# Paralome -- a reproducible, checkpointed framework for omics analysis.
# Copyright (C) 2026 Kenneth L. Jones
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License, version 3, as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# A separate commercial licence is available; see NOTICE.

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Optional



DEFAULT_STAR_SORT_RAM_FRACTION = 0.90


@dataclass(frozen=True)
class NodeResources:
    cpus: int
    mem_mb: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampleParallelPlan:
    requested_threads: int
    sample_count: int
    threads_per_sample: int
    concurrent_jobs: int
    peak_cpus: int
    max_threads_per_sample: Optional[int] = None
    requested_jobs: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StarResourcePlan:
    sample_plan: SampleParallelPlan
    mem: str
    mem_mb: int
    sort_ram_bytes: int
    sort_ram_fraction: float
    memory_source: str
    node_cpus: Optional[int] = None
    node_mem_mb: Optional[int] = None
    fallback_mem: Optional[str] = None

    @property
    def requested_threads(self) -> int:
        return self.sample_plan.requested_threads

    @property
    def sample_count(self) -> int:
        return self.sample_plan.sample_count

    @property
    def threads_per_sample(self) -> int:
        return self.sample_plan.threads_per_sample

    @property
    def concurrent_jobs(self) -> int:
        return self.sample_plan.concurrent_jobs

    @property
    def peak_cpus(self) -> int:
        return self.sample_plan.peak_cpus

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "requested_threads": self.requested_threads,
                "samples": self.sample_count,
                "threads_per_sample": self.threads_per_sample,
                "array_concurrency": self.concurrent_jobs,
                "peak_cpus": self.peak_cpus,
            }
        )
        return data


def parse_mem_to_mb(value: Optional[str]) -> int:
    """Parse a SLURM-style memory string into megabytes.

    Accepts integers (megabytes) and the K, M, G, T and P suffixes SLURM does,
    with or without a trailing B. Returns 0 only when no value was given.

    K and P are parsed, and anything else unparsable raises PLM-0110.
    The option check accepts K and P, this returned 0 for them, and the STAR
    memory plan read 0 as "not given" and asked for 32G without a word.
    """
    if value is None:
        return 0
    text = str(value).strip().upper()
    if not text:
        return 0
    multipliers = (
        ("PB", 1024.0 ** 3),
        ("P", 1024.0 ** 3),
        ("TB", 1024.0 ** 2),
        ("T", 1024.0 ** 2),
        ("GB", 1024.0),
        ("G", 1024.0),
        ("MB", 1.0),
        ("M", 1.0),
        ("KB", 1.0 / 1024),
        ("K", 1.0 / 1024),
    )
    try:
        for suffix, multiplier in multipliers:
            if text.endswith(suffix):
                megabytes = float(text[: -len(suffix)]) * multiplier
                break
        else:
            megabytes = float(text)
        if megabytes != megabytes or megabytes in (float("inf"), float("-inf")) or megabytes < 0:
            raise ValueError(text)
    except ValueError:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            110, f"Memory value {str(value)!r} cannot be read; use values such as 4G or 32000M"
        ) from None
    # A positive amount below one megabyte is one megabyte, never "not given".
    return max(1, int(megabytes)) if megabytes > 0 else 0


def format_mem_mb(mem_mb: int) -> str:
    return f"{max(1, int(mem_mb))}M"


def read_proc_meminfo_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(int(parts[1]) / 1024)
    except Exception:
        return None
    return None


def detect_allocated_node_resources() -> Optional[NodeResources]:
    """Detect resources from the current SLURM allocation, if one exists."""
    cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    mem = os.environ.get("SLURM_MEM_PER_NODE")
    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")

    parsed_cpus: Optional[int] = None
    if cpus:
        # SLURM_JOB_CPUS_PER_NODE can look like "80(x1)".
        token = str(cpus).split("(", 1)[0].split(",", 1)[0].strip()
        try:
            parsed_cpus = int(float(token))
        except ValueError:
            parsed_cpus = None

    # What SLURM put in the environment is diagnostic; a value this cannot read
    # there means "allocation unknown", not a refusal.
    def _allocated_mb(text: str) -> int:
        try:
            return parse_mem_to_mb(text)
        except RuntimeError:
            return 0

    if parsed_cpus and mem:
        mem_mb = _allocated_mb(mem)
        if mem_mb > 0:
            return NodeResources(cpus=parsed_cpus, mem_mb=mem_mb, source="slurm_allocation")

    if parsed_cpus and mem_per_cpu:
        mem_mb = _allocated_mb(mem_per_cpu)
        if mem_mb > 0:
            return NodeResources(cpus=parsed_cpus, mem_mb=parsed_cpus * mem_mb, source="slurm_allocation_per_cpu")

    return None


def detect_slurm_node_resources(partition: Optional[str] = None) -> Optional[NodeResources]:
    """Return representative SLURM node resources before submission.

    Before submission there is no allocated node, so this queries node
    definitions through sinfo and chooses the largest CPU node matching the
    optional partition. Returns None if SLURM is unavailable or unparseable.
    """
    allocated = detect_allocated_node_resources()
    if allocated is not None:
        return allocated

    if shutil.which("sinfo") is None:
        return None
    cmd = ["sinfo", "-h", "-N", "-o", "%c %m"]
    if partition:
        cmd[1:1] = ["-p", str(partition)]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        return None

    candidates: list[tuple[int, int]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            cpus = int(float(parts[0]))
            mem_mb = int(float(parts[1]))
        except ValueError:
            continue
        if cpus > 0 and mem_mb > 0:
            candidates.append((cpus, mem_mb))

    if not candidates:
        proc_mem = read_proc_meminfo_mb()
        proc_cpus = os.cpu_count()
        if proc_mem and proc_cpus:
            return NodeResources(cpus=int(proc_cpus), mem_mb=int(proc_mem), source="proc_meminfo")
        return None

    cpus, mem_mb = max(candidates, key=lambda x: (x[0], x[1]))
    return NodeResources(cpus=cpus, mem_mb=mem_mb, source="sinfo")


_MAX_CPUS_PER_NODE: dict[Optional[str], Optional[int]] = {}


def max_cpus_per_node(partition: Optional[str] = None, *, refresh: bool = False) -> Optional[int]:
    """The largest CPU count any node in this partition offers, or None.

    A CPU request above the biggest node is not a tight fit, it is
    UNPLACEABLE: sbatch refuses it with "CPU count per node can not be
    satisfied" and nothing queues, however long you wait (a run's --threads
    above the largest node, given to a stage that asks for --threads CPUs, say).

    sinfo's node DEFINITIONS are the only honest source for this, and they are
    read once per partition per process. Deliberately not
    detect_slurm_node_resources: that prefers the current allocation, so a
    supervisor running inside a two-CPU job would clamp every script it writes
    to two CPUs. Returns None when sinfo is absent or unreadable, and then
    nothing is clamped -- a guess here would be worse than letting the
    scheduler answer for itself.
    """
    key = str(partition) if partition else None
    if not refresh and key in _MAX_CPUS_PER_NODE:
        return _MAX_CPUS_PER_NODE[key]

    answer: Optional[int] = None
    if shutil.which("sinfo") is not None:
        cmd = ["sinfo", "-h", "-N", "-o", "%c"]
        if partition:
            cmd[1:1] = ["-p", str(partition)]
        try:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except Exception:
            proc = None
        if proc is not None:
            counts = []
            for line in proc.stdout.split():
                try:
                    value = int(float(line))
                except ValueError:
                    continue
                if value > 0:
                    counts.append(value)
            if counts:
                answer = max(counts)

    _MAX_CPUS_PER_NODE[key] = answer
    return answer


_MAX_TMP_DISK_MB: dict[Optional[str], Optional[int]] = {}


def max_tmp_disk_mb(partition: Optional[str] = None, *, refresh: bool = False) -> Optional[int]:
    """The largest TmpDisk (MB) any node in this partition defines, or None when sinfo cannot say.

    For ``--tmp`` (see ``paralome.pipeline.slurm.tmp_request``). As
    with CPUs, a request above every node's TmpDisk is refused outright, not
    queued. 0 means no node defines one: SLURM then does no scratch accounting
    and a --tmp request could never be placed. Read from sinfo's node
    definitions, once per partition per process, for the reason given in
    max_cpus_per_node.
    """
    key = str(partition) if partition else None
    if not refresh and key in _MAX_TMP_DISK_MB:
        return _MAX_TMP_DISK_MB[key]

    answer: Optional[int] = None
    if shutil.which("sinfo") is not None:
        cmd = ["sinfo", "-h", "-N", "-o", "%d"]
        if partition:
            cmd[1:1] = ["-p", str(partition)]
        try:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except Exception:
            proc = None
        if proc is not None:
            sizes = []
            for line in proc.stdout.split():
                try:
                    sizes.append(int(float(line)))
                except ValueError:
                    continue
            if sizes:
                answer = max(0, max(sizes))

    _MAX_TMP_DISK_MB[key] = answer
    return answer


def plan_sample_parallel(
    *,
    total_threads: int,
    sample_count: int,
    max_threads_per_sample: Optional[int] = None,
    jobs: Optional[int] = None,
) -> SampleParallelPlan:
    """Plan CPU usage for a sample-parallel SLURM array stage.

    total_threads is interpreted as the total CPU budget for the stage.
    jobs limits concurrent array tasks. If jobs is omitted or <= 0, all samples
    may run concurrently.
    """
    requested_threads = max(1, int(total_threads))
    samples = max(1, int(sample_count))
    if jobs is None or int(jobs) <= 0:
        requested_concurrency = samples
    else:
        requested_concurrency = max(1, min(int(jobs), samples))

    # A one-CPU minimum applies to every concurrent task, so concurrency itself
    # must never exceed the total stage CPU budget.  This guarantees
    # peak_cpus <= requested_threads even when task_count > total_threads.
    concurrent_jobs = max(1, min(requested_concurrency, requested_threads))
    threads_per_sample = max(1, requested_threads // concurrent_jobs)
    if max_threads_per_sample is not None:
        threads_per_sample = min(int(max_threads_per_sample), threads_per_sample)

    return SampleParallelPlan(
        requested_threads=requested_threads,
        sample_count=samples,
        threads_per_sample=threads_per_sample,
        concurrent_jobs=concurrent_jobs,
        peak_cpus=threads_per_sample * concurrent_jobs,
        max_threads_per_sample=max_threads_per_sample,
        requested_jobs=jobs,
    )


#: The pieces of a fused alignment task's memory default. A task that marks duplicates on STAR's output
#: holds the aligner and the samtools chain at once, so its request is their sum, worked out from the
#: genome's index on disk rather than a fixed figure each run would have to override with --stage-mem
#: alignment=.
STAR_INDEX_FILES = ("Genome", "SA", "SAindex")
#: STAR beyond its index: I/O buffers and the per-thread output chunks.
STAR_WORKING_BASE_BYTES = 2 * 1024 ** 3
STAR_WORKING_PER_THREAD_BYTES = 256 * 1024 ** 2
#: samtools sort's own default -m. Paralome does not pass -m to the chain
#: (stream.markdup_stream_bash says why), so this is what each sort holds per
#: thread; SAMTOOLS_SORT_OVERHEAD covers what it holds beyond -m.
SAMTOOLS_SORT_MEM_PER_THREAD_BYTES = 768 * 1024 ** 2
SAMTOOLS_SORT_OVERHEAD = 1.15
SAMTOOLS_MARKDUP_BYTES = 2 * 1024 ** 3
FASTP_IN_TASK_BYTES = 2 * 1024 ** 3
FUSED_MEMORY_MARGIN = 0.15
#: Never asked for less than the stage's fixed default.
FUSED_MEMORY_FLOOR_BYTES = 32 * 1024 ** 3


def star_index_bytes(reference: Any) -> Optional[int]:
    """The bytes STAR loads from a genome index: Genome, SA and SAindex.

    None when the index files are not there to measure, which the caller
    records rather than guessing a size.
    """
    from pathlib import Path

    folder = Path(reference)
    total = 0
    found = False
    for name in STAR_INDEX_FILES:
        path = folder / name
        try:
            if path.is_file():
                total += path.stat().st_size
                found = True
        except OSError:
            continue
    return total if found and total > 0 else None


@dataclass(frozen=True)
class FusedAlignmentMemory:
    """What a fused alignment task asks for by default, and why."""

    mem: str
    total_bytes: int
    index_bytes: Optional[int]
    index_source: str
    threads: int
    components: dict
    phase_while_aligning_bytes: int
    phase_after_aligning_bytes: int
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_fused_alignment_memory(
    *,
    reference: Any,
    threads: int,
    buffer_bytes: int = 0,
    fastp_in_task: bool = False,
    aligner_index: Optional[tuple[Optional[int], str]] = None,
) -> FusedAlignmentMemory:
    """The default --mem of an alignment task that marks duplicates itself.

    Two phases, and the request covers the larger:

    * while STAR runs: its index and working memory, the name sort filling
      from STAR's output, the read buffers and fastp when trimming is fused;
    * after STAR exits: the name sort handing its reads on, the position sort
      filling, and markdup.

    Every piece is recorded, so a run's summary shows what the number is made
    of. --mem, or --stage-mem alignment= in the pipeline, replaces it.
    """
    gib = 1024 ** 3
    threads = max(1, int(threads))
    if aligner_index is not None:
        # bwa-mem2 (--aligner bwa): its index's bytes and where the number came from.
        index, index_source = aligner_index
    else:
        index = star_index_bytes(reference)
        index_source = "Genome + SA + SAindex on disk"
    if index is None:
        index_used = FUSED_MEMORY_FLOOR_BYTES - STAR_WORKING_BASE_BYTES
        index_source = "index files not found; assumed the old 32G STAR request"
    else:
        index_used = index
    star_working = STAR_WORKING_BASE_BYTES + STAR_WORKING_PER_THREAD_BYTES * threads
    one_sort = int(SAMTOOLS_SORT_MEM_PER_THREAD_BYTES * threads * SAMTOOLS_SORT_OVERHEAD)
    buffers = max(0, int(buffer_bytes))
    fastp = FASTP_IN_TASK_BYTES if fastp_in_task else 0
    while_aligning = index_used + star_working + one_sort + buffers + fastp
    after_aligning = one_sort + one_sort + SAMTOOLS_MARKDUP_BYTES
    peak = max(while_aligning, after_aligning)
    total = max(FUSED_MEMORY_FLOOR_BYTES, int(peak * (1 + FUSED_MEMORY_MARGIN)))
    whole_gb = -(-total // gib)
    return FusedAlignmentMemory(
        mem=f"{whole_gb}G",
        total_bytes=whole_gb * gib,
        index_bytes=index,
        index_source=index_source,
        threads=threads,
        components={
            "star_index": index_used,
            "star_working": star_working,
            "samtools_name_sort": one_sort,
            "samtools_position_sort": one_sort,
            "samtools_markdup": SAMTOOLS_MARKDUP_BYTES,
            "read_buffers": buffers,
            "fastp": fastp,
        },
        phase_while_aligning_bytes=while_aligning,
        phase_after_aligning_bytes=after_aligning,
        margin=FUSED_MEMORY_MARGIN,
    )


def plan_star_memory(
    *,
    sample_plan: SampleParallelPlan,
    partition: Optional[str] = None,
    fallback_mem: str = "32G",
    sort_ram_fraction: float = DEFAULT_STAR_SORT_RAM_FRACTION,
    discover_node_memory: bool = False,
) -> StarResourcePlan:
    """Plan STAR per-task memory and BAM sort RAM.

    ``fallback_mem`` is the authoritative per-task scheduler request. Node
    discovery is diagnostic-only by default because scaling memory from CPU
    share can under-allocate STAR's largely fixed genome-index footprint.
    ``sort_ram_fraction`` controls STAR's internal sort limit independently of
    the scheduler request.
    """
    node = detect_slurm_node_resources(partition) if discover_node_memory else None
    fallback_mb = parse_mem_to_mb(fallback_mem)
    if not fallback_mb and fallback_mem is not None and str(fallback_mem).strip():
        # A given 0 is refused, not silently replaced by 32G.
        from paralome.errors import ParalomeError

        raise ParalomeError(
            110,
            f"Memory value {str(fallback_mem)!r} is zero; STAR's memory and BAM sort limit need an "
            f"explicit amount such as 32G",
        )
    fallback_mb = fallback_mb or 32 * 1024

    if node:
        per_task_mb = max(
            fallback_mb,
            int(node.mem_mb * (sample_plan.threads_per_sample / max(1, node.cpus))),
        )
        source = f"max(explicit,{node.source}_cpu_fraction)"
        node_cpus = node.cpus
        node_mem_mb = node.mem_mb
    else:
        per_task_mb = fallback_mb
        source = "explicit_per_task_mem"
        node_cpus = None
        node_mem_mb = None

    fraction = max(0.01, min(float(sort_ram_fraction), 0.99))
    sort_ram_bytes = max(1, int(per_task_mb * 1024 * 1024 * fraction))

    return StarResourcePlan(
        sample_plan=sample_plan,
        mem=format_mem_mb(per_task_mb),
        mem_mb=per_task_mb,
        sort_ram_bytes=sort_ram_bytes,
        sort_ram_fraction=fraction,
        memory_source=source,
        node_cpus=node_cpus,
        node_mem_mb=node_mem_mb,
        fallback_mem=fallback_mem,
    )
