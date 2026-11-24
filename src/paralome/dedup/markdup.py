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
import csv
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from paralome.checkpoints import stage_checkpoint_path
from paralome.dedup.counts import FLAGSTAT_SIDECAR, in_parallel, primary_counts, recorded_flagstat, run_flagstat
from paralome.stream import markdup_stream_bash
from paralome.fastq.manifest import read_sample_inputs
from paralome.pipeline.naming import slurm_job_name
from paralome.pipeline.profiles import ManifestProfile
from paralome.pipeline.resources import plan_sample_parallel
from paralome.pipeline.slurm import placeable_cpus, tmp_directive
from paralome.sample_manifest import read_sample_manifest
from paralome.scratch import job_scratch_bash, job_scratch_dir


SCHEMA_PLAN = "paralome.dedup.plan.v1"
SCHEMA_CHECKPOINT = "paralome.dedup.checkpoint.v1"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def require_tool(name: str) -> str:
    """The tool to invoke, from what setup recorded -- not from PATH.

    See binaries.resolve_tool. Returns the resolved absolute path so
    the caller can invoke THAT rather than the bare name it asked about; a
    recorded path that no longer resolves raises PLM-0410 instead of silently
    falling back to whatever this account's PATH happens to find.
    """
    from paralome.binaries import resolve_tool

    exe = resolve_tool(name)
    if exe is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")
    return exe


def file_info(path: str | Optional[Path]) -> dict[str, Any]:
    """Local variant of ``paralome.checkpoints.file_info`` -- DELIBERATELY NOT MERGED.

    It differs from the shared one in one line:

        size = p.stat().st_size if exists and p.is_file() else 0   # here
        size = p.stat().st_size if exists else 0                   # checkpoints.file_info

    That is a behavioural fork, not a formatting one. Handed a DIRECTORY, this
    version reports ``size 0, nonempty False``; the shared version calls
    ``stat()`` on the directory, gets its inode size (typically 64-4096, never
    0) and reports ``nonempty True``.

    Merging them means choosing which behaviour every caller gets, and the
    shared function is what ``pipeline/state.py`` uses to decide whether a stage
    is finished: adopting the ``is_file()`` guard everywhere could flip an
    already-complete stage to incomplete in existing project folders if any
    recorded output path is a directory, and dropping it here could do the
    reverse. That is a decision about pipeline state, so the two are kept apart.
    """
    if path is None:
        return {"path": None, "exists": False, "nonempty": False, "size": 0}
    p = Path(path)
    exists = p.exists()
    size = p.stat().st_size if exists and p.is_file() else 0
    return {"path": str(p), "exists": exists, "nonempty": bool(exists and size > 0), "size": size}


def ensure_dirs(workdir: Path) -> None:
    for name in ("inputs", "bam", "metrics", "logs", "scripts", "temp", "checkpoints", "results"):
        (workdir / name).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DedupSample:
    index: int
    sample_id: str
    group: str
    condition: str
    affected: str
    relationship: str
    family_id: str
    sex: str
    mother_id: str
    father_id: str
    input_bam: str
    output_bam: str

    @property
    def output_bai(self) -> str:
        return self.output_bam + ".bai"

    def to_manifest_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "family_id": self.family_id,
            "relationship": self.relationship,
            "affected": self.affected,
            "sex": self.sex,
            "condition": self.condition,
            "group": self.group,
            "mother_id": self.mother_id,
            "father_id": self.father_id,
            "bam": self.output_bam,
        }


def read_bam_samples(
    samples_csv: Union[str, Path],
    workdir: Path,
    manifest_profile: str = "family",
) -> list[DedupSample]:
    profile = (
        manifest_profile
        if isinstance(manifest_profile, ManifestProfile)
        else ManifestProfile(str(manifest_profile).strip().lower())
    )
    out: list[DedupSample] = []

    if profile is ManifestProfile.GENERAL:
        records = read_sample_inputs(
            Path(samples_csv),
            manifest_profile=ManifestProfile.GENERAL,
            validate_schema=True,
        )
        for idx, sample in enumerate(records, start=1):
            if not sample.bam:
                raise ValueError(f"Sample {sample.sample_id} has an empty BAM path")
            bam = Path(sample.bam)
            if not bam.exists():
                raise FileNotFoundError(f"BAM for sample {sample.sample_id} not found: {bam}")
            condition = sample.condition if sample.condition not in {"", "."} else sample.group
            condition = condition if condition not in {"", "."} else "unknown"
            out.append(
                DedupSample(
                    index=idx,
                    sample_id=sample.sample_id,
                    group=condition,
                    condition=condition,
                    affected="unknown",
                    relationship="unknown",
                    family_id="default",
                    sex="unknown",
                    mother_id=".",
                    father_id=".",
                    input_bam=str(bam),
                    output_bam=str(workdir / "bam" / f"{sample.sample_id}.markdup.bam"),
                )
            )
        return out

    samples = read_sample_manifest(Path(samples_csv))
    for idx, sample in enumerate(samples, start=1):
        bam = Path(sample.bam)
        if not bam.exists():
            raise FileNotFoundError(f"BAM for sample {sample.sample_id} not found: {bam}")
        out_bam = workdir / "bam" / f"{sample.sample_id}.markdup.bam"
        out.append(
            DedupSample(
                index=idx,
                sample_id=sample.sample_id,
                group=sample.group,
                condition=".",
                affected=sample.group,
                relationship=sample.relationship,
                family_id=sample.family_id,
                sex=sample.sex,
                mother_id=sample.mother_id or ".",
                father_id=sample.father_id or ".",
                input_bam=str(bam),
                output_bam=str(out_bam),
            )
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, ".") for k in fieldnames})
    return path


def build_plan(
    *,
    samples_csv: Path,
    workdir: Path,
    samples: list[DedupSample],
    threads: int,
    jobs: int,
    backend: str,
    remove_duplicates: bool,
    overwrite: bool,
    submit: bool,
    manifest_profile: str,
) -> dict[str, Any]:
    resource_plan = plan_sample_parallel(
        total_threads=threads, sample_count=len(samples), jobs=jobs
    )
    return {
        "schema": SCHEMA_PLAN,
        "stage": "deduplication",
        "created": now_iso(),
        "workdir": str(workdir),
        "samples_manifest": str(samples_csv),
        "samples": len(samples),
        "requested_threads": resource_plan.requested_threads,
        "threads_per_sample": resource_plan.threads_per_sample,
        "jobs": int(jobs),
        "concurrent_jobs": resource_plan.concurrent_jobs,
        "peak_cpus": resource_plan.peak_cpus,
        "backend": backend,
        "remove_duplicates": bool(remove_duplicates),
        "mark_duplicates": True,
        "overwrite": bool(overwrite),
        "submit": bool(submit),
        "manifest_profile": manifest_profile,
        "outputs": {
            "dedup_manifest": str(workdir / "inputs" / "dedup_samples.csv"),
            "summary": str(workdir / "results" / "dedup_summary.json"),
            "checkpoint": str(stage_checkpoint_path(workdir, "deduplication")),
        },
        "sample_outputs": {s.sample_id: {"bam": s.output_bam, "bai": s.output_bai, "input_bam": s.input_bam} for s in samples},
    }


def write_checkpoint(workdir: Path, plan: dict[str, Any], *, complete: bool, submitted: bool = False, job_id: Optional[str] = None, message: str = "", metrics: Optional[dict[str, Any]] = None) -> Path:
    ckpt = stage_checkpoint_path(workdir, "deduplication")
    state = {
        "schema": SCHEMA_CHECKPOINT,
        "stage": "deduplication",
        "complete": bool(complete),
        "submitted": bool(submitted),
        "job_id": job_id,
        "started": plan.get("created"),
        "updated": now_iso(),
        "finished": now_iso() if complete else None,
        "message": message,
        "mark_duplicates": True,
        "remove_duplicates": plan.get("remove_duplicates"),
        "samples": plan.get("samples"),
        "threads_per_sample": plan.get("threads_per_sample"),
        "jobs": plan.get("jobs"),
        "backend": plan.get("backend"),
        "outputs": plan.get("outputs", {}),
        "sample_outputs": plan.get("sample_outputs", {}),
        "metrics": metrics or {},
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return ckpt


def bam_complete(path: Union[str, Path]) -> bool:
    bam = Path(path)
    bai = Path(str(bam) + ".bai")
    alt_bai = bam.with_suffix(bam.suffix + ".bai")
    return bam.exists() and bam.stat().st_size > 0 and ((bai.exists() and bai.stat().st_size > 0) or (alt_bai.exists() and alt_bai.stat().st_size > 0))


def count_reads(bam: Path, flag: Optional[int] = None) -> int:
    cmd = [require_tool("samtools"), "view", "-c"]
    if flag is not None:
        cmd += ["-f", str(flag)]
    cmd.append(str(bam))
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return int(proc.stdout.strip() or "0")


def flagstat(bam: Path, *, recorded: bool = True) -> dict:
    """A BAM's flagstat: what its task wrote beside it (recorded), else one ``samtools flagstat`` pass."""
    data = recorded_flagstat(bam) if recorded else None
    return data if data is not None else run_flagstat(bam, require_tool("samtools"))


def _duplicate_metrics(out_bam: Path, input_bam: Path, *, remove_duplicates: bool) -> dict[str, Any]:
    """Read counts for one finished BAM, in primary reads.

    ``total_reads`` is the mapped reads (flagstat ``primary mapped``), ``duplicate_reads`` the duplicates
    among them (``primary duplicates``) and ``duplicate_rate`` the one over the other: a read is counted
    once, as ``samtools flagstat`` gives it by hand, so the rate can be set against the mapping table.
    ``properly_paired_reads``, ``paired_reads`` and ``primary_reads`` come along for the sequence QC table.

    With --remove-duplicates (markdup -r) the duplicates are gone from the output, so they are the input's
    mapped reads less the output's (sort and fixmate drop no reads), and the paired counts are the input's.
    An input BAM that is no longer there leaves the duplicates unavailable (None, shown as ".") rather than 0.
    """
    out = primary_counts(flagstat(out_bam))
    if not remove_duplicates:
        mapped, dup = out["mapped_reads"], out["primary_duplicates"]
        return {"total_reads": mapped, "duplicate_reads": dup, "duplicate_rate": (dup / mapped) if mapped and dup is not None else None,
                "properly_paired_reads": out["properly_paired_reads"], "paired_reads": out["paired_reads"],
                "primary_reads": out["primary_reads"]}
    if not input_bam.is_file():
        return {"total_reads": out["mapped_reads"], "duplicate_reads": None, "duplicate_rate": None,
                "properly_paired_reads": None, "paired_reads": out["paired_reads"], "primary_reads": None}
    before = primary_counts(flagstat(input_bam, recorded=False))
    mapped = before["mapped_reads"]
    dup = max(0, (mapped or 0) - (out["mapped_reads"] or 0))
    return {"total_reads": mapped, "duplicate_reads": dup, "duplicate_rate": (dup / mapped) if mapped else None,
            "properly_paired_reads": before["properly_paired_reads"], "paired_reads": before["paired_reads"],
            "primary_reads": before["primary_reads"]}


def _current_counts(workdir: Path) -> dict[str, dict[str, Any]]:
    """Each sample's counts in the stage's summary that are still its BAM's: in primary reads (properly_paired_reads is there) and written after the BAM was."""
    path = Path(workdir) / "results" / "dedup_summary.json"
    try:
        summary = json.loads(path.read_text())
        written = path.stat().st_mtime
    except (OSError, ValueError):
        return {}
    out = {}
    for m in summary.get("sample_metrics") or []:
        if not isinstance(m, dict) or "properly_paired_reads" not in m or m.get("duplicate_reads") is None:
            continue
        try:
            if Path(str(m.get("output_bam"))).stat().st_mtime > written:
                continue
        except OSError:
            continue
        out[str(m.get("sample_id"))] = m
    return out


def run_one_sample(sample: DedupSample, *, workdir: Path, threads: int, remove_duplicates: bool, overwrite: bool, clip_overlap: bool = False) -> dict[str, Any]:
    samtools = require_tool("samtools")
    input_bam = Path(sample.input_bam)
    out_bam = Path(sample.output_bam)
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = workdir / "temp" / sample.sample_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    if bam_complete(out_bam) and not overwrite:
        return {
            "sample_id": sample.sample_id,
            "complete": True,
            "reused": True,
            "input_bam": str(input_bam),
            "output_bam": str(out_bam),
            **_duplicate_metrics(out_bam, input_bam, remove_duplicates=remove_duplicates),
        }

    markdup_cmd = [samtools, "markdup", "-@", str(threads)]
    if remove_duplicates:
        markdup_cmd.append("-r")
    # Written and indexed under a temporary name, then moved into
    # place with the old index taken away first (see the SLURM task script):
    # a final-path BAM and index only ever come from one finished run.
    partial_bam = temp_dir / f"{sample.sample_id}.markdup.partial.bam"
    partial_bai = Path(str(partial_bam) + ".bai")
    for stale in (partial_bam, partial_bai):
        stale.unlink(missing_ok=True)
    # The same four steps as one stream, matching the SLURM task
    # script -- uncompressed between steps, each sort's temporary files on this
    # node's scratch, and nothing written to shared storage but the result.
    # `sort -n` and the default sort memory are kept on purpose; see the long
    # note in the task script for why either one would change the output.
    try:
        need = os.path.getsize(input_bam) * DEDUP_SCRATCH_FACTOR
    except OSError:
        need = 0
    with job_scratch_dir(temp_dir, "dedup", need_bytes=need) as scratch_dir:
        markdup_cmd += ["-T", str(scratch_dir / "markdup"), "-", str(partial_bam)]
        name_sort = subprocess.Popen(
            [samtools, "sort", "-n", "-@", str(threads), "-u",
             "-T", str(scratch_dir / "namesort"), "-o", "-", str(input_bam)],
            stdout=subprocess.PIPE)
        fixmate = subprocess.Popen(
            [samtools, "fixmate", "-m", "-u", "-@", str(threads), "-", "-"],
            stdin=name_sort.stdout, stdout=subprocess.PIPE)
        position_sort = subprocess.Popen(
            [samtools, "sort", "-@", str(threads), "-u",
             "-T", str(scratch_dir / "possort"), "-o", "-", "-"],
            stdin=fixmate.stdout, stdout=subprocess.PIPE)
        markdup = subprocess.Popen(markdup_cmd, stdin=position_sort.stdout)
        # Each stage closes its own copy of the pipe it wrote into, so a
        # failure upstream reaches the next one as end of input rather than
        # hanging it. Statuses are checked from the end back to the start, so
        # the error reported is the first one that happened.
        for handle in (name_sort.stdout, fixmate.stdout, position_sort.stdout):
            if handle is not None:
                handle.close()
        stages = (("sort -n", name_sort), ("fixmate", fixmate),
                  ("sort", position_sort), ("markdup", markdup))
        for _, proc in stages:
            proc.wait()
        for name, proc in stages:
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, f"samtools {name}")
        if clip_overlap:
            # The mates' overlap soft-clipped, as the SLURM task does in its stream.
            from paralome.stream import BAMUTIL_TOOL, CLIP_OVERLAP_ARGS

            clipped = partial_bam.with_name(partial_bam.name + ".clip.bam")
            args = [require_tool(BAMUTIL_TOOL), "clipOverlap", "--in", str(partial_bam), "--out", str(clipped),
                    *[a for a in CLIP_OVERLAP_ARGS if a not in ("--in", "-.bam")]]
            subprocess.run(args, check=True)
            clipped.replace(partial_bam)
    subprocess.run([samtools, "index", "-@", str(threads), str(partial_bam)], check=True)
    Path(str(out_bam) + ".bai").unlink(missing_ok=True)
    os.replace(partial_bam, out_bam)
    os.replace(partial_bai, Path(str(out_bam) + ".bai"))


    return {
        "sample_id": sample.sample_id,
        "complete": bam_complete(out_bam),
        "reused": False,
        "input_bam": str(input_bam),
        "output_bam": str(out_bam),
        **_duplicate_metrics(out_bam, input_bam, remove_duplicates=remove_duplicates),
    }


def write_sample_manifests(workdir: Path, samples: list[DedupSample]) -> tuple[Path, Path]:
    manifest = workdir / "inputs" / "dedup_mark_manifest.csv"
    dedup_samples = workdir / "inputs" / "dedup_samples.csv"
    write_csv(
        manifest,
        [asdict(s) | {"output_bai": s.output_bai} for s in samples],
        ["index", "sample_id", "group", "relationship", "family_id", "sex", "mother_id", "father_id", "input_bam", "output_bam", "output_bai"],
    )
    write_csv(
        dedup_samples,
        [s.to_manifest_row() for s in samples],
        ["sample_id", "family_id", "relationship", "affected", "sex", "condition", "group", "mother_id", "father_id", "bam"],
    )
    return manifest, dedup_samples


def write_summary_script(workdir: Path, *, partition: Optional[str]) -> Path:
    """The job that writes the summary after a SLURM run's tasks: ``dedup summarize``."""
    from paralome.binaries import worker_argv
    from paralome.pipeline.slurm import SlurmOptions, write_sbatch

    return write_sbatch(
        workdir / "scripts" / "dedup_summarize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(workdir, "dedup_summary"), cpus_per_task=4, mem="4G", partition=partition,
            output=workdir / "logs" / "dedup_summarize_%j.out", error=workdir / "logs" / "dedup_summarize_%j.err",
        ),
        shlex.join([*worker_argv(), "dedup", "summarize", "--workdir", str(workdir)]),
    )


def summarize_dedup_stage(workdir: Union[str, Path]) -> dict[str, Any]:
    """Write the stage's summary and checkpoint from the duplicate-marked BAMs that are there.

    Each sample's read and duplicate counts come from what its task recorded beside the BAM, else one
    flagstat pass, the samples in parallel (dedup.counts). Nothing is marked again.
    """
    workdir = Path(workdir)
    plan_json = workdir / "inputs" / "dedup_plan.json"
    if not plan_json.is_file():
        raise FileNotFoundError(f"{plan_json} is not there: duplicate marking has not been run in {workdir}")
    plan = json.loads(plan_json.read_text())
    samples = read_bam_samples(plan["samples_manifest"], workdir, manifest_profile=plan.get("manifest_profile") or "family")
    remove = bool(plan.get("remove_duplicates"))
    metrics = collect_metrics(samples, remove_duplicates=remove)
    summary = write_summary(workdir, plan, metrics)
    write_checkpoint(
        workdir, plan, complete=bool(metrics.get("complete")),
        message=("summary written from the finished tasks" if metrics.get("complete")
                 else "not every sample has its duplicate-marked BAM yet"),
        metrics=metrics,
    )
    return {"summary": str(summary), **metrics}


def write_summary(workdir: Path, plan: dict[str, Any], metrics: dict[str, Any]) -> Path:
    out = workdir / "results" / "dedup_summary.json"
    summary = {
        "schema": "paralome.dedup.summary.v1",
        "stage": "deduplication",
        "complete": metrics.get("complete"),
        "samples": metrics.get("samples"),
        "completed_samples": metrics.get("completed_samples"),
        "total_reads": metrics.get("total_reads"),
        "duplicate_reads": metrics.get("duplicate_reads"),
        "duplicate_rate": metrics.get("duplicate_rate"),
        "remove_duplicates": plan.get("remove_duplicates"),
        "outputs": plan.get("outputs", {}),
        "sample_metrics": metrics.get("sample_metrics", []),
        "finished": now_iso(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out


def collect_metrics(
    samples: list[DedupSample],
    *,
    scan_read_counts: bool = True,
    remove_duplicates: bool = False,
    known: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Collect dedup completion and optional read-count metrics.

    ``scan_read_counts=False`` is used when SLURM outputs already exist and the
    stage is being recovered or finalized. In that path BAM/BAI presence is
    sufficient for completion, and Paralome must not rescan every BAM with
    ``samtools view -c`` merely to rebuild metadata.

    ``remove_duplicates`` says the output BAMs no longer hold their
    duplicates, so the count is taken against the input BAM instead (see
    _duplicate_metrics).
    """
    def one(s: DedupSample) -> dict[str, Any]:
        complete = bam_complete(s.output_bam)
        counts: dict[str, Any] = {"total_reads": None, "duplicate_reads": None, "duplicate_rate": None,
                                  "properly_paired_reads": None, "paired_reads": None, "primary_reads": None}
        if complete and scan_read_counts:
            counts = _duplicate_metrics(Path(s.output_bam), Path(s.input_bam), remove_duplicates=remove_duplicates)
        elif complete:
            # Without a scan, the counts already there are kept (a rerun with nothing to do wrote
            # the summary without them, over the ones `dedup summarize` had written): what the task recorded
            # beside the BAM, else the summary's own (see _current_counts).
            if not remove_duplicates and recorded_flagstat(Path(s.output_bam)) is not None:
                counts = _duplicate_metrics(Path(s.output_bam), Path(s.input_bam), remove_duplicates=False)
            elif (known or {}).get(s.sample_id):
                counts = {k: known[s.sample_id].get(k) for k in counts}
        return {
            "sample_id": s.sample_id,
            "complete": complete,
            "input_bam": s.input_bam,
            "output_bam": s.output_bam,
            **counts,
        }

    # The samples at once, as many as there are CPUs.
    sample_metrics = in_parallel(samples, one)
    completed = sum(1 for m in sample_metrics if m["complete"])
    total_reads = None
    duplicate_reads = None
    duplicate_rate = None
    if scan_read_counts or any(m.get("total_reads") is not None for m in sample_metrics):
        total_reads = sum(int(m.get("total_reads") or 0) for m in sample_metrics)
        # Over the samples whose duplicates could be counted; a rate over all
        # the reads with some samples uncounted would understate it.
        counted = [m for m in sample_metrics if m.get("duplicate_reads") is not None]
        if counted:
            duplicate_reads = sum(int(m["duplicate_reads"]) for m in counted)
            counted_reads = sum(int(m.get("total_reads") or 0) for m in counted)
            duplicate_rate = (duplicate_reads / counted_reads) if counted_reads else None
    return {
        "complete": completed == len(samples) and len(samples) > 0,
        "samples": len(samples),
        "completed_samples": completed,
        "read_counts_collected": bool(scan_read_counts) or all(m.get("duplicate_reads") is not None for m in sample_metrics if m["complete"]),
        "total_reads": total_reads,
        "duplicate_reads": duplicate_reads,
        "duplicate_rate": duplicate_rate,
        "sample_metrics": sample_metrics,
    }


#: The scratch space guard's estimate of a sample's sort spill: this multiple
#: of its input BAM, which is about what an uncompressed sort spill comes to.
DEDUP_SCRATCH_FACTOR = 4


def _largest_input_bam(manifest: Path, indices: list[int]) -> int:
    """The biggest input BAM among the array's tasks, for its --tmp request."""
    wanted = {str(i) for i in indices}
    largest = 0
    try:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("index")) in wanted:
                    try:
                        largest = max(largest, os.path.getsize(row.get("input_bam") or ""))
                    except OSError:
                        pass
    except OSError:
        return 0
    return largest


def slurm_array_spec(indices: list[int], jobs: int) -> str:
    base = ",".join(str(i) for i in indices)
    if jobs and jobs > 0:
        return f"{base}%{int(jobs)}"
    return base


def write_slurm_script(*, workdir: Path, manifest: Path, threads: int, jobs: int, remove_duplicates: bool, overwrite: bool, partition: Optional[str], mem: str, active_indices: list[int], clip_overlap: bool = False) -> Path:
    script = workdir / "scripts" / "dedup_mark_array.sbatch"
    script.parent.mkdir(parents=True, exist_ok=True)
    resource_plan = plan_sample_parallel(
        total_threads=threads, sample_count=len(active_indices), jobs=jobs
    )
    task_threads = resource_plan.threads_per_sample
    array_spec = slurm_array_spec(active_indices, resource_plan.concurrent_jobs)
    partition_line = f"#SBATCH --partition={shlex.quote(str(partition))}\n" if partition else ""
    remove_flag = "-r" if remove_duplicates else ""
    overwrite_int = "1" if overwrite else "0"
    # resolve_tool rather than require_tool: building a submission script must
    # not need the tool present on the machine doing the building. A stale
    # RECORDED path still raises PLM-0410 here, which is right -- that is a
    # configuration error, not an absence -- but a workstation that simply has
    # no samtools can still generate a script for a cluster that does.
    from paralome.binaries import resolve_tool

    samtools_exe = shlex.quote(resolve_tool("samtools") or "samtools")
    # The sorts' temporary files go on this node's scratch when the
    # site has one. They are the bulk of dedup's I/O and nobody reads them
    # afterwards, so they have no business on shared storage. The space guard
    # is given four times the input BAM, which is about what an uncompressed
    # sort spill comes to.
    scratch_bash = job_scratch_bash(
        '"$WORKDIR/temp/$SAMPLE_ID"', "dedup", var="DEDUP_SCRATCH",
        need_bytes=f'$(( $(stat -c %s "$INPUT_BAM" 2>/dev/null || echo 0) * {DEDUP_SCRATCH_FACTOR} ))',
    )
    # And SLURM is asked for that space (see paralome.scratch).
    tmp_line = tmp_directive(_largest_input_bam(manifest, active_indices) * DEDUP_SCRATCH_FACTOR, partition)
    # The stream itself comes from paralome.stream, which the fused
    # alignment task uses too. It was written out here and would have had to be
    # kept in step by hand with the copy that reads STAR's output on a pipe.
    from paralome.stream import BAMUTIL_TOOL

    dedup_stream = markdup_stream_bash(source='"$INPUT_BAM"', clip_var="BAMUTIL" if clip_overlap else "")
    bamutil_line = f"BAMUTIL={shlex.quote(resolve_tool(BAMUTIL_TOOL) or BAMUTIL_TOOL)}\n" if clip_overlap else ""
    script.write_text(f'''#!/usr/bin/env bash
{partition_line}#SBATCH --job-name={slurm_job_name(workdir, "dedup")}
#SBATCH --cpus-per-task={placeable_cpus(task_threads, partition, job_name="dedup")}
#SBATCH --mem={shlex.quote(str(mem))}
#SBATCH --array={array_spec}
{tmp_line}#SBATCH --output={shlex.quote(str(workdir))}/logs/dedup_%A_%a.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/dedup_%A_%a.err

set -euo pipefail

# The path `paralome setup` recorded, resolved at SUBMIT time on the login node and written into the
# script. A compute node's PATH is its own, so `command -v samtools` there could find a different
# build from the one the submitting account validated.
SAMTOOLS={samtools_exe}
{bamutil_line}if ! command -v "$SAMTOOLS" >/dev/null 2>&1; then
  echo "ERROR: samtools not found: $SAMTOOLS" >&2
  exit 1
fi

WORKDIR={shlex.quote(str(workdir))}
MANIFEST={shlex.quote(str(manifest))}
THREADS="${{SLURM_CPUS_PER_TASK:-{task_threads}}}"
REMOVE_FLAG={remove_flag!r}
OVERWRITE={overwrite_int}

line=$(awk -F',' -v id="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==id {{print; exit}}' "$MANIFEST")
if [[ -z "$line" ]]; then
  echo "ERROR: could not find manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=',' read -r IDX SAMPLE_ID GROUP REL FAMILY SEX MOTHER_ID FATHER_ID INPUT_BAM OUTPUT_BAM OUTPUT_BAI <<< "$line"

mkdir -p "$WORKDIR/temp/$SAMPLE_ID" "$WORKDIR/bam" "$WORKDIR/logs"

if [[ -s "$OUTPUT_BAM" && -s "$OUTPUT_BAM.bai" && "$OVERWRITE" != "1" ]]; then
  echo "Duplicate-marked BAM exists; skipping $SAMPLE_ID: $OUTPUT_BAM"
  exit 0
fi
# With overwrite the earlier index goes when this task starts, so
# the old BAM does not read finished while it is being redone.
if [[ "$OVERWRITE" == "1" ]]; then
  rm -f "$OUTPUT_BAM.bai"
fi

{scratch_bash}

# Written and indexed beside the final name, then moved into place (the index taken away first), so a
# task killed part-way cannot leave a truncated BAM with an index that every completion test accepts.
PARTIAL_BAM="$WORKDIR/temp/$SAMPLE_ID/$SAMPLE_ID.markdup.partial.bam"
rm -f "$PARTIAL_BAM" "$PARTIAL_BAM.bai"

# ONE STREAM, NOT FOUR FILES. Name-sort, fixmate, position-sort and markdup run in order with the
# same tools, passing uncompressed BAM between them and keeping each sort's temporary files on this
# node's scratch, rather than writing a full compressed BAM to shared storage after each step.
#
# WHAT IS DELIBERATELY KEPT, because changing it would change the output:
#   * `sort -n`. `samtools collate` is faster and groups mates, but it
#     does not order them, so the position sort downstream would receive a
#     different order and reads tied at one coordinate could come out in a
#     different order -- and markdup's choice of which duplicate to keep
#     follows that order.
#   * No `-m`. Changing a sort's memory changes how many temporary chunks it
#     makes, and the merge of equal keys follows the chunking.
# Compression and temporary-file location do not affect decoded content, so
# the duplicate-marked BAM is the same as one made through files.
{dedup_stream}
"$SAMTOOLS" index -@ "$THREADS" "$PARTIAL_BAM"
rm -f "$OUTPUT_BAM.bai"
mv -f "$PARTIAL_BAM" "$OUTPUT_BAM"
mv -f "$PARTIAL_BAM.bai" "$OUTPUT_BAM.bai"
# The read and duplicate counts, taken here with the task's threads, for the summary (dedup summarize).
"$SAMTOOLS" flagstat -@ "$THREADS" -O json "$OUTPUT_BAM" > "$OUTPUT_BAM{FLAGSTAT_SIDECAR}"
echo "Wrote $OUTPUT_BAM"
''')
    script.chmod(0o755)
    return script


@dataclass(frozen=True)
class DedupResult:
    workdir: str
    samples: int
    completed_samples: int
    dedup_manifest: str
    plan_json: str
    checkpoint: str
    summary: str
    backend: str
    submitted: bool
    job_id: Optional[str]
    complete: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _requests_path(workdir: Path) -> Path:
    return workdir / "inputs" / "dedup_requests.json"


def _load_requests(workdir: Path) -> dict[str, Any]:
    """What each finished sample was made from and with, as recorded."""
    try:
        payload = json.loads(_requests_path(workdir).read_text())
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not isinstance(payload.get("samples"), dict):
        payload["samples"] = {}
    return payload


def _save_requests(workdir: Path, payload: dict[str, Any]) -> None:
    path = _requests_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sample_request(sample: DedupSample, *, remove_duplicates: bool) -> dict[str, Any]:
    """Everything that decides a sample's duplicate-marked BAM.

    The input BAM (its path and size) and whether duplicates are
    removed or only marked. Threads, memory, jobs and partition are left out:
    they do not change the result.
    """
    from paralome.fastq.fastp import input_signature

    return {
        "remove_duplicates": bool(remove_duplicates),
        "inputs": {"bam": input_signature(sample.input_bam)},
    }


def _prepare_requests(
    samples: list[DedupSample], workdir: Path, *, overwrite: bool, remove_duplicates: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """This run's request per sample; refuses when a finished one was made otherwise.

    As FASTQ QC and STAR do (PLM-0309): each sample's request is written down
    when it is made, and a finished sample whose record differs from what is
    asked now (a re-aligned input BAM, --remove-duplicates added or dropped)
    is refused before anything starts, rather than reused by its output file
    alone. A BAM with no record was made by an older Paralome and is accepted.
    """
    from paralome.fastq.fastp import refuse_mixed_results, request_differences

    record = _load_requests(workdir)
    requests = {s.sample_id: _sample_request(s, remove_duplicates=remove_duplicates) for s in samples}
    if not overwrite:
        changed: dict[str, list[str]] = {}
        for sample in samples:
            made_with = record["samples"].get(sample.sample_id)
            if not isinstance(made_with, dict) or not bam_complete(sample.output_bam):
                continue
            differences = request_differences(made_with, requests[sample.sample_id])
            if differences:
                changed[sample.sample_id] = differences
        refuse_mixed_results("Duplicate marking", workdir, changed)
    return record, requests


def _record_requests(
    samples: list[DedupSample], workdir: Path, record: dict[str, Any],
    requests: dict[str, dict[str, Any]], *, overwrite: bool,
) -> None:
    """Write the request down for every sample that is about to be made."""
    for sample in samples:
        if overwrite or not bam_complete(sample.output_bam):
            record["samples"][sample.sample_id] = requests[sample.sample_id]
    _save_requests(workdir, record)


def _invalidate_previous_run(workdir: Path, plan: dict[str, Any], samples: list[DedupSample], *, overwrite: bool) -> None:
    """Withdraw the earlier run's claim of completion before this run starts.

    A run that adds a sample to a finished project must not leave the earlier
    run's summary saying complete:true beside a manifest naming the new
    sample (the state reader takes that flag over the files). The summary is
    removed here and the checkpoint set incomplete; whichever path finishes
    the run writes both.

    With --overwrite every sample is redone, but the earlier run's BAM and
    index stay in place until each task replaces them, and a BAM with its
    index is a finished sample to every reader. Its index is removed now, so
    a sample reads finished again only once this run has made it.
    """
    (workdir / "results" / "dedup_summary.json").unlink(missing_ok=True)
    write_checkpoint(workdir, plan, complete=False, submitted=False, message="Duplicate marking started.")
    if overwrite:
        for sample in samples:
            Path(str(sample.output_bam) + ".bai").unlink(missing_ok=True)


def run_dedup_stage(
    *,
    samples_csv: Union[str, Path],
    workdir: Union[str, Path],
    backend: str = "local",
    threads: int = 4,
    jobs: int = 0,
    remove_duplicates: bool = False,
    overwrite: bool = False,
    submit: bool = False,
    dry_run: bool = False,
    partition: Optional[str] = None,
    mem: str = "16G",
    manifest_profile: str = "family",
    # Soft-clip the overlap of a pair's mates after marking (bamUtil clipOverlap).
    clip_overlap: bool = False,
) -> DedupResult:
    require_tool("samtools")
    samples_csv = Path(samples_csv)
    workdir = Path(workdir)
    backend = backend.lower().strip()
    if backend not in {"local", "slurm"}:
        raise ValueError("backend must be local or slurm")
    if not samples_csv.exists():
        raise FileNotFoundError(f"Sample manifest not found: {samples_csv}")

    ensure_dirs(workdir)
    samples = read_bam_samples(samples_csv, workdir, manifest_profile=manifest_profile)
    plan = build_plan(
        samples_csv=samples_csv, workdir=workdir, samples=samples, threads=threads,
        jobs=jobs, backend=backend, remove_duplicates=remove_duplicates,
        overwrite=overwrite, submit=submit, manifest_profile=manifest_profile,
    )
    # Refused before anything is written, a dry run included.
    record, requests = _prepare_requests(
        samples, workdir, overwrite=overwrite, remove_duplicates=remove_duplicates,
    )
    # Only a run that starts work withdraws the earlier run: a SLURM scripts-only run changes nothing
    # on disk, and the task script removes its own index.
    if not dry_run and (backend == "local" or submit):
        _invalidate_previous_run(workdir, plan, samples, overwrite=overwrite)
    if not dry_run:
        # The request is written for a scripts-only run too: it is what the
        # scripts will make when they are submitted.
        _record_requests(samples, workdir, record, requests, overwrite=overwrite)
    manifest, dedup_manifest = write_sample_manifests(workdir, samples)
    plan_json = workdir / "inputs" / "dedup_plan.json"
    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    shutil.copy2(samples_csv, workdir / "inputs" / "original_samples.csv")

    if dry_run:
        write_checkpoint(workdir, plan, complete=False, submitted=False, message="Dry run complete; duplicate marking was not executed.")
        return DedupResult(str(workdir), len(samples), 0, str(dedup_manifest), str(plan_json), str(stage_checkpoint_path(workdir, "deduplication")), str(workdir / "results" / "dedup_summary.json"), backend, False, None, False, "Dry run complete; duplicate marking was not executed.")

    active = [s.index for s in samples if overwrite or not bam_complete(s.output_bam)]

    if backend == "slurm":
        if not active:
            metrics = collect_metrics(samples, scan_read_counts=False, remove_duplicates=remove_duplicates,
                                      known=_current_counts(workdir))
            summary = write_summary(workdir, plan, metrics)
            uncounted = [m["sample_id"] for m in metrics["sample_metrics"] if m["complete"] and m.get("duplicate_reads") is None]
            if uncounted and submit:
                # BAMs with no counts yet (marked before the counts were written beside the BAM)
                # are counted by the summary job, on the cluster, rather than in this process.
                summary_script = write_summary_script(workdir, partition=partition)
                proc = subprocess.run(["sbatch", "--parsable", str(summary_script)], check=True, text=True, capture_output=True)
                job_id = proc.stdout.strip().split(";")[0]
                message = f"All duplicate-marked BAMs complete; counting the reads of {len(uncounted)} (dedup summarize, job {job_id})."
                # The BAMs are the stage's output and they are complete; the job only adds their counts, and
                # its ID is printed (print_result), so the pipeline waits for it before the next stage.
                write_checkpoint(workdir, plan, complete=True, submitted=True, job_id=job_id, message=message, metrics=metrics)
                return DedupResult(
                    str(workdir), len(samples), metrics["completed_samples"],
                    str(dedup_manifest), str(plan_json),
                    str(stage_checkpoint_path(workdir, "deduplication")),
                    str(summary), backend, True, job_id, True, message,
                )
            message = (
                "All duplicate-marked BAMs and indexes already complete; "
                "metadata refreshed without rescanning BAM read counts."
            )
            write_checkpoint(
                workdir,
                plan,
                complete=True,
                submitted=False,
                message=message,
                metrics=metrics,
            )
            return DedupResult(
                str(workdir), len(samples), metrics["completed_samples"],
                str(dedup_manifest), str(plan_json),
                str(stage_checkpoint_path(workdir, "deduplication")),
                str(summary), backend, False, None, True, message,
            )
        script = write_slurm_script(workdir=workdir, manifest=manifest, threads=threads, jobs=jobs, remove_duplicates=remove_duplicates, overwrite=overwrite, partition=partition, mem=mem, active_indices=active, clip_overlap=clip_overlap)
        summary_script = write_summary_script(workdir, partition=partition)
        job_id = None
        submitted = False
        if submit:
            proc = subprocess.run(["sbatch", "--parsable", str(script)], check=True, text=True, capture_output=True)
            job_id = proc.stdout.strip().split(";")[0]
            submitted = True
            # The summary, once every task has finished (every run has one): a SLURM run removes the
            # earlier summary when it starts, so this job writes the new one when its tasks finish.
            subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{job_id}", str(summary_script)],
                           check=True, text=True, capture_output=True)
        write_checkpoint(workdir, plan, complete=False, submitted=submitted, job_id=job_id, message="SLURM duplicate-marking jobs submitted." if submitted else "SLURM duplicate-marking script written.")
        return DedupResult(str(workdir), len(samples), len(samples)-len(active), str(dedup_manifest), str(plan_json), str(stage_checkpoint_path(workdir, "deduplication")), str(workdir / "results" / "dedup_summary.json"), backend, submitted, job_id, False, "SLURM duplicate-marking jobs submitted." if submitted else "SLURM duplicate-marking script written.")

    sample_metrics = []
    for sample in samples:
        sample_metrics.append(run_one_sample(sample, workdir=workdir, threads=threads, remove_duplicates=remove_duplicates, overwrite=overwrite, clip_overlap=clip_overlap))
    metrics = collect_metrics(samples, remove_duplicates=remove_duplicates)
    summary = write_summary(workdir, plan, metrics)
    write_checkpoint(workdir, plan, complete=bool(metrics["complete"]), submitted=False, message="Duplicate marking complete.", metrics=metrics)
    return DedupResult(str(workdir), len(samples), metrics["completed_samples"], str(dedup_manifest), str(plan_json), str(stage_checkpoint_path(workdir, "deduplication")), str(summary), backend, False, None, bool(metrics["complete"]), "Duplicate marking complete.")


def _check_dedup_sample_status(item: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    """Fast per-sample dedup output check.

    This intentionally uses filesystem metadata only. It does not run
    ``samtools view -c`` or otherwise scan BAM contents, so status/verify remain
    fast even for large exome/WGS BAMs. Full BAM validation remains available via
    explicit ``samtools quickcheck`` outside this lightweight workflow check.
    """
    idx, sample_id, info = item
    bam = info.get("bam")
    bai = str(bam) + ".bai" if bam else None
    return {
        "sample_id": sample_id,
        "bam": file_info(bam),
        "bai": file_info(bai),
        "complete": bam_complete(bam) if bam else False,
    }


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text())
    return {}


def _summary_metrics(summary: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return available dedup metrics without scanning BAMs."""
    if summary:
        return {
            "samples": summary.get("samples"),
            "completed_samples": summary.get("completed_samples"),
            "total_reads": summary.get("total_reads"),
            "duplicate_reads": summary.get("duplicate_reads"),
            "duplicate_rate": summary.get("duplicate_rate"),
            "sample_metrics": summary.get("sample_metrics", []),
        }
    metrics = checkpoint.get("metrics") or {}
    return {
        "samples": metrics.get("samples"),
        "completed_samples": metrics.get("completed_samples"),
        "total_reads": metrics.get("total_reads"),
        "duplicate_reads": metrics.get("duplicate_reads"),
        "duplicate_rate": metrics.get("duplicate_rate"),
        "sample_metrics": metrics.get("sample_metrics", []),
    }


def verify_dedup_stage(workdir: Union[str, Path], threads: int = 4) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    workdir = Path(workdir)
    plan_json = workdir / "inputs" / "dedup_plan.json"
    checkpoint = stage_checkpoint_path(workdir, "deduplication")
    summary_json = workdir / "results" / "dedup_summary.json"
    dedup_manifest = workdir / "inputs" / "dedup_samples.csv"

    plan = _load_json_if_present(plan_json)
    ckpt = _load_json_if_present(checkpoint)
    summary = _load_json_if_present(summary_json)

    sample_outputs = plan.get("sample_outputs") or ckpt.get("sample_outputs") or {}
    items = [(i, sample_id, info) for i, (sample_id, info) in enumerate(sample_outputs.items(), start=1)]

    workers = max(1, int(threads or 1))
    if items:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            samples = list(executor.map(_check_dedup_sample_status, items))
    else:
        samples = []

    completed = sum(1 for s in samples if s["complete"])
    complete = bool(samples) and completed == len(samples)
    metrics = _summary_metrics(summary, ckpt)

    return {
        "schema": "paralome.dedup.verify.v1",
        "stage": "deduplication",
        "workdir": str(workdir),
        "complete": complete,
        "samples": len(samples),
        "completed_samples": completed,
        "checkpoint": file_info(checkpoint),
        "plan_json": file_info(plan_json),
        "summary": file_info(summary_json),
        "dedup_manifest": file_info(dedup_manifest),
        "remove_duplicates": ckpt.get("remove_duplicates") if ckpt else plan.get("remove_duplicates"),
        "metrics": metrics,
        "sample_status": samples,
    }


def print_result(result: DedupResult) -> None:
    print("Duplicate marking stage")
    print("=======================")
    print(f"Workdir             : {result.workdir}")
    print(f"Samples             : {result.samples}")
    print(f"Completed samples   : {result.completed_samples}/{result.samples}")
    print(f"Backend             : {result.backend}")
    print(f"Submitted           : {result.submitted}")
    if result.job_id:
        print(f"Job ID              : {result.job_id}")
    print(f"Dedup manifest      : {result.dedup_manifest}")
    print(f"Plan JSON           : {result.plan_json}")
    print(f"Checkpoint          : {result.checkpoint}")
    print(f"Summary             : {result.summary}")
    print(f"Complete            : {result.complete}")
    print(f"Message             : {result.message}")


def print_verify(report: dict[str, Any]) -> None:
    metrics = report.get("metrics") or {}
    print("Duplicate marking verification")
    print("==============================")
    print(f"Workdir             : {report.get('workdir')}")
    print(f"Complete            : {report.get('complete')}")
    print(f"Samples             : {report.get('completed_samples')}/{report.get('samples')}")
    print(f"Remove duplicates   : {report.get('remove_duplicates')}")
    print(f"Duplicate reads     : {metrics.get('duplicate_reads')}")
    rate = metrics.get("duplicate_rate")
    print(f"Duplicate rate      : {rate * 100:.3f}%" if isinstance(rate, (int, float)) else "Duplicate rate      : .")
    print(f"Dedup manifest      : {'OK' if report['dedup_manifest']['nonempty'] else 'MISSING'} {report['dedup_manifest']['path']}")
    print(f"Checkpoint          : {'OK' if report['checkpoint']['nonempty'] else 'MISSING'} {report['checkpoint']['path']}")
    print("")
    print("Samples")
    print("-------")
    for row in report.get("sample_status", []):
        flag = "OK" if row.get("complete") else "MISSING"
        print(f"{row.get('sample_id')}\t{flag}\t{row.get('bam', {}).get('path')}")


#: What a fused alignment task writes, and therefore what this stage adopts.
FUSED_BAM_SUFFIX = ".markdup.bam"


def _fused_samples(workdir: Path, samples_csv: Union[str, Path]) -> list[DedupSample]:
    """The stage's sample rows, built from the FASTQ sheet a fused run started from.

    The metadata matters and cannot be skipped: the dedup manifest this writes
    is what MACS reads for each sample's condition and what the variant stages
    read for family, sex and affected status. A row with those blank would let
    the pipeline run on and produce comparisons with no groups in them.
    """
    from paralome.fastq.manifest import read_sample_inputs as _read

    rows: list[DedupSample] = []
    for i, sample in enumerate(_read(Path(samples_csv)), start=1):
        extra = dict(sample.extra or {})
        rows.append(DedupSample(
            index=i,
            sample_id=sample.sample_id,
            group=sample.group,
            condition=sample.condition,
            affected=extra.get("affected", "."),
            relationship=sample.relationship,
            family_id=sample.family_id,
            sex=sample.sex,
            mother_id=extra.get("mother_id", "."),
            father_id=extra.get("father_id", "."),
            # A fused run never wrote one; the field is kept so the manifest
            # has the shape every reader expects.
            input_bam="",
            output_bam=str(workdir / "bam" / f"{sample.sample_id}{FUSED_BAM_SUFFIX}"),
        ))
    return rows


def adopt_fused_results(
    workdir: Union[str, Path],
    *,
    sample_ids: Optional[list[str]] = None,
    samples_csv: Optional[Union[str, Path]] = None,
    remove_duplicates: bool,
    alignment_workdir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Complete this stage from BAMs a fused alignment task already wrote.

    With `--fuse-stages` the duplicate marking happens inside the
    alignment task, on STAR's output as it arrives, so there is no aligned BAM
    to run this stage against and nothing left for it to compute. What the
    stage still owes its readers is its own paperwork: the plan, the summary
    and the checkpoint that every downstream stage and every report looks for.
    This writes those from the BAMs that are there.

    It does not re-mark anything. If a BAM is missing the stage is reported
    incomplete, which is the truthful answer -- the alignment task that owed it
    did not finish.

    WHAT CANNOT BE MEASURED HERE. With ``remove_duplicates`` the duplicates are
    gone from the output and there is no input BAM to count them against (that
    BAM is exactly what the fusion stopped writing), so the duplicate counts
    are reported as unavailable rather than as zero. Keeping the duplicates --
    marking without removing, which is Paralome's default -- costs nothing in a
    fused run and keeps the rate measurable, which is a good reason to prefer
    it.
    """
    workdir = Path(workdir)
    if samples_csv is not None:
        samples = _fused_samples(workdir, samples_csv)
    elif sample_ids:
        samples = [
            DedupSample(
                index=i, sample_id=sample_id, group=".", condition=".", affected=".",
                relationship=".", family_id=".", sex=".", mother_id=".", father_id=".",
                input_bam="",
                output_bam=str(workdir / "bam" / f"{sample_id}{FUSED_BAM_SUFFIX}"),
            )
            for i, sample_id in enumerate(sample_ids, start=1)
        ]
    else:
        raise ValueError("adopt_fused_results needs samples_csv or sample_ids")
    metrics = collect_metrics(
        samples,
        # Counting duplicates needs only the output BAM's flags when they are
        # still in it; with -r it would need the input BAM, which no longer
        # exists in a fused run.
        scan_read_counts=not remove_duplicates,
        remove_duplicates=False,
    )
    plan = {
        "schema": SCHEMA_PLAN,
        "stage": "deduplication",
        "backend": "fused-alignment",
        "remove_duplicates": remove_duplicates,
        "alignment_workdir": str(alignment_workdir) if alignment_workdir else None,
        "samples": [s.sample_id for s in samples],
        "outputs": {s.sample_id: {"bam": s.output_bam, "bai": s.output_bai} for s in samples},
        "note": ("duplicate marking ran inside the alignment task; this stage adopted "
                 "its output rather than reading an aligned BAM, which a fused run "
                 "never writes"),
        "created": now_iso(),
    }
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    (workdir / "inputs").mkdir(parents=True, exist_ok=True)
    # The manifests the rest of the pipeline reads: MACS takes each sample's
    # condition from dedup_samples.csv, and the variant stages take family,
    # sex and affected status from it.
    _manifest, dedup_samples = write_sample_manifests(workdir, samples)
    plan["dedup_manifest"] = str(dedup_samples)
    write_summary(workdir, plan, metrics)
    write_checkpoint(
        workdir, plan,
        complete=bool(metrics.get("complete")),
        message=("adopted from the fused alignment task" if metrics.get("complete")
                 else "a fused alignment task has not produced every duplicate-marked BAM yet"),
        metrics=metrics,
    )
    return {
        "workdir": str(workdir),
        "complete": bool(metrics.get("complete")),
        "samples": metrics.get("samples"),
        "completed_samples": metrics.get("completed_samples"),
        "duplicate_rate": metrics.get("duplicate_rate"),
        "dedup_manifest": str(dedup_samples),
        "backend": "fused-alignment",
    }
