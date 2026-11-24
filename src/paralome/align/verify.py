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

import csv
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Union

from paralome.checkpoints import load_stage_checkpoint, stage_checkpoint_path, update_stage_checkpoint
from paralome.csvio import skip_comment_lines


@dataclass
class FileStatus:
    path: str
    exists: bool
    nonempty: bool
    size: int

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "FileStatus":
        p = Path(path)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return cls(path=str(p), exists=exists, nonempty=exists and size > 0, size=size)


@dataclass
class AlignSampleStatus:
    sample_id: str
    bam: FileStatus
    bai: FileStatus
    complete: bool
    # Why a BAM that is present was not accepted, and its mapped reads.
    problem: Optional[str] = None
    mapped_reads: Optional[int] = None
    # True when the BAM was read and found unusable, not merely left unchecked.
    rejected: bool = False
    # False when samtools could not be run or gave no verdict on this BAM.
    checked: bool = True
    # True while --overwrite has marked the sample and it is not aligned again.
    to_be_replaced: bool = False


@dataclass
class StarResourceStatus:
    backend: str
    threads_per_sample: Optional[int]
    concurrent_jobs: Optional[int]
    peak_cpus: Optional[int]
    slurm_array: Optional[str]
    slurm_cpus_per_task: Optional[int]
    slurm_mem: Optional[str]
    samtools_index_threads: Optional[int] = None
    max_samtools_index_threads: Optional[int] = None
    star_sort_ram_bytes: Optional[int] = None


@dataclass
class StarVerifyReport:
    workdir: str
    samples: int
    complete_samples: int
    incomplete_samples: int
    resources: StarResourceStatus
    aligned_manifest: FileStatus
    script: FileStatus
    sample_status: list[AlignSampleStatus]
    checkpoint: FileStatus | None = None
    missing_sample_ids: Optional[list[str]] = None
    failed_sample_ids: Optional[list[str]] = None

    @property
    def complete(self) -> bool:
        return self.samples > 0 and self.samples == self.complete_samples

    @property
    def could_not_check(self) -> bool:
        """Some BAM is there and samtools gave no verdict on it. The supervisor reads this key."""
        return any(not s.checked for s in self.sample_status)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["complete"] = self.complete
        d["could_not_check"] = self.could_not_check
        return d


#: How a sample's ``failed`` field starts when verification rejected its BAM.
VERIFY_REJECTED = "align verify: "


#: The mark `align star --overwrite` leaves in a sample's checkpoint metadata.
TO_BE_REPLACED = "to_be_replaced"


def _file_can_be_read(path: str) -> bool:
    """Can this process read the start and the end of the file?"""
    try:
        with open(path, "rb") as fh:
            fh.read(4096)
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            fh.read(4096)
    except OSError:
        return False
    return True


def _samtools_answers(samtools: str) -> bool:
    """Does this samtools run at all? One with a missing library exits non-zero on everything."""
    try:
        return subprocess.run([samtools, "--version"], text=True, capture_output=True, check=False).returncode == 0
    except OSError:
        return False


def _inspect_bam(bam: FileStatus, bai: FileStatus) -> tuple[Optional[str], Optional[int], bool, bool]:
    """Why this BAM is not accepted, or None; its mapped reads; rejected; checked.

    Asked of samtools: quickcheck for a whole file, idxstats for the count, so
    a BAM holding a header and no reads, or one cut short, is not accepted.

    REJECTED IS A VERDICT ON THE BAM, AND ONLY THAT. Two things are: a
    quickcheck that read the file and failed, and an index that was read and
    counts no mapped read. Every other failure (no samtools, a samtools that
    does not run, a file this process cannot read, an idxstats that failed)
    says nothing about the BAM: it is not accepted, and it is reported as not
    checked, so a stale file handle on one node never condemns a good BAM.
    """
    if not (bam.nonempty and bai.nonempty):
        return None, None, False, True
    from paralome.binaries import resolve_tool

    samtools = resolve_tool("samtools")
    if samtools is None:
        return "samtools was not found, so the BAM could not be checked", None, False, False
    try:
        check = subprocess.run([samtools, "quickcheck", bam.path], text=True, capture_output=True, check=False)
        if check.returncode != 0:
            # Killed, not executable, not found (126, 127), or failing on a
            # file that cannot be read here: none of that is the BAM's fault.
            ran = 0 < check.returncode < 126
            if ran and _file_can_be_read(bam.path) and _samtools_answers(samtools):
                return "the BAM is truncated or not a BAM (samtools quickcheck failed)", None, True, True
            message = (check.stderr or "").strip().splitlines()
            return (
                "samtools quickcheck gave no verdict (" + (message[-1] if message else f"exit {check.returncode}")
                + "), so the BAM could not be checked"
            ), None, False, False
        stats = subprocess.run([samtools, "idxstats", bam.path], text=True, capture_output=True, check=False)
    except OSError as exc:
        return f"samtools could not be run ({exc}), so the BAM could not be checked", None, False, False
    if stats.returncode != 0:
        message = (stats.stderr or "").strip().splitlines()
        return (
            "samtools idxstats failed (" + (message[-1] if message else f"exit {stats.returncode}")
            + "), so the mapped reads could not be counted"
        ), None, False, False
    mapped = 0
    counted = False
    for line in stats.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and fields[2].strip().isdigit():
            mapped += int(fields[2])
            counted = True
    if not counted:
        return "samtools idxstats listed no sequences, so the mapped reads could not be counted", None, False, False
    if mapped == 0:
        return "the BAM has no mapped reads", 0, True, True
    return None, mapped, False, True


def _sample_status(sample_id: str, bam_path: Union[str, Path], bai_path: Union[str, Path]) -> AlignSampleStatus:
    bam = FileStatus.from_path(bam_path)
    bai = FileStatus.from_path(bai_path)
    problem, mapped, rejected, checked = _inspect_bam(bam, bai)
    return AlignSampleStatus(
        sample_id=sample_id, bam=bam, bai=bai,
        complete=bam.nonempty and bai.nonempty and problem is None,
        problem=problem, mapped_reads=mapped, rejected=rejected, checked=checked,
    )


def _awaiting_replacement(sample_id: str, bam_path: Union[str, Path], bai_path: Union[str, Path]) -> AlignSampleStatus:
    """A sample --overwrite marked and no task has aligned again: whatever BAM is there is the old one."""
    return AlignSampleStatus(
        sample_id=sample_id, bam=FileStatus.from_path(bam_path), bai=FileStatus.from_path(bai_path),
        complete=False, problem="is still to be aligned again (--overwrite); the BAM in place is the earlier one",
        to_be_replaced=True,
    )


def no_reads_advice(sample_ids: list[str]) -> str:
    """What is true of a sample that aligned no reads, for the report and for PLM-0312.

    Such a sample (an empty negative control, the wrong species) was
    told to remove its BAM and align again, which gives the same empty BAM.
    """
    return (
        "Aligned no reads: " + ", ".join(sample_ids) + ". Aligning again gives the same empty BAM. "
        "To go on without such a sample, take it out of the sample sheet (put '#' at the start of its "
        "line) and run again. If its FASTQ is the wrong file, correct that, remove the sample's BAM "
        "and .bai, and run again."
    )


def _withdraw_completion(workdir: Path, problems: dict[str, str], accepted: Optional[set[str]] = None) -> None:
    """Mark samples whose BAM failed the check as not complete in the checkpoint.

    The pipeline takes the stage as complete when either this report
    or the stage checkpoint says so, and the array task has already marked the
    sample complete there. `align star` reads the mark (VERIFY_REJECTED) and
    refuses to go on over such a BAM. A sample in `accepted` that carries the
    mark from an earlier verification loses it, so a BAM rejected once over a
    passing I/O error is not rejected for ever.
    """
    def withdraw(stage: Any) -> None:
        for sample_id, problem in problems.items():
            sample = stage.samples.get(sample_id)
            if sample is not None:
                sample.complete = False
                sample.finished = None
                sample.failed = f"{VERIFY_REJECTED}{problem}"
        for sample_id in accepted or ():
            sample = stage.samples.get(sample_id)
            if sample is not None and str(sample.failed or "").startswith(VERIFY_REJECTED):
                stage.mark_complete(sample_id)

    try:
        update_stage_checkpoint(workdir, "alignment", withdraw)
    except FileNotFoundError:
        pass


def _parse_sbatch_value(script_text: str, key: str) -> Optional[str]:
    pattern = rf"^#SBATCH\s+--{re.escape(key)}=(.+)$"
    for line in script_text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            return m.group(1).strip()
    return None


def _parse_array(array_value: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    if not array_value:
        return None, None
    # Examples: 1-3%3, 1-400%80, 1-3
    throttle = None
    spec = array_value
    if "%" in array_value:
        spec, throttle_s = array_value.split("%", 1)
        try:
            throttle = int(throttle_s)
        except ValueError:
            throttle = None
    return throttle, array_value


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(skip_comment_lines(fh))
        return [dict(row) for row in reader]


def _sample_id_from_manifest_row(row: dict[str, str]) -> str:
    for key in ("sample_id", "sample", "id", "Sample", "SAMPLE"):
        value = row.get(key)
        if value:
            return value
    return row.get("bam", "unknown")


def _parse_assignment_int(script_text: str, name: str) -> Optional[int]:
    pattern = rf"^{re.escape(name)}=(\d+)\s*$"
    for line in script_text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _parse_index_thread_cap(script_text: str) -> Optional[int]:
    # Generated STAR scripts cap samtools index threads with:
    #   if (( INDEX_THREADS > 4 )); then
    for line in script_text.splitlines():
        m = re.search(r"INDEX_THREADS\s*>\s*(\d+)", line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _resource_int(resources: dict[str, Any], key: str) -> Optional[int]:
    value = resources.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def verify_star_alignment(workdir: Union[str, Path], *, write_json: bool = True) -> StarVerifyReport:
    workdir = Path(workdir)
    script = workdir / "scripts" / "star_array.sbatch"
    aligned_manifest = workdir / "inputs" / "aligned_samples.csv"
    checkpoint_file = stage_checkpoint_path(workdir, "alignment")
    checkpoint = load_stage_checkpoint(workdir, "alignment")
    checkpoint_resources: dict[str, Any] = checkpoint.resources if checkpoint is not None else {}

    script_text = script.read_text() if script.exists() else ""
    cpus_s = _parse_sbatch_value(script_text, "cpus-per-task")
    mem = _parse_sbatch_value(script_text, "mem")
    array_value = _parse_sbatch_value(script_text, "array")

    cpus = None
    if cpus_s:
        try:
            cpus = int(cpus_s)
        except ValueError:
            cpus = None

    concurrent, array_raw = _parse_array(array_value)

    # Local backend generally has no sbatch script. If the script exists, call it slurm.
    backend = "slurm" if script.exists() and array_value else "local"
    # Unless the checkpoint says it ran here. A fused local run
    # writes the same task script and runs it on this machine.
    if str(checkpoint_resources.get("backend") or "") == "local":
        backend = "local"
    peak = cpus * concurrent if cpus is not None and concurrent is not None else None

    max_index_threads = (
        _resource_int(checkpoint_resources, "max_samtools_index_threads")
        or _parse_index_thread_cap(script_text)
    )
    index_threads = _resource_int(checkpoint_resources, "samtools_index_threads")
    if index_threads is None and cpus is not None and "samtools index -@" in script_text:
        index_threads = max(1, int(cpus))
        if max_index_threads is not None:
            index_threads = min(max_index_threads, index_threads)
    star_sort_ram_bytes = (
        _resource_int(checkpoint_resources, "star_sort_ram_bytes")
        or _parse_assignment_int(script_text, "STAR_SORT_RAM")
    )

    rows = _read_manifest(aligned_manifest)
    sample_status: list[AlignSampleStatus] = []
    for row in rows:
        sample_id = _sample_id_from_manifest_row(row)
        bam_path = row.get("bam") or row.get("bam_path") or str(workdir / "bam" / f"{sample_id}.sorted.bam")
        bai_path = str(Path(bam_path).with_suffix(Path(bam_path).suffix + ".bai")) if bam_path.endswith(".bam") else bam_path + ".bai"
        if checkpoint is None:
            sample_status.append(_sample_status(sample_id, bam_path, bai_path))

    # If manifest not populated yet, infer sample IDs from BAM names.
    if not rows and checkpoint is None:
        for bam_path in sorted((workdir / "bam").glob("*.bam")):
            sample_id = bam_path.name.replace(".sorted.bam", "").replace(".bam", "")
            sample_status.append(_sample_status(sample_id, bam_path, str(bam_path) + ".bai"))

    # Prefer checkpoint sample universe if available, and refresh completion from BAM/BAI files.
    if checkpoint is not None:
        refreshed: list[AlignSampleStatus] = []
        for sample_id, sample_ckpt in checkpoint.samples.items():
            bam_path = sample_ckpt.outputs.get("bam") or str(workdir / "bam" / f"{sample_id}.sorted.bam")
            bai_path = sample_ckpt.outputs.get("bai") or (str(bam_path) + ".bai")
            if sample_ckpt.metadata.get(TO_BE_REPLACED):
                # Not inspected. Its old BAM would pass, and be counted as done.
                refreshed.append(_awaiting_replacement(sample_id, bam_path, bai_path))
            else:
                refreshed.append(_sample_status(sample_id, bam_path, bai_path))
        sample_status = refreshed
        problems = {s.sample_id: s.problem for s in sample_status if s.rejected and s.problem}
        cleared = {
            s.sample_id for s in sample_status
            if s.complete and str(checkpoint.samples[s.sample_id].failed or "").startswith(VERIFY_REJECTED)
        }
        if (problems or cleared) and write_json:
            _withdraw_completion(workdir, problems, cleared)
            checkpoint = load_stage_checkpoint(workdir, "alignment") or checkpoint

    complete_samples = sum(1 for s in sample_status if s.complete)
    missing_sample_ids = [s.sample_id for s in sample_status if not s.complete]
    failed_sample_ids = checkpoint.failed_samples() if checkpoint is not None else []
    report = StarVerifyReport(
        workdir=str(workdir),
        samples=len(sample_status),
        complete_samples=complete_samples,
        incomplete_samples=len(sample_status) - complete_samples,
        resources=StarResourceStatus(
            backend=backend,
            threads_per_sample=cpus,
            concurrent_jobs=concurrent,
            peak_cpus=peak,
            slurm_array=array_raw,
            slurm_cpus_per_task=cpus,
            slurm_mem=mem,
            samtools_index_threads=index_threads,
            max_samtools_index_threads=max_index_threads,
            star_sort_ram_bytes=star_sort_ram_bytes,
        ),
        aligned_manifest=FileStatus.from_path(aligned_manifest),
        script=FileStatus.from_path(script),
        sample_status=sample_status,
        checkpoint=FileStatus.from_path(checkpoint_file),
        missing_sample_ids=missing_sample_ids,
        failed_sample_ids=failed_sample_ids,
    )

    if write_json:
        out = workdir / "results" / "star_alignment_verify.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")

    return report


def print_star_verify_report(report: StarVerifyReport) -> None:
    r = report.resources
    print("STAR alignment verification")
    print("===========================")
    print(f"Workdir              : {report.workdir}")
    print(f"Samples              : {report.samples}")
    print(f"Completed            : {report.complete_samples}/{report.samples}")
    print(f"Incomplete           : {report.incomplete_samples}")
    print(f"Backend              : {r.backend}")
    print(f"Threads/sample       : {r.threads_per_sample if r.threads_per_sample is not None else 'unknown'}")
    if r.samtools_index_threads is not None:
        print(f"Index threads/sample : {r.samtools_index_threads}")
    if r.max_samtools_index_threads is not None:
        print(f"Max index threads    : {r.max_samtools_index_threads}")
    print(f"Concurrent jobs      : {r.concurrent_jobs if r.concurrent_jobs is not None else 'unknown'}")
    print(f"Peak CPUs            : {r.peak_cpus if r.peak_cpus is not None else 'unknown'}")
    if r.slurm_array and r.backend == "slurm":
        print(f"SLURM array          : {r.slurm_array}")
    if r.slurm_mem and r.backend == "slurm":
        print(f"Memory/sample        : {r.slurm_mem}")
    if r.star_sort_ram_bytes is not None:
        print(f"STAR sort RAM        : {r.star_sort_ram_bytes} bytes")
    print(f"Aligned manifest     : {'OK' if report.aligned_manifest.nonempty else 'missing'} {report.aligned_manifest.path}")
    print(f"STAR script          : {'OK' if report.script.nonempty else 'missing'} {report.script.path}")
    if report.checkpoint is not None:
        print(f"Checkpoint           : {'OK' if report.checkpoint.nonempty else 'missing'} {report.checkpoint.path}")
    print(f"Complete             : {str(report.complete).lower()}")
    if report.missing_sample_ids:
        print(f"Missing samples      : {', '.join(report.missing_sample_ids)}")
    if report.failed_sample_ids:
        print(f"Failed samples       : {', '.join(report.failed_sample_ids)}")

    if report.sample_status:
        print("")
        print("Samples")
        print("-------")
        for s in report.sample_status:
            status = "OK" if s.complete else "INCOMPLETE"
            print(f"{s.sample_id}\t{status}\tBAM={s.bam.path}\tBAI={s.bai.path}")
        unusable = [s for s in report.sample_status if s.problem]
        if unusable:
            # Nothing is deleted here. The user removes what failed.
            print("")
            print("Not accepted")
            print("------------")
            for s in unusable:
                print(f"{s.sample_id}: {s.problem}: {s.bam.path}")
            no_reads = [s.sample_id for s in unusable if s.rejected and s.mapped_reads == 0]
            if any(s.rejected and s.mapped_reads != 0 for s in unusable):
                print("Remove the rejected BAMs and their .bai files (or the whole alignment folder) and run the alignment again.")
            if no_reads:
                print(no_reads_advice(no_reads))
            if report.could_not_check:
                print("A BAM that could not be checked is not rejected: nothing is wrong with it that is known. Run `align verify` again where samtools works.")
