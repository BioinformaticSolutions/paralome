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

import contextlib
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from paralome.csvio import skip_comment_lines
from paralome.stream import (FUSED_BUFFER_BYTES, FUSED_READER, fastp_fifo_bash,
                             fused_buffer_overhead, fused_read_files_command,
                             markdup_stream_bash, star_stdout_arguments,
                             wait_for_fastp_bash)
from paralome.binaries import shell_join, worker_argv
from paralome.align.verify import TO_BE_REPLACED, VERIFY_REJECTED
from paralome.dedup.counts import FLAGSTAT_SIDECAR
from paralome.fastq.fastp import input_signature, refuse_mixed_results, refuse_while_jobs_are_alive, request_differences
from paralome.pipeline.naming import slurm_job_name
from paralome.scratch import job_scratch_bash, job_scratch_dir
from paralome.pipeline.resources import (
    parse_mem_to_mb, plan_fused_alignment_memory, plan_sample_parallel, plan_star_memory,
)
from paralome.pipeline.scheduler import detect_scheduler_capabilities, format_plan_diagnostics, plan_array_tasks
from paralome.pipeline.slurm import SlurmOptions, afterok, placeable_cpus, sbatch, tmp_directive, write_sbatch

from paralome.checkpoints import SampleCheckpoint, StageCheckpoint, load_stage_checkpoint, save_stage_checkpoint, slurm_array_spec, stage_checkpoint_path, update_stage_checkpoint


MAX_STAR_THREADS_PER_SAMPLE = 40
STAR_SORT_RAM_FRACTION = 0.90


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class FastqSample:
    sample_id: str
    group: str = "."
    relationship: str = "."
    family_id: str = "."
    sex: str = "."
    condition: str = "."
    mother_id: str = "."
    father_id: str = "."
    fastq_r1: str = ""
    fastq_r2: str = ""

    def to_bam_row(self, bam: Path) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "family_id": self.family_id,
            "relationship": self.relationship,
            "affected": self.group,
            "sex": self.sex,
            "condition": self.condition,
            "group": self.group,
            "mother_id": self.mother_id,
            "father_id": self.father_id,
            "bam": str(bam),
        }


def _pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] not in (None, "", "."):
            return str(row[name]).strip()
    return ""


def read_fastq_manifest(path: Union[str, Path]) -> list[FastqSample]:
    path = Path(path)
    # Utf-8-sig, as read_sample_inputs and read_sample_manifest
    # already do. An Excel-saved CSV carries a BOM in front of the
    # first header, so `sample_id` read as `\ufeffsample_id` and the sheet was
    # refused with "missing sample_id" when handed straight to `align star`.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(skip_comment_lines(fh))
        if not reader.fieldnames:
            raise ValueError(f"Sample manifest has no header: {path}")
        samples: list[FastqSample] = []
        for row in reader:
            sid = _pick(row, "sample_id", "sample", "id", "name")
            if not sid:
                raise ValueError(f"Manifest row is missing sample_id: {row}")
            r1 = _pick(row, "fastq_r1", "fastq1", "r1", "R1")
            r2 = _pick(row, "fastq_r2", "fastq2", "r2", "R2")
            if not r1:
                raise ValueError(f"Sample {sid} is missing fastq_r1/R1")
            group = _pick(row, "affected", "group", "case_control", "status") or "."
            samples.append(FastqSample(
                sample_id=sid,
                group=group,
                relationship=_pick(row, "relationship", "relation") or ".",
                family_id=_pick(row, "family_id", "family") or ".",
                sex=_pick(row, "sex", "gender") or ".",
                condition=_pick(row, "condition", "group", "status") or group,
                mother_id=_pick(row, "mother_id", "maternal_id") or ".",
                father_id=_pick(row, "father_id", "paternal_id") or ".",
                fastq_r1=r1,
                fastq_r2=r2,
            ))
        return samples


def validate_fastqs(samples: list[FastqSample]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for s in samples:
        if s.sample_id in seen:
            errors.append(f"Duplicate sample_id: {s.sample_id}")
        seen.add(s.sample_id)
        for label, fp in (("R1", s.fastq_r1), ("R2", s.fastq_r2)):
            if not fp:
                if label == "R2":
                    continue
                errors.append(f"{s.sample_id}: missing {label}")
                continue
            p = Path(fp)
            if not p.exists():
                errors.append(f"{s.sample_id}: {label} not found: {p}")
            elif p.stat().st_size == 0:
                errors.append(f"{s.sample_id}: {label} is empty: {p}")
    return errors


from paralome.binaries import resolve_tool


def _require_tool(name: str) -> str:
    """The tool to invoke, from what setup recorded -- not from PATH.

    See binaries.resolve_tool. Returns the resolved absolute path so
    the caller can invoke THAT rather than the bare name it asked about; a
    recorded path that no longer resolves raises PLM-0410 instead of silently
    falling back to whatever this account's PATH happens to find.
    """
    exe = resolve_tool(name)
    if exe is None:
        # The lookup consults the setup record first and PATH second, and the message names both.
        raise RuntimeError(
            f"Required executable not found (recorded path or PATH): {name}")
    return exe


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, ".") for k in fieldnames})


def _write_checkpoint(workdir: Path, data: dict[str, Any]) -> None:
    ckpt = stage_checkpoint_path(workdir, "pipeline")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if ckpt.exists():
        try:
            existing = json.loads(ckpt.read_text())
        except Exception:
            existing = {}
    existing.update(data)
    existing["updated"] = now_iso()
    tmp = ckpt.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    tmp.replace(ckpt)


def index_already_has_junctions(reference: Union[str, Path]) -> bool:
    """Was this STAR index built with --sjdbGTFfile?

    Passing --sjdbGTFfile to an index that already holds the annotation makes
    STAR build the junction database twice (its log marks the parameter
    `~RE-DEFINED`), and reads that cross a junction stop aligning; they are
    reported as "unmapped: too short". STAR's manual reserves on-the-fly
    insertion for indexes generated WITHOUT annotation, so the GTF is passed
    only when the index lacks it.

    The answer is read from the index's genomeParameters.txt, the same key
    ``setup.genome_scan._star_index_sources`` reads to find the GTF at
    registration.

    Unreadable or absent parameters return False: the GTF is passed and STAR
    decides. A missing file is not evidence that the index has junctions.
    """
    parameters = Path(reference) / "genomeParameters.txt"
    try:
        text = parameters.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        # The `### STAR --runMode genomeGenerate ...` comment line repeats the
        # same paths and must not be read as a key/value pair.
        if line.startswith("###"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == "sjdbGTFfile":
            value = parts[1].strip()
            return bool(value) and value != "-"
    return False


_MEM_UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def mem_plus_bytes(mem: str, extra_bytes: int) -> str:
    """A SLURM --mem string with room added, rounded up to whole GB.

    A fused task holds two read buffers that the aligner knows
    nothing about. Asking for STAR's memory alone put them outside the cgroup
    limit, where the kernel kills the task rather than the buffer.
    """
    text = str(mem).strip().upper().rstrip("B")
    unit = text[-1] if text and text[-1] in _MEM_UNITS else "M"
    number = text[:-1] if text and text[-1] in _MEM_UNITS else text
    try:
        base = int(float(number) * _MEM_UNITS[unit])
    except ValueError:
        # Not a shape we can do arithmetic on; leave the caller's request alone
        # rather than guess, and let the scheduler reject it as it would have.
        return str(mem)
    total = base + max(0, int(extra_bytes))
    return f"{-(-total // 1024 ** 3)}G"


#: STAR's DNA mode, by assay: the largest distance between the mates of a pair.
UNSPLICED_MATES_GAP = {"dna": 1000, "atac": 2000}
#: The arguments DNA mode adds or withholds; a BAM made before DNA mode differs from a new request
#: only in these, and is accepted as it is (_refuse_mixed_alignment).
_DNA_MODE_KEYS = ("--alignIntronMax", "--alignMatesGapMax", "--sjdbGTFfile")

#: Mismatches are limited by the fraction of read length alone. STAR has no switch to turn its absolute cap off and
#: defaults it to 10, so it is always given a cap no read reaches; it is not an option.
MISMATCH_NMAX_OFF = 999
#: A read with two equally good places gets its primary record at one of them at random (STAR's fixed --runRNGseed keeps runs reproducible),
#: rather than always the same one; the other stays secondary (--outSAMprimaryFlag OneBestScore, the default).
MULTIMAPPER_ORDER = ("--outMultimapperOrder", "Random")
#: Fixed by Paralome, not settable: a BAM aligned by an older Paralome (cap 10, STAR's default order) differs from a
#: new request only in these, and is accepted as it is, as a BAM aligned before DNA mode is.
_MISMATCH_NMAX_FLAG = "--outFilterMismatchNmax"
_SETTLED_KEYS = (_MISMATCH_NMAX_FLAG, MULTIMAPPER_ORDER[0])


def unspliced_arguments(mates_gap: int) -> list[str]:
    return ["--alignIntronMax", "1", "--alignMatesGapMax", str(int(mates_gap))]


def _without_dna_mode(arguments: Any, keys: tuple = _DNA_MODE_KEYS) -> Any:
    if not isinstance(arguments, list):
        return arguments
    kept: list[Any] = []
    skip = False
    for item in arguments:
        if skip:
            skip = False
            continue
        if item in keys:
            skip = True
            continue
        kept.append(item)
    return kept


def _star_result_arguments(
    *,
    reference: Path,
    gtf: Optional[Path],
    extra_args: str,
    read_files_command: str = "zcat",
    out_reads_unmapped: str = "None",
    out_filter_multimap_nmax: int = 2,
    out_filter_mismatch_nover_read_lmax: float = 0.04,
    out_filter_mismatch_nmax: int = MISMATCH_NMAX_OFF,
    out_filter_score_min_over_lread: float = 0.66,
    out_filter_match_nmin_over_lread: float = 0.66,
    sorted_output: bool = True,
    unspliced_mates_gap: Optional[int] = None,
) -> list[str]:
    """Every STAR argument that decides the alignment and is the same for all samples.

    Built once, for the local command and for the array script, so the two
    cannot differ. It is also what is recorded with each finished sample, so a
    resumed run can tell a different request from the same one.
    """
    args: list[str] = []
    # An empty --star-read-files-command means uncompressed FASTQ
    # and no --readFilesCommand at all. The array script passed STAR an empty
    # argument for it.
    read_command = str(read_files_command or "").strip()
    if read_command:
        args += ["--readFilesCommand", read_command]
    # A sorted BAM file, or an unsorted stream on stdout for a task
    # that goes straight into duplicate marking -- which sorts by name and then
    # by position anyway, so sorting here was work thrown away.
    args += star_stdout_arguments(sorted_output=sorted_output)
    args += [
        "--outReadsUnmapped", str(out_reads_unmapped),
        "--outFilterMultimapNmax", str(int(out_filter_multimap_nmax)),
        "--outFilterMismatchNoverReadLmax", str(float(out_filter_mismatch_nover_read_lmax)),
        # STAR's absolute mismatch cap defaults to 10 and applies alongside the fraction; it is given
        # MISMATCH_NMAX_OFF, so the fraction alone decides.
        _MISMATCH_NMAX_FLAG, str(int(out_filter_mismatch_nmax)),
        *MULTIMAPPER_ORDER,
        "--outFilterScoreMinOverLread", str(float(out_filter_score_min_over_lread)),
        "--outFilterMatchNminOverLread", str(float(out_filter_match_nmin_over_lread)),
    ]
    if unspliced_mates_gap:
        # DNA mode (DNA and ATAC): no splicing. No intron may open and the mates of a pair lie
        # at most this far apart; no GTF, so no junctions are inserted.
        args += unspliced_arguments(unspliced_mates_gap)
    elif gtf is not None and not index_already_has_junctions(reference):
        args += ["--sjdbGTFfile", str(gtf)]
    if extra_args.strip():
        args += extra_args.strip().split()
    return args


def _star_command(
    sample: FastqSample,
    out_prefix: Path,
    reference: Path,
    gtf: Optional[Path],
    threads: int,
    extra_args: str,
    read_files_command: str = "zcat",
    out_reads_unmapped: str = "None",
    out_filter_multimap_nmax: int = 2,
    out_filter_mismatch_nover_read_lmax: float = 0.04,
    out_filter_mismatch_nmax: int = MISMATCH_NMAX_OFF,
    out_filter_score_min_over_lread: float = 0.66,
    out_filter_match_nmin_over_lread: float = 0.66,
    # Recorded, not inferred. The four numbers above fully
    # determine a loose run today, but deriving it that way would break
    # silently the first time a default moves -- and 'was this run
    # loose' is exactly the question someone asks months later.
    loose: bool = False,
    sort_ram_bytes: Optional[int] = None,
    out_tmp_dir: Optional[Union[str, Path]] = None,
    sorted_output: bool = True,
    unspliced_mates_gap: Optional[int] = None,
) -> list[str]:
    """One sample's STAR command, in the order the array script writes it."""
    cmd = [
        # The recorded path, not the bare name. On one cluster a site
        # bin directory held a newer STAR than /usr/bin, and which one an account got
        # depended on the order of two directories in its PATH.
        resolve_tool("STAR") or "STAR",
        "--runThreadN", str(threads),
        "--genomeDir", str(reference),
        "--readFilesIn", sample.fastq_r1,
    ]
    if sample.fastq_r2:
        cmd.append(sample.fastq_r2)
    cmd += ["--outFileNamePrefix", str(out_prefix)]
    # STAR's working directory, on this node's scratch when the site
    # has one. Without it STAR writes _STARtmp beside the output -- shared
    # storage on a cluster, and on a login node just as often. Both backends
    # pass it, in the same position, because the two command lines are checked
    # against each other.
    if out_tmp_dir is not None:
        cmd += ["--outTmpDir", str(out_tmp_dir)]
    cmd += [
        "--outSAMattrRGline",
        f"ID:{sample.sample_id}",
        f"SM:{sample.sample_id}",
        "PL:ILLUMINA",
        f"LB:{sample.sample_id}",
        f"PU:{sample.sample_id}",
    ]
    if sort_ram_bytes is not None:
        cmd += ["--limitBAMsortRAM", str(int(sort_ram_bytes))]
    cmd += _star_result_arguments(
        reference=reference,
        gtf=gtf,
        extra_args=extra_args,
        read_files_command=read_files_command,
        out_reads_unmapped=out_reads_unmapped,
        out_filter_multimap_nmax=out_filter_multimap_nmax,
        out_filter_mismatch_nover_read_lmax=out_filter_mismatch_nover_read_lmax,
        out_filter_mismatch_nmax=out_filter_mismatch_nmax,
        out_filter_score_min_over_lread=out_filter_score_min_over_lread,
        out_filter_match_nmin_over_lread=out_filter_match_nmin_over_lread,
        sorted_output=sorted_output,
        unspliced_mates_gap=unspliced_mates_gap,
    )
    return cmd


def _sample_request(sample: FastqSample, *, reference: Path, arguments: list[str], loose: bool) -> dict[str, Any]:
    """Everything that decides a sample's BAM: genome, STAR arguments, FASTQs.

    Threads, memory, jobs and partition are left out: they do not
    change the alignment.
    """
    return {
        "reference": os.path.abspath(str(reference)),
        "star_arguments": list(arguments),
        "loose": bool(loose),
        "inputs": {
            "fastq_r1": input_signature(sample.fastq_r1),
            "fastq_r2": input_signature(sample.fastq_r2),
        },
    }


def _is_unspliced(request: dict[str, Any]) -> bool:
    arguments = request.get("star_arguments")
    return isinstance(arguments, list) and "--alignIntronMax" in arguments


def _refuse_mixed_alignment(workdir: Path, samples: list[FastqSample], requests: dict[str, dict[str, Any]]) -> None:
    """Refuse to resume when a finished sample was aligned under another request.

    A BAM with no record was made by an older Paralome and is accepted.
    """
    existing = load_stage_checkpoint(workdir, "alignment")
    if existing is None:
        return
    changed: dict[str, list[str]] = {}
    for sample in samples:
        previous = existing.samples.get(sample.sample_id)
        made_with = previous.metadata.get("made_with") if previous is not None else None
        if not isinstance(made_with, dict) or not _sample_done(workdir, sample, existing):
            continue
        request = requests[sample.sample_id]
        if _is_unspliced(request):
            # A BAM aligned before STAR ran in DNA mode is kept: only a new alignment gets DNA mode.
            made_with = {**made_with, "star_arguments": _without_dna_mode(made_with.get("star_arguments"))}
            request = {**request, "star_arguments": _without_dna_mode(request.get("star_arguments"))}
        # Nor is one aligned before the absolute mismatch cap was switched off and placement made random (_SETTLED_KEYS).
        made_with = {**made_with, "star_arguments": _without_dna_mode(made_with.get("star_arguments"), _SETTLED_KEYS)}
        request = {**request, "star_arguments": _without_dna_mode(request.get("star_arguments"), _SETTLED_KEYS)}
        differences = request_differences(made_with, request)
        if differences:
            changed[sample.sample_id] = differences
    refuse_mixed_results("STAR alignment", workdir, changed)


def _refuse_rejected_bams(workdir: Path, samples: list[FastqSample]) -> None:
    """Refuse to go on over a BAM that `align verify` read and rejected.

    Such a BAM is present and indexed, so it would be taken as
    finished again here and rejected again there, for ever. It is not deleted
    for the user: it is their output, and they are told which file to remove.

    The BAM is looked at again first, and only one that still fails is
    refused, so a passing fault (a stale file handle on one node) does not
    condemn it for good. A BAM that cannot be checked here is not refused:
    nothing is known against it.
    """
    existing = load_stage_checkpoint(workdir, "alignment")
    if existing is None:
        return
    from paralome.align.verify import _sample_status, no_reads_advice
    from paralome.errors import ParalomeError

    broken: dict[str, str] = {}
    no_reads: dict[str, str] = {}
    for sample in samples:
        previous = existing.samples.get(sample.sample_id)
        if previous is None or not str(previous.failed or "").startswith(VERIFY_REJECTED):
            continue
        if not _sample_done(workdir, sample, existing):
            continue
        bam, bai = _bam_paths(workdir, sample)
        status = _sample_status(sample.sample_id, bam, bai)
        if not status.rejected:
            continue
        line = f"{sample.sample_id}: {status.problem}: {bam}"
        if status.mapped_reads == 0:
            no_reads[sample.sample_id] = line
        else:
            broken[sample.sample_id] = line
    if not broken and not no_reads:
        return
    advice = []
    if broken:
        advice.append(
            "Remove the BAM and .bai of " + ", ".join(broken) + f" (or remove {workdir}) and run again."
        )
    if no_reads:
        # Aligning such a sample again gives the same empty BAM.
        advice.append(no_reads_advice(list(no_reads)))
    fix = "\n".join(advice)
    raise ParalomeError(
        312,
        f"STAR alignment in {workdir} holds BAMs that failed verification. Nothing was started.\n"
        + "\n".join([*broken.values(), *no_reads.values()]) + "\n" + fix,
        fix=fix,
    )


def _raise_open_file_limit() -> None:
    """Give STAR the open-file limit the array script asks for (`ulimit -n`).

    Sorting a BAM opens a file per bin and thread, and the local
    backend ran STAR under whatever limit the shell had.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65535 if hard == resource.RLIM_INFINITY else min(65535, hard)
        if soft != resource.RLIM_INFINITY and soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, ValueError, OSError):
        pass


def run_star_local(
    *,
    samples_csv: Union[str, Path],
    out: Union[str, Path],
    reference: Union[str, Path],
    gtf: str | Optional[Path] = None,
    threads: int = 8,
    overwrite: bool = False,
    mem: str = "32G",
    star_extra: str = "",
    read_files_command: str = "zcat",
    out_reads_unmapped: str = "None",
    out_filter_multimap_nmax: int = 2,
    out_filter_mismatch_nover_read_lmax: float = 0.04,
    out_filter_mismatch_nmax: int = MISMATCH_NMAX_OFF,
    out_filter_score_min_over_lread: float = 0.66,
    out_filter_match_nmin_over_lread: float = 0.66,
    # Recorded, not inferred. The four numbers above fully
    # determine a loose run today, but deriving it that way would break
    # silently the first time a default moves -- and 'was this run
    # loose' is exactly the question someone asks months later.
    loose: bool = False,
    # DNA and ATAC: STAR in DNA mode (UNSPLICED_MATES_GAP); no GTF is given to STAR.
    unspliced_mates_gap: Optional[int] = None,
) -> dict[str, Any]:
    _require_tool("STAR")
    _require_tool("samtools")
    workdir = Path(out)
    reference = Path(reference)
    gtf_path = Path(gtf) if gtf and not unspliced_mates_gap else None
    samples = read_fastq_manifest(samples_csv)
    errors = validate_fastqs(samples)
    if errors:
        raise FileNotFoundError("FASTQ validation failed:\n" + "\n".join(errors))

    # The same thread cap and BAM sort memory as an array task.
    # Samples run one after another here, so each gets the whole budget.
    sample_plan = plan_sample_parallel(
        total_threads=int(threads),
        sample_count=1,
        max_threads_per_sample=MAX_STAR_THREADS_PER_SAMPLE,
        jobs=1,
    )
    star_threads = sample_plan.threads_per_sample
    star_memory = plan_star_memory(sample_plan=sample_plan, partition=None, fallback_mem=mem)
    result_arguments = _star_result_arguments(
        reference=reference,
        gtf=gtf_path,
        extra_args=star_extra,
        read_files_command=read_files_command,
        out_reads_unmapped=out_reads_unmapped,
        out_filter_multimap_nmax=out_filter_multimap_nmax,
        out_filter_mismatch_nover_read_lmax=out_filter_mismatch_nover_read_lmax,
        out_filter_mismatch_nmax=out_filter_mismatch_nmax,
        out_filter_score_min_over_lread=out_filter_score_min_over_lread,
        out_filter_match_nmin_over_lread=out_filter_match_nmin_over_lread,
        unspliced_mates_gap=unspliced_mates_gap,
    )
    requests = {
        s.sample_id: _sample_request(s, reference=reference, arguments=result_arguments, loose=loose)
        for s in samples
    }
    if not overwrite:
        # The rejected first: a sample whose FASTQ was corrected because its
        # BAM was rejected is that case, not a changed request.
        _refuse_rejected_bams(workdir, samples)
        _refuse_mixed_alignment(workdir, samples, requests)
    earlier = load_stage_checkpoint(workdir, "alignment")
    earlier_job_ids = [str(x) for x in (earlier.resources.get("submitted_job_ids") or [])] if earlier else []
    if overwrite or any(not _sample_done(workdir, s, earlier) for s in samples):
        # As the SLURM backend asks: run here beside an array that is still going, this would align the
        # same samples into the same files and rewrite the checkpoint the tasks are writing to.
        refuse_while_jobs_are_alive(
            "STAR alignment", workdir, earlier_job_ids,
            recorded_in='the entry "submitted_job_ids" (under "resources") from '
                        f'{stage_checkpoint_path(workdir, "alignment")}',
        )
        earlier_job_ids = []

    for d in ["inputs", "bam", "logs", "results", "checkpoints", "scripts", "temp"]:
        (workdir / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(samples_csv, workdir / "inputs" / "original_samples.csv")

    completed = []
    bam_rows: list[dict[str, str]] = []
    if overwrite:
        # The old BAMs are not removed here. Every sample is marked in the checkpoint as to be replaced
        # (_alignment_checkpoint) and loses its old BAM when its own turn comes, as on SLURM.
        _invalidate_previous_verification(workdir)
    aligned_manifest = workdir / "inputs" / "aligned_samples.csv"
    # The stage checkpoint the SLURM backend keeps. This path wrote
    # only checkpoints/pipeline.json, which nothing downstream reads, so a
    # local alignment was never seen as finished.
    ckpt = _alignment_checkpoint(
        workdir=workdir,
        samples=samples,
        reference=reference,
        gtf=gtf_path,
        threads=star_threads,
        jobs=1,
        backend="local",
        aligned_manifest=aligned_manifest,
        mem=star_memory.mem,
        overwrite=overwrite,
        requests=requests,
        submitted_job_ids=earlier_job_ids,
    )
    save_stage_checkpoint(workdir, ckpt)
    to_replace = {sid for sid, entry in ckpt.samples.items() if entry.metadata.get(TO_BE_REPLACED)}
    _raise_open_file_limit()
    for sample in samples:
        prefix = workdir / "temp" / f"{sample.sample_id}."
        final_bam = workdir / "bam" / f"{sample.sample_id}.sorted.bam"
        # The index too, as the SLURM task and `align verify` both
        # require. A run stopped between moving the BAM into place and
        # indexing it was skipped here for ever while verification kept
        # reporting the sample incomplete.
        if _bam_complete(workdir, sample) and sample.sample_id not in to_replace:
            completed.append(sample.sample_id)
            bam_rows.append(sample.to_bam_row(final_bam))
            continue
        _discard_previous_alignment(workdir, sample)
        try:
            star_need = (os.path.getsize(sample.fastq_r1)
                         + (os.path.getsize(sample.fastq_r2) if sample.fastq_r2 else 0)) * 3
        except OSError:
            star_need = 0
        star_scratch_stack = contextlib.ExitStack()
        star_scratch = star_scratch_stack.enter_context(
            job_scratch_dir(Path(prefix).parent, "star", need_bytes=star_need))
        # STAR refuses a --outTmpDir that already exists, so it is named inside
        # the scratch directory rather than being it.
        star_tmp = star_scratch / f"_STARtmp_{sample.sample_id}"
        cmd = _star_command(
            sample,
            prefix,
            reference,
            gtf_path,
            star_threads,
            star_extra,
            read_files_command=read_files_command,
            out_reads_unmapped=out_reads_unmapped,
            out_filter_multimap_nmax=out_filter_multimap_nmax,
            out_filter_mismatch_nover_read_lmax=out_filter_mismatch_nover_read_lmax,
            out_filter_mismatch_nmax=out_filter_mismatch_nmax,
            out_filter_score_min_over_lread=out_filter_score_min_over_lread,
            out_filter_match_nmin_over_lread=out_filter_match_nmin_over_lread,
            loose=loose,
            sort_ram_bytes=star_memory.sort_ram_bytes,
            out_tmp_dir=star_tmp,
            unspliced_mates_gap=unspliced_mates_gap,
        )
        log = workdir / "logs" / f"{sample.sample_id}.STAR.log"
        with star_scratch_stack:
            shutil.rmtree(star_tmp, ignore_errors=True)
            with log.open("w") as out_fh:
                subprocess.run(cmd, check=True, stdout=out_fh, stderr=subprocess.STDOUT, text=True)
        star_bam = Path(str(prefix) + "Aligned.sortedByCoord.out.bam")
        if not star_bam.exists():
            raise FileNotFoundError(f"STAR did not create expected BAM: {star_bam}")
        shutil.move(str(star_bam), final_bam)
        index_threads = min(4, max(1, int(star_threads)))
        subprocess.run(
            [resolve_tool("samtools") or "samtools", "index",
             "-@", str(index_threads), str(final_bam)],
            check=True,
        )
        update_stage_checkpoint(
            workdir, "alignment",
            lambda stage, sample_id=sample.sample_id, bam=final_bam: _mark_aligned(stage, sample_id, bam),
        )
        completed.append(sample.sample_id)
        bam_rows.append(sample.to_bam_row(final_bam))

    write_csv(
        aligned_manifest,
        bam_rows,
        [
            "sample_id", "family_id", "relationship", "affected", "sex",
            "condition", "group", "mother_id", "father_id", "bam",
        ],
    )
    summary = {
        "workdir": str(workdir),
        "samples": len(samples),
        "completed_samples": len(completed),
        "aligned_manifest": str(aligned_manifest),
        "completed": completed,
        "reference": str(reference),
        "parameters": {
            "read_files_command": read_files_command,
            "out_reads_unmapped": out_reads_unmapped,
            "out_filter_multimap_nmax": out_filter_multimap_nmax,
            "out_filter_mismatch_nover_read_lmax": out_filter_mismatch_nover_read_lmax,
            "out_filter_mismatch_nmax": out_filter_mismatch_nmax,
            "out_filter_score_min_over_lread": out_filter_score_min_over_lread,
            "out_filter_match_nmin_over_lread": out_filter_match_nmin_over_lread,
            "loose": bool(loose),
            "star_extra": star_extra,
            # Recorded so the run report can show the numbers a run produces beside the settings
            # that produced them.
            "sjdbGTFfile_passed": bool(gtf_path) and not index_already_has_junctions(reference),
        },
        "gtf": str(gtf_path) if gtf_path else None,
        "threads_per_sample": star_threads,
        "samtools_index_threads": min(4, max(1, int(star_threads))),
        "star_sort_ram_bytes": star_memory.sort_ram_bytes,
        "checkpoint": str(stage_checkpoint_path(workdir, "alignment")),
        "finished": now_iso(),
    }
    (workdir / "results" / "star_alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_checkpoint(workdir, {"alignment": {"star": {"complete": len(completed) == len(samples), **summary}}})
    return summary



def _bam_paths(workdir: Path, sample: FastqSample) -> tuple[Path, Path]:
    bam = workdir / "bam" / f"{sample.sample_id}.sorted.bam"
    return bam, Path(str(bam) + ".bai")


def _bam_complete(workdir: Path, sample: FastqSample) -> bool:
    bam, bai = _bam_paths(workdir, sample)
    return bam.exists() and bam.stat().st_size > 0 and bai.exists() and bai.stat().st_size > 0


def _sample_done(workdir: Path, sample: FastqSample, existing: Optional[StageCheckpoint]) -> bool:
    """BAM and index in place, and not waiting to be replaced under --overwrite.

    The one test of "this sample is finished" for planning on both
    backends. See _alignment_checkpoint for the mark.
    """
    previous = existing.samples.get(sample.sample_id) if existing is not None else None
    if previous is not None and previous.metadata.get(TO_BE_REPLACED):
        return False
    return _bam_complete(workdir, sample)


def _mark_aligned(stage: StageCheckpoint, sample_id: str, bam: Path) -> None:
    """A sample has just been aligned: complete, and no longer to be replaced."""
    stage.mark_complete(sample_id, outputs={"bam": str(bam), "bai": str(bam) + ".bai"})
    stage.samples[sample_id].metadata.pop(TO_BE_REPLACED, None)


def _discard_previous_alignment(workdir: Path, sample: FastqSample) -> None:
    """Remove a sample's BAM and index before it is aligned again.

    STAR writes under temp/ and the BAM is moved into place at the
    end, so a re-alignment that fails leaves whatever was there before. What
    was there before is exactly what the user asked to replace; left in place
    it is taken for the new result by the next run.
    """
    for path in _bam_paths(workdir, sample):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _invalidate_previous_verification(workdir: Path) -> None:
    """Withdraw an earlier run's "alignment verified" before re-aligning.

    Results/star_alignment_verify.json is what the pipeline reads to
    decide the stage is complete, and under --overwrite the one on disk
    describes the BAMs about to be replaced.
    """
    path = workdir / "results" / "star_alignment_verify.json"
    if not path.is_file():
        return
    try:
        previous = json.loads(path.read_text())
    except ValueError:
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    # Whatever the report says, it describes BAMs that are being replaced, so it is withdrawn;
    # could_not_check goes with it, as a finding about the old BAMs.
    previous.pop("could_not_check", None)
    previous["complete"] = False
    previous["invalidated"] = "--overwrite: the alignments are being replaced"
    path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")


def _alignment_checkpoint(
    *,
    workdir: Path,
    samples: list[FastqSample],
    reference: Path,
    gtf: Optional[Path],
    threads: int,
    jobs: int,
    backend: str,
    aligned_manifest: Path,
    script: Optional[Path] = None,
    mem: Optional[str] = None,
    overwrite: bool = False,
    requests: Optional[dict[str, dict[str, Any]]] = None,
    submitted_job_ids: Optional[list[str]] = None,
) -> StageCheckpoint:
    existing = load_stage_checkpoint(workdir, "alignment")
    resources = {
        "backend": backend,
        "threads_per_sample": threads,
        "jobs": jobs,
        "peak_cpus": threads * jobs if jobs else threads,
        "reference": str(reference),
        "gtf": str(gtf) if gtf else None,
    }
    if script is not None:
        resources["script"] = str(script)
    if mem is not None:
        resources["mem"] = mem
    if submitted_job_ids is not None:
        resources["submitted_job_ids"] = [str(job_id) for job_id in submitted_job_ids]
    sample_ckpts: list[SampleCheckpoint] = []
    for idx, sample in enumerate(samples, start=1):
        bam, bai = _bam_paths(workdir, sample)
        previous = existing.samples.get(sample.sample_id) if existing else None
        # TO BE REPLACED. --overwrite marks every sample, here, when the stage is planned, and deletes
        # nothing, so a refused submission leaves the BAMs in place. A marked sample is not finished
        # whatever is on disk; its task removes the old BAM when it starts and clears the mark when it
        # has finished. A later run without --overwrite finds the marks still set on the samples not
        # yet replaced and runs exactly those.
        marked = overwrite or bool(previous is not None and previous.metadata.get(TO_BE_REPLACED))
        done = _bam_complete(workdir, sample) and not marked
        ckpt = (None if overwrite else previous) or SampleCheckpoint(sample_id=sample.sample_id, index=idx)
        ckpt.index = idx
        ckpt.outputs.update({"bam": str(bam), "bai": str(bai)})
        ckpt.metadata.update({"group": sample.group, "relationship": sample.relationship})
        # Which FASTQs and which request a BAM was made with stay as they were recorded while that
        # BAM is on disk, so a finished sample is never recorded under a FASTQ path or options it was
        # not aligned with. A sample to be replaced is recorded under the current request even
        # though its old BAM is on disk: that BAM is not what will stand.
        if previous is not None and done:
            for key in ("fastq_r1", "fastq_r2", "made_with"):
                if key in previous.metadata:
                    ckpt.metadata[key] = previous.metadata[key]
        else:
            ckpt.metadata.update({"fastq_r1": sample.fastq_r1, "fastq_r2": sample.fastq_r2})
            ckpt.metadata.pop("made_with", None)
            if requests is not None and not done:
                ckpt.metadata["made_with"] = requests[sample.sample_id]
        if marked:
            ckpt.metadata[TO_BE_REPLACED] = True
        # COMPLETE MEANS THE BAM IS THERE AND IS NOT BEING REPLACED. Under --overwrite a sample is
        # not complete until it has been aligned again, so a task that dies cannot leave the old BAM
        # standing as a finished sample under the new settings; and a sample whose BAM was deleted
        # is not complete, whatever was recorded before.
        if not done:
            ckpt.complete = False
            ckpt.failed = None
            ckpt.finished = None
        else:
            ckpt.complete = True
            ckpt.failed = None
            ckpt.finished = ckpt.finished or now_iso()
        sample_ckpts.append(ckpt)

    return StageCheckpoint.create(
        stage="alignment",
        workdir=workdir,
        samples=sample_ckpts,
        resources=resources,
        outputs={"aligned_manifest": str(aligned_manifest)},
    )

def _record_submitted_jobs(workdir: Path, job_ids: list[str]) -> None:
    """Keep the stage's queued job IDs with its checkpoint.

    So that `align star --submit` run again by hand can see that an
    earlier submission is still running. Under the checkpoint's lock, because
    array tasks write to the same file as they finish.
    """
    def record(stage: StageCheckpoint) -> None:
        stage.resources["submitted_job_ids"] = [str(job_id) for job_id in job_ids]

    update_stage_checkpoint(workdir, "alignment", record)


def write_star_slurm(
    *,
    samples_csv: Union[str, Path],
    out: Union[str, Path],
    reference: Union[str, Path],
    gtf: str | Optional[Path] = None,
    threads: int = 8,
    jobs: int = 0,
    submit: bool = False,
    overwrite: bool = False,
    partition: Optional[str] = None,
    # None is the default. A task that marks duplicates itself
    # asks for its index, STAR's working memory and the samtools chain added
    # up (resources.plan_fused_alignment_memory); any other task asks for 32G
    # as before. A value given here replaces the default either way.
    mem: Optional[str] = None,
    star_extra: str = "",
    read_files_command: str = "zcat",
    out_reads_unmapped: str = "None",
    out_filter_multimap_nmax: int = 2,
    out_filter_mismatch_nover_read_lmax: float = 0.04,
    out_filter_mismatch_nmax: int = MISMATCH_NMAX_OFF,
    out_filter_score_min_over_lread: float = 0.66,
    out_filter_match_nmin_over_lread: float = 0.66,
    # Recorded, not inferred. The four numbers above fully
    # determine a loose run today, but deriving it that way would break
    # silently the first time a default moves -- and 'was this run
    # loose' is exactly the question someone asks months later.
    loose: bool = False,
    # The two fusions. `trim_in_task` makes this task run fastp
    # itself, reading the RAW FASTQs named in the manifest and streaming the
    # trimmed reads to STAR through FIFOs, so the trimmed FASTQs are never
    # written. `fuse_dedup` sends STAR's unsorted output straight into
    # duplicate marking, so the aligned-but-not-deduplicated BAM is never
    # written either, and STAR's coordinate sort is not performed at all.
    # Both are set by the orchestrator; a caller that passes neither gets STAR alone.
    trim_in_task: Optional[dict[str, Any]] = None,
    fuse_dedup: Optional[dict[str, Any]] = None,
    # Duplicate marking inside this task for an assay with no
    # deduplication stage (RNA). The marks go into the alignment's own
    # bam/<sample>.sorted.bam -- the path and name quantification already
    # reads -- so nothing downstream knows anything changed. markdup MARKS
    # rather than removes, so the file is a strict superset of the plain
    # aligned BAM and counting it unfiltered gives the same numbers; RNA
    # should not filter duplicates without UMIs, because a highly expressed
    # gene legitimately produces identical fragments.
    mark_in_place: bool = False,
    fused_buffer_bytes: int = FUSED_BUFFER_BYTES,
    # The same tasks run on this machine by run_star_fused_local;
    # the checkpoint and summary then say local, which is where they ran.
    record_backend: str = "slurm",
    # --aligner: star (default) or bwa. With bwa, ``reference`` is the genome's FASTA (align.bwa).
    aligner: str = "star",
    bwa_extra: str = "",
    # DNA and ATAC: STAR in DNA mode (UNSPLICED_MATES_GAP); no GTF is given to STAR.
    unspliced_mates_gap: Optional[int] = None,
    # With --aligner bwa, the pair filter's thresholds (align.bwa.filter_settings).
    bwa_filter: Optional[dict] = None,
) -> dict[str, Any]:
    from paralome.align import bwa as bwa_align

    aligner = bwa_align.normalize_aligner(aligner)
    use_bwa = aligner == "bwa"
    workdir = Path(out)
    reference = Path(reference)
    if use_bwa:
        reference = bwa_align.require_fasta(reference)
    # A FIFO carries uncompressed FASTQ, and STAR reads it through
    # Paralome's own buffer rather than `cat`. `cat` adds no capacity, so the
    # whole channel was two 64 KB kernel pipes -- and STAR consumes mates in
    # lockstep while fastp's two writers run ahead of each other, so any skew
    # past 64 KB deadlocked the pair. See paralome.buffer. Recorded with every
    # sample, so a resumed run can tell one arrangement from the other.
    if trim_in_task:
        read_files_command = fused_read_files_command(buffer_bytes=fused_buffer_bytes)
    mark_duplicates_here = bool(fuse_dedup) or bool(mark_in_place)
    fused_sort = not mark_duplicates_here
    gtf_path = Path(gtf) if gtf and not unspliced_mates_gap and not use_bwa else None
    samples = read_fastq_manifest(samples_csv)
    errors = validate_fastqs(samples)
    if errors:
        raise FileNotFoundError("FASTQ validation failed:\n" + "\n".join(errors))
    for s in samples:
        for label, value in (("sample_id", s.sample_id), ("fastq_r1", s.fastq_r1), ("fastq_r2", s.fastq_r2)):
            if "\t" in value or "\n" in value:
                raise ValueError(
                    f"Sample {s.sample_id!r}: {label} contains a tab or a line break, "
                    "which the array task cannot read; remove it from the sample sheet"
                )

    requested_threads = int(threads)
    sample_count = len(samples)
    sample_plan = plan_sample_parallel(
        total_threads=requested_threads,
        sample_count=sample_count,
        max_threads_per_sample=MAX_STAR_THREADS_PER_SAMPLE,
        jobs=jobs,
    )
    capabilities = detect_scheduler_capabilities("slurm")
    array_plan = plan_array_tasks(
        task_count=sample_count,
        concurrency=sample_plan.concurrent_jobs,
        capabilities=capabilities,
    )
    if array_plan.batch_count > 1:
        raise ValueError(
            "STAR sample count exceeds the scheduler MaxArraySize; "
            "split the manifest or raise PARALOME_SLURM_MAX_ARRAY_SIZE"
        )
    star_memory = plan_star_memory(
        sample_plan=sample_plan,
        partition=partition,
        fallback_mem=mem,
    )
    requested_throttle = sample_plan.concurrent_jobs
    star_threads = sample_plan.threads_per_sample
    star_mem = star_memory.mem
    star_sort_ram = star_memory.sort_ram_bytes
    buffer_overhead = (
        # Two buffers for paired reads, one for single-end. See
        # stream.fused_buffer_overhead for why this is not optional.
        fused_buffer_overhead(fused_buffer_bytes, paired=any(sample.fastq_r2 for sample in samples))
        if trim_in_task else 0
    )
    fused_memory = None
    if mem is None and mark_duplicates_here:
        # The samtools chain runs in this task beside
        # STAR, so the request is the two added up, from this genome's index.
        # STAR does not sort here, so its sort limit plays no part.
        fused_memory = plan_fused_alignment_memory(
            reference=reference,
            threads=star_threads,
            buffer_bytes=buffer_overhead,
            fastp_in_task=bool(trim_in_task),
            aligner_index=bwa_align.memory_index(reference) if use_bwa else None,
        )
        star_mem = fused_memory.mem
        memory_basis = "calculated: fused alignment and duplicate marking"
    else:
        if buffer_overhead:
            star_mem = mem_plus_bytes(star_mem, buffer_overhead)
        memory_basis = "given (--mem)" if mem is not None else "default 32G"

    if use_bwa:
        result_arguments = bwa_align.result_arguments(extra_args=bwa_extra, sorted_output=fused_sort,
                                                      pair_filter=bwa_filter)
    else:
        result_arguments = _star_result_arguments(
            reference=reference,
            gtf=gtf_path,
            extra_args=star_extra,
            read_files_command=read_files_command,
            out_reads_unmapped=out_reads_unmapped,
            out_filter_multimap_nmax=out_filter_multimap_nmax,
            out_filter_mismatch_nover_read_lmax=out_filter_mismatch_nover_read_lmax,
            out_filter_mismatch_nmax=out_filter_mismatch_nmax,
            out_filter_score_min_over_lread=out_filter_score_min_over_lread,
            out_filter_match_nmin_over_lread=out_filter_match_nmin_over_lread,
            sorted_output=fused_sort,
            unspliced_mates_gap=unspliced_mates_gap,
        )
    requests = {
        s.sample_id: _sample_request(s, reference=reference, arguments=result_arguments, loose=loose)
        for s in samples
    }
    if use_bwa:
        # Recorded only for BWA, so a STAR request reads as it always did and a project
        # switched between the two aligners is refused as a mixed alignment (--overwrite to redo).
        for request in requests.values():
            request["aligner"] = "bwa"
    if not overwrite:
        _refuse_rejected_bams(workdir, samples)
        _refuse_mixed_alignment(workdir, samples, requests)
    earlier = load_stage_checkpoint(workdir, "alignment")
    earlier_job_ids = [str(x) for x in (earlier.resources.get("submitted_job_ids") or [])] if earlier else []
    will_run = overwrite or any(not _sample_done(workdir, s, earlier) for s in samples)
    if will_run:
        # Also when only the scripts are being written: they and the checkpoint
        # belong to the tasks still running.
        refuse_while_jobs_are_alive(
            "STAR alignment", workdir, earlier_job_ids,
            recorded_in='the entry "submitted_job_ids" (under "resources") from '
                        f'{stage_checkpoint_path(workdir, "alignment")}',
        )
        # The queue answered and they are gone, so they are not asked about again (a squeue outage
        # would otherwise stop the stage over jobs that ended long before).
        earlier_job_ids = []
        if submit:
            _require_tool("sbatch")

    # Nothing is written until the stage is certain to go ahead. The
    # manifest of aligned samples was written before the MaxArraySize refusal
    # above, and by itself it reads as a finished alignment.
    for d in ["inputs", "bam", "logs", "results", "checkpoints", "scripts", "temp"]:
        (workdir / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(samples_csv, workdir / "inputs" / "original_samples.csv")

    manifest = workdir / "inputs" / "star_manifest.tsv"

    aligned_manifest = workdir / "inputs" / "aligned_samples.csv"
    bam_rows = [s.to_bam_row(workdir / "bam" / f"{s.sample_id}.sorted.bam") for s in samples]
    write_csv(
        aligned_manifest,
        bam_rows,
        [
            "sample_id", "family_id", "relationship", "affected", "sex",
            "condition", "group", "mother_id", "father_id", "bam",
        ],
    )

    script = workdir / "scripts" / "star_array.sbatch"
    ckpt = _alignment_checkpoint(
        workdir=workdir,
        samples=samples,
        reference=reference,
        gtf=gtf_path,
        threads=star_threads,
        jobs=requested_throttle,
        backend=record_backend,
        aligned_manifest=aligned_manifest,
        script=script,
        mem=star_mem,
        overwrite=overwrite,
        requests=requests,
        submitted_job_ids=earlier_job_ids,
    )
    save_stage_checkpoint(workdir, ckpt)
    if overwrite:
        _invalidate_previous_verification(workdir)

    # Tab-separated, and only what the task reads. It was written
    # with CSV quoting and read back in bash by splitting on commas, so a comma
    # in a group or relationship cell shifted the FASTQ columns. The last
    # column is the sample's "to be replaced" mark: with 1 the task does not
    # take the BAM in place as finished.
    with manifest.open("w", newline="") as fh:
        fh.write("index\tsample_id\tfastq_r1\tfastq_r2\treplace\n")
        for i, s in enumerate(samples, start=1):
            replace = "1" if ckpt.samples[s.sample_id].metadata.get(TO_BE_REPLACED) else "0"
            fh.write(f"{i}\t{s.sample_id}\t{s.fastq_r1}\t{s.fastq_r2 or '.'}\t{replace}\n")

    active_indices = ckpt.missing_indices()
    array_spec = slurm_array_spec(active_indices, requested_throttle)

    partition_line = f"#SBATCH --partition={shlex.quote(str(partition))}\n" if partition else ""
    # The SLURM path makes the same decision as the local one.
    #
    # The affected run went through here, and this is where the GTF was handed to
    # STAR for an index that already carried its junctions. Deciding it in
    # Python rather than in the generated bash means the emitted script SAYS
    # what was decided -- the sbatch is the artefact anyone reads afterwards
    # when asking why a run behaved as it did.
    junctions_are_baked_in = (not use_bwa) and bool(gtf_path) and index_already_has_junctions(reference)
    junction_note = (
        "# --sjdbGTFfile withheld: this index was generated with the GTF, so\n"
        "# its junctions are already loaded. Passing it again makes STAR insert\n"
        "# them a second time on the fly, and junction-spanning reads stop\n"
        "# aligning (they are reported as 'unmapped: too short').\n"
        if junctions_are_baked_in else
        "# --sjdbGTFfile passed: this index carries no annotation of its own.\n"
        if gtf_path and not use_bwa else ""
    )
    # The space guard is given three times the read files, which is
    # about what STAR's per-thread chunks and unmapped output come to. The
    # fallback is the sample's own temp folder, where _STARtmp went before.
    # A fused task keeps more than STAR's chunks here -- the
    # duplicate-marking sorts spill into the same directory, and the FIFOs
    # themselves hold no data but the sorts certainly do. Three times the read
    # files covered STAR alone; eight covers the sorts beside it.
    scratch_factor = 8 if (trim_in_task or mark_duplicates_here) else 3
    star_scratch_bash = job_scratch_bash(
        '"$(dirname "$PREFIX")"', "star", var="STAR_SCRATCH",
        need_bytes=f'$(( ( $(stat -c %s "$R1" 2>/dev/null || echo 0) '
                   f'+ $(stat -c %s "$R2" 2>/dev/null || echo 0) ) * {scratch_factor} ))',
    )
    # And SLURM is asked for the largest task's share of that space
    # (see paralome.scratch), so the scheduler counts it.
    def _reads_bytes(sample) -> int:
        total = 0
        for path in (sample.fastq_r1, sample.fastq_r2):
            if path:
                try:
                    total += os.path.getsize(path)
                except OSError:
                    pass
        return total

    tmp_line = tmp_directive(
        max((_reads_bytes(samples[i - 1]) for i in active_indices if 0 < i <= len(samples)), default=0)
        * scratch_factor,
        partition,
    )

    if active_indices:
        star_exe = shlex.quote(resolve_tool("STAR") or "STAR")
        bwa_exe = shlex.quote(resolve_tool(bwa_align.BWA_TOOL) or bwa_align.BWA_TOOL)
        aligner_exe = bwa_exe if use_bwa else star_exe
        samtools_exe = shlex.quote(resolve_tool("samtools") or "samtools")
        # bamUtil soft-clips the overlap of a pair's mates after duplicate marking.
        from paralome.stream import BAMUTIL_TOOL

        clip_overlap = bool((fuse_dedup or {}).get("clip_overlap")) and not mark_in_place
        bamutil_line = f"BAMUTIL={shlex.quote(resolve_tool(BAMUTIL_TOOL) or BAMUTIL_TOOL)}" if clip_overlap else ""
        # The interpreter that wrote this script, not whichever
        # python3 the job's PATH finds; that one may not have Paralome at all.
        python_exe = shlex.quote(sys.executable)
        # STAR runs the buffer once per read file, deep inside its own startup. Checked here
        # instead, so a task whose PATH has no paralome-buffer says so in its first second rather
        # than failing with a STAR error about an unreadable input.
        fused_reader_check = (
            f"require_tool {shlex.quote(FUSED_READER)}" if trim_in_task else ""
        )
        # The body of the task, composed here rather than written inline, because it has three
        # shapes: STAR alone, STAR fed by fastp through FIFOs, and either of those piped straight
        # into duplicate marking. Composing it in Python means the generated sbatch shows exactly
        # one of them, with nothing switched at run time.
        if mark_in_place:
            # No deduplication stage to own it, so the marked BAM
            # keeps the alignment's own path and name.
            bam_lines = "\n".join([
                'PREFIX="$WORKDIR/temp/${SAMPLE_ID}."',
                'BAM="$WORKDIR/bam/${SAMPLE_ID}.sorted.bam"',
                'mkdir -p "$WORKDIR/temp" "$WORKDIR/bam" "$WORKDIR/logs" "$WORKDIR/checkpoints"',
            ])
        elif fuse_dedup:
            # The one BAM this task keeps is the duplicate-marked one, and it
            # belongs to the duplicate marking stage: every reader downstream
            # looks for it there, so a fused run leaves the same tree behind as
            # an unfused one.
            dedup_bam_dir = shlex.quote(str(Path(fuse_dedup["workdir"]) / "bam"))
            bam_lines = "\n".join([
                'PREFIX="$WORKDIR/temp/${SAMPLE_ID}."',
                f'BAM={dedup_bam_dir}/"${{SAMPLE_ID}}".markdup.bam',
                f'mkdir -p "$WORKDIR/temp" "$WORKDIR/bam" "$WORKDIR/logs" "$WORKDIR/checkpoints" {dedup_bam_dir}',
            ])
        else:
            bam_lines = "\n".join([
                'PREFIX="$WORKDIR/temp/${SAMPLE_ID}."',
                'BAM="$WORKDIR/bam/${SAMPLE_ID}.sorted.bam"',
                'mkdir -p "$WORKDIR/temp" "$WORKDIR/bam" "$WORKDIR/logs" "$WORKDIR/checkpoints"',
            ])
        body: list[str] = [junction_note.rstrip("\n")] if junction_note else []
        body += [star_scratch_bash]
        body += [
            "",
            "# STAR's working directory on this node's scratch. Without",
            "# --outTmpDir it writes _STARtmp beside the output, which is shared",
            "# storage: per-thread chunk files written and read back once, then",
            "# deleted. The path must NOT exist when STAR starts -- STAR refuses a",
            "# directory that is already there -- so it is named inside the scratch",
            "# directory rather than being it.",
            "# Named for the sample. Where no scratch is configured the",
            "# fallback is the stage's temp/ folder, shared by every task, and tasks",
            "# running at once collided on one _STARtmp (\"could not make temporary",
            "# directory\"). A scratch directory is the job's own; this costs nothing there.",
            'STAR_TMP="$STAR_SCRATCH/_STARtmp_${SAMPLE_ID}"',
            'rm -rf "$STAR_TMP"',
        ]
        if mark_duplicates_here:
            body += [
                "",
                "# Duplicate marking runs here, on STAR's output as it",
                "# arrives. Its sorts spill into the same scratch directory.",
                'DEDUP_SCRATCH="$STAR_SCRATCH"',
                # Never -r when marking in place: the marked BAM is the only
                # copy, and RNA must keep its duplicates.
                f"REMOVE_FLAG={'-r' if (fuse_dedup or {}).get('remove_duplicates') and not mark_in_place else ''!r}",
                'PARTIAL_BAM="$STAR_SCRATCH/${SAMPLE_ID}.markdup.partial.bam"',
                'rm -f "$PARTIAL_BAM" "$PARTIAL_BAM.bai"',
            ]
        if trim_in_task:
            # The trimming arguments as bash arrays, each element quoted, the way the FASTQ QC stage
            # writes them, so a path with a space in it reaches fastp as one argument.
            trim_args_text = " ".join(shlex.quote(str(x)) for x in trim_in_task.get("args") or [])
            trim_paired_text = " ".join(shlex.quote(str(x)) for x in trim_in_task.get("paired_args") or [])
            body += ["", f"TRIM_ARGS=({trim_args_text})", f"TRIM_PAIRED_ARGS=({trim_paired_text})"]
            body += ["", fastp_fifo_bash(
                fastp_exe=trim_in_task["fastp_exe"],
                scratch_var="STAR_SCRATCH",
                json_dir=shlex.quote(str(trim_in_task["json_dir"])),
                threads=int(trim_in_task.get("threads") or 4),
            )]
            body += [
                'READS=("$TRIM_R1")',
                'if [[ -n "$TRIM_R2" ]]; then',
                '  READS+=("$TRIM_R2")',
                "fi",
            ]
        else:
            body += [
                "",
                'READS=("$R1")',
                'if [[ -n "$R2" && "$R2" != "." ]]; then',
                '  READS+=("$R2")',
                "fi",
            ]
        star_lines = [] if use_bwa else [
            "",
            "# The sample's own arguments first, then the list the local",
            "# backend uses, written out in full -- not as bash arrays, which may be",
            "# empty, and bash before 4.4 calls an empty array unbound under set -u.",
            f"{star_exe} \\",
            '  --runThreadN "$THREADS" \\',
            '  --genomeDir "$REFERENCE" \\',
            '  --readFilesIn "${READS[@]}" \\',
            '  --outFileNamePrefix "$PREFIX" \\',
            '  --outTmpDir "$STAR_TMP" \\',
            "  --outSAMattrRGline \\",
            '  ID:"$SAMPLE_ID" \\',
            '  SM:"$SAMPLE_ID" \\',
            "  PL:ILLUMINA \\",
            '  LB:"$SAMPLE_ID" \\',
            '  PU:"$SAMPLE_ID" \\',
        ]
        if use_bwa:
            chain = None
            if mark_duplicates_here:
                chain = [
                    line for line in markdup_stream_bash(samtools_var="SAMTOOLS", source="-", clip_var="BAMUTIL" if clip_overlap else "").splitlines()
                    if not line.startswith("# shellcheck")
                ]
                chain[0] = "  | " + chain[0]
            star_lines = bwa_align.align_lines(
                bwa_exe=bwa_exe, extra_args=bwa_extra,
                read_command=read_files_command if trim_in_task else None,
                fused_chain=chain, sorted_bam='"${PREFIX}Aligned.sortedByCoord.out.bam"',
                pair_filter=bwa_filter,
            )
        elif fused_sort:
            star_lines.append('  --limitBAMsortRAM "$STAR_SORT_RAM" \\')
        if use_bwa:
            pass
        elif mark_duplicates_here:
            # STAR's unsorted BAM goes down the pipe; the chain is the one
            # paralome.stream gives the duplicate marking stage, so the two
            # cannot drift apart.
            chain = [
                line for line in markdup_stream_bash(samtools_var="SAMTOOLS", source="-", clip_var="BAMUTIL" if clip_overlap else "").splitlines()
                if not line.startswith("# shellcheck")
            ]
            chain[0] = "  | " + chain[0]
            star_lines.append(f"  {shell_join(result_arguments)} \\")
            star_lines += chain
        else:
            star_lines.append(f"  {shell_join(result_arguments)}")
        body += star_lines
        if trim_in_task:
            body += ["", wait_for_fastp_bash()]
        body += [
            "",
            'INDEX_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            "if (( INDEX_THREADS > 4 )); then",
            "  INDEX_THREADS=4",
            "fi",
            "if (( INDEX_THREADS < 1 )); then",
            "  INDEX_THREADS=1",
            "fi",
            'echo "samtools index threads: $INDEX_THREADS"',
        ]
        if mark_duplicates_here:
            body += [
                # Indexed beside the final name and moved into place with the old index taken away
                # first, so a task killed part way cannot leave a truncated BAM that every
                # completion test accepts.
                '"$SAMTOOLS" index -@ "$INDEX_THREADS" "$PARTIAL_BAM"',
                'rm -f "$BAM.bai"',
                'mv -f "$PARTIAL_BAM" "$BAM"',
                'mv -f "$PARTIAL_BAM.bai" "$BAM.bai"',
                # The read and duplicate counts, taken here with the task's threads, so the stage that
                # adopts these BAMs reads them instead of counting every BAM again (dedup.markdup).
                f'"$SAMTOOLS" flagstat -@ "$INDEX_THREADS" -O json "$BAM" > "$BAM{FLAGSTAT_SIDECAR}"',
            ]
        else:
            body += [
                'STAR_BAM="${PREFIX}Aligned.sortedByCoord.out.bam"',
                'if [[ ! -s "$STAR_BAM" ]]; then',
                f'  echo "ERROR: {"bwa-mem2 and samtools sort" if use_bwa else "STAR"} did not create $STAR_BAM" >&2',
                "  exit 1",
                "fi",
                'mv -f "$STAR_BAM" "$BAM"',
                f'{samtools_exe} index -@ "$INDEX_THREADS" "$BAM"',
            ]
        if use_bwa:
            body += bwa_align.metrics_lines()
        body += ["mark_checkpoint_complete", 'echo "Wrote $BAM"']
        align_body = "\n".join(body)
        script.write_text(f'''#!/usr/bin/env bash
{partition_line}#SBATCH --job-name={slurm_job_name(workdir, "bwa" if use_bwa else "star")}
#SBATCH --cpus-per-task={placeable_cpus(star_threads, partition, job_name="star")}
#SBATCH --mem={shlex.quote(str(star_mem))}
#SBATCH --array={array_spec}
{tmp_line}#SBATCH --output={shlex.quote(str(workdir))}/logs/star_%A_%a.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/star_%A_%a.err

set -euo pipefail

require_tool() {{
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required executable not found in PATH: $1" >&2
    exit 1
  fi
}}
require_tool {aligner_exe}
require_tool {samtools_exe}
{fused_reader_check}

ulimit -n 65535 2>/dev/null || ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true
echo "ulimit -n: $(ulimit -n)"

WORKDIR={shlex_quote(str(workdir))}
MANIFEST={shlex_quote(str(manifest))}
REFERENCE={shlex_quote(str(reference))}
{"BWA_INDEX=" + shlex_quote(str(bwa_align.index_prefix(reference))) if use_bwa else ""}
THREADS="${{SLURM_CPUS_PER_TASK:-{star_threads}}}"
STAR_SORT_RAM={star_sort_ram}
SAMTOOLS={samtools_exe}
{bamutil_line}

line=$(awk -F'\\t' -v id="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==id {{print; exit}}' "$MANIFEST")
if [[ -z "$line" ]]; then
  echo "ERROR: could not find manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=$'\\t' read -r IDX SAMPLE_ID R1 R2 REPLACE <<< "$line"
{bam_lines}

mark_checkpoint_complete() {{
  PARALOME_WORKDIR="$WORKDIR" PARALOME_SAMPLE_ID="$SAMPLE_ID" PARALOME_BAM="$BAM" {python_exe} - <<'PYCHECK'
import os
from pathlib import Path
# Only what the heredoc uses, so no array task pays to import more.
from paralome.checkpoints import update_stage_checkpoint
workdir = Path(os.environ["PARALOME_WORKDIR"])
sample_id = os.environ["PARALOME_SAMPLE_ID"]
bam = Path(os.environ["PARALOME_BAM"])
def mark_complete(ckpt):
    if sample_id not in ckpt.samples:
        raise KeyError(f"Sample {{sample_id}} is absent from the alignment checkpoint")
    ckpt.mark_complete(sample_id, outputs={{"bam": str(bam), "bai": str(bam) + ".bai"}})
    ckpt.samples[sample_id].metadata.pop("{TO_BE_REPLACED}", None)

update_stage_checkpoint(workdir, "alignment", mark_complete)
PYCHECK
}}

# REPLACE is 1 while this sample is still to be replaced under
# --overwrite; the BAM in place is then the old one, not a finished sample.
if [[ -s "$BAM" && -s "$BAM.bai" && "${{REPLACE:-0}}" != "1" ]]; then
  echo "BAM already exists; skipping $SAMPLE_ID: $BAM"
  mark_checkpoint_complete
  exit 0
fi
# Whatever is at $BAM now is not this run's. STAR writes under temp/
# and the BAM is moved into place at the end, so if this task dies the previous
# BAM would be left standing as the result. It goes before STAR starts.
rm -f "$BAM" "$BAM.bai"

{align_body}
''')
        script.chmod(0o755)
    else:
        script.write_text("#!/usr/bin/env bash\necho 'All STAR alignment samples are already complete; nothing to run.'\n")
        script.chmod(0o755)

    summary = {
        "workdir": str(workdir),
        "samples": len(samples),
        "expected_samples": len(samples),
        "completed_samples": len(ckpt.complete_samples()),
        "missing_samples": ckpt.missing_samples(),
        "active_indices": active_indices,
        "array_spec": array_spec,
        "aligned_manifest": str(aligned_manifest),
        "script": str(script),
        "backend": record_backend,
        "submitted": False,
        "requested_threads": requested_threads,
        "threads_per_sample": star_threads,
        "max_star_threads_per_sample": MAX_STAR_THREADS_PER_SAMPLE,
        "array_concurrency": requested_throttle,
        "jobs": requested_throttle,
        "peak_cpus": star_threads * requested_throttle if requested_throttle else star_threads,
        "samtools_index_threads": min(4, max(1, int(star_threads))),
        # The SLURM path records what it decided too. This summary
        # is written at SUBMISSION, which is the only moment these are known
        # in one place -- the array tasks each see only their own sample.
        "parameters": {
            "read_files_command": read_files_command,
            "out_reads_unmapped": out_reads_unmapped,
            "out_filter_multimap_nmax": out_filter_multimap_nmax,
            "out_filter_mismatch_nover_read_lmax": out_filter_mismatch_nover_read_lmax,
            "out_filter_mismatch_nmax": out_filter_mismatch_nmax,
            "out_filter_score_min_over_lread": out_filter_score_min_over_lread,
            "out_filter_match_nmin_over_lread": out_filter_match_nmin_over_lread,
            "loose": bool(loose),
            "star_extra": star_extra,
            "sjdbGTFfile_passed": not junctions_are_baked_in and bool(gtf_path),
        },
        "max_samtools_index_threads": 4,
        "mem_per_sample": star_mem,
        "mem_per_sample_mb": parse_mem_to_mb(star_mem) or star_memory.mem_mb,
        "mem_basis": memory_basis,
        "fused_memory_plan": fused_memory.to_dict() if fused_memory else None,
        "star_sort_ram_bytes": star_sort_ram,
        "star_memory_source": star_memory.memory_source,
        "star_memory_node_cpus": star_memory.node_cpus,
        "star_memory_node_mem_mb": star_memory.node_mem_mb,
        "scheduler_capabilities": capabilities.to_dict(),
        "array_plan": array_plan.to_dict(),
        "planner_diagnostics": format_plan_diagnostics(
            "STAR Scheduler Plan",
            (
                ("Samples", sample_count),
                ("Requested threads", requested_threads),
                ("Task CPUs", star_threads),
                ("Concurrent tasks", requested_throttle),
                ("Memory/task", f"{star_mem} ({memory_basis})"),
                ("STAR sort RAM", star_sort_ram),
                ("MaxArraySize", capabilities.max_array_size),
            ),
        ),
        "reference": str(reference),
        "aligner": aligner,
        "bwa_index": str(bwa_align.index_prefix(reference)) if use_bwa else None,
        "bwa_extra": bwa_extra if use_bwa else None,
        "gtf": str(gtf_path) if gtf_path else None,
        "checkpoint": str(stage_checkpoint_path(workdir, "alignment")),
        "complete": len(active_indices) == 0,
    }

    finalize_script = write_sbatch(
        workdir / "scripts" / "star_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(workdir, "bwa_finalize" if use_bwa else "star_finalize"),
            cpus_per_task=1,
            mem="4G",
            partition=partition,
            output=workdir / "logs" / "star_finalize_%j.out",
            error=workdir / "logs" / "star_finalize_%j.err",
        ),
        shlex.join([
            *worker_argv(), "align", "verify", "--workdir", str(workdir),
            # In a fused run the fastp reports land in the FASTQ QC
            # stage's own directories as each task finishes, and nothing had
            # summarized them for an assay without a deduplication stage to
            # adopt them (RNA). The alignment finalizer is the first moment
            # every task is done, so it belongs here rather than in
            # `dedup adopt`.
            *(["--fastq-qc-workdir", str(trim_in_task["workdir"])]
              if trim_in_task and trim_in_task.get("workdir") else []),
        ]),
    )
    summary["finalize_script"] = str(finalize_script)
    summary["submitted_job_ids"] = []

    if submit and active_indices:
        _require_tool("sbatch")
        # Each ID is reported and recorded the moment its sbatch returns, so a refused finalizer
        # never leaves an array queued with nothing naming it. stderr: the supervisor reads it.
        index_job = []
        if use_bwa and not bwa_align.index_ready(reference):
            # The first BWA run on this genome: one job builds the index, and the array waits on it.
            fasta_bytes = reference.stat().st_size
            index_script = write_sbatch(
                workdir / "scripts" / "bwa_index.sbatch",
                SlurmOptions(
                    job_name=slurm_job_name(workdir, "bwa_index"), cpus_per_task=1,
                    mem=f"{max(8, -(-40 * fasta_bytes // 1024 ** 3))}G", partition=partition,
                    output=workdir / "logs" / "bwa_index_%j.out", error=workdir / "logs" / "bwa_index_%j.err",
                ),
                bwa_align.index_build_bash(reference),
            )
            index_job = [sbatch(index_script)]
            print(f"Submitted bwa-mem2 index job: {index_job[0]}", file=sys.stderr, flush=True)
            _record_submitted_jobs(workdir, index_job)
            summary["bwa_index_job_id"] = index_job[0]
        array_job_id = sbatch(script, dependency=afterok(index_job[0])) if index_job else sbatch(script)
        print(f"Submitted {'BWA' if use_bwa else 'STAR'} array job: {array_job_id}", file=sys.stderr, flush=True)
        _record_submitted_jobs(workdir, [*index_job, array_job_id])
        finalize_job_id = sbatch(finalize_script, dependency=afterok(array_job_id))
        print(f"Submitted STAR finalizer job: {finalize_job_id}", file=sys.stderr, flush=True)
        _record_submitted_jobs(workdir, [*index_job, array_job_id, finalize_job_id])
        summary["submitted"] = True
        summary["job_id"] = array_job_id
        summary["finalize_job_id"] = finalize_job_id
        summary["submitted_job_ids"] = [*index_job, array_job_id, finalize_job_id]
    elif submit and not active_indices:
        summary["submitted"] = False
        summary["job_id"] = None
        summary["finalize_job_id"] = None
        summary["message"] = "All STAR alignment samples are already complete."

    (workdir / "results" / "star_alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_checkpoint(workdir, {"alignment": {"star": {"complete": len(active_indices) == 0, **summary}}})
    return summary

def run_star_fused_local(*, jobs: int = 0, **options: Any) -> dict[str, Any]:
    """Run the fused alignment tasks on this machine.

    This writes exactly the task script the SLURM backend writes -- fastp
    streaming into STAR, STAR streaming into duplicate marking -- and runs each
    sample's task here, with the array index and CPU count a SLURM task would
    be given. One script for both backends means the two cannot drift apart.
    """
    summary = write_star_slurm(jobs=jobs, submit=False, record_backend="local", **options)
    workdir = Path(summary["workdir"])
    if summary.get("aligner") == "bwa" and summary.get("active_indices"):
        from paralome.align.bwa import build_index_here

        build_index_here(Path(summary["reference"]))
    script = Path(summary["script"])
    indices = list(summary.get("active_indices") or [])
    threads = int(summary.get("threads_per_sample") or 1)
    at_once = max(1, int(summary.get("array_concurrency") or 1))
    env = dict(os.environ)
    # paralome-buffer and the worker sit beside the interpreter that runs this.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["SLURM_CPUS_PER_TASK"] = str(threads)
    logs = workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def _task(index: int) -> tuple[int, int]:
        task_env = dict(env, SLURM_ARRAY_TASK_ID=str(index))
        with (logs / f"star_local_{index}.out").open("w") as out_fh, \
                (logs / f"star_local_{index}.err").open("w") as err_fh:
            done = subprocess.run(["bash", str(script)], env=task_env, stdout=out_fh, stderr=err_fh)
        return index, done.returncode

    from concurrent.futures import ThreadPoolExecutor

    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=at_once) as pool:
        for index, code in pool.map(_task, indices):
            if code != 0:
                failed.append(index)
    if failed:
        samples = read_fastq_manifest(options["samples_csv"])
        names = [f"{samples[i - 1].sample_id} (logs/star_local_{i}.err)" for i in failed if 0 < i <= len(samples)]
        raise RuntimeError(
            f"STAR alignment in {workdir}: the task failed for " + ", ".join(names)
            + ". The other samples are kept; run the same command again to retry these."
        )
    summary["submitted"] = False
    summary["ran_here"] = indices
    return summary


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)
