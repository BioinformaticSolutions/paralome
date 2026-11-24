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
import os
from dataclasses import replace
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from paralome.binaries import worker_argv
from paralome.stream import fastp_speed_args
from paralome.pipeline.naming import slurm_job_name
from paralome.pipeline.resources import plan_sample_parallel
from paralome.pipeline.slurm import SlurmOptions, afterok, placeable_cpus, sbatch, write_sbatch

from .manifest import SampleInput, read_sample_inputs, validate_input_paths, write_fastq_manifest
from .summary import replace_mark, summarize_fastp_workdir

FASTP_MAX_THREADS_PER_SAMPLE = 15

#: How long a stage command keeps asking a SLURM that does not
#: answer about its earlier jobs before it refuses, and how often it asks.
SQUEUE_RETRY_SECONDS = 600
SQUEUE_RETRY_INTERVAL_SECONDS = 30


def _require_tool(name: str) -> str:
    """The tool to invoke, from what setup recorded -- not from PATH.

    See binaries.resolve_tool. Returns the resolved absolute path so
    the caller can invoke THAT rather than the bare name it asked about; a
    recorded path that no longer resolves raises PLM-0410 instead of silently
    falling back to whatever this account's PATH happens to find.
    """
    from paralome.binaries import resolve_tool

    exe = resolve_tool(name)
    if exe is None:
        # The tool is looked for where `paralome setup` recorded it first, then on PATH.
        raise RuntimeError(
            f"Required tool not found: {name} (not recorded by `paralome setup`, not on PATH)"
        )
    return exe


def _sample_outputs(workdir: Path, sample: SampleInput) -> dict[str, Path]:
    trimmed = workdir / "trimmed"
    json_dir = workdir / "fastp_json"
    log_dir = workdir / "logs"
    for d in (trimmed, json_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "r1": trimmed / f"{sample.sample_id}.R1.fastq.gz",
        "r2": trimmed / f"{sample.sample_id}.R2.fastq.gz",
        "json": json_dir / f"{sample.sample_id}.json",
        "log": log_dir / f"{sample.sample_id}.fastp.log",
    }


def _sample_finished(out: dict[str, Path]) -> bool:
    """A sample is finished when its report and its R1 are there and not empty.

    One test for both backends.
    """
    return all(out[key].is_file() and out[key].stat().st_size > 0 for key in ("json", "r1"))


def _sample_done(workdir: Path, sample: SampleInput) -> bool:
    """Finished, and not waiting to be replaced under an earlier --overwrite."""
    return not replace_mark(workdir, sample.sample_id).exists() and _sample_finished(_sample_outputs(workdir, sample))


def _same_input(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> bool:
    """The same file: the same path and the same size.

    Not the date: FASTQs copied or restored without their times (cp, rsync
    without -t) are not changed, and redoing QC gives the trimmed FASTQs new
    dates, which alignment would then refuse in the same way.
    """
    if not before or not after:
        return not before and not after
    return before.get("path") == after.get("path") and before.get("size") == after.get("size")


def input_signature(path: Union[str, Path, None]) -> Optional[dict[str, Any]]:
    """Path, size and modification time of one input file, for the record.

    A file that is not there is recorded with no size, so a path can
    be recorded before it is validated. The time is recorded and not compared;
    see _same_input.
    """
    if not path or str(path) == ".":
        return None
    absolute = os.path.abspath(str(path))
    try:
        stat = os.stat(absolute)
    except OSError:
        return {"path": absolute, "size": None, "mtime_ns": None}
    return {"path": absolute, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _by_option(tokens: list[Any]) -> dict[str, list[str]]:
    """A tool's arguments as {--option: its values}, in the order given."""
    grouped: dict[str, list[str]] = {}
    option = ""
    for token in tokens:
        if str(token).startswith("--"):
            option = str(token)
            grouped.setdefault(option, [])
        else:
            grouped.setdefault(option, []).append(str(token))
    return grouped


def _option_text(values: Optional[list[str]]) -> str:
    if values is None:
        return "not given"
    return " ".join(values) if values else "given"


def request_differences(made_with: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """What differs between how a result was made and what is asked for now.

    Empty when nothing does. Lists (the tool's arguments) are shown
    as the tokens each side has that the other lacks; inputs by file.
    """
    differences: list[str] = []
    for key in sorted(set(made_with) | set(request)):
        old, new = made_with.get(key), request.get(key)
        if old == new:
            continue
        if key == "inputs" and isinstance(old, dict) and isinstance(new, dict):
            for label in sorted(set(old) | set(new)):
                before, after = old.get(label), new.get(label)
                if _same_input(before, after):
                    continue
                if not before or not after or before.get("path") != after.get("path"):
                    differences.append(
                        f"{label} was {before.get('path') if before else 'not given'}, "
                        f"now {after.get('path') if after else 'not given'}"
                    )
                else:
                    differences.append(f"{label} {after.get('path')} has changed since (its size)")
        elif isinstance(old, list) and isinstance(new, list):
            before, after = _by_option(old), _by_option(new)
            for option in list(before) + [name for name in after if name not in before]:
                if before.get(option) != after.get(option):
                    differences.append(
                        f"{option or key} was {_option_text(before.get(option))}, "
                        f"now {_option_text(after.get(option))}"
                    )
            if before == after:                  # same options, another order
                differences.append(f"{key} are in another order")
        else:
            differences.append(f"{key} was {old!r}, now {new!r}")
    return differences


def refuse_mixed_results(what: str, workdir: Path, changed: dict[str, list[str]]) -> None:
    """Stop before a stage ends up holding results from two different requests.

    A finished sample is reused only when it was made with the same request
    (options and FASTQ), so a resumed run with other options cannot report the
    whole stage under the new request. Nothing is replaced without being
    asked: the user removes the folder.
    """
    if not changed:
        return
    from paralome.errors import ParalomeError

    lines = [f"{sample_id}: " + "; ".join(reasons) for sample_id, reasons in changed.items()]
    # The console prints the detail and not the fix, so the detail says both.
    fix = (
        f"Remove {workdir} (or pass --overwrite) and run again to redo every sample, "
        "or repeat the earlier options to finish the stage as it was started."
    )
    raise ParalomeError(
        309,
        f"{what} in {workdir} already holds finished samples made with a different request. "
        "Nothing was started.\n" + "\n".join(lines) + "\n" + fix,
        fix=fix,
    )


def refuse_while_jobs_are_alive(
    what: str, workdir: Path, job_ids: Iterable[Any], *, recorded_in: str = "",
) -> None:
    """Stop a second submission beside one that is still running.

    The IDs of what was queued are recorded, so a stage command run by hand
    twice does not submit every unfinished sample twice. Returning means the
    queue answered and none of these jobs is working, so the caller can
    forget the IDs. A SLURM that does not answer is asked again for
    SQUEUE_RETRY_SECONDS, so a short outage does not end the supervisor's
    run. After that this refuses, and submits nothing, rather than guess.

    ``recorded_in`` is the key and file the IDs are kept in, for the refusal
    on a machine that has no squeue to ask.
    """
    ids = {str(job_id).strip() for job_id in job_ids if str(job_id).strip().isdigit()}
    if not ids:
        return
    from paralome.errors import ParalomeError
    from paralome.pipeline.supervisor import SlurmQueryUnavailable, _squeue_job_records, _working_roots

    listed = ", ".join(sorted(ids, key=int))
    if not shutil.which("squeue"):
        # A machine with no squeue will not grow one by being asked
        # again. This said squeue had not answered and to run again when it
        # did, which was untrue and left no way on. The record is named so
        # that someone who knows the jobs have ended can say so.
        record = recorded_in or f"the record of them under {workdir}"
        fix = (
            "Run this on the cluster, where squeue can be asked. Or, if you know those "
            f"jobs have ended, remove {record} and run again."
        )
        raise ParalomeError(
            311,
            f"{what} in {workdir} was submitted before as job(s) {listed}, and squeue is not on "
            "this machine, so Paralome cannot tell whether they are still running. "
            f"Nothing was started.\n{fix}",
            fix=fix,
        )
    deadline = time.monotonic() + SQUEUE_RETRY_SECONDS
    told = False
    while True:
        try:
            alive = _working_roots(_squeue_job_records(ids))
            break
        except SlurmQueryUnavailable as exc:
            if time.monotonic() >= deadline:
                fix = "Run the same command again when squeue answers."
                raise ParalomeError(
                    311,
                    f"{what} in {workdir} was submitted before as job(s) {listed}, and SLURM could not be "
                    f"asked whether they are still running ({exc}). Nothing was submitted.\n{fix}",
                    fix=fix,
                )
            if not told:
                # Worded so that the supervisor does not read these as new job IDs.
                print(
                    f"{what}: SLURM could not be asked whether earlier job(s) {listed} are still "
                    f"running ({exc}). Asking again for up to {SQUEUE_RETRY_SECONDS // 60} minutes.",
                    file=sys.stderr, flush=True,
                )
                told = True
            time.sleep(SQUEUE_RETRY_INTERVAL_SECONDS)
    if alive:
        fix = "Wait for them to finish, or scancel them, then run again."
        raise ParalomeError(
            311,
            f"{what} in {workdir} is still running from an earlier submission: job(s) "
            + ", ".join(sorted(alive, key=int)) + f". Nothing was submitted.\n{fix}",
            fix=fix,
        )


def _requests_path(workdir: Path) -> Path:
    return workdir / "inputs" / "fastp_requests.json"


def _load_requests(workdir: Path) -> dict[str, Any]:
    """What each finished sample was made with, and the jobs last submitted."""
    try:
        payload = json.loads(_requests_path(workdir).read_text())
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not isinstance(payload.get("samples"), dict):
        payload["samples"] = {}
    if not isinstance(payload.get("submitted_job_ids"), list):
        payload["submitted_job_ids"] = []
    return payload


def _save_requests(workdir: Path, payload: dict[str, Any]) -> None:
    path = _requests_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sample_request(sample: SampleInput, **trimming: Any) -> dict[str, Any]:
    """Everything that decides a sample's trimmed reads: arguments and inputs.

    Threads, memory and partition are left out: they do not change the result.
    """
    return {
        "fastp_args": fastp_trimming_args(paired=bool(sample.fastq_r2), **trimming),
        "inputs": {
            "fastq_r1": input_signature(sample.fastq_r1),
            "fastq_r2": input_signature(sample.fastq_r2),
        },
    }


def _prepare_requests(
    samples: list[SampleInput],
    workdir: Path,
    *,
    overwrite: bool,
    trimming: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """This run's request per sample; refuses when a finished one was made otherwise."""
    record = _load_requests(workdir)
    requests = {sample.sample_id: _sample_request(sample, **trimming) for sample in samples}
    if not overwrite:
        changed: dict[str, list[str]] = {}
        for sample in samples:
            made_with = record["samples"].get(sample.sample_id)
            # No record: finished by an older Paralome, accepted.
            if not isinstance(made_with, dict) or not _sample_done(workdir, sample):
                continue
            differences = request_differences(made_with, requests[sample.sample_id])
            if differences:
                changed[sample.sample_id] = differences
        refuse_mixed_results("FASTQ QC", workdir, changed)
    return record, requests


def _record_requests(
    samples: list[SampleInput],
    workdir: Path,
    record: dict[str, Any],
    requests: dict[str, dict[str, Any]],
) -> None:
    """Write the request down for every sample that is about to be made.

    A sample marked to be replaced is about to be made, whatever is on disk.
    """
    for sample in samples:
        if not _sample_done(workdir, sample):
            record["samples"][sample.sample_id] = requests[sample.sample_id]
    _drop_marks_off_sheet(samples, workdir, record)
    _save_requests(workdir, record)


def _drop_marks_off_sheet(samples: list[SampleInput], workdir: Path, record: dict[str, Any]) -> list[str]:
    """Withdraw --overwrite marks left for samples no longer on the sheet.

    A mark is cleared only by the sample's own task, and the state reader
    (pipeline.state.summarize_fastq_qc) calls the stage incomplete while any
    mark exists. A sample marked by `fastq qc --overwrite` and then taken off
    the sheet (its FASTQ was corrupt, say) would keep its mark for ever, so
    such a mark is withdrawn here, with what it said was the old run's: the
    sample's report and trimmed reads, and its request record. Put back on
    the sheet, the sample is trimmed again rather than reused as if it had
    been made with this request.
    """
    on_sheet = {sample.sample_id for sample in samples}
    marks = replace_mark(workdir, "sample").parent
    dropped: list[str] = []
    if not marks.is_dir():
        return dropped
    for mark in sorted(marks.iterdir()):
        if mark.name in on_sheet:
            continue
        for path in _outputs_by_id(workdir, mark.name):
            path.unlink(missing_ok=True)
        record["samples"].pop(mark.name, None)
        mark.unlink(missing_ok=True)
        dropped.append(mark.name)
    return dropped


def _outputs_by_id(workdir: Path, sample_id: str) -> list[Path]:
    """One sample's report and trimmed reads, by its ID alone (no sheet row needed)."""
    return [
        workdir / "fastp_json" / f"{sample_id}.json",
        workdir / "trimmed" / f"{sample_id}.R1.fastq.gz",
        workdir / "trimmed" / f"{sample_id}.R2.fastq.gz",
    ]


#: The stage's own summaries. The first is what the pipeline reads as "complete".
_STAGE_SUMMARIES = ("fastq_qc_summary.json", "fastq_qc_summary.tsv", "fastp_run_summary.json")


def _mark_for_replacement(samples: list[SampleInput], workdir: Path) -> None:
    """--overwrite: every sample is to be replaced, and the stage is not complete.

    Nothing is deleted here, so a refused submission leaves the old results.
    A mark (inputs/to_be_replaced/<sample>) says the sample's outputs are the
    old ones: the skip test in both backends and in the array task reads it,
    the task removes that sample's old outputs when it starts, and only a
    sample that finishes loses its mark. A later run without --overwrite
    therefore redoes exactly the samples not yet replaced.

    The summaries are set aside under another name, not removed: a summary in
    place reads as a finished stage.
    """
    for sample in samples:
        mark = replace_mark(workdir, sample.sample_id)
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text("--overwrite: to be replaced\n")
    for name in _STAGE_SUMMARIES:
        path = workdir / "results" / name
        if path.is_file():
            path.replace(path.with_name(name + ".before_overwrite"))


def _discard_sample_outputs(workdir: Path, sample: SampleInput) -> None:
    """Remove one sample's report and trimmed reads, just before it is made again."""
    out = _sample_outputs(workdir, sample)
    for key in ("json", "r1", "r2"):
        out[key].unlink(missing_ok=True)


#: A standard Trimmomatic RNA-seq trimming, expressed in fastp.
#:
#:     java -jar trimmomatic-0.39.jar SE -phred33 ... \
#:         ILLUMINACLIP:TruSeq_PE.fa:2:30:10 SLIDINGWINDOW:4:15 MINLEN:50
#:
#: WHY THESE ARE DEFAULTS AND NOT SUGGESTIONS. fastp's own `length_required` is
#: 15, so reads Trimmomatic discards below 50bp would survive QC and then be
#: refused by STAR and counted as "unmapped: too short". A read kept by QC and
#: rejected by the aligner is worse than a read dropped by QC, because only the
#: second one is counted anywhere a person looks.
DEFAULT_CUT_RIGHT = True
DEFAULT_CUT_RIGHT_WINDOW_SIZE = 4
DEFAULT_CUT_RIGHT_MEAN_QUALITY = 15
DEFAULT_LENGTH_REQUIRED = 50


def fastp_trimming_args(
    *,
    adapter_fasta: Optional[Path] = None,
    cut_right: bool = DEFAULT_CUT_RIGHT,
    cut_right_window_size: Optional[int] = None,
    cut_right_mean_quality: Optional[int] = None,
    length_required: Optional[int] = None,
    extra_fastp_options: Optional[str] = None,
    paired: bool = False,
    keep_files: bool = True,
) -> list[str]:
    """Every trimming argument, built once for both backends, so the two cannot differ.

    WHY --cut_right AND NOT --cut_tail. They are not variants of one idea:

        --cut_tail   scans 3' -> 5' and stops at the first window that PASSES,
                     nibbling only the tail
        --cut_right  scans 5' -> 3' and, at the first window that FAILS, drops
                     that window and everything right of it

    Trimmomatic's SLIDINGWINDOW is the second one, so cut_right is used, with
    the 4:15 window.
    """
    args: list[str] = []
    if adapter_fasta:
        args.extend(["--adapter_fasta", str(adapter_fasta)])
    elif paired:
        # Single-end needs no equivalent: fastp auto-detects the adapter from
        # overrepresented end k-mers for SE by default. It is still a guess,
        # which is why --fastp-adapter-fasta is worth setting per site.
        args.append("--detect_adapter_for_pe")
    if cut_right:
        args.append("--cut_right")
        args.extend([
            "--cut_right_window_size",
            str(int(cut_right_window_size if cut_right_window_size is not None
                    else DEFAULT_CUT_RIGHT_WINDOW_SIZE)),
        ])
        args.extend([
            "--cut_right_mean_quality",
            str(int(cut_right_mean_quality if cut_right_mean_quality is not None
                    else DEFAULT_CUT_RIGHT_MEAN_QUALITY)),
        ])
    args.extend([
        "--length_required",
        str(int(length_required if length_required is not None
                else DEFAULT_LENGTH_REQUIRED)),
    ])
    # The two arguments that cost nothing and save real time.
    # --dont_eval_duplication drops a number nothing reads (Paralome marks
    # duplicates in its own stage); -z 1 stops the stage compressing hard a
    # file that is read exactly once, by the aligner, and then never again.
    # `keep_files` is False in a fused task, where the reads go into a FIFO
    # and there is nothing to compress.
    args.extend(fastp_speed_args(keep_files=keep_files))
    if extra_fastp_options:
        args.extend(shlex.split(extra_fastp_options))
    return args


def _fastp_cmd(
    sample: SampleInput,
    workdir: Path,
    threads: int,
    overwrite: bool,
    adapter_fasta: Optional[Path] = None,
    cut_right: bool = DEFAULT_CUT_RIGHT,
    cut_right_window_size: Optional[int] = None,
    cut_right_mean_quality: Optional[int] = None,
    length_required: Optional[int] = None,
    extra_fastp_options: Optional[str] = None,
) -> list[str]:
    if not sample.fastq_r1:
        raise ValueError(f"Sample {sample.sample_id} has no fastq_r1")
    out = _sample_outputs(workdir, sample)
    if _sample_done(workdir, sample) and not overwrite:
        return []
    from paralome.binaries import resolve_tool

    cmd = [
        # What setup recorded, not what this account's PATH finds.
        resolve_tool("fastp") or "fastp",
        "-i", str(sample.fastq_r1),
        "-o", str(out["r1"]),
        "-j", str(out["json"]),
        # No per-sample HTML report: the JSON is what the stage reads, and the viewer shows the summary.
        "-h", "/dev/null",
        "-w", str(int(threads)),
    ]
    if sample.fastq_r2:
        cmd.extend(["-I", str(sample.fastq_r2), "-O", str(out["r2"])])
    cmd.extend(fastp_trimming_args(
        adapter_fasta=adapter_fasta,
        cut_right=cut_right,
        cut_right_window_size=cut_right_window_size,
        cut_right_mean_quality=cut_right_mean_quality,
        length_required=length_required,
        extra_fastp_options=extra_fastp_options,
        paired=bool(sample.fastq_r2),
    ))
    return cmd


def _write_trimmed_manifest(samples: list[SampleInput], workdir: Path) -> Path:
    path = workdir / "inputs" / "trimmed_samples.csv"
    trimmed: list[SampleInput] = []
    for sample in samples:
        out = _sample_outputs(workdir, sample)
        extra = dict(sample.extra or {})
        extra["fastq_r1"] = str(out["r1"])
        extra["fastq_r2"] = str(out["r2"]) if sample.fastq_r2 else "."
        trimmed.append(
            replace(
                sample,
                fastq_r1=str(out["r1"]),
                fastq_r2=str(out["r2"]) if sample.fastq_r2 else None,
                bam=None,
                extra=extra,
            )
        )
    return write_fastq_manifest(trimmed, path)


def run_fastp_local(
    samples_csv: Union[str, Path],
    workdir: Union[str, Path],
    *,
    threads: int = 4,
    overwrite: bool = False,
    adapter_fasta: Optional[Path] = None,
    cut_right: bool = DEFAULT_CUT_RIGHT,
    cut_right_window_size: Optional[int] = None,
    cut_right_mean_quality: Optional[int] = None,
    length_required: Optional[int] = None,
    extra_fastp_options: Optional[str] = None,
) -> dict:
    _require_tool("fastp")
    workdir = Path(workdir)
    samples = read_sample_inputs(samples_csv)
    errors = validate_input_paths(samples)
    if errors:
        raise FileNotFoundError("Input path validation failed:\n" + "\n".join(errors))
    fastq_samples = [s for s in samples if s.fastq_r1]
    if not fastq_samples:
        raise ValueError("No FASTQ samples found. Expected fastq_r1/fastq_r2 columns.")
    resource_plan = plan_sample_parallel(
        total_threads=threads,
        sample_count=len(fastq_samples),
        max_threads_per_sample=FASTP_MAX_THREADS_PER_SAMPLE,
        jobs=1,
    )
    trimming = dict(
        adapter_fasta=adapter_fasta,
        cut_right=cut_right,
        cut_right_window_size=cut_right_window_size,
        cut_right_mean_quality=cut_right_mean_quality,
        length_required=length_required,
        extra_fastp_options=extra_fastp_options,
    )
    record, requests = _prepare_requests(fastq_samples, workdir, overwrite=overwrite, trimming=trimming)
    if overwrite or not all(_sample_done(workdir, s) for s in fastq_samples):
        # As the SLURM backend asks. Run here beside an array that
        # was still going, this trimmed the same samples into the same files.
        refuse_while_jobs_are_alive(
            "FASTQ QC", workdir, record["submitted_job_ids"],
            recorded_in=f'the entry "submitted_job_ids" from {_requests_path(workdir)}',
        )
        record["submitted_job_ids"] = []
    (workdir / "inputs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(samples_csv, workdir / "inputs" / "original_samples.csv")
    if overwrite:
        _mark_for_replacement(fastq_samples, workdir)
    _record_requests(fastq_samples, workdir, record, requests)

    completed = []
    skipped = []
    for sample in fastq_samples:
        cmd = _fastp_cmd(
            sample,
            workdir,
            threads,
            overwrite,
            adapter_fasta=adapter_fasta,
            cut_right=cut_right,
            cut_right_window_size=cut_right_window_size,
            cut_right_mean_quality=cut_right_mean_quality,
            length_required=length_required,
            extra_fastp_options=extra_fastp_options,
        )
        out = _sample_outputs(workdir, sample)
        if not cmd:
            skipped.append(sample.sample_id)
            continue
        # As in the SLURM task -- an earlier run's report must not outlive the FASTQs it described. The
        # old trimmed reads go here too, when the sample's turn comes, and not all at the start.
        _discard_sample_outputs(workdir, sample)
        with out["log"].open("w") as log:
            log.write("COMMAND: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        replace_mark(workdir, sample.sample_id).unlink(missing_ok=True)
        completed.append(sample.sample_id)

    trimmed_manifest = _write_trimmed_manifest(fastq_samples, workdir)
    summary = summarize_fastp_workdir(workdir)
    summary["completed"] = completed
    summary["skipped"] = skipped
    summary["trimmed_manifest"] = str(trimmed_manifest)
    summary["parameters"] = {
        "adapter_fasta": str(adapter_fasta) if adapter_fasta else None,
        "cut_right": cut_right,
        "cut_right_window_size": cut_right_window_size,
        "cut_right_mean_quality": cut_right_mean_quality,
        "length_required": length_required,
        "extra_fastp_options": extra_fastp_options,
        "resource_allocation": {
            "requested_threads": resource_plan.requested_threads,
            "samples": resource_plan.sample_count,
            "threads_per_sample": int(threads),
            "max_fastp_threads": FASTP_MAX_THREADS_PER_SAMPLE,
            "array_concurrency": resource_plan.concurrent_jobs,
            "peak_cpus": int(threads),
        },
    }
    (workdir / "results" / "fastp_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_fastp_slurm(
    samples_csv: Union[str, Path],
    workdir: Union[str, Path],
    *,
    threads: int = 4,
    jobs: int = 0,
    submit: bool = False,
    overwrite: bool = False,
    partition: Optional[str] = None,
    mem: str = "8G",
    adapter_fasta: Optional[Path] = None,
    cut_right: bool = DEFAULT_CUT_RIGHT,
    cut_right_window_size: Optional[int] = None,
    cut_right_mean_quality: Optional[int] = None,
    length_required: Optional[int] = None,
    extra_fastp_options: Optional[str] = None,
) -> dict:
    workdir = Path(workdir)
    samples = [s for s in read_sample_inputs(samples_csv) if s.fastq_r1]
    if not samples:
        raise ValueError("No FASTQ samples found. Expected fastq_r1/fastq_r2 columns.")
    # As the local runner does: a missing FASTQ is refused before the scripts are written and the
    # array submitted.
    errors = validate_input_paths(samples)
    if errors:
        raise FileNotFoundError("Input path validation failed:\n" + "\n".join(errors))
    trimming = dict(
        adapter_fasta=adapter_fasta,
        cut_right=cut_right,
        cut_right_window_size=cut_right_window_size,
        cut_right_mean_quality=cut_right_mean_quality,
        length_required=length_required,
        extra_fastp_options=extra_fastp_options,
    )
    record, requests = _prepare_requests(samples, workdir, overwrite=overwrite, trimming=trimming)
    if overwrite or not all(_sample_done(workdir, s) for s in samples):
        # Also when only the scripts are being written: they and the record
        # of each sample's request belong to the tasks still running.
        refuse_while_jobs_are_alive(
            "FASTQ QC", workdir, record["submitted_job_ids"],
            recorded_in=f'the entry "submitted_job_ids" from {_requests_path(workdir)}',
        )
        # The queue answered and they are gone, so they are not asked about again (a squeue outage
        # would otherwise stop the stage over jobs that ended long before).
        record["submitted_job_ids"] = []
    if submit:
        _require_tool("sbatch")
    for d in (workdir / "inputs", workdir / "scripts", workdir / "logs", workdir / "trimmed", workdir / "fastp_json", workdir / "results"):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(samples_csv, workdir / "inputs" / "original_samples.csv")
    manifest = workdir / "inputs" / "fastq_qc_manifest.tsv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_id", "r1", "r2"])
        for sample in samples:
            writer.writerow([sample.sample_id, sample.fastq_r1, sample.fastq_r2 or "."])
    _write_trimmed_manifest(samples, workdir)

    # The same builder the local runner uses. This list was written
    # out a second time here and had already lost --detect_adapter_for_pe.
    # Decided per sample, as the local runner does. One paired
    # sample on the sheet gave --detect_adapter_for_pe to every task, the
    # single-end ones included. What only a paired sample takes is added in
    # the task, where it knows whether it has an R2.
    fastp_args = fastp_trimming_args(paired=False, **trimming)
    paired_only_args = [
        arg for arg in fastp_trimming_args(paired=True, **trimming) if arg not in fastp_args
    ]
    paired_only_text = " ".join(shlex.quote(x) for x in paired_only_args)
    # Written into the script as a bash array, each argument quoted, so a path with a space in it
    # reaches fastp as one argument.
    fastp_args_text = " ".join(shlex.quote(x) for x in fastp_args)

    sample_count = len(samples)
    # --jobs is how many samples run at once, here as in alignment
    # and duplicate marking; --threads is the stage's total, divided among them.
    resource_plan = plan_sample_parallel(
        total_threads=threads,
        sample_count=sample_count,
        max_threads_per_sample=FASTP_MAX_THREADS_PER_SAMPLE,
        jobs=jobs,
    )
    fastp_threads = resource_plan.threads_per_sample
    array_throttle = resource_plan.concurrent_jobs

    params_path = workdir / "inputs" / "fastp_parameters.json"
    params_path.write_text(json.dumps({
        "adapter_fasta": str(adapter_fasta) if adapter_fasta else None,
        "cut_right": cut_right,
        "cut_right_window_size": cut_right_window_size,
        "cut_right_mean_quality": cut_right_mean_quality,
        "length_required": length_required,
        "extra_fastp_options": extra_fastp_options,
        "fastp_args": fastp_args,
        "fastp_args_paired_end_only": paired_only_args,
        "resource_allocation": {
            "requested_threads": resource_plan.requested_threads,
            "samples": resource_plan.sample_count,
            "threads_per_sample": resource_plan.threads_per_sample,
            "max_fastp_threads": FASTP_MAX_THREADS_PER_SAMPLE,
            "array_concurrency": resource_plan.concurrent_jobs,
            "peak_cpus": resource_plan.peak_cpus,
        },
    }, indent=2, sort_keys=True) + "\n")
    partition_line = f"#SBATCH --partition={shlex.quote(str(partition))}\n" if partition else ""
    from paralome.binaries import resolve_tool

    fastp_exe = resolve_tool("fastp") or "fastp"
    script = workdir / "scripts" / "fastp_array.sbatch"
    script.write_text(f'''#!/usr/bin/env bash
{partition_line}#SBATCH --job-name={slurm_job_name(workdir, "fastq_qc")}
#SBATCH --cpus-per-task={placeable_cpus(fastp_threads, partition, job_name="fastq_qc")}
#SBATCH --mem={shlex.quote(str(mem))}
#SBATCH --array=1-{sample_count}%{array_throttle}
#SBATCH --output={shlex.quote(str(workdir))}/logs/fastp_%A_%a.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/fastp_%A_%a.err

set -euo pipefail

MANIFEST={shlex.quote(str(manifest))}
LINE=$(awk -v n="$SLURM_ARRAY_TASK_ID" 'NR==n+1 {{print}}' "$MANIFEST")
SAMPLE=$(echo "$LINE" | cut -f1)
R1=$(echo "$LINE" | cut -f2)
R2=$(echo "$LINE" | cut -f3)

OUTDIR={shlex.quote(str(workdir / 'trimmed'))}
JSONDIR={shlex.quote(str(workdir / 'fastp_json'))}
MARK={shlex.quote(str(replace_mark(workdir, 'x').parent))}/"$SAMPLE"
EXTRA_ARGS=({fastp_args_text})
PAIRED_ARGS=({paired_only_text})
mkdir -p "$OUTDIR" "$JSONDIR"

CMD=({shlex.quote(fastp_exe)} -i "$R1" -o "$OUTDIR/${{SAMPLE}}.R1.fastq.gz" -j "$JSONDIR/${{SAMPLE}}.json" -h /dev/null -w "${{SLURM_CPUS_PER_TASK:-{fastp_threads}}}")
if [[ "$R2" != "." && -n "$R2" ]]; then
  CMD+=( -I "$R2" -O "$OUTDIR/${{SAMPLE}}.R2.fastq.gz")
  CMD+=( ${{PAIRED_ARGS[@]+"${{PAIRED_ARGS[@]}}"}} )
fi
# The ${{X[@]+...}} form: bash before 4.4 calls an empty array unbound under set -u.
CMD+=( ${{EXTRA_ARGS[@]+"${{EXTRA_ARGS[@]}}"}} )
# $MARK is there while this sample is still to be replaced under
# --overwrite; its report and reads are then the old ones, not a finished sample.
if [[ -s "$JSONDIR/${{SAMPLE}}.json" && -s "$OUTDIR/${{SAMPLE}}.R1.fastq.gz" && ! -e "$MARK" ]]; then
  echo "Skipping existing fastp output for $SAMPLE"
  exit 0
fi

# The report goes first. fastp writes the FASTQs as it runs and the JSON at the end, and "JSON present
# and R1 non-empty" is the skip test above, so under --overwrite the previous run's JSON must go before
# fastp starts: a task killed part way must not leave a truncated FASTQ beside an old JSON. The old
# reads go here too, when the task starts.
rm -f "$JSONDIR/${{SAMPLE}}.json" "$OUTDIR/${{SAMPLE}}.R1.fastq.gz" "$OUTDIR/${{SAMPLE}}.R2.fastq.gz"
echo "Running: ${{CMD[*]}}"
"${{CMD[@]}}"
rm -f "$MARK"
''')
    script.chmod(0o755)

    finalize_script = write_sbatch(
        workdir / "scripts" / "fastp_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(workdir, "fastq_qc_finalize"),
            cpus_per_task=1,
            mem="2G",
            partition=partition,
            output=workdir / "logs" / "fastp_finalize_%j.out",
            error=workdir / "logs" / "fastp_finalize_%j.err",
        ),
        shlex.join([*worker_argv(), "fastq", "summarize", "--workdir", str(workdir)]),
    )

    job_id = None
    finalize_job_id = None
    submitted_job_ids: list[str] = []
    if overwrite:
        _mark_for_replacement(samples, workdir)
    _record_requests(samples, workdir, record, requests)
    if submit:
        # Each ID is reported and recorded the moment its sbatch returns, so a refused finalizer never
        # leaves an array queued with nothing naming it. stderr: the supervisor reads it.
        job_id = sbatch(script)
        print(f"Submitted FASTQ QC array job: {job_id}", file=sys.stderr, flush=True)
        record["submitted_job_ids"] = [job_id]
        _save_requests(workdir, record)
        finalize_job_id = sbatch(finalize_script, dependency=afterok(job_id))
        print(f"Submitted FASTQ QC finalizer job: {finalize_job_id}", file=sys.stderr, flush=True)
        submitted_job_ids = [job_id, finalize_job_id]
        record["submitted_job_ids"] = list(submitted_job_ids)
        _save_requests(workdir, record)
    return {
        "workdir": str(workdir),
        "samples": len(samples),
        "script": str(script),
        "finalize_script": str(finalize_script),
        "job_id": job_id,
        "finalize_job_id": finalize_job_id,
        "submitted_job_ids": submitted_job_ids,
        "trimmed_manifest": str(workdir / "inputs" / "trimmed_samples.csv"),
        "resource_allocation": {
            "requested_threads": resource_plan.requested_threads,
            "samples": resource_plan.sample_count,
            "threads_per_sample": resource_plan.threads_per_sample,
            "max_fastp_threads": FASTP_MAX_THREADS_PER_SAMPLE,
            "array_concurrency": resource_plan.concurrent_jobs,
            "peak_cpus": resource_plan.peak_cpus,
        },
    }


def defer_fastq_qc(
    samples_csv: Union[str, Path],
    workdir: Union[str, Path],
    **trimming: Any,
) -> dict:
    """Plan this stage without trimming anything, because alignment will do it.

    With fused stages the trimming happens inside each alignment task,
    streaming into STAR, so there are no trimmed FASTQs for this stage to
    write and nothing for it to run. What it still owes its readers is the
    paperwork: the manifest the alignment stage takes as its input, and a
    summary that says the stage is settled.

    THE MANIFEST NAMES THE RAW FASTQS. The alignment stage prefers a finished
    QC stage's trimmed manifest over the sample sheet, so pointing that
    manifest at the raw reads sends the fused task exactly what it needs, and
    no other stage has to know the difference.

    The real QC numbers do not exist yet -- fastp has not run. The summary
    written here records that, and `paralome-worker dedup adopt --fastq-qc-workdir`
    replaces it with the true one once the fused tasks have written their JSON
    reports into this directory.
    """
    workdir = Path(workdir)
    samples = [s for s in read_sample_inputs(samples_csv) if s.fastq_r1]
    if not samples:
        raise ValueError("No FASTQ samples found. Expected fastq_r1/fastq_r2 columns.")
    errors = validate_input_paths(samples)
    if errors:
        raise FileNotFoundError("Input path validation failed:\n" + "\n".join(errors))
    for d in (workdir / "inputs", workdir / "fastp_json", workdir / "results"):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(samples_csv, workdir / "inputs" / "original_samples.csv")
    # The raw reads, under the name the alignment stage looks for.
    manifest = write_fastq_manifest(samples, workdir / "inputs" / "trimmed_samples.csv")
    args = fastp_trimming_args(paired=False, keep_files=False, **trimming)
    paired_only = [a for a in fastp_trimming_args(paired=True, keep_files=False, **trimming)
                   if a not in args]
    (workdir / "inputs" / "fastp_parameters.json").write_text(json.dumps({
        "deferred_to": "alignment",
        "fastp_args": args,
        "fastp_args_paired_end_only": paired_only,
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in trimming.items()},
    }, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "paralome.fastq_qc.summary.v1",
        "stage": "fastq_qc",
        "deferred_to": "alignment",
        "note": ("trimming runs inside each alignment task and streams into STAR, so no "
                 "trimmed FASTQ is written; fastp's own JSON and HTML reports land in this "
                 "directory as the tasks finish, and `paralome-worker dedup adopt "
                 "--fastq-qc-workdir` then writes the real summary over this one"),
        "totals": {"samples": len(samples), "raw_reads": None, "clean_reads": None},
        "planned": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (workdir / "results" / "fastq_qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "workdir": str(workdir),
        "samples": len(samples),
        "trimmed_manifest": str(manifest),
        "deferred_to": "alignment",
        "backend": "fused-alignment",
    }
