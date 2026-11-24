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

import sys
from pathlib import Path
from typing import Optional

from paralome.fastq.fastp import DEFAULT_CUT_RIGHT, run_fastp_local, write_fastp_slurm
from paralome.fastq.manifest import detect_manifest_mode, read_sample_inputs, validate_input_paths
from paralome.fastq.summary import summarize_fastp_workdir


def inspect_fastq_manifest(samples: Path, validate_paths: bool = False) -> None:
    parsed = read_sample_inputs(samples)
    mode = detect_manifest_mode(parsed)
    print(f"Samples: {len(parsed)}")
    print(f"Input mode: {mode}")
    if validate_paths:
        errors = validate_input_paths(parsed)
        if errors:
            print("Missing inputs:")
            for err in errors:
                print(f"  ERROR: {err}")
        else:
            print("Input path validation: OK")
    for s in parsed:
        print(f"{s.sample_id}\t{s.group}\t{s.relationship}\t{s.input_mode}")


def run_fastq_qc(
    samples: Path,
    out: Path,
    backend: str = "local",
    threads: int = 4,
    jobs: int = 0,
    submit: bool = False,
    overwrite: bool = False,
    partition: Optional[str] = None,
    mem: str = "8G",
    adapter_fasta: Optional[Path] = None,
    cut_right: bool = DEFAULT_CUT_RIGHT,  # The CLI's default, not another one
    cut_right_window_size: Optional[int] = None,
    cut_right_mean_quality: Optional[int] = None,
    length_required: Optional[int] = None,
    extra_fastp_options: Optional[str] = None,
    defer_trimming: bool = False,
) -> None:
    backend = backend.lower().strip()
    if adapter_fasta is None:
        # The site's adapter file, as the pipeline uses. The option's
        # help promised it; run by itself this command let fastp guess instead.
        from paralome.setup.site import site_adapter_fasta

        adapter_fasta = site_adapter_fasta()
        if adapter_fasta is not None:
            print(f"Adapter FASTA: {adapter_fasta} (recorded by `paralome setup site`)")
    if defer_trimming:
        # Nothing to run here, and nothing to wait for. The stage
        # writes its manifest and its placeholder summary and returns.
        from paralome.fastq.fastp import defer_fastq_qc

        planned = defer_fastq_qc(
            samples,
            out,
            adapter_fasta=adapter_fasta,
            cut_right=cut_right,
            cut_right_window_size=cut_right_window_size,
            cut_right_mean_quality=cut_right_mean_quality,
            length_required=length_required,
            extra_fastp_options=extra_fastp_options,
        )
        print(f"FASTQ QC deferred to the alignment stage: {planned['workdir']}")
        print(f"Samples: {planned['samples']}")
        print(f"Manifest (raw reads): {planned['trimmed_manifest']}")
        print("No trimmed FASTQ will be written; fastp's reports arrive as the alignment tasks finish.")
        return
    if backend == "local":
        summary = run_fastp_local(
            samples,
            out,
            threads=threads,
            overwrite=overwrite,
            adapter_fasta=adapter_fasta,
            cut_right=cut_right,
            cut_right_window_size=cut_right_window_size,
            cut_right_mean_quality=cut_right_mean_quality,
            length_required=length_required,
            extra_fastp_options=extra_fastp_options,
        )
        totals = summary.get("totals", {})
        print(f"FASTQ QC complete: {out}")
        print(f"Samples: {totals.get('samples')}")
        print(f"Raw reads: {totals.get('raw_reads')}")
        print(f"Clean reads: {totals.get('clean_reads')}")
        print(f"Reads surviving QC: {totals.get('pct_reads_surviving')}%")
        print(f"Summary: {out / 'results' / 'fastq_qc_summary.tsv'}")
        print(f"Trimmed manifest: {summary.get('trimmed_manifest')}")
        return
    if backend == "slurm":
        result = write_fastp_slurm(
            samples,
            out,
            threads=threads,
            jobs=jobs,
            submit=submit,
            overwrite=overwrite,
            partition=partition,
            mem=mem,
            adapter_fasta=adapter_fasta,
            cut_right=cut_right,
            cut_right_window_size=cut_right_window_size,
            cut_right_mean_quality=cut_right_mean_quality,
            length_required=length_required,
            extra_fastp_options=extra_fastp_options,
        )
        print(f"FASTQ QC SLURM setup: {out}")
        print(f"Samples: {result['samples']}")
        resources = result.get("resource_allocation", {})
        if resources:
            print(f"Requested threads: {resources.get('requested_threads')}")
            print(f"Threads per sample: {resources.get('threads_per_sample')}")
            print(f"Array concurrency: {resources.get('array_concurrency')}")
        print(f"Script: {result['script']}")
        print(f"Trimmed manifest: {result['trimmed_manifest']}")
        if result.get("submitted_job_ids"):
            print("Job ID: " + ",".join(str(value) for value in result["submitted_job_ids"]))
            print(f"Submitted array job: {result['job_id']}")
            print(f"Submitted finalizer job: {result['finalize_job_id']}")
        else:
            print("Not submitted. Add --submit to submit the array and dependent finalizer.")
        return
    raise ValueError(f"Unknown FASTQ QC backend: {backend}")


def summarize_fastq_qc_if_ready(workdir: Path, *, attempts: int = 3, pause: float = 5.0) -> bool:
    """Summarize a fused run's fastp reports, and never fail the caller.

    The reports are written by array tasks on several nodes and read
    back by one finalizer, over NFS. On that read came up
    short-handed -- every report present on disk, none visible to the glob --
    and the stage command died with PLM-0001 "Unexpected internal error". It
    survived only because the stage that happened to own the call had already
    written its own paperwork, so the supervisor's retry found the stage
    complete and moved on.

    This is called from the ALIGNMENT finalizer, where that luck does not
    apply: a raise there fails the alignment stage after every task has done
    its work. So it retries a few times for the directory to settle, and then
    says so and returns False rather than raising. The stage's own reports are
    on disk either way; what is missing is a summary of them, which the next
    run writes.

    Returns True when a summary was written.
    """
    import time

    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            summarize_fastq_qc(workdir)
            return True
        except Exception as exc:  # noqa: BLE001 -- bookkeeping must not fail the stage
            if attempt >= attempts:
                print(
                    f"[paralome] WARNING: the FASTQ QC summary for {workdir} was not written "
                    f"({type(exc).__name__}: {exc}). fastp's reports are still there; the next "
                    "run summarizes them. The alignment itself is unaffected.",
                    file=sys.stderr, flush=True,
                )
                return False
            time.sleep(pause)
    return False


def _deferred_and_waiting(workdir: Path) -> bool:
    """A fused stage whose fastp reports have not all arrived yet.

    The supervisor closes every stage that ran here with
    `fastq summarize`, and a deferred stage has nothing to summarize until the
    alignment tasks have trimmed the reads. It was reported as an internal
    error (PLM-0001) and "the closing check did not run to the end".
    """
    import json

    try:
        params = json.loads((Path(workdir) / "inputs" / "fastp_parameters.json").read_text())
    except (OSError, ValueError):
        return False
    if params.get("deferred_to") != "alignment":
        return False
    try:
        summary = json.loads((Path(workdir) / "results" / "fastq_qc_summary.json").read_text())
    except (OSError, ValueError):
        return True
    return summary.get("deferred_to") == "alignment"


def summarize_fastq_qc(workdir: Path) -> None:
    if _deferred_and_waiting(workdir):
        try:
            summary = summarize_fastp_workdir(workdir)
        except (FileNotFoundError, RuntimeError):
            # Nothing trimmed yet: the alignment stage writes the summary.
            return
    else:
        summary = summarize_fastp_workdir(workdir)
    totals = summary.get("totals", {})
    print(f"Workdir: {workdir}")
    print(f"Samples: {totals.get('samples')}")
    print(f"Raw reads: {totals.get('raw_reads')}")
    print(f"Clean reads: {totals.get('clean_reads')}")
    print(f"Reads surviving QC: {totals.get('pct_reads_surviving')}%")
    print(f"Summary: {summary.get('summary_tsv')}")


# fastq_app was declared inline in cli.py, which meant the worker
# binary could not register it -- and the fastq_qc stage IS machine surface
# (the supervisor's finalizer invokes `fastq summarize`). Moving the group here
# puts it where every other command group already lives, so both entry points
# can mount it.
import typer

fastq_app = typer.Typer(help="Inspect FASTQ inputs and run read-level QC")

@fastq_app.command("inspect")
def fastq_inspect_command(
    samples: Path = typer.Option(..., "--samples", help="Sample manifest CSV"),
    validate_paths: bool = typer.Option(False, "--validate-paths", help="Check that BAM/FASTQ paths exist"),
):
    from paralome.commands.fastq import inspect_fastq_manifest

    inspect_fastq_manifest(samples, validate_paths=validate_paths)

@fastq_app.command("qc")
def fastq_qc_command(
    samples: Path = typer.Option(..., "--samples", help="FASTQ sample manifest CSV"),
    out: Path = typer.Option(..., "--out", help="FASTQ QC output work directory"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(4, "--threads", help="Total FASTQ QC CPU budget; divided across samples up to fastp cap"),
    jobs: int = typer.Option(0, "--jobs", help="Samples trimmed at once with --backend slurm; default all, limited by --threads. Local trims one at a time"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM job array"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing fastp outputs"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("8G", "--mem", help="SLURM memory per sample"),
    adapter_fasta: Optional[Path] = typer.Option(None, "--adapter-fasta", help="Adapter FASTA to trim against. Defaults to what `setup site --adapter-fasta` recorded; without one, fastp guesses the adapter from the reads."),
    cut_right: bool = typer.Option(True, "--cut-right/--no-cut-right", help="Sliding-window quality trim, 5' to 3', cutting the first failing window and everything after it. Trimmomatic's SLIDINGWINDOW. On by default."),
    cut_right_window_size: Optional[int] = typer.Option(None, "--cut-right-window-size", help="Window for --cut-right; default 4 (SLIDINGWINDOW:4:15)"),
    cut_right_mean_quality: Optional[int] = typer.Option(None, "--cut-right-mean-quality", help="Mean quality a --cut-right window must reach; default 15"),
    length_required: Optional[int] = typer.Option(None, "--length-required", help="Discard reads shorter than this AFTER trimming. Default 50. fastp's own default is 15, which keeps reads too short to align."),
    extra_fastp_options: Optional[str] = typer.Option(None, "--extra-fastp-options", help="Additional fastp options as a quoted string"),
    defer_trimming: bool = typer.Option(False, "--defer-trimming", help="Plan the stage but run no fastp: the trimming happens inside each alignment task (`paralome-worker align star --trim-in-task`), streaming into STAR so no trimmed FASTQ is ever written. Writes the stage's manifest pointing at the raw reads."),
):
    from paralome.commands.fastq import run_fastq_qc

    run_fastq_qc(
        samples,
        out,
        backend=backend,
        threads=threads,
        jobs=jobs,
        submit=submit,
        overwrite=overwrite,
        partition=partition,
        mem=mem,
        adapter_fasta=adapter_fasta,
        cut_right=cut_right,
        cut_right_window_size=cut_right_window_size,
        cut_right_mean_quality=cut_right_mean_quality,
        length_required=length_required,
        extra_fastp_options=extra_fastp_options,
        defer_trimming=defer_trimming,
    )

@fastq_app.command("summarize")
def fastq_summarize_command(
    workdir: Path = typer.Option(..., "--workdir", help="FASTQ QC work directory"),
):
    from paralome.commands.fastq import summarize_fastq_qc

    summarize_fastq_qc(workdir)

