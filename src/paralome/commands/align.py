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

from pathlib import Path
from typing import Optional

import typer

from paralome.align.star import run_star_local, write_star_slurm
from paralome.commands.align_verify import run_align_verify

align_app = typer.Typer(help="Align reads and summarize alignment results")


def _fusion_plans(*, mark_in_place, trim_in_task, fuse_dedup, trim_adapter_fasta, trim_length_required,
                  trim_extra, fused_buffer, trim_threads, clip_overlap=False, trim_cut_right=True,
                  trim_cut_right_window_size=None, trim_cut_right_mean_quality=None):
    """The fused shapes' plans (fastp in the task, duplicate marking in the task), for either aligner."""
    # Made the fused shapes SLURM-only. Fusing
    # is the default, so a local run takes the same task script and runs it
    # here (run_star_fused_local) rather than refusing at alignment.
    if mark_in_place and fuse_dedup:
        raise typer.BadParameter("--mark-in-place and --fuse-dedup both mark duplicates; pass one")
    trim_plan = None
    buffer_bytes = 0
    if trim_in_task:
        from paralome.binaries import resolve_tool
        from paralome.buffer import parse_bytes
        from paralome.fastq.fastp import fastp_trimming_args
        from paralome.stream import FUSED_FASTP_THREADS

        try:
            buffer_bytes = parse_bytes(fused_buffer)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--fused-buffer") from exc

        trimming = dict(
            adapter_fasta=trim_adapter_fasta,
            # The run's own sliding-window settings; a fused task trimmed with the defaults
            # whatever the pipeline was given.
            cut_right=trim_cut_right,
            cut_right_window_size=trim_cut_right_window_size,
            cut_right_mean_quality=trim_cut_right_mean_quality,
            length_required=trim_length_required,
            extra_fastp_options=trim_extra or None,
            keep_files=False,
        )
        base = fastp_trimming_args(paired=False, **trimming)
        paired_only = [a for a in fastp_trimming_args(paired=True, **trimming) if a not in base]
        trim_plan = {
            "fastp_exe": resolve_tool("fastp") or "fastp",
            "workdir": str(trim_in_task),
            "json_dir": Path(trim_in_task) / "fastp_json",
            "args": base,
            "paired_args": paired_only,
            "threads": int(trim_threads) if trim_threads else FUSED_FASTP_THREADS,
        }
    dedup_plan = None
    if fuse_dedup:
        dedup_plan = {"workdir": str(fuse_dedup), "remove_duplicates": False, "clip_overlap": bool(clip_overlap)}
    return trim_plan, dedup_plan, buffer_bytes


def _report(summary, *, backend: str, out: Path, trim_in_task: Optional[Path], aligner: str) -> None:
    print(f"{aligner} alignment workdir: {summary['workdir']}")
    print(f"Samples: {summary['samples']}")
    if summary.get("backend") == "slurm":
        print(f"Requested threads: {summary.get('requested_threads')}")
        print(f"Threads per sample: {summary.get('threads_per_sample')}")
        print(f"Array concurrency: {summary.get('array_concurrency', summary.get('jobs'))}")
    else:
        print(f"Threads per sample: {summary.get('threads_per_sample')}")
    print(f"Aligned manifest: {summary['aligned_manifest']}")
    if summary.get("submitted"):
        # Write_star_slurm says each job as it is submitted; said
        # again here, every SLURM run printed both lines twice.
        print("Job ID: " + ",".join(str(value) for value in summary.get("submitted_job_ids", [])))
    else:
        print("Submitted: no")
    if backend == "local":
        # What the SLURM finalizer does after the array. A local
        # alignment left no verification, no mapping table and no kept STAR
        # logs, so nothing downstream could take it as finished.
        print("")
        if not run_align_verify(out).complete:
            raise typer.Exit(code=1)
        if trim_in_task:
            # What star_finalize.sbatch does on SLURM: the fastp reports the
            # tasks wrote become the FASTQ QC stage's summary.
            from paralome.commands.fastq import summarize_fastq_qc_if_ready

            summarize_fastq_qc_if_ready(Path(trim_in_task))


@align_app.command("star")
def align_star_command(
    samples: Path = typer.Option(..., "--samples", help="FASTQ sample manifest CSV"),
    reference: Path = typer.Option(..., "--reference", help="STAR genomeDir path"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="Optional GTF passed to STAR"),
    out: Path = typer.Option(..., "--out", help="Alignment output work directory"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(8, "--threads", help="Total STAR CPU budget; divided across samples up to STAR cap"),
    jobs: int = typer.Option(0, "--jobs", help="Maximum concurrent SLURM array tasks; default all samples"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM array"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing BAM outputs"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: Optional[str] = typer.Option(None, "--mem", help="Memory per sample: the SLURM request, and on both backends what STAR's BAM sort is limited to. Default: 32G; on SLURM, when duplicates are marked in the task (--fuse-dedup, --mark-in-place), the genome index's size plus STAR's working memory, the samtools sorts and markdup, and the --fused-buffer buffers, with a 15% margin, never under 32G"),
    star_read_files_command: str = typer.Option("zcat", "--star-read-files-command", help="STAR --readFilesCommand"),
    star_out_reads_unmapped: str = typer.Option("None", "--star-out-reads-unmapped", help="STAR --outReadsUnmapped"),
    star_out_filter_multimap_nmax: int = typer.Option(2, "--star-out-filter-multimap-nmax", help="STAR --outFilterMultimapNmax"),
    star_out_filter_mismatch_nover_read_lmax: float = typer.Option(0.04, "--star-out-filter-mismatch-nover-read-lmax", help="STAR --outFilterMismatchNoverReadLmax; DNA default 0.04, RNA default 0.06"),
    star_out_filter_mismatch_nmax: int = typer.Option(999, "--star-out-filter-mismatch-nmax", hidden=True, help="STAR --outFilterMismatchNmax; 999, switched off (the fraction alone limits mismatches)"),
    star_out_filter_score_min_over_lread: float = typer.Option(0.66, "--star-out-filter-score-min-over-lread", help="STAR --outFilterScoreMinOverLread; with the option below, decides 'unmapped: too short'"),
    star_out_filter_match_nmin_over_lread: float = typer.Option(0.66, "--star-out-filter-match-nmin-over-lread", help="STAR --outFilterMatchNminOverLread"),
    star_loose: bool = typer.Option(False, "--loose", help="Record that this run's filters were deliberately opened"),
    star_extra: str = typer.Option("", "--star-extra", "--extra-star-options", help="Additional STAR arguments as a single quoted string"),
    unspliced_mates_gap: Optional[int] = typer.Option(None, "--unspliced-mates-gap", help="STAR in DNA mode (the DNA and ATAC pipelines set it): --alignIntronMax 1 and --alignMatesGapMax this many bases; no GTF is given to STAR. A BAM aligned before DNA mode is kept as it is."),
    mark_in_place: bool = typer.Option(False, "--mark-in-place", help="With --trim-in-task and no deduplication stage (RNA): mark duplicates inside the alignment task and write them into bam/<sample>.sorted.bam, the path quantification already reads. Duplicates are marked, never removed."),
    trim_in_task: Optional[Path] = typer.Option(None, "--trim-in-task", help="FASTQ QC work directory: run fastp inside each alignment task, streaming the trimmed reads to STAR through FIFOs so they are never written as files. The manifest must then name the RAW FASTQs."),
    fuse_dedup: Optional[Path] = typer.Option(None, "--fuse-dedup", help="Duplicate marking work directory: pipe STAR's unsorted output straight into duplicate marking, so the aligned BAM is never written and STAR does not sort. Complete the stage afterwards with `paralome-worker dedup adopt`."),
    clip_overlap: bool = typer.Option(False, "--clip-overlap", help="With --fuse-dedup (DNA): soft-clip the overlap of a pair's mates after duplicate marking (bamUtil clipOverlap), so a fragment base is counted once"),
    trim_adapter_fasta: Optional[Path] = typer.Option(None, "--trim-adapter-fasta", help="With --trim-in-task: fastp --adapter_fasta"),
    trim_length_required: Optional[int] = typer.Option(None, "--trim-length-required", help="With --trim-in-task: fastp --length_required"),
    trim_cut_right: bool = typer.Option(True, "--trim-cut-right/--no-trim-cut-right", help="With --trim-in-task: fastp --cut_right (the pipeline's --cut-right)"),
    trim_cut_right_window_size: Optional[int] = typer.Option(None, "--trim-cut-right-window-size", help="With --trim-in-task: fastp --cut_right_window_size"),
    trim_cut_right_mean_quality: Optional[int] = typer.Option(None, "--trim-cut-right-mean-quality", help="With --trim-in-task: fastp --cut_right_mean_quality"),
    trim_extra: str = typer.Option("", "--trim-extra", help="With --trim-in-task: additional fastp arguments as a single quoted string"),
    fused_buffer: str = typer.Option("8G", "--fused-buffer", help="With --trim-in-task: RAM each of the two read streams may buffer between fastp and STAR. A kernel pipe is 64 KB, STAR consumes mates in lockstep, and fastp's two writers run ahead of each other -- skew past the buffer deadlocks them. Added to the stage's memory request automatically."),
    trim_threads: Optional[int] = typer.Option(None, "--trim-threads", help="With --trim-in-task: fastp's threads inside each task (default 8; fastp's own cap is 16). The rest of the task's CPUs go to STAR."),
):
    """Align FASTQs with STAR and produce a BAM sample manifest."""
    backend = backend.lower().strip()
    trim_plan, dedup_plan, buffer_bytes = _fusion_plans(
        mark_in_place=mark_in_place, trim_in_task=trim_in_task, fuse_dedup=fuse_dedup,
        trim_adapter_fasta=trim_adapter_fasta, trim_length_required=trim_length_required, trim_extra=trim_extra,
        fused_buffer=fused_buffer, trim_threads=trim_threads, clip_overlap=clip_overlap, trim_cut_right=trim_cut_right,
        trim_cut_right_window_size=trim_cut_right_window_size, trim_cut_right_mean_quality=trim_cut_right_mean_quality,
    )
    fused = bool(trim_in_task or fuse_dedup or mark_in_place)
    if backend == "local" and fused:
        from paralome.align.star import run_star_fused_local

        summary = run_star_fused_local(
            samples_csv=samples,
            out=out,
            reference=reference,
            gtf=gtf,
            threads=threads,
            jobs=jobs,
            overwrite=overwrite,
            mem=mem,
            star_extra=star_extra,
            unspliced_mates_gap=unspliced_mates_gap,
            read_files_command=star_read_files_command,
            out_reads_unmapped=star_out_reads_unmapped,
            out_filter_multimap_nmax=star_out_filter_multimap_nmax,
            out_filter_mismatch_nover_read_lmax=star_out_filter_mismatch_nover_read_lmax,
            out_filter_mismatch_nmax=star_out_filter_mismatch_nmax,
            out_filter_score_min_over_lread=star_out_filter_score_min_over_lread,
            out_filter_match_nmin_over_lread=star_out_filter_match_nmin_over_lread,
            loose=star_loose,
            trim_in_task=trim_plan,
            fuse_dedup=dedup_plan,
            mark_in_place=mark_in_place,
            fused_buffer_bytes=buffer_bytes,
        )
    elif backend == "local":
        summary = run_star_local(
            samples_csv=samples,
            out=out,
            reference=reference,
            gtf=gtf,
            threads=threads,
            overwrite=overwrite,
            mem=mem or "32G",
            star_extra=star_extra,
            unspliced_mates_gap=unspliced_mates_gap,
            read_files_command=star_read_files_command,
            out_reads_unmapped=star_out_reads_unmapped,
            out_filter_multimap_nmax=star_out_filter_multimap_nmax,
            out_filter_mismatch_nover_read_lmax=star_out_filter_mismatch_nover_read_lmax,
            out_filter_mismatch_nmax=star_out_filter_mismatch_nmax,
            out_filter_score_min_over_lread=star_out_filter_score_min_over_lread,
            out_filter_match_nmin_over_lread=star_out_filter_match_nmin_over_lread,
            loose=star_loose,
        )
    elif backend == "slurm":
        summary = write_star_slurm(
            samples_csv=samples,
            out=out,
            reference=reference,
            gtf=gtf,
            threads=threads,
            jobs=jobs,
            submit=submit,
            overwrite=overwrite,
            partition=partition,
            mem=mem,
            star_extra=star_extra,
            unspliced_mates_gap=unspliced_mates_gap,
            read_files_command=star_read_files_command,
            out_reads_unmapped=star_out_reads_unmapped,
            out_filter_multimap_nmax=star_out_filter_multimap_nmax,
            out_filter_mismatch_nover_read_lmax=star_out_filter_mismatch_nover_read_lmax,
            out_filter_mismatch_nmax=star_out_filter_mismatch_nmax,
            out_filter_score_min_over_lread=star_out_filter_score_min_over_lread,
            out_filter_match_nmin_over_lread=star_out_filter_match_nmin_over_lread,
            loose=star_loose,
            trim_in_task=trim_plan,
            fuse_dedup=dedup_plan,
            mark_in_place=mark_in_place,
            fused_buffer_bytes=buffer_bytes,
        )
    else:
        raise typer.BadParameter("backend must be local or slurm")

    _report(summary, backend=backend, out=out, trim_in_task=trim_in_task, aligner="STAR")


@align_app.command("bwa")
def align_bwa_command(
    samples: Path = typer.Option(..., "--samples", help="FASTQ sample manifest CSV"),
    reference: Path = typer.Option(..., "--reference", help="The genome's FASTA; the bwa-mem2 index beside it (<folder>/bwa-mem2/) is built on the first BWA run"),
    out: Path = typer.Option(..., "--out", help="Alignment output work directory"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(8, "--threads", help="Total CPU budget; divided across samples"),
    jobs: int = typer.Option(0, "--jobs", help="Maximum concurrent SLURM array tasks; default all samples"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM array (after an index job when the index is not built yet)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing BAM outputs"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: Optional[str] = typer.Option(None, "--mem", help="Memory per sample: the SLURM request, and what samtools sort is limited to. Default 32G"),
    bwa_extra: str = typer.Option("", "--bwa-extra", help="Additional bwa-mem2 mem arguments as a single quoted string"),
    mark_in_place: bool = typer.Option(False, "--mark-in-place", help="As for `align star`: mark duplicates in the task into bam/<sample>.sorted.bam (RNA)"),
    trim_in_task: Optional[Path] = typer.Option(None, "--trim-in-task", help="As for `align star`: fastp in each task, streaming into bwa-mem2 through FIFOs"),
    fuse_dedup: Optional[Path] = typer.Option(None, "--fuse-dedup", help="As for `align star`: bwa-mem2's output straight into duplicate marking"),
    clip_overlap: bool = typer.Option(False, "--clip-overlap", help="With --fuse-dedup (DNA): soft-clip the overlap of a pair's mates after duplicate marking (bamUtil clipOverlap), so a fragment base is counted once"),
    trim_adapter_fasta: Optional[Path] = typer.Option(None, "--trim-adapter-fasta", help="With --trim-in-task: fastp --adapter_fasta"),
    trim_length_required: Optional[int] = typer.Option(None, "--trim-length-required", help="With --trim-in-task: fastp --length_required"),
    trim_cut_right: bool = typer.Option(True, "--trim-cut-right/--no-trim-cut-right", help="With --trim-in-task: fastp --cut_right (the pipeline's --cut-right)"),
    trim_cut_right_window_size: Optional[int] = typer.Option(None, "--trim-cut-right-window-size", help="With --trim-in-task: fastp --cut_right_window_size"),
    trim_cut_right_mean_quality: Optional[int] = typer.Option(None, "--trim-cut-right-mean-quality", help="With --trim-in-task: fastp --cut_right_mean_quality"),
    trim_extra: str = typer.Option("", "--trim-extra", help="With --trim-in-task: additional fastp arguments as a single quoted string"),
    fused_buffer: str = typer.Option("8G", "--fused-buffer", help="With --trim-in-task: RAM each read stream may buffer between fastp and bwa-mem2"),
    trim_threads: Optional[int] = typer.Option(None, "--trim-threads", help="With --trim-in-task: fastp's threads inside each task (default 8)"),
    mismatch_fraction: Optional[float] = typer.Option(None, "--mismatch-fraction", help="Pair filter: mismatches, both mates together, at most this fraction of their length (the run's STAR --outFilterMismatchNoverReadLmax). Without it, bwa-mem2's output is not filtered"),
    match_fraction: float = typer.Option(0.66, "--match-fraction", help="Pair filter: aligned bases at least this fraction of the pair's length (STAR --outFilterMatchNminOverLread)"),
    multimap_nmax: int = typer.Option(2, "--multimap-nmax", help="Pair filter: at most this many equally good places for the pair (STAR --outFilterMultimapNmax)"),
    mates_gap: Optional[int] = typer.Option(None, "--mates-gap", help="Pair filter: mates at most this far apart (STAR's DNA-mode --alignMatesGapMax); none for RNA"),
):
    """Align FASTQs with bwa-mem2 (--aligner bwa) and produce the same BAMs and sample manifest as STAR."""
    from paralome.align.star import run_star_fused_local

    backend = backend.lower().strip()
    trim_plan, dedup_plan, buffer_bytes = _fusion_plans(
        mark_in_place=mark_in_place, trim_in_task=trim_in_task, fuse_dedup=fuse_dedup,
        trim_adapter_fasta=trim_adapter_fasta, trim_length_required=trim_length_required, trim_extra=trim_extra,
        fused_buffer=fused_buffer, trim_threads=trim_threads, clip_overlap=clip_overlap, trim_cut_right=trim_cut_right,
        trim_cut_right_window_size=trim_cut_right_window_size, trim_cut_right_mean_quality=trim_cut_right_mean_quality,
    )
    options = dict(samples_csv=samples, out=out, reference=reference, threads=threads, overwrite=overwrite,
                   mem=mem, trim_in_task=trim_plan, fuse_dedup=dedup_plan, mark_in_place=mark_in_place,
                   fused_buffer_bytes=buffer_bytes, aligner="bwa", bwa_extra=bwa_extra)
    if mismatch_fraction is not None:
        from paralome.align.bwa import filter_settings

        options["bwa_filter"] = filter_settings(mismatch_fraction=mismatch_fraction, match_fraction=match_fraction,
                                                multimap_nmax=multimap_nmax, mates_gap=mates_gap)
    if backend == "local":
        # One task script for both backends, run here a sample at a time (the index built first).
        summary = run_star_fused_local(jobs=jobs, **options)
    elif backend == "slurm":
        summary = write_star_slurm(jobs=jobs, submit=submit, partition=partition, **options)
    else:
        raise typer.BadParameter("backend must be local or slurm")
    _report(summary, backend=backend, out=out, trim_in_task=trim_in_task, aligner="BWA")


@align_app.command("verify")
def align_verify_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="STAR alignment work directory"),
    fastq_qc_workdir: Optional[Path] = typer.Option(None, "--fastq-qc-workdir", help="A fused run's FASTQ QC work directory: summarize the fastp reports the alignment tasks wrote there. Every assay, with or without a deduplication stage."),
):
    """Verify a finished STAR alignment stage.

    Checks each sample's BAM and index, writes the mapping table and keeps each
    sample's STAR Log.final.out under logs/.
    """
    # Development history moved out of the help text. This is the
    # finalizer the supervisor and star_finalize.sbatch call after every SLURM
    # alignment; option spelling (incl. --out) matches `dedup verify`.
    run_align_verify(workdir)
    if fastq_qc_workdir:
        # The fused tasks write fastp's JSON and HTML into the FASTQ
        # QC stage's own directories as they finish. This is the first moment
        # every task is done, so it is where the stage's summary belongs.
        # ..._if_ready, not the raising form. A short-handed read of
        # the report directory over NFS must not fail the alignment stage after
        # every task has finished its work. See summarize_fastq_qc_if_ready.
        from paralome.commands.fastq import summarize_fastq_qc_if_ready

        summarize_fastq_qc_if_ready(Path(fastq_qc_workdir))
