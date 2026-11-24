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

from paralome.dedup.markdup import print_result, print_verify, run_dedup_stage, verify_dedup_stage


dedup_app = typer.Typer(help="Duplicate marking commands")


@dedup_app.command("mark")
def dedup_mark_command(
    samples: Path = typer.Option(..., "--samples", help="BAM sample manifest CSV"),
    out: Path = typer.Option(..., "--out", help="Duplicate marking work directory"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(4, "--threads", help="samtools threads per sample/job"),
    jobs: int = typer.Option(0, "--jobs", help="Maximum concurrent SLURM array tasks; default all samples"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM array"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing duplicate-marked BAMs"),
    remove_duplicates: bool = typer.Option(False, "--remove-duplicates", help="Remove duplicates with samtools markdup -r instead of only marking them"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write plan/checkpoint but do not run samtools"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("16G", "--mem", help="SLURM memory per sample"),
    manifest_profile: str = typer.Option("family", "--manifest-profile", help="Manifest schema: family or general"),
    clip_overlap: bool = typer.Option(False, "--clip-overlap", help="DNA: soft-clip the overlap of a pair's mates after marking (bamUtil clipOverlap), so a fragment base is counted once"),
):
    """Mark duplicate reads with samtools markdup and emit a BAM manifest."""
    result = run_dedup_stage(
        samples_csv=samples,
        workdir=out,
        backend=backend,
        threads=threads,
        jobs=jobs,
        remove_duplicates=remove_duplicates,
        overwrite=overwrite,
        submit=submit,
        dry_run=dry_run,
        clip_overlap=clip_overlap,
        partition=partition,
        mem=mem,
        manifest_profile=manifest_profile,
    )
    print_result(result)


@dedup_app.command("verify")
def dedup_verify_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="Duplicate marking work directory"),
    threads: int = typer.Option(4, "--threads", help="Parallel file-check threads"),
):
    """Verify a duplicate-marking workflow stage."""
    report = verify_dedup_stage(workdir, threads=threads)
    print_verify(report)


@dedup_app.command("status")
def dedup_status_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="Duplicate marking work directory"),
    threads: int = typer.Option(4, "--threads", help="Parallel file-check threads"),
):
    """Print concise duplicate-marking status."""
    report = verify_dedup_stage(workdir, threads=threads)
    metrics = report.get("metrics") or {}
    print("Duplicate marking status")
    print("========================")
    print(f"Workdir             : {workdir}")
    print(f"Complete            : {report.get('complete')}")
    print(f"Samples             : {report.get('completed_samples')}/{report.get('samples')}")
    rate = metrics.get("duplicate_rate")
    print(f"Duplicate rate      : {rate * 100:.3f}%" if isinstance(rate, (int, float)) else "Duplicate rate      : .")
    print(f"Dedup manifest      : {report.get('dedup_manifest', {}).get('path')}")


@dedup_app.command("summarize")
def dedup_summarize_command(
    workdir: Path = typer.Option(..., "--workdir", help="Duplicate marking work directory"),
):
    """Write the stage's summary (results/dedup_summary.json) from the duplicate-marked BAMs that are there.

    A SLURM run submits this after its tasks; run it by hand for a project whose summary is missing, has no
    read counts, or was written by an older Paralome (whose counts were BAM records, not primary reads).
    Nothing is marked again.
    """
    from paralome.dedup.markdup import summarize_dedup_stage

    result = summarize_dedup_stage(workdir)
    print(f"Summary  : {result['summary']}")
    print(f"Samples  : {result.get('completed_samples')}/{result.get('samples')}")
    for m in result.get("sample_metrics", []):
        rate = m.get("duplicate_rate")
        print(f"  {m['sample_id']}: mapped reads {m.get('total_reads')}  duplicates {m.get('duplicate_reads')}  "
              f"rate {('%.2f%%' % (100 * rate)) if isinstance(rate, (int, float)) else '.'}")


@dedup_app.command("adopt")
def dedup_adopt_command(
    workdir: Path = typer.Option(..., "--workdir", help="Duplicate marking work directory"),
    samples: Path = typer.Option(..., "--samples", help="Sample manifest naming the samples a fused alignment task marked"),
    remove_duplicates: bool = typer.Option(False, "--remove-duplicates/--keep-duplicates", help="What the fused alignment task was told to do; recorded, and decides whether the duplicate rate is measurable"),
    alignment_workdir: Optional[Path] = typer.Option(None, "--alignment-workdir", help="The alignment work directory the fused task ran in; recorded for provenance"),
    fastq_qc_workdir: Optional[Path] = typer.Option(None, "--fastq-qc-workdir", help="Also summarize the FASTQ QC stage from the fastp reports the fused task wrote there"),
):
    """Complete duplicate marking from BAMs a fused alignment task already wrote.

    With `align star --fuse-dedup` the marking happens inside the
    alignment task, so this stage has nothing to compute -- only its own plan,
    summary and checkpoint to write, which every downstream stage and report
    looks for. Nothing is re-marked; a missing BAM is reported as an
    incomplete stage, which is the truthful answer.
    """
    from paralome.dedup.markdup import adopt_fused_results

    result = adopt_fused_results(
        workdir,
        samples_csv=samples,
        remove_duplicates=remove_duplicates,
        alignment_workdir=alignment_workdir,
    )
    print(f"Duplicate marking workdir: {result['workdir']}")
    print("Adopted from             : fused alignment task")
    print(f"Samples                  : {result['completed_samples']}/{result['samples']}")
    print(f"Dedup manifest           : {result['dedup_manifest']}")
    rate = result.get("duplicate_rate")
    print(f"Duplicate rate           : {rate * 100:.3f}%" if isinstance(rate, (int, float))
          else "Duplicate rate           : not measurable (duplicates were removed in the fused task)")
    print(f"Complete                 : {result['complete']}")
    if fastq_qc_workdir:
        # The fused task wrote fastp's JSON and HTML into the FASTQ QC stage's
        # own directories; this is the stage's usual summarizer, run now that
        # they are there.
        from paralome.commands.fastq import summarize_fastq_qc_if_ready

        summarize_fastq_qc_if_ready(Path(fastq_qc_workdir))
    if not result["complete"]:
        raise typer.Exit(code=1)
