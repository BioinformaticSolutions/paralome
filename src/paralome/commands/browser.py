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

"""``paralome-worker output genotype`` and ``output statistics``: the genotype_output (DNA) and
statistics_output (every other arm) stages."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

browser_app = typer.Typer(help="The genotype_output and statistics_output stages: every arm's export for OmicView")


@browser_app.command("genotype")
def browser_run(
    segregation: Path = typer.Option(..., "--segregation", help="The segregation stage's whole-genome table"),
    out: Path = typer.Option(..., "--out", help="Stage directory (genotype_output)"),
    prefix: str = typer.Option(..., "--prefix", help="Project prefix for the bundle's name"),
    burden: Optional[Path] = typer.Option(None, "--burden", help="gene_burden's results folder (its tables)"),
    filter_results: Optional[Path] = typer.Option(None, "--filter", help="The filtering stage's results folder"),
    qc_root: Optional[Path] = typer.Option(None, "--qc-root", help="The project folder: its FASTQ-to-alignment summaries (fastq_qc, star, dedup) go into the zip"),
    filter_profile: Optional[str] = typer.Option(None, "--filter-profile", help="The filter profile the browser's pipeline filter follows; default: the one the filtering stage ran first"),
    threads: int = typer.Option(1, "--threads", help="CPUs DuckDB builds the bundle on"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Build again even when the bundle answers this request"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the project's omicview/<stage> folder (default: the stage folder's results/)"),
    former_folder: Optional[Path] = typer.Option(None, "--former-folder", help="This stage's folder in an older project (<project>/browser_bundle), renamed in place when the new one is not there"),
) -> None:
    import sys

    from paralome.browser.arm import take_former_folder
    from paralome.browser.stage import STAGE, run_genotype_output

    echo = lambda text: print(text, flush=True, file=sys.stdout)  # noqa: E731
    take_former_folder(former_folder, out, STAGE, echo)
    run_genotype_output(
        segregation_table=segregation, workdir=out, prefix=prefix, burden_results=burden,
        filter_results=filter_results, filter_profile=filter_profile, threads=threads, overwrite=overwrite,
        zip_dir=zip_dir, echo=echo, qc_root=qc_root,
    )


# The statistics_output stage of every other arm -- see paralome.browser.arm.
@browser_app.command("statistics")
def browser_arm(
    arm: str = typer.Option(..., "--arm", help="The pipeline: RNA, ATAC, snATAC, scRNA or proteomics"),
    out: Path = typer.Option(..., "--out", help="Stage directory (statistics_output)"),
    prefix: str = typer.Option(..., "--prefix", help="Project prefix for the bundle's name"),
    results: list[Path] = typer.Option([], "--results", help="A folder holding statistic folders (repeatable)"),
    motifs: list[Path] = typer.Option([], "--motifs", help="A motif enrichment results folder (repeatable)"),
    samples_table: Optional[Path] = typer.Option(None, "--samples-table", help="The sample sheet or pseudobulk manifest whose columns describe the samples"),
    threads: int = typer.Option(1, "--threads", help="CPUs"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Build again even when the bundle answers this request"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the project's omicview/<stage> folder (default: the stage folder's results/)"),
    qc_root: Optional[Path] = typer.Option(None, "--qc-root", help="The project folder: its FASTQ-to-alignment summaries (fastq_qc, star, dedup, peak_quantification) go into the zip"),
    former_folder: Optional[Path] = typer.Option(None, "--former-folder", help="This stage's folder in an older project (<project>/browser_bundle), renamed in place when the new one is not there"),
) -> None:
    import sys

    from paralome.browser.arm import STAGE, run_statistics_output, take_former_folder

    echo = lambda text: print(text, flush=True, file=sys.stdout)  # noqa: E731
    take_former_folder(former_folder, out, STAGE, echo)
    # The statistics only; the ATAC tracks are the genome_tracks stage's again.
    run_statistics_output(
        arm=arm, workdir=out, prefix=prefix, results_roots=results, motif_roots=motifs,
        samples_table=samples_table, threads=threads, overwrite=overwrite, zip_dir=zip_dir, echo=echo,
        qc_root=qc_root,
    )


# The DNA arm's genome_tracks stage -- see paralome.variants.genome_tracks.
@browser_app.command("dna-genome-tracks")
def dna_genome_tracks(
    vcf: Path = typer.Option(..., "--vcf", help="The variant calling's VCF (every site is sliced)"),
    samples: Path = typer.Option(..., "--samples", help="The deduplication manifest: each sample's BAM, family and affected status"),
    out: Path = typer.Option(..., "--out", help="Stage directory (genome_tracks)"),
    prefix: str = typer.Option(..., "--prefix", help="Project prefix for the zip's name"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Registered genome whose FASTA the reads are shown against"),
    reference: Optional[Path] = typer.Option(None, "--reference", help="The FASTA the BAMs were aligned to (default: the --genome entry's)"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="GTF for the gene models drawn above the reads (default: the --genome entry's)"),
    threads: int = typer.Option(1, "--threads", help="CPUs: the BAMs sliced at once, each samtools on its share, and the reference written"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the project's omicview/genome_tracks folder (default: the stage folder's results/)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Build again even when the reads answer this request"),
) -> None:
    """The reads FreeBayes used at every called site, <prefix>_DNA_reads.zip, for OmicView's variant panel."""
    import sys

    from paralome.variants.genome_tracks import run_dna_genome_tracks

    run_dna_genome_tracks(vcf=vcf, samples=samples, workdir=out, prefix=prefix, genome=genome, reference=reference,
                          gtf=gtf, threads=threads, zip_dir=zip_dir, overwrite=overwrite,
                          echo=lambda text: print(text, flush=True, file=sys.stdout))
