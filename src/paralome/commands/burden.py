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

"""``paralome-worker burden run``: the gene_burden stage driver."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

burden_app = typer.Typer(help="Gene burden of a proband cohort against gnomAD, as a pipeline stage")


@burden_app.command("run")
def burden_run(
    vcf: Path = typer.Option(..., "--vcf", help="ANNOVAR-annotated VCF (the annotation stage's output)"),
    samples: Path = typer.Option(..., "--samples", help="Sample sheet: the affected samples are the probands"),
    out: Path = typer.Option(..., "--out", help="Stage directory (gene_burden)"),
    reference: Optional[Path] = typer.Option(None, "--reference", help="The genome's burden reference table; without one the stage is skipped"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Project prefix for the output files"),
    ancestry: str = typer.Option("auto", "--ancestry", help="auto (estimate per proband) or one gnomAD population for all"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Genome name, for the command that builds a missing reference"),
    species: Optional[str] = typer.Option(None, "--species", help="The genome's species; gnomAD describes humans only"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Compute again even when the result answers this request"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="Genome GTF: only its protein_coding genes are tested; default: the --genome entry's gtf"),
    threads: int = typer.Option(1, "--threads", help="Processes reading the VCF, one contig each (needs its tabix index); at most 16"),
    alpha: str = typer.Option("P0.05", "--alpha", help="The p Evidome selects genes by (its --alpha); the null selection probability is computed at it"),
) -> None:
    import sys

    from paralome.burden.stage import run_gene_burden
    from paralome.commands.atac import _parse_alpha

    if gtf is None and genome:
        gtf = genome_gtf(genome)
    run_gene_burden(
        vcf=vcf, samples=samples, workdir=out, reference=reference, prefix=prefix,
        ancestry=ancestry, overwrite=overwrite, genome=genome, species=species, gtf=gtf, threads=threads, alpha=_parse_alpha(alpha)[1],
        echo=lambda text: print(text, flush=True, file=sys.stdout),
    )


def genome_gtf(genome: str) -> Optional[Path]:
    """The genome entry's gtf, or None when it records none."""
    from paralome.pipeline.resource_catalog import select_genome_config

    _name, entry = select_genome_config(genome)
    value = entry.get("gtf")
    return Path(str(value)) if value else None
