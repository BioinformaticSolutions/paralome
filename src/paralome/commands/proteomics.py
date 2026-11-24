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

"""`paralome proteomics`: proteins, and paired phosphopeptides, from an exporter's table.

The stages and what each decides are in
``paralome.massspec.stages``; this is the command line only.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from paralome.stats import BLOCK_HELP, FACTOR_HELP, STRATIFY_HELP

_DESIGN = "The comparison"


def proteomics_command(
    proteins: Path = typer.Option(..., "--proteins", help="The exporter's protein table (Proteome Discoverer, MaxQuant proteinGroups.txt, FragPipe combined_protein.tsv, DIA-NN pg_matrix.tsv)"),
    phospho: Optional[Path] = typer.Option(None, "--phospho", help="The paired phosphopeptide table from the same samples"),
    samples: Path = typer.Option(..., "--samples", help="Sample sheet: column (abundance column or the sample name in it), sample_id, then metadata columns"),
    project: Path = typer.Option(..., "--project", help="This analysis's folder; the import goes in the parent project's data/ folder"),
    genome: str = typer.Option(..., "--genome", help="Configured genome; its GTF gives each protein its Ensembl gene"),
    statistic: str = typer.Option(..., "--statistic", rich_help_panel=_DESIGN, help="Required: t, paired-t, lm, anova, or several comma-separated; add panel to fit the design's elastic-net panels (anova,panel). There is no default"),
    factor: List[str] = typer.Option([], "--factor", rich_help_panel=_DESIGN, help=FACTOR_HELP),
    pair_by: Optional[str] = typer.Option(None, "--pair-by", rich_help_panel=_DESIGN, help="Sheet column pairing samples, for paired-t"),
    covariate: Optional[str] = typer.Option(None, "--covariate", rich_help_panel=_DESIGN, help="Numeric sheet column, for lm"),
    block_by: Optional[str] = typer.Option(None, "--block-by", rich_help_panel=_DESIGN, help=BLOCK_HELP),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", rich_help_panel=_DESIGN, help=STRATIFY_HELP),
    alpha: str = typer.Option("P0.05", "--alpha", rich_help_panel=_DESIGN, help="Significance metric and cutoff, e.g. Q0.05 or P0.10"),
    backend: str = typer.Option("local", "--backend", help="Where Evidome runs: local or slurm"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition for Evidome"),
    threads: int = typer.Option(0, "--threads", min=0, help="CPUs for --statistic ...,panel. --backend slurm: each panel job asks for this many (default: resources.threads in your config, 8 as shipped); --backend local: worker processes (default: every CPU this process may use)"),
) -> None:
    """Proteomics, with paired phosphoproteomics: import, identity, scaling, QC, statistics, Evidome.

    A missing value of a protein (or phosphopeptide form) is set to half the
    feature's lowest measured value. With
    --phospho, three questions: did the protein change (protein_change), did
    the phosphopeptide change (phospho_change), and -- with t or paired-t --
    did it change differently from its protein (phospho_vs_protein_change, the
    2x2 ANOVA interaction); a combined table reads them together, and one
    Evidome run takes all three.
    """
    from paralome.commands.atac import _parse_alpha
    from paralome.commands.pipeline import _require_a_genome
    from paralome.errors import ParalomeError
    from paralome.massspec.stages import run_proteomics
    from paralome.pipeline.resource_catalog import ResourceResolutionError, resolve_genome_resources

    if backend not in ("local", "slurm"):
        raise typer.BadParameter("--backend must be local or slurm")
    metric, value = _parse_alpha(alpha)
    _require_a_genome(analysis="proteomics", genome=genome, explicit_paths_cover_it=False)
    try:
        gtf = resolve_genome_resources(genome=genome, required=("gtf",)).path("gtf")
    except ResourceResolutionError as exc:
        raise ParalomeError(406, str(exc)) from exc
    run_proteomics(
        proteins=proteins, phospho=phospho, samples=samples, project=project, gtf=str(gtf),
        statistic=statistic, factors=factor, pair_by=pair_by, covariate=covariate, block_by=block_by, stratify_by=stratify_by,
        alpha_metric=metric.lower(), alpha_value=value, backend=backend, partition=partition, threads=threads,
    )
