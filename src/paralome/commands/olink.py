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

"""`paralome olink`: Olink targeted protein panels from Olink's own plate exports.

The stages and what each decides are in ``paralome.olink.stages``; this is the command line
only.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from paralome.stats import BLOCK_HELP, FACTOR_HELP, STRATIFY_HELP

_DESIGN = "The comparison"


def olink_command(
    plate: List[Path] = typer.Option(..., "--plate", help="An Olink plate export (long format: NPX Signature CSV, .xlsx or .parquet). Repeatable, once per file; every other plate is bridged to the first by the sample controls they share"),
    samples: Path = typer.Option(..., "--samples", help="Sample sheet: PlateID, SampleID, sample_id, then the design's columns -- one sheet for every question of the plates. Wells given one sample_id are averaged; control wells are passed over; a sample with no value in a column the run uses (--factor, --block-by, --covariate, --pair-by) is not in the run"),
    project: Path = typer.Option(..., "--project", help="This analysis's folder; the plates' import goes in the parent project's data/ folder"),
    genome: str = typer.Option(..., "--genome", help="Configured genome; its GTF gives each assay its Ensembl gene"),
    statistic: str = typer.Option(..., "--statistic", rich_help_panel=_DESIGN, help="Required: t, paired-t, lm, anova, or several comma-separated; add panel to fit the design's elastic-net panels (anova,panel). There is no default"),
    factor: List[str] = typer.Option([], "--factor", rich_help_panel=_DESIGN, help=FACTOR_HELP),
    pair_by: Optional[str] = typer.Option(None, "--pair-by", rich_help_panel=_DESIGN, help="Sheet column pairing samples, for paired-t"),
    block_by: Optional[str] = typer.Option(None, "--block-by", rich_help_panel=_DESIGN, help=BLOCK_HELP),
    covariate: Optional[str] = typer.Option(None, "--covariate", rich_help_panel=_DESIGN, help="Numeric sheet column, for lm"),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", rich_help_panel=_DESIGN, help=STRATIFY_HELP),
    stratify_value: Optional[str] = typer.Option(None, "--stratify-value", rich_help_panel=_DESIGN, help="Run only these strata of --stratify-by (comma-separated; the value or its folder name), e.g. one experiment"),
    alpha: str = typer.Option("P0.05", "--alpha", rich_help_panel=_DESIGN, help="Significance metric and cutoff, e.g. Q0.05 or P0.10"),
    backend: str = typer.Option("local", "--backend", help="Where Evidome runs: local or slurm"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition for Evidome"),
    threads: int = typer.Option(0, "--threads", min=0, help="CPUs for --statistic ...,panel. --backend slurm: each panel job asks for this many (default: resources.threads in your config, 8 as shipped); --backend local: worker processes (default: every CPU this process may use)"),
) -> None:
    """Olink panels: import, plate bridging, identity, statistics, Evidome.

    The value is Olink's QuantifiedValue (pg/mL), tested on log2(x + 1). Each plate
    after the first is brought onto the first plate's scale, assay by assay, by the
    median ratio of the sample controls both ran. A value Olink flags below its limit
    of detection is missing, and set to half the assay's lowest measured value.
    """
    from paralome.commands.atac import _parse_alpha
    from paralome.commands.pipeline import _require_a_genome
    from paralome.errors import ParalomeError
    from paralome.olink.stages import run_olink
    from paralome.pipeline.resource_catalog import ResourceResolutionError, resolve_genome_resources

    if backend not in ("local", "slurm"):
        raise typer.BadParameter("--backend must be local or slurm")
    metric, value = _parse_alpha(alpha)
    _require_a_genome(analysis="olink", genome=genome, explicit_paths_cover_it=False)
    try:
        resources = resolve_genome_resources(genome=genome, required=("gtf",))
    except ResourceResolutionError as exc:
        raise ParalomeError(406, str(exc)) from exc
    gtf = resources.path("gtf")
    # The genome's Ensembl UniProt table, when registered, finds an assay named by no gene symbol.
    uniprot = resources.path("uniprot_table")
    run_olink(
        plates=plate, samples=samples, project=project, gtf=str(gtf), statistic=statistic, factors=factor,
        uniprot_table=str(uniprot) if uniprot else None,
        pair_by=pair_by, block_by=block_by, covariate=covariate, stratify_by=stratify_by, stratify_value=stratify_value,
        alpha_metric=metric.lower(), alpha_value=value, backend=backend, partition=partition, threads=threads,
    )
