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

"""`paralome-worker trajectories run`: the trajectories stage -- see
paralome.singlecell.trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

trajectories_app = typer.Typer(help="Single-cell trajectories (Monocle3 or Slingshot) as a pipeline stage")


@trajectories_app.command("run")
def trajectories_run(
    extraction: Path = typer.Option(..., "--extraction", help="The object's extraction folder (<parent>/data/extraction)"),
    samples: Path = typer.Option(..., "--samples", help="The run's pseudobulk sample manifest (its replicates and their columns)"),
    out: Path = typer.Option(..., "--out", help="Stage directory (<run>/trajectories)"),
    prefix: str = typer.Option(..., "--prefix", help="Project prefix: names the zip and the statistics files"),
    root: str = typer.Option(..., "--root", help="<column>:<value>: the cell-metadata column whose clusters the lineages are fitted on, and the cluster they start from"),
    replicate_column: str = typer.Option(..., "--replicate-column", help="The cell-metadata column naming each cell's biological sample (the run's)"),
    statistic: str = typer.Option(..., "--statistic", help="The run's --statistic: the composition's, each entry at the replicate level"),
    modality: str = typer.Option("scRNA", "--modality", help="scRNA"),
    bins: int = typer.Option(5, "--bins", min=2, help="Composition segments per lineage, equal numbers of its cells by pseudotime"),
    model: Optional[str] = typer.Option(None, "--model", help="The run's --trajectory-model: ls (the default) or monocle3, the spline's df after a colon (ls:df=4)"),
    lineages: Optional[str] = typer.Option(None, "--lineages", help="The run's --trajectory-lineages: fate (the default), exit or route"),
    fit_cells: Optional[str] = typer.Option(None, "--fit-cells", help="How the lineages are fitted (the run's --trajectory-fit-cells). Slingshot: <n>:random, <n>:centroid, or cells=<n>,dispersion=random|centroid,stretch=<s>,extend=y|n|pc1,reduction=<name>,df=<n>,shrink=<0-1>. Monocle3: reduction=<name>,ncenter=<n>,minimal_branch_len=<n>,use_partition=y|n,close_loop=y|n,geodesic_distance_ratio=<x>,euclidean_distance_ratio=<x>,orthogonal_proj_tip=y|n,prune_graph=y|n,l1_gamma=<x>,l1_sigma=<x>"),
    method: Optional[str] = typer.Option(None, "--method", help="The run's --trajectory-method: monocle3 or slingshot (default: the method the fit was made with)"),
    dims: Optional[int] = typer.Option(None, "--dims", min=2, help="Fit on the reduction's first n dimensions (the run's --trajectory-dims)"),
    factor: List[str] = typer.Option([], "--factor", help="The run's --factor"),
    pair_by: Optional[str] = typer.Option(None, "--pair-by"),
    covariate: Optional[str] = typer.Option(None, "--covariate"),
    covariate_feature: Optional[str] = typer.Option(None, "--covariate-feature"),
    remove: List[str] = typer.Option([], "--remove", help="The run's --remove"),
    genome: Optional[str] = typer.Option(None, "--genome", help="The run's genome: its GTF names the genes, as in the run's pseudobulk"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="Explicit GTF, overriding --genome's"),
    alpha: str = typer.Option("P0.05", "--alpha"),
    rscript: str = typer.Option("Rscript", "--rscript"),
    threads: int = typer.Option(1, "--threads", min=1),
    fit: Optional[Path] = typer.Option(None, "--fit", help="The fit's folder (the singlecell_rna stage fits the lineages there; default: <--out>/fit)"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the run's omicview/trajectories folder (default: the stage folder's results/)"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    from paralome.singlecell.trajectories import run_trajectories

    resolved_gtf = gtf
    if genome:
        from paralome.pipeline.resource_catalog import resolve_genome_resources

        resolved_gtf = resolve_genome_resources(genome=genome, overrides={"gtf": gtf}).path("gtf")
    run_trajectories(
        extraction, samples, out, root=root, prefix=prefix, modality=modality, replicate_column=replicate_column,
        statistic=statistic, factors=list(factor), pair_by=pair_by, covariate=covariate,
        covariate_feature=covariate_feature, remove=list(remove), gtf=Path(resolved_gtf) if resolved_gtf else None,
        bins=bins, fit_cells=fit_cells, dims=dims, method=method, alpha=alpha, jobs=threads, rscript=rscript, zip_dir=zip_dir, overwrite=overwrite,
        fit_dir=fit, model=model, lineage_set=lineages,
        echo=lambda text: print(f"[paralome] {text}", flush=True),
    )
