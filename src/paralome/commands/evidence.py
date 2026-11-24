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

"""``paralome-worker evidence run``: the functional_evidence stage driver."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

evidence_app = typer.Typer(help="Functional evidence (Evidome) as a pipeline stage")


@evidence_app.command("run")
def evidence_run(
    statistics: Optional[Path] = typer.Option(
        None, "--statistics", help="One table for Evidome (DNA's gene burden). Give this or --differential-checkpoint"),
    differential_checkpoint: Optional[Path] = typer.Option(
        None, "--differential-checkpoint",
        help="A differential stage's checkpoint: Evidome runs once per statistic folder (per stratum) it records"),
    trajectory_checkpoint: Optional[Path] = typer.Option(
        None, "--trajectory-checkpoint",
        help="The trajectories stage's segment statistics (its differential's checkpoint): one Evidome run "
             "per statistic folder per segment, besides the differential's"),
    out: Path = typer.Option(..., "--out", help="Stage directory (functional_evidence)"),
    prefix: str = typer.Option(..., "--prefix", help="Project prefix: names Evidome's output folder and files"),
    backend: str = typer.Option("local", "--backend", help="local, or slurm to have Evidome submit its own jobs"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition for Evidome's jobs"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the project's omicview/<stage> folder (default: the stage folder's results/)"),
    alpha: str = typer.Option("P0.05", "--alpha", help="Passed to Evidome as its --alpha: the p it selects genes by"),
) -> None:
    from paralome.commands.atac import _parse_alpha
    from paralome.evidence import run_functional_evidence

    if (statistics is None) == (differential_checkpoint is None):
        raise typer.BadParameter("give exactly one of --statistics or --differential-checkpoint")
    checkpoint = run_functional_evidence(
        statistics, out, prefix, differential_checkpoint=differential_checkpoint,
        backend=backend, partition=partition, zip_dir=zip_dir,
        trajectory_checkpoint=trajectory_checkpoint, alpha=_parse_alpha(alpha)[1],
    )
    # Skipped is not an error in the analysis (Evidome is not set up here); failed is.
    raise typer.Exit(1 if checkpoint["status"] == "failed" else 0)


@evidence_app.command("place")
def evidence_place(
    out: Path = typer.Option(..., "--out", help="Stage directory (functional_evidence)"),
    zip_dir: Path = typer.Option(..., "--zip-dir", help="The project's omicview/functional_evidence folder"),
) -> None:
    """Move each finished Evidome archive of the stage into --zip-dir (the job submitted after Evidome's)."""
    from paralome.evidence import place_archives

    state = place_archives(out, zip_dir)
    typer.echo(f"functional evidence: {state.get('status')}; archives in {zip_dir}")
