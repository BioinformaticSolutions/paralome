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

"""`paralome-worker panels run`: one stratum's --statistic ...,panel, in the SLURM job `paralome olink` /
`paralome proteomics` submit with --backend slurm."""

from __future__ import annotations

from pathlib import Path

import typer

panels_app = typer.Typer(help="The --statistic panel jobs")


@panels_app.command("run")
def panels_run(
    plan: Path = typer.Option(..., "--plan", help="The panel_plan.json the command wrote"),
    threads: int = typer.Option(0, "--threads", min=0, help="Worker processes (0: every CPU this job may use)"),
) -> None:
    import sys

    from paralome.massspec.panel import run_panel_plan

    result = run_panel_plan(plan, threads=threads, echo=lambda text: print(text, flush=True, file=sys.stdout))
    print(f"[paralome] panels: {result['fitted']} fitted, {result['not_fitted']} not fitted -> {result['performance']}",
          flush=True)
