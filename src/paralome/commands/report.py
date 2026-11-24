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

report_app = typer.Typer(help="Summarize Paralome results and pipeline outputs")


@report_app.command("summary")
def report_summary(
    workdir: Path = typer.Option(Path("."), "--workdir", "--project", help="Project folder: the one that holds project_state.json"),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Optional output JSON path"),
):
    """Write a project's recorded state (what `paralome status --project` shows) as JSON."""
    from paralome.reports.summary import write_pipeline_summary

    # Read from project_state.json, the file `paralome status` reads.
    # This read the variant caller's checkpoint for a "stages" map it never
    # had, wrote an empty summary and printed "Status: unknown" every time.
    try:
        summary = write_pipeline_summary(
            workdir=workdir,
            output_json=json_out,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    outputs = summary.get("outputs") or {}
    print(f"Summary written for: {workdir}")
    for kind in ("json",):
        if outputs.get(kind):
            print(f"  {kind.upper():<5}: {outputs[kind]}")
    print(f"Status: {summary.get('overall_status') or 'unknown'}")
    if summary.get("next_action"):
        print(f"Next  : {summary['next_action']}")
