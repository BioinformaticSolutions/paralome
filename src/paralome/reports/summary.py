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

"""The `paralome report summary` writer: a project's recorded state as JSON and HTML.

It reads what `paralome status` reads -- the project's project_state.json,
written by every run -- and refuses, saying so, when a folder has none.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Optional


def write_summary_json(path: Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def read_project_state(workdir: Path) -> dict[str, Any]:
    """The project_state.json of ``workdir``, as the last run recorded it.

    Raises FileNotFoundError, naming the file and the way to get one, when the
    folder has none: a project appears once it has run at least once.
    """
    from paralome.pipeline.survey import STATE_FILENAME

    path = Path(workdir) / STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{Path(workdir)} has no {STATE_FILENAME}, so there is no recorded state to summarize. "
            "That file is written once a pipeline run has started; `paralome status --project "
            f"{Path(workdir)} --samples <sheet> --write` builds it from the stage folders."
        )
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not hold a project state object")
    return state


def summarize_project_state(state: dict[str, Any]) -> dict[str, Any]:
    """The summary written: the workflow section of a project state, with each stage's own status.

    ``overall_status`` is the workflow state (complete, running, ready,
    waiting, failed ...), ``next_action`` what `paralome status` would say to
    do next.
    """
    pipeline = state.get("pipeline") if isinstance(state.get("pipeline"), dict) else {}
    overall = state.get("overall") if isinstance(state.get("overall"), dict) else {}
    project = state.get("project") if isinstance(state.get("project"), dict) else {}
    workflow_stages = pipeline.get("stages") if isinstance(pipeline.get("stages"), dict) else {}
    stage_names = list(workflow_stages) or [
        name for name in (overall.get("stages") or {}) if isinstance(state.get(name), dict)
    ]
    stages: dict[str, Any] = {}
    for name in stage_names:
        own = state.get(name) if isinstance(state.get(name), dict) else {}
        workflow = workflow_stages.get(name) if isinstance(workflow_stages.get(name), dict) else {}
        stages[name] = {
            "label": workflow.get("label", name),
            "status": own.get("status", workflow.get("status", "unknown")),
            "state": workflow.get("state"),
            "complete": bool(workflow.get("complete", own.get("complete", False))),
            "optional": bool(workflow.get("optional", False)),
            "stale": bool(workflow.get("stale", False)) or bool(own.get("stale")),
            "workdir": own.get("workdir"),
        }
        for key in ("reason", "out_of_date", "error", "skip_reason"):
            if own.get(key):
                stages[name][key] = own.get(key)
    return {
        "schema": "paralome.report_summary.v2",
        # The Paralome that wrote the state. Variant runs made before
        # Carry only a hard-coded "gv_version"; it is reported as it
        # was recorded, under the key that says what it is.
        "paralome_version": state.get("paralome_version") or state.get("version"),
        "recorded_gv_version": state.get("gv_version"),
        "created": state.get("created"),
        "updated": state.get("updated"),
        "project": project,
        "pipeline_profile": pipeline.get("pipeline_profile") or project.get("pipeline_profile") or project.get("profile"),
        "overall_status": pipeline.get("state") or ("complete" if pipeline.get("complete") else "unknown"),
        "complete": bool(pipeline.get("complete", False)),
        "current_stage": pipeline.get("current_stage"),
        "next_stage": pipeline.get("next_stage"),
        "next_action": overall.get("next_action"),
        "stages": stages,
    }


def write_pipeline_summary(workdir: Path, output_json: Optional[Path] = None) -> dict[str, Any]:
    workdir = Path(workdir)
    summary = summarize_project_state(read_project_state(workdir))
    if output_json is None:
        output_json = workdir / "results" / f"{workdir.name}.pipeline.summary.json"
    write_summary_json(output_json, summary)
    summary["outputs"] = {"json": str(output_json)}
    return summary
