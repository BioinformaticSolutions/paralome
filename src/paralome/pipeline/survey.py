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

"""Finding every project below here, and saying where each one got to.

A bare ``paralome status`` walks down from where it is run, with no depth
limit, reads every ``project_state.json`` it finds, and prints one line each,
so the user does not have to re-enter the run parameters.

Paralome writes project_state.json into stage workdirs as well as project
roots (the orchestrator writes a snapshot into each stage workdir it runs).
Two rules keep those snapshots out of the listing:

- Once a directory yields a state file the walk stops there, so a project's
  own ``variants/`` directory is never listed as a separate project.
- A state file in a directory that the state itself records as a stage
  workdir, and that does not hold the other recorded stage workdirs, is a
  stage snapshot written outside the project (the shared single-cell data
  folder, an ``--align-workdir`` elsewhere). It is not listed.

Hidden directories are skipped as well, to keep VCS and workflow-tool
folders out of a listing.

READ-ONLY, so an interrupted scan leaves nothing behind.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

#: What marks a directory as a project. Written by the pipeline after every
#: step (commands/pipeline.py) and by the supervisor (pipeline/supervisor.py),
#: so its presence means a run happened here, and its contents are current.
STATE_FILENAME = "project_state.json"


@dataclass(frozen=True)
class ProjectSummary:
    """One line of the survey."""

    path: Path
    profile: str
    updated: str
    next_stage: str
    complete: bool

    #: Where the survey was run from, so the listing can print a path relative
    #: to it rather than a bare directory name.
    root: Optional[Path] = None

    @property
    def name(self) -> str:
        """What to print -- and what can be pasted straight into --project.

        What a listing prints has to be what can be pasted into --project:
        two projects with the same directory name in different places must
        be told apart, and a bare directory name is not a path. Relative to
        the scan root, so it stays short, and absolute when it cannot be.
        """
        if self.root is not None:
            try:
                return str(self.path.resolve().relative_to(Path(self.root).resolve()))
            except ValueError:
                pass
        return str(self.path)


def _is_stage_snapshot(directory: Path, state: object) -> bool:
    """True for a stage workdir's state snapshot outside its project.

    The directory is one of the stage workdirs the state records, and at
    least one other recorded stage workdir is not inside it (a project root
    holds its own stage folders; a stage folder outside the project does not).
    """
    if not isinstance(state, dict) or not isinstance(state.get("project"), dict):
        return False
    here = Path(directory).resolve()
    recorded: list[Path] = []
    for key, value in state["project"].items():
        if (str(key).endswith("_workdir") or key == "data_workdir_path") and value:
            try:
                recorded.append(Path(str(value)).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                continue
    if here not in recorded:
        return False
    others = [path for path in recorded if path != here]
    return any(here not in path.parents for path in others)


def find_projects(root: Path) -> Iterator[Path]:
    """Every directory at or below ``root`` holding a project state file.

    Walked with os.walk and pruned in place: assigning to ``dirnames`` is what
    stops os.walk descending, which is the whole reason this is not a
    recursive glob.
    """
    root = Path(root).expanduser().resolve()

    def snapshot(directory: Path) -> bool:
        try:
            state = json.loads((directory / STATE_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return _is_stage_snapshot(directory, state)

    if (root / STATE_FILENAME).is_file():
        if not snapshot(root):
            yield root
        return

    for current, dirnames, filenames in os.walk(root, topdown=True):
        here = Path(current)
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        if STATE_FILENAME in filenames:
            if not snapshot(here):
                yield here
            # See the module docstring: a state file below this one belongs
            # to a stage workdir, not to a separate project.
            dirnames[:] = []


def read_summary(project: Path, root: Optional[Path] = None) -> Optional[ProjectSummary]:
    """One project's headline, or None if its state file cannot be read.

    Unreadable is not an error worth stopping a survey for: a half-written
    file from a job that died mid-write should cost that one line, not the
    other eleven.
    """
    path = Path(project) / STATE_FILENAME
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None

    details = state.get("project") if isinstance(state.get("project"), dict) else {}
    pipeline = state.get("pipeline") if isinstance(state.get("pipeline"), dict) else {}

    next_stage = str(pipeline.get("next_stage") or "")
    # A state file without a pipeline section says nothing about
    # progress; show "?" rather than "complete".
    if not pipeline:
        next_stage = "?"
    complete = not next_stage or next_stage.lower() in {"complete", "none", "-"}
    # A finished project says "complete", not the whole sentence overall
    # carries -- "scRNA workflow is complete through differential expression."
    # is a paragraph in a column of stage names, and the column is for
    # scanning.
    return ProjectSummary(
        path=Path(project),
        profile=str(details.get("profile") or details.get("pipeline_profile") or "?"),
        updated=str(state.get("updated") or "")[:16].replace("T", " "),
        next_stage="complete" if complete else next_stage,
        complete=complete,
        root=root,
    )


def survey(root: Path, profile: Optional[str] = None) -> List[ProjectSummary]:
    """Every readable project below ``root``, newest first.

    ``profile`` narrows it, for example to the scRNA projects only.

    MATCHED EXACTLY, capitals included: profile names are case-sensitive
    everywhere in the CLI ("scRNA", not "scrna"; `paralome scrna` fails
    loudly and normalize_pipeline_profile refuses "scrna"), so the filter
    does not accept a spelling the rest of the CLI refuses.

    What makes that bearable rather than merely strict is print_survey, which
    names the profiles actually present when a filter matches nothing.
    """
    found = [read_summary(path, root) for path in find_projects(root)]
    summaries = [item for item in found if item is not None]
    if profile:
        wanted = str(profile).strip()
        summaries = [item for item in summaries if item.profile == wanted]
    return sorted(summaries, key=lambda item: item.updated, reverse=True)


def print_survey(
    root: Path, summaries: List[ProjectSummary], profile: Optional[str] = None,
) -> None:
    """The listing. Says what was scanned, so an empty result is not a mystery."""
    narrowed = f" ({profile} only)" if profile else ""
    print(f"Projects under {Path(root).resolve()}{narrowed}")
    print("")
    if not summaries:
        print("  None found.")
        print("")
        if profile:
            present = sorted({item.profile for item in survey(root)})
            print(f"  Nothing here is {profile}.")
            if present:
                print(f"  Profiles found under this directory: {', '.join(present)}")
                print("  They are case-sensitive: scRNA and snATAC, not scrna.")
            print("  Drop --profile to see everything.")
        else:
            print("  A project appears here once it has been run at least once --")
            print(f"  that is when {STATE_FILENAME} is written. Nothing has been.")
        return

    width = max(len("PROJECT"), max(len(item.name) for item in summaries))
    profile_width = max(len("PROFILE"), max(len(item.profile) for item in summaries))
    print(f"  {'PROJECT'.ljust(width)}  {'PROFILE'.ljust(profile_width)}  "
          f"{'UPDATED'.ljust(16)}  NEXT STAGE")
    for item in summaries:
        print(f"  {item.name.ljust(width)}  {item.profile.ljust(profile_width)}  "
              f"{item.updated.ljust(16)}  {item.next_stage}")
    print("")
    unfinished = [item for item in summaries if not item.complete]
    print(f"  {len(summaries)} project(s), {len(unfinished)} unfinished.")
    print("")
    print("  Detail for one:  paralome status --project <path>")
