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

"""``paralome collect`` -- gather files from anywhere under a project into one
flat folder, chosen by ``--filter`` and nothing else.

ONE RULE, AND THE USER TYPES IT. No directory name is special: every file under
the project is a candidate and ``--filter`` decides. A hardcoded directory name
would be a second filter the user cannot see or override, and would silently
return nothing when a layout changes. A bare ``paralome collect --project X``
with the default ``*`` filter therefore copies everything -- .sbatch scripts,
SLURM .err logs and checkpoint JSON included. Pass a --filter.

- **Destination**: a sibling directory, ``<project>_collected/`` by default
  (overridable via ``--out``). The destination subtree is pruned from the walk
  -- not as a naming rule but as self-reference protection, since ``--out`` may
  legitimately point inside ``--project``.
- **Flat.** Every matching file lands directly in the destination by its own
  filename -- no mirrored tree. Paralome's file names are unique for this.
- **Copy, never move.** The project tree is left exactly as it was.
- **Idempotent.** Re-collecting an unchanged file is a no-op; a file whose
  content changed since the last collect is written beside the old one with a
  dated suffix rather than overwriting it. The same dated suffix is given to
  the second of two DIFFERENT files that share a name (summary.json in two
  stage folders): collect cannot tell the two cases apart, and the command's
  report lists every renamed file with its source path. Either way a file
  already collected under a dated name is recognised on every later collect,
  so a collect that finds nothing new copies nothing.

PLANNED FIRST, THEN COPIED IN BATCHES:

  1. PLAN, in walk order, on one CPU. Every decision that depends on what is
     already collected -- copy, skip, or write a dated version -- is made here,
     including against files this same run has planned but not yet written. So
     the outcome cannot depend on which copy finishes first.
  2. COPY, ``--jobs`` at a time (default 4). Each copy is independent by then:
     a source and the one destination the plan gave it. Locally they run in
     threads; with ``--backend slurm`` they are one array task per file,
     throttled to ``--jobs``, and a finalize job checks every file arrived.

``--jobs`` means what it means everywhere else: how many things run
at once. A copy uses one CPU, so there is no ``--threads``. The ceiling is the
storage, not the CPUs -- more copies than the disks can serve only share the
same throughput.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

_DEFAULT_FILTER = "*"
#: Files copied at once when --jobs is not given. Enough to help on fast storage without flooding slow storage.
DEFAULT_JOBS = 4
#: Where a SLURM collect keeps its plan, scripts and logs, inside the destination. Hidden, and never collected
#: (the destination is pruned from the walk).
CONTROL_DIRNAME = ".paralome_collect"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _matches_filter(filename: str, filter_text: str) -> bool:
    """Whether this filename is wanted. A blank filter is refused, not answered.

    The blank case is the whole reason this has a docstring: `""` is
    a substring of every string, so `"" in name` is True and an empty filter
    matched EVERYTHING -- silently, and by the ordinary behaviour of `in`
    rather than by any rule anyone wrote. Removing the earlier
    `if not filter_text` short-circuit did not change that, which is the trap:
    the surprising answer came from Python, not from this function.

    So blank raises. run_collect refuses it earlier with a better message, and
    this is the backstop that keeps any other caller from rediscovering it.
    """
    if not str(filter_text).strip():
        raise ValueError("filter_text is blank; pass a substring, or '*' for everything")
    if filter_text == _DEFAULT_FILTER:
        return True
    return filter_text.lower() in filename.lower()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source, temp_path)
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path.exists():
            temp_path.unlink()


class _Planner:
    """Decides where each source lands, against the disk AND against this run's own earlier decisions.

    With the copies deferred, a destination this run has already promised to
    an earlier source is as taken as one on disk, and its content is that
    source's. Checksums are computed only when two files compete for one name,
    and each file is read at most once.
    """

    def __init__(self) -> None:
        self.planned: Dict[Path, Path] = {}        # destination -> the source that will be written there
        self._digests: Dict[Path, str] = {}

    def _digest(self, path: Path) -> str:
        if path not in self._digests:
            self._digests[path] = _sha256_file(path)
        return self._digests[path]

    def _taken(self, destination: Path) -> bool:
        return destination in self.planned or destination.exists()

    def _content(self, destination: Path) -> str:
        return self._digest(self.planned.get(destination, destination))

    def _dated_copies(self, destination: Path) -> List[Path]:
        """The dated forms of ``destination`` already on disk or planned by this run, in name order."""
        stem, suffix = destination.stem, destination.suffix
        dated = re.compile(rf"^{re.escape(stem)}_\d{{4}}-\d{{2}}-\d{{2}}(?:_\d+)?{re.escape(suffix)}$")
        found = {path for path in self.planned if path.parent == destination.parent and dated.match(path.name)}
        if destination.parent.is_dir():
            found.update(path for path in destination.parent.glob(f"{stem}_*{suffix}")
                         if path.is_file() and dated.match(path.name))
        return sorted(found)

    def place(self, source: Path, destination: Path) -> "_Placement":
        """Where ``source`` should land, honoring the checksum/dated-version rule.

        A fresh copy, a version alongside a different file of the same name, or
        a genuine no-op (identical content already collected, or already planned).
        """
        if not self._taken(destination):
            self.planned[destination] = source
            return _Placement(destination, "copied")
        if self._content(destination) == self._digest(source):
            return _Placement(destination, "unchanged")

        stem, suffix = destination.stem, destination.suffix
        # EVERY dated copy counts, not only today's. A copy already on disk, or planned by this run,
        # with this content under any dated form of the name is this file, already collected -- so an
        # unchanged file is not copied again on a later day, and the second of two different files
        # sharing a name (summary.json in several stage folders) is not taken as changed each time.
        for earlier in self._dated_copies(destination):
            if self._content(earlier) == self._digest(source):
                return _Placement(earlier, "unchanged")

        today = date.today().isoformat()
        candidate = destination.with_name(f"{stem}_{today}{suffix}")
        counter = 2
        while self._taken(candidate):
            if self._content(candidate) == self._digest(source):
                return _Placement(candidate, "unchanged")
            candidate = destination.with_name(f"{stem}_{today}_{counter}{suffix}")
            counter += 1
        self.planned[candidate] = source
        return _Placement(candidate, "versioned")


@dataclass
class _Placement:
    path: Path
    status: str  # "copied" | "unchanged" | "versioned"


@dataclass
class CollectEntry:
    source: str
    destination: str
    status: str  # "copied" | "unchanged" | "versioned" | "would_copy" | "would_version" | "skipped"


@dataclass
class CollectReport:
    """Two counters, and they are not the same number.

    ``files_scanned`` is every file the walk visited. ``files_matched`` is the
    subset that passed ``--filter``, and it is the one that must equal
    ``copied + unchanged + versioned``. Reported separately, so a filtered
    collect does not look as if the filter did nothing.
    """

    project: str
    destination: str
    filter_text: str
    dry_run: bool
    files_scanned: int
    files_matched: int = 0
    copied: int = 0
    unchanged: int = 0
    versioned: int = 0
    # Matched names that are not readable regular files (e.g. a dangling symlink); not in files_matched.
    skipped: int = 0
    entries: List[CollectEntry] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "project": self.project,
            "destination": self.destination,
            "filter": self.filter_text,
            "dry_run": self.dry_run,
            "files_scanned": self.files_scanned,
            "files_matched": self.files_matched,
            "copied": self.copied,
            "unchanged": self.unchanged,
            "versioned": self.versioned,
            "skipped": self.skipped,
            "entries": [
                {"source": e.source, "destination": e.destination, "status": e.status}
                for e in self.entries
            ],
        }


def _iter_files(project_dir: Path, destination_dir: Path):
    """Yield every file under ``project_dir``. No directory name is special.

    The only rule is the one the user typed (--filter); a hardcoded directory
    name would be a second, invisible filter that silently returns nothing when
    a layout changes.

    The destination is pruned. That is not a naming convention, it is
    self-reference: ``--out`` may legitimately sit inside ``--project``, and
    without this collect would copy its own output back into itself.
    """
    destination_resolved = destination_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(project_dir):
        current = Path(dirpath).resolve()
        # Collect's own bookkeeping folder (plan, scripts and logs of a SLURM collect) is pruned for
        # the same reason as the destination -- self-reference, not a naming rule about the user's results.
        # Sorted, so the walk -- and with it which of two same-named files keeps the plain name -- is the same
        # on every filesystem and every run.
        dirnames[:] = sorted(
            d for d in dirnames
            if d != CONTROL_DIRNAME and (current / d).resolve() != destination_resolved
        )
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def plan_collect(
    project_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    filter_text: str = _DEFAULT_FILTER,
    dry_run: bool = False,
) -> Tuple["CollectReport", List[Tuple[Path, Path]]]:
    """Phase 1: walk, match and decide. Returns the report and the copies still to make, in walk order.

    Nothing is written. ``dry_run`` only changes the status words in the report
    ("would_copy" for "copied"), so a dry run and a real run can never disagree
    about what will happen -- they are the same plan.
    """
    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"--project is not a directory: {project}")

    # A BLANK FILTER IS A MISTAKE, NOT A SELECTION.
    #
    # `_matches_filter` returned True for an empty string, so `--filter ""` --
    # an unset shell variable, most likely -- collected the entire project. A
    # whitespace-only one did the opposite and matched nothing, reporting a
    # successful run that copied no files. Neither is anything a person meant,
    # so neither is guessed at.
    #
    # Refused here as well as in the command, because the two failures need
    # different shapes: a library caller wants an exception, someone at a
    # terminal wants a usage error rather than a traceback.
    if not str(filter_text).strip():
        raise ValueError(
            "--filter needs something to search for. Pass a substring of the "
            'filenames you want, e.g. --filter evidome, or --filter "*" to take '
            "every result file."
        )

    destination = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else project.with_name(f"{project.name}_collected")
    )

    report = CollectReport(
        project=str(project),
        destination=str(destination),
        filter_text=filter_text,
        dry_run=dry_run,
        files_scanned=0,
    )
    planner = _Planner()
    copies: List[Tuple[Path, Path]] = []
    words = {"copied": "would_copy", "versioned": "would_version"} if dry_run else {}

    for source_file in _iter_files(project, destination):
        report.files_scanned += 1
        if not _matches_filter(source_file.name, filter_text):
            continue
        # A dangling symlink (or FIFO, socket) cannot be copied; it aborted the whole collect
        # part-way (local) or while writing the plan (SLURM). Listed as skipped instead.
        if not source_file.is_file():
            report.skipped += 1
            report.entries.append(CollectEntry(str(source_file), "", "skipped"))
            continue
        report.files_matched += 1
        # Flat destination: every file this collects
        # is already uniquely named via its comparison's file_prefix,
        # so no mirrored subdirectory tree is needed -- everything
        # lands directly in `destination` by filename alone.
        placement = planner.place(source_file, destination / source_file.name)
        if placement.status == "unchanged":
            report.unchanged += 1
        else:
            if placement.status == "copied":
                report.copied += 1
            else:
                report.versioned += 1
            copies.append((source_file, placement.path))
        report.entries.append(
            CollectEntry(str(source_file), str(placement.path), words.get(placement.status, placement.status))
        )
    return report, copies


def _copy_all(copies: List[Tuple[Path, Path]], jobs: int) -> None:
    """Phase 2, locally: ``jobs`` copies at a time. Every copy is attempted; the first failure is raised."""
    if jobs <= 1 or len(copies) <= 1:
        for source, target in copies:
            _copy_atomic(source, target)
        return
    from concurrent.futures import ThreadPoolExecutor

    # Threads, not processes: a copy spends its time in the kernel, where the interpreter lock is released.
    with ThreadPoolExecutor(max_workers=min(jobs, len(copies))) as pool:
        futures = [pool.submit(_copy_atomic, source, target) for source, target in copies]
    errors = [f.exception() for f in futures if f.exception() is not None]
    if errors:
        raise errors[0]


def run_collect(
    project_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    filter_text: str = _DEFAULT_FILTER,
    dry_run: bool = False,
    jobs: int = DEFAULT_JOBS,
) -> Dict[str, object]:
    """Collect matching files across ``project_dir`` into one flat destination, ``jobs`` copies at a time.

    See module docstring for the full design (destination default, copy-only,
    filter matching, checksum-based idempotent re-runs, plan then copy).
    """
    report, copies = plan_collect(project_dir, out_dir, filter_text=filter_text, dry_run=dry_run)
    if not dry_run:
        Path(report.destination).mkdir(parents=True, exist_ok=True)
        _copy_all(copies, max(1, int(jobs)))
    return report.as_dict()


# ---------------------------------------------------------------------------
# The same two phases under SLURM: plan here, copy as an array, check at the end
# ---------------------------------------------------------------------------

#: SLURM refuses an array larger than its MaxArraySize (1001 by default). A collect of thousands of small files
#: is spread over at most this many tasks, each copying every Nth file of the plan.
MAX_ARRAY_TASKS = 1000
PLAN_NAME = "plan.json"


def _task_count(copies: int) -> int:
    return max(1, min(int(copies), MAX_ARRAY_TASKS))


def submit_collect(
    project_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    filter_text: str = _DEFAULT_FILTER,
    jobs: int = DEFAULT_JOBS,
    partition: Optional[str] = None,
    node: Optional[str] = None,
    mem: str = "2G",
    submit: bool = False,
) -> Dict[str, object]:
    """Plan here, then copy as a SLURM array of ``jobs`` at a time, then a finalize job that checks every file.

    Fire-and-forget like every other SLURM-backed command: it writes the plan
    and the two scripts, submits them when ``submit`` is set, and returns the
    job IDs. Nothing is copied by this process.
    """
    import json
    from datetime import datetime

    from paralome.binaries import worker_command
    from paralome.pipeline.naming import slurm_job_name
    from paralome.pipeline.slurm import SlurmOptions, afterany, sbatch, write_sbatch
    from paralome.version import __version__

    report, copies = plan_collect(project_dir, out_dir, filter_text=filter_text)
    result = report.as_dict()
    result.update({"backend": "slurm", "submitted": False, "submitted_job_ids": [], "jobs": int(jobs)})
    if not copies:
        return result              # nothing to copy: the plan is the whole answer

    destination = Path(report.destination)
    control = destination / CONTROL_DIRNAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    (control / "logs").mkdir(parents=True, exist_ok=True)
    plan_path = control / PLAN_NAME
    tasks = _task_count(len(copies))
    plan = {
        "paralome_version": __version__,
        "report": result,
        "tasks": tasks,
        "copies": [{"source": str(s), "destination": str(d), "bytes": s.stat().st_size} for s, d in copies],
    }
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    project = Path(report.project)
    worker = worker_command()
    quoted_plan = shlex.quote(str(plan_path))
    array_script = write_sbatch(
        control / "collect_copy.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(project, "collect_copy"), cpus_per_task=1, mem=mem, partition=partition,
            node=node, array=f"1-{tasks}%{max(1, int(jobs))}",
            output=control / "logs" / "copy_%A_%a.out", error=control / "logs" / "copy_%A_%a.err",
        ),
        f'\nexec {worker} atac collect-task --plan {quoted_plan} --index "${{SLURM_ARRAY_TASK_ID:?}}"\n',
    )
    final_script = write_sbatch(
        control / "collect_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(project, "collect_finalize"), cpus_per_task=1, mem="1G", partition=partition,
            node=node, output=control / "logs" / "finalize_%j.out", error=control / "logs" / "finalize_%j.err",
        ),
        f"\nexec {worker} atac collect-finalize --plan {quoted_plan}\n",
    )
    result.update({"plan": str(plan_path), "scripts": [str(array_script), str(final_script)],
                   "logs": str(control / "logs"), "array_tasks": tasks})
    if not submit:
        return result
    array_id = sbatch(array_script)
    final_id = sbatch(final_script, dependency=afterany(array_id))
    result.update({"submitted": True, "array_job_id": array_id, "finalize_job_id": final_id,
                   "submitted_job_ids": [array_id, final_id]})
    return result


def _read_plan(plan_path: Union[str, Path]) -> Dict[str, object]:
    import json

    return json.loads(Path(plan_path).read_text(encoding="utf-8"))


def _arrived(copy: Dict[str, object]) -> bool:
    target = Path(str(copy["destination"]))
    return target.is_file() and target.stat().st_size == int(copy["bytes"])


def run_collect_task(plan_path: Union[str, Path], index: int) -> Dict[str, int]:
    """One array task: copy every ``tasks``-th file of the plan, starting at ``index`` (1-based).

    A file already at its destination with the planned size is left alone, so a
    resubmitted array only does what an earlier one did not finish.
    """
    plan = _read_plan(plan_path)
    copies = plan["copies"]
    tasks = int(plan["tasks"])
    if not 1 <= int(index) <= tasks:
        raise ValueError(f"--index {index} is outside this plan's 1-{tasks}")
    done = skipped = 0
    for copy in copies[int(index) - 1::tasks]:
        if _arrived(copy):
            skipped += 1
            continue
        _copy_atomic(Path(copy["source"]), Path(copy["destination"]))
        done += 1
    return {"copied": done, "already_there": skipped}


def finalize_collect(plan_path: Union[str, Path]) -> Dict[str, object]:
    """After the array: check every planned file arrived whole, and write the report beside the plan.

    Returns the report with ``missing`` listing every destination that is absent
    or not the planned size -- a failed or cancelled array task shows up here,
    by name, rather than as a count that is quietly one short.
    """
    import json

    plan_path = Path(plan_path)
    plan = _read_plan(plan_path)
    result = dict(plan["report"])
    missing = [str(c["destination"]) for c in plan["copies"] if not _arrived(c)]
    result.update({"missing": missing, "complete": not missing})
    (plan_path.parent / "collect_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
