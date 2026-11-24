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

"""Functional evidence: Evidome as the last stage of an expression pipeline.

This runs as a final step in every pipeline. Evidome is a separate program --
the sister project that turns a differential-expression table into pathway,
gene-set and upstream-regulator evidence for an interpreting agent -- and
Paralome runs it the way it runs STAR: found through
``binaries.resolve_tool`` (the recorded path, else PATH), given input files,
left to write its own outputs. Nothing of Evidome is imported.

What this stage promises:

* It never touches the differential results. They are its input, read-only.
* A machine without Evidome, or without its reference data for this
  species, is not an error in the analysis: the stage is recorded as
  **skipped** with the reason, and the pipeline completes. Those reasons are
  Evidome's own error codes for "not set up here" (``SKIP_CODES``).
* Any other failure is recorded as **failed**, with Evidome's message and a
  log, so it is seen rather than hidden.
* Under SLURM, Evidome submits its own chain of jobs and prints them as
  ``{"submitted_job_ids": [...]}``; this command passes that line through, so
  the supervisor waits on them exactly as it waits on its own stages. The
  submitted stage is complete when Evidome's results archive exists -- the
  archive alone. Evidome writes it whole (a temporary file renamed); OmicView's
  Evidence Browser opens it.

Output: ``<workdir>/<prefix>/`` holds everything Evidome writes, including
``<prefix>_evidome_output.zip`` (the evidence bundle, opened in OmicView's
Evidence Browser). The run log is ``<workdir>/logs/evidome.log``.

ONE EVIDOME RUN PER STATISTIC.  A differential stage records one folder
of results per statistic (``paralome.results``), each with its own Evidome
table; given the differential's checkpoint (``--differential-checkpoint``),
this stage runs Evidome on every one of them, into
``<workdir>/<folder>/<prefix>[_<stratum>]/`` with a log each, and the stage
is complete when every run's archive exists. An ANOVA folder's
table holds its F-tests and its contrasts together, so Evidome gives each
F-test its direction from its own contrasts. DNA's gene burden is one table
(``--statistics``) and keeps the single-run layout above.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

STAGE = "functional_evidence"
SCHEMA = "paralome.functional_evidence.v2"
PROGRAM = "evidome"

#: Evidome's error codes that mean "this machine or species is not set up
#: for it" rather than "something went wrong": no reference data (0013), the
#: input's species is not among those fetched (0014), no gene-set index
#: (0019), no regulator index (0025). The stage is skipped, with the reason.
#: Evidome 1.0 prefixes its codes EVD-; earlier Evidome builds wrote
#: PRV-. Both are read, so either Evidome works with this Paralome.
SKIP_NUMBERS = ("0013", "0014", "0019", "0025")
EVIDOME_CODE_PREFIXES = ("EVD", "PRV")
SKIP_CODES = frozenset(f"{prefix}-{number}" for prefix in EVIDOME_CODE_PREFIXES for number in SKIP_NUMBERS)

#: PRV-0014 ("Species not identified") is a skip only when the
#: species could not be FETCHED here -- no network to ask Ensembl, or the
#: species not downloaded. Evidome raises the same code when the input is at
#: fault: its IDs are not Ensembl IDs (a RefSeq or symbol GTF), they belong
#: to several species, or too few match any species. A 0014 whose line carries
#: none of these phrases is therefore a failure, with Evidome's line.
_NOT_FETCHED_0014 = ("could not ask Ensembl which species", "not fetched")


def is_setup_skip(code: Optional[str], line: str) -> bool:
    """Whether Evidome's stopping error means "not set up here" (a skip) rather than a failure."""
    if code not in SKIP_CODES:
        return False
    if code.endswith("-0014"):
        return any(marker in (line or "") for marker in _NOT_FETCHED_0014)
    return True

#: Why a skipped stage was skipped, for a program to read: the
#: checkpoint's ``skip_reason``. The supervisor uses it to look again once
#: Evidome, or its data, has been installed.
SKIP_NOT_INSTALLED = "evidome_not_installed"
SKIP_DATA_MISSING = "evidome_data_missing"
#: The differential wrote no gene-level table for Evidome (snATAC
#: without a peak annotation has no gene for a peak). Nothing to ask Evidome
#: until the differential runs again with one, so the supervisor leaves it.
SKIP_NO_TABLE = "no_evidome_table"

#: Evidome reports the error it stopped on as a line that BEGINS
#: with its code. Only such a line is read as "the error", so a code mentioned
#: in a warning or a progress line cannot turn a real failure into a skip.
_PRV_LINE = re.compile(r"^\s*((?:EVD|PRV)-\d{4})\b")


def _stopping_error(stderr: str, stdout: str) -> Tuple[Optional[str], str]:
    """(code, line) of the error Evidome exited on, or (None, "").

    The last code-led line it wrote, on stderr if there is one there: what a
    program says last before a non-zero exit is why it stopped.
    """
    for text in (stderr or "", stdout or ""):
        for line in reversed(text.splitlines()):
            match = _PRV_LINE.match(line)
            if match:
                return match.group(1), line.strip()
    return None, ""


def bundle_dir(workdir: Path, prefix: str) -> Path:
    return Path(workdir) / prefix


def bundle_archive(workdir: Path, prefix: str) -> Path:
    """Evidome names its archive after its --out folder: ``<prefix>_evidome_output.zip`` (``_evidome_results.zip`` before, with Evidome 1.1 as it was)."""
    return bundle_dir(workdir, prefix) / f"{prefix}_evidome_output.zip"


def bundle_workbook(workdir: Path, prefix: str) -> Path:
    """The workbook an older Evidome wrote after the archive: ``<prefix>_evidome_workbook.xlsx``.

    Evidome writes none now; one an earlier run left is set aside before a new run, and named in the stage's
    summary when it is there. It is never required.
    """
    return bundle_dir(workdir, prefix) / f"{prefix}_evidome_workbook.xlsx"


def _set_aside(path: Path) -> None:
    """Keep an earlier run's file under another name, so nothing it left reads as this run's."""
    if path.is_file():
        path.replace(path.with_name(path.name + ".previous"))


def evidome_command(executable: str, statistics: Path, workdir: Path, prefix: str, *, backend: str,
                      partition: Optional[str] = None, alpha: Optional[float] = None) -> List[str]:
    cmd = [executable, str(statistics), "--out", str(bundle_dir(workdir, prefix))]
    # The run's --alpha: the p Evidome selects genes by.
    if alpha is not None:
        cmd += ["--alpha", f"{float(alpha):g}"]
    if backend == "slurm":
        cmd += ["--backend", "slurm"]
        if partition:
            cmd += ["--partition", partition]
    return cmd


def _submitted_ids(text: str) -> List[str]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "submitted_job_ids" in line:
            try:
                return [str(v) for v in json.loads(line)["submitted_job_ids"]]
            except (ValueError, KeyError, TypeError):
                return []
    return []


def evidence_inputs_from_checkpoint(path: Path) -> List[Dict[str, str]]:
    """Every Evidome table a differential stage's checkpoint records: one per statistic folder (per stratum)."""
    from paralome.pipeline.state import evidome_inputs

    return evidome_inputs(json.loads(Path(path).read_text()))


def _run_label_dir(workdir: Path, run: Mapping[str, str], prefix: str) -> Tuple[Path, str]:
    """(the folder Evidome's --out sits in, the bundle's name) for one run."""
    folder = str(run.get("folder") or "")
    stratum = str(run.get("stratum") or "")
    name = prefix
    if stratum:
        from paralome.atac.singlecell import sanitize_sample_id

        name = f"{prefix}_{sanitize_sample_id(stratum)}"
    # The bundle names its statistic, as every result file does.
    if folder:
        name = f"{name}_{folder}"
    return (Path(workdir) / folder if folder else Path(workdir)), name


def _run_one(
    run: Dict[str, object], executable: str, workdir: Path, prefix: str, *, backend: str,
    partition: Optional[str], runner: Callable[..., "subprocess.CompletedProcess"],
    echo: Callable[[str], None], zip_dir: Optional[Path] = None, alpha: Optional[float] = None,
) -> Dict[str, object]:
    """One Evidome run on one table. Fills and returns ``run``."""
    from paralome.checkpoints import atomic_write_text

    base, name = _run_label_dir(workdir, run, prefix)
    archive, workbook = bundle_archive(base, name), bundle_workbook(base, name)
    log = Path(workdir) / "logs" / (f"evidome_{run['label'].replace('/', '_')}.log" if run.get("label") else "evidome.log")
    run.update(bundle_dir=str(bundle_dir(base, name)), archive=str(archive), workbook=str(workbook), log=str(log))
    if not Path(str(run["statistics"])).is_file():
        run.update(status="failed", reason=f"differential statistics not found: {run['statistics']}")
        return run
    # AN ARCHIVE FROM AN EARLIER RUN IS NOT THIS RUN'S RESULT. The
    # old archive and workbook are set aside, not deleted, before Evidome is
    # started; whatever is at their paths afterwards was written by this run.
    _set_aside(archive)
    _set_aside(workbook)
    # And the archive an earlier run moved into the omicview folder, back beside Evidome's
    # folder under the same name, so the omicview folder holds only finished archives.
    placed = Path(zip_dir) / archive.name if zip_dir else None
    if placed is not None and placed.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(placed, archive.with_name(archive.name + ".previous"))
    cmd = evidome_command(executable, Path(str(run["statistics"])), base, name, backend=backend, partition=partition, alpha=alpha)
    run["command"] = cmd
    try:
        completed = runner(cmd, check=False, text=True, capture_output=True)
    except OSError as exc:
        run.update(status="failed", reason=f"{PROGRAM} could not be started: {exc}")
        return run
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    log.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(log, "$ " + " ".join(cmd) + "\n" + output + "\n")
    if completed.stdout:
        # Everything but the job-ID line, which is passed on once, for every
        # run together, at the end (the supervisor waits on all of them).
        shown = [line for line in completed.stdout.splitlines() if "submitted_job_ids" not in line]
        if shown:
            echo("\n".join(shown))
    if completed.returncode != 0:
        code, line = _stopping_error(completed.stderr, completed.stdout)
        if is_setup_skip(code, line):
            run.update(status="skipped", reason=f"{PROGRAM} is not set up for this input here: {line}",
                       evidome_code=code, skip_reason=SKIP_DATA_MISSING)
        else:
            run.update(status="failed", reason=line or f"{PROGRAM} exited with status {completed.returncode}",
                       evidome_code=code, returncode=completed.returncode)
        return run
    jobs = _submitted_ids(completed.stdout or "")
    if jobs:
        run.update(status="submitted", submitted_job_ids=jobs)
        return run
    if not archive.is_file():                              # The archive alone is the finished run
        run.update(status="failed", reason=f"{PROGRAM} exited normally but wrote no {archive.name}")
    else:
        run.update(status="complete")
    return run


def run_functional_evidence(
    statistics: Optional[Path],
    workdir: Path,
    prefix: str,
    *,
    differential_checkpoint: Optional[Path] = None,
    backend: str = "local",
    partition: Optional[str] = None,
    resolve: Optional[Callable[[str], Optional[str]]] = None,
    runner: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
    echo: Callable[[str], None] = print,
    zip_dir: Optional[Path] = None,
    trajectory_checkpoint: Optional[Path] = None,
    submit: Optional[Callable[..., str]] = None,
    alpha: Optional[float] = None,
) -> Dict[str, object]:
    """Run Evidome on one table, or on every statistic folder a differential recorded.

    Evidome's jobs submitted (``backend`` slurm) with a ``zip_dir``, one more job waits on them all and
    moves each finished archive there (``submit_placing``; ``submit`` is sbatch unless given) -- the commands that
    run Evidome with no supervisor watching (olink, proteomics) left the archive in Evidome's folder.

    And on every statistic folder of every segment the trajectories stage recorded
    (``trajectory_checkpoint``, its differential's checkpoint), after the differential's own.

    Writes and returns the stage checkpoint. Never raises for Evidome's sake. Each finished
    run's archive is moved into ``zip_dir`` (the project's omicview/functional_evidence folder) -- here when Evidome
    ran here, and by summarize when its jobs finish.
    """
    from paralome.checkpoints import atomic_write_text, now_iso, stage_checkpoint_path
    from paralome.version import __version__

    workdir = Path(workdir)
    (workdir / "logs").mkdir(parents=True, exist_ok=True)
    checkpoint: Dict[str, object] = {
        "schema": SCHEMA, "stage": STAGE, "paralome_version": __version__, "updated": now_iso(),
        "prefix": prefix, "backend": backend,
        "differential_checkpoint": str(differential_checkpoint) if differential_checkpoint else None,
        "statistics": str(statistics) if statistics else None,
        "trajectory_checkpoint": str(trajectory_checkpoint) if trajectory_checkpoint else None,
    }

    def finish(status: str, **fields: object) -> Dict[str, object]:
        checkpoint.update(fields, status=status, complete=status == "complete")
        if zip_dir:
            checkpoint["zip_dir"] = str(zip_dir)
            _place_archives(checkpoint, Path(zip_dir), echo, workdir)
        atomic_write_text(stage_checkpoint_path(workdir, STAGE), json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
        echo(f"functional evidence: {status}" + (f" -- {fields['reason']}" if fields.get("reason") else ""))
        return checkpoint

    if (statistics is None) == (differential_checkpoint is None):
        return finish("failed", reason="give either --statistics (one table) or --differential-checkpoint")
    runs: List[Dict[str, object]] = []
    if differential_checkpoint is not None:
        try:
            inputs = evidence_inputs_from_checkpoint(Path(differential_checkpoint))
        except (OSError, ValueError) as exc:
            return finish("failed", reason=f"cannot read {differential_checkpoint}: {exc}")
        if not inputs:
            return finish("skipped", reason=(
                f"{differential_checkpoint} records no Evidome table -- the differential had no gene "
                "for its features (peaks are given genes from --genome or --gtf); the differential results are complete"
            ), skip_reason=SKIP_NO_TABLE)
        runs = [{"label": item["label"], "folder": item["folder"], "stratum": item["stratum"],
                 "statistics": item["path"]} for item in inputs]
        if trajectory_checkpoint is not None:
            try:
                along = evidence_inputs_from_checkpoint(Path(trajectory_checkpoint))
            except (OSError, ValueError) as exc:
                return finish("failed", reason=f"cannot read {trajectory_checkpoint}: {exc}")
            runs += [{"label": item["label"], "folder": item["folder"], "stratum": item["stratum"],
                      "statistics": item["path"]} for item in along]
    else:
        runs = [{"label": "", "folder": "", "stratum": "", "statistics": str(statistics)}]
    checkpoint["runs"] = runs

    if resolve is None:
        from paralome.binaries import resolve_tool as resolve
    try:
        executable = resolve(PROGRAM)
    except Exception as exc:  # a recorded path that no longer resolves (PLM-0410): say so, do not crash
        return finish("failed", reason=str(exc).splitlines()[0])
    if not executable:
        return finish("skipped", reason=f"{PROGRAM} is not installed on this machine (not recorded by "
                                        f"`paralome setup`, not on PATH); the differential results are complete",
                      skip_reason=SKIP_NOT_INSTALLED)

    for run in runs:
        _run_one(run, executable, workdir, prefix, backend=backend, partition=partition, runner=runner, echo=echo,
                 zip_dir=zip_dir, alpha=alpha)
        if run.get("label"):
            echo(f"functional evidence [{run['label']}]: {run['status']}"
                 + (f" -- {run['reason']}" if run.get("reason") else ""))

    statuses = [str(run["status"]) for run in runs]
    jobs = [job for run in runs for job in (run.get("submitted_job_ids") or [])]
    if jobs and zip_dir and backend == "slurm":
        placed = submit_placing(workdir, Path(zip_dir), jobs, partition=partition, submit=submit, echo=echo)
        checkpoint["place_job"] = placed
        if placed.get("job_id"):
            jobs = jobs + [str(placed["job_id"])]
    if jobs:
        echo(json.dumps({"submitted_job_ids": jobs}))   # to the supervisor, once, for every run
    failed = [run for run in runs if run["status"] == "failed"]
    if failed:
        return finish("failed", reason="; ".join(
            (f"{run['label']}: " if run.get("label") else "") + str(run.get("reason")) for run in failed
        ), evidome_code=failed[0].get("evidome_code"), submitted_job_ids=jobs or None)
    if jobs:
        return finish("submitted", submitted_job_ids=jobs)
    if all(status == "skipped" for status in statuses):
        first = runs[0]
        return finish("skipped", reason=str(first.get("reason")), evidome_code=first.get("evidome_code"),
                      skip_reason=first.get("skip_reason"))
    return finish("complete")


def submit_placing(workdir: Path, zip_dir: Path, jobs: List[str], *, partition: Optional[str] = None,
                   submit: Optional[Callable[..., str]] = None, echo: Callable[[str], None] = print) -> Dict[str, object]:
    """A SLURM job, after Evidome's jobs end (afterany), that moves each finished archive into ``zip_dir``.

    `paralome olink` and `paralome proteomics` submit Evidome and end, so no
    supervisor reads the stage again to move the archive; this job runs
    `paralome-worker evidence place`, the same move. Never raises: a refused
    submission is recorded and said, and the archive stays where Evidome wrote
    it."""
    import shlex

    from paralome.binaries import worker_command
    from paralome.pipeline.slurm import SlurmOptions, afterany, sbatch, write_sbatch

    logs = Path(workdir) / "logs"
    script = Path(workdir) / "scripts" / "place_evidome_archives.sbatch"
    body = (f"{worker_command()} evidence place --out {shlex.quote(str(Path(workdir).resolve()))} "
            f"--zip-dir {shlex.quote(str(Path(zip_dir).resolve()))}\n")
    try:
        write_sbatch(script, SlurmOptions(job_name="paralome_evidome_place", cpus_per_task=1, mem="1G",
                                          partition=partition, output=logs / "place_archives_%j.out",
                                          error=logs / "place_archives_%j.err"), body)
        job_id = (submit or sbatch)(script, dependency=":".join([afterany(jobs[0]), *jobs[1:]]))
    except Exception as exc:          # Evidome's run stands; only the move is not queued
        echo(f"functional evidence: the job that moves the zip into {zip_dir} was not submitted -- {exc}")
        return {"script": str(script), "submitted": False, "reason": str(exc)}
    echo(f"functional evidence: job {job_id} moves the zip into {zip_dir} when Evidome's jobs end")
    return {"script": str(script), "submitted": True, "job_id": str(job_id), "after": list(jobs)}


def place_archives(workdir: Path, zip_dir: Path) -> Dict[str, object]:
    """`paralome-worker evidence place`: each finished run's archive moved into ``zip_dir`` (``summarize``'s move)."""
    return summarize(Path(workdir), Path(zip_dir))


def _within(path: Path, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def _place_archives(checkpoint: Dict[str, object], zip_dir: Path, echo: Callable[[str], None] = print,
                    workdir: Optional[Path] = None) -> bool:
    """Each finished run's archive moved into ``zip_dir``; the run's
    record names where it now is. True when one was moved. Nothing reruns: the archive alone is the finished run.
    Only an archive inside this stage's folder is moved: a copied project's record may still name the original's."""
    moved = False
    for run in checkpoint.get("runs") or []:
        recorded = Path(str(run.get("archive") or ""))
        if not run.get("archive") or not recorded.is_file():
            continue
        if workdir is not None and not _within(recorded, workdir):
            continue
        target = Path(zip_dir) / recorded.name
        if recorded.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(recorded, target)
        run["archive"] = str(target)
        echo(f"functional evidence: {recorded} moved to {target}")
        moved = True
    return moved


def _in_this_copy(wd: Path, recorded: Path, folder: str = "") -> Path:
    """The recorded file, or its copy under this stage folder when the original is gone.

    The checkpoint holds the archive's absolute path. In a project
    copied elsewhere, with the original gone, a finished stage read as
    incomplete and Evidome was run again. The file is <prefix>/<name> under
    the stage folder -- under the run's statistic folder
    (``<folder>/<prefix>/<name>``) for a differential's runs: the copy at
    that place here is the same result.
    """
    if str(recorded) and not recorded.is_file():
        here = wd / folder / recorded.parent.name / recorded.name
        if here.is_file():
            return here
    return recorded


def _run_state(wd: Path, run: Mapping[str, object], recorded_status: str,
               zip_dir: Optional[Path] = None) -> Tuple[str, Optional[str], Path, Path]:
    """(status, reason, archive, workbook) of one Evidome run, read from what is on disk now.

    The archive and workbook are where they are in this folder (_in_this_copy).
    """
    folder = str(run.get("folder") or "")
    archive = _in_this_copy(wd, Path(str(run.get("archive") or "")), folder)
    if str(archive) not in ("", ".") and zip_dir and (Path(zip_dir) / archive.name).is_file():
        archive = Path(zip_dir) / archive.name            # a copied project (its omicview folder)
    workbook = _in_this_copy(wd, Path(str(run.get("workbook") or "")), folder)
    status = str(run.get("status") or recorded_status)
    if status == "complete":
        return ("complete" if archive.is_file() else "incomplete"), None, archive, workbook
    if status == "submitted":
        # The archive, written whole, is the finished run; no workbook is waited for.
        if archive.is_file():
            return "complete", None, archive, workbook
        return "submitted", None, archive, workbook
    return status, (str(run.get("reason")) if run.get("reason") else None), archive, workbook


def summarize(workdir: Optional[Path], zip_dir: Optional[Path] = None) -> Dict[str, object]:
    """Stage state for project_state.json: complete once every run's archive exists.

    With ``zip_dir`` (the project's omicview/functional_evidence folder), a finished run's
    archive found elsewhere -- Evidome's own folder, where its jobs wrote it, or an older layout -- is
    moved there first and the checkpoint says so.
    """
    from paralome.checkpoints import atomic_write_text, stage_checkpoint_path

    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    path = stage_checkpoint_path(wd, STAGE)
    try:
        ckpt = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "not_available", "workdir": str(wd), "complete": False}
    if zip_dir and str(ckpt.get("status")) in ("complete", "submitted") and _place_archives(ckpt, Path(zip_dir), lambda _: None, wd):
        atomic_write_text(path, json.dumps(ckpt, indent=2, sort_keys=True) + "\n")
    status = str(ckpt.get("status") or "unknown")
    reason = ckpt.get("reason")
    runs = list(ckpt.get("runs") or [])
    if not runs and ckpt.get("archive"):
        # A v1 checkpoint (one table): its archive and workbook at the top level.
        prefix = str(ckpt.get("prefix") or Path(str(ckpt["archive"])).parent.name)
        runs = [{"label": "", "status": status, "archive": ckpt["archive"],
                 "workbook": ckpt.get("workbook") or str(Path(str(ckpt["archive"])).parent / f"{prefix}_evidome_workbook.xlsx"),
                 "log": ckpt.get("log"), "reason": reason}]
    run_states = []
    if status in ("complete", "submitted") and runs:
        for run in runs:
            run_status, run_reason, archive, workbook = _run_state(wd, run, status, zip_dir)
            run_states.append({"label": run.get("label"), "status": run_status, "reason": run_reason,
                               "archive": str(archive) if str(archive) != "." else None,
                               "workbook": str(workbook) if str(workbook) != "." and workbook.is_file() else None})
        if all(item["status"] == "complete" or item["status"] == "skipped" for item in run_states) and any(
                item["status"] == "complete" for item in run_states):
            status = "complete"
        elif any(item["status"] == "incomplete" for item in run_states):
            status = "incomplete"
            reason = "; ".join(
                (f"{item['label']}: " if item["label"] else "") + str(item["reason"])
                for item in run_states if item["status"] == "incomplete" and item["reason"]
            ) or reason
        elif status == "complete":
            status = "incomplete"
    first = runs[0] if runs else {}
    shown = run_states[0] if run_states else first
    return {
        "status": status, "complete": status == "complete", "workdir": str(wd),
        "reason": reason, "skip_reason": ckpt.get("skip_reason"),
        "runs": run_states or [{"label": run.get("label"), "status": run.get("status")} for run in runs],
        "archive": shown.get("archive"), "workbook": shown.get("workbook"),
        "bundle_dir": first.get("bundle_dir"), "log": first.get("log"),
        "submitted_job_ids": ckpt.get("submitted_job_ids"),
    }
