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

import errno
import fcntl
import glob
import hashlib
import inspect
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from paralome.checkpoints import stage_checkpoint_path
from paralome.binaries import worker_argv
from paralome.errors import ParalomeError
from paralome.layout import stale_layout_warning
from paralome.pipeline.jsonio import write_json
from paralome.pipeline.orchestrator import (
    REQUEST_RECORD_FORMAT,
    SELF_CHECKING_STAGES,
    STAGES_THAT_ALWAYS_RECOMPUTE,
    PipelineRunResult,
    StageCommandError,
    build_stage_command,
    describe_request_change,
    only_writes_scripts,
    request_records_differ,
    run_pipeline_next_stage,
    stage_request_record,
)
from paralome.pipeline.profiles import STAGE_PARTS, blocks, resolve_pipeline_profile, stage_reads
from paralome.pipeline.state import build_project_state, now_iso, write_project_state


# DEADLINE and REVOKED are terminal too.
FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "DEADLINE", "REVOKED"}
FAILED_QUEUE_REASONS = {"DEPENDENCYNEVERSATISFIED"}
# Failures a cluster may answer by running the job again. One of
# these is final only once the stage has had nothing running for two polls in a
# row; a requeued job comes back as PENDING and is waited on like any other.
REQUEUE_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
# No SLURM query may hang the supervisor. A query that has not
# answered in this long counts as "no answer" -- never as "nothing there".
SLURM_QUERY_TIMEOUT_SECONDS = 120
PIPELINE_STAGE_ORDER = [
    "fastq_qc",
    "alignment",
    "rna_quantification",
    "differential_expression",
    "deduplication",
    "macs_peak_calling",
    "peak_merge",
    "peak_quantification",
    "differential_accessibility",
    # Motif analysis is two stages (singles / pairs). The legacy unsplit
    # name is kept so an older project_state.json or supervisor checkpoint
    # still validates without a migration.
    "motif_analysis",
    "motif_analysis_singles",
    "motif_analysis_pairs",
    "variants",
    "annotation",
    "segregation",
    "filtering",
    "reporting",
    # DNA's gene burden against gnomAD, after the candidate reports
    # and before functional_evidence (last, below), which reads its table.
    "gene_burden",
    # The variant browser's bundle, after gene burden. Every arm's export for OmicView, the stage before
    # functional_evidence in every pipeline. Genotype_output on DNA, statistics_output on every other arm.
    "genotype_output",
    "statistics_output",
    # The ATAC pipelines' genome browser tracks, after statistics_output.
    "genome_tracks",
    "singlecell_atac",
    "singlecell_differential",
    "singlecell_motif_analysis",
    "singlecell_motif_analysis_singles",
    "singlecell_motif_analysis_pairs",
    # The single-cell RNA stages. This list duplicates the stage graphs in
    # `pipeline.profiles.PROFILE_REGISTRY`; a test asserts every registered
    # profile's stage_graph is a subset of it.
    #
    # Order matters: validate_pipeline_state() derives execution order from
    # THIS list when the caller supplies stages as a set, so pseudobulk must
    # precede differential.
    "singlecell_rna",
    "singlecell_rna_differential",
    # The lineages and the statistics along them, after the differential.
    "trajectories",
    # Evidome, last in every expression pipeline -- so last here.
    "functional_evidence",
]
TERMINAL_STAGE_STATES = {"complete", "skipped"}
SUPERVISOR_SCHEMA = "paralome.pipeline_supervisor.v2"
LEGACY_SUPERVISOR_SCHEMAS = {
    "paralome.pipeline_supervisor.v1",
    SUPERVISOR_SCHEMA,
    # Older schema names. The payload structure is identical; only the prefix differs, so checkpoints
    # written under the old name are still accepted.
    "omicsflow.pipeline_supervisor.v1",
    "omicsflow.pipeline_supervisor.v2",
}


class SupervisorValidationError(RuntimeError):
    """Raised when the supervisor cannot prove that pipeline state is safe."""

    def __init__(self, message: str, diagnostics: Optional[list[dict[str, Any]]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class SlurmQueryUnavailable(RuntimeError):
    """The controller did not answer, so the state of the queue is UNKNOWN.

    Once squeue failed for a while (network), the supervisor
    read that failure as an empty queue, concluded that nothing was running,
    and cancelled a healthy run. A failed query and an empty queue are opposite
    facts and must never share a return value. Raised by the queue readers and
    caught in the wait loops, which then wait and cancel nothing.
    """


class SupervisorInterrupted(KeyboardInterrupt):
    """The supervisor was told to stop: SIGTERM (scancel, kill) or SIGHUP.

    A KeyboardInterrupt, so everything above this module treats it as
    the user stopping the program. It exists so the supervisor can record what
    it was waiting on before it goes.
    """

    def __init__(self, signal_name: str):
        super().__init__(signal_name)
        self.signal_name = signal_name


@dataclass
class StateValidation:
    valid: bool
    complete: bool
    next_stage: Optional[str]
    completed_stages: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineSupervisorResult:
    project: Path
    complete: bool
    stages_run: list[str]
    report_workdir: Path
    state_path: Path


def _state_fingerprint(state: Mapping[str, Any]) -> str:
    pipeline = state.get("pipeline", {}) if isinstance(state.get("pipeline"), dict) else {}
    stable = {
        "project": state.get("project", {}),
        "pipeline": {
            "complete": pipeline.get("complete"),
            "current_stage": pipeline.get("current_stage"),
            "next_stage": pipeline.get("next_stage"),
            "stages": pipeline.get("stages", {}),
        },
    }
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _diagnostic(code: str, message: str, **details: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if details:
        item["details"] = details
    return item


def validate_pipeline_state(
    state: Mapping[str, Any],
    *,
    previous_state: Optional[Mapping[str, Any]] = None,
    expected_stage: Optional[str] = None,
) -> StateValidation:
    """Validate state-machine invariants and stage-transition legality.

    The project-state builders provide observations. This function is the
    authoritative decision point that determines whether those observations
    permit execution to start, resume, or advance.
    """
    diagnostics: list[dict[str, Any]] = []
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return StateValidation(False, False, None, diagnostics=[
            _diagnostic("missing_pipeline_state", "Project state has no pipeline state-machine section.")
        ])

    stages = pipeline.get("stages")
    if not isinstance(stages, Mapping):
        # Project-state payloads that carry only complete/next_stage are still accepted; full invariants
        # apply whenever the canonical stage map is present.
        declared_complete = bool(pipeline.get("complete"))
        declared_next = pipeline.get("next_stage")
        declared_next = str(declared_next) if declared_next not in (None, "") else None
        if declared_complete and declared_next is None:
            return StateValidation(True, True, None)
        if not declared_complete and declared_next in PIPELINE_STAGE_ORDER:
            legacy_diagnostics = []
            if expected_stage is not None and declared_next == expected_stage:
                legacy_diagnostics.append(_diagnostic(
                    "stage_not_advanced",
                    f"Stage {expected_stage} returned but did not validate as complete.",
                    stage=expected_stage,
                ))
            return StateValidation(not legacy_diagnostics, False, declared_next, diagnostics=legacy_diagnostics)
        return StateValidation(False, False, declared_next, diagnostics=[
            _diagnostic("missing_stage_state", "Pipeline state has no stage map and cannot be interpreted as a compatible legacy state.")
        ])

    declared_graph = pipeline.get("stage_graph")
    if isinstance(declared_graph, list) and declared_graph:
        stage_order = [str(name) for name in declared_graph]
    else:
        stage_order = [name for name in PIPELINE_STAGE_ORDER if name in stages] or list(PIPELINE_STAGE_ORDER)

    unknown_stages = [name for name in stage_order if name not in PIPELINE_STAGE_ORDER]
    if unknown_stages:
        diagnostics.append(_diagnostic(
            "invalid_stage_graph",
            "Pipeline stage graph contains unknown stages.",
            unknown_stages=unknown_stages,
        ))
    if len(stage_order) != len(set(stage_order)):
        diagnostics.append(_diagnostic(
            "duplicate_stage_graph_entry",
            "Pipeline stage graph contains duplicate stage names.",
            stage_graph=stage_order,
        ))

    project = state.get("project", {}) if isinstance(state.get("project"), Mapping) else {}
    try:
        resolved_profile = resolve_pipeline_profile(
            project.get("pipeline_profile") or project.get("profile"),
            project.get("segregation_profile"),
        )
    except ValueError as exc:
        diagnostics.append(_diagnostic("invalid_pipeline_profile", str(exc)))
        resolved_profile = None

    if resolved_profile is not None:
        expected_graph = list(resolved_profile.stage_graph)
        if not (isinstance(declared_graph, list) and declared_graph):
            stage_order = expected_graph
        if stage_order != expected_graph:
            diagnostics.append(_diagnostic(
                "profile_stage_graph_mismatch",
                "Pipeline stage graph does not match the resolved pipeline profile.",
                pipeline_profile=resolved_profile.pipeline_profile.value,
                declared=stage_order,
                expected=expected_graph,
            ))
        metadata_checks = {
            "pipeline_profile": resolved_profile.pipeline_profile.value,
            "segregation_profile": resolved_profile.segregation_profile,
            "alignment_profile": resolved_profile.alignment_profile.value,
            "manifest_profile": resolved_profile.manifest_profile.value,
        }
        for key, expected in metadata_checks.items():
            declared = pipeline.get(key)
            if declared is not None and declared != expected:
                diagnostics.append(_diagnostic(
                    "profile_metadata_mismatch",
                    f"Pipeline metadata {key} does not match the resolved project profile.",
                    field=key,
                    declared=declared,
                    expected=expected,
                ))

    completed: list[str] = []
    first_incomplete: Optional[str] = None
    # A finished stage is out of order only when a stage it reads (profiles.STAGE_READS), or one those
    # read, is not finished -- for a profile read as a line, any earlier stage. With the trajectory
    # statistics and the differential run together, either may finish first.
    reads = _reads_of(state, stage_order)
    unfinished: dict[str, Any] = {}
    blocked: set[str] = set()
    for name in stage_order:
        stage = stages.get(name)
        if any(blocks(read, unfinished, blocked) for read in reads.get(name, ())):
            blocked.add(name)
        if not isinstance(stage, Mapping):
            diagnostics.append(_diagnostic("missing_stage", f"Pipeline state is missing stage {name}.", stage=name))
            unfinished[name] = None
            if first_incomplete is None:
                first_incomplete = name
            continue
        stage_state = str(stage.get("state") or "").lower()
        if stage_state not in {"not_started", "waiting", "ready", "running", "complete", "failed", "skipped"}:
            diagnostics.append(_diagnostic("invalid_stage_state", f"Stage {name} has invalid state {stage_state!r}.", stage=name, state=stage_state))
        is_terminal = stage_state in TERMINAL_STAGE_STATES
        if is_terminal:
            completed.append(name)
            if name in blocked and not bool(stage.get("optional")):
                diagnostics.append(_diagnostic(
                    "noncontiguous_completion",
                    f"Stage {name} is complete although a required stage it reads is incomplete.",
                    stage=name,
                ))
        elif not bool(stage.get("optional")):
            unfinished[name] = _parts_of(stage)
            if first_incomplete is None:
                first_incomplete = name

    declared_complete = bool(pipeline.get("complete"))
    declared_next = pipeline.get("next_stage")
    declared_next = str(declared_next) if declared_next not in (None, "") else None
    expected_next = None if declared_complete else first_incomplete

    if declared_complete and first_incomplete is not None:
        diagnostics.append(_diagnostic(
            "false_complete",
            "Pipeline is marked complete but at least one required stage is incomplete.",
            first_incomplete=first_incomplete,
        ))
    if not declared_complete and declared_next is None:
        diagnostics.append(_diagnostic("missing_next_stage", "Incomplete pipeline has no next stage."))
    if declared_next is not None and declared_next not in stage_order:
        diagnostics.append(_diagnostic("unknown_next_stage", f"Unknown next stage {declared_next!r}.", next_stage=declared_next))
    elif declared_next != expected_next:
        diagnostics.append(_diagnostic(
            "next_stage_mismatch",
            f"Declared next stage {declared_next!r} does not match the first incomplete stage {expected_next!r}.",
            declared=declared_next,
            expected=expected_next,
        ))

    if expected_stage is not None and not declared_complete and declared_next == expected_stage:
        diagnostics.append(_diagnostic(
            "stage_not_advanced",
            f"Stage {expected_stage} returned but did not validate as complete.",
            stage=expected_stage,
        ))

    if previous_state is not None:
        previous_pipeline = previous_state.get("pipeline", {}) if isinstance(previous_state.get("pipeline"), Mapping) else {}
        previous_stages = previous_pipeline.get("stages", {}) if isinstance(previous_pipeline.get("stages"), Mapping) else {}
        for name in stage_order:
            old = previous_stages.get(name, {}) if isinstance(previous_stages.get(name), Mapping) else {}
            new = stages.get(name, {}) if isinstance(stages.get(name), Mapping) else {}
            old_state = str(old.get("state") or "").lower()
            new_state = str(new.get("state") or "").lower()
            if old_state in TERMINAL_STAGE_STATES and new_state not in TERMINAL_STAGE_STATES:
                diagnostics.append(_diagnostic(
                    "completed_stage_regressed",
                    f"Previously completed stage {name} regressed to {new_state or 'unknown'}.",
                    stage=name,
                    previous=old_state,
                    current=new_state,
                ))

    return StateValidation(
        valid=not diagnostics,
        complete=declared_complete and not diagnostics,
        next_stage=declared_next,
        completed_stages=completed,
        diagnostics=diagnostics,
    )


def _load_supervisor_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise SupervisorValidationError(
            f"Supervisor checkpoint is unreadable: {path}",
            [_diagnostic("invalid_supervisor_checkpoint", str(exc), path=str(path))],
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") not in LEGACY_SUPERVISOR_SCHEMAS:
        raise SupervisorValidationError(
            f"Supervisor checkpoint has an unsupported schema: {path}",
            [_diagnostic("unsupported_supervisor_checkpoint", "Unsupported or missing checkpoint schema.", path=str(path), schema=payload.get("schema") if isinstance(payload, dict) else None)],
        )
    return payload


def _validate_restart(checkpoint: Mapping[str, Any], state: Mapping[str, Any], validation: StateValidation) -> None:
    if not checkpoint:
        return
    if bool(checkpoint.get("complete")) and not validation.complete:
        # A prior run finished with every stage complete, but the freshly
        # computed state (built by inspecting current on-disk outputs and
        # per-stage checkpoints, not by trusting this old supervisor
        # checkpoint) now shows further work is required. This is expected
        # and supported: e.g. supplying new inputs to an already-complete
        # pipeline (such as --candidate-genes on the variant profile)
        # legitimately invalidates a downstream stage (reporting) so it can
        # regenerate with the new inputs. That is a reopening, not
        # corruption, and `validation` is authoritative for what to do next.
        #
        # What *would* be unsafe is resuming when the recomputed state is
        # itself inconsistent (e.g. a required stage's outputs vanished out
        # from under a non-contiguous completion pattern) -- that is still
        # caught, because such cases surface as their own diagnostics and
        # validation.valid is False.
        if not validation.valid:
            raise SupervisorValidationError(
                "A completed supervisor checkpoint conflicts with the current pipeline outputs.",
                [
                    _diagnostic(
                        "stale_completed_checkpoint",
                        "Completed checkpoint no longer matches validated outputs, "
                        "and the recomputed state is itself inconsistent.",
                    )
                ]
                + list(validation.diagnostics),
            )
        print(
            "Note: the previous run completed, but new inputs mean additional "
            f"work is required starting at stage {validation.next_stage or '?'}. "
            "Resuming from the current validated state.",
            flush=True,
        )
    checkpoint_stage = checkpoint.get("active_stage")
    if checkpoint_stage and checkpoint_stage not in PIPELINE_STAGE_ORDER:
        raise SupervisorValidationError(
            "Supervisor checkpoint contains an unknown active stage.",
            [_diagnostic("invalid_restart_stage", "Checkpoint active stage is not recognized.", active_stage=checkpoint_stage)],
        )
    # v2 fingerprints are advisory for interrupted runs: output state may have
    # legitimately advanced after the final checkpoint write. Record staleness
    # in the next checkpoint, but decide recovery from validated outputs.


def _normalize_slurm_state(raw: str) -> str:
    """Normalize a state token returned by squeue or sacct."""
    token = raw.strip().split()[0] if raw.strip() else ""
    return token.split("+", 1)[0].upper()


def _normalize_queue_reason(raw: str) -> str:
    """Normalize squeue's or scontrol's Reason field for comparison.

    Squeue prints a pending job's reason in parentheses --
    ``(DependencyNeverSatisfied)`` -- and scontrol prints it bare. Both forms
    are accepted, so a finalizer whose dependency can never be met is
    recognised from either command rather than counted as work in progress.
    """
    text = raw.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return _normalize_slurm_state(text)


def _is_held_reason(raw: str) -> bool:
    """Whether a queue reason says the job is held and will not start by itself."""
    return "held" in raw.lower()


def _run_slurm_query(command: list[str]) -> Optional[Any]:
    """Run one SLURM query; ``None`` when it could not be run or did not answer.

    Every query has a time limit, so a command that hangs on a dead network is
    one more way of getting no answer, and the caller decides what no answer
    means.
    """
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=SLURM_QUERY_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def _query_says_job_is_gone(proc: Any) -> bool:
    """SLURM's own answer for a job that has left the controller.

    ``squeue -j`` and ``scontrol show job`` exit non-zero with "Invalid job id
    specified" once a finished job has aged out. That is an answer -- the job is
    not in the queue -- and the only non-zero exit that is one.
    """
    text = f"{getattr(proc, 'stderr', '') or ''}\n{getattr(proc, 'stdout', '') or ''}"
    return "invalid job id" in text.lower()


def _first_line(text: Optional[str], fallback: str) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else fallback


def _squeue_job_records(job_ids: set[str]) -> list[tuple[str, str, str]]:
    """Return live controller records as ``(root, state, reason)`` tuples.

    Raises ``SlurmQueryUnavailable`` when squeue gave no usable answer. An empty
    list means the controller answered and none of these jobs is in the queue.
    """
    if not job_ids:
        return []
    proc = _run_slurm_query(
        [
            "squeue",
            "-h",
            "-r",
            "-j",
            ",".join(sorted(job_ids)),
            "-o",
            # %F, the array's base job ID (the job ID itself for a
            # job that is not an array). This was %A, which for a running array
            # element is that element's OWN job ID -- so every element except
            # the one that happens to carry the base ID failed the
            # `root in job_ids` test below, and an array could be read as gone
            # from the queue while most of it was running.
            "%F|%T|%R",
        ]
    )
    if proc is None:
        raise SlurmQueryUnavailable(
            f"squeue did not answer within {SLURM_QUERY_TIMEOUT_SECONDS} seconds"
        )
    if proc.returncode != 0:
        if _query_says_job_is_gone(proc):
            # Completed jobs disappear from the live controller before the
            # supervisor polls again. Terminal resolution belongs to sacct.
            return []
        # Any other failure is NOT an empty queue. See
        # SlurmQueryUnavailable.
        raise SlurmQueryUnavailable(
            f"squeue exited {proc.returncode}: "
            + _first_line(getattr(proc, "stderr", ""), "no message")
        )

    records: list[tuple[str, str, str]] = []
    for raw in proc.stdout.splitlines():
        fields = raw.split("|", 2)
        if len(fields) < 2:
            continue
        root = fields[0].strip()
        state = _normalize_slurm_state(fields[1])
        reason = fields[2].strip() if len(fields) > 2 else ""
        if root in job_ids:
            records.append((root, state, reason))
    return records


def _scontrol_job_records(job_ids: set[str]) -> list[tuple[str, str, str]]:
    """Return controller state/reason records directly for submitted roots.

    This is a fallback for clusters where a mixed array/finalizer ``squeue -j``
    query omits a dependency-blocked child. It uses the same controller Reason
    field and preserves the established terminal-failure policy.
    """
    records: list[tuple[str, str, str]] = []
    for root in sorted(job_ids):
        proc = _run_slurm_query(["scontrol", "show", "job", "-o", root])
        if proc is None:
            raise SlurmQueryUnavailable(
                f"scontrol did not answer within {SLURM_QUERY_TIMEOUT_SECONDS} seconds"
            )
        if proc.returncode != 0:
            if _query_says_job_is_gone(proc):
                continue
            # As for squeue -- a failed query is not "no such job".
            raise SlurmQueryUnavailable(
                f"scontrol exited {proc.returncode}: "
                + _first_line(getattr(proc, "stderr", ""), "no message")
            )
        # One record per LINE. For an array root scontrol prints a
        # line for every element; this split the whole output on whitespace
        # into one dictionary, so each element overwrote the one before and
        # only the last element's state survived -- an array with a failed
        # task read as COMPLETED when the last task happened to succeed.
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            fields: dict[str, str] = {}
            for token in line.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            state = _normalize_slurm_state(fields.get("JobState", ""))
            reason = fields.get("Reason", "")
            records.append((root, state, reason))
    return records


def _working_roots(records: list[tuple[str, str, str]]) -> set[str]:
    """Roots with work that is running or is able to start.

    Narrower than "in the queue and not failed". Left out:

    * COMPLETING -- the job's script has ended and only the node's cleanup is
      left; a node stuck there can hold the state for hours.
    * PENDING on a dependency -- such a job is waiting on other jobs, and is
      work in prospect only while one of THOSE is still going. When nothing
      else is working, a job waiting on a dependency is waiting on nothing: a
      chain three jobs deep keeps its last job PENDING (Dependency) for ever
      behind a middle job whose own dependency can never be satisfied.
    """
    working: set[str] = set()
    for root, state, reason in records:
        reason_state = _normalize_queue_reason(reason)
        if state in FAILED_STATES or state in {"COMPLETED", "COMPLETING"}:
            continue
        if reason_state in FAILED_QUEUE_REASONS:
            continue
        if state == "PENDING" and reason_state == "DEPENDENCY":
            continue
        working.add(root)
    return working


def _squeue_active_roots(job_ids: set[str]) -> set[str]:
    """Return submitted root IDs with live, nonterminal allocations.

    Raises ``SlurmQueryUnavailable`` when the controller did not answer.
    """
    active: set[str] = set()
    for root, state, reason in _squeue_job_records(job_ids):
        reason_state = _normalize_queue_reason(reason)
        if state == "COMPLETED" or state in FAILED_STATES or reason_state in FAILED_QUEUE_REASONS:
            continue
        active.add(root)
    return active


def _slurm_job_root(job_id: str) -> str:
    """Return the submitted numeric root for a SLURM job or array record."""
    token = job_id.strip()
    if not token:
        return ""
    token = token.split(".", 1)[0]
    return token.split("_", 1)[0]


# sacct --format field sets, most informative first.
#
# `JobIDRaw` is not accepted by every SLURM build, so a controller that
# rejects it falls back to the shorter field set. An empty answer from sacct
# is usually slurmdbd lag, not a rejected query.
_SACCT_FORMATS: tuple[str, ...] = (
    "JobID,JobIDRaw,State,ExitCode",
    "JobID,State,ExitCode",
)
_SACCT_FORMAT_CACHE: dict[str, str] = {}


def _reset_sacct_format_cache() -> None:
    """Forget which --format this controller accepted.

    Process-global state is fine in production -- one supervisor process talks
    to one controller -- but tests share an interpreter, so a test that
    exercises the fallback would otherwise pin the narrowed format for every
    test after it.
    """
    _SACCT_FORMAT_CACHE.clear()


def _sacct_job_records(
    job_ids: set[str],
) -> tuple[dict[str, list[tuple[str, str, str]]], Optional[str]]:
    """Return ``(records grouped by submitted ID, error description or None)``.

    The error is an explicit return value, so a failed query (a broken
    ``--format``, say) can be told from a genuinely empty database (normal
    accounting lag) and reported.

    Array elements are associated with their submitted root through the
    displayed ``JobID`` prefix, because ``JobIDRaw`` is an independent child
    identifier on some SLURM installations.
    """
    grouped: dict[str, list[tuple[str, str, str]]] = {job_id: [] for job_id in job_ids}
    if not job_ids:
        return grouped, None

    formats = (_SACCT_FORMAT_CACHE["format"],) if "format" in _SACCT_FORMAT_CACHE else _SACCT_FORMATS
    proc = None
    errors: list[str] = []
    for fmt in formats:
        attempt = _run_slurm_query(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                ",".join(sorted(job_ids)),
                "--parsable2",
                f"--format={fmt}",
            ]
        )
        if attempt is None:
            # A sacct that hangs is a sacct that is not answering.
            # Reported through the same channel as a rejected query; trying the
            # narrower format against a dead database would only wait again.
            return grouped, (
                f"sacct did not answer within {SLURM_QUERY_TIMEOUT_SECONDS} seconds"
            )
        if attempt.returncode == 0:
            _SACCT_FORMAT_CACHE["format"] = fmt
            proc = attempt
            break
        detail = (attempt.stderr or "").strip().splitlines()
        errors.append(f"--format={fmt} exited {attempt.returncode}: {detail[0] if detail else 'no stderr'}")

    if proc is None:
        # Every field set was rejected. Surface it: this will not heal on its
        # own, and the caller must not mistake it for accounting lag.
        _SACCT_FORMAT_CACHE.pop("format", None)
        return grouped, "sacct query failed (" + "; ".join(errors) + ")"

    # A successful exit is not the same as a healthy one. When slurmdbd is
    # unreachable or behind, sacct commonly exits 0, prints no rows, and puts
    # the reason on stderr; that text is surfaced, even on success. It is not
    # fatal and does not shorten the grace period, because accounting lag
    # heals. getattr is used because not every stand-in for CompletedProcess
    # has a stderr attribute, and a warning must never break the poll loop.
    stderr_text = (getattr(proc, "stderr", None) or "").strip()
    if stderr_text:
        print(
            "SLURM accounting warning (sacct exited 0): " + stderr_text.splitlines()[0],
            flush=True,
        )

    for raw in proc.stdout.splitlines():
        fields = raw.split("|")
        while fields and fields[-1] == "":
            fields.pop()
        if len(fields) >= 6:
            # Compatibility with fixtures/older output that include
            # ArrayJobID and ArrayTaskID between JobIDRaw and State.
            display_id = fields[0].strip()
            raw_id = fields[1].strip()
            array_job_id = fields[2].strip()
            state = _normalize_slurm_state(fields[4])
            exit_code = fields[5].strip()
        elif len(fields) == 5:
            # Legacy layout: JobIDRaw,ArrayJobID,ArrayTaskID,State,ExitCode.
            raw_id = fields[0].strip()
            display_id = raw_id
            array_job_id = fields[1].strip()
            state = _normalize_slurm_state(fields[3])
            exit_code = fields[4].strip()
        elif len(fields) >= 4:
            display_id = fields[0].strip()
            raw_id = fields[1].strip()
            array_job_id = ""
            state = _normalize_slurm_state(fields[2])
            exit_code = fields[3].strip()
        elif len(fields) == 3:
            # the JobID,State,ExitCode fallback, for controllers that
            # reject JobIDRaw. Without this branch the narrowed query would parse
            # to nothing and look identical to the failure it exists to survive.
            display_id = fields[0].strip()
            raw_id = ""
            array_job_id = ""
            state = _normalize_slurm_state(fields[1])
            exit_code = fields[2].strip()
        else:
            continue

        if not display_id or "." in display_id:
            continue

        candidates = [_slurm_job_root(display_id)]
        if array_job_id:
            candidates.append(array_job_id)
        if raw_id:
            candidates.append(raw_id)
        root = next((candidate for candidate in candidates if candidate in job_ids), None)
        if root is None:
            continue
        grouped[root].append((display_id, state, exit_code))
    return grouped, None


def _cancel_jobs(job_ids: set[str]) -> None:
    """Cancel every submitted job in ``job_ids`` via ``scancel``.

    Called when the supervisor is about to fail a stage, so a dependent
    finalizer left in ``DependencyNeverSatisfied`` (or any other job still
    sitting in the queue behind a failed worker/array) does not linger
    requiring manual ``scancel`` cleanup. Cancelling an already-terminal job
    is a harmless no-op on SLURM, so this is safe to call unconditionally
    for the full job set rather than trying to track which ones are still
    live.
    """
    if not job_ids:
        return
    proc = _run_slurm_query(["scancel", *sorted(job_ids)])
    if proc is None:
        print(
            "Pipeline supervisor could not cancel SLURM jobs (scancel did not answer): "
            + ",".join(sorted(job_ids))
            + ". Cancel them by hand if any is still queued.",
            flush=True,
        )
        return
    print(
        # Not always after a failure (accounting deadlines too);
        # the error raised next says why.
        "Pipeline supervisor cancelled the stage's remaining SLURM jobs: "
        + ",".join(sorted(job_ids))
        + (f" ({proc.stderr.strip()})" if proc.returncode != 0 and proc.stderr.strip() else ""),
        flush=True,
    )


def stage_log_hint(workdir: Optional[Union[str, Path]], job_ids) -> str:
    """Where to read the failure, named in the failure itself.

    A stage's logs are under its own workdir, which can be outside the
    project folder (the shared data workdir is a sibling of the project), so
    the failure message names the log directory and any matching log files.
    """
    if not workdir:
        return ""
    logs = Path(workdir) / "logs"
    matches: list[str] = []
    for job_id in sorted({str(j) for j in (job_ids or ())}):
        # Escaped, so an array id such as 123_[5-20] is literal.
        matches += [str(path) for path in sorted(logs.glob(f"*{glob.escape(job_id)}*"))]
    if matches:
        return " Logs: " + ", ".join(matches[:6])
    return f" Logs expected under {logs} (none found for these job ids)."


def _wait_for_jobs(
    job_ids: set[str],
    poll_seconds: int,
    *,
    completion_probe: Optional[Callable[[], bool]] = None,
    workdir: Optional[Union[str, Path]] = None,
) -> None:
    if not job_ids:
        return
    print(
        "Pipeline supervisor waiting for SLURM jobs: "
        + ",".join(sorted(job_ids)),
        flush=True,
    )

    missing_since: dict[str, float | None] = {job_id: None for job_id in job_ids}
    # Since when sacct has shown a root as not finished while
    # neither squeue nor scontrol reports it (a runaway accounting record).
    runaway_since: dict[str, float | None] = {job_id: None for job_id in job_ids}
    # Last-printed poll lines, so the loop can stay quiet while nothing changes.
    last_poll_line: str | None = None
    last_active_line: str | None = None
    # Printed when an accounting deadline is deferred because work is
    # still running. De-duplicated like the others -- an hour-long stage must not
    # bury its own start line under sixty copies of this.
    last_deferral_line: str | None = None
    accounting_grace_seconds = 7200
    # a rejected sacct query is not accounting lag -- it will not
    # resolve by waiting, so it gets its own much shorter deadline. Kept
    # non-zero rather than failing on the first poll because slurmdbd can be
    # briefly unreachable across a restart, which does heal.
    broken_accounting_grace_seconds = 300
    sacct_error_since: float | None = None
    # State, each printed when it changes and not once a minute.
    last_controller_line: str | None = None   # the controller is not answering
    last_held_line: str | None = None         # jobs held in the queue
    last_failure_line: str | None = None      # failures waiting for the rest
    quiet_failure_polls = 0                   # polls with a failure and nothing running

    while True:
        # THE QUEUE IS EITHER KNOWN OR IT IS NOT. A squeue that fails says nothing about the jobs, so it is
        # not read as "none of them is in the queue" (which would apply every accounting deadline below and
        # cancel live work). In the unknown case the supervisor waits indefinitely: the queue comes back or
        # it does not, and a person following up on a quiet run is told why it is quiet.
        controller_known = True
        controller_error = ""
        try:
            queue_records = _squeue_job_records(job_ids)
        except SlurmQueryUnavailable as exc:
            controller_known = False
            controller_error = str(exc)
            queue_records = []
        queue_active: set[str] = set()
        queue_failures: list[str] = []
        for root, state, reason in queue_records:
            reason_state = _normalize_queue_reason(reason)
            if state in FAILED_STATES:
                queue_failures.append(f"{root}:{state}:{reason or '?'}")
            elif reason_state in FAILED_QUEUE_REASONS:
                queue_failures.append(f"{root}:{reason_state}")
            elif state != "COMPLETED":
                queue_active.add(root)

        records_by_root, sacct_error = _sacct_job_records(job_ids)
        if sacct_error is None:
            sacct_error_since = None
        elif sacct_error_since is None:
            sacct_error_since = time.monotonic()
        # Preserve the vetted squeue/sacct lifecycle, but fill one production
        # controller gap: a dependency-blocked finalizer can be omitted from a
        # mixed array/finalizer squeue query while sacct only reports PENDING.
        # Query scontrol only for nonterminal chain members missing from squeue.
        represented = {root for root, _, _ in queue_records}
        known_queue_failure = any(
            state in FAILED_STATES or _normalize_queue_reason(reason) in FAILED_QUEUE_REASONS
            for _, state, reason in queue_records
        )
        fallback_roots: set[str] = set()
        # This applies to single-job stages too: a job that fails almost
        # instantly can leave squeue before sacct reports it, and without
        # this cross-check it would wait out the 2-hour accounting grace
        # period. scontrol retains recently-finished job state for a short window
        # (MinJobAge, commonly ~5 minutes) independent of sacct/slurmdbd, so
        # querying it here for ANY root missing from this poll's squeue
        # output -- not just when there's more than one job -- resolves this
        # quickly instead of waiting out the full grace period. This is a
        # shared supervisor fix: it benefits every pipeline profile's
        # single-job SLURM stages equally, not just snATAC.
        if controller_known and not known_queue_failure:
            for root in job_ids - represented:
                records = records_by_root.get(root, [])
                if not records or any(state not in FAILED_STATES | {"COMPLETED"} for _, state, _ in records):
                    fallback_roots.add(root)
        fallback_failures: list[str] = []
        if fallback_roots:
            try:
                fallback_records = _scontrol_job_records(fallback_roots)
            except SlurmQueryUnavailable as exc:
                # squeue answered and scontrol did not: the jobs squeue left
                # out are unaccounted for, so the queue is not known after all.
                controller_known = False
                controller_error = str(exc)
                fallback_records = []
            queue_records.extend(fallback_records)
            for root, state, reason in fallback_records:
                reason_state = _normalize_queue_reason(reason)
                if state in FAILED_STATES:
                    fallback_failures.append(f"{root}:{state}:{reason or '?'}")
                elif reason_state in FAILED_QUEUE_REASONS:
                    fallback_failures.append(f"{root}:{reason_state}")
            queue_active = set()
            for root, state, reason in queue_records:
                reason_state = _normalize_queue_reason(reason)
                if state not in FAILED_STATES and reason_state not in FAILED_QUEUE_REASONS and state != "COMPLETED":
                    queue_active.add(root)

        active: set[str] = set(queue_active)
        failures: list[str] = list(queue_failures) + fallback_failures
        unresolved: list[str] = []

        for root in sorted(job_ids):
            records = records_by_root.get(root, [])
            if not records:
                if root in queue_active:
                    missing_since[root] = None
                else:
                    if missing_since[root] is None:
                        missing_since[root] = time.monotonic()
                    unresolved.append(root)
                active.add(root)
                continue

            missing_since[root] = None
            unfinished_in_accounting = False
            for record_id, state, exit_code in records:
                if state in FAILED_STATES:
                    failures.append(f"{record_id}:{state}:{exit_code or '?'}")
                elif state == "COMPLETED":
                    if exit_code and exit_code != "0:0":
                        failures.append(f"{record_id}:COMPLETED:{exit_code}")
                else:
                    active.add(root)
                    unfinished_in_accounting = True
            if controller_known and unfinished_in_accounting and root not in queue_active:
                if runaway_since[root] is None:
                    runaway_since[root] = time.monotonic()
            else:
                runaway_since[root] = None

        queue_summary = ",".join(
            f"{root}:{state}:{reason or '-'}"
            for root, state, reason in queue_records
        ) or "none"
        if not controller_known:
            queue_summary = "no answer: " + controller_error
        accounting_summary = ",".join(
            f"{record_id}:{state}:{exit_code or '-'}"
            for root in sorted(records_by_root)
            for record_id, state, exit_code in records_by_root[root]
        ) or "none"
        # Say WHICH kind of nothing this is: "the database has no rows yet" and "the query was rejected"
        # need opposite responses from whoever is reading the log.
        if sacct_error is not None:
            accounting_summary = sacct_error
        # Printed only when it changes, and always when sacct is erroring,
        # so a rejected query stays visible without repeating an unchanged
        # line every poll.
        poll_line = f"SLURM poll diagnostics: squeue=[{queue_summary}] sacct=[{accounting_summary}]"
        if poll_line != last_poll_line or sacct_error is not None:
            print(poll_line, flush=True)
            last_poll_line = poll_line

        if not controller_known:
            controller_line = (
                "SLURM is not answering (" + controller_error + "). The state of "
                + ",".join(sorted(job_ids))
                + " is unknown, so nothing is cancelled and no deadline runs. "
                "Waiting for the queue to answer again, for as long as it takes."
            )
            if controller_line != last_controller_line:
                print(controller_line, flush=True)
                last_controller_line = controller_line
        elif last_controller_line is not None:
            print("SLURM is answering again; resuming normal checks.", flush=True)
            last_controller_line = None

        # A held job will not start until somebody releases it, and
        # it counts as active for as long as it sits there. Waiting is right --
        # only a person knows whether the hold is deliberate -- but waiting in
        # silence looked exactly like a long stage.
        held = sorted({
            f"{root} ({reason.strip().strip('()')})"
            for root, state, reason in queue_records
            if state == "PENDING" and _is_held_reason(reason)
        })
        held_line = (
            "Held in the queue and not going to start by themselves: "
            + ", ".join(held)
            + ". Paralome keeps waiting; `scontrol release <job>` lets one run, "
            "`scancel <job>` gives it up."
        ) if held else None
        if held_line != last_held_line:
            if held_line is not None:
                print(held_line, flush=True)
            last_held_line = held_line

        if failures:
            # A FAILURE DOES NOT TAKE THE REST OF THE STAGE WITH IT. The failure is announced at once and the
            # tasks still running are left to finish (one out-of-memory chunk of a 1,000-chunk array must not
            # stop the 999 that are fine); the stage stops when nothing in it is running. Their results are
            # checkpointed, so the re-run repeats only what failed.
            #
            # What "running" means here is _working_roots: the controller's own
            # list of what is executing or able to start. The accounting
            # database is not asked -- a stale RUNNING record there outlives the
            # job it describes, and would hold a failed stage open for ever.
            #
            # While the controller is not answering, nothing at all is known
            # about what is running, so nothing is cancelled: the failure waits
            # with everything else until the queue can be read.
            working = _working_roots(queue_records)
            if not controller_known or working:
                quiet_failure_polls = 0
                shown = ", ".join(failures[:5]) + (
                    f" and {len(failures) - 5} more" if len(failures) > 5 else ""
                )
                failure_line = "Failed so far: " + shown + ". " + (
                    "The rest of this stage (" + ",".join(sorted(working))
                    + ") is left to finish; the stage stops after that."
                    if controller_known else
                    "SLURM is not answering, so the stage is left as it is until it does."
                )
                if failure_line != last_failure_line:
                    print(failure_line, flush=True)
                    last_failure_line = failure_line
                time.sleep(max(1, poll_seconds))
                continue

            # A node failure or a preemption may be followed by a requeue, and
            # there is a moment when the old record says NODE_FAIL and the new
            # PENDING one is not listed yet. Such a failure is believed only
            # when it is still the whole story on the next poll.
            quiet_failure_polls += 1
            may_requeue = any(
                part in REQUEUE_STATES for failure in failures for part in failure.split(":")
            )
            if may_requeue and quiet_failure_polls < 2:
                time.sleep(max(1, poll_seconds))
                continue

            # Nothing is running, so this cancels nothing that could have
            # finished. It clears what is left behind: a finalizer whose
            # dependency can never be satisfied stays in the queue forever.
            _cancel_jobs(job_ids)
            # PLM-0301: the most common failure a user sees, numbered, with the path of the .err file that
            # holds the actual error ("Read the .err file named in the message -- that is the real error.").
            raise ParalomeError(
                301,
                "SLURM stage failed: " + ", ".join(failures)
                + stage_log_hint(workdir, [f.split(":", 1)[0] for f in failures]),
                jobs=failures,
                workdir=str(workdir),
            )
        quiet_failure_polls = 0

        # ONLY ONCE THE CONTROLLER SAYS THE JOBS ARE GONE.
        #
        # This check exists for one case: the jobs have left the queue and the accounting database has not
        # caught up. Several stages read as complete as soon as their final file exists, so while the
        # stage's jobs are still running (an annotation merge still writing its VCF) this must not run.
        #
        # A job that is COMPLETING has finished its own work and is not waited
        # on here: a node that sticks in that state must not hold a finished
        # stage open.
        queue_unfinished = {
            root for root, state, reason in queue_records
            if root in queue_active and state != "COMPLETING"
        }
        if (
            completion_probe is not None
            and controller_known
            and not queue_unfinished
            and completion_probe()
        ):
            print(
                "Pipeline supervisor observed a validated stage checkpoint; "
                "continuing despite delayed SLURM accounting.",
                flush=True,
            )
            return

        # NEITHER ACCOUNTING DEADLINE MAY KILL LIVE WORK.
        #
        # Both branches below end in _cancel_jobs(job_ids) -- the WHOLE set, not
        # the unresolvable member. One job whose exit code cannot be confirmed
        # (for example a finished job while sacct is erroring) must not take
        # down the other jobs of the stage that are still running.
        #
        # The rule: while the controller still reports ANY job in this set as
        # active, the accounting deadlines do not apply. sacct answers "how did it
        # end"; squeue answers "is it still going", and the second question is the
        # only one that licenses a teardown. A degraded accounting database is a
        # reporting problem and must never be paid for with running work.
        #
        # The deadlines apply when nothing is alive (a rejected query will not
        # heal by waiting), and then there is also nothing left to cancel.
        # And not while the controller is not answering at all --
        # "squeue says nothing is alive" and "squeue says nothing" are not the
        # same statement, and only the first one licenses a teardown.
        accounting_deadline_applies = controller_known and not queue_active
        if controller_known and not accounting_deadline_applies and sacct_error is not None:
            deferral_line = (
                "SLURM accounting is unusable, but the controller still reports "
                + ",".join(sorted(queue_active))
                + " as running -- continuing to wait rather than cancelling live "
                "work. Terminal states will be confirmed from squeue and the "
                "stage's own checkpoints."
            )
            if deferral_line != last_deferral_line:
                print(deferral_line, flush=True)
                last_deferral_line = deferral_line

        # An unusable sacct gets a short deadline and an error that names the query, not the database: a
        # rejected --format will not heal by waiting.
        if (
            accounting_deadline_applies
            and sacct_error is not None
            and unresolved
            and sacct_error_since is not None
        ):
            if time.monotonic() - sacct_error_since >= broken_accounting_grace_seconds:
                _cancel_jobs(job_ids)
                raise ParalomeError(                       # PLM-0303
                    303,
                    "SLURM accounting is unusable, so the terminal state of "
                    + ",".join(sorted(unresolved))
                    + " cannot be confirmed: " + sacct_error,
                    fix="fix the sacct query or the accounting database; waiting will not resolve it",
                    unresolved=sorted(unresolved),
                    sacct_error=sacct_error,
                )

        grace = accounting_grace_seconds if sacct_error is None else broken_accounting_grace_seconds
        exhausted = [
            root
            for root in unresolved
            if missing_since[root] is not None and time.monotonic() - missing_since[root] >= grace
        ] if accounting_deadline_applies else []
        if exhausted:
            _cancel_jobs(job_ids)
            raise ParalomeError(                           # PLM-0302
                302,
                "SLURM accounting did not provide terminal records for: "
                + ",".join(exhausted)
                + (f" ({sacct_error})" if sacct_error else ""),
                jobs=exhausted,
                sacct_error=sacct_error,
            )
        # A root that accounting still shows as running or pending
        # while the controller (squeue and scontrol) has not known it for the
        # accounting grace period will never finish: refuse instead of
        # waiting for ever.
        runaway = [
            root
            for root in sorted(job_ids)
            if runaway_since[root] is not None
            and time.monotonic() - runaway_since[root] >= accounting_grace_seconds
        ] if accounting_deadline_applies else []
        if runaway:
            _cancel_jobs(job_ids)
            raise ParalomeError(                           # PLM-0302
                302,
                "SLURM accounting still reports "
                + ",".join(runaway)
                + " as not finished, but neither squeue nor scontrol has known it for "
                + f"{accounting_grace_seconds // 3600} hours: a runaway accounting record. "
                "The job is gone and its outputs did not validate.",
                fix="ask the SLURM administrator to check `sacctmgr show runawayjobs`, "
                    "then run the same command again",
                jobs=runaway,
            )

        if not active:
            break
        # Announce the active set when it changes, not once a minute: a stage that is simply still running
        # says nothing, which is what "no news" should look like.
        active_line = "Still active: " + ",".join(sorted(active))
        if active_line != last_active_line:
            print(active_line, flush=True)
            last_active_line = active_line
        time.sleep(max(1, poll_seconds))

    print(
        "Pipeline supervisor confirmed terminal SLURM success: "
        + ",".join(sorted(job_ids)),
        flush=True,
    )


def _earlier_job_ids(checkpoint: Mapping[str, Any]) -> set[str]:
    """The job ids an earlier run recorded as possibly still running."""
    return {str(value) for value in (checkpoint.get("active_job_ids") or []) if str(value).strip()}


def _wait_for_jobs_from_an_earlier_run(
    checkpoint: Mapping[str, Any],
    poll_seconds: int,
) -> None:
    """Wait for jobs an earlier, interrupted run left in the queue.

    The supervisor can stop while its jobs carry on: a closed terminal, a
    kill, an error between submitting and waiting. The supervisor checkpoint
    records the job IDs, so running the same command again does not put a
    second set of jobs on the same output files: if any of them is still
    queued or running, this waits for them and submits nothing. How they ended
    is not judged here: when they are gone the normal loop reads the stage's
    own outputs, exactly as it does on any other start, and re-runs only what
    is missing.
    """
    job_ids = _earlier_job_ids(checkpoint)
    if not job_ids:
        return
    last_line: str | None = None
    waited = False
    while True:
        try:
            # Anything still queued that could yet run, INCLUDING a job pending on a dependency. The
            # single-cell stages report only their gather job, which waits (afterany) on an array this
            # checkpoint never heard of. From here it cannot be known what a recorded job is waiting on, so
            # it is waited for, and nothing is cancelled. Only a dependency SLURM itself reports as never to
            # be satisfied is passed over.
            queue_records = _squeue_job_records(job_ids)
            alive = {
                root for root, state, reason in queue_records
                if state != "COMPLETED" and state not in FAILED_STATES
                and _normalize_queue_reason(reason) not in FAILED_QUEUE_REASONS
            }
            # As in _wait_for_jobs, a dependency-blocked job that
            # squeue leaves out is looked up with scontrol before it is read
            # as gone.
            missing = job_ids - {root for root, _, _ in queue_records}
            if missing:
                for root, state, reason in _scontrol_job_records(missing):
                    if (
                        state != "COMPLETED" and state not in FAILED_STATES
                        and _normalize_queue_reason(reason) not in FAILED_QUEUE_REASONS
                    ):
                        alive.add(root)
            line = (
                "Jobs from the earlier run of this project are still in the queue: "
                + ",".join(sorted(alive))
                + ". Waiting for them to finish rather than submitting the stage again."
                " (`scancel` any that should not run.)"
            ) if alive else None
        except SlurmQueryUnavailable as exc:
            alive = job_ids
            line = (
                f"SLURM is not answering ({exc}), so it is not known whether the "
                "earlier run's jobs (" + ",".join(sorted(job_ids)) + ") are still "
                "going. Waiting for the queue to answer before submitting anything."
            )
        if not alive:
            break
        waited = True
        if line != last_line:
            print(line, flush=True)
            last_line = line
        time.sleep(max(1, poll_seconds))
    if waited:
        print("The earlier run's jobs have left the queue; continuing.", flush=True)


class _StopSignals:
    """Turn SIGTERM and SIGHUP into SupervisorInterrupted while supervising.

    Without a handler either signal ends the process on the spot: no
    checkpoint, no word about the jobs it was waiting on. Ctrl-C already
    arrives as KeyboardInterrupt and needs nothing here. The previous handlers
    are put back on the way out, and nothing is installed off the main thread,
    where Python does not allow it.
    """

    # SIGINT as well: left to Python it would raise KeyboardInterrupt at once, in the middle of a
    # submission, which is the one moment hold() exists to protect.
    _SIGNALS = ("SIGTERM", "SIGHUP", "SIGINT")

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}
        self._holding = False
        self._held: Optional[str] = None

    def __enter__(self) -> "_StopSignals":
        if threading.current_thread() is not threading.main_thread():
            return self
        for name in self._SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue
            # A signal the caller has chosen to ignore stays ignored: `nohup`
            # works by ignoring SIGHUP, and taking it over here would make a
            # nohup'd run stop on the very hangup it was protected from.
            if signal.getsignal(number) is signal.SIG_IGN:
                continue
            try:
                self._previous[number] = signal.signal(number, self._handle)
            except (ValueError, OSError):
                continue
        return self

    def _handle(self, number: int, frame: Any) -> None:
        name = signal.Signals(number).name
        if self._holding:
            # See hold(). The stop is honoured as soon as the stage command
            # has returned and its job IDs are on record.
            if self._held is None:
                print(
                    f"\nStop requested ({name}). Finishing the stage command in "
                    "progress first, so the jobs it submits are recorded.",
                    flush=True,
                )
            self._held = name
            return
        if name == "SIGINT":
            raise KeyboardInterrupt
        raise SupervisorInterrupted(name)

    @contextmanager
    def hold(self):
        """Postpone a stop while a stage command is submitting its jobs.

        A stage command makes several sbatch calls and prints the job IDs when
        it has made them all. Stopped part way through, it would leave some
        jobs submitted and none of their IDs reported, and the next run would
        submit the stage again beside them. Call ``release()`` afterwards.
        """
        self._holding = True
        try:
            yield
        finally:
            self._holding = False

    def release(self) -> None:
        """Raise the stop that arrived during ``hold()``, if one did."""
        if self._held is not None:
            name, self._held = self._held, None
            if name == "SIGINT":
                raise KeyboardInterrupt
            raise SupervisorInterrupted(name)

    def __exit__(self, *exc_info: Any) -> None:
        for number, previous in self._previous.items():
            try:
                signal.signal(number, previous)
            except (TypeError, ValueError, OSError):
                pass


def _reads_of(state: Mapping[str, Any], order: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """What each stage of ``order`` reads -- the profile's table, else a line (profiles.stage_reads)."""
    project = state.get("project", {}) if isinstance(state.get("project"), Mapping) else {}
    try:
        profile = resolve_pipeline_profile(project.get("pipeline_profile") or project.get("profile"),
                                           project.get("segregation_profile")).pipeline_profile
    except (ValueError, KeyError):
        return {name: tuple(list(order)[index - 1:index]) for index, name in enumerate(order)}
    return stage_reads(profile, list(order))


def _parts_of(stage: Mapping[str, Any]) -> Optional[frozenset]:
    """What of an unfinished stage is to run -- None (all of it) or the parts the state names."""
    parts = stage.get("unfinished_parts")
    return frozenset(str(part) for part in parts) if parts else None


def _tandem_stages(state: Mapping[str, Any], active: str) -> list[str]:
    """The stages to submit alongside ``active`` -- every other required stage not finished whose reads
    are (the state's ready_stages, from profiles.STAGE_READS). None for a profile read as a line."""
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return []
    stages = pipeline.get("stages") if isinstance(pipeline.get("stages"), Mapping) else {}
    return [
        str(name) for name in (pipeline.get("ready_stages") or [])
        if str(name) != active and isinstance(stages.get(name), Mapping) and not stages[name].get("optional")
    ]


def _stages_behind_an_incomplete_one(state: Mapping[str, Any]) -> list[str]:
    """Finished stages that read a required stage which is not finished.

    In graph order. These are stale: whatever they hold was made from
    the earlier stage's previous output. Optional stages are left alone.
    A stage that reads it, or reads one that does (profiles.STAGE_READS); for a profile read as a line,
    every stage after it. A trajectory fit run again leaves the differential as it is.
    """
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, Mapping) or not isinstance(pipeline.get("stages"), Mapping):
        return []
    stages = pipeline["stages"]
    graph = pipeline.get("stage_graph")
    order = [str(name) for name in graph] if isinstance(graph, list) and graph else list(stages)
    reads = _reads_of(state, order)
    stale: list[str] = []
    unfinished: dict[str, Any] = {}
    blocked: set[str] = set()
    for name in order:
        stage = stages.get(name)
        if any(blocks(read, unfinished, blocked) for read in reads.get(name, ())):
            blocked.add(name)
        if not isinstance(stage, Mapping) or bool(stage.get("optional")):
            continue
        if str(stage.get("state") or "").lower() in TERMINAL_STAGE_STATES:
            if name in blocked:
                stale.append(name)
        else:
            unfinished[name] = _parts_of(stage)
    return stale


def _finished_stages_left_out_behind_a_rerun(state: Mapping[str, Any]) -> list[str]:
    """Finished stages this run leaves out that come after one it runs again.

    A stage past --stop-at, or switched off by an option, is not
    run, so it was never marked either: `--stop-at alignment --loose` aligned
    again, and the next full run found quantification and the statistics
    "complete" -- made from the alignment that had just been replaced. They
    are put on the stale list, where the next run that includes them finds
    them.
    """
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, Mapping) or not isinstance(pipeline.get("stages"), Mapping):
        return []
    stages = pipeline["stages"]
    order = _stage_graph_of(state)
    reads = _reads_of(state, order)          # Behind = reading it (a line: after it)
    left_out: list[str] = []
    unfinished: dict[str, Any] = {}
    blocked: set[str] = set()
    for name in order:
        stage = stages.get(name)
        if any(blocks(read, unfinished, blocked) for read in reads.get(name, ())):
            blocked.add(name)
        if not isinstance(stage, Mapping):
            continue
        if bool(stage.get("optional")):
            if name in blocked and bool(stage.get("complete")):
                left_out.append(name)
        elif str(stage.get("state") or "").lower() not in TERMINAL_STAGE_STATES:
            unfinished[name] = _parts_of(stage)
    return left_out


def _stage_graph_of(state: Mapping[str, Any]) -> list[str]:
    pipeline = state.get("pipeline")
    if not isinstance(pipeline, Mapping) or not isinstance(pipeline.get("stages"), Mapping):
        return []
    graph = pipeline.get("stage_graph")
    return [str(name) for name in graph] if isinstance(graph, list) and graph else list(pipeline["stages"])


def _stage_entry(state: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    pipeline = state.get("pipeline")
    stages = pipeline.get("stages") if isinstance(pipeline, Mapping) else None
    entry = stages.get(stage) if isinstance(stages, Mapping) else None
    return entry if isinstance(entry, Mapping) else {}


def _stage_workdir(state: Mapping[str, Any], stage: str) -> Optional[str]:
    summary = state.get(stage)
    workdir = summary.get("workdir") if isinstance(summary, Mapping) else None
    return str(workdir) if workdir else None


def _recorded_stale_stages(checkpoint: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The stale list an earlier run left in its checkpoint.

    One record per stage: why it is stale, whether its command has
    been issued again and, if so, the jobs that submitted and the folder they
    write to. A stage leaves the list only when its re-run has finished and
    validated, so a stop or a failure in between cannot turn its old outputs
    back into a finished stage.

    The list is written once a stage command has been issued, not before:
    until then nothing on disk has moved, and the next start works the list
    out again for the request it is given. A run that is refused, or stopped
    while its plan is on the screen, therefore leaves no mark behind.
    """
    recorded = checkpoint.get("stale_stages")
    if not isinstance(recorded, Mapping):
        return {}
    stale: dict[str, dict[str, Any]] = {}
    for name, record in recorded.items():
        record = record if isinstance(record, Mapping) else {}
        stale[str(name)] = {
            "reason": str(record.get("reason") or "recorded by an earlier run"),
            "issued": bool(record.get("issued")),
            "job_ids": sorted(str(value) for value in (record.get("job_ids") or []) if str(value).strip()),
            "workdir": str(record["workdir"]) if record.get("workdir") else None,
            # A part of the stage only (profiles.STAGE_PARTS), or the whole of it.
            "part": str(record["part"]) if record.get("part") else None,
        }
    return stale


def _stale_entries(stale: Mapping[str, Any]) -> tuple[str, ...]:
    """The stale list as the state reads it -- "stage", or "stage.part" for a part of it."""
    return tuple(
        f"{name}.{record['part']}" if isinstance(record, Mapping) and record.get("part") else str(name)
        for name, record in stale.items()
    )


def _recorded_jobs_succeeded(job_ids: set[str], poll_seconds: int, attempts: int = 5) -> tuple[bool, str]:
    """Whether jobs that have left the queue all ended in success.

    Asked of the accounting database, a few times over because it
    runs behind the queue. Success is every job on record as COMPLETED with
    exit 0:0. Anything else -- a failure, no record, no answer -- is "no", with
    the reason: the caller then runs the stage again rather than trust it.
    """
    why = "nothing was recorded"
    for attempt in range(max(1, attempts)):
        records, error = _sacct_job_records(job_ids)
        failed: list[str] = []
        unsettled: list[str] = []
        for root in sorted(job_ids):
            rows = records.get(root) or []
            if not rows:
                unsettled.append(root)
            for record_id, state, exit_code in rows:
                if state == "COMPLETED" and (not exit_code or exit_code == "0:0"):
                    continue
                if state in FAILED_STATES or state == "COMPLETED":
                    failed.append(f"{record_id}:{state}:{exit_code or '?'}")
                else:
                    unsettled.append(f"{record_id}:{state}")
        if failed:
            return False, "failed: " + ", ".join(failed[:5])
        if not unsettled:
            return True, ""
        why = "SLURM accounting has no final record for " + ", ".join(unsettled[:5]) + (
            f" ({error})" if error else ""
        )
        if attempt + 1 < max(1, attempts):
            time.sleep(max(1, poll_seconds))
    return False, why


def _current_version() -> str:
    from paralome.version import __version__

    return str(__version__)


def _changed_requests(
    state: Mapping[str, Any],
    requests: dict[str, Any],
    stage_kwargs: Mapping[str, Any],
    project: Optional[Path],
    say: Callable[[str], None] = print,
) -> dict[str, str]:
    """The stages whose recorded request differs from this run's, each with how.

    A stage whose request differs only in the options of one of its parts (profiles.STAGE_PARTS) is in
    the result's ``parts`` with that part: only that part is to run again.

    Once a stage read complete, a changed option did nothing: a new
    --factor, a different annotation, other STAR settings, a sample commented
    out. Each stage's request is recorded when its command is issued; here the
    command is built again for the present request and compared. This is done
    at every start and from the files as they are, so an option that was
    changed and then put back is no change.

    Stages are taken in order up to and including the first unfinished one --
    past that, a stage's inputs are not settled and its command cannot be
    trusted to build the same way. ``state`` is the state as read from disk,
    without the stale list applied. A stage with no record (run by an earlier
    version, or by hand) is left alone, and so is one the sample sheet leaves
    out. A command that cannot be built is reported and left alone: not
    knowing is no reason to call days of work stale.

    A record written by another Paralome is not compared either (except where it differs only in a
    part's own options, which the part runs again for). A release
    that builds a stage's command differently would otherwise make every
    finished stage of every project read as changed. Such a record is replaced
    by this version's, taken as the same request, and compared from then on.
    So is one written under other rules by a build with the same version
    string: the record carries the number of the rules it was written by
    (orchestrator.REQUEST_RECORD_FORMAT), and one with another number or none
    read as "--gtf /abs/genes.gtf -> genes.gtf (10 bytes)", a refusal with no
    option to put back.
    """
    version = _current_version()
    adopted: list[str] = []
    adopted_but_differ: dict[str, str] = {}
    changed = _ChangedRequests()
    for name in _stage_graph_of(state):
        entry = _stage_entry(state, name)
        if not entry:
            break
        if bool(entry.get("optional")) or _not_applicable(state, name):
            continue
        recorded = requests.get(name)
        if isinstance(recorded, Mapping) and recorded.get("fingerprint"):
            try:
                _, command = build_stage_command(name, **{**dict(stage_kwargs), "stale_stages": ()})
                current = stage_request_record(
                    name, command, project=project, samples=stage_kwargs.get("samples"),
                    remove_samples=tuple(stage_kwargs.get("remove_samples") or ()),
                ) if command else None
            except Exception as exc:
                say(
                    f"Note: could not work out whether the request for {name} has changed "
                    f"({_first_line(str(exc), type(exc).__name__)}); its results are kept as they are."
                )
                current = None
            if current is not None and (
                str(recorded.get("paralome_version") or "") != version
                or recorded.get("format") != REQUEST_RECORD_FORMAT
            ):
                # A record of the same format can still be compared;
                # a difference is not acted on (it may be only how this version
                # builds the command), but it is reported, not swallowed.
                differs = (recorded.get("format") == REQUEST_RECORD_FORMAT
                           and request_records_differ(recorded, current, project))
                # A difference only in a part's own options (profiles.STAGE_PARTS: the trajectory fit) is the
                # run's own choice, not how a release builds the command -- that part runs again, and keeps
                # itself if its own record says nothing it is made from changed. The record is left for the
                # part's run to write.
                part = _changed_part(name, recorded, current, project) if differs else None
                if part:
                    changed[name] = describe_request_change(recorded, current, project)
                    changed.parts[name] = part
                else:
                    if differs:
                        adopted_but_differ[name] = describe_request_change(recorded, current, project)
                    requests[name] = {**current, "paralome_version": version}
                    adopted.append(name)
            elif current is not None and request_records_differ(recorded, current, project):
                changed[name] = describe_request_change(recorded, current, project)
                part = _changed_part(name, recorded, current, project)
                if part:
                    changed.parts[name] = part
        if str(entry.get("state") or "").lower() not in TERMINAL_STAGE_STATES:
            break
    if adopted:
        say(
            "Note: the requests on record for " + ", ".join(adopted) + " were written by another "
            f"Paralome version, or by an earlier build of this one, and cannot be compared with this one's. "
            f"They are recorded afresh for {version} and the results are kept; "
            "a change of options is noticed from the next run on."
        )
    for name, how in adopted_but_differ.items():
        say(
            f"Warning: the request for {name} differs from the one on record ({how}). "
            "The record was written by another Paralome version, so this may be only how the "
            f"command is now built, and the {name} results are kept. If you changed these "
            f"options in this run, remove the {name} stage folder and run again so they take effect."
        )
    return changed


class _ChangedRequests(dict):
    """{stage: how its request changed}, and ``parts``: {stage: the one part whose options are all that changed}."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: dict[str, str] = {}


def _changed_part(stage: str, recorded: Mapping[str, Any], current: Mapping[str, Any],
                  project: Optional[Path]) -> Optional[str]:
    """The one part of ``stage`` whose options are all that changed (profiles.STAGE_PARTS), or None."""
    own_parts = STAGE_PARTS.get(stage) or {}
    if not own_parts:
        return None
    own = {option for options in own_parts.values() for option in options}

    def without(record: Mapping[str, Any]) -> dict[str, Any]:
        return {**dict(record), "options": {key: value for key, value in (record.get("options") or {}).items()
                                            if key not in own}}

    if request_records_differ(without(recorded), without(current), project):
        return None
    before, after = recorded.get("options") or {}, current.get("options") or {}
    differing = [part for part, options in own_parts.items()
                 if any(before.get(option) != after.get(option) for option in options)]
    return differing[0] if len(differing) == 1 else None


def _not_applicable(state: Mapping[str, Any], stage: str) -> bool:
    """Whether the sample sheet leaves this stage out (a sheet of BAMs is not trimmed or aligned)."""
    summary = state.get(stage)
    return isinstance(summary, Mapping) and str(summary.get("status") or "") == "not_applicable"


def _evidence_skip_to_ask_again(state: Mapping[str, Any]) -> Optional[str]:
    """Why a skipped functional-evidence stage should be tried again, or None.

    The stage is skipped when Evidome, or its data for this species, is not
    installed, and "skipped" must not read as finished for ever. Whether the
    program is there now can be checked here. Whether its data is there only
    Evidome can say, so for that kind of skip it is asked again; it answers at
    once.
    """
    entry = _stage_entry(state, "functional_evidence")
    summary = state.get("functional_evidence")
    if not entry or bool(entry.get("optional")) or not isinstance(summary, Mapping):
        return None
    if str(summary.get("status") or "") != "skipped" or not summary.get("workdir"):
        return None
    try:
        recorded = json.loads(stage_checkpoint_path(summary["workdir"], "functional_evidence").read_text())
        from paralome.binaries import resolve_tool

        installed = bool(resolve_tool("evidome"))
    except Exception:
        return None
    if not installed or not isinstance(recorded, Mapping):
        return None
    # The checkpoint says which kind of skip it was (paralome.evidence). One
    # written before it did is read as it was: a skip with one of Evidome's
    # codes was for its data, any other for the program.
    from paralome.evidence import SKIP_DATA_MISSING, SKIP_NO_TABLE, SKIP_NOT_INSTALLED

    kind = recorded.get("skip_reason")
    if kind == SKIP_NO_TABLE:
        return None          # installing Evidome changes nothing: there is no table to give it
    if kind not in (SKIP_DATA_MISSING, SKIP_NOT_INSTALLED):
        kind = SKIP_DATA_MISSING if recorded.get("evidome_code") else SKIP_NOT_INSTALLED
    if kind == SKIP_DATA_MISSING:
        code = f" ({recorded['evidome_code']})" if recorded.get("evidome_code") else ""
        return f"it was skipped because Evidome was not set up for this input{code}; Evidome is asked again"
    return "it was skipped because Evidome was not installed, and it is now"


#: The options of a run that name what it reads. Whatever one of them names
#: is never a stage's to be told to remove.
_INPUT_OPTIONS = (
    "samples", "rds", "reference", "star_genome", "gtf", "regions",
    "fastp_adapter_fasta", "annovar_dir", "annovar_db", "snpeff_jar", "snpeff_config",
)


def _run_inputs(stage_kwargs: Mapping[str, Any]) -> list[str]:
    """Where this run's inputs are: the options above and the files the sheet lists."""
    named = [stage_kwargs.get(option) for option in _INPUT_OPTIONS]
    if stage_kwargs.get("samples"):
        try:
            from paralome.fastq.manifest import read_sample_inputs

            for row in read_sample_inputs(stage_kwargs["samples"]):
                named += [row.bam, row.fastq_r1, row.fastq_r2]
        except Exception:
            pass                             # an unreadable sheet is reported by the run itself
    return [os.path.realpath(str(value)) for value in named if value]


def _is_in(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder + os.sep)


_OUTSIDE_THE_PROJECT = "is outside the project folder"


def _is_named_inside(folder: str, project: Path) -> bool:
    """Whether ``folder``, as it is spelt, is a path inside the project.

    `<project>/star` that is a link to scratch is the alignment's
    own folder: the project names it. Followed to where it leads it read as
    "outside the project", like a folder given with --align-workdir.
    """
    spelt = os.path.abspath(folder)
    through_its_parent = os.path.join(os.path.realpath(os.path.dirname(spelt)), os.path.basename(spelt))
    roots = (os.path.abspath(str(project)), os.path.realpath(str(project)))
    return any(_is_in(path, root) and path != root for path in (spelt, through_its_parent) for root in roots)


def _why_its_folder_is_not_named(
    state: Mapping[str, Any],
    stage: str,
    project: Optional[Path],
    inputs: tuple[str, ...] = (),
    going: tuple[str, ...] = (),
) -> Optional[str]:
    """Why nobody is told to remove ``stage``'s folder, or None when they may be.

    PLM-0308 NAMES A FOLDER FOR REMOVAL ONLY IF IT IS THAT STAGE'S
    OWN, INSIDE THE PROJECT. It named the summarizer's folder whatever that
    was, and for the single-cell aggregation that is the study's data folder:
    outside the project, holding the extraction and the other modality, shared
    by every comparison. Not named are a folder outside the project, the
    project folder itself, one that holds the results of an earlier stage
    (``going``: the stages told to remove theirs in the same message), and one
    that holds an input of this run. Stages after this one are stale whenever
    it is, so a folder shared with one of those may go.

    A folder the project names that is a link to somewhere else is the
    stage's own. What it leads to is held to the same rules: not the project
    or a folder around it, no earlier stage's results, no input.
    """
    folder = _stage_workdir(state, stage)
    if not folder:
        return "is not known"
    real = os.path.realpath(folder)
    root = os.path.realpath(str(project)) if project is not None else None
    outside = root is None or not (_is_in(real, root) or _is_named_inside(folder, project))
    shared = None
    graph = _stage_graph_of(state)
    if real == root:
        shared = "is the project folder itself"
    elif root is not None and _is_in(root, real):
        shared = "holds the project folder" if outside else "is a link to a folder that holds the project folder"
    else:
        for other in graph[:graph.index(stage)] if stage in graph else graph:
            theirs = _stage_workdir(state, other)
            if other != stage and other not in going and theirs and _is_in(os.path.realpath(theirs), real):
                shared = f"also holds {other}"
                break
        held = [value for value in inputs if _is_in(value, real)]
        if shared is None and held:
            shared = f"holds an input of this run ({held[0]})"
    if outside:
        # Known to be shared as well: said, so that it is not named whole.
        return _OUTSIDE_THE_PROJECT + (f" and {shared}" if shared else "")
    return shared


def _holds_something(folder: str) -> bool:
    """Whether a stage folder is there with anything in it.

    An emptied folder is a removed one. Where the folder is a link
    to scratch, emptying it is what the person is told to do.
    """
    try:
        with os.scandir(folder) as entries:
            return any(True for _ in entries)
    except OSError:
        return False


def _alignment_results(folder: Path) -> list[str]:
    """What `align star` and the state reader go by in an alignment folder.

    The per-sample BAMs and indexes (a sample whose BAM and index are in
    place is not aligned again, whatever else is gone), the stage checkpoint
    and the reports under results/. By name, so that in a folder shared with
    something else nothing of that is listed.
    """
    found = sorted(str(path) for path in (folder / "bam").glob("*.sorted.bam*"))
    checkpoint = stage_checkpoint_path(folder, "alignment")
    if checkpoint.exists():
        found.append(str(checkpoint))
    return found + sorted(str(path) for path in (folder / "results").glob("star_alignment_*.json"))


#: What a stage keeps in a folder that cannot be named for removal,
#: for the stages where that can be listed: the files, and how the message
#: names them. For a stage not listed here the files cannot be told from what
#: else may lie in the folder.
_RESULTS_OF_A_STAGE: dict[str, tuple[Callable[[Path], list[str]], str]] = {
    "alignment": (
        _alignment_results,
        "the per-sample BAMs and their indexes ({folder}/bam/*.sorted.bam and .bai), and its records "
        "({folder}/checkpoints/alignment.json and {folder}/results/star_alignment_*.json)",
    ),
}


def _files_a_stage_is_judged_by(state: Mapping[str, Any], stage: str) -> list[str]:
    """The files in ``stage``'s folder that the state reader found and judges it by."""
    folder = _stage_workdir(state, stage)
    found: list[str] = []
    if not folder:
        return found
    real = os.path.realpath(folder)

    def look(value: Any) -> None:
        if isinstance(value, Mapping) and "path" in value and "exists" in value:
            path = str(value.get("path") or "")
            if path and value.get("exists") and path not in found and _is_in(os.path.realpath(path), real):
                found.append(path)
        elif isinstance(value, Mapping):
            for item in value.values():
                look(item)
        elif isinstance(value, list):
            for item in value:
                look(item)

    look(state.get(stage))
    return found


def _what_there_is_to_remove(
    state: Mapping[str, Any],
    stage: str,
    project: Optional[Path],
    inputs: tuple[str, ...] = (),
    going: tuple[str, ...] = (),
) -> tuple[list[str], Optional[str]]:
    """What of ``stage`` is there to be removed, and why that is not its folder.

    One answer for the message (PLM-0308), for the test of whether
    the person has done it (_forget_what_is_gone) and for `status`. An empty
    list is a stage with nothing left to keep.

    The folder, when it may be named. Otherwise what the stage itself keeps in
    it, where that can be listed (_RESULTS_OF_A_STAGE). Otherwise, for a
    folder outside the project, the folder again: all of it is what the stage
    keeps, and it is named to be emptied. Only then the files the state reader
    judges the stage by (removing only those, an alignment would find its old
    BAMs, align nothing and read as done afresh).
    """
    why_not = _why_its_folder_is_not_named(state, stage, project, inputs, going)
    folder = _stage_workdir(state, stage)
    if why_not is None or (why_not == _OUTSIDE_THE_PROJECT and stage not in _RESULTS_OF_A_STAGE):
        return ([str(folder)] if _holds_something(str(folder)) else []), why_not
    if folder and stage in _RESULTS_OF_A_STAGE:
        return _RESULTS_OF_A_STAGE[stage][0](Path(folder)), why_not
    return _files_a_stage_is_judged_by(state, stage), why_not


def _refuse_stale_stages_that_would_keep_their_result(
    state: Mapping[str, Any],
    stale: Mapping[str, Mapping[str, Any]],
    changed: Mapping[str, str],
    *,
    project: Optional[Path] = None,
    inputs: tuple[str, ...] = (),
    put_back: frozenset = frozenset(),
) -> None:
    """Stop before anything is submitted if a stale stage would keep what it has.

    THE SUPERVISOR NEVER ADDS --overwrite. A stage that does not
    check its own result against its request and inputs, run again over what
    it wrote, keeps it; made to overwrite, it would destroy results nobody
    asked to replace. So it is not touched: its folder has to go first, and
    that is for the person to do. Every such stage is named at once, with why
    and its folder, so the command is run again once and not once per stage.
    ``state`` has the stale list applied; ``changed`` is _changed_requests';
    ``put_back`` is the stages that are stale only because of a changed
    request (_work_out_what_is_stale).
    """
    graph = _stage_graph_of(state)
    blocked = [
        name for name in graph
        if name in stale and name not in SELF_CHECKING_STAGES
        and not bool(_stage_entry(state, name).get("optional"))
    ]
    if not blocked:
        return
    lines = [
        "These stages hold results that no longer answer this run, and Paralome does not "
        "replace a stage's results on its own account:"
    ]
    named: list[str] = []
    only_folders = True
    for name in blocked:
        what, why_not = _what_there_is_to_remove(state, name, project, inputs, tuple(blocked))
        named += [item for item in what if item not in named]
        why = (
            f"its request changed: {changed[name]}" if name in changed
            else str(stale[name].get("reason") or "an earlier stage has to run again")
        )
        # Marked stale and read as finished from its files, or never finished.
        finished = bool(_stage_entry(state, name).get("stale"))
        folder = _stage_workdir(state, name)
        if why_not is None:
            real = os.path.realpath(str(folder))
            also = [
                other for other in graph
                if other not in blocked and other != name and _stage_workdir(state, other)
                and Path(str(_stage_workdir(state, other))).exists()
                and _is_in(os.path.realpath(str(_stage_workdir(state, other))), real)
            ]
            if _is_in(real, os.path.realpath(str(project))):
                line = f"{name}: remove {folder}"
            else:
                # The stage's own folder, kept elsewhere through a link.
                only_folders = False
                line = (
                    f"{name}: remove everything in {folder} (a link to {real}; its contents are what "
                    "has to go, the link may stay)"
                )
            line += (
                f" (that folder also holds {', '.join(also)}, which no longer answers this run either)" if also else ""
            )
        elif name in _RESULTS_OF_A_STAGE and folder:
            # What the stage itself goes by, and nothing else of the folder.
            only_folders = False
            unknown = ", where Paralome cannot know what else there is" if why_not.startswith(_OUTSIDE_THE_PROJECT) else ""
            line = (
                f"{name}: its folder {folder} {why_not}{unknown}, so Paralome does not name it for removal. "
                f"What {name} keeps in it has to be removed before the same command is run again: "
                + _RESULTS_OF_A_STAGE[name][1].format(folder=folder)
                + ("" if what else " (none of them is left there)")
            )
        elif why_not == _OUTSIDE_THE_PROJECT and folder:
            only_folders = False
            line = (
                f"{name}: its folder {folder} {why_not}, where Paralome cannot know what else there is. "
                f"All of that folder is what {name} keeps: everything in it has to be removed before "
                "the same command is run again"
            )
        else:
            # See _why_its_folder_is_not_named. The person is told what Paralome
            # will look for, not to remove a folder that is not this stage's alone.
            only_folders = False
            line = (
                f"{name}: its folder" + (f" {folder}" if folder else "") + f" {why_not}, so Paralome does not "
                f"name it for removal. Clear the results of {name} out of it yourself; {name} is taken as not run once "
                + ("these files of it are gone: " + ", ".join(what) if what else "none of its files is left there")
            )
        lines.append(line + ("" if finished else f" ({name} was started and did not finish)") + f" -- {why}")
    lines.append(
        ("Remove the folders named" if only_folders else "Remove what is named")
        + ", then run the same command again."
    )
    # Only for the stages it is true of, not whenever a blocked stage's request changed while others were
    # stale for another reason.
    origins = [name for name in graph if name in changed and name in put_back]
    kept = [name for name in blocked if name in put_back]
    if origins and len(kept) == len(blocked):
        lines.append(
            "Or put the changed option back as it was (" + ", ".join(origins)
            + "): the results on disk answer that request and are kept."
        )
    elif origins and kept:
        lines.append(
            "Putting the changed option back as it was (" + ", ".join(origins) + ") keeps the results of "
            + ", ".join(kept) + " only. Not those of " + ", ".join(name for name in blocked if name not in kept)
            + ": whatever is put back, they no longer answer this run."
        )
    lines.append("Nothing was submitted.")
    raise ParalomeError(
        308,
        "\n".join(lines),
        fix=(
            ("remove the folders named above" if only_folders else "remove what is named above")
            + (" (or put a changed option back)" if origins and kept else "")
            + ", then run the same command again"
        ),
        stages=blocked,
        folders=named,
    )


def _take_project_lock(checkpoint_dir: Path) -> Optional[Any]:
    """Hold the project for one supervisor; refuse a second.

    Two supervisors on one project (a nohup'd run whose terminal was
    lost, started again) both waited for the first one's jobs, then both
    submitted the next stage. The lock is flock on a file that is never
    deleted: the kernel drops it when the process ends, however it ends, so
    there is no stale lock to clear. A filesystem that cannot lock is said so
    and the run goes ahead without one.
    """
    path = checkpoint_dir / "pipeline_supervisor.lock"
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+")
    except OSError as exc:
        print(f"Note: no lock could be taken on this project ({exc}); a second run on it would not be noticed.", flush=True)
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            try:
                handle.seek(0)
                holder = json.loads(handle.read() or "{}")
            except Exception:
                holder = {}
            handle.close()
            holder = holder if isinstance(holder, dict) else {}
            raise ParalomeError(
                310,
                "Another Paralome supervisor is already running on this project: "
                f"host {holder.get('host') or '?'}, pid {holder.get('pid') or '?'}, "
                f"started {holder.get('started') or '?'} (lock file {path}). "
                "Two runs on one project would submit the same stage twice. Nothing was submitted. "
                "If that host is down, or that process no longer exists there, and the lock is "
                "still held (a network filesystem can keep one), the lock file may be removed.",
                fix="wait for that run, or stop it (kill the pid on that host), then run the same command again",
                lock=str(path),
            ) from None
        handle.close()
        print(
            f"Note: this filesystem does not support file locks ({exc}); "
            "a second run on this project would not be noticed.",
            flush=True,
        )
        return None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": now_iso()}) + "\n")
        handle.flush()
    except OSError:
        pass
    return handle


def _release_project_lock(handle: Optional[Any]) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass


def _finalize_stage(stage: str, workdir: Path, threads: int, *, quiet: bool = False) -> None:
    commands = {
        "fastq_qc": [*worker_argv(), "fastq", "summarize", "--workdir", str(workdir)],
        "alignment": [*worker_argv(), "align", "verify", "--workdir", str(workdir)],
        "deduplication": [*worker_argv(), "dedup", "verify", "--workdir", str(workdir), "--threads", str(max(1, threads))],
    }
    cmd = commands.get(stage)
    if not cmd:
        return
    if not quiet:
        subprocess.run(cmd, check=True)
        return
    # A stage that ran here has already printed this report itself;
    # printed again it doubled every local run's summary. Shown only if the
    # check fails, where it is the explanation.
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        raise subprocess.CalledProcessError(done.returncode, cmd, done.stdout, done.stderr)


def _refresh_stage_state(
    *,
    state_kwargs: Mapping[str, Any],
    project: Path,
    previous_state: Mapping[str, Any],
    expected_stage: str,
    attempts: int = 12,
    delay_seconds: int = 5,
) -> tuple[dict[str, Any], Path, StateValidation]:
    """Retry post-stage state validation to tolerate checkpoint visibility races."""
    attempts = max(1, int(attempts))
    last_state: dict[str, Any] = {}
    last_path = project / "project_state.json"
    last_validation = StateValidation(False, False, expected_stage)
    for attempt in range(1, attempts + 1):
        last_state = build_project_state(**dict(state_kwargs))
        last_path = write_project_state(project, last_state)
        last_validation = validate_pipeline_state(
            last_state,
            previous_state=previous_state,
            expected_stage=expected_stage,
        )
        if last_validation.valid:
            return last_state, last_path, last_validation
        if attempt < attempts:
            print(
                f"Stage {expected_stage} validation not yet stable; "
                f"retrying ({attempt}/{attempts})",
                flush=True,
            )
            time.sleep(max(1, int(delay_seconds)))
    return last_state, last_path, last_validation


def _checkpoint_payload(
    *,
    run_id: str,
    status: str,
    complete: bool,
    stages_run: list[str],
    state: Mapping[str, Any],
    state_path: Path,
    active_stage: Optional[str] = None,
    diagnostics: Optional[list[dict[str, Any]]] = None,
    failure: Optional[str] = None,
    recovered_from: Optional[Mapping[str, Any]] = None,
    active_job_ids: Optional[set[str]] = None,
    stale_stages: Optional[Mapping[str, Any]] = None,
    stage_requests: Optional[Mapping[str, Any]] = None,
    pending_stage_requests: Optional[Mapping[str, Any]] = None,
    tandem_stages: Optional[list[str]] = None,
) -> dict[str, Any]:
    payload = {
        "schema": SUPERVISOR_SCHEMA,
        "run_id": run_id,
        "status": status,
        "complete": complete,
        "active_stage": active_stage,
        # The SLURM jobs this run submitted for the active stage and
        # has not yet seen through. The next run reads these before it submits
        # anything -- see _wait_for_jobs_from_an_earlier_run.
        "active_job_ids": sorted(active_job_ids or ()),
        # Finished stages that have to run again, each with why and
        # how far its re-run got. In every checkpoint until the re-run has
        # validated -- see _recorded_stale_stages.
        "stale_stages": {name: dict(record) for name, record in (stale_stages or {}).items()},
        # What each stage was asked to compute when its command was
        # last issued -- see _first_changed_request.
        "stage_requests": {name: dict(record) for name, record in (stage_requests or {}).items()},
        # The request of a stage submitted as one SLURM job, held
        # here until the job is known to have succeeded -- see
        # _settle_pending_requests.
        "pending_stage_requests": {name: dict(entry) for name, entry in (pending_stage_requests or {}).items()},
        "stages_run": list(stages_run),
        "project_state": str(state_path),
        "state_fingerprint": _state_fingerprint(state),
        "updated": now_iso(),
        "diagnostics": diagnostics or [],
        "failure": failure,
        "recovery": {
            "resumed": bool(recovered_from),
            "previous_run_id": recovered_from.get("run_id") if recovered_from else None,
            "previous_status": recovered_from.get("status") if recovered_from else None,
        },
    }
    if tandem_stages:
        # The stages submitted alongside the active one (profiles.STAGE_READS); their jobs are in
        # active_job_ids with its own.
        payload["tandem_stages"] = list(tandem_stages)
    return payload


def _terminal_workdir(
    state: Mapping[str, Any],
    state_kwargs: Mapping[str, Any],
    project: Path,
) -> Path:
    project_info = state.get("project", {}) if isinstance(state.get("project"), Mapping) else {}
    profile = resolve_pipeline_profile(
        state_kwargs.get("profile"),
        state_kwargs.get("segregation_profile"),
    ).pipeline_profile.value
    if profile == "ATAC":
        return Path(project_info.get("motif_analysis_workdir") or (project / "motif_analysis"))
    if profile == "snATAC":
        return Path(
            state_kwargs.get("singlecell_motif_workdir")
            or project_info.get("singlecell_motif_workdir")
            or (project / "motif_analysis")
        )
    # The folder of the last stage that produced something.
    for name in reversed(_stage_graph_of(state)):
        if bool(_stage_entry(state, name).get("complete")) and _stage_workdir(state, name):
            return Path(_stage_workdir(state, name))
    return project


def _print_restart_summary(
    state: Mapping[str, Any],
    validation: StateValidation,
    stage_kwargs: Mapping[str, Any],
) -> None:
    project_info = state.get("project", {}) if isinstance(state.get("project"), Mapping) else {}
    macs = state.get("macs_peak_calling", {}) if isinstance(state.get("macs_peak_calling"), Mapping) else {}
    dedup = state.get("deduplication", {}) if isinstance(state.get("deduplication"), Mapping) else {}
    sample_count = macs.get("manifest_rows") or macs.get("samples") or dedup.get("samples") or "."
    completed = ",".join(validation.completed_stages) or "."
    print("Pipeline execution plan", flush=True)
    print("=======================", flush=True)
    print(f"Profile               : {project_info.get('pipeline_profile') or project_info.get('profile') or '.'}", flush=True)
    print(f"Restart stage         : {validation.next_stage or '.'}", flush=True)
    print(f"Completed/skipped     : {completed}", flush=True)
    print(f"Samples               : {sample_count}", flush=True)
    print(f"Backend               : {stage_kwargs.get('backend') or '.'}", flush=True)
    print(f"Threads               : {stage_kwargs.get('atac_threads') or stage_kwargs.get('dedup_threads') or '.'}", flush=True)
    print(f"Jobs                  : {stage_kwargs.get('jobs') or '.'}", flush=True)
    print(f"Memory                : {stage_kwargs.get('mem') or '.'}", flush=True)

    # A pre-rename tree next door is the difference between reusing a 1.4 GB
    # extraction and silently redoing it. This plan block is where a person
    # looks before letting a run proceed, so it is where the warning belongs --
    # printed once, not per stage.
    data_dir = stage_kwargs.get("data_workdir_path")
    if data_dir:
        stale = stale_layout_warning(data_dir)
        if stale:
            print("", flush=True)
            print(f"WARNING: {stale}", flush=True)


def _state_kwargs_from(stage_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The state reader's arguments, taken from the stage runner's.

    For the dry run, which is handed only the latter. They share their names;
    the --factor columns are the one that differs.
    """
    wanted = inspect.signature(build_project_state).parameters
    kwargs = {name: stage_kwargs[name] for name in wanted if name in stage_kwargs}
    kwargs.pop("stale_stages", None)
    kwargs.pop("run_plan", None)
    if stage_kwargs.get("rna_factors"):
        kwargs["factors"] = tuple(stage_kwargs["rna_factors"])
    return kwargs


def stage_has_nothing_left(
    state: Mapping[str, Any],
    stage: str,
    project: Optional[Path],
    inputs: tuple[str, ...] = (),
) -> bool:
    """Whether what PLM-0308 names for ``stage`` is gone: see _what_there_is_to_remove."""
    return bool(_stage_workdir(state, stage)) and not _what_there_is_to_remove(state, stage, project, inputs)[0]


def _settle_pending_requests(
    state: Mapping[str, Any],
    pending: dict[str, Any],
    requests: dict[str, Any],
    *,
    slurm_mode: bool,
    poll_seconds: int,
    say: Callable[[str], None] = print,
) -> None:
    """Put on record the held request of a one-job stage whose job succeeded.

    A stage submitted as one SLURM job (segregation, filtering,
    reporting, gene burden) has its request held, not recorded, until its job
    is known to have succeeded. An earlier run that was stopped while the job
    ran leaves it held. Once the job has ended: if the stage reads finished
    and accounting says every job of it succeeded, the request goes on record,
    so a later change is still seen; otherwise it is dropped, as a failed
    command's is. Either way it is no longer held.
    """
    for name in list(pending):
        entry = pending.pop(name)
        record = entry.get("record") if isinstance(entry, Mapping) else None
        if not isinstance(record, Mapping):
            continue
        finished = str(_stage_entry(state, name).get("state") or "").lower() in TERMINAL_STAGE_STATES
        why = "it did not finish"
        job_ids = {str(job) for job in (entry.get("job_ids") or []) if str(job).strip()}
        # Whatever this run's backend, a job's request goes on
        # record only when accounting confirms the job succeeded.
        if finished and job_ids and not slurm_mode and shutil.which("sacct") is None:
            finished, why = False, "this machine has no sacct to ask how its job ended"
        elif finished and job_ids:
            finished, why = _recorded_jobs_succeeded(job_ids, poll_seconds)
        if finished:
            requests[name] = dict(record)
            say(f"Note: the {name} job an earlier run submitted ({', '.join(sorted(job_ids)) or 'no id'}) "
                "finished; what it was asked to compute is now on record.")
        else:
            say(f"Note: the {name} job an earlier run submitted is not known to have succeeded ({why}); "
                "nothing is on record for it.")


def _forget_what_is_gone(
    state: Mapping[str, Any],
    stale: dict[str, dict[str, Any]],
    requests: dict[str, Any],
    project: Optional[Path] = None,
    inputs: tuple[str, ...] = (),
) -> None:
    """Drop what is on record for a stage that has nothing left to keep.

    A stage whose folder has been removed is an ordinary stage that has not
    run: it is not stale, and the request it once ran with is not held
    against the one it is given now. This is done BEFORE requests are
    compared, so a person who has removed the folders PLM-0308 named is not
    refused again for the same stage. A stage the sample sheet now leaves out,
    or that this pipeline does not have, is not stale either.

    "Removed" is what PLM-0308 names for the stage and nothing else: see
    _what_there_is_to_remove.
    """
    graph = _stage_graph_of(state)
    for name in list(dict.fromkeys([*stale, *requests])):
        removed = stage_has_nothing_left(state, name, project, inputs)
        if removed:
            requests.pop(name, None)
        if removed or (graph and name not in graph) or _not_applicable(state, name):
            stale.pop(name, None)


def _differential_groups_not_the_sheets(
    state: Mapping[str, Any], samples: Optional[Union[str, Path]],
) -> Optional[str]:
    """How the groups a finished ATAC differential compared differ from the sheet's, or None.

    A relabelled sheet must not leave a result tested on its old groups
    standing as current. The groups the differential used are in its plan;
    here they are compared with the sheet's. The groups are the values of the
    --factor columns. Only samples in both are compared; who is in the test is
    the request's business. Not knowing (no plan, a sheet that cannot be read)
    is no reason to call the result stale.
    """
    summary = state.get("differential_accessibility")
    if not samples or not isinstance(summary, Mapping) or not summary.get("complete") or not summary.get("workdir"):
        return None
    plan_path = Path(str(summary["workdir"])) / "inputs" / "differential_accessibility_plan.json"
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, ValueError):
        return None
    rows = plan.get("sample_rows") if isinstance(plan, Mapping) else None
    spec = plan.get("differential") if isinstance(plan, Mapping) else None
    if not isinstance(rows, list) or not isinstance(spec, Mapping):
        return None
    # The groups are the --factor columns, so those are
    # what is compared; there is no condition column any more.
    columns = [str(f).split(":", 1)[0].strip() for f in (spec.get("factors") or ())]
    if not columns:
        return None
    try:
        import csv

        from paralome.csvio import skip_comment_lines

        with Path(samples).open(newline="") as handle:
            sheet = {
                str(row.get("sample_id") or "").strip(): row
                for row in csv.DictReader(skip_comment_lines(handle))
            }
    except Exception:
        return None
    differ = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id not in sheet:
            continue
        for column in columns:
            used = str(row.get(column) or "").strip()
            now = str(sheet[sample_id].get(column) or "").strip()
            if used and now and used != now:
                differ.append(f"{sample_id} was tested as {column}={used!r}, the sheet says {now!r}")
    if not differ:
        return None
    return "it compared groups the sample sheet no longer gives: " + "; ".join(differ[:6]) + (
        f"; and {len(differ) - 6} more" if len(differ) > 6 else "")


def _work_out_what_is_stale(
    *,
    state: dict[str, Any],
    stale: dict[str, dict[str, Any]],
    requests: dict[str, Any],
    stage_kwargs: Mapping[str, Any],
    state_kwargs: Mapping[str, Any],
    project: Optional[Path],
    say: Callable[[str], None] = print,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str], frozenset]:
    """Which stages hold results that no longer answer this run, and why.

    One function for the run and for the dry run, so both reach the
    same verdicts; it reads and writes no file. ``state`` is the state as read
    from disk; ``stale`` is the list carried from an earlier run and is added
    to. Returns the state with the list applied, the list in pipeline order,
    the stages whose request changed (see _changed_requests), and those that
    are stale for no other reason than a changed request: putting the option
    back keeps their results.

    Stale are: a stage whose request changed; a finished stage behind one that
    is not finished; functional evidence skipped for a reason that
    may be gone; and a finished stage this run leaves out behind one it runs.
    """
    graph = _stage_graph_of(state)
    for name in graph:
        summary = state.get(name)
        not_done = summary.get("samples_not_done") if isinstance(summary, Mapping) else None
        if not_done:
            # Said here, because the stage has results and would otherwise be
            # run again with no word of why.
            say(
                f"Note: {name} has not done every sample in the sheet (" + ", ".join(str(item) for item in not_done)
                + "); it is run for those, and the samples it has done are kept."
            )
    # A stage whose result was made by a method since corrected
    # (an annotation split, the segregation table, the gene-burden method)
    # reads incomplete and is run again; this says why, as the MACS3 marker's
    # reason is said in the state. Without it the person saw only that a
    # finished stage was being run again.
    for name in graph:
        summary = state.get(name)
        why = summary.get("out_of_date") if isinstance(summary, Mapping) else None
        if why and name not in stale:
            say(f"Note: {name} is run again: {why}.")
    carried = [name for name in graph if name in stale]
    if carried:
        say(
            "Note: still to be run again, from an earlier run of this project: "
            + "; ".join(f"{name} ({stale[name]['reason']})" for name in carried) + "."
        )

    put_back: set[str] = set()

    def mark_stale(name: str, reason: str, for_a_change: bool = False, part: Optional[str] = None) -> None:
        if name not in stale:
            stale[name] = {"reason": reason, "issued": False, "job_ids": [], "workdir": None, "part": part}
            if for_a_change:
                put_back.add(name)
        elif part is None and stale[name].get("part"):
            # The whole of it, not only a part.
            stale[name].update(part=None, reason=reason)

    changed = _changed_requests(state, requests, stage_kwargs, project, say)
    changed_parts: dict[str, str] = dict(getattr(changed, "parts", {}) or {})
    for name, what in changed.items():
        if name in changed_parts:
            say(f"Note: what {name} is asked to compute has changed since it ran, in its {changed_parts[name]} only "
                f"(only that part runs again, and only what reads it is stale). Changed: {what}.")
        else:
            say(f"Note: what {name} is asked to compute has changed since it ran. Changed: {what}.")
        mark_stale(name, f"its request changed: {what}", for_a_change=True, part=changed_parts.get(name))
    behind = _stages_behind_an_incomplete_one(state)
    if behind:
        first_incomplete = (state.get("pipeline") or {}).get("next_stage") or "an earlier stage"
        say(
            f"Note: {first_incomplete} has to run again, so the stages after it that "
            "were already finished no longer answer it: " + ", ".join(behind) + "."
        )
        for name in behind:
            mark_stale(name, f"{first_incomplete} has to run again")
    regroup = None if "differential_accessibility" in stale else _differential_groups_not_the_sheets(
        state, stage_kwargs.get("samples"))
    if regroup:
        say(f"Note: differential_accessibility no longer answers this run: {regroup}.")
        mark_stale("differential_accessibility", regroup)
    ask_again = None if "functional_evidence" in stale else _evidence_skip_to_ask_again(state)
    if ask_again:
        say(f"Note: functional_evidence will be run: {ask_again}.")
        mark_stale("functional_evidence", ask_again)
    if stale:
        # Whatever was finished behind a stale stage is stale with it.
        for _ in range(len(graph) + 1):
            state = build_project_state(**{**state_kwargs, "stale_stages": _stale_entries(stale)})
            further = [name for name in _stages_behind_an_incomplete_one(state) if name not in stale]
            if not further:
                break
            origin = (state.get("pipeline") or {}).get("next_stage") or "an earlier stage"
            say(
                f"Note: {origin} has to run again, so the stages after it that "
                "were already finished no longer answer it: " + ", ".join(further) + "."
            )
            for name in further:
                # Behind a stage that is stale only for its changed request.
                mark_stale(name, f"{origin} has to run again", for_a_change=origin in put_back)
    # And what this run leaves out -- see the function.
    left_out = [name for name in _finished_stages_left_out_behind_a_rerun(state) if name not in stale]
    if left_out:
        origin = (state.get("pipeline") or {}).get("next_stage") or "an earlier stage"
        say(
            f"Note: {origin} has to run again. Finished earlier and left out of this run "
            "(--stop-at, or an option that leaves a stage out): " + ", ".join(left_out)
            + ". They are held as stale for the next run that includes them."
        )
        for name in left_out:
            mark_stale(name, f"{origin} was run again by a run that left this stage out")
    if stale:
        ordered = [name for name in graph if name in stale] + [name for name in stale if name not in graph]
        stale = {name: stale[name] for name in ordered}
        if left_out:
            state = build_project_state(**{**state_kwargs, "stale_stages": _stale_entries(stale)})
        reissued = [
            name for name in graph
            if name in stale and name in SELF_CHECKING_STAGES
            and not bool(_stage_entry(state, name).get("optional"))
        ]
        if reissued:
            say(
                "Note: issued again, without --overwrite: " + ", ".join(reissued) + ". Each of these checks "
                "its own result against what it is now given, and computes again only if that no longer answers."
            )
    return state, stale, changed, frozenset(put_back)


def _say_what_the_state_reader_set_aside(state: Mapping[str, Any], say: Callable[[str], None] = print) -> None:
    """One line each for two things the state was read around."""
    alignment = state.get("alignment")
    if not isinstance(alignment, Mapping):
        return
    if alignment.get("results_not_used"):
        # A sheet of BAMs given to a project that has an alignment.
        say(
            "Note: the sample sheet lists BAM files, so those are this run's input; the alignment "
            f"already in {alignment.get('workdir')} is not used."
        )
    if alignment.get("could_not_check"):
        say(
            "Note: the alignment could not be verified on this host (samtools could not be run), "
            "so it is judged on its checkpoint and its files."
        )


def _verify_an_alignment_in_doubt(
    state: Mapping[str, Any],
    threads: int,
    say: Callable[[str], None] = print,
    run: bool = True,
) -> bool:
    """Verify the alignment once more when its report and its checkpoint disagree.

    For older projects, which could leave a report saying incomplete beside a
    finished alignment (a supervisor restarted during a retry, or `align
    verify` run by hand while the array was still going). The report decides,
    so such a project would read as not aligned and everything after it as
    stale: the report is made again, from the BAMs as they are, before
    anything is concluded from it. Returns whether the state has to be read
    again.
    """
    alignment = state.get("alignment")
    if not (
        isinstance(alignment, Mapping) and alignment.get("report_disagrees_with_checkpoint")
        and alignment.get("workdir") and "alignment" in _stage_graph_of(state)
    ):
        return False
    if not run:
        say(
            "Note: the alignment's verification report says it is not finished and its checkpoint says it is. "
            "A run verifies the alignment again before it decides; this dry run goes by the report as it stands."
        )
        return False
    say(
        "Note: the alignment's verification report says it is not finished and its checkpoint says it is. "
        "Verifying the alignment again before deciding what has to run."
    )
    try:
        _finalize_stage("alignment", Path(str(alignment["workdir"])), threads)
    except (subprocess.CalledProcessError, OSError) as exc:
        say(f"The verification did not run to the end ({_first_line(str(exc), type(exc).__name__)}); the report stands.")
    return True


def preview_pipeline_run(
    *,
    stage_kwargs: Mapping[str, Any],
    project: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """What a run of this request would find stale, worked out and not acted on.

    For --dry-run. The dry run built its state without the stale
    list and without comparing requests, so on a finished project with a
    changed option it reported that there was nothing to do. This reads what
    the run reads and reaches the same verdicts through the same functions;
    it takes no lock, waits for no job, verifies nothing and writes nothing.

    Returns the state with the stale list applied (``state``), the list
    (``stale``), the next stage (``next_stage``) and, under ``refusal``, the
    PLM-0308 error the run would stop with, or None.
    """
    state_kwargs = _state_kwargs_from(stage_kwargs)
    checkpoint: Mapping[str, Any] = {}
    if project is not None:
        checkpoint = _load_supervisor_checkpoint(stage_checkpoint_path(Path(project), "pipeline_supervisor"))
    stale = _recorded_stale_stages(checkpoint)
    recorded_requests = checkpoint.get("stage_requests")
    requests: dict[str, Any] = dict(recorded_requests) if isinstance(recorded_requests, Mapping) else {}
    jobs = sorted(str(value) for value in (checkpoint.get("active_job_ids") or []) if str(value).strip())
    if jobs:
        print(
            "Note: the earlier run of this project recorded SLURM jobs " + ",".join(jobs)
            + "; a run waits for any of them still queued before it reads the project. This dry run does not.",
            flush=True,
        )
    for name, record in stale.items():
        if record["issued"]:
            # How its jobs ended is for the run to ask; here it is still stale.
            record.update(issued=False, job_ids=[])
    say = lambda line: print(line, flush=True)
    state = build_project_state(**state_kwargs)
    _verify_an_alignment_in_doubt(state, 1, say, run=False)
    _say_what_the_state_reader_set_aside(state, say)
    # A held request of a one-job stage that reads finished is
    # compared as the run would compare it once accounting confirms the job
    # (_settle_pending_requests); this dry run does not ask, and says so.
    recorded_pending = checkpoint.get("pending_stage_requests")
    for name, entry in (recorded_pending.items() if isinstance(recorded_pending, Mapping) else ()):
        record = entry.get("record") if isinstance(entry, Mapping) else None
        if not isinstance(record, Mapping):
            continue
        if str(_stage_entry(state, name).get("state") or "").lower() not in TERMINAL_STAGE_STATES:
            continue
        job_ids = sorted(str(job) for job in (entry.get("job_ids") or []) if str(job).strip())
        slurm_mode = bool(stage_kwargs.get("backend") == "slurm" and stage_kwargs.get("submit"))
        if job_ids and not slurm_mode and shutil.which("sacct") is None:
            say(f"Note: the {name} job an earlier run submitted is not known to have succeeded (this machine "
                "has no sacct to ask how its job ended); nothing is on record for it.")
            continue
        requests[name] = dict(record)
        say(f"Note: the {name} job an earlier run submitted ({', '.join(job_ids) or 'no id'}) reads finished; "
            "this dry run compares what it was asked to compute, which a run puts on record only once "
            "SLURM accounting says the job succeeded.")
    project = Path(project) if project is not None else None
    inputs = tuple(_run_inputs(stage_kwargs))
    _forget_what_is_gone(state, stale, requests, project, inputs)
    state, stale, changed, put_back = _work_out_what_is_stale(
        state=state, stale=stale, requests=requests, stage_kwargs=stage_kwargs,
        state_kwargs=state_kwargs, project=project, say=say,
    )
    refusal: Optional[ParalomeError] = None
    try:
        _refuse_stale_stages_that_would_keep_their_result(
            state, stale, changed, project=project, inputs=inputs, put_back=put_back,
        )
    except ParalomeError as exc:
        refusal = exc
    return {
        "state": state,
        "stale": stale,
        "next_stage": (state.get("pipeline") or {}).get("next_stage"),
        "refusal": refusal,
    }


def run_pipeline_to_completion(
    *,
    stage_kwargs: dict[str, Any],
    state_kwargs: dict[str, Any],
    project: Path,
    poll_seconds: int = 60,
    max_stages: int = 16,
) -> PipelineSupervisorResult:
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    # One supervisor per project, for the whole of the run.
    lock = None
    if not stage_kwargs.get("dry_run"):
        lock = _take_project_lock(stage_checkpoint_path(project, "pipeline_supervisor").parent)
    try:
        return _supervise_pipeline(
            stage_kwargs=stage_kwargs,
            state_kwargs=state_kwargs,
            project=project,
            poll_seconds=poll_seconds,
            max_stages=max_stages,
        )
    finally:
        _release_project_lock(lock)


def _supervise_pipeline(
    *,
    stage_kwargs: dict[str, Any],
    state_kwargs: dict[str, Any],
    project: Path,
    poll_seconds: int,
    max_stages: int,
) -> PipelineSupervisorResult:
    checkpoint_path = stage_checkpoint_path(project, "pipeline_supervisor")
    previous_checkpoint = _load_supervisor_checkpoint(checkpoint_path)
    run_id = str(uuid.uuid4())
    stages_run: list[str] = []
    last_state: dict[str, Any] = {}
    state_path = project / "project_state.json"
    active_stage: Optional[str] = None
    # The jobs submitted for the active stage and not yet seen
    # through. Written into every checkpoint while they are outstanding.
    active_job_ids: set[str] = set()
    # The stale list. `carried` is what an earlier run left and is
    # what a checkpoint holds until this run issues its first stage command;
    # from then on it is `stale`, with what this start added. Until a command
    # is issued nothing on disk has moved, so a run that is refused, or
    # stopped while its plan is on the screen, leaves the list as it found it
    # (an option mistyped, noticed and corrected must leave no stage marked).
    stale: dict[str, dict[str, Any]] = _recorded_stale_stages(previous_checkpoint)
    carried: Optional[dict[str, dict[str, Any]]] = stale
    recorded_requests = previous_checkpoint.get("stage_requests")
    requests: dict[str, Any] = dict(recorded_requests) if isinstance(recorded_requests, Mapping) else {}
    # Requests of one-job stages whose job has not been seen to succeed.
    recorded_pending = previous_checkpoint.get("pending_stage_requests")
    pending: dict[str, Any] = {
        name: dict(entry) for name, entry in recorded_pending.items() if isinstance(entry, Mapping)
    } if isinstance(recorded_pending, Mapping) else {}
    report_threads = int(stage_kwargs.get("report_threads") or 1)
    # The state reader checks the sample sheet as the stages do, so
    # it needs the --factor columns that stand in for `condition`.
    if stage_kwargs.get("rna_factors") and "factors" not in state_kwargs:
        state_kwargs = {**state_kwargs, "factors": tuple(stage_kwargs["rna_factors"])}

    def save(status: str, *, complete: bool = False, **fields: Any) -> None:
        write_json(checkpoint_path, _checkpoint_payload(
            run_id=run_id,
            status=status,
            complete=complete,
            stages_run=stages_run,
            state=last_state,
            state_path=state_path,
            recovered_from=previous_checkpoint or None,
            stale_stages=stale if carried is None else carried,
            stage_requests=requests,
            pending_stage_requests=pending,
            **fields,
        ), atomic=True)

    stop_signals = _StopSignals()
    stop_signals.__enter__()
    try:
        slurm_mode = bool(
            stage_kwargs.get("backend") == "slurm" and stage_kwargs.get("submit")
        )
        # Before anything is read or submitted. An earlier run's jobs
        # may still be writing the files the state below is built from.
        if slurm_mode:
            _wait_for_jobs_from_an_earlier_run(previous_checkpoint, poll_seconds)
        elif _earlier_job_ids(previous_checkpoint):
            # A local or scripts-only run waits for them too; a
            # machine that cannot ask SLURM refuses rather than run over them.
            earlier = sorted(_earlier_job_ids(previous_checkpoint))
            try:
                _squeue_job_records(set(earlier))
                cannot_ask = None
            except SlurmQueryUnavailable as exc:
                cannot_ask = "this machine has no squeue to ask" if shutil.which("squeue") is None else str(exc)
            if cannot_ask is not None:
                raise ParalomeError(
                    311,
                    "An earlier SLURM run of this project recorded jobs "
                    + ",".join(earlier)
                    + " that may still be writing its stage folders, and whether they are cannot be "
                    f"found out ({cannot_ask}). Nothing was started.",
                    fix=("run the command where squeue is available, or, if those jobs have ended, "
                         f"remove \"active_job_ids\" from {checkpoint_path} and run again"),
                    jobs=earlier,
                )
            _wait_for_jobs_from_an_earlier_run(previous_checkpoint, poll_seconds)

        # A STALE STAGE WHOSE RE-RUN THE EARLIER RUN STARTED.
        #
        # Its jobs have left the queue. If every one of them succeeded and the
        # stage now validates, the re-run is done and is not done twice. If
        # they failed, or how they ended cannot be found out, the stage is
        # still stale and its command is issued again -- which costs little:
        # only a stage that checks its own result is ever issued while stale,
        # and one that did recompute finds that its result answers.
        for name in list(stale):
            record = stale[name]
            if not record["issued"]:
                continue
            job_ids = set(record["job_ids"])
            settled, why = False, "its command did not finish"
            if job_ids and slurm_mode:
                settled, why = _recorded_jobs_succeeded(job_ids, poll_seconds)
                if settled:
                    try:
                        if record["workdir"]:
                            _finalize_stage(name, Path(record["workdir"]), report_threads)
                        observed = build_project_state(
                            **{**state_kwargs, "stale_stages": _stale_entries({other: record for other, record in stale.items() if other != name})}
                        )
                        settled = str(_stage_entry(observed, name).get("state") or "").lower() in TERMINAL_STAGE_STATES
                        why = "its outputs do not validate"
                    except Exception as exc:
                        settled, why = False, _first_line(str(exc), type(exc).__name__)
            elif job_ids:
                why = "this run does not ask SLURM how its jobs ended"
            if settled:
                del stale[name]
                print(
                    f"The re-run of {name} that the earlier run submitted (jobs "
                    + ",".join(sorted(job_ids)) + ") finished and validated; it is not run again.",
                    flush=True,
                )
            else:
                record.update(issued=False, job_ids=[])
                print(
                    f"The re-run of {name} that the earlier run started is not known to have "
                    f"succeeded ({why}); {name} will be issued again.",
                    flush=True,
                )

        say = lambda line: print(line, flush=True)
        last_state = build_project_state(**state_kwargs)
        # Before anything is concluded from the alignment's report.
        if _verify_an_alignment_in_doubt(last_state, report_threads, say):
            last_state = build_project_state(**state_kwargs)
        _say_what_the_state_reader_set_aside(last_state, say)
        # Before requests are compared -- see the function.
        _settle_pending_requests(last_state, pending, requests, slurm_mode=slurm_mode,
                                 poll_seconds=poll_seconds, say=say)
        # Before requests are compared -- see the function.
        inputs = tuple(_run_inputs(stage_kwargs))
        _forget_what_is_gone(last_state, stale, requests, project, inputs)
        carried = {name: dict(record) for name, record in stale.items()}
        # Finished stages behind one that has to run again no longer
        # answer it. Nor does a stage whose request has changed.
        # Copies of the two argument sets carry the list: to the state
        # builder, so those stages read as not done, and to the stage runner,
        # so it selects them in their turn.
        last_state, stale, changed, put_back = _work_out_what_is_stale(
            state=last_state, stale=stale, requests=requests, stage_kwargs=stage_kwargs,
            state_kwargs=state_kwargs, project=project, say=say,
        )
        if stale:
            state_kwargs = {**state_kwargs, "stale_stages": _stale_entries(stale)}
            stage_kwargs = {**stage_kwargs, "stale_stages": _stale_entries(stale)}
        state_path = write_project_state(project, last_state)
        validation = validate_pipeline_state(last_state)
        if not validation.valid:
            raise SupervisorValidationError("Pipeline state is invalid; execution cannot start.", validation.diagnostics)
        # Before anything is submitted.
        _refuse_stale_stages_that_would_keep_their_result(
            last_state, stale, changed, project=project, inputs=inputs, put_back=put_back,
        )
        _validate_restart(previous_checkpoint, last_state, validation)
        _print_restart_summary(last_state, validation, stage_kwargs)

        for _ in range(max_stages):
            if validation.complete:
                save("complete", complete=True)
                return PipelineSupervisorResult(
                    project=project,
                    complete=True,
                    stages_run=stages_run,
                    report_workdir=_terminal_workdir(last_state, state_kwargs, project),
                    state_path=state_path,
                )

            active_stage = validation.next_stage
            if not active_stage:
                raise SupervisorValidationError("Pipeline is incomplete but no legal next stage was resolved.")

            # From here a stage command is issued, so the stale list
            # is no longer one that can be worked out afresh; and a stale stage
            # is on record as issued before its command can replace anything.
            # Whether the command only writes scripts is known once it is
            # built, and this is then taken back (see below).
            carried_until_now, carried = carried, None
            if active_stage in stale:
                stale[active_stage].update(issued=True, job_ids=[])
            save("running", active_stage=active_stage)
            print(f"\nPipeline supervisor: starting {active_stage}", flush=True)

            def request_of(command: list[str], stage: Optional[str] = None) -> Optional[dict[str, Any]]:
                try:
                    return {
                        **stage_request_record(
                            stage or active_stage, list(command), project=project,
                            samples=stage_kwargs.get("samples"),
                            remove_samples=tuple(stage_kwargs.get("remove_samples") or ()),
                        ),
                        "paralome_version": _current_version(),
                    }
                except Exception:
                    return None

            def record_request(command: list[str], stage: Optional[str] = None) -> None:
                # What the stage was asked for, as issued. Not for a
                # run that only writes scripts: nothing was computed from it.
                stage = stage or active_stage
                if not command or only_writes_scripts(command):
                    return
                record = request_of(command, stage)
                if record is None:
                    requests.pop(stage, None)
                else:
                    requests[stage] = record

            def hold_request(command: list[str], job_ids: set[str], stage: Optional[str] = None) -> None:
                # A stage submitted as ONE job computes all of its result or none of it. Its request goes on
                # record when the job is known to have succeeded, as it does for a stage run here once its
                # command has returned; until then it is held, so a failed job leaves no request that no result
                # was made from.
                stage = stage or active_stage
                record = request_of(command, stage) if command else None
                if record is None:
                    pending.pop(stage, None)
                    return
                pending[stage] = {"record": record, "job_ids": sorted(job_ids)}

            try:
                if slurm_mode:
                    with stop_signals.hold():
                        result: PipelineRunResult = run_pipeline_next_stage(**stage_kwargs)
                else:
                    result = run_pipeline_next_stage(**stage_kwargs)
            except StageCommandError as exc:
                # The command failed after submitting some of its jobs. They are
                # real and still queued; record them so the next run waits on
                # them instead of submitting the stage over the top of them.
                active_job_ids = set(exc.submitted_job_ids)
                # A stale stage whose command failed is on record as not issued, so the success of the part of
                # an array that got in is not taken for the stage's.
                if active_stage in stale:
                    stale[active_stage].update(issued=False, job_ids=[], workdir=str(exc.workdir))
                if only_writes_scripts(exc.command):
                    carried = carried_until_now
                # And a command that failed without submitting
                # anything asked for nothing that is being computed -- an
                # option its own parser refused must not go on record as the
                # request the stage's results were made with.
                if exc.submitted_job_ids:
                    record_request(exc.command)
                raise
            # A COMMAND THAT ONLY WRITES SLURM SCRIPTS RUNS NOTHING,
            # and that is read from the command, stage by stage. It was read
            # from the run's options (SLURM backend, no --submit), but
            # segregation, filtering and reporting have no backend and run
            # where they are started: such a run did run them, recorded no
            # request for them, and a later change to the sheet was not seen.
            scripts_only = bool(result.executed and only_writes_scripts(list(result.command)))
            if scripts_only:
                # Nothing was issued to run: the list is as it was found.
                carried = carried_until_now
                if active_stage in stale:
                    stale[active_stage].update(issued=False, job_ids=[])
            # On record before anything below can raise.
            submitted_job_ids = set(result.submitted_job_ids)
            if slurm_mode:
                active_job_ids = set(submitted_job_ids)
            if active_stage in stale and not scripts_only:
                stale[active_stage].update(
                    job_ids=sorted(submitted_job_ids) if slurm_mode else [],
                    workdir=str(result.workdir),
                )
            # Every submitted SLURM stage, not only one-job stages,
            # holds its request until its jobs are seen to succeed; a failed
            # array left its request on record and a corrected option was
            # then refused as a changed request (PLM-0308).
            one_job = bool(slurm_mode and submitted_job_ids)
            if result.executed and result.next_stage == active_stage:
                if one_job:
                    hold_request(list(result.command), submitted_job_ids)
                else:
                    record_request(result.command)
            # A stale stage is judged without its mark only once its re-run has ended, and comes off the list
            # only if it validates. NEVER after a run that only wrote scripts: the stage's old outputs would
            # then read complete, and the next stage would be computed from them.
            remaining = {name: record for name, record in stale.items() if name != active_stage}
            settled_kwargs = (
                {**state_kwargs, "stale_stages": _stale_entries(remaining)}
                if active_stage in stale and not scripts_only else state_kwargs
            )
            stop_signals.release()
            if result.next_stage and result.next_stage != active_stage:
                raise SupervisorValidationError(
                    f"Executor selected {result.next_stage}, but supervisor authorized {active_stage}.",
                    [_diagnostic("executor_stage_mismatch", "Executor did not run the authorized stage.", authorized=active_stage, selected=result.next_stage)],
                )
            stages_run.append(active_stage)
            if result.returncode not in (None, 0):
                raise ParalomeError(                       # PLM-0304
                    304,
                    f"Stage {active_stage} returned {result.returncode}.",
                    stage=active_stage,
                    returncode=result.returncode,
                )

            # Trajectory statistics run in tandem with the differential statistics; they need only the
            # trajectories. Every other stage whose reads are finished (profiles.STAGE_READS; none for a
            # profile read as a line) is submitted now, alongside this one, and seen through after it.
            tandem: list[tuple[str, PipelineRunResult, set[str]]] = []
            if slurm_mode and submitted_job_ids and not scripts_only:
                for companion in _tandem_stages(last_state, active_stage):
                    if companion in stale:
                        stale[companion].update(issued=True, job_ids=[])
                    save("running", active_stage=active_stage, active_job_ids=active_job_ids,
                         tandem_stages=[name for name, _r, _j in tandem] + [companion])
                    print(f"\nPipeline supervisor: starting {companion}, alongside {active_stage} "
                          f"(it reads {', '.join(_reads_of(last_state, _stage_graph_of(last_state)).get(companion, ())) or 'nothing'}"
                          ", which is finished)", flush=True)
                    try:
                        with stop_signals.hold():
                            companion_result: PipelineRunResult = run_pipeline_next_stage(
                                **stage_kwargs, _run_stage=companion)
                    except StageCommandError as exc:
                        active_job_ids |= set(exc.submitted_job_ids)
                        if companion in stale:
                            stale[companion].update(issued=False, job_ids=[], workdir=str(exc.workdir))
                        if exc.submitted_job_ids:
                            record_request(exc.command, companion)
                        raise
                    stop_signals.release()
                    if companion_result.next_stage != companion:
                        raise SupervisorValidationError(
                            f"Executor selected {companion_result.next_stage}, but supervisor authorized {companion}.",
                            [_diagnostic("executor_stage_mismatch", "Executor did not run the authorized stage.",
                                         authorized=companion, selected=companion_result.next_stage)],
                        )
                    if companion_result.returncode not in (None, 0):
                        raise ParalomeError(304, f"Stage {companion} returned {companion_result.returncode}.",
                                            stage=companion, returncode=companion_result.returncode)
                    companion_jobs = set(companion_result.submitted_job_ids)
                    active_job_ids |= companion_jobs
                    if companion in stale:
                        stale[companion].update(job_ids=sorted(companion_jobs), workdir=str(companion_result.workdir))
                    if companion_result.executed:
                        if companion_jobs:
                            hold_request(list(companion_result.command), companion_jobs, companion)
                        else:
                            record_request(companion_result.command, companion)
                    stages_run.append(companion)
                    tandem.append((companion, companion_result, companion_jobs))
                if tandem:
                    remaining = {name: record for name, record in remaining.items()
                                 if name not in {companion for companion, _r, _j in tandem}}
                    if active_stage in stale or any(companion in stale for companion, _r, _j in tandem):
                        settled_kwargs = {**state_kwargs, "stale_stages": _stale_entries(remaining)}

            if slurm_mode and submitted_job_ids:
                save("running", active_stage=active_stage, active_job_ids=active_job_ids,
                     tandem_stages=[name for name, _r, _j in tandem])

                def _stage_checkpoint_complete() -> bool:
                    # For a stale stage this is built with the stage still on the stale list, so it never
                    # passes: old outputs are not this run's, however complete they look. The one exception
                    # sets its earlier result aside before it submits, so what is there now is this run's, and
                    # it can be confirmed where SLURM accounting does not answer.
                    observed = build_project_state(**(
                        settled_kwargs if active_stage in STAGES_THAT_ALWAYS_RECOMPUTE else state_kwargs
                    ))
                    observed_validation = validate_pipeline_state(
                        observed,
                        previous_state=last_state,
                        expected_stage=active_stage,
                    )
                    return observed_validation.valid

                try:
                    _wait_for_jobs(
                        submitted_job_ids,
                        poll_seconds,
                        completion_probe=_stage_checkpoint_complete,
                        workdir=result.workdir,
                    )
                except ParalomeError:
                    # The job failed, or was cancelled for want of
                    # an answer: no result was made from its request.
                    pending.pop(active_stage, None)
                    raise
                # And a one-job stage's request goes on record now.
                if active_stage in pending:
                    requests[active_stage] = dict(pending.pop(active_stage)["record"])
                active_job_ids -= submitted_job_ids
                _finalize_stage(active_stage, result.workdir, report_threads)
                # Then each stage submitted alongside it.
                for companion, companion_result, companion_jobs in tandem:
                    if companion_jobs:
                        save("running", active_stage=companion, active_job_ids=active_job_ids,
                             tandem_stages=[name for name, _r, _j in tandem])

                        def _companion_complete(name: str = companion) -> bool:
                            observed = build_project_state(**(
                                settled_kwargs if name in STAGES_THAT_ALWAYS_RECOMPUTE else state_kwargs))
                            entry = _stage_entry(observed, name)
                            return str(entry.get("state") or "").lower() in TERMINAL_STAGE_STATES

                        try:
                            _wait_for_jobs(companion_jobs, poll_seconds, completion_probe=_companion_complete,
                                           workdir=companion_result.workdir)
                        except ParalomeError:
                            pending.pop(companion, None)
                            raise
                        if companion in pending:
                            requests[companion] = dict(pending.pop(companion)["record"])
                        active_job_ids -= companion_jobs
                    _finalize_stage(companion, companion_result.workdir, report_threads)
            elif result.executed and not scripts_only:
                # A stage that ran here is finalized like one that ran
                # as jobs. Without this a local alignment was never verified,
                # so it never read complete and the run stopped after it. A
                # stage that wrote nothing cannot be finalized; that is for
                # the validation below to say, in its own words.
                try:
                    _finalize_stage(active_stage, result.workdir, report_threads, quiet=True)
                except subprocess.CalledProcessError as exc:
                    print(
                        f"The closing check of {active_stage} did not run to the end "
                        f"(exit {exc.returncode}); the stage is judged on what it wrote.",
                        flush=True,
                    )

            slurm_submitted = bool(slurm_mode and submitted_job_ids)
            # A scripts-only run executed nothing to wait for.
            synchronous_execution = bool(result.executed and not submitted_job_ids and not scripts_only)
            refreshed, refreshed_path, refreshed_validation = _refresh_stage_state(
                state_kwargs=settled_kwargs,
                project=project,
                previous_state=last_state,
                expected_stage=active_stage,
                attempts=12 if slurm_submitted else (3 if synchronous_execution else 1),
                delay_seconds=5,
            )
            if not refreshed_validation.valid:
                if scripts_only:
                    # Not a failure of the run: the scripts are what a SLURM run without --submit produces,
                    # and that is said.
                    print(
                        f"\nStage {active_stage}: the SLURM scripts were written under {result.workdir} "
                        "and NOTHING WAS SUBMITTED, because --submit was not given.\n"
                        "Run the same command again with --submit to submit the jobs "
                        "(or with --backend local to run the stage on this machine).",
                        flush=True,
                    )
                    save("not_submitted", active_stage=active_stage)
                    return PipelineSupervisorResult(
                        project=project,
                        complete=False,
                        stages_run=stages_run,
                        report_workdir=Path(result.workdir),
                        state_path=state_path,
                    )
                if slurm_mode and synchronous_execution:
                    message = (
                        f"Stage {active_stage} returned no SLURM job ID and did not "
                        "complete synchronously."
                    )
                else:
                    message = f"Stage {active_stage} finished but supervisor validation failed."
                raise SupervisorValidationError(
                    message,
                    refreshed_validation.diagnostics,
                )
            for companion, _r, _j in tandem:
                if str(_stage_entry(refreshed, companion).get("state") or "").lower() not in TERMINAL_STAGE_STATES:
                    raise SupervisorValidationError(
                        f"Stage {companion} finished but supervisor validation failed.",
                        [_diagnostic("stage_not_advanced", f"Stage {companion} returned but did not validate as "
                                     "complete.", stage=companion)],
                    )
            if (active_stage in stale and not scripts_only) or tandem:
                stale = remaining
                state_kwargs = settled_kwargs
                stage_kwargs = {**stage_kwargs, "stale_stages": _stale_entries(stale)}
            last_state = refreshed
            state_path = refreshed_path
            validation = refreshed_validation
            active_stage = None

        raise SupervisorValidationError(
            f"Pipeline exceeded the safety limit of {max_stages} stage transitions.",
            [_diagnostic("transition_limit_exceeded", "Safety transition limit reached.", max_stages=max_stages)],
        )
    except Exception as exc:
        diagnostics = exc.diagnostics if isinstance(exc, SupervisorValidationError) else [
            _diagnostic("execution_failure", str(exc), stage=active_stage)
        ]
        if last_state:
            save(
                "failed",
                active_stage=active_stage,
                diagnostics=diagnostics,
                failure=str(exc),
                active_job_ids=active_job_ids,
            )
        raise
    except KeyboardInterrupt as exc:
        # STOPPING THE SUPERVISOR DOES NOT STOP THE WORK.
        #
        # Ctrl-C, scancel or kill of the supervisor, a closed terminal. The submitted jobs belong to the
        # scheduler and keep running; cancelling hours of healthy work because the process watching it was
        # stopped would be the wrong default. The checkpoint names the jobs, the console says what to do,
        # and the same command run again waits for them (see _wait_for_jobs_from_an_earlier_run).
        how = exc.signal_name if isinstance(exc, SupervisorInterrupted) else "Ctrl-C"
        if last_state:
            save(
                "interrupted",
                active_stage=active_stage,
                failure=f"interrupted by {how}",
                active_job_ids=active_job_ids,
            )
        if active_job_ids:
            jobs = ",".join(sorted(active_job_ids))
            print(
                f"\nPipeline supervisor stopped ({how}) during {active_stage or 'the run'}. "
                f"SLURM jobs {jobs} were left running.\n"
                "Run the same command again to pick them up and continue; "
                f"`scancel {jobs.replace(',', ' ')}` stops the work.",
                flush=True,
            )
        else:
            print(f"\nPipeline supervisor stopped ({how}). No SLURM jobs were outstanding.", flush=True)
        raise
    finally:
        stop_signals.__exit__(None, None, None)
