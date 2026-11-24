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

"""Shared ``--stratify-by`` fan-out for the single-cell differential drivers.

``--stratify-by`` always fans out, on all pipelines, and every option is
wired through to the CLI because ``pipeline run`` is where all pipelines go
through.

Before this, ``rna singlecell-rna-differential`` and ``atac
singlecell-differential`` REQUIRED ``--stratify-value`` -- one invocation per
cell type, so three contrasts across fifteen cell types was forty-five commands
driven by an external loop. The global PCA already fanned out internally (for
the same reason); these two were the holdouts.

The pattern: build a dict of slices keyed by the stratify column, and narrow
it to one entry only when a value was named.

Layout. Each stratum runs into ``<out>/<slug>/`` and writes its own checkpoint
there in the UNCHANGED single-stratum schema. The fan-out then writes an index
checkpoint at the top level carrying ``schema_version: 2`` and a ``strata``
map. That split is what makes the change safe: anything pointed at a stratum
directory -- including a future stage that wants one cell type's statistics --
sees exactly the file layout it saw before, and only readers of the TOP-level
checkpoint need to understand v2. See ``summarize_*`` in ``pipeline.state`` for
the v1/v2 branch.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from paralome.atac.singlecell import sanitize_sample_id
from paralome.checkpoints import stage_checkpoint_path
from paralome.errors import ParalomeError
# The same comma/whitespace splitter --remove uses, so
# "A,B" means the same thing on every option that takes a list.
from paralome.singlecell.sample_removal import split_remove_values

#: What a stratum needs is decided by the plan. Each requested statistic is planned on the stratum's own
#: samples by ``comparisons.plan_differential`` (``stats_core.plan_each_statistic``) -- the same call the
#: stratum's run makes -- so the planner and the run cannot disagree about which statistics a stratum can
#: support.


def resolve_strata(
    rows: Sequence[Mapping[str, object]],
    stratify_by: str,
    stratify_value: Optional[str] = None,
) -> "Dict[str, List[Mapping[str, object]]]":
    """Ordered ``{value: rows}``, narrowed to one entry if a value was named.

    Raises if ``stratify_by`` is not a column, or if a named value is not
    present -- both are user errors worth failing on immediately, and the error
    lists what IS available, since the usual cause is a cell-type label that
    does not match the metadata exactly.
    """
    if not rows:
        raise ValueError("Sample manifest is empty; nothing to stratify")
    from paralome.stratify import missing_columns, row_strata

    absent = missing_columns(stratify_by, list(rows[0]))
    if absent:
        raise ValueError(
            f"Sample manifest is missing --stratify-by column(s) {', '.join(map(repr, absent))} "
            f"(available: {', '.join(sorted(str(key) for key in rows[0]))})"
        )

    # Several columns (every combination present), a comma list in a cell puts the sample
    # in each of its values, and a sample with no value is in no stratum (paralome.stratify).
    slices: "Dict[str, List[Mapping[str, object]]]" = {}
    for row in rows:
        for key in row_strata(row, stratify_by):
            slices.setdefault(key, []).append(row)
    slices = {value: slices[value] for value in sorted(slices)}

    if stratify_value is not None:
        if str(stratify_value) not in slices:
            raise ValueError(
                f"--stratify-value {stratify_value!r} not found in --stratify-by "
                f"{stratify_by!r} column (available: {', '.join(sorted(slices))})"
            )
        return {str(stratify_value): slices[str(stratify_value)]}
    return slices


def stratum_slugs(values: Sequence[str]) -> Dict[str, str]:
    """``{value: directory-safe slug}``, refusing collisions rather than merging.

    Same sanitizer the pseudobulk sample_ids use, so a stratum directory name
    matches the sample_id tokens in the tables inside it. Two values that
    sanitize alike would silently overwrite each other's results, so that is an
    error naming both -- copied from the global PCA's fan-out, which hit the
    same hazard.
    """
    slugs: Dict[str, str] = {}
    seen: Dict[str, str] = {}
    from paralome.stratify import JOIN

    for value in values:
        # A stratum of several columns is named by each value's slug, joined by __.
        slug = "__".join(sanitize_sample_id(part) for part in str(value).split(JOIN))
        if slug in seen:
            raise ValueError(
                f"stratify values {seen[slug]!r} and {value!r} both sanitize to the "
                f"directory name {slug!r} -- rename one upstream in the metadata"
            )
        seen[slug] = str(value)
        slugs[str(value)] = slug
    return slugs


def resolve_stratum_names(
    requested: Sequence[str], available: Sequence[str], *, flag: str
) -> List[str]:
    """Map user-typed stratum names onto the manifest's own spellings.

    Accepts EITHER the raw metadata value or the directory-safe slug, because
    those differ for most cell types and the slug is the one the user sees
    everywhere:

        --stratify-value 'CD8 TEM'   ->  files named CD8_TEM_*
        --stratify-value 'B naive'   ->  files named B_naive_*

    and it is not even consistent -- ``NK_CD56bright`` genuinely contains an
    underscore in the metadata while ``NK Proliferating`` contains a space, so
    no rule applied by eye gets both right. Requiring the raw spelling would
    mean the names in every filename, folder and plot title are the ones that
    do not work.

    Order follows ``available`` (the manifest's), not the order typed, so two
    invocations naming the same strata differently produce identical output.

    Raises naming both spellings of everything available: an unmatched cell
    type is the single most likely mistake here, and a list is what fixes it.
    """
    slugs = stratum_slugs(list(available))
    by_slug = {slug: value for value, slug in slugs.items()}
    chosen, unknown = set(), []
    for token in split_remove_values(requested):
        if token in slugs:
            chosen.add(token)
        elif token in by_slug:
            chosen.add(by_slug[token])
        else:
            unknown.append(token)
    if unknown:
        catalogue = ", ".join(
            value if value == slugs[value] else f"{value!r} (as {slugs[value]})"
            for value in available
        )
        raise ValueError(
            f"{flag} value(s) not found: {', '.join(unknown)}. "
            f"Available strata: {catalogue}"
        )
    return [value for value in available if value in chosen]


#: Column carrying each pseudobulk sample's cell count, written by the
#: aggregation into ``sample_manifest.csv``.
CELL_COUNT_COLUMN = "n_cells"


def stratum_cell_counts(
    slices: "Dict[str, List[Mapping[str, object]]]",
    *,
    cell_count_key: str = CELL_COUNT_COLUMN,
) -> Dict[str, int]:
    """Total cells per stratum, summed across its pseudobulk samples.

    The filter measures cell counts per group (here, per cell type). Total rather than per-sample, because ``--min-cells-per-group``
    already governs the per-sample floor and applies during AGGREGATION -- this
    is the other question, asked of the finished matrix.

    A row with a missing or unparseable count contributes zero rather than
    raising. The count is provenance the aggregation wrote, not user input, and
    a stratum whose counts cannot be read should fall below any threshold and
    be reported as skipped -- not abort a fourteen-stratum run.
    """
    totals: Dict[str, int] = {}
    for value, rows in slices.items():
        total = 0
        for row in rows:
            try:
                total += int(float(str(row.get(cell_count_key, 0) or 0)))
            except (TypeError, ValueError):
                continue
        totals[value] = total
    return totals


def strata_to_run(
    rows: Sequence[Mapping[str, object]],
    stratify_by: str,
    *,
    only: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
    min_cells_per_stratum: int = 0,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """``(values to run, [{value, reason, ...} skipped])`` for ANY fan-out.

    A cell-count filter applies to any option that fans out that grouping
    (the scRNA PCA among them).

    It was already two places with different behaviour: the differential's
    ``run_stratified_fan_out`` and the global PCA's own loop, which sorted its
    own values, ran its own slug-collision check and built its own skip list.
    Adding the filter to one of them would have made the two disagree about
    which cell types exist -- the failure this codebase produces every time one
    concept has two implementations. So the selection lives here and both call
    it.

    Explicit selection is applied BEFORE the cell-count floor, and a stratum
    named outright still gets floored: asking for a cell type does not make it
    testable, and silently honouring the request would defeat the point of the
    threshold. It IS reported as skipped, so a named-but-dropped cell type is
    visible rather than absent.
    """
    slices = select_strata(resolve_strata(rows, stratify_by), only=only, skip=skip)
    if min_cells_per_stratum <= 0:
        return list(slices), []

    totals = stratum_cell_counts(slices)
    keep, skipped = [], []
    for value in slices:
        if totals[value] < min_cells_per_stratum:
            skipped.append({
                "value": value,
                "samples": len(slices[value]),
                "cells": totals[value],
                "reason": (
                    f"{totals[value]:,} cells, below --min-cells-per-stratum "
                    f"({min_cells_per_stratum:,})"
                ),
            })
        else:
            keep.append(value)
    return keep, skipped


def require_nonempty_strata(
    keep: Sequence[str],
    floored: Sequence[Mapping[str, Any]],
    *,
    stratify_by: str,
    min_cells_per_stratum: int,
    available: Iterable[str],
) -> None:
    """Raise if a fan-out has no strata left, naming the reason it can evidence.

    Nothing checked this, so a selection or a threshold that removed
    every stratum produced a SUCCESSFUL run with an empty output tree and exit
    code 0 -- indistinguishable from a run whose ``--stratify-by`` named the
    wrong column, and from one that genuinely had nothing to do.

    A user may set ``--min-cells-per-stratum 500`` deliberately, as a high bar
    for cell type analyses that are not reliable. That is exactly the usage that can empty the list. A high bar is a legitimate
    choice; producing nothing without saying so is not a legitimate response to
    it.

    Lives here, called by both ``run_stratified_fan_out`` and the global PCA's
    loop, because those are the two places that consume ``strata_to_run`` and
    writing the check twice is the precise mistake this patch exists to fix.

    ``strata_to_run`` deliberately does NOT raise this itself -- a caller that
    wants to report an empty selection rather than fail on it stays possible.

    The totals are printed because the next question after this fires is always
    "then what WOULD have passed", and re-running at a lower threshold to find
    out is an expensive way to ask.
    """
    if keep:
        return
    # ``floored`` carries ONLY the cell-count removals; values dropped by
    # --stratify-value/--stratify-skip are not in it. So the threshold can only
    # be blamed when it actually removed something, or a selection that emptied
    # the list would be reported as a cell-count failure.
    if floored:
        detail = ", ".join(
            f"{item['value']} ({int(item['cells']):,} cells)"
            for item in sorted(floored, key=lambda i: -int(i["cells"]))
        )
        raise ValueError(
            f"--min-cells-per-stratum {min_cells_per_stratum:,} removed every "
            f"{stratify_by!r} group, so there is nothing to run. Groups and "
            f"their totals, largest first: {detail}. Lower the threshold, or "
            f"check that --stratify-by names the column you meant."
        )
    listed = ", ".join(sorted(available)) or "(none)"
    raise ValueError(
        f"No {stratify_by!r} group is left to run after --stratify-value/"
        f"--stratify-skip. Groups present in the manifest: {listed}."
    )


def resolve_one_stratum(sample_manifest: Any, stratify_by: str, value: str) -> str:
    """The raw metadata spelling of one stratum, given raw OR slug.

    The single-stratum path filters the manifest by string equality deep in
    ``stats_core``, so the name has to be canonical before it gets there. This
    is the one place that translation happens for that path; the fan-out gets
    it from ``select_strata``.
    """
    rows = read_manifest_rows(sample_manifest)
    available = list(resolve_strata(rows, stratify_by))
    return resolve_stratum_names([value], available, flag="--stratify-value")[0]


def read_manifest_rows(sample_manifest: Any) -> List[Dict[str, Any]]:
    """The sample manifest's rows, as the planners read them.

    Opened as ``utf-8-sig``, as stats_core._read_pseudobulk_manifest is, so a
    manifest saved from Excel with a byte-order mark reads its first header as
    ``sample_id``. One reader for both planners, so they cannot drift.
    """
    import csv

    path = Path(sample_manifest).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def manifest_digest(sample_manifest: Any) -> str:
    """sha256 of the manifest's bytes (see write_gather_plan)."""
    import hashlib

    return hashlib.sha256(Path(sample_manifest).expanduser().resolve().read_bytes()).hexdigest()


def split_stratify_values(stratify_value: Optional[str]) -> List[str]:
    """``"CD8 TEM,CD4 Naive"`` -> two entries; None/"" -> none.

    One entry keeps the single-stratum behaviour exactly, which matters beyond
    taste: the fan-out re-invokes itself once per stratum with a single
    ``--stratify-value``, and that invocation must take the single-stratum
    path or the fan-out would recurse.
    """
    return split_remove_values([stratify_value]) if stratify_value else []


def select_strata(
    slices: "Dict[str, List[Mapping[str, object]]]",
    *,
    only: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
) -> "Dict[str, List[Mapping[str, object]]]":
    """Narrow the fan-out set by inclusion or exclusion.

    ``--remove`` resolves against the REPLICATE column (it drops a donor
    across every cell type); this narrows the cell types.

    ``only`` and ``skip`` are mutually exclusive at the CLI. Both are resolved
    through ``resolve_stratum_names``, so either spelling works and an unknown
    name fails loudly rather than silently narrowing to nothing -- which is the
    failure mode that matters, since an empty fan-out and a fan-out the user
    did not want look identical in the output.
    """
    available = list(slices)
    if only:
        keep = set(resolve_stratum_names(only, available, flag="--stratify-value"))
    elif skip:
        dropped = set(resolve_stratum_names(skip, available, flag="--stratify-skip"))
        keep = {value for value in available if value not in dropped}
        if not keep:
            raise ValueError(
                "--stratify-skip excluded every stratum; nothing would be tested"
            )
    else:
        return slices
    return {value: slices[value] for value in available if value in keep}


def plan_stratum_concurrency(n_strata: int, jobs: int) -> Tuple[int, int]:
    """``(strata to run at once, model workers each)`` for a job with ``jobs`` CPUs.

    ``--threads`` is split evenly by the number of concurrent tests: each test
    is single threaded, but all the analyses run in parallel -- 20 strata and
    20 CPUs gives 20 concurrent at 1 worker each. One worker takes the
    in-process branch of ``_run_peak_chunks`` only on the main thread, so one
    worker means one PROCESS, not a thread sharing a GIL with the others.

    The product stays bounded by ``jobs``, so nothing runs that SLURM does not
    know about and a node is not overloaded.

    Returns (1, jobs) when there is nothing to gain: the strata run serially.
    """
    jobs = max(1, int(jobs or 1))
    if n_strata <= 1 or jobs <= 1:
        return 1, jobs
    concurrent = min(n_strata, jobs)
    return concurrent, max(1, jobs // concurrent)


def fan_out(
    *,
    values: Sequence[str],
    run_one: Callable[[str, str, Path, int], Dict[str, Any]],
    out_dir: Path,
    skip_check: Optional[Callable[[str], Optional[str]]] = None,
    max_workers: int = 1,
    stratum_jobs: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run ``run_one`` per stratum, collecting results and skips.

    ``run_one(value, slug, results_dir)`` does one stratum's work and returns
    its result dict. ``skip_check(value)`` returns a reason string to skip, or
    None to proceed.

    Every stratum receives the SAME ``results_dir``, so results are not nested
    too deeply: the file prefix is what distinguishes the strata, and it
    survives a file being copied out of the tree, which a directory does not.
    ``slug`` is passed through for the callee to use as that prefix and as its
    checkpoint variant.

    One thin stratum does not cost the others -- the same policy the global
    PCA's fan-out uses ("skip and report"): a pDC or dnT
    population too small to test should not block the other cell types. But
    if EVERY stratum is skipped the caller raises, so a silently empty run
    still trips ``set -Eeuo pipefail`` rather than reporting success with no
    output.
    """
    slugs = stratum_slugs(values)
    completed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Skips are decided first and serially: the check is cheap, and deciding it
    # up front means the pool below only ever holds real work.
    to_run: List[str] = []
    for value in values:
        reason = skip_check(value) if skip_check else None
        if reason:
            skipped.append({"value": value, "slug": slugs[value], "reason": reason})
            continue
        to_run.append(value)

    if max_workers <= 1 or len(to_run) <= 1:
        for value in to_run:
            completed.append({
                "value": value, "slug": slugs[value],
                **run_one(value, slugs[value], Path(out_dir), stratum_jobs),
            })
        return completed, skipped

    # Strata run concurrently in PROCESSES, forked from this one, so everything a stratum does around its
    # numpy fit -- planning, matching cells, writing its tables -- runs in parallel rather than taking turns
    # on one interpreter. fork keeps what threads gave: run_one is a closure and is never pickled (the child
    # inherits it, see _fan_out_call), and the counts matrix loaded before the fork is shared copy-on-write,
    # not copied per stratum.
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    print(
        f"[paralome] fanning out {len(to_run)} strata, {max_workers} at a time",
        file=sys.stderr, flush=True,
    )
    sys.stdout.flush()
    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, BaseException] = {}
    global _FAN_OUT_RUN
    _FAN_OUT_RUN = run_one
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = {
                pool.submit(_fan_out_call, value, slugs[value], str(out_dir), stratum_jobs): value
                for value in to_run
            }
            for future in futures:
                value = futures[future]
                try:
                    outcome = future.result()
                except BaseException as exc:  # noqa: BLE001  (a worker that died: killed, out of memory)
                    errors[value] = exc
                    continue
                if outcome[0] == "ok":
                    results[value] = outcome[1]
                else:
                    errors[value] = _rebuild_error(outcome[1])
    finally:
        _FAN_OUT_RUN = None

    if errors:
        first = sorted(errors)[0]
        raise RuntimeError(
            f"{len(errors)} of {len(to_run)} strata failed: "
            + "; ".join(f"{v}: {type(errors[v]).__name__}: {errors[v]}" for v in sorted(errors))
        ) from errors[first]

    # Rebuilt in the ORIGINAL stratum order, not completion order, so the
    # checkpoint, the printed summary and the file listing are identical
    # whether this ran serially or nine-wide.
    for value in to_run:
        completed.append({"value": value, "slug": slugs[value], **results[value]})
    return completed, skipped


#: The stratum function of the fan-out under way; a forked worker inherits it.
_FAN_OUT_RUN: Optional[Callable[[str, str, Path, int], Dict[str, Any]]] = None


def _fan_out_call(value: str, slug: str, out_dir: str, stratum_jobs: int):
    """One stratum in a forked worker: ``("ok", result)`` or ``("error", what to rebuild it from)``.

    The exception is sent back as its parts, not pickled: Paralome's errors take keyword context their
    constructors do not accept back from a pickle."""
    try:
        return ("ok", _FAN_OUT_RUN(value, slug, Path(out_dir), stratum_jobs))
    except BaseException as exc:  # noqa: BLE001
        import traceback

        return ("error", {
            "type": type(exc).__name__, "code": getattr(exc, "code", None),
            "detail": getattr(exc, "detail", None) or str(exc), "fix": getattr(exc, "fix", None),
            "traceback": "".join(traceback.format_exception(exc)),
        })
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def _rebuild_error(parts: Mapping[str, Any]) -> BaseException:
    """The worker's exception, as close to itself as its parts allow (the traceback goes to the log)."""
    from paralome.errors import ParalomeError

    print(parts.get("traceback", ""), file=sys.stderr, flush=True)
    if parts.get("code") is not None:
        try:
            error = ParalomeError(int(parts["code"]), str(parts.get("detail") or ""),
                                  **({"fix": parts["fix"]} if parts.get("fix") else {}))
            return error
        except Exception:  # noqa: BLE001
            pass
    error_type = {"ValueError": ValueError, "KeyError": KeyError, "RuntimeError": RuntimeError}.get(
        str(parts.get("type")), RuntimeError)
    return error_type(str(parts.get("detail") or parts.get("type")))


def format_differential_summary(result: Mapping[str, Any]) -> List[str]:
    """Human-readable summary of a differential run, fanned out or not.

    Shared by both modalities so the two cannot diverge in what they report.
    One line per statistic folder (where it is, what it tested and
    found), and the statistics a stratum could not run.
    """
    def folders(item: Mapping[str, Any]) -> List[str]:
        lines = []
        for folder, manifest in sorted(((item or {}).get("statistics") or {}).items()):
            manifest = manifest or {}
            comparisons = manifest.get("comparisons") or {}
            lines.append(f"{folder}: {len(comparisons)} comparison(s) -> {manifest.get('directory', '?')}")
        for folder, why in sorted(((item or {}).get("statistics_not_run") or {}).items()):
            lines.append(f"{folder}: NOT RUN -- {why}")
        return lines

    if int(result.get("schema_version") or 1) < 2:
        lines = [f"Statistic: {result.get('statistic')}"]
        lines.extend(f"  {line}" for line in folders(result))
        if result.get("statistic_comparison"):
            lines.append(f"Statistic comparison: {result['statistic_comparison']}")
        return lines

    strata = dict(result.get("strata") or {})
    skipped = list(result.get("skipped_strata") or [])
    lines = [
        f"Stratified by {result.get('stratify_by')!r}: {len(strata)} stratum/strata tested"
        + (f", {len(skipped)} skipped" if skipped else "")
    ]
    for value, item in strata.items():
        lines.append(f"  {value} -> {(item or {}).get('workdir')}")
        lines.extend(f"    {line}" for line in folders(item or {}))
    for item in skipped:
        lines.append(f"  SKIPPED {item.get('value')}: {item.get('reason')}")
    return lines


#: What each stratum contributes to the fan-out INDEX checkpoint. The per-stratum
#: result is filtered down to this, so a key absent here does not reach any
#: downstream reader however faithfully the stratum recorded it.
#:
#: The checkpoint contract every differential stage carries --
#: ``statistics`` (one manifest per statistic folder, which holds each
#: comparison's machine table, report and the folder's power files),
#: ``statistic`` (the --statistic as given) and ``plan`` -- plus
#: ``statistics_not_run``, the statistics this stratum's samples could not
#: support, and ``warnings`` (a t_cell comparison run on one sample per group).
_STRATUM_INDEX_KEYS = frozenset({
    "statistics", "statistic", "plan", "statistics_not_run", "warnings",
    "checkpoint", "plan_id",
})


# ---------------------------------------------------------------------------
# Which submission a stratum result belongs to
# ---------------------------------------------------------------------------
#
# Each submission carries a token. The submitter records it in the gather plan and exports it into every
# array task; a task stamps it into its stratum checkpoint, whether it computed the stratum or reused an
# earlier result that still matches its plan_id. The gather accepts only checkpoints carrying THIS
# submission's token, so a task that died before writing (OOM-killed, walltime, cancelled, a lost node) is
# reported as missing (PLM-0305) instead of being stood in for by an older file of the same name: partial
# output read as complete is the one outcome worse than failing. Reuse is kept: a reusing task re-stamps,
# it does not recompute.

#: The environment variable the array script exports into each task.
SUBMISSION_ENV = "PARALOME_STRATUM_SUBMISSION"

#: The key the token is recorded under, in the gather plan and in the stratum
#: checkpoint alike.
SUBMISSION_KEY = "submission"


def new_submission_token() -> str:
    """A fresh token for one stratum-array submission."""
    import uuid

    return uuid.uuid4().hex


def current_submission() -> Optional[str]:
    """The token of the array this process is a task of, if it is one."""
    import os

    token = os.environ.get(SUBMISSION_ENV, "").strip()
    return token or None


def stamp_submission(result: Dict[str, Any]) -> Dict[str, Any]:
    """Record the current submission in a stratum result about to be written."""
    token = current_submission()
    if token:
        result[SUBMISSION_KEY] = token
    return result


def keep_for_submission(
    existing: Dict[str, Any],
    checkpoint_path: Path,
    write: Callable[[Path, Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Reuse a stratum checkpoint on behalf of the current submission.

    A task that reuses a still-valid result must still say so in the
    checkpoint, or the gather -- which accepts only this submission's token --
    would report a stratum that is in fact complete as missing. ``write`` is
    the modality's own checkpoint writer, so the rewrite is byte-for-byte what
    a fresh run would have produced, plus the token.
    """
    token = current_submission()
    if token and existing.get(SUBMISSION_KEY) != token:
        existing = dict(existing)
        existing[SUBMISSION_KEY] = token
        write(checkpoint_path, existing)
    return existing


def withdraw_checkpoint(
    checkpoint_path: Path,
    reason: str,
    write: Optional[Callable[[Path, Dict[str, Any]], None]] = None,
) -> None:
    """Mark a complete checkpoint as no longer describing what is on disk.

    A stratum that is about to be recomputed rewrites its files under the same
    names its previous checkpoint lists, so a kill part way must not leave the
    old checkpoint vouching for a mixture of two runs. Same convention as
    pipeline.jsonio.invalidate_checkpoint: ``complete`` goes false and the
    reason is recorded, atomically.
    """
    import json

    try:
        previous = json.loads(Path(checkpoint_path).read_text())
    except (OSError, ValueError):
        return
    if not isinstance(previous, dict) or not previous.get("complete"):
        return
    body = dict(previous)
    body["complete"] = False
    body["invalidated"] = reason
    body.pop(SUBMISSION_KEY, None)
    if write is None:
        from paralome.pipeline.jsonio import write_json

        write_json(Path(checkpoint_path), body, atomic=True, default=str)
    else:
        write(Path(checkpoint_path), body)


def withdraw_index(checkpoint_path: Path, *, stage: str, reason: str) -> None:
    """Replace a fan-out's top-level index with an incomplete marker.

    Both index writers (the local fan-out and the SLURM gather) write the
    index only at the END, so the previous run's index is replaced by this
    marker first: a rerun that is killed, whose stratum fails, or whose gather
    raises PLM-0305 must not leave an index saying ``complete: true`` over
    per-stratum files the new run has partly rewritten.

    The marker carries no ``strata`` map on purpose: a reader that walks the
    strata without consulting ``complete`` (atac.motifs reads the map
    directly) must find nothing to read, not the superseded run's paths.
    """
    if not Path(checkpoint_path).is_file():
        return
    import json

    from paralome.pipeline.jsonio import write_json

    try:
        previous = json.loads(Path(checkpoint_path).read_text())
    except (OSError, ValueError):
        previous = {}
    write_json(Path(checkpoint_path), {
        "schema_version": 2,
        "stage": stage,
        "complete": False,
        "invalidated": reason,
        "superseded_plan_id": (previous or {}).get("plan_id") if isinstance(previous, dict) else None,
    }, atomic=True, default=str)


def _write_index(checkpoint_path: Path, payload: Mapping[str, Any]) -> None:
    """Write the v2 index atomically, so a kill mid-write cannot leave a truncated index for the next reader."""
    from paralome.pipeline.jsonio import write_json

    write_json(Path(checkpoint_path), payload, atomic=True, default=str)


@dataclass(frozen=True)
class StratumPlan:
    """Which strata run, which are skipped, and what to say about each.

    Planning is separated from running: everything needed is in the
    sample manifest, so both the in-process thread fan-out and the SLURM array
    path call this, and cannot disagree about which strata exist. The skips and
    the "every stratum was skipped" refusal happen at SUBMISSION.
    """

    runnable: List[str]
    skipped: List[Dict[str, Any]]
    # Dropped by --min-cells-per-stratum rather than by a skip check; they
    # carry a `cells` count the index reports.
    floored: List[Mapping[str, Any]]
    #: stratum -> {statistic folder: why this stratum's samples cannot support it}
    dropped_statistics: Dict[str, Dict[str, str]]
    slices: Dict[str, List[Mapping[str, object]]]


def plan_strata(
    *,
    sample_manifest: Any,
    stratify_by: str,
    stratify_only: Optional[Sequence[str]] = None,
    stratify_skip: Optional[Sequence[str]] = None,
    min_cells_per_stratum: int = 0,
    statistic: str = "t",
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> StratumPlan:
    """Decide which strata are testable, from the manifest alone.

    A stratum is testable by a statistic when
    ``comparisons.plan_differential`` can plan that statistic on the
    stratum's samples -- the call the stratum's own run makes
    (``stats_core.plan_each_statistic``). A stratum no requested statistic can
    be planned on is skipped, with each statistic's reason; one that supports
    only some runs those and records the rest. --covariate-feature's values
    are in the CPM matrix, not the manifest, so here they stand in as
    distinct numbers; the stratum's run checks the real ones.
    """
    import numpy as np

    from paralome.singlecell.stats_core import parse_request, plan_each_statistic

    request = parse_request(
        statistic, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature,
    )
    rows: List[Mapping[str, object]] = list(read_manifest_rows(sample_manifest))

    keep, floored = strata_to_run(
        rows, stratify_by,
        only=stratify_only, skip=stratify_skip,
        min_cells_per_stratum=min_cells_per_stratum,
    )
    all_slices = resolve_strata(rows, stratify_by)
    require_nonempty_strata(
        keep, floored, stratify_by=stratify_by,
        min_cells_per_stratum=min_cells_per_stratum, available=all_slices,
    )
    slices = {value: all_slices[value] for value in keep}

    dropped: Dict[str, Dict[str, str]] = {}
    runnable: List[str] = []
    skipped: List[Dict[str, Any]] = []
    slugs = stratum_slugs(list(slices))

    for value in slices:
        stratum_rows = [dict(row) for row in slices[value]]
        able, refused = plan_each_statistic(
            stratum_rows, request,
            factors=factors, pair_by=pair_by, covariate=covariate,
            covariate_feature=covariate_feature,
            covariate_feature_values=(
                np.arange(len(stratum_rows), dtype=float) if covariate_feature else None
            ),
        )
        if not able:
            skipped.append({
                "value": value, "slug": slugs[value],
                "reason": "; ".join(f"{folder}: {why}" for folder, why in sorted(refused.items())),
            })
            continue
        if refused:
            dropped[value] = refused
        runnable.append(value)

    # Refused here so the SLURM submitters say why before writing anything.
    if not runnable:
        every = list(floored) + skipped
        # A refusal with a number. As a ValueError it reached the
        # user as PLM-0001 "Unexpected internal error".
        raise ParalomeError(
            230,
            f"--stratify-by {stratify_by!r} produced no testable strata; all "
            f"{len(every)} were skipped: "
            + "; ".join(f"{item['value']} ({item['reason']})" for item in every),
            fix="Read each stratum's reason above and change what it names",
        )

    return StratumPlan(
        runnable=runnable, skipped=skipped, floored=list(floored),
        dropped_statistics=dropped, slices=slices,
    )


#: Which modality a stage's results are stamped with. The run-level power
#: figure is named through the same convention as everything else, and this is
#: the only place the stage-to-modality mapping is written down.
_STAGE_MODALITY = {
    "singlecell_rna_differential": "rna",
    "singlecell_differential": "atac_peaks",
}


def assemble_index(
    *,
    stage: str,
    stratify_by: str,
    output_dir: Path,
    checkpoint_path: Path,
    plan: "StratumPlan",
    file_prefix: str,
    # The token of the array this gather belongs to. Required, and
    # None is refused rather than read as "accept anything" -- see the block on
    # SUBMISSION_ENV for the stale-checkpoint failure it closes.
    submission: Optional[str],
    statistic: Optional[str] = None,
) -> Dict[str, Any]:
    """Read each stratum's own checkpoint and write the v2 index over them.

    A SLURM array task writes its own stratum checkpoint, and a dependency job
    runs THIS to produce the index -- the same in shape as the in-process
    path's, because it is the same function.

    A stratum whose checkpoint is missing is reported, not skipped over. An
    array task can fail (OOM, walltime, a node going away) and the whole point
    of a numbered per-stratum job is that the others still count -- but an index
    that quietly omitted a failure would claim a complete run, which is the one
    outcome worse than failing. Only a checkpoint that is complete and carries
    THIS submission's ``submission`` token counts.
    """
    import json

    if not submission:
        # A gather plan written records no token, so nothing
        # can tell this array's results from an earlier run's. Refused rather
        # than indexed on trust; resubmitting writes a plan that has one.
        withdraw_index(
            checkpoint_path, stage=stage,
            reason="gather plan records no submission token; resubmit the stage",
        )
        raise ParalomeError(
            305,
            f"0 of {len(plan.runnable)} strata can be attributed to this submission: "
            "the gather plan records no submission token (written by an earlier "
            "Paralome), so a stratum result cannot be told apart from a previous run's",
            fix="re-issue the stratified differential command so it writes a new plan",
            missing=list(plan.runnable), stage=stage,
        )

    slugs = stratum_slugs(list(plan.slices))
    completed: List[Dict[str, Any]] = []
    missing: List[str] = []
    stale: List[str] = []

    for value in plan.runnable:
        slug = slugs[value]
        stratum_checkpoint = stage_checkpoint_path(
            output_dir, stage, variant=f"{file_prefix}_{slug}"
        )
        try:
            payload = json.loads(stratum_checkpoint.read_text())
        except (OSError, ValueError):
            missing.append(value)
            continue
        if not isinstance(payload, dict) or not payload.get("complete"):
            missing.append(value)
            continue
        if payload.get(SUBMISSION_KEY) != submission:
            # On disk and complete, but from another submission: the task for
            # this stratum never got as far as writing (or re-stamping) it.
            missing.append(value)
            stale.append(value)
            continue
        item: Dict[str, Any] = {
            "value": value, "slug": slug, "workdir": str(output_dir),
            **{k: v for k, v in payload.items() if k in _STRATUM_INDEX_KEYS},
        }
        completed.append(item)

    if missing:
        # The previous index is withdrawn before refusing. This
        # gather's array has already rewritten some strata under the names the
        # old index lists, so leaving it complete:true would vouch for a mix.
        withdraw_index(
            checkpoint_path, stage=stage,
            reason=f"{len(missing)} of {len(plan.runnable)} strata produced no result "
                   "in the latest submission",
        )
        raise ParalomeError(
            305,
            f"{len(missing)} of {len(plan.runnable)} strata produced no result: "
            + ", ".join(missing)
            + (
                f" ({len(stale)} of them have only a checkpoint from an earlier run, "
                "which is not used: " + ", ".join(stale) + ")"
                if stale else ""
            ),
            fix="read the .err file for the failed array task(s)",
            missing=missing, stage=stage,
        )

    _say_what_did_not_run(plan.dropped_statistics)

    payload = index_checkpoint(
        stage=stage, stratify_by=stratify_by,
        completed=completed,
        skipped=list(plan.floored) + list(plan.skipped),
        plan_id="|".join(str(item.get("plan_id", "")) for item in completed),
        extra={# Read by pipeline.state, so a pre-bump index is not complete.
               "method": _sc_diff_method(),
               "workdir": str(output_dir), "statistic": statistic,
               SUBMISSION_KEY: submission,
               },
    )
    _write_index(checkpoint_path, payload)
    payload["checkpoint"] = str(checkpoint_path)
    return payload


def write_gather_plan(path: Path, **fields: Any) -> Path:
    """Record everything the gather needs to reproduce the submitter's plan.

    One file, written once, read once, rather than a dozen CLI flags on the
    gather command, each a chance for the submitter and the gather to disagree
    about which strata were expected -- which would present as an index that
    quietly omits a stratum.

    The manifest's digest is recorded with it (``manifest_sha256``): the
    gather re-plans from the manifest on disk at gather time, and refuses when
    the manifest is not the one the array was planned from (one re-aggregated
    while the array was queued could give a different set of expected strata).
    """
    import json

    fields = dict(fields)
    manifest = fields.get("sample_manifest")
    if manifest and MANIFEST_DIGEST_KEY not in fields:
        try:
            fields[MANIFEST_DIGEST_KEY] = manifest_digest(manifest)
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, sort_keys=True, default=str) + "\n")
    return path


#: The gather-plan key holding the sha256 of the manifest the plan was made from.
MANIFEST_DIGEST_KEY = "manifest_sha256"


def _refuse_a_changed_manifest(fields: Mapping[str, Any], plan_path: Path) -> None:
    """Refuse to gather over a manifest that is not the one submitted."""
    recorded = fields.get(MANIFEST_DIGEST_KEY)
    manifest = fields.get("sample_manifest")
    if not recorded or not manifest:
        return
    try:
        current = manifest_digest(manifest)
    except OSError as exc:
        raise ParalomeError(
            104,
            f"the sample manifest the stratum array was planned from cannot be read at gather "
            f"time: {manifest} ({exc})",
            fix="restore the manifest at that path, then re-issue the stratified differential command",
        ) from exc
    if current != recorded:
        checkpoint_path = fields.get("checkpoint_path")
        if checkpoint_path:
            withdraw_index(
                Path(checkpoint_path), stage=str(fields.get("stage", "")),
                reason="the sample manifest changed between submission and gather",
            )
        raise ParalomeError(
            305,
            f"the sample manifest {manifest} changed after the stratum array was submitted "
            f"(recorded in {plan_path}), so the strata the array ran cannot be matched to the "
            "strata the manifest now describes. Nothing was indexed.",
            fix="re-issue the stratified differential command so the array and its gather are "
                "planned from the same manifest",
            stage=fields.get("stage"),
        )


def run_gather_from_plan(plan_path: Union[str, Path]) -> Dict[str, Any]:
    """Re-plan from the recorded arguments, then index whatever the array wrote.

    Runs as its own small SLURM job, dependent on the array with ``afterany``:
    if three of twenty strata failed, this still runs, and its job is to say
    WHICH three rather than to pretend the run finished.
    """
    import json

    fields = json.loads(Path(plan_path).read_text())
    if "statistic" not in fields:
        # A plan written before the statistic folders records the
        # retired model list, which nothing reads any more.
        raise ParalomeError(
            305,
            f"the gather plan {plan_path} was written by an earlier Paralome (it records no "
            "--statistic), so the strata it expects cannot be re-planned",
            fix="re-issue the stratified differential command so it writes a new plan",
        )
    # The same manifest the submitter planned from, or no index.
    _refuse_a_changed_manifest(fields, Path(plan_path))
    plan = plan_strata(
        sample_manifest=fields["sample_manifest"],
        stratify_by=fields["stratify_by"],
        stratify_only=fields.get("stratify_only"),
        stratify_skip=fields.get("stratify_skip"),
        min_cells_per_stratum=int(fields.get("min_cells_per_stratum") or 0),
        statistic=fields["statistic"],
        factors=fields.get("factors") or (),
        pair_by=fields.get("pair_by"),
        covariate=fields.get("covariate"),
        covariate_feature=fields.get("covariate_feature"),
    )
    return assemble_index(
        stage=fields["stage"], stratify_by=fields["stratify_by"],
        output_dir=Path(fields["output_dir"]),
        checkpoint_path=Path(fields["checkpoint_path"]),
        plan=plan, file_prefix=fields["file_prefix"],
        submission=fields.get(SUBMISSION_KEY),
        statistic=fields["statistic"],
    )


def _say_what_did_not_run(dropped: Mapping[str, Mapping[str, str]]) -> None:
    for value, refused in sorted(dropped.items()):
        print(
            f"[paralome] {value}: ran without " + ", ".join(sorted(refused))
            + " -- " + "; ".join(f"{folder}: {why}" for folder, why in sorted(refused.items())),
            file=sys.stderr,
        )


def run_stratified_fan_out(
    *,
    stage: str,
    sample_manifest: Any,
    output_dir: Path,
    checkpoint_path: Path,
    stratify_by: str,
    # Gains a fourth argument, the CPU share this stratum may use for
    # its own worker pool.
    run_one: Callable[[str, str, Path, int], Dict[str, Any]],
    # The RUN-level prefix, without a stratum -- the name the pooled
    # power figures are written under.
    file_prefix: Optional[str] = None,
    overwrite: bool = False,
    stratify_only: Optional[Sequence[str]] = None,
    stratify_skip: Optional[Sequence[str]] = None,
    min_cells_per_stratum: int = 0,
    # What the planner plans each stratum with -- the same request
    # and companions the stratum's own run receives.
    statistic: str = "t",
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    # The CPU budget for the WHOLE fan-out, split between strata and
    # each stratum's own pool by plan_stratum_concurrency.
    jobs: int = 1,
) -> Dict[str, Any]:
    """Run one differential comparison per stratum and write the v2 index.

    Shared by ``rna.singlecell_stats`` and ``atac.singlecell_stats`` so the two
    modalities cannot drift in layout, skip policy or checkpoint shape. The
    skip check (``plan_strata``) runs BEFORE any statistic does.
    """
    # Planning is shared with the SLURM array path (see plan_strata).
    plan = plan_strata(
        sample_manifest=sample_manifest, stratify_by=stratify_by,
        stratify_only=stratify_only, stratify_skip=stratify_skip,
        min_cells_per_stratum=min_cells_per_stratum,
        statistic=statistic, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature,
    )
    slices = plan.slices
    skipped_by_plan = {item["value"]: item["reason"] for item in plan.skipped}

    def _skip(value: str) -> Optional[str]:
        return skipped_by_plan.get(value)

    # The previous index is withdrawn before any stratum runs, after
    # planning, so a refused plan leaves the old result untouched.
    withdraw_index(checkpoint_path, stage=stage, reason="being recomputed by a new run")

    stratum_workers, per_stratum_jobs = plan_stratum_concurrency(len(slices), jobs)
    completed, skipped = fan_out(
        max_workers=stratum_workers,
        stratum_jobs=per_stratum_jobs,
        values=list(slices),
        run_one=lambda value, slug, results_dir, stratum_jobs: {
            "workdir": str(results_dir),
            **{
                key: value_
                for key, value_ in run_one(value, slug, results_dir, stratum_jobs).items()
                if key in _STRATUM_INDEX_KEYS
            },
        },
        out_dir=output_dir,
        skip_check=_skip,
    )

    skipped = list(plan.floored) + list(skipped)

    if not completed:
        raise ParalomeError(
            230,
            f"--stratify-by {stratify_by!r} produced no testable strata; all "
            f"{len(skipped)} were skipped: "
            + "; ".join(f"{item['value']} ({item['reason']})" for item in skipped),
            fix="Read each stratum's reason above and change what it names",
        )

    _say_what_did_not_run(plan.dropped_statistics)

    payload = index_checkpoint(
        stage=stage, stratify_by=stratify_by,
        completed=completed, skipped=skipped,
        plan_id="|".join(str(item.get("plan_id", "")) for item in completed),
        extra={# Read by pipeline.state, so a pre-bump index is not complete.
               "method": _sc_diff_method(),
               "workdir": str(output_dir),
               "statistic": statistic,
               },
    )
    _write_index(checkpoint_path, payload)
    payload["checkpoint"] = str(checkpoint_path)
    return payload


def _sc_diff_method() -> str:
    """Stats_core.SC_DIFF_METHOD, imported late (stats_core is heavy)."""
    from paralome.singlecell.stats_core import SC_DIFF_METHOD

    return SC_DIFF_METHOD


def index_checkpoint(
    *,
    stage: str,
    stratify_by: str,
    completed: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    plan_id: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The top-level v2 checkpoint payload for a fanned-out stage.

    ``schema_version`` is explicit and absent from v1, which is what the
    summarizers branch on -- inferring the version from the presence of a
    ``strata`` key would work today and break the first time a v1 payload
    gained an unrelated key of that name.
    """
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "complete": True,
        "stage": stage,
        "stratify_by": stratify_by,
        "plan_id": plan_id,
        # The entry is BUILT from _STRATUM_INDEX_KEYS rather than from a hand-written literal, so a key a
        # stratum records reaches the index by construction.
        "strata": {
            str(item["value"]): {
                "slug": item["slug"],
                **{
                    # An absent value stays null, not the word "None".
                    key: (dict(item[key]) if isinstance(item.get(key), dict)
                          else None if key in item and item[key] is None
                          else str(item.get(key, "")))
                    for key in ("workdir", *sorted(_STRATUM_INDEX_KEYS))
                    if key in item or key in ("workdir", "checkpoint", "statistics")
                },
            }
            for item in completed
        },
        "skipped_strata": [dict(item) for item in skipped],
    }
    if extra:
        payload.update(dict(extra))
    return payload
