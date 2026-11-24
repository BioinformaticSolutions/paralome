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

"""Bulk ATAC differential accessibility: peak blocks fitted by the shared engine.

Every statistic --statistic asks for (t, paired-t, lm, anova, in
any combination) is planned by ``paralome.comparisons.plan_differential``,
fitted per peak block by ``paralome.engine.fit_plan`` on log2(CPM +
pseudocount), corrected once over every peak by ``engine.finalize``, and
written by ``paralome.results.write_all_statistics`` into one folder per
statistic under ``<out>/results/`` -- the same names and columns bulk RNA and
the pseudobulk single-cell pipelines write. Nothing here computes a test.

What this module still owns: the block layout (prepare / block / finalize,
locally or as SLURM arrays), the checkpoints, and the stage-level outputs that
are not per statistic -- PCA, the sample-outlier QC, the candidate-gene tables
and the candidate-gene locus views.

The plan is decided ONCE, in the foreground, over every sample of the sheet,
so every refusal (an unusable --factor, a --pair-by that does not pair, a
non-numeric --covariate, a --covariate-feature that is not in the matrix)
happens before a single block is queued. Each block task rebuilds the same
plan from what the plan file records, and everything the plan depends on is
in both the plan id and each block's signature, so a block computed for one
request is never reused for another.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

import numpy as np

from paralome.comparisons import DifferentialPlan, plan_differential
from paralome.engine import ComparisonResult, concatenate, finalize, fit_plan
from paralome.results import Features, Samples, peak_gene, write_all_statistics
from paralome.stats import parse_statistic_request

from paralome.binaries import shell_join, worker_argv
from paralome.pipeline.jsonio import version_stale_reason
from paralome.checkpoints import atomic_write_text, exclusive_checkpoint_lock, stage_checkpoint_path, temporary_beside
from paralome.csvio import skip_comment_lines
from paralome.pipeline.resources import SampleParallelPlan, plan_sample_parallel
from paralome.pipeline.scheduler import detect_scheduler_capabilities, plan_array_tasks, require_array_task_ids
from paralome.pipeline.slurm import SlurmOptions, afterok, compress_array_ids, sbatch, write_sbatch
from paralome.pipeline.naming import slurm_job_name
from paralome.atac.naming import (
    ATAC_PEAK_MODALITY, derive_file_prefix, prefixed_path, resolve_prefixed_file,
    result_path, stage_banner,
)
from paralome.version import __version__


DIFF_PLAN_SCHEMA = "paralome.atac.differential_accessibility_plan.v3"
DIFF_PREPARE_SCHEMA = "paralome.atac.differential_accessibility_prepare.v1"
#: v2: a block's result is a compressed .npz of every column of
#: every comparison the engine fitted, not a TSV of the Student columns.
DIFF_BLOCK_SCHEMA = "paralome.atac.differential_accessibility_block.v2"
#: v6: the checkpoint carries ``statistics`` (one manifest per
#: statistic folder), ``statistic`` and ``plan``.
DIFF_STAGE_SCHEMA = "paralome.atac.differential_accessibility.v6"
#: Recorded in the plan id and in the checkpoint; pipeline/state.py compares it
#: so a result made another way reads out of date and is computed again.
#: The shared engine, one folder per statistic.
DIFF_METHOD = "shared_engine_statistic_folders.v7"
#: In each block's signature: what fitted the block.
BLOCK_ALGORITHM = "paralome.engine.fit_plan.v1"

#: The column every bulk ATAC CPM matrix keys its peaks by.
PEAK_ID_COLUMN = "peak_id"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _identity_matches(identity: Mapping[str, object], path: Path) -> bool:
    try:
        current = _file_identity(path)
    except (FileNotFoundError, OSError):
        return False
    return all(current.get(key) == identity.get(key) for key in ("path", "size", "sha256"))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    # See pipeline.jsonio.CHECKPOINT_VERSION_KEY.
    from paralome.pipeline.jsonio import stamp_version

    atomic_write_text(
        path, json.dumps(stamp_version(payload), indent=2, sort_keys=True, default=str) + "\n"
    )


@contextmanager
def _atomic_text_writer(path: Path, *, newline: Optional[str] = None) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # See temporary_beside; mkstemp left the tables owner-only.
    fd, temporary_path = temporary_beside(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@contextmanager
def _atomic_output_path(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=f".tmp{path.suffix}",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    temporary_path.unlink(missing_ok=True)
    try:
        yield temporary_path
        if not temporary_path.is_file():
            raise RuntimeError(f"Expected output was not created: {temporary_path}")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sniff_cpm_delimiter(path: Path) -> str:
    """CPM matrices are comma-delimited, but an older upstream stage's output
    can still be tab-delimited and named .tsv. Sniff the real delimiter instead
    of assuming comma, so such a stage does not make every downstream stage
    think its sample columns are missing.
    """
    with open(path, "r", newline="") as handle:
        first_line = handle.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","



def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)



def _resolve_project_root_from_cpm_matrix(cpm_matrix: Union[str, Path]) -> Optional[Path]:
    """Best-effort resolution of the Paralome project root from the ATAC CPM
    matrix path.

    Mirrors ``_resolve_candidate_view_gtf``'s ``quant_dir = cpm_matrix.parent.parent``
    (the ``peak_quantification`` stage workdir): one more level up is the project
    root, since ``peak_quantification`` and ``fastq_qc``/``alignment`` are sibling
    stage workdirs. Returns None rather than raising if the
    matrix was supplied from outside that layout.
    """
    try:
        return Path(cpm_matrix).resolve().parents[2]
    except IndexError:
        return None




def _resolve_coverage_summary(cpm_matrix: Path) -> Optional[Path]:
    search_dirs = [cpm_matrix.parent, cpm_matrix.parent / "machine"]
    for bare_name in ("peak_counting_summary.csv", "peak_counting_summary.tsv"):
        # Pre-Patch-109 runs may still have the older tab-delimited name on disk.
        for search_dir in search_dirs:
            found = resolve_prefixed_file(search_dir, bare_name)
            if found is not None:
                return found
    return None


def _read_sample_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(skip_comment_lines(handle))
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError("Sample manifest missing sample_id column")
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Sample manifest row {row_number} has an empty sample_id")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in sample manifest: {sample_id}")
            seen.add(sample_id)
            rows.append({**row, "sample_id": sample_id})
    if not rows:
        raise ValueError(f"Sample manifest contains no samples: {path}")
    return rows


def _matrix_layout(path: Path, sample_ids: Sequence[str]) -> Tuple[List[str], int]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=_sniff_cpm_delimiter(path))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CPM matrix is empty: {path}") from exc
        if len(header) != len(set(header)):
            raise ValueError(f"CPM matrix contains duplicate columns: {path}")
        missing = [sample_id for sample_id in sample_ids if sample_id not in header]
        if missing:
            raise ValueError("CPM matrix missing sample column(s): " + ", ".join(missing))
        count = sum(1 for row in reader if row)
    if count < 1:
        raise ValueError(f"CPM matrix contains no peaks: {path}")
    return header, count


def _balanced_ranges(row_count: int, task_count: int) -> List[Tuple[int, int]]:
    task_count = max(1, min(int(task_count), int(row_count)))
    quotient, remainder = divmod(row_count, task_count)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for index in range(task_count):
        size = quotient + (1 if index < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _covariate_feature_values(
    cpm_path: Path, sample_ids: Sequence[str], pseudocount: float, wanted: str,
) -> Tuple[List[float], str]:
    """``--covariate-feature``'s x: one named peak's log2(CPM + pseudocount) across the samples.

    The matching rule: the first row of the CPM matrix in which ANY identifier
    column (peak_id, the nearest/TSS gene id or name, ...) equals ``wanted``
    exactly. Resolved once, at plan time, and stored in the plan (the values
    and the peak they came from), so a name that is not in the matrix is
    refused before a block is queued and every block regresses on the same x.
    Returns (values, the matched peak_id).
    """
    sample_set = set(sample_ids)
    with cpm_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=_sniff_cpm_delimiter(cpm_path))
        identity_fields = [field for field in (reader.fieldnames or []) if field not in sample_set]
        for raw in reader:
            if any(str(raw.get(field)) == wanted for field in identity_fields):
                values = np.asarray([float(raw[name]) for name in sample_ids], dtype=float)
                return [float(v) for v in np.log2(values + pseudocount)], str(raw.get(PEAK_ID_COLUMN, ""))
    raise ValueError(
        f"--covariate-feature {wanted!r} is not in this CPM matrix under any "
        f"of its identifier columns ({', '.join(identity_fields)})."
    )


def _differential_plan(plan: Mapping[str, object]) -> DifferentialPlan:
    """The DifferentialPlan this stage's plan file records, rebuilt the same way every time.

    The planner, every block task and the finalizer all call this, on the
    same recorded inputs, so they cannot disagree about which comparisons
    exist, which samples each reads, or what they are named.
    """
    spec = dict(plan["differential"])
    covariate = spec.get("covariate")
    covariate_feature = spec.get("covariate_feature")
    request = parse_statistic_request(
        str(spec["statistic"]), factors=list(spec.get("factors") or ()),
        covariate=covariate or covariate_feature, pair_by=spec.get("pair_by"),
    )
    values = spec.get("covariate_feature_values")
    sample_rows = list(plan["sample_rows"])
    return plan_differential(
        sample_rows,
        request=request,
        factors=list(spec.get("factors") or ()),
        pair_by=spec.get("pair_by"),
        covariate=covariate,
        covariate_feature=covariate_feature,
        covariate_feature_values=None if values is None else np.asarray(values, dtype=float),
    )


def _build_differential_accessibility_plan(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    pseudocount: float,
    min_mean_cpm: float,
    alpha: float,
    backend: str,
    threads: int,
    jobs: int,
    alpha_metric: str = "p",
    statistic: str = "t",
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Tuple[Path, Dict[str, object]]:
    if pseudocount < 0 or min_mean_cpm < 0:
        raise ValueError("pseudocount and min_mean_cpm must be >= 0")
    # Log2(0 + 0) = -inf for any zero-CPM peak; PCA then fails in finalize.
    if not pseudocount > 0:
        raise ValueError("--pseudocount must be > 0 (log2 of a zero CPM is undefined at 0)")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    normalized_alpha_metric = alpha_metric.strip().lower()
    if normalized_alpha_metric not in {"p", "q"}:
        raise ValueError("alpha_metric must be 'p' or 'q'")
    if covariate and covariate_feature:
        # One x per regression: taking one and dropping the other would fit
        # something the run did not say.
        raise ValueError(
            f"--covariate {covariate} and --covariate-feature {covariate_feature} were both "
            "given; lm regresses on one of them. Give one"
        )

    cpm_path = Path(cpm_matrix).expanduser().resolve()
    manifest_path = Path(sample_manifest).expanduser().resolve()
    if not cpm_path.is_file():
        raise FileNotFoundError(f"CPM matrix does not exist: {cpm_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sample manifest does not exist: {manifest_path}")

    sample_rows = _read_sample_manifest(manifest_path)
    sample_ids = [row["sample_id"] for row in sample_rows]
    matrix_fields, peak_count = _matrix_layout(cpm_path, sample_ids)
    metadata_fields = [field for field in matrix_fields if field not in sample_ids]
    if PEAK_ID_COLUMN not in metadata_fields:
        raise ValueError(
            f"CPM matrix has no {PEAK_ID_COLUMN!r} column ({cpm_path}); the peak "
            "quantification stage writes one, and every result is keyed by it"
        )

    requested_factors = [str(name) for name in (factors or ())]
    request = parse_statistic_request(
        statistic, factors=requested_factors,
        covariate=covariate or covariate_feature, pair_by=pair_by,
    )
    feature_values: Optional[List[float]] = None
    feature_peak: Optional[str] = None
    if covariate_feature and "lm" in request.statistics:
        feature_values, feature_peak = _covariate_feature_values(
            cpm_path, sample_ids, float(pseudocount), str(covariate_feature),
        )
    differential_spec: Dict[str, object] = {
        "statistic": str(statistic),
        "factors": requested_factors,
        "pair_by": str(pair_by) if pair_by else None,
        "covariate": str(covariate) if covariate else None,
        "covariate_feature": str(covariate_feature) if covariate_feature else None,
        "covariate_feature_peak": feature_peak,
        "covariate_feature_values": feature_values,
    }
    # Every refusal the plan makes happens here, before a block is queued.
    differential = _differential_plan({"differential": differential_spec, "sample_rows": sample_rows})

    requested_task_count = jobs if int(jobs) > 0 else threads
    task_count = min(peak_count, max(1, int(requested_task_count)))
    resource_plan = plan_sample_parallel(
        total_threads=threads,
        sample_count=task_count,
        max_threads_per_sample=1,
        jobs=jobs,
    )

    out = Path(output_dir).expanduser().resolve()
    for directory in (
        out,
        out / "inputs",
        out / "blocks",
        out / "results",
        out / "results" / "machine",
        out / "checkpoints",
        out / "scripts",
        out / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, object]] = []
    for task_id, (start_row, end_row) in enumerate(_balanced_ranges(peak_count, task_count), start=1):
        token = f"block_{task_id:06d}"
        tasks.append({
            "task_id": task_id,
            "token": token,
            "start_row": start_row,
            "end_row": end_row,
            "row_count": end_row - start_row,
            "input": str(out / "blocks" / f"{token}.input.tsv"),
            "statistics": str(out / "blocks" / f"{token}.results.npz"),
            "checkpoint": str(stage_checkpoint_path(out, f"{token}")),
        })

    results = out / "results"
    # Every --out directory is a unique "project ID" per comparison by convention, so each output
    # file's prefix is derived from that directory's own basename. Checkpoints and blocks/ are NOT
    # prefixed -- prepare/block/finalize tasks read them back from fixed relative paths.
    file_prefix = derive_file_prefix(out)
    plan: Dict[str, object] = {
        "schema": DIFF_PLAN_SCHEMA,
        "stage": "differential_accessibility",
        "backend": backend,
        "file_prefix": file_prefix,
        "cpm_matrix": str(cpm_path),
        "cpm_identity": _file_identity(cpm_path),
        "sample_manifest": str(manifest_path),
        "sample_manifest_identity": _file_identity(manifest_path),
        "sample_rows": sample_rows,
        "sample_ids": sample_ids,
        "matrix_fields": matrix_fields,
        "metadata_fields": metadata_fields,
        "peak_count": peak_count,
        "pseudocount": float(pseudocount),
        "min_mean_cpm": float(min_mean_cpm),
        "alpha": float(alpha),
        "alpha_metric": normalized_alpha_metric,
        "alpha_value": float(alpha),
        # Everything the DifferentialPlan is rebuilt from, in one
        # place; every block task and the finalizer rebuild it from this.
        "statistic": str(statistic),
        "differential": differential_spec,
        "differential_plan": differential.as_record(),
        "threads": int(threads),
        "jobs": int(jobs),
        "resource_plan": resource_plan.to_dict(),
        "tasks": tasks,
        "output_dir": str(out),
        "outputs": {
            # Checkpoints stay at fixed, unprefixed relative paths --
            # prepare/block/finalize tasks locate them independently of plan_id.
            "prepare_checkpoint": str(stage_checkpoint_path(out, "differential_accessibility_prepare")),
            "checkpoint": str(stage_checkpoint_path(out, "differential_accessibility")),
            "results_dir": str(results),
        },
    }
    plan["plan_id"] = _stable_digest({
        "cpm_identity": plan["cpm_identity"],
        "sample_manifest_identity": plan["sample_manifest_identity"],
        "sample_rows": sample_rows,
        "pseudocount": pseudocount,
        "min_mean_cpm": min_mean_cpm,
        "alpha": alpha,
        "alpha_metric": normalized_alpha_metric,
        # The whole request -- statistic, factors, pairing, covariate, the covariate feature and its
        # values, the arms -- is in this hash and in the block signature, so a rerun that changes any
        # of them never reuses old blocks.
        "differential": differential_spec,
        "tasks": [
            {"task_id": task["task_id"], "start_row": task["start_row"], "end_row": task["end_row"]}
            for task in tasks
        ],
        "method": DIFF_METHOD,
    })
    plan_path = out / "inputs" / "differential_accessibility_plan.json"
    _atomic_json(plan_path, plan)
    return plan_path, plan


def _read_block_input(task: Mapping[str, object], plan: Mapping[str, object]) -> Tuple[List[int], List[Dict[str, str]], np.ndarray]:
    sample_ids = list(plan["sample_ids"])
    path = Path(str(task["input"]))
    indices: List[int] = []
    metadata: List[Dict[str, str]] = []
    values: List[List[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = ["_row_index", *list(plan["matrix_fields"])]
        if list(reader.fieldnames or []) != expected_fields:
            raise RuntimeError(f"Invalid differential block input header: {path}")
        for row in reader:
            indices.append(int(row["_row_index"]))
            metadata.append({key: row[key] for key in plan["metadata_fields"]})
            values.append([float(row[sample_id]) for sample_id in sample_ids])
    expected_indices = list(range(int(task["start_row"]), int(task["end_row"])))
    if indices != expected_indices:
        raise RuntimeError(f"Differential block row order is invalid: {path}")
    return indices, metadata, np.asarray(values, dtype=float)


def _write_prepared_blocks(plan: Mapping[str, object]) -> List[Dict[str, object]]:
    tasks = list(plan["tasks"])
    matrix_fields = list(plan["matrix_fields"])
    task_cursor = 0
    row_counts = [0 for _ in tasks]
    with ExitStack() as stack:
        writers: List[csv.DictWriter] = []
        for task in tasks:
            handle = stack.enter_context(_atomic_text_writer(Path(str(task["input"])), newline=""))
            writer = csv.DictWriter(
                handle,
                fieldnames=["_row_index", *matrix_fields],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writers.append(writer)
        cpm_matrix_path = Path(str(plan["cpm_matrix"]))
        with cpm_matrix_path.open(newline="") as source:
            # The master CPM matrix is comma-delimited going forward,
            # but an in-progress project can still have a pre-Patch-109
            # tab-delimited one from an already-completed upstream stage --
            # sniff instead of assuming comma. The per-block scratch file this
            # writes to stays tab-delimited internal plumbing either way.
            reader = csv.DictReader(source, delimiter=_sniff_cpm_delimiter(cpm_matrix_path))
            if list(reader.fieldnames or []) != matrix_fields:
                raise RuntimeError("CPM matrix header changed after differential planning")
            for row_index, row in enumerate(reader):
                while row_index >= int(tasks[task_cursor]["end_row"]):
                    task_cursor += 1
                output = {"_row_index": row_index, **row}
                writers[task_cursor].writerow(output)
                row_counts[task_cursor] += 1
    prepared: List[Dict[str, object]] = []
    for task, observed_count in zip(tasks, row_counts):
        if observed_count != int(task["row_count"]):
            raise RuntimeError(
                f"Prepared block {task['task_id']} has {observed_count} rows; expected {task['row_count']}"
            )
        path = Path(str(task["input"]))
        prepared.append({
            "task_id": int(task["task_id"]),
            "input": str(path),
            "rows": observed_count,
            "sha256": _sha256_file(path),
        })
    return prepared


def _validate_preparation_checkpoint(plan: Mapping[str, object]) -> Optional[Dict[str, object]]:
    checkpoint_path = Path(str(plan["outputs"]["prepare_checkpoint"]))
    if not checkpoint_path.is_file():
        return None
    if not _identity_matches(plan["cpm_identity"], Path(str(plan["cpm_matrix"]))):
        return None
    if not _identity_matches(plan["sample_manifest_identity"], Path(str(plan["sample_manifest"]))):
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("schema") != DIFF_PREPARE_SCHEMA or checkpoint.get("complete") is not True:
            return None
        if version_stale_reason(checkpoint) is not None:
            return None
        if checkpoint.get("plan_id") != plan.get("plan_id"):
            return None
        prepared = checkpoint.get("blocks") or []
        if len(prepared) != len(plan["tasks"]):
            return None
        for task, record in zip(plan["tasks"], prepared):
            path = Path(str(task["input"]))
            if record.get("task_id") != task.get("task_id") or not path.is_file():
                return None
            if record.get("sha256") != _sha256_file(path):
                return None
            indices, _, _ = _read_block_input(task, plan)
            if len(indices) != int(task["row_count"]):
                return None
        return checkpoint
    except Exception:
        return None


def _run_prepare_unlocked(plan: Mapping[str, object], *, overwrite: bool = False) -> Dict[str, object]:
    if not overwrite:
        cached = _validate_preparation_checkpoint(plan)
        if cached is not None:
            return {**cached, "reused": True}
    checkpoint_path = Path(str(plan["outputs"]["prepare_checkpoint"]))
    try:
        if not _identity_matches(plan["cpm_identity"], Path(str(plan["cpm_matrix"]))):
            raise RuntimeError("CPM matrix changed after differential planning")
        if not _identity_matches(plan["sample_manifest_identity"], Path(str(plan["sample_manifest"]))):
            raise RuntimeError("Sample manifest changed after differential planning")
        prepared = _write_prepared_blocks(plan)
        result: Dict[str, object] = {
            "schema": DIFF_PREPARE_SCHEMA,
            "stage": "differential_accessibility_prepare",
            "complete": True,
            "plan_id": str(plan["plan_id"]),
            "peaks": int(plan["peak_count"]),
            "block_count": len(plan["tasks"]),
            "blocks": prepared,
            "worker_pid": os.getpid(),
            "reused": False,
        }
        _atomic_json(checkpoint_path, result)
        return result
    except Exception as exc:
        _atomic_json(checkpoint_path, {
            "schema": DIFF_PREPARE_SCHEMA,
            "stage": "differential_accessibility_prepare",
            "complete": False,
            "plan_id": str(plan["plan_id"]),
            "error": f"{type(exc).__name__}: {exc}",
            "worker_pid": os.getpid(),
        })
        raise


def run_differential_accessibility_prepare_task(
    plan_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).expanduser().resolve().read_text())
    checkpoint_path = Path(str(plan["outputs"]["prepare_checkpoint"]))
    with exclusive_checkpoint_lock(checkpoint_path):
        return _run_prepare_unlocked(plan, overwrite=overwrite)


def _block_task_signature(
    task: Mapping[str, object],
    plan: Mapping[str, object],
    preparation: Mapping[str, object],
) -> str:
    prepared_record = next(
        record for record in preparation["blocks"] if int(record["task_id"]) == int(task["task_id"])
    )
    spec = dict(plan["differential"])
    return _stable_digest({
        "plan_id": plan["plan_id"],
        "task_id": task["task_id"],
        "start_row": task["start_row"],
        "end_row": task["end_row"],
        "input_sha256": prepared_record["sha256"],
        "statistics_algorithm": BLOCK_ALGORITHM,
        # Everything the fit depends on, by name, beside the plan
        # id that also holds it -- a block made for one request is never read
        # as another's.
        "statistic": spec.get("statistic"),
        "factors": list(spec.get("factors") or ()),
        "pair_by": spec.get("pair_by"),
        "covariate": spec.get("covariate"),
        "covariate_feature": spec.get("covariate_feature"),
        "covariate_feature_values": spec.get("covariate_feature_values"),
        "pseudocount": float(plan["pseudocount"]),
        "min_mean_cpm": float(plan["min_mean_cpm"]),
        "alpha": float(plan["alpha_value"]),
    })


def _fit_block(
    task: Mapping[str, object], plan: Mapping[str, object], differential: DifferentialPlan,
) -> List[ComparisonResult]:
    """Every comparison of every requested statistic, fitted on this block's peaks.

    Y is log2(CPM + pseudocount), raw the CPM itself; --min-mean-cpm is the
    engine's expression floor on raw, judged per comparison over the samples
    that comparison reads.
    """
    _indices, _metadata, cpm = _read_block_input(task, plan)
    log_values = np.log2(cpm + float(plan["pseudocount"]))
    return fit_plan(
        log_values, differential, raw=cpm, min_mean=float(plan["min_mean_cpm"]),
        alpha=float(plan["alpha_value"]), replication_unit="samples",
    )


def _result_key(folder: str, comparison: str, column: str) -> str:
    return f"{folder}/{comparison}/{column}"


def _write_block_results(path: Path, results: Sequence[ComparisonResult]) -> None:
    """One block's results: every column of every comparison, keyed folder/comparison/column."""
    arrays: Dict[str, np.ndarray] = {}
    meta: List[Dict[str, object]] = []
    for result in results:
        for column, values in result.columns.items():
            arrays[_result_key(result.folder, result.name, column)] = np.asarray(values)
        meta.append({
            "statistic": result.statistic,
            "folder": result.folder,
            "comparison": result.name,
            "columns": list(result.columns),
            "facts": result.facts,
        })
    arrays["__meta__"] = np.asarray(json.dumps(meta, sort_keys=True, default=str))
    with _atomic_output_path(path) as temporary_path:
        np.savez_compressed(str(temporary_path), **arrays)


def _read_block_results(
    path: Path, differential: DifferentialPlan, expected_rows: int,
) -> List[ComparisonResult]:
    """A block's results, in the plan's order, refused when they are not the plan's comparisons."""
    comparisons = [
        (statistic_plan, comparison)
        for statistic_plan in differential.statistics
        for comparison in statistic_plan.comparisons
    ]
    with np.load(str(path), allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        recorded = [(entry["folder"], entry["comparison"]) for entry in meta]
        expected = [(sp.folder, c.name) for sp, c in comparisons]
        if recorded != expected:
            raise RuntimeError(
                f"differential block {path} holds comparisons {recorded}; the plan makes {expected}"
            )
        results: List[ComparisonResult] = []
        for entry, (statistic_plan, comparison) in zip(meta, comparisons):
            columns = {
                column: np.asarray(data[_result_key(entry["folder"], entry["comparison"], column)])
                for column in entry["columns"]
            }
            if any(len(values) != int(expected_rows) for values in columns.values()):
                raise RuntimeError(f"differential block {path} does not hold {expected_rows} peaks")
            results.append(ComparisonResult(
                statistic=statistic_plan.statistic, folder=statistic_plan.folder,
                comparison=comparison, columns=columns, facts=dict(entry.get("facts") or {}),
            ))
    return results


def _validate_block_checkpoint(
    task: Mapping[str, object],
    plan: Mapping[str, object],
    preparation: Mapping[str, object],
    differential: Optional[DifferentialPlan] = None,
) -> Optional[Dict[str, object]]:
    checkpoint_path = Path(str(task["checkpoint"]))
    statistics_path = Path(str(task["statistics"]))
    if not checkpoint_path.is_file() or not statistics_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("schema") != DIFF_BLOCK_SCHEMA or checkpoint.get("complete") is not True:
            return None
        if version_stale_reason(checkpoint) is not None:
            return None
        if checkpoint.get("plan_id") != plan.get("plan_id"):
            return None
        if int(checkpoint.get("task_id", -1)) != int(task["task_id"]):
            return None
        if checkpoint.get("task_signature") != _block_task_signature(task, plan, preparation):
            return None
        if checkpoint.get("statistics") != str(statistics_path):
            return None
        if checkpoint.get("statistics_sha256") != _sha256_file(statistics_path):
            return None
        if int(checkpoint.get("rows", -1)) != int(task["row_count"]):
            return None
        _read_block_results(
            statistics_path, differential or _differential_plan(plan), int(task["row_count"]),
        )
        return checkpoint
    except Exception:
        return None


def _run_block_unlocked(
    task: Mapping[str, object],
    plan: Mapping[str, object],
    preparation: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    differential = _differential_plan(plan)
    if not overwrite:
        cached = _validate_block_checkpoint(task, plan, preparation, differential)
        if cached is not None:
            return {**cached, "reused": True}
    checkpoint_path = Path(str(task["checkpoint"]))
    statistics_path = Path(str(task["statistics"]))
    signature = _block_task_signature(task, plan, preparation)
    try:
        results = _fit_block(task, plan, differential)
        _write_block_results(statistics_path, results)
        rows = len(results[0]) if results else 0
        if rows != int(task["row_count"]):
            raise RuntimeError(
                f"differential block {task['task_id']} fitted {rows} peaks; expected {task['row_count']}"
            )
        result: Dict[str, object] = {
            "schema": DIFF_BLOCK_SCHEMA,
            "stage": "differential_accessibility_block",
            "complete": True,
            "plan_id": str(plan["plan_id"]),
            "task_id": int(task["task_id"]),
            "start_row": int(task["start_row"]),
            "end_row": int(task["end_row"]),
            "rows": rows,
            "comparisons": len(results),
            "task_signature": signature,
            "statistics": str(statistics_path),
            "statistics_sha256": _sha256_file(statistics_path),
            "worker_pid": os.getpid(),
            "reused": False,
        }
        _atomic_json(checkpoint_path, result)
        return result
    except Exception as exc:
        _atomic_json(checkpoint_path, {
            "schema": DIFF_BLOCK_SCHEMA,
            "stage": "differential_accessibility_block",
            "complete": False,
            "plan_id": str(plan["plan_id"]),
            "task_id": int(task["task_id"]),
            "task_signature": signature,
            "error": f"{type(exc).__name__}: {exc}",
            "worker_pid": os.getpid(),
        })
        raise


def run_differential_accessibility_block_task(
    plan_path: Union[str, Path],
    task_id: int,
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    resolved_plan = Path(plan_path).expanduser().resolve()
    plan = json.loads(resolved_plan.read_text())
    tasks = list(plan["tasks"])
    if task_id < 1 or task_id > len(tasks):
        raise ValueError(f"task_id must be between 1 and {len(tasks)}")
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        raise RuntimeError("Differential-accessibility preparation is incomplete or invalid")
    task = tasks[task_id - 1]
    checkpoint_path = Path(str(task["checkpoint"]))
    with exclusive_checkpoint_lock(checkpoint_path):
        return _run_block_unlocked(task, plan, preparation, overwrite=overwrite)


def _run_block_payload(task_id: int, plan_path: Union[str, Path], overwrite: bool) -> Dict[str, object]:
    return run_differential_accessibility_block_task(plan_path, task_id, overwrite=overwrite)


def _mark_stage_incomplete(
    plan: Mapping[str, object],
    pending_task_ids: Sequence[int],
    *,
    preparation_required: bool,
) -> None:
    checkpoint_path = Path(str(plan["outputs"]["checkpoint"]))
    _atomic_json(checkpoint_path, {
        "schema": DIFF_STAGE_SCHEMA,
        "stage": "differential_accessibility",
        "complete": False,
        "plan_id": str(plan["plan_id"]),
        "preparation_required": bool(preparation_required),
        "pending_task_ids": list(pending_task_ids),
        "block_count": len(plan["tasks"]),
        "resource_plan": plan["resource_plan"],
    })


def _pending_task_ids(
    plan: Mapping[str, object],
    *,
    overwrite: bool,
) -> Tuple[bool, List[int]]:
    if overwrite:
        return True, [int(task["task_id"]) for task in plan["tasks"]]
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        return True, [int(task["task_id"]) for task in plan["tasks"]]
    differential = _differential_plan(plan)
    pending = [
        int(task["task_id"])
        for task in plan["tasks"]
        if _validate_block_checkpoint(task, plan, preparation, differential) is None
    ]
    return False, pending


#: What bulk ATAC wrote, one table and one workbook for the
#: Student t, beside the results root. Removed when the stage finalizes, so a
#: superseded table is not left looking current next to the statistic folders.
_LEGACY_OUTPUTS = (
    ("machine", "all_peaks_statistics.csv"),
    ("machine", "all_peaks_statistics.tsv"),
    ("machine", "result_file_summary.csv"),
    ("machine", "candidate_gene_results.csv"),
    ("machine", "power_summary_atac.csv"),
    ("", "student_t_results.xlsx"),
    ("", "power_coverage_atac.svg"),
    ("", "power_coverage_atac.pdf"),
    ("", "power_coverage_atac.csv"),
    ("", "power_by_n_atac.svg"),
    ("", "power_by_n_atac.pdf"),
    ("", "power_by_n_atac.csv"),
)
_LEGACY_SUBSET = re.compile(r"^student_[pq]_[0-9][0-9.eE+-]*\.csv$")


def _remove_legacy_outputs(results_dir: Path, file_prefix: Optional[str]) -> None:
    candidates = [results_dir / sub / prefixed_path(Path(name), file_prefix).name
                  for sub, name in _LEGACY_OUTPUTS]
    machine = results_dir / "machine"
    if machine.is_dir():
        lead = f"{file_prefix}_" if file_prefix else ""
        for path in machine.iterdir():
            if path.name.startswith(lead) and _LEGACY_SUBSET.match(path.name[len(lead):]):
                candidates.append(path)
    for path in candidates:
        if path.is_file():
            path.unlink()
            print(f"[paralome] differential_accessibility: removed {path.name}, written by an "
                  "older Paralome; the results are one folder per statistic under "
                  f"{results_dir}.", file=sys.stderr, flush=True)


def _remove_statistic_folders_not_run(
    results_dir: Path, file_prefix: str, kept: Sequence[str],
) -> None:
    """A statistic an earlier run of this prefix asked for and this one did not: remove its files.

    The shared writer removes what an earlier run wrote in a folder this run
    writes again; a folder this run does not write at all would otherwise be
    left whole beside the current results, reading as current.
    """
    from paralome.results import remove_stale_results

    if not results_dir.is_dir():
        return
    for folder in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if folder.name in kept or folder.name in ("machine", "candidate_gene_views"):
            continue
        manifest = result_path(folder / "machine", f"{folder.name}_manifest.json",
                               file_prefix=file_prefix, modality=ATAC_PEAK_MODALITY)
        if not manifest.is_file():
            continue
        remove_stale_results(manifest, [], f"differential_accessibility {folder.name}")
        _candidate_results_path(folder, file_prefix).unlink(missing_ok=True)
        manifest.unlink()


def _candidate_results_path(folder: Path, file_prefix: str) -> Path:
    return result_path(folder, "candidate_gene_results.csv",
                       file_prefix=file_prefix, modality=ATAC_PEAK_MODALITY)


def _cell(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if np.isfinite(number) else ""



def _view_comparison(results: Sequence[ComparisonResult]) -> Optional[ComparisonResult]:
    """The comparison the locus views show: the first statistic's main comparison, else its first."""
    if not results:
        return None
    first_folder = results[0].folder
    own = [result for result in results if result.folder == first_folder]
    return next((result for result in own if result.comparison.role == "main"), own[0])


def finalize_differential_accessibility_plan(plan_path: Union[str, Path]) -> Dict[str, object]:
    resolved_plan = Path(plan_path).expanduser().resolve()
    plan = json.loads(resolved_plan.read_text())
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        raise RuntimeError("Differential-accessibility preparation is incomplete or invalid")
    differential = _differential_plan(plan)

    blocks: List[List[ComparisonResult]] = []
    block_checkpoints: List[Dict[str, object]] = []
    metadata: List[Dict[str, str]] = []
    cpm_parts: List[np.ndarray] = []
    for task in plan["tasks"]:
        checkpoint = _validate_block_checkpoint(task, plan, preparation, differential)
        if checkpoint is None:
            raise RuntimeError(f"Differential block {task['task_id']} is incomplete or invalid")
        block_checkpoints.append(checkpoint)
        blocks.append(_read_block_results(Path(str(task["statistics"])), differential, int(task["row_count"])))
        _indices, block_metadata, block_cpm = _read_block_input(task, plan)
        metadata.extend(block_metadata)
        cpm_parts.append(block_cpm)
    cpm = np.vstack(cpm_parts)
    if cpm.shape[0] != int(plan["peak_count"]):
        raise RuntimeError("Differential block outputs do not form one complete ordered peak set")

    alpha_metric = str(plan.get("alpha_metric", "q"))
    alpha_value = float(plan.get("alpha_value", plan["alpha"]))
    results = finalize(concatenate(blocks), alpha_metric=alpha_metric, alpha_value=alpha_value)

    outputs = plan["outputs"]
    out = Path(str(plan["output_dir"]))
    results_dir = out / "results"
    file_prefix = str(plan.get("file_prefix") or derive_file_prefix(out))
    _remove_legacy_outputs(results_dir, file_prefix)

    sample_rows = list(plan["sample_rows"])
    sample_ids = [str(row["sample_id"]) for row in sample_rows]

    genes = [peak_gene(row) for row in metadata]
    features = Features(
        key=[str(row[PEAK_ID_COLUMN]) for row in metadata],
        key_column=PEAK_ID_COLUMN,
        gene_id=[gene_id for gene_id, _name in genes],
        gene_name=[gene_name for _id, gene_name in genes],
        annotation=metadata,
        peaks=True,
    )
    manifests = write_all_statistics(
        results,
        root=results_dir,
        file_prefix=file_prefix,
        stamp=ATAC_PEAK_MODALITY,
        features=features,
        samples=Samples(ids=sample_ids, values=cpm, unit="CPM"),
        version=__version__,
        prefer_raw_means=True,
        raw_fpkm=False,
        alpha_metric=alpha_metric,
        alpha_value=alpha_value,
        stage="differential_accessibility",
    )
    _remove_statistic_folders_not_run(results_dir, file_prefix, list(manifests))

    checkpoint_path = Path(str(outputs["checkpoint"]))
    result: Dict[str, object] = {
        "schema": DIFF_STAGE_SCHEMA,
        "stage": "differential_accessibility",
        "complete": True,
        "plan_id": str(plan["plan_id"]),
        "method": DIFF_METHOD,  # Lets the pipeline summary spot older results
        # The BRIEF contract every differential stage carries.
        "statistics": manifests,
        "statistic": str(plan["differential"]["statistic"]),
        "plan": differential.as_record(),
        "peaks": int(cpm.shape[0]),
        "results_dir": str(results_dir),
        "block_count": len(plan["tasks"]),
        "completed_blocks": len(block_checkpoints),
        "resource_plan": plan["resource_plan"],
        "multiple_testing": "fdr_bh per comparison",
    }
    with exclusive_checkpoint_lock(checkpoint_path):
        _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    return result


def run_differential_accessibility_finalize_task(plan_path: Union[str, Path]) -> Dict[str, object]:
    return finalize_differential_accessibility_plan(plan_path)


def prepare_differential_accessibility_slurm_plan(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    pseudocount: float = 1.0,
    min_mean_cpm: float = 1.0,
    alpha: float = 0.05,
    threads: int = 1,
    jobs: int = 0,
    partition: Optional[str] = None,
    mem: str = "8G",
    overwrite: bool = False,
    alpha_metric: str = "p",
    statistic: str = "t",
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Dict[str, object]:
    plan_path, plan = _build_differential_accessibility_plan(
        cpm_matrix,
        sample_manifest,
        output_dir,
        pseudocount=pseudocount,
        min_mean_cpm=min_mean_cpm,
        alpha=alpha,
        backend="slurm",
        threads=threads,
        jobs=jobs,
        alpha_metric=alpha_metric,
        statistic=statistic,
        factors=factors,
        pair_by=pair_by,
        covariate=covariate,
        covariate_feature=covariate_feature,
    )
    preparation_required, pending = _pending_task_ids(plan, overwrite=overwrite)
    if preparation_required or pending:
        _mark_stage_incomplete(plan, pending, preparation_required=preparation_required)
    capabilities = detect_scheduler_capabilities("slurm")
    require_array_task_ids(pending, capabilities, stage="Differential accessibility")
    concurrency = int(plan["resource_plan"]["concurrent_jobs"])
    array_plan = plan_array_tasks(
        task_count=len(pending),
        concurrency=concurrency,
        capabilities=capabilities,
    )
    plan["preparation_required"] = preparation_required
    plan["pending_task_ids"] = pending
    plan["scheduler_capabilities"] = capabilities.to_dict()
    plan["array_plan"] = array_plan.to_dict()
    _atomic_json(plan_path, plan)

    cli = shell_join(worker_argv())
    out = Path(str(plan["output_dir"]))
    overwrite_flag = " --overwrite" if overwrite else ""
    prepare_script = write_sbatch(
        out / "scripts" / "differential_accessibility_prepare.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(out, "atac_diff_prepare"),
            cpus_per_task=1,
            mem=mem,
            partition=partition,
            output=out / "logs" / "atac_diff_prepare_%j.out",
            error=out / "logs" / "atac_diff_prepare_%j.err",
        ),
        f"{cli} atac differential-prepare --plan {shlex.quote(str(plan_path))}{overwrite_flag}",
    )
    array_scripts: List[Path] = []
    for batch in array_plan.batches:
        task_ids = pending[batch.first_task - 1: batch.last_task]
        array_spec = compress_array_ids(task_ids, throttle=min(batch.throttle, len(task_ids)))
        array_scripts.append(write_sbatch(
            out / "scripts" / f"differential_accessibility_blocks_{batch.batch_index:03d}.sbatch",
            SlurmOptions(
                job_name=slurm_job_name(out, f"atac_diff_{batch.batch_index:03d}"),
                cpus_per_task=1,
                mem=mem,
                partition=partition,
                array=array_spec,
                output=out / "logs" / f"atac_diff_{batch.batch_index:03d}_%A_%a.out",
                error=out / "logs" / f"atac_diff_{batch.batch_index:03d}_%A_%a.err",
            ),
            f'{cli} atac differential-block --plan {shlex.quote(str(plan_path))} '
            f'--task-id "$SLURM_ARRAY_TASK_ID"{overwrite_flag}',
        ))
    finalize_script = write_sbatch(
        out / "scripts" / "differential_accessibility_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(out, "atac_diff_finalize"),
            cpus_per_task=1,
            mem=mem,
            partition=partition,
            output=out / "logs" / "atac_diff_finalize_%j.out",
            error=out / "logs" / "atac_diff_finalize_%j.err",
        ),
        f"{cli} atac differential-finalize --plan {shlex.quote(str(plan_path))}",
    )
    return {
        "plan": plan_path,
        "prepare_script": prepare_script,
        "array_scripts": array_scripts,
        "finalize_script": finalize_script,
        "preparation_required": preparation_required,
        "task_count": len(pending),
        "total_blocks": len(plan["tasks"]),
        "reused_blocks": len(plan["tasks"]) - len(pending),
        "concurrency": concurrency,
        "resource_plan": plan["resource_plan"],
        "scheduler_capabilities": capabilities.to_dict(),
        "array_plan": array_plan.to_dict(),
        "plan_id": plan["plan_id"],
    }


def submit_differential_accessibility_slurm_plan(
    plan_info: Mapping[str, object],
    *,
    submit: bool,
) -> Dict[str, object]:
    result = dict(plan_info)
    result.update({"submitted": False, "job_id": None, "submitted_job_ids": [], "complete": False})
    if not submit:
        return result
    # Each job ID is reported the moment its sbatch returns, so a refused later sbatch never leaves
    # queued jobs unnamed. stderr, which the supervisor reads as well, so the JSON on stdout stays whole.
    prepare_job_id = sbatch(plan_info["prepare_script"])
    print(f"Submitted differential-accessibility prepare job: {prepare_job_id}", file=sys.stderr, flush=True)
    dependency: Optional[str] = afterok(prepare_job_id)
    block_job_ids: List[str] = []
    for script in plan_info["array_scripts"]:
        job_id = sbatch(script, dependency=dependency)
        print(f"Submitted differential-accessibility array job: {job_id}", file=sys.stderr, flush=True)
        block_job_ids.append(job_id)
        dependency = afterok(job_id)
    finalize_job_id = sbatch(plan_info["finalize_script"], dependency=dependency)
    print(f"Submitted differential-accessibility finalizer job: {finalize_job_id}", file=sys.stderr, flush=True)
    job_ids = [prepare_job_id, *block_job_ids, finalize_job_id]
    result.update({
        "submitted": True,
        "job_id": ",".join(job_ids),
        "submitted_job_ids": job_ids,
        "prepare_job_id": prepare_job_id,
        "block_job_ids": block_job_ids,
        "finalize_job_id": finalize_job_id,
    })
    return result


def run_differential_accessibility(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    pseudocount: float = 1.0,
    min_mean_cpm: float = 1.0,
    alpha: float = 0.05,
    threads: int = 1,
    jobs: int = 0,
    overwrite: bool = False,
    alpha_metric: str = "p",
    statistic: str = "t",
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Dict[str, object]:
    print(
        stage_banner(
            "differential_accessibility", derive_file_prefix(Path(output_dir)), output_dir,
        ),
        flush=True,
    )
    plan_path, plan = _build_differential_accessibility_plan(
        cpm_matrix,
        sample_manifest,
        output_dir,
        pseudocount=pseudocount,
        min_mean_cpm=min_mean_cpm,
        alpha=alpha,
        backend="local",
        threads=threads,
        jobs=jobs,
        alpha_metric=alpha_metric,
        statistic=statistic,
        factors=factors,
        pair_by=pair_by,
        covariate=covariate,
        covariate_feature=covariate_feature,
    )
    preparation_required, pending_ids = _pending_task_ids(plan, overwrite=overwrite)
    if preparation_required or pending_ids:
        _mark_stage_incomplete(plan, pending_ids, preparation_required=preparation_required)
    prepare_result = run_differential_accessibility_prepare_task(plan_path, overwrite=overwrite)
    plan = json.loads(plan_path.read_text())
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        raise RuntimeError("Differential-accessibility preparation failed validation")
    differential = _differential_plan(plan)
    pending = [
        task for task in plan["tasks"]
        if overwrite or _validate_block_checkpoint(task, plan, preparation, differential) is None
    ]
    statuses: List[Dict[str, object]] = []
    resource_plan = SampleParallelPlan(**plan["resource_plan"])
    if len(pending) <= 1 or resource_plan.concurrent_jobs <= 1:
        for task in pending:
            try:
                statuses.append(_run_block_payload(int(task["task_id"]), plan_path, overwrite))
            except Exception as exc:
                raise RuntimeError(f"Differential block {task['task_id']} failed: {exc}") from exc
    else:
        with ProcessPoolExecutor(max_workers=min(resource_plan.concurrent_jobs, len(pending))) as executor:
            future_to_task = {
                executor.submit(_run_block_payload, int(task["task_id"]), plan_path, overwrite): task
                for task in pending
            }
            try:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        statuses.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(f"Differential block {task['task_id']} failed: {exc}") from exc
            except Exception:
                for future in future_to_task:
                    future.cancel()
                raise

    result = finalize_differential_accessibility_plan(plan_path)
    result.update({
        "preparation_reused": bool(prepare_result.get("reused")),
        "executed_blocks": len(pending),
        "reused_blocks": len(plan["tasks"]) - len(pending),
        "block_task_results": sorted(statuses, key=lambda item: int(item["task_id"])),
        "concurrency": resource_plan.concurrent_jobs,
        "peak_cpus": resource_plan.peak_cpus,
    })
    return result
