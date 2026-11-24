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

"""Single-cell differential statistics: the snATAC driver and the cell-level fits.

The request is ``--statistic`` -- any combination of t, paired-t,
lm and anova -- on the pseudobulk values (the default) or, with ``:cell``, on
the cells. Both levels answer the SAME plan
(``paralome.comparisons.plan_differential``), and every statistic writes its
own folder through ``paralome.results``, named by the flag that asked for it:
``t``, ``paired_t``, ``lm``, ``anova``; ``t_cell`` ... ``anova_cell``.

PSEUDOBULK. The design's log2 CPM goes to ``paralome.engine.fit_plan`` -- the
fit bulk RNA and bulk ATAC run -- so a pseudobulk folder reads exactly as a
bulk one (``singlecell.stats_core``).

CELLS (``fit_cell_plan``), each statistic fitted with the cells as the
observations and returned as the engine's ``ComparisonResult``:

* t_cell -- Student's t on the per-cell log2 CPM of the comparison's two
  groups' cells (``engine.fit_two_groups``, i.e. ``stats.student_t_from_moments``).
* paired_t_cell -- Per pair, the mean per-cell log2 CPM of the treatment
  sample's cells minus that of the control sample's cells; the differences are
  tested across pairs (``engine.fit_paired``, i.e. ``stats.paired_t_from_pairs``).
* lm_cell -- OLS of per-cell log2 CPM on the covariate with the cells as N
  (``engine.fit_slope``, i.e. ``stats.ols_covariate_fit``): the sample's
  --covariate value on each of its cells, or the cell's own log2 CPM of
  --covariate-feature.
* anova_cell -- anova is what is used when there ARE interactions. One
  --factor: the one-way ANOVA. Two or more: the factorial, every interaction
, in the treatment coding of ``stats.design_matrix``.
   So
  the cells are the observations and a sample's cells are pooled with its
  replicates': least squares over the cells (``engine.fit_anova_cells``, from
  each sample's sums), Type II F per term and every contrast on
  df = cells - columns, every feature, vectorised. The random-intercept model
  it had, its feature subset and --peak-subset-limit are
  retired.

The groups of every statistic are the --factor columns' levels, each against
the control level its --factor names.

The vectorised cell statistics are chunked over features
(--naive-cell-t-peak-chunk-size) and every cell statistic is spread over
--threads worker processes (``_run_peak_chunks``).
"""

import json
import math
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from paralome.errors import ParalomeError
from paralome.checkpoints import exclusive_checkpoint_lock, stage_checkpoint_path
from paralome.pipeline.jsonio import version_stale_reason
from paralome.atac.naming import (
    ATAC_PEAK_MODALITY,
    assert_results_are_stamped,
    derive_file_prefix,
    naming_signature,
    run_scoped_prefix,
    stage_banner,
)
from paralome.singlecell.strata import (
    keep_for_submission,
    resolve_one_stratum,
    split_stratify_values,
    stamp_submission,
    stratum_slugs,
    withdraw_checkpoint,
)
from paralome.atac.singlecell import (
    SinglecellInputs,
    _atomic_json,
    _stable_digest,
    load_singlecell_inputs,
)
from paralome.singlecell.stats_core import (
    DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE,
    SC_DIFF_METHOD,
    compute_statistics,
    covariate_feature_row,
    input_file_identity,
    kept_comparison_samples,
    load_pseudobulk_comparison,
    parse_request,
    peak_features,
    stamped_files,
    statistics_on_disk,
)
from paralome.stats import DATA_LEVEL_CELL

SC_DIFF_SCHEMA = "paralome.atac.singlecell_differential.v2"

#: Every key in this stage's checkpoint that maps to the files it wrote. The
#: reuse guard requires them present AND every file they list on disk, so an
#: output kind added later invalidates prior checkpoints by construction.
_CHECKPOINT_OUTPUT_KEYS = ("statistics",)


# --------------------------------------------------------------------------
# The cells of the design
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellLevelComparison:
    """The cells of every design sample, and the features to fit on them.

    ``peak_ids``/``peak_positions`` are the features (peaks or genes) and their
    rows in the extraction's counts matrix; ``cell_positions`` the cells'
    columns; ``cell_sample`` each cell's design-row position (its pseudobulk
    sample's row in the plan); ``library_size`` each cell's total count over
    every feature.
    """

    peak_ids: List[str]
    peak_positions: np.ndarray
    cell_positions: np.ndarray
    cell_sample: np.ndarray
    sample_ids: List[str]
    library_size: np.ndarray

    def restricted(self, rows: Sequence[int]) -> "CellLevelComparison":
        """The same cells, fitted on a subset of the features (rows of this one)."""
        rows = list(rows)
        return replace(
            self, peak_ids=[self.peak_ids[i] for i in rows],
            peak_positions=np.asarray(self.peak_positions, dtype=int)[rows],
        )

    def cells_of(self, samples: Sequence[int]) -> np.ndarray:
        """Column positions (into this comparison's cells) of the cells of ``samples``."""
        return np.flatnonzero(np.isin(self.cell_sample, np.asarray(list(samples), dtype=int)))


def _normalised_metadata_value(value: object) -> str:
    """One spelling for a value read from the manifest CSV and from the cell metadata TSV.

    The manifest is written by csv.DictWriter and the cell metadata
    read by pandas, so 3 may arrive as "3" on one side and "3.0" on the other,
    and a missing value as "" or "nan". Numbers compare as numbers and the
    missing spellings compare equal; anything else compares as stripped text.
    """
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "<na>", "na"):
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return repr(number)


def _cell_sample_positions(
    metadata: "pd.DataFrame",
    kept_samples: Sequence[Mapping[str, object]],
) -> "pd.Series":
    """The design-row position of each cell's pseudobulk sample; NaN for every other cell.

    A cell belongs to a sample when it carries that sample's value in every
    manifest column the cell metadata also has. The match is CHECKED: each
    sample must get exactly its manifest n_cells, or the cell statistics would
    test a different cohort from the pseudobulk ones.
    """
    def _refuse(code: int, detail: str) -> ParalomeError:
        return ParalomeError(
            code, detail,
            fix="use the sample_manifest.csv that pseudobulk wrote from this same extraction "
                "(re-run pseudobulk if the extraction has changed since)",
        )

    if not kept_samples:
        raise _refuse(210, "No pseudobulk sample is in scope, so the cell-level statistics have no cells.")

    def _source(column: str) -> Optional[str]:
        if column == "barcode" or column not in metadata.columns:
            return None
        return column

    shared = None
    for sample in kept_samples:
        columns = [c for c in sample["values"] if _source(str(c)) is not None]
        shared = columns if shared is None else [c for c in shared if c in columns]
    shared = list(dict.fromkeys(shared or []))
    if not shared:
        raise _refuse(
            210,
            "The sample manifest has no column the cell metadata also has, so the cells of each "
            "pseudobulk sample cannot be identified for the cell-level statistics.",
        )

    separator = "\x1f"
    cell_key = None
    for column in shared:
        raw = metadata[_source(column)].astype(str)
        spelling = {value: _normalised_metadata_value(value) for value in raw.unique()}
        part = raw.map(spelling)
        cell_key = part if cell_key is None else cell_key + separator + part

    position_by_key: Dict[str, int] = {}
    sample_by_key: Dict[str, Mapping[str, object]] = {}
    for sample in kept_samples:
        key = separator.join(_normalised_metadata_value(sample["values"].get(c, "")) for c in shared)
        if key in sample_by_key:
            raise _refuse(
                210,
                f"Pseudobulk samples {sample_by_key[key]['sample_id']!r} and {sample['sample_id']!r} have the "
                f"same values in every column the cell metadata shares ({', '.join(shared)}), so their "
                f"cells cannot be told apart for the cell-level statistics.",
            )
        sample_by_key[key] = sample
        position_by_key[key] = int(sample["position"])

    found = cell_key.isin(list(sample_by_key))
    if not found.any():
        raise ParalomeError(
            206,
            "No cell of any pseudobulk sample in scope was found in the cell metadata, matching on "
            f"{', '.join(shared)}. The samples are drawn from values the cell metadata does not hold.",
            fix="use the sample_manifest.csv written from this extraction, or drop the empty "
                "stratum with --stratify-skip",
        )
    counts = cell_key[found].value_counts()
    wrong = []
    for key, sample in sample_by_key.items():
        got = int(counts.get(key, 0))
        if got != int(sample["n_cells"]):
            wrong.append(f"{sample['sample_id']} ({got} cells matched, manifest n_cells {sample['n_cells']})")
    if wrong:
        raise _refuse(
            210,
            "The cell metadata does not hold exactly the cells the sample manifest says each "
            f"pseudobulk sample has, matching on {', '.join(shared)}: " + "; ".join(wrong[:5])
            + (f" (+{len(wrong) - 5} more)" if len(wrong) > 5 else "")
            + ". The cell-level statistics would test a different cohort from the pseudobulk ones.",
        )
    return cell_key.map(position_by_key)


def build_cell_level_comparison(
    inputs: SinglecellInputs,
    feature_ids: Sequence[str],
    kept_samples: Sequence[Mapping[str, object]],
    *,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    extraction_feature_ids: Optional[Sequence[str]] = None,
) -> CellLevelComparison:
    """The cells of every design sample (``kept_comparison_samples``), and the features.

    ``extraction_feature_ids`` is the extraction's own feature list in
    ``inputs.matrix`` row order (RNA: ``[gene.feature_name for gene in
    inputs.genes]``); by default the ATAC peaks.
    """
    metadata = inputs.metadata
    position = _cell_sample_positions(metadata, kept_samples)
    mask = position.notna()
    if stratify_by is not None and stratify_value is not None:
        from paralome.stratify import frame_in_stratum

        mask = mask & frame_in_stratum(metadata, stratify_by, stratify_value).to_numpy()
    selected = metadata.loc[mask]

    barcode_to_position = {barcode: index for index, barcode in enumerate(inputs.cells)}
    cell_positions = np.array([barcode_to_position[b] for b in selected["barcode"]], dtype=int)
    cell_sample = position.loc[selected.index].to_numpy().astype(int)
    library_size = np.asarray(inputs.matrix.sum(axis=0)).ravel()[cell_positions].astype(float)

    if extraction_feature_ids is not None:
        feature_to_row = {feature_id: index for index, feature_id in enumerate(extraction_feature_ids)}
    else:
        feature_to_row = {peak.peak_id: index for index, peak in enumerate(inputs.peaks)}
    missing = [f for f in feature_ids if f not in feature_to_row]
    if missing:
        raise ValueError(f"Feature id(s) not present in the extraction's feature set: {missing[:5]}")
    ordered = sorted(kept_samples, key=lambda sample: int(sample["position"]))
    return CellLevelComparison(
        peak_ids=list(feature_ids),
        peak_positions=np.array([feature_to_row[f] for f in feature_ids], dtype=int),
        cell_positions=cell_positions, cell_sample=cell_sample,
        sample_ids=[str(sample["sample_id"]) for sample in ordered],
        library_size=library_size,
    )


_selected = (None, None, None)   # (inputs, cell_level, their cells' counts as CSR)
_selected_lock = threading.Lock()


def _selected_cells(inputs, cell_level):
    """The comparison's cells' counts, features x cells in cell_level order, as CSR -- made once per comparison.

    The cells are taken once (a column slice, cheap on CSC) and each block of
    features is a row slice of that (cheap on CSR), rather than row-slicing the
    whole object's CSC matrix for every block."""
    global _selected
    with _selected_lock:
        held_inputs, held_level, held = _selected
        if held_inputs is not inputs or held_level is not cell_level:
            held = inputs.matrix[:, cell_level.cell_positions].tocsr()
            _selected = (inputs, cell_level, held)
        return held


def _cell_values(
    inputs, cell_level: CellLevelComparison, start: int, stop: int, pseudocount: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """(log2 CPM, CPM) of features [start, stop) in every cell: features x cells."""
    rows = cell_level.peak_positions[start:stop]
    counts = _selected_cells(inputs, cell_level)[rows].toarray().astype(float)
    library = cell_level.library_size[np.newaxis, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cpm = np.where(library > 0, counts / library * 1_000_000.0, 0.0)
    return np.log2(cpm + pseudocount), cpm


# --------------------------------------------------------------------------
# Multiprocess chunking for the cell-level fits.
#
# `_run_peak_chunks` splits the features into up to `jobs` contiguous shards and
# either runs one shard in-process (jobs <= 1 on the main thread) or fans the
# shards out across a `ProcessPoolExecutor`. `inputs`/`cell_level` go to each
# worker once, through the pool's initializer.
# --------------------------------------------------------------------------


def _slurm_clamped_jobs(requested_jobs: int) -> int:
    """Cap a requested worker count at SLURM's own CPU allocation for this job.

    Running more processes than SLURM knows about can crash a node; SLURM sets
    SLURM_CPUS_PER_TASK in every job's environment, so it is read here, where
    the workers are made, whatever a caller passed.
    """
    requested = max(1, int(requested_jobs or 1))
    slurm_limit = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_limit:
        try:
            limit = int(slurm_limit)
        except ValueError:
            limit = 0
        if limit > 0:
            return min(requested, limit)
    return requested


def _peak_chunk_bounds(n_peaks: int, jobs: int) -> List[Tuple[int, int]]:
    """Split [0, n_peaks) into up to `jobs` contiguous, roughly equal shards."""
    if n_peaks <= 0:
        return []
    jobs = max(1, min(_slurm_clamped_jobs(jobs), n_peaks))
    base, remainder = divmod(n_peaks, jobs)
    bounds: List[Tuple[int, int]] = []
    start = 0
    for i in range(jobs):
        size = base + (1 if i < remainder else 0)
        if size <= 0:
            continue
        bounds.append((start, start + size))
        start += size
    return bounds


_worker_inputs = None  # type: Optional[SinglecellInputs]
_worker_cell_level = None  # type: Optional[CellLevelComparison]


def _init_peak_chunk_worker(inputs, cell_level) -> None:
    global _worker_inputs, _worker_cell_level
    _worker_inputs = inputs
    _worker_cell_level = cell_level


def _run_peak_chunks(inputs, cell_level, jobs: int, chunk_fn, chunk_args: tuple) -> List[Dict[str, object]]:
    """Dispatch `chunk_fn(start, stop, *chunk_args)` over every feature of `cell_level`."""
    n_peaks = len(cell_level.peak_ids)
    bounds = _peak_chunk_bounds(n_peaks, jobs)
    if not bounds:
        return []
    if len(bounds) == 1 and threading.current_thread() is threading.main_thread():
        # In the calling thread only when nothing else is running: off
        # the main thread means the stratified fan-out is going, its siblings
        # want the GIL, and one shard goes to a subprocess (fork shares the
        # counts matrix copy-on-write).
        _init_peak_chunk_worker(inputs, cell_level)
        return [chunk_fn(bounds[0][0], bounds[0][1], *chunk_args)]
    with ProcessPoolExecutor(
        max_workers=len(bounds), initializer=_init_peak_chunk_worker, initargs=(inputs, cell_level)
    ) as executor:
        futures = [executor.submit(chunk_fn, start, stop, *chunk_args) for start, stop in bounds]
        return [future.result() for future in futures]


# --------------------------------------------------------------------------
# t_cell, paired_t_cell, lm_cell: vectorised over features, chunked
# --------------------------------------------------------------------------


def _untested(keys: Sequence[str], n: int) -> Dict[str, np.ndarray]:
    columns = {key: np.full(n, np.nan) for key in keys}
    columns["tested"] = np.zeros(n, dtype=bool)
    return columns


def _sample_means(values: np.ndarray, cell_sample: np.ndarray, n_samples: int) -> np.ndarray:
    """Mean over each sample's cells: features x samples (NaN for a sample with no cell)."""
    indicator = np.zeros((cell_sample.shape[0], n_samples))
    indicator[np.arange(cell_sample.shape[0]), cell_sample] = 1.0
    counts = indicator.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, (values @ indicator) / counts, np.nan)


def _fit_vector_block(
    y: np.ndarray, raw: np.ndarray, cell_level: CellLevelComparison, plan,
    eligible: Mapping[Tuple[str, str], np.ndarray], alpha: float,
    covariate_cells: Optional[np.ndarray],
) -> List[object]:
    """t_cell / paired_t_cell / lm_cell / anova_cell for one block of features, in plan order."""
    from paralome import engine
    from paralome.comparisons import KIND_FTEST

    results: List[object] = []
    n = y.shape[0]
    n_samples = len(plan.rows)
    sample_means = None
    for statistic_plan in plan.statistics:
        statistic = statistic_plan.statistic
        if statistic == "anova" and statistic_plan.one_way:
            # each level's cells are its
            # observations; no random intercept (with one sample per level it
            # and the level are the same column).
            groups = [cell_level.cells_of(members) for members in statistic_plan.level_members]
            first = statistic_plan.comparisons[0].name
            fitted = engine.fit_one_way(
                y, raw, statistic_plan, eligible[(statistic_plan.folder, first)], groups,
            )
            samples_in = {level: len(m) for level, m in zip(statistic_plan.levels, statistic_plan.level_members)}
            for comparison in statistic_plan.comparisons:
                columns = fitted[comparison.name]
                if comparison.kind != KIND_FTEST:
                    columns["n_control_samples"] = np.full(n, float(samples_in[str(comparison.control)]))
                    columns["n_treatment_samples"] = np.full(n, float(samples_in[str(comparison.treatment)]))
                results.append(engine.ComparisonResult(
                    statistic=statistic, folder=statistic_plan.folder, comparison=comparison,
                    columns=columns, facts=engine.comparison_facts(statistic_plan, comparison, "cells"),
                ))
            continue
        if statistic == "anova" and statistic_plan.cells_as_n:
            # There are only two models for Seurat data, pseudobulk and cell; in the cell model the
            # replicates' cells are pooled as they are in Seurat. The cells are the observations,
            # every feature, from each sample's sums over its cells.
            indicator = np.zeros((cell_level.cell_sample.shape[0], n_samples))
            indicator[np.arange(cell_level.cell_sample.shape[0]), cell_level.cell_sample] = 1.0
            counts = indicator.sum(axis=0)
            first = statistic_plan.comparisons[0].name
            fitted = engine.fit_anova_cells(
                y @ indicator, np.einsum("ij,ij->i", y, y), counts, raw @ indicator,
                statistic_plan, eligible[(statistic_plan.folder, first)],
            )
            for comparison in statistic_plan.comparisons:
                columns = fitted[comparison.name]
                if comparison.kind != KIND_FTEST:
                    columns["n_control_samples"] = np.full(n, float(len(comparison.control_index)))
                    columns["n_treatment_samples"] = np.full(n, float(len(comparison.treatment_index)))
                results.append(engine.ComparisonResult(
                    statistic=statistic, folder=statistic_plan.folder, comparison=comparison,
                    columns=columns, facts=engine.comparison_facts(statistic_plan, comparison, "cells"),
                ))
            continue
        if statistic == "anova":
            continue
        for comparison in statistic_plan.comparisons:
            mask = eligible[(statistic_plan.folder, comparison.name)]
            unit = "cells"
            if statistic == "t":
                cells = replace(
                    comparison, pairs=(),
                    control_index=tuple(cell_level.cells_of(comparison.control_index)),
                    treatment_index=tuple(cell_level.cells_of(comparison.treatment_index)),
                )
                columns = engine.fit_two_groups(y, raw, cells, mask, alpha)
                columns["n_control_samples"] = np.full(n, float(len(comparison.control_index)))
                columns["n_treatment_samples"] = np.full(n, float(len(comparison.treatment_index)))
            elif statistic == "paired-t":
                # Per pair: the treatment sample's cells' mean minus the control
                # sample's; those differences are what is tested, across pairs.
                if sample_means is None:
                    sample_means = (
                        _sample_means(y, cell_level.cell_sample, n_samples),
                        _sample_means(raw, cell_level.cell_sample, n_samples),
                    )
                columns = engine.fit_paired(sample_means[0], sample_means[1], comparison, mask, alpha)
                # The test's replicate is the pair of samples, so its effect
                # size and detectable effect are in samples, not cells.
                unit = "samples"
            elif statistic == "lm":
                members = cell_level.cells_of(comparison.regressed_index)
                x = np.asarray(covariate_cells, dtype=float)
                if members.size < 3 or np.ptp(x[members]) == 0:
                    columns = _untested(
                        [*engine.CONTRAST_COLUMNS, *engine.EXTRA_COLUMNS["lm"]], n,
                    )
                    for drop in ("significant", "q_value"):
                        columns.pop(drop, None)
                else:
                    cells = replace(
                        comparison, pairs=(), regressed_index=tuple(members),
                        control_index=tuple(cell_level.cells_of(comparison.control_index)),
                        treatment_index=tuple(cell_level.cells_of(comparison.treatment_index)),
                    )
                    columns = engine.fit_slope(
                        y, raw, replace(statistic_plan, covariate_values=x), cells, mask,
                    )
            else:  # pragma: no cover - the parser accepts nothing else
                raise ValueError(statistic)
            results.append(engine.ComparisonResult(
                statistic=statistic, folder=statistic_plan.folder, comparison=comparison,
                columns=columns,
                facts=engine.comparison_facts(statistic_plan, comparison, unit),
            ))
    return results


def _vector_chunk(
    start: int, stop: int, plan, eligible: Mapping[Tuple[str, str], np.ndarray],
    alpha: float, pseudocount: float, chunk_size: int, covariate_cells: Optional[np.ndarray],
) -> Dict[str, object]:
    """The vectorised cell statistics for features [start, stop), densifying
    ``chunk_size`` features' per-cell counts at a time."""
    from paralome import engine

    inputs = _worker_inputs
    cell_level = _worker_cell_level
    size = chunk_size if chunk_size and chunk_size > 0 else max(stop - start, 1)
    blocks = []
    for inner_start in range(start, stop, size):
        inner_stop = min(inner_start + size, stop)
        y, raw = _cell_values(inputs, cell_level, inner_start, inner_stop, pseudocount)
        blocks.append(_fit_vector_block(
            y, raw, cell_level, plan,
            {key: mask[inner_start:inner_stop] for key, mask in eligible.items()},
            alpha, covariate_cells,
        ))
    return {"start": start, "stop": stop, "results": engine.concatenate(blocks)}


# --------------------------------------------------------------------------
# The plan, fitted on the cells
# --------------------------------------------------------------------------


def _expand(results: Sequence[object], rows: Sequence[int], n_total: int) -> List[object]:
    """Results computed on ``rows`` of the features, placed into full-length columns."""
    from paralome import engine

    rows = np.asarray(list(rows), dtype=int)
    out = []
    for result in results:
        columns: Dict[str, np.ndarray] = {}
        for name, values in result.columns.items():
            values = np.asarray(values)
            if values.dtype == bool:
                full = np.zeros(n_total, dtype=bool)
            elif values.dtype == object:
                full = np.full(n_total, None, dtype=object)
            else:
                full = np.full(n_total, np.nan)
            full[rows] = values
            columns[name] = full
        out.append(engine.ComparisonResult(
            statistic=result.statistic, folder=result.folder, comparison=result.comparison,
            columns=columns, facts=dict(result.facts),
        ))
    return out


def _floor_masks(plan, cpm: np.ndarray, min_mean_cpm: float) -> Dict[Tuple[str, str], np.ndarray]:
    """The expression floor, per comparison, on the pseudobulk CPM of the samples
    that comparison reads -- the engine's rule, so a feature is tested at the
    cell level where it is tested at the pseudobulk level."""
    masks: Dict[Tuple[str, str], np.ndarray] = {}
    every = list(range(cpm.shape[1]))
    for statistic_plan in plan.statistics:
        for comparison in statistic_plan.comparisons:
            samples = every if statistic_plan.statistic == "anova" else list(comparison.samples())
            masks[(statistic_plan.folder, comparison.name)] = cpm[:, samples].mean(axis=1) >= float(min_mean_cpm)
    return masks


def fit_cell_plan(
    inputs,
    cell_level: CellLevelComparison,
    comparison,
    plan,
    *,
    min_mean_cpm: float,
    alpha: float,
    pseudocount: float = 1.0,
    peak_chunk_size: int = DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE,
    jobs: int = 1,
    covariate_feature: Optional[str] = None,
) -> List[object]:
    """Every statistic of ``plan`` fitted with the cells as observations (uncorrected).

    ``comparison`` is the stage's ``PseudobulkComparison`` (its features are
    ``cell_level``'s, in order); the expression floor is judged on its CPM.
    Every statistic, the factorial ANOVA included, with
    the cells as N and a sample's cells pooled with its replicates' -- 
    """
    from paralome import engine

    n_features = len(cell_level.peak_ids)
    masks = _floor_masks(plan, comparison.cpm, min_mean_cpm)
    results: List[object] = []

    vector_plans = list(plan.statistics)
    if vector_plans:
        keys = [(p.folder, c.name) for p in vector_plans for c in p.comparisons]
        universe = np.flatnonzero(np.any([masks[key] for key in keys], axis=0))
        covariate_cells = None
        lm_plan = next((p for p in vector_plans if p.statistic == "lm"), None)
        if lm_plan is not None:
            if lm_plan.covariate_kind == "feature":
                row = covariate_feature_row(comparison.feature_ids, comparison.annotation, str(covariate_feature))
                values, _raw = _cell_values(inputs, cell_level, row, row + 1, pseudocount)
                covariate_cells = values[0]
            else:
                covariate_cells = np.asarray(lm_plan.covariate_values, dtype=float)[cell_level.cell_sample]
        if not universe.size:
            # Nothing reaches the floor in any comparison: one feature is still
            # fitted, flagged untested, so every comparison writes its table.
            universe = np.arange(min(1, n_features))
        subset = cell_level.restricted(universe)
        chunks = _run_peak_chunks(
            inputs, subset, jobs, _vector_chunk,
            (plan, {key: masks[key][universe] for key in keys}, alpha, pseudocount,
             peak_chunk_size, covariate_cells),
        )
        chunks.sort(key=lambda chunk: chunk["start"])
        joined = engine.concatenate([chunk["results"] for chunk in chunks])
        results.extend(_expand(joined, universe, n_features))

    return results


def make_cell_fitter(
    *,
    load_inputs: Callable[[], object],
    comparison,
    sample_manifest: Union[str, Path],
    stratify_by: Optional[str],
    stratify_value: Optional[str],
    extraction_feature_ids: Optional[Callable[[object], Sequence[str]]],
    min_mean_cpm: float,
    alpha: float,
    pseudocount: float,
    peak_chunk_size: int,
    jobs: int,
    covariate_feature: Optional[str],
    report: Dict[str, object],
) -> Callable[[object], List[object]]:
    """``fit_cells(plan)`` for ``stats_core.compute_statistics``: one implementation
    of the cell-level path for both drivers."""
    def fit_cells(plan) -> List[object]:
        # Inputs that hold several strata's cells only as needed (the trajectory segments) give the
        # stratum's own.
        for_stratum = getattr(load_inputs, "for_stratum", None)
        inputs = for_stratum(stratify_value) if for_stratum is not None else load_inputs()
        cell_level = build_cell_level_comparison(
            inputs, comparison.feature_ids,
            kept_comparison_samples(sample_manifest, comparison),
            stratify_by=stratify_by, stratify_value=stratify_value,
            extraction_feature_ids=None if extraction_feature_ids is None else extraction_feature_ids(inputs),
        )
        report["n_cells"] = int(cell_level.cell_positions.size)
        report["n_samples"] = len(cell_level.sample_ids)
        return fit_cell_plan(
            inputs, cell_level, comparison, plan,
            min_mean_cpm=min_mean_cpm, alpha=alpha, pseudocount=pseudocount,
            peak_chunk_size=peak_chunk_size,
            jobs=jobs, covariate_feature=covariate_feature,
        )

    return fit_cells


# --------------------------------------------------------------------------
# The snATAC driver
# --------------------------------------------------------------------------


def run_singlecell_differential(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    extraction_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    statistic: str = "t",
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    stratify_skip: Optional[str] = None,
    # `atac singlecell-differential` passes it unconditionally.
    min_cells_per_stratum: int = 0,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    naive_cell_t_peak_chunk_size: int = DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE,
    alpha: float = 0.05,
    alpha_metric: str = "p",
    pseudocount: float = 1.0,
    min_mean_cpm: float = 1.0,
    overwrite: bool = False,
    jobs: int = 1,
    file_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Run every statistic --statistic asks for on one snATAC comparison.

    ``cpm_matrix``/``sample_manifest`` are the pseudobulk outputs;
    ``extraction_dir`` the Seurat extraction (read only for ``:cell``). Each
    statistic writes its own folder under ``out_dir`` (``<out>/<folder>/``).
    ``jobs`` is the worker processes for the cell-level fits.
    """
    alpha_metric = alpha_metric.strip().lower()
    if alpha_metric not in {"p", "q"}:
        raise ValueError("alpha_metric must be 'p' or 'q'")
    from paralome.singlecell.stats_core import refuse_a_factor_aggregation_dropped

    refuse_a_factor_aggregation_dropped(sample_manifest, extraction_dir, factors)
    request = parse_request(
        statistic, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature,
    )
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A SLURM array task runs one stratum with --stratify-value and has
    # no flag for file_prefix, so it is derived here through the SAME
    # run_scoped_prefix the in-process fan-out uses.
    _single = split_stratify_values(stratify_value) if stratify_by else []
    if file_prefix is None and len(_single) == 1:
        file_prefix = run_scoped_prefix(
            derive_file_prefix(output_dir), None,
            stratum_slugs(_single)[_single[0]],
        )
    checkpoint_variant = file_prefix
    file_prefix = file_prefix or derive_file_prefix(output_dir)
    checkpoint_path = stage_checkpoint_path(
        output_dir, "singlecell_differential", variant=checkpoint_variant,
    )
    # One write per line, so strata fanned out in threads do not run their lines together.
    print(stage_banner("singlecell_differential", file_prefix, output_dir) + "\n", end="", flush=True)

    stratify_values = split_stratify_values(stratify_value)
    if stratify_by and len(stratify_values) != 1:
        from paralome.singlecell.strata import run_stratified_fan_out

        return run_stratified_fan_out(
            stage="singlecell_differential",
            file_prefix=file_prefix,
            sample_manifest=sample_manifest, output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            stratify_by=stratify_by, overwrite=overwrite,
            stratify_only=stratify_values or None, stratify_skip=stratify_skip,
            min_cells_per_stratum=min_cells_per_stratum,
            statistic=statistic, factors=factors, pair_by=pair_by,
            covariate=covariate, covariate_feature=covariate_feature,
            jobs=jobs,
            # The fan-out's per-stratum call is the second implementation of this signature's contract:
            # a parameter added above must be added here too, or every stratum runs without it.
            run_one=lambda value, slug, results_dir, _per_stratum_jobs: run_singlecell_differential(
                cpm_matrix, sample_manifest, extraction_dir, results_dir,
                statistic=statistic,
                stratify_by=stratify_by, stratify_value=value,
                factors=factors, pair_by=pair_by, covariate=covariate,
                covariate_feature=covariate_feature,
                naive_cell_t_peak_chunk_size=naive_cell_t_peak_chunk_size,
                alpha=alpha, alpha_metric=alpha_metric,
                pseudocount=pseudocount, min_mean_cpm=min_mean_cpm,
                # The per-stratum SHARE of the CPU budget.
                overwrite=overwrite, jobs=_per_stratum_jobs,
                file_prefix=run_scoped_prefix(file_prefix, None, slug),
            ),
        )

    # The slug spelling in every filename becomes the raw metadata spelling
    # ('CD8 TEM' vs CD8_TEM) before anything filters on it.
    if stratify_by and stratify_values:
        stratify_value = resolve_one_stratum(sample_manifest, stratify_by, stratify_values[0])

    plan_id = _stable_digest({
        "cpm_matrix": str(Path(cpm_matrix).expanduser().resolve()),
        "sample_manifest": str(Path(sample_manifest).expanduser().resolve()),
        "extraction_dir": str(Path(extraction_dir).expanduser().resolve()),
        # What the files hold, not only where they are.
        "inputs_identity": [input_file_identity(cpm_matrix), input_file_identity(sample_manifest)],
        "stratify_by": stratify_by,
        "stratify_value": stratify_value,
        "statistic": statistic,
        "factors": list(factors),
        "pair_by": pair_by,
        "covariate": covariate,
        "covariate_feature": covariate_feature,
        "alpha": alpha,
        "alpha_metric": alpha_metric,
        "pseudocount": pseudocount if request.has_cell else None,
        "min_mean_cpm": min_mean_cpm,
        # The prefix and the naming convention name every
        # output file, so a different one is a different result.
        "file_prefix": file_prefix,
        "naming": naming_signature(file_prefix=file_prefix, modality=ATAC_PEAK_MODALITY),
        # Which computation produced the result.
        "method": SC_DIFF_METHOD,
    })
    # jobs and the chunk size are NOT in plan_id: they change how a result is
    # computed, never the result.
    if not overwrite and checkpoint_path.is_file():
        existing = json.loads(checkpoint_path.read_text())
        # Plan_id covers what the stage was ASKED for, the
        # version check WHICH CODE produced it, and the files check what it
        # PRODUCED -- every file every statistic's manifest lists.
        version_reason = version_stale_reason(existing)
        reusable = (
            existing.get("complete")
            and existing.get("plan_id") == plan_id
            and version_reason is None
            and all(key in existing for key in _CHECKPOINT_OUTPUT_KEYS)
            and statistics_on_disk(existing.get("statistics"))
        )
        if reusable:
            # A reusing array task still records this submission.
            return keep_for_submission(existing, checkpoint_path, _locked_checkpoint_write)
        if version_reason is not None and existing.get("complete"):
            print(
                "[paralome] re-running differential"
                + (f" for {stratify_by}={stratify_value!r}" if stratify_value else "")
                + f": checkpoint {version_reason}.",
                file=sys.stderr, flush=True,
            )

    # Withdrawn first so a kill part way cannot leave it vouching
    # for a mixture of two runs.
    withdraw_checkpoint(checkpoint_path, "being recomputed", _locked_checkpoint_write)

    comparison = load_pseudobulk_comparison(
        cpm_matrix, sample_manifest,
        stratify_by=stratify_by, stratify_value=stratify_value,
    )
    cell_report: Dict[str, object] = {}
    fit_cells = make_cell_fitter(
        load_inputs=lambda: load_singlecell_inputs(extraction_dir),
        comparison=comparison, sample_manifest=sample_manifest,
        stratify_by=stratify_by, stratify_value=stratify_value,
        extraction_feature_ids=None,
        min_mean_cpm=min_mean_cpm, alpha=alpha, pseudocount=pseudocount,
        peak_chunk_size=naive_cell_t_peak_chunk_size,
        jobs=jobs, covariate_feature=covariate_feature,
        report=cell_report,
    )
    from paralome.version import __version__

    computed = compute_statistics(
        comparison, request,
        statistic_text=statistic, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature,
        min_mean_cpm=min_mean_cpm, alpha_metric=alpha_metric, alpha_value=alpha,
        output_dir=output_dir, file_prefix=file_prefix, stamp=ATAC_PEAK_MODALITY,
        features=peak_features(comparison), version=__version__,
        stage="snATAC differential",
        one_stratum_of_many=bool(stratify_by and stratify_value is not None),
        fit_cells=fit_cells, jobs=jobs,
    )

    result: Dict[str, object] = {
        "schema": SC_DIFF_SCHEMA,
        "stage": "singlecell_differential",
        "complete": True,
        "plan_id": plan_id,
        # Read by pipeline.state, so a pre-bump result is not complete.
        "method": SC_DIFF_METHOD,
        "stratify_by": stratify_by,
        "stratify_value": stratify_value,
        **computed,
        "cell_level": cell_report if request.has_cell else None,
    }
    # Every file the stage declares carries the modality stamp, so it
    # cannot collide with the RNA half of the same comparison.
    assert_results_are_stamped(
        {"statistics": stamped_files(computed["statistics"]),
         "statistic_comparison": computed.get("statistic_comparison")},
        modality=ATAC_PEAK_MODALITY, stage="singlecell_differential",
    )
    # Stamped with the array submission it was computed for.
    _locked_checkpoint_write(checkpoint_path, stamp_submission(result))
    result["checkpoint"] = str(checkpoint_path)
    return result


def _locked_checkpoint_write(path: Path, payload: Dict[str, object]) -> None:
    """The leaf checkpoint writer, shared by a fresh run, a reuse re-stamp and
    a withdrawal so all three take the same lock and stamp the same version."""
    with exclusive_checkpoint_lock(path):
        _atomic_json(path, json.loads(json.dumps(payload, default=str)))
