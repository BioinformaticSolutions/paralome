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

"""The modality-free half of the single-cell differential stage.

The single-cell differential has no statistics of its own. Both drivers
(snATAC peaks, scRNA genes) do this:

  1. ``load_pseudobulk_comparison`` reads the pseudobulk CPM matrix and the
     sample manifest: every sample in scope (after any stratum narrowing) is a
     design row, and the --factor columns carry its levels.
  2. ``plan_comparison`` asks ``paralome.comparisons.plan_differential`` which
     comparisons each requested statistic makes -- the same plan bulk RNA and
     bulk ATAC get.
  3. Pseudobulk (the default): ``fit_pseudobulk`` hands the design's log2 CPM
     to ``paralome.engine.fit_plan``, which calls ``paralome.stats``.
     ``:cell``: ``paralome.atac.singlecell_stats.fit_cell_plan`` answers the
     same plan with the cells as the observations, returning the engine's
     ``ComparisonResult`` objects.
  4. ``write_statistics`` writes one folder per statistic through
     ``paralome.results`` -- the writer every pipeline uses.


The cell-level fits live in ``paralome.atac.singlecell_stats`` because they are
typed on the extraction loaders defined beside it; they are reached from here
lazily (PEP 562 ``__getattr__``) so the two modules do not import each other at
load time.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from paralome.errors import ParalomeError
from paralome.qc_stats import pearson
from paralome.stats import (
    DATA_LEVEL_CELL,
    DATA_LEVEL_PSEUDOBULK,
    DATA_LEVELS,
    StatisticRequest,
    parse_statistic_request,
    statistic_folder,
)

#: How many features' per-cell counts the vectorised cell-level statistics
#: (t_cell, paired_t_cell, lm_cell) densify at once. Bounds memory; changes no
#: number. (The option that sets it is spelled --naive-cell-t-peak-chunk-size.)
DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE = 2000

#: The pseudobulk values are log2(CPM + 1).
PSEUDOBULK_PSEUDOCOUNT = 1.0

#: The method stamp of the single-cell differential stage -- the date it last
#: changed what it computes -- in both drivers' plan ids and checkpoints
#: (pipeline.state reads it), so a result recorded under another method is
#: recomputed rather than returned. One folder per statistic through the shared plan,
#: engine and writer; the cell-level statistics rebuilt on the same plan.
#: Anova_cell is least squares on the cells, replicates
#: pooled; its random-intercept model and --peak-subset-limit are retired.
SC_DIFF_METHOD = "2026-11-06"

def _read_pseudobulk_manifest(path: Path) -> List[Dict[str, str]]:
    # A byte-order mark at the top of the file is ignored.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Sample manifest contains no samples: {path}")
    if "sample_id" not in rows[0]:
        raise ValueError(f"Sample manifest is missing its sample_id column: {path}")
    if "n_cells" not in rows[0]:
        raise ValueError(
            f"Sample manifest is missing n_cells (the cell-level statistics check each "
            f"sample's cells against it): {path}"
        )
    return rows


def _sniff_cpm_delimiter(path: Path) -> str:
    """CPM matrices are comma-delimited, but a pseudobulk output from an older
    wheel can still be tab-delimited under a .tsv name."""
    with open(path, "r", newline="") as handle:
        first_line = handle.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


@dataclass(frozen=True)
class _CpmTable:
    header: List[str]
    annotation_columns: List[str]
    annotation: "Any"                  # DataFrame of the annotation columns, as text
    column: Dict[str, int]             # sample column -> its column in values
    values: np.ndarray                 # features x sample columns


_CPM_CACHE: Dict[Tuple[str, str, frozenset], _CpmTable] = {}
_CPM_LOCK = threading.Lock()


def _cpm_table(cpm_path: Path, sample_ids: Any) -> _CpmTable:
    """The CPM matrix, parsed once per file version and kept for the process (the last one only).

    pandas' C reader with round-trip float parsing, so every value is exactly float() of its text; annotation
    columns stay text."""
    import pandas as pd

    key = (str(cpm_path), input_file_identity(cpm_path), frozenset(map(str, sample_ids)))
    with _CPM_LOCK:
        cached = _CPM_CACHE.get(key)
        if cached is not None:
            return cached
        delimiter = _sniff_cpm_delimiter(cpm_path)
        with cpm_path.open(newline="") as handle:
            header = next(csv.reader(handle, delimiter=delimiter), [])
        samples = set(key[2])
        annotation_columns = [c for c in header if c not in samples and c != "sample_id"]
        value_columns = [c for c in header if c in samples]
        frame = pd.read_csv(cpm_path, sep=delimiter, dtype={c: str for c in annotation_columns},
                            keep_default_na=False, float_precision="round_trip", engine="c")
        values = frame[value_columns].to_numpy(dtype=float) if value_columns else np.zeros((len(frame), 0))
        table = _CpmTable(header=list(header), annotation_columns=annotation_columns,
                          annotation=frame[annotation_columns].astype(str),
                          column={c: i for i, c in enumerate(value_columns)}, values=values)
        _CPM_CACHE.clear()
        _CPM_CACHE[key] = table
        return table


def input_file_identity(path: Union[str, Path]) -> str:
    """"<bytes>:<mtime ns>" of an input, or "missing".

    Pseudobulk always rewrites the same paths, so the plan ids hash what the
    files hold as well as where they are.
    """
    try:
        stat = Path(path).expanduser().stat()
    except OSError:
        return "missing"
    return f"{stat.st_size}:{stat.st_mtime_ns}"


# ---------------------------------------------------------------------------
# The pseudobulk comparison: every sample in scope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudobulkComparison:
    """The pseudobulk values of every sample in scope.

    The samples are the DESIGN: every manifest sample in scope
    (after --stratify-by narrowing) that has a column in the CPM matrix.
    Which groups are compared is the --factor columns'
    business, read from each sample's manifest row by the shared plan -- there
    are no arms to resolve here.
    """

    feature_ids: List[str]
    annotation: List[Dict[str, object]]
    sample_ids: List[str]
    sample_rows: List[Dict[str, str]]
    n_cells: np.ndarray            # per sample
    cpm: np.ndarray                # features x samples
    log2_cpm: np.ndarray           # features x samples, log2(CPM + 1)


def load_pseudobulk_comparison(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    *,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    feature_id_column: str = "peak_id",
) -> PseudobulkComparison:
    """Read one comparison's pseudobulk values: every sample in scope."""
    manifest_path = Path(sample_manifest).expanduser().resolve()
    cpm_path = Path(cpm_matrix).expanduser().resolve()
    all_rows = _read_pseudobulk_manifest(manifest_path)
    # Every sample_id the manifest lists anywhere is a sample column of the CPM
    # matrix -- computed from the FULL manifest so another stratum's sample
    # column is never mistaken for an annotation column.
    all_sample_ids = {row["sample_id"] for row in all_rows}

    rows = all_rows
    if stratify_by is not None:
        from paralome.stratify import in_stratum, missing_columns

        absent = missing_columns(stratify_by, list(rows[0]))
        if absent:
            raise ValueError(f"Sample manifest is missing --stratify-by column(s) {', '.join(map(repr, absent))}")
        if stratify_value is not None:
            # A sample is in every stratum its columns list.
            rows = [row for row in rows if in_stratum(row, stratify_by, stratify_value)]
    if not rows:
        raise ValueError(f"No sample of the manifest is in stratum {stratify_value!r}")

    # The matrix is parsed once per process and shared (_cpm_table); every stratum of a fan-out takes its own
    # columns from it, rather than each parsing the whole file again.
    table = _cpm_table(cpm_path, all_sample_ids)
    header = table.header
    annotation_columns = table.annotation_columns
    sample_columns = set(table.column)
    if feature_id_column not in header:
        raise ValueError(
            f"CPM matrix {cpm_path} has no {feature_id_column!r} column; it has "
            f"{', '.join(annotation_columns[:8])}"
        )
    # Manifest order, restricted to samples the matrix has a column for
    # (--min-cells-per-group can drop a sample x stratum cell).
    rows = [dict(row) for row in rows if str(row["sample_id"]) in sample_columns]
    if not rows:
        raise ValueError(f"CPM matrix {cpm_path} has no column for any sample in scope")
    sample_ids = [str(row["sample_id"]) for row in rows]
    feature_ids: List[str] = list(table.annotation[feature_id_column])
    annotation: List[Dict[str, object]] = table.annotation[annotation_columns].to_dict("records")
    if not feature_ids:
        raise ValueError(f"CPM matrix contains no features: {cpm_path}")

    cpm = table.values[:, [table.column[s] for s in sample_ids]].astype(float, copy=True)
    return PseudobulkComparison(
        feature_ids=feature_ids, annotation=annotation,
        sample_ids=sample_ids, sample_rows=rows,
        n_cells=np.asarray([float(row["n_cells"]) for row in rows], dtype=float),
        cpm=cpm, log2_cpm=np.log2(cpm + PSEUDOBULK_PSEUDOCOUNT),
    )


def kept_comparison_samples(
    sample_manifest: Union[str, Path], comparison: PseudobulkComparison,
) -> List[Dict[str, object]]:
    """Every design sample, with the manifest values that identify its cells.

    Every sample in scope, because the cell-level ANOVA and lm read all of
    them. Each entry is ``{"sample_id", "position", "n_cells", "values"}``:
    ``position`` is the sample's row in the plan, ``values`` every manifest
    column except sample_id and n_cells. A pseudobulk sample IS the cells
    sharing its grouping values, and every such column is in the manifest, so
    these values identify the sample's cells exactly;
    ``build_cell_level_comparison`` matches them and checks the count against
    n_cells.
    """
    rows = {row["sample_id"]: row for row in _read_pseudobulk_manifest(Path(sample_manifest).expanduser().resolve())}
    kept: List[Dict[str, object]] = []
    for position, sample_id in enumerate(comparison.sample_ids):
        row = rows.get(str(sample_id))
        if row is None:
            raise ParalomeError(
                210,
                f"Pseudobulk sample {sample_id!r} is in this comparison but not in the sample "
                f"manifest {sample_manifest}, so its cells cannot be found for the cell-level statistics.",
                fix="point --samples at the sample_manifest.csv written beside this CPM matrix",
            )
        try:
            n_cells = int(float(str(row.get("n_cells", "")).strip()))
        except ValueError:
            raise ParalomeError(
                210,
                f"Pseudobulk sample {sample_id!r} has no usable n_cells in {sample_manifest} "
                f"({row.get('n_cells')!r}); the cell-level statistics check each sample's cells against it.",
                fix="point --samples at the sample_manifest.csv written beside this CPM matrix",
            ) from None
        kept.append({
            "sample_id": str(sample_id),
            "position": position,
            "n_cells": n_cells,
            "values": {
                str(column): ("" if value is None else str(value))
                for column, value in row.items()
                if column not in ("sample_id", "n_cells") and column is not None
            },
        })
    return kept


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def parse_request(
    statistic: str,
    *,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> StatisticRequest:
    """--statistic, read by the shared parser with the single-cell data levels."""
    if covariate and covariate_feature:
        raise ValueError(
            "--statistic lm needs exactly one of --covariate (a numeric sample-manifest "
            "column) or --covariate-feature (another feature to regress against); got both"
        )
    return parse_statistic_request(
        statistic, data_levels=DATA_LEVELS, default_data_level=DATA_LEVEL_PSEUDOBULK,
        factors=list(factors), covariate=covariate or covariate_feature, pair_by=pair_by,
    )


def covariate_feature_row(
    feature_ids: Sequence[str], annotation: Sequence[Mapping[str, object]], wanted: str,
) -> Optional[int]:
    """The row of --covariate-feature: by feature id, else by any annotation value
    (so a gene SYMBOL works where the id column holds Ensembl ids)."""
    wanted = str(wanted)
    for index, name in enumerate(feature_ids):
        if str(name) == wanted:
            return index
    for index, values in enumerate(annotation):
        if any(str(value) == wanted for value in values.values()):
            return index
    return None


def refuse_a_factor_aggregation_dropped(
    sample_manifest: Union[str, Path], extraction_dir: Union[str, Path], factors: Sequence[str],
) -> None:
    """PLM-0203 for a --factor column that varies within a pseudobulk sample.

    Aggregation carries into the manifest only the cell-metadata columns that
    are one value within each pseudobulk sample, so such a column is not in the
    manifest at all, and the plan could only say there is no column of that
    name. When the cells have it, say what is wrong and how to fix it.
    """
    from paralome.comparisons import FactorNotConstantError
    from paralome.stats import factor_columns

    columns = factor_columns(factors)
    if not columns:
        return
    try:
        with Path(sample_manifest).open(newline="") as handle:
            manifest = next(csv.reader(line for line in handle if not line.startswith("#")), [])
        with (Path(extraction_dir) / "cell_metadata.tsv").open(newline="") as handle:
            cells = next(csv.reader(handle, delimiter="\t"), [])
    except (OSError, StopIteration):
        return
    for column in columns:
        if column not in manifest and column in cells:
            raise FactorNotConstantError(
                f"--factor {column}: the cells have a {column!r} column, but it is not one value "
                f"within each --replicate-column sample, so aggregation left it out of {sample_manifest}. "
                f"Check that --replicate-column names the sample column, or fix the cell metadata"
            )


def _group_of(sample_ids: Sequence[str]) -> Dict[int, str]:
    """Each design row is one pseudobulk group (its sample)."""
    return {index: str(sample_id) for index, sample_id in enumerate(sample_ids)}


def plan_statistics(
    rows: Sequence[Mapping[str, str]],
    request: StatisticRequest,
    *,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    covariate_feature_values: Optional[np.ndarray] = None,
    block_by: Optional[str] = None,
):
    """``comparisons.plan_differential`` over these rows: the one place it is called here."""
    from paralome.comparisons import plan_differential

    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    return plan_differential(
        rows, request=request,
        factors=list(factors), pair_by=pair_by, covariate=covariate,
        covariate_feature=covariate_feature,
        covariate_feature_values=covariate_feature_values,
        group_of=_group_of(sample_ids), block_by=block_by,
    )


def plan_each_statistic(
    rows: Sequence[Mapping[str, str]],
    request: StatisticRequest,
    *,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    covariate_feature_values: Optional[np.ndarray] = None,
    block_by: Optional[str] = None,
) -> Tuple[List[Tuple[str, Optional[str]]], Dict[str, str]]:
    """``([(statistic, data level) that can be planned on these rows], {folder: why not})``.

    Used where a stratum may hold fewer samples than a statistic needs: the
    fan-out's planner (from the manifest alone) and a stratum's own run. Each
    statistic is planned on its own by ``plan_differential``, so a stratum that
    can run t but has no complete design for the ANOVA runs t and records why
    anova did not run.
    """
    from paralome.stats import StatisticRequest as _Request

    runnable: List[Tuple[str, Optional[str]]] = []
    refused: Dict[str, str] = {}
    for statistic, level in request.pairs:       # Each at its own data level
        try:
            plan_statistics(
                rows, _Request((statistic,), level),
                factors=factors,
                pair_by=pair_by if statistic == "paired-t" else None,
                covariate=covariate if statistic == "lm" else None,
                covariate_feature=covariate_feature if statistic == "lm" else None,
                covariate_feature_values=covariate_feature_values if statistic == "lm" else None,
                block_by=block_by if statistic in ("anova", "lm") else None,
            )
        except ValueError as error:      # DesignError (PLM-0221): the reason, without the number
            from paralome.stats import TwoControlsError

            if isinstance(error, TwoControlsError):     # Refused, not a stratum skipped
                raise
            refused[statistic_folder(statistic, level)] = getattr(error, "detail", "") or str(error)
            continue
        runnable.append((statistic, level))
    return runnable, refused


def merge_plans(plans, data_level):
    """One DifferentialPlan from the plans of each data level over the same samples."""
    if len(plans) == 1:
        return plans[0]
    from dataclasses import replace

    first = plans[0]
    return replace(first, statistics=[s for plan in plans for s in plan.statistics], data_level=data_level)


def plan_comparison(
    comparison: PseudobulkComparison,
    request: StatisticRequest,
    *,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    block_by: Optional[str] = None,
):
    """The plan for one comparison: every requested statistic, every comparison it makes."""
    values = None
    if covariate_feature:
        row = covariate_feature_row(comparison.feature_ids, comparison.annotation, covariate_feature)
        values = None if row is None else comparison.log2_cpm[row]
    return plan_statistics(
        comparison.sample_rows, request,
        factors=factors, pair_by=pair_by, covariate=covariate,
        covariate_feature=covariate_feature, covariate_feature_values=values, block_by=block_by,
    )


def fit_pseudobulk(
    comparison: PseudobulkComparison, plan, *, min_mean_cpm: float, alpha: float,
):
    """The plan fitted on the design's log2 CPM by the shared engine (uncorrected)."""
    from paralome.engine import fit_plan

    return fit_plan(
        comparison.log2_cpm, plan, raw=comparison.cpm, min_mean=min_mean_cpm,
        alpha=alpha, replication_unit="samples",
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def peak_features(comparison: PseudobulkComparison, feature_id_column: str = "peak_id"):
    """snATAC peaks as the writer sees them: gene by the TSS-else-nearest rule."""
    from paralome.results import Features, peak_gene

    genes = [peak_gene(annotation) for annotation in comparison.annotation]
    return Features(
        key=list(comparison.feature_ids), key_column=feature_id_column,
        gene_id=[gene for gene, _name in genes], gene_name=[name for _gene, name in genes],
        annotation=[dict(annotation) for annotation in comparison.annotation],
        peaks=True,
    )


def gene_features(
    comparison: PseudobulkComparison,
    feature_id_column: str = "feature_name",
    gene_id_column: str = "gene_id",
    gene_name_column: str = "common_id",
):
    """scRNA genes as the writer sees them: the key is the feature name; gene_id is
    '.' when the annotation could not place the gene (the writer leaves those out
    of the Evidome table)."""
    from paralome.report_schema import MISSING_VALUE
    from paralome.results import Features

    def text(annotation: Mapping[str, object], column: str) -> str:
        value = annotation.get(column)
        value = "" if value is None else str(value).strip()
        return MISSING_VALUE if value in ("", "nan", "NA", "None") else value

    return Features(
        key=list(comparison.feature_ids), key_column=feature_id_column,
        gene_id=[text(a, gene_id_column) for a in comparison.annotation],
        gene_name=[text(a, gene_name_column) for a in comparison.annotation],
        annotation=[
            {k: v for k, v in a.items() if k not in (gene_id_column,)}
            for a in comparison.annotation
        ],
        peaks=False,
    )


#: The single-cell result names an older layout wrote (one flat set per model in --out). results.py removes
#: what ITS earlier runs wrote; these never went through it, so they are swept by exact name.
_RETIRED_SLUGS = (
    "pseudobulk_t", "pseudobulk_paired_t", "pseudobulk_lm", "pseudobulk_anova",
    "mixed_model", "naive_cell_t",
)
_RETIRED_SINGLE_NAMES = (
    "model_comparison.json", "power_summary.csv",
    "power_coverage.svg", "power_coverage.pdf", "power_coverage.csv",
    "power_by_n.svg", "power_by_n.pdf", "power_by_n.csv",
)


def sweep_retired_outputs(output_dir: Union[str, Path], file_prefix: str, stamp: str, *, stage: str) -> List[str]:
    """Remove this prefix's older per-model files from the flat --out.

    Exact names only (``<prefix>_<model>_<stamp>.csv``, its p/q lists, its
    peak_statistics table, model_comparison.json and the flat power files), so
    another stratum's, modality's or command's files cannot match. Said on
    stderr per file.
    """
    from paralome.atac.naming import result_path
    from paralome.report_schema import remove_unwritten_results

    candidates = []
    for slug in _RETIRED_SLUGS:
        for name in (f"{slug}.csv", f"{slug}_p0.05.csv", f"{slug}_q0.05.csv",
                     f"{slug}_peak_statistics.csv"):
            candidates.append(result_path(output_dir, name, file_prefix=file_prefix, modality=stamp))
    for name in _RETIRED_SINGLE_NAMES:
        candidates.append(result_path(output_dir, name, file_prefix=file_prefix, modality=stamp))
    return remove_unwritten_results(
        candidates, [], stage=f"{stage} (results named by the retired per-model layout)",
    )


def sweep_unrequested_statistics(
    output_dir: Union[str, Path], file_prefix: str, stamp: str, written: Sequence[str], *, stage: str,
) -> List[str]:
    """Remove what an earlier run of this prefix wrote into a statistic folder
    this run did not write (a statistic no longer requested, or the other data
    level). paralome.results cleans the folders it writes; a folder it does not
    write is never opened, so its last run's files would stay beside the new
    results looking current. Only the files that folder's manifest for THIS
    prefix lists are removed, then the manifest.
    """
    from paralome.results import _stamped
    from paralome.stats import ALL_STATISTICS

    removed: List[str] = []
    for statistic in ALL_STATISTICS:
        for level in (None, DATA_LEVEL_CELL):
            folder = statistic_folder(statistic, level)
            if folder in written:
                continue
            manifest = _stamped(Path(output_dir) / folder / "machine", file_prefix,
                                f"{folder}_manifest", stamp, ".json")
            if not manifest.is_file():
                continue
            try:
                listed = [str(p) for p in (json.loads(manifest.read_text()).get("files") or [])]
            except (OSError, ValueError):
                listed = []
            from paralome.report_schema import remove_unwritten_results

            removed.extend(remove_unwritten_results(
                [*listed, str(manifest)], [], stage=f"{stage} {folder} (not requested by this run)",
            ))
    return removed


def statistic_comparison(
    results: Sequence[object], *, alpha_metric: str, alpha_value: float,
) -> List[Dict[str, object]]:
    """Per comparison name present in more than one statistic's folder: do they agree?

    For each pair of folders holding a comparison of the same name (t and
    paired_t both make KO_vs_WT): how many features both tested, the Pearson r
    of their log2 fold changes and the Spearman r of their p values. No call of
    significance. Never a combined table -- each statistic's numbers stay in its folder.
    """
    by_name: Dict[str, Dict[str, object]] = {}
    for result in results:
        by_name.setdefault(result.name, {})[result.folder] = result
    entries: List[Dict[str, object]] = []
    for name, folders in by_name.items():
        if len(folders) < 2:
            continue
        for a_folder, b_folder in combinations(list(folders), 2):
            a, b = folders[a_folder], folders[b_folder]
            both = (np.asarray(a.columns["tested"], dtype=bool)
                    & np.asarray(b.columns["tested"], dtype=bool))
            entry: Dict[str, object] = {
                "comparison": name, "statistic_a": a_folder, "statistic_b": b_folder,
                "features_tested_by_both": int(both.sum()),
            }
            if both.sum() >= 2:
                fc_a = np.asarray(a.columns["log2_fold_change"], dtype=float)[both]
                fc_b = np.asarray(b.columns["log2_fold_change"], dtype=float)[both]
                valid = np.isfinite(fc_a) & np.isfinite(fc_b)
                entry["log2fc_pearson_r"] = pearson(fc_a[valid], fc_b[valid], min_n=2, undefined=None)
                # Agreement on the p values themselves, not on a call: their rank correlation.
                from paralome.qc_stats import spearman

                p_a = np.asarray(a.columns["p_value"], dtype=float)[both]
                p_b = np.asarray(b.columns["p_value"], dtype=float)[both]
                ok = np.isfinite(p_a) & np.isfinite(p_b)
                entry["p_value_spearman_r"] = spearman(p_a[ok], p_b[ok], min_n=2, undefined=None)
            else:
                entry["log2fc_pearson_r"] = None
                entry["p_value_spearman_r"] = None
            entries.append(entry)
    return entries


def statistics_files(statistics: Mapping[str, object]) -> List[str]:
    """Every file the statistics manifests of a checkpoint say were written."""
    files: List[str] = []
    for manifest in (statistics or {}).values():
        if isinstance(manifest, Mapping):
            files.extend(str(path) for path in manifest.get("files") or [] if path)
    return files


def statistics_on_disk(statistics: object) -> bool:
    """A checkpoint's results are reusable only when every listed file exists."""
    if not isinstance(statistics, Mapping) or not statistics:
        return False
    files = statistics_files(statistics)
    return bool(files) and all(Path(path).is_file() for path in files)


def stamped_files(statistics: Mapping[str, object]) -> List[str]:
    """The written files the modality-stamp check applies to: all of them.

    Every file paralome.results writes ends its stem with the stamp.
    """
    return statistics_files(statistics)


def single_sample_notes(plan) -> Dict[str, Dict[str, str]]:
    """{folder: {comparison: note}} for each t_cell (or one-way, or one-sample-per-combination anova_cell) comparison with a one-sample group.

    T:cell's replicate is the cell, so it runs at
    one sample per group -- recorded rather than blocked, because that sample
    is then perfectly confounded with its group.
    """
    notes: Dict[str, Dict[str, str]] = {}
    for statistic_plan in plan.statistics:
        if (statistic_plan.statistic == "anova" and statistic_plan.one_way
                and statistic_plan.data_level == DATA_LEVEL_CELL):
            # The one-way anova_cell has the same footing (cells as N, always), so it carries the same record.
            thin = [f"{statistic_plan.factor} {level}"
                    for level, members in zip(statistic_plan.levels, statistic_plan.level_members)
                    if len(members) < 2]
            if thin:
                note = (
                    f"ran with a single sample in {', '.join(thin)}. Its replicate is the cell, so "
                    "it computes normally, but each such sample is perfectly confounded with its "
                    "level: any difference between the samples (batch, depth, health) is reported "
                    f"as a {statistic_plan.factor} effect. Treat the p-values as describing THESE "
                    "CELLS, not the levels."
                )
                for comparison in statistic_plan.comparisons:
                    notes.setdefault(statistic_plan.folder, {})[comparison.name] = note
            continue
        if statistic_plan.statistic == "anova" and statistic_plan.cells_as_n:
            # The factorial anova_cell with one sample in a
            # combination of levels, on the same footing.
            combinations = {}
            for row in plan.rows:
                key = tuple(str(row.get(f) or "").strip() for f in plan.factors)
                combinations[key] = combinations.get(key, 0) + 1
            if all(count > 1 for count in combinations.values()):
                continue
            single = [" / ".join(key) for key, count in combinations.items() if count == 1]
            note = (
                f"ran with a single sample in {len(single)} of {len(combinations)} combinations of "
                f"{', '.join(plan.factors)} ({', '.join(single[:4])}{' ...' if len(single) > 4 else ''}). "
                "Its replicate is the cell, so it computes normally, but such a sample is perfectly "
                "confounded with its combination: any difference between the samples (batch, "
                "depth, health) is reported as an effect of the factors. Treat the p-values as "
                "describing THESE CELLS, not the groups."
            )
            for comparison in statistic_plan.comparisons:
                notes.setdefault(statistic_plan.folder, {})[comparison.name] = note
            continue
        if statistic_plan.statistic != "t" or statistic_plan.data_level != DATA_LEVEL_CELL:
            continue
        for comparison in statistic_plan.comparisons:
            thin = [
                f"{label} ({len(index)} sample)"
                for label, index in ((comparison.control, comparison.control_index),
                                     (comparison.treatment, comparison.treatment_index))
                if len(index) < 2
            ]
            if thin:
                notes.setdefault(statistic_plan.folder, {})[comparison.name] = (
                    f"ran with a single sample in {' and '.join(thin)}. Its replicate is the "
                    "cell, so it computes normally, but that sample is perfectly confounded "
                    "with its group: any difference between the samples (batch, depth, "
                    "health) is reported as a group effect. Treat the p-values as "
                    "describing THESE CELLS, not the groups."
                )
    return notes


def compute_statistics(
    comparison: PseudobulkComparison,
    request: StatisticRequest,
    *,
    statistic_text: str,
    factors: Sequence[str],
    pair_by: Optional[str],
    covariate: Optional[str],
    covariate_feature: Optional[str],
    min_mean_cpm: float,
    alpha_metric: str,
    alpha_value: float,
    output_dir: Path,
    file_prefix: str,
    stamp: str,
    features,
    version: str,
    stage: str,
    one_stratum_of_many: bool = False,
    fit_cells: Optional[Callable[[object], Sequence[object]]] = None,
    fit_values: Optional[Callable[[object], Sequence[object]]] = None,
    sample_unit: str = "CPM",
    jobs: int = 1,
    block_by: Optional[str] = None,
) -> Dict[str, object]:
    """Plan, fit, correct and write one comparison. The body both drivers share.

    ``block_by``: --block-by, the untested term of anova's and lm's models.

    ``jobs``: the processes that write the result folders' per-comparison files
, the same budget the :cell fit uses.

    ``fit_values(plan)``, when given, fits the plan on the sample-level values in
    place of ``fit_pseudobulk`` and returns the engine's uncorrected results
    (the mass-spec arm, whose features may be missing in some
    samples, routes each set of measured samples through the same engine).

    ``one_stratum_of_many``: this is one stratum of a --stratify-by run. A
    statistic its samples cannot support (a stratum with no complete design, a
    level with one sample) is then left out and recorded under
    ``statistics_not_run``, as the fan-out's planner already said it would be;
    the other strata still run it. Outside a fan-out a statistic that cannot
    be planned is refused, by plan_differential's own message.

    ``fit_cells(plan)`` fits the plan with the cells as observations (the
    ``:cell`` data level); it returns the engine's uncorrected results.
    """
    from paralome.engine import finalize
    from paralome.results import Samples, write_all_statistics

    not_run: Dict[str, str] = {}
    if one_stratum_of_many:
        values = None
        if covariate_feature:
            row = covariate_feature_row(comparison.feature_ids, comparison.annotation, covariate_feature)
            values = None if row is None else comparison.log2_cpm[row]
        runnable, not_run = plan_each_statistic(
            comparison.sample_rows, request,
            factors=factors, pair_by=pair_by, covariate=covariate,
            covariate_feature=covariate_feature, covariate_feature_values=values, block_by=block_by,
        )
        if not runnable:
            raise ValueError(
                "No requested statistic can be run on this stratum: "
                + "; ".join(f"{folder}: {why}" for folder, why in sorted(not_run.items()))
            )
        for folder, why in sorted(not_run.items()):
            print(f"[paralome] {stage}: {folder} not run here: {why}", file=sys.stderr, flush=True)
        request = StatisticRequest.of_pairs(runnable)

    # Each data level is planned and fitted on its own (a request may
    # hold both, e.g. anova,anova:cell), then the results are written together,
    # each statistic in its own folder.
    by_level = request.by_level()
    level_plans = [plan_comparison(
        comparison, part, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature, block_by=block_by,
    ) for part in by_level]
    plan = merge_plans(level_plans, request.data_level)
    notes = single_sample_notes(plan)
    # Said once per statistic folder and note, naming how many comparisons it covers, not once per
    # comparison (a 4-factor anova_cell has dozens a stratum). Each comparison's own record of it is unchanged.
    for folder, by_comparison in sorted(notes.items()):
        grouped: Dict[str, List[str]] = {}
        for name, note in sorted(by_comparison.items()):
            grouped.setdefault(note, []).append(name)
        for note, names in grouped.items():
            which = names[0] if len(names) == 1 else f"{len(names)} comparisons ({names[0]} ...)"
            sys.stderr.write(f"[paralome] {stage}: {folder} {which}: {note}\n")
            sys.stderr.flush()
    started = time.monotonic()
    fitted = []
    for part, part_plan in zip(by_level, level_plans):
        if part.data_level == DATA_LEVEL_CELL:
            if fit_cells is None:  # pragma: no cover - both drivers pass it
                raise ValueError("this stage was asked for :cell statistics and has no cell-level data")
            fitted.extend(fit_cells(part_plan))
        elif fit_values is not None:
            fitted.extend(fit_values(part_plan))
        else:
            fitted.extend(fit_pseudobulk(comparison, part_plan, min_mean_cpm=min_mean_cpm, alpha=alpha_value))
    results = finalize(fitted, alpha_metric=alpha_metric, alpha_value=alpha_value)
    runtime = time.monotonic() - started

    sweep_retired_outputs(output_dir, file_prefix, stamp, stage=stage)
    sweep_unrequested_statistics(
        output_dir, file_prefix, stamp, [plan_statistic.folder for plan_statistic in plan.statistics],
        stage=stage,
    )
    statistics = write_all_statistics(
        results, root=output_dir, file_prefix=file_prefix, stamp=stamp,
        features=features,
        samples=Samples(ids=list(comparison.sample_ids), values=comparison.cpm, unit=sample_unit),
        version=version, alpha_metric=alpha_metric, alpha_value=alpha_value, stage=stage, jobs=jobs,
    )

    from paralome.atac.naming import result_path

    comparison_path = result_path(
        output_dir, "statistic_comparison.json", file_prefix=file_prefix, modality=stamp,
    )
    entries = statistic_comparison(results, alpha_metric=alpha_metric, alpha_value=alpha_value)
    if entries:
        from paralome.pipeline.jsonio import write_json

        write_json(comparison_path, {
            "statistic": statistic_text,
            "data_level": request.level_records(),
            "note": (
                "For each comparison made by more than one statistic: the Spearman r of their "
                "p values and the Pearson r of their log2 fold changes. Each "
                "statistic's results are in its own folder."
            ),
            "comparisons": entries,
        }, atomic=True, default=str)
    else:
        from paralome.report_schema import remove_unwritten_results

        remove_unwritten_results([comparison_path], [], stage=stage)

    return {
        "statistic": statistic_text,
        "data_level": request.level_records(),
        "plan": plan.as_record(),
        "statistics": statistics,
        "statistics_not_run": not_run,
        "warnings": notes,
        "statistic_comparison": str(comparison_path) if entries else None,
        "runtime_seconds": runtime,
    }


# --------------------------------------------------------------------------
# Resident in paralome.atac.singlecell_stats -- see the module docstring.
# --------------------------------------------------------------------------

_ATAC_RESIDENT = (
    "CellLevelComparison",
    "build_cell_level_comparison",
    "fit_cell_plan",
)


def __getattr__(name: str):
    if name in _ATAC_RESIDENT:
        from paralome.atac import singlecell_stats as _atac_engine

        return getattr(_atac_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ATAC_RESIDENT))


__all__ = [
    "DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE",
    "PseudobulkComparison",
    "SC_DIFF_METHOD",
    "compute_statistics",
    "covariate_feature_row",
    "fit_pseudobulk",
    "gene_features",
    "input_file_identity",
    "kept_comparison_samples",
    "load_pseudobulk_comparison",
    "parse_request",
    "peak_features",
    "plan_comparison",
    "plan_each_statistic",
    "plan_statistics",
    "statistic_comparison",
    "statistics_files",
    "statistics_on_disk",
    "sweep_retired_outputs",
    "sweep_unrequested_statistics",
    *_ATAC_RESIDENT,
]
