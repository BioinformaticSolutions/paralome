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

"""Shared cell-grouping/summing core for pseudobulk aggregation (ATAC peaks, RNA genes).

extracted from ``paralome.atac.singlecell.aggregate_pseudobulk`` so
the exact same grouping/summing logic (and the cell-counts-per-sample table
below) is shared between ATAC and RNA rather than duplicated -- this function
only depends on a plain features x cells sparse matrix, a feature-id list, a
cell metadata frame, and the cell/barcode order; it has no ATAC-specific
knowledge (peaks vs. genes is just which feature-id list is passed in).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from paralome.errors import ParalomeError
from paralome.validate import check_metadata_columns
from paralome.singlecell.sample_removal import resolve_removals


def _is_empty(value: object) -> bool:
    """'' (how R writes NA), None or NaN."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() == ""


def _constant_value(series: pd.Series) -> Optional[str]:
    unique = series.unique()
    if len(unique) == 1:
        return str(unique[0])
    return None


def aggregate_pseudobulk_core(
    matrix: spmatrix,
    feature_ids: Sequence[str],
    metadata: pd.DataFrame,
    cells: Sequence[str],
    *,
    sanitize_sample_id,
    group_by: Sequence[str],
    stratify_by: Optional[str] = None,
    min_cells_per_group: int = 1,
    remove_samples: Optional[Sequence[str]] = None,
    replicate_column: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]], Dict[str, object]]:
    """Group cells into pseudobulk samples and sum their per-feature counts.

    Returns ``(raw_counts, manifest_rows, aggregation_report)`` where
    ``raw_counts`` maps ``sample_id -> {feature_id: count}`` and
    ``manifest_rows`` is ready to hand to a modality's ``write_*_outputs``.
    ``sanitize_sample_id`` is injected (rather than imported) to avoid a
    circular import between this module and ``paralome.atac.singlecell``.

    ``remove_samples``: pseudobulk sample_id(s) to exclude outright -- these are dropped from
    ``raw_counts``/``manifest_rows`` after grouping, the same way an
    under-``min_cells_per_group`` group is dropped, and surface identically
    in ``write_cell_counts_table`` (``included=False``,
    ``excluded_reason="--remove"``) so an excluded sample stays visible
    rather than silently vanishing.
    """
    # PLM-0201, through the shared checker (validate.check_metadata_columns):
    # the full column list and a near-miss suggestion, at the earliest point the
    # answer is knowable. No --condition-column. Every
    # metadata column that is one value within a group is carried into the
    # manifest, and the differential's --factor names the columns it compares.
    from paralome.stratify import stratum_columns

    strata_columns = stratum_columns(stratify_by)          # Several, 
    check_metadata_columns(
        [("--replicate-column", column) for column in group_by if column not in strata_columns]
        + [("--stratify-by", column) for column in strata_columns],
        list(metadata.columns),
    )
    if not group_by:
        raise ParalomeError(                                   # PLM-0202
            202,
            "No sample column was named, so there is nothing to aggregate into "
            "pseudobulk samples.",
            available_columns=list(metadata.columns),
        )

    group_key_columns = list(group_by)
    grouped = metadata.groupby(group_key_columns, dropna=False, sort=True)

    reserved = set(group_key_columns) | {"barcode"}
    reserved.update(strata_columns)
    candidate_passthrough = [c for c in metadata.columns if c not in reserved]

    raw_counts: Dict[str, Dict[str, float]] = {}
    manifest_rows: List[Dict[str, object]] = []
    dropped_small_groups: List[Dict[str, object]] = []
    non_constant_columns: set[str] = set()
    used_sample_ids: set[str] = set()
    barcode_to_position = {barcode: index for index, barcode in enumerate(cells)}
    # The column that names a sample (as resolve_removals reads it). Cells
    # with no value there (R writes NA as '') are not one more sample.
    sample_column = replicate_column if replicate_column in group_key_columns else group_key_columns[0]
    sample_position = group_key_columns.index(sample_column)

    for key, group_frame in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        n_cells = len(group_frame)
        label = "__".join(sanitize_sample_id(str(value)) for value in key_values)
        if _is_empty(key_values[sample_position]):
            dropped_small_groups.append({
                **dict(zip(group_key_columns, [str(v) for v in key_values])),
                "n_cells": n_cells,
                "reason": f"empty {sample_column}",
            })
            print(f"WARNING: {n_cells} cells with an empty {sample_column!r} "
                  f"({', '.join(f'{c}={v}' for c, v in zip(group_key_columns, key_values))}) "
                  "are left out: they belong to no sample.", file=sys.stderr)
            continue
        if n_cells < min_cells_per_group:
            dropped_small_groups.append({
                **dict(zip(group_key_columns, [str(v) for v in key_values])),
                "n_cells": n_cells,
                "reason": f"fewer than --min-cells-per-group ({min_cells_per_group})",
            })
            continue

        stratum_values: Dict[str, object] = {}
        for strata_column in strata_columns:
            stratum_values[strata_column] = _constant_value(group_frame[strata_column])
            if stratum_values[strata_column] is None:
                raise ParalomeError(                           # PLM-0204
                    204,
                    f"Pseudobulk group {label!r} has more than one distinct value of "
                    f"--stratify-by {strata_column!r}: "
                    f"{', '.join(map(repr, sorted(group_frame[strata_column].unique())))}",
                    fix=f"check --replicate-column and --stratify-by {strata_column!r}",
                    group=label,
                    grouped_by=group_key_columns,
                )

        sample_id = label
        suffix = 2
        while sample_id in used_sample_ids:
            sample_id = f"{label}_{suffix}"
            suffix += 1
        used_sample_ids.add(sample_id)

        cell_positions = [barcode_to_position[barcode] for barcode in group_frame["barcode"]]
        summed = np.asarray(matrix[:, cell_positions].sum(axis=1)).ravel()
        raw_counts[sample_id] = {feature_id: float(count) for feature_id, count in zip(feature_ids, summed)}

        row: Dict[str, object] = {
            "sample_id": sample_id,
            "n_cells": n_cells,
        }
        row.update(stratum_values)
        for column, value in zip(group_key_columns, key_values):
            row.setdefault(column, str(value))
        for column in candidate_passthrough:
            value = _constant_value(group_frame[column])
            if value is None:
                non_constant_columns.add(column)
            else:
                row.setdefault(column, value)
        manifest_rows.append(row)

    if remove_samples:
        # a --remove value names a SAMPLE (a value of the sample-ID
        # column -- --replicate-column, else the first --group-by column), not
        # necessarily a composite pseudobulk sample_id. Dropping one bad donor
        # should not mean transcribing each of its per-cell-type sample_ids.
        requested, _removal_report = resolve_removals(
            remove_samples, manifest_rows,
            group_by=group_key_columns, replicate_column=replicate_column,
        )
        kept_rows = []
        for row in manifest_rows:
            if row["sample_id"] in requested:
                dropped_small_groups.append({
                    **{c: row.get(c) for c in group_key_columns},
                    "sample_id": row["sample_id"],    # The real id
                    "n_cells": row.get("n_cells"),
                    "reason": "--remove",
                })
                raw_counts.pop(row["sample_id"], None)
            else:
                kept_rows.append(row)
        manifest_rows = kept_rows

    if not manifest_rows:
        # Two codes, because they are two different mistakes with
        # two different fixes: --remove took everything (0207), or the floor
        # did (0209). A single message covering both made the reader work out
        # which, from a list of dropped groups.
        if remove_samples:
            raise ParalomeError(                               # PLM-0207
                207,
                "--remove excluded every pseudobulk sample; nothing is left to "
                f"aggregate. Removed: {', '.join(map(str, remove_samples))}",
                dropped_groups=dropped_small_groups,
            )
        raise ParalomeError(                                   # PLM-0209
            209,
            "Every pseudobulk group fell below --min-cells-per-group "
            f"({min_cells_per_group}); nothing is left to aggregate.",
            fix="lower --min-cells-per-group, or group by fewer columns",
            dropped_groups=dropped_small_groups,
        )

    for column in non_constant_columns:
        for row in manifest_rows:
            row.pop(column, None)

    report = {
        "groups": len(manifest_rows),
        "dropped_small_groups": dropped_small_groups,
        "non_constant_metadata_columns_dropped": sorted(non_constant_columns),
        "passthrough_metadata_columns": sorted(
            c for c in candidate_passthrough if c not in non_constant_columns
        ),
    }
    return raw_counts, manifest_rows, report


def write_cell_counts_table(
    manifest_rows: Sequence[Dict[str, object]],
    dropped_small_groups: Sequence[Dict[str, object]],
    group_by: Sequence[str],
    out_path,
) -> None:
    """Write a plain cells-per-sample table -- 

    Includes every group that made it into the pseudobulk aggregation
    (``included=True``) AND every group dropped for having fewer cells than
    ``--min-cells-per-group`` (``included=False``, with the drop reason) --
    so a sample that got silently excluded is just as visible here as one
    that made it in with a suspiciously low count. Sorted by ``n_cells``
    ascending, smallest (most outlier-suspect) first.
    """
    rows: List[Dict[str, object]] = []
    for row in manifest_rows:
        entry: Dict[str, object] = {"sample_id": row.get("sample_id")}
        for column in group_by:
            entry[column] = row.get(column)
        entry["n_cells"] = row.get("n_cells")
        entry["included"] = True
        entry["excluded_reason"] = ""
        rows.append(entry)
    for row in dropped_small_groups:
        entry = {"sample_id": row.get("sample_id") or "__".join(str(row.get(c, "")) for c in group_by)}
        for column in group_by:
            entry[column] = row.get(column)
        entry["n_cells"] = row.get("n_cells")
        entry["included"] = False
        entry["excluded_reason"] = row.get("reason", "")
        rows.append(entry)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("n_cells", ascending=True, kind="stable")
    frame.to_csv(out_path, index=False)


def format_pseudobulk_summary(result: Mapping[str, Any], feature_noun: str) -> List[str]:
    """Plain-text summary of a pseudobulk run, for interactive use.

    The full payload (dropped small groups, dropped metadata columns, digests,
    every output path, the R helper's stdout) runs to hundreds of lines. Nothing
    is lost: all of it is written to the stage checkpoint, which the summary
    points at, and ``--json`` still prints the payload verbatim for anything
    parsing it. This only changes what a human sees by default, and it matches
    what ``differential`` and ``global-pca`` do.

    Shared by both modalities so ATAC and RNA cannot drift in what they report;
    ``feature_noun`` is the modality's feature name ("genes" / "peaks").
    """
    extraction = dict(result.get("extraction") or {})
    dropped = list(result.get("aggregation", {}).get("dropped_small_groups") or [])
    group_by = result.get("group_by") or []
    if isinstance(group_by, (list, tuple)):
        group_by = " x ".join(str(g) for g in group_by)

    n_features = result.get("genes") if feature_noun == "genes" else result.get("peaks")
    lines = ["Pseudobulk aggregation complete" if result.get("complete")
             else "Pseudobulk aggregation incomplete"]

    if extraction:
        cells = extraction.get("n_cells")
        feats = extraction.get("n_genes") if feature_noun == "genes" else extraction.get("n_peaks")
        reused = " (reused)" if extraction.get("complete") and not extraction.get("executed", True) else ""
        if cells is not None and feats is not None:
            lines.append(f"  Extraction : {feats:,} {feature_noun} x {cells:,} cells{reused}")

    lines.append(f"  Groups     : {result.get('groups', result.get('samples'))} from {group_by}")
    if n_features is not None:
        lines.append(f"  {feature_noun.capitalize():<11}: {n_features:,}")
    if dropped:
        # Each reason counted on its own (--remove is not "below the floor").
        counts: Dict[str, int] = {}
        for item in dropped:
            reason = str(item.get("reason", ""))
            if "(" in reason and ")" in reason:
                reason = f"below {reason.split('(')[-1].rstrip(')')} cells"
            counts[reason] = counts.get(reason, 0) + 1
        detail = ", ".join(f"{n} {reason}" for reason, n in counts.items())
        lines.append(f"  Dropped    : {len(dropped)} groups ({detail}) -- see the checkpoint for the list")
    for key, label in (("raw_counts", "Raw counts"), ("cpm", "CPM"), ("log2_cpm", "log2 CPM"),
                       ("sample_manifest", "Manifest"), ("qc_run_dir", "QC"),
                       ("checkpoint", "Checkpoint")):
        value = result.get(key)
        if value:
            lines.append(f"  {label:<11}: {value}")
    return lines
