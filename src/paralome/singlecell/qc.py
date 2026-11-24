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

"""Project-level QC tables written to ``<out>/qc/``.

The workflow this serves: when a new project starts, its data are QC'd
first. Extracting an RDS object writes a QC folder under the output folder
holding a summary table of all the available metadata, from which the user
decides on the variables to use for grouping and for the PCA.

The tables are written to disk as CSVs, are modality-neutral, and read only
``cell_metadata.tsv`` (never the counts matrix), so they cost nothing
appreciable and work against an already-cached extraction.

Three files are written:

- ``metadata_columns.csv`` -- one row per metadata column: dtype, distinct
  count, whether it is usable as a grouping variable, missing count, and
  min/median/max for numeric columns.
- ``metadata_values.csv`` -- one row per column+value: cell count and percent.
  This is where the exact spelling of a column's values becomes visible,
  which is what a ``--factor COLUMN:CONTROL`` needs.
- ``metadata_summary.csv`` -- the two above denormalized into one file.

Two different retention rules apply here:

- The metadata tables above are a direct export of the extraction, so they
  are overwritten by a new extraction -- they are a pure function of
  the extracted object, and re-deriving them from an unchanged extraction
  produces byte-identical files. They live flat in ``qc/`` and are rewritten
  whenever an extraction actually runs, and written if simply absent
  otherwise.
- Everything aggregated -- cells-per-sample tables and the global PCAs --
  is timestamped and never overwritten, so the user has a paper trail of
  what was tried. Those land
  in ``qc/<timestamp>_<modality>/``, each with a ``run_params.json``, and each
  run appends a row to ``qc/runs.csv`` so the whole history is readable at a
  glance without opening any of them.

Everything is written atomically, so concurrent SLURM array tasks
cannot tear each other's output or race to a partial
file.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from paralome.checkpoints import atomic_write_text, exclusive_checkpoint_lock

QC_DIRNAME = "qc"
RUN_INDEX_CSV = "runs.csv"
RUN_PARAMS_JSON = "run_params.json"

METADATA_COLUMNS_CSV = "metadata_columns.csv"
METADATA_VALUES_CSV = "metadata_values.csv"
METADATA_SUMMARY_CSV = "metadata_summary.csv"

# A column with more distinct values than this is not usable as a
# --replicate-column/--stratify-by/--factor: it would produce (nearly) one
# pseudobulk sample per cell. Same spirit as singlecell-inspect's own
# reporting cutoff, made explicit here because it drives `grouping_suitable`.
_MAX_GROUPING_DISTINCT = 50

_COLUMNS_FIELDS = [
    "column", "dtype", "n_distinct", "grouping_suitable", "n_missing",
    "min", "median", "max",
]
_VALUES_FIELDS = ["column", "value", "n_cells", "pct_of_cells"]
_SUMMARY_FIELDS = [
    "column", "dtype", "n_distinct", "grouping_suitable",
    "value", "n_cells", "pct_of_cells", "min", "median", "max",
]


def qc_dir(out_dir: Union[str, Path]) -> Path:
    """``<out>/qc`` -- a sibling of ``atac/``, ``rna/`` and ``singlecell_extraction/``.

    QC has its own subfolder at the same level as RNA and ATAC.
    """
    return Path(out_dir).expanduser().resolve() / QC_DIRNAME


def _numeric_series(series: pd.Series) -> Optional[pd.Series]:
    """Parse a column as numeric, or None if it isn't one.

    ``cell_metadata.tsv`` is deliberately read as all-strings
    (``dtype=str, keep_default_na=False`` in ``read_cell_metadata``) so that
    grouping labels survive verbatim -- nothing downstream wants pandas
    guessing that a cluster ID is an integer. That means dtype has to be
    inferred here.

    A column counts as numeric when every non-empty value parses AND either
    of these holds:

    - any value is non-integral (``percent.mt``, ``TSS.enrichment``,
      ``ATAC.weight``) -- a fractional value is never a grouping label; or
    - it has more than ``_MAX_GROUPING_DISTINCT`` distinct values
      (``nCount_ATAC``, ``nFeature_RNA``).

    Both halves are needed. Without the first, a float QC metric in a small
    object would be reported as a categorical with one "value" per cell.
    Without the second, ``seurat_clusters`` -- which parses cleanly as
    integers but is a grouping variable, not a measurement -- would get
    min/median/max instead of the value counts actually needed in order to
    choose ``--replicate-column``.
    """
    non_empty = series[series.astype(str).str.strip() != ""]
    if non_empty.empty:
        return None
    parsed = pd.to_numeric(non_empty, errors="coerce")
    if parsed.isna().any():
        return None
    has_fractional = bool((parsed != parsed.round()).any())
    if not has_fractional and non_empty.nunique() <= _MAX_GROUPING_DISTINCT:
        return None
    return parsed


def summarize_metadata(metadata: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    """Build the per-column and per-value QC rows from a cell-metadata frame."""
    total_cells = len(metadata)
    column_rows: List[Dict[str, object]] = []
    value_rows: List[Dict[str, object]] = []

    for column in metadata.columns:
        if column == "barcode":
            continue
        series = metadata[column]
        n_missing = int((series.astype(str).str.strip() == "").sum())
        numeric = _numeric_series(series)

        if numeric is not None:
            column_rows.append({
                "column": column,
                "dtype": "numeric",
                "n_distinct": int(series.nunique()),
                "grouping_suitable": "no",
                "n_missing": n_missing,
                "min": _round(numeric.min()),
                "median": _round(numeric.median()),
                "max": _round(numeric.max()),
            })
            continue

        counts = series.value_counts()
        n_distinct = int(len(counts))
        column_rows.append({
            "column": column,
            "dtype": "character",
            "n_distinct": n_distinct,
            "grouping_suitable": "yes" if n_distinct <= _MAX_GROUPING_DISTINCT else "no",
            "n_missing": n_missing,
            "min": "", "median": "", "max": "",
        })
        # Value counts are only meaningful (and only fit) for columns small
        # enough to actually group by; a 16,000-value barcode-like column
        # would otherwise dominate the file.
        if n_distinct <= _MAX_GROUPING_DISTINCT:
            for value, count in counts.items():
                value_rows.append({
                    "column": column,
                    "value": str(value),
                    "n_cells": int(count),
                    "pct_of_cells": round(100.0 * int(count) / total_cells, 2) if total_cells else "",
                })

    column_rows.sort(key=lambda row: str(row["column"]))
    return {"columns": column_rows, "values": value_rows, "n_cells": total_cells}


def _round(value: object) -> object:
    """Trim float noise the way singlecell-inspect trims its output."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    # Inf (a ratio over zero, say) has no int(); written as R writes it.
    if not math.isfinite(number):
        return "NaN" if math.isnan(number) else ("Inf" if number > 0 else "-Inf")
    if number == int(number) and abs(number) < 1e15:
        return int(number)
    return round(number, 4)


def _summary_rows(
    column_rows: Sequence[Dict[str, object]], value_rows: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Denormalize the two tables into the single-file layout."""
    by_column: Dict[str, List[Dict[str, object]]] = {}
    for row in value_rows:
        by_column.setdefault(str(row["column"]), []).append(row)

    rows: List[Dict[str, object]] = []
    for column_row in column_rows:
        name = str(column_row["column"])
        shared = {
            "column": name,
            "dtype": column_row["dtype"],
            "n_distinct": column_row["n_distinct"],
            "grouping_suitable": column_row["grouping_suitable"],
        }
        matching = by_column.get(name, [])
        if not matching:
            rows.append({
                **shared, "value": "", "n_cells": "", "pct_of_cells": "",
                "min": column_row["min"], "median": column_row["median"],
                "max": column_row["max"],
            })
            continue
        for value_row in matching:
            rows.append({
                **shared,
                "value": value_row["value"],
                "n_cells": value_row["n_cells"],
                "pct_of_cells": value_row["pct_of_cells"],
                "min": "", "median": "", "max": "",
            })
    return rows


def _render_csv(fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def write_metadata_qc(
    metadata: pd.DataFrame, out_dir: Union[str, Path], *, overwrite: bool = False
) -> Dict[str, object]:
    """Write the three metadata QC CSVs into ``<out>/qc/``, skipping any that exist.

    A file that exists is left alone; a missing one is built. Each file
    is written atomically via ``atomic_write_text`` (mkstemp + os.replace), so
    concurrent array tasks that both find a file missing still leave a
    complete file behind rather than a half-written one.

    Returns the three paths plus ``written``: the subset actually created this
    call, so callers can report "already present" without re-deriving it.
    """
    directory = qc_dir(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    summary = summarize_metadata(metadata)
    payloads = {
        METADATA_COLUMNS_CSV: _render_csv(_COLUMNS_FIELDS, summary["columns"]),
        METADATA_VALUES_CSV: _render_csv(_VALUES_FIELDS, summary["values"]),
        METADATA_SUMMARY_CSV: _render_csv(
            _SUMMARY_FIELDS, _summary_rows(summary["columns"], summary["values"])
        ),
    }

    written: List[str] = []
    paths: Dict[str, str] = {}
    for name, text in payloads.items():
        path = directory / name
        paths[name] = str(path)
        if overwrite or not path.is_file():
            atomic_write_text(path, text)
            written.append(name)

    return {
        "qc_dir": str(directory),
        "n_cells": summary["n_cells"],
        "n_columns": len(summary["columns"]),
        "grouping_candidates": [
            str(row["column"]) for row in summary["columns"]
            if row["grouping_suitable"] == "yes"
        ],
        # The rows themselves, so a caller can print the table
        # instead of naming a CSV and leaving the reader to go open it.
        # The message names the metadata file and reports its columns
        # directly on screen.
        "columns": list(summary["columns"]),
        "files": paths,
        "written": written,
    }


# --------------------------------------------------------------------------
# Timestamped aggregation runs -- the paper trail
# --------------------------------------------------------------------------

_RUN_INDEX_FIELDS = [
    "run_id", "timestamp", "modality", "stage", "group_by",
    "stratify_by", "min_cells_per_group", "removed_samples", "n_samples", "plan_id", "path",
]


def _run_timestamp() -> str:
    """Sortable, filesystem-safe, second-resolution local timestamp."""
    return datetime.now().strftime("%Y-%m-%dT%H%M%S")


def new_run_dir(out_dir: Union[str, Path], modality: str) -> Path:
    """Create and return a fresh ``qc/<timestamp>_<modality>/`` directory.

    Never reuses or overwrites an existing run: aggregated tables and PCAs
    are timestamped and kept. Two runs starting inside the same second
    (concurrent array tasks can) get ``_2``, ``_3`` suffixes rather than colliding.
    """
    directory = qc_dir(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{_run_timestamp()}_{modality}"
    candidate = directory / base
    suffix = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = directory / f"{base}_{suffix}"
            suffix += 1


def write_run_params(run_dir: Union[str, Path], params: Mapping[str, object]) -> Path:
    """Record exactly which parameters produced this run's tables and plots."""
    path = Path(run_dir) / RUN_PARAMS_JSON
    atomic_write_text(path, json.dumps(dict(params), indent=2, sort_keys=True, default=str) + "\n")
    return path


def append_run_index(out_dir: Union[str, Path], record: Mapping[str, object]) -> Path:
    """Append one row to ``qc/runs.csv`` -- the readable history of every run.

    This is the actual paper trail: one line per aggregation/PCA run showing
    the grouping, stratification, removals and resulting
    sample count, so "what did I already try, and where did it land" is
    answerable without opening any run folder. Appends are serialized under
    the same checkpoint lock style used elsewhere, since an array job can have
    several tasks finishing at once.
    """
    directory = qc_dir(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_INDEX_CSV
    row = {field: record.get(field, "") for field in _RUN_INDEX_FIELDS}
    for key, value in list(row.items()):
        if isinstance(value, (list, tuple)):
            row[key] = ";".join(str(item) for item in value)
    with exclusive_checkpoint_lock(path):
        exists = path.is_file()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_RUN_INDEX_FIELDS, lineterminator="\n")
            if not exists or os.fstat(handle.fileno()).st_size == 0:
                writer.writeheader()
            writer.writerow(row)
    return path


def record_run(
    out_dir: Union[str, Path],
    modality: str,
    stage: str,
    params: Mapping[str, object],
) -> Dict[str, object]:
    """Create a timestamped run directory, write its params, and index it.

    Convenience wrapper used by both the aggregation and global-PCA stages so
    the two produce an identical paper trail.
    """
    run_dir = new_run_dir(out_dir, modality)
    run_id = run_dir.name
    write_run_params(run_dir, {"run_id": run_id, "modality": modality, "stage": stage, **dict(params)})
    append_run_index(out_dir, {
        "run_id": run_id,
        "timestamp": run_id.split("_")[0],
        "modality": modality,
        "stage": stage,
        "path": str(run_dir),
        **{key: params.get(key, "") for key in (
            "group_by", "stratify_by",
            "min_cells_per_group", "removed_samples", "n_samples", "plan_id",
        )},
    })
    return {"run_dir": str(run_dir), "run_id": run_id}
