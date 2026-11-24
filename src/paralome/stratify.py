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

"""Which strata a sample is in: one rule for every pipeline.

A sample can be in several experiments, so a cell can be in several experiments too, but it is always summed into
the same sample, never into several samples.

* ``--stratify-by A,B`` names several columns; a stratum is one value of each, every combination present. Its key
  is the values joined by ``|`` (``Exp1|Retina``); its folder and file names join the values' slugs with ``__``.
* A cell of a stratum column may list several values, comma-separated: the sample is in each of them.
* A sample with no value in a stratum column is in no stratum, and is not tested by a stratified run.

A pseudobulk sample is summed once (replicate x the stratum columns' values); these rules only say which of the
finished samples each stratum's analysis takes. Nothing here imports the rest of Paralome.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Between the columns' values in a stratum's key.
JOIN = "|"
#: Read as "no value" in a stratum column (pandas writes a missing value as nan).
_BLANK = {"", "nan", "none"}


def stratum_columns(stratify_by: Optional[str]) -> List[str]:
    """``"Experiment, tissue_strata"`` -> ``["Experiment", "tissue_strata"]``; None/"" -> []."""
    return [part.strip() for part in str(stratify_by or "").split(",") if part.strip()]


def cell_values(value: Any) -> List[str]:
    """The values one cell of a stratum column lists: comma-separated, blanks dropped, order kept."""
    seen: List[str] = []
    for part in str("" if value is None else value).split(","):
        part = part.strip()
        if part.lower() not in _BLANK and part not in seen:
            seen.append(part)
    return seen


def row_strata(row: Mapping[str, Any], stratify_by: Optional[str]) -> List[str]:
    """Every stratum key this sample is in ([] when a stratum column has no value for it)."""
    columns = stratum_columns(stratify_by)
    if not columns:
        return []
    lists = [cell_values(row.get(column)) for column in columns]
    if any(not values for values in lists):
        return []
    return [JOIN.join(combo) for combo in product(*lists)]


def in_stratum(row: Mapping[str, Any], stratify_by: Optional[str], key: Optional[str]) -> bool:
    return key is None or str(key) in row_strata(row, stratify_by)


def frame_in_stratum(frame, stratify_by: Optional[str], key: Optional[str]):
    """A boolean Series: the frame's rows (cells, samples) in stratum ``key``."""
    import pandas as pd

    columns = stratum_columns(stratify_by)
    if key is None or not columns or any(c not in frame.columns for c in columns):
        return pd.Series(True, index=frame.index)
    combos = frame[columns].astype(str)
    memo: Dict[tuple, bool] = {}
    flags = []
    for values in combos.itertuples(index=False, name=None):
        if values not in memo:
            memo[values] = str(key) in row_strata(dict(zip(columns, values)), stratify_by)
        flags.append(memo[values])
    return pd.Series(flags, index=frame.index)


def stratum_label(key: str, stratify_by: Optional[str]) -> str:
    """``tissue_strata: Retina`` / ``Experiment: Exp1, tissue_strata: Retina``, for titles and logs."""
    columns = stratum_columns(stratify_by)
    parts = str(key).split(JOIN)
    if len(parts) != len(columns):
        return str(key)
    return ", ".join(f"{column}: {part}" for column, part in zip(columns, parts))


def missing_columns(stratify_by: Optional[str], available: Sequence[str]) -> List[str]:
    have = set(available)
    return [column for column in stratum_columns(stratify_by) if column not in have]
