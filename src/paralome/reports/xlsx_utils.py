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

"""Significant-figure rounding for values written as report text.

Paralome writes no workbooks; this is the rounding the candidate report
applies to the float annotation values it writes as text.
"""

from __future__ import annotations

import math

DEFAULT_SIGNIFICANT_FIGURES = 6


def round_significant(
    value: object, significant_figures: int = DEFAULT_SIGNIFICANT_FIGURES
) -> object:
    """Round a float to a fixed number of significant figures.

    Only ``float`` values are affected -- every other type (``int``,
    ``str``, ``bool``, ``None``, etc.) passes through unchanged, as do
    non-finite floats (NaN/+Inf/-Inf) and exactly ``0.0``.

    Rounding to significant figures, rather than a fixed number of
    decimal places, is deliberate. These tables mix values that span many
    orders of magnitude within the same column family: a p-value column
    can hold both 0.04 and 4e-210; an expression/CPM column can hold both
    0.001 and 100000. A fixed decimal-place round would either collapse
    small p-values to 0.0000 (destroying the exact information a reader
    most wants to see) or leave large values with meaningless trailing
    precision. Rounding to N significant figures preserves relative
    precision regardless of magnitude.

    This only ever affects the value handed to a specific output (an
    Excel cell, a formatted report string). It never mutates the
    underlying row dict, so full float64 precision is preserved wherever
    else that same value is written -- TSV "machine" files, JSON
    checkpoints, and any in-memory filtering/statistics that run before
    a workbook is built.
    """
    if not isinstance(value, float) or not math.isfinite(value) or value == 0.0:
        return value
    magnitude = math.floor(math.log10(abs(value)))
    decimals = significant_figures - 1 - int(magnitude)
    return round(value, decimals)
