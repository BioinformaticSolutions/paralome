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

"""What a pseudobulk sample is: the --replicate-column, split by --stratify-by.

There can be several ID strata in a cell table, so the user names the one
column whose value is a cell's biological sample; the cells of each sample are
summed into one pseudobulk sample, one per stratum when --stratify-by is given.
--factor columns are not part of the grouping: each must be one value per
sample, and a column that is not is refused (PLM-0203) rather than split on.

--replicate-column is required on every single-cell run, pseudobulk and :cell
alike. --group-by, which made the user list the grouping columns by hand,
is retired: naming it stops with PLM-0119.

Kept free of numpy/pandas so the command modules can import it at start-up.
"""

from __future__ import annotations

from typing import List, Optional

from paralome.errors import ParalomeError

REPLICATE_COLUMN_HELP = (
    "Required. The cell-metadata column naming each cell's biological sample (library, "
    "animal, donor). Its values are the pseudobulk samples -- one per --stratify-by value "
    "when that is given -- and what --remove matches. Every --factor must be one value per "
    "sample."
)
RETIRED_GROUP_BY_HELP = "Retired: name the sample column with --replicate-column."


def pile_columns(
    replicate_column: Optional[str],
    stratify_by: Optional[str] = None,
    *,
    retired_group_by: Optional[str] = None,
) -> List[str]:
    """The cell-metadata columns whose values make one pseudobulk sample.

    ``[replicate_column]``, then ``stratify_by`` when given (and not the same
    column), so a stratified run has one sample per replicate per stratum -- the
    same samples, and the same sample_ids, as ``--group-by <replicate>,<stratum>``
    made.
    """
    if retired_group_by is not None:
        raise ParalomeError(                                   # PLM-0119
            119,
            f"--group-by is retired (given: {retired_group_by!r}). Name the column that "
            "identifies each biological sample with --replicate-column; the pseudobulk "
            "samples are its values, one per --stratify-by value when that is given.",
        )
    column = str(replicate_column or "").strip()
    if not column:
        raise ParalomeError(101, "--replicate-column is required: the cell-metadata column "
                                 "naming each cell's biological sample")
    if "," in column:
        raise ParalomeError(                                   # PLM-0202
            202,
            f"--replicate-column names one column, got {column!r}. The samples are that "
            "column's values; a second grouping is --stratify-by.",
        )
    columns = [column]
    # --stratify-by may name several columns; each is part of the sample.
    from paralome.stratify import stratum_columns

    for stratum in stratum_columns(stratify_by):
        if stratum != column and stratum not in columns:
            columns.append(stratum)
    return columns
