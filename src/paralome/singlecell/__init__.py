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

"""Modality-neutral single-cell machinery shared by ATAC and RNA.


Every statistic is computed by ``paralome.stats`` through the
shared plan and engine (``paralome.comparisons``, ``paralome.engine``), at the
pseudobulk level for both modalities. ``stats_core`` holds what the two
single-cell drivers share around that -- loading the pseudobulk comparison,
fitting it, writing it through ``paralome.results`` -- and the cell-level
(``:cell``) fits live in ``paralome.atac.singlecell_stats``, which both
modalities call, since their cell matrices are read the same way.
"""
