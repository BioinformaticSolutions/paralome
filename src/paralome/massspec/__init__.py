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

"""Mass spectrometry: proteomics, paired phosphoproteomics, and (prepared) metabolomics.

One command per omics (`paralome proteomics`, later `paralome metabolomics`)
over one shared layer:

    exporters   every quantification exporter's table -> one ``MsTable``
    samplesheet which abundance column is which sample, and its metadata
    prepare     log-scale detection, zeros as missing, per-sample scaling,
                the paired-loading test, the half-minimum fill
    identity    proteins to Ensembl genes, through the core gene_identity
    stages      each stage run and checkpointed with the core checkpoint helpers
    differential the core statistics (singlecell.stats_core / engine), missing
                values at half the feature's lowest

Nothing here computes a statistic of its own or keeps its own checkpoint
format: those are the core's.
"""
