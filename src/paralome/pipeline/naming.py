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

import re
from pathlib import Path

# Every conventional stage-workdir basename `pipeline run`/`continue` ever
# creates (e.g. project/motif_analysis, project/peak_quantification) -- kept
# in sync with paralome.atac.naming._GENERIC_STAGE_DIRNAMES, which solves the
# identical "climb past the generic stage dirname to the real per-comparison
# project directory" problem for output *file* prefixes. Before this list was
# widened, most of these weren't here at all, so `slurm_job_name()` treated
# the stage's own workdir (e.g. "motif_analysis") as the "project" -- every
# concurrently-running comparison's motif/peak/differential jobs showed up in
# squeue with the exact same name, indistinguishable from one another.
_STAGE_DIRS = {
    "fastq_qc", "alignment", "star", "dedup", "freebayes", "annotation", "annovar",
    "segregation", "filter", "candidate_report", "reporting",
    "rna_quantification", "differential_expression", "macs", "merged_peaks",
    "peak_quantification", "differential_accessibility", "motif_analysis",
    "singlecell_atac", "singlecell_differential",
    "singlecell_motif_analysis", "pseudobulk", "motifs", "variant_run",
    # The shared per-dataset folder and the extraction inside it. "pseudobulk" stays: older project
    # trees still use it as the folder name, and a basename this list does not know makes
    # slurm_job_name() treat it as the project (every comparison's jobs would share one name in squeue).
    "data", "extraction", "singlecell_extraction",
    "singlecell_rna", "singlecell_rna_differential",
    "functional_evidence",
    # The DNA gene-burden stage's folder.
    "gene_burden",
    # The variant browser bundle stage's folder (renamed in place on the next run).
    "browser_bundle",
    # Its folders, and the omicview folder every zip goes in.
    "statistics_output", "genotype_output", "omicview",
    # The genome tracks stage's folder.
    "genome_tracks",
    # The trajectories stage's folder.
    "trajectories",
}

# the directory layout has a modality layer between the project
# directory and ATAC/RNA-specific stage dirs (e.g.
# project/atac/singlecell_differential). The climb passes the modality dir
# with the same single-level check.
# "data" joins these because the shared per-dataset folder is the same kind
# of intermediate layer: <parent>/data/extraction is two levels deep, so
# without it project_identifier() stops at "data" and every dataset's
# extraction job is named "data_singlecell_extract".
_MODALITY_DIRS = {"atac", "rna", "data"}

def project_identifier(workdir: str | Path) -> str:
    """Return a concise scheduler-safe project identifier for a stage workdir.

    The climb is iterative: the shared data folder adds
    ``<parent>/data/<modality>/`` and ``<parent>/data/extraction`` to the tree,
    so ``scRNA/data/rna`` must climb past both ``rna`` and ``data`` for jobs
    to be told apart in squeue.

    Bounded so a pathological path cannot climb to the filesystem root, and it
    never consumes the last component -- a project genuinely named ``motifs``
    keeps its name rather than resolving to its parent.
    """
    path = Path(workdir)
    for _ in range(4):
        if path.name in _STAGE_DIRS or path.name in _MODALITY_DIRS:
            parent = path.parent
            if str(parent) in ("", ".", "/") or not parent.name:
                break
            path = parent
            continue
        break
    candidate = path.name
    for suffix in ("_variant", "-variant", "_project", "-project"):
        if candidate.lower().endswith(suffix):
            candidate = candidate[:-len(suffix)]
            break
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_.-")
    return candidate or "paralome"

def slurm_job_name(workdir: str | Path, stage: str) -> str:
    project = project_identifier(workdir)
    stage_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage)).strip("_.-") or "stage"
    # The project part is shortened, never the stage token, so a
    # long project id still gives each stage its own job name.
    stage_token = stage_token[:120]
    project = project[: max(1, 128 - len(stage_token) - 1)]
    return f"{project}_{stage_token}"
