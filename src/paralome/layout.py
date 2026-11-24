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

"""Canonical on-disk directory names for a single-cell project.

One generic ``data`` folder with the extractions within it, serving every
pipeline that starts with an R object, and one place for the names.

WHY ``data`` AND NOT ``pseudobulk``. Pseudobulk is one specific operation --
collapsing per-cell counts into one value per sample x stratum group -- and
the shared folder holds much more than it: the raw per-cell extraction, both
modalities' aggregated matrices, and the QC tables. The extraction in
particular does NOT pseudobulk anything: ``extract_seurat_assays.R`` writes
one column per cell; the collapse to groups happens afterwards in
``singlecell/pseudobulk_core.py``. In this repository ``pseudobulk`` means the
aggregation and nothing else.

WHY THEY LIVE HERE. A layout name written out by hand in several modules
changes in some copies and not others, silently; every module asks here
instead.

    <project-parent>/data/          <- shared across every comparison
        extraction/                 <- per-cell matrices from the rds
            checkpoints/singlecell_extract.json
        rna/                        <- pseudobulk aggregation, RNA
        atac/                       <- pseudobulk aggregation, ATAC
        qc/

``data/`` sits BESIDE the project rather than inside it because one rds
extraction serves many comparisons; re-reading a multi-gigabyte object per contrast
would be the expensive kind of wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

#: The shared per-dataset working folder. Everything derived from one rds.
DATA_DIRNAME = "data"

#: Per-cell matrices and cell metadata read out of the rds. Not aggregated.
EXTRACTION_DIRNAME = "extraction"

#: Aggregation QC tables written alongside the extraction.
QC_DIRNAME = "qc"


def data_workdir(project: Union[str, Path], *, rds: Optional[Union[str, Path]] = None) -> Path:
    """The shared data folder for ``project``.

    The data folder belongs to the parent project, the
    folder every analysis of one object sits under, however deep. The
    nearest ``data/`` at or above ``project`` that already holds an extraction
    is that folder (``parent_data_folder``). Only when there is none does the
    rule below choose one.

    Beside the project when it has a real parent, inside it otherwise -- a
    bare relative ``--project foo`` has no meaningful sibling directory.
    """
    found = parent_data_folder(project, rds=rds)
    if found is not None:
        return found
    return beside_project(project)


def beside_project(project: Union[str, Path]) -> Path:
    """The rule for a project with no parent data folder yet: beside it, or inside a bare one."""
    project = Path(project)
    parent = project.parent
    if str(parent) in ("", "."):
        return project / DATA_DIRNAME
    return parent / DATA_DIRNAME


def _extracted_object(data: Path) -> Optional[str]:
    """The object ``data``'s extraction was read from, or None when it has none."""
    import json

    from paralome.checkpoints import stage_checkpoint_path

    point = stage_checkpoint_path(Path(data) / EXTRACTION_DIRNAME, "singlecell_extract")
    try:
        stored = json.loads(point.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict) or not stored.get("complete"):
        return None
    return str(stored.get("rds") or "")


def parent_data_folder(
    project: Union[str, Path], *, rds: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """The nearest ``data/`` at or above ``project`` holding a completed extraction.

    Analyses of one object are subfolders, at any depth,
    of one parent project; ``paralome extract --project <parent>`` puts the
    extraction in ``<parent>/data``, and every run below it finds it here.

    ``rds``, when the caller knows it, must be the object that extraction was
    read from. A ``data/`` higher up that belongs to another object is passed
    over rather than used -- using it would re-extract over somebody else's
    data. Callers that only read an extraction pass no ``rds``.
    """
    start = Path(project).expanduser().resolve()
    wanted = str(Path(str(rds)).expanduser().resolve()) if rds is not None else None
    for folder in (start, *start.parents):
        data = folder / DATA_DIRNAME
        source = _extracted_object(data)
        if source is None:
            continue
        if wanted is not None and str(Path(source).resolve()) != wanted:
            continue
        return data
    return None


def parent_folder_holding(
    project: Union[str, Path], *, subdir: str, stage: str, key: str, source: Union[str, Path],
) -> Optional[Path]:
    """The nearest ``data/`` at or above ``project`` whose ``subdir`` holds a
    complete ``stage`` checkpoint made from ``source`` (recorded under ``key``).

    parent-project rule for any data-level stage, not only
    the single-cell extraction -- the mass-spec import uses it, so every
    analysis under one parent project reads the one import of one export.
    """
    import json

    from paralome.checkpoints import stage_checkpoint_path

    start = Path(project).expanduser().resolve()
    wanted = str(Path(str(source)).expanduser().resolve())
    for folder in (start, *start.parents):
        data = folder / DATA_DIRNAME
        point = stage_checkpoint_path(data / subdir, stage)
        try:
            stored = json.loads(point.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(stored, dict) and stored.get("complete") and str(stored.get(key) or "") == wanted:
            return data
    return None


def extraction_dir(data_dir: Union[str, Path]) -> Path:
    """Where the per-cell extraction lives under a data folder."""
    return Path(data_dir) / EXTRACTION_DIRNAME


def qc_dir(data_dir: Union[str, Path]) -> Path:
    """Where the aggregation QC tables live under a data folder."""
    return Path(data_dir) / QC_DIRNAME


#: Folder names this layout used before the rename. Read-only: nothing writes
#: here, and there is no compatibility shim, by policy. They exist
#: so a stale tree can be RECOGNISED and reported rather than silently ignored.
LEGACY_DATA_DIRNAMES = ("pseudobulk",)
LEGACY_EXTRACTION_DIRNAMES = ("singlecell_extraction",)


def stale_layout_warning(data_dir: Union[str, Path]) -> Optional[str]:
    """Warn when a pre-rename tree sits where the data folder should be.

    A script still passing ``--out scRNA/pseudobulk`` writes there, while the
    pipeline defaults to ``scRNA/data`` -- so the pipeline would find no
    checkpoint and re-read an rds it already had, with neither command saying
    the two point at different directories. A directory that does not exist is
    indistinguishable from one that was never made, unless someone looks next
    door.
    """
    data_dir = Path(data_dir)
    if data_dir.exists():
        # The folder renamed (or --data-workdir pointed at the old
        # one) but the extraction inside it still under its old name.
        inner = _legacy_extraction_in(data_dir)
        if inner is None:
            return None
        return (
            f"{data_dir} holds {inner}, the pre-rename name of the extraction folder; "
            f"the extraction is now read from {extraction_dir(data_dir)}. It will be "
            f"rebuilt from the rds unless you rename it:  mv {inner} {extraction_dir(data_dir)}"
        )
    parent = data_dir.parent
    for legacy in LEGACY_DATA_DIRNAMES:
        candidate = parent / legacy
        if candidate.exists():
            # The extraction inside is renamed too, or it is not found.
            inner = _legacy_extraction_in(candidate)
            inner_step = (
                f" && mv {data_dir / inner.name} {extraction_dir(data_dir)}" if inner is not None else ""
            )
            pointed = (
                f" (after  mv {inner} {extraction_dir(candidate)})" if inner is not None else ""
            )
            return (
                f"{data_dir} does not exist, but {candidate} does. That is the "
                f"pre-rename layout: the shared folder is now {DATA_DIRNAME!r} and the "
                f"extraction inside it is {EXTRACTION_DIRNAME!r}. Everything here will be "
                f"rebuilt from the rds unless you point --data-workdir at {candidate}{pointed} or "
                f"rename it:  mv {candidate} {data_dir}{inner_step}"
            )
    return None


def _legacy_extraction_in(folder: Path) -> Optional[Path]:
    """An old-named extraction folder in ``folder`` with no new-named one beside it."""
    if extraction_dir(folder).exists():
        return None
    for legacy in LEGACY_EXTRACTION_DIRNAMES:
        candidate = Path(folder) / legacy
        if candidate.is_dir():
            return candidate
    return None
