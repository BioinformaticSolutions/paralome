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

"""The trajectories stage: an object's lineages, and how treatment changes them.

Decisions: the whole object; the cluster column and the reduction chosen here, not by the user; the root
demanded (``--trajectory-root <column>:<value>``) unless the run opts out (``--no-trajectories``); the whole
transcriptome; the statistics use the same statistical methodology as the project, no different; 5 bins by
default.

1. Monocle3 (the default) or Slingshot (``r/slingshot_trajectories.R``) fits the lineages on
   every cell of the extraction, from the chosen reduction and cluster column, rooted at the named cluster: each cell's pseudotime and weight on every lineage,
   the lineage it is on (``branch``, Slingshot's own assignment: ``L1,L2`` for a cell before the lineages part),
   and the curves drawn in each 2D/3D embedding the object holds.
2. The routes the fit made are grouped into the lineages the statistics follow (--trajectory-lineages:
   fate, one per end state, the default; exit, one per way out of the root and end state; route, every route).
3. Expression (paralome.singlecell.trajectory_genes): each gene along each lineage's pseudotime by the
   run's full factorial crossed with a natural spline of pseudotime, fitted on the lineage's cells by least squares
   (--trajectory-model ls, the default) or by Monocle3's fit_models (monocle3). There are no per-segment statistics.
4. Composition: each lineage's cells are cut into ``bins`` segments of equal cells by pseudotime
   (``trajectory_L1_bin1`` ...; a cell on two lineages is in a segment of each); per replicate, the share of its
   lineage's cells in each segment (along pseudotime) and the share of its fitted cells on each branch (between
   lineages), tested by the run's differential with the same statistic at the replicate level (anova:cell is anova
   there: a share is a replicate's).

Everything is written under the stage folder and zipped into ``<prefix>_<modality>_trajectory_output.zip`` in the
run's omicview/trajectories folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paralome.checkpoints import atomic_text_writer, atomic_write_text, now_iso, stage_checkpoint_path
from paralome.errors import ParalomeError

STAGE = "trajectories"
METHOD = "trajectories.v1"
SCHEMA = "paralome.trajectory_output.v1"
ZIP_SUFFIX = "trajectory_output"
DEFAULT_BINS = 5
SEGMENT_COLUMN = "trajectory_segment"
SEGMENT_PREFIX = "trajectory_"
MISSING = "."

R_SCRIPT = Path(__file__).resolve().parent / "r" / "slingshot_trajectories.R"
# Monocle3's
# principal graph in the object's UMAP, its lineages every path from the root to a leaf ("root to leaves is what I
# am expecting.  How each leaf was derived is the idea"). The default.
MONOCLE3_R_SCRIPT = Path(__file__).resolve().parent / "r" / "monocle3_trajectories.R"
METHODS = ("monocle3", "slingshot")
DEFAULT_METHOD = "monocle3"
R_SCRIPTS = {"slingshot": R_SCRIPT, "monocle3": MONOCLE3_R_SCRIPT}
# What each method writes beside cells.tsv, lineages.tsv, mst.tsv and the curves: its record and its log.
FIT_OUTPUTS = ("cells.tsv", "lineages.tsv", "mst.tsv", "labels.tsv", "tips.tsv", "graph.rds",
               *(f"{name}.{ext}" for name in METHODS for ext in ("json", "log")))
# What is kept when only the welds change -- Monocle3's learned graph and the run's labels.
GRAPH_FILES = ("graph.rds", "labels.tsv")
# --trajectory-max-routes.
DEFAULT_MAX_ROUTES = 200
# The fit's settings applied once the graph is learned: changing only these keeps it.
AFTER_THE_GRAPH = ("welds", "max_routes")
# What is not compared when a learned graph is to be kept: those, and the whole script's checksum (its part that
# learns the graph is compared as graph_script).
_NOT_THE_GRAPH = (*AFTER_THE_GRAPH, "r_script")
_WELD = re.compile(r"^L\d+(=L\d+)+$|^L\d+~L\d+$|^L\d+->L\d+$")


def parse_welds(welds: Optional[Sequence[str]], method: Optional[str] = None) -> List[str]:
    """--trajectory-weld,
    repeatable: L39=L42 joins the two tips; L30=L33=L35 joins each tip to a new node at their middle; L39~L42 joins
    L39's tip to the nearest node of L42's own branch; L39->L42 joins L39's tip to L42's, L42's tip staying an end
    ("a leaf is the end point, but another leaf is shorted and should have extended to the true leaf"). The names are the learned graph's tips (tips.tsv). Monocle3's
    only."""
    out: List[str] = []
    for raw in welds or ():
        weld = str(raw).strip().upper().replace(" ", "")
        if re.match(r"^L\d+-$", weld):
            raise ParalomeError(                               # PLM-0127
                127, f"--trajectory-weld {raw!r}: quote a directional weld, e.g. --trajectory-weld 'L39->L42' -- "
                     f"unquoted, the shell reads > as a redirect (and wrote a file named after the second tip)",
            )
        if not _WELD.match(weld):
            raise ParalomeError(                               # PLM-0127
                127, f"--trajectory-weld {raw!r}: give L<a>=L<b> (tip to tip), L<a>=L<b>=L<c>... (the tips to a "
                     f"new node at their middle), L<a>~L<b> (L<a>'s tip to the nearest point of L<b>'s branch) or "
                     f"'L<a>->L<b>' (L<a>'s tip to L<b>'s, L<b>'s staying an end; quoted), the lineages named as the fit's "
                     f"tips.tsv names them",
            )
        named = re.split(r"->|=|~", weld)
        if len(set(named)) != len(named):
            raise ParalomeError(127, f"--trajectory-weld {raw!r}: a tip is named twice")   # PLM-0127
        out.append(weld)
    if out and parse_method(method) != "monocle3":
        raise ParalomeError(                                   # PLM-0125
            125, "--trajectory-weld joins tips of Monocle3's graph; Slingshot has none. Leave it out, or use "
                 "--trajectory-method monocle3",
        )
    return out


def parse_method(text: Optional[str]) -> str:
    """--trajectory-method: monocle3 (the default) or slingshot, any case; "monocle" is monocle3."""
    raw = str(text or DEFAULT_METHOD).strip().lower()
    raw = "monocle3" if raw == "monocle" else raw
    if raw not in METHODS:
        raise ParalomeError(                                   # PLM-0125
            125, f"--trajectory-method {text!r}: give monocle3 (the default) or slingshot",
        )
    return raw


def fit_method(fit_dir: Path) -> str:
    """The method a fit folder's fit was made with, from the record it wrote (slingshot.json / monocle3.json)."""
    for name in METHODS:
        if (Path(fit_dir) / f"{name}.json").is_file():
            return name
    return "slingshot"


# --- the request -------------------------------------------------------------------------------------------------

def parse_root(text: Optional[str]) -> Tuple[str, str]:
    """``<column>:<value>`` (split at the first colon, as the other column:value options are)."""
    raw = str(text or "").strip()
    column, sep, value = raw.partition(":")
    column, value = column.strip(), value.strip()
    if not raw or not sep or not column or not value:
        raise ParalomeError(                                   # PLM-0121
            121,
            f"--trajectory-root must be <column>:<value>, the cell-metadata column and the cluster the lineages "
            f"start from (given: {raw!r})",
        )
    return column, value


FIT_BY = ("random", "centroid")


FIT_EXTEND = ("y", "n", "pc1")
FIT_KEYS = ("cells", "dispersion", "stretch", "extend", "reduction", "df", "shrink")
# Monocle3's own, learn_graph's: the graph's number of nodes, how long a branch must be to be kept, and
# whether a graph is learned in each partition of the UMAP (y, Monocle3's default) or across every cell (n).
MONOCLE3_KEYS = ("reduction", "ncenter", "minimal_branch_len", "use_partition",
                 # learn_graph's
                 # own, by Monocle3's names (L1.gamma, L1.sigma as l1_gamma, l1_sigma).
                 "close_loop", "geodesic_distance_ratio", "euclidean_distance_ratio", "orthogonal_proj_tip",
                 "prune_graph", "l1_gamma", "l1_sigma")
# Each Monocle3 key: how it is read, and the R script's option for it.
_MONOCLE3_YES_NO = ("use_partition", "close_loop", "orthogonal_proj_tip", "prune_graph")
_MONOCLE3_POSITIVE = ("geodesic_distance_ratio", "euclidean_distance_ratio", "l1_gamma", "l1_sigma")
METHOD_KEYS = {"slingshot": FIT_KEYS, "monocle3": MONOCLE3_KEYS}
ALL_FIT_KEYS = FIT_KEYS + tuple(key for key in MONOCLE3_KEYS if key not in FIT_KEYS)


def parse_fit_cells(text: Optional[str], method: str = "slingshot") -> Optional[Dict[str, Any]]:
    """--trajectory-fit-cells: how the curves are fitted. None (every cell, Slingshot's defaults) when not given.

    the curves are fitted on <n> cells of each cluster
    and every cell is then placed on them (slingshot's predict), for "speed and accuracy trials, something like
    10, 100, and 1000".
    A comma list of key=value: cells=<n> (per cluster), dispersion=random|centroid (how they are
    chosen; random when cells is given alone), stretch=<s> (how far each curve runs past its last cluster;
    Slingshot's default 2), extend=y|n|pc1 (how a curve's ends are drawn; Slingshot's default y),
    reduction=<name> (the reduction the curves are fitted in, e.g. umap; by default the object's integrated one,
    else pca). <n>:random and <n>:centroid are still cells=<n>,dispersion=random|centroid.
    df=<n> (the
    smoothing spline's degrees of freedom for every curve, more than 1; by default Slingshot lets smooth.spline
    choose them by cross-validation) and shrink=<0-1> (how far the lineages' shared start is pulled together;
    Slingshot's default 1).
    Under --trajectory-method monocle3 the keys are Monocle3's: reduction=<name> (the 2D/3D UMAP the
    graph is learned in; by default the object's UMAP), ncenter=<n>, minimal_branch_len=<n>, use_partition=y|n;
    Also close_loop=y|n, geodesic_distance_ratio=<x>, euclidean_distance_ratio=<x>,
    orthogonal_proj_tip=y|n, prune_graph=y|n, l1_gamma=<x>, l1_sigma=<x> (learn_graph's own; x more than 0).
    Each method's keys are refused under the other.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    method = parse_method(method)
    allowed = METHOD_KEYS[method]

    def refuse(why: str) -> ParalomeError:
        if method == "monocle3":
            return ParalomeError(                              # PLM-0123
                123, f"--trajectory-fit-cells {raw!r}: {why}. With --trajectory-method monocle3 give key=value "
                     f"pairs separated by commas: reduction=<name>, ncenter=<n>, minimal_branch_len=<n>, "
                     f"use_partition=y|n, close_loop=y|n, geodesic_distance_ratio=<x>, euclidean_distance_ratio=<x>, "
                     f"orthogonal_proj_tip=y|n, prune_graph=y|n, l1_gamma=<x>, l1_sigma=<x>; "
                     f"e.g. use_partition=n,ncenter=1000",
            )
        return ParalomeError(                                  # PLM-0123
            123, f"--trajectory-fit-cells {raw!r}: {why}. Give <n>:random or <n>:centroid, or key=value pairs "
                 f"separated by commas: cells=<n>, dispersion=random|centroid, stretch=<s>, extend=y|n|pc1, "
                 f"reduction=<name>, df=<n>, shrink=<0-1>; e.g. cells=1000,dispersion=random,stretch=0,extend=n,df=3",
        )

    settings: Dict[str, Any] = {}
    if "=" not in raw:
        if method == "monocle3":
            raise refuse("<n>:random and <n>:centroid choose the cells Slingshot fits; Monocle3 learns its graph "
                         "on every cell")
        count, sep, how = raw.partition(":")
        how = how.strip().lower()
        try:
            number = int(count.strip())
        except ValueError:
            number = 0
        if not sep or number < 1 or how not in FIT_BY:
            raise refuse("not <n>:random or <n>:centroid")
        return {"cells": number, "dispersion": how}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        key, value = key.strip().lower(), value.strip()
        if not sep or key not in allowed:
            other = [name for name in METHODS if name != method and key in METHOD_KEYS[name]]
            if sep and other:
                raise refuse(f"{key}= is {other[0]}'s, not {method}'s (--trajectory-method)")
            raise refuse(f"{item.strip()!r} is not one of {', '.join(f'{k}=' for k in allowed)}")
        if key in settings:
            raise refuse(f"{key} is given twice")
        if key == "cells":
            try:
                settings[key] = int(value)
            except ValueError:
                settings[key] = 0
            if settings[key] < 1:
                raise refuse("cells must be a whole number of cells per cluster, 1 or more")
        elif key == "dispersion":
            if value.lower() not in FIT_BY:
                raise refuse("dispersion must be random or centroid")
            settings[key] = value.lower()
        elif key == "stretch":
            try:
                settings[key] = float(value)
            except ValueError:
                settings[key] = -1.0
            if not settings[key] >= 0:
                raise refuse("stretch must be 0 or more")
        elif key == "extend":
            if value.lower() not in FIT_EXTEND:
                raise refuse("extend must be y, n or pc1")
            settings[key] = value.lower()
        elif key == "reduction":
            if not value:
                raise refuse("reduction needs a name")
            settings[key] = value
        elif key == "df":
            try:
                settings[key] = float(value)
            except ValueError:
                settings[key] = 0.0
            if not settings[key] > 1:
                raise refuse("df must be more than 1 (2 is close to straight)")
        elif key == "shrink":
            try:
                settings[key] = float(value)
            except ValueError:
                settings[key] = -1.0
            if not 0 <= settings[key] <= 1:
                raise refuse("shrink must be from 0 to 1")
        elif key in ("ncenter", "minimal_branch_len"):
            try:
                settings[key] = int(value)
            except ValueError:
                settings[key] = 0
            if settings[key] < 1:
                raise refuse(f"{key} must be a whole number, 1 or more")
        elif key in _MONOCLE3_YES_NO:
            if value.lower() not in ("y", "n"):
                raise refuse(f"{key} must be y or n")
            settings[key] = value.lower()
        elif key in _MONOCLE3_POSITIVE:
            try:
                settings[key] = float(value)
            except ValueError:
                settings[key] = 0.0
            if not settings[key] > 0:
                raise refuse(f"{key} must be a number more than 0")
    if "dispersion" in settings and "cells" not in settings:
        raise refuse("dispersion chooses the cells: give cells=<n> with it")
    if "cells" in settings:
        settings.setdefault("dispersion", "random")
    return settings or None


def fit_cells_record(settings: Optional[Mapping[str, Any]]) -> Any:
    """How a fit's settings are recorded in its request: [n, how] for cells and dispersion alone (so an older
    fit is the same fit now), else every setting given."""
    if not settings:
        return None
    if set(settings) <= {"cells", "dispersion"} and "cells" in settings:
        return [int(settings["cells"]), str(settings["dispersion"])]
    return {key: settings[key] for key in ALL_FIT_KEYS if key in settings}


def zip_name(prefix: str, modality: str) -> str:
    return f"{prefix}_{modality}_{ZIP_SUFFIX}.zip"


def _reduction_dims(path: Path) -> int:
    with path.open() as handle:
        return max(0, len(handle.readline().rstrip("\n").split("\t")) - 1)


def choose_reduction(extraction_dir: Path, modality: str) -> Tuple[str, Path, bool]:
    """(name, file, drop the first dimension). The reductions the extraction wrote (every one the object held).

    RNA (scRNA): an integrated reduction (a name with harmony, then integrated) when the object has one, else
    pca. ATAC (snATAC): harmony, then integrated, else lsi without its first dimension. Only a reduction of more
    than three dimensions is a candidate: a 2D/3D one (UMAP, t-SNE) is for drawing, not fitting.
    """
    folder = Path(extraction_dir) / "reductions"
    found = {path.stem: path for path in sorted(folder.glob("*.tsv"))} if folder.is_dir() else {}
    fit = {name: path for name, path in found.items() if _reduction_dims(path) > 3}
    lower = {name.lower(): name for name in fit}

    def first(*patterns: str) -> Optional[str]:
        for pattern in patterns:
            for low, name in sorted(lower.items()):
                if pattern in low:
                    return name
        return None

    if modality == "snATAC":
        name = first("harmony", "integrated")
        if name:
            return name, fit[name], False
        name = first("lsi")
        if name:
            return name, fit[name], True
    else:
        name = first("harmony", "integrated") or first("pca")
        if name:
            return name, fit[name], False
    raise ParalomeError(                                       # PLM-0235
        235,
        f"The object holds no reduction the lineages can be fitted on ({'pca' if modality != 'snATAC' else 'lsi'} "
        f"or an integrated one, more than three dimensions); the extraction wrote: "
        f"{', '.join(f'{n} ({_reduction_dims(p)} dims)' for n, p in found.items()) or 'none'}",
    )


def named_reduction(extraction_dir: Path, name: str) -> Tuple[str, Path]:
    """The reduction --trajectory-fit-cells reduction=<name> names (as the extraction wrote it, any case)."""
    folder = Path(extraction_dir) / "reductions"
    found = {path.stem: path for path in sorted(folder.glob("*.tsv"))} if folder.is_dir() else {}
    for stem, path in found.items():
        if stem.lower() == name.strip().lower():
            return stem, path
    raise ParalomeError(                                       # PLM-0235
        235,
        f"--trajectory-fit-cells reduction={name}: the object holds no reduction of that name; the extraction "
        f"wrote: {', '.join(f'{n} ({_reduction_dims(p)} dims)' for n, p in found.items()) or 'none'}",
    )


def display_embeddings(extraction_dir: Path) -> List[Tuple[str, Path]]:
    """Every 2D or 3D reduction (UMAP, t-SNE): the curves are drawn in each."""
    folder = Path(extraction_dir) / "reductions"
    if not folder.is_dir():
        return []
    return [(path.stem, path) for path in sorted(folder.glob("*.tsv")) if _reduction_dims(path) in (2, 3)]


def read_cell_metadata(extraction_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(extraction_dir) / "cell_metadata.tsv", sep="\t", dtype=str, keep_default_na=False)


def check_root(metadata: pd.DataFrame, column: str, value: str) -> None:
    if column not in metadata.columns:
        raise ParalomeError(                                   # PLM-0234
            234,
            f"--trajectory-root {column}:{value}: the object's cell metadata has no column {column!r}. "
            f"Columns: {', '.join(c for c in metadata.columns if c != 'barcode')}",
        )
    values = set(metadata[column].astype(str))
    if value not in values:
        shown = sorted(values)
        raise ParalomeError(                                   # PLM-0234
            234,
            f"--trajectory-root {column}:{value}: no cell has {column} = {value!r}. Its values: "
            f"{', '.join(shown[:40])}{' ...' if len(shown) > 40 else ''}",
        )


# --- Slingshot ---------------------------------------------------------------------------------------------------

COMMANDS_TABLE = "seurat_commands.tsv"


def _span(columns: Sequence[int]) -> str:
    """1..30 for a run of consecutive dimensions, else the list."""
    columns = list(columns)
    if columns == list(range(columns[0], columns[0] + len(columns))):
        return f"{columns[0]}:{columns[-1]}"
    return ",".join(map(str, columns))


def fit_columns(extraction_dir: Path, reduction: str, stored: int, drop_first: bool,
                dims: Optional[int]) -> Tuple[Optional[List[int]], str]:
    """The reduction's dimensions the lineages are fitted on (1-based, as the object numbers them), and where that
    came from.

    ``--trajectory-dims n``: the first n (after LSI's first). Else, the dims the object's RunUMAP on this reduction was run with (its
    FindNeighbors' when no RunUMAP is logged), from the extraction's seurat_commands.tsv (the object's command
    log). Else every dimension. ``None`` columns means every one (and LSI's first left out)."""
    available = stored - int(drop_first)
    if dims is not None:
        if not 2 <= int(dims) <= available:
            raise ParalomeError(                               # PLM-0124
                124, f"--trajectory-dims {dims}: the lineages are fitted on {reduction}"
                     f"{' without its first dimension' if drop_first else ''}, which has {available} dimensions; "
                     f"give 2 to {available}",
            )
        first = 2 if drop_first else 1
        columns = list(range(first, first + int(dims)))
        return columns, f"dimensions {_span(columns)} (--trajectory-dims {dims})"
    table = Path(extraction_dir) / COMMANDS_TABLE
    if table.is_file():
        logged = pd.read_csv(table, sep="\t", dtype=str, keep_default_na=False)
        for function in ("RunUMAP", "FindNeighbors"):
            rows = logged[(logged["function_name"] == function) & (logged["reduction"] == reduction)]
            for text in reversed(list(rows["dims"])):      # the object's latest
                try:
                    columns = [int(part) for part in text.split(",") if part.strip()]
                except ValueError:
                    continue
                if columns and min(columns) >= 1 and max(columns) <= stored and len(columns) >= 2:
                    return columns, f"dimensions {_span(columns)} (the object's {function} on {reduction})"
    every = f"{2 if drop_first else 1}:{stored}"
    why = ("the object logs no RunUMAP or FindNeighbors on it" if table.is_file() else
           "the extraction has no command log (extracted by an older Paralome, or the object kept none)")
    return None, f"every dimension, {every} ({why}; --trajectory-dims sets them)"


def run_slingshot(extraction_dir: Path, out: Path, *, column: str, root: str, modality: str,
                  rscript: str = "Rscript", echo: Callable[[str], None] = print,
                  fit_cells: Optional[Mapping[str, Any]] = None, dims: Optional[int] = None) -> Dict[str, Any]:
    settings = dict(fit_cells or {})
    if settings.get("reduction"):
        # the curves fitted in the reduction named, whatever its dimensions.
        reduction, reduction_path = named_reduction(extraction_dir, str(settings["reduction"]))
        drop_first = False
    else:
        reduction, reduction_path, drop_first = choose_reduction(extraction_dir, modality)
    available = _reduction_dims(reduction_path) - int(drop_first)
    columns, dims_from = fit_columns(extraction_dir, reduction, _reduction_dims(reduction_path), drop_first, dims)
    out.mkdir(parents=True, exist_ok=True)
    metadata = read_cell_metadata(extraction_dir)
    labels = out / "labels.tsv"
    with atomic_text_writer(labels, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["barcode", "cluster"])
        writer.writerows(zip(metadata["barcode"], metadata[column]))
    argv = [rscript, str(R_SCRIPT), "--reduction", str(reduction_path), "--labels", str(labels),
            "--root", root, "--out", str(out)]
    if drop_first:
        argv.append("--drop-first")
    if columns is not None:
        argv += ["--columns", ",".join(str(c) for c in columns)]
    if settings.get("cells"):
        argv += ["--fit-cells", str(int(settings["cells"])), "--fit-by", str(settings.get("dispersion") or "random")]
    if "stretch" in settings:
        argv += ["--stretch", repr(float(settings["stretch"]))]
    if settings.get("extend"):
        argv += ["--extend", str(settings["extend"])]
    if "df" in settings:
        argv += ["--df", repr(float(settings["df"]))]
    if "shrink" in settings:
        argv += ["--shrink", repr(float(settings["shrink"]))]
    for name, path in display_embeddings(extraction_dir):
        argv += ["--embedding", f"{name}={path}"]
    how = []
    if settings.get("cells"):
        how.append(f"curves fitted on {settings['cells']} cells per cluster ({settings.get('dispersion')})")
    if "stretch" in settings:
        how.append(f"stretch {settings['stretch']:g}")
    if settings.get("extend"):
        how.append(f"extend {settings['extend']}")
    if "df" in settings:
        how.append(f"df {settings['df']:g}")
    if "shrink" in settings:
        how.append(f"shrink {settings['shrink']:g}")
    echo(f"trajectories: Slingshot on {reduction}{' (without its first dimension)' if drop_first else ''}"
         f", {dims_from}, clusters {column}, root {root}{''.join('; ' + part for part in how)}")
    _run_r(argv, out, "slingshot", rscript, echo)
    try:
        recorded = json.loads((out / "slingshot.json").read_text())
    except (OSError, ValueError):
        recorded = {}
    return {"method": "slingshot", "reduction": reduction, "reduction_dims": available, "dims": recorded.get("dimensions", available),
            "columns": columns, "dims_from": dims_from,
            "drop_first_dimension": drop_first, "cluster_column": column, "root": root,
            "fit_cells_per_cluster": settings.get("cells"),
            "fit_by": settings.get("dispersion") if settings.get("cells") else None,
            "stretch": settings.get("stretch"), "extend": settings.get("extend"),
            "df": settings.get("df"), "shrink": settings.get("shrink"),
            "reduction_named": settings.get("reduction"), "fitted_cells": recorded.get("fitted_cells"),
            "cells": recorded.get("cells"),
            "embeddings": [name for name, _ in display_embeddings(extraction_dir)]}


def _run_r(argv: Sequence[str], out: Path, method: str, rscript: str, echo: Callable[[str], None]) -> None:
    """Run a method's R script, its output written to <method>.log as R writes it, its lines echoed."""
    said: List[str] = []
    log = out / f"{method}.log"
    with log.open("w") as handle:
        process = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            said.append(line.rstrip("\n"))
            if line.startswith("  "):
                echo(f"trajectories:{line.rstrip()[1:]}")
        process.wait()
    missing = [line.split()[-1] for line in said if line.startswith("PARALOME_MISSING_R_PACKAGE")]
    if missing:
        if method == "monocle3":
            raise ParalomeError(418, f"The R that runs {rscript} has no {missing[0]} package")   # PLM-0418
        raise ParalomeError(417, f"The R that runs {rscript} has no slingshot package")      # PLM-0417
    for line in said:
        if line.startswith("PARALOME_WELD_UNKNOWN"):
            _tag, weld, name, tips = line.split()
            raise ParalomeError(                               # PLM-0127
                127, f"--trajectory-weld {weld}: the learned graph has no tip {name}; its tips are L1-L{tips} "
                     f"({out / 'tips.tsv'})",
            )
        if line.startswith("PARALOME_TOO_MANY_ROUTES"):
            fields = line.split()
            across = [entry.split()[1:] for entry in said if entry.startswith("PARALOME_ROUTES_ACROSS")]
            raise ParalomeError(                               # PLM-0127
                127, f"--trajectory-weld: the welds make {fields[1]} routes from the root to its ends, more than "
                     f"{fields[2] if len(fields) > 2 else DEFAULT_MAX_ROUTES} (--trajectory-max-routes). The routes "
                     f"across each weld: " + ", ".join(f"{spec} {count}" for spec, count in across)
                     + ". Make the busiest one-way ('A->B' or A~B), join them in one weld, or raise "
                       "--trajectory-max-routes",
            )
    if process.returncode != 0:
        if method == "monocle3":
            why = (" (the root's part of the graph has no leaf: no lineage to follow)"
                   if any("PARALOME_NO_LEAVES" in line for line in said) else "")
            raise ParalomeError(                               # PLM-0314
                314, f"Monocle3 failed (exit {process.returncode}){why}; the last lines of {log}:\n"
                     + "\n".join(said[-15:]),
            )
        raise ParalomeError(                                   # PLM-0313
            313, f"Slingshot failed (exit {process.returncode}); the last lines of {log}:\n" + "\n".join(said[-15:]),
        )


def monocle3_embedding(extraction_dir: Path, name: Optional[str] = None) -> Tuple[str, Path]:
    """The 2D or 3D UMAP Monocle3 learns its graph in -- the one reduction=<name> names, else the
    object's UMAP (a 2D/3D reduction with umap in its name), else its only 2D/3D one."""
    shown = display_embeddings(extraction_dir)
    if name:
        stem, path = named_reduction(extraction_dir, name)
        if _reduction_dims(path) not in (2, 3):
            raise ParalomeError(                               # PLM-0126
                126, f"--trajectory-fit-cells reduction={name}: Monocle3 learns its graph in a 2D or 3D embedding "
                     f"(a UMAP); {stem} has {_reduction_dims(path)} dimensions. The object's 2D/3D ones: "
                     f"{', '.join(n for n, _ in shown) or 'none'}",
            )
        return stem, path
    umaps = [(n, p) for n, p in shown if "umap" in n.lower()]
    if umaps:
        return umaps[0]
    if len(shown) == 1:
        return shown[0]
    raise ParalomeError(                                       # PLM-0126
        126, f"Monocle3 learns its graph in the object's UMAP, and the object holds "
             f"{'no 2D or 3D embedding' if not shown else 'no UMAP but ' + ', '.join(n for n, _ in shown)}; "
             f"name one with --trajectory-fit-cells reduction=<name>, or --trajectory-method slingshot",
    )


def run_monocle3(extraction_dir: Path, out: Path, *, column: str, root: str, rscript: str = "Rscript",
                 echo: Callable[[str], None] = print, settings: Optional[Mapping[str, Any]] = None,
                 welds: Sequence[str] = (), reuse_graph: bool = False,
                 max_routes: Optional[int] = None) -> Dict[str, Any]:
    """The lineages with Monocle3 (r/monocle3_trajectories.R) -- one principal graph learned in the
    object's UMAP, every path from the root's node to a leaf a lineage -- written in the layout Slingshot's are."""
    settings = dict(settings or {})
    name, path = monocle3_embedding(extraction_dir, settings.get("reduction"))
    out.mkdir(parents=True, exist_ok=True)
    metadata = read_cell_metadata(extraction_dir)
    labels = out / "labels.tsv"
    with atomic_text_writer(labels, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["barcode", "cluster"])
        writer.writerows(zip(metadata["barcode"], metadata[column]))
    argv = [rscript, str(MONOCLE3_R_SCRIPT), "--reduction", str(path), "--name", name, "--labels", str(labels),
            "--root", root, "--out", str(out)]
    how = []
    for key in MONOCLE3_KEYS:
        if key in settings and key != "reduction":
            value = settings[key]
            argv += ["--" + key.replace("_", "-"), repr(value) if isinstance(value, float) else str(value)]
            how.append(f"{key} {value:g}" if isinstance(value, float) else f"{key} {value}")
    for weld in welds:
        argv += ["--weld", weld]
    if reuse_graph:
        argv += ["--reuse-graph", "y"]
    if max_routes is not None:
        argv += ["--max-routes", str(int(max_routes))]
    echo(f"trajectories: Monocle3 on {name} ({_reduction_dims(path)} dimensions), clusters {column}, root {root}"
         f"{''.join('; ' + part for part in how)}{'; welds ' + ' '.join(welds) if welds else ''}"
         f"{' (the learned graph kept: only the welds changed)' if reuse_graph else ''}")
    _run_r(argv, out, "monocle3", rscript, echo)
    try:
        recorded = json.loads((out / "monocle3.json").read_text())
    except (OSError, ValueError):
        recorded = {}
    return {"method": "monocle3", "reduction": name, "dims": recorded.get("dimensions", _reduction_dims(path)),
            "cluster_column": column, "root": root, "root_node": recorded.get("root_node"),
            "ncenter": settings.get("ncenter"), "minimal_branch_len": settings.get("minimal_branch_len"),
            "use_partition": settings.get("use_partition"), "reduction_named": settings.get("reduction"),
            **{key: settings.get(key) for key in MONOCLE3_KEYS[4:]},
            "graph_nodes": recorded.get("graph_nodes"), "partitions": recorded.get("partitions"),
            "tips": recorded.get("tips"), "welds": recorded.get("welds") or [],
            "graph_reused": recorded.get("graph_reused"),
            "cells": recorded.get("cells"), "cells_placed": recorded.get("cells_placed"),
            "fitted_cells": recorded.get("cells"), "embeddings": [name]}


def lineage_names(cells: pd.DataFrame) -> List[str]:
    return [c[len("pseudotime_"):] for c in cells.columns if c.startswith("pseudotime_")]


def assign_segments(cells: pd.DataFrame, bins: int) -> pd.DataFrame:
    """One row per cell per lineage it is on: barcode, lineage, pseudotime, bin (1..bins, equal cells by
    pseudotime) and its segment (trajectory_<lineage>_bin<n>)."""
    rows = []
    for lineage in lineage_names(cells):
        on = cells["branch"].fillna("").astype(str).str.split(",").apply(lambda parts: lineage in parts)
        sub = cells.loc[on, ["barcode", f"pseudotime_{lineage}"]].dropna()
        if sub.empty:
            continue
        rank = sub[f"pseudotime_{lineage}"].rank(method="first")
        bin_ = np.minimum(bins, np.floor((rank - 1) / len(sub) * bins).astype(int) + 1)
        rows.append(pd.DataFrame({
            "barcode": sub["barcode"].to_numpy(), "lineage": lineage,
            "pseudotime": sub[f"pseudotime_{lineage}"].to_numpy(), "bin": bin_.to_numpy(),
            "segment": [f"{SEGMENT_PREFIX}{lineage}_bin{b}" for b in bin_],
        }))
    if not rows:
        return pd.DataFrame(columns=["barcode", "lineage", "pseudotime", "bin", "segment"])
    return pd.concat(rows, ignore_index=True)


# --- the composition ------------------------------------------------------------------------------------------

def composition_tables(cells: pd.DataFrame, segments: pd.DataFrame, metadata: pd.DataFrame,
                       replicate_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(counts, shares): features x replicates. ``<lineage>_bin<n>``: the replicate's cells of that lineage in the
    segment, its share of the replicate's cells on the lineage. ``branch_<lineages>``: the replicate's fitted cells
    on that branch (L1, L1_L2 ...), its share of the replicate's fitted cells."""
    replicate = metadata.set_index("barcode")[replicate_column].astype(str)
    seg = segments.assign(replicate=replicate.reindex(segments["barcode"]).to_numpy())
    seg = seg[seg["replicate"].notna() & (seg["replicate"] != "")]
    seg = seg.assign(feature=seg["lineage"] + "_bin" + seg["bin"].astype(str))
    along = seg.groupby(["feature", "replicate"]).size().unstack(fill_value=0)
    along_totals = seg.groupby(["lineage", "replicate"]).size().unstack(fill_value=0)
    lineage_of = {feature: feature.split("_bin")[0] for feature in along.index}
    along_share = along.div(along_totals.loc[[lineage_of[f] for f in along.index]].to_numpy())
    fitted = cells[cells["branch"].fillna("") != ""].assign(
        replicate=replicate.reindex(cells["barcode"]).to_numpy())
    fitted = fitted[fitted["replicate"].notna() & (fitted["replicate"] != "")]
    fitted = fitted.assign(feature="branch_" + fitted["branch"].str.replace(",", "_", regex=False))
    between = fitted.groupby(["feature", "replicate"]).size().unstack(fill_value=0)
    between_share = between / between.sum(axis=0)
    counts = pd.concat([along, between]).fillna(0)
    shares = pd.concat([along_share, between_share]).fillna(0.0)
    return counts, shares[counts.columns]


def write_composition_inputs(counts: pd.DataFrame, shares: pd.DataFrame, sample_manifest: Path, out: Path, *,
                             replicate_column: str, pseudocount: float,
                             covariate_values: Optional[Mapping[str, float]] = None,
                             covariate_column: Optional[str] = None) -> Dict[str, Any]:
    """The composition as the differential reads a pseudobulk: cpm.csv (share x 1e6), log2_cpm.csv,
    raw_counts.csv (cells) and sample_manifest.csv (one row per replicate, its columns from the run's own)."""
    out.mkdir(parents=True, exist_ok=True)
    run_manifest = pd.read_csv(sample_manifest, dtype=str, keep_default_na=False)
    keep = [c for c in run_manifest.columns if c not in ("sample_id", "n_cells")]
    rows = []
    for replicate in counts.columns:
        match = run_manifest[run_manifest[replicate_column].astype(str) == str(replicate)] \
            if replicate_column in run_manifest.columns else run_manifest.iloc[0:0]
        if match.empty:
            continue
        # The replicate's own name: with --stratify-by the run's samples are replicate x stratum.
        row = {"sample_id": str(replicate), "n_cells": int(counts[replicate].sum())}
        if covariate_values is not None and covariate_column:
            row[covariate_column] = repr(float(covariate_values[str(replicate)]))
        for column in keep:
            values = set(match[column])
            if len(values) == 1:
                row[column] = next(iter(values))
        rows.append((replicate, row))
    if not rows:
        raise ParalomeError(                                   # PLM-0236
            236, "No replicate of the trajectories is one of the run's pseudobulk samples: nothing to compare",
        )
    replicates = [r for r, _ in rows]
    ids = [row["sample_id"] for _, row in rows]

    def write(path: Path, frame: pd.DataFrame) -> None:
        with atomic_text_writer(path, newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["feature_name", "gene_id", "common_id", *ids])
            for feature, values in frame[replicates].iterrows():
                writer.writerow([feature, MISSING, MISSING, *[repr(float(v)) for v in values]])

    cpm = shares * 1_000_000.0
    write(out / "raw_counts.csv", counts.astype(float))
    write(out / "cpm.csv", cpm)
    write(out / "log2_cpm.csv", np.log2(cpm + pseudocount))
    fields: List[str] = []
    for _, row in rows:
        fields += [k for k in row if k not in fields]
    with atomic_text_writer(out / "sample_manifest.csv", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for _, row in rows)
    return {"cpm": str(out / "cpm.csv"), "sample_manifest": str(out / "sample_manifest.csv"),
            "raw_counts": str(out / "raw_counts.csv"), "log2_cpm": str(out / "log2_cpm.csv"),
            "features": len(counts), "samples": len(rows)}


def replicate_gene_covariate(extraction_dir: Path, gene: str, replicate_column: str, pseudocount: float,
                             gtf: Optional[Path] = None) -> Dict[str, float]:
    """The gene's log2(CPM + pseudocount) in each replicate's whole-object pseudobulk (every cell of the replicate,
    across every stratum): the --covariate-feature of the composition's lm. The gene is matched as the differential matches it: by any of its identifiers."""
    from paralome.rna.singlecell import _with_gene_identity_from, load_singlecell_rna_inputs

    inputs = load_singlecell_rna_inputs(extraction_dir)
    if gtf is not None:
        inputs = _with_gene_identity_from(inputs, gtf)
    rows = [index for index, entry in enumerate(inputs.genes)
            if gene in {str(getattr(entry, name, "")) for name in ("feature_name", "gene_id", "common_id")}]
    if len(rows) != 1:
        raise ParalomeError(                                   # PLM-0239
            239, f"--covariate-feature {gene!r} names {'no' if not rows else 'more than one'} gene of the object",
        )
    metadata = inputs.metadata
    barcodes = metadata["barcode"] if "barcode" in metadata.columns else pd.Series(metadata.index, index=metadata.index)
    replicate = pd.Series(metadata[replicate_column].astype(str).to_numpy(), index=barcodes.astype(str).to_numpy())
    of_cell = replicate.reindex(list(inputs.cells)).to_numpy()
    library = np.asarray(inputs.matrix.sum(axis=0)).ravel()
    counts = np.asarray(inputs.matrix[rows[0]].todense()).ravel()
    values: Dict[str, float] = {}
    for name in sorted({str(r) for r in of_cell if isinstance(r, str) and r}):
        mask = of_cell == name
        total = float(library[mask].sum())
        values[name] = float(np.log2((counts[mask].sum() / total * 1_000_000.0 if total > 0 else 0.0) + pseudocount))
    return values


def composition_statistic(statistic: str) -> str:
    """The statistic the composition is tested with: the run's --statistic, each entry at the replicate level.

    A share is a replicate's, not a cell's, so ``anova:cell`` is ``anova`` there (a run whose
    statistics are all :cell, as a 2x2x2x2 study's may be, still has its composition tested by its own model).
    """
    entries: List[str] = []
    for entry in (part.strip() for part in str(statistic).split(",")):
        if not entry:
            continue
        name = entry[: -len(":cell")] if entry.lower().endswith(":cell") else entry
        if name not in entries:
            entries.append(name)
    if not entries:
        raise ParalomeError(238, f"--statistic {statistic!r} names no statistic")   # PLM-0238
    return ",".join(entries)


# --- the stage ---------------------------------------------------------------------------------------------------

def _identity(path: Path) -> Dict[str, Any]:
    try:
        stat = Path(path).stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(path), "missing": True}


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _zip_tree(zip_path: Path, members: Sequence[Tuple[Path, str]], manifest: Mapping[str, Any], *,
              jobs: int = 1) -> None:
    """The members compressed on ``jobs`` CPUs at once (paralome.zipping): the zip can be tens of GB."""
    from paralome.zipping import write_zip

    write_zip(zip_path, members, texts=[("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")],
              jobs=jobs)


def _result_files(folder: Path) -> List[Path]:
    """The differential's output an agent or OmicView reads: its tables and manifests, not its scripts or logs."""
    skip = {"scripts", "logs", "checkpoints", "pseudobulk", "tmp"}
    found = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or skip & set(path.relative_to(folder).parts):
            continue
        if path.suffix.lower() in (".tsv", ".csv", ".json", ".txt", ".md", ".parquet", ".gz", ".png", ".html"):
            found.append(path)
    return found


def trajectories_to_cell_export(extraction_dir: Path, fit_dir: Path, *, root: str = "",
                                echo: Callable[[str], None] = print,
                                export_dir: Optional[Path] = None) -> Optional[Path]:
    """The fit written into the object's cell export (<omicview>/<rds>_cell_expression_embeddings.zip,
    where the extraction put it): pseudotime columns in obs and the curves in uns, for OmicView's Cells viewer."""
    from paralome.singlecell.cell_export import (add_trajectories, build_cell_export, export_home, export_zip_path,
                                                 rds_stem)

    record = stage_checkpoint_path(Path(extraction_dir), "singlecell_extract")
    try:
        rds = json.loads(record.read_text()).get("rds")
    except (OSError, ValueError):
        rds = None
    if not rds or not (Path(fit_dir) / "cells.tsv").is_file():
        echo("trajectories: no cell export to add the lineages to (the extraction names no object)")
        return None
    # the run's omicview folder
    # (export_dir: the parent of the stage's zip folder), with the run's other outputs; the object's own export
    # (<parent>/omicview, the extraction's) is copied there and left as it is.
    shared = export_zip_path(export_home(Path(extraction_dir).parent), rds_stem(rds))
    target = export_zip_path(export_dir, rds_stem(rds)) if export_dir else shared
    if not target.is_file() and target != shared and shared.is_file():
        echo(f"trajectories: the cell export copied into the run's folder: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        shutil.copyfile(shared, part)
        os.replace(part, target)
    if not target.is_file():
        # An export can sit where an older extraction wrote it (<parent>/data/), and a run that only reuses
        # the extraction never writes one in <parent>/omicview/. The stage writes it from the extraction (never
        # re-extracting the object), then adds the lineages.
        echo(f"trajectories: no cell export at {target}; writing it from the extraction")
        target = build_cell_export(Path(extraction_dir), target.parent, rds=rds, log=echo)
        if target is None or not Path(target).is_file():
            echo("trajectories: the extraction gave no cell export to add the lineages to")
            return None
    if add_trajectories(target, Path(fit_dir), root=root, log=echo):
        return target
    echo(f"trajectories: {target.name} already carries this fit")
    return target


def _export_record(target: Optional[Path], fit_dir: Path) -> Dict[str, Any]:
    """What the checkpoint keeps of the cell export: where it is and the fit it carries (None: there is none)."""
    from paralome.singlecell.cell_export import trajectory_fingerprint

    return {"zip": str(target) if target else None, "fit_sha256": trajectory_fingerprint(Path(fit_dir))}


# --- the fit ------------------------------------------------------------------------------------------
# The trajectories run as a subfunction of the extraction, serially after it, so the cell export and the
# trajectories are free while the rest works, and Paralome can be run in a loop over trials with the number
# of cells altered without touching the rest of the pipeline. The singlecell_rna stage fits the lineages after
# its extraction and pseudobulk (run_trajectory_fit) and writes them into the run's cell export; the
# trajectories stage then reads the fit (run_trajectories(fit_dir=)) for its segments and statistics. The fit
# keeps or redoes itself on what it is made from: the root, --trajectory-fit-cells, --trajectory-dims, the
# extraction's files and the R script.
FIT_RECORD = "trajectory_fit"
FIT_METHOD = "trajectory_fit.v1"
# What a fit is made from, as an older trajectories stage's request recorded it (that stage fitted too): a
# fit it made is taken as it is when these agree.
_FIT_FIELDS = ("root", "modality", "fit_cells", "dims", "seurat_commands", "cell_metadata", "reductions", "r_script")


def check_fit_options(method: Optional[str], fit_cells: Optional[str], dims: Optional[int],
                      welds: Optional[Sequence[str]] = None, max_routes: Optional[int] = None) -> str:
    """The method, its --trajectory-fit-cells and --trajectory-dims read together (and
    --trajectory-weld; --trajectory-max-routes); the method returned."""
    method = parse_method(method)
    parse_fit_cells(fit_cells, method)
    parse_welds(welds, method)
    if max_routes is not None:
        if method != "monocle3":
            raise ParalomeError(                               # PLM-0125
                125, "--trajectory-max-routes limits the routes of Monocle3's graph; Slingshot has none. Leave it "
                     "out, or use --trajectory-method monocle3",
            )
        if int(max_routes) < 1:
            raise ParalomeError(127, f"--trajectory-max-routes {max_routes}: give 1 or more (default "   # PLM-0127
                                     f"{DEFAULT_MAX_ROUTES})")
    if method == "monocle3" and dims is not None:
        raise ParalomeError(                                   # PLM-0125
            125, f"--trajectory-dims {dims} chooses the dimensions Slingshot fits on; Monocle3 learns its graph in "
                 f"the object's UMAP. Leave it out, or add --trajectory-method slingshot",
        )
    return method


def fit_request(extraction_dir: Path, *, root: str, modality: str = "scRNA", fit_cells: Optional[str] = None,
                dims: Optional[int] = None, method: Optional[str] = None,
                welds: Optional[Sequence[str]] = None, max_routes: Optional[int] = None) -> Dict[str, Any]:
    column, value = parse_root(root)
    method = check_fit_options(method, fit_cells, dims, welds, max_routes)
    fit = parse_fit_cells(fit_cells, method)
    extraction_dir = Path(extraction_dir)
    return {
        "method": FIT_METHOD, "trajectory_method": method, "root": [column, value], "modality": modality,
        "welds": parse_welds(welds, method), "max_routes": int(max_routes) if max_routes is not None else None,
        "fit_cells": fit_cells_record(fit), "dims": dims,
        "seurat_commands": _identity(extraction_dir / COMMANDS_TABLE),
        "cell_metadata": _identity(extraction_dir / "cell_metadata.tsv"),
        "reductions": sorted(str(p.name) for p in (extraction_dir / "reductions").glob("*.tsv"))
        if (extraction_dir / "reductions").is_dir() else [],
        "r_script": hashlib.sha256(R_SCRIPTS[method].read_bytes()).hexdigest(),
        # The part of the script that learns the graph -- a change elsewhere in it keeps graph.rds.
        "graph_script": _graph_section_digest(R_SCRIPTS[method]) if method == "monocle3" else None,
    }


def _graph_section_digest(script: Path) -> str:
    """The checksum of the lines between the R script's marks around what learns the graph."""
    text = script.read_text()
    start, end = text.index("# >>> the learned graph"), text.index("# <<< the learned graph")
    return hashlib.sha256(text[start:end].encode()).hexdigest()


def _earlier_stage_fit(fit_dir: Path, request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The fit the trajectories stage made in this folder, when it was made from the same things."""
    try:
        earlier = json.loads(stage_checkpoint_path(Path(fit_dir).parent, STAGE).read_text())
    except (OSError, ValueError):
        return None
    asked = earlier.get("request") or {}
    if request.get("trajectory_method") != "slingshot":
        return None                         # the trajectories stage fitted with Slingshot only
    if not (earlier.get("complete") and isinstance(earlier.get("fit"), Mapping)
            and (Path(fit_dir) / "cells.tsv").is_file()):
        return None
    if any(asked.get(name) != request.get(name) for name in _FIT_FIELDS):
        return None
    # An older stage fitted with Slingshot and names no method; a newer one names the method the fit it read
    # was made with, which is not a fit of the stage's own.
    if asked.get("trajectory_method", "slingshot") != "slingshot" or "trajectory_method" in asked:
        return None
    return dict(earlier["fit"])


def run_trajectory_fit(extraction_dir: Path, fit_dir: Path, *, root: str, modality: str = "scRNA",
                       fit_cells: Optional[str] = None, dims: Optional[int] = None, rscript: str = "Rscript",
                       export_dir: Optional[Path] = None, overwrite: bool = False,
                       echo: Callable[[str], None] = print, method: Optional[str] = None,
                       welds: Optional[Sequence[str]] = None, max_routes: Optional[int] = None) -> Dict[str, Any]:
    """Fit the object's lineages into ``fit_dir`` (kept when nothing it is made from changed) and write them into the
    run's cell export (``export_dir``). Returns the fit's record. When only ``welds`` changed, Monocle3's
    learned graph is kept and only the welds, the cells' order and the lineages are made again."""
    from paralome.singlecell.cell_export import trajectory_fingerprint
    from paralome.version import __version__

    if modality != "scRNA":
        raise ParalomeError(                                   # PLM-0237
            237, f"The trajectories stage runs on scRNA objects in this version; {modality} is not yet built",
        )
    extraction_dir, fit_dir = Path(extraction_dir), Path(fit_dir)
    column, value = parse_root(root)
    check_root(read_cell_metadata(extraction_dir), column, value)
    method = check_fit_options(method, fit_cells, dims, welds, max_routes)
    request = fit_request(extraction_dir, root=root, modality=modality, fit_cells=fit_cells, dims=dims, method=method,
                          welds=welds, max_routes=max_routes)
    request_id = _digest(request)
    record_path = stage_checkpoint_path(fit_dir, FIT_RECORD)
    fitted: Optional[Dict[str, Any]] = None
    try:
        existing = json.loads(record_path.read_text())
    except (OSError, ValueError):
        existing = {}
    earlier = existing.get("request") if isinstance(existing.get("request"), Mapping) else {}
    # The graph Monocle3 learned is kept when nothing but the welds changed. Whether or not
    # that fit then finished -- a fit refused for its welds (too many routes) still learned and saved its graph.
    # graph.rds is there only as the last fit left it: one that learns a graph removes the old one first.
    reuse_graph = (not overwrite and method == "monocle3"
                   and (fit_dir / "graph.rds").is_file()
                   and {k: v for k, v in earlier.items() if k not in _NOT_THE_GRAPH}
                   == {k: v for k, v in request.items() if k not in _NOT_THE_GRAPH})
    if not overwrite:
        if existing.get("complete") and existing.get("request_id") == request_id and (fit_dir / "cells.tsv").is_file():
            fitted = dict(existing.get("fit") or {})
            echo(f"trajectories: kept the fit in {fit_dir} (nothing it was made from has changed)")
        else:
            fitted = _earlier_stage_fit(fit_dir, request)
            if fitted is not None:
                echo(f"trajectories: kept the fit the trajectories stage made in {fit_dir} (made from the same)")
    if fitted is None:
        atomic_write_text(record_path, json.dumps({"stage": FIT_RECORD, "method": FIT_METHOD, "complete": False,
                                                   "request": request, "request_id": request_id,
                                                   "updated": now_iso()}, indent=2, sort_keys=True) + "\n")
        # One fit folder, whichever method made it -- the other method's files go first, so what is in
        # it is the fit this record describes.
        for stale in [*(fit_dir / name for name in FIT_OUTPUTS), *fit_dir.glob("curve_*.tsv")]:
            if stale.is_file() and not (reuse_graph and stale.name in GRAPH_FILES):
                stale.unlink()
        if method == "monocle3":
            fitted = run_monocle3(extraction_dir, fit_dir, column=column, root=value, rscript=rscript, echo=echo,
                                  settings=parse_fit_cells(fit_cells, method), welds=request["welds"],
                                  reuse_graph=reuse_graph, max_routes=request["max_routes"])
        else:
            fitted = run_slingshot(extraction_dir, fit_dir, column=column, root=value, modality=modality,
                                   rscript=rscript, echo=echo, fit_cells=parse_fit_cells(fit_cells), dims=dims)
    record: Dict[str, Any] = {
        "stage": FIT_RECORD, "method": FIT_METHOD, "complete": True, "paralome_version": __version__,
        "updated": now_iso(), "request": request, "request_id": request_id, "fit": fitted,
        "fit_dir": str(fit_dir), "fit_sha256": trajectory_fingerprint(fit_dir),
    }
    record["cell_export"] = _export_record(
        trajectories_to_cell_export(extraction_dir, fit_dir, root=root, echo=echo, export_dir=export_dir), fit_dir)
    atomic_write_text(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def fit_settings_asked(root: Optional[str], fit_cells: Optional[str] = None, dims: Optional[int] = None,
                       method: Optional[str] = None, welds: Optional[Sequence[str]] = None,
                       max_routes: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The fit's own settings a run asks for, as the fit's record holds them (fit_request) -- or None
    when there is no root, or they cannot be read (the run refuses them elsewhere, before anything is queued)."""
    if not root:
        return None
    try:
        method = check_fit_options(method, fit_cells, dims, welds, max_routes)
        return {"root": list(parse_root(root)), "trajectory_method": method,
                "fit_cells": fit_cells_record(parse_fit_cells(fit_cells, method)), "dims": dims,
                "welds": parse_welds(welds, method),
                "max_routes": int(max_routes) if max_routes is not None else None}
    except ParalomeError:
        return None


# What a fit's record held for these settings before they were recorded (Slingshot, no welds).
_FIT_SETTING_DEFAULTS = {"trajectory_method": "slingshot", "welds": [], "max_routes": None}


def fit_unfinished(fit_dir: Path, export_dir: Optional[Path] = None,
                   asked: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Why singlecell_rna's trajectory fit is to run again (the part only), or None.

    The fit is not there (a run fitted by the trajectories stage, or never), or the cell export it
    wrote is not in the run's omicview folder, or no longer carries it (written again since); or the run
    asks for other settings (``asked``: fit_settings_asked) than the fit's own record says it was made with --
    read from the fit itself, not from the stage's request record, which a run of another version takes over
    without the work having been done."""
    fit_dir = Path(fit_dir)
    try:
        record = json.loads(stage_checkpoint_path(fit_dir, FIT_RECORD).read_text())
    except (OSError, ValueError):
        record = {}
    if not record.get("complete") or not (fit_dir / "cells.tsv").is_file():
        return "the trajectories are fitted by singlecell_rna , and there is no fit of its own yet"
    if asked:
        made = record.get("request") if isinstance(record.get("request"), Mapping) else {}
        differ = [key for key, value in asked.items()
                  if json.loads(json.dumps(value)) != made.get(key, _FIT_SETTING_DEFAULTS.get(key))]
        if differ:
            return "this run asks for other " + ", ".join(
                f"{key} ({made.get(key, _FIT_SETTING_DEFAULTS.get(key))} -> {json.loads(json.dumps(asked[key]))})"
                for key in differ) + " than the fit was made with"
    export = record.get("cell_export") if isinstance(record.get("cell_export"), Mapping) else {}
    if not export.get("zip"):
        return None                         # no export could be made (the stage said why); not asked again
    target = Path(str(export["zip"]))
    if export_dir is not None and target.parent.resolve() != Path(export_dir).resolve():
        return f"its cell export is to be in the run's folder {export_dir}, not {target.parent}"
    try:
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read(f"{target.stem}_manifest.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return f"the cell export it wrote its lineages into is not there or not readable ({target})"
    if (manifest.get("trajectories") or {}).get("fit_sha256") != record.get("fit_sha256"):
        return f"the cell export no longer carries its lineages (written again since: {target.name})"
    return None


def read_trajectory_fit(fit_dir: Path) -> Dict[str, Any]:
    """The fit's record, refused when there is none (the singlecell_rna stage fits; PLM-0239)."""
    fit_dir = Path(fit_dir)
    try:
        record = json.loads(stage_checkpoint_path(fit_dir, FIT_RECORD).read_text())
    except (OSError, ValueError):
        record = {}
    if not record.get("complete") or not (fit_dir / "cells.tsv").is_file():
        raise ParalomeError(                                    # PLM-0244
            244, f"There is no trajectory fit in {fit_dir}: the singlecell_rna stage fits the lineages "
                 "(with --trajectory-root) before the trajectories stage runs",
            fix="run the pipeline again so singlecell_rna fits them, or name the fit's folder with --fit",
        )
    return record


def run_trajectories(
    extraction_dir: Path, sample_manifest: Path, workdir: Path, *, root: str, prefix: str, modality: str = "scRNA",
    replicate_column: str, statistic: str, factors: Sequence[str] = (), pair_by: Optional[str] = None,
    covariate: Optional[str] = None, covariate_feature: Optional[str] = None, remove: Sequence[str] = (),
    min_cells_per_group: int = 10, gtf: Optional[Path] = None, bins: int = DEFAULT_BINS, alpha: str = "P0.05",
    pseudocount: float = 1.0, min_mean_cpm: float = 1.0, jobs: int = 1, rscript: str = "Rscript",
    zip_dir: Optional[Path] = None, overwrite: bool = False, echo: Callable[[str], None] = print,
    fit_cells: Optional[str] = None, dims: Optional[int] = None, fit_dir: Optional[Path] = None,
    method: Optional[str] = None, model: Optional[str] = None, lineage_set: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare and zip the fit in ``fit_dir`` (<workdir>/fit; the singlecell_rna stage made it);
    returns the stage checkpoint (kept when nothing it was made from changed). ``method``: the run's
    --trajectory-method (not given, the method the fit was made with). ``model``: --trajectory-model
    (ls, the default, or monocle3; ``<engine>:df=<n>`` the spline's df); ``lineage_set``:
    --trajectory-lineages (fate, the default, exit or route)."""
    from paralome.singlecell.cell_export import trajectory_fingerprint

    from paralome.rna.singlecell_stats import run_singlecell_rna_differential
    from paralome.commands.atac import _parse_alpha
    from paralome.singlecell import trajectory_genes
    from paralome.version import __version__

    if modality != "scRNA":
        raise ParalomeError(                                   # PLM-0237
            237, f"The trajectories stage runs on scRNA objects in this version; {modality} is not yet built",
        )
    if bins < 2:
        raise ParalomeError(122, f"--trajectory-bins must be 2 or more (given: {bins})")   # PLM-0122
    extraction_dir, workdir = Path(extraction_dir), Path(workdir)
    column, value = parse_root(root)
    method = parse_method(method) if method else fit_method(Path(fit_dir) if fit_dir else workdir / "fit")
    fit = parse_fit_cells(fit_cells, method)
    metadata = read_cell_metadata(extraction_dir)
    check_root(metadata, column, value)
    kept_statistic = ",".join(part.strip() for part in str(statistic).split(",") if part.strip())
    shares_statistic = composition_statistic(statistic)
    engine, spline_df = trajectory_genes.parse_model(model)
    lineage_set = trajectory_genes.parse_lineage_set(lineage_set)
    out_zip = (Path(zip_dir) if zip_dir else workdir / "results") / zip_name(prefix, modality)
    request = {
        "method": METHOD, "trajectory_method": method, "root": [column, value], "bins": bins, "modality": modality,
        "fit_cells": fit_cells_record(fit), "dims": dims,
        "seurat_commands": _identity(extraction_dir / COMMANDS_TABLE),
        "replicate_column": replicate_column, "statistic": kept_statistic, "composition_statistic": shares_statistic,
        "factors": list(factors),
        "pair_by": pair_by, "covariate": covariate, "covariate_feature": covariate_feature,
        "remove": sorted(map(str, remove)), "min_cells_per_group": min_cells_per_group,
        "gtf": str(gtf) if gtf else None, "alpha": alpha, "pseudocount": pseudocount, "min_mean_cpm": min_mean_cpm,
        "cell_metadata": _identity(extraction_dir / "cell_metadata.tsv"),
        "reductions": sorted(str(p.name) for p in (extraction_dir / "reductions").glob("*.tsv"))
        if (extraction_dir / "reductions").is_dir() else [],
        "sample_manifest": _identity(Path(sample_manifest)),
        "r_script": hashlib.sha256(R_SCRIPT.read_bytes()).hexdigest(),
        # The gene model along pseudotime, and the lineages the statistics follow.
        "gene_model": [engine, spline_df], "lineage_set": lineage_set,
        "gene_model_r_script": hashlib.sha256(trajectory_genes.GENE_MODELS_R_SCRIPT.read_bytes()).hexdigest()
        if engine == "monocle3" else None,
    }
    request_id = _digest(request)
    checkpoint_path = stage_checkpoint_path(workdir, STAGE)
    fit_dir = Path(fit_dir) if fit_dir else workdir / "fit"
    fit_record = read_trajectory_fit(fit_dir)
    fit_sha = fit_record.get("fit_sha256") or trajectory_fingerprint(fit_dir)
    if checkpoint_path.is_file() and not overwrite:
        try:
            existing = json.loads(checkpoint_path.read_text())
        except (OSError, ValueError):
            existing = {}
        recorded_zip = Path(str(existing.get("zip") or ""))
        # And made from this fit (an older checkpoint names none: its own fit is the one in the folder, which
        # the fit took over when made from the same).
        if existing.get("complete") and existing.get("request_id") == request_id and existing.get(
                "fit_sha256", fit_sha) == fit_sha and (out_zip.is_file() or recorded_zip.is_file()):
            if not out_zip.is_file() and recorded_zip.is_file() and _within(recorded_zip, workdir):
                out_zip.parent.mkdir(parents=True, exist_ok=True)
                os.replace(recorded_zip, out_zip)
                existing["zip"] = str(out_zip)
                atomic_write_text(checkpoint_path, json.dumps(existing, indent=2, sort_keys=True) + "\n")
            echo(f"trajectories: kept {existing.get('zip')} (nothing it was made from has changed)")
            existing["fit_sha256"] = fit_sha
            atomic_write_text(checkpoint_path, json.dumps(existing, indent=2, sort_keys=True) + "\n")
            return existing
    checkpoint: Dict[str, Any] = {"schema": SCHEMA, "stage": STAGE, "method": METHOD, "complete": False,
                                  "paralome_version": __version__, "updated": now_iso(), "request": request,
                                  "request_id": request_id, "fit_sha256": fit_sha}
    atomic_write_text(checkpoint_path, json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")

    # The singlecell_rna stage fitted the lineages and wrote them into the cell export.
    fitted = dict(fit_record.get("fit") or {})
    cells = pd.read_csv(fit_dir / "cells.tsv", sep="\t", dtype={"barcode": str, "branch": str},
                        keep_default_na=False, na_values=[""])
    cells["branch"] = cells["branch"].fillna("")
    routes = pd.read_csv(fit_dir / "lineages.tsv", sep="\t", dtype=str, keep_default_na=False)
    # --trajectory-lineages: the
    # statistics follow one lineage per end state (fate), per way out of the root and end state (exit), or every
    # route (route).
    cells, lineages = trajectory_genes.group_lineages(cells, routes, lineage_set, value)
    lineages.to_csv(workdir / "lineages.tsv", sep="\t", index=False)
    segments = assign_segments(cells, bins)
    segments.to_csv(workdir / "segments.tsv", sep="\t", index=False)
    echo(f"trajectories: {len(lineages)} lineage(s) by {lineage_set} from the fit's {len(routes)} route(s), "
         f"{bins} composition segments each, {segments['barcode'].nunique()} cells")

    alpha_metric, alpha_value = _parse_alpha(alpha)
    common = dict(statistic=kept_statistic, factors=list(factors), pair_by=pair_by, covariate=covariate,
                  covariate_feature=covariate_feature, alpha=alpha_value, alpha_metric=alpha_metric,
                  pseudocount=pseudocount, min_mean_cpm=min_mean_cpm, overwrite=True, jobs=jobs)

    # The composition first -- it takes seconds, and a design it cannot test is said before the hours
    # of the expression statistics, not after them.
    counts, shares = composition_tables(cells, segments, metadata, replicate_column)
    # Lm on a gene (--covariate-feature) regresses the shares on that gene's expression in each
    # replicate (its whole-object pseudobulk), carried as a column of the composition's sample manifest.
    shares_covariate, gene_values = covariate, None
    if covariate_feature:
        shares_covariate = "covariate_feature_" + "".join(c if c.isalnum() else "_" for c in covariate_feature)
        gene_values = replicate_gene_covariate(extraction_dir, covariate_feature, replicate_column, pseudocount, gtf)
    composition_inputs = write_composition_inputs(
        counts, shares, Path(sample_manifest), workdir / "composition" / "pseudobulk",
        replicate_column=replicate_column, pseudocount=pseudocount,
        covariate_values=gene_values, covariate_column=shares_covariate if gene_values is not None else None)
    echo(f"trajectories: composition -- {composition_inputs['features']} shares x "
         f"{composition_inputs['samples']} replicates")
    composition_runs, composition_not_run = composition_statistics(
        composition_inputs["sample_manifest"], shares_statistic, factors=factors, pair_by=pair_by,
        covariate=shares_covariate)
    for folder, why in sorted(composition_not_run.items()):
        echo(f"trajectories: composition {folder} not run -- {why}")
    composition: Mapping[str, Any] = {}
    if composition_runs:
        composition = run_singlecell_rna_differential(
            composition_inputs["cpm"], composition_inputs["sample_manifest"], extraction_dir, workdir / "composition",
            file_prefix=f"{prefix}_trajectory_composition",
            **{**common, "statistic": composition_runs, "min_mean_cpm": 0.0, "covariate": shares_covariate,
               "covariate_feature": None})

    # The question is how genes change across trajectories by treatment (the per-bin assay is covered by the
    # main design): each gene along each lineage's pseudotime by the run's full factorial
    # (paralome.singlecell.trajectory_genes).
    genes = trajectory_genes.run_gene_models(
        extraction_dir, cells, lineages, Path(sample_manifest), workdir / "expression", prefix=prefix,
        factors=factors, replicate_column=replicate_column, remove=remove, gtf=gtf, engine=engine, df=spline_df,
        min_mean_cpm=min_mean_cpm, pseudocount=pseudocount, alpha=(alpha_metric, alpha_value), jobs=jobs,
        rscript=rscript, echo=echo)

    members: List[Tuple[Path, str]] = [(fit_dir / "cells.tsv", "fit/cells.tsv"),
                                       (fit_dir / "lineages.tsv", "fit/lineages.tsv"),
                                       (fit_dir / "mst.tsv", "fit/mst.tsv"),
                                       (fit_dir / f"{fit_method(fit_dir)}.json", f"fit/{fit_method(fit_dir)}.json"),
                                       (workdir / "lineages.tsv", "lineages.tsv"),
                                       (workdir / "segments.tsv", "composition/segments.tsv")]
    members += [(path, f"fit/{path.name}") for path in sorted(fit_dir.glob("curve_*.tsv"))]
    if (fit_dir / "tips.tsv").is_file():                 # Monocle3's learned tips, and the welds made
        members.append((fit_dir / "tips.tsv", "fit/tips.tsv"))
    for name in ("raw_counts.csv", "cpm.csv", "sample_manifest.csv"):
        members.append((workdir / "composition" / "pseudobulk" / name, f"composition/inputs/{name}"))
    for path in _result_files(workdir / "composition"):
        members.append((path, f"composition/{path.relative_to(workdir / 'composition')}"))
    expression_dir = workdir / "expression"
    members += [(path, f"expression/{path.name}") for path in sorted((expression_dir / engine).glob("*.tsv"))]
    manifest = {
        "schema": SCHEMA, "paralome_version": __version__, "created": now_iso(), "modality": modality,
        "what": ("One single-cell object's lineages (fitted on the cells of every condition together: by Monocle3 -- "
                 "one principal graph learned in the object's UMAP, each lineage a path from the root to a leaf of "
                 "the graph, a cell on every lineage its graph node lies on -- or by Slingshot -- every cell, or with "
                 "--trajectory-fit-cells a number per cluster, every cell then placed on the curves; fit.method says "
                 "which) "
                 "and how the conditions differ along and between them. fit/: each cell's pseudotime and weight on "
                 "every route of the fit and its branch (L1,L2 = before the routes part), the clusters along each "
                 "route, the graph, and the curves drawn in each embedding. lineages.tsv: the lineages the statistics "
                 "follow (lineage_set: fate, one per end state; exit, one per way out of the root and end state; "
                 "route, every route), the routes each holds, its exit and end. expression/: each gene along each "
                 "lineage's pseudotime, by the run's factors (gene_model): the cells' expression modelled as the full "
                 "factorial of the factors crossed with a natural spline of pseudotime; per term (each factorial "
                 "term; pseudotime: the gene changes along the lineage; <term>:pseudotime: the factors change how it "
                 "moves along it) its statistic (F by least squares, ls; LLR by Monocle3's fit_models, monocle3), "
                 "df, p and BH q, one table per lineage (<prefix>_trajectory_<lineage>_pseudotime_<engine>.tsv) and a "
                 "summary of every lineage and term. composition/: the run's statistics, at the replicate level, on "
                 "each replicate's share of its lineage's cells in each segment (<lineage>_bin<n>: shifts along "
                 "pseudotime; segments.tsv: each cell's segment) and of its fitted cells on each branch "
                 "(branch_<lineages>: shifts between lineages)."),
        "fit": {"method": fit_method(fit_dir), **fitted, "routes": routes.to_dict(orient="records"),
                # The cells the curves were fitted on, and every cell placed on them.
                "cells_fitted": int(fitted.get("fitted_cells") or len(cells)), "cells_placed": int(len(cells)),
                fit_method(fit_dir): json.loads((fit_dir / f"{fit_method(fit_dir)}.json").read_text())},
        "lineage_set": lineage_set, "lineages": lineages.to_dict(orient="records"),
        "segments": {"bins": bins, "rule": "equal numbers of a lineage's cells, by pseudotime, bin1 nearest the root",
                     "column": SEGMENT_COLUMN, "used_by": "composition"},
        "gene_model": {"engine": engine, "spline_df": spline_df, "factors": list(factors),
                       "data": f"each cell's log2(CPM + {pseudocount}), its whole library's CPM",
                       "genes_tested": f"mean CPM over the replicates (each one's cells on the lineage pooled) of "
                                       f"at least {min_mean_cpm}",
                       "lineages": genes["lineages"]},
        "statistics": {"statistic": kept_statistic, "composition_statistic": shares_statistic,
                       "composition_covariate": (f"{covariate_feature}: its log2(CPM+{pseudocount}) in each "
                                                 "replicate's whole-object pseudobulk") if covariate_feature
                       else covariate,
                       "factors": list(factors),
                       "replicate_column": replicate_column, "alpha": alpha,
                       "composition_folders": sorted(_folders(composition)),
                       # A composition statistic the replicates cannot support, and why.
                       "composition_not_run": composition_not_run},
    }
    started = time.monotonic()
    total = sum(path.stat().st_size for path, _name in members)
    echo(f"trajectories: zipping {len(members):,} files ({total / 1e9:.1f} GB) on {max(1, int(jobs or 1))} CPU(s)")
    _zip_tree(out_zip, members, manifest, jobs=jobs)
    echo(f"trajectories: wrote {out_zip} ({out_zip.stat().st_size / 1e9:.1f} GB) in "
         f"{time.monotonic() - started:.0f} s")
    checkpoint.update(complete=True, updated=now_iso(), zip=str(out_zip), lineages=len(lineages),
                      lineage_set=lineage_set, gene_model=genes,
                      composition_checkpoint=str(stage_checkpoint_path(workdir / "composition",
                                                                       "singlecell_rna_differential")),
                      fit=fitted, fit_dir=str(fit_dir), fit_sha256=fit_sha)
    if zip_dir:
        checkpoint["zip_dir"] = str(zip_dir)
    atomic_write_text(checkpoint_path, json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    return checkpoint


def composition_statistics(sample_manifest: Path, statistic: str, *, factors: Sequence[str],
                           pair_by: Optional[str], covariate: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """``(the composition's --statistic its replicates support, {folder: why not} for the rest)``.

    The shares are one value per replicate; a design with one replicate in every combination of the
    factors leaves the full-interaction anova no error to test against. The shared planner decides, as it does
    for every stratum, before the hours of the expression statistics; what it refuses is recorded."""
    from paralome.singlecell.stats_core import parse_request, plan_each_statistic
    from paralome.stats import statistic_folder

    rows = pd.read_csv(sample_manifest, dtype=str, keep_default_na=False).to_dict("records")
    request = parse_request(statistic, factors=factors, pair_by=pair_by, covariate=covariate)
    able, refused = plan_each_statistic(rows, request, factors=factors, pair_by=pair_by, covariate=covariate)
    folders = {statistic_folder(name, level) for name, level in able}
    # The composition's statistics are replicate-level (composition_statistic took off any :cell).
    kept = [name.strip() for name in statistic.split(",") if name.strip() and statistic_folder(name.strip()) in folders]
    return ",".join(kept), dict(refused)


def _folders(result: Mapping[str, Any]) -> List[str]:
    names = set()
    for key in ("statistics", "folders"):
        value = result.get(key)
        if isinstance(value, Mapping):
            names.update(map(str, value))
        elif isinstance(value, list):
            names.update(str(v.get("folder") if isinstance(v, Mapping) else v) for v in value)
    for stratum in (result.get("strata") or {}).values() if isinstance(result.get("strata"), Mapping) else []:
        if isinstance(stratum, Mapping):
            names.update(_folders(stratum))
    return sorted(names)


def _within(path: Path, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def summarize(workdir: Optional[Path], zip_dir: Optional[Path] = None) -> Dict[str, Any]:
    """The stage's state: complete when its checkpoint says so, under this method, and its zip is where it goes."""
    if not workdir:
        return {"status": "not_provided", "complete": False, "workdir": None}
    wd = Path(workdir)
    path = stage_checkpoint_path(wd, STAGE)
    try:
        ckpt = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "incomplete", "complete": False, "workdir": str(wd)}
    recorded = Path(str(ckpt.get("zip") or ""))
    here = Path(zip_dir) / recorded.name if zip_dir and recorded.name else recorded
    complete = bool(ckpt.get("complete")) and ckpt.get("method") == METHOD and here.is_file()
    # The cell export is the fit's (the singlecell_rna stage writes it); the statistics are finished
    # when they are made from the fit there now.
    out_of_date = None
    if complete and ckpt.get("fit_sha256"):
        from paralome.singlecell.cell_export import trajectory_fingerprint

        fit_dir = Path(str(ckpt.get("fit_dir") or wd / "slingshot"))
        if (fit_dir / "cells.tsv").is_file() and trajectory_fingerprint(fit_dir) != ckpt["fit_sha256"]:
            out_of_date = "the trajectory fit it was made from has been fitted again"
    return {"status": "complete" if complete and not out_of_date else "incomplete",
            "complete": complete and not out_of_date, "workdir": str(wd),
            "zip": str(here) if here.is_file() else None, "request_id": ckpt.get("request_id"),
            "out_of_date": out_of_date,
            "expression_checkpoint": ckpt.get("expression_checkpoint") if complete else None}
