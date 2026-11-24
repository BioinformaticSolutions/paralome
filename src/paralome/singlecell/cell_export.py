"""One object's cells, genes, expression and embeddings in one file, for an AI agent.

The Evidome results
hold the statistics, not the cells; a UMAP or spatial overlay of a gene needs
every cell's expression and coordinates. So:

* a ZIP of its own, not inside Evidome's;
* one dataset per .rds -- a folder of objects gives one ZIP per object;
* every cell, and every gene that is nonzero in at least one cell;
* log2(CPM + 1), CPM taken over every gene of the cell (as the :cell
  statistics take it);
* an .h5ad (AnnData) file, with a small manifest beside it that an agent can
  read before it opens the file;
* built by the extraction, every time an extraction is written; never by
  re-extracting an object on its own -- an extraction made before the
  coordinates were exported keeps its files, and the manifest says so.

``<stem>_cell_expression_embeddings.zip`` holds
``<stem>_cell_expression_embeddings.h5ad`` and
``<stem>_cell_expression_embeddings_manifest.json``. OmicView's Cell Viewer
draws it (the viewer no longer travels in the ZIP).

The .h5ad is written with h5py to the AnnData on-disk specification
(https://anndata.readthedocs.io/en/latest/fileformat-prose.html), so anndata
itself is not a dependency:

* ``X``       cells x genes, CSR, float32, log2(CPM + 1)
* ``obs``     every cell_metadata.tsv column (numeric where every value is a
              number, otherwise text), indexed by barcode; ``image`` when the
              object has tissue coordinates
* ``var``     feature_name (index), gene_id, common_id
* ``obsm``    ``X_<reduction>`` for each reduction, ``spatial`` (x, y); a cell a
              reduction or image does not cover is NaN
* ``uns``     ``paralome_cell_export`` -- what the file is and where it came from;
              ``spatial`` -- each tissue image the object holds,
              in the layout scanpy reads: ``spatial[<image>]["images"]``
              ``["lowres" or "hires"]`` (8-bit RGB, height x width x 3) and
              ``["scalefactors"]`` (``tissue_lowres_scalef``,
              ``tissue_hires_scalef``, ``spot_diameter_fullres``,
              ``fiducial_diameter_fullres``), so ``scanpy.pl.spatial(a,
              color=<gene>, library_id=<image>)`` draws the gene over the H&E.
              """

from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from paralome.errors import ParalomeError

EXPORT_SUFFIX = "_cell_expression_embeddings"
SCHEMA = "paralome.cell_expression_embeddings.v1"
VALUE = "log2(CPM + 1); CPM over every gene of the cell"
#: Where the file says what it is: ``uns[UNS_KEY]``.
UNS_KEY = "paralome_cell_export"


#: Where the export is drawn: OmicView, the viewer product; no viewer page is written into the ZIP.
OPEN_WITH = "OmicView's Cell Viewer"


def export_name(stem: str) -> str:
    return f"{stem}{EXPORT_SUFFIX}"


def export_home(data_dir: Union[str, Path]) -> Path:
    """Where an object's cell export goes -- ``<parent>/omicview``, beside the data folder
    (``<parent>/data``) the extraction is in, where every run of that object puts its own outputs in a folder of its
    own."""
    from paralome.pipeline.state import OMICVIEW_FOLDER

    return Path(data_dir).parent / OMICVIEW_FOLDER


def export_zip_path(export_dir: Union[str, Path], stem: str) -> Path:
    return Path(export_dir) / f"{export_name(stem)}.zip"


def rds_stem(rds: Union[str, Path]) -> str:
    """The dataset's name: the object's file name without .rds/.qs2."""
    name = Path(str(rds)).name
    for suffix in (".rds", ".qs2"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def is_current(zip_path: Path, checkpoint_path: Path) -> bool:
    """Written after the extraction it describes, by this version of Paralome.

    An export another version wrote is written again from the same extraction,
    so the viewer always reads the layout it expects. Rebuilding the export
    never re-extracts the object. The version is read from the manifest in the
    ZIP.
    """
    from paralome.version import __version__

    if not (zip_path.is_file() and checkpoint_path.is_file()
            and zip_path.stat().st_mtime >= checkpoint_path.stat().st_mtime):
        return False
    try:
        with zipfile.ZipFile(zip_path) as archive:
            manifest = json.loads(archive.read(f"{zip_path.stem}_manifest.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False
    return manifest.get("paralome_version") == __version__


def _require_h5py():
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ParalomeError(
            415, "The cell export (<rds>_cell_expression_embeddings.zip) writes an .h5ad "
                 "file with the h5py package, and this Python cannot import it.",
            fix="python3 -m pip install h5py, into the Python that runs Paralome",
        ) from exc
    import h5py
    return h5py


def _read_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        return header, [row for row in reader]


def _numeric(values: Sequence[str]):
    """A float64 array when every non-empty value is a number, else None."""
    import numpy as np

    out = np.empty(len(values), dtype="float64")
    seen = False
    for index, value in enumerate(values):
        if value == "" or value == "NA":
            out[index] = np.nan
            continue
        try:
            out[index] = float(value)
        except ValueError:
            return None
        seen = True
    return out if seen else None


#: Only reductions a plot draws -- two or three columns (UMAP, t-SNE, a 3-D UMAP). The export is for
#: images and 2D/3D plots, deliberately not for re-analysis; PCA, Harmony and the like (50 columns) would
#: also be a large part of the ZIP.
MAX_REDUCTION_COLUMNS = 3


def _reductions(extraction_dir: Path, barcodes: Sequence[str]) -> Dict[str, "object"]:
    import numpy as np

    out: Dict[str, object] = {}
    folder = extraction_dir / "reductions"
    if not folder.is_dir():
        return out
    position = {barcode: index for index, barcode in enumerate(barcodes)}
    for path in sorted(folder.glob("*.tsv")):
        header, rows = _read_table(path)
        if not 2 <= len(header) - 1 <= MAX_REDUCTION_COLUMNS:
            continue
        matrix = np.full((len(barcodes), len(header) - 1), np.nan, dtype="float32")
        for row in rows:
            index = position.get(row[0])
            if index is None:
                continue
            matrix[index] = [float(v) if v not in ("", "NA") else np.nan for v in row[1:]]
        out[path.stem] = matrix
    return out


def _spatial(extraction_dir: Path, barcodes: Sequence[str]):
    import numpy as np

    path = extraction_dir / "spatial_coordinates.tsv"
    if not path.is_file():
        return None, None
    header, rows = _read_table(path)
    column = {name: header.index(name) for name in ("barcode", "image", "x", "y")}
    position = {barcode: index for index, barcode in enumerate(barcodes)}
    xy = np.full((len(barcodes), 2), np.nan, dtype="float64")
    image = [""] * len(barcodes)
    for row in rows:
        index = position.get(row[column["barcode"]])
        if index is None:
            continue
        xy[index] = (float(row[column["x"]]), float(row[column["y"]]))
        image[index] = row[column["image"]]
    return xy, image


#: Seurat's scale-factor names, as scanpy names them.
_SCALEFACTORS = {"lowres": "tissue_lowres_scalef", "hires": "tissue_hires_scalef",
                 "spot": "spot_diameter_fullres", "fiducial": "fiducial_diameter_fullres"}


def _tissue_images(extraction_dir: Path, xy, image: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Each image the extraction wrote (images/<name>.rgb + .tsv), with its resolution named.

    Seurat keeps one picture per image and does not say which: the lowres H&E
    Load10X_Spatial reads by default, or the hires one when it was loaded. The
    one whose scale factor maps the spots' full-resolution extent onto the
    picture's width is the one it is.
    """
    import math

    import numpy as np

    out: Dict[str, Dict[str, object]] = {}
    folder = extraction_dir / "images"
    if not folder.is_dir():
        return out
    for described in sorted(folder.glob("*.tsv")):
        name = described.stem
        raw = folder / f"{name}.rgb"
        if not raw.is_file():
            continue
        fields = {}
        for line in described.read_text().splitlines():
            key, _, value = line.partition("\t")
            if value:
                fields[key] = float(value)
        height, width = int(fields.get("height", 0)), int(fields.get("width", 0))
        data = np.frombuffer(raw.read_bytes(), dtype="uint8")
        if height <= 0 or width <= 0 or data.size != height * width * 3:
            raise ParalomeError(211, f"images/{raw.name} holds {data.size} bytes, not the "
                                     f"{height} x {width} x 3 its {described.name} describes")
        factors = {_SCALEFACTORS[key]: value for key, value in fields.items() if key in _SCALEFACTORS}
        resolution = "lowres"
        if xy is not None and {"tissue_lowres_scalef", "tissue_hires_scalef"} <= set(factors):
            columns = [xy[i, 0] for i, label in enumerate(image) if label == name and not math.isnan(xy[i, 0])]
            if columns:
                extent = max(columns)
                resolution = min(("lowres", "hires"), key=lambda res: abs(math.log(
                    width / max(extent * factors[_SCALEFACTORS[res]], 1e-9))))
        out[name] = {"array": data.reshape(height, width, 3), "scalefactors": factors,
                     "resolution": resolution}
    return out


def _log_cpm_cells_by_genes(counts_genes_by_cells):
    """log2(CPM + 1) per cell, as CSR cells x genes; genes zero in every cell dropped."""
    import numpy as np
    from scipy.sparse import csc_matrix

    # A copy: the extraction's reader hands back read-only buffers.
    matrix = csc_matrix(counts_genes_by_cells, dtype="float64", copy=True)
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    scale = np.divide(1e6, totals, out=np.zeros_like(totals), where=totals > 0)
    values = matrix.data * np.repeat(scale, np.diff(matrix.indptr))
    logged = csc_matrix((np.log2(values + 1.0), matrix.indices.copy(), matrix.indptr.copy()),
                        shape=matrix.shape)
    logged.eliminate_zeros()
    kept = np.flatnonzero(np.diff(logged.tocsr().indptr) > 0)
    cells_by_genes = logged[kept, :].T.tocsr().astype("float32")
    return cells_by_genes, kept


# --- the AnnData on-disk specification, written with h5py -------------------

def _attrs(node, kind: str, version: str) -> None:
    node.attrs["encoding-type"] = kind
    node.attrs["encoding-version"] = version


def _strings(h5py, group, name: str, values: Sequence[str]):
    dataset = group.create_dataset(name, data=list(values), dtype=h5py.string_dtype("utf-8"))
    _attrs(dataset, "string-array", "0.2.0")
    return dataset


def _array(group, name: str, values, compress: bool = True):
    # The shuffle filter before gzip groups bytes of equal weight;
    # the gene indices shrink by about a third, losslessly, and every HDF5
    # reader undoes it on its own.
    dataset = group.create_dataset(name, data=values, **(
        {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        if compress and getattr(values, "size", 0) > 1024 else {}))
    _attrs(dataset, "array", "0.2.0")
    return dataset


def _categorical(h5py, group, name: str, values: Sequence[str]) -> None:
    """A text column as AnnData's categorical: codes plus the distinct values.

    A sample or stratum name repeated per cell is a few bytes as a code rather
    than the whole string per cell.
    """
    import numpy as np

    categories = sorted(set(values))
    code = {value: index for index, value in enumerate(categories)}
    dtype = "int8" if len(categories) < 128 else "int16" if len(categories) < 32768 else "int32"
    node = group.create_group(name)
    _attrs(node, "categorical", "0.2.0")
    node.attrs["ordered"] = False
    _array(node, "codes", np.fromiter((code[value] for value in values), dtype=dtype, count=len(values)))
    _strings(h5py, node, "categories", categories)


def _is_categorical(values: Sequence[str]) -> bool:
    """Any text that repeats; a column of unique values (an ID per cell) stays text."""
    return len(set(values)) < len(values)


def _dataframe(h5py, parent, name: str, index: Sequence[str], columns: Mapping[str, object]) -> None:
    group = parent.create_group(name)
    _attrs(group, "dataframe", "0.2.0")
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = list(columns) if columns else h5py.Empty("f")
    _strings(h5py, group, "_index", index)
    for column, values in columns.items():
        if isinstance(values, list) and _is_categorical(values):
            _categorical(h5py, group, column, values)
        elif isinstance(values, list):
            _strings(h5py, group, column, values)
        else:
            _array(group, column, values)


def _empty_dict(parent, name: str):
    group = parent.create_group(name)
    _attrs(group, "dict", "0.1.0")
    return group


def write_h5ad(path: Path, *, matrix, barcodes: Sequence[str], obs: Mapping[str, object],
               genes: Mapping[str, Sequence[str]], obsm: Mapping[str, object],
               uns: Mapping[str, str], tissue_images: Optional[Mapping[str, Mapping[str, object]]] = None) -> None:
    h5py = _require_h5py()
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w") as handle:
        _attrs(handle, "anndata", "0.1.0")
        x = handle.create_group("X")
        # CSC -- one gene's cells are one contiguous slice, which is
        # what an overlay reads (the viewer reads one gene at a time). AnnData
        # and scanpy read either layout.
        by_gene = matrix.tocsc()
        by_gene.sort_indices()
        _attrs(x, "csc_matrix", "0.1.0")
        x.attrs["shape"] = list(matrix.shape)
        _array(x, "data", by_gene.data.astype("float32"))
        _array(x, "indices", by_gene.indices.astype("int32"))
        _array(x, "indptr", by_gene.indptr.astype("int64"))
        _dataframe(h5py, handle, "obs", barcodes, obs)
        _dataframe(h5py, handle, "var", genes["feature_name"],
                   {name: list(values) for name, values in genes.items() if name != "feature_name"})
        group = _empty_dict(handle, "obsm")
        for name, values in obsm.items():
            _array(group, name, values)
        for name in ("varm", "obsp", "varp", "layers"):
            _empty_dict(handle, name)
        group = _empty_dict(handle, "uns")
        provenance = _empty_dict(group, UNS_KEY)
        for key, value in uns.items():
            dataset = provenance.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))
            _attrs(dataset, "string", "0.2.0")
        if tissue_images:
            spatial = _empty_dict(group, "spatial")
            for name, picture in tissue_images.items():
                library = _empty_dict(spatial, name)
                images = _empty_dict(library, "images")
                _array(images, str(picture["resolution"]), picture["array"])
                factors = _empty_dict(library, "scalefactors")
                for key, value in picture["scalefactors"].items():
                    dataset = factors.create_dataset(key, data=float(value))
                    _attrs(dataset, "numeric-scalar", "0.2.0")
                _empty_dict(library, "metadata")
    os.replace(tmp, path)


# --- the export ---------------------------------------------------------------

def build_cell_export(
    extraction_dir: Union[str, Path],
    export_dir: Union[str, Path],
    *,
    rds: Union[str, Path],
    log=print,
) -> Optional[Path]:
    """Write ``<stem>_cell_expression_embeddings.zip`` for one object's extraction.

    An object with no RNA half (snATAC alone) is exported too, without genes: its cells'
    metadata and embeddings, which OmicView's Cells viewer colours by any column. Returns the ZIP.
    """
    import numpy as np

    from paralome.atac.singlecell import load_counts_matrix
    from paralome.version import __version__

    extraction_dir = Path(extraction_dir)
    stem = rds_stem(rds)
    has_rna = (extraction_dir / "rna_counts.plmsp").is_file() or (extraction_dir / "rna_counts.mtx").is_file()
    _require_h5py()

    meta_header, meta_rows = _read_table(extraction_dir / "cell_metadata.tsv")
    if (extraction_dir / "cells.tsv").is_file():
        barcodes = [line for line in (extraction_dir / "cells.tsv").read_text().splitlines() if line]
    else:
        barcodes = [row[0] for row in meta_rows if row]
    if has_rna:
        counts_file = extraction_dir / ("rna_counts.plmsp" if (extraction_dir / "rna_counts.plmsp").is_file()
                                        else "rna_counts.mtx")
        counts = load_counts_matrix(counts_file)
        if counts.shape[1] != len(barcodes):
            raise ParalomeError(211, f"{counts_file.name} has {counts.shape[1]} cells and cells.tsv "
                                     f"{len(barcodes)}; the extraction's files disagree")
        matrix, kept = _log_cpm_cells_by_genes(counts)

        gene_header, gene_rows = _read_table(extraction_dir / "rna_genes.tsv")
        gene_column = {name: gene_header.index(name) for name in gene_header}
        genes = {name: [gene_rows[i][gene_column[name]] if gene_column[name] < len(gene_rows[i]) else ""
                        for i in kept]
                 for name in ("feature_name", "gene_id", "common_id") if name in gene_column}
    else:
        from scipy.sparse import csr_matrix

        matrix = csr_matrix((len(barcodes), 0), dtype=np.float32)
        genes = {"feature_name": [], "gene_id": [], "common_id": []}

    by_barcode = {row[0]: row for row in meta_rows}
    ordered = [by_barcode.get(barcode, [barcode] + [""] * (len(meta_header) - 1)) for barcode in barcodes]
    obs: Dict[str, object] = {}
    for column_index, column in enumerate(meta_header):
        if column == "barcode":
            continue
        values = [row[column_index] if column_index < len(row) else "" for row in ordered]
        numeric = _numeric(values)
        obs[column] = numeric if numeric is not None else values

    obsm = {f"X_{name}": values for name, values in _reductions(extraction_dir, barcodes).items()}
    xy, image = _spatial(extraction_dir, barcodes)
    if xy is not None:
        obsm["spatial"] = xy.astype("float32")
        if "image" not in obs:
            obs["image"] = image

    tissue_images = _tissue_images(extraction_dir, xy, image or [])
    coordinates_exported = (extraction_dir / "reductions").is_dir() or xy is not None
    extraction_manifest = {}
    manifest_path = extraction_dir / "extraction_manifest.json"
    if manifest_path.is_file():
        extraction_manifest = json.loads(manifest_path.read_text())
    knows_coordinates = "reductions" in extraction_manifest
    name = export_name(stem)
    uns = {
        "schema": SCHEMA, "source_rds": str(rds), "value": VALUE,
        "paralome_version": __version__,
    }
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    h5ad = export_dir / f"{name}.h5ad"
    write_h5ad(h5ad, matrix=matrix, barcodes=barcodes, obs=obs, genes=genes, obsm=obsm, uns=uns,
               tissue_images=tissue_images)

    manifest = {
        "schema": SCHEMA,
        "what": "One single-cell or spatial object's cells, genes, expression and embeddings, for "
                "plotting (UMAP, spatial gene overlays, violins, dot plots) by any tool that reads "
                "AnnData (.h5ad): Python anndata/scanpy, R anndata/zellkonverter.",
        "file": h5ad.name,
        "source_rds": str(rds),
        "paralome_version": __version__,
        "n_cells": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "genes_left_out": "genes with no count in any cell",
        "X": (VALUE + "; sparse CSC (one gene's cells contiguous), cells x genes") if has_rna else
             "none: the object has no RNA half, so there is no expression; the cells' metadata and embeddings only",
        "obs": {column: ("number" if not isinstance(values, list) else "text") for column, values in obs.items()},
        "var": {"index": "feature_name (the assay's row name)",
                "gene_id": "Ensembl gene ID from the --genome GTF, '.' when it has none",
                "common_id": "gene symbol, '.' when it has none"},
        "obsm": {key: f"{values.shape[1]} columns" for key, values in obsm.items()},
        "reductions_left_out": "any reduction of more than 3 dimensions (PCA, Harmony, ...): this file is "
                               "for plotting, not for re-analysis",
        "images": sorted({value for value in image if value}) if xy is not None else [],
        "tissue_images": {
            name: {"in": f"uns['spatial']['{name}']['images']['{picture['resolution']}']",
                   "resolution": picture["resolution"],
                   "height": int(picture["array"].shape[0]), "width": int(picture["array"].shape[1]),
                   "scalefactors": picture["scalefactors"]}
            for name, picture in tissue_images.items()},
        "read_with": "import anndata as ad; a = ad.read_h5ad('" + h5ad.name + "')",
    }
    if tissue_images:
        first = sorted(tissue_images)[0]
        manifest["plot_over_the_tissue_with"] = (
            f"import scanpy as sc; sc.pl.spatial(a[a.obs['image'] == '{first}'], color='<gene>', "
            f"library_id='{first}', img_key='{tissue_images[first]['resolution']}')")
    elif xy is not None and "tissue_images" not in extraction_manifest:
        manifest["tissue_image"] = (
            "none: this extraction was made by an older Paralome, before it exported the object's tissue "
            "image. Move the extraction folder aside and run again to extract the object anew; "
            "nothing re-extracts it on its own")
    if not coordinates_exported:
        manifest["coordinates"] = (
            "none: " + ("the object has no reductions or images" if knows_coordinates else
                        "this extraction was made by an older Paralome, before it exported reductions and "
                        "tissue coordinates. Move the extraction folder aside and run again to "
                        "extract the object anew; nothing re-extracts it on its own"))
    zip_path = export_zip_path(export_dir, stem)
    tmp = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(tmp, "w") as archive:
        archive.write(h5ad, h5ad.name, compress_type=zipfile.ZIP_STORED)   # compressed inside already
        archive.writestr(f"{name}_manifest.json", json.dumps(manifest, indent=2) + "\n",
                         compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, zip_path)
    h5ad.unlink()
    log(f"[paralome] cell export: {zip_path} ({matrix.shape[0]} cells x {matrix.shape[1]} genes"
        f"{'; obsm ' + ', '.join(obsm) if obsm else '; no reductions or coordinates'})")
    return zip_path


# --- the trajectories, as cell metadata ----------------------------------------------------------------
# The trajectory itself, as metadata for the cell embeddings -- not the trajectory zip's statistics (tens of GB, for an AI agent). The trajectories stage writes its
# fit into the object's export: obs gains pseudotime (each cell's pseudotime averaged over its lineages, weighted by
# its weights, as slingshot::slingAvgPseudotime), pseudotime_<lineage> (blank off that lineage) and
# trajectory_lineages (the lineages a cell is on); uns["paralome_trajectories"] holds each lineage's clusters and its
# curve in each embedding the curves were drawn in (curves/<embedding>: lineage, coordinates). An export the
# extraction writes again (another Paralome version) is given them again the next time the stage runs.
TRAJECTORY_UNS = "paralome_trajectories"


def trajectory_fingerprint(fit_dir: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in [fit_dir / "cells.tsv", fit_dir / "lineages.tsv", *sorted(fit_dir.glob("curve_*.tsv"))]:
        if path.is_file():
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _trajectory_columns(fit_dir: Path, barcodes: Sequence[str]):
    """{obs column: values in the export's cell order} from the fit's cells.tsv; the lineage names."""
    import numpy as np
    import pandas as pd

    cells = pd.read_csv(fit_dir / "cells.tsv", sep="\t", dtype={"barcode": str, "branch": str},
                        keep_default_na=False, na_values=[""]).set_index("barcode")
    lineages = [c[len("pseudotime_"):] for c in cells.columns if c.startswith("pseudotime_")]
    cells = cells.reindex(list(barcodes))
    times = np.column_stack([cells[f"pseudotime_{L}"].to_numpy(dtype=float) for L in lineages])
    weights = np.column_stack([cells[f"weight_{L}"].to_numpy(dtype=float) for L in lineages])
    use = np.isfinite(times) & (np.nan_to_num(weights) > 0)
    total = np.where(use, weights, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        average = np.where(total > 0, np.where(use, times * weights, 0.0).sum(axis=1) / total, np.nan)
    columns: Dict[str, object] = {"pseudotime": average.astype("float32")}
    for k, L in enumerate(lineages):
        columns[f"pseudotime_{L}"] = times[:, k].astype("float32")
    branch = cells["branch"] if "branch" in cells.columns else None
    columns["trajectory_lineages"] = [
        (str(branch.iloc[i]) if branch is not None and isinstance(branch.iloc[i], str) else
         ",".join(L for k, L in enumerate(lineages) if np.isfinite(times[i, k])))
        for i in range(len(cells))]
    return columns, lineages


def add_trajectories(zip_path: Path, fit_dir: Path, *, root: str = "", log=print) -> bool:
    """Write the fit in ``fit_dir`` into the export ``zip_path``; False when it already holds this fit."""
    import numpy as np

    zip_path, fit_dir = Path(zip_path), Path(fit_dir)
    fingerprint = trajectory_fingerprint(fit_dir)
    manifest_name = f"{zip_path.stem}_manifest.json"
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read(manifest_name))
        if (manifest.get("trajectories") or {}).get("fit_sha256") == fingerprint:
            return False
        h5ad_name = manifest.get("file") or f"{zip_path.stem}.h5ad"
        work = zip_path.with_name(zip_path.name + ".h5ad.part")
        with archive.open(h5ad_name) as source, open(work, "wb") as target:
            shutil.copyfileobj(source, target, 16 << 20)
    h5py = _require_h5py()
    try:
        with h5py.File(work, "r+") as handle:
            obs = handle["obs"]
            index = obs[obs.attrs.get("_index", "_index")]
            barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in index[()]]
            columns, lineages = _trajectory_columns(fit_dir, barcodes)
            order = [str(c) for c in (obs.attrs.get("column-order") if isinstance(obs.attrs.get("column-order"), np.ndarray) else [])]
            uns = handle["uns"]
            earlier = []
            if TRAJECTORY_UNS in uns:
                earlier = [str(c.decode() if isinstance(c, bytes) else c) for c in uns[TRAJECTORY_UNS]["obs_columns"][()]]
                del uns[TRAJECTORY_UNS]
            for name in set(earlier) | set(columns):
                if name in obs:
                    del obs[name]
            order = [c for c in order if c not in earlier and c not in columns]
            for name, values in columns.items():
                if isinstance(values, list):
                    _categorical(h5py, obs, name, values)
                else:
                    _array(obs, name, values)
            obs.attrs["column-order"] = order + list(columns)
            group = _empty_dict(uns, TRAJECTORY_UNS)
            _strings(h5py, group, "obs_columns", list(columns))
            paths = {}
            if (fit_dir / "lineages.tsv").is_file():
                header, rows = _read_table(fit_dir / "lineages.tsv")
                paths = {row[0]: (row[1] if len(row) > 1 else "") for row in rows}
            _strings(h5py, group, "lineages", lineages)
            _strings(h5py, group, "lineage_clusters", [paths.get(L, "") for L in lineages])
            for key, value in (("root", root), ("fit_sha256", fingerprint)):
                dataset = group.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))
                _attrs(dataset, "string", "0.2.0")
            curves = _empty_dict(group, "curves")
            drawn = []
            for path in sorted(fit_dir.glob("curve_*.tsv")):
                header, rows = _read_table(path)
                if not rows:
                    continue
                name = path.stem[len("curve_"):]
                rows.sort(key=lambda row: (row[0], float(row[1])))
                one = _empty_dict(curves, f"X_{name}")
                _strings(h5py, one, "lineage", [row[0] for row in rows])
                _array(one, "coordinates", np.array([[float(x) for x in row[2:]] for row in rows], dtype="float32"),
                       compress=False)
                drawn.append(f"X_{name}")
        placed = int(np.isfinite(columns["pseudotime"]).sum())
        manifest.setdefault("obs", {}).update({name: ("text" if isinstance(values, list) else "number")
                                               for name, values in columns.items()})
        manifest["trajectories"] = {
            "what": "the run's lineages (Monocle3 or Slingshot: method) as cell metadata: pseudotime (the cell's "
                    "pseudotime averaged over its lineages, weighted by its weights), pseudotime_<lineage> (blank "
                    "off that lineage), trajectory_lineages (the lineages a cell is on); each lineage's curve in "
                    "uns['paralome_trajectories']['curves']['<embedding>'] (lineage, coordinates)",
            "lineages": len(lineages), "cells_placed": placed, "root": root, "curves_in": drawn,
            "fit_sha256": fingerprint,
            # Which method made the fit.
            "method": next((name for name in ("monocle3", "slingshot") if (fit_dir / f"{name}.json").is_file()),
                           "slingshot"),
        }
        from paralome.version import __version__

        manifest["trajectories"]["paralome_version"] = __version__
        tmp = zip_path.with_name(zip_path.name + ".part")
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as archive:
            archive.write(work, h5ad_name, compress_type=zipfile.ZIP_STORED)
            archive.writestr(manifest_name, json.dumps(manifest, indent=2) + "\n", compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp, zip_path)
    finally:
        if work.exists():
            work.unlink()
    log(f"[paralome] cell export: {zip_path.name} now carries the trajectories ({len(lineages)} lineages, "
        f"{placed} of {len(barcodes)} cells placed; curves in {', '.join(drawn) or 'no embedding'})")
    return True
