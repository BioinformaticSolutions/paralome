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

"""Single-cell ATAC pseudobulk bridge.

Entry point for single-cell ATAC data (peaks already called by Signac in a
Seurat object; the bulk pipeline's ``macs``/``peak_merge`` front end is not
needed). The user exports exactly one file from R -- ``saveRDS(object, "x.rds")``
-- and everything else happens on the Paralome side:

1. ``run_seurat_extraction`` shells out to the bundled R helper
   (``atac/r/extract_seurat_assays.R``) to pull the peak set,
   sparse counts matrix, and the *full* cell metadata out of the object. The user
   never has to know this internal file contract; it is regenerated on every
   run and checkpointed like every other Paralome stage.
2. ``aggregate_pseudobulk`` groups cells into pseudobulk "samples" by
   whichever Seurat metadata column(s) the caller chooses (e.g.
   ``sample,cell_type``), sums counts per group, and emits the exact
   ``raw_counts`` / ``cpm`` / ``log2_cpm`` matrix shape that
   ``paralome.atac.quantification`` already produces, plus a sample manifest
   compatible with ``paralome.atac.statistics``' reader. Every metadata
   column that is constant within a pseudobulk group rides along in the
   manifest as a report-only column, so nothing the user tagged onto their
   cells gets silently dropped.
3. The differential is ``atac singlecell-differential`` (pseudobulk by
   default, ``--stratify-by`` for one comparison per stratum); it is the only
   single-cell differential.
"""

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix

from paralome.layout import extraction_dir as layout_extraction_dir
from paralome.stratify import stratum_columns
from paralome.checkpoints import (
    atomic_text_writer,
    atomic_write_text,
    exclusive_checkpoint_lock,
    stage_checkpoint_path,
)
from paralome.pipeline.jsonio import version_stale_reason
from paralome.pipeline.resource_catalog import resolve_genome_resources
from paralome.errors import ParalomeError
from paralome.validate import check_bounded_number
from paralome.atac.naming import derive_file_prefix, stage_banner
from paralome.atac.quantification import (
    ATLAS_FILTER,
    AtlasPeak,
    _write_matrix,
    annotate_peaks,
    is_mitochondrial_contig,
    load_tss,
    read_atlas,
)
from paralome.singlecell.pseudobulk_core import (
    aggregate_pseudobulk_core,
    write_cell_counts_table,
)

SC_EXTRACT_SCHEMA = "paralome.atac.singlecell_extract.v1"
SC_PSEUDOBULK_SCHEMA = "paralome.atac.singlecell_atac.v1"
# The pseudobulk method: the date the aggregation last changed what it computes
# (chrM peaks excluded, empty samples dropped). Recorded in the checkpoint and
# required by pipeline/state.py: a checkpoint under another method reads out of
# date, so the pipeline aggregates again before the differential reads cpm.csv.
SC_PSEUDOBULK_METHOD = "2026-10-11"

#: What R's serialisation formats look like in the first bytes. A
#: 2 GB object written with save() instead of saveRDS() produced only
#: "Error in readRDS(opts$rds) : unknown input format", four stack frames deep,
#: in a log directory the failure message never named. The file is already read
#: in full to compute its sha256; looking at the first four bytes of that same
#: read costs nothing and says which mistake was made.
_RDS_MAGIC = {
    b"RDX2": "an R workspace written by save() -- not an RDS",
    b"RDX3": "an R workspace written by save() -- not an RDS",
    b"RDA2": "an R workspace written by save() -- not an RDS",
}
_HDF5_MAGIC = b"\x89HDF"


def describe_rds_format(path: Path) -> Optional[str]:
    """Why this file is not an RDS, or None if it looks like one.

    Recognises the near-misses that produce an identical-looking file with the
    same extension and roughly the same size: save() instead of saveRDS(),
    SaveH5Seurat instead of either. R's own message for all of them is
    "unknown input format".
    """
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(4)
    except OSError:
        return None
    if not head:
        return f"{Path(path).name} is empty."
    if head[:2] == b"\x1f\x8b" or head[:2] == b"BZ" or head[:2] == b"X\n":
        return None                      # gzip / bzip2 / uncompressed RDS
    if head[:3] == b"\xfd7z":
        return None                      # xz
    if head == _HDF5_MAGIC:
        return (
            f"{Path(path).name} is an HDF5 file, not an RDS -- probably an "
            ".h5Seurat saved with SaveH5Seurat(). Re-save it with "
            "saveRDS(object, path), or point --rds at the RDS."
        )
    described = _RDS_MAGIC.get(head)
    if described:
        return (
            f"{Path(path).name} is {described} (magic {head!r}).\n"
            "Convert it:\n"
            "  e <- new.env(); nm <- load(<path>, envir = e)\n"
            "  saveRDS(e[[nm[1]]], <new path>)\n"
            "then point --rds at the new file."
        )
    return None


# ATAC does not always come with RNA, nor RNA with ATAC, so the extraction reads whichever assays are
# present. The R helper discovers which halves the object has, so this side cannot predict its
# outputs from its own arguments -- it reads extraction_manifest.json's has_atac/has_rna and checks
# what was promised.
_SHARED_EXTRACTION_OUTPUTS = ("cells.tsv", "cell_metadata.tsv")
_ATAC_EXTRACTION_OUTPUTS = ("peaks.bed", "peaks_order.tsv", "counts.mtx")
_RNA_EXTRACTION_OUTPUTS = ("rna_counts.mtx", "rna_genes.tsv", "rna_genes_order.tsv")

# Retained: the former name, still the full ATAC contract, imported by callers that mean "an ATAC
# extraction's files".
_EXTRACTION_OUTPUTS = _ATAC_EXTRACTION_OUTPUTS + _SHARED_EXTRACTION_OUTPUTS


def _raise_assay_error(output_dir: Path) -> None:
    """Turn the R helper's assay_error.json into a numbered error, if it wrote one.

    Returns quietly when there is no such file: the extraction failed for some
    other reason and the caller's own message is the better one.
    """
    from paralome.errors import ParalomeError

    path = Path(output_dir) / "assay_error.json"
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    code = int(report.get("code") or 0)
    if code not in (214, 216, 217):
        return
    available = report.get("available_assays") or []
    detail = str(report.get("message") or "Assay resolution failed")
    if available:
        detail += f"\nAssays present: {', '.join(str(a) for a in available)}"
    raise ParalomeError(code, detail, requested=report.get("requested"),
                        available_assays=available)


def _raise_extraction_error(output_dir: Path) -> None:
    """Turn the R helper's extraction_error.json into a numbered error.

    Same contract as ``_raise_assay_error`` above, generalised: R writes a code
    and a message, Python raises it with that number. Returns quietly when there
    is no such file, so an unstructured R failure still reaches the caller's own
    message rather than being swallowed. (For example: a .qs2 object on a
    cluster where qs2 is installed on the login node but not on the compute
    node arrives as its own numbered error, not as an unexpected internal one.)
    """
    from paralome.errors import REGISTRY, ParalomeError

    path = Path(output_dir) / "extraction_error.json"
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    code = int(report.get("code") or 0)
    if code not in REGISTRY:
        return
    raise ParalomeError(
        code,
        str(report.get("message") or "The extraction helper failed"),
        fix=report.get("fix") or None,
        extraction_dir=str(output_dir),
    )


def _halves_present(manifest: Mapping[str, object]) -> Dict[str, bool]:
    """Which halves a manifest or checkpoint says were extracted.

    Records written before ``has_atac``/``has_rna`` existed have neither, and
    every one of them described an extraction that necessarily had an ATAC
    half. So a missing key is read as that contract, inferred from the assay
    names present, not as "no assays found" (which would make every such
    checkpoint look empty and re-extract it).
    """
    def _flag(key: str, legacy: object) -> bool:
        value = manifest.get(key)
        if value is None:
            return bool(legacy)
        return bool(value)

    atac_legacy = manifest.get("assay") or manifest.get("n_peaks")
    return {
        "atac": _flag("has_atac", atac_legacy),
        "rna": _flag("has_rna", manifest.get("rna_assay")),
    }


#: Counts matrices the extraction may have written either way: the binary form, or the text form an
#: older extraction directory holds.
_MATRIX_ALTERNATIVES = {
    "counts.mtx": "counts.plmsp",
    "rna_counts.mtx": "rna_counts.plmsp",
}


def _outputs_for(
    has_atac: bool, has_rna: bool, output_dir: Optional[Path] = None
) -> List[str]:
    """The files this extraction is expected to have produced.

    The counts matrices are named by what is actually on disk. The R script
    writes ``counts.plmsp`` (binary); an older extraction directory holding
    ``counts.mtx`` (MatrixMarket text) keeps validating against the file it
    really has, rather than being declared incomplete and re-extracted.
    """
    names = list(_SHARED_EXTRACTION_OUTPUTS)
    if has_atac:
        names = list(_ATAC_EXTRACTION_OUTPUTS) + names
    if has_rna:
        names += list(_RNA_EXTRACTION_OUTPUTS)
    if output_dir is None:
        return names
    resolved = []
    for name in names:
        binary = _MATRIX_ALTERNATIVES.get(name)
        if binary and not (output_dir / name).is_file() and (output_dir / binary).is_file():
            resolved.append(binary)
        else:
            resolved.append(name)
    return resolved

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime, "sha256": _sha256_file(path)}


def _cached_identity(checkpoint_path: Path) -> Optional[Mapping[str, object]]:
    """The ``rds_identity`` recorded by a previous extraction, if readable."""
    try:
        stored = json.loads(checkpoint_path.read_text())
    except (OSError, ValueError):
        return None
    identity = stored.get("rds_identity")
    return identity if isinstance(identity, Mapping) else None


def _file_identity_cached(path: Path, checkpoint_path: Path) -> Dict[str, object]:
    """``_file_identity``, but skip the full SHA-256 when path/size/mtime are unchanged.

    The extraction checkpoint is keyed on a signature that includes
    a SHA-256 of the whole .rds -- so computing the key meant streaming the
    entire object (often hundreds of MB) before the checkpoint could even be
    consulted, which can take minutes for an extraction that is already
    complete and valid.

    ``stat()`` is free and ``_file_identity`` already recorded ``size`` and
    ``mtime`` alongside the hash. When both match what the checkpoint stored
    for the same path, the stored SHA-256 is reused verbatim, so the resulting
    signature is bit-identical to what a full re-hash would have produced --
    this changes only how long it takes to arrive at the same answer, never
    the answer. Any difference in path, size or mtime (or a missing/corrupt
    checkpoint) falls back to the full hash, so a genuinely changed object is
    still caught.
    """
    stat = path.stat()
    cheap: Dict[str, object] = {"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime}
    cached = _cached_identity(checkpoint_path)
    if cached and cached.get("sha256") and all(cached.get(key) == value for key, value in cheap.items()):
        return {**cheap, "sha256": str(cached["sha256"])}
    return {**cheap, "sha256": _sha256_file(path)}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    # See pipeline.jsonio.CHECKPOINT_VERSION_KEY.
    from paralome.pipeline.jsonio import stamp_version

    atomic_write_text(path, json.dumps(stamp_version(payload), indent=2, sort_keys=True) + "\n")


def sanitize_sample_id(value: str) -> str:
    """Turn an arbitrary metadata-derived label into a safe pseudobulk sample_id token."""
    cleaned = _SANITIZE_RE.sub("_", str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unlabeled"


#: Sha256 of extraction scripts whose outputs the current one only
#: adds to (reductions/, spatial_coordinates.tsv, images/). Their
#: extractions are reused, and the cell export says what they lack. Never a script whose outputs
#: changed: that one must re-extract.
_OUTPUT_COMPATIBLE_R_SCRIPTS = frozenset({
    "36faaa5891e2c3c09b0a0b4119cc26c686d3e04000cd4c24f38bebe20dcec331",
    "594640368952e4a396f3c81b88db9d2a0cf4e96047d59775df32ee73861a0928",
    # Adds seurat_commands.tsv (the object's command log, for the trajectories' dims).
    "48651bc5b4be3ff88d10ac9e20cb7c3e1b734c44563509a66dbd2ef80451594b",
})


def _default_r_script_path() -> Path:
    return Path(__file__).resolve().parent / "r" / "extract_seurat_assays.R"


def _run_rscript_streaming(argv: Sequence[str], output_dir: Path) -> subprocess.CompletedProcess:
    """Run the extraction R script, echoing its output live instead of swallowing it.

    R's progress reporting is streamed line-by-line to stderr as it arrives
    (so it lands in the SLURM ``.err`` for the task, and a long extraction over
    network storage is visibly alive) AND written to
    ``<extraction>/logs/seurat_extraction.log``, while still being captured
    for the RuntimeError message on failure.
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "seurat_extraction.log"

    lines: List[str] = []
    process = subprocess.Popen(
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    with log_path.open("w") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            log_handle.write(line)
            log_handle.flush()
            print(line.rstrip("\n"), file=sys.stderr, flush=True)
    returncode = process.wait()
    combined = "".join(lines)
    return subprocess.CompletedProcess(list(argv), returncode, stdout=combined, stderr="")


def _record_aggregation_qc(
    project_dir: Path,
    modality: str,
    written: Mapping[str, object],
    *,
    group_by: Sequence[str],
    stratify_by: Optional[str],
    min_cells_per_group: int,
    remove: Optional[Sequence[str]],
    plan_id: str,
    n_samples: int,
) -> Dict[str, object]:
    """Copy this aggregation's cells-per-sample table into a timestamped qc/ run.

    All aggregated tables and PCAs are time stamped and kept, so the user has
    a paper trail to understand and go back to. ``cell_counts_per_sample.csv`` is
    written by ``write_pseudobulk_outputs`` to a fixed path that each new
    grouping overwrites. The canonical copy stays where it is (pipeline state
    and the --remove help text both reference it); this additionally preserves
    a dated copy alongside a ``run_params.json`` recording the grouping that
    produced it, and indexes it in ``qc/runs.csv``.

    Never fatal: losing the paper trail must not lose the aggregation.
    """
    from paralome.singlecell.qc import record_run

    try:
        run = record_run(project_dir, modality, "singlecell_atac", {
            "group_by": list(group_by),
            "stratify_by": stratify_by,
            "min_cells_per_group": min_cells_per_group,
            "removed_samples": sorted(str(value) for value in remove) if remove else [],
            "plan_id": plan_id,
            "n_samples": n_samples,
        })
        source = written.get("cell_counts_per_sample")
        if source and Path(str(source)).is_file():
            destination = Path(run["run_dir"]) / "cell_counts_per_sample.csv"
            destination.write_text(Path(str(source)).read_text())
            run["qc_cell_counts_per_sample"] = str(destination)

        # The pseudobulk sample QC is the viewer's (from the statistics output); none is drawn here.
        return {"qc_run_id": run["run_id"], "qc_run_dir": run["run_dir"],
                **{k: v for k, v in run.items() if k.startswith("qc_")}}
    except Exception as error:  # pragma: no cover - QC must never break aggregation
        print(f"WARNING: could not record aggregation QC run: {error}", file=sys.stderr)
        return {}


def _write_extraction_qc(
    extraction: Mapping[str, object], extraction_dir: Path, *, overwrite: bool
) -> None:
    """Write the ``<out>/qc/`` metadata tables for a completed extraction.

    Whenever Paralome (from any entry point) reads the RDS to extract, the
    RDS checks -- which metadata columns exist, etc. -- are written as a
    metadata summary to the same folder at the same time. Hooking it here rather than in
    any one command means every entry point -- `pipeline extract`,
    `pipeline run`, both modalities' `singlecell-samples`, and
    `singlecell-global-pca` -- produces it identically.

    ``overwrite`` is True only when an extraction actually ran: metadata
    tables that are a direct export of the extraction are overwritten by a
    new extraction. On a cache hit the tables are
    written only if absent, so an existing project picks them up without
    being forced to re-extract.

    Reads ``cell_metadata.tsv`` directly -- deliberately NOT
    ``load_singlecell_inputs``, which would pull the entire counts matrix
    through ``mmread`` just to look at metadata. Never fatal: a QC-table
    failure must not lose a completed extraction.
    """
    from paralome.singlecell.qc import write_metadata_qc

    try:
        metadata = read_cell_metadata(extraction_dir / "cell_metadata.tsv")
        write_metadata_qc(metadata, extraction_dir.parent, overwrite=overwrite)
    except Exception as error:  # pragma: no cover - QC must never break extraction
        print(f"WARNING: could not write QC metadata tables: {error}", file=sys.stderr)



def _apply_gene_identity(output_dir: Path, gtf: Union[str, Path]):
    """Annotate ``rna_genes.tsv`` with an Ensembl ``gene_id`` and symbol ``common_id``.

    The downstream plumbing for this already existed and had simply never been
    fed: the R helper writes both columns, the aggregation reads them into
    ``GeneFeature`` and propagates them, which is why a ``log2_cpm.csv``
    produced without a genome still carries an empty ``common_id``. Filling the
    file here makes every later stage correct with no further change.

    UPDATES TWO COLUMNS IN PLACE; TOUCHES NOTHING ELSE. The first version of
    this rewrote the file with only ``gene_id`` and ``common_id`` and destroyed
    the rest, which broke a later run and silently disabled
    ``--top-variable-features`` (``global_pca`` ranks on the
    ``variance_standardized`` column this deleted). The R helper's table is the
    authority on its own columns; this function is a visitor to two of them.

    ``feature_name`` is never written here. It is the row's identity, it is what
    ``rna_genes_order.tsv`` records, and it is the input this resolution reads --
    so this function is idempotent, and rerunning it can never make the table
    disagree with the matrix.

    The match rate is printed because it is the only signal that the wrong
    genome was passed -- a human object mapped against a mouse GTF does not
    fail, it quietly matches almost nothing.
    """
    import csv as _csv

    from paralome.gene_identity import resolve_gene_identities

    genes_path = output_dir / "rna_genes.tsv"
    with genes_path.open(newline="") as handle:
        reader = _csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        original = list(reader)
    if "feature_name" not in fieldnames:
        raise RuntimeError(
            f"{genes_path} has no feature_name column. That column is the row's "
            "identity and is written by extract_seurat_assays.R; a table without "
            "it comes from an extraction predating this contract and must be "
            "re-extracted rather than annotated."
        )
    names = [str(row.get("feature_name", "")).strip() for row in original]

    resolved, report = resolve_gene_identities(names, gtf)
    for line in report.summary_lines():
        print(line, flush=True)

    for row, identity in zip(original, resolved):
        row["gene_id"] = identity["gene_id"]
        row["common_id"] = identity["common_id"]

    with atomic_text_writer(genes_path, newline="") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(original)
    return report


def run_seurat_extraction(
    rds: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    assay: Optional[str] = None,
    slot: str = "counts",
    rna_assay: Optional[str] = None,
    rna_slot: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
    gtf: Optional[Union[str, Path]] = None,
    rscript: str = "Rscript",
    r_script_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    require_existing: bool = False,
    require: Sequence[str] = (),
    required_columns: Sequence[str] = (),
    write_qc: bool = True,
    cell_export_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Extract whichever assays a Seurat .rds object has, via the bundled R helper.

    Every extraction of an object with an RNA half also writes
    ``<rds stem>_cell_expression_embeddings.zip`` (singlecell.cell_export) into
    ``cell_export_dir`` -- by default ``<parent>/omicview``, beside the folder
    holding the extraction (the parent project's data/). A reused extraction
    rebuilds it only when it is missing or older than the extraction; nothing
    re-extracts an object for it. A pipeline run names its own omicview folder
    (singlecell-samples --cell-export-dir), so the run's export is the only one
    written.

    ``rds`` may be a folder of objects. Each is extracted
    by this function on its own and the results are stacked
    (``singlecell.rds_input``); ``required_columns`` -- --replicate-column and
    --stratify-by -- are what every object in a folder must carry.
    ``write_qc`` is off for a folder's members: the QC tables describe what the
    run analyses, which is the stacked extraction.

    ATAC does not always come with RNA, nor RNA with ATAC: the R helper
    discovers both halves and exports each if and only if it is present;
    ``assay``/``rna_assay`` only break ties.

    ``require`` is the caller stating what it NEEDS, which is a different
    question from what the object HAS. `paralome scRNA` requires ``("rna",)``,
    so pointing it at an ATAC-only object fails -- naming the assays that were
    actually found, rather than mid-pipeline on a missing file. Passing nothing
    accepts whatever is there. Because both halves are always extracted when
    present, an ATAC run and an RNA run against one object produce the SAME
    outputs.

    ``require_existing``: reuse an extraction or fail; never run R. For the QC
    and PCA commands, which explore an extraction rather than produce one.

    Idempotent: if a prior extraction checkpoint exists and matches the
    source .rds identity plus the requested assay/slot(s), the R subprocess
    is skipped and the existing outputs are reused (unless ``overwrite``).

    RNA and ATAC are exported in one R subprocess call, sharing one
    ``cells.tsv``/``cell_metadata.tsv`` -- one ``readRDS``, not two. Passing
    ``rna_assay`` exports that assay's gene-level counts (``rna_counts.plmsp``/
    ``rna_genes.tsv``/``rna_genes_order.tsv``).
    """
    if Path(str(rds)).expanduser().is_dir():
        from paralome.singlecell.rds_input import run_folder_extraction

        return run_folder_extraction(
            rds, out_dir, required_columns=required_columns, overwrite=overwrite,
            require_existing=require_existing, assay=assay, slot=slot,
            rna_assay=rna_assay, rna_slot=rna_slot, gene_symbol_column=gene_symbol_column,
            gtf=gtf, rscript=rscript, r_script_path=r_script_path, require=require,
            cell_export_dir=cell_export_dir,
        )
    rds_path = Path(rds).expanduser().resolve()
    if not rds_path.is_file():
        raise ParalomeError(104, f"Seurat object not found: {rds_path}")
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Flat, like every unmoved stage -- but resolved through the
    # shared helper anyway, so there is exactly one way a stage checkpoint
    # location is decided in this package.
    checkpoint_path = stage_checkpoint_path(output_dir, "singlecell_extract")

    rds_identity = _file_identity_cached(rds_path, checkpoint_path)
    # rna_assay/rna_slot/gene_symbol_column deliberately left OUT of the signature, so an RNA run
    # and an ATAC run pointed at the same --out compute the same signature for the same object. The
    # signature identifies the extracted OBJECT (rds identity + ATAC assay/slot + the R script
    # itself); which assays have been exported out of it is a question about which output FILES
    # exist, checked separately below. An RNA run against an ATAC-only extraction directory
    # re-extracts once (to add the rna_counts/rna_genes outputs, written in the same single readRDS
    # as the ATAC ones) and every subsequent run of either modality reuses it.
    def _signature_for(script_sha256: str) -> str:
        return _stable_digest({
            "rds_identity": rds_identity,
            "assay": assay,
            "slot": slot,
            "r_script_sha256": script_sha256,
        })

    signature = _signature_for(
        _sha256_file(Path(r_script_path) if r_script_path else _default_r_script_path()))
    # An extraction by an earlier script whose outputs this one only adds to is still this object's
    # extraction. Re-extracting is the user's choice (move the extraction aside), never automatic.
    reusable_signatures = {signature} | (
        set() if r_script_path else {_signature_for(sha) for sha in _OUTPUT_COMPATIBLE_R_SCRIPTS})
    from paralome.singlecell.cell_export import export_home

    export_dir = Path(cell_export_dir).expanduser().resolve() if cell_export_dir else export_home(output_dir.parent)

    def _cell_export(record: Mapping[str, object]) -> None:
        from paralome.singlecell.cell_export import build_cell_export, export_zip_path, is_current, rds_stem

        # A reuse-only caller (QC, PCA, the folder's pending check on the
        # login node) reads an extraction; it does not produce its export.
        # An object with an ATAC half only gets one too -- its cells' metadata and
        # embeddings, for OmicView's Cells viewer (there is no expression to draw).
        if require_existing or not any(_halves_present(record).values()):
            return
        if is_current(export_zip_path(export_dir, rds_stem(rds_path)), checkpoint_path):
            return
        build_cell_export(output_dir, export_dir, rds=rds_path)

    wanted = {str(name).lower() for name in require}
    unknown = wanted - {"atac", "rna"}
    if unknown:
        raise ValueError(f"require= accepts 'atac' and/or 'rna', not: {sorted(unknown)}")

    def _check_required(manifest: Mapping[str, object], where: str) -> None:
        """The caller needed a half the object does not have.

        Worth its own error because the alternative is a FileNotFoundError on
        rna_counts.mtx several stages later, which says nothing about why.
        """
        found = {half for half, present in _halves_present(manifest).items() if present}
        missing = wanted - found
        if not missing:
            return
        assays = manifest.get("available_assays") or manifest.get("assays")
        detail = f" Assays found: {', '.join(assays)}." if assays else ""
        raise RuntimeError(
            f"{where} has no {'/'.join(sorted(missing))} assay, "
            f"which this command requires.{detail}\n"
            f"Extracted: {', '.join(sorted(found)) or 'nothing'}.\n"
            f"If the assay is present under a different name, pass "
            f"{'--assay' if 'atac' in missing else '--rna-assay'} to name it."
        )

    # A reuse-only caller (require_existing) always takes the reuse
    # path; its --overwrite is for the aggregation, not the extraction.
    if (not overwrite or require_existing) and checkpoint_path.is_file():
        existing = json.loads(checkpoint_path.read_text())
        # Deliberately does NOT version-guard this one, and it is the
        # only ATAC reuse guard exempted.
        #
        # Seurat extraction writes into `data/extraction`, which is shared: the
        # `rna_assay` branch above adds rna_counts.mtx/rna_genes.tsv to the
        # expected outputs, so this stage feeds the RNA side too. A user can
        # delete the ATAC folders while keeping `data/` and the RNA folders. A
        # version stamp here would refuse that kept extraction and force it to
        # be recomputed --
        # invalidating shared, still-valid data to solve an ATAC checkpoint
        # problem that never involved this stage.
        #
        # The writer above still stamps, so the provenance is recorded and a
        # guard can be added the day extraction itself changes shape.
        if existing.get("complete") and existing.get("signature") in reusable_signatures:
            # A prior run recorded what it wrote. That record, not a fresh guess, is what "still
            # complete" means, since the file set depends on the object rather than on the arguments.
            previous = list(existing.get("outputs") or {}) or list(_EXTRACTION_OUTPUTS)
            outputs_present = all((output_dir / name).is_file() for name in previous)
            # The RNA export is parameterized, so reusing it requires the
            # stored parameters to match what is being asked for now -- not
            # merely that the files exist.
            rna_compatible = (not rna_assay) or (
                str(existing.get("rna_assay") or "") == str(rna_assay)
                and str(existing.get("rna_slot") or "") == str(rna_slot or slot)
                and str(existing.get("gene_symbol_column") or "") == str(gene_symbol_column or "")
            )
            # --rna-slot / --gene-symbol-column given without
            # --rna-assay are compared too, whenever the extraction has an RNA half.
            if rna_compatible and not rna_assay and _halves_present(existing)["rna"]:
                if rna_slot and str(existing.get("rna_slot") or "") != str(rna_slot):
                    rna_compatible = False
                if gene_symbol_column and str(existing.get("gene_symbol_column") or "") != str(gene_symbol_column):
                    rna_compatible = False
            if outputs_present and rna_compatible:
                _check_required(existing, str(rds_path))
                if write_qc:
                    _write_extraction_qc(existing, output_dir, overwrite=False)
                _cell_export(existing)
                return existing

    # The PCA is for exploring the PCA space with different groupings, so it
    # belongs in the QC subroutine rather than far into the analysis.
    #
    # An exploratory tool must never START one of these. Extraction reads a
    # multi-gigabyte rds, takes tens of minutes, and produces the annotated
    # gene table every later stage depends on -- work that belongs to a pipeline
    # stage the user asked for, not a side effect of asking to look at a plot.
    #
    # Refusing here rather than in each caller means no caller can forget, and
    # it also removes the missing---genome hazard by construction: a command
    # that cannot extract cannot write a gene table without gene identities.
    if require_existing:
        raise FileNotFoundError(
            f"No usable extraction in {output_dir}.\n"
            f"This command reads an existing extraction and will not create one -- "
            f"extraction re-reads the .rds and is a pipeline stage, not a side effect "
            f"of a QC command.\n"
            f"Run `paralome scRNA samples --project <project> --genome <genome> ...` "
            f"(or `paralome snATAC samples`) first, then re-run this against the same "
            f"--project."
        )

    script_path = Path(r_script_path).expanduser().resolve() if r_script_path else _default_r_script_path()
    if not script_path.is_file():
        raise FileNotFoundError(f"Seurat extraction R script not found: {script_path}")

    # Say WHY before R says "unknown input format" four frames down.
    misformatted = describe_rds_format(rds_path)
    if misformatted:
        raise RuntimeError(f"Seurat extraction cannot read {rds_path}:\n{misformatted}")

    argv = [rscript, str(script_path), "--rds", str(rds_path), "--out", str(output_dir), "--slot", slot]
    if assay:
        argv.extend(["--assay", assay])
    if rna_assay:
        argv.extend(["--rna-assay", rna_assay])
        argv.extend(["--rna-slot", rna_slot or slot])
    elif rna_slot:
        # --rna-slot reaches R even when the RNA assay is auto-detected.
        argv.extend(["--rna-slot", rna_slot])
    if gene_symbol_column:
        argv.extend(["--gene-symbol-column", gene_symbol_column])

    # Error reports from an earlier failed run must not be raised
    # for this run's failure.
    for stale_report in ("assay_error.json", "extraction_error.json"):
        try:
            (output_dir / stale_report).unlink()
        except FileNotFoundError:
            pass
    completed = _run_rscript_streaming(argv, output_dir)
    if completed.returncode != 0:
        # The R helper writes assay_error.json when it is an assay it could not resolve, so that
        # failure arrives as its own number with the assay list attached, rather than as PLM-0001
        # wrapping R's stderr.
        _raise_assay_error(output_dir)
        # The general case. R writes extraction_error.json with a
        # registered code for any failure it can name; anything it cannot name
        # still falls through to the message below.
        _raise_extraction_error(output_dir)
        raise RuntimeError(
            "Seurat extraction failed "
            f"(rscript={rscript!r}, exit={completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    manifest_path = output_dir / "extraction_manifest.json"
    extraction_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    halves = _halves_present(extraction_manifest)
    has_atac, has_rna = halves["atac"], halves["rna"]
    expected_outputs = _outputs_for(has_atac, has_rna, output_dir)

    missing = [name for name in expected_outputs if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "Seurat extraction did not produce expected output(s): " + ", ".join(missing)
        )

    _check_required(extraction_manifest, str(rds_path))

    # Use the registered genomes and GTFs: Paralome should know the genome as
    # a general rule.
    #
    # Resolved HERE -- once per dataset, immediately after the R helper wrote
    # the gene table and before anything downstream has read it. The shared
    # data/ folder exists so one rds serves every comparison; doing this at
    # report time would re-parse the GTF once per stratum per contrast, and two
    # contrasts run with different genomes would silently disagree about what a
    # gene is. Done in Python rather than in R, keeping the established split
    # where R only reads the object and all annotation logic stays this side.
    identity_report = None
    if gtf is not None and (output_dir / "rna_genes.tsv").is_file():
        identity_report = _apply_gene_identity(output_dir, gtf)

    result: Dict[str, object] = {
        "schema": SC_EXTRACT_SCHEMA,
        "stage": "singlecell_extract",
        "complete": True,
        "signature": signature,
        "rds": str(rds_path),
        "rds_identity": rds_identity,
        "assay": extraction_manifest.get("assay", assay),
        "has_atac": has_atac,
        "has_rna": has_rna,
        "slot": slot,
        "n_cells": extraction_manifest.get("n_cells"),
        "n_peaks": extraction_manifest.get("n_peaks"),
        "seurat_version": extraction_manifest.get("seurat_version"),
        "signac_version": extraction_manifest.get("signac_version"),
        "metadata_columns": extraction_manifest.get("metadata_columns"),
        "outputs": {name: str(output_dir / name) for name in expected_outputs},
        "output_sha256": {
            name: _sha256_file(output_dir / name) for name in expected_outputs
        },
        "gene_identity": identity_report.to_dict() if identity_report else None,
        "rna_assay": extraction_manifest.get("rna_assay", rna_assay),
        "rna_slot": extraction_manifest.get("rna_slot", rna_slot),
        "n_genes": extraction_manifest.get("n_genes"),
        "gene_symbol_column": extraction_manifest.get("gene_symbol_column", gene_symbol_column),
        "stdout": completed.stdout,
    }
    with exclusive_checkpoint_lock(checkpoint_path):
        _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    # An extraction actually ran, so the metadata tables are a fresh export of
    # a (possibly new) object -- overwrite whatever was there.
    if write_qc:
        _write_extraction_qc(result, output_dir, overwrite=True)
    _cell_export(result)
    return result


def read_cell_metadata(path: Union[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "barcode" not in frame.columns:
        raise ValueError(f"cell_metadata.tsv is missing the required barcode column: {path}")
    return frame.set_index("barcode", drop=False)


#: In-process cache for `load_counts_matrix`, keyed by (path, size, mtime_ns).
#: Cleared only by process exit -- these are read-only inputs for the life of a
#: run, and a stage that re-reads the same extraction nine times is the case
#: this exists for.
_MATRIX_MEMO: Dict[Tuple[str, int, int], "csc_matrix"] = {}


#: Header of the binary sparse format written by extract_seurat_assays.R.
#: 8-byte magic, then nrow/ncol/nnz/indptr_n as float64, then 24 reserved bytes.
_PLMSP_MAGIC = b"PLMSPR01"
_PLMSP_HEADER = 64


def _read_plmsp(path: Path) -> "csc_matrix":
    """Read the binary sparse matrix written by the R extraction.

    The file holds exactly scipy's CSC arrays, so this is three ``frombuffer``
    calls over one ``read()`` -- no parsing, and no copy beyond the read itself.

    EVERY FIELD IS CHECKED against the file's own length before use. A truncated
    write -- a full disk, a killed job, an NFS hiccup -- would otherwise produce
    a matrix that is silently short, and a counts matrix missing its tail is the
    kind of wrong that survives all the way into a p-value. Anything
    inconsistent raises, and the caller falls back to the .mtx if one exists.
    """
    import numpy as np
    from scipy.sparse import csc_matrix

    raw = path.read_bytes()
    if len(raw) < _PLMSP_HEADER or raw[:8] != _PLMSP_MAGIC:
        raise ValueError(f"{path.name} does not carry the PLMSPR01 header")

    nrow, ncol, nnz, indptr_n = (
        int(v) for v in np.frombuffer(raw, dtype="<f8", count=4, offset=8)
    )
    if indptr_n != ncol + 1:
        raise ValueError(
            f"{path.name}: header says {indptr_n} indptr entries for {ncol} "
            f"columns; expected {ncol + 1}"
        )

    expected = _PLMSP_HEADER + 8 * indptr_n + 4 * nnz + 8 * nnz
    if len(raw) != expected:
        raise ValueError(
            f"{path.name} is {len(raw)} bytes, expected {expected} for "
            f"{nrow}x{ncol} with {nnz} stored values -- truncated or corrupt"
        )

    offset = _PLMSP_HEADER
    indptr = np.frombuffer(raw, dtype="<f8", count=indptr_n, offset=offset).astype(np.int64)
    offset += 8 * indptr_n
    indices = np.frombuffer(raw, dtype="<i4", count=nnz, offset=offset)
    offset += 4 * nnz
    data = np.frombuffer(raw, dtype="<f8", count=nnz, offset=offset)

    if indptr[-1] != nnz:
        raise ValueError(
            f"{path.name}: indptr ends at {indptr[-1]} but the file carries "
            f"{nnz} values"
        )
    return csc_matrix((data, indices, indptr), shape=(nrow, ncol))


def load_counts_matrix(mtx_path: Union[str, Path]) -> "csc_matrix":
    """The counts matrix, from a binary cache when one is valid.

    ``mmread`` parses Matrix Market TEXT, single-threaded, and a real
    ``rna_counts.mtx`` can approach 1 GB -- minutes per read, once per stratum
    per job. Two layers, both keyed on the SOURCE file's identity:

    1. an in-process memo, so the strata in one job share one matrix;
    2. a ``.npz`` sidecar written next to the ``.mtx`` on first read, so every
       later process, job and array task loads binary instead of text.

    ON CORRECTNESS, which matters more here than the speed. A stale cache would
    silently feed the wrong counts to every model -- far worse than being slow.
    So the sidecar records the source's size and mtime_ns and is used ONLY when
    both still match; any mismatch, any unreadable metadata, any malformed npz
    falls through to ``mmread``. The cache is never trusted on its own
    existence.

    Failure to WRITE the cache is not an error: a read-only extraction
    directory should be slow, not broken.
    """
    import json as _json

    source = Path(mtx_path).expanduser().resolve()

    # Prefer the binary file the extraction writes: the .plmsp file IS the CSC arrays, so there is
    # nothing to encode, parse or convert.
    #
    # The .mtx path below is kept for extraction directories written in text. An old extraction must
    # not become unreadable because the writer improved.
    binary = source.with_suffix(".plmsp")
    if binary.is_file():
        source = binary

    stat = source.stat()
    key = (str(source), stat.st_size, stat.st_mtime_ns)
    memo = _MATRIX_MEMO.get(key)
    if memo is not None:
        return memo

    if source.suffix == ".plmsp":
        matrix = _read_plmsp(source)
        _MATRIX_MEMO[key] = matrix
        return matrix

    cache_path = source.with_suffix(source.suffix + ".npz")
    meta_path = source.with_suffix(source.suffix + ".npz.json")
    matrix: Optional["csc_matrix"] = None
    if cache_path.is_file() and meta_path.is_file():
        try:
            meta = _json.loads(meta_path.read_text())
            if (
                int(meta.get("source_size", -1)) == stat.st_size
                and int(meta.get("source_mtime_ns", -1)) == stat.st_mtime_ns
            ):
                from scipy.sparse import load_npz

                matrix = load_npz(str(cache_path)).tocsc()
        except Exception:  # noqa: BLE001
            # A corrupt or half-written cache must never be fatal, and must
            # never be used. Fall through and re-read the authoritative text.
            matrix = None

    if matrix is None:
        matrix = mmread(str(source)).tocsc()
        try:
            from scipy.sparse import save_npz

            # Temp + rename: several array tasks may reach this at once, and a
            # half-written .npz that another task then reads is exactly the
            # silent-wrong-data failure this whole function is trying to avoid.
            # Ends in .npz so scipy writes exactly this path rather than
            # appending the extension itself, which would leave the rename
            # pointing at a file that does not exist.
            tmp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
            save_npz(str(tmp), matrix)
            os.replace(str(tmp), str(cache_path))
            meta_tmp = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
            meta_tmp.write_text(_json.dumps({
                "source": str(source),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            }) + "\n")
            os.replace(str(meta_tmp), str(meta_path))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[paralome] could not cache {source.name} as .npz ({exc}); "
                f"continuing with the text matrix. Reads stay slow.",
                file=sys.stderr,
            )

    _MATRIX_MEMO[key] = matrix
    return matrix


def read_barcode_list(path: Union[str, Path]) -> List[str]:
    with Path(path).open() as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


@dataclass(frozen=True)
class SinglecellInputs:
    peaks: List[AtlasPeak]
    matrix: csc_matrix  # peaks x cells, matches `cells` column order and `peaks` row order
    cells: List[str]
    metadata: pd.DataFrame  # indexed by barcode, row order matches `cells`
    mitochondrial_peaks_excluded: int = 0


def load_singlecell_inputs(extraction_dir: Union[str, Path]) -> SinglecellInputs:
    extraction_dir = Path(extraction_dir).expanduser().resolve()
    peaks = read_atlas(extraction_dir / "peaks.bed")
    peak_order_check = read_barcode_list(extraction_dir / "peaks_order.tsv")
    if [peak.peak_id for peak in peaks] != peak_order_check:
        raise RuntimeError(
            "peaks.bed and peaks_order.tsv disagree on peak order -- Seurat extraction is inconsistent"
        )
    cells = read_barcode_list(extraction_dir / "cells.tsv")
    # Same binary cache and per-process memo as the RNA loader.
    matrix = load_counts_matrix(extraction_dir / "counts.mtx")
    if matrix.shape != (len(peaks), len(cells)):
        raise RuntimeError(
            f"counts.mtx shape {matrix.shape} does not match "
            f"{len(peaks)} peaks x {len(cells)} cells"
        )
    metadata = read_cell_metadata(extraction_dir / "cell_metadata.tsv")
    metadata = metadata.loc[cells]
    if list(metadata["barcode"]) != cells:
        raise RuntimeError("cell_metadata.tsv row order does not match cells.tsv")
    # ChrM peaks are left out, as in bulk quantification
    # (quantification.ATLAS_FILTER), so they are neither tested nor part of a
    # pseudobulk sample's or a cell's library size.
    nuclear = [index for index, peak in enumerate(peaks) if not is_mitochondrial_contig(peak.chrom)]
    excluded = len(peaks) - len(nuclear)
    if excluded:
        if not nuclear:
            raise RuntimeError(f"The extraction in {extraction_dir} has no peaks off the mitochondrial genome")
        peaks = [peaks[index] for index in nuclear]
        matrix = csc_matrix(matrix[nuclear, :])
    return SinglecellInputs(
        peaks=peaks, matrix=matrix, cells=cells, metadata=metadata,
        mitochondrial_peaks_excluded=excluded,
    )


_MAX_VALUE_COUNTS_REPORTED = 25


def summarize_metadata_columns(inputs: SinglecellInputs) -> Dict[str, object]:
    """Describe every cell-metadata column so the user can pick --replicate-column/--stratify-by/--factor etc.

    Nothing prompts interactively (Paralome runs unattended on SLURM). The user
    picks --replicate-column/--factor COLUMN:CONTROL/--stratify-by explicitly
    on the CLI, exactly like every other `atac` subcommand's --factor. This
    function exists so the user isn't guessing blind at column names or the
    exact spelling of a condition's values in an object they may not have
    generated themselves: it surfaces every metadata column, how many distinct
    values it has, and (for low-cardinality columns -- the ones usable as
    --factor/--replicate-column/--stratify-by) the value counts, so the right
    control-level spellings and column names are visible before committing to
    a full pseudobulk run.
    """
    columns: List[Dict[str, object]] = []
    for column in inputs.metadata.columns:
        if column == "barcode":
            continue
        series = inputs.metadata[column]
        value_counts = series.value_counts()
        entry: Dict[str, object] = {
            "column": column,
            "distinct_values": int(series.nunique()),
        }
        if len(value_counts) <= _MAX_VALUE_COUNTS_REPORTED:
            entry["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
            entry["likely_grouping_column"] = len(value_counts) < len(series)
        else:
            entry["value_counts"] = None
            entry["likely_grouping_column"] = False
            entry["sample_values"] = [str(v) for v in series.unique()[:5]]
        columns.append(entry)
    return {
        "n_cells": len(inputs.cells),
        "n_peaks": len(inputs.peaks),
        "columns": sorted(columns, key=lambda entry: entry["column"]),
    }


def run_singlecell_inspect(
    rds: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    assay: Optional[str] = None,
    slot: str = "counts",
    genome: Optional[str] = None,
    gtf: Optional[Union[str, Path]] = None,
    rscript: str = "Rscript",
    r_script_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Extract (or reuse an already-extracted) object and report its metadata schema.

    Meant to be run once, interactively, before choosing --replicate-column/
    --stratify-by/--factor for
    `singlecell-samples`/`singlecell-differential` -- it shares the same
    extraction checkpoint, so this does not force a redundant R invocation
    if `singlecell-samples` is run against the same --out afterward.
    """

    # One rule for genome -> GTF, the same one run_singlecell_atac uses:
    # every pipeline asks for --genome rather than --gtf.
    resolved_gtf: Optional[Union[str, Path]] = gtf
    if genome:
        resolved_gtf = resolve_genome_resources(
            genome=genome, overrides={"gtf": gtf}
        ).path("gtf")

    extraction_dir = layout_extraction_dir(Path(out_dir).expanduser().resolve())
    # NOT read-only, unlike the PCA -- deliberately. QC is the BOOTSTRAP step: the user runs it to
    # discover which metadata columns exist so they can choose --replicate-column/--factor, which
    # they need before they can run pseudobulk at all. Requiring an existing extraction here would
    # be a deadlock.
    #
    # It is safe for it to extract because it takes --genome and resolves the GTF above, so the
    # extraction it produces is annotated and is exactly the one the pipeline will go on to reuse.
    extraction = run_seurat_extraction(
        rds, extraction_dir, require=("atac",), assay=assay, slot=slot, rscript=rscript,
        r_script_path=r_script_path, overwrite=overwrite, gtf=resolved_gtf,
    )
    inputs = load_singlecell_inputs(extraction_dir)
    summary = summarize_metadata_columns(inputs)
    return {"extraction": extraction, **summary}


def _constant_value(series: pd.Series) -> Optional[str]:
    unique = series.unique()
    if len(unique) == 1:
        return str(unique[0])
    return None


def aggregate_pseudobulk(
    inputs: SinglecellInputs,
    *,
    group_by: Sequence[str],
    stratify_by: Optional[str] = None,
    min_cells_per_group: int = 1,
    remove: Optional[Sequence[str]] = None,
    replicate_column: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]], Dict[str, object]]:
    """Group cells into pseudobulk samples and sum their peak counts.

    Returns ``(raw_counts, manifest_rows, aggregation_report)`` where
    ``raw_counts`` maps ``sample_id -> {peak_id: count}`` and
    ``manifest_rows`` is ready to hand to ``write_pseudobulk_outputs``.

    The actual grouping/summing logic lives in
    ``paralome.singlecell.pseudobulk_core.aggregate_pseudobulk_core`` (shared
    with the RNA side -- see ``paralome.rna.singlecell.aggregate_rna_pseudobulk``);
    this is a thin wrapper unpacking ATAC's ``SinglecellInputs``.

    ``remove``: pseudobulk sample_id(s) to exclude outright (e.g. outliers
    seen in the PCAs; wired all the way through to pipeline run).
    """
    peak_ids = [peak.peak_id for peak in inputs.peaks]
    return aggregate_pseudobulk_core(
        inputs.matrix, peak_ids, inputs.metadata, inputs.cells,
        sanitize_sample_id=sanitize_sample_id,
        group_by=group_by,
        stratify_by=stratify_by, min_cells_per_group=min_cells_per_group,
        remove_samples=remove, replicate_column=replicate_column,
    )


def write_pseudobulk_outputs(
    peaks: Sequence[AtlasPeak],
    raw_counts: Mapping[str, Mapping[str, float]],
    manifest_rows: Sequence[Mapping[str, object]],
    out_dir: Union[str, Path],
    *,
    gtf: Optional[Union[str, Path]] = None,
    tss_upstream: int = 2000,
    tss_downstream: int = 500,
    pseudocount: float = 1.0,
    group_by: Optional[Sequence[str]] = None,
    dropped_small_groups: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[str, object]:
    # Log2(CPM + 0) has no value for a zero-count peak.
    check_bounded_number(pseudocount, flag="--pseudocount", minimum=0, exclusive=True)
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if gtf is not None:
        annotation_rows = annotate_peaks(peaks, load_tss(gtf), tss_upstream, tss_downstream)
    else:
        annotation_rows = [
            {"peak_id": peak.peak_id, "chrom": peak.chrom, "start": peak.start,
             "end": peak.end, "width": peak.end - peak.start}
            for peak in peaks
        ]

    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    peak_ids = [row["peak_id"] for row in annotation_rows]

    cpm: Dict[str, Dict[str, float]] = {}
    log2_cpm: Dict[str, Dict[str, float]] = {}
    for sample_id in sample_ids:
        counts = raw_counts[sample_id]
        library_size = sum(counts.values())
        cpm[sample_id] = {
            peak_id: (counts[peak_id] / library_size * 1_000_000.0 if library_size else 0.0)
            for peak_id in peak_ids
        }
        log2_cpm[sample_id] = {
            peak_id: math.log2(cpm[sample_id][peak_id] + pseudocount)
            for peak_id in peak_ids
        }

    raw_path = output_dir / "raw_counts.csv"
    cpm_path = output_dir / "cpm.csv"
    log_path = output_dir / "log2_cpm.csv"
    manifest_path = output_dir / "sample_manifest.csv"

    _write_matrix(raw_path, annotation_rows, sample_ids, raw_counts)
    _write_matrix(cpm_path, annotation_rows, sample_ids, cpm)
    _write_matrix(log_path, annotation_rows, sample_ids, log2_cpm)
    # Sample manifests elsewhere in the ATAC pipeline are read as generic CSV
    # via csv.DictReader; write comma-delimited for direct drop-in
    # compatibility with `atac differential --samples`.
    # Atomic, like the matrices above.
    with atomic_text_writer(manifest_path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    result: Dict[str, object] = {
        "raw_counts": str(raw_path),
        "cpm": str(cpm_path),
        "log2_cpm": str(log_path),
        "sample_manifest": str(manifest_path),
        "peaks": len(peak_ids),
        "samples": len(sample_ids),
    }

    # A table of how many cells there are per sample, to help identify
    # outliers before starting a project. Written
    # whenever the caller has group_by (every real pseudobulk run does) --
    # optional purely so any other historical caller of this function that
    # doesn't pass it isn't affected.
    if group_by:
        cell_counts_path = output_dir / "cell_counts_per_sample.csv"
        write_cell_counts_table(manifest_rows, dropped_small_groups or [], group_by, cell_counts_path)
        result["cell_counts_per_sample"] = str(cell_counts_path)

    return result


def run_singlecell_atac(
    rds: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    group_by: Sequence[str],
    stratify_by: Optional[str] = None,
    assay: Optional[str] = None,
    slot: str = "counts",
    genome: Optional[str] = None,
    gtf: Optional[Union[str, Path]] = None,
    tss_upstream: int = 2000,
    tss_downstream: int = 500,
    pseudocount: float = 1.0,
    min_cells_per_group: int = 10,
    rscript: str = "Rscript",
    r_script_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    remove: Optional[Sequence[str]] = None,
    replicate_column: Optional[str] = None,
    require_existing_extraction: bool = False,
    aggregate_root: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Full single-cell-ATAC-to-pseudobulk pipeline: extract, then aggregate.

    ``remove``: pseudobulk sample_id(s) to exclude outright (e.g. outliers
    seen in the PCAs; wired all the way through to pipeline run) -- identify them from a prior run's
    ``cell_counts_per_sample.csv``/global-PCA plot. Included in this stage's
    checkpoint plan_id, so changing ``--remove`` invalidates a prior
    aggregation and reruns it.

    Peak annotation resolves its GTF from ``--genome`` through the same genome
    catalog motif analysis uses (resolve_genome_resources) -- an explicit
    ``gtf`` still wins as an override, same as ``resolve_motif_resources``'s
    reference_fasta/motif_database overrides.
    """
    # Refuse before the (long) extraction, not in the writer after it.
    check_bounded_number(pseudocount, flag="--pseudocount", minimum=0, exclusive=True)
    resolved_gtf: Optional[Union[str, Path]] = gtf
    if genome:
        # Only hit the genome catalog when --genome was actually supplied --
        # a plain --gtf-only call (no genome catalog configured at all) must
        # keep working exactly as it did before, with zero catalog dependency.
        genome_resources = resolve_genome_resources(genome=genome, overrides={"gtf": gtf})
        resolved_gtf = genome_resources.path("gtf")

    # `out_dir` is the SHARED data folder (the extraction lives directly under it, since a combined
    # multiome RDS's extraction can be shared with an `rna singlecell-samples` run against the same
    # --out -- see run_seurat_extraction's --rna-assay passthrough). This function's own aggregation
    # output (raw_counts/cpm/log2_cpm/sample_manifest + its checkpoint) is ATAC-specific, so it nests
    # under `atac/` rather than the shared pseudobulk root: a folder layer for RNA and ATAC analyses
    # keeps the project folder down to major directories.
    output_dir = Path(out_dir).expanduser().resolve()
    extraction_dir = layout_extraction_dir(output_dir)
    extraction = run_seurat_extraction(
        # The ATAC entry point requires peaks, and says so here rather than
        # letting a missing counts.mtx surface two stages downstream. An RNA
        # half, if the object has one, is extracted alongside and ignored.
        rds, extraction_dir, require=("atac",), assay=assay, slot=slot, rscript=rscript,
        r_script_path=r_script_path, overwrite=overwrite, gtf=resolved_gtf,
        require_existing=require_existing_extraction,
        required_columns=[*group_by, *stratum_columns(stratify_by)],   # Several
    )

    # See rna/singlecell.py: the aggregation half's root, redirected by the PCA
    # so an exploratory grouping cannot overwrite the analysis matrices or the
    # pipeline stage's own checkpoint. Defaults to the shared data folder.
    aggregate_dir = (
        Path(aggregate_root).expanduser().resolve() if aggregate_root else output_dir
    )
    atac_dir = aggregate_dir / "atac"
    # See rna/singlecell.py -- one resolver shared with the reader.
    checkpoint_path = stage_checkpoint_path(aggregate_dir, "singlecell_atac")
    print(
        stage_banner(
            "singlecell_atac", derive_file_prefix(aggregate_dir), aggregate_dir,
        ),
        flush=True,
    )
    plan_id = _stable_digest({
        "extraction_signature": extraction["signature"],
        "group_by": list(group_by),
        "stratify_by": stratify_by,
        "min_cells_per_group": min_cells_per_group,
        "pseudocount": pseudocount,
        "tss_upstream": tss_upstream,
        "tss_downstream": tss_downstream,
        "genome": genome,
        "gtf": str(Path(resolved_gtf).resolve()) if resolved_gtf else None,
        "remove": sorted(str(v) for v in remove) if remove else [],
        # ChrM peaks are excluded (load_singlecell_inputs).
        "atlas_filter": ATLAS_FILTER,
        # Groups with an empty sample value are dropped, not kept
        # as an "unlabeled" sample.
        "empty_sample": "dropped.v1",
    })

    # An existing checkpoint is READ back first, so a rerun with nothing to redo rewrites nothing.
    # The whole check-then-aggregate-then-write section runs inside the same
    # exclusive_checkpoint_lock as the checkpoint write below, so concurrent comparisons sharing one
    # data directory (a SLURM array, say) serialize on this stage instead of racing each other to
    # recompute and overwrite the same files.
    with exclusive_checkpoint_lock(checkpoint_path):
        if not overwrite and checkpoint_path.is_file():
            existing = json.loads(checkpoint_path.read_text())
            if (
                existing.get("complete")
                and existing.get("plan_id") == plan_id
                and version_stale_reason(existing) is None
                and all(
                    Path(str(existing.get(key, ""))).is_file()
                    for key in ("raw_counts", "cpm", "log2_cpm", "sample_manifest")
                )
                and _pseudobulk_outputs_match(existing)
                and existing.get("method") in (None, SC_PSEUDOBULK_METHOD)
            ):
                if existing.get("method") is None:
                    # Same plan (atlas_filter and empty_sample are
                    # in it), written before the method was recorded: the
                    # outputs are this method's, so the record says so.
                    existing["method"] = SC_PSEUDOBULK_METHOD
                    _atomic_json(checkpoint_path, existing)
                existing["checkpoint"] = str(checkpoint_path)
                return existing

        # Withdrawn before the shared files are rewritten, so a run
        # that dies part way cannot leave the old checkpoint vouching for them.
        if checkpoint_path.is_file():
            _atomic_json(checkpoint_path, {
                "schema": SC_PSEUDOBULK_SCHEMA,
                "stage": "singlecell_atac",
                "complete": False,
                "plan_id": plan_id,
                "reason": "being recomputed",
            })

        inputs = load_singlecell_inputs(extraction_dir)
        raw_counts, manifest_rows, aggregation_report = aggregate_pseudobulk(
            inputs, group_by=group_by,
            stratify_by=stratify_by, min_cells_per_group=min_cells_per_group,
            remove=remove, replicate_column=replicate_column,
        )
        written = write_pseudobulk_outputs(
            inputs.peaks, raw_counts, manifest_rows, atac_dir,
            gtf=resolved_gtf, tss_upstream=tss_upstream, tss_downstream=tss_downstream,
            pseudocount=pseudocount, group_by=group_by,
            dropped_small_groups=aggregation_report.get("dropped_small_groups"),
        )
        qc_run = _record_aggregation_qc(
            output_dir, "atac", written, group_by=group_by,
            stratify_by=stratify_by,
            min_cells_per_group=min_cells_per_group, remove=remove,
            plan_id=plan_id, n_samples=len(manifest_rows),
        )

        result: Dict[str, object] = {
            "schema": SC_PSEUDOBULK_SCHEMA,
            "stage": "singlecell_atac",
            "complete": True,
            "plan_id": plan_id,
            # Required by pipeline/state.py.
            "method": SC_PSEUDOBULK_METHOD,
            "extraction": extraction,
            "group_by": list(group_by),
            "stratify_by": stratify_by,
            "aggregation": aggregation_report,
            "mitochondrial_peaks_excluded": inputs.mitochondrial_peaks_excluded,
            **written,
            **qc_run,
            "output_sha256": {
                key: _sha256_file(Path(str(written[key])))
                for key in ("raw_counts", "cpm", "log2_cpm", "sample_manifest")
            },
        }
        _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    return result


def _pseudobulk_outputs_match(existing: Mapping[str, object]) -> bool:
    """The files still hold what the checkpoint recorded writing."""
    recorded = existing.get("output_sha256")
    if not isinstance(recorded, Mapping):
        return False
    try:
        return all(
            recorded.get(key) == _sha256_file(Path(str(existing.get(key, ""))))
            for key in ("raw_counts", "cpm", "log2_cpm", "sample_manifest")
        )
    except OSError:
        return False
