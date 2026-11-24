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

"""Single-cell RNA pseudobulk bridge.

Mirrors ``paralome.atac.singlecell`` for RNA: gene-level counts instead of
peak-level counts, from the SAME combined multiome Seurat object and the
SAME (now dual-assay) extraction step -- RNA and ATAC are exported in one
export rather than twice. ``run_singlecell_rna`` calls the exact same
``run_seurat_extraction`` ATAC's pseudobulk stage calls; whichever modality's
command runs first against a shared ``--out`` does the one combined R
extraction, the second reuses it via checkpoint.

Feature identity is deliberately minimal: only the Ensembl (master) ID and
the common (secondary) ID are needed for RNA analyses, not the full
annotation -- no GTF parsing, no chrom/start/end.
"""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import csv
import math
from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csc_matrix

from paralome.layout import extraction_dir as layout_extraction_dir
from paralome.stratify import stratum_columns
from paralome.report_schema import MISSING_VALUE
from paralome.atac.naming import derive_file_prefix, stage_banner
from paralome.atac.singlecell import (
    _record_aggregation_qc,
    load_counts_matrix,
    read_barcode_list,
    read_cell_metadata,
    run_seurat_extraction,
    sanitize_sample_id,
)
from paralome.checkpoints import (
    atomic_text_writer,
    atomic_write_text,
    exclusive_checkpoint_lock,
    stage_checkpoint_path,
)
from paralome.pipeline.jsonio import invalidate_checkpoint
from paralome.singlecell.pseudobulk_core import (
    aggregate_pseudobulk_core,
    write_cell_counts_table,
)

SC_RNA_PSEUDOBULK_SCHEMA = "paralome.rna.singlecell_rna.v1"
# The pseudobulk method: the date the aggregation last changed what it computes
# (empty samples dropped). Recorded in the checkpoint and required by
# pipeline/state.py: a checkpoint under another method reads out of date, so the
# pipeline aggregates again before the differential reads cpm.csv.
SC_RNA_PSEUDOBULK_METHOD = "2026-10-11"

# NOT a default, and deliberately still exported, as the first entry of the R
# helper's tiebreak order for objects that genuinely have several gene-level
# assays. RNA, Spatial and Xenium are all RNA analyses under different names (a
# Visium object has one assay, `Spatial`), and a Seurat user can name the assay
# anything, so the extraction does not hard-code it: an explicit --rna-assay
# would short-circuit the R helper's detection.
DEFAULT_RNA_ASSAY = "RNA"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    import json
    # Deliberately does NOT stamp here. The ATAC writers record the
    # paralome --version that wrote each checkpoint so a stage can refuse to
    # reuse one from another release; RNA is deliberately left as it was, so
    # existing RNA runs keep resolving, and the cheapest way to be sure a
    # change cannot disturb them is not to make it. If an RNA consumer
    # ever needs the stamp, stamp_version is already shared; add the call.
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stable_digest(payload: Mapping[str, object]) -> str:
    import hashlib
    import json
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gene_matrix(
    path: Path,
    metadata: Sequence[Mapping[str, object]],
    samples: Sequence[str],
    values: Mapping[str, Mapping[str, float]],
) -> None:
    """Same shape as ``atac.quantification._write_matrix``, keyed by
    ``feature_name`` instead of ``peak_id`` -- a separate small function rather
    than parameterizing the shared ATAC one, to avoid touching that stable
    module for a purely cosmetic column-header difference.
    """
    if not metadata:
        raise ValueError("Cannot write an empty gene matrix")
    fields = list(metadata[0].keys()) + list(samples)
    # Through atomic_text_writer (temp file, fsync, rename), so a re-aggregation killed by walltime or OOM
    # cannot leave a truncated cpm.csv where the differential reads it.
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in metadata:
            out = dict(row)
            feature_name = str(row["feature_name"])
            out.update({sample: values[sample][feature_name] for sample in samples})
            writer.writerow(out)


@dataclass(frozen=True)
class GeneFeature:
    """One row of the extracted gene table: an identity, plus annotation about it.

    ``feature_name`` is the assay rowname. It indexes the counts matrix, it is
    unique by construction, and it is the ONLY one of these three fields that
    may be used as a key.

    ``gene_id`` (Ensembl) and ``common_id`` (symbol) are what the GTF was able
    to say about that row, and either can be ``.`` -- on a real object of
    36,601 genes, roughly 30,000 resolved to accessions and several thousand
    to ``.``.
    Keying on ``gene_id`` therefore collapses every unresolved gene onto one
    entry, which does not raise: it silently reports one gene's numbers under
    thousands of names. That is the failure this separation exists to prevent,
    and it is why ``feature_name`` is a field rather than a convention.
    """

    feature_name: str
    gene_id: str = ""
    common_id: str = ""


@dataclass(frozen=True)
class SinglecellRnaInputs:
    genes: List[GeneFeature]
    matrix: csc_matrix  # genes x cells, matches `cells` column order and `genes` row order
    cells: List[str]
    metadata: pd.DataFrame  # indexed by barcode, shared with the ATAC side


def load_singlecell_rna_inputs(extraction_dir: Union[str, Path]) -> SinglecellRnaInputs:
    """Load the RNA half of a (possibly dual-assay) extraction directory.

    ``cells.tsv``/``cell_metadata.tsv`` are the SAME shared files the ATAC
    loader (``paralome.atac.singlecell.load_singlecell_inputs``) reads: both
    modalities read one shared cell/metadata table rather than each deriving
    their own.
    """
    extraction_dir = Path(extraction_dir).expanduser().resolve()
    genes_path = extraction_dir / "rna_genes.tsv"
    if not genes_path.is_file():
        raise FileNotFoundError(
            f"rna_genes.tsv not found in {extraction_dir} -- was extraction run with --rna-assay?"
        )
    genes_frame = pd.read_csv(genes_path, sep="\t", dtype=str, keep_default_na=False)
    if "feature_name" not in genes_frame.columns:
        raise RuntimeError(
            f"{genes_path} has no feature_name column -- it was written by an "
            "extraction predating this contract. Re-extract (the R script hash is "
            "part of the extraction signature, so a normal rerun will do it)."
        )
    genes = [
        GeneFeature(
            feature_name=row["feature_name"],
            gene_id=row.get("gene_id", "") or "",
            common_id=row.get("common_id", "") or "",
        )
        for _, row in genes_frame.iterrows()
    ]
    gene_order_check = read_barcode_list(extraction_dir / "rna_genes_order.tsv")
    # Compares feature_name, NOT gene_id. Both sides are written by the R helper from the same rownames,
    # so this checks that the table's row order matches the counts matrix; an annotated gene_id is not a
    # positional identifier.
    if [gene.feature_name for gene in genes] != gene_order_check:
        raise RuntimeError(
            "rna_genes.tsv and rna_genes_order.tsv disagree on gene order -- Seurat extraction is inconsistent"
        )
    if len(set(gene.feature_name for gene in genes)) != len(genes):
        raise RuntimeError(
            f"{genes_path} contains duplicate feature_name values -- feature_name is "
            "the matrix row key and must be unique"
        )
    cells = read_barcode_list(extraction_dir / "cells.tsv")
    # Binary cache + per-process memo. See load_counts_matrix.
    matrix = load_counts_matrix(extraction_dir / "rna_counts.mtx")
    if matrix.shape != (len(genes), len(cells)):
        raise RuntimeError(
            f"rna_counts.mtx shape {matrix.shape} does not match "
            f"{len(genes)} genes x {len(cells)} cells"
        )
    metadata = read_cell_metadata(extraction_dir / "cell_metadata.tsv")
    metadata = metadata.loc[cells]
    if list(metadata["barcode"]) != cells:
        raise RuntimeError("cell_metadata.tsv row order does not match cells.tsv")
    return SinglecellRnaInputs(genes=genes, matrix=matrix, cells=cells, metadata=metadata)


def aggregate_rna_pseudobulk(
    inputs: SinglecellRnaInputs,
    *,
    group_by: Sequence[str],
    stratify_by: Optional[str] = None,
    min_cells_per_group: int = 1,
    remove: Optional[Sequence[str]] = None,
    replicate_column: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]], Dict[str, object]]:
    """Group cells into pseudobulk samples and sum their per-gene counts.

    Thin wrapper over the shared ``aggregate_pseudobulk_core`` -- identical
    grouping/summing logic to ATAC's ``aggregate_pseudobulk``, just handed
    gene IDs instead of peak IDs.

    ``remove``: pseudobulk sample_id(s) to exclude outright (outliers seen on
    the PCAs), wired all the way through to pipeline run.
    """
    # feature_name, not gene_id: aggregate_pseudobulk_core builds per-sample
    # dicts keyed by this, and gene_id is "." for every gene the GTF could not
    # resolve.
    feature_names = [gene.feature_name for gene in inputs.genes]
    return aggregate_pseudobulk_core(
        inputs.matrix, feature_names, inputs.metadata, inputs.cells,
        sanitize_sample_id=sanitize_sample_id,
        group_by=group_by,
        stratify_by=stratify_by, min_cells_per_group=min_cells_per_group,
        remove_samples=remove, replicate_column=replicate_column,
    )


def write_rna_pseudobulk_outputs(
    genes: Sequence[GeneFeature],
    raw_counts: Mapping[str, Mapping[str, float]],
    manifest_rows: Sequence[Mapping[str, object]],
    out_dir: Union[str, Path],
    *,
    pseudocount: float = 1.0,
    group_by: Optional[Sequence[str]] = None,
    dropped_small_groups: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """RNA counterpart of ``atac.singlecell.write_pseudobulk_outputs``.

    Same raw_counts/cpm/log2_cpm/sample_manifest four-file contract; feature
    annotation is just gene_id + common_id (the Ensembl master ID and the
    common secondary ID; the full annotation is not needed) -- no GTF, no chrom/start/end/width.
    """
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # feature_name leads, and is the key every value dict is built on. gene_id
    # and common_id ride alongside as annotation. They are all three written to
    # the matrix files so a reader can join on any of them, but only the first
    # is unique.
    annotation_rows = [
        {
            "feature_name": gene.feature_name,
            "gene_id": gene.gene_id or MISSING_VALUE,
            "common_id": gene.common_id or MISSING_VALUE,
        }
        for gene in genes
    ]
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    feature_names = [row["feature_name"] for row in annotation_rows]

    cpm: Dict[str, Dict[str, float]] = {}
    log2_cpm: Dict[str, Dict[str, float]] = {}
    for sample_id in sample_ids:
        counts = raw_counts[sample_id]
        library_size = sum(counts.values())
        cpm[sample_id] = {
            name: (counts[name] / library_size * 1_000_000.0 if library_size else 0.0)
            for name in feature_names
        }
        log2_cpm[sample_id] = {
            name: math.log2(cpm[sample_id][name] + pseudocount)
            for name in feature_names
        }

    raw_path = output_dir / "raw_counts.csv"
    cpm_path = output_dir / "cpm.csv"
    log_path = output_dir / "log2_cpm.csv"
    manifest_path = output_dir / "sample_manifest.csv"

    _write_gene_matrix(raw_path, annotation_rows, sample_ids, raw_counts)
    _write_gene_matrix(cpm_path, annotation_rows, sample_ids, cpm)
    _write_gene_matrix(log_path, annotation_rows, sample_ids, log2_cpm)
    with atomic_text_writer(manifest_path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    result: Dict[str, object] = {
        "raw_counts": str(raw_path),
        "cpm": str(cpm_path),
        "log2_cpm": str(log_path),
        "sample_manifest": str(manifest_path),
        "genes": len(feature_names),
        "samples": len(sample_ids),
    }
    if group_by:
        cell_counts_path = output_dir / "cell_counts_per_sample.csv"
        write_cell_counts_table(manifest_rows, dropped_small_groups or [], group_by, cell_counts_path)
        result["cell_counts_per_sample"] = str(cell_counts_path)
    return result


def _gtf_identity(gtf: Optional[Union[str, Path]]) -> Optional[Dict[str, object]]:
    """Which GTF the gene identities are taken from: where it is and its size."""
    if gtf is None:
        return None
    path = Path(gtf).expanduser().resolve()
    try:
        size: Optional[int] = int(path.stat().st_size)
    except OSError:
        size = None
    return {"path": str(path), "size": size}


def _with_gene_identity_from(inputs: SinglecellRnaInputs, gtf: Union[str, Path]) -> SinglecellRnaInputs:
    """The same inputs, each gene's gene_id and common_id resolved from ``gtf``.

    The resolution the extraction applies to its gene table
    (atac.singlecell._apply_gene_identity), so the same GTF gives the same
    identities. feature_name, the key, is not touched.
    """
    from dataclasses import replace

    from paralome.gene_identity import resolve_gene_identities

    resolved, report = resolve_gene_identities([gene.feature_name for gene in inputs.genes], gtf)
    for line in report.summary_lines():
        print(line, flush=True)
    genes = [
        GeneFeature(feature_name=gene.feature_name, gene_id=identity["gene_id"], common_id=identity["common_id"])
        for gene, identity in zip(inputs.genes, resolved)
    ]
    return replace(inputs, genes=genes)


#: The extraction files the aggregation reads. Named so the plan can
#: say WHICH export of them it was built from (see _extraction_inputs).
_AGGREGATION_INPUT_FILES = (
    "rna_counts.mtx", "rna_counts.plmsp", "rna_genes.tsv", "rna_genes_order.tsv",
    "cells.tsv", "cell_metadata.tsv",
)

#: Matrices the stage writes and the checkpoint vouches for.
_PSEUDOBULK_MATRIX_KEYS = ("raw_counts", "cpm", "log2_cpm", "sample_manifest")


def _extraction_inputs(
    extraction: Mapping[str, object], extraction_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """What the aggregation actually reads from the extraction, as a plan field.

    The extraction's signature deliberately leaves out rna_assay, rna_slot and
    gene_symbol_column (atac.singlecell), so a re-export with another
    --rna-slot / --rna-assay / --gene-symbol-column keeps the SAME signature.
    The extraction checkpoint records the RNA options it exported with and the
    SHA-256 of every file it wrote, at the moment it wrote them, so those are
    what go in the plan: a re-export changes them and nothing else does. No
    file is re-hashed here (the counts matrix runs to a gigabyte).

    An extraction recorded without output_sha256 predates that field; the
    files' size and modification time on disk are used instead -- a copy then
    costs one recompute, never a stale reuse.
    """
    recorded = extraction.get("output_sha256")
    recorded = recorded if isinstance(recorded, Mapping) else {}
    files: Dict[str, object] = {
        name: str(recorded[name]) for name in _AGGREGATION_INPUT_FILES if recorded.get(name)
    }
    if not files and extraction_dir is not None:
        for name in _AGGREGATION_INPUT_FILES:
            try:
                stat = (Path(extraction_dir) / name).stat()
            except OSError:
                continue
            files[name] = f"stat:{stat.st_size}:{stat.st_mtime_ns}"
    return {
        "rna_assay": extraction.get("rna_assay"),
        "rna_slot": extraction.get("rna_slot"),
        "gene_symbol_column": extraction.get("gene_symbol_column"),
        "files": files,
    }


def run_singlecell_rna(
    rds: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    group_by: Sequence[str],
    stratify_by: Optional[str] = None,
    rna_assay: Optional[str] = None,
    rna_slot: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
    gtf: Optional[Union[str, Path]] = None,
    assay: Optional[str] = None,
    slot: str = "counts",
    pseudocount: float = 1.0,
    min_cells_per_group: int = 10,
    rscript: str = "Rscript",
    r_script_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    remove: Optional[Sequence[str]] = None,
    replicate_column: Optional[str] = None,
    require_existing_extraction: bool = False,
    aggregate_root: Optional[Union[str, Path]] = None,
    cell_export_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Full single-cell-RNA-to-pseudobulk pipeline: extract (shared with ATAC), then aggregate.

    ``require_existing_extraction``: reuse an extraction or fail; never run R.

    ``aggregate_root``: where the AGGREGATION half writes (matrices and its own
    checkpoint), when that must not be the shared data folder. The extraction is
    still read from ``out_dir``, which is the point -- one extraction, several
    groupings.

    Both exist for the PCA. It is there to explore the PCA space with
    different groupings, so it re-aggregates by design, and its matrix must not
    land where the differential reads. Sharing ``out_dir`` would also have made
    it overwrite ``rna/checkpoints/singlecell_rna.json`` -- the
    pipeline stage's own completion record -- and the supervisor would then
    believe the pipeline had aggregated with the PCA's --group-by.

    ``remove``: pseudobulk sample_id(s) to exclude outright (outliers seen on
    the PCAs), wired all the way through to pipeline run. Included in this stage's checkpoint
    plan_id, so changing ``--remove`` correctly invalidates a prior
    aggregation and reruns it.

    ``out_dir`` is the SAME shared data folder ``atac
    singlecell-samples`` uses for this project -- whichever of the two
    commands runs first against it triggers the one combined R extraction
    (see ``run_seurat_extraction``'s ``rna_assay`` passthrough); this
    function's own aggregated output (raw_counts/cpm/log2_cpm/sample_manifest
    + its checkpoint) nests under ``out_dir/rna/``, matching ATAC's
    ``out_dir/atac/`` nesting from the same directory-layout cleanup.
    """
    output_dir = Path(out_dir).expanduser().resolve()
    extraction_dir = layout_extraction_dir(output_dir)
    extraction = run_seurat_extraction(
        # This is the RNA entry point, so it requires an RNA half and
        # is indifferent to whether an ATAC one is also present -- a multiome
        # object extracts both in the same readRDS and this function reads the
        # genes. Before this it demanded ATAC from objects that have none, which
        # is why `paralome scRNA` could not process scRNA data.
        rds, extraction_dir, require=("rna",), assay=assay, slot=slot,
        rna_assay=rna_assay, rna_slot=rna_slot, gene_symbol_column=gene_symbol_column,
        # A caller that may only reuse the extraction (the PCA) has
        # its --overwrite apply to the aggregation alone; passed through, it
        # skipped the reuse branch and the extraction was refused as missing.
        rscript=rscript, r_script_path=r_script_path,
        overwrite=overwrite and not require_existing_extraction, gtf=gtf,
        require_existing=require_existing_extraction,
        required_columns=[*group_by, *stratum_columns(stratify_by)],   # Several
        # The run's omicview folder, so the run's export is the only one written.
        cell_export_dir=cell_export_dir,
    )

    # The aggregation half's root. Defaults to the shared data folder, so the
    # pipeline is unchanged; the PCA redirects it so an exploratory grouping
    # cannot overwrite the analysis matrices or their checkpoint.
    aggregate_dir = (
        Path(aggregate_root).expanduser().resolve() if aggregate_root else output_dir
    )
    rna_dir = aggregate_dir / "rna"
    # Resolved through the shared helper, which pipeline/state.py's summarizer also calls, so writer and
    # reader cannot drift apart.
    checkpoint_path = stage_checkpoint_path(aggregate_dir, "singlecell_rna")
    print(
        stage_banner(
            "singlecell_rna", derive_file_prefix(aggregate_dir), aggregate_dir,
        ),
        flush=True,
    )
    plan = {
        "extraction_signature": extraction["signature"],
        "group_by": list(group_by),
        "stratify_by": stratify_by,
        "min_cells_per_group": min_cells_per_group,
        "pseudocount": pseudocount,
        "rna_assay": rna_assay,
        "gene_symbol_column": gene_symbol_column,
        "remove": sorted(str(v) for v in remove) if remove else [],
        # Groups with an empty sample value are dropped, not kept
        # as an "unlabeled" sample. Also changes the legacy ids below.
        "empty_sample": "dropped.v1",
    }
    if remove:
        # --replicate-column decides which column a --remove value
        # is matched against, so it is part of the plan whenever --remove is
        # given. Without --remove it changes nothing and plan_id is unchanged.
        plan["replicate_column"] = replicate_column
    # THE GTF IS PART OF THE PLAN, as it is of the ATAC one: the matrices carry gene_id and common_id, so a
    # project run again with another --genome is a different plan.
    gene_identity_gtf = _gtf_identity(gtf)
    # Which export of the extraction this is (see _extraction_inputs).
    extraction_inputs = _extraction_inputs(extraction, extraction_dir)
    plan_id = _stable_digest({**plan, "gtf": gene_identity_gtf, "extraction_inputs": extraction_inputs})
    # Older checkpoints hold the digest of the plan without extraction_inputs (or a GTF). They are reused
    # only when the extraction they recorded was the same export as this one -- the checkpoint carries
    # that extraction's own record -- and otherwise made again.
    legacy_plan_ids = {_stable_digest({**plan, "gtf": gene_identity_gtf})}
    plan_id_without_gtf = _stable_digest(plan)

    def _legacy_reusable(existing: Mapping[str, object]) -> bool:
        if not extraction_inputs["files"]:
            return False
        recorded_inputs = existing.get("extraction_inputs")
        if recorded_inputs is None:
            recorded = existing.get("extraction")
            if not isinstance(recorded, Mapping):
                return False
            recorded_inputs = _extraction_inputs(recorded)
        if recorded_inputs != extraction_inputs:
            return False
        return existing.get("plan_id") in legacy_plan_ids or (
            "gene_identity_gtf" not in existing and existing.get("plan_id") == plan_id_without_gtf
        )

    with exclusive_checkpoint_lock(checkpoint_path):
        existing: Optional[Dict[str, object]] = None
        if checkpoint_path.is_file():
            import json
            try:
                existing = json.loads(checkpoint_path.read_text())
            except (OSError, ValueError):
                existing = None
        if not overwrite and existing is not None:
            if (
                existing.get("complete")
                and (existing.get("plan_id") == plan_id or _legacy_reusable(existing))
                and all(
                    Path(str(existing.get(key, ""))).is_file()
                    for key in _PSEUDOBULK_MATRIX_KEYS
                )
                and existing.get("method") in (None, SC_RNA_PSEUDOBULK_METHOD)
            ):
                if existing.get("method") is None:
                    # Same plan (empty_sample is in it, and in the
                    # legacy ids), written before the method was recorded: the
                    # outputs are this method's, so the record says so.
                    existing["method"] = SC_RNA_PSEUDOBULK_METHOD
                    _atomic_json(checkpoint_path, existing)
                existing["checkpoint"] = str(checkpoint_path)
                return existing
        if existing is not None and existing.get("complete"):
            # Say the old result is going BEFORE the matrices are rewritten, so a re-aggregation killed part
            # way leaves complete: false, not a complete checkpoint over a mixture of two plans' files.
            invalidate_checkpoint(
                checkpoint_path, existing,
                "re-aggregating: the plan changed or --overwrite was given",
            )

        inputs = load_singlecell_rna_inputs(extraction_dir)
        if gtf is not None:
            # The identities are those of the GTF in the plan. The
            # extraction holds the ones of the GTF it was first made with, and
            # it is every comparison's: it is read here and not written to.
            inputs = _with_gene_identity_from(inputs, gtf)
        raw_counts, manifest_rows, aggregation_report = aggregate_rna_pseudobulk(
            inputs, group_by=group_by,
            stratify_by=stratify_by, min_cells_per_group=min_cells_per_group,
            remove=remove, replicate_column=replicate_column,
        )
        written = write_rna_pseudobulk_outputs(
            inputs.genes, raw_counts, manifest_rows, rna_dir,
            pseudocount=pseudocount, group_by=group_by,
            dropped_small_groups=aggregation_report.get("dropped_small_groups"),
        )
        # same timestamped paper trail the ATAC path records.
        qc_run = _record_aggregation_qc(
            output_dir, "rna", written, group_by=group_by,
            stratify_by=stratify_by,
            min_cells_per_group=min_cells_per_group, remove=remove,
            plan_id=plan_id, n_samples=len(manifest_rows),
        )

        result: Dict[str, object] = {
            "schema": SC_RNA_PSEUDOBULK_SCHEMA,
            "stage": "singlecell_rna",
            "complete": True,
            "plan_id": plan_id,
            # Required by pipeline/state.py.
            "method": SC_RNA_PSEUDOBULK_METHOD,
            "gene_identity_gtf": gene_identity_gtf,
            "extraction_inputs": extraction_inputs,
            "extraction": extraction,
            "group_by": list(group_by),
            "stratify_by": stratify_by,
            "aggregation": aggregation_report,
            **written,
            **qc_run,
            "output_sha256": {
                key: _sha256_file(Path(str(written[key])))
                for key in _PSEUDOBULK_MATRIX_KEYS
            },
        }
        _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    return result
