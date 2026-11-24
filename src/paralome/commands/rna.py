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

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

from paralome.layout import DATA_DIRNAME, data_workdir, extraction_dir as layout_extraction_dir
from paralome.config.user import resolve_rna_expression_gate
from paralome.stats import FACTOR_HELP, STATISTIC_HELP, STRATIFY_HELP
from paralome.rna.differential import differential_expression
from paralome.rna.fpkm import quantify_fpkm
from paralome.rna.preflight import project_bam_index_dir
from paralome.rna.strandedness import detect_strandedness
from paralome.rna.parallel import (
    prepare_fpkm_slurm_plan, submit_fpkm_slurm_plan, run_fpkm_prepare, run_fpkm_sample, run_fpkm_finalize,
    prepare_differential_slurm_plan, submit_differential_slurm_plan, run_differential_prepare, run_differential_block, run_differential_finalize,
)
from paralome.commands.atac import (
    DEFAULT_ALPHA_SPEC,
    _cli_argv,
    _parse_alpha,
    _print_submission,
    _relay_opt,
    _resolve_data_out,
    _submit_single_job,
)
from paralome.pipeline.resource_catalog import resolve_genome_resources
from paralome.singlecell.pseudobulk_core import format_pseudobulk_summary
from paralome.singlecell.replicates import REPLICATE_COLUMN_HELP, RETIRED_GROUP_BY_HELP, pile_columns
from paralome.atac.naming import derive_file_prefix
from paralome.checkpoints import stage_checkpoint_path
from paralome.commands.atac import submit_stratum_array
from paralome.singlecell.sample_removal import split_remove_values
from paralome.singlecell.strata import (
    new_submission_token,
    plan_strata,
    run_gather_from_plan,
    split_stratify_values,
    write_gather_plan,
)
from paralome.rna.singlecell import run_singlecell_rna
from paralome.singlecell.stats_core import parse_request as parse_singlecell_request
from paralome.rna.singlecell_stats import run_singlecell_rna_differential
from paralome.singlecell.strata import format_differential_summary

rna_app = typer.Typer(help="Bulk RNA-seq quantification and expression analysis")








def _resolve_rna_data_paths(
    data_dir: Optional[Path],
    cpm: Optional[Path],
    samples: Optional[Path],
    extraction: Optional[Path],
) -> tuple:
    """Fill in --cpm/--samples/--extraction from --data-dir.

    Mirrors ``atac._resolve_data_paths``: RNA's own aggregated output
    (cpm.csv/sample_manifest.csv) nests under data_dir/rna/, while the
    Seurat extraction stays directly under data_dir -- shared with the
    ATAC side, since both may point at the same --out.
    """
    if data_dir is not None:
        cpm = cpm or (data_dir / "rna" / "cpm.csv")
        samples = samples or (data_dir / "rna" / "sample_manifest.csv")
        extraction = extraction or layout_extraction_dir(data_dir)
    missing = []
    if cpm is None:
        missing.append("--cpm (or --data-dir)")
    if samples is None:
        missing.append("--samples (or --data-dir)")
    if extraction is None:
        missing.append("--extraction (or --data-dir)")
    if missing:
        raise typer.BadParameter("Missing required path(s): " + ", ".join(missing))
    return cpm, samples, extraction


# RNA's single-cell stages submit through `_submit_single_job` (imported from commands/atac.py), which
# routes job_name through `slurm_job_name(workdir, ...)` to prefix the project/comparison, so
# concurrently running comparisons are distinguishable in squeue. There is no RNA-local copy.


#: The detection record from the most recent _resolve_strandedness_option
#: call, so the stage can record it without threading a second return value
#: through every caller. None when a concrete number was passed and no detection
#: ran -- which is itself worth recording, since "0 because you said so" and "0
#: because nothing could tell" are different facts. `rna quantify-fpkm` writes
#: it to inputs/strandedness_detection.json under --out, on both backends --
#: see _record_strandedness_decision.
LAST_STRANDEDNESS_DETECTION: Optional[dict] = None


def _record_strandedness_decision(out: Path, requested: str, resolved: int) -> Path:
    """Write what --strandedness was given and what it came to, beside the plan."""
    from paralome.pipeline.jsonio import write_json

    return write_json(
        Path(out) / "inputs" / "strandedness_detection.json",
        {
            "schema": "paralome.rna.strandedness_decision.v1",
            "requested": requested.strip().lower(),
            "strandedness": int(resolved),
            "detection": LAST_STRANDEDNESS_DETECTION,
        },
        atomic=True, default=str,
    )


def _resolve_strandedness_option(
    strandedness: str, *, samples_path: Path, gtf_path: Path, paired_end: bool,
    index_dir: Optional[Path] = None,
) -> int:
    """Resolve --strandedness's ``"0"``/``"1"``/``"2"``/``"auto"`` string
    into a concrete validated int, running the auto-detection and printing its
    evidence to stderr when ``"auto"`` was requested.

    Deliberately done here, at the CLI layer, rather than inside
    ``quantify_fpkm()`` itself: the stage function's own contract (and its
    checkpoint-staleness comparison, which compares whatever concrete
    strandedness value was actually used) stays simple -- "auto" is resolved
    to a real number before it ever reaches the stage, not a new kind of value
    the stage has to understand.
    """
    global LAST_STRANDEDNESS_DETECTION
    LAST_STRANDEDNESS_DETECTION = None
    normalized = strandedness.strip().lower()
    if normalized == "auto":
        try:
            # Auto reads each BAM's mapped-read total from its
            # index, and runs before preflight -- so a missing index is made
            # first, as preflight would make it (see ensure_bam_indexes).
            detection = detect_strandedness(samples_path, gtf_path, paired_end=paired_end,
                                            index_dir=index_dir, make_missing_indexes=True)
        except (OSError, ValueError) as exc:
            # A BAM without an index, a sheet or GTF that is not
            # there -- said as a refusal of the option, not as a traceback.
            raise typer.BadParameter(f"--strandedness auto could not run: {exc}") from None
        LAST_STRANDEDNESS_DETECTION = dict(detection)
        print(
            f"[paralome] rna_fpkm_quantification: auto-detected strandedness "
            f"from sample {detection['sample_id']!r} -- {detection['reason']}",
            file=sys.stderr,
        )
        if detection["detected_strandedness"] is None:
            # RESOLVES TO UNSTRANDED rather than refusing the run.
            #
            # A failed detection suggests the library is not stranded, and the
            # costs are asymmetric: counting unstranded keeps every read and adds
            # antisense noise, where guessing a direction wrong sends reads to
            # the strand the gene is not on and the signal is simply lost.
            # Falling back to the recoverable failure is the correct policy.
            #
            # BUT IT IS SAID OUT LOUD, EVERY TIME. "Inconclusive" covers a
            # genuinely non-directional library AND a panel that could not see
            # enough reads to tell. Those resolve to the same number and
            # deserve to be distinguishable later, so the reason is printed
            # here and recorded beside the run by the caller.
            print(
                "[paralome] rna_fpkm_quantification: strandedness detection was "
                f"inconclusive, using 0 (unstranded) -- {detection['reason']}",
                file=sys.stderr,
            )
            return 0
        return int(detection["detected_strandedness"])
    try:
        value = int(normalized)
    except ValueError:
        raise typer.BadParameter("--strandedness must be 0, 1, 2, or 'auto'") from None
    if value not in {0, 1, 2}:
        raise typer.BadParameter("--strandedness must be 0, 1, 2, or 'auto'")
    return value


@rna_app.command("detect-strandedness")
def detect_strandedness_command(
    samples: Path = typer.Option(..., "--samples"),
    gtf: Path = typer.Option(..., "--gtf"),
    paired_end: Optional[bool] = typer.Option(
        None,
        "--paired-end/--single-end",
        help="Sequencing mode is required; there is no default.",
    ),
):
    """Preview the auto-strandedness verdict without running a full
    quantification -- prints the panel size, per-mode recovery, and the
    detected value (or an explanation of why it was inconclusive)."""
    if paired_end is None:
        raise typer.BadParameter(
            "One of --paired-end or --single-end is required."
        )
    result = detect_strandedness(samples, gtf, paired_end=paired_end)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


@rna_app.command("quantify-fpkm")
def quantify_fpkm_command(
    samples: Path = typer.Option(..., "--samples"),
    gtf: Path = typer.Option(..., "--gtf"),
    out: Path = typer.Option(..., "--out"),
    threads: int = typer.Option(4, "--threads", min=1),
    jobs: int = typer.Option(0, "--jobs", min=0),
    strandedness: str = typer.Option(
        "0", "--strandedness",
        help=(
            "0 (unstranded), 1 (first-strand), 2 (second-strand), or 'auto' "
            "to detect it from a small GTF-derived gene panel checked "
            "against the largest BAM (see 'paralome RNA "
            "strandedness' to preview the decision without running "
            "quantification)."
        ),
    ),
    paired_end: Optional[bool] = typer.Option(
        None,
        "--paired-end/--single-end",
        help="Sequencing mode is required; there is no default, to avoid silently assuming paired-end.",
    ),
    require_both_ends_mapped: bool = typer.Option(
        True,
        "--require-both-ends-mapped/--allow-single-mapped-end",
    ),
    exclude_chimeric_fragments: bool = typer.Option(
        True,
        "--exclude-chimeric-fragments/--allow-chimeric-fragments",
    ),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    backend: str = typer.Option("local", "--backend", help="local or slurm"),
    submit: bool = typer.Option(False, "--submit"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    mem: str = typer.Option("16G", "--mem"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    backend = backend.strip().lower()
    if backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if paired_end is None:
        raise typer.BadParameter(
            "One of --paired-end or --single-end is required; RNA quantification does not "
            "silently default to paired-end."
        )
    # 0 passed the option's own minimum and failed with "math domain
    # error" at log2(0 + 0), after every BAM had been counted.
    if not pseudocount > 0:
        raise typer.BadParameter(
            f"--pseudocount must be greater than 0 (got {pseudocount:g}); "
            "log2(FPKM + pseudocount) has no value for a gene with no reads otherwise."
        )

    resolved_strandedness = _resolve_strandedness_option(
        strandedness, samples_path=samples, gtf_path=gtf, paired_end=paired_end,
        index_dir=project_bam_index_dir(Path(out).expanduser().resolve()),
    )

    if backend == "slurm":
        plan_info = prepare_fpkm_slurm_plan(
            samples, gtf, out, threads=threads, jobs=jobs,
            strandedness=resolved_strandedness, paired_end=paired_end,
            require_both_ends_mapped=require_both_ends_mapped,
            exclude_chimeric_fragments=exclude_chimeric_fragments,
            pseudocount=pseudocount, partition=partition, mem=mem, overwrite=overwrite,
        )
        result = submit_fpkm_slurm_plan(plan_info, submit=submit)
    else:
        result = quantify_fpkm(
            samples,
            gtf,
            out,
            threads=threads,
            jobs=jobs,
            strandedness=resolved_strandedness,
            paired_end=paired_end,
            require_both_ends_mapped=require_both_ends_mapped,
            exclude_chimeric_fragments=exclude_chimeric_fragments,
            pseudocount=pseudocount,
            overwrite=overwrite,
        )
    _record_strandedness_decision(out, strandedness, resolved_strandedness)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


@rna_app.command("differential")
def differential_command(
    expression: Path = typer.Option(..., "--expression", help="gene_log2_fpkm.tsv"),
    samples: Path = typer.Option(
        ...,
        "--samples",
        help="CSV metadata with sample_id and the --factor columns",
    ),
    out: Path = typer.Option(..., "--out"),
    statistic: str = typer.Option(..., "--statistic", help=STATISTIC_HELP),
    factor: List[str] = typer.Option([], "--factor", help=FACTOR_HELP),
    pair_by: Optional[str] = typer.Option(None, "--pair-by", help="paired-t: the sample-sheet column whose shared value makes two samples a pair (one per level), e.g. the subject"),
    covariate: Optional[str] = typer.Option(None, "--covariate", help="lm: a numeric sample-sheet column to regress against"),
    covariate_feature: Optional[str] = typer.Option(None, "--covariate-feature", help="lm: another feature to regress every feature against, e.g. --covariate-feature SOX2"),
    block_size: int = typer.Option(5000, "--block-size", min=1),
    backend: str = typer.Option("local", "--backend", help="local or slurm"),
    submit: bool = typer.Option(False, "--submit"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    mem: str = typer.Option("16G", "--mem"),
    threads: int = typer.Option(1, "--threads", min=1),
    jobs: int = typer.Option(0, "--jobs", min=0),
    overwrite: bool = typer.Option(False, "--overwrite"),
    min_expression: Optional[float] = typer.Option(
        None, "--min-expression",
        help=(
            "Low-expression gate, written as each machine table's "
            "passes_expression_gate column (not the statistical test itself): "
            "a gene must reach this mean expression in at least one level of the "
            "comparison. Raw FPKM units. Defaults to "
            "the configured rna.min_expression value (5.0 out of the box; see "
            "'paralome setup config'); pass 0.0 explicitly to disable this gate."
        ),
    ),
    min_expression_difference: Optional[float] = typer.Option(
        None, "--min-expression-difference",
        help=(
            "Minimum magnitude of difference between the two compared level means, raw "
            "FPKM units, same scope as --min-expression. Defaults to the "
            "configured rna.min_expression_difference (5.0 out of the box); "
            "pass 0.0 explicitly to disable."
        ),
    ),
    pseudocount: float = typer.Option(
        1.0, "--pseudocount", min=0.0,
        help=(
            "Pseudocount to subtract back out when back-transforming "
            "--expression's log2(FPKM + pseudocount) values to raw FPKM for "
            "the reports' Control_mean/Treatment_mean/Fold_change columns "
            "and per-sample values, and for --min-expression/"
            "--min-expression-difference. Must match the pseudocount used to "
            "build --expression (the `rna quantify-fpkm` default is "
            "1.0); does not affect the statistical test itself."
        ),
    ),
    refine_responders: Optional[int] = typer.Option(
        None, "--refine-responders", min=2,
        help=(
            "Also re-run the run's own two-group statistics (t, paired-t, lm; with "
            "its --pair-by and --covariate) on a trimmed population, per "
            "comparison the QC audit defines from --factor (each level against its control level): the N "
            "animals of the compared level with the highest leave-one-out responder score (the "
            "same score panel J's hover shows) and the N control animals "
            "closest to the control mean on that same axis -- the best "
            "responders and the most typical controls, not just any N of "
            "either. Written to results/qc/refined_top<N>/<comparison>/, "
            "beside and never replacing the full-cohort result. Requires "
            "--factor; a run with no factors has no comparisons to refine, "
            "and is refused."
        ),
    ),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut a next stage selects by (motif peak sets, the phospho pairing test, Evidome's gene selection), P or Q and a value: P0.05 (default) or Q0.10. Paralome writes no call of significance; the viewer applies the user's thresholds",
    ),
):
    backend = backend.strip().lower()
    if backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    alpha_metric, alpha_value = _parse_alpha(alpha)
    # As 'rna quantify-fpkm' -- no matrix was built with 0.
    if not pseudocount > 0:
        raise typer.BadParameter(
            f"--pseudocount must be greater than 0 (got {pseudocount:g}); give the "
            "pseudocount --expression was built with (1.0 unless quantification was told otherwise)."
        )
    resolved_min_expression, resolved_min_expression_difference = resolve_rna_expression_gate(
        min_expression, min_expression_difference
    )

    if backend == "slurm":
        plan_info = prepare_differential_slurm_plan(
            expression, samples, out, statistic=statistic,
            factors=factor,
            pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
            block_size=block_size, threads=threads,
            jobs=jobs, partition=partition, mem=mem, overwrite=overwrite,
            min_expression=resolved_min_expression,
            min_expression_difference=resolved_min_expression_difference,
            pseudocount=pseudocount,
            refine_responders=refine_responders or 0,
            alpha_metric=alpha_metric,
            alpha_value=alpha_value,
        )
        result = submit_differential_slurm_plan(plan_info, submit=submit)
    else:
        result = differential_expression(
            expression,
            samples,
            out,
            statistic=statistic,
            factors=factor,
            pair_by=pair_by,
            covariate=covariate,
            covariate_feature=covariate_feature,
            block_size=block_size,
            overwrite=overwrite,
            min_expression=resolved_min_expression,
            min_expression_difference=resolved_min_expression_difference,
            pseudocount=pseudocount,
            refine_responders=refine_responders,
            alpha_metric=alpha_metric,
            alpha_value=alpha_value,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _require_positive_pseudocount(pseudocount: float) -> None:
    """The single-cell commands' version of quantify-fpkm's check."""
    if not pseudocount > 0:
        raise typer.BadParameter(
            f"--pseudocount must be greater than 0 (got {pseudocount:g}); "
            "log2(CPM + pseudocount) has no value for a gene with no counts otherwise."
        )


def _parse_threshold_list(raw: str, *, option_name: str) -> List[float]:
    """Parse a comma-separated list of non-negative FPKM thresholds, e.g.
    ``"5,4"`` -> ``[5.0, 4.0]``. The pipeline's --rna-min-expression and
    --rna-min-expression-difference take one value and refuse more."""
    values: List[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{option_name} must be a comma-separated list of numbers, e.g. \"5,4,3,2,1\" (got {piece!r})"
            ) from exc
        if value < 0:
            raise typer.BadParameter(f"{option_name} values must be non-negative (got {value!r})")
        values.append(value)
    if not values:
        raise typer.BadParameter(f"{option_name} must supply at least one value")
    return values


@rna_app.command("fpkm-prepare")
def fpkm_prepare_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_fpkm_prepare(plan), indent=2, sort_keys=True, default=str))

@rna_app.command("fpkm-sample")
def fpkm_sample_command(plan: Path = typer.Option(..., "--plan"), task_id: int = typer.Option(..., "--task-id"), overwrite: bool = typer.Option(False, "--overwrite")):
    print(json.dumps(run_fpkm_sample(plan, task_id, overwrite), indent=2, sort_keys=True, default=str))

@rna_app.command("fpkm-finalize")
def fpkm_finalize_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_fpkm_finalize(plan), indent=2, sort_keys=True, default=str))

@rna_app.command("differential-prepare")
def differential_prepare_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_differential_prepare(plan), indent=2, sort_keys=True, default=str))

@rna_app.command("differential-block")
def differential_block_command(plan: Path = typer.Option(..., "--plan"), task_id: int = typer.Option(..., "--task-id"), overwrite: bool = typer.Option(False, "--overwrite")):
    print(json.dumps(run_differential_block(plan, task_id, overwrite), indent=2, sort_keys=True, default=str))

@rna_app.command("differential-finalize")
def differential_finalize_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_differential_finalize(plan), indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------
# Single-cell RNA -- mirrors paralome.atac's singlecell-* commands, up to
# the differential stage. RNA and ATAC can come from the same export: these
# commands share the same --rds/--out (shared data folder) as `atac singlecell-*`;
# whichever runs first against a shared --out does the one combined Seurat
# extraction, the second reuses it via checkpoint.
# --------------------------------------------------------------------------


@rna_app.command("singlecell-samples")
def singlecell_rna_command(
    rds: str = typer.Option(..., "--rds", help="Seurat/Signac .rds object (single-cell RNA; may be the same combined multiome object as the ATAC side); or a folder of objects, written with a trailing /"),
    project: Optional[Path] = typer.Option(
        None, "--project",
        help="The analysis project directory, exactly as passed to `paralome scRNA`. The shared "
             "data folder is the nearest data/ above it holding this object's extraction, else "
             "<project-parent>/data, so this stage and the "
             "pipeline always agree on where the extraction lives. Prefer this over --out.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Explicit shared data directory, used verbatim -- for standalone work outside a "
             "project. Mutually exclusive with --project.",
    ),
    replicate_column: Optional[str] = typer.Option(None, "--replicate-column", help=REPLICATE_COLUMN_HELP),
    group_by: Optional[str] = typer.Option(None, "--group-by", hidden=True, help=RETIRED_GROUP_BY_HELP),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", help=STRATIFY_HELP),
    # No default. This defaulted to "RNA", which reached the R helper as
    # an explicit request and disabled the assay detection entirely -- a Visium
    # object whose only assay is `Spatial` could not be extracted at all.
    rna_assay: Optional[str] = typer.Option(
        None, "--rna-assay",
        help="Seurat assay holding the gene-level counts. Auto-detected: anything that is "
             "not ATAC is gene-level, whatever it is named (RNA, Spatial, Xenium, ADT, "
             "yours). Pass this only to break a tie between several such assays.",
    ),
    gene_symbol_column: Optional[str] = typer.Option(
        None, "--gene-symbol-column",
        help="Column in the RNA assay's meta.features carrying the gene symbol/common name "
             "(secondary ID). Omit for gene_id (Ensembl/primary ID) only.",
    ),
    assay: Optional[str] = typer.Option(None, "--assay", help="ATAC assay name, only relevant if this call is also the one that triggers the shared extraction"),
    slot: str = typer.Option("counts", "--slot", help="Seurat assay layer/slot to read"),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    min_cells_per_group: int = typer.Option(10, "--min-cells-per-group", min=1),
    rscript: str = typer.Option("Rscript", "--rscript", help="Rscript executable"),
    remove: List[str] = typer.Option(
        [], "--remove",
        help="Pseudobulk sample_id to exclude outright (repeatable) -- identify outliers from a "
             "prior run's cell_counts_per_sample.csv or global-PCA plot.",
    ),
    genome: Optional[str] = typer.Option(
        None, "--genome",
        help="Configured genome bundle. REQUIRED unless --gtf is given: it resolves the "
             "GTF that attaches an Ensembl ID to every gene during extraction, once, so "
             "every later stage inherits it.",
    ),
    gtf: Optional[Path] = typer.Option(
        None, "--gtf", help="Explicit GTF, overriding whatever --genome would resolve."
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit"),
    threads: int = typer.Option(1, "--threads", min=1),
    mem: str = typer.Option("32G", "--mem"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="A folder of objects with --backend slurm: objects extracted at once (0: all)"),
    # The trajectories are fitted here, right after the extraction and the pseudobulk.
    trajectory_root: Optional[str] = typer.Option(
        None, "--trajectory-root",
        help="<column>:<value>: fit the object's lineages (Slingshot) after the extraction, starting from that cluster "
             "of that cell-metadata column, and write them into the run's cell export. Omit to fit none.",
    ),
    trajectory_fit_cells: Optional[str] = typer.Option(
        None, "--trajectory-fit-cells", help="How the lineages are fitted. Slingshot: <n>:random, <n>:centroid, or cells=<n>,dispersion=random|centroid,stretch=<s>,extend=y|n|pc1,reduction=<name>,df=<n>,shrink=<0-1>. Monocle3: reduction=<name>,ncenter=<n>,minimal_branch_len=<n>,use_partition=y|n,close_loop=y|n,geodesic_distance_ratio=<x>,euclidean_distance_ratio=<x>,orthogonal_proj_tip=y|n,prune_graph=y|n,l1_gamma=<x>,l1_sigma=<x>"),
    trajectory_method: Optional[str] = typer.Option(
        None, "--trajectory-method", help="monocle3 (default: one graph learned in the UMAP, each root-to-leaf path a lineage) or slingshot"),
    trajectory_dims: Optional[int] = typer.Option(
        None, "--trajectory-dims", min=2, help="Fit on the reduction's first n dimensions"),
    trajectory_weld: List[str] = typer.Option(
        [], "--trajectory-weld", help="Monocle3: join tips of the learned graph (repeatable): L39=L42 tip to tip, L30=L33=L35 to a new node at their middle, L39~L42 L39's tip to the nearest point of L42's branch, 'L39->L42' (quoted) L39's tip to L42's, L42's staying an end"),
    trajectory_max_routes: Optional[int] = typer.Option(
        None, "--trajectory-max-routes", help="Monocle3: the most routes the welds may make (default 200)"),
    trajectory_fit_dir: Optional[Path] = typer.Option(
        None, "--trajectory-fit-dir", help="Where the fit goes (the pipeline: <run>/trajectories/fit)"),
    trajectory_export_dir: Optional[Path] = typer.Option(
        None, "--trajectory-export-dir",
        help="The run's omicview folder: the cell export with the lineages goes there (default: the object's)"),
    cell_export_dir: Optional[Path] = typer.Option(
        None, "--cell-export-dir",
        help="Where the object's cell export is written (the pipeline: the run's omicview folder; default "
             "<parent of --out>/omicview)"),
):
    """Read a Seurat single-cell RNA object and aggregate it into bulk-style pseudobulk matrices.

    Mirrors `atac singlecell-samples`. Writes into --out/rna/ (raw_counts/
    cpm/log2_cpm/sample_manifest/cell_counts_per_sample), alongside --out/atac/
    if that stage also ran against this same --out.
    """
    # As quantify-fpkm. log2(CPM + 0) has no value for a gene with
    # no counts, and that failure came only after the extraction.
    _require_positive_pseudocount(pseudocount)
    group_by_columns = pile_columns(replicate_column, stratify_by, retired_group_by=group_by)
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if normalized_backend == "local" and submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    if not genome and not gtf:
        raise typer.BadParameter(
            "scRNA samples: --genome is required (or --gtf to name one directly). "
            "The extraction resolves an Ensembl ID for every gene from the GTF and "
            "stores it once, so every downstream comparison inherits it. Checked here, "
            "before the object is read, rather than after."
        )
    resolved_gtf = gtf
    if genome:
        resolved_gtf = resolve_genome_resources(
            genome=genome, overrides={"gtf": gtf}
        ).path("gtf")
    # One object, or a folder of objects written with a trailing '/'.
    from paralome.singlecell.rds_input import parse_rds_option, rds_argument

    rds = parse_rds_option(rds)
    out = _resolve_data_out(project, out, command="scRNA samples", rds=rds)
    if trajectory_root:
        from paralome.singlecell.trajectories import check_fit_options, parse_root

        parse_root(trajectory_root)
        check_fit_options(trajectory_method, trajectory_fit_cells, trajectory_dims, trajectory_weld,
                          trajectory_max_routes)
        if Path(str(rds)).is_dir():
            raise typer.BadParameter("--trajectory-root fits one object's lineages; --rds names a folder of objects")
        if trajectory_fit_dir is None:
            raise typer.BadParameter("--trajectory-root needs --trajectory-fit-dir, the folder the fit goes to")
    if normalized_backend == "slurm":
        argv = _cli_argv() + [
            "rna", "singlecell-samples",
            "--rds", rds_argument(rds), "--out", str(out),
            "--replicate-column", group_by_columns[0],
            "--slot", slot,
            "--pseudocount", str(pseudocount), "--min-cells-per-group", str(min_cells_per_group),
            "--rscript", rscript, "--backend", "local",
        ]
        # --rna-assay is an optional flag with no default, so it is relayed only when set; unset, it
        # would write `--rna-assay None` into the sbatch.
        _relay_opt(argv, "--rna-assay", rna_assay)
        # A REQUIRED option must be relayed into the job too, or the in-job invocation dies on the
        # requirement the outer call satisfied.
        #
        # The resolved GTF is relayed rather than --genome, so the in-job run
        # cannot re-resolve to something else if the genome catalog differs on
        # the compute node.
        _relay_opt(argv, "--gtf", resolved_gtf)
        if stratify_by:
            argv += ["--stratify-by", stratify_by]
        if gene_symbol_column:
            argv += ["--gene-symbol-column", gene_symbol_column]
        if assay:
            argv += ["--assay", assay]
        for sample_id in remove:
            argv += ["--remove", sample_id]
        if overwrite:
            argv += ["--overwrite"]
        if trajectory_root:
            argv += ["--trajectory-root", trajectory_root, "--trajectory-fit-dir", str(trajectory_fit_dir)]
            _relay_opt(argv, "--trajectory-fit-cells", trajectory_fit_cells)
            _relay_opt(argv, "--trajectory-method", trajectory_method)
            _relay_opt(argv, "--trajectory-dims", trajectory_dims)
            for weld in trajectory_weld or ():
                argv += ["--trajectory-weld", weld]
            _relay_opt(argv, "--trajectory-max-routes", trajectory_max_routes)
            _relay_opt(argv, "--trajectory-export-dir", trajectory_export_dir)
        _relay_opt(argv, "--cell-export-dir", cell_export_dir)
        # A folder of objects is extracted one SLURM task per
        # object, --jobs at a time; this stage's job follows the array.
        after = None
        if Path(str(rds)).is_dir() and not overwrite:
            from paralome.singlecell.rds_input import submit_member_extraction

            after = submit_member_extraction(
                rds, layout_extraction_dir(Path(out)), workdir=Path(out), jobs=jobs, mem=mem,
                partition=partition, submit=submit, gtf=resolved_gtf, assay=assay, slot=slot,
                rna_assay=rna_assay, gene_symbol_column=gene_symbol_column, rscript=rscript,
                cell_export_dir=cell_export_dir,
            )
        result = _submit_single_job(
            job_name="sc_rna_samples", workdir=out, argv=argv,
            threads=threads, mem=mem, partition=partition, submit=submit, after=after,
        )
        _print_submission(result)
        return
    result = run_singlecell_rna(
        rds, out, group_by=group_by_columns, replicate_column=group_by_columns[0],
        stratify_by=stratify_by, rna_assay=rna_assay, gene_symbol_column=gene_symbol_column,
        assay=assay, slot=slot, pseudocount=pseudocount, min_cells_per_group=min_cells_per_group,
        rscript=rscript, overwrite=overwrite, remove=remove or None,
        gtf=resolved_gtf, cell_export_dir=cell_export_dir,
    )
    for line in format_pseudobulk_summary(result, "genes"):
        print(line)
    if trajectory_root:
        from paralome.singlecell.trajectories import run_trajectory_fit

        run_trajectory_fit(
            layout_extraction_dir(Path(out)), trajectory_fit_dir, root=trajectory_root, modality="scRNA",
            fit_cells=trajectory_fit_cells, dims=trajectory_dims, rscript=rscript,
            export_dir=trajectory_export_dir or cell_export_dir, overwrite=overwrite, method=trajectory_method,
            welds=trajectory_weld, max_routes=trajectory_max_routes,
            echo=lambda text: print(f"[paralome] {text}", flush=True),
        )


@rna_app.command("singlecell-differential")
def singlecell_rna_differential_command(
    project: Optional[Path] = typer.Option(
        None, "--project",
        help="Analysis project directory. The shared data folder is derived as <project-parent>/data, and --cpm/--samples/--extraction are read from it. Use this rather than spelling the three paths out.",
    ),
    cpm: Optional[Path] = typer.Option(None, "--cpm", help="Gene CPM matrix; derived from --project/--data-dir if omitted"),
    samples: Optional[Path] = typer.Option(None, "--samples", help="Sample manifest; derived from --data-dir if omitted"),
    extraction: Optional[Path] = typer.Option(None, "--extraction", help="Seurat extraction directory; derived from --data-dir if omitted"),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir",
        help="The shared data directory -- supplies --cpm/--samples/--extraction automatically "
             "(rna/cpm.csv, rna/sample_manifest.csv, extraction/ underneath it)",
    ),
    out: Path = typer.Option(..., "--out", help="Single-cell differential-expression output directory"),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by"),
    stratify_value: Optional[str] = typer.Option(
        None, "--stratify-value",
        help=(
            "Comma-separated --stratify-by value(s) to run, e.g. 'CD8 TEM,CD4 Naive'. "
            "Either spelling works: the raw metadata value, or the underscored slug "
            "you see in filenames (CD8_TEM). Omit to run every stratum. Mutually "
            "exclusive with --stratify-skip."
        ),
    ),
    stratify_skip: Optional[str] = typer.Option(
        None, "--stratify-skip",
        help=(
            "Comma-separated --stratify-by value(s) to EXCLUDE, running all the "
            "others. Same spelling rules as --stratify-value. Mutually exclusive "
            "with it."
        ),
    ),
    min_cells_per_stratum: int = typer.Option(
        0, "--min-cells-per-stratum", min=0,
        help=(
            "Skip a --stratify-by group whose TOTAL cells, summed across its "
            "pseudobulk samples, fall below this. 0 disables it. Distinct from "
            "--min-cells-per-group, which drops one sample (donor x group) during "
            "AGGREGATION and therefore changes these totals -- raise that and these "
            "numbers move. Skipped groups are reported, not silently absent."
        ),
    ),
    statistic: str = typer.Option(..., "--statistic", help=STATISTIC_HELP + ". Pseudobulk is the default and has no term; :cell after a statistic runs that statistic on the cells (folders t_cell, paired_t_cell, lm_cell, anova_cell), e.g. anova,anova:cell runs both levels."),
    pair_by: Optional[str] = typer.Option(None, "--pair-by", help="paired-t: the sample-manifest column whose shared value makes two samples a pair (one per level), e.g. the subject or litter"),
    covariate: Optional[str] = typer.Option(None, "--covariate", help="lm: a numeric sample-manifest column to regress expression against"),
    covariate_feature: Optional[str] = typer.Option(None, "--covariate-feature", help="lm: another feature to regress every feature against, e.g. --covariate-feature SOX2"),
    factor: List[str] = typer.Option([], "--factor", help=FACTOR_HELP + ". The column is read from the pseudobulk sample manifest, which carries any cell-metadata column that is constant within a group."),
    naive_cell_t_peak_chunk_size: int = typer.Option(
        2000, "--naive-cell-t-peak-chunk-size", min=1,
        help="How many genes' per-cell counts the :cell statistics densify at once "
             "(memory only; changes no number).",
    ),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    min_mean_cpm: float = typer.Option(1.0, "--min-mean-cpm", min=0.0),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut a next stage selects by (motif peak sets, the phospho pairing test, Evidome's gene selection), P or Q and a value: P0.05 (default) or Q0.10. Paralome writes no call of significance; the viewer applies the user's thresholds",
    ),
    # ScRNA takes snATAC's --alpha spec. Two older flags are declared only to refuse a stale command
    # line by name.
    alpha_metric: Optional[str] = typer.Option(None, "--alpha-metric", hidden=True),
    alpha_value: Optional[float] = typer.Option(None, "--alpha-value", hidden=True),
    overwrite: bool = typer.Option(False, "--overwrite"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit"),
    threads: int = typer.Option(1, "--threads", min=1),
    # --threads and --mem are PER STRATUM under --backend slurm; each
    # stratum becomes its own array task and SLURM does the accounting.
    jobs: int = typer.Option(0, "--jobs", min=0, help="Worker processes for the :cell statistics; 0 (the default) uses --threads"),
    max_concurrent_strata: int = typer.Option(
        0, "--max-concurrent-strata", min=0,
        help="Cap simultaneously running stratum tasks (SLURM array %%K). 0 runs them all at once.",
    ),
    mem: str = typer.Option("32G", "--mem"),
    partition: Optional[str] = typer.Option(None, "--partition"),
):
    """Run single-cell RNA differential expression: any combination of t, paired-t, lm and anova.

    Mirrors `atac singlecell-differential` -- the same --statistic, read by the
    same parser, on the pseudobulk values by default or on the cells with
    :cell -- and writes the same folders: one per statistic, each with the
    per-comparison reports, Evidome table, power files and
    machine tables bulk RNA writes. Genes carry the Ensembl (primary) ID and the
    common (secondary) ID.
    """
    # As quantify-fpkm. log2(CPM + 0) has no value for a gene with
    # no counts, and that failure came only after the extraction.
    _require_positive_pseudocount(pseudocount)
    if alpha_metric is not None or alpha_value is not None:
        raise typer.BadParameter(
            "--alpha-metric/--alpha-value are replaced by --alpha, the spec snATAC takes: "
            "e.g. --alpha Q0.05 or --alpha P0.10"
        )
    alpha_metric, alpha_value = _parse_alpha(alpha)
    # The shared parser, given everything it validates against, so anova, lm
    # and paired-t without their companion option are refused HERE, before a
    # stage is queued.
    try:
        parse_singlecell_request(
            statistic, factors=list(factor), pair_by=pair_by,
            covariate=covariate, covariate_feature=covariate_feature,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    # --stratify-by alone fans out over every value of that column;
    # --stratify-value narrows the fan-out to one stratum, which is what the
    # SLURM array tasks pass. --project derives the shared data folder the same
    # way every other stage does; --data-dir names it verbatim.
    if project is not None and data_dir is not None:
        raise typer.BadParameter(
            f"scRNA differential: pass --project (the data folder is derived as "
            f"<project-parent>/{DATA_DIRNAME}) or --data-dir (used verbatim), not both."
        )
    if project is not None:
        data_dir = data_workdir(project)
    if stratify_value and stratify_skip:
        raise typer.BadParameter(
            "--stratify-value names the strata to RUN and --stratify-skip names the "
            "strata to EXCLUDE; pass one or the other, not both."
        )
    cpm, samples, extraction = _resolve_rna_data_paths(data_dir, cpm, samples, extraction)
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if normalized_backend == "local" and submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    if normalized_backend == "slurm":
        argv = _cli_argv() + [
            "rna", "singlecell-differential",
            "--cpm", str(cpm), "--samples", str(samples), "--extraction", str(extraction), "--out", str(out),
            "--statistic", statistic,
            *[token for name in factor for token in ("--factor", str(name))],
            "--naive-cell-t-peak-chunk-size", str(naive_cell_t_peak_chunk_size),
            "--pseudocount", str(pseudocount), "--min-mean-cpm", str(min_mean_cpm),
            "--alpha", alpha,
            "--jobs", str(jobs or threads), "--backend", "local",
        ]
        # Relayed with _relay_opt rather than spliced into the argv literal,
        # where a possibly-None parameter would reach the command as "None".
        _relay_opt(argv, "--pair-by", pair_by)
        _relay_opt(argv, "--covariate", covariate)
        _relay_opt(argv, "--covariate-feature", covariate_feature)
        if stratify_by:
            argv += ["--stratify-by", stratify_by]
            # Relayed only when set: str(None) would hand the in-job invocation a stratum literally named
            # "None", and the fan-out keys off `stratify_by and stratify_value is None`.
            _relay_opt(argv, "--stratify-value", stratify_value)
        _relay_opt(argv, "--stratify-skip", stratify_skip)
        if min_cells_per_stratum:
            argv += ["--min-cells-per-stratum", str(min_cells_per_stratum)]
        if overwrite:
            argv += ["--overwrite"]

        # Each stratum is its own SLURM process, so the parent need not
        # keep track of the resources.
        #
        # So --threads and --mem are PER STRATUM here. The planning is done in
        # the FOREGROUND, on the login node, from the sample manifest alone --
        # every skip decision and the "no testable strata" refusal happen
        # before anything queues, instead of inside a job.
        if stratify_by and not split_stratify_values(stratify_value):
            plan = plan_strata(
                sample_manifest=samples, stratify_by=stratify_by,
                stratify_skip=split_remove_values([stratify_skip]) if stratify_skip else None,
                min_cells_per_stratum=min_cells_per_stratum,
                statistic=statistic, factors=list(factor), pair_by=pair_by,
                covariate=covariate, covariate_feature=covariate_feature,
            )
            file_prefix = derive_file_prefix(out)
            # One token per submission, recorded in the plan and
            # exported into every array task; see strata.SUBMISSION_ENV.
            submission = new_submission_token()
            index_path = stage_checkpoint_path(out, "singlecell_rna_differential")
            plan_path = write_gather_plan(
                Path(out) / "scripts" / "sc_rna_diff.plan.json",
                stage="singlecell_rna_differential",
                submission=submission,
                sample_manifest=str(samples), output_dir=str(out),
                checkpoint_path=str(index_path),
                file_prefix=file_prefix, stratify_by=stratify_by,
                stratify_skip=split_remove_values([stratify_skip]) if stratify_skip else None,
                statistic=statistic, factors=list(factor), pair_by=pair_by,
                covariate=covariate, covariate_feature=covariate_feature,
                min_cells_per_stratum=min_cells_per_stratum,
            )
            result = submit_stratum_array(
                job_name="sc_rna_diff", workdir=out, argv=argv,
                values=plan.runnable,
                gather_argv=_cli_argv() + [
                    "rna", "singlecell-differential-gather", "--plan", str(plan_path),
                ],
                threads=threads, mem=mem, partition=partition,
                submission=submission, index_checkpoint=index_path,
                submit=submit, throttle=max_concurrent_strata or None,
            )
            _print_submission(result)
            return

        result = _submit_single_job(
            # Shortened from "sc_rna_differential" so it matches ATAC's
            # "sc_atac_diff" and so project_sc_rna_diff still fits the %.35j
            # squeue column -- "Tolerant_vs_HUU_" plus the long
            # form is 36 characters and truncates exactly where the modality is.
            job_name="sc_rna_diff", workdir=out, argv=argv,
            threads=threads, mem=mem, partition=partition, submit=submit,
        )
        _print_submission(result)
        return
    result = run_singlecell_rna_differential(
        cpm, samples, extraction, out,
        statistic=statistic,
        stratify_by=stratify_by, stratify_value=stratify_value,
        stratify_skip=stratify_skip,
        min_cells_per_stratum=min_cells_per_stratum,
        factors=list(factor),
        pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
        naive_cell_t_peak_chunk_size=naive_cell_t_peak_chunk_size,
        alpha=alpha_value, alpha_metric=alpha_metric,
        pseudocount=pseudocount, min_mean_cpm=min_mean_cpm,
        # --jobs 0 means --threads, on both backends.
        overwrite=overwrite, jobs=jobs or threads,
    )
    for line in format_differential_summary(result):
        print(line)


@rna_app.command("singlecell-differential-gather")
def singlecell_rna_differential_gather_command(
    plan: Path = typer.Option(..., "--plan", help="Plan file written by the array submitter"),
) -> None:
    """Index the per-stratum results an array produced. Machine surface.

    Submitted by ``rna singlecell-differential --backend slurm`` with
    ``--dependency=afterany`` on the stratum array, so it runs even when some
    tasks failed -- naming which is the point. Not intended to be run by hand;
    it takes no analysis arguments, only the plan the submitter recorded.
    """

    payload = run_gather_from_plan(plan)
    strata = payload.get("strata") or {}
    print(f"Indexed {len(strata)} stratum result(s) -> {payload.get('checkpoint')}")
