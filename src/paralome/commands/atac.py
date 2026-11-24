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
import shlex
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import typer

from paralome.stats import FACTOR_HELP, STATISTIC_HELP, STRATIFY_HELP
from paralome.layout import DATA_DIRNAME, data_workdir, extraction_dir as layout_extraction_dir
from paralome.pipeline.slurm import (
    SlurmOptions,
    afterany,
    compress_array_ids,
    sbatch,
    write_sbatch,
)
from paralome.atac.naming import derive_file_prefix
from paralome.checkpoints import stage_checkpoint_path
from paralome.pipeline.naming import slurm_job_name
from paralome.singlecell.sample_removal import split_remove_values
from paralome.singlecell.strata import (
    SUBMISSION_ENV,
    new_submission_token,
    plan_strata,
    run_gather_from_plan,
    split_stratify_values,
    withdraw_index,
    write_gather_plan,
)

from paralome.atac.peak_merge import (
    prepare_peak_merge_slurm_plan, run_peak_merge, run_peak_merge_finalize_task,
    run_peak_merge_sample_task, submit_peak_merge_slurm_plan,
)
from paralome.atac.quantification import (
    prepare_peak_quantification_slurm_plan, run_peak_quantification,
    run_peak_quantification_finalize_task, run_peak_quantification_prepare_task,
    run_peak_quantification_sample_task, submit_peak_quantification_slurm_plan,
)
from paralome.atac.statistics import (
    prepare_differential_accessibility_slurm_plan,
    run_differential_accessibility,
    run_differential_accessibility_block_task,
    run_differential_accessibility_finalize_task,
    run_differential_accessibility_prepare_task,
    submit_differential_accessibility_slurm_plan,
)
from paralome.atac.motifs import (
    DEFAULT_MAX_PEAK_WIDTH,
    run_motif_analysis, run_motif_parallel_analysis, prepare_motif_slurm_plan, submit_motif_slurm_plan,
    run_motif_prepare_task, run_motif_scan_task, run_motif_finalize_task,
)
from paralome.atac.singlecell import (
    run_singlecell_inspect, run_singlecell_atac,
)
from paralome.atac.collect import finalize_collect, run_collect, run_collect_task, submit_collect
from paralome.singlecell.pseudobulk_core import format_pseudobulk_summary
from paralome.singlecell.replicates import REPLICATE_COLUMN_HELP, RETIRED_GROUP_BY_HELP, pile_columns
# `paralome.atac.singlecell_stats` requires statsmodels' MixedLM/OLS, which is
# already a hard dependency of `paralome.atac.statistics` -- imported directly
# (not lazily) since CLI startup already pays that cost for `atac differential`.
from paralome.atac.singlecell_stats import (
    DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE, run_singlecell_differential,
)
from paralome.singlecell.stats_core import parse_request as parse_singlecell_request

# `paralome.binaries.worker_argv` is the one resolver of the worker command.
from paralome.binaries import worker_argv as _cli_argv



def _print_submission(result: dict) -> None:
    """Report a submitted SLURM job concisely.

    In interactive mode the full payload was too much text, most of it not
    actionable. A submission payload is four useful facts; it
    was being printed as pretty JSON. Lives here rather than in rna.py because
    rna.py already imports from this module -- the other direction would be a
    cycle.
    """
    jobs = ", ".join(str(j) for j in (result.get("submitted_job_ids") or [])) or "none"
    if result.get("submitted"):
        print(f"Submitted SLURM job {jobs}")
    else:
        print("Not submitted (script written only)")
    if result.get("script"):
        print(f"  Script     : {result['script']}")



def _resolve_data_out(project, out, *, command: str, rds=None):
    """Resolve the shared data directory from --project or an explicit --out.

    The pipeline derives <project-parent>/data from --project, and this stage
    command does the same through layout.data_workdir(): one implementation of
    where the data folder lives, so the stage command and the pipeline cannot
    point at different directories (and extract the rds twice). Lives here
    rather than in rna.py because rna.py already imports from this module; the
    other direction would be a cycle.

    --out is kept, verbatim, for standalone work. It is NOT combined with
    --project: accepting both would mean silently ignoring one, and two paths
    that disagree is the exact failure this exists to prevent.
    """
    if project is not None and out is not None:
        raise typer.BadParameter(
            f"{command}: pass --project (the data folder is derived as "
            f"<project-parent>/{DATA_DIRNAME}) or --out (used verbatim), not both."
        )
    if project is not None:
        # The parent project's data folder, when one above
        # --project already holds this object's extraction.
        return data_workdir(project, rds=rds)
    if out is not None:
        return out
    raise typer.BadParameter(
        f"{command}: --project or --out is required. --project is the one to use -- it "
        f"derives the same data folder the pipeline uses, so the extraction is shared "
        f"instead of rebuilt."
    )


def _relay_opt(argv: list[str], flag: str, value: object) -> None:
    """Append ``flag value`` to a self-relayed argv, skipping ``None``.

    The SLURM-backed sub-commands re-invoke themselves inside the
    job they generate, rebuilding argv by hand from their own parsed
    parameters. ``_submit_single_job`` renders that argv with
    ``shlex.quote(str(value))`` -- so a parameter whose value is ``None``
    would not vanish, it would become the four-character string ``None``, and
    the in-job invocation would parse it as a real value.

    This is the same contract ``pipeline.orchestrator._add_opt`` enforces on
    the orchestrator-to-CLI hop. Every optional parameter relayed into a
    generated sbatch must go through here rather than being spliced into the
    list literal.
    """
    if value is None:
        return
    argv.extend([flag, str(value)])


def _submit_single_job(
    *,
    job_name: str,
    workdir: Path,
    argv: list[str],
    threads: int,
    mem: str,
    partition: Optional[str],
    submit: bool,
    after: Optional[dict] = None,
) -> dict:
    """Wrap a single (non-array) local CLI invocation as one SLURM job.

    ``after``: a submission this job waits for (``afterany``) --
    a folder's per-object extraction array -- whose job ID is reported with
    this one's, so the caller tracks both.

    For stages whose real parallelism is internal thread/process pooling
    rather than a per-sample scatter dimension (single-cell ATAC pseudobulk
    aggregation over one Seurat object; singlecell-differential over one
    filtered manifest) -- there is no independent-
    sample axis to array over, so the whole step is queued as one job
    instead of run on whatever node invoked the CLI.

    Fire-and-forget, same as every other SLURM-backed ATAC stage: this
    function's job is only to submit and report ``submitted_job_ids`` for the
    caller (a human, or the pipeline supervisor's ``_wait_for_jobs``) to
    track. It must not block waiting for the job itself: completion is
    tracked by the supervisor, from checkpoints, with its own squeue/scontrol
    fallbacks and accounting-lag grace period.
    """
    scripts_dir = workdir / "scripts"
    logs_dir = workdir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{job_name}.sbatch"
    # Every SLURM job name carries stage-specific nomenclature, so it shows
    # which pipeline and stage the server is working on: slurm_job_name()
    # prefixes the project/comparison directory name so concurrently-running
    # comparisons are distinguishable in squeue, not just the stage.
    scoped_job_name = slurm_job_name(workdir, job_name)
    options = SlurmOptions(
        job_name=scoped_job_name,
        cpus_per_task=max(1, threads),
        mem=mem,
        partition=partition,
        output=logs_dir / f"{job_name}.%j.out",
        error=logs_dir / f"{job_name}.%j.err",
    )
    body = "\n" + " ".join(shlex.quote(str(value)) for value in argv) + "\n"
    write_sbatch(script_path, options, body)
    result: dict = {
        "backend": "slurm", "script": str(script_path),
        "submitted": False, "job_id": None, "submitted_job_ids": [],
    }
    if not submit:
        return result
    before = (after or {}).get("job_id")
    job_id = sbatch(script_path, dependency=afterany(before) if before else None)
    result["submitted"] = True
    result["job_id"] = job_id
    result["submitted_job_ids"] = ([before] if before else []) + [job_id]
    return result


def submit_stratum_array(
    *,
    job_name: str,
    workdir: Path,
    argv: list[str],
    values: Sequence[str],
    gather_argv: list[str],
    threads: int,
    mem: str,
    partition: Optional[str],
    submit: bool,
    # The token the gather plan records (strata.new_submission_token)
    # and the top-level index this array will replace. Both required: without
    # the token the gather cannot tell this array's results from an earlier
    # run's, and without the index path a failed rerun leaves the old index
    # saying complete.
    submission: str,
    index_checkpoint: Path,
    throttle: Optional[int] = None,
) -> dict:
    """One SLURM task per stratum, then a gather job that writes the index.

    Each stratum is its own SLURM process. The parent process needs
    very little (2 threads and 2G RAM), so --threads is the number of threads
    each stratum process uses; as those are spun into SLURM, the parent does
    not need to keep track of the resources.

    So ``threads`` and ``mem`` here are PER STRATUM, not a budget to divide, and
    SLURM does the accounting it exists to do. That deletes
    plan_stratum_concurrency's arithmetic, the GIL contention, and the
    fork-from-a-thread hazard in one move -- none of them are questions any more.

    The strata are passed as a FILE indexed by SLURM_ARRAY_TASK_ID rather than
    interpolated into the body. Stratum values are Seurat metadata labels: they
    contain spaces ("CD14 Mono"), slashes, and parentheses, and every one of
    those has already cost this pipeline a run when it reached a shell
    unquoted. A file read by line number cannot be re-parsed by anything.

    Follows the array-then-gather pattern variants/freebayes.py already uses:
    submit the array, then submit the gather with afterany on it. `afterany`
    rather than `afterok` on purpose -- if three of twenty strata fail, the
    gather must still run so it can say WHICH three, and the seventeen that
    worked are on disk either way.
    """
    scripts_dir = workdir / "scripts"
    logs_dir = workdir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    values_path = scripts_dir / f"{job_name}.strata"
    values_path.write_text("\n".join(str(v) for v in values) + "\n")

    scoped = slurm_job_name(workdir, job_name)
    array_script = scripts_dir / f"{job_name}.sbatch"
    array_options = SlurmOptions(
        job_name=scoped,
        cpus_per_task=max(1, threads),
        mem=mem,
        partition=partition,
        array=compress_array_ids(range(len(values)), throttle=throttle),
        output=logs_dir / f"{job_name}.%A_%a.out",
        error=logs_dir / f"{job_name}.%A_%a.err",
    )
    # sed -n "Np" is the portable way to take the Nth line; the value keeps its
    # spaces because it goes into a shell variable and then a quoted argument.
    if not str(submission or "").strip():
        raise ValueError("submit_stratum_array needs the submission token the gather plan records")
    # Every task stamps this token into its stratum checkpoint, and
    # the gather accepts only checkpoints carrying it. See strata.SUBMISSION_ENV.
    body = "\n".join([
        "",
        f"export {SUBMISSION_ENV}={shlex.quote(str(submission))}",
        f'PARALOME_STRATUM="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {shlex.quote(str(values_path))})"',
        'if [ -z "$PARALOME_STRATUM" ]; then',
        '  echo "no stratum for array index $SLURM_ARRAY_TASK_ID" >&2',
        "  exit 1",
        "fi",
        " ".join(shlex.quote(str(value)) for value in argv)
        + ' --stratify-value "$PARALOME_STRATUM"',
        "",
    ])
    write_sbatch(array_script, array_options, body)

    gather_script = scripts_dir / f"{job_name}_gather.sbatch"
    gather_options = SlurmOptions(
        job_name=f"{scoped}_gather",
        # The gather reads N small JSON files and writes one, so it needs
        # very little.
        cpus_per_task=2,
        mem="2G",
        partition=partition,
        output=logs_dir / f"{job_name}_gather.%j.out",
        error=logs_dir / f"{job_name}_gather.%j.err",
    )
    write_sbatch(
        gather_script, gather_options,
        "\n" + " ".join(shlex.quote(str(value)) for value in gather_argv) + "\n",
    )

    result: dict = {
        "backend": "slurm", "script": str(array_script),
        "gather_script": str(gather_script), "strata": list(values),
        "submitted": False, "job_id": None, "submitted_job_ids": [],
    }
    if not submit:
        return result

    # The previous index is withdrawn before the array can start rewriting the files it names, so a
    # rerun whose tasks fail, or whose gather refuses with PLM-0305, cannot leave the old index saying
    # complete over a mixture of two runs. Only when actually submitting: writing the scripts alone
    # changes nothing on disk.
    withdraw_index(
        Path(index_checkpoint), stage=Path(index_checkpoint).stem,
        reason="being recomputed by a new SLURM stratum array",
    )
    array_job_id = sbatch(array_script)
    try:
        gather_job_id = sbatch(gather_script, dependency=f"afterany:{array_job_id}")
    except BaseException:
        # The array is queued and nothing had named it. A command
        # that failed here told the supervisor of no job, and the next start
        # submitted the strata again beside the running ones. Only on this
        # path: when both go in, the supervisor is handed the gather alone.
        print(f"Submitted stratum array job: {array_job_id}", file=sys.stderr, flush=True)
        raise
    result["submitted"] = True
    result["array_job_id"] = array_job_id
    # The stage is complete when the INDEX exists, not when the array drains,
    # so the supervisor is handed the gather. Waiting on the array would let
    # the next stage start against a directory with no index in it.
    result["job_id"] = gather_job_id
    result["submitted_job_ids"] = [gather_job_id]
    return result


atac_app = typer.Typer(help="ATAC-seq peak processing utilities")

DEFAULT_ALPHA_SPEC = "P0.05"


def _parse_alpha(value: str) -> tuple[str, float]:
    """Parse a combined significance-metric + cutoff spec, e.g. "Q0.05" or "P0.10".

    The user declares both the cutoff and which version of P to use in one
    value (--alpha Q0.05, or P0.10), split on the '0.'. Splits on the first "0." so "Q0.05" -> ("Q", "05") -> (metric="q",
    alpha=0.05); "P0.10" -> ("P", "10") -> (metric="p", alpha=0.10).
    """
    raw = str(value).strip()
    if "0." not in raw:
        raise typer.BadParameter(
            f"--alpha must look like Q0.05 or P0.10 (metric letter + '0.' + digits), got {value!r}"
        )
    metric_text, _, fraction = raw.partition("0.")
    metric = metric_text.strip().lower()
    if metric not in {"p", "q"}:
        raise typer.BadParameter(f"--alpha metric must be P or Q, got {metric_text!r} in {value!r}")
    try:
        alpha_value = float("0." + fraction)
    except ValueError as exc:
        raise typer.BadParameter(f"--alpha cutoff is not a valid number in {value!r}") from exc
    if not 0 <= alpha_value <= 1:
        raise typer.BadParameter(f"--alpha cutoff must be between 0 and 1, got {alpha_value} in {value!r}")
    return metric, alpha_value


@atac_app.command("merge-peaks")
def merge_peaks_command(
    peak_manifest: Path = typer.Option(..., "--peak-manifest", help="MACS peak manifest CSV"),
    out: Path = typer.Option(..., "--out", help="Peak merge output directory"),
    within_sample_gap: int = typer.Option(0, "--within-sample-gap", min=0),
    across_sample_gap: int = typer.Option(0, "--across-sample-gap", min=0),
    max_peak_width: Optional[int] = typer.Option(5000, "--max-peak-width", min=1),
    oversized_peak_policy: str = typer.Option(
        "center", "--oversized-peak-policy", help="center or error"
    ),
    oversized_peak_width: int = typer.Option(
        500, "--oversized-peak-width", min=1,
        help="Summit/center window used when a source MACS interval exceeds --max-peak-width",
    ),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(1, "--threads", min=1, help="Total peak-merge CPU budget"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="Maximum simultaneous sample tasks"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("4G", "--mem", help="SLURM memory per sample task/finalizer"),
    submit: bool = typer.Option(False, "--submit", help="Submit the SLURM arrays and dependent finalizer"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Merge samples independently, then construct a deterministic cohort peak atlas."""
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if max_peak_width is None:
        raise typer.BadParameter("--max-peak-width is required")
    if normalized_backend == "slurm":
        plan_info = prepare_peak_merge_slurm_plan(
            peak_manifest, out,
            within_sample_gap=within_sample_gap,
            across_sample_gap=across_sample_gap,
            max_peak_width=max_peak_width,
            oversized_peak_policy=oversized_peak_policy,
            oversized_peak_width=oversized_peak_width,
            threads=threads,
            jobs=jobs,
            partition=partition,
            mem=mem,
            overwrite=overwrite,
        )
        result = submit_peak_merge_slurm_plan(plan_info, submit=submit)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if result.get("submitted"):
            print(f"Submitted peak-merge job: {result['job_id']}")
        return
    if submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    result = run_peak_merge(
        peak_manifest, out, within_sample_gap, across_sample_gap, max_peak_width,
        oversized_peak_policy, oversized_peak_width,
        threads=threads, jobs=jobs, overwrite=overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@atac_app.command("quantify-peaks")
def quantify_peaks_command(
    atlas: Path = typer.Option(..., "--atlas", help="Cohort peak BED4"),
    samples: Path = typer.Option(..., "--samples", help="Deduplicated BAM manifest CSV"),
    gtf: Path = typer.Option(..., "--gtf", help="Genome annotation GTF"),
    out: Path = typer.Option(..., "--out", help="Peak quantification work directory"),
    tss_upstream: int = typer.Option(2000, "--tss-upstream", min=0),
    tss_downstream: int = typer.Option(500, "--tss-downstream", min=0),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    samtools: str = typer.Option("samtools", "--samtools", help="samtools executable"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(1, "--threads", min=1, help="Total peak-quantification CPU budget"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="Maximum simultaneous BAM-counting tasks"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("8G", "--mem", help="SLURM memory per preparation, sample, and finalizer task"),
    submit: bool = typer.Option(False, "--submit", help="Submit preparation, sample arrays, and finalizer"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Prepare annotation, count BAMs independently, and finalize deterministic matrices."""
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if normalized_backend == "slurm":
        plan_info = prepare_peak_quantification_slurm_plan(
            atlas, samples, gtf, out,
            upstream=tss_upstream, downstream=tss_downstream,
            pseudocount=pseudocount, samtools=samtools,
            threads=threads, jobs=jobs, partition=partition, mem=mem,
            overwrite=overwrite,
        )
        result = submit_peak_quantification_slurm_plan(plan_info, submit=submit)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if result.get("submitted"):
            print(f"Submitted peak-quantification job: {result['job_id']}")
        return
    if submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    result = run_peak_quantification(
        atlas, samples, gtf, out, tss_upstream, tss_downstream, pseudocount, samtools,
        threads=threads, jobs=jobs, overwrite=overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@atac_app.command("differential")
def differential_command(
    cpm: Path = typer.Option(..., "--cpm", help="Peak CPM matrix TSV"),
    samples: Path = typer.Option(..., "--samples", help="Sample manifest with sample_id and the --factor columns"),
    out: Path = typer.Option(..., "--out", help="Differential-accessibility output directory"),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    min_mean_cpm: float = typer.Option(1.0, "--min-mean-cpm", min=0.0, help="A peak is tested in a comparison when its mean CPM over that comparison's samples reaches this"),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut a next stage selects by (e.g. motif peak sets), P or Q and a value: P0.05 (default) or Q0.10. Paralome writes no call of significance; the viewer applies the user's thresholds",
    ),
    statistic: str = typer.Option(..., "--statistic", help=STATISTIC_HELP),
    factor: List[str] = typer.Option([], "--factor", help=FACTOR_HELP),
    pair_by: Optional[str] = typer.Option(None, "--pair-by", help="paired-t: the sample-sheet column whose shared value makes two samples a pair (one per level), e.g. the subject"),
    covariate: Optional[str] = typer.Option(None, "--covariate", help="lm: a numeric sample-sheet column to regress against"),
    covariate_feature: Optional[str] = typer.Option(None, "--covariate-feature", help="lm: a peak to regress every peak against, named by peak_id or by its nearest/TSS gene id or name, e.g. --covariate-feature SOX2"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    threads: int = typer.Option(1, "--threads", min=1, help="Total differential-accessibility CPU budget"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="Maximum simultaneous peak-block tasks"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("8G", "--mem", help="SLURM memory per preparation, block, and finalizer task"),
    submit: bool = typer.Option(False, "--submit", help="Submit preparation, peak-block arrays, and global finalizer"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Run peak blocks independently, then apply global FDR correction and reporting."""
    alpha_metric, alpha_value = _parse_alpha(alpha)
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    options = dict(
        pseudocount=pseudocount, min_mean_cpm=min_mean_cpm, alpha=alpha_value,
        threads=threads, jobs=jobs,
        overwrite=overwrite, alpha_metric=alpha_metric,
        statistic=statistic, factors=factor,
        pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
    )
    if normalized_backend == "slurm":
        plan_info = prepare_differential_accessibility_slurm_plan(
            cpm, samples, out, partition=partition, mem=mem, **options,
        )
        result = submit_differential_accessibility_slurm_plan(plan_info, submit=submit)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if result.get("submitted"):
            print(f"Submitted differential-accessibility job: {result['job_id']}")
        return
    if submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    result = run_differential_accessibility(cpm, samples, out, **options)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


@atac_app.command("motifs")
def motifs_command(
    cpm: Path = typer.Option(..., "--cpm", help="Peak CPM matrix TSV"),
    samples: Path = typer.Option(..., "--samples", help="Sample manifest (sample_id and the design columns)"),
    out: Path = typer.Option(..., "--out", help="Motif-analysis output directory"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Registered genome bundle"),
    reference: Optional[Path] = typer.Option(None, "--reference", help="Reference FASTA override"),
    motifs: Optional[Path] = typer.Option(None, "--motifs", help="MEME/JASPAR motif database override"),
    motif_species: Optional[str] = typer.Option(None, "--motif-species", help="Species override; normally inferred from --genome"),
    motif_library: str = typer.Option("jaspar", "--motif-library"),
    motif_p_threshold: float = typer.Option(1e-4, "--motif-p-threshold", min=1e-12, max=0.5),
    jaspar_release: str = typer.Option("2026", "--jaspar-release"),
    download_motifs: bool = typer.Option(True, "--download-motifs/--no-download-motifs"),
    differential_checkpoint: Optional[Path] = typer.Option(
        None, "--differential-checkpoint",
        help=(
            "The differential stage's checkpoint JSON (bulk `atac differential` or the "
            "single-cell differential, stratified or not). Motif enrichment runs once per "
            "comparison of every statistic folder it records (and per stratum), writing "
            "results/<folder>/<prefix>[_<stratum>]_<comparison>_*. When neither this nor "
            "--differential-statistics is given, <project>/differential_accessibility/"
            "checkpoints/differential_accessibility.json beside the quantification is used."
        ),
    ),
    differential_statistics: Optional[Path] = typer.Option(
        None, "--differential-statistics",
        help=(
            "One comparison's per-peak machine statistics table (peak_id, tested, "
            "significant, log2_fold_change columns), classified as the single comparison "
            "the table's control_condition/treatment_condition columns name; results go "
            "directly under results/."
        ),
    ),
    pseudocount: float = typer.Option(
        1.0, "--pseudocount", min=0.0,
        help="Deprecated, unused (CPM-based motif activity was removed). Kept for CLI backward compatibility only.",
    ),
    min_motif_peaks: int = typer.Option(5, "--min-motif-peaks", min=1),
    min_pair_peaks: int = typer.Option(5, "--min-pair-peaks", min=1),
    max_motifs_per_peak: int = typer.Option(25, "--max-motifs-per-peak", min=2),
    scanner_backend: str = typer.Option("auto", "--scanner-backend", help="auto, moods, or python"),
    max_peak_width: int = typer.Option(
        DEFAULT_MAX_PEAK_WIDTH, "--max-peak-width", min=1,
        help="Peaks wider than this are skipped by the motif scan and reported, "
             "not treated as an error. Motif scanning cost is proportional to "
             "total base pairs, so this bounds it; raise it to scan wide loci.",
    ),
    motif_batch_size: int = typer.Option(64, "--motif-batch-size", min=1),
    sequence_batch_bases: int = typer.Option(2000000, "--sequence-batch-bases", min=10000),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut that selects the up and down peak sets motif enrichment tests, P or Q and a value: P0.05 (default) or Q0.10. The enrichment tables are written whole",
    ),
    motifs_per_peak: int = typer.Option(5, "--motifs-per-peak", min=1),
    skip_motif_pairs: bool = typer.Option(
        False, "--skip-motif-pairs/--include-motif-pairs",
        help=(
            "Skip motif-pair co-occurrence analysis (Fisher's-exact enrichment "
            "between every candidate motif pair). Runs by default (matching prior "
            "behavior); pass --skip-motif-pairs to turn it off. It's by far the "
            "slowest part of motif finalize -- a single-threaded combinatorial "
            "loop that can run well over an hour on a full motif library. When "
            "skipped, the pair activity/statistics/workbook outputs are still "
            "written but empty, same as when zero pairs pass --min-pair-peaks."
        ),
    ),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit generated SLURM array and dependent finalizer"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write plans and scripts without execution"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option(
        "2G", "--mem",
        help=(
            "SLURM memory per motif SCAN shard (flat request, same for every array task "
            "regardless of --jobs width -- measured scan-task usage is ~0.5G, so 2G "
            "leaves 4x headroom; the old 16G default was ~32x oversubscribed, which starves "
            "other jobs of schedulable memory and shows up as finalize jobs stuck PD(Resources) "
            "even though nothing is actually OOM-killed). This value is ALSO the floor "
            "_motif_finalizer_memory scales up from, but finalize's own per-sample-count "
            "scaling and 32G/128G caps dominate in practice, so lowering this default does not "
            "under-provision finalize."
        ),
    ),
    threads: int = typer.Option(1, "--threads", min=1, help="Total motif-stage CPU budget"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="Motif shard count and maximum concurrent array tasks"),
    stage: str = typer.Option(
        "all", "--stage",
        help=(
            "Which phase to run: 'all' (default -- scan/export, single-motif "
            "outputs, and motif-pair co-occurrence all in one call), 'singles' "
            "(scan/export + all single-motif outputs only; writes the hand-off "
            "inputs/motif_singles_state.pkl and stops -- no pair computation "
            "happens at all), or 'pairs' (resumes from that hand-off when it was "
            "computed from the same peaks, samples, genome, motifs and thresholds, "
            "skipping the scan and export and going straight to motif-pair "
            "co-occurrence, which is the slow part; a hand-off from other inputs, "
            "or none, is reported and the scan runs again). Use singles then pairs "
            "as two separate submissions to checkpoint and troubleshoot the pairs "
            "step in isolation."
        ),
    ),
    precomputed_hit_database: Optional[Path] = typer.Option(None, "--precomputed-hit-database", hidden=True),
    precomputed_plan: Optional[Path] = typer.Option(None, "--precomputed-plan", hidden=True),
):
    """Scan known motifs and compare motif-associated accessibility and motif pairs."""
    alpha_metric, alpha_value = _parse_alpha(alpha)
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    normalized_stage = stage.strip().lower()
    if normalized_stage not in {"all", "singles", "pairs"}:
        raise typer.BadParameter("--stage must be all, singles, or pairs")
    if normalized_backend == "slurm" and precomputed_hit_database is None:
        # Refuse an unclassifiable differential table here, before
        # the prepare job and scan array are submitted, rather than in the
        # finalizer after they have run. See _require_classifiable_differential.
        from paralome.atac.motifs import _require_classifiable_differential

        _require_classifiable_differential(
            differential_statistics, differential_checkpoint, cpm,
        )
        argv = list(sys.argv[1:])
        # The finalizer runs worker_argv() + this, and paralome-worker has no
        # snATAC group: the command words before the options become `atac motifs`.
        leading = next((i for i, token in enumerate(argv) if token.startswith("-")), len(argv))
        if "motifs" in argv[:leading]:
            argv[:argv[:leading].index("motifs") + 1] = ("atac", "motifs")
        cleaned: list[str] = []
        skip_next = False
        for index, token in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            if token == "--submit":
                continue
            if token == "--backend" and index + 1 < len(argv):
                skip_next = True
                cleaned.extend(["--backend", "local"])
                continue
            if token.startswith("--backend="):
                cleaned.append("--backend=local")
                continue
            cleaned.append(token)
        plan_info = prepare_motif_slurm_plan(
            max_peak_width=max_peak_width,
            cpm_matrix=cpm, sample_manifest=samples, output_dir=out, genome=genome,
            reference_fasta=reference, motif_database=motifs, motif_species=motif_species,
            motif_library=motif_library, motif_p_threshold=motif_p_threshold,
            jaspar_release=jaspar_release, download_motifs=download_motifs,
            scanner_backend=scanner_backend, motif_batch_size=motif_batch_size,
            sequence_batch_bases=sequence_batch_bases, threads=threads, jobs=jobs,
            partition=partition, mem=mem, final_argv=cleaned,
            skip_motif_pairs=skip_motif_pairs,
            # Threaded through so --stage pairs can skip straight to finalize when a completed
            # singles hand-off makes the scan redundant.
            stage=normalized_stage,
        )
        result = submit_motif_slurm_plan(plan_info, submit=submit and not dry_run)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return
    if submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    runner = run_motif_analysis if precomputed_hit_database is not None else run_motif_parallel_analysis
    kwargs = dict(
        genome=genome, reference_fasta=reference, motif_database=motifs,
        motif_species=motif_species, motif_library=motif_library, motif_p_threshold=motif_p_threshold,
        jaspar_release=jaspar_release, download_motifs=download_motifs,
        differential_statistics=differential_statistics,
        differential_checkpoint=differential_checkpoint,
        pseudocount=pseudocount, min_motif_peaks=min_motif_peaks, min_pair_peaks=min_pair_peaks,
        max_motifs_per_peak=max_motifs_per_peak, scanner_backend=scanner_backend,
        motif_batch_size=motif_batch_size, sequence_batch_bases=sequence_batch_bases, alpha=alpha_value,
        alpha_metric=alpha_metric,
        motifs_per_peak=motifs_per_peak,
        skip_motif_pairs=skip_motif_pairs,
        threads=threads,
        # Spawned threads are constrained to the max resources / --jobs: run_motif_analysis's own
        # worker pools (motif-hit export, motif-pair feature building) cap concurrency at
        # min(threads, jobs) when --jobs is narrower than --threads, so this reaches BOTH runners.
        jobs=jobs,
        stage=normalized_stage,
    )
    # --max-peak-width reaches the SLURM finalizer too, so the --stage singles hand-off it writes
    # records the width the user asked for.
    kwargs.update(max_peak_width=max_peak_width)
    if runner is run_motif_analysis:
        kwargs.update(
            precomputed_hit_database=precomputed_hit_database,
            precomputed_plan=precomputed_plan,
        )
    else:
        kwargs.update(dry_run=dry_run)
    result = runner(cpm, samples, out, **kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


def _shorten_numeric_str(value: str) -> str:
    """Round a numeric-looking string to 4 decimal places for terminal display.

    Real single-cell QC metadata (ATAC.weight, percent.mt, chrX_fraction,
    etc.) is often float64 full-precision (e.g. "0.44445333459084") or
    scientific notation (e.g. "7e-04") -- unreadable at a glance in a table
    of 5 examples per column. Whole-number-looking strings (cluster IDs like
    "11", counts like "16948") are left untouched so they don't grow a
    spurious ".0000".
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number != number:  # NaN
        return value
    if "." not in value and "e" not in value.lower() and "E" not in value:
        return value  # looks like a plain integer string; leave as-is
    return f"{number:.4f}"


def _format_singlecell_inspect_text(result: dict) -> str:
    """Human-readable rendering of run_singlecell_inspect()'s result.

    The raw result dict nests one entry per metadata column (often 15-25+
    for a real object), which reads as a wall of JSON on a terminal. This
    lays out the same information as an aligned plain-text table instead.
    """
    extraction = result.get("extraction", {})
    lines = []
    lines.append("Seurat extraction")
    lines.append(f"  rds:        {extraction.get('rds', '?')}")
    assay = extraction.get("assay", "?")
    slot = extraction.get("slot", "?")
    lines.append(f"  assay:      {assay}        slot: {slot}")
    seurat_version = extraction.get("seurat_version") or "?"
    signac_version = extraction.get("signac_version") or "?"
    lines.append(f"  versions:   Seurat {seurat_version} / Signac {signac_version}")
    lines.append(f"  cells:      {result.get('n_cells', '?')}        peaks: {result.get('n_peaks', '?')}")
    lines.append(f"  checkpoint: {extraction.get('checkpoint', '?')}")
    lines.append("")

    columns = result.get("columns", [])
    lines.append(f"Metadata columns ({len(columns)})")
    lines.append("")

    if columns:
        name_width = max(len("column"), max(len(str(c["column"])) for c in columns))
        distinct_width = max(len("distinct"), max(len(str(c["distinct_values"])) for c in columns))
        header = f"  {'column'.ljust(name_width)}  {'distinct'.rjust(distinct_width)}  grouping?  values"
        lines.append(header)
        lines.append(f"  {'-' * name_width}  {'-' * distinct_width}  ---------  " + "-" * 40)
        for entry in columns:
            name = str(entry["column"]).ljust(name_width)
            distinct = str(entry["distinct_values"]).rjust(distinct_width)
            grouping = "yes" if entry.get("likely_grouping_column") else "no"
            value_counts = entry.get("value_counts")
            if value_counts:
                values = ", ".join(
                    f"{_shorten_numeric_str(str(k))}={v}" for k, v in value_counts.items()
                )
            else:
                sample_values = entry.get("sample_values") or []
                shortened = [_shorten_numeric_str(v) for v in sample_values]
                values = ("e.g. " + ", ".join(shortened)) if shortened else ""
            lines.append(f"  {name}  {distinct}  {grouping.ljust(9)}  {values}")
    return "\n".join(lines)


@atac_app.command("singlecell-inspect")
def singlecell_inspect_command(
    rds: str = typer.Option(..., "--rds", help="Seurat/Signac .rds object (single-cell ATAC); or a folder of objects, written with a trailing /"),
    project: Optional[Path] = typer.Option(
        None, "--project",
        help="Analysis project directory. The shared data folder is the nearest data/ above it holding this object's extraction, else <project-parent>/data -- the SAME one the pipeline uses, so this reads that extraction instead of building its own. Use this, not --out.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Explicit shared data directory, used verbatim -- for standalone work outside a project. Not combined with --project.",
    ),
    assay: Optional[str] = typer.Option(None, "--assay", help="Seurat assay name; auto-detected when omitted"),
    slot: str = typer.Option("counts", "--slot", help="Seurat assay layer/slot to read"),
    genome: str = typer.Option(
        ..., "--genome",
        help="Genome key from the catalog (e.g. human_grch38). Required: this command may CREATE the extraction every later stage reuses, and the gene table it writes is annotated from this genome's GTF. Without it the extraction carries no Ensembl IDs and every downstream result inherits that.",
    ),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="Override the GTF --genome resolves to"),
    rscript: str = typer.Option("Rscript", "--rscript", help="Rscript executable"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Extract a Seurat/Signac object and report its metadata schema, before running singlecell-samples.

    Nothing in Paralome prompts interactively for comparison choices -- every
    other ATAC command (e.g. ``atac differential --factor``)
    already takes them as explicit flags, and this stays consistent with
    that. This command exists so those flag values (--replicate-column,
    --stratify-by, --factor and its control level) don't have to
    be guessed blind: it lists every cell-metadata column, how many distinct
    values it has, and (for low-cardinality columns -- the ones usable as a
    grouping/factor/stratification column) the value counts. Run this
    first, read off the column names and values you want to compare, then
    pass them to ``singlecell-samples``/``singlecell-differential``. Uses
    the same extraction checkpoint under --out, so a subsequent
    singlecell-samples run against the same --out does not re-invoke R.
    """
    # One object, or a folder of objects written with a trailing '/'.
    from paralome.singlecell.rds_input import parse_rds_option

    rds = parse_rds_option(rds)
    out = _resolve_data_out(project, out, command="singlecell-inspect", rds=rds)
    result = run_singlecell_inspect(
        rds, out, assay=assay, slot=slot, genome=genome, gtf=gtf,
        rscript=rscript, overwrite=overwrite,
    )
    print(_format_singlecell_inspect_text(result))


@atac_app.command("singlecell-samples")
def singlecell_atac_command(
    rds: str = typer.Option(..., "--rds", help="Seurat/Signac .rds object (single-cell ATAC; peaks already called); or a folder of objects, written with a trailing /"),
    project: Optional[Path] = typer.Option(
        None, "--project",
        help="The analysis project directory, exactly as passed to `paralome snATAC`. The shared "
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
    stratify_by: Optional[str] = typer.Option(
        None, "--stratify-by",
        help=STRATIFY_HELP,
    ),
    assay: Optional[str] = typer.Option(None, "--assay", help="Seurat assay name; auto-detected when omitted"),
    slot: str = typer.Option("counts", "--slot", help="Seurat assay layer/slot to read"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Registered genome bundle (resolves GTF for peak annotation, same as bulk ATAC/motif analysis; --gtf overrides)"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="GTF override for TSS annotation of peaks; normally resolved from --genome"),
    tss_upstream: int = typer.Option(2000, "--tss-upstream", min=0),
    tss_downstream: int = typer.Option(500, "--tss-downstream", min=0),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    min_cells_per_group: int = typer.Option(10, "--min-cells-per-group", min=1),
    rscript: str = typer.Option("Rscript", "--rscript", help="Rscript executable"),
    remove: List[str] = typer.Option(
        [], "--remove",
        help="Pseudobulk sample_id to exclude outright (repeatable) -- identify from a prior run's "
             "cell_counts_per_sample.csv or global-PCA plot, e.g. an outlier seen on the PCA.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit as a single SLURM job; returns immediately -- use `paralome status` or squeue/sacct to track it"),
    threads: int = typer.Option(1, "--threads", min=1, help="SLURM --cpus-per-task when --backend slurm"),
    mem: str = typer.Option("32G", "--mem", help="SLURM --mem when --backend slurm"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="A folder of objects with --backend slurm: objects extracted at once (0: all)"),
):
    """Read a Seurat/Signac single-cell ATAC object and aggregate it into bulk-style pseudobulk matrices.

    Only one file needs to be exported from R (``saveRDS(object, "x.rds")``);
    this command shells out to a bundled R helper to extract the peak set,
    counts matrix, and full cell metadata, then aggregates cells into
    pseudobulk samples using arbitrary metadata columns. The resulting
    raw_counts/cpm/log2_cpm matrices and sample manifest are drop-in
    compatible with ``atac differential`` and ``atac singlecell-differential``.

    There's no independent-sample axis here to array across (one Seurat
    object in, one pseudobulk aggregation out) -- --backend slurm queues the
    whole step as a single job instead of running it wherever the CLI was
    invoked from (e.g. a login node).
    """
    group_by_columns = pile_columns(replicate_column, stratify_by, retired_group_by=group_by)
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if normalized_backend == "local" and submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    if not pseudocount > 0:  # Refuse before submitting, not in the job
        raise typer.BadParameter(
            f"--pseudocount must be greater than 0 (got {pseudocount:g}); "
            "log2(CPM + pseudocount) has no value for a peak with no counts otherwise."
        )
    # One object, or a folder of objects written with a trailing '/'.
    from paralome.singlecell.rds_input import parse_rds_option, rds_argument

    rds = parse_rds_option(rds)
    out = _resolve_data_out(project, out, command="snATAC samples", rds=rds)
    if normalized_backend == "slurm":
        argv = _cli_argv() + [
            "atac", "singlecell-samples",
            "--rds", rds_argument(rds), "--out", str(out),
            "--replicate-column", group_by_columns[0],
            "--slot", slot, "--tss-upstream", str(tss_upstream), "--tss-downstream", str(tss_downstream),
            "--pseudocount", str(pseudocount), "--min-cells-per-group", str(min_cells_per_group),
            "--rscript", rscript, "--backend", "local",
        ]
        if stratify_by:
            argv += ["--stratify-by", stratify_by]
        if assay:
            argv += ["--assay", assay]
        if genome:
            argv += ["--genome", genome]
        if gtf:
            argv += ["--gtf", str(gtf)]
        for sample_id in remove:
            argv += ["--remove", sample_id]
        if overwrite:
            argv += ["--overwrite"]
        # A folder of objects is extracted one SLURM task per
        # object, --jobs at a time; this stage's job follows the array.
        after = None
        if Path(str(rds)).is_dir() and not overwrite:
            from paralome.pipeline.resource_catalog import resolve_genome_resources
            from paralome.singlecell.rds_input import submit_member_extraction

            after = submit_member_extraction(
                rds, layout_extraction_dir(Path(out)), workdir=Path(out), jobs=jobs, mem=mem,
                partition=partition, submit=submit,
                gtf=(resolve_genome_resources(genome=genome, overrides={"gtf": gtf}).path("gtf")
                     if genome else gtf),
                assay=assay, slot=slot, rscript=rscript,
            )
        result = _submit_single_job(
            # Was "sc_pseudobulk"; "samples", as the command
            # is named. ATAC carried no modality while RNA
            # carried "_rna_", so squeue showed an ATAC job under a name that
            # read as either: ATAC processes whose descriptor looked like RNA.
            job_name="sc_atac_samples", workdir=Path(out), argv=argv,
            threads=threads, mem=mem, partition=partition, submit=submit, after=after,
        )
        _print_submission(result)
        return
    result = run_singlecell_atac(
        rds, out,
        group_by=group_by_columns,
        replicate_column=group_by_columns[0],
        stratify_by=stratify_by,
        assay=assay,
        slot=slot,
        genome=genome,
        gtf=gtf,
        tss_upstream=tss_upstream,
        tss_downstream=tss_downstream,
        pseudocount=pseudocount,
        min_cells_per_group=min_cells_per_group,
        rscript=rscript,
        overwrite=overwrite,
        remove=remove or None,
    )
    for line in format_pseudobulk_summary(result, "peaks"):
        print(line)


def _resolve_data_paths(
    data_dir: Optional[Path],
    cpm: Optional[Path],
    samples: Optional[Path],
    extraction: Optional[Path] = None,
    *,
    need_extraction: bool,
) -> tuple:
    """Fill in --cpm/--samples/(--extraction) from --data-dir when given.

    singlecell-samples always writes cpm.tsv/sample_manifest.csv directly
    into its own --out, plus a singlecell_extraction/ subdirectory -- once
    that run is done, every downstream command needing its output is really
    just pointing at one directory. Rather than making the user retype three
    (or two) separately-tracked paths for the common case, --data-dir
    supplies all of them at once; any path also passed explicitly overrides
    the --data-dir-derived one, so mixing files from elsewhere still
    works. This mirrors the framework's existing preference for explicit
    stage-to-stage paths (no hidden auto-discovery across --out directories
    that weren't named) while removing the redundant typing once one
    directory is already known.
    """
    if data_dir is not None:
        # ATAC's own aggregated output (cpm.csv/sample_manifest.csv)
        # nests under data_dir/atac/ (directory-layout cleanup); the shared Seurat extraction stays directly
        # under data_dir, since it may also serve an `rna
        # singlecell-samples` run pointed at the same --out.
        cpm = cpm or (data_dir / "atac" / "cpm.csv")
        samples = samples or (data_dir / "atac" / "sample_manifest.csv")
        if need_extraction:
            extraction = extraction or layout_extraction_dir(data_dir)

    missing = []
    if cpm is None:
        missing.append("--cpm (or --data-dir)")
    if samples is None:
        missing.append("--samples (or --data-dir)")
    if need_extraction and extraction is None:
        missing.append("--extraction (or --data-dir)")
    if missing:
        raise typer.BadParameter("Missing required path(s): " + ", ".join(missing))
    return cpm, samples, extraction


def _format_singlecell_differential_text(result: dict) -> str:
    """Human-readable rendering of run_singlecell_differential()'s result.

    Routed through the shared formatter, fanned out or not, so scRNA and
    snATAC say the same thing about the same payload.
    """
    from paralome.singlecell.strata import format_differential_summary

    lines = []
    if int(result.get("schema_version") or 1) < 2:
        lines.append(
            f"Single-cell ATAC differential accessibility: "
            f"{result.get('control') or '-'} vs {result.get('treatment') or '-'}"
        )
        if result.get("stratify_by"):
            lines.append(f"  stratified by: {result.get('stratify_by')} = {result.get('stratify_value')}")
    lines.extend(format_differential_summary(result))
    lines.append(f"Checkpoint: {result.get('checkpoint', '?')}")
    return "\n".join(lines)


@atac_app.command("singlecell-differential")
def singlecell_differential_command(
    cpm: Optional[Path] = typer.Option(None, "--cpm", help="Peak CPM matrix TSV; derived from --data-dir if omitted"),
    samples: Optional[Path] = typer.Option(
        None, "--samples", help="Pseudobulk sample manifest; derived from --data-dir if omitted"
    ),
    extraction: Optional[Path] = typer.Option(
        None, "--extraction",
        help="Seurat extraction directory; supplies the per-cell counts any "
             "--statistic ...:cell needs; derived from --data-dir if omitted",
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir",
        help="The shared data directory -- supplies --cpm/--samples/--extraction automatically "
             "(atac/cpm.csv, atac/sample_manifest.csv, extraction/ underneath it) so they need not be typed separately",
    ),
    out: Path = typer.Option(..., "--out", help="Single-cell differential-accessibility output directory"),
    # --threads and --mem are PER STRATUM under --backend slurm.
    max_concurrent_strata: int = typer.Option(
        0, "--max-concurrent-strata", min=0,
        help="Cap simultaneously running stratum tasks (SLURM array %%K). 0 runs them all at once.",
    ),
    stratify_by: Optional[str] = typer.Option(
        None, "--stratify-by", help="Sample-manifest column to restrict this comparison to one value of, e.g. cell_type"
    ),
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
    statistic: str = typer.Option(
        ..., "--statistic",
        help=STATISTIC_HELP + ". Pseudobulk is the default and has no term; :cell after a "
             "statistic runs that statistic on the cells (folders t_cell, paired_t_cell, lm_cell, "
             "anova_cell), e.g. anova,anova:cell runs both levels.",
    ),
    pair_by: Optional[str] = typer.Option(
        None, "--pair-by",
        help="paired-t: the sample-manifest column whose shared value makes two samples a "
             "pair (one per level), e.g. the subject or litter.",
    ),
    covariate: Optional[str] = typer.Option(
        None, "--covariate",
        help="lm: a numeric sample-manifest column to regress accessibility against.",
    ),
    covariate_feature: Optional[str] = typer.Option(
        None, "--covariate-feature",
        help="lm: another feature to regress every feature against.",
    ),
    factor: List[str] = typer.Option(
        [], "--factor",
        help=FACTOR_HELP + ". The column is read from the pseudobulk sample manifest, which "
             "carries any cell-metadata column that is constant within a group.",
    ),
    naive_cell_t_peak_chunk_size: int = typer.Option(
        DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE, "--naive-cell-t-peak-chunk-size", min=1,
        help="How many peaks' per-cell counts the :cell statistics densify at once "
             "(memory only; changes no number).",
    ),
    pseudocount: float = typer.Option(1.0, "--pseudocount", min=0.0),
    min_mean_cpm: float = typer.Option(1.0, "--min-mean-cpm", min=0.0),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut a next stage selects by (e.g. motif peak sets), P or Q and a value: P0.05 (default) or Q0.10. Paralome writes no call of significance; the viewer applies the user's thresholds",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit as a single SLURM job; returns immediately -- use `paralome status` or squeue/sacct to track it"),
    threads: int = typer.Option(
        1, "--threads", min=1,
        help="Worker processes for the :cell statistics (the pseudobulk statistics are "
             "vectorised and do not use them); also SLURM --cpus-per-task when --backend slurm",
    ),
    jobs: int = typer.Option(0, "--jobs", min=0, help="Unused for this command -- there is no per-sample array dimension to throttle"),
    mem: str = typer.Option("32G", "--mem", help="SLURM --mem when --backend slurm"),
    partition: Optional[str] = typer.Option(None, "--partition"),
):
    """Run single-cell ATAC differential accessibility.

    --statistic is any combination of t, paired-t, lm and anova, and each
    writes its own folder of results under --out, named by the flag:

    \b
      * --statistic t             Student's t on the pseudobulk values, one per
        (the default)             sample (folder t). Exactly what bulk ATAC
                                  writes for the same comparison.
      * paired-t, lm, anova       the paired t (--pair-by), the regression
                                  (--covariate / --covariate-feature) and the
                                  ANOVA (one --factor: the one-way F with its
                                  linear trend and lack of fit, and each level
                                  against the one before; two: the 2x2's three
                                  F-tests and six contrasts), on the pseudobulk
                                  values.
      * <statistic>:cell          that statistic with the cells as the
                                  observations (t_cell, paired_t_cell, lm_cell,
                                  anova_cell), the cells as N and each sample's
                                  cells pooled with its replicates'; a list may
                                  hold both levels (anova,anova:cell).

    --cpm/--samples/--extraction all live under the same singlecell-samples
    --out directory -- pass that directory as --data-dir once.
    """
    if stratify_value and stratify_skip:
        raise typer.BadParameter(
            "--stratify-value names the strata to RUN and --stratify-skip names the "
            "strata to EXCLUDE; pass one or the other, not both."
        )
    alpha_metric, alpha_value = _parse_alpha(alpha)
    # The shared parser, given everything it validates against, so
    # anova/lm/paired-t without their companion option are refused before a
    # stage is queued.
    try:
        parse_singlecell_request(
            statistic, factors=list(factor), pair_by=pair_by,
            covariate=covariate, covariate_feature=covariate_feature,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    # --stratify-by alone fans out over every value of that column.
    # --stratify-value narrows the fan-out to one stratum, which is what the
    # SLURM array tasks pass.
    cpm, samples, extraction = _resolve_data_paths(
        data_dir, cpm, samples, extraction, need_extraction=True,
    )
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise typer.BadParameter("--backend must be local or slurm")
    if normalized_backend == "local" and submit:
        raise typer.BadParameter("--submit requires --backend slurm")
    if normalized_backend == "slurm":
        # One CPM matrix + one extraction in, one set of statistic folders out:
        # the "single non-array job" wrapper, or one array task per stratum.
        argv = _cli_argv() + [
            "atac", "singlecell-differential",
            "--cpm", str(cpm), "--samples", str(samples), "--extraction", str(extraction), "--out", str(out),
            "--statistic", statistic,
            *[token for name in factor for token in ("--factor", str(name))],
            "--naive-cell-t-peak-chunk-size", str(naive_cell_t_peak_chunk_size),
            "--pseudocount", str(pseudocount), "--min-mean-cpm", str(min_mean_cpm), "--alpha", alpha,
            # --threads reaches the command that runs inside the job, so the
            # :cell fits use the CPUs the job reserved.
            "--threads", str(threads),
            "--backend", "local",
        ]
        # Possibly-None values go through _relay_opt, never into the argv literal.
        _relay_opt(argv, "--pair-by", pair_by)
        _relay_opt(argv, "--covariate", covariate)
        _relay_opt(argv, "--covariate-feature", covariate_feature)
        if stratify_by:
            argv += ["--stratify-by", stratify_by]
        if stratify_value:
            argv += ["--stratify-value", stratify_value]
        if min_cells_per_stratum:
            argv += ["--min-cells-per-stratum", str(min_cells_per_stratum)]
        if overwrite:
            argv += ["--overwrite"]
        # One SLURM task per stratum. --threads and --mem are PER
        # STRATUM; see submit_stratum_array and the scRNA twin in commands/rna.py.
        if stratify_by and not split_stratify_values(stratify_value):
            plan = plan_strata(
                sample_manifest=samples, stratify_by=stratify_by,
                stratify_skip=split_remove_values([stratify_skip]) if stratify_skip else None,
                min_cells_per_stratum=min_cells_per_stratum,
                statistic=statistic, factors=list(factor), pair_by=pair_by,
                covariate=covariate, covariate_feature=covariate_feature,
            )
            # One token per submission, recorded in the plan and
            # exported into every array task; see strata.SUBMISSION_ENV.
            submission = new_submission_token()
            index_path = stage_checkpoint_path(Path(out), "singlecell_differential")
            plan_path = write_gather_plan(
                Path(out) / "scripts" / "sc_atac_diff.plan.json",
                stage="singlecell_differential",
                submission=submission,
                sample_manifest=str(samples), output_dir=str(out),
                checkpoint_path=str(index_path),
                file_prefix=derive_file_prefix(Path(out)),
                stratify_by=stratify_by,
                stratify_skip=split_remove_values([stratify_skip]) if stratify_skip else None,
                statistic=statistic, factors=list(factor), pair_by=pair_by,
                covariate=covariate, covariate_feature=covariate_feature,
                min_cells_per_stratum=min_cells_per_stratum,
            )
            result = submit_stratum_array(
                job_name="sc_atac_diff", workdir=Path(out), argv=argv,
                values=plan.runnable,
                gather_argv=_cli_argv() + [
                    "atac", "singlecell-differential-gather", "--plan", str(plan_path),
                ],
                threads=threads, mem=mem, partition=partition,
                submission=submission, index_checkpoint=index_path,
                submit=submit, throttle=max_concurrent_strata or None,
            )
            _print_submission(result)
            return

        # Relayed only to the single job. The stratum array's plan has
        # already applied the skip, and each task's --stratify-value excludes it.
        if stratify_skip:
            argv += ["--stratify-skip", stratify_skip]
        result = _submit_single_job(
            job_name="sc_atac_diff", workdir=Path(out), argv=argv,
            threads=threads, mem=mem, partition=partition, submit=submit,
        )
        _print_submission(result)
        return
    cli_warning_log = Path(out).expanduser().resolve() / "logs" / "cli_warnings.log"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_singlecell_differential(
            cpm, samples, extraction, out,
            statistic=statistic,
            stratify_by=stratify_by,
            stratify_value=stratify_value,
            stratify_skip=stratify_skip,
            min_cells_per_stratum=min_cells_per_stratum,
            factors=list(factor),
            pair_by=pair_by,
            covariate=covariate,
            covariate_feature=covariate_feature,
            naive_cell_t_peak_chunk_size=naive_cell_t_peak_chunk_size,
            alpha=alpha_value,
            alpha_metric=alpha_metric,
            pseudocount=pseudocount,
            min_mean_cpm=min_mean_cpm,
            overwrite=overwrite,
            jobs=threads,
        )
    if caught:
        cli_warning_log.parent.mkdir(parents=True, exist_ok=True)
        header_line = f"{len(caught)} warning(s) raised outside the per-feature model fits\n\n"
        body = "\n".join(f"{w.category.__name__}: {w.message}" for w in caught)
        cli_warning_log.write_text(header_line + body + "\n")

    print(_format_singlecell_differential_text(result))
    if caught:
        print(f"\n{len(caught)} additional warning(s) written to {cli_warning_log}")


def _format_collect_text(result: dict) -> str:
    lines = []
    lines.append(f"Project:      {result.get('project', '?')}")
    lines.append(f"Destination:  {result.get('destination', '?')}")
    lines.append(f"Filter:       {result.get('filter', '*')}")
    if result.get("dry_run"):
        lines.append("(dry run -- nothing was written)")
    lines.append("")
    # Two numbers, each labelled for what it counts: every file walked, and the files the filter
    # matched.
    # Under SLURM the plan is printed before anything has been copied, so it speaks in the future
    # too -- until the finalize job, which has checked the files and says what happened.
    pending = result.get("dry_run") or (result.get("backend") == "slurm" and "complete" not in result)
    verb = "would copy" if result.get("dry_run") else ("to copy" if pending else "copied")
    version_verb = "would version" if result.get("dry_run") else ("to version" if pending else "versioned")
    width = max(len(verb), len(version_verb), len("unchanged")) + 1
    lines.append(f"Files walked:            {result.get('files_scanned', 0)}")
    lines.append(f"Files matching --filter: {result.get('files_matched', 0)}")
    lines.append(f"  {(verb + ':').ljust(width)} {result.get('copied', 0)}")
    lines.append(f"  {'unchanged:'.ljust(width)} {result.get('unchanged', 0)}")
    lines.append(f"  {(version_verb + ':').ljust(width)} {result.get('versioned', 0)}")
    interesting = [e for e in result.get("entries", []) if e["status"] not in ("unchanged", "skipped")]
    if interesting:
        lines.append("")
        lines.append("Copied/versioned files:")
        for entry in interesting:
            lines.append(f"  [{entry['status']}] {entry['destination']}")

    # A rename means two DIFFERENT files wanted the same name. That is right for "collected again on
    # a later day" and wrong for "different comparisons that happen to be named alike", so the source
    # path of every renamed file is printed: it is the one thing that still distinguishes them.
    versioned = [e for e in result.get("entries", []) if e["status"] in ("versioned", "would_version")]
    if versioned:
        lines.append("")
        lines.append(
            f"WARNING: {len(versioned)} file(s) were renamed because a DIFFERENT "
            f"file already had that name."
        )
        lines.append(
            "  If these came from separate comparisons, the new names carry no "
            "record of which is which -- the numbering is collection order. The "
            "source path below is the only remaining distinction."
        )
        for entry in versioned:
            lines.append(f"  {entry['destination']}")
            lines.append(f"      <- {entry['source']}")

    # Matched names that could not be copied (dangling symlink, not a regular file).
    skipped = [e for e in result.get("entries", []) if e["status"] == "skipped"]
    if skipped:
        lines.append("")
        lines.append(f"WARNING: {len(skipped)} matching name(s) were skipped: not a readable regular file "
                     "(e.g. a symlink whose target is gone).")
        for entry in skipped:
            lines.append(f"  {entry['source']}")

    # What a SLURM collect did with the plan above.
    if result.get("backend") == "slurm" and "complete" not in result:
        lines.append("")
        if not result.get("plan"):
            lines.append("Nothing to copy; no jobs were needed.")
        elif result.get("submitted"):
            lines.append(f"Submitted: copy array {result['array_job_id']} ({result['array_tasks']} task(s), "
                         f"{result['jobs']} at a time), then finalize job {result['finalize_job_id']}.")
            lines.append(f"  logs:   {result['logs']}")
            lines.append("  The finalize log ends with the summary, and lists any file that did not arrive.")
        else:
            lines.append("Not submitted (no --submit). The plan and scripts are written:")
            lines.append(f"  plan:    {result['plan']}")
            for script in result.get("scripts", []):
                lines.append(f"  script:  {script}")
    if result.get("missing"):
        lines.append("")
        lines.append(f"ERROR: {len(result['missing'])} file(s) did not arrive, or arrived short:")
        for path in result["missing"]:
            lines.append(f"  {path}")
        lines.append("  Run the same collect again: files already collected are skipped.")
    elif result.get("complete"):
        lines.append("")
        lines.append("Every planned file is at its destination, at its full size.")
    return "\n".join(lines)


@atac_app.command("collect")
def collect_command(
    project: Path = typer.Option(..., "--project", help="Project directory to walk for results/ output"),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Destination directory; defaults to a sibling '<project>_collected' directory",
    ),
    # REQUIRED, WITH NO DEFAULT. A default of "*" (every filename) would make
    # forgetting the option copy the whole project into a flat folder; a default
    # like "csv" would silently drop every figure and table that is not a .csv.
    # Either looks like a successful run. Collecting everything is still
    # available and still spelled "*"; it just has to be asked for. Note the
    # match is a SUBSTRING of the filename, not an extension: "csv" also matches
    # csv_summary.pdf.
    filter_text: str = typer.Option(
        ..., "--filter",
        help='Case-insensitive substring to match against filenames, e.g. "evidome". '
             'Pass "*" -- quoted -- to take every result file.',
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be collected without copying anything"),
    # Large collections (e.g. 80 files of 20 GB) are copied
    # in batches of X at a time, similar to how --threads and --jobs work
    # together. --jobs is how many run at once, as
    # everywhere else; a copy uses one CPU, so there is no --threads.
    jobs: int = typer.Option(
        4, "--jobs", min=1,
        help="Files copied at once (default 4). The storage sets the ceiling, not the CPUs: more copies than "
             "the disks can serve only share the same throughput.",
    ),
    backend: str = typer.Option(
        "local", "--backend",
        help="local: copy here, --jobs at a time. slurm: one array task per file, --jobs at a time, then a "
             "job that checks every file arrived.",
    ),
    submit: bool = typer.Option(False, "--submit", help="With --backend slurm: submit the jobs (otherwise the "
                                                        "plan and scripts are written and nothing runs)"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    slurm_node: Optional[str] = typer.Option(None, "--slurm-node", help="Optional SLURM node constraint, e.g. "
                                                                         "the node that owns the disks"),
    mem: str = typer.Option("2G", "--mem", help="Memory per SLURM copy task"),
):
    """Copy files from anywhere under --project into one flat folder.

    Walks the whole --project subtree and copies every file whose name
    matches --filter into --out (default: "<project>_collected", a sibling
    of --project). No directory name is special: --filter is required and
    is the only control, so --filter "*" copies everything, including
    .sbatch scripts, SLURM .err logs and checkpoint JSON.

    Every file lands directly in that destination directory by its own
    filename -- no mirrored subdirectory tree. This relies on every
    filename already being unique via its comparison's own file-prefix
    convention, so result files from different comparisons/analyses do
    not collide. Files that are not results do: with --filter "*" the
    summary.json, plan.json and checkpoint JSON of several stage folders
    all share a name, and the second and later ones are written under a
    dated name by the versioning rule below. The report lists every
    renamed file with its source path, which is the only record of which
    is which.

    Source files are never modified, moved, or deleted -- this is a
    copy-only operation. Re-running is safe and idempotent: an unchanged
    file is skipped (its checksum already matches a copy already
    collected, under the plain name or any dated one); a file whose
    content changed since the last collect, or a different file of the
    same name, is written alongside the previously-collected copy with
    today's date inserted before the extension (e.g. "foo.csv" ->
    "foo_2026-07-28.csv"), so nothing already collected is ever silently
    overwritten.

    Use --filter to collect only files whose name contains specific text --
    e.g. "atac collect --project my_project --filter evidome" pulls out just
    the *_evidome_input_*.csv files from every sub-analysis in one pass,
    instead of every workbook, plot, and report alongside them. Use --dry-run first on a large project
    to preview what would be copied/versioned before committing to it.
    """
    if not filter_text.strip():
        raise typer.BadParameter(
            "needs something to search for. Pass a substring of the filenames "
            'you want, e.g. --filter evidome, or --filter "*" to take every '
            "result file.",
            param_hint="--filter",
        )
    if backend not in ("local", "slurm"):
        raise typer.BadParameter("is local or slurm", param_hint="--backend")
    if backend == "slurm" and not dry_run:
        result = submit_collect(project, out, filter_text=filter_text, jobs=jobs, partition=partition,
                                node=slurm_node, mem=mem, submit=submit)
        print(_format_collect_text(result))
        return
    if submit:
        raise typer.BadParameter("goes with --backend slurm", param_hint="--submit")
    result = run_collect(project, out, filter_text=filter_text, dry_run=dry_run, jobs=jobs)
    print(_format_collect_text(result))


@atac_app.command("extract-member", hidden=True)
def extract_member_command(
    plan: Path = typer.Option(..., "--plan", help="rds_members_plan.json written for a folder's extraction"),
    index: int = typer.Option(..., "--index", min=1, help="Array task number"),
):
    """One SLURM array task of a folder's extraction: one object, as the folder run extracts it."""
    from paralome.singlecell.rds_input import run_member_task

    result = run_member_task(plan, index)
    print(f"task {index}: {result.get('n_cells')} cells extracted")


@atac_app.command("collect-task", hidden=True)
def collect_task_command(
    plan: Path = typer.Option(..., "--plan", help="plan.json written by collect --backend slurm"),
    index: int = typer.Option(..., "--index", min=1, help="Array task number"),
):
    """One SLURM array task of a collect: copy this task's share of the plan."""
    counts = run_collect_task(plan, index)
    print(f"task {index}: copied {counts['copied']}, already there {counts['already_there']}")


@atac_app.command("collect-finalize", hidden=True)
def collect_finalize_command(
    plan: Path = typer.Option(..., "--plan", help="plan.json written by collect --backend slurm"),
):
    """After a collect array: check every planned file arrived, and print the summary."""
    result = finalize_collect(plan)
    print(_format_collect_text(result))
    if result["missing"]:
        raise typer.Exit(1)


@atac_app.command("differential-prepare")
def differential_prepare_command(
    plan: Path = typer.Option(..., "--plan"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    print(json.dumps(
        run_differential_accessibility_prepare_task(plan, overwrite=overwrite),
        indent=2, sort_keys=True,
    ))


@atac_app.command("differential-block")
def differential_block_command(
    plan: Path = typer.Option(..., "--plan"),
    task_id: int = typer.Option(..., "--task-id", min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    print(json.dumps(
        run_differential_accessibility_block_task(plan, task_id, overwrite=overwrite),
        indent=2, sort_keys=True,
    ))


@atac_app.command("differential-finalize")
def differential_finalize_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(
        run_differential_accessibility_finalize_task(plan),
        indent=2, sort_keys=True,
    ))


@atac_app.command("motif-prepare")
def motif_prepare_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_motif_prepare_task(plan), indent=2, sort_keys=True))


@atac_app.command("motif-shard")
def motif_shard_command(
    plan: Path = typer.Option(..., "--plan"),
    task_id: int = typer.Option(..., "--task-id", min=1),
):
    print(json.dumps(run_motif_scan_task(plan, task_id), indent=2, sort_keys=True))


@atac_app.command("motif-finalize")
def motif_finalize_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_motif_finalize_task(plan), indent=2, sort_keys=True))

@atac_app.command("peak-merge-sample")
def peak_merge_sample_command(
    plan: Path = typer.Option(..., "--plan"),
    task_id: int = typer.Option(..., "--task-id", min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    print(json.dumps(run_peak_merge_sample_task(plan, task_id, overwrite=overwrite), indent=2, sort_keys=True))


@atac_app.command("peak-merge-finalize")
def peak_merge_finalize_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(run_peak_merge_finalize_task(plan), indent=2, sort_keys=True))

@atac_app.command("peak-quant-prepare")
def peak_quant_prepare_command(
    plan: Path = typer.Option(..., "--plan"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    print(json.dumps(
        run_peak_quantification_prepare_task(plan, overwrite=overwrite),
        indent=2, sort_keys=True,
    ))


@atac_app.command("peak-quant-sample")
def peak_quant_sample_command(
    plan: Path = typer.Option(..., "--plan"),
    task_id: int = typer.Option(..., "--task-id", min=1),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    print(json.dumps(
        run_peak_quantification_sample_task(plan, task_id, overwrite=overwrite),
        indent=2, sort_keys=True,
    ))


@atac_app.command("peak-quant-finalize")
def peak_quant_finalize_command(plan: Path = typer.Option(..., "--plan")):
    print(json.dumps(
        run_peak_quantification_finalize_task(plan),
        indent=2, sort_keys=True,
    ))


@atac_app.command("motif-partners")
def motif_partner_shift_command(
    project: Path = typer.Option(
        ..., "--project",
        help="The project directory the comparisons live under, e.g. the snATAC/ folder "
             "containing StateA_vs_CTL, StateB_vs_CTL and StateC_vs_CTL. Same sense the "
             "bulk pipeline uses --project in: the directory its sub-analyses sit under.",
    ),
    comparisons: str = typer.Option(
        ..., "--comparisons",
        help="ORDERED, comma-delimited comparison directory names under --project, e.g. "
             "StateA_vs_CTL,StateB_vs_CTL,StateC_vs_CTL. The order is the progression: "
             "the first is the baseline and each later one a successive state, which is what "
             "makes 'acquired' and 'lost' meaningful rather than merely 'different somewhere'.",
    ),
    out: Path = typer.Option(..., "--out", help="Directory to write the analysis into."),
    anchor: Optional[str] = typer.Option(
        None, "--anchor",
        help="Motif to treat as the anchor whose partners are examined, e.g. IRF (matches "
             "IRF1..IRF9 and composites like STAT2::IRF9) or ^IRF3$ to pin one member. "
             "Omit for an unguided run over every pair -- same computation, no filter.",
    ),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut that selects each comparison's peak buckets and the pairs whose loci and genes are "
             "detailed, P or Q and a value: P0.05 (default) or Q0.10. The pair and contrast tables are "
             "written whole.",
    ),
    threads: int = typer.Option(
        4, "--threads", min=1,
        help="Worker processes for reading the per-stratum tables. The files are read in "
             "two passes and both parallelise over them, so this scales close to linearly "
             "until the filesystem saturates. Set it to your SLURM allocation, not the "
             "node's core count.",
    ),
    no_overwrite: bool = typer.Option(
        False, "--no-overwrite",
        help="Reuse an output already in --out when it is recorded as a finished output of "
             "the same inputs and settings (anchor, alpha, top-pairs, tables read). For the "
             "locus export and the gene annotation this skips the WORK, not just the write -- "
             "those are the expensive ones, and re-running to add a single table should not "
             "cost the hours the first run did. A file left by an interrupted run, an earlier "
             "version or different settings is recomputed, and says so. Every reused path is "
             "printed and recorded in the checkpoint.",
    ),
    loci_max_rows: int = typer.Option(
        50_000_000, "--loci-max-rows", min=0,
        help="Refuse to write a locus export larger than this many rows, after reporting the "
             "size it would have been. 0 disables the guard. Exists because an unbounded "
             "export once wrote for nine hours and was only discovered by looking at the "
             "filesystem.",
    ),
    top_pairs: int = typer.Option(
        25, "--top-pairs", min=0,
        help="How many changed pairs get detailed annotation -- BOTH the locus export and the "
             "gene table -- ranked by effect_lower_bound. Defaults to 25 because "
             "annotating every pair that clears the cut ran to 79.9 million gene rows, 10.9 GB "
             "and 97 GB of RAM on a three-state project, which is not what anyone wants from a "
             "first run. The run reports how many it left out. Pass 0 for all of them. The "
             "motif_instances table is found automatically -- every comparison writes an "
             "identical copy, so there is nothing to point at.",
    ),
):
    """How an anchor motif's co-occurring partners change ACROSS comparisons.

    A secondary analysis: it runs after the primary ATAC analyses are done and
    evaluated, reads their written outputs, and belongs to no stage graph.

    Every comparison in a project scans the same cohort peak atlas with the same
    motif library, so each pair's co-occurring peak set is fixed and identical
    between comparisons. A motif cannot move and a peak cannot widen between
    them, because there is one of each. What varies is only which of those fixed
    peaks a comparison called significant -- so this reports the FRACTION of each
    pair's peak set that a comparison captured, against the rate that comparison
    opened peaks generally. Indiscriminate opening scores near 1.
    """
    from paralome.atac.motif_partners import MotifPartnerError, run_motif_partner_shift

    metric, alpha_value = _parse_alpha(alpha)
    names = [item.strip() for item in str(comparisons).split(",") if item.strip()]
    try:
        result = run_motif_partner_shift(
            project, names, out,
            anchor=anchor,
            alpha=alpha_value, alpha_metric=metric,
            threads=threads,
            top_pairs=top_pairs,
            loci_max_rows=loci_max_rows,
            overwrite=not no_overwrite,
        )
    except MotifPartnerError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2)
    # One fork per statistic folder (and comparison), each its own result.
    for fork_key, fork in dict(result["forks"]).items():
        print(f"\n== {fork_key or 'results/'} ==")
        print(f"Pairs examined : {fork['pairs_examined']}")
        print(f"Pairs reported : {fork['pairs_reported']}")
        if fork.get("changed_pairs_total"):
            print(f"Changed pairs  : {fork['changed_pairs_total']:,} "
                  f"(detailed {fork.get('pairs_detailed', 0):,})")
        print(f"\nWrote {fork['motif_partner_shift']}")
        for key in ("motif_partner_statistics", "motif_partner_cut",
                    "motif_partner_contrasts", "motif_partner_contrasts_cut",
                    "motif_partner_genes",
                    "motif_partner_loci"):
            if fork.get(key):
                print(f"      {fork[key]}")
    for note in result.get("forks_skipped") or []:
        print(f"\nNot analysed: {note}")


@atac_app.command("singlecell-differential-gather")
def singlecell_differential_gather_command(
    plan: Path = typer.Option(..., "--plan", help="Plan file written by the array submitter"),
) -> None:
    """Index the per-stratum results an array produced. Machine surface.

    The snATAC twin of `rna singlecell-differential-gather`; see that command.
    """

    payload = run_gather_from_plan(plan)
    print(f"Indexed {len(payload.get('strata') or {})} stratum result(s) -> {payload.get('checkpoint')}")


# ---------------------------------------------------------------------------------------------------------------
# The genome_tracks stage -- see paralome.atac.genome_tracks.


@atac_app.command("genome-tracks")
def genome_tracks_command(
    differential_checkpoint: Path = typer.Option(..., "--differential-checkpoint",
                                                 help="differential_accessibility's checkpoint (its tested peaks)"),
    samples: Path = typer.Option(..., "--samples", help="Deduplicated BAM manifest (sample_id, condition, bam): the profiles drawn"),
    coverage_summary: Path = typer.Option(..., "--coverage-summary",
                                          help="peak_quantification's summary: each sample's eligible fragments"),
    out: Path = typer.Option(..., "--out", help="Stage directory (genome_tracks)"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Project prefix for the zip's name; default: from --out"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Registered genome (its GTF for the gene track)"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="GTF for the gene track; overrides --genome's"),
    threads: int = typer.Option(1, "--threads", min=1, help="BAMs read at once"),
    bundled_tracks: Optional[Path] = typer.Option(None, "--bundled-tracks", help="The browser_bundle stage folder of an older project, whose tracks are taken over when they still answer"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the project's omicview/<stage> folder (default: the stage folder's results/)"),
    former_tracks: Optional[Path] = typer.Option(None, "--former-tracks", help="snATAC's genome_tracks folder in an older project (<project>/atac/genome_tracks), whose tracks are moved here when they still answer"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Build again even when the zip answers this request"),
) -> None:
    """Bulk ATAC: the viable peaks, each comparison's peaks and one 25 bp CPM profile per sample, in one zip."""
    from paralome.atac.genome_tracks import adopt_bundled_tracks, run_bulk_genome_tracks

    name = prefix or derive_file_prefix(Path(out).resolve())
    adopt_bundled_tracks(bundled_tracks, out, name, "ATAC", zip_dir=zip_dir,
                         echo=lambda text: print(text, flush=True, file=sys.stdout))
    run_bulk_genome_tracks(
        differential_checkpoint=differential_checkpoint, samples=samples, coverage_summary=coverage_summary,
        workdir=out, prefix=name, genome=genome, gtf=gtf,
        threads=threads, overwrite=overwrite, zip_dir=zip_dir, echo=lambda text: print(text, flush=True, file=sys.stdout),
    )


@atac_app.command("singlecell-genome-tracks")
def singlecell_genome_tracks_command(
    differential_checkpoint: Path = typer.Option(..., "--differential-checkpoint",
                                                 help="singlecell_differential's checkpoint (its tested peaks, every stratum)"),
    samples: Path = typer.Option(..., "--samples", help="The pseudobulk sample manifest: the profiles drawn"),
    extraction: Path = typer.Option(..., "--extraction", help="The single-cell extraction (cell_metadata.tsv)"),
    rds: Path = typer.Option(..., "--rds", help="The Seurat/Signac object, or folder of objects, whose fragment files are read"),
    replicate_column: Optional[str] = typer.Option(None, "--replicate-column", help="The --replicate-column the pseudobulk samples were made with"),
    group_by: Optional[str] = typer.Option(None, "--group-by", hidden=True, help=RETIRED_GROUP_BY_HELP),
    out: Path = typer.Option(..., "--out", help="Stage directory (genome_tracks)"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Project prefix for the zip's name; default: from --out"),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", help="The run's --stratify-by, recorded on each track"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Registered genome (its GTF and reference)"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="GTF for the gene track; overrides --genome's"),
    reference: Optional[Path] = typer.Option(None, "--reference", help="Reference FASTA (its .fai gives contig lengths)"),
    rscript: str = typer.Option("Rscript", "--rscript", help="Rscript executable"),
    bundled_tracks: Optional[Path] = typer.Option(None, "--bundled-tracks", help="The browser_bundle stage folder of an older project, whose tracks are taken over when they still answer"),
    zip_dir: Optional[Path] = typer.Option(None, "--zip-dir", help="Where the zip goes: the project's omicview/<stage> folder (default: the stage folder's results/)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Build again even when the zip answers this request"),
) -> None:
    """snATAC: the viable peaks, each comparison's peaks and one 25 bp CPM profile per pseudobulk sample."""
    from paralome.atac.genome_tracks import adopt_bundled_tracks, run_singlecell_genome_tracks

    name = prefix or derive_file_prefix(Path(out).resolve())
    adopt_bundled_tracks(bundled_tracks, out, name, "snATAC", zip_dir=zip_dir,
                         echo=lambda text: print(text, flush=True, file=sys.stdout))
    run_singlecell_genome_tracks(
        differential_checkpoint=differential_checkpoint, samples=samples, extraction=extraction, rds=rds,
        group_by=pile_columns(replicate_column, stratify_by, retired_group_by=group_by), stratify_by=stratify_by,
        workdir=out, prefix=name, genome=genome, gtf=gtf,
        reference=reference, rscript=rscript, overwrite=overwrite, zip_dir=zip_dir,
        echo=lambda text: print(text, flush=True, file=sys.stdout),
    )
