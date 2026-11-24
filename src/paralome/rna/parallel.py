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

import csv
import json
import os
from paralome import __version__
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from paralome.checkpoints import stage_checkpoint_path
from paralome.pipeline.jsonio import input_fingerprint, invalidate_checkpoint, write_json
from paralome.pipeline.slurm import SlurmOptions, afterany, afterok, compress_array_ids, sbatch, write_sbatch
from paralome.rna.fpkm import (
    _read_sample_rows as _read_fpkm_sample_rows,
    _read_samples as _read_fpkm_samples, _request_fingerprint as _fpkm_request_fingerprint,
    _write_matrix, genes_left_out_summary, quantification_fingerprints,
    kept_result, quantification_input_stamps, quantification_stale_reason,
    quantify_fpkm, write_genes_left_out, LIBRARY_SIZE_RULE,
)
from paralome.rna.native_counter import STRANDEDNESS_ASSIGNMENT_RULES, STRANDEDNESS_LABELS
from paralome.rna.preflight import preflight_rna_quantification, project_bam_index_dir
from paralome.rna.differential import (
    _raw_fpkm,
    _read_matrix,
    _read_samples as _read_diff_samples,
    check_pseudocount,
    check_pseudocount_matches_matrix,
    check_request_options,
    covariate_feature_values,
    differential_request_fingerprint,
    differential_stale_reason,
    fit_blocks,
    plan_bulk_differential,
    power_alpha,
    say_kept_without_matrix,
    write_differential_result,
)
from paralome.binaries import worker_command


def _cli() -> str:
    """The worker invocation embedded in every generated RNA sbatch body.

    Resolved by the same function every other sbatch writer uses
    (``binaries.worker_argv``): the installed console script, else the module
    through the current interpreter.
    """
    return worker_command()


def _announce_job(job_id: str) -> str:
    """Say a job ID the moment sbatch returns it, in the supervisor's format.

    So when a later submission is refused, the supervisor -- which reads IDs
    from the output of a failed command too -- still knows the earlier jobs
    are in the queue. On stderr, so stdout stays the one JSON document it is.
    """
    print(f"Job ID: {job_id}", file=sys.stderr, flush=True)
    return job_id


def _replaces(checkpoint_path: Path, reason: str) -> Dict[str, str]:
    """The finished result a plan would replace, for _withdraw_replaced."""
    return {"checkpoint": str(checkpoint_path), "reason": reason}


def _withdraw_replaced(info: Mapping[str, object]) -> None:
    """Mark the result this plan replaces as withdrawn. Called only to submit.

    Done when the first job is in the queue, not when the plan is made, so a
    plan made WITHOUT --submit takes nothing away. From then on, and for good
    if the array fails (finalize is afterok), nothing downstream carries on
    from the old result. If sbatch refuses that first job (a mistyped
    --partition), nothing was submitted and the finished result stands.
    """
    replaced = info.get("replaces")
    if not isinstance(replaced, Mapping):
        return
    path = Path(str(replaced["checkpoint"]))
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    if previous.get("complete"):
        invalidate_checkpoint(path, previous, str(replaced["reason"]))


def _submit_chain(info: Mapping[str, object]) -> List[str]:
    """Prepare, then the array after it, then finalize after the array."""
    prepare = _announce_job(sbatch(info["prepare_script"]))
    # The prepare job writes no result; nothing else can start before this.
    _withdraw_replaced(info)
    array = _announce_job(sbatch(info["array_script"], dependency=afterok(prepare)))
    finalize = _announce_job(sbatch(info["finalize_script"], dependency=afterok(array)))
    return [prepare, array, finalize]


def _already_complete(stage: str, checkpoint_path: Path, out: Path) -> Dict[str, object]:
    """What a plan returns when the finished result answers the request."""
    print(
        f"[paralome] {stage}: the finished result in {out} answers this request; "
        "nothing is planned or submitted.",
        file=sys.stderr,
    )
    return {"reused": True, "complete": True, "checkpoint": checkpoint_path,
            "output_dir": out, "task_count": 0}


def _write_matrix_atomic(path: Path, metadata, values) -> None:
    """A merged matrix appears whole or not at all."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        _write_matrix(temporary, metadata, values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resource_split(threads: int, jobs: int, tasks: int) -> tuple[int, int]:
    concurrency = min(max(1, jobs or tasks), max(1, tasks), max(1, threads))
    return concurrency, max(1, int(threads) // concurrency)


def prepare_fpkm_slurm_plan(
    samples: Path,
    gtf: Path,
    out: Path,
    *,
    threads: int,
    jobs: int,
    strandedness: int,
    paired_end: bool,
    require_both_ends_mapped: bool,
    exclude_chimeric_fragments: bool,
    pseudocount: float,
    partition: Optional[str],
    mem: str,
    overwrite: bool,
) -> Dict[str, object]:
    out = Path(out).resolve()
    # The decision quantify_fpkm makes, made here by the same function and, as there, BEFORE preflight: a
    # finished result whose BAMs have since been deleted still answers the request it was made for, and a
    # changed request does not.
    fingerprint, request = quantification_fingerprints(Path(samples), Path(gtf))
    stamps = quantification_input_stamps(Path(samples), Path(gtf))
    checkpoint_path = stage_checkpoint_path(out, "rna_fpkm_quantification")
    replaces: Optional[Dict[str, str]] = None
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text())
        if previous.get("complete"):
            stale_reason = "--overwrite" if overwrite else quantification_stale_reason(
                previous,
                strandedness=strandedness,
                paired_end=paired_end,
                pseudocount=pseudocount,
                require_both_ends_mapped=require_both_ends_mapped,
                exclude_chimeric_fragments=exclude_chimeric_fragments,
                fingerprint=fingerprint,
                request=request,
                stamps=stamps,
            )
            if stale_reason is None:
                kept_result(previous, stamps)
                info = _already_complete("rna_fpkm_quantification", checkpoint_path, out)
                return info
            if not overwrite:
                print(
                    f"[paralome] rna_fpkm_quantification: {stale_reason}; "
                    "recomputing instead of reusing the existing checkpoint.",
                    file=sys.stderr,
                )
            # Withdrawn when the jobs are submitted -- see _withdraw_replaced.
            replaces = _replaces(checkpoint_path, stale_reason)
    # Preflight runs before any directory, plan, or sbatch script is written,
    # so an invalid configuration submits zero worker jobs.
    # A BAM without an index is indexed here, on the login node --
    # see ensure_bam_indexes.
    preflight_rna_quantification(samples, gtf, strandedness=strandedness, paired_end=paired_end,
                                 index_dir=project_bam_index_dir(Path(out).resolve()))
    sample_rows = _read_fpkm_samples(samples)
    if not fingerprint:
        fingerprint = input_fingerprint(
            stamped=[Path(gtf), *(sample.bam for sample in sample_rows)],
            content=[[sample.sample_id, sample.condition] for sample in sample_rows],
        )
    for d in (out / "inputs", out / "scripts", out / "logs", out / "blocks", out / "checkpoints"):
        d.mkdir(parents=True, exist_ok=True)
    concurrency, task_threads = _resource_split(threads, jobs, len(sample_rows))
    tasks = []
    for i, sample in enumerate(sample_rows, 1):
        task_dir = out / "blocks" / f"sample_{i:06d}_{sample.sample_id}"
        manifest = out / "inputs" / f"sample_{i:06d}.csv"
        with manifest.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["sample_id", "bam", "condition"])
            w.writeheader(); w.writerow({"sample_id": sample.sample_id, "bam": str(sample.bam), "condition": sample.condition})
        tasks.append({"task_id": i, "sample_id": sample.sample_id, "manifest": str(manifest), "task_dir": str(task_dir)})
    plan = {
        # V3 -- the tasks count the library size by fpkm.LIBRARY_SIZE_RULE.
        "schema": "paralome.rna.fpkm_parallel_plan.v3",
        "library_size_rule": LIBRARY_SIZE_RULE,
        "samples": str(Path(samples).resolve()), "gtf": str(Path(gtf).resolve()), "output_dir": str(out),
        "threads": int(threads), "jobs": int(jobs),
        "concurrency": concurrency, "threads_per_task": task_threads, "mem": mem,
        "partition": partition,
        "input_fingerprint": fingerprint, "request_fingerprint": request,
        # Each input on its own -- see quantification_input_stamps.
        "input_stamps": stamps,
        "strandedness": int(strandedness), "paired_end": bool(paired_end),
        "require_both_ends_mapped": bool(require_both_ends_mapped),
        "exclude_chimeric_fragments": bool(exclude_chimeric_fragments),
        "pseudocount": float(pseudocount), "overwrite": bool(overwrite), "tasks": tasks,
    }
    plan_path = out / "inputs" / "rna_fpkm_parallel_plan.json"
    write_json(plan_path, plan, atomic=True)
    overwrite_flag = " --overwrite" if overwrite else ""
    prepare_script = write_sbatch(out / "scripts" / "rna_fpkm_prepare.sbatch", SlurmOptions(
        job_name="rna_fpkm_prepare", cpus_per_task=1, mem=mem, partition=partition,
        output=out / "logs" / "rna_fpkm_prepare_%j.out", error=out / "logs" / "rna_fpkm_prepare_%j.err"),
        f"{_cli()} rna fpkm-prepare --plan {shlex.quote(str(plan_path))}")
    array_script = write_sbatch(out / "scripts" / "rna_fpkm_samples.sbatch", SlurmOptions(
        job_name="rna_fpkm", cpus_per_task=task_threads, mem=mem, partition=partition,
        array=compress_array_ids(range(1, len(tasks)+1), throttle=concurrency),
        output=out / "logs" / "rna_fpkm_%A_%a.out", error=out / "logs" / "rna_fpkm_%A_%a.err"),
        f'{_cli()} rna fpkm-sample --plan {shlex.quote(str(plan_path))} --task-id "$SLURM_ARRAY_TASK_ID"{overwrite_flag}')
    finalize_script = write_sbatch(out / "scripts" / "rna_fpkm_finalize.sbatch", SlurmOptions(
        job_name="rna_fpkm_finalize", cpus_per_task=1, mem=mem, partition=partition,
        output=out / "logs" / "rna_fpkm_finalize_%j.out", error=out / "logs" / "rna_fpkm_finalize_%j.err"),
        f"{_cli()} rna fpkm-finalize --plan {shlex.quote(str(plan_path))}")
    return {"plan": plan_path, "prepare_script": prepare_script, "array_script": array_script,
            "finalize_script": finalize_script, "task_count": len(tasks), "concurrency": concurrency,
            "threads_per_task": task_threads, **({"replaces": replaces} if replaces else {})}


def submit_fpkm_slurm_plan(info: Mapping[str, object], *, submit: bool) -> Dict[str, object]:
    result = dict(info); result.update({"submitted": False, "submitted_job_ids": [], "complete": bool(info.get("reused"))})
    if not submit or info.get("reused"): return result
    p, a, f = _submit_chain(info)
    result.update({"submitted": True, "job_id": ",".join([p,a,f]), "submitted_job_ids": [p,a,f],
                   "prepare_job_id": p, "array_job_id": a, "finalize_job_id": f})
    return result


def run_fpkm_prepare(plan_path: Path) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).read_text())
    payload = {"schema": "paralome.rna.fpkm_prepare.v1", "complete": True, "task_count": len(plan["tasks"])}
    write_json(stage_checkpoint_path(Path(plan["output_dir"]), "rna_fpkm_prepare"), payload, atomic=True)
    return payload


def run_fpkm_sample(plan_path: Path, task_id: int, overwrite: bool=False) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).read_text()); task = plan["tasks"][int(task_id)-1]
    return quantify_fpkm(Path(task["manifest"]), Path(plan["gtf"]), Path(task["task_dir"]),
        threads=int(plan["threads_per_task"]), strandedness=int(plan["strandedness"]),
        paired_end=bool(plan["paired_end"]), require_both_ends_mapped=bool(plan["require_both_ends_mapped"]),
        exclude_chimeric_fragments=bool(plan["exclude_chimeric_fragments"]), pseudocount=float(plan["pseudocount"]),
        overwrite=overwrite or bool(plan.get("overwrite")),
        # The index the login node made at submit time, not a second one per task.
        index_dir=project_bam_index_dir(Path(plan["output_dir"])))


def run_fpkm_finalize(plan_path: Path) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).read_text()); out = Path(plan["output_dir"])
    machine = out / "results" / "machine"; machine.mkdir(parents=True, exist_ok=True)
    metadata = None; counts: Dict[str,List[float]] = {}; fpkm: Dict[str,List[float]] = {}; log2: Dict[str,List[float]] = {}
    qc_rows=[]; library_fragments={}; skipped_genes: Dict[str,str] = {}; partly_kept_genes: Dict[str,str] = {}
    for task in plan["tasks"]:
        cp = stage_checkpoint_path(Path(task["task_dir"]), "rna_fpkm_quantification")
        if not cp.exists(): raise RuntimeError(f"Missing RNA FPKM task checkpoint: {cp}")
        c = json.loads(cp.read_text())
        if not c.get("complete"): raise RuntimeError(f"Incomplete RNA FPKM task: {task['sample_id']}")
        sid=task["sample_id"]
        for key,target in (("gene_fragment_counts",counts),("gene_fpkm",fpkm),("gene_log2_fpkm",log2)):
            m, arr = _read_matrix(Path(c[key]), [sid])
            if metadata is None: metadata=m
            elif m != metadata: raise RuntimeError("RNA quantification task gene metadata mismatch")
            target[sid]=arr[:,0].tolist()
        with Path(c["sample_expression_qc"]).open(newline="") as h: qc_rows.extend(csv.DictReader(h, delimiter="\t"))
        library_fragments.update(c.get("library_fragments",{}))
        # The genes left out of every matrix (on two strands or two
        # chromosomes), as each task found them in the one GTF.
        skipped_genes.update(c.get("skipped_genes") or {})
        partly_kept_genes.update(c.get("partly_kept_genes") or {})
    assert metadata is not None
    counts_path=machine/"gene_fragment_counts.tsv"; fpkm_path=machine/"gene_fpkm.tsv"; log2_path=machine/"gene_log2_fpkm.tsv"
    # A result this step is about to replace does not stay marked
    # complete while its matrices are rewritten (the plan withdraws it too;
    # this covers a plan written by an earlier install).
    final_checkpoint=stage_checkpoint_path(out,"rna_fpkm_quantification")
    replaced: Dict[str,object]={}
    if final_checkpoint.exists():
        replaced=json.loads(final_checkpoint.read_text())
        if replaced.get("complete"): invalidate_checkpoint(final_checkpoint,replaced,"being replaced by a new SLURM run")
    _write_matrix_atomic(counts_path,metadata,counts); _write_matrix_atomic(fpkm_path,metadata,fpkm); _write_matrix_atomic(log2_path,metadata,log2)
    qc_path=machine/"sample_expression_qc.tsv"; qc_temporary=qc_path.with_suffix(".tsv.tmp")
    with qc_temporary.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(qc_rows[0]),delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(qc_rows)
    os.replace(qc_temporary,qc_path)
    # The table of genes not held whole, and the line about them,
    # as quantify_fpkm writes and says them.
    left_out_path=write_genes_left_out(machine/"genes_left_out.tsv",metadata,skipped_genes,partly_kept_genes)
    summary=genes_left_out_summary(skipped_genes,partly_kept_genes,left_out_path)
    if summary: print(summary,file=sys.stderr)
    # What the tasks were planned from, recorded as quantify_fpkm records it, so either backend can tell
    # whether the other's result still answers. An older plan carries none; it is worked out here.
    fingerprint=plan.get("input_fingerprint")
    if not fingerprint:
        try:
            planned=_read_fpkm_samples(Path(plan["samples"]))
            fingerprint=input_fingerprint(stamped=[Path(plan["gtf"]),*(s.bam for s in planned)],content=[[s.sample_id,s.condition] for s in planned])
        except (OSError,ValueError):
            fingerprint=""
    # And what was asked for, with the genes left out, under the keys
    # quantify_fpkm uses. A plan from an earlier install carries no request.
    request=plan.get("request_fingerprint")
    if request is None:
        try:
            request=_fpkm_request_fingerprint(Path(plan["samples"]),Path(plan["gtf"]))
        except (OSError,ValueError,KeyError):
            request=""
    strandedness_key = int(plan["strandedness"])
    # V4 and the library-size rule, as quantify_fpkm records them --
    # see fpkm.LIBRARY_SIZE_RULE. Every task counted under this install's rule.
    checkpoint={"schema":"paralome.rna.fpkm_quantification.v4","stage":"rna_fpkm_quantification","complete":True,
        "library_size_rule":LIBRARY_SIZE_RULE,
        "parallel":True,"counting_backend":"native","sample_count":len(plan["tasks"]),"gene_count":len(metadata),"paired_end":plan["paired_end"],
        "strandedness":plan["strandedness"],"strandedness_label":STRANDEDNESS_LABELS[strandedness_key],"strand_assignment_rule":STRANDEDNESS_ASSIGNMENT_RULES[strandedness_key],
        "require_both_ends_mapped":bool(plan["require_both_ends_mapped"]),"exclude_chimeric_fragments":bool(plan["exclude_chimeric_fragments"]),
        "input_fingerprint":fingerprint,"request_fingerprint":request,
        # Each input on its own, as quantify_fpkm records it. A plan
        # from an earlier install carries none, and none is recorded.
        "input_stamps":plan.get("input_stamps"),
        "skipped_gene_count":len(skipped_genes),"skipped_genes":dict(sorted(skipped_genes.items())),
        "partly_kept_gene_count":len(partly_kept_genes),"partly_kept_genes":dict(sorted(partly_kept_genes.items())),
        "genes_left_out":str(left_out_path),
        "pseudocount":plan["pseudocount"],"gene_fragment_counts":str(counts_path),"gene_fpkm":str(fpkm_path),
        "gene_log2_fpkm":str(log2_path),"sample_expression_qc":str(qc_path),"library_fragments":library_fragments,
        "task_count":len(plan["tasks"]),"concurrency":plan["concurrency"],"threads_per_task":plan["threads_per_task"]}
    write_json(out/"checkpoints"/"rna_fpkm_quantification.json",checkpoint,atomic=True)
    return checkpoint


def _differential_options(plan: Mapping[str, object]) -> Dict[str, object]:
    """The request a plan records, as plan_bulk_differential takes it."""
    return {
        "statistic": str(plan["statistic"]),
        "factors": list(plan.get("factors") or ()),
        "pair_by": plan.get("pair_by"),
        "covariate": plan.get("covariate"),
        "covariate_feature": plan.get("covariate_feature"),
    }


def _differential_inputs(plan: Mapping[str, object]):
    """The sheet, the matrix and the plan of comparisons, as every task of one plan reads them.

    The same ``plan_bulk_differential`` the local run calls, on the same
    rows, so every block fits -- and the finalize names -- exactly the
    comparisons the plan checked. --covariate-feature is found in the WHOLE
    matrix, not a block's slice of it: the feature need not be in the block.
    """
    sample_rows = _read_diff_samples(Path(str(plan["samples"])))
    ids = [str(row["sample_id"]).strip() for row in sample_rows]
    meta, matrix = _read_matrix(Path(str(plan["expression"])), ids)
    options = _differential_options(plan)
    feature_values = (
        covariate_feature_values(meta, matrix, str(options["covariate_feature"]))
        if options["covariate_feature"] and not options["covariate"] else None
    )
    comparisons = plan_bulk_differential(
        sample_rows, covariate_feature_values=feature_values, **options,
    )
    return sample_rows, meta, matrix, comparisons


def prepare_differential_slurm_plan(
    expression: Path,
    samples: Path,
    out: Path,
    *,
    statistic: str,
    factors: Sequence[str],
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    block_size: int = 5000,
    threads: int,
    jobs: int,
    partition: Optional[str],
    mem: str,
    overwrite: bool,
    min_expression: float = 0.0,
    min_expression_difference: float = 0.0,
    pseudocount: float = 1.0,
    refine_responders: int = 0,
    alpha_metric: str = "p",
    alpha_value: float = 0.05,
) -> Dict[str, object]:
    """Plan the SLURM run of 'rna differential': prepare, a block array, finalize.

    The checks differential_expression makes, made before anything
    is written or submitted, and in its order: the request against the sheet,
    then whether a finished result answers, then the matrix.
    """
    check_pseudocount(pseudocount)
    options = dict(
        statistic=statistic, factors=list(factors),
        pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
    )
    check_request_options(refine_responders=refine_responders, **options)
    # The pseudocount the matrix was built with -- see
    # check_pseudocount_matches_matrix.
    check_pseudocount_matches_matrix(expression, pseudocount)
    planned = False
    try:
        sample_rows = _read_diff_samples(samples)
    except (OSError, ValueError):
        sample_rows = None
    if sample_rows is not None:
        plan_bulk_differential(sample_rows, **options)
        planned = True
    out = Path(out).resolve()
    try:
        if sample_rows is None:
            raise ValueError("the sample sheet could not be read")
        fingerprint = input_fingerprint(hashed=[expression], content=sample_rows)
    except (OSError, ValueError):
        fingerprint = ""
    request_fingerprint = (
        differential_request_fingerprint(expression, sample_rows) if sample_rows is not None else ""
    )
    refine = int(refine_responders or 0)
    final_checkpoint = stage_checkpoint_path(out, "differential_expression")
    replaces: Optional[Dict[str, str]] = None
    if final_checkpoint.exists():
        previous = json.loads(final_checkpoint.read_text())
        if previous.get("complete"):
            stale_reason = "--overwrite" if overwrite else differential_stale_reason(
                previous, min_expression=min_expression,
                min_expression_difference=min_expression_difference, pseudocount=pseudocount,
                alpha_metric=alpha_metric, alpha_value=alpha_value, fingerprint=fingerprint,
                request_fingerprint=request_fingerprint, output_dir=out, **options,
            )
            # The local path refines a finished result in place. A job would
            # have to be submitted for that here, so the stage is run again.
            refined = previous.get("refine_responders") or 0
            if stale_reason is None and refine and refined != refine:
                if previous.get("refine_responders_attempted") == refine:
                    # Tried with this N on this unchanged result and
                    # it did not complete -- see record_refined_outputs.
                    reason = previous.get("refine_responders_failed") or (
                        f"--refine-responders {refine} did not complete")
                    print(f"[paralome] differential_expression: --refine-responders {refine} was "
                          f"already tried on this result and did not complete: {reason}. Not "
                          "resubmitting; pass --overwrite to try again.", file=sys.stderr)
                else:
                    stale_reason = f"--refine-responders {refine} has not been run on this result"
            if stale_reason is None:
                say_kept_without_matrix(expression)
                return _already_complete("differential_expression", final_checkpoint, out)
            if not overwrite:
                print(f"[paralome] differential_expression: {stale_reason}; recomputing instead "
                      "of reusing the existing checkpoint.", file=sys.stderr)
            # Withdrawn when the jobs are submitted -- see _withdraw_replaced.
            replaces = _replaces(final_checkpoint, stale_reason)
    if sample_rows is None:
        sample_rows = _read_diff_samples(samples)
    if not planned:
        plan_bulk_differential(sample_rows, **options)
    meta, matrix = _read_matrix(expression, [r["sample_id"] for r in sample_rows])
    if covariate_feature and not covariate:
        # Refused here, on the login node, not in every array task.
        plan_bulk_differential(
            sample_rows, covariate_feature_values=covariate_feature_values(
                meta, matrix, covariate_feature), **options,
        )
    if not fingerprint:
        fingerprint = input_fingerprint(hashed=[expression], content=sample_rows)
    for d in (out / "inputs", out / "scripts", out / "logs", out / "blocks", out / "checkpoints"):
        d.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, start in enumerate(range(0, len(meta), block_size), 1):
        tasks.append({"task_id": i, "start": start, "stop": min(start + block_size, len(meta)),
                      "task_dir": str(out / "blocks" / f"block_{i:06d}")})
    concurrency, task_threads = _resource_split(threads, jobs, len(tasks))
    plan = {
        "schema": "paralome.rna.differential_parallel_plan.v4",
        "expression": str(Path(expression).resolve()), "samples": str(Path(samples).resolve()),
        "output_dir": str(out),
        # The whole request every block fits and the finalize names:
        # --statistic as given, the arms, the factors, --pair-by, --covariate
        # and --covariate-feature.
        **options,
        "block_size": block_size,
        "min_expression": min_expression, "min_expression_difference": min_expression_difference,
        "pseudocount": pseudocount, "refine_responders": refine, "input_fingerprint": fingerprint,
        # From --expression as it was given -- see differential_request_fingerprint.
        "request_fingerprint": request_fingerprint or differential_request_fingerprint(expression, sample_rows),
        "alpha_metric": alpha_metric, "alpha_value": alpha_value,
        "threads": threads, "jobs": jobs, "concurrency": concurrency, "threads_per_task": task_threads,
        "mem": mem, "overwrite": overwrite, "tasks": tasks,
    }
    plan_path = out / "inputs" / "rna_differential_parallel_plan.json"
    write_json(plan_path, plan, atomic=True)
    of = " --overwrite" if overwrite else ""
    prep = write_sbatch(out / "scripts" / "rna_differential_prepare.sbatch", SlurmOptions(
        job_name="rna_diff_prepare", cpus_per_task=1, mem=mem, partition=partition,
        output=out / "logs" / "rna_diff_prepare_%j.out", error=out / "logs" / "rna_diff_prepare_%j.err"),
        f"{_cli()} rna differential-prepare --plan {shlex.quote(str(plan_path))}")
    arr = write_sbatch(out / "scripts" / "rna_differential_blocks.sbatch", SlurmOptions(
        job_name="rna_diff", cpus_per_task=task_threads, mem=mem, partition=partition,
        array=compress_array_ids(range(1, len(tasks) + 1), throttle=concurrency),
        output=out / "logs" / "rna_diff_%A_%a.out", error=out / "logs" / "rna_diff_%A_%a.err"),
        f'{_cli()} rna differential-block --plan {shlex.quote(str(plan_path))} --task-id "$SLURM_ARRAY_TASK_ID"{of}')
    fin = write_sbatch(out / "scripts" / "rna_differential_finalize.sbatch", SlurmOptions(
        job_name="rna_diff_finalize", cpus_per_task=1, mem=mem, partition=partition,
        output=out / "logs" / "rna_diff_finalize_%j.out", error=out / "logs" / "rna_diff_finalize_%j.err"),
        f"{_cli()} rna differential-finalize --plan {shlex.quote(str(plan_path))}")
    return {"plan": plan_path, "prepare_script": prep, "array_script": arr, "finalize_script": fin,
            "task_count": len(tasks), "concurrency": concurrency, "threads_per_task": task_threads,
            **({"replaces": replaces} if replaces else {})}


def submit_differential_slurm_plan(info: Mapping[str, object], *, submit: bool) -> Dict[str, object]:
    result = dict(info)
    result.update({"submitted": False, "submitted_job_ids": [], "complete": bool(info.get("reused"))})
    if not submit or info.get("reused"):
        return result
    p, a, f = _submit_chain(info)
    result.update({"submitted": True, "job_id": ",".join([p, a, f]), "submitted_job_ids": [p, a, f],
                   "prepare_job_id": p, "array_job_id": a, "finalize_job_id": f})
    return result


def run_differential_prepare(plan_path: Path) -> Dict[str, object]:
    p = json.loads(Path(plan_path).read_text())
    payload = {"schema": "paralome.rna.differential_prepare.v1", "complete": True, "task_count": len(p["tasks"])}
    write_json(Path(p["output_dir"]) / "checkpoints" / "differential_expression_prepare.json", payload, atomic=True)
    return payload


def _plan_digest(plan_path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()


def _block_checkpoint_path(task: Mapping[str, object]) -> Path:
    return Path(str(task["task_dir"])) / "checkpoints" / "differential_block.json"


def run_differential_block(plan_path: Path, task_id: int, overwrite: bool = False) -> Dict[str, object]:
    """One array task: the engine's fit of this block's genes, and nothing else.

    It fits ONLY the engine's columns for its genes (uncorrected; BH needs
    every gene, so it is the finalize's) and keeps them at full precision in
    ``block_statistics.npz``. The finalize joins the blocks in order and
    writes the result through the same writer the local run uses.
    """
    import numpy as np

    plan_path = Path(plan_path).resolve()
    p = json.loads(plan_path.read_text())
    task = p["tasks"][int(task_id) - 1]
    task_dir = Path(str(task["task_dir"]))
    checkpoint_path = _block_checkpoint_path(task)
    digest = _plan_digest(plan_path)
    if checkpoint_path.exists() and not (overwrite or p.get("overwrite")):
        try:
            previous = json.loads(checkpoint_path.read_text())
        except ValueError:
            previous = {}
        if (previous.get("complete") and previous.get("plan_sha256") == digest
                and previous.get("paralome_version") == __version__
                and Path(str(previous.get("output"))).is_file()):
            return previous
    sample_rows, meta, matrix, comparisons = _differential_inputs(p)
    start, stop = int(task["start"]), int(task["stop"])
    raw = _raw_fpkm(matrix, float(p.get("pseudocount", 1.0)))
    results = fit_blocks(
        matrix, raw, comparisons, start=start, stop=stop, block_size=int(p["block_size"]),
        alpha=power_alpha(str(p.get("alpha_metric", "q")), float(p.get("alpha_value", 0.05))),
    )
    arrays: Dict[str, object] = {}
    layout = []
    for position, result in enumerate(results):
        names = list(result.columns)
        layout.append({"folder": result.folder, "comparison": result.name, "columns": names})
        for name in names:
            arrays[f"r{position}__{name}"] = np.asarray(result.columns[name])
    (task_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    output = task_dir / "block_statistics.npz"
    temporary = task_dir / "block_statistics.partial.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, output)
    payload = {
        "schema": "paralome.rna.differential_block.v2", "complete": True,
        "paralome_version": __version__, "plan_sha256": digest,
        "task_id": int(task_id), "start": start, "stop": stop,
        "genes": [str(row.get("gene_id") or "") for row in meta[start:stop]],
        "results": layout, "output": str(output),
    }
    write_json(checkpoint_path, payload, atomic=True)
    return payload


def _load_block(task: Mapping[str, object], comparisons, gene_ids: Sequence[str]):
    """One finished block's results, as the engine's ComparisonResult objects."""
    import numpy as np

    from paralome.engine import ComparisonResult

    checkpoint_path = _block_checkpoint_path(task)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing differential block checkpoint: {checkpoint_path}")
    block = json.loads(checkpoint_path.read_text())
    if not block.get("complete"):
        raise RuntimeError(f"Differential block {task['task_id']} did not finish: {checkpoint_path}")
    start, stop = int(task["start"]), int(task["stop"])
    if [int(block.get("start", -1)), int(block.get("stop", -1))] != [start, stop] or (
            list(block.get("genes") or ()) != list(gene_ids[start:stop])):
        raise RuntimeError(
            f"Differential block {task['task_id']} ({checkpoint_path}) was computed for other "
            "genes than the plan gives it; run 'rna differential' again to plan the stage anew")
    expected = [(statistic.folder, comparison) for statistic in comparisons.statistics
                for comparison in statistic.comparisons]
    layout = block.get("results") or []
    if [(entry["folder"], entry["comparison"]) for entry in layout] != [
            (folder, comparison.name) for folder, comparison in expected]:
        raise RuntimeError(
            f"Differential block {task['task_id']} holds other comparisons than the plan makes; "
            "run 'rna differential' again to plan the stage anew")
    statistic_of = {statistic.folder: statistic.statistic for statistic in comparisons.statistics}
    with np.load(str(block["output"]), allow_pickle=False) as arrays:
        results = []
        for position, ((folder, comparison), entry) in enumerate(zip(expected, layout)):
            results.append(ComparisonResult(
                statistic=statistic_of[folder], folder=folder, comparison=comparison,
                columns={name: np.array(arrays[f"r{position}__{name}"]) for name in entry["columns"]},
            ))
    return results, str(block.get("paralome_version", "unknown"))


def run_differential_finalize(plan_path: Path) -> Dict[str, object]:
    """Join the blocks in order, and write the result as the local run writes it."""
    from paralome.engine import comparison_facts, concatenate

    p = json.loads(Path(plan_path).read_text())
    out = Path(p["output_dir"])
    sample_rows, meta, matrix, comparisons = _differential_inputs(p)
    # The blocks were computed from the inputs as they were when the
    # plan was made. If the sheet or the matrix has changed since, merging
    # them would describe inputs they were not computed from.
    fingerprint = p.get("input_fingerprint") or ""
    current = input_fingerprint(hashed=[Path(p["expression"])], content=sample_rows)
    if fingerprint and current != fingerprint:
        raise RuntimeError(
            f"The sample sheet or the expression matrix changed after {plan_path} was "
            "written, so the finished blocks do not describe them. Run 'rna differential' "
            "again to plan and submit the stage from the inputs as they are now.")
    fingerprint = fingerprint or current
    gene_ids = [str(row.get("gene_id") or "") for row in meta]
    blocks, block_versions = [], set()
    for task in p["tasks"]:
        results, version = _load_block(task, comparisons, gene_ids)
        blocks.append(results)
        block_versions.add(version)
    raw_results = concatenate(blocks)
    if not raw_results or len(raw_results[0]) != len(meta):
        raise RuntimeError("The differential blocks do not cover every gene of the matrix")
    by_name = {(statistic.folder, comparison.name): statistic for statistic in comparisons.statistics
               for comparison in statistic.comparisons}
    for result in raw_results:
        result.facts = comparison_facts(by_name[(result.folder, result.name)], result.comparison, "samples")
    # SAY IT OUT LOUD: version skew inside one job is an environment fault the
    # pipeline cannot fix, but it can refuse to let it be silent.
    if block_versions - {__version__}:
        print(f"  WARNING: version skew -- finalize is {__version__}, "
              f"blocks were {', '.join(sorted(block_versions))}. "
              "Outputs may mix formats; re-run this stage with one install.",
              flush=True)
    # The result this one replaces was withdrawn when the plan was
    # submitted; its files go now, just before the new ones are written.
    final_checkpoint = stage_checkpoint_path(out, "differential_expression")
    superseded = None
    if final_checkpoint.exists():
        superseded = json.loads(final_checkpoint.read_text())
        if superseded.get("complete"):
            invalidate_checkpoint(final_checkpoint, superseded, "being replaced by a new SLURM run")
    return write_differential_result(
        raw_results, comparisons,
        out=out, expression_matrix=Path(p["expression"]), samples_path=Path(p["samples"]),
        sample_rows=sample_rows, meta=meta, matrix=matrix,
        min_expression=float(p.get("min_expression", 0.0)),
        min_expression_difference=float(p.get("min_expression_difference", 0.0)),
        pseudocount=float(p.get("pseudocount", 1.0)),
        alpha_metric=str(p.get("alpha_metric", "q")), alpha_value=float(p.get("alpha_value", 0.05)),
        file_prefix=None, fingerprint=fingerprint,
        request_fingerprint=p.get("request_fingerprint") or differential_request_fingerprint(
            p["expression"], sample_rows),
        superseded=superseded,
        refine_responders=int(p.get("refine_responders", 0) or 0) or None, run_qc=True,
        block_count=len(p["tasks"]),
        extra={
            "parallel": True,
            "block_paralome_versions": sorted(block_versions),
            "concurrency": p.get("concurrency"),
            "threads_per_task": p.get("threads_per_task"),
        },
        **_differential_options(p),
    )
