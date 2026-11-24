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
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from paralome.atac.naming import (
    assert_outputs_are_prefixed,
    derive_file_prefix,
    prefixed_name,
    require_file_prefix,
    snapshot_results,
    stage_banner,
)
from paralome.checkpoints import (
    file_info,
    recorded_output_sizes,
    stage_checkpoint_path,
)
from paralome.errors import ParalomeError
from paralome.pipeline.region_shards import check_region_convention, merge_tsv_fragments, read_regions, split_vcf_by_regions
from paralome.segregation.table import BUCKETS, HEADER, SEGREGATION_METHOD, write_segregation_outputs


SCHEMA_PLAN = "paralome.segregation.plan.v2"
SCHEMA_CHECKPOINT = "paralome.segregation.checkpoint.v2"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def ensure_dirs(workdir: Path) -> None:
    for name in ("inputs", "results", "checkpoints", "logs", "scripts", "temp"):
        (workdir / name).mkdir(parents=True, exist_ok=True)


def default_prefix(workdir: Path, prefix: Optional[str] = None) -> str:
    """The project/family ID this run's outputs are named for.

    Not ``workdir.name`` (literally "segregation"): a result file must be
    identifiable with its directory stripped off, once anything collects,
    emails or uploads it. ``derive_file_prefix`` has "segregation", "filter"
    and "candidate_report" in its generic-stage-directory list, so it climbs
    past them to the project ID. Same helper, same prefix position, same
    spelling as the ATAC/RNA outputs.
    """
    if prefix and prefix.strip():
        return prefix.strip()
    return derive_file_prefix(workdir)


@dataclass(frozen=True)
class SegregationStageResult:
    workdir: Path
    input_vcf: Path
    samples: Path
    output_tsv: Path
    plan_json: Path
    checkpoint_json: Path
    summary_json: Path
    provenance_json: Path
    bucket_outputs: dict[str, Path]
    executed: bool
    complete: bool
    rows: int
    bucket_counts: dict[str, int]
    message: str
    threads: int = 1
    shards: int = 1
    regions: Optional[Path] = None
    execution_mode: str = "serial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workdir": str(self.workdir),
            "input_vcf": str(self.input_vcf),
            "samples": str(self.samples),
            "output_tsv": str(self.output_tsv),
            "plan_json": str(self.plan_json),
            "checkpoint_json": str(self.checkpoint_json),
            "summary_json": str(self.summary_json),
            "provenance_json": str(self.provenance_json),
            "bucket_outputs": {k: str(v) for k, v in self.bucket_outputs.items()},
            "executed": self.executed,
            "complete": self.complete,
            "rows": self.rows,
            "bucket_counts": self.bucket_counts,
            "message": self.message,
            "threads": self.threads,
            "shards": self.shards,
            "regions": str(self.regions) if self.regions else None,
            "execution_mode": self.execution_mode,
        }


def count_tsv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open() as fh:
        return max(0, sum(1 for _ in fh) - 1)


def bucket_output_paths(workdir: Path, prefix: Optional[str] = None) -> dict[str, Path]:
    """The four bucket files, carrying the project ID."""
    results = workdir / "results"
    run_prefix = default_prefix(workdir, prefix)
    return {
        bucket: results / prefixed_name(f"{bucket}.tsv", run_prefix)
        for bucket in (
            "segregation_pass",
            "segregation_partial",
            "segregation_fail",
            "segregation_uninformative",
        )
    }


def build_segregation_plan(
    *,
    vcf: Path,
    samples: Path,
    workdir: Path,
    prefix: Optional[str] = None,
    overwrite: bool = False,
    threads: int = 1,
    shards: int = 1,
    regions: Optional[Path] = None,
) -> dict[str, Any]:
    run_prefix = default_prefix(workdir, prefix)
    results = workdir / "results"
    all_tsv = results / prefixed_name("all_variants.segregation.tsv", run_prefix)
    summary = results / prefixed_name("segregation_summary.json", run_prefix)
    # Provenance stays unprefixed: it is a provenance record that never leaves the workdir, and renaming
    # it would churn the resume path for no benefit.
    provenance = results / "provenance.json"
    buckets = bucket_output_paths(workdir, run_prefix)

    return {
        "schema": SCHEMA_PLAN,
        "stage": "segregation",
        "stage_type": "post_annotation_family_segregation",
        "created": now_iso(),
        "workdir": str(workdir),
        "input_vcf": str(vcf),
        "samples": str(samples),
        "prefix": run_prefix,
        "overwrite": overwrite,
        "threads": int(max(1, threads)),
        "shards": int(max(1, shards)),
        "regions": str(regions) if regions else None,
        "execution_mode": "parallel_region_sharded" if (regions or int(max(1, threads)) > 1 or int(max(1, shards)) > 1) else "serial",
        "lossless": True,
        "outputs": {
            "all_variants": str(all_tsv),
            "segregation_tsv": str(all_tsv),
            "summary": str(summary),
            "provenance": str(provenance),
            **{bucket: str(path) for bucket, path in buckets.items()},
        },
    }


def write_provenance(*, workdir: Path, plan: dict[str, Any], summary: dict[str, Any]) -> Path:
    path = Path(plan["outputs"]["provenance"])
    provenance = {
        "schema": "paralome.segregation.provenance.v1",
        "stage": "segregation",
        "stage_type": "post_annotation_family_segregation",
        "created": now_iso(),
        "algorithm_scope": "genotype_and_pedigree_only",
        "excluded_inputs": [
            "population_frequency",
            "functional_consequence",
            "clinical_significance",
            "gene_panel_membership",
        ],
        "input_vcf": plan.get("input_vcf"),
        "samples": plan.get("samples"),
        "outputs": plan.get("outputs", {}),
        "validation": {
            "input_rows": summary.get("rows", 0),
            "bucket_sum": summary.get("bucket_sum", 0),
            "lossless": bool(summary.get("lossless")),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return path


def write_checkpoint(
    *,
    workdir: Path,
    plan: dict[str, Any],
    complete: bool,
    rows: int = 0,
    bucket_counts: dict[str, int] | None = None,
    dry_run: bool = False,
    message: str = "",
    inputs_identity: Optional[dict[str, Any]] = None,
) -> Path:
    checkpoint = stage_checkpoint_path(workdir, "segregation")
    plan_outputs = plan.get("outputs", {}) if isinstance(plan.get("outputs"), dict) else {}
    outputs = plan.get("outputs", {}) if isinstance(plan.get("outputs"), dict) else {}
    bucket_counts = bucket_counts or {bucket: 0 for bucket in BUCKETS}
    bucket_sum = sum(int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS)
    state = {
        "schema": SCHEMA_CHECKPOINT,
        "stage": "segregation",
        "stage_type": "post_annotation_family_segregation",
        "complete": complete,
        "dry_run": dry_run,
        "started": plan.get("created"),
        "updated": now_iso(),
        "finished": now_iso() if complete else None,
        "input_vcf": plan.get("input_vcf"),
        "samples": plan.get("samples"),
        "outputs": outputs,
        "threads": int(plan.get("threads", 1)),
        "shards": int(plan.get("shards", 1)),
        "regions": plan.get("regions"),
        "execution_mode": plan.get("execution_mode", "serial"),
        "rows": rows,
        "bucket_counts": bucket_counts,
        "bucket_sum": bucket_sum,
        "lossless": bool(rows == bucket_sum) if rows or bucket_sum else False,
        "message": message,
        # Which VCF and sheet the tables were scored from (see _inputs_identity).
        "inputs_identity": inputs_identity,
        # The size of each table this checkpoint vouches for, so
        # a reader can tell when the files beside it are no longer those
        # tables (see checkpoints.recorded_output_sizes).
        "output_bytes": (
            recorded_output_sizes({key: plan_outputs.get(key) for key in ("all_variants", *BUCKETS)})
            if complete else {}
        ),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return checkpoint


def _inputs_identity(vcf: Path, samples: Path) -> dict[str, Any]:
    """The input VCF (path, size, mtime) and the sheet's SHA-256.

    Reuse compared neither, so `segregate run` again with a corrected sheet or
    another VCF in the same --out returned the earlier tables. A copied VCF
    costs one recompute, never a stale reuse.
    """
    import hashlib

    stat = vcf.stat()
    return {
        "input_vcf": str(vcf.resolve()),
        "input_vcf_size": int(stat.st_size),
        "input_vcf_mtime_ns": int(stat.st_mtime_ns),
        "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
    }


def _complete_from_outputs(outputs: dict[str, Any]) -> bool:
    required = ["all_variants", "summary", "provenance", *BUCKETS]
    return all(file_info(outputs.get(key)).get("exists") for key in required)


def _tables_match_summary(outputs: dict[str, Any], summary: dict[str, Any]) -> tuple[bool, str]:
    """Whether the tables on disk hold the rows the summary says they hold.

    A lossless summary and the presence of the files are not proof: a run
    killed part way through an --overwrite could leave an earlier summary
    beside truncated tables. The rows are counted here -- once, on the reuse
    path only -- and any disagreement means the stage runs again.
    """
    rows = summary.get("rows")
    if rows is None:
        return False, "the summary records no row count"
    on_disk = count_tsv_rows(Path(outputs["all_variants"]))
    if on_disk != int(rows):
        return False, f"all_variants holds {on_disk} rows, the summary says {int(rows)}"
    recorded = summary.get("bucket_counts") or {}
    for bucket in BUCKETS:
        if bucket not in recorded:
            return False, f"the summary records no count for {bucket}"
        counted = count_tsv_rows(Path(outputs[bucket]))
        if counted != int(recorded[bucket]):
            return False, f"{bucket} holds {counted} rows, the summary says {int(recorded[bucket])}"
    return True, ""


def _invalidate_previous_run(workdir: Path, plan: dict[str, Any], summary_json: Path, provenance_json: Path) -> None:
    """Withdraw the earlier run's claim of completion before its tables are rewritten.

    The tables are rewritten in place (opened with "w"), and the summary and
    checkpoint are written only once the last row is out. So the checkpoint
    says incomplete and the summary and provenance are gone for as long as the
    tables are being written: a run that stops early (a SLURM time limit,
    scancel, Ctrl-C) leaves a stage that reads incomplete and is redone.
    """
    write_checkpoint(
        workdir=workdir,
        plan=plan,
        complete=False,
        rows=0,
        dry_run=False,
        message="Segregation running; the tables are being rewritten.",
    )
    for stale in (summary_json, provenance_json):
        stale.unlink(missing_ok=True)



def _segregation_shard_worker(job: dict[str, Any]) -> dict[str, Any]:
    shard_vcf = Path(job["shard_vcf"])
    samples = Path(job["samples"])
    fragment_dir = Path(job["fragment_dir"])
    source_vcf = Path(job["source_vcf"])
    summary = write_segregation_outputs(
        vcf=shard_vcf,
        samples=samples,
        results_dir=fragment_dir,
        all_tsv=fragment_dir / "all_variants.segregation.tsv",
        summary_json=fragment_dir / "segregation_summary.json",
        source_vcf=source_vcf,
    )
    return {
        "shard_id": int(job["shard_id"]),
        "shard_vcf": str(shard_vcf),
        "fragment_dir": str(fragment_dir),
        "rows": int(summary.get("rows", 0)),
        "bucket_counts": {bucket: int((summary.get("bucket_counts") or {}).get(bucket, 0)) for bucket in BUCKETS},
        "class_counts": {key: int(value) for key, value in (summary.get("class_counts") or {}).items()},
        "lossless": bool(summary.get("lossless")),
        # The same for every shard (one sheet); carried to the merged summary.
        "scored": {key: summary.get(key) for key in ("scoring", "samples_scored", "samples_left_out")},
    }


def _segregation_shard_pool_worker(job: dict[str, Any]) -> dict[str, Any]:
    """A shard worker for the process pool.

    A ParalomeError cannot be unpickled in the parent, so a numbered error
    raised in a shard (a sheet sample missing from the VCF, say) would break
    the pool and reach the user as BrokenProcessPool. It is carried back as
    plain fields and raised again in the parent.
    """
    try:
        return _segregation_shard_worker(job)
    except ParalomeError as exc:
        return {"paralome_error": {"code": int(exc.code), "detail": str(exc.detail), "fix": exc.fix}}


def _run_segregation_sharded(
    *,
    vcf: Path,
    samples: Path,
    workdir: Path,
    output_tsv: Path,
    summary_json: Path,
    bucket_outputs: dict[str, Path],
    regions: Optional[Path],
    threads: int,
    shards: int,
    region_convention: Optional[str] = None,
) -> tuple[dict[str, Any], list[int], list[dict[str, Any]]]:
    threads = max(int(threads or 1), 1)
    shards = max(int(shards or 1), 1)
    region_list = read_regions(regions, region_convention)
    shard_dir = workdir / "inputs" / "segregation_shards"
    fragment_root = workdir / "results" / "shard_fragments"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    if fragment_root.exists():
        shutil.rmtree(fragment_root)
    shard_vcfs, shard_counts = split_vcf_by_regions(input_vcf=vcf, shard_dir=shard_dir, shards=shards, regions=region_list)
    jobs = [
        {
            "shard_id": idx,
            "shard_vcf": str(path),
            "samples": str(samples),
            "fragment_dir": str(fragment_root / f"shard_{idx:04d}"),
            "source_vcf": str(vcf),
        }
        for idx, path in enumerate(shard_vcfs)
    ]
    shard_results: list[dict[str, Any]] = []
    workers = min(threads, len(jobs)) if jobs else 1
    if workers <= 1:
        for job in jobs:
            shard_results.append(_segregation_shard_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_segregation_shard_pool_worker, job) for job in jobs]
            for fut in as_completed(futures):
                result = fut.result()
                error = result.get("paralome_error")
                if error:
                    raise ParalomeError(int(error["code"]), error["detail"], fix=error["fix"])
                shard_results.append(result)
    shard_results.sort(key=lambda x: int(x["shard_id"]))
    fragment_dirs = [Path(result["fragment_dir"]) for result in shard_results]
    # Shard fragments keep their bare names -- they live under
    # results/shard_fragments/<n>/ and are deleted after a lossless merge (with
    # the shard VCFs), so they are never collected and never seen.
    # PASSED IN, not re-derived. The first cut of the rename called
    # bucket_output_paths(workdir, prefix) here, and `prefix` is not a parameter
    # of this function -- a NameError on the sharded path, which is the path
    # exome runs usually take. It also re-derived paths the plan already
    # holds, which is the two-implementations shape all over again: the serial
    # branch read them from the plan and this one rebuilt them.
    bucket_paths = {bucket: Path(path) for bucket, path in bucket_outputs.items()}
    merge_tsv_fragments(
        fragment_paths=[d / "all_variants.segregation.tsv" for d in fragment_dirs],
        output_path=output_tsv,
        fieldnames=HEADER,
    )
    for bucket, path in bucket_paths.items():
        merge_tsv_fragments(fragment_paths=[d / f"{bucket}.tsv" for d in fragment_dirs], output_path=path, fieldnames=HEADER)
    bucket_counts: dict[str, int] = {bucket: 0 for bucket in BUCKETS}
    class_counts: Counter[str] = Counter()
    for result in shard_results:
        for bucket, value in (result.get("bucket_counts") or {}).items():
            if bucket in bucket_counts:
                bucket_counts[bucket] += int(value)
        class_counts.update({key: int(value) for key, value in (result.get("class_counts") or {}).items()})
    total_rows = sum(int(x.get("rows", 0)) for x in shard_results)
    summary = {
        "schema": "paralome.segregation.summary.v2",
        "stage": "segregation",
        "input_vcf": str(vcf),
        "samples": str(samples),
        "outputs": {"all_variants": str(output_tsv), **{bucket: str(path) for bucket, path in bucket_paths.items()}},
        "rows": int(total_rows),
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS},
        "class_counts": {key: int(value) for key, value in sorted(class_counts.items())},
        "bucket_sum": int(sum(bucket_counts.values())),
        "lossless": bool(total_rows == sum(bucket_counts.values())),
        "threads": int(threads),
        "shards": int(shards),
        "regions": str(regions) if regions else None,
        "usable_regions": len(region_list),
        "shard_row_counts": shard_counts,
        "shard_results": shard_results,
        "execution_mode": "parallel_region_sharded",
        # See segregation.table._write_summary.
        "method": SEGREGATION_METHOD,
        **((shard_results[0].get("scored") or {}) if shard_results else {}),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["lossless"]:
        raise RuntimeError(f"Segregation row-preservation validation failed: rows={total_rows} bucket_sum={summary['bucket_sum']}")
    # The fragments and shard VCFs (uncompressed copies of the input) are removed once the merge is
    # lossless.
    shutil.rmtree(fragment_root, ignore_errors=True)
    shutil.rmtree(shard_dir, ignore_errors=True)
    return summary, shard_counts, shard_results


def run_segregation_stage(
    *,
    vcf: Union[str, Path],
    samples: Union[str, Path],
    workdir: Union[str, Path],
    prefix: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    threads: int = 1,
    shards: int = 1,
    regions: Union[str, Path, None] = None,
    region_convention: Optional[str] = None,
) -> SegregationStageResult:
    vcf = Path(vcf)
    samples = Path(samples)
    workdir = Path(workdir)
    threads = max(int(threads or 1), 1)
    shards = max(int(shards or 1), 1)
    regions_path = Path(regions) if regions else None
    region_convention = check_region_convention(region_convention)

    if not vcf.exists():
        raise FileNotFoundError(f"Input VCF not found: {vcf}")
    if not samples.exists():
        raise FileNotFoundError(f"Sample manifest not found: {samples}")

    ensure_dirs(workdir)

    # Resolve and VALIDATE the project ID before anything is written, and say which version resolved what:
    # a prefix that failed to resolve must not silently produce bare filenames.
    run_prefix = require_file_prefix(default_prefix(workdir, prefix), stage="segregation")
    print(stage_banner("segregation", run_prefix, workdir), flush=True)
    # What was already here, so the post-write check can tell this run's
    # mistakes from a previous convention's leftovers.
    _preexisting = snapshot_results(workdir / "results")

    plan = build_segregation_plan(
        vcf=vcf,
        samples=samples,
        workdir=workdir,
        prefix=run_prefix,
        overwrite=overwrite,
        threads=threads,
        shards=shards,
        regions=regions_path,
    )

    plan_json = workdir / "inputs" / "segregation_plan.json"
    outputs = plan["outputs"]
    output_tsv = Path(outputs["all_variants"])
    summary_json = Path(outputs["summary"])
    provenance_json = Path(outputs["provenance"])
    checkpoint_json = stage_checkpoint_path(workdir, "segregation")
    bucket_outputs = {bucket: Path(outputs[bucket]) for bucket in BUCKETS}

    shutil.copy2(samples, workdir / "inputs" / "samples.csv")
    (workdir / "inputs" / "input_vcf.txt").write_text(str(vcf) + "\n")
    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    if dry_run:
        write_checkpoint(
            workdir=workdir,
            plan=plan,
            complete=False,
            rows=0,
            dry_run=True,
            message="Dry run complete; segregation was not executed.",
        )
        return SegregationStageResult(
            workdir=workdir,
            input_vcf=vcf,
            samples=samples,
            output_tsv=output_tsv,
            plan_json=plan_json,
            checkpoint_json=checkpoint_json,
            summary_json=summary_json,
            provenance_json=provenance_json,
            bucket_outputs=bucket_outputs,
            executed=False,
            complete=False,
            rows=0,
            bucket_counts={bucket: 0 for bucket in BUCKETS},
            message="Dry run complete; segregation was not executed.",
            threads=threads,
            shards=shards,
            regions=regions_path,
            execution_mode=plan.get("execution_mode", "serial"),
        )

    existing_summary = read_json(summary_json)
    inputs_identity = _inputs_identity(vcf, samples)
    previous_identity = read_json(checkpoint_json).get("inputs_identity")
    # Only a table this method wrote is reused. Before, any lossless
    # table was -- one written before the relationship, family and per-genotype
    # quality fixes would have been kept, and its rows read as current.
    stale = bool(existing_summary) and existing_summary.get("method") != SEGREGATION_METHOD
    if stale and not overwrite:
        print(f"[paralome] segregation: the table in {workdir / 'results'} was written by "
              f"{existing_summary.get('method') or 'a method before ' + SEGREGATION_METHOD}; "
              "scoring it again.", flush=True)
    # Nor a table written (no depth/GQ/alt-read columns, recessive score 0).
    from paralome.segregation.table import missing_table_columns

    reusable = bool(existing_summary.get("lossless") and not stale and _complete_from_outputs(outputs)
                    and not overwrite and not missing_table_columns(Path(output_tsv)))
    if reusable and previous_identity != inputs_identity:
        reusable = False
        print("Existing segregation outputs are not reused: they were not scored from this VCF "
              "and sample sheet (or predate the record of them); segregation runs again.", flush=True)
    if reusable:
        reusable, why_not = _tables_match_summary(outputs, existing_summary)
        if not reusable:
            # Said, not skipped silently. See _tables_match_summary.
            print(
                f"Existing segregation outputs are not reused: {why_not}. "
                "An earlier run probably stopped part way; segregation runs again.",
                flush=True,
            )
    if reusable:
        # The reuse branch reports complete without writing anything,
        # so it has to meet the same bar. If a previous run left wrongly-named
        # files here, reusing them silently is exactly how a stale name
        # survived five releases.
        assert_outputs_are_prefixed(workdir / "results", run_prefix, stage="segregation",
                                preexisting=_preexisting)
        rows = int(existing_summary.get("rows", count_tsv_rows(output_tsv)))
        bucket_counts = {bucket: int((existing_summary.get("bucket_counts") or {}).get(bucket, 0)) for bucket in BUCKETS}
        write_checkpoint(
            workdir=workdir,
            plan=plan,
            complete=True,
            rows=rows,
            bucket_counts=bucket_counts,
            dry_run=False,
            message="Existing lossless segregation outputs reused.",
            inputs_identity=inputs_identity,
        )
        return SegregationStageResult(
            workdir=workdir,
            input_vcf=vcf,
            samples=samples,
            output_tsv=output_tsv,
            plan_json=plan_json,
            checkpoint_json=checkpoint_json,
            summary_json=summary_json,
            provenance_json=provenance_json,
            bucket_outputs=bucket_outputs,
            executed=False,
            complete=True,
            rows=rows,
            bucket_counts=bucket_counts,
            message="Existing lossless segregation outputs reused.",
            threads=threads,
            shards=shards,
            regions=regions_path,
            execution_mode=plan.get("execution_mode", "serial"),
        )

    _invalidate_previous_run(workdir, plan, summary_json, provenance_json)

    if threads > 1 or shards > 1 or regions_path is not None:
        summary, _shard_counts, _shard_results = _run_segregation_sharded(
            vcf=vcf,
            samples=samples,
            workdir=workdir,
            output_tsv=output_tsv,
            summary_json=summary_json,
            bucket_outputs=bucket_outputs,
            regions=regions_path,
            threads=threads,
            shards=shards,
            region_convention=region_convention,
        )
    else:
        summary = write_segregation_outputs(
            vcf=vcf,
            samples=samples,
            results_dir=workdir / "results",
            all_tsv=output_tsv,
            summary_json=summary_json,
            bucket_outputs=bucket_outputs,
        )
    write_provenance(workdir=workdir, plan=plan, summary=summary)

    # Look at what actually landed on disk before claiming success.
    # Every failure would have stopped here.
    assert_outputs_are_prefixed(workdir / "results", run_prefix, stage="segregation",
                                preexisting=_preexisting)

    rows = int(summary.get("rows", 0))
    bucket_counts = {bucket: int((summary.get("bucket_counts") or {}).get(bucket, 0)) for bucket in BUCKETS}
    complete = bool(summary.get("lossless") and _complete_from_outputs(outputs))
    message = "Segregation complete; all variants preserved in lossless buckets." if complete else "Segregation output missing or not lossless."

    write_checkpoint(
        workdir=workdir,
        plan=plan,
        complete=complete,
        rows=rows,
        bucket_counts=bucket_counts,
        dry_run=False,
        message=message,
        inputs_identity=inputs_identity if complete else None,
    )

    return SegregationStageResult(
        workdir=workdir,
        input_vcf=vcf,
        samples=samples,
        output_tsv=output_tsv,
        plan_json=plan_json,
        checkpoint_json=checkpoint_json,
        summary_json=summary_json,
        provenance_json=provenance_json,
        bucket_outputs=bucket_outputs,
        executed=True,
        complete=complete,
        rows=rows,
        bucket_counts=bucket_counts,
        message=message,
        threads=threads,
        shards=shards,
        regions=regions_path,
        execution_mode=plan.get("execution_mode", "serial"),
    )


def verify_segregation_stage(workdir: Union[str, Path]) -> dict[str, Any]:
    workdir = Path(workdir)
    checkpoint = stage_checkpoint_path(workdir, "segregation")
    plan_json = workdir / "inputs" / "segregation_plan.json"

    checkpoint_data = read_json(checkpoint)
    plan_data = read_json(plan_json)
    source = checkpoint_data if checkpoint_data else plan_data
    outputs = source.get("outputs", {}) if isinstance(source.get("outputs"), dict) else {}

    output_tsv = outputs.get("all_variants") or outputs.get("segregation_tsv")
    summary_json = outputs.get("summary") or str(workdir / "results" / "segregation_summary.json")
    provenance_json = outputs.get("provenance") or str(workdir / "results" / "provenance.json")
    summary_data = read_json(Path(summary_json)) if summary_json else {}

    bucket_files = {bucket: file_info(outputs.get(bucket) or (workdir / "results" / f"{bucket}.tsv")) for bucket in BUCKETS}
    rows = count_tsv_rows(Path(output_tsv)) if output_tsv else 0
    output = file_info(output_tsv)
    bucket_counts = {bucket: count_tsv_rows(Path(info["path"])) if info.get("path") else 0 for bucket, info in bucket_files.items()}
    bucket_sum = sum(bucket_counts.values())
    lossless = bool(rows == bucket_sum and output.get("exists"))
    # The tables must also hold what the summary says they hold: a killed --overwrite run truncates the
    # all_variants table and the buckets together, so they would still sum to each other.
    summary_rows = summary_data.get("rows")
    matches_summary = summary_rows is not None and int(summary_rows) == rows

    # A table written by an older method is not a finished result.
    current = summary_data.get("method") == SEGREGATION_METHOD
    complete = bool(
        (checkpoint_data.get("complete") or summary_data.get("lossless"))
        and current
        and output.get("exists")
        and file_info(summary_json).get("nonempty")
        and file_info(provenance_json).get("nonempty")
        and lossless
        and matches_summary
    )

    return {
        "workdir": str(workdir),
        "complete": complete,
        "lossless": lossless,
        "checkpoint": file_info(checkpoint),
        "plan_json": file_info(plan_json),
        "summary_json": file_info(summary_json),
        "provenance_json": file_info(provenance_json),
        "input_vcf": checkpoint_data.get("input_vcf") or plan_data.get("input_vcf"),
        "samples": checkpoint_data.get("samples") or plan_data.get("samples"),
        "output_tsv": output,
        "bucket_outputs": bucket_files,
        "rows": rows,
        "summary_rows": summary_rows,
        "bucket_counts": bucket_counts,
        "bucket_sum": bucket_sum,
        "threads": checkpoint_data.get("threads") or plan_data.get("threads") or summary_data.get("threads"),
        "shards": checkpoint_data.get("shards") or plan_data.get("shards") or summary_data.get("shards"),
        "regions": checkpoint_data.get("regions") or plan_data.get("regions") or summary_data.get("regions"),
        "execution_mode": checkpoint_data.get("execution_mode") or plan_data.get("execution_mode") or summary_data.get("execution_mode"),
        "checkpoint_data": checkpoint_data,
        "summary_data": summary_data,
        "method": summary_data.get("method"),
        "method_current": current,
    }


def print_segregation_stage_result(result: SegregationStageResult) -> None:
    print("Segregation stage")
    print("=================")
    print(f"Workdir             : {result.workdir}")
    print(f"Input VCF           : {result.input_vcf}")
    print(f"Samples             : {result.samples}")
    print(f"All variants TSV    : {result.output_tsv}")
    print(f"Summary             : {result.summary_json}")
    print(f"Provenance          : {result.provenance_json}")
    print(f"Plan JSON           : {result.plan_json}")
    print(f"Checkpoint          : {result.checkpoint_json}")
    print(f"Executed            : {result.executed}")
    print(f"Complete            : {result.complete}")
    print(f"Rows                : {result.rows}")
    print(f"Threads             : {result.threads}")
    print(f"Shards              : {result.shards}")
    print(f"Regions             : {result.regions or '.'}")
    print(f"Execution mode      : {result.execution_mode}")
    if result.bucket_counts:
        print("Bucket counts       : " + " ".join(f"{k}={v}" for k, v in result.bucket_counts.items()))
    print(f"Message             : {result.message}")


def print_segregation_verify(report: dict[str, Any]) -> None:
    print("Segregation verification")
    print("========================")
    print(f"Workdir             : {report.get('workdir')}")
    print(f"Complete            : {report.get('complete')}")
    print(f"Lossless            : {report.get('lossless')}")
    print(f"Input VCF           : {report.get('input_vcf')}")
    print(f"Samples             : {report.get('samples')}")
    print(f"Rows                : {report.get('rows')}")
    print(f"Threads             : {report.get('threads') or '.'}")
    print(f"Shards              : {report.get('shards') or '.'}")
    print(f"Regions             : {report.get('regions') or '.'}")
    print(f"Execution mode      : {report.get('execution_mode') or '.'}")
    print(f"Bucket sum          : {report.get('bucket_sum')}")
    print(f"Checkpoint          : {'OK' if report['checkpoint']['nonempty'] else 'MISSING'} {report['checkpoint']['path']}")
    print(f"Plan JSON           : {'OK' if report['plan_json']['nonempty'] else 'MISSING'} {report['plan_json']['path']}")
    print(f"Summary             : {'OK' if report['summary_json']['nonempty'] else 'MISSING'} {report['summary_json']['path']}")
    print(f"Provenance          : {'OK' if report['provenance_json']['nonempty'] else 'MISSING'} {report['provenance_json']['path']}")
    out = report["output_tsv"]
    print(f"All variants TSV    : {'OK' if out['nonempty'] else 'MISSING'} {out['path']}")
    print("")
    print("Buckets")
    print("-------")
    counts = report.get("bucket_counts") or {}
    for bucket, info in (report.get("bucket_outputs") or {}).items():
        state = "OK" if info.get("exists") else "MISSING"
        print(f"{bucket:<26}: {state} rows={counts.get(bucket, 0)} {info.get('path')}")
