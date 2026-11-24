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
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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
    unprefixed_outputs,
)
from paralome.checkpoints import (
    count_table_rows,
    file_info as _file_info,
    output_sizes_agree,
    recorded_output_sizes,
    stage_checkpoint_path,
)
from paralome.filters.config import config_sha256, default_filter_config_path
from paralome.filters.profiles import get_profile
from paralome.filters.quality import check_profile_quality
from paralome.filters.table import (
    _profile_outputs as _table_profile_outputs,
    FILTER_METHOD,
    PROFILE_BUCKETS,
    require_applicable_scoring,
    require_quality_columns,
    filter_profile_scope_table,
    filter_profiles_scope_table,
    normalize_profiles,
    normalize_scopes,
)
from paralome.pipeline.region_shards import check_region_convention, merge_tsv_fragments, read_regions, split_tsv_by_regions
from paralome.errors import ParalomeError


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")




def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open() as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class FilterStageResult:
    workdir: Path
    profile: str
    input_segregation: Path
    out_prefix: Path
    plan_json: Path
    checkpoint_json: Path
    summary_json: Path
    candidates_tsv: Path
    complete: bool
    executed: bool
    message: str
    counts: dict[str, int] | None = None
    profiles: list[str] | None = None
    scope: list[str] | None = None
    profile_summaries: dict[str, dict[str, Any]] | None = None
    aggregate_summary_json: Path | None = None
    threads: int = 1
    shards: int = 1
    regions: Path | None = None
    execution_mode: str = "serial"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def _ensure_dirs(workdir: Path) -> None:
    for name in ("inputs", "results", "logs", "scripts", "checkpoints", "temp"):
        (workdir / name).mkdir(parents=True, exist_ok=True)


def _prefix_for(workdir: Path, profile: str, prefix: Optional[str] = None) -> Path:
    slug = (profile or "none").strip().lower().replace("_", "-")
    # Workdir.name is literally "filter", so this produced
    # filter.family-segregating for every family ever run. The project ID was
    # always the intent -- the `prefix` argument existed and nothing passed it.
    root = prefix or derive_file_prefix(workdir)
    return workdir / "results" / f"{root}.{slug}"


def _profile_outputs(workdir: Path, profile: str, prefix: Optional[str] = None) -> dict[str, str]:
    """Per-profile filter tables, carrying the project ID.

    The profile directory already disambiguates these WITHIN a project, but
    unique in its own directory is not unique as a filename, and `pass.tsv` is
    the extreme case of that.
    """
    profile_dir = workdir / "results" / profile
    run_prefix = prefix or derive_file_prefix(workdir)
    def _out(name: str) -> str:
        return str(profile_dir / prefixed_name(name, run_prefix))
    return {
        "all": _out("all.tsv"),
        "pass": _out("pass.tsv"),
        "fail_quality": _out("fail_quality.tsv"),
        "fail_population": _out("fail_population.tsv"),
        "fail_annotation": _out("fail_annotation.tsv"),
        "fail_segregation": _out("fail_segregation.tsv"),
        "fail_multiple": _out("fail_multiple.tsv"),
        "summary": _out("filter_summary.json"),
    }


def build_filter_plan(
    *,
    segregation: Union[str, Path],
    workdir: Union[str, Path],
    profile: str = "family-segregating",
    profiles: str | list[str] | tuple[str, ...] | None = None,
    scope: str | list[str] | tuple[str, ...] | None = None,
    filter_config: str | Optional[Path] = None,
    prefix: Optional[str] = None,
    overwrite: bool = False,
    threads: int = 1,
    shards: int = 1,
    regions: Optional[Path] = None,
) -> dict[str, Any]:
    workdir = Path(workdir)
    segregation = Path(segregation)
    threads = max(int(threads or 1), 1)
    shards = max(int(shards or 1), 1)
    filter_config_path = Path(filter_config) if filter_config else None
    profile_names = normalize_profiles(profiles, profile)
    scopes = normalize_scopes(scope)

    cfg_path = filter_config_path if filter_config_path else default_filter_config_path()
    # Validate early.
    profile_descriptions: dict[str, str] = {}
    for name in profile_names:
        prof = get_profile(name, filter_config_path)
        # A threshold no test reads is refused here, before any work.
        check_profile_quality(prof)
        profile_descriptions[prof.name] = prof.description

    primary_profile = profile_names[0]
    out_prefix = _prefix_for(workdir, primary_profile, prefix)
    run_prefix = prefix or derive_file_prefix(workdir)
    profile_outputs = {name: _profile_outputs(workdir, name, run_prefix) for name in profile_names}
    aggregate_summary = workdir / "results" / prefixed_name("filtering_summary.json", run_prefix)

    plan = {
        "schema": "paralome.filtering.plan.v2",
        "stage": "filtering",
        "created": _now(),
        "workdir": str(workdir),
        "profile": primary_profile,
        "profiles": profile_names,
        "profile_descriptions": profile_descriptions,
        "scope": scopes,
        "input_segregation": str(segregation),
        "filter_config": str(cfg_path),
        "filter_config_sha256": config_sha256(filter_config_path),
        "out_prefix": str(out_prefix),
        "outputs": {
            "aggregate_summary": str(aggregate_summary),
            "profiles": profile_outputs,
            # Legacy keys retained so older status readers do not crash.
            "candidates": str(Path(str(out_prefix) + ".candidates.tsv")),
            "segregating": str(Path(str(out_prefix) + ".segregating.tsv")),
            "non_segregating": str(Path(str(out_prefix) + ".non_segregating.tsv")),
            "uninformative": str(Path(str(out_prefix) + ".uninformative.tsv")),
            "other": str(Path(str(out_prefix) + ".other.tsv")),
            "summary": str(aggregate_summary),
        },
        "overwrite": bool(overwrite),
        "threads": int(max(1, threads)),
        "shards": int(max(1, shards)),
        "regions": str(regions) if regions else None,
        "execution_mode": "parallel_region_sharded" if (regions or int(max(1, threads)) > 1 or int(max(1, shards)) > 1) else "serial",
    }
    return plan


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(checkpoint)
    checkpoint["updated"] = _now()
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")


def _aggregate_counts(profile_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The stage's counts across its profiles.

    ``scanned``, ``total`` and ``pass`` are SUMS over the profiles, and every
    profile reads the whole table, so with two profiles they are twice the
    variant count. The sums are kept under their names (for older readers),
    each profile's own counts stand beside them, and the printers say what the
    sums are.
    """
    total_profiles = len(profile_summaries)
    by_profile = {name: (s.get("counts") or {}) for name, s in sorted(profile_summaries.items())}
    scanned = sum(int(c.get("scanned", 0)) for c in by_profile.values())
    total = sum(int(c.get("total", 0)) for c in by_profile.values())
    passed = sum(int(c.get("pass", 0)) for c in by_profile.values())
    return {
        "profiles": total_profiles, "scanned": scanned, "total": total, "pass": passed,
        "total_by_profile": {name: int(c.get("total", 0)) for name, c in by_profile.items()},
        "pass_by_profile": {name: int(c.get("pass", 0)) for name, c in by_profile.items()},
        "note": "scanned, total and pass are summed over the profiles; each profile reads the whole table",
    }


def counts_label(counts: dict[str, Any]) -> str:
    """How a printer should qualify the aggregate total and pass."""
    profiles = int(counts.get("profiles") or 0)
    return f" (summed over {profiles} profiles)" if profiles > 1 else ""


def _write_aggregate_summary(plan: dict[str, Any], profile_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    outputs = plan.get("outputs", {}) if isinstance(plan.get("outputs"), dict) else {}
    path = Path(outputs.get("aggregate_summary") or outputs.get("summary") or Path(plan["workdir"]) / "results" / "filtering_summary.json")
    counts = _aggregate_counts(profile_summaries)
    lossless_by_profile = {
        name: bool(summary.get("lossless")) for name, summary in sorted(profile_summaries.items())
    }
    summary = {
        "schema": "paralome.filtering.summary.v2",
        "stage": "filtering",
        "created": plan.get("created"),
        "finished": _now(),
        "input_segregation": plan.get("input_segregation"),
        "profiles": list(profile_summaries),
        "scope": plan.get("scope", ["segregation_pass"]),
        "filter_config": plan.get("filter_config"),
        "filter_config_sha256": plan.get("filter_config_sha256"),
        "counts": counts,
        "lossless_by_profile": lossless_by_profile,
        "lossless": all(lossless_by_profile.values()) if lossless_by_profile else False,
        "profile_summaries": {
            name: {
                "summary": summary.get("outputs", {}).get("summary"),
                "counts": summary.get("counts", {}),
                "outputs": summary.get("outputs", {}),
                "lossless": summary.get("lossless"),
            }
            for name, summary in sorted(profile_summaries.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _checkpoint_from_summary(plan: dict[str, Any], summary: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    outputs = plan.get("outputs", {}) if isinstance(plan.get("outputs"), dict) else {}
    return {
        "schema": "paralome.filtering.checkpoint.v2",
        "stage": "filtering",
        "complete": bool(complete),
        "profile": plan.get("profile"),
        "profiles": plan.get("profiles", [plan.get("profile")]),
        "scope": plan.get("scope", ["segregation_pass"]),
        "input_segregation": plan.get("input_segregation"),
        "filter_config": plan.get("filter_config"),
        "filter_config_sha256": plan.get("filter_config_sha256"),
        "outputs": outputs,
        "threads": int(plan.get("threads", 1)),
        "shards": int(plan.get("shards", 1)),
        "regions": plan.get("regions"),
        "execution_mode": plan.get("execution_mode", "serial"),
        "counts": summary.get("counts", {}) if isinstance(summary, dict) else {},
        "lossless": summary.get("lossless") if isinstance(summary, dict) else None,
        "started": plan.get("started"),
        "finished": _now() if complete else None,
    }


def _profile_tables_agree(outmap: dict[str, Any], counts: dict[str, Any], recorded: Any) -> bool:
    """Whether a profile's tables on disk are the ones its record describes.

    The tables are rewritten in place, so a run killed part way through an
    --overwrite could leave the earlier run's summary (lossless, its counts)
    beside truncated tables. A checkpoint records each table's size as it
    closes; without one (an older run) the rows are counted and compared with
    the summary's counts.
    """
    tables = {key: path for key, path in outmap.items() if key != "summary"}
    if isinstance(recorded, dict) and recorded:
        return output_sizes_agree(recorded, tables)
    expected = {"all": counts.get("total"), **{bucket: counts.get(bucket) for bucket in PROFILE_BUCKETS}}
    for key, rows in expected.items():
        if rows is None or not tables.get(key):
            return False
        if count_table_rows(tables[key]) != int(rows):
            return False
    return True


def _invalidate_previous_run(plan: dict[str, Any], checkpoint_path: Path) -> None:
    """Withdraw the earlier run's claim of completion before its tables are rewritten.

    The bucket tables are opened with "w" and rewritten in place, and the
    checkpoint and summaries are written only at the end. So the checkpoint
    says incomplete, and the summaries this run will write are gone, for as
    long as the tables are being written: a run killed in between cannot
    leave a complete checkpoint beside truncated tables.
    """
    checkpoint = _checkpoint_from_summary(plan, {"counts": {}, "lossless": False}, complete=False)
    checkpoint["message"] = "Filtering running; the tables are being rewritten."
    _write_checkpoint(checkpoint_path, checkpoint)
    outputs = plan.get("outputs", {}) if isinstance(plan.get("outputs"), dict) else {}
    stale = [outputs.get("aggregate_summary"), outputs.get("summary")]
    profile_outputs = outputs.get("profiles", {}) if isinstance(outputs.get("profiles"), dict) else {}
    stale.extend(outmap.get("summary") for outmap in profile_outputs.values() if isinstance(outmap, dict))
    for path in stale:
        if path:
            Path(path).unlink(missing_ok=True)


def _recorded_profile_sizes(plan: dict[str, Any]) -> dict[str, dict[str, int]]:
    outputs = plan.get("outputs", {}) if isinstance(plan.get("outputs"), dict) else {}
    profile_outputs = outputs.get("profiles", {}) if isinstance(outputs.get("profiles"), dict) else {}
    return {
        name: recorded_output_sizes({key: path for key, path in outmap.items() if key != "summary"})
        for name, outmap in profile_outputs.items()
        if isinstance(outmap, dict)
    }


def _profile_status_from_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    profiles = outputs.get("profiles", {}) if isinstance(outputs.get("profiles"), dict) else {}
    status: dict[str, Any] = {}
    for name, outmap in profiles.items():
        if not isinstance(outmap, dict):
            continue
        files = {key: _file_info(path) for key, path in outmap.items()}
        summary = _read_json(Path(outmap.get("summary", ""))) if outmap.get("summary") else {}
        status[name] = {
            "outputs": files,
            "paths": dict(outmap),
            "summary": files.get("summary"),
            "counts": summary.get("counts", {}),
            "scope": summary.get("scope"),
            "lossless": summary.get("lossless"),
            "method": summary.get("method"),
        }
    return status



def _filter_shard_worker(job: dict[str, Any]) -> dict[str, Any]:
    profile = str(job["profile"])
    summary = filter_profile_scope_table(
        segregation_tsv=Path(job["shard_path"]),
        results_dir=Path(job["fragment_dir"]),
        profile_name=profile,
        scope=job.get("scope"),
        filter_config=Path(job["filter_config"]) if job.get("filter_config") else None,
    )
    return {
        "shard_id": int(job["shard_id"]),
        "profile": profile,
        "fragment_dir": str(Path(job["fragment_dir"]) / profile),
        "summary": summary,
    }


def _merge_filter_profile_fragments(
    *,
    profile: str,
    fragment_dirs: list[Path],
    results_dir: Path,
    fieldnames: list[str],
    prefix: Optional[str] = None,
) -> dict[str, Any]:
    # THE canonical map, imported, not a hand-written copy, so the plan, the serial writer and this
    # merge all name the same files.
    outputs = _table_profile_outputs(results_dir, profile, prefix)
    counts_by_bucket: dict[str, int] = {}
    for bucket in ("all", *PROFILE_BUCKETS):
        counts_by_bucket[bucket] = merge_tsv_fragments(
            fragment_paths=[d / f"{bucket}.tsv" for d in fragment_dirs],
            output_path=outputs[bucket],
            fieldnames=fieldnames,
        )
    failed_stage_counts: Counter[str] = Counter()
    not_tested: Counter[str] = Counter()
    scanned = skipped = total = 0
    metadata: dict[str, Any] = {}
    for frag_dir in fragment_dirs:
        data = _read_json(frag_dir / "filter_summary.json")
        if not data:
            continue
        metadata = metadata or data
        if not metadata.get("segregation_scoring") and data.get("segregation_scoring"):
            metadata = {**metadata, "segregation_scoring": data.get("segregation_scoring")}
        c = data.get("counts", {}) if isinstance(data.get("counts"), dict) else {}
        scanned += int(c.get("scanned", 0))
        skipped += int(c.get("skipped_by_scope", 0))
        total += int(c.get("total", 0))
        failed_stage_counts.update({k: int(v) for k, v in (data.get("failed_stage_counts") or {}).items()})
        not_tested.update({k: int(v) for k, v in (data.get("quality_not_tested") or {}).items()})
    counts = {
        "scanned": int(scanned),
        "skipped_by_scope": int(skipped),
        "total": int(total),
        "pass": int(counts_by_bucket.get("pass", 0)),
        "fail_quality": int(counts_by_bucket.get("fail_quality", 0)),
        "fail_population": int(counts_by_bucket.get("fail_population", 0)),
        "fail_annotation": int(counts_by_bucket.get("fail_annotation", 0)),
        "fail_segregation": int(counts_by_bucket.get("fail_segregation", 0)),
        "fail_multiple": int(counts_by_bucket.get("fail_multiple", 0)),
    }
    counts["bucket_sum"] = sum(int(counts.get(bucket, 0)) for bucket in PROFILE_BUCKETS)
    summary = {
        "schema": "paralome.filtering.profile_summary.v2",
        "stage": "filtering",
        "profile": profile,
        "description": metadata.get("description", ""),
        "mode": metadata.get("mode"),
        "rarity": metadata.get("rarity"),
        "scope": metadata.get("scope"),
        "thresholds": metadata.get("thresholds", {}),
        "quality": metadata.get("quality", {}),
        # See filters.table.filter_profile_scope_table.
        "quality_applied": metadata.get("quality_applied"),
        "method": metadata.get("method"),
        "segregation_scoring": metadata.get("segregation_scoring"),
        "population": metadata.get("population", {}),
        "annotation": metadata.get("annotation", {}),
        "filter_config": metadata.get("filter_config"),
        "filter_config_sha256": metadata.get("filter_config_sha256"),
        "counts": counts,
        "failed_stage_counts": {k: int(v) for k, v in sorted(failed_stage_counts.items())},
        # Summed over the shards (filters.table.filter_profile_scope_table).
        "quality_not_tested": {k: int(v) for k, v in sorted(not_tested.items())},
        "input": metadata.get("input"),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "lossless": bool(counts["total"] == counts["bucket_sum"]),
    }
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["lossless"]:
        raise RuntimeError(f"Filtering row-preservation validation failed for {profile}: total={counts['total']} bucket_sum={counts['bucket_sum']}")
    return summary


def _filter_profiles_scope_table_parallel(
    *,
    segregation_tsv: Path,
    results_dir: Path,
    profiles: list[str],
    scope: list[str],
    filter_config: Optional[Path],
    workdir: Path,
    threads: int,
    shards: int,
    regions: Optional[Path],
    prefix: Optional[str] = None,
    region_convention: Optional[str] = None,
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    threads = max(int(threads or 1), 1)
    shards = max(int(shards or 1), 1)
    # The whole table is checked against each profile before it is
    # split: a shard holding no rows could not refuse on its own.
    with open(segregation_tsv, newline="") as fh:
        header = next(csv.reader(fh, delimiter="\t"), [])
    for name in profiles:
        prof = get_profile(name, filter_config)
        require_quality_columns(header, prof, segregation_tsv)
        require_applicable_scoring(segregation_tsv, prof)
    region_list = read_regions(regions, region_convention)
    shard_dir = workdir / "inputs" / "filter_shards"
    fragment_root = workdir / "results" / "shard_fragments"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    if fragment_root.exists():
        shutil.rmtree(fragment_root)
    shard_paths, input_fields, shard_counts = split_tsv_by_regions(
        input_tsv=segregation_tsv,
        shard_dir=shard_dir,
        shards=shards,
        regions=region_list,
        prefix="shard",
    )
    fieldnames = list(input_fields)
    for col in ("filter_profile", "filter_scope", "filter_bucket", "filter_failed_stages", "filter_reasons", "quality_score", "population_score", "annotation_score", "segregation_score_filter"):
        if col not in fieldnames:
            fieldnames.append(col)
    jobs: list[dict[str, Any]] = []
    for profile in profiles:
        for idx, shard_path in enumerate(shard_paths):
            jobs.append(
                {
                    "shard_id": idx,
                    "profile": profile,
                    "shard_path": str(shard_path),
                    "fragment_dir": str(fragment_root / profile / f"shard_{idx:04d}"),
                    "scope": scope,
                    "filter_config": str(filter_config) if filter_config else None,
                }
            )
    shard_results: list[dict[str, Any]] = []
    workers = min(threads, len(jobs)) if jobs else 1
    if workers <= 1:
        for job in jobs:
            shard_results.append(_filter_shard_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_filter_shard_worker, job) for job in jobs]
            for fut in as_completed(futures):
                shard_results.append(fut.result())
    summaries: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        profile_results = sorted((r for r in shard_results if r.get("profile") == profile), key=lambda x: int(x["shard_id"]))
        fragment_dirs = [Path(r["fragment_dir"]) for r in profile_results]
        summaries[profile] = _merge_filter_profile_fragments(
            profile=profile,
            fragment_dirs=fragment_dirs,
            results_dir=results_dir,
            fieldnames=fieldnames,
            prefix=prefix,
        )
        summaries[profile]["threads"] = int(threads)
        summaries[profile]["shards"] = int(shards)
        summaries[profile]["regions"] = str(regions) if regions else None
        summaries[profile]["usable_regions"] = len(region_list)
        summaries[profile]["shard_row_counts"] = shard_counts
        summaries[profile]["execution_mode"] = "parallel_region_sharded"
    return summaries, shard_counts


def verify_filter_stage(workdir: Union[str, Path]) -> dict[str, Any]:
    workdir = Path(workdir)
    checkpoint_path = stage_checkpoint_path(workdir, "filtering")
    plan_path = workdir / "inputs" / "filter_plan.json"
    checkpoint = _read_json(checkpoint_path)
    plan = _read_json(plan_path)
    source = checkpoint if checkpoint else plan
    outputs = source.get("outputs", {}) if isinstance(source.get("outputs"), dict) else {}

    file_status = {name: _file_info(path) for name, path in outputs.items() if name != "profiles"}
    profile_status = _profile_status_from_outputs(outputs)
    aggregate_summary = _read_json(Path(outputs.get("aggregate_summary") or outputs.get("summary") or "")) if outputs.get("aggregate_summary") or outputs.get("summary") else {}

    complete = bool(checkpoint.get("complete"))
    if not complete:
        complete = bool(file_status.get("aggregate_summary", file_status.get("summary", {})).get("nonempty"))
    # A run that withdrew the checkpoint and did not finish.
    if checkpoint and checkpoint.get("complete") is False and not checkpoint.get("dry_run"):
        complete = False
    if profile_status:
        complete = bool(complete and all(bool(p.get("summary", {}).get("nonempty")) and p.get("lossless") is not False for p in profile_status.values()))
    else:
        # Nothing to judge the tables by is not a finished stage.
        complete = False
    tables_match_record = None
    if complete:
        recorded = checkpoint.get("output_bytes") if checkpoint.get("complete") else None
        recorded = recorded if isinstance(recorded, dict) else {}
        tables_match_record = all(
            _profile_tables_agree(p.get("paths") or {}, p.get("counts") or {}, recorded.get(name))
            for name, p in profile_status.items()
        )
        complete = bool(tables_match_record)
    # A result filtered by an older method is not reused.
    stale_method = bool(profile_status) and any(p.get("method") != FILTER_METHOD for p in profile_status.values())
    if stale_method:
        complete = False

    return {
        "schema": "paralome.filtering.verify.v2",
        "stage": "filtering",
        "workdir": str(workdir),
        "complete": bool(complete),
        "status": "complete" if complete else "incomplete",
        "tables_match_record": tables_match_record,
        "stale_method": stale_method,
        "checkpoint": _file_info(checkpoint_path),
        "plan": _file_info(plan_path),
        "profile": checkpoint.get("profile") or plan.get("profile") or aggregate_summary.get("profile"),
        "profiles": checkpoint.get("profiles") or plan.get("profiles") or aggregate_summary.get("profiles") or list(profile_status),
        "scope": checkpoint.get("scope") or plan.get("scope") or aggregate_summary.get("scope"),
        "input_segregation": checkpoint.get("input_segregation") or plan.get("input_segregation") or aggregate_summary.get("input_segregation"),
        "filter_config": checkpoint.get("filter_config") or plan.get("filter_config") or aggregate_summary.get("filter_config"),
        "filter_config_sha256": checkpoint.get("filter_config_sha256") or plan.get("filter_config_sha256") or aggregate_summary.get("filter_config_sha256"),
        "outputs": file_status,
        "profile_outputs": profile_status,
        "counts": checkpoint.get("counts") or aggregate_summary.get("counts", {}),
        "lossless": checkpoint.get("lossless") if "lossless" in checkpoint else aggregate_summary.get("lossless"),
        "threads": checkpoint.get("threads") or plan.get("threads") or aggregate_summary.get("threads"),
        "shards": checkpoint.get("shards") or plan.get("shards") or aggregate_summary.get("shards"),
        "regions": checkpoint.get("regions") or plan.get("regions") or aggregate_summary.get("regions"),
        "execution_mode": checkpoint.get("execution_mode") or plan.get("execution_mode") or aggregate_summary.get("execution_mode"),
    }


def _request_differs(existing: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """What a finished run was made for that this plan does not ask for. Empty: the same request."""
    differences: list[str] = []
    recorded_profiles = [str(p) for p in (existing.get("profiles") or [])]
    if sorted(recorded_profiles) != sorted(str(p) for p in (plan.get("profiles") or [])):
        differences.append(f"profiles {','.join(recorded_profiles) or '.'} -> {','.join(plan.get('profiles') or [])}")
    recorded_scope = [str(x) for x in (existing.get("scope") or [])]
    if sorted(recorded_scope) != sorted(str(x) for x in (plan.get("scope") or [])):
        differences.append(f"scope {','.join(recorded_scope) or '.'} -> {','.join(plan.get('scope') or [])}")
    recorded_sha = existing.get("filter_config_sha256")
    if recorded_sha and plan.get("filter_config_sha256") and recorded_sha != plan.get("filter_config_sha256"):
        differences.append("filter config contents")
    return differences


def run_filter_stage(
    *,
    segregation: Union[str, Path],
    workdir: Union[str, Path],
    profile: str = "family-segregating",
    profiles: str | list[str] | tuple[str, ...] | None = None,
    scope: str | list[str] | tuple[str, ...] | None = None,
    filter_config: str | Optional[Path] = None,
    prefix: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    threads: int = 1,
    shards: int = 1,
    regions: Union[str, Path, None] = None,
    region_convention: Optional[str] = None,
) -> FilterStageResult:
    workdir = Path(workdir)
    segregation = Path(segregation)
    threads = max(int(threads or 1), 1)
    shards = max(int(shards or 1), 1)
    regions_path = Path(regions) if regions else None
    region_convention = check_region_convention(region_convention)
    if not segregation.exists():
        raise ParalomeError(104, f"Segregation table not found: {segregation}")

    # Resolved ONCE here, from the workdir, and threaded into every
    # writer. The writers must never derive it themselves -- results_dir is
    # <workdir>/results and "results" is not a generic stage name, so deriving
    # there yields "results".
    # And validated, then announced. A falsy prefix reaching
    # prefixed_name() silently produces bare filenames.
    run_prefix = require_file_prefix(prefix or derive_file_prefix(workdir), stage="filtering")
    print(stage_banner("filtering", run_prefix, workdir), flush=True)

    _ensure_dirs(workdir)
    # What was already here, so the post-write check can tell this run's
    # mistakes from a previous convention's leftovers.
    _preexisting = snapshot_results(workdir / "results")
    plan = build_filter_plan(
        segregation=segregation,
        workdir=workdir,
        profile=profile,
        profiles=profiles,
        scope=scope,
        filter_config=filter_config,
        prefix=prefix,
        overwrite=overwrite,
        threads=threads,
        shards=shards,
        regions=regions_path,
    )
    plan["started"] = _now()
    plan_path = workdir / "inputs" / "filter_plan.json"
    checkpoint_path = stage_checkpoint_path(workdir, "filtering")

    # The input as this run read it, copied every run. It was
    # refreshed only under --overwrite, so the archived input could be an
    # earlier table's, and a copy that failed was passed over in silence.
    input_copy = workdir / "inputs" / "segregation.tsv"
    if segregation.resolve() != input_copy.resolve():
        shutil.copy2(segregation, input_copy)

    # Reuse is conditional on the outputs actually carrying THIS run's prefix. Not "having a prefix":
    # the wrong prefix is what a half-migrated tree is full of. `verify_filter_stage` is a second
    # completeness reader for this stage, beside state.py, and the two must agree, or the supervisor
    # would re-select a stage that keeps declining to run.
    existing = verify_filter_stage(workdir)
    reusable = bool(existing.get("complete")) and not unprefixed_outputs(
        workdir / "results", run_prefix
    )
    # AND IT MUST BE THE SAME REQUEST. verify judges the stage by the checkpoint's own profiles, so a
    # result made for other profiles (a rerun with --segregation-profile unrelated-probands after a
    # family-segregating run, say), another scope or another profile config is not this request's,
    # and the stage runs.
    reusable = reusable and not _request_differs(existing, plan)
    if reusable and not overwrite:
        outputs = plan["outputs"]
        profile_names = list(plan.get("profiles") or [plan.get("profile")])
        primary = profile_names[0]
        primary_outputs = outputs.get("profiles", {}).get(primary, {}) if isinstance(outputs.get("profiles"), dict) else {}
        return FilterStageResult(
            workdir=workdir,
            profile=str(plan["profile"]),
            profiles=profile_names,
            scope=list(plan.get("scope") or []),
            input_segregation=segregation,
            out_prefix=Path(plan["out_prefix"]),
            plan_json=plan_path,
            checkpoint_json=checkpoint_path,
            summary_json=Path(primary_outputs.get("summary") or outputs["summary"]),
            aggregate_summary_json=Path(outputs["aggregate_summary"]),
            candidates_tsv=Path(primary_outputs.get("pass") or outputs["candidates"]),
            complete=True,
            executed=False,
            message="Filtering already complete. Use --overwrite to rerun.",
            counts=existing.get("counts", {}),
            profile_summaries=existing.get("profile_outputs", {}),
            threads=threads,
            shards=shards,
            regions=regions_path,
            execution_mode=plan.get("execution_mode", "serial"),
        )

    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    outputs = plan["outputs"]

    if dry_run:
        checkpoint = _checkpoint_from_summary(plan, {"counts": {}, "lossless": False}, complete=False)
        checkpoint["dry_run"] = True
        _write_checkpoint(checkpoint_path, checkpoint)
        return FilterStageResult(
            workdir=workdir,
            profile=str(plan["profile"]),
            profiles=list(plan.get("profiles") or []),
            scope=list(plan.get("scope") or []),
            input_segregation=segregation,
            out_prefix=Path(plan["out_prefix"]),
            plan_json=plan_path,
            checkpoint_json=checkpoint_path,
            summary_json=Path(outputs["summary"]),
            aggregate_summary_json=Path(outputs["aggregate_summary"]),
            candidates_tsv=Path(outputs["candidates"]),
            complete=False,
            executed=False,
            message="Dry run complete; filtering was not executed.",
            counts={},
            profile_summaries={},
            threads=threads,
            shards=shards,
            regions=regions_path,
            execution_mode=plan.get("execution_mode", "serial"),
        )

    _invalidate_previous_run(plan, checkpoint_path)

    if threads > 1 or shards > 1 or regions_path is not None:
        profile_summaries, shard_counts = _filter_profiles_scope_table_parallel(
            segregation_tsv=segregation,
            results_dir=workdir / "results",
            profiles=list(plan.get("profiles") or []),
            scope=list(plan.get("scope") or []),
            filter_config=Path(filter_config) if filter_config else None,
            workdir=workdir,
            threads=threads,
            shards=shards,
            regions=regions_path,
            prefix=run_prefix,
            region_convention=region_convention,
        )
    else:
        shard_counts = []
        profile_summaries = filter_profiles_scope_table(
            segregation_tsv=segregation,
            results_dir=workdir / "results",
            profiles=plan.get("profiles"),
            scope=plan.get("scope"),
            filter_config=Path(filter_config) if filter_config else None,
            prefix=run_prefix,
        )
    aggregate_summary = _write_aggregate_summary(plan, profile_summaries)
    if threads > 1 or shards > 1 or regions_path is not None:
        aggregate_summary["threads"] = int(threads)
        aggregate_summary["shards"] = int(shards)
        aggregate_summary["regions"] = str(regions_path) if regions_path else None
        aggregate_summary["shard_row_counts"] = shard_counts
        aggregate_summary["execution_mode"] = "parallel_region_sharded"
        Path(outputs["aggregate_summary"]).write_text(json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n")
    # Before the checkpoint says complete, confirm on disk. Writing
    # `complete: true` next to a directory of bare filenames is what armed
    # every subsequent skip.
    assert_outputs_are_prefixed(workdir / "results", run_prefix, stage="filtering",
                                preexisting=_preexisting)

    checkpoint = _checkpoint_from_summary(plan, aggregate_summary, complete=True)
    # The size of every table this checkpoint vouches for; see
    # _profile_tables_agree.
    checkpoint["output_bytes"] = _recorded_profile_sizes(plan)
    _write_checkpoint(checkpoint_path, checkpoint)

    profile_names = list(plan.get("profiles") or [])
    primary = profile_names[0]
    primary_outputs = outputs.get("profiles", {}).get(primary, {}) if isinstance(outputs.get("profiles"), dict) else {}
    return FilterStageResult(
        workdir=workdir,
        profile=str(plan["profile"]),
        profiles=profile_names,
        scope=list(plan.get("scope") or []),
        input_segregation=segregation,
        out_prefix=Path(plan["out_prefix"]),
        plan_json=plan_path,
        checkpoint_json=checkpoint_path,
        summary_json=Path(primary_outputs.get("summary") or outputs["summary"]),
        aggregate_summary_json=Path(outputs["aggregate_summary"]),
        candidates_tsv=Path(primary_outputs.get("pass") or outputs["candidates"]),
        complete=True,
        executed=True,
        message="Filtering complete.",
        counts=aggregate_summary.get("counts", {}) if isinstance(aggregate_summary, dict) else {},
        profile_summaries=profile_summaries,
        threads=threads,
        shards=shards,
        regions=regions_path,
        execution_mode=plan.get("execution_mode", "serial"),
    )


def print_filter_stage_result(result: FilterStageResult) -> None:
    print("Filtering stage")
    print("===============")
    print(f"Workdir             : {result.workdir}")
    print(f"Profiles            : {','.join(result.profiles or [result.profile])}")
    print(f"Scope               : {','.join(result.scope or []) or '.'}")
    print(f"Input segregation   : {result.input_segregation}")
    print(f"Plan JSON           : {result.plan_json}")
    print(f"Checkpoint          : {result.checkpoint_json}")
    print(f"Aggregate summary   : {result.aggregate_summary_json or result.summary_json}")
    print(f"Primary pass TSV    : {result.candidates_tsv}")
    print(f"Executed            : {result.executed}")
    print(f"Complete            : {result.complete}")
    print(f"Threads             : {result.threads}")
    print(f"Shards              : {result.shards}")
    print(f"Regions             : {result.regions or '.'}")
    print(f"Execution mode      : {result.execution_mode}")
    print(f"Message             : {result.message}")
    counts = result.counts or {}
    if counts:
        print(
            "Counts              : "
            f"profiles={counts.get('profiles', 0)} "
            f"total={counts.get('total', 0)} "
            f"pass={counts.get('pass', 0)}" + counts_label(counts)
        )
    summaries = result.profile_summaries or {}
    if summaries:
        print("")
        print("Profiles")
        print("--------")
        for name, summary in sorted(summaries.items()):
            counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
            print(
                f"{name:<22} total={counts.get('total', 0)} pass={counts.get('pass', 0)} "
                f"fail_quality={counts.get('fail_quality', 0)} "
                f"fail_population={counts.get('fail_population', 0)} "
                f"fail_annotation={counts.get('fail_annotation', 0)} "
                f"fail_segregation={counts.get('fail_segregation', 0)} "
                f"fail_multiple={counts.get('fail_multiple', 0)}"
            )


def print_filter_verify(report: dict[str, Any]) -> None:
    print("Filtering verification")
    print("======================")
    print(f"Workdir             : {report.get('workdir')}")
    print(f"Profiles            : {','.join(report.get('profiles') or [])}")
    print(f"Scope               : {','.join(report.get('scope') or [])}")
    print(f"Complete            : {report.get('complete')}")
    print(f"Lossless            : {report.get('lossless')}")
    print(f"Threads             : {report.get('threads') or '.'}")
    print(f"Shards              : {report.get('shards') or '.'}")
    print(f"Regions             : {report.get('regions') or '.'}")
    print(f"Execution mode      : {report.get('execution_mode') or '.'}")
    print(f"Input segregation   : {report.get('input_segregation')}")
    print(f"Checkpoint          : {'OK' if report.get('checkpoint', {}).get('nonempty') else 'MISSING'} {report.get('checkpoint', {}).get('path')}")
    print(f"Plan JSON           : {'OK' if report.get('plan', {}).get('nonempty') else 'MISSING'} {report.get('plan', {}).get('path')}")
    print("")
    print("Outputs")
    print("-------")
    for name, info in sorted((report.get("outputs") or {}).items()):
        state = "OK" if info.get("nonempty") else ("EMPTY" if info.get("exists") else "MISSING")
        print(f"{name:<18}: {state} {info.get('path')}")
    profiles = report.get("profile_outputs") or {}
    if profiles:
        print("")
        print("Profiles")
        print("--------")
        for name, prof in sorted(profiles.items()):
            counts = prof.get("counts", {}) if isinstance(prof, dict) else {}
            print(
                f"{name:<22}: total={counts.get('total', 0)} pass={counts.get('pass', 0)} "
                f"bucket_sum={counts.get('bucket_sum', 0)} lossless={prof.get('lossless')}"
            )
    counts = report.get("counts") or {}
    if counts:
        print("")
        print("Aggregate counts" + counts_label(counts))
        print("----------------")
        for key in ("profiles", "scanned", "total", "pass"):
            print(f"{key:<18}: {counts.get(key, 0)}")
