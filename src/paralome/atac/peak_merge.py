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
import hashlib
import json
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from paralome.binaries import shell_join, worker_argv
from paralome.pipeline.jsonio import version_stale_reason
from paralome.checkpoints import (
    atomic_text_writer as _atomic_text_writer,
    atomic_write_text,
    exclusive_checkpoint_lock,
    stage_checkpoint_path,
)
from paralome.pipeline.resources import SampleParallelPlan, plan_sample_parallel
from paralome.pipeline.scheduler import detect_scheduler_capabilities, plan_array_tasks, require_array_task_ids
from paralome.pipeline.slurm import SlurmOptions, afterok, compress_array_ids, sbatch, write_sbatch
from paralome.atac.naming import sample_file_token as _sample_file_token
from paralome.pipeline.naming import slurm_job_name
from paralome.errors import ParalomeError
from paralome.csvio import skip_comment_lines


PEAK_MERGE_PLAN_SCHEMA = "paralome.atac.peak_merge_plan.v1"
PEAK_MERGE_SAMPLE_SCHEMA = "paralome.atac.peak_merge_sample.v1"
PEAK_MERGE_STAGE_SCHEMA = "paralome.atac.peak_merge.v3"
#: In every sample task signature and the plan identity, so peaks merged before width-capped regions
#: were made non-overlapping are merged again.
PEAK_MERGE_METHOD = "width_cap_non_overlapping.v2"


@dataclass(frozen=True)
class Peak:
    chrom: str
    start: int
    end: int
    sample_id: str
    source_id: str
    score: float = 0.0
    summit: Optional[int] = None


@dataclass
class MergedPeak:
    chrom: str
    start: int
    end: int
    samples: Set[str]
    components: List[Peak]


def read_narrowpeak(path: Union[str, Path], sample_id: str) -> List[Peak]:
    peaks: List[Peak] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError("%s:%d has fewer than 3 columns" % (path, line_no))
            chrom, start_text, end_text = fields[:3]
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start:
                raise ValueError("%s:%d has invalid interval %s:%d-%d" % (path, line_no, chrom, start, end))
            source_id = fields[3] if len(fields) > 3 and fields[3] else "%s:%d" % (sample_id, line_no)
            score = float(fields[6]) if len(fields) > 6 and fields[6] not in ("", ".") else 0.0
            summit = None
            if len(fields) > 9 and fields[9] not in ("", ".", "-1"):
                offset = int(fields[9])
                candidate = start + offset
                if start <= candidate < end:
                    summit = candidate
            peaks.append(Peak(chrom, start, end, sample_id, source_id, score, summit))
    return peaks


def _center_peak(peak: Peak, width: int) -> Peak:
    if width < 1:
        raise ValueError("oversized_peak_width must be >= 1")
    center = peak.summit if peak.summit is not None else (peak.start + peak.end) // 2
    start = max(0, center - width // 2)
    end = start + width
    return Peak(peak.chrom, start, end, peak.sample_id, peak.source_id, peak.score, center)


def normalize_oversized_peaks(
    peaks: Sequence[Peak],
    *,
    max_peak_width: int,
    oversized_peak_policy: str,
    oversized_peak_width: int,
) -> Tuple[List[Peak], int]:
    policy = oversized_peak_policy.strip().lower()
    if policy not in {"center", "error"}:
        raise ValueError("oversized_peak_policy must be 'center' or 'error'")
    if max_peak_width < 1:
        raise ValueError("max_peak_width must be >= 1")
    if oversized_peak_width < 1 or oversized_peak_width > max_peak_width:
        raise ValueError("oversized_peak_width must be between 1 and max_peak_width")

    normalized: List[Peak] = []
    changed = 0
    for peak in peaks:
        width = peak.end - peak.start
        if width <= max_peak_width:
            normalized.append(peak)
            continue
        if policy == "error":
            raise ValueError(
                "Peak %s from sample %s is %d bp, exceeding max_peak_width=%d"
                % (peak.source_id, peak.sample_id, width, max_peak_width)
            )
        normalized.append(_center_peak(peak, oversized_peak_width))
        changed += 1
    return normalized, changed


def merge_intervals(peaks: Sequence[Peak], max_gap: int = 0,
                    max_width: Optional[int] = None) -> List[MergedPeak]:
    if max_gap < 0:
        raise ValueError("max_gap must be >= 0")
    if max_width is not None and max_width < 1:
        raise ValueError("max_width must be >= 1")
    ordered = sorted(peaks, key=lambda p: (p.chrom, p.start, p.end, p.sample_id, p.source_id))
    if max_width is not None:
        oversized = [peak for peak in ordered if peak.end - peak.start > max_width]
        if oversized:
            peak = oversized[0]
            raise ValueError(
                "Input peak %s from sample %s is %d bp, exceeding max_width=%d"
                % (peak.source_id, peak.sample_id, peak.end - peak.start, max_width)
            )
    merged: List[MergedPeak] = []
    current: Optional[MergedPeak] = None
    for peak in ordered:
        if current is None:
            current = MergedPeak(peak.chrom, peak.start, peak.end, {peak.sample_id}, [peak])
            continue
        candidate_end = max(current.end, peak.end)
        joins = peak.chrom == current.chrom and peak.start <= current.end + max_gap
        if joins and max_width is not None and candidate_end - current.start > max_width:
            joins = False
        # A peak overlapping the previous region supports it too,
        # whichever region it is placed in.
        previous = merged[-1] if merged else None
        if previous is not None and previous.chrom == peak.chrom and peak.start < previous.end:
            previous.samples.add(peak.sample_id)
        if joins:
            current.end = candidate_end
            current.samples.add(peak.sample_id)
            current.components.append(peak)
        else:
            merged.append(current)
            # When the width cap refuses a peak that overlaps the
            # region, the new region starts at the region's end, so atlas
            # regions never overlap and no fragment is counted in two peaks.
            start = peak.start
            if peak.chrom == current.chrom and start < current.end:
                current.samples.add(peak.sample_id)
                start = current.end
            current = MergedPeak(peak.chrom, start, peak.end, {peak.sample_id}, [peak])
    if current is not None:
        merged.append(current)
    if max_width is not None and any(region.end - region.start > max_width for region in merged):
        raise RuntimeError("Peak merge produced an interval wider than max_width")
    return merged


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    # Stamped here rather than at each call site, so every checkpoint
    # this module writes records the version that produced it. See
    # pipeline.jsonio.CHECKPOINT_VERSION_KEY.
    from paralome.pipeline.jsonio import stamp_version

    atomic_write_text(path, json.dumps(stamp_version(payload), indent=2, sort_keys=True) + "\n")






def _read_peak_manifest(path: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    seen: Set[str] = set()
    path = path.expanduser().resolve()
    with path.open(newline="") as handle:
        # '#' lines are comments, as in every other manifest.
        reader = csv.DictReader(skip_comment_lines(handle))
        required = {"sample_id", "narrowpeak"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Peak manifest missing column(s): %s" % ", ".join(sorted(missing)))
        for row_no, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            peak_text = (row.get("narrowpeak") or "").strip()
            if not sample_id or not peak_text:
                raise ValueError("Invalid peak manifest row %d" % row_no)
            if sample_id in seen:
                raise ValueError("Duplicate sample_id in peak manifest: %s" % sample_id)
            seen.add(sample_id)
            peak_path = Path(peak_text).expanduser()
            if peak_path.is_absolute():
                candidates = [peak_path.resolve()]
            else:
                # Relative to the manifest's folder first, then cwd.
                candidates = [(parent / peak_path).resolve() for parent in path.parents]
                candidates.append((Path.cwd() / peak_path).resolve())
            peak_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            if not peak_path.is_file():
                raise ParalomeError(104, "Peak file does not exist for sample %r: %s" % (sample_id, peak_path))
            rows.append((sample_id, peak_path))
    if not rows:
        raise ValueError("Peak manifest contains no samples: %s" % path)
    return rows


def _sample_task_signature(task: Mapping[str, object]) -> str:
    return _stable_digest({
        "sample_id": task["sample_id"],
        "input_sha256": task["input_sha256"],
        "within_sample_gap": int(task["within_sample_gap"]),
        "max_peak_width": int(task["max_peak_width"]),
        "oversized_peak_policy": str(task["oversized_peak_policy"]),
        "oversized_peak_width": int(task["oversized_peak_width"]),
        "merge_method": PEAK_MERGE_METHOD,
    })


def _count_bed_rows(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                count += 1
    return count


def _validate_sample_checkpoint(task: Mapping[str, object]) -> Optional[Dict[str, object]]:
    checkpoint_path = Path(str(task["checkpoint"]))
    sample_bed = Path(str(task["sample_bed"]))
    if not checkpoint_path.is_file() or not sample_bed.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("schema") != PEAK_MERGE_SAMPLE_SCHEMA:
            return None
        if checkpoint.get("complete") is not True:
            return None
        if version_stale_reason(checkpoint) is not None:
            return None
        if checkpoint.get("sample_id") != task.get("sample_id"):
            return None
        if checkpoint.get("task_signature") != _sample_task_signature(task):
            return None
        if checkpoint.get("sample_bed") != str(sample_bed):
            return None
        if checkpoint.get("sample_bed_sha256") != _sha256_file(sample_bed):
            return None
        if int(checkpoint.get("merged_peaks", -1)) != _count_bed_rows(sample_bed):
            return None
        return checkpoint
    except Exception:
        return None


def _write_sample_bed(path: Path, sample_id: str, merged: Sequence[MergedPeak]) -> None:
    with _atomic_text_writer(path) as handle:
        for idx, region in enumerate(merged, start=1):
            name = "%s_peak_%06d" % (sample_id, idx)
            handle.write("%s\t%d\t%d\t%s\t%d\n" % (
                region.chrom, region.start, region.end, name, len(region.components)
            ))


def _run_sample_task_unlocked(task: Mapping[str, object], *, overwrite: bool = False) -> Dict[str, object]:
    if not overwrite:
        cached = _validate_sample_checkpoint(task)
        if cached is not None:
            return {**cached, "reused": True}

    sample_id = str(task["sample_id"])
    peak_path = Path(str(task["peak_path"]))
    sample_bed = Path(str(task["sample_bed"]))
    checkpoint_path = Path(str(task["checkpoint"]))
    task_signature = _sample_task_signature(task)
    try:
        source_peaks = read_narrowpeak(peak_path, sample_id)
        normalized, changed = normalize_oversized_peaks(
            source_peaks,
            max_peak_width=int(task["max_peak_width"]),
            oversized_peak_policy=str(task["oversized_peak_policy"]),
            oversized_peak_width=int(task["oversized_peak_width"]),
        )
        merged = merge_intervals(
            normalized,
            int(task["within_sample_gap"]),
            int(task["max_peak_width"]),
        )
        _write_sample_bed(sample_bed, sample_id, merged)
        result: Dict[str, object] = {
            "schema": PEAK_MERGE_SAMPLE_SCHEMA,
            "complete": True,
            "sample_id": sample_id,
            "task_id": int(task["task_id"]),
            "task_signature": task_signature,
            "input_peak": str(peak_path),
            "input_sha256": str(task["input_sha256"]),
            "source_peaks": len(source_peaks),
            "normalized_oversized_peaks": changed,
            "merged_peaks": len(merged),
            "sample_bed": str(sample_bed),
            "sample_bed_sha256": _sha256_file(sample_bed),
            "worker_pid": os.getpid(),
            "reused": False,
        }
        _atomic_json(checkpoint_path, result)
        return result
    except Exception as exc:
        _atomic_json(checkpoint_path, {
            "schema": PEAK_MERGE_SAMPLE_SCHEMA,
            "complete": False,
            "sample_id": sample_id,
            "task_id": int(task["task_id"]),
            "task_signature": task_signature,
            "input_peak": str(peak_path),
            "input_sha256": str(task["input_sha256"]),
            "error": "%s: %s" % (type(exc).__name__, exc),
            "worker_pid": os.getpid(),
        })
        raise


def _run_sample_task_payload(task: Mapping[str, object], overwrite: bool = False) -> Dict[str, object]:
    checkpoint_path = Path(str(task["checkpoint"]))
    with exclusive_checkpoint_lock(checkpoint_path):
        return _run_sample_task_unlocked(task, overwrite=overwrite)


def _build_peak_merge_plan(
    peak_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    within_sample_gap: int,
    across_sample_gap: int,
    max_peak_width: int,
    oversized_peak_policy: str,
    oversized_peak_width: int,
    backend: str,
    threads: int,
    jobs: int,
) -> Tuple[Path, Dict[str, object]]:
    if within_sample_gap < 0 or across_sample_gap < 0:
        raise ValueError("Peak merge gaps must be >= 0")
    if max_peak_width < 1:
        raise ValueError("max_peak_width must be >= 1")
    if oversized_peak_policy.strip().lower() not in {"center", "error"}:
        raise ValueError("oversized_peak_policy must be 'center' or 'error'")
    if oversized_peak_width < 1 or oversized_peak_width > max_peak_width:
        raise ValueError("oversized_peak_width must be between 1 and max_peak_width")

    manifest_path = Path(peak_manifest).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    for directory in (out, out / "inputs", out / "per_sample", out / "checkpoints", out / "scripts", out / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    sample_rows = _read_peak_manifest(manifest_path)
    resource_plan = plan_sample_parallel(
        total_threads=threads,
        sample_count=len(sample_rows),
        max_threads_per_sample=1,
        jobs=jobs,
    )
    tasks: List[Dict[str, object]] = []
    tokens: Set[str] = set()
    for task_id, (sample_id, peak_path) in enumerate(sample_rows, start=1):
        token = _sample_file_token(sample_id)
        if token in tokens:
            token = f"{token}_{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:10]}"
        tokens.add(token)
        task = {
            "task_id": task_id,
            "sample_id": sample_id,
            "sample_token": token,
            "peak_path": str(peak_path),
            "input_sha256": _sha256_file(peak_path),
            "sample_bed": str(out / "per_sample" / f"{token}.merged_peaks.bed"),
            "checkpoint": str(stage_checkpoint_path(out, f"sample_{token}")),
            "within_sample_gap": int(within_sample_gap),
            "max_peak_width": int(max_peak_width),
            "oversized_peak_policy": oversized_peak_policy.strip().lower(),
            "oversized_peak_width": int(oversized_peak_width),
        }
        task["task_signature"] = _sample_task_signature(task)
        tasks.append(task)

    identity = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "samples": [{"sample_id": task["sample_id"], "task_signature": task["task_signature"]} for task in tasks],
        "within_sample_gap": int(within_sample_gap),
        "across_sample_gap": int(across_sample_gap),
        "max_peak_width": int(max_peak_width),
        "oversized_peak_policy": oversized_peak_policy.strip().lower(),
        "oversized_peak_width": int(oversized_peak_width),
        "merge_method": PEAK_MERGE_METHOD,
    }
    plan: Dict[str, object] = {
        "schema": PEAK_MERGE_PLAN_SCHEMA,
        "stage": "peak_merge",
        "plan_id": _stable_digest(identity),
        "backend": backend,
        "peak_manifest": str(manifest_path),
        "output_dir": str(out),
        "input_identity": identity,
        "tasks": tasks,
        "within_sample_gap": int(within_sample_gap),
        "across_sample_gap": int(across_sample_gap),
        "max_peak_width": int(max_peak_width),
        "oversized_peak_policy": oversized_peak_policy.strip().lower(),
        "oversized_peak_width": int(oversized_peak_width),
        "threads": int(threads),
        "jobs": int(jobs),
        "resource_plan": resource_plan.to_dict(),
        "outputs": {
            "cohort_bed": str(out / "cohort_peaks.bed"),
            "metadata": str(out / "cohort_peaks.csv"),
            "support": str(out / "cohort_peak_support.csv"),
            "checkpoint": str(stage_checkpoint_path(out, "peak_merge")),
        },
    }
    plan_path = out / "inputs" / "peak_merge_plan.json"
    _atomic_json(plan_path, plan)
    return plan_path, plan


def _read_sample_bed(task: Mapping[str, object], *, max_peak_width: int) -> List[Peak]:
    status = _validate_sample_checkpoint(task)
    if status is None:
        raise RuntimeError("Peak-merge sample task failed validation: %s" % task["sample_id"])
    sample_id = str(task["sample_id"])
    sample_bed = Path(str(task["sample_bed"]))
    peaks: List[Peak] = []
    previous: Optional[Tuple[str, int, int, str]] = None
    seen: Set[Tuple[str, int, int, str]] = set()
    with sample_bed.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise RuntimeError("%s:%d has fewer than 4 columns" % (sample_bed, line_no))
            chrom, start_text, end_text, source_id = fields[:4]
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start or end - start > max_peak_width:
                raise RuntimeError("Invalid per-sample merged peak %s:%d-%d" % (chrom, start, end))
            key = (chrom, start, end, source_id)
            if key in seen:
                raise RuntimeError("Duplicate per-sample merged peak: %r" % (key,))
            if previous is not None and key < previous:
                raise RuntimeError("Per-sample merged peaks are not deterministically ordered: %s" % sample_bed)
            previous = key
            seen.add(key)
            score = float(fields[4]) if len(fields) > 4 and fields[4] else 0.0
            peaks.append(Peak(chrom, start, end, sample_id, source_id, score, None))
    if len(peaks) != int(status["merged_peaks"]):
        raise RuntimeError("Per-sample merged peak count changed after checkpoint validation: %s" % sample_id)
    return peaks


def _validate_atlas(atlas: Sequence[MergedPeak], max_peak_width: int) -> None:
    if not atlas:
        raise RuntimeError("Cohort peak atlas contains no peaks")
    previous: Optional[Tuple[str, int, int]] = None
    seen: Set[Tuple[str, int, int]] = set()
    for region in atlas:
        key = (region.chrom, region.start, region.end)
        if region.start < 0 or region.end <= region.start:
            raise RuntimeError("Cohort peak atlas contains an invalid interval: %r" % (key,))
        if region.end - region.start > max_peak_width:
            raise RuntimeError("Cohort peak atlas contains an interval wider than max_peak_width")
        if key in seen:
            raise RuntimeError("Cohort peak atlas contains duplicate intervals: %r" % (key,))
        if previous is not None and key < previous:
            raise RuntimeError("Cohort peak atlas is not deterministically ordered")
        # Overlapping atlas peaks would count a fragment twice.
        if previous is not None and previous[0] == region.chrom and region.start < previous[2]:
            raise RuntimeError("Cohort peak atlas contains overlapping intervals: %r and %r" % (previous, key))
        previous = key
        seen.add(key)
        if not region.samples or not region.components:
            raise RuntimeError("Cohort peak atlas contains missing support metadata")


def _mark_stage_incomplete(plan: Mapping[str, object], pending_task_ids: Sequence[int]) -> None:
    checkpoint_path = Path(str(plan["outputs"]["checkpoint"]))
    payload = {
        "schema": PEAK_MERGE_STAGE_SCHEMA,
        "stage": "peak_merge",
        "complete": False,
        "plan_id": plan["plan_id"],
        "backend": plan["backend"],
        "samples": len(plan["tasks"]),
        "completed_samples": len(plan["tasks"]) - len(pending_task_ids),
        "pending_task_ids": [int(task_id) for task_id in pending_task_ids],
        "resource_plan": plan["resource_plan"],
        "input_identity": plan["input_identity"],
    }
    with exclusive_checkpoint_lock(checkpoint_path):
        _atomic_json(checkpoint_path, payload)


def _stage_checkpoint_current(plan: Mapping[str, object]) -> bool:
    """True when the stage checkpoint is complete for this very plan."""
    try:
        existing = json.loads(Path(str(plan["outputs"]["checkpoint"])).read_text())
    except (OSError, ValueError):
        return False
    return existing.get("complete") is True and existing.get("plan_id") == plan["plan_id"]


def finalize_peak_merge_plan(plan_path: Union[str, Path]) -> Dict[str, object]:
    plan_path = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PEAK_MERGE_PLAN_SCHEMA:
        raise ValueError("Unsupported peak-merge plan schema: %r" % plan.get("schema"))
    tasks = list(plan["tasks"])
    max_peak_width = int(plan["max_peak_width"])
    sample_merged: List[Peak] = []
    statuses: List[Dict[str, object]] = []
    for task in tasks:
        status = _validate_sample_checkpoint(task)
        if status is None:
            raise RuntimeError("Peak-merge sample task failed validation: %s" % task["sample_id"])
        statuses.append(status)
        sample_merged.extend(_read_sample_bed(task, max_peak_width=max_peak_width))

    atlas = merge_intervals(sample_merged, int(plan["across_sample_gap"]), max_peak_width)
    _validate_atlas(atlas, max_peak_width)
    outputs = dict(plan["outputs"])
    bed_path = Path(str(outputs["cohort_bed"]))
    metadata_path = Path(str(outputs["metadata"]))
    support_path = Path(str(outputs["support"]))

    with _atomic_text_writer(bed_path) as bed, _atomic_text_writer(metadata_path, newline="") as meta, _atomic_text_writer(support_path, newline="") as support:
        meta_writer = csv.writer(meta, delimiter=",", lineterminator="\n")
        support_writer = csv.writer(support, delimiter=",", lineterminator="\n")
        meta_writer.writerow(["peak_id", "chrom", "start", "end", "width", "sample_support", "component_count", "samples"])
        support_writer.writerow(["peak_id", "sample_id"])
        for idx, region in enumerate(atlas, start=1):
            # chr:start-end, so a peak_id can be pasted into a genome browser.
            # _validate_atlas has checked that regions do not overlap, so the
            # id is unique across the atlas.
            peak_id = "%s:%d-%d" % (region.chrom, region.start, region.end)
            samples = sorted(region.samples)
            bed.write("%s\t%d\t%d\t%s\n" % (region.chrom, region.start, region.end, peak_id))
            meta_writer.writerow([
                peak_id, region.chrom, region.start, region.end, region.end - region.start,
                len(samples), len(region.components), ",".join(samples),
            ])
            for sample_id in samples:
                support_writer.writerow([peak_id, sample_id])

    output_hashes = {
        "cohort_bed_sha256": _sha256_file(bed_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "support_sha256": _sha256_file(support_path),
    }
    result: Dict[str, object] = {
        "schema": PEAK_MERGE_STAGE_SCHEMA,
        "stage": "peak_merge",
        "complete": True,
        "plan_id": plan["plan_id"],
        "backend": plan["backend"],
        "samples": len(tasks),
        "completed_samples": len(statuses),
        "source_peaks": sum(int(status["source_peaks"]) for status in statuses),
        "normalized_oversized_peaks": sum(int(status["normalized_oversized_peaks"]) for status in statuses),
        "sample_merged_peaks": sum(int(status["merged_peaks"]) for status in statuses),
        "cohort_peaks": len(atlas),
        "cohort_bed": str(bed_path),
        "metadata": str(metadata_path),
        "support": str(support_path),
        "within_sample_gap": int(plan["within_sample_gap"]),
        "across_sample_gap": int(plan["across_sample_gap"]),
        "max_peak_width": max_peak_width,
        "oversized_peak_policy": str(plan["oversized_peak_policy"]),
        "oversized_peak_width": int(plan["oversized_peak_width"]),
        "threads": int(plan["threads"]),
        "jobs": int(plan["jobs"]),
        "resource_plan": plan["resource_plan"],
        "input_identity": plan["input_identity"],
        "sample_checkpoints": [str(task["checkpoint"]) for task in tasks],
        **output_hashes,
    }
    checkpoint_path = Path(str(outputs["checkpoint"]))
    _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    result["plan"] = str(plan_path)
    return result


def run_peak_merge_sample_task(plan_path: Union[str, Path], task_id: int, *, overwrite: bool = False) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).expanduser().resolve().read_text())
    tasks = list(plan["tasks"])
    if task_id < 1 or task_id > len(tasks):
        raise ValueError("task_id must be between 1 and %d" % len(tasks))
    pending = _pending_task_ids(plan, overwrite=False)
    if overwrite and task_id not in pending:
        pending.append(task_id)
    _mark_stage_incomplete(plan, pending or [task_id])
    return _run_sample_task_payload(tasks[task_id - 1], overwrite=overwrite)


def run_peak_merge_finalize_task(plan_path: Union[str, Path]) -> Dict[str, object]:
    return finalize_peak_merge_plan(plan_path)




def _pending_task_ids(plan: Mapping[str, object], *, overwrite: bool) -> List[int]:
    if overwrite:
        return [int(task["task_id"]) for task in plan["tasks"]]
    return [
        int(task["task_id"])
        for task in plan["tasks"]
        if _validate_sample_checkpoint(task) is None
    ]


def prepare_peak_merge_slurm_plan(
    peak_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    within_sample_gap: int = 0,
    across_sample_gap: int = 0,
    max_peak_width: int = 5000,
    oversized_peak_policy: str = "center",
    oversized_peak_width: int = 500,
    threads: int = 1,
    jobs: int = 0,
    partition: Optional[str] = None,
    mem: str = "4G",
    overwrite: bool = False,
) -> Dict[str, object]:
    plan_path, plan = _build_peak_merge_plan(
        peak_manifest,
        output_dir,
        within_sample_gap=within_sample_gap,
        across_sample_gap=across_sample_gap,
        max_peak_width=max_peak_width,
        oversized_peak_policy=oversized_peak_policy,
        oversized_peak_width=oversized_peak_width,
        backend="slurm",
        threads=threads,
        jobs=jobs,
    )
    pending = _pending_task_ids(plan, overwrite=overwrite)
    # Also when only the plan changed (e.g. --across-sample-gap) and
    # no sample task is pending, so the old result never reads complete.
    if pending or overwrite or not _stage_checkpoint_current(plan):
        _mark_stage_incomplete(plan, pending)
    capabilities = detect_scheduler_capabilities("slurm")
    require_array_task_ids(pending, capabilities, stage="Peak merge")
    concurrency = int(plan["resource_plan"]["concurrent_jobs"])
    array_plan = plan_array_tasks(
        task_count=len(pending), concurrency=concurrency, capabilities=capabilities,
    )
    plan["pending_task_ids"] = pending
    plan["scheduler_capabilities"] = capabilities.to_dict()
    plan["array_plan"] = array_plan.to_dict()
    _atomic_json(plan_path, plan)
    executable_argv = worker_argv()
    cli = shell_join(executable_argv)
    out = Path(str(plan["output_dir"]))
    array_scripts: List[Path] = []
    for batch in array_plan.batches:
        task_ids = pending[batch.first_task - 1: batch.last_task]
        array_spec = compress_array_ids(task_ids, throttle=min(batch.throttle, len(task_ids)))
        batch_index = batch.batch_index
        overwrite_flag = " --overwrite" if overwrite else ""
        array_scripts.append(write_sbatch(
            out / "scripts" / f"peak_merge_samples_{batch_index:03d}.sbatch",
            SlurmOptions(
                job_name=slurm_job_name(out, f"peak_merge_{batch_index:03d}"),
                cpus_per_task=1,
                mem=mem,
                partition=partition,
                array=array_spec,
                output=out / "logs" / f"peak_merge_{batch_index:03d}_%A_%a.out",
                error=out / "logs" / f"peak_merge_{batch_index:03d}_%A_%a.err",
            ),
            f'{cli} atac peak-merge-sample --plan {shlex.quote(str(plan_path))} '
            f'--task-id "$SLURM_ARRAY_TASK_ID"{overwrite_flag}',
        ))
    finalize_script = write_sbatch(
        out / "scripts" / "peak_merge_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(out, "peak_merge_finalize"),
            cpus_per_task=1,
            mem=mem,
            partition=partition,
            output=out / "logs" / "peak_merge_finalize_%j.out",
            error=out / "logs" / "peak_merge_finalize_%j.err",
        ),
        f'{cli} atac peak-merge-finalize --plan {shlex.quote(str(plan_path))}',
    )
    return {
        "plan": plan_path,
        "array_scripts": array_scripts,
        "finalize_script": finalize_script,
        "task_count": len(pending),
        "total_samples": len(plan["tasks"]),
        "reused_samples": len(plan["tasks"]) - len(pending),
        "concurrency": concurrency,
        "resource_plan": plan["resource_plan"],
        "scheduler_capabilities": capabilities.to_dict(),
        "array_plan": array_plan.to_dict(),
        "plan_id": plan["plan_id"],
    }


def submit_peak_merge_slurm_plan(plan_info: Mapping[str, object], *, submit: bool) -> Dict[str, object]:
    result = dict(plan_info)
    result.update({"submitted": False, "job_id": None, "submitted_job_ids": [], "complete": False})
    if not submit:
        return result
    sample_job_ids: List[str] = []
    dependency: Optional[str] = None
    # Each job ID is reported the moment its sbatch returns, so a refused later sbatch never leaves
    # queued jobs unnamed. stderr, which the supervisor reads as well, so the JSON on stdout stays whole.
    for script in plan_info["array_scripts"]:
        job_id = sbatch(script, dependency=dependency)
        print(f"Submitted peak-merge array job: {job_id}", file=sys.stderr, flush=True)
        sample_job_ids.append(job_id)
        # Array batches are chained so their independent throttles cannot exceed
        # the stage-wide --threads/--jobs concurrency contract in aggregate.
        dependency = afterok(job_id)
    finalize_job_id = sbatch(plan_info["finalize_script"], dependency=dependency)
    print(f"Submitted peak-merge finalizer job: {finalize_job_id}", file=sys.stderr, flush=True)
    job_ids = [*sample_job_ids, finalize_job_id]
    result.update({
        "submitted": True,
        "job_id": ",".join(job_ids),
        "submitted_job_ids": job_ids,
        "sample_job_ids": sample_job_ids,
        "finalize_job_id": finalize_job_id,
    })
    return result


def run_peak_merge(
    peak_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    within_sample_gap: int = 0,
    across_sample_gap: int = 0,
    max_peak_width: Optional[int] = 5000,
    oversized_peak_policy: str = "center",
    oversized_peak_width: int = 500,
    *,
    threads: int = 1,
    jobs: int = 0,
    overwrite: bool = False,
) -> Dict[str, object]:
    if max_peak_width is None:
        raise ValueError("max_peak_width must be set for ATAC cohort peak construction")
    plan_path, plan = _build_peak_merge_plan(
        peak_manifest,
        output_dir,
        within_sample_gap=within_sample_gap,
        across_sample_gap=across_sample_gap,
        max_peak_width=max_peak_width,
        oversized_peak_policy=oversized_peak_policy,
        oversized_peak_width=oversized_peak_width,
        backend="local",
        threads=threads,
        jobs=jobs,
    )
    tasks = list(plan["tasks"])
    pending = [task for task in tasks if overwrite or _validate_sample_checkpoint(task) is None]
    # See prepare_peak_merge_slurm_plan.
    if pending or overwrite or not _stage_checkpoint_current(plan):
        _mark_stage_incomplete(plan, [int(task["task_id"]) for task in pending])
    statuses: List[Dict[str, object]] = []
    resource_plan = SampleParallelPlan(**plan["resource_plan"])
    if len(pending) <= 1 or resource_plan.concurrent_jobs <= 1:
        for task in pending:
            statuses.append(_run_sample_task_payload(task, overwrite=overwrite))
    else:
        with ProcessPoolExecutor(
            max_workers=min(resource_plan.concurrent_jobs, len(pending)),
        ) as executor:
            future_to_task = {
                executor.submit(_run_sample_task_payload, task, overwrite): task
                for task in pending
            }
            try:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        statuses.append(future.result())
                    except Exception as exc:
                        raise RuntimeError("Peak merge failed for sample %s: %s" % (task["sample_id"], exc)) from exc
            except Exception:
                for future in future_to_task:
                    future.cancel()
                raise

    result = finalize_peak_merge_plan(plan_path)
    result.update({
        "executed_samples": len(pending),
        "reused_samples": len(tasks) - len(pending),
        "sample_task_results": sorted(statuses, key=lambda item: int(item["task_id"])),
        "concurrency": resource_plan.concurrent_jobs,
        "peak_cpus": resource_plan.peak_cpus,
    })
    return result
