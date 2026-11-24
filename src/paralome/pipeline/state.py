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
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from paralome.checkpoints import (
    count_table_rows,
    file_info as _file_info,
    legacy_stage_checkpoint_path,
    output_sizes_agree,
    stage_checkpoint_path,
    vcf_with_index_ready,
)

from .jsonio import version_stale_reason, write_json
from .profiles import blocks, resolve_pipeline_profile, split_read, stage_ancestors, stage_reads


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: str | Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}




def _prefixed_fallback(results: Path, name: str, expected: str | None = None) -> str:
    """A results path for ``name``, resolving ONLY the project-ID-prefixed form.

    The bare (unprefixed) name is never resolved: a workdir that holds only
    ``segregation_pass.tsv`` reads as incomplete, so its stage re-runs and
    writes the prefixed names. Segregation, filter and report are cheap to
    re-run; nothing is deleted.

    A file with some other prefix is not accepted either: the reader asks
    "is this file named ``<FAM01>_<name>``", not "does it have a prefix",
    because files with a wrong prefix can be present beside the right ones.

    ``expected`` is the prefix this project is supposed to be using, derived
    from the workdir the same way the writers derive it. When it is known,
    only that exact name resolves. The glob remains as a fallback for callers
    that genuinely cannot know the prefix.
    """
    if expected:
        exact = results / f"{expected}_{name}"
        if exact.exists():
            return str(exact)
        # Deliberately NOT falling back to the glob here. A tree that has an
        # `X_pass.tsv` but no `FAM01_pass.tsv` is a half-migrated tree, and
        # X must not be treated as good enough.
        return str(results / f".unprefixed.{name}")
    matches = sorted(results.glob(f"*_{name}")) if results.exists() else []
    if matches:
        return str(matches[0])
    # A path that CANNOT exist. Returning ``results / name`` here would resolve
    # the bare file whenever one is present, which is the exact behaviour being
    # removed -- the stage would read complete and never re-run.
    return str(results / f".unprefixed.{name}")


def _expected_prefix(workdir: Path) -> str | None:
    """The prefix this workdir's outputs should carry, or None if underivable.

    Uses the same helper the writers use, so the reader and the
    writer cannot disagree about what the family is called -- which they did,
    silently, for five releases.
    """
    try:
        from paralome.atac.naming import derive_file_prefix

        prefix = derive_file_prefix(workdir)
    except Exception:
        return None
    return prefix if prefix and prefix != "output" else None


def _reject_if_bare(path: str | None, bare_name: str, expected: str | None = None) -> str | None:
    """Drop a recorded output path that is not named for this project.

    The checkpoint records the paths a stage wrote, and ``summarize_*``
    trusts them before falling back to a glob, so a recorded path that is not
    named for this project (bare, ``results/segregation_pass.tsv``, or carrying
    another prefix) must be refused here as well as by the glob, or a stage
    with wrongly named outputs would read complete.
    """
    if not path:
        return path
    actual = Path(str(path)).name
    if actual == bare_name:
        return None
    if expected and actual != f"{expected}_{bare_name}":
        return None
    return path


def _in_this_copy(wd: Path, recorded: Any) -> Any:
    """What a checkpoint recorded, with its paths as they are in the folder being read.

    Stage checkpoints hold the absolute paths of their outputs. In a project
    that was copied, or that is reached under another mount point, those name
    the files of the other tree. A recorded path that is not in ``wd`` but ran
    through a folder of the same name is read at the same place under ``wd``
    when the file is there. Anything else is returned as recorded.
    """
    folder = os.path.abspath(wd)
    name = os.path.basename(folder)

    def read(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: read(item) for key, item in value.items()}
        if isinstance(value, list):
            return [read(item) for item in value]
        if not isinstance(value, str) or not value.startswith(os.sep) or value.startswith(folder + os.sep):
            return value
        parts = Path(value).parts
        if name not in parts[:-1]:
            return value
        try:
            if (os.path.realpath(value) + os.sep).startswith(os.path.realpath(folder) + os.sep):
                return value
            here = os.path.join(folder, *parts[max(at for at, part in enumerate(parts[:-1]) if part == name) + 1:])
            return here if os.path.exists(here) else value
        except (OSError, ValueError):
            return value

    return read(recorded)


def _csv_or_tsv(wd: Path, *parts: str) -> Path:
    """Machine outputs are CSV; fall back to the .tsv path of a run completed before CSV."""
    csv_path = wd.joinpath(*parts[:-1], parts[-1].replace(".tsv", ".csv"))
    if csv_path.exists():
        return csv_path
    tsv_path = wd.joinpath(*parts)
    if tsv_path.exists():
        return tsv_path
    return csv_path


def _resolve_manifest_artifact(manifest: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = [(Path.cwd() / candidate).resolve()]
    candidates.extend((parent / candidate).resolve() for parent in manifest.parents)
    return next((path for path in candidates if path.is_file()), candidates[0])


def _safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _glob_first(patterns: list[str]) -> Optional[Path]:
    import glob

    for pattern in patterns:
        matches = sorted(Path(x) for x in glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def summarize_fastq_qc(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    summary_path = wd / "results" / "fastq_qc_summary.json"
    trimmed_manifest = wd / "inputs" / "trimmed_samples.csv"
    summary = _read_json(summary_path)
    totals = summary.get("totals", {}) if isinstance(summary.get("totals", {}), dict) else {}
    complete = _file_info(summary_path)["nonempty"] and _file_info(trimmed_manifest)["nonempty"]
    # The samples the summary has a report for; None when it does
    # not list them. Not the trimmed manifest: the SLURM stage writes that for
    # every sample of the sheet before any of them is trimmed.
    listed = summary.get("samples")
    done_sample_ids = (
        [str(item.get("sample_id")) for item in listed if isinstance(item, dict) and item.get("sample_id")]
        if isinstance(listed, list) else None
    )
    # And every sample with a fastp report: the report is written when fastp finishes, so a summary
    # written while the array's reports were still arriving (or read short-handed over NFS) can list fewer
    # samples than were trimmed, and must not make the later stages read out of date.
    reports = wd / "fastp_json"
    if done_sample_ids is not None and reports.is_dir():
        reported = sorted(path.stem for path in reports.glob("*.json") if path.stat().st_size)
        done_sample_ids = list(dict.fromkeys([*done_sample_ids, *reported]))
    # A sample `fastq qc --overwrite` marked and no task has trimmed
    # again still has its earlier reads in place. The stage is not finished
    # while there is one, whatever summary is there.
    from paralome.fastq.summary import replace_mark

    marks = replace_mark(wd, "sample").parent
    to_be_replaced = sorted(path.name for path in marks.iterdir()) if marks.is_dir() else []
    if to_be_replaced:
        complete = False
        if done_sample_ids is not None:
            done_sample_ids = [name for name in done_sample_ids if name not in to_be_replaced]
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "summary": _file_info(summary_path),
        "trimmed_manifest": _file_info(trimmed_manifest),
        "done_sample_ids": done_sample_ids,
        "to_be_replaced": to_be_replaced,
        "samples": totals.get("samples"),
        "raw_reads": totals.get("raw_reads"),
        "clean_reads": totals.get("clean_reads"),
        "surviving_percent": (
            totals.get("surviving_percent")
            or totals.get("pct_surviving")
            or totals.get("pct_reads_surviving")
        ),
    }



def summarize_alignment(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint_path = stage_checkpoint_path(wd, "alignment")
    metrics_path = wd / "results" / "alignment_metrics.json"
    if not metrics_path.is_file():
        metrics_path = wd / "results" / "star_alignment_metrics.json"   # an older project's name
    verify_path = wd / "results" / "star_alignment_verify.json"
    manifest_path = wd / "inputs" / "aligned_samples.csv"

    checkpoint = _read_json(checkpoint_path)
    metrics = _read_json(metrics_path)
    verify = _read_json(verify_path)

    expected = checkpoint.get("expected") or verify.get("samples") or metrics.get("samples")
    completed = checkpoint.get("completed_samples") or verify.get("complete_samples")

    manifest_ok = _file_info(manifest_path)["nonempty"]
    # WHEN THERE IS A VERIFICATION REPORT, IT DECIDES. This was
    # "the report or the checkpoint", so a report that found a BAM missing or
    # unusable was overruled by the checkpoint the array task had already
    # marked complete. The checkpoint speaks only where nothing was verified.
    #
    # UNLESS THE REPORT SAYS IT COULD NOT CHECK. Where samtools
    # cannot be run (a login node without it) `align verify` accepts no BAM and
    # says so with `could_not_check`; that is no verdict, and taken as one a
    # good alignment could never read complete there. Such a report still
    # says which files are there, so the stage is judged on those: every
    # sample either accepted, or left unchecked with its BAM and index in
    # place. A sample the report did check and did not accept stays refused.
    # A report that lists no samples leaves it to the checkpoint.
    listed = verify.get("sample_status") if verify else None
    could_not_check = bool(verify.get("could_not_check"))

    def in_place(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if item.get("complete"):
            return True
        files = [item.get("bam"), item.get("bai")]
        return item.get("checked") is False and all(isinstance(f, dict) and f.get("nonempty") for f in files)

    if could_not_check and isinstance(listed, list) and listed:
        complete = all(in_place(item) for item in listed)
    elif could_not_check:
        complete = bool(checkpoint.get("complete"))
    else:
        complete = bool(verify.get("complete")) if verify else bool(checkpoint.get("complete"))
    if complete and not manifest_ok:
        complete = False
    # NOR WHILE A SAMPLE IS STILL TO BE ALIGNED AGAIN. `align star --overwrite` marks every sample in the
    # checkpoint and a sample's own task takes its mark off. Until then the BAM in place is the earlier one.
    from paralome.align.verify import TO_BE_REPLACED

    recorded_samples = checkpoint.get("samples")
    to_be_replaced = sorted(
        str(sample_id) for sample_id, entry in recorded_samples.items()
        if isinstance(entry, dict) and isinstance(entry.get("metadata"), dict) and entry["metadata"].get(TO_BE_REPLACED)
    ) if isinstance(recorded_samples, dict) else []
    if to_be_replaced:
        complete = False
    # A report that says incomplete beside a checkpoint that says complete (a supervisor restarted during
    # a retry, `align verify` run by hand while the array was running). The supervisor verifies once more
    # before it believes either -- see supervisor._verify_an_alignment_in_doubt.
    report_disagrees = bool(
        verify and not could_not_check and not verify.get("complete")
        and checkpoint.get("complete") and manifest_ok and not to_be_replaced
    )
    # The samples known to be aligned, for build_project_state to hold against
    # the sample sheet; None when neither file names them.
    if verify:
        done_sample_ids = (
            [str(item.get("sample_id")) for item in listed
             if isinstance(item, dict) and (item.get("complete") or (could_not_check and in_place(item)))]
            if isinstance(listed, list) else None
        )
    else:
        listed = checkpoint.get("complete_sample_ids")
        done_sample_ids = [str(name) for name in listed] if isinstance(listed, list) else None
    if done_sample_ids is not None:
        done_sample_ids = [name for name in done_sample_ids if name not in to_be_replaced]

    totals = metrics.get("totals", {}) if isinstance(metrics.get("totals", {}), dict) else {}
    resources = checkpoint.get("resources") or verify.get("resources") or {}

    if complete:
        status = "complete"
    elif expected and completed not in (None, 0):
        status = "partial"
    else:
        status = "incomplete"

    return {
        "status": status,
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint_path),
        "verify": _file_info(verify_path),
        "metrics": _file_info(metrics_path),
        "aligned_manifest": _file_info(manifest_path),
        "expected_samples": expected,
        "completed_samples": completed,
        "missing_sample_ids": checkpoint.get("missing_sample_ids") or verify.get("missing_sample_ids") or [],
        "done_sample_ids": done_sample_ids,
        "to_be_replaced": to_be_replaced,
        "could_not_check": could_not_check,
        "report_disagrees_with_checkpoint": report_disagrees,
        "resources": resources,
        "input_reads": totals.get("input_reads"),
        "uniquely_mapped_reads": totals.get("uniquely_mapped_reads"),
        "weighted_uniquely_mapped_percent": totals.get("weighted_uniquely_mapped_percent"),
        "total_mapped_reads": totals.get("total_mapped_reads"),
        "weighted_total_mapped_percent": totals.get("weighted_total_mapped_percent"),
        # The per-sample rows and the settings that produced them.
        #
        # The cohort totals hide whether a low unique-mapping rate is uniform
        # across samples (systematic) or confined to one library; the
        # per-sample rows show it.
        "samples_detail": metrics.get("samples_detail") or [],
        # The stage writes these under "parameters", in results/
        # star_alignment_summary.json and again into the checkpoint. Read from
        # the summary first because the SLURM path writes it at submission,
        # before any array task has run.
        "star_settings": (
            _read_json(wd / "results" / "star_alignment_summary.json").get("parameters")
            or (checkpoint.get("alignment", {}) or {}).get("star", {}).get("parameters")
            or checkpoint.get("parameters")
            or {}
        ),
    }


def summarize_dedup(workdir: str | Optional[Path], manifest_profile: str | None = None) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint_path = stage_checkpoint_path(wd, "deduplication")
    summary_path = wd / "results" / "dedup_summary.json"
    manifest_path = wd / "inputs" / "dedup_samples.csv"

    ckpt = _read_json(checkpoint_path)
    summary = _read_json(summary_path)
    manifest = _file_info(manifest_path)

    rows = []
    if manifest.get("nonempty"):
        import csv
        try:
            with Path(manifest_path).open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            rows = []

    manifest_compatible = True
    manifest_issue = None

    completed = 0
    done_ids: list[str] = []
    for row in rows:
        bam = row.get("bam") or row.get("bam_path") or row.get("path")
        if not bam:
            continue
        # Read where the BAM is in this copy of the project (see _in_this_copy): the files decide
        # completion, and the manifest holds absolute paths.
        bam = _in_this_copy(wd, bam)
        bam_info = _file_info(bam)
        bai_info = _file_info(str(bam) + ".bai")
        if bam_info.get("nonempty") and bai_info.get("nonempty"):
            completed += 1
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                done_ids.append(sample_id)

    expected = len(rows) or ckpt.get("expected_samples") or ckpt.get("samples") or summary.get("samples")
    complete_by_files = bool(rows) and completed == len(rows)
    # The files decide, and only the files: the run rewrites the manifest for the samples now in the sheet
    # without touching the earlier run's summary, so a sample added to a finished project must not read
    # as finished on that summary's complete:true. Every row of the manifest must have its BAM and index;
    # a stage with no manifest names no BAMs and is not finished (the manifest is written before any
    # sample is processed).
    complete = manifest_compatible and complete_by_files

    if complete:
        status = "complete"
    elif ckpt.get("submitted") and completed:
        status = "partial"
    elif ckpt.get("submitted"):
        status = "running"
    elif not manifest_compatible:
        status = "manifest_incompatible"
    elif manifest.get("nonempty"):
        status = "incomplete"
    else:
        status = "not_available"

    metrics = summary if summary else (ckpt.get("metrics") if isinstance(ckpt.get("metrics"), dict) else {})

    return {
        "status": status,
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint_path),
        "summary": _file_info(summary_path),
        "dedup_manifest": manifest,
        "manifest_profile": manifest_profile,
        "manifest_compatible": manifest_compatible,
        "manifest_issue": manifest_issue,
        "samples": expected,
        # The samples this stage has done; None without a manifest.
        # Those with their BAM and index on disk, not every row of
        # the manifest. The manifest is rewritten for the whole sheet when a
        # run starts, so it named a sample whose task had not run, and the
        # sheet guard (_hold_per_sample_stages_to_the_sheet) found nothing
        # missing.
        "done_sample_ids": done_ids if rows else None,
        "completed_samples": completed if rows else (ckpt.get("completed_samples") or summary.get("completed_samples")),
        "remove_duplicates": ckpt.get("remove_duplicates") if ckpt else summary.get("remove_duplicates"),
        "duplicate_rate": metrics.get("duplicate_rate") if isinstance(metrics, dict) else None,
        "duplicate_reads": metrics.get("duplicate_reads") if isinstance(metrics, dict) else None,
        "total_reads": metrics.get("total_reads") if isinstance(metrics, dict) else None,
    }

def summarize_macs(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "macs_peak_calling")
    manifest = wd / "results" / "macs_peaks.csv"
    ckpt = _read_json(checkpoint)
    manifest_info = _file_info(manifest)
    rows = 0
    valid_rows = 0
    issues: list[str] = []
    seen: set[str] = set()
    # The MACS3 settings the plan records, compared with each marker's.
    planned_parameters = _read_json(wd / "inputs" / "macs_plan.json").get("parameters")
    if manifest_info["nonempty"]:
        try:
            with manifest.open(newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"sample_id", "narrowpeak", "summits"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    issues.append("missing columns: " + ",".join(sorted(missing)))
                else:
                    for row_no, row in enumerate(reader, start=2):
                        rows += 1
                        sample_id = str(row.get("sample_id") or "").strip()
                        if not sample_id:
                            issues.append(f"row {row_no}: empty sample_id")
                            continue
                        if sample_id in seen:
                            issues.append(f"row {row_no}: duplicate sample_id {sample_id}")
                            continue
                        seen.add(sample_id)
                        row_valid = True
                        for field in ("narrowpeak", "summits"):
                            value = str(row.get(field) or "").strip()
                            candidate = _resolve_manifest_artifact(manifest, value) if value else None
                            if candidate is None or not candidate.is_file() or candidate.stat().st_size <= 0:
                                issues.append(f"row {row_no}: invalid {field}")
                                row_valid = False
                        if str(row.get("status") or "complete").strip().lower() not in {"", "complete"}:
                            issues.append(f"row {row_no}: status is not complete")
                            row_valid = False
                        # Peaks called before MACS3 was given the BAM without
                        # duplicates have no marker beside them; they are called again.
                        peaks_value = str(row.get("narrowpeak") or "").strip()
                        peaks_file = _resolve_manifest_artifact(manifest, peaks_value) if peaks_value else None
                        marker = (peaks_file.parent / f"{sample_id}.paralome_macs.json") if peaks_file is not None else None
                        if row_valid and marker is not None and not marker.is_file():
                            issues.append(f"row {row_no}: peaks called with PCR duplicates (by an older Paralome)")
                            row_valid = False
                        if row_valid and marker is not None:
                            # Peaks must record the MACS3 settings they were called with.
                            recorded = _read_json(marker).get("parameters")
                            if not isinstance(recorded, dict):
                                issues.append(f"row {row_no}: peaks record no MACS3 parameters (by an older Paralome)")
                                row_valid = False
                            elif isinstance(planned_parameters, dict) and recorded != planned_parameters:
                                issues.append(f"row {row_no}: peaks called with other MACS3 parameters than the plan records")
                                row_valid = False
                        if row_valid:
                            valid_rows += 1
        except Exception as exc:
            issues.append(f"manifest unreadable: {exc}")
    expected = ckpt.get("samples")
    completed = ckpt.get("completed_samples")
    try:
        expected_matches = expected in (None, "") or int(expected) == rows
    except (TypeError, ValueError):
        expected_matches = False
        issues.append("checkpoint samples value is invalid")
    try:
        completed_matches = completed in (None, "") or int(completed) == rows
    except (TypeError, ValueError):
        completed_matches = False
        issues.append("checkpoint completed_samples value is invalid")
    complete = (
        bool(ckpt.get("complete"))
        and manifest_info["nonempty"]
        and rows > 0
        and valid_rows == rows
        and expected_matches
        and completed_matches
        and not issues
    )
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "peak_manifest": manifest_info,
        "samples": expected,
        "completed_samples": ckpt.get("completed_samples"),
        "manifest_rows": rows,
        "validated_peak_files": valid_rows * 2,
        "validation_issues": issues,
    }

def summarize_peak_merge(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "peak_merge")
    bed = wd / "cohort_peaks.bed"
    ckpt = _read_json(checkpoint)
    complete = bool(ckpt.get("complete")) and _file_info(bed)["nonempty"]
    return {"status": "complete" if complete else "incomplete", "complete": complete, "workdir": str(wd), "checkpoint": _file_info(checkpoint), "cohort_peaks": _file_info(bed), "peaks": ckpt.get("cohort_peaks")}


def summarize_peak_quantification(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "peak_quantification")
    ckpt = _read_json(checkpoint)
    raw = _csv_or_tsv(wd, "results", "peak_raw_counts.tsv")
    cpm = _csv_or_tsv(wd, "results", "peak_cpm.tsv")
    log2_cpm = _csv_or_tsv(wd, "results", "peak_log2_cpm.tsv")
    sample_qc = _csv_or_tsv(wd, "results", "sample_qc.tsv")
    annotated = _csv_or_tsv(wd, "results", "cohort_peaks.annotated.tsv")
    # Counts made before the contig-alias counting are made again,
    # as the stage's own reuse check (_stage_checkpoint_current) decides.
    from paralome.atac.quantification import PEAK_COUNTING_METHOD

    complete = (
        bool(ckpt.get("complete"))
        and ckpt.get("counting_method") == PEAK_COUNTING_METHOD
        and _file_info(raw)["nonempty"]
        and _file_info(cpm)["nonempty"]
        and _file_info(log2_cpm)["nonempty"]
        and _file_info(sample_qc)["nonempty"]
        and _file_info(annotated)["nonempty"]
    )
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "raw_counts": _file_info(raw),
        "cpm": _file_info(cpm),
        "log2_cpm": _file_info(log2_cpm),
        "sample_qc": _file_info(sample_qc),
        "annotated_peaks": _file_info(annotated),
        "peaks": ckpt.get("peaks"),
        "samples": ckpt.get("samples"),
    }


# ---------------------------------------------------------------------------
# Every differential stage records its results the same way --
# ``statistics: {folder: manifest}``, one folder per statistic the run asked
# for (paralome.results), at the top of its checkpoint or, for a stratified
# single-cell run, in each stratum of its v2 index. Every reader below reads
# that one shape, through these two functions.
# ---------------------------------------------------------------------------


def differential_statistics(ckpt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{"" or stratum: {folder: manifest}}`` from a differential checkpoint (v1 or v2)."""
    if int(ckpt.get("schema_version") or 1) >= 2:
        return {
            str(stratum): dict((entry or {}).get("statistics") or {})
            for stratum, entry in dict(ckpt.get("strata") or {}).items()
            if (entry or {}).get("statistics")
        }
    statistics = ckpt.get("statistics")
    return {"": dict(statistics)} if isinstance(statistics, Mapping) and statistics else {}


def statistics_written(ckpt: Mapping[str, Any]) -> bool:
    """Whether every comparison's machine table a checkpoint records is on disk."""
    by_stratum = differential_statistics(ckpt)
    if not by_stratum:
        return False
    for folders in by_stratum.values():
        for manifest in folders.values():
            for entry in dict((manifest or {}).get("comparisons") or {}).values():
                if not _file_info((entry or {}).get("statistics"))["nonempty"]:
                    return False
    return True


def evidome_inputs(ckpt: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every Evidome table a differential wrote: one per statistic folder (per stratum)."""
    inputs: list[dict[str, str]] = []
    for stratum, folders in differential_statistics(ckpt).items():
        for folder, manifest in folders.items():
            table = (manifest or {}).get("evidome_input")
            if table:
                inputs.append({
                    "folder": str(folder), "stratum": str(stratum),
                    "label": f"{folder}/{stratum}" if stratum else str(folder),
                    "path": str(table),
                })
    return inputs


def summarize_differential_accessibility(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "differential_accessibility")
    # In a copied project the recorded paths are read from this copy.
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    # Complete when the checkpoint says so, under this version's
    # method, AND every comparison's machine table it records is on disk --
    # one folder per statistic (see differential_statistics).
    complete = bool(ckpt.get("complete")) and statistics_written(ckpt)
    if complete:
        # A result of another method is computed again, so it is
        # not complete here either. Imported only here: statistics.py is heavy
        # and this module is read by light commands.
        from paralome.atac.statistics import DIFF_METHOD

        complete = ckpt.get("method") == DIFF_METHOD
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "statistic": ckpt.get("statistic"),
        "statistics": differential_statistics(ckpt),
        "evidome_inputs": evidome_inputs(ckpt),
        "factors": ckpt.get("factors"),
        "blocks": ckpt.get("block_count"),
    }


# Motif analysis is two real, independently checkpointable
# pipeline stages -- motif_analysis_singles and motif_analysis_pairs (same
# split for the snATAC profile) -- instead of one stage with a bespoke
# `atac motifs --stage` escape hatch bolted on the side. Both summarizers
# read the SAME checkpoint file (run_motif_analysis writes one checkpoint
# regardless of which --stage produced it) but each looks at the field that
# actually reflects ITS half of the work: "singles_complete" for the singles
# stage, "pairs_complete" for the pairs stage. Each field is the single
# source of truth for whether its stage is done.
def _motif_out_of_date(ckpt: dict[str, Any]) -> Optional[str]:
    """Why a finished motif analysis is not current, or None.

    motifs.run_motif_analysis records MOTIF_ANALYSIS_METHOD as ``method``. A
    checkpoint under another value, or none, was computed with the skipped
    peaks inside the enrichment universe, with its comparison name unchecked
    against the differential, and -- when the pipeline drove it -- from the
    raw sample sheet rather than the deduplicated-BAM manifest. It reads
    incomplete, the supervisor says why, and the stage runs again; the motif
    stages check their own result (orchestrator.SELF_CHECKING_STAGES), so
    nothing has to be removed first. Imported late: motifs.py is heavy and
    this module is read by light commands.
    """
    from paralome.atac.motifs import MOTIF_ANALYSIS_METHOD

    recorded = ckpt.get("method")
    if recorded == MOTIF_ANALYSIS_METHOD:
        return None
    return (
        f"computed by motif method {recorded or 'from before 1.0'}; this version computes "
        f"{MOTIF_ANALYSIS_METHOD} (the enrichment universe is the scanned peaks, the comparison "
        f"is checked against the differential, and the pipeline's motif stage reads the "
        f"deduplicated-BAM manifest)"
    )


def summarize_motif_analysis_singles(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "motif_analysis")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    # No "motif_activity" entry: there is no CPM activity matrix, so motif_activity_by_sample.csv is not
    # produced, and a file-existence check on it would make this stage report incomplete forever.
    # motifs.py's own "singles_complete" checkpoint flag is the signal.
    outputs = {
        "motif_instances": _file_info(ckpt.get("motif_instances") or _csv_or_tsv(wd, "results", "machine", "motif_instances.tsv")),
        "peak_motif_matrix": _file_info(ckpt.get("peak_motif_matrix") or _csv_or_tsv(wd, "results", "machine", "peak_motif_matrix.tsv")),
        "motif_metadata": _file_info(ckpt.get("motif_metadata") or _csv_or_tsv(wd, "results", "machine", "motif_metadata.tsv")),
        "scan_summary": _file_info(ckpt.get("scan_summary") or _csv_or_tsv(wd, "results", "machine", "motif_scan_summary.tsv")),
    }
    complete = bool(ckpt.get("singles_complete")) and all(info["nonempty"] for info in outputs.values())
    # A finished result of an earlier method is out of date; see
    # _motif_out_of_date. A withdrawn record (a run in progress or one that
    # died) already reads incomplete through singles_complete.
    out_of_date = _motif_out_of_date(ckpt) if complete else None
    complete = complete and out_of_date is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "out_of_date": out_of_date,
        "withdrawn": ckpt.get("withdrawn"),
        **outputs,
        "motifs_scanned": ckpt.get("motifs_scanned"),
        "peaks_scanned": ckpt.get("peaks_scanned"),
        "motifs_tested": ckpt.get("motifs_tested"),
        "backend": ckpt.get("backend"),
        "resource_plan": ckpt.get("resource_plan"),
        "task_count": ckpt.get("task_count"),
    }


def summarize_motif_analysis_pairs(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "motif_analysis")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    # No "motif_pair_activity" entry either: this dict is intentionally empty, and `pairs_complete` below
    # is the sole signal.
    outputs: dict[str, Any] = {}
    # A prior stage="singles" checkpoint has no pairs_complete entry at all
    # yet -- that correctly reads as incomplete here (this stage hasn't run),
    # same as any other not-yet-started stage.
    complete = bool(ckpt.get("pairs_complete"))
    # Pair statistics written before the Fisher margins were counted over the same top-N universe as the
    # co-occurrence are biased toward depletion. Such a checkpoint is not complete, so the pairs stage
    # recomputes rather than handing those numbers on. Imported here because motifs.py is heavy and this
    # module is read by light commands.
    from paralome.atac.motifs import MOTIF_PAIR_STATISTICS_METHOD

    pair_method_current = ckpt.get("pair_statistics_method") == MOTIF_PAIR_STATISTICS_METHOD
    out_of_date = None
    if complete and not pair_method_current:
        complete = False
        out_of_date = (
            f"pair statistics computed by {ckpt.get('pair_statistics_method') or 'a method from an older Paralome'}; "
            f"this version computes {MOTIF_PAIR_STATISTICS_METHOD}"
        )
    # The pairs are counted over the same universe as the single
    # motifs, so an out-of-date singles method makes the pairs out of date.
    if complete:
        out_of_date = _motif_out_of_date(ckpt)
        complete = out_of_date is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "out_of_date": out_of_date,
        "withdrawn": ckpt.get("withdrawn"),
        **outputs,
        "motif_pairs_tested": ckpt.get("motif_pairs_tested"),
        "pair_statistics_method": ckpt.get("pair_statistics_method"),
        "backend": ckpt.get("backend"),
        "resource_plan": ckpt.get("resource_plan"),
        "task_count": ckpt.get("task_count"),
        "comparisons": ckpt.get("comparisons"),
    }


def _resolve_stage_checkpoint(workdir: Path, stage: str) -> Path:
    """Locate ``stage``'s checkpoint using the same resolver its writer uses.

    A checkpoint looked for in the wrong place reads as ``{}``, never complete,
    so readers resolve exactly as writers do. The canonical location wins. The
    older flat path is honoured only as a fallback, and only when the
    canonical file is absent, so older project folders still summarize
    correctly.
    """
    canonical = stage_checkpoint_path(workdir, stage)
    if canonical.is_file():
        return canonical
    legacy = legacy_stage_checkpoint_path(workdir, stage)
    if legacy is not None and legacy.is_file():
        return legacy
    return canonical


def _pseudobulk_out_of_date(ckpt: dict[str, Any], current: str) -> Optional[str]:
    """Why a finished pseudobulk is not current, or None.

    The writers record their pseudobulk method (atac/singlecell.py
    SC_PSEUDOBULK_METHOD, rna/singlecell.py SC_RNA_PSEUDOBULK_METHOD). A
    checkpoint under another method, or none, was aggregated before chrM peaks
    were excluded and empty samples dropped: it reads incomplete, so the
    pipeline aggregates again before the differential reads its cpm.csv.
    """
    recorded = ckpt.get("method")
    if recorded == current:
        return None
    return (
        f"aggregated by pseudobulk method {recorded or 'from an older Paralome'}; "
        f"this version aggregates with {current}"
    )


def summarize_singlecell_atac(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = _resolve_stage_checkpoint(wd, "singlecell_atac")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    # ATAC's aggregated pseudobulk output nests under <workdir>/atac/ (beside RNA's <workdir>/rna/, in the
    # <parent>/data/{atac,rna}/ layout). The checkpoint's own recorded path is used first; the bare
    # wd/cpm.csv fallback below only matters for older checkpoints that record no "cpm" field.
    nested_cpm, legacy_cpm = wd / "atac" / "cpm.csv", wd / "cpm.csv"
    cpm = _file_info(ckpt.get("cpm") or (nested_cpm if nested_cpm.exists() else legacy_cpm))
    nested_manifest, legacy_manifest = wd / "atac" / "sample_manifest.csv", wd / "sample_manifest.csv"
    manifest = _file_info(ckpt.get("sample_manifest") or (nested_manifest if nested_manifest.exists() else legacy_manifest))
    complete = bool(ckpt.get("complete")) and cpm["nonempty"] and manifest["nonempty"]
    out_of_date = None
    if complete:
        # Imported late, and only for a finished checkpoint.
        from paralome.atac.singlecell import SC_PSEUDOBULK_METHOD
        out_of_date = _pseudobulk_out_of_date(ckpt, SC_PSEUDOBULK_METHOD)
    complete = complete and out_of_date is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "out_of_date": out_of_date,
        "cpm": cpm,
        "sample_manifest": manifest,
        "peaks": ckpt.get("peaks"),
        "samples": ckpt.get("samples"),
        "group_by": ckpt.get("group_by"),
        "stratify_by": ckpt.get("stratify_by"),
    }


def summarize_singlecell_rna(workdir: str | Optional[Path]) -> dict[str, Any]:
    """RNA counterpart of ``summarize_singlecell_atac``. Aggregated RNA
    pseudobulk output nests under <workdir>/rna/."""
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = _resolve_stage_checkpoint(wd, "singlecell_rna")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    cpm = _file_info(ckpt.get("cpm") or (wd / "rna" / "cpm.csv"))
    manifest = _file_info(ckpt.get("sample_manifest") or (wd / "rna" / "sample_manifest.csv"))
    complete = bool(ckpt.get("complete")) and cpm["nonempty"] and manifest["nonempty"]
    out_of_date = None
    if complete:
        # Imported late, and only for a finished checkpoint.
        from paralome.rna.singlecell import SC_RNA_PSEUDOBULK_METHOD
        out_of_date = _pseudobulk_out_of_date(ckpt, SC_RNA_PSEUDOBULK_METHOD)
    complete = complete and out_of_date is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "out_of_date": out_of_date,
        "cpm": cpm,
        "sample_manifest": manifest,
        "genes": ckpt.get("genes"),
        "samples": ckpt.get("samples"),
        "group_by": ckpt.get("group_by"),
        "stratify_by": ckpt.get("stratify_by"),
    }




def _stratum_fields(ckpt: dict[str, Any]) -> dict[str, Any]:
    """The per-stratum view of a differential checkpoint, v1 or v2.

    A fanned-out run writes ``schema_version: 2`` at the top level
    with a ``strata`` map, while each stratum keeps the v1 schema in its own
    subdirectory. Both shapes are read:

    - v2 -> ``strata`` (value -> {slug, workdir, checkpoint, statistics}) and
      ``skipped_strata``, with ``stratified`` True.
    - v1 -> a single-entry ``strata`` keyed by that run's ``stratify_value``,
      so a consumer can iterate uniformly without caring which schema it got.
      An unstratified v1 run has no ``stratify_value`` and yields no entries.

    Branching on the explicit ``schema_version`` rather than on the presence of
    a ``strata`` key: inferring works today and breaks the first time a v1
    payload gains an unrelated key by that name.
    """
    if int(ckpt.get("schema_version") or 1) >= 2:
        return {
            "stratified": True,
            "strata": dict(ckpt.get("strata") or {}),
            "skipped_strata": list(ckpt.get("skipped_strata") or []),
        }
    value = ckpt.get("stratify_value")
    strata: dict[str, Any] = {}
    if value:
        strata[str(value)] = {
            "slug": None,
            "workdir": ckpt.get("workdir"),
            "checkpoint": ckpt.get("checkpoint"),
            "statistics": dict(ckpt.get("statistics") or {}),
        }
    return {"stratified": False, "strata": strata, "skipped_strata": []}


def _singlecell_differential_out_of_date(ckpt: dict[str, Any]) -> Optional[str]:
    """Why a finished single-cell differential is not current, or None.

    Both drivers and the stratified index record stats_core.SC_DIFF_METHOD; a
    result recorded under another method (or none) is issued again, and the
    stage's plan id then recomputes it. Imported late: stats_core is heavy.
    """
    from paralome.singlecell.stats_core import SC_DIFF_METHOD

    recorded = ckpt.get("method")
    if recorded == SC_DIFF_METHOD:
        return None
    return (
        f"computed by single-cell differential method {recorded or 'from an older Paralome'}; "
        f"this version computes {SC_DIFF_METHOD}"
    )


def summarize_singlecell_differential(workdir: str | Optional[Path]) -> dict[str, Any]:
    # Reads the checkpoint written by `atac.singlecell_stats.run_singlecell_differential`
    # (the CLI's `atac singlecell-differential`, which the pipeline's
    # `singlecell_differential` stage drives -- see orchestrator.py): one
    # folder of results per statistic, recorded under `statistics`.
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = _resolve_stage_checkpoint(wd, "singlecell_differential")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    complete = (bool(ckpt.get("complete")) and _file_info(checkpoint)["nonempty"]
                and statistics_written(ckpt))
    out_of_date = _singlecell_differential_out_of_date(ckpt) if complete else None
    complete = complete and out_of_date is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "out_of_date": out_of_date,
        "stratify_by": ckpt.get("stratify_by"),
        "stratify_value": ckpt.get("stratify_value"),
        "factors": ckpt.get("factors"),
        "statistic": ckpt.get("statistic"),
        "statistics": differential_statistics(ckpt),
        # What functional_evidence reads -- every statistic's table.
        "evidome_inputs": evidome_inputs(ckpt),
        **_stratum_fields(ckpt),
    }


def summarize_singlecell_rna_differential(workdir: str | Optional[Path]) -> dict[str, Any]:
    """RNA counterpart of ``summarize_singlecell_differential`` -- reads the
    checkpoint written by ``rna.singlecell_stats.run_singlecell_rna_differential``
    (``rna singlecell-differential``, driven by the pipeline's
    ``singlecell_rna_differential`` stage)."""
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = _resolve_stage_checkpoint(wd, "singlecell_rna_differential")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    complete = (bool(ckpt.get("complete")) and _file_info(checkpoint)["nonempty"]
                and statistics_written(ckpt))
    out_of_date = _singlecell_differential_out_of_date(ckpt) if complete else None
    complete = complete and out_of_date is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "out_of_date": out_of_date,
        "stratify_by": ckpt.get("stratify_by"),
        "stratify_value": ckpt.get("stratify_value"),
        "factors": ckpt.get("factors"),
        "statistic": ckpt.get("statistic"),
        "statistics": differential_statistics(ckpt),
        # What functional_evidence reads -- every statistic's table.
        "evidome_inputs": evidome_inputs(ckpt),
        **_stratum_fields(ckpt),
    }


def summarize_rna_quantification(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "rna_fpkm_quantification")
    ckpt = _read_json(checkpoint)
    gene_fpkm = wd / "results" / "machine" / "gene_fpkm.tsv"
    gene_log2 = wd / "results" / "machine" / "gene_log2_fpkm.tsv"
    sample_qc = wd / "results" / "machine" / "sample_expression_qc.tsv"
    complete = bool(ckpt.get("complete")) and all(
        _file_info(path)["nonempty"] for path in (gene_fpkm, gene_log2, sample_qc)
    )
    # The stage's own rule, against what the result records, so a
    # result 'rna quantify-fpkm' would recompute (e.g. an older library-size
    # rule) is not complete and is issued again before the differential.
    stale_reason = None
    if complete:
        from paralome.rna.fpkm import quantification_stale_reason

        recorded = _in_this_copy(wd, ckpt)
        stale_reason = quantification_stale_reason(
            recorded,
            strandedness=recorded.get("strandedness"),
            paired_end=recorded.get("paired_end"),
            pseudocount=recorded.get("pseudocount"),
            require_both_ends_mapped=recorded.get("require_both_ends_mapped"),
            exclude_chimeric_fragments=recorded.get("exclude_chimeric_fragments"),
            fingerprint=str(recorded.get("input_fingerprint") or ""),
            request=str(recorded.get("request_fingerprint") or ""),
        )
        complete = stale_reason is None
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "stale_reason": stale_reason,
        "gene_fpkm": _file_info(gene_fpkm),
        "gene_log2_fpkm": _file_info(gene_log2),
        "sample_qc": _file_info(sample_qc),
        "samples": ckpt.get("sample_count") or ckpt.get("samples"),
        "genes": ckpt.get("gene_count") or ckpt.get("genes"),
    }


def summarize_differential_expression(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint = stage_checkpoint_path(wd, "differential_expression")
    ckpt = _in_this_copy(wd, _read_json(checkpoint))
    # The supervisor reads this to decide whether the stage is done, without
    # calling differential_expression() and its own version check, so a
    # checkpoint written by a different Paralome version is not "complete"
    # here either. And every comparison's machine table it records
    # is on disk (one folder per statistic).
    complete = (
        bool(ckpt.get("complete"))
        and statistics_written(ckpt)
        and version_stale_reason(ckpt) is None
    )
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint),
        "statistic": ckpt.get("statistic"),
        "statistics": differential_statistics(ckpt),
        # The inputs of the functional_evidence stage.
        "evidome_inputs": evidome_inputs(ckpt),
        "factors": ckpt.get("factors"),
        "genes_tested": ckpt.get("genes_tested"),
        "blocks": ckpt.get("block_count"),
    }

def summarize_functional_evidence(workdir: str | Optional[Path], zip_dir: Optional[Path] = None) -> dict[str, Any]:
    """Evidome's stage -- see paralome.evidence.summarize (which moves a finished run's archive
    into ``zip_dir``)."""
    from paralome.evidence import summarize

    return summarize(Path(workdir) if workdir else None, zip_dir)


#: The folder every zip a run makes goes in: all OmicView output in one folder, organized by analysis
#: (<parent>/omicview/, with each run's output in its own subfolder, e.g.
#: <parent>/omicview/<run>/functional_evidence).
OMICVIEW_FOLDER = "omicview"


#: The folder of the run's trajectories folder the lineages are fitted into, by either method.
TRAJECTORY_FIT_FOLDER = "fit"

def omicview_dir(project_root: str | Optional[Path], data_workdir: str | Optional[Path] = None,
                 single_cell: bool = False) -> Optional[Path]:
    """Where a run's zips go. Single-cell: ``<parent>/omicview/<run>``, the parent being the
    folder holding the extraction (``<parent>/data``) and ``<run>`` this run's folder's path inside it; the
    cell exports go in the run's folder too. Bulk: ``<project>/omicview``, inside the project. Each stage's
    zips in a subfolder named for the stage."""
    if not project_root:
        return None
    root = Path(project_root)
    if single_cell and data_workdir:
        parent = Path(data_workdir).parent
        try:
            run = root.resolve().relative_to(parent.resolve())
        except ValueError:
            run = Path(root.name)
        if str(run) in ("", "."):
            run = Path(root.name)
        return parent / OMICVIEW_FOLDER / run
    return root / OMICVIEW_FOLDER


def summarize_statistics_output(workdir: str | Optional[Path], zip_dir: Optional[Path] = None) -> dict[str, Any]:
    """The statistics_output stage -- see paralome.browser.arm.summarize."""
    from paralome.browser.arm import summarize

    return summarize(Path(workdir) if workdir else None, zip_dir)


def summarize_trajectories(workdir: str | Optional[Path], zip_dir: Optional[Path] = None) -> dict[str, Any]:
    """-- see paralome.singlecell.trajectories.summarize."""
    from paralome.singlecell.trajectories import summarize

    return summarize(Path(workdir) if workdir else None, zip_dir)


def summarize_genotype_output(workdir: str | Optional[Path], zip_dir: Optional[Path] = None) -> dict[str, Any]:
    """The genotype_output stage -- see paralome.browser.stage.summarize."""
    from paralome.browser.stage import summarize

    return summarize(Path(workdir) if workdir else None, zip_dir)


def summarize_genome_tracks(workdir: str | Optional[Path], zip_dir: Optional[Path] = None) -> dict[str, Any]:
    """The ATAC pipelines' genome browser tracks -- see paralome.atac.genome_tracks.summarize."""
    from paralome.atac.genome_tracks import summarize

    return summarize(Path(workdir) if workdir else None, zip_dir)


def summarize_gene_burden(workdir: str | Optional[Path]) -> dict[str, Any]:
    """The DNA gene-burden stage -- see paralome.burden.stage.summarize."""
    from paralome.burden.stage import summarize

    return summarize(Path(workdir) if workdir else None)


def _functional_evidence_follows_gene_burden(state: dict[str, Any]) -> None:
    """For DNA, Evidome reads gene_burden's table; no table, no Evidome.

    A skipped gene_burden (no burden reference, too few probands, a species
    gnomAD does not describe) leaves functional_evidence nothing to read. It is
    skipped with it, with the reason, unless it has a finished result of its
    own -- and it is read afresh every time, so once gene_burden runs, Evidome
    is next again.
    """
    burden = state.get("gene_burden") if isinstance(state.get("gene_burden"), dict) else {}
    evidence = state.get("functional_evidence") if isinstance(state.get("functional_evidence"), dict) else {}
    if str(burden.get("status") or "") != "skipped":
        return
    # A finished result is kept only while the stage is not on the
    # stale list. Stale means gene_burden ran again (and was skipped), so that
    # result was built from a burden table the project no longer has.
    project = state.get("project") if isinstance(state.get("project"), dict) else {}
    stale = "functional_evidence" in {str(n) for n in (project.get("stale_stages") or ())}
    if evidence.get("complete") and not stale:
        return
    state["functional_evidence"] = {
        **evidence,
        "results_not_used": bool(evidence.get("complete")),
        "status": "skipped",
        "complete": False,
        "reason": f"gene_burden was skipped, so there is no gene-burden table for Evidome: {burden.get('reason') or 'see gene_burden'}",
        "skip_reason": "gene_burden_skipped",
    }


def summarize_annotation(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoints = sorted((wd / "checkpoints").glob("annotation_*.json")) if (wd / "checkpoints").exists() else []
    # One checkpoint per engine chain; with more than one left in
    # the folder, read the most recently written, not the first by name. A
    # checkpoint `annotate` marked superseded_by (annotation/standalone.py
    # begin_annotation_checkpoint) stands for an earlier chain, so
    # it is passed over even when that marking made it the newest file.
    current = [p for p in checkpoints if "superseded_by" not in (_read_json(p) or {})] or checkpoints
    ckpt_path = max(current, key=lambda p: (p.stat().st_mtime, p.name)) if current else stage_checkpoint_path(wd, "annotation")
    ckpt = _in_this_copy(wd, _read_json(ckpt_path))
    output = ckpt.get("output_vcf") or None
    if not output:
        candidate = _glob_first([str(wd / "results" / "*.annotated.vcf.gz")])
        output = str(candidate) if candidate else None
    out_info = _file_info(output)
    idx_info = _file_info(str(output) + ".tbi") if output else {"path": None, "exists": False, "nonempty": False, "size": 0}
    # A non-empty VCF is not a finished one -- see vcf_with_index_ready. Without a checkpoint saying so,
    # the stage is complete only when the VCF has an index written after it.
    # Complete:false in the checkpoint is final: `annotate` writes complete:false before it starts
    # (annotation/standalone.py begin_annotation_checkpoint), and that holds here even while the previous
    # VCF and its index are still on disk waiting for the job to replace them.
    recorded = ckpt.get("complete")
    if recorded is False:
        complete = False
    else:
        complete = bool(recorded) or vcf_with_index_ready(output)
    # An annotation whose calls were split for ANNOVAR by an earlier split method (annovar.SPLIT_METHOD)
    # reads incomplete, so it is made again and the stages after it no longer answer (supervisor: those
    # that do not check themselves are named to be removed; gene burden computes again on the new VCF).
    # Without a checkpoint the split cannot be known; an ANNOVAR file name says one was made.
    from paralome.annotation.annovar import SPLIT_METHOD

    engine = str(ckpt.get("engine") or "")
    if not engine and output:
        engine = Path(str(output)).name
    split_by_annovar = "annovar" in engine.lower().replace(".", "_").split("_")
    out_of_date = None
    # `recorded or complete`, not `complete`: an annotation the
    # checkpoint calls finished under an earlier split is out of date whether
    # or not its files are all still there, and the person is told why.
    if (recorded or complete) and split_by_annovar and ckpt.get("split_method") != SPLIT_METHOD:
        recorded = ckpt.get("split_method")
        out_of_date = (
            f"its calls were split for ANNOVAR by {'an earlier split (' + str(recorded) + ')' if recorded else 'a split from an older Paralome, or one not recorded'}; "
            f"the split is now {SPLIT_METHOD}, which corrects which samples carry complex and multi-allelic calls"
        )
        complete = False
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "out_of_date": out_of_date,
        "workdir": str(wd),
        "checkpoint": _file_info(ckpt_path),
        "engine": ckpt.get("engine"),
        "input_vcf": ckpt.get("input_vcf"),
        "output_vcf": out_info,
        "output_index": idx_info,
    }


def summarize_segregation(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint_path = stage_checkpoint_path(wd, "segregation")
    plan_path = wd / "inputs" / "segregation_plan.json"
    ckpt = _in_this_copy(wd, _read_json(checkpoint_path))
    plan = _in_this_copy(wd, _read_json(plan_path))
    outputs = ckpt.get("outputs") if isinstance(ckpt.get("outputs"), dict) else {}
    if not outputs:
        outputs = plan.get("outputs") if isinstance(plan.get("outputs"), dict) else {}

    # The prefix this workdir's files are SUPPOSED to carry, derived exactly as the writers derive it.
    # Everything below is checked against it, so a file left behind by a half-finished migration does not
    # pass for a correctly named one.
    expected = _expected_prefix(wd)

    output = (_reject_if_bare(outputs.get("all_variants"), "all_variants.segregation.tsv", expected)
              or _reject_if_bare(outputs.get("segregation_tsv"), "all_variants.segregation.tsv", expected))
    if not output:
        # ONLY the prefixed form -- the exact prefix when it is known, not any prefix -- so a tree of bare
        # names never reads complete.
        patterns = ([str(wd / "results" / f"{expected}_all_variants.segregation.tsv")] if expected
                    else [str(wd / "results" / "*_all_variants.segregation.tsv")])
        candidate = _glob_first(patterns)
        output = str(candidate) if candidate else None

    summary_json = (_reject_if_bare(outputs.get("summary"), "segregation_summary.json", expected)
                    or _prefixed_fallback(wd / "results", "segregation_summary.json", expected))
    provenance_json = outputs.get("provenance") or str(wd / "results" / "provenance.json")
    summary = _read_json(Path(summary_json)) if summary_json else {}
    bucket_names = ["segregation_pass", "segregation_partial", "segregation_fail", "segregation_uninformative"]
    bucket_outputs = {
        bucket: _file_info(_reject_if_bare(outputs.get(bucket), f"{bucket}.tsv", expected)
                           or _prefixed_fallback(wd / "results", f"{bucket}.tsv", expected))
        for bucket in bucket_names
    }
    # Counts only from a finished, real run's checkpoint. A dry run
    # rewrites it with rows 0 and every bucket 0, which made a finished stage
    # read incomplete; then the summary (as the reuse gate uses) answers.
    counted = ckpt if (ckpt.get("complete") and not ckpt.get("dry_run")) else {}
    bucket_counts = dict(counted.get("bucket_counts") or summary.get("bucket_counts") or {})
    out_info = _file_info(output)
    rows = counted.get("rows") or summary.get("rows")
    bucket_sum = counted.get("bucket_sum") or summary.get("bucket_sum")
    lossless = bool(ckpt.get("lossless") or summary.get("lossless"))
    complete = bool(ckpt.get("complete") and out_info.get("exists") and lossless) or bool(
        out_info.get("exists") and _file_info(summary_json).get("nonempty") and lossless
    )
    # And the tables on disk must be the ones that claim was made about: a `segregate run --overwrite`
    # killed part way must not leave a complete checkpoint and lossless summary beside truncated tables.
    # A checkpoint records each table's size as it closes, compared here without reading the tables;
    # an older checkpoint, or a summary standing alone, is checked by counting the rows.
    tables_agree = None
    if complete:
        table_paths = {"all_variants": output, **{b: bucket_outputs[b].get("path") for b in bucket_names}}
        recorded_sizes = ckpt.get("output_bytes") if ckpt.get("complete") else None
        if isinstance(recorded_sizes, dict) and recorded_sizes:
            tables_agree = output_sizes_agree(recorded_sizes, table_paths)
        else:
            tables_agree = rows is not None and count_table_rows(output) == int(rows)
            if tables_agree and bucket_counts:
                tables_agree = all(
                    count_table_rows(table_paths[b]) == int(bucket_counts.get(b, 0)) for b in bucket_names
                )
        complete = bool(tables_agree)
    # A table an older segregation method wrote is not complete: its rows lack the per-genotype evidence
    # filtering needs. Incomplete, the stage runs again (and the stage itself does not reuse it:
    # segregation.runner).
    from paralome.segregation.table import SEGREGATION_METHOD

    stale_method = complete and summary.get("method") != SEGREGATION_METHOD
    if stale_method:
        complete = False
    # A table lacking the depth, GQ and alt-read columns the filter thresholds read is not complete: the
    # stage is written again, and the stages after it no longer answer.
    out_of_date = None
    if out_info.get("exists"):
        from paralome.segregation.table import missing_table_columns

        missing = missing_table_columns(Path(str(output)))
        if missing:
            out_of_date = (
                "its table was written by an earlier Paralome (it has no " + ", ".join(missing[:6])
                + (" ..." if len(missing) > 6 else "") + " column; its recessive score was 0 "
                "and the columns the filter thresholds read were missing)"
            )
            complete = False
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "tables_match_record": tables_agree,
        "stale_method": stale_method,
        "out_of_date": out_of_date,
        "lossless": lossless,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint_path),
        "plan": _file_info(plan_path),
        "input_vcf": ckpt.get("input_vcf") or plan.get("input_vcf"),
        "samples": ckpt.get("samples") or plan.get("samples"),
        "segregation_tsv": out_info,
        "all_variants_tsv": out_info,
        "summary_json": _file_info(summary_json),
        "provenance_json": _file_info(provenance_json),
        "bucket_outputs": bucket_outputs,
        "bucket_counts": bucket_counts,
        "bucket_sum": bucket_sum,
        "rows": rows,
    }


def summarize_variants(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    pipeline_path = stage_checkpoint_path(wd, "pipeline")
    dispatch_plan = wd / "inputs" / "dispatch_plan.json"
    run_plan = wd / "inputs" / "pipeline_run_plan.json"
    pipe = _in_this_copy(wd, _read_json(pipeline_path))

    final_output = _safe_get(pipe, "final", "output") or _safe_get(pipe, "merge", "output")
    if not final_output:
        # Prefer likely final results if checkpoint is incomplete or old.
        final = _glob_first([str(wd / "results" / "*.annotated.vcf.gz"), str(wd / "results" / "*.merged.vcf.gz")])
        final_output = str(final) if final else None

    final_info = _file_info(final_output)
    index_info = _file_info(str(final_output) + ".tbi") if final_output else {"path": None, "exists": False, "nonempty": False, "size": 0}
    # As for annotation -- the index, not just the file.
    complete = bool(_safe_get(pipe, "final", "complete", default=False)) or vcf_with_index_ready(final_output)

    freebayes = pipe.get("freebayes", {}) if isinstance(pipe.get("freebayes", {}), dict) else {}
    summary_path = freebayes.get("summary") or str(wd / "results" / "freebayes_summary.json")
    freebayes_summary = _read_json(Path(summary_path)) if summary_path else {}
    annotation = pipe.get("annotation", {}) if isinstance(pipe.get("annotation", {}), dict) else {}
    segregation = pipe.get("segregation", {}) if isinstance(pipe.get("segregation", {}), dict) else {}

    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "workdir": str(wd),
        "pipeline_json": _file_info(pipeline_path),
        "dispatch_plan": _file_info(dispatch_plan),
        "pipeline_run_plan": _file_info(run_plan),
        "sample_manifest": _safe_get(pipe, "inputs", "sample_manifest"),
        "final_output": final_info,
        "final_index": index_info,
        "freebayes": freebayes,
        "freebayes_summary": freebayes_summary,
        "annotation": annotation,
        "segregation": segregation,
    }


def summarize_filtering(workdir: str | Optional[Path], profile: Optional[str] = None) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False, "profiles": {}}
    wd = Path(workdir)
    checkpoint_path = stage_checkpoint_path(wd, "filtering")
    plan_path = wd / "inputs" / "filter_plan.json"
    checkpoint = _in_this_copy(wd, _read_json(checkpoint_path))
    plan = _in_this_copy(wd, _read_json(plan_path))
    results = wd / "results"
    # See summarize_segregation. The expected prefix, not any prefix.
    expected = _expected_prefix(wd)

    profiles: dict[str, Any] = {}
    # Compare canonical names (the filter writes the canonical one),
    # and apply the requested-profile filter to every source below.
    from paralome.filters.profiles import normalize_profile_name

    wanted = normalize_profile_name(profile)

    def not_requested(prof: str) -> bool:
        return wanted != "none" and normalize_profile_name(prof) != wanted

    def add_summary(path: Path) -> None:
        data = _read_json(path)
        prof = str(data.get("profile") or path.parent.name or path.name.replace(".filter_summary.json", ""))
        if not_requested(prof):
            return
        if prof in profiles:
            return
        profiles[prof] = {
            "summary": _file_info(path),
            "method": data.get("method"),
            "counts": data.get("counts", {}),
            "description": data.get("description"),
            "scope": data.get("scope"),
            "lossless": data.get("lossless"),
            "outputs": data.get("outputs", {}),
            "filter_config": data.get("filter_config"),
            "filter_config_sha256": data.get("filter_config_sha256"),
        }

    outputs = checkpoint.get("outputs", {}) if isinstance(checkpoint.get("outputs"), dict) else {}
    profile_outputs = outputs.get("profiles", {}) if isinstance(outputs.get("profiles"), dict) else {}
    for outmap in profile_outputs.values():
        # A checkpoint recording a bare summary is an older layout's output; not accepted as complete.
        recorded = _reject_if_bare(outmap.get("summary") if isinstance(outmap, dict) else None,
                                   "filter_summary.json", expected)
        if recorded:
            add_summary(Path(recorded))

    # The aggregate summary is not accepted on its own either (it carries profile_summaries, which would
    # populate `profiles`).
    aggregate_summary_path = (_reject_if_bare(outputs.get("aggregate_summary"), "filtering_summary.json", expected)
                              or _reject_if_bare(outputs.get("summary"), "filtering_summary.json", expected))
    aggregate_summary = _in_this_copy(wd, _read_json(Path(aggregate_summary_path))) if aggregate_summary_path else {}
    aggregate_profiles = aggregate_summary.get("profile_summaries", {}) if isinstance(aggregate_summary.get("profile_summaries"), dict) else {}
    for prof, obj in aggregate_profiles.items():
        if not_requested(str(prof)):
            continue
        if prof not in profiles and isinstance(obj, dict):
            summary_path = obj.get("summary")
            profiles[prof] = {
                "summary": _file_info(summary_path),
                "method": _read_json(Path(summary_path)).get("method") if summary_path else None,
                "counts": obj.get("counts", {}),
                "scope": aggregate_summary.get("scope"),
                "lossless": obj.get("lossless"),
                "outputs": obj.get("outputs", {}),
                "filter_config": aggregate_summary.get("filter_config"),
                "filter_config_sha256": aggregate_summary.get("filter_config_sha256"),
            }

    patterns: list[str] = []
    # The exact expected prefix when it is derivable: `*_filter_summary.json` does not match the bare name,
    # but it DOES match a wrongly-prefixed one.
    stem = f"{expected}_filter_summary.json" if expected else "*_filter_summary.json"
    if wanted != "none":
        patterns.append(f"{wanted}/{stem}")
    patterns.extend([f"*/{stem}", stem])
    if results.exists():
        for pattern in patterns:
            for path in sorted(results.glob(pattern)):
                add_summary(path)

    # The checkpoint alone is not sufficient. It says "this ran", not "its outputs follow the current
    # convention".
    complete = bool(profiles)
    # `filter run --overwrite` rewrites the tables in place, and the run withdraws its checkpoint and
    # summaries before it writes. Here a profile counts only with its summary on disk, not marked lossy,
    # and its tables the ones the record describes -- by the sizes the checkpoint records, or, for an
    # older record, by counting rows.
    filter_issues: list[str] = []
    if complete and checkpoint and checkpoint.get("complete") is False and not checkpoint.get("dry_run"):
        filter_issues.append("the last filtering run started and did not finish")
    recorded_by_profile = checkpoint.get("output_bytes") if checkpoint.get("complete") else None
    if not isinstance(recorded_by_profile, dict):
        recorded_by_profile = {}
    filter_buckets = ("pass", "fail_quality", "fail_population", "fail_annotation", "fail_segregation", "fail_multiple")
    for prof, info in profiles.items() if complete else ():
        if not (info.get("summary") or {}).get("nonempty"):
            filter_issues.append(f"{prof}: its summary is missing")
            continue
        if info.get("lossless") is False:
            filter_issues.append(f"{prof}: its summary records a lossy run")
            continue
        tables = _in_this_copy(wd, info.get("outputs") or {})
        tables = tables if isinstance(tables, dict) else {}
        recorded = recorded_by_profile.get(prof)
        if isinstance(recorded, dict) and recorded:
            if not output_sizes_agree(recorded, tables):
                filter_issues.append(f"{prof}: its tables are not the ones the checkpoint recorded")
            continue
        counts = info.get("counts") or {}
        expected_rows = {"all": counts.get("total"), **{b: counts.get(b) for b in filter_buckets}}
        for key, expected_count in expected_rows.items():
            if expected_count is None or not tables.get(key):
                filter_issues.append(f"{prof}: no recorded row count or path for {key}")
                break
            if count_table_rows(tables[key]) != int(expected_count):
                filter_issues.append(f"{prof}: {key} does not hold the {int(expected_count)} rows its summary records")
                break
    if filter_issues:
        complete = False
    # A profile filtered by an older method is not a finished result (filters.table FILTER_METHOD).
    from paralome.filters.table import FILTER_METHOD

    stale_method = sorted(name for name, entry in profiles.items() if entry.get("method") != FILTER_METHOD)
    if stale_method:
        complete = False
    status = "complete" if complete else ("incomplete" if profiles else "not_available")
    return {
        "status": status,
        "complete": complete,
        "issues": filter_issues,
        "stale_method": stale_method,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint_path),
        "plan": _file_info(plan_path),
        "profile": checkpoint.get("profile") or plan.get("profile") or profile,
        "profiles_requested": checkpoint.get("profiles") or plan.get("profiles") or list(profiles),
        "scope": checkpoint.get("scope") or plan.get("scope") or aggregate_summary.get("scope"),
        "input_segregation": checkpoint.get("input_segregation") or plan.get("input_segregation") or aggregate_summary.get("input_segregation"),
        "filter_config": checkpoint.get("filter_config") or plan.get("filter_config") or aggregate_summary.get("filter_config"),
        "filter_config_sha256": checkpoint.get("filter_config_sha256") or plan.get("filter_config_sha256") or aggregate_summary.get("filter_config_sha256"),
        "outputs": outputs,
        "counts": checkpoint.get("counts") or aggregate_summary.get("counts", {}),
        "lossless": checkpoint.get("lossless") if "lossless" in checkpoint else aggregate_summary.get("lossless"),
        "aggregate_summary": _file_info(aggregate_summary_path),
        "profiles": profiles,
    }


def summarize_reporting(workdir: str | Optional[Path]) -> dict[str, Any]:
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    checkpoint_path = stage_checkpoint_path(wd, "candidate_report")
    checkpoint = _in_this_copy(wd, _read_json(checkpoint_path))
    outputs = checkpoint.get("outputs") if isinstance(checkpoint.get("outputs"), dict) else {}
    # See summarize_segregation. The expected prefix, not any prefix.
    expected = _expected_prefix(wd)
    summary_path = (_reject_if_bare(outputs.get("summary_json"), "candidate_report_summary.json", expected)
                    or _prefixed_fallback(wd / "results", "candidate_report_summary.json", expected))
    manifest_path = (_reject_if_bare(outputs.get("audit_manifest_json"), "candidate_report_manifest.json", expected)
                     or _prefixed_fallback(wd / "results", "candidate_report_manifest.json", expected))
    summary = _read_json(summary_path)
    required = {
        "summary": _file_info(summary_path),
        "manifest": _file_info(manifest_path),
        "all_reviewable": _file_info(
            _reject_if_bare(outputs.get("all_reviewable_tsv"), "all_reviewable_candidates.tsv", expected)
            or _prefixed_fallback(wd / "results", "all_reviewable_candidates.tsv", expected)),
    }
    complete = bool(checkpoint.get("complete")) and all(v.get("nonempty") for v in required.values())
    # A report is complete when it is CURRENT: written under this version's classification
    # (reports.candidates.REPORT_METHOD and CHECKPOINT_SCHEMA) from the segregation table as it is on disk
    # now, so a table rewritten in place (an --overwrite rerun after a pedigree correction) makes the
    # stage run again.
    from paralome.reports.candidates import report_staleness

    stale = report_staleness(checkpoint) if complete else None
    stale_method = bool(stale)
    if stale:
        complete = False
    return {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "stale": stale,
        "stale_method": stale_method,
        "workdir": str(wd),
        "checkpoint": _file_info(checkpoint_path),
        "outputs": required,
        "method": checkpoint.get("method"),
        "rows_scanned": checkpoint.get("rows_scanned") or summary.get("rows_scanned"),
        "rows_reviewable": checkpoint.get("rows_reviewable") or summary.get("rows_reviewable"),
        "bucket_tsv_total_rows": checkpoint.get("bucket_tsv_total_rows") or summary.get("bucket_tsv_total_rows"),
        "threads": checkpoint.get("threads"),
        "shards": checkpoint.get("shards"),
        "execution_mode": checkpoint.get("execution_mode"),
    }


CANONICAL_STAGE_STATES = {
    "not_started",
    "waiting",
    "ready",
    "running",
    "complete",
    "failed",
    "skipped",
}


def canonical_stage_state(stage_name: str, stage: dict[str, Any], *, prerequisites_complete: bool = True) -> str:
    """Return a normalized workflow-state label for a project stage."""
    status = str(stage.get("status") or "unknown").lower()
    complete = bool(stage.get("complete"))

    if complete or status == "complete":
        return "complete"
    if status in {"failed", "error"}:
        return "failed"
    if status in {"running", "submitted", "pending"}:
        return "running"
    if status in {"skipped", "not_applicable"}:
        return "skipped"
    if status in {"not_available", "none", "not_provided", "missing"}:
        return "not_started" if prerequisites_complete else "waiting"
    if status in {"partial", "incomplete"}:
        return "ready" if prerequisites_complete else "waiting"
    return "ready" if prerequisites_complete else "waiting"


#: The human name for every stage any profile can run.
#:
#: compute_pipeline_state and `print_pipeline_dashboard` both read this one
#: table, so every profile's stages are labelled the same way everywhere.
STAGE_LABELS: dict[str, str] = {
    "fastq_qc": "FASTQ QC",
    "alignment": "STAR alignment",
    "rna_quantification": "RNA FPKM quantification",
    "differential_expression": "Differential expression",
    "deduplication": "Duplicate marking",
    "macs_peak_calling": "MACS3 peak calling",
    "peak_merge": "Cohort peak merge",
    "peak_quantification": "Peak quantification",
    "differential_accessibility": "Differential accessibility",
    "motif_analysis_singles": "Motif analysis (single motifs)",
    "motif_analysis_pairs": "Motif analysis (motif pairs)",
    "variants": "Variants",
    "annotation": "Annotation",
    "segregation": "Segregation",
    "filtering": "Filtering",
    "reporting": "Candidate reporting",
    "singlecell_atac": "Single-cell pseudobulk aggregation",
    "singlecell_differential": "Single-cell differential accessibility",
    "singlecell_motif_analysis_singles": "Single-cell motif analysis (single motifs)",
    "singlecell_motif_analysis_pairs": "Single-cell motif analysis (motif pairs)",
    "singlecell_rna": "Single-cell RNA pseudobulk aggregation",
    "singlecell_rna_differential": "Single-cell RNA differential expression",
    "functional_evidence": "Functional evidence (Evidome)",
    "gene_burden": "Gene burden (probands vs gnomAD)",
    "statistics_output": "Statistics output (OmicView)",
    "genotype_output": "Genotype output (OmicView)",
    "genome_tracks": "Genome browser tracks",
    #
    "trajectories": "Trajectories (Slingshot lineages and the statistics along them)",
    # The legacy unsplit motif stage names, still accepted in older
    # project_state.json files, so they are given labels too.
    "motif_analysis": "Motif analysis (legacy combined stage)",
    "singlecell_motif_analysis": "Single-cell motif analysis (legacy combined stage)",
}


def compute_pipeline_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build the workflow state-machine section for project_state.json."""
    labels = STAGE_LABELS

    project = state.get("project", {}) if isinstance(state.get("project"), dict) else {}
    resolved_profile = resolve_pipeline_profile(
        project.get("pipeline_profile") or project.get("profile"),
        project.get("segregation_profile"),
    )
    stage_order = list(resolved_profile.stage_graph)
    segregation_profile = str(resolved_profile.segregation_profile or "none").strip().lower()

    # --stop-at: everything strictly after the requested stage is treated as
    # not_started/optional -- same mechanism as --skip-motif below -- so the
    # pipeline reports itself complete once the requested stage finishes,
    # without ever touching later stages' checkpoints. A later run/continue
    # invocation made *without* --stop-at sees those stages as ordinary
    # not_started stages again and simply resumes past this point.
    stale_stages = {str(name) for name in (project.get("stale_stages") or ())}
    # "stage.part" on the stale list -- a part of a finished stage is to run again (profiles.STAGE_PARTS).
    stale_parts: dict[str, set[str]] = {}
    for entry in list(stale_stages):
        stage, part = split_read(entry)
        if part:
            stale_stages.discard(entry)
            stale_parts.setdefault(stage, set()).add(part)
    stop_at_raw = str(project.get("stop_at") or "").strip().lower()
    stop_at_index = stage_order.index(stop_at_raw) if stop_at_raw in stage_order else None
    # What each stage reads (profiles.STAGE_READS; a line where a profile has none). A stage's
    # prerequisites are the stages it reads, and theirs -- for a line, every stage before it. --stop-at
    # keeps the stage named and what it reads: for a line, every stage up to it.
    reads = stage_reads(resolved_profile.pipeline_profile, stage_order)
    kept_by_stop = (stage_ancestors(reads, stop_at_raw) | {stop_at_raw}) if stop_at_index is not None else None

    stages: dict[str, Any] = {}
    unfinished: dict[str, Any] = {}             # required stage -> None (all of it) or its parts not finished
    blocked: set[str] = set()                   # stages that read something unfinished
    current_stage = None
    next_stage = None
    ready_stages: list[str] = []
    pipeline_state = "complete"

    for stage_index, name in enumerate(stage_order):
        obj = state.get(name, {}) if isinstance(state.get(name), dict) else {}
        prerequisites_complete = not any(blocks(read, unfinished, blocked) for read in reads.get(name, ()))
        if not prerequisites_complete:
            blocked.add(name)
        canonical = canonical_stage_state(name, obj, prerequisites_complete=prerequisites_complete)
        # A finished stage with a part to run again (its summary says, or the stale list).
        parts_to_run = set(obj.get("unfinished_parts") or ()) | stale_parts.get(name, set())

        # Filtering is optional unless a non-none filter profile was requested.
        # Keep its display state honest as not_started, but do not make an
        # otherwise complete variant-only workflow look incomplete.
        optional_stage = False
        if name == "filtering" and str(obj.get("status") or "").lower() in {"not_available", "not_provided", "none"}:
            if segregation_profile in {"", "none", "."}:
                canonical = "not_started"
                optional_stage = True

        # --skip-motif lets a run bypass TF motif finding entirely -- both
        # halves of it (bulk ATAC's motif_analysis_singles/_pairs stages, or
        # snATAC's singlecell_motif_analysis_singles/_pairs stages). Same
        # treatment as filtering above: report it honestly as not_started
        # rather than letting it block/represent the pipeline as incomplete.
        if (
            name in {
                "motif_analysis_singles", "motif_analysis_pairs",
                "singlecell_motif_analysis_singles", "singlecell_motif_analysis_pairs",
            }
            and bool(project.get("skip_motif"))
        ):
            canonical = "not_started"
            optional_stage = True

        # --skip-motif-pairs marks only the pairs HALF of motif analysis optional, as --skip-motif does for
        # the whole thing above -- singles still runs and is required. Pairs is its own stage with its own
        # "pairs_complete" checkpoint field (see summarize_motif_analysis_pairs): a later run made without
        # --skip-motif-pairs does not mark it optional, and the ordinary completeness check (which reads
        # pairs_complete=False from the skipped run) sends it back to "ready".
        if (
            name in {"motif_analysis_pairs", "singlecell_motif_analysis_pairs"}
            and bool(project.get("skip_motif_pairs"))
        ):
            canonical = "not_started"
            optional_stage = True

        # --no-genome-track, read as --skip-evidence is.
        if name == "genome_tracks" and bool(project.get("no_genome_track")):
            canonical = "not_started"
            optional_stage = True

        # --no-trajectories, the same.
        if name == "trajectories" and bool(project.get("no_trajectories")):
            canonical = "not_started"
            optional_stage = True

        # --skip-evidence, read the same way --skip-motif is. Evidome is on every graph, so a run that says no
        # to it reports the stage as not started and optional rather than as the thing holding the pipeline
        # short of complete.
        if name == "functional_evidence" and bool(project.get("skip_evidence")):
            canonical = "not_started"
            optional_stage = True

        # DNA's functional_evidence, skipped because gene_burden was
        # (_functional_evidence_follows_gene_burden). Nothing of its own is
        # there to be stale: it is read afresh from gene_burden every time, so
        # it is never put on the stale list and re-issued with nothing to read.
        # Shown as not started (its status says skipped, and why), like the
        # optional stages above: once gene_burden has a table it is an
        # ordinary stage again, and a stage read as finished may not turn
        # back into one that has to run (the supervisor's regression check).
        if name == "functional_evidence" and obj.get("skip_reason") == "gene_burden_skipped":
            canonical = "not_started"
            optional_stage = True

        # --stop-at: stages beyond the requested stop point never block/
        # represent the pipeline as incomplete, regardless of their real
        # (not_started) status.
        if kept_by_stop is not None and name not in kept_by_stop:
            canonical = "not_started"
            optional_stage = True

        # A STAGE BEHIND ONE THAT IS BEING RE-RUN IS NOT DONE.
        #
        # Its outputs exist, but they were computed from the earlier stage's previous outputs. Reported as
        # complete it would produce a state the supervisor refuses outright ("a later stage is complete
        # although an earlier one is not"), and a finished project in which an earlier stage came to read
        # incomplete could not be started again. The supervisor names these stages; here they are simply not
        # complete, and are run again in their turn.
        stale_stage = name in stale_stages and canonical in {"complete", "skipped"} and not optional_stage
        if stale_stage:
            canonical = "ready" if prerequisites_complete else "waiting"
        part_stage = bool(parts_to_run) and not stale_stage and canonical in {"complete", "skipped"} \
            and not optional_stage
        if part_stage:
            canonical = "ready" if prerequisites_complete else "waiting"

        stages[name] = {
            "label": labels.get(name, name),
            "status": obj.get("status", "unknown"),
            "state": canonical,
            "complete": bool(obj.get("complete")) and not stale_stage and not part_stage,
            "optional": optional_stage,
        }
        if stale_stage:
            stages[name]["stale"] = True
        if part_stage:
            stages[name]["unfinished_parts"] = sorted(parts_to_run)

        if optional_stage:
            continue

        if canonical not in {"complete", "skipped"}:
            if current_stage is None:
                current_stage = name
                next_stage = name
                pipeline_state = canonical
            if prerequisites_complete:
                ready_stages.append(name)
            unfinished[name] = frozenset(parts_to_run) if part_stage else None

    if current_stage is None:
        next_stage = None
        pipeline_state = "complete"

    return {
        "schema_version": 2,
        "state": pipeline_state,
        "complete": pipeline_state == "complete",
        "current_stage": current_stage,
        "next_stage": next_stage,
        # Every required stage not finished whose reads are, the next stage first -- what the
        # supervisor submits together (one at a time for a profile read as a line: only the next one is ready).
        "ready_stages": ready_stages,
        "stage_reads": {name: list(read) for name, read in reads.items()},
        "stages": stages,
        "pipeline_profile": resolved_profile.pipeline_profile.value,
        "segregation_profile": resolved_profile.segregation_profile,
        "alignment_profile": resolved_profile.alignment_profile.value,
        "manifest_profile": resolved_profile.manifest_profile.value,
        "stage_graph": list(resolved_profile.stage_graph),
        "updated": now_iso(),
    }

def project_stage_root(
    dedup_workdir: str | Optional[Path] = None,
    align_workdir: str | Optional[Path] = None,
) -> Optional[Path]:
    """The folder the derived stage folders (rna_quantification, macs, ...) sit in.

    One rule, used by this module and by the orchestrator. The
    orchestrator wrote the RNA stages beside --align-workdir while this module
    read them beside the dedup folder, so `--align-workdir <elsewhere>` wrote
    rna_quantification where the state reader never looked.
    """
    if dedup_workdir:
        return Path(dedup_workdir).parent
    if align_workdir:
        return Path(align_workdir).parent
    return None


def _hold_per_sample_stages_to_the_sheet(state: dict[str, Any], samples: Path) -> None:
    """A per-sample stage is finished only if every sample now in the sheet is done.

    Read QC, alignment and duplicate marking are held against the sheet, so a
    sample added to the sheet of a finished project is trimmed and aligned. A
    stage with a sample still to do reads "partial" and is run again as it
    stands -- its command skips the samples it has -- so nothing already made
    is made twice.

    Only a sample that is IN the sheet and not done counts. One taken out or
    commented out leaves these stages finished: its files are simply not used.
    A stage that does not name its samples is left as it was.

    Raises what reading the sheet raises.
    """
    from paralome.fastq.manifest import detect_manifest_mode, read_sample_inputs

    rows = read_sample_inputs(samples)
    mode = detect_manifest_mode(rows)
    with_reads = [row.sample_id for row in rows if row.fastq_r1]
    wanted = {
        # Samples given as BAMs are not trimmed or aligned here.
        "fastq_qc": with_reads,
        "alignment": with_reads,
        # Every sample is duplicate-marked, when all come in the same way.
        "deduplication": [row.sample_id for row in rows] if mode != "mixed" else [],
    }
    for name, sample_ids in wanted.items():
        stage = state.get(name)
        if not isinstance(stage, dict) or not stage.get("complete"):
            continue
        done = stage.get("done_sample_ids")
        if not isinstance(done, list):
            continue
        missing = [sample_id for sample_id in sample_ids if sample_id not in set(done)]
        if missing:
            state[name] = {
                **stage,
                "complete": False,
                "status": "partial",
                "samples_not_done": missing,
                "reason": "in the sample sheet and not yet done by this stage: " + ", ".join(missing),
            }


def build_project_state(
    *,
    samples: str | Optional[Path] = None,
    fastq_qc_workdir: str | Optional[Path] = None,
    align_workdir: str | Optional[Path] = None,
    dedup_workdir: str | Optional[Path] = None,
    variant_workdir: str | Optional[Path] = None,
    annotation_workdir: str | Optional[Path] = None,
    segregation_workdir: str | Optional[Path] = None,
    filter_workdir: str | Optional[Path] = None,
    report_workdir: str | Optional[Path] = None,
    rds: str | Optional[Path] = None,
    data_workdir_path: str | Optional[Path] = None,
    singlecell_differential_workdir: str | Optional[Path] = None,
    singlecell_motif_workdir: str | Optional[Path] = None,
    singlecell_rna_workdir: str | Optional[Path] = None,
    singlecell_rna_differential_workdir: str | Optional[Path] = None,
    profile: Optional[str] = None,
    segregation_profile: Optional[str] = None,
    skip_motif: bool = False,
    skip_motif_pairs: bool = False,
    skip_evidence: bool = False,
    no_genome_track: bool = False,
    no_trajectories: bool = False,
    # The fit's own settings this run asks for, compared with the fit's record.
    trajectory_root: Optional[str] = None,
    trajectory_fit_cells: Optional[str] = None,
    trajectory_dims: Optional[int] = None,
    trajectory_method: Optional[str] = None,
    trajectory_weld: Optional[Sequence[str]] = None,
    trajectory_max_routes: Optional[int] = None,
    stop_at: Optional[str] = None,
    run_plan: Optional[dict[str, Any]] = None,
    stale_stages: tuple[str, ...] = (),
    factors: tuple[str, ...] = (),
) -> dict[str, Any]:
    resolved_profile = resolve_pipeline_profile(profile, segregation_profile)
    project_root = project_stage_root(dedup_workdir, align_workdir)
    macs_workdir = project_root / "macs" if project_root else None
    peak_merge_workdir = project_root / "merged_peaks" if project_root else None
    peak_quantification_workdir = project_root / "peak_quantification" if project_root else None
    differential_accessibility_workdir = project_root / "differential_accessibility" if project_root else None
    motif_analysis_workdir = project_root / "motif_analysis" if project_root else None
    rna_quantification_workdir = project_root / "rna_quantification" if project_root else None
    differential_expression_workdir = project_root / "differential_expression" if project_root else None
    functional_evidence_workdir = project_root / "functional_evidence" if project_root else None
    # Beside the other DNA stage folders, <project>/gene_burden.
    gene_burden_workdir = project_root / "gene_burden" if project_root else None
    # <project>/statistics_output, or <project>/genotype_output on DNA (an older project's
    # <project>/browser_bundle); the zips in the omicview folder.
    statistics_output_workdir = project_root / "statistics_output" if project_root else None
    genotype_output_workdir = project_root / "genotype_output" if project_root else None
    single_cell = resolved_profile.pipeline_profile.value in ("snATAC", "scRNA")
    omicview = omicview_dir(project_root, data_workdir_path, single_cell)
    zips = (lambda stage: omicview / stage if omicview else None)
    # <project>/genome_tracks, its own project folder like Evidome's and the statistics output. SnATAC's
    # too (an older snATAC project's is <project>/atac/genome_tracks).
    genome_tracks_workdir = project_root / "genome_tracks" if project_root else None
    # <run>/trajectories, its zip in the run's omicview/trajectories.
    trajectories_workdir = project_root / "trajectories" if project_root else None
    state: dict[str, Any] = {
        "schema": "paralome.project_state.v2",
        "created": now_iso(),
        "updated": now_iso(),
        "project": {
            "samples_manifest": str(samples) if samples else None,
            "fastq_qc_workdir": str(fastq_qc_workdir) if fastq_qc_workdir else None,
            "align_workdir": str(align_workdir) if align_workdir else None,
            "dedup_workdir": str(dedup_workdir) if dedup_workdir else None,
            "macs_workdir": str(macs_workdir) if macs_workdir else None,
            "peak_merge_workdir": str(peak_merge_workdir) if peak_merge_workdir else None,
            "peak_quantification_workdir": str(peak_quantification_workdir) if peak_quantification_workdir else None,
            "differential_accessibility_workdir": str(differential_accessibility_workdir) if differential_accessibility_workdir else None,
            "motif_analysis_workdir": str(motif_analysis_workdir) if motif_analysis_workdir else None,
            "rna_quantification_workdir": str(rna_quantification_workdir) if rna_quantification_workdir else None,
            "differential_expression_workdir": str(differential_expression_workdir) if differential_expression_workdir else None,
            "functional_evidence_workdir": str(functional_evidence_workdir) if functional_evidence_workdir else None,
            "gene_burden_workdir": str(gene_burden_workdir) if gene_burden_workdir else None,
            "statistics_output_workdir": str(statistics_output_workdir) if statistics_output_workdir else None,
            "genotype_output_workdir": str(genotype_output_workdir) if genotype_output_workdir else None,
            # Where every zip of this run is.
            "omicview_dir": str(omicview) if omicview else None,
            "genome_tracks_workdir": str(genome_tracks_workdir) if genome_tracks_workdir else None,
            "trajectories_workdir": str(trajectories_workdir) if trajectories_workdir else None,
            "variant_workdir": str(variant_workdir) if variant_workdir else None,
            "annotation_workdir": str(annotation_workdir) if annotation_workdir else None,
            "segregation_workdir": str(segregation_workdir) if segregation_workdir else None,
            "filter_workdir": str(filter_workdir) if filter_workdir else None,
            "report_workdir": str(report_workdir) if report_workdir else None,
            "rds": str(rds) if rds else None,
            "data_workdir_path": str(data_workdir_path) if data_workdir_path else None,
            "singlecell_differential_workdir": str(singlecell_differential_workdir) if singlecell_differential_workdir else None,
            "singlecell_motif_workdir": str(singlecell_motif_workdir) if singlecell_motif_workdir else None,
            "singlecell_rna_workdir": str(singlecell_rna_workdir) if singlecell_rna_workdir else None,
            "singlecell_rna_differential_workdir": str(singlecell_rna_differential_workdir) if singlecell_rna_differential_workdir else None,
            "profile": resolved_profile.pipeline_profile.value,
            "pipeline_profile": resolved_profile.pipeline_profile.value,
            "segregation_profile": resolved_profile.segregation_profile,
            "alignment_profile": resolved_profile.alignment_profile.value,
            "manifest_profile": resolved_profile.manifest_profile.value,
            "skip_motif": bool(skip_motif),
            "skip_motif_pairs": bool(skip_motif_pairs),
            "skip_evidence": bool(skip_evidence),
            "no_genome_track": bool(no_genome_track),
            "no_trajectories": bool(no_trajectories),
            "stop_at": stop_at,
            # Stages whose outputs are on disk but were made from an
            # earlier stage that is being run again. See compute_pipeline_state.
            "stale_stages": sorted(stale_stages),
        },
        "inputs": {},
        "fastq_qc": summarize_fastq_qc(fastq_qc_workdir),
        "alignment": summarize_alignment(align_workdir),
        "deduplication": summarize_dedup(
            dedup_workdir, manifest_profile=resolved_profile.manifest_profile.value
        ),
        "macs_peak_calling": summarize_macs(macs_workdir),
        "peak_merge": summarize_peak_merge(peak_merge_workdir),
        "peak_quantification": summarize_peak_quantification(peak_quantification_workdir),
        "differential_accessibility": summarize_differential_accessibility(differential_accessibility_workdir),
        "motif_analysis_singles": summarize_motif_analysis_singles(motif_analysis_workdir),
        "motif_analysis_pairs": summarize_motif_analysis_pairs(motif_analysis_workdir),
        "rna_quantification": summarize_rna_quantification(rna_quantification_workdir),
        "differential_expression": summarize_differential_expression(differential_expression_workdir),
        "functional_evidence": summarize_functional_evidence(functional_evidence_workdir, zips("functional_evidence")),
        "variants": summarize_variants(variant_workdir),
        "annotation": summarize_annotation(annotation_workdir),
        "segregation": summarize_segregation(segregation_workdir),
        "filtering": summarize_filtering(filter_workdir, profile=resolved_profile.segregation_profile),
        "reporting": summarize_reporting(report_workdir),
        "gene_burden": summarize_gene_burden(gene_burden_workdir),
        "statistics_output": summarize_statistics_output(statistics_output_workdir, zips("statistics_output")),
        "genotype_output": summarize_genotype_output(genotype_output_workdir, zips("genotype_output")),
        # Its own stage again.
        "genome_tracks": summarize_genome_tracks(genome_tracks_workdir, zips("genome_tracks")),
        #
        "trajectories": summarize_trajectories(trajectories_workdir, zips("trajectories")),
        "singlecell_atac": summarize_singlecell_atac(data_workdir_path),
        "singlecell_differential": summarize_singlecell_differential(singlecell_differential_workdir),
        "singlecell_motif_analysis_singles": summarize_motif_analysis_singles(singlecell_motif_workdir),
        "singlecell_motif_analysis_pairs": summarize_motif_analysis_pairs(singlecell_motif_workdir),
        "singlecell_rna": summarize_singlecell_rna(singlecell_rna_workdir or data_workdir_path),
        "singlecell_rna_differential": summarize_singlecell_rna_differential(singlecell_rna_differential_workdir),
    }
    # ScRNA's singlecell_rna fits the trajectories as its last step (profiles.STAGE_PARTS). Finished
    # without this run's fit, only that part runs again, and only what reads it is stale.
    if (resolved_profile.pipeline_profile.value == "scRNA" and not no_trajectories and project_root
            and isinstance(state.get("singlecell_rna"), dict) and state["singlecell_rna"].get("complete")):
        from paralome.singlecell.trajectories import fit_settings_asked, fit_unfinished

        why = fit_unfinished(project_root / "trajectories" / TRAJECTORY_FIT_FOLDER,
                             omicview_dir(project_root, data_workdir_path or singlecell_rna_workdir, True),
                             asked=fit_settings_asked(trajectory_root, trajectory_fit_cells, trajectory_dims,
                                                      trajectory_method, trajectory_weld, trajectory_max_routes))
        if why:
            state["singlecell_rna"] = {**state["singlecell_rna"], "unfinished_parts": ["trajectory_fit"],
                                       "out_of_date": f"its trajectory fit: {why}"}

    if samples:
        try:
            from paralome.pipeline.dispatcher import build_dispatch_plan
            plan = build_dispatch_plan(
                samples=Path(samples),
                fastq_qc_workdir=Path(fastq_qc_workdir) if fastq_qc_workdir else None,
                align_workdir=Path(align_workdir) if align_workdir else None,
                variant_workdir=Path(variant_workdir) if variant_workdir else None,
                manifest_profile=resolved_profile.manifest_profile.value,
                stage_graph=resolved_profile.stage_graph,
                # The --factor columns stand in for `condition`.
                factors=tuple(factors),
            )
            state["inputs"]["dispatch_plan"] = plan.to_dict()
            # A sheet that lists BAMs and no FASTQs has nothing to
            # trim or align. Both stages read as not done, so the run was sent
            # to fastq_qc with a sheet that has no reads in it. They do not
            # apply to this sheet, and the first stage is the one after them.
            #
            # And that holds when the project already has a finished
            # alignment. The sheet is the input the run was given; with the
            # earlier alignment left "complete" the next stage was handed its
            # BAMs and the sheet was ignored without a word.
            if plan.input_manifest.mode == "bam":
                for name in ("fastq_qc", "alignment"):
                    if name in resolved_profile.stage_graph:
                        state[name] = {
                            **state[name],
                            "status": "not_applicable",
                            "complete": False,
                            "results_not_used": bool(state[name].get("complete")),
                            "reason": "the sample sheet lists BAM files, not FASTQ files",
                        }
        except Exception as exc:
            state["inputs"]["dispatch_error"] = str(exc)
        try:
            _hold_per_sample_stages_to_the_sheet(state, Path(samples))
        except Exception as exc:
            # An unreadable sheet is reported by the run itself; here it is on record.
            state["inputs"].setdefault("dispatch_error", str(exc))

    if "gene_burden" in resolved_profile.stage_graph:
        _functional_evidence_follows_gene_burden(state)

    if run_plan is not None:
        state["orchestrator"] = {"last_run_plan": run_plan}

    state["overall"] = compute_overall_status(state)
    state["pipeline"] = compute_pipeline_state(state)
    return state


def compute_overall_status(state: dict[str, Any]) -> dict[str, Any]:
    project = state.get("project", {}) if isinstance(state.get("project"), dict) else {}
    resolved_profile = resolve_pipeline_profile(
        project.get("pipeline_profile") or project.get("profile"),
        project.get("segregation_profile"),
    )
    summary = {}
    for stage in resolved_profile.stage_graph:
        obj = state.get(stage, {}) if isinstance(state.get(stage), dict) else {}
        summary[stage] = obj.get("status", "unknown")

    if resolved_profile.pipeline_profile.value == "scRNA" and summary.get("singlecell_rna_differential") == "complete":
        next_action = "scRNA workflow is complete through differential expression."
    elif resolved_profile.pipeline_profile.value == "scRNA" and summary.get("singlecell_rna") == "complete":
        next_action = "Run single-cell RNA differential-expression statistics."
    elif resolved_profile.pipeline_profile.value == "scRNA":
        next_action = "Aggregate the Seurat single-cell RNA object into pseudobulk matrices."
    elif resolved_profile.pipeline_profile.value == "snATAC" and summary.get("singlecell_motif_analysis_pairs") == "complete":
        next_action = "snATAC workflow is complete through motif analysis."
    elif resolved_profile.pipeline_profile.value == "snATAC" and summary.get("singlecell_motif_analysis_singles") == "complete":
        next_action = "Run single-cell motif-pair co-occurrence analysis."
    elif resolved_profile.pipeline_profile.value == "snATAC" and summary.get("singlecell_differential") == "complete":
        next_action = "Run checkpointed single-cell motif preparation, scanning, and global finalization."
    elif resolved_profile.pipeline_profile.value == "snATAC" and summary.get("singlecell_atac") == "complete":
        next_action = "Run single-cell stratified differential-accessibility statistics."
    elif resolved_profile.pipeline_profile.value == "snATAC":
        next_action = "Aggregate the Seurat/Signac object into pseudobulk matrices."
    elif resolved_profile.pipeline_profile.value == "ATAC" and summary.get("motif_analysis_pairs") == "complete":
        next_action = "ATAC workflow is complete through motif analysis."
    elif resolved_profile.pipeline_profile.value == "ATAC" and summary.get("motif_analysis_singles") == "complete":
        next_action = "Run motif-pair co-occurrence analysis."
    elif resolved_profile.pipeline_profile.value == "ATAC" and summary.get("differential_accessibility") == "complete":
        next_action = "Run checkpointed motif preparation, scanning, and global finalization."
    elif resolved_profile.pipeline_profile.value == "ATAC" and summary.get("peak_quantification") == "complete":
        next_action = "Run differential-accessibility statistics."
    elif resolved_profile.pipeline_profile.value == "ATAC" and summary.get("peak_merge") == "complete":
        next_action = "Quantify fragments across the cohort peak atlas."
    elif resolved_profile.pipeline_profile.value == "RNA" and summary.get("functional_evidence") == "complete":
        next_action = "Bulk RNA-seq workflow is complete through functional evidence (Evidome)."
    elif resolved_profile.pipeline_profile.value == "RNA" and summary.get("functional_evidence") == "skipped":
        next_action = "Bulk RNA-seq workflow is complete through DEG reporting (functional evidence skipped: see status)."
    elif resolved_profile.pipeline_profile.value == "RNA" and summary.get("differential_expression") == "complete":
        next_action = "Run functional evidence (Evidome) on the differential statistics."
    elif resolved_profile.pipeline_profile.value == "RNA" and summary.get("rna_quantification") == "complete":
        next_action = "Run checkpointed differential-expression statistics and reporting."
    elif resolved_profile.pipeline_profile.value == "RNA" and summary.get("alignment") in {"complete", "not_applicable"}:
        next_action = "Quantify gene expression and calculate native FPKM matrices."
    elif summary.get("macs_peak_calling") == "complete":
        next_action = "Merge MACS3 peaks into a cohort atlas."
    elif summary.get("deduplication") == "complete" and resolved_profile.pipeline_profile.value == "ATAC":
        next_action = "Run per-sample MACS3 peak calling."
    # DNA ends with gene burden and Evidome on its table.
    elif "gene_burden" in summary and summary.get("functional_evidence") == "complete":
        next_action = ("DNA workflow is complete through functional evidence (Evidome) on the gene-burden "
                       "table. Open the genotype output in OmicView's Variant Browser.")
    elif "gene_burden" in summary and summary.get("functional_evidence") == "skipped":
        burden = state.get("gene_burden") if isinstance(state.get("gene_burden"), dict) else {}
        next_action = (
            "DNA workflow is complete through filtering; "
            + (f"gene burden skipped: {burden.get('reason')}" if summary.get("gene_burden") == "skipped"
               else "functional evidence skipped: see status.")
        )
    elif "gene_burden" in summary and summary.get("gene_burden") == "complete":
        next_action = "Run functional evidence (Evidome) on the gene-burden table."
    elif "gene_burden" in summary and summary.get("filtering") == "complete":
        next_action = "Run gene burden against gnomAD."
    elif summary.get("segregation") == "complete":
        next_action = "Run filtering."
    elif summary.get("annotation") == "complete":
        next_action = "Run segregation."
    elif summary.get("variants") == "complete":
        next_action = "Run annotation."
    elif summary.get("deduplication") == "complete":
        next_action = "Run variant calling."
    elif summary.get("alignment") in {"complete", "not_applicable"}:
        next_action = "Run duplicate marking."
    elif summary.get("fastq_qc") == "complete":
        next_action = "Run alignment."
    elif summary.get("fastq_qc") == "not_provided":
        next_action = "Provide workdirs or run pipeline inspection."
    else:
        next_action = "Run FASTQ QC."

    # The checks above read the last stages first, so a finished
    # later stage hid an earlier one that reads incomplete (an out-of-date
    # method, a new sample) and the project was called complete. The first
    # unfinished stage in the stage order, as the pipeline runs it, answers.
    pipeline = compute_pipeline_state(state)
    first = pipeline.get("next_stage")
    if first and not pipeline.get("complete"):
        order = list(pipeline.get("stage_graph") or [])
        stage_rows = pipeline.get("stages") or {}
        later_done = any(
            summary.get(name) in {"complete", "skipped"} and not (stage_rows.get(name) or {}).get("optional")
            for name in order[order.index(first) + 1:]
        ) if first in order else False
        if later_done:
            label = STAGE_LABELS.get(first, first)
            if pipeline.get("state") == "running":
                next_action = f"Wait for {label} to finish; the stages after it are then run again."
            else:
                reads = summary.get(first) or "incomplete"
                why = ("its results are out of date" if reads in {"complete", "skipped"}
                       else f"it reads {reads}")
                next_action = f"Run {label}: {why}, so the stages after it are run again behind it."

    return {"stages": summary, "next_action": next_action}


def write_project_state(workdir: Union[str, Path], state: dict[str, Any], *, filename: str = "project_state.json") -> Path:
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    path = wd / filename
    state = dict(state)
    state["updated"] = now_iso()
    return write_json(path, state)

