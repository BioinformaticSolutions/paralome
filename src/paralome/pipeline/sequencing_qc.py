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
from pathlib import Path
from typing import Dict, Optional, Union

# Columns published into the outlier-QC tab/sheet for every sample, regardless of
# whether the upstream fastq_qc/alignment summaries were found. Fixed column order
# keeps the TSV header and the Sample_QC workbook sheet stable even when some
# samples (or an entire upstream summary) are missing data.
SEQUENCING_QC_FIELDS = [
    "raw_reads",
    "clean_reads",
    "pct_reads_surviving",
    "duplication_rate",
    "star_input_reads",
    "uniquely_mapped_reads",
    "uniquely_mapped_percent",
    "total_mapped_reads",
    "total_mapped_percent",
]


def load_sequencing_qc(project_root: Optional[Union[str, Path]]) -> Dict[str, Dict[str, object]]:
    """Load per-sample sequencing QC (fastp read counts + STAR alignment rates).

    Best-effort, read-only cross-reference for the differential-expression and
    differential-accessibility "outlier QC" tabs: a reviewer can use raw/clean
    read depth and mapping rate to help understand *why* a sample was flagged as
    a statistical PCA/correlation outlier.

    ``fastq_qc`` and ``alignment`` (workdir name ``star``) are sibling stage
    workdirs under the same project root as the calling stage,
    so this only reads each stage's already-published TSV summary -- it never
    touches another stage's checkpoint file. Returns an empty dict, or partial
    per-sample dicts, if either upstream summary is missing or ``project_root``
    can't be resolved; callers should treat missing sequencing QC as normal
    (e.g. a matrix supplied from outside the standard project layout), not an
    error.
    """
    qc: Dict[str, Dict[str, object]] = {}
    if project_root is None:
        return qc
    root = Path(project_root)

    fastq_summary = root / "fastq_qc" / "results" / "fastq_qc_summary.tsv"
    if fastq_summary.exists():
        try:
            with fastq_summary.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    sample_id = str(row.get("sample_id") or "").strip()
                    if not sample_id:
                        continue
                    bucket = qc.setdefault(sample_id, {})
                    bucket["raw_reads"] = row.get("raw_reads", ".")
                    bucket["clean_reads"] = row.get("clean_reads", ".")
                    bucket["pct_reads_surviving"] = row.get("pct_reads_surviving", ".")
                    bucket["duplication_rate"] = row.get("duplication_rate", ".")
        except OSError:
            pass

    # The alignment folder: <project>/alignment for a run with --project (derive_project_workdirs; an
    # older project has <project>/star), or "star_align" beside a sample sheet when there is no project.
    # All are read, the project layout first. Its table is alignment_metrics.tsv
    # (star_alignment_metrics.tsv in an older project).
    star_summary = next(
        (root / folder / "results" / name
         for folder in ("alignment", "star", "star_align")
         for name in ("alignment_metrics.tsv", "star_alignment_metrics.tsv")
         if (root / folder / "results" / name).exists()),
        root / "alignment" / "results" / "alignment_metrics.tsv",
    )
    if star_summary.exists():
        try:
            with star_summary.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    sample_id = str(row.get("sample_id") or "").strip()
                    if not sample_id:
                        continue
                    bucket = qc.setdefault(sample_id, {})
                    bucket["star_input_reads"] = row.get("input_reads", ".")
                    bucket["uniquely_mapped_reads"] = row.get("uniquely_mapped_reads", ".")
                    bucket["uniquely_mapped_percent"] = row.get("uniquely_mapped_percent", ".")
                    bucket["total_mapped_reads"] = row.get("total_mapped_reads", ".")
                    bucket["total_mapped_percent"] = row.get("total_mapped_percent", ".")
        except OSError:
            pass

    return qc


def sequencing_qc_row(qc: Dict[str, Dict[str, object]], sample_id: str) -> Dict[str, object]:
    """Return a fixed-column sequencing-QC row for one sample, "." for anything missing."""
    sample_qc = qc.get(sample_id, {})
    return {field: sample_qc.get(field, ".") for field in SEQUENCING_QC_FIELDS}


# ---------------------------------------------------------------------------
# The sequence QC table: raw reads to duplicate-removed reads, one unit
# ---------------------------------------------------------------------------

#: The columns, in order. Every count is in reads (a pair is two), whatever the step reported in:
#: fastp counts reads, STAR and the BWA table count pairs, flagstat counts primary reads.
#: mapped_percent, uniquely_mapped_percent and multimapped_percent are of the trimmed reads (the aligner's
#: input); properly_paired_percent and duplicate_percent of the mapped reads; every other percent of the raw.
SEQUENCE_QC_COLUMNS = [
    "sample_id", "paired",
    "raw_reads",
    "trimmed_reads", "trimmed_percent",
    "mapped_reads", "mapped_percent",
    "uniquely_mapped_reads", "uniquely_mapped_percent",
    "multimapped_reads", "multimapped_percent",
    "properly_paired_reads", "properly_paired_percent",
    "duplicate_reads", "duplicate_percent",
    "deduplicated_reads", "deduplicated_percent",
]


def _int(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _alignment_summary(root: Path) -> Optional[Path]:
    """The mapping table (alignment/, star/ or star_align/; alignment_metrics.tsv or star_alignment_metrics.tsv)."""
    return next((root / folder / "results" / name
                 for folder in ("alignment", "star", "star_align")
                 for name in ("alignment_metrics.tsv", "star_alignment_metrics.tsv")
                 if (root / folder / "results" / name).is_file()), None)


def _rows(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open(newline="") as handle:
            return {str(r.get("sample_id") or "").strip(): r
                    for r in csv.DictReader(handle, delimiter="\t") if str(r.get("sample_id") or "").strip()}
    except OSError:
        return {}


def _fastp_paired(row: Dict[str, str]) -> Optional[bool]:
    """Whether fastp read a pair (its JSON has read2 counts), None when the JSON is not there."""
    import json

    path = Path(str(row.get("fastp_json") or ""))
    try:
        data = json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return None
    if data is None:
        return None
    return bool(data.get("read2_before_filtering"))


def _bam_counts(bam: Path) -> Optional[dict]:
    """A duplicate-marked BAM's primary-read counts: the flagstat its task wrote beside it, else one threaded
    ``samtools flagstat`` pass, kept beside the BAM (``<bam>.flagstat.json``) for the next build."""
    from paralome.dedup.counts import FLAGSTAT_SIDECAR, primary_counts, recorded_flagstat, run_flagstat

    data = recorded_flagstat(bam)
    if data is None:
        if not bam.is_file():
            return None
        try:
            from paralome.dedup.markdup import require_tool

            data = run_flagstat(bam, require_tool("samtools"), threads=4)
        except Exception:                                  # no samtools, an unreadable BAM: the cells stay empty
            return None
        try:
            import json

            Path(str(bam) + FLAGSTAT_SIDECAR).write_text(json.dumps(data, indent=1) + "\n")
        except OSError:
            pass
    return primary_counts(data)


def _dedup_rows(root: Path) -> Dict[str, Dict[str, object]]:
    """Each sample's primary-read counts in its duplicate-marked BAM.

    From the dedup summary when it has them; else from the BAM itself (_bam_counts), the samples in
    parallel -- an older summary has no primary counts, and a rerun leaves a finished dedup stage alone.
    With --remove-duplicates the BAM no longer holds them, so only the summary can give them."""
    import json

    from paralome.dedup.counts import in_parallel

    path = root / "dedup" / "results" / "dedup_summary.json"
    try:
        summary = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    removed = bool(summary.get("remove_duplicates"))
    out: Dict[str, Dict[str, object]] = {}
    pending = []
    for m in summary.get("sample_metrics") or []:
        if not isinstance(m, dict) or not m.get("sample_id"):
            continue
        sid = str(m["sample_id"])
        if "properly_paired_reads" in m and m.get("duplicate_reads") is not None:
            out[sid] = {"mapped": _int(m.get("total_reads")), "duplicates": _int(m.get("duplicate_reads")),
                        "properly_paired": _int(m.get("properly_paired_reads")), "paired": _int(m.get("paired_reads"))}
        elif not removed:
            bam = Path(str(m.get("output_bam") or "")) if m.get("output_bam") else root / "dedup" / "bam" / f"{sid}.markdup.bam"
            pending.append((sid, bam))
    # A sample the summary does not name (no summary, or one written before it was marked): its BAM, if there.
    named = {str(m.get("sample_id")) for m in summary.get("sample_metrics") or [] if isinstance(m, dict)}
    if not removed and (root / "dedup" / "bam").is_dir():
        for bam in sorted((root / "dedup" / "bam").glob("*.markdup.bam")):
            sid = bam.name[: -len(".markdup.bam")]
            if sid not in named:
                pending.append((sid, bam))
    for (sid, _bam), c in zip(pending, in_parallel(pending, lambda item: _bam_counts(item[1]))):
        if c is not None:
            out[sid] = {"mapped": c["mapped_reads"], "duplicates": c["primary_duplicates"],
                        "properly_paired": c["properly_paired_reads"], "paired": c["paired_reads"]}
    return out


def _logged_row(root: Path, sid: str) -> Optional[Dict[str, str]]:
    """A sample's mapping row from what its alignment task wrote, for a mapping table that has none: STAR's
    Log.final.out, else BWA's flagstat and MAPQ>=1 count (align.metrics). An older table may not name a fused
    run's BAMs, and a project may have no table at all; a rerun leaves a finished alignment stage, and its
    table, alone."""
    from paralome.align.metrics import _log_path_for_sample, parse_bwa_metrics, parse_star_log_final

    for folder in ("alignment", "star", "star_align"):
        wd = root / folder
        if not wd.is_dir():
            continue
        log = _log_path_for_sample(wd, sid)
        if log.is_file():
            try:
                return parse_star_log_final(log, sid).to_row()
            except Exception:
                pass
        m = parse_bwa_metrics(wd, sid) if (wd / "logs").is_dir() else None
        if m is not None:
            return m.to_row()
    return None


def _percent(part: Optional[int], whole: Optional[int]) -> Optional[float]:
    return round(100.0 * part / whole, 2) if part is not None and whole else None


def sequence_qc_rows(project_root: Optional[Union[str, Path]]) -> list:
    """The sequence QC table of a project Paralome trimmed and aligned (DNA, RNA, ATAC): one row per sample,
    SEQUENCE_QC_COLUMNS, None for what a step did not record (RNA has no duplicate marking). [] when the
    project has neither a FASTQ QC nor a mapping table."""
    if project_root is None:
        return []
    root = Path(project_root)
    fastq = _rows(root / "fastq_qc" / "results" / "fastq_qc_summary.tsv")
    aligned = _rows(_alignment_summary(root))
    dedup = _dedup_rows(root)
    rows = []
    for sid in sorted(set(fastq) | set(aligned) | set(dedup)):
        f, a, d = fastq.get(sid, {}), aligned.get(sid, {}), dedup.get(sid, {})
        if _int(a.get("uniquely_mapped_reads")) is None:
            a = _logged_row(root, sid) or a
        raw, trimmed = _int(f.get("raw_reads")), _int(f.get("clean_reads"))
        input_units = _int(a.get("input_reads"))
        # A pair is two reads: fastp's JSON says, else the duplicate-marked BAM's paired reads, else the
        # trimmed reads being twice what the aligner took in.
        paired = _fastp_paired(f) if f else None
        if paired is None and d.get("paired") is not None:
            paired = bool(d.get("paired"))
        if paired is None and trimmed and input_units:
            paired = round(trimmed / input_units) == 2
        per = 2 if paired else 1
        unique = _int(a.get("uniquely_mapped_reads"))
        multi = _int(a.get("multimapped_reads"))
        mapped = None if unique is None and multi is None else ((unique or 0) + (multi or 0)) * per
        unique = None if unique is None else unique * per
        multi = None if multi is None else multi * per
        if mapped is None and d.get("mapped") is not None:
            # No mapping row nor log: the duplicate-marked BAM's mapped reads (no unique/multi split).
            mapped = d.get("mapped")
        if trimmed is None and input_units is not None:
            trimmed = input_units * per
        marked_mapped, dups = d.get("mapped"), d.get("duplicates")
        dedup_reads = (marked_mapped - dups) if marked_mapped is not None and dups is not None else None
        proper = d.get("properly_paired") if paired else None
        rows.append({
            "sample_id": sid, "paired": "yes" if paired else ("no" if paired is not None else None),
            "raw_reads": raw,
            "trimmed_reads": trimmed, "trimmed_percent": _percent(trimmed, raw),
            "mapped_reads": mapped, "mapped_percent": _percent(mapped, trimmed),
            "uniquely_mapped_reads": unique, "uniquely_mapped_percent": _percent(unique, trimmed),
            "multimapped_reads": multi, "multimapped_percent": _percent(multi, trimmed),
            "properly_paired_reads": proper, "properly_paired_percent": _percent(proper, marked_mapped),
            "duplicate_reads": dups, "duplicate_percent": _percent(dups, marked_mapped),
            "deduplicated_reads": dedup_reads, "deduplicated_percent": _percent(dedup_reads, raw),
        })
    return rows
