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
from paralome.checkpoints import stage_checkpoint_path

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class StarSampleMetrics:
    sample_id: str
    log_final: str
    log_exists: bool
    input_reads: Optional[int] = None
    average_input_read_length: Optional[float] = None
    uniquely_mapped_reads: Optional[int] = None
    uniquely_mapped_percent: Optional[float] = None
    average_mapped_length: Optional[float] = None
    mismatch_rate_per_base_percent: Optional[float] = None
    deletion_rate_per_base_percent: Optional[float] = None
    insertion_rate_per_base_percent: Optional[float] = None
    multimapped_reads: Optional[int] = None
    multimapped_percent: Optional[float] = None
    too_many_loci_reads: Optional[int] = None
    too_many_loci_percent: Optional[float] = None
    unmapped_too_many_mismatches_percent: Optional[float] = None
    unmapped_too_short_percent: Optional[float] = None
    unmapped_other_percent: Optional[float] = None

    @property
    def total_mapped_reads(self) -> Optional[int]:
        vals = [self.uniquely_mapped_reads, self.multimapped_reads]
        if all(v is None for v in vals):
            return None
        return sum(v or 0 for v in vals)

    @property
    def total_mapped_percent(self) -> Optional[float]:
        vals = [self.uniquely_mapped_percent, self.multimapped_percent]
        if all(v is None for v in vals):
            return None
        return sum(v or 0.0 for v in vals)

    def to_row(self) -> dict[str, str]:
        d = asdict(self)
        d["total_mapped_reads"] = self.total_mapped_reads
        d["total_mapped_percent"] = self.total_mapped_percent
        row: dict[str, str] = {}
        for k, v in d.items():
            if v is None:
                row[k] = "."
            elif isinstance(v, float):
                row[k] = f"{v:.4f}"
            else:
                row[k] = str(v)
        return row


_METRIC_MAP = {
    "Number of input reads": ("input_reads", int),
    "Average input read length": ("average_input_read_length", float),
    "Uniquely mapped reads number": ("uniquely_mapped_reads", int),
    "Uniquely mapped reads %": ("uniquely_mapped_percent", float),
    "Average mapped length": ("average_mapped_length", float),
    "Mismatch rate per base, %": ("mismatch_rate_per_base_percent", float),
    "Deletion rate per base": ("deletion_rate_per_base_percent", float),
    "Insertion rate per base": ("insertion_rate_per_base_percent", float),
    "Number of reads mapped to multiple loci": ("multimapped_reads", int),
    "% of reads mapped to multiple loci": ("multimapped_percent", float),
    "Number of reads mapped to too many loci": ("too_many_loci_reads", int),
    "% of reads mapped to too many loci": ("too_many_loci_percent", float),
    "% of reads unmapped: too many mismatches": ("unmapped_too_many_mismatches_percent", float),
    "% of reads unmapped: too short": ("unmapped_too_short_percent", float),
    "% of reads unmapped: other": ("unmapped_other_percent", float),
}


_FIELDNAMES = [
    "sample_id",
    "log_final",
    "log_exists",
    "input_reads",
    "average_input_read_length",
    "uniquely_mapped_reads",
    "uniquely_mapped_percent",
    "multimapped_reads",
    "multimapped_percent",
    "total_mapped_reads",
    "total_mapped_percent",
    "too_many_loci_reads",
    "too_many_loci_percent",
    "unmapped_too_many_mismatches_percent",
    "unmapped_too_short_percent",
    "unmapped_other_percent",
    "average_mapped_length",
    "mismatch_rate_per_base_percent",
    "deletion_rate_per_base_percent",
    "insertion_rate_per_base_percent",
]


#: The per-sample mapping table (either aligner); a project aligned by an older Paralome has star_alignment_metrics.*.
METRICS_TSV = "alignment_metrics.tsv"
METRICS_JSON = "alignment_metrics.json"


def _clean_key(key: str) -> str:
    return re.sub(r"\s+", " ", key.strip())


def _parse_numeric(value: str, typ: type) -> int | Optional[float]:
    value = value.strip().replace("%", "")
    if value in ("", ".", "NA", "N/A"):
        return None
    try:
        if typ is int:
            return int(float(value))
        return float(value)
    except ValueError:
        return None


def parse_star_log_final(path: Union[str, Path], sample_id: str) -> StarSampleMetrics:
    path = Path(path)
    m = StarSampleMetrics(sample_id=sample_id, log_final=str(path), log_exists=path.exists() and path.stat().st_size > 0)
    if not m.log_exists:
        return m

    with path.open(errors="replace") as fh:
        for raw in fh:
            if "|" not in raw:
                continue
            key, val = raw.split("|", 1)
            key = _clean_key(key)
            if key not in _METRIC_MAP:
                continue
            attr, typ = _METRIC_MAP[key]
            parsed = _parse_numeric(val, typ)
            if parsed is not None:
                setattr(m, attr, parsed)
    return m


def parse_bwa_metrics(workdir: Path, sample_id: str) -> Optional[StarSampleMetrics]:
    """A BWA sample's row of the same table, from what its task wrote (align.bwa.metrics_lines); None if none.

    Counted in read pairs when the library is paired, as STAR counts them: input = primary reads, uniquely
    mapped = primary mapped at MAPQ >= 1, multimapped = the rest of the primary mapped reads.
    """
    from paralome.align.bwa import FLAGSTAT_SUFFIX, MAPQ1_SUFFIX

    flagstat = Path(workdir) / "logs" / f"{sample_id}{FLAGSTAT_SUFFIX}"
    mapq1 = Path(workdir) / "logs" / f"{sample_id}{MAPQ1_SUFFIX}"
    if not (flagstat.is_file() and flagstat.stat().st_size and mapq1.is_file()):
        return None
    try:
        passed = json.loads(flagstat.read_text()).get("QC-passed reads") or {}
        unique = int(mapq1.read_text().strip() or 0)
    except (OSError, ValueError):
        return None
    per = 2 if int(passed.get("paired in sequencing") or 0) else 1
    primary = int(passed.get("primary") or 0) // per
    # The pair filter (align.bwa.FILTER_AWK) leaves out of the BAM what STAR would not have written;
    # those pairs are still the aligner's input, and count as not mapped, as STAR's do.
    from paralome.align.bwa import FILTER_SUFFIX

    counts = Path(workdir) / "logs" / f"{sample_id}{FILTER_SUFFIX}"
    try:
        filtered = dict(line.split("\t", 1) for line in counts.read_text().splitlines() if "\t" in line)
        primary = int(filtered["pairs_in"])
    except (OSError, KeyError, ValueError):
        pass
    mapped = int(passed.get("primary mapped") or 0) // per
    unique //= per
    m = StarSampleMetrics(sample_id=sample_id, log_final=str(flagstat), log_exists=True)
    m.input_reads = primary
    m.uniquely_mapped_reads = unique
    m.multimapped_reads = max(0, mapped - unique)
    if primary:
        m.uniquely_mapped_percent = round(100.0 * unique / primary, 2)
        m.multimapped_percent = round(100.0 * m.multimapped_reads / primary, 2)
    return m


def _read_aligned_manifest(workdir: Path) -> list[dict[str, str]]:
    manifest = workdir / "inputs" / "aligned_samples.csv"
    if not manifest.exists():
        return []
    with manifest.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _sample_ids_from_checkpoint(workdir: Path) -> list[str]:
    path = stage_checkpoint_path(workdir, "alignment")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    samples = data.get("samples") or {}
    if isinstance(samples, dict):
        return list(samples.keys())
    return []


def _sample_ids(workdir: Path) -> list[str]:
    ids: list[str] = []
    for row in _read_aligned_manifest(workdir):
        sid = row.get("sample_id") or row.get("sample") or row.get("id")
        if sid:
            ids.append(str(sid))
    if ids:
        return ids
    ids = _sample_ids_from_checkpoint(workdir)
    if ids:
        return ids
    return sorted(p.name.replace(".Log.final.out", "") for p in (workdir / "temp").glob("*.Log.final.out"))


def _log_path_for_sample(workdir: Path, sample_id: str) -> Path:
    candidates = [
        workdir / "temp" / f"{sample_id}.Log.final.out",
        workdir / "logs" / f"{sample_id}.Log.final.out",
        workdir / "logs" / f"{sample_id}.STAR.Log.final.out",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _samples_to_be_replaced(workdir: Path) -> set[str]:
    """Samples `align star --overwrite` marked and no task has aligned again."""
    from paralome.align.verify import TO_BE_REPLACED

    path = stage_checkpoint_path(workdir, "alignment")
    try:
        samples = json.loads(path.read_text()).get("samples") or {}
    except Exception:
        return set()
    if not isinstance(samples, dict):
        return set()
    return {
        str(sample_id) for sample_id, entry in samples.items()
        if isinstance(entry, dict) and isinstance(entry.get("metadata"), dict)
        and entry["metadata"].get(TO_BE_REPLACED)
    }


def _samples_without_a_bam(workdir: Path, ids: list[str]) -> set[str]:
    """Samples whose BAM or index is missing or empty.

    An earlier run's Log.final.out stays in temp/ and logs/ when the BAM is
    removed to align a sample again, and was counted as this run's when the
    new alignment had not finished. Without its BAM and index a sample has no
    result yet, whatever log is beside it.
    """
    bams: dict[str, str] = {}
    for row in _read_aligned_manifest(workdir):
        sid = row.get("sample_id") or row.get("sample") or row.get("id")
        bam = row.get("bam") or row.get("bam_path")
        if sid and bam:
            bams[str(sid)] = bam
    try:
        samples = json.loads(stage_checkpoint_path(workdir, "alignment").read_text()).get("samples") or {}
    except Exception:
        samples = {}
    def there(bam: str) -> bool:
        bam_path = Path(bam)
        if not bam_path.is_absolute() and not bam_path.is_file():
            # A relative path in the aligned manifest is as the run
            # wrote it -- from where the run was started, the workdir included
            # (`--project TRT_vs_CON` writes TRT_vs_CON/star/bam/...). Joined
            # to the workdir again it named a file that is never there, so
            # every sample read as unaligned and the status showed no mapping
            # rate. Kept as the fallback for a path written inside the workdir.
            bam_path = workdir / bam_path
        bai_path = Path(str(bam_path) + ".bai")
        return all(p.is_file() and p.stat().st_size > 0 for p in (bam_path, bai_path))

    missing: set[str] = set()
    for sid in ids:
        entry = samples.get(sid) if isinstance(samples, dict) else None
        outputs = entry.get("outputs") if isinstance(entry, dict) else None
        # The BAM the sample's task recorded comes first. A fused run's BAM is the duplicate-marked
        # one in the deduplication folder; the aligned manifest still names bam/<sample>.sorted.bam, never made,
        # and every fused run's mapping table came out empty.
        candidates = [b for b in ((outputs.get("bam") if isinstance(outputs, dict) else None), bams.get(sid)) if b]
        if not any(there(b) for b in (candidates or [str(workdir / "bam" / f"{sid}.sorted.bam")])):
            missing.add(sid)
    return missing


def collect_star_metrics(workdir: Union[str, Path], *, write_outputs: bool = True) -> dict[str, Any]:
    workdir = Path(workdir)
    ids = _sample_ids(workdir)
    # A sample still to be aligned again under --overwrite keeps its
    # earlier Log.final.out until its task has finished, and that log was
    # counted as this run's mapping rate. Such a sample has no log yet.
    pending = _samples_to_be_replaced(workdir) | _samples_without_a_bam(workdir, ids)
    metrics = [
        StarSampleMetrics(sample_id=sid, log_final=str(_log_path_for_sample(workdir, sid)), log_exists=False)
        if sid in pending else (parse_bwa_metrics(workdir, sid)
                                or parse_star_log_final(_log_path_for_sample(workdir, sid), sid))
        for sid in ids
    ]

    total_input = sum(m.input_reads or 0 for m in metrics)
    total_unique = sum(m.uniquely_mapped_reads or 0 for m in metrics)
    total_multi = sum(m.multimapped_reads or 0 for m in metrics)
    total_mapped = total_unique + total_multi
    weighted_unique_pct = (100.0 * total_unique / total_input) if total_input else None
    weighted_mapped_pct = (100.0 * total_mapped / total_input) if total_input else None

    summary = {
        "workdir": str(workdir),
        "samples": len(metrics),
        "samples_with_logs": sum(1 for m in metrics if m.log_exists),
        "complete": len(metrics) > 0 and all(m.log_exists for m in metrics),
        "pending_sample_ids": [sid for sid in ids if sid in pending],
        "totals": {
            "input_reads": total_input,
            "uniquely_mapped_reads": total_unique,
            "multimapped_reads": total_multi,
            "total_mapped_reads": total_mapped,
            "weighted_uniquely_mapped_percent": weighted_unique_pct,
            "weighted_total_mapped_percent": weighted_mapped_pct,
        },
        "samples_detail": [asdict(m) | {"total_mapped_reads": m.total_mapped_reads, "total_mapped_percent": m.total_mapped_percent} for m in metrics],
    }

    if write_outputs:
        results = workdir / "results"
        results.mkdir(parents=True, exist_ok=True)
        tsv = results / METRICS_TSV
        with tsv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for m in metrics:
                writer.writerow({k: m.to_row().get(k, ".") for k in _FIELDNAMES})
        (results / METRICS_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary

