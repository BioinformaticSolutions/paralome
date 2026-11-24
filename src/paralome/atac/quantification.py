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
import math
import os
import re
import shlex
import sys
import shutil
import subprocess
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from paralome.binaries import shell_join, worker_argv
from paralome.pipeline.jsonio import version_stale_reason
from paralome.checkpoints import (
    atomic_text_writer as _atomic_text_writer,
    atomic_write_text,
    exclusive_checkpoint_lock,
    stage_checkpoint_path,
)
from paralome.csvio import skip_comment_lines
from paralome.pipeline.resources import SampleParallelPlan, plan_sample_parallel
from paralome.pipeline.scheduler import detect_scheduler_capabilities, plan_array_tasks, require_array_task_ids
from paralome.pipeline.slurm import SlurmOptions, afterok, compress_array_ids, sbatch, write_sbatch
from paralome.atac.naming import sample_file_token as _sample_file_token
from paralome.pipeline.naming import slurm_job_name


PEAK_QUANT_PLAN_SCHEMA = "paralome.atac.peak_quantification_plan.v1"
PEAK_QUANT_PREPARE_SCHEMA = "paralome.atac.peak_quantification_prepare.v1"
PEAK_QUANT_SAMPLE_SCHEMA = "paralome.atac.peak_quantification_sample.v1"
PEAK_QUANT_STAGE_SCHEMA = "paralome.atac.peak_quantification.v2"


@dataclass(frozen=True)
class AtlasPeak:
    peak_id: str
    chrom: str
    start: int
    end: int


@dataclass(frozen=True)
class TSSRecord:
    chrom: str
    position: int
    strand: str
    gene_id: str
    gene_name: str
    gene_biotype: str


@dataclass(frozen=True)
class _TSSWindow:
    left: int
    right: int
    record: TSSRecord


@dataclass(frozen=True)
class _TSSOverlapIndex:
    lefts: Tuple[int, ...]
    windows: Tuple[_TSSWindow, ...]
    prefix_max_right: Tuple[int, ...]

    @classmethod
    def build(
        cls,
        records: Sequence[TSSRecord],
        *,
        upstream: int,
        downstream: int,
    ) -> "_TSSOverlapIndex":
        windows: List[_TSSWindow] = []
        for record in records:
            left = record.position - (upstream if record.strand == "+" else downstream)
            right = record.position + (downstream if record.strand == "+" else upstream) + 1
            windows.append(_TSSWindow(left, right, record))
        windows.sort(key=lambda item: (
            item.left,
            item.right,
            item.record.position,
            item.record.gene_id,
        ))
        prefix: List[int] = []
        maximum = -1
        for window in windows:
            maximum = max(maximum, window.right)
            prefix.append(maximum)
        return cls(
            tuple(window.left for window in windows),
            tuple(windows),
            tuple(prefix),
        )

    def overlapping(self, start: int, end: int) -> List[TSSRecord]:
        """Return TSS windows overlapping ``[start, end)`` using an interval index."""
        high = bisect_left(self.lefts, end)
        cursor = high - 1
        matches: List[TSSRecord] = []
        while cursor >= 0 and self.prefix_max_right[cursor] > start:
            window = self.windows[cursor]
            if window.right > start:
                matches.append(window.record)
            cursor -= 1
        matches.sort(key=lambda record: (record.position, record.gene_id))
        return matches


def _gtf_attributes(text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for key, value in re.findall(r'(\S+)\s+"([^"]*)"', text):
        attrs[key] = value
    return attrs


def read_atlas(path: Union[str, Path]) -> List[AtlasPeak]:
    atlas_path = Path(path).expanduser().resolve()
    peaks: List[AtlasPeak] = []
    seen: Set[str] = set()
    with atlas_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{atlas_path}:{line_no} requires BED4")
            chrom, start_text, end_text, peak_id = fields[:4]
            start, end = int(start_text), int(end_text)
            if not chrom or not peak_id or start < 0 or end <= start:
                raise ValueError(f"{atlas_path}:{line_no} has an invalid BED4 interval")
            if peak_id in seen:
                raise ValueError(f"Duplicate peak_id in atlas: {peak_id}")
            seen.add(peak_id)
            peaks.append(AtlasPeak(peak_id, chrom, start, end))
    if not peaks:
        raise ValueError(f"Peak atlas is empty: {atlas_path}")
    return peaks



#: Contigs never counted in bulk peak quantification.
MITOCHONDRIAL_CONTIGS = frozenset({"M", "MT", "chrM", "chrMT", "chrm", "chrmt"})
#: Recorded in the preparation signature, so a quantification made before it is
#: prepared and counted again rather than reused.
ATLAS_FILTER = "no_mitochondrial.v1"


def is_mitochondrial_contig(chrom: str) -> bool:
    name = str(chrom)
    bare = name[3:] if name.lower().startswith("chr") else name
    return name in MITOCHONDRIAL_CONTIGS or bare.upper() in {"M", "MT"}


def nuclear_atlas_peaks(path: Union[str, Path]) -> Tuple[List[AtlasPeak], int]:
    """The atlas's peaks off the mitochondrial genome, and how many were left out.

    MACS3 calls peaks on chrM reads and the merge keeps them, but chrM load
    varies widely between ATAC libraries (often 5-60%), so a library size that
    included it would move every nuclear peak's CPM, in one direction per
    sample. chrM peaks are not counted, tested or part of the library size.
    """
    peaks = read_atlas(path)
    kept = [peak for peak in peaks if not is_mitochondrial_contig(peak.chrom)]
    if not kept:
        raise ValueError(f"Peak atlas has no peaks off the mitochondrial genome: {path}")
    return kept, len(peaks) - len(kept)

def read_bam_manifest(path: Union[str, Path]) -> List[Dict[str, str]]:
    manifest_path = Path(path).expanduser().resolve()
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(skip_comment_lines(handle))
        required = {"sample_id", "bam"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("BAM manifest missing column(s): " + ", ".join(sorted(missing)))
        for row_no, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            bam_text = (row.get("bam") or "").strip()
            if not sample_id or not bam_text:
                raise ValueError(f"Invalid BAM manifest row {row_no}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in BAM manifest: {sample_id}")
            seen.add(sample_id)
            bam = Path(bam_text).expanduser()
            if not bam.is_absolute():
                candidates = [manifest_path.parent / bam, Path.cwd() / bam]
                if bam.parts and manifest_path.parent.parent.parent.name == bam.parts[0]:
                    candidates.append(manifest_path.parent.parent.parent.parent / bam)
                bam = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            bam = bam.resolve()
            if not bam.is_file():
                raise FileNotFoundError(f"Invalid BAM manifest row for {sample_id!r}: {bam}")
            rows.append({
                "sample_id": sample_id,
                "condition": (row.get("condition") or row.get("group") or "unknown").strip(),
                "bam": str(bam),
            })
    if not rows:
        raise ValueError(f"BAM manifest contains no samples: {manifest_path}")
    return rows


def load_tss(gtf: Union[str, Path]) -> List[TSSRecord]:
    records: Dict[Tuple[str, str], TSSRecord] = {}
    with Path(gtf).expanduser().resolve().open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"gene", "transcript"}:
                continue
            chrom, _, feature, start, end, _, strand, _, attr_text = fields
            attrs = _gtf_attributes(attr_text)
            gene_id = attrs.get("gene_id", "")
            if not gene_id or strand not in {"+", "-"}:
                continue
            position = int(start) - 1 if strand == "+" else int(end) - 1
            key = (chrom, gene_id)
            record = TSSRecord(
                chrom=chrom,
                position=position,
                strand=strand,
                gene_id=gene_id,
                gene_name=attrs.get("gene_name", gene_id),
                gene_biotype=attrs.get("gene_biotype") or attrs.get("gene_type", ""),
            )
            previous = records.get(key)
            if previous is None or feature == "gene":
                records[key] = record
    return sorted(records.values(), key=lambda record: (record.chrom, record.position, record.gene_id))


def _nearest_tss(
    rows: Sequence[TSSRecord],
    positions: Sequence[int],
    center: int,
) -> Optional[TSSRecord]:
    if not rows:
        return None
    index = bisect_left(positions, center)
    candidates = rows[max(0, index - 2): min(len(rows), index + 2)]
    return min(candidates, key=lambda record: (abs(center - record.position), record.gene_id))


def _normalize_chrom_name(chrom: str) -> str:
    """Strip a leading "chr"/"Chr"/"CHR" so "chr1" and "1" compare equal.

    Peaks extracted from a Seurat/Signac object (often UCSC/Cell Ranger
    "chr1"-style contigs) and the registered genome's GTF (often Ensembl
    "1"-style) can disagree on this convention for the same chromosome.
    Matching names literally would leave every peak on such a chromosome
    without TSS candidates.
    """
    text = str(chrom).strip()
    if text[:3].lower() == "chr":
        return text[3:]
    return text


def _chrom_alias_map(chroms: Iterable[str]) -> Dict[str, str]:
    """Map every spelling (exact, and "chr"-stripped) of each chrom in
    ``chroms`` to one canonical key actually present in ``chroms``. Exact
    matches always win (registered first), so this is a no-op when peak and
    GTF chrom naming already agree.
    """
    alias: Dict[str, str] = {}
    for chrom in chroms:
        alias.setdefault(chrom, chrom)
        alias.setdefault(_normalize_chrom_name(chrom), chrom)
    return alias


def annotate_peaks(
    peaks: Sequence[AtlasPeak],
    tss_records: Sequence[TSSRecord],
    upstream: int = 2000,
    downstream: int = 500,
) -> List[Dict[str, object]]:
    if upstream < 0 or downstream < 0:
        raise ValueError("TSS upstream/downstream windows must be >= 0")
    by_chrom: Dict[str, List[TSSRecord]] = {}
    for record in tss_records:
        by_chrom.setdefault(record.chrom, []).append(record)
    positions = {
        chrom: [record.position for record in rows]
        for chrom, rows in by_chrom.items()
    }
    overlap_indexes = {
        chrom: _TSSOverlapIndex.build(rows, upstream=upstream, downstream=downstream)
        for chrom, rows in by_chrom.items()
    }
    chrom_alias = _chrom_alias_map(by_chrom.keys())

    annotated: List[Dict[str, object]] = []
    for peak in peaks:
        resolved_chrom = chrom_alias.get(peak.chrom) or chrom_alias.get(_normalize_chrom_name(peak.chrom))
        rows = by_chrom.get(resolved_chrom, []) if resolved_chrom else []
        center = (peak.start + peak.end) // 2
        nearest = _nearest_tss(rows, positions.get(resolved_chrom, []) if resolved_chrom else [], center)
        overlap_index = overlap_indexes.get(resolved_chrom) if resolved_chrom else None
        overlaps = overlap_index.overlapping(peak.start, peak.end) if overlap_index else []
        primary = min(overlaps, key=lambda record: (abs(center - record.position), record.gene_id)) if overlaps else None
        annotated.append({
            "peak_id": peak.peak_id,
            "chrom": peak.chrom,
            "start": peak.start,
            "end": peak.end,
            "width": peak.end - peak.start,
            "nearest_gene_id": nearest.gene_id if nearest else "",
            "nearest_gene_name": nearest.gene_name if nearest else "",
            "nearest_gene_biotype": nearest.gene_biotype if nearest else "",
            "nearest_tss_position": nearest.position if nearest else "",
            # Genomic-coordinate difference (peak centre minus TSS); positive
            # means to the right of the TSS on the reference, whatever the strand.
            "distance_to_nearest_tss": center - nearest.position if nearest else "",
            "tss_overlap": bool(overlaps),
            "tss_gene_id": primary.gene_id if primary else "",
            "tss_gene_name": primary.gene_name if primary else "",
            "tss_gene_biotype": primary.gene_biotype if primary else "",
            "tss_overlap_count": len(overlaps),
            "tss_gene_ids": ";".join(record.gene_id for record in overlaps),
            "tss_gene_names": ";".join(record.gene_name for record in overlaps),
        })
    return annotated


def _peak_index(peaks: Sequence[AtlasPeak]) -> Dict[str, Tuple[List[int], List[AtlasPeak]]]:
    grouped: Dict[str, List[AtlasPeak]] = {}
    for peak in peaks:
        grouped.setdefault(peak.chrom, []).append(peak)
    index: Dict[str, Tuple[List[int], List[AtlasPeak]]] = {}
    for chrom, rows in grouped.items():
        ordered = sorted(rows, key=lambda peak: (peak.start, peak.end, peak.peak_id))
        index[chrom] = ([peak.start for peak in ordered], ordered)
    return index


def _prefix_max_ends(rows: Sequence[AtlasPeak]) -> List[int]:
    """Running maximum of peak ends, in start order."""
    ends: List[int] = []
    maximum = -1
    for peak in rows:
        maximum = max(maximum, peak.end)
        ends.append(maximum)
    return ends


def _count_unpaired_reads(bam: Union[str, Path], samtools: str) -> int:
    """Mapped primary reads that are not part of a pair (single-end)."""
    completed = subprocess.run(
        [samtools, "view", "-c", "-F", "3853", str(bam)],
        stdout=subprocess.PIPE, text=True, check=False,
    )
    try:
        return int((completed.stdout or "0").strip() or 0)
    except ValueError:
        return 0


def fragment_interval(fields: Sequence[str]) -> Optional[Tuple[int, int]]:
    """(start, end) 0-based half-open of a properly paired read's fragment, from its SAM fields; None if none.

    SAM's TLEN runs from the leftmost base of the leftmost mate to
    the rightmost base of the rightmost one. When the read is the rightmost
    mate, TLEN is negative and the fragment starts at the MATE's position
    (PNEXT), not at this read's -- min/max of POS and POS+TLEN would shift the
    fragment left by the read's length for about half of all fragments.
    """
    pos0 = int(fields[3]) - 1
    tlen = int(fields[8])
    if tlen == 0:
        return None
    if tlen > 0:
        start = pos0
    else:
        start = int(fields[7]) - 1
    end = start + abs(tlen)
    return (start, end) if end > start else None


#: BAM contigs are matched to atlas contigs with or without "chr"
#: (and chrM/MT); recorded in the sample signature and the stage checkpoint.
PEAK_COUNTING_METHOD = "contig_alias.v1"


def _contig_key(chrom: str) -> str:
    """"chr1"/"1" -> "1"; "chrM"/"M"/"MT" -> "MT"."""
    bare = _normalize_chrom_name(chrom)
    return "MT" if bare.upper() in {"M", "MT"} else bare


def _bam_contig_resolver(atlas_chroms: Iterable[str]):
    """Map a BAM contig to the atlas contig it names, or None.

    An exact name wins; otherwise the "chr"-insensitive name is used when it
    names exactly one atlas contig (ambiguous spellings are not guessed).
    """
    exact = set(atlas_chroms)
    by_key: Dict[str, Set[str]] = {}
    for chrom in exact:
        by_key.setdefault(_contig_key(chrom), set()).add(chrom)
    cache: Dict[str, Optional[str]] = {}

    def resolve(chrom: str) -> Optional[str]:
        if chrom in cache:
            return cache[chrom]
        if chrom in exact:
            found: Optional[str] = chrom
        else:
            matches = by_key.get(_contig_key(chrom), set())
            found = next(iter(matches)) if len(matches) == 1 else None
        cache[chrom] = found
        return found

    return resolve


def count_fragments(
    bam: Union[str, Path],
    peaks: Sequence[AtlasPeak],
    samtools: str = "samtools",
) -> Tuple[Dict[str, int], int, int]:
    index = _peak_index(peaks)
    resolve_contig = _bam_contig_resolver(index.keys())
    unmatched: Dict[str, int] = {}
    # The backward scan stops on the running maximum of ends, so a
    # short peak nested in an earlier, longer one is still reached.
    max_ends = {chrom: _prefix_max_ends(rows) for chrom, (_starts, rows) in index.items()}
    counts = {peak.peak_id: 0 for peak in peaks}
    eligible = 0
    assigned = 0
    command = [samtools, "view", "-f", "66", "-F", "3852", str(bam)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9:
            continue
        chrom = resolve_contig(fields[2])
        if chrom is None:
            unmatched[fields[2]] = unmatched.get(fields[2], 0) + 1
            continue
        interval = fragment_interval(fields)
        if interval is None:
            continue
        start, end = interval
        eligible += 1
        starts, rows = index[chrom]
        chrom_max_ends = max_ends[chrom]
        idx = bisect_left(starts, end)
        hit = False
        cursor = idx - 1
        while cursor >= 0:
            if chrom_max_ends[cursor] <= start:
                break
            peak = rows[cursor]
            if peak.end > start and peak.start < end:
                counts[peak.peak_id] += 1
                hit = True
            cursor -= 1
        if hit:
            assigned += 1
    return_code = proc.wait()
    if return_code:
        raise RuntimeError(f"samtools view failed for {bam} with exit code {return_code}")
    if eligible == 0 and unmatched:
        # Reads exist, but on no contig the atlas names.
        shown = ", ".join(sorted(unmatched)[:5])
        atlas_shown = ", ".join(sorted(index)[:5])
        raise RuntimeError(
            f"{bam} shares no contigs with the peak atlas: its {sum(unmatched.values())} paired "
            f"fragments are on contigs ({shown}) the atlas does not name (atlas contigs: {atlas_shown}). "
            "Align the sample to the same reference as the atlas, or remove it from the BAM manifest."
        )
    if eligible == 0:
        # Zero fragments would enter the cohort as 0 CPM at every
        # peak. Peak counting needs paired-end reads (read 1 of a proper pair).
        unpaired = _count_unpaired_reads(bam, samtools)
        if unpaired:
            raise RuntimeError(
                f"{bam} has {unpaired} single-end reads and no properly paired fragments. "
                "ATAC peak counting needs paired-end reads; single-end ATAC is not supported. "
                "Remove this sample from the BAM manifest or supply its paired-end alignment."
            )
        raise RuntimeError(
            f"{bam} has no properly paired, mapped, non-duplicate fragments to count. "
            "Check the alignment, or remove this sample from the BAM manifest."
        )
    return counts, eligible, assigned


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_identity(path: Path) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _identity_matches(identity: Mapping[str, object], path: Path) -> bool:
    try:
        return dict(identity) == _file_identity(path)
    except OSError:
        return False


def _tool_identity(command: str) -> Dict[str, object]:
    resolved = shutil.which(command)
    if resolved is None and Path(command).is_file():
        resolved = str(Path(command).expanduser().resolve())
    identity: Dict[str, object] = {"command": command, "resolved": resolved}
    if resolved and Path(resolved).is_file():
        identity["file"] = _file_identity(Path(resolved))
    return identity


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    # See pipeline.jsonio.CHECKPOINT_VERSION_KEY.
    from paralome.pipeline.jsonio import stamp_version

    atomic_write_text(path, json.dumps(stamp_version(payload), indent=2, sort_keys=True) + "\n")






def _write_annotation(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Peak annotation contains no rows")
    # This writes cohort_peaks.annotated.csv, a final/machine-facing
    # output -- comma-delimited, matching _read_annotation below.
    fieldnames = list(rows[0].keys())
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_annotation(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows or "peak_id" not in rows[0]:
        raise RuntimeError(f"Invalid prepared peak annotation: {path}")
    peak_ids = [row["peak_id"] for row in rows]
    if len(set(peak_ids)) != len(peak_ids):
        raise RuntimeError(f"Prepared peak annotation contains duplicate peak IDs: {path}")
    return rows


def _write_counts(path: Path, peaks: Sequence[AtlasPeak], counts: Mapping[str, int]) -> None:
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["peak_id", "raw_count"])
        for peak in peaks:
            writer.writerow([peak.peak_id, int(counts[peak.peak_id])])


def _read_counts(path: Path, expected_peak_ids: Sequence[str]) -> Dict[str, int]:
    values: Dict[str, int] = {}
    observed: List[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if set(reader.fieldnames or []) != {"peak_id", "raw_count"}:
            raise RuntimeError(f"Invalid sample count header: {path}")
        for row in reader:
            peak_id = row["peak_id"]
            if peak_id in values:
                raise RuntimeError(f"Duplicate peak_id in sample counts: {peak_id}")
            value = int(row["raw_count"])
            if value < 0:
                raise RuntimeError(f"Negative raw count for {peak_id}: {path}")
            observed.append(peak_id)
            values[peak_id] = value
    if observed != list(expected_peak_ids):
        raise RuntimeError(f"Sample count peak order does not match the prepared atlas: {path}")
    return values


def _write_matrix(
    path: Path,
    metadata: Sequence[Mapping[str, object]],
    samples: Sequence[str],
    values: Mapping[str, Mapping[str, float]],
) -> None:
    if not metadata:
        raise ValueError("Cannot write an empty peak matrix")
    fields = list(metadata[0].keys()) + list(samples)
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in metadata:
            out = dict(row)
            peak_id = str(row["peak_id"])
            out.update({sample: values[sample][peak_id] for sample in samples})
            writer.writerow(out)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _preparation_signature(plan: Mapping[str, object]) -> str:
    return _stable_digest({
        "atlas_identity": plan["atlas_identity"],
        "gtf_identity": plan["gtf_identity"],
        "tss_upstream": int(plan["tss_upstream"]),
        "tss_downstream": int(plan["tss_downstream"]),
        "annotation_algorithm": "indexed_tss_windows.v1",
        "atlas_filter": ATLAS_FILTER,
    })


def _build_peak_quantification_plan(
    atlas: Union[str, Path],
    bam_manifest: Union[str, Path],
    gtf: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    upstream: int,
    downstream: int,
    pseudocount: float,
    samtools: str,
    backend: str,
    threads: int,
    jobs: int,
) -> Tuple[Path, Dict[str, object]]:
    if upstream < 0 or downstream < 0:
        raise ValueError("TSS upstream/downstream windows must be >= 0")
    # Log2(CPM + pseudocount) needs pseudocount > 0 (zero-count peaks).
    if not pseudocount > 0:
        raise ValueError("--pseudocount must be > 0 (log2-CPM of zero-count peaks is undefined at 0)")
    atlas_path = Path(atlas).expanduser().resolve()
    manifest_path = Path(bam_manifest).expanduser().resolve()
    gtf_path = Path(gtf).expanduser().resolve()
    for path, label in ((atlas_path, "atlas"), (manifest_path, "BAM manifest"), (gtf_path, "GTF")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    out = Path(output_dir).expanduser().resolve()
    for directory in (
        out,
        out / "inputs",
        out / "results",
        out / "counts",
        out / "checkpoints",
        out / "scripts",
        out / "logs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    samples = read_bam_manifest(manifest_path)
    resource_plan = plan_sample_parallel(
        total_threads=threads,
        sample_count=len(samples),
        max_threads_per_sample=1,
        jobs=jobs,
    )
    tokens: Set[str] = set()
    tasks: List[Dict[str, object]] = []
    for task_id, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        token = _sample_file_token(sample_id)
        if token in tokens:
            token = f"{token}_{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:10]}"
        tokens.add(token)
        bam_path = Path(sample["bam"])
        tasks.append({
            "task_id": task_id,
            "sample_id": sample_id,
            "sample_token": token,
            "condition": sample["condition"],
            "bam": str(bam_path),
            "bam_identity": _file_identity(bam_path),
            "counts": str(out / "counts" / f"{token}.counts.tsv"),
            "checkpoint": str(stage_checkpoint_path(out, f"sample_{token}")),
        })

    samtools_identity = _tool_identity(samtools)
    resolved_samtools = str(samtools_identity.get("resolved") or samtools)

    plan: Dict[str, object] = {
        "schema": PEAK_QUANT_PLAN_SCHEMA,
        "stage": "peak_quantification",
        "backend": backend,
        "atlas": str(atlas_path),
        "atlas_identity": _file_identity(atlas_path),
        "bam_manifest": str(manifest_path),
        "bam_manifest_sha256": _sha256_file(manifest_path),
        "gtf": str(gtf_path),
        "gtf_identity": _file_identity(gtf_path),
        "tss_upstream": int(upstream),
        "tss_downstream": int(downstream),
        "pseudocount": float(pseudocount),
        "samtools": resolved_samtools,
        "samtools_identity": samtools_identity,
        "threads": int(threads),
        "jobs": int(jobs),
        "resource_plan": resource_plan.to_dict(),
        "tasks": tasks,
        "outputs": {
            "annotated_peaks": str(out / "results" / "cohort_peaks.annotated.csv"),
            "raw_counts": str(out / "results" / "peak_raw_counts.csv"),
            "cpm": str(out / "results" / "peak_cpm.csv"),
            "log2_cpm": str(out / "results" / "peak_log2_cpm.csv"),
            "sample_qc": str(out / "results" / "sample_qc.csv"),
            "legacy_summary": str(out / "results" / "peak_counting_summary.csv"),
            "prepare_checkpoint": str(stage_checkpoint_path(out, "peak_quantification_prepare")),
            "checkpoint": str(stage_checkpoint_path(out, "peak_quantification")),
        },
        "output_dir": str(out),
    }
    plan["preparation_signature"] = _preparation_signature(plan)
    plan["plan_id"] = _stable_digest({
        "atlas_identity": plan["atlas_identity"],
        "bam_manifest_sha256": plan["bam_manifest_sha256"],
        "gtf_identity": plan["gtf_identity"],
        "preparation_signature": plan["preparation_signature"],
        "pseudocount": plan["pseudocount"],
        "samtools_identity": plan["samtools_identity"],
        "samples": [
            {
                "sample_id": task["sample_id"],
                "condition": task["condition"],
                "bam_identity": task["bam_identity"],
            }
            for task in tasks
        ],
    })
    plan_path = out / "inputs" / "peak_quantification_plan.json"
    _atomic_json(plan_path, plan)
    return plan_path, plan


def _validate_preparation_checkpoint(plan: Mapping[str, object]) -> Optional[Dict[str, object]]:
    checkpoint_path = Path(str(plan["outputs"]["prepare_checkpoint"]))
    annotation_path = Path(str(plan["outputs"]["annotated_peaks"]))
    if not checkpoint_path.is_file() or not annotation_path.is_file():
        return None
    if not _identity_matches(plan["atlas_identity"], Path(str(plan["atlas"]))):
        return None
    if not _identity_matches(plan["gtf_identity"], Path(str(plan["gtf"]))):
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("schema") != PEAK_QUANT_PREPARE_SCHEMA:
            return None
        if checkpoint.get("complete") is not True:
            return None
        if version_stale_reason(checkpoint) is not None:
            return None
        if checkpoint.get("preparation_signature") != plan.get("preparation_signature"):
            return None
        if checkpoint.get("annotated_peaks") != str(annotation_path):
            return None
        if checkpoint.get("annotated_sha256") != _sha256_file(annotation_path):
            return None
        rows = _read_annotation(annotation_path)
        if int(checkpoint.get("peaks", -1)) != len(rows):
            return None
        return checkpoint
    except Exception:
        return None


def _run_prepare_unlocked(plan: Mapping[str, object], *, overwrite: bool = False) -> Dict[str, object]:
    if not overwrite:
        cached = _validate_preparation_checkpoint(plan)
        if cached is not None:
            return {**cached, "reused": True}
    checkpoint_path = Path(str(plan["outputs"]["prepare_checkpoint"]))
    annotation_path = Path(str(plan["outputs"]["annotated_peaks"]))
    try:
        if not _identity_matches(plan["atlas_identity"], Path(str(plan["atlas"]))):
            raise RuntimeError("Peak atlas changed after quantification planning")
        if not _identity_matches(plan["gtf_identity"], Path(str(plan["gtf"]))):
            raise RuntimeError("GTF changed after quantification planning")
        peaks, mitochondrial_peaks = nuclear_atlas_peaks(str(plan["atlas"]))
        tss_records = load_tss(str(plan["gtf"]))
        annotation = annotate_peaks(
            peaks,
            tss_records,
            int(plan["tss_upstream"]),
            int(plan["tss_downstream"]),
        )
        _write_annotation(annotation_path, annotation)
        result: Dict[str, object] = {
            "schema": PEAK_QUANT_PREPARE_SCHEMA,
            "stage": "peak_quantification_prepare",
            "complete": True,
            "plan_id": str(plan["plan_id"]),
            "preparation_signature": str(plan["preparation_signature"]),
            "atlas": str(plan["atlas"]),
            "gtf": str(plan["gtf"]),
            "peaks": len(peaks),
            "mitochondrial_peaks_excluded": mitochondrial_peaks,
            "tss_records": len(tss_records),
            "annotation_algorithm": "indexed_tss_windows.v1",
            "annotated_peaks": str(annotation_path),
            "annotated_sha256": _sha256_file(annotation_path),
            "worker_pid": os.getpid(),
            "reused": False,
        }
        _atomic_json(checkpoint_path, result)
        return result
    except Exception as exc:
        _atomic_json(checkpoint_path, {
            "schema": PEAK_QUANT_PREPARE_SCHEMA,
            "stage": "peak_quantification_prepare",
            "complete": False,
            "plan_id": str(plan["plan_id"]),
            "preparation_signature": str(plan["preparation_signature"]),
            "error": f"{type(exc).__name__}: {exc}",
            "worker_pid": os.getpid(),
        })
        raise


def run_peak_quantification_prepare_task(
    plan_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).expanduser().resolve().read_text())
    checkpoint_path = Path(str(plan["outputs"]["prepare_checkpoint"]))
    with exclusive_checkpoint_lock(checkpoint_path):
        return _run_prepare_unlocked(plan, overwrite=overwrite)


def _sample_task_signature(
    task: Mapping[str, object],
    plan: Mapping[str, object],
    preparation: Mapping[str, object],
) -> str:
    return _stable_digest({
        "sample_id": task["sample_id"],
        "bam_identity": task["bam_identity"],
        "preparation_signature": preparation["preparation_signature"],
        "samtools_identity": plan["samtools_identity"],
        "fragment_filter": {"include_flags": 66, "exclude_flags": 3852},
        # The fragment interval method (fragment_interval), so counts made with an older interval
        # rule are counted again.
        "fragment_interval": "tlen_from_leftmost_mate.v2",
        # Nested peaks are reached (prefix-max scan) and a sample
        # with no paired fragments is refused, so earlier counts are redone.
        "peak_scan": "prefix_max_end.v2",
        "counting_method": PEAK_COUNTING_METHOD,
    })


def _validate_sample_checkpoint(
    task: Mapping[str, object],
    plan: Mapping[str, object],
    preparation: Mapping[str, object],
    expected_peak_ids: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, object]]:
    checkpoint_path = Path(str(task["checkpoint"]))
    counts_path = Path(str(task["counts"]))
    if not checkpoint_path.is_file() or not counts_path.is_file():
        return None
    if not _identity_matches(task["bam_identity"], Path(str(task["bam"]))):
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("schema") != PEAK_QUANT_SAMPLE_SCHEMA:
            return None
        if checkpoint.get("complete") is not True:
            return None
        if version_stale_reason(checkpoint) is not None:
            return None
        if checkpoint.get("sample_id") != task.get("sample_id"):
            return None
        signature = _sample_task_signature(task, plan, preparation)
        if checkpoint.get("task_signature") != signature:
            return None
        if checkpoint.get("counts") != str(counts_path):
            return None
        if checkpoint.get("counts_sha256") != _sha256_file(counts_path):
            return None
        peak_ids = list(expected_peak_ids) if expected_peak_ids is not None else [
            row["peak_id"] for row in _read_annotation(Path(str(plan["outputs"]["annotated_peaks"])))
        ]
        counts = _read_counts(counts_path, peak_ids)
        if int(checkpoint.get("peaks", -1)) != len(counts):
            return None
        if int(checkpoint.get("library_size_used_for_cpm", -1)) != sum(counts.values()):
            return None
        return checkpoint
    except Exception:
        return None


def _run_sample_unlocked(
    task: Mapping[str, object],
    plan: Mapping[str, object],
    preparation: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    annotation_rows = _read_annotation(Path(str(plan["outputs"]["annotated_peaks"])))
    expected_peak_ids = [row["peak_id"] for row in annotation_rows]
    if not overwrite:
        cached = _validate_sample_checkpoint(
            task,
            plan,
            preparation,
            expected_peak_ids,
        )
        if cached is not None:
            return {**cached, "reused": True}

    checkpoint_path = Path(str(task["checkpoint"]))
    counts_path = Path(str(task["counts"]))
    signature = _sample_task_signature(task, plan, preparation)
    try:
        if not _identity_matches(task["bam_identity"], Path(str(task["bam"]))):
            raise RuntimeError(f"BAM changed after quantification planning: {task['bam']}")
        peaks, _ = nuclear_atlas_peaks(str(plan["atlas"]))
        if [peak.peak_id for peak in peaks] != expected_peak_ids:
            raise RuntimeError("Prepared annotation peak order differs from the source atlas")
        counts, eligible, assigned = count_fragments(
            str(task["bam"]),
            peaks,
            samtools=str(plan["samtools"]),
        )
        _write_counts(counts_path, peaks, counts)
        library_size = sum(counts.values())
        result: Dict[str, object] = {
            "schema": PEAK_QUANT_SAMPLE_SCHEMA,
            "stage": "peak_quantification_sample",
            "complete": True,
            "sample_id": str(task["sample_id"]),
            "condition": str(task["condition"]),
            "task_id": int(task["task_id"]),
            "task_signature": signature,
            "bam": str(task["bam"]),
            "bam_identity": task["bam_identity"],
            "peaks": len(peaks),
            "eligible_fragments": eligible,
            "assigned_fragments": assigned,
            "fraction_in_peaks": assigned / eligible if eligible else 0.0,
            "library_size_used_for_cpm": library_size,
            "counts": str(counts_path),
            "counts_sha256": _sha256_file(counts_path),
            "worker_pid": os.getpid(),
            "reused": False,
        }
        _atomic_json(checkpoint_path, result)
        return result
    except Exception as exc:
        _atomic_json(checkpoint_path, {
            "schema": PEAK_QUANT_SAMPLE_SCHEMA,
            "stage": "peak_quantification_sample",
            "complete": False,
            "sample_id": str(task["sample_id"]),
            "condition": str(task["condition"]),
            "task_id": int(task["task_id"]),
            "task_signature": signature,
            "bam": str(task["bam"]),
            "bam_identity": task["bam_identity"],
            "error": f"{type(exc).__name__}: {exc}",
            "worker_pid": os.getpid(),
        })
        raise


def _run_sample_payload(
    task: Mapping[str, object],
    plan_path: Union[str, Path],
    overwrite: bool = False,
) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).expanduser().resolve().read_text())
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        raise RuntimeError("Peak-quantification preparation is missing or invalid")
    checkpoint_path = Path(str(task["checkpoint"]))
    with exclusive_checkpoint_lock(checkpoint_path):
        return _run_sample_unlocked(
            task,
            plan,
            preparation,
            overwrite=overwrite,
        )


def run_peak_quantification_sample_task(
    plan_path: Union[str, Path],
    task_id: int,
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    resolved = Path(plan_path).expanduser().resolve()
    plan = json.loads(resolved.read_text())
    tasks = list(plan["tasks"])
    if task_id < 1 or task_id > len(tasks):
        raise ValueError(f"task_id must be between 1 and {len(tasks)}")
    return _run_sample_payload(tasks[task_id - 1], resolved, overwrite)


def _mark_stage_incomplete(
    plan: Mapping[str, object],
    pending_task_ids: Sequence[int],
    *,
    preparation_required: bool,
) -> None:
    checkpoint_path = Path(str(plan["outputs"]["checkpoint"]))
    with exclusive_checkpoint_lock(checkpoint_path):
        _atomic_json(checkpoint_path, {
            "schema": PEAK_QUANT_STAGE_SCHEMA,
            "stage": "peak_quantification",
            "complete": False,
            "plan_id": str(plan["plan_id"]),
            "samples": len(plan["tasks"]),
            "pending_task_ids": list(pending_task_ids),
            "preparation_required": bool(preparation_required),
            "resource_plan": plan["resource_plan"],
        })


def _pending_task_ids(
    plan: Mapping[str, object],
    *,
    overwrite: bool,
) -> Tuple[bool, List[int]]:
    preparation = None if overwrite else _validate_preparation_checkpoint(plan)
    preparation_required = preparation is None
    if preparation_required:
        return True, [int(task["task_id"]) for task in plan["tasks"]]
    annotation_rows = _read_annotation(Path(str(plan["outputs"]["annotated_peaks"])))
    peak_ids = [row["peak_id"] for row in annotation_rows]
    pending = [
        int(task["task_id"])
        for task in plan["tasks"]
        if overwrite or _validate_sample_checkpoint(task, plan, preparation, peak_ids) is None
    ]
    return False, pending


def _stage_checkpoint_current(plan: Mapping[str, object]) -> bool:
    """True when the stage checkpoint is complete for this very plan."""
    try:
        existing = json.loads(Path(str(plan["outputs"]["checkpoint"])).read_text())
    except (OSError, ValueError):
        return False
    return (
        existing.get("complete") is True
        and existing.get("plan_id") == plan["plan_id"]
        and existing.get("counting_method") == PEAK_COUNTING_METHOD
    )


def finalize_peak_quantification_plan(plan_path: Union[str, Path]) -> Dict[str, object]:
    resolved = Path(plan_path).expanduser().resolve()
    plan = json.loads(resolved.read_text())
    if not float(plan["pseudocount"]) > 0:  # Plans written before the plan-time check
        raise ValueError("--pseudocount must be > 0; re-run peak quantification with a positive --pseudocount")
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        _mark_stage_incomplete(plan, [], preparation_required=True)
        raise RuntimeError("Peak-quantification preparation failed validation")

    annotation_rows = _read_annotation(Path(str(plan["outputs"]["annotated_peaks"])))
    peak_ids = [row["peak_id"] for row in annotation_rows]
    tasks = list(plan["tasks"])
    invalid = [
        int(task["task_id"])
        for task in tasks
        if _validate_sample_checkpoint(task, plan, preparation, peak_ids) is None
    ]
    if invalid:
        _mark_stage_incomplete(plan, invalid, preparation_required=False)
        labels = [str(tasks[task_id - 1]["sample_id"]) for task_id in invalid]
        raise RuntimeError("Peak-quantification sample task(s) failed validation: " + ", ".join(labels))

    sample_ids = [str(task["sample_id"]) for task in tasks]
    raw_counts: Dict[str, Dict[str, float]] = {}
    cpm: Dict[str, Dict[str, float]] = {}
    log2_cpm: Dict[str, Dict[str, float]] = {}
    qc_rows: List[Dict[str, object]] = []
    for task in tasks:
        sample_id = str(task["sample_id"])
        checkpoint = json.loads(Path(str(task["checkpoint"])).read_text())
        sample_counts = _read_counts(Path(str(task["counts"])), peak_ids)
        raw_counts[sample_id] = {peak_id: float(sample_counts[peak_id]) for peak_id in peak_ids}
        library_size = sum(sample_counts.values())
        cpm[sample_id] = {
            peak_id: (sample_counts[peak_id] / library_size * 1_000_000.0 if library_size else 0.0)
            for peak_id in peak_ids
        }
        log2_cpm[sample_id] = {
            peak_id: math.log2(cpm[sample_id][peak_id] + float(plan["pseudocount"]))
            for peak_id in peak_ids
        }
        qc_rows.append({
            "sample_id": sample_id,
            "condition": str(task["condition"]),
            "bam": str(task["bam"]),
            "eligible_fragments": int(checkpoint["eligible_fragments"]),
            "assigned_fragments": int(checkpoint["assigned_fragments"]),
            "fraction_in_peaks": float(checkpoint["fraction_in_peaks"]),
            "library_size_used_for_cpm": library_size,
        })

    outputs = plan["outputs"]
    raw_path = Path(str(outputs["raw_counts"]))
    cpm_path = Path(str(outputs["cpm"]))
    log_path = Path(str(outputs["log2_cpm"]))
    sample_qc_path = Path(str(outputs["sample_qc"]))
    legacy_summary_path = Path(str(outputs["legacy_summary"]))
    _write_matrix(raw_path, annotation_rows, sample_ids, raw_counts)
    _write_matrix(cpm_path, annotation_rows, sample_ids, cpm)
    _write_matrix(log_path, annotation_rows, sample_ids, log2_cpm)
    _write_rows(sample_qc_path, qc_rows)
    _write_rows(legacy_summary_path, qc_rows)

    checkpoint_path = Path(str(outputs["checkpoint"]))
    result: Dict[str, object] = {
        "schema": PEAK_QUANT_STAGE_SCHEMA,
        "stage": "peak_quantification",
        "complete": True,
        "plan_id": str(plan["plan_id"]),
        "plan": str(resolved),
        "samples": len(tasks),
        "peaks": len(annotation_rows),
        "sample_order": sample_ids,
        "raw_counts": str(raw_path),
        "cpm": str(cpm_path),
        "log2_cpm": str(log_path),
        "annotated_peaks": str(outputs["annotated_peaks"]),
        "sample_qc": str(sample_qc_path),
        "summary": str(legacy_summary_path),
        "counts_dir": str(Path(str(plan["output_dir"])) / "counts"),
        "pseudocount": float(plan["pseudocount"]),
        "tss_upstream": int(plan["tss_upstream"]),
        "tss_downstream": int(plan["tss_downstream"]),
        "annotation_algorithm": "indexed_tss_windows.v1",
        "counting_method": PEAK_COUNTING_METHOD,
        "resource_plan": plan["resource_plan"],
        "output_sha256": {
            "raw_counts": _sha256_file(raw_path),
            "cpm": _sha256_file(cpm_path),
            "log2_cpm": _sha256_file(log_path),
            "annotated_peaks": _sha256_file(Path(str(outputs["annotated_peaks"]))),
            "sample_qc": _sha256_file(sample_qc_path),
            "summary": _sha256_file(legacy_summary_path),
        },
    }
    with exclusive_checkpoint_lock(checkpoint_path):
        _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    return result


def run_peak_quantification_finalize_task(plan_path: Union[str, Path]) -> Dict[str, object]:
    return finalize_peak_quantification_plan(plan_path)




def prepare_peak_quantification_slurm_plan(
    atlas: Union[str, Path],
    bam_manifest: Union[str, Path],
    gtf: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    upstream: int = 2000,
    downstream: int = 500,
    pseudocount: float = 1.0,
    samtools: str = "samtools",
    threads: int = 1,
    jobs: int = 0,
    partition: Optional[str] = None,
    mem: str = "8G",
    overwrite: bool = False,
) -> Dict[str, object]:
    plan_path, plan = _build_peak_quantification_plan(
        atlas,
        bam_manifest,
        gtf,
        output_dir,
        upstream=upstream,
        downstream=downstream,
        pseudocount=pseudocount,
        samtools=samtools,
        backend="slurm",
        threads=threads,
        jobs=jobs,
    )
    preparation_required, pending = _pending_task_ids(plan, overwrite=overwrite)
    # Also when only the plan changed (e.g. --pseudocount, a
    # condition relabel) and no task is pending, so the old result never
    # reads complete under the new request.
    if preparation_required or pending or overwrite or not _stage_checkpoint_current(plan):
        _mark_stage_incomplete(
            plan,
            pending,
            preparation_required=preparation_required,
        )
    capabilities = detect_scheduler_capabilities("slurm")
    require_array_task_ids(pending, capabilities, stage="Peak quantification")
    concurrency = int(plan["resource_plan"]["concurrent_jobs"])
    array_plan = plan_array_tasks(
        task_count=len(pending),
        concurrency=concurrency,
        capabilities=capabilities,
    )
    plan["preparation_required"] = preparation_required
    plan["pending_task_ids"] = pending
    plan["scheduler_capabilities"] = capabilities.to_dict()
    plan["array_plan"] = array_plan.to_dict()
    _atomic_json(plan_path, plan)

    cli = shell_join(worker_argv())
    out = Path(str(plan["output_dir"]))
    overwrite_flag = " --overwrite" if overwrite else ""
    prepare_script = write_sbatch(
        out / "scripts" / "peak_quantification_prepare.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(out, "peak_quant_prepare"),
            cpus_per_task=1,
            mem=mem,
            partition=partition,
            output=out / "logs" / "peak_quant_prepare_%j.out",
            error=out / "logs" / "peak_quant_prepare_%j.err",
        ),
        f"{cli} atac peak-quant-prepare --plan {shlex.quote(str(plan_path))}{overwrite_flag}",
    )
    array_scripts: List[Path] = []
    for batch in array_plan.batches:
        task_ids = pending[batch.first_task - 1: batch.last_task]
        array_spec = compress_array_ids(task_ids, throttle=min(batch.throttle, len(task_ids)))
        batch_index = batch.batch_index
        array_scripts.append(write_sbatch(
            out / "scripts" / f"peak_quantification_samples_{batch_index:03d}.sbatch",
            SlurmOptions(
                job_name=slurm_job_name(out, f"peak_quant_{batch_index:03d}"),
                cpus_per_task=1,
                mem=mem,
                partition=partition,
                array=array_spec,
                output=out / "logs" / f"peak_quant_{batch_index:03d}_%A_%a.out",
                error=out / "logs" / f"peak_quant_{batch_index:03d}_%A_%a.err",
            ),
            f'{cli} atac peak-quant-sample --plan {shlex.quote(str(plan_path))} '
            f'--task-id "$SLURM_ARRAY_TASK_ID"{overwrite_flag}',
        ))
    finalize_script = write_sbatch(
        out / "scripts" / "peak_quantification_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(out, "peak_quant_finalize"),
            cpus_per_task=1,
            mem=mem,
            partition=partition,
            output=out / "logs" / "peak_quant_finalize_%j.out",
            error=out / "logs" / "peak_quant_finalize_%j.err",
        ),
        f"{cli} atac peak-quant-finalize --plan {shlex.quote(str(plan_path))}",
    )
    return {
        "plan": plan_path,
        "prepare_script": prepare_script,
        "array_scripts": array_scripts,
        "finalize_script": finalize_script,
        "preparation_required": preparation_required,
        "task_count": len(pending),
        "total_samples": len(plan["tasks"]),
        "reused_samples": len(plan["tasks"]) - len(pending),
        "concurrency": concurrency,
        "resource_plan": plan["resource_plan"],
        "scheduler_capabilities": capabilities.to_dict(),
        "array_plan": array_plan.to_dict(),
        "plan_id": plan["plan_id"],
    }


def submit_peak_quantification_slurm_plan(
    plan_info: Mapping[str, object],
    *,
    submit: bool,
) -> Dict[str, object]:
    result = dict(plan_info)
    result.update({
        "submitted": False,
        "job_id": None,
        "submitted_job_ids": [],
        "complete": False,
    })
    if not submit:
        return result
    # Each job ID is reported the moment its sbatch returns. They
    # were reported only after the last one, so when a later sbatch was
    # refused the command failed having named none of the jobs already queued,
    # and the next start submitted the stage again beside them. stderr, which
    # the supervisor reads as well, so the JSON on stdout stays whole.
    prepare_job_id = sbatch(plan_info["prepare_script"])
    print(f"Submitted peak-quantification prepare job: {prepare_job_id}", file=sys.stderr, flush=True)
    dependency: Optional[str] = afterok(prepare_job_id)
    sample_job_ids: List[str] = []
    for script in plan_info["array_scripts"]:
        job_id = sbatch(script, dependency=dependency)
        print(f"Submitted peak-quantification array job: {job_id}", file=sys.stderr, flush=True)
        sample_job_ids.append(job_id)
        dependency = afterok(job_id)
    finalize_job_id = sbatch(plan_info["finalize_script"], dependency=dependency)
    print(f"Submitted peak-quantification finalizer job: {finalize_job_id}", file=sys.stderr, flush=True)
    job_ids = [prepare_job_id, *sample_job_ids, finalize_job_id]
    result.update({
        "submitted": True,
        "job_id": ",".join(job_ids),
        "submitted_job_ids": job_ids,
        "prepare_job_id": prepare_job_id,
        "sample_job_ids": sample_job_ids,
        "finalize_job_id": finalize_job_id,
    })
    return result


def run_peak_quantification(
    atlas: Union[str, Path],
    bam_manifest: Union[str, Path],
    gtf: Union[str, Path],
    output_dir: Union[str, Path],
    upstream: int = 2000,
    downstream: int = 500,
    pseudocount: float = 1.0,
    samtools: str = "samtools",
    *,
    threads: int = 1,
    jobs: int = 0,
    overwrite: bool = False,
) -> Dict[str, object]:
    plan_path, plan = _build_peak_quantification_plan(
        atlas,
        bam_manifest,
        gtf,
        output_dir,
        upstream=upstream,
        downstream=downstream,
        pseudocount=pseudocount,
        samtools=samtools,
        backend="local",
        threads=threads,
        jobs=jobs,
    )
    preparation_required, pending_ids = _pending_task_ids(plan, overwrite=overwrite)
    # See prepare_peak_quantification_slurm_plan.
    if preparation_required or pending_ids or overwrite or not _stage_checkpoint_current(plan):
        _mark_stage_incomplete(
            plan,
            pending_ids,
            preparation_required=preparation_required,
        )
    prepare_result = run_peak_quantification_prepare_task(plan_path, overwrite=overwrite)
    plan = json.loads(plan_path.read_text())
    preparation = _validate_preparation_checkpoint(plan)
    if preparation is None:
        raise RuntimeError("Peak-quantification preparation failed validation")
    annotation_rows = _read_annotation(Path(str(plan["outputs"]["annotated_peaks"])))
    peak_ids = [row["peak_id"] for row in annotation_rows]
    tasks = list(plan["tasks"])
    pending = [
        task
        for task in tasks
        if overwrite or _validate_sample_checkpoint(task, plan, preparation, peak_ids) is None
    ]
    statuses: List[Dict[str, object]] = []
    resource_plan = SampleParallelPlan(**plan["resource_plan"])
    if len(pending) <= 1 or resource_plan.concurrent_jobs <= 1:
        for task in pending:
            try:
                statuses.append(_run_sample_payload(task, plan_path, overwrite))
            except Exception as exc:
                raise RuntimeError(f"Peak quantification failed for sample {task['sample_id']}: {exc}") from exc
    else:
        with ProcessPoolExecutor(max_workers=min(resource_plan.concurrent_jobs, len(pending))) as executor:
            future_to_task = {
                executor.submit(_run_sample_payload, task, plan_path, overwrite): task
                for task in pending
            }
            try:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        statuses.append(future.result())
                    except Exception as exc:
                        raise RuntimeError(f"Peak quantification failed for sample {task['sample_id']}: {exc}") from exc
            except Exception:
                for future in future_to_task:
                    future.cancel()
                raise

    result = finalize_peak_quantification_plan(plan_path)
    result.update({
        "preparation_reused": bool(prepare_result.get("reused")),
        "executed_samples": len(pending),
        "reused_samples": len(tasks) - len(pending),
        "sample_task_results": sorted(statuses, key=lambda item: int(item["task_id"])),
        "concurrency": resource_plan.concurrent_jobs,
        "peak_cpus": resource_plan.peak_cpus,
    })
    return result
