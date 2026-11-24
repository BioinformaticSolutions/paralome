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

"""The genome_tracks stage: an ATAC run's peaks and profiles for a genome browser.

It runs on both ATAC pipelines (bulk and snATAC) after all peak QC and sample
merges, so the tracks hold only viable peaks: after the differential (bulk:
differential_accessibility; snATAC: singlecell_differential) and right after
statistics_output. It writes one file::

    <workdir>/results/<prefix>_<ATAC|snATAC>_genome_tracks.zip

stored (not compressed: every track in it is compressed already, and a reader
opens the members in place). Its parts:

    manifest.json                       what is in the zip, for the viewer
    chrom.sizes                         the contigs the tracks are on, and their lengths
    peaks/<prefix>_viable_peaks.bb      THE VIABLE PEAKS: the union of the peaks tested in any comparison
                                        of any statistic (and, for snATAC, any stratum)
    peaks/<comparison>.bb               one per comparison: every peak it tested, coloured by the direction
                                        of its log2 fold change (higher red, lower blue; an F-test, no
                                        direction, grey), each carrying its log2 fold change, p and q --
                                        no call of significance
    coverage/<sample>.bw                ONE PROFILE PER SAMPLE (per sample, not by group): fragment
                                        coverage inside the viable peaks in 25 bp bins, in CPM -- covering fragments per base
                                        per million eligible fragments of the sample. Outside the viable
                                        peaks a track has no data at all, not zero
    annotation/genes.bb                 the run's GTF as gene models (one transcript per gene where the
                                        GTF tags Ensembl_canonical, every transcript otherwise)

Bulk ATAC reads its fragments from the deduplicated BAMs exactly as peak
quantification counts them (samtools view -f 66 -F 3852, the fragment from
POS/PNEXT/TLEN, contigs matched with or without "chr"), and scales each sample
by the factor the candidate-gene views use: one million over its eligible
fragments (peak_quantification's summary). snATAC reads the fragment files the
Signac object names and its
samples are the pseudobulk samples --replicate-column made (after --remove): every
cell's fragments are added to its sample's profile, and the sample's eligible
fragments are all its cells' fragments off the mitochondrial genome.

bigWig and bigBed are written by the UCSC tools bedGraphToBigWig and
bedToBigBed, resolved like every other tool
(binaries.resolve_tool); missing, the stage stops on PLM-0416. Paralome does
not ship them.

SELF-CHECKING, as statistics_output is: issued again, it compares its inputs
(each file's path, size and modification time) and options with the request in
its checkpoint, keeps the zip when nothing changed, and builds it again in
place otherwise.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paralome.checkpoints import atomic_write_text, file_info, stage_checkpoint_path
from paralome.comparisons import display_group, recorded_factors
from paralome.errors import ParalomeError

STAGE = "genome_tracks"
SCHEMA = "paralome.genome_tracks.checkpoint.v1"
METHOD = "paralome.genome_tracks.v1"
#: The DNA arm's reads (paralome.variants.genome_tracks): the same stage and checkpoint, <prefix>_DNA_reads.zip.
DNA_METHOD = "paralome.genome_tracks.dna_reads.v2"
MANIFEST_KIND = "paralome.genome_tracks"
MANIFEST_SCHEMA = 1
#: 
BIN_SIZE = 25
#: The viewer the zip is made for (OmicView).
OPEN_WITH = "OmicView's Genome Browser"
#: How many of the strongest differential peaks the manifest offers as places to start.
START_LOCI = 100
#: The UCSC tools the stage writes its tracks with.
TOOLS = ("bedGraphToBigWig", "bedToBigBed")

#: itemRgb of a comparison's peaks.
COLOUR_UP = "215,25,28"
COLOUR_DOWN = "44,123,182"
COLOUR_NO_DIRECTION = "150,150,150"

COMPARISON_AUTOSQL = """table paralomeDifferentialPeak
"A peak one comparison tested: its fold change, q-value and call (Paralome genome_tracks)"
(
string chrom;          "Chromosome"
uint   chromStart;     "Start (0-based)"
uint   chromEnd;       "End"
string name;           "Peak id"
uint   score;          "-10 x log10(p), at most 1000"
char[1] strand;        "No strand"
uint   thickStart;     "As chromStart"
uint   thickEnd;       "As chromEnd"
uint   reserved;       "Colour: higher red, lower blue, no direction grey"
string log2FoldChange; "log2 fold change, treatment over control (empty for an F-test)"
string pValue;         "p-value"
string qValue;         "q-value, BH within this comparison"
string direction;      "higher, lower or none"
)
"""


# ---------------------------------------------------------------------------------------------------------------
# what the differential tested


@dataclass(frozen=True)
class Peak:
    peak_id: str
    chrom: str
    start: int
    end: int


@dataclass
class ComparisonPeaks:
    """The peaks one comparison of one statistic (of one stratum) tested, with its results."""

    stratum: str
    folder: str
    comparison: str
    kind: str
    control: Optional[str]
    treatment: Optional[str]
    source: str
    #: (peak_id, log2 fold change or None, p-value or None, q-value or None)
    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = field(default_factory=list)

    @property
    def label(self) -> str:
        parts = [self.stratum, self.folder, self.comparison]
        return "__".join(part for part in parts if part)

    @property
    def directional(self) -> bool:
        return any(value is not None for _p, value, _q, _s in self.rows)


_PEAK_ID = re.compile(r"^(?P<chrom>.+?)[:\-_](?P<start>\d+)[\-_](?P<end>\d+)$")
_TRUE = {"true", "1", "yes", "t"}


def _number(text: object) -> Optional[float]:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _table_rows(path: Path) -> Iterator[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        first = handle.readline()
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
        handle.seek(0)
        yield from csv.DictReader(handle, delimiter=delimiter)


def _coordinates(row: Mapping[str, str]) -> Optional[Tuple[str, int, int]]:
    """A peak's place: its chrom/start/end columns, or else its id (chr1-100-200, chr1:100-200)."""
    chrom = str(row.get("chrom") or "").strip()
    start, end = _number(row.get("start")), _number(row.get("end"))
    if chrom and start is not None and end is not None and end > start:
        return chrom, int(start), int(end)
    match = _PEAK_ID.match(str(row.get("peak_id") or "").strip())
    if match and int(match["end"]) > int(match["start"]):
        return match["chrom"], int(match["start"]), int(match["end"])
    return None


def read_differential(checkpoint: Path) -> Tuple[Dict[str, Peak], List[ComparisonPeaks]]:
    """The union of the tested peaks, and each comparison's tested peaks, from a differential checkpoint.

    The checkpoint is read as motif enrichment reads it (atac.motifs._differential_sources): every comparison
    of every statistic folder, in every stratum of a stratified single-cell run. A peak counts as viable when
    any of them tested it (``tested`` in its machine table: it reached the expression floor of that
    comparison's samples).
    """
    from paralome.atac.motifs import _differential_sources

    sources = _differential_sources(None, Path(checkpoint), Path(checkpoint).parent)
    peaks: Dict[str, Peak] = {}
    comparisons: List[ComparisonPeaks] = []
    for source in sources:
        table = Path(str(source["path"]))
        entry = ComparisonPeaks(
            stratum=str(source.get("stratum") or ""), folder=str(source.get("folder") or ""),
            comparison=str(source.get("comparison") or ""), kind=str(source.get("kind") or ""),
            control=source.get("control"), treatment=source.get("treatment"), source=str(table),
        )
        for row in _table_rows(table):
            if str(row.get("tested") or "").strip().lower() not in _TRUE:
                continue
            peak_id = str(row.get("peak_id") or "").strip()
            place = _coordinates(row)
            if not peak_id or place is None:
                continue
            peaks.setdefault(peak_id, Peak(peak_id, *place))
            entry.rows.append((
                peak_id, _number(row.get("log2_fold_change")), _number(row.get("p_value")), _number(row.get("q_value")),
            ))
        comparisons.append(entry)
    if not peaks:
        raise ParalomeError(
            232,
            f"The differential in {checkpoint} tested no peak, so there are no viable peaks to export.",
            fix="Check the differential's results (each comparison's 'tested' count) before exporting tracks",
        )
    return peaks, comparisons


# ---------------------------------------------------------------------------------------------------------------
# where the profiles are drawn: the viable peaks, merged where they overlap, cut into 25 bp bins


def chrom_sort_key(chrom: str) -> bytes:
    """The order bedGraphToBigWig and bedToBigBed need: names compared byte by byte (``sort -k1,1``, C locale)."""
    return str(chrom).encode()


@dataclass
class Regions:
    """The viable peaks of one contig, merged where they overlap or touch, and their 25 bp bins.

    Coverage is held in REGION SPACE: the bases of the regions laid end to end, so an array per contig is as
    long as the peaks on it, not as the contig.
    """

    chrom: str
    starts: np.ndarray            # region starts, sorted
    ends: np.ndarray              # region ends
    offsets: np.ndarray           # region-space index of each region's first base
    length: int                   # region-space length (bases in all regions)
    bin_starts: np.ndarray        # genomic start of each bin
    bin_ends: np.ndarray          # genomic end of each bin
    bin_from: np.ndarray          # region-space index of each bin's first base
    bin_to: np.ndarray            # and one past its last

    @classmethod
    def build(cls, chrom: str, intervals: Iterable[Tuple[int, int]], bin_size: int = BIN_SIZE) -> "Regions":
        merged: List[List[int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        starts = np.array([m[0] for m in merged], dtype=np.int64)
        ends = np.array([m[1] for m in merged], dtype=np.int64)
        widths = ends - starts
        offsets = np.concatenate(([0], np.cumsum(widths)[:-1])).astype(np.int64) if len(widths) else np.zeros(0, np.int64)
        bins_s: List[np.ndarray] = []
        bins_e: List[np.ndarray] = []
        from_s: List[np.ndarray] = []
        for start, end, offset in zip(starts, ends, offsets):
            # genome-aligned 25 bp bins, cut at the region's edges
            first = (int(start) // bin_size + 1) * bin_size
            cuts = np.arange(first, int(end), bin_size, dtype=np.int64)
            b_s = np.concatenate(([start], cuts))
            b_e = np.concatenate((cuts, [end]))
            bins_s.append(b_s)
            bins_e.append(b_e)
            from_s.append(b_s - start + offset)
        bin_starts = np.concatenate(bins_s) if bins_s else np.zeros(0, np.int64)
        bin_ends = np.concatenate(bins_e) if bins_e else np.zeros(0, np.int64)
        bin_from = np.concatenate(from_s) if from_s else np.zeros(0, np.int64)
        return cls(chrom=chrom, starts=starts, ends=ends, offsets=offsets, length=int(widths.sum()),
                   bin_starts=bin_starts, bin_ends=bin_ends, bin_from=bin_from,
                   bin_to=bin_from + (bin_ends - bin_starts))


def build_regions(peaks: Iterable[Peak], bin_size: int = BIN_SIZE) -> Dict[str, Regions]:
    by_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for peak in peaks:
        by_chrom.setdefault(peak.chrom, []).append((peak.start, peak.end))
    return {chrom: Regions.build(chrom, intervals, bin_size) for chrom, intervals in by_chrom.items()}


def clip_to_regions(regions: Regions, starts: np.ndarray, ends: np.ndarray, owners: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each fragment's overlap with each region it touches, in region space: (from, to, owner)."""
    out_from: List[np.ndarray] = []
    out_to: List[np.ndarray] = []
    out_owner: List[np.ndarray] = []
    n = len(regions.starts)
    if not n or not len(starts):
        empty = np.zeros(0, np.int64)
        return empty, empty, empty
    index = np.searchsorted(regions.ends, starts, side="right")        # first region ending after the start
    starts, ends, owners = np.asarray(starts), np.asarray(ends), np.asarray(owners)
    while len(index):
        live = index < n
        index, starts, ends, owners = index[live], starts[live], ends[live], owners[live]
        if not len(index):
            break
        r_start = regions.starts[index]
        hit = r_start < ends
        index, starts, ends, owners, r_start = index[hit], starts[hit], ends[hit], owners[hit], r_start[hit]
        if not len(index):
            break
        r_end = regions.ends[index]
        lo = np.maximum(starts, r_start)
        hi = np.minimum(ends, r_end)
        offset = regions.offsets[index]
        out_from.append(offset + (lo - r_start))
        out_to.append(offset + (hi - r_start))
        out_owner.append(owners)
        more = ends > r_end                               # runs past this region: the next one may hold more
        index, starts, ends, owners = index[more] + 1, starts[more], ends[more], owners[more]
    # Fragments that touch no region (a batch between peaks, or past the last) give nothing, not
    # a failed concatenate (an snATAC run, the browser bundle's tracks: "need at least one array")
    if not out_from:
        empty = np.zeros(0, np.int64)
        return empty, empty, empty
    return np.concatenate(out_from), np.concatenate(out_to), np.concatenate(out_owner)


class BinSums:
    """Covered bases per bin, per sample, for every contig: the sum a profile is made from.

    ``add`` takes fragments contig by contig; a contig's coverage is turned into bin sums when the next contig
    begins (``flush``), so only one contig's region-space arrays are ever held. Input must therefore come in
    contig order -- a coordinate-sorted BAM and a tabix-indexed fragment file both do; a contig met again after
    another refuses.
    """

    def __init__(self, regions: Mapping[str, Regions], samples: int):
        self.regions = regions
        self.samples = samples
        self.sums: Dict[str, np.ndarray] = {}
        self._chrom: Optional[str] = None
        self._from: List[np.ndarray] = []
        self._to: List[np.ndarray] = []
        self._owner: List[np.ndarray] = []
        self._done: set = set()

    def add(self, chrom: str, starts, ends, owners) -> None:
        if chrom not in self.regions:
            return
        if chrom != self._chrom:
            self.flush()
            if chrom in self._done:
                raise ParalomeError(
                    233, f"The fragments are not sorted by contig: {chrom} came again after another contig.",
                    fix="Sort the BAM by coordinate (samtools sort) or the fragment file (sort -k1,1 -k2,2n) and index it",
                )
            self._chrom = chrom
        f, t, o = clip_to_regions(self.regions[chrom], np.asarray(starts, np.int64), np.asarray(ends, np.int64),
                                  np.asarray(owners, np.int64))
        if len(f):
            self._from.append(f)
            self._to.append(t)
            self._owner.append(o)

    def flush(self) -> None:
        chrom = self._chrom
        if chrom is None:
            return
        regions = self.regions[chrom]
        length = regions.length
        sums = self.sums.setdefault(chrom, np.zeros((self.samples, len(regions.bin_starts)), dtype=np.float64))
        if self._from:
            f = np.concatenate(self._from)
            t = np.concatenate(self._to)
            o = np.concatenate(self._owner)
            for sample in np.unique(o):
                mine = o == sample
                diff = (np.bincount(f[mine], minlength=length + 1)[: length + 1]
                        - np.bincount(t[mine], minlength=length + 1)[: length + 1])
                coverage = np.cumsum(diff[:length])
                prefix = np.concatenate(([0], np.cumsum(coverage)))
                sums[int(sample)] += prefix[regions.bin_to] - prefix[regions.bin_from]
        self._from, self._to, self._owner = [], [], []
        self._done.add(chrom)
        self._chrom = None


# ---------------------------------------------------------------------------------------------------------------
# writing the tracks


def require_tools() -> Dict[str, str]:
    """bedGraphToBigWig and bedToBigBed, by absolute path; PLM-0416 naming the missing ones."""
    from paralome.binaries import resolve_tool

    found = {name: resolve_tool(name) for name in TOOLS}
    missing = [name for name, path in found.items() if not path]
    if missing:
        raise ParalomeError(
            416,
            f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed or not on PATH; the "
            "genome_tracks stage writes its bigWig and bigBed files with them.",
            fix="conda install -c bioconda ucsc-bedgraphtobigwig ucsc-bedtobigbed, then paralome setup",
        )
    return {name: str(path) for name, path in found.items()}


def _run(command: Sequence[str]) -> None:
    proc = subprocess.run([str(part) for part in command], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{Path(str(command[0])).name} failed ({proc.returncode}): "
                           f"{(proc.stderr or proc.stdout).strip()[:2000]}")


def write_chrom_sizes(path: Path, sizes: Mapping[str, int]) -> None:
    lines = [f"{chrom}\t{int(size)}" for chrom, size in sorted(sizes.items(), key=lambda kv: chrom_sort_key(kv[0]))]
    Path(path).write_text("\n".join(lines) + "\n")


def write_bigwig(path: Path, sums: Mapping[str, np.ndarray], regions: Mapping[str, Regions], factor: float,
                 chrom_sizes: Path, tool: str, scratch: Path) -> Dict[str, object]:
    """One sample's bin sums as a bigWig: covering fragments per base, scaled by ``factor`` (CPM)."""
    graph = Path(scratch) / (Path(path).stem + ".bedGraph")
    bins = 0
    peak_value = 0.0
    with graph.open("w") as handle:
        for chrom in sorted(sums, key=chrom_sort_key):
            r = regions[chrom]
            if not len(r.bin_starts):
                continue
            widths = (r.bin_ends - r.bin_starts).astype(np.float64)
            values = sums[chrom] / widths * factor
            bins += len(values)
            if len(values):
                peak_value = max(peak_value, float(values.max()))
            handle.write("".join(
                f"{chrom}\t{s}\t{e}\t{v:.6g}\n" for s, e, v in zip(r.bin_starts.tolist(), r.bin_ends.tolist(), values.tolist())
            ))
    _run([tool, str(graph), str(chrom_sizes), str(path)])
    graph.unlink(missing_ok=True)
    return {"bins": bins, "max_cpm": round(peak_value, 6)}


def _bigbed(path: Path, lines: List[str], chrom_sizes: Path, tool: str, scratch: Path, *,
            bed_type: str, autosql: Optional[str] = None) -> int:
    bed = Path(scratch) / (Path(path).stem + ".bed")
    bed.write_text("".join(lines))
    command = [tool, "-tab", f"-type={bed_type}"]
    if autosql:
        spec = Path(scratch) / (Path(path).stem + ".as")
        spec.write_text(autosql)
        command.append(f"-as={spec}")
    _run(command + [str(bed), str(chrom_sizes), str(path)])
    bed.unlink(missing_ok=True)
    return len(lines)


def _sorted_peaks(peaks: Iterable[Peak]) -> List[Peak]:
    return sorted(peaks, key=lambda p: (chrom_sort_key(p.chrom), p.start, p.end, p.peak_id))


def write_viable_peaks(path: Path, peaks: Mapping[str, Peak], sizes: Mapping[str, int], chrom_sizes: Path,
                       tool: str, scratch: Path) -> int:
    lines = [f"{p.chrom}\t{p.start}\t{p.end}\t{p.peak_id}\n" for p in _sorted_peaks(peaks.values()) if p.chrom in sizes]
    return _bigbed(path, lines, chrom_sizes, tool, scratch, bed_type="bed4")


def comparison_direction(log2fc: Optional[float], directional: bool) -> Tuple[str, str]:
    """(direction, itemRgb) of one tested peak: its fold change's sign, no call of significance."""
    if not directional or log2fc is None or log2fc == 0:
        return "none", COLOUR_NO_DIRECTION
    return ("higher", COLOUR_UP) if log2fc > 0 else ("lower", COLOUR_DOWN)


def write_comparison_peaks(path: Path, entry: ComparisonPeaks, peaks: Mapping[str, Peak], sizes: Mapping[str, int],
                           chrom_sizes: Path, tool: str, scratch: Path) -> Dict[str, int]:
    directional = entry.directional
    rows = []
    counts = {"tested": 0, "higher": 0, "lower": 0, "none": 0}
    for peak_id, log2fc, p, q in entry.rows:
        peak = peaks[peak_id]
        if peak.chrom not in sizes:
            continue
        direction, colour = comparison_direction(log2fc, directional)
        score = 0 if p is None else int(min(1000, round(-10 * math.log10(max(p, 1e-100)))))
        rows.append((peak, (
            f"{peak.chrom}\t{peak.start}\t{peak.end}\t{peak.peak_id}\t{score}\t.\t{peak.start}\t{peak.end}\t{colour}\t"
            f"{'' if log2fc is None else f'{log2fc:.6g}'}\t{'' if p is None else f'{p:.6g}'}\t"
            f"{'' if q is None else f'{q:.6g}'}\t{direction}\n"
        )))
        counts["tested"] += 1
        counts[direction] += 1
    rows.sort(key=lambda item: (chrom_sort_key(item[0].chrom), item[0].start, item[0].end, item[0].peak_id))
    _bigbed(path, [line for _peak, line in rows], chrom_sizes, tool, scratch, bed_type="bed9+4", autosql=COMPARISON_AUTOSQL)
    return counts


# --- gene models from the GTF ----------------------------------------------------------------------------------

_ATTR = re.compile(r'([A-Za-z0-9_]+)\s+"([^"]*)"')


def _contig_key(chrom: str) -> str:
    bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
    return "MT" if bare.upper() in {"M", "MT"} else bare


def _contig_namer(names: Iterable[str]):
    """A GTF's contig name to the tracks' spelling of it, with or without "chr" (as peak counting matches)."""
    exact = set(names)
    by_key: Dict[str, set] = {}
    for name in exact:
        by_key.setdefault(_contig_key(name), set()).add(name)

    def name(chrom: str) -> Optional[str]:
        if chrom in exact:
            return chrom
        found = by_key.get(_contig_key(chrom), set())
        return next(iter(found)) if len(found) == 1 else None

    return name


def gene_models(gtf: Path, sizes: Mapping[str, int]) -> List[str]:
    """BED12 lines, one per transcript: where the GTF tags Ensembl_canonical, those transcripts only."""
    opener = gzip.open if str(gtf).endswith(".gz") else open
    namer = _contig_namer(sizes)
    transcripts: Dict[str, Dict[str, object]] = {}
    canonical_seen = False
    with opener(gtf, "rt") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in ("transcript", "exon"):
                continue
            attributes = dict(_ATTR.findall(fields[8]))
            tid = attributes.get("transcript_id")
            if not tid:
                continue
            chrom = namer(fields[0])
            if chrom is None:
                continue
            entry = transcripts.setdefault(tid, {
                "chrom": chrom, "strand": fields[6] if fields[6] in "+-" else ".",
                "name": attributes.get("gene_name") or attributes.get("gene_id") or tid,
                "exons": [], "start": None, "end": None, "canonical": False,
            })
            start, end = int(fields[3]) - 1, int(fields[4])
            if fields[2] == "transcript":
                entry["start"], entry["end"] = start, end
                if "Ensembl_canonical" in fields[8]:
                    entry["canonical"] = True
                    canonical_seen = True
            else:
                entry["exons"].append((start, end))
    lines = []
    for tid, entry in transcripts.items():
        if canonical_seen and not entry["canonical"]:
            continue
        exons = sorted(entry["exons"])
        if not exons:
            continue
        start = min(entry["start"] if entry["start"] is not None else exons[0][0], exons[0][0])
        end = max(entry["end"] if entry["end"] is not None else exons[-1][1], exons[-1][1])
        chrom = str(entry["chrom"])
        if end > int(sizes.get(chrom, 0)):
            continue
        sizes_text = ",".join(str(e - s) for s, e in exons) + ","
        starts_text = ",".join(str(s - start) for s, _e in exons) + ","
        name = str(entry["name"]).replace(" ", "_")
        lines.append((chrom, start, end, f"{chrom}\t{start}\t{end}\t{name}\t0\t{entry['strand']}\t{start}\t{end}\t0\t"
                                         f"{len(exons)}\t{sizes_text}\t{starts_text}\n"))
    lines.sort(key=lambda item: (chrom_sort_key(item[0]), item[1], item[2]))
    return [line for *_place, line in lines]


def write_genes(path: Path, gtf: Path, sizes: Mapping[str, int], chrom_sizes: Path, tool: str, scratch: Path) -> int:
    lines = gene_models(gtf, sizes)
    if not lines:
        return 0
    return _bigbed(path, lines, chrom_sizes, tool, scratch, bed_type="bed12")


# ---------------------------------------------------------------------------------------------------------------
# the zip


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text)).strip("._")
    return cleaned or "track"


def tracks_zip_path(workdir: Path, prefix: str, assay: str, zip_dir: Optional[Path] = None) -> Path:
    """``<prefix>_<ATAC|snATAC>_genome_tracks.zip`` -- the modality after the prefix.
    In ``zip_dir``, the project's omicview/genome_tracks folder, else workdir/results."""
    return Path(zip_dir if zip_dir else Path(workdir) / "results") / f"{prefix}_{assay}_genome_tracks.zip"


def start_loci(comparisons: Sequence[ComparisonPeaks], peaks: Mapping[str, Peak], limit: int = START_LOCI) -> List[Dict]:
    """The peaks with the smallest p across the comparisons, for the viewer to offer first (an order, not a cut)."""
    best: Dict[str, Tuple[float, Dict]] = {}
    for entry in comparisons:
        for peak_id, log2fc, p, q in entry.rows:
            if p is None:
                continue
            if peak_id in best and best[peak_id][0] <= p:
                continue
            peak = peaks[peak_id]
            best[peak_id] = (p, {
                "locus": f"{peak.chrom}:{peak.start + 1}-{peak.end}", "peak_id": peak_id,
                "comparison": entry.label, "p_value": p, "q_value": q, "log2_fold_change": log2fc,
            })
    ranked = sorted(best.values(), key=lambda item: (item[0], item[1]["peak_id"]))
    return [record for _q, record in ranked[:limit]]


def build_zip(out: Path, members: Sequence[Tuple[Path, str]], manifest: Mapping[str, object]) -> None:
    """Write the zip stored, manifest first, beside ``out`` and then into its place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for source, name in members:
            archive.write(source, name)
    os.replace(partial, out)


# ---------------------------------------------------------------------------------------------------------------
# the stage: request, checkpoint, summary


def _fingerprint(path: Path) -> Dict[str, object]:
    stat = Path(path).stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _request_id(request: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def zip_members(path: Path) -> Dict[str, int]:
    """Each member of a zip and its size; {} when it is not a readable zip."""
    try:
        with zipfile.ZipFile(path) as archive:
            return {info.filename: info.file_size for info in archive.infolist()}
    except (OSError, zipfile.BadZipFile):
        return {}


def tracks_there(checkpoint: Mapping[str, object], path: Optional[Path]) -> bool:
    """Whether the zip a checkpoint names still holds its tracks.

    The tracks travelled in the arm's browser bundle, which held the results beside them, so
    the zip is checked for the members the tracks wrote (their names and sizes), not for its own size (their own zip again). A checkpoint written recorded only the zip's size."""
    if path is None or not Path(path).is_file():
        return False
    members = checkpoint.get("members")
    if isinstance(members, dict) and members:
        present = zip_members(Path(path))
        return all(present.get(name) == size for name, size in members.items())
    return Path(path).stat().st_size == checkpoint.get("zip_bytes")


def recorded_zip(checkpoint: Mapping[str, object], workdir: Path) -> Optional[Path]:
    """The zip the tracks are in: where the checkpoint says, else by name in workdir/results (a copied project)."""
    recorded = (checkpoint.get("outputs") or {}).get("zip") if isinstance(checkpoint.get("outputs"), dict) else None
    if not recorded:
        return None
    path = Path(str(recorded))
    if not path.is_file() and (Path(workdir) / "results" / path.name).is_file():
        path = Path(workdir) / "results" / path.name
    return path


def _kept(checkpoint_path: Path, out: Path, request_id: str, overwrite: bool) -> Optional[Dict[str, object]]:
    try:
        previous = json.loads(checkpoint_path.read_text())
    except (OSError, ValueError):
        return None
    if overwrite or previous.get("status") != "complete" or previous.get("request_id") != request_id:
        return None
    if previous.get("method") != METHOD:
        return None
    where = recorded_zip(previous, checkpoint_path.parent.parent) or out
    # The tracks are this stage's own zip; tracks recorded in another zip (a browser bundle of an
    # older Paralome) are taken over by adopt_bundled_tracks, not kept where they are.
    if Path(where).resolve() != Path(out).resolve():      # And in the place this run writes it
        return None
    if tracks_there(previous, where):
        return previous
    return None


def _inside(path: Path, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def _move_former_tracks(former: Path, workdir: Path, out: Path, mine_path: Path, assay: str,
                        echo: Callable[[str], None]) -> bool:
    """SnATAC's tracks moved from their former folder, <project>/atac/genome_tracks, to
    <project>/genome_tracks with their checkpoint (renamed to this version's name on the way); kept there when
    their request still matches. True when moved."""
    try:
        ckpt = json.loads(stage_checkpoint_path(former, STAGE).read_text())
    except (OSError, ValueError):
        return False
    archive = recorded_zip(ckpt, former)
    if (ckpt.get("status") != "complete" or ckpt.get("method") != METHOD or ckpt.get("assay") != assay
            or not archive or Path(archive).parent != former / "results" or not tracks_there(ckpt, archive)):
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(archive, out)
    except OSError:
        shutil.copy2(archive, out)
    taken = dict(ckpt)
    taken["outputs"] = dict(ckpt.get("outputs") or {}, zip=str(out))
    taken["moved_from"] = str(archive)
    mine_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(mine_path, json.dumps(taken, indent=2, sort_keys=True, default=str) + "\n")
    echo(f"genome_tracks: {archive} moved to {out}; kept if they still answer this run")
    return True


def adopt_bundled_tracks(bundle_workdir: Optional[Path], workdir: Path, prefix: str, assay: str,
                         echo: Callable[[str], None] = print, former_workdir: Optional[Path] = None,
                         zip_dir: Optional[Path] = None) -> None:
    """Tracks an older browser_bundle stage made inside its bundle are copied out, member by member, into this
    stage's own zip, with their checkpoint, so they are not made again when nothing they read has changed (the
    request is still compared). The bundle is left as it is. A zip under the older name
    <prefix>_genome_tracks.zip is renamed <prefix>_<assay>_genome_tracks.zip."""
    out = tracks_zip_path(workdir, prefix, assay, zip_dir)
    mine_path = stage_checkpoint_path(Path(workdir), STAGE)
    try:
        mine = json.loads(mine_path.read_text())
    except (OSError, ValueError):
        mine = {}
    here = recorded_zip(mine, Path(workdir))
    if mine.get("status") == "complete" and tracks_there(mine, here):
        if Path(here).resolve() == out.resolve():
            return
        if mine.get("assay") == assay and _inside(Path(here), Path(workdir)):
            # This stage's own zip where an older Paralome put it -- under its old name,
            # <prefix>_genome_tracks.zip, or outside the omicview folder: moved.
            out.parent.mkdir(parents=True, exist_ok=True)
            os.replace(here, out)
            mine["outputs"] = dict(mine.get("outputs") or {}, zip=str(out))
            atomic_write_text(mine_path, json.dumps(mine, indent=2, sort_keys=True, default=str) + "\n")
            echo(f"genome_tracks: {here} moved to {out}")
            return
    if former_workdir and Path(former_workdir).resolve() != Path(workdir).resolve() and \
            _move_former_tracks(Path(former_workdir), Path(workdir), out, mine_path, assay, echo):
        return
    if not bundle_workdir:
        return
    try:
        ckpt = json.loads(stage_checkpoint_path(Path(bundle_workdir), STAGE).read_text())
    except (OSError, ValueError):
        return
    archive = recorded_zip(ckpt, Path(bundle_workdir))
    members = ckpt.get("members")
    if (ckpt.get("status") != "complete" or ckpt.get("method") != METHOD or ckpt.get("assay") != assay
            or not isinstance(members, dict)
            or not members or not tracks_there(ckpt, archive)):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(archive) as source, \
            zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
        for info in source.infolist():
            if info.filename in members:
                with source.open(info) as reader, target.open(info.filename, "w", force_zip64=True) as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)
    os.replace(partial, out)
    taken = dict(ckpt)
    taken["outputs"] = dict(ckpt.get("outputs") or {}, zip=str(out))
    taken["zip_bytes"] = out.stat().st_size
    taken["members"] = zip_members(out)
    taken["adopted_from"] = str(archive)
    mine_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(mine_path, json.dumps(taken, indent=2, sort_keys=True, default=str) + "\n")
    echo(f"genome_tracks: the tracks in {archive} (an older browser_bundle) copied to {out}; kept if they "
         "still answer this run")


def _begin(checkpoint_path: Path, request: Mapping[str, object], request_id: str) -> str:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    atomic_write_text(checkpoint_path, json.dumps({
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "incomplete", "complete": False,
        "request": request, "request_id": request_id, "started": started,
        "reason": f"building since {started}; a run that stopped here did not finish",
    }, indent=2, sort_keys=True, default=str) + "\n")
    return started


def summarize(workdir: Optional[Path], zip_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Stage state for project_state.json: complete when the checkpoint says so, its method is this one and the
    zip it names is there at its recorded size -- in ``zip_dir`` when given."""
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    path = stage_checkpoint_path(wd, STAGE)
    try:
        ckpt = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "not_available", "workdir": str(wd), "complete": False, "checkpoint": file_info(path)}
    archive = recorded_zip(ckpt, wd)
    if archive is not None and zip_dir and (Path(zip_dir) / Path(archive).name).is_file():
        archive = Path(zip_dir) / Path(archive).name   # a copied project (its omicview folder)
    info = file_info(str(archive) if archive else None)
    status = str(ckpt.get("status") or "unknown")
    reason = ckpt.get("reason")
    if status == "complete":
        if ckpt.get("method") not in (METHOD, DNA_METHOD):
            status, reason = "incomplete", f"built by {ckpt.get('method')}; the method is now {METHOD}"
        elif not tracks_there(ckpt, archive):
            status, reason = "incomplete", "the tracks zip is missing or not the one recorded"
        elif zip_dir and archive is not None and Path(archive).resolve() != (Path(zip_dir) / Path(archive).name).resolve():
            # Issued again, the stage moves it there (adopt_bundled_tracks).
            status, reason = "incomplete", f"{Path(archive).name} is not in {zip_dir} (the zips are in the omicview folder)"
    return {"status": status, "complete": status == "complete", "workdir": str(wd), "reason": reason,
            "checkpoint": file_info(path), "zip": info}


@dataclass
class TrackSample:
    """One profile: who it is, what it is labelled with, and its CPM factor."""

    sample_id: str
    group: str = ""
    stratum: str = ""
    factor: Optional[float] = None
    eligible_fragments: Optional[int] = None


def _finish(
    *, assay: str, prefix: str, workdir: Path, out: Path, checkpoint_path: Path, request: Mapping[str, object],
    request_id: str, started: str, peaks: Mapping[str, Peak], comparisons: Sequence[ComparisonPeaks],
    regions: Mapping[str, Regions], sizes: Mapping[str, int], samples: Sequence[TrackSample],
    sums: Mapping[str, np.ndarray], gtf: Optional[Path], genome: Optional[str], tools: Mapping[str, str],
    echo: Callable[[str], None], extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Write every track, the manifest and the zip; then the checkpoint."""
    from paralome.version import __version__

    with tempfile.TemporaryDirectory(prefix="genome_tracks_", dir=str(workdir)) as tmp:
        scratch = Path(tmp)
        chrom_sizes = scratch / "chrom.sizes"
        write_chrom_sizes(chrom_sizes, sizes)
        members: List[Tuple[Path, str]] = [(chrom_sizes, "chrom.sizes")]

        viable = scratch / f"{prefix}_viable_peaks.bb"
        viable_count = write_viable_peaks(viable, peaks, sizes, chrom_sizes, tools["bedToBigBed"], scratch)
        members.append((viable, f"peaks/{viable.name}"))
        echo(f"genome_tracks: {viable_count} viable peaks (the union of every comparison's tested peaks)")

        comparison_records = []
        used: set = set()
        for entry in comparisons:
            name = safe_name(entry.label) or "comparison"
            while name in used:
                name += "_"
            used.add(name)
            path = scratch / f"{name}.bb"
            counts = write_comparison_peaks(path, entry, peaks, sizes, chrom_sizes, tools["bedToBigBed"], scratch)
            members.append((path, f"peaks/{path.name}"))
            comparison_records.append({
                "name": entry.label, "path": f"peaks/{path.name}", "stratum": entry.stratum or None,
                "folder": entry.folder, "comparison": entry.comparison, "kind": entry.kind,
                "control": entry.control, "treatment": entry.treatment, "directional": entry.directional,
                **counts,
            })

        track_records = []
        used = set()
        for index, sample in enumerate(samples):
            name = safe_name(sample.sample_id)
            while name in used:
                name += "_"
            used.add(name)
            path = scratch / f"{name}.bw"
            sample_sums = {chrom: matrix[index] for chrom, matrix in sums.items()}
            facts = write_bigwig(path, sample_sums, regions, float(sample.factor or 0.0), chrom_sizes,
                                 tools["bedGraphToBigWig"], scratch)
            members.append((path, f"coverage/{path.name}"))
            track_records.append({
                "name": sample.sample_id, "path": f"coverage/{path.name}", "type": "bigwig",
                "group": sample.group or None, "stratum": sample.stratum or None,
                "eligible_fragments": sample.eligible_fragments, "cpm_factor": sample.factor, **facts,
            })
        echo(f"genome_tracks: {len(track_records)} sample profiles, {BIN_SIZE} bp bins inside the viable peaks")

        genes_record = None
        if gtf and Path(gtf).is_file():
            genes = scratch / "genes.bb"
            count = write_genes(genes, Path(gtf), sizes, chrom_sizes, tools["bedToBigBed"], scratch)
            if count:
                members.append((genes, "annotation/genes.bb"))
                genes_record = {"path": "annotation/genes.bb", "source": str(gtf), "transcripts": count}

        manifest = {
            "kind": MANIFEST_KIND, "schema": MANIFEST_SCHEMA, "program": "Paralome", "program_version": __version__,
            "open_with": OPEN_WITH, "assay": assay, "project": prefix, "genome": genome,
            "chrom_sizes": "chrom.sizes", "bin_size": BIN_SIZE,
            "coverage": ("fragment coverage inside the viable peaks, in CPM: covering fragments per base per "
                         "million eligible fragments of the sample; no data outside the viable peaks"),
            "viable_peaks": {"path": f"peaks/{viable.name}", "count": viable_count,
                             "rule": "the union of the peaks tested in any comparison"},
            "comparisons": comparison_records,
            "tracks": track_records,
            "genes": genes_record,
            "loci": start_loci(comparisons, peaks),
        }
        build_zip(out, members, manifest)

    checkpoint = {
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "complete", "complete": True,
        "request": request, "request_id": request_id, "started": started, "finished": now_iso(),
        "program_version": __version__, "open_with": OPEN_WITH, "assay": assay,
        "outputs": {"zip": str(out)}, "zip_bytes": out.stat().st_size, "members": zip_members(out),
        "viable_peaks": viable_count, "comparisons": len(comparison_records), "samples": len(track_records),
        "bin_size": BIN_SIZE, "genes": bool(genes_record),
        **(dict(extra) if extra else {}),
    }
    atomic_write_text(checkpoint_path, json.dumps(checkpoint, indent=2, sort_keys=True, default=str) + "\n")
    echo(f"genome_tracks: {out} ({out.stat().st_size / 1e6:.1f} MB); open it in {OPEN_WITH}")
    return checkpoint


def _resolve_annotation(genome: Optional[str], gtf: Optional[Path], reference: Optional[Path]
                        ) -> Tuple[Optional[Path], Optional[Path]]:
    """The GTF and the reference FASTA: explicit paths first, then the registered genome's."""
    if genome:
        from paralome.pipeline.resource_catalog import resolve_genome_resources

        resolved = resolve_genome_resources(genome=genome, overrides={"gtf": gtf, "reference": reference})
        gtf = gtf or resolved.path("gtf")
        reference = reference or resolved.path("reference")
    return (Path(gtf) if gtf else None), (Path(reference) if reference else None)


# ---------------------------------------------------------------------------------------------------------------
# bulk ATAC: the deduplicated BAMs


def bam_contig_sizes(bam: Path, samtools: str) -> Dict[str, int]:
    proc = subprocess.run([samtools, "view", "-H", str(bam)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"samtools view -H failed for {bam}: {proc.stderr.strip()[:500]}")
    sizes: Dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("@SQ"):
            continue
        tags = dict(part.split(":", 1) for part in line.split("\t")[1:] if ":" in part)
        if "SN" in tags and "LN" in tags:
            sizes[tags["SN"]] = int(tags["LN"])
    return sizes


def bam_sample_sums(bam: str, regions: Mapping[str, Regions], samtools: str, batch: int = 500_000
                    ) -> Dict[str, np.ndarray]:
    """One sample's bin sums from its deduplicated BAM, its fragments as peak quantification takes them."""
    from paralome.atac.quantification import _bam_contig_resolver, fragment_interval

    resolve = _bam_contig_resolver(regions.keys())
    sums = BinSums(regions, 1)
    chrom: Optional[str] = None
    starts: List[int] = []
    ends: List[int] = []

    def flush_batch() -> None:
        if chrom is not None and starts:
            sums.add(chrom, starts, ends, np.zeros(len(starts), np.int64))
        starts.clear()
        ends.clear()

    proc = subprocess.Popen([samtools, "view", "-f", "66", "-F", "3852", str(bam)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        fields = line.split("\t", 9)
        if len(fields) < 9:
            continue
        contig = resolve(fields[2])
        if contig is None:
            continue
        interval = fragment_interval(fields)
        if interval is None:
            continue
        if contig != chrom:
            flush_batch()
            chrom = contig
        starts.append(interval[0])
        ends.append(interval[1])
        if len(starts) >= batch:
            flush_batch()
    if proc.wait():
        raise RuntimeError(f"samtools view failed for {bam} with exit code {proc.returncode}")
    flush_batch()
    sums.flush()
    return {chrom: matrix[0] for chrom, matrix in sums.sums.items()}


def _bam_job(args: Tuple[str, Mapping[str, Regions], str]) -> Dict[str, np.ndarray]:
    bam, regions, samtools = args
    return bam_sample_sums(bam, regions, samtools)


def run_bulk_genome_tracks(
    *,
    differential_checkpoint: Path,
    samples: Path,
    coverage_summary: Path,
    workdir: Path,
    prefix: str,
    genome: Optional[str] = None,
    gtf: Optional[Path] = None,
    threads: int = 1,
    zip_dir: Optional[Path] = None,
    overwrite: bool = False,
    echo: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Bulk ATAC: the viable peaks, each comparison's peaks and one profile per sample of the manifest."""
    from paralome.atac.locus_view import load_coverage_scaling
    from paralome.binaries import resolve_tool

    differential_checkpoint, samples, coverage_summary = Path(differential_checkpoint), Path(samples), Path(coverage_summary)
    for label, path in (("the differential's checkpoint", differential_checkpoint), ("the sample manifest", samples),
                        ("peak quantification's summary", coverage_summary)):
        if not path.is_file():
            raise ParalomeError(104, f"{label} {path} is not there", fix="Run the stages before genome_tracks first")
    with samples.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get("sample_id") or "").strip()]
    bams = {str(row["sample_id"]).strip(): str(row.get("bam") or "").strip() for row in rows}
    missing = [sample for sample, bam in bams.items() if not bam or not Path(bam).is_file()]
    if missing:
        raise ParalomeError(104, f"no deduplicated BAM for {', '.join(missing)} in {samples}",
                            fix="Run deduplication first, or check the bam column of the manifest")
    gtf_path, _reference = _resolve_annotation(genome, gtf, None)
    workdir = Path(workdir)
    out = tracks_zip_path(workdir, prefix, "ATAC", zip_dir)
    checkpoint_path = stage_checkpoint_path(workdir, STAGE)
    request = {
        "method": METHOD, "assay": "ATAC", "bin_size": BIN_SIZE,
        "differential": _fingerprint(differential_checkpoint),
        "samples": {sample: _fingerprint(Path(bam)) for sample, bam in sorted(bams.items())},
        "coverage_summary": _fingerprint(coverage_summary),
        "gtf": _fingerprint(gtf_path) if gtf_path and gtf_path.is_file() else None,
    }
    request_id = _request_id(request)
    kept = _kept(checkpoint_path, out, request_id, overwrite)
    if kept is not None:
        echo(f"genome_tracks: the tracks in {(recorded_zip(kept, workdir) or out).name} answer this request; kept")
        return kept
    tools = require_tools()
    samtools = resolve_tool("samtools") or "samtools"
    started = _begin(checkpoint_path, request, request_id)

    peaks, comparisons = read_differential(differential_checkpoint)
    regions = build_regions(peaks.values())
    factors = load_coverage_scaling(coverage_summary)
    unscaled = [sample for sample in bams if sample not in factors]
    if unscaled:
        raise ParalomeError(
            104, f"peak quantification's summary {coverage_summary} has no eligible fragments for {', '.join(unscaled)}",
            fix="Run peak_quantification again for these samples before genome_tracks",
        )
    first = next(iter(bams.values()))
    from paralome.atac.quantification import _bam_contig_resolver

    resolve = _bam_contig_resolver(regions.keys())
    sizes: Dict[str, int] = {}
    for contig, length in bam_contig_sizes(Path(first), samtools).items():
        name = resolve(contig)
        if name is not None:
            sizes[name] = length
    lost = sorted(set(regions) - set(sizes))
    if lost:
        echo(f"genome_tracks: the BAM header names no length for {', '.join(lost)}; their peaks are left out")
        regions = {chrom: r for chrom, r in regions.items() if chrom in sizes}
        peaks = {pid: p for pid, p in peaks.items() if p.chrom in sizes}

    order = list(bams)
    echo(f"genome_tracks: reading {len(order)} BAM(s) on {max(1, min(int(threads or 1), len(order)))} process(es)")
    if int(threads or 1) > 1 and len(order) > 1:
        with ProcessPoolExecutor(max_workers=min(int(threads), len(order))) as pool:
            per_sample = list(pool.map(_bam_job, [(bams[s], regions, samtools) for s in order]))
    else:
        per_sample = [bam_sample_sums(bams[s], regions, samtools) for s in order]
    sums = {chrom: np.vstack([one.get(chrom, np.zeros(len(r.bin_starts))) for one in per_sample])
            for chrom, r in regions.items()}
    # Each profile is labelled with its value of the run's first --factor.
    column, _control = display_group(recorded_factors(differential_checkpoint))
    groups = {str(row["sample_id"]).strip(): str(row.get(column) or "").strip() if column else "" for row in rows}
    track_samples = [TrackSample(sample_id=s, group=groups.get(s, ""), factor=factors[s],
                                 eligible_fragments=int(round(1_000_000.0 / factors[s]))) for s in order]
    return _finish(
        assay="ATAC", prefix=prefix, workdir=workdir, out=out, checkpoint_path=checkpoint_path, request=request,
        request_id=request_id, started=started, peaks=peaks, comparisons=comparisons, regions=regions, sizes=sizes,
        samples=track_samples, sums=sums, gtf=gtf_path, genome=genome, tools=tools, echo=echo,
    )


# ---------------------------------------------------------------------------------------------------------------
# snATAC: the fragment files the Signac object names


R_SCRIPT = Path(__file__).resolve().parent / "r" / "fragment_sources.R"


def fragment_sources(rds: Path, out: Path, *, rscript: str = "Rscript", assay: Optional[str] = None) -> Path:
    """Run fragment_sources.R on one object: a table of (path, cell, barcode) for every fragment file it names."""
    command = [rscript, str(R_SCRIPT), "--rds", str(rds), "--out", str(out)]
    if assay:
        command += ["--assay", assay]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"fragment_sources.R failed for {rds} ({proc.returncode}): "
                           f"{(proc.stderr or proc.stdout).strip()[-2000:]}")
    return out


def read_fragment_sources(path: Path, cell_prefix: str = "") -> Dict[str, Dict[str, str]]:
    """{fragment file: {fragment-file barcode: cell name in the extraction}}."""
    files: Dict[str, Dict[str, str]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            file_path = str(row.get("path") or "").strip()
            cell = str(row.get("cell") or "").strip()
            barcode = str(row.get("barcode") or "").strip() or cell
            if file_path and cell:
                files.setdefault(file_path, {})[barcode] = cell_prefix + cell
    return files


def check_fragment_files(files: Mapping[str, object], objects: Sequence[Path]) -> None:
    """Every fragment file the object(s) name, there with its tabix index; PLM-0231 listing them all otherwise."""
    missing = [path for path in files if not Path(path).is_file()]
    if missing or not files:
        listed = "\n".join(f"  {'MISSING ' if path in missing else 'found   '}{path}" for path in files) or "  (none)"
        raise ParalomeError(
            231,
            f"{'The Signac object names no fragment file' if not files else f'{len(missing)} fragment file(s) the Signac object names cannot be found'} "
            f"({', '.join(str(o) for o in objects)}). Every fragment file the object holds:\n{listed}",
            fix="In R: Fragments(obj)[[i]] <- UpdatePath(Fragments(obj)[[i]], new.path = '<real path>'); saveRDS(obj, ...)",
        )


def pseudobulk_membership(extraction_dir: Path, manifest: Path, group_by: Sequence[str],
                          stratify_by: Optional[str] = None, group_column: Optional[str] = None,
                          ) -> Tuple[Dict[str, int], List[TrackSample]]:
    """Each extracted cell's pseudobulk sample, as the aggregation made them (only the samples in the manifest).

    A profile is labelled with its value of ``group_column``, the run's first --factor.
    """
    from paralome.atac.singlecell import read_cell_metadata, sanitize_sample_id

    with Path(manifest).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get("sample_id") or "").strip()]
    from paralome.stratify import JOIN, missing_columns, stratum_columns

    strata = stratum_columns(stratify_by) if rows and not missing_columns(stratify_by, list(rows[0])) else []
    samples = [TrackSample(sample_id=str(row["sample_id"]).strip(),
                           group=str(row.get(group_column) or "").strip() if group_column else "",
                           stratum=JOIN.join(str(row.get(c) or "").strip() for c in strata) if strata else "")
               for row in rows]
    index = {sample.sample_id: i for i, sample in enumerate(samples)}
    metadata = read_cell_metadata(Path(extraction_dir) / "cell_metadata.tsv")
    columns = list(group_by)
    missing = [c for c in columns if c not in metadata.columns]
    if missing:
        raise ParalomeError(201, f"--replicate-column/--stratify-by column(s) {', '.join(missing)} are not in the extraction's cell metadata")
    labels = metadata[columns].astype(str).apply(
        lambda values: "__".join(sanitize_sample_id(str(v)) for v in values), axis=1)
    membership = {str(cell): index[label] for cell, label in zip(metadata["barcode"], labels) if label in index}
    return membership, samples


def fragment_file_sums(path: str, barcodes: Mapping[str, str], membership: Mapping[str, int],
                       regions: Mapping[str, Regions], samples: int, sums: Dict[str, np.ndarray],
                       eligible: np.ndarray, sizes: Dict[str, int], chunk: int = 2_000_000) -> int:
    """Add one fragment file's fragments to the samples' bin sums; returns the fragments read."""
    import pandas as pd

    from paralome.atac.quantification import is_mitochondrial_contig

    owner_of = {barcode: membership[cell] for barcode, cell in barcodes.items() if cell in membership}
    namer = _contig_namer(regions.keys())
    file_sums = BinSums(regions, samples)
    read = 0
    reader = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "barcode"],
                         dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str}, chunksize=chunk)
    for frame in reader:
        read += len(frame)
        owners = frame["barcode"].map(owner_of)
        keep = owners.notna().to_numpy()
        if not keep.any():
            continue
        frame = frame.loc[keep]
        owners = owners[keep].astype(np.int64).to_numpy()
        chroms = frame["chrom"].to_numpy()
        nuclear = np.array([not is_mitochondrial_contig(c) for c in chroms]) if len(chroms) else np.zeros(0, bool)
        np.add.at(eligible, owners[nuclear], 1)
        starts = frame["start"].to_numpy()
        ends = frame["end"].to_numpy()
        for c in pd.unique(chroms):
            name = namer(str(c))
            mine = chroms == c
            if name is None:
                continue
            sizes[name] = max(sizes.get(name, 0), int(ends[mine].max()))
            file_sums.add(name, starts[mine], ends[mine], owners[mine])
    file_sums.flush()
    for chrom, matrix in file_sums.sums.items():
        sums[chrom] += matrix
    return read


def _fasta_sizes(reference: Optional[Path]) -> Dict[str, int]:
    if not reference:
        return {}
    fai = Path(str(reference) + ".fai")
    if not fai.is_file():
        return {}
    sizes = {}
    for line in fai.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].isdigit():
            sizes[parts[0]] = int(parts[1])
    return sizes


def run_singlecell_genome_tracks(
    *,
    differential_checkpoint: Path,
    samples: Path,
    extraction: Path,
    rds: Path,
    group_by: Sequence[str],
    workdir: Path,
    prefix: str,
    genome: Optional[str] = None,
    gtf: Optional[Path] = None,
    reference: Optional[Path] = None,
    stratify_by: Optional[str] = None,
    rscript: str = "Rscript",
    zip_dir: Optional[Path] = None,
    overwrite: bool = False,
    echo: Callable[[str], None] = print,
    sources: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, object]:
    """snATAC: the viable peaks, each comparison's peaks and one profile per pseudobulk sample.

    ``sources`` ({fragment file: {barcode: extracted cell}}) is what fragment_sources.R reports; given, R is not
    run (tests, and a caller that has read it already).
    """
    differential_checkpoint, samples, extraction, rds = Path(differential_checkpoint), Path(samples), Path(extraction), Path(rds)
    for label, path in (("the differential's checkpoint", differential_checkpoint), ("the pseudobulk manifest", samples)):
        if not path.is_file():
            raise ParalomeError(104, f"{label} {path} is not there", fix="Run the stages before genome_tracks first")
    if not (extraction / "cell_metadata.tsv").is_file():
        raise ParalomeError(104, f"the single-cell extraction {extraction} is not there",
                            fix="Run singlecell_atac before genome_tracks")
    workdir = Path(workdir)
    out = tracks_zip_path(workdir, prefix, "snATAC", zip_dir)
    checkpoint_path = stage_checkpoint_path(workdir, STAGE)
    gtf_path, reference_path = _resolve_annotation(genome, gtf, reference)

    objects: List[Path]
    if sources is None:
        from paralome.singlecell.rds_input import member_prefix, rds_members

        objects = rds_members(rds) if rds.is_dir() else [rds]
        found: Dict[str, Dict[str, str]] = {}
        with tempfile.TemporaryDirectory(prefix="fragment_sources_", dir=str(workdir) if workdir.is_dir() else None) as tmp:
            for obj in objects:
                assay = None
                for manifest_dir in (extraction / "rds_members" / member_prefix(obj), extraction):
                    try:
                        assay = json.loads((manifest_dir / "extraction_manifest.json").read_text()).get("assay") or None
                        break
                    except (OSError, ValueError):
                        continue
                table = fragment_sources(obj, Path(tmp) / f"{safe_name(obj.stem)}.tsv", rscript=rscript, assay=assay)
                prefix_cells = (member_prefix(obj) + "_") if rds.is_dir() else ""
                for path, barcodes in read_fragment_sources(table, prefix_cells).items():
                    found.setdefault(path, {}).update(barcodes)
        sources = found
    else:
        objects = [rds]
    check_fragment_files(sources, objects)

    request = {
        "method": METHOD, "assay": "snATAC", "bin_size": BIN_SIZE, "group_by": list(group_by),
        "stratify_by": stratify_by,
        "differential": _fingerprint(differential_checkpoint), "samples": _fingerprint(samples),
        "cell_metadata": _fingerprint(extraction / "cell_metadata.tsv"),
        "fragments": {path: _fingerprint(Path(path)) for path in sorted(sources)},
        "gtf": _fingerprint(gtf_path) if gtf_path and gtf_path.is_file() else None,
    }
    request_id = _request_id(request)
    kept = _kept(checkpoint_path, out, request_id, overwrite)
    if kept is not None:
        echo(f"genome_tracks: the tracks in {(recorded_zip(kept, workdir) or out).name} answer this request; kept")
        return kept
    tools = require_tools()
    started = _begin(checkpoint_path, request, request_id)

    peaks, comparisons = read_differential(differential_checkpoint)
    regions = build_regions(peaks.values())
    membership, track_samples = pseudobulk_membership(
        extraction, samples, group_by, stratify_by,
        group_column=display_group(recorded_factors(differential_checkpoint))[0],
    )
    count = len(track_samples)
    sums = {chrom: np.zeros((count, len(r.bin_starts)), dtype=np.float64) for chrom, r in regions.items()}
    eligible = np.zeros(count, dtype=np.int64)
    observed: Dict[str, int] = {}
    for path, barcodes in sources.items():
        echo(f"genome_tracks: reading {path}")
        fragment_file_sums(path, barcodes, membership, regions, count, sums, eligible, observed)
    sizes = _fasta_sizes(reference_path)
    namer = _contig_namer(regions.keys())
    sizes = {namer(c): n for c, n in sizes.items() if namer(c)} if sizes else {}
    for chrom, r in regions.items():
        if chrom not in sizes:
            sizes[chrom] = max(int(r.ends.max()) if len(r.ends) else 0, observed.get(chrom, 0))
    for index, sample in enumerate(track_samples):
        sample.eligible_fragments = int(eligible[index])
        sample.factor = (1_000_000.0 / eligible[index]) if eligible[index] else 0.0
    empty = [s.sample_id for s in track_samples if not s.eligible_fragments]
    if empty:
        echo(f"genome_tracks: no fragments found for {', '.join(empty)}; their tracks are empty")
    return _finish(
        assay="snATAC", prefix=prefix, workdir=workdir, out=out, checkpoint_path=checkpoint_path, request=request,
        request_id=request_id, started=started, peaks=peaks, comparisons=comparisons, regions=regions, sizes=sizes,
        samples=track_samples, sums=sums, gtf=gtf_path, genome=genome, tools=tools, echo=echo,
        extra={"fragment_files": sorted(sources)},
    )
