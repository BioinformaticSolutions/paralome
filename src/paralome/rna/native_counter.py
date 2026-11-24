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

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from paralome.rna.gene_model import GeneModel

STRANDEDNESS_LABELS = {0: "unstranded", 1: "first-strand", 2: "second-strand"}
STRANDEDNESS_ASSIGNMENT_RULES = {
    0: "ignore annotated transcript strand",
    1: "fragment orientation opposite annotated transcript strand",
    2: "fragment orientation matches annotated transcript strand",
}


@dataclass(frozen=True)
class _ExonInterval:
    start: int
    end: int
    gene_id: str
    strand: str


@dataclass(frozen=True)
class _ChromIndex:
    """Bisect-indexed exon intervals for one chromosome.

    Mirrors the interval-index pattern already used for ATAC TSS/peak lookups
    (``paralome.atac.quantification``): entries are sorted by start position
    with a running prefix-max of interval ends, so overlap queries walk
    backward only as far as intervals could possibly still overlap.
    """

    lefts: Tuple[int, ...]
    intervals: Tuple[_ExonInterval, ...]
    prefix_max_end: Tuple[int, ...]

    @classmethod
    def build(cls, intervals: Sequence[_ExonInterval]) -> "_ChromIndex":
        ordered = sorted(intervals, key=lambda iv: (iv.start, iv.end, iv.gene_id))
        prefix: List[int] = []
        running_max = -1
        for interval in ordered:
            running_max = max(running_max, interval.end)
            prefix.append(running_max)
        return cls(
            tuple(interval.start for interval in ordered),
            tuple(ordered),
            tuple(prefix),
        )

    def overlapping(self, start: int, end: int) -> List[_ExonInterval]:
        high = bisect_left(self.lefts, end)
        cursor = high - 1
        matches: List[_ExonInterval] = []
        while cursor >= 0 and self.prefix_max_end[cursor] > start:
            interval = self.intervals[cursor]
            if interval.end > start:
                matches.append(interval)
            cursor -= 1
        return matches


def _build_chrom_index(genes: Sequence[GeneModel]) -> Dict[str, _ChromIndex]:
    by_chrom: Dict[str, List[_ExonInterval]] = {}
    for gene in genes:
        for start, end in gene.exons:
            by_chrom.setdefault(gene.chrom, []).append(
                _ExonInterval(start, end, gene.gene_id, gene.strand)
            )
    return {chrom: _ChromIndex.build(intervals) for chrom, intervals in by_chrom.items()}


def _orientation_matches(strandedness: int, fragment_strand: str, gene_strand: str) -> bool:
    if strandedness == 1:
        return fragment_strand != gene_strand
    if strandedness == 2:
        return fragment_strand == gene_strand
    raise ValueError(f"Unsupported strandedness: {strandedness}")


def _assign_blocks(
    index_by_chrom: Mapping[str, _ChromIndex],
    chrom: Optional[str],
    blocks: Sequence[Tuple[int, int]],
    strandedness: int,
    fragment_strand: str,
) -> Set[str]:
    if not chrom:
        return set()
    chrom_index = index_by_chrom.get(chrom)
    if chrom_index is None:
        return set()
    hits: Set[str] = set()
    for start, end in blocks:
        if end <= start:
            continue
        for interval in chrom_index.overlapping(start, end):
            if strandedness != 0 and not _orientation_matches(strandedness, fragment_strand, interval.strand):
                continue
            hits.add(interval.gene_id)
    return hits


def _count_single_end(bam, index_by_chrom: Mapping[str, _ChromIndex], counts: Dict[str, int], strandedness: int) -> Tuple[int, int]:
    eligible = 0
    assigned = 0
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        blocks = read.get_blocks()
        if not blocks:
            continue
        eligible += 1
        fragment_strand = "-" if read.is_reverse else "+"
        hit_genes = _assign_blocks(index_by_chrom, read.reference_name, blocks, strandedness, fragment_strand)
        if len(hit_genes) == 1:
            counts[next(iter(hit_genes))] += 1
            assigned += 1
    return eligible, assigned


def _count_paired_end(
    bam,
    index_by_chrom: Mapping[str, _ChromIndex],
    counts: Dict[str, int],
    strandedness: int,
    *,
    require_both_ends_mapped: bool,
    exclude_chimeric_fragments: bool,
) -> Tuple[int, int]:
    eligible = 0
    assigned = 0
    pending: Dict[str, object] = {}

    def process(mate1, mate2) -> None:
        nonlocal eligible, assigned
        reads = [read for read in (mate1, mate2) if read is not None]
        if not reads:
            return
        if len(reads) == 2:
            if exclude_chimeric_fragments and (
                not reads[0].is_proper_pair
                or not reads[1].is_proper_pair
                or reads[0].reference_name != reads[1].reference_name
            ):
                return
        blocks: List[Tuple[int, int]] = []
        for read in reads:
            blocks.extend(read.get_blocks())
        if not blocks:
            return
        eligible += 1
        chrom = reads[0].reference_name
        read1 = next((read for read in reads if read.is_read1), None)
        reference = read1 if read1 is not None else reads[0]
        fragment_strand = "-" if reference.is_reverse else "+"
        if read1 is None and reference.is_read2:
            # Only read2 is available for this fragment; read2 sequences the
            # opposite strand from read1, so invert the inferred orientation
            # to keep it consistent with a read1-referenced strand call.
            fragment_strand = "+" if fragment_strand == "-" else "-"
        if all(read.reference_name == chrom for read in reads):
            hit_genes = _assign_blocks(index_by_chrom, chrom, blocks, strandedness, fragment_strand)
        else:
            # Mates on different chromosomes (reachable only with --allow-chimeric-fragments). Each mate is
            # looked up where it aligned; a fragment touching two genes is ambiguous, as it is anywhere else.
            hit_genes = set()
            for read in reads:
                hit_genes |= _assign_blocks(
                    index_by_chrom, read.reference_name, read.get_blocks(), strandedness, fragment_strand,
                )
        if len(hit_genes) == 1:
            counts[next(iter(hit_genes))] += 1
            assigned += 1

    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if not read.is_paired:
            # Malformed/mixed input for a nominally paired-end BAM: treat the
            # lone read as its own fragment rather than silently dropping it.
            process(read, None)
            continue
        name = read.query_name
        mate = pending.pop(name, None)
        if mate is None:
            pending[name] = read
            continue
        if read.is_read1:
            process(read, mate)
        else:
            process(mate, read)

    for read in pending.values():
        if require_both_ends_mapped:
            continue
        if read.is_read1:
            process(read, None)
        else:
            process(None, read)

    return eligible, assigned


def count_gene_fragments(
    bam_path: Union[str, Path],
    genes: Sequence[GeneModel],
    *,
    paired_end: bool,
    strandedness: int,
    require_both_ends_mapped: bool = True,
    exclude_chimeric_fragments: bool = True,
) -> Tuple[Dict[str, int], int, int]:
    """Assign primary mapped reads/fragments in ``bam_path`` to genes.

    A fragment (a single-end read, or the combined footprint of a mapped
    read pair) is assigned to exactly one gene when its aligned blocks
    overlap that gene's union-exon model and no other gene's, honoring the
    requested strandedness rule. Fragments overlapping zero or more than one
    gene are counted as eligible but unassigned, matching standard
    meta-feature counting semantics (e.g. featureCounts without ``-O``).

    Returns ``(counts, eligible_fragments, assigned_fragments)``.
    """
    if strandedness not in {0, 1, 2}:
        raise ValueError("strandedness must be 0, 1, or 2")

    import pysam

    index_by_chrom = _build_chrom_index(genes)
    counts: Dict[str, int] = {gene.gene_id: 0 for gene in genes}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        if paired_end:
            eligible, assigned = _count_paired_end(
                bam,
                index_by_chrom,
                counts,
                strandedness,
                require_both_ends_mapped=require_both_ends_mapped,
                exclude_chimeric_fragments=exclude_chimeric_fragments,
            )
        else:
            eligible, assigned = _count_single_end(bam, index_by_chrom, counts, strandedness)
    return counts, eligible, assigned
