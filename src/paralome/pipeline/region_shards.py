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
import gzip
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from paralome.csvio import configure_csv_field_size_limit


CSV_FIELD_SIZE_LIMIT = configure_csv_field_size_limit()


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int
    name: str


def norm_chrom(chrom: Any) -> str:
    text = str(chrom or '').strip()
    if text.lower().startswith('chr'):
        return text[3:]
    return text


#: How a ``chrom:start-end`` line of a region map is read. The two
#: sources disagree, and a line does not say which it came from:
#:
#: ``one-based``  a list written by a person, samtools style: 1-based, both
#:                ends included. The default of every --regions that takes a
#:                region map for sharding (segregate, filter, report), and
#:                unchanged from.
#: ``freebayes``  a FreeBayes region file: what FreeBayes is given as
#:                --region, which its help defines as "0-base coordinates,
#:                end_position not included (same as BED format)". Since
#:                Every region file Paralome writes for FreeBayes is
#:                in it (variants/freebayes.py FREEBAYES_REGION_CONVENTION),
#:                and the DNA pipeline's own --regions is handed to FreeBayes
#:                as it is. The pipeline says so (--regions-convention
#:                freebayes) whenever it passes one of those on.
#:
#: BED lines (three or more columns) are 0-based, end-exclusive in both.
#:
#: Whichever is used, a region map only DEALS rows into shards: every row goes
#: to exactly one shard -- the first region (by start) that holds it, else the
#: round-robin fallback -- so a wrong reading moves a boundary row to a
#: neighbouring shard and can never lose or repeat it.
REGION_CONVENTION_ONE_BASED = "one-based"
REGION_CONVENTION_FREEBAYES = "freebayes"
REGION_CONVENTIONS = (REGION_CONVENTION_ONE_BASED, REGION_CONVENTION_FREEBAYES)


def check_region_convention(convention: Optional[str]) -> str:
    value = (convention or REGION_CONVENTION_ONE_BASED).strip().lower()
    if value not in REGION_CONVENTIONS:
        raise ValueError(
            f"Unknown region convention {convention!r}; use one of {', '.join(REGION_CONVENTIONS)}"
        )
    return value


def parse_region_token(text: str, idx: int, convention: Optional[str] = None) -> Optional[Region]:
    """One line of a region map as a 1-based, both-ends-included Region.

    ``convention`` says how a ``chrom:start-end`` line is read (see
    REGION_CONVENTIONS). The contig is everything before the LAST colon, as
    FreeBayes reads it, so a contig whose name holds colons stays whole.
    """
    convention = check_region_convention(convention)
    token = text.strip()
    if not token or token.startswith('#'):
        return None
    parts = token.replace('\t', ' ').split()
    if len(parts) >= 3:
        chrom = parts[0]
        try:
            start = int(float(parts[1])) + 1
            end = int(float(parts[2]))
        except ValueError:
            return None
        name = parts[3] if len(parts) > 3 else f'region_{idx:05d}'
        if end < start:
            return None
        return Region(norm_chrom(chrom), start, end, name)
    chrom, colon, rest = token.rpartition(':')
    if colon and chrom and '-' in rest:
        start_text, end_text = rest.split('-', 1)
        try:
            start = int(start_text.replace(',', ''))
            end = int(end_text.replace(',', ''))
        except ValueError:
            return None
        if convention == REGION_CONVENTION_FREEBAYES:
            start += 1  # 0-based, end-exclusive -> 1-based, both ends included
        if end < start:
            return None
        return Region(norm_chrom(chrom), start, end, f'region_{idx:05d}')
    return None


def read_regions(path: Union[str, Path, None], convention: Optional[str] = None) -> list[Region]:
    convention = check_region_convention(convention)
    if path in (None, ''):
        return []
    p = Path(path)
    regions: list[Region] = []
    # A byte-order mark at the top of the file is ignored.
    with p.open(encoding='utf-8-sig') as fh:
        for idx, line in enumerate(fh, start=1):
            region = parse_region_token(line, idx, convention)
            if region is not None:
                regions.append(region)
    return regions


class RegionIndex(dict):
    """Regions by contig, sorted, with a lookup by bisection.

    A dict of contig -> sorted [(start, end, region index)] that also answers
    "the first region, in that order, holding this position" in O(log n),
    rather than a walk over every region of the contig for every row.

    The answer is exactly the walk's. With the regions sorted by start, and
    ``reach[i]`` the largest end among the first i+1, the first region whose
    end reaches the position is the first i with reach[i] >= pos; it holds
    the position if its start is not past it, and if its start is past it so
    is every later start, and no region holds it.
    """

    def __init__(self, by_chrom: Optional[dict] = None) -> None:
        super().__init__()
        self._reach: dict[str, list[int]] = {}
        for chrom, entries in (by_chrom or {}).items():
            self[chrom] = sorted(entries)

    def __setitem__(self, chrom: str, entries: list) -> None:
        entries = sorted(entries)
        super().__setitem__(chrom, entries)
        reach: list[int] = []
        furthest: Optional[int] = None
        for _start, end, _idx in entries:
            furthest = end if furthest is None or end > furthest else furthest
            reach.append(furthest)
        self._reach[chrom] = reach

    def find(self, chrom: str, pos: int) -> Optional[int]:
        """The region index the walk would have returned, or None."""
        entries = self.get(chrom)
        if not entries:
            return None
        i = bisect_left(self._reach[chrom], pos)
        if i < len(entries) and entries[i][0] <= pos:
            return entries[i][2]
        return None


def region_indexes_by_chrom(regions: Sequence[Region]) -> RegionIndex:
    by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for idx, region in enumerate(regions):
        by_chrom.setdefault(norm_chrom(region.chrom), []).append((region.start, region.end, idx))
    return RegionIndex(by_chrom)


def assign_region_shard(
    *,
    chrom: Any,
    pos: Any,
    regions_by_chrom: dict[str, list[tuple[int, int, int]]],
    shard_count: int,
    fallback_index: int,
) -> int:
    shard_count = max(int(shard_count or 1), 1)
    chrom_key = norm_chrom(chrom)
    try:
        pos_i = int(float(str(pos)))
    except (TypeError, ValueError):
        return fallback_index % shard_count
    index = regions_by_chrom if isinstance(regions_by_chrom, RegionIndex) else RegionIndex(regions_by_chrom)
    region_idx = index.find(chrom_key, pos_i)
    if region_idx is not None:
        return region_idx % shard_count
    return fallback_index % shard_count


def _open_text(path: Path, mode: str = 'rt'):
    path = Path(path)
    if path.suffix == '.gz':
        return gzip.open(path, mode)
    return path.open(mode)


def split_tsv_by_regions(
    *,
    input_tsv: Union[str, Path],
    shard_dir: Union[str, Path],
    shards: int,
    regions: Sequence[Region] | None = None,
    prefix: str = 'shard',
) -> tuple[list[Path], list[str], list[int]]:
    input_tsv = Path(input_tsv)
    shard_dir = Path(shard_dir)
    shards = max(int(shards or 1), 1)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f'{prefix}_{idx:04d}.tsv' for idx in range(shards)]
    regions_by_chrom = region_indexes_by_chrom(regions or [])
    handles = []
    counts = [0 for _ in range(shards)]
    # Without a region map, rows go to the shards in CONTIGUOUS
    # runs, not dealt round-robin. The merge concatenates shard by shard, so
    # dealing put the output in the order 1000,1003,1001,1004,1002 for three
    # shards -- neither the input's order nor genomic. Contiguous runs merge
    # back in the input's order, as the serial path writes it.
    rows_per_shard = 1
    if not regions:
        with input_tsv.open(newline='') as count_fh:
            total_rows = max(sum(1 for _ in count_fh) - 1, 0)
        rows_per_shard = max(-(-total_rows // shards), 1)
    with input_tsv.open(newline='') as in_fh:
        reader = csv.DictReader(in_fh, delimiter='\t')
        if not reader.fieldnames:
            raise ValueError(f'TSV has no header: {input_tsv}')
        fieldnames = list(reader.fieldnames)
        try:
            writers = []
            for path in shard_paths:
                fh = path.open('w', newline='')
                handles.append(fh)
                writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t', lineterminator='\n', extrasaction='ignore')
                writer.writeheader()
                writers.append(writer)
            for idx, row in enumerate(reader):
                if regions:
                    shard_idx = assign_region_shard(
                        chrom=row.get('chrom') or row.get('CHROM') or row.get('#CHROM'),
                        pos=row.get('pos') or row.get('POS'),
                        regions_by_chrom=regions_by_chrom,
                        shard_count=shards,
                        fallback_index=idx,
                    )
                else:
                    shard_idx = min(idx // rows_per_shard, shards - 1)
                writers[shard_idx].writerow(row)
                counts[shard_idx] += 1
        finally:
            for fh in handles:
                fh.close()
    return shard_paths, fieldnames, counts


def split_vcf_by_regions(
    *,
    input_vcf: Union[str, Path],
    shard_dir: Union[str, Path],
    shards: int,
    regions: Sequence[Region] | None = None,
    prefix: str = 'shard',
) -> tuple[list[Path], list[int]]:
    input_vcf = Path(input_vcf)
    shard_dir = Path(shard_dir)
    shards = max(int(shards or 1), 1)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f'{prefix}_{idx:04d}.vcf' for idx in range(shards)]
    regions_by_chrom = region_indexes_by_chrom(regions or [])
    handles = [path.open('w') for path in shard_paths]
    counts = [0 for _ in range(shards)]
    header_lines: list[str] = []
    try:
        with _open_text(input_vcf, 'rt') as fh:
            variant_idx = 0
            for line in fh:
                if line.startswith('#'):
                    header_lines.append(line)
                    for out in handles:
                        out.write(line)
                    continue
                if not line.strip():
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                chrom = parts[0]
                pos = parts[1]
                if regions:
                    shard_idx = assign_region_shard(
                        chrom=chrom,
                        pos=pos,
                        regions_by_chrom=regions_by_chrom,
                        shard_count=shards,
                        fallback_index=variant_idx,
                    )
                else:
                    shard_idx = variant_idx % shards
                handles[shard_idx].write(line)
                counts[shard_idx] += 1
                variant_idx += 1
    finally:
        for fh in handles:
            fh.close()
    return shard_paths, counts


def append_tsv_without_header(src: Union[str, Path], dst_handle: Any) -> int:
    src = Path(src)
    if not src.exists():
        return 0
    count = 0
    with src.open(newline='') as fh:
        fh.readline()
        for line in fh:
            dst_handle.write(line)
            count += 1
    return count


def merge_tsv_fragments(
    *,
    fragment_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    fieldnames: Sequence[str],
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open('w', newline='') as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=list(fieldnames), delimiter='\t', lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        for path in fragment_paths:
            total += append_tsv_without_header(Path(path), out_fh)
    return total
