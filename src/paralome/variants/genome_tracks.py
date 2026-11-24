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

"""The DNA genome_tracks stage: the reads behind every variant call, for OmicView's variant panel.

The individual reads that contributed to each variant call -- the reads FreeBayes used, in a zip of their own (the ATAC arms keep their coverage tracks). The whole BAMs are too big to
give the client; this stage slices them at the called sites::

    <omicview>/genome_tracks/<prefix>_DNA_reads.zip      (stored: OmicView reads each member in place)

    manifest.json                 what is in the zip (kind paralome.dna_reads); each sample's family, relationship,
                                  affected status and sex (from the deduplication manifest)
    reads/<sample>.bam (+ .bai)   the sample's reads over every site the VCF holds, as FreeBayes read them:
                                  primary, not duplicate, not QC-failed, MAPQ >= the run's --min-mapping-quality
                                  (taken from the VCF's own freebayes command line)
    reference/genome.fa.gz        the genome's sequence REFERENCE_FLANK bases either side of every site, N elsewhere,
     (+ .fai, .gzi)               bgzip-compressed and indexed, so the viewer shows each read's mismatches
    annotation/genes.bb           the genome's GTF as gene models, exons and UTRs (as the ATAC tracks), drawn above
                                  the reads

Nothing is called or filtered beyond what the caller itself read: every site of the VCF is sliced.

SELF-CHECKING as the ATAC tracks are: issued again, it keeps the zip while its inputs (each file's path, size and
modification time) and the read filter are what the checkpoint recorded.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import struct
import subprocess
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from paralome.atac.genome_tracks import (
    SCHEMA, STAGE, _begin, require_tools, write_chrom_sizes, write_genes, _fingerprint, _request_id, _resolve_annotation, bam_contig_sizes, build_zip,
    chrom_sort_key, now_iso, recorded_zip, safe_name, tracks_there, zip_members,
)
from paralome.checkpoints import atomic_write_text, stage_checkpoint_path
from paralome.errors import ParalomeError

METHOD = "paralome.genome_tracks.dna_reads.v2"
OPEN_WITH = "OmicView's Variants viewer, with the genotype output: each variant's panel shows its reads"
MANIFEST_KIND = "paralome.dna_reads"
MANIFEST_SCHEMA = 1
ASSAY = "DNA"
#: Reference sequence kept either side of each site: more than a read's length, so every read over the site is
#: compared with the genome along its whole length.
REFERENCE_FLANK = 1000
#: What FreeBayes does not read: unmapped, secondary, QC-failed, duplicate, supplementary.
EXCLUDED_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
# FreeBayes's own default, for a VCF whose command line names no cut (the pipeline passes 0).
DEFAULT_MIN_MAPPING_QUALITY = 1
LINE = 60


def reads_zip_path(workdir: Path, prefix: str, zip_dir: Optional[Path] = None) -> Path:
    """``<prefix>_DNA_reads.zip``: in ``zip_dir`` (the project's omicview/genome_tracks folder), else workdir/results."""
    return Path(zip_dir if zip_dir else Path(workdir) / "results") / f"{prefix}_{ASSAY}_reads.zip"


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_sites(vcf: Path) -> Tuple[List[Tuple[str, int, int]], Optional[int]]:
    """Every record's REF span (contig as the VCF names it, 0-based start, end) and the MAPQ cut on the
    VCF's freebayes command line (None when the header has none)."""
    sites: List[Tuple[str, int, int]] = []
    mapq: Optional[int] = None
    with _open_text(vcf) as handle:
        for line in handle:
            if line.startswith("##"):
                if mapq is None and "freebayes" in line:
                    found = re.search(r"(?:--min-mapping-quality|-m)[ =](\d+)", line)
                    if found:
                        mapq = int(found.group(1))
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t", 5)
            if len(parts) < 5:
                continue
            start = int(parts[1]) - 1
            sites.append((parts[0], start, start + max(1, len(parts[3]))))
    return sites, mapq


def merged(regions: Sequence[Tuple[str, int, int]], sizes: Mapping[str, int], flank: int = 0) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for chrom, start, end in sorted(regions, key=lambda r: (chrom_sort_key(r[0]), r[1], r[2])):
        if chrom not in sizes:
            continue
        lo, hi = max(0, start - flank), min(sizes[chrom], end + flank)
        if out and out[-1][0] == chrom and lo <= out[-1][2]:
            out[-1] = (chrom, out[-1][1], max(out[-1][2], hi))
        else:
            out.append((chrom, lo, hi))
    return out


# --------------------------------------------------------------------------------------------------- reference
class _Fasta:
    """Random access to a plain FASTA through its .fai."""

    def __init__(self, path: Path):
        self.path = Path(path)
        fai = Path(str(path) + ".fai")
        if not fai.is_file():
            subprocess.run(["samtools", "faidx", str(path)], check=True)
        self.index: Dict[str, Tuple[int, int, int, int]] = {}
        for line in fai.read_text().splitlines():
            name, length, offset, bases, width = line.split("\t")[:5]
            self.index[name] = (int(length), int(offset), int(bases), int(width))
        self.handle = open(path, "rb")

    def fetch(self, name: str, start: int, end: int) -> bytes:
        length, offset, bases, width = self.index[name]
        end = min(end, length)
        if start >= end:
            return b""
        first = offset + (start // bases) * width + start % bases
        last = offset + ((end - 1) // bases) * width + (end - 1) % bases
        self.handle.seek(first)
        return self.handle.read(last - first + 1).replace(b"\n", b"").replace(b"\r", b"")[: end - start].upper()


class _Bgzf:
    """A bgzip-compressed file written block by block, with its .gzi (the offsets igv.js reads it by)."""

    BLOCK = 0xFF00

    def __init__(self, path: Path):
        self.out: BinaryIO = open(path, "wb")
        self.buffer = bytearray()
        self.compressed = 0
        self.uncompressed = 0
        self.starts: List[Tuple[int, int]] = []

    def write(self, data: bytes) -> None:
        self.buffer += data
        while len(self.buffer) >= self.BLOCK:
            self._block(bytes(self.buffer[: self.BLOCK]))
            del self.buffer[: self.BLOCK]

    @staticmethod
    def pack(data: bytes) -> bytes:
        """One BGZF block holding ``data`` (at most BLOCK bytes)."""
        packer = zlib.compressobj(6, zlib.DEFLATED, -15)
        body = packer.compress(data) + packer.flush()
        size = len(body) + 25
        head = struct.pack("<4BI2BH2BHH", 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, size)
        tail = struct.pack("<2I", zlib.crc32(data) & 0xFFFFFFFF, len(data))
        return head + body + tail

    @staticmethod
    def finish(path: Path, gzi: Path, starts: List[Tuple[int, int]], compressed: int, uncompressed: int) -> None:
        """End the BGZF file at ``path`` and write its .gzi.

        The empty block that ends a BGZF file is indexed too: igv.js reads a block only up to the next one the
        .gzi names, so without it the last block of sequence would never be read."""
        starts = list(starts) + [(compressed, uncompressed)]
        with open(path, "ab") as handle:
            handle.write(bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000"))
        with open(gzi, "wb") as handle:
            handle.write(struct.pack("<Q", len(starts)))
            for c, u in starts:
                handle.write(struct.pack("<2Q", c, u))

    def _block(self, data: bytes) -> None:
        if self.compressed:
            self.starts.append((self.compressed, self.uncompressed))
        block = self.pack(data)
        self.out.write(block)
        self.compressed += len(block)
        self.uncompressed += len(data)

    def close(self, gzi: Path) -> None:
        if self.buffer:
            self._block(bytes(self.buffer))
            self.buffer.clear()
        self.out.close()
        self.finish(self.out.name, gzi, self.starts, self.compressed, self.uncompressed)


#: BGZF blocks compressed per piece of the parallel reference writer: 256 blocks, about 16 MB of text.
REFERENCE_PIECE_BLOCKS = 256


def _contig_layout(sizes: Mapping[str, int], keep: Sequence[Tuple[str, int, int]]) -> List[Dict[str, object]]:
    """Every contig the reference holds, in order: its header, length, kept windows and where its text starts.

    The text is ``>chrom\n`` then the bases LINE to a line, each line ending in a newline, so where every
    contig starts in the uncompressed stream is known before a byte is written."""
    by_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for chrom, start, end in keep:
        by_chrom.setdefault(chrom, []).append((start, end))
    layout: List[Dict[str, object]] = []
    offset = 0
    for chrom in sorted(by_chrom, key=chrom_sort_key):
        length = int(sizes[chrom])
        header = f">{chrom}\n".encode()
        text = len(header) + length + (length + LINE - 1) // LINE
        layout.append({"chrom": chrom, "header": header, "length": length, "windows": sorted(by_chrom[chrom]),
                       "start": offset, "sequence_start": offset + len(header), "end": offset + text})
        offset += text
    return layout


def _contig_text(source: "_Fasta", contig: Mapping[str, object], lo: int, hi: int) -> bytes:
    """Bytes ``lo:hi`` of one contig's text (offsets within that contig's text)."""
    header: bytes = contig["header"]  # type: ignore[assignment]
    length: int = contig["length"]  # type: ignore[assignment]
    out = bytearray()
    if lo < len(header):
        out += header[lo:min(hi, len(header))]
    lo, hi = max(0, lo - len(header)), hi - len(header)
    if hi <= lo:
        return bytes(out)
    # Body offsets to bases: every line is LINE bases and a newline (the last line may be shorter).
    first_line, last_line = lo // (LINE + 1), (hi - 1) // (LINE + 1)
    base_lo, base_hi = first_line * LINE, min(length, (last_line + 1) * LINE)
    bases = bytearray(b"N" * (base_hi - base_lo))
    for start, end in contig["windows"]:  # type: ignore[union-attr]
        s, e = max(start, base_lo), min(end, base_hi)
        if s < e:
            bases[s - base_lo:e - base_lo] = source.fetch(contig["chrom"], s, e)  # type: ignore[arg-type]
    body = bytearray()
    for line in range(first_line, last_line + 1):
        a = line * LINE - base_lo
        body += bases[a:a + min(LINE, length - line * LINE)]
        body += b"\n"
    skip = lo - first_line * (LINE + 1)
    out += body[skip:skip + (hi - lo)]
    return bytes(out)


def _compress_piece(fasta: str, layout: Sequence[Mapping[str, object]], lo: int, hi: int) -> Tuple[bytes, List[int]]:
    """Text ``lo:hi`` of the whole reference stream as BGZF blocks: (the blocks, each block's compressed size).

    ``lo`` is a multiple of the block size, so the pieces' blocks are the blocks one writer would make."""
    source = _Fasta(Path(fasta))
    text = bytearray()
    for contig in layout:
        start, end = int(contig["start"]), int(contig["end"])  # type: ignore[arg-type]
        if end <= lo or start >= hi:
            continue
        text += _contig_text(source, contig, max(lo, start) - start, min(hi, end) - start)
    blocks = bytearray()
    sizes: List[int] = []
    for at in range(0, len(text), _Bgzf.BLOCK):
        block = _Bgzf.pack(bytes(text[at:at + _Bgzf.BLOCK]))
        blocks += block
        sizes.append(len(block))
    return bytes(blocks), sizes


def write_masked_reference(fasta: Path, sizes: Mapping[str, int], keep: Sequence[Tuple[str, int, int]],
                           out: Path, threads: int = 1) -> Tuple[Path, Path, Path]:
    """The genome's sequence over ``keep`` (reference contig names), N elsewhere: ``out`` bgzipped, its .fai, .gzi.

    Only the contigs with a site are written, each at its full length, so a read sits at its own coordinates.
    The stream is cut into pieces of whole BGZF blocks, each made and compressed by its own process
    (``threads`` at once), and the pieces are joined in order: the same file one writer would make."""
    layout = _contig_layout(sizes, keep)
    total = int(layout[-1]["end"]) if layout else 0
    step = _Bgzf.BLOCK * REFERENCE_PIECE_BLOCKS
    pieces = [(at, min(total, at + step)) for at in range(0, total, step)]
    workers = max(1, min(int(threads or 1), len(pieces) or 1))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_compress_piece, [str(fasta)] * len(pieces), [layout] * len(pieces),
                                 [lo for lo, _hi in pieces], [hi for _lo, hi in pieces]))
    else:
        done = [_compress_piece(str(fasta), layout, lo, hi) for lo, hi in pieces]
    starts: List[Tuple[int, int]] = []
    compressed = uncompressed = 0
    with open(out, "wb") as handle:
        for blocks, block_sizes in done:
            handle.write(blocks)
            for size in block_sizes:
                if compressed:
                    starts.append((compressed, uncompressed))
                compressed += size
                uncompressed += min(_Bgzf.BLOCK, total - uncompressed)
    gzi = Path(str(out) + ".gzi")
    _Bgzf.finish(out, gzi, starts, compressed, uncompressed)
    fai = Path(str(out) + ".fai")
    fai.write_text("".join(f"{c['chrom']}\t{c['length']}\t{c['sequence_start']}\t{LINE}\t{LINE + 1}\n"
                           for c in layout))
    return out, fai, gzi


# ------------------------------------------------------------------------------------------------------- reads
def slice_reads(bam: Path, sites_bed: Path, out: Path, samtools: str, min_mapping_quality: int,
                threads: int = 1) -> int:
    """The sample's reads over the sites as FreeBayes read them, indexed; returns how many.

    ``threads`` is the sample's share of the job's CPUs: samtools decompresses and compresses on that many."""
    extra = ["-@", str(max(0, int(threads) - 1))] if int(threads or 1) > 1 else []
    subprocess.run([samtools, "view", *extra, "-b", "-M", "-L", str(sites_bed), "-q", str(min_mapping_quality),
                    "-F", str(EXCLUDED_FLAGS), "-o", str(out), str(bam)], check=True)
    subprocess.run([samtools, "index", *extra, str(out)], check=True)
    count = subprocess.run([samtools, "view", *extra, "-c", str(out)], check=True, capture_output=True, text=True)
    return int(count.stdout.strip() or 0)


def _manifest_rows(samples: Path) -> List[Dict[str, str]]:
    with Path(samples).open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if str(row.get("sample_id") or "").strip()]


def run_dna_genome_tracks(
    *,
    vcf: Path,
    samples: Path,
    workdir: Path,
    prefix: str,
    genome: Optional[str] = None,
    reference: Optional[Path] = None,
    gtf: Optional[Path] = None,
    threads: int = 1,
    zip_dir: Optional[Path] = None,
    overwrite: bool = False,
    echo: Callable[[str], None] = print,
) -> Dict[str, object]:
    from paralome.atac.quantification import _bam_contig_resolver
    from paralome.binaries import resolve_tool
    from paralome.version import __version__

    vcf, samples = Path(vcf), Path(samples)
    for label, path in (("the variant calling's VCF", vcf), ("the deduplication manifest", samples)):
        if not path.is_file():
            raise ParalomeError(104, f"{label} {path} is not there", fix="Run the stages before genome_tracks first")
    rows = _manifest_rows(samples)
    bams = {str(r["sample_id"]).strip(): str(r.get("bam") or r.get("output_bam") or "").strip() for r in rows}
    missing = [s for s, bam in bams.items() if not bam or not Path(bam).is_file()]
    if missing:
        raise ParalomeError(104, f"no deduplicated BAM for {', '.join(missing)} in {samples}",
                            fix="Run deduplication first, or check the bam column of the manifest")
    gtf_path, fasta = _resolve_annotation(genome, gtf, reference)
    if fasta is None or not Path(fasta).is_file():
        raise ParalomeError(104, f"the genome's FASTA ({fasta}) is not there",
                            fix="Give --genome (its reference), or --reference, so the reads can be shown against it")
    sites_raw, mapq = read_sites(vcf)
    mapq_source = "the VCF's freebayes command line"
    if mapq is None:
        mapq, mapq_source = DEFAULT_MIN_MAPPING_QUALITY, "FreeBayes's own default (the VCF names none)"

    workdir = Path(workdir)
    out = reads_zip_path(workdir, prefix, zip_dir)
    checkpoint_path = stage_checkpoint_path(workdir, STAGE)
    request = {
        "method": METHOD, "assay": ASSAY, "flank": REFERENCE_FLANK, "min_mapping_quality": mapq,
        "excluded_flags": EXCLUDED_FLAGS, "vcf": _fingerprint(vcf),
        "samples": {s: _fingerprint(Path(b)) for s, b in sorted(bams.items())},
        "manifest": _fingerprint(samples), "reference": _fingerprint(Path(fasta)),
        "gtf": _fingerprint(gtf_path) if gtf_path and Path(gtf_path).is_file() else None,
    }
    request_id = _request_id(request)
    kept = _kept(checkpoint_path, out, request_id, overwrite)
    if kept is not None:
        echo(f"genome_tracks: the reads in {(recorded_zip(kept, workdir) or out).name} answer this request; kept")
        return kept
    samtools = resolve_tool("samtools") or "samtools"
    started = _begin(checkpoint_path, request, request_id)

    sizes = bam_contig_sizes(Path(next(iter(bams.values()))), samtools)
    resolve = _bam_contig_resolver(sizes.keys())
    sites = [(resolve(c), s, e) for c, s, e in sites_raw]
    sites = [x for x in sites if x[0] is not None and x[0] in sizes]
    echo(f"genome_tracks: {len(sites)} sites in {vcf.name}; reads at MAPQ >= {mapq} ({mapq_source}), "
         "primary, not duplicate or QC-failed")
    by_sample = {str(r["sample_id"]).strip(): r for r in rows}
    fasta_names = set(_Fasta(Path(fasta)).index)
    missing_contigs = sorted({c for c, _s, _e in sites if c not in fasta_names}, key=chrom_sort_key)
    if missing_contigs:
        raise ParalomeError(104, f"the FASTA {fasta} has no {', '.join(missing_contigs[:5])}, which the BAMs use",
                            fix="Give the FASTA the BAMs were aligned to (--genome or --reference)")

    with tempfile.TemporaryDirectory(prefix="genome_tracks_", dir=str(workdir)) as tmp:
        scratch = Path(tmp)
        sites_bed = scratch / "sites.bed"
        sites_bed.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in merged(sites, sizes)))
        members: List[Tuple[Path, str]] = []
        names: Dict[str, str] = {}
        for sample in bams:
            name = safe_name(sample)
            while name in names.values():
                name += "_"
            names[sample] = name

        order = list(bams)
        workers = max(1, min(int(threads or 1), len(order)))
        # Every sample at once where the CPUs allow, each sample's samtools on its share of the rest.
        per_sample = max(1, int(threads or 1) // workers)

        def one(sample: str) -> int:
            return slice_reads(Path(bams[sample]), sites_bed, scratch / f"{names[sample]}.bam", samtools, mapq,
                               threads=per_sample)

        echo(f"genome_tracks: slicing {len(order)} BAM(s), {workers} at a time, {per_sample} CPU(s) each")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            counts = dict(zip(order, pool.map(one, order)))
        echo(f"genome_tracks: writing the reference around the sites on {max(1, int(threads or 1))} CPU(s)")
        ref_out = scratch / "genome.fa.gz"
        ref, fai, gzi = write_masked_reference(Path(fasta), sizes, merged(sites, sizes, REFERENCE_FLANK), ref_out,
                                               threads=int(threads or 1))
        members += [(ref, "reference/genome.fa.gz"), (fai, "reference/genome.fa.gz.fai"), (gzi, "reference/genome.fa.gz.gzi")]
        genes_record = None
        if gtf_path and Path(gtf_path).is_file():
            chrom_sizes = scratch / "chrom.sizes"
            write_chrom_sizes(chrom_sizes, sizes)
            genes = scratch / "genes.bb"
            count = write_genes(genes, Path(gtf_path), sizes, chrom_sizes, require_tools()["bedToBigBed"], scratch)
            if count:
                members.append((genes, "annotation/genes.bb"))
                genes_record = {"path": "annotation/genes.bb", "source": str(gtf_path), "transcripts": count}
        else:
            echo("genome_tracks: no GTF for this genome; the reads are shown without gene models")

        track_records = []
        for sample in order:
            row = by_sample.get(sample, {})
            bam = scratch / f"{names[sample]}.bam"
            members += [(bam, f"reads/{bam.name}"), (Path(str(bam) + ".bai"), f"reads/{bam.name}.bai")]
            track_records.append({
                "name": sample, "path": f"reads/{bam.name}", "index": f"reads/{bam.name}.bai", "reads": counts[sample],
                "family_id": (row.get("family_id") or "").strip() or None,
                "relationship": (row.get("relationship") or "").strip() or None,
                "affected": (row.get("affected") or "").strip() or None,
                "sex": (row.get("sex") or "").strip() or None,
            })
        manifest = {
            "kind": MANIFEST_KIND, "schema": MANIFEST_SCHEMA, "program": "Paralome", "program_version": __version__,
            "open_with": OPEN_WITH, "assay": ASSAY, "project": prefix, "genome": genome,
            "sites": len(sites), "source": str(vcf),
            "reads": (f"the reads FreeBayes read over every site: primary, not duplicate or QC-failed, MAPQ >= {mapq} "
                      f"({mapq_source})"),
            "min_mapping_quality": mapq,
            "reference": {"fasta": "reference/genome.fa.gz", "fai": "reference/genome.fa.gz.fai",
                          "gzi": "reference/genome.fa.gz.gzi",
                          "note": f"the genome's sequence {REFERENCE_FLANK} bases either side of every site, N elsewhere"},
            "genes": genes_record,
            "tracks": track_records,
        }
        build_zip(out, members, manifest)

    checkpoint = {
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "complete", "complete": True,
        "request": request, "request_id": request_id, "started": started, "finished": now_iso(),
        "program_version": __version__, "open_with": OPEN_WITH, "assay": ASSAY,
        "outputs": {"zip": str(out)}, "zip_bytes": out.stat().st_size, "members": zip_members(out),
        "sites": len(sites), "samples": len(track_records), "min_mapping_quality": mapq,
    }
    atomic_write_text(checkpoint_path, json.dumps(checkpoint, indent=2, sort_keys=True, default=str) + "\n")
    echo(f"genome_tracks: {out} ({out.stat().st_size / 1e6:.1f} MB); open it in {OPEN_WITH}")
    return checkpoint


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
    if Path(where).resolve() != Path(out).resolve() or not tracks_there(previous, where):
        return None
    return previous
