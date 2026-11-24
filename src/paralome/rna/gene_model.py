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

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

GTF_GENE_FEATURE_ALIASES = ("exon",)


# A key is never allowed to swallow the ";" that ended the pair before it, so `transcript_id
# "T";gene_id "G"` (no space after the semicolon) yields gene_id. A quoted value is taken whole, so it
# may hold a ";".
_GTF_ATTRIBUTE = re.compile(r'([^\s;"]+)(?:\s*"([^"]*)"|\s+([^\s;"]+))')


def _gtf_attributes(text: str) -> Dict[str, str]:
    """Parse GTF attribute text (``key "value";`` pairs) into a mapping."""
    attrs: Dict[str, str] = {}
    for key, quoted, bare in _GTF_ATTRIBUTE.findall(text):
        attrs[key] = quoted or bare
    return attrs


def open_gtf_text(gtf_path: Union[str, Path]):
    """Open a GTF for reading as text, gzip-compressed or not.

    A ``.gtf.gz`` straight from Ensembl was read as text and failed
    with "'utf-8' codec can't decode byte 0x8b". Compression is recognised by
    the file's first two bytes, not its name.
    """
    path = Path(gtf_path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open()


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent half-open ``[start, end)`` intervals."""
    ordered = sorted(intervals)
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class GeneModel:
    """A gene's union-exon model, used for native fragment assignment.

    ``exons`` is the merged (non-overlapping, sorted) set of exonic
    ``[start, end)`` intervals for the gene, in 0-based half-open coordinates.
    ``length_bp`` is the sum of merged exon widths -- the same "union exon
    length" convention used by featureCounts, so FPKM values calculated from
    this model are comparable to prior featureCounts-derived runs.
    """

    gene_id: str
    chrom: str
    strand: str
    exons: Tuple[Tuple[int, int], ...]
    gene_name: str = ""

    @property
    def start(self) -> int:
        return self.exons[0][0]

    @property
    def end(self) -> int:
        return self.exons[-1][1]

    @property
    def length_bp(self) -> int:
        return sum(end - start for start, end in self.exons)


def parse_gtf_genes(gtf_path: Union[str, Path]) -> List[GeneModel]:
    """Build one union-exon :class:`GeneModel` per ``gene_id`` in a GTF.

    Only ``exon`` feature rows are used to build gene models, matching the
    standard ``featureCounts -t exon -g gene_id`` convention this replaces.
    A gene with exons on both strands of a chromosome is left out, and a gene
    on several chromosomes keeps only one; :func:`parse_gtf_left_out`
    also says which those were.
    """
    return parse_gtf_gene_models(gtf_path)[0]


def parse_gtf_gene_models(
    gtf_path: Union[str, Path],
) -> Tuple[List[GeneModel], Dict[str, str]]:
    """The gene models of a GTF, and the genes that could not be given one.

    Real annotations hold genes with exons on both strands, or on two
    chromosomes (trans-spliced genes; pseudoautosomal genes sharing an id
    across X and Y), and a union-exon model has no single strand or length for
    them. They are left out of quantification and returned as
    ``{gene_id: reason}`` so the caller can say so; the GTF is not refused.

    Only a gene left out WHOLE is returned here; :func:`parse_gtf_left_out`
    also returns the genes that were kept without their exons elsewhere.
    """
    genes, skipped, _partial = parse_gtf_left_out(gtf_path)
    return genes, skipped


#: A primary chromosome by its name, once a leading "chr" is taken off: a
#: short name of letters and digits (1..99, X, Y, M, MT, 2L, 3R, Z, W, Mito,
#: the roman numerals of yeast and worm) or a RefSeq NC_ accession.
_SHORT_CHROMOSOME_NAME = re.compile(r"^[A-Za-z0-9]{1,5}$")
_ROMAN_NUMERAL = re.compile(r"^(?=[IVX])X{0,3}(IX|IV|V?I{0,3})$")
_REFSEQ_CHROMOSOME = re.compile(r"^NC_\d+(\.\d+)?$")


def _is_primary_chromosome(chrom: str) -> bool:
    """Whether a sequence name is a chromosome and not a scaffold or patch."""
    if _REFSEQ_CHROMOSOME.match(chrom):
        return True
    name = chrom[3:] if chrom.lower().startswith("chr") else chrom
    if "_" in name or "." in name or name.lower().startswith("un"):
        return False
    return bool(_SHORT_CHROMOSOME_NAME.match(name) or _ROMAN_NUMERAL.match(name))


def _kept_chromosome(exons_by_chromosome: Dict[str, List[Tuple[int, int]]]) -> str:
    """The one chromosome a gene annotated on several is kept on.

    Chosen in this order: a primary chromosome before anything else, not Y
    while there is another (so X before Y), the most exon bases, and then the
    name, so that the order of the GTF never decides.

    "Primary" is decided by _is_primary_chromosome, not by "a name without an
    underscore" (which holds for UCSC names only: Ensembl and GENCODE
    scaffolds such as KI270762.1 carry none, and every RefSeq name carries
    one). Which RefSeq accession is a Y cannot be known from its name, and is
    not guessed.
    ``exons_by_chromosome`` holds merged exons.
    """
    def rank(item) -> Tuple[int, int, int, str]:
        chrom, exons = item
        name = chrom[3:] if chrom.lower().startswith("chr") else chrom
        return (
            0 if _is_primary_chromosome(chrom) else 1,
            1 if name.lower() == "y" else 0,
            -sum(end - start for start, end in exons),
            chrom,
        )

    return min(exons_by_chromosome.items(), key=rank)[0]


def parse_gtf_left_out(
    gtf_path: Union[str, Path],
) -> Tuple[List[GeneModel], Dict[str, str], Dict[str, str]]:
    """Gene models, genes left out whole, and genes kept at one of their places.

    In a GTF keyed by gene symbol (UCSC/RefSeq style) one gene_id is
    on chrX and chrY, or on a chromosome and its alternate contigs. Leaving
    such a gene out whole dropped the primary copy with the others. It is kept
    in ONE place and only its exons on other chromosomes are left out: the
    third value, ``{gene_id: reason}``. A gene on both strands of the
    chromosome it is kept on still has no single model and is left out whole
    (the second).

    The place does not depend on the order of the GTF's lines --
    see :func:`_kept_chromosome`.
    """
    path = Path(gtf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GTF not found: {path}")

    # Per gene, per chromosome in the order the GTF first names them: the
    # exons, the strands seen and the gene_name given there.
    exons_at: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    strands_at: Dict[str, Dict[str, List[str]]] = {}
    gene_name_at: Dict[str, Dict[str, str]] = {}

    with open_gtf_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, _source, feature, start_text, end_text, _score, strand, _frame, attr_text = fields
            if feature not in GTF_GENE_FEATURE_ALIASES:
                continue
            if strand not in {"+", "-"}:
                continue
            attrs = _gtf_attributes(attr_text)
            gene_id = attrs.get("gene_id", "")
            if not gene_id:
                continue
            try:
                start = int(start_text) - 1  # GTF is 1-based closed; convert to 0-based half-open.
                end = int(end_text)
            except ValueError:
                continue
            if end <= start:
                continue

            exons_at.setdefault(gene_id, {}).setdefault(chrom, []).append((start, end))
            strands = strands_at.setdefault(gene_id, {}).setdefault(chrom, [])
            if strand not in strands:
                strands.append(strand)
            # Carry the GTF's gene_name (gene symbol / "common
            # name") through, same attribute and fallback-to-gene_id pattern
            # already used independently by paralome.atac.quantification and
            # paralome.variants.freebayes. The RNA-seq reports key each gene
            # by both its Ensembl ID and this human-readable symbol.
            gene_name = attrs.get("gene_name", "").strip()
            if gene_name:
                gene_name_at.setdefault(gene_id, {})[chrom] = gene_name

    skipped: Dict[str, str] = {}
    partial: Dict[str, str] = {}
    genes: List[GeneModel] = []
    for gene_id, by_chrom in exons_at.items():
        merged = {chrom: _merge_intervals(exons) for chrom, exons in by_chrom.items()}
        chrom = _kept_chromosome(merged)
        if len(strands_at[gene_id][chrom]) > 1:
            skipped[gene_id] = f"exons on both strands of {chrom}"
            continue
        if len(merged) > 1:
            partial[gene_id] = (
                "exons on " + ", ".join(other for other in merged if other != chrom) + " left out"
            )
        genes.append(
            GeneModel(
                gene_id=gene_id,
                chrom=chrom,
                strand=strands_at[gene_id][chrom][0],
                exons=tuple(merged[chrom]),
                gene_name=gene_name_at.get(gene_id, {}).get(chrom, gene_id),
            )
        )
    if not genes:
        raise ValueError(
            f"GTF contains no usable exon records with gene_id: {path}"
            + (f" ({len(skipped)} gene(s) are on both strands of a chromosome)" if skipped else "")
        )

    genes.sort(key=lambda gene: (gene.chrom, gene.start, gene.gene_id))
    return genes, skipped, partial
