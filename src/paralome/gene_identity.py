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

"""Resolve each feature to an Ensembl ID and a common name, once, from the GTF.

Each feature gets its Ensembl ID in ``gene_id`` and its common name in
``common_id``, taken from the GTF of the configured genome.

WHERE THIS RUNS, AND WHY THERE. At EXTRACTION, once per dataset -- not at report
time, once per stratum per contrast.

The shared ``data/`` folder exists precisely so one rds is read once and serves
every comparison. Resolving identifiers at report time would re-parse the GTF
once per stratum per contrast, and worse, two contrasts run with
different ``--genome`` values would silently carry different identifiers for the
same gene. Extraction is also where the gene universe is fixed: ``rna_genes.tsv``
IS the gene list, so the identifier mapping belongs beside it. The R helper
writes ``rna_genes.tsv`` with ``gene_id`` and ``common_id`` columns, and the
aggregation reads both into ``GeneFeature`` and propagates them.

THE MATCH RATE IS A GENOME CHECK. A Seurat object built against GENCODE v44
and mapped against a mouse GTF will not fail -- it will quietly map almost
nothing. So the rate is reported and a low rate is a loud warning, because
"3% of your genes matched" is the only signal that the wrong genome was
passed.

WHAT IS NOT DONE HERE. No gene is dropped and no symbol is guessed at. A feature
whose symbol maps to several ``gene_id`` values is ambiguous and gets no Ensembl
ID rather than an arbitrary one; a feature absent from the GTF gets none either.
Both keep their original name and are counted in the report: the row is
kept, ``gene_id`` is ``.``, and the match rate is reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from paralome.report_schema import MISSING_VALUE

#: An Ensembl gene accession, with or without a version suffix, for any species
#: (ENSG human, ENSMUSG mouse, ENSDARG zebrafish...). Used only to decide which
#: way round a feature list already is -- never to validate one.
_ENSEMBL_GENE = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$")

#: Below this fraction matched, the genome is almost certainly wrong rather than
#: the object unusual. Not an error: a legitimately custom reference exists, and
#: refusing to run would be worse than saying so clearly.
LOW_MATCH_WARNING_FRACTION = 0.50


def looks_like_ensembl(value: str) -> bool:
    """Whether a feature name is an Ensembl gene accession."""
    return bool(_ENSEMBL_GENE.match(str(value).strip()))


def strip_version(value: str) -> str:
    """``ENSG00000141510.17`` -> ``ENSG00000141510``.

    GTFs and Seurat objects disagree about version suffixes more often than they
    agree, and a version mismatch is not an identity mismatch.
    """
    text = str(value).strip()
    return text.split(".", 1)[0] if _ENSEMBL_GENE.match(text) else text


@dataclass(frozen=True)
class GeneIdentityReport:
    """What the mapping achieved, for the checkpoint and for the user."""

    orientation: str                      # "symbol" | "ensembl" | "mixed" | "unknown"
    total: int = 0
    mapped: int = 0
    ambiguous: List[str] = field(default_factory=list)
    absent: List[str] = field(default_factory=list)
    gtf: str = ""
    # Names the GTF holds but the gene models leave out (exons on
    # both strands, or no exon rows). They also get gene_id ".".
    left_out: List[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return (self.mapped / self.total) if self.total else 0.0

    def summary_lines(self) -> List[str]:
        pct = f"{self.fraction:.1%}"
        lines = [
            f"Gene identity: matched {self.mapped:,} / {self.total:,} features ({pct}) "
            f"against {self.gtf or 'the configured GTF'}",
            f"  feature names look like: {self.orientation}",
        ]
        if self.ambiguous:
            lines.append(
                f"  ambiguous (name maps to >1 gene): {len(self.ambiguous):,} "
                f"-> {MISSING_VALUE}   e.g. {', '.join(self.ambiguous[:5])}"
            )
        if self.absent:
            lines.append(
                f"  absent from the GTF            : {len(self.absent):,} "
                f"-> {MISSING_VALUE}   e.g. {', '.join(self.absent[:5])}"
            )
        if self.left_out:
            lines.append(
                f"  in the GTF, no gene model      : {len(self.left_out):,} "
                f"-> {MISSING_VALUE}   e.g. {', '.join(self.left_out[:5])}"
            )
        if self.total and self.fraction < LOW_MATCH_WARNING_FRACTION:
            lines.append(
                f"  WARNING: under {LOW_MATCH_WARNING_FRACTION:.0%} matched. That is almost "
                f"always the wrong --genome for this object rather than an unusual "
                f"reference. Check the species and annotation release before trusting "
                f"any downstream gene identifier."
            )
        return lines

    def to_dict(self) -> Dict[str, object]:
        return {
            "orientation": self.orientation,
            "total": self.total,
            "mapped": self.mapped,
            "fraction": round(self.fraction, 6),
            "ambiguous": len(self.ambiguous),
            "absent": len(self.absent),
            "left_out": len(self.left_out),
            "gtf": self.gtf,
            # Bounded samples: a checkpoint recording 20,000 unmatched names
            # helps nobody and bloats every project directory.
            "ambiguous_examples": self.ambiguous[:25],
            "absent_examples": self.absent[:25],
            "left_out_examples": self.left_out[:25],
        }


def _gtf_named_anywhere(gtf_path: Union[str, Path]) -> Tuple[set, set]:
    """Every ``gene_id`` (version stripped) and ``gene_name`` on any GTF row.

    Used only to tell "absent from the GTF" from "in the GTF but
    left out of the gene models"; the identifiers resolved are unchanged.
    """
    from paralome.rna.gene_model import _gtf_attributes, open_gtf_text

    ids: set = set()
    names: set = set()
    with open_gtf_text(gtf_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            attrs = _gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id", "").strip()
            if gene_id:
                ids.add(strip_version(gene_id))
            gene_name = attrs.get("gene_name", "").strip()
            if gene_name:
                names.add(gene_name)
    return ids, names


def on_an_alternate_locus(contig: str) -> bool:
    """A contig that is an alternate haplotype or a patch, not the primary assembly.

    Ensembl names them CHR_* (CHR_HSCHR6_MHC_COX_CTG1, CHR_HG2266_PATCH); UCSC and GENCODE *_alt and *_fix
    (chr6_GL000250v2_alt). The unlocalised and unplaced scaffolds (KI270..., GL000..., chrUn_...) are primary
    assembly and are not counted here.
    """
    name = str(contig or "").strip()
    upper = name.upper()
    return upper.startswith(("CHR_", "HSCHR")) or upper.endswith(("_ALT", "_FIX")) or "_ALT_" in upper


def prefer_the_primary_assembly(genes: Sequence[str], contig_of: Mapping[str, str]) -> List[str]:
    """Of several genes one name or accession points to, the one on the primary assembly when every other is on an
    alternate haplotype or patch; otherwise all of them, still ambiguous.

    For example HLA-DRA, AGER and MOG each have one gene on chromosome 6 of GRCh38 and six to eight more on the
    CHR_HSCHR6_MHC_* haplotypes -- the same genes on other versions of the MHC, not other genes.
    """
    distinct = list(dict.fromkeys(genes))
    if len(distinct) < 2:
        return distinct
    primary = [g for g in distinct if not on_an_alternate_locus(contig_of.get(g, ""))]
    return primary if len(primary) == 1 else distinct


def gtf_gene_index(gtf_path: Union[str, Path]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """``(symbol -> gene_id, gene_id -> symbol, gene_id -> contig)``; ``build_gtf_index`` plus each gene's contig."""
    from paralome.rna.gene_model import parse_gtf_genes

    claims: Dict[str, List[str]] = {}
    by_id: Dict[str, str] = {}
    contig_of: Dict[str, str] = {}

    for gene in parse_gtf_genes(gtf_path):
        gene_id = strip_version(gene.gene_id)
        symbol = (gene.gene_name or "").strip()
        if gene_id:
            by_id[gene_id] = symbol
            contig_of[gene_id] = str(gene.chrom)
        if symbol and gene_id:
            claims.setdefault(symbol, [])
            if gene_id not in claims[symbol]:
                claims[symbol].append(gene_id)

    by_symbol: Dict[str, str] = {}
    for symbol, genes in claims.items():
        chosen = prefer_the_primary_assembly(genes, contig_of)
        by_symbol[symbol] = chosen[0] if len(chosen) == 1 else ""
    return by_symbol, by_id, contig_of


def build_gtf_index(gtf_path: Union[str, Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(symbol -> gene_id, gene_id -> symbol)`` from a GTF.

    A symbol claimed by more than one ``gene_id`` maps to the empty string
    rather than to whichever line was read last. GENCODE genuinely contains
    duplicated ``gene_name`` values -- PAR_Y copies, and a handful of real
    collisions -- so "last one wins" would assign a defensible-looking but
    arbitrary accession, which is worse than admitting the ambiguity.

    Except when every other claim is on an alternate haplotype or a patch
    (``prefer_the_primary_assembly``): the symbol is then the primary
    assembly's gene. An Ensembl GRCh38 GTF carries the MHC's alternate
    haplotypes, so HLA-DRA, AGER, MOG and the rest of the MHC would otherwise
    be ambiguous in every pipeline.
    """
    by_symbol, by_id, _contigs = gtf_gene_index(gtf_path)
    return by_symbol, by_id


#: The columns of Ensembl's UniProt cross-reference table this reads
#: (``<Species>.<assembly>.<release>.uniprot.tsv.gz``, ftp.ensembl.org/pub/release-<n>/tsv/<species>/).
UNIPROT_TABLE_COLUMNS = ("gene_stable_id", "xref")


def read_uniprot_table(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Ensembl's UniProt cross-references: ``{accession: [gene_id, ...]}``.

    The genome's own table (registered as ``uniprot_table``, from the same Ensembl release as its GTF): Olink
    assays named for a protein form (Abeta40, pTau217) are found by the UniProt accession Olink gives them. An
    accession's isoform suffix (P05067-2) is its accession."""
    import csv
    import gzip

    from paralome.errors import ParalomeError

    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    found: Dict[str, List[str]] = {}
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in UNIPROT_TABLE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ParalomeError(                                                         # PLM-0245
                245, f"{path} is not Ensembl's UniProt table: it has no {', '.join(missing)} column "
                     f"(its columns: {', '.join(reader.fieldnames or []) or 'none'})",
            )
        for row in reader:
            accession = (row.get("xref") or "").strip().split("-")[0]
            gene = strip_version((row.get("gene_stable_id") or "").strip())
            if accession and gene:
                genes = found.setdefault(accession, [])
                if gene not in genes:
                    genes.append(gene)
    return found


def resolve_gene_identities(
    feature_names: Sequence[str],
    gtf_path: Union[str, Path],
) -> Tuple[List[Dict[str, str]], GeneIdentityReport]:
    """Map an extraction's feature names to ``{gene_id, common_id}`` rows.

    Handles both orientations. A Seurat object may carry Ensembl accessions as
    rownames or gene symbols. Which way round it is, is
    detected from the names themselves rather than assumed, so the same code
    serves an object built either way -- and so a mixed or unrecognisable list
    is reported as such instead of being silently mapped at 0%.

    Order is preserved exactly: this is the extraction's feature order, and the
    counts matrix is indexed by position against it.
    """
    names = [str(name).strip() for name in feature_names]
    ensembl_like = sum(1 for name in names if looks_like_ensembl(name))
    # The orientation is REPORTED, and each name is resolved for what it is: a list that is half
    # accessions is not handled as all accessions, and the usual 10x list -- symbols, with accessions
    # for the unnamed genes -- gets gene_ids for those accessions too.
    if not names:
        orientation = "unknown"
    elif ensembl_like == len(names):
        orientation = "ensembl"
    elif ensembl_like == 0:
        orientation = "symbol"
    else:
        orientation = "mixed"

    by_symbol, by_id = build_gtf_index(gtf_path)

    rows: List[Dict[str, str]] = []
    mapped = 0
    ambiguous: List[str] = []
    absent: List[str] = []

    for name in names:
        # An accession the GTF does not hold is still tried as a name below: a
        # GTF without gene names is indexed by accession on both sides.
        if looks_like_ensembl(name) and (
            strip_version(name) in by_id or name not in by_symbol
        ):
            gene_id = strip_version(name)
            symbol = by_id.get(gene_id, "")
            if symbol:
                mapped += 1
            else:
                absent.append(name)
            rows.append({"gene_id": gene_id, "common_id": symbol or MISSING_VALUE})
            continue

        # Symbol -> Ensembl. An empty mapping means the symbol was claimed by
        # more than one gene_id; a missing key means it is not in the GTF.
        if name in by_symbol:
            gene_id = by_symbol[name]
            if gene_id:
                mapped += 1
                rows.append({"gene_id": gene_id, "common_id": name})
            else:
                ambiguous.append(name)
                rows.append({"gene_id": MISSING_VALUE, "common_id": name})
        else:
            absent.append(name)
            rows.append({"gene_id": MISSING_VALUE, "common_id": name})

    left_out: List[str] = []
    if absent:
        any_ids, any_names = _gtf_named_anywhere(gtf_path)
        in_gtf = [
            name for name in absent
            if (name in any_names or strip_version(name) in any_ids)
            and strip_version(name) not in by_id and name not in by_symbol
        ]
        if in_gtf:
            held = set(in_gtf)
            left_out = in_gtf
            absent = [name for name in absent if name not in held]

    report = GeneIdentityReport(
        orientation=orientation,
        total=len(names),
        mapped=mapped,
        ambiguous=ambiguous,
        absent=absent,
        gtf=str(gtf_path),
        left_out=left_out,
    )
    return rows, report
