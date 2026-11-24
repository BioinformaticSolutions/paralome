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

"""Which alleles qualify for gene burden, and for which gene.

One set of rules, read by BOTH sides of the burden test -- the
gnomAD reference table built at genome setup (burden/reference.py) and the
cohort's annotated VCF (the gene_burden stage). The two sides are only
comparable if an allele counts the same way on each, so nothing here reads a
file: every function takes the ANNOVAR values it judges.

Values may come from an ANNOVAR ``-vcfinput`` INFO block (escaped: ``\\x3b``
for ';', ``\\x3d`` for '=', spaces written as '_') or from a
``*_multianno.txt`` column (plain text). ``decode_annovar_value`` makes the
two the same, and every function here calls it, so either may be passed.

Gene key
    The Ensembl gene id from ``Gene.ensGene`` (version suffix removed). A
    variant annotated to several genes counts for each. The symbol from
    ``Gene.refGeneWithVer`` is for display only and never a key.

Coding tiers (consequence from ``Func.ensGene`` / ``ExonicFunc.ensGene``)
    ``ptv``
        ExonicFunc 'stopgain', 'startloss', 'frameshift insertion',
        'frameshift deletion', 'frameshift block substitution';
        or Func token 'splicing' (also as 'exonic;splicing'). ANNOVAR calls a
        variant 'splicing' when it lies within 2 bp of a splice junction by
        default (-splicing_threshold 2); when GeneDetail gives the offset
        (``c.123+2T>C``, ``c.124-1G>A``) an offset above
        ``SPLICE_MAX_DISTANCE`` is not counted, so a database annotated with
        a wider threshold does not widen the tier. 'ncRNA_splicing' is a
        non-coding transcript and is not a PTV.
        'stoploss' is NOT a PTV. 'nonframeshift ...', 'synonymous SNV' and
        'unknown' are not.
    ``ptv_missense``
        every ``ptv`` allele, plus 'nonsynonymous SNV' with
        REVEL_score >= ``REVEL_MIN`` or AlphaMissense_pred == 'P'. When dbNSFP
        gives several values (one per transcript) the highest REVEL and any
        'P' count.

    Tiers are judged without frequency; ``passes_frequency`` is the separate
    model gate on gnomAD AF_grpmax.

Regulatory tier (``regulatory``; run only with gnomAD GENOME frequencies)
    Func token in UTR5, UTR3, upstream, downstream, intronic, or 'intergenic'
    with a flanking gene within ``REGULATORY_MAX_DISTANCE`` bp (GeneDetail
    ``dist=N``). For intergenic each flanking gene within the window is
    assigned (both, when both are). No damage score is used. Frequency comes
    from ``gnomad41_genome_AF*``. 'ncRNA_*' functions are not counted. The
    tier is never mixed into the coding tiers: ``coding_tiers`` never returns
    it and ``regulatory_genes`` is a separate call.

Multi-token Func values
    ANNOVAR writes 'exonic;splicing' or 'upstream;downstream' when a variant
    has one role in one gene and another in a second; the Gene field then
    holds one ';' group per role. When the counts agree the roles are paired
    with their genes; otherwise every gene receives every role.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Union

__all__ = [
    "ALL_POPULATIONS",
    "POPULATIONS_AND_ALL",
    "CODING_TIERS",
    "REGULATORY_TIER",
    "TIERS",
    "MODELS",
    "DOMINANT_MAX_AF",
    "RECESSIVE_MAX_AF",
    "MODEL_MAX_AF",
    "REVEL_MIN",
    "ALPHAMISSENSE_PATHOGENIC",
    "SPLICE_MAX_DISTANCE",
    "REGULATORY_MAX_DISTANCE",
    "PTV_EXONIC_FUNCS",
    "MISSENSE_EXONIC_FUNC",
    "REGULATORY_FUNCS",
    "GNOMAD_SOURCES",
    "KEYS",
    "AlleleCall",
    "decode_annovar_value",
    "parse_float",
    "parse_floats",
    "af_key",
    "allele_frequencies",
    "grpmax_af",
    "passes_frequency",
    "ensembl_gene_ids",
    "gene_symbols",
    "func_tokens",
    "is_ptv",
    "is_damaging_missense",
    "coding_tiers",
    "coding_tiers_by_gene",
    "regulatory_genes",
    "classify_allele",
]

ALL_POPULATIONS = ("afr", "amr", "asj", "eas", "fin", "mid", "nfe", "sas")
# 'all' is gnomAD's overall AF (the key without a population suffix).
POPULATIONS_AND_ALL = ("all",) + ALL_POPULATIONS

CODING_TIERS = ("ptv", "ptv_missense")
REGULATORY_TIER = "regulatory"
TIERS = CODING_TIERS + (REGULATORY_TIER,)
MODELS = ("dominant", "recessive")

DOMINANT_MAX_AF = 0.001
RECESSIVE_MAX_AF = 0.01
MODEL_MAX_AF = {"dominant": DOMINANT_MAX_AF, "recessive": RECESSIVE_MAX_AF}

REVEL_MIN = 0.5
ALPHAMISSENSE_PATHOGENIC = "P"
SPLICE_MAX_DISTANCE = 2
REGULATORY_MAX_DISTANCE = 10_000

PTV_EXONIC_FUNCS = frozenset({
    "stopgain",
    "startloss",
    "frameshift insertion",
    "frameshift deletion",
    "frameshift block substitution",
})
MISSENSE_EXONIC_FUNC = "nonsynonymous SNV"
SPLICING_FUNC = "splicing"
REGULATORY_FUNCS = frozenset({"UTR5", "UTR3", "upstream", "downstream", "intronic"})
INTERGENIC_FUNC = "intergenic"

GNOMAD_SOURCES = {"exome": "gnomad41_exome", "genome": "gnomad41_genome"}

# The ANNOVAR column / INFO key names both sides read (exact spelling).
KEYS = {
    "func": "Func.ensGene",
    "exonic_func": "ExonicFunc.ensGene",
    "gene": "Gene.ensGene",
    "gene_detail": "GeneDetail.ensGene",
    "symbol": "Gene.refGeneWithVer",
    "revel": "REVEL_score",
    "alphamissense_pred": "AlphaMissense_pred",
}

_MISSING = ("", ".", "NA", "nan", "NaN", "None", None)
_ENSEMBL_GENE = re.compile(r"^(ENS[A-Z]*G\d+)(?:\.\d+)?$")
_SPLICE_OFFSET = re.compile(r"c\.-?\*?\d+([+-])(\d+)")
_DIST = re.compile(r"dist=(\d+|NONE)", re.IGNORECASE)

Value = Union[str, float, int, None]


def decode_annovar_value(value: Value) -> str:
    """Undo ANNOVAR's VCF escaping. Plain text passes through unchanged.

    ``\\x3b`` -> ';', ``\\x3d`` -> '='. A bare ``x3b``/``x3d`` (the backslash
    lost by an earlier tool) is decoded too.
    """
    if value is None:
        return ""
    text = str(value)
    if "x3" in text:
        text = (text.replace("\\x3b", ";").replace("\\x3d", "=")
                .replace("x3b", ";").replace("x3d", "="))
    return text


def _is_missing(text: Value) -> bool:
    return text in _MISSING or (isinstance(text, str) and text.strip() in _MISSING)


def parse_floats(value: Value) -> list[float]:
    """Every number in a field ('.' and blanks skipped; ';' ',' '|' split)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [] if math.isnan(float(value)) else [float(value)]
    text = decode_annovar_value(value)
    out: list[float] = []
    for token in re.split(r"[;,|]", text):
        token = token.strip()
        if _is_missing(token):
            continue
        try:
            number = float(token)
        except ValueError:
            continue
        if not math.isnan(number):
            out.append(number)
    return out


def parse_float(value: Value) -> Optional[float]:
    """One number, or None when absent. Several values: the largest."""
    numbers = parse_floats(value)
    return max(numbers) if numbers else None


# ---------------------------------------------------------------- frequency

def af_key(source: str, population: str) -> str:
    """INFO key / column name of a gnomAD AF.

    ``af_key('exome','all')`` -> ``gnomad41_exome_AF``;
    ``af_key('exome','grpmax')`` -> ``gnomad41_exome_AF_grpmax``;
    ``af_key('genome','nfe')`` -> ``gnomad41_genome_AF_nfe``.
    """
    prefix = GNOMAD_SOURCES.get(source, source)
    if population == "all":
        return f"{prefix}_AF"
    return f"{prefix}_AF_{population}"


def allele_frequencies(fields: Mapping[str, Value], source: str = "exome") -> dict[str, Optional[float]]:
    """{'all', each population, 'grpmax'} -> AF or None when absent."""
    keys = POPULATIONS_AND_ALL + ("grpmax",)
    return {pop: parse_float(fields.get(af_key(source, pop))) for pop in keys}


def grpmax_af(fields: Mapping[str, Value], source: str = "exome") -> Optional[float]:
    return parse_float(fields.get(af_key(source, "grpmax")))


def passes_frequency(model: str, grpmax: Value) -> bool:
    """The model's rarity gate on gnomAD AF_grpmax. Absent ('.') passes.

    dominant: AF_grpmax < DOMINANT_MAX_AF; recessive: < RECESSIVE_MAX_AF.
    """
    try:
        limit = MODEL_MAX_AF[model]
    except KeyError:
        raise ValueError(f"unknown burden model {model!r}; expected one of {MODELS}") from None
    af = parse_float(grpmax)
    return af is None or af < limit


# --------------------------------------------------------------------- genes

def _groups(value: Value) -> list[str]:
    """The ';'-separated groups of a gene field (after decoding)."""
    text = decode_annovar_value(value)
    if _is_missing(text):
        return []
    return [part.strip() for part in text.split(";")]


def _ids_in(text: str, symbol_ids: Optional[Mapping[str, str]] = None) -> list[str]:
    """Ensembl gene ids in one group of a gene field.

    ``symbol_ids`` maps a gene NAME to its Ensembl id. ANNOVAR's hg38 ensGene
    names most genes by symbol (A1BG, TTN) and only the unnamed ones by ENSG
    id, so names must be mapped or nearly every protein-coding gene would be
    dropped.
    """
    out = []
    for token in text.split(","):
        token = token.strip()
        match = _ENSEMBL_GENE.match(token)
        if match:
            out.append(match.group(1))
        elif symbol_ids and token in symbol_ids:
            out.append(symbol_ids[token])
    return out


def _unique(items) -> tuple:
    seen: dict = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)


def ensembl_gene_ids(value: Value, symbol_ids: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Ensembl gene ids in an ANNOVAR gene field, versions stripped, in order.

    Separators ';' ',' (and the escaped ``\\x3b``) are all accepted; 'NONE',
    '.' and non-Ensembl tokens are dropped.
    """
    ids: list[str] = []
    for group in _groups(value):
        ids.extend(_ids_in(group, symbol_ids))
    return _unique(ids)


def gene_symbols(value: Value) -> tuple[str, ...]:
    """Display symbols from Gene.refGeneWithVer (never a key)."""
    out = []
    for group in _groups(value):
        for token in group.split(","):
            token = token.strip()
            if token and not _is_missing(token) and token.upper() != "NONE":
                out.append(token)
    return _unique(out)


def func_tokens(func: Value) -> tuple[str, ...]:
    text = decode_annovar_value(func)
    if _is_missing(text):
        return ()
    return tuple(t.strip() for t in text.split(";") if t.strip())


def _exonic_func(value: Value) -> str:
    # -vcfinput writes spaces as '_' ('nonsynonymous_SNV').
    text = decode_annovar_value(value).strip()
    return "" if _is_missing(text) else text.replace("_", " ")


# ------------------------------------------------------------------- tiers

def _splice_within_limit(gene_detail: Value) -> bool:
    """False only when every offset GeneDetail states exceeds the limit."""
    text = decode_annovar_value(gene_detail)
    offsets = [int(m.group(2)) for m in _SPLICE_OFFSET.finditer(text)]
    if not offsets:
        return True
    return min(offsets) <= SPLICE_MAX_DISTANCE


def is_ptv(func: Value, exonic_func: Value, gene_detail: Value = None) -> bool:
    if _exonic_func(exonic_func) in PTV_EXONIC_FUNCS:
        return True
    if SPLICING_FUNC in func_tokens(func):
        return _splice_within_limit(gene_detail)
    return False


def is_damaging_missense(exonic_func: Value, revel: Value, alphamissense_pred: Value) -> bool:
    if _exonic_func(exonic_func) != MISSENSE_EXONIC_FUNC:
        return False
    score = parse_float(revel)
    if score is not None and score >= REVEL_MIN:
        return True
    preds = re.split(r"[;,|]", decode_annovar_value(alphamissense_pred))
    return any(p.strip() == ALPHAMISSENSE_PATHOGENIC for p in preds)


def coding_tiers(func: Value, exonic_func: Value, revel: Value = None,
                 alphamissense_pred: Value = None, gene_detail: Value = None) -> frozenset:
    """The coding tiers an allele qualifies for, frequency not considered."""
    if is_ptv(func, exonic_func, gene_detail):
        return frozenset(CODING_TIERS)
    if is_damaging_missense(exonic_func, revel, alphamissense_pred):
        return frozenset({"ptv_missense"})
    return frozenset()


def _paired(func: Value, gene: Value,
            symbol_ids: Optional[Mapping[str, str]] = None) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """[(func tokens, gene ids)] -- paired by position when the counts agree."""
    tokens = func_tokens(func)
    groups = _groups(gene)
    if len(tokens) > 1 and len(groups) == len(tokens):
        return [((tok,), _unique(_ids_in(grp, symbol_ids))) for tok, grp in zip(tokens, groups)]
    return [(tokens, ensembl_gene_ids(gene, symbol_ids))]


def coding_tiers_by_gene(fields: Mapping[str, Value],
                         symbol_ids: Optional[Mapping[str, str]] = None) -> dict[str, frozenset]:
    """{ENSG: tiers} for one allele's ANNOVAR fields; genes with no tier omitted."""
    exonic = fields.get(KEYS["exonic_func"])
    detail = fields.get(KEYS["gene_detail"])
    revel = fields.get(KEYS["revel"])
    am = fields.get(KEYS["alphamissense_pred"])
    out: dict[str, set] = {}
    pairs = _paired(fields.get(KEYS["func"]), fields.get(KEYS["gene"]), symbol_ids)
    for tokens, genes in pairs:
        # Split roles: ExonicFunc belongs to the 'exonic' gene only.
        role_exonic = exonic if len(pairs) == 1 or "exonic" in tokens else None
        tiers = coding_tiers(";".join(tokens), role_exonic, revel, am, detail)
        if not tiers:
            continue
        for gene in genes:
            out.setdefault(gene, set()).update(tiers)
    return {gene: frozenset(t) for gene, t in out.items()}


def _distances(gene_detail: Value) -> list[Optional[int]]:
    text = decode_annovar_value(gene_detail)
    out: list[Optional[int]] = []
    for match in _DIST.finditer(text):
        raw = match.group(1)
        out.append(None if raw.upper() == "NONE" else int(raw))
    return out


def regulatory_genes(func: Value, gene: Value, gene_detail: Value = None,
                     max_distance: int = REGULATORY_MAX_DISTANCE,
                     symbol_ids: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Genes a non-coding allele is assigned to for the regulatory tier.

    Empty when the allele is not regulatory (coding, splicing, ncRNA_*,
    intergenic beyond ``max_distance``).
    """
    tokens = func_tokens(func)
    if not tokens:
        return ()
    if tokens == (INTERGENIC_FUNC,):
        # 'ENSG1,ENSG2' with GeneDetail 'dist=N1;dist=N2' (either may be NONE).
        names = [t.strip() for t in decode_annovar_value(gene).replace(";", ",").split(",")]
        dists = _distances(gene_detail)
        out = []
        for index, name in enumerate(names):
            match = _ENSEMBL_GENE.match(name)
            gene_id = match.group(1) if match else (symbol_ids or {}).get(name)
            if not gene_id or index >= len(dists):
                continue
            dist = dists[index]
            if dist is not None and dist <= max_distance:
                out.append(gene_id)
        return _unique(out)
    if not all(tok in REGULATORY_FUNCS for tok in tokens):
        return ()
    return _unique(g for _toks, genes in _paired(func, gene, symbol_ids) for g in genes)


@dataclass(frozen=True)
class AlleleCall:
    """One allele judged: coding tiers per gene, and regulatory genes."""

    coding: Mapping[str, frozenset] = field(default_factory=dict)
    regulatory: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()

    def genes_for(self, tier: str) -> tuple[str, ...]:
        if tier == REGULATORY_TIER:
            return self.regulatory
        return tuple(g for g, tiers in self.coding.items() if tier in tiers)


def classify_allele(fields: Mapping[str, Value],
                    max_distance: int = REGULATORY_MAX_DISTANCE,
                    symbol_ids: Optional[Mapping[str, str]] = None) -> AlleleCall:
    """Judge one allele from its ANNOVAR fields (INFO block or multianno row).

    ``symbol_ids``: gene name -> Ensembl id, for a gene field that
    names genes by symbol (gene_symbol_ids).
    """
    return AlleleCall(
        coding=coding_tiers_by_gene(fields, symbol_ids),
        regulatory=regulatory_genes(fields.get(KEYS["func"]), fields.get(KEYS["gene"]),
                                    fields.get(KEYS["gene_detail"]), max_distance, symbol_ids),
        symbols=gene_symbols(fields.get(KEYS["symbol"])),
    )


# ------------------------------------------------------------ gene names

#: Recorded in a burden reference's provenance; gene_burden refuses a reference
#: without it (one built without mapping gene symbols to ids).
GENE_MAPPING = "ensembl_id_or_symbol_via_gtf.v1"

_TRANSCRIPT_VERSION = re.compile(r"\.\d+$")


def _attribute(attributes: str, key: str) -> Optional[str]:
    marker = key + " "
    for item in attributes.split(";"):
        item = item.strip()
        if item.startswith(marker):
            return item[len(marker):].strip().strip('"')
    return None


def gene_symbol_ids(gtf: Optional[str], ensgene_txt: Optional[str] = None) -> tuple[dict[str, str], dict]:
    """({gene name: Ensembl gene id}, what was found) for gene fields that name genes by symbol.

    Two routes, the first preferred:
    * through the transcripts: ANNOVAR's ``<buildver>_ensGene.txt`` gives each
      transcript (column 2) its gene name (column 13); the GTF gives each
      transcript its gene id. A name whose transcripts all belong to one gene
      maps to it, whatever the GTF calls that gene now.
    * the GTF's own gene_name, when that names exactly one gene.
    A name that would map to more than one gene is left out (counted), so an
    allele is never counted for a gene it may not be in.
    """
    import gzip as _gzip
    from collections import defaultdict
    from pathlib import Path as _Path

    stats: dict = {"gtf": gtf, "ensgene": ensgene_txt, "names_by_transcript": 0, "names_by_gene_name": 0,
                   "ambiguous_names": 0}
    if not gtf or not _Path(gtf).is_file():
        stats["reason"] = "no GTF: gene names that are symbols cannot be mapped"
        return {}, stats
    by_name: dict[str, set] = defaultdict(set)
    transcript_gene: dict[str, str] = {}
    opener = _gzip.open if str(gtf).endswith(".gz") else open
    with opener(gtf, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in ("gene", "transcript"):
                continue
            gene = _attribute(parts[8], "gene_id")
            match = _ENSEMBL_GENE.match(gene or "")
            if not match:
                continue
            gene = match.group(1)
            if parts[2] == "gene":
                name = _attribute(parts[8], "gene_name")
                if name:
                    by_name[name].add(gene)
            else:
                transcript = _attribute(parts[8], "transcript_id")
                if transcript:
                    transcript_gene[_TRANSCRIPT_VERSION.sub("", transcript)] = gene
    via_transcripts: dict[str, set] = defaultdict(set)
    if ensgene_txt and _Path(ensgene_txt).is_file():
        with open(ensgene_txt, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 13 or _ENSEMBL_GENE.match(parts[12]):
                    continue
                gene = transcript_gene.get(_TRANSCRIPT_VERSION.sub("", parts[1]))
                if gene:
                    via_transcripts[parts[12]].add(gene)
    out: dict[str, str] = {}
    for name in set(via_transcripts) | set(by_name):
        genes = via_transcripts.get(name) or by_name.get(name) or set()
        if len(genes) == 1:
            out[name] = next(iter(genes))
            stats["names_by_transcript" if name in via_transcripts else "names_by_gene_name"] += 1
        else:
            stats["ambiguous_names"] += 1
    stats["names_mapped"] = len(out)
    return out, stats


def ensgene_table(annovar_db: Optional[str], buildver: Optional[str]) -> Optional[str]:
    """ANNOVAR's ensGene table in ``annovar_db`` (``<buildver>_ensGene.txt``), or None."""
    from pathlib import Path as _Path

    if not annovar_db or not buildver:
        return None
    path = _Path(annovar_db) / f"{buildver}_ensGene.txt"
    return str(path) if path.is_file() else None


def unmapped_gene_names(value: Value, symbol_ids: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Names in a gene field that are neither an Ensembl id nor in ``symbol_ids``."""
    out = []
    for group in _groups(value):
        for token in group.split(","):
            token = token.strip()
            if (not token or _is_missing(token) or token.upper() == "NONE" or _ENSEMBL_GENE.match(token)
                    or (symbol_ids and token in symbol_ids)):
                continue
            out.append(token)
    return _unique(out)
