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

"""The cohort side of gene burden: who carries what, read from the annotated VCF.

One streaming pass over the ANNOVAR-annotated VCF (tens of millions
of records for a whole-genome cohort) that produces, per gene x tier x
model, the probands who carry a qualifying allele, and the genotypes at common
coding SNPs that the ancestry estimate reads.

ANNOVAR PER-ALLELE BLOCKS. ``table_annovar.pl -vcfinput`` appends to INFO, for
each ALT allele in ALT order, a block of keys that starts with
``ANNOVAR_DATE=...`` and ends with the flag ``ALLELE_END``. A multi-allelic
record therefore carries the same keys once per ALT, and a reader that builds
one dict from INFO keeps only the last ALT's annotation for every allele.
``split_annovar_blocks`` keeps them apart. Values are escaped (``\\x3b`` for
';'), so splitting INFO on ';' is safe.

WHY THE PRE-FILTER IS TEXT, NOT ``bcftools view -i``. htslib reads a repeated
INFO key as its first occurrence, so an expression on ``ExonicFunc.ensGene``
judges every ALT of a multi-allelic record by the first ALT's consequence, and
several dbNSFP keys are not valid VCF IDs at all. Every allele the coding tiers
or the ancestry SNPs can use has 'exonic' or 'splicing' in its Func.ensGene
value, so a record without either word anywhere in its line cannot contribute
and is dropped before Python sees it: by ``grep`` behind ``bgzip -dc`` (or
``gzip -dc``) when those are on PATH, by the same substring test in Python
otherwise. The filter is a superset by construction; everything that passes is
judged by ``qualify``. With the regulatory tier on, nothing is pre-filtered.

GENOTYPE FILTERS: site QUAL >= 30; sample DP >= 10 (FORMAT
DP, else the sum of the allele depths); GQ >= 20 when the sample has one; a
heterozygous call's allele balance -- the second allele's share of the two
alleles' depths, from AD or RO/AO -- within 0.25-0.75. A call that fails, or is
missing, carries nothing.

DUPLICATE AND OVERLAPPING RECORDS. FreeBayes run in
adjacent regions can call the same variant twice at a boundary, or two
overlapping representations of one event. An allele seen again at the same
chrom, pos, REF and ALT is read only for the probands whose call was not
already read from a copy passing QUAL (per proband, the first
passing copy counts; an allele with nothing left to read is counted in
``duplicate_alleles_skipped``); the window is the current position, so a
sorted VCF (what the variants stage writes) costs no memory, and an unsorted
one switches to remembering every allele. For the recessive model, two
qualifying alleles of one proband in one gene whose REF spans overlap are ONE
event -- the more damaging is kept (PTV over missense) -- so an overlapping
pair is never a "possible compound heterozygote".

SEX. Common SNPs on chrX outside the pseudo-autosomal regions are
collected apart from the ancestry SNPs (chrX and chrY are left out of the
ancestry estimate: a male's hemizygous calls are not Hardy-Weinberg draws);
burden/ancestry.py infers sex from them when the sheet does not give it.

ABSENT FROM gnomAD. An allele with neither an overall AF nor an
AF_grpmax in the source gnomAD table counts as rare -- correct in principle --
but recurrent caller artefacts look exactly like that. Every qualifying absent
allele's carriers are counted (``absent_allele_probands``) so the stage can
flag genes whose carriers rest on an absent allele seen in several probands.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from paralome.burden import qualify


ANNOVAR_BLOCK_START = "ANNOVAR_DATE"
ANNOVAR_BLOCK_END = "ALLELE_END"

#: INFO keys the coding tiers cannot be judged without (exact spelling).
REQUIRED_INFO_KEYS = (
    qualify.KEYS["func"], qualify.KEYS["exonic_func"], qualify.KEYS["gene"],
    qualify.af_key("exome", "grpmax"),
)
#: The cohort-side key that switches the regulatory tier on (with a reference section for it).
GENOME_GRPMAX_KEY = qualify.af_key("genome", "grpmax")
#: A single-pass scan reports progress every this many records.
PROGRESS_EVERY = 2_000_000
#: INFO key the annotation stage writes before splitting calls (annotation/annovar.py ORIGIN_TAG).
ORIGIN_KEY = "PARALOME_ORIGIN"

#: Common coding SNPs for the ancestry estimate: overall gnomAD exome AF at least this.
ANCESTRY_MIN_AF = 0.05

#: The words a line must contain to be worth parsing for the coding tiers and the ancestry SNPs.
CODING_LINE_WORDS = ("exonic", "splicing")


@dataclass(frozen=True)
class GenotypeFilters:
    min_qual: float = 30.0
    min_depth: int = 10
    min_gq: float = 20.0
    het_balance_low: float = 0.25
    het_balance_high: float = 0.75

    def to_dict(self) -> dict[str, float]:
        return {
            "min_qual": self.min_qual, "min_depth": self.min_depth, "min_gq": self.min_gq,
            "het_balance_low": self.het_balance_low, "het_balance_high": self.het_balance_high,
        }


DEFAULT_FILTERS = GenotypeFilters()


# ------------------------------------------------------------------ probands

_FAMILY_COLUMNS = ("family_id", "family", "pedigree")
ANCESTRY_COLUMN = "ancestry"


@dataclass(frozen=True)
class ProbandChoice:
    used: tuple                 # sample ids, sheet order
    not_used: tuple             # ({"sample_id", "reason"}, ...)
    ancestry: Mapping[str, str]  # sample id -> the sheet's 'ancestry' value, where one is given
    #: Sample id -> 'male' / 'female' / 'unknown' from the sheet's sex (or gender) column, read by
    #: the normaliser sample sheets use (sample_manifest); anything it does not call male or female is unknown.
    sexes: Mapping[str, str] = field(default_factory=dict)


def select_probands(samples: Path) -> ProbandChoice:
    """The affected samples of the sheet, one per family.

    Read with the readers segregation uses: ``read_sample_roles`` decides who
    is affected (the same words), ``read_manifest_rows`` gives the family and
    the optional ``ancestry`` column. Where a family has more than one
    affected member, its proband (relationship 'proband') is used, else the
    first listed; the others are recorded with the reason -- relatives are not
    independent draws, and the test assumes they are.
    """
    from paralome.pipeline.manifest import read_manifest_rows
    from paralome.segregation.table import read_sample_roles

    roles = read_sample_roles(Path(samples))
    _fields, rows = read_manifest_rows(Path(samples))
    from paralome.fastq.manifest import FAMILY_ALIASES, SAMPLE_ALIASES

    # The id and family columns are found by the aliases read_sample_roles
    # accepts; keyed on 'sample_id' alone, a 'sample' sheet lost families, sex and ancestry.
    lowered_fields = [str(f).strip().lower() for f in (_fields or []) if isinstance(f, str)]
    id_column = next((a for a in SAMPLE_ALIASES if a in lowered_fields), "sample_id")
    family_columns = tuple(FAMILY_ALIASES) + tuple(c for c in _FAMILY_COLUMNS if c not in FAMILY_ALIASES)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        lowered = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if isinstance(k, str)}
        sample_id = lowered.get(id_column, "")
        if sample_id:
            by_id[sample_id] = lowered
    affected = [role for role in roles if role.is_affected]
    missing = [role.sample_id for role in affected if role.sample_id not in by_id]
    if missing:
        raise ValueError(
            f"{samples}: affected samples {', '.join(missing[:5])} not found under the sheet's id column "
            f"{id_column!r}; their family, sex and ancestry cannot be read. Give the id column one "
            f"name ({', '.join(SAMPLE_ALIASES)}) and one row per sample")
    families: dict[str, list] = {}
    order: list[str] = []
    for role in affected:
        row = by_id.get(role.sample_id, {})
        family = next((row[c] for c in family_columns if row.get(c)), "") or role.sample_id
        if family not in families:
            order.append(family)
        families.setdefault(family, []).append(role)
    used: list[str] = []
    not_used: list[dict[str, str]] = []
    for family in order:
        members = families[family]
        chosen = next((m for m in members if m.relationship == "proband"), members[0])
        used.append(chosen.sample_id)
        for member in members:
            if member is not chosen:
                not_used.append({
                    "sample_id": member.sample_id,
                    "reason": f"another affected member of family {family} ({chosen.sample_id}) is used; "
                              "relatives are not independent",
                })
    ancestry = {
        sample: by_id.get(sample, {}).get(ANCESTRY_COLUMN, "")
        for sample in used if by_id.get(sample, {}).get(ANCESTRY_COLUMN, "")
    }
    sexes = {sample: sheet_sex(by_id.get(sample, {})) for sample in used}
    return ProbandChoice(tuple(used), tuple(not_used), ancestry, sexes)


_SEX_COLUMNS = ("sex", "gender")


def sheet_sex(row: Mapping[str, str]) -> str:
    """'male', 'female' or 'unknown' from a sheet row (lower-cased keys)."""
    from paralome.sample_manifest import _normalize_sex

    value = next((row[c] for c in _SEX_COLUMNS if row.get(c)), "")
    sex = _normalize_sex(value)
    return sex if sex in ("male", "female") else "unknown"


# --------------------------------------------------------------------- header

@dataclass(frozen=True)
class VcfHeader:
    info_ids: frozenset
    samples: tuple


def read_vcf_header(path: Path) -> VcfHeader:
    info: set[str] = set()
    samples: tuple = ()
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("##INFO=<ID="):
                info.add(line[len("##INFO=<ID="):].split(",", 1)[0])
                continue
            if line.startswith("#CHROM"):
                samples = tuple(line.rstrip("\n").split("\t")[9:])
                break
            if not line.startswith("#"):
                break
    return VcfHeader(frozenset(info), samples)


def _open_text(path: Path):
    path = Path(path)
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


# ------------------------------------------------------------------- reading

@dataclass
class LineSource:
    """Where the records come from, and how they were narrowed (for the summary)."""

    method: str
    words: tuple = ()


def iter_record_lines(path: Path, *, words: Sequence[str] = (), source: Optional[LineSource] = None) -> Iterator[str]:
    """Data lines of the VCF, optionally only those containing one of ``words``.

    External ``grep`` behind ``bgzip -dc``/``gzip -dc`` when available (fast on
    25 M records); the same test in Python otherwise. Both give the same lines.
    """
    path = Path(path)
    words = tuple(words)
    grep = shutil.which("grep") if words else None
    decompress = None
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if grep and compressed:
        for tool, argv in (("bgzip", ["-dc"]), ("gzip", ["-dc"])):
            found = shutil.which(tool)
            if found:
                decompress = [found, *argv, str(path)]
                break
    if grep and (decompress or not compressed):
        yield from _iter_external(path, words, grep, decompress, source)
        return
    if source is not None:
        source.method, source.words = ("python-filter" if words else "python"), words
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            if words and not any(word in line for word in words):
                continue
            yield line


def _iter_external(path: Path, words, grep, decompress, source) -> Iterator[str]:
    pattern = "|".join(words)
    env = {**os.environ, "LC_ALL": "C"}
    grep_cmd = [grep, "-E", pattern] + ([] if decompress else [str(path)])
    first = None
    if decompress:
        first = subprocess.Popen(decompress, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    second = subprocess.Popen(
        grep_cmd, stdin=first.stdout if first else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    if first is not None and first.stdout is not None:
        first.stdout.close()          # grep owns it now; the decompressor sees SIGPIPE if grep dies
    if source is not None:
        source.method = (Path(decompress[0]).name + "|grep") if decompress else "grep"
        source.words = tuple(words)
    assert second.stdout is not None
    try:
        for raw in second.stdout:
            line = raw.decode("utf-8", errors="replace")
            if line.startswith("#"):
                continue
            yield line
    finally:
        second.stdout.close()
        grep_code = second.wait()
        first_code = first.wait() if first is not None else 0
        first_err = first.stderr.read().decode(errors="replace").strip() if first is not None and first.stderr else ""
        if first is not None and first.stderr:
            first.stderr.close()
        grep_err = second.stderr.read().decode(errors="replace").strip() if second.stderr else ""
        if second.stderr:
            second.stderr.close()
    # grep exits 1 when nothing matched, which is an answer, not a failure.
    if first_code != 0:
        raise RuntimeError(f"{decompress[0]} could not read {path} (exit {first_code}): {first_err}")
    if grep_code not in (0, 1):
        raise RuntimeError(f"grep failed on {path} (exit {grep_code}): {grep_err}")


# ------------------------------------------------------------- ANNOVAR blocks

def split_annovar_blocks(info: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    """(site INFO, [one dict per ANNOVAR allele block, in ALT order]).

    Keys before the first ``ANNOVAR_DATE`` are the caller's own (FreeBayes'
    DP, AO, ...). A block runs from ``ANNOVAR_DATE`` to the flag
    ``ALLELE_END``. A record annotated without -vcfinput blocks (no
    ANNOVAR_DATE) gives its whole INFO as one block, so a single-ALT record
    from such a file is still read.
    """
    site: dict[str, str] = {}
    blocks: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None
    if not info or info == ".":
        return site, blocks
    for item in info.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item, "true"
        if key == ANNOVAR_BLOCK_START:
            current = {key: value}
            blocks.append(current)
            continue
        if key == ANNOVAR_BLOCK_END:
            current = None
            continue
        if current is not None:
            current[key] = value
        else:
            site[key] = value
    if not blocks and qualify.KEYS["func"] in site:
        blocks = [dict(site)]
    return site, blocks


# ----------------------------------------------------------------- genotypes

def _int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def allele_depths(values: Mapping[str, str], n_alleles: int) -> Optional[list[int]]:
    """Depth per allele (REF first): FORMAT AD, else RO and AO. None if neither is complete."""
    ad = values.get("AD")
    if ad and ad != ".":
        parts = [_int(x) for x in ad.split(",")]
        if len(parts) == n_alleles and all(p is not None for p in parts):
            return [int(p) for p in parts]  # type: ignore[arg-type]
    ro, ao = values.get("RO"), values.get("AO")
    if ro not in (None, "", ".") and ao not in (None, "", "."):
        ref = _int(str(ro))
        alts = [_int(x) for x in str(ao).split(",")]
        if ref is not None and len(alts) == n_alleles - 1 and all(a is not None for a in alts):
            return [ref, *[int(a) for a in alts]]  # type: ignore[misc]
    return None


def filtered_alleles(
    fmt_keys: Sequence[str], sample_value: str, n_alleles: int, filters: GenotypeFilters = DEFAULT_FILTERS,
    *, split: Union[bool, str] = False,
) -> tuple[Optional[tuple[int, ...]], str]:
    """(allele indices of a call that passes the genotype filters, or None; why not).

    The QUAL filter is the site's, applied by the caller.

    ``split`` -- the record was split out of a multi-allelic or
    complex call before annotation (INFO/PARALOME_ORIGIN). True: the allele
    balance is not tested; depth and GQ still are, from the original call.

    ``split="multi_alt"`` (the scan's value for a piece of a record
    with two or more ALTs; "ref_zero", the former name, means the same). A
    1/2 call becomes 1/0 with AD [RO, AO1] and 0/1 with AD [RO, AO2]; the other
    ALT's reads are in FORMAT/DP, not in the piece's AD. The call passes when
    its balance is in range against REF or against the other ALTs' share of DP.
    """
    raw = sample_value.split(":")
    values = {key: (raw[i] if i < len(raw) else ".") for i, key in enumerate(fmt_keys)}
    gt = values.get("GT", raw[0] if raw else ".")
    tokens = gt.replace("|", "/").split("/")
    alleles: list[int] = []
    for token in tokens:
        index = _int(token)
        if index is None:
            return None, "no_call"
        if index < 0 or index >= n_alleles:
            return None, "bad_allele_index"
        alleles.append(index)
    if not alleles:
        return None, "no_call"
    if len(alleles) == 1:
        alleles = alleles * 2            # a haploid call counts as homozygous
    depths = allele_depths(values, n_alleles)
    depth = _int(values.get("DP", "."))
    if depth is None and depths is not None:
        depth = sum(depths)
    if depth is None:
        return None, "no_depth"
    if depth < filters.min_depth:
        return None, "low_depth"
    gq = values.get("GQ")
    if gq not in (None, "", "."):
        quality = _float(str(gq))
        if quality is None or quality < filters.min_gq:
            return None, "low_gq"
    distinct = sorted(set(alleles))
    if split in ("multi_alt", "ref_zero") and len(distinct) == 2:
        # A piece of a record with two or more ALTs. bcftools norm -m keeps the real REF depth on
        # each half of a 1/2 call (AD 1,15,14 becomes 1,15 and 1,14), so one stray REF read --
        # routine for indels in repeats -- must not put the half through the ordinary het test.
        # The piece's partner allele on the other haplotype is REF or another ALT of the record; the
        # other ALTs' reads are what FORMAT/DP holds beyond this piece's REF and ALT depths. The call
        # passes when its balance is in range against either partner: against REF it is the ordinary
        # 0/1 test, against the other ALTs it is the test the unsplit 1/2 call would have had (AO1 vs AO2).
        if depths is None:
            return None, "no_allele_depths"
        low, high = distinct
        reported = _int(values.get("DP", "."))
        other = max(reported - depths[low] - depths[high], 0) if reported is not None else 0
        # The other ALTs are the partner only when they outnumber REF (a 0/1
        # with 18 REF and 2 ALT reads does not pass by pairing with 1 stray read).
        partners = [p for p in (depths[low], other if other >= depths[low] else 0) if p > 0]
        if partners and not any(
            filters.het_balance_low <= depths[high] / (depths[high] + p) <= filters.het_balance_high
            for p in partners
        ):
            return None, "allele_balance"
        # No partner read at all (a 1/2 half with RO 0 and no DP beyond the piece): not tested.
        split = True
    if len(distinct) == 2 and not split:
        if depths is None:
            return None, "no_allele_depths"
        low, high = distinct
        pair = depths[low] + depths[high]
        if pair <= 0:
            return None, "no_allele_depths"
        balance = depths[high] / pair
        if not (filters.het_balance_low <= balance <= filters.het_balance_high):
            return None, "allele_balance"
    elif len(distinct) > 2:
        return None, "polyploid_call"
    return tuple(alleles), ""


# ------------------------------------------------------------------ the scan

#: Consequences ranked for "the more damaging" of two overlapping alleles.
_PTV_CONSEQUENCES = frozenset(c.replace(" ", "_") for c in qualify.PTV_EXONIC_FUNCS) | {"splicing"}


@dataclass
class Hit:
    """One qualifying allele carried by one proband in one gene."""

    variant: str            # chrom:pos:ref>alt
    copies: int             # 1 heterozygous, 2 homozygous
    consequence: str
    grpmax: Optional[float]
    revel: Optional[float]
    alphamissense: str
    absent: bool = False    # No AF and no AF_grpmax in the source gnomAD table
    #: "<chrom>:<POS>" of the FreeBayes record this allele was split from before annotation
    #: (INFO/PARALOME_ORIGIN; the record, not the ALT). "" for a call that was not split.
    origin: str = ""
    #: Which genotype slot holds this allele in a split record -- "a" (1/0), "b" (0/1) or "ab"
    #: (1/1). bcftools norm -m keeps a 1/2 call's order, so the two ALTs' pieces sit in different slots.
    slot: str = ""
    #: The allele's gnomAD AF per population (qualify.POPULATIONS_AND_ALL order, None where gnomAD
    #: has none) from the tier's table, so an allele dropped as too common in the cohort (burden/artefacts.py)
    #: can be taken out of the gene's expectation too.
    afs: Optional[tuple] = None

    def label(self) -> str:
        af = "." if self.grpmax is None else f"{self.grpmax:.3g}"
        revel = "." if self.revel is None else f"{self.revel:.3g}"
        zygosity = "hom" if self.copies >= 2 else "het"
        absent = " absent_from_gnomAD" if self.absent else ""
        return (f"{self.variant} {zygosity} {self.consequence} "
                f"AF_grpmax={af} REVEL={revel} AlphaMissense={self.alphamissense or '.'}{absent}")

    def span(self) -> tuple[str, int, int]:
        """(chrom, first, last) of the REF bases, 1-based inclusive, from ``variant``."""
        chrom, pos, alleles = self.variant.rsplit(":", 2)
        ref = alleles.split(">", 1)[0]
        start = int(pos)
        return chrom, start, start + max(len(ref), 1) - 1

    def damage(self) -> int:
        """2 a PTV consequence, 1 anything else that qualified (missense)."""
        tokens = set(self.consequence.replace(";", ",").split(","))
        return 2 if tokens & _PTV_CONSEQUENCES else 1


@dataclass
class GeneSite:
    chrom: str
    start: int
    end: int
    symbols: set = field(default_factory=set)


@dataclass
class CohortScan:
    """What the pass found. ``hits[model][tier][gene][proband] -> [Hit]``."""

    probands: tuple
    hits: dict = field(default_factory=dict)
    genes: dict = field(default_factory=dict)          # gene -> GeneSite, from the cohort's own variants
    ancestry_afs: list = field(default_factory=list)   # [[AF per ALL_POPULATIONS]] per SNP
    ancestry_dosages: list = field(default_factory=list)  # [[alt dosage or -1 per proband]] per SNP
    #: Common chrX non-PAR SNPs for sex inference: overall AF, and dosages as above.
    sex_afs: list = field(default_factory=list)
    sex_dosages: list = field(default_factory=list)
    #: Qualifying alleles absent from gnomAD -> how many probands carry each.
    absent_allele_probands: Counter = field(default_factory=Counter)
    #: Distinct ANNOVAR_DATE values seen in the records (annotation provenance).
    annovar_dates: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    failed_calls: Counter = field(default_factory=Counter)
    source: dict = field(default_factory=dict)

    def add(self, model: str, tier: str, gene: str, proband: str, hit: Hit) -> None:
        self.hits.setdefault(model, {}).setdefault(tier, {}).setdefault(gene, {}).setdefault(proband, []).append(hit)

    def carriers(self, model: str, tier: str, *, sexes: Optional[Mapping[str, str]] = None,
                 regions: Optional[Mapping[str, Optional[str]]] = None,
                 hits: Optional[dict] = None) -> dict[str, dict[str, list[Hit]]]:
        """gene -> {proband: hits} of the probands who count under ``model``.

        Dominant: at least one qualifying allele. Recessive: homozygous for one,
        or two or more qualifying heterozygous events in the gene (phase
        unknown -- a possible compound heterozygote; overlapping alleles are one
        event). With ``sexes`` (proband -> male/female/unknown) and ``regions``
        (gene -> 'X'/'Y' outside the PARs, see stats.gene_region): a male there
        counts with one qualifying allele under both models; a female on Y, and
        anyone of unknown sex on X or Y, is not in the test. ``hits``: another
        tree of the same shape to count from (the hits with the
        alleles too common for gnomAD taken out, burden/artefacts.py).
        """
        out: dict[str, dict[str, list[Hit]]] = {}
        for gene, by_proband in (self.hits if hits is None else hits).get(model, {}).get(tier, {}).items():
            region = (regions or {}).get(gene) if sexes is not None else None
            kept = {}
            for proband, hits in by_proband.items():
                sex = (sexes or {}).get(proband, "unknown")
                if carrier_status(model, hits, sex, region):
                    kept[proband] = hits
            if kept:
                out[gene] = kept
        return out


def events(hits: Sequence[Hit]) -> list[Hit]:
    """One hit per event: hits whose REF spans overlap (same chromosome) are one event, represented by the
    more damaging (PTV over missense; then more copies; then the variant name, for a stable choice). Exact
    duplicates are one event too. Two different ALTs of one position and REF are two events (in trans).

    Hits split out of one FreeBayes record before annotation (same ``origin``) are grouped first, then by
    the genotype slot each piece carries: pieces in slot "a" are one haplotype, pieces in
    slot "b" the other, so a 1/2 call gives at most two events, in trans; a piece on both ("ab", 1/1 or a
    piece bcftools joined from both ALTs) belongs to both. Within a slot the most damaging piece stands
    for the haplotype; the two slots' representatives stay two events through the overlap
    merge, whatever REF spans the left-aligned pieces have. A group whose pieces carry no slot is one
    event, unless it holds two different ALTs at one position. Neither rule invents a compound
    heterozygote from the pieces of a single allele.
    """
    rank = lambda h: (-h.damage(), -h.copies, h.variant)  # noqa: E731
    groups: dict[str, list[Hit]] = {}
    unsplit: list[Hit] = []
    #: The two haplotypes' representatives of one record, known to be in trans.
    in_trans: list[tuple[Hit, Hit]] = []
    for hit in hits:
        (groups.setdefault(hit.origin, []) if hit.origin else unsplit).append(hit)
    for group in groups.values():
        # One event per haplotype of the record. The pieces in genotype slot "a" (1/0) are one
        # haplotype and those in "b" (0/1) the other: the two ALTs of a 1/2 call, in trans, whatever
        # positions the left-aligned pieces landed on. A piece with no slot recorded falls back to
        # grouping by the record alone.
        slotted = [h for h in group if h.slot]
        if slotted and len(slotted) == len(group):
            chosen = {}
            for side in ("a", "b"):
                side_hits = [h for h in group if side in h.slot]
                if side_hits:
                    best = min(side_hits, key=rank)
                    chosen[side] = best
                    if best not in unsplit:
                        unsplit.append(best)
            if len(chosen) == 2 and chosen["a"] != chosen["b"]:
                in_trans.append((chosen["a"], chosen["b"]))
            continue
        by_site: dict[str, dict[str, Hit]] = {}
        for hit in group:
            site, alt = hit.variant.rsplit(">", 1)
            by_site.setdefault(site, {}).setdefault(alt, hit)
        pairs = [alts for alts in by_site.values() if len(alts) >= 2]
        if pairs:
            unsplit.extend(sorted(pairs[0].values(), key=rank)[:2])
        else:
            unsplit.append(min(group, key=rank))
    hits = unsplit
    ordered = sorted(hits, key=lambda h: h.span())
    clusters: list[list[Hit]] = []
    last: Optional[tuple[str, int]] = None
    for hit in ordered:
        chrom, start, end = hit.span()
        if clusters and last is not None and last[0] == chrom and start <= last[1]:
            clusters[-1].append(hit)
            last = (chrom, max(last[1], end))
        else:
            clusters.append([hit])
            last = (chrom, end)
    out: list[Hit] = []
    for cluster in clusters:
        # The two slot representatives of one record are two events even when their REF spans overlap
        # (the left-aligned halves of a 1/2 indel call, ACAC>A and ACACAC>A, overlap with different
        # REFs), so the compound heterozygote is kept. Overlapping hits of other records are the same
        # event as one of the two haplotypes.
        pair = next(((a, b) for a, b in in_trans if a in cluster and b in cluster), None)
        if pair is not None:
            out.extend(sorted(pair, key=rank))
            continue
        out.append(min(cluster, key=lambda h: (-h.damage(), -h.copies, h.variant)))
        # Two different ALTs at one position and REF (a 1/2 call, or the same site split into two records)
        # cannot sit on one haplotype: they are two events, in trans.
        by_site: dict[str, dict[str, Hit]] = {}
        for hit in cluster:
            site, alt = hit.variant.rsplit(">", 1)
            by_site.setdefault(site, {}).setdefault(alt, hit)
        pairs = [alts for alts in by_site.values() if len(alts) >= 2]
        if pairs:
            ranked = sorted(pairs[0].values(), key=lambda h: (-h.damage(), -h.copies, h.variant))
            out[-1] = ranked[0]
            out.append(ranked[1])
    return out


def recessive_status(hits: Sequence[Hit]) -> str:
    """'homozygous', 'possible_compound_het' or '' for one proband's hits in one gene (diploid)."""
    kept = events(hits)
    if any(hit.copies >= 2 for hit in kept):
        return "homozygous"
    # Two calls of one variant, or of overlapping variants, are one event; count events, not hits.
    if len(kept) >= 2:
        return "possible_compound_het"
    return ""


def carrier_status(model: str, hits: Sequence[Hit], sex: str = "female", region: Optional[str] = None) -> str:
    """Why a proband counts ('carrier', 'homozygous', 'possible_compound_het', 'hemizygous'), or ''.

    ``region`` None: diploid. 'X'/'Y': outside the PARs -- a male is hemizygous
    (one qualifying allele counts, under both models); a female on X is diploid;
    a female on Y, or a proband of unknown sex, is not in the test ('').
    """
    if not hits:
        return ""
    if region is not None:
        if sex == "male":
            return "hemizygous"
        if sex != "female" or region == "Y":
            return ""
    if model == "dominant":
        return "carrier"
    return recessive_status(hits)


def in_sex_aware_test(sex: str, region: Optional[str]) -> bool:
    """Whether a proband of ``sex`` is in a gene's test at all (see carrier_status)."""
    if region is None:
        return True
    return sex == "male" or (sex == "female" and region == "X")


def _consequence(fields: Mapping[str, str]) -> str:
    exonic = qualify.decode_annovar_value(fields.get(qualify.KEYS["exonic_func"])).replace("_", " ").strip()
    func = qualify.decode_annovar_value(fields.get(qualify.KEYS["func"])).strip()
    if exonic and exonic != ".":
        return exonic.replace(" ", "_")
    return func or "."


def _ancestry_afs(fields: Mapping[str, str]) -> Optional[list[float]]:
    overall = qualify.parse_float(fields.get(qualify.af_key("exome", "all")))
    if overall is None or overall < ANCESTRY_MIN_AF:
        return None
    afs = []
    for population in qualify.ALL_POPULATIONS:
        value = qualify.parse_float(fields.get(qualify.af_key("exome", population)))
        if value is None:
            return None
        afs.append(value)
    return afs


def _absent(fields: Mapping[str, str], source: str) -> bool:
    """No overall AF and no AF_grpmax in the source gnomAD table: absent from gnomAD."""
    return (qualify.parse_float(fields.get(qualify.af_key(source, "all"))) is None
            and qualify.grpmax_af(fields, source) is None)


def _scan_lines(
    lines: Iterable[str],
    samples: Sequence[str],
    probands: Sequence[str],
    *,
    regulatory: bool,
    filters: GenotypeFilters,
    par: Mapping[str, Sequence[tuple]],
    progress: Optional[Callable[[int], None]] = None,
    symbol_ids: Optional[Mapping[str, str]] = None,
) -> CohortScan:
    """The scan of some of the VCF's record lines (all of them, or one contig's). ``samples``: the header's.

    ``symbol_ids``: gene name -> Ensembl id, for an ensGene that names genes by symbol.
    """
    from paralome.burden.stats import gene_region, sex_chromosome

    index = {sample: i for i, sample in enumerate(samples)}
    columns = [(sample, index[sample]) for sample in probands]
    scan = CohortScan(probands=tuple(probands))
    tiers = qualify.CODING_TIERS + ((qualify.REGULATORY_TIER,) if regulatory else ())

    # Exact duplicate alleles are read once. Sorted input: only the current position is remembered.
    # Each allele maps to the probands whose call was already read at a site passing QUAL; a
    # duplicate is read again for the others, so a first copy that failed QUAL or a genotype filter does
    # not hide a passing second copy.
    window_key: Optional[tuple[str, int]] = None
    window: dict = {}
    everything: Optional[dict] = None

    for line in lines:
        if progress is not None and scan.counts["records_read"] and not scan.counts["records_read"] % PROGRESS_EVERY:
            progress(scan.counts["records_read"])
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 8:
            scan.counts["malformed_records"] += 1
            continue
        scan.counts["records_read"] += 1
        chrom, pos_text, _id, ref, alt_text, qual_text, _filter, info = parts[:8]
        alts = alt_text.split(",")
        pos = _int(pos_text)
        if pos is None:
            scan.counts["malformed_records"] += 1
            continue
        site_info, blocks = split_annovar_blocks(info)
        # The FreeBayes record this one was split from, "<chrom>:<POS>" (the ALT number is dropped:
        # bcftools joins identical pieces of two ALTs into one record that keeps only ALT1's value).
        origin_value = site_info.get(ORIGIN_KEY) or ""
        origin_parts = origin_value.split("_")
        record_origin = f"{chrom}:{origin_parts[0]}" if origin_value else ""
        # A piece of a single-ALT record is tested as it stands. A piece of a record with two or more
        # ALTs is tested against its partner allele, REF or the other ALTs' share of DP (see
        # filtered_alleles). A tag that has no ALT count is treated as multi-allelic.
        if not origin_value:
            split_record = False
        elif len(origin_parts) >= 3 and origin_parts[2].isdigit() and int(origin_parts[2]) <= 1:
            split_record = False
        else:
            split_record = "multi_alt"
        if not blocks:
            scan.counts["records_without_annovar_annotation"] += 1
            continue
        if len(blocks) != len(alts):
            # Refused rather than guessed: which block belongs to which ALT is
            # exactly what cannot be known here. Counted and reported.
            scan.counts["records_with_block_count_not_matching_alts"] += 1
            continue
        for fields in blocks:
            date = fields.get(ANNOVAR_BLOCK_START)
            if date and (date in scan.annovar_dates or len(scan.annovar_dates) < 20):
                scan.annovar_dates[date] += 1

        if everything is None and (chrom, pos) != window_key:
            if window_key is not None and chrom == window_key[0] and pos < window_key[1]:
                scan.counts["records_out_of_order"] += 1
                everything = dict(window)         # unsorted: remember every allele from here on
            window_key, window = (chrom, pos), {}
        seen = window if everything is None else everything
        duplicate = []
        already: list = []
        for alt in alts:
            key = (chrom, pos, ref, alt)
            duplicate.append(key in seen)
            already.append(frozenset(seen.setdefault(key, set())))
        if all(duplicate) and all(len(done) >= len(probands) for done in already):
            scan.counts["duplicate_records_skipped"] += 1
            continue

        qual = _float(qual_text)
        site_passes = qual is not None and qual >= filters.min_qual
        if not site_passes:
            scan.counts["records_below_qual"] += 1

        calls: dict[str, Optional[tuple[int, ...]]] = {}
        if site_passes:
            fmt_keys = parts[8].split(":") if len(parts) > 8 else []
            n_alleles = len(alts) + 1
            for sample, column in columns:
                value = parts[9 + column] if 9 + column < len(parts) else "."
                alleles, why = filtered_alleles(fmt_keys, value, n_alleles, filters, split=split_record)
                if alleles is None:
                    scan.failed_calls[why] += 1
                calls[sample] = alleles
            for alt in alts:
                seen[(chrom, pos, ref, alt)].update(s for s, a in calls.items() if a is not None)

        # Common coding SNPs: the ancestry estimate (autosomes) and sex inference (chrX outside the PARs).
        if (site_passes and not duplicate[0] and len(alts) == 1 and len(ref) == 1 and len(alts[0]) == 1
                and alts[0] != "*"):
            fields = blocks[0]
            if "exonic" in qualify.func_tokens(fields.get(qualify.KEYS["func"])):
                dosages = [
                    -1 if calls.get(sample) is None else sum(1 for a in calls[sample] if a == 1)  # type: ignore[union-attr]
                    for sample in probands
                ]
                if sex_chromosome(chrom) is not None:
                    if gene_region(chrom, pos, pos, par) == "X":
                        overall = qualify.parse_float(fields.get(qualify.af_key("exome", "all")))
                        if overall is not None and ANCESTRY_MIN_AF <= overall <= 1.0 - ANCESTRY_MIN_AF:
                            scan.sex_afs.append(overall)
                            scan.sex_dosages.append(dosages)
                else:
                    afs = _ancestry_afs(fields)
                    if afs is not None:
                        scan.ancestry_afs.append(afs)
                        scan.ancestry_dosages.append(dosages)

        for allele_index, (alt, fields) in enumerate(zip(alts, blocks), start=1):
            if alt == "*":
                continue
            done = already[allele_index - 1]
            if duplicate[allele_index - 1] and (not site_passes or len(done) >= len(probands)):
                scan.counts["duplicate_alleles_skipped"] += 1
                continue
            call = qualify.classify_allele(fields, symbol_ids=symbol_ids)
            any_tier = False
            absent_anywhere = False
            for tier in tiers:
                genes = call.genes_for(tier)
                if not genes:
                    continue
                any_tier = True
                source_name = "genome" if tier == qualify.REGULATORY_TIER else "exome"
                grpmax = qualify.grpmax_af(fields, source_name)
                absent = _absent(fields, source_name)
                in_reference = not absent
                if absent and source_name == "exome" and not _absent(fields, "genome"):
                    # An allele outside gnomAD's exome capture is not in its exome table,
                    # and was read as never seen: GNB1 led a 40-proband run's PTV tables on two common intronic
                    # alleles (gnomAD genomes 0.41 and 0.25) that ensGene calls splicing. The genome table
                    # judges its frequency; it is absent only when neither table has it. (The exome
                    # reference has no such allele, so nothing comes out of the expectation for it.)
                    grpmax, absent = qualify.grpmax_af(fields, "genome"), False
                absent_anywhere = absent_anywhere or absent
                hit_base = None
                for model in qualify.MODELS:
                    if not qualify.passes_frequency(model, grpmax):
                        continue
                    for gene in genes:
                        site = scan.genes.get(gene)
                        if site is None:
                            scan.genes[gene] = GeneSite(chrom, pos, pos, set(call.symbols))
                        else:
                            if site.chrom == chrom:
                                site.start, site.end = min(site.start, pos), max(site.end, pos)
                            site.symbols.update(call.symbols)
                    if not site_passes:
                        continue
                    for sample in probands:
                        alleles = calls.get(sample)
                        if not alleles or sample in done:
                            continue
                        copies = sum(1 for a in alleles if a == allele_index)
                        if copies <= 0:
                            continue
                        if hit_base is None:
                            hit_base = Hit(
                                variant=f"{chrom}:{pos}:{ref}>{alt}", copies=0,
                                consequence=_consequence(fields), grpmax=grpmax,
                                revel=qualify.parse_float(fields.get(qualify.KEYS["revel"])),
                                alphamissense=qualify.decode_annovar_value(
                                    fields.get(qualify.KEYS["alphamissense_pred"])).strip(),
                                absent=absent,
                                afs=(tuple(qualify.allele_frequencies(fields, source_name)[pop]
                                           for pop in qualify.POPULATIONS_AND_ALL) if in_reference else None),
                                origin=record_origin if len(alts) == 1 else "",
                            )
                        slot = ""
                        if hit_base.origin:
                            slot = "".join(name for name, a in zip("ab", alleles) if a == allele_index)
                        hit = Hit(**{**hit_base.__dict__, "copies": copies, "slot": slot})
                        for gene in genes:
                            scan.add(model, tier, gene, sample, hit)
            if any_tier and not duplicate[allele_index - 1]:
                scan.counts["qualifying_alleles_seen"] += 1
            if absent_anywhere and site_passes:
                carriers = sum(1 for sample in probands
                               if sample not in done and calls.get(sample)
                               and any(a == allele_index for a in calls[sample]))  # type: ignore[union-attr]
                if carriers:
                    scan.absent_allele_probands[f"{chrom}:{pos}:{ref}>{alt}"] += carriers

    return scan


def _tabix_index(vcf: Path) -> Optional[Path]:
    for suffix in (".tbi", ".csi"):
        index = Path(str(vcf) + suffix)
        if index.is_file():
            return index
    return None


def vcf_contigs(vcf: Path) -> Optional[list[str]]:
    """The contigs of an indexed VCF, in index order (``tabix -l``); None when that cannot be read."""
    tabix = shutil.which("tabix")
    if tabix is None or _tabix_index(vcf) is None:
        return None
    completed = subprocess.run([tabix, "-l", str(vcf)], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _iter_contig_lines(vcf: Path, contig: str, words: Sequence[str]) -> Iterator[str]:
    """One contig's record lines through ``tabix``, filtered by ``grep -E`` for ``words`` when given."""
    tabix = shutil.which("tabix") or "tabix"
    if words:
        yield from _iter_external(vcf, words, shutil.which("grep") or "grep", [tabix, str(vcf), contig], None)
        return
    process = subprocess.Popen([tabix, str(vcf), contig], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ, "LC_ALL": "C"})
    assert process.stdout is not None
    try:
        for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("#"):
                yield line
    finally:
        process.stdout.close()
        code = process.wait()
        error = process.stderr.read().decode(errors="replace").strip() if process.stderr else ""
        if process.stderr:
            process.stderr.close()
    if code != 0:
        raise RuntimeError(f"tabix could not read {contig} of {vcf} (exit {code}): {error}")


def _scan_contig(vcf: str, contig: str, samples: tuple, probands: tuple, regulatory: bool,
                 filters: GenotypeFilters, words: tuple, par: dict,
                 symbol_ids: Optional[dict] = None) -> tuple[str, CohortScan]:
    """One worker's part of the parallel scan (module level, so a process pool can run it)."""
    lines = _iter_contig_lines(Path(vcf), contig, words)
    return contig, _scan_lines(lines, samples, probands, regulatory=regulatory, filters=filters, par=par,
                               symbol_ids=symbol_ids)


def merge_scans(parts: Sequence[CohortScan], probands: Sequence[str]) -> CohortScan:
    """The scans of several contigs as one, in the order given (the VCF's contig order).

    Hits are joined per model, tier, gene and proband; a gene's span is the
    widest on its chromosome; SNP lists are joined in order; counters add.
    """
    out = CohortScan(probands=tuple(probands))
    for part in parts:
        for model, by_tier in part.hits.items():
            for tier, by_gene in by_tier.items():
                for gene, by_proband in by_gene.items():
                    for proband, hits in by_proband.items():
                        (out.hits.setdefault(model, {}).setdefault(tier, {}).setdefault(gene, {})
                         .setdefault(proband, []).extend(hits))
        for gene, site in part.genes.items():
            kept = out.genes.get(gene)
            if kept is None:
                out.genes[gene] = GeneSite(site.chrom, site.start, site.end, set(site.symbols))
                continue
            if kept.chrom == site.chrom:
                kept.start, kept.end = min(kept.start, site.start), max(kept.end, site.end)
            kept.symbols.update(site.symbols)
        out.ancestry_afs.extend(part.ancestry_afs)
        out.ancestry_dosages.extend(part.ancestry_dosages)
        out.sex_afs.extend(part.sex_afs)
        out.sex_dosages.extend(part.sex_dosages)
        out.absent_allele_probands.update(part.absent_allele_probands)
        out.annovar_dates.update(part.annovar_dates)
        out.counts.update(part.counts)
        out.failed_calls.update(part.failed_calls)
    return out


def scan_cohort(
    vcf: Path,
    probands: Sequence[str],
    *,
    regulatory: bool = False,
    filters: GenotypeFilters = DEFAULT_FILTERS,
    words: Optional[Sequence[str]] = None,
    par: Optional[Mapping[str, Sequence[tuple]]] = None,
    threads: int = 1,
    echo: Optional[Callable[[str], None]] = None,
    symbol_ids: Optional[Mapping[str, str]] = None,
) -> CohortScan:
    """One pass over the annotated VCF. See the module docstring.

    ``par``: the pseudo-autosomal regions of the build (stats.PAR_GRCH38 by
    default), for telling chrX SNPs that infer sex from PAR ones.

    ``threads`` > 1 and a tabix-indexed VCF: the contigs are read
    in parallel, one ``tabix VCF CONTIG`` per worker process, and the parts
    merged in the VCF's contig order (``merge_scans``). The 40-proband genome
    (22.5 GB, regulatory tier on, nothing pre-filtered) took 2 h 46 min on one
    core. Without an index, or with one thread, it is the single pass it was.
    ``echo`` reports progress: each contig as it finishes, or every
    ``PROGRESS_EVERY`` records of a single pass.
    """
    from paralome.burden.stats import PAR_GRCH38

    par = PAR_GRCH38 if par is None else par
    header = read_vcf_header(vcf)
    line_words = () if regulatory else (CODING_LINE_WORDS if words is None else tuple(words))
    say = echo or (lambda _text: None)
    threads = max(1, int(threads or 1))
    contigs = vcf_contigs(Path(vcf)) if threads > 1 else None
    if threads > 1 and not contigs:
        say(f"gene_burden: {vcf} has no tabix index (or tabix is not on PATH); reading it in one pass")
    if contigs:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        workers = min(threads, len(contigs))
        say(f"gene_burden: reading {len(contigs)} contigs with {workers} processes")
        done: dict[str, CohortScan] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_scan_contig, str(vcf), contig, tuple(header.samples), tuple(probands), regulatory,
                            filters, tuple(line_words), {k: list(v) for k, v in par.items()},
                            dict(symbol_ids or {}))
                for contig in contigs
            ]
            for future in as_completed(futures):
                contig, part = future.result()
                done[contig] = part
                if part.counts["records_read"] or len(done) == len(contigs):
                    say(f"gene_burden: {contig} read ({part.counts['records_read']:,} records); "
                        f"{len(done)} of {len(contigs)} contigs done")
        scan = merge_scans([done[contig] for contig in contigs], probands)
        scan.source = {"method": f"tabix by contig, {workers} processes" + ("|grep" if line_words else ""),
                       "line_filter": list(line_words)}
        return scan
    source = LineSource(method="")
    scan = _scan_lines(
        iter_record_lines(vcf, words=line_words, source=source), header.samples, probands,
        regulatory=regulatory, filters=filters, par=par,
        progress=(lambda n: say(f"gene_burden: {n:,} records read")) if echo else None,
        symbol_ids=symbol_ids,
    )
    scan.source = {"method": source.method, "line_filter": list(source.words)}
    return scan
