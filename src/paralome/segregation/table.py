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

import sys
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from paralome.csvio import skip_comment_lines
from paralome.population_af import INFO_AF_KEYS, is_missing, read_population_af

from .genotypes import VariantRecord, extract_gene, iter_vcf_records
from .scorer import SampleRole, check_group, project_to_allele, score_variant


BUCKETS = [
    "segregation_pass",
    "segregation_partial",
    "segregation_fail",
    "segregation_uninformative",
]

BUCKET_FILENAMES = {
    "segregation_pass": "segregation_pass.tsv",
    "segregation_partial": "segregation_partial.tsv",
    "segregation_fail": "segregation_fail.tsv",
    "segregation_uninformative": "segregation_uninformative.tsv",
}

HEADER = [
    "chrom",
    "pos",
    "ref",
    "alt",
    # The one ALT allele whose consequence, gene, ClinVar and
    # frequency the row reports (see choose_reported_allele). A table written
    # before this column existed is still read: without it, readers take the
    # largest frequency of every allele, as they did.
    "reported_alt",
    "gene",
    "variant_id",
    "vcf_id",
    "qual",
    "filter",
    "source_vcf",
    "source_vcf_record",
    "source_stage",
    "source_bucket",
    "annovar_gene",
    "annovar_func",
    "annovar_exonic_func",
    "annovar_aa_change",
    "snpeff_ann",
    "snpeff_impact",
    "snpeff_effect",
    "snpeff_gene",
    "clinvar_significance",
    "gnomad_af",
    "gnomad_af_grpmax",
    "max_population_af",
    "population_af_source",
    "lof_annotation",
    "nmd_annotation",
    "affected_alt_count",
    "affected_total",
    "affected_ref_count",
    "unaffected_alt_count",
    "unaffected_total",
    "unaffected_ref_count",
    "missing_count",
    "informative_sample_count",
    "proband_gt",
    "mother_gt",
    "father_gt",
    "sibling_case_count",
    "sibling_control_count",
    "cousin_case_count",
    "cousin_control_count",
    "segregation_bucket",
    "segregation_model",
    "segregation_class",
    "segregation_score",
    "compatible_ad",
    "compatible_ar",
    "compatible_xlinked",
    "compatible_denovo",
    "segregation_reasons",
    # What the filter profiles' site and genotype thresholds read
    # (filters/quality.py evaluate_quality). The table had none of them, so
    # every min_dp, min_gq, min_alt_reads, min_site_depth, min_site_alt_reads
    # and max_missing_rate was skipped, while filter_summary.json listed them
    # as applied. Over the sheet's samples; '.' where the VCF does not say.
    "site_depth",
    "site_alt_reads",
    "missing_rate",
    # How the sheet was scored and each deciding genotype's own numbers, so each threshold is tested per
    # genotype (see genotype_quality_columns): one unrelated person's shallow reference call must not fail
    # every carrier's variant, and one weak carrier must not sink a strong one. A table without these
    # columns is refused by filters.table.require_quality_columns, so it is written again.
    "segregation_scoring",
    "genotype_evidence",
]

#: What this module writes. Raised whenever a row's content
#: changes for the same VCF and sheet, so a table written under an older
#: method is never reused (segregation/runner.py run_segregation_stage,
#: pipeline/state.py summarize_segregation). v3: the relationship column read
#: under every accepted spelling, a blank family_id read as the sample's own
#: family, per-genotype quality evidence (GQ worked out from GL/PL where the
#: caller wrote none) and the scoring layout on every row.
#: v4: a multi-allelic record scored for its reported allele only,
#: compatible_xlinked only off the PARs of chrX (carrier mother allowed).
#: v5: on a multi-allelic record, alt reads (per genotype and
#: site) are the reported allele's own AO/AD entry; '.' entries keep their place.
#: v6: unrelated / pooled_probands scoring reads no single proband
#: (proband_gt '.', compatible_ar from every affected sample).
SEGREGATION_METHOD = "paralome.segregation.table.v6"

#: How a sheet was scored, written on every row (segregation_scoring).
#: pedigree -- one family, its relatives' calls decide the class;
#: unrelated -- independent samples, no relative decides another's class;
#: pooled_probands -- several pedigrees cut to one affected sample each
#: (see _one_proband_per_pedigree), no family evidence left.
SCORING_PEDIGREE = "pedigree"
SCORING_UNRELATED = "unrelated"
SCORING_POOLED = "pooled_probands"
SCORING_LAYOUTS = (SCORING_PEDIGREE, SCORING_UNRELATED, SCORING_POOLED)

#: Relationships whose genotype the scorer reads as a relative's.
RELATIVE_RELATIONSHIPS = frozenset({"mother", "father", "sibling", "cousin", "second-cousin"})


def missing_table_columns(path: Path) -> list[str]:
    """The columns of HEADER a written segregation table lacks ([] when it has them all).

    How a table written by an earlier Paralome is told apart from a
    current one (pipeline state, and the runner before it keeps a table). A
    file that cannot be read has nothing to compare and gives [].
    """
    try:
        with Path(path).open(newline="") as handle:
            first = handle.readline()
    except OSError:
        return []
    present = {name.strip() for name in first.rstrip("\r\n").lstrip("#").split("\t")}
    return [name for name in HEADER if name not in present]


#: The one list, in paralome.population_af (filtering and reporting
#: read the columns written from it through the same module).
POPULATION_AF_KEYS = INFO_AF_KEYS


def _info_value(info: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = info.get(key)
        if value not in (None, "", "."):
            return str(value)
    return "."


#: ANNOVAR -vcfinput's per-allele block: from ANNOVAR_DATE to the flag ALLELE_END.
_BLOCK_START = "ANNOVAR_DATE"
_BLOCK_END = "ALLELE_END"


def parse_info(info_text: str) -> tuple[dict[str, str], list[dict[str, str]], dict[str, str]]:
    """(the INFO as one dict, each ALT allele's ANNOVAR block in ALT order,
    the keys outside every block).

    One pass over the INFO text. The dict is what VariantRecord.info_dict
    gives (a later key wins, so a multi-allelic record's dict holds its LAST
    allele's block); the blocks are what burden.cohort.split_annovar_blocks
    gives. Splitting the INFO once rather than three times matters on a
    whole-genome VCF. A record without ANNOVAR blocks gives the whole INFO as
    its one block.
    """
    info: dict[str, str] = {}
    blocks: list[dict[str, str]] = []
    site: dict[str, str] = {}
    if not info_text or info_text == ".":
        return info, [info], site
    current: Optional[dict[str, str]] = None
    for item in info_text.split(";"):
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep:
            value = "true"
        info[key] = value
        if key == _BLOCK_START:
            current = {key: value}
            blocks.append(current)
        elif key == _BLOCK_END:
            current = None
        elif current is not None:
            current[key] = value
        else:
            site[key] = value
    return info, (blocks or [info]), site


#: How damaging an ANNOVAR consequence is, for the last tie-break
#: between alleles carried equally often and equally rare. Higher is more.
_LOSS_OF_FUNCTION_TERMS = ("stopgain", "stop_gain", "stoploss", "stop_loss", "startloss", "start_loss",
                           "frameshift")
_ALLELE_AF_KEYS = INFO_AF_KEYS


def _damage_rank(block: dict[str, str]) -> int:
    exonic = " ".join(str(block.get(key) or "") for key in (
        "ExonicFunc.refGeneWithVer", "ExonicFunc.refGene", "ExonicFunc.ensGene", "ExonicFunc")).lower()
    func = " ".join(str(block.get(key) or "") for key in (
        "Func.refGeneWithVer", "Func.refGene", "Func.ensGene", "Func")).lower()
    if "nonframeshift" in exonic:
        return 4
    if any(term in exonic for term in _LOSS_OF_FUNCTION_TERMS):
        return 5
    if "splicing" in func.replace("ncrna_splicing", ""):
        return 5
    if "nonsynonymous" in exonic:
        return 3
    if any(term in func for term in ("utr5", "utr3", "ncrna_exonic", "ncrna_splicing", "upstream")):
        return 2
    if exonic.strip(". ") or func.strip(". "):
        return 1
    return 0


def _allele_view(blocks: list[dict[str, str]], index: int, n_alts: int) -> dict[str, str]:
    """The annotation of ALT allele ``index``: its own ANNOVAR block, or, where
    the record has one block for all of them (snpEff, or ANNOVAR without
    -vcfinput), that block with every per-allele list ('0.3,0.001') cut to
    this allele's value."""
    if len(blocks) == n_alts:
        return blocks[index]
    block = blocks[0] if blocks else {}
    if n_alts <= 1:
        return block
    view: dict[str, str] = {}
    for key, value in block.items():
        tokens = str(value).split(",")
        view[key] = tokens[index].strip() if len(tokens) == n_alts else value
    return view


def _carriers_per_allele(record: VariantRecord, roles: Optional[list[SampleRole]], n_alts: int) -> list[int]:
    """How many samples carry each ALT allele: the affected samples when the
    sheet names any, otherwise every sample in the VCF."""
    sample_ids: list[str]
    affected = [role.sample_id for role in (roles or ()) if role.is_affected]
    sample_ids = affected if affected else list(record.calls)
    counts = [0] * n_alts
    for sample_id in sample_ids:
        call = record.calls.get(sample_id)
        if call is None or not call.has_alt:
            continue
        seen = set()
        for allele in call.alleles:
            if allele in seen:
                continue
            seen.add(allele)
            try:
                number = int(allele)
            except ValueError:
                continue
            if 1 <= number <= n_alts:
                counts[number - 1] += 1
    return counts


def choose_reported_allele(
    record: VariantRecord,
    blocks: list[dict[str, str]],
    roles: Optional[list[SampleRole]] = None,
) -> int:
    """The ALT allele (0-based index) one row of a multi-allelic record reports.

    A record is one row, and a row has one consequence, one gene and one
    frequency, which must belong to the SAME allele (at an A>G,T site with a
    common missense G, AF 0.35, and a rare stop-gain T, AF 2e-5, mixing them
    would read "stopgain, 0.35, common"). The allele chosen is:

    1. the one carried by the most affected samples (every sample when the
       sheet names no affected one);
    2. on a tie, the rarest (a frequency absent from gnomAD counts as rarest);
    3. then the most damaging consequence;
    4. then the first in ALT order.

    A '*' or symbolic ALT is chosen only when every ALT is one.
    """
    alts = [alt.strip() for alt in str(record.alt).split(",")]
    n_alts = len(alts)
    if n_alts <= 1:
        return 0
    carriers = _carriers_per_allele(record, roles, n_alts)
    best_index = 0
    best_key: Optional[tuple] = None
    for index in range(n_alts):
        view = _allele_view(blocks, index, n_alts)
        found = read_population_af(view, _ALLELE_AF_KEYS, source_column=None)
        frequency = found.af if found.af is not None else 0.0
        # A spanning deletion ('*') or a symbolic ALT ('<NON_REF>') is not a variant of its own here, and
        # ANNOVAR writes no block for it, so it is reported only when nothing else is there.
        real = not (alts[index] == "*" or alts[index].startswith("<"))
        key = (real, carriers[index], -frequency, _damage_rank(view), -index)
        if best_key is None or key > best_key:
            best_key, best_index = key, index
    return best_index


def _allele_values(blocks: list[dict[str, str]], *keys: str) -> str:
    """Per ALT allele, the first of ``keys`` that has a value;
    joined in ALT order ('0.3,.') for a multi-allelic record, '.' when no
    allele has one."""
    values = []
    for block in blocks:
        value = "."
        for key in keys:
            if not is_missing(block.get(key)):
                value = str(block.get(key)).strip()
                break
        values.append(value)
    if all(value == "." for value in values):
        return "."
    return ",".join(values)


def _parse_snpeff_ann(record: VariantRecord, info: dict[str, str], prefer: Optional[str] = None) -> dict[str, str]:
    ann = info.get("ANN") or "."
    if ann in ("", "."):
        return {"ann": ".", "impact": ".", "effect": ".", "gene": "."}

    priority = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
    best_fields: list[str] | None = None
    best_score = -1
    requested_alleles = {a.strip() for a in str(record.alt).split(",") if a.strip()}

    for entry in ann.split(","):
        fields = entry.split("|")
        if len(fields) < 4:
            continue
        allele = fields[0].strip()
        impact = fields[2].strip().upper()
        score = priority.get(impact, 0)
        if requested_alleles and allele in requested_alleles:
            score += 10
        # The reported allele's own effect before any other allele's.
        if prefer and allele == prefer:
            score += 20
        if score > best_score:
            best_score = score
            best_fields = fields

    if not best_fields:
        return {"ann": ann, "impact": ".", "effect": ".", "gene": "."}
    return {
        "ann": ann,
        "effect": best_fields[1].strip() or ".",
        "impact": best_fields[2].strip().upper() or ".",
        "gene": best_fields[3].strip() or ".",
    }


_GENE_KEYS = ("Gene.refGeneWithVer", "Gene.refGene", "Gene.ensGene", "Gene")


def _allele_value(view: dict[str, str], site: dict[str, str], *keys: str) -> str:
    """The first of ``keys`` with a value in the allele's annotation, else in
    the keys outside the ANNOVAR blocks (a caller's own INFO). Never another
    allele's block."""
    value = _info_value(view, *keys)
    return value if value != "." else _info_value(site, *keys)


def normalized_annotation(record: VariantRecord, roles: Optional[list[SampleRole]] = None,
                          *, parsed: Optional[tuple] = None,
                          reported_index: Optional[int] = None) -> dict[str, str]:
    """The annotation columns of one record's row, all of ONE allele.

    The allele is choose_reported_allele's, written as ``reported_alt``; the
    consequence, gene, ClinVar and the frequency (``max_population_af`` and
    its source) are that allele's. ``gnomad_af`` and ``gnomad_af_grpmax``
    keep every allele's value in ALT order ('0.35,2e-05'); a reader picks the
    reported allele's out of them by ``reported_alt`` (paralome.population_af).
    """
    info, blocks, site = parsed if parsed is not None else parse_info(record.info)
    alts = [alt.strip() for alt in str(record.alt).split(",")]
    index = choose_reported_allele(record, blocks, roles) if reported_index is None else reported_index
    view = _allele_view(blocks, index, len(alts)) if len(alts) > 1 else (blocks[0] if len(blocks) == 1 else info)
    snpeff = _parse_snpeff_ann(record, info, prefer=alts[index] if len(alts) > 1 else None)
    found = read_population_af(view, _ALLELE_AF_KEYS, source_column=None)
    gnomad_af = _allele_values(blocks, "gnomad41_exome_AF", "gnomad41_genome_AF", "gnomAD_exome_AF", "gnomAD_genome_AF", "gnomAD_AF")
    gnomad_grpmax = _allele_values(blocks, "gnomad41_exome_AF_grpmax", "gnomad41_genome_AF_grpmax", "AF_popmax")
    return {
        "reported_alt": alts[index] if alts and alts[index] else ".",
        "annovar_gene": _allele_value(view, site, *_GENE_KEYS),
        "annovar_func": _allele_value(view, site, "Func.refGeneWithVer", "Func.refGene", "Func.ensGene", "Func"),
        "annovar_exonic_func": _allele_value(view, site, "ExonicFunc.refGeneWithVer", "ExonicFunc.refGene", "ExonicFunc.ensGene", "ExonicFunc"),
        "annovar_aa_change": _allele_value(view, site, "AAChange.refGeneWithVer", "AAChange.refGene", "AAChange.ensGene", "AAChange"),
        "snpeff_ann": snpeff["ann"],
        "snpeff_impact": snpeff["impact"],
        "snpeff_effect": snpeff["effect"],
        "snpeff_gene": snpeff["gene"],
        "clinvar_significance": _allele_value(view, site, "CLNSIG", "ClinVar", "clinvar", "clinvar_clnsig"),
        "gnomad_af": gnomad_af,
        "gnomad_af_grpmax": gnomad_grpmax,
        "max_population_af": "." if found.af is None else f"{found.af:.8g}",
        "population_af_source": found.source if found.af is not None else ".",
        "lof_annotation": _info_value(info, "LOF"),
        "nmd_annotation": _info_value(info, "NMD"),
    }


@dataclass(frozen=True)
class ScoredRoles:
    """The samples segregation scores, how, and who was left out (with why)."""

    roles: list[SampleRole]
    scoring: str
    left_out: tuple[tuple[str, str], ...] = ()


#: The group columns, in the order a row's first non-empty one is taken (matched without regard to case).
_GROUP_COLUMNS = ("affected", "group", "status")


def _read_sheet(path: Path) -> tuple[list[SampleRole], Optional[dict[str, list[str]]]]:
    """The sheet's roles in sheet order, and its families (None: no family column)."""
    # The columns are found as the rest of the pipeline finds them (fastq.manifest's aliases, any case:
    # 'relation', 'Relationship', ...), so the proband, mother and father are found in the user's own
    # sheet. Values go through the sheet normaliser's table too (brother/sister are siblings).
    from paralome.fastq.manifest import FAMILY_ALIASES, RELATIONSHIP_ALIASES, SAMPLE_ALIASES
    from paralome.sample_manifest import RELATIONSHIP_NORMALIZATION

    roles: list[SampleRole] = []
    # Utf-8-sig drops the byte-order mark Excel writes; with it left
    # in, the first column was not called sample_id and the sheet was refused.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        # Skip '#' lines, as every other sample-sheet reader does. A
        # commented-out sample was read as a family member called "#NAME" with
        # no genotype in the VCF: it raised the missing count on every variant
        # and, placed above the real proband, was taken for the proband.
        reader = csv.DictReader(skip_comment_lines(fh))
        by_lower: dict[str, str] = {}
        for name in reader.fieldnames or []:
            if isinstance(name, str):
                by_lower.setdefault(name.strip().lower(), name)

        def column(aliases: tuple[str, ...]) -> Optional[str]:
            return next((by_lower[alias] for alias in aliases if alias in by_lower), None)

        sample_column = column(tuple(SAMPLE_ALIASES))
        group_columns = [by_lower[name] for name in _GROUP_COLUMNS if name in by_lower]
        relationship_column = column(tuple(RELATIONSHIP_ALIASES))
        family_column = column(tuple(FAMILY_ALIASES))
        missing = []
        if sample_column is None:
            missing.append("sample_id")
        if not group_columns:
            missing.append("affected/group/status")
        if missing:
            raise ValueError(f"Sample manifest missing required columns: {', '.join(sorted(missing))}")
        families: Optional[dict[str, list[str]]] = {} if family_column else None
        for row in reader:
            sample_id = (row.get(sample_column) or "").strip()
            if not sample_id:
                continue
            if families is not None:
                # A blank family cell is the sample's own family, as gene burden reads it
                # (burden.cohort.select_probands), not one shared family ''.
                family = (row.get(family_column) or "").strip()
                if family in ("", "."):
                    family = sample_id
                families.setdefault(family, []).append(sample_id)
            group = next(
                (value for value in ((row.get(name) or "").strip() for name in group_columns) if value),
                "unknown",
            ).lower()
            # Refused unless case, control or unknown (PED digits included).
            group = check_group(group, sample_id)
            relationship = (row.get(relationship_column) or "").strip().lower().replace("_", "-") \
                if relationship_column else ""
            if relationship in ("", "."):
                relationship = "unknown"
            relationship = RELATIONSHIP_NORMALIZATION.get(relationship, relationship)
            roles.append(SampleRole(sample_id=sample_id, group=group, relationship=relationship))
    return roles, families


def read_sample_roles(path: Path, *, one_family: bool = False) -> list[SampleRole]:
    """Every sample of the sheet; with ``one_family``, the samples segregation scores."""
    if one_family:
        return read_scored_roles(path).roles
    roles, _families = _read_sheet(path)
    return roles


def read_scored_roles(path: Path) -> ScoredRoles:
    """The samples segregation scores and the layout they are scored as.

    Several families of which one has relatives: one affected sample per
    family (pooled_probands, see _one_proband_per_pedigree). Otherwise every
    sample: a pedigree when the sheet is one family of two or more, or names a
    relative (mother, father, sibling, cousin); unrelated when it is not.
    """
    roles, families = _read_sheet(path)
    if families is not None and len(families) >= 2 and any(len(ids) >= 2 for ids in families.values()):
        return _one_proband_per_pedigree(path, families, roles)
    one_family = families is not None and len(families) == 1 and len(roles) >= 2
    named_relatives = any(role.relationship in RELATIVE_RELATIONSHIPS for role in roles)
    return ScoredRoles(roles=roles, scoring=SCORING_PEDIGREE if (one_family or named_relatives) else SCORING_UNRELATED)


def scoring_of(roles: list[SampleRole]) -> str:
    """The layout of a list of roles given without its sheet (older callers): a pedigree when it names a relative."""
    return SCORING_PEDIGREE if any(role.relationship in RELATIVE_RELATIONSHIPS for role in roles) else SCORING_UNRELATED


def _one_proband_per_pedigree(
    path: Path, families: dict[str, list[str]], roles: list[SampleRole],
) -> ScoredRoles:
    """The roles segregation scores when the sheet holds several pedigrees.

    The scorer pools every affected and every unaffected sample, and takes the
    first proband, mother and father listed, so several pedigrees in one sheet
    cannot be scored as families together. Each family gives its proband
    (else its first affected sample) and its relatives are left out, so the
    families are scored as unrelated probands, without the family evidence,
    and the run goes on to the burden test the sheet was made for. For family
    segregation, run each family as its own project (`paralome batch run`).

    A family of one with no affected sample (an unrelated control) is kept:
    it is no proband's relative. Every sample left out is named, with the
    reason, on the console and in the segregation summary. The layout is
    recorded on every row (segregation_scoring = pooled_probands): a family
    profile cannot be applied to such a table, and filtering says so instead
    of silently keeping only the variants shared by half the probands
    (filters.table).
    """
    by_id = {role.sample_id: role for role in roles}
    kept: list[SampleRole] = []
    left_out: list[tuple[str, str]] = []
    for family, ids in families.items():
        family_roles = [by_id[i] for i in ids if i in by_id]
        chosen = (next((r for r in family_roles if r.relationship == "proband" and r.is_affected), None)
                  or next((r for r in family_roles if r.is_affected), None))
        if chosen is not None:
            kept.append(SampleRole(sample_id=chosen.sample_id, group=chosen.group, relationship="proband"))
            left_out.extend((r.sample_id, f"a relative of {chosen.sample_id} (family {family})")
                            for r in family_roles if r is not chosen)
        elif len(family_roles) == 1:
            only = family_roles[0]
            kept.append(SampleRole(sample_id=only.sample_id, group=only.group, relationship="unrelated"))
        else:
            left_out.extend((r.sample_id, f"family {family} has no affected member to score") for r in family_roles)
    pedigrees = sum(1 for ids in families.values() if len(ids) >= 2)
    named = "; ".join(f"{sample} ({reason})" for sample, reason in left_out) or "none"
    print(
        f"[paralome] {path} holds {len(families)} families, {pedigrees} with relatives. Segregation scores "
        f"one family's pedigree at a time, so these are scored as unrelated probands (pooled_probands): one "
        f"affected sample per family, {len(kept)} samples in all. Left out: {named}. A family filter profile "
        "cannot be applied to this table: filter it with --segregation-profile unrelated-probands, or for family "
        "segregation run each family as its own project (paralome batch run).",
        file=sys.stderr, flush=True,
    )
    return ScoredRoles(roles=kept, scoring=SCORING_POOLED, left_out=tuple(left_out))


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _ints(text: Optional[str]) -> list[int]:
    out: list[int] = []
    for piece in str(text or "").split(","):
        try:
            out.append(int(float(piece)))
        except ValueError:
            continue
    return out


def _positional_ints(text: Optional[str]) -> list[Optional[int]]:
    """A per-allele list (AO, AD) with '.' kept in place as None;
    _ints dropped it, so AD '10,3,.,4' read allele 3 from the wrong slot."""
    out: list[Optional[int]] = []
    for piece in str(text or "").split(","):
        try:
            out.append(int(float(piece)))
        except ValueError:
            out.append(None)
    return out if any(v is not None for v in out) else []


def _floats(text: Optional[str]) -> Optional[list[float]]:
    """Every value of a per-genotype list, or None when one is missing or unreadable."""
    out: list[float] = []
    for piece in str(text or "").split(","):
        try:
            value = float(piece)
        except ValueError:
            return None
        if math.isnan(value):
            return None
        out.append(value)
    return out or None


def _genotype_index(alleles: tuple[str, ...]) -> Optional[int]:
    """The VCF likelihood index of a called genotype (haploid or diploid), else None."""
    try:
        numbers = sorted(int(allele) for allele in alleles)
    except ValueError:
        return None
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        j, k = numbers
        return k * (k + 1) // 2 + j
    return None


#: The largest GQ written, as callers cap it.
_GQ_CAP = 99


def genotype_quality(values: dict[str, str], alleles: tuple[str, ...]) -> Optional[int]:
    """The call's GQ: FORMAT GQ, else worked out from PL or GL.

    FreeBayes writes no GQ unless it is given --genotype-qualities, which the
    pipeline does not pass; its FORMAT is GT:DP:AD:RO:QR:AO:QA:GL. Every
    min_gq of the filter profiles was therefore skipped on pipeline output
    while filter_summary.json listed it. Where the caller wrote likelihoods
    and no GQ, GQ is what GATK and bcftools define it as: the phred-scaled
    likelihood of the next most likely genotype less that of the called one
    (PL = -10 x GL), at least 0 and at most 99. None when the call has neither.
    """
    gq = _ints(values.get("GQ"))
    if gq:
        return gq[0]
    phred = _floats(values.get("PL"))
    if phred is None:
        log10 = _floats(values.get("GL"))
        phred = [-10.0 * value for value in log10] if log10 is not None else None
    index = _genotype_index(alleles)
    if phred is None or index is None or index >= len(phred) or len(phred) < 2:
        return None
    others = min(value for i, value in enumerate(phred) if i != index)
    return int(max(0, min(_GQ_CAP, round(others - phred[index]))))


def _allele_reads(values: dict[str, str], alleles: tuple[str, ...],
                  reported: Optional[int] = None) -> tuple[Optional[int], Optional[int]]:
    """(reads for the ALT alleles the call carries, reads for REF): FORMAT AO/RO, else AD.

    ``reported`` (allele number, multi-allelic records) counts that
    allele's reads only, not every ALT the call carries; a missing or '.'
    entry gives None (not tested as passed) instead of the sum of every ALT.
    """
    carried = [reported] if reported is not None else sorted({int(a) for a in alleles if a.isdigit() and int(a) > 0})

    def pick(reads: list[Optional[int]], offset: int) -> Optional[int]:
        picked = [reads[a - offset] if 0 <= a - offset < len(reads) else None for a in carried]
        return None if any(v is None for v in picked) else sum(picked)  # type: ignore[arg-type]

    ao = _positional_ints(values.get("AO"))
    ro = _positional_ints(values.get("RO"))
    if ao or ro:
        return (pick(ao, 1) if ao else None), (ro[0] if ro else None)
    ad = _positional_ints(values.get("AD"))
    if len(ad) > 1:
        return pick(ad, 0), ad[0]
    return None, None


def _evidence_value(value: Optional[int]) -> str:
    return "." if value is None else str(value)


def genotype_quality_columns(
    record: VariantRecord, roles: list[SampleRole], info: dict[str, str], scoring: Optional[str] = None,
    allele_index: Optional[int] = None,
) -> list[str]:
    """site_depth, site_alt_reads, missing_rate, segregation_scoring, genotype_evidence.

    site_depth is INFO DP and site_alt_reads the sum of INFO AO; missing_rate
    the share of the scored samples not called.

    genotype_evidence holds the genotypes the profile's genotype thresholds
    are tested on, one ``sample:tag:GT:DP:GQ:ALT_READS:REF_READS`` per
    genotype, ';' between them ('.' for a number the VCF does not give). Tag
    A, U or C is an ALT carrier (affected, unaffected, status unknown); tag R
    is a called non-carrier whose call decides the class -- every relative in
    a pedigree (a parent's reference call on two reads is what makes a false
    de novo), nobody when the samples are unrelated (filters.quality.evaluate_quality).
    """
    scoring = scoring or scoring_of(roles)
    keys = record.fmt.split(":") if record.fmt else []
    site_depth = _ints(info.get("DP"))
    site_alt = _ints(info.get("AO"))
    # A multi-allelic row's alt reads are its reported allele's (allele_index, 0-based).
    reported = int(allele_index) + 1 if allele_index is not None and "," in str(record.alt) else None
    if reported is not None:
        per_allele = _positional_ints(info.get("AO"))
        value = per_allele[reported - 1] if reported - 1 < len(per_allele) else None
        site_alt = [value] if value is not None else []
    called = missing = 0
    evidence: list[str] = []
    for role in roles:
        call = record.calls.get(role.sample_id)
        if call is None or not call.is_called:
            missing += 1
            continue
        called += 1
        if call.has_alt:
            tag = "A" if role.is_affected else ("U" if role.is_unaffected else "C")
        elif scoring == SCORING_PEDIGREE:
            tag = "R"
        else:
            continue
        values = dict(zip(keys, str(call.raw).split(":")))
        dp = _ints(values.get("DP"))
        alt_reads, ref_reads = _allele_reads(values, call.alleles, reported)
        evidence.append(":".join((
            role.sample_id, tag, call.gt.replace(":", "/"),
            _evidence_value(dp[0] if dp else None),
            _evidence_value(genotype_quality(values, call.alleles)),
            _evidence_value(alt_reads), _evidence_value(ref_reads),
        )))
    total = called + missing
    return [
        str(site_depth[0]) if site_depth else ".",
        str(sum(site_alt)) if site_alt else ".",
        f"{missing / total:.4f}" if total else ".",
        scoring,
        ";".join(evidence) if evidence else ".",
    ]


def record_to_row(record: VariantRecord, roles: list[SampleRole], *, source_vcf: Path,
                  scoring: Optional[str] = None) -> list[str]:
    # The INFO is parsed once for the whole row.
    parsed = parse_info(record.info)
    # Scored for the allele the row reports (see project_to_allele).
    reported_index = choose_reported_allele(record, parsed[1], roles)
    scoring = scoring or scoring_of(roles)
    result = score_variant(record, roles, allele_index=reported_index, scoring=scoring)
    annotation = normalized_annotation(record, roles, parsed=parsed, reported_index=reported_index)
    source_bucket = result.segregation_bucket if result.segregation_bucket in BUCKETS else "segregation_uninformative"
    gene = extract_gene(record, info=parsed[0])
    if "ANN" not in parsed[0] and annotation["annovar_gene"] != ".":
        # The reported allele's gene, not the last allele's.
        gene = annotation["annovar_gene"]
    return [
        record.chrom,
        str(record.pos),
        record.ref,
        record.alt,
        annotation["reported_alt"],
        gene,
        record.key,
        record.variant_id,
        record.qual,
        record.filt,
        str(source_vcf),
        record.key,
        "segregation",
        source_bucket,
        annotation["annovar_gene"],
        annotation["annovar_func"],
        annotation["annovar_exonic_func"],
        annotation["annovar_aa_change"],
        annotation["snpeff_ann"],
        annotation["snpeff_impact"],
        annotation["snpeff_effect"],
        annotation["snpeff_gene"],
        annotation["clinvar_significance"],
        annotation["gnomad_af"],
        annotation["gnomad_af_grpmax"],
        annotation["max_population_af"],
        annotation["population_af_source"],
        annotation["lof_annotation"],
        annotation["nmd_annotation"],
        str(result.affected_alt_count),
        str(result.affected_called),
        str(result.affected_ref_count),
        str(result.unaffected_alt_count),
        str(result.unaffected_called),
        str(result.unaffected_ref_count),
        str(result.missing_count),
        str(result.informative_sample_count),
        result.proband_gt,
        result.mother_gt,
        result.father_gt,
        str(result.sibling_case_count),
        str(result.sibling_control_count),
        str(result.cousin_case_count),
        str(result.cousin_control_count),
        result.segregation_bucket,
        result.segregation_model,
        result.segregation_class,
        f"{result.segregation_score:.4f}",
        _bool(result.compatible_ad),
        _bool(result.compatible_ar),
        _bool(result.compatible_xlinked),
        _bool(result.compatible_denovo),
        result.notes,
        *genotype_quality_columns(project_to_allele(record, reported_index), roles, parsed[0], scoring,
                                  reported_index),
    ]


def scored_record(scored: ScoredRoles) -> dict[str, Any]:
    """The summary's account of who was scored, as what, and who was left out."""
    return {
        "scoring": scored.scoring,
        "samples_scored": [role.sample_id for role in scored.roles],
        "samples_left_out": [{"sample_id": sample, "reason": reason} for sample, reason in scored.left_out],
    }


def _empty_bucket_counts() -> dict[str, int]:
    return {bucket: 0 for bucket in BUCKETS}


def _write_summary(
    *,
    summary_json: Path,
    input_vcf: Path,
    samples: Path,
    all_tsv: Path,
    bucket_paths: dict[str, Path],
    total_rows: int,
    bucket_counts: dict[str, int],
    class_counts: Counter[str],
    scored: Optional[ScoredRoles] = None,
) -> dict[str, Any]:
    bucket_sum = sum(bucket_counts.get(bucket, 0) for bucket in BUCKETS)
    lossless = total_rows == bucket_sum
    summary = {
        "schema": "paralome.segregation.summary.v1",
        "stage": "segregation",
        "input_vcf": str(input_vcf),
        "samples": str(samples),
        "outputs": {
            "all_variants": str(all_tsv),
            **{bucket: str(path) for bucket, path in bucket_paths.items()},
        },
        "rows": total_rows,
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS},
        "class_counts": {key: int(value) for key, value in sorted(class_counts.items())},
        "bucket_sum": int(bucket_sum),
        "lossless": bool(lossless),
        # What wrote the rows and how the sheet was scored. The
        # stage reuses a table only when its method is this one.
        "method": SEGREGATION_METHOD,
        **(scored_record(scored) if scored is not None else {}),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _require_sheet_samples_in_vcf(roles: list[SampleRole], samples: Path, vcf: Path) -> None:
    """Refuse a sheet sample that is not a VCF sample (PLM-0205).

    Every sheet sample must be a VCF sample. FreeBayes names its
    samples from each BAM's read-group SM; on a sheet of BAMs nothing makes
    that the sheet's sample_id, and a sample not in the VCF was read as
    uncalled on every record -- a carrier parent became "missing" and her
    variants segregation_pass. Gene burden already checks its probands.
    """
    from paralome.burden.cohort import read_vcf_header
    from paralome.errors import ParalomeError

    vcf_samples = list(read_vcf_header(vcf).samples)
    absent = [role.sample_id for role in roles if role.sample_id not in set(vcf_samples)]
    if absent:
        raise ParalomeError(
            205,
            f"sample(s) {', '.join(absent)} in {samples} are not in the VCF {vcf} "
            f"(its samples: {', '.join(vcf_samples[:10])}{' ...' if len(vcf_samples) > 10 else ''})",
            fix="A BAM's read-group SM tag names its VCF sample: make the sheet's sample_id match it, "
                "or comment out the samples that were not called",
        )


def write_segregation_outputs(
    *,
    vcf: Union[str, Path],
    samples: Union[str, Path],
    results_dir: Union[str, Path],
    all_tsv: Union[str, Path] | None = None,
    summary_json: Union[str, Path] | None = None,
    source_vcf: Union[str, Path, None] = None,
    bucket_outputs: Union[dict, None] = None,
) -> dict[str, Any]:
    """Write lossless segregation annotation and one mutually exclusive bucket file per variant.

    Segregation is intentionally genotype/pedigree-only. It does not consume or filter by
    population frequency, functional consequence, ClinVar state, or other biological annotations.
    """
    vcf = Path(vcf)
    samples = Path(samples)
    source_vcf_path = Path(source_vcf) if source_vcf is not None else vcf
    results_dir = Path(results_dir)
    scored = read_scored_roles(samples)
    roles = scored.roles
    _require_sheet_samples_in_vcf(roles, samples, vcf)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_path = Path(all_tsv) if all_tsv else results_dir / "all_variants.segregation.tsv"
    summary_path = Path(summary_json) if summary_json else results_dir / "segregation_summary.json"
    # The caller supplies these so the project ID can be in the name.
    # Deriving them here unconditionally was why every family on disk had a
    # file called segregation_pass.tsv -- unique in its own directory and
    # indistinguishable the moment `collect` flattens the tree.
    bucket_paths = (
        {bucket: Path(path) for bucket, path in bucket_outputs.items()}
        if bucket_outputs
        else {bucket: results_dir / filename for bucket, filename in BUCKET_FILENAMES.items()}
    )

    total = 0
    bucket_counts = _empty_bucket_counts()
    class_counts: Counter[str] = Counter()

    handles: dict[str, Any] = {}
    writers: dict[str, csv.writer] = {}
    try:
        all_path.parent.mkdir(parents=True, exist_ok=True)
        handles["all"] = open(all_path, "w", newline="")
        writers["all"] = csv.writer(handles["all"], delimiter="\t", lineterminator="\n")
        writers["all"].writerow(HEADER)

        for bucket, path in bucket_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[bucket] = open(path, "w", newline="")
            writers[bucket] = csv.writer(handles[bucket], delimiter="\t", lineterminator="\n")
            writers[bucket].writerow(HEADER)

        bucket_idx = HEADER.index("segregation_bucket")
        class_idx = HEADER.index("segregation_class")
        for record in iter_vcf_records(vcf):
            row = record_to_row(record, roles, source_vcf=source_vcf_path, scoring=scored.scoring)
            bucket = row[bucket_idx] if row[bucket_idx] in BUCKETS else "segregation_uninformative"
            writers["all"].writerow(row)
            writers[bucket].writerow(row)
            bucket_counts[bucket] += 1
            class_counts[row[class_idx]] += 1
            total += 1
    finally:
        for fh in handles.values():
            fh.close()

    summary = _write_summary(
        summary_json=summary_path,
        input_vcf=vcf,
        samples=samples,
        all_tsv=all_path,
        bucket_paths=bucket_paths,
        total_rows=total,
        bucket_counts=bucket_counts,
        class_counts=class_counts,
        scored=scored,
    )
    if not summary["lossless"]:
        raise RuntimeError(
            f"Segregation row-preservation validation failed: rows={total} bucket_sum={summary['bucket_sum']}"
        )
    return summary


def write_segregation_table(vcf: Path, samples: Path, out_tsv: Path) -> int:
    """Backward-compatible writer used by older callers.

    New stage code should use write_segregation_outputs() so bucket files and summary JSON
    are generated in the same pass.
    """
    scored = read_scored_roles(samples)
    # The same sample-name check as the stage writer.
    _require_sheet_samples_in_vcf(scored.roles, Path(samples), Path(vcf))
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_tsv, "w", newline="") as out_fh:
        writer = csv.writer(out_fh, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for record in iter_vcf_records(Path(vcf)):
            writer.writerow(record_to_row(record, scored.roles, source_vcf=Path(vcf), scoring=scored.scoring))
            count += 1
    return count


def main(argv: Optional[list[str]] = None) -> int:
    raise SystemExit(
        "paralome.segregation.table is a library module, not a command.\n"
        "\n"
        "it has no caller in the pipeline (the freebayes annotation-merge job does not run it),\n"
        "which ran it inline and produced output named from the VARIANT workdir\n"
        "('freebayes.segregation.tsv'), a third naming convention that no amount\n"
        "of fixing segregation/runner.py could reach.\n"
        "\n"
        "Use the stage, which prefixes its output with the family ID, writes the\n"
        "bucket files and the summary JSON, checkpoints, and now fails\n"
        "loudly rather than writing a bare name:\n"
        "\n"
        # `paralome-worker`, not `paralome`: `segregate` is on the machine-facing binary. See filters/table.py.
        "    paralome-worker segregate run --vcf <vcf> --samples <csv> --out <workdir> --prefix FAM01\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
