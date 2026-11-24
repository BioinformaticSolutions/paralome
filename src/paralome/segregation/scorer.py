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

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from paralome.sample_manifest import AFFECTED_GROUPS, GROUP_ALIASES, UNAFFECTED_GROUPS

from .genotypes import GenotypeCall, VariantRecord

# One vocabulary with sample_manifest (the dedup reader). '1' and '0' are not read as affected /
# unaffected: under PED coding 2 is affected and 1 unaffected, so they are ambiguous. check_group refuses
# them.
CASE_GROUPS = set(AFFECTED_GROUPS) | {k for k, v in GROUP_ALIASES.items() if v == "affected"}
CONTROL_GROUPS = set(UNAFFECTED_GROUPS) | {k for k, v in GROUP_ALIASES.items() if v == "unaffected"}
UNKNOWN_GROUPS = {"unknown", "", ".", "na", "n/a", "missing"}


def check_group(value: str, sample_id: str) -> str:
    """The sheet's affected value, lower-cased; refused when it is none of
    case/control/unknown."""
    group = str(value or "").strip().lower()
    if group in UNKNOWN_GROUPS:
        return "unknown"
    if group in CASE_GROUPS or group in CONTROL_GROUPS:
        return group
    hint = (" Numeric codes are ambiguous (PED uses 2 = affected, 1 = unaffected; "
            "others use 1 = affected, 0 = unaffected): write affected or unaffected."
            if group.lstrip("-").isdigit() else "")
    raise ValueError(
        f"Invalid affected value {value!r} for sample {sample_id}. Use affected/unaffected, "
        f"case/control, yes/no or unknown.{hint}"
    )


PASS_CLASSES = {"affected_only", "shared_affected", "possible_de_novo", "possible_recessive"}
PARTIAL_CLASSES = {"affected_enriched"}
FAIL_CLASSES = {"non_segregating"}


@dataclass(frozen=True)
class SampleRole:
    sample_id: str
    group: str
    relationship: str
    # Worked out once, not twice per sample per record.
    is_affected: bool = field(init=False, repr=False, compare=False)
    is_unaffected: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        group = str(self.group).lower()
        object.__setattr__(self, "is_affected", group in CASE_GROUPS)
        object.__setattr__(self, "is_unaffected", group in CONTROL_GROUPS)


@dataclass(frozen=True)
class SegregationResult:
    affected_alt_count: int
    affected_called: int
    unaffected_alt_count: int
    unaffected_called: int
    affected_ref_count: int
    unaffected_ref_count: int
    missing_count: int
    informative_sample_count: int
    proband_gt: str
    mother_gt: str
    father_gt: str
    sibling_case_count: int
    sibling_control_count: int
    cousin_case_count: int
    cousin_control_count: int
    segregation_class: str
    segregation_bucket: str
    segregation_model: str
    segregation_score: float
    compatible_ad: bool
    compatible_ar: bool
    compatible_xlinked: bool
    compatible_denovo: bool
    notes: str


def bucket_for_class(segregation_class: str) -> str:
    """Map a genotype-derived segregation class to a lossless stage bucket."""
    if segregation_class in PASS_CLASSES:
        return "segregation_pass"
    if segregation_class in PARTIAL_CLASSES:
        return "segregation_partial"
    if segregation_class in FAIL_CLASSES:
        return "segregation_fail"
    return "segregation_uninformative"


def _gt(record: VariantRecord, sample_id: Optional[str]) -> str:
    if not sample_id:
        return "."
    call = record.calls.get(sample_id)
    return call.gt if call else "."


def _first_role(roles: Iterable[SampleRole], relationship: str) -> SampleRole | None:
    for role in roles:
        if role.relationship == relationship:
            return role
    return None


def _count_alt(record: VariantRecord, roles: list[SampleRole]) -> tuple[int, int]:
    alt = 0
    called = 0
    for role in roles:
        call = record.calls.get(role.sample_id)
        if not call or not call.is_called:
            continue
        called += 1
        if call.has_alt:
            alt += 1
    return alt, called


def _missing_count(record: VariantRecord, roles: list[SampleRole]) -> int:
    missing = 0
    for role in roles:
        call = record.calls.get(role.sample_id)
        if not call or not call.is_called:
            missing += 1
    return missing


def _model_from_flags(
    *,
    compatible_denovo: bool,
    compatible_ar: bool,
    compatible_ad: bool,
    compatible_xlinked: bool,
) -> str:
    models: list[str] = []
    if compatible_denovo:
        models.append("de_novo")
    if compatible_ar:
        models.append("autosomal_recessive")
    if compatible_ad:
        models.append("autosomal_dominant")
    if compatible_xlinked:
        models.append("x_linked")
    return ",".join(models) if models else "."


def _project_call(call: GenotypeCall, allele: str) -> GenotypeCall:
    count = sum(1 for a in call.alleles if a == allele) if call.is_called else 0
    return replace(call, has_alt=count > 0, alt_count=count)


def project_to_allele(record: VariantRecord, allele_index: Optional[int]) -> VariantRecord:
    """A multi-allelic record seen for one ALT (0-based index): a
    sample carries it only when its genotype holds that allele number. Pooling
    every ALT made a parent carrying ALT1 an unaffected carrier of ALT2, hiding
    de novo calls. Bi-allelic records come back unchanged."""
    if allele_index is None or "," not in str(record.alt):
        return record
    allele = str(int(allele_index) + 1)
    return replace(record, calls={sid: _project_call(c, allele) for sid, c in record.calls.items()})


#: Pseudoautosomal regions of X, GRCh37 and GRCh38 together (a
#: position in either build's PAR is not treated as X-linked).
_X_PAR = ((10_001, 2_781_479), (154_931_044, 156_030_895))


def _non_par_x(record: VariantRecord) -> bool:
    chrom = str(record.chrom)
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom.upper() not in {"X", "23"}:
        return False
    return not any(start <= int(record.pos) <= end for start, end in _X_PAR)


def score_variant(record: VariantRecord, roles: list[SampleRole],
                  allele_index: Optional[int] = None, scoring: Optional[str] = None) -> SegregationResult:
    record = project_to_allele(record, allele_index)
    # Outside a pedigree (scoring 'unrelated' or 'pooled_probands')
    # there is no single proband: every kept sample may be one, and the first
    # listed decided proband_gt, compatible_ar and possible_recessive.
    single_family = scoring is None or scoring == "pedigree"
    affected = [r for r in roles if r.is_affected]
    unaffected = [r for r in roles if r.is_unaffected]

    affected_alt, affected_called = _count_alt(record, affected)
    unaffected_alt, unaffected_called = _count_alt(record, unaffected)

    affected_ref = max(0, affected_called - affected_alt)
    unaffected_ref = max(0, unaffected_called - unaffected_alt)
    missing = _missing_count(record, roles)
    informative = affected_called + unaffected_called

    proband = _first_role(roles, "proband") if single_family else None
    mother = _first_role(roles, "mother") if single_family else None
    father = _first_role(roles, "father") if single_family else None

    proband_call = record.calls.get(proband.sample_id) if proband else None
    mother_call = record.calls.get(mother.sample_id) if mother else None
    father_call = record.calls.get(father.sample_id) if father else None

    sibling_cases = [r for r in roles if r.relationship == "sibling" and r.is_affected]
    sibling_controls = [r for r in roles if r.relationship == "sibling" and r.is_unaffected]
    cousin_cases = [r for r in roles if r.relationship in {"cousin", "second-cousin"} and r.is_affected]
    cousin_controls = [r for r in roles if r.relationship in {"cousin", "second-cousin"} and r.is_unaffected]

    sibling_case_alt, _ = _count_alt(record, sibling_cases)
    sibling_control_alt, _ = _count_alt(record, sibling_controls)
    cousin_case_alt, _ = _count_alt(record, cousin_cases)
    cousin_control_alt, _ = _count_alt(record, cousin_controls)

    affected_rate = affected_alt / affected_called if affected_called else 0.0
    unaffected_rate = unaffected_alt / unaffected_called if unaffected_called else 0.0
    score = round(affected_rate - unaffected_rate, 4)

    notes: list[str] = []
    if affected_called == 0:
        seg_class = "uninformative"
        notes.append("no_called_affected_samples")
    elif affected_alt == 0:
        seg_class = "non_segregating"
        notes.append("absent_from_called_affected_samples")
    elif affected_alt == affected_called and unaffected_called > 0 and unaffected_alt == 0:
        seg_class = "affected_only"
    elif affected_alt >= 2 and unaffected_alt == 0:
        seg_class = "shared_affected"
    elif affected_rate > unaffected_rate:
        seg_class = "affected_enriched"
    elif unaffected_called > 0 and unaffected_alt > 0 and affected_alt > 0:
        seg_class = "non_segregating"
        notes.append("present_in_unaffected_samples")
    else:
        seg_class = "uninformative"

    compatible_denovo = False
    compatible_ad = False
    compatible_ar = False
    compatible_xlinked = False

    if proband_call and proband_call.has_alt:
        parents_called = bool(mother_call and mother_call.is_called) and bool(father_call and father_call.is_called)
        parents_alt = bool(mother_call and mother_call.has_alt) or bool(father_call and father_call.has_alt)
        compatible_denovo = bool(parents_called and not parents_alt and unaffected_alt == 0)
        if compatible_denovo:
            seg_class = "possible_de_novo"
            notes.append("proband_alt_both_parents_ref")
        if mother_call and mother_call.has_alt and not (father_call and father_call.has_alt):
            notes.append("maternal_shared")
        if father_call and father_call.has_alt and not (mother_call and mother_call.has_alt):
            notes.append("paternal_shared")
        if mother_call and mother_call.has_alt and father_call and father_call.has_alt:
            notes.append("both_parents_alt")

    # RECESSIVE INHERITANCE. The proband and every called affected relative homozygous ALT, no called
    # unaffected relative homozygous ALT, and at least one unaffected relative called (carriers allowed) is
    # possible_recessive, a pass class -- so a homozygous proband with heterozygous parents, the expected
    # recessive pattern, is not counted against by its carrier parents. A sheet with no unaffected relative
    # gives no recessive evidence and keeps the class it had.
    affected_calls = [record.calls.get(r.sample_id) for r in affected]
    affected_calls = [c for c in affected_calls if c and c.is_called]
    unaffected_hom = any(
        (c := record.calls.get(r.sample_id)) is not None and c.is_called and c.is_hom_alt for r in unaffected
    )
    # Without a single proband, every called affected sample decides (as it already did).
    proband_hom = bool(proband_call and proband_call.is_hom_alt) if single_family else True
    recessive = bool(
        proband_hom
        and affected_calls and all(c.is_hom_alt for c in affected_calls)
        and unaffected_called > 0 and not unaffected_hom
    )
    if recessive and seg_class in {"non_segregating", "affected_enriched", "uninformative"}:
        seg_class = "possible_recessive"
        if "present_in_unaffected_samples" in notes:
            notes.remove("present_in_unaffected_samples")
        notes.append("affected_hom_alt_unaffected_not_hom_alt")
        # The score of a recessive pattern is the homozygous rate in affected relatives less that in
        # unaffected ones (affected_rate - unaffected_rate would be 0.0 for a homozygous proband of carrier
        # parents, below the family profiles' min_segregation_score of 0.5).
        unaffected_calls = [c for c in (record.calls.get(r.sample_id) for r in unaffected) if c and c.is_called]
        affected_hom = sum(1 for c in affected_calls if c.is_hom_alt) / len(affected_calls)
        unaffected_hom_rate = (sum(1 for c in unaffected_calls if c.is_hom_alt) / len(unaffected_calls)
                               if unaffected_calls else 0.0)
        score = round(affected_hom - unaffected_hom_rate, 4)

    compatible_ad = bool(affected_alt > 0 and unaffected_alt == 0 and affected_rate >= 0.5)
    if single_family:
        compatible_ar = bool(proband_call and proband_call.is_hom_alt and not unaffected_hom)
    else:
        # Any called affected sample homozygous ALT, none unaffected.
        compatible_ar = bool(any(c.is_hom_alt for c in affected_calls) and not unaffected_hom)
    # X-linked only off the PARs of chrX, and the classic recessive
    # pattern counts: a heterozygous unaffected mother when she is the only
    # unaffected carrier and every affected carrier is hemizygous / hom-alt.
    unaffected_carriers = [r for r in unaffected
                           if (c := record.calls.get(r.sample_id)) is not None and c.is_called and c.has_alt]
    carrier_mother_only = bool(
        unaffected_carriers and mother is not None
        and all(r.sample_id == mother.sample_id for r in unaffected_carriers)
        and mother_call is not None and not mother_call.is_hom_alt
        and all(c.is_hom_alt for c in affected_calls if c.has_alt)
    )
    compatible_xlinked = bool(
        _non_par_x(record) and affected_alt > 0
        and (unaffected_alt == 0 or carrier_mother_only)
        and (father_call is None or not father_call.has_alt)
    )

    if affected_alt >= 2 and affected_called >= 2:
        notes.append("variant_seen_in_multiple_affected_relatives")
    if sibling_case_alt > 0:
        notes.append("affected_sibling_alt")
    if sibling_control_alt > 0:
        notes.append("unaffected_sibling_alt")
    if cousin_case_alt > 0:
        notes.append("affected_cousin_alt")
    if cousin_control_alt > 0:
        notes.append("unaffected_cousin_alt")

    bucket = bucket_for_class(seg_class)
    model = _model_from_flags(
        compatible_denovo=compatible_denovo,
        compatible_ar=compatible_ar,
        compatible_ad=compatible_ad,
        compatible_xlinked=compatible_xlinked,
    )

    return SegregationResult(
        affected_alt_count=affected_alt,
        affected_called=affected_called,
        unaffected_alt_count=unaffected_alt,
        unaffected_called=unaffected_called,
        affected_ref_count=affected_ref,
        unaffected_ref_count=unaffected_ref,
        missing_count=missing,
        informative_sample_count=informative,
        proband_gt=_gt(record, proband.sample_id if proband else None),
        mother_gt=_gt(record, mother.sample_id if mother else None),
        father_gt=_gt(record, father.sample_id if father else None),
        sibling_case_count=sibling_case_alt,
        sibling_control_count=sibling_control_alt,
        cousin_case_count=cousin_case_alt,
        cousin_control_count=cousin_control_alt,
        segregation_class=seg_class,
        segregation_bucket=bucket,
        segregation_model=model,
        segregation_score=score,
        compatible_ad=compatible_ad,
        compatible_ar=compatible_ar,
        compatible_xlinked=compatible_xlinked,
        compatible_denovo=compatible_denovo,
        notes=";".join(notes) if notes else ".",
    )
