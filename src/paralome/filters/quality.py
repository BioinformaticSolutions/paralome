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

from dataclasses import dataclass, field
from typing import Any, Optional

from paralome.population_af import read_population_af

from .profiles import AnalysisProfile


def _float(row: dict[str, str], keys: list[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value in (None, "", "."):
            continue
        try:
            return float(str(value).replace("%", ""))
        except ValueError:
            continue
    return default


def _int(row: dict[str, str], keys: list[str], default: Optional[int] = None) -> Optional[int]:
    value = _float(row, keys, None)
    if value is None:
        return default
    return int(value)


def _text(row: dict[str, str], keys: list[str], default: str = ".") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "."):
            return str(value)
    return default


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class FilterEvaluation:
    passes: bool = True
    reasons: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)

    def fail(self, reason: str) -> None:
        self.passes = False
        self.reasons.append(reason)



#: Every key a profile's quality block may set, by block, each with
#: its test in evaluate_quality. A profile with a key outside this list is
#: refused (check_profile_quality), so no threshold is listed as applied
#: without being tested.
QUALITY_KEYS: dict[str, frozenset] = {
    "site": frozenset({"min_qual", "require_pass", "allowed_filters", "min_site_depth",
                       "min_site_alt_reads", "max_missing_rate"}),
    "genotype": frozenset({"min_dp", "min_gq", "min_alt_reads", "min_ref_reads",
                           "min_alt_fraction_het", "min_alt_fraction_hom"}),
    "family": frozenset({"min_affected_carriers", "max_unaffected_carriers", "min_total_carriers",
                         "min_sequenced_samples", "require_proband"}),
}

#: Keys Paralome cannot apply, with the reason. The somatic
#: profiles' tumor and normal allele fractions need tumor and normal sample
#: roles, which no stage reads: a somatic profile filtered with no tumor or
#: normal test at all.
UNSUPPORTED_QUALITY_KEYS: dict[tuple[str, str], str] = {
    ("genotype", "min_alt_fraction_tumor"): "Paralome has no tumor/normal sample roles, so a tumor allele "
                                            "fraction cannot be tested",
    ("genotype", "max_alt_fraction_normal"): "Paralome has no tumor/normal sample roles, so a normal allele "
                                             "fraction cannot be tested",
}

#: The genotype thresholds tested per genotype on genotype_evidence.
GENOTYPE_THRESHOLDS = ("min_dp", "min_gq", "min_alt_reads", "min_ref_reads",
                       "min_alt_fraction_het", "min_alt_fraction_hom")


def check_profile_quality(profile: AnalysisProfile) -> None:
    """Refuse a profile whose quality block sets a threshold Paralome does not test."""
    quality = profile.quality if isinstance(profile.quality, dict) else {}
    problems: list[str] = []
    for block, values in quality.items():
        if block not in QUALITY_KEYS:
            problems.append(f"quality.{block} (not a block Paralome reads: {', '.join(sorted(QUALITY_KEYS))})")
            continue
        if values is not None and not isinstance(values, dict):
            problems.append(f"quality.{block} (must be an object of thresholds)")
            continue
        for key, value in (values or {}).items():
            if (block, key) in UNSUPPORTED_QUALITY_KEYS:
                if value is not None:
                    problems.append(f"quality.{block}.{key} ({UNSUPPORTED_QUALITY_KEYS[(block, key)]})")
            elif key not in QUALITY_KEYS[block]:
                problems.append(f"quality.{block}.{key} (no test reads it)")
    if problems:
        raise ValueError(
            f"Profile {profile.name!r} sets thresholds that would not be applied: " + "; ".join(problems)
            + ". Remove them from the filter profile config, or choose another profile."
        )


@dataclass(frozen=True)
class GenotypeEvidence:
    """One genotype of a segregation row's genotype_evidence (segregation.table.genotype_quality_columns)."""

    sample_id: str
    tag: str            # A / U / C: ALT carrier (affected, unaffected, unknown); R: deciding relative
    gt: str
    dp: Optional[int]
    gq: Optional[int]
    alt_reads: Optional[int]
    ref_reads: Optional[int]

    @property
    def carrier(self) -> bool:
        return self.tag in ("A", "U", "C")

    @property
    def homozygous_alt(self) -> bool:
        alleles = [a for a in self.gt.replace("|", "/").split("/") if a]
        return bool(alleles) and all(a not in ("0", ".") for a in alleles)

    @property
    def alt_fraction(self) -> Optional[float]:
        if self.alt_reads is None or self.ref_reads is None or self.alt_reads + self.ref_reads <= 0:
            return None
        return self.alt_reads / (self.alt_reads + self.ref_reads)


def _evidence_int(text: str) -> Optional[int]:
    try:
        return None if text in ("", ".") else int(float(text))
    except ValueError:
        return None


def parse_genotype_evidence(text: Optional[str]) -> list[GenotypeEvidence]:
    out: list[GenotypeEvidence] = []
    for entry in str(text or "").split(";"):
        if not entry or entry == ".":
            continue
        parts = entry.rsplit(":", 6)
        if len(parts) != 7:
            continue
        sample, tag, gt, dp, gq, alt, ref = parts
        out.append(GenotypeEvidence(sample, tag, gt, _evidence_int(dp), _evidence_int(gq),
                                    _evidence_int(alt), _evidence_int(ref)))
    return out


def _carrier_failures(call: GenotypeEvidence, genotype: dict[str, Any]) -> list[str]:
    """Why one carrier's genotype fails the profile's genotype thresholds ([]: it passes)."""
    failures: list[str] = []

    def at_least(value: Optional[float], key: str, name: str) -> None:
        threshold = genotype.get(key)
        if threshold is None:
            return
        if value is None:
            failures.append(f"{name}_unknown")
        elif value < float(threshold):
            failures.append(f"{name}<{threshold}")

    at_least(call.dp, "min_dp", "dp")
    at_least(call.gq, "min_gq", "gq")
    at_least(call.alt_reads, "min_alt_reads", "alt_reads")
    if call.homozygous_alt:
        at_least(call.alt_fraction, "min_alt_fraction_hom", "hom_alt_fraction")
    else:
        at_least(call.ref_reads, "min_ref_reads", "ref_reads")
        at_least(call.alt_fraction, "min_alt_fraction_het", "het_alt_fraction")
    return failures


def evaluate_genotypes(row: dict[str, str], profile: AnalysisProfile, ev: FilterEvaluation) -> None:
    """The profile's genotype thresholds, tested per genotype.

    * the carriers the variant rests on -- the affected carriers, or every
      carrier of unknown status on a sheet that names none -- are tested one by
      one on min_dp, min_gq, min_alt_reads and, for a heterozygous call,
      min_ref_reads and min_alt_fraction_het, for a homozygous one
      min_alt_fraction_hom; at least family.min_affected_carriers (1 when unset)
      of them must pass. A carrier whose VCF lacks a tested number fails it
      ("dp_unknown"): a threshold that could not be tested is not passed.
    * in a pedigree, every called relative that does not carry the ALT must
      pass min_dp and min_gq: a parent's reference call on two reads is what
      makes a false de novo. Unrelated samples are never tested for another's
      variant.

    So one sample's shallow call never fails another sample's variant, and one
    weak carrier does not sink a strong one. Unaffected carriers are not
    tested: their call counts against the variant in segregation, and a weak
    one cannot rescue it here.
    """
    genotype = profile.quality.get("genotype", {}) if isinstance(profile.quality, dict) else {}
    family = profile.quality.get("family", {}) if isinstance(profile.quality, dict) else {}
    if not any(genotype.get(key) is not None for key in GENOTYPE_THRESHOLDS):
        return
    calls = parse_genotype_evidence(row.get("genotype_evidence"))
    carriers = [c for c in calls if c.tag == "A"] or [c for c in calls if c.tag == "C"]
    if carriers:
        failures = [_carrier_failures(c, genotype) for c in carriers]
        passing = sum(1 for f in failures if not f)
        needed = family.get("min_affected_carriers")
        needed = 1 if needed is None else max(1, int(needed))
        if passing < needed:
            reasons = sorted({reason for f in failures for reason in f})
            for reason in reasons:
                ev.fail(f"carrier_{reason}")
            ev.fail(f"carriers_passing_genotype<{needed}")
    elif _int(row, ["affected_alt_count"], 0):
        # Affected carriers counted, but no genotype to test them on.
        ev.fail("carrier_genotype_unknown")
    for relative in (c for c in calls if c.tag == "R"):
        for key, name, value in (("min_dp", "dp", relative.dp), ("min_gq", "gq", relative.gq)):
            threshold = genotype.get(key)
            if threshold is None:
                continue
            if value is None:
                reason = f"relative_{name}_unknown"
            elif value < float(threshold):
                reason = f"relative_{name}<{threshold}"
            else:
                continue
            if reason not in ev.reasons:
                ev.fail(reason)


#: The segregation-table columns each per-variant site threshold
#: reads. A row whose column is '.' (no INFO AO in a GATK VCF, no INFO DP in a
#: DeepVariant one) is not failed on that threshold; site_thresholds_not_tested
#: names it so filter_summary.json can count it.
SITE_THRESHOLD_COLUMNS: dict[str, list[str]] = {
    "min_qual": ["qual", "QUAL", "site_qual"],
    "min_site_depth": ["site_depth", "total_depth", "DP", "INFO_DP"],
    "min_site_alt_reads": ["site_alt_reads", "total_alt_reads", "AO", "INFO_AO"],
    "max_missing_rate": ["missing_rate", "sample_missing_rate"],
}


def site_thresholds_not_tested(row: dict[str, str], profile: AnalysisProfile) -> list[str]:
    """The site thresholds the profile sets that this row has no value to test."""
    site = profile.quality.get("site", {}) if isinstance(profile.quality, dict) else {}
    site = site or {}
    return [
        f"site.{key}" for key, columns in SITE_THRESHOLD_COLUMNS.items()
        if site.get(key) is not None and _float(row, columns) is None
    ]


def evaluate_quality(row: dict[str, str], profile: AnalysisProfile) -> FilterEvaluation:
    ev = FilterEvaluation()
    site = profile.quality.get("site", {}) if isinstance(profile.quality, dict) else {}
    family = profile.quality.get("family", {}) if isinstance(profile.quality, dict) else {}

    qual = _float(row, SITE_THRESHOLD_COLUMNS["min_qual"])
    min_qual = site.get("min_qual")
    if qual is not None and min_qual is not None and qual < float(min_qual):
        ev.fail(f"qual<{min_qual}")

    filt = _text(row, ["filter", "FILTER"], ".")
    allowed_filters = {str(x) for x in site.get("allowed_filters", ["PASS", "."])}
    if site.get("require_pass", False) and filt not in allowed_filters:
        ev.fail("filter_not_pass")

    site_depth = _int(row, SITE_THRESHOLD_COLUMNS["min_site_depth"])
    min_site_depth = site.get("min_site_depth")
    if site_depth is not None and min_site_depth is not None and site_depth < int(min_site_depth):
        ev.fail(f"site_depth<{min_site_depth}")

    site_alt_reads = _int(row, SITE_THRESHOLD_COLUMNS["min_site_alt_reads"])
    min_site_alt = site.get("min_site_alt_reads")
    if site_alt_reads is not None and min_site_alt is not None and site_alt_reads < int(min_site_alt):
        ev.fail(f"site_alt_reads<{min_site_alt}")

    missing_rate = _float(row, SITE_THRESHOLD_COLUMNS["max_missing_rate"])
    max_missing = site.get("max_missing_rate")
    if missing_rate is not None and max_missing is not None and missing_rate > float(max_missing):
        ev.fail(f"missing_rate>{max_missing}")

    evaluate_genotypes(row, profile, ev)

    affected_carriers = _int(row, ["affected_alt_count", "affected_carriers"], 0) or 0
    min_affected = family.get("min_affected_carriers")
    if min_affected is not None and affected_carriers < int(min_affected):
        ev.fail(f"affected_carriers<{min_affected}")

    unaffected_carriers = _int(row, ["unaffected_alt_count", "unaffected_carriers"], 0) or 0
    max_unaffected = family.get("max_unaffected_carriers")
    if max_unaffected is not None and unaffected_carriers > int(max_unaffected):
        ev.fail(f"unaffected_carriers>{max_unaffected}")

    # The rest of the family block, never read before.
    min_total = family.get("min_total_carriers")
    if min_total is not None:
        evidence = parse_genotype_evidence(row.get("genotype_evidence"))
        total_carriers = (sum(1 for c in evidence if c.carrier) if evidence
                          else affected_carriers + unaffected_carriers)
        if total_carriers < int(min_total):
            ev.fail(f"total_carriers<{min_total}")
    min_sequenced = family.get("min_sequenced_samples")
    if min_sequenced is not None:
        sequenced = _int(row, ["informative_sample_count"], 0) or 0
        if sequenced < int(min_sequenced):
            ev.fail(f"sequenced_samples<{min_sequenced}")
    if family.get("require_proband"):
        proband = _text(row, ["proband_gt"], ".").replace("|", "/").split("/")
        if not any(a not in ("0", ".", "") for a in proband):
            ev.fail("proband_not_carrier")

    ev.component_scores["quality"] = 1.0 if ev.passes else 0.0
    return ev


def evaluate_population(row: dict[str, str], profile: AnalysisProfile) -> FilterEvaluation:
    ev = FilterEvaluation()
    pop = profile.population or {}
    # Read as reporting reads it (paralome.population_af): the same
    # columns, the same missing values, the largest of them. This took the
    # first column that parsed, so a multi-allelic "0.002,0.3" was skipped
    # and a rarer column behind it decided. A variant with no frequency
    # (absent from gnomAD, or unreadable) is not failed here, as before.
    # Of the allele the row reports (its reported_alt), whose
    # consequence and gene the row carries -- not the largest of every ALT.
    found = read_population_af(row)
    af = found.af
    min_af = pop.get("min_gnomad_af")
    max_af = pop.get("max_gnomad_af")
    # A FLOOR CANNOT BE PASSED BY A VARIANT WITH NO FREQUENCY. The
    # test above fits a rarity ceiling (max_gnomad_af): absent from gnomAD is
    # rare. It inverted a common-variant floor (germline-common's
    # min_gnomad_af 0.01): a private variant, or one not annotated at all, has
    # no frequency and so was never below the floor, and stood in the "common"
    # pass table beside true polymorphisms. With a floor set, a row with no
    # frequency fails it, saying which: gnomad_af_missing (no value for the
    # allele) or gnomad_af_unparseable (a value that is not a number). A profile
    # that wants such rows anyway sets population.missing_af_passes_floor true.
    if min_af is not None and af is None and not pop.get("missing_af_passes_floor"):
        ev.fail("gnomad_af_unparseable" if found.status == "unparseable" else "gnomad_af_missing")
    if af is not None and min_af is not None and af < float(min_af):
        ev.fail(f"gnomad_af<{min_af}")
    if af is not None and max_af is not None and af > float(max_af):
        ev.fail(f"gnomad_af>{max_af}")
    ev.component_scores["population"] = 1.0 if ev.passes else 0.0
    return ev


def evaluate_annotation(row: dict[str, str], profile: AnalysisProfile) -> FilterEvaluation:
    ev = FilterEvaluation()
    ann = profile.annotation or {}

    # The impact is the report's reading of the row
    # (reports.candidates.protein_consequence_bucket): snpEff's IMPACT where
    # snpEff ran, otherwise ANNOVAR's own words mapped onto HIGH / MODERATE /
    # LOW / MODIFIER. This took the first non-'.' value of a list that fell
    # through to ANNOVAR's Func ('exonic', 'splicing'), which is never HIGH or
    # MODERATE, and failed the row because the 'snpeff_impact' column exists
    # (with '.') -- so on an ANNOVAR-only run every annotated variant failed
    # germline-rare. A row with no consequence at all is still not failed here.
    allowed = {str(x).lower() for x in (ann.get("allowed_impacts") or ann.get("impact") or [])}
    if allowed:
        from paralome.reports.candidates import protein_consequence_bucket

        bucket, described = protein_consequence_bucket(row)
        if described not in ("", ".") and bucket.lower() not in allowed:
            ev.fail("impact_not_allowed")

    clinvar = _text(row, ["clinvar_significance", "clinvar", "ClinVar", "CLNSIG", "clinvar_20250721"], ".")
    exclude = {str(x).lower().replace(" ", "_") for x in ann.get("exclude_clinvar", [])}
    clinvar_norm = clinvar.lower().replace(" ", "_")
    if clinvar != "." and exclude and any(x in clinvar_norm for x in exclude):
        ev.fail("clinvar_excluded")

    ev.component_scores["annotation"] = 1.0 if ev.passes else 0.0
    return ev

