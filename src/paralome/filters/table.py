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
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from paralome.atac.naming import prefixed_name
from paralome.csvio import configure_csv_field_size_limit


CSV_FIELD_SIZE_LIMIT = configure_csv_field_size_limit()

from .config import config_sha256, default_filter_config_path
from .profiles import AnalysisProfile, get_profile, normalize_profile_name
from .quality import (
    GENOTYPE_THRESHOLDS,
    check_profile_quality,
    evaluate_annotation,
    evaluate_population,
    evaluate_quality,
    site_thresholds_not_tested,
)
from paralome.errors import ParalomeError


LEGACY_BUCKETS = ("candidates", "segregating", "non_segregating", "uninformative", "other")
PROFILE_BUCKETS = ("pass", "fail_quality", "fail_population", "fail_annotation", "fail_segregation", "fail_multiple")
VALID_SCOPES = {"all", "segregation_pass", "segregation_partial", "segregation_fail", "segregation_uninformative"}
AUDIT_COLUMNS = (
    "filter_profile",
    "filter_scope",
    "filter_bucket",
    "filter_failed_stages",
    "filter_reasons",
    "quality_score",
    "population_score",
    "annotation_score",
    "segregation_score_filter",
)


@dataclass
class FilterCounts:
    total: int = 0
    candidates: int = 0
    segregating: int = 0
    non_segregating: int = 0
    uninformative: int = 0
    other: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "candidates": self.candidates,
            "segregating": self.segregating,
            "non_segregating": self.non_segregating,
            "uninformative": self.uninformative,
            "other": self.other,
        }


@dataclass
class ProfileBucketCounts:
    scanned: int = 0
    skipped_by_scope: int = 0
    total: int = 0
    passed: int = 0
    fail_quality: int = 0
    fail_population: int = 0
    fail_annotation: int = 0
    fail_segregation: int = 0
    fail_multiple: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "skipped_by_scope": self.skipped_by_scope,
            "total": self.total,
            "pass": self.passed,
            "fail_quality": self.fail_quality,
            "fail_population": self.fail_population,
            "fail_annotation": self.fail_annotation,
            "fail_segregation": self.fail_segregation,
            "fail_multiple": self.fail_multiple,
            "bucket_sum": self.bucket_sum,
        }

    @property
    def bucket_sum(self) -> int:
        return self.passed + self.fail_quality + self.fail_population + self.fail_annotation + self.fail_segregation + self.fail_multiple

    @property
    def lossless(self) -> bool:
        return self.total == self.bucket_sum


@dataclass
class StageEvaluation:
    passed: bool
    failed_stages: list[str]
    reasons: list[str]
    scores: dict[str, float]


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value in (None, "", "."):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in (None, "", "."):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _seg_class(row: dict[str, str]) -> str:
    return (row.get("segregation_class") or "uninformative").strip().lower()


def _seg_bucket(row: dict[str, str]) -> str:
    return (row.get("segregation_bucket") or "").strip().lower()


def normalize_scopes(scope: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if scope is None:
        return ["segregation_pass"]
    raw: list[str] = []
    if isinstance(scope, str):
        raw = [part.strip() for part in scope.replace(";", ",").split(",")]
    else:
        for item in scope:
            raw.extend(str(item).replace(";", ",").split(","))
    scopes: list[str] = []
    for item in raw:
        key = item.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in VALID_SCOPES:
            raise ValueError(f"Unknown filtering scope '{item}'. Valid scopes: {', '.join(sorted(VALID_SCOPES))}")
        if key == "all":
            return ["all"]
        if key not in scopes:
            scopes.append(key)
    return scopes or ["segregation_pass"]


def normalize_profiles(profiles: str | list[str] | tuple[str, ...] | None, profile: str | None = None) -> list[str]:
    raw: list[str] = []
    if profiles:
        if isinstance(profiles, str):
            raw = [part.strip() for part in profiles.replace(";", ",").split(",")]
        else:
            for item in profiles:
                raw.extend(str(item).replace(";", ",").split(","))
    elif profile:
        raw = [profile]
    else:
        raw = ["family-segregating"]
    normalized: list[str] = []
    for item in raw:
        key = normalize_profile_name(item)
        if key and key not in normalized:
            normalized.append(key)
    return normalized or ["family-segregating"]


def _row_in_scope(row: dict[str, str], scopes: list[str]) -> bool:
    if "all" in scopes:
        return True
    bucket = _seg_bucket(row)
    if bucket:
        return bucket in scopes
    # A row with no segregation_bucket (a table from before the column existed)
    # is in no named scope; only --scope all, handled above, takes it.
    return False


def _scope_label(scopes: list[str]) -> str:
    return ",".join(scopes)


def _stage_reasons(stage: str, reasons: list[str]) -> list[str]:
    return [f"{stage}:{reason}" for reason in reasons]


#: Each threshold of a profile's quality block and the table column it needs.
#: The genotype thresholds are tested per genotype on genotype_evidence.
_QUALITY_THRESHOLD_COLUMNS = (
    ("site", "min_site_depth", "site_depth"),
    ("site", "min_site_alt_reads", "site_alt_reads"),
    ("site", "max_missing_rate", "missing_rate"),
    *(("genotype", key, "genotype_evidence") for key in GENOTYPE_THRESHOLDS),
    ("family", "min_total_carriers", "genotype_evidence"),
    ("family", "min_sequenced_samples", "informative_sample_count"),
    ("family", "require_proband", "proband_gt"),
)

#: How filtering applies a profile. Raised whenever a row's verdict
#: for the same table and profile changes; recorded in every filter summary,
#: and a filtering result with another method is not reused
#: (filtering.runner.verify_filter_stage, pipeline.state.summarize_filtering).
#: v3: genotype thresholds per genotype, the allele-fraction, ref-read and
#: family thresholds applied, unknown keys refused, family profiles refused on
#: a pooled-probands table.
#: v4: a profile no single independent proband's variant can pass
#: refused on an unrelated table of 2+ affected samples too; and a population
#: floor (min_gnomad_af) fails a row with no population frequency
#: (quality.evaluate_population) -- a result filtered under v3 by a profile
#: with a floor holds unannotated variants as "common".
FILTER_METHOD = "paralome.filtering.v4"

#: What filter_summary.json says each kind of threshold was tested on.
QUALITY_APPLIED = {
    "site": ("per variant, on the segregation table's site columns; a row whose column is '.' (INFO DP or AO "
             "absent from the VCF) is not failed on that threshold, and quality_not_tested counts such rows"),
    "genotype": ("per genotype, on genotype_evidence: at least family.min_affected_carriers (1 when unset) of the "
                 "affected carriers (every carrier when the sheet names no affected sample) must pass min_dp, "
                 "min_gq, min_alt_reads and, heterozygous, min_ref_reads and min_alt_fraction_het, homozygous "
                 "min_alt_fraction_hom; a number the VCF does not give fails the test. In a pedigree, every "
                 "called relative not carrying the ALT must pass min_dp and min_gq. GQ is the VCF's, or from "
                 "PL/GL where the caller wrote none"),
    "family": "per variant, on the carrier and called-sample counts of the segregation table",
}


def _threshold_set(value: Any) -> bool:
    return value is not None and value is not False


def require_quality_columns(fieldnames: list[str], profile: AnalysisProfile, table: object) -> None:
    """Refuse a profile whose thresholds the table cannot be tested against.

    A segregation table written has none of these
    columns, and each such threshold was skipped without a word while
    filter_summary.json listed it as applied.

    A threshold no test reads is refused too (check_profile_quality),
    and a family profile needs the table's segregation_scoring.
    """
    check_profile_quality(profile)
    quality = profile.quality if isinstance(profile.quality, dict) else {}
    missing = [
        f"{block}.{key} (needs column {column})"
        for block, key, column in _QUALITY_THRESHOLD_COLUMNS
        if _threshold_set((quality.get(block) or {}).get(key)) and column not in fieldnames
    ]
    if profile.mode == "family" and "segregation_scoring" not in fieldnames:
        missing.append("a family profile (needs column segregation_scoring)")
    if missing:
        raise ValueError(
            f"Profile {profile.name!r} sets thresholds that {table} cannot be tested against: "
            + "; ".join(missing)
            + ". The table was written by an older Paralome: run segregation again to write these columns "
            "(the pipeline does so by itself; `paralome-worker segregate run` with --overwrite)."
        )


def table_scoring(segregation_tsv: Union[str, Path]) -> Optional[str]:
    """The segregation_scoring of a table's first row (one value per table), or None."""
    with open(segregation_tsv, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            value = (row.get("segregation_scoring") or "").strip()
            return value or None
    return None


def _affected_scored(segregation_tsv: Union[str, Path], enough: int = 2) -> int:
    """The most affected samples called on any row (affected_total), read until ``enough`` is seen."""
    most = 0
    with open(segregation_tsv, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            most = max(most, _int_value(row, "affected_total", 0))
            if most >= enough:
                break
    return most


def _private_variant_blocked(profile: AnalysisProfile, affected: int) -> Optional[str]:
    """Why a variant one of ``affected`` independent probands carries fails the profile, or None.

    Such a variant is affected_enriched, scored 1/N (segregation.scorer).
    """
    if profile.name == "none":
        return None
    if profile.mode == "family":
        return "it is a family profile"
    if profile.candidate_classes and "affected_enriched" not in profile.candidate_classes:
        return "its candidate_classes leave out affected_enriched"
    if affected and profile.min_segregation_score > 1.0 / affected:
        return f"its min_segregation_score {profile.min_segregation_score} is above 1/{affected}"
    return None


def require_applicable_scoring(segregation_tsv: Union[str, Path], profile: AnalysisProfile) -> Optional[str]:
    """Refuse a family profile on a table whose families were pooled into unrelated probands.

    A sheet of several pedigrees is scored one affected sample per family
    (segregation.table._one_proband_per_pedigree). A variant carried by one
    of N probands is then affected_enriched, scored 1/N, in the partial
    bucket: family-segregating never reads that bucket and wants a score of
    0.5, so only the variants shared by half the probands would be kept, and
    the run would finish with a near-empty candidate list that reads as
    complete.

    An 'unrelated' table (a sheet of independent samples) is scored
    the same way, so with two or more affected samples any profile a variant
    private to one proband cannot pass (family mode, affected_enriched not a
    candidate class, min_segregation_score above 1/N) is refused there too.
    """
    scoring = table_scoring(segregation_tsv)
    if scoring == "pooled_probands" and profile.mode == "family":
        raise ValueError(
            f"{segregation_tsv} holds several families, scored as unrelated probands (one affected sample per "
            f"family, relatives left out: see its segregation summary), so the family profile {profile.name!r} "
            "would keep only the variants shared by half the probands. Filter it with --segregation-profile "
            "unrelated-probands (filter run: --profile unrelated-probands), or run each family as its own "
            "project (paralome batch run) for family segregation."
        )
    if scoring in ("unrelated", "pooled_probands") and profile.name != "none":
        affected = _affected_scored(segregation_tsv)
        why = _private_variant_blocked(profile, affected) if affected >= 2 else None
        if why:
            raise ValueError(
                f"{segregation_tsv} was scored as {scoring} samples (two or more affected, no relative decides a "
                f"variant), and the profile {profile.name!r} cannot keep a variant only one of them carries "
                f"({why}: such a variant is affected_enriched, scored 1/N), so it would keep only the variants "
                "shared by several probands. Filter it with --segregation-profile unrelated-probands (filter "
                "run: --profile unrelated-probands), or give each family its relatives on the sheet (mother, "
                "father, sibling) for family segregation."
            )
    return scoring


#: What the pipeline filters with when a sheet's layout rules out
#: the default profile and nobody chose one.
SHEET_FALLBACK_PROFILE = "unrelated-probands"


def sheet_profile_conflict(samples: Union[str, Path], profile_name: Optional[str]) -> Optional[str]:
    """Why filtering would refuse ``profile_name`` on the table segregation writes for this sheet, or None.

    The same rule as require_applicable_scoring, read from the sheet before
    any job runs, so a run is not refused at filtering after alignment,
    calling and annotation. None when the sheet cannot be read (segregation
    says why) or the profile is unknown (filtering says so).
    """
    import contextlib
    import io

    from paralome.segregation.table import read_scored_roles

    if not profile_name or normalize_profile_name(profile_name) == "none":
        return None
    try:
        profile = get_profile(profile_name)
        with contextlib.redirect_stderr(io.StringIO()):
            scored = read_scored_roles(Path(samples))
    except (OSError, ValueError, KeyError, ParalomeError):
        return None
    affected = sum(1 for role in scored.roles if role.is_affected)
    if scored.scoring == "pooled_probands" and profile.mode == "family":
        return (f"the sheet holds several families, scored as unrelated probands (pooled_probands), and "
                f"{profile.name!r} is a family profile")
    if scored.scoring in ("unrelated", "pooled_probands") and affected >= 2:
        why = _private_variant_blocked(profile, affected)
        if why:
            return (f"the sheet's {affected} affected samples are scored as {scored.scoring} probands, and "
                    f"{profile.name!r} cannot keep a variant only one of them carries ({why})")
    return None


def applicable_pipeline_profile(
    samples: Union[str, Path, None], profile_name: Optional[str], *, chosen_by: Optional[str] = None,
) -> Optional[str]:
    """The filter profile a DNA run uses for this sheet.

    A profile nobody chose (``chosen_by`` None: Paralome's default) that the
    sheet rules out is replaced by unrelated-probands, saying so; one someone
    chose is refused here, before any job runs, naming where it was set.
    """
    if samples is None or not profile_name:
        return profile_name
    conflict = sheet_profile_conflict(samples, profile_name)
    if conflict is None:
        return profile_name
    if chosen_by is None:
        print(
            f"[paralome] {samples}: {conflict}, so filtering would refuse it. Filtering with "
            f"{SHEET_FALLBACK_PROFILE} instead (pass --segregation-profile to choose another; for family "
            "segregation run each family as its own project: paralome batch run).",
            file=sys.stderr, flush=True,
        )
        return SHEET_FALLBACK_PROFILE
    raise ValueError(
        f"Segregation profile {profile_name!r} ({chosen_by}) cannot be used with {samples}: {conflict}, so "
        f"filtering would refuse it after alignment, calling and annotation. Use --segregation-profile "
        f"{SHEET_FALLBACK_PROFILE}, or run each family as its own project (paralome batch run) for family "
        "segregation."
    )


def evaluate_segregation(row: dict[str, str], profile: AnalysisProfile) -> tuple[bool, list[str], float]:
    if profile.name == "none":
        return True, [], 1.0
    seg_class = _seg_class(row)
    score = _float_value(row, "segregation_score", 0.0)
    affected_alt = _int_value(row, "affected_alt_count", 0)
    unaffected_alt = _int_value(row, "unaffected_alt_count", 0)
    reasons: list[str] = []

    if profile.require_affected_alt and affected_alt <= 0:
        reasons.append("no_affected_alt")
    if profile.require_control_absence and unaffected_alt > 0:
        reasons.append("control_alt_present")
    if not profile.allow_unaffected_alt and unaffected_alt > 0:
        reasons.append("unaffected_alt_not_allowed")
    if profile.candidate_classes and seg_class not in profile.candidate_classes:
        reasons.append("segregation_class_not_candidate")
    if score < profile.min_segregation_score:
        reasons.append(f"segregation_score<{profile.min_segregation_score}")

    return not reasons, reasons, 1.0 if not reasons else 0.0


def evaluate_row_by_stage(row: dict[str, str], profile: AnalysisProfile) -> StageEvaluation:
    failed: list[str] = []
    reasons: list[str] = []
    scores: dict[str, float] = {}

    segregation_pass, segregation_reasons, segregation_score = evaluate_segregation(row, profile)
    scores["segregation"] = segregation_score
    if not segregation_pass:
        failed.append("segregation")
        reasons.extend(_stage_reasons("segregation", segregation_reasons))

    quality = evaluate_quality(row, profile)
    scores.update(quality.component_scores)
    if not quality.passes:
        failed.append("quality")
        reasons.extend(_stage_reasons("quality", quality.reasons))

    population = evaluate_population(row, profile)
    scores.update(population.component_scores)
    if not population.passes:
        failed.append("population")
        reasons.extend(_stage_reasons("population", population.reasons))

    annotation = evaluate_annotation(row, profile)
    scores.update(annotation.component_scores)
    if not annotation.passes:
        failed.append("annotation")
        reasons.extend(_stage_reasons("annotation", annotation.reasons))

    return StageEvaluation(passed=not failed, failed_stages=failed, reasons=reasons, scores=scores)


def row_is_candidate(row: dict[str, str], profile: AnalysisProfile) -> tuple[bool, list[str], dict[str, float]]:
    ev = evaluate_row_by_stage(row, profile)
    scores = {k: v for k, v in ev.scores.items() if k != "segregation"}
    return ev.passed, ev.reasons, scores


def classify_row(row: dict[str, str], profile: AnalysisProfile) -> tuple[str, list[str], dict[str, float]]:
    is_candidate, reasons, scores = row_is_candidate(row, profile)
    if is_candidate:
        return "candidates", reasons, scores
    seg_class = _seg_class(row)
    if seg_class in profile.segregating_classes:
        return "segregating", reasons, scores
    if seg_class in profile.non_segregating_classes:
        return "non_segregating", reasons, scores
    if seg_class in profile.uninformative_classes:
        return "uninformative", reasons, scores
    return "other", reasons, scores


def classify_profile_bucket(evaluation: StageEvaluation) -> str:
    if evaluation.passed:
        return "pass"
    failed = set(evaluation.failed_stages)
    if len(failed) != 1:
        return "fail_multiple"
    stage = next(iter(failed))
    return {
        "quality": "fail_quality",
        "population": "fail_population",
        "annotation": "fail_annotation",
        "segregation": "fail_segregation",
    }.get(stage, "fail_multiple")


def _clean_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    cleaned = {}
    for key in fieldnames:
        value = row.get(key, ".")
        if value is None or value == "":
            value = "."
        cleaned[key] = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return cleaned


def _writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    return writer, fh


def _profile_outputs(results_dir: Path, profile_name: str,
                     prefix: Optional[str] = None) -> dict[str, Path]:
    """The files this profile actually writes, carrying the project ID.

    THIS is the function that names the filter outputs -- not the
    same-named helper in ``filtering/runner.py``, which only records paths in
    the plan; the two must agree.

    The prefix is PASSED IN, never derived here. ``results_dir`` is
    ``<workdir>/results`` and "results" is not in derive_file_prefix's
    generic-stage list, so deriving from it would yield "results" and put
    results_pass.tsv on disk -- confidently wrong, which is worse than
    unprefixed. Only the caller knows the workdir.
    """
    profile_dir = results_dir / normalize_profile_name(profile_name)
    return {
        key: profile_dir / prefixed_name(name, prefix)
        for key, name in (
            ("all", "all.tsv"),
            ("pass", "pass.tsv"),
            ("fail_quality", "fail_quality.tsv"),
            ("fail_population", "fail_population.tsv"),
            ("fail_annotation", "fail_annotation.tsv"),
            ("fail_segregation", "fail_segregation.tsv"),
            ("fail_multiple", "fail_multiple.tsv"),
            ("summary", "filter_summary.json"),
        )
    }


def filter_profile_scope_table(
    *,
    segregation_tsv: Union[str, Path],
    results_dir: Union[str, Path],
    profile_name: str,
    scope: str | list[str] | tuple[str, ...] | None = None,
    filter_config: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> dict[str, Any]:
    profile = get_profile(profile_name, filter_config)
    scopes = normalize_scopes(scope)
    segregation_tsv = Path(segregation_tsv)
    results_dir = Path(results_dir)
    outputs = _profile_outputs(results_dir, profile.name, prefix)

    if not segregation_tsv.exists():
        raise ParalomeError(104, f"Segregation table not found: {segregation_tsv}")

    counts = ProfileBucketCounts()
    bucket_counts: Counter[str] = Counter()
    failed_stage_counts: Counter[str] = Counter()
    not_tested: Counter[str] = Counter()

    with open(segregation_tsv, newline="") as in_fh:
        reader = csv.DictReader(in_fh, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Segregation table has no header: {segregation_tsv}")
        fieldnames = list(reader.fieldnames)
        require_quality_columns(fieldnames, profile, segregation_tsv)
        scoring = require_applicable_scoring(segregation_tsv, profile)
        for col in AUDIT_COLUMNS:
            if col not in fieldnames:
                fieldnames.append(col)

        writers: dict[str, csv.DictWriter] = {}
        handles = []
        try:
            writers["all"], fh = _writer(outputs["all"], fieldnames)
            handles.append(fh)
            for bucket in PROFILE_BUCKETS:
                writers[bucket], fh = _writer(outputs[bucket], fieldnames)
                handles.append(fh)

            for row in reader:
                counts.scanned += 1
                if not _row_in_scope(row, scopes):
                    counts.skipped_by_scope += 1
                    continue

                counts.total += 1
                evaluation = evaluate_row_by_stage(row, profile)
                bucket = classify_profile_bucket(evaluation)
                bucket_counts[bucket] += 1
                for stage in evaluation.failed_stages:
                    failed_stage_counts[stage] += 1
                not_tested.update(site_thresholds_not_tested(row, profile))

                out_row = dict(row)
                out_row["filter_profile"] = profile.name
                out_row["filter_scope"] = _scope_label(scopes)
                out_row["filter_bucket"] = bucket
                out_row["filter_failed_stages"] = ";".join(evaluation.failed_stages) if evaluation.failed_stages else "."
                out_row["filter_reasons"] = ";".join(evaluation.reasons) if evaluation.reasons else "."
                out_row["quality_score"] = f"{evaluation.scores.get('quality', 1.0):.3f}"
                out_row["population_score"] = f"{evaluation.scores.get('population', 1.0):.3f}"
                out_row["annotation_score"] = f"{evaluation.scores.get('annotation', 1.0):.3f}"
                out_row["segregation_score_filter"] = f"{evaluation.scores.get('segregation', 1.0):.3f}"

                clean = _clean_row(out_row, fieldnames)
                writers["all"].writerow(clean)
                writers[bucket].writerow(clean)
        finally:
            for fh in handles:
                fh.close()

    counts.passed = bucket_counts.get("pass", 0)
    counts.fail_quality = bucket_counts.get("fail_quality", 0)
    counts.fail_population = bucket_counts.get("fail_population", 0)
    counts.fail_annotation = bucket_counts.get("fail_annotation", 0)
    counts.fail_segregation = bucket_counts.get("fail_segregation", 0)
    counts.fail_multiple = bucket_counts.get("fail_multiple", 0)

    summary = {
        "schema": "paralome.filtering.profile_summary.v2",
        "stage": "filtering",
        "profile": profile.name,
        "description": profile.description,
        "mode": profile.mode,
        "rarity": profile.rarity,
        "scope": scopes,
        "thresholds": profile.thresholds,
        "quality": profile.quality,
        # How each kind of threshold was tested, the method, and how
        # the table's sheet was scored.
        "quality_applied": QUALITY_APPLIED,
        "method": FILTER_METHOD,
        "segregation_scoring": scoring,
        "population": profile.population,
        "annotation": profile.annotation,
        "filter_config": str(filter_config) if filter_config else str(default_filter_config_path()),
        "filter_config_sha256": config_sha256(filter_config),
        "counts": counts.to_dict(),
        "failed_stage_counts": {k: int(v) for k, v in sorted(failed_stage_counts.items())},
        # Rows each site threshold could not be tested on (and so did not fail).
        "quality_not_tested": {k: int(v) for k, v in sorted(not_tested.items())},
        "input": str(segregation_tsv),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "lossless": counts.lossless,
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not counts.lossless:
        raise RuntimeError(
            f"Filtering row-preservation validation failed for {profile.name}: "
            f"total={counts.total} bucket_sum={counts.bucket_sum}"
        )
    return summary


def filter_profiles_scope_table(
    *,
    segregation_tsv: Union[str, Path],
    results_dir: Union[str, Path],
    profiles: str | list[str] | tuple[str, ...] | None = None,
    profile: str | None = None,
    scope: str | list[str] | tuple[str, ...] | None = None,
    filter_config: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> dict[str, Any]:
    names = normalize_profiles(profiles, profile)
    summaries = {}
    for name in names:
        summaries[name] = filter_profile_scope_table(
            segregation_tsv=segregation_tsv,
            results_dir=results_dir,
            prefix=prefix,
            profile_name=name,
            scope=scope,
            filter_config=filter_config,
        )
    return summaries


def filter_segregation_table(
    segregation_tsv: Path,
    out_prefix: Path,
    profile_name: str,
    filter_config: Optional[Path] = None,
) -> dict[str, object]:
    """Legacy final-bucket splitter (candidates / segregating / non_segregating / uninformative / other).

    No command runs it: the filtering stage (`paralome-worker filter run`,
    filtering.runner.run_filter_stage) writes per-profile pass/fail buckets
    through filter_profile_scope_table(). Kept for library callers that rely
    on its output contract.
    """
    profile = get_profile(profile_name, filter_config)
    segregation_tsv = Path(segregation_tsv)
    out_prefix = Path(out_prefix)

    if not segregation_tsv.exists():
        raise ParalomeError(104, f"Segregation table not found: {segregation_tsv}")

    counts = FilterCounts()
    not_tested: Counter[str] = Counter()
    outputs = {
        "candidates": Path(str(out_prefix) + ".candidates.tsv"),
        "segregating": Path(str(out_prefix) + ".segregating.tsv"),
        "non_segregating": Path(str(out_prefix) + ".non_segregating.tsv"),
        "uninformative": Path(str(out_prefix) + ".uninformative.tsv"),
        "other": Path(str(out_prefix) + ".other.tsv"),
        "summary": Path(str(out_prefix) + ".filter_summary.json"),
    }

    with open(segregation_tsv, newline="") as in_fh:
        reader = csv.DictReader(in_fh, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Segregation table has no header: {segregation_tsv}")
        fieldnames = list(reader.fieldnames)
        require_quality_columns(fieldnames, profile, segregation_tsv)
        scoring = require_applicable_scoring(segregation_tsv, profile)
        for col in ("filter_profile", "filter_bucket", "filter_reasons", "quality_score", "population_score", "annotation_score"):
            if col not in fieldnames:
                fieldnames.append(col)

        writers: dict[str, csv.DictWriter] = {}
        handles = []
        try:
            for name in LEGACY_BUCKETS:
                writers[name], fh = _writer(outputs[name], fieldnames)
                handles.append(fh)

            for row in reader:
                counts.total += 1
                not_tested.update(site_thresholds_not_tested(row, profile))
                bucket, reasons, scores = classify_row(row, profile)
                out_row = dict(row)
                out_row["filter_profile"] = profile.name
                out_row["filter_bucket"] = bucket
                out_row["filter_reasons"] = ";".join(reasons) if reasons else "."
                out_row["quality_score"] = f"{scores.get('quality', 1.0):.3f}"
                out_row["population_score"] = f"{scores.get('population', 1.0):.3f}"
                out_row["annotation_score"] = f"{scores.get('annotation', 1.0):.3f}"
                writers[bucket].writerow(_clean_row(out_row, fieldnames))
                if bucket == "candidates":
                    counts.candidates += 1
                elif bucket == "segregating":
                    counts.segregating += 1
                elif bucket == "non_segregating":
                    counts.non_segregating += 1
                elif bucket == "uninformative":
                    counts.uninformative += 1
                else:
                    counts.other += 1
        finally:
            for fh in handles:
                fh.close()

    summary = {
        "profile": profile.name,
        "description": profile.description,
        "mode": profile.mode,
        "rarity": profile.rarity,
        "thresholds": profile.thresholds,
        "quality": profile.quality,
        "quality_applied": QUALITY_APPLIED,
        "method": FILTER_METHOD,
        "segregation_scoring": scoring,
        "population": profile.population,
        "annotation": profile.annotation,
        "filter_config": str(filter_config) if filter_config else str(default_filter_config_path()),
        "filter_config_sha256": config_sha256(filter_config),
        "counts": counts.to_dict(),
        "quality_not_tested": {k: int(v) for k, v in sorted(not_tested.items())},
        "input": str(segregation_tsv),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    # What to run instead is what the user is told.
    raise SystemExit(
        "paralome.filters.table is a library module, not a command.\n"
        "\n"
        "Use the filtering stage, which prefixes its output with the project ID,\n"
        "writes the bucket files and the summary JSON, and checkpoints:\n"
        "\n"
        # `paralome-worker`, not `paralome`: `filter` belongs to the
        # machine-facing binary.
        "    paralome-worker filter run --segregation <tsv> --out <workdir> --prefix <project ID, e.g. FAM01>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
