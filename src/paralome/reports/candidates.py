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
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from paralome.atac.naming import (
    assert_outputs_are_prefixed,
    derive_file_prefix,
    prefixed_name,
    require_file_prefix,
    snapshot_results,
    stage_banner,
)
from paralome.checkpoints import stage_checkpoint_path
from paralome.csvio import configure_csv_field_size_limit
from paralome.pipeline.region_shards import RegionIndex
from paralome.population_af import (
    ROW_AF_KEYS,
    STATUS_ABSENT,
    STATUS_PRESENT,
    STATUS_UNPARSEABLE,
    STATUSES,
    UNPARSEABLE_MEANING,
    PopulationAF,
    absent_meaning,
    read_population_af,
)
from paralome.reports.xlsx_utils import round_significant


CSV_FIELD_SIZE_LIMIT = configure_csv_field_size_limit()

#: What this module computes, raised whenever a row's tier, bucket or
#: classification changes for the same input table. A report written under an
#: older method is not read as complete (pipeline/state.py summarize_reporting,
#: verify_candidate_review_report), so it is written again rather than reused.
#: v2: ANNOVAR's \x3b-escaped gene lists are decoded before the known-risk
#: match; a non-coding RNA "splicing" Func is not a protein splice-site; the QC
#: flag is read from the columns segregation writes (segregation_class
#: possible_de_novo, compatible_denovo); the checkpoint records the input
#: table's size and digest.
#: v3: the outputs follow the study design -- a family (one pedigree
#: scored as such) or a cohort (unrelated or pooled probands) -- as lists named
#: for what their rows are, each holding every row its rule takes (a row may be
#: in several), recorded per row in report_lists; the review tiers are gone.
REPORT_METHOD = "paralome.candidate_report.v3"
#: Checkpoint v2 -- fail/uninformative rows carry
#: review_tier=not_reviewable. A checkpoint naming another schema is not current.
#: Checkpoint v3 -- the design and its lists.
CHECKPOINT_SCHEMA = "paralome.candidate_report.checkpoint.v3"
#: Bytes digested at each end of the input table for its record.
INPUT_DIGEST_BYTES = 1 << 20


SEGREGATION_REVIEW_BUCKETS = {"segregation_pass", "segregation_partial", "pass", "partial"}
SEGREGATION_ARCHIVE_BUCKETS = {"segregation_fail", "segregation_uninformative", "fail", "uninformative"}
POPULATION_BUCKETS = ("common", "rare", "ultrarare", "unknown")
CONSEQUENCE_BUCKETS = ("high", "moderate", "low", "modifier")
REVIEW_COLUMNS = (
    # The report lists this row is in (see REPORT_LISTS), ';'-joined.
    "report_lists",
    # The affected samples carrying the ALT, from genotype_evidence.
    "n_affected_carriers",
    "affected_carriers",
    "segregation_bucket_normalized",
    "population_bucket",
    "protein_consequence_bucket",
    "clinical_bucket",
    "candidate_gene_match",
    "known_risk_gene",
    "max_population_af",
    "population_af_source",
    # Why a variant's population bucket is "unknown": absent (not in
    # gnomAD's exomes) or unparseable. See paralome.population_af.
    "population_af_status",
    "consequence_terms",
)
CORE_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    # The ALT whose consequence and frequency the row reports.
    "reported_alt",
    "gene",
    "variant_id",
    "segregation_bucket",
    "segregation_class",
    "affected_alt_count",
    "affected_ref_count",
    "unaffected_alt_count",
    "unaffected_ref_count",
    "missing_count",
    "inferred_carrier_samples",
    "penetrance_conflict",
    "possible_denovo",
    "mendelian_compatible",
    "gnomad_af",
    "gnomad_af_grpmax",
    "max_population_af",
    "population_af_source",
    "snpeff_impact",
    "snpeff_effect",
    "annovar_func",
    "annovar_exonic_func",
    "clinvar_significance",
    "impact",
    "CLNSIG",
    "ANN",
    "Func.refGene",
    "ExonicFunc.refGene",
)

BUCKETED_WORKBOOK_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    # The ALT whose consequence and frequency the row reports.
    "reported_alt",
    "gene",
    "variant_id",
    "source_vcf_record",
    "source_bucket",
    "segregation_bucket_normalized",
    "segregation_bucket",
    "segregation_class",
    "population_bucket",
    "protein_consequence_bucket",
    "clinical_bucket",
    "candidate_gene_match",
    "known_risk_gene",
    "report_lists",
    "n_affected_carriers",
    "affected_carriers",
    "max_population_af",
    "population_af_source",
    "population_af_status",
    "gnomad_af",
    "gnomad_af_grpmax",
    "snpeff_impact",
    "snpeff_effect",
    "annovar_func",
    "annovar_exonic_func",
    "clinvar_significance",
    "affected_alt_count",
    "affected_ref_count",
    "unaffected_alt_count",
    "unaffected_ref_count",
    "missing_count",
    "proband_gt",
    "segregation_reasons",
)

SEGREGATING_EXCEL_BUCKETS = ("pass", "partial")
NONSEGREGATING_EXCEL_BUCKETS = ("fail", "uninformative")
#: The columns a frequency is read from -- the ones filtering reads,
#: through the same function (paralome.population_af.read_population_af).
POP_AF_KEYS = ROW_AF_KEYS
IMPACT_KEYS = (
    "snpeff_impact",
    "impact",
    "snpEff_impact",
    "ANN_impact",
    "IMPACT",
)
CONSEQUENCE_KEYS = (
    "consequence",
    "Consequence",
    "annovar_func",
    "annovar_exonic_func",
    "snpeff_effect",
    "Func.refGeneWithVer",
    "ExonicFunc.refGeneWithVer",
    "Func.refGene",
    "ExonicFunc.refGene",
    "ANN",
)
CLINICAL_KEYS = (
    "clinvar_significance",
    "CLNSIG",
    "ClinVar",
    "clinvar",
    "clinvar_clnsig",
)


@dataclass
class CandidateReportResult:
    workdir: Path
    input_segregation: Path
    summary_json: Path
    audit_manifest_json: Path
    segregation_summary_csv: Path
    matrix_csv: Path
    all_reviewable_tsv: Path
    #: "family" or "cohort" (see study_design).
    design: str
    #: Each report list written: its key (REPORT_LISTS) -> its file.
    lists: dict[str, Path]
    bucketed_workbook_rows: dict[str, int]
    bucket_tsv_dir: Optional[Path]
    bucket_tsv_rows: dict[str, int]
    threads: int
    shards: int
    rows_scanned: int
    rows_reviewable: int
    complete: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data["lists"] = {key: str(value) for key, value in self.lists.items()}
        return data


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe(value: Any) -> str:
    if value is None or value == "":
        return "."
    if isinstance(value, float):
        # Rounded as the other table writers round (paralome.reports.xlsx_utils), so a float column (a
        # population allele frequency, say) does not carry full float64 precision as text. This sheet
        # stringifies every value already (typed and missing/"." values mixed), so this only shortens the
        # text, not the cell type.
        value = round_significant(value)
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _first_value(row: dict[str, str], keys: Iterable[str]) -> tuple[str, str]:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "."):
            return key, str(value)
    return "", ""


def max_population_af(row: dict[str, str]) -> tuple[Optional[float], str]:
    found = read_population_af(row)
    return found.af, found.source


def population_reading(row: dict[str, str]) -> tuple[str, PopulationAF]:
    """(population bucket, the frequency it was read from).

    "unknown" is a variant with no frequency; ``.status`` says whether it is
    absent from gnomAD (for a whole-genome run on gnomAD exomes, most
    variants) or carries a value that is not a frequency.
    """
    found = read_population_af(row)
    af = found.af
    if af is None:
        return "unknown", found
    if af <= 0.001:
        return "ultrarare", found
    if af <= 0.01:
        return "rare", found
    return "common", found


def population_bucket(row: dict[str, str]) -> tuple[str, Optional[float], str]:
    bucket, found = population_reading(row)
    return bucket, found.af, found.source


#: ANNOVAR -vcfinput writes the characters a VCF INFO field cannot
#: hold as hex escapes: ';' as \x3b, '=' as \x3d, ',' as \x2c. They are decoded
#: before any value is matched, so "SCN1A\x3bSCN1A-AS1" is two genes and
#: "ncRNA_exonic\x3bsplicing" is two Func tokens (burden/qualify.py decodes the
#: same way).
_ANNOVAR_ESCAPES = (("\\x3b", ";"), ("\\x3d", "="), ("\\x2c", ","))


def decode_annovar(text: Any) -> str:
    value = str(text)
    for escaped, plain in _ANNOVAR_ESCAPES:
        value = value.replace(escaped, plain).replace(escaped.upper(), plain)
    return value


def _row_text(row: dict[str, str], keys: Iterable[str]) -> str:
    values = []
    for key in keys:
        val = row.get(key)
        if val not in (None, "", "."):
            values.append(decode_annovar(val))
    return " | ".join(values)


_TOKEN_SPLIT = re.compile(r"[;,|/\s]+")


def _high_consequence_text(row: dict[str, str]) -> str:
    """The consequence text the high terms are tested on.

    Per Func value (ANNOVAR writes "ncRNA_exonic;splicing" for a non-coding
    RNA exon/splice variant): where a value has a non-coding RNA token, its
    "splicing" is the RNA's, not a protein splice-site, and is taken out with
    the ncRNA tokens. "nonframeshift" contains "frameshift" and is taken out
    too, so an in-frame indel stays moderate.
    """
    kept: list[str] = []
    for key in list(IMPACT_KEYS) + list(CONSEQUENCE_KEYS):
        val = row.get(key)
        if val in (None, "", "."):
            continue
        tokens = [t for t in _TOKEN_SPLIT.split(decode_annovar(val).lower()) if t]
        if any("ncrna" in token for token in tokens):
            tokens = [t for t in tokens if "ncrna" not in t and t != "splicing"]
        kept.extend(tokens)
    return " ".join(kept).replace("nonframeshift", "")


def protein_consequence_bucket(row: dict[str, str]) -> tuple[str, str]:
    _, impact = _first_value(row, IMPACT_KEYS)
    text = _row_text(row, list(IMPACT_KEYS) + list(CONSEQUENCE_KEYS)).lower()
    impact_norm = impact.strip().upper()

    high_terms = (
        "high",
        "stop_gained",
        "stopgain",
        "stop-gain",
        "frameshift",
        "splice_acceptor",
        "splice_donor",
        "start_lost",
        "stop_lost",
        "nonsense",
        # ANNOVAR's own words (the pipeline's default engine).
        # Without them a canonical splice-site, stop-loss or start-loss
        # variant fell through to "modifier" and out of the review tiers.
        "splicing",
        "stoploss",
        "startloss",
    )
    moderate_terms = (
        "moderate",
        "missense",
        "nonsynonymous",
        "inframe_insertion",
        "inframe_deletion",
        "protein_altering",
        "nonframeshift",
    )
    low_terms = (
        "low",
        "synonymous",
        "stop_retained",
        "start_retained",
    )

    # "nonframeshift" contains "frameshift", and a non-coding RNA
    # "splicing" is not a protein splice-site: both are taken out before the
    # high terms are tested. Per Func token (_high_consequence_text).
    high_text = _high_consequence_text(row)
    if impact_norm == "HIGH" or any(term in high_text for term in high_terms):
        return "high", _row_text(row, list(IMPACT_KEYS) + list(CONSEQUENCE_KEYS)) or "."
    if impact_norm == "MODERATE" or any(term in text for term in moderate_terms):
        return "moderate", _row_text(row, list(IMPACT_KEYS) + list(CONSEQUENCE_KEYS)) or "."
    if impact_norm == "LOW" or any(term in text for term in low_terms):
        return "low", _row_text(row, list(IMPACT_KEYS) + list(CONSEQUENCE_KEYS)) or "."
    return "modifier", _row_text(row, list(IMPACT_KEYS) + list(CONSEQUENCE_KEYS)) or "."


#: The ClinVar buckets that go to clinical review.
CLINICAL_REVIEW_BUCKETS = frozenset({"pathogenic_or_likely_pathogenic", "vus", "conflicting"})


def clinical_bucket(row: dict[str, str]) -> str:
    text = _row_text(row, CLINICAL_KEYS).lower()
    if not text:
        return "unknown"
    normalized = text.replace(" ", "_").replace("-", "_")
    # ClinVar's "Conflicting_classifications_of_pathogenicity"
    # (earlier "..._interpretations_of_pathogenicity") contains "pathogenic"
    # and was counted as pathogenic/likely pathogenic. It is its own bucket,
    # reviewed with VUS.
    if "conflicting" in normalized:
        return "conflicting"
    if "pathogenic" in normalized and "benign" not in normalized:
        return "pathogenic_or_likely_pathogenic"
    if "uncertain" in normalized or "vus" in normalized:
        return "vus"
    if "benign" in normalized:
        return "benign_or_likely_benign"
    return "other_clinical"


def parse_gene_set(value: Union[str, Sequence[str], None]) -> set[str]:
    if value is None:
        return set()
    raw: list[str] = []
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        for item in value:
            raw.extend(str(item).replace(";", ",").split(","))
    return {x.strip().upper() for x in raw if x.strip()}


def read_gene_file(path: Union[str, Path, None]) -> set[str]:
    if path in (None, ""):
        return set()
    p = Path(path)
    genes: set[str] = set()
    # A byte-order mark at the top of the file is ignored.
    with p.open(encoding="utf-8-sig") as fh:
        for line in fh:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            for token in text.replace(";", ",").split(","):
                token = token.strip()
                if token:
                    genes.add(token.upper())
    return genes


def gene_symbol(row: dict[str, str]) -> str:
    """The first gene named on the row.

    ANNOVAR's escaped ';' is decoded and the list split on ',' and ';' both,
    so "SCN1A\x3bSCN1A-AS1" (an exonic or intronic variant with an overlapping
    antisense gene) is SCN1A and matches --candidate-genes.
    """
    for key in ("gene", "annovar_gene", "snpeff_gene", "Gene.refGeneWithVer", "Gene.refGene", "Gene", "SYMBOL"):
        val = row.get(key)
        if val not in (None, "", "."):
            for token in re.split(r"[,;]", decode_annovar(val)):
                token = token.strip()
                if token and token != ".":
                    return token
            continue
    ann = row.get("ANN") or ""
    if ann and "|" in ann:
        fields = ann.split("|")
        if len(fields) > 3 and fields[3]:
            return fields[3].strip()
    return "."


def normalized_segregation_bucket(row: dict[str, str]) -> str:
    bucket = _norm_text(row.get("segregation_bucket"))
    if bucket in {"segregation_pass", "pass"}:
        return "pass"
    if bucket in {"segregation_partial", "partial"}:
        return "partial"
    if bucket in {"segregation_fail", "fail"}:
        return "fail"
    if bucket in {"segregation_uninformative", "uninformative"}:
        return "uninformative"
    return bucket or "unknown"


def truthy(row: dict[str, str], keys: Iterable[str]) -> bool:
    for key in keys:
        val = _norm_text(row.get(key))
        if val in {"1", "true", "yes", "y"}:
            return True
    return False


def qc_review_flag(row: dict[str, str]) -> bool:
    """Whether the row goes to qc_review (a pedigree or genotype QC flag).

    Read from the columns the segregation table WRITES. The flag
    keyed on possible_denovo, penetrance_conflict and mendelian_compatible,
    none of which segregation/table.py has ever written, so qc_review.tsv
    was empty on every trio run and a possible de novo call went to tier_2
    with no QC flag. Segregation says it as segregation_class ==
    "possible_de_novo" and compatible_denovo == true (a proband ALT with both
    parents called reference, which is the call most often an artefact and
    the one to check first). The three older names are still honoured for a
    table that carries them.
    """
    if _norm_text(row.get("segregation_class")) == "possible_de_novo":
        return True
    if truthy(row, ("compatible_denovo", "possible_denovo", "penetrance_conflict")):
        return True
    return _norm_text(row.get("mendelian_compatible")) in {"false", "no"}


# ---------------------------------------------------------------------------
# The study design and its report lists
# ---------------------------------------------------------------------------

#: The design a report is written for. A family: one pedigree, scored with its
#: relatives (segregation_scoring "pedigree"), so pass/partial is family
#: segregation. A cohort: unrelated probands, or several pedigrees cut to one
#: proband each ("unrelated", "pooled_probands"), so pass/partial means "carried
#: by affected samples" and there is no family evidence to report.
DESIGN_FAMILY = "family"
DESIGN_COHORT = "cohort"
COHORT_SCORINGS = frozenset({"unrelated", "pooled_probands"})

#: Every list a row can be in, and the rule that puts it there (report_lists).
REPORT_LISTS = (
    "candidate_genes",               # reviewable, in a --candidate-genes gene
    "rare_damaging",                 # reviewable, rare/ultrarare/unknown, high/moderate
    "nonsegregating_rare_damaging",  # fail/uninformative, rare/ultrarare/unknown, high/moderate
    "clinical",                      # reviewable, ClinVar P/LP, VUS or conflicting
    "de_novo",                       # reviewable, segregation's possible de novo
    "common",                        # reviewable, common
)
RARE_OR_UNKNOWN = frozenset({"rare", "ultrarare", "unknown"})
DAMAGING = frozenset({"high", "moderate"})

#: Each design's lists, in the order they are written and shown, with their
#: file names. A list the design does not have is not written.
DESIGN_LISTS = {
    DESIGN_FAMILY: {
        "candidate_genes": "candidate_genes.tsv",
        "rare_damaging": "segregating_rare_damaging.tsv",
        "nonsegregating_rare_damaging": "nonsegregating_rare_damaging.tsv",
        "clinical": "clinical_variants.tsv",
        "de_novo": "de_novo_check.tsv",
        "common": "common_variants.tsv",
    },
    DESIGN_COHORT: {
        "candidate_genes": "candidate_genes.tsv",
        "rare_damaging": "rare_damaging_variants.tsv",
        "clinical": "clinical_variants.tsv",
        "common": "common_variants.tsv",
    },
}
#: The client lists (the common list stays in-house).
CLIENT_LISTS = {
    DESIGN_FAMILY: ("candidate_genes", "rare_damaging", "nonsegregating_rare_damaging", "clinical", "de_novo"),
    DESIGN_COHORT: ("candidate_genes", "rare_damaging", "clinical"),
}
SHEET_TITLES = {
    DESIGN_FAMILY: {
        "candidate_genes": "Candidate_genes", "rare_damaging": "Segregating_rare_damaging",
        "nonsegregating_rare_damaging": "Nonsegregating_rare_damaging", "clinical": "Clinical",
        "de_novo": "De_novo_check",
    },
    DESIGN_COHORT: {"candidate_genes": "Candidate_genes", "rare_damaging": "Rare_damaging", "clinical": "Clinical"},
}
LIST_MEANING = {
    DESIGN_FAMILY: {
        "candidate_genes": "segregating (pass/partial) variants in a --candidate-genes gene",
        "rare_damaging": "segregating (pass/partial), rare/ultrarare or not in gnomAD, high/moderate consequence",
        "nonsegregating_rare_damaging": "failing or uninformative for segregation, rare/ultrarare or not in "
                                        "gnomAD, high/moderate consequence",
        "clinical": "segregating (pass/partial) variants with a ClinVar P/LP, VUS or conflicting call",
        "de_novo": "segregating (pass/partial) variants segregation called a possible de novo: check the "
                   "parents' calls before reporting",
        "common": "segregating (pass/partial) common variants",
    },
    DESIGN_COHORT: {
        "candidate_genes": "variants carried by affected samples in a --candidate-genes gene",
        "rare_damaging": "variants carried by affected samples, rare/ultrarare or not in gnomAD, high/moderate "
                         "consequence; most affected carriers first",
        "clinical": "variants carried by affected samples with a ClinVar P/LP, VUS or conflicting call",
        "common": "common variants carried by affected samples",
    },
}


def report_lists(
    *, segregation: str, population: str, consequence: str, clinical: str,
    known_gene: bool, de_novo: bool,
) -> list[str]:
    """Every report list (REPORT_LISTS) a row belongs in, by its own rule.

    The lists are not exclusive: a rare damaging variant with a ClinVar call
    is in the clinical list and the rare-damaging one. A list holds every row
    its rule describes.
    """
    lists: list[str] = []
    reviewable = segregation in {"pass", "partial"}
    rare_damaging = population in RARE_OR_UNKNOWN and consequence in DAMAGING
    if reviewable and known_gene:
        lists.append("candidate_genes")
    if reviewable and rare_damaging:
        lists.append("rare_damaging")
    if segregation in {"fail", "uninformative"} and rare_damaging:
        lists.append("nonsegregating_rare_damaging")
    if reviewable and clinical in CLINICAL_REVIEW_BUCKETS:
        lists.append("clinical")
    if reviewable and de_novo:
        lists.append("de_novo")
    if reviewable and population == "common":
        lists.append("common")
    return lists


def affected_carriers(row: Mapping[str, Any]) -> list[str]:
    """The affected samples carrying the ALT, from segregation's genotype_evidence.

    genotype_evidence is ``sample:tag:GT:DP:GQ:ALT_READS:REF_READS`` per
    genotype, ';' between them; tag A is an affected ALT carrier.
    """
    text = str(row.get("genotype_evidence") or "")
    carriers: list[str] = []
    for entry in text.split(";"):
        parts = entry.rsplit(":", 6)
        if len(parts) == 7 and parts[1] == "A" and parts[0] not in carriers:
            carriers.append(parts[0])
    return carriers


def study_design(scorings: Iterable[str]) -> tuple[str, str]:
    """(design, why) from the segregation_scoring values the table holds.

    A table that does not say (written before segregation recorded it) is read
    as a family, as it always was.
    """
    seen = {str(value).strip() for value in scorings if str(value or "").strip() not in ("", ".")}
    if not seen:
        return DESIGN_FAMILY, "the segregation table does not record how it was scored; read as a family"
    if seen <= COHORT_SCORINGS:
        return DESIGN_COHORT, f"segregation scored the samples as {', '.join(sorted(seen))}"
    return DESIGN_FAMILY, f"segregation scored the samples as {', '.join(sorted(seen))}"


def _output_paths(workdir: Path, prefix: str | None = None) -> dict[str, Path]:
    """Every candidate-report output, carrying the project ID.

    Every file a report writes carries the family ID, the files most likely to
    be sent on above all.

    bucket_tsv_dir is a directory and keeps its name; the files inside it are
    prefixed by their writer.
    """
    results = workdir / "results"
    run_prefix = prefix or derive_file_prefix(workdir)

    def _out(name: str) -> Path:
        return results / prefixed_name(name, run_prefix)

    paths = {
        "summary_json": _out("candidate_report_summary.json"),
        "audit_manifest_json": _out("candidate_report_manifest.json"),
        "segregation_summary_csv": _out("segregation_summary.csv"),
        "matrix_csv": _out("population_consequence_matrix.csv"),
        "bucket_tsv_summary_csv": _out("bucket_tsv_summary.csv"),
        "all_reviewable_tsv": _out("all_reviewable_candidates.tsv"),
        "bucket_tsv_dir": results / "bucket_tsv",
    }
    return paths


def _list_path(workdir: Path, prefix: str, design: str, key: str) -> Path:
    """Where report list ``key`` of ``design`` is written."""
    return workdir / "results" / prefixed_name(DESIGN_LISTS[design][key], prefix)


#: The names the report wrote before its lists followed the design
#: (v2: review tiers, the segregation-centred workbook under the client name).
#: Removed, by name, when a report runs again, with every name either design
#: writes now -- so a report rerun as the other design leaves nothing behind.
_FORMER_OUTPUT_NAMES = (
    "tier_1_known_risk.tsv", "tier_2_family_supported.tsv", "tier_3_rare_damaging.tsv",
    "tier_4_common_modifiers.tsv", "clinical_review.tsv", "qc_review.tsv", "segregating_review.xlsx",
    # Workbooks are not written; old ones are removed when a report runs again.
    "candidate_review.xlsx", "candidate_variants.xlsx", "bucket_review.xlsx", "failed_uninformative.xlsx",
)


def _every_output_name() -> list[str]:
    names = list(_FORMER_OUTPUT_NAMES)
    for design in (DESIGN_FAMILY, DESIGN_COHORT):
        names += list(DESIGN_LISTS[design].values())
    return list(dict.fromkeys(names))


def _make_fieldnames(input_fields: list[str]) -> list[str]:
    fields: list[str] = []
    for col in REVIEW_COLUMNS + CORE_COLUMNS:
        if col not in fields:
            fields.append(col)
    for col in input_fields:
        if col not in fields:
            fields.append(col)
    return fields


def _writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    return writer, fh


def _write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe(row.get(key, ".")) for key in fieldnames})


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _bucket_filename(seg: str, pop: str, consequence: str) -> str:
    return f"{_sheet_name(seg, pop, consequence)}.tsv"


def _open_bucket_tsv_writers(bucket_dir: Path, prefix: str | None = None) -> tuple[dict[tuple[str, str, str], csv.DictWriter], list[Any]]:
    """Open one compact TSV writer per evidence bucket.

    Bucket TSVs are the canonical reporting layer (there are no Excel
    workbooks; OmicView's variant browser reads the segregation table).
    """
    bucket_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, str, str], csv.DictWriter] = {}
    handles: list[Any] = []
    for seg in ("pass", "partial", "fail", "uninformative"):
        for pop in POPULATION_BUCKETS:
            for consequence in CONSEQUENCE_BUCKETS:
                # 64 files per family. Left bare they are the largest
                # single source of collisions after a flat collect -- ten
                # families give 640 files across ten identical name sets.
                path = bucket_dir / prefixed_name(_bucket_filename(seg, pop, consequence), prefix)
                fh = path.open("w", newline="")
                writer = csv.DictWriter(
                    fh,
                    fieldnames=list(BUCKETED_WORKBOOK_COLUMNS),
                    delimiter="\t",
                    lineterminator="\n",
                    extrasaction="ignore",
                )
                writer.writeheader()
                writers[(seg, pop, consequence)] = writer
                handles.append(fh)
    return writers, handles


def _count_tsv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _write_bucket_tsv_summary(bucket_dir: Path, summary_path: Path, prefix: str | None = None) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for seg in ("pass", "partial", "fail", "uninformative"):
        for pop in POPULATION_BUCKETS:
            for consequence in CONSEQUENCE_BUCKETS:
                name = _sheet_name(seg, pop, consequence)
                path = bucket_dir / prefixed_name(f"{name}.tsv", prefix)
                count = _count_tsv_rows(path)
                counts[name] = count
                rows.append(
                    {
                        "source_bucket": name,
                        "segregation_bucket": seg,
                        "population_bucket": pop,
                        "protein_consequence_bucket": consequence,
                        "rows": str(count),
                        "path": str(path),
                    }
                )
    _write_csv_rows(
        summary_path,
        ["source_bucket", "segregation_bucket", "population_bucket", "protein_consequence_bucket", "rows", "path"],
        rows,
    )
    return counts


def _sheet_name(seg: str, pop: str, consequence: str) -> str:
    seg_short = {"uninformative": "uninf"}.get(seg, seg)
    name = f"{seg_short}_{pop}_{consequence}"
    return name[:31]


def openpyxl_temp_dir(workdir: Path) -> Path:
    """Where openpyxl spooled a workbook's sheets; an interrupted
    run's leftovers there are still cleared when the report runs again."""
    return Path(workdir) / "temp" / "openpyxl"


@dataclass(frozen=True)
class _Region:
    chrom: str
    start: int
    end: int
    name: str


def _norm_chrom(chrom: Any) -> str:
    text = str(chrom or "").strip()
    if text.lower().startswith("chr"):
        return text[3:]
    return text


def _parse_region_token(text: str, idx: int, convention: Optional[str] = None) -> Optional[_Region]:
    """One parser for every region map (pipeline.region_shards), so
    reporting and segregation read a FreeBayes region file the same way."""
    from paralome.pipeline.region_shards import parse_region_token

    region = parse_region_token(text, idx, convention)
    return None if region is None else _Region(region.chrom, region.start, region.end, region.name)


def read_report_regions(path: Union[str, Path, None], convention: Optional[str] = None) -> list[_Region]:
    from paralome.pipeline.region_shards import check_region_convention

    convention = check_region_convention(convention)
    if path in (None, ""):
        return []
    p = Path(path)
    regions: list[_Region] = []
    # A byte-order mark at the top of the file is ignored.
    with p.open(encoding="utf-8-sig") as fh:
        for idx, line in enumerate(fh, start=1):
            region = _parse_region_token(line, idx, convention)
            if region is not None:
                regions.append(region)
    return regions


def _region_indexes_by_chrom(regions: Sequence[_Region]) -> RegionIndex:
    """A RegionIndex, looked up by bisection (pipeline.region_shards).
    The walk over every region of the contig, for every row, cost about 8
    minutes on 13.2 M rows over 34,000 regions."""
    by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for idx, region in enumerate(regions):
        by_chrom[_norm_chrom(region.chrom)].append((region.start, region.end, idx))
    return RegionIndex(by_chrom)


def _assign_region_shard(row: dict[str, str], regions_by_chrom: Mapping[str, list[tuple[int, int, int]]], shard_count: int, fallback_index: int) -> int:
    chrom = _norm_chrom(row.get("chrom") or row.get("CHROM") or row.get("#CHROM"))
    pos_text = row.get("pos") or row.get("POS") or ""
    try:
        pos = int(float(str(pos_text)))
    except ValueError:
        return fallback_index % shard_count
    index = regions_by_chrom if isinstance(regions_by_chrom, RegionIndex) else RegionIndex(dict(regions_by_chrom))
    region_idx = index.find(chrom, pos)
    if region_idx is not None:
        return region_idx % shard_count
    return fallback_index % shard_count


def _classify_candidate_row(row: dict[str, str], known_genes: set[str]) -> tuple[dict[str, Any], str, str, str, str, list[str], bool, str]:
    """(row with the report's columns, segregation, population, consequence,
    clinical, report lists, de novo flag, bucket name)."""
    seg = normalized_segregation_bucket(row)
    pop, found = population_reading(row)
    af, af_source = found.af, found.source
    consequence, terms = protein_consequence_bucket(row)
    clinical = clinical_bucket(row)
    gene = gene_symbol(row)
    known = gene.upper() in known_genes if gene and gene != "." else False
    de_novo = qc_review_flag(row)
    lists = report_lists(segregation=seg, population=pop, consequence=consequence, clinical=clinical,
                         known_gene=known, de_novo=de_novo)
    carriers = affected_carriers(row)
    source_bucket = _sheet_name(seg, pop, consequence)
    out = dict(row)
    out.update(
        {
            "source_bucket": source_bucket,
            "report_lists": ";".join(lists) or ".",
            "n_affected_carriers": str(len(carriers)),
            "affected_carriers": ",".join(carriers) or ".",
            "segregation_bucket_normalized": seg,
            "population_bucket": pop,
            "protein_consequence_bucket": consequence,
            "clinical_bucket": clinical,
            "candidate_gene_match": "true" if known else "false",
            "known_risk_gene": "true" if known else "false",
            "max_population_af": "." if af is None else f"{af:.8g}",
            "population_af_source": af_source or ".",
            "population_af_status": found.status,
            "consequence_terms": terms,
            "gene": gene,
        }
    )
    return out, seg, pop, consequence, clinical, lists, de_novo, source_bucket


def _count_population(out: dict[str, Any], status: Counter, sources: Counter) -> None:
    """How many rows had a frequency, none (absent), or an unreadable one."""
    value = str(out.get("population_af_status") or STATUS_ABSENT)
    status[value] += 1
    if value == STATUS_PRESENT:
        sources[str(out.get("population_af_source") or ".")] += 1


def population_af_summary(status: Counter, sources: Counter) -> dict[str, Any]:
    """What the "unknown" population bucket is made of, in words.

    A whole-genome cohort run had 94% of its rows "unknown" with nothing to say
    why; on a whole-genome run annotated with gnomAD's exomes only, most variants
    are not coding and gnomAD's exomes do not hold them. The two causes are
    counted apart: absent from gnomAD, and a value that is not a frequency.
    """
    meaning = absent_meaning(sources)
    absent_key = "absent_from_gnomad" if "genomes" in meaning else "absent_from_gnomad_exome"
    return {
        "status_counts": {name: int(status.get(name, 0)) for name in STATUSES},
        "sources": {key: int(value) for key, value in sorted(sources.items())},
        "unknown_bucket": {
            absent_key: int(status.get(STATUS_ABSENT, 0)),
            "unparseable": int(status.get(STATUS_UNPARSEABLE, 0)),
        },
        "meaning": {
            STATUS_ABSENT: meaning,
            STATUS_UNPARSEABLE: UNPARSEABLE_MEANING,
        },
    }


def _open_fragment_writer(path: Path, fieldnames: Sequence[str]) -> tuple[csv.DictWriter, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    return writer, fh


def _parents_called(row: Mapping[str, Any]) -> bool:
    """Whether this row has a called genotype for both parents."""
    return all(str(row.get(column) or ".").strip() not in ("", ".", "./.", ".|.") for column in ("mother_gt", "father_gt"))


def _process_report_shard(job: dict[str, Any]) -> dict[str, Any]:
    shard_path = Path(job["shard_path"])
    out_dir = Path(job["out_dir"])
    fieldnames = list(job["fieldnames"])
    known_genes = set(job.get("known_genes") or [])
    compact_fields = list(BUCKETED_WORKBOOK_COLUMNS)
    shard_id = int(job["shard_id"])
    progress_every = int(job.get("progress_every") or 0)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, csv.DictWriter] = {}
    handles: list[Any] = []
    for key in _FRAGMENT_KEYS:
        writers[key], fh = _open_fragment_writer(out_dir / f"{key}.tsv", fieldnames)
        handles.append(fh)
    bucket_writers: dict[tuple[str, str, str], csv.DictWriter] = {}
    for seg in ("pass", "partial", "fail", "uninformative"):
        for pop in POPULATION_BUCKETS:
            for consequence in CONSEQUENCE_BUCKETS:
                writer, fh = _open_fragment_writer(out_dir / "bucket_tsv" / _bucket_filename(seg, pop, consequence), compact_fields)
                bucket_writers[(seg, pop, consequence)] = writer
                handles.append(fh)

    rows_scanned = 0
    rows_reviewable = 0
    rows_segregating_workbook = 0
    rows_nonsegregating_workbook = 0
    seg_counts: Counter[str] = Counter()
    matrix: Counter[tuple[str, str, str]] = Counter()
    clinical_counts: Counter[str] = Counter()
    list_counts: Counter[str] = Counter()
    bucket_tsv_rows: Counter[str] = Counter()
    population_status: Counter[str] = Counter()
    population_sources: Counter[str] = Counter()
    unrecognized: Counter[str] = Counter()
    # What the design is read from (study_design), and whether any
    # row has both parents called (de_novo_check is written only then).
    scorings: Counter[str] = Counter()
    parents_called = False

    try:
        with shard_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                rows_scanned += 1
                if progress_every and rows_scanned % progress_every == 0:
                    print(f"Candidate report rows processed (shard {shard_id}): {rows_scanned}", flush=True)
                out, seg, pop, consequence, clinical, lists, _de_novo, source_bucket = _classify_candidate_row(row, known_genes)
                _count_population(out, population_status, population_sources)
                scorings[str(row.get("segregation_scoring") or "").strip()] += 1
                if not parents_called and _parents_called(row):
                    parents_called = True
                seg_counts[seg] += 1
                matrix[(seg, pop, consequence)] += 1
                clinical_counts[clinical] += 1
                compact = {key: _safe(out.get(key, ".")) for key in compact_fields}
                if (seg, pop, consequence) in bucket_writers:
                    bucket_writers[(seg, pop, consequence)].writerow(compact)
                    bucket_tsv_rows[source_bucket] += 1
                else:
                    # A segregation bucket the report does not know
                    # has no bucket file; counted, so the invariant check names it.
                    unrecognized[seg] += 1
                clean = None
                if seg in {"pass", "partial"}:
                    rows_reviewable += 1
                    rows_segregating_workbook += 1
                    clean = {key: _safe(out.get(key, ".")) for key in fieldnames}
                    writers["all_reviewable"].writerow(clean)
                elif seg in {"fail", "uninformative"}:
                    rows_nonsegregating_workbook += 1
                for key in lists:
                    if clean is None:
                        clean = {name: _safe(out.get(name, ".")) for name in fieldnames}
                    writers[key].writerow(clean)
                    list_counts[key] += 1
    finally:
        for fh in handles:
            fh.close()

    return {
        "shard_id": shard_id,
        "rows_scanned": rows_scanned,
        "rows_reviewable": rows_reviewable,
        "rows_segregating_workbook": rows_segregating_workbook,
        "rows_nonsegregating_workbook": rows_nonsegregating_workbook,
        "seg_counts": dict(seg_counts),
        "matrix": {"|".join(k): v for k, v in matrix.items()},
        "clinical_counts": dict(clinical_counts),
        "list_counts": dict(list_counts),
        "scorings": dict(scorings),
        "parents_called": bool(parents_called),
        "bucket_tsv_rows": dict(bucket_tsv_rows),
        "population_af_status": dict(population_status),
        "population_af_sources": dict(population_sources),
        "unrecognized_segregation_buckets": dict(unrecognized),
        "fragment_dir": str(out_dir),
    }


def _append_tsv_without_header(src: Path, dst_handle: Any) -> int:
    if not src.exists():
        return 0
    count = 0
    with src.open(newline="") as fh:
        fh.readline()       # the header, written by the caller
        for line in fh:
            dst_handle.write(line)
            count += 1
    return count


def _merge_fragment_files(fragment_dirs: Sequence[Path], output_path: Path, fieldnames: Sequence[str], fragment_name: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        total = 0
        for frag_dir in fragment_dirs:
            total += _append_tsv_without_header(frag_dir / fragment_name, out_fh)
    return total


def _merge_bucket_fragments(fragment_dirs: Sequence[Path], bucket_dir: Path,
                            prefix: str | None = None) -> dict[str, int]:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for seg in ("pass", "partial", "fail", "uninformative"):
        for pop in POPULATION_BUCKETS:
            for consequence in CONSEQUENCE_BUCKETS:
                name = _sheet_name(seg, pop, consequence)
                out_path = bucket_dir / prefixed_name(f"{name}.tsv", prefix)
                with out_path.open("w", newline="") as out_fh:
                    writer = csv.DictWriter(out_fh, fieldnames=list(BUCKETED_WORKBOOK_COLUMNS), delimiter="\t", lineterminator="\n", extrasaction="ignore")
                    writer.writeheader()
                    total = 0
                    for frag_dir in fragment_dirs:
                        total += _append_tsv_without_header(frag_dir / "bucket_tsv" / f"{name}.tsv", out_fh)
                counts[name] = total
    return counts


#: What the bucket files must add up to.
BUCKET_INVARIANT_RULE = (
    "every row of the input table is written to exactly one bucket TSV, the one for its "
    "(segregation bucket, population bucket, consequence bucket); source_bucket is that file's name"
)


def _bucket_invariant(
    *, input_rows: int, rows_scanned: int, bucket_counts: dict[str, int],
    unrecognized: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Rows in == rows in the bucket files, as counted on disk.

    The input's own count is recorded beside the bucket files', and a stage
    whose files do not add up fails instead of finishing.
    """
    bucket_rows = int(sum(int(v) for v in bucket_counts.values()))
    holds = int(input_rows) == bucket_rows == int(rows_scanned)
    return {
        "rule": BUCKET_INVARIANT_RULE,
        "input_rows": int(input_rows),
        "rows_classified": int(rows_scanned),
        "bucket_rows": bucket_rows,
        "unrecognized_segregation_buckets": {k: int(v) for k, v in sorted((unrecognized or {}).items())},
        "holds": bool(holds),
    }


def _refuse_a_broken_invariant(invariant: dict[str, Any], segregation: Path) -> None:
    if invariant["holds"]:
        return
    unknown = invariant.get("unrecognized_segregation_buckets") or {}
    why = (
        " Rows with a segregation_bucket the report does not know (" +
        ", ".join(f"{k!r}: {v}" for k, v in unknown.items()) + ") have no bucket file."
    ) if unknown else ""
    raise RuntimeError(
        "Candidate report bucket check failed: the input table "
        f"{segregation} has {invariant['input_rows']} rows, {invariant['rows_classified']} were classified "
        f"and the bucket TSVs hold {invariant['bucket_rows']}; each input row must be in exactly one "
        f"bucket file.{why} The stage is not marked complete."
    )


#: The files of one report shard's fragment folder (see
#: _process_report_shard), and the names openpyxl gives its spool files.
_FRAGMENT_KEYS = ("all_reviewable", *REPORT_LISTS)
#: The fragment names of a v2 run, removed with this run's.
_FORMER_FRAGMENT_KEYS = ("tier_1", "tier_2", "tier_3", "tier_4", "qc")
_SHARD_FILE = re.compile(r"^shard_\d{4}\.tsv$")
_SHARD_FOLDER = re.compile(r"^shard_\d{4}$")
_OPENPYXL_SPOOL = re.compile(r"^openpyxl\..+")


def _bucket_names(prefix: Optional[str]) -> list[str]:
    return [
        prefixed_name(_bucket_filename(seg, pop, consequence), prefix)
        for seg in ("pass", "partial", "fail", "uninformative")
        for pop in POPULATION_BUCKETS for consequence in CONSEQUENCE_BUCKETS
    ]


def _unlink_own(path: Path, removed: list[str]) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
        removed.append(str(path))


def _rmdir_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass                       # not empty: something not the stage's is in it


def _remove_openpyxl_spool(folder: Path) -> list[str]:
    """openpyxl's spool files in ``folder`` (``openpyxl.*``), then the folder if empty."""
    removed: list[str] = []
    folder = Path(folder)
    if folder.is_dir() and not folder.is_symlink():
        for path in folder.iterdir():
            if _OPENPYXL_SPOOL.match(path.name):
                _unlink_own(path, removed)
        _rmdir_if_empty(folder)
    return removed


def input_record(path: Union[str, Path]) -> dict[str, Any]:
    """What the checkpoint records about the input table: its size and a digest.

    So a segregation table rewritten IN PLACE (an --overwrite rerun after a
    pedigree correction) does not leave the earlier report standing as
    complete. The digest covers the first and last INPUT_DIGEST_BYTES and the
    size, so it costs the same on a 13 M-row table as on a small one; it is
    compared without reading the table.
    """
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "digest": input_digest(path, stat.st_size),
    }


def input_digest(path: Union[str, Path], size: Optional[int] = None) -> str:
    """sha256 over (size, first INPUT_DIGEST_BYTES, last INPUT_DIGEST_BYTES)."""
    path = Path(path)
    if size is None:
        size = path.stat().st_size
    digest = hashlib.sha256(str(int(size)).encode("ascii"))
    with path.open("rb") as fh:
        digest.update(fh.read(INPUT_DIGEST_BYTES))
        if size > INPUT_DIGEST_BYTES:
            fh.seek(max(INPUT_DIGEST_BYTES, size - INPUT_DIGEST_BYTES))
            digest.update(fh.read(INPUT_DIGEST_BYTES))
    return "sha256:" + digest.hexdigest()


def input_unchanged(record: Mapping[str, Any], path: Union[str, Path, None] = None) -> tuple[bool, str]:
    """(True, "") when the table at ``path`` (default: the recorded one) is the recorded one.

    Size first, then the digest. The modification time alone is not held
    against the table (a copy or a touch is not a new result); a different
    size or digest is.
    """
    target = Path(path) if path is not None else Path(str(record.get("path") or ""))
    if not str(target) or not target.is_file():
        return False, f"the input table {target} is not there"
    size = target.stat().st_size
    if int(record.get("bytes", -1)) != size:
        return False, (f"the input table {target} has changed since the report was written "
                       f"({size} bytes now, {record.get('bytes')} when the report was made)")
    recorded = str(record.get("digest") or "")
    if recorded and input_digest(target, size) != recorded:
        return False, f"the input table {target} has changed since the report was written (same size, different content)"
    return True, ""


def report_staleness(checkpoint: Mapping[str, Any]) -> Optional[str]:
    """Why a complete-looking candidate_report checkpoint is not current, or None.

    Two reasons: it was written under an earlier REPORT_METHOD
    (its rows were classified by rules this version has corrected), or the
    input table it was made from is not the one on disk now. A checkpoint
    that names no method made no claim about its method and is judged on its
    schema and its input record.
    """
    if not checkpoint:
        return None
    if "schema" in checkpoint and checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        # A v1 report gave fail/uninformative rows review tiers.
        return (f"the report was written under an earlier method (checkpoint "
                f"{checkpoint.get('schema')}); this Paralome writes {CHECKPOINT_SCHEMA} "
                f"and classifies with {REPORT_METHOD}")
    # A checkpoint that names the current schema but no method (one written by
    # hand, or by a build that recorded only the schema) is judged on its input.
    if "method" in checkpoint and checkpoint.get("method") != REPORT_METHOD:
        return (f"the report was written under {checkpoint.get('method') or 'an earlier method'}; "
                f"this Paralome classifies with {REPORT_METHOD}")
    record = checkpoint.get("input_segregation_record")
    if isinstance(record, dict) and record:
        ok, why = input_unchanged(record)
        if not ok:
            return why
    return None


def _remove_shards_and_fragments(workdir: Path, removed: list[str], *, shards: bool = True) -> None:
    """The input shards and shard fragments of the parallel path, by name.

    Called at the start of a run (_start_clean) and, once the bucket
    invariant holds, at the end of a parallel run with ``shards=False``: the
    fragments are a full copy of every output and serve nothing after the
    merge. The input shards (inputs/report_shards) are kept at the end of a
    run: they are what shows how the table was split (a region-sharded run is
    checked against them), and the next run replaces them. BY NAME, NEVER BY
    FOLDER, as _start_clean says.
    """
    shard_dir = workdir / "inputs" / "report_shards"
    if shards and shard_dir.is_dir() and not shard_dir.is_symlink():
        for path in shard_dir.iterdir():
            if _SHARD_FILE.match(path.name):
                _unlink_own(path, removed)
        _rmdir_if_empty(shard_dir)
    fragment_root = workdir / "results" / "shard_fragments"
    if fragment_root.is_dir() and not fragment_root.is_symlink():
        for folder in fragment_root.iterdir():
            if not (_SHARD_FOLDER.match(folder.name) and folder.is_dir() and not folder.is_symlink()):
                continue
            for key in (*_FRAGMENT_KEYS, *_FORMER_FRAGMENT_KEYS):
                _unlink_own(folder / f"{key}.tsv", removed)
            buckets = folder / "bucket_tsv"
            if buckets.is_dir() and not buckets.is_symlink():
                for name in _bucket_names(None):
                    _unlink_own(buckets / name, removed)
                _rmdir_if_empty(buckets)
            _rmdir_if_empty(folder)
        _rmdir_if_empty(fragment_root)


def _start_clean(workdir: Path, paths: dict[str, Path], prefix: Optional[str] = None) -> list[str]:
    """A run starts from none of the stage's own earlier outputs.

    The run replaces exactly these -- the checkpoint (so a failure cannot
    leave an earlier "complete" standing over half-replaced files), every
    named output of this prefix, the bucket TSVs, the input shards, the shard
    fragments and openpyxl's spool -- and nothing else.

    BY NAME, NEVER BY FOLDER: a file the person has put in one of these
    folders must survive. Each folder loses only the files this stage writes
    there, under the names it writes them (the 64 bucket TSVs with this run's
    prefix; shard_NNNN.tsv; each fragment folder's own files; openpyxl.*), and
    is removed only when that leaves it empty.
    """
    removed: list[str] = []
    files = [stage_checkpoint_path(workdir, "candidate_report")]
    files += [path for key, path in paths.items() if key != "bucket_tsv_dir"]
    # The lists and client workbook of either design, and the v2 names.
    files += [workdir / "results" / prefixed_name(name, prefix) for name in _every_output_name()]
    for path in files:
        _unlink_own(Path(path), removed)
    bucket_dir = Path(paths["bucket_tsv_dir"])
    if bucket_dir.is_dir() and not bucket_dir.is_symlink():
        for name in _bucket_names(prefix):
            _unlink_own(bucket_dir / name, removed)
    _remove_shards_and_fragments(workdir, removed)
    removed += _remove_openpyxl_spool(openpyxl_temp_dir(workdir))
    if removed:
        print(f"Candidate report: replacing {len(removed)} output(s) of an earlier run", flush=True)
    return removed


def _split_report_input(
    *,
    segregation: Path,
    shard_dir: Path,
    shards: int,
    regions: Sequence[_Region],
) -> tuple[list[Path], list[str], list[int]]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    regions_by_chrom = _region_indexes_by_chrom(regions)
    with segregation.open(newline="") as in_fh:
        reader = csv.DictReader(in_fh, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Segregation table has no header: {segregation}")
        input_fields = list(reader.fieldnames)
        shard_paths = [shard_dir / f"shard_{idx:04d}.tsv" for idx in range(shards)]
        handles = [path.open("w", newline="") for path in shard_paths]
        writers = [csv.DictWriter(fh, fieldnames=input_fields, delimiter="\t", lineterminator="\n", extrasaction="ignore") for fh in handles]
        counts = [0 for _ in range(shards)]
        try:
            for writer in writers:
                writer.writeheader()
            for idx, row in enumerate(reader):
                if regions:
                    shard_idx = _assign_region_shard(row, regions_by_chrom, shards, idx)
                else:
                    shard_idx = idx % shards
                writers[shard_idx].writerow(row)
                counts[shard_idx] += 1
        finally:
            for fh in handles:
                fh.close()
    return shard_paths, input_fields, counts


def _gather(fragment_dirs: Sequence[Path], fragment_name: str, output_path: Path, fieldnames: Sequence[str]) -> None:
    """One output from its shard fragments: moved when there is one, merged otherwise."""
    if len(fragment_dirs) == 1 and (Path(fragment_dirs[0]) / fragment_name).is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(Path(fragment_dirs[0]) / fragment_name, output_path)
        return
    _merge_fragment_files(fragment_dirs, output_path, fieldnames, fragment_name)


def _sort_by_carriers(path: Path) -> None:
    """Rows of a list TSV, most affected carriers first; the file's order kept within a count.

    For the cohort's rare damaging list. One pass records each row's
    offset and count; the rows are then written in that order to a file beside
    it, which replaces it. Memory is two integers a row, not the rows.
    """
    offsets: list[int] = []
    counts: list[int] = []
    with path.open("rb") as fh:
        header = fh.readline()
        names = header.decode("utf-8").rstrip("\n").split("\t")
        if "n_affected_carriers" not in names:
            return
        at = names.index("n_affected_carriers")
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            fields = line.rstrip(b"\n").split(b"\t")
            try:
                count = int(fields[at])
            except (IndexError, ValueError):
                count = 0
            offsets.append(offset)
            counts.append(count)
    order = sorted(range(len(offsets)), key=lambda i: -counts[i])
    if order == list(range(len(offsets))):
        return
    sorted_path = path.with_name(path.name + ".sorting")
    with path.open("rb") as src, sorted_path.open("wb") as out:
        out.write(header)
        for i in order:
            src.seek(offsets[i])
            out.write(src.readline())
    os.replace(sorted_path, path)


def write_candidate_review_report(
    *,
    segregation: Union[str, Path],
    workdir: Union[str, Path],
    disease_genes: Union[str, Sequence[str], None] = None,
    candidate_gene_file: Union[str, Path, None] = None,
    workbook_prefix: Optional[str] = None,
    threads: int = 1,
    shards: int = 1,
    progress_every: int = 100000,
    regions: Union[str, Path, None] = None,
    region_convention: Optional[str] = None,
) -> CandidateReportResult:
    """The candidate report of a segregation table.

    One implementation: a run with one shard and no region map reads the
    input table itself as its only shard, with nothing copied, and its
    fragments are moved into place rather than merged.

    The study design (study_design, from the table's segregation_scoring)
    decides which lists are written and what they are called (DESIGN_LISTS).
    No workbooks -- the lists are the TSVs, and OmicView's
    variant browser reads the run's segregation table.
    """
    from paralome.pipeline.region_shards import check_region_convention

    segregation = Path(segregation)
    workdir = Path(workdir)
    region_convention = check_region_convention(region_convention)
    if not segregation.exists():
        raise FileNotFoundError(f"Segregation table not found: {segregation}")
    threads = max(int(threads or 1), 1)
    shards = max(int(shards or 1), 1)
    # Default to the project, not to the product name: resolved ONCE, validated, and reused for every
    # output.
    prefix = require_file_prefix(
        (workbook_prefix or "").strip() or derive_file_prefix(workdir),
        stage="candidate_report",
    )
    print(stage_banner("candidate_report", prefix, workdir), flush=True)
    paths = _output_paths(workdir, prefix)
    for name in ("inputs", "results", "checkpoints"):
        (workdir / name).mkdir(parents=True, exist_ok=True)
    # The stage's own outputs of an earlier (or killed) run go first.
    _start_clean(workdir, paths, prefix)
    # What was already here, so the post-write check can tell this run's
    # mistakes from a previous convention's leftovers.
    _preexisting = snapshot_results(workdir / "results")
    # The input as it is now, for the checkpoint (see input_record).
    segregation_record = input_record(segregation)

    known_genes = parse_gene_set(disease_genes) | read_gene_file(candidate_gene_file)
    report_regions = read_report_regions(regions, region_convention)
    shard_dir = workdir / "inputs" / "report_shards"
    fragment_root = workdir / "results" / "shard_fragments"
    split = bool(regions) or shards > 1
    execution_mode = "parallel_region_sharded" if (split or threads > 1) else "serial"

    if split:
        print(f"Candidate report stratification: launching {min(threads, shards)} workers over {shards} shards")
        if regions:
            print(f"Candidate report stratification: region map={regions} usable_regions={len(report_regions)}")
        print("Splitting segregation table into report shards...")
        shard_paths, input_fields, shard_counts = _split_report_input(
            segregation=segregation, shard_dir=shard_dir, shards=shards, regions=report_regions)
    else:
        with segregation.open(newline="") as fh:
            header = next(csv.reader(fh, delimiter="\t"), None)
        if not header:
            raise ValueError(f"Segregation table has no header: {segregation}")
        shard_paths, input_fields, shard_counts = [segregation], list(header), None
    fieldnames = _make_fieldnames(input_fields)
    _write_json(
        workdir / "inputs" / "candidate_report_plan.json",
        {
            "schema": "paralome.candidate_report.plan.v2",
            "stage": "candidate_report",
            "created": _now(),
            "input_segregation": str(segregation),
            "workdir": str(workdir),
            "disease_genes": sorted(known_genes),
            "candidate_gene_file": str(candidate_gene_file) if candidate_gene_file else None,
            "workbook_prefix": workbook_prefix,
            "threads": int(threads),
            "shards": int(shards),
            "progress_every": int(progress_every),
            "regions": str(regions) if regions else None,
            "region_convention": region_convention if regions else None,
            "usable_regions": len(report_regions),
            "shard_row_counts": shard_counts,
            "execution_mode": execution_mode,
        },
    )

    jobs = [
        {
            "shard_id": idx,
            "shard_path": str(shard_path),
            "out_dir": str(fragment_root / f"shard_{idx:04d}"),
            "fieldnames": fieldnames,
            "known_genes": sorted(known_genes),
            "progress_every": int(progress_every) if not split else 0,
        }
        for idx, shard_path in enumerate(shard_paths)
    ]
    shard_results: list[dict[str, Any]] = []
    if threads == 1 or len(jobs) == 1:
        for i, job in enumerate(jobs, start=1):
            shard_results.append(_process_report_shard(job))
            if progress_every != 0 and split:
                print(f"Report shard {i}/{len(jobs)} complete")
    else:
        with ProcessPoolExecutor(max_workers=min(threads, len(jobs))) as pool:
            futures = [pool.submit(_process_report_shard, job) for job in jobs]
            complete = 0
            for future in as_completed(futures):
                shard_results.append(future.result())
                complete += 1
                if progress_every != 0:
                    print(f"Report shard {complete}/{len(jobs)} complete")
    shard_results.sort(key=lambda x: int(x["shard_id"]))
    fragment_dirs = [Path(x["fragment_dir"]) for x in shard_results]

    rows_scanned = sum(int(x["rows_scanned"]) for x in shard_results)
    rows_reviewable = sum(int(x["rows_reviewable"]) for x in shard_results)
    rows_segregating_workbook = sum(int(x["rows_segregating_workbook"]) for x in shard_results)
    rows_nonsegregating_workbook = sum(int(x["rows_nonsegregating_workbook"]) for x in shard_results)
    seg_counts: Counter[str] = Counter()
    matrix: Counter[tuple[str, str, str]] = Counter()
    clinical_counts: Counter[str] = Counter()
    list_counts: Counter[str] = Counter()
    scorings: Counter[str] = Counter()
    population_status: Counter[str] = Counter()
    population_sources: Counter[str] = Counter()
    unrecognized: Counter[str] = Counter()
    parents_called = False
    for result in shard_results:
        population_status.update({k: int(v) for k, v in result.get("population_af_status", {}).items()})
        population_sources.update({k: int(v) for k, v in result.get("population_af_sources", {}).items()})
        unrecognized.update({k: int(v) for k, v in result.get("unrecognized_segregation_buckets", {}).items()})
        seg_counts.update({k: int(v) for k, v in result.get("seg_counts", {}).items()})
        clinical_counts.update({k: int(v) for k, v in result.get("clinical_counts", {}).items()})
        list_counts.update({k: int(v) for k, v in result.get("list_counts", {}).items()})
        scorings.update({k: int(v) for k, v in result.get("scorings", {}).items()})
        parents_called = parents_called or bool(result.get("parents_called"))
        for key, value in result.get("matrix", {}).items():
            seg, pop, cons = key.split("|", 2)
            matrix[(seg, pop, cons)] += int(value)

    # The design, and so which lists are written and under what names.
    design, design_reason = study_design(scorings)
    not_written: dict[str, str] = {}
    if not known_genes:
        not_written["candidate_genes"] = "no candidate genes were given (--candidate-genes)"
    # A table with mother_gt/father_gt: written when both parents are called on some row. An older table
    # has only segregation's de novo calls, which it made with the parents: written when it has one.
    parent_columns = {"mother_gt", "father_gt"} <= set(input_fields)
    has_parents = parents_called if parent_columns else bool(list_counts.get("de_novo"))
    if design == DESIGN_FAMILY and not has_parents:
        not_written["de_novo"] = "no variant has a called genotype for both parents"
    if design == DESIGN_COHORT:
        not_written["nonsegregating_rare_damaging"] = "a cohort has no family segregation to fail"
        not_written["de_novo"] = "a cohort has no parents"
    print(f"Candidate report: {design} design ({design_reason})", flush=True)

    print("Gathering the report lists...")
    _gather(fragment_dirs, "all_reviewable.tsv", paths["all_reviewable_tsv"], fieldnames)
    lists: dict[str, Path] = {}
    for key in DESIGN_LISTS[design]:
        if key in not_written:
            continue
        lists[key] = _list_path(workdir, prefix, design, key)
        _gather(fragment_dirs, f"{key}.tsv", lists[key], fieldnames)
    if design == DESIGN_COHORT and "rare_damaging" in lists:
        _sort_by_carriers(lists["rare_damaging"])
    if len(fragment_dirs) == 1:
        moved = {bucket: Path(fragment_dirs[0]) / "bucket_tsv" / f"{bucket}.tsv" for bucket in
                 (_sheet_name(seg, pop, cons) for seg in ("pass", "partial", "fail", "uninformative")
                  for pop in POPULATION_BUCKETS for cons in CONSEQUENCE_BUCKETS)}
        paths["bucket_tsv_dir"].mkdir(parents=True, exist_ok=True)
        for bucket, source in moved.items():
            os.replace(source, paths["bucket_tsv_dir"] / prefixed_name(f"{bucket}.tsv", prefix))
    else:
        _merge_bucket_fragments(fragment_dirs, paths["bucket_tsv_dir"], prefix)
    bucket_tsv_counts = _write_bucket_tsv_summary(
        paths["bucket_tsv_dir"], paths["bucket_tsv_summary_csv"], prefix)

    segregation_summary_rows = [{"segregation_bucket": key, "count": str(val)} for key, val in sorted(seg_counts.items())]
    _write_csv_rows(paths["segregation_summary_csv"], ["segregation_bucket", "count"], segregation_summary_rows)
    matrix_rows = []
    for seg in ("pass", "partial", "fail", "uninformative"):
        for pop in POPULATION_BUCKETS:
            for cons in CONSEQUENCE_BUCKETS:
                matrix_rows.append({"segregation_bucket": seg, "population_bucket": pop, "protein_consequence_bucket": cons, "count": str(matrix.get((seg, pop, cons), 0))})
    _write_csv_rows(paths["matrix_csv"], ["segregation_bucket", "population_bucket", "protein_consequence_bucket", "count"], matrix_rows)

    # Rows in, rows in the bucket files (counted on disk), checked
    # before anything is built on them. Unsplit, the table was read once, as
    # its only shard, and its row count is the rows read.
    input_rows = sum(int(x) for x in shard_counts) if shard_counts is not None else rows_scanned
    invariant = _bucket_invariant(
        input_rows=input_rows, rows_scanned=rows_scanned,
        bucket_counts=bucket_tsv_counts, unrecognized=dict(unrecognized),
    )
    population_notes = population_af_summary(population_status, population_sources)
    if not invariant["holds"]:
        _write_json(paths["summary_json"], {
            "schema": "paralome.candidate_report.summary.v2", "stage": "candidate_report",
            "created": _now(), "input_segregation": str(segregation), "complete": False,
            "bucket_invariant": invariant, "shard_row_counts": shard_counts,
        })
        _refuse_a_broken_invariant(invariant, segregation)
    # The outputs hold every row (the invariant just said so); the
    # fragments left are copies, or lists this design does not write, and go.
    _scratch_removed: list[str] = []
    _remove_shards_and_fragments(workdir, _scratch_removed, shards=False)
    if _scratch_removed:
        print(f"Candidate report: removed {len(_scratch_removed)} shard fragment file(s) after the merge", flush=True)

    list_rows = {key: int(list_counts.get(key, 0)) for key in lists}
    outputs = {k: str(v) for k, v in paths.items()}
    outputs.update({f"list:{key}": str(path) for key, path in lists.items()})
    report = {
        "method": REPORT_METHOD,
        "input_segregation": str(segregation),
        "input_segregation_record": segregation_record,
        "design": design,
        "design_reason": design_reason,
        "segregation_scoring_counts": {k: int(v) for k, v in sorted(scorings.items())},
        "parents_called": bool(parents_called),
        "lists": {key: str(path) for key, path in lists.items()},
        "list_rows": list_rows,
        "lists_not_written": dict(not_written),
        "rows_scanned": rows_scanned,
        "rows_reviewable": rows_reviewable,
        "rows_segregating_workbook": rows_segregating_workbook,
        "rows_nonsegregating_workbook": rows_nonsegregating_workbook,
        "bucketed_workbook_rows": {"segregating": int(rows_segregating_workbook), "nonsegregating": int(rows_nonsegregating_workbook)},
        "bucket_tsv_rows": {k: int(v) for k, v in sorted(bucket_tsv_counts.items())},
        "bucket_tsv_total_rows": int(sum(bucket_tsv_counts.values())),
        "threads": int(threads),
        "shards": int(shards),
        "regions": str(regions) if regions else None,
        "region_convention": region_convention if regions else None,
        "usable_regions": len(report_regions),
        "execution_mode": execution_mode,
        "bucket_invariant": invariant,
    }
    summary = {
        "schema": "paralome.candidate_report.summary.v2",
        "stage": "candidate_report",
        "created": _now(),
        **report,
        "segregation_counts": {k: int(v) for k, v in sorted(seg_counts.items())},
        "list_counts": {k: int(v) for k, v in sorted(list_counts.items())},
        "clinical_counts": {k: int(v) for k, v in sorted(clinical_counts.items())},
        "population_consequence_matrix": {"|".join(key): int(val) for key, val in sorted(matrix.items())},
        "disease_genes": sorted(known_genes),
        "candidate_gene_file": str(candidate_gene_file) if candidate_gene_file else None,
        "workbook_prefix": workbook_prefix,
        "shard_row_counts": shard_counts,
        "shard_results": shard_results,
        "population_af": population_notes,
        "outputs": outputs,
    }
    _write_json(paths["summary_json"], summary)
    _write_json(paths["audit_manifest_json"], {**summary, "manifest_created": _now()})
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "candidate_report",
        "complete": True,
        **report,
        "outputs": outputs,
        "finished": _now(),
    }
    # The files that get emailed carry the family ID. Confirm
    # on disk before the checkpoint claims this stage is done.
    assert_outputs_are_prefixed(workdir / "results", prefix, stage="candidate_report",
                                preexisting=_preexisting)
    _write_json(stage_checkpoint_path(workdir, "candidate_report"), checkpoint)
    return CandidateReportResult(
        workdir=workdir,
        input_segregation=segregation,
        summary_json=paths["summary_json"],
        audit_manifest_json=paths["audit_manifest_json"],
        segregation_summary_csv=paths["segregation_summary_csv"],
        matrix_csv=paths["matrix_csv"],
        all_reviewable_tsv=paths["all_reviewable_tsv"],
        design=design,
        lists=dict(lists),
        bucketed_workbook_rows={"segregating": int(rows_segregating_workbook), "nonsegregating": int(rows_nonsegregating_workbook)},
        bucket_tsv_dir=paths["bucket_tsv_dir"],
        bucket_tsv_rows={k: int(v) for k, v in sorted(bucket_tsv_counts.items())},
        threads=int(threads),
        shards=int(shards),
        rows_scanned=rows_scanned,
        rows_reviewable=rows_reviewable,
        complete=True,
        message="candidate report complete",
    )


def verify_candidate_review_report(workdir: Union[str, Path]) -> dict[str, Any]:
    workdir = Path(workdir)
    checkpoint_path = stage_checkpoint_path(workdir, "candidate_report")
    if not checkpoint_path.exists():
        return {"complete": False, "checkpoint": {"path": str(checkpoint_path), "exists": False}}
    checkpoint = json.loads(checkpoint_path.read_text())
    outputs = checkpoint.get("outputs") or {}
    # V1 reports gave fail/uninformative rows review tiers; rerun.
    current = checkpoint.get("schema") == CHECKPOINT_SCHEMA
    files = {}
    for key, value in outputs.items():
        path = Path(value)
        files[key] = {
            "path": str(path),
            "exists": path.exists(),
            "nonempty": path.exists() and path.stat().st_size > 0,
            "size": path.stat().st_size if path.exists() else 0,
        }
    required = [
        "summary_json",
        "audit_manifest_json",
        "segregation_summary_csv",
        "matrix_csv",
        "all_reviewable_tsv",
        "bucket_tsv_summary_csv",
    ]
    complete = bool(checkpoint.get("complete")) and all(files.get(k, {}).get("nonempty") for k in required)
    # And current -- the same schema, the same method, the same input table.
    stale = report_staleness(checkpoint) if complete else None
    if stale or not current:
        complete = False
    return {
        "complete": complete,
        "stale": stale,
        "method": checkpoint.get("method"),
        "input_segregation_record": checkpoint.get("input_segregation_record"),
        "checkpoint": {"path": str(checkpoint_path), "exists": True, "schema": checkpoint.get("schema"), "current": current},
        "rows_scanned": checkpoint.get("rows_scanned"),
        "rows_reviewable": checkpoint.get("rows_reviewable"),
        "rows_segregating_workbook": checkpoint.get("rows_segregating_workbook"),
        "rows_nonsegregating_workbook": checkpoint.get("rows_nonsegregating_workbook"),
        "bucketed_workbook_rows": checkpoint.get("bucketed_workbook_rows"),
        "bucket_tsv_rows": checkpoint.get("bucket_tsv_rows"),
        "bucket_tsv_total_rows": checkpoint.get("bucket_tsv_total_rows"),
        "threads": checkpoint.get("threads"),
        "shards": checkpoint.get("shards"),
        "regions": checkpoint.get("regions"),
        "usable_regions": checkpoint.get("usable_regions"),
        "execution_mode": checkpoint.get("execution_mode"),
        "outputs": files,
    }
