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

"""The gene_burden stage: probands against gnomAD, gene by gene.

Inputs: the annotated VCF (annotation's output), the sample sheet, and the
genome's burden reference (burden/reference.py, built once per genome by
``paralome setup genome burden-reference NAME``). Outputs, under
``<workdir>/results``:

    <prefix>_burden_<model>_<tier>.tsv          one per model x tier run
    <prefix>_ancestry.tsv
    <prefix>_burden_summary.json
    <prefix>_evidome_input.tsv                  the coding tiers, for Evidome

and the checkpoint ``checkpoints/gene_burden.json``.

THE PER-GENE TABLES add ``n_tested`` (probands in the
gene's test: on chrX/chrY outside the PARs a female on Y, or a proband of
unknown sex, is not), ``flag`` ('absent_recurrent' when a carrier rests only on
an allele absent from gnomAD that at least ``RECURRENT_ABSENT_MIN_PROBANDS``
probands carry), ``carriers_from_absent_alleles`` (carriers who would not count
without the alleles absent from gnomAD) and ``recurrent_absent_alleles``
(variant(probands), ...); recessive tables list ``hemizygous`` males beside
``possible_compound_het``. A gene with no proband in its test is left out. The
recessive expectation is P(>= 2 qualifying alleles) (burden/stats.py), with the
reference's s2 columns when it has them. Sex: the sheet's ``sex`` column, else
inferred from chrX (burden/ancestry.py), reported in the ancestry table. The
summary records the cohort-wide count of absent and recurrent absent alleles,
and the annotation provenance of both sides (the VCF's ANNOVAR_DATE values and
ANNOVAR header lines; the reference's protocol) so a mismatch is visible --
nothing checks they agree.

THE EVIDOME INPUT (contract shared with evidome.wide_stats): per comparison
``{C}_Log2_fold_change``, ``{C}_P-value``, ``{C}_Q-value`` and
``{C}_Null_selection_probability`` -- the probability that the gene reaches
p < --alpha (default 0.05; Paralome passes the same --alpha to Evidome) under the null, from the same Poisson-binomial
as its p; Evidome weights its burden ORA null by it. Flags are not written
there: the contract has no column for them.

SKIPPED, NOT FAILED (status 'skipped', reason recorded; the pipeline goes on,
and functional_evidence for DNA is skipped with it): no burden reference for
the genome (the reason names the command that builds it), fewer than
``MIN_PROBANDS`` probands, a species gnomAD does not describe.

SELF-CHECKING. Issued again, the stage compares what it is given -- the VCF,
the sheet's probands and ancestry values, the reference, the options -- with
the request in its checkpoint. Unchanged, with every recorded output present:
the result is kept and nothing is written. Otherwise it computes again and
replaces the files it records; it never deletes anything.

The regulatory tier runs only when the VCF carries gnomAD GENOME frequencies
(``gnomad41_genome_AF_grpmax``) AND the reference has a regulatory table;
otherwise it is recorded as not run, with the reason. It is never mixed into
the coding tiers and never written to the Evidome input. Its test is an
allele count against 2Nq with a Poisson tail, not the carrier test of the
coding tiers (burden/stats.py says why); its tables add ``n_alleles``, list
every proband carrying a qualifying allele, and their ``expected`` is alleles.

* PROTEIN-CODING GENES ONLY. Every tier tests only genes the genome's GTF
  calls protein_coding (gene_biotype, or GENCODE's gene_type), so novel and
  read-through Ensembl genes ("BARHL2,LINC02609") whose qualifying alleles
  are their neighbours' do not lead the tables. A
  gene the GTF does not list is left out too (it cannot be shown to be
  protein-coding); both counts are in the summary. A GTF with no biotype at
  all filters nothing, and the summary says so. Without a GTF (no --gtf and
  no gtf in the genome entry) nothing is filtered, and the summary says so.
* GENOMIC CONTROL FOR THE REGULATORY TIER. Its allele-count test is
  inflated on a whole genome (linkage, the reference's genome AFs against a
  joint call). The tables keep ``p`` and ``q`` and add ``p_gc`` -- the
  chi-square (1 df) of p divided by lambda (stats.genomic_inflation), when
  lambda > 1 -- and ``q_gc`` (BH of p_gc). The viewer draws the
  Manhattan and QQ from the tables (no figure is written here). The coding tiers are not
  corrected (their exact test is conservative, and lambda for them is biased
  low). The regulatory tier is still never written to the Evidome input.
* THE SCAN RUNS BY CONTIG IN PARALLEL (``--threads``, cohort.scan_cohort),
  and reports its progress.
* ALLELES TOO COMMON FOR GNOMAD ARE DROPPED (burden/artefacts.py).
  Before any gene is scored, an allele whose number of carrier probands is
  too unlikely for its gnomAD frequency (P < 1e-6) is taken out of the
  carriers and, where gnomAD has it, out of the gene's expectation. The
  tables add ``carriers_from_overrepresented_alleles`` and
  ``overrepresented_alleles``; the summary lists every allele dropped.
* AN ALLELE OUTSIDE THE EXOME CAPTURE IS NOT "ABSENT". The coding
  tiers judge frequency on gnomAD's exome table; an allele that table lacks
  but the genome table has is judged on the genome table.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from paralome.burden import artefacts, qualify
from paralome.burden import stats as burden_stats
from paralome.burden.ancestry import (
    ANCESTRY_COLUMNS,
    AUTO,
    MIN_SEX_SNPS,
    SEX_F_FEMALE,
    SEX_F_MALE,
    ancestry_rows,
    assign_sexes,
    check_ancestry_value,
    estimate_ancestry,
)
from paralome.burden.cohort import (
    DEFAULT_FILTERS,
    GENOME_GRPMAX_KEY,
    REQUIRED_INFO_KEYS,
    carrier_status,
    in_sex_aware_test,
    read_vcf_header,
    scan_cohort,
    select_probands,
)
from paralome.checkpoints import atomic_text_writer, atomic_write_text, stage_checkpoint_path
from paralome.errors import ParalomeError
from paralome.stats import benjamini_hochberg

STAGE = "gene_burden"
SCHEMA = "paralome.gene_burden.checkpoint.v1"
#: The method of this stage's results, recorded in its checkpoint; a result made under another
#: method is computed again. v2: recessive P(>= 2 qualifying alleles) with s2; hemizygous males;
#: absent-allele flags; overlapping alleles one event; the Evidome input's null selection probability.
#: v3: protein-coding genes only; genomic control (p_gc, q_gc) for the regulatory tier; split
#: (atomized) calls from the annotation stage, alleles of one origin one event.
#: v4: allele balance skipped only for the halves of a two-ALT call, tested for every other indel and
#: piece; one event per haplotype (genotype slot) of a split record; complex alleles not atomized.
#: v5: gene names that are symbols mapped to Ensembl ids.
#: v6: the annotation split joins same-allele pieces of two ALTs.
#: v7: the halves of a 1/2 call tested against the other ALT's reads; a 1/2 indel pair (overlapping,
#: different REFs) two events in trans.
#: v8: a duplicate record's carriers read from the first copy that passes QUAL and the genotype
#: filters, per proband; no stale regulatory table when the reference records the tier absent.
#: v9: probands, families, sex and ancestry read under any accepted id-column alias ('sample', 'id', 'name').
#: v10: alleles too common in the cohort for gnomAD dropped; an allele outside the exome capture judged
#: on gnomAD's genome table, not read as absent.
METHOD = "paralome.gene_burden.v10"
#: The most worker processes the scan uses, whatever --threads says.
MAX_SCAN_THREADS = 16
#: GTF biotype values that make a gene protein-coding.
PROTEIN_CODING = "protein_coding"
MIN_PROBANDS = 5
#: Species gnomAD describes. Anything else is skipped: there is no reference to compare with.
GNOMAD_SPECIES = frozenset({"homo sapiens", "human"})

#: The four comparisons Evidome is given, in this order (pD contract).
EVIDOME_COMPARISONS = tuple(f"{model}_{tier}" for model in qualify.MODELS for tier in qualify.CODING_TIERS)
EVIDOME_HEADER = "# {name}: control=gnomAD, treatment=probands, test=burden"

TABLE_COLUMNS = ("gene_id", "gene_name", "chrom", "start", "end", "n_probands", "n_tested", "probands",
                 "expected", "log2_obs_exp", "p", "q", "flag", "carriers_from_absent_alleles",
                 "recurrent_absent_alleles", "carriers_from_overrepresented_alleles", "overrepresented_alleles")
#: Probands counted with one copy (males outside the PARs) are listed too.
RECESSIVE_EXTRA = ("possible_compound_het", "hemizygous")
#: An allele absent from gnomAD seen in at least this many probands is "recurrent" (a caller
#: artefact looks exactly like this; so could the finding of a single-disease cohort -- flagged, never dropped).
RECURRENT_ABSENT_MIN_PROBANDS = 3
FLAG_ABSENT_RECURRENT = "absent_recurrent"
#: The Evidome input's optional role (Evidome's wide_stats contract): the probability, under the null,
#: that the gene reaches Evidome's ORA threshold (p < the run's --alpha, passed to Evidome too).
EVIDOME_NULL_ROLE = "Null_selection_probability"
#: The regulatory tier counts alleles (burden/stats.py): its tables
#: carry the allele count; "expected" there is an expected number of alleles.
REGULATORY_EXTRA = ("n_alleles", "p_gc", "q_gc")

#: Excel's limit on the text of one cell. A gene's allele listing
#: (``variants``) longer than this runs on into ``variants_2``, ``variants_3``,
#: ... -- as many as the table's longest listing needs -- cut between two
#: alleles, never inside one. A regulatory-tier listing ran to 471,592
#: characters, and Excel broke the row there.
EXCEL_CELL_CHARS = 32_767
#: What the listing puts between two alleles ("; ") and two probands (" | ").
#: Neither occurs inside an allele's label: ANNOVAR's multi-value fields
#: (AlphaMissense "P;B") carry no space after the semicolon.
_VARIANTS_SEPARATOR = re.compile(r"(; | \| )")

REFERENCE_REQUIRED = ("gene_id", "gene_name", "chrom", "start", "end", "tier", "model",
                      "n_qualifying_alleles") + tuple(f"q_{p}" for p in qualify.POPULATIONS_AND_ALL)


def _s2_keys() -> tuple:
    from paralome.burden.reference import S2_KEYS

    return S2_KEYS


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_reference_command(genome: Optional[str]) -> str:
    try:
        from paralome.burden.reference import build_command

        return build_command(genome or "NAME")
    except Exception:  # pragma: no cover - reference.py is part of the package
        return f"paralome setup genome burden-reference {genome or 'NAME'}"


def default_prefix(workdir: Path) -> str:
    """The project's name, derived as every other stage derives it: ``gene_burden``
    is one of naming's generic stage folders, so the walk goes on to the project."""
    from paralome.atac.naming import derive_file_prefix

    return derive_file_prefix(Path(workdir).expanduser().resolve())


# ------------------------------------------------------------------ reference

def regulatory_reference_of(reference: Path) -> Optional[Path]:
    """The regulatory (gnomAD genome) table beside ``reference``, when one was built."""
    provenance = reference.parent / "burden_reference.json"
    try:
        recorded = json.loads(provenance.read_text())
    except (OSError, ValueError):
        recorded = {}
    regulatory = recorded.get("regulatory") if isinstance(recorded, dict) else None
    if isinstance(regulatory, dict) and regulatory.get("status") == "built" and regulatory.get("output"):
        path = Path(str(regulatory["output"]))
        if not path.is_file():
            path = reference.parent / path.name
        if path.is_file():
            return path
    if isinstance(regulatory, dict):
        # The provenance decides; a genome table left by an earlier build is not used.
        return None
    sibling = reference.parent / reference.name.replace(
        qualify.GNOMAD_SOURCES["exome"], qualify.GNOMAD_SOURCES["genome"], 1)
    return sibling if sibling != reference and sibling.is_file() else None


def read_reference(*paths: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """(model, tier) -> gene -> {gene_name, chrom, start, end, n, q: {pop: q}}."""
    table: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = [c for c in REFERENCE_REQUIRED if c not in (reader.fieldnames or [])]
            s2_keys = _s2_keys()
            has_s2 = all(f"s2_{k}" in (reader.fieldnames or []) for k in s2_keys)
            if missing:
                raise ParalomeError(
                    106, f"{path} is not a burden reference table: it lacks {', '.join(missing)}",
                    fix=f"Build it again with `{build_reference_command(None)} --rebuild`",
                )
            for row in reader:
                key = (row["model"], row["tier"])
                try:
                    n = int(row["n_qualifying_alleles"] or 0)
                    q = {pop: float(row[f"q_{pop}"] or 0.0) for pop in qualify.POPULATIONS_AND_ALL}
                except ValueError as exc:
                    raise ParalomeError(106, f"{path}: unreadable number in gene {row.get('gene_id')}: {exc}") from exc
                try:
                    # Second power sums (reference.py); None in a table built before them.
                    s2 = ({k: float(row[f"s2_{k}"] or 0.0) for k in s2_keys} if has_s2 else None)
                except ValueError as exc:
                    raise ParalomeError(106, f"{path}: unreadable number in gene {row.get('gene_id')}: {exc}") from exc
                table.setdefault(key, {})[row["gene_id"]] = {
                    "gene_name": row.get("gene_name") or "",
                    "chrom": row.get("chrom") or "",
                    "start": int(row["start"]) if (row.get("start") or "").strip().isdigit() else None,
                    "end": int(row["end"]) if (row.get("end") or "").strip().isdigit() else None,
                    "n": n,
                    "q": q,
                    "s2": s2,
                }
    return table


def reference_provenance(reference: Path) -> dict[str, Any]:
    """burden_reference.json beside the reference ({} when there is none, e.g. a table made by hand)."""
    try:
        recorded = json.loads((Path(reference).parent / "burden_reference.json").read_text())
    except (OSError, ValueError):
        return {}
    return recorded if isinstance(recorded, dict) else {}


def vcf_annotation_header(vcf: Path, limit: int = 20) -> list[str]:
    """The VCF header lines that say how it was annotated (ANNOVAR's own, and the ensGene INFO lines)."""
    from paralome.burden.cohort import _open_text

    lines: list[str] = []
    with _open_text(Path(vcf)) as handle:
        for line in handle:
            if not line.startswith("##"):
                break
            text = line.rstrip("\n")
            lowered = text.lower()
            if "annovar" in lowered or "ensgene" in lowered:
                lines.append(text)
                if len(lines) >= limit:
                    break
    return lines


# ----------------------------------------------------------- fingerprints

def _describe_file(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {"path": str(p), "exists": False}
    return {"path": str(p.resolve()), "exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _fingerprint(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_checkpoint(workdir: Path) -> dict[str, Any]:
    try:
        data = json.loads(stage_checkpoint_path(workdir, STAGE).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_checkpoint(workdir: Path, checkpoint: Mapping[str, Any]) -> Path:
    path = stage_checkpoint_path(workdir, STAGE)
    atomic_write_text(path, json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    return path


def output_paths(workdir: Path, prefix: str, comparisons) -> dict[str, Path]:
    results = Path(workdir) / "results"
    out: dict[str, Path] = {}
    for name in comparisons:
        out[f"{name}_table"] = results / f"{prefix}_burden_{name}.tsv"
    out["ancestry"] = results / f"{prefix}_ancestry.tsv"
    out["summary"] = results / f"{prefix}_burden_summary.json"
    out["evidome_input"] = results / f"{prefix}_evidome_input.tsv"
    return out


# ------------------------------------------------------------------- the stage

def gene_key(gene_id: str) -> str:
    """An Ensembl gene id without its version: ENSG00000141510.18 -> ENSG00000141510."""
    gene_id = str(gene_id).strip()
    head, dot, tail = gene_id.rpartition(".")
    return head if dot and head and tail.isdigit() else gene_id


def _gtf_attribute(attributes: str, key: str) -> Optional[str]:
    marker = key + " "
    for item in attributes.split(";"):
        item = item.strip()
        if item.startswith(marker):
            return item[len(marker):].strip().strip('"')
    return None


def gtf_biotypes(gtf: Path) -> dict[str, str]:
    """Ensembl gene id (no version) -> biotype, from the GTF's gene lines (any line when it has none)."""
    biotypes: dict[str, str] = {}
    any_line: dict[str, str] = {}
    with open(gtf, "rb") as probe:
        opener = gzip.open if probe.read(2) == b"\x1f\x8b" else open
    with opener(gtf, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            gene_id = _gtf_attribute(parts[8], "gene_id")
            if not gene_id:
                continue
            biotype = _gtf_attribute(parts[8], "gene_biotype") or _gtf_attribute(parts[8], "gene_type") or ""
            if parts[2] == "gene":
                biotypes[gene_key(gene_id)] = biotype
            elif not biotypes:
                any_line.setdefault(gene_key(gene_id), biotype)
    return biotypes or any_line


def protein_coding_filter(gtf: Optional[Path]) -> tuple[Optional[frozenset], dict[str, Any]]:
    """(protein-coding gene ids or None for no filter, what the summary says about it)."""
    if gtf is None:
        return None, {"applied": False, "reason": "no GTF (--gtf, or gtf in the genome entry); every gene tested"}
    gtf = Path(gtf)
    if not gtf.is_file():
        raise ParalomeError(104, f"--gtf {gtf} does not exist")
    biotypes = gtf_biotypes(gtf)
    if not any(biotypes.values()):
        return None, {"applied": False, "gtf": str(gtf),
                      "reason": "the GTF gives no gene_biotype or gene_type; every gene tested"}
    coding = frozenset(g for g, b in biotypes.items() if b == PROTEIN_CODING)
    return coding, {"applied": True, "gtf": str(gtf), "genes_in_gtf": len(biotypes),
                    "protein_coding_genes_in_gtf": len(coding)}


def run_gene_burden(
    *,
    vcf: Path,
    samples: Path,
    workdir: Path,
    reference: Optional[Path] = None,
    prefix: Optional[str] = None,
    ancestry: str = AUTO,
    overwrite: bool = False,
    genome: Optional[str] = None,
    species: Optional[str] = None,
    gtf: Optional[Path] = None,
    threads: int = 1,
    alpha: float = burden_stats.EVIDOME_ORA_ALPHA,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run (or reuse, or skip) the stage. Returns the checkpoint written or kept."""
    from paralome.version import __version__

    workdir = Path(workdir)
    for name in ("inputs", "results", "checkpoints", "logs"):
        (workdir / name).mkdir(parents=True, exist_ok=True)
    run_prefix = (prefix or "").strip() or default_prefix(workdir)
    ancestry = (ancestry or AUTO).strip().lower()
    problem = check_ancestry_value(ancestry)
    if problem:
        raise ParalomeError(108, f"--ancestry {problem}")
    started = now_iso()
    previous = _read_checkpoint(workdir)

    base: dict[str, Any] = {
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "paralome_version": __version__,
        "started": started, "workdir": str(workdir), "prefix": run_prefix, "genome": genome,
        "species": species, "options": {"ancestry": ancestry},
        "inputs": {"vcf": _describe_file(vcf), "samples": _describe_file(samples),
                   "reference": _describe_file(reference), "gtf": _describe_file(gtf)},
    }

    def skip(reason: str, **extra: Any) -> dict[str, Any]:
        checkpoint = {
            **base, **extra, "status": "skipped", "complete": False, "reason": reason,
            "finished": now_iso(), "outputs": {},
            "earlier_outputs_left_in_place": sorted((previous.get("outputs") or {}).values())
            if previous.get("status") == "complete" else [],
        }
        _write_checkpoint(workdir, checkpoint)
        echo(f"gene_burden skipped: {reason}")
        return checkpoint

    if species and species.strip().lower().replace("_", " ") not in GNOMAD_SPECIES:
        return skip(f"gnomAD describes human populations only; this genome's species is {species}")
    if reference is None:
        named = f"genome {genome!r} has" if genome else "this run names no genome (--genome), so it has"
        return skip(
            f"{named} no burden reference. Build it once with: {build_reference_command(genome)} "
            "-- then run the pipeline again; gene_burden and functional_evidence will run."
        )
    reference = Path(reference)
    if not reference.is_file():
        # The preflight says the same as a note (pipeline/preflight.py): a
        # genome whose reference was recorded and has since gone skips the
        # stage; it does not stop the DNA run.
        return skip(
            f"the burden reference recorded for genome {genome or '?'} is not there ({reference}). "
            f"Build it again with: {build_reference_command(genome)} --rebuild"
        )
    for flag, path in (("--vcf", Path(vcf)), ("--samples", Path(samples))):
        if not path.is_file():
            raise ParalomeError(104, f"{flag} {path} does not exist")

    choice = select_probands(Path(samples))
    header = read_vcf_header(Path(vcf))
    absent = [sample for sample in choice.used if sample not in header.samples]
    if absent:
        raise ParalomeError(
            205,
            f"affected sample(s) {', '.join(absent)} in {samples} are not in the VCF {vcf} "
            f"(its samples: {', '.join(header.samples[:10])}{' ...' if len(header.samples) > 10 else ''})",
            fix="Correct the sample ids in the sheet, or comment out the samples that were not called",
        )
    for sample, value in choice.ancestry.items():
        problem = check_ancestry_value(value)
        if problem:
            raise ParalomeError(108, f"the sheet's ancestry for {sample}: {problem}",
                                fix="Use a gnomAD population (afr, amr, asj, eas, fin, mid, nfe, sas), 'auto', or leave it empty")
    missing_keys = [key for key in REQUIRED_INFO_KEYS if key not in header.info_ids]
    if missing_keys:
        raise ParalomeError(
            106,
            f"{vcf} lacks the INFO key(s) {', '.join(missing_keys)}; gene burden needs ANNOVAR's ensGene "
            "annotation and gnomad41_exome frequencies",
            fix="Annotate with a protocol that includes ensGene and gnomad41_exome",
        )

    regulatory_table = regulatory_reference_of(reference)
    provenance = reference_provenance(reference)
    buildver = str(provenance.get("buildver") or "").strip()
    if buildver and buildver.lower() not in burden_stats.PAR_BUILDS:
        raise ParalomeError(
            106,
            f"the burden reference {reference} was built for {buildver}; gene_burden knows the pseudo-autosomal "
            f"regions of {', '.join(sorted(burden_stats.PAR_BUILDS))} only, and needs them to count chrX and chrY "
            "carriers by sex",
            fix="Use a GRCh38 (hg38) genome and its burden reference",
        )
    par = burden_stats.PAR_BUILDS.get(buildver.lower(), burden_stats.PAR_GRCH38)
    par_build = buildver or "GRCh38 (assumed: the reference records no build)"
    probands_record = {
        "used": list(choice.used), "not_used": list(choice.not_used),
        "ancestry_in_sheet": dict(choice.ancestry),
        "sex_in_sheet": dict(choice.sexes),
    }
    request = {
        "method": METHOD,
        "vcf": base["inputs"]["vcf"], "reference": base["inputs"]["reference"],
        "gtf": base["inputs"]["gtf"],
        "regulatory_reference": _describe_file(regulatory_table),
        "probands": probands_record, "ancestry": ancestry, "prefix": run_prefix,
        "filters": DEFAULT_FILTERS.to_dict(),
        "thresholds": {
            "dominant_max_af": qualify.DOMINANT_MAX_AF, "recessive_max_af": qualify.RECESSIVE_MAX_AF,
            "revel_min": qualify.REVEL_MIN, "q_floor": burden_stats.Q_FLOOR,
            "evidome_ora_alpha": float(alpha),
            "recurrent_absent_min_probands": RECURRENT_ABSENT_MIN_PROBANDS,
        },
    }
    fingerprint = _fingerprint(request)
    base["request"] = {"fingerprint": fingerprint, **request}
    base["probands"] = probands_record

    if len(choice.used) < MIN_PROBANDS:
        return skip(f"{len(choice.used)} independent affected proband(s) in the sheet; the burden test "
                    f"needs at least {MIN_PROBANDS}")

    recorded = previous.get("outputs") or {}
    if (
        not overwrite and previous.get("status") == "complete" and previous.get("complete")
        and (previous.get("request") or {}).get("fingerprint") == fingerprint
        and recorded and all(Path(str(p)).is_file() for p in recorded.values())
    ):
        echo(f"gene_burden: the result in {workdir} answers this request (inputs and options unchanged); kept.")
        return previous
    # The earlier result stops reading as complete before any table is rewritten.
    _write_checkpoint(workdir, {
        **base, "status": "incomplete", "complete": False, "outputs": {},
        "reason": f"computing since {started}; a run that stopped here did not finish",
        "earlier_outputs_left_in_place": sorted(str(p) for p in recorded.values())
        if previous.get("status") == "complete" else [],
    })

    # --- the regulatory tier: both halves or not at all.
    if GENOME_GRPMAX_KEY not in header.info_ids:
        regulatory = {"status": "not_run", "reason": "needs gnomad41_genome: the VCF has no "
                      f"{GENOME_GRPMAX_KEY} (ANNOVAR's gnomad41_genome table is not in this genome's protocol)"}
    elif regulatory_table is None:
        regulatory = {"status": "not_run", "reason": "needs gnomad41_genome: the burden reference has no "
                      "regulatory table (rebuild it once hg38_gnomad41_genome.txt is in humandb)"}
    else:
        regulatory = {"status": "run", "reference": str(regulatory_table)}
    run_regulatory = regulatory["status"] == "run"

    reference_table = read_reference(reference, *([regulatory_table] if run_regulatory else []))
    coding_genes, protein_coding = protein_coding_filter(Path(gtf) if gtf else None)
    reference_genes = {gene_key(g) for genes in reference_table.values() for g in genes}
    if coding_genes is not None and reference_genes and not (reference_genes & coding_genes):
        # A GTF keyed by other ids (NCBI RefSeq) would drop every gene as not protein-coding.
        raise ParalomeError(
            106,
            f"the GTF {gtf} shares no gene id with the burden reference {reference} (Ensembl ids, ENSG...); "
            "every gene would be dropped as not protein-coding",
            fix="Pass --gtf with an Ensembl or GENCODE GTF for this genome",
        )
    if coding_genes is not None:
        echo(f"gene_burden: testing the {len(coding_genes):,} protein-coding genes of {gtf}")
    else:
        echo(f"gene_burden: not limited to protein-coding genes: {protein_coding['reason']}")
    echo(f"gene_burden: reading {vcf} for {len(choice.used)} probands"
         + (" (regulatory tier on)" if run_regulatory else ""))
    # Gene names that are symbols. ANNOVAR's hg38 ensGene names most genes by symbol and only the
    # unnamed ones by Ensembl id, so a reference built without mapping symbols to ids would cover only
    # a small fraction of genes: refused.
    if provenance and (provenance.get("gene_mapping") or {}).get("method") != qualify.GENE_MAPPING:
        raise ParalomeError(
            106,
            f"the burden reference {reference} was built by an older Paralome, which dropped every gene "
            "ANNOVAR's ensGene names by symbol (most protein-coding genes); its expectations cover only the "
            "genes named by Ensembl id",
            fix=f"Build it again: {build_reference_command(genome)} --rebuild",
        )
    map_gtf = ((provenance.get("gene_coordinates") or {}).get("gtf") if provenance else None) or (
        str(gtf) if gtf else None)
    annovar_record = (provenance.get("annovar") or {}) if provenance else {}
    symbol_ids, symbol_stats = qualify.gene_symbol_ids(
        map_gtf, qualify.ensgene_table(annovar_record.get("annovar_db"), provenance.get("buildver") if provenance else None))
    echo(f"gene_burden: {len(symbol_ids):,} gene names mapped to Ensembl ids ({map_gtf or 'no GTF'})")
    scan = scan_cohort(Path(vcf), choice.used, regulatory=run_regulatory, par=par,
                       threads=min(MAX_SCAN_THREADS, max(1, int(threads or 1))), echo=echo,
                       symbol_ids=symbol_ids)
    echo(f"gene_burden: {scan.counts['records_read']:,} records read; scoring")
    estimates = estimate_ancestry(
        choice.used, scan.ancestry_dosages, scan.ancestry_afs, option=ancestry, sheet=choice.ancestry,
    )
    weights = [est.weights for est in estimates]
    sex_calls = assign_sexes(choice.used, choice.sexes, scan.sex_dosages, scan.sex_afs)
    sex_of = {call.sample_id: call.sex for call in sex_calls}
    sex_list = [sex_of[sample] for sample in choice.used]
    recurrent_absent = {variant: n for variant, n in scan.absent_allele_probands.items()
                        if n >= RECURRENT_ABSENT_MIN_PROBANDS}
    # Alleles too common in the cohort for their gnomAD frequency (caller artefacts:
    # homopolymer indels, pseudogene reads) are dropped before any gene is scored -- from the carriers and,
    # where gnomAD has them, from the expectation -- and listed in the tables and the summary.
    too_common = artefacts.overrepresented(scan.hits, len(choice.used))
    kept_hits = artefacts.without(scan.hits, too_common)
    if too_common:
        echo(f"gene_burden: {len(too_common):,} alleles dropped as too common for gnomAD "
             f"(P of their carrier count < {artefacts.OVERREPRESENTED_P:g})")

    tiers = qualify.CODING_TIERS + ((qualify.REGULATORY_TIER,) if run_regulatory else ())
    comparisons = [(model, tier) for model in qualify.MODELS for tier in tiers]
    names = [f"{model}_{tier}" for model, tier in comparisons]
    paths = output_paths(workdir, run_prefix, names)

    per_comparison: dict[str, dict[str, Any]] = {}
    evidome_columns: dict[str, dict[str, tuple[float, ...]]] = {}
    gene_names: dict[str, str] = {}
    second_order: dict[str, bool] = {}

    def place(gene: str, ref: Optional[Mapping[str, Any]]) -> tuple[str, Optional[int], Optional[int]]:
        site = scan.genes.get(gene)
        chrom = (ref or {}).get("chrom") or (site.chrom if site else "")
        start = (ref or {}).get("start") if ref and ref.get("chrom") else (site.start if site else None)
        end = (ref or {}).get("end") if ref and ref.get("chrom") else (site.end if site else None)
        return chrom, start, end

    for (model, tier), name in zip(comparisons, names):
        ref_genes = reference_table.get((model, tier), {})
        regulatory_tier = tier == qualify.REGULATORY_TIER
        candidate_genes = sorted({g for g, row in ref_genes.items() if row["n"] > 0}
                                 | set(scan.hits.get(model, {}).get(tier, {})))
        regions = {g: burden_stats.gene_region(*place(g, ref_genes.get(g)), par) for g in candidate_genes}
        def tier_carriers(tree: Optional[dict]) -> dict:
            if not regulatory_tier:
                return scan.carriers(model, tier, sexes=sex_of, regions=regions, hits=tree)
            # Every proband carrying a qualifying allele, counted in alleles (one copy for a male
            # outside the PARs; nobody outside the test).
            out = {}
            for g, by in (scan.hits if tree is None else tree).get(model, {}).get(tier, {}).items():
                kept = {pr: hits for pr, hits in by.items() if hits and in_sex_aware_test(sex_of[pr], regions.get(g))}
                if kept:
                    out[g] = kept
            return out

        # Counted without the alleles too common for gnomAD; with them, for the table's record.
        carriers = tier_carriers(kept_hits)
        carriers_all = tier_carriers(None)
        source = artefacts.source_of(tier)
        dropped_in = {}
        for g, by in scan.hits.get(model, {}).get(tier, {}).items():
            found = {(source, h.variant) for hits in by.values() for h in hits} & too_common.keys()
            if found:
                dropped_in[g] = sorted((too_common[key] for key in found), key=lambda a: a.variant)
        ref_genes = {g: (artefacts.expectation_less(row, dropped_in[g]) if g in dropped_in else row)
                     for g, row in ref_genes.items()}
        universe_ids = sorted({g for g, row in ref_genes.items() if row["n"] > 0} | set(carriers))
        not_coding = []
        if coding_genes is not None:
            not_coding = [g for g in universe_ids if gene_key(g) not in coding_genes]
            dropped = set(not_coding)
            universe_ids = [g for g in universe_ids if g not in dropped]
            carriers = {g: by for g, by in carriers.items() if g not in dropped}
        universe = {
            gene: (ref_genes[gene]["q"] if gene in ref_genes else {p: 0.0 for p in qualify.POPULATIONS_AND_ALL})
            for gene in universe_ids
        }
        has_s2 = bool(ref_genes) and all(row.get("s2") is not None for row in ref_genes.values())
        second_order[name] = has_s2
        if regulatory_tier:
            observed_alleles = {
                g: burden_stats.allele_count(by, {pr: (1 if (regions.get(g) and sex_of[pr] == "male") else 2)
                                                  for pr in by})
                for g, by in carriers.items()
            }
            raw = burden_stats.score_allele_counts(universe, observed_alleles, weights, sexes=sex_list,
                                                   regions=regions)
            scored = {g: burden_stats.GeneScore(*raw[g], sum(1 for sx in sex_list
                                                             if burden_stats.copies_in_test(sx, regions.get(g))),
                                                float("nan"))
                      for g in universe_ids}
        else:
            second = {g: row["s2"] for g, row in ref_genes.items()} if has_s2 else None
            scored = burden_stats.score_genes(model, universe, {g: list(c) for g, c in carriers.items()}, weights,
                                              second=second, sexes=sex_list, regions=regions, alpha=float(alpha))
        # A gene no proband is in the test of (chrY with no male, chrX/chrY with every sex unknown) is
        # not a test: left out of the table, BH and Evidome, and counted.
        untested = [g for g in universe_ids if scored[g].n_tested == 0]
        universe_ids = [g for g in universe_ids if scored[g].n_tested > 0]
        # The shared Benjamini-Hochberg (paralome.stats), as every pipeline uses; burden's own
        # copy is retired. n is the genes tested; a p that is not finite would stay NaN.
        qvalues = dict(zip(universe_ids, benjamini_hochberg([scored[g].p for g in universe_ids]).tolist()))
        rows = []
        for gene in universe_ids:
            score = scored[gene]
            ref = ref_genes.get(gene)
            site = scan.genes.get(gene)
            gene_name = (ref or {}).get("gene_name") or (",".join(sorted(site.symbols)) if site else "") or gene
            gene_names.setdefault(gene, gene_name)
            chrom, start, end = place(gene, ref)
            by_proband = carriers.get(gene, {})
            region = regions.get(gene)
            # Carriers resting on alleles absent from gnomAD, and on recurrent ones.
            from_absent = []
            from_recurrent = []
            for proband, hits in by_proband.items():
                sex = sex_of[proband]
                if regulatory_tier:
                    still = [h for h in hits if not h.absent]
                    still_recurrent = [h for h in hits if h.variant not in recurrent_absent]
                else:
                    still = [h for h in hits if not h.absent]
                    still = still if carrier_status(model, still, sex, region) else []
                    still_recurrent = [h for h in hits if h.variant not in recurrent_absent]
                    still_recurrent = still_recurrent if carrier_status(model, still_recurrent, sex, region) else []
                if not still:
                    from_absent.append(proband)
                if not still_recurrent:
                    from_recurrent.append(proband)
            recurrent_here = sorted({h.variant for hits in by_proband.values() for h in hits
                                     if h.variant in recurrent_absent})
            row = {
                "gene_id": gene, "gene_name": gene_name, "chrom": chrom,
                "start": "" if start is None else start, "end": "" if end is None else end,
                "n_probands": len(by_proband) if regulatory_tier else score.observed,
                "n_tested": score.n_tested,
                "probands": ",".join(sorted(by_proband)) or ".",
                "expected": score.expected, "log2_obs_exp": score.log2_obs_exp, "p": score.p, "q": qvalues[gene],
                "flag": FLAG_ABSENT_RECURRENT if from_recurrent else ".",
                "carriers_from_absent_alleles": len(from_absent),
                "recurrent_absent_alleles": ",".join(f"{v}({recurrent_absent[v]})" for v in recurrent_here) or ".",
                "carriers_from_overrepresented_alleles": len(set(carriers_all.get(gene, {})) - set(by_proband)),
                "overrepresented_alleles": ",".join(f"{a.variant}({len(a.probands)})" for a in dropped_in.get(gene, ())) or ".",
                "variants": " | ".join(
                    f"{proband}: " + "; ".join(hit.label() for hit in by_proband[proband])
                    for proband in sorted(by_proband)
                ) or ".",
                "_selection": score.null_selection,
            }
            if regulatory_tier:
                row["n_alleles"] = score.observed
            elif model == "recessive":
                statuses = {pr: carrier_status(model, hits, sex_of[pr], region) for pr, hits in by_proband.items()}
                row["possible_compound_het"] = ",".join(
                    sorted(pr for pr, st in statuses.items() if st == "possible_compound_het")) or "."
                row["hemizygous"] = ",".join(sorted(pr for pr, st in statuses.items() if st == "hemizygous")) or "."
            rows.append(row)
        rows.sort(key=lambda r: (r["p"], r["gene_id"]))
        inflation, lambda_genes = burden_stats.genomic_inflation([r["p"] for r in rows], [r["expected"] for r in rows])
        gc_applied = False
        if regulatory_tier:
            corrected, gc_applied = burden_stats.genomic_control([r["p"] for r in rows], inflation)
            q_corrected = benjamini_hochberg(corrected).tolist()
            for row, p_gc, q_gc in zip(rows, corrected, q_corrected):
                row["p_gc"], row["q_gc"] = p_gc, q_gc
        _write_gene_table(paths[f"{name}_table"], rows, recessive=(model == "recessive" and not regulatory_tier),
                          regulatory=regulatory_tier)
        p_key = "p_gc" if regulatory_tier else "p"
        per_comparison[name] = {
            "model": model, "tier": tier, "genes_tested": len(rows),
            "genes_with_carriers": sum(1 for r in rows if r["n_probands"] > 0),
            "test": ("allele count vs 2Nq, Poisson upper tail" if regulatory_tier
                     else "carriers, exact Poisson-binomial upper tail"),
            "recessive_probability": (
                None if model != "recessive" or regulatory_tier
                else ("1-exp(-(2q+s2))(1+2q+2s2)" if has_s2
                      else "1-exp(-2q)(1+2q) (Poisson form: the reference has no s2 columns; conservative; "
                           "rebuild the reference for the second-order form)")),
            "genes_only_in_cohort": sum(1 for g in universe_ids if g not in ref_genes),
            "genes_with_no_proband_in_the_test": len(untested),
            "genes_on_x_or_y_outside_par": sum(1 for g in universe_ids if regions.get(g)),
            "genes_flagged_absent_recurrent": sum(1 for r in rows if r["flag"] == FLAG_ABSENT_RECURRENT),
            "p_for_this_tier": p_key,
            "lambda": inflation, "lambda_genes": lambda_genes,
            "genomic_control": (None if not regulatory_tier else
                                {"applied": gc_applied, "lambda": inflation,
                                 "rule": "p_gc = chi2_1.sf(chi2_1.isf(p) / lambda) when lambda > 1; q_gc = BH(p_gc)"}),
            "genes_not_protein_coding_left_out": len(not_coding),
            "top_genes": [
                {k: r[k] for k in ("gene_id", "gene_name", "n_probands", "expected", "p", "q", "flag")}
                for r in rows[:10]
            ],
        }
        if name in EVIDOME_COMPARISONS:
            evidome_columns[name] = {r["gene_id"]: (r["log2_obs_exp"], r["p"], r["q"], r["_selection"]) for r in rows}

    _write_ancestry(paths["ancestry"], estimates, sex_calls)
    _write_evidome_input(paths["evidome_input"], evidome_columns, gene_names)

    tiers_not_run = {} if run_regulatory else {qualify.REGULATORY_TIER: regulatory["reason"]}
    annotation = {
        "vcf_annovar_date": dict(scan.annovar_dates),
        "vcf_header_lines": vcf_annotation_header(Path(vcf)),
        "reference_protocol": {source: (info or {}).get("protocol")
                               for source, info in (provenance.get("sources") or {}).items()},
        "reference_annovar": provenance.get("annovar"),
        "reference_built": provenance.get("built"),
        "reference_buildver": provenance.get("buildver"),
        "note": "gene_burden does not check that the VCF was annotated with the same ANNOVAR and ensGene as the "
                "reference; compare these fields",
    }
    summary = {
        "schema": "paralome.gene_burden.summary.v1",
        "prefix": run_prefix, "created": now_iso(), "paralome_version": __version__,
        "vcf": str(vcf), "reference": str(reference), "regulatory": regulatory,
        "probands": probands_record, "n_probands": len(choice.used),
        "protein_coding": protein_coding,
        "gene_names": symbol_stats,
        "ancestry": {"option": ancestry, "per_proband": ancestry_rows(estimates, sex_calls),
                     "snps_available": len(scan.ancestry_afs)},
        "sex": {
            "per_proband": {call.sample_id: {"sex": call.sex, "source": call.source, "chrx_snps": call.n_snps,
                                             "chrx_het_f": call.f} for call in sex_calls},
            "counts": {sx: sex_list.count(sx) for sx in ("male", "female", "unknown")},
            "chrx_snps_available": len(scan.sex_afs),
            "rule": (f"sheet's sex column when male/female; otherwise F = 1 - observed/expected heterozygosity over "
                     f"common chrX non-PAR coding SNPs (expected sum 2p(1-p), p = gnomAD overall AF): male when "
                     f"F >= {SEX_F_MALE}, female when F <= {SEX_F_FEMALE}, unknown in between or with fewer than "
                     f"{MIN_SEX_SNPS} SNPs; unknown sex is left out of chrX/chrY non-PAR genes"),
            "pseudo_autosomal_regions": {"build": par_build, "regions": {k: [list(r) for r in v] for k, v in par.items()}},
        },
        "absent_from_gnomad": {
            "qualifying_alleles_absent": len(scan.absent_allele_probands),
            "recurrent_alleles": len(recurrent_absent),
            "recurrent_min_probands": RECURRENT_ABSENT_MIN_PROBANDS,
            "recurrent": dict(sorted(recurrent_absent.items(), key=lambda kv: (-kv[1], kv[0]))[:200]),
            "note": "alleles absent from gnomAD count as rare; now one too common for that (below) is "
                    "dropped. A gene whose carrier rests only on a recurrent absent allele left in is flagged "
                    "absent_recurrent. An allele missing from gnomAD's exome table but in its genome table "
                    "(outside the exome capture) is judged on the genome table and is not absent ",
        },
        # Every allele dropped as too common in the cohort for its gnomAD frequency.
        "overrepresented_alleles": {
            "rule": (f"dropped when P(X >= carriers | X >= 1) < {artefacts.OVERREPRESENTED_P:g}, X ~ Binomial(probands, "
                     "1-(1-f)^2), f = gnomAD AF_grpmax (genome table for the regulatory tier, exome otherwise; an "
                     "allele gnomAD lacks gets the gene floor); taken out of the carriers and, where gnomAD has it, "
                     "out of the gene's expectation"),
            "threshold": artefacts.OVERREPRESENTED_P,
            "probands": len(choice.used),
            "dropped": len(too_common),
            "alleles": [a.summary() for a in sorted(too_common.values(), key=lambda a: (-len(a.probands), a.variant))][:1000],
        },
        "annotation_provenance": annotation,
        "comparisons": per_comparison,
        "tiers_run": list(tiers), "tiers_not_run": tiers_not_run,
        "evidome_comparisons": [n for n in EVIDOME_COMPARISONS if n in evidome_columns],
        "scan": {**dict(scan.counts), "failed_genotype_calls": dict(scan.failed_calls), "source": scan.source},
        "method": {
            "test": "exact one-sided Poisson-binomial upper tail P(X >= observed), one probability per proband",
            "carrier_probability": {
                "dominant": "1-(1-q)^2",
                "recessive": "P(>= 2 qualifying alleles on the two haplotypes, any phase) = "
                             "1-exp(-(2q+s2))(1+2q+2s2), s2 = sum AF^2 (Poisson form 1-exp(-2q)(1+2q) when the "
                             "reference has no s2)",
                "hemizygous": "q (both models): a male on chrX/chrY outside the PARs; a 1/1 call is one copy",
            },
            "second_order_reference": second_order,
            "q": "sum of gnomAD AF over qualifying alleles passing the model's AF_grpmax gate, mixed by ancestry weights",
            "q_floor": burden_stats.Q_FLOOR,
            "multiple_testing": "Benjamini-Hochberg per (model, tier)",
            "log2_obs_exp_pseudocount": burden_stats.PSEUDOCOUNT,
            "lambda_min_expected": burden_stats.LAMBDA_MIN_EXPECTED,
            "lambda_note": "lambda from genes expecting at least lambda_min_expected carriers; biased low for an "
                           "exact discrete test; reported for the coding tiers, applied to the regulatory tier "
                           "as genomic control (p_gc, q_gc)",
            "genes": "protein-coding genes of the genome's GTF only (see protein_coding)",
            "recessive_phase": "unknown: two qualifying heterozygous variants count as a possible compound "
                               "heterozygote; alleles whose REF spans overlap are one event (the more damaging kept), "
                               "and so are alleles split from one called allele (INFO/PARALOME_ORIGIN)",
            "duplicates": "an allele at the same chrom, pos, REF and ALT as one already read is skipped",
            "regulatory_test": (
                "allele count: observed = qualifying alleles carried by probands (het 1, hom 2; one copy for a "
                "male outside the PARs); expected = sum over probands of copies*q_i (ancestry-weighted gnomAD "
                "genome AF sum, not capped); Poisson upper tail P(X >= observed); the model only sets the "
                "AF_grpmax gate"
            ),
            "q_floor_regulatory": burden_stats.Q_FLOOR_GENOME,
            "evidome_null_selection_probability": (
                f"per gene, P(p < {float(alpha):g}) (--alpha) under the null: P(X >= k*) with k* the "
                "smallest count whose p is below that, from the same Poisson-binomial as p"),
            "genotype_filters": DEFAULT_FILTERS.to_dict(),
            "thresholds": request["thresholds"],
        },
    }
    atomic_write_text(paths["summary"], json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

    outputs = {key: str(path) for key, path in paths.items()}
    left = sorted(set(str(v) for v in recorded.values()) - set(outputs.values())) if recorded else []
    checkpoint = {
        **base, "status": "complete", "complete": True, "finished": now_iso(), "reason": None,
        "outputs": outputs, "tiers_run": list(tiers), "tiers_not_run": tiers_not_run,
        "regulatory": regulatory, "comparisons": names,
        "evidome_comparisons": summary["evidome_comparisons"],
        "earlier_outputs_left_in_place": left,
    }
    _write_checkpoint(workdir, checkpoint)
    echo(f"gene_burden complete: {len(names)} tables in {workdir / 'results'}; Evidome input {paths['evidome_input']}")
    return checkpoint


# --------------------------------------------------------------------- writers

def _number(value: float) -> str:
    return format(float(value), ".6g")


def _evidome_triple_ok(value) -> bool:
    """(log2 O/E, p, q[, null selection probability]) all finite numbers, p, q and the probability in [0, 1]."""
    import math

    if value is None:
        return False
    try:
        numbers = [float(v) for v in value]
    except (TypeError, ValueError):
        return False
    if len(numbers) not in (3, 4) or not all(math.isfinite(v) for v in numbers):
        return False
    return all(0.0 <= v <= 1.0 for v in numbers[1:])


def split_variants(cell: str, limit: int = EXCEL_CELL_CHARS) -> list[str]:
    """A ``variants`` listing as consecutive pieces of at most ``limit`` characters.

    Cut only after a separator (_VARIANTS_SEPARATOR), so a piece
    holds whole alleles; joining the pieces in order gives the listing back.
    """
    if len(cell) <= limit:
        return [cell]
    tokens = _VARIANTS_SEPARATOR.split(cell)           # allele, separator, allele, ...
    pieces: list[str] = []
    current = ""
    for i in range(0, len(tokens), 2):
        unit = tokens[i] + (tokens[i + 1] if i + 1 < len(tokens) else "")
        if current and len(current) + len(unit) > limit:
            pieces.append(current)
            current = ""
        current += unit
    if current:
        pieces.append(current)
    return pieces


def variants_columns(width: int) -> tuple:
    """``variants`` and its continuation columns, ``width`` in all."""
    return ("variants",) + tuple(f"variants_{k}" for k in range(2, int(width) + 1))


def joined_variants(row: Mapping[str, Any]) -> str:
    """A table row's whole allele listing: ``variants`` and its continuations, in order."""
    text = str(row.get("variants") or "")
    k = 2
    while f"variants_{k}" in row:
        text += str(row.get(f"variants_{k}") or "")
        k += 1
    return text


def _write_gene_table(path: Path, rows, *, recessive: bool, regulatory: bool = False) -> None:
    pieces = [split_variants(str(row.get("variants", ""))) for row in rows]
    width = max((len(p) for p in pieces), default=1)
    columns = (TABLE_COLUMNS + (RECESSIVE_EXTRA if recessive else ())
               + (REGULATORY_EXTRA if regulatory else ()))
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns + variants_columns(width))
        for row, listing in zip(rows, pieces):
            out = []
            for column in columns:
                value = row.get(column, "")
                out.append(_number(value) if column in ("expected", "log2_obs_exp", "p", "q", "p_gc", "q_gc")
                           else str(value))
            writer.writerow(out + listing + [""] * (width - len(listing)))


def _write_ancestry(path: Path, estimates, sexes=None) -> None:
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANCESTRY_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(ancestry_rows(estimates, sexes))


def _write_evidome_input(path: Path, columns: Mapping[str, Mapping[str, tuple]], gene_names: Mapping[str, str]) -> None:
    """The wide table Evidome reads (Evidome's wide_stats contract): comment lines, then one row per gene.

    A gene outside a comparison's universe is '.' there: Evidome records it as
    untested in that comparison, never as p = 1.

    A comparison whose values carry a fourth number (the null
    selection probability) gets the column ``{C}_Null_selection_probability``.

    (checked against Evidome's own reader, evidome.wide_stats):
    a comparison with no column gets no header line (Evidome refuses a header
    naming a comparison the file has no columns for), and a triple with any
    value that is not a finite number -- or a p or q outside [0, 1] -- is
    written '.' whole. Evidome refuses an infinite P-value or Q-value outright,
    reads 'nan' as missing (so one bad cell would leave a gene tested with no
    direction, or a direction with no p), and a table with no comparison at
    all is refused here rather than handed on.
    """
    names = [n for n in EVIDOME_COMPARISONS if n in columns]
    if not names:
        raise ValueError("no burden comparison to write for Evidome (Evidome refuses a table without one)")
    # A comparison whose values carry a fourth number gets the optional Null_selection_probability
    # column (Evidome's weighted ORA null for burden tests); with three, the table is as before.
    weighted = {name: bool(columns[name]) and all(v is not None and len(v) == 4 for v in columns[name].values())
                for name in names}
    genes = sorted({gene for name in names for gene in columns[name]})
    with atomic_text_writer(path, newline="") as handle:
        for name in names:
            handle.write(EVIDOME_HEADER.format(name=name) + "\n")
        header = ["gene_id", "gene_name"]
        for name in names:
            header += [f"{name}_Log2_fold_change", f"{name}_P-value", f"{name}_Q-value"]
            if weighted[name]:
                header.append(f"{name}_{EVIDOME_NULL_ROLE}")
        handle.write("\t".join(header) + "\n")
        for gene in genes:
            cells = [gene, gene_names.get(gene) or gene]
            for name in names:
                width = 4 if weighted[name] else 3
                value = columns[name].get(gene)
                value = None if value is None else tuple(value)[:width]
                cells += ["."] * width if not _evidome_triple_ok(value) else [_number(v) for v in value]
            handle.write("\t".join(cells) + "\n")


# ------------------------------------------------------------------ state side

def summarize(workdir: Optional[Path]) -> dict[str, Any]:
    """Stage state for project_state.json (pipeline/state.py).

    complete: the checkpoint says so and its summary and Evidome input are
    there; skipped: as recorded, with the reason; otherwise not finished.
    """
    from paralome.checkpoints import file_info

    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    path = stage_checkpoint_path(wd, STAGE)
    try:
        ckpt = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "not_available", "workdir": str(wd), "complete": False, "checkpoint": file_info(path)}
    outputs = ckpt.get("outputs") if isinstance(ckpt.get("outputs"), dict) else {}

    def here(value: Any) -> Optional[str]:
        # A copied project: the checkpoint's absolute path, or the same name in this results folder.
        if not value:
            return None
        recorded = Path(str(value))
        if recorded.is_file():
            return str(recorded)
        local = wd / "results" / recorded.name
        return str(local) if local.is_file() else str(recorded)

    evidome_input = file_info(here(outputs.get("evidome_input")))
    summary = file_info(here(outputs.get("summary")))
    status = str(ckpt.get("status") or "unknown")
    if status == "complete":
        status = "complete" if (ckpt.get("complete") and evidome_input["nonempty"] and summary["nonempty"]) else "incomplete"
    # A result computed by an earlier method is not finished. The pipeline never issues a stage
    # that reads complete, so the method is checked here too: read as incomplete, the stage is
    # issued again and its own check computes it afresh.
    out_of_date = None
    if status == "complete" and ckpt.get("method") != METHOD:
        out_of_date = (f"it was computed by {ckpt.get('method') or 'a method not recorded'}; "
                       f"the method is now {METHOD}")
        status = "incomplete"
    return {
        "status": status, "complete": status == "complete", "workdir": str(wd),
        "out_of_date": out_of_date,
        "checkpoint": file_info(path),
        "reason": (f"{out_of_date}; run gene_burden again" if out_of_date else ckpt.get("reason")),
        "evidome_input": evidome_input, "summary": summary,
        "tables": {key[:-len("_table")]: file_info(here(value)) for key, value in outputs.items() if key.endswith("_table")},
        "ancestry": file_info(here(outputs.get("ancestry"))),
        "tiers_run": ckpt.get("tiers_run") or [], "tiers_not_run": ckpt.get("tiers_not_run") or {},
        "n_probands": len(((ckpt.get("probands") or {}).get("used")) or []),
    }
