"""The variants of a list of genes, one file per gene, after gene_burden.

A bolt-on: it runs no stage and makes none stale.
Clients find their candidate genes in the burden results; finding those
genes' variants meant searching a whole-genome segregation table or a burden
table's allele listings by hand.

``paralome DNA --project P --extract-damaging-variants A,B`` writes, per gene,
the alleles gene_burden counted (the qualifying, "damaging" ones), read from
the burden tables' listings: one row per allele and proband, with the burden
tests (model and tier) it counted in.

``paralome DNA --project P --extract-all-variants A,B`` writes, per gene,
every row of the segregation table (the annotated VCF as a table: every
annotation and every sample's genotype) that names the gene, as it is.

A gene is asked for by its Ensembl id (with or without the version) or its
name, either case. The files go to ``<project>/gene_burden/extracted``.
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from paralome.checkpoints import atomic_text_writer
from paralome.burden.stage import gene_key, joined_variants, summarize

#: Where the files go, under the gene_burden folder.
EXTRACT_FOLDER = "extracted"

COLUMNS = ("gene_id", "gene_name", "proband", "variant", "zygosity", "consequence",
           "AF_grpmax", "REVEL", "AlphaMissense", "absent_from_gnomAD", "burden_tests")

#: One allele, as burden.cohort.Hit.label() writes it:
#:   chrom:pos:REF>ALT het|hom <consequence> AF_grpmax=x REVEL=y AlphaMissense=z[ absent_from_gnomAD]
_ALLELE = re.compile(
    r"^(?P<variant>\S+:\d+:\S+>\S+) (?P<zygosity>het|hom) (?P<consequence>.*?) "
    r"AF_grpmax=(?P<af>\S+) REVEL=(?P<revel>\S+) AlphaMissense=(?P<am>.*?)"
    r"(?P<absent> absent_from_gnomAD)?$"
)
#: Where one allele ends and the next begins, and one proband's alleles end and the next's begin.
_NEXT_ALLELE = re.compile(r"; (?=\S+:\d+:\S+>\S+ (?:het|hom) )")
_NEXT_PROBAND = re.compile(r" \| (?=[^\s|:]+: \S+:\d+:\S+>\S+ (?:het|hom) )")


class ListingError(ValueError):
    """A ``variants`` listing that does not read back as the alleles it lists."""


def parse_listing(cell: str) -> list[tuple[str, dict]]:
    """[(proband, allele fields)] of one gene's listing ('.' or '' -> []).

    The parse is rebuilt into the listing and compared: a listing that does
    not come back exactly is refused rather than read wrongly.
    """
    if cell in ("", "."):
        return []
    out: list[tuple[str, dict]] = []
    for block in _NEXT_PROBAND.split(cell):
        proband, sep, alleles = block.partition(": ")
        if not sep:
            raise ListingError(f"no 'proband: ' in {block[:80]!r}")
        for text in _NEXT_ALLELE.split(alleles):
            match = _ALLELE.match(text)
            if not match:
                raise ListingError(f"an allele that does not read: {text[:120]!r}")
            out.append((proband, match.groupdict()))
    if _rebuild(out) != cell:
        raise ListingError("the listing does not read back exactly")
    return out


def _rebuild(parsed: Sequence[tuple[str, dict]]) -> str:
    order: list[str] = []
    labels: dict[str, list[str]] = {}
    for proband, a in parsed:
        if proband not in labels:
            order.append(proband)
            labels[proband] = []
        labels[proband].append(f"{a['variant']} {a['zygosity']} {a['consequence']} AF_grpmax={a['af']} "
                               f"REVEL={a['revel']} AlphaMissense={a['am']}{a['absent'] or ''}")
    return " | ".join(f"{p}: " + "; ".join(labels[p]) for p in order) or "."


def parse_gene_list(text: str) -> list[str]:
    """The genes of ``--extract-damaging-variants`` or ``--extract-all-variants``: comma-separated, blanks dropped, order kept, no repeats."""
    seen: list[str] = []
    for token in str(text or "").split(","):
        token = token.strip()
        if token and token.upper() not in {g.upper() for g in seen}:
            seen.append(token)
    return seen


def _file_token(gene: str) -> str:
    """The gene as it may appear in a file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", gene).strip("_") or "gene"


@dataclass
class ExtractResult:
    folder: Path
    kind: str = "damaging"
    files: dict = field(default_factory=dict)          # gene asked for -> file written
    rows: dict = field(default_factory=dict)           # gene asked for -> rows written
    not_found: list = field(default_factory=list)      # genes found nowhere
    no_alleles: list = field(default_factory=list)     # genes tested, with no allele counted
    unreadable: dict = field(default_factory=dict)     # "<table> <gene_id>" -> why


@dataclass
class BurdenAlleles:
    prefix: str
    found: dict          # gene asked for -> {(gene_id, proband, variant): row}
    matched: set         # genes asked for that are in a burden table
    names: dict          # gene asked for -> the names and ids the burden tables give it (upper case)
    unreadable: dict


def read_burden_alleles(burden_workdir: Path, wanted: Sequence[str]) -> BurdenAlleles:
    """Every allele gene_burden counted for the genes ``wanted``, from its tables.

    Raises ValueError when gene_burden is not complete: there is nothing
    finished to read.
    """
    burden_workdir = Path(burden_workdir)
    state = summarize(burden_workdir)
    if not state.get("complete"):
        why = state.get("reason") or state.get("status") or "not finished"
        raise ValueError(f"gene_burden in {burden_workdir} is not complete ({why})")
    tables = {name: Path(info["path"]) for name, info in sorted((state.get("tables") or {}).items())
              if info.get("path") and Path(info["path"]).is_file()}
    by_id = {gene_key(g).upper(): g for g in wanted}
    by_name = {g.upper(): g for g in wanted}
    found: dict[str, dict[tuple, dict]] = {g: {} for g in wanted}
    names: dict[str, set] = {g: {g.upper(), gene_key(g).upper()} for g in wanted}
    unreadable: dict[str, str] = {}
    matched: set[str] = set()
    prefix = ""
    previous = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    try:
        for name, path in tables.items():
            prefix = prefix or path.name.split("_burden_", 1)[0]
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    gene_id = str(row.get("gene_id") or "")
                    row_names = {n.strip().upper() for n in str(row.get("gene_name") or "").split(",") if n.strip()}
                    asked = by_id.get(gene_key(gene_id).upper())
                    if asked is None:
                        asked = next((by_name[n] for n in row_names if n in by_name), None)
                    if asked is None:
                        continue
                    matched.add(asked)
                    names[asked] |= row_names | {gene_key(gene_id).upper()}
                    try:
                        alleles = parse_listing(joined_variants(row))
                    except ListingError as exc:
                        unreadable[f"{path.name} {gene_id}"] = str(exc)
                        continue
                    for proband, a in alleles:
                        key = (gene_id, proband, a["variant"])
                        out = found[asked].setdefault(key, {
                            "gene_id": gene_id, "gene_name": row.get("gene_name") or ".", "proband": proband,
                            "variant": a["variant"], "zygosity": a["zygosity"], "consequence": a["consequence"],
                            "AF_grpmax": a["af"], "REVEL": a["revel"], "AlphaMissense": a["am"],
                            "absent_from_gnomAD": "yes" if a["absent"] else "no", "burden_tests": [],
                        })
                        if name not in out["burden_tests"]:
                            out["burden_tests"].append(name)
    finally:
        csv.field_size_limit(previous)
    return BurdenAlleles(prefix=prefix, found=found, matched=matched, names=names, unreadable=unreadable)


def extract_damaging_variants(project: Path, genes: Iterable[str]) -> ExtractResult:
    """``<prefix>_<GENE>_damaging_variants.tsv`` per gene: the alleles gene_burden counted."""
    wanted = parse_gene_list(",".join(genes))
    burden_workdir = Path(project) / "gene_burden"
    burden = read_burden_alleles(burden_workdir, wanted)
    folder = burden_workdir / EXTRACT_FOLDER
    result = ExtractResult(folder=folder, kind="damaging", unreadable=burden.unreadable)
    for gene in wanted:
        rows = burden.found[gene]
        if not rows:
            (result.no_alleles if gene in burden.matched else result.not_found).append(gene)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{burden.prefix}_{_file_token(gene)}_damaging_variants.tsv"
        ordered = sorted(rows.values(), key=lambda r: (r["gene_id"], _position(r["variant"]), r["proband"]))
        with atomic_text_writer(path, newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(COLUMNS)
            for r in ordered:
                writer.writerow([",".join(r[c]) if c == "burden_tests" else r[c] for c in COLUMNS])
        result.files[gene] = path
        result.rows[gene] = len(ordered)
    return result


#: The segregation table's columns that name a row's gene(s).
_SEGREGATION_GENE_COLUMNS = ("gene", "annovar_gene", "snpeff_gene")
_SEGREGATION_TABLE = "all_variants.segregation.tsv"


def segregation_table(project: Path) -> Path:
    """The project's segregation table: the one its checkpoint names, else the one in results/."""
    import json

    from paralome.checkpoints import stage_checkpoint_path

    workdir = Path(project) / "segregation"
    try:
        outputs = json.loads(stage_checkpoint_path(workdir, "segregation").read_text()).get("outputs") or {}
    except (OSError, ValueError, AttributeError):
        outputs = {}
    for key in ("all_variants", "segregation_tsv"):
        value = outputs.get(key)
        if value and Path(value).is_file():
            return Path(value)
        if value and (workdir / "results" / Path(value).name).is_file():
            return workdir / "results" / Path(value).name     # a copied project
    found = sorted((workdir / "results").glob(f"*_{_SEGREGATION_TABLE}"))
    if len(found) == 1:
        return found[0]
    raise ValueError(f"no segregation table ({_SEGREGATION_TABLE}) in {workdir / 'results'}"
                     if not found else f"{len(found)} segregation tables in {workdir / 'results'}; "
                     "expected one")


def _row_genes(row: dict) -> set:
    from paralome.reports.candidates import decode_annovar

    genes: set = set()
    for column in _SEGREGATION_GENE_COLUMNS:
        value = row.get(column)
        if value in (None, "", "."):
            continue
        for token in re.split(r"[,;]", decode_annovar(value)):
            token = token.strip()
            if token and token != ".":
                genes.add(token.upper())
                genes.add(gene_key(token).upper())
    return genes


def extract_all_variants(project: Path, genes: Iterable[str]) -> ExtractResult:
    """``<prefix>_<GENE>_all_variants.tsv`` per gene: the segregation table's rows in the gene, as they are.

    One
    pass over the table; a line is parsed only when one of the genes' names
    appears in it, and written, unchanged, when a gene column names the gene.
    An Ensembl id is matched through the names gene_burden gives it, when
    gene_burden is complete.
    """
    from contextlib import ExitStack

    wanted = parse_gene_list(",".join(genes))
    project = Path(project)
    table = segregation_table(project)
    prefix = table.name[: -len(_SEGREGATION_TABLE)].rstrip("_") or project.name
    folder = project / "gene_burden" / EXTRACT_FOLDER
    result = ExtractResult(folder=folder, kind="all")
    names = {g: {g.upper(), gene_key(g).upper()} for g in wanted}
    try:
        names = read_burden_alleles(project / "gene_burden", wanted).names
    except ValueError:
        pass
    owner: dict[str, str] = {}
    for gene in wanted:
        for name in names[gene]:
            owner.setdefault(name, gene)
    screen = re.compile("|".join(sorted((re.escape(n) for n in owner if n), key=len, reverse=True)),
                        re.IGNORECASE)
    previous = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    try:
        with ExitStack() as stack, table.open(newline="") as fh:
            header_line = fh.readline()
            header = next(csv.reader([header_line], delimiter="\t"))
            handles: dict = {}
            for line in fh:
                if not screen.search(line):
                    continue
                row = dict(zip(header, next(csv.reader([line], delimiter="\t"))))
                for gene in sorted({owner[n] for n in _row_genes(row) if n in owner}):
                    if gene not in handles:
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f"{prefix}_{_file_token(gene)}_all_variants.tsv"
                        handles[gene] = stack.enter_context(atomic_text_writer(path, newline=""))
                        handles[gene].write(header_line)
                        result.files[gene] = path
                        result.rows[gene] = 0
                    handles[gene].write(line)
                    result.rows[gene] += 1
    finally:
        csv.field_size_limit(previous)
    result.not_found = [gene for gene in wanted if gene not in result.files]
    return result


def _position(variant: str) -> tuple:
    chrom, pos, alleles = variant.rsplit(":", 2)
    try:
        return chrom, int(pos), alleles
    except ValueError:
        return chrom, 0, alleles


def print_extract_result(result: ExtractResult, out=None) -> None:
    out = out or sys.stdout
    what = "damaging" if result.kind == "damaging" else "all"
    for gene, path in result.files.items():
        print(f"Extracted {what} {gene:<16}: {result.rows[gene]} row(s) -> {path}", file=out)
    for gene in result.not_found:
        where = "any gene_burden table" if result.kind == "damaging" else "the segregation table"
        print(f"Extracted {what} {gene:<16}: not in {where}; no file", file=out)
    for gene in result.no_alleles:
        print(f"Extracted {what} {gene:<16}: tested, no allele counted in any burden test; no file", file=out)
    for where, why in result.unreadable.items():
        print(f"Not read: {where}: {why}", file=out)
