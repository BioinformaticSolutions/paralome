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

"""Every quantification exporter's table, read into one ``MsTable``.

The search (raw files to identified, quantified features) stays in the
exporter's own software; Paralome starts at its table, the way the
single-cell arms start at a Seurat object. Each exporter's layout is
RECOGNISED from its header, never guessed: a table no layout claims is
refused with PLM-0227, naming what was looked for.

Layouts (``LAYOUTS``), in the order they are tried:

    proteome_discoverer   Thermo Proteome Discoverer protein, peptide-group and
                          phosphopeptide exports (CSV or tab), with or without
                          the title line above the header
    maxquant_proteins     MaxQuant proteinGroups.txt
    fragpipe_proteins     FragPipe / IonQuant combined_protein.tsv
    diann_proteins        DIA-NN report.pg_matrix.tsv

What every reader does the same way:

* A zero is NOT MEASURED, not "none": exporters write 0 for a feature with no
  signal in a sample. Zeros and blanks both become NaN here, before anything
  else sees the numbers.
* The abundance columns are found by the layout's own column rule, in its own
  order of preference (MaxQuant: reporter intensities, then LFQ, then raw
  intensities), and which family was used is recorded.
* Each abundance column keeps its full header and gets a short ``label`` --
  the sample's own name inside it (``Ctrl_1`` in
  ``Abundances (TMT 126, Ctrl_1)``) -- which the sample sheet may use
  instead of the header.
* Decoys, contaminants and site-only identifications are removed where the
  exporter marks them, and counted.

Phosphopeptides: the feature is the peptide form as
exported, one row each; its sites are annotated (residue, position in the
protein, localization probability) and never combined.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

KIND_PROTEIN = "protein"
KIND_PHOSPHO = "phospho"
KIND_METABOLITE = "metabolite"


@dataclass
class MsTable:
    """One exporter table, as every later stage reads it."""

    exporter: str                     # the layout that read it
    kind: str                         # protein | phospho | metabolite
    source: str                       # the file, resolved
    quantity: str                     # which abundance family was read
    columns: List[str]                # abundance column headers, as written
    labels: List[str]                 # the sample name inside each header
    keys: List[str]                   # one unique id per feature (row)
    features: List[Dict[str, str]]    # per feature: identifiers and annotation
    values: np.ndarray                # features x columns, NaN = not measured
    log_header: Optional[bool] = None  # the header says log (True) / linear (False) / silent
    notes: List[str] = field(default_factory=list)
    removed: Dict[str, int] = field(default_factory=dict)


class LayoutError(ValueError):
    """No layout recognises the table, or the table contradicts its layout."""


# --------------------------------------------------------------------------
# Reading the text
# --------------------------------------------------------------------------

def _rows(path: Path) -> List[List[str]]:
    """Every row of a CSV or tab-separated export, the byte-order mark ignored."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    sample = text[:20000]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    return [row for row in csv.reader(io.StringIO(text), delimiter=delimiter)]


def _header_index(rows: Sequence[Sequence[str]], required: Callable[[List[str]], bool], limit: int = 20) -> int:
    """The first of the first ``limit`` rows that ``required`` accepts as the header.

    Proteome Discoverer exports made for a report carry a title line ("Table 2:
    Comparison of ...") above the real header; others start at the header.
    """
    for index, row in enumerate(rows[:limit]):
        if required([cell.strip() for cell in row]):
            return index
    return -1


def _number(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return float("nan")
    try:
        value = float(text.replace(",", "")) if text.count(",") and "." in text else float(text)
    except ValueError:
        return float("nan")
    if not np.isfinite(value) or value == 0.0:
        return float("nan")
    return value


def _matrix(body: Sequence[Sequence[str]], positions: Sequence[int]) -> np.ndarray:
    values = np.full((len(body), len(positions)), np.nan)
    for i, row in enumerate(body):
        for j, position in enumerate(positions):
            if position < len(row):
                values[i, j] = _number(row[position])
    return values


def _unique(keys: Sequence[str]) -> List[str]:
    """Keys made unique by a #2, #3 suffix where an exporter repeats one."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
        out.append(key if seen[key] == 1 else f"{key}#{seen[key]}")
    return out


_GN = re.compile(r"\bGN=(\S+)")
_OS = re.compile(r"\bOS=(.+?)\s+OX=")


def _symbol_from_description(description: str) -> str:
    match = _GN.search(description or "")
    return match.group(1) if match else ""


def _header_says_log(columns: Sequence[str]) -> Optional[bool]:
    text = " ".join(columns).lower()
    if re.search(r"\blog\s*2|\blog2|\blog10|\blog\b", text):
        return True
    return None


# --------------------------------------------------------------------------
# Proteome Discoverer
# --------------------------------------------------------------------------

#: Proteome Discoverer's per-sample abundance families, most raw first.
#: "Grouped" abundances are per-group summaries, never samples, and are
#: never read.
_PD_FAMILIES = (
    ("Abundance:", "abundance"),
    ("Abundances (Normalized):", "normalized abundance"),
    ("Abundances (Scaled):", "scaled abundance"),
    ("Abundances (", "abundance"),
)


def _pd_family(header: List[str]) -> Tuple[str, List[int]]:
    for prefix, name in _PD_FAMILIES:
        positions = [
            i for i, column in enumerate(header)
            if column.startswith(prefix) and "Grouped" not in column and "Count" not in column
        ]
        if positions:
            return name, positions
    return "", []


def _pd_label(column: str) -> str:
    """The sample's own name inside a PD abundance header.

    ``Abundances (TMT 126, Ctrl_1)`` -> ``Ctrl_1``;
    ``Abundances (Vehicle 1)`` -> ``Vehicle 1``;
    ``Abundance: F1: 126, Sample, Ctrl, 1`` -> ``F1: 126, Sample, Ctrl, 1``.
    """
    text = column.strip()
    inner = re.match(r"^Abundances?(?: \([A-Za-z]+\))?\s*\((.*)\)\s*$", text)
    if inner:
        parts = [part.strip() for part in inner.group(1).split(",")]
        if len(parts) > 1 and re.match(r"^(TMT\S*\s*)?\d{3}[NC]?$", parts[0]):
            return ", ".join(parts[1:])
        return inner.group(1).strip()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _pd_is_header(row: List[str]) -> bool:
    has_accession = any(cell in ("Accession", "Accessions", "Master Protein Accessions") for cell in row)
    return has_accession and any(cell.startswith("Abundance") for cell in row)


_PD_PHOSPHO_SITE = re.compile(r"(\d+)\s*×\s*Phospho\s*\[([^\]]*)\]")
_SITE = re.compile(r"([STYH])(\d+)(?:\(([\d.]+)\))?")


def phospho_sites(modifications: str, positions: str) -> Tuple[List[Dict[str, object]], int]:
    """The phosphosites a PD row carries, placed in its first master protein.

    ``modifications`` is PD's ``1×TMTpro [N-Term]; 2×Phospho [S3(100); T9(97.5)]``;
    ``positions`` its ``[54-65]`` (``[54-65]; [12-23]`` for several proteins).
    Returns ``(sites, count)``: each localized site's residue, offset in the
    peptide, position in the protein, and probability; ``count`` is the number
    of phosphates, which an unlocalized ``S/T`` entry still counts.
    """
    match = _PD_PHOSPHO_SITE.search(modifications or "")
    if not match:
        return [], 0
    count = int(match.group(1))
    starts = re.findall(r"\[(\d+)-\d+\]", positions or "")
    start = int(starts[0]) if starts else None
    sites: List[Dict[str, object]] = []
    for entry in match.group(2).split(";"):
        entry = entry.strip()
        site = _SITE.fullmatch(entry)
        if site:
            offset = int(site.group(2))
            sites.append({
                "residue": site.group(1), "offset": offset,
                "position": (start + offset - 1) if start is not None else None,
                "probability": float(site.group(3)) if site.group(3) else None,
            })
        elif entry:
            sites.append({"residue": entry, "offset": None, "position": None, "probability": None})
    return sites, count


def _site_text(sites: Sequence[Dict[str, object]]) -> str:
    parts = []
    for site in sites:
        if site["position"] is None:
            parts.append(f"{site['residue']}(unlocalized)")
        else:
            probability = "" if site["probability"] is None else f"({site['probability']:g})"
            parts.append(f"{site['residue']}{site['position']}{probability}")
    return ";".join(parts)


def _read_proteome_discoverer(path: Path, rows: List[List[str]], start: int) -> MsTable:
    header = [cell.strip() for cell in rows[start]]
    body = [row for row in rows[start + 1:] if any(cell.strip() for cell in row)]
    quantity, positions = _pd_family(header)
    if not positions:
        raise LayoutError("Proteome Discoverer header with no per-sample abundance column")
    column = {name: i for i, name in enumerate(header) if name}

    def cell(row: List[str], *names: str) -> str:
        for name in names:
            if name in column and column[name] < len(row):
                return row[column[name]].strip()
        return ""

    phospho = "Modifications" in column and ("Annotated Sequence" in column or "Sequence" in column)
    kind = KIND_PHOSPHO if phospho else KIND_PROTEIN
    keys: List[str] = []
    features: List[Dict[str, str]] = []
    kept_body: List[List[str]] = []
    removed: Dict[str, int] = {}
    for row in body:
        if phospho:
            modifications = cell(row, "Modifications")
            if "Phospho" not in modifications:
                removed["not phosphorylated"] = removed.get("not phosphorylated", 0) + 1
                continue
            accessions = cell(row, "Master Protein Accessions", "Accessions", "Protein Accessions")
            description = cell(row, "Master Protein Descriptions", "Protein Descriptions")
            positions_text = cell(row, "Positions in Master Proteins", "Positions", "Position")
            sites, count = phospho_sites(modifications, positions_text)
            sequence = cell(row, "Annotated Sequence", "Sequence")
            core = sequence.split(".")[1] if sequence.count(".") >= 2 else sequence
            first = accessions.split(";")[0].strip()
            site_text = _site_text(sites)
            keys.append(f"{first}|{site_text or 'phospho'}|{core}")
            features.append({
                "accession": first, "accessions": accessions,
                "gene_symbol": _symbol_from_description(description.split(";")[0]),
                "description": description,
                "sequence": sequence, "modifications": modifications,
                "positions": positions_text, "sites": site_text,
                "phospho_count": str(count),
                "localized": "yes" if sites and all(s["position"] is not None for s in sites) else "no",
                "min_localization_probability": (
                    f"{min(s['probability'] for s in sites if s['probability'] is not None):g}"
                    if any(s["probability"] is not None for s in sites) else ""
                ),
            })
        else:
            accession = cell(row, "Accession")
            description = cell(row, "Description")
            symbol = cell(row, "Gene Symbol") or _symbol_from_description(description)
            keys.append(accession)
            features.append({
                "accession": accession, "accessions": accession,
                "gene_symbol": symbol, "description": description,
                "coverage_pct": cell(row, "Coverage [%]", "Coverage"),
                "peptides": cell(row, "# Peptides"),
                "unique_peptides": cell(row, "# Unique Peptides"),
            })
        kept_body.append(row)
    columns = [header[i] for i in positions]
    species = {m.group(1) for f in features for m in [_OS.search(f.get("description", ""))] if m}
    notes = [f"species named in the descriptions: {', '.join(sorted(species))}"] if species else []
    return MsTable(
        exporter="proteome_discoverer", kind=kind, source=str(path), quantity=quantity,
        columns=columns, labels=[_pd_label(c) for c in columns],
        keys=_unique(keys), features=features, values=_matrix(kept_body, positions),
        log_header=_header_says_log(columns), notes=notes, removed=removed,
    )


# --------------------------------------------------------------------------
# MaxQuant proteinGroups.txt
# --------------------------------------------------------------------------

_MQ_FAMILIES = (
    ("Reporter intensity corrected ", "reporter intensity corrected"),
    ("LFQ intensity ", "LFQ intensity"),
    ("Intensity ", "intensity"),
)


def _mq_is_header(row: List[str]) -> bool:
    return "Protein IDs" in row and "Majority protein IDs" in row


def _read_maxquant_proteins(path: Path, rows: List[List[str]], start: int) -> MsTable:
    header = [cell.strip() for cell in rows[start]]
    column = {name: i for i, name in enumerate(header)}
    quantity, positions = "", []
    for prefix, name in _MQ_FAMILIES:
        positions = [i for i, c in enumerate(header) if c.startswith(prefix)]
        if positions:
            quantity = name
            break
    if not positions:
        raise LayoutError("MaxQuant proteinGroups.txt with no per-sample intensity column")
    removed: Dict[str, int] = {}
    keys: List[str] = []
    features: List[Dict[str, str]] = []
    kept: List[List[str]] = []
    flags = (("Reverse", "decoy (Reverse)"), ("Potential contaminant", "potential contaminant"),
             ("Contaminant", "potential contaminant"), ("Only identified by site", "only identified by site"))
    for row in rows[start + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        why = next((label for name, label in flags
                    if name in column and column[name] < len(row) and row[column[name]].strip() == "+"), None)
        if why:
            removed[why] = removed.get(why, 0) + 1
            continue
        get = lambda name: row[column[name]].strip() if name in column and column[name] < len(row) else ""
        majority = get("Majority protein IDs")
        keys.append(majority.split(";")[0] or get("Protein IDs").split(";")[0])
        features.append({
            "accession": majority.split(";")[0], "accessions": get("Protein IDs"),
            "gene_symbol": get("Gene names").split(";")[0], "gene_symbols": get("Gene names"),
            "description": get("Protein names") or get("Fasta headers"),
            "peptides": get("Peptides"), "unique_peptides": get("Unique peptides"),
        })
        kept.append(row)
    columns = [header[i] for i in positions]
    prefix = next(p for p, n in _MQ_FAMILIES if n == quantity)
    return MsTable(
        exporter="maxquant_proteins", kind=KIND_PROTEIN, source=str(path), quantity=quantity,
        columns=columns, labels=[c[len(prefix):].strip() for c in columns],
        keys=_unique(keys), features=features, values=_matrix(kept, positions),
        log_header=_header_says_log(columns), removed=removed,
    )


# --------------------------------------------------------------------------
# FragPipe combined_protein.tsv
# --------------------------------------------------------------------------

def _fp_is_header(row: List[str]) -> bool:
    return "Protein ID" in row and ("Entry Name" in row or "Protein" in row) and any(
        c.endswith(" Intensity") for c in row)


def _read_fragpipe_proteins(path: Path, rows: List[List[str]], start: int) -> MsTable:
    header = [cell.strip() for cell in rows[start]]
    column = {name: i for i, name in enumerate(header)}
    positions = [i for i, c in enumerate(header) if c.endswith(" MaxLFQ Intensity")]
    quantity, suffix = "MaxLFQ intensity", " MaxLFQ Intensity"
    if not positions:
        positions = [i for i, c in enumerate(header)
                     if c.endswith(" Intensity") and not c.endswith(("Total Intensity", "Unique Intensity", "Razor Intensity"))]
        quantity, suffix = "intensity", " Intensity"
    if not positions:
        raise LayoutError("FragPipe combined_protein.tsv with no per-sample intensity column")
    keys, features, kept = [], [], []
    for row in rows[start + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        get = lambda name: row[column[name]].strip() if name in column and column[name] < len(row) else ""
        accession = get("Protein ID")
        keys.append(accession)
        features.append({
            "accession": accession, "accessions": accession + (";" + get("Indistinguishable Proteins") if get("Indistinguishable Proteins") else ""),
            "gene_symbol": get("Gene"), "description": get("Protein Description") or get("Description"),
            "entry_name": get("Entry Name"),
        })
        kept.append(row)
    columns = [header[i] for i in positions]
    return MsTable(
        exporter="fragpipe_proteins", kind=KIND_PROTEIN, source=str(path), quantity=quantity,
        columns=columns, labels=[c[: -len(suffix)].strip() for c in columns],
        keys=_unique(keys), features=features, values=_matrix(kept, positions),
        log_header=_header_says_log(columns),
    )


# --------------------------------------------------------------------------
# DIA-NN report.pg_matrix.tsv
# --------------------------------------------------------------------------

_DIANN_ANNOTATION = ("Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description")


def _diann_is_header(row: List[str]) -> bool:
    return "Protein.Group" in row and "Genes" in row


def _read_diann_proteins(path: Path, rows: List[List[str]], start: int) -> MsTable:
    header = [cell.strip() for cell in rows[start]]
    column = {name: i for i, name in enumerate(header)}
    positions = [i for i, c in enumerate(header) if c and c not in _DIANN_ANNOTATION]
    if not positions:
        raise LayoutError("DIA-NN pg_matrix with no run column")
    keys, features, kept = [], [], []
    for row in rows[start + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        get = lambda name: row[column[name]].strip() if name in column and column[name] < len(row) else ""
        group = get("Protein.Group")
        keys.append(group.split(";")[0])
        features.append({
            "accession": group.split(";")[0], "accessions": get("Protein.Ids") or group,
            "gene_symbol": get("Genes").split(";")[0], "gene_symbols": get("Genes"),
            "description": get("First.Protein.Description"), "entry_name": get("Protein.Names"),
        })
        kept.append(row)
    columns = [header[i] for i in positions]
    labels = [re.sub(r"\.(raw|d|wiff|mzML|dia)$", "", Path(c.replace("\\", "/")).name, flags=re.I) for c in columns]
    return MsTable(
        exporter="diann_proteins", kind=KIND_PROTEIN, source=str(path), quantity="protein group quantity",
        columns=columns, labels=labels, keys=_unique(keys), features=features,
        values=_matrix(kept, positions), log_header=_header_says_log(columns),
    )


#: (name, header test, reader), tried in order.
LAYOUTS: Tuple[Tuple[str, Callable, Callable], ...] = (
    ("proteome_discoverer", _pd_is_header, _read_proteome_discoverer),
    ("maxquant_proteins", _mq_is_header, _read_maxquant_proteins),
    ("fragpipe_proteins", _fp_is_header, _read_fragpipe_proteins),
    ("diann_proteins", _diann_is_header, _read_diann_proteins),
)


def read_export(path: Union[str, Path]) -> MsTable:
    """Read one exporter table. Raises PLM-0227 when no layout recognises it."""
    from paralome.errors import ParalomeError

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ParalomeError(104, f"Mass-spec export not found: {path}")
    rows = _rows(path)
    for name, is_header, reader in LAYOUTS:
        start = _header_index(rows, is_header)
        if start >= 0:
            try:
                table = reader(path, rows, start)
            except LayoutError as error:
                raise ParalomeError(227, f"{path}: {error}") from error
            if not table.keys:
                raise ParalomeError(227, f"{path}: read as {name}, but it holds no features")
            return table
    first = ", ".join(cell.strip() for cell in (rows[0] if rows else [])[:8])
    raise ParalomeError(
        227,
        f"{path}: no known exporter layout recognises this table. Its first row begins: {first}",
    )
