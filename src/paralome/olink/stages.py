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

"""Every stage of `paralome olink`.

* The input is Olink's own per-plate long export (NPX Signature), one ``--plate`` per file, read by Olink's
  column names; the study design is in a sample sheet keyed by ``PlateID`` + ``SampleID``, mimicking the
  original exports. A file without Olink's columns is refused, naming what is missing.
* The value tested is ``QuantifiedValue`` (pg/mL), log2 through the shared statistics' log2(x + 1).
* Below the limit of detection, as in proteomics: a value Olink flags ``BelowLOD`` (or reports no
  concentration for) is missing, and set to half the assay's lowest measured value (``half_min``) through
  the proteomics arm's own differential.
* Plate bridging: every other plate is brought onto the first ``--plate``'s scale assay by assay, by the
  median ratio of the sample controls (``SAMPLE_CONTROL``) both plates ran, each control counted where both
  of its values are measured and above the LOD.
* Replicated samples are averaged: wells the sheet gives one ``sample_id`` are averaged (pg/mL, after
  bridging, over the wells that measured the assay).
* A run takes the wells its sheet lists; the plates' other study wells are left out of it and counted, so
  each question (controls, treatments, ...) is a sheet.
* --block-by: the subject term of anova and lm, for the repeated measures.
* Bridged only where a question spans plates: a question (the run, or one stratum of it) whose samples'
  wells are all on one plate is tested on that plate's own values (within a plate the bridged and unbridged
  values rank every sample alike, and an assay no shared control measured on both plates stays usable).
* Identity: an assay named by no gene symbol (Abeta40, Abeta42, pTau217) is found by its UniProt accession
  in the genome's Ensembl UniProt table, and ``measured_form`` keeps the assay's own name (the cleavage
  product or phospho-form, which an AI agent can use).

data/olink/ holds the plates' import (read, checked, bridged), shared by every analysis of those plates;
<project>/olink/proteins/ the run's samples, and the differential, statistics_output and Evidome are the
proteomics arm's (``massspec.stages.differential``), so the results read as every other arm's.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paralome.errors import ParalomeError

METHOD = "paralome.olink.v2"          # Unbridged_wells.tsv, measured_form
SUBDIR = "olink"
EVIDOME_NAME = "olink"
UNIT = "pg/mL, bridged"
VALUE = "QuantifiedValue"

#: The columns of Olink's long export Paralome reads (NPX Signature); the rest are carried, not needed.
REQUIRED = ("PlateID", "SampleID", "SampleType", "Assay", "UniProt", "OlinkID", VALUE, "BelowLOD", "Unit")
SAMPLE, CONTROL = "SAMPLE", "SAMPLE_CONTROL"
SHEET_KEYS = ("PlateID", "SampleID", "sample_id")


# --------------------------------------------------------------------------
# The exports
# --------------------------------------------------------------------------

def read_plate(path: Path):
    """One Olink long export as a DataFrame of Olink's columns (CSV with , ; or tab, an .xlsx, or .parquet)."""
    import pandas as pd

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ParalomeError(240, f"--plate {path}: no such file")                          # PLM-0240
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path, dtype=str)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path).astype(str)
    else:
        with path.open(newline="") as handle:
            head = handle.readline()
        sep = max((",", ";", "\t"), key=head.count)
        frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ParalomeError(                                                                 # PLM-0240
            240, f"{path}: not an Olink long export -- it has no {', '.join(missing)} column"
                 f"{'s' if len(missing) > 1 else ''} (Olink's columns are read by name: {', '.join(REQUIRED)})",
        )
    frame = frame[frame["Assay"].astype(str).str.strip().ne("") & frame["Assay"].notna()].copy()
    frame["source"] = str(path)
    return frame


def _number(text) -> float:
    try:
        value = float(str(text).strip())
    except ValueError:
        return float("nan")
    return value


def plate_values(frame) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, Dict[str, str]], Dict[Tuple[str, str], str]]:
    """``({(PlateID, SampleID): {OlinkID: pg/mL or NaN}}, {OlinkID: assay}, {(PlateID, SampleID): SampleType})``.

    A value is missing when Olink reports no concentration or flags it ``BelowLOD``."""
    values: Dict[Tuple[str, str], Dict[str, float]] = {}
    assays: Dict[str, Dict[str, str]] = {}
    kinds: Dict[Tuple[str, str], str] = {}
    for row in frame.to_dict("records"):
        well = (str(row["PlateID"]).strip(), str(row["SampleID"]).strip())
        oid = str(row["OlinkID"]).strip()
        assays.setdefault(oid, {"OlinkID": oid, "Assay": str(row["Assay"]).strip(),
                                "UniProt": str(row["UniProt"]).strip(), "Unit": str(row["Unit"]).strip()})
        kinds[well] = str(row["SampleType"]).strip()
        value = _number(row[VALUE])
        # Olink's .xlsx gives TRUE/FALSE, its CSV 1/0 or TRUE/FALSE (the plate_2 export is TRUE/FALSE).
        below = str(row["BelowLOD"]).strip().lower() in ("true", "1", "1.0", "yes")
        values.setdefault(well, {})[oid] = value if (np.isfinite(value) and value > 0 and not below) else float("nan")
    return values, assays, kinds


def bridge(plates: Sequence[str], values, kinds, assay_ids: Sequence[str]):
    """Each plate's factor per assay onto the first plate: the median, over the sample controls both ran, of
    first-plate pg/mL / this plate's pg/mL. ``({plate: {OlinkID: factor}}, report rows)``."""
    reference = plates[0]
    controls = {p: {s for (pp, s), k in kinds.items() if pp == p and k == CONTROL} for p in plates}
    factors: Dict[str, Dict[str, float]] = {reference: {oid: 1.0 for oid in assay_ids}}
    rows: List[Dict[str, object]] = []
    for plate in plates[1:]:
        shared = sorted(controls[plate] & controls[reference])
        if not shared:
            raise ParalomeError(                                                             # PLM-0241
                241, f"plate {plate} ran no SAMPLE_CONTROL that the first --plate ({reference}) also ran "
                     f"({plate}: {', '.join(sorted(controls[plate])) or 'none'}; {reference}: "
                     f"{', '.join(sorted(controls[reference])) or 'none'}), so it cannot be bridged to it",
            )
        factors[plate] = {}
        for oid in assay_ids:
            ratios = []
            for sample in shared:
                ref_value = values.get((reference, sample), {}).get(oid, float("nan"))
                this = values.get((plate, sample), {}).get(oid, float("nan"))
                if np.isfinite(ref_value) and np.isfinite(this):
                    ratios.append(ref_value / this)
            factor = float(np.median(ratios)) if ratios else float("nan")
            factors[plate][oid] = factor
            rows.append({"plate": plate, "reference_plate": reference, "OlinkID": oid, "controls": ";".join(shared),
                         "controls_used": len(ratios), "factor": factor,
                         "log2_factor": float(np.log2(factor)) if np.isfinite(factor) else float("nan")})
    return factors, rows


# --------------------------------------------------------------------------
# The sample sheet
# --------------------------------------------------------------------------

def read_sheet(path: Path) -> List[Dict[str, str]]:
    """The run's wells: PlateID, SampleID, sample_id, then the design's columns (CSV or tab)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ParalomeError(242, f"--samples {path}: no such file")                         # PLM-0242
    with path.open(newline="") as handle:
        head = handle.readline()
        handle.seek(0)
        rows = [{str(k).strip(): str(v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(handle, delimiter="\t" if head.count("\t") > head.count(",") else ",")]
    faults = []
    if not rows:
        faults.append("it has no rows")
    else:
        missing = [c for c in SHEET_KEYS if c not in rows[0]]
        if missing:
            faults.append(f"it has no {', '.join(missing)} column (an Olink sheet names each well by PlateID and "
                          "SampleID, and the sample it is by sample_id)")
        seen = set()
        for number, row in enumerate(rows, start=2):
            well = (row.get("PlateID", ""), row.get("SampleID", ""))
            if not all(well):
                faults.append(f"line {number}: PlateID and SampleID are both needed")
            elif well in seen:
                faults.append(f"line {number}: well {well[0]} {well[1]} is listed twice")
            seen.add(well)
    if faults:
        raise ParalomeError(242, f"--samples {path}: " + "; ".join(faults[:12]))            # PLM-0242
    return rows


def samples_of(sheet: Sequence[Mapping[str, str]], kinds: Mapping[Tuple[str, str], str],
               needed: Sequence[str] = ()):
    """``(one row per sample_id, {sample_id: [wells]}, wells of the plates left out, {column: samples without a
    value})``; refuses a sheet that names a well the plates do not hold, or replicates whose design columns disagree.

    One sheet for every question of the plates. Its control wells are passed over (they bridge
    the plates), and a sample with no value in a column the run uses (``needed``: the --factor columns,
    --covariate, --pair-by, --block-by) is not in this run -- The --stratify-by columns leave theirs out at the strata (paralome.stratify)."""
    faults = []
    wells_of: Dict[str, List[Tuple[str, str]]] = {}
    rows: Dict[str, Dict[str, str]] = {}
    blank: Dict[str, List[str]] = {}
    for row in sheet:
        well = (row["PlateID"], row["SampleID"])
        if well not in kinds:
            faults.append(f"{well[0]} {well[1]} is in no --plate")
            continue
        if kinds[well] != SAMPLE:
            continue
        if not row.get("sample_id"):
            faults.append(f"{well[0]} {well[1]} has no sample_id (a control well may leave it blank; a sample may not)")
            continue
        empty = [column for column in needed if not str(row.get(column, "")).strip()]
        if empty:
            for column in empty:
                if row["sample_id"] not in blank.setdefault(column, []):
                    blank[column].append(row["sample_id"])
            continue
        design = {k: v for k, v in row.items() if k not in ("PlateID", "SampleID")}
        sid = row["sample_id"]
        if sid in rows and rows[sid] != design:
            differs = sorted(k for k in design if rows[sid].get(k) != design.get(k))
            faults.append(f"sample_id {sid}: its wells disagree on {', '.join(differs)} -- wells averaged into one "
                          "sample must be the same sample")
            continue
        rows.setdefault(sid, design)
        wells_of.setdefault(sid, []).append(well)
    if faults:
        raise ParalomeError(243, "The sample sheet does not match the plates: " + "; ".join(faults[:12])  # PLM-0243
                            + (f"; ... {len(faults) - 12} more" if len(faults) > 12 else ""))
    listed = {(r["PlateID"], r["SampleID"]) for r in sheet}
    left_out = sorted(w for w, k in kinds.items() if k == SAMPLE and w not in listed)
    if not rows:
        raise ParalomeError(243, "No sample of the sheet is in this run: "               # PLM-0243
                            + ("; ".join(f"{len(ids)} have no {column}" for column, ids in blank.items())
                               or "it lists no SAMPLE well"))
    return list(rows.values()), wells_of, left_out, blank


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def import_plates(plates: Sequence[Path], folder: Path, *, gtf: str, uniprot_table: Optional[str] = None,
                  echo=print) -> Dict[str, object]:
    """olink_import: read, check, bridge every plate; data/olink/ (shared by every analysis of these plates)."""
    from paralome.checkpoints import stage_checkpoint_path
    from paralome.massspec.identity import resolve_proteins
    from paralome.massspec.stages import _reusable, _write_checkpoint
    from paralome.singlecell.stats_core import input_file_identity

    plates = [Path(p).expanduser().resolve() for p in plates]
    plan_id = _digest({"plates": [str(p) for p in plates], "identity": [input_file_identity(p) for p in plates],
                       "gtf": gtf, "gtf_identity": input_file_identity(gtf), "method": METHOD,
                       "uniprot_table": str(uniprot_table) if uniprot_table else None,
                       "uniprot_identity": input_file_identity(uniprot_table) if uniprot_table else None})
    checkpoint = stage_checkpoint_path(folder, "olink_import")
    stored = _reusable(checkpoint, plan_id)
    if stored:
        echo(f"[paralome] olink_import: reused ({folder})")
        return stored
    import pandas as pd

    frames = [read_plate(p) for p in plates]
    order: List[str] = []
    for path, frame in zip(plates, frames):
        ids = list(dict.fromkeys(frame["PlateID"].astype(str).str.strip()))
        clash = [i for i in ids if i in order]
        if clash:
            raise ParalomeError(240, f"{path}: PlateID {', '.join(clash)} is in an earlier --plate too")  # PLM-0240
        order += ids
    frame = pd.concat(frames, ignore_index=True)
    values, assays, kinds = plate_values(frame)
    units = {a["Unit"] for a in assays.values()}
    assay_ids = sorted(assays)
    factors, bridge_rows = bridge(order, values, kinds, assay_ids)
    wells = sorted(kinds)
    matrix = np.full((len(assay_ids), len(wells)), np.nan)
    unbridged = np.full((len(assay_ids), len(wells)), np.nan)
    for j, well in enumerate(wells):
        for i, oid in enumerate(assay_ids):
            unbridged[i, j] = values.get(well, {}).get(oid, float("nan"))
            matrix[i, j] = unbridged[i, j] * factors[well[0]].get(oid, float("nan"))
    folder.mkdir(parents=True, exist_ok=True)
    # Each plate's own values beside the bridged ones, for a question within one plate.
    raw = pd.DataFrame(unbridged, columns=[f"{p}\t{s}" for p, s in wells])
    raw.insert(0, "OlinkID", assay_ids)
    raw.to_csv(folder / "unbridged_wells.tsv", sep="\t", index=False, na_rep="NA", float_format="%.17g")
    table = pd.DataFrame(matrix, columns=[f"{p}\t{s}" for p, s in wells])
    table.insert(0, "OlinkID", assay_ids)
    table.to_csv(folder / "bridged_wells.tsv", sep="\t", index=False, na_rep="NA", float_format="%.17g")
    _write_tsv(folder / "wells.tsv", [{"PlateID": p, "SampleID": s, "SampleType": kinds[(p, s)]} for p, s in wells],
               ["PlateID", "SampleID", "SampleType"])
    _write_tsv(folder / "bridging.tsv", bridge_rows,
               ["plate", "reference_plate", "OlinkID", "controls", "controls_used", "factor", "log2_factor"])
    features = [{"accession": assays[o]["UniProt"], "gene_symbol": assays[o]["Assay"],
                 "description": f"Olink {assays[o]['Assay']} ({o})"} for o in assay_ids]
    identities, report = resolve_proteins(features, gtf, uniprot_table=uniprot_table)
    rows = [{"feature": o, **ident, "assay": assays[o]["Assay"], "OlinkID": o, "accession": assays[o]["UniProt"],
             "measured_form": measured_form(assays[o]["Assay"], ident),
             "description": f"Olink {assays[o]['Assay']} ({o})"} for o, ident in zip(assay_ids, identities)]
    _write_tsv(folder / "features.tsv", rows, ["feature", "gene_id", "gene_name", "identity_evidence",
                                                "identity_note", "assay", "OlinkID", "accession", "measured_form",
                                                "description"])
    unbridged = sorted({r["OlinkID"] for r in bridge_rows if not np.isfinite(r["factor"])})
    present = {p: sorted({o for (pp, _s), v in values.items() if pp == p for o, x in v.items()}) for p in order}
    record = {
        "schema": "paralome.olink.import.v1", "stage": "olink_import", "complete": True, "plan_id": plan_id,
        "plates": [str(p) for p in plates], "source": str(plates[0]), "plate_order": order,
        "reference_plate": order[0], "value": VALUE, "units": sorted(units), "assays": len(assay_ids),
        "assays_missing_from": {p: sorted(set(assay_ids) - set(present[p])) for p in order
                                if set(assay_ids) - set(present[p])},
        "wells": {p: {k: sum(1 for (pp, _s), kk in kinds.items() if pp == p and kk == k) for k in (SAMPLE, CONTROL)}
                  for p in order},
        "below_lod_or_missing": int(sum(1 for v in values.values() for x in v.values() if not np.isfinite(x))),
        "unbridged_assays": unbridged, "identity": report,
        "files": [str(folder / n) for n in ("bridged_wells.tsv", "unbridged_wells.tsv", "wells.tsv", "bridging.tsv",
                                            "features.tsv")],
    }
    _write_checkpoint(checkpoint, record)
    echo(f"[paralome] olink_import: {len(plates)} plate(s) ({', '.join(order)}), {len(assay_ids)} assays, "
         f"values {VALUE} ({', '.join(sorted(units))}); bridged to {order[0]}")
    for p in order[1:]:
        used = [r for r in bridge_rows if r["plate"] == p and np.isfinite(r["factor"])]
        logs = [r["log2_factor"] for r in used]
        echo(f"[paralome] olink_import: {p} -> {order[0]} by {used[0]['controls'] if used else 'no control'}: "
             f"median log2 factor {np.median(logs):+.3f} over {len(used)} assays" if logs else
             f"[paralome] olink_import: {p}: no assay bridged")
    if unbridged:
        echo(f"[paralome] olink_import: not bridged (no control measured on both plates), left missing on the "
             f"plates that need it: {', '.join(unbridged)}")
    echo(f"[paralome] protein_identity olink: {report['mapped']:,} / {report['total']:,} given an Ensembl gene; "
         f"{report['by_evidence']}")
    return record


def measured_form(assay: str, identity: Mapping[str, str]) -> str:
    """The form of the protein an assay measures, when its name is not its gene's: Olink's Abeta40 on
    APP, pTau217 on MAPT. Blank when the assay is named for its gene, or has none."""
    from paralome.report_schema import MISSING_VALUE

    gene = str(identity.get("gene_name") or "").strip()
    if identity.get("gene_id") in (None, "", MISSING_VALUE) or not gene:
        return ""
    return "" if str(assay).strip().lower() == gene.lower() else str(assay).strip()


def prepare_samples(data: Path, project: Path, sheet_path: Path, *, needed: Sequence[str] = (),
                    echo=print) -> Dict[str, object]:
    """The run's samples: the sheet's wells, averaged by sample_id -> <project>/olink/proteins/scaled.tsv."""
    import pandas as pd

    from paralome.massspec.stages import _write_matrix

    folder = data / SUBDIR
    table = pd.read_csv(folder / "bridged_wells.tsv", sep="\t", na_values=["NA"], keep_default_na=False,
                        dtype={"OlinkID": str})
    own = pd.read_csv(folder / "unbridged_wells.tsv", sep="\t", na_values=["NA"], keep_default_na=False,
                      dtype={"OlinkID": str})
    kinds = {}
    for row in csv.DictReader((folder / "wells.tsv").open(newline=""), delimiter="\t"):
        kinds[(row["PlateID"], row["SampleID"])] = row["SampleType"]
    sheet = read_sheet(sheet_path)
    rows, wells_of, left_out, blank = samples_of(sheet, kinds, needed)
    keys = list(table["OlinkID"])
    out = np.full((len(keys), len(rows)), np.nan)
    out_own = np.full((len(keys), len(rows)), np.nan)
    averaged = []
    plates_of = []
    for j, row in enumerate(rows):
        wells = wells_of[row["sample_id"]]
        plates_of.append({"sample_id": row["sample_id"], "plates": ";".join(sorted({p for p, _s in wells}))})
        for source, target_matrix in ((table, out), (own, out_own)):
            block = source[[f"{p}\t{s}" for p, s in wells]].to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    target_matrix[:, j] = np.nanmean(block, axis=1)
        if len(wells) > 1:
            averaged.append({"sample_id": row["sample_id"], "wells": ";".join(f"{p}:{s}" for p, s in wells)})
    target = Path(project) / SUBDIR / "proteins"
    target.mkdir(parents=True, exist_ok=True)
    _write_matrix(target / "scaled.tsv", keys, [r["sample_id"] for r in rows], out)
    _write_matrix(target / "unbridged.tsv", keys, [r["sample_id"] for r in rows], out_own)
    _write_tsv(target / "sample_plates.tsv", plates_of, ["sample_id", "plates"])
    (target / "features.tsv").write_text((folder / "features.tsv").read_text())
    _write_tsv(target / "averaged_wells.tsv", averaged, ["sample_id", "wells"])
    _write_tsv(target / "left_out_wells.tsv", [{"PlateID": p, "SampleID": s} for p, s in left_out],
               ["PlateID", "SampleID"])
    _write_tsv(target / "samples_not_in_this_run.tsv",
               [{"sample_id": sid, "no_value_in": column} for column, ids in blank.items() for sid in ids],
               ["sample_id", "no_value_in"])
    fields = list(rows[0]) if rows else ["sample_id"]
    _write_tsv(target / "samples.tsv", rows, fields)
    echo(f"[paralome] olink_samples: {len(rows)} samples from {sum(len(w) for w in wells_of.values())} wells"
         f"{f' ({len(averaged)} averaged from replicate wells)' if averaged else ''}; "
         f"{len(left_out)} study well(s) of the plates not in this sheet, left out")
    for column, ids in blank.items():
        echo(f"[paralome] olink_samples: {len(ids)} sample(s) with no {column} are not in this run")
    return {"rows": rows, "averaged": averaged, "left_out": len(left_out), "folder": str(target),
            "plates": [r["plates"].split(";") for r in plates_of],
            "not_in_this_run": {column: len(ids) for column, ids in blank.items()}}


#: The Olink assays' annotation in the results (measured_form, the protein form an assay measures).
OLINK_ANNOTATION = ("accession", "identity_evidence", "identity_note", "measured_form", "description")


def values_by_plate(folder: Path, plates: Sequence[Sequence[str]], *, echo=print):
    """The matrix each question is tested on: its plate's own values when every one of its samples'
    wells is on one plate, the bridged values when they span plates. Returns ``values_for(scope)`` for
    ``massspec.stages.differential``: ``scope`` the question's sample positions, the matrix all the run's
    samples."""
    from paralome.massspec.stages import _read_matrix

    _keys, _samples, own = _read_matrix(folder / "unbridged.tsv")
    said = set()

    def values_for(scope: Sequence[int]):
        on = sorted({p for i in scope for p in plates[i]})
        if len(on) == 1:
            if on[0] not in said:
                said.add(on[0])
                echo(f"[paralome] olink: a question on {on[0]} alone is tested on its own values (not bridged)")
            return own
        return None

    return values_for


class _Sheet:
    """What massspec.stages.differential reads of a sample sheet."""

    def __init__(self, path: str, rows: List[Dict[str, str]]):
        self.path, self.rows = path, rows

    @property
    def sample_ids(self) -> List[str]:
        return [row["sample_id"] for row in self.rows]

    def metadata_columns(self) -> List[str]:
        return [c for c in (self.rows[0] if self.rows else {}) if c != "sample_id"]


def data_folder(project: Path, first_plate: Path) -> Path:
    """The parent project's data folder holding these plates' import, else the default."""
    from paralome.layout import beside_project, parent_folder_holding

    found = parent_folder_holding(project, subdir=SUBDIR, stage="olink_import", key="source", source=first_plate)
    return found if found is not None else beside_project(project)


def run_olink(
    *, plates: Sequence[Path], samples: Path, project: Path, gtf: str, statistic: str,
    factors: Sequence[str] = (), pair_by: Optional[str] = None, block_by: Optional[str] = None,
    covariate: Optional[str] = None, stratify_by: Optional[str] = None, stratify_value: Optional[str] = None,
    alpha_metric: str = "p",
    alpha_value: float = 0.05, backend: str = "local", partition: Optional[str] = None, evidence: bool = True,
    uniprot_table: Optional[str] = None, echo=print, threads: int = 0,
) -> Dict[str, object]:
    """Every stage of `paralome olink`, in order (``threads``, the panels' worker processes)."""
    from paralome.atac.naming import derive_file_prefix
    from paralome.massspec import stages as ms
    from paralome.singlecell.stats_core import parse_request
    from paralome.stats import factor_columns

    project = Path(project).expanduser().resolve()
    plates = [Path(p).expanduser().resolve() for p in plates]
    if not plates:
        raise ParalomeError(240, "give each Olink plate export with --plate")                # PLM-0240
    from paralome.massspec.panel import split_statistic

    statistic, panel = split_statistic(statistic)        # --statistic ...,panel
    if panel and not statistic:
        raise ValueError(ms.PANEL_ALONE)
    request = parse_request(statistic, factors=factors, pair_by=pair_by, covariate=covariate)
    data = data_folder(project, plates[0])
    echo(f"[paralome] olink: data folder {data}; project {project}")
    from paralome.stratify import stratum_columns

    sheet_rows = read_sheet(Path(samples))
    columns = [c for c in sheet_rows[0] if c not in SHEET_KEYS]
    named = [(flag, column) for flag, column in (
        *(("--stratify-by", c) for c in stratum_columns(stratify_by)), ("--pair-by", pair_by),
        ("--block-by", block_by), ("--covariate", covariate), *(("--factor", f) for f in factor_columns(factors)))
        if column]
    for flag, column in named:
        if column not in columns:
            raise ParalomeError(243, f"{flag} {column!r} is not a column of the sample sheet {samples} "  # PLM-0243
                                     f"(it has: {', '.join(columns)})")
    imported = import_plates(plates, data / SUBDIR, gtf=str(gtf), uniprot_table=uniprot_table, echo=echo)
    prepared = prepare_samples(data, project, Path(samples), echo=echo,
                               needed=[column for flag, column in named if flag != "--stratify-by"])
    sheet = _Sheet(str(Path(samples).expanduser().resolve()), prepared["rows"])
    prefix = derive_file_prefix(project)
    panel_plans = [] if panel and backend == "slurm" else None       # SLURM jobs
    records = ms.differential(
        project, project, ms.ROLE_PROTEINS, sheet, request=request, statistic=statistic, factors=factors,
        pair_by=pair_by, covariate=covariate, stratify_by=stratify_by, alpha_metric=alpha_metric,
        alpha_value=alpha_value, file_prefix=prefix, echo=echo, block_by=block_by, subdir=SUBDIR,
        evidome_name=EVIDOME_NAME, unit=UNIT, group_columns=factor_columns(factors), stratify_value=stratify_value,
        annotation=OLINK_ANNOTATION,
        values_for=values_by_plate(Path(prepared["folder"]), prepared["plates"], echo=echo), panel=panel, threads=threads,
        panel_plans=panel_plans,
    )
    from paralome.browser.arm import STAGE as OUTPUT_STAGE, run_statistics_output
    from paralome.pipeline.state import omicview_dir

    zips = omicview_dir(project)
    bundle = run_statistics_output(
        arm="olink", workdir=project / OUTPUT_STAGE, prefix=prefix,
        results_roots=[project / "protein_differential"],
        samples_table=Path(prepared["folder"]) / "samples.tsv", zip_dir=zips / OUTPUT_STAGE,
        echo=lambda text: echo(f"[paralome] {text}"),
    )
    panel_jobs = None
    if panel_plans:
        from paralome.massspec.panel import submit_arm_panels

        panel_jobs = submit_arm_panels(
            panel_plans, arm="olink", project=project, prefix=prefix, results_roots=[project / "protein_differential"],
            samples_table=Path(prepared["folder"]) / "samples.tsv", zip_dir=zips / OUTPUT_STAGE, threads=threads,
            partition=partition, echo=echo)
    evidence_records = []
    if evidence:
        from paralome.evidence import run_functional_evidence

        for record in records:
            for table in record["evidome_tables"]:
                workdir = project / "evidence" / record["stage"]
                if table.get("slug"):
                    workdir = workdir / table["slug"]
                evidence_records.append(run_functional_evidence(
                    Path(table["path"]), workdir / table["folder"], table["name"],
                    backend=backend, partition=partition, echo=echo, zip_dir=zips / "functional_evidence",
                    alpha=alpha_value,
                ))
    return {"data": str(data), "project": str(project), "import": imported, "samples": prepared,
            "panel_jobs": panel_jobs,
            "differential": records, "statistics_output": bundle, "evidence": evidence_records}
