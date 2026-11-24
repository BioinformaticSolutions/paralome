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

"""`paralome proteomics`: the stages, in order, each checkpointed by the core.

    ms_import            the export(s) and the sample sheet -> data/proteomics/<role>/
    protein_identity     every feature to an Ensembl gene, from the genome's GTF
    ms_normalize         log scale tested, per-sample scaling, the paired-loading test
    ms_qc                sample totals, missing values, detection, on/off features
    protein_differential Q1, protein_change: the core statistics on proteins,
                         missing values at half the feature's lowest
    phospho_differential Q2, phospho_change: the same on phosphopeptides; Q3,
                         phospho_vs_protein_change: the 2x2 ANOVA interaction
                         (measure x group) per phosphopeptide whose protein
                         was measured; the combined table of all three
    statistics_output    the results for OmicView (paralome.browser.arm): the zip in
                         <project>/omicview/statistics_output, and Evidome's in
                         omicview/functional_evidence
    evidence             Evidome, through the core evidence stage: one run per
                         statistic (and stratum), on one table holding every question

The data-level stages (import to QC) write to the parent project's
``data/proteomics/`` (``layout.parent_folder_holding``): every analysis of one
export under one parent project reads the one import. The differential stages
and Evidome write into ``--project``.

A stage whose checkpoint records the same inputs (files by size and time, the
sheet, the genome's GTF, this method) and whose outputs exist is reused; any
change recomputes it and everything after it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from paralome.massspec import exporters, prepare
from paralome.massspec.exporters import KIND_PHOSPHO, KIND_PROTEIN

METHOD = "paralome.massspec.v1"
SUBDIR = "proteomics"
ROLE_PROTEINS = "proteins"
ROLE_PHOSPHO = "phospho"
#: The question each result answers, in its file names:
#: Q1 did the protein change, Q2 did the phosphopeptide change, Q3 did the
#: phosphopeptide change differently from its protein; the three read together.
PROTEIN_CHANGE = "protein_change"
PHOSPHO_CHANGE = "phospho_change"
PHOSPHO_VS_PROTEIN_CHANGE = "phospho_vs_protein_change"
EVIDOME_NAME = "proteomics"
QUESTION = {ROLE_PROTEINS: PROTEIN_CHANGE, ROLE_PHOSPHO: PHOSPHO_CHANGE}
#: Q3 is a two-group question: it runs beside t and paired-t.
TWO_GROUP_STATISTICS = ("t", "paired-t")
#: Panel follows a design another statistic names.
PANEL_ALONE = ("--statistic panel runs beside the statistic whose design it follows: name it with t, paired-t, lm "
               "or anova (e.g. --statistic anova,panel)")
TWO_GROUP_FOLDERS = ("t", "paired_t")
UNIT = "scaled abundance"


def stamp(question: str) -> str:
    """What every result name ends with: its question."""
    return question


def _plan_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    from paralome.pipeline.jsonio import write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(path), dict(payload), atomic=True, default=str)


def _reusable(path: Path, plan_id: str, files: Sequence[str] = ()) -> Optional[Dict[str, object]]:
    try:
        stored = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if stored.get("complete") and stored.get("plan_id") == plan_id and all(
            Path(str(f)).is_file() for f in (files or stored.get("files") or [])):
        return stored
    return None


def _write_matrix(path: Path, keys: Sequence[str], sample_ids: Sequence[str], values: np.ndarray) -> None:
    import pandas as pd

    frame = pd.DataFrame(values, columns=list(sample_ids))
    frame.insert(0, "feature", list(keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g")


def _read_matrix(path: Path):
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", na_values=["NA"], keep_default_na=False, dtype={"feature": str})
    return list(frame["feature"]), [c for c in frame.columns[1:]], frame.iloc[:, 1:].to_numpy(dtype=float)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], fields: Optional[Sequence[str]] = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or (list(rows[0]) if rows else []))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


# --------------------------------------------------------------------------
# ms_import + protein_identity
# --------------------------------------------------------------------------

def import_table(source: Path, role: str, sheet, folder: Path, *, gtf: str, echo=print) -> Dict[str, object]:
    """Read one export into ``folder``: raw linear values in sheet order, features with identity."""
    from paralome.checkpoints import stage_checkpoint_path
    from paralome.errors import ParalomeError
    from paralome.massspec.identity import resolve_proteins
    from paralome.massspec.samplesheet import match_columns
    from paralome.singlecell.stats_core import input_file_identity

    source = Path(source).expanduser().resolve()
    plan_id = _plan_id({
        "source": str(source), "source_identity": input_file_identity(source),
        "sheet": sheet.path, "sheet_identity": input_file_identity(sheet.path),
        "gtf": gtf, "gtf_identity": input_file_identity(gtf), "role": role, "method": METHOD,
    })
    checkpoint = stage_checkpoint_path(folder, "ms_import")
    abundance, features_path = folder / "abundance.tsv", folder / "features.tsv"
    stored = _reusable(checkpoint, plan_id, [str(abundance), str(features_path)])
    if stored:
        echo(f"[paralome] ms_import {role}: reused ({folder})")
        return stored
    table = exporters.read_export(source)
    want = KIND_PROTEIN if role == ROLE_PROTEINS else KIND_PHOSPHO
    if table.kind != want:
        raise ParalomeError(227, f"{source}: given as --{role}, but it is a {table.kind} table "
                                 f"({table.exporter})")
    order = match_columns(sheet, table.columns, table.labels, export=str(source))
    raw = table.values[:, order]
    header_text = " ".join(table.columns)
    decision = prepare.decide_scale(raw, table.log_header, header_text)
    linear = prepare.to_linear(raw, decision.scale)
    notes = list(table.notes)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_means = np.nanmean(raw, axis=1) if decision.scale == "linear" else None
    if row_means is not None and np.isfinite(row_means).any() and np.nanmedian(np.abs(row_means - 100.0)) < 0.5:
        notes.append("the exporter scaled every feature to a mean of 100 across the samples, so the table "
                     "holds no feature intensities: the missing-by-intensity QC cannot show detection limits")
    identities, report = resolve_proteins(table.features, gtf)
    features = [{"feature": key, **identity, **feature}
                for key, identity, feature in zip(table.keys, identities, table.features)]
    _write_matrix(abundance, table.keys, sheet.sample_ids, linear)
    fields = ["feature", "gene_id", "gene_name", "identity_evidence", "identity_note"] + [
        k for k in table.features[0] if k not in ("gene_id", "gene_name")]
    _write_rows(features_path, features, fields)
    record = {
        "schema": "paralome.massspec.import.v1", "stage": "ms_import", "complete": True, "plan_id": plan_id,
        "role": role, "source": str(source), "sheet": sheet.path, "exporter": table.exporter,
        "kind": table.kind, "quantity": table.quantity, "columns": table.columns, "labels": table.labels,
        "scale": decision.scale, "scale_reason": decision.reason, "n_features": len(table.keys),
        "removed": table.removed, "notes": notes, "identity": report,
        "files": [str(abundance), str(features_path)],
    }
    _write_checkpoint(stage_checkpoint_path(folder, "protein_identity"), {
        "schema": "paralome.massspec.identity.v1", "stage": "protein_identity", "complete": True,
        "plan_id": plan_id, **report, "files": [str(features_path)],
    })
    _write_checkpoint(checkpoint, record)
    echo(f"[paralome] ms_import {role}: {table.exporter}, {len(table.keys):,} {table.kind} features, "
         f"{len(sheet.sample_ids)} samples; values {decision.scale} ({decision.reason})")
    for note in notes:
        echo(f"[paralome] ms_import {role}: {note}")
    echo(f"[paralome] protein_identity {role}: {report['mapped']:,} / {report['total']:,} "
         f"({report['fraction']:.1%}) given an Ensembl gene; {report['by_evidence']}")
    return record


# --------------------------------------------------------------------------
# ms_normalize + ms_qc
# --------------------------------------------------------------------------

def normalize(data: Path, roles: Sequence[str], sheet, *, echo=print, alpha: float = prepare.PAIRED_P) -> Dict[str, object]:
    """Scale every table; a phospho table by the protein factors only when the loadings pair."""
    from paralome.checkpoints import stage_checkpoint_path

    folder = data / SUBDIR
    imports = {role: json.loads(stage_checkpoint_path(folder / role, "ms_import").read_text()) for role in roles}
    plan_id = _plan_id({role: imports[role]["plan_id"] for role in roles} | {"method": METHOD, "pairing_alpha": float(alpha)})
    checkpoint = stage_checkpoint_path(folder, "ms_normalize")
    stored = _reusable(checkpoint, plan_id)
    if stored:
        echo(f"[paralome] ms_normalize: reused ({folder})")
        return stored
    groups = prepare.design_groups(sheet.rows)
    linear = {role: _read_matrix(folder / role / "abundance.tsv") for role in roles}
    factors: Dict[str, List[float]] = {}
    basis: Dict[str, str] = {}
    pairing = None
    protein_factors, basis[ROLE_PROTEINS] = prepare.scaling_factors(linear[ROLE_PROTEINS][2])
    factors[ROLE_PROTEINS] = list(protein_factors)
    if ROLE_PHOSPHO in roles:
        test = prepare.pairing_test(linear[ROLE_PROTEINS][2], linear[ROLE_PHOSPHO][2], groups, alpha=alpha)
        pairing = test.as_dict()
        if test.paired:
            factors[ROLE_PHOSPHO] = list(protein_factors)
            basis[ROLE_PHOSPHO] = "the protein table's factors (the loadings pair)"
        else:
            own, basis[ROLE_PHOSPHO] = prepare.scaling_factors(linear[ROLE_PHOSPHO][2])
            factors[ROLE_PHOSPHO] = list(own)
        echo(f"[paralome] ms_normalize: {test.reason}")
    files = []
    for role in roles:
        keys, samples, values = linear[role]
        scaled = values * np.asarray(factors[role])[None, :]
        path = folder / role / "scaled.tsv"
        _write_matrix(path, keys, samples, scaled)
        files.append(str(path))
    record = {
        "schema": "paralome.massspec.normalize.v1", "stage": "ms_normalize", "complete": True,
        "plan_id": plan_id, "sample_ids": sheet.sample_ids, "design_groups": groups,
        "factors": factors, "basis": basis, "pairing": pairing, "files": files,
    }
    _write_checkpoint(checkpoint, record)
    write_qc(data, roles, sheet, groups, factors, pairing)
    return record


def write_qc(data: Path, roles: Sequence[str], sheet, groups: Sequence[str], factors, pairing) -> None:
    """ms_qc: the tables that say what the values are before anything is tested."""
    from paralome.checkpoints import stage_checkpoint_path

    folder = data / SUBDIR / "qc"
    sample_rows, bins, on_off = [], [], []
    group_arr = np.asarray(list(groups))
    for role in roles:
        keys, samples, values = _read_matrix(data / SUBDIR / role / "abundance.tsv")
        measured = np.isfinite(values)
        complete = measured.all(axis=1)
        totals = values[complete].sum(axis=0) if complete.any() else np.nansum(values, axis=0)
        offsets = prepare.loading_offsets(values)
        for j, sample in enumerate(samples):
            column = values[:, j][measured[:, j]]
            sample_rows.append({
                "table": role, "sample_id": sample, "design_group": groups[j],
                "total_before_scaling": f"{totals[j]:.6g}", "relative_total": f"{totals[j] / totals.mean():.4f}",
                "loading_offset_log2": f"{offsets[j]:.4f}", "scaling_factor": f"{factors[role][j]:.4f}",
                "features_measured": int(measured[:, j].sum()), "features_missing": int((~measured[:, j]).sum()),
                "lowest_measured": f"{column.min():.6g}" if column.size else "",
                "median_measured": f"{np.median(column):.6g}" if column.size else "",
            })
        # Missing values against intensity: are the missing ones the faint ones?
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a feature measured nowhere has no mean
            mean_log = np.nanmean(np.log2(values + 1.0), axis=1)
        some = np.isfinite(mean_log)
        if some.sum() >= 10:
            edges = np.quantile(mean_log[some], np.linspace(0, 1, 11))
            which = np.clip(np.searchsorted(edges, mean_log[some], side="right") - 1, 0, 9)
            missing_fraction = (~measured[some]).mean(axis=1)
            for b in range(10):
                in_bin = which == b
                if in_bin.any():
                    bins.append({"table": role, "intensity_decile": b + 1,
                                 "mean_log2_from": f"{edges[b]:.3f}", "mean_log2_to": f"{edges[b + 1]:.3f}",
                                 "features": int(in_bin.sum()),
                                 "fraction_of_values_missing": f"{missing_fraction[in_bin].mean():.4f}"})
        # On/off: measured in at least half of one group, in no sample of another.
        unique = list(dict.fromkeys(groups))
        for i, key in enumerate(keys):
            present = [g for g in unique if measured[i, group_arr == g].sum() >= np.ceil((group_arr == g).sum() / 2)]
            absent = [g for g in unique if not measured[i, group_arr == g].any()]
            if present and absent:
                on_off.append({"table": role, "feature": key, "measured_in": ";".join(present),
                               "absent_from": ";".join(absent)})
    _write_rows(folder / "samples.tsv", sample_rows)
    _write_rows(folder / "missing_by_intensity.tsv", bins,
                ["table", "intensity_decile", "mean_log2_from", "mean_log2_to", "features", "fraction_of_values_missing"])
    _write_rows(folder / "on_off_features.tsv", on_off, ["table", "feature", "measured_in", "absent_from"])
    _write_checkpoint(stage_checkpoint_path(folder, "ms_qc"), {
        "schema": "paralome.massspec.qc.v1", "stage": "ms_qc", "complete": True, "pairing": pairing,
        "files": [str(folder / n) for n in ("samples.tsv", "missing_by_intensity.tsv", "on_off_features.tsv")],
    })


# --------------------------------------------------------------------------
# protein_differential / phospho_differential
# --------------------------------------------------------------------------

def _features(rows: Sequence[Mapping[str, str]], key_column: str, annotation_fields: Sequence[str]):
    from paralome.results import Features

    return Features(
        key=[row["feature"] for row in rows], key_column=key_column,
        gene_id=[row.get("gene_id", "") for row in rows], gene_name=[row.get("gene_name", "") for row in rows],
        annotation=[{name: row.get(name, "") for name in annotation_fields} for row in rows],
    )


_PROTEIN_ANNOTATION = ("accession", "accessions", "identity_evidence", "identity_note", "description")
_PHOSPHO_ANNOTATION = ("accession", "sites", "phospho_count", "localized", "min_localization_probability",
                       "sequence", "modifications", "identity_evidence", "description")


def _two_group_comparisons(results: Sequence[object]):
    """[(comparison, control positions, treatment positions)] of the run's two-group tests.

    Every level against its factor's control level, from the first two-group
    statistic the run asked for (t, else paired-t), so Q3 is answered for each
    --factor level.
    """
    for wanted in TWO_GROUP_STATISTICS:
        found = []
        for result in results:
            comparison = result.comparison
            if result.statistic != wanted or comparison.kind != "contrast":
                continue
            control = list(comparison.control_index) or [c for c, _t, _p in comparison.pairs]
            treatment = list(comparison.treatment_index) or [t for _c, t, _p in comparison.pairs]
            if control and treatment:
                found.append((comparison, control, treatment))
        if found:
            return found
    return []


def phospho_vs_protein(
    two_group, phospho: np.ndarray, protein: np.ndarray, eligible: np.ndarray,
    sample_ids: Sequence[str], keys: Sequence[str], *, data_level: str, alpha_metric: str,
    alpha_value: float,
):
    """Q3: does the phosphopeptide change differently from its protein?

    The core two-way ANOVA (measure: protein, phospho x group: the comparison's
    control and treated level) over the comparison's samples, each measured twice; only its
    interaction F-test is kept -- one pooled error over the four cells. With
    protein and control the reference levels its log2_fold_change is
    (phospho treatment - control) - (protein treatment - control).
    Returns (the finalized result, its Samples).
    """
    from paralome.comparisons import KIND_FTEST
    from paralome.engine import finalize
    from paralome.massspec.differential import fit_by_measured_samples
    from paralome.results import Samples
    from paralome.singlecell.stats_core import PSEUDOBULK_PSEUDOCOUNT, PseudobulkComparison, plan_statistics
    from paralome.stats import StatisticRequest

    comparison, control, treatment = two_group
    rows, columns = [], []
    # Protein and the comparison's control level are the two factors' control
    # levels, named as every --factor names one.
    for measure, matrix in (("protein", protein), ("phospho", phospho)):
        for level, positions in ((comparison.control, control), (comparison.treatment, treatment)):
            for position in positions:
                rows.append({"sample_id": f"{sample_ids[position]}_{measure}", "measure": measure,
                             "group": str(level)})
                columns.append(matrix[:, position])
    linear = np.column_stack(columns)
    with np.errstate(invalid="ignore"):
        logged = np.log2(linear + PSEUDOBULK_PSEUDOCOUNT)
    stacked = PseudobulkComparison(
        feature_ids=list(keys), annotation=[{} for _ in keys], sample_ids=[row["sample_id"] for row in rows],
        sample_rows=rows, n_cells=np.ones(len(rows)), cpm=linear, log2_cpm=logged,
    )
    request = StatisticRequest(("anova",), data_level)
    two_factors = ("measure:protein", f"group:{comparison.control}")
    plan = plan_statistics(rows, request, factors=two_factors)
    fitted = fit_by_measured_samples(stacked, request, eligible=eligible, alpha=alpha_value,
                                     factors=two_factors)(plan)
    interaction = next(r for r in fitted if r.kind == KIND_FTEST and ":" in str(r.comparison.term))
    result = replace(interaction, comparison=replace(
        interaction.comparison, name=comparison.name, control=comparison.control,
        treatment=comparison.treatment,
        description=(f"phosphopeptide change relative to its protein's change, {comparison.treatment} vs "
                     f"{comparison.control}: the measure (phospho, protein) x group interaction of a "
                     "two-way ANOVA, pooled error over the four cells"),
    ))
    finalize([result], alpha_metric=alpha_metric, alpha_value=alpha_value)
    return result, Samples(ids=list(stacked.sample_ids), values=linear, unit=UNIT)


def _on_rows(result, positions: np.ndarray, n_rows: int):
    """``result`` re-laid on ``n_rows`` rows: row ``positions[i]`` holds its feature i, the rest untested."""
    columns = {}
    for name, values in result.columns.items():
        values = np.asarray(values)
        if values.dtype == bool:
            full = np.zeros(n_rows, dtype=bool)
        elif values.dtype.kind in "fiu":
            full = np.full(n_rows, np.nan)
        else:
            full = np.full(n_rows, "", dtype=object)
        full[positions] = values
        columns[name] = full
    return replace(result, columns=columns)


def _named(result, question: str):
    """The result as one block of the proteomics Evidome table: its comparison, then its question."""
    return replace(result, comparison=replace(result.comparison, name=f"{result.name}_{question}"))


def _add_to_manifest(manifest: Mapping[str, object], paths: Sequence[str]) -> None:
    """List files this arm writes beside a statistic folder's own in its manifest, so the core's
    stale-result sweep removes them with the folder's results."""
    from paralome.pipeline.jsonio import write_json

    where = Path(str(manifest["manifest"]))
    recorded = json.loads(where.read_text())
    recorded["files"] = list(dict.fromkeys([*recorded.get("files", []), *paths]))
    write_json(where, recorded, atomic=True, stamp=True)


def differential(
    data: Path, project: Path, role: str, sheet, *, request, statistic: str,
    factors: Sequence[str], pair_by: Optional[str],
    covariate: Optional[str], stratify_by: Optional[str], alpha_metric: str, alpha_value: float,
    file_prefix: str, phospho_follows: bool = False,
    protein_results: Optional[Mapping[Optional[str], Sequence[object]]] = None,
    collected: Optional[Dict[Optional[str], List[object]]] = None, echo=print,
    block_by: Optional[str] = None, subdir: str = SUBDIR, evidome_name: str = EVIDOME_NAME,
    unit: str = UNIT, group_columns: Optional[Sequence[str]] = None,
    stratify_value: Optional[str] = None, annotation: Optional[Sequence[str]] = None,
    values_for=None, panel: bool = False, threads: int = 0, panel_plans: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """One table, missing values at half the feature's minimum: core statistics, core writer, core checkpoint.

    Written in <project>/<stage>/ (no half_min folder, no _missing_half_min in a name).

    ``panel`` (--statistic ...,panel) -- each stratum's design also gets its panels
    (paralome.massspec.panel), on the values its statistics are tested on, in <stratum>/panel/; Across ``threads`` worker processes (0: every CPU this process may use). Given ``panel_plans``
    (--backend slurm), each stratum's panels are written as a plan and its path appended, for a SLURM job to run.

    For the Olink arm: ``annotation`` (the features' columns carried into the results, the role's
    own when not given) and ``values_for(scope)`` -- the matrix of all the run's samples a question (the run, or a
    stratum) is tested on, or None for the table's own (Olink: a plate's own values when the question is on one
    plate).

    The Olink arm runs its table through here too -- ``subdir`` (its folder under data/),
    ``evidome_name``, ``unit``, ``group_columns`` (the design groups a feature must be measured in half of one
    of; the sheet's shared columns when not given) and ``block_by`` (--block-by).

    Proteins answer Q1 (protein_change). Phosphopeptides answer Q2
    (phospho_change) and, beside it, Q3 (phospho_vs_protein_change) for each
    phosphopeptide whose protein was measured; with them, the combined table
    and the one Evidome table of all three. ``protein_results`` are Q1's
    results by stratum, as ``collected`` returns them; ``phospho_follows``
    says a phosphopeptide table will be run after this protein table, so
    Evidome waits for the combined table. Each record lists the Evidome tables to run under ``evidome_tables``.
    """
    from paralome.atac.naming import run_scoped_prefix
    from paralome.checkpoints import stage_checkpoint_path
    from paralome.engine import by_folder
    from paralome.massspec.differential import build_comparison, fit_by_measured_samples
    from paralome.results import Features, _evidome_table, _stamped, write_all_statistics
    from paralome.singlecell.stats_core import compute_statistics, sweep_unrequested_statistics
    from paralome.singlecell.strata import stratum_slugs
    from paralome.version import __version__

    folder = data / subdir
    stage = "protein_differential" if role == ROLE_PROTEINS else "phospho_differential"
    keys, samples, scaled = _read_matrix(folder / role / "scaled.tsv")
    features = _read_rows(folder / role / "features.tsv")
    rows = [dict(row) for row in sheet.rows]
    if group_columns is not None:
        groups = ["|".join(f"{c}={row.get(c, '')}" for c in group_columns) for row in rows]
    else:
        groups = prepare.design_groups(sheet.rows)
    link = p_scaled = None
    if role == ROLE_PHOSPHO and (folder / ROLE_PROTEINS / "scaled.tsv").is_file():
        p_keys, _s, p_scaled = _read_matrix(folder / ROLE_PROTEINS / "scaled.tsv")
        where = {k: i for i, k in enumerate(p_keys)}
        link = np.asarray([where.get(row.get("accession", ""), -1) for row in features])
        p_features = _read_rows(folder / ROLE_PROTEINS / "features.tsv")
    strata = [None]
    if stratify_by:
        # Several columns, comma lists, blanks in none (paralome.stratify).
        from paralome.stratify import row_strata

        strata = sorted({key for row in rows for key in row_strata(row, stratify_by)})
        if stratify_value:
            # --stratify-value, as the single-cell fan-out reads it (raw or slug, comma list).
            from paralome.errors import ParalomeError
            from paralome.singlecell.strata import resolve_stratum_names

            try:
                strata = resolve_stratum_names([stratify_value], strata, flag="--stratify-value")
            except ValueError as error:
                raise ParalomeError(243, str(error)) from error           # PLM-0243
    slugs = stratum_slugs([s for s in strata if s is not None]) if stratify_by else {}
    feature_table = _features(features, "protein" if role == ROLE_PROTEINS else "phosphopeptide",
                              tuple(annotation) if annotation is not None else
                              _PROTEIN_ANNOTATION if role == ROLE_PROTEINS else _PHOSPHO_ANNOTATION)
    question = QUESTION[role]
    records = []
    captured: Dict[Optional[str], List[object]] = {}
    skipped_strata: List[Dict[str, str]] = []
    if stratify_by:
        from paralome.errors import ParalomeError

        # As the single-cell fan-out does, a stratum no requested statistic can be planned on
        # (its control level absent, one level of the factor) is skipped with the reason; none left is refused.
        from paralome.singlecell.stats_core import plan_each_statistic

        runnable_strata = []
        for value in strata:
            scope_rows = [row for row in rows if value in row_strata(row, stratify_by)]
            able, refused = plan_each_statistic(scope_rows, request, factors=factors, pair_by=pair_by,
                                                covariate=covariate, block_by=block_by)
            if able:
                runnable_strata.append(value)
            else:
                skipped_strata.append({"stratum": value, "reason": "; ".join(
                    f"{folder}: {why}" for folder, why in sorted(refused.items()))})
                echo(f"[paralome] {stage}: stratum {value} skipped -- {skipped_strata[-1]['reason']}")
        if not runnable_strata:
            raise ParalomeError(                               # PLM-0230
                230, f"--stratify-by {stratify_by!r} produced no testable strata; all {len(strata)} were skipped: "
                     + "; ".join(f"{item['stratum']} ({item['reason']})" for item in skipped_strata),
                fix="Read each stratum's reason above and change what it names",
            )
        strata = runnable_strata
    filled: Dict[int, np.ndarray] = {}

    def the_values(matrix: np.ndarray) -> np.ndarray:
        if id(matrix) not in filled:
            filled[id(matrix)] = prepare.half_minimum_fill(matrix)
        return filled[id(matrix)]

    out = project / stage
    computed_by_stratum = {}
    panels_by_stratum: Dict[Optional[str], Dict[str, object]] = {}
    q3_by_stratum: Dict[Optional[str], Dict[str, object]] = {}
    evidome_tables: List[Dict[str, object]] = []
    for value in strata:
        captured[value] = []
        scope = [i for i, row in enumerate(rows) if value is None or value in row_strata(row, stratify_by)]
        sub_rows = [rows[i] for i in scope]
        chosen = values_for(scope) if values_for is not None else None
        base = scaled if chosen is None else chosen
        values = the_values(base)
        eligible = prepare.testable(base[:, scope], [groups[i] for i in scope])
        comparison = build_comparison(
            keys, [{} for _ in keys], values[:, scope], sub_rows,
            stratify_by=stratify_by, stratify_value=value, all_rows=rows,
        )
        target = out if value is None else out / slugs[value]
        prefix = file_prefix if value is None else run_scoped_prefix(file_prefix, None, slugs[value])
        target.mkdir(parents=True, exist_ok=True)
        computed = compute_statistics(
            comparison, request, statistic_text=statistic, factors=factors, pair_by=pair_by,
            covariate=covariate, covariate_feature=None, min_mean_cpm=0.0,
            alpha_metric=alpha_metric, alpha_value=alpha_value, output_dir=target,
            file_prefix=prefix, stamp=stamp(question), features=feature_table,
            version=__version__, stage=stage,
            one_stratum_of_many=value is not None,
            fit_values=fit_by_measured_samples(
                comparison, request, eligible=eligible, alpha=alpha_value, factors=factors,
                pair_by=pair_by, covariate=covariate, captured=captured[value], block_by=block_by,
            ),
            sample_unit=unit, block_by=block_by,
        )
        computed_by_stratum[value] = computed
        if panel:
            from paralome.massspec.panel import FOLDER as PANEL_FOLDER, run_panels

            panel_args = (np.asarray(comparison.log2_cpm, dtype=float), keys, features, sub_rows)
            panel_kwargs = dict(factors=factors, covariate=covariate, blocks_by=block_by or pair_by,
                                eligible=eligible, folder=target / PANEL_FOLDER, prefix=prefix,
                                stamp=stamp(question), version=__version__)
            if panel_plans is not None:
                from paralome.massspec.panel import write_panel_plan

                plan = write_panel_plan(*panel_args, **panel_kwargs)
                panel_plans.append(plan)
                panels_by_stratum[value] = {"plan": plan, "submitted_as_a_job": True}
            else:
                panels_by_stratum[value] = run_panels(*panel_args, **panel_kwargs, echo=echo, threads=threads)
        results = captured[value]
        if collected is not None:
            collected[value] = results
        manifests = computed.get("statistics") or {}
        if role == ROLE_PROTEINS:
            if not phospho_follows:
                # No phosphopeptides: Evidome runs on Q1's own table.
                for statistic_folder, manifest in manifests.items():
                    if (manifest or {}).get("evidome_input"):
                        evidome_tables.append({
                            "folder": statistic_folder, "stratum": value, "slug": slugs.get(value),
                            "path": manifest["evidome_input"],
                            "name": f"{prefix}_{statistic_folder}_{evidome_name}",
                        })
            continue

        # Q3, beside Q2 in this stratum: one per two-group comparison.
        q3_results: Dict[str, object] = {}
        q3_stamp = stamp(PHOSPHO_VS_PROTEIN_CHANGE)
        two_groups = _two_group_comparisons(results) if link is not None else []
        if two_groups:
            p_values = prepare.half_minimum_fill(p_scaled)
            matched = np.where(link[:, None] >= 0, p_values[np.maximum(link, 0)], np.nan)
            unfilled = np.where(link[:, None] >= 0, p_scaled[np.maximum(link, 0)], np.nan)
            both = (link >= 0) & eligible & prepare.testable(unfilled[:, scope], [groups[i] for i in scope])
            q3_samples = None
            for two_group in two_groups:
                q3, q3_samples = phospho_vs_protein(
                    two_group, values[:, scope], matched[:, scope], both, [row["sample_id"] for row in sub_rows],
                    keys, data_level=request.data_level, alpha_metric=alpha_metric, alpha_value=alpha_value,
                )
                q3_results[q3.name] = q3
            q3_manifest = write_all_statistics(
                list(q3_results.values()), root=target, file_prefix=prefix, stamp=q3_stamp,
                features=feature_table, samples=q3_samples, version=__version__, alpha_metric=alpha_metric,
                alpha_value=alpha_value, stage=f"{stage} ({PHOSPHO_VS_PROTEIN_CHANGE})",
            )
            q3_by_stratum[value] = {"statistics": q3_manifest, "comparisons": list(q3_results)}
        else:
            why = ("no protein table was given" if link is None else
                   f"it asks a two-group question; {statistic} makes no two-group comparison "
                   "(it runs with t or paired-t)")
            sweep_unrequested_statistics(target, prefix, q3_stamp, [], stage=stage)
            q3_by_stratum[value] = {"not_run": why}
            echo(f"[paralome] {stage}: {PHOSPHO_VS_PROTEIN_CHANGE} not run -- {why}")

        # The one Evidome table of all three questions (no combined copy for a human: the viewer reads the zip).
        protein_by_folder = by_folder((protein_results or {}).get(value, []))
        for statistic_folder, phospho_group in by_folder(results).items():
            manifest = manifests.get(statistic_folder)
            if not manifest:
                continue
            protein_group = protein_by_folder.get(statistic_folder, [])
            written = []
            blocks = []
            n_proteins = len(p_features) if link is not None else 0
            union = Features(
                key=[row["feature"] for row in (p_features if link is not None else [])] + list(keys),
                key_column="feature",
                gene_id=[row.get("gene_id", "") for row in (p_features if link is not None else [])]
                + [row.get("gene_id", "") for row in features],
                gene_name=[row.get("gene_name", "") for row in (p_features if link is not None else [])]
                + [row.get("gene_name", "") for row in features],
            )
            on_proteins = np.arange(n_proteins)
            on_phospho = np.arange(len(features)) + n_proteins
            for result in protein_group if link is not None else []:
                blocks.append(_named(_on_rows(result, on_proteins, len(union)), PROTEIN_CHANGE))
            for result in phospho_group:
                blocks.append(_named(_on_rows(result, on_phospho, len(union)), PHOSPHO_CHANGE))
            if statistic_folder in TWO_GROUP_FOLDERS:
                for q3 in q3_results.values():
                    blocks.append(_named(_on_rows(q3, on_phospho, len(union)), PHOSPHO_VS_PROTEIN_CHANGE))
            table = _evidome_table(
                _stamped(target / statistic_folder, prefix, f"{statistic_folder}_evidome_input",
                         evidome_name, ".csv"),
                blocks, union, version=__version__, raw_fpkm=False, prefer_raw=True,
            )
            if table:
                written.append(table)
                evidome_tables.append({
                    "folder": statistic_folder, "stratum": value, "slug": slugs.get(value), "path": table,
                    "name": f"{prefix}_{statistic_folder}_{evidome_name}",
                })
            _add_to_manifest(manifest, written)
    if stratify_by:
        # The core's stratified checkpoint shape (pipeline.state.differential_statistics).
        computed = {"schema_version": 2, "strata": {v: c for v, c in computed_by_stratum.items()}}
    else:
        computed = computed_by_stratum[None]
    checkpoint = stage_checkpoint_path(out, stage)
    record = {
        "schema": "paralome.massspec.differential.v2", "stage": stage, "complete": True,
        "role": role, "question": question, "statistic": statistic,
        "factors": list(factors), "pair_by": pair_by, "covariate": covariate, "block_by": block_by,
        "stratify_by": stratify_by, **computed, "skipped_strata": skipped_strata,
        "evidome_tables": evidome_tables,
    }
    if panel:
        record["panel"] = panels_by_stratum if stratify_by else panels_by_stratum.get(None)
    if role == ROLE_PHOSPHO:
        record[PHOSPHO_VS_PROTEIN_CHANGE] = (q3_by_stratum if stratify_by else q3_by_stratum.get(None))
    _write_checkpoint(checkpoint, record)
    record["checkpoint"] = str(checkpoint)
    records.append(record)
    echo(f"[paralome] {stage}: written to {out}")
    return records


# --------------------------------------------------------------------------
# The whole arm
# --------------------------------------------------------------------------

def data_folder(project: Path, proteins: Path) -> Path:
    """The parent project's data folder holding this export's import, else the default."""
    from paralome.layout import beside_project, parent_folder_holding

    found = parent_folder_holding(project, subdir=f"{SUBDIR}/{ROLE_PROTEINS}", stage="ms_import",
                                  key="source", source=proteins)
    return found if found is not None else beside_project(project)


def run_proteomics(
    *, proteins: Path, phospho: Optional[Path], samples: Path, project: Path, gtf: str,
    statistic: str = "t", factors: Sequence[str] = (), pair_by: Optional[str] = None,
    covariate: Optional[str] = None, stratify_by: Optional[str] = None, alpha_metric: str = "p",
    alpha_value: float = 0.05, backend: str = "local", partition: Optional[str] = None,
    evidence: bool = True, echo=print, block_by: Optional[str] = None, threads: int = 0,
) -> Dict[str, object]:
    """Every stage of `paralome proteomics`, in order (--block-by)."""
    from paralome.atac.naming import derive_file_prefix
    from paralome.errors import ParalomeError
    from paralome.massspec.samplesheet import read_sample_sheet
    from paralome.singlecell.stats_core import parse_request

    project = Path(project).expanduser().resolve()
    proteins = Path(proteins).expanduser().resolve()
    sheet = read_sample_sheet(samples)
    from paralome.stats import factor_columns

    from paralome.stratify import stratum_columns

    for flag, column in (*(("--stratify-by", c) for c in stratum_columns(stratify_by)), ("--pair-by", pair_by),
                         ("--covariate", covariate), ("--block-by", block_by),
                         *(("--factor", f) for f in factor_columns(factors))):
        if column and column not in sheet.rows[0]:
            raise ParalomeError(228, f"{flag} {column!r} is not a column of the sample sheet {sheet.path} "
                                     f"(it has: {', '.join(sheet.metadata_columns())})")
    gtf = str(gtf)
    from paralome.massspec.panel import split_statistic

    statistic, panel = split_statistic(statistic)
    if panel and not statistic:
        raise ValueError(PANEL_ALONE)
    request = parse_request(statistic, factors=factors, pair_by=pair_by, covariate=covariate)
    data = data_folder(project, proteins)
    roles = [ROLE_PROTEINS] + ([ROLE_PHOSPHO] if phospho else [])
    echo(f"[paralome] proteomics: data folder {data}; project {project}")
    import_table(proteins, ROLE_PROTEINS, sheet, data / SUBDIR / ROLE_PROTEINS, gtf=gtf, echo=echo)
    if phospho:
        import_table(Path(phospho), ROLE_PHOSPHO, sheet, data / SUBDIR / ROLE_PHOSPHO, gtf=gtf, echo=echo)
    normalized = normalize(data, roles, sheet, echo=echo, alpha=alpha_value)
    prefix = derive_file_prefix(project)
    options = dict(request=request, statistic=statistic, factors=factors, pair_by=pair_by, covariate=covariate,
                   stratify_by=stratify_by, alpha_metric=alpha_metric, alpha_value=alpha_value,
                   file_prefix=prefix, echo=echo, block_by=block_by, panel=panel, threads=threads,
                   panel_plans=[] if panel and backend == "slurm" else None)
    protein_results: Dict[Optional[str], List[object]] = {}
    records = differential(data, project, ROLE_PROTEINS, sheet, phospho_follows=bool(phospho),
                           collected=protein_results, **options)
    if phospho:
        records += differential(data, project, ROLE_PHOSPHO, sheet, protein_results=protein_results, **options)
    # The browser bundle, as every arm's: the results of both questions' arms for OmicView.
    # The statistics_output stage, its zip in <project>/omicview/statistics_output.
    from paralome.browser.arm import STAGE as OUTPUT_STAGE, run_statistics_output, take_former_folder
    from paralome.pipeline.state import omicview_dir

    zips = omicview_dir(project)
    take_former_folder(project / "browser_bundle", project / OUTPUT_STAGE, OUTPUT_STAGE,
                       lambda text: echo(f"[paralome] {text}"))
    bundle = run_statistics_output(
        arm="proteomics", workdir=project / OUTPUT_STAGE, prefix=prefix,
        results_roots=[project / "protein_differential", project / "phospho_differential"],
        samples_table=Path(sheet.path), zip_dir=zips / OUTPUT_STAGE, echo=lambda text: echo(f"[paralome] {text}"),
    )
    panel_jobs = None
    if options["panel_plans"]:
        # --backend slurm -- the panels are SLURM jobs; the zip is built again when they end.
        from paralome.massspec.panel import submit_arm_panels

        panel_jobs = submit_arm_panels(
            options["panel_plans"], arm="proteomics", project=project, prefix=prefix,
            results_roots=[project / "protein_differential", project / "phospho_differential"],
            samples_table=Path(sheet.path), zip_dir=zips / OUTPUT_STAGE, threads=threads, partition=partition,
            echo=echo)
    evidence_records = []
    if evidence:
        from paralome.evidence import run_functional_evidence

        # One Evidome run per statistic (and stratum), on the table holding every question.
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
    return {"data": str(data), "project": str(project), "normalize": normalized, "panel_jobs": panel_jobs,
            "differential": records, "statistics_output": bundle, "evidence": evidence_records}
