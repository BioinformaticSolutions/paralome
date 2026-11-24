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

"""Each gene along each lineage's pseudotime, by the run's factors.

the
per-segment ANOVA the stage ran is dropped ("covered by the main design"); "Code in the full model
and we can test that"; the spline's df 3, with a setting; "install LS as default and monocle's as a switch".

For each lineage, each gene's log2(CPM + pseudocount) in each of the lineage's cells is modelled as the run's full
factorial (``stats.design_matrix``: every --factor, every interaction, treatment-coded from each factor's control
level) crossed with a natural spline of the cell's pseudotime on that lineage (``stats.natural_spline_basis``, df 3
unless ``--trajectory-model <engine>:df=<n>``). The terms tested, each Type II as the run's ANOVA tests them
(``stats.term_reference_columns``):

* each factorial term (A, A:B ...): the factors' effect on the gene's level along the lineage;
* ``pseudotime``: the gene changes along the lineage;
* each factorial term crossed with it (A:pseudotime, A:B:pseudotime ...): the factors change how the gene moves
  along the lineage -- the question the stage is for.

Two engines fit the same models:

* ``ls`` (the default): least squares on the cells' log2(CPM + pseudocount), every gene of a lineage in one solve
  (``stats.crossproduct_anova``); F, df, p, and BH q per term over the genes tested.
* ``monocle3``: Monocle3's own ``fit_models`` (a negative binomial GLM of each gene's counts, the cell's size factor
  as its offset) for each model the tests need, and each term by Monocle3's own comparison of two fits (its
  ``compare_models``: twice the difference of the log-likelihoods, on the difference of their residual df);
  LLR, df, p, and BH q per term over the genes tested. Negative binomial, not Monocle3's default quasi-Poisson:
  a quasi-Poisson fit has no log-likelihood, so compare_models gives it no p.

A gene is tested on a lineage when its mean CPM over the replicates (each replicate's cells on the lineage pooled)
is at least the run's --min-mean-cpm, the expression floor the run's statistics use.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paralome.checkpoints import atomic_text_writer, atomic_write_text
from paralome.errors import ParalomeError

ENGINES = ("ls", "monocle3")
DEFAULT_ENGINE = "ls"
DEFAULT_SPLINE_DF = 3
TIME_TERM = "pseudotime"
# Lineage minimisation is a switch (fate the default), so the choices can be time-trialled.
LINEAGE_SETS = ("fate", "exit", "route")
DEFAULT_LINEAGE_SET = "fate"
GENE_MODELS_R_SCRIPT = Path(__file__).resolve().parent / "r" / "monocle3_gene_models.R"
MISSING = "."


def parse_model(text: Optional[str]) -> Tuple[str, int]:
    """--trajectory-model: ``ls`` (the default) or ``monocle3``, its spline's df after a colon (``ls:df=4``)."""
    raw = str(text or DEFAULT_ENGINE).strip()
    engine, _colon, rest = raw.partition(":")
    engine = engine.strip().lower()

    def refuse(why: str) -> ParalomeError:
        return ParalomeError(                                   # PLM-0128
            128, f"--trajectory-model {raw!r}: {why}. Give ls or monocle3, its spline's df after a colon: ls:df=4",
            fix="--trajectory-model ls, monocle3, or e.g. ls:df=4",
        )

    if engine not in ENGINES:
        raise refuse(f"{engine or 'nothing'} is not an engine")
    df = DEFAULT_SPLINE_DF
    for part in (p.strip() for p in rest.split(",") if p.strip()):
        key, equals, value = part.partition("=")
        if key.strip().lower() != "df" or not equals:
            raise refuse(f"{part!r} is not a setting it knows (df=<n>)")
        try:
            df = int(value.strip())
        except ValueError:
            raise refuse(f"df={value.strip()!r} is not a whole number") from None
        if df < 1:
            raise refuse("df must be 1 or more (1 is a straight line)")
    return engine, df


def parse_lineage_set(text: Optional[str]) -> str:
    """--trajectory-lineages: fate (the default), exit or route."""
    value = str(text or DEFAULT_LINEAGE_SET).strip().lower()
    if value not in LINEAGE_SETS:
        raise ParalomeError(                                   # PLM-0129
            129, f"--trajectory-lineages {text!r} is not fate, exit or route",
            fix="--trajectory-lineages fate, exit or route",
        )
    return value


# --- the lineages the statistics follow -----------------------------------------------------------------------------

def _clusters(text: object) -> List[str]:
    return [part.strip() for part in str(text or "").split("->") if part.strip()]


def _name(text: str) -> str:
    from paralome.atac.singlecell import sanitize_sample_id

    return sanitize_sample_id(text)


def group_lineages(cells: pd.DataFrame, lineages: pd.DataFrame, lineage_set: str,
                   root_value: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The fit's routes as the lineages the statistics follow: (cells, lineages).

    ``route``: every route the fit made, as it made them. ``fate``: one lineage per end state -- every route ending
    in the same cluster, named for it. ``exit``: one per way out of the root and end state -- the routes that leave
    the root cluster into the same cluster and end in the same one, named ``<exit>_to_<end>``. A cell is on a
    grouped lineage when it is on any of its routes, its pseudotime there its earliest on them (on a route the
    shortest way to its node, Monocle3's own). ``cells`` comes back with barcode, branch and pseudotime_<lineage>;
    ``lineages`` with lineage, routes (the routes it holds), clusters (its first route's), exit, end and cells.
    """
    lineage_set = parse_lineage_set(lineage_set)
    names = [str(n) for n in lineages["lineage"]]
    along = {str(row["lineage"]): _clusters(row.get("clusters")) for _, row in lineages.iterrows()}
    groups: Dict[str, List[str]] = {}
    exits: Dict[str, str] = {}
    ends: Dict[str, str] = {}
    for route in names:
        clusters = along.get(route) or []
        end = str(clusters[-1]) if clusters else route
        if "leaf_cluster" in lineages.columns:
            recorded = lineages.loc[lineages["lineage"].astype(str) == route, "leaf_cluster"].astype(str).iloc[0]
            end = recorded if recorded and recorded != "nan" else end
        leaving = [c for c in clusters if c != str(root_value)]
        exit_ = leaving[0] if leaving else str(root_value)
        if lineage_set == "route":
            key = route
        elif lineage_set == "fate":
            key = _name(end)
        else:
            key = f"{_name(exit_)}_to_{_name(end)}"
        groups.setdefault(key, []).append(route)
        exits.setdefault(key, exit_)
        ends.setdefault(key, end)
    out = pd.DataFrame({"barcode": cells["barcode"].astype(str).to_numpy()})
    on: Dict[str, np.ndarray] = {}
    for key, routes in groups.items():
        times = cells[[f"pseudotime_{r}" for r in routes if f"pseudotime_{r}" in cells.columns]]
        values = times.min(axis=1, skipna=True).to_numpy(dtype=float) if times.shape[1] else \
            np.full(len(cells), np.nan)
        out[f"pseudotime_{key}"] = values
        on[key] = np.isfinite(values)
    keys = list(groups)
    out["branch"] = [",".join(k for k in keys if on[k][i]) for i in range(len(out))]
    table = pd.DataFrame({
        "lineage": keys,
        "routes": [",".join(groups[k]) for k in keys],
        "clusters": [" -> ".join(along.get(groups[k][0]) or []) for k in keys],
        "exit": [exits[k] for k in keys],
        "end": [ends[k] for k in keys],
        "cells": [int(on[k].sum()) for k in keys],
    })
    return out, table


# --- the model --------------------------------------------------------------------------------------------------------

def lineage_design(rows: Sequence[Mapping[str, str]], factors: Sequence[str], pseudotime: np.ndarray,
                   df: int) -> Tuple[np.ndarray, List[str], Dict[str, List[int]], int]:
    """(X, column names, term -> columns, the spline's df) for one lineage's cells.

    The run's full factorial (``stats.design_matrix``, each factor's control level the one its --factor names),
    each of its columns repeated times each column of the natural spline of pseudotime. Terms: the factorial's,
    ``pseudotime`` (the intercept times the spline), and ``<term>:pseudotime``. Refuses (FactorSpecError) a lineage
    whose cells lack a factor's control level."""
    from paralome.stats import baseline_first_levels, design_matrix, natural_spline_basis, parse_factor

    specs = [parse_factor(spec) for spec in factors]
    levels = {column: baseline_first_levels(rows, column, control) for column, control in specs}
    base, base_names, base_terms = design_matrix(rows, [column for column, _ in specs], levels)
    spline = natural_spline_basis(pseudotime, df)
    columns = [base[:, j] for j in range(base.shape[1])]
    names = list(base_names)
    term_of = {j: term for term, cols in base_terms.items() for j in cols}
    terms: Dict[str, List[int]] = {term: list(cols) for term, cols in base_terms.items()}
    timed: Dict[str, List[int]] = {}
    for j, name in enumerate(base_names):
        term = TIME_TERM if j == 0 else f"{term_of[j]}:{TIME_TERM}"
        for k in range(spline.shape[1]):
            timed.setdefault(term, []).append(len(columns))
            columns.append(base[:, j] * spline[:, k])
            names.append(f"{TIME_TERM}[{k + 1}]" if j == 0 else f"{name}:{TIME_TERM}[{k + 1}]")
    terms[TIME_TERM] = timed.get(TIME_TERM, [])
    for term in base_terms:
        terms[f"{term}:{TIME_TERM}"] = timed.get(f"{term}:{TIME_TERM}", [])
    return np.column_stack(columns), names, terms, int(spline.shape[1])


def _log_values(counts, library: np.ndarray, pseudocount: float):
    """(log2(CPM + pseudocount) - log2(pseudocount) as a sparse genes x cells matrix, log2(pseudocount)): a cell
    with none of a gene's counts is 0 in the matrix, so the matrix stays as sparse as the counts."""
    from scipy import sparse

    values = sparse.csc_matrix(counts, dtype=float, copy=True)
    values.eliminate_zeros()
    cell_of = np.repeat(np.arange(values.shape[1]), np.diff(values.indptr))
    shift = float(np.log2(pseudocount))
    with np.errstate(divide="ignore", invalid="ignore"):
        values.data = np.log2(values.data / library[cell_of] * 1_000_000.0 + pseudocount) - shift
    return values, shift


def _crossproducts(values, shift: float, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(y'y, X'y) of every gene, y = values + shift."""
    n = x.shape[0]
    sums = np.asarray(values.sum(axis=1)).ravel()
    squares = np.asarray(values.multiply(values).sum(axis=1)).ravel()
    yty = squares + 2.0 * shift * sums + n * shift * shift
    xty = np.asarray(values @ x) + shift * x.sum(axis=0)[np.newaxis, :]
    return yty, xty


def _replicate_floor(counts, replicate_codes: np.ndarray, n_replicates: int) -> np.ndarray:
    """Each gene's mean CPM over the replicates, each replicate's cells pooled."""
    from scipy import sparse

    membership = sparse.csc_matrix((np.ones(replicate_codes.size), (np.arange(replicate_codes.size),
                                                                     replicate_codes)),
                                   shape=(replicate_codes.size, n_replicates))
    pooled = np.asarray((sparse.csr_matrix(counts) @ membership).todense(), dtype=float)
    library = pooled.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpm = np.where(library > 0, pooled / library * 1_000_000.0, 0.0)
    return cpm.mean(axis=1)


def _write_table(path: Path, genes, rows_tested: np.ndarray, mean_cpm: np.ndarray, results: Mapping[str, Any],
                 terms: Sequence[str], statistic: str, df_error: Optional[float]) -> None:
    """One lineage's table: every gene of the object, tested or not, and per term its statistic, df, p and q."""
    from paralome.stats import benjamini_hochberg

    n = len(genes)
    tested = np.zeros(n, dtype=bool)
    tested[rows_tested] = True
    columns: Dict[str, np.ndarray] = {}
    for term in terms:
        stat = np.full(n, np.nan)
        p = np.full(n, np.nan)
        df_term = np.full(n, np.nan)
        entry = results.get(term) or {}
        if entry:
            stat[rows_tested] = np.asarray(entry["statistic"], dtype=float)
            p[rows_tested] = np.asarray(entry["p"], dtype=float)
            df_term[rows_tested] = np.asarray(entry["df"], dtype=float)
        columns[f"{term}_{statistic}"] = stat
        columns[f"{term}_df"] = df_term
        columns[f"{term}_p"] = p
        columns[f"{term}_q"] = benjamini_hochberg(p)

    def cell(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        value = float(value)
        if not np.isfinite(value):
            return MISSING
        return repr(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.6g}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        header = ["feature_name", "gene_id", "common_id", "tested", "mean_cpm"]
        if df_error is not None:
            header.append("df_error")
        writer.writerow(header + list(columns))
        for i, gene in enumerate(genes):
            row = [gene.feature_name, gene.gene_id or MISSING, gene.common_id or MISSING, cell(tested[i]),
                   cell(mean_cpm[i])]
            if df_error is not None:
                row.append(cell(df_error) if tested[i] else MISSING)
            writer.writerow(row + [cell(columns[name][i]) for name in columns])


def table_name(prefix: str, lineage: str, engine: str) -> str:
    """<prefix>_trajectory_<lineage>_pseudotime_<engine>.tsv: the lineage, and the model that made it."""
    return f"{prefix}_trajectory_{lineage}_pseudotime_{engine}.tsv"


def run_gene_models(
    extraction_dir: Path, cells: pd.DataFrame, lineages: pd.DataFrame, sample_manifest: Path, out: Path, *,
    prefix: str, factors: Sequence[str], replicate_column: str, remove: Sequence[str] = (),
    gtf: Optional[Path] = None, engine: str = DEFAULT_ENGINE, df: int = DEFAULT_SPLINE_DF,
    min_mean_cpm: float = 1.0, pseudocount: float = 1.0, alpha: Tuple[str, float] = ("p", 0.05), jobs: int = 1,
    rscript: str = "Rscript", echo: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Every lineage's gene model (see the module's docstring); returns the record the stage keeps: per lineage its
    cells, genes tested, the table written and how long it took, or why it was not tested; and a summary table
    (lineage x term: df, genes tested, genes significant at the run's --alpha)."""
    from paralome.rna.singlecell import _with_gene_identity_from, load_singlecell_rna_inputs
    from paralome.stats import FactorSpecError, crossproduct_anova, factor_columns

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_singlecell_rna_inputs(extraction_dir)
    if gtf is not None:
        inputs = _with_gene_identity_from(inputs, gtf)
    position = {barcode: index for index, barcode in enumerate(inputs.cells)}
    library_all = np.asarray(inputs.matrix.sum(axis=0)).ravel().astype(float)
    metadata = inputs.metadata
    barcodes = metadata["barcode"].astype(str).to_numpy() if "barcode" in metadata.columns else \
        np.asarray(metadata.index, dtype=str)
    replicate_of = pd.Series(metadata[replicate_column].astype(str).to_numpy(), index=barcodes)

    # Each replicate's factor levels, from the run's sample manifest (one value each).
    manifest = pd.read_csv(sample_manifest, dtype=str, keep_default_na=False)
    columns = factor_columns(factors)
    levels_of: Dict[str, Dict[str, str]] = {}
    removed = {str(r) for r in remove}
    for replicate, rows in manifest.groupby(replicate_column) if replicate_column in manifest.columns else []:
        if str(replicate) in removed or removed & set(rows.get("sample_id", pd.Series(dtype=str))):
            continue
        values = {}
        for column in columns:
            seen = {v for v in rows.get(column, pd.Series(dtype=str)) if str(v).strip()}
            if len(seen) == 1:
                values[column] = next(iter(seen))
        if len(values) == len(columns):
            levels_of[str(replicate)] = values

    tables = out / engine
    record: Dict[str, Any] = {"engine": engine, "spline_df": df, "factors": list(factors), "lineages": {}}
    summary: List[Dict[str, Any]] = []
    jobs_for_r: List[Dict[str, Any]] = []
    for _, lineage_row in lineages.iterrows():
        lineage = str(lineage_row["lineage"])
        started = time.monotonic()
        times = cells[f"pseudotime_{lineage}"].to_numpy(dtype=float)
        on = np.isfinite(times)
        chosen = [(position[b], t, replicate_of.get(b)) for b, t, keep in
                  zip(cells["barcode"].astype(str), times, on) if keep and b in position]
        chosen = [(p, t, r) for p, t, r in chosen if isinstance(r, str) and r in levels_of]
        entry: Dict[str, Any] = {"cells": len(chosen)}
        if len(chosen) < 3:
            entry["not_tested"] = "fewer than 3 of its cells are in the run's replicates"
            record["lineages"][lineage] = entry
            echo(f"trajectories: genes on {lineage} not tested -- {entry['not_tested']}")
            continue
        positions = np.asarray([p for p, _, _ in chosen])
        pseudotime = np.asarray([t for _, t, _ in chosen], dtype=float)
        replicates = [r for _, _, r in chosen]
        rows = [levels_of[r] for r in replicates]
        try:
            x, names, terms, used_df = lineage_design(rows, factors, pseudotime, df)
        except FactorSpecError as error:
            entry["not_tested"] = str(error)
            record["lineages"][lineage] = entry
            echo(f"trajectories: genes on {lineage} not tested -- {error}")
            continue
        if used_df == 0:
            entry["not_tested"] = "every one of its cells has the same pseudotime"
            record["lineages"][lineage] = entry
            echo(f"trajectories: genes on {lineage} not tested -- {entry['not_tested']}")
            continue
        counts = inputs.matrix[:, positions]
        codes, unique = pd.factorize(pd.Series(replicates))
        mean_cpm = _replicate_floor(counts, np.asarray(codes), len(unique))
        tested_rows = np.flatnonzero(mean_cpm >= float(min_mean_cpm))
        entry.update(replicates=len(unique), genes_tested=int(tested_rows.size), spline_df=used_df,
                     columns=len(names))
        path = tables / table_name(prefix, lineage, engine)
        tested_terms = [t for t in terms]
        if engine == "ls":
            values, shift = _log_values(counts[tested_rows], library_all[positions], pseudocount)
            yty, xty = _crossproducts(values, shift, x)
            fitted = crossproduct_anova(yty, xty, x.T @ x, len(positions), terms)
            results = {term: {"statistic": fitted[term]["F"], "p": fitted[term]["p"],
                              "df": np.full(tested_rows.size, fitted[term]["df_term"])} for term in tested_terms}
            df_error = next(iter(fitted.values()))["df_error"] if fitted else None
            _write_table(path, inputs.genes, tested_rows, mean_cpm, results, tested_terms, "F", df_error)
            entry.update(table=str(path), seconds=round(time.monotonic() - started, 1), df_error=df_error)
            record["lineages"][lineage] = entry
            _summarise(summary, lineage, results, tested_terms, tested_rows.size, alpha)
            echo(f"trajectories: genes on {lineage} -- {len(positions):,} cells, {tested_rows.size:,} genes, "
                 f"{len(names)} columns, {entry['seconds']} s")
        else:
            jobs_for_r.append(dict(lineage=lineage, entry=entry, positions=positions, x=x, names=names,
                                   terms=terms, tested_rows=tested_rows, mean_cpm=mean_cpm, path=path,
                                   library=library_all[positions], started=started))
    if jobs_for_r:
        _monocle3_models(extraction_dir, out / "monocle3_work", jobs_for_r, inputs, jobs=jobs, rscript=rscript,
                         echo=echo)
        for job in jobs_for_r:
            results = job["results"]
            terms = list(job["terms"])
            _write_table(job["path"], inputs.genes, job["tested_rows"], job["mean_cpm"], results, terms, "LLR", None)
            job["entry"].update(table=str(job["path"]), seconds=job.get("seconds"))
            record["lineages"][job["lineage"]] = job["entry"]
            _summarise(summary, job["lineage"], results, terms, job["tested_rows"].size, alpha)
    # No summary table: the lineage tables are the result.
    atomic_write_text(out / f"gene_models_{engine}.json", json.dumps(record, indent=2, sort_keys=True,
                                                                     default=str) + "\n")
    return record


def _summarise(summary: List[Dict[str, Any]], lineage: str, results: Mapping[str, Any], terms: Sequence[str],
               tested: int, alpha: Tuple[str, float]) -> None:
    from paralome.stats import benjamini_hochberg

    metric, threshold = alpha
    for term in terms:
        p = np.asarray(results[term]["p"], dtype=float) if term in results else np.array([])
        values = benjamini_hochberg(p) if str(metric).lower().startswith("q") else p
        dfs = np.asarray(results[term]["df"], dtype=float) if term in results else np.array([])
        finite = dfs[np.isfinite(dfs)]
        summary.append({"lineage": lineage, "term": term,
                        "df": (int(np.median(finite)) if finite.size else MISSING),
                        "genes_tested": int(tested),
                        "genes_significant": int(np.sum(np.isfinite(values) & (values <= float(threshold))))})


# --- Monocle3's fit_models ------------------------------------------------------------------------------------------

def _monocle3_models(extraction_dir: Path, work: Path, jobs_for_r: List[Dict[str, Any]], inputs, *, jobs: int,
                     rscript: str, echo: Callable[[str], None]) -> None:
    """Every lineage's models fitted by monocle3_gene_models.R; each job gets ``results`` (term -> statistic, p,
    df over its tested genes) and ``seconds``."""
    from paralome.stats import term_reference_columns

    work.mkdir(parents=True, exist_ok=True)
    listing = []
    for index, job in enumerate(jobs_for_r, start=1):
        folder = work / f"lineage_{index}"
        folder.mkdir(parents=True, exist_ok=True)
        x = job["x"]
        width = x.shape[1]
        design = pd.DataFrame(x[:, 1:], columns=[f"c{j}" for j in range(1, width)])
        design.insert(0, "column", job["positions"] + 1)
        design.insert(1, "size_factor", job["library"] / np.exp(np.mean(np.log(np.maximum(job["library"], 1e-300)))))
        design.to_csv(folder / "design.tsv", sep="\t", index=False)
        np.savetxt(folder / "genes.txt", job["tested_rows"] + 1, fmt="%d")
        models: Dict[Tuple[int, ...], str] = {}
        tests = []
        for term, columns in job["terms"].items():
            reference = tuple(i for i in term_reference_columns(term, job["terms"], width) if i != 0)
            reduced = tuple(i for i in reference if i not in set(columns))
            if not columns:
                continue
            for key in (reference, reduced):
                models.setdefault(key, f"m{len(models) + 1}")
            tests.append((term, models[reference], models[reduced]))
        with atomic_text_writer(folder / "models.tsv", newline="") as handle:
            handle.write("model\tcolumns\n")
            for key, name in models.items():
                handle.write(f"{name}\t{','.join(f'c{i}' for i in key)}\n")
        with atomic_text_writer(folder / "tests.tsv", newline="") as handle:
            handle.write("term\tfull\treduced\n")
            for term, full, reduced in tests:
                handle.write(f"{term}\t{full}\t{reduced}\n")
        listing.append(folder)
        job["folder"] = folder
    log = work / "monocle3_gene_models.log"
    argv = [rscript, str(GENE_MODELS_R_SCRIPT), "--counts", str(Path(extraction_dir) / "rna_counts.mtx"),
            "--cores", str(max(1, int(jobs or 1))), *[token for folder in listing for token in ("--lineage",
                                                                                               str(folder))]]
    echo(f"trajectories: Monocle3 fit_models on {len(listing)} lineage(s), {max(1, int(jobs or 1))} core(s) "
         f"(log: {log})")
    with open(log, "w") as handle:
        process = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, text=True)
    said = Path(log).read_text(errors="replace").splitlines()
    for line in said:
        if line.startswith("PARALOME_LINEAGE_SECONDS "):
            _tag, folder, seconds = line.split(" ", 2)
            for job in jobs_for_r:
                if str(job["folder"]) == folder:
                    job["seconds"] = round(float(seconds), 1)
    if process.returncode != 0:
        raise ParalomeError(                                   # PLM-0315
            315, f"Monocle3's fit_models failed (exit {process.returncode}); the last lines of {log}:\n"
                 + "\n".join(said[-15:]),
        )
    for job in jobs_for_r:
        frame = pd.read_csv(job["folder"] / "results.tsv", sep="\t")
        n = job["tested_rows"].size
        where = {row: i for i, row in enumerate(job["tested_rows"] + 1)}
        results: Dict[str, Dict[str, np.ndarray]] = {}
        for term, part in frame.groupby("term", sort=False):
            statistic = np.full(n, np.nan)
            p = np.full(n, np.nan)
            dfs = np.full(n, np.nan)
            index = np.asarray([where[int(r)] for r in part["gene_row"]])
            statistic[index] = part["LLR"].to_numpy(dtype=float)
            p[index] = part["p_value"].to_numpy(dtype=float)
            dfs[index] = part["df"].to_numpy(dtype=float)
            results[str(term)] = {"statistic": statistic, "p": p, "df": dfs}
        job["results"] = results
