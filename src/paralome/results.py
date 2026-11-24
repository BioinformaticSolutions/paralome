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

"""Every differential result, written the same way: one folder per statistic.

When multiple statistics are used, each carries its own mathematical reality,
and each gets its own forked set of files: a folder named by the flag that
asked for it -- ``t``, ``paired_t``, ``lm``, ``anova``, and ``t_cell`` ...
with ``:cell`` -- and every pipeline writes into it through this module, with
the same names and the same columns:

    <folder>/<prefix>_<folder>_evidome_input_<stamp>.csv  this statistic's Evidome table
    <folder>/machine/<prefix>_<comparison>_statistics_<stamp>.csv
                                                          full precision, every column, then the
                                                          comparison's sample values (s:<sample>)
    <folder>/machine/<prefix>_<folder>_sample_matrix_<stamp>.csv  every feature x sample: the PCA input

No human-readable report, volcano or power file: OmicView draws them all from the statistics
output zip, with the user's own thresholds.

Generic column names throughout -- ``p_value``,
``q_value``, ``log2_fold_change`` -- because the folder and the file already say
which statistic and which comparison a column belongs to.

No call of significance anywhere: p and q are the result.

No Excel workbook: OmicView reads the reports, the machine tables and the
manifest (the statistics output).

THE EVIDOME TABLE holds everything the statistic produced -- for an ANOVA the
three F-tests and the six contrasts, which is what lets Evidome give each F-test
its direction from its own contrasts. Peak results become genes by one rule: the gene whose
TSS the peak overlaps, otherwise its nearest gene, and the lowest-p peak stands
for the gene in each comparison. An lm's slope is written as its log2 fold
change.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from paralome.comparisons import KIND_FTEST, KIND_SLOPE
from paralome.engine import ComparisonResult
from paralome.report_schema import MISSING_VALUE


@dataclass
class Features:
    """Who the rows are. ``key`` is unique; ``gene_id``/``gene_name`` are what a reader and Evidome see."""

    key: List[str]
    key_column: str
    gene_id: List[str]
    gene_name: List[str]
    annotation: Optional[List[Mapping[str, object]]] = None
    #: Peaks carry their peak_id into the report, beside the gene they map to.
    peaks: bool = False

    def __len__(self) -> int:
        return len(self.key)


@dataclass
class Samples:
    """The per-sample values written beside each result, and what they are."""

    ids: List[str]
    values: np.ndarray
    #: e.g. "raw FPKM", "CPM", "log2 CPM" -- written into the file headers.
    unit: str


def peak_gene(annotation: Mapping[str, object]) -> tuple:
    """(gene_id, gene_name) a peak stands for: the closest TSS or the closest gene.

    The gene whose TSS the peak overlaps, when it overlaps one; otherwise its
    nearest gene. Missing values come back as "".
    """
    def text(name: str) -> str:
        value = annotation.get(name)
        value = "" if value is None else str(value).strip()
        return "" if value in (MISSING_VALUE, "nan", "NA", "None") else value

    overlap = text("tss_overlap").lower() in ("true", "1", "yes")
    if overlap and text("tss_gene_id"):
        return text("tss_gene_id"), text("tss_gene_name")
    return text("nearest_gene_id"), text("nearest_gene_name")


def _number(value) -> object:
    """A cell as written to a CSV: '' for missing, the number otherwise."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return ""
    if float(number).is_integer() and abs(number) < 1e15:
        return int(number)
    return number


def _number_column(values) -> List[object]:
    """``[_number(v) for v in values]``, a column at a time.

    Identical cells: '' for a missing or non-finite number, an int for a whole number under 1e15, else the
    float; bools stay bools; anything that is not a number column goes through _number cell by cell. Writing
    the results cell by cell was most of a trajectory segment's time (15.7M _number calls for 3,000 genes)."""
    if values is None:
        return []
    array = np.asarray(values)
    if array.dtype == bool:
        return array.tolist()
    if array.dtype.kind not in "iuf" or array.ndim != 1:
        return [_number(v) for v in (array.tolist() if array.ndim else [array.item()])]
    numbers = array.astype(float)
    finite = np.isfinite(numbers)
    out = numbers.tolist()
    for i in np.flatnonzero(~finite).tolist():
        out[i] = ""
    whole = np.flatnonzero(finite & (np.abs(numbers) < 1e15) & (numbers == np.floor(numbers)))
    for i, v in zip(whole.tolist(), numbers[whole].astype(np.int64).tolist()):
        out[i] = v
    return out


def _shown_column(values) -> List[object]:
    """``[_shown(v) for v in values]``, a column at a time."""
    return [MISSING_VALUE if v == "" else v for v in _number_column(values)]


def _shown(value) -> object:
    """A cell as a reader sees it: '.' for missing (MISSING_VALUE)."""
    out = _number(value)
    return MISSING_VALUE if out == "" else out


def _stamped(directory: Path, prefix: str, stem: str, stamp: Optional[str], ext: str) -> Path:
    name = f"{prefix}_{stem}" if prefix else stem
    if stamp:
        name = f"{name}_{stamp}"
    return directory / f"{name}{ext}"


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Sequence[object]],
               header_lines: Sequence[str] = ()) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        writer = csv.writer(handle)
        writer.writerow(list(fields))
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)
    return str(path)


def _comparison_samples(result: ComparisonResult, n_samples: int) -> List[int]:
    """The samples a comparison's statistic read, control group first."""
    comparison = result.comparison
    if result.kind == KIND_FTEST:
        return list(range(n_samples))
    if result.kind == KIND_SLOPE:
        return list(comparison.regressed_index)
    if comparison.pairs:
        return [c for c, _t, _p in comparison.pairs] + [t for _c, t, _p in comparison.pairs]
    return list(comparison.control_index) + list(comparison.treatment_index)


def _means(result: ComparisonResult, prefer_raw: bool) -> tuple:
    columns = result.columns
    if prefer_raw and "raw_control_mean" in columns and np.isfinite(
            np.asarray(columns["raw_control_mean"], dtype=float)).any():
        return columns["raw_control_mean"], columns["raw_treatment_mean"]
    return columns.get("control_mean"), columns.get("treatment_mean")


#: A column a pipeline may add to a result, written into every machine table:
#: bulk RNA's optional raw-FPKM gate (--min-expression and
#: --min-expression-difference) -- an export rule, never a test.
EXPORT_GATE_COLUMN = "passes_expression_gate"


def _evidome_table(
    path: Path, results: Sequence[ComparisonResult], features: Features, *,
    version: str, raw_fpkm: bool, prefer_raw: bool,
) -> Optional[str]:
    """This statistic's table in Evidome's input grammar: one row per gene, one block per comparison."""
    genes: Dict[str, int] = {}
    order: List[str] = []
    for i, gene in enumerate(features.gene_id):
        gene = (gene or "").strip()
        if gene and gene != MISSING_VALUE and gene not in genes:
            genes[gene] = len(order)
            order.append(gene)
    if not order:
        return None
    gene_rows: Dict[str, List[int]] = {gene: [] for gene in order}
    for i, gene in enumerate(features.gene_id):
        gene = (gene or "").strip()
        if gene in gene_rows:
            gene_rows[gene].append(i)
    symbol = {}
    for gene, members in gene_rows.items():
        symbol[gene] = next((features.gene_name[i] for i in members if features.gene_name[i]), gene)

    header = [f"generated by paralome {version}" if version else "generated by paralome",
              "unfiltered: every gene with a result; a peak table is collapsed to one row per gene "
              "(the TSS gene, else the nearest gene; the lowest-p peak per comparison)",
              "comparisons in this file:"]
    fields = ["Gene", "Common_name"]
    blocks = []
    for result in results:
        name = result.name
        p = np.asarray(result.columns["p_value"], dtype=float)
        # The representative row of each gene in THIS comparison: its lowest p.
        chosen = {}
        for gene, members in gene_rows.items():
            best = None
            for i in members:
                if math.isfinite(p[i]) and (best is None or p[i] < p[best]):
                    best = i
            chosen[gene] = best if best is not None else members[0]
        if result.kind == KIND_FTEST:
            header.append(f"  {name}: ANOVA F-test, {result.comparison.description}")
            columns = [("F", "F"), ("ANOVA_p", "p_value"), ("ANOVA_q", "q_value")]
            if "trend_p" in result.columns:
                # Evidome's F-test block is F, ANOVA_p and ANOVA_q
                # and nothing else, so the one-way ANOVA's linear trend -- the
                # direction of its F -- goes to Evidome as a directed block of
                # its own.
                trend = f"{result.comparison.term}_linear_trend"
                blocks.append((chosen, [(f"{name}_{role}", np.asarray(result.columns.get(source), dtype=float))
                                        for role, source in columns]))
                fields.extend(column for column, _a in blocks[-1][1])
                header.append(
                    f"  {trend}: linear trend of log2 expression per unit of "
                    f"{result.comparison.term} within the one-way ANOVA (pooled error from that model)"
                )
                name = trend
                slope = np.asarray(result.columns["trend_slope"], dtype=float)
                slope_se = np.asarray(result.columns["trend_slope_se"], dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    trend_t = np.where(slope_se > 0, slope / slope_se, np.nan)
                columns = [("Log2_fold_change", "trend_slope"), ("Effect", "trend_slope"),
                           ("T-statistic", trend_t), ("P-value", "trend_p"), ("Q-value", "trend_q")]
        elif result.kind == KIND_SLOPE:
            header.append(
                f"  {name}: linear regression slope of log2 expression per unit of "
                f"{result.facts.get('covariate', '')} ({result.comparison.description})"
            )
            columns = [("Log2_fold_change", "log2_fold_change"), ("Effect", "log2_fold_change"),
                       ("T-statistic", "statistic"), ("P-value", "p_value"), ("Q-value", "q_value")]
        else:
            description = result.comparison.description.replace(",", ";").replace("=", " ")
            header.append(
                f"  {name}: {description}, control={result.comparison.control}, "
                f"treatment={result.comparison.treatment}"
            )
            mean_names = (("Raw_FPKM_Control_mean", "Raw_FPKM_Treatment_mean") if raw_fpkm
                          else ("Control_mean", "Treatment_mean"))
            control_mean, treatment_mean = _means(result, prefer_raw or raw_fpkm)
            columns = [(mean_names[0], control_mean), (mean_names[1], treatment_mean),
                       ("Log2_fold_change", "log2_fold_change"), ("Effect", "log2_fold_change"),
                       ("T-statistic", "statistic"), ("P-value", "p_value"), ("Q-value", "q_value")]
        resolved = []
        for role, source in columns:
            array = source if not isinstance(source, str) else result.columns.get(source)
            if array is None:
                continue
            resolved.append((f"{name}_{role}", np.asarray(array, dtype=float)))
        fields.extend(column for column, _a in resolved)
        blocks.append((chosen, resolved))

    shown_blocks = [(chosen, [_shown_column(array) for _column, array in resolved]) for chosen, resolved in blocks]

    def rows():
        for gene in order:
            row = [gene, symbol.get(gene) or gene]
            for chosen, shown in shown_blocks:
                i = chosen[gene]
                row.extend(column[i] for column in shown)
            yield row

    return _write_csv(path, fields, rows(), header_lines=header)


#: The prefix of a sample's value column in a machine table (the statistics output reads it as is).
SAMPLE_COLUMN_PREFIX = "s:"


def write_sample_matrix(path: Path, features: Features, samples: Samples) -> Optional[str]:
    """Every feature (rows, its key and gene) by every sample (columns): the values the statistics read."""
    values = np.asarray(samples.values, dtype=float)
    if values.ndim != 2 or not values.shape[1]:
        return None
    columns = [_number_column(values[:, j]) for j in range(values.shape[1])]

    identity = [features.key_column] + [n for n in ("gene_id", "gene_name") if n != features.key_column]

    def rows():
        for i in range(len(features)):
            known = {"gene_id": features.gene_id[i], "gene_name": features.gene_name[i]}
            yield [features.key[i], *(known[n] for n in identity[1:]), *(c[i] for c in columns)]

    return _write_csv(path, [*identity, *samples.ids], rows())


def remove_stale_results(manifest_path: Path, written: Sequence[str], stage: str) -> None:
    """Remove what the previous run of this prefix wrote in this folder and this run did not."""
    try:
        previous = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return
    keep = {str(Path(p).resolve()) for p in written if p}
    for entry in previous.get("files", []):
        path = Path(str(entry))
        if str(path.resolve()) in keep or not path.is_file():
            continue
        try:
            path.unlink()
            print(f"[paralome] {stage}: removed {path.name} -- an earlier run's result that this "
                  "run did not reproduce.", file=sys.stderr, flush=True)
        except OSError as error:
            print(f"[paralome] {stage}: could not remove {path}, left by an earlier run and not "
                  f"reproduced by this one ({error}). Delete it by hand: it does not reflect this "
                  "result.", file=sys.stderr, flush=True)


def _write_comparison(
    result: ComparisonResult, *, features: Features, samples: Samples, directory: Path, machine: Path,
    named: str, stamp: Optional[str], folder: str, prefer_raw_means: bool,
) -> tuple:
    """One comparison's machine table, report and volcano: ``(paths written, its manifest entry)``."""
    written: List[str] = []
    name = result.name
    # Machine table: every column, full precision, identity first.
    annotation_fields: List[str] = []
    if features.annotation:
        annotation_fields = [
            k for k in features.annotation[0]
            if k not in (features.key_column, "gene_id", "gene_name")
        ]
    order = result.column_order()
    # The feature's key first, then its gene -- once: bulk RNA's key IS the gene id.
    identity = [features.key_column] + [
        name for name in ("gene_id", "gene_name") if name != features.key_column
    ]
    # The samples the comparison read, control first, as s:<sample> after the statistics: what the viewer
    # draws a feature's values from (the report that carried them is not written).
    sample_order = _comparison_samples(result, len(samples.ids))
    machine_fields = [*identity, *annotation_fields, *order,
                      *(SAMPLE_COLUMN_PREFIX + samples.ids[j] for j in sample_order)]

    def machine_rows(result=result, order=order, annotation_fields=annotation_fields, sample_order=sample_order):
        columns = [_number_column(result.columns[c]) for c in order]
        values_matrix = np.asarray(samples.values, dtype=float)
        sample_columns = [_number_column(values_matrix[:, j]) for j in sample_order]
        for i in range(len(features)):
            values = {"gene_id": features.gene_id[i], "gene_name": features.gene_name[i]}
            row = [features.key[i]] + [values[name] for name in identity[1:]]
            if annotation_fields:
                row.extend(features.annotation[i].get(k, "") for k in annotation_fields)
            row.extend(column[i] for column in columns)
            row.extend(column[i] for column in sample_columns)
            yield row

    machine_path = _write_csv(
        _stamped(machine, named, f"{name}_statistics", stamp, ".csv"),
        machine_fields, machine_rows(),
    )
    written.append(machine_path)

    entry = {
        "kind": result.kind,
        "role": result.comparison.role,
        "description": result.comparison.description,
        "control": result.comparison.control,
        "treatment": result.comparison.treatment,
        "statistics": machine_path,
        "tested": int(result.facts.get("tested", 0)),
    }
    return written, entry


_SHARED: Dict[str, object] = {}


def _share_tables(features: Features, samples: Samples) -> None:
    """A writer process's copy of the folder's features and samples, sent once."""
    _SHARED["features"], _SHARED["samples"] = features, samples


def _write_shared_comparison(result: ComparisonResult, args: Mapping[str, object]) -> tuple:
    return _write_comparison(result, features=_SHARED["features"], samples=_SHARED["samples"], **args)


def _write_workers(jobs: int, n_comparisons: int) -> int:
    """Processes for a folder's comparisons: ``jobs``, at most one per comparison, within SLURM's CPUs."""
    from paralome.atac.singlecell_stats import _slurm_clamped_jobs

    return max(1, min(_slurm_clamped_jobs(jobs), n_comparisons)) if int(jobs or 1) > 1 else 1


def write_statistic_folder(
    results: Sequence[ComparisonResult],
    *,
    root,
    file_prefix: str,
    stamp: Optional[str],
    features: Features,
    samples: Samples,
    version: str = "",
    prefer_raw_means: bool = True,
    raw_fpkm: bool = False,
    alpha_metric: str = "p",
    alpha_value: float = 0.05,
    stage: str = "differential",
    jobs: int = 1,
) -> Dict[str, object]:
    """Write one statistic's folder for one prefix (project, stratum). Returns its manifest.

    ``results`` are the finalized results of ONE statistic (one folder).
    ``stamp`` is the modality stamp every result name ends with (``rna``,
    ``atac_peaks``), so RNA and ATAC results collected into one place do not
    collide. ``raw_fpkm`` marks bulk RNA, whose linear means are FPKM and are
    named so in the Evidome table.
    """
    if not results:
        return {}
    folder = results[0].folder
    statistic = results[0].statistic
    directory = Path(root) / folder
    # EVERY FILE NAMES ITS STATISTIC, so the user can tell which test made it once files are out of their
    # folders (a t and a paired-t result of KO_vs_WT are different file names).
    named = f"{file_prefix}_{folder}" if file_prefix else folder
    machine = directory / "machine"
    written: List[str] = []
    comparisons: Dict[str, object] = {}

    # Each comparison's files are its own, so with ``jobs`` they are written by that many processes (a
    # large :cell run can have over a hundred comparisons a stratum).
    args = dict(directory=directory, machine=machine, named=named, stamp=stamp, folder=folder,
                prefer_raw_means=prefer_raw_means)
    workers = _write_workers(jobs, len(results))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, initializer=_share_tables,
                                 initargs=(features, samples)) as pool:
            done = list(pool.map(_write_shared_comparison, results, [args] * len(results)))
    else:
        done = [_write_comparison(result, features=features, samples=samples, **args) for result in results]
    for result, (paths, entry) in zip(results, done):
        written.extend(paths)
        comparisons[result.name] = entry

    evidome = _evidome_table(
        _stamped(directory, file_prefix, f"{folder}_evidome_input", stamp, ".csv"),
        results, features, version=version, raw_fpkm=raw_fpkm, prefer_raw=prefer_raw_means,
    )
    if evidome:
        written.append(evidome)

    # Every feature by every sample of this folder: the matrix the viewer's PCA and outlier views read.
    matrix_path = write_sample_matrix(
        _stamped(machine, named, "sample_matrix", stamp, ".csv"), features, samples)
    if matrix_path:
        written.append(matrix_path)

    manifest_path = _stamped(machine, file_prefix, f"{folder}_manifest", stamp, ".json")
    remove_stale_results(manifest_path, written, f"{stage} {folder}")
    manifest = {
        "folder": folder,
        "statistic": statistic,
        "directory": str(directory),
        "comparisons": comparisons,
        "evidome_input": evidome,
        "matrix": matrix_path,
        "files": written,
    }
    from paralome.pipeline.jsonio import write_json

    write_json(manifest_path, manifest, atomic=True, stamp=True)
    manifest["manifest"] = str(manifest_path)
    return manifest


def write_all_statistics(
    results: Sequence[ComparisonResult], **kwargs,
) -> Dict[str, Dict[str, object]]:
    """``write_statistic_folder`` for every statistic in ``results``: folder -> manifest."""
    from paralome.engine import by_folder

    return {folder: write_statistic_folder(group, **kwargs) for folder, group in by_folder(results).items()}
