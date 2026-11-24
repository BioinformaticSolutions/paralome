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

"""The results part of a browser bundle: an arm's differential results for OmicView.


Every arm that tests features (bulk RNA, bulk ATAC, snATAC, scRNA, proteomics)
writes its statistic folders through one writer (paralome.results), and every
such folder carries a manifest (``machine/<prefix>_<folder>_manifest_<stamp>.json``)
naming each comparison's machine table and report. This module reads those
manifests -- whatever the arm -- and writes, under ``results/`` of the bundle::

    results/manifest.json          the arm, the comparisons (grouped by statistic folder and stratum), the
                                   samples, the QC tables, and which file holds what
    results/c/<NNNN>.parquet       one per comparison: every column of its machine table (full precision),
                                   then the value of each sample the comparison read, as the report gives it,
                                   in columns named "s:<sample>"
    results/m/<NNNN>.parquet       ATAC: the comparison's motif enrichment (its motif_statistics table)
    results/p/<NNNN>.parquet       ATAC: its motif-pair enrichment
    results/qc/<NNNN>.parquet      the sample PCA scores, PCA variance and outlier tables beside the folders
    results/samples.parquet        every sample and its sheet columns (sample_id first), when given

The machine table and the report are written by one writer in one feature
order, so a report's sample values are joined to the machine table by row.
Parquet is written by DuckDB (a dependency), from the CSVs as
they are; '.' and '' are read as missing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RESULTS_KIND = "paralome.results"
RESULTS_SCHEMA = 1
from paralome.results import SAMPLE_COLUMN_PREFIX  # noqa: E402  the machine tables' own
_MANIFEST_RE = re.compile(r"^(?P<prefix>.*)_(?P<folder>[^_].*?)_manifest_(?P<stamp>[a-z_]+)\.json$")
_SAMPLE_ID_COLUMNS = ("sample_id", "sample", "id", "name")
_QC_PATTERNS = (("pca_scores", "*pca_scores*"), ("pca_variance", "*pca_variance*"),
                ("sample_outliers", "*sample_outlier_metrics*"), ("sample_matrix", "*_sample_matrix_*"))


def _fingerprint(path: Path) -> Dict[str, object]:
    stat = Path(path).stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _local(recorded: Optional[str], directory: Path) -> Optional[Path]:
    """A file a manifest records: where it says, else by its name in ``directory`` (a moved or copied project)."""
    if not recorded:
        return None
    path = Path(recorded)
    if path.is_file():
        return path
    candidate = directory / path.name
    return candidate if candidate.is_file() else None


def find_statistic_manifests(roots: Iterable[Path]) -> List[Path]:
    """Every statistic folder's manifest under ``roots`` (machine/*_manifest_*.json with comparisons)."""
    found: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("machine/*_manifest_*.json")):
            if "motif_analysis" in path.relative_to(root).parts:
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and isinstance(data.get("comparisons"), dict) and data.get("folder"):
                found.append(path)
    return found


def _qc_tables(roots: Iterable[Path]) -> List[Tuple[str, Path]]:
    """The sample PCA and outlier tables an arm writes beside its statistic folders."""
    out: List[Tuple[str, Path]] = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for kind, pattern in _QC_PATTERNS:
            for path in sorted(root.rglob(f"machine/{pattern}")):
                if path.suffix.lower() not in (".csv", ".tsv") or path in seen:
                    continue
                seen.add(path)
                out.append((kind, path))
    return out


#: The per-sample summaries from the FASTQ to the alignment (and ATAC's peak counts), by the project's stage
#: folders: the viewer's QC-to-alignment view. (kind, candidate paths under the project root; the first found.)
_SEQUENCING_QC = (
    ("fastq_qc", ("fastq_qc/results/fastq_qc_summary.tsv",)),
    ("alignment", ("alignment/results/alignment_metrics.tsv", "star/results/alignment_metrics.tsv",
                   "star/results/star_alignment_metrics.tsv", "star_align/results/alignment_metrics.tsv",
                   "star_align/results/star_alignment_metrics.tsv")),
    ("deduplication", ("dedup/results/dedup_summary.json",)),
    ("peak_sample_qc", ("peak_quantification/results/sample_qc.csv",)),
)


def sequencing_qc_tables(qc_root: Optional[Path]) -> List[Tuple[str, Path]]:
    """(kind, file) of each QC-to-alignment summary under the project root that exists."""
    if not qc_root:
        return []
    found = []
    for kind, candidates in _SEQUENCING_QC:
        for relative in candidates:
            path = Path(qc_root) / relative
            if path.is_file():
                found.append((kind, path))
                break
    return found


def _sequencing_qc_source(kind: str, path: Path, scratch: Path) -> Tuple[Path, str]:
    """A table the Parquet writer reads: the dedup summary's per-sample metrics come out of its JSON."""
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text()).get("sample_metrics") or []
        columns = list(dict.fromkeys(key for row in rows if isinstance(row, Mapping) for key in row))
        target = scratch / f"{kind}.csv"
        _write_rows_csv(target, columns, [{k: row.get(k, "") for k in columns} for row in rows
                                          if isinstance(row, Mapping)])
        return target, ","
    return path, "\t" if path.suffix.lower() == ".tsv" else ","


def sequence_qc_source(qc_root: Optional[Path], scratch: Path) -> Optional[Path]:
    """The project's sequence QC table (pipeline.sequencing_qc.sequence_qc_rows) as a CSV, None when it has none.

    Raw reads to duplicate-removed reads in one unit (reads) with their percents: OmicView shows it as is.
    """
    if not qc_root:
        return None
    from paralome.pipeline.sequencing_qc import SEQUENCE_QC_COLUMNS, sequence_qc_rows

    rows = sequence_qc_rows(qc_root)
    if not rows:
        return None
    Path(scratch).mkdir(parents=True, exist_ok=True)
    target = Path(scratch) / "sequence_qc.csv"
    _write_rows_csv(target, SEQUENCE_QC_COLUMNS,
                    [{k: ("" if row.get(k) is None else row.get(k)) for k in SEQUENCE_QC_COLUMNS} for row in rows])
    return target


def _motif_tables(roots: Iterable[Path]) -> Dict[str, Path]:
    """Motif enrichment tables by file name (motif_statistics and motif_pair_statistics)."""
    tables: Dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("machine/*_motif_*statistics_*.csv")):
            tables.setdefault(path.name, path)
    return tables


def collect_inputs(
    results_roots: Sequence[Path], motif_roots: Sequence[Path] = (), samples_table: Optional[Path] = None,
    qc_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """What the results part is built from, fingerprinted: the request a rebuild is decided on."""
    manifests = find_statistic_manifests(results_roots)
    files: List[Dict[str, object]] = []
    for manifest in manifests:
        files.append(_fingerprint(manifest))
        data = json.loads(manifest.read_text())
        for entry in (data.get("comparisons") or {}).values():
            for key, directory in (("statistics", manifest.parent), ("report", manifest.parent.parent)):
                path = _local(entry.get(key), directory)
                if path is not None:
                    files.append(_fingerprint(path))
    files.extend(_fingerprint(path) for _kind, path in _qc_tables(results_roots))
    files.extend(_fingerprint(path) for _root, path in panel_tables(results_roots))
    files.extend(_fingerprint(path) for path in _motif_tables(motif_roots).values())
    files.extend(_fingerprint(path) for _kind, path in sequencing_qc_tables(qc_root))
    if samples_table and Path(samples_table).is_file():
        files.append(_fingerprint(Path(samples_table)))
    return {"files": files, "statistic_folders": len(manifests)}


def panel_tables(results_roots: Sequence[Path]) -> List[Tuple[Path, Path]]:
    """(root, table) of each --statistic panel table under the roots (<stratum>/panel/*_panel_*.tsv)."""
    found = []
    for root in results_roots:
        root = Path(root)
        if root.is_dir():
            found += [(root, p) for p in sorted(root.rglob("*_panel_*.tsv")) if p.parent.name == "panel"]
    return found


def _read_header(path: Path) -> List[str]:
    with Path(path).open(newline="") as handle:
        return next(csv.reader(handle), [])


def _merged_csv(statistics: Path, report: Optional[Path], samples: Sequence[str], out: Path) -> List[str]:
    """The machine table with each of the comparison's sample values appended by row; returns the sample names."""
    with Path(statistics).open(newline="") as m_handle:
        machine = csv.reader(m_handle)
        header = next(machine, [])
        if report is None:
            with out.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(machine)
            return []
        with Path(report).open(newline="") as r_handle:
            reader = csv.reader(r_handle)
            report_header = next(reader, [])
            wanted = set(samples)
            index = [i for i, name in enumerate(report_header) if name in wanted]
            names = [report_header[i] for i in index]
            with out.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header + [SAMPLE_COLUMN_PREFIX + name for name in names])
                for row, values in zip(machine, reader):
                    writer.writerow(row + [values[i] if i < len(values) else "" for i in index])
    return names


def _report_samples(report: Optional[Path], layout_known: Iterable[str], samples: Optional[Sequence[str]]) -> List[str]:
    """The sample columns of a report: the sheet's samples when known, else the columns after its statistics."""
    if report is None:
        return []
    header = _read_header(report)
    if samples:
        wanted = set(samples)
        return [name for name in header if name in wanted]
    known = set(layout_known)
    tail: List[str] = []
    for name in reversed(header):
        if name in known:
            break
        tail.append(name)
    return list(reversed(tail))


#: The columns a report puts before its samples (paralome.results._report_layout and the identity columns).
_REPORT_COLUMNS = {
    "Gene", "Common_name", "peak_id", "F", "Fold_change", "P-value", "Q-value", "df_term", "df_error",
    "trend_slope", "trend_slope_se", "trend_F", "trend_p", "trend_q", "lack_of_fit_F", "lack_of_fit_df",
    "lack_of_fit_p", "lack_of_fit_q", "Control_mean", "Treatment_mean", "statistic", "df", "cohens_d",
    "observed_power", "n_pairs", "mean_within_pair_difference", "sd_within_pair_difference", "slope",
    "slope_se", "intercept", "pearson_r", "r_squared", "n_regressed", "standard_error", "n_control_cells",
    "n_treatment_cells", "fit_method", "converged",
}


def read_samples_table(path: Optional[Path]) -> Tuple[List[str], List[Dict[str, str]]]:
    """A sample sheet or manifest: its columns (the sample id renamed sample_id, first) and active rows."""
    if not path or not Path(path).is_file():
        return [], []
    from paralome.csvio import skip_comment_lines

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(skip_comment_lines(handle), delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    lower = {name.lower(): name for name in fields}
    key = next((lower[c] for c in _SAMPLE_ID_COLUMNS if c in lower), None)
    if key is None:
        return [], []
    columns = ["sample_id"] + [name for name in fields if name != key]
    out = []
    for row in rows:
        sid = str(row.get(key) or "").strip()
        if not sid:
            continue
        record = {"sample_id": sid}
        record.update({name: str(row.get(name) or "") for name in fields if name != key})
        out.append(record)
    return columns, out


def _write_rows_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class _Parquet:
    """CSV -> Parquet through one DuckDB connection."""

    def __init__(self, threads: int = 1, memory_limit: Optional[str] = None):
        import duckdb

        self.con = duckdb.connect()
        self.con.execute(f"SET threads={max(1, int(threads))}")
        if memory_limit:
            self.con.execute(f"SET memory_limit='{memory_limit}'")

    def write(self, source: Path, target: Path, *, delimiter: str = ",", all_varchar: bool = False) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        options = (f"header=true, delim='{delimiter}', nullstr=['.', ''], sample_size=-1, "
                   f"all_varchar={'true' if all_varchar else 'false'}, quote='\"'")
        # Named, not positional. DuckDB 1.5 binds the two ? of this
        # COPY the other way round from 2.0 (read_csv was handed the target), and
        # 2.0 does not run on CPUs with AVX but no AVX2; a bundle built with positional
        # parameters stopped with "No files found ... 0001.parquet".
        self.con.execute(
            f"COPY (SELECT * FROM read_csv($source, {options})) TO $target (FORMAT parquet, COMPRESSION zstd)",
            {"source": str(source), "target": str(target)},
        )
        return int(self.con.execute("SELECT count(*) FROM read_parquet(?)", [str(target)]).fetchone()[0])

    def close(self) -> None:
        self.con.close()


def _export_comparison(job: Mapping[str, Any]) -> Dict[str, Any]:
    """One comparison: its machine table joined with its samples' values, as Parquet (and its motif tables)."""
    staging = Path(job["staging"])
    statistics = Path(job["statistics"])
    report = Path(job["report"]) if job.get("report") else None
    merged = Path(job["scratch"]) / f"{job['counter']:04d}.csv"
    wanted = _report_samples(report, _REPORT_COLUMNS, job.get("sheet_samples"))
    names = _merged_csv(statistics, report, wanted, merged)
    if report is None:
        # The machine table carries the comparison's sample values itself (s:<sample>).
        names = [h[len(SAMPLE_COLUMN_PREFIX):] for h in _read_header(statistics) if h.startswith(SAMPLE_COLUMN_PREFIX)]
    parquet = _Parquet(1, job.get("memory_limit"))
    try:
        rows = parquet.write(merged, staging / f"results/c/{job['counter']:04d}.parquet")
        motifs = {}
        for part, source in (job.get("motifs") or {}).items():
            target = f"results/{part}/{job['counter']:04d}.parquet"
            parquet.write(Path(source), staging / target)
            motifs[part] = target
    finally:
        parquet.close()
    merged.unlink()
    header = _read_header(statistics)
    return {"rows": rows, "key": header[0] if header else None, "samples": names, "motifs": motifs}


def _export_all(jobs: Sequence[Mapping[str, Any]], threads: int) -> List[Dict[str, Any]]:
    """Every comparison, in order -- on ``threads`` forked processes when there are several."""
    threads = max(1, int(threads or 1))
    if threads == 1 or len(jobs) <= 1:
        return [_export_comparison(job) for job in jobs]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    workers = min(threads, len(jobs))
    # Each process's DuckDB gets its share of the stage's memory (ONE_JOB_STAGE_MEM, 64G; 48 GB of it shared out),
    # not DuckDB's default of 80 % of the machine apiece.
    limit = f"{max(1, 48 // workers)}GB"
    jobs = [dict(job, memory_limit=limit) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(_export_comparison, jobs, chunksize=1))


def _label(manifest: Path, data: Mapping[str, Any], root_of: Callable[[Path], Optional[Path]]) -> Dict[str, str]:
    """Where a statistic folder sits: its folder, its file prefix (project and stratum) and a readable label."""
    folder = str(data.get("folder"))
    match = _MANIFEST_RE.match(manifest.name)
    prefix = ""
    stamp = ""
    if match and match.group("folder") == folder:
        prefix, stamp = match.group("prefix"), match.group("stamp")
    else:
        stem = manifest.name[: -len(".json")]
        marker = f"_{folder}_manifest_"
        if marker in stem:
            prefix, stamp = stem.split(marker, 1)
    folder_dir = manifest.parent.parent
    root = root_of(manifest)
    try:
        where = str(folder_dir.relative_to(root)) if root else folder
    except ValueError:
        where = folder
    return {"folder": folder, "prefix": prefix, "stamp": stamp, "where": where}


def build_results(
    staging: Path,
    *,
    arm: str,
    project: str,
    results_roots: Sequence[Path],
    motif_roots: Sequence[Path] = (),
    samples_table: Optional[Path] = None,
    threads: int = 1,
    echo: Callable[[str], None] = print,
    qc_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write ``staging/results/...``; returns the results manifest. Nothing to read -> a manifest of nothing."""
    from paralome.version import __version__

    staging = Path(staging)
    base = staging / "results"
    base.mkdir(parents=True, exist_ok=True)
    roots = [Path(r) for r in results_roots if r]

    def root_of(path: Path) -> Optional[Path]:
        for root in roots:
            try:
                path.relative_to(root)
                return root
            except ValueError:
                continue
        return None

    columns, sample_rows = read_samples_table(samples_table)
    sheet_samples = [row["sample_id"] for row in sample_rows] or None
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    motifs = _motif_tables(motif_roots)
    counter = 0
    feature_kind = None
    parquet = None
    try:
        with tempfile.TemporaryDirectory(prefix="results_part_", dir=str(staging)) as tmp:
            scratch = Path(tmp)
            # What each comparison needs is gathered first (cheap); the comparisons are then written on
            # every CPU at once (_export_comparison, one forked process each).
            work: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for manifest in find_statistic_manifests(roots):
                data = json.loads(manifest.read_text())
                where = _label(manifest, data, root_of)
                key = (where["where"], where["prefix"])
                group = groups.setdefault(key, {
                    "label": where["where"] if not where["prefix"] or where["prefix"] == project
                    else f"{where['where']} · {where['prefix'][len(project) + 1:] if where['prefix'].startswith(project + '_') else where['prefix']}",
                    "folder": where["folder"], "statistic": data.get("statistic"), "where": where["where"],
                    "prefix": where["prefix"],
                    "stratum": (where["prefix"][len(project) + 1:] if where["prefix"].startswith(project + "_") else None),
                    "stamp": where["stamp"], "comparisons": [],
                })
                for name, entry in (data.get("comparisons") or {}).items():
                    statistics = _local(entry.get("statistics"), manifest.parent)
                    if statistics is None:
                        echo(f"statistics_output: {name} ({where['where']}): its machine table is not there; left out")
                        continue
                    report = _local(entry.get("report"), manifest.parent.parent)
                    counter += 1
                    motif_sources = {}
                    for part, word in (("m", "motif_statistics"), ("p", "motif_pair_statistics")):
                        for stamp in ("atac_motifs", "atac_motif_pairs"):
                            source = motifs.get(f"{where['prefix']}_{name}_{word}_{stamp}.csv")
                            if source is not None and source.parent.parent.name == where["folder"]:
                                motif_sources[part] = str(source)
                    record: Dict[str, Any] = {
                        "name": name, "kind": entry.get("kind"), "role": entry.get("role"),
                        "description": entry.get("description"), "control": entry.get("control"),
                        "treatment": entry.get("treatment"), "tested": entry.get("tested"),
                        "table": f"results/c/{counter:04d}.parquet",
                    }
                    group["comparisons"].append(record)
                    work.append((record, {
                        "counter": counter, "statistics": str(statistics), "report": str(report) if report else None,
                        "sheet_samples": sheet_samples, "scratch": str(scratch), "staging": str(staging),
                        "motifs": motif_sources,
                    }))
            for record, done in zip((r for r, _job in work), _export_all([job for _r, job in work], threads)):
                record.update(rows=done["rows"], key=done["key"], samples=done["samples"])
                for part, target in done["motifs"].items():
                    record["motifs" if part == "m" else "motif_pairs"] = target
                if feature_kind is None and done["key"] is not None:
                    feature_kind = "peak" if done["key"] == "peak_id" else done["key"]
            # After the forks: a DuckDB connection is not carried into a fork.
            parquet = _Parquet(threads)
            qc_records = []
            for index, (kind, path) in enumerate(_qc_tables(roots), start=1):
                target = f"results/qc/{index:04d}.parquet"
                parquet.write(path, staging / target, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
                root = root_of(path)
                try:
                    where = str(path.parent.parent.relative_to(root)) if root else ""
                except ValueError:
                    where = ""
                qc_records.append({"kind": kind, "path": target, "source": path.name, "where": where or "."})
            for kind, path in sequencing_qc_tables(qc_root):
                index = len(qc_records) + 1
                target = f"results/qc/{index:04d}.parquet"
                source, delimiter = _sequencing_qc_source(kind, path, scratch)
                parquet.write(source, staging / target, delimiter=delimiter)
                qc_records.append({"kind": kind, "path": target, "source": path.name,
                                   "where": str(path.parent.parent.name)})
            sequence_qc = sequence_qc_source(qc_root, scratch)
            if sequence_qc is not None:
                target = f"results/qc/{len(qc_records) + 1:04d}.parquet"
                parquet.write(sequence_qc, staging / target, delimiter=",", all_varchar=True)
                qc_records.append({"kind": "sequence_qc", "path": target, "source": sequence_qc.name, "where": "."})
            samples_record = None
            if sample_rows:
                source = scratch / "samples.csv"
                _write_rows_csv(source, columns, sample_rows)
                parquet.write(source, staging / "results/samples.parquet", all_varchar=True)
                samples_record = {"path": "results/samples.parquet", "columns": columns, "count": len(sample_rows)}
    finally:
        if parquet is not None:
            parquet.close()

    manifest = {
        "kind": RESULTS_KIND, "schema": RESULTS_SCHEMA, "program": "Paralome", "program_version": __version__,
        "arm": arm, "project": project, "feature": feature_kind,
        "sample_column_prefix": SAMPLE_COLUMN_PREFIX,
        "groups": [group for group in groups.values() if group["comparisons"]],
        "qc": qc_records, "samples": samples_record,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    count = sum(len(g["comparisons"]) for g in manifest["groups"])
    echo(f"statistics_output: {count} comparison(s) in {len(manifest['groups'])} statistic folder(s) exported")
    return manifest


def request_id(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()[:16]
