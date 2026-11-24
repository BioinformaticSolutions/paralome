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

import json
import csv
from pathlib import Path
from typing import Any, Optional, Union


def parse_fastp_json(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    before = summary.get("before_filtering", {})
    after = summary.get("after_filtering", {})
    filtering = data.get("filtering_result", {})
    duplication = data.get("duplication", {})
    insert_size = data.get("insert_size", {})

    raw_reads = int(before.get("total_reads") or 0)
    clean_reads = int(after.get("total_reads") or 0)
    raw_bases = int(before.get("total_bases") or 0)
    clean_bases = int(after.get("total_bases") or 0)

    return {
        "raw_reads": raw_reads,
        "clean_reads": clean_reads,
        "reads_removed": raw_reads - clean_reads,
        "pct_reads_surviving": round((clean_reads / raw_reads * 100.0), 3) if raw_reads else 0.0,
        "raw_bases": raw_bases,
        "clean_bases": clean_bases,
        "pct_bases_surviving": round((clean_bases / raw_bases * 100.0), 3) if raw_bases else 0.0,
        "q20_rate_before": before.get("q20_rate", "."),
        "q30_rate_before": before.get("q30_rate", "."),
        "q20_rate_after": after.get("q20_rate", "."),
        "q30_rate_after": after.get("q30_rate", "."),
        "gc_content_before": before.get("gc_content", "."),
        "gc_content_after": after.get("gc_content", "."),
        "too_short_reads": filtering.get("too_short_reads", "."),
        "too_many_N_reads": filtering.get("too_many_N_reads", "."),
        "low_quality_reads": filtering.get("low_quality_reads", "."),
        "duplication_rate": duplication.get("rate", "."),
        "insert_size_peak": insert_size.get("peak", "."),
    }


def _current_sample_ids(workdir: Path) -> Optional[set[str]]:
    """The samples of the sheet this workdir was last run with, or None.

    Both backends write inputs/trimmed_samples.csv from the sheet before any
    report is summarised. None is a workdir without one, where there is
    nothing to say which reports are current.
    """
    manifest = workdir / "inputs" / "trimmed_samples.csv"
    if not manifest.is_file():
        return None
    with manifest.open(newline="", encoding="utf-8-sig") as fh:
        return {
            (row.get("sample_id") or "").strip()
            for row in csv.DictReader(fh)
            if (row.get("sample_id") or "").strip()
        }


def replace_mark(workdir: Union[str, Path], sample_id: str) -> Path:
    """Where --overwrite records that a sample's outputs are still the old ones.

    Written by `fastq qc --overwrite` for every sample, removed by
    the sample's own task when it has finished. See fastp._mark_for_replacement.
    """
    return Path(workdir) / "inputs" / "to_be_replaced" / sample_id


def summarize_fastp_workdir(workdir: Union[str, Path]) -> dict[str, Any]:
    workdir = Path(workdir)
    json_dir = workdir / "fastp_json"
    result_dir = workdir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # The samples of the current sheet only. Every report in the
    # folder was counted, so a sample since taken off the sheet stayed in the
    # table and the stage totals, which then disagreed with the run.
    current = _current_sample_ids(workdir)
    rows: list[dict[str, Any]] = []
    waiting: list[str] = []
    for p in sorted(json_dir.glob("*.json")):
        sample = p.stem
        if current is not None and sample not in current:
            continue
        if replace_mark(workdir, sample).exists():
            waiting.append(sample)
            continue
        row = {"sample_id": sample, "fastp_json": str(p)}
        row.update(parse_fastp_json(p))
        rows.append(row)

    if waiting:
        # Their reports are the old run's. A summary in place reads
        # as a finished stage, so none is written while any is still waiting.
        raise RuntimeError(
            f"FASTQ QC in {workdir} was started again with --overwrite and these samples have not been "
            "trimmed again yet: " + ", ".join(waiting) + ". No summary was written. "
            "Run `fastq qc` again (without --overwrite) to trim them."
        )
    # A sample of the sheet with no report (its fastp task failed)
    # was left out of the totals without a word. No summary while one is missing.
    missing = sorted((current or set()) - {row["sample_id"] for row in rows} - set(waiting))
    if missing:
        raise FileNotFoundError(
            f"FASTQ QC in {workdir} has no fastp report for these samples of the sheet: "
            + ", ".join(missing) + f" (expected {json_dir}/<sample>.json). No summary was written. "
            "Run `fastq qc` again to trim them, or take them off the sample sheet."
        )
    if not rows:
        raise FileNotFoundError(f"No fastp JSON reports found in {json_dir}")

    tsv = result_dir / "fastq_qc_summary.tsv"
    fields = list(rows[0].keys())
    with tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    totals = {
        "samples": len(rows),
        "raw_reads": sum(int(r.get("raw_reads") or 0) for r in rows),
        "clean_reads": sum(int(r.get("clean_reads") or 0) for r in rows),
        "raw_bases": sum(int(r.get("raw_bases") or 0) for r in rows),
        "clean_bases": sum(int(r.get("clean_bases") or 0) for r in rows),
    }
    totals["pct_reads_surviving"] = round((totals["clean_reads"] / totals["raw_reads"] * 100.0), 3) if totals["raw_reads"] else 0.0
    totals["pct_bases_surviving"] = round((totals["clean_bases"] / totals["raw_bases"] * 100.0), 3) if totals["raw_bases"] else 0.0

    summary = {
        "workdir": str(workdir),
        "summary_tsv": str(tsv),
        "totals": totals,
        "samples": rows,
    }
    out_json = result_dir / "fastq_qc_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
