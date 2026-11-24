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

"""The genotype_output stage (formerly browser_bundle): the DNA run's results for the variant browser.

Every DNA run ends its own work with it,
after gene_burden and before Evidome. It reads what the run already wrote --
the segregation stage's whole-genome table, every gene_burden table, the
filtering stage's pass table -- and writes one file::

    <workdir>/results/<prefix>_DNA_genotype_output.zip

stored (not compressed). Open it in OmicView's Variant Browser, which reads
the zip in place on the user's computer (browser/bundle.py).

SELF-CHECKING. Issued again, it compares its inputs (each file's path, size
and modification time, and the options) with the request in its checkpoint.
Unchanged, with the zip there, it keeps the zip and writes nothing; otherwise
it builds the zip again in place of the old one. gene_burden or filtering that
has not run (a genome without a burden reference) leaves its part out of the
bundle, said in the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from paralome.checkpoints import atomic_write_text, file_info, stage_checkpoint_path
from paralome.errors import ParalomeError

STAGE = "genotype_output"
SCHEMA = "paralome.browser_bundle.checkpoint.v1"
#: v2: each variant row carries the candidate lists' columns (bundle.LIST_COLUMNS), and the QC-to-alignment
#: summaries go in (qc/).
METHOD = "paralome.browser_bundle.v6"  # v3: the sequence QC table; v4: its duplicates counted from the BAMs; v5: a BWA run's mapping read from its logs; v6: STAR's too, with no mapping table


def bundle_path(workdir: Path, prefix: str, zip_dir: Optional[Path] = None) -> Path:
    return Path(zip_dir if zip_dir else Path(workdir) / "results") / f"{prefix}_DNA_genotype_output.zip"


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _inputs(segregation_table: Path, burden_results: Optional[Path], filter_results: Optional[Path],
            qc_root: Optional[Path] = None) -> dict:
    from paralome.browser.results import sequencing_qc_tables

    burden = sorted(Path(burden_results).glob("*_burden_*.tsv")) if burden_results and Path(burden_results).is_dir() else []
    passes = sorted(Path(filter_results).glob("*/*_pass.tsv")) if filter_results and Path(filter_results).is_dir() else []
    summaries = sorted(Path(filter_results).glob("*filtering_summary.json")) if passes else []
    return {
        "segregation_table": _fingerprint(Path(segregation_table)),
        "burden_tables": [_fingerprint(p) for p in burden],
        "filter_pass_tables": [_fingerprint(p) for p in passes],
        "filter_summaries": [_fingerprint(p) for p in summaries],
        "qc_tables": [_fingerprint(p) for _kind, p in sequencing_qc_tables(qc_root)],
    }


def _request_id(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_genotype_output(
    *,
    segregation_table: Path,
    workdir: Path,
    prefix: str,
    burden_results: Optional[Path] = None,
    filter_results: Optional[Path] = None,
    filter_profile: Optional[str] = None,
    threads: int = 1,
    overwrite: bool = False,
    zip_dir: Optional[Path] = None,
    echo: Callable[[str], None] = print,
    qc_root: Optional[Path] = None,
) -> dict[str, Any]:
    from paralome.browser.bundle import OPEN_WITH, build_bundle
    from paralome.version import __version__

    segregation_table = Path(segregation_table)
    if not segregation_table.is_file():
        raise ParalomeError(104, f"the segregation table {segregation_table} is not there",
                            fix="Run the segregation stage first")
    workdir = Path(workdir)
    out = bundle_path(workdir, prefix, zip_dir)
    checkpoint_path = stage_checkpoint_path(workdir, STAGE)
    request = {
        "method": METHOD,
        "inputs": _inputs(segregation_table, burden_results, filter_results, qc_root),
        "filter_profile": filter_profile,
    }
    request_id = _request_id(request)
    try:
        previous = json.loads(checkpoint_path.read_text())
    except (OSError, ValueError):
        previous = {}
    # A bundle this request built under its former name, <prefix>_browser_bundle.zip, is
    # renamed, not built again (the variant bundle takes hours). And moved into the omicview
    # folder, from where it was recorded or, the stage folder renamed, from this folder's results/.
    former = Path(str((previous.get("outputs") or {}).get("bundle") or ""))
    if former.name and not former.is_file() and (workdir / "results" / former.name).is_file():
        former = workdir / "results" / former.name
    try:                                                   # only this stage's own file (not a copy's original)
        former.resolve().relative_to(workdir.resolve())
    except ValueError:
        former = Path("")
    if (not overwrite and previous.get("status") == "complete" and previous.get("request_id") == request_id
            and not out.is_file() and former.name and former != out and former.is_file()
            and former.stat().st_size == previous.get("bundle_bytes")):
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(former, out)
        previous = dict(previous, outputs=dict(previous.get("outputs") or {}, bundle=str(out)))
        atomic_write_text(checkpoint_path, json.dumps(previous, indent=2, sort_keys=True) + "\n")
        echo(f"genotype_output: {former} moved to {out}")
    if (not overwrite and previous.get("status") == "complete" and previous.get("request_id") == request_id
            and out.is_file() and out.stat().st_size == previous.get("bundle_bytes")):
        echo(f"genotype_output: {out.name} answers this request; kept")
        return previous

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    atomic_write_text(checkpoint_path, json.dumps({
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "incomplete", "complete": False,
        "request": request, "request_id": request_id, "started": started,
        "reason": f"building since {started}; a run that stopped here did not finish",
    }, indent=2, sort_keys=True) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    built = build_bundle(
        segregation_table, out,
        burden_results=burden_results if burden_results and Path(burden_results).is_dir() else None,
        filter_results=filter_results if filter_results and Path(filter_results).is_dir() else None,
        filter_profile=filter_profile, threads=threads, echo=echo, qc_root=qc_root,
    )
    checkpoint = {
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "complete", "complete": True,
        "request": request, "request_id": request_id, "started": started, "finished": now_iso(),
        "program_version": __version__, "open_with": OPEN_WITH,
        "outputs": {"bundle": str(out)}, "bundle_bytes": out.stat().st_size,
        "variants": built.get("variants"), "candidate_variants": built.get("candidates"),
        "bins": built.get("bins"), "seconds": built.get("seconds"),
        "burden": bool(built.get("burden")), "genotypes": bool(built.get("genotypes")),
        "pipeline_filter": built.get("filter"),
        "left_out": [part for part, there in (("gene_burden tables", built.get("burden")),
                                              ("pipeline filter", built.get("filter"))) if not there],
    }
    atomic_write_text(checkpoint_path, json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    echo(f"genotype_output: {out} ({out.stat().st_size / 1e9:.2f} GB); open it in {OPEN_WITH}")
    return checkpoint


def summarize(workdir: Optional[Path], zip_dir: Optional[Path] = None) -> dict[str, Any]:
    """Stage state for project_state.json (pipeline/state.py): complete when the checkpoint says so, its method
    is this one and the zip it names is there at its recorded size."""
    if not workdir:
        return {"status": "not_provided", "workdir": None, "complete": False}
    wd = Path(workdir)
    path = stage_checkpoint_path(wd, STAGE)
    try:
        ckpt = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "not_available", "workdir": str(wd), "complete": False, "checkpoint": file_info(path)}
    recorded = (ckpt.get("outputs") or {}).get("bundle")
    bundle = Path(recorded) if recorded else None
    if bundle is not None and not bundle.is_file() and (wd / "results" / bundle.name).is_file():
        bundle = wd / "results" / bundle.name          # a copied project
    if bundle is not None and zip_dir and (Path(zip_dir) / bundle.name).is_file():
        bundle = Path(zip_dir) / bundle.name           # a copied project (its omicview folder)
    info = file_info(str(bundle) if bundle else None)
    status = str(ckpt.get("status") or "unknown")
    reason = ckpt.get("reason")
    if status == "complete":
        if ckpt.get("method") != METHOD:
            status, reason = "incomplete", f"built by {ckpt.get('method')}; the method is now {METHOD}"
        elif not (info.get("exists") and info.get("size") == ckpt.get("bundle_bytes")):
            status, reason = "incomplete", "the bundle zip is missing or not the one recorded"
        elif bundle is not None and not bundle.name.endswith("_DNA_genotype_output.zip"):
            # Issued again, the stage renames it (run_genotype_output).
            status, reason = "incomplete", f"{bundle.name} is named <prefix>_DNA_genotype_output.zip now"
        elif zip_dir and bundle is not None and bundle.parent.resolve() != Path(zip_dir).resolve():
            # Issued again, the stage moves it there.
            status, reason = "incomplete", f"{bundle.name} is not in {zip_dir} (the zips are in the omicview folder)"
    return {"status": status, "complete": status == "complete", "workdir": str(wd), "reason": reason,
            "checkpoint": file_info(path), "bundle": info, "left_out": ckpt.get("left_out") or []}
