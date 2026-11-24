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

"""The statistics_output stage of every arm but DNA (formerly browser_bundle).

Every arm exports its results for OmicView in the stage before Evidome, as DNA
has (DNA's is genotype_output, paralome.browser.stage). One file::

    <omicview>/statistics_output/<prefix>_<arm>_statistics_output.zip      stored, not compressed

written straight there (--zip-dir); the stage folder,
<project>/statistics_output (an older <project>/browser_bundle is renamed in
place), keeps the checkpoint and logs.

The zip holds ``bundle.json`` (what the zip holds, for OmicView) and the
RESULTS (paralome.browser.results): each comparison of each statistic folder
and stratum, its samples' values, the sample PCA and outlier tables, and on
ATAC the motif enrichment. OmicView's Results viewer opens them. Zips are named
for what is inside them, with the modality after the prefix. The genome
tracks, the slow part, are the genome_tracks stage's own
<prefix>_<ATAC|snATAC>_genome_tracks.zip (paralome.atac.genome_tracks); an
older zip that carried them hands its tracks over (adopt_bundled_tracks).

SELF-CHECKING, as every bundle stage: issued again, the stage compares the
results' inputs with its checkpoint and keeps the zip when nothing changed.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from paralome.checkpoints import atomic_write_text, file_info, stage_checkpoint_path

STAGE = "statistics_output"
#: The stage's name and folder before.
FORMER_STAGE = "browser_bundle"
SCHEMA = "paralome.browser_bundle.checkpoint.v2"
#: V2, the statistics only (v1, carried the ATAC tracks).
METHOD = "paralome.browser_bundle.arm.v6"  # v3: the sequence QC table; v4: its duplicates counted from the BAMs; v5: a BWA run's mapping read from its logs; v6: STAR's too, with no mapping table
BUNDLE_KIND = "paralome.browser_bundle"
OPEN_WITH = "OmicView"


def bundle_path(workdir: Path, prefix: str, arm: str, zip_dir: Optional[Path] = None) -> Path:
    """``<prefix>_<arm>_statistics_output.zip`` -- RNA, scRNA, ATAC, snATAC or proteomics --
    in ``zip_dir`` (the project's omicview folder), else the stage folder's results/."""
    return Path(zip_dir if zip_dir else Path(workdir) / "results") / f"{prefix}_{arm}_statistics_output.zip"


def take_former_folder(former: Optional[Path], workdir: Path, stage: str, echo: Callable[[str], None] = print) -> None:
    """An older stage folder, <project>/browser_bundle, renamed in place to this stage's name, its
    checkpoint with it, when the new folder is not there yet."""
    if not former or not Path(former).is_dir() or Path(workdir).exists():
        return
    Path(workdir).parent.mkdir(parents=True, exist_ok=True)
    os.rename(former, workdir)
    old = stage_checkpoint_path(Path(workdir), FORMER_STAGE)
    if old.is_file():
        os.replace(old, stage_checkpoint_path(Path(workdir), stage))
    echo(f"{stage}: {former} renamed {workdir}")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _compose(out: Path, staging: Path, bundle: Dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("bundle.json", json.dumps(bundle, indent=2, sort_keys=True, default=str) + "\n")
        for part in ("results", "panel"):            # The --statistic panel tables, as written
            folder = staging / part
            if folder.is_dir():
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        archive.write(path, str(path.relative_to(staging)))
    os.replace(partial, out)


def run_statistics_output(
    *,
    arm: str,
    workdir: Path,
    prefix: str,
    results_roots: Sequence[Path] = (),
    motif_roots: Sequence[Path] = (),
    samples_table: Optional[Path] = None,
    threads: int = 1,
    overwrite: bool = False,
    zip_dir: Optional[Path] = None,
    echo: Callable[[str], None] = print,
    qc_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build (or keep) ``<zip_dir>/<prefix>_<arm>_statistics_output.zip``.

    ``qc_root``: the project folder whose FASTQ-to-alignment summaries go in too (results/qc, kinds fastq_qc,
    alignment, deduplication, peak_sample_qc)."""
    from paralome.browser.results import build_results, collect_inputs, panel_tables, request_id
    from paralome.version import __version__

    workdir = Path(workdir)
    out = bundle_path(workdir, prefix, arm, zip_dir)
    checkpoint_path = stage_checkpoint_path(workdir, STAGE)

    request = {"method": METHOD, "arm": arm, "results": collect_inputs(results_roots, motif_roots, samples_table, qc_root)}
    rid = request_id(request)
    try:
        previous = json.loads(checkpoint_path.read_text())
    except (OSError, ValueError):
        previous = {}
    if (not overwrite and previous.get("status") == "complete" and previous.get("request_id") == rid
            and out.is_file() and out.stat().st_size == previous.get("bundle_bytes")):
        echo(f"statistics_output: {out.name} answers this request; kept")
        return previous

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    atomic_write_text(checkpoint_path, json.dumps({
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "incomplete", "complete": False,
        "request": request, "request_id": rid, "started": started,
        "reason": f"building since {started}; a run that stopped here did not finish",
    }, indent=2, sort_keys=True, default=str) + "\n")

    parts = []
    with tempfile.TemporaryDirectory(prefix="statistics_output_", dir=str(workdir)) as tmp:
        staging = Path(tmp)
        results = build_results(staging, arm=arm, project=prefix, results_roots=results_roots,
                                motif_roots=motif_roots, samples_table=samples_table, threads=threads, echo=echo,
                                qc_root=qc_root)
        if results.get("groups"):
            parts.append("results")
        panels = []
        for root, table in panel_tables(results_roots):
            dest = staging / "panel" / root.name / table.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(table, dest)
            panels.append(str(dest.relative_to(staging)))
        if panels:
            parts.append("panel")
        bundle = {
            "kind": BUNDLE_KIND, "schema": 2, "program": "Paralome", "program_version": __version__,
            "arm": arm, "project": prefix, "parts": parts, "open_with": OPEN_WITH,
            "results": "results/manifest.json" if "results" in parts else None, "tracks": None,
            "panel": panels,
        }
        _compose(out, staging, bundle)

    checkpoint = {
        "schema": SCHEMA, "stage": STAGE, "method": METHOD, "status": "complete", "complete": True,
        "request": request, "request_id": rid, "started": started, "finished": now_iso(),
        "program_version": __version__, "open_with": OPEN_WITH, "arm": arm, "parts": parts,
        "outputs": {"bundle": str(out)}, "bundle_bytes": out.stat().st_size,
        "comparisons": sum(len(g["comparisons"]) for g in results.get("groups") or []),
        "left_out": [] if parts else ["nothing to export: no statistic folder"],
    }
    atomic_write_text(checkpoint_path, json.dumps(checkpoint, indent=2, sort_keys=True, default=str) + "\n")
    echo(f"statistics_output: {out} ({out.stat().st_size / 1e6:.1f} MB; {', '.join(parts) or 'empty'}); "
         f"open it in {OPEN_WITH}")
    return checkpoint


def summarize(workdir: Optional[Path], zip_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Stage state for project_state.json. Complete only with the zip in ``zip_dir`` when given."""
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
        bundle = wd / "results" / bundle.name
    if bundle is not None and zip_dir and (Path(zip_dir) / bundle.name).is_file():
        bundle = Path(zip_dir) / bundle.name           # a copied project (its omicview folder)
    info = file_info(str(bundle) if bundle else None)
    status = str(ckpt.get("status") or "unknown")
    reason = ckpt.get("reason")
    if status == "complete" and not (info.get("exists") and info.get("size") == ckpt.get("bundle_bytes")):
        status, reason = "incomplete", "the bundle zip is missing or not the one recorded"
    elif status == "complete" and zip_dir and bundle is not None and bundle.parent.resolve() != Path(zip_dir).resolve():
        status, reason = "incomplete", f"{bundle.name} is not in {zip_dir} (the zips are in the omicview folder)"
    elif status == "complete" and ckpt.get("method") != METHOD:
        status, reason = "incomplete", (f"built by {ckpt.get('method')} as {bundle.name if bundle else 'a bundle'}; "
                                        "this version writes <prefix>_<arm>_statistics_output.zip, the statistics only")
    return {"status": status, "complete": status == "complete", "workdir": str(wd), "reason": reason,
            "checkpoint": file_info(path), "bundle": info, "parts": ckpt.get("parts") or [],
            "left_out": ckpt.get("left_out") or []}
