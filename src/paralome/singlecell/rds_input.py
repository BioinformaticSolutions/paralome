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

"""``--rds`` takes a single object or a folder of objects.

Gene lists are compared by Ensembl ID, genes with none by their feature name;
metadata columns only some objects carry are dropped and listed; a folder
holding one object is a folder like any other.

The way the path is WRITTEN decides which it is, and the disk must agree:
``*.rds`` (any case) is one object -- as is ``*.qs2``, which ``--rds`` has
read -- a trailing ``/`` is a folder, anything else is
refused. Python's path handling drops a trailing ``/``, so the option
is read as text and decided here, before it ever becomes a ``Path``. Past the
command line the decision is carried by the disk (a folder is a directory),
and ``rds_argument`` puts the ``/`` back on whenever the path is handed to
another Paralome command.

A folder is never merged in R. Each object is extracted on its own by the
existing extraction, one in memory at a time, into ``rds_members/<name>/``
under the extraction directory, each with its own checkpoint -- so a rerun
after fixing one file re-extracts only that file. Every member is then
checked; nothing is stacked until all pass. The stacked extraction is written
where a single object's would be, with a real checkpoint whose signature
covers every member, and every later stage reads it exactly as it reads a
single object's extraction.

No file is the authority. The command line names the metadata columns every
file must carry; differing gene (or peak) lists stop the run with every group
reported and no file chosen as the reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from paralome.errors import ParalomeError

STACK_SCHEMA = "paralome.singlecell.rds_folder_extract.v1"
#: The date the stacking last changed what it writes; a stack recorded under another is made again.
STACK_METHOD = "2026-10-21"
MEMBERS_DIRNAME = "rds_members"
GENE_GROUPS_FILE = "rds_folder_gene_groups.tsv"
PEAK_GROUPS_FILE = "rds_folder_peak_groups.tsv"
#: How many differing names are printed per group; the full lists go to a file.
_SHOWN = 10
_MISSING = (".", "")


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def parse_rds_option(text: Union[str, Path, None], *, flag: str = "--rds") -> Optional[Path]:
    """The ``--rds`` value as typed -> the object or folder it names.

    ``*.rds`` (or ``*.qs2``) in any case is a single object and must be a regular file; a
    trailing ``/`` (``slides/``, even ``my.rds/``) is a folder and must be a
    directory; anything else is refused with the rule spelled out.
    A ``Path`` (an internal caller) has already been decided and is returned.
    """
    if text is None:
        return None
    if isinstance(text, Path):
        return text
    raw = str(text)
    path = Path(raw).expanduser()
    if raw.endswith("/"):
        if not path.exists():
            raise ParalomeError(104, f"{flag} {raw!r} does not exist.")
        if not path.is_dir():
            raise ParalomeError(
                118, f"{flag} {raw!r} ends in '/', which names a folder of objects, "
                     f"but it is not a folder.")
        return path
    if raw.lower().endswith((".rds", ".qs2")):
        if not path.exists():
            from paralome.validate import check_input_path

            check_input_path(path, flag=flag)
        if not path.is_file():
            raise ParalomeError(
                118, f"{flag} {raw!r} ends in {path.suffix}, which names a single object, "
                     f"but it is not a file. A folder of objects is written with "
                     f"a trailing '/': {raw}/")
        return path
    hint = f" -- for a folder of objects write {raw}/" if path.is_dir() else ""
    raise ParalomeError(
        118, f"{flag} {raw!r} is neither a .rds (or .qs2) object nor a folder written with "
             f"a trailing '/'{hint}.")


def rds_argument(path: Union[str, Path]) -> str:
    """``path`` as another Paralome command must be given it: a folder keeps its ``/``."""
    text = str(path)
    if Path(text).expanduser().is_dir() and not text.endswith("/"):
        return text + "/"
    return text


def is_rds_folder(rds: Union[str, Path]) -> bool:
    return Path(str(rds)).expanduser().is_dir()


def rds_members(folder: Union[str, Path]) -> List[Path]:
    """Every ``*.rds`` (any case) directly in ``folder``, by name.

    Subfolders are not read; every other file is left alone, neither read nor
    refused.
    """
    folder = Path(folder).expanduser().resolve()
    members = sorted(
        (entry for entry in folder.iterdir()
         if entry.is_file() and entry.name.lower().endswith(".rds")),
        key=lambda entry: entry.name,
    )
    if not members:
        raise ParalomeError(222, f"{folder} holds no .rds object (subfolders are not read).")
    stems: Dict[str, List[str]] = {}
    for entry in members:
        stems.setdefault(member_prefix(entry), []).append(entry.name)
    clashing = {stem: names for stem, names in stems.items() if len(names) > 1}
    if clashing:
        detail = "; ".join(f"{', '.join(names)} -> {stem}_" for stem, names in clashing.items())
        raise ParalomeError(223, f"In {folder}: {detail}")
    return members


def member_prefix(path: Path) -> str:
    """``HPC_03.rds`` -> ``HPC_03``: what its barcodes are prefixed with (plus ``_``)."""
    return path.name[: -len(".rds")]


# ---------------------------------------------------------------------------
# the extraction
# ---------------------------------------------------------------------------

def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _header(path: Path) -> List[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"), [])


def _lines(path: Path) -> List[str]:
    with path.open() as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _matrix(directory: Path, name: str):
    """A member's counts, read directly (not memoised: each is read once)."""
    from scipy.io import mmread

    from paralome.atac.singlecell import _read_plmsp

    binary = directory / name.replace(".mtx", ".plmsp")
    if binary.is_file():
        return _read_plmsp(binary)
    return mmread(str(directory / name)).tocsc()


def write_plmsp(matrix, path: Path) -> None:
    """The R helper's binary CSC layout, written from Python (read by ``_read_plmsp``)."""
    import numpy as np

    import os

    matrix = matrix.tocsc()
    matrix.sort_indices()
    nrow, ncol = matrix.shape
    header = b"PLMSPR01" + np.array(
        [nrow, ncol, matrix.nnz, ncol + 1], dtype="<f8").tobytes()
    header += b"\0" * (64 - len(header))
    body = (
        np.asarray(matrix.indptr, dtype="<f8").tobytes()
        + np.asarray(matrix.indices, dtype="<i4").tobytes()
        + np.asarray(matrix.data, dtype="<f8").tobytes()
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(header + body)
    os.replace(temporary, path)


def _gene_keys(genes: Sequence[Mapping[str, str]]) -> List[str]:
    """Ensembl ID where the GTF gave one; the feature name where it did not."""
    keys = []
    for row in genes:
        gene_id = str(row.get("gene_id") or "").strip()
        if gene_id in _MISSING:
            keys.append(f"name:{str(row.get('feature_name') or '').strip()}")
        else:
            keys.append(gene_id.split(".")[0] if gene_id.upper().startswith("ENS") else gene_id)
    return keys


def _occurrence_keys(keys: Sequence[str]) -> List[Tuple[str, int]]:
    """Unique row keys: a key that repeats is told apart by its occurrence."""
    seen: Counter = Counter()
    out = []
    for key in keys:
        out.append((key, seen[key]))
        seen[key] += 1
    return out


def _groups(lists: Mapping[str, Sequence[str]]) -> List[Tuple[List[str], Counter]]:
    """Files grouped by identical (unordered) list."""
    grouped: Dict[Tuple, List[str]] = {}
    counters: Dict[Tuple, Counter] = {}
    for name, keys in lists.items():
        counter = Counter(keys)
        signature = tuple(sorted(counter.items()))
        grouped.setdefault(signature, []).append(name)
        counters[signature] = counter
    return [(files, counters[signature]) for signature, files in grouped.items()]


def _report_differing(
    kind: str,
    groups: List[Tuple[List[str], Counter]],
    labels: Mapping[str, str],
    out_dir: Path,
    filename: str,
    code: int,
    evidence: Mapping[str, str],
) -> None:
    """Every group, what it has that the others lack and the reverse; no winner."""
    from paralome.checkpoints import atomic_text_writer

    rows: List[Tuple[int, str, str, str, str]] = []
    text = [f"The objects do not all have the same {kind}s; {len(groups)} groups. "
            f"Paralome does not choose between them: decide which file is wrong."]
    for number, (files, counter) in enumerate(groups, start=1):
        others = Counter()
        for other_files, other in groups:
            if other_files is not files:
                others |= other
        here = set(counter)
        elsewhere = set(others)
        not_everywhere = sorted(
            key for key in here
            if any(key not in other for other_files, other in groups if other_files is not files)
        )
        lacking = sorted(elsewhere - here)
        text.append(f"\nGroup {number}: {len(files)} file(s), {sum(counter.values()):,} {kind}s"
                    + (f"; {evidence[files[0]]}" if evidence.get(files[0]) else ""))
        text.append(f"  files: {', '.join(files)}")
        text.append(f"  has {len(not_everywhere):,} {kind}s another group lacks"
                    + (f": {', '.join(labels.get(k, k) for k in not_everywhere[:_SHOWN])}"
                       + (" ..." if len(not_everywhere) > _SHOWN else "") if not_everywhere else ""))
        text.append(f"  lacks {len(lacking):,} {kind}s another group has"
                    + (f": {', '.join(labels.get(k, k) for k in lacking[:_SHOWN])}"
                       + (" ..." if len(lacking) > _SHOWN else "") if lacking else ""))
        for key in not_everywhere:
            rows.append((number, ";".join(files), key, labels.get(key, key), "has_others_lack"))
        for key in lacking:
            rows.append((number, ";".join(files), key, labels.get(key, key), "lacks_others_have"))
    target = out_dir / filename
    with atomic_text_writer(target, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["group", "files", "key", "name", "status"])
        writer.writerows(rows)
    text.append(f"\nFull lists: {target}")
    raise ParalomeError(code, "\n".join(text))


#: The SLURM jobs a folder's extraction is split into.
MEMBER_JOB = "sc_extract"
MEMBER_PLAN = "rds_members_plan.json"


def pending_members(folder: Union[str, Path], out_dir: Union[str, Path], **extract_kwargs) -> List[Path]:
    """The objects in ``folder`` whose extraction cannot be reused, by the extraction's own reuse rule."""
    from paralome.atac.singlecell import run_seurat_extraction

    members_root = Path(out_dir).expanduser().resolve() / MEMBERS_DIRNAME
    pending = []
    for member in rds_members(folder):
        try:
            run_seurat_extraction(member, members_root / member_prefix(member), require_existing=True,
                                  write_qc=False, **extract_kwargs)
        except Exception:  # noqa: BLE001 - anything short of a reusable extraction is work to do
            pending.append(member)
    return pending


def submit_member_extraction(
    folder: Union[str, Path], out_dir: Union[str, Path], *, workdir: Union[str, Path], jobs: int,
    mem: str, partition: Optional[str], submit: bool, cell_export_dir: Optional[Union[str, Path]] = None,
    **extract_kwargs,
) -> Dict[str, object]:
    """One SLURM array task per object still to extract, ``jobs`` at a time (0: all at once).

    Each task is the extraction a folder run makes of that object, into the
    same ``rds_members/<name>/`` with the same checkpoint, so the job that
    then extracts the folder finds every member done and only checks and
    stacks. That job is submitted by the caller, after this array
    (``afterany``): a task that failed is extracted again there, in order, and
    its error is the run's. Objects already extracted are not submitted.
    """
    from paralome.binaries import worker_command
    from paralome.pipeline.naming import slurm_job_name
    from paralome.pipeline.slurm import SlurmOptions, sbatch, write_sbatch
    import shlex

    folder = Path(folder).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()
    # --overwrite: every object again (the reuse check would find them done).
    overwrite = bool(extract_kwargs.pop("overwrite", False))
    pending = (rds_members(folder) if overwrite
               else pending_members(folder, output_dir, **extract_kwargs))
    extract_kwargs["overwrite"] = overwrite
    result: Dict[str, object] = {"members": len(rds_members(folder)), "to_extract": [p.name for p in pending],
                                 "submitted": False, "job_id": None, "script": None}
    if not pending:
        return result
    workdir = Path(workdir).expanduser().resolve()
    plan_path = workdir / "scripts" / MEMBER_PLAN
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({
        "folder": str(folder), "out_dir": str(output_dir), "members": [str(p) for p in pending],
        "extract": {k: (str(v) if isinstance(v, Path) else v) for k, v in extract_kwargs.items()},
        # The run's omicview folder, when the run names one.
        "cell_export_dir": str(cell_export_dir) if cell_export_dir else None,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    throttle = f"%{int(jobs)}" if jobs and int(jobs) > 0 else ""
    script = write_sbatch(
        workdir / "scripts" / f"{MEMBER_JOB}.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(workdir, MEMBER_JOB), cpus_per_task=1, mem=mem, partition=partition,
            array=f"1-{len(pending)}{throttle}",
            output=workdir / "logs" / f"{MEMBER_JOB}.%A_%a.out", error=workdir / "logs" / f"{MEMBER_JOB}.%A_%a.err",
        ),
        f'\nexec {worker_command()} atac extract-member --plan {shlex.quote(str(plan_path))} '
        f'--index "${{SLURM_ARRAY_TASK_ID:?}}"\n',
    )
    result.update(script=str(script), plan=str(plan_path))
    if submit:
        result.update(submitted=True, job_id=sbatch(script))
    return result


def run_member_task(plan_path: Union[str, Path], index: int) -> Dict[str, object]:
    """One array task: extract the ``index``-th (1-based) object of the plan, as a folder run would."""
    from paralome.atac.singlecell import run_seurat_extraction
    from paralome.singlecell.cell_export import export_home

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    members = plan["members"]
    if not 1 <= int(index) <= len(members):
        raise ValueError(f"--index {index} is outside this plan's 1-{len(members)}")
    member = Path(members[int(index) - 1])
    print(f"[paralome] --rds folder: object {index} of {len(members)} in this array: {member.name}", flush=True)
    return run_seurat_extraction(
        member, Path(plan["out_dir"]) / MEMBERS_DIRNAME / member_prefix(member), write_qc=False,
        cell_export_dir=plan.get("cell_export_dir") or export_home(Path(plan["out_dir"]).parent), **plan["extract"],
    )


def run_folder_extraction(
    folder: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    required_columns: Sequence[str] = (),
    overwrite: bool = False,
    require_existing: bool = False,
    cell_export_dir: Optional[Union[str, Path]] = None,
    **extract_kwargs,
) -> Dict[str, object]:
    """Extract every object in ``folder`` and stack them into ``out_dir``.

    ``extract_kwargs`` go to ``run_seurat_extraction`` for every member
    (``--rna-assay``, when given, applies to every file). ``required_columns``
    are ``--replicate-column`` and ``--stratify-by``: every file must carry them.
    """
    from paralome.atac.singlecell import (
        SC_EXTRACT_SCHEMA, _atomic_json, _sha256_file, _write_extraction_qc,
        run_seurat_extraction,
    )
    from paralome.checkpoints import exclusive_checkpoint_lock, stage_checkpoint_path
    from paralome.singlecell.cell_export import export_home

    folder = Path(folder).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    members = rds_members(folder)
    members_root = output_dir / MEMBERS_DIRNAME

    # 1. Each object on its own, one in memory at a time, each cached.
    extracted: Dict[str, Mapping[str, object]] = {}
    for index, member in enumerate(members, start=1):
        print(f"[paralome] --rds folder: object {index} of {len(members)}: {member.name}", flush=True)
        # Each object is its own dataset; its cell export goes in <parent>/omicview, one ZIP per object.
        extracted[member.name] = run_seurat_extraction(
            member, members_root / member_prefix(member),
            overwrite=overwrite, require_existing=require_existing, write_qc=False,
            cell_export_dir=cell_export_dir or export_home(output_dir.parent), **extract_kwargs,
        )
    dirs = {name: members_root / member_prefix(folder / name) for name in extracted}

    # 2. Every check, every file, before anything is stacked.
    #    Metadata: the command line is the authority.
    wanted = [column for column in dict.fromkeys(required_columns) if column]
    headers = {name: _header(dirs[name] / "cell_metadata.tsv") for name in extracted}
    lacking = {
        name: [column for column in wanted if column not in header]
        for name, header in headers.items()
    }
    lacking = {name: columns for name, columns in lacking.items() if columns}
    if lacking:
        detail = "\n".join(f"  {name}: {', '.join(columns)}" for name, columns in lacking.items())
        raise ParalomeError(
            224, f"{len(lacking)} of {len(members)} objects in {folder} lack a column "
                 f"--replicate-column/--stratify-by names:\n{detail}")

    halves = {name: (bool(result.get("has_atac")), bool(result.get("has_rna")))
              for name, result in extracted.items()}

    #    Gene lists: never a winner. By Ensembl ID; unmapped genes by name.
    has_rna = all(rna for _, rna in halves.values())
    gene_rows: Dict[str, List[Dict[str, str]]] = {}
    gene_fields: List[str] = []
    if any(rna for _, rna in halves.values()):
        key_lists: Dict[str, List[str]] = {}
        labels: Dict[str, str] = {}
        evidence: Dict[str, str] = {}
        for name in extracted:
            if not halves[name][1]:
                key_lists[name] = []
                continue
            fields, rows = _read_tsv(dirs[name] / "rna_genes.tsv")
            gene_fields = gene_fields or fields
            gene_rows[name] = rows
            keys = _gene_keys(rows)
            key_lists[name] = keys
            for key, row in zip(keys, rows):
                labels.setdefault(key, row.get("feature_name") or key)
            identity = extracted[name].get("gene_identity") or {}
            if identity:
                evidence[name] = (f"{float(identity.get('fraction') or 0):.1%} of its genes "
                                  f"match the --genome GTF")
        groups = _groups(key_lists)
        if len(groups) > 1:
            _report_differing("gene", groups, labels, output_dir, GENE_GROUPS_FILE, 225, evidence)

    #    Peaks, likewise, by coordinates.
    has_atac = all(atac for atac, _ in halves.values())
    peak_rows: Dict[str, List[List[str]]] = {}
    if any(atac for atac, _ in halves.values()):
        key_lists = {}
        for name in extracted:
            if not halves[name][0]:
                key_lists[name] = []
                continue
            with (dirs[name] / "peaks.bed").open(newline="") as handle:
                rows = [row for row in csv.reader(handle, delimiter="\t") if row]
            peak_rows[name] = rows
            key_lists[name] = [f"{row[0]}:{row[1]}-{row[2]}" for row in rows]
        groups = _groups(key_lists)
        if len(groups) > 1:
            _report_differing("peak", groups, {}, output_dir, PEAK_GROUPS_FILE, 226, {})

    # Metadata columns only some objects carry are dropped, and named.
    shared_columns = [c for c in headers[next(iter(headers))] if all(c in h for h in headers.values())]
    dropped = sorted({c for h in headers.values() for c in h} - set(shared_columns))

    # 3. Stack -- or reuse the stacked extraction these members already made.
    checkpoint_path = stage_checkpoint_path(output_dir, "singlecell_extract")
    signature = _digest({
        "method": STACK_METHOD,
        "folder": str(folder),
        "members": [
            {"name": name, "signature": result.get("signature"),
             "outputs": result.get("output_sha256"),
             "rna_assay": result.get("rna_assay"), "rna_slot": result.get("rna_slot"),
             "gene_symbol_column": result.get("gene_symbol_column")}
            for name, result in extracted.items()
        ],
    })
    if checkpoint_path.is_file():
        try:
            existing = json.loads(checkpoint_path.read_text())
        except (OSError, ValueError):
            existing = {}
        if (existing.get("complete") and existing.get("signature") == signature
                and all((output_dir / name).is_file() for name in existing.get("outputs") or {})):
            _write_extraction_qc(existing, output_dir, overwrite=False)
            return existing
    if require_existing:
        raise FileNotFoundError(
            f"No usable stacked extraction in {output_dir}; run the pipeline stage first.")

    from scipy.sparse import hstack

    from paralome.checkpoints import atomic_text_writer, atomic_write_text

    names = list(extracted)
    first = names[0]
    outputs: List[str] = ["cells.tsv", "cell_metadata.tsv"]

    # Cells and metadata, barcodes prefixed with their file's name.
    cells: List[str] = []
    with atomic_text_writer(output_dir / "cell_metadata.tsv", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(shared_columns)
        for name in names:
            prefix = member_prefix(folder / name) + "_"
            member_cells = _lines(dirs[name] / "cells.tsv")
            cells.extend(prefix + barcode for barcode in member_cells)
            _, rows = _read_tsv(dirs[name] / "cell_metadata.tsv")
            for row in rows:
                values = [row.get(column, "") for column in shared_columns]
                if "barcode" in shared_columns:
                    position = shared_columns.index("barcode")
                    values[position] = prefix + values[position]
                writer.writerow(values)
    atomic_write_text(output_dir / "cells.tsv", "\n".join(cells) + "\n")

    if has_rna:
        order = _occurrence_keys(_gene_keys(gene_rows[first]))
        blocks = []
        for name in names:
            position = {key: row for row, key in enumerate(_occurrence_keys(_gene_keys(gene_rows[name])))}
            matrix = _matrix(dirs[name], "rna_counts.mtx")
            blocks.append(matrix[[position[key] for key in order], :])
        write_plmsp(hstack(blocks, format="csc"), output_dir / "rna_counts.plmsp")
        del blocks
        # The first file's names for the rows; per-object statistics (Seurat's
        # variable-feature metrics) describe one object each and are not carried.
        keep = [field for field in ("feature_name", "gene_id", "common_id") if field in gene_fields]
        with atomic_text_writer(output_dir / "rna_genes.tsv", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(keep)
            for row in gene_rows[first]:
                writer.writerow([row.get(field, "") for field in keep])
        atomic_write_text(output_dir / "rna_genes_order.tsv",
                          "\n".join(row["feature_name"] for row in gene_rows[first]) + "\n")
        outputs += ["rna_counts.plmsp", "rna_genes.tsv", "rna_genes_order.tsv"]

    if has_atac:
        coordinates = [f"{row[0]}:{row[1]}-{row[2]}" for row in peak_rows[first]]
        blocks = []
        for name in names:
            position = {f"{row[0]}:{row[1]}-{row[2]}": index
                        for index, row in enumerate(peak_rows[name])}
            matrix = _matrix(dirs[name], "counts.mtx")
            blocks.append(matrix[[position[key] for key in coordinates], :])
        write_plmsp(hstack(blocks, format="csc"), output_dir / "counts.plmsp")
        del blocks
        atomic_write_text(output_dir / "peaks.bed",
                          "".join("\t".join(row[:4]) + "\n" for row in peak_rows[first]))
        atomic_write_text(output_dir / "peaks_order.tsv",
                          "".join(row[3] + "\n" for row in peak_rows[first]))
        atomic_write_text(output_dir / "peaks_features.tsv",
                          "peak_id\n" + "".join(row[3] + "\n" for row in peak_rows[first]))
        outputs = ["peaks.bed", "peaks_order.tsv", "counts.plmsp"] + outputs

    first_result = extracted[first]
    manifest = {
        "schema": "paralome.atac.singlecell_extraction_manifest.v1",
        "has_atac": has_atac,
        "has_rna": has_rna,
        "n_cells": len(cells),
        "n_peaks": len(peak_rows[first]) if has_atac else 0,
        "n_genes": len(gene_rows[first]) if has_rna else None,
        "metadata_columns": shared_columns,
        "rds_folder": str(folder),
        "rds_members": names,
    }
    atomic_write_text(output_dir / "extraction_manifest.json",
                      json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result: Dict[str, object] = {
        "schema": SC_EXTRACT_SCHEMA,
        "stage": "singlecell_extract",
        "complete": True,
        "signature": signature,
        "rds": str(folder),
        "rds_folder": True,
        "rds_members": {
            name: {"rds": str(folder / name), "signature": result.get("signature"),
                   "n_cells": result.get("n_cells"), "extraction": str(dirs[name])}
            for name, result in extracted.items()
        },
        "stack_method": STACK_METHOD,
        "metadata_columns": shared_columns,
        "metadata_columns_dropped": dropped,
        "assay": first_result.get("assay"),
        "has_atac": has_atac,
        "has_rna": has_rna,
        "slot": first_result.get("slot"),
        "n_cells": len(cells),
        "n_peaks": manifest["n_peaks"],
        "n_genes": manifest["n_genes"],
        "seurat_version": first_result.get("seurat_version"),
        "signac_version": first_result.get("signac_version"),
        "outputs": {name: str(output_dir / name) for name in outputs},
        "output_sha256": {name: _sha256_file(output_dir / name) for name in outputs},
        "gene_identity": {name: result.get("gene_identity") for name, result in extracted.items()},
        "rna_assay": first_result.get("rna_assay"),
        "rna_slot": first_result.get("rna_slot"),
        "gene_symbol_column": first_result.get("gene_symbol_column"),
    }
    with exclusive_checkpoint_lock(checkpoint_path):
        _atomic_json(checkpoint_path, result)
    result["checkpoint"] = str(checkpoint_path)
    if dropped:
        print(f"[paralome] metadata columns not in every object, left out: {', '.join(dropped)}",
              flush=True)
    print(f"[paralome] stacked {len(names)} objects: {len(cells):,} cells", flush=True)
    _write_extraction_qc(result, output_dir, overwrite=True)
    return result
