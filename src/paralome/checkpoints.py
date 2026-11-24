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

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar, Union


T = TypeVar("T")


def file_info(path: Union[str, Path, None]) -> dict:
    """Describe an output file for stage-state and provenance records.

    This existed four times over, AST-identical in every copy:
    ``annotation/standalone.py:file_info``, ``filtering/runner.py:_file_info``,
    ``pipeline/state.py:_file_info`` and ``segregation/runner.py:file_info`` --
    two of them public, two private, all four the same seven lines.

    It lives here because "does this output exist and is it non-empty" is the
    question every stage-completion check asks, and this module already owns the
    rest of the durable-state-on-disk vocabulary: the stage checkpoint resolver,
    the atomic writers and the cross-process lock. ``pipeline/state.py``, which
    accounts for 88 of the call sites, already imported from here.

    The returned shape is unchanged, because ``project_state.json`` contains it
    and older project folders must keep parsing.
    """
    if path is None:
        return {"path": None, "exists": False, "nonempty": False, "size": 0}
    p = Path(path)
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    return {"path": str(p), "exists": exists, "nonempty": bool(exists and size > 0), "size": size}


def count_table_rows(path: Union[str, Path, None]) -> int:
    """Data lines in a tab-separated table with one header line; 0 when absent."""
    if path is None:
        return 0
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    lines = 0
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            lines += block.count(b"\n")
        # A last line without its newline is still a line.
        if p.stat().st_size and not _ends_with_newline(p):
            lines += 1
    return max(0, lines - 1)


def _ends_with_newline(p: Path) -> bool:
    with p.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def recorded_output_sizes(outputs: dict[str, Any]) -> dict[str, int]:
    """Byte size of each named output, for a checkpoint to record as it closes.

    Stages that rewrite their tables in place (segregation, filtering) record
    what the tables measured when the checkpoint was written, so a reader can
    see that the files beside a "lossless, N rows" claim are no longer those
    tables (a later --overwrite run killed part way, say) without reading them.
    Keyed by output name, not path, so a copied project still compares.
    """
    sizes: dict[str, int] = {}
    for key, path in outputs.items():
        if not path:
            continue
        info = file_info(path)
        if info["exists"]:
            sizes[str(key)] = int(info["size"])
    return sizes


def output_sizes_agree(recorded: Any, outputs: dict[str, Any]) -> bool:
    """Whether every output a checkpoint recorded a size for still has that size."""
    if not isinstance(recorded, dict) or not recorded:
        return False
    for key, size in recorded.items():
        path = outputs.get(key)
        if not path:
            return False
        info = file_info(path)
        if not info["exists"] or int(info["size"]) != int(size):
            return False
    return True


def vcf_with_index_ready(path: Union[str, Path, None]) -> bool:
    """Whether a compressed VCF is finished: present, and indexed after it was written.

    "The file exists and is not empty" is true of a VCF from the first block
    written to it, and the annotation merge writes its output in place. The
    index is the last thing every Paralome VCF writer produces, and it cannot
    be built from a truncated file, so its presence is the evidence that the
    VCF is whole. It must not be older than the VCF: an index left over from an
    earlier run says nothing about the file beside it now (tabix makes the same
    comparison and warns).
    """
    if path is None:
        return False
    vcf = Path(path)
    try:
        vcf_stat = vcf.stat()
    except OSError:
        return False
    if vcf_stat.st_size <= 0:
        return False
    for suffix in (".tbi", ".csi"):
        try:
            index_stat = Path(str(vcf) + suffix).stat()
        except OSError:
            continue
        if index_stat.st_size > 0 and index_stat.st_mtime >= vcf_stat.st_mtime:
            return True
    return False


@contextmanager
def exclusive_checkpoint_lock(path: Union[str, Path]) -> Iterator[None]:
    """Serialize checkpoint read-modify-write operations across processes."""
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def temporary_beside(path: Union[str, Path], *, suffix: str = ".tmp") -> tuple[int, Path]:
    """(descriptor, name) of a new file in ``path``'s folder, to rename onto it.

    Not mkstemp. It makes the file readable by its owner alone and
    the rename carried that onto every checkpoint, matrix and workbook, so a
    colleague in a shared project could not read them. Created as open() would
    create it: the umask decides, and so do the folder's setgid bit and ACL.
    """
    path = Path(path)
    while True:
        name = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{suffix}"
        try:
            return os.open(str(name), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666), name
        except FileExistsError:
            continue


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write a file atomically using a unique temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = temporary_beside(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@contextmanager
def atomic_text_writer(path: Union[str, Path], *, newline: Optional[str] = None) -> Iterator[Any]:
    """Stream text to ``path`` atomically: temp file, fsync, rename, fsync dir.

    Identical copies of this lived in ``atac/peak_merge.py``
    and ``atac/quantification.py``. It belongs beside ``atomic_write_text``,
    which is the same durability contract for the case where the whole payload
    is already a string in memory; this is the streaming form, for the matrices
    and BED files too large to build up first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = temporary_beside(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class SampleCheckpoint:
    sample_id: str
    index: Optional[int] = None
    complete: bool = False
    started: Optional[str] = None
    finished: Optional[str] = None
    failed: Optional[str] = None
    outputs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleCheckpoint":
        return cls(
            sample_id=str(data.get("sample_id", "")),
            index=data.get("index"),
            complete=bool(data.get("complete", False)),
            started=data.get("started"),
            finished=data.get("finished"),
            failed=data.get("failed"),
            outputs=dict(data.get("outputs") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "index": self.index,
            "complete": self.complete,
            "started": self.started,
            "finished": self.finished,
            "failed": self.failed,
            "outputs": self.outputs,
            "metadata": self.metadata,
        }


@dataclass
class StageCheckpoint:
    stage: str
    workdir: str
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    expected: int = 0
    resources: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    samples: dict[str, SampleCheckpoint] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StageCheckpoint":
        path = Path(path)
        data = json.loads(path.read_text())
        ckpt = cls(
            stage=str(data.get("stage", "unknown")),
            workdir=str(data.get("workdir", path.parent.parent)),
            created=str(data.get("created", now_iso())),
            updated=str(data.get("updated", now_iso())),
            expected=int(data.get("expected", 0) or 0),
            resources=dict(data.get("resources") or {}),
            outputs=dict(data.get("outputs") or {}),
        )
        ckpt.samples = {
            sid: SampleCheckpoint.from_dict(v)
            for sid, v in (data.get("samples") or {}).items()
        }
        return ckpt

    @classmethod
    def create(cls, *, stage: str, workdir: Union[str, Path], samples: Iterable[SampleCheckpoint], resources: Optional[dict[str, Any]] = None, outputs: dict[str, str] | None = None) -> "StageCheckpoint":
        sample_map = {s.sample_id: s for s in samples}
        return cls(
            stage=stage,
            workdir=str(workdir),
            expected=len(sample_map),
            resources=dict(resources or {}),
            outputs=dict(outputs or {}),
            samples=sample_map,
        )

    def to_dict(self) -> dict[str, Any]:
        complete_ids = self.complete_samples()
        failed_ids = self.failed_samples()
        missing_ids = self.missing_samples()
        return {
            "stage": self.stage,
            "workdir": self.workdir,
            "created": self.created,
            "updated": self.updated,
            "expected": self.expected,
            "complete": len(complete_ids) == self.expected and self.expected > 0,
            "completed_samples": len(complete_ids),
            "missing_samples": len(missing_ids),
            "failed_samples": len(failed_ids),
            "complete_sample_ids": complete_ids,
            "missing_sample_ids": missing_ids,
            "failed_sample_ids": failed_ids,
            "resources": self.resources,
            "outputs": self.outputs,
            "samples": {sid: sample.to_dict() for sid, sample in self.samples.items()},
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        self.updated = now_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def complete_samples(self) -> list[str]:
        return [sid for sid, sample in self.samples.items() if sample.complete]

    def failed_samples(self) -> list[str]:
        return [sid for sid, sample in self.samples.items() if sample.failed and not sample.complete]

    def missing_samples(self) -> list[str]:
        return [sid for sid, sample in self.samples.items() if not sample.complete]

    def missing_indices(self) -> list[int]:
        return [sample.index for sample in self.samples.values() if not sample.complete and sample.index is not None]

    def mark_complete(self, sample_id: str, *, outputs: dict[str, str] | None = None, metadata: Optional[dict[str, Any]] = None) -> None:
        sample = self.samples[sample_id]
        sample.complete = True
        sample.failed = None
        sample.finished = now_iso()
        if outputs:
            sample.outputs.update(outputs)
        if metadata:
            sample.metadata.update(metadata)



# Stages whose checkpoint does NOT sit at <workdir>/checkpoints/.
#
# The two pseudobulk stages write their output one level down -- ATAC's into <workdir>/atac/,
# RNA's into <workdir>/rna/ -- so one shared data folder can hold both modalities beside a single
# singlecell_extraction/, and their checkpoints sit with their output.
#
# The mapping lives here, once, and both writers and readers (pipeline/state.py) resolve through
# stage_checkpoint_path() below; a checkpoint looked for in the wrong place would read as `{}`,
# never complete. A stage absent from this dict uses the flat default, which is every other stage
# in the package.
STAGE_CHECKPOINT_SUBDIR: dict[str, str] = {
    "singlecell_atac": "atac",
    "singlecell_rna": "rna",
}


def stage_checkpoint_path(
    workdir: Union[str, Path], stage: str, *, variant: Optional[str] = None
) -> Path:
    """The single authoritative location of ``stage``'s checkpoint under ``workdir``.

    Every writer and every reader must resolve through this function rather
    than composing the path by hand, so moving the write path cannot leave the
    read path behind.

    ``variant`` distinguishes several runs of one stage that share a workdir --
    in practice the stratum slug, since the stratified fan-out writes every
    stratum into one flat directory rather than a subdirectory each. It is a
    parameter here rather than a caller-side f-string for the same reason: the
    fan-out's reader resolves the same way its writer does.
    """
    subdir = STAGE_CHECKPOINT_SUBDIR.get(stage)
    base = Path(workdir) / subdir if subdir else Path(workdir)
    name = f"{variant}_{stage}.json" if variant else f"{stage}.json"
    return base / "checkpoints" / name


def legacy_stage_checkpoint_path(workdir: Union[str, Path], stage: str) -> Optional[Path]:
    """Where ``stage``'s checkpoint lived nested it, if it moved.

    Read-side only. Project folders created before that layout change still
    have their checkpoint at the flat location, and a summarizer that ignored
    them would report a genuinely finished stage as incomplete -- the same
    failure this patch fixes, in the other direction. Returns None for stages
    that never moved.
    """
    if stage not in STAGE_CHECKPOINT_SUBDIR:
        return None
    return Path(workdir) / "checkpoints" / f"{stage}.json"


def checkpoint_path(workdir: Union[str, Path], stage: str) -> Path:
    return stage_checkpoint_path(workdir, stage)


def load_stage_checkpoint(workdir: Union[str, Path], stage: str) -> Optional[StageCheckpoint]:
    path = checkpoint_path(workdir, stage)
    if not path.exists():
        return None
    return StageCheckpoint.load(path)


def save_stage_checkpoint(workdir: Union[str, Path], checkpoint: StageCheckpoint) -> Path:
    path = checkpoint_path(workdir, checkpoint.stage)
    checkpoint.save(path)
    return path


def update_stage_checkpoint(
    workdir: Union[str, Path],
    stage: str,
    updater: Callable[[StageCheckpoint], T],
) -> T:
    """Lock, reload, mutate, and atomically save a shared stage checkpoint."""
    path = checkpoint_path(workdir, stage)
    with exclusive_checkpoint_lock(path):
        if not path.exists():
            raise FileNotFoundError(f"Stage checkpoint does not exist: {path}")
        checkpoint = StageCheckpoint.load(path)
        result = updater(checkpoint)
        checkpoint.save(path)
        return result


def compress_int_ranges(values: Iterable[int]) -> str:
    nums = sorted({int(v) for v in values})
    if not nums:
        return ""
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def slurm_array_spec(indices: Iterable[int], throttle: Optional[int] = None) -> str:
    spec = compress_int_ranges(indices)
    if throttle and throttle > 0 and spec:
        return f"{spec}%{int(throttle)}"
    return spec
