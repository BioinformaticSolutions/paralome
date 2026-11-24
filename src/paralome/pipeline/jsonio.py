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
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: The key every checkpoint carries recording which paralome wrote it.
#:
#: The plan hash records what a stage's result depends on, but the list of
#: things that can change is the whole program. A checkpoint written by a
#: different version is not a result the current code produced, whatever its
#: parameters say. Resumability is preserved WITHIN a version and forced across
#: one.
CHECKPOINT_VERSION_KEY = "paralome_version"


def _current_version() -> str:
    # Imported lazily: paralome.version is a leaf with no dependencies, but
    # jsonio is imported by pipeline modules during package import and a
    # module-level import here would put version resolution on that path for
    # no benefit.
    from paralome.version import __version__

    return str(__version__)


def stamp_version(payload: Mapping[str, Any], *, version: Optional[str] = None) -> Dict[str, Any]:
    """``payload`` plus the version that wrote it. Returns a copy.

    Applied by the shared JSON writers rather than by each of the ~35 call
    sites that build a checkpoint dict, so a new stage gets the stamp by
    construction instead of by remembering. An explicit ``version`` is honoured
    so a test can write a checkpoint that claims to be from another release
    without monkeypatching the module.
    """
    stamped = dict(payload)
    stamped[CHECKPOINT_VERSION_KEY] = str(version) if version is not None else _current_version()
    return stamped


def version_stale_reason(
    previous: Mapping[str, Any], *, version: Optional[str] = None
) -> Optional[str]:
    """Why ``previous`` cannot be reused by this version, or ``None``.

    Deliberately shaped like ``checkpoint_stale_reason``: it answers "is this
    stale, and why" by NAME, so a re-run prints a sentence a user can act on
    rather than recomputing in silence ("it did not re-run" and "it re-ran and
    produced the same thing" look identical from outside).

    An UNSTAMPED checkpoint is stale: it predates the stamp and is not trusted.
    """
    current = str(version) if version is not None else _current_version()
    recorded = previous.get(CHECKPOINT_VERSION_KEY)
    if not recorded:
        return (
            f"written by a paralome older than {current} (no "
            f"{CHECKPOINT_VERSION_KEY} recorded); re-running because a checkpoint "
            f"from before 0.0.146 cannot say what produced it"
        )
    if str(recorded) != current:
        return f"written by paralome {recorded}, running {current}"
    return None


def checkpoint_stale_reason(previous: Mapping[str, Any], **expected: Any) -> Optional[str]:
    """Compare requested stage parameters against what a checkpoint recorded.

    A checkpointed stage (``rna quantify-fpkm``, ``rna differential``, and by
    extension anything that delegates into them -- the per-block
    parallel/SLURM path included) reuses a cached result only when the
    parameters that produced it match: rerunning ``rna quantify-fpkm`` with a
    new ``--strandedness`` value recomputes. Pass the stage's own checkpoint
    dict plus the *current* invocation's parameters as keyword arguments, and
    this reports whether anything that actually matters changed.

    Returns ``None`` when every provided keyword argument matches the
    checkpoint's recorded value for that same key (the checkpoint is still
    valid and safe to reuse). Otherwise returns a short, human-readable
    description of the first mismatch found -- e.g.
    ``"strandedness changed from 1 to 2"`` -- meant to be logged so a user
    can see *why* a stage recomputed.

    List/tuple values are compared order-sensitively but container-type-
    insensitively (a caller's tuple vs. a checkpoint's JSON-round-tripped
    list for the same logical value should not count as a mismatch).

    This is deliberately a comparison of *parameters*, not input file
    contents; input files are compared by ``inputs_signature``.
    """
    for key, value in expected.items():
        recorded = previous.get(key)
        normalized_value = list(value) if isinstance(value, (list, tuple)) else value
        normalized_recorded = list(recorded) if isinstance(recorded, (list, tuple)) else recorded
        if normalized_recorded != normalized_value:
            return f"{key} changed from {recorded!r} to {value!r}"
    return None


def input_fingerprint(
    *,
    hashed: Sequence[Any] = (),
    stamped: Sequence[Any] = (),
    content: Any = None,
) -> str:
    """One string that changes when a stage's inputs change.

    ``checkpoint_stale_reason`` compares parameters, not inputs. This covers
    the inputs: comment a sample out of the sheet, regenerate the expression
    matrix, point at another GTF, and a completed stage recomputes rather than
    returning the old result (and, under SLURM, blocks computed from an old
    sample set are never merged under a checkpoint describing a new one).

    ``hashed`` files are identified by content (matrices; hashing them is a
    second's work). ``stamped`` files are identified by path, size and
    modification time (BAMs and annotations, too large to read for this).
    ``content`` is anything JSON can hold -- the parsed sample rows, so that
    re-saving a sheet without changing it is not a change.

    Raises ``OSError`` when a file is not there. Callers treat that as "cannot
    compare" and leave the decision to the other checks: a finished result
    whose input has since been deleted is still a finished result.
    """
    import hashlib

    digest = hashlib.sha256()
    for item in hashed:
        path = Path(item)
        digest.update(b"H")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    for item in stamped:
        path = Path(item)
        stat = path.stat()
        digest.update(f"S|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    digest.update(json.dumps(content, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def inputs_changed_reason(previous: Mapping[str, Any], fingerprint: str) -> Optional[str]:
    """Why ``previous`` was not computed from these inputs, or ``None``.

    A checkpoint with no recorded fingerprint predates the check and is NOT
    called stale by it: doing so would recompute every finished quantification
    on disk the first time it was looked at. The protection starts with the
    next result each stage writes.
    """
    recorded = previous.get("input_fingerprint")
    if not recorded or recorded == fingerprint:
        return None
    return "its inputs have changed since it was computed (samples, matrix, annotation or alignments)"


def invalidate_checkpoint(path: Path, previous: Mapping[str, Any], reason: str) -> None:
    """Record that ``path``'s result is being replaced, before replacing it.

    So a stage that recomputes and is killed part way cannot leave its old
    checkpoint saying ``complete: true`` over a mixture of old and new files.
    """
    body = dict(previous)
    body["complete"] = False
    body["invalidated"] = reason
    write_json(Path(path), body, atomic=True)


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    atomic: bool = False,
    stamp: bool = False,
    default: Any = None,
) -> Path:
    """Write deterministic, newline-terminated JSON.

    Atomic writes use a sibling temporary file and ``os.replace`` so readers
    never observe a partially written checkpoint.

    ``stamp=True`` records CHECKPOINT_VERSION_KEY. It is opt-in: this function
    is the writer for rna/fpkm.py, rna/differential.py and rna/parallel.py,
    whose existing checkpoints must keep resolving, and stamping by default
    would change the contents of every RNA checkpoint.

    The ATAC writers stamp through their own ``_atomic_json`` wrappers, so the
    ATAC checkpoints are covered without this default touching anything else.

    ``default`` mirrors ``json.dumps``, for callers that were passing it.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = stamp_version(payload) if stamp else dict(payload)
    rendered = json.dumps(body, indent=2, sort_keys=True, default=default) + "\n"

    if not atomic:
        destination.write_text(rendered, encoding="utf-8")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def missing_checkpoint_outputs(
    previous: Mapping[str, Any], keys: Sequence[str]
) -> List[str]:
    """Which of ``keys``' recorded output paths are no longer on disk.

    ``checkpoint_stale_reason`` compares PARAMETERS, which answers
    "was this computed the way I am asking for it now?" It cannot answer "is
    the result still there?", and a checkpoint saying ``complete: true`` next
    to a deleted output is as misleading as one next to a wrongly-named output.
    Deleting a results directory and re-running is a routine case while
    iterating, and the stage must recompute rather than hand back the old
    checkpoint with no files.

    Returns the missing paths so the caller can name them, rather than a bool
    -- "recomputing because X is gone" is actionable; "recomputing" is not.
    Entries that are None or empty are skipped: an optional output that was
    never produced is not a missing one.
    """
    from pathlib import Path

    missing: List[str] = []
    for key in keys:
        recorded = previous.get(key)
        if not recorded:
            continue
        if not Path(str(recorded)).exists():
            missing.append(str(recorded))
    return missing
