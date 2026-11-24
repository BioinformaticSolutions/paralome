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

"""Check what the user typed, before anything is submitted.

Every user-defined parameter from the CLI gets a specific error for
its data type and a suggestion on how to fix it.

The options on the human-facing surface fall into seven KINDS, and this module
is one implementation per kind -- the same convention the rest of the package
runs on (one shared implementation per cross-cutting concern). Adding an option
means choosing a kind, not writing a validator.

  input_path        --rds --samples --gtf --reference --star-genome
  output_path       --project --out                    (in pipeline/logging)
  metadata_column   --replicate-column --stratify-by --factor
  choice            --backend
  bounded_number    --threads --jobs --alpha --min-cells-per-group
  factor_spec       --factor COLUMN:CONTROL            (parsing lives in stats.parse_factor)
  genome_bundle     --genome

WHY IN THE FOREGROUND. Nothing about "this file does not exist" requires a
scheduler to discover. Every check here is one that can be answered on the
login node, in milliseconds, before a job is queued, rather than on a compute
node one poll cycle later.

The checks that genuinely CANNOT be answered early -- which metadata columns an
unread Seurat object has -- are exposed here anyway, so the stage can run them
at the first moment the answer exists, and so a second run against an existing
extraction gets them in the foreground after all.
"""

from __future__ import annotations

import difflib
import os
import pwd
import stat
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from paralome.errors import ParalomeError

__all__ = [
    "check_input_path",
    "check_metadata_columns",
    "check_choice",
    "check_bounded_number",
    "extraction_metadata_columns",
]


def _owner(path: Path) -> str:
    try:
        info = path.stat()
    except OSError:
        return "unknown"
    try:
        name = pwd.getpwuid(info.st_uid).pw_name
    except (KeyError, AttributeError):
        name = str(info.st_uid)
    return f"{name}, mode {stat.filemode(info.st_mode)}"


def _running_as() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, AttributeError):  # pragma: no cover - unusual environments
        return str(os.getuid())


def _near_misses(name: str, candidates: Iterable[str], *, limit: int = 3) -> Sequence[str]:
    """Names close enough to be worth suggesting.

    Deliberately generous (cutoff 0.6): the cost of an unhelpful suggestion is
    one line of output, and the cost of not suggesting is an afternoon
    spent opening cell_metadata.tsv to find out that a column was not there.
    """
    pool = [str(c) for c in candidates]
    close = difflib.get_close_matches(str(name), pool, n=limit, cutoff=0.6)
    if close:
        return close
    # difflib is poor at substring relationships -- 'sample_integrated_rds.rds'
    # against 'sample_integrated.rds' scores below the cutoff at these lengths.
    lowered = str(name).lower()
    return [c for c in pool if lowered in c.lower() or c.lower() in lowered][:limit]


def check_input_path(
    value: Union[str, Path, None],
    *,
    flag: str,
    must_exist: bool = True,
) -> Optional[Path]:
    """PLM-0104 / PLM-0105. A file the run reads.

    Names near-misses in the same directory, because a renamed or mistyped
    input is the overwhelmingly common case and the correct name is usually
    sitting right next to the wrong one.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if not must_exist:
        return path

    if not path.exists():
        # One line on the console (see errors.render_console). The near-miss and
        # the directory listing are real help, but they are help for AFTER the
        # user knows what broke -- they go to the log and to `paralome explain`.
        directory = path.parent if str(path.parent) else Path(".")
        try:
            siblings = sorted(p.name for p in directory.iterdir() if p.is_file())
        except OSError:
            siblings = []
        suggestions = _near_misses(path.name, siblings)
        raise ParalomeError(
            104,
            f"{flag} {str(path)!r} does not exist.",
            fix=(f"did you mean {suggestions[0]!r}?" if suggestions else None),
            searched=str(directory),
            near_misses=list(suggestions),
        )

    if path.is_dir():
        return path
    if not os.access(path, os.R_OK):
        raise ParalomeError(
            105,
            f"{flag} {str(path)!r} exists but is not readable.",
            fix=f"chmod +r {path}, or ask its owner to",
            owner=_owner(path),
            running_as=_running_as(),
        )
    return path


def extraction_metadata_columns(extraction_dir: Union[str, Path]) -> Optional[Sequence[str]]:
    """The metadata columns of an existing extraction, or None if there isn't one.

    Reads only the header line. This is what makes PLM-0201 a foreground error
    on every run after the first: once the object has been read once, the answer
    is on disk and there is no reason to submit a job to rediscover it.
    """
    path = Path(extraction_dir).expanduser() / "cell_metadata.tsv"
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").rstrip("\r")
    except OSError:
        return None
    if not header:
        return None
    return [column for column in header.split("\t") if column]


def check_metadata_columns(
    requested: Sequence[tuple],
    available: Optional[Sequence[str]],
) -> None:
    """PLM-0201. ``requested`` is ``[(flag, column), ...]``; blanks are skipped.

    ``available=None`` means the object has not been read yet, so nothing can be
    said -- the check simply does not run, rather than guessing.
    """
    if not available:
        return
    known = list(available)
    missing = [
        (flag, column) for flag, column in requested
        if column and column not in known
    ]
    if not missing:
        return

    named = ", ".join(f"{flag} {column!r}" for flag, column in missing)
    suggestions = {
        column: _near_misses(column, known) for _flag, column in missing
    }
    first = next((s[0] for s in suggestions.values() if s), None)
    # The column list goes on the CONSOLE for this one code, wrapped: it keeps
    # the user from digging through the metadata without being so wordy as to
    # lose the reader. It is the single fact a user would otherwise hunt through
    # cell_metadata.tsv for, and every other code stays at two lines.
    raise ParalomeError(
        201,
        f"Not a column of this object's cell metadata: {named}\n"
        f"Present: {', '.join(known)}",
        fix=(f"did you mean {first!r}?" if first else None),
        near_misses={k: v for k, v in suggestions.items() if v},
    )


def _reject_choice(value: str, flag: str, allowed: Sequence[str]) -> None:
    detail = f"{flag} {value!r} is not one of: {', '.join(allowed)}"
    suggestions = _near_misses(value, allowed, limit=1)
    fix = f"did you mean {suggestions[0]!r}?" if suggestions else None
    # Case-only mismatches deserve saying out loud: Paralome is case-sensitive
    # everywhere and has no aliases, which surprises people exactly once.
    for candidate in allowed:
        if candidate.lower() == value.lower() and candidate != value:
            fix = f"use {candidate!r} -- Paralome is case-sensitive"
            break
    raise ParalomeError(108, detail, fix=fix)


def check_choice(
    value: Optional[str],
    *,
    flag: str,
    allowed: Sequence[str],
    multiple: bool = False,
    aliases: Sequence[str] = (),
) -> Optional[str]:
    """PLM-0108. One of a fixed set, case-sensitively.

    ``multiple`` accepts a comma-separated list and validates each element.
    ``aliases`` are extra whole-value spellings that are not themselves members
    of ``allowed`` (an ``"all"`` a command expands itself).

    A comma-separated option is validated ELEMENT by element, so a typo in a
    two-item list names the item that is wrong. (--statistic is not checked
    here: it has its own grammar, read by
    ``paralome.stats.parse_statistic_request``.)
    """
    if value is None:
        return None
    if value in aliases:
        return value
    if not multiple:
        if value in allowed:
            return value
        _reject_choice(value, flag, allowed)

    parts = [part.strip() for part in value.split(",")]
    if not any(parts):
        raise ParalomeError(
            108, f"{flag} is empty",
            fix=f"name at least one of: {', '.join(allowed)}",
        )
    for part in parts:
        if not part:
            raise ParalomeError(
                108, f"{flag} {value!r} has an empty entry",
                fix="remove the stray comma",
            )
        if part not in allowed and part not in aliases:
            _reject_choice(part, flag, allowed)
    return value


def check_bounded_number(
    value,
    *,
    flag: str,
    minimum=None,
    maximum=None,
    exclusive: bool = False,
) -> None:
    """PLM-0112. A number inside its range.

    ``exclusive`` for things like --alpha, where 0 and 1 are as meaningless as
    -1 is: an alpha of 0 selects nothing and an alpha of 1 selects everything.
    """
    if value is None:
        return
    low_bad = minimum is not None and (value <= minimum if exclusive else value < minimum)
    high_bad = maximum is not None and (value >= maximum if exclusive else value > maximum)
    if not (low_bad or high_bad):
        return

    if minimum is not None and maximum is not None:
        span = (f"strictly between {minimum} and {maximum}" if exclusive
                else f"between {minimum} and {maximum}")
    elif minimum is not None:
        span = f"greater than {minimum}" if exclusive else f"at least {minimum}"
    else:
        span = f"less than {maximum}" if exclusive else f"at most {maximum}"

    fix = f"use a value {span}"
    if minimum == 1 and value == 0:
        # Worth its own sentence: most tools read 0 as "auto", and Paralome
        # does not, because on a shared cluster the CPU count is a scheduling
        # decision rather than a property of whatever node the job lands on.
        fix = "name the number explicitly -- 0 does not mean 'auto' here"
    raise ParalomeError(112, f"{flag} is {value}; it must be {span}.", fix=fix)
