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

import csv
import sys
from typing import Iterable, Iterator


def skip_comment_lines(handle: Iterable[str]) -> Iterator[str]:
    """Filter out full-line comments before they reach ``csv.DictReader``.

    A line whose first character is ``#`` is dropped entirely. This is the
    convention every Paralome sample manifest reader uses to let a line be
    "soft-disabled" without deleting it or maintaining a second, divergent
    copy of the samplesheet -- e.g. a real PCA/outlier-QC outlier found
    after a completed run, which should be excluded from the next round of
    quantification/statistics without touching the alignment data already
    computed for every other sample. Only a genuine first-character ``#``
    counts: a ``#`` appearing anywhere else in a line (mid-field) is left
    alone, so a legitimate data value that happens to contain ``#`` is
    never silently dropped.

    This must run *before* ``csv.DictReader`` ever sees the file, since the
    ``csv`` module has no native comment-line concept:

        csv.DictReader(skip_comment_lines(handle))

    Not a general CSV-comment parser: a quoted field containing an
    embedded newline whose continuation line happens to start with ``#``
    would be (incorrectly) dropped. Accepted limitation for the flat,
    one-line-per-sample manifests Paralome actually reads with this
    helper -- none of them use multi-line quoted fields.
    """
    first = True
    for line in handle:
        # A byte-order mark at the top of the file is not data. Left
        # in, it became part of the first column's name ("\ufeffsample_id") and
        # hid a '#' on the first line. Dropped here so every reader that uses
        # this helper tolerates one, however it opened the file.
        if first:
            first = False
            if line.startswith("\ufeff"):
                line = line[1:]
        if line.startswith("#"):
            continue
        yield line


def configure_csv_field_size_limit(preferred: int | None = None) -> int:
    """Raise Python's process-wide CSV field limit as high as the platform permits.

    Annotation and INFO-derived columns can legitimately exceed the csv module's
    128 KiB default.  CPython accepts a C ``long`` here, so progressively reduce
    ``sys.maxsize`` on platforms where it overflows.
    """
    limit = int(preferred if preferred is not None else sys.maxsize)
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10
    raise RuntimeError("Could not configure a usable CSV field-size limit")
