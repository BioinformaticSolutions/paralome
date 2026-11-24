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

"""The sample sheet: which abundance column is which sample, and what it is.

One row per sample. Two columns are required:

    column      the abundance column, as its header is written in the export or
                as the sample name inside it (``Ctrl_1`` for
                ``Abundances (TMT 126, Ctrl_1)``)
    sample_id   the name Paralome uses for the sample

Every other column is metadata (treatment, source, batch, ...), read by
--factor, --stratify-by and --pair-by exactly as every other pipeline reads
its sample sheet. A paired protein and phospho export
share one sheet: their channels are the same samples.

Every abundance column must be in the sheet and every sheet row must name an
abundance column. All faults are reported at once (PLM-0228), as the preflight
does, rather than the first.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

REQUIRED = ("column", "sample_id")


@dataclass
class SampleSheet:
    path: str
    rows: List[Dict[str, str]]         # one per sample, sheet order

    @property
    def sample_ids(self) -> List[str]:
        return [row["sample_id"] for row in self.rows]

    def metadata_columns(self) -> List[str]:
        return [c for c in self.rows[0] if c not in REQUIRED] if self.rows else []


def read_sample_sheet(path: Union[str, Path]) -> SampleSheet:
    from paralome.errors import ParalomeError

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ParalomeError(104, f"Sample sheet not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    header = [h.strip() for h in (reader.fieldnames or [])]
    faults: List[str] = []
    missing = [name for name in REQUIRED if name not in header]
    if missing:
        raise ParalomeError(228, f"{path}: the sample sheet has no {' or '.join(missing)} column "
                                 f"(it has: {', '.join(header) or 'nothing'})")
    rows: List[Dict[str, str]] = []
    for number, raw in enumerate(reader, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k is not None}
        if not any(row.values()):
            continue
        for name in REQUIRED:
            if not row.get(name):
                faults.append(f"line {number}: no {name}")
        rows.append(row)
    seen: Dict[str, int] = {}
    for row in rows:
        seen[row.get("sample_id", "")] = seen.get(row.get("sample_id", ""), 0) + 1
    faults += [f"sample_id {sid!r} is on {n} rows" for sid, n in seen.items() if sid and n > 1]
    if not rows:
        faults.append("the sheet lists no samples")
    if faults:
        raise ParalomeError(228, f"{path}:\n  " + "\n  ".join(faults))
    return SampleSheet(path=str(path), rows=rows)


def match_columns(sheet: SampleSheet, columns: Sequence[str], labels: Sequence[str], *, export: str) -> List[Optional[int]]:
    """For each sheet row, the index of its abundance column; every fault at once (PLM-0228)."""
    from paralome.errors import ParalomeError

    by_header = {c.strip(): i for i, c in enumerate(columns)}
    by_label: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label.strip(), []).append(i)
    order: List[Optional[int]] = []
    faults: List[str] = []
    for row in sheet.rows:
        name = row["column"]
        if name in by_header:
            order.append(by_header[name])
        elif len(by_label.get(name, [])) == 1:
            order.append(by_label[name][0])
        elif len(by_label.get(name, [])) > 1:
            faults.append(f"{name!r} names {len(by_label[name])} columns; give the full header")
            order.append(None)
        else:
            faults.append(f"sample {row['sample_id']!r}: column {name!r} is not in the export")
            order.append(None)
    used = {i for i in order if i is not None}
    faults += [f"abundance column {columns[i]!r} is not in the sheet" for i in range(len(columns)) if i not in used]
    if faults:
        raise ParalomeError(
            228, f"{export} and the sample sheet {sheet.path} do not match:\n  " + "\n  ".join(faults)
            + "\n  The export's sample names are: " + ", ".join(labels),
        )
    return order
