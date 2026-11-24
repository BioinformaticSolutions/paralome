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

"""A duplicate-marked BAM's read and duplicate counts.

The fused alignment task writes them beside the BAM (``<bam>.flagstat.json``) with its own threads, so the
deduplication stage that adopts the BAMs reads them rather than counting every BAM again. A BAM without
them (made before this file was written) is counted with one threaded ``samtools flagstat`` pass, the
samples in parallel.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

FLAGSTAT_SIDECAR = ".flagstat.json"

T = TypeVar("T")
R = TypeVar("R")


def _from_flagstat(data: dict) -> tuple[int, int]:
    passed, failed = data.get("QC-passed reads") or {}, data.get("QC-failed reads") or {}
    total = int(passed.get("total") or 0) + int(failed.get("total") or 0)
    duplicates = int(passed.get("duplicates") or 0) + int(failed.get("duplicates") or 0)
    return total, duplicates


#: The primary-read counts a sequence QC table is built from (flagstat key, name), QC-passed and -failed together.
#: Reads, not records: secondary and supplementary alignments are not counted, so a read is counted once.
PRIMARY_COUNTS = (
    ("primary", "primary_reads"),
    ("primary mapped", "mapped_reads"),
    ("primary duplicates", "primary_duplicates"),
    ("properly paired", "properly_paired_reads"),
    ("paired in sequencing", "paired_reads"),
)


def primary_counts(data: dict) -> dict:
    """A flagstat's primary-read counts (PRIMARY_COUNTS), None for a key this samtools does not write."""
    passed, failed = data.get("QC-passed reads") or {}, data.get("QC-failed reads") or {}
    out = {}
    for key, name in PRIMARY_COUNTS:
        if key not in passed and key not in failed:
            out[name] = None
        else:
            out[name] = int(passed.get(key) or 0) + int(failed.get(key) or 0)
    return out


def recorded_flagstat(bam: Path) -> Optional[dict]:
    """The flagstat the task wrote beside this BAM, or None when there is none, or it is older than the BAM."""
    sidecar = Path(str(bam) + FLAGSTAT_SIDECAR)
    try:
        if sidecar.stat().st_mtime < Path(bam).stat().st_mtime or not sidecar.stat().st_size:
            return None
        return json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None


def run_flagstat(bam: Path, samtools: str, threads: int = 1) -> dict:
    """One threaded ``samtools flagstat`` pass, as JSON."""
    proc = subprocess.run([samtools, "flagstat", "-@", str(max(1, threads)), "-O", "json", str(bam)],
                          check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def recorded_counts(bam: Path) -> Optional[tuple[int, int]]:
    """(reads, duplicates) the task wrote for this BAM, or None when there are none for this BAM."""
    data = recorded_flagstat(bam)
    return None if data is None else _from_flagstat(data)


def counted(bam: Path, samtools: str, threads: int = 1) -> tuple[int, int]:
    """(reads, duplicates): one ``samtools flagstat`` pass; every record, as ``view -c`` and ``-f 1024``."""
    return _from_flagstat(run_flagstat(bam, samtools, threads))


def available_cpus() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def in_parallel(items: Iterable[T], work: Callable[[T], R]) -> list[R]:
    """``work`` over ``items`` in order, as many at once as there are CPUs."""
    items = list(items)
    workers = max(1, min(len(items), available_cpus()))
    if workers == 1:
        return [work(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, items))
