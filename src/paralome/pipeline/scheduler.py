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

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence


DEFAULT_SLURM_MAX_ARRAY_SIZE = 1001


@dataclass(frozen=True)
class SchedulerCapabilities:
    backend: str
    supports_arrays: bool
    max_array_size: int
    source: str
    max_cpus_per_task: Optional[int] = None

    @property
    def max_array_task_id(self) -> int:
        """The highest array task number this scheduler accepts.

        SLURM's MaxArraySize is one MORE than the highest index it
        allows: with MaxArraySize=1001 the valid indices are 0-1000, and
        ``--array=1-1001`` is rejected with "Invalid job array specification".
        Paralome numbers tasks from 1, so the most tasks one array can hold is
        MaxArraySize - 1.
        """
        size = max(1, int(self.max_array_size))
        return max(1, size - 1) if self.supports_arrays else size

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["max_array_task_id"] = self.max_array_task_id
        return data


@dataclass(frozen=True)
class ArrayBatch:
    batch_index: int
    first_task: int
    last_task: int
    task_count: int
    throttle: int

    @property
    def array_spec(self) -> str:
        return f"{self.first_task}-{self.last_task}%{self.throttle}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["array_spec"] = self.array_spec
        return data


@dataclass(frozen=True)
class ArrayPlan:
    task_count: int
    requested_concurrency: int
    effective_concurrency: int
    max_array_size: int
    batches: tuple[ArrayBatch, ...]

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "requested_concurrency": self.requested_concurrency,
            "effective_concurrency": self.effective_concurrency,
            "max_array_size": self.max_array_size,
            "batch_count": self.batch_count,
            "batches": [batch.to_dict() for batch in self.batches],
        }


def _positive_int(value: object) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_max_array_size(text: str) -> Optional[int]:
    match = re.search(r"(?:^|\s)MaxArraySize\s*=\s*(\d+)", text, flags=re.MULTILINE)
    return _positive_int(match.group(1)) if match else None


def detect_scheduler_capabilities(
    backend: str,
    *,
    max_array_size: Optional[int] = None,
) -> SchedulerCapabilities:
    normalized = str(backend).strip().lower()
    if normalized != "slurm":
        return SchedulerCapabilities(
            backend=normalized or "local",
            supports_arrays=False,
            max_array_size=1,
            source="backend_default",
            max_cpus_per_task=os.cpu_count(),
        )

    explicit = _positive_int(max_array_size)
    if explicit is not None:
        return SchedulerCapabilities(
            backend="slurm",
            supports_arrays=True,
            max_array_size=explicit,
            source="explicit",
        )

    env_value = _positive_int(os.environ.get("PARALOME_SLURM_MAX_ARRAY_SIZE"))
    if env_value is not None:
        return SchedulerCapabilities(
            backend="slurm",
            supports_arrays=True,
            max_array_size=env_value,
            source="environment",
        )

    if shutil.which("scontrol") is not None:
        try:
            proc = subprocess.run(
                ["scontrol", "show", "config"],
                check=True,
                text=True,
                capture_output=True,
            )
            detected = _parse_max_array_size(proc.stdout)
            if detected is not None:
                return SchedulerCapabilities(
                    backend="slurm",
                    supports_arrays=True,
                    max_array_size=detected,
                    source="scontrol",
                )
        except Exception:
            pass

    return SchedulerCapabilities(
        backend="slurm",
        supports_arrays=True,
        max_array_size=DEFAULT_SLURM_MAX_ARRAY_SIZE,
        source="default",
    )


def plan_array_tasks(
    *,
    task_count: int,
    concurrency: int,
    capabilities: SchedulerCapabilities,
) -> ArrayPlan:
    tasks = max(0, int(task_count))
    requested = max(1, int(concurrency))
    # Tasks are numbered from 1, so one array holds max_array_task_id of them
    # (MaxArraySize - 1 under SLURM), not MaxArraySize.
    max_size = capabilities.max_array_task_id
    effective = min(requested, max_size, tasks) if tasks else 0

    batches: list[ArrayBatch] = []
    first = 1
    batch_index = 1
    while first <= tasks:
        last = min(tasks, first + max_size - 1)
        count = last - first + 1
        batches.append(
            ArrayBatch(
                batch_index=batch_index,
                first_task=first,
                last_task=last,
                task_count=count,
                throttle=min(max(1, effective), count),
            )
        )
        first = last + 1
        batch_index += 1

    return ArrayPlan(
        task_count=tasks,
        requested_concurrency=requested,
        effective_concurrency=effective,
        max_array_size=max(1, int(capabilities.max_array_size)),
        batches=tuple(batches),
    )


def require_array_task_ids(
    task_ids: Sequence[int],
    capabilities: SchedulerCapabilities,
    *,
    stage: str,
) -> None:
    """Refuse, before anything is submitted, task numbers the scheduler rejects.

    A stage whose array tasks are numbered beyond the scheduler's
    limit cannot be submitted as written, however the tasks are batched: the
    limit is on the task NUMBER, and a second batch keeps its real numbers.
    Before this check the first batch was submitted and the second was rejected
    by sbatch, leaving a half-submitted stage. Called by the stages that submit
    one array task per pending item under its own number.
    """
    ids = [int(value) for value in task_ids]
    if not ids:
        return
    highest = max(ids)
    limit = capabilities.max_array_task_id
    if highest <= limit:
        return
    from paralome.errors import ParalomeError

    raise ParalomeError(
        307,
        f"{stage} needs array task {highest}, and this scheduler accepts task "
        f"numbers up to {limit} (MaxArraySize={capabilities.max_array_size}, "
        f"from {capabilities.source}).",
        stage=stage,
        highest_task=highest,
        limit=limit,
    )


def format_plan_diagnostics(title: str, rows: Sequence[tuple[str, object]]) -> str:
    width = max((len(label) for label, _ in rows), default=0)
    body = [title, "=" * len(title)]
    body.extend(f"{label.ljust(width)} : {value}" for label, value in rows)
    return "\n".join(body)
