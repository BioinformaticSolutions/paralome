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

from dataclasses import dataclass, replace
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable, Optional, Sequence, Union

from paralome.errors import ParalomeError
from paralome.pipeline.resources import max_cpus_per_node, max_tmp_disk_mb


@dataclass(frozen=True)
class SlurmOptions:
    job_name: str
    cpus_per_task: int = 1
    mem: str = "8G"
    partition: Optional[str] = None
    node: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    output: Optional[Path] = None
    error: Optional[Path] = None
    array: Optional[str] = None
    #: Node scratch the job needs (sbatch --tmp, e.g. "2048M");
    #: built by tmp_request, None for a job that writes none.
    tmp: Optional[str] = None
    extra_directives: tuple[str, ...] = ()

    def header(self) -> str:
        # Values that can contain a space (job names, log paths built from
        # labels such as "Group A") are shell-quoted: sbatch tokenizes
        # #SBATCH lines on whitespace, so an unquoted space would split the
        # path and the job would fail before writing any log.
        def q(value: object) -> str:
            return shlex.quote(str(value))

        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={q(self.job_name)}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={q(self.mem)}",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={q(self.partition)}")
        if self.node:
            lines.append(f"#SBATCH --nodelist={q(self.node)}")
        if self.account:
            lines.append(f"#SBATCH --account={q(self.account)}")
        if self.qos:
            lines.append(f"#SBATCH --qos={q(self.qos)}")
        if self.array:
            lines.append(f"#SBATCH --array={self.array}")
        if self.tmp:
            lines.append(f"#SBATCH --tmp={q(self.tmp)}")
        if self.output:
            lines.append(f"#SBATCH --output={q(self.output)}")
        if self.error:
            lines.append(f"#SBATCH --error={q(self.error)}")
        lines.extend(self.extra_directives)
        lines.extend(["", "set -euo pipefail", ""])
        return "\n".join(str(x) for x in lines)


def compress_array_ids(ids: Iterable[int], throttle: Optional[int] = None) -> str:
    values = sorted(set(int(x) for x in ids))
    if not values:
        raise ValueError("Cannot build SLURM array expression from an empty ID list")
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    expr = ",".join(ranges)
    if throttle:
        expr = f"{expr}%{throttle}"
    return expr


def tmp_request(need_bytes: int, partition: Optional[str] = None) -> Optional[str]:
    """The sbatch ``--tmp`` value for a job that writes ``need_bytes`` to node scratch, or None.

    ``paralome.scratch`` says SLURM schedules by what jobs ask for,
    not what they write; jobs that used scratch asked for nothing, so the
    scheduler put as many on a node as its CPUs and memory allowed and the
    scratch disk was never part of the decision. ``need_bytes`` is the same
    estimate the job's own space guard uses (for an array, the largest task's).

    None -- no directive -- when scratch is not set up here (``none`` or no
    setting: the job writes to shared storage) or the job needs nothing, and
    when sinfo says no node in the partition defines TmpDisk: such a request
    could never be placed, and there the scratch ledger does the accounting
    (see ``paralome.scratch``). A need above the largest TmpDisk is asked as
    that TmpDisk, which some node can satisfy; the job's space guard still
    falls back to shared storage if the disk cannot hold it. When sinfo cannot
    say, the need is asked as it is.
    """
    from paralome.scratch import NO_SCRATCH, configured_setting

    setting = configured_setting()
    if not setting or setting == NO_SCRATCH:
        return None
    try:
        need = int(need_bytes)
    except (TypeError, ValueError):
        return None
    if need <= 0:
        return None
    mb = -(-need // (1024 * 1024))
    largest = max_tmp_disk_mb(partition)
    if largest is not None and largest <= 0:
        return None
    if largest is not None and mb > largest:
        print(
            f"Note: a job needs about {mb:,} MB of node scratch and the largest TmpDisk "
            f"is {largest:,} MB; requesting {largest:,} MB, which the scheduler can place.",
            file=sys.stderr, flush=True,
        )
        mb = int(largest)
    return f"{mb}M"


def tmp_directive(need_bytes: int, partition: Optional[str] = None) -> str:
    """``#SBATCH --tmp=...`` and a newline for a hand-written header, or "" (see tmp_request)."""
    value = tmp_request(need_bytes, partition)
    return f"#SBATCH --tmp={value}\n" if value else ""


#: The most of the largest node one job may ask for. A job asking
#: for the whole node waits until the node is EMPTY, and a node running
#: anything at all -- a one-CPU pipeline supervisor is always somewhere --
#: never is: the ATAC motif finalizer and the scRNA samples each sat pending
#: on (Resources) beside a 1-CPU supervisor. A quarter stays free.
ONE_JOB_NODE_SHARE = 0.75


def one_job_cpu_ceiling(partition: Optional[str] = None) -> Optional[int]:
    """The most CPUs one job is given: ONE_JOB_NODE_SHARE of the largest node, or None without sinfo."""
    largest = max_cpus_per_node(partition)
    if not largest:
        return None
    return max(1, int(largest * ONE_JOB_NODE_SHARE))


def placeable_cpus(cpus: int, partition: Optional[str] = None, *, job_name: str = "a job") -> int:
    """``cpus`` brought down to one_job_cpu_ceiling, for a header written by hand."""
    return placeable_options(SlurmOptions(job_name=job_name, cpus_per_task=max(1, int(cpus)),
                                          partition=partition)).cpus_per_task


def placeable_options(options: SlurmOptions) -> SlurmOptions:
    """The same request, with a CPU count a node can actually start.

    --threads is a budget for a STAGE, and several stages hand it
    straight to one job's --cpus-per-task. Ask for more CPUs than the largest
    node has and sbatch does not queue the job and wait: it refuses it outright
    ("CPU count per node can not be satisfied"), so the stage simply never
    runs. Clamping here rather than at the call sites means no stage can get it
    wrong by omission, which is the same reason this function's neighbour
    creates the log directories.

    Brought down to ONE_JOB_NODE_SHARE of the largest node, not to
    the whole node. A whole-node request is placeable and still never starts
    while anything else runs there.

    When sinfo cannot say how big a node is, the request is left exactly as it
    was asked for.
    """
    ceiling = one_job_cpu_ceiling(options.partition)
    if not ceiling or options.cpus_per_task <= ceiling:
        return options
    print(
        f"Note: {options.job_name} asked for {options.cpus_per_task} CPUs; requesting {ceiling}, "
        f"{int(ONE_JOB_NODE_SHARE * 100)}% of the largest node ({max_cpus_per_node(options.partition)}), "
        "so it can start beside other jobs. A whole-node request waits for an empty node.",
        file=sys.stderr, flush=True,
    )
    return replace(options, cpus_per_task=int(ceiling))


def write_sbatch(path: Union[str, Path], options: SlurmOptions, body: str) -> Path:
    """Write an sbatch script, creating the directories it will need.

    That includes the directories --output/--error point at. SLURM does not
    create them: if the log directory is missing the job is killed before the
    body runs, with exit code 1 and no output anywhere to say why -- the file
    that would have explained it is the file that could not be created. Doing
    it here means no call site has to, and a new one cannot get it wrong by
    omission.
    """
    path = Path(path)
    options = placeable_options(options)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stream in (options.output, options.error):
        if stream:
            # %j/%a are expanded by SLURM in the FILENAME, never in a directory
            # component, so the parent is always a literal path.
            Path(stream).parent.mkdir(parents=True, exist_ok=True)
    path.write_text(options.header() + body.rstrip() + "\n")
    path.chmod(0o755)
    return path


def parse_sbatch_jobid(output: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    if not match:
        raise ValueError(f"Could not parse sbatch job id from output: {output!r}")
    return match.group(1)


def sbatch(script: Union[str, Path], *, dependency: Optional[str] = None, extra_args: Sequence[str] = ()) -> str:
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.extend(extra_args)
    cmd.append(str(script))
    # SAY WHAT SBATCH SAID. check=True raises CalledProcessError, whose message carries the command and
    # the exit status and NOT the scheduler's own sentence, so the reason is captured and reported: it is
    # the only useful part.
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or "no output"
        raise ParalomeError(
            414,
            f"sbatch refused {script}: {detail}",
            fix=("Run the same sbatch by hand to see the scheduler's answer in full: "
                 f"{' '.join(shlex.quote(part) for part in cmd)}"),
        )
    return parse_sbatch_jobid(result.stdout)


def afterany(jobid: str | int) -> str:
    return f"afterany:{jobid}"


def afterok(jobid: str | int) -> str:
    return f"afterok:{jobid}"


# Single-job SLURM stages are submitted without waiting here; waiting and
# completion detection are done by `pipeline.supervisor._wait_for_jobs`.
