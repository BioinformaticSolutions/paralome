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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Optional, Union


import os
import pwd


def writable_reason(target: Union[str, Path]) -> Optional[str]:
    """Why this path cannot be written into, naming owner, mode and user.

    Root created a project tree under a new user's home and never
    chowned it. Paralome died creating its own directory inside, and because
    the SLURM --output path pointed into that same tree, the job's stderr had
    nowhere to go -- the failure erased its own explanation. `write_sbatch`
    already carries a comment calling that "the single least debuggable failure
    this pipeline can produce"; it was guarded there and nowhere else.

    Walks up to the nearest existing ancestor, because the target itself
    usually does not exist yet -- that is the whole point of creating it.
    """
    path = Path(target).expanduser()
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    if os.access(probe, os.W_OK | os.X_OK):
        return None
    try:
        info = probe.stat()
        owner = pwd.getpwuid(info.st_uid).pw_name
    except (OSError, KeyError):
        owner, info = "?", None
    mode = format(info.st_mode & 0o777, "04o") if info else "?"
    try:
        me = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        me = str(os.getuid())
    return (
        f"Cannot write into {path}\n"
        f"  nearest existing parent : {probe}\n"
        f"  owner / mode            : {owner} / {mode}\n"
        f"  running as              : {me} (uid {os.getuid()})\n"
        f"Fix with:  chown -R {me} {probe}"
    )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class RunLogger:
    workdir: Path
    verbosity: int = 0
    echo_stderr: bool = False

    @property
    def path(self) -> Path:
        return Path(self.workdir) / "logs" / "pipeline.log"

    def write(self, message: str, level: int = 1, *, stderr: Optional[bool] = None) -> None:
        line = f"{now_iso()} {message}"
        # A logger that raises while reporting a failure converts a
        # diagnosable error into silence. If the log cannot be written, say so
        # once on stderr and carry on -- the message still reaches the user,
        # which is the whole job.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            if not getattr(self, "_log_write_failed", False):
                object.__setattr__(self, "_log_write_failed", True)
                print(
                    f"[paralome] cannot write {self.path}: {exc}. "
                    "Continuing with console output only.",
                    file=sys.stderr, flush=True,
                )
        if self.verbosity >= level:
            print(message, flush=True)
        if stderr if stderr is not None else self.echo_stderr:
            print(message, file=sys.stderr, flush=True)


    def info(self, message: str) -> None:
        self.write(message, level=1)

    def detail(self, message: str) -> None:
        self.write(message, level=2)


    def error(self, message: str) -> None:
        self.write(f"ERROR: {message}", level=0, stderr=True)

    def section(self, title: str, level: int = 1) -> None:
        self.write(title, level=level)
        self.write("-" * len(title), level=level)



    def command(self, command: str, level: int = 2) -> None:
        self.write("Command:", level=level)
        for line in command.splitlines():
            self.write(f"  {line}", level=level)

    def key_values(self, title: str, values: dict[str, Any], level: int = 1) -> None:
        self.section(title, level=level)
        for key, value in values.items():
            self.write(f"{key}: {value}", level=level)
