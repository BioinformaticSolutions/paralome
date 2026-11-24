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

"""The one place that answers "how do I invoke Paralome's console scripts?".

The package has two console scripts (``paralome`` and ``paralome-worker``),
and every site that puts a worker command into argv or a generated script asks
here. It resolves the installed console script, and falls back to running the
module through the current interpreter (``sys.executable -m
paralome.worker``), so a ``PYTHONPATH``-only deployment (what ``bin/paralome``
sets up, with no worker launcher) works as well as an installed one.

Resolution happens on the submitting host at script-generation time; compute
nodes share the filesystem, so the absolute path stays valid inside the job, and
it does not depend on the job's ``PATH`` being set up the same way as the login
shell's.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import List, Optional

#: The machine-facing console script. Invoked by the pipeline supervisor, by
#: ``pipeline/orchestrator.py``, and by every generated sbatch body.
WORKER_SCRIPT = "paralome-worker"

#: The human-facing console script. Generated scripts reach for this only when
#: the thing being run is something a person would also type -- today that is
#: ``batch``'s per-family analysis command and nothing else.
HUMAN_SCRIPT = "paralome"

_WORKER_MODULE = "paralome.worker"
_HUMAN_MODULE = "paralome.cli"


def _resolve(script: str, module: str) -> List[str]:
    # THE SCRIPT BESIDE THE RUNNING PROGRAM COMES FIRST.
    #
    # A machine can hold several Paralome versions at once, each installed into
    # its own prefix and reached through a module that puts that prefix's bin/
    # first on PATH. Generated sbatch bodies name the worker by absolute path,
    # so the one chosen here is the one every job runs: it must belong to the
    # install that is running now, never to another version.
    #
    # The console script that started this process knows which install it is:
    # sys.argv[0] is its path (the shell passes the full path it found), and
    # pip writes all of an install's scripts into the same bin/. So the sibling
    # of argv[0] is looked at first.
    #
    # Then the sibling of the interpreter: a virtualenv keeps its
    # scripts beside its python. NOT .resolve(): a virtualenv's
    # bin/python3 is a symlink to the system interpreter, and resolving it jumps
    # out of the environment being identified.
    #
    # A prefix install runs on the system interpreter, whose sibling bin/ may
    # hold a different, system-wide Paralome -- which is why argv[0] comes
    # before it. Then PATH, where the loaded module's bin/ comes first; then the
    # module itself.
    for here in _launch_dirs():
        sibling = here / script
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return [str(sibling)]
    executable = shutil.which(script)
    if executable:
        return [str(Path(executable).resolve())]
    return [sys.executable, "-m", module]


def _launch_dirs() -> List[Path]:
    """Where this install's console scripts are: beside the one that started the process, then beside the
    interpreter."""
    dirs: List[Path] = []
    launched = sys.argv[0] if sys.argv else ""
    if launched and Path(launched).name.startswith(HUMAN_SCRIPT):
        dirs.append(Path(launched).absolute().parent)
    dirs.append(Path(sys.executable).parent)
    return dirs


def install_root() -> Path:
    """The directory this Paralome is installed under.

    Discovered rather than recorded: no install path is hardcoded in src/ -- a package that knows its own location cannot be
    installed twice, and every version is installed into a prefix of its own.

    NOT .resolve(). An install is sometimes a virtualenv, and a
    virtualenv's bin/python3 is a symlink to the system interpreter -- so
    `Path(sys.executable).resolve().parent.parent` returned `/usr` rather than
    the venv. `paralome --version` would then name the system install while
    running the venv's code: the one command whose whole purpose is answering
    "which install is this" pointing at the wrong one.
    """
    return Path(sys.executable).parent.parent


def is_development(version: Optional[str] = None) -> bool:
    """True when ``version`` is a pre-release of the next production release.

    Takes the string rather than importing it, because the caller may have
    resolved it from INSTALLED distribution metadata while the imported module
    says something else -- a stale install running against a source checkout is
    exactly the confusion this function exists to clear up, so it must not be
    the thing that reintroduces it.
    """
    if version is None:
        from paralome.version import __version__ as version

    return ".dev" in str(version)


def worker_argv() -> List[str]:
    """argv prefix that reaches ``paralome-worker``."""
    return _resolve(WORKER_SCRIPT, _WORKER_MODULE)


def human_argv() -> List[str]:
    """argv prefix that reaches the human ``paralome`` command."""
    return _resolve(HUMAN_SCRIPT, _HUMAN_MODULE)


def worker_command() -> str:
    """``worker_argv()`` as one shell-quoted token sequence, for sbatch bodies."""
    return shell_join(worker_argv())


def shell_join(argv) -> str:
    """Quote and join an argv for embedding in a generated shell script.

    This existed four times over as a private ``_shell_join``
    in ``atac/peak_merge.py``, ``atac/motifs.py``, ``atac/quantification.py`` and
    ``atac/statistics.py`` -- character-identical in every copy, sitting directly
    beneath four character-identical copies of the worker resolver above. Both
    belong to the same job: turning "run this Paralome command" into a line of
    bash.
    """
    return " ".join(shlex.quote(str(value)) for value in argv)


# ---------------------------------------------------------------------------
# Per-user installs on a machine that already has a shared one
# ---------------------------------------------------------------------------


def package_root() -> Path:
    """The directory this package's code actually lives in.

    Distinct from ``install_root()``, which is derived from the interpreter and
    therefore says ``/usr`` for a ``--user`` install -- the two disagree in
    exactly the case below, which is why both exist.
    """
    return Path(__file__).resolve().parent


def user_site_install() -> bool:
    """Whether this Paralome lives in the calling account's private packages.

    NOT "was this installed without root". A virtualenv is also not root, and
    development installs are often one -- ``install_root()`` was corrected in
    For that reason. A venv disables user-site by default and its
    packages are under its own prefix, so testing for USER SITE specifically
    leaves every legitimate unprivileged arrangement alone and catches only
    ``pip install --user``.
    """
    import site

    if not getattr(site, "ENABLE_USER_SITE", False):
        return False
    try:
        user_site = site.getusersitepackages()
    except Exception:  # pragma: no cover - platforms without a user site
        return False
    if not user_site:
        return False
    candidates = [user_site] if isinstance(user_site, str) else list(user_site)
    here = package_root()
    for candidate in candidates:
        try:
            resolved = Path(candidate).resolve()
        except OSError:  # pragma: no cover
            continue
        if resolved in here.parents:
            return True
    return False


def refuse_private_install_on_a_shared_machine() -> None:
    """Stop a ``pip install --user`` from quietly islanding somebody.

    0.4. Individual per-account installations are not allowed: they lead to
    isolation and genome redundancy, which shared installs exist to avoid.

    A WHEEL CANNOT REFUSE AT INSTALL TIME. Wheels are unpacked, not executed --
    that is the point of the format -- so there is no hook. And doing nothing
    does not produce a failure: a ``--user`` install works for everything
    except publishing, so the account can register its own genomes, build its
    own copy of a reference somebody already built, and run every pipeline. The
    outcome of silence is not a user who gives up; it is a user who succeeds
    privately, which is the thing being prevented. So the refusal has to be
    here, at run time.

    EVERY MACHINE, not only the ones that already share. An account that is not
    the admin on the computer it installs on fails out. The
    first version of this required the machine to be shared or to already have
    a registry, which left the case that actually matters uncovered -- the
    account that installs privately on a machine BEFORE anyone sets up sharing
    is the one that then has a private genome nobody can see.

    The test is user-site, not "did this need root". A virtualenv is also not
    root, and development installs are often one; venvs disable user-site and
    live under their own prefix, so they are untouched. What is caught is
    `pip install --user` and nothing else.
    """
    from paralome.errors import ParalomeError

    if not user_site_install():
        return
    raise ParalomeError(
        408,
        f"Installed under {package_root().parent}, which is your home directory "
        f"rather than the machine.",
    )


# ---------------------------------------------------------------------------
# Where the external tools are
# ---------------------------------------------------------------------------


def _recorded_executables() -> dict:
    """What `paralome setup` wrote down, or an empty mapping.

    A machine with no config must fall back to PATH rather than take every
    run down with it -- the recording is an improvement on PATH, not a
    precondition for running at all. It raises for a config that is refused
    by number: a key nothing reads, or a file that is not JSON.
    """
    from paralome.errors import ParalomeError

    try:
        from paralome.config.user import load_config

        executables = load_config(initialize=False).get("executables")
    except ParalomeError:
        # A key nothing reads, or a file that is not JSON, is
        # refused by name, here as everywhere. Swallowed, a standalone stage
        # command took its tools from PATH and said nothing about the
        # configuration it had just set aside.
        raise
    except Exception:  # pragma: no cover - a config we cannot read is not fatal
        return {}
    return executables if isinstance(executables, dict) else {}


def tool_path(name: str) -> Optional[str]:
    """The absolute path this machine recorded for a tool, or None.

    `paralome setup` records the absolute path of every external tool it
    finds, and every stage invokes the recorded path, so the tool a run uses
    does not depend on the order of directories in one account's PATH (on a
    shared cluster, two accounts could otherwise get different builds of STAR
    or FreeBayes from the same command, and nothing would record it).

    A RECORDED PATH THAT NO LONGER RESOLVES REFUSES, rather than falling back
    to PATH with a warning: a path that does not resolve stops the run at the
    start rather than surfacing later as a number that looks fine.

    Returns None when nothing was recorded -- an unrecorded tool is not an
    error, it is a machine whose setup predates the recording or never ran.
    Callers fall back to PATH there.
    """
    entry = _recorded_executables().get(name)
    if not isinstance(entry, dict):
        return None
    recorded = entry.get("path")
    if not isinstance(recorded, str) or not recorded.strip():
        return None

    candidate = Path(recorded).expanduser()
    if os.access(str(candidate), os.X_OK) and candidate.is_file():
        return str(candidate)

    from paralome.errors import ParalomeError

    raise ParalomeError(
        410,
        f"{name} was recorded at {candidate}\n"
        f"That path is {'not executable by you' if candidate.exists() else 'gone'}.",
    )


def relocate_tool(name: str) -> Optional[tuple]:
    """Find a tool whose recorded path is gone on this account's PATH, and record where it is now.

    Tools do not stay in one place: an administrator may move them, and a user
    will not know. So a tool whose recorded path is gone but which is on the
    account's PATH is flagged and re-recorded rather than stopping the run.

    Called by the preflight, on the login node, before anything is submitted. Returns (old path, new path)
    when the tool was found and re-recorded -- with its version, as `paralome setup` records it -- so every
    stage of the run uses the new path; None when it is not on PATH either, which stays a fault.
    """
    found = shutil.which(name)
    if not found:
        return None
    from paralome.config.user import load_config, save_config
    from paralome.setup.doctor import discover_executables

    config = load_config(initialize=False)
    executables = config.setdefault("executables", {})
    old = (executables.get(name) or {}).get("path")
    entry = discover_executables().get(name) or {}
    if entry.get("path") != found:
        entry = {"path": found, "found": True}
    executables[name] = entry
    save_config(config)
    return old, found


def resolve_tool(name: str) -> Optional[str]:
    """The tool to invoke for ``name``: the recorded path, else PATH, else None.

    The one function every stage should use. Raises PLM-0410 when a recorded
    path is stale, returns None when the tool cannot be found at all -- which
    each caller reports in its own words, because "STAR is not installed" and
    "tabix is not installed" need different advice.
    """
    recorded = tool_path(name)
    if recorded:
        return recorded
    return shutil.which(name)
