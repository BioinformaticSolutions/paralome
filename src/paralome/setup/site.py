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

"""What the MACHINE has decided, as opposed to what a user prefers.

A fact about the machine has to be recorded on the machine. Where the machine
keeps its genomes must be known to every account, including one created next
month with a fresh config -- otherwise the natural thing to do is build a
private copy somewhere in a home directory, which is the duplication the
whole shared arrangement exists to prevent.

WHERE. Beside the registry, under the install prefix:

    <prefix>/share/paralome/site.json      what this machine decided
    <prefix>/share/paralome/genomes.d/     the registry, by default

``<prefix>/share/paralome/`` is where Paralome keeps machine-level state, so
there is no further location to document and no further discovery rule. The
prefix is derived from ``sys.executable`` rather than written down -- see
``binaries.install_root`` -- so a virtualenv looks for its own file and finds
none, which keeps development environments independent of whatever the system
install decided.

AUTHORITATIVE WHEN PRESENT. When this file exists, the environment variables
and the per-user config keys are ignored for the paths it names: site.json
wins outright. The machine has decided; a user disagreeing with it is the
isolation this is built to stop.

The file is root-owned in a root-owned directory, so an unprivileged
``setup site`` fails at the ``open``. The enforcement is the filesystem's,
not this module's -- which is the right place for it.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

#: The machine-level file, inside the directory that already holds the registry.
SITE_CONFIG_FILENAME = "site.json"

#: Keys. ``root`` is the one an administrator sets; the other two are derived
#: from it unless they are recorded explicitly, which a site with an unusual
#: layout may need.
ROOT_KEY = "shared_root"
REPOSITORY_KEY = "genome_repository"
REGISTRY_KEY = "site_registry"
#: The adapter FASTA every project on this machine trims against.
#:
#: The adapter file should be a default and not a flag retyped per project --
#: and its path is a fact about the machine, not about Paralome, so it is
#: recorded here rather than compiled in. Unset, fastp falls
#: back to auto-detecting the adapter from overrepresented end k-mers, which
#: works but is a guess.
ADAPTER_FASTA_KEY = "adapter_fasta"

#: Subdirectory names under the shared root. SIBLINGS, not nested: a registry inside the genome folder is
#: one `chown -R` away from losing its mode, and people run recursive commands on genome repositories.
REPOSITORY_DIRNAME = "genomes"
REGISTRY_DIRNAME = "registry"


def site_directory() -> Path:
    """The machine-level Paralome directory under the install prefix."""
    from paralome.binaries import install_root
    from paralome.config.user import DEFAULT_CONFIG_DIRNAME

    return install_root() / "share" / DEFAULT_CONFIG_DIRNAME.lstrip(".")


def site_config_path() -> Path:
    """Where this machine's decisions are recorded. May not exist."""
    return site_directory() / SITE_CONFIG_FILENAME


#: Every key this file may hold.
#: Where jobs put temporary files (an absolute path, "TMPDIR" or "none"; paralome.scratch).
SCRATCH_KEY = "scratch"
KNOWN_KEYS = (ROOT_KEY, REPOSITORY_KEY, REGISTRY_KEY, ADAPTER_FASTA_KEY, SCRATCH_KEY)

#: How many callers are, right now, only reporting on this machine's setup.
_reporting = 0


@contextlib.contextmanager
def reporting_on_the_site_file() -> Iterator[None]:
    """Read the site file as it is, without refusing it, for a report.

    One key nothing reads in site.json refuses every run for every
    account, which is the point -- but it also stopped `setup doctor`, `setup
    where` (which Evidome parses) and `setup site`, the three commands somebody
    turns to when the machine is refused, and ordinary users cannot edit that
    root-owned file. Inside this block the file loads whatever it holds, and
    the caller reports site_config_problem() as a finding.
    """
    global _reporting
    _reporting += 1
    try:
        yield
    finally:
        _reporting -= 1


def _read_site_file() -> Tuple[Dict[str, Any], str, str]:
    """(what the file holds, what is wrong with it, how to put it right).

    Never raises. An absent or unreadable file is a machine nobody has set
    up: an empty mapping and no problem.
    """
    path = site_config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}, "", ""
    owner = "it belongs to this machine's administrator"
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return {}, f"{path} is not valid JSON: {exc}.", f"Correct the JSON in that file; {owner}"
    if not isinstance(payload, dict):
        return {}, f"{path} holds JSON that is not an object.", f"Correct that file; {owner}"
    for key in payload:
        if key not in KNOWN_KEYS:
            import difflib

            close = difflib.get_close_matches(str(key), KNOWN_KEYS, n=1, cutoff=0.5)
            return (
                payload,
                f"{path}: unknown key {str(key)!r}."
                + (f" Did you mean {close[0]!r}?" if close else "")
                + f" Nothing reads that key. Valid here: {', '.join(sorted(KNOWN_KEYS))}.",
                f"Correct the key in that file; {owner}",
            )
    return payload, "", ""


def site_config_error() -> Optional[Exception]:
    """The refusal a run gets for this machine's site file, or None."""
    payload, problem, fix = _read_site_file()
    if not problem:
        return None
    from paralome.errors import ParalomeError

    # A key nothing reads is PLM-0411. A file that is not a JSON object is
    # Too: it was read as "no site file", and every account then
    # went by its own settings on a machine that had decided otherwise.
    return ParalomeError(411 if payload else 412, problem, fix=fix)


def site_config_problem() -> str:
    """The same refusal as one line for a report, or an empty string."""
    error = site_config_error()
    return str(error) if error is not None else ""


def load_site_config() -> Dict[str, Any]:
    """The machine's decisions, or an empty mapping.

    Never raises for a file that is absent or unreadable. A machine with no
    site config is the normal case -- a workstation, a laptop, any install
    nobody has run ``setup site`` on.

    It does raise for a key nothing reads, and for a file that is
    not a JSON object. A mistyped key -- `adaptor_fasta` -- was ignored and
    left every account on the machine trimming on fastp's guess with nothing
    to say why. Not inside reporting_on_the_site_file().
    """
    payload, problem, _fix = _read_site_file()
    if problem and not _reporting:
        raise site_config_error()
    return payload


def _recorded(key: str, fallback: Optional[str] = None) -> Optional[Path]:
    site = load_site_config()
    value = site.get(key)
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser().resolve()
    if fallback is None:
        return None
    root = site.get(ROOT_KEY)
    if isinstance(root, str) and root.strip():
        return (Path(root).expanduser() / fallback).resolve()
    return None


def shared_root() -> Optional[Path]:
    """The one directory an administrator names. Everything else derives."""
    return _recorded(ROOT_KEY)


def site_genome_repository() -> Optional[Path]:
    """Where this machine keeps genome folders, if it has decided."""
    return _recorded(REPOSITORY_KEY, REPOSITORY_DIRNAME)


def site_registry() -> Optional[Path]:
    """Where this machine keeps the drop-in registry, if it has decided."""
    return _recorded(REGISTRY_KEY, REGISTRY_DIRNAME)


def site_adapter_fasta() -> Optional[Path]:
    """The adapter FASTA this machine trims against, if it has decided.

    No fallback under the shared root: an adapter file is a sequencing-kit
    fact, not a Paralome directory, and guessing at one would be worse than
    letting fastp auto-detect.
    """
    return _recorded(ADAPTER_FASTA_KEY)


def has_site_config() -> bool:
    """Whether this machine has been set up with ``paralome setup site``."""
    return bool(load_site_config())


def describe_source(kind: str) -> str:
    """Where a path came from, for a report that has to be trusted.

    A user looking at a registry path they did not expect needs to know which
    of four places decided it, and the answer is not guessable from the value.
    """
    if kind not in ("registry", "repository"):
        raise ValueError(f"Unknown path kind: {kind}")
    if kind == "registry":
        if site_registry() is not None:
            return "this machine's site.json"
        if os.environ.get("PARALOME_SITE_REGISTRY", "").strip():
            return "PARALOME_SITE_REGISTRY"
        return "your config, or the built-in default"
    if site_genome_repository() is not None:
        return "this machine's site.json"
    if os.environ.get("PARALOME_GENOME_REPOSITORY", "").strip():
        return "PARALOME_GENOME_REPOSITORY"
    return "your config"


def save_site_config(data: Dict[str, Any], *, path: Optional[Path] = None) -> Path:
    """Write the machine's decisions. Root, in practice.

    The caller is responsible for being root; this just writes, and the
    filesystem refuses if they are not. Written 0644 so every account can read
    it -- the whole point is that a user who has never seen this machine
    before can find out where its genomes live.
    """
    target = Path(path) if path is not None else site_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(target, 0o644)
        os.chmod(target.parent, 0o755)
    except OSError:  # pragma: no cover - a mode we cannot set is not fatal here
        pass
    return target


# ---------------------------------------------------------------------------
# `paralome setup where` -- what this machine has resolved, for other programs
# ---------------------------------------------------------------------------

#: Bumped only when a key is removed or changes meaning; adding keys is not a
#: schema change. Other programs (Evidome first) parse this.
WHERE_SCHEMA = "paralome.where.v1"


def _scratch_for_report() -> Optional[str]:
    from paralome.scratch import configured_setting

    return configured_setting()


def where_report() -> Dict[str, Any]:
    """Everything this machine has resolved about Paralome's shared locations.

    For Evidome, Paralome's sister program, which needs the genome registry
    and the communal root and asks Paralome rather than re-deriving them.
    Paralome resolves the registry through four steps -- site.json,
    PARALOME_SITE_REGISTRY, the user's config, the built-in default -- and a
    second program re-deriving that would be a second implementation of one
    contract, free to disagree with this one the first time either changed.

    Nothing is created, written or checked for permission: this reports. A
    machine that has not run ``setup site`` says so (``site_configured``) --
    the caller decides what that means for it.
    """
    from paralome.binaries import install_root
    from paralome.setup.genomes import (
        config_for_a_report, genome_repository, site_registry_path,
    )
    from paralome.version import __version__

    # This reports, also beside a configuration or a site file
    # that is refused; the site file's fault is in the answer.
    with reporting_on_the_site_file():
        configured = has_site_config()
        root = shared_root()
        config = config_for_a_report()
        registry = site_registry_path(config)
        repository = genome_repository(config)
        sources = {
            "repository": describe_source("repository") if repository is not None else None,
            "registry": describe_source("registry"),
        }
        problem = site_config_problem()
    return {
        "schema": WHERE_SCHEMA,
        "paralome_version": __version__,
        "install_root": str(install_root()),
        "site_config": str(site_config_path()),
        # Where jobs put temporary files (None: not set up, shared storage as before).
        "scratch": _scratch_for_report(),
        "site_configured": configured,
        "shared_root": str(root) if root is not None else None,
        "genome_repository": str(repository) if repository is not None else None,
        "genome_repository_source": sources["repository"],
        "genome_registry": str(registry),
        "genome_registry_source": sources["registry"],
        "genome_registry_exists": registry.is_dir(),
        # None when the site file is absent or sound. Otherwise why every run
        # on this machine is refused.
        "site_config_problem": problem or None,
    }


def shared_group_membership(
    *,
    root: Optional[Path] = None,
    groups: Optional[Iterable[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Whether the running account is in the shared root's group.

    An account outside the group can read the shared root but cannot add to
    it. Every run checks this first, with its other missing pieces, and
    `setup doctor` reports it.

    The group is the shared GENOME REPOSITORY's (made 3775, setgid, by the
    administrator), not the shared root's (often root, which no account is
    in). The Evidome reference beside it carries the same group.

    None when this machine has no shared repository, or it is not present
    here: there is no group to be in. ``root`` and ``groups`` are for tests,
    which may run as root.
    """
    import grp

    if root is None:
        with reporting_on_the_site_file():
            root = site_genome_repository()
    if root is None:
        return None
    try:
        gid = os.stat(root).st_gid
    except OSError:
        return None
    if groups is None:
        groups = set(os.getgroups()) | {os.getegid()}
    try:
        name = grp.getgrgid(gid).gr_name
    except (KeyError, OverflowError):
        name = str(gid)
    return {
        "shared_root": str(root),
        "group": name,
        "gid": gid,
        "member": gid in set(groups),
    }


def not_in_shared_group_text(membership: Dict[str, Any]) -> str:
    """One sentence, the same in the preflight and in doctor."""
    return (
        f"this account is not in group '{membership['group']}' "
        f"(gid {membership['gid']}), which owns the shared genome repository "
        f"{membership['shared_root']}; ask the administrator to add it "
        f"(usermod -aG {membership['group']} <account>), then log in again"
    )
