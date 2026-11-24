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

"""Scratch space for a job's temporary files.

WHY. A job's temporary files -- the annotation split and its sort, the
ANNOVAR working files of each shard, the burden reference's per-chunk ANNOVAR
run, MACS3's duplicate-free BAM -- would otherwise go to the project or genome
folder on shared storage, beside the results. On a cluster that is the slowest
disk there is, shared by every job at once.

WHERE. Sites differ, so nothing is assumed: node-local disks are usually at
/scratch, /lscratch, /local/scratch or /local_scratch; large clusters often
mount a shared parallel filesystem at /scratch; many SLURM sites give every job
a private $TMPDIR; small machines have only /tmp. The administrator decides
once (`paralome setup site --scratch PATH|TMPDIR|none`, which reports what this
machine has); a user may override it with PARALOME_SCRATCH (same values).
Unset, temporary files stay beside the results.

ON THE NODE. A job script decides when it runs, on the node it landed on: the
scratch directory must exist there and be writable, else the job says so in its
log and uses the shared location -- a node without the drive is slower, never
broken. Each job works in its own directory,
<scratch>/paralome-<user>-<job>-<task>-<what>.XXXXXX, removed when the job
ends, however it ends. Results never go to scratch.

CONCURRENT JOBS. Array tasks that start together on one node would each see
the same free space and each decide it fitted, so two things account for the
others. Every job that writes to scratch asks SLURM for it (``--tmp``, see
``paralome.pipeline.slurm.tmp_request``), so on nodes that define TmpDisk the
scheduler does the arithmetic. Everywhere else the space guard subtracts what
the jobs already running there have reserved: each job appends what it expects
to write to a ledger in the scratch root (``RESERVATIONS_FILE``), under an
flock, and removes its entry when it exits. A reservation counts only what its
job has not yet written (its directory's size is subtracted), and entries whose
job is gone -- killed without running its exit trap -- are dropped by the next
job that reads the ledger. Where flock cannot be had (no flock command, a
filesystem mounted without lock support, a ledger another user cannot write)
the guard works on free space alone, and says so.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

#: The site.json key, and the environment variable that overrides it.
SCRATCH_KEY = "scratch"
SCRATCH_ENV = "PARALOME_SCRATCH"
#: The analysis a stage runs for (ATAC, DNA, RNA, scRNA, snATAC),
#: set by the pipeline for every stage command it starts. It goes into the name
#: of the job's scratch directory, after "paralome-", so a node's scratch shows
#: which analysis each directory belongs to. A stage command run by itself has
#: none, and its directory is named as before.
PIPELINE_ENV = "PARALOME_PIPELINE"
#: Values that are not a path.
USE_TMPDIR = "TMPDIR"
NO_SCRATCH = "none"

#: Where sites put scratch, in the order a suggestion prefers them.
CANDIDATES = ("/scratch", "/lscratch", "/local/scratch", "/local_scratch", "/scratch/local", "/tmp")
#: Filesystem types that are not on the node itself.
NETWORK_FILESYSTEMS = frozenset({
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "ceph", "cephfs", "glusterfs",
    "fuse.glusterfs", "fuse.sshfs", "panfs", "wekafs", "afs", "9p", "fuse.s3fs", "orangefs",
})
#: A suggestion needs at least this much free space (bytes).
SUGGEST_MIN_FREE = 100 * 1024 ** 3


def configured_setting() -> Optional[str]:
    """PARALOME_SCRATCH, else the site's ``scratch``, else None (not set up: shared storage as before)."""
    value = os.environ.get(SCRATCH_ENV, "").strip()
    if value:
        return value
    try:
        from paralome.setup.site import load_site_config

        recorded = load_site_config().get(SCRATCH_KEY)
    except Exception:  # an unreadable site file is reported by the commands that read it
        recorded = None
    return str(recorded).strip() if isinstance(recorded, str) and recorded.strip() else None


def check_setting(value: str) -> Optional[str]:
    """Why ``value`` cannot be a scratch setting, or None."""
    if value in (USE_TMPDIR, NO_SCRATCH):
        return None
    if not value.startswith("/"):
        return f"--scratch {value!r}: an absolute path, {USE_TMPDIR} or {NO_SCRATCH}"
    return None


def _mounts() -> List[tuple]:
    """(mount point, filesystem type) of every mount, longest mount point first."""
    out = []
    try:
        with open("/proc/mounts", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    out.append((parts[1].replace("\\040", " "), parts[2]))
    except OSError:
        return []
    return sorted(out, key=lambda m: len(m[0]), reverse=True)


def describe(path: str) -> Dict[str, object]:
    """What a candidate scratch directory is: exists, writable, filesystem, local or network, free bytes."""
    p = Path(path)
    info: Dict[str, object] = {"path": str(p), "exists": p.is_dir()}
    if not p.is_dir():
        return info
    real = os.path.realpath(p)
    mount, fstype = next(((m, t) for m, t in _mounts() if real == m or real.startswith(m.rstrip("/") + "/")),
                         ("?", "?"))
    try:
        free = shutil.disk_usage(real).free
    except OSError:
        free = None
    info.update({
        "writable": os.access(real, os.W_OK | os.X_OK),
        "mount": mount,
        "filesystem": fstype,
        "network": fstype in NETWORK_FILESYSTEMS,
        "free_bytes": free,
    })
    return info


def detect(candidates=CANDIDATES) -> List[Dict[str, object]]:
    """Every candidate that exists on this machine, described."""
    return [describe(c) for c in candidates if Path(c).is_dir()]


def suggest(found: List[Dict[str, object]]) -> Optional[str]:
    """The first local, writable candidate with room that is not /tmp; None when there is none."""
    for info in found:
        if (info.get("writable") and not info.get("network") and info.get("path") != "/tmp"
                and (info.get("free_bytes") or 0) >= SUGGEST_MIN_FREE):
            return str(info["path"])
    return None


#: Scratch is used only while this share of its filesystem stays free after the job's estimated need.
KEEP_FREE_PERCENT = 10

#: The reservations of the jobs using a scratch directory, kept in
#: its root. One line per job: host, PID, bytes reserved, the job's directory.
#: Removed by the last job to leave it.
RESERVATIONS_FILE = ".paralome-scratch-reservations"
#: How long a job waits for the ledger's lock before going on without it.
RESERVATION_LOCK_SECONDS = 300


#: One EXIT trap runs a list of cleanups that each caller adds to with
#: ``paralome_on_exit CMD ARGS...`` (newest first, then any trap set before it).
#: Rebuilding trap text from ``trap -p`` left bash's quote escapes in place, so a
#: second chained trap could not be parsed at exit and the job failed with 2.
EXIT_CLEANUP_BASH = "\n".join([
    "if ! declare -F paralome_on_exit >/dev/null; then",
    "  PARALOME_ON_EXIT=()",
    "  PARALOME_EXIT_EARLIER=''",
    "  paralome_exit_earlier() { if [[ -n \"$1\" ]]; then eval \"set -- $1\"; PARALOME_EXIT_EARLIER=${3:-}; fi; }",
    "  paralome_exit_earlier \"$(trap -p EXIT)\"",
    "  paralome_run_on_exit() {",
    "    local i",
    "    for (( i=${#PARALOME_ON_EXIT[@]}-1; i>=0; i-- )); do eval \"${PARALOME_ON_EXIT[i]}\" || true; done",
    "    if [[ -n \"$PARALOME_EXIT_EARLIER\" ]]; then eval \"$PARALOME_EXIT_EARLIER\"; fi",
    "  }",
    "  paralome_on_exit() { PARALOME_ON_EXIT+=(\"$(printf '%q ' \"$@\")\"); }",
    "  trap paralome_run_on_exit EXIT",
    "fi",
])


#: The reservation ledger, in bash (see the module docstring).
#: Defined once per script however many scratch directories it makes.
#:   paralome_scratch_lock LEDGER     open and flock it; 1 when that cannot be done.
#:     A ledger removed by another job while this one waited is not the one at
#:     the path any more; the lock is taken again on the file that is there.
#:   paralome_scratch_reserved LEDGER sets PARALOME_SCRATCH_RESERVED (bytes the
#:     live entries have yet to write) and keeps only the live entries.
#:   paralome_scratch_release LEDGER DIR  drops DIR's entry (and dead ones);
#:     the ledger goes when nothing is left in it.
RESERVATION_BASH = "\n".join([
    "if ! declare -F paralome_scratch_lock >/dev/null; then",
    "  PARALOME_SCRATCH_LOCK_FD=''",
    "  paralome_scratch_lock() {",
    "    local tries=0",
    "    command -v flock >/dev/null 2>&1 || return 1",
    "    while (( tries < 10 )); do",
    "      exec {PARALOME_SCRATCH_LOCK_FD}>>\"$1\" || return 1",
    "      chmod a+rw \"$1\" 2>/dev/null || true",
    f"      if ! flock -w {int(RESERVATION_LOCK_SECONDS)} \"$PARALOME_SCRATCH_LOCK_FD\"; then",
    "        exec {PARALOME_SCRATCH_LOCK_FD}>&-; return 1",
    "      fi",
    "      if [[ \"$1\" -ef \"/dev/fd/$PARALOME_SCRATCH_LOCK_FD\" ]]; then return 0; fi",
    "      exec {PARALOME_SCRATCH_LOCK_FD}>&-",
    "      tries=$(( tries + 1 ))",
    "    done",
    "    return 1",
    "  }",
    "  paralome_scratch_unlock() { exec {PARALOME_SCRATCH_LOCK_FD}>&-; }",
    "  paralome_scratch_reserved() {",
    "    local host pid bytes dir used me",
    "    me=$(uname -n)",
    "    PARALOME_SCRATCH_RESERVED=0",
    "    PARALOME_SCRATCH_LIVE=()",
    "    PARALOME_SCRATCH_LIVE_N=0",
    "    while read -r host pid bytes dir; do",
    "      if [[ ! \"$bytes\" =~ ^[0-9]+$ || ! \"$pid\" =~ ^[0-9]+$ || -z \"$dir\" ]]; then continue; fi",
    "      if [[ \"$host\" == \"$me\" && ! -d \"/proc/$pid\" ]]; then continue; fi",
    "      if [[ ! -d \"$dir\" ]]; then continue; fi",
    "      if [[ -n \"${2:-}\" && \"$dir\" == \"$2\" ]]; then continue; fi",
    # Another job may remove its directory between the -d test above and du, and under the task script's
    # set -euo pipefail du's failure would end the whole task with nothing on stderr. A directory that went
    # is a reservation that ended: measured first, kept only if it is still there.
    "      used=$(du -sb -- \"$dir\" 2>/dev/null | awk '{print $1}') || used=''",
    "      if [[ ! -d \"$dir\" ]]; then continue; fi",
    "      used=${used:-0}",
    "      PARALOME_SCRATCH_LIVE+=(\"$host $pid $bytes $dir\")",
    "      PARALOME_SCRATCH_LIVE_N=$(( PARALOME_SCRATCH_LIVE_N + 1 ))",
    "      if (( bytes > used )); then PARALOME_SCRATCH_RESERVED=$(( PARALOME_SCRATCH_RESERVED + bytes - used )); fi",
    "    done < \"$1\"",
    # Counted, and expanded only when filled: bash before 4.4 calls an empty
    # array unbound under set -u.
    "    if (( PARALOME_SCRATCH_LIVE_N )); then",
    "      printf '%s\\n' ${PARALOME_SCRATCH_LIVE[@]+\"${PARALOME_SCRATCH_LIVE[@]}\"} > \"$1\"",
    "    else",
    "      : > \"$1\"",
    "    fi",
    "  }",
    "  paralome_scratch_release() {",
    "    paralome_scratch_lock \"$1\" || return 0",
    "    paralome_scratch_reserved \"$1\" \"$2\"",
    "    if (( ! PARALOME_SCRATCH_LIVE_N )); then rm -f -- \"$1\"; fi",
    "    paralome_scratch_unlock",
    "  }",
    "fi",
])


def job_scratch_bash(fallback: str, what: str, var: str = "JOB_SCRATCH", need_bytes: str = "0") -> str:
    """Bash that sets ``$<var>`` to this job's scratch directory on this node, or to ``fallback``.

    ``fallback`` is a bash word (already quoted): the shared directory used when
    scratch is not set up, or not usable on this node. Under scratch, the
    directory is the job's own and is removed when the script exits (through
    ``EXIT_CLEANUP_BASH``, which keeps any trap set before it).

    ``need_bytes`` is a bash arithmetic expression: what the job expects to
    write there. Scratch is used only when that much is free on it AND a tenth
    of the filesystem stays free besides; otherwise the job says so and uses
    the shared location. SLURM schedules by what jobs ask for (``--tmp``), not
    what they write, so this check is what keeps concurrent jobs from filling
    the disk.

    What the other jobs on this scratch have reserved and not yet
    written is taken off the free space first, and this job's own need is
    reserved until it exits (``RESERVATION_BASH``). The script that carries
    this should also ask SLURM for the space: ``tmp_request`` in
    ``paralome.pipeline.slurm``.
    """
    setting = configured_setting() or ""
    label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in what) or "job"
    pipeline = "".join(ch for ch in os.environ.get(PIPELINE_ENV, "") if ch.isalnum())
    prefix = f"paralome-{pipeline}-" if pipeline else "paralome-"
    return "\n".join([
        f"# Temporary files on this node's scratch ({SCRATCH_ENV} overrides the site's setting).",
        f"PARALOME_SCRATCH_SETTING=\"${{{SCRATCH_ENV}:-{setting}}}\"",
        "PARALOME_SCRATCH_QUIET=0",
        "case \"$PARALOME_SCRATCH_SETTING\" in",
        # '' and 'none' both mean no scratch, but they are not the same thing. 'none' is a decision someone
        # made, so the job says nothing; '' means no site file was found on this node, which is worth a
        # line, so temporary files going to shared storage never goes unnoticed.
        f"  {NO_SCRATCH}) PARALOME_SCRATCH_BASE=''; PARALOME_SCRATCH_QUIET=1 ;;",
        "  '') PARALOME_SCRATCH_BASE='' ;;",
        f"  {USE_TMPDIR}) PARALOME_SCRATCH_BASE=\"${{TMPDIR:-}}\" ;;",
        "  *) PARALOME_SCRATCH_BASE=\"$PARALOME_SCRATCH_SETTING\" ;;",
        "esac",
        EXIT_CLEANUP_BASH,
        RESERVATION_BASH,
        f"{var}=''",
        f"PARALOME_SCRATCH_NEED=$(( {need_bytes} ))",
        "PARALOME_SCRATCH_ROOM=0",
        "PARALOME_SCRATCH_RESERVED=0",
        "PARALOME_SCRATCH_LOCKED=0",
        # Why scratch was not used, said out loud in every case. A
        # run with no site setting on this node fell back in silence, which is
        # how a whole cluster's jobs wrote their temporary files to shared
        # storage without anyone seeing it.
        "PARALOME_SCRATCH_WHY='no scratch is configured on this node (no site setting and no "
        "PARALOME_SCRATCH in the environment)'",
        "if [[ -n \"$PARALOME_SCRATCH_BASE\" && ! -d \"$PARALOME_SCRATCH_BASE\" ]]; then",
        "  PARALOME_SCRATCH_WHY=\"$PARALOME_SCRATCH_BASE is not a directory on $(hostname)\"",
        "elif [[ -n \"$PARALOME_SCRATCH_BASE\" && ! -w \"$PARALOME_SCRATCH_BASE\" ]]; then",
        "  PARALOME_SCRATCH_WHY=\"$PARALOME_SCRATCH_BASE is not writable on $(hostname)\"",
        "fi",
        "if [[ -n \"$PARALOME_SCRATCH_BASE\" && -d \"$PARALOME_SCRATCH_BASE\" && -w \"$PARALOME_SCRATCH_BASE\" ]]; then",
        # The ledger is locked from here until this job's own
        # reservation is in it, so two jobs cannot both count the same space.
        f"  PARALOME_SCRATCH_LEDGER=\"$PARALOME_SCRATCH_BASE/{RESERVATIONS_FILE}\"",
        "  if paralome_scratch_lock \"$PARALOME_SCRATCH_LEDGER\" 2>/dev/null; then",
        "    PARALOME_SCRATCH_LOCKED=1",
        "    paralome_scratch_reserved \"$PARALOME_SCRATCH_LEDGER\"",
        "  else",
        "    echo \"scratch: the reservations in $PARALOME_SCRATCH_LEDGER could not be locked on $(hostname); "
        "the space check counts free space only\" >&2",
        "  fi",
        "  read -r PARALOME_SCRATCH_SIZE_KB PARALOME_SCRATCH_FREE_KB < <(df -Pk \"$PARALOME_SCRATCH_BASE\" | awk 'NR==2 {print $2, $4}')",
        f"  if (( PARALOME_SCRATCH_FREE_KB * 1024 >= PARALOME_SCRATCH_NEED + PARALOME_SCRATCH_RESERVED + PARALOME_SCRATCH_SIZE_KB * 1024 * {int(KEEP_FREE_PERCENT)} / 100 )); then",
        "    PARALOME_SCRATCH_ROOM=1",
        "  else",
        "    PARALOME_SCRATCH_WHY=\"not enough room on $PARALOME_SCRATCH_BASE ($(( PARALOME_SCRATCH_FREE_KB / 1048576 )) GB free, "
        "about $(( PARALOME_SCRATCH_RESERVED / 1073741824 )) GB of it reserved by other jobs, "
        "this job needs about $(( PARALOME_SCRATCH_NEED / 1073741824 + 1 )) GB and a tenth stays free)\"",
        "  fi",
        "fi",
        "if (( PARALOME_SCRATCH_ROOM )); then",
        f"  {var}=$(mktemp -d \"$PARALOME_SCRATCH_BASE/{prefix}${{USER:-user}}-${{SLURM_JOB_ID:-$$}}-"
        f"${{SLURM_ARRAY_TASK_ID:-0}}-{label}.XXXXXX\") || {var}=''",
        f"  if [[ -z \"${var}\" ]]; then",
        "    PARALOME_SCRATCH_WHY=\"a directory could not be created under $PARALOME_SCRATCH_BASE on $(hostname)\"",
        "  fi",
        "fi",
        f"if (( PARALOME_SCRATCH_LOCKED )) && [[ -n \"${var}\" ]] && (( PARALOME_SCRATCH_NEED > 0 )); then",
        f"  printf '%s %s %s %s\\n' \"$(uname -n)\" \"$$\" \"$PARALOME_SCRATCH_NEED\" \"${var}\" >> \"$PARALOME_SCRATCH_LEDGER\"",
        # Registered before the directory's removal, so it runs after it.
        f"  paralome_on_exit paralome_scratch_release \"$PARALOME_SCRATCH_LEDGER\" \"${var}\"",
        "elif (( PARALOME_SCRATCH_LOCKED )) && (( ! PARALOME_SCRATCH_LIVE_N )); then",
        "  rm -f -- \"$PARALOME_SCRATCH_LEDGER\"",
        "fi",
        "if (( PARALOME_SCRATCH_LOCKED )); then paralome_scratch_unlock; fi",
        f"if [[ -n \"${var}\" ]]; then",
        f"  paralome_on_exit rm -rf -- \"${var}\"",
        f"  echo \"scratch: ${var} on $(hostname)\"",
        "else",
        f"  {var}={fallback}",
        f"  mkdir -p \"${var}\"",
        "  if (( ! PARALOME_SCRATCH_QUIET )); then",
        f"    echo \"scratch: not used on $(hostname) -- $PARALOME_SCRATCH_WHY; "
        f"temporary files go to ${var}\" >&2",
        "  fi",
        "fi",
    ])


def uses_scratch(script_text: str) -> bool:
    """Whether a generated script carries ``job_scratch_bash`` (its sbatch then asks for --tmp)."""
    return "PARALOME_SCRATCH_SETTING=" in str(script_text)


@contextlib.contextmanager
def job_scratch_dir(fallback: Path, what: str, need_bytes: int = 0) -> Iterator[Path]:
    """A directory for this process's temporary files: on scratch when set up and usable here, else ``fallback``.

    The scratch directory is removed on exit; ``fallback`` is created and left to its caller (as before).
    """
    setting = configured_setting()
    base: Optional[str] = None
    if setting == USE_TMPDIR:
        base = os.environ.get("TMPDIR") or None
    elif setting and setting != NO_SCRATCH:
        base = setting
    made: Optional[Path] = None
    room = False
    ledger: Optional[Path] = None
    if base and os.path.isdir(base) and os.access(base, os.W_OK | os.X_OK):
        # Under the ledger's lock until this job's reservation is in
        # it, as the bash guard does (RESERVATION_BASH).
        ledger = Path(base) / RESERVATIONS_FILE
        held = _lock_ledger(ledger)
        try:
            reserved, live = _reserved(ledger) if held else (0, 0)
            if held is None:
                import sys

                print(f"scratch: the reservations in {ledger} could not be locked on {os.uname().nodename}; "
                      "the space check counts free space only", file=sys.stderr, flush=True)
            try:
                usage = shutil.disk_usage(base)
                room = usage.free >= int(need_bytes) + reserved + usage.total * KEEP_FREE_PERCENT / 100
            except OSError:
                room = False
            if not room:
                import sys

                print(f"scratch: not enough room on {base} on {os.uname().nodename} for about "
                      f"{int(need_bytes) // 1024 ** 3 + 1} GB with about {reserved // 1024 ** 3} GB reserved by "
                      "other jobs and a tenth kept free; using the shared location",
                      file=sys.stderr, flush=True)
            if room:
                label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in what) or "job"
                user = os.environ.get("USER") or str(os.getuid())
                job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
                task = os.environ.get("SLURM_ARRAY_TASK_ID") or "0"
                try:
                    pipeline = "".join(ch for ch in os.environ.get(PIPELINE_ENV, "") if ch.isalnum())
                    start = f"paralome-{pipeline}-" if pipeline else "paralome-"
                    made = Path(tempfile.mkdtemp(prefix=f"{start}{user}-{job}-{task}-{label}.", dir=base))
                except OSError:
                    made = None
            if held is not None:
                if made is not None and int(need_bytes) > 0:
                    with open(ledger, "a", encoding="utf-8") as handle:
                        handle.write(f"{os.uname().nodename} {os.getpid()} {int(need_bytes)} {made}\n")
                elif not live:
                    ledger.unlink(missing_ok=True)
        finally:
            if held is not None:
                held.close()
    if made is None:
        if base and not (os.path.isdir(base) and os.access(base, os.W_OK | os.X_OK)):
            import sys

            print(f"scratch: {base} is not a writable directory on {os.uname().nodename}; "
                  f"temporary files go to {fallback}", file=sys.stderr, flush=True)
        Path(fallback).mkdir(parents=True, exist_ok=True)
        yield Path(fallback)
        return
    # SIGTERM (scancel, time limit) would end the process without
    # running the finally below; turn it into SystemExit so the directory goes,
    # as the bash jobs' EXIT trap does. A handler set by someone else stays.
    previous = _exit_on_sigterm()
    try:
        yield made
    finally:
        try:
            shutil.rmtree(made, ignore_errors=True)
            if ledger is not None:
                _release(ledger, made)
        finally:
            _restore_sigterm(previous)


def _lock_ledger(ledger: Path):
    """The ledger open and flocked, or None when that cannot be had.

    A ledger another job removed while this one waited is no longer the file at
    the path; the lock is taken again on the one that is there.
    """
    import fcntl
    import time

    for _attempt in range(10):
        try:
            handle = open(ledger, "a+", encoding="utf-8")
        except OSError:
            return None
        try:
            os.chmod(ledger, 0o666)
        except OSError:
            pass
        deadline = time.monotonic() + RESERVATION_LOCK_SECONDS
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    return None
                time.sleep(0.2)
            except OSError:
                handle.close()
                return None
        try:
            mine, there = os.fstat(handle.fileno()), os.stat(ledger)
            if (mine.st_dev, mine.st_ino) == (there.st_dev, there.st_ino):
                return handle
        except OSError:
            pass
        handle.close()
    return None


def _bytes_in(directory: str) -> int:
    total = 0
    for root, dirs, files in os.walk(directory):
        for name in dirs + files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _reserved(ledger: Path, dropping: Optional[Path] = None) -> tuple:
    """(bytes the live reservations have yet to write, how many are live); the ledger keeps only those.

    Called with the ledger locked. Live: its job's process is still there (when
    on this host) and its directory exists. ``dropping`` is left out as well.
    """
    me = os.uname().nodename
    live: List[str] = []
    reserved = 0
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(" ", 3)
        if len(parts) != 4 or not parts[1].isdigit() or not parts[2].isdigit() or not parts[3]:
            continue
        host, pid, need, directory = parts
        if host == me and not os.path.isdir(f"/proc/{pid}"):
            continue
        if not os.path.isdir(directory) or (dropping is not None and directory == str(dropping)):
            continue
        live.append(line)
        reserved += max(0, int(need) - _bytes_in(directory))
    try:
        with open(ledger, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{line}\n" for line in live))
    except OSError:
        pass
    return reserved, len(live)


def _release(ledger: Path, directory: Path) -> None:
    """This job's reservation out of the ledger; the ledger goes when it is empty."""
    held = _lock_ledger(ledger)
    if held is None:
        return
    try:
        _reserved_count = _reserved(ledger, dropping=directory)[1]
        if not _reserved_count:
            ledger.unlink(missing_ok=True)
    finally:
        held.close()


def _raise_exit_on_sigterm(signum, frame):  # noqa: ARG001 -- signal handler signature
    raise SystemExit(128 + int(signum))


def _exit_on_sigterm():
    """Install the SIGTERM-to-SystemExit handler when SIGTERM has its default action; what to restore, or None."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
            return None
        return signal.signal(signal.SIGTERM, _raise_exit_on_sigterm)
    except (ValueError, OSError, AttributeError):
        return None


def _restore_sigterm(previous) -> None:
    if previous is None:
        return
    import signal

    try:
        signal.signal(signal.SIGTERM, previous)
    except (ValueError, OSError):
        pass


def quoted(path: Path) -> str:
    return shlex.quote(str(path))
