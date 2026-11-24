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

"""Retiring a genome, and telling the people who adopted it.

WHY WITHDRAWAL IS NOT DELETION. `setup genome remove` takes an entry out of the
owner's config, which stops NEW adoptions and tells nobody. Everyone who
already adopted keeps an independent copy -- adoption copies rather than
references -- so a deletion is silent precisely when it most needs not to be.
Withdrawal leaves the entry in place carrying a reason, because a removed
record cannot carry a message.

TWO SEVERITIES, AND THE DISTINCTION IS THE POINT. hg19 was a perfectly good
genome for a long time, and was replaced by hg38. Nothing about hg19 is wrong.
Years of correct results came off it, and somebody mid-analysis should not
have it pulled out from under them -- they should simply not START new work on
it.

That is a different thing from an index built against the wrong FASTA, where
every result is wrong and the only safe answer is to stop. Collapsing the two
into one flag would either block work that is fine or wave through work that is
not, so:

    SUPERSEDED  warns, every time, and proceeds.
    FATAL       refuses, and refuses for the OWNER too -- a genome that
                produces wrong answers does so regardless of who runs it.

HOW AN ADOPTER FINDS OUT. Their copy carries no withdrawal; the marker lives in
the owner's config. But an adopted entry records ``discovered_from``, so the
owner is known by name and their config can be read -- the same read discovery
already does, and still nothing is ever written outside the caller's own home.

WHAT SUPERSEDES WHAT IS COMPUTED, NOT TYPED. Same species, same assembly, higher annotation release
is a newer build of the same genome, and that is arithmetic rather than
bookkeeping somebody has to remember to do.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

#: The key an owner's entry carries once retired.
WITHDRAWN_KEY = "withdrawn"

FATAL = "fatal"
SUPERSEDED = "superseded"
SEVERITIES = (SUPERSEDED, FATAL)


def withdrawal_record(
    severity: str, reason: str, by: str
) -> Dict[str, Any]:
    """The block written into the owner's entry.

    Dated and attributed, because a retirement is a claim somebody made and an
    adopter reading it months later needs to know who and when before deciding
    whether to act on it.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, not {severity!r}")
    return {
        "severity": severity,
        "reason": reason.strip(),
        "by": by,
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def withdrawal_of(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The withdrawal recorded on this entry itself, if any."""
    found = entry.get(WITHDRAWN_KEY)
    if isinstance(found, dict) and found.get("severity") in SEVERITIES:
        return dict(found)
    return None


def _owners_entry(name: str, account: str) -> Optional[Dict[str, Any]]:
    """The named account's own record of this genome, read where permitted.

    Read-only, and silent about anything it cannot reach: a colleague whose
    home is mode 700 simply cannot tell us, which is the filesystem answering
    rather than a failure. Never writes.
    """
    from paralome.config.user import CONFIG_FILENAME, DEFAULT_CONFIG_DIRNAME

    try:
        import pwd  # Inside the try; Windows has no pwd.

        home = Path(pwd.getpwnam(account).pw_dir)
    except (KeyError, AttributeError, ImportError):
        return None
    config = home / DEFAULT_CONFIG_DIRNAME / CONFIG_FILENAME
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    genomes = payload.get("genomes") if isinstance(payload, dict) else None
    if not isinstance(genomes, dict):
        return None
    entry = genomes.get(name)
    return dict(entry) if isinstance(entry, dict) else None


def _registry_entry(name: str) -> Optional[Dict[str, Any]]:
    """The shared registry's copy of this genome, if the machine has one.

    Best effort and never fatal: a site with no shared registry, or one nobody
    can read, simply yields nothing. When no entry has the name, an
    entry that records it among its former names (a rename) is the copy.
    """
    from paralome.errors import ParalomeError

    try:
        from paralome.setup.genome_names import former_names
        from paralome.setup.genomes import load_site_registry

        registry = load_site_registry()
        found = registry.get(name)
        if found is None:
            renamed = [item for item in registry.values()
                       if isinstance(item, dict) and name in former_names(item)]
            found = renamed[0] if len(renamed) == 1 else None
    except ParalomeError:
        # A refusal with a number is not a broken registry. Finding
        # the registry reads the configuration, and a key nothing reads in it
        # is refused by name here as everywhere.
        raise
    except Exception:  # pragma: no cover - a broken registry is not this
        return None                                         # function's problem
    return dict(found) if isinstance(found, dict) else None


def effective_withdrawal(
    name: str, entry: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Whether this genome is withdrawn, checking wherever the owner can say so.

    An adopted copy never carries the marker -- the owner wrote it afterwards.
    So this follows the channel the copy came through, and there are two:

    THE SHARED REGISTRY, checked first because it works regardless of how
    private anyone's home is. ``load_site_registry`` merges drop-in files
    verbatim and stamps no ``discovered_from``, so a genome adopted from the
    registry carries no pointer to anybody; the registry's copy holds the
    withdrawal, since `withdraw` republishes. On a machine where users cannot
    read each other's homes, that is the only channel there is.

    THE OWNER'S OWN CONFIG, via ``discovered_from``, for copies taken through
    passwd discovery.

    The owner's word wins over the local copy either way: an adopter cannot
    un-withdraw somebody else's genome by holding an older copy of it.

    A registry entry is only trusted when it is the SAME genome -- same files,
    compared by definition rather than by name. Two accounts can register
    different references under one name, and blocking somebody's run because a
    stranger's unrelated genome shares a string would be worse than the gap
    this closes.
    """
    # Every channel is consulted and a FATAL anywhere wins (a local SUPERSEDED marker must not hide the
    # owner's FATAL one); otherwise the first record found.
    found_all: List[Dict[str, Any]] = []
    own = withdrawal_of(entry)
    if own is not None:
        found_all.append(own)

    from paralome.setup.genomes import DERIVED_KEYS, differing_keys

    shared = _registry_entry(name)
    # The owner's later additions (burden_reference, annovar tables) and a
    # rename do not make it another genome; the files decide.
    ignored = set(DERIVED_KEYS) | {"name", "annovar_protocol", "annovar_operation"}
    differing = [key for key in differing_keys(shared, entry) if key not in ignored] if shared else []
    if shared is not None and not differing:
        found = withdrawal_of(shared)
        if found is not None:
            found.setdefault("upstream", str(shared.get("registered_by") or "the registry"))
            found_all.append(found)
    elif shared is not None and withdrawal_of(shared) is not None:
        import sys

        print(f"Warning: the shared registry's {name} is withdrawn, but differs from yours in "
              f"{', '.join(differing)}; not applied to your copy. Compare them (paralome setup genome sync).",
              file=sys.stderr)

    found_all.extend(_provenance_withdrawals(name, entry))
    for found in found_all:
        if found.get("severity") == FATAL:
            return found
    return found_all[0] if found_all else None


#: How many accounts a provenance chain is followed through before stopping.
MAX_PROVENANCE_HOPS = 8


def _provenance_withdrawals(name: str, entry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Withdrawals recorded anywhere along this copy's provenance chain.

    Not only ``discovered_from`` but also ``registered_by``, and on
    through each holder's own entry, so a copy of a copy -- or one whose
    ``discovered_from`` names an intermediate adopter -- still hears the
    registrant. Read-only, bounded, and cycle-safe.
    """
    from paralome.setup.genomes import _current_user

    me = _current_user()
    queue = [entry.get("discovered_from"), entry.get("registered_by")]
    seen: set = set()
    found: List[Dict[str, Any]] = []
    while queue and len(seen) < MAX_PROVENANCE_HOPS:
        account = str(queue.pop(0) or "")
        if not account or account in seen or account == me:
            continue
        seen.add(account)
        upstream = _owners_entry(name, account)
        if upstream is None:
            continue
        record = withdrawal_of(upstream)
        if record is not None:
            record.setdefault("upstream", account)
            found.append(record)
        queue.extend([upstream.get("discovered_from"), upstream.get("registered_by")])
    return found


# ---------------------------------------------------------------------------
# What replaces what
# ---------------------------------------------------------------------------


#: Recorded on an entry the user has altered (fluorescent genes -- GFP, mCherry -- added to assemblies
#: and GTFs, say).
MODIFIED_KEY = "modified"
MODIFIED_NOTE_KEY = "modified_note"


class Replacement(NamedTuple):
    """Something newer than this genome, and how far it can be trusted.

    ``caution`` is empty for a true drop-in and carries a sentence otherwise.
    It exists because the two useful answers here are not "yes" and "no" but
    "yes, swap it" and "yes, something newer exists, and it is not a swap" --
    and a system that only says the first two would either hide the second case
    or misrepresent it.
    """

    name: str
    relation: str          # "annotation" or "assembly"
    detail: str            # "Ensembl 113", "GRCh38"
    caution: str = ""

    @property
    def headline(self) -> str:
        """How this is introduced, in one phrase.

        "Drop-in replacement" is reserved for the case that actually is one.
        A line reading "Drop-in replacement" above a caution reading "not a
        swap" is a contradiction, and the reader is entitled to believe the
        first line -- so only a same-assembly, same-source, unmodified upgrade
        earns the phrase.
        """
        if self.caution:
            return "Newer assembly" if self.relation == "assembly" else "Newer annotation"
        return "Drop-in replacement"


def is_modified(entry: Mapping[str, Any]) -> bool:
    """Whether the user altered these files after they left their source."""
    return bool(entry.get(MODIFIED_KEY))


def _modification_note(entry: Mapping[str, Any]) -> str:
    return str(entry.get(MODIFIED_NOTE_KEY) or "").strip()


def _known(value: Any) -> Optional[str]:
    """A recorded field, or None when it is absent or explicitly unknown."""
    text = str(value or "").strip()
    if not text or text.lower() == "unknown":
        return None
    return text


_TRAILING_NUMBER = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+(?:\.\d+)*)$")


def _ordered(value: Optional[str]) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """Split a version-ish string into a prefix and a number, for comparison.

    GRCh38 -> ("grch", (38,)). v44 -> ("v", (44,)). vM33 -> ("vm", (33,)).
    Mmul_10 -> ("mmul_", (10,)). ARS-UI_Ramb_v3.0 -> ("ars-ui_ramb_v", (3,)).
    The number is compared component by component (v2.10 after
    v2.9), trailing zero components dropped (3.0 is 3).

    Two values are only ever compared when their PREFIXES match exactly, which
    is what keeps this honest. GENCODE's human (v44) and mouse (vM33) lines do
    not compare to each other, and neither do hg38 and GRCh38 -- the same
    assembly under two naming conventions, which Paralome deliberately does not
    try to alias. Refusing to compare is always available and always safe; a
    wrong comparison here moves somebody's coordinates.
    """
    if value is None:
        return None
    match = _TRAILING_NUMBER.match(value.strip())
    if not match:
        return None
    try:
        parts = [int(part) for part in match.group("number").split(".")]
        while len(parts) > 1 and parts[-1] == 0:
            parts.pop()
        return match.group("prefix").lower(), tuple(parts)
    except ValueError:  # pragma: no cover - the regex guarantees a number
        return None


def _release_of(entry: Mapping[str, Any]) -> Optional[str]:
    """The annotation release, falling back to reading it out of the GTF name.

    The fallback covers entries written before the field existed and
    hand-written ones.
    """
    recorded = _known(entry.get("annotation_release"))
    if recorded:
        return recorded
    from paralome.setup.genome_scan import annotation_release

    return annotation_release(entry.get("gtf"))


def _source_of(entry: Mapping[str, Any]) -> Optional[str]:
    """Which annotation project, falling back to the GTF name."""
    recorded = _known(entry.get("annotation_source"))
    if recorded:
        return recorded.lower()
    from paralome.setup.genome_scan import annotation_identity

    return _known(annotation_identity(entry.get("gtf")).get("annotation_source"))


def replacements_for(
    name: str, entry: Mapping[str, Any], available: Mapping[str, Mapping[str, Any]]
) -> List[Replacement]:
    """Everything newer than this genome, separated by what "newer" means.

    A reference bundle carries two independently versioned things, and they
    supersede along different axes with different consequences:

    ANNOTATION -- same species, same assembly, same annotation source, higher
    release. The coordinate system is identical and only the gene models moved,
    so this is a drop-in: re-run and the results are comparable to the old
    ones. No caution.

    ASSEMBLY -- same species, a later assembly. GRCh37 to GRCh38 is an
    honourable retirement, and it is NOT a drop-in: every coordinate changes,
    and old and new results cannot be compared without lifting over. Offered,
    because "hg38 exists" is exactly what somebody on hg19 needs to hear, but
    never in the same words as a drop-in.

    Both are refused across annotation sources, because Ensembl 113 and GENCODE
    v44 are both current and an integer comparison between them would be
    meaningless.

    MODIFIED GENOMES COMPARE ONLY TO THEMSELVES. Stock GRCh38 offered as a
    "newer build" of a genome with GFP and mCherry added would silently drop
    the transgenes, and offering a modified copy to somebody else would
    silently add sequence they never asked for. So anything newer is still
    REPORTED -- knowing it exists is useful -- but always carrying the caution
    that says why it is not a swap.
    """
    species = _known(entry.get("species"))
    if not species:
        return []
    species = species.lower()

    assembly = _known(entry.get("assembly"))
    source = _source_of(entry)
    release = _release_of(entry)
    mine_modified = is_modified(entry)

    found: List[Replacement] = []
    for other_name, other in available.items():
        if other_name == name or withdrawal_of(other):
            continue
        other_species = _known(other.get("species"))
        if not other_species or other_species.lower() != species:
            continue

        other_assembly = _known(other.get("assembly"))
        theirs_modified = is_modified(other)
        caution = ""
        if mine_modified or theirs_modified:
            whose = _modification_note(entry) if mine_modified else _modification_note(other)
            which = "yours is" if mine_modified else f"{other_name} is"
            caution = (
                f"not a swap -- {which} user-modified"
                + (f" ({whose})" if whose else "")
                + "; compare them yourself"
            )

        # Same coordinate system: a newer annotation of it is a drop-in.
        if assembly and other_assembly and assembly.lower() == other_assembly.lower():
            other_source = _source_of(other)
            if not source or not other_source or source != other_source:
                continue
            mine, theirs = _ordered(release), _ordered(_release_of(other))
            if mine and theirs and mine[0] == theirs[0] and theirs[1] > mine[1]:
                found.append(
                    Replacement(
                        other_name,
                        "annotation",
                        f"{other_source} {_release_of(other)}",
                        caution,
                    )
                )
            continue

        # A later assembly: real, useful, and not a drop-in.
        mine, theirs = _ordered(assembly), _ordered(other_assembly)
        if mine and theirs and mine[0] == theirs[0] and theirs[1] > mine[1]:
            migration = (
                "a different coordinate system -- results are not comparable "
                "without lifting over"
            )
            found.append(
                Replacement(
                    other_name,
                    "assembly",
                    str(other_assembly),
                    f"{caution}; {migration}" if caution else migration,
                )
            )
    # Annotation upgrades first: a drop-in is more actionable than a migration.
    return sorted(found, key=lambda r: (r.relation != "annotation", r.name))


def newer_builds(
    name: str, entry: Mapping[str, Any], available: Mapping[str, Mapping[str, Any]]
) -> List[Replacement]:
    """Drop-in replacements only. Kept for callers that want just those."""
    return [
        item for item in replacements_for(name, entry, available)
        if item.relation == "annotation" and not item.caution
    ]


# ---------------------------------------------------------------------------
# The run-time gate
# ---------------------------------------------------------------------------


def enforce(name: str, entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The run-time gate. Fatal stops; superseded warns and returns.

    

    A warning rather than an echo because this sits underneath every pipeline,
    and the ones that run under SLURM have no terminal to echo to -- a warning
    lands in the job's stderr alongside everything else that went wrong, and
    ``-W error`` turns it into a stop for anyone who wants that.

    Returns whatever it found so callers that want to record it can, and None
    when the genome is in good standing.
    """
    import warnings

    from paralome.errors import ParalomeError

    found = effective_withdrawal(name, entry)
    if found is None:
        return None
    if found.get("severity") == FATAL:
        raise ParalomeError(
            407,
            describe(name, found),
            genome=name,
            withdrawn_by=found.get("upstream") or found.get("by"),
            withdrawn_at=found.get("at"),
        )
    warnings.warn(
        describe(name, found)
        + "  -- work already under way against it is still valid; "
        "`paralome setup genome list` names anything newer.",
        UserWarning,
        stacklevel=2,
    )
    return found


# ---------------------------------------------------------------------------
# Writing the marker
# ---------------------------------------------------------------------------


def withdraw(
    name: str,
    config: Dict[str, Any],
    *,
    severity: str,
    reason: str,
    by: str,
) -> Dict[str, Any]:
    """Mark the caller's OWN entry withdrawn, in the config given.

    Only the owner's copy carries the marker, which is why this writes into the
    caller's config and nowhere else. Withdrawing a genome you adopted retires
    YOUR copy of it -- a reasonable thing to want, and not a claim about anyone
    else's.
    """
    genomes = config.setdefault("genomes", {})
    entry = genomes.get(name)
    if not isinstance(entry, dict):
        raise KeyError(name)
    entry[WITHDRAWN_KEY] = withdrawal_record(severity, reason, by)
    return entry[WITHDRAWN_KEY]


def restore(name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Undo a withdrawal on the caller's own entry. Returns what it removed."""
    genomes = config.get("genomes") or {}
    entry = genomes.get(name)
    if not isinstance(entry, dict):
        raise KeyError(name)
    return entry.pop(WITHDRAWN_KEY, None)


def describe(name: str, withdrawal: Mapping[str, Any]) -> str:
    """The sentence a user sees, wherever they meet it."""
    severity = withdrawal.get("severity")
    who = withdrawal.get("upstream") or withdrawal.get("by") or "its owner"
    when = str(withdrawal.get("at") or "")[:10]
    headline = (
        "WITHDRAWN AS UNUSABLE" if severity == FATAL
        else "superseded"
    )
    return (
        f"{name} was {headline} by {who}"
        + (f" on {when}" if when else "")
        + f": {withdrawal.get('reason') or 'no reason given'}"
    )
