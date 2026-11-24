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

"""What a genome is called, decided by what it IS.

THE PROBLEM IT SOLVES IS DUPLICATION, not tidiness. A check that refuses to
publish a name somebody else already registered is evadable by typing a
different name: `Human_GRCh38.e81` and `Human38` pointing at one reference are
two registry entries with nothing in common that any check can see, which is
the dozen-human-genomes outcome the shared repository exists to prevent.

A name derived from the genome's own identity cannot be evaded, because there
is no free text left to vary. Two people registering Ensembl 81 of GRCh38
produce the same name whether they meant to or not, and the collision is
detected at the moment it is created.

THE FORM, and why there is no lookup table:

    <Species>.<Assembly>.<source-initial><release>

    Homo_sapiens.GRCh38.e81
    Mus_musculus.GRCm39.e112
    Macaca_mulatta.Mmul_10.e113
    Ovis_aries_rambouillet.ARS-UI_Ramb_v2_0.e113

Common names (Human, Mouse, Sheep) would read better but need a table mapping
every binomial anyone might register to a word somebody chose, which is wrong
the first time a lab registers a rotifer, and its gaps are silent. The binomial
is already in the entry, comes from the FASTA filename via
identity_from_fasta, and is what the rest of biology uses. So it costs a
longer sheep and buys a rule with no exceptions. A species misspelt in a
hand-typed name comes out right, because the typo was in the name and never
in the identity.

WHAT THE IDENTITY KEY IS FOR, separately. ``identity_key`` is the same
derivation casefolded, and it is computed rather than stored -- so a genome
registered before any of this existed still has one, and no backfill is needed
anywhere. Duplicate detection compares keys; display uses names.

FORMER NAMES ARE KEPT. Renaming a genome that people have already run against
would invalidate every project that recorded which genome produced a result,
so ``rename`` records what the entry used to be called and
``select_genome_config`` matches those too. A name is a handle, and old
handles keep working.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

#: The initial that stands for each annotation source in a derived name (`e` for Ensembl, as in
#: `Homo_sapiens.GRCh38.e81`).
SOURCE_INITIALS = {
    "ensembl": "e",
    "gencode": "g",
    "refseq": "r",
    "custom": "c",
    "unknown": "u",
}

#: The key an entry records the names it used to answer to under.
FORMER_NAMES_KEY = "former_names"

#: Characters a name field may contain. Everything else becomes an underscore,
#: including the dot -- the dot is the field separator, so an assembly like
#: ARS-UI_Ramb_v2.0 must not be allowed to introduce a fourth field.
_SAFE = re.compile(r"[^A-Za-z0-9-]+")


def _field(value: Any) -> str:
    """One component of a name, safe to join with dots.

    The word ``unknown`` is no value. `setup genome add --yes`
    records it for a species, assembly or release the scan could not read, and
    read as a real value it named every such genome ``unknown.unknown.uunknown``
    -- so the second one registered replaced the first.
    """
    text = str(value or "").strip()
    if text.casefold() == "unknown":
        return ""
    return _SAFE.sub("_", text).strip("_")


def derive_genome_name(entry: Mapping[str, Any]) -> Optional[str]:
    """The name this genome should be called, or None if it cannot be derived.

    None rather than a guess. A genome with no species or no assembly recorded
    cannot be named by what it is, and inventing a partial name would produce
    exactly the silent collisions this exists to prevent -- two different
    assemblies of one species arriving at one name because the assembly was
    blank in both.
    """
    species = _field(entry.get("species"))
    assembly = _field(entry.get("assembly"))
    if not species or not assembly:
        return None

    source = str(entry.get("annotation_source") or "").strip().lower()
    release = _field(entry.get("annotation_release"))
    # A MODIFIED GENOME IS NOT THE STOCK ONE. The name and identity
    # left `modified` out, so registering a GFP/mCherry build adopted another
    # account's stock entry of the same assembly and release (and the reverse),
    # and the run aligned to files without the transgenes. The note is part of
    # it, so two different modifications are two genomes. Kept inside the last
    # field: the dot separates fields.
    modified = ""
    if entry.get("modified"):
        full = _field(entry.get("modified_note"))
        note = full[:32].strip("_")
        if len(full) > 32:
            # A hash of the whole note, so notes that differ after 32 characters differ here.
            import hashlib

            note += "_" + hashlib.sha256(full.encode("utf-8")).hexdigest()[:8]
        modified = "-mod" + (f"_{note}" if note else "")
    if not release:
        # A reference with no annotation release is still a genome; it is just
        # one whose annotation nobody recorded. Named by assembly alone rather
        # than refused, because the FASTA and the assembly identify it.
        return f"{species}.{assembly}{modified}"
    initial = SOURCE_INITIALS.get(source, SOURCE_INITIALS["unknown"])
    return f"{species}.{assembly}.{initial}{release}{modified}"


def identity_key(entry: Mapping[str, Any]) -> Optional[str]:
    """What two entries must share to be the same genome. Computed, not stored.

    Casefolded so that a difference of capitalisation between two registrations
    -- "Homo sapiens" against "homo sapiens" -- is not a difference of genome.
    """
    derived = derive_genome_name(entry)
    return derived.casefold() if derived else None


def same_genome(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Whether two entries describe one genome, by identity rather than name.

    Two entries with no derivable identity are never called the same, because
    the answer would rest on nothing. Silence is the honest result there.
    """
    left, right = identity_key(first), identity_key(second)
    return bool(left) and left == right


def former_names(entry: Mapping[str, Any]) -> list[str]:
    """Every name this entry has previously answered to."""
    recorded = entry.get(FORMER_NAMES_KEY)
    if isinstance(recorded, (list, tuple)):
        return [str(item) for item in recorded if str(item).strip()]
    return []


def remember_former_name(entry: dict, previous: str) -> dict:
    """Record that this genome used to be called something else.

    Idempotent, and it never records the name the genome already has -- a
    rename that changes nothing should not leave a trail suggesting it did.
    """
    previous = str(previous or "").strip()
    if not previous or previous == str(entry.get("name") or ""):
        return entry
    names = former_names(entry)
    if previous not in names:
        names.append(previous)
    entry[FORMER_NAMES_KEY] = names
    return entry
