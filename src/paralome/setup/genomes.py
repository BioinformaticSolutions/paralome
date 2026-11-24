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

"""Registering a genome, and the machine-wide registry that makes it worth doing once.

`paralome setup genome add` writes the genome entry -- fourteen possible keys,
each with two to five accepted aliases (people hand-write these and spell them
differently) -- so nobody has to hand-edit `~/.paralome/config.json`.

WHY THE REGISTRY IS PER-MACHINE AND NOT SHIPPED. Genome file identities and
paths are computer specific (server or desktop); there is nothing to gain by
shipping a config of where one machine's files are. A genome entry is a set of
absolute paths. Shipped to another institution they name nothing, and
`paralome setup config` would list a genome that fails the moment it is used
-- worse than an empty registry, which at least tells the truth. So the
registry is a file on the machine the paths describe, and the wheel carries no
genomes at all.

WHY IT IS NOT INSIDE THE INSTALL. An upgrade with
`pip install --no-deps --force-reinstall` replaces the package directory
wholesale, so a registry under `site-packages/paralome/` would lose every
genome added since the last install, silently. It also would not be writable
by most users of a shared install -- exactly the people a shared registry is
for.

TWO LAYERS, USER FIRST. Registration writes both the user's own config and the
machine registry. The user's copy wins on a name clash, so anyone can point
`human_grch38` at their own GTF without touching what everyone else uses, and
without needing permission to.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from paralome.config.user import config_path, load_config, save_config
from paralome.errors import ParalomeError

ENV_SITE_REGISTRY = "PARALOME_SITE_REGISTRY"

#: The shared registry is a DIRECTORY of one file per user, not a single file.
#:
#: There may be no group that encompasses everyone, while anyone on the
#: machine should have access. The Unix answer to that is a 1777 directory -- world
#: writable with the sticky bit, like /tmp. But sticky means you may only
#: delete or rename files you own, and a single shared genomes.json is written
#: by replacing it: the first person to register would own it and everyone
#: else's write would fail with EPERM. Dropping the sticky bit instead would
#: let any user delete every genome on the machine with one careless command.
#:
#: One file per user escapes the choice entirely. Everybody writes only their
#: own file, so the sticky bit protects rather than obstructs, the atomic
#: replace keeps working, attribution is structural rather than a field anyone
#: could mis-set, and one corrupt file does not take the other four with it.
#: This is how /etc/profile.d and every conf.d directory works, for the same
#: reasons.
SITE_REGISTRY_DIRNAME = "genomes.d"

#: The older single-file layout, still READ so a registry written in it is not silently lost. Never
#: written.
LEGACY_SITE_REGISTRY_FILENAME = "genomes.json"

#: Anyone may create a file here; only the owner may remove one. The mode is
#: forced with chmod after mkdir because the process umask would otherwise
#: strip the group and other write bits straight back off.
SHARED_DIR_MODE = 0o1777

#: Recorded in the user config when the default location will not do -- a
#: cluster whose install directory is root-owned, most often.
USER_CONFIG_SITE_KEY = "site_registry"

#: Where this machine keeps its genome folders, so a bare name can be used
#: instead of a full path, such as /shared/genomes/<folder>: the part that
#: never changes is recorded once.
USER_CONFIG_REPOSITORY_KEY = "genome_repository"


def genome_repository(config: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    """The recorded root, if there is one.

    THE MACHINE WINS. If this machine has a site.json, the path it
    names is the answer and neither the environment variable nor the user's own
    config is consulted.

    The reason is the whole point of recording it machine-side. A per-user
    override is exactly how somebody ends up with a private repository nobody
    else can see, which is the duplication the shared arrangement exists to
    end -- and there is no legitimate case that needs it: an administrator
    who wants a different location edits site.json, which is theirs to edit.

    Below that: environment, then the user's own config, then nothing.
    """
    from paralome.setup.site import site_genome_repository

    decided = site_genome_repository()
    if decided is not None:
        return decided
    override = os.environ.get("PARALOME_GENOME_REPOSITORY")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    if config is None:
        try:
            config = load_config(initialize=False)
        except ParalomeError:
            # A key nothing reads is refused by name, here as
            # everywhere; only a configuration that cannot be read is set aside.
            raise
        except Exception:  # pragma: no cover - unreadable config is not fatal
            config = {}
    recorded = (config or {}).get(USER_CONFIG_REPOSITORY_KEY)
    if isinstance(recorded, str) and recorded.strip():
        return Path(recorded).expanduser().resolve()
    return None


def config_for_a_report() -> Dict[str, Any]:
    """The user configuration for a command that only reports on a setup.

    `setup doctor` and `setup where` answer on a machine whose
    configuration is refused or cannot be read -- that is when they are asked.
    Doctor prints the refusal itself; neither acts on a setting. Every other
    caller of the two functions above lets the refusal through.
    """
    try:
        return load_config(initialize=False)
    except Exception:
        return {}


def resolve_genome_folder(
    given: str, config: Optional[Mapping[str, Any]] = None
) -> Path:
    """A path as typed, or a bare name resolved against the repository root.

    Anything containing a separator, or that exists as given, is taken
    literally -- a recorded root must never change the meaning of a path
    somebody wrote out in full.
    """
    candidate = Path(given).expanduser()
    if candidate.is_dir() or os.sep in str(given):
        return candidate.resolve()
    root = genome_repository(config)
    if root is not None and (root / given).is_dir():
        return (root / given).resolve()
    return candidate.resolve()


#: Where an account records the genomes it has decided are not for it.
#: A TOP-LEVEL config key, never inside ``genomes`` and never published: the
#: drop-in writer copies ``genomes`` alone, so nothing here can reach another
#: account's view of the machine.
DECLINED_KEY = "declined"


def declined_genomes(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Names this account has said are not for it, with when and why.

    Two different statements could be made with one verb:

        "this genome is bad"    a judgement about the artifact.  The owner's
                                to make, broadcast to everyone (withdrawal;
                                superseded, as hg19 is).
        "I don't need this"     a statement about one person's config.

    Conflating them would turn one person's housekeeping into evidence about
    a genome that is perfectly fine. So a decline is recorded HERE, where only
    its owner can see it, and never in the entry, the registry, or anything
    another account reads.

    It exists because ``Reconciliation`` computes ``only_site`` as everything
    in the registry the user lacks. Without a record of the decision, removing
    a genome you do not need would mean ``sync`` offers it back forever.
    """
    found = config.get(DECLINED_KEY)
    return dict(found) if isinstance(found, dict) else {}


def decline(
    name: str, config: Dict[str, Any], *, note: str = ""
) -> Dict[str, Any]:
    """Record, privately, that this account does not want ``name``.

    Says nothing about the genome. Returns the record written.
    """
    record: Dict[str, Any] = {
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if note:
        record["note"] = note
    declined = declined_genomes(config)
    declined[name] = record
    config[DECLINED_KEY] = declined
    return record


def undecline(name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Forget a decline, so ``sync`` offers the genome again."""
    declined = declined_genomes(config)
    removed = declined.pop(name, None)
    if removed is not None:
        config[DECLINED_KEY] = declined
    return removed


def registered_by(entry: Mapping[str, Any]) -> str:
    """Who put this genome on the machine, as recorded at registration.

    ``stamp`` sets this with ``setdefault`` on a verbatim copy, so an adopted
    entry keeps the ORIGINAL registrant rather than acquiring the adopter --
    which is what makes an adopted copy distinguishable from an owned one
    without any new field.
    """
    return str(entry.get("registered_by") or "")


def is_mine(entry: Mapping[str, Any]) -> bool:
    """Whether this account registered the genome, as opposed to adopting it.

    An entry with no ``registered_by`` at all predates the stamp; it is treated
    as the caller's own, because the alternative is telling someone their own
    pre-0.3 genome belongs to a stranger.
    """
    owner = registered_by(entry)
    return not owner or owner == _current_user()


def remove_genome(
    name: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Drop a genome from this user's config and from their drop-in file.

    ONLY THIS USER'S OWN ENTRIES. The drop-in file we may rewrite is ours; a
    genome another account registered is theirs to remove, and the sticky bit
    on a shared registry enforces that even if this did not. So a name present
    only in a colleague's config is reported as not yours rather than silently
    doing nothing.
    """
    cfg = config if config is not None else load_config(initialize=True)
    genomes = cfg.get("genomes", {}) or {}
    outcome: Dict[str, Any] = {
        "genome": name,
        "removed_from_config": False,
        "removed_from_registry": False,
        "declined": False,
        "entry": None,
        "error": None,
    }
    if name in genomes:
        entry = dict(genomes[name])
        outcome["entry"] = entry
        del genomes[name]
        cfg["genomes"] = genomes
        # Dropping a genome somebody else registered is housekeeping,
        # not a verdict. Remember it so sync stops offering it back -- privately,
        # because the fact that this account does not need hg19 is nobody else's
        # business and certainly not evidence about hg19. Removing one's OWN
        # genome needs no such record: it leaves the registry at the same time,
        # so there is nothing left to be re-offered.
        if not is_mine(entry):
            decline(name, cfg, note="removed a copy registered by "
                                    f"{registered_by(entry) or 'another account'}")
            outcome["declined"] = True
        save_config(cfg)
        outcome["removed_from_config"] = True

    target = Path(registry_path) if registry_path is not None else site_registry_path(cfg)
    try:
        mine = my_registered_genomes(target)
        if name in mine:
            del mine[name]
            save_site_registry(mine, target)
            outcome["removed_from_registry"] = True
    except OSError as exc:
        outcome["error"] = str(exc)

    if not outcome["removed_from_config"] and not outcome["removed_from_registry"]:
        outcome["error"] = outcome["error"] or (
            f"{name} is not registered by this account"
        )
    return outcome

REGISTRY_SCHEMA = "paralome.genome_registry.v1"

#: Bookkeeping this module adds, which is not part of the genome definition.
#: Excluded when two entries are compared, so a genome is not reported as
#: differing because two people registered the same paths on different days.
#:
#: ``analyses`` is here for a different reason: it records what the registrant
#: intends to RUN, not what the genome IS. Two people can want different things
#: from byte-identical files, and reporting that as a conflict would ask them to
#: resolve a disagreement that does not exist.
PROVENANCE_KEYS = (
    "registered_by", "registered_at", "registered_on", "analyses",
    # Where we read the entry from, not anything about the genome itself.
    "discovered_from",
    # A retirement is a judgement ABOUT the genome, not part of what
    # the genome is. Two entries naming byte-identical files are still the same
    # genome after one of them is withdrawn -- and leaving this in the
    # definition would make sync report "differing: withdrawn" and offer to
    # resolve it by hand, when the correct handling is the adopt-time check.
    "withdrawn",
    # What a genome used to be CALLED is history, not identity. Two entries naming byte-identical files
    # are the same genome whether one of them has been through `setup genome rename` and the other has
    # not, so sync does not report "differing: former_names" while accounts catch up.
    "former_names",
)


# ---------------------------------------------------------------------------
# Which keys a genome actually needs
#
# Derived from where the resolver is CONSUMED, not from RESOURCE_KEYS, which
# lists everything the resolver will accept. Asking a user for a key nothing
# reads is the defect this repository has shipped five times -- a flag that
# parses and reaches nothing.
#
# `taxonomy_id` is the live example: declared in RESOURCE_KEYS with three
# aliases and read by no module in the package. It is deliberately absent below.
# ---------------------------------------------------------------------------

#: Every analysis resolves these two. reference_fasta reaches alignment, the
#: MACS effective genome size (atac/genome_size.py) and motif scanning, where it
#: is the one hard requirement (motifs.py passes required=("reference_fasta",)).
#: gtf reaches annotation on every path that has one.
UNIVERSAL_KEYS = ("reference_fasta", "gtf")

ANALYSIS_KEYS: Dict[str, Tuple[str, ...]] = {
    # commands/pipeline.py resolves star_index for alignment.
    "RNA": ("star_index",),
    "scRNA": ("star_index",),
    # atac/motifs.py: resources.value("motif_species") and .path("motif_database").
    "ATAC": ("motif_database", "motif_species"),
    "snATAC": ("motif_database", "motif_species"),
    # commands/annotate.py, both annotators:
    #   required=("annovar_dir", "annovar_db")
    #   required=("snpeff_jar", "snpeff_config", "snpeff_genome")
    "DNA": (
        "annovar_dir", "annovar_db", "annovar_buildver",
        "annovar_protocol", "annovar_operation",
        "snpeff_jar", "snpeff_config", "snpeff_genome",
    ),
}

#: What each key is, whether it names a file/directory/plain text, and -- when
#: the key can be left out -- what happens if it is.
#:
#: THE THIRD FIELD IS THE POINT. Some keys have a working fallback in the code
#: that consumes them (there is no JASPAR collection for rhesus macaque, and the
#: motif keys fall back), so the walkthrough must not invent a requirement the
#: pipeline does not have.
#:
#: A note here means the key may be skipped. None means the consumer passes it
#: in a `required=` tuple, or nothing downstream can proceed without it.
KEY_PROMPTS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "reference_fasta": ("Reference genome FASTA", "file", None),
    "gtf": ("Gene annotation GTF", "file", None),
    "star_index": ("STAR index directory", "directory", None),
    # atac/motifs.py falls back to the JASPAR CORE vertebrates non-redundant
    # set, cached under ~/.paralome/motifs/ and downloaded once if absent.
    # Correct for any vertebrate without its own collection, which is most of
    # them -- JASPAR CORE is organised by taxonomic group, not by species.
    "motif_database": (
        "Motif database, JASPAR MEME format", "file",
        "skip to use JASPAR CORE vertebrates -- right for any vertebrate; "
        "give a path for plants, insects, fungi or nematodes",
    ),
    # infer_motif_species reads the genome name for mouse/human tokens and
    # otherwise returns "vertebrates".
    "motif_species": (
        "Motif species common name, e.g. human", "text",
        "skip to infer it from the genome name",
    ),
    "annovar_dir": ("ANNOVAR install directory, holding table_annovar.pl", "directory", None),
    "annovar_db": ("ANNOVAR database directory, e.g. humandb", "directory", None),
    # Skipping these three records NOTHING: there is no human fallback. A DNA run with ANNOVAR refuses until
    # the key is recorded or given as an option.
    "annovar_buildver": (
        "ANNOVAR build version, e.g. hg38", "text",
        "skip to record none -- a DNA run with ANNOVAR then needs --annovar-buildver",
    ),
    "annovar_protocol": (
        "ANNOVAR protocols, comma separated", "text",
        "skip to record none -- a DNA run with ANNOVAR then needs --annovar-protocol",
    ),
    "annovar_operation": (
        "ANNOVAR operations, comma separated, one per protocol", "text",
        "skip to record none -- a DNA run with ANNOVAR then needs --annovar-operation",
    ),
    "snpeff_jar": ("snpEff jar", "file", None),
    "snpeff_config": ("snpEff config file", "text_or_file", None),
    "snpeff_genome": ("snpEff genome database name, e.g. GRCh38.99", "text", None),
    # BUILT, never asked for -- no analysis lists it in ANALYSIS_KEYS,
    # so the walkthrough does not prompt. Here for its kind (a file, so the
    # listing checks it) and for what its absence means.
    "burden_reference": (
        "Gene-burden gnomAD reference table", "file",
        "skip -- built by `paralome setup genome burden-reference`; without it "
        "a DNA run skips the gene_burden stage",
    ),
}


def is_optional(key: str) -> bool:
    """Whether the walkthrough will accept an empty answer for this key."""
    entry = KEY_PROMPTS.get(key)
    return bool(entry and entry[2])


#: The key a registered genome records its intended analyses under.
ANALYSES_KEY = "analyses"

#: Validation sections, and what puts each one in scope. A section is checked
#: when the genome was registered for one of its analyses, OR when any of its
#: keys is present -- the second half is what makes this work for genomes
#: registered, which carry no analyses list.
DOCTOR_SECTIONS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "reference": ((), ("reference_fasta", "gtf")),
    "star": (("RNA", "scRNA"), ("star_index",)),
    "motifs": (("ATAC", "snATAC"), ("motif_database", "motif_species")),
    "annovar": (
        ("DNA",),
        ("annovar_dir", "annovar_db", "annovar_buildver",
         "annovar_protocol", "annovar_operation"),
    ),
    "snpeff": (("DNA",), ("snpeff_jar", "snpeff_config", "snpeff_genome")),
    # Absent is not a fault (doctor reports it as not built yet).
    "burden": (("DNA",), ("burden_reference",)),
}


def sections_in_scope(entry: Mapping[str, Any]) -> Dict[str, bool]:
    """Which validation sections this genome should be judged on.

    A genome that is entirely correct for what it is for must not be called
    invalid (for missing ANNOVAR and snpEff keys on a genome registered for
    RNA and ATAC, say), or the word stops meaning anything and the next real
    failure gets waved past.

    ``reference`` is always in scope: every analysis resolves a FASTA and a GTF.

    Everything else is in scope when the genome says it is for one of that
    section's analyses, or when it carries one of that section's keys. The
    second condition is not redundant -- it is how a genome registered before
    the analyses list existed still gets checked properly. It also means a
    genome registered for DNA with the ANNOVAR keys left blank is still
    reported as missing them, which is the case worth catching.
    """
    declared = {str(item) for item in entry.get(ANALYSES_KEY) or ()}
    scope: Dict[str, bool] = {}
    for section, (analyses, keys) in DOCTOR_SECTIONS.items():
        if not analyses:
            scope[section] = True
            continue
        scope[section] = bool(declared & set(analyses)) or any(
            str(entry.get(key) or "").strip() for key in keys
        )
    return scope

#: Identity fields, asked for every genome regardless of analysis.
IDENTITY_KEYS = ("species", "assembly")


#: The order analyses are OFFERED in, which is not alphabetical.
#:
#: DNA is the most foreign: it is almost only human or mouse, almost never a custom species. Registering a
#: custom species means taking everything except DNA, and with DNA first that is one selection from the end
#: of the line rather than a re-type around a gap in the middle.
ANALYSIS_ORDER = ("DNA", "RNA", "scRNA", "snATAC", "ATAC")

#: The order resource keys are ASKED in, independent of the order the analyses
#: were named. Someone who picks every analysis should still be asked for the
#: reference before ANNOVAR's protocol list, whatever sequence they typed.
KEY_ORDER = (
    "reference_fasta", "gtf",
    "star_index",
    "motif_database", "motif_species",
    "annovar_dir", "annovar_db", "annovar_buildver",
    "annovar_protocol", "annovar_operation",
    "snpeff_jar", "snpeff_config", "snpeff_genome",
    # Last, and never asked (no analysis lists it); built instead.
    "burden_reference",
)


def keys_for_analyses(analyses: Iterable[str]) -> List[str]:
    """The resource keys those analyses resolve, in a fixed order.

    Ordered rather than a set so the walkthrough asks in a stable sequence and
    two runs of it produce the same conversation -- and ordered by KEY_ORDER
    rather than by the order the analyses were named, so the sequence does not
    change because someone typed "ATAC, RNA" instead of "RNA, ATAC".
    """
    wanted = set(UNIVERSAL_KEYS)
    for analysis in analyses:
        wanted.update(ANALYSIS_KEYS.get(analysis, ()))
    return [key for key in KEY_ORDER if key in wanted]


# ---------------------------------------------------------------------------
# Where the machine registry lives
# ---------------------------------------------------------------------------


def default_site_registry_path() -> Path:
    """Beside the install, never inside the package.

    ``install_root()`` is discovered from ``sys.executable`` rather than
    recorded: no install path is hardcoded in ``src/`` -- a package that knows its own location cannot be installed
    twice, and every version is installed into a prefix of its own.

    THE DIRECTORY NAME COMES FROM THE CONFIG DIRECTORY (``DEFAULT_CONFIG_DIRNAME``,
    ".paralome"), not from the package name: every installed version shares
    one genome registry, and this is the machine-wide half of it.

    NOTE this default usually needs one root action to become usable. It lands
    under a root-owned prefix, and no amount of arranging makes an unprivileged
    process able to create or chmod something there. ``setup_hint`` says what
    to do about it.
    """
    from paralome.binaries import install_root
    from paralome.config.user import DEFAULT_CONFIG_DIRNAME

    return (
        install_root() / "share" / DEFAULT_CONFIG_DIRNAME.lstrip(".")
        / SITE_REGISTRY_DIRNAME
    )


def site_registry_path(config: Optional[Mapping[str, Any]] = None) -> Path:
    """The shared registry DIRECTORY. Environment, then user config, then default.

    The two overrides exist because the default frequently needs root to set
    up. On a cluster the install prefix is root-owned, so the registry has to
    move somewhere the site already owns -- and that is a property of the
    machine, not something this package can guess.
    """
    from paralome.setup.site import site_registry

    # The machine's decision outranks everything -- see
    # genome_repository above for why a per-user override is not wanted here.
    decided = site_registry()
    if decided is not None:
        return decided
    override = os.environ.get(ENV_SITE_REGISTRY)
    if override and override.strip():
        return Path(override).expanduser().resolve()
    if config is None:
        try:
            config = load_config(initialize=False)
        except ParalomeError:
            # A key nothing reads is refused by name, here as
            # everywhere; only a configuration that cannot be read is set aside.
            raise
        except Exception:  # pragma: no cover - unreadable config is not fatal here
            config = {}
    recorded = (config or {}).get(USER_CONFIG_SITE_KEY)
    if isinstance(recorded, str) and recorded.strip():
        return Path(recorded).expanduser().resolve()
    return default_site_registry_path()


def user_registry_file(directory: Optional[Path] = None, user: Optional[str] = None) -> Path:
    """This account's own drop-in file. The only file this user ever writes."""
    target = Path(directory) if directory is not None else site_registry_path()
    return target / f"{user or _current_user()}.json"


def ensure_registry_dir(directory: Path) -> Path:
    """Create the shared directory world-writable, if we are the one creating it.

    The mode is set with an explicit chmod rather than passed to mkdir, because
    mkdir's mode is masked by the process umask -- which on nearly every system
    would strip exactly the group and other write bits that make the directory
    shared at all.

    Only the directory WE create is chmodded. An existing one is left alone: a
    sysadmin may have set it up deliberately, and silently widening someone
    else's permissions is not this command's business.

    Raises OSError when the parent is not writable, which is the /usr/share
    case and is handled by the caller rather than here.
    """
    directory = Path(directory)
    if directory.is_dir():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, SHARED_DIR_MODE)
    except OSError:
        # Created but not shareable -- on a filesystem that refuses the sticky
        # bit, say. Better than failing: the registry works for this user and
        # `setup genome list` will show others cannot reach it.
        pass
    return directory


def setup_hint(directory: Optional[Path] = None) -> str:
    """What to do when the shared registry cannot be written.

    This replaces printing a bare errno. `[Errno 13] Permission denied:
    '/usr/share/paralome'` says what happened and nothing about what to do,
    and the thing to do is not guessable -- it depends on whether the person
    has root, which Paralome cannot know.
    """
    target = Path(directory) if directory is not None else site_registry_path()
    return (
        f"The shared genome registry is {target}, and you cannot write there.\n"
        "\n"
        "Anything registered stays in your own configuration and works for you;\n"
        "it is only invisible to everyone else on this machine.\n"
        "\n"
        "Either ask an administrator for, once and for all users:\n"
        f"    sudo mkdir -p {target}\n"
        f"    sudo chmod 1777 {target}\n"
        "\n"
        "or have an administrator run `paralome setup site`, which decides the\n"
        "registry's location for the whole machine.\n"
        "\n"
        # The routes that exist.
        "Without an administrator, point every account at a directory you own:\n"
        f"    export {ENV_SITE_REGISTRY}=/shared/path/genomes.d\n"
        f"or the key \"{USER_CONFIG_SITE_KEY}\" in each account's configuration.\n"
        "Paralome creates that directory world-writable when it first writes there."
    )


#: Below this, an account is a daemon rather than a person.
#:
#: 1000 on Linux, where anything under it is root, bin, systemd and friends.
#: 500 on macOS, where the first real user is 501 and the system accounts are
#: the underscore-prefixed ones below it. Hardcoding 1000 excluded EVERY user
#: on a Mac, so discovery there silently found nothing -- a wrong answer that
#: looks exactly like a correct one, since "no colleagues registered anything"
#: is a perfectly ordinary result.
FIRST_HUMAN_UID = 500 if sys.platform == "darwin" else 1000

#: Bounds the scan. A login node with a directory service can enumerate
#: thousands of accounts, and stat-ing every one of their home directories to
#: find five genomes is not a trade worth making. Sorted by uid first, so the
#: cut is at least deterministic rather than whatever order NSS replied in.
MAX_ACCOUNTS_SCANNED = 500


def resource_state(value: object, kind: str = "file") -> str:
    """"ok", "unreadable" or "missing" for one genome resource.

    EXISTENCE IS NOT ACCESS. ``Path.exists()`` needs only traversal of the
    parent directories; opening the file needs the read bit on the file
    itself. So a reference FASTA at mode 600 inside a traversable directory
    exists for everybody and is readable by its owner alone; a colleague
    adopting that genome would get a permission error on a compute node hours
    after registration, the listing and the doctor had all said it was fine.

    A directory needs execute as well as read: without ``x`` you cannot open
    anything inside it, which for a STAR index is the whole point of naming it.

    Three states rather than a boolean, because "there but not yours to read"
    and "not there" call for completely different fixes -- a chmod by its owner
    against rebuilding or re-registering the path.
    """
    if not value:
        return "missing"
    path = Path(str(value)).expanduser()
    try:
        if kind == "directory":
            if not path.is_dir():
                return "missing"
            return "ok" if os.access(path, os.R_OK | os.X_OK) else "unreadable"
        if not path.is_file():
            return "missing"
        return "ok" if os.access(path, os.R_OK) else "unreadable"
    except OSError:
        # A parent we cannot traverse raises rather than returning False. From
        # here that is indistinguishable from the file being absent, and the
        # remedy is the same conversation with whoever owns the path.
        return "missing"


def _readable_file(path: Path) -> bool:
    """Does this exist and may we read it -- answered without raising.

    ``Path.is_file()`` raises PermissionError when a parent directory is not
    traversable, which is the ordinary state of a home directory at mode 700.
    Discovery walks every account on the machine, so on any cluster where some
    homes are private the bare is_file() call turns a routine `setup genome
    list` into a traceback. Caught here rather than around each caller, because
    "not readable" and "not there" are the same answer for our purposes: the
    genome is unavailable either way.
    """
    try:
        return path.is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def peer_config_paths() -> List[Tuple[str, Path]]:
    """Other accounts' Paralome configs, as (account, path), if they exist.

    The passwd file beats the alternative: it KNOWS which directories are home
    directories. Scanning the parent of your own home only guesses, and on a
    layout like /shared/home/projects/<user> the siblings may be project
    folders belonging to nobody.

    ``pwd.getpwall`` rather than reading /etc/passwd directly: same data, plus
    whatever NSS adds, and it is what ``getent passwd`` reports. The catch is
    that clusters backed by LDAP or SSSD normally disable enumeration, so
    getpwall returns only local accounts -- which is why the parent-of-home
    scan stays as a supplement rather than being replaced.

    NOTHING IS OPENED HERE, and nothing anywhere in this module writes outside
    the current user's own home. Discovery is read-only by construction, and
    the filesystem decides what is readable: a home directory at mode 700 or a
    config at 600 simply yields nothing, which is the correct answer and needs
    no mechanism of ours to enforce.
    """
    from paralome.config.user import CONFIG_FILENAME, DEFAULT_CONFIG_DIRNAME

    me = _current_user()
    found: Dict[str, Path] = {}

    # `pwd` IS UNIX ONLY. On Windows the import raises ImportError, and an
    # unguarded one here would turn `paralome setup genome` into a traceback
    # on a platform where the command otherwise works fine -- pip installs the
    # wheel happily, so the failure would appear at first use rather than at
    # install. Windows falls through to the sibling scan, which uses nothing
    # platform-specific.
    try:
        import pwd
    except ImportError:  # pragma: no cover - exercised by the Windows test
        accounts = []
    else:
        try:
            accounts = sorted(pwd.getpwall(), key=lambda entry: entry.pw_uid)
        except Exception:  # pragma: no cover - no passwd database at all
            accounts = []
    for account in accounts[:MAX_ACCOUNTS_SCANNED]:
        if account.pw_uid < FIRST_HUMAN_UID or account.pw_name == me:
            continue
        candidate = Path(account.pw_dir) / DEFAULT_CONFIG_DIRNAME / CONFIG_FILENAME
        if _readable_file(candidate):
            found[account.pw_name] = candidate

    # Supplement, for the accounts a non-enumerating directory service hides.
    # The home's parent is where colleagues live on a shared cluster, and a
    # directory that holds `.paralome/config.json` is somebody's home whatever
    # passwd was willing to say.
    try:
        siblings = sorted(Path.home().parent.glob(
            f"*/{DEFAULT_CONFIG_DIRNAME}/{CONFIG_FILENAME}"
        ))
    except OSError:  # pragma: no cover - unreadable parent is not an error
        siblings = []
    for candidate in siblings[:MAX_ACCOUNTS_SCANNED]:
        account = candidate.parent.parent.name
        if account != me:
            found.setdefault(account, candidate)

    return sorted(found.items())


def discovered_genomes() -> Dict[str, Dict[str, Any]]:
    """Genomes registered by other accounts on this machine, by genome name.

    The crowd-sourced master list. Everybody already writes their own config;
    the union of those configs IS the registry, computed when it is read. No
    root, no common group, no 1777 directory, no environment variable, and no
    setup step -- because there is nothing to set up when nothing is ever
    written outside your own home.

    Each entry carries ``discovered_from``, the account it was read from, so
    the provenance survives into anything that displays or adopts it. An
    unreadable or malformed config is skipped without comment: one colleague
    with a broken file must not cost everyone else their discovery.

    On a name several accounts hold, the REGISTRANT'S own entry wins over any
    copy, so an adopter's clean copy cannot hide the owner's withdrawal.
    Between two copies, or two owners, the first account by name wins; sync
    surfaces the disagreement.
    """
    found: Dict[str, Dict[str, Any]] = {}
    authoritative: Dict[str, bool] = {}
    for account, path in peer_config_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        genomes = payload.get("genomes") if isinstance(payload, dict) else None
        if not isinstance(genomes, dict):
            continue
        for name, entry in genomes.items():
            if not isinstance(entry, dict):
                continue
            owner = str(entry.get("registered_by") or "")
            owns_it = not owner or owner == account
            if name in found and (authoritative[name] or not owns_it):
                continue
            found[name] = dict(entry, discovered_from=account)
            authoritative[name] = owns_it
    return found


def registry_files(directory: Optional[Path] = None) -> List[Path]:
    """Every drop-in file, in a deterministic order, oldest layout last.

    Sorted so two accounts reading the same directory merge it the same way --
    a registry whose contents depend on filesystem iteration order would give
    two people different answers for the same genome name.
    """
    target = Path(directory) if directory is not None else site_registry_path()
    files = sorted(target.glob("*.json")) if target.is_dir() else []
    # The pre-drop-in single file, if it is still there. Read last so a
    # per-user file wins over the layout it replaced.
    legacy = target.parent / LEGACY_SITE_REGISTRY_FILENAME
    if legacy.is_file():
        files.append(legacy)
    return files


def unreadable_registry_paths(directory: Optional[Path] = None) -> List[str]:
    """This machine's registry and genome repository, where this account cannot read them.

    Every reader here skips what it cannot open, so without this a permission
    problem would look exactly like an empty machine. Each entry is the path
    and the error, for the message that has to say so.
    """
    from paralome.setup.site import reporting_on_the_site_file, site_genome_repository

    target = Path(directory) if directory is not None else site_registry_path()
    problems: List[str] = []

    def denied(path: Path, exc: OSError) -> None:
        problems.append(f"{path} ({exc.strerror or exc})")

    try:
        names = sorted(os.listdir(target))
    except FileNotFoundError:
        names = []
    except PermissionError as exc:
        denied(target, exc)
        names = []
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(target / name, "rb"):
                pass
        except PermissionError as exc:
            denied(target / name, exc)
        except OSError:
            continue
    with reporting_on_the_site_file():
        repository = site_genome_repository()
    if repository is not None:
        try:
            os.listdir(repository)
        except PermissionError as exc:
            denied(repository, exc)
        except OSError:
            pass
    return problems


def registry_permission_text(directory: Optional[Path] = None) -> Optional[str]:
    """What a genome lookup says when part of the registry was unreadable, else None."""
    from paralome.setup.site import not_in_shared_group_text, shared_group_membership

    problems = unreadable_registry_paths(directory)
    if not problems:
        return None
    text = ("Permission denied reading this machine's genome registry, so genomes registered "
            "there cannot be seen from this account: " + "; ".join(problems) + ".")
    membership = shared_group_membership()
    if membership is not None and not membership["member"]:
        return (f"{text} Ask the administrator to set the permissions: "
                f"{not_in_shared_group_text(membership)}.")
    return (f"{text} Ask the administrator to set the permissions: the registry's files "
            "readable by every account that uses Paralome, and its directories searchable.")


def load_site_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    """Every genome registered on this machine, merged across all drop-in files.

    A missing, unreadable or malformed FILE is skipped rather than raised on,
    and the rest are still returned. That is the main practical advantage of
    the drop-in layout: one person writing a bad file does not take everyone
    else's genomes down with it. `paralome setup genome sync` is where a broken
    file is meant to become visible.

    THE OWNER'S FILE WINS, not the lexically first one. Two accounts can
    publish the same name (an adopter who withdraws their own copy republishes
    it), and if the machine's answer came from whichever publisher's username
    sorted first, an adopter's clean copy could become the machine's answer
    and the owner's withdrawal would reach nobody.

    ``registered_by`` survives adoption verbatim -- ``stamp`` uses setdefault on
    a copy -- so a file publishing an entry it did not register is identifiable
    by comparing the entry's registrant to the file's own name. That entry is a
    COPY, and a copy never outranks the original. Between two copies, or two
    entries with no registrant at all, the first wins.
    """
    merged: Dict[str, Any] = {}
    authoritative: Dict[str, bool] = {}
    for candidate in registry_files(path):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        genomes = payload.get("genomes") if isinstance(payload, dict) else None
        if not isinstance(genomes, dict):
            continue
        for name, entry in genomes.items():
            if not isinstance(entry, Mapping):
                continue
            owner = str(entry.get("registered_by") or "")
            # No registrant recorded means a pre-stamp entry, not an imposter.
            owns_it = not owner or owner == candidate.stem
            if name not in merged:
                merged[name] = entry
                authoritative[name] = owns_it
            elif owns_it and not authoritative.get(name, False):
                merged[name] = entry
                authoritative[name] = True
    return merged


def site_registry_is_writable(path: Optional[Path] = None) -> bool:
    """Whether this user could publish, without publishing to find out.

    Checks the file we would write, then the directory, then -- when neither
    exists yet -- whether we could create the directory at all. The last case
    is the /usr/share one, and answering it correctly is what lets the command
    offer a fix instead of a traceback.
    """
    target = Path(path) if path is not None else site_registry_path()
    own = user_registry_file(target)
    if own.exists():
        return os.access(own, os.W_OK)
    if target.is_dir():
        return os.access(target, os.W_OK)
    parent = target.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def save_site_registry(genomes: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    """Write THIS USER'S drop-in file. Raises OSError if this user may not.

    Only ever this account's own file, which is what makes a 1777 directory
    workable: the sticky bit stops you touching anyone else's, and you never
    need to.

    The caller is expected to catch the OSError and carry on -- a registration
    that cannot reach the shared directory still succeeds locally. Failing the
    whole command would block precisely the users without write permission on
    a shared install, who are the ones a machine-wide registry exists to help.
    """
    directory = Path(path) if path is not None else site_registry_path()
    ensure_registry_dir(directory)
    target = user_registry_file(directory)
    payload = {
        "schema": REGISTRY_SCHEMA,
        "registered_by": _current_user(),
        "genomes": dict(sorted(genomes.items())),
    }
    # Written whole, then moved into place, so a reader never sees a partial
    # file. The rename is safe under the sticky bit because the target is ours.
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    try:
        os.chmod(target, 0o644)
    except OSError:  # pragma: no cover - not fatal, the file is already written
        pass
    return target


def my_registered_genomes(path: Optional[Path] = None) -> Dict[str, Any]:
    """Just this account's drop-in file, which is the only one we may rewrite."""
    target = user_registry_file(Path(path) if path is not None else None)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # An unreadable or corrupt file is not an empty one; rewriting it
        # from {} would drop every other genome in it. Callers treat OSError as "not written".
        raise OSError(f"cannot read your registry file {target}: {exc}") from exc
    genomes = payload.get("genomes") if isinstance(payload, dict) else None
    return genomes if isinstance(genomes, dict) else {}


# ---------------------------------------------------------------------------
# Comparing what the user has against what the machine has
# ---------------------------------------------------------------------------


def definition_of(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The genome itself, with the bookkeeping stripped.

    Two entries naming identical files are the same genome even when they were
    registered by different people on different days.
    """
    return {
        key: value for key, value in entry.items()
        if key not in PROVENANCE_KEYS
    }


def differing_keys(first: Mapping[str, Any], second: Mapping[str, Any]) -> List[str]:
    """Which keys two genome definitions disagree on. Empty means identical."""
    left, right = definition_of(first), definition_of(second)
    return sorted(
        key for key in set(left) | set(right)
        if str(left.get(key, "")) != str(right.get(key, ""))
    )


class Reconciliation:
    """What syncing would do, computed before anything is written.

    Four outcomes, and they need different handling:

    ``only_user``   the user has it, the machine does not -- publish upward
                    (genomes already built, donated to the master list).
    ``only_site``   the machine has it, the user does not -- adopt downward.
                    How a new account picks up everything already built.
    ``identical``   both have it, same paths. Nothing to do, and worth saying
                    so rather than leaving it unmentioned.
    ``conflicting`` same name, different files.
                    Never resolved automatically -- silently overwriting a
                    shared entry with one person's local override is the
                    failure that would be hardest to trace afterwards.
    """

    def __init__(
        self,
        user_genomes: Mapping[str, Any],
        site_genomes: Mapping[str, Any],
        declined: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.only_user: List[str] = []
        self.only_site: List[str] = []
        self.identical: List[str] = []
        self.conflicting: Dict[str, List[str]] = {}
        #: On the machine, and this account has already said no. Reported so it
        #: is visible, never offered. See ``declined_genomes``.
        self.declined: List[str] = []
        said_no = set(declined or ())
        for name in sorted(set(user_genomes) | set(site_genomes)):
            in_user, in_site = name in user_genomes, name in site_genomes
            if in_user and not in_site:
                self.only_user.append(name)
            elif in_site and not in_user:
                (self.declined if name in said_no else self.only_site).append(name)
            else:
                differences = differing_keys(user_genomes[name], site_genomes[name])
                if differences:
                    self.conflicting[name] = differences
                else:
                    self.identical.append(name)

    @property
    def nothing_to_do(self) -> bool:
        return not (self.only_user or self.only_site or self.conflicting)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "only_user": list(self.only_user),
            "only_site": list(self.only_site),
            "identical": list(self.identical),
            "conflicting": {k: list(v) for k, v in self.conflicting.items()},
        }


def reconcile(
    user_genomes: Mapping[str, Any],
    site_genomes: Mapping[str, Any],
    declined: Optional[Mapping[str, Any]] = None,
) -> Reconciliation:
    return Reconciliation(user_genomes, site_genomes, declined)


# ---------------------------------------------------------------------------
# Writing a genome into both places
# ---------------------------------------------------------------------------


def stamp(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Add who registered this and when.

    Every registration is published to the shared registry silently, rather
    than behind a prompt or a flag. These two fields are what make that safe
    to live with: an entry pointing somewhere it should not can be traced to a
    person and a date instead of being anonymous.
    """
    stamped = dict(entry)
    stamped.setdefault("registered_by", _current_user())
    stamped.setdefault(
        "registered_at", datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    return stamped


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - no account name available
        return "unknown"


#: Paralome's own directory and config, once a genome has been registered in them. World READABLE,
#: owner-writable only.
DISCOVERABLE_DIR_MODE = 0o755
DISCOVERABLE_FILE_MODE = 0o644


def blocking_ancestors(path: Optional[Path] = None) -> List[Path]:
    """Every directory on the way to ``path`` that others cannot traverse.

    Permissions compose down the path, so every link of the chain is checked,
    not only the home. A deep home layout is where that matters. Accounts may
    live at

        /shared/home/projects/<user>

    so reaching one config needs traversal of four directories, and any single
    one of them at 0750 hides every account from every other at once.

    Returned as a list, innermost last, because the fix is per-directory and
    one of them is likely not the user's to change: a home they can chmod
    themselves, a shared parent they must ask an administrator about. Saying
    which is which is the difference between an instruction and a complaint.

    The root directory is not included -- if / is not traversable, nothing on
    the machine works and Paralome is not the right messenger.
    """
    target = (path or Path.home()).resolve()
    blocked: List[Path] = []
    for ancestor in reversed(target.parents):
        if ancestor == Path(ancestor.root):
            continue
        try:
            if not ancestor.stat().st_mode & 0o001:
                blocked.append(ancestor)
        except OSError:
            # Cannot stat it, which from here is indistinguishable from not
            # being able to traverse it, and is reported the same way.
            blocked.append(ancestor)
    try:
        if not target.stat().st_mode & 0o001:
            blocked.append(target)
    except OSError:  # pragma: no cover - no readable home
        blocked.append(target)
    return blocked


def make_config_discoverable() -> Dict[str, Any]:
    """Let other accounts read this one's genome list. Never let them write it.

    Only Paralome's own directory and its own config file are touched, and only
    ever to ADD the read bit -- the owner's permissions are preserved and no
    write bit is granted to anyone. Registering a genome is the act of wanting
    it shared, so it is the moment to make it visible; running `paralome setup`
    is not, which is why this is not in save_config.

    Best effort throughout. A filesystem that refuses the chmod, an ACL-managed
    home, a config owned by someone else -- none of that should fail a
    registration that has otherwise succeeded.
    """
    blocked = blocking_ancestors()
    outcome: Dict[str, Any] = {
        "config_dir": None, "config_file": None,
        "home_traversable": not blocked,
        # Named individually so the report can tell the user which of these is
        # theirs to fix and which needs an administrator.
        "blocked_by": [str(item) for item in blocked],
    }
    for path, mode, key in (
        (config_path().parent, DISCOVERABLE_DIR_MODE, "config_dir"),
        (config_path(), DISCOVERABLE_FILE_MODE, "config_file"),
    ):
        try:
            current = path.stat().st_mode & 0o777
            # Add the read bits, keep everything the owner already had. A
            # user who deliberately widened or narrowed something else of
            # theirs keeps that choice.
            os.chmod(path, current | (mode & 0o055))
            outcome[key] = str(path)
        except OSError:
            continue
    return outcome


#: Keys a Paralome build step writes into an EXISTING entry.
DERIVED_KEYS = ("burden_reference",)


def record_derived_key(
    name: str,
    key: str,
    value: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Write one built resource into a genome this account already has.

    For `setup genome burden-reference`. Not a way in: a name with
    no entry here is left alone (returns None), and only DERIVED_KEYS may be
    written, so nothing about what the genome IS changes. Published to the
    machine registry only when the entry is this account's own -- an adopted
    genome is its registrant's to publish.
    """
    if key not in DERIVED_KEYS:
        raise ValueError(f"{key} is not a key a build step records")
    cfg = config if config is not None else load_config(initialize=True)
    genomes = cfg.get("genomes", {}) or {}
    if name not in genomes:
        return None
    entry = dict(genomes[name])
    entry[key] = value
    return register_genome(
        name, entry, config=cfg, registry_path=registry_path, publish=is_mine(entry),
    )


def record_annovar_table(
    name: str,
    table: str,
    operation: str = "f",
    *,
    config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Add one ANNOVAR table to the protocol of a genome this account already has.

    For `setup genome annovar-table`. Like record_derived_key, not a
    way in: a name with no entry here is left alone (None). The table goes at
    the end of annovar_protocol and its operation at the end of
    annovar_operation; a table already there changes nothing. Published to the
    machine registry only when the entry is this account's own.
    """
    from paralome.setup.annovar_tables import with_protocol

    cfg = config if config is not None else load_config(initialize=True)
    genomes = cfg.get("genomes", {}) or {}
    if name not in genomes:
        return None
    entry = with_protocol(genomes[name], table, operation)
    return register_genome(
        name, entry, config=cfg, registry_path=registry_path, publish=is_mine(entry),
    )


def register_genome(
    name: str,
    entry: Mapping[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    registry_path: Optional[Path] = None,
    publish: bool = True,
) -> Dict[str, Any]:
    """Write one genome to the user's config and, when possible, the machine registry.

    Returns what happened, rather than printing it, so the command layer owns
    presentation and the tests do not have to read stdout.

    The shared write is best-effort by design. If it fails the registration
    still stands locally and the caller reports that the machine copy was not
    updated -- see ``save_site_registry`` for why this must not be fatal.
    """
    stamped = stamp(entry)

    cfg = config if config is not None else load_config(initialize=True)
    genomes = cfg.setdefault("genomes", {})
    genomes[name] = dict(stamped)
    save_config(cfg)

    outcome: Dict[str, Any] = {
        "genome": name,
        "user_config": str(config_path()),
        "site_registry": None,
        "published": False,
        "publish_error": None,
        # Registering is the act of wanting a genome shared, so this is where
        # the config becomes readable by the accounts that would discover it.
        "discoverable": make_config_discoverable(),
        # And the genome's folder in the shared repository open to the repository's group, so any member can
        # add to it later (an index built on a first run: bwa-mem2/ beside the FASTA).
        "group_writable": share_genome_folders(stamped),
    }
    if not publish:
        return outcome

    target = Path(registry_path) if registry_path is not None else site_registry_path(cfg)
    outcome["site_registry"] = str(target)
    try:
        # THIS USER'S file, not the merged registry. Rewriting the merge would
        # copy everyone else's genomes into our drop-in, so a later correction
        # by their owner would be shadowed by our stale copy of it -- and the
        # sticky directory would not stop us, because we would be writing our
        # own file the whole time.
        mine = my_registered_genomes(target)
        mine[name] = dict(stamped)
        outcome["site_registry_file"] = str(save_site_registry(mine, target))
        outcome["published"] = True
    except OSError as exc:
        outcome["publish_error"] = str(exc)
        outcome["setup_hint"] = setup_hint(target)
    return outcome


def share_genome_folders(entry: Mapping[str, Any]) -> List[str]:
    """Give the group write (and setgid) on each folder of the shared repository this entry's files live in.

    Only folders this account owns, and only ever adding bits: a genome registered from an account with the
    usual 022 umask left its folder group read-only, and the first BWA run of another account in the group
    could not create the index beside the FASTA. Best effort: returns the folders changed.
    """
    import stat

    try:
        from paralome.setup.site import site_genome_repository

        repository = site_genome_repository()
    except Exception:  # noqa: BLE001 -- no shared repository on this machine
        repository = None
    if repository is None:
        return []
    root = Path(repository).resolve()
    changed: List[str] = []
    folders = set()
    for value in _recorded_paths(entry):
        try:
            relative = Path(value).resolve().relative_to(root)
        except (ValueError, OSError):
            continue
        if relative.parts:
            folders.add(root / relative.parts[0])
    for folder in sorted(folders):
        try:
            info = folder.stat()
            if not folder.is_dir() or info.st_uid != os.getuid():
                continue
            wanted = info.st_mode | stat.S_IWGRP | stat.S_ISGID
            if wanted != info.st_mode:
                os.chmod(folder, stat.S_IMODE(wanted))
                changed.append(str(folder))
        except OSError:
            continue
    return changed


def _recorded_paths(entry: Mapping[str, Any]) -> List[str]:
    """Every absolute path an entry records, at any depth."""
    found: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str) and value.startswith("/"):
            found.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(entry)
    return found


def refresh_adopted_genome(name: str) -> Optional[str]:
    """Take the machine's current entry for an adopted genome whose recorded files are gone.

    The shared genomes may be moved and the machine registry updated, while an
    account's adopted copy still names the old place. Only an ADOPTED copy is
    refreshed: one this account registered is its own to move. The machine's
    entry is taken only when every path it records exists. Returns what was
    done, for the preflight to say; None when nothing needed doing or nothing
    could be done (the preflight then reports the paths).
    """
    from paralome.config.user import load_config, save_config

    config = load_config(initialize=False)
    genomes = config.get("genomes") or {}
    entry = genomes.get(name)
    if not isinstance(entry, Mapping) or is_mine(entry):
        return None
    gone = [path for path in _recorded_paths(entry) if not Path(path).exists()]
    if not gone:
        return None
    available = dict(discovered_genomes())
    available.update(load_site_registry())
    current = available.get(name)
    if not isinstance(current, Mapping) or registered_by(current) != registered_by(entry):
        return None
    if not _recorded_paths(current) or any(not Path(p).exists() for p in _recorded_paths(current)):
        return None
    genomes[name] = dict(current)
    config["genomes"] = genomes
    save_config(config)
    return (f"{name}: your copy named {len(gone)} file(s) that are gone (e.g. {gone[0]}); took the machine "
            f"registry's current entry, whose files are all there, and this run uses it")
