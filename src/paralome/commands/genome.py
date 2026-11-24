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

"""`paralome setup genome` -- register a genome, and reconcile with the machine.

The first interactive command in Paralome. Everything else here is
built to run unattended under SLURM, which is why this one has to be careful
about the terminal: a prompt issued into a batch job's closed stdin does not
error, it hangs, and the job burns its wall clock waiting for an answer nobody
can give. So the prompting path checks for a TTY first and refuses with an
explanation naming the flags that would have worked instead.

NO SCANNING. An earlier design offered to search a directory for a FASTA, a
GTF, a STAR index. Paralome cannot know where to scan: reference bundles
on a cluster are scattered across group shares by whoever built them, and a
guess that lands on the wrong GTF is worse than a prompt, because it gets
accepted without anyone reading it.

WHAT IT ASKS FOR is decided by what will be run against the genome, not by the
resolver's full vocabulary. `RESOURCE_KEYS` accepts fourteen keys; a scRNA
genome needs three. Walking someone through ANNOVAR's protocol/operation pairs
to register a genome for single-cell RNA would be the kind of thing people
learn to press enter through.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import typer

from paralome.config.user import load_config
from paralome.setup.genome_names import derive_genome_name
from paralome.setup.genomes import (
    ANALYSIS_ORDER,
    IDENTITY_KEYS,
    KEY_PROMPTS,
    Reconciliation,
    definition_of,
    load_site_registry,
    reconcile,
    register_genome,
    save_site_registry,
    site_registry_is_writable,
    site_registry_path,
)

genome_app = typer.Typer(
    help="Register genomes and keep this machine's registry in step.",
    no_args_is_help=False,
)


def _interactive() -> bool:
    """A real terminal on both ends.

    stdin alone is not enough: a job with input redirected from /dev/null has a
    readable stdin that answers every prompt with EOF, which typer turns into an
    abort part-way through a registration.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def _require_terminal(what: str, *, hint: str) -> None:
    """Refuse rather than prompt when nobody can answer.

    The hint is per-caller because the alternatives differ: a registration can
    be driven entirely by flags, a sync cannot -- it is a conversation about
    specific genomes by design.
    """
    if _interactive():
        return

    # NUMBERED, like every other refusal: the code + one-line fix + log file every other failure
    # produces, and something for `paralome explain` to say. The 0100 block is "what the user typed --
    # CLI parameters, before any work starts", and a refusal to prompt unattended is exactly that.
    from paralome.errors import ParalomeError

    raise ParalomeError(115, f"{what} needs a terminal.\n{hint}")


def _describe(name: str, entry: Mapping[str, Any]) -> str:
    """One line per genome: what it is, and whether its files are here.

    The path check is the point. A registry entry is only ever as good as the
    filesystem it names, and a genome registered on one machine and copied to
    another looks identical in JSON while being entirely wrong.
    """
    from paralome.setup.genomes import resource_state
    from paralome.setup.withdrawal import describe as describe_withdrawal
    from paralome.setup.withdrawal import effective_withdrawal, is_modified

    # A retired genome says so here, on its own line, before the
    # file tally. A listing that reports "6 of 6 files readable" and nothing
    # else is the exact reassurance a withdrawn genome must not give.
    retired = effective_withdrawal(name, entry)
    banner = f"\n      ** {describe_withdrawal(name, retired)}" if retired else ""

    definition = definition_of(entry)
    paths = [
        (key, value, KEY_PROMPTS.get(key, ("", "text", None))[1])
        for key, value in definition.items()
        if KEY_PROMPTS.get(key, ("", "text", None))[1] in ("file", "directory")
    ]
    identity = " ".join(
        str(definition[key]) for key in IDENTITY_KEYS if definition.get(key)
    )
    # The annotation, not only the assembly. Two entries reading
    # "Homo sapiens GRCh38" that differ by thirty Ensembl releases looked
    # identical in this listing, which is the confusion the whole supersession
    # feature exists to remove.
    annotation = " ".join(
        str(definition[key])
        for key in ("annotation_source", "annotation_release")
        if definition.get(key) and str(definition[key]).lower() != "unknown"
    )
    if annotation:
        identity = f"{identity} / {annotation}" if identity else annotation
    suffix = f"  [{identity}]" if identity else ""
    # A user-modified genome says so everywhere it appears. Users add GFP and
    # mCherry to assemblies and GTFs, and somebody reading a gene table that
    # contains GFP needs this to be the first thing they can find out.
    if is_modified(definition):
        note = str(definition.get("modified_note") or "").strip()
        suffix += f"  MODIFIED{f' ({note})' if note else ''}"
    if not paths:
        return f"  {name}{suffix}  (no file paths recorded){banner}"

    # READABLE BY YOU, not merely present. A genome registered by a
    # colleague names files in their space, and existence is not access -- see
    # resource_state. Reporting "present" for a FASTA nobody but its owner can
    # open is how a run fails on a compute node after the registration, the
    # listing and the doctor all said it was fine.
    states = {key: resource_state(value, kind) for key, value, kind in paths}
    readable = sum(1 for state in states.values() if state == "ok")
    unreadable = sorted(k for k, v in states.items() if v == "unreadable")
    missing = sorted(k for k, v in states.items() if v == "missing")

    state = f"{readable} of {len(paths)} files readable"
    if missing:
        state += f"  missing: {', '.join(missing)}"
    if unreadable:
        state += f"  NOT READABLE BY YOU: {', '.join(unreadable)}"
    return f"  {name}{suffix}  {state}{banner}"


def _show_discovered(user_genomes: Mapping[str, Any]) -> Dict[str, Any]:
    """Genomes other accounts on this machine have registered.

    The master list is the union of everyone's own config (accounts found
    through the passwd file), read where the filesystem permits, rather than a
    shared file somebody needs write access to.

    Printed with the account each came from, deliberately. Absorbing a
    colleague's entry should be a thing you can see happening, not something
    that quietly appears in your list.
    """
    from paralome.setup.genomes import discovered_genomes

    found = {k: v for k, v in discovered_genomes().items() if k not in user_genomes}
    if not found:
        return {}
    accounts = sorted({str(entry.get("discovered_from")) for entry in found.values()})
    typer.echo(
        f"Registered by other accounts on this machine "
        f"({len(found)} in {len(accounts)}: {', '.join(accounts)}):"
    )
    for name in sorted(found):
        origin = found[name].get("discovered_from")
        typer.echo(f"{_describe(name, found[name])}   [{origin}]")
    typer.echo("")
    typer.echo("`paralome setup genome sync` adopts any of these without retyping them.")
    typer.echo("")
    return found


def _show_existing(user_genomes: Mapping[str, Any], site_genomes: Mapping[str, Any]) -> None:
    """What is already registered, before anyone builds another one.

    Shown first so a user can adopt a genome that is already built instead of
    building their own.
    """
    if user_genomes:
        typer.echo("Already in your configuration:")
        for name in sorted(user_genomes):
            typer.echo(_describe(name, user_genomes[name]))
        typer.echo("")
    unadopted = {k: v for k, v in site_genomes.items() if k not in user_genomes}
    if unadopted:
        typer.echo(f"Registered on this machine, not yet yours ({site_registry_path()}):")
        for name in sorted(unadopted):
            typer.echo(_describe(name, unadopted[name]))
        typer.echo("")
        typer.echo("`paralome setup genome sync` adopts any of these without retyping them.")
        typer.echo("")
    if not user_genomes and not site_genomes:
        typer.echo("No genomes are registered yet, here or on this machine.\n")


IDENTITY_PROMPTS = (
    ("species", "Species, e.g. Homo sapiens"),
    ("assembly", "Assembly, e.g. GRCh38"),
    ("annotation_source", "Annotation source (ensembl/gencode/refseq/custom)"),
    ("annotation_release", "Annotation release, e.g. 113 or v44"),
)


def _annovar_for_the_confirmed_assembly(entry: Dict[str, Any], folder: Path) -> None:
    """Set the entry's ANNOVAR build and protocol from its confirmed assembly.

    An empty build (several builds present, none known for the assembly) is
    left out of the entry: an annotation then refuses for want of one, naming
    the key, rather than annotating against the wrong build.
    """
    from paralome.setup.genome_scan import find_annovar

    findings = find_annovar(folder, entry.get("assembly"))
    if "annovar_buildver" not in findings:
        return
    before = entry.get("annovar_buildver")
    # The database directory follows the confirmed assembly too
    # (humandb/ and mousedb/ in one install).
    if findings.get("annovar_db") is not None and findings["annovar_db"].value:
        entry["annovar_db"] = findings["annovar_db"].value
    for key in ("annovar_buildver", "annovar_protocol", "annovar_operation"):
        found = findings.get(key)
        if found is not None and found.value:
            entry[key] = found.value
        else:
            entry.pop(key, None)
    after = entry.get("annovar_buildver")
    if after != before:
        typer.echo(
            f"ANNOVAR build for assembly {entry.get('assembly') or '?'}: "
            + (after if after else f"none recorded ({findings['annovar_buildver'].note})")
            + (f" (the scan proposed {before})" if before else "")
        )


def _ask_identity(
    *,
    supplied: Mapping[str, Optional[str]],
    scanned: Mapping[str, Any],
    modified: Optional[bool],
    modified_note: Optional[str],
    interactive: bool,
) -> Dict[str, Any]:
    """The five fields that decide what supersedes what.

    These are the key elements that decide supersession, so they are asked
    for explicitly.

    WHY ASKED AND NOT INFERRED. Inference from filenames is unreliable, and it
    fails SILENTLY -- a user who is offered no replacement cannot tell "nothing
    newer exists" from "your filenames did not parse", and those need opposite
    responses. Asking converts a silent gap into an answered question.

    UNKNOWN IS ACCEPTED, and recorded as ``unknown`` rather than left absent.
    A recorded unknown is a statement; a missing key is an accident. Both keep
    the genome out of supersession, but only one of them can be reported back
    to the user as the reason.

    THE MODIFICATION FLAG. Users add fluorescent (GFP, mCherry) genes to
    assemblies and GTFs, and those must be flagged as user-modified. Both
    halves of the bundle, which is why the question is about the genome rather
    than about the GTF: stock GRCh38 is not a replacement for a GFP-bearing one
    -- swapping would drop the transgene and the reads over it would simply
    stop mapping -- and a modified copy is not a replacement for anybody
    else's either.
    """
    from paralome.setup.genome_scan import ANNOTATION_SOURCES, UNKNOWN

    found: Dict[str, Any] = {}
    if interactive:
        typer.echo("")
        typer.echo("What this genome IS. Used to tell you when something newer exists.")
        typer.echo("Press enter to accept a scanned value; 'unknown' is a real answer.")

    for key, label in IDENTITY_PROMPTS:
        value = supplied.get(key)
        default = str(scanned.get(key) or "").strip()
        if value is None and interactive:
            value = typer.prompt(f"  {label}", default=default or UNKNOWN).strip()
        value = (value or default or UNKNOWN).strip()
        if key == "annotation_source":
            value = value.lower()
            if value not in ANNOTATION_SOURCES:
                # Not rejected: a source nobody anticipated is real, and
                # refusing the registration over it would be worse than
                # recording it. It simply will not order against anything.
                typer.echo(
                    f"    (recording '{value}' -- not one of "
                    f"{', '.join(ANNOTATION_SOURCES)}, so it will not be "
                    f"compared to other releases)"
                )
        found[key] = value

    if modified is None:
        modified = (
            typer.confirm(
                "  Have you modified these files? (added transgenes, filtered biotypes)",
                default=False,
            )
            if interactive
            else False
        )
    found["modified"] = bool(modified)
    if modified:
        note = modified_note
        if note is None and interactive:
            note = typer.prompt("    What did you change", default="").strip()
        if note:
            found["modified_note"] = note

    if interactive:
        unknowns = [k for k, _ in IDENTITY_PROMPTS if found.get(k) == UNKNOWN]
        if unknowns:
            typer.echo("")
            typer.echo(
                f"    Recorded as unknown: {', '.join(unknowns)}. This genome will "
                f"not be offered a newer build, and will not be offered as one."
            )
    return found


def _show_scan(result) -> None:
    """What a folder offered, before anything is written.

    Marked lines are the ones Paralome inferred rather than read. The
    distinction is the whole reason this prints instead of just registering:
    a GTF named by genomeParameters.txt is a fact the index recorded, while an
    ANNOVAR operation letter is a guess from a protocol name, and the person
    confirming deserves to see which is which.
    """
    from paralome.setup.genome_scan import analyses_for

    typer.echo(f"Scanning {result.folder}")
    typer.echo("")
    if not result.findings:
        typer.echo("  Nothing recognisable here.")
    for key in sorted(result.findings):
        found = result.findings[key]
        mark = " " if found.confident else "?"
        typer.echo(f" {mark} {key:18s} {found.value}")
        typer.echo(f"      {found.source}")
        if found.note:
            typer.echo(f"      {found.note}")
    if result.problems:
        typer.echo("")
        for problem in result.problems:
            typer.echo(f" ! {problem}")
    analyses = analyses_for(result)
    typer.echo("")
    typer.echo(f"  Equipped for: {', '.join(analyses) if analyses else '(nothing)'}")
    if any(not f.confident for f in result.findings.values()):
        typer.echo("  ? marks a value Paralome inferred rather than read. Check those.")
    typer.echo("")


@genome_app.callback(invoke_without_command=True)
def genome_command(ctx: typer.Context) -> None:
    """Genomes registered here, and the commands that change them.

    With no subcommand, prints the same listing as `setup genome list`.
    """
    # There is one registration path, `genome add`, so the shared-repository
    # rule (PLM-0409) and derived naming apply to every registration. Configs
    # that still carry the per-account site_registry/repository keys are still
    # read; `setup site` records those places for every account.
    if ctx.invoked_subcommand is not None:
        return
    genome_list_command()


def _report_visibility(outcome: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> None:
    """Why colleagues cannot see this genome -- and what to do ON THIS MACHINE.

    The advice is chosen by what kind of machine this is: on a real
    distributed system such as a university HPC, users should not browse
    unaffiliated users' homes. On a cluster the registry is the whole answer
    and the home is not mentioned as an option; on a lab server both are
    offered (naming the directory that blocks); on a single-user workstation
    there is nobody to be visible to and Paralome says so rather than inventing
    a problem. See paralome.setup.deployment.
    """
    from paralome.setup.deployment import (
        infer_deployment, may_open_own_home, sharing_advice,
    )

    # PUBLISHED MEANS VISIBLE, and nothing else needs saying. The registry is read by every account
    # without touching anybody's home, so once an entry is in it, whether this home is private is not
    # a question anyone needs answered (on a cluster nobody's home is traversable, by design).
    if outcome.get("published"):
        return

    discoverable = outcome.get("discoverable") or {}
    if discoverable.get("home_traversable", True):
        return

    deployment = infer_deployment(config)
    typer.echo("")
    typer.echo(f"Not visible to other accounts on this machine ({deployment}).")
    typer.echo("")
    for line in sharing_advice(deployment, site_registry_path()).splitlines():
        typer.echo(f"  {line}")

    # WHICH directory blocks is only worth printing when opening one is on the
    # table. On a cluster it is a fact the user can do nothing with, and
    # listing it would read as an invitation.
    if not may_open_own_home(deployment):
        return

    blocked = [Path(item) for item in discoverable.get("blocked_by") or []]
    if not blocked:
        return
    mine = [item for item in blocked if item == Path.home() or _owned_by_me(item)]
    theirs = [item for item in blocked if item not in mine]

    typer.echo("")
    typer.echo("  Permissions compose down the whole path, and these block it:")
    for item in blocked:
        typer.echo(f"      {item}  ({'yours' if item in mine else 'not yours to change'})")
    if mine:
        typer.echo("")
        typer.echo("  You could open yours with:")
        for item in mine:
            typer.echo(f"      chmod o+x {item}")
    if theirs:
        typer.echo("")
        typer.echo("  An administrator has to open the rest. Until then nothing below")
        typer.echo("  them is discoverable -- for you or for anyone else under them.")


def _owned_by_me(path: Path) -> bool:
    try:
        return path.stat().st_uid == os.getuid()
    except (OSError, AttributeError):  # pragma: no cover - Windows, or no stat
        return False


def _split(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _report_registration(
    name: str,
    outcome: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    """What happened, said once. Both registration paths print this.

    ``config`` reaches _report_visibility, which needs it to tell a cluster
    from a workstation. Optional rather than required because the value is
    advisory -- without it the deployment is inferred from the machine alone,
    which is right often enough that a caller lacking a config should still get
    a report rather than an error.
    """
    typer.echo("")
    typer.echo(f"Registered {name}.")
    typer.echo(f"  your config    : {outcome['user_config']}")
    if outcome.get("published"):
        typer.echo(f"  this machine   : {outcome['site_registry']}")
    else:
        typer.echo("  this machine   : NOT updated")
        typer.echo("")
        typer.echo(outcome.get("setup_hint") or str(outcome.get("publish_error")))
    _report_visibility(outcome, config)
    typer.echo("")
    typer.echo(f"Check it with:  paralome setup doctor {name}")


def _settle_a_rival_claim(name: str, mine: Mapping[str, Any], *, yes: bool) -> str:
    """A genome somebody else registered is theirs to publish, not yours.

    When a second account runs `genome add` on a genome another account has
    already registered, two drop-ins would both assert ownership of one name.
    ``load_site_registry`` resolves that first-authoritative-wins, so which
    entry becomes the machine's answer would depend on how the usernames sort,
    and a withdrawal by the real owner could reach nobody. This refuses the
    second registration instead.

    MATCHED ON IDENTITY, NOT ON TEXT. Names are derived from species, assembly
    and annotation, so two registrations of one genome arrive at one name by
    construction; the identity key is compared as well as the name, which
    catches entries that still carry hand-made names (`Human38` for the
    reference registered as `Human_GRCh38.e81`).

    There is no "register mine under another name" branch: a shared name
    means a shared species, assembly and annotation release, which is not a
    coincidence, it is the same genome.
    """
    from paralome.setup.genome_names import identity_key
    from paralome.setup.genomes import is_mine, registered_by

    registry = load_site_registry()
    theirs = registry.get(name)
    if theirs is None:
        wanted = identity_key(mine)
        if wanted:
            theirs = next(
                (item for item in registry.values() if identity_key(item) == wanted),
                None,
            )
    if theirs is None or is_mine(theirs):
        return name

    held = str(theirs.get("name") or name)
    typer.echo("")
    typer.echo(f"This genome is already registered on this machine, as {held}.")
    typer.echo(f"  Registered by : {registered_by(theirs) or 'unknown'}")
    typer.echo(f"  Reference     : {theirs.get('reference_fasta') or '(none recorded)'}")
    typer.echo("")
    typer.echo(
        "Registering it again publishes a SECOND entry for one genome, and which\n"
        "one this machine believes is then settled by whose username sorts first\n"
        "-- including whether a withdrawal by its owner is ever seen."
    )
    typer.echo("")

    if yes:
        typer.echo("--yes: adopting the existing registration rather than publishing a rival.")
        choice = "a"
    else:
        _require_terminal(
            "Settling a duplicate registration",
            hint=f"`paralome setup genome adopt {held}` takes the existing one.",
        )
        typer.echo("  [a] adopt theirs   [q] quit")
        choice = (typer.prompt("  Choice", default="a").strip().lower() or "a")[:1]

    if choice == "a":
        genome_adopt_command(held)
        raise typer.Exit(code=0)
    typer.echo("Nothing registered.")
    raise typer.Exit(code=1)


def _with_existence(value: Any) -> str:
    """A configured value, and whether it is actually there.

    `sync` prints the two sides of a conflict. After a repository migration
    every `yours` path can be stale and every `theirs` path real, and the
    default answer is [s] skip, which KEEPS the dead path -- so each path is
    marked when it is missing. A question whose right answer is visible on
    disk should not require the history to be remembered. Only paths are
    annotated: "(missing)" beside a species name or an ANNOVAR protocol list
    would be nonsense.
    """
    if value is None:
        return "(unset)"
    text = str(value)
    if not text.startswith("/") and not text.startswith("~"):
        return text
    try:
        there = Path(text).expanduser().exists()
    except OSError:
        return text
    return f"{text}  [{'present' if there else 'MISSING'}]"


def _require_the_shared_repository(folder: Path) -> None:
    """A genome on a shared machine belongs in the shared place.

    Genomes registered outside the shared place are refused. Otherwise people
    would register their own private copies that nobody else could use, and
    the machine would end up with a dozen human and a dozen mouse genomes.

    ONLY WHEN THE MACHINE HAS DECIDED. The test is site.json specifically, not
    the per-user ``genome_repository`` key -- that one is a convenience for
    typing bare names, and a person who set it for themselves has not thereby
    asked to be forbidden from registering anything outside it. An
    administrator running `setup site` has.

    So a workstation, a laptop, and anyone who merely recorded a favourite
    directory are all unaffected.
    """
    from paralome.errors import ParalomeError
    from paralome.setup.site import site_genome_repository

    repository = site_genome_repository()
    if repository is None:
        return
    try:
        folder.relative_to(repository)
        return
    except ValueError:
        pass
    raise ParalomeError(
        409,
        f"{folder}\nThis machine's genome repository is {repository}",
    )


def _refuse_to_replace_a_different_genome(
    name: str, entry: Dict[str, Any], config: Dict[str, Any]
) -> None:
    """A name already held by other files is not overwritten.

    Registering wrote ``genomes[name]`` whatever was there. Two
    folders whose names could not be derived from what they hold arrived at one
    name, and the second silently replaced the first in the config and the
    shared registry. Registering the SAME files again is still an update.
    """
    from paralome.pipeline.resource_catalog import PATH_RESOURCES

    genomes = config.get("genomes")
    existing = genomes.get(name) if isinstance(genomes, dict) else None
    if not isinstance(existing, dict):
        return
    differing = [
        key for key in PATH_RESOURCES
        if str(existing.get(key) or "").strip()
        and str(entry.get(key) or "").strip()
        and str(existing.get(key)).strip() != str(entry.get(key)).strip()
    ]
    if not differing:
        return
    held = "\n".join(f"  {key}: {existing.get(key)}" for key in differing)
    raise typer.BadParameter(
        f"The name {name!r} is already taken by a genome with different files:\n{held}\n"
        "Nothing was registered. Give this genome its own identity with "
        "--species, --assembly and --annotation-release so it is named for what "
        "it is; or, if it replaces the registered one, run "
        f"`paralome setup genome remove {name}` first."
    )


def _report_the_name_to_fix(folder: Path, target: Path, why: str) -> None:
    """Nothing registered: the folder must carry the genome's name."""
    typer.echo("")
    typer.echo(f"Cannot rename the folder: {why}.")
    typer.echo("Nothing was registered. A genome's folder must carry the genome's name,")
    typer.echo(f"{target.name}. Rename the folder to that, then register it:")
    typer.echo("")
    typer.echo(f"  mv {folder} {target}")
    typer.echo(f"  paralome setup genome add {target}")
    raise typer.Exit(code=1)


def _refuse_unless_the_folder_can_be_renamed(folder: Path, target: Path) -> None:
    """Say before anything is asked whether the rename is possible.

    Two things stop it: a folder of that name already there, and a parent this
    account may not rename entries in -- not writable, or sticky (the shared
    repository is 3775) with the folder owned by somebody else.
    """
    import stat

    if target.exists():
        _report_the_name_to_fix(folder, target, f"{target} already exists")
    parent = folder.parent
    if not os.access(parent, os.W_OK | os.X_OK):
        _report_the_name_to_fix(folder, target, f"this account cannot rename entries in {parent}")
    try:
        parent_stat, folder_stat = parent.stat(), folder.stat()
    except OSError as exc:
        _report_the_name_to_fix(folder, target, str(exc))
        return
    uid = os.geteuid()
    if (parent_stat.st_mode & stat.S_ISVTX and uid != 0
            and uid not in (folder_stat.st_uid, parent_stat.st_uid)):
        try:
            import pwd

            owner = pwd.getpwuid(folder_stat.st_uid).pw_name
        except KeyError:
            owner = str(folder_stat.st_uid)
        _report_the_name_to_fix(
            folder, target,
            f"{parent} lets only a folder's owner rename it, and this one belongs to {owner}",
        )


@genome_app.command("add")
def genome_add_command(
    folder: str = typer.Argument(
        ..., help="A genome folder, or a bare name under the recorded repository",
    ),
    analyses: Optional[str] = typer.Option(
        None, "--analyses", help=f"Override, from: {', '.join(ANALYSIS_ORDER)}",
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before registering"),
    species: Optional[str] = typer.Option(None, "--species", help="Override the scan"),
    assembly: Optional[str] = typer.Option(None, "--assembly", help="Override the scan"),
    annotation_source: Optional[str] = typer.Option(
        None, "--annotation-source", help="ensembl, gencode, refseq, custom or unknown",
    ),
    annotation_release: Optional[str] = typer.Option(
        None, "--annotation-release", help="e.g. 113, or v44 for GENCODE",
    ),
    modified: Optional[bool] = typer.Option(
        None, "--modified/--stock",
        help="Have you altered the assembly or the GTF? e.g. added GFP",
    ),
    modified_note: Optional[str] = typer.Option(
        None, "--modified-note", help="What you changed, e.g. 'GFP, mCherry added'",
    ),
    no_publish: bool = typer.Option(
        False, "--no-publish", help="Write only your own config",
    ),
) -> None:
    """Read a genome folder and register what is in it.

    The scan proposes the genome's identity (species, assembly, annotation)
    and you confirm it; the name is derived from that identity. Without a
    terminal, pass --yes.
    """
    # A subcommand rather than a positional on the group: a Click group cannot
    # have both a positional argument and subcommands without `list` and
    # `sync` being read as folder names.
    from paralome.setup.genome_scan import (
        analyses_for, annotation_identity, identity_from_fasta,
        scan_genome_folder,
    )
    from paralome.setup.genomes import resolve_genome_folder

    config = load_config(initialize=True)
    chosen_folder = resolve_genome_folder(folder, config)
    _require_the_shared_repository(chosen_folder)
    scan = scan_genome_folder(chosen_folder)
    _show_scan(scan)
    if not scan.findings:
        raise typer.BadParameter(f"Nothing to register in {scan.folder}")

    chosen = _split(analyses) if analyses else analyses_for(scan)
    entry: Dict[str, Any] = {"analyses": list(chosen)}

    # The scan PROPOSES the identity and the user confirms it. The
    # reference filename carries species and assembly, the annotation filename
    # carries the source and release -- but only for the tidy cases, and a
    # wrong or absent inference that nobody was shown is exactly how a genome
    # ends up quietly excluded from supersession.
    # REFUSE BEFORE ASKING, not after: the identity walkthrough is guarded by _require_terminal
    # too, so a batch job is refused with a hint rather than dying at a prompt.
    if not yes:
        _require_terminal(
            "Registering a genome",
            # --yes is the only way past this gate; the identity
            # flags override the scan but still need --yes without a terminal.
            hint="Pass --yes to register without being asked; add "
                 "--species/--assembly/--annotation-source/--annotation-release "
                 "with it to override what the scan found.",
        )

    scanned: Dict[str, Any] = dict(identity_from_fasta(scan.value("reference_fasta")))
    scanned.update(annotation_identity(scan.value("gtf")))
    entry.update(
        _ask_identity(
            supplied={
                "species": species,
                "assembly": assembly,
                "annotation_source": annotation_source,
                "annotation_release": annotation_release,
            },
            scanned=scanned,
            modified=modified,
            modified_note=modified_note,
            interactive=not yes,
        )
    )
    entry.update(scan.as_entry())
    # THE ANNOVAR BUILD FROM THE CONFIRMED ASSEMBLY, not from the FASTA's file name (which says
    # nothing for GENCODE, UCSC or RefSeq names).
    _annovar_for_the_confirmed_assembly(entry, Path(scan.folder))

    # THE NAME COMES FROM THE GENOME, not from whoever is typing.
    # It has to be derived here rather than earlier, because the identity it is
    # derived FROM is what _ask_identity just collected and confirmed.
    #
    # The fallback to the folder name is for a genome with no species or no
    # assembly recorded, which derive_genome_name refuses to name rather than
    # guess at. That is rare through this command -- _ask_identity asks for
    # both -- but it is reachable with --species '' and must not crash.
    derived = derive_genome_name(entry)
    if derived is None:
        typer.echo("")
        typer.echo(
            f"Cannot derive a name: species and assembly are both needed.\n"
            f"Falling back to the folder name, {scan.name!r}."
        )
    name = derived or scan.name
    entry["name"] = name
    name = _settle_a_rival_claim(name, entry, yes=yes)
    entry["name"] = name
    _refuse_to_replace_a_different_genome(name, entry, config)

    # THE FOLDER CARRIES THE GENOME'S NAME: a folder that differs from the name Paralome derives would
    # confuse people. Warn, rename, report; and when the rename cannot be done, register nothing and
    # say what to call it. Only a DERIVED name is imposed: the folder-name fallback above already
    # matches by definition.
    folder_path = Path(scan.folder)
    rename_to: Optional[Path] = None
    if derived is not None and folder_path.name != name:
        rename_to = folder_path.parent / name
        typer.echo("")
        typer.echo(f"The folder is named {folder_path.name}; Paralome names this genome {name}.")
        typer.echo("A genome's folder carries the genome's name, so the folder will be renamed:")
        typer.echo(f"  {folder_path}")
        typer.echo(f"  -> {rename_to}")
        _refuse_unless_the_folder_can_be_renamed(folder_path, rename_to)

    if not yes:
        _require_terminal(
            "Confirming a registration",
            hint="Pass --yes to register without being asked.",
        )
        question = (f"Rename the folder and register this as {name!r}?" if rename_to
                    else f"Register this as {name!r}?")
        if not typer.confirm(question, default=True):
            raise typer.Exit(code=1)

    if rename_to is not None:
        try:
            folder_path.rename(rename_to)
        except OSError as exc:
            _report_the_name_to_fix(folder_path, rename_to, f"the rename failed: {exc.strerror or exc}")
        # Every path the scan found was inside the old folder; read them again
        # from the new one. The FASTA and GTF are matched by file name (the
        # STAR index records basenames), so the move cannot lose them.
        moved = scan_genome_folder(rename_to)
        entry.update(moved.as_entry())
        _annovar_for_the_confirmed_assembly(entry, rename_to)

    outcome = register_genome(name, entry, config=config, publish=not no_publish)
    if rename_to is not None:
        typer.echo("")
        typer.echo("Renamed the folder to match the genome:")
        typer.echo(f"  {folder_path}")
        typer.echo(f"  -> {rename_to}")
    _report_registration(name, outcome, config)
    note = _burden_reference_note(name, entry)
    if note:
        typer.echo("")
        typer.echo(note)


@genome_app.command("remove")
def genome_remove_command(
    name: str = typer.Argument(..., help="The genome to unregister"),
    force: bool = typer.Option(False, "--force", help="Do not ask"),
) -> None:
    """Unregister a genome. The files on disk are never touched.

    Only the registry ENTRY is removed; nothing under the genome folder is
    deleted. A fatally withdrawn genome is refused without --force, because
    its entry is what warns everyone who adopted it.
    """
    # Adoption copies rather than references, so an adopter's entry carries no
    # withdrawal marker of its own; the reason is read from the owner's
    # registry file each time. Removing that entry would make a genome the
    # owner declared unusable valid again for every adopter. Superseded
    # genomes are not blocked: work against them is still correct.
    from paralome.setup.genomes import remove_genome

    from paralome.setup.genomes import my_registered_genomes

    config = load_config(initialize=True)
    entry = (config.get("genomes", {}) or {}).get(name)
    # The published copy (this account's drop-in) carries the
    # withdrawal adopters read, so it is checked too, and stands in for a
    # config entry that is gone: the guard and the question apply either way.
    unreadable = None
    try:
        published = my_registered_genomes().get(name)
    except (OSError, ValueError) as exc:
        published, unreadable = None, exc
    if entry is None:
        if unreadable is not None:
            raise typer.BadParameter(f"{name} is not in your config, and {unreadable}")
        if published is None:
            raise typer.BadParameter(
                f"{name} is not registered by this account. "
                f"`paralome setup genome list` shows what is."
            )
        entry = published if isinstance(published, dict) else {}
    if entry is not None:
        from paralome.setup.withdrawal import FATAL, describe, withdrawal_of

        retired = withdrawal_of(entry)
        if (retired is None or retired.get("severity") != FATAL) and isinstance(published, dict):
            retired = withdrawal_of(published) or retired
        if retired is not None and retired.get("severity") == FATAL and not force:
            typer.echo(describe(name, retired))
            typer.echo("")
            typer.echo(
                "REFUSING. This entry is what warns everyone who adopted this\n"
                "genome. Their copies carry no marker of their own -- they read\n"
                "your reason out of the shared registry each time they look.\n"
                "\n"
                "Removing it here takes it out of the registry too, and the\n"
                "genome silently becomes valid again for every one of them.\n"
                "\n"
                "Leave it. A withdrawn entry costs a line in a listing and is the\n"
                "only record of why the genome must not be used.\n"
                "\n"
                f"If you are certain -- nobody adopted it, or the files are long\n"
                f"gone -- `paralome setup genome remove {name} --force`."
            )
            raise typer.Exit(code=1)

    if entry is not None and not force:
        typer.echo(_describe(name, entry))
        typer.echo("")
        typer.echo("The folder and its files are NOT touched -- only this entry.")
        _require_terminal(
            "Confirming a removal",
            hint="Pass --force to remove without being asked.",
        )
        if not typer.confirm(f"Unregister {name}?", default=False):
            raise typer.Exit(code=1)

    outcome = remove_genome(name, config=config)
    if outcome["removed_from_config"]:
        typer.echo(f"Removed {name} from {config_summary()}")
    if outcome["removed_from_registry"]:
        typer.echo(f"Removed {name} from {site_registry_path()}")
    if outcome["error"]:
        typer.echo(outcome["error"])
        raise typer.Exit(code=1)


@genome_app.command("withdraw")
def genome_withdraw_command(
    name: str = typer.Argument(..., help="The genome to retire"),
    reason: str = typer.Option(..., "--reason", help="Why, in a sentence others will read"),
    fatal: bool = typer.Option(
        False,
        "--fatal/--superseded",
        help="Fatal refuses every future run; superseded warns and proceeds",
    ),
    yes: bool = typer.Option(
        False, "--yes",
        help="Skip the confirmation a fatal withdrawal asks for.",
    ),
) -> None:
    """Retire a genome, with a reason, without removing it.

    --fatal refuses every future run against the genome; --superseded (the
    default) warns and proceeds. The marker goes in your config and the
    machine registry, where adopters' runs read it.

    On a genome you adopted rather than registered, this removes your copy
    instead; only the owner can withdraw a genome for everyone.
    """
    # --superseded is the default because retiring a genome is nearly always
    # the hg19 case: nothing wrong with it, something newer exists. See
    # ``declined_genomes`` for why a non-owner's withdraw removes their copy.
    from paralome.setup.genomes import (
        _current_user, is_mine, registered_by, register_genome, remove_genome,
    )
    from paralome.setup.withdrawal import (
        FATAL, SUPERSEDED, describe, replacements_for, withdraw,
    )

    config = load_config(initialize=True)
    genomes = config.get("genomes", {}) or {}
    if name not in genomes:
        raise typer.BadParameter(
            f"{name} is not registered by this account, so there is nothing for "
            f"you to withdraw. `paralome setup genome list` shows what is.\n"
            f"A genome another account owns can only be withdrawn by that account."
        )

    if not is_mine(genomes[name]):
        owner = registered_by(genomes[name]) or "another account"
        typer.echo(f"{name} was registered by {owner}, not by you.")
        typer.echo("")
        typer.echo(
            "Withdrawing is a judgement ABOUT the genome, and it reaches everyone\n"
            f"who has adopted it -- so it stays {owner}'s call. What you can do is\n"
            "drop your own copy, which is what this has done.\n"
            "\n"
            "Nothing was published and nothing was flagged: another account cannot\n"
            "tell you removed it, because your not needing a genome is not evidence\n"
            "about the genome.\n"
            "\n"
            f"If you believe it is genuinely broken, tell {owner} so they can\n"
            "withdraw it for everybody."
        )
        outcome = remove_genome(name, config=config)
        if outcome["removed_from_config"]:
            typer.echo("")
            typer.echo(f"Removed {name} from {config_summary()}")
        if outcome["removed_from_registry"]:
            typer.echo(f"Removed your published copy from {site_registry_path()}")
        if outcome["declined"]:
            typer.echo("`sync` will not offer it to you again.")
        if outcome["error"]:
            typer.echo(outcome["error"])
            raise typer.Exit(code=1)
        return

    severity = FATAL if fatal else SUPERSEDED

    # --fatal SAYS WHAT IT WILL DO, AND ASKS.
    #
    # Typing --fatal is deliberate, so this is not a second permission gate -- it is the one place in Paralome where a command
    # reaches other people's work, and the person typing it deserves to see
    # that spelled out before it happens rather than in the line printed
    # afterwards. --superseded is untouched: it warns and proceeds, which harms
    # nobody, and adding friction there would train people to hurry past it.
    #
    # Default NO. Somebody who hits return without reading has changed nothing.
    if severity == FATAL and not yes:
        typer.echo("")
        typer.echo(f"FATAL withdrawal of {name}")
        typer.echo("")
        typer.echo("  Every future run against this genome will stop with PLM-0407.")
        typer.echo("  That applies to every account on this machine that has")
        typer.echo("  adopted it, and to you. Nobody will be able to use it until")
        typer.echo("  you restore it with `paralome setup genome restore`.")
        typer.echo("")
        typer.echo(f"  Reason others will read:  {reason}")
        typer.echo("")
        typer.echo("Reserve this for a genome that produces WRONG results. For one")
        typer.echo("that is merely older, --superseded warns and lets work continue.")
        typer.echo("")
        _require_terminal(
            "A fatal withdrawal",
            hint="Pass --yes to withdraw without being asked.",
        )
        if not typer.confirm("Withdraw it for everyone?", default=False):
            typer.echo("Nothing withdrawn.")
            raise typer.Exit(code=1)

    record = withdraw(
        name, config, severity=severity, reason=reason, by=_current_user()
    )
    # Written back through register_genome so the shared registry sees it too:
    # an entry published to the site copy and then withdrawn only in the user's
    # config would keep offering itself to everyone who syncs.
    register_genome(name, genomes[name], config=config)

    typer.echo(describe(name, record))
    if severity == FATAL:
        typer.echo("Future runs against it will stop with PLM-0407, yours included.")
    else:
        typer.echo("Runs against it will warn and proceed.")
    for item in replacements_for(name, genomes[name], genomes):
        typer.echo(f"  {item.headline}: {item.name} ({item.detail})")
        if item.caution:
            typer.echo(f"      {item.caution}")


@genome_app.command("restore")
def genome_restore_command(
    name: str = typer.Argument(..., help="The genome to un-retire"),
) -> None:
    """Undo a withdrawal on your own entry.

    Withdrawals are judgements, and judgements are sometimes wrong -- a genome
    condemned for an index built against the wrong FASTA turns out to have been
    the right FASTA and a stale STAR version. Nothing about the flag should be
    harder to undo than it was to apply.
    """
    from paralome.setup.genomes import is_mine, register_genome, registered_by
    from paralome.setup.withdrawal import restore

    config = load_config(initialize=True)
    genomes = config.get("genomes", {}) or {}
    if name not in genomes:
        raise typer.BadParameter(f"{name} is not registered by this account.")
    mine = is_mine(genomes[name])
    removed = restore(name, config)
    if removed is None:
        typer.echo(f"{name} was not withdrawn. Nothing to do.")
        return
    # An adopted copy is never published (a second entry for
    # somebody else's genome), and the owner's withdrawal still wins, so only
    # say "back in service" when nothing else withdraws it.
    register_genome(name, genomes[name], config=config, publish=mine)
    if not mine:
        from paralome.setup.withdrawal import describe, effective_withdrawal

        owner = registered_by(genomes[name]) or "another account"
        still = effective_withdrawal(name, genomes[name])
        if still is not None:
            typer.echo(f"Cleared the withdrawal on your copy of {name}, but {owner} registered it "
                       f"and their withdrawal still applies:\n  {describe(name, still)}\n"
                       f"Only {owner} can restore it for everyone.")
            return
    typer.echo(f"{name} is back in service (was: {removed.get('reason')})")


@genome_app.command("rename")
def genome_rename_command(
    name: Optional[str] = typer.Argument(
        None, help="One genome to rename; omit for every genome of yours",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report and stop"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before renaming"),
) -> None:
    """Move your genomes onto names derived from their identity.

    Genomes registered with hand-typed names (e.g. `Human_GRCh38.e81`) are
    renamed to the name derived from species, assembly and annotation. Only
    genomes you registered are renamed; adopted ones are reported and skipped.

    The old name is kept in ``former_names`` and still resolves, so recorded
    runs and `--genome <old name>` keep working.
    """
    from paralome.config.user import save_config
    from paralome.setup.genome_names import remember_former_name
    from paralome.setup.genomes import is_mine

    config = load_config(initialize=True)
    genomes = config.get("genomes", {}) or {}
    if not genomes:
        # _nothing_registered_yet prints for itself and talks about
        # a repository line rename never printed.
        typer.echo("You have no genomes to rename. `paralome setup genome list` "
                   "shows what is registered.")
        return

    wanted = [name] if name else sorted(genomes)
    planned: List[tuple] = []
    skipped: List[str] = []
    for current in wanted:
        entry = genomes.get(current)
        if not isinstance(entry, dict):
            raise typer.BadParameter(f"{current} is not one of your genomes.")
        if not is_mine(entry):
            skipped.append(current)
            continue
        derived = derive_genome_name(entry)
        if derived is None or derived == current:
            continue
        if derived in genomes:
            skipped.append(f"{current} -> {derived} (that name is already yours)")
            continue
        planned.append((current, derived, entry))
    # Two of your genomes deriving one name would overwrite each
    # other (one entry lost, its old name no longer resolving); rename neither.
    claims: Dict[str, List[str]] = {}
    for current, derived, _ in planned:
        claims.setdefault(derived, []).append(current)
    for current, derived, _ in planned:
        if len(claims[derived]) > 1:
            others = ", ".join(c for c in claims[derived] if c != current)
            skipped.append(f"{current} -> {derived} ({others} would also be renamed to {derived}; "
                           "remove or rename the duplicate first)")
    planned = [p for p in planned if len(claims[p[1]]) == 1]

    for item in skipped:
        typer.echo(f"Skipped: {item}")
    if not planned:
        typer.echo("Nothing to rename.")
        return

    typer.echo("")
    for current, derived, _ in planned:
        typer.echo(f"  {current}  ->  {derived}")
    typer.echo("")
    if dry_run:
        typer.echo("--dry-run: nothing written.")
        return
    if not yes:
        _require_terminal(
            "Renaming genomes",
            hint="`paralome setup genome rename --dry-run` reports without writing; "
                 "pass --yes to rename without being asked.",
        )
        if not typer.confirm(f"Rename {len(planned)} genome(s)?", default=True):
            raise typer.Exit(code=1)

    for current, derived, entry in planned:
        moved = dict(entry)
        moved["name"] = derived
        remember_former_name(moved, current)
        genomes.pop(current, None)
        genomes[derived] = moved
    config["genomes"] = genomes
    save_config(config)
    typer.echo(f"Renamed {len(planned)} genome(s) in {config_summary()}.")

    published = {
        derived: genomes[derived] for _, derived, _ in planned if derived in genomes
    }
    if published:
        try:
            from paralome.setup.genomes import my_registered_genomes

            # Rewrite only this account's own drop-in file. The
            # merged view would copy every other account's genomes into it.
            site = my_registered_genomes()
            site.update(published)
            for current, _, _ in planned:
                site.pop(current, None)
            save_site_registry(site)
            typer.echo("This machine's registry updated.")
        except OSError as exc:
            typer.echo(f"This machine's registry NOT updated -- {exc}")
    typer.echo("")
    typer.echo("Old names still resolve; they are recorded in former_names.")


@genome_app.command("adopt")
def genome_adopt_command(
    name: str = typer.Argument(..., help="A genome on this machine to take"),
) -> None:
    """Take one genome from this machine's registry, by name.

    Takes a single named genome without the questions `sync` asks, and clears
    an earlier decline of it. Your copy is not republished: adoption copies
    the entry into your config only.
    """
    from paralome.config.user import save_config
    from paralome.setup.genomes import (
        discovered_genomes, undecline,
    )
    from paralome.setup.withdrawal import FATAL, describe, effective_withdrawal

    config = load_config(initialize=True)
    genomes = config.get("genomes", {}) or {}

    available = dict(discovered_genomes())
    available.update(load_site_registry())
    if name not in available:
        from paralome.setup.genomes import registry_permission_text

        denied = registry_permission_text()
        if denied:
            raise typer.BadParameter(f"{name} was not found. {denied}")
        raise typer.BadParameter(
            f"{name} is not on this machine. "
            f"`paralome setup genome list` shows what is."
        )
    if name in genomes:
        typer.echo(f"You already have {name}. Nothing to do.")
        if undecline(name, config) is not None:
            save_config(config)
        return

    entry = dict(available[name])
    retired = effective_withdrawal(name, entry)
    if retired and retired.get("severity") == FATAL:
        typer.echo(describe(name, retired))
        typer.echo("")
        typer.echo(
            "REFUSING. Its owner has withdrawn it as unusable, and adoption is\n"
            "the cheapest moment to act on that -- before it is in your config\n"
            "and before a run has been queued against it."
        )
        raise typer.Exit(code=1)

    genomes[name] = entry
    config["genomes"] = genomes
    was_declined = undecline(name, config) is not None
    save_config(config)

    typer.echo(_describe(name, entry))
    typer.echo(f"Added {name} to {config_summary()}")
    if was_declined:
        typer.echo("`sync` will offer it again from now on.")
    if retired:
        typer.echo("")
        typer.echo(describe(name, retired))


def _show_repository(config: Mapping[str, Any]) -> None:
    """Where genomes go, and whether this account can put one there.

    Printed on EVERY listing rather than only the empty one: somebody with
    four genomes still needs to know where the fifth goes.

    The writability check is the part that earns its place. It is the
    difference between learning you are not in the group now and learning it
    after spending two hours building an index you cannot move.
    """
    from paralome.setup.genomes import genome_repository

    repository = genome_repository(config)
    if repository is None:
        return
    typer.echo(f"Genome repository : {repository}")
    if not repository.is_dir():
        typer.echo("                    (recorded, but not present on this machine)")
        return
    if os.access(repository, os.W_OK | os.X_OK):
        typer.echo("                    you can add genomes here")
    else:
        typer.echo("                    read-only for you -- ask an administrator")
        typer.echo("                    to add you to its group, then log out and in")


def _nothing_registered_yet(config: Mapping[str, Any]) -> None:
    """The first user on a fresh machine, who currently gets a blank stare.

    An empty listing is not an error and never was, but it answered nothing:
    no genomes, no indication of where one would go, no next step. The first
    user to set up a genome gets an empty list with a statement of where the
    folder is.
    """
    from paralome.setup.genomes import genome_repository

    typer.echo("No genomes are registered on this machine yet -- you would be "
               "the first.")
    typer.echo("")
    repository = genome_repository(config)
    if repository is None:
        typer.echo("Register a genome folder wherever it lives:")
        typer.echo("")
        typer.echo("    paralome setup genome add /path/to/the/folder")
        return
    typer.echo("Put a genome folder in the repository above, then register it:")
    typer.echo("")
    typer.echo(f"    paralome setup genome add {repository}/<folder>")
    typer.echo("")
    typer.echo("A bare name works too, once the folder is there.")


@genome_app.command("list")
def genome_list_command(
) -> None:
    """Every genome available here, and whether its files are on this machine.

    Read-only: it never creates a configuration. An account with no
    configuration has no genomes of its own, and still sees what other
    accounts have registered.
    """
    print_genome_listing()


def print_genome_listing() -> None:
    """The listing itself, so the three commands that show it show one thing.

    Three places want this: `genome list`, the bare group, and
    `setup doctor <a name nobody registered>` -- where nothing is validated,
    because the genome does not exist, and what somebody needs is the list of
    names that DO exist.
    """
    from paralome.errors import ParalomeError

    try:
        config = load_config(initialize=False)
    except ParalomeError:
        # A key nothing reads is refused by name, here as everywhere.
        raise
    except (FileNotFoundError, ValueError):
        config = {}
    user_genomes = config.get("genomes", {}) or {}
    site_genomes = load_site_registry()
    typer.echo(f"Your config       : {config_summary()}")
    typer.echo(f"Shared registry   : {site_registry_path()}")
    _show_repository(config)
    typer.echo("")
    from paralome.setup.genomes import registry_permission_text

    denied = registry_permission_text()
    if denied:
        typer.echo(denied)
        typer.echo("")
    if not user_genomes and not site_genomes:
        _nothing_registered_yet(config)
        return
    _show_existing(user_genomes, site_genomes)
    _show_discovered(user_genomes)


def config_summary() -> str:
    from paralome.config.user import config_path

    return str(config_path())


@genome_app.command("sync")
def genome_sync_command(
    adopt: bool = typer.Option(True, "--adopt/--no-adopt", help="Take machine genomes you lack"),
    publish: bool = typer.Option(True, "--publish/--no-publish", help="Give the machine genomes it lacks"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would change and stop"),
) -> None:
    """Reconcile your genomes with this machine's registry, both directions.

    Offers to publish genomes you have that the machine lacks, and to adopt
    machine genomes you lack. This is how genomes that predate the registry
    get into it, and how a new account picks up everything already built
    without typing a path.
    """
    from paralome.setup.genomes import declined_genomes, discovered_genomes

    config = load_config(initialize=True)
    user_genomes = config.get("genomes", {}) or {}
    # Genomes this account has already said no to. Without this,
    # removing one you do not need means being asked about it at every sync
    # forever, and the only way to stop is to keep it.
    said_no = declined_genomes(config)
    # The shared registry if the site has one, plus everything other accounts
    # have registered in their own configs. The second source needs no shared
    # writable location and no administrator, which is why it exists -- see
    # discovered_genomes. A shared-registry entry wins on a name held by both,
    # since somebody put it there on purpose.
    site_genomes = dict(discovered_genomes())
    site_genomes.update(load_site_registry())
    report = reconcile(user_genomes, site_genomes, said_no)
    # An adopted entry whose name the owner has since renamed is
    # not "yours only". Publishing it would put a second, stale entry for the
    # owner's genome into the registry, so report it and do not offer it.
    renamed = _renamed_by_their_owner(report.only_user, user_genomes, site_genomes)
    report.only_user = [n for n in report.only_user if n not in renamed]

    typer.echo(f"Machine registry: {site_registry_path()}")
    if not site_registry_is_writable():
        typer.echo("  (not writable by you -- nothing can be published from this account)")
    typer.echo("")
    _print_reconciliation(report)
    for old, new in sorted(renamed.items()):
        typer.echo(f"{old} was renamed by its owner to {new}. Take the new name with")
        typer.echo(f"  paralome setup genome adopt {new}")
        typer.echo(f"and then drop the old copy with `paralome setup genome remove {old}`.")

    if report.nothing_to_do:
        typer.echo("Nothing to reconcile.")
        return
    if dry_run:
        typer.echo("--dry-run: nothing written.")
        return

    _require_terminal(
        "Syncing",
        hint="`paralome setup genome sync --dry-run` reports what would change "
             "without asking anything; run the real sync from a login shell.",
    )
    changed_user, changed_site = False, False
    # What this account publishes. Only these go into its own
    # drop-in file; discovered and other accounts' entries are never written.
    to_publish: Dict[str, Any] = {}

    if publish and report.only_user:
        typer.echo("")
        for name in report.only_user:
            typer.echo(_describe(name, user_genomes[name]))
            if typer.confirm(f"  Add {name} to this machine's registry?", default=True):
                site_genomes[name] = dict(user_genomes[name])
                to_publish[name] = site_genomes[name]
                changed_site = True

    if adopt and report.only_site:
        from paralome.setup.withdrawal import FATAL, effective_withdrawal

        typer.echo("")
        for name in report.only_site:
            typer.echo(_describe(name, site_genomes[name]))
            # Adoption is the moment a withdrawal is cheapest to
            # act on -- before the entry is in anyone's config and before a
            # run has been queued against it. A fatally withdrawn genome is
            # not offered at all, because there is no answer to "would you
            # like a genome that produces wrong results?" worth collecting.
            # A superseded one is still offered, with the default flipped.
            retired = effective_withdrawal(name, site_genomes[name])
            if retired and retired.get("severity") == FATAL:
                typer.echo("  Not offered: withdrawn as unusable by its owner.")
                continue
            if typer.confirm(
                f"  Add {name} to your configuration?", default=not retired
            ):
                user_genomes[name] = dict(site_genomes[name])
                changed_user = True

    for name, keys in report.conflicting.items():
        typer.echo("")
        typer.echo(f"{name} exists in both, with different files.")
        typer.echo(f"  differing: {', '.join(keys)}")
        for key in keys:
            typer.echo(f"    yours  {key}: {_with_existence(user_genomes[name].get(key))}")
            typer.echo(f"    theirs {key}: {_with_existence(site_genomes[name].get(key))}")
        typer.echo("  [m] use this machine's version   [r] keep yours, under a new name   [s] skip")
        choice = typer.prompt("  Choice", default="s").strip().lower()
        if choice.startswith("m"):
            user_genomes[name] = dict(site_genomes[name])
            changed_user = True
        elif choice.startswith("r"):
            new_name = typer.prompt("  New name for your version").strip()
            if not new_name or new_name in user_genomes or new_name in site_genomes:
                typer.echo("    Not a free name; skipped.")
                continue
            user_genomes[new_name] = dict(user_genomes.pop(name))
            user_genomes[name] = dict(site_genomes[name])
            site_genomes[new_name] = dict(user_genomes[new_name])
            to_publish[new_name] = site_genomes[new_name]
            changed_user = changed_site = True

    if changed_user:
        from paralome.config.user import save_config

        config["genomes"] = user_genomes
        save_config(config)
        typer.echo("")
        typer.echo("Your configuration updated.")
    if changed_site:
        try:
            from paralome.setup.genomes import my_registered_genomes

            mine = my_registered_genomes()
            mine.update(to_publish)
            save_site_registry(mine)
            typer.echo("This machine's registry updated.")
        except OSError as exc:
            typer.echo(f"This machine's registry NOT updated -- {exc}")
    if not (changed_user or changed_site):
        typer.echo("")
        typer.echo("Nothing changed.")


def _renamed_by_their_owner(
    names: List[str], user_genomes: Mapping[str, Any], site_genomes: Mapping[str, Any],
) -> Dict[str, str]:
    """Adopted names that a registry entry now lists as a former name."""
    from paralome.setup.genome_names import former_names
    from paralome.setup.genomes import is_mine, registered_by

    found: Dict[str, str] = {}
    for old in names:
        entry = user_genomes.get(old)
        if not isinstance(entry, Mapping) or is_mine(entry):
            continue
        for new, site_entry in site_genomes.items():
            if (isinstance(site_entry, Mapping) and old in former_names(site_entry)
                    and registered_by(site_entry) == registered_by(entry)):
                found[old] = new
                break
    return found


def _print_reconciliation(report: Reconciliation) -> None:
    if report.identical:
        typer.echo(f"Already in step ({len(report.identical)}): {', '.join(report.identical)}")
    if report.only_user:
        typer.echo(f"Yours only, this machine lacks them: {', '.join(report.only_user)}")
    if report.only_site:
        typer.echo(f"On this machine, not yours: {', '.join(report.only_site)}")
    if report.declined:
        # Shown, not offered. Silence would leave someone wondering where a
        # genome went; a prompt would be the thing the decline exists to stop.
        typer.echo(f"Not offered, you removed these: {', '.join(report.declined)}")
        typer.echo("  `paralome setup genome adopt <name>` if you want one back.")
    if report.conflicting:
        typer.echo("Same name, different files:")
        for name, keys in report.conflicting.items():
            typer.echo(f"  {name}: {', '.join(keys)}")


# ---------------------------------------------------------------------------
# The gnomAD reference burden table
# ---------------------------------------------------------------------------


def _numbered(exc) -> "Exception":
    """A planning refusal from burden/reference.py, as a numbered error."""
    from paralome.errors import ParalomeError

    return ParalomeError(exc.code, exc.detail, fix=exc.fix)


def _burden_reference_note(name: str, entry: Mapping[str, Any]) -> Optional[str]:
    """What `genome add` says about the burden reference, or None.

    Said, not started. The build annotates every rare gnomAD exome
    allele with ensGene and dbNSFP -- hours on a cluster -- and an interactive
    registration is not the place to begin that unasked.
    """
    from paralome.burden.reference import build_command, gnomad_tables_present

    if entry.get("burden_reference"):
        return None
    tables = gnomad_tables_present(entry.get("annovar_db"), entry.get("annovar_buildver"))
    if "exome" not in tables:
        return None
    return (
        "The DNA gene_burden stage needs this genome's gnomAD burden reference,\n"
        f"built once from {tables['exome']}. Build it with:\n"
        f"  {build_command(name)}\n"
        f"  (or `{build_command(name, 'local')} --threads N` on this machine)\n"
        "Until then a DNA run skips gene_burden and says why."
    )


def record_burden_reference(name: str, path: Path) -> Optional[Dict[str, Any]]:
    """Record ``burden_reference`` in the genome's existing entry, or None."""
    from paralome.setup.genomes import record_derived_key

    return record_derived_key(name, "burden_reference", str(path))


def _genome_entry_for(name: str):
    """(registered name, entry) for a genome this account has, or PLM-0401/0407."""
    from paralome.errors import ParalomeError
    from paralome.pipeline.resource_catalog import ResourceResolutionError, select_genome_config
    from paralome.setup.withdrawal import enforce

    config = load_config(initialize=True)
    try:
        genome, entry = select_genome_config(name, config=config)
    except ResourceResolutionError as exc:
        raise ParalomeError(401, str(exc))
    enforce(str(genome), entry)
    return str(genome), dict(entry)


@genome_app.command("burden-reference")
def genome_burden_reference_command(
    name: str = typer.Argument(..., help="A registered genome with ANNOVAR and gnomAD"),
    backend: str = typer.Option("local", "--backend", help="local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit the SLURM jobs (with --backend slurm)"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    threads: Optional[int] = typer.Option(None, "--threads", help="Chunks at once on this machine (--backend local; default 1)"),
    max_parallel: Optional[int] = typer.Option(
        None, "--max-parallel",
        help="At most N SLURM array tasks at once (--backend slurm; default 48, 0 for no limit)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Build again over an existing reference"),
    mem: Optional[str] = typer.Option(None, "--mem", help="Memory per SLURM chunk task (default 6G)"),
) -> None:
    """Build the per-gene gnomAD expected table the DNA gene_burden stage reads.

    Once per genome. Written to <genome folder>/burden/ and recorded
    in the genome entry as burden_reference when it is complete. Run again
    after a failure, it keeps the chunks already finished (same gnomAD table,
    ANNOVAR inputs, chunk size and thresholds) and computes only the rest.
    """
    from paralome.burden import reference as R
    from paralome.errors import ParalomeError

    if backend not in ("local", "slurm"):
        raise ParalomeError(108, f"--backend {backend!r}: expected local or slurm")
    if submit and backend != "slurm":
        raise ParalomeError(103, "--submit submits SLURM jobs; it needs --backend slurm")
    if partition and backend != "slurm":
        raise ParalomeError(103, "--partition is a SLURM partition; it needs --backend slurm")
    # --threads is this machine's. The array's concurrency limit is its own option, off unless given.
    if threads is not None and backend != "local":
        raise ParalomeError(103, "--threads is for --backend local; limit the SLURM array with --max-parallel N")
    if max_parallel is not None and backend != "slurm":
        raise ParalomeError(103, "--max-parallel limits the SLURM array; it needs --backend slurm")
    if threads is not None and threads < 1:
        raise ParalomeError(112, f"--threads {threads}: at least 1")
    if max_parallel is not None and max_parallel < 0:
        raise ParalomeError(112, f"--max-parallel {max_parallel}: 0 (no limit) or more")
    # The defaults are the module's, and the array is limited, so a build does not fill the nodes a
    # project's runs need.
    mem = mem or R.DEFAULT_MEM
    if backend == "slurm" and max_parallel is None:
        max_parallel = R.DEFAULT_MAX_PARALLEL

    genome, entry = _genome_entry_for(name)
    try:
        plan = R.plan_reference(genome=genome, entry=entry, rebuild=rebuild)
    except R.BurdenReferenceError as exc:
        raise _numbered(exc)
    plan_path = R.write_plan(plan)
    sources = ", ".join(f"{s} ({Path(i['table']).name})" for s, i in plan["sources"].items())
    todo = R.missing_chunks(plan)
    typer.echo(f"Burden reference for {genome}: {len(plan['chunks'])} chunk(s) from {sources}")
    if plan.get("reused_plan"):
        typer.echo(f"  same inputs as the earlier plan: {len(plan['chunks']) - len(todo)} chunk(s) finished "
                   f"earlier are kept, {len(todo)} to compute")
    if plan.get("regulatory_absent_reason"):
        typer.echo(f"  {plan['regulatory_absent_reason']}")
    typer.echo(f"  plan     : {plan_path}")
    typer.echo(f"  output   : {plan['sources']['exome']['output']}")

    if backend == "local":
        try:
            provenance = R.run_local(plan, threads=threads or 1, echo=typer.echo)
        except R.BurdenReferenceError as exc:
            raise _numbered(exc)
        _report_burden_reference(genome, provenance)
        return

    scripts = R.write_slurm_scripts(plan, plan_path, partition=partition, max_parallel=max_parallel, mem=mem)
    if scripts["array"] is not None:
        typer.echo(f"  scripts  : {scripts['array']}")
        typer.echo(f"             {scripts['finalize']}")
    else:
        typer.echo(f"  scripts  : {scripts['finalize']} (every chunk is finished; only the sum is left)")
    if not submit:
        import shlex

        again = R.build_command(genome)
        if partition:
            again += f" --partition {shlex.quote(partition)}"
        if max_parallel != R.DEFAULT_MAX_PARALLEL:
            again += f" --max-parallel {max_parallel}"
        if mem != R.DEFAULT_MEM:
            again += f" --mem {shlex.quote(mem)}"
        if rebuild:
            again += " --rebuild"
        typer.echo("")
        typer.echo("Not submitted. Submit with:")
        if scripts["array"] is not None:
            typer.echo(f"  jid=$(sbatch --parsable {scripts['array']})")
            typer.echo(f"  sbatch --dependency=afterok:$jid {scripts['finalize']}")
        else:
            typer.echo(f"  sbatch {scripts['finalize']}")
        typer.echo(f"or run: {again}")
        return
    R.submit_slurm(scripts, echo=typer.echo)
    typer.echo("The finalize job records burden_reference in the genome entry when it completes.")


@genome_app.command("annovar-table")
def genome_annovar_table_command(
    name: str = typer.Argument(..., help="A registered genome with ANNOVAR"),
    table: str = typer.Argument(..., help="The ANNOVAR table, e.g. gnomad41_genome"),
    operation: str = typer.Option("f", "--operation", help="Its ANNOVAR operation: f filter (default), g gene, r region"),
    backend: str = typer.Option("local", "--backend", help="local (this machine, now) or slurm (one job)"),
    submit: bool = typer.Option(False, "--submit", help="Submit the SLURM job (with --backend slurm)"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    time: Optional[str] = typer.Option(None, "--time", help="SLURM time limit, e.g. 2-00:00:00"),
) -> None:
    """Download an ANNOVAR table into a genome's database and add it to the genome's protocol.

    \b
    For a whole-genome project:
      paralome setup genome annovar-table <genome> gnomad41_genome
    gives every variant a gnomAD GENOME frequency (intronic, intergenic and
    regulatory ones included) from the next annotation on, and lets the
    gene-burden regulatory tier run. Then build (or --rebuild) the burden
    reference, so it has the regulatory table too. A table already there is
    not downloaded again; it is only added to the protocol.
    """
    from paralome.errors import ParalomeError
    from paralome.setup import annovar_tables as A

    if backend not in ("local", "slurm"):
        raise ParalomeError(108, f"--backend {backend!r}: expected local or slurm")
    if submit and backend != "slurm":
        raise ParalomeError(103, "--submit submits a SLURM job; it needs --backend slurm")
    if (partition or time) and backend != "slurm":
        raise ParalomeError(103, "--partition and --time are SLURM options; they need --backend slurm")
    genome, entry = _genome_entry_for(name)
    try:
        table = A.check_table_name(table)
        A.with_protocol(entry, table, operation)          # refuses misaligned lists before any download
        _annovar_dir, annovar_db, buildver = A.genome_annovar(entry)
    except A.AnnovarTableError as exc:
        raise ParalomeError(exc.code, exc.detail, fix=exc.fix)

    if backend == "slurm":
        script = A.slurm_script(genome, table, operation, annovar_db, partition=partition, time=time)
        typer.echo(f"ANNOVAR table {table} ({buildver}) for {genome}: {script}")
        if not submit:
            typer.echo("Not submitted. Submit with:")
            typer.echo(f"  sbatch {script}")
            typer.echo("A compute node needs to reach www.openbioinformatics.org; if yours cannot, run this "
                       "with --backend local on the login node instead (under nohup or screen: it is a "
                       "long download).")
            return
        from paralome.pipeline import slurm

        job = slurm.sbatch(script)
        typer.echo(f"Submitted: job {job}. When it completes, {table} is in the protocol of {genome}.")
        return
    _fetch_and_record(genome, entry, table, operation)


def _fetch_and_record(genome: str, entry: Mapping[str, Any], table: str, operation: str) -> None:
    from paralome.errors import ParalomeError
    from paralome.setup import annovar_tables as A
    from paralome.setup.genomes import record_annovar_table

    _annovar_dir, annovar_db, _buildver = A.genome_annovar(entry)
    try:
        A.fetch_table(entry, table, echo=typer.echo,
                      log=annovar_db / "paralome_jobs" / "logs" / f"annovar_table_{table}.download.log")
        outcome = record_annovar_table(genome, table, operation)
    except A.AnnovarTableError as exc:
        raise ParalomeError(exc.code, exc.detail, fix=exc.fix)
    if outcome is None:
        typer.echo(f"{genome} is not in your own configuration; {table} was not added to its protocol.")
        return
    protocols = A.split_list(A.with_protocol(entry, table, operation).get("annovar_protocol"))
    typer.echo(f"{table} is in the ANNOVAR protocol of {genome} ({len(protocols)} tables).")
    if table == "gnomad41_genome":
        from paralome.burden.reference import build_command

        typer.echo("Next: build the burden reference (with --rebuild if it was built before) so it has the "
                   f"regulatory table:\n  {build_command(genome)}")
    typer.echo("Projects annotated before this keep their annotation: remove a project's annotation folder, "
               "and the stages after it, to annotate it again with this table.")


def _report_burden_reference(genome: str, provenance: Mapping[str, Any]) -> None:
    path = Path(str(provenance["burden_reference"]))
    outcome = record_burden_reference(genome, path)
    typer.echo("")
    typer.echo(f"Built {path}")
    for source, info in (provenance.get("sources") or {}).items():
        typer.echo(f"  {source}: {info['rows']} rows, {info['genes']} genes, "
                   f"{info['alleles_annotated']} alleles annotated")
    regulatory = provenance.get("regulatory") or {}
    if regulatory.get("status") != "built":
        typer.echo(f"  {regulatory.get('reason')}")
    if outcome is None:
        typer.echo(f"  {genome} is not in your own configuration; burden_reference was not recorded.")
    else:
        typer.echo(f"  recorded as burden_reference of {genome}")


#: The two array-task internals of `burden-reference`. Machine
#: surface, for paralome-worker only (see worker.py on why the split).
genome_worker_app = typer.Typer(help="Genome-setup jobs run by generated scripts.")


@genome_worker_app.command("burden-chunk")
def genome_burden_chunk_command(
    plan: str = typer.Option(..., "--plan", help="work/plan.json of the build"),
    chunk: int = typer.Option(..., "--chunk", help="1-based chunk index"),
) -> None:
    from paralome.burden import reference as R

    try:
        loaded = R.read_plan(Path(plan))
        path = R.run_chunk(loaded, chunk)
    except R.BurdenReferenceError as exc:
        raise _numbered(exc)
    typer.echo(f"chunk {chunk} done: {path}")


@genome_worker_app.command("annovar-table")
def genome_annovar_table_worker(
    genome: str = typer.Option(..., "--genome"),
    table: str = typer.Option(..., "--table"),
    operation: str = typer.Option("f", "--operation"),
) -> None:
    """The SLURM job of `setup genome annovar-table`: download, check, record."""
    from paralome.errors import ParalomeError
    from paralome.setup import annovar_tables as A

    name, entry = _genome_entry_for(genome)
    try:
        A.with_protocol(entry, A.check_table_name(table), operation)
    except A.AnnovarTableError as exc:
        raise ParalomeError(exc.code, exc.detail, fix=exc.fix)
    _fetch_and_record(name, entry, table, operation)


@genome_worker_app.command("burden-finalize")
def genome_burden_finalize_command(
    plan: str = typer.Option(..., "--plan", help="work/plan.json of the build"),
) -> None:
    from paralome.burden import reference as R

    try:
        loaded = R.read_plan(Path(plan))
        provenance = R.finalize(loaded)
    except R.BurdenReferenceError as exc:
        raise _numbered(exc)
    _report_burden_reference(str(loaded["genome"]), provenance)
