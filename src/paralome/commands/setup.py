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

import json
from pathlib import Path
from typing import Optional

import typer

from paralome.setup.doctor import doctor_report, initialize_setup, print_doctor_report, print_setup_result
from paralome.config.user import DEFAULT_CONFIG_DIRNAME, config_path, load_config

#: The default group allowed to contribute genomes.
#:
#: Derived rather than written as a quoted literal, as default_site_registry_path
#: derives its directory name: the bare quoted name of the human console script is
#: kept out of modules whose job is not naming it, because a stale one at a machine
#: call site would invoke the human CLI from a generated sbatch body.
#:
#: DEFAULT_CONFIG_DIRNAME is ".paralome" and is deliberately NOT renamed in the
#: development build -- the two share one registry, and one group -- so this is
#: the same string in both.
DEFAULT_GENOME_GROUP = DEFAULT_CONFIG_DIRNAME.lstrip(".")

setup_app = typer.Typer(help="Paralome setup commands")

# Registering a genome. Under `setup` rather than `config`
# because it is part of making a machine usable, which is what a person
# looking for it would reach for -- and `config` is for inspecting what is
# already there, not for building it.
from paralome.commands.genome import genome_app  # noqa: E402

setup_app.add_typer(genome_app, name="genome")

# `config` lives here: inspecting the configuration is part of setting a machine up, not a separate
# activity. It is a COMMAND rather than a group. See commands/config.py.
from paralome.commands.config import config_command  # noqa: E402

setup_app.command("config")(config_command)


@setup_app.callback(invoke_without_command=True)
def setup_command(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-apply shipped default VALUES over ones you have changed. "
             "Never touches genomes or profiles.",
    ),
):
    """Initialize Paralome user config and discover local executables."""
    from paralome.cli import READ_ONLY_SETUP, _refuse_administration_inside_a_job

    # Here, not in the top-level callback, which cannot see which
    # setup command it is: `setup where` only answers (Evidome asks it from
    # inside the functional_evidence job); every other setup command is
    # refused inside a SLURM job as before.
    if ctx.invoked_subcommand not in READ_ONLY_SETUP:
        _refuse_administration_inside_a_job("setup")
    if ctx.invoked_subcommand is not None:
        return
    config = initialize_setup(force=force)
    print_setup_result(config)


@setup_app.command("doctor")
def setup_doctor_command(
    genome: Optional[str] = typer.Argument(None, help="Configured genome name to inspect and validate"),
    init: bool = typer.Option(False, "--init", help="Create default config before checking if missing"),
):
    """Validate the current Paralome setup."""
    from paralome.setup.site import reporting_on_the_site_file

    # Doctor is what somebody runs when the machine is refused, so
    # it answers beside a site file that is, and names the fault.
    with reporting_on_the_site_file():
        _setup_doctor(genome, init)


def _setup_doctor(genome: Optional[str], init: bool) -> None:
    report = doctor_report(initialize=init, genome=genome)
    print_doctor_report(report)
    # A name nobody registered is not a broken genome, and the
    # answer to it is the list of names that exist -- printed, not pointed at.
    # Bare `setup doctor genome` and bare `setup genome` report the same as
    # `setup genome list`.
    #
    # Still a non-zero exit. The listing is help, not success: `setup doctor
    # $NAME` in a script asked a question about a genome and there is no such
    # genome, which a script has to be able to tell from a healthy one.
    if genome is not None and report.get("genome", {}).get("unknown"):
        from paralome.commands.genome import print_genome_listing

        print_genome_listing()
        raise typer.Exit(code=1)
    if genome is not None and not report.get("genome", {}).get("valid"):
        raise typer.Exit(code=1)


@setup_app.command("where")
def setup_where_command(
    as_json: bool = typer.Option(
        False, "--json",
        help="Machine-readable output, for other programs (e.g. Evidome).",
    ),
) -> None:
    """Where this machine keeps Paralome's shared things, and what decided each.

    Written for Evidome, which asks rather than re-deriving
    Paralome's lookup rules, and readable by a person for the same question.
    Always exits 0: an unconfigured machine is an answer, not an error.
    """
    from paralome.setup.site import where_report

    report = where_report()
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"Paralome {report['paralome_version']}  (installed under {report['install_root']})")
    if report["site_configured"]:
        print(f"This machine : set up -- {report['site_config']}")
        print(f"Shared root  : {report['shared_root'] or '(not recorded)'}")
    else:
        print(f"This machine : NOT set up -- no {report['site_config']}")
        print("               An administrator runs `paralome setup site` to set it up.")
    repo = report["genome_repository"]
    print(f"Genomes      : {repo or '(none recorded)'}"
          + (f"  [{report['genome_repository_source']}]" if repo else ""))
    exists = "" if report["genome_registry_exists"] else "  (does not exist yet)"
    print(f"Registry     : {report['genome_registry']}  [{report['genome_registry_source']}]{exists}")
    if report.get("site_config_problem"):
        # Answered all the same; this is why runs here are refused.
        print(f"Site file    : REFUSED -- {report['site_config_problem']}")


@setup_app.command("reset")
def setup_reset_command(
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm removal without prompting"),
):
    """Remove this account's Paralome configuration, naming what goes with it.

    Lists the genomes and profiles in the file before removing it, and
    removes nothing without --yes. Genomes still published to this machine's
    registry can be taken back with `paralome setup genome sync`; anything
    only in this file cannot.
    """
    path = config_path()
    if not path.exists():
        print(f"Configuration is already absent: {path}")
        return

    genomes: list[str] = []
    profiles: list[str] = []
    try:
        existing = load_config(initialize=False)
        genomes = sorted(str(name) for name in (existing.get("genomes") or {}))
        profiles = sorted(str(name) for name in (existing.get("profiles") or {}))
    except Exception:  # a config too broken to read is one worth removing
        print("This configuration cannot be read; removing it is reasonable.")

    print(f"Configuration : {path}")
    if genomes:
        print(f"Genomes       : {len(genomes)}  ({', '.join(genomes)})")
    if profiles:
        print(f"Profiles      : {len(profiles)}  ({', '.join(profiles)})")
    if genomes or profiles:
        print("")
        print("These go with it. Genomes still published to this machine's")
        print("registry can be taken back with `paralome setup genome sync`;")
        print("anything only in this file cannot.")
        print("")

    if not yes:
        raise typer.BadParameter("Refusing to remove config without --yes")
    path.unlink()
    print(f"Removed configuration: {path}")


def _acl_supported(directory: Path) -> Optional[bool]:
    """Whether POSIX ACLs work here. None when it cannot be determined.

    Probed rather than assumed. An NFSv4 share carries NFSv4 ACLs rather than
    POSIX ones, so setfacl fails with EOPNOTSUPP -- and
    a setup command that silently skipped the step would leave an administrator
    believing content was protected when it was not.
    """
    import shutil
    import subprocess

    if shutil.which("getfacl") is None:
        return None
    try:
        done = subprocess.run(
            ["getfacl", "-c", str(directory)],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # pragma: no cover - a probe that fails proves nothing
        return None
    if done.returncode != 0:
        return False
    return True


@setup_app.command("site")
def setup_site_command(
    root: Optional[str] = typer.Option(
        None, "--root",
        help="The shared directory. <root>/genomes and <root>/registry are created under it.",
    ),
    group: str = typer.Option(
        DEFAULT_GENOME_GROUP, "--group",
        help="The group allowed to contribute genomes",
    ),
    adapter_fasta: Optional[str] = typer.Option(
        None, "--adapter-fasta",
        help="Adapter FASTA every project on this machine trims against, e.g. /path/to/adapters.fa",
    ),
    scratch: Optional[str] = typer.Option(
        None, "--scratch",
        help="Where jobs put temporary files: an absolute path on every node (e.g. /scratch), "
             "TMPDIR (SLURM's per-job directory) or none. Plain `setup site` shows what this machine has.",
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Make the changes. Without this, nothing is written.",
    ),
) -> None:
    """Set this machine up so everyone on it shares one set of genomes.

    It replaces eight commands and two modes an administrator would otherwise
    have to transcribe correctly:

        groupadd, usermod, mkdir, chown, chmod 3775, mkdir, chown, chmod 1777

    with `3775` and `1777` each encoding three separate decisions that look
    like typos. Getting one wrong produces a machine that works until the
    second person uses it.

    WHAT IT DECIDES, and why the two are siblings rather than nested:

        <root>/genomes    3775 root:GROUP   the genome files
        <root>/registry   1777 root:root    who has published what

    The two are siblings by default. A registry inside the genome folder is one `chown -R` away from losing its mode, and
    people run recursive commands on genome repositories. As a sibling it
    cannot be caught by an operation aimed at the genomes.

    DRY RUN IS THE DEFAULT. This creates directories, creates a group and
    writes a file that every account on the machine then obeys; none of that
    should happen because somebody was exploring the help text.
    """
    from paralome.setup.site import reporting_on_the_site_file

    # This is the command that shows and rewrites the site file, so
    # it works beside one that every other command refuses.
    # --adapter-fasta and --group are acted on only with --root; without it they are refused rather
    # than dropped in silence.
    if root is None and (adapter_fasta or group != DEFAULT_GENOME_GROUP):
        given = " and ".join(
            ([f"--adapter-fasta {adapter_fasta}"] if adapter_fasta else [])
            + ([f"--group {group}"] if group != DEFAULT_GENOME_GROUP else [])
        )
        raise typer.BadParameter(
            f"{given} is recorded only with --root; re-run with --root <shared directory> "
            "(the one already recorded, to keep it; `paralome setup site` shows it)."
        )
    with reporting_on_the_site_file():
        if root is None and scratch is not None:
            _setup_scratch(scratch, apply)
            return
        _setup_site(root, group, adapter_fasta, apply, scratch)


def _scratch_report(say) -> None:
    """What this machine has for scratch, and what is recorded."""
    from paralome.scratch import configured_setting, detect, suggest

    found = detect()
    say(f"  recorded       {configured_setting() or '(none: temporary files go beside the results)'}")
    for info in found:
        free = info.get("free_bytes")
        say(f"  {str(info['path']):14s} {info.get('filesystem', '?'):8s} "
            f"{'network' if info.get('network') else 'local':8s} "
            f"{'writable' if info.get('writable') else 'NOT writable':13s} "
            f"{(str(round(free / 1024 ** 3)) + ' GB free') if free is not None else ''}")
    suggestion = suggest(found)
    if suggestion:
        say(f"  suggested      --scratch {suggestion}")
    say("  It must exist at the same path on every compute node; a job on a node without it")
    say("  says so in its log and uses the shared location. TMPDIR uses SLURM's per-job directory.")


def _setup_scratch(scratch: str, apply: bool) -> None:
    """`setup site --scratch X` on its own: record where jobs put temporary files."""
    import os

    from paralome.scratch import SCRATCH_KEY, check_setting
    from paralome.setup.site import load_site_config, save_site_config

    problem = check_setting(scratch)
    if problem:
        raise typer.BadParameter(problem)
    print("\nScratch\n-------")
    _scratch_report(print)
    if not apply:
        print(f"\nDRY RUN -- would record scratch = {scratch}. Re-run with --apply.")
        return
    if os.geteuid() != 0:
        print("Must run as root: this writes a file every account obeys.")
        raise typer.Exit(code=1)
    payload = dict(load_site_config())
    payload[SCRATCH_KEY] = scratch
    print(f"\nRecorded scratch = {scratch} in {save_site_config(payload)}")


def _setup_site(
    root: Optional[str], group: str, adapter_fasta: Optional[str], apply: bool,
    scratch: Optional[str] = None,
) -> None:
    import grp
    import os
    import subprocess

    from paralome.setup.site import (
        KNOWN_KEYS, REGISTRY_DIRNAME, REPOSITORY_DIRNAME, ROOT_KEY,
        ADAPTER_FASTA_KEY,
        load_site_config, save_site_config, site_config_path, site_config_problem,
    )

    def say(text: str = "") -> None:
        print(text)

    def head(text: str) -> None:
        print(f"\n{text}\n{'-' * len(text)}")

    existing = load_site_config()
    problem = site_config_problem()

    def say_the_problem() -> None:
        head("REFUSED")
        say(problem)
        say("")
        say(f"Every run on this machine is refused until {site_config_path()}")
        say("is corrected. A key is corrected by hand: `setup site --apply` keeps")
        say("what the file holds and does not remove a key it does not know.")

    if root is None:
        head("This machine")
        if problem and not existing:
            say(f"Recorded in : {site_config_path()}")
            say_the_problem()
        elif existing:
            # On a machine already set up, say how to migrate to a different site configuration:
            # `setup site --root <new>` prints exactly that, under "Already set up".
            say(f"Recorded in : {site_config_path()}")
            for key, value in sorted(existing.items()):
                say(f"  {key:16s} {value}" + ("" if key in KNOWN_KEYS else "   <- UNKNOWN KEY"))
            if problem:
                say_the_problem()
            recorded = existing.get(ROOT_KEY)
            if recorded:
                say("")
                say(f"  {'genomes':16s} {Path(recorded) / REPOSITORY_DIRNAME}")
                say(f"  {'registry':16s} {Path(recorded) / REGISTRY_DIRNAME}")
            say("")
            # Asked often enough to answer unprompted: --group is not in the
            # file, and that is not an omission. The group is carried by the
            # genomes directory itself, set at creation and inherited by
            # everything added since, so a recorded copy could only ever
            # disagree with it.
            say("The group that may contribute is not recorded here -- it is")
            say("carried by the genomes directory. `paralome setup doctor`")
            say("checks both directories' modes (not their owners).")
            say("")
            say("To change it:")
            say("")
            say("  paralome setup site --root /new/path")
            say("")
            say("which shows what would change and writes nothing without")
            say("--apply. It does not move anything: genomes already registered")
            say("keep their recorded paths. Moving the contents is yours to do.")
            head("Scratch (temporary files of jobs)")
            _scratch_report(say)
        else:
            say("No shared setup has been recorded on this machine.")
            say("")
            say("  paralome setup site --root /storage/paralome")
            say("")
            say("creates the genome repository and the registry under that")
            say("directory and records where they are, so every account -- "
                "including")
            say("ones that do not exist yet -- finds them without being told.")
            # The scratch report is shown here too, as --scratch's help says.
            head("Scratch (temporary files of jobs)")
            _scratch_report(say)
        return

    shared = Path(root).expanduser().resolve()
    repository = shared / REPOSITORY_DIRNAME
    registry = shared / REGISTRY_DIRNAME

    head("Mode")
    if apply:
        if os.geteuid() != 0:
            say("Must run as root: this writes a file every account obeys, into")
            say(f"a root-owned directory ({site_config_path().parent}).")
            raise typer.Exit(code=1)
        say("APPLY -- changes will be made.")
    else:
        say("DRY RUN -- nothing will be written. Re-run with --apply.")

    head("What will exist")
    say(f"  {str(repository):50s} 3775 root:{group}")
    say(f"  {str(registry):50s} 1777 root:root")
    say(f"  {str(site_config_path()):50s} 0644 root:root")
    say("")
    say("3775 is setgid + sticky + group write, and each bit is doing a job:")
    say("  group write   any member of the group may add a genome")
    say("  setgid        what they add inherits the group, so the next person")
    say("                can read it without anyone remembering to chgrp")
    say("  sticky        but only the owner may delete their own genome")
    say("")
    say("1777 is the /tmp arrangement, for the same reason: everyone may")
    say("publish their own registry file, nobody may remove another's.")

    if existing and existing.get(ROOT_KEY) not in (None, str(shared)):
        head("Already set up")
        say(f"This machine already records {existing.get(ROOT_KEY)}.")
        say("")
        say("Changing it does NOT move anything. Genomes already registered")
        say("keep their recorded paths, and registry files written to the old")
        say("location stay there -- Paralome will simply stop looking at them.")
        say("Move the contents yourself if that is what you mean.")

    head("Group")
    try:
        grp.getgrnam(group)
        say(f"  {group} exists")
    except KeyError:
        say(f"  {group} does not exist")
        if apply:
            subprocess.run(["groupadd", group], check=True)
            say(f"  + groupadd {group}")
        else:
            say(f"  would run: groupadd {group}")
    say("")
    say("Membership is what grants access, not file ownership. Add people with")
    say(f"  usermod -aG {group} <account>")
    say("and have them log out and back in -- a live session keeps the groups")
    say("it started with.")

    head("Directories")
    for target, mode, owner_group in (
        (repository, 0o3775, group),
        (registry, 0o1777, "root"),
    ):
        if apply:
            target.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.chown(target, user="root", group=owner_group)
            os.chmod(target, mode)
            say(f"  + {target}  {oct(mode)[-4:]} root:{owner_group}")
        else:
            state = "exists" if target.is_dir() else "would be created"
            say(f"  {target}  ({state}) -> {oct(mode)[-4:]} root:{owner_group}")

    head("Group write on new content")
    say("Setgid gives new content the right GROUP. It does not give it group")
    say("WRITE -- that comes from each contributor's umask, which Paralome")
    say("cannot set. A umask of 002 would let any member delete the contents")
    say("of somebody else's genome.")
    say("")
    supported = _acl_supported(repository) if repository.is_dir() else None
    acl_failed = False
    if supported is True:
        if apply:
            done = subprocess.run(
                ["setfacl", "-d", "-m", "g::rx", str(repository)],
                check=False, capture_output=True, text=True,
            )
            # Success is reported only when setfacl succeeded.
            if done.returncode == 0:
                say("  + default ACL set: new content is group r-x whatever the umask")
            else:
                reason = (done.stderr or "").strip() or f"exit status {done.returncode}"
                say(f"  ! setfacl FAILED ({reason}); no default ACL is set.")
                acl_failed = True
        else:
            say(f"  would run: setfacl -d -m g::rx {repository}")
    if supported is False or acl_failed:
        if supported is False:
            say("  This filesystem does not carry POSIX ACLs -- NFS, most likely.")
        say("  Pin the umask for every account instead:")
        say("")
        say("      echo 'umask 022' > /etc/profile.d/paralome_umask.sh")
        say("      chmod 644 /etc/profile.d/paralome_umask.sh")
        say("")
        say("  Check it before trusting it: a syntax error in /etc/profile.d")
        say("  breaks login for every account including root.")
    elif supported is None:
        say("  (checked once the repository exists)")

    if apply:
        head("Recorded")
        # Start from what is recorded. This rebuilt the file from
        # the root alone, so re-running without --adapter-fasta dropped the
        # recorded adapter file (and any genome_repository or site_registry
        # written by hand), and every later project trimmed on fastp's guess.
        payload = dict(load_site_config())
        payload[ROOT_KEY] = str(shared)
        if scratch is not None:
            from paralome.scratch import SCRATCH_KEY, check_setting

            if check_setting(scratch):
                raise typer.BadParameter(check_setting(scratch))
            payload[SCRATCH_KEY] = scratch
        if adapter_fasta:
            # The default is to trim against this machine's adapter
            # file. Recorded once here rather than retyped per project -- and kept out of the package, because
            # the path is a fact about this machine's sequencing kit.
            payload[ADAPTER_FASTA_KEY] = str(Path(adapter_fasta).expanduser().resolve())
        written = save_site_config(payload)
        say(f"  {written}")
        say("")
        say("Every account on this machine now finds the repository and the")
        say("registry without being told, including accounts created later.")

    # Asked again: --apply has just rewritten a file that could not be read.
    problem = site_config_problem()
    if problem:
        say_the_problem()

    head("Then")
    say("  paralome setup doctor        # confirms the modes and the group")
    say("  paralome setup genome list   # what a user sees")
