#!/usr/bin/env python3
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

"""The human command line. Everything here is something a person types.

Only what a person would type is in the help or the CLI; anything the
software calls on behalf of the user is not. The machine surface -- the stage
drivers the supervisor invokes and the array-task internals written into
sbatch bodies -- lives on `paralome-worker`, a separate console script. Not
hidden -- absent. `paralome --help` cannot leak a worker command, because the
worker commands are not registered on this application.

What remains is organised by what the user is trying to do, with the analysis
named first: `paralome scRNA ...`, not `paralome pipeline run --profile snRNA`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from typer.core import TyperGroup

from paralome.errors import REGISTRY, format_code, install_error_handler, lookup

from paralome.commands.analyses import ASSAY_APPS
from paralome.commands.atac import collect_command
from paralome.commands.batch import batch_app
from paralome.commands.pipeline import (
    pipeline_extract_command,
    pipeline_status_command,
)
from paralome.commands.setup import setup_app


# Typer's rich tracebacks are off. They are on by default and they
# print every local variable in every frame -- which is how a one-line "column
# not found" arrived as several screens containing the entire pseudobulk sample
# list, which was far too much error output for a command line.
# The traceback still exists; it goes to the log (see paralome.errors).
#: The panels, in the order they are to be printed. Options,
#: Pipelines, then Utilities.
#:
#: Options is Click's own and always comes first. The other two are ours, and
#: forcing their order takes a group class because BOTH typer versions in play
#: decide it by accident rather than by intent: rich_utils builds a
#: ``panel_to_commands`` dict and prints it in insertion order, so the panel
#: order is whichever panel the first listed command happens to belong to.
#:
#: The two disagree about that order, which is the argument for pinning it.
#: typer 0.12 sorts commands alphabetically, so `ATAC` comes first and
#: Pipelines wins; typer 0.27 lists plain commands before sub-apps, so
#: `extract` comes first and Utilities wins. Same source, two different help
#: screens, and the one a site renders is not the one these tests render.
PANEL_ORDER = ("Pipelines", "Utilities")


class _OrderedPanels(TyperGroup):
    """List commands grouped by panel, in PANEL_ORDER, alphabetical within each.

    Overriding ``list_commands`` is enough: the renderer walks that sequence to
    build its dict, so controlling the sequence controls both the panel order
    and the order of the commands inside each panel.
    """

    def list_commands(self, ctx):
        # ALPHABETICAL INSIDE EACH PANEL, which also keeps the single-cell
        # commands together: sorting the assays gives
        #
        #     ATAC  DNA  RNA  scRNA  snATAC
        #
        # so the three bulk assays come first and the two single-cell ones land
        # side by side at the end, without anybody having to encode that.
        #
        # typer 0.12's list_commands sorts and 0.27's does not, so this does
        # not call super() -- doing the sort here is what makes the two agree.
        def rank(name: str) -> tuple:
            command = self.commands[name]
            panel = getattr(command, "rich_help_panel", None) or ""
            try:
                position = PANEL_ORDER.index(panel)
            except ValueError:
                # A panel nobody named goes last rather than first, so adding a
                # command without one cannot silently displace the pipelines.
                position = len(PANEL_ORDER)
            return (position, name.lower())

        return sorted(self.commands, key=rank)


app = typer.Typer(
    cls=_OrderedPanels,
    pretty_exceptions_enable=False,
    # Typer adds --install-completion and --show-completion to every
    # root app unless told not to. They are turned off entirely.
    #
    # They took two of the three lines in the top-level Options box, so the
    # first thing anyone reading `paralome --help` saw under Options was two
    # commands they would never run, above the one they wanted.
    #
    # --install-completion was also the only command in Paralome that edits a
    # user's shell configuration -- appending to ~/.bashrc, writing ~/.zfunc --
    # and it did it with no dry run and little account of what it changed. That
    # sits badly beside `setup site`, which writes nothing without --apply.
    add_completion=False,
    help=(
        "Paralome: reproducible sequencing analyses.\n\n"
        "Name the analysis and hand it a cohort:\n"
        "    paralome scRNA --rds obj.rds --project treated_vs_control --statistic t --factor treatment:VEH --submit\n\n"
        "Everything is checkpointed, so re-running the same command resumes at "
        "the first incomplete stage."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)

#: Commands that must keep working even when the install itself is refused.
#: A refusal nobody can read the explanation of is just a broken program --
#: `explain 408` is the whole answer, and `version` is how you find out which
#: copy is objecting.
# `version` and `docs` became root OPTIONS, so the only command that
# still has to be let past the private-install refusal is `explain`. It reads
# nothing but its own error registry, and somebody hitting PLM-0408 needs a way
# to look up PLM-0408.
#
# --version and --docs need no entry here at all: an eager option runs during
# parsing, before this callback's body, so both answer on a private install
# without being exempted. That is deliberate -- "which paralome am I running"
# is the first question somebody refused by 0408 will ask.
_ALWAYS_ALLOWED = {"explain"}


#: What may run inside a SLURM job: essentially only the true genomics
#: pipelines and batch need to be submittable.
#:
#: `batch` is here because it IS a submitted thing, and the analyses are here
#: because batch invokes the HUMAN command for each family from inside an array
#: task -- see commands/batch.py's _paralome_argv. An allowlist of the worker
#: binary alone would break every batch run.
#:
#: `extract` is here because it is WORK, not administration. It is the heaviest
#: single-cell step -- an R subprocess over an object that can be tens of
#: gigabytes -- and sbatching it for a large RDS is a reasonable thing to do.
#: It writes nothing shared: extraction goes into the project directory the
#: caller named, not into the config or the registry.
#:
#: `proteomics` and `olink` run their stages in the command itself, and their panels use every CPU
#: the process may use, so they are submittable too. They write only into the --project and data
#: folders they are given, as extract does. Every command in the Pipelines panel is on this list.
SUBMITTABLE = frozenset({"ATAC", "DNA", "RNA", "scRNA", "snATAC", "proteomics", "olink", "batch", "extract"})


#: `setup` subcommands that only answer a question and write nothing.
#: Evidome asks `paralome setup where --json` for the shared reference root, and
#: in a pipeline it runs inside the functional_evidence SLURM job; refused there
#: with PLM-0116, Evidome found no reference data and every DNA and RNA run's
#: functional evidence was skipped.
READ_ONLY_SETUP = frozenset({"where"})


def _refuse_administration_inside_a_job(subcommand: Optional[str]) -> None:
    """Administration belongs at a prompt, not in a queue.

    The config lives in ~/.paralome/config.json, on shared storage, and the
    tool paths recorded in it are what every run invokes. `paralome setup`
    inside a job would record THAT NODE's paths into the file the login node
    reads, silently repointing every future run at one node's binaries.

    The rest are refused for a smaller reason: they ask questions, write to a
    shared registry, or print a report nobody is watching.

    NOT APPLIED TO paralome-worker, which is a different console script with
    its own app -- the machine-facing CLI is what generated job scripts invoke
    and is submittable by definition.
    """
    if subcommand is None or subcommand in SUBMITTABLE:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        return

    from paralome.errors import ParalomeError

    # The console prints two lines and no "Fix:" -- see render_console, which
    # prints only the number and what happened. So the instruction goes here.
    raise ParalomeError(
        116,
        f"`paralome {subcommand}` inside SLURM job "
        f"{os.environ.get('SLURM_JOB_ID')}. Run it on the login node.",
    )


def _print_version_and_exit(value: bool) -> None:
    """Print the version, the install, and the AGPL notice. An option.

    An option rather than a command, as convention has it -- `--version` is
    where everyone looks first -- and it takes no arguments.
    """
    if not value:
        return
    from paralome.binaries import is_development

    # The version of the code that is running; installed metadata can be older.
    from paralome.version import RELEASE_LABEL, __version__ as resolved

    try:
        metadata = package_version("paralome")
    except PackageNotFoundError:  # pragma: no cover - source checkout
        metadata = None
    channel = "development" if is_development(resolved) else (RELEASE_LABEL or "production")
    typer.echo(f"Paralome version {resolved} ({channel})")
    if metadata is not None and metadata != resolved:
        typer.echo(f"  (the installed package metadata says {metadata}; it is out of date)")
    # Where the running code is, as Evidome reports it. install_root() names the interpreter's
    # prefix, which for a per-version prefix install is the system's, not this version's.
    typer.echo(f"  code imported from {Path(__file__).resolve().parent}")
    typer.echo(f"  running {sys.executable}")
    # AGPL section 13 asks that a program offering an interactive interface
    # show "Appropriate Legal Notices" -- the copyright, the absence of a
    # warranty, and how to read the licence. The licence says a prominent item
    # in a command list satisfies this, and this is the first thing anybody
    # runs to ask what they are running, so the notice belongs here.
    typer.echo("")
    typer.echo("Copyright (C) 2026 Kenneth L. Jones")
    typer.echo("Licensed under the GNU Affero General Public License, version 3.")
    typer.echo("This program comes with ABSOLUTELY NO WARRANTY.")
    typer.echo("Full terms: <https://www.gnu.org/licenses/agpl-3.0.html>")
    typer.echo("A separate commercial licence is available; see NOTICE.")
    typer.echo("")
    raise typer.Exit()


def _refresh_docs_and_exit(value: bool) -> None:
    """Copy the shipped documents where the user can find them, and say where.

    The shipped docs sit in an install directory users should not
    have to look in, so they are copied to ~/.paralome beside the user's config
    and --docs lists where the files are. It does not print a document.

    REFRESHED EVERY TIME, which is the point rather than an optimisation --
    see docs.refresh_user_copy.
    """
    if not value:
        return
    from paralome import docs as shipped

    destination, placed = shipped.refresh_user_copy()
    typer.echo(f"Paralome documentation was written to {destination}:")
    typer.echo("")
    for filename in placed:
        size = (destination / filename).stat().st_size
        typer.echo(f"  {filename}  ({size // 1024} KB)")
    typer.echo("")
    # The refresh is not mentioned; no one needs to know that.
    # It matters that it happens, not that the reader is told it happens --
    # the value of always-current documents is that nobody has to think about
    # whether they are current.
    typer.echo("Documentation to provide an AI assistant for customized help:")
    typer.echo(f"  {shipped.BRIEF}")
    typer.echo("")
    raise typer.Exit()


@app.callback()
def _before_any_command(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_print_version_and_exit, is_eager=True,
        help="Print the version and which install is running, then exit.",
    ),
    docs: bool = typer.Option(
        False, "--docs", callback=_refresh_docs_and_exit, is_eager=True,
        help="Refresh the documents into ~/.paralome/docs and list them.",
    ),
) -> None:
    """Refuse a `pip install --user` copy before it does any work.

    0.4. Individual per-account installations are not allowed: they lead to
    isolation and genome redundancy, which shared installs exist to avoid.

    Here rather than at install time because a wheel cannot refuse at install
    time -- wheels are unpacked, not executed. And doing nothing produces no
    failure at all: a private install works for everything except publishing,
    so the account succeeds quietly with its own duplicate genomes, which is
    the outcome being prevented rather than an acceptable fallback.
    """
    from paralome.binaries import refuse_private_install_on_a_shared_machine

    # This test came second, so `paralome explain 116` typed inside a
    # SLURM allocation was itself refused with PLM-0116.
    if ctx.invoked_subcommand in _ALWAYS_ALLOWED:
        return
    # `setup` is refused in its own callback, which knows the
    # subcommand (`setup where` only reads, and is answered inside a job).
    if ctx.invoked_subcommand != "setup":
        _refuse_administration_inside_a_job(ctx.invoked_subcommand)
    refuse_private_install_on_a_shared_machine()


# --- Analyses ---------------------------------------------------------------
# The group IS the verb -- there is no `run` and no `continue`. Registered under
# their canonical, case-sensitive acronyms. scRNA is not scrna, so there are
# no lower-case aliases; `paralome scrna` fails with "No such command", which is
# loud, rather than quietly resolving to something the user did not type.
#
# TWO PANELS, because most of this list is not a pipeline: the pipelines a user
# comes to run, and the utilities that support them, in separate boxes.
PIPELINES = "Pipelines"
UTILITIES = "Utilities"

for _name, _assay_app in ASSAY_APPS.items():
    app.add_typer(_assay_app, name=_name, rich_help_panel=PIPELINES)

# Proteomics runs its stages in the command itself; it is
# not yet one of the orchestrated profiles.
from paralome.commands.proteomics import proteomics_command  # noqa: E402

app.command("proteomics", rich_help_panel=PIPELINES)(proteomics_command)

# Olink panels, the same way.
from paralome.commands.olink import olink_command  # noqa: E402

app.command("olink", rich_help_panel=PIPELINES)(olink_command)

# `batch` is a UTILITY, not a primary pipeline. It runs pipelines rather than being one -- somebody
# choosing what analysis to do is not choosing between scRNA and batch.
app.add_typer(
    batch_app, name="batch", rich_help_panel=UTILITIES,
    help="Run many projects with family-level admission control.",
)

# --- Utilities --------------------------------------------------------------
# extract and QC are single-cell preparation rather than assays: you run them
# against an object before choosing what to do with it, and they take no
# --factor. They sit here for that reason, not because they are
# unimportant -- `extract` is the first command of any single-cell project.
app.command("extract", rich_help_panel=UTILITIES)(pipeline_extract_command)
# Single-cell QC is part of `extract`, which prints the metadata column table.
# `paralome-worker atac singlecell-inspect` gives the full per-column table with value counts.
app.command("collect", rich_help_panel=UTILITIES)(collect_command)
app.command("status", rich_help_panel=UTILITIES)(pipeline_status_command)
# What a run will do is shown by --dry-run: the preflight validates (every run does, whether asked
# or not) and the dispatch plan is printed, with no run parameters re-typed into a second command.
# EVERYTHING THAT TOUCHES SETUP OR CONFIGURATION LIVES UNDER `setup`, config and doctor included,
# each with one implementation.
app.add_typer(
    setup_app, name="setup", rich_help_panel=UTILITIES,
    help="Install, configure and check this machine",
)


@app.command(rich_help_panel=UTILITIES)
def explain(
    code: str = typer.Argument(
        ...,
        help="An error number, e.g. 104 or PLM-0104",
    ),
) -> None:
    """Explain a numbered Paralome error.

    The console prints the number and what went wrong; this is the rest of it,
    with the fix. Same text as the appendix in docs/MANUAL.md, from the same
    source, so the two cannot disagree.
    """
    spec = lookup(code)
    if spec is None:
        known = ", ".join(format_code(c) for c in sorted(REGISTRY))
        typer.echo(f"No such error code: {code}\n\nKnown codes: {known}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{format_code(spec.code)}  {spec.summary}\n")
    typer.echo(spec.explanation)
    typer.echo(f"\nFix: {spec.fix}")


# The console script is this, not `app`, so that every failure passes through
# the numbering and logging handler on its way out.
main = install_error_handler(app)


if __name__ == "__main__":  # pragma: no cover
    main()
