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

"""The assay-first human CLI: ATAC, snATAC, RNA, scRNA, DNA.


THE GROUP IS THE VERB. There is no `run` sub-command and no `continue`:

    paralome scRNA --rds X.rds --project A_vs_NEG --control "NEG:..." --submit

`run` was redundant -- invoking the analysis with a cohort already says you want
it run. `continue` was redundant twice over: it took the same 82 options as
`run` and differed only by `check_complete=True, run_all=False`, while the
orchestrator already selects the first INCOMPLETE stage from the checkpoints.

Re-running your own command is the resume path.

--profile is gone as well; the profile IS the command name.

WHY THIS DELEGATES RATHER THAN REDECLARES. `pipeline run` carried 82 options.
Copying them into five assay groups would be five copies of one contract, which
is precisely the failure mode Patches 146-153 were spent on -- one concept
written more than once, with nothing keeping the copies in step. Instead each
group forwards its argv to the single existing implementation with the profile
injected, so there remains exactly one declaration of what an analysis takes.

THE DELEGATION WAS RIGHT AND THE HIDING WAS NOT.

Forwarding argv meant the five groups declared no options at all, so
`paralome scRNA --help` printed a help line, `--help` and four stage
subcommands -- and nothing whatsoever about --rds, --project, --control or
--submit. 

The premise above ("copying them would be five copies") was true; the conclusion
drawn from it was not. Not copying them never required hiding them. Each group
now declares the subset of the ONE signature that applies to it, taken from
pipeline/assay_options.py, which is data rather than a second declaration -- and
a test pins that table to the signature so neither can move without the other.

WHY THE SIGNATURE AND NOT THE CLICK PARAMS. The obvious generator reads the
built click command and filters `cmd.params`. It was not used: typer 0.27
vendors its own click, so `TyperOption` inherits `typer._click.core.Parameter`
and `isinstance(param, click.Option)` is False -- while on typer 0.12, the pin
of the time, it was True. Building from `inspect.signature` and handing that to
Typer keeps Typer responsible for whichever click it has.

Paralome now requires typer 0.27, and this module imports
nothing from click -- click is no longer installed with Paralome. The context
type and echo come from typer's public names.
"""

import inspect
from typing import Any, Callable, Dict

import typer
from typer.core import TyperGroup

from paralome.pipeline import assay_options
from paralome.pipeline.profiles import PipelineProfile


_ASSAY_HELP = {
    PipelineProfile.ATAC: "Bulk ATAC-seq: peaks, quantification, differential accessibility, motifs.",
    PipelineProfile.SNATAC: "Single-nuclei ATAC: pseudobulk, differential accessibility, motifs.",
    PipelineProfile.RNA: "Bulk RNA-seq: quantification and differential expression.",
    PipelineProfile.SCRNA: "Single-cell RNA: pseudobulk and differential expression.",
    PipelineProfile.DNA: "DNA variants (--exome/--genome): calling, annotation, segregation, reporting.",
}


#: Implementation parameters the callback supplies itself, so they must be left
#: out of BOTH the offered options and the filled defaults -- passing one twice
#: is a TypeError at the call, which is how this list earned its own name.
_BOUND_BY_THE_COMMAND_NAME = frozenset({"profile"})

#: Set by parse_args so the callback can tell `paralome scRNA` from
#: `paralome scRNA --rds x.rds`. Every option has a default, so "the user gave
#: nothing" cannot be read off the parsed values -- a run with only defaults and
#: a bare invocation look identical by the time the callback sees them.
_GIVEN_ANY_ARGS = "paralome.analysis_given_args"


def _signature_for(profile: PipelineProfile) -> Dict[str, inspect.Parameter]:
    """The parameters of the one implementation that this assay offers.

    Declaration order is preserved from the signature, so an option's position
    on the help screen is decided by assay_options.sort_key rather than by the
    accident of where it sits in a 157-line parameter list.
    """
    from paralome.commands.pipeline import pipeline_run_command

    wanted = assay_options.options_for(profile.value)
    declared = inspect.signature(pipeline_run_command).parameters
    return {name: declared[name] for name in wanted if name in declared}


def _resolved_hints() -> Dict[str, Any]:
    """The implementation's annotations as objects rather than strings.

    ``commands/pipeline.py`` uses ``from __future__ import annotations``, so its
    annotations are strings naming Path, List, Optional and so on. Typer calls
    ``get_type_hints`` on the callback, which resolves strings against the
    FUNCTION's module globals -- and the generated callback lives here, where
    those names are not imported. So they are resolved once against
    pipeline.py's own globals, and the resolved objects handed over instead of
    the strings.
    """
    import typing

    from paralome.commands import pipeline as pipeline_module

    return typing.get_type_hints(pipeline_module.pipeline_run_command)


def _defaults_for(profile: PipelineProfile) -> Dict[str, Any]:
    """What to pass for the options this assay does NOT offer.

    The implementation still takes all of them. An option scRNA never shows has
    to arrive as its declared default rather than as the OptionInfo object left
    behind by an unpassed typer.Option -- which is truthy, and fails the moment
    anything reads it.
    """
    from paralome.commands.pipeline import pipeline_run_command

    offered = assay_options.options_for(profile.value)
    out: Dict[str, Any] = {}
    for name, param in inspect.signature(pipeline_run_command).parameters.items():
        if name in offered or name in _BOUND_BY_THE_COMMAND_NAME:
            continue
        default = param.default
        value = getattr(default, "default", default)
        # A List[str] option defaults to a shared [] in the signature; hand each
        # call its own so one run cannot append into the next one's default.
        out[name] = list(value) if isinstance(value, list) else value
    return out


class AnalysisGroup(TyperGroup):
    """A group whose own options are the analysis, and a bare word selects a step.

    The group DECLARES the analysis's options, so `--rds` is a parameter Click
    knows and the help documents it. Ordinary Group parsing handles both
    shapes: options are consumed as options, and a leading bare word resolves
    as a step. (``allow_extra_args``/``ignore_unknown_options`` in a
    sub-``Typer``'s ``context_settings`` are not carried over by
    ``add_typer``, so they cannot be relied on for this.)

    This must subclass ``typer.core.TyperGroup``, not ``click.Group``: Typer
    asserts the class identity when building the group and rejects a plain
    Click group outright, and typer 0.27 vendors its own click, so
    `click.Group` is not even the right base class there.

    What it adds is one line of bookkeeping. Every option has a default, so by
    the time the callback runs, `paralome scRNA` and a run where the user
    happened to pass nothing but defaults are indistinguishable -- and the
    first should print help while the second should not.
    """

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        ctx.meta[_GIVEN_ANY_ARGS] = bool(args)
        return super().parse_args(ctx, args)


def _build_entry(profile: PipelineProfile) -> Callable[..., None]:
    """The group callback, carrying this assay's own options in its signature.

    ``__signature__`` is what Typer reads, so assigning a synthesised one is how
    five commands get five different option sets out of one declaration. The
    function itself takes ``**kwargs`` because the names differ per profile;
    Typer never sees that, and the signature it does see is exact.
    """
    offered = _signature_for(profile)
    defaults = _defaults_for(profile)

    def _entry(ctx: typer.Context, **given: Any) -> None:
        # A step was named (e.g. `paralome scRNA samples`); let Typer route
        # to it rather than treating this invocation as a run.
        if ctx.invoked_subcommand is not None:
            return
        if not ctx.meta.get(_GIVEN_ANY_ARGS):
            typer.echo(ctx.get_help())
            raise typer.Exit(0)

        if given.get("stratify_value") and given.get("stratify_skip"):
            raise typer.BadParameter(
                "--stratify-value names the strata to RUN and --stratify-skip "
                "names the strata to EXCLUDE; pass one or the other, not both."
            )

        from paralome.commands.pipeline import _run_common

        _run_common(
            **defaults,
            **given,
            # Bound by which command was typed rather than offered as an option, so
            # `paralome scRNA --profile DNA` cannot run DNA.
            profile=profile.value,
            title=f"{profile.value} run",
            check_complete=False,
            run_all=True,
        )

    hints = _resolved_hints()
    context = inspect.Parameter(
        "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context,
    )
    ordered = sorted(offered, key=assay_options.sort_key)
    parameters = [context] + [
        offered[name].replace(
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=hints[name],
            default=_panelled(offered[name].default, assay_options.OPTION_PANEL[name],
                              help=_profile_help(profile, name)),
        )
        for name in ordered
    ]
    _entry.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    _entry.__annotations__ = {
        "ctx": typer.Context,
        **{name: hints[name] for name in ordered},
        "return": None,
    }
    _entry.__doc__ = _ASSAY_HELP[profile]
    _entry.__name__ = f"{profile.value}_run"
    return _entry


def _profile_help(profile: PipelineProfile, name: str) -> Any:
    """Help text that has to name this analysis's own stages, or None.

    --stop-at offers the stages this analysis runs, read from its stage graph
    (RNA and DNA have no motif stage).
    """
    if name != "stop_at":
        return None
    from paralome.pipeline.profiles import PROFILE_REGISTRY

    stages = PROFILE_REGISTRY[profile].stage_graph
    return (
        "Run through this stage and stop: " + ", ".join(stages) + ". "
        "Earlier stages are checkpointed and reused; run again without "
        "--stop-at to continue past it."
    )


def _panelled(option: Any, panel: str, *, help: Any = None) -> Any:
    """A copy of one typer.Option with its help panel set.

    COPIED rather than mutated. The same OptionInfo object is shared by every
    profile that offers the option -- they agree on the panel, so mutating would
    work today, but it would make the five commands quietly interdependent and
    the first per-profile difference would be a mystery.
    """
    import copy

    duplicate = copy.copy(option)
    if hasattr(duplicate, "rich_help_panel"):
        duplicate.rich_help_panel = panel
    if help is not None and hasattr(duplicate, "help"):
        duplicate.help = help
    return duplicate


def build_assay_app(profile: PipelineProfile) -> typer.Typer:
    """One top-level analysis command whose group callback runs the analysis."""
    assay = typer.Typer(
        cls=AnalysisGroup,
        help=_ASSAY_HELP[profile],
        context_settings={"help_option_names": ["-h", "--help"]},
        invoke_without_command=True,
        no_args_is_help=False,
    )
    assay.callback()(_build_entry(profile))
    return assay


ATAC_app = build_assay_app(PipelineProfile.ATAC)
snATAC_app = build_assay_app(PipelineProfile.SNATAC)
RNA_app = build_assay_app(PipelineProfile.RNA)
scRNA_app = build_assay_app(PipelineProfile.SCRNA)
DNA_app = build_assay_app(PipelineProfile.DNA)


ASSAY_APPS: dict[str, typer.Typer] = {
    PipelineProfile.ATAC.value: ATAC_app,
    PipelineProfile.SNATAC.value: snATAC_app,
    PipelineProfile.RNA.value: RNA_app,
    PipelineProfile.SCRNA.value: scRNA_app,
    PipelineProfile.DNA.value: DNA_app,
}


# --------------------------------------------------------------------------
# Per-assay steps -- the escape hatches for running one stage alone.
# --------------------------------------------------------------------------
#
# Registered under the assay that owns them and named for what they do. The
# modality prefix is dropped because the group already says it: `paralome
# scRNA samples`, not `rna singlecell-rna-pseudobulk`.

from paralome.commands.atac import (  # noqa: E402
    motif_partner_shift_command,
    motifs_command,
    singlecell_differential_command,
    singlecell_atac_command,
)
from paralome.commands.rna import (  # noqa: E402
    detect_strandedness_command,
    singlecell_rna_differential_command,
    singlecell_rna_command,
)

snATAC_app.command("samples")(singlecell_atac_command)
snATAC_app.command("differential")(singlecell_differential_command)
snATAC_app.command("motifs")(motifs_command)

scRNA_app.command("samples")(singlecell_rna_command)
scRNA_app.command("differential")(singlecell_rna_differential_command)

# Registered on both ATAC groups from one callable: it reads written motif_pair_statistics tables and
# does not care which pipeline produced them. Bulk and single-cell runs emit
# the identical schema, so a second registration is the whole port.
#
# It is a SECONDARY analysis -- deliberately absent from every stage graph: a
# new analysis set run after the primary analyses have been run and evaluated,
# still part of the ATAC pipeline, as PCA is to RNA.
ATAC_app.command("motif-partners")(motif_partner_shift_command)
snATAC_app.command("motif-partners")(motif_partner_shift_command)

RNA_app.command("strandedness")(detect_strandedness_command)
