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
from typing import Any, List, Optional, Sequence, Tuple

import typer

from paralome.align.bwa import normalize_aligner

from paralome.layout import DATA_DIRNAME, data_workdir, extraction_dir as layout_extraction_dir
from paralome.errors import ParalomeError
from paralome.singlecell.replicates import REPLICATE_COLUMN_HELP, RETIRED_GROUP_BY_HELP, pile_columns
from paralome.stats import FACTOR_HELP, STATISTIC_HELP, STRATIFY_HELP
from paralome.pipeline.logging import writable_reason
from paralome.validate import (
    check_bounded_number,
    check_input_path,
    check_metadata_columns,
    extraction_metadata_columns,
)
from paralome.pipeline.state import build_project_state, write_project_state
from paralome.pipeline.dashboard import print_pipeline_dashboard
from paralome.pipeline.orchestrator import DEFAULT_FUSE_STAGES, run_pipeline_next_stage
from paralome.pipeline.supervisor import run_pipeline_to_completion
from paralome.pipeline.project import cfg_get, choose_path, derive_project_workdirs, load_pipeline_config, stage_resource
from paralome.pipeline.resource_catalog import resolve_genome_resources
from paralome.pipeline.profiles import (
    PipelineProfile,
    star_output_filters,
    resolve_pipeline_profile,
    resolve_stage_alias,
)

# The --profile help is generated from the enum, so the help cannot drift from the accepted set.
_PROFILE_HELP = "Pipeline modality (case-sensitive): " + ", ".join(
    profile.value for profile in PipelineProfile
)
from paralome.commands.atac import DEFAULT_ALPHA_SPEC, _parse_alpha
from paralome.commands.rna import _parse_threshold_list
from paralome.atac.singlecell import read_cell_metadata, run_seurat_extraction
from paralome.singlecell.qc import write_metadata_qc

# There is no `pipeline` group. pipeline_run_command is a plain function: the ONE declaration of
# what a run takes -- analyses.py builds the five assay signatures from it -- and callable directly,
# which is how tests drive a run without going through argv.


def _validate_mem(value: str, *, source: str) -> str:
    import re
    token = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGTP])?", token, re.IGNORECASE):
        raise typer.BadParameter(f"Invalid memory value {token!r} from {source}; use values such as 4G or 32000M")
    return token.upper()


def _stage_mem_names() -> tuple[str, ...]:
    """Every stage name a memory setting may be given for.

    The stages of every pipeline, taken from the profile registry so
    a new stage is accepted without anyone remembering to list it here, plus
    ``variant_calling`` -- the name the configuration has always used for the
    stage the pipeline calls ``variants``.
    """
    from paralome.pipeline.profiles import PROFILE_REGISTRY

    names: list[str] = ["variant_calling"]
    for spec in PROFILE_REGISTRY.values():
        for stage in spec.stage_graph:
            if stage not in names:
                names.append(stage)
    return tuple(names)


def _parse_stage_mem(values: List[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise typer.BadParameter("--stage-mem requires STAGE=MEM, for example --stage-mem alignment=40G")
        stage, value = (part.strip() for part in raw.split("=", 1))
        if not stage:
            raise typer.BadParameter("--stage-mem stage name cannot be empty")
        parsed[stage] = _validate_mem(value, source=f"--stage-mem {stage}")
    return parsed


def _resolved_adapter_fasta(cfg: Any, given: Optional[Path]) -> Optional[Path]:
    """The adapter FASTA to trim against: flag, then config, then site.

    Returns None when no layer names a file, which leaves fastp to auto-detect.
    """
    if given is not None:
        return Path(given)
    configured = cfg_get(cfg, "fastq_qc", "adapter_fasta", None)
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser()
    from paralome.setup.site import site_adapter_fasta

    return site_adapter_fasta()


#: The annotation engines `paralome annotate --engine` accepts, by every
#: spelling it accepts. Kept beside the check below; the stage is the authority.
_ANNOTATION_ENGINES = {
    "annovar": "annovar",
    "snpeff": "snpeff",
    "snp_eff": "snpeff",
    "snp-eff": "snpeff",
}


def _annotation_engines(value: str, *, source: str) -> list[str]:
    """The engine chain a DNA run will annotate with, checked before it starts.

    `none` was silently run as ANNOVAR, and any other wrong word --
    `all` included, which `variants freebayes` advertises for its own option --
    was refused by the annotation stage, after calling had finished.
    """
    engines: list[str] = []
    for item in str(value or "").split(","):
        word = item.strip().lower()
        if not word:
            continue
        if word not in _ANNOTATION_ENGINES:
            extra = (
                " A DNA run always annotates; there is no way to skip that stage."
                if word == "none" else ""
            )
            raise ParalomeError(
                108,
                f"{source} {item.strip()!r} is not an annotation engine. Use annovar, "
                f"snpeff, or both as annovar,snpeff.{extra}",
            )
        engine = _ANNOTATION_ENGINES[word]
        if engine in engines:
            raise ParalomeError(108, f"{source} names {engine} twice: {value!r}")
        engines.append(engine)
    if not engines:
        raise ParalomeError(108, f"{source} names no annotation engine; use annovar or snpeff")
    return engines


def _load_pipeline_config(pipeline_config: Optional[Path]) -> dict[str, Any]:
    """The merged settings, with a --pipeline-config that is not JSON refused.

    A stray comma in the file was "PLM-0001 Unexpected internal
    error", and a file holding a JSON list was read as an empty one, so the
    run went ahead without any of its settings.
    """
    if pipeline_config is not None and Path(pipeline_config).is_file():
        fix = "Correct the JSON in that file; `python3 -m json.tool FILE` shows where it breaks"
        try:
            given = json.loads(Path(pipeline_config).read_text())
        except ValueError as exc:
            raise ParalomeError(
                412, f"--pipeline-config {pipeline_config} is not valid JSON: {exc}.", fix=fix,
            ) from exc
        if not isinstance(given, dict):
            raise ParalomeError(
                412,
                f"--pipeline-config {pipeline_config} holds JSON that is not an object "
                f"({type(given).__name__}); none of it would be read.",
                fix=fix,
            )
    return load_pipeline_config(pipeline_config)


def _configured_setting(
    section: str, key: str, pipeline_config: Optional[Path],
) -> tuple[Any, str]:
    """A setting someone wrote down, and the file it is in.

    --pipeline-config first, then the user configuration, as they are merged.
    Paralome's packaged defaults are not consulted: (None, "") when neither
    file sets it.
    """
    from paralome.config.user import config_path, load_config

    sources: list[tuple[Any, str]] = []
    if pipeline_config is not None:
        try:
            sources.append((json.loads(Path(pipeline_config).read_text()), str(pipeline_config)))
        except (OSError, ValueError):
            pass
    try:
        sources.append((load_config(initialize=False), str(config_path())))
    except FileNotFoundError:
        pass
    for data, where in sources:
        block = data.get(section) if isinstance(data, dict) else None
        if isinstance(block, dict) and block.get(key) not in (None, ""):
            return block[key], f"{section}.{key} in {where}"
    return None, ""


def _strandedness_source(given: Optional[str], pipeline_config: Optional[Path]) -> str:
    """Where this run's RNA strandedness came from.

    A configuration's rna.strandedness held a copied default of 0 for every
    account set up before detection existed, and nothing said so: every RNA
    run was counted unstranded. A value Paralome wrote and nobody changed is
    named as its default.
    """
    if given is not None:
        return "--rna-strandedness"
    _, where = _configured_setting("rna", "strandedness", pipeline_config)
    if not where:
        return "Paralome's default"
    if where.endswith(str(pipeline_config)) and pipeline_config is not None:
        return where
    from paralome.config.user import holds_shipped_default, load_config

    try:
        if holds_shipped_default(load_config(initialize=False), "rna.strandedness"):
            return f"Paralome's default, as written in {where.split(' in ', 1)[1]}"
    except FileNotFoundError:
        pass
    return where


def _backend_source(backend: Optional[str], pipeline_config: Optional[Path]) -> str:
    """Where this run's backend came from, in the order the settings are merged."""
    if backend:
        return "--backend"
    for section, source in (("resources", pipeline_config), ("defaults", None)):
        _, where = _configured_setting(section, "backend", source)
        if where:
            return where
    return "Paralome's default; no configuration sets one"


#: The genome's build names, which no configuration file may set: (key in the
#: `annotation` section, the option, the genome entry's key).
_BUILD_SETTINGS = (
    ("annovar_buildver", "--annovar-buildver"),
    ("snpeff_genome", "--snpeff-genome"),
)


def _refuse_a_configured_build(pipeline_config: Optional[Path]) -> None:
    """Refuse annotation.annovar_buildver / snpeff_genome in a configuration.

    With --genome required, the build comes from the genome entry and nothing
    reads these two keys; one a user wrote is refused rather than loaded and
    ignored.
    """
    for key, option in _BUILD_SETTINGS:
        value, where = _configured_setting("annotation", key, pipeline_config)
        if where:
            raise ParalomeError(
                411,
                f"{where} ({value!r}) is no longer read. The build comes from the "
                f"genome entry named by --genome (its key {key}), or from {option}.",
                fix=f"Remove that key; record {key} in the genome entry instead",
            )


def _refuse_an_operation_from_elsewhere(
    *,
    pipeline_config: Optional[Path],
    typed: dict[str, Any],
) -> None:
    """Refuse an ANNOVAR protocol list and operation list given in two places.

    THE OPERATION LIST COMES FROM WHERE THE PROTOCOL LIST DOES.
    ANNOVAR pairs them entry by entry. Each was looked up by itself, so three
    protocols from a configuration file ran with a genome entry's five
    operations: that passed the preflight and stopped after variant calling.
    A source with one list and not the other is refused, and named.
    """
    def given(name: str) -> bool:
        return typed.get(name) not in (None, "")

    def refuse(code: int, has: str, lacks: str, fix: str) -> None:
        raise ParalomeError(
            code,
            f"{has} is given without {lacks}. ANNOVAR pairs the protocol and operation "
            "lists entry by entry, so Paralome takes both from one place and never "
            "completes one with a list written for another.",
            fix=fix,
        )

    if given("annovar_protocol") != given("annovar_operation"):
        has, lacks = (
            ("--annovar-protocol", "--annovar-operation") if given("annovar_protocol")
            else ("--annovar-operation", "--annovar-protocol")
        )
        refuse(101, has, lacks, f"Add {lacks}, one entry for each entry of {has}; or drop {has}")
    if given("annovar_protocol"):
        return
    protocol_where = _configured_setting("annotation", "annovar_protocol", pipeline_config)[1]
    operation_where = _configured_setting("annotation", "annovar_operation", pipeline_config)[1]
    if not protocol_where and not operation_where:
        return
    protocol_file = protocol_where.partition(" in ")[2]
    operation_file = operation_where.partition(" in ")[2]
    if protocol_file != operation_file:
        has, lacks, where = (
            (protocol_where, "annotation.annovar_operation", protocol_file) if protocol_where
            else (operation_where, "annotation.annovar_protocol", operation_file)
        )
        elsewhere = operation_where if protocol_where else protocol_where
        refuse(
            117, has, f"{lacks} in that file"
            + (f" ({elsewhere} is a different file)" if elsewhere else ""),
            f"Write {lacks} in {where} as well; or remove the key that is there, and the "
            "lists come from the genome entry",
        )


def _annotation_settings(
    resolved: dict[str, Any],
    engines: list[str],
    *,
    pipeline_config: Optional[Path],
    typed: dict[str, Any],
) -> None:
    """Fill in the build, protocols and operations a DNA run annotates with,
    and print each with where it came from.

    The order is the option, then annotation.annovar_protocol /
    annovar_operation in a configuration file, then the genome entry. Making
    --genome mandatory had quietly moved a run from the configured protocol
    list to the genome entry's. There is no built-in value: one that nothing
    supplies is left unset, and the preflight refuses the run for it.
    """
    labels = {
        "annovar_buildver": ("annovar", "ANNOVAR build", "--annovar-buildver"),
        "annovar_protocol": ("annovar", "ANNOVAR protocol", "--annovar-protocol"),
        "annovar_operation": ("annovar", "ANNOVAR operation", "--annovar-operation"),
        "snpeff_genome": ("snpeff", "snpEff genome", "--snpeff-genome"),
    }
    if "annovar" in engines:
        _refuse_an_operation_from_elsewhere(pipeline_config=pipeline_config, typed=typed)
    entry = None
    for name, (engine, label, option) in labels.items():
        if engine not in engines:
            continue
        source = ""
        if typed.get(name) not in (None, ""):
            source = option
        elif resolved.get(name) not in (None, ""):
            source = _configured_setting("annotation", name, pipeline_config)[1]
        elif resolved.get("genome"):
            if entry is None:
                entry = resolve_genome_resources(genome=resolved["genome"])
            recorded = entry.resources.get(name)
            if recorded is not None:
                resolved[name] = recorded.value
                source = f"genome entry {entry.genome}, key {recorded.config_key}"
            elif name == "annovar_operation" and "annovar_protocol" in entry.resources:
                # The same rule for the genome entry, said here
                # because the preflight would offer the option or the
                # configuration as the way to complete it.
                raise ParalomeError(
                    117,
                    f"genome entry {entry.genome} records "
                    f"{entry.resources['annovar_protocol'].config_key} and no annovar_operation. "
                    "ANNOVAR pairs the two lists entry by entry, so Paralome takes both "
                    "from one place.",
                    fix="Record annovar_operation in that genome entry (`paralome setup config` "
                        "shows the file); or pass --annovar-protocol and --annovar-operation together",
                )
        if resolved.get(name) not in (None, ""):
            print(f"{label:<20}: {resolved[name]} ({source})")


def _configured_genomes_sentence() -> str:
    """Which genomes --genome could name, for the refusal that asks for one."""
    from paralome.pipeline.resource_catalog import retired_default_genome_note

    try:
        from paralome.config.user import load_config

        config = load_config(initialize=False)
    except FileNotFoundError:
        config = {}
    genomes = config.get("genomes")
    names = sorted(str(name) for name in genomes) if isinstance(genomes, dict) else []
    if not names:
        return "No genome is configured yet; register one with `paralome setup genome add <folder>`"
    return "Configured genomes: " + ", ".join(names) + retired_default_genome_note(config)


def what_explicit_paths_lack(
    *,
    annotation_engines: list[str],
    star_genome: Any, reference: Any,
    annovar_db: Any, annovar_buildver: Any,
    annovar_protocol: Any, annovar_operation: Any,
    snpeff_jar: Any, snpeff_config: Any, snpeff_genome: Any,
    aligner: str = "star",
) -> list[str]:
    """The options a DNA run without --genome still has to be given.

    The protocol and operation lists are asked for too. The
    preflight requires them, so a run that named neither passed this rule,
    and a batch submitted every family to be stopped by PLM-0117.
    ``annovar_protocol`` and ``annovar_operation`` are the option or, failing
    that, the configured value: both stand without a genome.
    """
    # --aligner bwa aligns against the FASTA (--reference) and needs no STAR index.
    needed = ([] if str(aligner or "").strip().lower() == "bwa" else [("--star-genome", star_genome is not None)])
    needed += [("--reference", reference is not None)]
    if "annovar" in annotation_engines:
        needed += [
            ("--annovar-db", annovar_db is not None),
            ("--annovar-buildver", bool(annovar_buildver)),
            ("--annovar-protocol", bool(annovar_protocol)),
            ("--annovar-operation", bool(annovar_operation)),
        ]
    if "snpeff" in annotation_engines:
        needed += [
            ("--snpeff-jar", snpeff_jar is not None),
            ("--snpeff-config", snpeff_config is not None),
            ("--snpeff-genome", bool(snpeff_genome)),
        ]
    return [option for option, given in needed if not given]


def explicit_paths_cover_the_genome(*, is_variant: bool, **given: Any) -> bool:
    """Whether a DNA run's own options stand in for --genome.

    One rule, for the analysis and for `batch run`, which checks each family
    command with it when the plan is built. The build is named too: without
    it the genome's build would have to be assumed.
    """
    return bool(is_variant and not what_explicit_paths_lack(**given))


def _require_a_genome(
    *,
    analysis: str,
    genome: Optional[str],
    explicit_paths_cover_it: bool,
    lacking: tuple[str, ...] = (),
) -> None:
    """Refuse a run that names no genome, or one that is not configured.

    A needed --genome has no default: a default would be wrong for most users
    and say nothing. A DNA run given every genome path as an option needs no
    --genome.
    """
    if genome:
        from paralome.pipeline.resource_catalog import (
            ResourceResolutionError, select_genome_config,
        )

        from paralome.setup.genomes import registry_permission_text

        try:
            select_genome_config(genome)
        except ResourceResolutionError as exc:
            # A registry this account cannot read is said, not
            # reported as a genome that does not exist.
            denied = registry_permission_text()
            raise ParalomeError(401, f"{exc} {denied}" if denied else str(exc),
                                **({"fix": "Ask the administrator to set the permissions named above"} if denied else {})) from exc
        except FileNotFoundError as exc:
            # An account that has never run setup: no file to look the name up
            # in. It was an unnumbered internal error.
            denied = registry_permission_text()
            raise ParalomeError(
                401, f"--genome {genome}: {exc}" + (f" {denied}" if denied else ""),
                fix=("Ask the administrator to set the permissions named above, then run `paralome setup genome adopt "
                     f"{genome}`") if denied else
                    "Run `paralome setup`, then `paralome setup genome list` to see what this machine has",
            ) from exc
        return
    if explicit_paths_cover_it:
        return
    listed = _configured_genomes_sentence()
    instead = (
        " (or give every path yourself: --star-genome, --reference, and "
        "--annovar-db with --annovar-buildver, --annovar-protocol and "
        "--annovar-operation, or --snpeff-jar with "
        "--snpeff-config and --snpeff-genome)"
        if analysis == "DNA" else ""
    )
    # A run that set out to give every path is told which are left.
    short = (
        f" This command gives genome paths but not {', '.join(lacking)}."
        + (" The two ANNOVAR lists may instead be annotation.annovar_protocol and "
           "annotation.annovar_operation in your configuration."
           if {"--annovar-protocol", "--annovar-operation"} & set(lacking) else "")
        if lacking else ""
    )
    raise ParalomeError(
        101,
        f"--genome is required for {analysis}{instead}. Paralome never chooses "
        f"a genome for you.{short} {listed}",
    )


def _variants_setting(cfg: Any, key: str, given: Any, option: str, kind: type) -> Any:
    """--max-region-length / --stall-minutes, or variants.<key> in the
    configuration. None when neither is given (the stage's default then
    holds). A configured value that is not a number of 0 or more is refused,
    named, as the option is."""
    if given is not None:
        return kind(given)
    value = cfg_get(cfg, "variants", key, None)
    if value is None:
        return None
    try:
        number = kind(value)
        if isinstance(value, bool) or number < 0 or (kind is int and float(value) != number):
            raise ValueError
    except (TypeError, ValueError):
        raise ParalomeError(
            112,
            f"variants.{key} in the configuration is {value!r}: it takes a number of 0 or more, as {option} does.",
            fix=f"Set variants.{key} to a number (0 or more), or remove it to use the default",
        ) from None
    return number


def _resolve_pipeline_inputs(
    *,
    project: Optional[Path],
    pipeline_config: Optional[Path],
    samples: Optional[Path] = None,
    rds: Optional[Path] = None,
    replicate_column: Optional[str] = None,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    stratify_skip: Optional[str] = None,
    min_cells_per_stratum: int = 0,
    rscript: Optional[str] = None,
    data_workdir_path: Optional[Path] = None,
    rna_assay: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
    singlecell_rna_workdir: Optional[Path] = None,
    singlecell_rna_differential_workdir: Optional[Path] = None,
    threads: Optional[int],
    jobs: Optional[int],
    backend: Optional[str],
    fastq_qc_workdir: Optional[Path],
    align_workdir: Optional[Path],
    dedup_workdir: Optional[Path],
    variant_workdir: Optional[Path],
    annotation_workdir: Optional[Path],
    segregation_workdir: Optional[Path],
    filter_workdir: Optional[Path],
    report_workdir: Optional[Path],
    fastp_threads: Optional[int],
    star_threads: Optional[int],
    dedup_threads: Optional[int],
    freebayes_threads: Optional[int],
    fastp_adapter_fasta: Optional[Path],
    fastp_cut_right: bool,
    fastp_cut_right_window_size: Optional[int],
    fastp_cut_right_mean_quality: Optional[int],
    fastp_length_required: Optional[int],
    fastp_extra: Optional[str],
    star_extra: Optional[str],
    aligner: str = "star",
    bwa_extra: Optional[str] = None,
    star_read_files_command: Optional[str],
    star_out_reads_unmapped: Optional[str],
    star_out_filter_multimap_nmax: Optional[int],
    star_out_filter_mismatch_nover_read_lmax: Optional[float],
    # Default to None/False so callers that predate the loose
    # switch keep working and resolve to the profile defaults. Every
    # command path passes them explicitly.
    star_out_filter_mismatch_nmax: Optional[int] = None,
    star_out_filter_score_min_over_lread: Optional[float] = None,
    star_out_filter_match_nmin_over_lread: Optional[float] = None,
    star_loose: bool = False,
    annotation: Optional[str],
    profile: Optional[str],
    segregation_profile: Optional[str] = None,
    genome: Optional[str],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
    snpeff_genome: Optional[str],
    mark_duplicates: Optional[bool],
    remove_duplicates: Optional[bool],
    fuse_stages: Optional[bool] = None,
    fused_buffer: Optional[str] = None,
    rna_strandedness: Optional[str] = None,
    rna_min_expression: Optional[str] = None,
    rna_min_expression_difference: Optional[str] = None,
    stage_mem: List[str] = [],
    max_region_length: Optional[int] = None,
    stall_minutes: Optional[float] = None,
) -> dict[str, Any]:
    cfg = _load_pipeline_config(pipeline_config)
    naming_source = samples or rds or Path("unnamed")
    workdirs = derive_project_workdirs(project, naming_source) if (project is not None or samples is not None) else None
    cli_stage_mem = _parse_stage_mem(stage_mem)
    # Every stage, not a hand-kept list, so every stage's configured memory is read.
    known_stages = _stage_mem_names()
    # variant_calling and variants are one stage under two names. Both are
    # filed under `variants`, so a --stage-mem given under either name replaces
    # a configured value given under either name.
    def _stage_mem_key(stage: str) -> str:
        return "variants" if stage == "variant_calling" else stage

    stage_memories: dict[str, str] = {}
    for stage in known_stages:
        configured = stage_resource(cfg, stage, "mem", None)
        if configured is not None:
            stage_memories[_stage_mem_key(stage)] = _validate_mem(str(configured), source=f"stage_resources.{stage}.mem")
    stage_memories.update({_stage_mem_key(stage): value for stage, value in cli_stage_mem.items()})
    # A STRING, and it defaults to "auto".
    #
    # This was Optional[int] bounded 0-2, so an RNA run could not express
    # "auto" at all -- the detection built and improved earlier
    # was reachable only by invoking `rna quantify-fpkm` by hand. Every
    # pipeline run silently assumed unstranded, although strandedness is
    # usually not supplied. Paralome's answer to that was to guess, and not to
    # say so.
    #
    # Unstranded is the right FALLBACK but the wrong DEFAULT. Counting
    # unstranded keeps every read and adds antisense noise; guessing a
    # direction wrong throws real signal away. Asymmetric costs, so detection
    # runs first and the recoverable failure is where it lands if it cannot
    # decide.
    configured_strandedness = cfg_get(cfg, "rna", "strandedness", "auto")
    resolved_rna_strandedness = str(
        configured_strandedness if rna_strandedness is None else rna_strandedness
    ).strip().lower()
    if resolved_rna_strandedness not in {"0", "1", "2", "auto"}:
        # PLM-0114, which was in the registry and raised nowhere, so
        # `paralome explain 114` described a refusal that came unnumbered.
        raise ParalomeError(
            114,
            ("--rna-strandedness" if rna_strandedness is not None
             else "rna.strandedness in the configuration")
            + f" {resolved_rna_strandedness!r}: use 0 (unstranded), 1 (first-strand), "
            "2 (second-strand), or auto to detect it from the data",
        )
    configured_min_expression = cfg_get(cfg, "rna", "min_expression", 5.0)
    configured_min_expression_difference = cfg_get(cfg, "rna", "min_expression_difference", 5.0)
    # A string, parsed by _parse_threshold_list; one value.
    resolved_rna_min_expression = str(
        configured_min_expression if rna_min_expression is None else rna_min_expression
    )
    resolved_rna_min_expression_difference = str(
        configured_min_expression_difference
        if rna_min_expression_difference is None
        else rna_min_expression_difference
    )
    global_threads = int(threads if threads is not None else cfg_get(cfg, "resources", "threads", 8))
    configured_pipeline_profile = cfg_get(cfg, "pipeline", "profile", None)
    legacy_filter_profile = cfg_get(cfg, "filtering", "profile", None)
    configured_segregation_profile = cfg_get(cfg, "filtering", "segregation_profile", legacy_filter_profile)
    requested_profile = profile or configured_pipeline_profile or legacy_filter_profile
    preliminary_profile = resolve_pipeline_profile(requested_profile)
    selected_segregation_profile = segregation_profile
    if preliminary_profile.is_variant and selected_segregation_profile is None:
        selected_segregation_profile = configured_segregation_profile
    resolved_profile = resolve_pipeline_profile(
        requested_profile,
        selected_segregation_profile,
    )
    if resolved_profile.is_variant:
        _refuse_a_configured_build(pipeline_config)
    if resolved_profile.is_variant and samples is not None:
        # The profile checked against the sheet's scoring layout
        # before any job runs (filters.table.applicable_pipeline_profile).
        from paralome.filters.table import applicable_pipeline_profile

        chosen_by = "--segregation-profile" if segregation_profile else None
        for section, key in (("filtering", "segregation_profile"), ("filtering", "profile"),
                             ("defaults", "filter_profile")):
            if chosen_by:
                break
            chosen_by = _configured_setting(section, key, pipeline_config)[1] or None
        try:
            fitted = applicable_pipeline_profile(samples, resolved_profile.segregation_profile, chosen_by=chosen_by)
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
        if fitted != resolved_profile.segregation_profile:
            resolved_profile = resolve_pipeline_profile(requested_profile, fitted)
    # PLM-0111, which has been in the registry unraised. A stage name
    # this analysis does not have was accepted and then never looked up, so the
    # run went ahead on the default memory and nothing said so.
    stage_mem_names = [
        *resolved_profile.stage_graph,
        *(["variant_calling"] if "variants" in resolved_profile.stage_graph else []),
    ]
    unknown_stage_mem = sorted(set(cli_stage_mem) - set(stage_mem_names))
    if unknown_stage_mem:
        raise ParalomeError(
            111,
            f"--stage-mem {unknown_stage_mem[0]}=... : the {resolved_profile.pipeline_profile.value} "
            f"analysis has no stage called {unknown_stage_mem[0]!r}. Its stages: "
            + ", ".join(stage_mem_names),
            unknown=unknown_stage_mem,
            stages=stage_mem_names,
        )
    # Duplicate marking cannot be switched off. --no-mark-duplicates
    # (or mark_duplicates: false in a config) was accepted, read QC and
    # alignment ran, and the deduplication stage then refused it. Refused
    # here instead, naming where it was set.
    resolved_mark_duplicates = bool(
        mark_duplicates if mark_duplicates is not None
        else cfg_get(cfg, "resources", "mark_duplicates", True)
    )
    if not resolved_mark_duplicates and "deduplication" in resolved_profile.stage_graph:
        where = (
            "mark_duplicates is false in the configuration (defaults.mark_duplicates "
            "or resources.mark_duplicates)"
            if mark_duplicates is None
            else "--no-mark-duplicates was given"
        )
        raise ParalomeError(
            108,
            f"{where}, but the {resolved_profile.pipeline_profile.value} analysis always "
            "marks duplicates before it goes on.",
            fix="Remove that setting; use --remove-duplicates to drop duplicates rather than mark them",
        )
    # The data folder is written in the same default place every time, so the
    # user need not specify it. This mirrors the same walk-up convention naming.py's
    # derive_file_prefix() already uses for output filenames: a bare
    # `--project name` (no parent directory component) keeps pseudobulk
    # nested inside it (name/data); a `--project`
    # with a parent directory (e.g. snATAC/tissue_vs_NEG) is read as "this
    # comparison lives under a shared study directory", so pseudobulk
    # defaults to living in that parent (snATAC/data) instead of
    # being duplicated under every comparison's own subfolder. Every
    # comparison under the same parent -- same --rds/--replicate-column/
    # --stratify-by/--genome -- then sees that stage
    # already complete via its checkpoint and skips straight to
    # singlecell_differential, with zero extra flags. --data-workdir
    # still overrides this when the default guess is wrong (e.g. a
    # genuinely standalone one-off analysis that just happens to sit in a
    # subdirectory of something else).
    default_pseudobulk_workdir = None
    if workdirs is not None:
        # Or the parent project's, when a data folder
        # above --project already holds this object's extraction.
        default_pseudobulk_workdir = (
            data_workdir(workdirs.project, rds=rds)
        )
        # ONE DATA FOLDER PER PROJECT. The rule above reads the
        # --project as typed, so `--project foo` and `--project /abs/foo` name
        # one project and two data folders, and the second spelling quietly
        # extracted the object again. If the folder this spelling points at is
        # not there and the other spelling's is, stop and ask which.
        if data_workdir_path is None and resolved_profile.pipeline_profile.value in ("snATAC", "scRNA"):
            absolute_project = Path(workdirs.project).expanduser().resolve()
            chosen = default_pseudobulk_workdir.expanduser().resolve()
            others = [
                candidate
                for candidate in (
                    absolute_project / chosen.name,
                    absolute_project.parent / chosen.name,
                )
                if candidate != chosen and layout_extraction_dir(candidate).is_dir()
            ]
            if others and not layout_extraction_dir(chosen).is_dir():
                raise ParalomeError(
                    101,
                    f"--data-workdir: --project {str(project)!r} puts this project's data "
                    f"folder at {chosen}, which holds no extraction, while {others[0]} does "
                    "-- the folder the other spelling of this --project (relative or "
                    "absolute) uses.",
                    fix=f"Pass --data-workdir {others[0]} to use the existing extraction, "
                        f"or --data-workdir {chosen} to make a new one",
                )
        # ScRNA WITH --data-workdir, FOR A PROJECT AN OLDER PARALOME MADE, whose RNA pseudobulk is in
        # the default folder and not in the named one: the command would find none in the named folder
        # and aggregate again under results that are already finished, so only that case is refused.
        # What else either folder holds (a combined object's ATAC half, another project's extraction)
        # is not evidence about this project.
        if (
            data_workdir_path is not None
            and singlecell_rna_workdir is None
            and resolved_profile.pipeline_profile.value == "scRNA"
        ):
            from paralome.checkpoints import stage_checkpoint_path
            from paralome.pipeline.state import summarize_singlecell_rna_differential

            def has_rna_pseudobulk(folder: Path) -> bool:
                return stage_checkpoint_path(folder, "singlecell_rna").is_file()

            given = Path(data_workdir_path).expanduser().resolve()
            absolute_project = Path(workdirs.project).expanduser().resolve()
            default = default_pseudobulk_workdir.expanduser().resolve()
            finished = summarize_singlecell_rna_differential(
                choose_path(singlecell_rna_differential_workdir, absolute_project / "rna")
            )["complete"]
            holding = [
                candidate
                for candidate in dict.fromkeys((
                    default,
                    absolute_project / default.name,
                    absolute_project.parent / default.name,
                ))
                if candidate != given and has_rna_pseudobulk(candidate)
            ] if finished and not has_rna_pseudobulk(given) else []
            if holding:
                raise ParalomeError(
                    101,
                    f"--data-workdir {str(data_workdir_path)!r} ({given}) holds no RNA "
                    f"pseudobulk for project {absolute_project}, whose single-cell RNA "
                    f"results are finished; the default data folder {holding[0]} holds "
                    "one. An older Paralome's scRNA ignored --data-workdir and wrote the "
                    "pseudobulk into the default folder, so that is most likely this "
                    "project's.",
                    fix=f"Pass --data-workdir {holding[0]} to go on from it",
                )
    # The directory layout keeps the project root clean of per-modality clutter -- ATAC's own
    # single-cell differential/motif workdirs nest under <project>/atac/, RNA's under <project>/rna/,
    # leaving only pseudobulk/, atac/, rna/ at the top level, with no singlecell_differential/ level.
    #
    # Safe only because every result filename carries the project, the stratum and the modality --
    # see naming.result_path.
    default_singlecell_differential_workdir = (workdirs.project / "atac") if workdirs else None
    default_singlecell_motif_workdir = (workdirs.project / "atac" / "motif_analysis") if workdirs else None
    default_singlecell_rna_differential_workdir = (workdirs.project / "rna") if workdirs else None
    return {
        "project_root": workdirs.project if workdirs else None,
        "fastq_qc_workdir": choose_path(fastq_qc_workdir, workdirs.fastq_qc if workdirs else None),
        "align_workdir": choose_path(align_workdir, workdirs.alignment if workdirs else None),
        "dedup_workdir": choose_path(dedup_workdir, workdirs.deduplication if workdirs else None),
        "variant_workdir": choose_path(variant_workdir, workdirs.variants if workdirs else None),
        "annotation_workdir": choose_path(annotation_workdir, workdirs.annotation if workdirs else None),
        "segregation_workdir": choose_path(segregation_workdir, workdirs.segregation if workdirs else None),
        "filter_workdir": choose_path(filter_workdir, workdirs.filtering if workdirs else None),
        "report_workdir": choose_path(report_workdir, workdirs.reporting if workdirs else None),
        "threads": global_threads,
        # --backend has no default of its own, so defaults.backend in the user config is reached here.
        # SLURM and Local are read as slurm and local.
        "backend": str(backend or cfg_get(cfg, "resources", "backend", "slurm")).strip().lower(),
        "jobs": int(jobs if jobs is not None else cfg_get(cfg, "resources", "jobs", 0)),
        "stage_memories": stage_memories,
        "rna_strandedness": resolved_rna_strandedness,
        "rna_min_expression": resolved_rna_min_expression,
        "rna_min_expression_difference": resolved_rna_min_expression_difference,
        "fastp_threads": int(fastp_threads if fastp_threads is not None else global_threads),
        "star_threads": int(star_threads if star_threads is not None else global_threads),
        "dedup_threads": int(dedup_threads if dedup_threads is not None else global_threads),
        "freebayes_threads": int(freebayes_threads if freebayes_threads is not None else global_threads),
        # The option, else variants.* in the configuration, else the
        # stage's own default (None: not passed).
        "freebayes_max_region_length": _variants_setting(
            cfg, "max_region_length", max_region_length, "--max-region-length", int),
        "freebayes_stall_minutes": _variants_setting(
            cfg, "stall_minutes", stall_minutes, "--stall-minutes", float),
        # CLI, then this project's config, then the machine's site config. An adapter FASTA is a fact
        # about the sequencing kit and the machine, so `setup site --adapter-fasta` records it once and
        # every project inherits it.
        "fastp_adapter_fasta": _resolved_adapter_fasta(cfg, fastp_adapter_fasta),
        "fastp_cut_right": fastp_cut_right,
        "fastp_cut_right_window_size": fastp_cut_right_window_size,
        "fastp_cut_right_mean_quality": fastp_cut_right_mean_quality,
        "fastp_length_required": fastp_length_required,
        "fastp_extra": fastp_extra,
        "star_extra": star_extra,
        "aligner": normalize_aligner(aligner),
        "bwa_extra": bwa_extra,
        "star_read_files_command": star_read_files_command or "zcat",
        "star_out_reads_unmapped": star_out_reads_unmapped or "None",
        "star_out_filter_multimap_nmax": int(star_out_filter_multimap_nmax if star_out_filter_multimap_nmax is not None else 2),
        # All four STAR output filters resolved together, because
        # they interact. --outFilterMismatchNmax is an ABSOLUTE cap applied
        # alongside the fraction and STAR takes whichever is stricter, so
        # resolving the fraction alone -- which is what this did -- produced a
        # setting that could be raised without effect: raising 0.08 to 0.16
        # moved nothing while the cap of 10 stayed.
        **{f"star_{k}": v for k, v in star_output_filters(
            resolved_profile.alignment_profile,
            loose=bool(star_loose),
            mismatch_fraction=star_out_filter_mismatch_nover_read_lmax,
            mismatch_nmax=star_out_filter_mismatch_nmax,
            score_min_over_lread=star_out_filter_score_min_over_lread,
            match_nmin_over_lread=star_out_filter_match_nmin_over_lread,
        ).items() if k != "loose"},
        "star_loose": bool(star_loose),
        "annotation": annotation or str(cfg_get(cfg, "annotation", "engine", "annovar")),
        "profile": resolved_profile.pipeline_profile.value,
        "segregation_profile": resolved_profile.segregation_profile,
        "alignment_profile": resolved_profile.alignment_profile.value,
        "genome": genome,
        # The build is the option or, later, the genome entry;
        # never hg38 or GRCh38.99 of Paralome's own. The protocol list and
        # operations are the option, then a configuration file whether or not
        # --genome is given, then the genome entry (see _annotation_settings).
        "annovar_buildver": annovar_buildver,
        "annovar_protocol": annovar_protocol if annovar_protocol is not None else cfg_get(cfg, "annotation", "annovar_protocol", None),
        "annovar_operation": annovar_operation if annovar_operation is not None else cfg_get(cfg, "annotation", "annovar_operation", None),
        "snpeff_genome": snpeff_genome,
        "mark_duplicates": resolved_mark_duplicates,
        "remove_duplicates": bool(remove_duplicates if remove_duplicates is not None else cfg_get(cfg, "resources", "remove_duplicates", False)),
        # One task per sample for FASTQ QC, alignment and duplicate
        # marking, so the trimmed FASTQs and the aligned BAM are never written.
        # Configurable per site, because whether those intermediates are kept
        # is a policy and not a tuning.
        # FUSED IS THE DEFAULT. The intermediates a staged run
        # writes -- the trimmed FASTQs and the aligned BAM -- are deleted by
        # hand afterwards anyway, and on the 40-genome run they bloated 2.2 TB
        # of input several fold and filled the shared storage. Re-running a failed
        # sample costs less than carrying them. --separate-stages is the way
        # back when QC or mapping needs looking at.
        "fuse_stages": bool(fuse_stages if fuse_stages is not None else cfg_get(cfg, "resources", "fuse_stages", DEFAULT_FUSE_STAGES)),
        # What a fused task's two read streams may buffer. See
        # paralome.buffer -- a kernel pipe is 64 KB and the streams drift.
        "fused_buffer": str(fused_buffer if fused_buffer else cfg_get(cfg, "resources", "fused_buffer", "8G")),
        "rds": rds,
        "replicate_column": replicate_column,
        "stratify_by": stratify_by,
        "stratify_value": stratify_value,
        "stratify_skip": stratify_skip,
        "min_cells_per_stratum": min_cells_per_stratum,
        "rscript": rscript,
        "data_workdir_path": choose_path(data_workdir_path, default_pseudobulk_workdir),
        "singlecell_differential_workdir": default_singlecell_differential_workdir,
        "singlecell_motif_workdir": default_singlecell_motif_workdir,
        "rna_assay": rna_assay,
        "gene_symbol_column": gene_symbol_column,
        # --data-workdir is honoured by scRNA as it is by snATAC.
        # This fell back to the default alone and the stage prefers this
        # value, so scRNA extracted into the default folder whatever
        # --data-workdir said.
        "singlecell_rna_workdir": choose_path(
            singlecell_rna_workdir,
            choose_path(data_workdir_path, default_pseudobulk_workdir),
        ),
        "singlecell_rna_differential_workdir": choose_path(singlecell_rna_differential_workdir, default_singlecell_rna_differential_workdir),
    }


def _resolve_pipeline_genome_paths(
    *,
    genome: Optional[str],
    reference: Optional[Path],
    star_genome: Optional[Path],
    gtf: Optional[Path],
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Resolve stage-wide genome paths using the authoritative genome catalog.

    Explicit CLI paths retain precedence. When ``--genome`` is supplied, the
    configured reference FASTA, STAR index, and GTF become the defaults used by
    alignment and variant calling as well as annotation.
    """
    if not genome:
        return reference, star_genome, gtf

    resources = resolve_genome_resources(
        genome=genome,
        overrides={
            "reference_fasta": reference,
            "star_index": star_genome,
            "gtf": gtf,
        },
    )
    return (
        resources.path("reference_fasta"),
        resources.path("star_index"),
        resources.path("gtf"),
    )


def _state_target(resolved: dict[str, Any]) -> Path:
    return Path(
        resolved.get("report_workdir")
        or resolved.get("filter_workdir")
        or resolved.get("segregation_workdir")
        or resolved.get("annotation_workdir")
        or resolved.get("variant_workdir")
        or resolved.get("dedup_workdir")
        or resolved.get("align_workdir")
        or resolved.get("fastq_qc_workdir")
        or resolved.get("project_root")
        or Path(".")
    )


def _print_pipeline_run_result(title: str, result, *, recorded: bool = True) -> None:
    print(title)
    print("=" * len(title))
    print(f"Next stage          : {result.next_stage}")
    print(f"Workdir             : {result.workdir}")
    print(f"Executed            : {result.executed}")
    print(f"Message             : {result.message}")
    if result.command:
        import shlex
        print("Command             : " + " ".join(shlex.quote(x) for x in result.command))
        if recorded:
            print(f"Plan JSON           : {result.workdir / 'inputs' / 'pipeline_run_plan.json'}")
        else:
            # A dry run records nothing; say so rather than name a
            # file that is not there.
            print("Plan JSON           : not written (dry run)")
    else:
        print("Command             : .")


#: The panel the tie-breakers live in. Named for when you would reach for one
#: rather than for what they are, because nobody goes looking for "advanced
#: options" -- they go looking because the thing guessed wrong.
_WHEN_AMBIGUOUS = "When auto-detection needs help"


#: Memory for each extraction job, as the single-cell stages ask by default.
EXTRACT_JOB_MEM = "32G"


def _submit_extraction(
    rds: Path, data_dir: Path, *, genome: str, gtf, assay, slot, rna_assay, rna_slot, gene_symbol_column, rscript,
    overwrite: bool, jobs: int, partition: Optional[str], submit: bool,
) -> None:
    """`extract --backend slurm`: a folder one job per object (``jobs`` at a time), then the extract itself.

    The last job is this same command run locally on a node: every
    object's extraction is then found done and reused, so it only checks,
    combines and writes the QC tables. It waits for the array with
    ``afterany``, so an object whose job failed is extracted again there and
    its error is reported where the run's is. With --overwrite every object is
    extracted again in the array, and the last job does not repeat it.
    """
    from paralome.binaries import human_argv
    from paralome.commands.atac import _print_submission, _relay_opt, _submit_single_job
    from paralome.singlecell.rds_input import rds_argument, submit_member_extraction

    after = None
    if Path(rds).is_dir():
        after = submit_member_extraction(
            rds, layout_extraction_dir(data_dir), workdir=data_dir, jobs=jobs, mem=EXTRACT_JOB_MEM,
            partition=partition, submit=submit, overwrite=overwrite, gtf=gtf, assay=assay, slot=slot,
            rna_assay=rna_assay, rna_slot=rna_slot, gene_symbol_column=gene_symbol_column, rscript=rscript,
        )
        print(f"{len(after['to_extract'])} of {after['members']} objects to extract"
              + (f", {jobs} at a time" if jobs else ", all at once")
              + (f"; script {after['script']}" if after.get("script") else ""))
    argv = human_argv() + ["extract", "--rds", rds_argument(rds), "--out", str(data_dir),
                           "--slot", slot, "--rscript", rscript, "--backend", "local"]
    argv += ["--genome", genome]
    _relay_opt(argv, "--assay", assay)
    _relay_opt(argv, "--rna-assay", rna_assay)
    _relay_opt(argv, "--rna-slot", rna_slot)
    _relay_opt(argv, "--gene-symbol-column", gene_symbol_column)
    if overwrite and after is None:
        argv.append("--overwrite")
    result = _submit_single_job(
        job_name="sc_extract_combine" if after is not None else "sc_extract", workdir=data_dir, argv=argv,
        threads=1, mem=EXTRACT_JOB_MEM, partition=partition, submit=submit, after=after,
    )
    _print_submission(result)


def pipeline_extract_command(
    rds: str = typer.Option(..., "--rds", help="Seurat/Signac .rds object (the one combined, project-wide object); or a folder of objects, written with a trailing /"),
    project: Optional[Path] = typer.Option(
        None, "--project",
        help="The parent project folder. The extraction goes in <project>/data, and "
             "every run whose --project is inside this folder, at any depth, uses it.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="A data folder, used as given (extraction/ and qc/ are created under it). "
             "Use --project instead to share the extraction with a run.",
    ),
    # THESE ARE TIE-BREAKERS. The whole RDS is converted: every assay present is extracted and nothing
    # has to be named. They exist because ambiguity is real -- an object carrying RNA plus SCT plus
    # integrated gives auto-detection nothing to choose on. They sit in their own help panel.
    assay: Optional[str] = typer.Option(
        None, "--assay", rich_help_panel=_WHEN_AMBIGUOUS,
        help="Name the ATAC assay, when more than one could be it.",
    ),
    slot: str = typer.Option(
        "counts", "--slot", rich_help_panel=_WHEN_AMBIGUOUS,
        help="Layer to read. counts is the only one that is not derived.",
    ),
    rna_assay: Optional[str] = typer.Option(
        None, "--rna-assay", rich_help_panel=_WHEN_AMBIGUOUS,
        help="Name the RNA assay, when more than one could be it. RNA is "
             "exported whenever it is present; this only breaks the tie.",
    ),
    rna_slot: Optional[str] = typer.Option(
        None, "--rna-slot", rich_help_panel=_WHEN_AMBIGUOUS,
        help="RNA layer, if it differs from --slot.",
    ),
    gene_symbol_column: Optional[str] = typer.Option(
        None, "--gene-symbol-column", rich_help_panel=_WHEN_AMBIGUOUS,
        help="Feature metadata column holding gene symbols, when it is not "
             "where it is usually found.",
    ),
    genome: str = typer.Option(
        ..., "--genome",
        help="Configured genome bundle. REQUIRED: it resolves the GTF used to "
             "attach an Ensembl ID to every gene, once, here -- every later stage "
             "inherits it. The match rate is reported and doubles as a check that "
             "this really is the right genome for this object.",
    ),
    rscript: str = typer.Option(
        "Rscript", "--rscript", rich_help_panel=_WHEN_AMBIGUOUS,
        help="Rscript executable, if it is not the one on PATH.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Force re-extraction even when a valid one exists"),
    backend: str = typer.Option("local", "--backend", help="local: extract here. slurm: as SLURM jobs -- a folder of objects one job per object, --jobs at a time, then one job that combines them"),
    submit: bool = typer.Option(False, "--submit", help="With --backend slurm: submit the jobs (otherwise the scripts are written and nothing runs)"),
    jobs: int = typer.Option(0, "--jobs", min=0, help="With --backend slurm and a folder: objects extracted at once (0: all)"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
):
    """Extract a Seurat/Signac object once and write the project's QC tables.

    The first step with a new object: `paralome extract --rds obj.rds --project
    <project> --genome <genome>` writes QC tables from which to choose the
    grouping variables, before running the 3D PCA plots.

    --project is the parent project folder. The extraction goes in
    <project>/data, and every run whose --project is inside that folder, at
    any depth, finds it there instead of reading the object again.

    This takes NO grouping flags. Choosing --replicate-column/--factor/
    --stratify-by is what the QC tables it writes are for. It is also
    modality-neutral: whichever halves the object has come out, both in one
    readRDS, sharing one cell metadata table. Nothing has to be named.

    Writes into <project>/data (or --out):
      extraction/              the extracted matrices and cell metadata
      qc/metadata_columns.csv  one row per metadata column: dtype, distinct
      count, whether it is usable for grouping, and min/median/max for numeric
      columns

      qc/metadata_values.csv   one row per column+value with cell counts --
      the exact spellings a --factor control level needs

      qc/metadata_summary.csv  the two above denormalized into one file

    Re-running against an unchanged object is cheap: the extraction is reused
    via its checkpoint, and the .rds is not re-hashed in full to discover that.
    """
    # --genome is required rather than encouraged: without it every gene_id is '.' and no
    # annotation cross-check is possible, and a user only discovers that after
    # paying for a full extraction. Failing here costs seconds.
    # --gtf is gone; --genome replaced it. It was an override on a value --genome already resolves, and
    # keeping it meant a registered genome and a hand-typed GTF could disagree
    # with nothing to say which one the run used.
    if not genome:
        raise typer.BadParameter(
            "--genome is required. The extraction resolves an Ensembl ID for every "
            "gene from the GTF and stores it once, so every downstream comparison "
            "inherits it. Without a genome there is nothing to resolve against and "
            "no way to check the annotation matches the object."
        )
    # The name has to be a configured genome, and that genome has to
    # record a GTF. Without the second check a genome with none extracted
    # every gene with no Ensembl ID, which is what requiring --genome is for.
    _require_a_genome(analysis="extract", genome=genome, explicit_paths_cover_it=False)
    from paralome.pipeline.resource_catalog import ResourceResolutionError

    try:
        resolved_gtf = resolve_genome_resources(genome=genome, required=("gtf",)).path("gtf")
    except ResourceResolutionError as exc:
        raise ParalomeError(406, str(exc)) from exc

    # One object, or a folder of objects written with a trailing '/'.
    # Decided after the genome, so a configuration fault is reported first.
    from paralome.singlecell.rds_input import parse_rds_option

    rds = parse_rds_option(rds)
    # --project places the extraction where a run with
    # that --project looks for it -- the same function decides both.
    # --project is the PARENT project, the folder every
    # analysis of this object sits under at any depth. The extraction goes in
    # its data/ folder, and a run whose --project is anywhere below finds it
    # there (layout.parent_data_folder).
    if project is not None and out is not None:
        raise typer.BadParameter(
            f"extract: pass --project (the extraction goes in <project>/{DATA_DIRNAME}) "
            "or --out (used verbatim), not both."
        )
    if project is None and out is None:
        raise typer.BadParameter(
            "extract: --project or --out is required. --project is the one to use: "
            f"the extraction goes in <project>/{DATA_DIRNAME}, where every run with a "
            "--project under it finds it."
        )
    data_dir = (Path(project) / DATA_DIRNAME if project is not None else Path(out)).expanduser().resolve()
    normalized_backend = backend.strip().lower()
    if normalized_backend not in ("local", "slurm"):
        raise typer.BadParameter("is local or slurm", param_hint="--backend")
    if submit and normalized_backend != "slurm":
        raise typer.BadParameter("goes with --backend slurm", param_hint="--submit")
    if normalized_backend == "slurm":
        _submit_extraction(
            rds, data_dir, genome=genome, gtf=resolved_gtf, assay=assay, slot=slot, rna_assay=rna_assay,
            rna_slot=rna_slot, gene_symbol_column=gene_symbol_column, rscript=rscript,
            overwrite=overwrite, jobs=jobs, partition=partition, submit=submit,
        )
        return
    result = run_seurat_extraction(
        rds, layout_extraction_dir(data_dir),
        assay=assay, slot=slot, rna_assay=rna_assay, rna_slot=rna_slot,
        gene_symbol_column=gene_symbol_column, rscript=rscript, overwrite=overwrite,
        gtf=resolved_gtf,
    )
    qc = write_metadata_qc(
        read_cell_metadata(layout_extraction_dir(data_dir) / "cell_metadata.tsv"),
        data_dir,
        overwrite=False,
    )
    # `extract` is the escape hatch, so it requires nothing and reports
    # whichever halves the object turned out to have.
    if result.get("has_atac", True):
        print(f"Extracted {result.get('n_cells')} cells x {result.get('n_peaks')} peaks from {rds}")
    else:
        print(f"Extracted {result.get('n_cells')} cells from {rds} (no ATAC assay present)")
    if result.get("n_genes"):
        print(f"  plus {result['n_genes']} genes from RNA assay {result.get('rna_assay')!r}")
    # THE COLUMNS ARE PRINTED, not pointed at: the extraction exports the metadata as CSV, and this
    # names that file and reports its columns on screen.
    print()
    print(f"Cell metadata, one row per column:  {qc['files'].get('metadata_columns.csv', qc['qc_dir'])}")
    print()
    columns = qc.get("columns") or []
    if columns:
        width = max(len("column"), max(len(str(row["column"])) for row in columns))
        print(f"  {'column'.ljust(width)}  {'distinct':>8}  group by?")
        for row in columns:
            suitable = "yes" if row.get("grouping_suitable") == "yes" else ""
            print(f"  {str(row['column']).ljust(width)}  {row.get('n_distinct', '?'):>8}  {suitable}")
        print()
    candidates = qc["grouping_candidates"]
    if candidates:
        print("Columns marked `yes` can be used for --replicate-column, --factor")
        print("and --stratify-by. For the exact spelling of their values, see")
        print(f"  {qc['files'].get('metadata_values.csv', qc['qc_dir'])}")
    else:
        print("No column has few enough distinct values to group by.")
    print()
    print("Next: `paralome scRNA pca` (or `paralome snATAC pca`) with the")
    print("      --replicate-column/--stratify-by you chose, to see how samples cluster.")
    print()


def _check_factors_against_the_sheet(samples: Path, specs: Sequence[Tuple[str, str]]) -> None:
    """Each --factor column is in the sheet and holds its control level.

    Only the sheet's active samples count: a '#'-commented sample is not in the
    test. An unreadable sheet is left to the stages, which say why.
    """
    import csv

    from paralome.csvio import skip_comment_lines
    from paralome.stats import FactorSpecError

    try:
        with samples.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(skip_comment_lines(handle))
            header = [str(name).strip() for name in (reader.fieldnames or [])]
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return
    for column, control in specs:
        if column not in header:
            raise FactorSpecError(
                f"--factor {column}:{control}: the sample sheet {samples} has no column named "
                f"{column!r}. Its columns are: {', '.join(header)}"
            )
        levels = sorted({str(row.get(column) or "").strip() for row in rows} - {""})
        # A list of controls needs one of them in the sheet (each stratum is checked at its test).
        from paralome.stats import control_choices

        if not set(control_choices(control)) & set(levels):
            raise FactorSpecError(
                f"--factor {column}:{control}: {column} has no level {control!r} in {samples}. "
                f"Its levels are: {', '.join(levels) or 'none'}"
            )


def _run_common(
    *,
    title: str,
    samples: Optional[Path] = None,
    rds: Optional[Path] = None,
    replicate_column: Optional[str] = None,
    group_by: Optional[str] = None,       # Retired; naming it stops with PLM-0119
    # Relayed to the stratum array's %K throttle. _run_common takes
    # **locals() from both entry commands, so an option added to a command and
    # not here is a TypeError at invocation -- which no test caught, because
    # nothing crossed that boundary. See test_pipeline_options_reach_run_common.
    max_concurrent_strata: int = 0,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    stratify_skip: Optional[str] = None,
    min_cells_per_stratum: int = 0,
    rscript: Optional[str] = None,
    data_workdir_path: Optional[Path] = None,
    rna_assay: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
    singlecell_rna_workdir: Optional[Path] = None,
    singlecell_rna_differential_workdir: Optional[Path] = None,
    remove: List[str] = [],
    project: Optional[Path],
    pipeline_config: Optional[Path],
    reference: Optional[Path],
    star_genome: Optional[Path],
    gtf: Optional[Path],
    fastq_qc_workdir: Optional[Path],
    align_workdir: Optional[Path],
    dedup_workdir: Optional[Path],
    variant_workdir: Optional[Path],
    annotation_workdir: Optional[Path],
    segregation_workdir: Optional[Path],
    filter_workdir: Optional[Path],
    backend: Optional[str],
    submit: bool,
    overwrite: bool,
    dry_run: bool,
    threads: Optional[int],
    fastp_threads: Optional[int],
    star_threads: Optional[int],
    dedup_threads: Optional[int],
    freebayes_threads: Optional[int],
    motif_scanner_backend: str,
    alpha: str,
    rna_statistic: str,
    rna_factor: List[str],
    rna_pair_by: Optional[str],
    rna_covariate: Optional[str],
    rna_covariate_feature: Optional[str],
    rna_strandedness: Optional[str],
    stage_mem: List[str],
    rna_pseudocount: float,
    rna_block_size: int,
    rna_min_expression: Optional[str],
    rna_min_expression_difference: Optional[str],
    rna_refine_responders: Optional[int] = None,
    fastp_adapter_fasta: Optional[Path],
    fastp_cut_right: bool,
    fastp_cut_right_window_size: Optional[int],
    fastp_cut_right_mean_quality: Optional[int],
    fastp_length_required: Optional[int],
    fastp_extra: Optional[str],
    star_extra: Optional[str],
    aligner: str = "star",
    bwa_extra: Optional[str] = None,
    star_read_files_command: Optional[str],
    star_out_reads_unmapped: Optional[str],
    star_out_filter_multimap_nmax: Optional[int],
    star_out_filter_mismatch_nover_read_lmax: Optional[float],
    # Default to None/False so callers that predate the loose
    # switch keep working and resolve to the profile defaults. Every
    # command path passes them explicitly.
    star_out_filter_mismatch_nmax: Optional[int] = None,
    star_out_filter_score_min_over_lread: Optional[float] = None,
    star_out_filter_match_nmin_over_lread: Optional[float] = None,
    star_loose: bool = False,
    jobs: Optional[int],
    regions: Optional[Path],
    gene_padding: Optional[int] = None,
    min_alternate_count: int = 2,
    min_alternate_fraction: float = 0.05,
    haplotype_length: int = 3,
    min_qual: float = 20.0,
    min_depth: Optional[int] = None,
    min_gq: Optional[float] = None,
    annotation: Optional[str],
    profile: Optional[str],
    segregation_profile: Optional[str] = None,
    genome: Optional[str],
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
    partition: Optional[str],
    mark_duplicates: Optional[bool],
    remove_duplicates: Optional[bool],
    fuse_stages: Optional[bool] = None,
    fused_buffer: Optional[str] = None,
    check_complete: bool = False,
    run_all: bool,
    poll_seconds: int,
    skip_motif: bool = False,
    skip_motif_pairs: bool = False,
    skip_evidence: bool = False,
    no_genome_track: bool = False,
    no_trajectories: bool = False,
    trajectory_root: Optional[str] = None,
    trajectory_bins: int = 5,
    trajectory_fit_cells: Optional[str] = None,
    trajectory_dims: Optional[int] = None,
    trajectory_method: Optional[str] = None,
    trajectory_weld: Optional[List[str]] = None,
    trajectory_max_routes: Optional[int] = None,
    trajectory_model: Optional[str] = None,
    trajectory_lineages: Optional[str] = None,
    stop_at: Optional[str] = None,
    ancestry: Optional[str] = None,
    report_workbook: Optional[bool] = None,     # Retired, refused (PLM-0120)
    max_region_length: Optional[int] = None,
    stall_minutes: Optional[float] = None,
    extract_damaging_variants: Optional[str] = None,
    extract_all_variants: Optional[str] = None,
) -> None:
    # The samples are --replicate-column; --group-by is retired.
    if group_by is not None or replicate_column:
        pile_columns(replicate_column, retired_group_by=group_by)
    # The report's workbooks are gone.
    if report_workbook is not None:
        raise ParalomeError(120, f"--{'' if report_workbook else 'no-'}workbook: the report writes no workbooks "
                                 "now. The lists are the TSVs; OmicView's variant browser reads the "
                                 "segregation table.")
    # --extract-*-variants is a bolt-on after gene_burden, not part of the run: it must not affect
    # the rest of the run or trip an unnecessary rerun. Given, it reads the project's results and
    # nothing else: no stage is checked, run or made stale. Here, before anything else, so
    # `paralome DNA` and `pipeline run` both honour it.
    if extract_damaging_variants is not None or extract_all_variants is not None:
        if project is None:
            raise typer.BadParameter("--extract-damaging-variants/--extract-all-variants read the project's "
                                     "results; give --project")
        done = True
        if extract_damaging_variants is not None:
            done = _extract_gene_variants(Path(project), extract_damaging_variants, "damaging") and done
        if extract_all_variants is not None:
            done = _extract_gene_variants(Path(project), extract_all_variants, "all") and done
        if not done:
            raise typer.Exit(1)
        return
    # --ancestry is checked before any work, like the other values.
    if ancestry is not None:
        from paralome.burden.ancestry import check_ancestry_value

        problem = check_ancestry_value(ancestry)
        if problem:
            raise ParalomeError(108, f"--ancestry {problem}", fix="Use auto, or one of afr, amr, asj, eas, fin, mid, nfe, sas")
        ancestry = ancestry.strip().lower()
    # Validated here, before any work. One spec for every analysis.
    _alpha_specs_preview = [_parse_alpha(_piece.strip()) for _piece in alpha.split(",") if _piece.strip()]
    if not _alpha_specs_preview:
        raise typer.BadParameter("--alpha must supply at least one spec, e.g. Q0.05")
    if len(_alpha_specs_preview) > 1:
        raise typer.BadParameter(
            f"--alpha {alpha!r}: takes one spec, e.g. --alpha Q0.05; run once per cutoff."
        )
    # Zero was accepted and the fold change then took log2 of 0,
    # after counting had finished.
    if rna_pseudocount is not None and float(rna_pseudocount) <= 0:
        raise ParalomeError(
            112, f"--rna-pseudocount {rna_pseudocount} must be greater than 0; the default is 1.0",
        )
    # One value each.
    for _given, _option in ((rna_min_expression, "--rna-min-expression"),
                            (rna_min_expression_difference, "--rna-min-expression-difference")):
        if _given is not None and len(_parse_threshold_list(_given, option_name=_option)) > 1:
            raise typer.BadParameter(f"{_option} {_given!r}: takes one value, e.g. {_option} 5")
    # The refine defines its comparisons from --factor. Without one,
    # the run aligned and quantified every sample and was then refused at the
    # differential stage (see differential._check_request_options). Refused
    # here, before anything runs.
    if rna_refine_responders and not rna_factor:
        raise typer.BadParameter(
            f"--refine-responders {int(rna_refine_responders)} needs the comparisons "
            "the QC audit defines from --factor, and no --factor was given. Add the "
            "factor column(s) of the sample sheet (--factor <column>), or leave out "
            "--refine-responders."
        )
    preliminary = resolve_pipeline_profile(profile)
    stop_at_stage: Optional[str] = None
    if stop_at:
        try:
            stop_at_stage = resolve_stage_alias(stop_at, preliminary.stage_graph)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    # The analysis is named from the resolved profile, so the message always says what the user
    # actually typed, for every profile, with no list to maintain (names map all the way through,
    # case-sensitively: scRNA != scrna).
    analysis = preliminary.pipeline_profile.value
    if analysis == "snATAC":
        if samples is not None:
            raise ParalomeError(102, f"{analysis} takes --rds, not --samples")
        missing = [
            name for name, value in (
                # --condition-column is optional (it names a Seurat
                # metadata column; explicit --control/--treatment member lists
                # need nothing looked up). --stratify-value is
                # optional too -- omitting it fans out over every stratum.
                # --stratify-by dropped from this list, so snATAC and
                # scRNA require exactly the same five options. It was required
                # here and nowhere else, which meant the two profiles could not
                # be driven by the same script.
                # --control/--treatment dropped too. Whether a run
                # needs them is the statistic's question, answered by
                # comparisons.plan_differential (t/paired-t without them or a
                # --factor is refused there); here they blocked lm alone,
                # --factor alone, and anova, which refuses them.
                ("--rds", rds), ("--replicate-column", replicate_column),
            ) if not value
        ]
        if missing:
            raise ParalomeError(101, f"{analysis} requires: {', '.join(missing)}")
        if project is None:
            raise ParalomeError(101, f"--project is required for {analysis}")
    elif analysis == "scRNA":
        if samples is not None:
            raise ParalomeError(102, f"{analysis} takes --rds, not --samples")
        missing = [
            name for name, value in (
                # --condition-column optional -- see the snATAC branch.
                # --control/--treatment too -- see the snATAC branch.
                ("--rds", rds), ("--replicate-column", replicate_column),
            ) if not value
        ]
        if missing:
            raise ParalomeError(101, f"{analysis} requires: {', '.join(missing)}")
        if project is None:
            raise typer.BadParameter(f"--project is required for {analysis}")
    elif samples is None:
        raise typer.BadParameter(f"--samples is required for {analysis}")
    if analysis in ("RNA", "ATAC", "snATAC", "scRNA"):
        # The groups are --factor COLUMN:CONTROL. Checked
        # here, before anything is queued, rather than at the differential
        # stage hours later: each --factor names its control (PLM-0113), a
        # statistic that compares levels has a --factor, and on a bulk sheet
        # the column exists and holds the control level. A malformed
        # --statistic is left to the stage, which names what is wrong with it.
        from paralome.stats import parse_factor, parse_statistic_request

        specs = [parse_factor(spec) for spec in rna_factor]
        graph = list(preliminary.stage_graph)
        differential = next((i for i, name in enumerate(graph) if "differential" in name), None)
        # A run told to stop before its differential (--stop-at alignment, say)
        # tests nothing, and needs neither a statistic nor a factor.
        reaches_the_test = differential is not None and (
            stop_at_stage is None or stop_at_stage not in graph or graph.index(stop_at_stage) >= differential
        )
        # --statistic is required and has no default: the user types it.
        if rna_statistic is None and reaches_the_test:
            raise ParalomeError(
                101,
                "--statistic is required: name the test(s), any of t, paired-t, lm and anova, "
                "e.g. --statistic t or --statistic t,anova",
            )
        try:
            if rna_statistic is None:
                requested = ()
            elif analysis in ("snATAC", "scRNA"):
                from paralome.singlecell.stats_core import DATA_LEVELS, DATA_LEVEL_PSEUDOBULK

                requested = parse_statistic_request(
                    rna_statistic, data_levels=DATA_LEVELS,
                    default_data_level=DATA_LEVEL_PSEUDOBULK, validate_companions=False,
                ).statistics
            else:
                requested = parse_statistic_request(rna_statistic, validate_companions=False).statistics
        except ValueError:
            requested = ()
        compared = [s for s in requested if s in ("t", "paired-t", "anova")]
        if compared and not specs and reaches_the_test:
            raise ParalomeError(
                101,
                f"--statistic {','.join(compared)} compares the levels of a --factor: give "
                "--factor COLUMN:CONTROL (e.g. --factor treatment:VEH)",
            )
        if specs and samples is not None and analysis in ("RNA", "ATAC"):
            _check_factors_against_the_sheet(Path(samples), specs)
    if rna_statistic is None:
        # A run that never reaches a differential (DNA, or --stop-at before
        # it): no stage reads this, and the stage commands only need a value.
        rna_statistic = "t"
    resolved = _resolve_pipeline_inputs(
        project=project,
        pipeline_config=pipeline_config,
        samples=samples,
        rds=rds,
        replicate_column=replicate_column,
        stratify_by=stratify_by,
        stratify_value=stratify_value,
        stratify_skip=stratify_skip,
        min_cells_per_stratum=min_cells_per_stratum,
        rscript=rscript,
        data_workdir_path=data_workdir_path,
        rna_assay=rna_assay,
        gene_symbol_column=gene_symbol_column,
        singlecell_rna_workdir=singlecell_rna_workdir,
        singlecell_rna_differential_workdir=singlecell_rna_differential_workdir,
        threads=threads,
        jobs=jobs,
        backend=backend,
        fastq_qc_workdir=fastq_qc_workdir,
        align_workdir=align_workdir,
        dedup_workdir=dedup_workdir,
        variant_workdir=variant_workdir,
        annotation_workdir=annotation_workdir,
        segregation_workdir=segregation_workdir,
        filter_workdir=filter_workdir,
        report_workdir=None,
        fastp_threads=fastp_threads,
        star_threads=star_threads,
        dedup_threads=dedup_threads,
        freebayes_threads=freebayes_threads,
        max_region_length=max_region_length,
        stall_minutes=stall_minutes,
        fastp_adapter_fasta=fastp_adapter_fasta,
        fastp_cut_right=fastp_cut_right,
        fastp_cut_right_window_size=fastp_cut_right_window_size,
        fastp_cut_right_mean_quality=fastp_cut_right_mean_quality,
        fastp_length_required=fastp_length_required,
        fastp_extra=fastp_extra,
        star_extra=star_extra,
        aligner=aligner,
        bwa_extra=bwa_extra,
        star_read_files_command=star_read_files_command,
        star_out_reads_unmapped=star_out_reads_unmapped,
        star_out_filter_multimap_nmax=star_out_filter_multimap_nmax,
        star_out_filter_mismatch_nover_read_lmax=star_out_filter_mismatch_nover_read_lmax,
        star_out_filter_mismatch_nmax=star_out_filter_mismatch_nmax,
        star_out_filter_score_min_over_lread=star_out_filter_score_min_over_lread,
        star_out_filter_match_nmin_over_lread=star_out_filter_match_nmin_over_lread,
        star_loose=star_loose,
        annotation=annotation,
        profile=profile,
        segregation_profile=segregation_profile,
        genome=genome,
        annovar_buildver=annovar_buildver,
        annovar_protocol=annovar_protocol,
        annovar_operation=annovar_operation,
        snpeff_genome=snpeff_genome,
        mark_duplicates=mark_duplicates,
        remove_duplicates=remove_duplicates,
        fuse_stages=fuse_stages,
        fused_buffer=fused_buffer,
        rna_strandedness=rna_strandedness,
        rna_min_expression=rna_min_expression,
        rna_min_expression_difference=rna_min_expression_difference,
        # --stage-mem was parsed by the function and never handed
        # to it, so every analysis ran on the configured or built-in memory.
        stage_mem=stage_mem,
    )
    # A backend from the configuration is checked like one typed.
    if resolved["backend"] not in ("slurm", "local"):
        raise ParalomeError(
            108,
            f"{_backend_source(backend, pipeline_config)} "
            f"{resolved['backend']!r} is not a backend; use slurm or local",
        )
    # Say which backend this run uses and what chose it (setup records `local` where sbatch is not on
    # PATH; with --submit such a run trims and aligns on this machine).
    backend_source = _backend_source(backend, pipeline_config)
    if submit and resolved["backend"] == "local":
        because = (
            "--backend local was given" if backend
            else f"{backend_source} says so"
        )
        raise ParalomeError(
            103,
            "--submit is for the SLURM backend; this run's backend is local "
            f"because {because}.",
            fix="Pass --backend slurm"
                + ("" if backend else ", or change that setting,")
                + " to submit; or drop --submit to run here",
        )
    print(f"Backend             : {resolved['backend']} ({backend_source})")
    if resolve_pipeline_profile(profile).pipeline_profile.value == "RNA":
        print(f"RNA strandedness    : {resolved['rna_strandedness']} "
              f"({_strandedness_source(rna_strandedness, pipeline_config)})")
    annotation_engines = (
        _annotation_engines(
            resolved["annotation"],
            source="--annotation" if annotation else "annotation.engine in the configuration",
        )
        if preliminary.is_variant else []
    )
    given_paths = dict(
        annotation_engines=annotation_engines,
        star_genome=star_genome, reference=reference,
        annovar_db=annovar_db, annovar_buildver=annovar_buildver,
        annovar_protocol=resolved["annovar_protocol"],
        annovar_operation=resolved["annovar_operation"],
        snpeff_jar=snpeff_jar, snpeff_config=snpeff_config, snpeff_genome=snpeff_genome,
        aligner=resolved["aligner"],
    )
    _require_a_genome(
        analysis=analysis,
        genome=resolved["genome"],
        explicit_paths_cover_it=explicit_paths_cover_the_genome(
            is_variant=preliminary.is_variant, **given_paths,
        ),
        lacking=(
            tuple(what_explicit_paths_lack(**given_paths))
            if preliminary.is_variant and star_genome is not None and reference is not None
            else ()
        ),
    )
    # What the preflight checks, the same for a dry run and a real one. The
    # genome paths are the ones given as options, before the genome fills in
    # the rest below.
    run_stages = [
        stage for stage in preliminary.stage_graph
        if not (skip_motif and "motif_analysis" in stage)
        # --skip-evidence, the counterpart of --skip-motif. Evidome is on every graph, so there has to be
        # a way to say no to it that is not "stop the pipeline one stage early".
        and not (skip_evidence and stage == "functional_evidence")
        # --no-genome-track, 
        and not (no_genome_track and stage == "genome_tracks")
        # --no-trajectories.
        and not (no_trajectories and stage == "trajectories")
    ]
    if stop_at_stage in run_stages:
        run_stages = run_stages[: run_stages.index(stop_at_stage) + 1]
    # A run that reaches the trajectories names their root, or says no to them -- before
    # anything is queued.
    if "trajectories" in run_stages and not trajectory_root:
        raise ParalomeError(
            121,
            "This run reaches the trajectories stage, which needs the cluster its lineages start from: "
            "--trajectory-root <column>:<value> (e.g. --trajectory-root celltype:RPC). Or --no-trajectories "
            "to leave it out, and run again with --trajectory-root once the UMAP has been seen.",
        )
    # Singlecell_rna fits the lineages, so the root and the fit's settings are read before
    # anything is queued whenever the run has them -- --stop-at singlecell_rna too -- rather than
    # when the stage starts.
    if "trajectories" in run_stages or (trajectory_root and not no_trajectories and "singlecell_rna" in run_stages):
        from paralome.singlecell.trajectories import check_root, parse_root
        from paralome.singlecell.trajectories import read_cell_metadata as read_extracted_cells

        root_column, root_value = parse_root(trajectory_root)
        # The fit's options are read now, not when the stage starts.
        from paralome.singlecell.trajectories import check_fit_options

        check_fit_options(trajectory_method, trajectory_fit_cells, trajectory_dims, trajectory_weld,
                          trajectory_max_routes)
        # And the statistics' own.
        from paralome.singlecell.trajectory_genes import parse_lineage_set, parse_model

        parse_model(trajectory_model)
        parse_lineage_set(trajectory_lineages)
        # And when the object is already extracted (a run adding the trajectories afterwards), the
        # column and the value are checked against its cells now, not when the stage starts (PLM-0234).
        if resolved.get("data_workdir_path"):
            extracted = layout_extraction_dir(Path(resolved["data_workdir_path"]))
            if (extracted / "cell_metadata.tsv").is_file():
                check_root(read_extracted_cells(extracted), root_column, root_value)
    if "annotation" in run_stages:
        _annotation_settings(
            resolved, annotation_engines, pipeline_config=pipeline_config,
            typed={
                "annovar_buildver": annovar_buildver,
                "annovar_protocol": annovar_protocol,
                "annovar_operation": annovar_operation,
                "snpeff_genome": snpeff_genome,
            },
        )
    preflight_kwargs = dict(
        samples=samples,
        # The R whose slingshot the trajectories need.
        rscript=resolved.get("rscript") or rscript,
        genome=resolved["genome"],
        stages=tuple(run_stages),
        # Singlecell_rna fits the lineages when a root is named -- slingshot is checked for it too.
        trajectory_fit=bool(trajectory_root and not no_trajectories),
        trajectory_method=trajectory_method,
        trajectory_model=trajectory_model,
        aligner=resolved["aligner"],
        # The column rules the run's first stage applies to the sheet.
        manifest_profile=(
            resolve_pipeline_profile(
                resolved["profile"], resolved["segregation_profile"],
            ).manifest_profile.value
            if samples is not None else None
        ),
        factors=tuple(rna_factor),
        explicit={
            "reference_fasta": reference,
            "star_index": star_genome,
            "annovar_dir": annovar_dir,
            "annovar_db": annovar_db,
            "snpeff_jar": snpeff_jar,
            "snpeff_config": snpeff_config,
            "snpeff_genome": resolved["snpeff_genome"],
            "annovar_buildver": resolved["annovar_buildver"],
            "annovar_protocol": resolved["annovar_protocol"],
            "annovar_operation": resolved["annovar_operation"],
        },
        annotation=",".join(annotation_engines),
    )
    reference, star_genome, gtf = _resolve_pipeline_genome_paths(
        genome=resolved["genome"],
        reference=reference,
        star_genome=star_genome,
        gtf=gtf,
    )

    state = build_project_state(
        samples=samples,
        fastq_qc_workdir=resolved["fastq_qc_workdir"],
        align_workdir=resolved["align_workdir"],
        dedup_workdir=resolved["dedup_workdir"],
        variant_workdir=resolved["variant_workdir"],
        annotation_workdir=resolved["annotation_workdir"],
        segregation_workdir=resolved["segregation_workdir"],
        filter_workdir=resolved["filter_workdir"],
        report_workdir=resolved["report_workdir"],
        rds=resolved["rds"],
        data_workdir_path=resolved["data_workdir_path"],
        singlecell_differential_workdir=resolved["singlecell_differential_workdir"],
        singlecell_motif_workdir=resolved["singlecell_motif_workdir"],
        singlecell_rna_workdir=resolved["singlecell_rna_workdir"],
        singlecell_rna_differential_workdir=resolved["singlecell_rna_differential_workdir"],
        profile=resolved["profile"],
        segregation_profile=resolved["segregation_profile"],
        skip_motif=skip_motif,
        skip_evidence=skip_evidence,
        no_genome_track=no_genome_track,
        no_trajectories=no_trajectories,
        trajectory_root=trajectory_root, trajectory_fit_cells=trajectory_fit_cells,
        trajectory_dims=trajectory_dims, trajectory_method=trajectory_method,
        trajectory_weld=tuple(trajectory_weld or ()), trajectory_max_routes=trajectory_max_routes,
        skip_motif_pairs=skip_motif_pairs,
        stop_at=stop_at_stage,
        # The --factor columns stand in for `condition` in the sheet.
        factors=tuple(rna_factor),
    )
    pipeline = state.get("pipeline", {}) if isinstance(state.get("pipeline"), dict) else {}
    if check_complete and pipeline.get("state") == "complete" and not overwrite:
        target = _state_target(resolved)
        # A dry run writes nothing, this file included.
        out_path = None if dry_run else write_project_state(target, state)
        print(title)
        print("=" * len(title))
        print("Workflow state       : complete")
        print("Next stage           : .")
        print(f"Workdir              : {target}")
        print("Executed             : False")
        print("Message              : Pipeline is already complete. No action required.")
        print("Command              : .")
        if out_path is not None:
            print(f"Project state JSON   : {out_path}")
        return

    stage_kwargs = dict(
        samples=samples,
        #
        trajectory_root=trajectory_root,
        trajectory_bins=trajectory_bins,
        trajectory_fit_cells=trajectory_fit_cells,
        trajectory_dims=trajectory_dims,
        trajectory_method=trajectory_method,
        trajectory_weld=list(trajectory_weld or []),
        trajectory_max_routes=trajectory_max_routes,
        trajectory_model=trajectory_model,
        trajectory_lineages=trajectory_lineages,
        # The gene_burden stage's ancestry option (DNA).
        ancestry=ancestry,
        rds=resolved["rds"],
        replicate_column=resolved["replicate_column"],
        stratify_by=resolved["stratify_by"],
        stratify_value=resolved["stratify_value"],
        stratify_skip=resolved.get("stratify_skip"),
        # This value must survive every hop of the relay (resolved -> stage_kwargs -> the supervisor's
        # sbatch); each hop reads it back out explicitly, so the job gets the value given, not the
        # default.
        min_cells_per_stratum=resolved.get("min_cells_per_stratum", 0),
        max_concurrent_strata=max_concurrent_strata,
        rscript=resolved["rscript"],
        data_workdir_path=resolved["data_workdir_path"],
        singlecell_differential_workdir=resolved["singlecell_differential_workdir"],
        singlecell_motif_workdir=resolved["singlecell_motif_workdir"],
        rna_assay=resolved["rna_assay"],
        gene_symbol_column=resolved["gene_symbol_column"],
        singlecell_rna_workdir=resolved["singlecell_rna_workdir"],
        singlecell_rna_differential_workdir=resolved["singlecell_rna_differential_workdir"],
        remove_samples=tuple(remove),
        reference=reference,
        star_genome=star_genome,
        gtf=gtf,
        fastq_qc_workdir=resolved["fastq_qc_workdir"],
        align_workdir=resolved["align_workdir"],
        dedup_workdir=resolved["dedup_workdir"],
        variant_workdir=resolved["variant_workdir"],
        annotation_workdir=resolved["annotation_workdir"],
        segregation_workdir=resolved["segregation_workdir"],
        filter_workdir=resolved["filter_workdir"],
        report_workdir=resolved["report_workdir"],
        backend=resolved["backend"],
        submit=submit,
        overwrite=overwrite,
        dry_run=dry_run,
        fastp_threads=resolved["fastp_threads"],
        star_threads=resolved["star_threads"],
        dedup_threads=resolved["dedup_threads"],
        atac_threads=resolved["threads"],
        motif_scanner_backend=motif_scanner_backend,
        alpha=alpha,
        rna_statistic=rna_statistic,
        rna_factors=tuple(rna_factor),
        rna_pair_by=rna_pair_by,
        rna_covariate=rna_covariate,
        rna_covariate_feature=rna_covariate_feature,
        rna_strandedness=resolved["rna_strandedness"],
        stage_memories=resolved["stage_memories"],
        rna_pseudocount=rna_pseudocount,
        rna_block_size=rna_block_size,
        rna_min_expression=resolved["rna_min_expression"],
        rna_min_expression_difference=resolved["rna_min_expression_difference"],
        rna_refine_responders=rna_refine_responders,
        freebayes_threads=resolved["freebayes_threads"],
        # How FreeBayes calls are run; not part of the stage's request.
        freebayes_max_region_length=resolved["freebayes_max_region_length"],
        freebayes_stall_minutes=resolved["freebayes_stall_minutes"],
        report_threads=resolved["threads"],
        report_shards=max(1, resolved["jobs"] or resolved["threads"]),
        fastp_adapter_fasta=resolved["fastp_adapter_fasta"],
        fastp_cut_right=resolved["fastp_cut_right"],
        fastp_cut_right_window_size=resolved["fastp_cut_right_window_size"],
        fastp_cut_right_mean_quality=resolved["fastp_cut_right_mean_quality"],
        fastp_length_required=resolved["fastp_length_required"],
        fastp_extra=resolved["fastp_extra"],
        star_extra=resolved["star_extra"],
        aligner=resolved["aligner"],
        bwa_extra=resolved["bwa_extra"],
        star_read_files_command=resolved["star_read_files_command"],
        star_out_reads_unmapped=resolved["star_out_reads_unmapped"],
        star_out_filter_multimap_nmax=resolved["star_out_filter_multimap_nmax"],
        star_out_filter_mismatch_nover_read_lmax=resolved["star_out_filter_mismatch_nover_read_lmax"],
        star_out_filter_mismatch_nmax=resolved["star_out_filter_mismatch_nmax"],
        star_out_filter_score_min_over_lread=resolved["star_out_filter_score_min_over_lread"],
        star_out_filter_match_nmin_over_lread=resolved["star_out_filter_match_nmin_over_lread"],
        star_loose=resolved["star_loose"],
        jobs=resolved["jobs"],
        regions=regions,
        gene_padding=gene_padding,
        min_alternate_count=min_alternate_count,
        min_alternate_fraction=min_alternate_fraction,
        haplotype_length=haplotype_length,
        min_qual=min_qual,
        min_depth=min_depth,
        min_gq=min_gq,
        annotation=resolved["annotation"],
        profile=resolved["profile"],
        segregation_profile=resolved["segregation_profile"],
        genome=resolved["genome"],
        annovar_dir=annovar_dir,
        annovar_db=annovar_db,
        annovar_buildver=resolved["annovar_buildver"],
        annovar_protocol=resolved["annovar_protocol"],
        annovar_operation=resolved["annovar_operation"],
        snpeff_jar=snpeff_jar,
        snpeff_config=snpeff_config,
        snpeff_genome=resolved["snpeff_genome"],
        partition=partition,
        mark_duplicates=resolved["mark_duplicates"],
        remove_duplicates=resolved["remove_duplicates"],
        fuse_stages=resolved["fuse_stages"],
        fused_buffer=resolved["fused_buffer"],
        skip_motif=skip_motif,
        skip_evidence=skip_evidence,
        no_genome_track=no_genome_track,
        no_trajectories=no_trajectories,
        skip_motif_pairs=skip_motif_pairs,
        stop_at=stop_at_stage,
    )

    # A DRY RUN CHECKS TOO: the one invocation whose purpose is asking "would this work?" reports
    # these checks.
    #
    # Separate from the block below rather than folded into it: a dry run
    # should not require --project or a writable project root, because those
    # are conditions for SUBMITTING and a dry run submits nothing.
    if run_all and dry_run:
        from paralome.pipeline.preflight import (
            print_preflight, refuse_if_not_ready, run_preflight,
        )

        # With this run's stages, as the real run passes them. Without
        # them a dry run refused for a stale tool the run never uses.
        report = run_preflight(**preflight_kwargs)
        print_preflight(report)

        # The dispatch plan -- what kind of input this is and what runs first. Validation is the
        # preflight above, which always runs.
        if samples is not None and Path(samples).is_file():
            try:
                from paralome.pipeline.dispatcher import (
                    build_dispatch_plan, print_dispatch_plan,
                )

                print("")
                # For this run's profile and --factor columns, as
                # the run itself builds it. Without them the plan named
                # variant calling for every profile and could not read a sheet
                # that has factor columns in place of `condition`.
                plan_profile = resolve_pipeline_profile(
                    resolved["profile"], resolved["segregation_profile"],
                )
                print_dispatch_plan(build_dispatch_plan(
                    samples=samples,
                    fastq_qc_workdir=resolved.get("fastq_qc_workdir"),
                    align_workdir=resolved.get("align_workdir"),
                    variant_workdir=resolved.get("variant_workdir"),
                    validate_paths=False,
                    manifest_profile=plan_profile.manifest_profile.value,
                    stage_graph=plan_profile.stage_graph,
                    factors=tuple(rna_factor),
                ))
            except Exception as exc:
                print("")
                print(f"Dispatch plan unavailable: {exc}")

        refuse_if_not_ready(report)

    if run_all and not dry_run:
        if resolved.get("project_root") is None:
            raise ParalomeError(101, "--project is required for a complete pipeline run")
        # Fail HERE, in the foreground, where the user is standing -- not on a compute node one poll
        # cycle later. Nothing about "this path does not exist" needs a scheduler to find out. Every
        # check below is answerable on the login node in milliseconds.
        unwritable = writable_reason(resolved["project_root"])
        if unwritable:
            raise ParalomeError(107, unwritable)

        check_input_path(rds, flag="--rds")
        check_input_path(samples, flag="--samples")
        # No check of a GTF here: the genome's GTF is checked by the preflight below with its other paths.

        # OPEN the sample sheet, resolve the genome, check the
        # recorded tools -- before the first sbatch. The checks above confirm a
        # file EXISTS; these confirm what is inside it is usable.
        from paralome.pipeline.preflight import refuse_if_not_ready, run_preflight

        # Hand the preflight this run's stage graph, so a stale
        # recording for a tool the run never invokes is reported rather than
        # refused. An RNA run was otherwise blocked by stale bcftools, bgzip and
        # tabix -- three variant-calling tools an RNA stage graph does not
        # contain.
        refuse_if_not_ready(run_preflight(**preflight_kwargs))
        check_bounded_number(threads, flag="--threads", minimum=1)
        check_bounded_number(jobs, flag="--jobs", minimum=1)
        check_bounded_number(
            min_cells_per_stratum, flag="--min-cells-per-stratum", minimum=0
        )
        # Metadata columns are only knowable once the object has been read. On
        # the FIRST run they are checked inside the stage, at the earliest point
        # the answer exists. On every run after that the extraction is on disk,
        # so the same question can be answered here instead -- which is the
        # difference between a wrong column name costing a second and costing a
        # queue slot.
        if rds is not None and resolved.get("data_workdir_path"):
            columns = extraction_metadata_columns(
                layout_extraction_dir(Path(resolved["data_workdir_path"]))
            )
            # A --factor column of a single-cell run is a cell
            # metadata column carried onto the pseudobulk manifest.
            from paralome.stratify import stratum_columns

            requested = [("--stratify-by", column) for column in stratum_columns(stratify_by)] or [("--stratify-by", None)]
            requested += [("--factor", spec.split(":", 1)[0]) for spec in rna_factor]
            if replicate_column:
                requested.append(("--replicate-column", replicate_column))
            check_metadata_columns(requested, columns)
        state_kwargs = dict(
            samples=samples,
            fastq_qc_workdir=resolved["fastq_qc_workdir"],
            align_workdir=resolved["align_workdir"],
            dedup_workdir=resolved["dedup_workdir"],
            variant_workdir=resolved["variant_workdir"],
            annotation_workdir=resolved["annotation_workdir"],
            segregation_workdir=resolved["segregation_workdir"],
            filter_workdir=resolved["filter_workdir"],
            report_workdir=resolved["report_workdir"],
            rds=resolved["rds"],
            data_workdir_path=resolved["data_workdir_path"],
            singlecell_differential_workdir=resolved["singlecell_differential_workdir"],
            singlecell_motif_workdir=resolved["singlecell_motif_workdir"],
            singlecell_rna_workdir=resolved["singlecell_rna_workdir"],
            singlecell_rna_differential_workdir=resolved["singlecell_rna_differential_workdir"],
            profile=resolved["profile"],
            segregation_profile=resolved["segregation_profile"],
            skip_motif=skip_motif,
        skip_evidence=skip_evidence,
        no_genome_track=no_genome_track,
        no_trajectories=no_trajectories,
        trajectory_root=trajectory_root, trajectory_fit_cells=trajectory_fit_cells,
        trajectory_dims=trajectory_dims, trajectory_method=trajectory_method,
        trajectory_weld=tuple(trajectory_weld or ()), trajectory_max_routes=trajectory_max_routes,
            skip_motif_pairs=skip_motif_pairs,
            stop_at=stop_at_stage,
            factors=tuple(rna_factor),
        )
        completed = run_pipeline_to_completion(
            stage_kwargs=stage_kwargs,
            state_kwargs=state_kwargs,
            project=Path(resolved["project_root"]),
            poll_seconds=poll_seconds,
        )
        print(title)
        print("=" * len(title))
        print(f"Workflow state       : {'complete' if completed.complete else 'incomplete'}")
        print(f"Stages run           : {','.join(completed.stages_run) or '.'}")
        print(f"Final workdir        : {completed.report_workdir}")
        print(f"Project state JSON   : {completed.state_path}")
        return

    if dry_run:
        # A DRY RUN WRITES NOTHING AND CREATES NOTHING. It wrote
        # inputs/pipeline_run_plan.json under the stage folder, and the project
        # state below, so asking "would this work?" made the project folder.
        # Building the command for the stage that is next is the same work
        # without the plan file; the plan is printed instead.
        #
        # WITH THE RUN'S OWN VERDICTS. The state above knows nothing
        # of the stages an earlier run left stale, nor of an option changed
        # since a stage ran, so on a finished project with a changed option
        # the dry run said there was nothing to do. The supervisor works out,
        # and prints, what the run would find; it writes nothing either. What
        # the run would refuse (PLM-0308) the dry run refuses.
        from paralome.pipeline.supervisor import _stale_entries, preview_pipeline_run

        preview = preview_pipeline_run(
            stage_kwargs=stage_kwargs, project=resolved.get("project_root"),
        )
        if preview["refusal"] is not None:
            raise preview["refusal"]
        result = run_pipeline_next_stage(
            **stage_kwargs, stale_stages=_stale_entries(preview["stale"]),
            _only_build_stage=preview["next_stage"],
        )
        _print_pipeline_run_result(title, result, recorded=False)
        return

    result = run_pipeline_next_stage(**stage_kwargs)
    if resolved.get("project_root") is not None:
        # Keep a top-level project_state.json current after each step.
        try:
            refreshed = build_project_state(
                samples=samples,
                fastq_qc_workdir=resolved["fastq_qc_workdir"],
                align_workdir=resolved["align_workdir"],
                dedup_workdir=resolved["dedup_workdir"],
                variant_workdir=resolved["variant_workdir"],
                annotation_workdir=resolved["annotation_workdir"],
                segregation_workdir=resolved["segregation_workdir"],
                filter_workdir=resolved["filter_workdir"],
                report_workdir=resolved["report_workdir"],
                rds=resolved["rds"],
                data_workdir_path=resolved["data_workdir_path"],
                singlecell_differential_workdir=resolved["singlecell_differential_workdir"],
                singlecell_motif_workdir=resolved["singlecell_motif_workdir"],
                singlecell_rna_workdir=resolved["singlecell_rna_workdir"],
                singlecell_rna_differential_workdir=resolved["singlecell_rna_differential_workdir"],
                profile=resolved["profile"],
                skip_motif=skip_motif,
        skip_evidence=skip_evidence,
        no_genome_track=no_genome_track,
        no_trajectories=no_trajectories,
        trajectory_root=trajectory_root, trajectory_fit_cells=trajectory_fit_cells,
        trajectory_dims=trajectory_dims, trajectory_method=trajectory_method,
        trajectory_weld=tuple(trajectory_weld or ()), trajectory_max_routes=trajectory_max_routes,
                skip_motif_pairs=skip_motif_pairs,
                stop_at=stop_at_stage,
                factors=tuple(rna_factor),
            )
            write_project_state(Path(resolved["project_root"]), refreshed)
        except Exception:
            pass
    _print_pipeline_run_result(title, result)


def _extract_gene_variants(project: Path, genes: str, kind: str) -> bool:
    """--extract-damaging-variants / --extract-all-variants.

    Not a stage, and the pipeline is not run. True when it read what it
    reads (a gene with no file is named); False when there was nothing to
    read (no gene given, gene_burden not complete, no segregation table).
    """
    from paralome.burden.extract import (
        extract_all_variants, extract_damaging_variants, parse_gene_list, print_extract_result,
    )

    option = f"--extract-{kind}-variants"
    wanted = parse_gene_list(genes)
    if not wanted:
        print(f"Variants not extracted: {option} names no gene")
        return False
    try:
        result = (extract_damaging_variants if kind == "damaging" else extract_all_variants)(project, wanted)
    except ValueError as exc:
        print(f"Variants not extracted ({option}): {exc}")
        return False
    print_extract_result(result)
    return True


def pipeline_run_command(
    samples: Optional[Path] = typer.Option(None, "--samples", help="Input sample manifest CSV; may contain BAMs or FASTQs"),
    rds: Optional[str] = typer.Option(None, "--rds", help="Seurat/Signac .rds object holding the cells this run aggregates; or a folder of objects, written with a trailing /"),
    replicate_column: Optional[str] = typer.Option(None, "--replicate-column", help="scRNA/snATAC: " + REPLICATE_COLUMN_HELP),
    group_by: Optional[str] = typer.Option(None, "--group-by", hidden=True, help=RETIRED_GROUP_BY_HELP),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", help=STRATIFY_HELP),
    rscript: Optional[str] = typer.Option(None, "--rscript", help="snATAC: Rscript executable for Seurat/Signac extraction; pass an absolute path if it's not on PATH in the execution environment (e.g. only available after an HPC module load) -- default 'Rscript' if omitted"),
    stratify_value: Optional[str] = typer.Option(
        None, "--stratify-value",
        help=(
            "snATAC/scRNA: comma-separated --stratify-by value(s) to run, e.g. "
            "'CD8 TEM,CD4 Naive'. Either the raw metadata value or the underscored "
            "slug you see in filenames. Omit to run every stratum."
        ),
    ),
    stratify_skip: Optional[str] = typer.Option(
        None, "--stratify-skip",
        help=(
            "snATAC/scRNA: comma-separated --stratify-by value(s) to EXCLUDE, running "
            "all the others. Mutually exclusive with --stratify-value."
        ),
    ),
    min_cells_per_stratum: int = typer.Option(
        0, "--min-cells-per-stratum", min=0,
        help=(
            "Skip a --stratify-by group whose TOTAL cells, summed across its "
            "pseudobulk samples, fall below this. 0 disables it. Distinct from "
            "--min-cells-per-group, which drops one sample (donor x group) during "
            "AGGREGATION and therefore changes these totals -- raise that and these "
            "numbers move. Skipped groups are reported, not silently absent."
        ),
    ),
    data_workdir_path: Optional[Path] = typer.Option(
        None, "--data-workdir",
        help="snATAC/scRNA: override the shared data workdir (default <project-parent>/data). "
             "Holds the per-cell extraction, both modalities' pseudobulk matrices and the QC "
             "tables. Point several comparisons at one shared directory to run Seurat/Signac "
             "extraction and pseudobulk aggregation once instead of per comparison -- "
             "replicate-column/"
             "stratify-by/genome must be identical across every comparison sharing it; only "
             "--stratify-value/--statistic/--factor vary per comparison downstream. snATAC and scRNA "
             "profiles pointed at the same --out/--data-workdir share one combined Seurat extraction.",
    ),
    rna_assay: Optional[str] = typer.Option(
        None, "--rna-assay",
        help="scRNA: Seurat assay holding the gene-level counts. Auto-detected -- anything "
             "that is not ATAC is gene-level, whatever it is named. Pass this only to break "
             "a tie between several such assays.",
    ),
    gene_symbol_column: Optional[str] = typer.Option(
        None, "--gene-symbol-column",
        help="scRNA: column in the RNA assay's meta.features carrying the gene symbol/common name "
             "(secondary ID). Omit for gene_id (Ensembl/primary ID) only.",
    ),
    singlecell_rna_differential_workdir: Optional[Path] = typer.Option(
        None, "--singlecell-rna-differential-workdir",
        help="scRNA: override the single-cell RNA differential-expression stage workdir "
             "(default <project>/rna).",
    ),
    remove: List[str] = typer.Option(
        [], "--remove",
        help="snATAC/scRNA: pseudobulk sample_id to exclude outright from the pseudobulk aggregation "
             "(repeatable) -- identify outliers from a prior run's cell_counts_per_sample.csv "
             "or global-PCA plot.",
    ),
    project: Optional[Path] = typer.Option(None, "--project", help="Project workdir; stage workdirs are created under this directory"),
    pipeline_config: Optional[Path] = typer.Option(None, "--pipeline-config", help="Optional JSON pipeline defaults config"),
    reference: Optional[Path] = typer.Option(None, "--reference", help="Reference FASTA for FreeBayes when variants are the next stage"),
    star_genome: Optional[Path] = typer.Option(None, "--star-genome", help="STAR genomeDir when alignment is the next stage"),
    # NOT AN OPTION ANY MORE, and deliberately still a parameter;
    # --genome replaced it.
    # _resolve_pipeline_genome_paths already fills this from the registered
    # genome; the flag was an override that let a hand-typed GTF disagree with
    # the genome the run recorded, with nothing to say which one it used.
    gtf: Optional[Path] = None,
    fastq_qc_workdir: Optional[Path] = typer.Option(None, "--fastq-qc-workdir", help="Override FASTQ QC workdir"),
    align_workdir: Optional[Path] = typer.Option(None, "--align-workdir", help="Override STAR alignment workdir"),
    dedup_workdir: Optional[Path] = typer.Option(None, "--dedup-workdir", help="Override duplicate marking workdir"),
    variant_workdir: Optional[Path] = typer.Option(None, "--variant-workdir", "--out", help="Override variant/FreeBayes workdir"),
    annotation_workdir: Optional[Path] = typer.Option(None, "--annotation-workdir", help="Override standalone annotation workdir"),
    segregation_workdir: Optional[Path] = typer.Option(None, "--segregation-workdir", help="Override segregation workdir"),
    filter_workdir: Optional[Path] = typer.Option(None, "--filter-workdir", help="Override filtering workdir"),
    # No default of its own, so defaults.backend in the user config is reached.
    backend: Optional[str] = typer.Option(None, "--backend", help="Execution backend passed to each stage: slurm or local. Default: defaults.backend in your configuration, else slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit the run to SLURM (SLURM backend only)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs for selected next stage"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check the inputs and print the command that would run. Nothing is run, written or created"),
    threads: Optional[int] = typer.Option(None, "--threads", "--thread", help="Total CPUs each stage may use, divided among the samples running at once (--jobs); variant calling and annotation run this many one-CPU pieces at a time"),
    fastp_threads: Optional[int] = typer.Option(None, "--fastp-threads", help="Override fastp threads only", hidden=True),
    star_threads: Optional[int] = typer.Option(None, "--star-threads", help="Override STAR threads only", hidden=True),
    dedup_threads: Optional[int] = typer.Option(None, "--dedup-threads", help="Override markdup threads only", hidden=True),
    freebayes_threads: Optional[int] = typer.Option(None, "--freebayes-threads", help="Override FreeBayes threads only", hidden=True),
    motif_scanner_backend: str = typer.Option("auto", "--motif-scanner-backend", help="Motif scanner backend: auto, moods, or python"),
    alpha: str = typer.Option(
        DEFAULT_ALPHA_SPEC, "--alpha",
        help="The cut a next stage selects by (motif peak sets, the phospho pairing test, Evidome's gene selection), P or Q and a value: P0.05 (default) or Q0.10. Paralome writes no call of significance; the viewer applies the user's thresholds",
    ),
    rna_statistic: Optional[str] = typer.Option(
        None,
        "--statistic",
        help=(
            STATISTIC_HELP + ". Single-cell pipelines: :cell after a statistic runs that "
            "statistic with the cells as N (t_cell, paired_t_cell, lm_cell, anova_cell), so "
            "anova,anova:cell runs both levels; pseudobulk is the default and has no term. Bulk RNA "
            "and bulk ATAC take no colon."
        ),
    ),
    rna_factor: List[str] = typer.Option([], "--factor", help=FACTOR_HELP + ". On a single-cell run the column is read from the pseudobulk sample manifest, which carries any cell-metadata column constant within a group"),
    rna_pair_by: Optional[str] = typer.Option(None, "--pair-by", help="paired-t: the sample-sheet column whose shared value makes two samples a pair (one per level), e.g. the subject. On a single-cell run it is read from the pseudobulk sample manifest"),
    rna_covariate: Optional[str] = typer.Option(None, "--covariate", help="lm: a numeric sample-sheet column to regress against"),
    rna_covariate_feature: Optional[str] = typer.Option(None, "--covariate-feature", help="lm: another feature to regress every feature against, e.g. --covariate-feature SOX2"),
    rna_strandedness: Optional[str] = typer.Option(None, "--rna-strandedness", help="0 unstranded, 1 first-strand, 2 second-strand, or 'auto' (the default) to detect it from a gene panel checked against the largest BAM. An undetectable library resolves to 0, with the reasoning printed and recorded."),
    stage_mem: List[str] = typer.Option([], "--stage-mem", help="Repeatable stage-specific memory override, e.g. alignment=40G"),
    rna_pseudocount: float = typer.Option(1.0, "--rna-pseudocount", min=0.0, help="Added to both group means before the fold change is taken; must be greater than 0. Default 1.0"),
    rna_block_size: int = typer.Option(5000, "--rna-block-size", min=1),
    rna_min_expression: Optional[str] = typer.Option(None, "--rna-min-expression", help="Override configured RNA low-expression gate minimum (raw FPKM); config default is 5.0."),
    rna_min_expression_difference: Optional[str] = typer.Option(None, "--rna-min-expression-difference", help="Override configured RNA between-group minimum difference gate (raw FPKM); config default is 5.0."),
    rna_refine_responders: Optional[int] = typer.Option(
        None, "--refine-responders", min=2,
        help=(
            "Also re-run the real pairwise RNA differential test on a trimmed "
            "population, per comparison --factor defines: the N treated "
            "animals with the highest leave-one-out responder score and the N "
            "control animals closest to the control mean on that same axis. "
            "Written to results/qc/refined_top<N>/<comparison>/, beside and "
            "never replacing the full-cohort result. Requires --factor."
        ),
    ),
    fastp_adapter_fasta: Optional[Path] = typer.Option(None, "--fastp-adapter-fasta", "--adapter-fasta", help="Adapter FASTA to trim against. Defaults to what `setup site --adapter-fasta` recorded; without one, fastp guesses the adapter from the reads."),
    fastp_cut_right: bool = typer.Option(True, "--fastp-cut-right/--no-fastp-cut-right", "--cut-right/--no-cut-right", help="Sliding-window quality trim, 5' to 3', cutting the first failing window and everything after it. Trimmomatic's SLIDINGWINDOW. On by default."),
    fastp_cut_right_window_size: Optional[int] = typer.Option(None, "--fastp-cut-right-window-size", "--cut-right-window-size", help="Window for --cut-right; default 4 (SLIDINGWINDOW:4:15)"),
    fastp_cut_right_mean_quality: Optional[int] = typer.Option(None, "--fastp-cut-right-mean-quality", "--cut-right-mean-quality", help="Mean quality a --cut-right window must reach; default 15"),
    fastp_length_required: Optional[int] = typer.Option(None, "--fastp-length-required", "--length-required", help="Discard reads shorter than this AFTER trimming. Default 50. fastp's own default is 15, which keeps reads too short to align and hides them as STAR 'unmapped: too short'."),
    fastp_extra: Optional[str] = typer.Option(None, "--fastp-extra", help="Additional fastp options as a quoted string"),
    star_read_files_command: Optional[str] = typer.Option(None, "--star-read-files-command", help="STAR --readFilesCommand; default zcat"),
    star_out_reads_unmapped: Optional[str] = typer.Option(None, "--star-out-reads-unmapped", help="STAR --outReadsUnmapped; default None"),
    star_out_filter_multimap_nmax: Optional[int] = typer.Option(None, "--star-out-filter-multimap-nmax", help="STAR --outFilterMultimapNmax; default 2"),
    star_out_filter_mismatch_nover_read_lmax: Optional[float] = typer.Option(None, "--star-out-filter-mismatch-nover-read-lmax", help="Mismatches allowed, as a FRACTION OF READ LENGTH. Defaults: 0.06 for RNA, 0.04 for DNA/ATAC. The only mismatch limit: STAR's absolute cap is switched off. For paired reads, of both mates' length together. STAR --outFilterMismatchNoverReadLmax"),
    star_out_filter_score_min_over_lread: Optional[float] = typer.Option(None, "--star-out-filter-score-min-over-lread", help="Minimum alignment score as a fraction of read length; default 0.66. With the option below, this is what decides whether a read is reported 'unmapped: too short'. STAR --outFilterScoreMinOverLread"),
    star_out_filter_match_nmin_over_lread: Optional[float] = typer.Option(None, "--star-out-filter-match-nmin-over-lread", help="Minimum matched bases as a fraction of read length; default 0.66. STAR --outFilterMatchNminOverLread"),
    star_loose: bool = typer.Option(False, "--loose", help="Open every STAR output filter at once, for a divergent reference or a deliberately cheap library: mismatch fraction 0.3, score and match minima 0.3. EXPECT WORSE ALIGNMENTS, not more real ones -- on a 92bp sheep library this raised unique mapping 11%->29% while average aligned length fell from 87 to 65 bases. A diagnostic, not a default. Any filter you set explicitly still wins."),
    star_extra: Optional[str] = typer.Option(None, "--star-extra", "--extra-star-options", help="Additional STAR options as a quoted string"),
    aligner: str = typer.Option("star", "--aligner", help="The read aligner: star (default) or bwa (bwa-mem2 against the genome's FASTA; its index is built on the first BWA run). The --star-* options and --loose apply to STAR only"),
    bwa_extra: Optional[str] = typer.Option(None, "--bwa-extra", help="Additional bwa-mem2 mem options as a quoted string (--aligner bwa)"),
    jobs: Optional[int] = typer.Option(None, "--jobs", help="How many samples run at once in per-sample stages (read QC, alignment, duplicate marking); default all of them, never more than --threads. Variant calling and annotation are split by genome region instead, and keep --threads CPUs busy whatever this is."),
    regions: Optional[Path] = typer.Option(None, "--regions", help="FreeBayes region file when variants are the next stage"),
    max_region_length: Optional[int] = typer.Option(
        None, "--max-region-length", min=0,
        help="Variant calling: the longest region, in bases, one FreeBayes call is given; a longer one is cut into "
             "pieces that meet end to start. Default 100,000 (or variants.max_region_length in the pipeline "
             "configuration); 0 = no cap. It says how the calls are run, so changing it does not make a finished "
             "variant calling out of date; recorded in the FreeBayes summary.",
    ),
    stall_minutes: Optional[float] = typer.Option(
        None, "--stall-minutes", min=0,
        help="Variant calling: minutes a FreeBayes call may write nothing before it is stopped and its region "
             "finished in 2 kb windows. Default 30 (or variants.stall_minutes in the pipeline configuration); 0 "
             "turns the watchdog off. Recorded in the FreeBayes summary.",
    ),
    gene_padding: Optional[int] = typer.Option(
        None, "--gene-padding", min=0,
        help="Bases added to both ends of every GTF gene when variant calling is scheduled from the genome's genes "
             "(no --regions). Default 100. Pad wide on genome data to keep promoter and other flanking "
             "regulatory variants; they are reported whatever their coding effect. Ignored with --regions.",
    ),
    # Every threshold the run applies is one the user can see and set. FreeBayes's calling
    # thresholds, and the QUAL below which a call is left out of the VCF; the filtering stage's quality rules
    # follow the same values. No mapping- or base-quality cut is applied.
    min_alternate_count: int = typer.Option(2, "--min-alternate-count", min=1, help="Reads a sample needs showing an allele before FreeBayes considers it, and the variant reads a genotype needs at filtering. Default 2. FreeBayes --min-alternate-count"),
    min_alternate_fraction: float = typer.Option(0.05, "--min-alternate-fraction", min=0.0, max=1.0, help="Fraction of a sample's reads that must show an allele before FreeBayes considers it, and the variant-read fraction a genotype needs at filtering. Default 0.05. FreeBayes --min-alternate-fraction"),
    haplotype_length: int = typer.Option(3, "--haplotype-length", help="Variants this many matching bases apart or fewer on the same reads are called as one allele. Default 3; -1 calls each separately. FreeBayes --haplotype-length"),
    min_qual: float = typer.Option(20.0, "--min-qual", min=0.0, help="Calls below this QUAL are left out of the VCF; the filtering stage uses it too. Default 20"),
    min_depth: Optional[int] = typer.Option(None, "--min-depth", min=0, help="Filtering: genotype depth at least this. No depth cut unless given"),
    min_gq: Optional[float] = typer.Option(None, "--min-gq", min=0.0, help="Filtering: genotype quality (GQ) at least this. No GQ cut unless given"),
    annotation: Optional[str] = typer.Option(None, "--annotation", help="Standalone annotation engine; default from pipeline config"),
    # NOT AN OPTION. The command name IS the profile, bound by the assay callback, so nothing a user
    # types can override it (a --profile would otherwise let `paralome scRNA --profile DNA` run DNA).
    #
    # `status` still declares --profile and keeps it: narrowing a survey of
    # finished work is not choosing what to run.
    profile: Optional[str] = None,
    segregation_profile: Optional[str] = typer.Option(None, "--segregation-profile", help="Variant-only segregation/filter profile; default family-segregating. Use unrelated-probands for independent affected people with no relatives: every rare variant any of them carries, coding or not."),
    genome: Optional[str] = typer.Option(None, "--genome", help="Configured genome name used for annotation resources and genome-specific annotation settings"),
    annovar_dir: Optional[Path] = typer.Option(None, "--annovar-dir", help="ANNOVAR installation directory"),
    annovar_db: Optional[Path] = typer.Option(None, "--annovar-db", help="ANNOVAR humandb directory"),
    annovar_buildver: Optional[str] = typer.Option(None, "--annovar-buildver", help="ANNOVAR build version, e.g. hg38. Default: annovar_buildver in the genome entry named by --genome; no configuration file sets it"),
    annovar_protocol: Optional[str] = typer.Option(None, "--annovar-protocol", help="ANNOVAR protocol list, given with --annovar-operation. Default: annotation.annovar_protocol in your configuration, else the genome entry"),
    annovar_operation: Optional[str] = typer.Option(None, "--annovar-operation", help="ANNOVAR operation list, one entry per protocol, given with --annovar-protocol. Default: annotation.annovar_operation in your configuration, else the genome entry"),
    snpeff_jar: Optional[Path] = typer.Option(None, "--snpeff-jar", help="Path to snpEff.jar"),
    snpeff_config: Optional[Path] = typer.Option(None, "--snpeff-config", help="Path to snpEff config"),
    snpeff_genome: Optional[str] = typer.Option(None, "--snpeff-genome", help="SnpEff genome database name. Default: snpeff_genome in the genome entry named by --genome; no configuration file sets it"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    # DNA and ATAC always mark duplicates, so --no-mark-duplicates is refused before the run.
    # --mark-duplicates asks for what happens anyway and stays accepted, so a script that carries it
    # still runs. Hidden: there is nothing to choose.
    mark_duplicates: Optional[bool] = typer.Option(
        None, "--mark-duplicates/--no-mark-duplicates", hidden=True,
        help="Duplicates are always marked; --mark-duplicates changes nothing and --no-mark-duplicates is refused",
    ),
    # No default of its own, so defaults.remove_duplicates in the
    # configuration is reached when the flag is not given.
    remove_duplicates: Optional[bool] = typer.Option(None, "--remove-duplicates", help="Remove duplicates instead of only marking them. Default: defaults.remove_duplicates in your configuration, else off"),
    # See paralome.stream. One task per sample for FASTQ QC,
    # alignment and duplicate marking, with the trimmed reads streamed from
    # fastp into STAR through FIFOs and STAR's unsorted output streamed into
    # markdup. The trimmed FASTQs and the aligned BAM are never written, and
    # STAR's coordinate sort is not performed (markdup sorts anyway).
    fuse_stages: Optional[bool] = typer.Option(None, "--fuse-stages/--separate-stages", help="Run FASTQ QC, alignment and duplicate marking as one task per sample, streaming between them: the trimmed FASTQs and the aligned BAM are never written. fastp's QC reports are kept. A task that dies re-trims. THE DEFAULT. --separate-stages writes every intermediate instead, for looking at QC or mapping directly. Default: defaults.fuse_stages in your configuration, else on"),
    fused_buffer: Optional[str] = typer.Option(None, "--fused-buffer", help="With --fuse-stages: RAM each of the two read streams may buffer between fastp and STAR, e.g. 8G. A kernel pipe is 64 KB and the two streams can drift apart by far more, which deadlocks them. Added to the alignment stage's memory request automatically. Default: defaults.fused_buffer in your configuration, else 8G"),
    skip_motif: bool = typer.Option(False, "--skip-motif", help="Skip TF motif finding entirely (bulk ATAC motif_analysis / snATAC singlecell_motif_analysis stage)"),
    skip_motif_pairs: bool = typer.Option(False, "--skip-motif-pairs/--include-motif-pairs", help="Run motif differential stats but skip the combinatorial motif-pair co-occurrence analysis (same flag as `atac motifs --skip-motif-pairs`, now honored by the pipeline stage too)"),
    skip_evidence: bool = typer.Option(False, "--skip-evidence", help="Skip Evidome (the functional_evidence stage) for this run. Evidome already self-skips when it or its data for this species is not installed; this is the deliberate off switch, for when it IS installed and you do not want it"),
    # The trajectories stage (scRNA). The root is required, as Column:ID like the other choices; the
    # default is 5 bins; the stage runs by default, with an opt-out.
    trajectory_root: Optional[str] = typer.Option(None, "--trajectory-root", help="scRNA: <column>:<value>, the cell-metadata column whose clusters the lineages are fitted on and the cluster they start from, e.g. celltype:RPC. Required when the run reaches the trajectories stage"),
    trajectory_bins: int = typer.Option(5, "--trajectory-bins", min=2, help="scRNA: segments per lineage for its composition, each an equal share of its cells by pseudotime; each replicate's share of the lineage's cells in each is tested by the run's statistics"),
    # "install LS as default and monocle's as a
    # switch"; the spline's df "Inside the engine option".
    trajectory_model: Optional[str] = typer.Option(None, "--trajectory-model", help="scRNA: how each gene along each lineage's pseudotime is tested by the run's factors -- the cells' expression as the full factorial of the --factor columns crossed with a natural spline of pseudotime: ls (the default; least squares on each cell's log2(CPM+1), F tests) or monocle3 (Monocle3's fit_models, negative binomial on the counts, its likelihood-ratio tests); the spline's df after a colon, e.g. ls:df=4 (default 3; 1 is a straight line)"),
    # fate the default.
    trajectory_lineages: Optional[str] = typer.Option(None, "--trajectory-lineages", help="scRNA: the lineages the statistics follow: fate (the default; one per end state, every route ending in the same cluster together), exit (one per way out of the root's cluster and end state) or route (every route the fit made)"),
    # --trajectory-fit-cells 100:random samples 100 cells per cluster at a random draw, versus
    # --trajectory-fit-cells 100:centroid.
    trajectory_fit_cells: Optional[str] = typer.Option(None, "--trajectory-fit-cells", help="scRNA: how the lineages are fitted, as key=value pairs separated by commas. Monocle3 (the default --trajectory-method): reduction=<name> (the 2D/3D UMAP the graph is learned in; default the object's UMAP), ncenter=<n> (the graph's nodes), minimal_branch_len=<n> (the shortest branch kept; Monocle3 default 10), use_partition=y|n (a graph per partition of the UMAP, Monocle3's default, or one across every cell), close_loop=y|n (join ends of the graph that come close; default n, a tree), geodesic_distance_ratio=<x> and euclidean_distance_ratio=<x> (with close_loop=y, how far apart two ends may be and still be joined; Monocle3 defaults 1/3 and 1), orthogonal_proj_tip=y|n (extend each leaf out to the cells beyond it; default n), prune_graph=y|n (trim short branches; default y), l1_gamma=<x> (how hard the graph is pulled short and smooth; default 0.5), l1_sigma=<x> (how local each node's pull on the cells is; default 0.01). Slingshot: <n>:random or <n>:centroid (n cells of each cluster, drawn at random or the n nearest its centre), or cells=<n>, dispersion=random|centroid, stretch=<s> (how far a curve runs past its last cluster; Slingshot default 2), extend=y|n|pc1 (how its ends are drawn; default y), reduction=<name> (fit in that reduction, e.g. umap), df=<n> (the curves' spline degrees of freedom, more than 1; 2 is close to straight; lower is stiffer), shrink=<0-1> (how far the lineages' shared start is pulled together; default 1); every cell is then placed on the curves. Default: Monocle3's or Slingshot's own settings, every cell"),
    # 
    trajectory_method: Optional[str] = typer.Option(None, "--trajectory-method", help="scRNA: how the lineages are fitted: monocle3 (the default: one principal graph learned through the cells in the object's UMAP, each path from the root to a leaf of it a lineage) or slingshot (curves through the clusters' centres)"),
    # 
    trajectory_weld: List[str] = typer.Option([], "--trajectory-weld", help="scRNA, Monocle3: join tips of the learned graph, named as its lineages are (trajectories/fit/tips.tsv); repeatable. L39=L42 joins the two tips; L30=L33=L35 joins each to a new node at their middle; L39~L42 joins L39's tip to the nearest point of L42's own branch; 'L39->L42' (quoted: the shell reads > as a redirect) joins L39's tip to L42's, L42's tip staying an end (L39 stopped short of it). Each route from the root to an end is then a lineage (L44, and L44b, L44c ... by the other ways the welds open); where two ways around a loop meet is an end too. Changing only the welds keeps the learned graph"),
    # 
    trajectory_max_routes: Optional[int] = typer.Option(None, "--trajectory-max-routes", help="scRNA, Monocle3: the most routes from the root to the ends the welds may make (default 200); over it the fit stops, saying how many routes cross each weld. Changing it keeps the learned graph"),
    trajectory_dims: Optional[int] = typer.Option(None, "--trajectory-dims", min=2, help="scRNA: fit the lineages on the first n dimensions of the reduction (e.g. the dims the object's UMAP was built from). Default: every dimension"),
    no_trajectories: bool = typer.Option(False, "--no-trajectories", help="scRNA: leave the trajectories stage out of this run (run again later with --trajectory-root to add them)"),
    no_genome_track: bool = typer.Option(False, "--no-genome-track", help="DNA, ATAC and snATAC: skip the genome_tracks stage (DNA: the reads FreeBayes used at every called site, <prefix>_DNA_reads.zip; ATAC: the genome browser's peaks and profiles) for this run, so the run goes straight on to Evidome"),
    stop_at: Optional[str] = typer.Option(None, "--stop-at", help="Run through and stop after this stage (name or friendly alias, e.g. samples, diffs, motif_singles, motifs). Earlier stages are checkpointed/reused automatically; rerun later without --stop-at to continue past it. motif_singles stops after single-motif analysis, before the slow motif-pair co-occurrence step; motifs (or motif_pairs) runs pairs too."),
    max_concurrent_strata: int = typer.Option(
        0, "--max-concurrent-strata", min=0,
        help="scRNA/snATAC: cap simultaneously running stratum tasks. Each stratum becomes its "
             "own SLURM job sized by --threads and --stage-mem; 0 submits them all at once.",
    ),
    poll_seconds: int = typer.Option(60, "--poll-seconds", min=1, help="Seconds between SLURM/checkpoint checks while supervising the complete pipeline"),
    ancestry: str = typer.Option(
        "auto", "--ancestry",
        help="Gene burden: each proband's gnomAD population for the expected carrier rate. auto (the default) "
             "estimates a mixture per proband from the cohort's common coding SNPs; afr, amr, asj, eas, fin, mid, "
             "nfe or sas uses that population for everyone. An `ancestry` column in the sample sheet overrides "
             "it per sample.",
    ),
    # No workbooks; refused by name (PLM-0120).
    report_workbook: Optional[bool] = typer.Option(None, "--workbook/--no-workbook", hidden=True),
    # A bolt-on after gene_burden: the variants of these genes, one file per gene.
    extract_damaging_variants: Optional[str] = typer.Option(
        None, "--extract-damaging-variants",
        help="Gene burden: comma-separated genes (names or Ensembl ids). Runs no pipeline: writes the alleles "
             "--project's gene_burden counted for each gene to "
             "gene_burden/extracted/<prefix>_<GENE>_damaging_variants.tsv, one row per allele and proband with "
             "the burden tests it counted in. No stage is checked, run or made stale.",
    ),
    extract_all_variants: Optional[str] = typer.Option(
        None, "--extract-all-variants",
        help="Gene burden: comma-separated genes (names or Ensembl ids). Runs no pipeline: writes every row of "
             "--project's segregation table that names the gene, as it is, to "
             "gene_burden/extracted/<prefix>_<GENE>_all_variants.tsv. No stage is checked, run or made stale.",
    ),
):
    """Run the complete checkpointed pipeline."""
    # One object, or a folder of objects written with a trailing '/'.
    from paralome.singlecell.rds_input import parse_rds_option

    rds = parse_rds_option(rds)
    if stratify_value and stratify_skip:
        raise typer.BadParameter(
            "--stratify-value names the strata to RUN and --stratify-skip names the "
            "strata to EXCLUDE; pass one or the other, not both."
        )
    _run_common(**locals(), title="Pipeline run", check_complete=False, run_all=True)

#: Where the options that only serve the rebuild path live. A project that
#: has never run has no project_state.json, so its state must be reconstructed
#: from whatever stage outputs exist -- which needs the sample sheet and, if
#: the workdirs are not where they would be derived, each of them by hand.
#: A real case and a rare one, so these are kept off the front page.
_REBUILDING = "Rebuilding state for a project that has never run"


def pipeline_status_command(
    samples: Optional[Path] = typer.Option(None, "--samples", help="Original input sample manifest CSV", rich_help_panel=_REBUILDING),
    project: Optional[Path] = typer.Option(None, "--project", help="Project workdir; stage workdirs are read from this directory"),
    pipeline_config: Optional[Path] = typer.Option(None, "--pipeline-config", help="Optional JSON pipeline defaults config", rich_help_panel=_REBUILDING),
    fastq_qc_workdir: Optional[Path] = typer.Option(None, "--fastq-qc-workdir", help="Override FASTQ QC workdir", rich_help_panel=_REBUILDING),
    align_workdir: Optional[Path] = typer.Option(None, "--align-workdir", help="Override alignment workdir", rich_help_panel=_REBUILDING),
    dedup_workdir: Optional[Path] = typer.Option(None, "--dedup-workdir", help="Override duplicate marking workdir", rich_help_panel=_REBUILDING),
    variant_workdir: Optional[Path] = typer.Option(None, "--variant-workdir", "--workdir", help="Override variant/project workdir", rich_help_panel=_REBUILDING),
    annotation_workdir: Optional[Path] = typer.Option(None, "--annotation-workdir", help="Override standalone annotation workdir", rich_help_panel=_REBUILDING),
    segregation_workdir: Optional[Path] = typer.Option(None, "--segregation-workdir", help="Override segregation workdir", rich_help_panel=_REBUILDING),
    filter_workdir: Optional[Path] = typer.Option(None, "--filter-workdir", help="Override filtering workdir", rich_help_panel=_REBUILDING),
    profile: Optional[str] = typer.Option(None, "--profile", help=_PROFILE_HELP),
    segregation_profile: Optional[str] = typer.Option(None, "--segregation-profile", help="Variant-only segregation/filter profile; default from pipeline config", rich_help_panel=_REBUILDING),
    write: bool = typer.Option(False, "--write", help="Write project_state.json into the selected workdir", rich_help_panel=_REBUILDING),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Write state JSON to this path"),
    detail: bool = typer.Option(False, "--detail", help="Show detailed stage outputs, metrics, and file paths"),
    skip_motif: bool = typer.Option(False, "--skip-motif", help="Report as if TF motif finding was intentionally skipped for this run (match the --skip-motif the run was started with)", rich_help_panel=_REBUILDING),
    # The body passed skip_evidence on without the option existing,
    # so rebuilding a project's state ended in a NameError.
    skip_evidence: bool = typer.Option(False, "--skip-evidence", help="Report as if Evidome was intentionally skipped for this run (match the --skip-evidence the run was started with)", rich_help_panel=_REBUILDING),
    no_genome_track: bool = typer.Option(False, "--no-genome-track", help="Report as if the genome tracks were intentionally skipped for this run (match the --no-genome-track the run was started with)", rich_help_panel=_REBUILDING),
    no_trajectories: bool = typer.Option(False, "--no-trajectories", help="Report as if the trajectories were intentionally skipped for this run (match the --no-trajectories the run was started with)", rich_help_panel=_REBUILDING),
):
    """Where your projects got to. No arguments lists every one of them.

    \b
      paralome status                      every project below here
      paralome status --profile scRNA      only the scRNA ones
      paralome status --project PATH       one project, every stage
      paralome status --project PATH --detail    ...with outputs and metrics

    Paste the path the listing prints straight into --project; that is what it
    is printed for.

    A project appears once it has run at least once, which is when its
    project_state.json is written. To report on one that never has, the state
    has to be rebuilt from whatever stage outputs exist -- that needs --samples
    and the options in the second group.
    """
    from paralome.pipeline.survey import STATE_FILENAME, print_survey, survey

    # BARE `status` SURVEYS, and --project alone is enough: the state is read from
    # project_state.json, not rebuilt from inputs the user has to supply.
    if project is None and samples is None:
        # NOTE: this runs before `profile` is defaulted from the pipeline
        # config below, so None here genuinely means the user did not ask for
        # one -- a bare `status` lists everything rather than silently
        # narrowing to the config's default of DNA.
        root = Path.cwd()
        print_survey(root, survey(root, profile), profile)
        return

    if project is not None and samples is None:
        recorded = Path(project) / STATE_FILENAME
        if recorded.is_file():
            state = json.loads(recorded.read_text(encoding="utf-8"))
            print_pipeline_dashboard(state, detail=detail)
            # This is the state the last run recorded, not the folders as they are now (a stage folder
            # moved aside still shows that stage complete).
            from datetime import datetime as _dt

            written = _dt.fromtimestamp(recorded.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"\nAs recorded by the last run ({recorded}, {written}). "
                  "To read the stage folders as they are now, add --samples <sheet>.")
            if json_out:
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
                print(f"\nProject state JSON  : {json_out}")
            return
        raise typer.BadParameter(
            f"{project} has no {STATE_FILENAME}, so there is nothing recorded "
            f"to report.\nThat file is written once a run has started. To build "
            f"the state from the stage outputs instead, pass --samples."
        )

    cfg = _load_pipeline_config(pipeline_config)
    if profile is None:
        profile = str(cfg_get(cfg, "pipeline", "profile", "DNA"))
    # Paralome's default profile, when nobody set one, is fitted to
    # the sheet as `pipeline run` fits it, so status reads the same filtering.
    defaulted = segregation_profile is None and not any(
        _configured_setting(section, key, pipeline_config)[1]
        for section, key in (("filtering", "segregation_profile"), ("filtering", "profile"),
                             ("defaults", "filter_profile"))
    )
    if segregation_profile is None:
        segregation_profile = cfg_get(
            cfg,
            "filtering",
            "segregation_profile",
            cfg_get(cfg, "filtering", "profile", None),
        )
    if defaulted and samples is not None and resolve_pipeline_profile(profile).is_variant:
        from paralome.filters.table import applicable_pipeline_profile

        segregation_profile = applicable_pipeline_profile(samples, segregation_profile or "family-segregating")
    if project is not None:
        if samples is None:
            raise typer.BadParameter("--samples is required when --project is used")
        workdirs = derive_project_workdirs(project, samples)
        fastq_qc_workdir = fastq_qc_workdir or workdirs.fastq_qc
        align_workdir = align_workdir or workdirs.alignment
        dedup_workdir = dedup_workdir or workdirs.deduplication
        variant_workdir = variant_workdir or workdirs.variants
        annotation_workdir = annotation_workdir or workdirs.annotation
        segregation_workdir = segregation_workdir or workdirs.segregation
        filter_workdir = filter_workdir or workdirs.filtering
    state = build_project_state(
        samples=samples,
        fastq_qc_workdir=fastq_qc_workdir,
        align_workdir=align_workdir,
        dedup_workdir=dedup_workdir,
        variant_workdir=variant_workdir,
        annotation_workdir=annotation_workdir,
        segregation_workdir=segregation_workdir,
        filter_workdir=filter_workdir,
        report_workdir=None,
        profile=profile,
        segregation_profile=segregation_profile,
        skip_motif=skip_motif,
        skip_evidence=skip_evidence,
        no_genome_track=no_genome_track,
        no_trajectories=no_trajectories,
    )

    out_path = None
    if write:
        target = filter_workdir or segregation_workdir or annotation_workdir or variant_workdir or dedup_workdir or align_workdir or fastq_qc_workdir or project or Path(".")
        out_path = write_project_state(target, state)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        out_path = json_out

    print_pipeline_dashboard(state, detail=detail)
    if out_path:
        print(f"\nProject state JSON  : {out_path}")
