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

from typing import Any, Iterable, Mapping, Optional

from paralome.errors import ParalomeError

CONFIG_VERSION = 1
REQUIRED_TOP_LEVEL_KEYS = ("config_version", "defaults", "executables", "genomes", "profiles")

DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "defaults": {
        "backend": "slurm",
        "threads": 8,
        "jobs": 0,
        "annotation": "annovar",
        "filter_profile": "family-segregating",
        "mark_duplicates": True,
        "remove_duplicates": False,
    },
    "executables": {},
    "genomes": {},
    "profiles": {},
    "stage_resources": {},
    "rna": {
        # "auto", not 0: a shipped config pinning 0 would override the pipeline default for every
        # account that has one, silently, since a config value beating a default is exactly what a
        # config is for.
        "strandedness": "auto",
        # RNA-seq's most "significant" hits are often
        # false positives where both group means are near zero (e.g. a
        # gene going from 0 to 0.15 raw FPKM has an enormous, infinitely
        # signed fold-change but is not biologically meaningful). These
        # give every user a working low-expression/small-effect gate out
        # of the box (a common analysis convention), rather than
        # requiring each new project to discover and set them manually.
        # A gene must reach `min_expression` FPKM in at least one condition
        # AND differ between conditions by at least `min_expression_difference`
        # FPKM to appear on the Significant sheet / per-term stats tabs
        # (never on All_Genes, and never affects the statistical test
        # itself). Set both to 0.0 in the config file to disable.
        "min_expression": 5.0,
        "min_expression_difference": 5.0,
    },
}


class ConfigValidationError(ValueError):
    """Raised when the Paralome user configuration is malformed."""


class UnknownConfigKeyError(ParalomeError, ConfigValidationError):
    """A key no part of Paralome reads. A typo, almost always.

    A ParalomeError as well, so every command reports it with a number and the
    file's name rather than as an internal error; and still a
    ConfigValidationError for the code that catches those.
    """

    def __init__(self, detail: str) -> None:
        ParalomeError.__init__(
            self, 411, detail,
            fix="Correct or remove that key in the file named above",
        )


class MalformedConfigError(ParalomeError, ConfigValidationError):
    """A configuration file that is not JSON, or not a JSON object.

    A ParalomeError as well, like the class above, so a stray comma in the file
    is reported under a number naming the file, from whichever command reads
    it first.
    """

    def __init__(self, detail: str) -> None:
        ParalomeError.__init__(
            self, 412, detail,
            fix="Correct the JSON in the file named above; `python3 -m json.tool FILE` shows where it breaks",
        )


# ---------------------------------------------------------------------------
# THE KEYS PARALOME READS, so that one it does not read is refused.
#
# A key that matches nothing is refused, because a silently ignored key will
# ultimately bite the user: a mistyped key -- `stage_resources.alignmnt`,
# `defaults.thraeds` -- would otherwise validate, load and be ignored, and the
# run would go ahead on the default it was written to replace.
#
# Every name below is read or written somewhere in the package. Maps whose
# names the user chooses (genomes, executables, profiles, declined) are checked
# one level down, never at the name.
# ---------------------------------------------------------------------------

#: Sections of the pipeline schema: the packaged defaults, a --pipeline-config
#: file, and the same sections when they appear in the user configuration.
PIPELINE_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "annotation": (
        "engine", "annovar_buildver", "annovar_operation", "annovar_protocol",
        "snpeff_genome",
    ),
    # `profile` is the older spelling of segregation_profile and is still read.
    "filtering": ("segregation_profile", "profile"),
    "pipeline": ("profile",),
    "resources": ("backend", "jobs", "mark_duplicates", "remove_duplicates", "threads"),
    "rna": ("strandedness", "min_expression", "min_expression_difference"),
    "fastq_qc": ("adapter_fasta",),
    # How FreeBayes calls are run (--max-region-length, --stall-minutes).
    "variants": ("max_region_length", "stall_minutes"),
}

#: `defaults` in the user configuration. `genome` and `reference_genome` are
#: accepted and not used: they named a default genome, and there is none. A run
#: without --genome says so by name rather than the file failing to load for
#: every other command.
DEFAULTS_KEYS: tuple[str, ...] = (
    "backend", "threads", "jobs", "annotation", "filter_profile",
    "mark_duplicates", "remove_duplicates", "deployment",
    "genome", "reference_genome",
)

#: Top-level keys of the user configuration that are not pipeline sections.
USER_ONLY_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "config_version", "defaults", "executables", "genomes", "profiles",
    "stage_resources", "declined", "site_registry", "genome_repository",
    # The defaults Paralome wrote into the file (config.user).
    "paralome_defaults",
)

#: What `paralome setup` records for each tool.
EXECUTABLE_KEYS: tuple[str, ...] = ("path", "found", "version")

#: What a stage's resources may set. Only memory is read.
STAGE_RESOURCE_KEYS: tuple[str, ...] = ("mem",)

#: A genome entry's own facts, beside its resource keys and its bookkeeping.
GENOME_IDENTITY_KEYS: tuple[str, ...] = (
    "name", "species", "assembly", "annotation_source", "annotation_release",
    "modified", "modified_note",
)


def _nearest(key: str, known: Iterable[str]) -> str:
    import difflib

    close = difflib.get_close_matches(str(key), sorted(known), n=1, cutoff=0.5)
    return f" Did you mean {close[0]!r}?" if close else ""


def _refuse_unknown(
    found: Iterable[Any], known: Iterable[str], *, where: str, source: str,
    fold_case: bool = False,
) -> None:
    known = list(known)
    accepted = {name.lower() for name in known} if fold_case else set(known)
    for key in found:
        text = str(key)
        if (text.lower() if fold_case else text) in accepted:
            continue
        dotted = f"{where}.{text}" if where else text
        raise UnknownConfigKeyError(
            f"{source}: unknown key {dotted!r}.{_nearest(text, known)} "
            f"Nothing reads that key, so the setting would be ignored. "
            f"Valid here: {', '.join(sorted(known))}."
        )


def stage_resource_names() -> tuple[str, ...]:
    """Every stage `stage_resources` may name: all pipelines' stages, plus
    ``variant_calling``, the configuration's own name for ``variants``."""
    from paralome.pipeline.profiles import PROFILE_REGISTRY

    names = ["variant_calling"]
    for spec in PROFILE_REGISTRY.values():
        for stage in spec.stage_graph:
            if stage not in names:
                names.append(stage)
    return tuple(names)


def _genome_entry_keys() -> tuple[str, ...]:
    from paralome.pipeline.resource_catalog import RESOURCE_KEYS
    from paralome.setup.genomes import PROVENANCE_KEYS

    names = list(GENOME_IDENTITY_KEYS) + list(PROVENANCE_KEYS)
    for canonical, aliases in RESOURCE_KEYS.items():
        names.append(canonical)
        names.extend(aliases)
    return tuple(dict.fromkeys(names))


def validate_pipeline_sections(config: Mapping[str, Any], *, source: str) -> None:
    """Refuse a key the pipeline schema does not have, in the sections given."""
    for section, known in PIPELINE_SECTION_KEYS.items():
        block = config.get(section)
        if isinstance(block, dict):
            _refuse_unknown(block, known, where=section, source=source)
    stages = config.get("stage_resources")
    if isinstance(stages, dict):
        _refuse_unknown(stages, stage_resource_names(), where="stage_resources", source=source)
        for stage, block in stages.items():
            if isinstance(block, dict):
                _refuse_unknown(
                    block, STAGE_RESOURCE_KEYS,
                    where=f"stage_resources.{stage}", source=source,
                )


def validate_pipeline_config_keys(config: Mapping[str, Any], *, source: str) -> None:
    """Every key of a --pipeline-config file is one the pipeline reads."""
    _refuse_unknown(
        config, (*PIPELINE_SECTION_KEYS, "stage_resources"), where="", source=source,
    )
    validate_pipeline_sections(config, source=source)


def validate_user_config_keys(config: Mapping[str, Any], *, source: str) -> None:
    """Every key of the user configuration is one Paralome reads or writes."""
    _refuse_unknown(
        config, (*USER_ONLY_TOP_LEVEL_KEYS, *PIPELINE_SECTION_KEYS),
        where="", source=source,
    )
    defaults = config.get("defaults")
    if isinstance(defaults, dict):
        _refuse_unknown(defaults, DEFAULTS_KEYS, where="defaults", source=source)
    executables = config.get("executables")
    if isinstance(executables, dict):
        for tool, entry in executables.items():
            if isinstance(entry, dict):
                _refuse_unknown(
                    entry, EXECUTABLE_KEYS, where=f"executables.{tool}", source=source,
                )
    genomes = config.get("genomes")
    if isinstance(genomes, dict):
        known = _genome_entry_keys()
        for name, entry in genomes.items():
            if not isinstance(entry, dict):
                continue
            # Only keys holding a plain value. A key holding an object is an
            # older entry's grouping of resource keys, which the resolver still
            # walks into, or a record such as `withdrawn`.
            plain = [key for key, value in entry.items() if not isinstance(value, (dict, list))]
            _refuse_unknown(
                plain, known, where=f"genomes.{name}", source=source, fold_case=True,
            )
    validate_pipeline_sections(config, source=source)


def default_config() -> dict[str, Any]:
    """Return a fresh copy of the default user configuration."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


def validate_config(config: dict[str, Any], *, source: Optional[str] = None) -> None:
    """Validate the minimal Paralome config schema.

    This intentionally stays lightweight for the bootstrap layer. Later setup
    and doctor commands can do deeper validation of executables and genome
    bundles without making config loading fragile.

    ``source`` is the file being validated, for the message about a key
    nothing reads.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Configuration root must be a JSON object")

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ConfigValidationError("Configuration is missing required keys: " + ", ".join(missing))

    version = config.get("config_version")
    if version != CONFIG_VERSION:
        raise ConfigValidationError(
            f"Unsupported config_version {version!r}; expected {CONFIG_VERSION}"
        )

    for key in ("defaults", "executables", "genomes", "profiles"):
        if not isinstance(config.get(key), dict):
            raise ConfigValidationError(f"Configuration key '{key}' must be an object")
    for key in ("stage_resources", "rna"):
        if key in config and not isinstance(config.get(key), dict):
            raise ConfigValidationError(f"Configuration key '{key}' must be an object")
    rna_config = config.get("rna") or {}
    # Accepts the three numbers and "auto". Numbers stay numbers so an existing
    # config keeps validating unchanged; the comparison is on the string form so
    # 1 and "1" are the same answer to the same question.
    strandedness = rna_config.get("strandedness", "auto")
    if str(strandedness).strip().lower() not in {"0", "1", "2", "auto"}:
        raise ConfigValidationError(
            "rna.strandedness must be 0, 1, 2, or 'auto'"
        )
    for key in ("min_expression", "min_expression_difference"):
        if key not in rna_config:
            continue
        value = rna_config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigValidationError(f"rna.{key} must be a number >= 0")
    validate_user_config_keys(config, source=source or "The Paralome configuration")
