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

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from paralome.config.user import config_path, load_config


DEFAULT_GENOME_ALIASES = ("human_grch38", "GRCh38", "grch38", "hg38")

#: Which resolved resources name something on disk, and what kind. The rest of
#: RESOURCE_KEYS are values -- a build name, a taxonomy id, a comma-separated
#: protocol list -- and checking those against the filesystem would be nonsense.
PATH_RESOURCES: dict[str, str] = {
    "reference_fasta": "file",
    "gtf": "file",
    "motif_database": "file",
    "snpeff_jar": "file",
    "snpeff_config": "file",
    "star_index": "directory",
    "annovar_dir": "directory",
    "annovar_db": "directory",
    # Built by `setup genome burden-reference`, read by gene_burden.
    "burden_reference": "file",
}


def _require_the_files_to_be_there(
    genome_name: Optional[str],
    required: Iterable[str],
    resources: Mapping[str, "ResolvedResource"],
) -> None:
    """A resolved path has to exist and be readable, not merely be configured.

    ``missing`` above is computed as ``name not in resources`` -- presence of a
    KEY in the config. A key whose value points at a file that was deleted,
    moved or renamed would satisfy it completely, and the resolver would hand
    the caller a path to nothing. So existence and readability are checked
    here, as the doctor's ``_check_path`` checks them, and the run-time path
    is as strict as the doctor that certifies it: learning on a compute node
    that a genome file is gone is the most expensive place to learn it.

    Readability is separate from existence on purpose: a genome folder owned by
    a colleague at mode 600 is a chmod by them, not a re-registration by you,
    and PLM-0105 says so rather than sending someone to rebuild an index that
    is sitting right there.
    """
    from paralome.errors import ParalomeError
    from paralome.setup.genomes import resource_state

    for name in required:
        kind = PATH_RESOURCES.get(name)
        if kind is None:
            continue
        resource = resources.get(name)
        if resource is None:
            continue
        state = resource_state(resource.value, kind)
        if state == "ok":
            continue
        where = f"{name} for genome {genome_name}" if genome_name else name
        if state == "unreadable":
            raise ParalomeError(
                105,
                f"{where}: {resource.value}\n"
                "It is there, but this account cannot read it. Its owner needs "
                "to make it readable; re-registering the genome will not help.",
            )
        raise ParalomeError(
            104,
            f"{where}: {resource.value}\n"
            "The genome entry names this path and nothing is at it. Check the "
            "entry with `paralome setup doctor <genome>`.",
        )

RESOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "reference_fasta": (
        "reference_fasta",
        "reference",
        "fasta",
        "genome_fasta",
    ),
    "gtf": (
        "gtf",
        "annotation_gtf",
        "genes_gtf",
    ),
    "motif_database": (
        "motif_database",
        "motifs",
        "jaspar_meme",
        "jaspar_database",
    ),
    "motif_species": (
        "motif_species",
        "species_common_name",
        "species",
    ),
    "taxonomy_id": (
        "taxonomy_id",
        "taxon_id",
        "ncbi_taxonomy_id",
    ),
    "star_index": (
        "star_index",
        "star_genome",
        "star_genome_dir",
    ),
    "annovar_dir": (
        "annovar_dir",
        "annovar_home",
        "annovar_install",
        "annovar_install_dir",
        "annovar",
    ),
    "annovar_db": (
        "annovar_db",
        "annovar_db_dir",
        "annovar_humandb",
        "humandb",
        "humandb_dir",
    ),
    "snpeff_jar": (
        "snpeff_jar",
        "snpeff",
        "snpEff_jar",
        "snpeff_path",
        "snpeff_jar_path",
    ),
    "snpeff_config": (
        "snpeff_config",
        "snpEff_config",
        "snpeff_config_path",
        "snpeff_cfg",
    ),
    "snpeff_genome": (
        "snpeff_genome",
        "snpEff_genome",
        "snpeff_database",
        "snpeff_db",
    ),
    "annovar_buildver": (
        "annovar_buildver",
        "buildver",
        "annovar_build",
    ),
    "annovar_protocol": (
        "annovar_protocol",
        "protocol",
        "annovar_protocols",
    ),
    "annovar_operation": (
        "annovar_operation",
        "operation",
        "annovar_operations",
    ),
    # The per-gene gnomAD expected table (burden/reference.py).
    # No aliases: Paralome writes it, nobody hand-spells it.
    "burden_reference": (
        "burden_reference",
    ),
    # Ensembl's UniProt cross-references, found in the genome folder by `setup genome add`
    # (setup/genome_scan.find_uniprot_table); read by the Olink arm's protein identity.
    "uniprot_table": (
        "uniprot_table",
    ),
}


class ResourceResolutionError(ValueError):
    """Raised when configured pipeline resources cannot be resolved."""


@dataclass(frozen=True)
class ResolvedResource:
    name: str
    value: str
    source: str
    config_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedGenomeResources:
    genome: str
    config_file: str
    resources: Mapping[str, ResolvedResource]

    def value(self, name: str) -> Optional[str]:
        resource = self.resources.get(name)
        return resource.value if resource is not None else None

    def path(self, name: str) -> Optional[Path]:
        value = self.value(name)
        return Path(value).expanduser() if value else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome,
            "config_file": self.config_file,
            "resources": {
                name: resource.to_dict()
                for name, resource in sorted(self.resources.items())
            },
        }


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _find_value(config: Mapping[str, Any], aliases: tuple[str, ...]) -> tuple[Optional[str], Optional[str]]:
    accepted = {alias.lower() for alias in aliases}
    for block in _iter_dicts(config):
        for key, value in block.items():
            if str(key).lower() in accepted and value not in (None, ""):
                return str(value), str(key)
    return None, None


def retired_default_genome_note(config: Mapping[str, Any]) -> str:
    """A sentence for a configuration that still names a default genome.

    ``defaults.genome`` (or ``reference_genome``) is accepted in the file and
    not used, so the refusal for a missing --genome says so rather than
    leaving the setting to be ignored without a word.
    """
    defaults = config.get("defaults") if isinstance(config, Mapping) else None
    if not isinstance(defaults, Mapping):
        return ""
    named = defaults.get("genome") or defaults.get("reference_genome")
    if not named:
        return ""
    return (
        f". defaults.genome ({named}) in your configuration is no longer used: "
        "there is no default genome"
    )


def select_genome_config(
    genome: Optional[str] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> tuple[str, Mapping[str, Any]]:
    cfg = dict(config) if config is not None else load_config(initialize=False)
    genomes = cfg.get("genomes", {})
    if not isinstance(genomes, dict) or not genomes:
        raise ResourceResolutionError(
            f"No genomes are configured in {config_path()}; pass explicit annotation resource paths."
        )

    # THERE IS NO DEFAULT GENOME. A forgotten --genome on a non-human run must not be annotated against
    # human without a word: any default here would mislead the user.
    selected = genome
    if selected:
        selected_text = str(selected)
        entry = genomes.get(selected_text)
        if isinstance(entry, dict):
            return selected_text, entry

        casefold_matches = [
            (str(name), value)
            for name, value in genomes.items()
            if str(name).casefold() == selected_text.casefold() and isinstance(value, dict)
        ]
        if len(casefold_matches) == 1:
            return casefold_matches[0]

        # A NAME A GENOME USED TO HAVE STILL FINDS IT.
        #
        # Names are derived from species, assembly and annotation, and `setup genome rename` moves
        # existing genomes onto them. Every project records which genome produced a result, and every
        # user has commands and scripts with --genome in them, so a rename that stopped the old name
        # resolving would make finished work unreadable. `rename` records what an entry used to be
        # called; this is the half that honours it.
        #
        # Checked AFTER exact and casefold matching, so a former name can
        # never shadow a genome that currently holds that name.
        from paralome.setup.genome_names import former_names

        previous_matches = [
            (str(name), value)
            for name, value in genomes.items()
            if isinstance(value, dict)
            and any(
                str(old).casefold() == selected_text.casefold()
                for old in former_names(value)
            )
        ]
        if len(previous_matches) == 1:
            return previous_matches[0]

        grch38_aliases = {alias.casefold() for alias in DEFAULT_GENOME_ALIASES}
        if selected_text.casefold() in grch38_aliases:
            alias_matches = [
                (str(name), value)
                for name, value in genomes.items()
                if str(name).casefold() in grch38_aliases and isinstance(value, dict)
            ]
            if len(alias_matches) == 1:
                return alias_matches[0]

        raise ResourceResolutionError(
            f"Genome {selected_text!r} is not configured in {config_path()}. "
            f"Available genomes: {', '.join(sorted(str(k) for k in genomes))}"
        )

    raise ResourceResolutionError(
        "--genome is required: Paralome never chooses a genome for you. "
        f"Configured genomes: {', '.join(sorted(str(k) for k in genomes))}"
        + retired_default_genome_note(cfg)
    )


def resolve_genome_resources(
    *,
    genome: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    required: Iterable[str] = (),
    config: Optional[Mapping[str, Any]] = None,
    enforce_withdrawal: bool = True,
) -> ResolvedGenomeResources:
    """Resolve genome resources with one authoritative precedence policy.

    Precedence is explicit non-empty override, then configured genome value.
    Nested legacy config keys remain supported through RESOURCE_KEYS aliases.

    A caller who passes every required resource explicitly needs no genome
    config: the genome config is consulted in exactly one place below
    (``_find_value``), and an empty one simply finds nothing. So genome
    selection failure is deferred rather than raised: if the overrides satisfy
    ``required``, the call proceeds; if they do not, the selection error is
    raised at the bottom, because "no genomes are configured" is a better
    diagnosis than "could not resolve for genome None".

    Deferred ONLY when no genome was named: a mistyped --genome is refused,
    rather than resolving to no paths at all and passing preflight.
    """
    if genome:
        genome_name, genome_config = select_genome_config(genome, config=config)
    else:
        # No genome named: nothing is selected, and the configuration is not
        # even opened unless something below turns out to be missing.
        genome_name, genome_config = None, {}

    # The one chokepoint every pipeline passes through, which is why
    # the withdrawal check lives here rather than in each command. Fatal raises
    # PLM-0407; superseded warns and falls through.
    #
    # ``enforce_withdrawal=False`` is for the doctor alone: its job is to REPORT
    # the state of a genome, and a doctor that dies on the genome it was asked
    # to examine tells you less than one that prints the reason.
    if enforce_withdrawal and genome_name and genome_config:
        from paralome.setup.withdrawal import enforce as _enforce_withdrawal

        _enforce_withdrawal(str(genome_name), genome_config)

    supplied = dict(overrides or {})
    resources: dict[str, ResolvedResource] = {}

    for name, aliases in RESOURCE_KEYS.items():
        explicit = supplied.get(name)
        if explicit not in (None, ""):
            resources[name] = ResolvedResource(
                name=name,
                value=str(explicit),
                source="explicit_override",
            )
            continue

        value, key = _find_value(genome_config, aliases)
        if value is not None:
            resources[name] = ResolvedResource(
                name=name,
                value=value,
                source="genome_config",
                config_key=key,
            )

    # Preserve the historical ANNOVAR convenience rule.
    if "annovar_dir" not in resources and "annovar_db" in resources:
        db = resources["annovar_db"]
        resources["annovar_dir"] = ResolvedResource(
            name="annovar_dir",
            value=str(Path(db.value).expanduser().parent),
            source="derived_from_annovar_db",
            config_key=db.config_key,
        )

    missing = [name for name in required if name not in resources]
    if missing:
        # If there was no genome to consult, say THAT. Reporting
        # "could not resolve for genome None" would describe the symptom of
        # the deferred failure above rather than its cause.
        if not genome:
            # Raises: no genomes configured, or --genome is required with the
            # configured names listed.
            select_genome_config(None, config=config)
        raise ResourceResolutionError(
            f"Could not resolve resources for genome {genome_name!r}: {', '.join(missing)}. "
            "Configure them for that genome or pass the corresponding options explicitly."
        )

    _require_the_files_to_be_there(genome_name, required, resources)

    return ResolvedGenomeResources(
        genome=genome_name,
        config_file=str(config_path()),
        resources=resources,
    )
