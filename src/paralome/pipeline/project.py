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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from paralome.config.user import load_config as load_user_config
from paralome.errors import ParalomeError


DEFAULT_PIPELINE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "pipeline_defaults.json"


@dataclass(frozen=True)
class ProjectWorkdirs:
    project: Path
    fastq_qc: Path
    alignment: Path
    deduplication: Path
    variants: Path
    annotation: Path
    segregation: Path
    filtering: Path
    reporting: Path
    rna_quantification: Path
    differential_expression: Path
    # The DNA gene-burden stage. Beside the others, as the state
    # reader finds it (state.project_stage_root); with a default so a caller
    # that builds this record itself is unchanged.
    gene_burden: Optional[Path] = None
    # The export stages' folders and the omicview folder.
    statistics_output: Optional[Path] = None
    genotype_output: Optional[Path] = None

    def to_dict(self) -> dict[str, str]:
        return {
            "project": str(self.project),
            "fastq_qc": str(self.fastq_qc),
            "alignment": str(self.alignment),
            "deduplication": str(self.deduplication),
            "variants": str(self.variants),
            "annotation": str(self.annotation),
            "segregation": str(self.segregation),
            "filtering": str(self.filtering),
            "reporting": str(self.reporting),
            "rna_quantification": str(self.rna_quantification),
            "differential_expression": str(self.differential_expression),
            "gene_burden": str(self.gene_burden or Path(self.project) / "gene_burden"),
            "statistics_output": str(self.statistics_output or Path(self.project) / "statistics_output"),
            "genotype_output": str(self.genotype_output or Path(self.project) / "genotype_output"),
        }


def derive_project_name(samples: Union[str, Path]) -> str:
    stem = Path(samples).name
    for suffix in (".csv", ".tsv", ".txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for suffix in ("_samples_fastQ", "_samples_fastq", "_samples", ".samples"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "paralome_project"


def derive_project_workdirs(project: str | Optional[Path], samples: Union[str, Path]) -> ProjectWorkdirs:
    root = Path(project) if project else Path(derive_project_name(samples))
    annotation = root / "annotation"
    legacy_annotation = root / "annovar"
    if not annotation.exists() and legacy_annotation.exists():
        annotation = legacy_annotation
    # The alignment stage's folder, whichever aligner runs (STAR or BWA); a project aligned before it was
    # named so keeps its star/ folder, and its BAMs are reused there.
    alignment = root / "alignment"
    if not alignment.exists() and (root / "star").exists():
        alignment = root / "star"
    return ProjectWorkdirs(
        project=root,
        fastq_qc=root / "fastq_qc",
        alignment=alignment,
        deduplication=root / "dedup",
        variants=root / "freebayes",
        annotation=annotation,
        segregation=root / "segregation",
        filtering=root / "filter",
        reporting=root / "candidate_report",
        rna_quantification=root / "rna_quantification",
        differential_expression=root / "differential_expression",
        gene_burden=root / "gene_burden",
        statistics_output=root / "statistics_output",
        genotype_output=root / "genotype_output",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    # Under a number, naming the file; a file holding a JSON list is refused, not read as empty.
    from paralome.errors import ParalomeError

    fix = "Correct the JSON in that file; `python3 -m json.tool FILE` shows where it breaks"
    try:
        text = path.read_text()
    except OSError as exc:
        raise ParalomeError(105, f"Pipeline configuration {path} cannot be read: {exc}.") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ParalomeError(412, f"Pipeline configuration {path} is not valid JSON: {exc}.", fix=fix) from exc
    if not isinstance(data, dict):
        raise ParalomeError(
            412,
            f"Pipeline configuration {path} holds JSON that is not an object "
            f"({type(data).__name__}); none of it would be read.",
            fix=fix,
        )
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _user_pipeline_overlay(user: dict[str, Any]) -> dict[str, Any]:
    """Translate user-config defaults into the pipeline-config schema."""
    overlay: dict[str, Any] = {}
    defaults = user.get("defaults", {})
    if isinstance(defaults, dict):
        resources: dict[str, Any] = {}
        for key in (
            "backend",
            "jobs",
            "mark_duplicates",
            "remove_duplicates",
            "threads",
        ):
            if defaults.get(key) is not None:
                resources[key] = defaults[key]
        if resources:
            overlay["resources"] = resources

        if defaults.get("annotation") is not None:
            overlay.setdefault("annotation", {})["engine"] = defaults["annotation"]
        if defaults.get("filter_profile") is not None:
            overlay.setdefault("filtering", {})["segregation_profile"] = defaults[
                "filter_profile"
            ]

    # Fastq_qc joins them, so fastq_qc.adapter_fasta in the user configuration is read too.
    for section in ("annotation", "filtering", "pipeline", "resources", "stage_resources", "rna", "fastq_qc"):
        value = user.get(section)
        if isinstance(value, dict):
            overlay[section] = _deep_merge(overlay.get(section, {}), value)
    return overlay


def load_pipeline_config(config: str | Optional[Path] = None) -> dict[str, Any]:
    """Load merged pipeline defaults.

    Precedence, lowest to highest: packaged defaults, user config, explicit
    ``--pipeline-config``. Explicit CLI options are resolved later and remain
    authoritative.
    """
    from paralome.config.schema import validate_pipeline_config_keys

    merged = _read_json_object(DEFAULT_PIPELINE_CONFIG)
    try:
        user = load_user_config(initialize=False)
    except FileNotFoundError:
        user = {}
    merged = _deep_merge(merged, _user_pipeline_overlay(user))
    if config is not None:
        # A file the user NAMED must exist. _read_json_object treats a missing file as an empty one, which
        # is right for the packaged defaults and wrong here: a mistyped --pipeline-config would run the
        # whole analysis on defaults without a word.
        if not Path(config).is_file():
            raise ParalomeError(104, f"--pipeline-config {str(config)!r} does not exist.")
        given = _read_json_object(Path(config))
        # And every key in it must be one the pipeline reads. A
        # mistyped section or key was merged in and never looked at.
        validate_pipeline_config_keys(given, source=f"--pipeline-config {config}")
        merged = _deep_merge(merged, given)
    return merged


def cfg_get(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    block = config.get(section, {}) if isinstance(config, dict) else {}
    if not isinstance(block, dict):
        return default
    value = block.get(key, default)
    return default if value is None else value


def stage_resource(config: dict[str, Any], stage: str, key: str, default: Any = None) -> Any:
    """Return a validated stage-specific user/pipeline resource value."""
    blocks = config.get("stage_resources", {}) if isinstance(config, dict) else {}
    if not isinstance(blocks, dict):
        return default
    block = blocks.get(stage, {})
    if not isinstance(block, dict):
        return default
    value = block.get(key, default)
    return default if value is None else value


def choose_path(explicit: str | Optional[Path], fallback: Optional[Path]) -> Optional[Path]:
    return Path(explicit) if explicit is not None else fallback
