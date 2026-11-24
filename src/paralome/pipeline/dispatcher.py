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

from paralome.checkpoints import stage_checkpoint_path, vcf_with_index_ready
from paralome.pipeline.manifest import ManifestInspection, inspect_manifest


@dataclass
class FileStatus:
    path: str
    exists: bool
    nonempty: bool
    size: int

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "FileStatus":
        p = Path(path)
        exists = p.exists()
        size = p.stat().st_size if exists and p.is_file() else 0
        return cls(path=str(p), exists=exists, nonempty=size > 0, size=size)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "exists": self.exists, "nonempty": self.nonempty, "size": self.size}


@dataclass
class PipelineStage:
    name: str
    complete: bool
    status: str
    workdir: Optional[str] = None
    outputs: dict[str, str] | None = None
    details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "complete": self.complete,
            "status": self.status,
            "workdir": self.workdir,
            "outputs": self.outputs or {},
            "details": self.details or {},
        }


@dataclass
class DispatchPlan:
    input_manifest: ManifestInspection
    resolved_manifest: str
    resolved_manifest_role: str
    next_stage: str
    stages: list[PipelineStage]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_manifest": self.input_manifest.to_dict(),
            "resolved_manifest": self.resolved_manifest,
            "resolved_manifest_role": self.resolved_manifest_role,
            "next_stage": self.next_stage,
            "recommended_action": self.recommended_action,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _held_to_the_sheet(name: str, workdir: Path, samples: Optional[Path]) -> dict[str, Any]:
    """The state reader's verdict on a per-sample stage, held against the sheet.

    One rule, pipeline.state's, so the plan and the state cannot
    disagree. The plan took any summary as a finished read QC and any finished
    alignment as covering the sheet; a sample added to the sheet since was then
    missing from the manifest the plan handed to the next stage.
    """
    from paralome.pipeline import state as state_module

    summarize = {"fastq_qc": state_module.summarize_fastq_qc, "alignment": state_module.summarize_alignment}
    held = {name: summarize[name](workdir)}
    if samples is not None:
        # The plan has read this sheet already, so one that cannot
        # be read here is an error and is raised as one.
        state_module._hold_per_sample_stages_to_the_sheet(held, Path(samples))
    return held[name]


def _fastq_qc_stage(workdir: Optional[Path], samples: Optional[Path] = None) -> PipelineStage:
    if workdir is None:
        return PipelineStage("fastq_qc", False, "not_provided")
    summary = FileStatus.from_path(workdir / "results" / "fastq_qc_summary.json")
    trimmed = FileStatus.from_path(workdir / "inputs" / "trimmed_samples.csv")
    checkpoint = _load_json(stage_checkpoint_path(workdir, "pipeline"))
    held = _held_to_the_sheet("fastq_qc", workdir, samples)
    complete = bool(held["complete"])
    details = {
        "summary": summary.to_dict(),
        "trimmed_manifest": trimmed.to_dict(),
        "checkpoint_present": bool(checkpoint),
        "samples_not_done": list(held.get("samples_not_done") or []),
    }
    return PipelineStage(
        "fastq_qc",
        complete,
        str(held.get("status") or ("complete" if complete else "incomplete")),
        workdir=str(workdir),
        outputs={"trimmed_manifest": trimmed.path, "summary": summary.path},
        details=details,
    )


def _alignment_stage(workdir: Optional[Path], samples: Optional[Path] = None) -> PipelineStage:
    if workdir is None:
        return PipelineStage("alignment", False, "not_provided")
    aligned = FileStatus.from_path(workdir / "inputs" / "aligned_samples.csv")
    verify = _load_json(workdir / "results" / "star_alignment_verify.json")
    stage_checkpoint = _load_json(stage_checkpoint_path(workdir, "alignment"))
    checkpoint = _load_json(stage_checkpoint_path(workdir, "pipeline"))

    # `sample_count`. It was called `samples`, which is this
    # function's sheet: the count was then what the stage was held against
    # below, the hold failed without a word, and a finished alignment that
    # lacked a sample of the sheet read complete.
    if verify:
        complete_samples = verify.get("complete_samples")
        sample_count = verify.get("samples")
        resources = verify.get("resources", {})
        missing = verify.get("missing_sample_ids", [])
    elif stage_checkpoint:
        complete_samples = stage_checkpoint.get("completed_samples")
        sample_count = stage_checkpoint.get("expected")
        resources = stage_checkpoint.get("resources", {})
        missing = stage_checkpoint.get("missing_sample_ids", [])
    else:
        complete_samples = None
        sample_count = None
        resources = {}
        missing = []
    # One rule for "alignment is finished", the state reader's. This
    # took a bare aligned_samples.csv as a finished alignment; the stage writes
    # that file before any read is aligned, so a run that stopped there was
    # handed the BAM manifest as its FASTQ input from then on.
    held = _held_to_the_sheet("alignment", workdir, samples)
    complete = bool(held["complete"])

    details = {
        "aligned_manifest": aligned.to_dict(),
        "verify_present": bool(verify),
        "checkpoint_present": bool(checkpoint),
        "stage_checkpoint_present": bool(stage_checkpoint),
        "complete_samples": complete_samples,
        "samples": sample_count,
        "missing_sample_ids": missing,
        "samples_not_done": list(held.get("samples_not_done") or []),
        "resources": resources,
    }
    status = "complete" if complete else ("partial" if complete_samples else "incomplete")
    return PipelineStage(
        "alignment",
        complete,
        status,
        workdir=str(workdir),
        outputs={"aligned_manifest": aligned.path},
        details=details,
    )

def _variant_stage(workdir: Optional[Path]) -> PipelineStage:
    if workdir is None:
        return PipelineStage("variants", False, "not_provided")
    checkpoint = _load_json(stage_checkpoint_path(workdir, "pipeline"))
    final = checkpoint.get("final", {}) if isinstance(checkpoint, dict) else {}
    complete = bool(final.get("complete"))
    output = final.get("output")
    if not complete and output:
        # The same rule as pipeline.state.summarize_variants.
        complete = vcf_with_index_ready(output)
    return PipelineStage(
        "variants",
        complete,
        "complete" if complete else "incomplete",
        workdir=str(workdir),
        outputs={"final": str(output or "")},
        details={"checkpoint_present": bool(checkpoint)},
    )


def infer_related_workdir(samples: Path, suffix: str) -> Path:
    stem = samples.stem
    for token in ("_samples_fastQ", "_samples_fastq", "_samples", ".samples"):
        stem = stem.replace(token, "")
    return samples.parent / f"{stem}.{suffix}"


def _stage_after_alignment(stage_graph: tuple[str, ...]) -> str:
    """The first stage of ``stage_graph`` that works on aligned reads, or "complete"."""
    if "alignment" in stage_graph:
        later = stage_graph[stage_graph.index("alignment") + 1:]
    else:
        later = tuple(name for name in stage_graph if name != "fastq_qc")
    return later[0] if later else "complete"


def build_dispatch_plan(
    *,
    samples: Union[str, Path],
    fastq_qc_workdir: str | Optional[Path] = None,
    align_workdir: str | Optional[Path] = None,
    variant_workdir: str | Optional[Path] = None,
    validate_paths: bool = False,
    manifest_profile: str | None = None,
    stage_graph: tuple[str, ...] | list[str] | None = None,
    factors: tuple[str, ...] = (),
) -> DispatchPlan:
    samples_path = Path(samples)
    # With --factor the sheet needs the factor columns, not
    # `condition`. Read without them, a factor-only sheet was refused here.
    manifest = inspect_manifest(
        samples_path, manifest_profile=manifest_profile, validate_paths=validate_paths,
        factors=tuple(factors),
    )

    fq_workdir = Path(fastq_qc_workdir) if fastq_qc_workdir else infer_related_workdir(samples_path, "fastq_qc")
    al_workdir = Path(align_workdir) if align_workdir else infer_related_workdir(samples_path, "star_align")
    var_workdir = Path(variant_workdir) if variant_workdir else None
    active_graph = tuple(stage_graph or ("fastq_qc", "alignment", "variants"))

    stages: list[PipelineStage] = []
    fq_stage = _fastq_qc_stage(fq_workdir if fq_workdir.exists() else None, samples_path)
    al_stage = _alignment_stage(al_workdir if al_workdir.exists() else None, samples_path)
    var_stage = _variant_stage(var_workdir if var_workdir and var_workdir.exists() else None)
    stage_lookup = {stage.name: stage for stage in (fq_stage, al_stage, var_stage)}
    stages.extend(stage_lookup[name] for name in active_graph if name in stage_lookup)

    # What follows alignment in THIS pipeline. The plan named variant
    # calling for every profile, so a dry run printed "Next stage: variants"
    # for a sheet of BAMs while the run started at duplicate marking (DNA,
    # ATAC) or at quantification (RNA). Only variant calling can be seen to be
    # finished from here; any other stage is named as the one that comes next.
    after_alignment = _stage_after_alignment(active_graph)
    if after_alignment == "variants":
        after_alignment = "variants" if not var_stage.complete else "complete"

    if manifest.has_bam and manifest.mode == "bam":
        resolved = str(samples_path)
        role = "bam_manifest"
        next_stage = after_alignment
        if next_stage == "variants":
            action = "Run variant calling from BAM manifest."
        elif next_stage == "complete":
            action = "Workflow appears complete."
        else:
            action = f"Run {next_stage} from BAM manifest."
    elif manifest.has_fastq:
        if al_stage.complete and al_stage.outputs and al_stage.outputs.get("aligned_manifest"):
            resolved = al_stage.outputs["aligned_manifest"]
            role = "aligned_bam_manifest"
            next_stage = after_alignment
            if next_stage == "variants":
                action = "Run variant calling from STAR aligned BAM manifest."
            elif next_stage == "complete":
                action = "Alignment workflow appears complete."
            else:
                action = f"Run {next_stage} from STAR aligned BAM manifest."
        elif fq_stage.complete and fq_stage.outputs and fq_stage.outputs.get("trimmed_manifest"):
            resolved = fq_stage.outputs["trimmed_manifest"]
            role = "trimmed_fastq_manifest"
            next_stage = "alignment"
            action = "Run STAR alignment from trimmed FASTQ manifest."
        else:
            resolved = str(samples_path)
            role = "raw_fastq_manifest"
            next_stage = "fastq_qc"
            action = "Run FASTQ QC/trimming before alignment."
    else:
        resolved = str(samples_path)
        role = "unknown"
        next_stage = "inspect_manifest"
        action = "Manifest input type is unknown; inspect required columns."

    return DispatchPlan(
        input_manifest=manifest,
        resolved_manifest=resolved,
        resolved_manifest_role=role,
        next_stage=next_stage,
        stages=stages,
        recommended_action=action,
    )


def print_dispatch_plan(plan: DispatchPlan) -> None:
    print("Dispatch plan")
    print("-------------")
    print(f"Input manifest      : {plan.input_manifest.path}")
    print(f"Input mode          : {plan.input_manifest.mode}")
    print(f"Samples             : {plan.input_manifest.samples}")
    print(f"Resolved manifest   : {plan.resolved_manifest}")
    print(f"Resolved role       : {plan.resolved_manifest_role}")
    print(f"Next stage          : {plan.next_stage}")
    print(f"Recommended action  : {plan.recommended_action}")
    if plan.input_manifest.validation_errors:
        print("\nInput validation errors")
        print("-----------------------")
        for err in plan.input_manifest.validation_errors:
            print(f"ERROR: {err}")
    print("\nStages")
    print("------")
    for stage in plan.stages:
        label = "OK" if stage.complete else stage.status.upper()
        wd = stage.workdir or "."
        print(f"{stage.name}\t{label}\t{wd}")
