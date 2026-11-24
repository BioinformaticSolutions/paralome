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
from typing import Any, Optional

from paralome.checkpoints import stage_checkpoint_path

from .state import STAGE_LABELS


def _status_icon(status: Optional[str], complete: Optional[bool] = None) -> str:
    status = (status or "unknown").lower()
    if complete is True or status == "complete":
        return "OK"
    if status in {"partial", "running", "ready"}:
        return "PARTIAL"
    if status in {"not_available", "not_provided", "none", "skipped"}:
        return "NA"
    if status in {"failed", "error"}:
        return "FAIL"
    return "WAIT"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "."
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "."
    try:
        return f"{float(value):.3f}%"
    except Exception:
        return str(value)


def _file_line(label: str, info: Any) -> None:
    if not isinstance(info, dict):
        print(f"{label:<24}: .")
        return
    path = info.get("path")
    exists = info.get("exists")
    nonempty = info.get("nonempty")
    state = "OK" if nonempty else ("EMPTY" if exists else "MISSING")
    print(f"{label:<24}: {state} {path}")


def _stage_line(
    name: str,
    obj: dict[str, Any],
    flow_obj: Optional[dict[str, Any]] = None,
    *,
    width: int = 22,
    stale_reason: Optional[str] = None,
) -> None:
    status = obj.get("status", "unknown")
    flow = flow_obj or {}
    state = flow.get("state") or status
    # The icon is the workflow's verdict, not the raw summary's. A
    # stage about to be run again, or one past --stop-at, has its old outputs
    # on disk; taking "complete" from those printed OK beside "waiting".
    icon = _status_icon(state, None if flow.get("state") else obj.get("complete"))
    note = ""
    if stale_reason is not None or flow.get("stale"):
        icon = "STALE"
        note = "  to be run again" + (f": {stale_reason}" if stale_reason else "")
    elif obj.get("samples_not_done"):
        # Why a stage with results on disk is not finished.
        note = "  in the sample sheet and not done: " + ", ".join(str(name) for name in obj["samples_not_done"])
    print(f"{name:<{width}} {icon:<8} {state:<12} raw={status}{note}")


def _stage_graph_for(
    pipeline: dict[str, Any],
    pipeline_stages: dict[str, Any],
    project: dict[str, Any],
) -> list[str]:
    """Which stages this project actually runs, in order.

    Three sources, most authoritative first. `compute_pipeline_state` already
    derived the graph from the profile and wrote it into the state, so an
    ordinary `status` reads it straight back. An older state carries no
    `stage_graph` key but does carry `stages`, which is built by iterating the
    graph and so preserves it. Only if neither is present -- which means the
    pipeline section is missing entirely -- is the profile resolved again
    here. Never a literal list of stage names.
    """
    graph = pipeline.get("stage_graph")
    if isinstance(graph, list) and graph:
        return [str(name) for name in graph]
    if pipeline_stages:
        return [str(name) for name in pipeline_stages]

    from .profiles import resolve_pipeline_profile

    resolved = resolve_pipeline_profile(
        project.get("pipeline_profile") or project.get("profile"),
        project.get("segregation_profile"),
    )
    return [str(name) for name in resolved.stage_graph]


def _recorded_stale_stages(project: dict[str, Any]) -> tuple[dict[str, str], Optional[Path]]:
    """Stages the supervisor has on record as having to run again, with why,
    and the project folder the record was found in.

    `status` runs without the supervisor, and a state rebuilt from
    the stage folders knows nothing of the list -- every stale stage on it
    would read as finished. The list is in the supervisor's checkpoint, in the
    project folder, which is the parent of the stage folders named in the
    state. No checkpoint, or one that cannot be read, means no list.
    """
    folders = [
        Path(str(value)) for key, value in project.items()
        if value and (key.endswith("_workdir") or key == "data_workdir_path")
    ]
    # The single-cell stage folders sit one level further down.
    roots: list[Path] = []
    for root in [folder.parent for folder in folders] + [folder.parent.parent for folder in folders]:
        if root not in roots:
            roots.append(root)
    for root in roots:
        try:
            recorded = json.loads(stage_checkpoint_path(root, "pipeline_supervisor").read_text())
            # Only the checkpoint of this very folder: it names its own state file.
            if Path(str(recorded["project_state"])).parent.resolve() != root.resolve():
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        stale = recorded.get("stale_stages") if isinstance(recorded, dict) else None
        if isinstance(stale, dict):
            return {
                str(name): str((record or {}).get("reason") or "") if isinstance(record, dict) else ""
                for name, record in stale.items()
            }, root
        return {}, root
    return {}, None


def _print_stages(
    state: dict[str, Any],
    pipeline_stages: dict[str, Any],
    stage_graph: list[str],
) -> None:
    """The stage table, for the profile this project is -- not for DNA.

    The rows come from the profile's stage graph, so every stage the project
    runs is shown (a failed stage included) and no stage it does not run is.
    """
    graph = stage_graph

    def label_of(name: str) -> str:
        flow = pipeline_stages.get(name)
        if isinstance(flow, dict) and flow.get("label"):
            return str(flow["label"])
        return STAGE_LABELS.get(name, name)

    # The labels are profile-dependent and some are long ("Motif analysis (single motifs)"), so the
    # column is measured rather than assumed.
    width = max([len(label_of(name)) for name in graph] + [22])
    project = state.get("project") if isinstance(state.get("project"), dict) else {}
    recorded_stale, project_root = _recorded_stale_stages(project)
    inputs: tuple[str, ...] = ()
    if recorded_stale:
        # The supervisor's own test of what is gone, so the two
        # cannot disagree. Of the run's inputs the state names these two.
        from paralome.pipeline.supervisor import _run_inputs, stage_has_nothing_left

        inputs = tuple(_run_inputs({"samples": project.get("samples_manifest"), "rds": project.get("rds")}))

    for name in graph:
        flow = pipeline_stages.get(name)
        obj = state.get(name)
        flow = flow if isinstance(flow, dict) else {}
        # A stage past --stop-at or switched off is not going to be run again.
        # Nor is one whose folder has been removed: the next start drops it
        # from the list (supervisor._forget_what_is_gone), so it is not shown.
        stale = (
            name in recorded_stale and not flow.get("optional")
            and not stage_has_nothing_left(state, name, project_root, inputs)
        )
        _stage_line(
            label_of(name),
            obj if isinstance(obj, dict) else {},
            flow,
            width=width,
            stale_reason=recorded_stale[name] if stale else None,
        )


def _print_per_sample_journey(state: dict[str, Any]) -> None:
    """One row per sample: raw reads to QC to mapped, with the parameters used.

    It answers how many sequences passed QC and how many mapped, per sample.

    There are three -- fastq_qc_summary.tsv, star_alignment_metrics.tsv and
    sample_expression_qc.tsv -- and no way to read one sample across them. They
    do not even agree on a noun: fastp counts READS, STAR counts INPUT READS,
    the FPKM stage counts FRAGMENTS. For single-end those coincide; for
    paired-end they do not, so the units are named in the header rather than
    silently reconciled.

    THE PARAMETERS ARE PRINTED WITH IT, and that is not decoration. An 87%
    "unmapped: too short" rate was diagnosed over a day, and every step of the
    diagnosis was reading a setting out of a generated sbatch -- the mismatch
    fraction, the multimap cap, whether the GTF was passed to an index that
    already had it. A run whose numbers look wrong should show what produced
    them on the same screen.
    """
    fastq = state.get("fastq_qc", {}) if isinstance(state.get("fastq_qc"), dict) else {}
    alignment = state.get("alignment", {}) if isinstance(state.get("alignment"), dict) else {}
    per_sample = alignment.get("samples_detail")
    if not isinstance(per_sample, list) or not per_sample:
        return

    print("")
    print("Per sample")
    print("----------")
    print(f"  {'SAMPLE':28} {'INPUT':>12} {'UNIQUE':>9} {'UNIQUE%':>8} "
          f"{'MULTI%':>7} {'TOOSHORT%':>10}")
    for row in per_sample:
        if not isinstance(row, dict):
            continue
        print(
            f"  {str(row.get('sample_id', '.'))[:28]:28} "
            f"{_fmt_int(row.get('input_reads')):>12} "
            f"{_fmt_int(row.get('uniquely_mapped_reads')):>9} "
            f"{_fmt_pct(row.get('uniquely_mapped_percent')):>8} "
            f"{_fmt_pct(row.get('multimapped_percent')):>7} "
            f"{_fmt_pct(row.get('unmapped_too_short_percent')):>10}"
        )

    print("")
    print("  INPUT is reads entering STAR, which is what survived QC "
          f"({_fmt_pct(fastq.get('surviving_percent'))} of raw).")
    print("  TOOSHORT% is STAR's name for 'no acceptable alignment found' --")
    print("  it does not mean the read was short.")

    settings = alignment.get("star_settings")
    if isinstance(settings, dict) and settings:
        print("")
        print("  Aligned with:")
        fraction = settings.get("out_filter_mismatch_nover_read_lmax")
        if fraction is not None:
            print(f"    {'mismatch fraction':24}: {fraction}"
                  f"   (--star-out-filter-mismatch-nover-read-lmax)")
        # The absolute cap is printed NEXT TO the fraction, because
        # the two are applied together and STAR takes whichever is stricter.
        # Raising the fraction from 0.08 to 0.16 changes nothing for 92bp reads
        # while the cap is 10 (0.16 x 92 = 14.7), which shows only when both
        # are printed.
        nmax = settings.get("out_filter_mismatch_nmax")
        if nmax is not None:
            binding = ""
            if isinstance(fraction, (int, float)) and alignment.get("samples_detail"):
                lengths = [r.get("average_input_read_length")
                           for r in alignment["samples_detail"]
                           if isinstance(r, dict) and r.get("average_input_read_length")]
                if lengths:
                    allowed = float(fraction) * (sum(lengths) / len(lengths))
                    binding = ("   <- this is the binding limit"
                               if nmax < allowed else
                               f"   (the fraction allows {allowed:.1f}, so that binds)")
            print(f"    {'mismatch cap, absolute':24}: {nmax}{binding}")

        loci = settings.get("out_filter_multimap_nmax")
        if loci is not None:
            print(f"    {'max loci per read':24}: {loci}"
                  f"   reads mapping to more are discarded")

        score = settings.get("out_filter_score_min_over_lread")
        match = settings.get("out_filter_match_nmin_over_lread")
        if score is not None or match is not None:
            print(f"    {'min score / matched':24}: {score} / {match}"
                  f"   of read length -- below this a read is 'too short'")
        if settings.get("loose"):
            print(f"    {'':24}  RUN WAS --loose: filters opened deliberately.")
            print(f"    {'':24}  Expect partial alignments, not rescued reads.")
        if "sjdbGTFfile_passed" in settings:
            # Said as a sentence, because "False" is the answer to a question nobody asked: passing the GTF
            # to an index that already carries its junctions loses the reads that cross a junction.
            passed = settings["sjdbGTFfile_passed"]
            print(f"    {'splice junctions':24}: "
                  + ("from the GTF, inserted at run time"
                     if passed else
                     "from the index, which was built with the GTF"))



def print_pipeline_dashboard(state: dict[str, Any], *, detail: bool = False) -> None:
    project = state.get("project", {}) if isinstance(state.get("project"), dict) else {}
    overall = state.get("overall", {}) if isinstance(state.get("overall"), dict) else {}
    inputs = state.get("inputs", {}) if isinstance(state.get("inputs"), dict) else {}
    pipeline = state.get("pipeline", {}) if isinstance(state.get("pipeline"), dict) else {}
    pipeline_stages = pipeline.get("stages", {}) if isinstance(pipeline.get("stages"), dict) else {}

    # Resolved ONCE, and every section that needs to know which stages exist reads this rather than
    # carrying its own idea.
    stage_graph = _stage_graph_for(pipeline, pipeline_stages, project)

    fastq = state.get("fastq_qc", {}) if isinstance(state.get("fastq_qc"), dict) else {}
    alignment = state.get("alignment", {}) if isinstance(state.get("alignment"), dict) else {}
    dedup = state.get("deduplication", {}) if isinstance(state.get("deduplication"), dict) else {}
    # _print_stages reads every stage straight out of `state` by the name the graph gives it, so a stage
    # needs no local variable or print line of its own to be visible.
    variants = state.get("variants", {}) if isinstance(state.get("variants"), dict) else {}
    annotation = state.get("annotation", {}) if isinstance(state.get("annotation"), dict) else {}
    segregation = state.get("segregation", {}) if isinstance(state.get("segregation"), dict) else {}
    filtering = state.get("filtering", {}) if isinstance(state.get("filtering"), dict) else {}
    reporting = state.get("reporting", {}) if isinstance(state.get("reporting"), dict) else {}
    gene_burden = state.get("gene_burden", {}) if isinstance(state.get("gene_burden"), dict) else {}

    print("Paralome Pipeline Dashboard")
    print("=====================")
    print(f"Samples manifest     : {project.get('samples_manifest')}")
    # A workdir the project does not have is not printed as None.
    # An RNA run listed five of them -- Variant, Annotation, Segregation,
    # Filter, Report -- every one `None`, every one a stage RNA never runs.
    # Entries irrelevant to the profile are left out.
    # And only the stages this pipeline has. The state carries a
    # path for every one of them whatever the pipeline, so an RNA project
    # still listed six DNA folders it never makes.
    for label, key, stage in (
        ("FASTQ QC workdir", "fastq_qc_workdir", "fastq_qc"),
        ("Alignment workdir", "align_workdir", "alignment"),
        ("Dedup workdir", "dedup_workdir", "deduplication"),
        ("Variant workdir", "variant_workdir", "variants"),
        ("Annotation workdir", "annotation_workdir", "annotation"),
        ("Segregation workdir", "segregation_workdir", "segregation"),
        ("Filter workdir", "filter_workdir", "filtering"),
        ("Report workdir", "report_workdir", "reporting"),
    ):
        value = project.get(key)
        if value and (not stage_graph or stage in stage_graph):
            print(f"{label:<21}: {value}")
    # Only where the graph has the stage; every project has a root to derive it from.
    if "gene_burden" in stage_graph and project.get("gene_burden_workdir"):
        print(f"{'Gene burden workdir':<21}: {project.get('gene_burden_workdir')}")
    print(f"Pipeline profile     : {project.get('pipeline_profile') or project.get('profile')}")
    # Only if set. "Segregation profile  : ." on an RNA run is the same noise
    # as the None workdirs above -- and worse, it puts the word "Segregation"
    # on the screen of a project that has no segregation stage.
    if project.get("segregation_profile"):
        print(f"Segregation profile  : {project.get('segregation_profile')}")
    if project.get("alignment_profile"):
        print(f"Alignment profile    : {project.get('alignment_profile')}")
    if pipeline:
        print(f"Workflow state       : {pipeline.get('state')}")
        print(f"Next stage           : {pipeline.get('next_stage') or '.'}")

    dispatch = inputs.get("dispatch_plan") if isinstance(inputs.get("dispatch_plan"), dict) else {}
    input_manifest = dispatch.get("input_manifest") if isinstance(dispatch.get("input_manifest"), dict) else {}
    if input_manifest:
        print("")
        print("Inputs")
        print("------")
        print(f"Input mode           : {input_manifest.get('mode')}")
        print(f"Samples              : {input_manifest.get('samples')}")
        print(f"Resolved manifest    : {dispatch.get('resolved_manifest')}")
        print(f"Resolved role        : {dispatch.get('resolved_manifest_role')}")

    print("")
    print("Stages")
    print("------")
    _print_stages(state, pipeline_stages, stage_graph)

    # Every line under Performance belongs to one of these three stages, so a
    # profile that runs none of them gets no header rather than an empty one.
    if any(name in stage_graph for name in ("fastq_qc", "alignment", "deduplication")):
        print("")
        print("Performance")
        print("-----------")
    # Gated on the graph, like the dedup block below. scRNA and
    # snATAC start at pseudobulk -- their upstream is Cell Ranger's -- so
    # "FASTQ QC: ." on one of those was reporting a missing number for a
    # stage that is not missing, it is not theirs.
    if "fastq_qc" in stage_graph:
        if fastq.get("raw_reads") is not None:
            print(f"Raw reads            : {_fmt_int(fastq.get('raw_reads'))}")
            print(f"Clean reads          : {_fmt_int(fastq.get('clean_reads'))}")
            print(f"Reads surviving QC   : {_fmt_pct(fastq.get('surviving_percent'))}")
        else:
            print("FASTQ QC             : .")

    if "alignment" in stage_graph:
        if alignment.get("weighted_uniquely_mapped_percent") is not None:
            print(f"Alignment samples    : {alignment.get('completed_samples')}/{alignment.get('expected_samples')}")
            print(f"Unique mapping rate  : {_fmt_pct(alignment.get('weighted_uniquely_mapped_percent'))}")
            print(f"Total mapping rate   : {_fmt_pct(alignment.get('weighted_total_mapped_percent'))}")
            print(f"Input reads aligned  : {_fmt_int(alignment.get('input_reads'))}")
            print(f"Total mapped reads   : {_fmt_int(alignment.get('total_mapped_reads'))}")
        else:
            print("Alignment metrics    : .")

    # Gated on the stage graph, not on the summary's status: RNA runs no deduplication stage at all, so
    # its duplicate numbers are inapplicable, not merely missing.
    if "deduplication" in stage_graph and dedup.get("status") != "not_provided":
        print(f"Dedup samples        : {dedup.get('completed_samples') or '.'}/{dedup.get('samples') or '.'}")
        print(f"Duplicate rate       : {_fmt_pct((dedup.get('duplicate_rate') * 100) if isinstance(dedup.get('duplicate_rate'), (int, float)) else None)}")

    resources = alignment.get("resources") if isinstance(alignment.get("resources"), dict) else {}
    if resources:
        print("")
        print("Resources")
        print("---------")
        print(f"Backend              : {resources.get('backend', '.')}")
        print(f"Threads/sample       : {resources.get('threads_per_sample', '.')}")
        print(f"Concurrent jobs      : {resources.get('jobs') or resources.get('concurrent_jobs') or '.'}")
        print(f"Peak CPUs            : {resources.get('peak_cpus', '.')}")
        print(f"Memory/sample        : {resources.get('mem') or resources.get('slurm_mem') or '.'}")

    # Each output block gated on the stage graph as well as on
    # its own summary. summarize_* returns a populated dict for a workdir
    # that was never created, so an RNA run printed "Variant outputs",
    # "Annotation outputs", "Segregation outputs" and "Candidate
    # reporting" headers -- four sections for stages RNA does not run.
    if "variants" in stage_graph and variants:
        final_output = variants.get("final_output") if isinstance(variants.get("final_output"), dict) else {}
        print("")
        print("Variant outputs")
        print("---------------")
        _file_line("Final VCF", final_output)
        _file_line("Final VCF index", variants.get("final_index"))
        _file_line("Variant pipeline JSON", variants.get("pipeline_json"))
        fb_summary = variants.get("freebayes_summary") if isinstance(variants.get("freebayes_summary"), dict) else {}
        if fb_summary:
            print(f"Variant records         : {fb_summary.get('variant_count') or '.'}")
            print(f"FreeBayes shards        : {fb_summary.get('shards') or '.'}")
            print(f"FreeBayes summary       : {fb_summary.get('merged_vcf') and variants.get('freebayes', {}).get('summary') or '.'}")

    if "annotation" in stage_graph and annotation and annotation.get("status") != "not_provided":
        print("")
        print("Annotation outputs")
        print("------------------")
        _file_line("Annotated VCF", annotation.get("output_vcf"))
        _file_line("Annotated VCF index", annotation.get("output_index"))
        _file_line("Annotation checkpoint", annotation.get("checkpoint"))

    if "segregation" in stage_graph and segregation and segregation.get("status") != "not_provided":
        print("")
        print("Segregation outputs")
        print("-------------------")
        _file_line("All variants TSV", segregation.get("all_variants_tsv") or segregation.get("segregation_tsv"))
        _file_line("Segregation summary", segregation.get("summary_json"))
        _file_line("Segregation provenance", segregation.get("provenance_json"))
        _file_line("Segregation checkpoint", segregation.get("checkpoint"))
        print(f"Rows                 : {segregation.get('rows') or '.'}")
        print(f"Bucket sum           : {segregation.get('bucket_sum') or '.'}")
        print(f"Lossless             : {segregation.get('lossless')}")
        bucket_counts = segregation.get("bucket_counts") if isinstance(segregation.get("bucket_counts"), dict) else {}
        if bucket_counts:
            print("Bucket counts        : " + " ".join(f"{k}={v}" for k, v in sorted(bucket_counts.items())))

    profiles = filtering.get("profiles") if ("filtering" in stage_graph and isinstance(filtering.get("profiles"), dict)) else {}
    if profiles:
        print("")
        print("Filtering")
        print("---------")
        print(f"Scope               : {','.join(filtering.get('scope') or []) or '.'}")
        print(f"Lossless            : {filtering.get('lossless')}")
        for name, prof in sorted(profiles.items()):
            counts = prof.get("counts", {}) if isinstance(prof, dict) else {}
            print(
                f"{name:<22} total={counts.get('total', '.')} "
                f"pass={counts.get('pass', counts.get('candidates', '.'))} "
                f"fail_quality={counts.get('fail_quality', '.')} "
                f"fail_population={counts.get('fail_population', '.')} "
                f"fail_annotation={counts.get('fail_annotation', '.')} "
                f"fail_segregation={counts.get('fail_segregation', '.')} "
                f"fail_multiple={counts.get('fail_multiple', '.')}"
            )

    if "reporting" in stage_graph and reporting and reporting.get("status") != "not_provided":
        print("")
        print("Candidate reporting")
        print("-------------------")
        _file_line("Report checkpoint", reporting.get("checkpoint"))
        outputs = reporting.get("outputs") if isinstance(reporting.get("outputs"), dict) else {}
        _file_line("Report summary", outputs.get("summary"))
        _file_line("Report manifest", outputs.get("manifest"))
        _file_line("Review table", outputs.get("all_reviewable"))
        print(f"Rows scanned         : {reporting.get('rows_scanned') or '.'}")
        print(f"Rows reviewable      : {reporting.get('rows_reviewable') or '.'}")

    if "gene_burden" in stage_graph and gene_burden and gene_burden.get("status") not in ("not_provided", "not_available"):
        print("")
        print("Gene burden")
        print("-----------")
        if gene_burden.get("status") == "skipped":
            print(f"Skipped              : {gene_burden.get('reason') or '.'}")
        print(f"Probands             : {gene_burden.get('n_probands') or '.'}")
        tables = gene_burden.get("tables") if isinstance(gene_burden.get("tables"), dict) else {}
        for name, info in sorted(tables.items()):
            _file_line(name, info)
        _file_line("Evidome input", gene_burden.get("evidome_input"))
        _file_line("Burden summary", gene_burden.get("summary"))
        not_run = gene_burden.get("tiers_not_run") if isinstance(gene_burden.get("tiers_not_run"), dict) else {}
        for tier, why in sorted(not_run.items()):
            print(f"Not run ({tier})".ljust(21) + f": {why}")

    print("")
    print(f"Next action          : {overall.get('next_action')}")

    if detail:
        _print_per_sample_journey(state)
        print("")
        print("Detailed files")
        print("--------------")
        # Each line only for a stage this profile runs, like the
        # blocks above.
        detailed_files = (
            ("fastq_qc", "FASTQ summary", fastq.get("summary")),
            ("fastq_qc", "Trimmed manifest", fastq.get("trimmed_manifest")),
            ("alignment", "Alignment checkpoint", alignment.get("checkpoint")),
            ("alignment", "Alignment verify", alignment.get("verify")),
            ("alignment", "Alignment metrics", alignment.get("metrics")),
            ("alignment", "Aligned manifest", alignment.get("aligned_manifest")),
            ("deduplication", "Dedup checkpoint", dedup.get("checkpoint")),
            ("deduplication", "Dedup summary", dedup.get("summary")),
            ("deduplication", "Dedup manifest", dedup.get("dedup_manifest")),
            ("variants", "Dispatch plan", variants.get("dispatch_plan")),
            ("variants", "Pipeline run plan", variants.get("pipeline_run_plan")),
            ("annotation", "Annotation checkpoint", annotation.get("checkpoint")),
            ("annotation", "Annotated VCF", annotation.get("output_vcf")),
            ("segregation", "Segregation checkpoint", segregation.get("checkpoint")),
            ("segregation", "Segregation TSV", segregation.get("segregation_tsv")),
            ("filtering", "Filtering checkpoint", filtering.get("checkpoint")),
            ("filtering", "Filtering plan", filtering.get("plan")),
        )
        for stage_name, label, value in detailed_files:
            if stage_name in stage_graph:
                _file_line(label, value)
        print("")
        print("Workflow state machine")
        print("----------------------")
        for stage_name, obj in pipeline_stages.items():
            if isinstance(obj, dict):
                print(f"{stage_name}: state={obj.get('state')} raw={obj.get('status')} complete={obj.get('complete')}")

        print("")
        print("Stage data")
        print("----------")
        # Driven off the stage graph, like the table, so `--detail` prints Stage data only for the
        # stages this project has.
        for stage_name in stage_graph:
            obj = state.get(stage_name, {}) if isinstance(state.get(stage_name), dict) else {}
            print(f"{stage_name}: status={obj.get('status')} complete={obj.get('complete')}")
