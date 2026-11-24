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

from pathlib import Path
from typing import Optional

import typer

variants_app = typer.Typer(help="Call, inspect, and prepare variants")


@variants_app.command("freebayes")
def variants_freebayes(
    samples: Path = typer.Option(..., "--samples", help="Sample manifest CSV with sample_id, group, and bam columns"),
    reference: Path = typer.Option(..., "--reference", help="Reference FASTA indexed with samtools faidx"),
    out: Path = typer.Option(..., "--out", help="Output VCF or VCF.GZ"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Working directory for intermediate files"),
    fastq_qc_workdir: Optional[Path] = typer.Option(None, "--fastq-qc-workdir", help="Completed `paralome-worker fastq qc` workdir used when --samples is a FASTQ manifest"),
    align_workdir: Optional[Path] = typer.Option(None, "--align-workdir", help="Completed `paralome-worker align star` workdir used when --samples is a FASTQ manifest"),
    threads: int = typer.Option(8, "--threads", help="Chunks called at once: local worker processes, or the SLURM array throttle"),
    jobs: int = typer.Option(0, "--jobs", help="Pieces the calling regions are cut into, one SLURM array task each; 0 = automatic, 8 per --threads CPU (never more than there are regions)"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase Paralome verbosity (-v, -vv, -vvv)"),
    regions: Optional[Path] = typer.Option(None, "--regions", help="Optional freebayes regions file, one region per line"),
    targets: Optional[str] = typer.Option(None, "--targets", help="Target mode: auto, genome, or path to BED [default: auto]"),
    gtf: Optional[Path] = typer.Option(None, "--gtf", help="Optional GTF for gene-level region scheduling"),
    region_strategy: str = typer.Option("auto", "--region-strategy", help="Region strategy: auto, empirical, or gene. Auto uses gene regions when --gtf is supplied."),
    gene_padding: int = typer.Option(100, "--gene-padding", help="Padding added to gene regions when --region-strategy gene is used"),
    chunk_size: int = typer.Option(100000, "--chunk-size", help="Genome chunk size when generating regions from FASTA index"),
    # The region cap and the stall watchdog. Merged gene regions of
    # 0.5-1.6 Mb held five tasks at 100% CPU for up to two days.
    max_region_length: int = typer.Option(100000, "--max-region-length", help="Longest region, in bases, one FreeBayes call is given; longer regions are cut into consecutive pieces. 0 = no cap"),
    stall_minutes: float = typer.Option(30.0, "--stall-minutes", help="Minutes a FreeBayes call may write nothing before its region is finished in 2 kb windows from its last complete record. 0 = no watchdog"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("2G", "--mem", help="SLURM memory per FreeBayes shard/task"),
    slurm_node: Optional[str] = typer.Option(None, "--slurm-node", help="Optional SLURM node constraint, e.g. node01"),
    slurm_account: Optional[str] = typer.Option(None, "--slurm-account", help="Optional SLURM account"),
    slurm_qos: Optional[str] = typer.Option(None, "--slurm-qos", help="Optional SLURM QOS"),
    slurm_oversubscribe: bool = typer.Option(True, "--oversubscribe/--no-oversubscribe", help="Request SLURM oversubscription for one-CPU FreeBayes shard arrays"),
    submit: bool = typer.Option(False, "--submit/--no-submit", help="Submit SLURM array immediately"),
    min_alternate_count: int = typer.Option(2, "--min-alternate-count"),
    min_alternate_fraction: float = typer.Option(0.05, "--min-alternate-fraction"),
    # No base- or mapping-quality cut unless asked for (the pipeline passes 0 for both).
    min_base_quality: int = typer.Option(0, "--min-base-quality"),
    min_mapping_quality: int = typer.Option(0, "--min-mapping-quality"),
    min_qual: float = typer.Option(0.0, "--min-qual", help="Calls below this QUAL are left out of the merged VCF (the pipeline passes its --min-qual, default 20)"),
    haplotype_length: int = typer.Option(3, "--haplotype-length"),
    pooled_discrete: bool = typer.Option(False, "--pooled-discrete/--no-pooled-discrete"),
    pooled_continuous: bool = typer.Option(False, "--pooled-continuous/--no-pooled-continuous"),
    extra_freebayes_options: Optional[str] = typer.Option(None, "--extra-freebayes-options", help="Quoted string of additional freebayes options"),
    annotation: str = typer.Option("none", "--annotation", help="Inline annotation as dependent SLURM jobs (--backend slurm only): none, snpeff, annovar, or all. Needs --snpeff-jar / --annovar-dir and the genome's database options"),
    snpeff_jar: Optional[Path] = typer.Option(None, "--snpeff-jar", help="Path to snpEff.jar"),
    snpeff_config: Optional[Path] = typer.Option(None, "--snpeff-config", help="Path to snpEff config"),
    snpeff_genome: Optional[str] = typer.Option(None, "--snpeff-genome", help="SnpEff genome database name; required with --annotation snpeff or all"),
    snpeff_memory: str = typer.Option("8g", "--snpeff-memory", help="Java memory for SnpEff, e.g. 8g"),
    annovar_dir: Optional[Path] = typer.Option(None, "--annovar-dir", help="ANNOVAR installation directory"),
    annovar_db: Optional[Path] = typer.Option(None, "--annovar-db", help="ANNOVAR humandb directory"),
    annovar_buildver: Optional[str] = typer.Option(None, "--annovar-buildver", help="ANNOVAR build version; required with --annotation annovar or all"),
    annovar_protocol: Optional[str] = typer.Option(None, "--annovar-protocol", help="Comma-separated ANNOVAR protocols; required with --annotation annovar or all"),
    annovar_operation: Optional[str] = typer.Option(None, "--annovar-operation", help="Comma-separated ANNOVAR operations; required with --annotation annovar or all"),
    profile: str = typer.Option("none", "--profile", help="Accepted for the pipeline's sake and must be none: this command calls variants and does not filter them; the filtering stage takes the profile"),
):
    from paralome.pipeline.dispatcher import build_dispatch_plan
    from paralome.variants.freebayes import default_workdir, run_freebayes_workflow

    dispatch = build_dispatch_plan(
        samples=samples,
        fastq_qc_workdir=fastq_qc_workdir,
        align_workdir=align_workdir,
    )
    if getattr(dispatch, "resolved_manifest", None):
        samples = Path(dispatch.resolved_manifest)

    # Record dispatcher provenance before launching FreeBayes.
    import json
    from dataclasses import asdict, is_dataclass

    # In the run's workdir. This was always <--out>/inputs, which for
    # a local run whose --out names a VCF file made a directory of that name.
    dispatch_path = default_workdir(Path(out), backend, workdir) / "inputs" / "dispatch_plan.json"
    dispatch_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(dispatch, "to_dict"):
        dispatch_data = dispatch.to_dict()
    elif is_dataclass(dispatch):
        dispatch_data = asdict(dispatch)
    else:
        dispatch_data = dict(getattr(dispatch, "__dict__", {}))
    dispatch_path.write_text(json.dumps(dispatch_data, indent=2, sort_keys=True) + "\n")

    result = run_freebayes_workflow(
        samples=samples,
        reference=reference,
        out_vcf=out,
        backend=backend,
        workdir=workdir,
        threads=threads,
        jobs=jobs,
        verbose=verbose,
        regions=regions,
        targets=targets,
        chunk_size=chunk_size,
        gtf=gtf,
        region_strategy=region_strategy,
        gene_padding=gene_padding,
        partition=partition,
        mem=mem,
        slurm_node=slurm_node,
        slurm_account=slurm_account,
        slurm_qos=slurm_qos,
        slurm_oversubscribe=slurm_oversubscribe,
        submit=submit,
        min_alternate_count=min_alternate_count,
        min_alternate_fraction=min_alternate_fraction,
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
        haplotype_length=haplotype_length,
        min_qual=min_qual,
        pooled_discrete=pooled_discrete,
        pooled_continuous=pooled_continuous,
        extra_options=extra_freebayes_options,
        empirical_min_depth=8,
        empirical_min_samples=1,
        empirical_merge_distance=50,
        empirical_padding=100,
        annotation=annotation,
        snpeff_jar=snpeff_jar,
        snpeff_config=snpeff_config,
        snpeff_genome=snpeff_genome,
        snpeff_memory=snpeff_memory,
        annovar_dir=annovar_dir,
        annovar_db=annovar_db,
        annovar_buildver=annovar_buildver,
        annovar_protocol=annovar_protocol,
        annovar_operation=annovar_operation,
        profile=profile,
        max_region_length=max_region_length,
        stall_minutes=stall_minutes,
    )

    if result.backend == "slurm" and result.submitted:
        # The merge (and annotation) jobs are already queued with
        # dependencies; running the merge script by hand merged twice.
        print(f"Submitted last SLURM job: {result.job_id} (end of the chain)")
        print(f"Workdir: {result.workdir}")
        print("The merge (and any annotation) jobs run on their own after calling; nothing to run by hand.")
        print(f"Final VCF when they finish: {result.out_vcf}")
    elif result.backend == "slurm" and result.already_complete:
        # Nothing to submit; it said to submit the array again.
        print(f"Nothing to submit: the result is already complete: {result.out_vcf}")
    elif result.backend == "slurm":
        # The whole chain in order, annotation included; it named
        # only the array and the merge, so an annotated run ended unannotated.
        print(f"SLURM files written to: {result.workdir}")
        print("Submit, in this order:")
        for step in result.manual_steps:
            print(f"  {step}")
    else:
        print(f"FreeBayes VCF written to: {result.out_vcf}")



# What the generated merge scripts run once their VCF is in place.
@variants_app.command("freebayes-merge-record")
def variants_freebayes_merge_record(
    workdir: Path = typer.Option(..., "--workdir", help="The FreeBayes run's work directory"),
    merged_vcf: Path = typer.Option(..., "--merged-vcf", help="The merged, indexed VCF"),
    run_backend: str = typer.Option(..., "--run-backend", help="local or slurm, as the run was planned (recorded)"),
    shards: int = typer.Option(..., "--shards", help="Chunks the run was planned in"),
    sets_final: bool = typer.Option(..., "--sets-final/--no-sets-final",
                                    help="Record the merged VCF as the final result (no annotation chain)"),
    variant_count: str = typer.Option("", "--variant-count", help="Records in the merged VCF, as counted"),
    duplicates_removed: str = typer.Option("", "--duplicates-removed", help="Duplicate records dropped"),
) -> None:
    """Write results/freebayes_summary.json and record the merge in the run's checkpoint."""
    from paralome.variants.freebayes import record_freebayes_merge

    record_freebayes_merge(workdir=workdir, merged_vcf=merged_vcf, backend=run_backend, shards=shards,
                           sets_final=sets_final, variant_count=variant_count,
                           duplicates_removed=duplicates_removed)


@variants_app.command("freebayes-annotation-record")
def variants_freebayes_annotation_record(
    workdir: Path = typer.Option(..., "--workdir", help="The FreeBayes run's work directory"),
    mode: str = typer.Option(..., "--mode", help="The annotation mode that produced the VCF"),
    output: Path = typer.Option(..., "--output", help="The annotated, merged VCF"),
) -> None:
    """Record the annotated, merged VCF as the annotation's output and the run's final result."""
    from paralome.variants.freebayes import record_annotation_merge

    record_annotation_merge(workdir=workdir, mode=mode, output=output)
