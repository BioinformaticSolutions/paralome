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
from typing import Any, Optional

import typer

from paralome.binaries import WORKER_SCRIPT
from paralome.annotation.annovar import AnnovarConfig
from paralome.annotation.snpeff import SnpEffConfig
from paralome.annotation.standalone import run_annotation_engine_chain

annotate_app = typer.Typer(
    help="Annotate variant calls with one annotation stage and selectable annotation engine(s)",
    invoke_without_command=True,
)


def _print_result(result) -> None:
    print("Annotation stage")
    print("================")
    print(f"Engine              : {result.engine}")
    print(f"Backend             : {result.backend}")
    print(f"Workdir             : {result.workdir}")
    print(f"Input VCF           : {result.input_vcf}")
    print(f"Output VCF          : {result.output_vcf}")
    print(f"Script              : {result.script}")
    if result.plan:
        print(f"Plan JSON           : {result.plan}")
    if result.checkpoint:
        print(f"Checkpoint          : {result.checkpoint}")
    print(f"Submitted           : {result.submitted}")
    if result.job_id:
        print(f"Job ID              : {result.job_id}")
    print(f"Complete            : {result.complete}")


def _resolve_resources(
    *,
    genome: Optional[str],
    overrides: dict[str, Any],
    required: tuple[str, ...],
):
    from paralome.pipeline.resource_catalog import (
        ResourceResolutionError,
        resolve_genome_resources,
    )

    try:
        return resolve_genome_resources(
            genome=genome,
            overrides=overrides,
            required=required,
        )
    except ResourceResolutionError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_annovar_resources(
    *,
    genome: Optional[str],
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
) -> tuple[Path, Path, str, str, str]:
    # THE OPERATION LIST COMES FROM WHERE THE PROTOCOL LIST DOES, as
    # on the assay commands. Each was looked up by itself here, so a typed
    # protocol list ran with the genome entry's operations. This command reads
    # no configuration file: the two places are the options and the entry.
    from paralome.commands.pipeline import _refuse_an_operation_from_elsewhere
    from paralome.errors import ParalomeError

    typed = (annovar_protocol not in (None, ""), annovar_operation not in (None, ""))
    if typed[0] != typed[1]:
        _refuse_an_operation_from_elsewhere(
            pipeline_config=None,
            typed={"annovar_protocol": annovar_protocol, "annovar_operation": annovar_operation},
        )
    resolved = _resolve_resources(
        genome=genome,
        overrides={
            "annovar_dir": annovar_dir,
            "annovar_db": annovar_db,
            "annovar_buildver": annovar_buildver,
            "annovar_protocol": annovar_protocol,
            "annovar_operation": annovar_operation,
        },
        required=("annovar_dir", "annovar_db"),
    )
    if not typed[0] and resolved.value("annovar_protocol") and not resolved.value("annovar_operation"):
        raise ParalomeError(
            117,
            f"genome entry {resolved.genome} records annovar_protocol and no annovar_operation. "
            "ANNOVAR pairs the two lists entry by entry, so Paralome takes both from one place.",
            fix="Record annovar_operation in that genome entry (`paralome setup config` "
                "shows the file); or pass --annovar-protocol and --annovar-operation together",
        )
    # NO BUILD AND NO PROTOCOL OF PARALOME'S OWN. A genome registered without them would otherwise
    # be annotated as human (an hg19 genome against hg38 tables). The rule for the genome holds for
    # its build.
    missing = [
        (name, option)
        for name, option in (
            ("annovar_buildver", "--annovar-buildver"),
            ("annovar_protocol", "--annovar-protocol"),
            ("annovar_operation", "--annovar-operation"),
        )
        if not resolved.value(name)
    ]
    if missing:
        entry = (
            f"genome entry {resolved.genome!r} records no "
            if resolved.genome else "no --genome was given to supply "
        )
        raise typer.BadParameter(
            "ANNOVAR needs " + ", ".join(name for name, _ in missing) + ": "
            + entry + " or ".join(name for name, _ in missing) + ", and "
            + " / ".join(option for _, option in missing) + " was not given. "
            "Paralome never assumes a genome build or its databases. Record the "
            "key(s) in the genome entry (`paralome setup config` shows the file) "
            "or pass the option(s)."
        )
    # One operation for each protocol, checked here and by number.
    # A mismatch reached AnnovarConfig.validate and came out as PLM-0001.
    protocols, operations = (
        [item.strip() for item in str(resolved.value(name)).split(",") if item.strip()]
        for name in ("annovar_protocol", "annovar_operation")
    )
    if len(protocols) != len(operations):
        where = (
            "--annovar-protocol and --annovar-operation" if typed[0]
            else f"genome entry {resolved.genome}"
        )
        raise ParalomeError(
            101 if typed[0] else 117,
            f"{where}: annovar_protocol has {len(protocols)} entries ({','.join(protocols)}) and "
            f"annovar_operation has {len(operations)} ({','.join(operations)}). "
            "ANNOVAR needs one operation for each protocol.",
            fix=(
                "Give one --annovar-operation entry for each --annovar-protocol entry" if typed[0]
                else "Correct annovar_operation in that genome entry (`paralome setup config` shows "
                     "the file); or pass --annovar-protocol and --annovar-operation together"
            ),
        )
    return (
        resolved.path("annovar_dir"),
        resolved.path("annovar_db"),
        resolved.value("annovar_buildver"),
        resolved.value("annovar_protocol"),
        resolved.value("annovar_operation"),
    )


def _resolve_snpeff_resources(
    *,
    genome: Optional[str],
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
) -> tuple[Path, Path, str]:
    resolved = _resolve_resources(
        genome=genome,
        overrides={
            "snpeff_jar": snpeff_jar,
            "snpeff_config": snpeff_config,
            "snpeff_genome": snpeff_genome,
        },
        required=("snpeff_jar", "snpeff_config", "snpeff_genome"),
    )
    return (
        resolved.path("snpeff_jar"),
        resolved.path("snpeff_config"),
        resolved.value("snpeff_genome"),
    )


def _resolve_reference_fasta(genome: Optional[str], reference_fasta: Optional[Path]) -> Optional[Path]:
    """--reference-fasta, else the genome entry's reference_fasta, else None.

    Calls are split before ANNOVAR (annotation/annovar.py, ORIGIN_TAG); the
    FASTA left-aligns indels as gnomAD's are. The pipeline does not pass it:
    it comes from the --genome entry the annotation is already resolved from,
    so a project's recorded annotation request does not change.
    """
    if reference_fasta is not None:
        return Path(reference_fasta)
    if not genome:
        return None
    resolved = _resolve_resources(genome=genome, overrides={}, required=())
    return resolved.path("reference_fasta") if resolved.value("reference_fasta") else None


def _normalize_engine(engine: str) -> str:
    value = (engine or "").strip().lower()
    aliases = {
        "annovar": "annovar",
        "snpeff": "snpeff",
        "snp_eff": "snpeff",
        "snp-eff": "snpeff",
    }
    if value not in aliases:
        raise typer.BadParameter("--engine must be a comma-delimited list using: annovar, snpeff")
    return aliases[value]


def _parse_engine_chain(engine: str) -> list[str]:
    raw = [item.strip() for item in (engine or "").split(",")]
    engines = [_normalize_engine(item) for item in raw if item]
    if not engines:
        raise typer.BadParameter("--engine must include at least one annotation engine")
    seen = set()
    ordered = []
    for item in engines:
        if item in seen:
            raise typer.BadParameter(f"Duplicate annotation engine in --engine chain: {item}")
        seen.add(item)
        ordered.append(item)
    return ordered


def _run_selected_annotation(
    *,
    engine: str,
    vcf: Path,
    out: Path,
    genome: Optional[str],
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
    snpeff_memory: str,
    prefix: Optional[str],
    backend: str,
    submit: bool,
    dry_run: bool,
    partition: Optional[str],
    mem: str,
    threads: int,
    jobs: int,
    variant_workdir: Optional[Path],
    shard_vcf_dir: Optional[Path],
    reference_fasta: Optional[Path] = None,
):
    selected_engines = _parse_engine_chain(engine)
    annovar_config = None
    snpeff_config_obj = None

    if "annovar" in selected_engines:
        (
            resolved_annovar_dir,
            resolved_annovar_db,
            resolved_buildver,
            resolved_protocol,
            resolved_operation,
        ) = _resolve_annovar_resources(
            genome=genome,
            annovar_dir=annovar_dir,
            annovar_db=annovar_db,
            annovar_buildver=annovar_buildver,
            annovar_protocol=annovar_protocol,
            annovar_operation=annovar_operation,
        )
        annovar_config = AnnovarConfig(
            annovar_dir=resolved_annovar_dir,
            database_dir=resolved_annovar_db,
            buildver=resolved_buildver,
            protocol=resolved_protocol,
            operation=resolved_operation,
            reference_fasta=_resolve_reference_fasta(genome, reference_fasta),
        )
        annovar_config.validate()

    if "snpeff" in selected_engines:
        resolved_jar, resolved_config, resolved_genome = _resolve_snpeff_resources(
            genome=genome,
            snpeff_jar=snpeff_jar,
            snpeff_config=snpeff_config,
            snpeff_genome=snpeff_genome,
        )
        snpeff_config_obj = SnpEffConfig(
            jar=resolved_jar,
            config=resolved_config,
            genome=resolved_genome,
            memory=snpeff_memory,
        )
        snpeff_config_obj.validate()

    return run_annotation_engine_chain(
        engines=selected_engines,
        vcf=vcf,
        workdir=out,
        annovar_config=annovar_config,
        snpeff_config=snpeff_config_obj,
        prefix=prefix,
        backend=backend,
        submit=submit,
        dry_run=dry_run,
        partition=partition,
        mem=mem,
        threads=threads,
        jobs=jobs,
        variant_workdir=variant_workdir,
        shard_vcf_dir=shard_vcf_dir,
    )


@annotate_app.callback(invoke_without_command=True)
def annotate_command(
    ctx: typer.Context,
    engine: Optional[str] = typer.Option(None, "--engine", help="Comma-delimited annotation engine chain, e.g. annovar or annovar,snpeff"),
    vcf: Optional[Path] = typer.Option(None, "--vcf", help="Input VCF/VCF.GZ"),
    out: Optional[Path] = typer.Option(None, "--out", help="Annotation work directory"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Configured genome name used to resolve annotation resources"),
    annovar_dir: Optional[Path] = typer.Option(None, "--annovar-dir", help="ANNOVAR installation directory; overrides genome config"),
    annovar_db: Optional[Path] = typer.Option(None, "--annovar-db", help="ANNOVAR humandb directory; overrides genome config"),
    annovar_buildver: Optional[str] = typer.Option(None, "--annovar-buildver", help="ANNOVAR build version; overrides genome config"),
    annovar_protocol: Optional[str] = typer.Option(None, "--annovar-protocol", help="Comma-separated ANNOVAR protocols; overrides genome config"),
    annovar_operation: Optional[str] = typer.Option(None, "--annovar-operation", help="Comma-separated ANNOVAR operations; overrides genome config"),
    snpeff_jar: Optional[Path] = typer.Option(None, "--snpeff-jar", help="Path to snpEff.jar; overrides genome config"),
    snpeff_config: Optional[Path] = typer.Option(None, "--snpeff-config", help="Path to SnpEff config; overrides genome config"),
    snpeff_genome: Optional[str] = typer.Option(None, "--snpeff-genome", help="SnpEff genome database name; overrides genome config"),
    snpeff_memory: str = typer.Option("8g", "--snpeff-memory", help="Java memory for SnpEff"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Output run prefix; default is workdir name"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM job when backend=slurm"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write scripts/plans without running"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("16G", "--mem", help="SLURM memory per annotation task"),
    threads: int = typer.Option(1, "--threads", help="Total annotation CPU budget; existing shard VCFs run as one-CPU tasks"),
    jobs: int = typer.Option(0, "--jobs", help="Number of annotation shards when splitting a merged VCF; ignored when existing shard VCFs are supplied"),
    variant_workdir: Optional[Path] = typer.Option(None, "--variant-workdir", help="Completed FreeBayes workdir; consumes its shards/shard_*.vcf files directly"),
    shard_vcf_dir: Optional[Path] = typer.Option(None, "--shard-vcf-dir", help="Directory containing existing shard_*.vcf or shard_*.vcf.gz inputs"),
    reference_fasta: Optional[Path] = typer.Option(None, "--reference-fasta", help="Genome FASTA for left-aligning indels when calls are split before ANNOVAR; default: the genome entry's reference_fasta"),
):
    """Run the unified annotation stage with an ordered annotation engine chain."""
    if ctx.invoked_subcommand is not None:
        return
    if engine is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    if vcf is None:
        raise typer.BadParameter(f"--vcf is required when running {WORKER_SCRIPT} annotate --engine")
    if out is None:
        raise typer.BadParameter(f"--out is required when running {WORKER_SCRIPT} annotate --engine")

    result = _run_selected_annotation(
        engine=engine,
        vcf=vcf,
        out=out,
        genome=genome,
        annovar_dir=annovar_dir,
        annovar_db=annovar_db,
        annovar_buildver=annovar_buildver,
        annovar_protocol=annovar_protocol,
        annovar_operation=annovar_operation,
        snpeff_jar=snpeff_jar,
        snpeff_config=snpeff_config,
        snpeff_genome=snpeff_genome,
        snpeff_memory=snpeff_memory,
        prefix=prefix,
        backend=backend,
        submit=submit,
        dry_run=dry_run,
        partition=partition,
        mem=mem,
        threads=threads,
        jobs=jobs,
        variant_workdir=variant_workdir,
        shard_vcf_dir=shard_vcf_dir,
        reference_fasta=reference_fasta,
    )
    _print_result(result)


@annotate_app.command("snpeff")
def annotate_snpeff_command(
    vcf: Path = typer.Option(..., "--vcf", help="Input VCF/VCF.GZ"),
    out: Path = typer.Option(..., "--out", help="Annotation work directory"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Configured genome name used to resolve SnpEff resources"),
    snpeff_jar: Optional[Path] = typer.Option(None, "--snpeff-jar", help="Path to snpEff.jar; overrides genome config"),
    snpeff_config: Optional[Path] = typer.Option(None, "--snpeff-config", help="Path to SnpEff config; overrides genome config"),
    snpeff_genome: Optional[str] = typer.Option(None, "--snpeff-genome", help="SnpEff genome database name; overrides genome config"),
    snpeff_memory: str = typer.Option("8g", "--snpeff-memory", help="Java memory for SnpEff"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Output run prefix; default is workdir name"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM job when backend=slurm"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write scripts/plans without running"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("16G", "--mem", help="SLURM memory per annotation task"),
    threads: int = typer.Option(1, "--threads", help="Total annotation CPU budget; existing shard VCFs run as one-CPU tasks"),
    jobs: int = typer.Option(0, "--jobs", help="Number of annotation shards when splitting a merged VCF; ignored when existing shard VCFs are supplied"),
    variant_workdir: Optional[Path] = typer.Option(None, "--variant-workdir", help="Completed FreeBayes workdir; consumes its shards/shard_*.vcf files directly"),
    shard_vcf_dir: Optional[Path] = typer.Option(None, "--shard-vcf-dir", help="Directory containing existing shard_*.vcf or shard_*.vcf.gz inputs"),
):
    """Compatibility alias for `paralome-worker annotate --engine snpeff`."""
    result = _run_selected_annotation(
        engine="snpeff",
        vcf=vcf,
        out=out,
        genome=genome,
        annovar_dir=None,
        annovar_db=None,
        annovar_buildver=None,
        annovar_protocol=None,
        annovar_operation=None,
        snpeff_jar=snpeff_jar,
        snpeff_config=snpeff_config,
        snpeff_genome=snpeff_genome,
        snpeff_memory=snpeff_memory,
        prefix=prefix,
        backend=backend,
        submit=submit,
        dry_run=dry_run,
        partition=partition,
        mem=mem,
        threads=threads,
        jobs=jobs,
        variant_workdir=variant_workdir,
        shard_vcf_dir=shard_vcf_dir,
    )
    _print_result(result)


@annotate_app.command("annovar")
def annotate_annovar_command(
    vcf: Path = typer.Option(..., "--vcf", help="Input VCF/VCF.GZ"),
    out: Path = typer.Option(..., "--out", help="Annotation work directory"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Configured genome name used to resolve ANNOVAR resources"),
    annovar_dir: Optional[Path] = typer.Option(None, "--annovar-dir", help="ANNOVAR installation directory; overrides genome config"),
    annovar_db: Optional[Path] = typer.Option(None, "--annovar-db", help="ANNOVAR humandb directory; overrides genome config"),
    annovar_buildver: Optional[str] = typer.Option(None, "--annovar-buildver", help="ANNOVAR build version; overrides genome config"),
    annovar_protocol: Optional[str] = typer.Option(None, "--annovar-protocol", help="Comma-separated ANNOVAR protocols; overrides genome config"),
    annovar_operation: Optional[str] = typer.Option(None, "--annovar-operation", help="Comma-separated ANNOVAR operations; overrides genome config"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Output run prefix; default is workdir name"),
    backend: str = typer.Option("local", "--backend", help="Execution backend: local or slurm"),
    submit: bool = typer.Option(False, "--submit", help="Submit SLURM job when backend=slurm"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write scripts/plans without running"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition"),
    mem: str = typer.Option("16G", "--mem", help="SLURM memory per annotation task"),
    threads: int = typer.Option(1, "--threads", help="Total annotation CPU budget; existing shard VCFs run as one-CPU tasks"),
    jobs: int = typer.Option(0, "--jobs", help="Number of annotation shards when splitting a merged VCF; ignored when existing shard VCFs are supplied"),
    variant_workdir: Optional[Path] = typer.Option(None, "--variant-workdir", help="Completed FreeBayes workdir; consumes its shards/shard_*.vcf files directly"),
    shard_vcf_dir: Optional[Path] = typer.Option(None, "--shard-vcf-dir", help="Directory containing existing shard_*.vcf or shard_*.vcf.gz inputs"),
    reference_fasta: Optional[Path] = typer.Option(None, "--reference-fasta", help="Genome FASTA for left-aligning indels when calls are split before ANNOVAR; default: the genome entry's reference_fasta"),
):
    """Compatibility alias for `paralome-worker annotate --engine annovar`."""
    result = _run_selected_annotation(
        engine="annovar",
        vcf=vcf,
        out=out,
        genome=genome,
        annovar_dir=annovar_dir,
        annovar_db=annovar_db,
        annovar_buildver=annovar_buildver,
        annovar_protocol=annovar_protocol,
        annovar_operation=annovar_operation,
        snpeff_jar=None,
        snpeff_config=None,
        snpeff_genome=None,
        snpeff_memory="8g",
        prefix=prefix,
        backend=backend,
        submit=submit,
        dry_run=dry_run,
        partition=partition,
        mem=mem,
        threads=threads,
        jobs=jobs,
        variant_workdir=variant_workdir,
        shard_vcf_dir=shard_vcf_dir,
        reference_fasta=reference_fasta,
    )
    _print_result(result)
