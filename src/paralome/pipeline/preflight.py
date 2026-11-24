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

"""Everything a run needs, checked on the login node before anything is queued.

The checks run at the start of every run rather than as a separate command:
they confirm that all the pieces are there, samples are in place and options
are set appropriately. Nobody has to remember to run them, and they cannot
drift from what the run actually does, because they ARE the run.

Fail HERE, in the foreground, where the user is standing -- not on a compute
node one poll cycle later. The sample sheet is opened and read: a mistyped
FASTQ path on row 47 is found here, not by the alignment stage after FASTQ QC
has run. ``validate_fastqs`` checks presence, zero bytes and duplicate sample
ids, collecting every fault rather than the first.

EVERY PROBLEM, NOT THE FIRST. A verdict that names one cause invites fixing
that cause and expecting to be done; a preflight that stopped at the first bad
FASTQ would cost a fix-and-resubmit cycle per bad row.

WHAT IS DELIBERATELY NOT CHECKED. Whether a tool is absent altogether is
`setup doctor`'s job and depends on which stages this profile runs -- freebayes
missing is fatal to DNA and irrelevant to scRNA, and a preflight that warned
about it either way would be noise. What IS checked is narrower and
unambiguous: a tool whose path was RECORDED and has since moved, in a stage
this run reaches. That is PLM-0410 arriving on the login node instead of at
the first stage that reaches for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence


@dataclass
class Preflight:
    """What is wrong, gathered rather than raised one at a time."""

    problems: List[str] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)
    # Said, and the run goes ahead -- a stage that will be skipped
    # for a reason the user may want to act on (gene_burden without its
    # gnomAD reference). Never a reason to refuse.
    notes: List[str] = field(default_factory=list)

    def fault(self, text: str) -> None:
        self.problems.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def pass_(self, text: str) -> None:
        self.checked.append(text)

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_sample_sheet(
    report: Preflight,
    samples: Optional[Path],
    *,
    manifest_profile: Optional[str] = None,
    factors: Sequence[str] = (),
    stages: Optional[Sequence[str]] = None,
) -> Optional[list]:
    """Open it, and check what it names actually exists and is not empty.

    Returns the rows it read, or None when there was nothing to read.

    THE GAP THIS CLOSES. `check_input_path(samples)` confirmed the CSV was
    there; nothing read it. A path typo inside it survived until the alignment
    stage read the file on a compute node, one completed FASTQ QC later.
    """
    if samples is None:
        return None
    path = Path(samples)
    if not path.is_file():
        return None  # check_input_path already reports this, with the flag name.

    # Imports outside the try, as in _check_genome: an ImportError here is a defect in Paralome, and
    # must not be reported as "your sample sheet is bad".
    #
    # THE READER THE RUN USES: the sheet is parsed with fastq.manifest, which accepts read1/fq_r1/fastq1
    # and the rest, and accepts a sheet of BAMs -- so the preflight takes exactly the sheets the run takes.
    from paralome.fastq.manifest import detect_manifest_mode, read_sample_inputs

    try:
        # And THE SCHEMA THE RUN USES. The run's first stage reads
        # the sheet with the analysis's column rules; this read it with none,
        # so a sheet missing a required column passed here and stopped the run
        # at FASTQ QC. Only when the caller says which analysis this is.
        rows = read_sample_inputs(
            path,
            manifest_profile=manifest_profile,
            validate_schema=manifest_profile is not None,
            factors=tuple(factors),
        )
    except Exception as exc:
        # A sheet that cannot be parsed is a fault in its own right, and its
        # message names the column that is missing -- see task #14 on why that
        # message currently arrives wearing PLM-0001.
        report.fault(f"{path.name}: {exc}")
        return None

    if not rows:
        report.fault(f"{path.name}: no samples in it")
        return None

    faults = _sample_file_faults(rows)
    # RNA quantification refuses a cohort that mixes single-end and paired-end samples; said here, before
    # anything is trimmed and aligned.
    if stages is not None and "rna_quantification" in stages and detect_manifest_mode(rows) == "fastq_mixed":
        single = [row.sample_id for row in rows if row.input_mode == "fastq_se"]
        faults.append(
            "single-end and paired-end samples are mixed, which RNA quantification "
            "does not support; no second read file is named for: " + ", ".join(single)
        )
    # A sheet naming BAMs for some samples and FASTQs for others.
    # Read QC kept only the FASTQ rows, so alignment, duplicate marking and
    # every stage after them ran without the BAM samples, and read complete.
    if detect_manifest_mode(rows) == "mixed":
        bams = [row.sample_id for row in rows if row.input_mode == "bam"]
        reads = [row.sample_id for row in rows if row.input_mode in ("fastq_pe", "fastq_se")]
        faults.append(
            f"{len(bams)} sample(s) are given as BAMs ({', '.join(bams[:5])}{' ...' if len(bams) > 5 else ''}) and "
            f"{len(reads)} as FASTQs ({', '.join(reads[:5])}{' ...' if len(reads) > 5 else ''}). A project takes "
            "one or the other: the BAM samples would be left out of every stage after read QC. Give every "
            "sample as FASTQs, or every sample as a BAM"
        )
    for fault in faults:
        report.fault(f"{path.name}: {fault}")
    if not faults:
        report.pass_(f"{len(rows)} sample(s) in {path.name}, every file present and non-empty")
    return rows


def _sample_file_faults(rows) -> list[str]:
    """Duplicate IDs, samples naming no input, and files missing or empty."""
    faults: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.sample_id in seen:
            faults.append(f"Duplicate sample_id: {row.sample_id}")
        seen.add(row.sample_id)
        named = [
            (label, value)
            for label, value in (("BAM", row.bam), ("R1", row.fastq_r1), ("R2", row.fastq_r2))
            if value
        ]
        if not (row.bam or row.fastq_r1):
            faults.append(f"{row.sample_id}: no FASTQ and no BAM named")
        for label, value in named:
            candidate = Path(value)
            if not candidate.exists():
                faults.append(f"{row.sample_id}: {label} not found: {candidate}")
            elif candidate.stat().st_size == 0:
                faults.append(f"{row.sample_id}: {label} is empty: {candidate}")
    return faults


#: What each stage reads from the genome. A stage absent from this map reads
#: nothing from it. Annotation is not here because what it needs depends on the
#: engine; see _genome_keys_needed.
STAGE_GENOME_KEYS: dict[str, tuple[str, ...]] = {
    "alignment": ("star_index",),
    "rna_quantification": ("gtf",),
    "macs_peak_calling": ("reference_fasta",),
    "peak_quantification": ("gtf",),
    "motif_analysis_singles": ("reference_fasta",),
    "variants": ("reference_fasta",),
    "singlecell_atac": ("gtf",),
    "singlecell_motif_analysis_singles": ("reference_fasta",),
    "singlecell_rna": ("gtf",),
    # See STAGE_SKIPPED_WITHOUT -- absent is a note, not a fault.
    "gene_burden": ("burden_reference",),
}

#: Genome keys whose absence SKIPS the stage that reads them rather
#: than refusing the run. The gnomAD burden reference is built once per genome
#: by hand (hours on a cluster); a DNA run on a genome without it still calls,
#: annotates and reports, and the gene_burden stage records why it did not run.
STAGE_SKIPPED_WITHOUT: dict[str, str] = {
    "burden_reference": "gene_burden",
}

#: What each annotation engine resolves, as commands/annotate.py requires it.
#:
#: The ANNOVAR build, protocols and operations are required like the
#: paths. They fell back to hg38 and the human databases in the annotation
#: stage, so a genome without them passed here and was annotated as human, or
#: stopped after variant calling.
ANNOTATION_GENOME_KEYS: dict[str, tuple[str, ...]] = {
    "annovar": (
        "annovar_dir", "annovar_db",
        "annovar_buildver", "annovar_protocol", "annovar_operation",
    ),
    "snpeff": ("snpeff_jar", "snpeff_config", "snpeff_genome"),
}

#: The option that supplies a genome value which is not a path, for the fault
#: that says it is missing.
VALUE_OPTIONS: dict[str, str] = {
    "annovar_buildver": "--annovar-buildver",
    "annovar_protocol": "--annovar-protocol, or annotation.annovar_protocol in your configuration",
    "annovar_operation": "--annovar-operation, or annotation.annovar_operation in your configuration",
    "snpeff_genome": "--snpeff-genome",
}


def _genome_keys_needed(
    stages: Optional[Sequence[str]], annotation: Optional[str], aligner: str = "star",
) -> list[tuple[str, str]]:
    """(resource, the first stage that needs it), for the stages this run has."""
    needed: list[tuple[str, str]] = []
    seen: set[str] = set()

    def want(name: str, stage: str) -> None:
        if name not in seen:
            seen.add(name)
            needed.append((name, stage))

    bwa = str(aligner or "star").strip().lower() == "bwa"
    for stage in stages or ():
        if stage == "alignment" and bwa:
            # --aligner bwa: bwa-mem2 against the FASTA (its index is built on the first BWA run).
            want("reference_fasta", stage)
            continue
        for name in STAGE_GENOME_KEYS.get(stage, ()):
            want(name, stage)
        # RNA alignment inserts the annotated junctions, so it reads the GTF
        # too. rna_quantification is in exactly the stage graph that does.
        if stage == "alignment" and "rna_quantification" in stages:
            want("gtf", stage)
        if stage == "annotation":
            for engine in str(annotation or "").lower().replace("-", "").replace("_", "").split(","):
                for name in ANNOTATION_GENOME_KEYS.get(engine.strip(), ()):
                    want(name, stage)
    return needed


def _skip_note(genome: Optional[str], stage: str, why: str) -> str:
    if stage == "gene_burden":
        from paralome.burden.reference import build_command

        how = (f"build it with `{build_command(genome)}`" if genome
               else "it comes from --genome")
    else:  # pragma: no cover - one entry today
        how = f"`paralome setup doctor {genome}` shows the entry"
    label = f"--genome {genome}" if genome else "no --genome"
    return f"{label}: {why}; the {stage} stage will be skipped -- {how}"


def _check_genome(
    report: Preflight,
    genome: Optional[str],
    *,
    stages: Optional[Sequence[str]] = None,
    explicit: Optional[Mapping[str, object]] = None,
    annotation: Optional[str] = None,
    aligner: str = "star",
) -> None:
    """Resolve it, and confirm the files it names are on disk.

    Made a run refuse a genome whose paths do not resolve; this is
    that check, moved to where the person is standing. A half-built genome then
    costs a second rather than a queue slot and a stage.
    """
    given = {
        name: value for name, value in (explicit or {}).items()
        if value not in (None, "")
    }
    if not genome and not given:
        return  # nothing to resolve; requiring --genome is the command's job
    needed = _genome_keys_needed(stages, annotation, aligner)

    # THE IMPORTS ARE OUTSIDE THE try. A name this module cannot import is a programming error and must
    # crash like one, where it names this file, rather than be swallowed by the `except` below and
    # reported as a fault in the user's genome. Only the RESOLUTION stays guarded, because failing to
    # resolve a genome really is something the user can fix.
    from paralome.pipeline.resource_catalog import (
        PATH_RESOURCES, resolve_genome_resources,
    )
    from paralome.setup.genomes import resource_state

    label = f"--genome {genome}" if genome else "no --genome"
    try:
        resolved = resolve_genome_resources(genome=genome or None, overrides=given)
    except Exception as exc:
        report.fault(f"{label}: {exc}")
        return

    # Every path the entry names, whatever this analysis reads, so the genome
    # is confirmed up to date.
    missing = []
    skipped_stages = []
    for name, kind in PATH_RESOURCES.items():
        resource = resolved.resources.get(name)
        if resource is None:
            continue
        if resource_state(resource.value, kind) != "ok":
            if name in STAGE_SKIPPED_WITHOUT:
                skipped_stages.append((name, f"{name} -> {resource.value} is not there"))
                continue
            missing.append(f"{name} -> {resource.value}")

    # And every resource this analysis's stages READ has to be recorded at all (a genome registered
    # without a STAR index is refused for an RNA run here, not by alignment after FASTQ QC).
    def _or_the_option(name: str) -> str:
        # A build or a protocol list is never assumed, so say how
        # to give one.
        option = VALUE_OPTIONS.get(name)
        return f"; record it in the genome entry or pass {option}" if option else ""

    absent = [
        (
            f"{name} is not recorded for it, and the {stage} stage needs it "
            f"(`paralome setup doctor {genome}` shows the entry)" + _or_the_option(name)
            if genome
            else f"the {stage} stage needs {name}, which comes from --genome"
            + (f" or {VALUE_OPTIONS[name]}" if name in VALUE_OPTIONS else "")
        )
        for name, stage in needed
        if name not in resolved.resources and name not in STAGE_SKIPPED_WITHOUT
    ]
    needed_names = {name for name, _ in needed}
    skipped_stages.extend(
        (name, f"{name} is not built for it")
        for name, _ in needed
        if name in STAGE_SKIPPED_WITHOUT and name not in resolved.resources
    )
    for name, why in skipped_stages:
        if name not in needed_names:
            continue  # recorded, gone, and nothing in this run reads it
        stage = STAGE_SKIPPED_WITHOUT[name]
        report.note(_skip_note(genome, stage, why))
    # ANNOVAR pairs its protocol and operation lists entry by entry.
    # Lists of different lengths passed here and stopped the run after variant
    # calling, with an unnumbered ValueError.
    lists = {
        name: [item.strip() for item in resolved.resources[name].value.split(",") if item.strip()]
        for name, _ in needed
        if name in ("annovar_protocol", "annovar_operation") and name in resolved.resources
    }
    if len(lists) == 2 and len(lists["annovar_protocol"]) != len(lists["annovar_operation"]):
        absent.append(
            f"annovar_protocol has {len(lists['annovar_protocol'])} entries "
            f"({','.join(lists['annovar_protocol'])}) and annovar_operation has "
            f"{len(lists['annovar_operation'])} ({','.join(lists['annovar_operation'])}); "
            "ANNOVAR needs one operation for each protocol"
        )
    if missing or absent:
        for item in missing + absent:
            report.fault(f"{label}: {item}")
    elif genome:
        report.pass_(f"--genome {genome} resolves, and its files are there")
    else:
        report.pass_("the genome paths given explicitly are there")


#: Which stage invokes each external tool. A stale recording can only break a
#: run that actually reaches the stage that calls it, so a run is never refused
#: for a tool it would not execute (bcftools, bgzip and tabix are invoked only
#: by variants/freebayes.py, which no RNA stage graph contains).
#:
#: A tool absent from this map is treated as REQUIRED BY EVERYTHING, so adding
#: a new external dependency and forgetting to list it here fails loudly rather
#: than silently skipping its check.
TOOL_STAGES: dict[str, tuple[str, ...]] = {
    "fastp": ("fastq_qc",),
    "STAR": ("alignment",),
    "bwa-mem2": ("alignment",),
    # "deduplication", not "dedup" -- the stage graph's own spelling. A typo
    # here silently makes a tool unreachable, which is the failure direction
    # that lets a genuinely broken run through.
    "samtools": ("alignment", "deduplication", "rna_quantification"),
    "bcftools": ("variants",),
    "bgzip": ("variants",),
    "tabix": ("variants",),
    "freebayes": ("variants",),
    "annovar": ("annotation",),
    "table_annovar.pl": ("annotation",),
    "convert2annovar.pl": ("annotation",),
    "snpEff": ("annotation",),
    # Java runs snpEff and nothing else, and evidome is the
    # functional_evidence stage alone. Unlisted, a moved java -- it is recorded
    # with symlinks resolved, so any OS Java update moves it -- refused RNA,
    # scRNA and ATAC runs that have no stage using it.
    "java": ("annotation",),
    "evidome": ("functional_evidence",),
    # The genome tracks' bigWig and bigBed writers (the genome_tracks stage's).
    "bedGraphToBigWig": ("genome_tracks",),
    "bedToBigBed": ("genome_tracks",),
    # bamUtil: clips the mates' overlap after duplicate marking, in the DNA pipeline only -- keyed
    # to variants, the stage only DNA has, since ATAC marks duplicates too and does not clip.
    "bam": ("variants",),
    "sbatch": (),
    "squeue": (),
    "scancel": (),
    "sacct": (),
}


def _tool_is_reachable(name: str, stages: Optional[Sequence[str]]) -> bool:
    """Would this run ever invoke ``name``?

    ``stages`` of None means the caller could not tell us the stage graph, in
    which case every tool is treated as reachable -- the earlier behaviour,
    which is wrong only in being too strict.
    """
    if stages is None:
        return True
    owners = TOOL_STAGES.get(name)
    if owners is None:
        return True  # unlisted: assume it matters, so a new tool is not skipped
    if not owners:
        return True  # scheduler and always-available tools
    return any(stage in set(stages) for stage in owners)


def _check_recorded_tools(
    report: Preflight, stages: Optional[Sequence[str]] = None
) -> None:
    """A tool whose recorded path has moved, in a stage this run will reach.

    Absence is profile-dependent and `setup doctor`'s job. A STALE RECORDING
    breaks any profile that uses that tool -- recorded paths are authoritative
    -- and "any profile that uses that tool" is the operative clause.

    The whole ParalomeError detail is kept, not just its first line. PLM-0410's
    detail is two lines -- "bcftools was recorded at /usr/bin/bcftools" then
    "That path is gone." -- and the SECOND line is the one that names the
    fault.
    """
    from paralome.binaries import _recorded_executables, relocate_tool, tool_path
    from paralome.errors import ParalomeError

    stale: List[str] = []
    unreachable: List[str] = []
    for name in _recorded_executables():
        try:
            tool_path(name)
        except ParalomeError as exc:
            if exc.code != 410:
                continue
            # A tool that moved is looked for on PATH and re-recorded; only one found nowhere is
            # a fault.
            moved = relocate_tool(name)
            if moved is not None:
                text = (f"{name} moved: recorded at {moved[0]}, found at {moved[1]} on PATH; "
                        "recorded there, and this run uses it")
                report.note(text)
                print(f"[paralome] {text}", flush=True)     # said on every run, not only a dry run's report
                continue
            detail = " ".join(str(exc.detail).split()) + " It is not on this account's PATH either."
            if _tool_is_reachable(name, stages):
                stale.append(detail)
            else:
                unreachable.append(name)

    for item in stale:
        report.fault(item)
    if unreachable:
        # Reported, because a stale recording is still worth repairing -- but
        # it is not this run's problem and must not refuse it.
        report.pass_(
            "recorded paths are stale for "
            + ", ".join(sorted(unreachable))
            + " -- not used by this run's stages; fix with `setup doctor` when convenient"
        )
    if not stale and not unreachable:
        report.pass_("recorded tool paths all resolve")


def _check_shared_group(report: Preflight) -> None:
    """First of all, is this account in the shared group (the shared genome repository's group, not the shared root's).

    Checked before anything else, and a fault like any other missing piece:
    an account outside the group reaches the shared genomes and Evidome
    reference for reading, then fails the first time a run has to add to
    them -- an index, a species, a registered genome.
    """
    from paralome.setup.site import not_in_shared_group_text, shared_group_membership

    membership = shared_group_membership()
    if membership is None:
        return
    if membership["member"]:
        report.pass_(f"account is in group '{membership['group']}' of the shared genome repository")
    else:
        report.fault(not_in_shared_group_text(membership))


def run_preflight(
    *,
    samples: Optional[Path] = None,
    genome: Optional[str] = None,
    stages: Optional[Sequence[str]] = None,
    manifest_profile: Optional[str] = None,
    factors: Sequence[str] = (),
    explicit: Optional[Mapping[str, object]] = None,
    annotation: Optional[str] = None,
    rscript: Optional[str] = None,
    trajectory_fit: bool = False,
    trajectory_method: Optional[str] = None,
    trajectory_model: Optional[str] = None,
    aligner: str = "star",
) -> Preflight:
    """Check a run's inputs and environment. Returns what is wrong, all of it.

    ``stages`` is this run's stage graph, so a stale recording for a tool the
    run never invokes is reported without refusing the run. Omitting it checks
    every recorded tool.

    ``stages`` also decides which genome resources must be recorded,
    ``explicit`` holds the ones given as options (resource name -> value),
    ``annotation`` is the annotation engine chain, and ``manifest_profile`` and
    ``factors`` are the column rules the run's first stage applies to the
    sample sheet.
    """
    report = Preflight()
    _check_shared_group(report)
    rows = _check_sample_sheet(
        report, samples,
        manifest_profile=manifest_profile, factors=factors, stages=stages,
    )
    # A sheet of BAMs starts after alignment, so the genome is not asked for
    # what read QC and alignment would have read.
    genome_stages = stages
    if stages is not None and rows and all(row.bam for row in rows):
        genome_stages = [stage for stage in stages if stage not in ("fastq_qc", "alignment")]
    # An adopted genome whose files moved takes the machine registry's current entry first.
    if genome:
        try:
            from paralome.setup.genomes import refresh_adopted_genome

            refreshed = refresh_adopted_genome(str(genome))
        except Exception:  # noqa: BLE001 - a refresh that cannot run leaves the check below to report
            refreshed = None
        if refreshed:
            report.note(refreshed)
            print(f"[paralome] {refreshed}", flush=True)
    _check_genome(
        report, genome, stages=genome_stages, explicit=explicit, annotation=annotation, aligner=aligner,
    )
    if str(aligner or "").strip().lower() == "bwa" and "alignment" in (genome_stages or ()):
        _check_bwa(report, genome, explicit)
    # Nor are the tools only read QC and alignment run.
    _check_recorded_tools(report, genome_stages)
    _check_track_tools(report, stages)
    _check_clip_tool(report, stages)
    _check_slingshot(report, stages, rscript, trajectory_fit=trajectory_fit, method=trajectory_method,
                     model=trajectory_model)
    return report


def _check_bwa(report: Preflight, genome: Optional[str] = None,
               explicit: Optional[Mapping[str, object]] = None) -> None:
    """--aligner bwa: bwa-mem2 aligns, and builds the index on the first run -- where this account must be
    able to write it (<FASTA folder>/bwa-mem2/), or the index job fails in the queue."""
    import shutil

    from paralome.align.bwa import BWA_TOOL, index_prefix, index_ready
    from paralome.binaries import resolve_tool

    found = resolve_tool(BWA_TOOL) or shutil.which(BWA_TOOL)
    if found:
        report.pass_(f"{BWA_TOOL} found: {found}")
    else:
        report.fault(f"--aligner bwa runs {BWA_TOOL}, which is not installed or not on PATH")
    fasta = (explicit or {}).get("reference_fasta")
    if not fasta and genome:
        try:
            from paralome.pipeline.resource_catalog import resolve_genome_resources

            fasta = resolve_genome_resources(genome=genome).path("reference_fasta")
        except Exception:  # noqa: BLE001 -- the genome check reports a genome that does not resolve
            fasta = None
    if not fasta or not Path(str(fasta)).is_file():
        return
    fasta = Path(str(fasta))
    if index_ready(fasta):
        report.pass_(f"bwa-mem2 index built: {index_prefix(fasta)}")
        return
    folder = index_prefix(fasta).parent
    where = folder if folder.is_dir() else fasta.parent
    if os.access(where, os.W_OK | os.X_OK):
        report.pass_(f"bwa-mem2 index not built yet; the first job builds it in {folder}")
    else:
        report.fault(
            f"the bwa-mem2 index for {fasta.name} is not built yet, and this account cannot write {where} to "
            f"build it ({folder}). Its owner (or sudo) gives the folder's group write: chmod g+ws {where}")


def _check_slingshot(
    report: Preflight, stages: Optional[Sequence[str]], rscript: Optional[str], *, trajectory_fit: bool = False,
    method: Optional[str] = None, model: Optional[str] = None,
) -> None:
    """A run that reaches the trajectories needs slingshot in the R the extraction runs; said
    before anything is submitted, not after the differential.

    Singlecell_rna fits the lineages when the run names a root, so a run that stops there
    (``trajectory_fit``) needs slingshot too.

    Or Monocle3, the default --trajectory-method.

    And Monocle3 for the trajectories stage's gene models when --trajectory-model is monocle3."""
    if stages is None:
        return
    if "trajectories" not in set(stages) and not (trajectory_fit and "singlecell_rna" in set(stages)):
        return
    import shutil
    import subprocess

    command = rscript or "Rscript"
    where = shutil.which(command) or command
    said = ""
    from paralome.singlecell.trajectories import parse_method

    method = parse_method(method)
    # DelayedMatrixStats too -- slingshot calls it while it fits, so a missing one is said here rather
    # than hours into the stage.
    needed = ("monocle3", "igraph") if method == "monocle3" else ("slingshot", "DelayedMatrixStats")
    from paralome.singlecell.trajectory_genes import parse_model

    gene_engine = parse_model(model)[0] if "trajectories" in set(stages) else None
    if gene_engine == "monocle3" and "monocle3" not in needed:
        needed = (*needed, "monocle3")
    probe = ("for (p in c(" + ", ".join(f'"{name}"' for name in needed) + ")) "
             "cat('PARALOME_PACKAGE', p, requireNamespace(p, quietly = TRUE), '\\n')")
    missing = list(needed)
    try:
        done = subprocess.run([command, "-e", probe], capture_output=True, text=True, timeout=600)
        loaded = {parts[1] for parts in (line.split() for line in done.stdout.splitlines())
                  if len(parts) == 3 and parts[0] == "PARALOME_PACKAGE" and parts[2] == "TRUE"}
        missing = [name for name in needed if name not in loaded]
        said = " | ".join(line for line in (done.stdout + "\n" + done.stderr).strip().splitlines()[-4:] if line)
        said = f"exit {done.returncode}: {said or 'no output'}"
    except subprocess.TimeoutExpired:
        said = "loading them took more than 600 s"
    except OSError as exc:
        said = f"{command} could not be run ({exc})"
    if not missing:
        report.pass_(f"{' and '.join(needed)} load in the R {where} runs, for the trajectories")
    elif method != "monocle3" and missing == ["monocle3"]:
        report.fault(f"monocle3 did not load in the R {where} runs ({said}); the trajectories stage fits its gene "
                     "models with Monocle3's fit_models (--trajectory-model monocle3): in that R, "
                     "remotes::install_github(\"cole-trapnell-lab/monocle3\") (PLM-0418), or --trajectory-model ls")
    elif method == "monocle3":
        report.fault(f"{' and '.join(missing)} did not load in the R {where} runs ({said}); the lineages are fitted "
                     "with Monocle3 (--trajectory-method monocle3, the default): in that R, "
                     "remotes::install_github(\"cole-trapnell-lab/monocle3\") (PLM-0418), or --trajectory-method "
                     "slingshot, or --no-trajectories")
    else:
        # What that R said, and which R it was -- not a bare "not installed".
        report.fault(f"{' and '.join(missing)} did not load in the R {where} runs ({said}); the trajectories stage "
                     "fits its lineages with slingshot: in that R, BiocManager::install(c("
                     + ", ".join(f'\"{name}\"' for name in missing) + ")) (PLM-0417), or --no-trajectories")


def _check_track_tools(report: Preflight, stages: Optional[Sequence[str]]) -> None:
    """A run that reaches genome_tracks needs the UCSC writers; said before anything is submitted,
    not after the differential has run for hours (--no-genome-track runs without them)."""
    if stages is None or "genome_tracks" not in set(stages):
        return
    from paralome.binaries import resolve_tool
    from paralome.errors import ParalomeError

    missing = []
    for name in ("bedGraphToBigWig", "bedToBigBed"):
        try:
            found = resolve_tool(name)
        except ParalomeError:
            found = None            # a stale recording is reported by _check_recorded_tools
        if not found:
            missing.append(name)
    if missing:
        report.fault(f"{' and '.join(missing)} not found (the genome_tracks stage writes its tracks with them): "
                     "conda install -c bioconda ucsc-bedgraphtobigwig ucsc-bedtobigbed, then paralome setup "
                     "(PLM-0416)")
    else:
        report.pass_("bedGraphToBigWig and bedToBigBed are there for the genome tracks")


def _check_clip_tool(report: Preflight, stages: Optional[Sequence[str]]) -> None:
    """A DNA run that marks duplicates clips the mates' overlap with bamUtil: said before anything
    is submitted."""
    if stages is None or "variants" not in set(stages) or not ({"alignment", "deduplication"} & set(stages)):
        return
    from paralome.binaries import resolve_tool
    from paralome.errors import ParalomeError
    from paralome.stream import BAMUTIL_TOOL

    try:
        found = resolve_tool(BAMUTIL_TOOL)
    except ParalomeError:
        found = None
    if not found:
        report.fault("bamUtil's `bam` not found (the DNA pipeline soft-clips the overlap of a pair's mates with "
                     "`bam clipOverlap` after duplicate marking): install bamUtil where every node sees it, put "
                     "`bam` on the PATH (or load its module), then paralome setup (PLM-0416)")
    else:
        report.pass_("bamUtil (bam clipOverlap) is there for the DNA pipeline")


def refuse_if_not_ready(report: Preflight) -> None:
    """Stop the run, naming every fault at once, or return quietly."""
    if report.ok:
        return
    from paralome.errors import ParalomeError

    detail = "\n".join(f"  {problem}" for problem in report.problems)
    raise ParalomeError(
        117,
        f"{len(report.problems)} problem(s) found before submitting:\n{detail}",
    )


def print_preflight(report: Preflight) -> None:
    """What was verified. Printed by a dry run, which is where it is the point.

    A dry run runs every check: the one invocation whose purpose is asking
    "would this work?" verifies, and the checks cost milliseconds and need no
    scheduler.

    Only the PASSES are printed here. The faults follow from PLM-0117, which
    lists every one of them, so printing them twice would just make the real
    answer harder to find.
    """
    print("")
    print("Preflight")
    print("---------")
    for line in report.checked:
        print(f"  OK  {line}")
    for line in report.notes:
        print(f"  NOTE  {line}")
    if report.ok and not report.checked:
        print("  nothing to check -- no --samples and no --genome given")
