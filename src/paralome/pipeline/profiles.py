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

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence


class PipelineProfile(str, Enum):
    """The five analyses Paralome runs, named as the field names them.

    These values are the CLI command names, the strings written into
    project_state.json, and the strings printed back in status output -- one
    spelling the whole way through: scRNA is not scrna.

    So they are CASE-SENSITIVE and have no aliases. `paralome scrna` fails with
    "No such command", which is loud; a silent alias that quietly resolves to
    something else is how two spellings of one concept get established.

    On the naming: sn = single nuclei; sc = single cell. ATAC is a
    nuclei-specific library prep, while RNA can be done on nuclei or whole
    cells. Hence snATAC but scRNA -- the asymmetry is the
    biology, and it means scATAC is not a value anyone can type, because it is
    not an experiment anyone can run.

    DNA covers exome and genome alike: exome versus genome is a property of one
    DNA analysis, not two analyses, and both run the same stage graph. No
    command-line option selects it; note that --genome names the reference
    genome entry, not a whole-genome target.
    """

    DNA = "DNA"
    RNA = "RNA"
    ATAC = "ATAC"
    SNATAC = "snATAC"
    SCRNA = "scRNA"


class DnaTarget(str, Enum):
    """Which DNA analysis is being run. Not yet acted on.

    No command passes it; there is no --exome option, and --genome
    is the reference genome entry. It changes no behaviour -- if an exome
    target should restrict calling
    and filtering to a capture-target BED, that is a scientific change deserving
    its own patch, not something smuggled in under a CLI cleanup.
    """

    EXOME = "exome"
    GENOME = "genome"


class AlignmentProfile(str, Enum):
    DNA = "dna"
    RNA = "rna"


class ManifestProfile(str, Enum):
    FAMILY = "family"
    GENERAL = "general"


# The segregation model applied to family DNA analyses: its own constant, not shared with any
# pipeline-profile name.
DEFAULT_SEGREGATION_PROFILE = "family-segregating"
DEFAULT_PIPELINE_PROFILE = PipelineProfile.DNA
DNA_MISMATCH_FRACTION = 0.04
RNA_MISMATCH_FRACTION = 0.06

# ---------------------------------------------------------------------------
# The four STAR output filters, and what --loose does to them
# ---------------------------------------------------------------------------
#
# The thresholds that decide whether a read is reported as "unmapped: too
# short" are reachable as options, not only through --star-extra.
#
# What they do, from one sample run several ways. Against a 92bp read with
# ~5-8% per-base divergence from its reference assembly (expected for that
# assembly and shallow single-read data, not a defect):
#
#     baseline        0.08 fraction, defaults elsewhere   11.34% unique
#     mismatch_wide   fraction 0.3 AND Nmax 30            15.59% unique
#     loose_score     Score/MatchNminOverLread 0.66->0.3  29.18% unique
#
# WHAT LOOSE COSTS, said out loud because the numbers flatter it. At 0.3 the
# recovered reads aligned 65.5 of 92 bases against 87.1 at baseline -- these
# are partial alignments, not rescued ones: loosening makes worse mappings. It
# is a diagnostic and a last resort for a deliberately cheap library, not a way
# to make a run look better.
# The mismatch fraction alone decides; the absolute cap is given a value no read reaches and is
# not an option (align.star.MISMATCH_NMAX_OFF), --loose included.
DEFAULT_MISMATCH_NMAX = 999
DEFAULT_SCORE_MIN_OVER_LREAD = 0.66  # STAR's own default
DEFAULT_MATCH_NMIN_OVER_LREAD = 0.66  # STAR's own default

LOOSE_MISMATCH_FRACTION = 0.3
LOOSE_MISMATCH_NMAX = 999
LOOSE_SCORE_MIN_OVER_LREAD = 0.3
LOOSE_MATCH_NMIN_OVER_LREAD = 0.3


@dataclass(frozen=True)
class PipelineProfileSpec:
    pipeline_profile: PipelineProfile
    alignment_profile: AlignmentProfile
    manifest_profile: ManifestProfile
    stage_graph: tuple[str, ...]
    variant_pipeline: bool


@dataclass(frozen=True)
class ResolvedPipelineProfile:
    pipeline_profile: PipelineProfile
    segregation_profile: Optional[str]
    alignment_profile: AlignmentProfile
    manifest_profile: ManifestProfile
    stage_graph: tuple[str, ...]
    dna_target: Optional[DnaTarget] = None

    @property
    def is_variant(self) -> bool:
        return self.pipeline_profile is PipelineProfile.DNA

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline_profile": self.pipeline_profile.value,
            "segregation_profile": self.segregation_profile,
            "alignment_profile": self.alignment_profile.value,
            "manifest_profile": self.manifest_profile.value,
            "stage_graph": list(self.stage_graph),
            "dna_target": self.dna_target.value if self.dna_target else None,
        }


_VARIANT_GRAPH = (
    "fastq_qc",
    "alignment",
    "deduplication",
    "variants",
    "annotation",
    "segregation",
    "filtering",
    # Gene burden of the probands against gnomAD, then Evidome on its table -- the DNA pipeline's last
    # two stages.
    "gene_burden",
    # The variant browser's bundle, on every DNA run; before Evidome, which stays last in every pipeline.
    # Genotype_output.
    "genotype_output",
    # The reads FreeBayes used at every called site, sliced from the BAMs (the individual reads
    # that contributed to each variant call), in <prefix>_DNA_reads.zip. --no-genome-track leaves it out.
    "genome_tracks",
    "functional_evidence",
)
_RNA_GRAPH = (
    "fastq_qc",
    "alignment",
    "rna_quantification",
    "differential_expression",
    # Every arm's export for OmicView, the stage before Evidome in every pipeline.
    "statistics_output",
    # Evidome as the final step of every
    # pipeline -- bulk RNA first; scRNA and ATAC follow. See paralome.evidence.
    "functional_evidence",
)
_ATAC_GRAPH = (
    "fastq_qc",
    "alignment",
    "deduplication",
    "macs_peak_calling",
    "peak_merge",
    "peak_quantification",
    "differential_accessibility",
    "motif_analysis_singles",
    "motif_analysis_pairs",
    # The export for OmicView -- the results and the motif enrichment; after the motif
    # stages, whose tables it carries. <prefix>_statistics_output.zip.
    "statistics_output",
    # The genome browser's tracks, once the viable peaks are known, after statistics_output: the
    # statistics are out before the slow part. --no-genome-track leaves it out.
    "genome_tracks",
    # Evidome on every pipeline, not two of five. It reads
    # this pipeline's own differential table and self-skips when Evidome or
    # its data for the species is not installed.
    "functional_evidence",
)
_SNATAC_GRAPH = (
    "singlecell_atac",
    "singlecell_differential",
    "singlecell_motif_analysis_singles",
    "singlecell_motif_analysis_pairs",
    # As bulk ATAC's.
    "statistics_output",
    # As bulk ATAC's.
    "genome_tracks",
    "functional_evidence",
)
# snRNA: hybrid of snATAC up through Diffs only, stopping at the
# differential for now. No motif-analysis stages exist for RNA.
_SCRNA_GRAPH = (
    "singlecell_rna",
    "singlecell_rna_differential",
    # The object's lineages (Slingshot) and the run's statistics along and between them, its own
    # zip (<prefix>_scRNA_trajectory_output.zip); before statistics_output and Evidome, which read its segments too.
    # It needs --trajectory-root <column>:<value>; --no-trajectories leaves it out.
    "trajectories",
    # The results for OmicView (the cell export is the extraction's, in data/).
    "statistics_output",
    "functional_evidence",
)


# WHAT EACH STAGE READS. Some stages (trajectories, say) are separate processes with their own input and
# output rather than links in one linear string: trajectory statistics run in tandem with the differential
# statistics, and need only the trajectories. A stage is ready when the stages it reads are finished, not
# every stage listed before it; the supervisor submits every ready stage at once, and a stage run again
# makes stale only the stages that read it (and theirs). Only scRNA has such a table; a profile without
# one reads as a line, each stage reading the one before it. The graph order stays the order stages are
# listed, shown and started in; a stage reads only stages listed before it.
STAGE_READS: dict[PipelineProfile, dict[str, tuple[str, ...]]] = {
    PipelineProfile.SCRNA: {
        # The extraction, the pseudobulk and the trajectory fit and the cell export with its lineages.
        "singlecell_rna": (),
        "singlecell_rna_differential": ("singlecell_rna",),
        # The segment statistics: singlecell_rna's trajectory fit (and the extraction), nothing of the differential.
        "trajectories": ("singlecell_rna.trajectory_fit",),
        "statistics_output": ("singlecell_rna_differential",),
        # Evidome: the differential's tables and the segments'.
        "functional_evidence": ("singlecell_rna_differential", "trajectories"),
    },
}


# A PART OF A STAGE. The trajectory fit is singlecell_rna's last step (a subfunction of the extraction), but
# only the trajectories read it: a stage reads "singlecell_rna.trajectory_fit" for the fit, and
# "singlecell_rna" for the rest. Fitted again (another --trajectory-fit-cells, or a fit not there),
# singlecell_rna is issued again for that part only, and only what reads the part is stale -- not the
# differential, so Paralome can be run in a loop over fit-cell counts as trials without re-running the
# rest of the pipeline. Each part names the stage options that are its own.
STAGE_PARTS: dict[str, dict[str, tuple[str, ...]]] = {
    "singlecell_rna": {
        "trajectory_fit": ("--trajectory-root", "--trajectory-fit-cells", "--trajectory-dims",
                           "--trajectory-fit-dir", "--trajectory-export-dir", "--trajectory-method",
                           "--trajectory-weld", "--trajectory-max-routes"),
    },
}


def split_read(read: str) -> tuple[str, Optional[str]]:
    """``"stage.part"`` -> ``("stage", "part")``; ``"stage"`` -> ``("stage", None)``."""
    stage, _, part = str(read).partition(".")
    return stage, (part or None)


def blocks(read: str, unfinished: Mapping[str, Optional[frozenset]], blocked: "set[str] | frozenset") -> bool:
    """Whether reading ``read`` waits: its stage is unfinished as a whole, or waits itself, or -- for a part -- that
    part is unfinished. ``unfinished``: stage -> None (the whole) or the parts not finished."""
    stage, part = split_read(read)
    if stage in blocked:
        return True
    if stage not in unfinished:
        return False
    parts = unfinished[stage]
    return parts is None or (part is not None and part in parts)


def blocked_stages(order: Sequence[str], reads: Mapping[str, Sequence[str]],
                   unfinished: Mapping[str, Optional[frozenset]]) -> set[str]:
    """The stages of ``order`` that read, directly or through others, something unfinished."""
    blocked: set[str] = set()
    for name in order:
        if any(blocks(read, unfinished, blocked) for read in reads.get(name, ())):
            blocked.add(name)
    return blocked


def stage_reads(pipeline_profile: "str | PipelineProfile | None", graph: Optional[Sequence[str]] = None
                ) -> dict[str, tuple[str, ...]]:
    """{stage: the stages it reads} for the profile's graph (or ``graph``); a line where there is no table."""
    canonical = normalize_pipeline_profile(pipeline_profile)
    order = [str(name) for name in (graph if graph else PROFILE_REGISTRY[canonical].stage_graph)]
    table = STAGE_READS.get(canonical)
    if table is None:
        return {name: tuple(order[index - 1:index]) for index, name in enumerate(order)}
    return {name: tuple(read for read in table.get(name, ()) if split_read(read)[0] in order) for name in order}


def stage_ancestors(reads: Mapping[str, Sequence[str]], name: str) -> set[str]:
    """Every stage ``name`` reads, and every stage those read."""
    seen: set[str] = set()
    todo = [split_read(read)[0] for read in reads.get(name, ())]
    while todo:
        stage = todo.pop()
        if stage not in seen:
            seen.add(stage)
            todo.extend(split_read(read)[0] for read in reads.get(stage, ()))
    return seen


PROFILE_REGISTRY: dict[PipelineProfile, PipelineProfileSpec] = {
    # One DNA analysis; exome versus genome is not a separate pipeline:
    # the former variant-exome and variant-genome profiles resolved to byte
    # identical specs and nothing in the package read the difference.
    PipelineProfile.DNA: PipelineProfileSpec(
        PipelineProfile.DNA,
        AlignmentProfile.DNA,
        ManifestProfile.FAMILY,
        _VARIANT_GRAPH,
        True,
    ),
    PipelineProfile.RNA: PipelineProfileSpec(
        PipelineProfile.RNA,
        AlignmentProfile.RNA,
        ManifestProfile.GENERAL,
        _RNA_GRAPH,
        False,
    ),
    PipelineProfile.ATAC: PipelineProfileSpec(
        PipelineProfile.ATAC,
        AlignmentProfile.DNA,
        ManifestProfile.GENERAL,
        _ATAC_GRAPH,
        False,
    ),
    # snATAC has no fastq/bam sample manifest at all -- its input is a single
    # Seurat/Signac .rds object (--rds). manifest_profile is unused for this
    # profile (build_project_state/build_dispatch_plan are only invoked when
    # --samples is set, which snATAC never sets), GENERAL is just an inert
    # placeholder to satisfy the dataclass field.
    PipelineProfile.SNATAC: PipelineProfileSpec(
        PipelineProfile.SNATAC,
        AlignmentProfile.DNA,
        ManifestProfile.GENERAL,
        _SNATAC_GRAPH,
        False,
    ),
    # scRNA shares snATAC's "no fastq/bam manifest, single Seurat .rds input"
    # shape -- see the SNATAC comment above; the same GENERAL placeholder
    # reasoning applies here.
    #
    # Why this is scRNA while its ATAC counterpart is snATAC: sn = single
    # nuclei; sc = single cell. ATAC is a nuclei-specific library prep, while
    # RNA can be done on nuclei or whole cells. The asymmetry is the library chemistry, and it means scATAC is
    # not a value anyone can type because it is not an experiment anyone can
    # run.
    PipelineProfile.SCRNA: PipelineProfileSpec(
        PipelineProfile.SCRNA,
        AlignmentProfile.RNA,
        ManifestProfile.GENERAL,
        _SCRNA_GRAPH,
        False,
    ),
}


# Friendly, memorable names for --stop-at (and any future stage-targeting
# option) that resolve to the canonical stage_graph identifiers above. Some
# aliases are ambiguous across profiles (e.g. "samples" means
# peak_quantification for bulk ATAC but singlecell_atac for snATAC) --
# resolve_stage_alias() picks whichever candidate is actually a member of the
# active profile's stage_graph.
STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "qc": ("fastq_qc",),
    "fastqc": ("fastq_qc",),
    "fastq_qc": ("fastq_qc",),
    "align": ("alignment",),
    "alignment": ("alignment",),
    "bam": ("alignment",),
    "dedup": ("deduplication",),
    "deduplication": ("deduplication",),
    "duplicates": ("deduplication",),
    "peaks": ("macs_peak_calling",),
    "macs": ("macs_peak_calling",),
    "peak_calling": ("macs_peak_calling",),
    "macs_peak_calling": ("macs_peak_calling",),
    "merge": ("peak_merge",),
    "peak_merge": ("peak_merge",),
    "samples": ("peak_quantification", "singlecell_atac", "singlecell_rna"),
    "extract": ("peak_quantification", "singlecell_atac", "singlecell_rna"),
    "quant": ("peak_quantification",),
    "quantification": ("peak_quantification",),
    "cpm": ("peak_quantification",),
    "peak_quantification": ("peak_quantification",),
    "singlecell_atac": ("singlecell_atac",),
    "singlecell_rna": ("singlecell_rna",),
    "diffs": ("differential_accessibility", "singlecell_differential", "differential_expression", "singlecell_rna_differential"),
    "diff": ("differential_accessibility", "singlecell_differential", "differential_expression", "singlecell_rna_differential"),
    "differential": ("differential_accessibility", "singlecell_differential", "differential_expression", "singlecell_rna_differential"),
    "stats": ("differential_accessibility", "singlecell_differential", "singlecell_rna_differential"),
    "differential_accessibility": ("differential_accessibility",),
    "singlecell_differential": ("singlecell_differential",),
    "differential_expression": ("differential_expression",),
    "singlecell_rna_differential": ("singlecell_rna_differential",),
    # "motifs"/"motif" mean "run motif analysis all the way through" -- i.e. stop at the LAST motif stage
    # (pairs).
    "motifs": ("motif_analysis_pairs", "singlecell_motif_analysis_pairs"),
    "motif": ("motif_analysis_pairs", "singlecell_motif_analysis_pairs"),
    "motif_singles": ("motif_analysis_singles", "singlecell_motif_analysis_singles"),
    "motifs_singles": ("motif_analysis_singles", "singlecell_motif_analysis_singles"),
    "singles": ("motif_analysis_singles", "singlecell_motif_analysis_singles"),
    "motif_pairs": ("motif_analysis_pairs", "singlecell_motif_analysis_pairs"),
    "motifs_pairs": ("motif_analysis_pairs", "singlecell_motif_analysis_pairs"),
    "pairs": ("motif_analysis_pairs", "singlecell_motif_analysis_pairs"),
    "motif_analysis_singles": ("motif_analysis_singles",),
    "motif_analysis_pairs": ("motif_analysis_pairs",),
    "singlecell_motif_analysis_singles": ("singlecell_motif_analysis_singles",),
    "singlecell_motif_analysis_pairs": ("singlecell_motif_analysis_pairs",),
    # Backward-compatible aliases for the earlier single combined stage
    # name, in case anyone has --stop-at motif_analysis baked into a script.
    "motif_analysis": ("motif_analysis_pairs", "singlecell_motif_analysis_pairs"),
    "singlecell_motif_analysis": ("singlecell_motif_analysis_pairs",),
    "rna_quant": ("rna_quantification",),
    "rna_quantification": ("rna_quantification",),
    "fpkm": ("rna_quantification",),
    "variants": ("variants",),
    "calling": ("variants",),
    "annotation": ("annotation",),
    "annotate": ("annotation",),
    "segregation": ("segregation",),
    "filtering": ("filtering",),
    "filter": ("filtering",),
    "burden": ("gene_burden",),
    "gene_burden": ("gene_burden",),
    # The stages named for their zips; browser_bundle and its aliases are retired.
    "statistics_output": ("statistics_output",),
    "genotype_output": ("genotype_output",),
    # Again.
    "genome_tracks": ("genome_tracks",),
    "tracks": ("genome_tracks",),
    #.
    "trajectories": ("trajectories",),
    "trajectory": ("trajectories",),
    "evidence": ("functional_evidence",),
    "functional_evidence": ("functional_evidence",),
    "evidome": ("functional_evidence",),
}


def resolve_stage_alias(value: str, stage_graph: tuple[str, ...]) -> str:
    """Resolve a --stop-at value (canonical stage name or a friendly alias)
    to the exact stage identifier used by this profile's stage_graph.

    Raises ValueError with a message listing the valid options for this
    profile if `value` cannot be resolved.
    """
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in stage_graph:
        return normalized
    for candidate in STAGE_ALIASES.get(normalized, ()):
        if candidate in stage_graph:
            return candidate
    raise ValueError(
        f"Unknown stage {value!r} for this pipeline profile. "
        f"Valid stages: {', '.join(stage_graph)} "
        "(friendly aliases also accepted, e.g. samples, diffs, motifs)."
    )


def normalize_pipeline_profile(value: str | PipelineProfile | None) -> PipelineProfile:
    """Resolve a profile name EXACTLY. No case folding, no aliases.

    Profile names map exactly through the CLI (scRNA is not scrna), and there
    are no deprecated spellings: an alias is a second spelling of one concept.
    `paralome scrna` fails loudly with the supported list rather than quietly
    resolving to something.
    """
    if isinstance(value, PipelineProfile):
        return value
    if value is None:
        return DEFAULT_PIPELINE_PROFILE
    text = str(value).strip()
    try:
        return PipelineProfile(text)
    except ValueError as exc:
        supported = ", ".join(profile.value for profile in PipelineProfile)
        raise ValueError(
            f"Unknown pipeline profile {value!r}. Supported profiles are case-sensitive: {supported}"
        ) from exc


def normalize_dna_target(value: str | DnaTarget | None) -> Optional[DnaTarget]:
    if value is None or isinstance(value, DnaTarget):
        return value
    text = str(value).strip()
    try:
        return DnaTarget(text)
    except ValueError as exc:
        supported = ", ".join(target.value for target in DnaTarget)
        raise ValueError(f"Unknown DNA target {value!r}. Supported targets: {supported}") from exc


def resolve_pipeline_profile(
    pipeline_profile: str | PipelineProfile | None,
    segregation_profile: Optional[str] = None,
    dna_target: str | DnaTarget | None = None,
) -> ResolvedPipelineProfile:
    canonical = normalize_pipeline_profile(pipeline_profile)
    spec = PROFILE_REGISTRY[canonical]
    segregation = segregation_profile.strip() if segregation_profile and segregation_profile.strip() else None
    target = normalize_dna_target(dna_target)
    if spec.variant_pipeline:
        segregation = segregation or DEFAULT_SEGREGATION_PROFILE
    else:
        if segregation is not None:
            raise ValueError(
                f"--segregation-profile is only valid for {PipelineProfile.DNA.value}; received {canonical.value}"
            )
        if target is not None:
            raise ValueError(
                f"A DNA target (exome/genome) is only valid for {PipelineProfile.DNA.value}; received {canonical.value}"
            )
    return ResolvedPipelineProfile(
        pipeline_profile=canonical,
        segregation_profile=segregation,
        alignment_profile=spec.alignment_profile,
        manifest_profile=spec.manifest_profile,
        stage_graph=spec.stage_graph,
        dna_target=target,
    )


def default_star_mismatch_fraction(alignment_profile: str | AlignmentProfile) -> float:
    profile = alignment_profile if isinstance(alignment_profile, AlignmentProfile) else AlignmentProfile(str(alignment_profile))
    return RNA_MISMATCH_FRACTION if profile is AlignmentProfile.RNA else DNA_MISMATCH_FRACTION


def star_output_filters(
    alignment_profile: str | AlignmentProfile,
    *,
    loose: bool = False,
    mismatch_fraction: Optional[float] = None,
    mismatch_nmax: Optional[int] = None,
    score_min_over_lread: Optional[float] = None,
    match_nmin_over_lread: Optional[float] = None,
) -> dict[str, float | int]:
    """The four STAR filters this run will use, resolved in one place.

    Precedence is: an explicitly given value, then --loose, then the
    profile default. AN EXPLICIT VALUE ALWAYS WINS, including over --loose --
    otherwise `--loose --star-out-filter-mismatch-nover-read-lmax 0.08` would
    silently ignore the number the user typed.

    Returned as a mapping rather than applied here so the caller can record it:
    a loose run has to be identifiable afterwards from the run's own files, not
    from whoever remembers typing the flag.
    """
    resolved: dict[str, float | int] = {
        "out_filter_mismatch_nover_read_lmax": (
            mismatch_fraction if mismatch_fraction is not None
            else LOOSE_MISMATCH_FRACTION if loose
            else default_star_mismatch_fraction(alignment_profile)
        ),
        "out_filter_mismatch_nmax": (
            mismatch_nmax if mismatch_nmax is not None
            else LOOSE_MISMATCH_NMAX if loose
            else DEFAULT_MISMATCH_NMAX
        ),
        "out_filter_score_min_over_lread": (
            score_min_over_lread if score_min_over_lread is not None
            else LOOSE_SCORE_MIN_OVER_LREAD if loose
            else DEFAULT_SCORE_MIN_OVER_LREAD
        ),
        "out_filter_match_nmin_over_lread": (
            match_nmin_over_lread if match_nmin_over_lread is not None
            else LOOSE_MATCH_NMIN_OVER_LREAD if loose
            else DEFAULT_MATCH_NMIN_OVER_LREAD
        ),
    }
    resolved["loose"] = bool(loose)
    return resolved


def alignment_requires_gtf(alignment_profile: str | AlignmentProfile) -> bool:
    profile = alignment_profile if isinstance(alignment_profile, AlignmentProfile) else AlignmentProfile(str(alignment_profile))
    return profile is AlignmentProfile.RNA
