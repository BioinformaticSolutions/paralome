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

"""Which options belong to which assay, and under which heading.

Each analysis is its own command (`paralome scRNA`, `paralome DNA`, ...), and
each shows and takes only its own options. The options are declared once, on
``pipeline_run_command`` in ``commands/pipeline.py``; this module says which
assays each one applies to and under which help heading, and the five assay
commands render that one declaration. One declaration, five renderings, so
the commands cannot drift apart.

This table's keys must be EXACTLY the parameters of ``pipeline_run_command``:
an option added without classifying it, or classified after it is gone, is an
error that names it.

HOW APPLICABILITY IS DECIDED. From ``PROFILE_REGISTRY[profile].stage_graph``,
which declares which stages each assay runs: an option that configures a stage
belongs to exactly the profiles that run it:

    DNA      fastq_qc alignment deduplication variants annotation
             segregation filtering reporting gene_burden functional_evidence
    RNA      fastq_qc alignment rna_quantification differential_expression
    ATAC     fastq_qc alignment deduplication macs_peak_calling peak_merge
             peak_quantification differential_accessibility motif_analysis_*
    snATAC   singlecell_atac singlecell_differential
             singlecell_motif_analysis_*
    scRNA    singlecell_rna singlecell_rna_differential

``--alpha``, ``--factor`` and ``--statistic`` are parameters of a differential
test, so every profile but DNA, whose comparison is the pedigree and
``--segregation-profile``. There is no ``--gtf``: --genome resolves it.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Tuple

#: Every assay, in the order the CLI lists them.
EVERY_PROFILE: Tuple[str, ...] = ("DNA", "RNA", "ATAC", "snATAC", "scRNA")

#: The three that start from a sample sheet and go through fastp and STAR.
#: snATAC and scRNA start at pseudobulk, from an object somebody already built.
BULK: Tuple[str, ...] = ("DNA", "RNA", "ATAC")

#: Everything with a differential test. DNA is the exception, and it is not an
#: oversight: its comparison is the pedigree, through --segregation-profile.
DIFFERENTIAL: Tuple[str, ...] = ("RNA", "ATAC", "snATAC", "scRNA")

#: The two that aggregate cells into pseudobulk samples.
SINGLE_CELL: Tuple[str, ...] = ("snATAC", "scRNA")

#: The two that scan motifs. scRNA does not: there are no peaks to scan.
MOTIF: Tuple[str, ...] = ("ATAC", "snATAC")

#: Headings, in the order they should appear on a help screen. Roughly the order
#: a run happens in: what goes in, how it is run, then each stage in turn.
PANEL_ORDER: Tuple[str, ...] = (
    "Inputs",
    "Running it",
    "Read QC",
    "Alignment",
    "Duplicate marking",
    "Variant calling",
    "Variant annotation",
    "Segregation and reporting",
    "Gene burden",
    "Expression quantification and testing",
    "Single-cell sampling",
    "Single-cell RNA",
    "Differential testing",
    "Motifs",
    "Prior knowledge",
    "Genome tracks",
    "Trajectories",
    "Functional evidence",
)

#: panel -> ((parameter name, profiles it applies to), ...)
#:
#: Parameter names, not flag strings, because the generator matches these
#: against ``pipeline_run_command``'s signature and a flag can have aliases
#: (--star-extra is also --extra-star-options) while a parameter cannot.
_TABLE: Tuple[Tuple[str, Tuple[Tuple[str, Tuple[str, ...]], ...]], ...] = (
    ("Inputs", (
        # Mutually exclusive BY PROFILE, which is why neither has to say so in its help.
        ("samples", BULK),
        ("rds", SINGLE_CELL),
        ("project", EVERY_PROFILE),
        ("pipeline_config", EVERY_PROFILE),
        ("genome", EVERY_PROFILE),
    )),
    ("Running it", (
        ("submit", EVERY_PROFILE),
        ("dry_run", EVERY_PROFILE),
        ("overwrite", EVERY_PROFILE),
        ("backend", EVERY_PROFILE),
        ("partition", EVERY_PROFILE),
        ("jobs", EVERY_PROFILE),
        ("threads", EVERY_PROFILE),
        ("stage_mem", EVERY_PROFILE),
        ("poll_seconds", EVERY_PROFILE),
        ("stop_at", EVERY_PROFILE),
        ("max_concurrent_strata", SINGLE_CELL),
    )),
    ("Read QC", (
        ("fastq_qc_workdir", BULK),
        ("fastp_threads", BULK),
        ("fastp_adapter_fasta", BULK),
        ("fastp_cut_right", BULK),
        ("fastp_cut_right_window_size", BULK),
        ("fastp_cut_right_mean_quality", BULK),
        ("fastp_length_required", BULK),
        ("fastp_extra", BULK),
    )),
    ("Alignment", (
        ("align_workdir", BULK),
        ("star_genome", BULK),
        ("star_threads", BULK),
        ("star_read_files_command", BULK),
        ("star_out_reads_unmapped", BULK),
        ("star_out_filter_multimap_nmax", BULK),
        ("star_out_filter_mismatch_nover_read_lmax", BULK),
        ("star_out_filter_score_min_over_lread", BULK),
        ("star_out_filter_match_nmin_over_lread", BULK),
        ("star_loose", BULK),
        ("star_extra", BULK),
        ("aligner", BULK),
        ("bwa_extra", BULK),
    )),
    ("Duplicate marking", (
        # RNA aligns and quantifies without marking duplicates, so these
        # are DNA and bulk ATAC only -- the two profiles whose stage graph
        # holds `deduplication`.
        ("dedup_workdir", ("DNA", "ATAC")),
        ("dedup_threads", ("DNA", "ATAC")),
        # Hidden, and kept so that --mark-duplicates in an existing
        # script is not "No such option" (Click then suggested
        # --remove-duplicates, the opposite). The negative form is refused.
        ("mark_duplicates", ("DNA", "ATAC")),
        ("remove_duplicates", ("DNA", "ATAC")),
        # Fuses FASTQ QC, alignment and duplicate marking. The
        # assays that have all three.
        # RNA too. The mapping is the same STAR call, so the
        # trimming fusion applies unchanged; RNA has no deduplication STAGE,
        # but markdup still runs inside the fused task and writes its marks
        # into the BAM quantification already reads (--mark-in-place).
        ("fuse_stages", ("DNA", "ATAC", "RNA")),
        ("fused_buffer", ("DNA", "ATAC", "RNA")),
    )),
    ("Variant calling", (
        ("variant_workdir", ("DNA",)),
        ("reference", ("DNA",)),
        ("regions", ("DNA",)),
        ("gene_padding", ("DNA",)),
        ("min_alternate_count", ("DNA",)),
        ("min_alternate_fraction", ("DNA",)),
        ("haplotype_length", ("DNA",)),
        ("min_qual", ("DNA",)),
        ("min_depth", ("DNA",)),
        ("min_gq", ("DNA",)),
        ("freebayes_threads", ("DNA",)),
        # The region cap and the stall watchdog of FreeBayes.
        ("max_region_length", ("DNA",)),
        ("stall_minutes", ("DNA",)),
    )),
    ("Variant annotation", (
        ("annotation_workdir", ("DNA",)),
        ("annotation", ("DNA",)),
        ("annovar_dir", ("DNA",)),
        ("annovar_db", ("DNA",)),
        ("annovar_buildver", ("DNA",)),
        ("annovar_protocol", ("DNA",)),
        ("annovar_operation", ("DNA",)),
        ("snpeff_jar", ("DNA",)),
        ("snpeff_config", ("DNA",)),
        ("snpeff_genome", ("DNA",)),
    )),
    ("Segregation and reporting", (
        ("segregation_workdir", ("DNA",)),
        ("segregation_profile", ("DNA",)),
        ("filter_workdir", ("DNA",)),
        # Reporting's --no-workbook; Retired, hidden, refused (PLM-0120).
        ("report_workbook", ("DNA",)),
    )),
    ("Gene burden", (
        # The probands' ancestry for the gnomAD expectation.
        ("ancestry", ("DNA",)),
        # A bolt-on after gene_burden: the variants of the genes named.
        ("extract_damaging_variants", ("DNA",)),
        ("extract_all_variants", ("DNA",)),
    )),
    ("Expression quantification and testing", (
        ("rna_strandedness", ("RNA",)),
        ("rna_pseudocount", ("RNA",)),
        ("rna_block_size", ("RNA",)),
        # --statistic/--factor belong to every pipeline that produces differential statistics, from the
        # one shared parser; bulk ATAC fits the same factorial model with the same code. (The parameter
        # names still say rna_.) One list for every statistics option -- every profile but DNA, whose
        # gene burden is its own statistic. The ANOVA always fits its interaction; no option turns it off.
        ("rna_statistic", DIFFERENTIAL),
        ("rna_factor", DIFFERENTIAL),
        ("rna_pair_by", DIFFERENTIAL),
        ("rna_covariate", DIFFERENTIAL),
        ("rna_covariate_feature", DIFFERENTIAL),
        ("rna_min_expression", ("RNA",)),
        ("rna_min_expression_difference", ("RNA",)),
        ("rna_refine_responders", ("RNA",)),
    )),
    ("Single-cell sampling", (
        # Several of these still say "snATAC:" in their help text while scRNA
        # uses them too. That prose is what applicability was NOT derived from.
        ("replicate_column", SINGLE_CELL),
        ("group_by", SINGLE_CELL),          # Retired, hidden; stops with PLM-0119
        ("stratify_by", SINGLE_CELL),
        ("stratify_value", SINGLE_CELL),
        ("stratify_skip", SINGLE_CELL),
        ("min_cells_per_stratum", SINGLE_CELL),
        ("data_workdir_path", SINGLE_CELL),
        ("remove", SINGLE_CELL),
        ("rscript", SINGLE_CELL),
    )),
    ("Single-cell RNA", (
        ("rna_assay", ("scRNA",)),
        ("gene_symbol_column", ("scRNA",)),
        ("singlecell_rna_differential_workdir", ("scRNA",)),
    )),
    ("Differential testing", (
        ("alpha", DIFFERENTIAL),
        # Single-cell only. RNA and bulk ATAC offered it, checked its
        # value and passed it to no stage; only the two single-cell
        # differential stages take a model.
    )),
    ("Motifs", (
        ("skip_motif", MOTIF),
        ("skip_motif_pairs", MOTIF),
        ("motif_scanner_backend", MOTIF),
    )),
    # The genome_tracks stage's off switch (--no-genome-track, to go on to Evidome faster).
    ("Genome tracks", (
        ("no_genome_track", ("DNA",) + MOTIF),
    )),
    # The trajectories stage, scRNA's for now.
    ("Trajectories", (
        ("trajectory_root", ("scRNA",)),
        ("trajectory_bins", ("scRNA",)),
        ("trajectory_fit_cells", ("scRNA",)),
        ("trajectory_dims", ("scRNA",)),
        ("trajectory_method", ("scRNA",)),
        ("trajectory_weld", ("scRNA",)),
        ("trajectory_max_routes", ("scRNA",)),
        ("trajectory_model", ("scRNA",)),
        ("trajectory_lineages", ("scRNA",)),
        ("no_trajectories", ("scRNA",)),
    )),
    # Functional_evidence is on every graph, so its off switch applies to every profile, under its own
    # heading.
    ("Functional evidence", (
        ("skip_evidence", EVERY_PROFILE),
    )),
)

#: parameter name -> the profiles whose --help offers it.
OPTION_PROFILES: Dict[str, FrozenSet[str]] = {
    name: frozenset(profiles)
    for _panel, entries in _TABLE
    for name, profiles in entries
}

#: parameter name -> the heading it appears under.
OPTION_PANEL: Dict[str, str] = {
    name: panel
    for panel, entries in _TABLE
    for name, _profiles in entries
}

#: Declared on `pipeline run` and deliberately not classified, so no assay command takes them:
#: --profile, because the command name IS the profile (a --profile would let `paralome scRNA
#: --profile DNA` run DNA); --gtf, because --genome resolves it.
RETIRED: FrozenSet[str] = frozenset({"profile", "gtf"})


def options_for(profile: str) -> FrozenSet[str]:
    """Every parameter name ``profile``'s --help should offer."""
    return frozenset(
        name for name, profiles in OPTION_PROFILES.items() if profile in profiles
    )


def panels_for(profile: str) -> List[str]:
    """The headings ``profile`` uses, in PANEL_ORDER, skipping empty ones.

    A DNA help screen has no Motifs heading and an scRNA one has no Alignment
    heading, because an empty box is a question the reader has to answer.
    """
    wanted = options_for(profile)
    return [
        panel for panel in PANEL_ORDER
        if any(OPTION_PANEL[name] == panel for name in wanted)
    ]


def sort_key(name: str) -> Tuple[int, int]:
    """Where an option sits on a help screen: panel order, then table order.

    Declaration order within a panel, which is the order the stage uses them in
    rather than alphabetical -- ``--annovar-dir`` before ``--annovar-db`` before
    ``--annovar-buildver`` reads as a sequence; alphabetised it does not.
    """
    panel = OPTION_PANEL[name]
    within = [n for _p, entries in _TABLE for n, _ in entries
              if OPTION_PANEL[n] == panel]
    return (PANEL_ORDER.index(panel), within.index(name))


def classified() -> FrozenSet[str]:
    """Every parameter name this table accounts for, retired ones included.

    What the coupling test compares against ``pipeline_run_command``'s
    signature.
    """
    return frozenset(OPTION_PROFILES) | RETIRED


def describe(profile: str) -> Iterable[Tuple[str, List[str]]]:
    """(panel, [option names]) for one profile, ready to print or assert on."""
    wanted = options_for(profile)
    for panel in panels_for(profile):
        names = sorted(
            (name for name in wanted if OPTION_PANEL[name] == panel),
            key=sort_key,
        )
        yield panel, names
