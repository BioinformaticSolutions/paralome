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

"""Automatic RNA-seq library strandedness detection .

Getting ``--strandedness`` wrong produces almost completely different
"significant" gene lists rather than just a scale difference. The mechanism:
stranded-library reads that get checked against the wrong strand simply
miss the gene entirely -- true antisense gene overlap is rare in most
genomes -- so aggregate signal collapses toward zero under the wrong
setting and stays high under the correct one. That is the aggregate-FPKM
check an analyst does by eye, and the same principle tools like RSeQC's
``infer_experiment.py`` use.

Design: a small objective panel of about 100 genes, half on each strand,
with no other gene overlapping on the opposite strand, applied to all three
modes on the BAM with the largest count. Decision rule: a truly stranded library
should have one mode clearly outperform the other and come close to the
unstranded ceiling; a truly unstranded library should split the signal
roughly evenly between both stranded modes, with neither winning.

Deliberately does *not* run a full unstranded pass to pick the panel (a
GTF-only, purely structural "no opposite-strand neighbor" filter is enough
and needs no BAM at all), and deliberately reuses raw assigned-fragment
counts rather than full FPKM for the recovery comparison -- the same BAM
and the same panel are used for all three modes, so the FPKM library-size
denominator is identical across all three and cancels out of the
ratio entirely; skipping it avoids a second full-BAM scan just for this
one-time decision, keeping this fast regardless of BAM size (only three
`count_gene_fragments()` passes over the same ~2,000-gene candidate pool,
read back for the ~100-gene panel, not a genome-wide scan for FPKM normalization).
"""

import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from paralome.rna.fpkm import RnaSample, _read_samples
from paralome.rna.gene_model import GeneModel, parse_gtf_genes
from paralome.rna.native_counter import _ChromIndex, _ExonInterval, count_gene_fragments

DEFAULT_PANEL_SIZE_PER_STRAND = 50
#: How many structurally-eligible genes per strand get counted once,
#: unstranded, before the panel is chosen from among them by observed expression.
#:
#: Most genes are silent in any one tissue, so a blind random draw from a
#: ~20,000-gene annotation would land mostly on genes that contribute nothing
#: to any of the three modes, and on a shallow library the survivors would carry
#: a handful of reads each.
#:
#: WHY NOT A HOUSEKEEPING LIST. A symbol list needs gene_name, and Ensembl
#: annotations for non-model organisms carry a large fraction of genes with
#: only an Ensembl identifier, so a symbol-matched panel can come back
#: near-empty on the genome it is needed for. A fixed list is not
#: strand-balanced, and the recovery ratios depend on ~50/50. And it would need
#: curating per species.
#:
#: Ranking by what THIS library actually expressed gets the benefit -- every
#: panel gene guaranteed non-silent in this tissue -- and keeps the properties
#: the structural selection was chosen for: species-agnostic, strand-balanced,
#: free of antisense-overlapping neighbours, and self-correcting for tissue
#: without anyone curating anything.
DEFAULT_CANDIDATE_POOL_PER_STRAND = 1000
DEFAULT_MIN_ASSIGNED_FRAGMENTS = 10
DEFAULT_WIN_RECOVERY = 0.70
DEFAULT_LOSE_RECOVERY = 0.30
DEFAULT_UNSTRANDED_MAX_RECOVERY = 0.65
DEFAULT_UNSTRANDED_RATIO_TOLERANCE = 1.5


def _genes_without_opposite_strand_overlap(genes: Sequence[GeneModel]) -> List[GeneModel]:
    """Keep only genes with no *other* gene overlapping their coordinates on
    the opposite strand -- reuses the same tested interval-overlap index
    ``native_counter.py`` already builds for real fragment assignment, just
    with one gene-envelope interval per gene instead of per-exon intervals,
    since only the gene's overall span matters for "is there strand
    ambiguity here", not its internal exon structure."""
    by_chrom: Dict[str, List[_ExonInterval]] = {}
    for gene in genes:
        by_chrom.setdefault(gene.chrom, []).append(
            _ExonInterval(gene.start, gene.end, gene.gene_id, gene.strand)
        )
    index_by_chrom = {chrom: _ChromIndex.build(intervals) for chrom, intervals in by_chrom.items()}
    clean: List[GeneModel] = []
    for gene in genes:
        hits = index_by_chrom[gene.chrom].overlapping(gene.start, gene.end)
        if all(hit.gene_id == gene.gene_id or hit.strand == gene.strand for hit in hits):
            clean.append(gene)
    return clean


def select_strandedness_detection_panel(
    genes: Sequence[GeneModel],
    *,
    panel_size_per_strand: int = DEFAULT_PANEL_SIZE_PER_STRAND,
    seed: int = 0,
    expression: Optional[Mapping[str, int]] = None,
) -> List[GeneModel]:
    """Pick a deterministic, roughly half-plus-strand/half-minus-strand
    panel of genes with no antisense-overlapping neighbor, straight from a
    parsed GTF -- no BAM, no species-specific gene list, so this works
    identically for mouse, human, sheep, dog, or anything else with a
    reasonably normal gene model.

    A fixed seed (rather than true randomness) makes the panel
    reproducible for the same GTF: two auto-detection runs against the
    same genome should reach the same panel, and therefore the same
    verdict, rather than risking an unlucky draw disagreeing with a
    previous one.
    """
    clean = _genes_without_opposite_strand_overlap(genes)
    plus = [gene for gene in clean if gene.strand == "+"]
    minus = [gene for gene in clean if gene.strand == "-"]

    if expression is not None:
        # Rank by what this library actually expressed. Ties break
        # on gene_id so the panel stays deterministic for the same GTF and BAM,
        # which is the property the fixed seed was protecting.
        def by_expression(gene: GeneModel):
            return (-int(expression.get(gene.gene_id, 0)), gene.gene_id)

        plus_sample = sorted(plus, key=by_expression)[:panel_size_per_strand]
        minus_sample = sorted(minus, key=by_expression)[:panel_size_per_strand]
    else:
        rng = random.Random(seed)
        plus_sample = rng.sample(plus, min(panel_size_per_strand, len(plus))) if plus else []
        minus_sample = rng.sample(minus, min(panel_size_per_strand, len(minus))) if minus else []

    panel = [*plus_sample, *minus_sample]
    panel.sort(key=lambda gene: (gene.chrom, gene.start, gene.gene_id))
    return panel


def select_candidate_pool(
    genes: Sequence[GeneModel],
    *,
    pool_size_per_strand: int = DEFAULT_CANDIDATE_POOL_PER_STRAND,
    seed: int = 0,
) -> List[GeneModel]:
    """The genes counted once, unstranded, to find out which are expressed.

    Structurally eligible and strand-balanced, as the panel is -- just wider,
    so there is something to rank. Still bounded and still not a genome-wide
    pass: one count over ~2,000 genes, once per project, for a decision that
    changes every number downstream of it.
    """
    return select_strandedness_detection_panel(
        genes, panel_size_per_strand=pool_size_per_strand, seed=seed
    )


def _largest_sample(samples: Sequence[RnaSample], index_dir: Optional[Path] = None) -> RnaSample:
    """Strandedness is a library-prep property shared by every sample in a
    project (every sample has the same prep), so checking one representative sample is enough --
    picking whichever has the most mapped reads (via the BAM index's own
    read-count metadata, not a scan) gives the most statistical power for
    the smallest chance of an ambiguous/noisy verdict.

    The index is looked up as preflight finds it, ``index_dir``
    included (where an index Paralome made for a BAM in an unwritable folder
    lives -- see preflight.ensure_bam_indexes), and handed to pysam by path.
    """
    import pysam

    from paralome.rna.preflight import _bam_index_path

    best: Optional[RnaSample] = None
    best_mapped = -1
    for sample in samples:
        index = _bam_index_path(Path(sample.bam), index_dir)
        opened = {"index_filename": str(index)} if index is not None else {}
        with pysam.AlignmentFile(str(sample.bam), "rb", **opened) as bam:
            try:
                mapped = bam.mapped
            except ValueError:
                # The mapped-read total lives in the index. Without
                # one pysam says "mapping information not recorded in index or
                # index not available", which names neither the BAM nor the fix.
                raise ValueError(
                    f"--strandedness auto picks the sample with the most mapped reads from "
                    f"each BAM's index, and sample {sample.sample_id!r} has no usable index: "
                    f"{sample.bam}\nIndex it with `samtools index {sample.bam}`, or pass "
                    "--strandedness 0, 1 or 2."
                ) from None
        if mapped > best_mapped:
            best_mapped = mapped
            best = sample
    assert best is not None  # samples is guaranteed non-empty by the caller
    return best


def _decide_strandedness(
    totals: Dict[int, int],
    *,
    panel_size: int,
    kept_gene_count: int,
    min_assigned_fragments: int,
    win_recovery: float = DEFAULT_WIN_RECOVERY,
    lose_recovery: float = DEFAULT_LOSE_RECOVERY,
) -> Dict[str, object]:
    """Pure decision logic, factored out of :func:`detect_strandedness` so
    it's directly unit-testable against hand-built fragment-count totals
    without needing a real BAM/pysam at all. ``totals`` maps strandedness
    mode (0/1/2) to the summed assigned-fragment count across whichever
    panel genes cleared ``min_assigned_fragments`` in at least one mode.
    """
    total_unstranded = totals[0]
    recovery_1 = (totals[1] / total_unstranded) if total_unstranded else 0.0
    recovery_2 = (totals[2] / total_unstranded) if total_unstranded else 0.0

    verdict: str
    detected: Optional[int]
    if not kept_gene_count or total_unstranded == 0:
        verdict = "inconclusive"
        detected = None
        reason = (
            f"none of the {panel_size} panel genes reached "
            f"{min_assigned_fragments} assigned fragments in any strandedness "
            "mode on the checked sample -- likely too little real signal "
            "at this panel to judge; pass --strandedness explicitly"
        )
    elif recovery_1 >= win_recovery and recovery_2 <= lose_recovery:
        verdict = "stranded"
        detected = 1
        reason = (
            f"mode 1 (first-strand) recovered {recovery_1:.0%} of the "
            f"unstranded signal across {kept_gene_count} expressed panel "
            f"genes; mode 2 recovered only {recovery_2:.0%}"
        )
    elif recovery_2 >= win_recovery and recovery_1 <= lose_recovery:
        verdict = "stranded"
        detected = 2
        reason = (
            f"mode 2 (second-strand) recovered {recovery_2:.0%} of the "
            f"unstranded signal across {kept_gene_count} expressed panel "
            f"genes; mode 1 recovered only {recovery_1:.0%}"
        )
    elif (
        0 < recovery_1 <= DEFAULT_UNSTRANDED_MAX_RECOVERY
        and 0 < recovery_2 <= DEFAULT_UNSTRANDED_MAX_RECOVERY
        and max(recovery_1, recovery_2) / min(recovery_1, recovery_2) <= DEFAULT_UNSTRANDED_RATIO_TOLERANCE
    ):
        verdict = "unstranded"
        detected = 0
        reason = (
            f"mode 1 recovered {recovery_1:.0%} and mode 2 recovered "
            f"{recovery_2:.0%} of the unstranded signal -- neither strand "
            "wins, matching a non-directional library"
        )
    else:
        verdict = "inconclusive"
        detected = None
        reason = (
            f"mode 1 recovered {recovery_1:.0%} and mode 2 recovered "
            f"{recovery_2:.0%} of the unstranded signal -- doesn't clearly "
            "match either a stranded or unstranded signature; pass "
            "--strandedness explicitly rather than risk a wrong guess"
        )

    return {
        "recovery_mode_1": recovery_1,
        "recovery_mode_2": recovery_2,
        "verdict": verdict,
        "detected_strandedness": detected,
        "reason": reason,
    }


def detect_strandedness(
    samples_path: Union[str, Path],
    gtf_path: Union[str, Path],
    *,
    paired_end: bool,
    panel_size_per_strand: int = DEFAULT_PANEL_SIZE_PER_STRAND,
    candidate_pool_per_strand: int = DEFAULT_CANDIDATE_POOL_PER_STRAND,
    min_assigned_fragments: int = DEFAULT_MIN_ASSIGNED_FRAGMENTS,
    win_recovery: float = DEFAULT_WIN_RECOVERY,
    lose_recovery: float = DEFAULT_LOSE_RECOVERY,
    seed: int = 0,
    index_dir: Optional[Path] = None,
    make_missing_indexes: bool = False,
) -> Dict[str, object]:
    """Auto-detect ``--strandedness`` (0/1/2) for one RNA-seq project.

    Runs entirely against a single representative BAM (the sample with the
    most mapped reads) and a small (~100-gene) structurally-unambiguous
    panel selected straight from the GTF -- no full genome-wide pass, no
    per-sample repetition, so this finishes in minutes regardless of cohort
    size.

    Returns a dict always containing ``verdict`` (one of ``"stranded"``,
    ``"unstranded"``, or ``"inconclusive"``), ``detected_strandedness``
    (0, 1, 2, or ``None`` if inconclusive), and a human-readable ``reason``
    -- callers must surface ``reason`` to the user rather than silently
    acting on ``detected_strandedness`` alone, given how consequential
    getting this wrong is.

    With ``make_missing_indexes`` (as ``rna quantify-fpkm`` asks),
    a BAM without an index is indexed first, beside it or under ``index_dir``
    -- see preflight.ensure_bam_indexes -- and refused only when it cannot be.
    """
    samples = _read_samples(Path(samples_path))
    if not samples:
        raise ValueError(f"No samples found in {samples_path}")
    if make_missing_indexes:
        from paralome.rna.preflight import ensure_bam_indexes

        _indexed, index_problems = ensure_bam_indexes(samples, index_dir)
        if index_problems:
            raise ValueError("\n".join(index_problems))

    genes = parse_gtf_genes(gtf_path)
    candidates = select_candidate_pool(
        genes, pool_size_per_strand=candidate_pool_per_strand, seed=seed
    )
    if not candidates:
        raise ValueError(
            "Could not find any genes without an opposite-strand-overlapping "
            "neighbor in this GTF -- cannot auto-detect strandedness from it. "
            "Pass --strandedness explicitly instead."
        )

    sample = _largest_sample(samples, index_dir)

    # Two stages. The unstranded pass over the candidate pool is the
    # SAME count the ceiling is derived from, so ranking by it costs one wider
    # pass rather than an extra one; modes 1 and 2 are read back for the ~100
    # genes that survive. Picking the panel by observed expression is what stops
    # a blind draw from filling it with genes this tissue does not transcribe.
    unstranded_pool_counts, _eligible, _assigned = count_gene_fragments(
        sample.bam, candidates, paired_end=paired_end, strandedness=0,
    )
    panel = select_strandedness_detection_panel(
        candidates,
        panel_size_per_strand=panel_size_per_strand,
        seed=seed,
        expression=unstranded_pool_counts,
    )

    panel_ids = {gene.gene_id for gene in panel}
    assigned_by_mode: Dict[int, Dict[str, int]] = {
        0: {gene_id: count for gene_id, count in unstranded_pool_counts.items()
            if gene_id in panel_ids},
    }
    for mode in (1, 2):
        # Counted over the same candidate pool as mode 0 and then
        # restricted to the panel, so all three modes see the same gene
        # universe (a fragment ambiguous between a panel gene and a pool
        # neighbour is dropped in every mode, not only in mode 0).
        counts, _eligible, _assigned = count_gene_fragments(
            sample.bam, candidates, paired_end=paired_end, strandedness=mode,
        )
        assigned_by_mode[mode] = {
            gene_id: count for gene_id, count in counts.items() if gene_id in panel_ids
        }

    kept_gene_ids = {
        gene.gene_id
        for gene in panel
        if any(
            assigned_by_mode[mode].get(gene.gene_id, 0) >= min_assigned_fragments
            for mode in (0, 1, 2)
        )
    }
    totals = {
        mode: sum(
            count for gene_id, count in assigned_by_mode[mode].items() if gene_id in kept_gene_ids
        )
        for mode in (0, 1, 2)
    }

    decision = _decide_strandedness(
        totals,
        panel_size=len(panel),
        kept_gene_count=len(kept_gene_ids),
        min_assigned_fragments=min_assigned_fragments,
        win_recovery=win_recovery,
        lose_recovery=lose_recovery,
    )

    return {
        "schema": "paralome.rna.strandedness_detection.v1",
        "sample_id": sample.sample_id,
        "sample_bam": str(sample.bam),
        "candidate_pool_size": len(candidates),
        "panel_gene_count": len(panel),
        "panel_gene_count_plus_strand": sum(1 for gene in panel if gene.strand == "+"),
        "panel_gene_count_minus_strand": sum(1 for gene in panel if gene.strand == "-"),
        "panel_genes_above_floor": len(kept_gene_ids),
        "min_assigned_fragments_floor": min_assigned_fragments,
        "assigned_fragments_by_mode": totals,
        **decision,
    }
