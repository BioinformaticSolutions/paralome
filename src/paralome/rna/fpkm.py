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

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from paralome.csvio import skip_comment_lines
from paralome.errors import ParalomeError
from paralome.pipeline.jsonio import (
    checkpoint_stale_reason,
    input_fingerprint,
    inputs_changed_reason,
    invalidate_checkpoint,
    missing_checkpoint_outputs,
    write_json,
)
from paralome.rna.gene_model import GeneModel, parse_gtf_left_out
from paralome.rna.native_counter import (
    STRANDEDNESS_ASSIGNMENT_RULES,
    STRANDEDNESS_LABELS,
    count_gene_fragments,
)
from paralome.rna.preflight import preflight_rna_quantification, project_bam_index_dir

NATIVE_COUNTING_ALGORITHM = "paralome.native_union_exon_assignment.v1"

#: How the FPKM library size (the per-million denominator) is
#: counted. v1 counted every primary mapped read1 of a paired run, orphans
#: (a mate unmapped or missing) and non-proper pairs included, while the
#: fragment counter discards them; v2 is the counter's own fragment total, so
#: numerator and denominator follow one rule. A result recorded under another
#: rule (or none) is recomputed, not reused -- see quantification_stale_reason.
LIBRARY_SIZE_RULE = "paralome.library_size.counted_fragments.v2"


@dataclass(frozen=True)
class RnaSample:
    sample_id: str
    bam: Path
    condition: str = "unknown"


def _read_sample_rows(path: Path) -> List[RnaSample]:
    """The sample sheet as written, without asking whether its BAMs exist."""
    rows: List[RnaSample] = []
    # Utf-8-sig, so a sheet saved by Excel with a byte-order mark
    # does not lose its first column ("missing column(s): sample_id").
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(skip_comment_lines(handle))
        fields = set(reader.fieldnames or [])
        missing = {"sample_id", "bam"} - fields
        if missing:
            raise ValueError(f"RNA sample sheet missing column(s): {', '.join(sorted(missing))}")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            bam_text = (row.get("bam") or "").strip()
            if not sample_id or not bam_text:
                raise ValueError("RNA sample sheet contains an empty sample_id or BAM path")
            rows.append(
                RnaSample(
                    sample_id=sample_id,
                    bam=Path(bam_text).expanduser().resolve(),
                    condition=(row.get("condition") or row.get("group") or "unknown").strip() or "unknown",
                )
            )
    if not rows:
        raise ValueError(f"RNA sample sheet contains no samples: {path}")
    duplicate_ids = sorted({sample.sample_id for sample in rows if sum(x.sample_id == sample.sample_id for x in rows) > 1})
    if duplicate_ids:
        raise ValueError("Duplicate RNA sample_id values: " + ",".join(duplicate_ids))
    return rows


def _read_samples(path: Path) -> List[RnaSample]:
    rows = _read_sample_rows(path)
    for sample in rows:
        if not sample.bam.exists():
            raise FileNotFoundError(f"BAM not found for {sample.sample_id}: {sample.bam}")
    return rows


def _request_fingerprint(samples_path: Path, gtf_path: Path) -> str:
    """What was ASKED for: the sheet's rows and the GTF, as paths, not as files.

    Unlike ``input_fingerprint`` this needs no file to exist, so it
    can tell "an input of this result has since been deleted" (same request)
    from "the sheet now names something else" (a different one). Paths are
    made absolute without following links, which a deleted target would change.
    """
    def named(path) -> str:
        return os.path.abspath(os.path.expanduser(str(path)))

    with Path(samples_path).open(newline="", encoding="utf-8-sig") as handle:
        written = {
            (row.get("sample_id") or "").strip(): named((row.get("bam") or "").strip())
            for row in csv.DictReader(skip_comment_lines(handle))
        }
    rows = _read_sample_rows(Path(samples_path))
    return input_fingerprint(content={
        "gtf": named(gtf_path),
        "samples": [[row.sample_id, row.condition, written[row.sample_id]] for row in rows],
    })


class BamNotReadable(Exception):
    """A BAM that could not be read through; its one argument is the sentence.

    Raised by the counting of one sample, which may be in a worker
    process: it has to survive being pickled back, so it carries only text.
    """


def _count_one_sample(task: Tuple) -> Tuple[List[float], int, int, int]:
    """One sample's counts, in gene order. Module-level so a worker process can run it."""
    bam, genes, paired_end, strandedness, require_both_ends_mapped, exclude_chimeric_fragments = task[:6]
    sample_id = task[6] if len(task) > 6 else Path(bam).name
    # A truncated BAM ended the stage with "OSError: no BGZF EOF
    # marker; file may be truncated" and neither the sample nor the file.
    try:
        gene_counts, eligible, assigned = count_gene_fragments(
            bam,
            genes,
            paired_end=paired_end,
            strandedness=strandedness,
            require_both_ends_mapped=require_both_ends_mapped,
            exclude_chimeric_fragments=exclude_chimeric_fragments,
        )
        # The library size is the fragments the counter itself
        # considered (``eligible``), under its rules: primary, mapped, and in
        # a paired run both mates mapped (--require-both-ends-mapped) as a
        # proper pair on one chromosome (--exclude-chimeric-fragments). It was
        # a second pass counting every primary mapped read1, so orphan reads
        # the counter discards still swelled the denominator. See
        # LIBRARY_SIZE_RULE.
        mapped_fragments = eligible
    except OSError as error:
        raise BamNotReadable(
            f"sample {sample_id!r}: the BAM could not be read through ({error}): {bam}"
        ) from error
    return (
        [float(gene_counts[gene.gene_id]) for gene in genes],
        mapped_fragments,
        eligible,
        assigned,
    )


def _count_samples(tasks: Sequence[Tuple], workers: int) -> List[Tuple[List[float], int, int, int]]:
    """Every sample's counts, in the order given, ``workers`` at a time."""
    try:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from concurrent.futures.process import BrokenProcessPool

            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    return list(pool.map(_count_one_sample, tasks))
            except BrokenProcessPool as error:
                # A worker that dies without a word has all but always been killed for memory, and fewer at
                # once is what helps.
                raise ParalomeError(
                    304,
                    f"A counting process was killed while {workers} samples were being counted at "
                    "once, most likely for memory: each sample being counted holds its unmatched "
                    "mates in memory. Run the same command with a lower --jobs (or --jobs 1).",
                    fix="Run the same command with a lower --jobs (or --jobs 1)",
                ) from error
        return [_count_one_sample(task) for task in tasks]
    except BamNotReadable as error:
        # The console shows the detail and not the fix, so both say what to do.
        remedy = ("The file is probably truncated (an alignment or a copy that did not finish): "
                  "make that BAM again, then run the same command")
        raise ParalomeError(105, f"{error}\n{remedy}.", fix=remedy) from error


def _gene_metadata_rows(genes: Sequence[GeneModel]) -> List[Dict[str, str]]:
    """The metadata columns every matrix row starts with.

    ``start`` and ``end`` are the gene model's own coordinates (gene_model.py):
    0-based half-open, as a BED file gives them, so a GTF gene at 1001-2000
    reads ``1000  2000``.
    """
    return [
        {
            "gene_id": gene.gene_id,
            "gene_name": gene.gene_name or gene.gene_id,
            "chromosome": gene.chrom,
            "start": str(gene.start),
            "end": str(gene.end),
            "strand": gene.strand,
            "length_bp": str(gene.length_bp),
        }
        for gene in genes
    ]


GENES_LEFT_OUT_FIELDS = (
    "gene_id", "left_out", "reason", "kept_chromosome", "kept_start", "kept_end", "kept_strand",
)


def write_genes_left_out(
    path: Path,
    metadata: Sequence[Mapping[str, str]],
    skipped_genes: Mapping[str, str],
    partly_kept_genes: Mapping[str, str],
) -> Path:
    """The genes the matrices do not hold whole, as a table beside them.

    One row per gene: left out whole (on both strands of one
    chromosome), or kept on one chromosome without its exons on the others
    (which one: gene_model._kept_chromosome). The kept place is given as the
    matrices give it.
    Written by both backends, with a header alone when there is nothing to say.
    """
    kept = {row["gene_id"]: row for row in metadata if row["gene_id"] in partly_kept_genes}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GENES_LEFT_OUT_FIELDS), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for gene_id in sorted(set(skipped_genes) | set(partly_kept_genes)):
            if gene_id in skipped_genes:
                writer.writerow({"gene_id": gene_id, "left_out": "whole gene", "reason": skipped_genes[gene_id]})
                continue
            place = kept.get(gene_id, {})
            writer.writerow({
                "gene_id": gene_id, "left_out": "exons elsewhere", "reason": partly_kept_genes[gene_id],
                "kept_chromosome": place.get("chromosome", ""), "kept_start": place.get("start", ""),
                "kept_end": place.get("end", ""), "kept_strand": place.get("strand", ""),
            })
    os.replace(temporary, path)
    return path


def genes_left_out_summary(
    skipped_genes: Mapping[str, str], partly_kept_genes: Mapping[str, str], table: Path,
) -> Optional[str]:
    """The one line said about them, or None when every gene is whole."""
    parts = []
    for genes, what in (
        (skipped_genes, "left out of every matrix (annotated on both strands of a chromosome)"),
        (partly_kept_genes, "counted on one chromosome only, the primary one where the names "
                            "tell (their exons on other chromosomes are left out)"),
    ):
        if genes:
            first = sorted(genes)[:5]
            parts.append(
                f"{len(genes)} gene(s) are {what}: {', '.join(first)}"
                + (", ..." if len(genes) > len(first) else "")
            )
    if not parts:
        return None
    return f"[paralome] rna_fpkm_quantification: {'; '.join(parts)}. All are listed in {table}."


def quantification_stale_reason(
    previous: Mapping[str, object],
    *,
    strandedness: int,
    paired_end: bool,
    pseudocount: float,
    require_both_ends_mapped: bool,
    exclude_chimeric_fragments: bool,
    fingerprint: str,
    request: str,
    stamps: Optional[Mapping[str, Optional[str]]] = None,
) -> Optional[str]:
    """Why a finished quantification does not answer this request, or ``None``.

    The decision ``quantify_fpkm`` makes, as a function, so the SLURM
    plan makes the same one. ``fingerprint`` is "" when an input
    cannot be read; ``request`` then decides -- see _request_fingerprint --
    and ``stamps`` (quantification_input_stamps) still compares, one by one,
    the input files that are there.
    """
    expected: Dict[str, object] = {
        "strandedness": strandedness, "paired_end": paired_end, "pseudocount": pseudocount,
    }
    # The SLURM finalize did not record these two before this
    # version. A key the checkpoint does not have cannot be compared, and
    # "changed from None to True" resubmitted every finished SLURM result.
    for key, value in (
        ("require_both_ends_mapped", require_both_ends_mapped),
        ("exclude_chimeric_fragments", exclude_chimeric_fragments),
    ):
        if key in previous:
            expected[key] = value
    stale_reason = checkpoint_stale_reason(previous, **expected)
    # The FPKM of a result whose library size was counted another
    # way differs (slightly, for a paired run with orphan reads) from what
    # this version computes, so it is made again -- see LIBRARY_SIZE_RULE.
    if stale_reason is None and previous.get("library_size_rule") != LIBRARY_SIZE_RULE:
        stale_reason = (
            "the FPKM library size is now counted with the fragment counter's own rules "
            "(orphan reads and discarded pairs left out), and this result was not"
        )
    # A result that records no fingerprint (every SLURM result of
    # And anything older) cannot vouch for its inputs, and was reused
    # whatever had happened to the BAMs, the sheet or the GTF.
    if stale_reason is None and not previous.get("input_fingerprint"):
        stale_reason = NO_FINGERPRINT_REASON
    if stale_reason is None and fingerprint:
        stale_reason = inputs_changed_reason(previous, fingerprint)
    recorded_request = previous.get("request_fingerprint")
    if stale_reason is None and not fingerprint and recorded_request and recorded_request != request:
        stale_reason = (
            "the sample sheet or --gtf no longer names what this result "
            "was computed from, and cannot be read as given"
        )
    # Each input that IS there is compared with what was recorded for it, so one missing BAM does not
    # switch the comparison off for the others; a checkpoint without the per-file record is judged on
    # its parameters.
    recorded_stamps = previous.get("input_stamps")
    if stale_reason is None and not fingerprint and stamps and isinstance(recorded_stamps, Mapping):
        changed = sorted(
            path for path, stamp in stamps.items()
            if stamp is not None and recorded_stamps.get(path) != stamp
        )
        if changed:
            stale_reason = (
                f"{changed[0]} "
                + (f"and {len(changed) - 1} more input file(s) have" if len(changed) > 1 else "has")
                + " changed since this result was computed"
            )
    # Parameters matching is not enough -- the outputs have to still be there. Results are often deleted
    # and re-run, and must not be answered with a checkpoint and no files on disk.
    gone = missing_checkpoint_outputs(previous, (
        "gene_fragment_counts", "gene_fpkm", "gene_log2_fpkm",
        "sample_expression_qc",
    ))
    if gone:
        stale_reason = (
            f"{len(gone)} recorded output(s) no longer exist, "
            f"e.g. {gone[0]}"
        )
    return stale_reason


#: Why a finished result with no input fingerprint is made again.
NO_FINGERPRINT_REASON = (
    "this result was made by an older Paralome that did not record what it was made from"
)


def quantification_input_stamps(samples_path: Path, gtf_path: Path) -> Optional[Dict[str, Optional[str]]]:
    """Each input file on its own: ``{path: "size|mtime"}``, None for one that is not there.

    The combined fingerprint cannot be had while ANY input is
    missing; this still says which of the others have changed. None when the
    sheet itself cannot be read.
    """
    try:
        paths = [Path(gtf_path).expanduser().resolve(),
                 *(sample.bam for sample in _read_sample_rows(Path(samples_path)))]
    except (OSError, ValueError):
        return None
    stamps: Dict[str, Optional[str]] = {}
    for path in paths:
        try:
            stat = path.stat()
            stamps[str(path)] = f"{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            stamps[str(path)] = None
    return stamps


def kept_result(previous, stamps: Optional[Mapping[str, Optional[str]]]):
    """``previous``, a result that is kept, after naming any input of it that is gone."""
    missing = sorted(path for path, stamp in (stamps or {}).items() if stamp is None)
    if missing:
        print(
            "[paralome] rna_fpkm_quantification: the finished result is kept although "
            f"{len(missing)} of its input file(s) no longer exist: {', '.join(missing)}",
            file=sys.stderr,
        )
    return previous


def quantification_fingerprints(samples_path: Path, gtf_path: Path) -> Tuple[str, str]:
    """(input fingerprint, request fingerprint); either is "" when it cannot be had."""
    # What this result is computed FROM -- see jsonio.input_fingerprint.
    # The sample sheet is read here (plain CSV, nothing imported) so that a
    # sample commented out, a different BAM or another GTF is a change.
    try:
        fingerprint_samples = _read_samples(Path(samples_path))
        fingerprint = input_fingerprint(
            stamped=[Path(gtf_path), *(sample.bam for sample in fingerprint_samples)],
            content=[[sample.sample_id, sample.condition] for sample in fingerprint_samples],
        )
    except (OSError, ValueError):
        fingerprint = ""        # an unreadable sheet is preflight's to report
    # "Cannot fingerprint" does not mean "reuse the old result": a sample added with a mistyped BAM path, a
    # duplicate sample_id or a GTF that does not exist must not return the finished result. Only a request
    # identical to the recorded one may keep a result whose input file has gone.
    try:
        request = _request_fingerprint(Path(samples_path), Path(gtf_path))
    except (OSError, ValueError, KeyError):
        request = ""
    return fingerprint, request


def _write_matrix(path: Path, metadata: Sequence[Dict[str, str]], values: Dict[str, Sequence[float]]) -> None:
    sample_ids = list(values)
    fields = ["gene_id", "gene_name", "chromosome", "start", "end", "strand", "length_bp", *sample_ids]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, record in enumerate(metadata):
            row = dict(record)
            for sample_id in sample_ids:
                value = values[sample_id][index]
                row[sample_id] = f"{value:.10g}"
            writer.writerow(row)


def quantify_fpkm(
    samples_path: Path,
    gtf_path: Path,
    output_dir: Path,
    *,
    threads: int = 4,
    jobs: int = 0,
    strandedness: int = 0,
    paired_end: bool,
    require_both_ends_mapped: bool = True,
    exclude_chimeric_fragments: bool = True,
    pseudocount: float = 1.0,
    overwrite: bool = False,
    index_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Quantify gene expression natively and make FPKM the primary RNA-seq matrix.

    Reads are assigned to genes directly from BAM alignment blocks and a
    GTF-derived union-exon gene model -- no external tool (featureCounts,
    Subread, Cufflinks) is required. FPKM is calculated from assigned gene
    fragments, union-exon gene length, and the number of fragments the
    counter considered in each BAM (under the counter's own rules,
    so orphan reads of a paired run are left out of both -- see
    LIBRARY_SIZE_RULE).

    ``paired_end`` has no default: callers must resolve single-end versus
    paired-end from the established sample manifest before calling this
    function, rather than relying on an assumed default.

    ``jobs`` is how many samples are counted at once (0: as many as
    ``threads`` allows). The result does not depend on it.

    ``index_dir`` is where a BAM in an unwritable folder has its
    index made or found (default: the project's inputs/bam_indexes under
    ``output_dir``). A SLURM sample task passes the submitting project's
    folder, where the login node already indexed it.
    """
    if strandedness not in {0, 1, 2}:
        raise ValueError("strandedness must be 0, 1, or 2")
    if threads < 1:
        raise ValueError("threads must be at least 1")
    if jobs < 0:
        raise ValueError("jobs cannot be negative")
    # 0 was accepted, and failed with "math domain error" at
    # log2(0 + 0) only after every BAM had been counted.
    if not pseudocount > 0:
        raise ValueError(
            f"pseudocount must be greater than 0 (got {pseudocount!r}): "
            "log2(FPKM + pseudocount) has no value for a gene with no reads otherwise"
        )

    out = Path(output_dir).expanduser().resolve()
    inputs = out / "inputs"
    results = out / "results"
    machine = results / "machine"
    logs = out / "logs"
    checkpoints = out / "checkpoints"
    for path in (inputs, machine, logs, checkpoints):
        path.mkdir(parents=True, exist_ok=True)

    # The parameter comparison runs *before* preflight, and before touching any BAM/GTF at all -- a cache
    # hit needs none of that validation, since nothing is about to be read. A matching cached checkpoint
    # can be returned even in an environment where pysam isn't importable.
    #
    # Every parameter that affects this stage's output is compared against what the existing checkpoint
    # recorded (a new --strandedness value, say): a real parameter change is treated as equivalent to
    # ``--overwrite`` for this stage, with no action required from the user beyond running the command
    # they meant to run.
    checkpoint_path = checkpoints / "rna_fpkm_quantification.json"
    fingerprint, request = quantification_fingerprints(Path(samples_path), Path(gtf_path))
    stamps = quantification_input_stamps(Path(samples_path), Path(gtf_path))
    # The finished result this run replaces, and why. It is withdrawn
    # after preflight, not before it: a request preflight refuses (a mistyped
    # BAM path) overwrites nothing, so the finished result it was aimed at is
    # still a finished result. --overwrite is no exception.
    replaced: Optional[Tuple[Dict[str, object], str]] = None
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text())
        if previous.get("complete") and overwrite:
            replaced = (previous, "--overwrite")
        elif previous.get("complete"):
            stale_reason = quantification_stale_reason(
                previous,
                strandedness=strandedness,
                paired_end=paired_end,
                pseudocount=pseudocount,
                require_both_ends_mapped=require_both_ends_mapped,
                exclude_chimeric_fragments=exclude_chimeric_fragments,
                fingerprint=fingerprint,
                request=request,
                stamps=stamps,
            )
            if stale_reason is None:
                return kept_result(previous, stamps)
            print(
                f"[paralome] rna_fpkm_quantification: {stale_reason}; "
                "recomputing instead of reusing the existing checkpoint.",
                file=sys.stderr,
            )
            replaced = (previous, stale_reason)

    preflight_rna_quantification(
        samples_path,
        gtf_path,
        strandedness=strandedness,
        paired_end=paired_end,
        # Where a BAM in an unwritable folder gets its index.
        index_dir=(Path(index_dir).expanduser().resolve() if index_dir is not None
                   else project_bam_index_dir(out)),
    )
    if replaced is not None:
        invalidate_checkpoint(checkpoint_path, *replaced)

    samples = _read_samples(Path(samples_path))
    # Genes on both strands are left out; a gene on several
    # chromosomes is kept on one. Said once, and written out below.
    genes, skipped_genes, partly_kept_genes = parse_gtf_left_out(gtf_path)
    left_out_path = machine / "genes_left_out.tsv"
    summary = genes_left_out_summary(skipped_genes, partly_kept_genes, left_out_path)
    if summary:
        print(summary, file=sys.stderr)

    sample_ids = [sample.sample_id for sample in samples]
    metadata = _gene_metadata_rows(genes)

    counts: Dict[str, List[float]] = {sample_id: [] for sample_id in sample_ids}
    library_fragments: Dict[str, int] = {}
    assignment_stats: Dict[str, Dict[str, int]] = {}
    # Samples are independent, so they are counted --jobs at a time (never more than --threads), and
    # collected in sheet order so the files do not depend on it.
    tasks = [
        (sample.bam, genes, paired_end, strandedness, require_both_ends_mapped, exclude_chimeric_fragments,
         sample.sample_id)
        for sample in samples
    ]
    workers = min(jobs or len(samples), len(samples), threads)
    counted = _count_samples(tasks, workers)
    for sample, (gene_counts, mapped_fragments, eligible, assigned) in zip(samples, counted):
        counts[sample.sample_id] = gene_counts
        library_fragments[sample.sample_id] = mapped_fragments
        assignment_stats[sample.sample_id] = {"eligible_fragments": eligible, "assigned_fragments": assigned}

    fpkm: Dict[str, List[float]] = {sample_id: [] for sample_id in sample_ids}
    log2_fpkm: Dict[str, List[float]] = {sample_id: [] for sample_id in sample_ids}
    for index, gene in enumerate(genes):
        length_bp = float(gene.length_bp)
        if length_bp <= 0:
            raise RuntimeError(f"Non-positive union-exon length for gene {gene.gene_id}")
        for sample_id in sample_ids:
            denominator = length_bp * float(library_fragments[sample_id])
            value = 0.0 if denominator <= 0 else counts[sample_id][index] * 1_000_000_000.0 / denominator
            fpkm[sample_id].append(value)
            log2_fpkm[sample_id].append(math.log2(value + pseudocount))

    counts_path = machine / "gene_fragment_counts.tsv"
    fpkm_path = machine / "gene_fpkm.tsv"
    log2_path = machine / "gene_log2_fpkm.tsv"
    _write_matrix(counts_path, metadata, counts)
    _write_matrix(fpkm_path, metadata, fpkm)
    _write_matrix(log2_path, metadata, log2_fpkm)
    write_genes_left_out(left_out_path, metadata, skipped_genes, partly_kept_genes)

    qc_path = machine / "sample_expression_qc.tsv"
    with qc_path.open("w", newline="") as handle:
        fields = [
            "sample_id", "condition", "bam", "mapped_primary_fragments",
            "assigned_gene_fragments", "assignment_fraction", "detected_genes",
            "median_nonzero_fpkm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            assigned = assignment_stats[sample.sample_id]["assigned_fragments"]
            mapped = library_fragments[sample.sample_id]
            nonzero = [value for value in fpkm[sample.sample_id] if value > 0]
            median = 0.0
            if nonzero:
                ordered = sorted(nonzero)
                middle = len(ordered) // 2
                median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "condition": sample.condition,
                    "bam": str(sample.bam),
                    "mapped_primary_fragments": mapped,
                    "assigned_gene_fragments": f"{assigned:.10g}",
                    "assignment_fraction": f"{(assigned / mapped) if mapped else 0.0:.10g}",
                    "detected_genes": sum(value > 0 for value in counts[sample.sample_id]),
                    "median_nonzero_fpkm": f"{median:.10g}",
                }
            )

    checkpoint = {
        # V4, the library size counted by LIBRARY_SIZE_RULE.
        "schema": "paralome.rna.fpkm_quantification.v4",
        "stage": "rna_fpkm_quantification",
        "complete": True,
        "quantification_model": "FPKM",
        "counting_backend": "native",
        "counting_algorithm": NATIVE_COUNTING_ALGORITHM,
        "library_size_rule": LIBRARY_SIZE_RULE,
        "sample_count": len(samples),
        "gene_count": len(metadata),
        "paired_end": paired_end,
        "strandedness": strandedness,
        "strandedness_label": STRANDEDNESS_LABELS[strandedness],
        "strand_assignment_rule": STRANDEDNESS_ASSIGNMENT_RULES[strandedness],
        "require_both_ends_mapped": require_both_ends_mapped,
        "exclude_chimeric_fragments": exclude_chimeric_fragments,
        "pseudocount": pseudocount,
        "gene_fragment_counts": str(counts_path),
        "gene_fpkm": str(fpkm_path),
        "gene_log2_fpkm": str(log2_path),
        "sample_expression_qc": str(qc_path),
        "library_fragments": library_fragments,
        "assignment_stats": assignment_stats,
        "input_fingerprint": fingerprint,
        "request_fingerprint": request,
        # Each input on its own -- see quantification_input_stamps.
        "input_stamps": stamps,
        "skipped_gene_count": len(skipped_genes),
        "skipped_genes": dict(sorted(skipped_genes.items())),
        "partly_kept_gene_count": len(partly_kept_genes),
        "partly_kept_genes": dict(sorted(partly_kept_genes.items())),
        "genes_left_out": str(left_out_path),
    }
    write_json(checkpoint_path, checkpoint, atomic=True)
    return checkpoint
