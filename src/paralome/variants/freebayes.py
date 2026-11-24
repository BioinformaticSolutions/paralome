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

import hashlib
import json
import os
import re
import shlex
import sys
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from paralome.annotation.annovar import info_id_rename_bash, normalize_for_annotation_bash
from paralome.binaries import worker_command
from paralome.checkpoints import stage_checkpoint_path
from typing import Optional, Sequence

import pysam

from paralome.sample_manifest import ordered_samples, read_sample_manifest
from paralome.pipeline.naming import slurm_job_name
from paralome.pipeline.resources import plan_sample_parallel
from paralome.pipeline.scheduler import detect_scheduler_capabilities, plan_array_tasks


@dataclass
class FreebayesRunResult:
    out_vcf: Path
    workdir: Path
    backend: str
    submitted: bool = False
    job_id: Optional[str] = None
    # For the SLURM backend without --submit (see commands/variants.py).
    already_complete: bool = False
    manual_steps: tuple = ()


@dataclass(frozen=True)
class AnnotationSettings:
    mode: str = "none"
    snpeff_jar: Optional[Path] = None
    snpeff_config: Optional[Path] = None
    # No build, database or protocol of Paralome's own: _resolve_annotation_settings refuses an engine that
    # is asked for without them.
    snpeff_genome: Optional[str] = None
    snpeff_memory: str = "8g"
    annovar_dir: Optional[Path] = None
    annovar_db: Optional[Path] = None
    annovar_buildver: Optional[str] = None
    annovar_protocol: Optional[str] = None
    annovar_operation: Optional[str] = None


@dataclass(frozen=True)
class GeneSpan:
    chrom: str
    start0: int
    end0: int
    gene_id: str
    gene_name: Optional[str] = None


def _progress(message: str, *, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{message}", flush=True)




def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _pinned_tools_bash() -> str:
    """Generated scripts call the tools setup recorded, not PATH's."""
    from paralome.annotation.standalone import _PINNED_TOOLS, recorded_tools_bash

    return recorded_tools_bash(*_PINNED_TOOLS)


def _require_tool(name: str) -> str:
    """The tool to invoke, from what setup recorded -- not from PATH.

    See binaries.resolve_tool: a recorded path that no longer
    resolves raises PLM-0410 rather than quietly falling back, because two
    accounts with different PATHs running different binaries for one command
    is how the same analysis produces two sets of numbers with nothing saying
    it did.
    """
    from paralome.binaries import resolve_tool

    exe = resolve_tool(name)
    if exe is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")
    return exe


def _manifest_bams(samples_csv: Path) -> list[Path]:
    samples = [s for _, s in ordered_samples(read_sample_manifest(samples_csv))]
    if not samples:
        raise ValueError(f"No samples found in manifest: {samples_csv}")

    bams = []
    for sample in samples:
        bam = Path(sample.bam)
        _require_file(bam, f"BAM for sample {sample.sample_id}")
        if _bam_index(bam) is None:
            raise FileNotFoundError(
                f"Missing BAM index for sample {sample.sample_id}: expected one of "
                + ", ".join(str(p) for p in _bam_index_candidates(bam))
                + f". Index it with `samtools index {bam}`."
            )
        bams.append(bam)
    return bams


def _bam_index_candidates(bam: Path) -> list[Path]:
    """Where a BAM's index may be, in the order it is looked for.

    ``x.bam.bai`` (samtools), ``x.bai`` (Picard MarkDuplicates CREATE_INDEX
    and GATK) and ``x.bam.csi`` (samtools index -c, for contigs longer than
    2^29 bases), so a Picard- or GATK-indexed BAM is accepted, as samtools,
    pysam and FreeBayes all read its index.
    """
    bam = Path(bam)
    candidates = [Path(str(bam) + ".bai")]
    if bam.suffix:
        candidates.append(bam.with_suffix(".bai"))
    candidates.append(Path(str(bam) + ".csi"))
    return candidates


def _bam_index(bam: Path) -> Optional[Path]:
    """The BAM's index file, or None when there is none."""
    for candidate in _bam_index_candidates(bam):
        if candidate.exists():
            return candidate
    return None



def _run_prefix(workdir: Path) -> str:
    return workdir.name


def _results_dir(workdir: Path) -> Path:
    return workdir / "results"


def _ensure_run_directories(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for name in (
        "checkpoints",
        "inputs",
        "logs",
        "regions",
        "results",
        "scripts",
        "shards",
        "temp",
    ):
        (workdir / name).mkdir(parents=True, exist_ok=True)


def _write_run_metadata(
    *,
    workdir: Path,
    samples: Path,
    reference: Path,
    backend: str,
    region_strategy: str,
    annotation_mode: str,
) -> None:
    from paralome import __version__

    metadata = {
        # The Paralome that made the run. This was "gv_version":
        # "0.16.1", a hard-coded version of a former project name, which the
        # pipeline summary then reported for every variant run.
        "paralome_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "samples": str(samples),
        "reference": str(reference),
        "backend": backend,
        "region_strategy": region_strategy,
        "annotation": annotation_mode,
        "workdir": str(workdir),
    }
    with open(workdir / "run.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")



def _pipeline_state_path(workdir: Path) -> Path:
    return stage_checkpoint_path(workdir, "pipeline")


def _write_pipeline_state(workdir: Path, state: dict) -> None:
    path = _pipeline_state_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def _force_init_pipeline_json(
    *,
    workdir: Path,
    samples: Path,
    reference: Path,
    backend: str,
    region_strategy: str,
    annotation_mode: str,
    jobs: int,
    threads: int,
) -> None:
    path = stage_checkpoint_path(workdir, "pipeline")
    path.parent.mkdir(parents=True, exist_ok=True)

    from paralome import __version__

    state = {
        "paralome_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "workdir": str(workdir),
        "inputs": {
            "complete": False,
            "sample_manifest": str(samples),
            "reference": str(reference),
        },
        "regions": {
            "complete": False,
            "strategy": region_strategy,
        },
        "freebayes": {
            "complete": False,
            "jobs": jobs,
            "threads": threads,
        },
        "merge": {
            "complete": False,
            "output": None,
        },
        "annotation": {
            "mode": annotation_mode,
            "snpeff": {"complete": False, "output": None},
            "annovar": {"complete": False, "output": None},
        },
        "final": {
            "complete": False,
            "output": None,
        },
    }

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)



def _pipeline_log_path(workdir: Path) -> Path:
    return workdir / "logs" / "pipeline.log"


def _append_pipeline_log(workdir: Path, message: str) -> None:
    path = _pipeline_log_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a") as fh:
        fh.write(f"{timestamp} {message}\n")


def _log_event(workdir: Path, stage: str, status: str, message: Optional[str] = None) -> None:
    path = stage_checkpoint_path(workdir, "pipeline")
    if not path.exists():
        return
    try:
        state = json.loads(path.read_text())
    except Exception:
        return
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
    }
    if message:
        event["message"] = message
    state.setdefault("events", []).append(event)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _gv_log(workdir: Path, message: str, *, verbosity: int = 0, level: int = 1, stage: Optional[str] = None) -> None:
    _append_pipeline_log(workdir, message)
    if stage:
        _log_event(workdir, stage, "message", message)
    if verbosity >= level:
        print(message, flush=True)


def _write_bam_list(bams: Sequence[Path], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for bam in bams:
            fh.write(str(bam) + "\n")
    return out


def _normalize_targets_value(targets: str | Optional[Path]) -> str:
    if targets is None:
        return "auto"
    value = str(targets).strip()
    if not value:
        return "auto"
    if value.lower() in {"auto", "genome"}:
        return value.lower()
    return value


def _merge_intervals(intervals: list[tuple[int, int]], merge_distance: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + merge_distance:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


_PRIMARY_NAME = re.compile(r"^[A-Za-z0-9]+$")

#: Accession-named chromosomes. RefSeq ``NC_`` is a complete
#: chromosome or organelle (NC_000001.11 is GRCh38 chr1, NC_056054.1 a sheep
#: chromosome of ARS-UI_Ramb_v3.0, NC_012920.1 the human mitochondrion);
#: scaffolds and contigs are NW_/NT_. GenBank ``CM`` is a chromosome-level
#: assembled sequence (CM000663.2 is GRCh38 chr1); scaffolds are KI/GL/JH...
_ACCESSION_CHROMOSOME = re.compile(r"^(NC_\d+|CM\d{6,})\.\d+$")

#: Ensembl's name for the Drosophila mitochondrion, the one placed chromosome
#: with an underscore in its name.
_NAMED_CHROMOSOMES = {"mitochondrion_genome"}

#: Decoys and spike-ins whose names pass the letters-and-digits
#: rule. hs37d5 is the 1000 Genomes GRCh37 decoy, hs38d1 the GRCh38 one; the
#: rest are sequencing controls. Lower-case.
_DECOYS_AND_SPIKE_INS = {
    "hs37d5", "hs38d1", "lambda", "phix", "phix174", "puc19", "ebv",
}


def _is_primary_contig_name(contig: str) -> bool:
    """True for a placed chromosome of any species; False for scaffolds, alts and decoys.

    A name (less any 'chr') is a chromosome when it is letters and digits only
    -- an accession (KI270728.1, JH584304.1), an alt or random contig
    (chr1_KI270706v1_random) or an HLA allele (HLA-A*01:01) is not -- and is
    not 'Un...', 'EBV', or a number above 200 (fly's unplaced
    211000022278279). So chicken 23-39, W and Z, fly 2L, 2R, 3L, 3R and 4, and
    every other species' chromosomes are kept; for human the set is 1-22, X,
    Y and MT/M.

    A RefSeq NC_ or GenBank CM accession IS a chromosome (a genome registered
    from a RefSeq FASTA has every chromosome as NC_0560xx.1). Fly's
    'mitochondrion_genome' is a chromosome too, as MT is for every other
    species. In the other direction hs37d5, lambda, phiX174 and pUC19 pass the
    letters-and-digits rule; they are decoys and spike-ins and are not called.
    """
    c = contig.strip()
    if c.lower() in _NAMED_CHROMOSOMES:
        return True
    if _ACCESSION_CHROMOSOME.match(c):
        return True
    if c.lower().startswith("chr"):
        c = c[3:]
    if not c or not _PRIMARY_NAME.match(c):
        return False
    if c.lower() in _DECOYS_AND_SPIKE_INS or c.lower().startswith("un"):
        return False
    if c.isdigit():
        return 0 < int(c) <= 200
    return True


def _report_contigs_left_out(all_contigs: Sequence[str], kept: Sequence[str], *, what: str) -> None:
    """Say which contigs the name rule left out, or that none were."""
    left_out = [c for c in all_contigs if c not in set(kept)]
    if not left_out:
        return
    shown = ", ".join(left_out[:8]) + (f", ... ({len(left_out):,} in all)" if len(left_out) > 8 else "")
    _progress(
        f"{len(left_out):,} of {len(all_contigs):,} contigs in {what} are not placed chromosomes by "
        f"name (scaffolds, alts, decoys) and are not called: {shown}. "
        "Give --regions or --targets BED to call them.",
        indent=2,
    )


def _refuse_no_primary_contigs(all_contigs: Sequence[str], *, what: str) -> None:
    """The name rule found no chromosome: say what was seen and how to go on."""
    shown = ", ".join(list(all_contigs)[:8]) + (" ..." if len(all_contigs) > 8 else "")
    raise ValueError(
        f"No placed chromosome could be told by name among the {len(all_contigs):,} contigs in "
        f"{what} ({shown}). Paralome calls chromosomes named 1-22/X/Y/MT, chr-prefixed, "
        "species letters-and-digits (2L, Z, W), or RefSeq NC_/GenBank CM accessions. "
        "For a draft or scaffold-level assembly, give the contigs to call with --regions "
        "(one contig or chrom:start-end per line) or --targets BED."
    )


def _primary_reference_contigs(reference: Path) -> list[str]:
    lengths = _reference_lengths(reference)
    kept = [contig for contig in lengths if _is_primary_contig_name(contig)]
    # What was left out is said, not just what was kept.
    _report_contigs_left_out(list(lengths), kept, what=f"{reference}.fai")
    return kept


def _callable_intervals_for_bam_contig(
    bam_path: Path,
    contig: str,
    *,
    min_depth: int,
    min_mapping_quality: int,
) -> tuple[str, list[tuple[int, int]]]:
    events: list[tuple[int, int]] = []

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        try:
            iterator = bam.fetch(contig)
        except ValueError:
            return contig, []

        for read in iterator:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            # Duplicate- and QC-fail-flagged reads do not count
            # towards callable depth, as FreeBayes does not read them.
            if read.is_duplicate or read.is_qcfail:
                continue
            if read.mapping_quality < min_mapping_quality:
                continue
            start = read.reference_start
            end = read.reference_end
            if start is None or end is None or end <= start:
                continue
            events.append((int(start), 1))
            events.append((int(end), -1))

    if not events:
        return contig, []

    events.sort()
    depth = 0
    last_pos: Optional[int] = None
    intervals: list[tuple[int, int]] = []

    for pos, delta in events:
        if last_pos is not None and pos > last_pos and depth >= min_depth:
            intervals.append((last_pos, pos))
        depth += delta
        last_pos = pos

    return contig, _merge_intervals(intervals, merge_distance=0)


def _sample_callable_intervals(
    bam_path: Path,
    *,
    contigs: Sequence[str],
    min_depth: int,
    min_mapping_quality: int,
    threads: int,
) -> dict[str, list[tuple[int, int]]]:
    """Infer per-sample callable intervals over primary chromosomes only.

    This remains region-level rather than per-base pileup-level. Empirical target
    discovery uses the existing --threads option and parallelizes by contig; the
    contigs it is given are the placed chromosomes, told by name
    (_is_primary_contig_name), so alternate contigs and scaffolds are skipped.
    """
    callable_by_contig: dict[str, list[tuple[int, int]]] = {}
    contigs = list(contigs)
    workers = max(1, min(int(threads), len(contigs) or 1))

    if workers == 1:
        for done, contig in enumerate(contigs, start=1):
            c, intervals = _callable_intervals_for_bam_contig(
                bam_path,
                contig,
                min_depth=min_depth,
                min_mapping_quality=min_mapping_quality,
            )
            if intervals:
                callable_by_contig[c] = intervals
            _progress(
                f"{bam_path.name}: {contig} complete ({len(intervals):,} intervals; {done}/{len(contigs)})",
                indent=4,
            )
        return callable_by_contig

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [
            exe.submit(
                _callable_intervals_for_bam_contig,
                bam_path,
                contig,
                min_depth=min_depth,
                min_mapping_quality=min_mapping_quality,
            )
            for contig in contigs
        ]
        done = 0
        for fut in as_completed(futures):
            contig, intervals = fut.result()
            done += 1
            if intervals:
                callable_by_contig[contig] = intervals
            _progress(
                f"{bam_path.name}: {contig} complete ({len(intervals):,} intervals; {done}/{len(contigs)})",
                indent=4,
            )

    return callable_by_contig


def _combine_sample_callable_intervals(
    per_sample: list[dict[str, list[tuple[int, int]]]],
    *,
    min_samples: int,
) -> dict[str, list[tuple[int, int]]]:
    combined: dict[str, list[tuple[int, int]]] = {}
    contigs = sorted({c for sample in per_sample for c in sample})

    for contig in contigs:
        events: list[tuple[int, int]] = []
        for sample in per_sample:
            for start, end in sample.get(contig, []):
                events.append((start, 1))
                events.append((end, -1))
        if not events:
            continue

        events.sort()
        support = 0
        last_pos: Optional[int] = None
        intervals: list[tuple[int, int]] = []
        for pos, delta in events:
            if last_pos is not None and pos > last_pos and support >= min_samples:
                intervals.append((last_pos, pos))
            support += delta
            last_pos = pos
        if intervals:
            combined[contig] = _merge_intervals(intervals, merge_distance=0)

    return combined


def _reference_lengths(reference: Path) -> dict[str, int]:
    fai = Path(str(reference) + ".fai")
    _require_file(fai, "reference FASTA index")
    lengths: dict[str, int] = {}
    with fai.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            chrom, length_s, *_ = line.rstrip("\n").split("\t")
            lengths[chrom] = int(length_s)
    return lengths


def _write_empirical_targets_bed(
    *,
    bams: Sequence[Path],
    reference: Path,
    out_bed: Path,
    min_depth: int,
    min_samples: int,
    merge_distance: int,
    padding: int,
    min_mapping_quality: int,
    threads: int,
) -> Path:
    stage_t0 = time.monotonic()
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    lengths = _reference_lengths(reference)
    contigs = _primary_reference_contigs(reference)
    if not contigs:
        _refuse_no_primary_contigs(list(lengths), what=f"{reference}.fai")

    _progress("[1/4] Empirical target discovery")
    _progress(
        f"Scanning {len(contigs)} primary chromosomes across {len(bams)} BAM(s) with {max(1, int(threads))} threads",
        indent=2,
    )
    _progress(
        f"Callable definition: depth >= {min_depth}, samples >= {min_samples}, MAPQ >= {min_mapping_quality}",
        indent=2,
    )
    min_samples = max(1, min_samples)

    per_sample = []
    discovery_t0 = time.monotonic()
    for sample_i, bam in enumerate(bams, start=1):
        sample_t0 = time.monotonic()
        _progress(f"BAM {sample_i}/{len(bams)}: {bam}", indent=2)
        sample_intervals = _sample_callable_intervals(
            bam,
            contigs=contigs,
            min_depth=min_depth,
            min_mapping_quality=min_mapping_quality,
            threads=threads,
        )
        sample_n = sum(len(v) for v in sample_intervals.values())
        _progress(
            f"BAM {sample_i}/{len(bams)} complete: {sample_n:,} callable intervals in {_fmt_seconds(time.monotonic() - sample_t0)}",
            indent=2,
        )
        per_sample.append(sample_intervals)

    _progress(f"Discovery completed in {_fmt_seconds(time.monotonic() - discovery_t0)}", indent=2)

    combine_t0 = time.monotonic()
    _progress("[2/4] Combining callable intervals across samples")
    raw_sample_intervals = sum(len(v) for sample in per_sample for v in sample.values())
    _progress(f"Raw per-sample intervals: {raw_sample_intervals:,}", indent=2)
    combined = _combine_sample_callable_intervals(per_sample, min_samples=min_samples)
    combined_n = sum(len(v) for v in combined.values())
    _progress(f"Combined intervals: {combined_n:,}", indent=2)
    _progress(f"Combination completed in {_fmt_seconds(time.monotonic() - combine_t0)}", indent=2)

    write_t0 = time.monotonic()
    _progress("[3/4] Padding, merging, and writing empirical BED")
    _progress(f"Padding: {padding} bp; merge distance: {merge_distance} bp", indent=2)
    n = 0
    with out_bed.open("w") as out:
        for contig in sorted(combined):
            length = lengths.get(contig)
            padded: list[tuple[int, int]] = []
            for start, end in combined[contig]:
                s0 = max(0, start - padding)
                e0 = end + padding
                if length is not None:
                    e0 = min(length, e0)
                if e0 > s0:
                    padded.append((s0, e0))
            merged = _merge_intervals(padded, merge_distance)
            for start, end in merged:
                out.write(f"{contig}\t{start}\t{end}\n")
                n += 1
            _progress(f"{contig}: {len(combined[contig]):,} combined -> {len(merged):,} padded/merged intervals", indent=4)

    if n == 0:
        raise ValueError("Empirical target detection produced no callable intervals")

    _progress(f"Wrote empirical targets BED: {out_bed}", indent=2)
    _progress(f"Final empirical intervals: {n:,}", indent=2)
    _progress(f"BED writing completed in {_fmt_seconds(time.monotonic() - write_t0)}", indent=2)
    _progress(f"Empirical target discovery total: {_fmt_seconds(time.monotonic() - stage_t0)}", indent=2)
    return out_bed


def _open_text(path: Path):
    """Open a GTF/GFF or BED for reading as text, gzip-compressed or not.

    Compression is recognised by the file's first two bytes, as
    rna.gene_model.open_gtf_text does, so a genome registered with a .gtf.gz
    is read. A byte-order mark at the top of the file is dropped either way.
    """
    import gzip

    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open(encoding="utf-8-sig", errors="replace")


def _parse_gtf_attributes(attr_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for raw in attr_text.strip().rstrip(";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if "=" in item and " " not in item.split("=", 1)[0]:
            key, value = item.split("=", 1)
            attrs[key] = value.strip().strip('"')
        elif " " in item:
            key, value = item.split(" ", 1)
            attrs[key] = value.strip().strip('"')
    return attrs


def _is_gene_level_feature(feature: str) -> bool:
    """Whether a GTF/GFF3 feature type is a gene locus.

    ``gene`` in GTF and GFF3; Ensembl GFF3 types every lncRNA, snRNA and
    miRNA gene ``ncRNA_gene`` and every pseudogene ``pseudogene``; NCBI GFF3
    uses ``pseudogene``; FlyBase and SGD have ``transposable_element_gene``,
    ``tRNA_gene``, ``snoRNA_gene``. Taking ``gene`` alone would silently skip
    tens of thousands of ncRNA genes and pseudogenes of an Ensembl GFF3. Not
    ``region`` (NCBI's whole-chromosome record), not transcripts, exons or
    CDS.
    """
    f = feature.strip().lower()
    return f == "gene" or f.endswith("_gene") or f.endswith("pseudogene")


def _read_gtf_gene_spans(gtf: Path, reference: Path, *, padding: int) -> dict[str, list[GeneSpan]]:
    """Read gene-level spans from a GTF or GFF3.

    Prefer explicit gene-level features (see _is_gene_level_feature). If the
    file has none, fall back to aggregating exons by gene: by ``gene_id`` in a
    GTF, and through ``Parent`` (exon -> transcript -> gene) in a GFF3, whose
    exons carry no gene_id. Coordinates returned are 0-based
    half-open and padded/clipped to the reference lengths.

    The counts by feature type used, the exon fallback, and genes
    on contigs the reference does not have are reported, so an annotation that
    does not match its reference (chr-prefixed GTF, bare reference) is said
    rather than coming out as "no FreeBayes regions".
    """
    _require_file(gtf, "GTF")
    lengths = _reference_lengths(reference)
    genes: dict[str, GeneSpan] = {}
    exon_accum: dict[tuple[str, str], tuple[int, int, Optional[str]]] = {}
    # GFF3: ID -> Parent, and the exons by their Parent, resolved at the end.
    parent_of: dict[str, str] = {}
    exons_by_parent: list[tuple[str, str, int, int]] = []
    used_by_type: dict[str, int] = {}
    off_reference: dict[str, int] = {}

    with _open_text(gtf) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _source, feature, start_s, end_s, _score, _strand, _frame, attrs_s = parts
            attrs = _parse_gtf_attributes(attrs_s)
            gene_level = _is_gene_level_feature(feature)
            if chrom not in lengths:
                if gene_level:
                    off_reference[chrom] = off_reference.get(chrom, 0) + 1
                continue
            record_id = attrs.get("ID")
            parent = attrs.get("Parent")
            if record_id and parent:
                parent_of[record_id] = parent.split(",", 1)[0]
            gene_id = attrs.get("gene_id") or record_id or attrs.get("gene")
            gene_name = attrs.get("gene_name") or attrs.get("Name")

            try:
                start0 = max(0, int(start_s) - 1 - padding)
                end0 = min(lengths[chrom], int(end_s) + padding)
            except ValueError:
                continue
            if end0 <= start0:
                continue

            if gene_level:
                if not gene_id:
                    continue
                key = f"{chrom}\t{gene_id}"
                old = genes.get(key)
                if old is None:
                    genes[key] = GeneSpan(chrom, start0, end0, gene_id, gene_name)
                else:
                    genes[key] = GeneSpan(chrom, min(old.start0, start0), max(old.end0, end0), gene_id, old.gene_name or gene_name)
                used_by_type[feature] = used_by_type.get(feature, 0) + 1
            elif feature == "exon":
                if attrs.get("gene_id"):
                    key2 = (chrom, attrs["gene_id"])
                    old2 = exon_accum.get(key2)
                    if old2 is None:
                        exon_accum[key2] = (start0, end0, gene_name)
                    else:
                        exon_accum[key2] = (min(old2[0], start0), max(old2[1], end0), old2[2] or gene_name)
                elif parent:
                    exons_by_parent.append((chrom, parent.split(",", 1)[0], start0, end0))

    if not genes:
        # GFF3 exons: climb Parent to the topmost record, which is the gene.
        for chrom, parent, start0, end0 in exons_by_parent:
            top = parent
            for _ in range(8):
                if top not in parent_of:
                    break
                top = parent_of[top]
            key2 = (chrom, top)
            old2 = exon_accum.get(key2)
            if old2 is None:
                exon_accum[key2] = (start0, end0, None)
            else:
                exon_accum[key2] = (min(old2[0], start0), max(old2[1], end0), old2[2])
        if exon_accum:
            for (chrom, gene_id), (start0, end0, gene_name) in exon_accum.items():
                genes[f"{chrom}\t{gene_id}"] = GeneSpan(chrom, start0, end0, gene_id, gene_name)
            _progress(
                f"No gene-level feature in {gtf.name}: {len(genes):,} gene spans taken from the "
                "union of their exons",
                indent=2,
            )
    if used_by_type:
        _progress(
            "Gene-level features used: "
            + ", ".join(f"{feature} {count:,}" for feature, count in sorted(used_by_type.items())),
            indent=2,
        )
    if off_reference:
        shown = ", ".join(f"{c} ({n:,})" for c, n in sorted(off_reference.items(), key=lambda i: -i[1])[:6])
        _progress(
            f"{sum(off_reference.values()):,} gene(s) on {len(off_reference):,} contig(s) the reference "
            f"does not have are not called: {shown}",
            indent=2,
        )
        if not genes:
            raise ValueError(
                f"Every gene in {gtf} lies on a contig that {reference}.fai does not have "
                f"(for example {next(iter(off_reference))}); the annotation does not match the "
                "reference. Use the GTF/GFF that belongs to this reference FASTA, or one whose "
                "contig names agree with it (chr1 against chr1, 1 against 1)."
            )

    by_chrom: dict[str, list[GeneSpan]] = {}
    for span in genes.values():
        by_chrom.setdefault(span.chrom, []).append(span)
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda g: (g.start0, g.end0, g.gene_id))
    return by_chrom


def _is_bed_header(line: str) -> bool:
    """A comment, ``track``, ``browser`` or column-header line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    first = stripped.split("\t", 1)[0].split(" ", 1)[0].lower()
    return first in {"track", "browser"}


def _read_bed_intervals_by_chrom(bed: Path) -> dict[str, list[tuple[int, int]]]:
    """BED intervals by contig, sorted and merged where they touch or overlap.

    Read through _open_text (gzip and a byte-order mark are
    handled); track/browser header lines are skipped. A line whose start or
    end is not a number (a column header) is skipped too.
    """
    intervals: dict[str, list[tuple[int, int]]] = {}
    with _open_text(bed) as fh:
        for line in fh:
            if _is_bed_header(line):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start0 = int(parts[1])
                end0 = int(parts[2])
            except ValueError:
                continue
            if end0 > start0:
                intervals.setdefault(chrom, []).append((start0, end0))
    for chrom in intervals:
        intervals[chrom] = _merge_intervals(intervals[chrom], merge_distance=0)
    return intervals


def _span_overlaps_any(span: GeneSpan, intervals: list[tuple[int, int]]) -> bool:
    # intervals are sorted and merged
    lo = 0
    hi = len(intervals)
    # first interval whose end is greater than span.start0
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= span.start0:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    return i < len(intervals) and intervals[i][0] < span.end0


def _gtf_gene_regions(
    *,
    gtf: Path,
    reference: Path,
    out: Path,
    padding: int,
    restrict_to_bed: Optional[Path],
) -> Path:
    t0 = time.monotonic()
    out.parent.mkdir(parents=True, exist_ok=True)
    _progress("[4/4] Building gene-level FreeBayes region list from GTF")
    _progress(f"GTF: {gtf}", indent=2)
    _progress(f"Gene padding: {padding} bp", indent=2)
    if restrict_to_bed is not None:
        _progress(f"Restricting to genes overlapping targets: {restrict_to_bed}", indent=2)

    genes_by_chrom = _read_gtf_gene_spans(gtf, reference, padding=padding)
    bed_by_chrom = _read_bed_intervals_by_chrom(restrict_to_bed) if restrict_to_bed is not None else None

    # The contigs with genes that the name rule leaves out are
    # named, with their gene counts; and when it leaves out every one the
    # refusal says what was seen instead of "produced no FreeBayes regions".
    primary = [chrom for chrom in genes_by_chrom if _is_primary_contig_name(chrom)]
    if genes_by_chrom and not primary:
        _refuse_no_primary_contigs(sorted(genes_by_chrom), what=f"{gtf.name} (contigs with genes)")
    left_out = sorted(set(genes_by_chrom) - set(primary))
    if left_out:
        shown = ", ".join(f"{c} ({len(genes_by_chrom[c]):,} genes)" for c in left_out[:8])
        _progress(
            f"{len(left_out):,} contig(s) with genes are not placed chromosomes by name and are not "
            f"called: {shown}{' ...' if len(left_out) > 8 else ''}. Give --regions or --targets BED "
            "to call them.",
            indent=2,
        )

    selected_by_chrom: dict[str, list[tuple[int, int]]] = {}
    selected_genes = 0
    total_genes = 0
    for chrom in sorted(primary):
        spans = genes_by_chrom[chrom]
        total_genes += len(spans)
        keep: list[tuple[int, int]] = []
        target_intervals = bed_by_chrom.get(chrom, []) if bed_by_chrom is not None else []
        for span in spans:
            if bed_by_chrom is not None and not _span_overlaps_any(span, target_intervals):
                continue
            keep.append((span.start0, span.end0))
            selected_genes += 1
        if keep:
            selected_by_chrom[chrom] = _merge_intervals(keep, merge_distance=0)
            _progress(f"{chrom}: selected {len(keep):,} genes -> {len(selected_by_chrom[chrom]):,} merged gene regions", indent=4)

    n_regions = 0
    with out.open("w") as dst:
        for chrom in sorted(selected_by_chrom):
            for start0, end0 in selected_by_chrom[chrom]:
                # Written as FreeBayes reads --region, 0-based and
                # end-exclusive (see FREEBAYES_REGION_CONVENTION).
                if end0 > start0:
                    dst.write(f"{chrom}:{start0}-{end0}\n")
                    n_regions += 1

    if n_regions == 0:
        raise ValueError("GTF gene region strategy produced no FreeBayes regions")

    _progress(f"Total genes in GTF/reference: {total_genes:,}", indent=2)
    _progress(f"Selected overlapping genes: {selected_genes:,}", indent=2)
    _progress(f"Merged gene regions: {n_regions:,}", indent=2)
    _progress(f"Wrote FreeBayes regions: {out}", indent=2)
    _progress(f"Gene region list completed in {_fmt_seconds(time.monotonic() - t0)}", indent=2)
    return out

def _bed_to_freebayes_regions(bed: Path, out: Path, reference: Optional[Path] = None) -> Path:
    """The regions FreeBayes is given from a targets BED.

    The intervals are sorted and merged where they touch or overlap, and
    clipped to the contig lengths when the reference is given. A capture BED
    padded by the vendor (overlapping neighbours) would otherwise have the
    overlaps called twice -- the merge's duplicate filter removes only exact
    CHROM/POS/REF/ALT repeats, so a boundary call read differently in the two
    regions would leave two records for one site -- and an interval running
    past a contig's end would make FreeBayes exit 1 ("outside of reference
    sequence bounds"). An interval on a contig the reference does not have is
    refused by name: the usual cause is a chr-prefixed BED against a bare
    reference, or the reverse, and FreeBayes would fail the task on it.
    """
    t0 = time.monotonic()
    out.parent.mkdir(parents=True, exist_ok=True)
    _progress("[4/4] Building FreeBayes region list from BED")
    _progress(f"Input BED: {bed}", indent=2)
    # Contigs in the order the BED first names them; the count of intervals read.
    raw = 0
    order: list[str] = []
    with _open_text(bed) as src:
        for line in src:
            if _is_bed_header(line):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3 or not (parts[1].strip().isdigit() and parts[2].strip().isdigit()):
                continue
            raw += 1
            if parts[0] not in order:
                order.append(parts[0])
    by_chrom = _read_bed_intervals_by_chrom(bed)
    lengths = _reference_lengths(reference) if reference is not None else None
    if lengths is not None:
        unknown = [chrom for chrom in order if chrom not in lengths]
        if unknown:
            raise ValueError(
                f"{len(unknown):,} contig(s) in {bed} are not in {reference}.fai: "
                + ", ".join(unknown[:8]) + (" ..." if len(unknown) > 8 else "")
                + ". The BED must name contigs as the reference does (chr1 against chr1, 1 against 1)."
            )
    n = 0
    clipped = 0
    with out.open("w") as dst:
        for chrom in order:
            for start0, end0 in by_chrom.get(chrom, []):
                if lengths is not None and end0 > lengths[chrom]:
                    end0 = lengths[chrom]
                    clipped += 1
                if end0 <= start0:
                    continue
                # BED coordinates ARE FreeBayes's --region coordinates;
                # they are copied, not shifted (see FREEBAYES_REGION_CONVENTION).
                dst.write(f"{chrom}:{start0}-{end0}\n")
                n += 1
    if n == 0:
        raise ValueError(f"No usable regions found in BED: {bed}")
    _progress(f"Wrote FreeBayes regions: {out}", indent=2)
    _progress(f"Regions: {n:,} (from {raw:,} BED intervals, merged where they touch or overlap)", indent=2)
    if clipped:
        _progress(f"Clipped to the contig's length: {clipped:,}", indent=2)
    _progress(f"Region list completed in {_fmt_seconds(time.monotonic() - t0)}", indent=2)
    return out


def _fai_to_regions(reference: Path, out: Path, chunk_size: int) -> Path:
    t0 = time.monotonic()
    fai = Path(str(reference) + ".fai")
    _require_file(fai, "reference FASTA index")
    out.parent.mkdir(parents=True, exist_ok=True)
    _progress("Building whole-genome FreeBayes region list from FASTA index")
    _progress(f"FAI: {fai}", indent=2)
    _progress(f"Chunk size: {chunk_size:,} bp", indent=2)
    n = 0
    with fai.open() as src, out.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            chrom, length_s, *_ = line.rstrip("\n").split("\t")
            length = int(length_s)
            chrom_n = 0
            # 0-based, end-exclusive windows that meet (0-100000, 100000-200000), as FreeBayes reads them, so
            # no boundary base is skipped.
            for start0 in range(0, length, chunk_size):
                end0 = min(length, start0 + chunk_size)
                dst.write(f"{chrom}:{start0}-{end0}\n")
                n += 1
                chrom_n += 1
            _progress(f"{chrom}: {chrom_n:,} regions", indent=4)
    if n == 0:
        raise ValueError(f"No regions generated from FASTA index: {fai}")
    _progress(f"Wrote FreeBayes regions: {out}", indent=2)
    _progress(f"Regions: {n:,}", indent=2)
    _progress(f"Region list completed in {_fmt_seconds(time.monotonic() - t0)}", indent=2)
    return out

def _resolve_targets_bed(
    *,
    targets: str | Optional[Path],
    reference: Path,
    workdir: Path,
    bams: Sequence[Path],
    empirical_min_depth: int,
    empirical_min_samples: int,
    empirical_merge_distance: int,
    empirical_padding: int,
    min_mapping_quality: int,
    threads: int,
) -> Optional[Path]:
    value = _normalize_targets_value(targets)
    if value == "genome":
        return None
    if value == "auto":
        return _write_empirical_targets_bed(
            bams=bams,
            reference=reference,
            out_bed=workdir / "regions" / "empirical_targets.bed",
            min_depth=empirical_min_depth,
            min_samples=empirical_min_samples,
            merge_distance=empirical_merge_distance,
            padding=empirical_padding,
            min_mapping_quality=min_mapping_quality,
            threads=threads,
        )

    bed = Path(value)
    _require_file(bed, "targets BED")
    return bed


def _prepare_regions(
    *,
    reference: Path,
    workdir: Path,
    regions: Optional[Path],
    targets: str | Optional[Path],
    bams: Sequence[Path],
    chunk_size: int,
    gtf: Optional[Path],
    region_strategy: str,
    gene_padding: int,
    empirical_min_depth: int,
    empirical_min_samples: int,
    empirical_merge_distance: int,
    empirical_padding: int,
    min_mapping_quality: int,
    threads: int,
) -> Path:
    if regions is not None:
        if _normalize_targets_value(targets) != "auto":
            raise ValueError("Use --regions or --targets, not both")
        _require_file(regions, "regions file")
        return regions

    strategy = region_strategy.lower().strip()
    if strategy not in {"auto", "empirical", "gene"}:
        raise ValueError(f"Unsupported region strategy: {region_strategy}. Use auto, empirical, or gene.")

    # For exome-scale workflows, gene-level regions are much faster than empirical
    # BAM target discovery or whole-genome chunking.  In auto mode, prefer gene
    # regions whenever a GTF is available.  Critically, gene mode must not call
    # empirical target discovery for --targets auto; that was observed to scan
    # all primary chromosomes and become prohibitively slow for exomes.
    value = _normalize_targets_value(targets)
    if strategy == "auto":
        # An explicit --targets genome asks for the whole genome, and chooses the whole-genome path even when a
        # GTF is given.
        if value == "genome":
            strategy = "empirical"
            if gtf is not None:
                _progress("--targets genome: calling the whole genome in chunks; the GTF is not used for regions", indent=2)
        else:
            strategy = "gene" if gtf is not None else "empirical"
            if strategy == "gene" and value != "auto":
                _progress(
                    f"--targets {value} with --gtf: the genes overlapping the BED are called (the gene "
                    "strategy); give --region-strategy empirical to call the BED intervals as given",
                    indent=2,
                )

    if strategy == "gene":
        if gtf is None:
            raise ValueError("--region-strategy gene requires --gtf")
        if value == "genome":
            from paralome.errors import ParalomeError

            raise ParalomeError(
                103,
                "--region-strategy gene calls the gene spans of --gtf, and --targets genome asks for "
                "the whole genome; the two cannot both be meant.",
                fix="Drop --targets genome to call gene spans, or use --region-strategy empirical "
                    "--targets genome to call the whole genome in chunks",
            )
        restrict_to_bed = None
        if value != "auto":
            restrict_to_bed = Path(value)
            _require_file(restrict_to_bed, "targets BED")
        return _gtf_gene_regions(
            gtf=gtf,
            reference=reference,
            out=workdir / "regions" / "freebayes.gene_regions.txt",
            padding=gene_padding,
            restrict_to_bed=restrict_to_bed,
        )

    bed = _resolve_targets_bed(
        targets=targets,
        reference=reference,
        workdir=workdir,
        bams=bams,
        empirical_min_depth=empirical_min_depth,
        empirical_min_samples=empirical_min_samples,
        empirical_merge_distance=empirical_merge_distance,
        empirical_padding=empirical_padding,
        min_mapping_quality=min_mapping_quality,
        threads=threads,
    )
    if bed is not None:
        return _bed_to_freebayes_regions(bed, workdir / "regions" / "freebayes.regions.txt", reference)

    return _fai_to_regions(reference, workdir / "regions" / "freebayes.regions.txt", chunk_size)


#: Pieces per CPU when the piece count is left to Paralome. Genes
#: differ in length by orders of magnitude; with one piece per CPU the run
#: waits on whichever CPU drew the longest genes. Regions are dealt round-robin
#: into pieces, so several per CPU evens the load, and the SLURM array stays in
#: the hundreds rather than one task per gene.
PIECES_PER_CPU = 8


def automatic_region_pieces(region_count: int, threads: int) -> int:
    """How many pieces to cut the calling regions into when --jobs is not given."""
    return max(1, min(int(region_count), max(1, int(threads)) * PIECES_PER_CPU))


#: How FreeBayes reads ``--region chrom:start-end``. Its own help
#: says "0-base coordinates, end_position not included (same as BED format)",
#: and freebayes-parallel's region generator writes chr:0-100000,
#: chr:100000-200000. Paralome's region files are written that way, and a
#: region is cut into pieces that meet end to start in it, so the pieces
#: together cover exactly the bases the whole region did.
FREEBAYES_REGION_CONVENTION = "0-based, end-exclusive (BED)"

#: No region handed to FreeBayes is longer than this by default.
#: Gene regions are merged where padded genes overlap, and on a production
#: cluster (2026-10) merged regions of 0.5-1.6 Mb held five tasks at 100% CPU for
#: up to two days writing nothing. The same bases in 2 kb windows finished in
#: about 30 s each. 0 means no cap.
DEFAULT_MAX_REGION_LENGTH = 100_000

#: A FreeBayes call whose output has not grown for this many minutes
#: is treated as stalled (see the array script). 0 turns the watchdog off.
DEFAULT_STALL_MINUTES = 30.0

#: The window, and the time each window is given, when a stalled region is
#: finished in small windows from its last complete record.
STALL_WINDOW_BP = 2_000
STALL_WINDOW_TIMEOUT_SECONDS = 3_600


def _parse_region(line: str) -> Optional[tuple[str, int, int]]:
    """``chrom:start-end`` as (chrom, start, end), or None for anything else.

    The contig is everything before the LAST colon, so a contig whose name
    holds one (HLA-A*01:01:01:01) is kept whole. Thousands separators are
    allowed, as FreeBayes allows them.
    """
    chrom, sep, span = line.strip().rpartition(":")
    if not sep or not chrom:
        return None
    start_text, dash, end_text = span.partition("-")
    if not dash:
        return None
    try:
        start = int(start_text.replace(",", ""))
        end = int(end_text.replace(",", ""))
    except ValueError:
        return None
    return chrom, start, end


def split_region(line: str, max_length: int) -> list[str]:
    """Cut one region into consecutive pieces no longer than ``max_length``.

    Pieces meet end to start (``c:0-100000``, ``c:100000-150000``):
    in FreeBayes's 0-based, end-exclusive reading that is no gap and no
    overlap. A line that is not ``chrom:start-end`` (a whole contig, say) is
    returned unchanged, as is everything when ``max_length`` is 0 or less.
    """
    text = line.strip()
    if max_length <= 0:
        return [text]
    parsed = _parse_region(text)
    if parsed is None:
        return [text]
    chrom, start, end = parsed
    if end - start <= max_length:
        return [text]
    pieces = []
    for piece_start in range(start, end, max_length):
        pieces.append(f"{chrom}:{piece_start}-{min(end, piece_start + max_length)}")
    return pieces


@dataclass(frozen=True)
class RegionCap:
    """What the region-length cap did to one region list."""

    region_file: Path
    source_file: Path
    max_region_length: int
    regions_before: int
    regions_after: int
    regions_split: int
    longest_before: int

    def to_dict(self) -> dict:
        return {
            "max_region_length": self.max_region_length,
            "source_region_file": str(self.source_file),
            "region_file": str(self.region_file),
            "regions_before_cap": self.regions_before,
            "regions_after_cap": self.regions_after,
            "regions_split": self.regions_split,
            "longest_region_before_cap": self.longest_before,
            "region_convention": FREEBAYES_REGION_CONVENTION,
        }


def cap_region_lengths(
    region_file: Path, workdir: Path, max_length: int, reference: Optional[Path] = None,
) -> RegionCap:
    """Apply --max-region-length to the region list FreeBayes will be given.

    Runs after _prepare_regions, for every strategy, so a --regions
    file, gene regions, a BED and whole-genome chunks are all capped the same
    way. The capped list is written to regions/freebayes.capped_regions.txt;
    the list it came from is left as it was.

    A line naming a whole contig (``chr1``) is expanded to
    ``chr1:0-<length>`` from the reference's .fai, when the reference is
    given, so the cap reaches it. It was passed through whole, with no
    message, and produced exactly the multi-day call the cap exists to
    prevent. A contig the .fai does not have is refused by name.
    """
    if max_length < 0:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            112,
            f"--max-region-length {max_length} is negative. It is the longest region, in "
            f"bases, one FreeBayes call is given; 0 means no cap and the default is "
            f"{DEFAULT_MAX_REGION_LENGTH:,}.",
            fix=f"Give 0 or a positive number of bases, e.g. --max-region-length {DEFAULT_MAX_REGION_LENGTH}",
        )
    lines = [line.strip() for line in Path(region_file).read_text().splitlines() if line.strip()]
    whole = 0
    if reference is not None:
        lengths = _reference_lengths(reference)
        expanded: list[str] = []
        for line in lines:
            if _parse_region(line) is None:
                if line not in lengths:
                    raise ValueError(
                        f"Region {line!r} in {region_file} is neither chrom:start-end nor a contig in "
                        f"{reference}.fai. Name contigs as the reference does."
                    )
                expanded.append(f"{line}:0-{lengths[line]}")
                whole += 1
            else:
                expanded.append(line)
        if whole:
            _progress(
                f"{whole:,} region line(s) name a whole contig; expanded to 0-<length> from "
                f"{Path(str(reference) + '.fai').name} so the length cap applies",
                indent=2,
            )
        lines = expanded
    longest = 0
    pieces: list[str] = []
    split = 0
    for line in lines:
        parsed = _parse_region(line)
        if parsed is not None:
            longest = max(longest, parsed[2] - parsed[1])
        cut = split_region(line, max_length)
        if len(cut) > 1:
            split += 1
        pieces.extend(cut)
    out = Path(region_file)
    if split or whole:
        out = Path(workdir) / "regions" / "freebayes.capped_regions.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(piece + "\n" for piece in pieces))
        _progress(
            f"Region length cap {max_length:,} bp: {split:,} of {len(lines):,} regions split; "
            f"{len(lines):,} -> {len(pieces):,} regions (longest was {longest:,} bp)",
            indent=2,
        )
    elif max_length > 0:
        _progress(f"Region length cap {max_length:,} bp: no region is longer (longest {longest:,} bp)", indent=2)
    return RegionCap(
        region_file=out,
        source_file=Path(region_file),
        max_region_length=max_length,
        regions_before=len(lines),
        regions_after=len(pieces),
        regions_split=split,
        longest_before=longest,
    )


def _deal_region_chunks(region_file: Path, jobs: int) -> list[str]:
    """The text of each chunk file: regions dealt round-robin into ``jobs`` pieces."""
    regions = [line.strip() for line in Path(region_file).read_text().splitlines() if line.strip()]
    if not regions:
        raise ValueError(f"No regions found in {region_file}")
    n_jobs = len(regions) if jobs <= 0 else min(jobs, len(regions))
    return ["".join(region + "\n" for region in regions[idx::n_jobs]) for idx in range(n_jobs)]


_CHUNK_FILE = re.compile(r"^chunk_\d+\.regions\.txt$")


def _clear_region_chunks(chunk_dir: Path) -> None:
    """Remove the chunk files an earlier plan wrote, and nothing else (not the folder whole, with anything a
    person had put in it)."""
    chunk_dir = Path(chunk_dir)
    if chunk_dir.is_dir() and not chunk_dir.is_symlink():
        for path in chunk_dir.iterdir():
            if _CHUNK_FILE.match(path.name) and (path.is_file() or path.is_symlink()):
                path.unlink()
    chunk_dir.mkdir(parents=True, exist_ok=True)


def _chunk_plan_path(workdir: Path) -> Path:
    return workdir / "inputs" / "freebayes_chunk_plan.json"


def _chunk_fingerprint(
    chunks: Sequence[str],
    opts: Sequence[str],
    bams: Sequence[Path],
    *,
    reference: Optional[Path] = None,
    freebayes: Optional[str] = None,
) -> str:
    """One hash of everything that decides what a shard holds.

    The chunk files' text, the FreeBayes options (reference included) and the
    BAMs in order. The stall watchdog's settings are left out: they change how
    a chunk is finished, not what it holds.

    Each BAM by path AND size and modification time, as
    pipeline.jsonio.input_fingerprint stamps BAMs, so a BAM made again in
    place (deduplication run again, under the same name) changes the
    fingerprint. The reference FASTA and the FreeBayes executable are stamped
    the same way (path, size, modification time), so a FASTA replaced in place
    or a FreeBayes build swapped under the recorded path does too.
    """
    def stamp(path) -> list:
        try:
            st = Path(path).stat()
        except OSError:
            return [str(path), None, None]
        return [str(path), st.st_size, st.st_mtime_ns]

    payload: dict = {"chunks": list(chunks), "options": list(opts), "bams": [stamp(b) for b in bams]}
    if reference is not None:
        payload["reference"] = stamp(reference)
    if freebayes:
        payload["freebayes"] = stamp(freebayes)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_chunk_plan(workdir: Path) -> dict:
    try:
        data = json.loads(_chunk_plan_path(workdir).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_chunk_plan(workdir: Path, plan: dict) -> Path:
    path = _chunk_plan_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _shard_path(workdir: Path, index: int) -> Path:
    return workdir / "shards" / f"shard_{index}.vcf"


def shard_is_finished(path: Path) -> bool:
    """Whether a shard is one a task finished: there, and with a VCF header.

    FreeBayes writes its header (ending in the #CHROM line) before
    it calls anything, also for a region with no variant; a shard whose
    lines start with a record has none, and was kept as finished when the
    stage was run again. Only the header lines are read.
    """
    try:
        with Path(path).open("r", errors="replace") as fh:
            for line in fh:
                if line.startswith("#CHROM"):
                    return True
                if not line.startswith("#"):
                    return False
    except OSError:
        return False
    return False


def _set_aside_previous_results(workdir: Path, *, shards: bool) -> Optional[Path]:
    """Move an earlier plan's results out of the way before calling.

    Nothing is deleted; what is moved goes to superseded/<time>/ in the
    workdir. Always the merged VCF, its index and the summary: the state
    reader falls back to any results/*.merged.vcf.gz with an index, so an
    earlier run's VCF left in place reads as this run finished while it is
    still calling. With ``shards`` (the chunks or options changed, so every
    chunk is called again) also the shards and the stall log: a shard
    numbered beyond the new chunk count would otherwise stay in shards/, where
    the annotation stage's shard_*.vcf glob picks it up.
    """
    moving: list[Path] = []
    merged = _results_dir(workdir) / f"{_run_prefix(workdir)}.freebayes.merged.vcf.gz"
    for path in (merged, Path(str(merged) + ".tbi"), Path(str(merged) + ".csi"),
                 _results_dir(workdir) / "freebayes_summary.json",
                 # And the inline annotation's final VCFs. The
                 # state reader globs results/*.annotated.vcf.gz FIRST, so an
                 # earlier plan's indexed annotated VCF left here read as this
                 # run's finished result while the new plan was still calling.
                 *_annotated_results(workdir)):
        if path.exists():
            moving.append(path)
    if shards:
        shard_dir = workdir / "shards"
        moving += sorted(shard_dir.glob("shard_*.vcf")) + sorted(shard_dir.glob("shard_*.vcf.tmp"))
        stall_log = workdir / "logs" / "freebayes_stalls.tsv"
        if stall_log.exists():
            moving.append(stall_log)
        # The inline annotation's shards too, kept apart by engine
        # (annovar/ and snpeff/ both hold shard_N.vcf). A rerun with fewer
        # chunks left the earlier plan's higher-numbered annotated shards here.
        for engine_dir in sorted((workdir / "annotation_shards").glob("*")):
            if engine_dir.is_dir():
                moving += sorted(engine_dir.glob("shard_*.vcf"))
    if not moving:
        return None
    dest = workdir / "superseded" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest.mkdir(parents=True, exist_ok=True)
    for path in moving:
        if path.parent.parent == workdir / "annotation_shards":
            target = dest / "annotation_shards" / path.parent.name
            target.mkdir(parents=True, exist_ok=True)
            path.replace(target / path.name)
        else:
            path.replace(dest / path.name)
    _progress(f"Set aside {len(moving):,} file(s) from an earlier run: {dest}", indent=2)
    # Superseded/ only grows -- Paralome deletes nothing (the
    # owner's rule) -- so its size is said each time something is added.
    total, sets = superseded_size(workdir)
    _progress(
        f"{workdir / 'superseded'} now holds {_human_size(total)} in {sets:,} set(s) set aside by earlier "
        "runs; Paralome never deletes them. Remove what you no longer need.",
        indent=2,
    )
    return dest


def _annotated_results(workdir: Path) -> list[Path]:
    """The inline annotation's final VCFs and their indexes, whether or not they exist."""
    paths: list[Path] = []
    for engine in ("snpeff", "annovar"):
        vcf = _results_dir(workdir) / f"{_run_prefix(workdir)}.{engine}.annotated.vcf.gz"
        paths += [vcf, Path(str(vcf) + ".tbi"), Path(str(vcf) + ".csi")]
    return paths


def _set_aside_annotated_results(workdir: Path) -> Optional[Path]:
    """Move an earlier run's annotated VCFs aside before annotating again.

    For the case plan_region_chunks does not reach: the chunk plan is
    unchanged and every shard is there, so nothing is set aside, but
    annotation was asked for and its chain will rewrite these files; the
    state reader must not take the earlier annotated VCF as this run's.
    """
    moving = [path for path in _annotated_results(workdir) if path.exists()]
    if not moving:
        return None
    dest = workdir / "superseded" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest.mkdir(parents=True, exist_ok=True)
    for path in moving:
        path.replace(dest / path.name)
    _progress(f"Set aside {len(moving):,} annotated file(s) from an earlier run: {dest}", indent=2)
    return dest


def superseded_size(workdir: Path) -> tuple[int, int]:
    """(bytes, sets) held in ``<workdir>/superseded``."""
    root = Path(workdir) / "superseded"
    if not root.is_dir():
        return 0, 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total, sum(1 for path in root.iterdir() if path.is_dir())


def _human_size(count: float) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover


def _array_spec(indices: Sequence[int], throttle: int) -> str:
    """``1-3,7,9-10%4``: SLURM array indices as ranges, with a throttle."""
    values = sorted(set(int(i) for i in indices))
    parts: list[str] = []
    run_start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(str(run_start) if run_start == prev else f"{run_start}-{prev}")
        run_start = prev = value
    parts.append(str(run_start) if run_start == prev else f"{run_start}-{prev}")
    return ",".join(parts) + f"%{max(1, min(int(throttle), len(values)))}"


@dataclass
class ChunkPlan:
    """The chunks a run calls, and which of them still need calling."""

    chunk_dir: Path
    n_chunks: int
    fingerprint: str
    resumed: bool
    missing: list[int]
    plan_path: Path
    previous_job_ids: list[str]
    region_cap: RegionCap


def plan_region_chunks(
    *,
    workdir: Path,
    region_cap: RegionCap,
    jobs: int,
    opts: Sequence[str],
    bams: Sequence[Path],
    stall_minutes: float,
    reference: Optional[Path] = None,
    freebayes: Optional[str] = None,
    fingerprint_extra: Optional[list] = None,
) -> ChunkPlan:
    """Write the chunk files, or keep them, and say which chunks still need calling.

    RE-RUNNING CALLS ONLY THE CHUNKS WITHOUT A SHARD. The chunks and
    options are fingerprinted into inputs/freebayes_chunk_plan.json. When the
    stage is run again and the fingerprint matches, and the chunk files on disk
    are the ones it describes, the finished shards are kept and only the
    chunks whose shard is missing are called. When anything differs the run is
    planned afresh, and the old shards are set aside rather than mixed in. A
    shard exists only once its task finished (it is written to .tmp and moved
    into place at the end), so a missing shard is exactly an unfinished chunk.
    """
    workdir = Path(workdir)
    chunks = _deal_region_chunks(region_cap.region_file, jobs)
    fingerprint = _chunk_fingerprint(chunks, [*opts, *(fingerprint_extra or [])], bams, reference=reference,
                                     freebayes=freebayes)
    chunk_dir = workdir / "regions" / "region_chunks"
    previous = _read_chunk_plan(workdir)
    previous_ids = [str(x) for x in previous.get("submitted_job_ids", []) if str(x).strip()]
    same = (
        previous.get("fingerprint") == fingerprint
        and int(previous.get("chunks", -1)) == len(chunks)
        and all(
            (chunk_dir / f"chunk_{idx}.regions.txt").is_file()
            and (chunk_dir / f"chunk_{idx}.regions.txt").read_text() == text
            for idx, text in enumerate(chunks, start=1)
        )
    )
    if same:
        missing = [idx for idx in range(1, len(chunks) + 1) if not shard_is_finished(_shard_path(workdir, idx))]
        _progress(
            f"Chunk plan unchanged ({len(chunks):,} chunks): {len(chunks) - len(missing):,} shard(s) "
            f"already finished and kept; {len(missing):,} chunk(s) to call",
            indent=2,
        )
        if missing:
            _set_aside_previous_results(workdir, shards=False)
    else:
        if previous:
            _progress("Chunk plan changed (regions, options or BAMs differ): every chunk is called again", indent=2)
        _set_aside_previous_results(workdir, shards=True)
        _clear_region_chunks(chunk_dir)
        for idx, text in enumerate(chunks, start=1):
            (chunk_dir / f"chunk_{idx}.regions.txt").write_text(text)
        missing = list(range(1, len(chunks) + 1))
        _progress(f"Wrote {len(chunks):,} region chunks: {chunk_dir}", indent=2)

    plan = {
        "schema": "paralome.freebayes.chunk_plan.v1",
        "created": previous.get("created") if same else datetime.now().isoformat(timespec="seconds"),
        "updated": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "chunks": len(chunks),
        "chunk_dir": str(chunk_dir),
        "resumed": same,
        "chunks_to_call": missing,
        "stall_minutes": stall_minutes,
        "stall_window_bp": STALL_WINDOW_BP,
        "stall_window_timeout_seconds": STALL_WINDOW_TIMEOUT_SECONDS,
        # Kept until the next submission has asked SLURM whether they are done.
        "submitted_job_ids": previous_ids,
        **region_cap.to_dict(),
    }
    plan_path = _write_chunk_plan(workdir, plan)
    return ChunkPlan(
        chunk_dir=chunk_dir,
        n_chunks=len(chunks),
        fingerprint=fingerprint,
        resumed=same,
        missing=missing,
        plan_path=plan_path,
        previous_job_ids=previous_ids,
        region_cap=region_cap,
    )


def _record_region_cap_in_dispatch_plan(workdir: Path, cap: RegionCap, chunk_plan: ChunkPlan) -> None:
    """Add the cap and the region counts to inputs/dispatch_plan.json, if there is one."""
    path = Path(workdir) / "inputs" / "dispatch_plan.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return
    if not isinstance(data, dict):
        return
    data["freebayes_regions"] = {
        **cap.to_dict(),
        "chunks": chunk_plan.n_chunks,
        "chunks_to_call": len(chunk_plan.missing),
        "resumed": chunk_plan.resumed,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _stall_seconds(stall_minutes: float) -> int:
    try:
        minutes = float(stall_minutes)
    except (TypeError, ValueError):
        minutes = -1.0
    if minutes < 0:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            112,
            f"--stall-minutes {stall_minutes} is negative. It is how long a FreeBayes call may "
            f"write nothing before its region is finished in {STALL_WINDOW_BP:,} bp windows; "
            f"0 turns the watchdog off and the default is {DEFAULT_STALL_MINUTES:g}.",
            fix=f"Give 0 or a positive number of minutes, e.g. --stall-minutes {DEFAULT_STALL_MINUTES:g}",
        )
    if minutes == 0:
        return 0
    return max(1, int(round(minutes * 60)))


def freebayes_array_script(
    *,
    workdir: Path,
    chunk_dir: Path,
    bam_list: Path,
    freebayes: str,
    opts: Sequence[str],
    stall_minutes: float,
    sbatch_header: str = "",
    window_bp: int = STALL_WINDOW_BP,
    window_timeout_seconds: int = STALL_WINDOW_TIMEOUT_SECONDS,
) -> str:
    """The script one array task runs: every region of its chunk, into one shard.

    THE STALL WATCHDOG. Each region's FreeBayes runs in the
    background into its own file, and the file's size is checked every second.
    FreeBayes block-buffers its output, so the size moves in steps; a call
    whose output has not grown for --stall-minutes is taken to be stuck. It is
    killed, the whole lines it wrote are kept (a cut-off last line is dropped),
    and the rest of the region is called in window_bp windows starting where
    its last complete record's REF ends, each window with
    window_timeout_seconds. A window that fails fails the task, naming the
    window, and no shard is written. Every stall is logged to
    logs/freebayes_stalls.tsv (array job, task, region, last position,
    windows, seconds, windows run again without a limit), and the merge
    counts them into results/freebayes_summary.json.

    Also:
    * every call runs under setsid and a stall or time-out kills its whole
      process group, so a wrapper `freebayes` cannot leave the real one
      running (coreutils timeout would kill only the wrapper);
    * the resume point is the end of the last record's REF (POS + len(REF)
      - 1, 1-based, which is the 0-based start of the next base), so an
      indel or complex record crossing it is not called twice;
    * a window that runs past window_timeout_seconds is run once more with
      no time limit (logged) instead of failing the task on every re-run;
    * a shard is written only with a VCF header: FreeBayes writes its header
      before it calls anything, also for a region without a variant, so a
      first region that wrote none is a failed call.

    The task reads SLURM_ARRAY_TASK_ID, which the local backend sets itself.
    Every path is absolute. Written for bash 4.2 under ``set -euo pipefail``.
    """
    stall_seconds = _stall_seconds(stall_minutes)
    workdir_q = shlex.quote(str(workdir))
    opts_s = " ".join(shlex.quote(str(x)) for x in opts)
    return f'''#!/usr/bin/env bash
{sbatch_header}
set -euo pipefail

FREEBAYES={shlex.quote(str(freebayes))}
FB_OPTS=({opts_s})
REGION_CHUNKS={shlex.quote(str(chunk_dir))}
BAMS={shlex.quote(str(bam_list))}
SHARDS={shlex.quote(str(workdir / 'shards'))}
TASK="${{SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}}"
ARRAY_JOB="${{SLURM_ARRAY_JOB_ID:-local}}"
TASK_REGIONS="$REGION_CHUNKS/chunk_${{TASK}}.regions.txt"
OUT="$SHARDS/shard_${{TASK}}.vcf"
TMP="$OUT.tmp"
WORK={workdir_q}/temp/freebayes_task_${{TASK}}
STALL_LOG={workdir_q}/logs/freebayes_stalls.tsv
# Seconds a call may write nothing before it is treated as stalled
# (0 = no watchdog), and how a stalled region is finished.
STALL_SECONDS={stall_seconds}
WINDOW_BP={int(window_bp)}
WINDOW_TIMEOUT={int(window_timeout_seconds)}

mapfile -t BAM_FILES < "$BAMS"
if [[ "${{#BAM_FILES[@]}}" -eq 0 ]]; then
    echo "ERROR: no BAMs listed in $BAMS" >&2
    exit 1
fi
if [[ "$STALL_SECONDS" -gt 0 ]] && ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: the stall watchdog needs setsid (util-linux), which is not on PATH" >&2
    exit 1
fi
# How often a running call is looked at.
POLL=1
if sleep 0.2 2>/dev/null; then POLL=0.2; fi

# The task's own scratch files, by name; never the folder whole.
clear_work() {{
    rm -f "$WORK/region.vcf" "$WORK/kept.vcf" "$WORK/recovered.vcf" "$WORK/window.vcf" "$WORK/stalls.tsv"
}}
rm -f "$OUT" "$TMP"
if [[ -d "$WORK" ]]; then clear_work; fi
mkdir -p "$WORK" "$SHARDS" "$(dirname "$STALL_LOG")"
first=1

file_size() {{
    local n=0
    if [[ -e "$1" ]]; then n=$(wc -c < "$1"); fi
    echo $(( n ))
}}

has_header() {{
    # A #CHROM line before the first record. FreeBayes writes its
    # header before it calls anything, also for a region with no variant, so
    # output without one is not a finished call.
    awk '/^#CHROM/ {{ found = 1; exit }} !/^#/ {{ exit }} END {{ exit !found }}' "$1"
}}

# The first region's output goes in whole; later regions add their records.
append_part() {{
    if [[ "$first" -eq 1 ]]; then
        # And it must carry the header: a first region that wrote none would leave a shard of bare
        # records.
        if ! has_header "$1"; then
            echo "ERROR: FreeBayes wrote no VCF header for region $2 (task $TASK). No shard was written." >&2
            exit 1
        fi
        cat "$1" > "$TMP"
        first=0
    else
        # Grep exits 1 for a region with no records; that is not
        # an error, a FreeBayes failure is (and is checked before this).
        {{ grep -v '^#' "$1" || true; }} >> "$TMP"
    fi
}}

# Stall rows are kept with the task's own files and appended to the run's log only once its shard is in
# place, so a task that stalled, failed and ran again does not log its stall twice.
log_stall() {{
    printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "$ARRAY_JOB" "$TASK" "$1" "$2" "$3" "$4" "$5" >> "$WORK/stalls.tsv"
}}

publish_stalls() {{
    if [[ ! -s "$WORK/stalls.tsv" ]]; then return 0; fi
    if [[ ! -e "$STALL_LOG" ]]; then
        ( set -o noclobber; printf 'array_job\\ttask\\tregion\\tlast_position\\twindows\\tseconds\\twindows_rerun_without_limit\\n' > "$STALL_LOG" ) 2>/dev/null || true
    fi
    cat "$WORK/stalls.tsv" >> "$STALL_LOG"
}}

# End a call and everything it started. Each call runs under setsid, so it leads a process group of its
# own; a `freebayes` that is a wrapper script (a module's, a conda shim) cannot leave the real FreeBayes
# running, orphaned, when the wrapper is killed.
kill_group() {{
    local pid=$1 waited=0
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt 10 ]]; do sleep 1; waited=$(( waited + 1 )); done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}}

# A call in a process group of its own is not reached by a signal
# to the task's group (Ctrl-C on a local run, a scheduler that signals the
# group), so the task ends the current call itself when it is stopped.
CUR_PID=""
on_stop() {{
    if [[ -n "$CUR_PID" ]]; then kill_group "$CUR_PID"; CUR_PID=""; fi
    exit "$1"
}}
trap 'on_stop 130' INT
trap 'on_stop 143' TERM
trap 'on_stop 129' HUP
trap 'if [[ -n "$CUR_PID" ]]; then kill_group "$CUR_PID"; fi' EXIT

# Run one FreeBayes call in its own process group and watch it. Sets CALL_RC:
# its exit code, 125 when its output did not grow for $3 seconds (0 = not
# watched for that), 124 when it ran longer than $4 seconds (0 = no limit).
watch_call() {{
    local region=$1 out=$2 stall=$3 limit=$4 pid size last_size=-1 t0=$SECONDS last_change=$SECONDS
    CALL_RC=0
    setsid "$FREEBAYES" "${{FB_OPTS[@]}}" --region "$region" "${{BAM_FILES[@]}}" < /dev/null > "$out" &
    pid=$!
    CUR_PID=$pid
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL"
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
        if [[ "$stall" -gt 0 ]]; then
            size=$(file_size "$out")
            if [[ "$size" -ne "$last_size" ]]; then
                last_size=$size
                last_change=$SECONDS
            elif [[ $(( SECONDS - last_change )) -ge "$stall" ]]; then
                kill_group "$pid"
                CUR_PID=""
                CALL_RC=125
                return 0
            fi
        fi
        if [[ "$limit" -gt 0 && $(( SECONDS - t0 )) -ge "$limit" ]]; then
            kill_group "$pid"
            CUR_PID=""
            CALL_RC=124
            return 0
        fi
    done
    wait "$pid" || CALL_RC=$?
    CUR_PID=""
    return 0
}}

# Finish a stalled region in windows, from its last complete record.
finish_stalled_region() {{
    local region=$1 part=$2 t0=$3
    local kept="$WORK/kept.vcf" assembled="$WORK/recovered.vcf" window_out="$WORK/window.vcf"
    local chrom span rstart rend have_header=0 last="" last_chrom="" last_pos="" last_end="" start x end w rc n=0
    local reruns=""
    # Whole lines only: the last line may have been cut off mid-write.
    : > "$kept"
    if [[ -s "$part" ]]; then
        if [[ -n "$(tail -c 1 "$part")" ]]; then sed '$d' "$part" > "$kept"; else cat "$part" > "$kept"; fi
    fi
    chrom=${{region%:*}}
    span=${{region##*:}}
    if [[ "$chrom" == "$region" || "$span" != *-* ]]; then
        echo "ERROR: FreeBayes stalled on region $region in task $TASK, and that region has no start-end to finish it from in windows. No shard was written." >&2
        exit 1
    fi
    rstart=${{span%%-*}}
    rend=${{span#*-}}
    rstart=${{rstart//,/}}
    rend=${{rend//,/}}
    if has_header "$kept"; then have_header=1; fi
    if [[ "$have_header" -eq 1 ]]; then
        # The last whole record: its contig, POS and the length of its REF.
        last=$(awk -F '\\t' '!/^#/ && NF >= 4 {{ c = $1; p = $2; l = length($4) }} END {{ if (p != "") print c "\\t" p "\\t" l }}' "$kept")
        cat "$kept" > "$assembled"
    else
        # Not even the header was written whole: the first window supplies it.
        : > "$assembled"
    fi
    start=$rstart
    if [[ -n "$last" ]]; then
        last_chrom=${{last%%$'\\t'*}}
        last_pos=${{last#*$'\\t'}}
        last_end=${{last_pos#*$'\\t'}}
        last_pos=${{last_pos%%$'\\t'*}}
        # Resume where the last record's REF ENDS. A record at 1-based POS p with a REF of l bases covers
        # p .. p+l-1, which is the 0-based bases p-1 .. p+l-2; the first base after it is 0-based p+l-1, so
        # an indel or complex record crossing that point is not called a second time, from inside it.
        last_end=$(( last_pos + last_end - 1 ))
        if [[ "$last_chrom" == "$chrom" && "$last_pos" -gt "$rstart" && "$last_pos" -le "$rend" ]]; then
            start=$last_end
            if [[ "$start" -gt "$rend" ]]; then start=$rend; fi
        fi
    fi
    echo "STALL: task $TASK, region $region wrote nothing for ${{STALL_SECONDS}}s; last whole record ${{last_chrom:-none}}:${{last_pos:-none}}; finishing $chrom:$start-$rend in ${{WINDOW_BP}} bp windows" >&2
    x=$start
    while [[ "$x" -lt "$rend" ]]; do
        end=$(( x + WINDOW_BP ))
        if [[ "$end" -gt "$rend" ]]; then end=$rend; fi
        w="$chrom:$x-$end"
        watch_call "$w" "$window_out" 0 "$WINDOW_TIMEOUT"
        rc=$CALL_RC
        if [[ "$rc" -eq 124 ]]; then
            # A window that runs past its time is run once more with no time limit, rather than failing the
            # task for good (every run of the stage would meet the same window).
            echo "STALL: FreeBayes window $w did not finish within ${{WINDOW_TIMEOUT}}s (task $TASK, finishing stalled region $region); running it once more with no time limit" >&2
            reruns="${{reruns:+$reruns,}}$w"
            rc=0
            "$FREEBAYES" "${{FB_OPTS[@]}}" --region "$w" "${{BAM_FILES[@]}}" < /dev/null > "$window_out" || rc=$?
            if [[ "$rc" -ne 0 ]]; then
                echo "ERROR: FreeBayes window $w exited $rc when run again with no time limit (task $TASK, finishing stalled region $region). No shard was written." >&2
                exit 1
            fi
        elif [[ "$rc" -ne 0 ]]; then
            echo "ERROR: FreeBayes window $w exited $rc (task $TASK, finishing stalled region $region). No shard was written." >&2
            exit 1
        fi
        if [[ "$have_header" -eq 0 ]]; then
            if ! has_header "$window_out"; then
                echo "ERROR: FreeBayes window $w wrote no VCF header (task $TASK, finishing stalled region $region). No shard was written." >&2
                exit 1
            fi
            grep '^#' "$window_out" > "$assembled"
            have_header=1
        fi
        {{ grep -v '^#' "$window_out" || true; }} >> "$assembled"
        n=$(( n + 1 ))
        x=$end
    done
    if [[ "$have_header" -eq 0 ]]; then
        echo "ERROR: FreeBayes stalled on region $region in task $TASK before writing a header, and nothing was left to call. No shard was written." >&2
        exit 1
    fi
    append_part "$assembled" "$region"
    log_stall "$region" "${{last_chrom:-none}}:${{last_pos:-none}}" "$n" "$(( SECONDS - t0 ))" "${{reruns:-none}}"
    echo "STALL: task $TASK, region $region finished in $n window(s)" >&2
}}

call_region() {{
    local region=$1 part="$WORK/region.vcf" t0=$SECONDS
    rm -f "$part"
    if [[ "$STALL_SECONDS" -le 0 ]]; then
        "$FREEBAYES" "${{FB_OPTS[@]}}" --region "$region" "${{BAM_FILES[@]}}" < /dev/null > "$part"
        append_part "$part" "$region"
        return 0
    fi
    watch_call "$region" "$part" "$STALL_SECONDS" 0
    if [[ "$CALL_RC" -eq 125 ]]; then
        finish_stalled_region "$region" "$part" "$t0"
        return 0
    fi
    if [[ "$CALL_RC" -ne 0 ]]; then
        echo "ERROR: FreeBayes exited $CALL_RC on region $region (task $TASK). No shard was written." >&2
        exit "$CALL_RC"
    fi
    append_part "$part" "$region"
}}

while IFS= read -r REGION <&3 || [[ -n "$REGION" ]]; do
    REGION=${{REGION%$'\\r'}}
    [[ -z "$REGION" ]] && continue
    echo "Running freebayes on $REGION"
    call_region "$REGION"
done 3< "$TASK_REGIONS"

if [[ "$first" -eq 1 ]]; then
    echo "ERROR: chunk $TASK_REGIONS lists no regions. No shard was written." >&2
    exit 1
fi
if ! has_header "$TMP"; then
    echo "ERROR: the shard for task $TASK has no VCF header. No shard was written." >&2
    exit 1
fi
mv "$TMP" "$OUT"
publish_stalls
clear_work
rmdir "$WORK" 2>/dev/null || true
'''


#: How the merge jobs update checkpoints/pipeline.json. Under
#: --annotation on SLURM the FreeBayes merge (afterany:array) and the
#: annotation chain (afterok:array) run at the same time and both read, change
#: and write this file, so the update is made under an exclusive flock on
#: pipeline.json.lock and written tmp+replace.
#: These are ordinary functions, called by the worker commands
#: `variants freebayes-merge-record` and `variants freebayes-annotation-record`
#: that the generated merge scripts run, so they are tested code running in an
#: interpreter that has imported this module.
def update_pipeline_state(path: Path, mutate) -> None:
    """Read, change and write pipeline.json under a lock, atomically."""
    import fcntl

    path = Path(path)
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(path.read_text()) if path.exists() else {}
            except ValueError:
                state = {}
            if not isinstance(state, dict):
                state = {}
            mutate(state)
            tmp = Path(str(path) + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, path)
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)


def _superseded_record(workdir: Path) -> dict:
    """What earlier runs set aside under superseded/ (never deleted by Paralome): its size and sets."""
    superseded_dir = workdir / "superseded"
    superseded_bytes = 0
    if superseded_dir.is_dir():
        for item in superseded_dir.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    superseded_bytes += item.stat().st_size
            except OSError:
                pass
    return {
        "path": str(superseded_dir),
        "bytes": superseded_bytes,
        "sets": len([d for d in superseded_dir.iterdir() if d.is_dir()]) if superseded_dir.is_dir() else 0,
    }


def record_freebayes_merge(
    *,
    workdir: Path,
    merged_vcf: Path,
    backend: str,
    shards: int,
    sets_final: bool,
    variant_count: str = "",
    duplicates_removed: str = "",
) -> dict:
    """Write results/freebayes_summary.json and record the merge in pipeline.json; the summary.

    Run by the merge script once the merged VCF is in place and indexed.
    ``variant_count`` and ``duplicates_removed`` are the script's own counts as
    text (None in the summary when not a number). ``sets_final`` is False under
    an annotation chain, which records the final result itself.
    """
    import csv

    workdir = Path(workdir)
    path = stage_checkpoint_path(workdir, "pipeline")
    now = datetime.now().isoformat(timespec="seconds")
    out = Path(merged_vcf)
    variant_count_text = str(variant_count or "").strip()
    duplicates_text = str(duplicates_removed or "").strip()
    try:
        chunk_plan = json.loads((workdir / "inputs" / "freebayes_chunk_plan.json").read_text())
    except (OSError, ValueError):
        chunk_plan = {}
    if not isinstance(chunk_plan, dict):
        chunk_plan = {}
    superseded = _superseded_record(workdir)
    if superseded["bytes"]:
        print(f"Set aside by earlier runs (never deleted by Paralome): {superseded['path']} holds "
              f"{superseded['bytes'] / 1024 ** 3:.2f} GB in {superseded['sets']} set(s)")
    # The stalls the watchdog finished, so the run reports them.
    stall_log = workdir / "logs" / "freebayes_stalls.tsv"
    stalls = []
    if stall_log.exists():
        with stall_log.open() as fh:
            stalls = [row for row in csv.DictReader(fh, delimiter="\t") if row.get("task")]
    summary = {
        "schema": "paralome.freebayes.summary.v1",
        "stage": "variants",
        "complete": True,
        "backend": str(backend),
        "shards": int(shards),
        "variant_count": int(variant_count_text) if variant_count_text.isdigit() else None,
        "merged_vcf": str(out),
        "merged_index": str(out) + ".tbi",
        "finished": now,
        "duplicate_records_removed": int(duplicates_text) if duplicates_text.isdigit() else None,
        # What earlier runs set aside. Nothing there is deleted by
        # Paralome; its size is reported so it can be seen growing.
        "superseded": superseded,
        "region_cap": {
            key: chunk_plan.get(key)
            for key in (
                "max_region_length", "regions_before_cap", "regions_after_cap", "regions_split",
                "longest_region_before_cap", "region_convention", "region_file",
            )
        },
        "chunk_plan": {
            key: chunk_plan.get(key)
            for key in ("fingerprint", "chunks", "resumed", "chunks_to_call")
        },
        "stalls": {
            "count": len(stalls),
            "tasks": sorted({int(row["task"]) for row in stalls if str(row.get("task", "")).isdigit()}),
            "log": str(stall_log),
            "stall_minutes": chunk_plan.get("stall_minutes"),
            "window_bp": chunk_plan.get("stall_window_bp"),
            "window_timeout_seconds": chunk_plan.get("stall_window_timeout_seconds"),
            "windows_rerun_without_limit": sum(
                len([w for w in str(row.get("windows_rerun_without_limit") or "").split(",") if w and w != "none"])
                for row in stalls
            ),
            "records": stalls,
        },
    }
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    summary_path = workdir / "results" / "freebayes_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def record(state):
        state.setdefault("merge", {}).update({"complete": True, "output": str(out), "finished": now})
        if sets_final:
            state.setdefault("final", {}).update({"complete": True, "output": str(out), "finished": now})
        state.setdefault("freebayes", {}).update({
            "complete": True,
            "summary": str(summary_path),
            "finished": now,
            "stalls": len(stalls),
        })

    update_pipeline_state(path, record)
    return summary


def record_annotation_merge(*, workdir: Path, mode: str, output: Path) -> None:
    """Record the annotated, merged VCF as the annotation mode's output and the run's final result."""
    workdir = Path(workdir)
    path = stage_checkpoint_path(workdir, "pipeline")
    now = datetime.now().isoformat(timespec="seconds")
    output = str(Path(output))

    def record(state):
        state.setdefault("annotation", {}).setdefault(str(mode), {}).update(
            {"complete": True, "output": output, "finished": now})
        state.setdefault("final", {}).update({"complete": True, "output": output, "finished": now})

    update_pipeline_state(path, record)


#: An ANNOVAR shard task's scratch need, as a multiple of its input shard:
#: the split, its sort and ANNOVAR's working files.
ANNOVAR_SHARD_SCRATCH_FACTOR = 6


def _copy_out_bash(out_var: str, copy_to: Optional[Path]) -> str:
    """Bash that copies the finished VCF (and index) to --out.

    For a SLURM run whose --out names a VCF file: the run's work goes in
    <stem>.freebayes_work beside it (default_workdir), and the job that
    writes the final VCF puts a copy where --out said. Nothing to do when
    --out is the workdir itself.
    """
    if copy_to is None:
        return ""
    target = shlex.quote(str(copy_to))
    if str(copy_to).lower().endswith(".gz"):
        return f'''
# --out named a VCF file: the result is copied there with its index.
mkdir -p "$(dirname {target})"
cp -f "${out_var}" {target}
cp -f "${out_var}.tbi" {target}.tbi
echo "Copied to {copy_to}"
'''
    return f'''
# --out named an uncompressed VCF: the result is decompressed there.
mkdir -p "$(dirname {target})"
bgzip -dc "${out_var}" > {target}.partial
mv -f {target}.partial {target}
echo "Copied to {copy_to}"
'''


def freebayes_merge_script(
    *,
    workdir: Path,
    merge_out_vcf: Path,
    n_chunks: int,
    backend: str,
    annotation_mode: str = "none",
    copy_to: Optional[Path] = None,
    min_qual: float = 0.0,
) -> str:
    """The script that joins the shards into the run's VCF and checkpoints it.

    A record whose QUAL is below ``min_qual`` (--min-qual) is left
    out of the merged VCF, and counted.

    Written by one function for both backends, with every path
    absolute; exact duplicate records (same CHROM, POS, REF and ALT) are
    dropped after sorting and counted; and the summary carries the region cap,
    the chunk plan and the stalls the watchdog finished.

    One stream (shard records | bcftools sort -T
    <workdir>/temp/bcftools_sort | duplicate filter | bgzip), so no
    uncompressed copy of the merged VCF is written; a shard without a VCF
    header is refused like a missing one; and the summary gives the size of
    superseded/, which only grows (Paralome deletes nothing there).

    With ``annotation_mode`` other than none this merge records
    ``merge`` and ``freebayes`` only, never ``final``. It set final.complete
    on the raw merged VCF while the annotation chain was still running, so
    the stage read as complete with an unannotated output; and when the sort
    of a large merge finished after the annotation merge, it put the raw VCF
    back as the final output for good. ``copy_to`` is --out when that names
    a VCF file (SLURM; the local backend copies in-process).
    """
    workdir_q = shlex.quote(str(workdir))
    sets_final = annotation_mode == "none"
    return f'''#!/usr/bin/env bash
set -euo pipefail

WORKDIR={workdir_q}
OUT={shlex.quote(str(merge_out_vcf))}
# The merge is ONE STREAM -- shard records, bcftools sort, the duplicate filter, bgzip -- with no
# uncompressed copy on disk (a whole-genome cohort VCF can be tens of GB uncompressed). bcftools sort
# spills to SORT_TMP, in the run's own folder: never /tmp.
SORT_TMP="$WORKDIR/temp/bcftools_sort"
PARTIAL="$OUT.partial"
DUPLICATES="$WORKDIR/temp/merged.duplicates.txt"
LOW_QUAL="$WORKDIR/temp/merged.low_qual.txt"
mkdir -p "$WORKDIR/temp" "$SORT_TMP" "$(dirname "$OUT")" "$WORKDIR/results" "$WORKDIR/logs"

require_tool() {{
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required executable not found in PATH: $1" >&2
        exit 1
    fi
}}

{_pinned_tools_bash()}
require_tool bcftools
require_tool bgzip
require_tool tabix

has_header() {{
    awk '/^#CHROM/ {{ found = 1; exit }} !/^#/ {{ exit }} END {{ exit !found }}' "$1"
}}

# EVERY CHUNK, OR NO MERGE.
#
# This job runs afterany, so it runs when array tasks have failed (the rest of an array is left to finish
# after one task fails). Merging whichever shards it found would give a VCF with whole regions missing
# and nothing to say so. The shards are also named, 1 to EXPECTED, rather than globbed: a shard left over
# from an earlier run with more chunks is not part of this one. And a shard without a VCF header is not a
# finished one.
EXPECTED={int(n_chunks)}
shards=()
missing=()
for i in $(seq 1 "$EXPECTED"); do
    f="$WORKDIR/shards/shard_$i.vcf"
    if [[ -s "$f" ]] && has_header "$f"; then
        shards+=("$f")
    else
        missing+=("$i")
    fi
done
if [[ "${{#missing[@]}}" -gt 0 ]]; then
    echo "ERROR: ${{#missing[@]}} of $EXPECTED FreeBayes chunks have no shard VCF (or one without a VCF header): ${{missing[*]}}" >&2
    echo "Their tasks failed; see $WORKDIR/logs/freebayes_*_<chunk>.err. Nothing was merged." >&2
    echo "Re-run the same command: the finished shards are kept and only these chunks are called again." >&2
    exit 1
fi

echo "Found ${{#shards[@]}} shard VCFs"
echo "Concatenating, sorting (temporary files in $SORT_TMP), removing duplicates and compressing, in one stream"
rm -f "$PARTIAL"
# A record called twice -- same CHROM, POS, REF and ALT -- is kept
# once. Pieces of a capped region meet without overlapping, and a stalled
# region resumes after its last complete record, so this should find none; it
# is here so that a boundary FreeBayes reads differently costs a count in the
# summary rather than a duplicated variant downstream.
{{
    grep '^#' "${{shards[0]}}"
    for f in "${{shards[@]}}"; do
        grep -v '^#' "$f" || true
    done
}} | bcftools sort -T "$SORT_TMP/bcftools." -O v - \
   | awk -F '\\t' -v OFS='\\t' -v DUPS="$DUPLICATES" -v LOWQ="$LOW_QUAL" -v MINQ={float(min_qual):g} '
    /^#/ {{ print; next }}
    $6 != "." && $6 + 0 < MINQ {{ lowq++; next }}
    {{
        if ($1 != chrom || $2 != pos) {{ split("", seen); chrom = $1; pos = $2 }}
        key = $4 "\\t" $5
        if (key in seen) {{ dropped++; next }}
        seen[key] = 1
        print
    }}
    END {{ print dropped + 0 > DUPS; print lowq + 0 > LOWQ }}
' | bgzip -c > "$PARTIAL"
mv -f "$PARTIAL" "$OUT"
tabix -f -p vcf "$OUT"
VARIANT_COUNT=$(bcftools index -n "$OUT")
DUPLICATES_REMOVED=$(cat "$DUPLICATES" 2>/dev/null || echo 0)
echo "Left out below QUAL {float(min_qual):g}: $(cat "$LOW_QUAL" 2>/dev/null || echo 0) records"

rm -f "$DUPLICATES" "$LOW_QUAL"
rmdir "$SORT_TMP" 2>/dev/null || true
echo "Wrote $OUT"
echo "Variant records: $VARIANT_COUNT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') Merge complete: $OUT" >> "$WORKDIR/logs/pipeline.log"
# The summary and the checkpoint are written by the worker; see freebayes.record_freebayes_merge.
{worker_command()} variants freebayes-merge-record --workdir "$WORKDIR" --merged-vcf "$OUT" \
    --run-backend {shlex.quote(str(backend))} --shards {int(n_chunks)} {"--sets-final" if sets_final else "--no-sets-final"} \
    --variant-count "$VARIANT_COUNT" --duplicates-removed "$DUPLICATES_REMOVED"
{_copy_out_bash("OUT", copy_to)}if [[ -s {shlex.quote(str(workdir / "logs" / "freebayes_stalls.tsv"))} ]]; then
    echo "FreeBayes stalls finished in windows: $(( $(wc -l < {shlex.quote(str(workdir / "logs" / "freebayes_stalls.tsv"))}) - 1 )) (see {workdir / "logs" / "freebayes_stalls.tsv"})"
fi
'''


def _freebayes_options(
    *,
    reference: Path,
    min_alternate_count: int,
    min_alternate_fraction: float,
    min_base_quality: int,
    min_mapping_quality: int,
    haplotype_length: int,
    pooled_discrete: bool,
    pooled_continuous: bool,
    extra_options: Optional[str],
) -> list[str]:
    opts = [
        "-f",
        str(reference),
        "--min-alternate-count",
        str(min_alternate_count),
        "--min-alternate-fraction",
        str(min_alternate_fraction),
        "--min-base-quality",
        str(min_base_quality),
        "--min-mapping-quality",
        str(min_mapping_quality),
        "--haplotype-length",
        str(haplotype_length),
    ]
    if pooled_discrete:
        opts.append("--pooled-discrete")
    if pooled_continuous:
        opts.append("--pooled-continuous")
    if extra_options:
        opts.extend(shlex.split(extra_options))
    return opts


def _is_vcf_path(path: Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".vcf.gz") or name.endswith(".vcf")


def default_workdir(out: Path, backend: str, workdir: Optional[Path] = None) -> Path:
    """The run's working directory, absolute.

    --workdir when given; otherwise --out itself, which is what the pipeline
    passes. For a run whose --out names a VCF file, the work goes
    in ``<name>.freebayes_work`` beside it, rather than in a directory
    created under the file's own name, and the job that writes the final VCF
    copies it to --out (see _copy_out_bash). For the SLURM backend too.
    """
    if workdir is not None:
        return _absolute(workdir)
    out = _absolute(out)
    if _is_vcf_path(out) and not out.is_dir():
        name = out.name
        stem = name[: -len(".vcf.gz")] if name.lower().endswith(".vcf.gz") else name[: -len(".vcf")]
        return out.parent / f"{stem}.freebayes_work"
    return out


def run_freebayes_local(
    *,
    samples: Path,
    reference: Path,
    out_vcf: Path,
    workdir: Path,
    threads: int,
    regions: Optional[Path],
    targets: str | Optional[Path],
    chunk_size: int,
    gtf: Optional[Path],
    region_strategy: str,
    gene_padding: int,
    empirical_min_depth: int,
    empirical_min_samples: int,
    empirical_merge_distance: int,
    empirical_padding: int,
    min_alternate_count: int,
    min_alternate_fraction: float,
    min_base_quality: int,
    min_mapping_quality: int,
    haplotype_length: int,
    min_qual: float = 0.0,
    pooled_discrete: bool,
    pooled_continuous: bool,
    extra_options: Optional[str],
    jobs: int = 0,
    verbose: int = 0,
    max_region_length: int = DEFAULT_MAX_REGION_LENGTH,
    stall_minutes: float = DEFAULT_STALL_MINUTES,
) -> FreebayesRunResult:
    """Call FreeBayes on this machine, the way the SLURM backend does.

    THE LOCAL BACKEND IS THE SLURM BACKEND RUN HERE. It prepares the same
    capped regions and chunks as write_freebayes_slurm, runs the same task
    script for each chunk (``threads`` at a time, each with the stall
    watchdog) and the same merge script, so it writes the same files:
    shards/shard_<i>.vcf, results/<workdir>.freebayes.merged.vcf.gz and .tbi,
    results/freebayes_summary.json and checkpoints/pipeline.json. Run again,
    it calls only the chunks without a shard. When --out names a VCF file
    rather than the workdir, the merged VCF is also copied there.
    """
    reference = _absolute(reference)
    samples = _absolute(samples)
    out_vcf = _absolute(out_vcf)
    workdir = _absolute(workdir)
    regions = _absolute(regions)
    gtf = _absolute(gtf)
    targets = _absolute_targets(targets)

    _progress("============================================================")
    _progress("Paralome FreeBayes (local)")
    _progress("============================================================")
    _ensure_run_directories(workdir)
    _require_file(reference, "reference FASTA")
    _require_file(Path(str(reference) + ".fai"), "reference FASTA index")
    for tool in ("freebayes", "bcftools", "bgzip", "tabix"):
        _require_tool(tool)

    # The same refusal write_freebayes_slurm makes,
    # before anything is rewritten. The local backend went straight to
    # _prepare_chunked_run, which rewrites the chunk files, bams.list and the
    # scripts and sets the shards aside, while an earlier SLURM array for this
    # workdir could still be pending: its tasks then ran the stored script
    # with the OLD options over the NEW chunks and wrote shards that counted
    # as finished for the new plan. squeue is asked whenever it exists.
    _refuse_beside_a_live_submission(workdir)

    run = _prepare_chunked_run(
        samples=samples,
        reference=reference,
        workdir=workdir,
        backend="local",
        regions=regions,
        targets=targets,
        chunk_size=chunk_size,
        gtf=gtf,
        region_strategy=region_strategy,
        gene_padding=gene_padding,
        jobs=jobs,
        threads=threads,
        verbose=verbose,
        empirical_min_depth=empirical_min_depth,
        empirical_min_samples=empirical_min_samples,
        empirical_merge_distance=empirical_merge_distance,
        empirical_padding=empirical_padding,
        min_alternate_count=min_alternate_count,
        min_alternate_fraction=min_alternate_fraction,
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
        haplotype_length=haplotype_length,
        min_qual=min_qual,
        pooled_discrete=pooled_discrete,
        pooled_continuous=pooled_continuous,
        extra_options=extra_options,
        max_region_length=max_region_length,
        stall_minutes=stall_minutes,
        annotation_mode="none",
    )
    chunks = run.chunks
    if run.already_merged:
        return FreebayesRunResult(out_vcf=_copy_out(run.merge_out_vcf, out_vcf), workdir=workdir, backend="local")
    if chunks.resumed and chunks.missing and len(chunks.missing) < chunks.n_chunks:
        _progress(
            f"Calling again only the {len(chunks.missing):,} of {chunks.n_chunks:,} chunks without a shard: "
            + ", ".join(str(i) for i in chunks.missing),
            indent=2,
        )
    workers = max(1, min(int(run.concurrency or 1), len(chunks.missing) or 1))
    _progress(f"Calling {len(chunks.missing):,} chunk(s), {workers} at a time", indent=2)
    logs = workdir / "logs"

    def _call_chunk(index: int) -> tuple[int, int]:
        env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(index)}
        env.pop("SLURM_ARRAY_JOB_ID", None)
        with (logs / f"freebayes_local_{index}.out").open("w") as out, \
                (logs / f"freebayes_local_{index}.err").open("w") as err:
            proc = subprocess.run(["bash", str(run.task_script)], env=env, stdout=out, stderr=err)
        return index, proc.returncode

    failed: list[int] = []
    if chunks.missing:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, code in pool.map(_call_chunk, chunks.missing):
                if code != 0:
                    failed.append(index)
                    _progress(f"Chunk {index} failed (exit {code}); see {logs / f'freebayes_local_{index}.err'}", indent=4)
    if failed:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            304,
            f"FreeBayes failed on {len(failed)} of {chunks.n_chunks} chunks: {', '.join(map(str, sorted(failed)))}. "
            f"Their errors are in {logs}/freebayes_local_<chunk>.err. Nothing was merged; the finished "
            "shards are kept, and running the same command again calls only the failed chunks.",
        )

    with (logs / "merge_local.out").open("w") as out, (logs / "merge_local.err").open("w") as err:
        merged = subprocess.run(["bash", str(run.merge_script)], stdout=out, stderr=err)
    if merged.returncode != 0:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            304,
            f"Merging the FreeBayes shards failed (exit {merged.returncode}); see {logs / 'merge_local.err'}.",
        )
    final = _copy_out(run.merge_out_vcf, out_vcf)
    _progress(f"FreeBayes complete: {run.merge_out_vcf}", indent=2)
    return FreebayesRunResult(out_vcf=final, workdir=workdir, backend="local")


def _refuse_beside_a_live_submission(workdir: Path, *, submitting: bool = False) -> None:
    """Refuse to plan over a workdir whose recorded SLURM jobs may still be running.

    For the SLURM backend, and shared with the local one. The recorded IDs are
    those in inputs/freebayes_chunk_plan.json; squeue is asked whenever it
    exists, and when it does not and nothing is being submitted the question
    cannot be answered on this machine, so the local run goes on
    (refuse_while_jobs_are_alive would otherwise stop every workstation run
    over a workdir once submitted from a cluster).
    """
    recorded_ids = [str(job) for job in (_read_chunk_plan(workdir).get("submitted_job_ids") or [])]
    if recorded_ids and (submitting or shutil.which("squeue")):
        from paralome.fastq.fastp import refuse_while_jobs_are_alive

        refuse_while_jobs_are_alive(
            "FreeBayes calling", workdir, recorded_ids,
            recorded_in=f"submitted_job_ids in {_chunk_plan_path(workdir)}",
        )


def _copy_out(merged: Path, out_vcf: Path) -> Path:
    """The merged VCF, also copied to --out when that names a VCF file."""
    if _is_vcf_path(out_vcf) and out_vcf != merged and not out_vcf.is_dir():
        out_vcf.parent.mkdir(parents=True, exist_ok=True)
        if out_vcf.name.lower().endswith(".gz"):
            shutil.copyfile(merged, out_vcf)
            shutil.copyfile(Path(str(merged) + ".tbi"), Path(str(out_vcf) + ".tbi"))
        else:
            import gzip

            with gzip.open(merged, "rb") as src, out_vcf.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        _progress(f"Copied the merged VCF to {out_vcf}", indent=2)
        return out_vcf
    return merged


def _normalize_annotation_mode(mode: Optional[str]) -> str:
    value = (mode or "none").strip().lower()
    aliases = {"false": "none", "0": "none", "no": "none", "yes": "all", "true": "all", "1": "all"}
    value = aliases.get(value, value)
    allowed = {"none", "snpeff", "annovar", "all"}
    if value not in allowed:
        raise ValueError(f"Unsupported annotation mode: {mode}. Expected one of: {', '.join(sorted(allowed))}")
    return value


def _annotation_uses_snpeff(settings: AnnotationSettings) -> bool:
    return settings.mode in {"snpeff", "all"}


def _annotation_uses_annovar(settings: AnnotationSettings) -> bool:
    return settings.mode in {"annovar", "all"}


def _resolve_snpeff_jar(path: Optional[Path]) -> Path:
    """--snpeff-jar, required.

    The jar is given, or the run is refused: no guess at where a snpEff.jar
    might be, as for every other tool (_require_tool).
    """
    if path is None:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            101,
            "--annotation snpeff needs --snpeff-jar, the snpEff.jar of the SnpEff this genome's "
            "database was built for. Paralome does not look for one.",
            fix="Add --snpeff-jar /path/to/snpEff.jar (and --snpeff-config when snpEff.config is "
                "not beside it); or call with --annotation none and run `paralome annotate` afterwards",
        )
    _require_file(path, "SnpEff jar")
    return Path(path)


def _resolve_snpeff_config(path: Optional[Path], snpeff_jar: Path) -> Path:
    """--snpeff-config, else snpEff.config beside the jar (SnpEff's own layout)."""
    if path is not None:
        _require_file(path, "SnpEff config")
        return Path(path)
    beside = snpeff_jar.parent / "snpEff.config"
    if beside.exists():
        return beside
    raise FileNotFoundError(
        f"No snpEff.config beside {snpeff_jar}. Use --snpeff-config to name it."
    )


def _snpeff_data_dir(config: Path) -> Path:
    """The folder SnpEff reads databases from: data.dir in the
    config (last one wins; '~' is home, a relative path is against the
    config's folder, as SnpEff resolves it), else <config dir>/data."""
    value = None
    try:
        text = config.read_text(errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        m = re.match(r"\s*data\.dir\s*[=:]\s*(.*?)\s*$", line)
        if m and m.group(1):
            value = m.group(1)
    if not value:
        return config.parent / "data"
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.parent / path


def _validate_snpeff_database(config: Path, genome: str) -> None:
    data_dir = _snpeff_data_dir(config) / genome
    if not data_dir.exists():
        raise FileNotFoundError(f"SnpEff genome database not found: {data_dir}")


def _resolve_annovar_dir(path: Optional[Path]) -> Path:
    """--annovar-dir, required (see _resolve_snpeff_jar)."""
    if path is None:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            101,
            "--annotation annovar needs --annovar-dir, the ANNOVAR folder holding table_annovar.pl. "
            "Paralome does not look for one.",
            fix="Add --annovar-dir /path/to/annovar (and --annovar-db when the databases are not in "
                "its humandb/); or call with --annotation none and run `paralome annotate` afterwards",
        )
    d = Path(path)
    _require_file(d / "table_annovar.pl", "ANNOVAR table_annovar.pl")
    return d


def _resolve_annovar_db(path: Optional[Path], annovar_dir: Path) -> Path:
    d = Path(path) if path is not None else annovar_dir / "humandb"
    if not d.exists():
        raise FileNotFoundError(f"ANNOVAR humandb directory not found: {d}")
    return d


def _validate_annovar_protocols(db: Path, buildver: str, protocol: str, operation: str) -> None:
    protocols = [p.strip() for p in protocol.split(",") if p.strip()]
    operations = [o.strip() for o in operation.split(",") if o.strip()]
    if len(protocols) != len(operations):
        raise ValueError(
            f"ANNOVAR protocol/operation length mismatch: {len(protocols)} protocols vs {len(operations)} operations"
        )
    missing: list[str] = []
    for proto in protocols:
        # ANNOVAR database files are usually named <buildver>_<protocol>.*.
        if not list(db.glob(f"{buildver}_{proto}*")):
            missing.append(proto)
    if missing:
        raise FileNotFoundError(
            "Missing ANNOVAR database files for build "
            f"{buildver}: {', '.join(missing)} in {db}. "
            "Download them with annotate_variation.pl -downdb -webfrom annovar where appropriate."
        )


def _resolve_annotation_settings(
    *,
    annotation: str,
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
    snpeff_memory: str,
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
) -> AnnotationSettings:
    mode = _normalize_annotation_mode(annotation)
    if mode == "none":
        return AnnotationSettings(mode="none")

    # NO BUILD OF PARALOME'S OWN. --annotation annovar with nothing
    # else annotated any genome against hg38 and the human clinvar/gnomAD
    # tables, and snpeff against GRCh38.99, without a word.
    needed = []
    if mode in {"snpeff", "all"}:
        needed.append(("--snpeff-genome", snpeff_genome))
    if mode in {"annovar", "all"}:
        needed += [
            ("--annovar-buildver", annovar_buildver),
            ("--annovar-protocol", annovar_protocol),
            ("--annovar-operation", annovar_operation),
        ]
    missing = [option for option, value in needed if not str(value or "").strip()]
    if missing:
        from paralome.errors import ParalomeError

        raise ParalomeError(
            101,
            f"--annotation {mode} needs {', '.join(missing)}. Paralome never assumes a genome "
            "build or its annotation databases.",
            fix="Add the option(s) named, for the genome these reads were aligned to; "
                "or call with --annotation none and run `paralome annotate --genome NAME` afterwards",
        )

    resolved_snpeff_jar = None
    resolved_snpeff_config = None
    resolved_annovar_dir = None
    resolved_annovar_db = None

    if mode in {"snpeff", "all"}:
        _require_tool("java")
        resolved_snpeff_jar = _resolve_snpeff_jar(snpeff_jar)
        resolved_snpeff_config = _resolve_snpeff_config(snpeff_config, resolved_snpeff_jar)
        _validate_snpeff_database(resolved_snpeff_config, snpeff_genome)
        _progress("SnpEff annotation enabled", indent=2)
        _progress(f"snpEff jar   : {resolved_snpeff_jar}", indent=4)
        _progress(f"snpEff config: {resolved_snpeff_config}", indent=4)
        _progress(f"snpEff genome: {snpeff_genome}", indent=4)

    if mode in {"annovar", "all"}:
        _require_tool("perl")
        resolved_annovar_dir = _resolve_annovar_dir(annovar_dir)
        resolved_annovar_db = _resolve_annovar_db(annovar_db, resolved_annovar_dir)
        _validate_annovar_protocols(resolved_annovar_db, annovar_buildver, annovar_protocol, annovar_operation)
        _progress("ANNOVAR annotation enabled", indent=2)
        _progress(f"ANNOVAR dir : {resolved_annovar_dir}", indent=4)
        _progress(f"ANNOVAR db  : {resolved_annovar_db}", indent=4)
        _progress(f"Build       : {annovar_buildver}", indent=4)
        _progress(f"Protocols   : {annovar_protocol}", indent=4)

    return AnnotationSettings(
        mode=mode,
        snpeff_jar=resolved_snpeff_jar,
        snpeff_config=resolved_snpeff_config,
        snpeff_genome=snpeff_genome,
        snpeff_memory=snpeff_memory,
        annovar_dir=resolved_annovar_dir,
        annovar_db=resolved_annovar_db,
        annovar_buildver=annovar_buildver,
        annovar_protocol=annovar_protocol,
        annovar_operation=annovar_operation,
    )



def _absolute(path):
    """An absolute path, keeping the folder names the user gave.

    Job scripts carry absolute paths, so a relative --out works wherever the
    job starts. os.path.abspath rather than resolve(): symlinked folders keep
    the name they were given.
    """
    if path is None:
        return None
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _absolute_targets(targets):
    value = _normalize_targets_value(targets)
    if value in {"auto", "genome"}:
        return targets
    return str(_absolute(value))


@dataclass
class ChunkedRun:
    """Everything one FreeBayes run prepared before calling."""

    workdir: Path
    bam_list: Path
    bams: list[Path]
    opts: list[str]
    chunks: ChunkPlan
    merge_out_vcf: Path
    task_script: Path
    merge_script: Path
    concurrency: int
    array_plan: object
    #: Every shard of this plan is there and was merged, under this
    #: same fingerprint, into the indexed VCF still in results/: nothing to do.
    already_merged: bool = False


def _merged_under_this_plan(workdir: Path, merge_out_vcf: Path, chunks: "ChunkPlan") -> bool:
    """Whether the merged VCF in results/ is this plan's, finished.

    Only when the plan was kept (same chunks, options and BAMs), no chunk is
    missing a shard, the VCF is indexed after it was written, and the summary
    the merge wrote names this plan's fingerprint. A stage issued again over
    such a result does not call or merge anything again.
    """
    from paralome.checkpoints import vcf_with_index_ready

    if not chunks.resumed or chunks.missing or not vcf_with_index_ready(merge_out_vcf):
        return False
    try:
        summary = json.loads((_results_dir(workdir) / "freebayes_summary.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(summary, dict) or summary.get("merged_vcf") != str(merge_out_vcf):
        return False
    return (summary.get("chunk_plan") or {}).get("fingerprint") == chunks.fingerprint


def _prepare_chunked_run(
    *,
    samples: Path,
    reference: Path,
    workdir: Path,
    backend: str,
    regions: Optional[Path],
    targets,
    chunk_size: int,
    gtf: Optional[Path],
    region_strategy: str,
    gene_padding: int,
    jobs: int,
    threads: int,
    verbose: int,
    empirical_min_depth: int,
    empirical_min_samples: int,
    empirical_merge_distance: int,
    empirical_padding: int,
    min_alternate_count: int,
    min_alternate_fraction: float,
    min_base_quality: int,
    min_mapping_quality: int,
    haplotype_length: int,
    min_qual: float = 0.0,
    pooled_discrete: bool,
    pooled_continuous: bool,
    extra_options: Optional[str],
    max_region_length: int,
    stall_minutes: float,
    annotation_mode: str,
    sbatch_lines: Optional[dict] = None,
    copy_to: Optional[Path] = None,
) -> ChunkedRun:
    """Regions, the cap, the chunk plan and both scripts, for either backend.

    The SLURM and local backends call the same chunks with the same
    task script and merge with the same merge script, so they write the same
    files: shards/shard_<i>.vcf, results/<workdir>.freebayes.merged.vcf.gz and
    its .tbi, results/freebayes_summary.json and checkpoints/pipeline.json.

    THE INPUTS ARE CHECKED AND THE REGIONS BUILT BEFORE pipeline.json IS RESET.
    The BAM checks and region discovery (for the empirical strategy, a scan of
    every BAM that can take hours) come first; no earlier result or checkpoint
    changes until the plan is known, and the set-aside of the earlier merged
    VCF and the reset happen together. So a run that fails in that window (a
    new sample's .bai missing, say) cannot leave the previous cohort's merged
    VCF in results/ to be taken by glob as the stage's complete result.
    """
    _stall_seconds(stall_minutes)  # refuses a negative value before any work
    freebayes = _require_tool("freebayes")
    # Read before it is reset below, to be put back if this run
    # finds the result already finished under the same plan.
    try:
        previous_checkpoint = json.loads(_pipeline_state_path(workdir).read_text())
    except (OSError, ValueError):
        previous_checkpoint = None
    merge_out_vcf = _results_dir(workdir) / f"{_run_prefix(workdir)}.freebayes.merged.vcf.gz"
    bams = [_absolute(b) for b in _manifest_bams(samples)]
    _progress(f"Samples/BAMs: {len(bams)}", indent=2)
    region_file = _prepare_regions(
        reference=reference,
        workdir=workdir,
        regions=regions,
        targets=targets,
        bams=bams,
        chunk_size=chunk_size,
        empirical_min_depth=empirical_min_depth,
        empirical_min_samples=empirical_min_samples,
        empirical_merge_distance=empirical_merge_distance,
        empirical_padding=empirical_padding,
        min_mapping_quality=min_mapping_quality,
        threads=threads,
        gtf=gtf,
        region_strategy=region_strategy,
        gene_padding=gene_padding,
    )
    cap = cap_region_lengths(region_file, workdir, int(max_region_length), reference)

    _progress("Building FreeBayes command options")
    opts = _freebayes_options(
        reference=reference,
        min_alternate_count=min_alternate_count,
        min_alternate_fraction=min_alternate_fraction,
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
        haplotype_length=haplotype_length,
        pooled_discrete=pooled_discrete,
        pooled_continuous=pooled_continuous,
        extra_options=extra_options,
    )

    scheduler_capabilities = detect_scheduler_capabilities(backend)
    requested_region_tasks = automatic_region_pieces(cap.regions_after, threads) if jobs <= 0 else jobs
    max_tasks = scheduler_capabilities.max_array_task_id if backend == "slurm" else max(1, requested_region_tasks)
    if backend == "slurm":
        requested_array_plan = plan_array_tasks(
            task_count=max(1, requested_region_tasks),
            concurrency=plan_sample_parallel(
                total_threads=threads,
                sample_count=max(1, requested_region_tasks),
                max_threads_per_sample=1,
                jobs=jobs,
            ).concurrent_jobs,
            capabilities=scheduler_capabilities,
        )
    # The most TASKS one array holds, which is MaxArraySize - 1.
    # This read max_array_size, so a request for more pieces than the limit was
    # consolidated to 1,001 chunks and submitted as --array=1-1001, which SLURM
    # rejects.
    planned_region_jobs = min(max(1, requested_region_tasks), max_tasks)
    # Everything is known: from here the workdir changes. The earlier result is
    # set aside (plan_region_chunks) and the state reset in the same breath.
    chunks = plan_region_chunks(
        workdir=workdir,
        region_cap=cap,
        jobs=planned_region_jobs,
        opts=opts,
        # The merge's QUAL cut decides the result too: a changed --min-qual merges again.
        fingerprint_extra=["--paralome-min-qual", f"{float(min_qual):g}"],
        bams=bams,
        stall_minutes=stall_minutes,
        reference=reference,
        freebayes=freebayes,
    )
    if annotation_mode != "none":
        # The annotation chain rewrites these; until it has, an
        # earlier run's annotated VCF must not read as this run's result.
        _set_aside_annotated_results(workdir)
    _write_run_metadata(
        workdir=workdir,
        samples=samples,
        reference=reference,
        backend=backend,
        region_strategy=region_strategy,
        annotation_mode=annotation_mode,
    )
    _force_init_pipeline_json(
        workdir=workdir,
        samples=samples,
        reference=reference,
        backend=backend,
        region_strategy=region_strategy,
        annotation_mode=annotation_mode,
        jobs=jobs,
        threads=threads,
    )
    _gv_log(workdir, "Paralome pipeline initialized", verbosity=verbose, level=1, stage="init")
    _gv_log(workdir, f"Run directory: {workdir}", verbosity=verbose, level=1, stage="init")
    _gv_log(workdir, f"Annotation mode: {annotation_mode}", verbosity=verbose, level=1, stage="init")
    bam_list = _write_bam_list(bams, workdir / "inputs" / "bams.list")
    _progress(f"Wrote BAM list: {bam_list}", indent=2)
    _gv_log(workdir, f"Wrote BAM list: {bam_list}", verbosity=verbose, level=2, stage="inputs")
    _gv_log(
        workdir,
        f"Region length cap {cap.max_region_length} bp: {cap.regions_before} -> {cap.regions_after} regions",
        verbosity=verbose, level=1, stage="regions",
    )
    _record_region_cap_in_dispatch_plan(workdir, cap, chunks)
    task_plan = plan_sample_parallel(
        total_threads=threads,
        sample_count=chunks.n_chunks,
        max_threads_per_sample=1,
        jobs=jobs,
    )
    array_plan = plan_array_tasks(
        task_count=chunks.n_chunks,
        concurrency=task_plan.concurrent_jobs,
        capabilities=scheduler_capabilities,
    ) if backend == "slurm" else None
    _gv_log(workdir, f"Prepared {chunks.n_chunks} region chunks in {chunks.chunk_dir}", verbosity=verbose, level=1, stage="regions")
    if backend == "slurm" and requested_array_plan.batch_count > 1:
        _gv_log(
            workdir,
            f"Requested {requested_region_tasks} tasks exceeds the "
            f"{scheduler_capabilities.max_array_task_id} one array holds (MaxArraySize="
            f"{scheduler_capabilities.max_array_size}); consolidated into {chunks.n_chunks} chunks",
            verbosity=verbose,
            level=1,
            stage="regions",
        )

    _progress("Writing FreeBayes task and merge scripts")
    header = ""
    if sbatch_lines is not None:
        header = (
            f"#SBATCH --job-name={slurm_job_name(workdir, 'freebayes')}\n"
            "#SBATCH --cpus-per-task=1\n"
            f"#SBATCH --mem={shlex.quote(str(sbatch_lines['mem']))}\n"
            f"#SBATCH --array={array_plan.batches[0].array_spec}\n"
            f"#SBATCH --output={shlex.quote(str(workdir))}/logs/freebayes_%A_%a.out\n"
            f"#SBATCH --error={shlex.quote(str(workdir))}/logs/freebayes_%A_%a.err\n"
            f"{sbatch_lines['extra']}"
        )
    task_script = workdir / "scripts" / "freebayes_array.sbatch"
    task_script.write_text(freebayes_array_script(
        workdir=workdir,
        chunk_dir=chunks.chunk_dir,
        bam_list=bam_list,
        freebayes=freebayes,
        opts=opts,
        stall_minutes=stall_minutes,
        sbatch_header=header,
    ))
    task_script.chmod(0o755)
    merge_script = workdir / "scripts" / "merge_freebayes_shards.sh"
    merge_script.write_text(freebayes_merge_script(
        workdir=workdir,
        merge_out_vcf=merge_out_vcf,
        min_qual=min_qual,
        n_chunks=chunks.n_chunks,
        backend=backend,
        annotation_mode=annotation_mode,
        # The merge copies to --out only when it writes the final VCF.
        copy_to=copy_to if annotation_mode == "none" else None,
    ))
    merge_script.chmod(0o755)
    already_merged = annotation_mode == "none" and _merged_under_this_plan(workdir, merge_out_vcf, chunks)
    if already_merged:
        if isinstance(previous_checkpoint, dict) and (previous_checkpoint.get("final") or {}).get("complete"):
            _write_pipeline_state(workdir, previous_checkpoint)
        _progress(
            f"Every one of the {chunks.n_chunks:,} chunks has its shard and the merged VCF is this "
            f"plan's: nothing to call or merge ({merge_out_vcf})",
            indent=2,
        )
    return ChunkedRun(
        workdir=workdir,
        bam_list=bam_list,
        bams=bams,
        opts=opts,
        chunks=chunks,
        merge_out_vcf=merge_out_vcf,
        task_script=task_script,
        merge_script=merge_script,
        concurrency=task_plan.concurrent_jobs,
        array_plan=array_plan,
        already_merged=already_merged,
    )



def write_freebayes_slurm(
    *,
    samples: Path,
    reference: Path,
    out_vcf: Path,
    workdir: Path,
    regions: Optional[Path],
    targets: str | Optional[Path],
    chunk_size: int,
    gtf: Optional[Path],
    region_strategy: str,
    gene_padding: int,
    jobs: int,
    threads: int,
    verbose: int = 0,
    partition: Optional[str],
    mem: str,
    slurm_node: Optional[str],
    slurm_account: Optional[str],
    slurm_qos: Optional[str],
    slurm_oversubscribe: bool,
    submit: bool,
    min_alternate_count: int,
    min_alternate_fraction: float,
    min_base_quality: int,
    min_mapping_quality: int,
    haplotype_length: int,
    min_qual: float = 0.0,
    pooled_discrete: bool,
    pooled_continuous: bool,
    extra_options: Optional[str],
    annotation: str,
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
    snpeff_memory: str,
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
    profile: str = "none",
    empirical_min_depth: int,
    empirical_min_samples: int,
    empirical_merge_distance: int,
    empirical_padding: int,
    max_region_length: int = DEFAULT_MAX_REGION_LENGTH,
    stall_minutes: float = DEFAULT_STALL_MINUTES,
) -> FreebayesRunResult:
    # Every path that goes into a job script is absolute.
    reference = _absolute(reference)
    samples = _absolute(samples)
    out_vcf = _absolute(out_vcf)
    workdir = _absolute(workdir)
    regions = _absolute(regions)
    gtf = _absolute(gtf)
    targets = _absolute_targets(targets)
    _ensure_run_directories(workdir)
    _results_dir(workdir).mkdir(exist_ok=True)

    _require_file(reference, "reference FASTA")
    _require_file(Path(str(reference) + ".fai"), "reference FASTA index")
    _require_tool("freebayes")
    _require_tool("bcftools")
    _require_tool("bgzip")
    _require_tool("tabix")
    _progress("Validating inputs")
    annotation_settings = _resolve_annotation_settings(
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
    )
    partition_line = f"#SBATCH --partition={shlex.quote(str(partition))}\n" if partition else ""
    node_line = f"#SBATCH -w {shlex.quote(str(slurm_node))}\n" if slurm_node else ""
    account_line = f"#SBATCH --account={shlex.quote(str(slurm_account))}\n" if slurm_account else ""
    qos_line = f"#SBATCH --qos={shlex.quote(str(slurm_qos))}\n" if slurm_qos else ""
    oversubscribe_line = "#SBATCH --oversubscribe\n" if slurm_oversubscribe else ""
    # BEFORE anything is planned or written: an earlier array's pending tasks must not call the new
    # chunks with their old options, nor its merge job run the new merge script. squeue is asked whenever
    # it exists, submitting or not.
    _refuse_beside_a_live_submission(workdir, submitting=submit)
    # --out naming a VCF file is honoured on SLURM too (the job
    # that writes the final VCF copies it there); see default_workdir.
    copy_to = out_vcf if _is_vcf_path(out_vcf) and not out_vcf.is_dir() and out_vcf != workdir else None
    run = _prepare_chunked_run(
        samples=samples,
        reference=reference,
        workdir=workdir,
        backend="slurm",
        regions=regions,
        targets=targets,
        chunk_size=chunk_size,
        gtf=gtf,
        region_strategy=region_strategy,
        gene_padding=gene_padding,
        jobs=jobs,
        threads=threads,
        verbose=verbose,
        empirical_min_depth=empirical_min_depth,
        empirical_min_samples=empirical_min_samples,
        empirical_merge_distance=empirical_merge_distance,
        empirical_padding=empirical_padding,
        min_alternate_count=min_alternate_count,
        min_alternate_fraction=min_alternate_fraction,
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
        haplotype_length=haplotype_length,
        min_qual=min_qual,
        pooled_discrete=pooled_discrete,
        pooled_continuous=pooled_continuous,
        extra_options=extra_options,
        max_region_length=max_region_length,
        stall_minutes=stall_minutes,
        annotation_mode=annotation_settings.mode,
        sbatch_lines={
            "mem": mem,
            "extra": f"{partition_line}{node_line}{account_line}{qos_line}{oversubscribe_line}",
        },
        copy_to=copy_to,
    )
    merge_out_vcf = run.merge_out_vcf
    n_regions = run.chunks.n_chunks
    array_plan = run.array_plan
    script = run.task_script
    merge_script = run.merge_script
    merge_sbatch = workdir / "scripts" / "merge_freebayes_shards.sbatch"
    merge_sbatch.write_text(f"""#!/usr/bin/env bash
{partition_line}{node_line}{account_line}{qos_line}#SBATCH --job-name={slurm_job_name(workdir, "freebayes_merge")}
#SBATCH --cpus-per-task=1
#SBATCH --mem={shlex.quote(str(mem))}
#SBATCH --output={shlex.quote(str(workdir))}/logs/merge_%j.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/merge_%j.err

set -euo pipefail
bash {shlex.quote(str(merge_script))}
""")
    merge_sbatch.chmod(0o755)

    annotation_jobs: list[tuple[str, Path]] = []
    annotation_final = merge_out_vcf

    # annotation is performed per FreeBayes shard using the same
    # shard count and SLURM throttle controlled by --jobs and --threads.  This
    # avoids a single long-running ANNOVAR process over the full merged VCF.
    annotation_shard_input_dir = workdir / "shards"
    annotation_final_shard_dir: Optional[Path] = None

    if _annotation_uses_snpeff(annotation_settings):
        snpeff_shard_dir = workdir / "annotation_shards" / "snpeff"
        snpeff_shard_dir.mkdir(parents=True, exist_ok=True)
        snpeff_script = workdir / "scripts" / "annotate_snpeff_array.sbatch"
        snpeff_script.write_text(f"""#!/usr/bin/env bash
{partition_line}{node_line}{account_line}{qos_line}#SBATCH --job-name={slurm_job_name(workdir, "snpeff")}
#SBATCH --cpus-per-task=1
#SBATCH --mem={shlex.quote(str(mem))}
#SBATCH --array={array_plan.batches[0].array_spec}
#SBATCH --output={shlex.quote(str(workdir))}/logs/snpeff_%A_%a.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/snpeff_%A_%a.err

set -euo pipefail

{_pinned_tools_bash()}
ID="${{SLURM_ARRAY_TASK_ID}}"
IN={shlex.quote(str(annotation_shard_input_dir))}/shard_${{ID}}.vcf
OUTDIR={shlex.quote(str(snpeff_shard_dir))}
RAW="$OUTDIR/shard_${{ID}}.raw.vcf"
OUT="$OUTDIR/shard_${{ID}}.vcf"
mkdir -p "$OUTDIR"

if [[ ! -s "$IN" ]]; then
    echo "ERROR: SnpEff input shard not found: $IN" >&2
    exit 1
fi

rm -f "$RAW" "$OUT"
echo "Running SnpEff on shard $ID: $IN"
java -Xmx{annotation_settings.snpeff_memory} \
  -jar {shlex.quote(str(annotation_settings.snpeff_jar))} \
  -c {shlex.quote(str(annotation_settings.snpeff_config))} \
  {shlex.quote(annotation_settings.snpeff_genome)} \
  "$IN" > "$RAW"

if [[ ! -s "$RAW" ]]; then
    echo "ERROR: SnpEff produced empty shard: $RAW" >&2
    exit 1
fi
mv -f "$RAW" "$OUT"
echo "Wrote $OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') SnpEff shard complete: $OUT" >> {shlex.quote(str(workdir / "logs" / "pipeline.log"))}
""")
        snpeff_script.chmod(0o755)
        annotation_jobs.append(("snpeff", snpeff_script))
        annotation_shard_input_dir = snpeff_shard_dir
        annotation_final_shard_dir = snpeff_shard_dir
        annotation_final = _results_dir(workdir) / f"{_run_prefix(workdir)}.snpeff.annotated.vcf.gz"

    if _annotation_uses_annovar(annotation_settings):
        annovar_shard_dir = workdir / "annotation_shards" / "annovar"
        annovar_tmp_dir = workdir / "temp" / "annovar_shards"
        annovar_shard_dir.mkdir(parents=True, exist_ok=True)
        annovar_tmp_dir.mkdir(parents=True, exist_ok=True)
        from paralome.pipeline.slurm import tmp_directive
        from paralome.scratch import job_scratch_bash  # ANNOVAR working files on node scratch

        # SLURM is asked for the scratch a task writes (see
        # paralome.scratch). Its input shard is written by the job before it,
        # so the request is sized from the shards already there (a resumed
        # run); on a fresh run there are none to size it from and only the
        # task's own space guard and the scratch ledger account for it.
        existing_shards = [path.stat().st_size for path in Path(annotation_shard_input_dir).glob("shard_*.vcf")
                           if path.is_file()]
        annovar_tmp_line = tmp_directive(max(existing_shards, default=0) * ANNOVAR_SHARD_SCRATCH_FACTOR,
                                         partition)
        if not annovar_tmp_line:
            annovar_tmp_line = ("# no --tmp: the input shards are written by the job before this one; the scratch\n"
                                "# space guard and its reservation ledger account for this task's working files.\n")
        annovar_script = workdir / "scripts" / "annotate_annovar_array.sbatch"
        annovar_script.write_text(f"""#!/usr/bin/env bash
{partition_line}{node_line}{account_line}{qos_line}#SBATCH --job-name={slurm_job_name(workdir, "annovar")}
#SBATCH --cpus-per-task=1
#SBATCH --mem={shlex.quote(str(mem))}
#SBATCH --array={array_plan.batches[0].array_spec}
{annovar_tmp_line}#SBATCH --output={shlex.quote(str(workdir))}/logs/annovar_%A_%a.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/annovar_%A_%a.err

set -euo pipefail

{_pinned_tools_bash()}
ID="${{SLURM_ARRAY_TASK_ID}}"
IN={shlex.quote(str(annotation_shard_input_dir))}/shard_${{ID}}.vcf
OUTDIR={shlex.quote(str(annovar_shard_dir))}
{job_scratch_bash(shlex.quote(str(annovar_tmp_dir)) + '/shard_"${ID}"', "annovar-shard", var="SHARD_TMP", need_bytes=f'$(stat -c %s "$IN" 2>/dev/null || echo 0) * {ANNOVAR_SHARD_SCRATCH_FACTOR}')}
TMPDIR="$SHARD_TMP"
PREFIX="$TMPDIR/shard_${{ID}}"
ANNOVAR_VCF="$PREFIX.{annotation_settings.annovar_buildver}_multianno.vcf"
OUT="$OUTDIR/shard_${{ID}}.vcf"
mkdir -p "$OUTDIR" "$TMPDIR"

if [[ ! -s "$IN" ]]; then
    echo "ERROR: ANNOVAR input shard not found: $IN" >&2
    exit 1
fi

rm -f "$OUT"
{normalize_for_annotation_bash('"$IN"', '"$TMPDIR/normalized.vcf"', reference, '"$TMPDIR/sort"')}
echo "Running ANNOVAR on shard $ID: $TMPDIR/normalized.vcf"
perl {shlex.quote(str(annotation_settings.annovar_dir / 'table_annovar.pl'))} \
  "$TMPDIR/normalized.vcf" \
  {shlex.quote(str(annotation_settings.annovar_db))} \
  -buildver {shlex.quote(annotation_settings.annovar_buildver)} \
  -out "$PREFIX" \
  -remove \
  -protocol {shlex.quote(annotation_settings.annovar_protocol)} \
  -operation {shlex.quote(annotation_settings.annovar_operation)} \
  -nastring . \
  -vcfinput

if [[ ! -s "$ANNOVAR_VCF" ]]; then
    echo "ERROR: ANNOVAR did not create expected VCF: $ANNOVAR_VCF" >&2
    exit 1
fi
{info_id_rename_bash('"$ANNOVAR_VCF"', '"$TMPDIR/info_id_renames.tsv"')}
mv -f "$ANNOVAR_VCF" "$OUT"
echo "Wrote $OUT"
echo "$(date '+%Y-%m-%dT%H:%M:%S') ANNOVAR shard complete: $OUT" >> {shlex.quote(str(workdir / "logs" / "pipeline.log"))}
""")
        annovar_script.chmod(0o755)
        annotation_jobs.append(("annovar", annovar_script))
        annotation_shard_input_dir = annovar_shard_dir
        annotation_final_shard_dir = annovar_shard_dir
        annotation_final = _results_dir(workdir) / f"{_run_prefix(workdir)}.annovar.annotated.vcf.gz"

    final_annotation_merge_sbatch: Optional[Path] = None
    if annotation_settings.mode != "none" and annotation_final_shard_dir is not None:
        final_annotation_merge_script = workdir / "scripts" / "merge_annotation_shards.sh"
        final_annotation_merge_sbatch = workdir / "scripts" / "merge_annotation_shards.sbatch"
        # THE ANNOTATION-MERGE JOB DOES NOT RUN SEGREGATION OR FILTERING, and does not mark them complete.
        # `segregation` and `filtering` are proper stages in _VARIANT_GRAPH: they write prefixed, bucketed,
        # checkpointed output into their own workdirs, and they fail loudly rather than write a bare name.
        # This job's one responsibility is merging annotated shards.
        filter_profile = (profile or "none").strip().lower().replace("_", "-")
        chunks_total = run.chunks.n_chunks
        # The checkpoint heredoc runs under sys.executable, not PATH's python3.
        final_annotation_merge_script.write_text(f"""#!/usr/bin/env bash
set -euo pipefail

WORKDIR={shlex.quote(str(workdir))}
SHARD_DIR={shlex.quote(str(annotation_final_shard_dir))}
OUT={shlex.quote(str(annotation_final))}
# One stream into $OUT.partial, as the FreeBayes merge; bcftools sort spills to SORT_TMP in the run's own
# folder, never /tmp.
PARTIAL="$OUT.partial"
SORT_TMP="$WORKDIR/temp/bcftools_sort"
mkdir -p "$WORKDIR/temp" "$SORT_TMP" "$(dirname "$OUT")"

require_tool() {{
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required executable not found in PATH: $1" >&2
        exit 1
    fi
}}

{_pinned_tools_bash()}
require_tool bcftools
require_tool bgzip
require_tool tabix

# The shards of THIS plan, by number, every one required (as the FreeBayes merge does), so an earlier
# plan's higher-numbered shards are never taken.
shards=()
for i in $(seq 1 {int(chunks_total)}); do
    f="$SHARD_DIR/shard_${{i}}.vcf"
    if [[ ! -s "$f" ]]; then
        echo "ERROR: annotated shard missing: $f" >&2
        exit 1
    fi
    shards+=("$f")
done

echo "Found ${{#shards[@]}} annotated shard VCFs"
rm -f "$PARTIAL"
{{
    grep '^#' "${{shards[0]}}"
    for f in "${{shards[@]}}"; do
        grep -v '^#' "$f" || true
    done
}} | bcftools sort -T "$SORT_TMP/bcftools." -O v - | bgzip -c > "$PARTIAL"
mv -f "$PARTIAL" "$OUT"
tabix -f -p vcf "$OUT"
rmdir "$SORT_TMP" 2>/dev/null || true
echo "Wrote $OUT"

# Segregation and filtering are their own stages; this job does not run them.

echo "$(date '+%Y-%m-%dT%H:%M:%S') Annotated shard merge complete: $OUT" >> {shlex.quote(str(workdir / "logs" / "pipeline.log"))}
# Recorded by the worker; see freebayes.record_annotation_merge.
{worker_command()} variants freebayes-annotation-record --workdir {shlex.quote(str(workdir))} \
    --mode {shlex.quote(str(annotation_settings.mode))} --output {shlex.quote(str(annotation_final))}
{_copy_out_bash("OUT", copy_to)}""")
        final_annotation_merge_script.chmod(0o755)
        final_annotation_merge_sbatch.write_text(f"""#!/usr/bin/env bash
{partition_line}{node_line}{account_line}{qos_line}#SBATCH --job-name={slurm_job_name(workdir, "annotation_merge")}
#SBATCH --cpus-per-task=1
#SBATCH --mem={shlex.quote(str(mem))}
#SBATCH --output={shlex.quote(str(workdir))}/logs/annotation_merge_%j.out
#SBATCH --error={shlex.quote(str(workdir))}/logs/annotation_merge_%j.err

set -euo pipefail
bash {shlex.quote(str(final_annotation_merge_script))}
""")
        final_annotation_merge_sbatch.chmod(0o755)

    if annotation_settings.mode != "none":
        _progress(f"Annotation mode       : {annotation_settings.mode}", indent=2)
        _progress(f"Annotation shards     : {n_regions}", indent=2)
        _progress(f"Annotation throttle   : {threads}", indent=2)
        _progress(f"Results directory    : {_results_dir(workdir)}", indent=2)
        _progress(f"Final annotated VCF  : {annotation_final}", indent=2)
        # No segregation/filter paths are announced here: this job does not produce them, and printing a
        # path this stage does not write would suggest the stage ran.
        if filter_profile != "none":
            _progress(f"Filter profile       : {filter_profile} (run by the filtering stage)", indent=2)
        for label, annot_script in annotation_jobs:
            _progress(f"Wrote {label} array sbatch: {annot_script}", indent=2)
        if final_annotation_merge_sbatch is not None:
            _progress(f"Wrote annotation merge sbatch: {final_annotation_merge_sbatch}", indent=2)

    job_id = None
    if slurm_node:
        _progress(f"SLURM node constraint: {slurm_node}", indent=2)
    if slurm_account:
        _progress(f"SLURM account        : {slurm_account}", indent=2)
    if slurm_qos:
        _progress(f"SLURM QOS            : {slurm_qos}", indent=2)
    _progress(f"Wrote SLURM script: {script}", indent=2)
    _progress(f"Wrote merge script : {merge_script}", indent=2)
    _progress(f"Wrote merge sbatch : {merge_sbatch}", indent=2)
    _progress(f"Work directory      : {workdir}", indent=2)
    _progress("Setup complete", indent=2)

    if submit and run.already_merged:
        # Issued again over its own finished result -- nothing to submit.
        submit = False
        _progress("Nothing submitted: the FreeBayes result is already complete for this plan", indent=2)
    if submit:
        _require_tool("sbatch")
        chunks = run.chunks
        # Not beside a submission that is still running. Re-running
        # calls only the chunks with no shard; a chunk whose task is still
        # working has none yet, and would be called twice at once.
        if chunks.previous_job_ids:
            from paralome.fastq.fastp import refuse_while_jobs_are_alive

            refuse_while_jobs_are_alive(
                "FreeBayes calling", workdir, chunks.previous_job_ids,
                recorded_in=f"submitted_job_ids in {chunks.plan_path}",
            )
        submitted_ids: list[str] = []

        def _remember(job: str) -> None:
            submitted_ids.append(job)
            plan = _read_chunk_plan(workdir)
            plan["submitted_job_ids"] = list(submitted_ids)
            _write_chunk_plan(workdir, plan)

        array_job_id: Optional[str] = None
        array_cmd = ["sbatch", "--parsable"]
        if chunks.resumed and chunks.missing and len(chunks.missing) < chunks.n_chunks:
            # Command-line --array overrides the script's own #SBATCH --array.
            spec = _array_spec(chunks.missing, run.concurrency)
            array_cmd.append(f"--array={spec}")
            _progress(
                f"Resubmitting {len(chunks.missing):,} of {chunks.n_chunks:,} FreeBayes chunks "
                f"(their shards are missing): {', '.join(str(i) for i in chunks.missing)}",
                indent=2,
            )
        if chunks.missing:
            submit_array = subprocess.run(
                [*array_cmd, str(script)],
                check=True,
                text=True,
                capture_output=True,
            )
            array_job_id = submit_array.stdout.strip().split(";")[0]
            _remember(array_job_id)
            # Named as soon as it is queued, so a refused merge never leaves the array running with no ID
            # reported.
            _progress(f"Submitted FreeBayes array job: {array_job_id}", indent=2)
        else:
            _progress(
                f"All {chunks.n_chunks:,} FreeBayes chunks already have a shard; only the merge is submitted",
                indent=2,
            )

        merge_cmd = ["sbatch", "--parsable"]
        if array_job_id:
            merge_cmd.append(f"--dependency=afterany:{array_job_id}")
        submit_merge = subprocess.run(
            [*merge_cmd, str(merge_sbatch)],
            check=True,
            text=True,
            capture_output=True,
        )
        merge_job_id = submit_merge.stdout.strip().split(";")[0]
        _remember(merge_job_id)
        _progress(f"Submitted dependent merge job: {merge_job_id}", indent=2)

        # Sharded annotation consumes the FreeBayes shard VCFs directly.
        # Therefore annotation must depend on the FreeBayes array, not on the
        # optional raw merged VCF job. The raw merge remains a convenience
        # output in results/.
        if annotation_settings.mode != "none":
            last_job_id = array_job_id
        else:
            last_job_id = merge_job_id

        for label, annot_script in annotation_jobs:
            dependency = [f"--dependency=afterok:{last_job_id}"] if last_job_id else []
            submit_annot = subprocess.run(
                ["sbatch", "--parsable", *dependency, str(annot_script)],
                check=True,
                text=True,
                capture_output=True,
            )
            last_job_id = submit_annot.stdout.strip().split(";")[0]
            _remember(last_job_id)
            _progress(f"Submitted dependent {label} annotation array job: {last_job_id}", indent=2)

        if annotation_settings.mode != "none" and final_annotation_merge_sbatch is not None:
            dependency = [f"--dependency=afterok:{last_job_id}"] if last_job_id else []
            submit_final_annot = subprocess.run(
                ["sbatch", "--parsable", *dependency, str(final_annotation_merge_sbatch)],
                check=True,
                text=True,
                capture_output=True,
            )
            last_job_id = submit_final_annot.stdout.strip().split(";")[0]
            _remember(last_job_id)
            _progress(f"Submitted dependent annotation merge job: {last_job_id}", indent=2)

        job_id = last_job_id

    # Without --submit, the same chain the submit branch above
    # queues, in order with its dependencies -- annotation included.
    manual_steps: list[str] = []
    if not submit and not run.already_merged:
        chunks = run.chunks
        prev = None
        if chunks.missing:
            array_opt = ""
            if chunks.resumed and len(chunks.missing) < chunks.n_chunks:
                array_opt = f" --array={_array_spec(chunks.missing, run.concurrency)}"
            manual_steps.append(f"ARRAY=$(sbatch --parsable{array_opt} {shlex.quote(str(script))})")
            prev = "$ARRAY"
        dep = f" --dependency=afterany:{prev}" if prev else ""
        manual_steps.append(f"MERGE=$(sbatch --parsable{dep} {shlex.quote(str(merge_sbatch))})")
        if annotation_settings.mode == "none":
            prev = "$MERGE"
        for label, annot_script in annotation_jobs:
            dep = f" --dependency=afterok:{prev}" if prev else ""
            var = label.upper()
            manual_steps.append(f"{var}=$(sbatch --parsable{dep} {shlex.quote(str(annot_script))})")
            prev = f"${var}"
        if annotation_settings.mode != "none" and final_annotation_merge_sbatch is not None:
            dep = f" --dependency=afterok:{prev}" if prev else ""
            manual_steps.append(f"sbatch{dep} {shlex.quote(str(final_annotation_merge_sbatch))}")

    return FreebayesRunResult(
        out_vcf=copy_to or annotation_final,
        workdir=workdir,
        backend="slurm",
        submitted=submit,
        job_id=job_id,
        already_complete=run.already_merged,
        manual_steps=tuple(manual_steps),
    )


def _refuse_options_this_command_does_not_run(*, backend: str, annotation: str, profile: str) -> None:
    """Refuse --annotation on the local backend and --profile on either.

    The local backend ran FreeBayes and merged, hard-coding annotation
    'none': ``--backend local --annotation annovar ...`` validated nothing,
    annotated nothing and exited 0 saying the VCF was written. --backend
    local is the default. And --profile was used in one progress line saying
    the filtering stage would run it, which this command never does, on
    either backend. An option that is accepted and does nothing looks like it
    took effect, so both are refused with the command that does the work.
    """
    from paralome.errors import ParalomeError

    mode = _normalize_annotation_mode(annotation)
    if str(backend).strip().lower() == "local" and mode != "none":
        raise ParalomeError(
            103,
            f"--annotation {mode} with --backend local: the local backend calls and merges, and does "
            "not annotate (inline annotation runs as dependent SLURM jobs on --backend slurm). "
            "Nothing was started.",
            fix="Run with --annotation none and annotate the merged VCF afterwards with "
                "`paralome annotate --vcf <merged.vcf.gz> --genome NAME`; or use --backend slurm",
        )
    wanted = (profile or "none").strip().lower().replace("_", "-")
    if wanted != "none":
        raise ParalomeError(
            102,
            f"--profile {wanted}: `variants freebayes` calls variants and does not filter them, on "
            "either backend. The profile belongs to the filtering stage, which the pipeline runs "
            "after segregation. Nothing was started.",
            fix="Drop --profile here; run the full pipeline, or the filtering command, with the profile",
        )


def run_freebayes_workflow(
    *,
    samples: Path,
    reference: Path,
    out_vcf: Path,
    backend: str,
    workdir: Optional[Path],
    threads: int,
    jobs: int,
    verbose: int = 0,
    regions: Optional[Path] = None,
    targets: str | Optional[Path],
    chunk_size: int,
    gtf: Optional[Path],
    region_strategy: str,
    gene_padding: int,
    partition: Optional[str],
    mem: str,
    slurm_node: Optional[str] = None,
    slurm_account: Optional[str] = None,
    slurm_qos: Optional[str] = None,
    slurm_oversubscribe: bool = True,
    submit: bool = False,
    min_alternate_count: int,
    min_alternate_fraction: float,
    min_base_quality: int,
    min_mapping_quality: int,
    haplotype_length: int,
    min_qual: float = 0.0,
    pooled_discrete: bool,
    pooled_continuous: bool,
    extra_options: Optional[str],
    annotation: str,
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
    snpeff_memory: str,
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
    profile: str = "none",
    empirical_min_depth: int,
    empirical_min_samples: int,
    empirical_merge_distance: int,
    empirical_padding: int,
    max_region_length: int = DEFAULT_MAX_REGION_LENGTH,
    stall_minutes: float = DEFAULT_STALL_MINUTES,
) -> FreebayesRunResult:
    out_vcf = _absolute(out_vcf)
    workdir = default_workdir(out_vcf, backend, workdir)
    _refuse_options_this_command_does_not_run(backend=backend, annotation=annotation, profile=profile)

    if backend == "local":
        return run_freebayes_local(
            samples=samples,
            reference=reference,
            out_vcf=out_vcf,
            workdir=workdir,
            threads=threads,
            regions=regions,
            targets=targets,
            chunk_size=chunk_size,
            gtf=gtf,
            region_strategy=region_strategy,
            gene_padding=gene_padding,
            empirical_min_depth=empirical_min_depth,
            empirical_min_samples=empirical_min_samples,
            empirical_merge_distance=empirical_merge_distance,
            empirical_padding=empirical_padding,
            min_alternate_count=min_alternate_count,
            min_alternate_fraction=min_alternate_fraction,
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
            haplotype_length=haplotype_length,
            min_qual=min_qual,
            pooled_discrete=pooled_discrete,
            pooled_continuous=pooled_continuous,
            extra_options=extra_options,
            jobs=jobs,
            verbose=verbose,
            max_region_length=max_region_length,
            stall_minutes=stall_minutes,
        )

    if backend == "slurm":
        return write_freebayes_slurm(
            samples=samples,
            reference=reference,
            out_vcf=out_vcf,
            workdir=workdir,
            regions=regions,
            targets=targets,
            chunk_size=chunk_size,
            gtf=gtf,
            region_strategy=region_strategy,
            gene_padding=gene_padding,
            empirical_min_depth=empirical_min_depth,
            empirical_min_samples=empirical_min_samples,
            empirical_merge_distance=empirical_merge_distance,
            empirical_padding=empirical_padding,
            jobs=jobs,
            threads=threads,
            verbose=verbose,
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
            extra_options=extra_options,
            annotation=annotation,
            profile=profile,
            snpeff_jar=snpeff_jar,
            snpeff_config=snpeff_config,
            snpeff_genome=snpeff_genome,
            snpeff_memory=snpeff_memory,
            annovar_dir=annovar_dir,
            annovar_db=annovar_db,
            annovar_buildver=annovar_buildver,
            annovar_protocol=annovar_protocol,
            annovar_operation=annovar_operation,
            max_region_length=max_region_length,
            stall_minutes=stall_minutes,
        )

    raise ValueError(f"Unsupported backend: {backend}")
