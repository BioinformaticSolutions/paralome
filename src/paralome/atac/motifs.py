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
import hashlib
import heapq
import gzip
import json
import math
import os
import multiprocessing
import pickle
import re
import sqlite3
import urllib.request
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, repeat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from paralome.stats import benjamini_hochberg, fisher_exact_2x2
from paralome.config.user import config_dir
from paralome.binaries import shell_join, worker_argv
from paralome.errors import ParalomeError
from paralome.pipeline.jsonio import version_stale_reason
from paralome.checkpoints import exclusive_checkpoint_lock

#: Peaks wider than this are skipped by the motif scan. The
#: offending loci are skipped and the number dropped is reported, so a user
#: who wants them back raises --max-peak-width; the run itself never fails
#: over them. The limit is a scan-cost guard, not a correctness one, so
#: exceeding it is a reason to skip a locus and say so -- never a reason to
#: lose the run.
DEFAULT_MAX_PEAK_WIDTH = 5000

#: A peak whose contig is not in the reference FASTA is skipped and
#: counted like an over-wide one (see IndexedFasta.find_contig). Above this
#: fraction of the peak set the names are no longer a few unplaced scaffolds
#: but a naming convention mismatch between the peaks and the genome, and the
#: stage refuses (PLM-0219) rather than scan a fraction of the data.
MAX_UNRESOLVED_CONTIG_FRACTION = 0.01

#: Names how the single-motif enrichment is computed, recorded in
#: checkpoints/motif_analysis.json as ``method``. pipeline.state reads a
#: checkpoint under another value, or none, as out of date, so the stage runs
#: again instead of its numbers being reused. v3: the enrichment universe
#: (global_peaks, and the up/down/non-significant buckets) is the SCANNED
#: peaks only, so a peak skipped for its width or its contig does not dilute
#: every rate; the comparison the results are named after is the one the
#: differential tested; and the pipeline's motif stage reads the
#: deduplicated-BAM manifest, not the raw sample sheet. v4: the enrichment
#: runs once per comparison of every statistic folder the differential wrote
#: (each statistic its own forked set of files), into results/<folder>/, and
#: the checkpoint records one entry per (folder, comparison, stratum).
MOTIF_ANALYSIS_METHOD = "fisher_exact_per_statistic.v4"

#: Names how the motif-pair Fisher table is built, and is recorded
#: in checkpoints/motif_analysis.json as ``pair_statistics_method``. The
#: co-occurrence count and the margins both count each peak's top
#: --max-motifs-per-peak motifs (counting every hit in the margins would bias
#: every pair toward depletion; see _build_pair_features), and the table's
#: total is the scanned peaks (see MOTIF_ANALYSIS_METHOD). A pairs checkpoint
#: that does not carry this value is read by the pipeline state as not
#: complete, and the pairs stage runs again.
MOTIF_PAIR_STATISTICS_METHOD = "fisher_exact_topn_margins.v3"

#: Schema of the --stage singles hand-off (the pickle and its JSON
#: identity sidecar). A hand-off without this schema predates the identity
#: check and is never resumed from.
MOTIF_SINGLES_STATE_SCHEMA = "paralome.atac.motif_singles_state.v2"


def _peak_width(row: Mapping[str, object]) -> int:
    return int(float(row['end'])) - int(float(row['start']))


from paralome.pipeline.resource_catalog import resolve_genome_resources
from paralome.pipeline.resources import plan_sample_parallel
from paralome.pipeline.scheduler import detect_scheduler_capabilities, plan_array_tasks
from paralome.pipeline.naming import slurm_job_name
from paralome.pipeline.slurm import SlurmOptions, afterok, sbatch, write_sbatch
from paralome.csvio import skip_comment_lines
from paralome.atac.naming import (
    ATAC_MOTIF_MODALITY, ATAC_MOTIF_PAIR_MODALITY, derive_file_prefix, resolve_prefixed_file, result_path, stage_banner,
)
from paralome.atac.statistics import (
    _sniff_cpm_delimiter,
)

BASES = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}
COMPLEMENT_COLUMNS = np.array([3, 2, 1, 0], dtype=int)
DEFAULT_JASPAR_RELEASE = "2026"
DEFAULT_JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/{release}/CORE/"
    "JASPAR{release}_CORE_vertebrates_non-redundant_pfms_meme.txt"
)


@dataclass(frozen=True)
class MotifProfile:
    motif_id: str
    name: str
    probabilities: np.ndarray
    source_url: str = ""

    @property
    def width(self) -> int:
        return int(self.probabilities.shape[0])


@dataclass(frozen=True)
class FastaRecord:
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    """Small random-access FASTA reader backed by a standard ``.fai`` index."""

    def __init__(self, fasta: Union[str, Path]):
        self.path = Path(fasta)
        fai = Path(str(self.path) + ".fai")
        if not fai.exists():
            raise FileNotFoundError(
                f"FASTA index does not exist: {fai}. Run 'samtools faidx {self.path}'."
            )
        self.records: Dict[str, FastaRecord] = {}
        with fai.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                self.records[fields[0]] = FastaRecord(
                    length=int(fields[1]),
                    offset=int(fields[2]),
                    line_bases=int(fields[3]),
                    line_width=int(fields[4]),
                )
        self._handle = self.path.open("rb")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "IndexedFasta":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def find_contig(self, chrom: str) -> Optional[str]:
        """The FASTA's name for ``chrom``, or None when the FASTA does not have it.

        The lookup without the KeyError, for callers that skip and count
        a peak on a contig the genome lacks (10x/Signac peak sets routinely
        carry unplaced scaffolds named KI270728.1 where a UCSC FASTA says
        chrUn_KI270728v1) instead of losing the run over it.
        """
        if chrom in self.records:
            return chrom
        candidates = []
        if chrom.startswith("chr"):
            candidates.append(chrom[3:])
        else:
            candidates.append("chr" + chrom)
        if chrom in {"MT", "M", "chrM", "chrMT"}:
            candidates.extend(["MT", "M", "chrM", "chrMT"])
        for candidate in candidates:
            if candidate in self.records:
                return candidate
        return None

    def resolve_contig(self, chrom: str) -> str:
        contig = self.find_contig(chrom)
        if contig is None:
            raise KeyError(f"Contig {chrom!r} is absent from {self.path}.fai")
        return contig

    def fetch(self, chrom: str, start: int, end: int) -> str:
        contig = self.resolve_contig(chrom)
        record = self.records[contig]
        start = max(0, min(int(start), record.length))
        end = max(start, min(int(end), record.length))
        if end <= start:
            return ""
        pieces: List[bytes] = []
        position = start
        while position < end:
            line_index, line_offset = divmod(position, record.line_bases)
            take = min(end - position, record.line_bases - line_offset)
            byte_offset = record.offset + line_index * record.line_width + line_offset
            self._handle.seek(byte_offset)
            pieces.append(self._handle.read(take))
            position += take
        return b"".join(pieces).decode("ascii").upper()


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return path.open()


@dataclass(frozen=True)
class UnscannedPeaks:
    """The peaks the motif scan leaves out, by CPM-matrix row index.

    ``oversize`` are wider than --max-peak-width; ``unresolved`` lie on a
    contig the reference FASTA does not have (``contigs`` names those). Both
    kinds keep their row index -- every shard header and every hit refers to
    the CPM row -- and simply produce no sequence and no hit. They are also
    OUT of the enrichment universe: a peak that was never scanned is not a
    peak without a motif, and counting it as one diluted every rate and
    pushed every Fisher test toward depletion.
    """

    oversize: frozenset
    unresolved: frozenset
    contigs: Tuple[str, ...]

    @property
    def indices(self) -> frozenset:
        return self.oversize | self.unresolved


def find_unscanned_peaks(
    peak_rows: Sequence[Mapping[str, object]],
    max_peak_width: int,
    fasta: Optional["IndexedFasta"],
) -> UnscannedPeaks:
    """Which peaks the scan skips, and why. One rule for every path (in-process,
    sharded, resumed from a singles hand-off), so the universe the finalizer
    tests over is the set of peaks that were actually scanned."""
    oversize = set()
    unresolved = set()
    contigs: Dict[str, int] = {}
    known: Dict[str, bool] = {}
    for index, row in enumerate(peak_rows):
        if _peak_width(row) > max_peak_width:
            oversize.add(index)
            continue
        if fasta is None:
            continue
        chrom = str(row.get("chrom", ""))
        found = known.get(chrom)
        if found is None:
            found = known[chrom] = fasta.find_contig(chrom) is not None
        if not found:
            unresolved.add(index)
            contigs[chrom] = contigs.get(chrom, 0) + 1
    return UnscannedPeaks(
        oversize=frozenset(oversize), unresolved=frozenset(unresolved),
        contigs=tuple(sorted(contigs, key=lambda name: (-contigs[name], name))),
    )


def refuse_if_contigs_do_not_match(unscanned: UnscannedPeaks, peak_count: int, fasta_path: Union[str, Path]) -> None:
    """PLM-0219 when more than MAX_UNRESOLVED_CONTIG_FRACTION of the peaks lie
    on contigs the FASTA does not have: that is a naming mismatch, not a few
    scaffolds, and scanning the rest would answer a different peak set."""
    if peak_count <= 0 or not unscanned.unresolved:
        return
    fraction = len(unscanned.unresolved) / peak_count
    if fraction <= MAX_UNRESOLVED_CONTIG_FRACTION:
        return
    named = ", ".join(unscanned.contigs[:8]) + (", ..." if len(unscanned.contigs) > 8 else "")
    raise ParalomeError(
        219,
        f"{len(unscanned.unresolved)} of {peak_count} peaks ({fraction:.1%}) lie on contigs "
        f"that {fasta_path} does not have ({named}); more than {MAX_UNRESOLVED_CONTIG_FRACTION:.0%} "
        f"means the peaks and the genome use different contig names.",
        fix="Register the genome the peaks were called on, or rename the peaks' contigs to the FASTA's",
        reference_fasta=str(fasta_path),
    )


def report_unscanned_peaks(unscanned: UnscannedPeaks, peak_count: int, max_peak_width: int, fasta_path: Union[str, Path]) -> None:
    """Say what the scan leaves out, once per run, on every path."""
    if unscanned.oversize:
        print(
            f"[paralome] motif scan: skipping {len(unscanned.oversize)} peak(s) wider than "
            f"--max-peak-width {max_peak_width} bp.",
            flush=True,
        )
    if unscanned.unresolved:
        named = ", ".join(unscanned.contigs[:8]) + (", ..." if len(unscanned.contigs) > 8 else "")
        print(
            f"[paralome] motif scan: skipping {len(unscanned.unresolved)} peak(s) on "
            f"{len(unscanned.contigs)} contig(s) that {fasta_path} does not have ({named}).",
            flush=True,
        )
    if unscanned.indices:
        print(
            f"[paralome] motif scan: scanning {peak_count - len(unscanned.indices)} of {peak_count} peaks; "
            f"the skipped peaks are outside the enrichment universe.",
            flush=True,
        )


def parse_meme_motifs(path: Union[str, Path]) -> List[MotifProfile]:
    """Parse MEME position-probability matrices, including JASPAR batch files."""
    motifs: List[MotifProfile] = []
    with _open_text(Path(path)) as handle:
        lines = iter(handle)
        current_id = ""
        current_name = ""
        for raw in lines:
            line = raw.strip()
            if not line.startswith("MOTIF "):
                continue
            pieces = line.split(maxsplit=2)
            current_id = pieces[1]
            current_name = pieces[2] if len(pieces) > 2 else pieces[1]
            matrix_header = ""
            for raw_header in lines:
                matrix_header = raw_header.strip()
                if matrix_header.startswith("letter-probability matrix:"):
                    break
                if matrix_header.startswith("MOTIF "):
                    raise ValueError(f"Motif {current_id} is missing a probability matrix")
            match = re.search(r"\bw\s*=\s*(\d+)", matrix_header)
            if not match:
                raise ValueError(f"Could not parse motif width for {current_id}")
            width = int(match.group(1))
            matrix: List[List[float]] = []
            for _ in range(width):
                row = next(lines).strip().split()
                if len(row) < 4:
                    raise ValueError(f"Incomplete probability row for {current_id}")
                matrix.append([float(value) for value in row[:4]])
            array = np.asarray(matrix, dtype=float)
            row_sums = array.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            array = array / row_sums
            motifs.append(MotifProfile(current_id, current_name, array))
    if not motifs:
        raise ValueError(f"No MEME motifs were found in {path}")
    return motifs


def parse_jaspar_pfms(path: Union[str, Path]) -> List[MotifProfile]:
    """Parse a batch JASPAR PFM file with A/C/G/T row records."""
    motifs: List[MotifProfile] = []
    motif_id = ""
    name = ""
    rows: Dict[str, List[float]] = {}

    def flush() -> None:
        nonlocal motif_id, name, rows
        if not motif_id:
            return
        if set(rows) != set(BASES):
            raise ValueError(f"JASPAR motif {motif_id} is missing one or more base rows")
        width = len(rows["A"])
        if any(len(rows[base]) != width for base in BASES):
            raise ValueError(f"JASPAR motif {motif_id} has inconsistent row widths")
        counts = np.asarray([[rows[base][index] for base in BASES] for index in range(width)])
        counts += 0.25
        probs = counts / counts.sum(axis=1, keepdims=True)
        motifs.append(MotifProfile(motif_id, name or motif_id, probs))
        motif_id = ""
        name = ""
        rows = {}

    with _open_text(Path(path)) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                pieces = line[1:].split(maxsplit=1)
                motif_id = pieces[0]
                name = pieces[1] if len(pieces) > 1 else motif_id
                continue
            match = re.match(r"^([ACGT])\s*\[?\s*(.*?)\s*\]?$", line)
            if match and motif_id:
                rows[match.group(1)] = [float(value) for value in match.group(2).split()]
    flush()
    if not motifs:
        raise ValueError(f"No JASPAR motifs were found in {path}")
    return motifs


def load_motifs(path: Union[str, Path]) -> List[MotifProfile]:
    path = Path(path)
    with _open_text(path) as handle:
        preview = "".join(handle.readline() for _ in range(6))
    if "MEME version" in preview or "letter-probability matrix" in preview:
        return parse_meme_motifs(path)
    return parse_jaspar_pfms(path)


#: Tokens that identify a species, checked against whatever text is available.
#: Latin binomials are here alongside the common names and the assembly
#: prefixes because the text reaching this function is frequently a binomial --
#: see the note in infer_motif_species.
_SPECIES_TOKENS = (
    ("mouse", ("mouse", "mus_musculus", "mus musculus", "grcm", "mm10", "mm39")),
    ("human", ("human", "homo_sapiens", "homo sapiens", "grch", "hg19", "hg38")),
)


def infer_motif_species(genome: Optional[str], configured: Optional[str] = None) -> str:
    """The species name the motif layer works in, whatever it was handed.

    The configured value is normalised too, not returned verbatim:
    resource_catalog aliases ``motif_species`` to

        ("motif_species", "species_common_name", "species")

    so a genome carrying ``species: "Homo sapiens"`` -- which every genome
    registered through `setup genome add` does -- resolves motif_species to
    "Homo sapiens" without anybody having set that key, and the token matching
    turns that into ``human``.

    The value is a provenance label: it is carried into the plan and recorded
    in the ATAC outputs, but no code selects or filters motifs by it -- the
    JASPAR CORE vertebrates non-redundant set is used whole.

    An unrecognised value is returned as given -- somebody who writes
    "drosophila" means it.
    """
    for candidate in (configured, genome):
        text = str(candidate or "").strip().casefold()
        if not text:
            continue
        for name, tokens in _SPECIES_TOKENS:
            if any(token in text for token in tokens):
                return name
        # A configured value that matches nothing is still the user's answer.
        if candidate is configured:
            return text
    return "vertebrates"


def _download_jaspar(path: Path, release: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = DEFAULT_JASPAR_URL.format(release=release)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the JASPAR CORE vertebrate motif library. "
            f"Configure motif_database for the genome or download {url} to {path}."
        ) from exc
    return path


def resolve_motif_resources(
    *,
    genome: Optional[str],
    reference_fasta: Optional[Union[str, Path]] = None,
    motif_database: Optional[Union[str, Path]] = None,
    motif_species: Optional[str] = None,
    motif_library: str = "jaspar",
    jaspar_release: str = DEFAULT_JASPAR_RELEASE,
    download_motifs: bool = True,
) -> Dict[str, object]:
    resources = resolve_genome_resources(
        genome=genome,
        overrides={
            "reference_fasta": reference_fasta,
            "motif_database": motif_database,
            "motif_species": motif_species,
        },
        required=("reference_fasta",),
    )
    resolved_reference = resources.path("reference_fasta")
    if resolved_reference is None:
        raise ValueError("Could not resolve the reference FASTA")
    configured_species = resources.value("motif_species")
    species = infer_motif_species(resources.genome, configured_species)
    database = resources.path("motif_database")
    database_source = "genome_registry" if database else "jaspar_cache"
    library = str(motif_library or "jaspar").strip().lower()
    if database is None:
        if library != "jaspar":
            raise ValueError("A motif database path is required for non-JASPAR libraries")
        database = (
            config_dir()
            / "motifs"
            / "jaspar"
            / str(jaspar_release)
            / f"JASPAR{jaspar_release}_CORE_vertebrates_non-redundant_pfms_meme.txt"
        )
        if not database.exists():
            if not download_motifs:
                raise FileNotFoundError(
                    f"Motif database is not cached: {database}. Configure motif_database or allow downloading."
                )
            _download_jaspar(database, str(jaspar_release))
            database_source = "downloaded_jaspar_cache"
    if not database.exists():
        raise ParalomeError(104, f"Motif database does not exist: {database}")
    return {
        "genome": resources.genome,
        "reference_fasta": resolved_reference,
        "reference_source": resources.resources["reference_fasta"].source,
        "motif_database": database,
        "motif_database_source": database_source,
        "motif_species": species,
        "motif_library": library,
        "jaspar_release": str(jaspar_release),
        "resource_config": resources.to_dict(),
    }


def _background_from_sequences(sequences: Sequence[str]) -> np.ndarray:
    counts = np.ones(4, dtype=float)
    for sequence in sequences:
        for base, count in Counter(sequence).items():
            index = BASE_TO_INDEX.get(base)
            if index is not None:
                counts[index] += count
    return counts / counts.sum()


def _log_odds_matrix(profile: MotifProfile, background: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    probabilities = np.asarray(profile.probabilities, dtype=float)
    probabilities = probabilities + pseudocount
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.log2(probabilities / np.asarray(background, dtype=float)[None, :])


def _reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    return pwm[::-1, :][:, COMPLEMENT_COLUMNS]


def _score_distribution(
    pwm: np.ndarray,
    background: np.ndarray,
    precision: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    distribution: Dict[int, float] = {0: 1.0}
    integer_pwm = np.rint(np.asarray(pwm) * precision).astype(int)
    for row in integer_pwm:
        next_distribution: Dict[int, float] = defaultdict(float)
        for score, probability in distribution.items():
            for base_index, base_probability in enumerate(background):
                next_distribution[score + int(row[base_index])] += probability * float(base_probability)
        distribution = {score: probability for score, probability in next_distribution.items() if probability > 1e-15}
    scores = np.asarray(sorted(distribution), dtype=float) / precision
    probs = np.asarray([distribution[int(round(score * precision))] for score in scores], dtype=float)
    tails = np.cumsum(probs[::-1])[::-1]
    return scores, np.minimum(1.0, tails)


def _threshold_from_distribution(scores: np.ndarray, tails: np.ndarray, p_threshold: float) -> float:
    indices = np.where(tails <= p_threshold)[0]
    if indices.size:
        return float(scores[int(indices[0])])
    return float(scores[-1] + 1e-9)


def _pvalue_from_score(score: float, scores: np.ndarray, tails: np.ndarray) -> float:
    index = int(np.searchsorted(scores, score, side="left"))
    if index >= len(scores):
        return float(tails[-1])
    return float(tails[index])


def _python_scan(sequence: str, matrices: Sequence[np.ndarray], thresholds: Sequence[float]) -> List[List[Tuple[int, float]]]:
    encoded = np.fromiter((BASE_TO_INDEX.get(base, -1) for base in sequence), dtype=np.int16)
    results: List[List[Tuple[int, float]]] = []
    for pwm, threshold in zip(matrices, thresholds):
        width = pwm.shape[0]
        if encoded.size < width:
            results.append([])
            continue
        windows = np.lib.stride_tricks.sliding_window_view(encoded, width)
        valid = np.all(windows >= 0, axis=1)
        indices = np.where(valid)[0]
        if indices.size == 0:
            results.append([])
            continue
        scores = pwm[np.arange(width)[None, :], windows[indices]].sum(axis=1)
        selected = np.where(scores >= threshold)[0]
        results.append([(int(indices[position]), float(scores[position])) for position in selected])
    return results


def _moods_scanner(matrices: Sequence[np.ndarray], background: np.ndarray, thresholds: Sequence[float]):
    """Create a MOODS scanner from Paralome's width-by-base PWMs.

    Paralome keeps motif score matrices in ``(width, 4)`` orientation for
    NumPy/Python scanning.  MOODS-python requires one row per alphabet symbol,
    i.e. ``(4, width)``.  Passing the internal orientation reaches the native
    ``Scanner.set_motifs`` binding and aborts the interpreter instead of
    raising a Python exception, so transpose only at this native boundary.
    """
    try:
        import MOODS.scan  # type: ignore
    except ImportError:
        return None

    native_matrices = []
    for index, matrix in enumerate(matrices):
        normalized = np.asarray(matrix, dtype=float)
        if normalized.ndim != 2 or normalized.shape[1] != 4:
            raise ValueError(
                "MOODS matrices must use Paralome (width, 4) orientation; "
                f"matrix {index} has shape {normalized.shape}"
            )
        native_matrices.append(normalized.T.tolist())

    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs(native_matrices, background.tolist(), list(thresholds))
    return scanner


def _iter_sequence_batches(
    peak_indices: Sequence[int],
    sequences: Sequence[str],
    separator_length: int,
    max_bases: int,
) -> Iterator[Tuple[str, List[Tuple[int, int, int]]]]:
    pieces: List[str] = []
    mappings: List[Tuple[int, int, int]] = []
    current = 0
    separator = "N" * max(1, separator_length)
    for peak_index in peak_indices:
        sequence = sequences[peak_index]
        addition = len(sequence) + (len(separator) if pieces else 0)
        if pieces and current + addition > max_bases:
            yield "".join(pieces), mappings
            pieces = []
            mappings = []
            current = 0
        if pieces:
            pieces.append(separator)
            current += len(separator)
        start = current
        pieces.append(sequence)
        current += len(sequence)
        mappings.append((start, current, peak_index))
    if pieces:
        yield "".join(pieces), mappings


def _map_hit(position: int, width: int, mappings: Sequence[Tuple[int, int, int]]) -> Optional[Tuple[int, int]]:
    # Batches contain at most a few thousand peaks; binary search avoids linear scans.
    lo = 0
    hi = len(mappings)
    while lo < hi:
        mid = (lo + hi) // 2
        if mappings[mid][0] <= position:
            lo = mid + 1
        else:
            hi = mid
    index = lo - 1
    if index < 0:
        return None
    start, end, peak_index = mappings[index]
    if position < start or position + width > end:
        return None
    return peak_index, position - start


def _prepare_hit_database(path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE hits (motif_idx INTEGER, peak_idx INTEGER, strand TEXT, rel_start INTEGER, rel_end INTEGER, score REAL, pvalue REAL)"
    )
    return connection


def scan_peak_sequences(
    motifs: Sequence[MotifProfile],
    sequences: Sequence[str],
    peak_rows: Sequence[Mapping[str, object]],
    background: np.ndarray,
    p_threshold: float,
    sqlite_path: Path,
    motif_batch_size: int = 64,
    sequence_batch_bases: int = 2_000_000,
    scanner_backend: str = "auto",
) -> Tuple[sqlite3.Connection, List[float], List[Tuple[np.ndarray, np.ndarray]]]:
    """Scan peak sequences and persist hits in a temporary SQLite database."""
    if not 0 < p_threshold < 1:
        raise ValueError("motif p-value threshold must be between zero and one")
    connection = _prepare_hit_database(sqlite_path)
    distributions: List[Tuple[np.ndarray, np.ndarray]] = []
    pwms: List[np.ndarray] = []
    thresholds: List[float] = []
    for profile in motifs:
        pwm = _log_odds_matrix(profile, background)
        scores, tails = _score_distribution(pwm, background)
        threshold = _threshold_from_distribution(scores, tails, p_threshold)
        pwms.append(pwm)
        distributions.append((scores, tails))
        thresholds.append(threshold)

    peak_indices = [index for index, sequence in enumerate(sequences) if sequence]
    max_width = max(profile.width for profile in motifs)
    backend = scanner_backend.strip().lower()
    if backend not in {"auto", "moods", "python"}:
        raise ValueError("scanner backend must be auto, moods, or python")

    for motif_start in range(0, len(motifs), max(1, motif_batch_size)):
        motif_end = min(len(motifs), motif_start + max(1, motif_batch_size))
        scan_matrices: List[np.ndarray] = []
        scan_thresholds: List[float] = []
        scan_meta: List[Tuple[int, str]] = []
        for motif_index in range(motif_start, motif_end):
            scan_matrices.append(pwms[motif_index])
            scan_thresholds.append(thresholds[motif_index])
            scan_meta.append((motif_index, "+"))
            scan_matrices.append(_reverse_complement_pwm(pwms[motif_index]))
            scan_thresholds.append(thresholds[motif_index])
            scan_meta.append((motif_index, "-"))
        scanner = None if backend == "python" else _moods_scanner(scan_matrices, background, scan_thresholds)
        if backend == "moods" and scanner is None:
            raise RuntimeError("MOODS-python is required for --scanner-backend moods")

        for batch_sequence, mappings in _iter_sequence_batches(
            peak_indices, sequences, max_width, max(10_000, int(sequence_batch_bases))
        ):
            if scanner is not None:
                scanned = scanner.scan(batch_sequence)
                normalized_results = [
                    [(int(hit.pos), float(hit.score)) for hit in motif_hits]
                    for motif_hits in scanned
                ]
            else:
                normalized_results = _python_scan(batch_sequence, scan_matrices, scan_thresholds)
            inserts: List[Tuple[int, int, str, int, int, float, float]] = []
            for matrix_index, motif_hits in enumerate(normalized_results):
                motif_index, strand = scan_meta[matrix_index]
                width = motifs[motif_index].width
                score_grid, tail_grid = distributions[motif_index]
                for position, score in motif_hits:
                    mapped = _map_hit(position, width, mappings)
                    if mapped is None:
                        continue
                    peak_index, relative_start = mapped
                    pvalue = _pvalue_from_score(score, score_grid, tail_grid)
                    if pvalue > p_threshold * 1.0000001:
                        continue
                    inserts.append(
                        (
                            motif_index,
                            peak_index,
                            strand,
                            relative_start,
                            relative_start + width,
                            score,
                            pvalue,
                        )
                    )
            if inserts:
                connection.executemany("INSERT INTO hits VALUES (?, ?, ?, ?, ?, ?, ?)", inserts)
                connection.commit()
    connection.execute("CREATE INDEX hits_motif_peak ON hits(motif_idx, peak_idx)")
    connection.execute("CREATE INDEX hits_peak ON hits(peak_idx)")
    connection.commit()
    return connection, thresholds, distributions



def _compute_thresholds_distributions(
    motifs: Sequence[MotifProfile],
    background: np.ndarray,
    p_threshold: float,
) -> Tuple[List[float], List[Tuple[np.ndarray, np.ndarray]]]:
    thresholds: List[float] = []
    distributions: List[Tuple[np.ndarray, np.ndarray]] = []
    for profile in motifs:
        pwm = _log_odds_matrix(profile, background)
        scores, tails = _score_distribution(pwm, background)
        thresholds.append(_threshold_from_distribution(scores, tails, p_threshold))
        distributions.append((scores, tails))
    return thresholds, distributions


def _read_peak_index_file(path: Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _read_motif_shard_fasta(path: Path) -> Tuple[List[int], List[str]]:
    indices: List[int] = []
    sequences: List[str] = []
    current: Optional[int] = None
    pieces: List[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current is not None:
                    indices.append(current)
                    sequences.append(''.join(pieces).upper())
                current = int(line[1:].split()[0])
                pieces = []
            else:
                pieces.append(line)
    if current is not None:
        indices.append(current)
        sequences.append(''.join(pieces).upper())
    return indices, sequences


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_stat(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        'path': str(path),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _stable_digest(payload: Mapping[str, object]) -> str:
    # default=str, matching the other three copies of this helper
    # (peak_merge, singlecell, rna.singlecell). This one alone omitted it, so a
    # payload carrying a Path -- which every other copy tolerates -- would raise
    # TypeError here instead of hashing. Digests of currently-hashed payloads are
    # unchanged: default= only fires on values json cannot otherwise encode.
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _sqlite_hit_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f'Motif SQLite output is missing or empty: {path}')
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        integrity = connection.execute('PRAGMA quick_check').fetchone()
        if not integrity or integrity[0] != 'ok':
            raise RuntimeError(f'Motif SQLite integrity check failed: {path}')
        columns = [row[1] for row in connection.execute('PRAGMA table_info(hits)')]
        expected = ['motif_idx', 'peak_idx', 'strand', 'rel_start', 'rel_end', 'score', 'pvalue']
        if columns != expected:
            raise RuntimeError(f'Motif SQLite schema mismatch in {path}: {columns!r}')
        return int(connection.execute('SELECT COUNT(*) FROM hits').fetchone()[0])
    finally:
        connection.close()


def _task_signature(plan: Mapping[str, object], task: Mapping[str, object]) -> str:
    return _stable_digest({
        'plan_id': plan['plan_id'],
        'task_id': int(task['task_id']),
        'fasta_sha256': task.get('fasta_sha256'),
        'motif_database_sha256': plan.get('motif_database_sha256'),
        'background': plan.get('background'),
        'motif_p_threshold': plan.get('motif_p_threshold'),
        'scanner_backend': plan.get('scanner_backend'),
        'motif_batch_size': plan.get('motif_batch_size'),
        'sequence_batch_bases': plan.get('sequence_batch_bases'),
    })


def _validate_prepared_task(plan: Mapping[str, object], task: Mapping[str, object]) -> bool:
    fasta_path = Path(str(task['fasta']))
    index_path = Path(str(task['peak_indices']))
    if not fasta_path.is_file() or not index_path.is_file():
        return False
    if fasta_path.stat().st_size <= 0 or index_path.stat().st_size <= 0:
        return False
    if task.get('fasta_sha256') != _sha256_file(fasta_path):
        return False
    if task.get('peak_indices_sha256') != _sha256_file(index_path):
        return False
    indices = _read_peak_index_file(index_path)
    return len(indices) == int(task.get('peak_count_prepared', -1))


def _validate_scan_checkpoint(plan: Mapping[str, object], task: Mapping[str, object]) -> Optional[Dict[str, object]]:
    checkpoint_path = Path(str(task['checkpoint']))
    sqlite_path = Path(str(task['sqlite']))
    if not checkpoint_path.is_file() or not sqlite_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get('complete') is not True:
            return None
        if checkpoint.get('plan_id') != plan.get('plan_id'):
            return None
        # A shard is scanned on a compute node, and a node-local
        # install can lag the login node's by several releases, so a wheel
        # built on the login node reaches a machine that runs nothing. A
        # version-stamped shard turns that from an invisible wrong answer
        # into a shard that simply re-runs.
        if version_stale_reason(checkpoint) is not None:
            return None
        if checkpoint.get('task_signature') != _task_signature(plan, task):
            return None
        if checkpoint.get('sqlite_sha256') != _sha256_file(sqlite_path):
            return None
        if int(checkpoint.get('hit_count', -1)) != _sqlite_hit_count(sqlite_path):
            return None
        return checkpoint
    except Exception:
        return None


def _run_motif_scan_task_unlocked(plan_path: Union[str, Path], task_id: int) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).read_text())
    prepare_checkpoint = Path(plan['prepare_checkpoint'])
    if not prepare_checkpoint.exists():
        raise RuntimeError('Motif preparation checkpoint is missing')
    prepare_status = json.loads(prepare_checkpoint.read_text())
    if prepare_status.get('complete') is not True or prepare_status.get('plan_id') != plan.get('plan_id'):
        raise RuntimeError('Motif preparation checkpoint is incomplete or stale')
    task = plan['tasks'][int(task_id) - 1]
    cached = _validate_scan_checkpoint(plan, task)
    if cached is not None:
        return cached
    if not _validate_prepared_task(plan, task):
        raise RuntimeError(f'Motif shard preparation is invalid: {task_id}')

    fasta_path = Path(task['fasta'])
    sqlite_path = Path(task['sqlite'])
    checkpoint_path = Path(task['checkpoint'])
    peak_indices, sequences = _read_motif_shard_fasta(fasta_path)
    motifs = load_motifs(Path(plan['motif_database']))
    background = np.asarray(plan['background'], dtype=float)
    temporary_sqlite = sqlite_path.with_name(f'.{sqlite_path.name}.tmp')
    temporary_sqlite.unlink(missing_ok=True)
    connection, _, _ = scan_peak_sequences(
        motifs,
        sequences,
        [{} for _ in sequences],
        background,
        float(plan['motif_p_threshold']),
        temporary_sqlite,
        motif_batch_size=int(plan['motif_batch_size']),
        sequence_batch_bases=int(plan['sequence_batch_bases']),
        scanner_backend=str(plan['scanner_backend']),
    )
    rows = list(connection.execute('SELECT rowid, peak_idx FROM hits'))
    mapping = {local: global_index for local, global_index in enumerate(peak_indices)}
    connection.executemany(
        'UPDATE hits SET peak_idx=? WHERE rowid=?',
        [(mapping[int(local_index)], int(rowid)) for rowid, local_index in rows],
    )
    connection.commit()
    hit_count = int(connection.execute('SELECT COUNT(*) FROM hits').fetchone()[0])
    connection.close()
    os.replace(temporary_sqlite, sqlite_path)
    validated_count = _sqlite_hit_count(sqlite_path)
    if validated_count != hit_count:
        raise RuntimeError(f'Motif shard hit count changed during validation: {task_id}')
    result = {
        'schema': 'paralome.atac.motif_shard.v2',
        'complete': True,
        'plan_id': plan['plan_id'],
        'task_id': int(task_id),
        'task_signature': _task_signature(plan, task),
        'peak_count': len(peak_indices),
        'base_count': sum(len(sequence) for sequence in sequences),
        'hit_count': hit_count,
        'sqlite': str(sqlite_path),
        'sqlite_sha256': _sha256_file(sqlite_path),
    }
    _atomic_json(checkpoint_path, result)
    return result


def run_motif_scan_task(plan_path: Union[str, Path], task_id: int) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).read_text())
    task = plan['tasks'][int(task_id) - 1]
    with exclusive_checkpoint_lock(Path(str(task['checkpoint']))):
        return _run_motif_scan_task_unlocked(plan_path, task_id)


def merge_motif_scan_tasks(plan_path: Union[str, Path]) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    merged = Path(plan['merged_sqlite'])
    temporary = merged.with_name(f'.{merged.name}.tmp')
    temporary.unlink(missing_ok=True)
    connection = _prepare_hit_database(temporary)
    expected_hits = 0
    try:
        for task in plan['tasks']:
            status = _validate_scan_checkpoint(plan, task)
            if status is None:
                raise RuntimeError(f'Motif shard failed validation: {task["task_id"]}')
            source = Path(task['sqlite'])
            expected_hits += int(status['hit_count'])
            alias = f'shard_{int(task["task_id"])}'
            connection.execute(f'ATTACH DATABASE ? AS {alias}', (str(source),))
            connection.execute(f'INSERT INTO hits SELECT * FROM {alias}.hits')
            connection.commit()
            connection.execute(f'DETACH DATABASE {alias}')
        connection.execute('CREATE INDEX hits_motif_peak ON hits(motif_idx, peak_idx)')
        connection.execute('CREATE INDEX hits_peak ON hits(peak_idx)')
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, merged)
    merged_hits = _sqlite_hit_count(merged)
    if merged_hits != expected_hits:
        raise RuntimeError(f'Merged motif hit count mismatch: expected {expected_hits}, observed {merged_hits}')
    return merged




def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    # See pipeline.jsonio.CHECKPOINT_VERSION_KEY. This stamps the
    # motif plan and the prepare/shard/finalize checkpoints alike -- the plan
    # is not a checkpoint, but plan_id is hashed from `identity_payload` in
    # memory, never from the written file, so an extra key here cannot move it.
    from paralome.pipeline.jsonio import stamp_version

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(stamp_version(payload), indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def motif_checkpoint_path(output_dir: Union[str, Path]) -> Path:
    return Path(output_dir) / "checkpoints" / "motif_analysis.json"


def withdraw_motif_checkpoint(output_dir: Union[str, Path], *, reason: str, keep_singles: bool = False) -> bool:
    """Mark the motif checkpoint not complete before a run replaces its results.

    Every path that is about to compute withdraws the record first: the
    local runner before it prepares, the SLURM submitter before sbatch, and
    the finalizer before it writes a byte. So a re-run that overwrites
    results and then fails, or a job still running or dead, never leaves a
    checkpoint saying complete over half-replaced results. ``complete``,
    ``singles_complete`` and ``pairs_complete`` are set False and the reason
    is recorded; the rest of the old record is kept for the reader.
    ``keep_singles`` is for a --stage pairs run that resumes from a verified
    singles hand-off: the single-motif outputs are not touched by it, so that
    half stays vouched for and a crash in pairs re-runs pairs, not the scan.

    Returns whether there was a complete record to withdraw.
    """
    path = motif_checkpoint_path(output_dir)
    try:
        recorded = json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):
        recorded = None
    if not isinstance(recorded, dict):
        return False
    was_complete = bool(recorded.get("complete")) or bool(recorded.get("singles_complete")) or bool(recorded.get("pairs_complete"))
    withdrawn = dict(recorded)
    withdrawn["complete"] = False
    withdrawn["pairs_complete"] = False
    if not keep_singles:
        withdrawn["singles_complete"] = False
    withdrawn["withdrawn"] = {
        "reason": reason,
        "singles_kept": bool(keep_singles and recorded.get("singles_complete")),
    }
    _atomic_json(path, withdrawn)
    if was_complete:
        print(
            f"[paralome] motif checkpoint {path} marked not complete: {reason}. It is written "
            f"again only when this run finishes.",
            flush=True,
        )
    return was_complete


#: The stems of every result file run_motif_analysis writes, under the
#: motif modality stamps. _remove_results_not_written_by_this_run goes by these
#: and nothing else, so another analysis's files in the same folder are never
#: touched.
_MOTIF_RESULT_FILE = re.compile(
    r"^(?:.+_)?(?:motif_statistics|motif_pair_statistics|motif_[pq]_[0-9.eE+-]+|motif_pair_[pq]_[0-9.eE+-]+"
    r"|motif_results|motif_pair_network|motif_instances"
    r"|peak_motif_matrix|candidate_gene_motifs|motif_metadata|motif_scan_summary)"
    r"_atac_motif(?:s|_pairs)\.(?:csv|xlsx|pdf)$"
)


def _motif_result_files(results_dir: Path) -> List[Path]:
    """Every file under ``results_dir`` that this stage's own writers name."""
    if not results_dir.is_dir():
        return []
    found = []
    for path in results_dir.rglob("*"):
        if path.is_file() and _MOTIF_RESULT_FILE.match(path.name):
            found.append(path)
    return sorted(found)


def _remove_results_not_written_by_this_run(results_dir: Path, written: Iterable[Path]) -> List[Path]:
    """Remove this stage's result files that the run now finishing did not write.

    A re-run with --skip-motif-pairs, with a stratum that classified nothing
    this time, with another --alpha, or with no tested motif would otherwise
    leave the earlier run's files under their current names beside the new
    results. Every file this stage writes carries one of two modality stamps
    and one of a fixed set of stems (_MOTIF_RESULT_FILE); whatever matches and
    was not written by this run is removed and named, so every result file on
    disk belongs to the run the checkpoint describes.
    """
    keep = {Path(path).resolve() for path in written}
    removed: List[Path] = []
    for path in _motif_result_files(results_dir):
        if path.resolve() in keep:
            continue
        path.unlink(missing_ok=True)
        removed.append(path)
    # Results are forked per statistic folder (results/<folder>/ and
    # its machine/), so a folder an earlier run wrote and this one did not --
    # its statistic no longer requested -- is left empty by the removal above;
    # the empty folder goes too, so the tree shows only this run's statistics.
    root = results_dir.resolve()
    for path in removed:
        directory = path.parent.resolve()
        while directory != root and root in directory.parents:
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
    if removed:
        print(
            f"[paralome] removed {len(removed)} result file(s) left by an earlier run that this "
            f"run did not produce: " + ", ".join(path.name for path in removed[:12])
            + (", ..." if len(removed) > 12 else ""),
            flush=True,
        )
    return removed


def _file_identity(path: Optional[Union[str, Path]]) -> Optional[Dict[str, object]]:
    """Name, size and mtime of ``path``, or None when there is no such file.

    The name rather than the full path, so a project copied with its
    timestamps (cp -a, rsync -a) still recognises its own hand-off; a file
    replaced or rewritten in place changes size or mtime.
    """
    if not path:
        return None
    resolved = Path(path).expanduser()
    try:
        stat = resolved.stat()
    except OSError:
        return None
    return {'name': resolved.name, 'size': int(stat.st_size), 'mtime_ns': int(stat.st_mtime_ns)}


def _singles_state_identity(
    *, cpm_matrix: Union[str, Path], sample_manifest: Union[str, Path],
    genome: Optional[str], reference_fasta: Optional[Union[str, Path]],
    motif_database: Optional[Union[str, Path]], motif_species: Optional[str],
    motif_library: str, jaspar_release: str, motif_p_threshold: float,
    max_peak_width: int, scanner_backend: str,
) -> Dict[str, object]:
    """What a --stage singles hand-off was computed from.

    Every input of the scan and of the pickled peak/sample rows -- peaks,
    samples, genome, motif library and thresholds -- is named here; a
    hand-off is resumed (and the SLURM planner skips the scan array) only when
    all of them still match, so pairs are never computed from an old peak set
    or an old scan.
    """
    return {
        'schema': MOTIF_SINGLES_STATE_SCHEMA,
        'cpm_matrix': _file_identity(cpm_matrix),
        'sample_manifest': _file_identity(sample_manifest),
        'genome': genome,
        'reference_fasta_override': _file_identity(reference_fasta),
        'motif_database_override': _file_identity(motif_database),
        'motif_species': motif_species,
        'motif_library': str(motif_library),
        'jaspar_release': str(jaspar_release),
        'motif_p_threshold': float(motif_p_threshold),
        'max_peak_width': int(max_peak_width),
        'scanner_backend': str(scanner_backend),
    }


def _singles_state_sidecar(state_path: Path) -> Path:
    return state_path.with_suffix('.json')


def _discard_singles_state(state_path: Path) -> None:
    """Remove a singles hand-off: the sidecar first, so a half-removed pair is never resumed."""
    _singles_state_sidecar(state_path).unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)


def _write_singles_state(state_path: Path, payload: Mapping[str, object], identity: Mapping[str, object]) -> None:
    """Write the singles hand-off atomically, then the sidecar that vouches for it.

    The pickle is written beside its final name and renamed, so a job killed
    mid-dump leaves no truncated file; the sidecar holding its identity is
    written after, and a resume requires both.
    """
    _discard_singles_state(state_path)
    temporary = state_path.with_name(f'.{state_path.name}.tmp')
    with temporary.open('wb') as handle:
        pickle.dump({**payload, 'schema': MOTIF_SINGLES_STATE_SCHEMA, 'identity': dict(identity)}, handle)
    os.replace(temporary, state_path)
    resolved = payload.get('resolved') or {}
    _atomic_json(_singles_state_sidecar(state_path), {
        'schema': MOTIF_SINGLES_STATE_SCHEMA,
        'identity': dict(identity),
        'pickle': _file_identity(state_path),
        # The genome FASTA and motif library the scan actually read, with
        # their paths, so a resource replaced under the same --genome name is
        # noticed too.
        'resolved_files': {
            key: {'path': str(resolved.get(key)), 'identity': _file_identity(resolved.get(key))}
            for key in ('reference_fasta', 'motif_database') if resolved.get(key)
        },
    })


def _singles_state_reuse_problem(state_path: Path, identity: Mapping[str, object]) -> Optional[str]:
    """Why the singles hand-off at ``state_path`` must not be resumed, or None.

    See _singles_state_identity. Reads only the small JSON sidecar, so the
    SLURM planner can decide without loading the pickle.
    """
    if not state_path.exists():
        return f'no singles hand-off at {state_path}'
    sidecar = _singles_state_sidecar(state_path)
    try:
        recorded = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return (
            f'{state_path.name} has no readable identity record ({sidecar.name}); it was '
            f'written by an older Paralome or by an interrupted run, so what it was computed '
            f'from cannot be checked'
        )
    if recorded.get('schema') != MOTIF_SINGLES_STATE_SCHEMA:
        return f'{sidecar.name} has schema {recorded.get("schema")!r}, expected {MOTIF_SINGLES_STATE_SCHEMA!r}'
    stale = version_stale_reason(recorded)
    if stale is not None:
        return f'the singles hand-off was {stale}'
    previous = dict(recorded.get('identity') or {})
    for key, value in identity.items():
        if previous.get(key) != value:
            return f'{key} changed since --stage singles ran (was {previous.get(key)!r}, now {value!r})'
    if _file_identity(state_path) != recorded.get('pickle'):
        return f'{state_path.name} is not the file its identity record describes'
    for key, value in dict(recorded.get('resolved_files') or {}).items():
        entry = dict(value or {})
        if _file_identity(entry.get('path')) != entry.get('identity'):
            return f'the {key.replace("_", " ")} the scan read ({entry.get("path")}) has changed or is gone'
    return None


def _inspect_peak_rows(
    cpm_matrix: Path,
    *,
    max_peak_width: int = DEFAULT_MAX_PEAK_WIDTH,
    max_total_bases: int = 2_000_000_000,
) -> Dict[str, int]:
    peak_count = 0
    total_bases = 0
    widest_peak = 0
    widest_peak_id = ''
    skipped_peaks = 0
    skipped_widest = 0
    skipped_widest_id = ''
    with cpm_matrix.open(newline='') as handle:
        # the CPM matrix is comma-delimited going forward, but an
        # already-completed pre-Patch-109 stage can still be tab-delimited --
        # sniff instead of assuming either.
        reader = csv.DictReader(handle, delimiter=_sniff_cpm_delimiter(cpm_matrix))
        required = {'peak_id', 'chrom', 'start', 'end'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError('CPM matrix is missing peak metadata columns: ' + ', '.join(sorted(missing)))
        for row_number, row in enumerate(reader, start=2):
            try:
                start = int(float(row['start']))
                end = int(float(row['end']))
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid peak coordinates on CPM row {row_number}') from exc
            if end <= start:
                raise ValueError(f'Peak end must exceed start on CPM row {row_number}')
            width = end - start
            peak_id = str(row.get('peak_id') or f'row_{row_number}')
            if width > max_peak_width:
                # Counted and reported, never fatal. total_bases deliberately
                # excludes it too: it is not going to be scanned, so charging
                # the budget for it would make the budget describe a scan that
                # never happens.
                skipped_peaks += 1
                if width > skipped_widest:
                    skipped_widest = width
                    skipped_widest_id = peak_id
                continue
            if width > widest_peak:
                widest_peak = width
                widest_peak_id = peak_id
            peak_count += 1
            total_bases += width
            if total_bases > max_total_bases:
                raise ValueError(
                    f'ATAC motif scan would cover more than {max_total_bases} total bases. '
                    'Rebuild or narrow the cohort peak atlas before motif scanning.'
                )
    if peak_count <= 0:
        # Every peak skipped is a different failure from an empty file, and the
        # only one where --max-peak-width is the answer. Saying "no peak rows"
        # would send the reader to look at the matrix.
        if skipped_peaks:
            raise ParalomeError(
                218,
                f'All {skipped_peaks} peaks are wider than --max-peak-width '
                f'({max_peak_width} bp); the widest is {skipped_widest_id} at '
                f'{skipped_widest} bp, so there is nothing left to scan.',
                fix=f'raise --max-peak-width above {skipped_widest}',
                cpm_matrix=str(cpm_matrix),
            )
        raise ValueError(f'CPM matrix has no peak rows: {cpm_matrix}')
    return {
        'peak_count': peak_count, 'total_bases': total_bases,
        'widest_peak': widest_peak, 'widest_peak_id': widest_peak_id,
        'skipped_peaks': skipped_peaks, 'skipped_widest': skipped_widest,
        'skipped_widest_id': skipped_widest_id, 'max_peak_width': max_peak_width,
    }


def _memory_to_mib(value: str) -> int:
    """Parse a SLURM memory string into MiB for resource scaling."""
    text = str(value).strip().upper()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?)B?", text)
    if not match:
        raise ValueError(f"Unsupported memory value: {value}")
    amount = float(match.group(1))
    unit = match.group(2) or "M"
    factors = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024 * 1024 * 1024}
    return max(1, int(np.ceil(amount * factors[unit])))


def _motif_finalizer_memory(mem: str, biological_samples: int, skip_motif_pairs: bool = False) -> str:
    """Scale motif finalizer memory without inflating the scan-array tasks.

    squeue showed several `motif_analysis_atac_motifs_finalize`
    jobs each reserving the full 128G cap concurrently on the same node --
    a real oversubscription concern. The combinatorial motif-pair
    co-occurrence step (`_build_pair_features`, skippable via
    `--skip-motif-pairs`) is the dominant driver of
    finalize's memory footprint -- its pair_peaks dict and per-pair
    distance/orientation lists can run into the tens of millions of
    entries for a full motif library. Without it, finalize only holds the
    motif-level hit dicts, activity matrices, and log2 CPM matrix -- a
    small fraction of that. Scale both the per-sample factor and the cap
    down substantially when the caller has opted out of pair analysis, so
    multiple concurrent finalize jobs reserve memory proportional to what
    they can actually use, instead of every job claiming the same 128G
    ceiling regardless of whether the expensive step even runs.
    """
    requested = _memory_to_mib(mem)
    if skip_motif_pairs:
        scaled = max(8 * 1024, 2 * 1024 * max(1, int(biological_samples)), requested)
        return f"{min(scaled, 32 * 1024)}M"
    scaled = max(16 * 1024, 5 * 1024 * max(1, int(biological_samples)), requested)
    return f"{min(scaled, 128 * 1024)}M"


#: Arguments that select what the ENRICHMENT is classified against. They do not
#: reach the scan, which reads peaks, a genome and a motif database and nothing
#: else -- so they must not be allowed to invalidate it.
_ENRICHMENT_ONLY_OPTIONS = ("--differential-statistics", "--differential-checkpoint")


def _scan_identity_argv(final_argv: Sequence[str]) -> List[str]:
    """``final_argv`` with the enrichment-only options removed, for the plan hash.

    The scan depends on the peaks, the genome, the motif library and the score
    thresholds. Every one of those is hashed separately above; `cpm_matrix` is
    hashed by size and mtime, so genuinely new peaks still invalidate it. What
    the differential says about those peaks (and the path of its table) is an
    input to the FINALIZER's Fisher's test, not to the scan, so a change there
    does not force a pointless rescan; re-running the finalizer is minutes.

    Removed as (option, value) pairs rather than by name alone, so a value that
    happens to look like an option cannot desynchronise the walk.
    """
    filtered: List[str] = []
    skip_next = False
    for token in final_argv:
        if skip_next:
            skip_next = False
            continue
        if token in _ENRICHMENT_ONLY_OPTIONS:
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in _ENRICHMENT_ONLY_OPTIONS):
            continue
        filtered.append(str(token))
    return filtered


def _prepare_motif_execution_plan(
    *, cpm_matrix: Path, sample_manifest: Path, output_dir: Path,
    genome: Optional[str], reference_fasta: Optional[Path], motif_database: Optional[Path],
    motif_species: Optional[str], motif_library: str, motif_p_threshold: float,
    jaspar_release: str, download_motifs: bool, scanner_backend: str,
    motif_batch_size: int, sequence_batch_bases: int, threads: int, jobs: int,
    backend: str, partition: Optional[str], mem: str, final_argv: Sequence[str],
    skip_motif_pairs: bool = False, stage: str = "all",
    max_peak_width: int = DEFAULT_MAX_PEAK_WIDTH,
) -> Dict[str, object]:
    """Create the shared motif work-unit plan for local or SLURM execution.

    When `stage == "pairs"` and a singles hand-off computed from these inputs
    is on disk, prepare+scan are skipped entirely and only the (cheap,
    dependency-free) finalize job is submitted, which resumes straight from
    the pickle exactly like the local-backend path.
    """
    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in {"local", "slurm"}:
        raise ValueError("Motif backend must be local or slurm")
    normalized_stage = str(stage).strip().lower()
    if normalized_stage not in {"all", "singles", "pairs"}:
        raise ValueError("Motif stage must be all, singles, or pairs")

    cpm_matrix = Path(cpm_matrix).expanduser().resolve()
    sample_manifest = Path(sample_manifest).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    if not cpm_matrix.is_file():
        raise ParalomeError(104, f'CPM matrix does not exist: {cpm_matrix}')
    if not sample_manifest.is_file():
        raise ParalomeError(104, f'Sample manifest does not exist: {sample_manifest}')
    for directory in (out / 'inputs', out / 'scripts', out / 'logs', out / 'checkpoints', out / 'shards'):
        directory.mkdir(parents=True, exist_ok=True)

    singles_state_path = out / 'inputs' / 'motif_singles_state.pkl'
    # The scan is skipped only for a hand-off computed from these
    # same inputs. A bare exists() test skipped it for any pickle at all, and
    # the finalizer then reported pairs from whatever peaks and thresholds the
    # earlier run had used. See _singles_state_identity.
    skip_scan = False
    if normalized_stage == 'pairs':
        resume_problem = _singles_state_reuse_problem(singles_state_path, _singles_state_identity(
            cpm_matrix=cpm_matrix, sample_manifest=sample_manifest, genome=genome,
            reference_fasta=reference_fasta, motif_database=motif_database,
            motif_species=motif_species, motif_library=motif_library,
            jaspar_release=jaspar_release, motif_p_threshold=motif_p_threshold,
            max_peak_width=max_peak_width, scanner_backend=scanner_backend,
        ))
        skip_scan = resume_problem is None
        if resume_problem is not None and singles_state_path.exists():
            print(
                f"[paralome] motif --stage pairs: not resuming from {singles_state_path}: "
                f"{resume_problem}. Scanning again.",
                file=sys.stderr, flush=True,
            )

    biological_sample_count = len(_load_sample_manifest(sample_manifest))
    finalizer_mem = (
        _motif_finalizer_memory(mem, biological_sample_count, skip_motif_pairs)
        if normalized_backend == 'slurm' else mem
    )
    # The CPUs its worker pools use (run_motif_analysis's
    # resource_max_workers), not --threads. It asked for --threads and used
    # min(--threads, --jobs), so a run with --jobs 30 waited for an empty
    # 80-CPU node to use 30 of it.
    finalize_cpus = max(1, min(int(threads), int(jobs)) if int(jobs) > 0 else int(threads))
    peak_summary = _inspect_peak_rows(cpm_matrix, max_peak_width=max_peak_width)
    peak_count = int(peak_summary['peak_count'])
    requested_tasks = int(jobs) if int(jobs) > 0 else max(1, int(threads))
    task_count = max(1, min(requested_tasks, peak_count))
    resource_plan = plan_sample_parallel(
        total_threads=threads, sample_count=task_count, max_threads_per_sample=1, jobs=jobs,
    )

    array_plan = None
    if normalized_backend == 'slurm' and not skip_scan:
        capabilities = detect_scheduler_capabilities('slurm')
        planned_arrays = plan_array_tasks(
            task_count=task_count,
            concurrency=resource_plan.concurrent_jobs,
            capabilities=capabilities,
        )
        if planned_arrays.batch_count != 1:
            raise RuntimeError('Motif task count exceeds one SLURM array; reduce --jobs to scheduler MaxArraySize')
        array_plan = planned_arrays.to_dict()

    tasks: List[Dict[str, object]] = []
    for index in range(1, (0 if skip_scan else task_count) + 1):
        tasks.append({
            'task_id': index,
            'peak_indices': str(out / 'inputs' / f'motif_shard_{index:04d}.indices'),
            'fasta': str(out / 'shards' / f'motif_{index:04d}.fa'),
            'sqlite': str(out / 'shards' / f'motif_{index:04d}.sqlite'),
            'checkpoint': str(out / 'checkpoints' / f'motif_shard_{index:04d}.json'),
        })

    executable_argv = worker_argv()
    identity_payload = {
        'backend': normalized_backend,
        'cpm_matrix': _path_stat(cpm_matrix),
        # NOTE: 'final_argv' below is filtered by _scan_identity_argv. See there.
        'sample_manifest': _path_stat(sample_manifest),
        'peak_count': peak_count,
        'total_peak_bases': int(peak_summary['total_bases']),
        'widest_peak': int(peak_summary['widest_peak']),
        'widest_peak_id': str(peak_summary['widest_peak_id']),
        'task_count': task_count,
        'genome': genome,
        'reference_fasta_override': str(Path(reference_fasta).expanduser().resolve()) if reference_fasta else None,
        'motif_database_override': str(Path(motif_database).expanduser().resolve()) if motif_database else None,
        'motif_species_override': motif_species,
        'motif_library': motif_library,
        'jaspar_release': jaspar_release,
        'download_motifs': bool(download_motifs),
        'motif_p_threshold': float(motif_p_threshold),
        'scanner_backend': scanner_backend,
        'motif_batch_size': int(motif_batch_size),
        'sequence_batch_bases': int(sequence_batch_bases),
        'final_argv': _scan_identity_argv(final_argv),
    }
    plan_id = _stable_digest(identity_payload)
    plan = {
        'schema': 'paralome.atac.motif_execution_plan.v4',
        'stage': 'motif_analysis',
        'requested_stage': normalized_stage,
        'skip_scan': skip_scan,
        'backend': normalized_backend,
        'plan_id': plan_id,
        'created_input_identity': identity_payload,
        'cpm_matrix': str(cpm_matrix),
        # The width limit travels IN THE PLAN. Shard preparation runs
        # in a separate process with only the plan path, so a limit passed as an
        # argument could not reach it -- and a second default there would be a
        # fourth place that decides what "too wide" means.
        'max_peak_width': int(max_peak_width),
        'sample_manifest': str(sample_manifest),
        'output_dir': str(out),
        'peak_count': peak_count,
        'total_peak_bases': int(peak_summary['total_bases']),
        'widest_peak': int(peak_summary['widest_peak']),
        'widest_peak_id': str(peak_summary['widest_peak_id']),
        'genome': genome,
        'reference_fasta_override': identity_payload['reference_fasta_override'],
        'motif_database_override': identity_payload['motif_database_override'],
        'motif_species_override': motif_species,
        'motif_library': motif_library,
        'jaspar_release': jaspar_release,
        'download_motifs': bool(download_motifs),
        'motif_p_threshold': float(motif_p_threshold),
        'scanner_backend': scanner_backend,
        'motif_batch_size': int(motif_batch_size),
        'sequence_batch_bases': int(sequence_batch_bases),
        'tasks': tasks,
        'merged_sqlite': str(out / 'inputs' / 'motif_hits_merged.sqlite'),
        'prepare_checkpoint': str(out / 'checkpoints' / 'motif_prepare.json'),
        'threads': int(threads),
        'jobs': int(jobs),
        'resource_plan': {**resource_plan.to_dict(), 'biological_sample_count': biological_sample_count, 'finalizer_mem': finalizer_mem},
        'array_plan': array_plan,
        'executable_argv': executable_argv,
        'final_argv': list(final_argv),
        # `final_argv` is sys.argv[1:] captured verbatim -- every path in it exactly as typed, possibly
        # relative -- and run_motif_finalize_task re-executes that argv. Rather than absolutize argv
        # token by token (which would change plan_id through _scan_identity_argv and force a rescan),
        # the plan records where it was prepared and the finalizer stands there. One value, no argv
        # rewriting, no effect on the hash.
        'cwd': str(Path.cwd()),
    }
    plan_path = out / 'inputs' / 'motif_scan_plan.json'
    _atomic_json(plan_path, plan)
    result: Dict[str, object] = {
        'plan': plan_path,
        'task_count': task_count,
        'concurrency': resource_plan.concurrent_jobs,
        'plan_id': plan_id,
        'backend': normalized_backend,
        'resource_plan': {**resource_plan.to_dict(), 'biological_sample_count': biological_sample_count, 'finalizer_mem': finalizer_mem},
    }

    if normalized_backend == 'local':
        return result

    cli = shell_join(executable_argv)
    thread_env = 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'

    # Stage-specific job names, so squeue says which pipeline and stage a
    # node is working on (motif_single vs motif_dual). Prepare/scan are
    # always the "singles" computation
    # (scanning + single-motif export), regardless of what triggered them --
    # including a `--stage pairs` run that had to fall back to a full
    # rescan because no singles pickle existed yet. Finalize's name reflects
    # what was actually REQUESTED: "all" -> motif_finalize (does both in one
    # shot), "singles" -> motif_single_finalize, "pairs" -> motif_dual_finalize.
    finalize_stage_label = {
        'all': 'motif_finalize',
        'singles': 'motif_single_finalize',
        'pairs': 'motif_dual_finalize',
    }[normalized_stage]

    if skip_scan:
        # A matching singles hand-off exists -- prepare+scan would just
        # redo the cheap ~10-minute scan for no reason. Submit ONLY the
        # finalize job, no dependency chain needed since there's nothing to
        # wait on.
        finalize_script = write_sbatch(
            out / 'scripts' / 'motif_finalize.sbatch',
            SlurmOptions(
                job_name=slurm_job_name(out, finalize_stage_label), cpus_per_task=finalize_cpus, mem=finalizer_mem,
                partition=partition, output=out / 'logs' / 'motif_finalize_%j.out',
                error=out / 'logs' / 'motif_finalize_%j.err',
            ),
            thread_env + f'{cli} atac motif-finalize --plan {shlex.quote(str(plan_path))}',
        )
        result.update({'finalize_script': finalize_script})
        return result

    assert array_plan is not None
    prepare_script = write_sbatch(
        out / 'scripts' / 'motif_prepare.sbatch',
        SlurmOptions(
            job_name=slurm_job_name(out, 'motif_single_prepare'), cpus_per_task=1, mem=mem,
            partition=partition, output=out / 'logs' / 'motif_prepare_%j.out',
            error=out / 'logs' / 'motif_prepare_%j.err',
        ),
        thread_env + f'{cli} atac motif-prepare --plan {shlex.quote(str(plan_path))}',
    )
    array_script = write_sbatch(
        out / 'scripts' / 'motif_scan_array.sbatch',
        SlurmOptions(
            job_name=slurm_job_name(out, 'motif_single_scan'), cpus_per_task=1, mem=mem,
            partition=partition, array=array_plan['batches'][0]['array_spec'],
            output=out / 'logs' / 'motif_scan_%A_%a.out',
            error=out / 'logs' / 'motif_scan_%A_%a.err',
        ),
        thread_env + f'{cli} atac motif-shard --plan {shlex.quote(str(plan_path))} --task-id "$SLURM_ARRAY_TASK_ID"',
    )
    finalize_script = write_sbatch(
        out / 'scripts' / 'motif_finalize.sbatch',
        SlurmOptions(
            job_name=slurm_job_name(out, finalize_stage_label), cpus_per_task=finalize_cpus, mem=finalizer_mem,
            partition=partition, output=out / 'logs' / 'motif_finalize_%j.out',
            error=out / 'logs' / 'motif_finalize_%j.err',
        ),
        thread_env + f'{cli} atac motif-finalize --plan {shlex.quote(str(plan_path))}',
    )
    result.update({
        'prepare_script': prepare_script,
        'array_script': array_script,
        'finalize_script': finalize_script,
    })
    return result


def prepare_motif_slurm_plan(
    *, cpm_matrix: Path, sample_manifest: Path, output_dir: Path,
    genome: Optional[str], reference_fasta: Optional[Path], motif_database: Optional[Path],
    motif_species: Optional[str], motif_library: str, motif_p_threshold: float,
    jaspar_release: str, download_motifs: bool, scanner_backend: str,
    motif_batch_size: int, sequence_batch_bases: int, threads: int, jobs: int,
    partition: Optional[str], mem: str, final_argv: Sequence[str],
    skip_motif_pairs: bool = False, stage: str = "all",
    max_peak_width: int = DEFAULT_MAX_PEAK_WIDTH,
) -> Dict[str, object]:
    """Write the lightweight three-phase SLURM motif plan."""
    return _prepare_motif_execution_plan(
        max_peak_width=max_peak_width,
        cpm_matrix=cpm_matrix,
        sample_manifest=sample_manifest,
        output_dir=output_dir,
        genome=genome,
        reference_fasta=reference_fasta,
        motif_database=motif_database,
        motif_species=motif_species,
        motif_library=motif_library,
        motif_p_threshold=motif_p_threshold,
        jaspar_release=jaspar_release,
        download_motifs=download_motifs,
        scanner_backend=scanner_backend,
        motif_batch_size=motif_batch_size,
        sequence_batch_bases=sequence_batch_bases,
        threads=threads,
        jobs=jobs,
        backend='slurm',
        partition=partition,
        mem=mem,
        final_argv=final_argv,
        skip_motif_pairs=skip_motif_pairs,
        stage=stage,
    )


def prepare_motif_local_plan(
    *, cpm_matrix: Path, sample_manifest: Path, output_dir: Path,
    genome: Optional[str], reference_fasta: Optional[Path], motif_database: Optional[Path],
    motif_species: Optional[str], motif_library: str, motif_p_threshold: float,
    jaspar_release: str, download_motifs: bool, scanner_backend: str,
    motif_batch_size: int, sequence_batch_bases: int, threads: int, jobs: int,
    max_peak_width: int = DEFAULT_MAX_PEAK_WIDTH,
) -> Dict[str, object]:
    """Write the same work-unit plan used by SLURM without scheduler scripts."""
    # Max_peak_width is passed on. It was not, so a local sharded
    # run scanned with the default width whatever --max-peak-width said, while
    # the finalizer and the singles hand-off recorded the requested one.
    return _prepare_motif_execution_plan(
        max_peak_width=max_peak_width,
        cpm_matrix=cpm_matrix,
        sample_manifest=sample_manifest,
        output_dir=output_dir,
        genome=genome,
        reference_fasta=reference_fasta,
        motif_database=motif_database,
        motif_species=motif_species,
        motif_library=motif_library,
        motif_p_threshold=motif_p_threshold,
        jaspar_release=jaspar_release,
        download_motifs=download_motifs,
        scanner_backend=scanner_backend,
        motif_batch_size=motif_batch_size,
        sequence_batch_bases=sequence_batch_bases,
        threads=threads,
        jobs=jobs,
        backend='local',
        partition=None,
        mem='0',
        final_argv=(),
    )

def _prepare_assignment_database(plan: Mapping[str, object], resolved: Mapping[str, object]) -> Tuple[Path, List[int], List[int]]:
    """Disk-backed longest-first partitioning; all heavy work runs in the preparation job."""
    out = Path(str(plan['output_dir']))
    database = out / 'inputs' / '.motif_assignments.sqlite.tmp'
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute('CREATE TABLE peaks (peak_idx INTEGER PRIMARY KEY, length INTEGER NOT NULL)')
    cpm_matrix_path = Path(str(plan['cpm_matrix']))
    with cpm_matrix_path.open(newline='') as handle:
        # sniff delimiter -- comma going forward, tab for an
        # already-completed pre-Patch-109 upstream stage.
        reader = csv.DictReader(handle, delimiter=_sniff_cpm_delimiter(cpm_matrix_path))
        rows = []
        for peak_idx, row in enumerate(reader):
            length = int(float(row['end'])) - int(float(row['start']))
            rows.append((peak_idx, length))
            if len(rows) >= 10000:
                connection.executemany('INSERT INTO peaks VALUES (?, ?)', rows)
                rows.clear()
        if rows:
            connection.executemany('INSERT INTO peaks VALUES (?, ?)', rows)
    connection.execute('CREATE TABLE assignments (peak_idx INTEGER PRIMARY KEY, task_id INTEGER NOT NULL)')
    heap = [(0, task_id) for task_id in range(len(plan['tasks']))]
    heapq.heapify(heap)
    peak_counts = [0] * len(plan['tasks'])
    base_totals = [0] * len(plan['tasks'])
    assignment_rows = []
    for peak_idx, length in connection.execute('SELECT peak_idx, length FROM peaks ORDER BY length DESC, peak_idx'):
        total, task_id = heapq.heappop(heap)
        assignment_rows.append((int(peak_idx), int(task_id)))
        peak_counts[task_id] += 1
        base_totals[task_id] += int(length)
        heapq.heappush(heap, (total + int(length), task_id))
        if len(assignment_rows) >= 10000:
            connection.executemany('INSERT INTO assignments VALUES (?, ?)', assignment_rows)
            assignment_rows.clear()
    if assignment_rows:
        connection.executemany('INSERT INTO assignments VALUES (?, ?)', assignment_rows)
    connection.execute('CREATE INDEX assignments_task ON assignments(task_id, peak_idx)')
    connection.commit()
    connection.close()
    return database, peak_counts, base_totals


def _run_motif_prepare_task_unlocked(plan_path: Union[str, Path]) -> Dict[str, object]:
    """Resolve resources and atomically materialize balanced shard FASTAs in a scheduled job."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    checkpoint_path = Path(plan['prepare_checkpoint'])
    resolved = resolve_motif_resources(
        genome=plan.get('genome'), reference_fasta=plan.get('reference_fasta_override'),
        motif_database=plan.get('motif_database_override'),
        motif_species=plan.get('motif_species_override'), motif_library=str(plan['motif_library']),
        jaspar_release=str(plan['jaspar_release']), download_motifs=bool(plan['download_motifs']),
    )
    reference_path = Path(resolved['reference_fasta'])
    motif_path = Path(resolved['motif_database'])
    motifs = load_motifs(motif_path)
    current_identity = {
        'plan_id': plan['plan_id'],
        'cpm_matrix_sha256': _sha256_file(Path(plan['cpm_matrix'])),
        'sample_manifest_sha256': _sha256_file(Path(plan['sample_manifest'])),
        'reference_fasta': _path_stat(reference_path),
        'reference_fai_sha256': _sha256_file(Path(str(reference_path) + '.fai')),
        'motif_database_sha256': _sha256_file(motif_path),
    }
    if checkpoint_path.exists():
        try:
            status = json.loads(checkpoint_path.read_text())
            if (
                status.get('complete') is True
                and status.get('input_identity') == current_identity
                and version_stale_reason(status) is None
                and all(_validate_prepared_task(plan, task) for task in plan['tasks'])
            ):
                return status
        except Exception:
            pass

    assignment_database, peak_counts, base_totals = _prepare_assignment_database(plan, resolved)
    assignment_connection = sqlite3.connect(assignment_database)
    assignment_cursor = iter(assignment_connection.execute('SELECT peak_idx, task_id FROM assignments ORDER BY peak_idx'))
    next_assignment = next(assignment_cursor, None)
    temporary_fastas: List[Path] = []
    temporary_indices: List[Path] = []
    fasta_handles = []
    index_handles = []
    base_counts = Counter()
    observed_peak_counts = [0] * len(plan['tasks'])
    observed_base_totals = [0] * len(plan['tasks'])
    try:
        for task in plan['tasks']:
            fasta_path = Path(task['fasta'])
            index_path = Path(task['peak_indices'])
            fasta_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_fasta = fasta_path.with_name(f'.{fasta_path.name}.tmp')
            temporary_index = index_path.with_name(f'.{index_path.name}.tmp')
            temporary_fasta.unlink(missing_ok=True)
            temporary_index.unlink(missing_ok=True)
            temporary_fastas.append(temporary_fasta)
            temporary_indices.append(temporary_index)
            fasta_handles.append(temporary_fasta.open('w'))
            index_handles.append(temporary_index.open('w'))
        cpm_matrix_path = Path(plan['cpm_matrix'])
        max_peak_width = int(plan.get('max_peak_width', DEFAULT_MAX_PEAK_WIDTH))
        oversize_count = 0
        unresolved_contigs: Dict[str, int] = {}
        unresolved_count = 0
        with cpm_matrix_path.open(newline='') as handle, IndexedFasta(reference_path) as fasta:
            # sniff delimiter -- comma going forward, tab for an
            # already-completed pre-Patch-109 upstream stage.
            reader = csv.DictReader(handle, delimiter=_sniff_cpm_delimiter(cpm_matrix_path))
            for peak_idx, peak in enumerate(reader):
                if next_assignment is None or int(next_assignment[0]) != peak_idx:
                    raise RuntimeError(f'Motif assignment is missing peak index {peak_idx}')
                task_id = int(next_assignment[1])
                next_assignment = next(assignment_cursor, None)
                # An over-wide peak is written as an EMPTY record.
                #
                # peak_idx stays the raw CPM row number -- the identity the
                # finalizer indexes peak_rows with -- so skipping must not
                # renumber anything. _read_shard_fasta skips blank lines, so
                # `>N\n\n` reads back as an empty sequence for index N, which
                # scan_peak_sequences already excludes.
                #
                # A peak on a contig the FASTA does not have is written
                # the same way, counted, and named below.
                chrom = str(peak['chrom'])
                if int(float(peak['end'])) - int(float(peak['start'])) > max_peak_width:
                    sequence = ''
                    oversize_count += 1
                elif fasta.find_contig(chrom) is None:
                    sequence = ''
                    unresolved_count += 1
                    unresolved_contigs[chrom] = unresolved_contigs.get(chrom, 0) + 1
                else:
                    sequence = fasta.fetch(chrom, int(float(peak['start'])), int(float(peak['end'])))
                fasta_handles[task_id].write(f'>{peak_idx}\n{sequence}\n')
                index_handles[task_id].write(f'{peak_idx}\n')
                valid = Counter(base for base in sequence if base in BASE_TO_INDEX)
                base_counts.update(valid)
                observed_peak_counts[task_id] += 1
                observed_base_totals[task_id] += len(sequence)
        if next_assignment is not None:
            raise RuntimeError('Motif assignment contains peak indices beyond the CPM matrix')
    finally:
        assignment_connection.close()
        assignment_database.unlink(missing_ok=True)
        for handle in fasta_handles + index_handles:
            handle.close()

    if observed_peak_counts != peak_counts:
        raise RuntimeError('Prepared motif shard peak counts do not match assignment counts')
    # Refuse a contig-name mismatch here, before any shard is scanned.
    unscanned = UnscannedPeaks(
        oversize=frozenset(), unresolved=frozenset(range(unresolved_count)),
        contigs=tuple(sorted(unresolved_contigs, key=lambda name: (-unresolved_contigs[name], name))),
    )
    try:
        refuse_if_contigs_do_not_match(unscanned, sum(observed_peak_counts), reference_path)
    except ParalomeError:
        for temporary in temporary_fastas + temporary_indices:
            temporary.unlink(missing_ok=True)
        raise
    if unresolved_count:
        named = ', '.join(unscanned.contigs[:8]) + (', ...' if len(unscanned.contigs) > 8 else '')
        print(
            f'[paralome] motif prepare: {unresolved_count} peak(s) on {len(unscanned.contigs)} contig(s) '
            f'that {reference_path} does not have ({named}) are written without sequence and will not be scanned.',
            flush=True,
        )
    for task, temp_fasta, temp_index, peak_count, base_count in zip(
        plan['tasks'], temporary_fastas, temporary_indices, observed_peak_counts, observed_base_totals
    ):
        if peak_count <= 0 or temp_fasta.stat().st_size <= 0 or temp_index.stat().st_size <= 0:
            raise RuntimeError(f'Prepared motif shard is empty: {task["task_id"]}')
        os.replace(temp_fasta, Path(task['fasta']))
        os.replace(temp_index, Path(task['peak_indices']))
        task['peak_count_prepared'] = int(peak_count)
        task['base_count_prepared'] = int(base_count)
        task['fasta_sha256'] = _sha256_file(Path(task['fasta']))
        task['peak_indices_sha256'] = _sha256_file(Path(task['peak_indices']))

    total = sum(base_counts.values())
    if total <= 0:
        background = np.full(4, 0.25, dtype=float)
    else:
        background = np.asarray([base_counts[base] / total for base in BASES], dtype=float)
        background = np.maximum(background, 1e-6)
        background = background / background.sum()
    plan['reference_fasta'] = str(reference_path)
    plan['motif_database'] = str(motif_path)
    plan['motif_database_source'] = resolved['motif_database_source']
    plan['motif_species'] = resolved['motif_species']
    plan['motif_database_sha256'] = current_identity['motif_database_sha256']
    plan['background'] = background.tolist()
    _atomic_json(plan_path, plan)
    result = {
        'schema': 'paralome.atac.motif_prepare.v2',
        'complete': True,
        'plan_id': plan['plan_id'],
        'input_identity': current_identity,
        'motifs': len(motifs),
        'peaks': sum(observed_peak_counts),
        'bases': sum(observed_base_totals),
        # What the shards leave out, so the record says what was scanned.
        'peaks_skipped_wide': int(oversize_count),
        'peaks_skipped_unresolved_contig': int(unresolved_count),
        'unresolved_contigs': {name: int(count) for name, count in sorted(unresolved_contigs.items())},
        'reference_fasta': str(reference_path),
        'motif_database': str(motif_path),
        'background': background.tolist(),
        'task_peak_counts': observed_peak_counts,
        'task_base_counts': observed_base_totals,
    }
    _atomic_json(checkpoint_path, result)
    return result


def run_motif_prepare_task(plan_path: Union[str, Path]) -> Dict[str, object]:
    plan = json.loads(Path(plan_path).read_text())
    with exclusive_checkpoint_lock(Path(str(plan['prepare_checkpoint']))):
        return _run_motif_prepare_task_unlocked(plan_path)


def submit_motif_slurm_plan(plan_info: Mapping[str, object], submit: bool) -> Dict[str, object]:
    result = dict(plan_info)
    result.update({'submitted': False, 'job_id': None, 'submitted_job_ids': [], 'complete': False})
    if not submit:
        return result
    # The record of the earlier run is withdrawn before the first sbatch,
    # so the stage does not read as complete while its jobs run or after one
    # of them dies; see withdraw_motif_checkpoint. A plan that skips the scan
    # resumes from a verified singles hand-off, so the singles half stays.
    try:
        plan = json.loads(Path(str(plan_info['plan'])).read_text())
    except (KeyError, OSError, ValueError):
        plan = {}
    if plan.get('output_dir'):
        withdraw_motif_checkpoint(
            plan['output_dir'],
            reason=f"a --stage {plan.get('requested_stage', 'all')} run was submitted to SLURM",
            keep_singles=bool(plan.get('skip_scan')),
        )
    if 'prepare_script' not in plan_info:
        # --stage pairs found an existing singles pickle, so
        # _prepare_motif_execution_plan skipped prepare+scan entirely and
        # only wrote a finalize_script -- submit it standalone, no
        # dependency chain needed since there's nothing to wait on.
        finalize_job = sbatch(plan_info['finalize_script'])
        job_ids = [finalize_job]
        result.update({'submitted': True, 'job_id': ','.join(job_ids), 'submitted_job_ids': job_ids})
        return result
    # Each job ID is reported the moment its sbatch returns, so a refused later sbatch never leaves
    # queued jobs unnamed. stderr, which the supervisor reads as well, so the JSON on stdout stays whole.
    prepare_job = sbatch(plan_info['prepare_script'])
    print(f"Submitted motif prepare job: {prepare_job}", file=sys.stderr, flush=True)
    array_job = sbatch(plan_info['array_script'], dependency=afterok(prepare_job))
    print(f"Submitted motif scan array job: {array_job}", file=sys.stderr, flush=True)
    finalize_job = sbatch(plan_info['finalize_script'], dependency=afterok(array_job))
    print(f"Submitted motif finalizer job: {finalize_job}", file=sys.stderr, flush=True)
    job_ids = [prepare_job, array_job, finalize_job]
    result.update({'submitted': True, 'job_id': ','.join(job_ids), 'submitted_job_ids': job_ids})
    return result


def run_motif_finalize_task(plan_path: Union[str, Path]) -> Dict[str, object]:
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get('skip_scan'):
        # prepare+scan were skipped for this plan (a --stage
        # pairs request found an existing singles pickle) -- there are no
        # shard sqlite files to merge, and run_motif_analysis(stage="pairs")
        # resumes entirely from the pickle, so --precomputed-hit-database/
        # --precomputed-plan are neither needed nor meaningful here.
        merged = None
        command = [
            *[str(value) for value in plan['executable_argv']],
            *[str(value) for value in plan['final_argv']],
        ]
    else:
        merged = merge_motif_scan_tasks(plan_path)
        command = [
            *[str(value) for value in plan['executable_argv']],
            *[str(value) for value in plan['final_argv']],
            '--precomputed-hit-database', str(merged),
            '--precomputed-plan', str(plan_path),
        ]
    # Run where the plan was prepared, so the relative paths in `final_argv` resolve the way they did
    # when they were typed. A plan that does not record it inherits the caller's directory.
    recorded_cwd = str(plan.get('cwd') or '')
    run_cwd: Optional[str] = None
    if recorded_cwd:
        if Path(recorded_cwd).is_dir():
            run_cwd = recorded_cwd
        else:
            # Named, not swallowed: a moved project is a real thing to fix, and
            # a bare chdir failure here would blame the finalizer.
            print(
                f"[paralome] plan recorded cwd {recorded_cwd!r}, which no longer exists; "
                f"running in {Path.cwd()} instead. Any relative path in the plan's argv "
                f"is now resolved against that directory.",
                file=sys.stderr, flush=True,
            )
    subprocess.run(command, check=True, cwd=run_cwd)
    return {
        'complete': True,
        'plan_id': plan['plan_id'],
        'cwd': run_cwd or str(Path.cwd()),
        'merged_sqlite': str(merged) if merged is not None else None,
        'command': command,
    }



def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


# The hit export is parallel and compact:
#
# 1. Per-hit storage is a small slotted dataclass (MotifPeakHit) rather than a
#    7-key dict -- no per-instance __dict__ and no dict hashing overhead, at the
#    scale of millions of (motif, peak) entries. __slots__ is declared on the
#    class body rather than with dataclass(slots=True), which needs Python 3.10.
# 2. The reconstruction is split into --threads chunks of motif indices and
#    run across a fork-based process pool. Each motif's hits (and its own BH
#    q-value correction) are independent of every other motif's, and the
#    SQLite hits table has a `hits_motif_peak` index for per-motif lookups, so
#    concurrent read-only connections are safe and cheap. Each worker streams
#    its slice of instance/matrix rows to a private temp file, which the
#    caller concatenates in motif-index order, so the final CSVs are the same
#    as a serial export.
@dataclass
class MotifPeakHit:
    __slots__ = (
        "hit_count", "best_score", "best_p", "best_q",
        "best_strand", "best_motif_start", "best_motif_end",
    )
    hit_count: int
    best_score: float
    best_p: float
    best_q: float
    best_strand: str
    best_motif_start: int
    best_motif_end: int


_EXPORT_WORKER_SQLITE_PATH: Optional[Path] = None
_EXPORT_WORKER_MOTIFS: Optional[Sequence[MotifProfile]] = None
_EXPORT_WORKER_PEAK_ROWS: Optional[Sequence[Mapping[str, object]]] = None
_EXPORT_WORKER_INSTANCES_BASE: Optional[Path] = None
_EXPORT_WORKER_MATRIX_BASE: Optional[Path] = None

_EXPORT_INSTANCE_FIELDS = [
    "peak_id", "chrom", "peak_start", "peak_end", "motif_id", "motif_name",
    "strand", "motif_start", "motif_end", "motif_score", "motif_p", "motif_q",
    "nearest_gene_id", "nearest_gene_name", "tss_gene_id", "tss_gene_name",
]
_EXPORT_MATRIX_FIELDS = [
    "peak_id", "motif_id", "motif_name", "hit_count", "best_score", "best_p", "best_q",
    "best_strand", "best_motif_start", "best_motif_end",
]


def _export_motif_chunk(motif_indices, worker_index):
    sqlite_path = _EXPORT_WORKER_SQLITE_PATH
    motifs = _EXPORT_WORKER_MOTIFS
    peak_rows = _EXPORT_WORKER_PEAK_ROWS
    assert sqlite_path is not None and motifs is not None and peak_rows is not None
    instances_tmp = _EXPORT_WORKER_INSTANCES_BASE.with_name(_EXPORT_WORKER_INSTANCES_BASE.name + f".{worker_index}")
    matrix_tmp = _EXPORT_WORKER_MATRIX_BASE.with_name(_EXPORT_WORKER_MATRIX_BASE.name + f".{worker_index}")
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    motif_to_peaks = {}
    peak_to_motifs = defaultdict(list)
    try:
        with instances_tmp.open("w", newline="") as instance_out, matrix_tmp.open("w", newline="") as matrix_out:
            instance_writer = csv.writer(instance_out, lineterminator="\n")
            matrix_writer = csv.writer(matrix_out, lineterminator="\n")
            for motif_index in motif_indices:
                motif = motifs[motif_index]
                raw_hits = list(
                    connection.execute(
                        "SELECT peak_idx, strand, rel_start, rel_end, score, pvalue FROM hits WHERE motif_idx=? ORDER BY pvalue, score DESC",
                        (motif_index,),
                    )
                )
                if not raw_hits:
                    continue
                qvalues = benjamini_hochberg([row[5] for row in raw_hits])
                aggregate = {}
                for raw, qvalue in zip(raw_hits, qvalues):
                    peak_index, strand, rel_start, rel_end, score, pvalue = raw
                    peak = peak_rows[int(peak_index)]
                    absolute_start = int(float(peak["start"])) + int(rel_start)
                    absolute_end = int(float(peak["start"])) + int(rel_end)
                    instance_writer.writerow(
                        [
                            peak.get("peak_id", ""), peak.get("chrom", ""), peak.get("start", ""), peak.get("end", ""),
                            motif.motif_id, motif.name, strand, absolute_start, absolute_end,
                            score, pvalue, float(qvalue),
                            peak.get("nearest_gene_id", ""), peak.get("nearest_gene_name", ""),
                            peak.get("tss_gene_id", ""), peak.get("tss_gene_name", ""),
                        ]
                    )
                    item = aggregate.get(int(peak_index))
                    if item is None:
                        item = MotifPeakHit(
                            hit_count=0, best_score=-math.inf, best_p=math.inf, best_q=math.inf,
                            best_strand="", best_motif_start=0, best_motif_end=0,
                        )
                        aggregate[int(peak_index)] = item
                    item.hit_count += 1
                    if float(pvalue) < item.best_p or (
                        float(pvalue) == item.best_p and float(score) > item.best_score
                    ):
                        item.best_score = float(score)
                        item.best_p = float(pvalue)
                        item.best_q = float(qvalue)
                        item.best_strand = strand
                        item.best_motif_start = absolute_start
                        item.best_motif_end = absolute_end
                for peak_index, item in sorted(aggregate.items()):
                    peak = peak_rows[peak_index]
                    matrix_writer.writerow(
                        [
                            peak.get("peak_id", ""), motif.motif_id, motif.name,
                            item.hit_count, item.best_score, item.best_p, item.best_q,
                            item.best_strand, item.best_motif_start, item.best_motif_end,
                        ]
                    )
                    peak_to_motifs[peak_index].append((motif_index, item.best_p))
                motif_to_peaks[motif_index] = aggregate
    finally:
        connection.close()
    return {
        "worker_index": worker_index,
        "motif_to_peaks": motif_to_peaks,
        "peak_to_motifs": dict(peak_to_motifs),
        "instances_tmp": str(instances_tmp),
        "matrix_tmp": str(matrix_tmp),
    }


def _chunk_motif_indices(motif_count, workers):
    workers = max(1, workers)
    chunk_size = max(1, math.ceil(motif_count / workers))
    return [
        list(range(start, min(motif_count, start + chunk_size)))
        for start in range(0, motif_count, chunk_size)
    ]


def export_hits_and_sparse_matrix(
    sqlite_path: Path,
    motifs: Sequence[MotifProfile],
    peak_rows: Sequence[Mapping[str, object]],
    machine_dir: Path,
    *,
    file_prefix: Optional[str] = None,
    max_workers: int = 1,
) -> Tuple[
    Dict[int, Dict[int, "MotifPeakHit"]],
    Dict[int, List[Tuple[int, float]]],
    Path,
    Path,
]:
    """Write hit-level and peak x motif sparse tables and return aggregate maps.

    ``sqlite_path`` must point to a closed (fully committed) hit database --
    each worker opens its own short-lived read-only connection to it.
    """
    instances_path = result_path(machine_dir, "motif_instances.csv", file_prefix=file_prefix, modality=ATAC_MOTIF_MODALITY)
    matrix_path = result_path(machine_dir, "peak_motif_matrix.csv", file_prefix=file_prefix, modality=ATAC_MOTIF_MODALITY)
    instances_base = machine_dir / f".{instances_path.name}.part"
    matrix_base = machine_dir / f".{matrix_path.name}.part"

    global _EXPORT_WORKER_SQLITE_PATH, _EXPORT_WORKER_MOTIFS, _EXPORT_WORKER_PEAK_ROWS
    global _EXPORT_WORKER_INSTANCES_BASE, _EXPORT_WORKER_MATRIX_BASE
    _EXPORT_WORKER_SQLITE_PATH = sqlite_path
    _EXPORT_WORKER_MOTIFS = motifs
    _EXPORT_WORKER_PEAK_ROWS = peak_rows
    _EXPORT_WORKER_INSTANCES_BASE = instances_base
    _EXPORT_WORKER_MATRIX_BASE = matrix_base

    chunks = _chunk_motif_indices(len(motifs), max(1, int(max_workers)))
    worker_results = []
    try:
        if max_workers > 1 and len(chunks) > 1:
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(int(max_workers), len(chunks)), mp_context=ctx) as executor:
                futures = [executor.submit(_export_motif_chunk, chunk, index) for index, chunk in enumerate(chunks)]
                for future in as_completed(futures):
                    worker_results.append(future.result())
        else:
            worker_results = [_export_motif_chunk(list(range(len(motifs))), 0)]
    finally:
        _EXPORT_WORKER_SQLITE_PATH = None
        _EXPORT_WORKER_MOTIFS = None
        _EXPORT_WORKER_PEAK_ROWS = None
        _EXPORT_WORKER_INSTANCES_BASE = None
        _EXPORT_WORKER_MATRIX_BASE = None

    worker_results.sort(key=lambda result: result["worker_index"])
    motif_to_peaks = {}
    peak_to_motifs = defaultdict(list)
    with instances_path.open("w", newline="") as instance_out, matrix_path.open("w", newline="") as matrix_out:
        instance_writer = csv.writer(instance_out, lineterminator="\n")
        matrix_writer = csv.writer(matrix_out, lineterminator="\n")
        instance_writer.writerow(_EXPORT_INSTANCE_FIELDS)
        matrix_writer.writerow(_EXPORT_MATRIX_FIELDS)
        for result in worker_results:
            instances_tmp = Path(str(result["instances_tmp"]))
            matrix_tmp = Path(str(result["matrix_tmp"]))
            with instances_tmp.open() as source:
                shutil.copyfileobj(source, instance_out)
            with matrix_tmp.open() as source:
                shutil.copyfileobj(source, matrix_out)
            instances_tmp.unlink(missing_ok=True)
            matrix_tmp.unlink(missing_ok=True)
            motif_to_peaks.update(result["motif_to_peaks"])
            for peak_index, entries in result["peak_to_motifs"].items():
                peak_to_motifs[peak_index].extend(entries)
    return motif_to_peaks, peak_to_motifs, instances_path, matrix_path


def _load_sample_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(skip_comment_lines(handle)):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            rows.append(
                {**row, "sample_id": sample_id}
            )
    if not rows:
        raise ValueError(f"No samples were found in {path}")
    return rows


def _differential_arms(rows: Sequence[Mapping[str, object]], path: Union[str, Path]) -> Optional[Tuple[str, str]]:
    """(control, treatment) as the differential table itself records them, or None.

    1.0. Both differential writers (atac/statistics.py for bulk, the
    single-cell drivers through stats_core) put ``control_condition`` and
    ``treatment_condition`` on every row, with the arm LABEL for a labelled
    arm. A table that names two different comparisons is refused: it is not
    one differential.
    """
    seen: Dict[Tuple[str, str], int] = {}
    for row in rows:
        control = str(row.get("control_condition", "") or "").strip()
        treatment = str(row.get("treatment_condition", "") or "").strip()
        if control and treatment:
            seen[(control, treatment)] = seen.get((control, treatment), 0) + 1
    if len(seen) > 1:
        raise ParalomeError(
            220,
            f"{Path(path).name} records more than one comparison in its control_condition/"
            f"treatment_condition columns: " + "; ".join(f"{c} vs {t}" for c, t in sorted(seen)),
            fix="Pass one differential table per comparison",
        )
    return next(iter(seen)) if seen else None


def _resolve_comparison(
    recorded: Mapping[str, Optional[Tuple[str, str]]],
) -> Tuple[Tuple[str, str], str]:
    """The comparison an explicit table's results are named after, and where the name came from.

    Never guessed from manifest order. The motif stage takes no --control/--treatment -- the groups
    belong to the differential's --factor -- so an explicit table is named by
    what it records (``control_condition``/``treatment_condition``); a table
    that records nothing is refused, and --differential-checkpoint, which
    names every comparison, is the way in.
    """
    by_table = {labels for labels in recorded.values() if labels}
    if len(by_table) > 1:
        raise ParalomeError(
            220,
            "The differential tables of this run do not all record the same comparison: "
            + "; ".join(
                f"{Path(path).name}: {labels[0]} vs {labels[1]}" for path, labels in sorted(recorded.items()) if labels
            ),
            fix="Give the motif stage the tables of one comparison",
        )
    derived = next(iter(by_table)) if by_table else None
    if derived is not None:
        return derived, "the differential's control_condition/treatment_condition columns"
    tables = ", ".join(Path(path).name for path in recorded) or "the differential"
    raise ParalomeError(
        220,
        f"{tables} does not record which comparison it holds (no control_condition/"
        f"treatment_condition columns), so the comparison the results would be named after -- "
        f"and which way 'up' points -- cannot be known.",
        fix="Pass --differential-checkpoint (the differential stage's checkpoint names every comparison)",
    )


def _classify_peaks_for_enrichment(
    differential_rows: Sequence[Mapping[str, object]],
    peak_id_to_index: Mapping[str, int],
    lfc_column: str = "log2_fold_change",
    tested_column: str = "tested",
    alpha_metric: str = "p",
    alpha: float = 0.05,
) -> Tuple[Dict[str, List[int]], Dict[int, float]]:
    """Classify every peak_id in a --differential-statistics file into
    'up' / 'down' / 'nonsig' (untested peaks are dropped -- they're never a
    valid foreground OR background for enrichment), restricted to peak_ids
    that also exist in this motif run's own peak set.

    This is the classification half of the counting-based enrichment -- a pure
    counting/labeling step. The statistic itself is Fisher's exact, in
    _enrichment_statistics below.
    """
    bucket_peak_indices: Dict[str, List[int]] = {"up": [], "down": [], "nonsig": []}
    peak_lfc: Dict[int, float] = {}
    for row in differential_rows:
        peak_id = str(row.get("peak_id", ""))
        peak_index = peak_id_to_index.get(peak_id)
        if peak_index is None:
            continue
        tested = str(row.get(tested_column, "")).strip().lower() in ("true", "1", "yes")
        if not tested:
            continue
        lfc_raw = row.get(lfc_column, "")
        try:
            lfc = float(lfc_raw)
        except (TypeError, ValueError):
            lfc = None
        if lfc is not None and not math.isfinite(lfc):
            # An F-test of a term with several columns (a factor of
            # three or more levels) has no direction; NaN > 0 is False, so it
            # was read as 'down'.
            lfc = None
        # The peak sets follow --alpha (default p < 0.05) on the differential's own p or q.
        try:
            value = float(row.get("q_value" if alpha_metric == "q" else "p_value", ""))
        except (TypeError, ValueError):
            value = float("nan")
        selected = math.isfinite(value) and value < float(alpha)
        if not selected or lfc is None:
            bucket = "nonsig"
        else:
            bucket = "up" if lfc > 0 else "down"
        bucket_peak_indices[bucket].append(peak_index)
        if lfc is not None:
            peak_lfc[peak_index] = lfc
    return bucket_peak_indices, peak_lfc


def _manifest_sources(
    statistics: object, stratum: str, where: str,
) -> List[Dict[str, object]]:
    """One source per comparison of every statistic folder in a ``statistics`` map.

    ``statistics`` is the ``{folder: manifest}`` map paralome.results writes
    into a differential checkpoint (top level, or one stratum of a v2 index).
    Every comparison of every folder is a source: an ANOVA folder gives its
    six contrasts AND its three F-tests, each classified on its own machine
    table (an F-test's log2_fold_change column carries the direction its
    contrasts give it, so the same up/down/non-significant rule applies).
    """
    if not isinstance(statistics, Mapping):
        return []
    sources: List[Dict[str, object]] = []
    for folder, manifest in statistics.items():
        if not isinstance(manifest, Mapping):
            continue
        for name, entry in dict(manifest.get("comparisons") or {}).items():
            if not isinstance(entry, Mapping):
                continue
            table = entry.get("statistics")
            if not table:
                raise ParalomeError(
                    106,
                    f"{where}: the {folder} folder's comparison {name!r} records no machine "
                    f"statistics table, so its peaks cannot be classified.",
                    fix="Re-run the differential stage so its checkpoint records every comparison's machine table",
                )
            sources.append({
                "stratum": str(stratum or ""),
                "folder": str(folder),
                "comparison": str(name),
                "kind": str(entry.get("kind") or ""),
                "control": entry.get("control"),
                "treatment": entry.get("treatment"),
                "path": Path(str(table)),
            })
    return sources


def _discover_differential_checkpoint(cpm_matrix: Path) -> Optional[Path]:
    """The bulk differential's checkpoint beside the quantification, when one is there.

    What ``atac motifs`` classifies against when it is given neither
    --differential-statistics nor --differential-checkpoint: the checkpoint
    names every statistic folder's machine tables.
    """
    project_root = Path(cpm_matrix).resolve().parent.parent.parent
    candidate = project_root / "differential_accessibility" / "checkpoints" / "differential_accessibility.json"
    return candidate if candidate.is_file() else None


def _differential_sources(
    explicit: Optional[Union[str, Path]],
    checkpoint: Optional[Union[str, Path]],
    cpm_matrix: Path,
) -> List[Dict[str, object]]:
    """Everything the enrichment is classified against, one source per table.

    Each source is ``{stratum, folder, comparison, kind, control, treatment,
    path}``. The scan is peak-level and shared, but the CLASSIFICATION is not:
    every comparison of every statistic folder ("each
    statistical reality needs its own forked set of files") and, for a
    stratified single-cell run, every cell type has its own
    differential, so the caller runs the enrichment once per source.

    With ``--differential-checkpoint`` (or one discovered beside the
    quantification) the sources are read from the checkpoint's
    ``statistics`` manifests -- at its top level, or in each stratum of a v2
    index -- and every comparison is named as the differential named it.
    An explicit ``--differential-statistics`` table is one source with folder
    "" and no name yet: the caller names it from the table's own
    control_condition/treatment_condition columns (_resolve_comparison).
    """
    if explicit and checkpoint:
        raise ParalomeError(
            220,
            "--differential-statistics and --differential-checkpoint were both given. The "
            "checkpoint names every comparison's table, so one or the other says what the "
            "enrichment is classified against, never both.",
            fix="Pass --differential-checkpoint alone, or one --differential-statistics table",
        )
    if not explicit and not checkpoint:
        checkpoint = _discover_differential_checkpoint(Path(cpm_matrix))
        if checkpoint is None:
            raise FileNotFoundError(
                "Motif enrichment requires a --differential-checkpoint (the differential "
                "stage's checkpoint, naming every statistic folder's per-peak machine "
                "tables) or a --differential-statistics table -- neither was given and no "
                "differential_accessibility/checkpoints/differential_accessibility.json was "
                "found beside the quantification. The enrichment classifies every peak as "
                "up, down or neither (its --alpha cut) from the differential and cannot run without it."
            )
        print(f"[paralome] motif enrichment: classifying against {checkpoint} (found beside the quantification).",
              flush=True)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"--differential-checkpoint does not exist: {checkpoint_path}"
            )
        state = json.loads(checkpoint_path.read_text())
        strata = state.get("strata") if int(state.get("schema_version") or 1) >= 2 else None
        sources: List[Dict[str, object]] = []
        if isinstance(strata, Mapping) and strata:
            for label, entry in strata.items():
                statistics = (entry or {}).get("statistics") if isinstance(entry, Mapping) else None
                sources.extend(_manifest_sources(statistics, str(label), f"{checkpoint_path} stratum {label}"))
        else:
            sources.extend(_manifest_sources(state.get("statistics"), "", str(checkpoint_path)))
        if not sources:
            raise FileNotFoundError(
                f"{checkpoint_path} records no statistics manifests (no 'statistics' map with "
                f"per-comparison machine tables{' in any stratum' if strata else ''}). A differential "
                f"written by an older Paralome recorded one model's table instead -- re-run the "
                f"differential stage to produce the per-statistic folders the enrichment reads."
            )
        return sources

    resolved = _resolve_differential_statistics(explicit)
    if not resolved or not Path(resolved).exists():
        raise FileNotFoundError(f"--differential-statistics does not exist: {explicit}")
    return [{
        "stratum": "", "folder": "", "comparison": None, "kind": "",
        "control": None, "treatment": None, "path": Path(resolved),
    }]


def _source_key(source: Mapping[str, object]) -> str:
    """``<folder>/<comparison>`` plus ``:<stratum>``; the comparison alone for an explicit table."""
    folder = str(source.get("folder") or "")
    comparison = str(source.get("comparison") or "")
    base = f"{folder}/{comparison}" if folder else comparison
    stratum = str(source.get("stratum") or "")
    return f"{base}:{stratum}" if stratum else base


#: The columns _classify_peaks_for_enrichment reads. A table
#: without them classifies nothing.
_CLASSIFICATION_COLUMNS = ("peak_id", "tested", "log2_fold_change", "p_value", "q_value")


def _table_header(path: Path) -> List[str]:
    """The column names of a comma- or tab-delimited table, from its first line."""
    with Path(path).open(newline="") as handle:
        first = ""
        for line in handle:
            if line.strip():
                first = line.rstrip("\r\n")
                break
    delimiter = "\t" if first.count("\t") > first.count(",") else ","
    return [field.strip().strip('"') for field in next(csv.reader([first], delimiter=delimiter))] if first else []


def _require_classifiable_differential(
    explicit: Optional[Union[str, Path]],
    checkpoint: Optional[Union[str, Path]],
    cpm_matrix: Path,
) -> List[Dict[str, object]]:
    """Resolve the differential tables and refuse any the enrichment cannot classify.

    Deriving `significant` here would mean deciding what makes a
    peak differential. That is for the differential to decide and record, not
    for this consumer to guess, so a table without the classification columns
    is refused -- before any scanning -- naming what is missing.
    """
    sources = _differential_sources(explicit, checkpoint, cpm_matrix)
    problems: List[str] = []
    for source in sources:
        path = Path(source["path"])
        try:
            header = _table_header(path)
        except OSError as error:
            problems.append(f"{path}: cannot be read ({error})")
            continue
        missing = [column for column in _CLASSIFICATION_COLUMNS if column not in header]
        if not missing:
            continue
        where = f"{path}" + (f" ({_source_key(source)})" if source.get("folder") or source.get("stratum") else "")
        problems.append(
            f"{where} lacks the column(s) {', '.join(missing)}; it looks like a "
            f"reporting CSV rather than a differential's machine statistics table"
        )
    if problems:
        fix = (
            "Pass --differential-checkpoint naming the differential stage's checkpoint, or "
            "--differential-statistics naming one comparison's machine table "
            "(<statistic folder>/machine/<prefix>_<comparison>_statistics_<stamp>.csv)."
        )
        raise ParalomeError(
            106,
            "Motif enrichment classifies each peak as up, down or neither (its --alpha cut) from "
            "the differential's peak_id, tested, log2_fold_change, p_value and q_value "
            "columns, and refused before scanning because: "
            + "; ".join(problems) + ". " + fix,
            fix=fix,
        )
    return sources


def _fc_distribution(values: Sequence[float]) -> Dict[str, object]:
    """Descriptive-only log2FC summary (median preferred over mean: the
    mean is pulled by outlier peaks; the median just reports where the
    middle of the distribution sits). Never feeds an enrichment p-value."""
    if not values:
        return {"n": 0, "median": "", "p25": "", "p75": ""}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def _enrichment_statistics(
    feature_rows: Sequence[Mapping[str, object]],
    feature_keys: Sequence[object],
    feature_to_peaks: Mapping[object, Sequence[int]],
    total_peaks: int,
    bucket_peak_indices: Mapping[str, Sequence[int]],
    peak_lfc: Mapping[int, float],
) -> List[Dict[str, object]]:
    """Counting-based motif/pair enrichment. Replaces the CPM-based
    activity/t-test entirely (the t-test, Cohen's d and power are all gone
    from here). One row per feature (motif OR pair
    -- same function, same design: single and paired motifs are the same basic
    design), with up_* and down_* column groups side by side on one row.

    For each direction (up-vs-nonsig, down-vs-nonsig):
      - Fisher's exact test (two-sided), foreground = that direction's
        significant peaks, background = non-significant tested peaks.
      - local_rate / background_rate = the feature's hit-rate within each.
      - global_rate = the feature's hit-rate across the WHOLE peak
        universe scanned in this run (significant or not, tested or not):
        if a motif is globally enriched in a majority of peaks, then a
        random draw of peaks will also show an enrichment for that motif.
      - specificity_ratio = local_rate / global_rate -- an index on which
        a motif enriched in differential peaks AND rare everywhere
        else means more than one that's just common throughout.
      - FC context (purely descriptive, never feeds the p-value): median/
        p25/p75 of log2_fold_change among this direction's foreground
        peaks that actually have a hit, plus that bucket's own baseline
        median for comparison.
    BH q-values are corrected per direction, across all features of this
    type -- matching the validated sensitivity-analysis prototype.
    """
    up_ids = set(int(i) for i in bucket_peak_indices.get("up", ()))
    down_ids = set(int(i) for i in bucket_peak_indices.get("down", ()))
    nonsig_ids = set(int(i) for i in bucket_peak_indices.get("nonsig", ()))
    n_up, n_down, n_nonsig = len(up_ids), len(down_ids), len(nonsig_ids)

    rows: List[Dict[str, object]] = []
    raw_pvalues: Dict[str, List[float]] = {"up": [], "down": []}
    for key, metadata in zip(feature_keys, feature_rows):
        hit_ids = set(int(i) for i in feature_to_peaks.get(key, ()))
        global_hits = len(hit_ids)
        global_rate = global_hits / total_peaks if total_peaks > 0 else float("nan")
        row: Dict[str, object] = {
            **dict(metadata),
            "global_peaks": total_peaks,
            "global_hits": global_hits,
            "global_rate": global_rate if np.isfinite(global_rate) else "",
        }
        for direction, fg_ids, fg_n in (("up", up_ids, n_up), ("down", down_ids, n_down)):
            bg_ids, bg_n = nonsig_ids, n_nonsig
            fg_hits = len(hit_ids & fg_ids)
            bg_hits = len(hit_ids & bg_ids)
            if fg_n == 0 or bg_n == 0:
                fisher_p, odds_ratio, local_rate, bg_rate, specificity = (float("nan"),) * 5
            else:
                local_rate = fg_hits / fg_n
                bg_rate = bg_hits / bg_n
                odds_ratio, fisher_p = fisher_exact_2x2(
                    fg_hits, fg_n - fg_hits, bg_hits, bg_n - bg_hits, alternative="two-sided"
                )
                specificity = (local_rate / global_rate) if global_rate and global_rate > 0 else float("inf")
            raw_pvalues[direction].append(float(fisher_p))
            fc_hit_stats = _fc_distribution([peak_lfc[i] for i in (hit_ids & fg_ids) if i in peak_lfc])
            fc_bucket_stats = _fc_distribution([peak_lfc[i] for i in fg_ids if i in peak_lfc])
            row.update({
                f"{direction}_foreground_peaks": fg_n,
                f"{direction}_foreground_hits": fg_hits,
                f"{direction}_local_rate": local_rate if np.isfinite(local_rate) else "",
                f"{direction}_background_rate": bg_rate if np.isfinite(bg_rate) else "",
                f"{direction}_specificity_ratio": specificity if np.isfinite(specificity) else "",
                f"{direction}_fisher_odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else "",
                f"{direction}_fisher_p": float(fisher_p) if np.isfinite(fisher_p) else "",
                f"{direction}_fisher_q": "",  # filled in below, after BH correction across all features
                f"{direction}_fc_hit_n": fc_hit_stats["n"],
                f"{direction}_fc_median": fc_hit_stats["median"],
                f"{direction}_fc_p25": fc_hit_stats["p25"],
                f"{direction}_fc_p75": fc_hit_stats["p75"],
                f"{direction}_fc_bucket_median": fc_bucket_stats["median"],
            })
        rows.append(row)

    for direction in ("up", "down"):
        qvalues = benjamini_hochberg(raw_pvalues[direction])
        for row, qvalue in zip(rows, qvalues):
            row[f"{direction}_fisher_q"] = float(qvalue) if np.isfinite(qvalue) else ""
    return rows


# There is no 3+-condition motif test. The counting-based enrichment is a
# two-group design (foreground vs. non-significant background, derived from one
# pairwise --differential-statistics comparison), so no motif_global_anova.csv /
# motif_pair_global_anova.csv is produced.


def _pair_feature_stats_chunk(payloads, total_peaks, motifs_meta):
    rows = []
    for motif_a, motif_b, a_count, b_count, peak_features in payloads:
        both = len(peak_features)
        a_only = a_count - both
        b_only = b_count - both
        neither = total_peaks - both - a_only - b_only
        # A negative cell is not clamped to 0. Every count comes
        # from this run's scanned peaks, so one below zero means the hits and
        # the scanned set disagree (a hit database from another scan), and a
        # clamped table would test a contingency that never existed.
        if min(a_only, b_only, neither) < 0:
            raise ValueError(
                f"motif pair {motif_a}/{motif_b}: inconsistent co-occurrence counts "
                f"(both={both}, {motif_a} only={a_only}, {motif_b} only={b_only}, "
                f"neither={neither}, peaks={total_peaks}); the motif hits do not match "
                "the peaks this run scanned"
            )
        odds, pvalue = fisher_exact_2x2(both, a_only, b_only, neither, alternative="greater")
        distances = []
        orientations = Counter()
        for center_a, strand_a, center_b, strand_b in peak_features:
            distances.append(abs(center_b - center_a))
            orientations[f"{strand_a}{strand_b}"] += 1
        expected = a_count * b_count / max(1, total_peaks)
        enrichment = both / expected if expected > 0 else math.inf
        motif_a_id, motif_a_name = motifs_meta[motif_a]
        motif_b_id, motif_b_name = motifs_meta[motif_b]
        rows.append(
            {
                "pair_id": f"{motif_a_id}|{motif_b_id}",
                "motif_a_index": motif_a,
                "motif_b_index": motif_b,
                "motif_a_id": motif_a_id,
                "motif_a_name": motif_a_name,
                "motif_b_id": motif_b_id,
                "motif_b_name": motif_b_name,
                "cooccurring_peak_count": both,
                "motif_a_peak_count": a_count,
                "motif_b_peak_count": b_count,
                "expected_cooccurrence": expected,
                "enrichment_ratio": enrichment,
                "log2_enrichment": math.log2(enrichment) if enrichment > 0 and math.isfinite(enrichment) else "",
                "fisher_odds_ratio": float(odds),
                "enrichment_p": float(pvalue),
                "median_motif_distance": float(np.median(distances)) if distances else "",
                "dominant_orientation": orientations.most_common(1)[0][0] if orientations else "",
            }
        )
    return rows


def _build_pair_features(
    peak_to_motifs,
    motif_to_peaks,
    motifs,
    total_peaks,
    min_pair_peaks,
    max_motifs_per_peak,
    max_workers=1,
):
    # Pass 1: count-only. No per-pair peak lists yet.
    pair_counts = Counter()
    per_peak_motifs = []
    # The Fisher margins are counted over the SAME top-N universe as
    # the co-occurrence. Pairs are formed only from each peak's strongest
    # `max_motifs_per_peak` motifs, so `both` is "peaks where A and B are both
    # among the top N", and a motif counts as present in a peak exactly when it
    # is among that peak's selected N, for the margins and the overlap alike.
    # Counting the margins over every peak with ANY hit would compare a
    # restricted overlap with unrestricted totals and push every pair toward
    # "depleted" (routine with ~1,000 JASPAR motifs on 300-1,000 bp peaks).
    selected_motif_peak_counts: Counter = Counter()
    for peak_index, motif_values in peak_to_motifs.items():
        selected = sorted(motif_values, key=lambda item: item[1])[: max(2, max_motifs_per_peak)]
        motif_indices = sorted({item[0] for item in selected})
        for motif_index in motif_indices:
            selected_motif_peak_counts[motif_index] += 1
        if len(motif_indices) < 2:
            continue
        per_peak_motifs.append((int(peak_index), motif_indices))
        for pair in combinations(motif_indices, 2):
            pair_counts[pair] += 1

    surviving_pairs = {pair for pair, count in pair_counts.items() if count >= min_pair_peaks}
    del pair_counts

    # Pass 2: build full peak-index lists only for pairs that will survive.
    pair_peaks = {pair: [] for pair in surviving_pairs}
    for peak_index, motif_indices in per_peak_motifs:
        for pair in combinations(motif_indices, 2):
            bucket = pair_peaks.get(pair)
            if bucket is not None:
                bucket.append(peak_index)
    del per_peak_motifs
    pair_peaks = {pair: sorted(peaks) for pair, peaks in pair_peaks.items()}

    # Pass 3: Fisher's-exact + distance/orientation stats, over survivors only.
    #
    # extract only what each pair's stats need -- motif-level
    # peak counts (ints) and, per co-occurring peak, the hit's center
    # position + strand -- BEFORE any forking, so workers never see
    # motif_to_peaks itself (see _pair_feature_stats_chunk's docstring for
    # why that giant shared dict was blowing up per-worker memory).
    # Top-N margins from Pass 1, not len(motif_to_peaks[m]); see
    # the comment there.
    motif_peak_counts = dict(selected_motif_peak_counts)
    motifs_meta = {index: (motif.motif_id, motif.name) for index, motif in enumerate(motifs)}
    pair_payloads = []
    for (motif_a, motif_b), peaks in pair_peaks.items():
        peak_features = []
        for peak_index in peaks:
            hit_a = motif_to_peaks[motif_a][peak_index]
            hit_b = motif_to_peaks[motif_b][peak_index]
            center_a = (float(hit_a.best_motif_start) + float(hit_a.best_motif_end)) / 2
            center_b = (float(hit_b.best_motif_start) + float(hit_b.best_motif_end)) / 2
            peak_features.append((center_a, hit_a.best_strand, center_b, hit_b.best_strand))
        pair_payloads.append((motif_a, motif_b, motif_peak_counts[motif_a], motif_peak_counts[motif_b], peak_features))
    pair_payloads.sort(key=lambda item: (item[0], item[1]))

    workers = max(1, int(max_workers))
    if workers > 1 and len(pair_payloads) > 1:
        chunk_size = max(1, math.ceil(len(pair_payloads) / workers))
        chunks = [pair_payloads[i : i + chunk_size] for i in range(0, len(pair_payloads), chunk_size)]
        rows = []
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks)), mp_context=ctx) as executor:
            for chunk_rows in executor.map(
                _pair_feature_stats_chunk, chunks,
                repeat(total_peaks), repeat(motifs_meta),
            ):
                rows.extend(chunk_rows)
    else:
        rows = _pair_feature_stats_chunk(pair_payloads, total_peaks, motifs_meta)

    rows.sort(key=lambda row: (row["motif_a_index"], row["motif_b_index"]))
    pvalues = [row["enrichment_p"] for row in rows]
    qvalues = benjamini_hochberg(pvalues)
    for row, qvalue in zip(rows, qvalues):
        row["enrichment_q"] = float(qvalue)
    return rows, pair_peaks


def _resolve_differential_statistics(explicit: Optional[Union[str, Path]]) -> Optional[Path]:
    """The explicit --differential-statistics table, or None when there is no such file.

    `explicit` may be a not-yet-prefix-aware name a caller built (e.g.
    "<dir>/machine/<name>.csv"); it is resolved against its own directory
    before giving up.
    """
    if not explicit:
        return None
    path = Path(explicit)
    if path.exists():
        return path
    return resolve_prefixed_file(path.parent, path.name)


def _resolve_coverage_summary(cpm_matrix: Path) -> Optional[Path]:
    search_dirs = [cpm_matrix.parent, cpm_matrix.parent / "machine"]
    for bare_name in ("peak_counting_summary.csv", "peak_counting_summary.tsv"):
        # Pre-Patch-109 runs may still have the older tab-delimited name on disk.
        for search_dir in search_dirs:
            found = resolve_prefixed_file(search_dir, bare_name)
            if found is not None:
                return found
    return None


def run_motif_parallel_analysis(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    genome: Optional[str] = None,
    reference_fasta: Optional[Union[str, Path]] = None,
    motif_database: Optional[Union[str, Path]] = None,
    motif_species: Optional[str] = None,
    motif_library: str = "jaspar",
    motif_p_threshold: float = 1e-4,
    jaspar_release: str = DEFAULT_JASPAR_RELEASE,
    download_motifs: bool = True,
    differential_statistics: Optional[Union[str, Path]] = None,
    # The v2 differential checkpoint, whose `strata` map lets the
    # enrichment fan out per cell type. Takes precedence over the single-file
    # --differential-statistics, which stays for unstratified runs.
    differential_checkpoint: Optional[Union[str, Path]] = None,
    pseudocount: float = 1.0,
    min_motif_peaks: int = 5,
    min_pair_peaks: int = 5,
    max_motifs_per_peak: int = 25,
    scanner_backend: str = "auto",
    motif_batch_size: int = 64,
    sequence_batch_bases: int = 2_000_000,
    alpha: float = 0.05,
    alpha_metric: str = "p",
    motifs_per_peak: int = 5,
    skip_motif_pairs: bool = False,
    threads: int = 1,
    jobs: int = 0,
    dry_run: bool = False,
    stage: str = "all",
    max_peak_width: int = DEFAULT_MAX_PEAK_WIDTH,
) -> Dict[str, object]:
    """Run motif preparation and shard scans locally using bounded processes."""
    print(
        stage_banner(
            "motif_analysis", derive_file_prefix(Path(output_dir)), output_dir,
        ),
        flush=True,
    )
    # Fail on an unclassifiable differential before preparing or
    # scanning anything; see _require_classifiable_differential.
    _require_classifiable_differential(
        differential_statistics, differential_checkpoint, Path(cpm_matrix),
    )
    resume_problem = None
    if stage == "pairs":
        resume_problem = _singles_state_reuse_problem(
            Path(output_dir) / "inputs" / "motif_singles_state.pkl",
            _singles_state_identity(
                cpm_matrix=cpm_matrix, sample_manifest=sample_manifest, genome=genome,
                reference_fasta=reference_fasta, motif_database=motif_database,
                motif_species=motif_species, motif_library=motif_library,
                jaspar_release=jaspar_release, motif_p_threshold=motif_p_threshold,
                max_peak_width=max_peak_width, scanner_backend=scanner_backend,
            ),
        )
    if stage == "pairs" and resume_problem is None:
        # Resuming pairs from a prior stage="singles" checkpoint needs
        # none of the prepare/scan/merge machinery below (that produced the
        # hit database the singles run already consumed and deleted) --
        # everything it needs comes back out of the pickle written by that
        # earlier run.
        #
        # Only when that hand-off matches these inputs. Otherwise
        # the sharded scan below runs and run_motif_analysis(stage="pairs")
        # builds pairs from it; before, a missing hand-off sent pairs into
        # run_motif_analysis's single-process scan, and a stale one was
        # resumed as if current.
        return run_motif_analysis(
            cpm_matrix, sample_manifest, output_dir,
            genome=genome, reference_fasta=reference_fasta, motif_database=motif_database,
            motif_species=motif_species, motif_library=motif_library, motif_p_threshold=motif_p_threshold,
            jaspar_release=jaspar_release, download_motifs=download_motifs,
            differential_statistics=differential_statistics, pseudocount=pseudocount,
            differential_checkpoint=differential_checkpoint,
            min_motif_peaks=min_motif_peaks, min_pair_peaks=min_pair_peaks,
            max_motifs_per_peak=max_motifs_per_peak, scanner_backend=scanner_backend,
            motif_batch_size=motif_batch_size, sequence_batch_bases=sequence_batch_bases,
            alpha=alpha, alpha_metric=alpha_metric, 
            motifs_per_peak=motifs_per_peak, skip_motif_pairs=skip_motif_pairs, threads=threads,
            jobs=jobs,
            stage="pairs",
            max_peak_width=max_peak_width,
        )
    plan_info = prepare_motif_local_plan(
        cpm_matrix=Path(cpm_matrix),
        sample_manifest=Path(sample_manifest),
        output_dir=Path(output_dir),
        genome=genome,
        reference_fasta=Path(reference_fasta) if reference_fasta else None,
        motif_database=Path(motif_database) if motif_database else None,
        motif_species=motif_species,
        motif_library=motif_library,
        motif_p_threshold=motif_p_threshold,
        jaspar_release=jaspar_release,
        download_motifs=download_motifs,
        scanner_backend=scanner_backend,
        motif_batch_size=motif_batch_size,
        sequence_batch_bases=sequence_batch_bases,
        threads=threads,
        jobs=jobs,
        max_peak_width=max_peak_width,
    )
    plan_path = Path(str(plan_info['plan']))
    if dry_run:
        return {**plan_info, 'complete': False, 'dry_run': True}

    # The old record is withdrawn before the scan starts, not only in the
    # finalizer; see withdraw_motif_checkpoint.
    withdraw_motif_checkpoint(output_dir, reason=f"a --stage {stage} run started (local backend)")
    preparation = run_motif_prepare_task(plan_path)
    plan = json.loads(plan_path.read_text())
    pending = [
        int(task['task_id'])
        for task in plan['tasks']
        if _validate_scan_checkpoint(plan, task) is None
    ]
    completed: List[Dict[str, object]] = []
    if pending:
        max_workers = min(int(plan['resource_plan']['concurrent_jobs']), len(pending))
        if max_workers <= 1:
            completed = [run_motif_scan_task(plan_path, task_id) for task_id in pending]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(run_motif_scan_task, plan_path, task_id): task_id
                    for task_id in pending
                }
                for future in as_completed(future_to_task):
                    completed.append(future.result())
    merged = merge_motif_scan_tasks(plan_path)
    result = run_motif_analysis(
        cpm_matrix,
        sample_manifest,
        output_dir,
        genome=genome,
        reference_fasta=reference_fasta,
        motif_database=motif_database,
        motif_species=motif_species,
        motif_library=motif_library,
        motif_p_threshold=motif_p_threshold,
        jaspar_release=jaspar_release,
        download_motifs=download_motifs,
        differential_statistics=differential_statistics,
        differential_checkpoint=differential_checkpoint,
        pseudocount=pseudocount,
        min_motif_peaks=min_motif_peaks,
        min_pair_peaks=min_pair_peaks,
        max_motifs_per_peak=max_motifs_per_peak,
        scanner_backend=scanner_backend,
        motif_batch_size=motif_batch_size,
        sequence_batch_bases=sequence_batch_bases,
        alpha=alpha,
        alpha_metric=alpha_metric,
        motifs_per_peak=motifs_per_peak,
        skip_motif_pairs=skip_motif_pairs,
        threads=threads,
        jobs=jobs,
        precomputed_hit_database=merged,
        precomputed_plan=plan_path,
        stage=stage,
        max_peak_width=max_peak_width,
    )
    result.update({
        'backend': 'local',
        'plan': str(plan_path),
        'plan_id': plan['plan_id'],
        'task_count': len(plan['tasks']),
        'completed_tasks': len(plan['tasks']),
        'newly_completed_tasks': len(completed),
        'resource_plan': plan['resource_plan'],
        'prepare_checkpoint': preparation,
    })
    return result

def run_motif_analysis(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    genome: Optional[str] = None,
    reference_fasta: Optional[Union[str, Path]] = None,
    motif_database: Optional[Union[str, Path]] = None,
    motif_species: Optional[str] = None,
    motif_library: str = "jaspar",
    motif_p_threshold: float = 1e-4,
    jaspar_release: str = DEFAULT_JASPAR_RELEASE,
    download_motifs: bool = True,
    differential_statistics: Optional[Union[str, Path]] = None,
    # The v2 differential checkpoint, whose `strata` map lets the
    # enrichment fan out per cell type. Takes precedence over the single-file
    # --differential-statistics, which stays for unstratified runs.
    differential_checkpoint: Optional[Union[str, Path]] = None,
    pseudocount: float = 1.0,
    min_motif_peaks: int = 5,
    min_pair_peaks: int = 5,
    max_motifs_per_peak: int = 25,
    scanner_backend: str = "auto",
    motif_batch_size: int = 64,
    sequence_batch_bases: int = 2_000_000,
    alpha: float = 0.05,
    alpha_metric: str = "p",
    motifs_per_peak: int = 5,
    skip_motif_pairs: bool = False,
    threads: int = 1,
    jobs: int = 0,
    precomputed_hit_database: Optional[Union[str, Path]] = None,
    precomputed_plan: Optional[Union[str, Path]] = None,
    stage: str = "all",
    max_peak_width: int = DEFAULT_MAX_PEAK_WIDTH,
) -> Dict[str, object]:
    """Scan known motifs and test motif-associated accessibility and co-occurrence.

    The activity and pair analyses describe sequence motif potential beneath ATAC
    peaks. They do not claim direct TF occupancy or physical co-binding.

    motif-pair co-occurrence (``_build_pair_features``) is a
    combinatorial loop -- one Fisher's-exact test plus a per-co-occurring-peak
    distance/orientation scan for every candidate motif pair -- and is by far
    the slowest step of motif finalize. ``skip_motif_pairs=True`` skips that
    call outright; every downstream consumer of the pair tables (per-comparison
    stats/cut files, the pair network plot,
    motif_pairs_tested in the scan summary/checkpoint) tolerates an empty pair
    set the same way it does when a real run has zero pairs passing
    --min-pair-peaks.

    The single-motif and the motif-pair halves are separate,
    checkpointable steps, so pairs (hours) can be inspected and re-run without
    the scan and export (minutes). ``stage`` controls that hand-off:

    - ``"all"`` (default): scan/export, singles, and (unless
      ``skip_motif_pairs``) pairs all run in one call. No interim checkpoint
      is written between singles and pairs: the record is withdrawn when the
      run starts (withdraw_motif_checkpoint) and written once, at the end.
    - ``"singles"``: runs scan/export + all single-motif outputs, then --
      instead of building pairs -- pickles everything the pairs stage needs
      (the exported hit maps, thresholds, background, resolved resources,
      peak/sample rows) to ``inputs/motif_singles_state.pkl`` with an identity
      sidecar, and writes the checkpoint with ``"complete": False`` and
      ``"singles_complete": True``. No pair-related outputs are produced.
    - ``"pairs"``: resumes from the hand-off a prior ``stage="singles"`` run
      left, skipping scan and export, when that hand-off was computed from the
      same inputs (_singles_state_identity). Otherwise the hand-off
      is discarded, the reason is printed, and the scan runs again in this
      process -- the sharded runners (run_motif_parallel_analysis, the SLURM
      plan) make the same decision and run their sharded scan instead. Then
      pairs are built, the per-comparison reporting is written (which also
      re-derives the cheap single-motif tables), and the final checkpoint says
      ``"complete": True``.

    The enrichment universe is the scanned peaks (find_unscanned_peaks), the
    comparison is the one the differential tested (_resolve_comparison), the
    old checkpoint is withdrawn before anything is written, and result files an
    earlier run left that this run did not produce are removed
    (_remove_results_not_written_by_this_run).
    """
    alpha_metric = alpha_metric.strip().lower()
    if alpha_metric not in {"p", "q"}:
        raise ValueError("alpha_metric must be 'p' or 'q'")
    stage = str(stage).strip().lower()
    if stage not in {"all", "singles", "pairs"}:
        raise ValueError(f"stage must be 'all', 'singles', or 'pairs', got {stage!r}")
    # The spawned worker processes are constrained to the run's resources. --threads sizes the
    # SLURM finalize job's own CPU request (finalize_cpus = threads); when --jobs is narrower than
    # --threads the worker pools (motif-hit export, motif-pair feature building) are capped at
    # --jobs too, so the finalize step never spawns more concurrent processes than the width the
    # run was sized for.
    resource_max_workers = min(int(threads), int(jobs)) if int(jobs) > 0 else int(threads)
    out = Path(output_dir)
    results = out / "results"
    machine = results  # Flat, like the differential -- see naming.py
    inputs = out / "inputs"
    checkpoints = out / "checkpoints"
    for directory in (results, machine, inputs, checkpoints):
        directory.mkdir(parents=True, exist_ok=True)
    # Every --out directory is already a unique "project ID" per run by
    # convention; derive each output file's prefix from that directory's own
    # basename rather than a separate option. Checkpoints are NOT prefixed.
    file_prefix = derive_file_prefix(out)
    print(stage_banner("motif_analysis", file_prefix, out), flush=True)

    stage_started = time.monotonic()
    singles_state_path = inputs / "motif_singles_state.pkl"

    # Refuse a differential table the enrichment cannot classify
    # BEFORE the scan, not after it. See _require_classifiable_differential.
    _require_classifiable_differential(
        differential_statistics, differential_checkpoint, Path(cpm_matrix),
    )

    # On the sharded paths the width the scan used is in the plan; the
    # finalizer goes by that, not by its own argument, so the peaks it leaves
    # out of the universe are exactly the ones the shards left unscanned
    #. A disagreement is said, not resolved quietly.
    if precomputed_plan is not None:
        planned_width = json.loads(Path(precomputed_plan).read_text()).get("max_peak_width")
        if planned_width is not None and int(planned_width) != int(max_peak_width):
            print(
                f"[paralome] motif finalizer: the scan plan used --max-peak-width {int(planned_width)}, "
                f"not {int(max_peak_width)}; the finalizer goes by the plan, which is what was scanned.",
                flush=True,
            )
            max_peak_width = int(planned_width)

    # A singles hand-off is resumed only when it was computed from
    # the inputs this call names (see _singles_state_identity), and any other
    # run discards the old one before it starts: an 'all' or 'singles' run
    # supersedes it, and leaving it would let a later --stage pairs resume
    # from a scan that no longer describes these peaks.
    singles_identity = _singles_state_identity(
        cpm_matrix=cpm_matrix, sample_manifest=sample_manifest, genome=genome,
        reference_fasta=reference_fasta, motif_database=motif_database,
        motif_species=motif_species, motif_library=motif_library,
        jaspar_release=jaspar_release, motif_p_threshold=motif_p_threshold,
        max_peak_width=max_peak_width, scanner_backend=scanner_backend,
    )
    singles_state: Optional[Mapping[str, object]] = None
    if stage == "pairs":
        resume_problem = _singles_state_reuse_problem(singles_state_path, singles_identity)
        if resume_problem is None:
            print(f"Motif finalizer phase: loading singles checkpoint from {singles_state_path}", flush=True)
            try:
                with singles_state_path.open("rb") as handle:
                    singles_state = pickle.load(handle)
            except Exception as error:  # truncated or unreadable: rescan, say why
                resume_problem = f"{singles_state_path.name} could not be read ({error})"
                singles_state = None
            if singles_state is not None and singles_state.get("identity") != singles_identity:
                resume_problem = f"{singles_state_path.name} does not match its identity record"
                singles_state = None
        if resume_problem is not None:
            print(
                f"Motif finalizer phase: not resuming --stage pairs from a singles hand-off: "
                f"{resume_problem}. Scanning again.",
                flush=True,
            )
            _discard_singles_state(singles_state_path)
    else:
        _discard_singles_state(singles_state_path)

    # Nothing below is written over the earlier run's results while its
    # record still says complete; see withdraw_motif_checkpoint. A pairs run
    # resuming from a verified hand-off leaves the single-motif outputs alone,
    # so that half of the record stands.
    withdraw_motif_checkpoint(
        out, reason=f"a --stage {stage} run started in the finalizer",
        keep_singles=singles_state is not None,
    )
    # Every result file this run writes, so the ones it did not can be told
    # apart at the end (_remove_results_not_written_by_this_run).
    written_results: List[Path] = []

    if singles_state is not None:
        # Resume from a prior stage="singles" run: skip resource loading,
        # scanning, and hit export entirely (the cheap part) and load its
        # outputs straight from the pickle instead.
        resolved = singles_state["resolved"]
        motifs = singles_state["motifs"]
        sample_rows = singles_state["sample_rows"]
        sample_ids = singles_state["sample_ids"]
        peak_rows = singles_state["peak_rows"]
        background = singles_state["background"]
        thresholds = singles_state["thresholds"]
        motif_to_peaks = singles_state["motif_to_peaks"]
        peak_to_motifs = singles_state["peak_to_motifs"]
        instances_path = singles_state["instances_path"]
        matrix_path = singles_state["matrix_path"]
        written_results += [Path(instances_path), Path(matrix_path)]
        # The same rule the scan applied, from the same peaks and FASTA
        # (the identity check above vouches for both), so the universe below
        # is the scanned peaks here too.
        with IndexedFasta(Path(resolved["reference_fasta"])) as fasta:
            unscanned = find_unscanned_peaks(peak_rows, max_peak_width, fasta)
    else:
        print("Motif finalizer phase: loading resources and matrices", flush=True)
        resolved = resolve_motif_resources(
            genome=genome,
            reference_fasta=reference_fasta,
            motif_database=motif_database,
            motif_species=motif_species,
            motif_library=motif_library,
            jaspar_release=jaspar_release,
            download_motifs=download_motifs,
        )
        motifs = load_motifs(Path(resolved["motif_database"]))
        sample_rows = _load_sample_manifest(Path(sample_manifest))
        sample_ids = [row["sample_id"] for row in sample_rows]
        # the CPM matrix is comma-delimited going forward; auto-sniff
        # (python engine) so a pre-Patch-109 tab-delimited matrix still reads.
        frame = pd.read_csv(cpm_matrix, sep=None, engine="python")
        missing_samples = [sample_id for sample_id in sample_ids if sample_id not in frame.columns]
        if missing_samples:
            raise ValueError("CPM matrix is missing samples: " + ", ".join(missing_samples))
        required_columns = {"peak_id", "chrom", "start", "end"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError("CPM matrix is missing peak metadata columns: " + ", ".join(sorted(missing)))
        peak_rows = frame.drop(columns=sample_ids).fillna("").to_dict(orient="records")

        # Peak_rows is NOT filtered, and must never be.
        #
        # `peak_index` means "row in the CPM matrix" everywhere in the motif
        # stage. The sharded path writes it explicitly into each shard FASTA
        # header (`>{peak_idx}` in prepare_motif_shards) and the finalizer
        # indexes peak_rows with whatever comes back out of the merged sqlite,
        # so removing rows here would shift every index after them.
        #
        # Skipping is expressed as an EMPTY SEQUENCE instead, which is the
        # mechanism scan_peak_sequences already uses:
        #     peak_indices = [i for i, sequence in enumerate(sequences) if sequence]
        # A skipped peak therefore keeps its index and simply produces no hits,
        # on both the in-process and the sharded path. The same goes for a peak
        # on a contig the FASTA does not have (find_unscanned_peaks); both kinds
        # are kept OUT of the enrichment universe below.
        with IndexedFasta(Path(resolved["reference_fasta"])) as fasta:
            unscanned = find_unscanned_peaks(peak_rows, max_peak_width, fasta)
        refuse_if_contigs_do_not_match(unscanned, len(peak_rows), resolved["reference_fasta"])
        report_unscanned_peaks(unscanned, len(peak_rows), max_peak_width, resolved["reference_fasta"])
        # The CPM matrix is loaded for its peak metadata (peak_id/chrom/start/end above); no CPM
        # activity matrix is computed -- the counting-based enrichment test below never touches CPM
        # values. `pseudocount` is accepted for CLI compatibility and unused.

        sqlite_path = inputs / "motif_hits.sqlite"
        if precomputed_hit_database is None:
            sequences: List[str] = []
            with IndexedFasta(Path(resolved["reference_fasta"])) as fasta:
                for index, peak in enumerate(peak_rows):
                    if index in unscanned.indices:
                        # Empty -> excluded by scan_peak_sequences' own filter,
                        # while the index stays aligned with peak_rows.
                        sequences.append("")
                        continue
                    sequences.append(
                        fasta.fetch(str(peak["chrom"]), int(float(peak["start"])), int(float(peak["end"])))
                    )
            background = _background_from_sequences(sequences)
            connection, thresholds, distributions = scan_peak_sequences(
                motifs, sequences, peak_rows, background, motif_p_threshold, sqlite_path,
                motif_batch_size=motif_batch_size, sequence_batch_bases=sequence_batch_bases,
                scanner_backend=scanner_backend,
            )
        else:
            source_database = Path(precomputed_hit_database)
            if not source_database.exists():
                raise FileNotFoundError(f"Precomputed motif hit database does not exist: {source_database}")
            if precomputed_plan is None:
                raise ValueError("precomputed_plan is required with precomputed_hit_database")
            prepared_plan = json.loads(Path(precomputed_plan).read_text())
            background = np.asarray(prepared_plan["background"], dtype=float)
            shutil.copy2(source_database, sqlite_path)
            connection = sqlite3.connect(sqlite_path)
            thresholds, distributions = _compute_thresholds_distributions(motifs, background, motif_p_threshold)
        print(f"Motif finalizer phase: exporting motif hits ({time.monotonic() - stage_started:.1f}s)", flush=True)
        connection.close()
        motif_to_peaks, peak_to_motifs, instances_path, matrix_path = export_hits_and_sparse_matrix(
            sqlite_path, motifs, peak_rows, machine, file_prefix=file_prefix, max_workers=resource_max_workers,
        )
        sqlite_path.unlink(missing_ok=True)
        written_results += [Path(instances_path), Path(matrix_path)]

    # The universe every rate and every Fisher table is counted over is
    # the peaks that were scanned. See UnscannedPeaks.
    scanned_indices = [index for index in range(len(peak_rows)) if index not in unscanned.indices]
    total_peaks = len(scanned_indices)
    if total_peaks <= 0:
        # The same refusal _inspect_peak_rows gives the sharded planner; a
        # contig mismatch of this size was refused above (PLM-0219).
        raise ParalomeError(
            218,
            f"No peak was scanned: all {len(peak_rows)} were skipped ({len(unscanned.oversize)} wider than "
            f"--max-peak-width {max_peak_width} bp, {len(unscanned.unresolved)} on contigs the reference lacks), "
            f"so there is nothing to test for enrichment.",
            fix="raise --max-peak-width above the widest peak",
        )

    metadata_rows: List[Dict[str, object]] = []
    motif_feature_to_peaks: Dict[int, List[int]] = {}
    for motif_index, profile in enumerate(motifs):
        peaks = sorted(motif_to_peaks.get(motif_index, {}))
        if len(peaks) < min_motif_peaks:
            continue
        motif_feature_to_peaks[motif_index] = peaks
        scores = [float(motif_to_peaks[motif_index][peak].best_score) for peak in peaks]
        metadata_rows.append(
            {
                "motif_index": motif_index,
                "motif_id": profile.motif_id,
                "motif_name": profile.name,
                "motif_width": profile.width,
                "motif_peak_count": len(peaks),
                "motif_instance_count": int(sum(int(motif_to_peaks[motif_index][peak].hit_count) for peak in peaks)),
                "median_best_score": float(np.median(scores)),
                "score_threshold": float(thresholds[motif_index]),
            }
        )
    motif_keys = list(motif_feature_to_peaks.keys())
    metadata_by_index = {int(row["motif_index"]): row for row in metadata_rows}
    motif_rows = [metadata_by_index[int(key)] for key in motif_keys]

    if stage == "singles":
        # Checkpointable hand-off point: finish the single motifs and stop,
        # so a later run can start at the pairs analysis. Persist everything
        # a later stage="pairs" run
        # needs so it can skip scanning/export (the cheap part) and go
        # straight to pair building (the slow part) without redoing any of
        # the singles work above.
        print(f"Motif finalizer phase: writing singles checkpoint for later --stage pairs ({time.monotonic() - stage_started:.1f}s)", flush=True)
        # Written atomically with the identity it was computed
        # from; see _write_singles_state and _singles_state_identity.
        _write_singles_state(
            singles_state_path,
            {
                "resolved": resolved,
                "motifs": motifs,
                "sample_rows": sample_rows,
                "sample_ids": sample_ids,
                "peak_rows": peak_rows,
                "background": background,
                "thresholds": thresholds,
                "motif_to_peaks": motif_to_peaks,
                "peak_to_motifs": peak_to_motifs,
                "instances_path": instances_path,
                "matrix_path": matrix_path,
            },
            singles_identity,
        )
        effective_skip_pairs = True
    else:
        effective_skip_pairs = skip_motif_pairs

    if effective_skip_pairs:
        reason = "--stage singles was set" if stage == "singles" else "--skip-motif-pairs was set"
        print(f"Motif finalizer phase: skipping motif-pair features, {reason} ({time.monotonic() - stage_started:.1f}s)", flush=True)
        pair_metadata, pair_to_peaks = [], {}
    else:
        print(f"Motif finalizer phase: building motif-pair features ({time.monotonic() - stage_started:.1f}s)", flush=True)
        pair_metadata, pair_to_peaks = _build_pair_features(
            peak_to_motifs,
            motif_to_peaks,
            motifs,
            total_peaks,  # the scanned peaks, as for the single motifs
            min_pair_peaks,
            max_motifs_per_peak,
            max_workers=resource_max_workers,
        )
    pair_keys = list(pair_to_peaks.keys())
    pair_metadata_by_key = {
        (int(row["motif_a_index"]), int(row["motif_b_index"])): row
        for row in pair_metadata
    }
    pair_rows_ordered = [pair_metadata_by_key[tuple(key)] for key in pair_keys]


    # The counting-based enrichment test needs the peak-level up/down/non-significant
    # classification from the peaks differential-accessibility test. Resolved once, before the
    # comparison loop, as candidate-view rendering does.
    print(f"Motif finalizer phase: resolving differential-accessibility statistics for enrichment ({time.monotonic() - stage_started:.1f}s)", flush=True)
    # The SAME sanitizer strata.stratum_slugs uses, so a motif result's prefix
    # matches its differential's byte for byte. Imported here rather than at
    # module scope because atac.singlecell imports this module back, lazily.
    from paralome.atac.singlecell import sanitize_sample_id

    # One source per comparison of every statistic folder (and per
    # stratum), not one table. The scan above is shared -- peaks do not change
    # between statistics or cell types -- but the classification is not.
    differential_sources = _differential_sources(
        differential_statistics, differential_checkpoint, Path(cpm_matrix),
    )
    # Only scanned peaks can be classified into the universe; a skipped
    # peak's row in the differential is left out, as the peak was.
    peak_id_to_index = {
        str(peak_rows[index].get("peak_id", "")): index for index in scanned_indices
    }

    enrichment_sources: List[Dict[str, object]] = []
    recorded_arms: Dict[str, Optional[Tuple[str, str]]] = {}
    skipped_sources: List[Dict[str, object]] = []
    for source in differential_sources:
        statistics_path = Path(source["path"])
        # Comma-delimited going forward; auto-sniffed so a tab-delimited
        # explicit table still reads.
        rows = pd.read_csv(
            statistics_path, sep=None, engine="python",
        ).fillna("").to_dict(orient="records")
        if source["comparison"] is None:
            # An explicit table: what it says it compared, checked below.
            recorded_arms[str(statistics_path)] = _differential_arms(rows, statistics_path)
        buckets, lfc = _classify_peaks_for_enrichment(rows, peak_id_to_index, alpha_metric=alpha_metric, alpha=alpha)
        classified = sum(len(indices) for indices in buckets.values())
        label = _source_key(source) or str(source.get("stratum") or "") or "(all)"
        print(
            f"Enrichment peak universe [{label}]: {total_peaks} peaks scanned; "
            f"{len(buckets['up'])} up, {len(buckets['down'])} down, "
            f"{len(buckets['nonsig'])} neither, at the --alpha cut (per {statistics_path.name}).",
            flush=True,
        )
        if not classified:
            # Named, and NOT fatal: one comparison or one thin stratum
            # classifying nothing must not discard the others, which is the
            # whole reason the fan-out exists.
            print(
                f"  SKIPPED {label}: no peak in {statistics_path.name} could be "
                f"classified -- no peak of this run's peak set was tested in it.",
                flush=True,
            )
            skipped_sources.append(source)
            continue
        if not lfc:
            # No tested peak carries a direction -- the F-test of a
            # factor of three or more levels. Its levels' contrasts are
            # classified each on their own.
            print(
                f"  SKIPPED {label}: {statistics_path.name} has no direction (an F-test of "
                "several levels); each level's contrast is enriched on its own.",
                flush=True,
            )
            skipped_sources.append(source)
            continue
        enrichment_sources.append({**source, "rows": rows, "buckets": buckets, "lfc": lfc})

    if not enrichment_sources:
        raise RuntimeError(
            f"No differential table produced a classifiable peak set from "
            f"{len(differential_sources)} table(s); there is nothing to "
            f"test for enrichment. See the per-table lines above for which files "
            f"were read."
        )

    # An explicit table is named as the differential tested it; see
    # _resolve_comparison. A checkpoint's comparisons are named by its
    # manifests, which say what each one compared.
    if any(source["comparison"] is None for source in differential_sources):
        (named_control, named_treatment), named_from = _resolve_comparison(recorded_arms)
        for source in (*differential_sources, *enrichment_sources):
            if source["comparison"] is None:
                source.update(comparison=f"{named_control}_vs_{named_treatment}",
                              control=named_control, treatment=named_treatment)
    else:
        named_from = "the differential checkpoint's statistics manifests"
    print(
        f"Motif finalizer phase: {len(enrichment_sources)} enrichment(s), named for "
        + ", ".join(_source_key(source) for source in enrichment_sources)
        + f" (from {named_from}).",
        flush=True,
    )

    comparison_results: Dict[str, Dict[str, object]] = {}
    for source in enrichment_sources:
        folder = str(source["folder"] or "")
        comparison_name = str(source["comparison"])
        stratum_label = str(source["stratum"] or "")
        result_key = _source_key(source)
        # Each statistic folder's results go in their own folder,
        # results/<folder>/, machine tables in results/<folder>/machine/, the
        # way the differential lays out its own; an explicit table (folder "")
        # writes under results/ as before.
        source_dir = results / folder if folder else results
        source_machine = source_dir / "machine" if folder else machine
        source_machine.mkdir(parents=True, exist_ok=True)
        # The stratum and the comparison ride in the FILE PREFIX --
        # <prefix>_<stratum>_<comparison> -- so a motif result sorts beside the
        # differential it was classified on. An explicit table keeps the
        # names it always had: <prefix>[_<stratum>].
        prefix_parts = [file_prefix, sanitize_sample_id(stratum_label) if stratum_label else ""]
        if folder:
            prefix_parts.append(comparison_name)
        source_prefix = "_".join(part for part in prefix_parts if part)
        bucket_peak_indices = source["buckets"]
        peak_lfc = source["lfc"]
        print(
            f"Motif finalizer phase: enrichment for {result_key} "
            f"({time.monotonic() - stage_started:.1f}s)",
            flush=True,
        )
        motif_stats = _enrichment_statistics(
            motif_rows, motif_keys, motif_feature_to_peaks, total_peaks, bucket_peak_indices, peak_lfc
        )
        pair_stats = _enrichment_statistics(
            pair_rows_ordered, pair_keys, pair_to_peaks, total_peaks, bucket_peak_indices, peak_lfc
        ) if pair_rows_ordered else []
        motif_stats_path = result_path(source_machine, "motif_statistics.csv", file_prefix=source_prefix, modality=ATAC_MOTIF_MODALITY)
        pair_stats_path = result_path(source_machine, "motif_pair_statistics.csv", file_prefix=source_prefix, modality=ATAC_MOTIF_PAIR_MODALITY)
        if motif_stats:
            _write_csv(motif_stats_path, motif_stats, list(motif_stats[0]))
        else:
            _write_csv(motif_stats_path, [], ["motif_id"])
        if pair_stats:
            _write_csv(pair_stats_path, pair_stats, list(pair_stats[0]))
        else:
            _write_csv(pair_stats_path, [], ["pair_id"])
        written_results += [motif_stats_path, pair_stats_path]

        # No cut copies and no network figure: the full tables are the result, and OmicView draws them.
        comparison_results[result_key] = {
            # Recorded, not just encoded in the filename, so a reader of the
            # checkpoint never parses a prefix to learn which statistic,
            # comparison or cell type produced a result.
            "folder": folder,
            "comparison": comparison_name,
            "stratum": stratum_label,
            "kind": str(source.get("kind") or ""),
            "control": source.get("control"),
            "treatment": source.get("treatment"),
            "differential_statistics": str(source["path"]),
            "up_peaks": len(bucket_peak_indices["up"]),
            "down_peaks": len(bucket_peak_indices["down"]),
            "nonsignificant_peaks": len(bucket_peak_indices["nonsig"]),
            "motif_statistics": str(motif_stats_path),
            "pair_statistics": str(pair_stats_path),
        }

    motif_metadata_rows = [
        {
            "motif_id": motif.motif_id,
            "motif_name": motif.name,
            "width": motif.width,
            "score_threshold": thresholds[index],
        }
        for index, motif in enumerate(motifs)
    ]
    motif_metadata_path = result_path(machine, "motif_metadata.csv", file_prefix=file_prefix, modality=ATAC_MOTIF_MODALITY)
    _write_csv(motif_metadata_path, motif_metadata_rows, list(motif_metadata_rows[0]))
    scan_summary_path = result_path(machine, "motif_scan_summary.csv", file_prefix=file_prefix, modality=ATAC_MOTIF_MODALITY)
    scan_summary_rows = [
        {
            "genome": resolved["genome"],
            "motif_species": resolved["motif_species"],
            "motif_library": resolved["motif_library"],
            "jaspar_release": resolved["jaspar_release"],
            "motifs_scanned": len(motifs),
            # Peaks_scanned is the universe -- the peaks the scan read.
            # The matrix's row count and what was left out are beside it.
            "peaks_scanned": total_peaks,
            "peaks_in_matrix": len(peak_rows),
            "peaks_skipped_wide": len(unscanned.oversize),
            "peaks_skipped_unresolved_contig": len(unscanned.unresolved),
            "motifs_tested": len(motif_rows),
            "motif_pairs_tested": len(pair_rows_ordered),
            "motif_p_threshold": motif_p_threshold,
            # Recorded because it DEFINES cooccurring_peak_count; see
            # the checkpoint entry below.
            "max_motifs_per_peak": int(max_motifs_per_peak),
            "max_peak_width": int(max_peak_width),
            "method": MOTIF_ANALYSIS_METHOD,
            "scanner_backend_requested": scanner_backend,
            "background_A": float(background[0]),
            "background_C": float(background[1]),
            "background_G": float(background[2]),
            "background_T": float(background[3]),
        }
    ]
    _write_csv(scan_summary_path, scan_summary_rows, list(scan_summary_rows[0]))
    written_results += [motif_metadata_path, scan_summary_path]

    # Whatever this stage wrote on an earlier run and did not write now
    # goes, before the checkpoint says these results are the run's. See
    # _remove_results_not_written_by_this_run.
    removed_results = _remove_results_not_written_by_this_run(results, written_results)

    checkpoint_path = checkpoints / "motif_analysis.json"
    result = {
        "complete": stage != "singles",
        "stage": "motif_analysis",
        "motif_run_stage": stage,
        "singles_complete": True,
        "pairs_complete": stage != "singles" and not effective_skip_pairs,
        # Read by pipeline.state; a checkpoint under another method is out
        # of date and the stage runs again. See MOTIF_ANALYSIS_METHOD.
        "method": MOTIF_ANALYSIS_METHOD,
        "interpretation": "motif potential and motif co-occurrence; not direct TF binding",
        "genome": resolved["genome"],
        "reference_fasta": str(resolved["reference_fasta"]),
        "motif_database": str(resolved["motif_database"]),
        "motif_database_source": resolved["motif_database_source"],
        "motif_species": resolved["motif_species"],
        "motif_library": resolved["motif_library"],
        "motif_p_threshold": motif_p_threshold,
        # Recorded because it DEFINES cooccurring_peak_count, and so
        # every selectivity number derived from it. Pairs are formed only from
        # the strongest `max_motifs_per_peak` motifs in each peak, so a
        # downstream reader joining motif_instances raw finds co-occurrences
        # this analysis never counted -- a peak with 200 hits gives C(200,2)
        # against C(25,2) here. That mismatch produced a 291 GB locus file
        # whose rows did not correspond to the |S| its q-values described.
        # Nothing could reproduce the rule because nothing recorded it.
        "max_motifs_per_peak": int(max_motifs_per_peak),
        "max_peak_width": int(max_peak_width),
        "sample_manifest": str(Path(sample_manifest).expanduser().resolve()),
        "motifs_scanned": len(motifs),
        # The scanned peaks -- the universe of every rate and test. The
        # matrix's row count and the peaks left out are recorded beside it.
        "peaks_scanned": total_peaks,
        "peaks_in_matrix": len(peak_rows),
        "peaks_skipped_wide": len(unscanned.oversize),
        "peaks_skipped_unresolved_contig": len(unscanned.unresolved),
        "unresolved_contigs": list(unscanned.contigs),
        # Every table the enrichment was asked to classify against,
        # one per comparison of every statistic folder (and stratum), and the
        # ones that classified no peak and so have no result.
        "sources": [
            {
                "key": _source_key(source),
                "folder": str(source["folder"] or ""),
                "comparison": str(source["comparison"]),
                "stratum": str(source["stratum"] or ""),
                "kind": str(source.get("kind") or ""),
                "control": source.get("control"),
                "treatment": source.get("treatment"),
                "differential_statistics": str(source["path"]),
                "enriched": _source_key(source) in comparison_results,
            }
            for source in differential_sources
        ],
        "comparisons_named_from": named_from,
        "sources_skipped": [_source_key(source) for source in skipped_sources],
        "motifs_tested": len(motif_rows),
        "motif_pairs_tested": len(pair_rows_ordered),
        "skip_motif_pairs": bool(effective_skip_pairs),
        # See MOTIF_PAIR_STATISTICS_METHOD.
        "pair_statistics_method": MOTIF_PAIR_STATISTICS_METHOD,
        "results_removed_from_earlier_run": [str(path) for path in removed_results],
        "motif_instances": str(instances_path),
        "peak_motif_matrix": str(matrix_path),
        # No motif_activity_by_sample.csv / motif_pair_activity_by_sample.csv: there is no CPM
        # activity matrix. pipeline/state.py's status summary reports these two keys as "missing".
        "motif_metadata": str(motif_metadata_path),
        "scan_summary": str(scan_summary_path),
        "comparisons": comparison_results,
    }
    if precomputed_plan is not None:
        execution_plan = json.loads(Path(precomputed_plan).read_text())
        result.update({
            "backend": execution_plan.get("backend"),
            "plan_id": execution_plan.get("plan_id"),
            "task_count": len(execution_plan.get("tasks") or []),
            "resource_plan": {**(execution_plan.get("resource_plan") or {}), "sample_count": len(sample_rows), "biological_sample_count": len(sample_rows)},
            "execution_plan": str(Path(precomputed_plan)),
        })
    print(f"Motif finalizer phase: writing checkpoint ({time.monotonic() - stage_started:.1f}s)", flush=True)
    _atomic_json(checkpoint_path, {"schema": "paralome.atac.motif_analysis.v3", **result})
    result["checkpoint"] = str(checkpoint_path)
    if stage == "pairs" and result["pairs_complete"]:
        # Pairs finished successfully off the singles pickle -- it has
        # served its purpose, clean it up rather than leaving a stale
        # multi-hundred-MB pickle around next to the real outputs.
        _discard_singles_state(singles_state_path)
    return result
