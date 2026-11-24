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
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from paralome.rna.gene_model import parse_gtf_genes


class RnaPreflightError(RuntimeError):
    """Raised when RNA quantification inputs fail preflight validation.

    Aggregates every problem found rather than stopping at the first, so a
    single preflight run reports everything that needs fixing. Callers must
    not submit any worker job (local or SLURM) when this is raised.
    """

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        message = "RNA quantification preflight failed:\n" + "\n".join(
            f"  - {problem}" for problem in self.problems
        )
        super().__init__(message)


def _bam_index_path(bam: Path, index_dir: Optional[Path] = None) -> Optional[Path]:
    # Both index namings: X.bam.bai, and X.bai (Picard) or X.bam.csi.
    bam = Path(bam)
    candidates = [Path(str(bam) + ".bai"), bam.with_suffix(".bai"), Path(str(bam) + ".csi")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # The index Paralome made itself, for a BAM in a folder it
    # could not write to -- see ensure_bam_indexes. Not used once the BAM is
    # newer than it (a BAM replaced under the same name).
    if index_dir is not None:
        made = project_bam_index_path(bam, index_dir)
        try:
            if made.is_file() and made.stat().st_mtime >= bam.stat().st_mtime:
                return made
        except OSError:
            return None
    return None


#: Where indexes Paralome made for BAMs in unwritable folders go,
#: under the quantification's --out.
BAM_INDEX_SUBDIR = Path("inputs") / "bam_indexes"


def project_bam_index_dir(out: Union[str, Path]) -> Path:
    """The project-owned folder for BAM indexes that could not go beside the BAM."""
    return Path(out) / BAM_INDEX_SUBDIR


def project_bam_index_path(bam: Union[str, Path], index_dir: Union[str, Path]) -> Path:
    """One BAM's index in ``index_dir``: its name plus a digest of its full path,
    so two samples' ``Aligned.sortedByCoord.out.bam`` do not share one index."""
    bam = Path(bam)
    try:
        resolved = str(bam.resolve())
    except OSError:
        resolved = str(bam.absolute())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", bam.name) or "bam"
    return Path(index_dir) / f"{stem}.{digest}.bai"


def _can_write_beside(bam: Path) -> bool:
    return os.access(str(bam.parent), os.W_OK | os.X_OK)


def _samtools_index(bam: Path, target: Path) -> Optional[str]:
    """Index ``bam`` into ``target`` with samtools; None, or why it failed.

    Written to a temporary name and renamed, so a failed or interrupted run
    never leaves a partial index that a later run would take for a real one.
    """
    from paralome.binaries import resolve_tool

    try:
        samtools = resolve_tool("samtools")
    except Exception as exc:  # noqa: BLE001 -- PLM-0410, a stale recorded path
        return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    if not samtools:
        return "samtools was not found (it is not recorded by `paralome setup` and not on PATH)"
    temporary = target.with_name(f"{target.name}.tmp{os.getpid()}")
    try:
        completed = subprocess.run(
            [samtools, "index", "-b", str(bam), str(temporary)],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        return f"samtools could not be run: {exc}"
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        lines = [line for line in (completed.stderr or "").splitlines() if line.strip()]
        return "samtools index failed (" + (lines[-1].strip() if lines else f"exit {completed.returncode}") + ")"
    os.replace(temporary, target)
    return None


def ensure_bam_indexes(
    samples: Sequence[object], index_dir: Optional[Path], *, stage: str = "rna_fpkm_quantification",
) -> Tuple[List[Tuple[str, Path, Path]], List[str]]:
    """Index every BAM of ``samples`` that has no index. (indexed, problems)

    A BAM without an index is indexed, not
    refused. The index is written beside the BAM as ``<bam>.bai`` when that
    folder can be written to, and otherwise under ``index_dir`` (the
    quantification's ``inputs/bam_indexes``), where ``_bam_index_path`` and
    every reader that asks it find it. Nothing reads the index but
    ``--strandedness auto``'s mapped-read totals; counting reads the BAM
    from start to end. Under SLURM this runs on the login node, before
    anything is submitted. Each index made is printed.
    """
    indexed: List[Tuple[str, Path, Path]] = []
    problems: List[str] = []
    for sample in samples:
        bam = Path(getattr(sample, "bam"))
        sample_id = getattr(sample, "sample_id", str(bam))
        if not bam.is_file() or _bam_index_path(bam, index_dir) is not None:
            continue
        beside = Path(str(bam) + ".bai")
        # One place, not a fallback: when samtools fails beside the BAM it
        # would fail the same way in the project folder.
        targets: List[Path] = []
        if _can_write_beside(bam):
            targets.append(beside)
        elif index_dir is not None:
            targets.append(project_bam_index_path(bam, index_dir))
        failures: List[str] = []
        for target in targets:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                failures.append(f"{target.parent} could not be created: {exc}")
                continue
            failure = _samtools_index(bam, target)
            if failure is None:
                indexed.append((str(sample_id), bam, target))
                print(f"[paralome] {stage}: sample {sample_id!r} had no BAM index; indexed {bam} -> {target}",
                      file=sys.stderr)
                break
            failures.append(failure)
        else:
            reason = "; ".join(failures) or (
                f"its folder {bam.parent} cannot be written to and this command has no "
                "project folder to put an index in"
            )
            problems.append(
                f"BAM index (.bai or .csi) not found for sample {sample_id!r} and it could not be "
                f"made: {reason}. Index it with `samtools index {bam}` (the index goes beside the "
                f"BAM, as {beside.name}) as someone who can write to {bam.parent}, or point the "
                f"sheet at a copy of the BAM in a folder you can write to: {bam}"
            )
    return indexed, problems


def _bam_looks_paired(bam_path, limit: int = 2000) -> Optional[bool]:
    """Whether the first mapped primary reads are paired; None if it cannot be told."""
    try:
        import pysam

        paired = single = 0
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if read.is_paired:
                    paired += 1
                else:
                    single += 1
                if paired + single >= limit:
                    break
    except Exception:  # noqa: BLE001 -- an unreadable BAM is reported elsewhere
        return None
    if paired and not single:
        return True
    if single and not paired:
        return False
    return None


def _bam_reference_names(bam_path) -> Optional[List[str]]:
    """The @SQ names of a BAM's header; None if the file cannot be opened."""
    try:
        import pysam

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            return list(bam.references)
    except Exception:  # noqa: BLE001 -- an unreadable BAM is reported elsewhere
        return None


def gtf_bam_chromosome_mismatch(
    gtf_chromosomes: Sequence[str], bam_references: Sequence[str],
) -> Optional[str]:
    """Why no fragment could reach a gene, or None when the names overlap.

    A chr-prefixed GTF against an Ensembl-named BAM (or the
    reverse) assigns every fragment to no gene: strandedness detection came
    back "inconclusive", and quantification wrote an all-zero matrix under
    complete=True. The names are compared here, before any job is created.
    """
    gtf_names = {str(name) for name in gtf_chromosomes if str(name)}
    bam_names = {str(name) for name in bam_references if str(name)}
    if not gtf_names or not bam_names or gtf_names & bam_names:
        return None

    def sample(names) -> str:
        shown = sorted(names)[:4]
        return ", ".join(shown) + (", ..." if len(names) > len(shown) else "")

    hint = ""
    stripped_gtf = {name[3:] if name.startswith("chr") else name for name in gtf_names}
    stripped_bam = {name[3:] if name.startswith("chr") else name for name in bam_names}
    if stripped_gtf & stripped_bam:
        hint = (
            " They differ only by the 'chr' prefix: use a GTF from the same source as "
            "the genome the reads were aligned to (both UCSC-style or both Ensembl-style)."
        )
    return (
        f"the GTF's chromosome names ({sample(gtf_names)}) match none of the BAM's "
        f"reference names ({sample(bam_names)}), so no fragment could be assigned to "
        f"any gene and every FPKM would be 0.{hint}"
    )


def preflight_rna_quantification(
    samples_path: Union[str, Path],
    gtf_path: Union[str, Path],
    *,
    strandedness: int,
    paired_end: bool,
    index_dir: Optional[Path] = None,
) -> dict:
    """Validate every RNA-quantification input before any job is created.

    Checks the sample sheet, each BAM file and its index (a
    missing index is made -- see ensure_bam_indexes; ``index_dir`` is where
    it goes when the BAM's folder cannot be written to), the GTF, the
    strandedness value, the resolved sequencing mode, and that pysam (the
    only counting dependency) is importable. Raises :class:`RnaPreflightError`
    listing every problem found if anything is invalid; on success returns a
    small summary dict.
    """
    problems: List[str] = []

    try:
        import pysam  # noqa: F401
    except ImportError:
        problems.append(
            "pysam is not importable; it is required for native RNA quantification"
        )

    if strandedness not in {0, 1, 2}:
        problems.append(f"strandedness must be 0, 1, or 2 (got {strandedness!r})")

    if not isinstance(paired_end, bool):
        problems.append(
            f"paired_end must be explicitly resolved to True or False before preflight, "
            f"not {paired_end!r}; refusing to guess sequencing mode"
        )

    samples: list = []
    try:
        from paralome.rna.fpkm import _read_samples

        samples = _read_samples(Path(samples_path))
    except Exception as exc:  # noqa: BLE001 - aggregate every problem, do not stop early
        problems.append(f"RNA sample sheet is invalid: {exc}")

    gene_count = 0
    gtf_chromosomes: List[str] = []
    try:
        genes = parse_gtf_genes(gtf_path)
        gene_count = len(genes)
        gtf_chromosomes = sorted({gene.chrom for gene in genes})
    except Exception as exc:  # noqa: BLE001
        problems.append(f"GTF is invalid or unusable: {exc}")

    # Indexed rather than refused -- see ensure_bam_indexes. A BAM
    # that is missing altogether is reported by the sheet reader above.
    indexed, index_problems = ensure_bam_indexes(samples, index_dir)
    problems.extend(index_problems)
    for sample in samples:
        # Nothing compared the BAM with the layout asked for. A
        # single-end BAM counted as paired-end has a library size of zero (the
        # size is counted from first mates), so every FPKM came out 0 under a
        # checkpoint saying complete; a paired BAM counted as single-end
        # counted each mate as a fragment.
        if isinstance(paired_end, bool):
            looks_paired = _bam_looks_paired(sample.bam)
            if looks_paired is not None and looks_paired != paired_end:
                problems.append(
                    f"sample {sample.sample_id!r}: the BAM holds "
                    f"{'paired-end' if looks_paired else 'single-end'} reads but the run was told "
                    f"{'--paired-end' if paired_end else '--single-end'}: {sample.bam}"
                )
        # The GTF's chromosomes against the BAM's references -- see
        # gtf_bam_chromosome_mismatch.
        if gtf_chromosomes:
            references = _bam_reference_names(sample.bam)
            if references is not None:
                mismatch = gtf_bam_chromosome_mismatch(gtf_chromosomes, references)
                if mismatch:
                    problems.append(f"sample {sample.sample_id!r}: {mismatch} ({sample.bam})")

    if problems:
        raise RnaPreflightError(problems)

    return {
        "sample_count": len(samples),
        "gene_count": gene_count,
        "strandedness": strandedness,
        "paired_end": paired_end,
        "indexed": [{"sample_id": sid, "bam": str(bam), "index": str(index)} for sid, bam, index in indexed],
    }
