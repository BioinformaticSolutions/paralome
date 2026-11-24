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

"""The MACS3 ``--gsize`` for ``--genome-size auto``: the summed length of the reference's canonical contigs.

WHAT 'auto' IS. MACS3's --gsize is meant to be the EFFECTIVE (mappable) genome
size; its shortcuts are hs = 2.7e9 and mm = 1.87e9 in MACS2, 2.9e9 and 1.87e9
in MACS3 (the deepTools non-N counts are 2.91e9 for GRCh38 and 2.65e9 for
GRCm38). 'auto' sums the lengths in the .fai of the contigs that are not
mitochondrial and not alternate, random, unplaced, fix or decoy sequence -- N
gaps and unmappable sequence included -- so for GRCh38 it gives about 3.1e9,
an UPPER BOUND on the effective size. A larger --gsize makes the genome-wide
background lambda smaller, so somewhat more weak peaks pass; the local lambda
usually dominates, and the effect is modest. The value and how it was derived
are printed when 'auto' is resolved, so a run shows what MACS was given, and
anyone wanting the MACS convention passes ``--genome-size hs`` (or mm, ce, dm,
or a number).

Contigs are excluded by the ASSEMBLY'S OWN NAMING, anchored: ``_alt``,
``_random``, ``_decoy``, ``_fix``, ``Un_`` (UCSC chrUn_*), and the GenBank
scaffold accessions ``KI``, ``GL``, ``JH`` followed by digits at the start of
the name or after an underscore (chr1_KI270706v1_random, chrUn_GL000195v1,
Ensembl's KI270706.1, mouse JH584304), so a canonical contig of a non-human
assembly whose name happens to contain those letters is kept.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from paralome.errors import ParalomeError

_EXCLUDED_MARKERS = ("_alt", "_random", "_decoy", "_fix", "Un_")
#: GenBank scaffold accessions of unplaced and unlocalized human and mouse
#: contigs, at the start of the name or after an underscore, followed by digits.
_SCAFFOLD_ACCESSION = re.compile(r"(?:^|_)(?:KI|GL|JH)\d+")


def _is_noncanonical(name: str) -> bool:
    return any(marker in name for marker in _EXCLUDED_MARKERS) or bool(_SCAFFOLD_ACCESSION.search(name))


def macs_genome_size_from_fai(reference_fasta: str | Path, *, include_all_contigs: bool = False,
                              report: bool = False) -> int:
    """The summed .fai length of the canonical contigs (N gaps included; see the module docstring).

    With ``report`` the value and its derivation are printed to stderr.
    """
    fasta = Path(reference_fasta)
    fai = Path(str(fasta) + ".fai")
    if not fai.exists():
        raise ParalomeError(104, f"FASTA index not found: {fai}")
    total = 0
    counted = 0
    excluded: list[str] = []
    for line in fai.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"Malformed FASTA index line: {line!r}")
        name, length = fields[0], int(fields[1])
        normalized = name[3:] if name.startswith("chr") else name
        mitochondrial = normalized in {"M", "MT"}
        if include_all_contigs or (not mitochondrial and not _is_noncanonical(name)):
            total += length
            counted += 1
        else:
            excluded.append(name)
    if total <= 0:
        raise ValueError(f"No eligible contigs found in {fai}")
    if report:
        shown = ", ".join(excluded[:6]) + (f" and {len(excluded) - 6} more" if len(excluded) > 6 else "")
        print(
            f"NOTE: MACS3 --gsize auto = {total:,} = the summed length of {counted} contig(s) in {fai}"
            + (f" ({len(excluded)} excluded as mitochondrial or non-canonical: {shown})" if excluded else "")
            + ". N gaps are included, so this is an upper bound on the effective genome size; "
            "pass --genome-size hs, mm, or a number for the MACS convention.",
            file=sys.stderr,
        )
    return total


def resolve_macs_genome_size(value: str | int | None, reference_fasta: str | Path | None = None) -> str:
    if value not in (None, "", "auto"):
        return str(value)
    if reference_fasta is None:
        raise ValueError("MACS genome size 'auto' requires a reference FASTA")
    return str(macs_genome_size_from_fai(reference_fasta, report=True))
