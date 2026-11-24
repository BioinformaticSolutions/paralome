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

"""One reading of a variant's population allele frequency.

Segregation writes the frequency columns of its table from the ANNOVAR INFO
keys; filtering (filters/quality.py evaluate_population) and reporting
(reports/candidates.py population_bucket) read them back. All three read
through ``read_population_af``:

* the same keys (``ROW_AF_KEYS`` for a table row, ``INFO_AF_KEYS`` for INFO);
* the same missing values (``MISSING_VALUES``: '', '.', 'NA', ...);
* the largest frequency of every key -- of the ONE allele the row reports
  (its ``reported_alt`` column), or, for a table written before
  that column existed, of every allele, so such a variant is never called
  rarer than one of its frequencies says;
* and a status that says WHY there is no frequency. ``absent``: every key is
  missing -- for ANNOVAR with gnomad41_exome this is a variant gnomAD's
  exomes do not hold, which on a whole-genome run is most of them (intronic,
  intergenic, regulatory). ``unparseable``: a value is there and is not a
  frequency (text, or outside 0-1). The report counts the two apart.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

#: Values that mean "no frequency": ANNOVAR's '.', a blank, and the NA/NaN
#: spellings other tools write.
MISSING_VALUES = frozenset({"", ".", "NA", "na", "N/A", "nan", "NaN", "NAN", "None", "none"})

#: INFO keys segregation reads a frequency from, exact spelling (ANNOVAR
#: gnomAD 4.1 exome first; the genome keys when that table is annotated too).
INFO_AF_KEYS = (
    "gnomad41_exome_AF_grpmax",
    "gnomad41_exome_AF",
    "gnomad41_genome_AF_grpmax",
    "gnomad41_genome_AF",
    "gnomAD_exome_AF",
    "gnomAD_genome_AF",
    "gnomAD_AF",
    "AF_popmax",
)

#: Columns a table row carries a frequency in: the segregation table's own
#: (``max_population_af``, ``gnomad_af_grpmax``, ``gnomad_af``) and, for a
#: table that kept the INFO keys as columns, those.
ROW_AF_KEYS = (
    "max_population_af",
    "gnomad_af_grpmax",
    "gnomad_af",
    *INFO_AF_KEYS,
    "popmax_af",
)

STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"
STATUS_UNPARSEABLE = "unparseable"
STATUSES = (STATUS_PRESENT, STATUS_ABSENT, STATUS_UNPARSEABLE)

#: What an absent frequency means, for the reader of a report. gnomAD exome
#: is the one frequency table on the owner's genome (no gnomad41_genome yet).
ABSENT_MEANING_EXOME = "not in gnomAD exomes (no frequency available)"
ABSENT_MEANING_EXOME_AND_GENOME = "not in gnomAD exomes or genomes (no frequency available)"
UNPARSEABLE_MEANING = "a frequency value is present but is not a number between 0 and 1"

_SPLIT = re.compile(r"[,;|]")


@dataclass(frozen=True)
class PopulationAF:
    af: Optional[float]
    source: str
    status: str


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() in MISSING_VALUES


def parse_af_values(value: Any) -> tuple[list[float], bool]:
    """(every frequency in the field, whether some token was not a frequency).

    A multi-allelic field ('0.1,.,0.3') gives each allele's value; missing
    tokens are skipped. A token that is not a number, or is not within 0-1,
    is not a frequency.
    """
    if is_missing(value):
        return [], False
    numbers: list[float] = []
    unparseable = False
    for token in _SPLIT.split(str(value)):
        token = token.strip()
        if token in MISSING_VALUES:
            continue
        try:
            number = float(token)
        except ValueError:
            unparseable = True
            continue
        if math.isnan(number) or math.isinf(number) or number < 0 or number > 1:
            unparseable = True
            continue
        numbers.append(number)
    return numbers, unparseable


#: The column naming the ALT allele a row reports (segregation
#: writes it; see paralome.segregation.table.choose_reported_allele).
REPORTED_ALT_COLUMN = "reported_alt"


def reported_allele_index(row: Mapping[str, Any]) -> tuple[Optional[int], int]:
    """(index of the reported ALT among the row's ALTs, number of ALTs).

    The index is None when the row names no reported allele (a table written), or names one that is not among its ALTs.
    """
    alts = [alt.strip() for alt in str(row.get("alt") or row.get("ALT") or "").split(",")]
    reported = row.get(REPORTED_ALT_COLUMN)
    if is_missing(reported):
        return None, len(alts)
    reported = str(reported).strip()
    if reported not in alts:
        return None, len(alts)
    return alts.index(reported), len(alts)


def _allele_value(value: Any, index: Optional[int], n_alts: int) -> Any:
    """The reported allele's token of a per-allele list ('0.35,2e-05')."""
    if index is None or n_alts <= 1 or is_missing(value):
        return value
    tokens = str(value).split(",")
    if len(tokens) != n_alts:
        return value
    return tokens[index]


def read_population_af(
    row: Mapping[str, Any],
    keys: Sequence[str] = ROW_AF_KEYS,
    *,
    source_column: Optional[str] = "population_af_source",
) -> PopulationAF:
    """The largest population AF of ``row`` over ``keys``, its key, and a status.

    ``source`` is the key the largest value was read from. Where that is the
    segregation table's ``max_population_af``, the INFO key segregation
    recorded in ``source_column`` is given instead, when there is one.

    Of the reported allele only, when the row names one in
    ``reported_alt``: a field holding one value per ALT ('0.35,2e-05' for
    ALT 'G,T') is read at that allele's place. The consequence and gene of
    the row are that allele's, so its frequency must be too.
    """
    index, n_alts = reported_allele_index(row)
    best: Optional[float] = None
    source = ""
    unparseable = False
    for key in keys:
        if key not in row:
            continue
        numbers, bad = parse_af_values(_allele_value(row.get(key), index, n_alts))
        unparseable = unparseable or bad
        for number in numbers:
            if best is None or number > best:
                best = number
                source = key
    if best is None:
        return PopulationAF(None, "", STATUS_UNPARSEABLE if unparseable else STATUS_ABSENT)
    if source == "max_population_af" and source_column and not is_missing(row.get(source_column)):
        source = str(row.get(source_column)).strip()
    return PopulationAF(best, source, STATUS_PRESENT)


def read_population_af_alleles(blocks: Iterable[Mapping[str, Any]], keys: Sequence[str] = INFO_AF_KEYS) -> PopulationAF:
    """``read_population_af`` over every ALT allele's annotation of one record.

    No longer what a segregation row reports -- that is the one
    allele the row reports (segregation.table.choose_reported_allele). Kept
    for a reader that wants the largest of all.
    """
    best = PopulationAF(None, "", STATUS_ABSENT)
    unparseable = False
    for block in blocks:
        found = read_population_af(block, keys, source_column=None)
        unparseable = unparseable or found.status == STATUS_UNPARSEABLE
        if found.af is not None and (best.af is None or found.af > best.af):
            best = found
    if best.af is None and unparseable:
        return PopulationAF(None, "", STATUS_UNPARSEABLE)
    return best


def absent_meaning(sources_seen: Iterable[str]) -> str:
    """What 'no frequency' means in a run whose frequencies came from ``sources_seen``."""
    if any("genome" in str(source).lower() for source in sources_seen):
        return ABSENT_MEANING_EXOME_AND_GENOME
    return ABSENT_MEANING_EXOME
