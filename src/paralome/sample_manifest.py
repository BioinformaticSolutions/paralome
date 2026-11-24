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
from pathlib import Path
import csv
from typing import Iterable, List, Optional

from paralome.csvio import skip_comment_lines


ALLOWED_GROUPS = {
    "case",
    "affected",
    "phenotype_positive",
    "control",
    "unaffected",
    "phenotype_negative",
    "unknown",
    "",
}

ALLOWED_RELATIONSHIPS = {
    "proband",
    "mother",
    "father",
    "sibling",
    "brother",
    "sister",
    "cousin",
    "second-cousin",
    "second_cousin",
    "unrelated",
    "unknown",
    "",
}

RELATIONSHIP_NORMALIZATION = {
    "": "unknown",
    "second_cousin": "second-cousin",
    "brother": "sibling",
    "sister": "sibling",
}

AFFECTED_GROUPS = {"case", "affected", "phenotype_positive"}
UNAFFECTED_GROUPS = {"control", "unaffected", "phenotype_negative"}

#: The other affected codings segregation/scorer.py accepts, read
#: here as affected / unaffected so the two readers agree on one sheet. Digits
#: are refused in both: 1 means affected in 0/1 coding and unaffected in PED.
GROUP_ALIASES = {
    "positive": "affected", "yes": "affected", "true": "affected",
    "negative": "unaffected", "no": "unaffected", "false": "unaffected",
}

#: What star.read_fastq_manifest writes for a blank cell.
BLANK_VALUES = {"", "."}


def _blank_to_empty(value: Optional[str]) -> str:
    text = (value or "").strip()
    return "" if text in BLANK_VALUES else text


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    group: str
    bam: Path
    relationship: str = "unknown"
    family_id: str = "default"
    sex: str = "unknown"
    mother_id: str = ""
    father_id: str = ""

    @property
    def affected(self) -> Optional[bool]:
        if self.group in AFFECTED_GROUPS:
            return True
        if self.group in UNAFFECTED_GROUPS:
            return False
        return None



def _normalize_group(value: Optional[str], sample_id: str) -> str:
    group = _blank_to_empty(value).lower() or "unknown"
    group = GROUP_ALIASES.get(group, group)
    if group not in ALLOWED_GROUPS:
        raise ValueError(
            f"Invalid group '{group}' for sample {sample_id}. "
            "Common values are case/control, affected/unaffected, or unknown."
            + (" Numeric codes are ambiguous (PED uses 2 = affected, 1 = unaffected; "
               "others use 1 = affected, 0 = unaffected): write affected or unaffected."
               if group.lstrip("-").isdigit() else "")
        )
    return group or "unknown"


def _normalize_relationship(value: Optional[str], sample_id: str) -> str:
    relationship = _blank_to_empty(value).lower() or "unknown"
    relationship = RELATIONSHIP_NORMALIZATION.get(relationship, relationship)
    if relationship not in ALLOWED_RELATIONSHIPS:
        raise ValueError(
            f"Invalid relationship '{relationship}' for sample {sample_id}. "
            "Allowed values: proband, mother, father, sibling, cousin, "
            "second-cousin, unrelated, unknown"
        )
    return relationship or "unknown"


def _normalize_sex(value: Optional[str]) -> str:
    sex = (value or "unknown").strip().lower()
    if sex in {"m", "male"}:
        return "male"
    if sex in {"f", "female"}:
        return "female"
    if sex in {"", ".", "u", "unknown", "na", "n/a"}:
        return "unknown"
    return sex


def read_sample_manifest(path: Path) -> List[SampleRecord]:
    path = Path(path)
    samples: List[SampleRecord] = []

    # Utf-8-sig, so the byte-order mark Excel writes at the top of a
    # CSV is dropped instead of becoming part of the first column's name.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        # A line starting with '#' is a sample switched off, in every
        # Paralome sample sheet. This reader did not skip them, so a
        # commented-out sample came through as a sample called "#NAME".
        reader = csv.DictReader(skip_comment_lines(fh))
        fieldnames = set(reader.fieldnames or [])
        required = {"sample_id", "bam"}
        missing = required - fieldnames
        if not ({"affected", "group", "status"} & fieldnames):
            missing.add("affected/group/status")
        if missing:
            raise ValueError(
                f"Sample manifest is missing required column(s): {', '.join(sorted(missing))}"
            )

        for row in reader:
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError("Sample manifest contains a row with an empty sample_id")

            # '.' is a blank cell here, so the next column is tried.
            group = _normalize_group(
                next((v for v in (row.get("affected"), row.get("group"), row.get("status"))
                      if _blank_to_empty(v)), ""),
                sample_id,
            )
            relationship = _normalize_relationship(row.get("relationship"), sample_id)
            family_id = _blank_to_empty(row.get("family_id")) or "default"
            sex = _normalize_sex(row.get("sex"))
            mother_id = _blank_to_empty(row.get("mother_id"))
            father_id = _blank_to_empty(row.get("father_id"))
            bam_text = (row.get("bam") or "").strip()
            if not bam_text:
                raise ValueError(f"Sample {sample_id} has an empty BAM path")

            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    group=group,
                    bam=Path(bam_text),
                    relationship=relationship,
                    family_id=family_id,
                    sex=sex,
                    mother_id=mother_id,
                    father_id=father_id,
                )
            )

    if not samples:
        raise ValueError(f"Sample manifest contains no samples: {path}")

    return samples


def ordered_samples(samples: Iterable[SampleRecord]):
    group_order = {
        "case": 0,
        "affected": 0,
        "phenotype_positive": 0,
        "control": 1,
        "unaffected": 1,
        "phenotype_negative": 1,
        "unknown": 2,
    }

    relationship_order = {
        "proband": 0,
        "mother": 1,
        "father": 2,
        "sibling": 3,
        "cousin": 4,
        "second-cousin": 5,
        "unrelated": 6,
        "unknown": 7,
    }

    return sorted(
        enumerate(samples),
        key=lambda x: (
            group_order.get(x[1].group, 99),
            relationship_order.get(x[1].relationship, 99),
            x[0],
        ),
    )

