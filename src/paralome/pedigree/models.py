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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PedigreeSample:
    sample_id: str
    family_id: str = "default"
    group: str = "unknown"
    relationship: str = "unknown"
    sex: str = "unknown"
    mother_id: str = ""
    father_id: str = ""
    bam: Optional[Path] = None

    @property
    def affected(self) -> Optional[bool]:
        # The same words, in any case, as the segregation scorer
        # (segregation.scorer CASE_GROUPS / CONTROL_GROUPS). A value that is
        # neither, and is not an explicit "unknown", is refused rather than
        # read as unknown.
        from paralome.segregation.scorer import CASE_GROUPS, CONTROL_GROUPS

        group = str(self.group or "").strip().lower()
        if group in CASE_GROUPS:
            return True
        if group in CONTROL_GROUPS:
            return False
        if group in {"", ".", "unknown", "na", "missing"}:
            return None
        raise ValueError(
            f"Sample {self.sample_id}: affected status {self.group!r} is not recognised. "
            f"Use one of {sorted(CASE_GROUPS)} for affected or {sorted(CONTROL_GROUPS)} for unaffected."
        )


@dataclass
class Family:
    family_id: str
    samples: dict[str, PedigreeSample] = field(default_factory=dict)

    def add(self, sample: PedigreeSample) -> None:
        self.samples[sample.sample_id] = sample

    @property
    def probands(self) -> list[PedigreeSample]:
        return [s for s in self.samples.values() if s.relationship == "proband"]

    @property
    def affected(self) -> list[PedigreeSample]:
        return [s for s in self.samples.values() if s.affected is True]

    @property
    def unaffected(self) -> list[PedigreeSample]:
        return [s for s in self.samples.values() if s.affected is False]


    def summary(self) -> dict:
        return {
            "family_id": self.family_id,
            "samples": len(self.samples),
            "probands": [s.sample_id for s in self.probands],
            "affected": [s.sample_id for s in self.affected],
            "unaffected": [s.sample_id for s in self.unaffected],
        }
