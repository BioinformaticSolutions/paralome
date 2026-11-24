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

from typing import Iterable

from .models import Family, PedigreeSample


def families_from_samples(samples: Iterable) -> dict[str, Family]:
    families: dict[str, Family] = {}
    for sample in samples:
        family_id = getattr(sample, "family_id", "default") or "default"
        fam = families.setdefault(family_id, Family(family_id=family_id))
        fam.add(
            PedigreeSample(
                sample_id=sample.sample_id,
                family_id=family_id,
                group=getattr(sample, "group", "unknown"),
                relationship=getattr(sample, "relationship", "unknown"),
                sex=getattr(sample, "sex", "unknown"),
                mother_id=getattr(sample, "mother_id", ""),
                father_id=getattr(sample, "father_id", ""),
                bam=getattr(sample, "bam", None),
            )
        )
    return families
