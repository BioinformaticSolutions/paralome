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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from paralome.csvio import skip_comment_lines
from paralome.pipeline.profiles import ManifestProfile

FASTQ_R1_ALIASES = ("fastq_r1", "fq_r1", "r1", "read1", "fastq1")
FASTQ_R2_ALIASES = ("fastq_r2", "fq_r2", "r2", "read2", "fastq2")
BAM_ALIASES = ("bam", "bam_path", "alignment")
SAMPLE_ALIASES = ("sample_id", "sample", "id", "name")
FAMILY_ALIASES = ("family_id", "family", "pedigree")
RELATIONSHIP_ALIASES = ("relationship", "relation")
GROUP_ALIASES = ("group", "status", "affected", "condition")
SEX_ALIASES = ("sex", "gender")

GENERAL_OPTIONAL_COLUMNS = ("replicate", "batch", "subject", "pair_id", "notes")


@dataclass
class SampleInput:
    sample_id: str
    group: str = "."
    relationship: str = "."
    family_id: str = "."
    sex: str = "."
    condition: str = "."
    bam: Optional[str] = None
    fastq_r1: Optional[str] = None
    fastq_r2: Optional[str] = None
    extra: dict[str, str] | None = None

    @property
    def input_mode(self) -> str:
        if self.bam:
            return "bam"
        if self.fastq_r1 and self.fastq_r2:
            return "fastq_pe"
        if self.fastq_r1:
            return "fastq_se"
        return "unknown"

    def to_row(self) -> dict[str, str]:
        row = dict(self.extra or {})
        row.update(
            {
                "sample_id": self.sample_id,
                "group": self.group,
                "relationship": self.relationship,
                "family_id": self.family_id,
                "sex": self.sex,
                "condition": self.condition,
            }
        )
        if self.bam:
            row["bam"] = self.bam
        if self.fastq_r1:
            row["fastq_r1"] = self.fastq_r1
        if self.fastq_r2:
            row["fastq_r2"] = self.fastq_r2
        return row


def _first(row: dict[str, str], keys: Iterable[str]) -> Optional[str]:
    low = {k.lower(): k for k in row}
    for key in keys:
        real = low.get(key.lower())
        if real is not None:
            value = (row.get(real) or "").strip()
            if value and value != ".":
                return value
    return None


def _header_has(fieldnames: Iterable[str], aliases: Iterable[str]) -> bool:
    lowered = {name.strip().lower() for name in fieldnames}
    return any(alias.lower() in lowered for alias in aliases)


def _coerce_manifest_profile(value: str | ManifestProfile | None) -> Optional[ManifestProfile]:
    if value is None:
        return None
    if isinstance(value, ManifestProfile):
        return value
    return ManifestProfile(str(value).strip().lower())


def detect_manifest_profile(fieldnames: Iterable[str]) -> ManifestProfile:
    fields = list(fieldnames)
    family_signals = (
        _header_has(fields, FAMILY_ALIASES),
        _header_has(fields, RELATIONSHIP_ALIASES),
        _header_has(fields, SEX_ALIASES),
    )
    if any(family_signals):
        return ManifestProfile.FAMILY
    if _header_has(fields, ("condition",)):
        return ManifestProfile.GENERAL
    # Legacy variant manifests commonly contain only sample_id/group plus BAM or FASTQ.
    # Preserve that interpretation when no explicit general-sequencing condition exists.
    if _header_has(fields, SAMPLE_ALIASES) and (
        _header_has(fields, BAM_ALIASES) or _header_has(fields, FASTQ_R1_ALIASES)
    ):
        return ManifestProfile.FAMILY
    raise ValueError(
        "Could not determine manifest profile. Family manifests require family metadata; "
        "general sequencing manifests require condition."
    )


def validate_manifest_columns(
    fieldnames: Iterable[str],
    profile: str | ManifestProfile,
    *,
    allow_bam: bool = True,
    factors: Iterable[str] = (),
) -> list[str]:
    profile = _coerce_manifest_profile(profile)
    assert profile is not None
    fields = list(fieldnames)
    errors: list[str] = []
    if not _header_has(fields, SAMPLE_ALIASES):
        errors.append("missing required column: sample_id")
    has_fastq = _header_has(fields, FASTQ_R1_ALIASES)
    has_bam = allow_bam and _header_has(fields, BAM_ALIASES)
    if not has_fastq and not has_bam:
        errors.append("missing required input column: fastq_r1 or bam")

    if profile is ManifestProfile.FAMILY:
        for label, aliases in (
            ("family_id", FAMILY_ALIASES),
            ("relationship", RELATIONSHIP_ALIASES),
            ("affected", ("affected", "group", "status")),
            ("sex", SEX_ALIASES),
        ):
            if not _header_has(fields, aliases):
                errors.append(f"missing required family column: {label}")
    else:
        # The groups are the --factor columns, so a sheet
        # needs those columns and no `condition`. --factor is COLUMN:CONTROL.
        wanted = [str(name).strip().partition(":")[0].strip() for name in factors if str(name).strip()]
        for name in wanted:
            if name and not _header_has(fields, (name,)):
                errors.append(f"missing --factor column: {name}")
    return errors


def read_sample_inputs(
    samples_csv: Union[str, Path],
    *,
    manifest_profile: str | ManifestProfile | None = None,
    validate_schema: bool = False,
    factors: Iterable[str] = (),
) -> list[SampleInput]:
    samples_csv = Path(samples_csv)
    # Utf-8-sig, so the byte-order mark Excel writes at the top of a
    # CSV is dropped instead of becoming part of the first column's name.
    with samples_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(skip_comment_lines(fh))
        if not reader.fieldnames:
            raise ValueError(f"Sample manifest has no header: {samples_csv}")
        # Column names stripped once, for the check and the rows alike, so ' condition' is read as
        # 'condition' everywhere.
        stripped = [name.strip() for name in reader.fieldnames]
        lowered = [name.lower() for name in stripped]
        clashes = sorted({name for name in lowered if lowered.count(name) > 1})
        if clashes and len({n.lower() for n in reader.fieldnames}) == len(reader.fieldnames):
            raise ValueError(
                f"Sample manifest {samples_csv} has columns that differ only in surrounding spaces: "
                + ", ".join(clashes) + "; rename or remove one of each"
            )
        reader.fieldnames = stripped
        profile = _coerce_manifest_profile(manifest_profile)
        if profile is None and validate_schema:
            profile = detect_manifest_profile(reader.fieldnames)
        if profile is not None and validate_schema:
            errors = validate_manifest_columns(reader.fieldnames, profile, factors=factors)
            if errors:
                raise ValueError(f"Invalid {profile.value} manifest {samples_csv}: " + "; ".join(errors))

        out: list[SampleInput] = []
        seen: set[str] = set()
        for idx, row in enumerate(reader, start=2):
            # Csv files a surplus cell under the key None, which the
            # column lookups below cannot lower-case. A trailing comma did it.
            if None in row:
                raise ValueError(
                    f"{samples_csv} line {idx} has more fields than the header "
                    f"({len(reader.fieldnames)}); remove the extra comma or value"
                )
            sample_id = _first(row, SAMPLE_ALIASES)
            if not sample_id:
                raise ValueError(f"Missing sample_id/sample in {samples_csv} line {idx}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {samples_csv} line {idx}")
            seen.add(sample_id)
            group = _first(row, GROUP_ALIASES) or "."
            relationship = _first(row, RELATIONSHIP_ALIASES) or "."
            family_id = _first(row, FAMILY_ALIASES) or "."
            sex = _first(row, SEX_ALIASES) or "."
            condition = _first(row, ("condition", "group", "status")) or "."
            bam = _first(row, BAM_ALIASES)
            r1 = _first(row, FASTQ_R1_ALIASES)
            r2 = _first(row, FASTQ_R2_ALIASES)
            if not bam and not r1:
                raise ValueError(f"Sample {sample_id!r} in {samples_csv} line {idx} has neither BAM nor FASTQ R1")
            out.append(
                SampleInput(
                    sample_id=sample_id,
                    group=group,
                    relationship=relationship,
                    family_id=family_id,
                    sex=sex,
                    condition=condition,
                    bam=bam,
                    fastq_r1=r1,
                    fastq_r2=r2,
                    extra={str(k): (v or "") for k, v in row.items()},
                )
            )
    return out


def detect_manifest_mode(samples: list[SampleInput]) -> str:
    modes = {s.input_mode for s in samples}
    if len(modes) == 1:
        return next(iter(modes))
    if modes <= {"fastq_pe", "fastq_se"}:
        return "fastq_mixed"
    return "mixed"


def validate_input_paths(samples: list[SampleInput]) -> list[str]:
    errors: list[str] = []
    for sample in samples:
        if sample.bam and not Path(sample.bam).exists():
            errors.append(f"{sample.sample_id}: BAM not found: {sample.bam}")
        if sample.fastq_r1 and not Path(sample.fastq_r1).exists():
            errors.append(f"{sample.sample_id}: FASTQ R1 not found: {sample.fastq_r1}")
        if sample.fastq_r2 and not Path(sample.fastq_r2).exists():
            errors.append(f"{sample.sample_id}: FASTQ R2 not found: {sample.fastq_r2}")
    return errors


def manifest_fieldnames(samples: list[SampleInput]) -> list[str]:
    preferred = [
        "sample_id", "family_id", "relationship", "affected", "sex", "condition",
        "replicate", "batch", "subject", "pair_id", "notes", "group", "bam", "fastq_r1", "fastq_r2",
    ]
    present: set[str] = set()
    for sample in samples:
        present.update((sample.extra or {}).keys())
        present.update(k for k, v in sample.to_row().items() if v not in (None, "", "."))
    return [name for name in preferred if name in present] + sorted(present - set(preferred))


def write_fastq_manifest(samples: list[SampleInput], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = manifest_fieldnames(samples)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            row = sample.to_row()
            writer.writerow({k: row.get(k, ".") or "." for k in fieldnames})
    return path
