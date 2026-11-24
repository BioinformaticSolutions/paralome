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
from typing import Union

from paralome.csvio import skip_comment_lines
from paralome.fastq.manifest import (
    detect_manifest_mode,
    detect_manifest_profile,
    read_sample_inputs,
    validate_input_paths,
)
from paralome.pipeline.profiles import ManifestProfile


@dataclass
class ManifestInspection:
    path: str
    samples: int
    mode: str
    manifest_profile: str
    sample_ids: list[str]
    has_bam: bool
    has_fastq: bool
    metadata_columns: list[str]
    validation_errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.validation_errors

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "samples": self.samples,
            "mode": self.mode,
            "manifest_profile": self.manifest_profile,
            "sample_ids": self.sample_ids,
            "has_bam": self.has_bam,
            "has_fastq": self.has_fastq,
            "metadata_columns": self.metadata_columns,
            "validation_errors": self.validation_errors,
            "ok": self.ok,
        }


def inspect_manifest(
    path: Union[str, Path],
    *,
    manifest_profile: str | ManifestProfile | None = None,
    validate_paths: bool = False,
    factors: tuple[str, ...] = (),
) -> ManifestInspection:
    path = Path(path)
    fieldnames, _ = read_manifest_rows(path)
    resolved_profile = ManifestProfile(manifest_profile) if manifest_profile else detect_manifest_profile(fieldnames)
    # With --factor the design is in the factor columns, and the
    # general schema then asks for those instead of `condition`.
    samples = read_sample_inputs(path, manifest_profile=resolved_profile, validate_schema=True, factors=factors)
    mode = detect_manifest_mode(samples)
    errors = validate_input_paths(samples) if validate_paths else []
    reserved = {
        "sample_id", "sample", "id", "name", "family_id", "family", "pedigree",
        "relationship", "relation", "affected", "group", "status", "sex", "gender",
        "condition", "bam", "bam_path", "alignment", "fastq_r1", "fq_r1", "r1",
        "read1", "fastq1", "fastq_r2", "fq_r2", "r2", "read2", "fastq2",
    }
    metadata_columns = [name for name in fieldnames if name.lower() not in reserved]
    return ManifestInspection(
        path=str(path),
        samples=len(samples),
        mode=mode,
        manifest_profile=resolved_profile.value,
        sample_ids=[s.sample_id for s in samples],
        has_bam=any(bool(s.bam) for s in samples),
        has_fastq=any(bool(s.fastq_r1) for s in samples),
        metadata_columns=metadata_columns,
        validation_errors=errors,
    )


def read_manifest_rows(path: Union[str, Path]) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    # Utf-8-sig drops a byte-order mark, as fastq.manifest does.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(skip_comment_lines(fh))
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]

