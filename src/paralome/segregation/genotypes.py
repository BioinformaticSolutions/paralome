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
from typing import Iterator, Optional
import gzip


@dataclass(frozen=True)
class GenotypeCall:
    sample_id: str
    raw: str
    gt: str
    alleles: tuple[str, ...]
    is_called: bool
    has_alt: bool
    alt_count: int
    allele_count: int

    @property
    def is_hom_alt(self) -> bool:
        return self.is_called and self.allele_count > 0 and self.alt_count == self.allele_count




@dataclass(frozen=True)
class VariantRecord:
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    qual: str
    filt: str
    info: str
    fmt: str
    calls: dict[str, GenotypeCall]

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"

    def info_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if not self.info or self.info == ".":
            return out
        for item in self.info.split(";"):
            if not item:
                continue
            if "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
            else:
                out[item] = "true"
        return out


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def parse_gt(sample_id: str, fmt: str, sample_value: str) -> GenotypeCall:
    keys = fmt.split(":") if fmt else []
    values = sample_value.split(":") if sample_value else []
    try:
        gt_index = keys.index("GT")
    except ValueError:
        gt = values[0] if values else "./."
    else:
        gt = values[gt_index] if gt_index < len(values) else "./."

    alleles = tuple(gt.replace("|", "/").split("/")) if gt else tuple()
    called = bool(alleles) and all(a not in {".", ""} for a in alleles)
    alt_count = 0
    allele_count = 0
    if called:
        for allele in alleles:
            allele_count += 1
            try:
                ai = int(allele)
            except ValueError:
                continue
            if ai > 0:
                alt_count += 1
    return GenotypeCall(
        sample_id=sample_id,
        raw=sample_value,
        gt=gt,
        alleles=alleles,
        is_called=called,
        has_alt=alt_count > 0,
        alt_count=alt_count,
        allele_count=allele_count,
    )


def iter_vcf_records(path: Path) -> Iterator[VariantRecord]:
    samples: list[str] = []
    with _open_text(Path(path)) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.split("\t")
                samples = parts[9:]
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            chrom, pos_s, variant_id, ref, alt, qual, filt, info = parts[:8]
            fmt = parts[8] if len(parts) > 8 else ""
            calls: dict[str, GenotypeCall] = {}
            for sid, sval in zip(samples, parts[9:]):
                calls[sid] = parse_gt(sid, fmt, sval)
            try:
                pos = int(pos_s)
            except ValueError:
                continue
            yield VariantRecord(
                chrom=chrom,
                pos=pos,
                variant_id=variant_id,
                ref=ref,
                alt=alt,
                qual=qual,
                filt=filt,
                info=info,
                fmt=fmt,
                calls=calls,
            )


def extract_gene(record: VariantRecord, info: Optional[dict[str, str]] = None) -> str:
    # ``info`` is the record's INFO already parsed (segregation
    # parses it once per row); parsed again here only when not given.
    if info is None:
        info = record.info_dict()

    # SnpEff ANN: Allele|Annotation|Impact|Gene_Name|Gene_ID|...
    ann = info.get("ANN")
    if ann:
        genes = []
        for entry in ann.split(","):
            fields = entry.split("|")
            if len(fields) > 3 and fields[3] and fields[3] not in genes:
                genes.append(fields[3])
        if genes:
            return ",".join(genes[:5])

    # ANNOVAR usually emits INFO fields derived from multianno columns.
    for key in (
        "Gene.refGeneWithVer",
        "Gene.refGene",
        "Gene.ensGene",
        "Gene",
    ):
        val = info.get(key)
        if val and val != ".":
            return val

    return "."
