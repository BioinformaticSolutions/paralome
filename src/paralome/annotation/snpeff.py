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
import shlex


@dataclass(frozen=True)
class SnpEffConfig:
    jar: Path
    config: Path
    # No default. It was GRCh38.99, the human database.
    genome: str
    memory: str = "8g"
    java: str = "java"

    def validate(self) -> None:
        if not Path(self.jar).exists():
            raise FileNotFoundError(f"snpEff jar not found: {self.jar}")
        if not Path(self.config).exists():
            raise FileNotFoundError(f"snpEff config not found: {self.config}")
        if not self.genome:
            raise ValueError(
                "snpEff genome is empty: pass --snpeff-genome, or record snpeff_genome "
                "in the genome entry. Paralome never assumes a genome build."
            )

    def raw_output(self, output_vcf: Path) -> Path:
        name = str(output_vcf)
        if name.endswith(".vcf.gz"):
            return Path(name[:-3])
        if name.endswith(".gz"):
            return Path(name[:-3])
        return Path(name + ".raw.vcf")

    def command(self, input_vcf: Path, output_vcf: Path) -> str:
        raw = self.raw_output(output_vcf)
        return "\n".join(
            [
                f"{shlex.quote(self.java)} -Xmx{shlex.quote(self.memory)} \\",
                f"  -jar {shlex.quote(str(self.jar))} \\",
                f"  -c {shlex.quote(str(self.config))} \\",
                f"  {shlex.quote(self.genome)} \\",
                f"  {shlex.quote(str(input_vcf))} > {shlex.quote(str(raw))}",
                f"bgzip -f {shlex.quote(str(raw))}",
                f"mv -f {shlex.quote(str(raw) + '.gz')} {shlex.quote(str(output_vcf))}",
                f"tabix -f -p vcf {shlex.quote(str(output_vcf))}",
            ]
        )

    def stage_name(self) -> str:
        return "snpeff"
