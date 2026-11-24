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

from pathlib import Path
from typing import Optional

import typer

from paralome.segregation.runner import (
    print_segregation_stage_result,
    print_segregation_verify,
    run_segregation_stage,
    verify_segregation_stage,
)

segregate_app = typer.Typer(help="Annotate variants against family segregation models")


@segregate_app.command("run")
def segregate_run_command(
    vcf: Path = typer.Option(..., "--vcf", help="Input annotated VCF/VCF.GZ"),
    samples: Path = typer.Option(..., "--samples", help="Sample manifest CSV with family/group/relationship columns"),
    out: Path = typer.Option(..., "--out", help="Segregation work directory"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Output file prefix; default is workdir name"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing segregation output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write plan/checkpoint but do not execute segregation"),
    threads: int = typer.Option(1, "--threads", help="Total local worker budget for shard execution"),
    shards: int = typer.Option(1, "--shards", help="Number of segregation shards to generate"),
    regions: Optional[Path] = typer.Option(None, "--regions", help="Optional regions.txt/BED-style region map for genomic shard assignment"),
    # See pipeline.region_shards REGION_CONVENTIONS.
    regions_convention: str = typer.Option("one-based", "--regions-convention", help="How a chr:start-end line of --regions is read: one-based (1-based, both ends included; a hand-written list) or freebayes (0-based, end-exclusive: a FreeBayes region file). BED lines are 0-based either way."),
):
    """Run segregation as a checkpointed workflow stage."""
    result = run_segregation_stage(
        vcf=vcf,
        samples=samples,
        workdir=out,
        prefix=prefix,
        overwrite=overwrite,
        dry_run=dry_run,
        threads=threads,
        shards=shards,
        regions=regions,
        region_convention=regions_convention,
    )
    print_segregation_stage_result(result)


@segregate_app.command("verify")
def segregate_verify_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="Segregation work directory"),
):
    """Verify a segregation workflow stage."""

    report = verify_segregation_stage(workdir)
    print_segregation_verify(report)


@segregate_app.command("status")
def segregate_status_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="Segregation work directory"),
):
    """Print concise segregation stage status."""
    report = verify_segregation_stage(workdir)
    print("Segregation status")
    print("==================")
    print(f"Workdir             : {workdir}")
    print(f"Complete            : {report.get('complete')}")
    print(f"Rows                : {report.get('rows')}")
    print(f"Bucket sum          : {report.get('bucket_sum')}")
    print(f"Lossless            : {report.get('lossless')}")
    out = report.get("output_tsv") or {}
    print(f"All variants TSV    : {out.get('path')}")
