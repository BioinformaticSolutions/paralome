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

import json
from pathlib import Path
from typing import Optional

import typer

from paralome.macs.runner import DEFAULT_MEM as MACS_DEFAULT_MEM, DEFAULT_THREADS as MACS_DEFAULT_THREADS

macs_app = typer.Typer(help="MACS3 peak-calling utilities")


@macs_app.command("check")
def macs_check():
    from paralome.macs.runner import check_macs
    if not check_macs():
        raise typer.Exit(code=1)


@macs_app.command("call")
def macs_call(
    samples: Path = typer.Option(..., "--samples", help="BAM sample manifest CSV"),
    out: Path = typer.Option(..., "--out", help="MACS3 stage work directory"),
    genome_size: str = typer.Option("auto", "--genome-size", help="MACS3 genome size, hs/mm shortcut, or auto from .fai"),
    reference: Optional[Path] = typer.Option(None, "--reference", help="Reference FASTA used when genome size is auto"),
    backend: str = typer.Option("local", "--backend"),
    submit: bool = typer.Option(False, "--submit"),
    jobs: int = typer.Option(0, "--jobs"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    mem: str = typer.Option(MACS_DEFAULT_MEM, "--mem",
                            help="SLURM memory per sample task (default %s)" % MACS_DEFAULT_MEM),
    threads: int = typer.Option(MACS_DEFAULT_THREADS, "--threads", min=1,
                                help="CPUs per sample task; they go to the duplicate-free "
                                     "samtools copy, MACS3 itself is single-threaded "
                                     "(default %d)" % MACS_DEFAULT_THREADS),
    qvalue: float = typer.Option(0.01, "--qvalue", min=0.0, max=1.0),
    keep_dup: str = typer.Option("all", "--keep-dup"),
    extra_options: Optional[str] = typer.Option(None, "--extra-macs3-options"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Call paired-end ATAC peaks independently for every sample using MACS3."""
    from paralome.atac.genome_size import resolve_macs_genome_size
    from paralome.macs.runner import run_macs_stage
    resolved_size = resolve_macs_genome_size(genome_size, reference)
    result = run_macs_stage(samples, out, resolved_size, qvalue, keep_dup, extra_options, overwrite, dry_run, backend, submit, jobs, partition, mem, threads)
    print(json.dumps(result, indent=2, sort_keys=True))


@macs_app.command("finalize")
def macs_finalize(samples: Path = typer.Option(..., "--samples"), out: Path = typer.Option(..., "--out")):
    from paralome.macs.runner import finalize_macs_stage
    print(json.dumps(finalize_macs_stage(samples, out), indent=2, sort_keys=True))
