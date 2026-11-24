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

"""The machine-facing entry point. Humans never run this.

The machine surface -- the stage drivers the supervisor invokes, and the
array-task internals written into sbatch bodies by this package's own script
writers (``motif-shard`` exists so that ``atac/motifs.py`` can put it in a job
script; nobody types it) -- lives on this separate binary.

Not hidden on the human CLI, absent from it: a hidden command is still in the
CLI, and still turns up in completion, in error messages, and in anyone's
exploration of the tool. Two binaries is the honest structure: ``paralome
--help`` cannot leak a worker command because the worker commands are not
registered on it. Git does this (``git-upload-pack``), Postgres does this
(``postgres`` vs ``pg_ctl``), Celery does this.

The split is by AUDIENCE, not by risk. Everything here is a legitimate, tested
operation; it simply has exactly one caller, and that caller is a program.

The docstring sits above the ``__future__`` import, so ``paralome.worker.__doc__``
is set.
"""

from __future__ import annotations

import typer

from paralome.errors import install_error_handler

from paralome.commands.align import align_app
from paralome.commands.annotate import annotate_app
from paralome.commands.atac import atac_app
from paralome.commands.batch import batch_worker_app
from paralome.commands.browser import browser_app
from paralome.commands.burden import burden_app
from paralome.commands.dedup import dedup_app
from paralome.commands.evidence import evidence_app
from paralome.commands.fastq import fastq_app
from paralome.commands.filter import filter_app
from paralome.commands.genome import genome_worker_app
from paralome.commands.macs import macs_app
from paralome.commands.report import report_app
from paralome.commands.rna import rna_app
from paralome.commands.segregate import segregate_app
from paralome.commands.trajectories import trajectories_app
from paralome.commands.variants import variants_app


# See cli.py. The worker's output lands in a SLURM .err file, where a
# locals dump is worse still -- it is what the user is told to go and read.
app = typer.Typer(
    pretty_exceptions_enable=False,
    help=(
        "Paralome internal worker. Invoked by the pipeline supervisor and by "
        "generated SLURM scripts. Not intended to be run by hand -- use the "
        "'paralome' command instead."
    )
)

# Stage drivers, one per pipeline stage, invoked by pipeline.orchestrator.
app.add_typer(fastq_app, name="fastq")
app.add_typer(align_app, name="align")
app.add_typer(dedup_app, name="dedup")
app.add_typer(macs_app, name="macs")
app.add_typer(variants_app, name="variants")
app.add_typer(annotate_app, name="annotate")
app.add_typer(segregate_app, name="segregate")
app.add_typer(filter_app, name="filter")
app.add_typer(report_app, name="report")
# The last stage of every expression pipeline -- Evidome, run as an external program.
app.add_typer(evidence_app, name="evidence")
# Gene burden of a proband cohort against gnomAD -- the DNA stage before Evidome.
app.add_typer(burden_app, name="burden")
# The genotype_output / statistics_output stages.
#: `output genotype` / `output statistics`.
app.add_typer(browser_app, name="output")
# The single-cell lineages and the statistics along them.
app.add_typer(trajectories_app, name="trajectories")
# The array task and finalize job of `paralome setup genome burden-reference`.
app.add_typer(genome_worker_app, name="genome")
# The --statistic panel job `paralome olink` / `paralome proteomics` submit with --backend slurm.
from paralome.commands.panels import panels_app  # noqa: E402

app.add_typer(panels_app, name="panels")

# Assay stage drivers plus every *-prepare/-sample/-block/-shard/-finalize
# array task. These groups keep their existing command names so the sbatch
# writers change only their argv[0].
app.add_typer(atac_app, name="atac")
app.add_typer(rna_app, name="rna")

# Family admission-control array task and the snapshot renderers.
#
# Only the array task of `batch` belongs here (`batch_worker_app`); `batch run` is on the human binary.
app.add_typer(batch_worker_app, name="batch")


main = install_error_handler(app)


if __name__ == "__main__":  # pragma: no cover
    main()
