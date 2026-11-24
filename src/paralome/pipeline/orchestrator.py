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

import contextlib
import csv
import io
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

from paralome.layout import extraction_dir as layout_extraction_dir
from paralome.csvio import skip_comment_lines
from paralome.binaries import worker_argv
from paralome.singlecell.rds_input import rds_argument
from paralome.checkpoints import stage_checkpoint_path
from paralome.errors import ParalomeError
from paralome.pipeline.dispatcher import DispatchPlan, build_dispatch_plan, infer_related_workdir
from paralome.pipeline.jsonio import write_json
from paralome.pipeline.manifest import read_manifest_rows
from paralome.atac.naming import derive_file_prefix
from paralome.pipeline.state import build_project_state, project_stage_root, write_project_state
from paralome.pipeline.profiles import alignment_requires_gtf, resolve_pipeline_profile
from paralome.commands.rna import _parse_threshold_list



#: Fused is the product's default, so a caller that does not say gets the fused pipeline. The CLI
#: resolves --fuse-stages/--separate-stages and defaults.fuse_stages against this same value.
DEFAULT_FUSE_STAGES = True

@dataclass
class PipelineRunResult:
    workdir: Path
    next_stage: str
    command: list[str]
    executed: bool
    returncode: Optional[int]
    dispatch_plan: dict[str, Any]
    message: str
    submitted_job_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workdir": str(self.workdir),
            "next_stage": self.next_stage,
            "command": self.command,
            "command_text": " ".join(shlex.quote(x) for x in self.command),
            "executed": self.executed,
            "returncode": self.returncode,
            "dispatch_plan": self.dispatch_plan,
            "message": self.message,
            "submitted_job_ids": list(self.submitted_job_ids),
        }


class StageCommandError(ParalomeError):
    """A stage command exited with an error (PLM-0304).

    Carries the job IDs the command reported before it failed. A
    stage that submits an array and then fails to submit its finalizer has left
    real jobs in the queue; raising a bare RuntimeError threw their IDs away,
    and the next run submitted the same work a second time on top of them. The
    supervisor records these IDs and waits on them when it is next started.
    """

    def __init__(
        self,
        returncode: int,
        command: list[str],
        *,
        workdir: Path,
        submitted_job_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            304,
            f"Pipeline stage command failed with exit code {returncode}: "
            + " ".join(shlex.quote(x) for x in command),
            returncode=returncode,
            workdir=str(workdir),
            submitted_job_ids=list(submitted_job_ids),
        )
        self.returncode = returncode
        self.command = list(command)
        self.workdir = Path(workdir)
        self.submitted_job_ids = tuple(submitted_job_ids)


_SUBMITTED_JOB_PATTERNS = (
    re.compile(r"Submitted(?:\s+[A-Za-z0-9_.-]+)*\s+job:?\s+(?P<job_ids>\d+(?:[\s,]+\d+)*)", re.IGNORECASE),
    re.compile(r"Job\s+ID\s*:\s*(?P<job_ids>\d+(?:[\s,]+\d+)*)", re.IGNORECASE),
)


def _parse_submitted_job_ids(output: str) -> tuple[str, ...]:
    """Return every scheduler job ID reported by stage submission output.

    Some stages submit an array plus a dependent merge job and report both IDs
    on one line, for example ``Job ID: 1001,1002``.  Preserve every
    reported ID in submission order so the supervisor waits for the complete
    stage chain before validating outputs.
    """
    job_ids: list[str] = []
    seen: set[str] = set()
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            structured = payload.get("submitted_job_ids")
            if isinstance(structured, list):
                for value in structured:
                    job_id = str(value).strip()
                    if job_id.isdigit() and job_id not in seen:
                        seen.add(job_id)
                        job_ids.append(job_id)
    for line in output.splitlines():
        for pattern in _SUBMITTED_JOB_PATTERNS:
            match = pattern.search(line)
            if match:
                for job_id in re.findall(r"\d+", match.group("job_ids")):
                    if job_id not in seen:
                        seen.add(job_id)
                        job_ids.append(job_id)
                break
    return tuple(job_ids)


def _parse_alpha_spec(value: str) -> tuple[str, float]:
    """Parse a combined significance-metric + cutoff spec (e.g. "Q0.05" or
    "P0.10") into (metric, alpha) -- same convention as
    ``paralome.commands.atac._parse_alpha``, duplicated here (not imported)
    to keep the orchestrator independent of the commands layer. Needed by
    the ``singlecell_rna_differential`` dispatch branch below: unlike ATAC's
    ``singlecell-differential`` (which accepts the combined "Q0.05" spec
    directly), RNA's ``singlecell-differential`` takes separate
    ``--alpha-metric``/``--alpha-value`` options.
    """
    raw = str(value).strip()
    if "0." not in raw:
        raise ValueError(f"--alpha must look like Q0.05 or P0.10, got {value!r}")
    metric_text, _, fraction = raw.partition("0.")
    metric = metric_text.strip().lower()
    if metric not in {"p", "q"}:
        raise ValueError(f"--alpha metric must be P or Q, got {metric_text!r} in {value!r}")
    alpha_value = float("0." + fraction)
    if not 0 <= alpha_value <= 1:
        raise ValueError(f"--alpha cutoff must be between 0 and 1, got {alpha_value} in {value!r}")
    return metric, alpha_value


@contextlib.contextmanager
def _pipeline_in_environment(pipeline: Optional[str]):
    """The analysis a stage runs for, in the environment its command inherits.

    Read by paralome.scratch when the stage writes its job scripts,
    so each job's scratch directory is named after the analysis
    (paralome-RNA-<user>-...). Restored afterwards, so nothing else in this
    process sees it.
    """
    from paralome.scratch import PIPELINE_ENV

    before = os.environ.get(PIPELINE_ENV)
    if pipeline:
        os.environ[PIPELINE_ENV] = str(pipeline)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(PIPELINE_ENV, None)
        else:
            os.environ[PIPELINE_ENV] = before


def _without_json_documents(text: str) -> str:
    """The stage command's output without the JSON records it prints.

    Several worker commands end by printing their checkpoint as a
    JSON document -- for the machine, and for tests that read it. Echoed by
    the supervisor it put two hundred lines of paths and fingerprints between
    the stages of every local run. A document is a line that is exactly `{`
    or `[` through the line at column 0 that closes it, and is dropped only if
    it parses; everything else is shown as it was printed. The job IDs are
    still read from the full output.
    """
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    i = 0
    while i < len(lines):
        opener = lines[i].rstrip("\r\n")
        end = None
        if opener in ("{", "["):
            closer = "}" if opener == "{" else "]"
            for j in range(i + 1, len(lines)):
                if lines[j].rstrip("\r\n") == closer:
                    try:
                        json.loads("".join(lines[i:j + 1]))
                        end = j
                    except ValueError:
                        pass
                    break
        if end is None:
            kept.append(lines[i])
            i += 1
        else:
            i = end + 1
    return "".join(kept)


def _run_stage_command(cmd: list[str], own_session: bool = False) -> tuple[int, tuple[str, ...]]:
    # A command that SUBMITS runs in a session of its own. Ctrl-C and a closed terminal are delivered to
    # every process in the foreground group, so a stage command could otherwise die between two sbatch
    # calls, with jobs in the queue and their IDs never printed. A command that runs its work here
    # (local backend, or a run without --submit) stays in the group, so stopping the run still stops
    # the tools it started.
    #
    # Decided by the caller from the run's mode, not from the word "--submit" in the command: the
    # functional-evidence command submits Evidome's jobs with `--backend slurm` and no --submit.
    completed = subprocess.run(
        cmd, check=False, text=True, capture_output=True,
        start_new_session=bool(own_session),
    )
    shown = _without_json_documents(completed.stdout or "")
    if shown.strip():
        print(shown, end="" if shown.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return completed.returncode, _parse_submitted_job_ids(combined)


#: The stages whose worker command does the work itself -- they
#: have no --backend of their own. On a run with --backend slurm --submit each
#: is submitted as ONE job running that same command, instead of running inside
#: the supervisor's process (on a cluster, the supervisor's own one-CPU job,
#: where a multi-million-variant filtering and reporting would run). Locally, and on a SLURM
#: run without --submit, they run where they are started, as before.
STAGES_SUBMITTED_AS_ONE_JOB = frozenset({"segregation", "filtering", "reporting", "gene_burden",
                                         # / formerly browser_bundle.
                                         "genotype_output", "statistics_output",
                                         # Reads every BAM or fragment file in its own job.
                                         "genome_tracks",
                                         # Slingshot and the statistics along its lineages.
                                         "trajectories"})
#: Memory for such a job when --stage-mem / stage_resources names none.
ONE_JOB_DEFAULT_MEM = "16G"
#: Reporting runs one worker process per shard in its one job, so it gets more.
#: Segregation is sized from what it uses. Measured on a 40-proband,
#: 28.3-million-variant cohort it held 218 MB resident while asking for 16G --
#: about seventy times what it needed, and on a CR_CPU_MEMORY cluster a job
#: reserves what it asks for whether it touches it or not. Sharded it runs one
#: worker per shard at roughly that size, so 8G leaves room for far more shards
#: than any run will use.
ONE_JOB_STAGE_MEM = {"reporting": "64G", "gene_burden": "64G", "segregation": "8G",
                     # DuckDB holds the whole-genome table (28 M rows for 40 genomes) while it bins it.
                     "genotype_output": "64G", "statistics_output": "64G",
                     # Every sample's 25 bp bins inside the viable peaks, and a contig of coverage at a
                     # time (its own stage again).
                     "genome_tracks": "32G",
                     # The whole extraction's counts, a column per cell per lineage it is on.
                     "trajectories": "64G"}
#: The CPUs the browser bundle's DuckDB build asks for, at most.
BROWSER_BUNDLE_MAX_THREADS = 32
#: Gene_burden reads the VCF by contig in this many processes at most (burden/stage.py).
GENE_BURDEN_MAX_THREADS = 16
#: Options of those commands that name a file or folder: written absolute in the job script.
_ONE_JOB_PATH_OPTIONS = frozenset({
    "--vcf", "--samples", "--out", "--segregation", "--regions", "--reference",
    "--candidate-gene-file",
    # The browser bundle's inputs.
    "--burden", "--filter",
})


def _option_value(cmd: list[str], flag: str) -> Optional[str]:
    tokens = [str(token) for token in cmd]
    if flag in tokens[:-1]:
        return tokens[tokens.index(flag) + 1]
    return None


def _absolute_command(cmd: list[str]) -> list[str]:
    """The command with every path option written absolute (not resolved through links)."""
    tokens = [str(token) for token in cmd]
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        out.append(token)
        if token in _ONE_JOB_PATH_OPTIONS and index + 1 < len(tokens):
            value = tokens[index + 1]
            value = os.path.abspath(value)
            out.append(value)
            index += 2
            continue
        index += 1
    return out


def write_stage_job_script(
    stage: str,
    cmd: list[str],
    workdir: Union[str, Path],
    *,
    mem: Optional[str] = None,
    partition: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> Path:
    """The sbatch script that runs ``cmd`` -- the stage command itself -- as one job.

    CPUs are the command's own --threads (1 when it takes none);
    memory is the stage's --stage-mem, else ONE_JOB_STAGE_MEM's or
    ONE_JOB_DEFAULT_MEM. Logs go to the stage's logs/, the script to its
    scripts/<stage>.sbatch.

    CPUS ARE WHAT THE COMMAND CAN USE, NOT ITS BUDGET. The report's
    --threads is the run's whole CPU budget (threads x jobs, e.g. 140)
    and it starts min(--threads, --shards) worker processes. A job asking
    for 140 CPUs on one node is refused by sbatch on a cluster whose nodes
    have fewer ("CPU count per node can not be satisfied").
    With --shards the job asks for min(--threads, --shards).
    """
    from paralome.binaries import shell_join
    from paralome.pipeline.naming import slurm_job_name
    from paralome.pipeline.slurm import SlurmOptions, write_sbatch

    workdir = Path(os.path.abspath(str(workdir)))
    threads_text = _option_value(cmd, "--threads")
    try:
        threads = max(1, int(threads_text)) if threads_text is not None else 1
    except ValueError:
        threads = 1
    shards_text = _option_value(cmd, "--shards")
    try:
        if shards_text is not None:
            threads = min(threads, max(1, int(shards_text)))
    except ValueError:
        pass
    logs = workdir / "logs"
    options = SlurmOptions(
        job_name=slurm_job_name(workdir, stage),
        cpus_per_task=threads,
        mem=str(mem or ONE_JOB_STAGE_MEM.get(stage, ONE_JOB_DEFAULT_MEM)),
        partition=partition,
        output=logs / f"{stage}_%j.out",
        error=logs / f"{stage}_%j.err",
    )
    body = (
        f"# Paralome {stage}: the pipeline's stage command, run as its own job.\n"
        # Python's output reaches the job log as it is printed. Block-
        # buffered, gene_burden's log stayed empty for the 2 h 46 min it ran.
        "export PYTHONUNBUFFERED=1\n"
        # The analysis, for the scratch directories this job's
        # command names (paralome.scratch.PIPELINE_ENV).
        + (f"export PARALOME_PIPELINE={shlex.quote(str(pipeline))}\n" if pipeline else "")
        + shell_join(_absolute_command(cmd)) + "\n"
    )
    return write_sbatch(workdir / "scripts" / f"{stage}.sbatch", options, body)


def _sbatch_stage_script(script: Path) -> tuple[int, str, str]:
    """Run ``sbatch <script>`` in a session of its own: (exit code, stdout, stderr)."""
    completed = subprocess.run(
        ["sbatch", str(script)], check=False, text=True, capture_output=True, start_new_session=True,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _submit_stage_as_one_job(
    stage: str,
    cmd: list[str],
    workdir: Union[str, Path],
    *,
    mem: Optional[str] = None,
    partition: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> tuple[int, tuple[str, ...]]:
    """Write the stage's job script, submit it, and say its job id at once."""
    from paralome.pipeline.slurm import parse_sbatch_jobid

    script = write_stage_job_script(stage, cmd, workdir, mem=mem, partition=partition, pipeline=pipeline)
    try:
        returncode, stdout, stderr = _sbatch_stage_script(script)
    except FileNotFoundError as exc:
        print(f"Could not submit {script}: {exc}", file=sys.stderr, flush=True)
        return 127, ()
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if returncode != 0:
        print(f"sbatch refused {script} (exit {returncode}).", file=sys.stderr, flush=True)
        return returncode, ()
    try:
        job_id = parse_sbatch_jobid(stdout)
    except ValueError:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
        print(f"sbatch gave no job id for {script}.", file=sys.stderr, flush=True)
        return 1, ()
    print(f"Submitted {stage} job: {job_id}", flush=True)
    print(f"  script: {script}", flush=True)
    print(f"  logs:   {Path(os.path.abspath(str(workdir))) / 'logs'}/{stage}_{job_id}.out", flush=True)
    return 0, (job_id,)


def _add_opt(cmd: list[str], flag: str, value: Union[str, Path] | int | Optional[float]) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _filter_profile_scope(profile: str) -> Optional[str]:
    """The profile's own filtering scope, or None for the filter stage's default.

    An unknown profile name is left for the filter stage to reject with its
    list of valid names, rather than failing here with a less useful message.
    """
    from paralome.filters.profiles import profile_scope

    try:
        return profile_scope(profile)
    except ValueError:
        return None


def _add_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)



def _resolve_region_map(regions: Optional[Path], variant_workdir: Optional[Union[str, Path]]) -> Optional[Path]:
    """The run's region map: the one given, else the one FreeBayes called against.

    Segregation and filtering shard by region too, and find it here.
    """
    if regions is not None:
        return regions
    if variant_workdir is None:
        return None
    for candidate in (
        Path(variant_workdir) / "regions" / "freebayes.gene_regions.txt",
        Path(variant_workdir) / "regions" / "freebayes.regions.txt",
        Path(variant_workdir) / "regions" / "gene_regions.txt",
        Path(variant_workdir) / "inputs" / "regions.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def _stage_mem(stage_memories: Mapping[str, str] | None, stage: str) -> Optional[str]:
    if not stage_memories:
        return None
    value = stage_memories.get(stage)
    return str(value) if value else None



def _rna_library_layout(align_workdir: Optional[Path], bam_sheet: Optional[Path] = None) -> str:
    """Resolve one cohort-wide RNA library layout from STAR's original manifest.

    Uses the same manifest column aliases and mode detection as the rest of
    the pipeline (``paralome.fastq.manifest``) rather than a bespoke check,
    so a manifest using ``read2``/``fq_r2``/etc. is read the same way here as
    everywhere else. There is no silent default: every branch either returns
    a definitive ``"single-end"``/``"paired-end"`` answer or raises a clear
    error, because guessing paired-end for a single-end cohort silently
    corrupts fragment assignment and FPKM values.
    """
    from paralome.fastq.manifest import detect_manifest_mode, read_sample_inputs

    if bam_sheet is not None:
        # A sheet of BAMs has no FASTQ columns to count, so the
        # layout is read from the BAMs themselves. Still no default: BAMs that
        # disagree, or cannot be read, are refused.
        from paralome.rna.preflight import _bam_looks_paired

        looks = {str(sample.bam): _bam_looks_paired(sample.bam) for sample in read_sample_inputs(bam_sheet)}
        answers = set(looks.values())
        if answers == {True}:
            return "paired-end"
        if answers == {False}:
            return "single-end"
        unclear = sorted(path for path, answer in looks.items() if answer is None)
        raise ValueError(
            "Cannot resolve RNA single-end/paired-end mode from the BAM files in "
            f"{bam_sheet}: "
            + (f"could not tell for {', '.join(unclear[:3])}" if unclear
               else "some hold paired-end reads and some single-end")
            + ". Refusing to assume paired-end."
        )
    if align_workdir is None:
        raise ValueError(
            "Cannot resolve RNA single-end/paired-end mode: no alignment workdir is known. "
            "RNA quantification requires the STAR alignment stage to have run first."
        )
    manifest = align_workdir / "inputs" / "original_samples.csv"
    if not manifest.is_file():
        raise ValueError(
            f"Cannot resolve RNA single-end/paired-end mode: original manifest not found at {manifest}. "
            "Refusing to assume paired-end."
        )
    samples = read_sample_inputs(manifest)
    if not samples:
        raise ValueError(f"Cannot resolve RNA single-end/paired-end mode: manifest has no samples: {manifest}")
    mode = detect_manifest_mode(samples)
    if mode == "fastq_pe":
        return "paired-end"
    if mode == "fastq_se":
        return "single-end"
    if mode == "fastq_mixed":
        raise ValueError(
            "RNA quantification does not support mixed single-end and paired-end samples in one cohort: "
            f"{manifest}"
        )
    raise ValueError(
        f"Cannot resolve RNA single-end/paired-end mode from manifest input mode {mode!r}: {manifest}"
    )


def _is_bam_sheet(alignment: Mapping[str, Any]) -> bool:
    """Whether the state says the sample sheet lists BAMs, so nothing was aligned here.

    The stages after alignment then read the sheet itself: there is
    no aligned manifest to hand them.
    """
    return str(alignment.get("status") or "") == "not_applicable"


def _file_path(obj: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    path = obj.get("path")
    if path in {None, "", "."}:
        return None
    return str(path)


def _filter_manifest_to_active_samples(
    bam_manifest: Path, top_level_samples: Optional[Path], out_dir: Path,
    *, write: bool = True,
) -> Path:
    """Restrict a downstream bam-augmented manifest to samples still active
    in the top-level --samples sheet.

    `differential_accessibility` reads the dedup-stage's bam-augmented
    manifest rather than the top-level --samples sheet directly (it needs a
    `bam` column the top-level sheet never has, for candidate-gene locus
    coverage). Dedup ran once for every original sample, so that manifest does
    not reflect samples later `#`-commented out of the top-level sheet to
    exclude an outlier from a differential-only rerun. If the top-level
    sheet's currently-active sample set is a subset of the bam manifest's
    samples, write a filtered copy containing only those rows and return its
    path; otherwise return the bam manifest unchanged.
    """
    return _active_sample_manifest(bam_manifest, top_level_samples, out_dir, write=write)[0]


def _manifest_rows(manifest: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(skip_comment_lines(handle))
        return list(reader.fieldnames or []), list(reader)


#: Sheet columns that name an input file. They are the earlier
#: stage's business; every other column of the sheet is the design, and the
#: sheet is what decides it.
_SHEET_FILE_COLUMN_WORDS = ("fastq", "read1", "read2", "fq_", "bam", "path", "file")


def _is_sheet_file_column(name: str) -> bool:
    """Whether a column names an input file.

    Matched against the known file-column aliases and against whole
    words of the name, not substrings, so a design column such as 'profile'
    is not taken for a file column.
    """
    from paralome.fastq.manifest import BAM_ALIASES, FASTQ_R1_ALIASES, FASTQ_R2_ALIASES

    lowered = str(name).strip().lower()
    if lowered in (*FASTQ_R1_ALIASES, *FASTQ_R2_ALIASES, *BAM_ALIASES):
        return True
    words = {word for word in re.split(r"[^a-z0-9]+", lowered) if word}
    return bool(words & ({word.strip("_") for word in _SHEET_FILE_COLUMN_WORDS} | {"fastq1", "fastq2"}))


def _sheet_id_column(fieldnames: list[str]) -> Optional[str]:
    """The sheet's sample-id column, by the aliases every sheet reader accepts, any case."""
    from paralome.fastq.manifest import SAMPLE_ALIASES

    lowered = {str(name).strip().lower(): name for name in fieldnames if isinstance(name, str)}
    return next((lowered[alias] for alias in SAMPLE_ALIASES if alias in lowered), None)


def _factor_columns(factors: Sequence[str]) -> list[str]:
    """The sheet columns the --factor values name (a malformed one is the stage's to refuse)."""
    return [str(spec).split(":", 1)[0].strip() for spec in factors or () if str(spec).split(":", 1)[0].strip()]


def _active_sample_manifest(
    bam_manifest: Path, top_level_samples: Optional[Path], out_dir: Path,
    *, write: bool = True, filename: str = "differential_samples.csv",
    factor_columns: Sequence[str] = (),
) -> tuple[Path, list[dict[str, str]]]:
    """The manifest the stage is given, and the rows it holds.

    With ``write=False`` nothing is written. The path returned is
    the one a real run writes, so a command built for a dry run, or built to
    compare against a recorded request, reads the same as the one that runs.
    The rows are returned because the file may not be there to read back.

    THE SHEET DECIDES WHO IS IN AND WHAT GROUP THEY ARE IN. The earlier
    stage's manifest (dedup, MACS) is a snapshot of the sheet as it was when
    that stage ran. So the id column is found by alias ('sample_id',
    'sample', 'id', 'name', 'Sample_ID'), any case (no id column is an error,
    not "everyone"); rows not active in the sheet are left out; and every
    design column the manifest carries (anything that does not name an input
    file) takes the sheet's value, and the group reaches 'condition' and
    'group' whichever the sheet calls it. Only when that changes something is
    a new manifest written (``filename`` under ``out_dir/inputs``): an
    unchanged project keeps the path it had, and so its recorded request.

    ``factor_columns`` (the --factor columns) are copied from the sheet even
    when the manifest does not carry them: the dedup manifest has a fixed set
    of columns, and the differential needs the groups it tests.
    """
    if not Path(bam_manifest).is_file():
        # Not written yet (a command built before the stage before it ran): nothing to restrict.
        return bam_manifest, []
    fieldnames, bam_rows = _manifest_rows(bam_manifest)
    if top_level_samples is None or not Path(top_level_samples).exists():
        return bam_manifest, bam_rows
    sheet_fields, active_rows = read_manifest_rows(Path(top_level_samples))
    id_column = _sheet_id_column(sheet_fields)
    if id_column is None:
        raise ValueError(
            f"{top_level_samples} has no sample id column (sample_id, sample, id or name), so the samples "
            "still active in it cannot be told apart from the ones taken out"
        )
    active = {
        str(row.get(id_column) or "").strip(): row for row in active_rows if str(row.get(id_column) or "").strip()
    }
    if not active:
        # A sheet with every sample '#'-commented out is refused,
        # not read as "no restriction".
        raise ValueError(
            f"{top_level_samples} has no active samples (every row is empty or '#'-commented out), so "
            f"no samples are left for this stage. Uncomment the samples to analyse."
        )
    # Only the columns the manifest already carries: a sheet column the stage never read (notes, sex)
    # is not a change to it. Matched by name in any case, as every sheet reader matches them
    # (fastq.manifest._first), so a sheet column 'Condition' reaches the manifest's 'condition'.
    sheet_lower = {n.strip().lower(): n for n in sheet_fields if isinstance(n, str)}
    id_lower = str(id_column).strip().lower()
    design = {
        name: sheet_lower[name.strip().lower()] for name in fieldnames
        if isinstance(name, str) and name.strip().lower() in sheet_lower and name.strip().lower() != id_lower
        and not _is_sheet_file_column(name)
    }
    # The stages read the group from 'condition' or 'group' first.
    # Where the manifest has either, its 'status' column is not a group (the
    # MACS manifest's holds complete/incomplete) and is never written.
    manifest_lower = {str(name).strip().lower() for name in fieldnames if isinstance(name, str)}
    group_targets = ("condition", "group") if manifest_lower & {"condition", "group"} else ("status",)
    if "status" not in group_targets:
        design = {name: sheet_name for name, sheet_name in design.items() if name.strip().lower() != "status"}
    merged_fields = list(fieldnames)
    for column in factor_columns:
        sheet_name = sheet_lower.get(str(column).strip().lower())
        if sheet_name is None or str(column) in design:
            continue
        design[str(column)] = sheet_name
        if str(column) not in merged_fields:
            merged_fields.append(str(column))
    out_rows: list[dict[str, str]] = []
    changed = False
    for row in bam_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        sheet_row = active.get(sample_id)
        if sheet_row is None:
            changed = True
            continue
        merged = dict(row)
        for name, sheet_name in design.items():
            value = str(sheet_row.get(sheet_name) or "").strip()
            if str(merged.get(name) or "").strip() != value:
                merged[name] = value
                changed = True
        # The stages read the group from 'condition', else 'group', else 'status', else 'affected'
        # (read_sample_inputs). A sheet that names it one way still updates a manifest that names it
        # another -- but only a manifest column the sheet does not carry itself: a sheet column of the
        # same name was copied above and is the sheet's own value. A '.' is no group, as there. A
        # manifest column left '.' is one its stage does not use (the family dedup manifest's
        # 'condition') and is not filled in: that would be a change to nothing the stage reads, and
        # would make an unchanged project's command differ from the one it ran.
        sheet_group = next((value for value in (
            str(sheet_row.get(sheet_lower[n]) or "").strip()
            for n in ("condition", "group", "status", "affected") if n in sheet_lower
        ) if value and value != "."), "")
        if sheet_group:
            for name in fieldnames:
                if (isinstance(name, str) and name.strip().lower() in group_targets
                        and name.strip().lower() not in sheet_lower
                        and str(row.get(name) or "").strip() not in ("", ".")
                        and str(merged.get(name) or "").strip() != sheet_group):
                    merged[name] = sheet_group
                    changed = True
        out_rows.append(merged)
    if not changed:
        return bam_manifest, bam_rows
    if not out_rows:
        raise ValueError(
            f"Restricting {bam_manifest} to the samples still active in {top_level_samples} left none -- "
            "check that the sample ids match between the two files."
        )
    filtered_path = out_dir / "inputs" / filename
    if not write:
        return filtered_path, out_rows
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=merged_fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)
    text = buffer.getvalue()
    # Rewritten only when its content changes. Each stage command is
    # built again at every start, and a rewrite of the same rows moved the
    # file's mtime -- so the motif pairs stage took the manifest for changed and
    # scanned every motif again instead of resuming from the singles hand-off.
    try:
        if filtered_path.read_text() == text:
            return filtered_path, out_rows
    except OSError:
        pass
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.write_text(text)
    return filtered_path, out_rows


def genome_burden_inputs(genome: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """(burden_reference, species) of the genome entry; (None, None) without a genome.

    The reference is a genome resource (built once by `paralome setup
    genome burden-reference NAME`); a genome without one is not an error here
    -- the gene_burden stage records itself skipped and names that command.
    """
    if not genome:
        return None, None
    from paralome.pipeline.resource_catalog import resolve_genome_resources, select_genome_config

    _name, entry = select_genome_config(genome)
    resources = resolve_genome_resources(genome=genome)
    resolved = resources.resources.get("burden_reference")
    reference = resolved.value if resolved is not None else entry.get("burden_reference")
    species = entry.get("species")
    return (str(reference) if reference else None), (str(species) if species else None)


def _trajectory_fit_dir(stage_root: Optional[Path]) -> Path:
    """Where singlecell_rna fits the lineages and the trajectories stage reads them -- the run's
    trajectories folder. <run>/trajectories/fit, whichever --trajectory-method made it (an older run's is
    <run>/trajectories/slingshot)."""
    from paralome.pipeline.state import TRAJECTORY_FIT_FOLDER

    return (Path(stage_root) / "trajectories" if stage_root else Path("paralome.trajectories")) / TRAJECTORY_FIT_FOLDER


def build_next_stage_command(
    *,
    samples: Optional[Path] = None,
    plan: Optional[DispatchPlan] = None,
    reference: Optional[Path],
    star_genome: Optional[Path],
    gtf: Optional[Path],
    fastq_qc_workdir: Optional[Path],
    align_workdir: Optional[Path],
    dedup_workdir: Optional[Path],
    variant_workdir: Optional[Path],
    annotation_workdir: Optional[Path],
    segregation_workdir: Optional[Path],
    filter_workdir: Optional[Path],
    report_workdir: Optional[Path],
    backend: str,
    submit: bool,
    overwrite: bool,
    # Fuse FASTQ QC, alignment and duplicate marking into one task
    # per sample, so the trimmed FASTQs and the aligned BAM are never written.
    # See paralome.stream. Off by default: it changes which files a run leaves
    # behind, which is a decision rather than a tuning.
    fuse_stages: bool = DEFAULT_FUSE_STAGES,
    # What each of a fused task's two read streams may buffer
    # between fastp and STAR. See paralome.buffer.
    fused_buffer: str = "8G",
    fastp_threads: int,
    star_threads: int,
    fastp_adapter_fasta: Optional[Path],
    fastp_cut_right: bool,
    fastp_cut_right_window_size: Optional[int],
    fastp_cut_right_mean_quality: Optional[int],
    fastp_length_required: Optional[int],
    fastp_extra: Optional[str],
    star_extra: Optional[str],
    aligner: str = "star",
    bwa_extra: Optional[str] = None,
    star_read_files_command: str,
    star_out_reads_unmapped: str,
    star_out_filter_multimap_nmax: int,
    star_out_filter_mismatch_nover_read_lmax: float,
    # STAR's own defaults, so a caller that predates these
    # three keeps working and gets STAR's behaviour rather than a
    # TypeError. The pipeline always passes them explicitly.
    star_out_filter_mismatch_nmax: int = 999,
    star_out_filter_score_min_over_lread: float = 0.66,
    star_out_filter_match_nmin_over_lread: float = 0.66,
    star_loose: bool = False,
    dedup_threads: int,
    freebayes_threads: int,
    report_threads: int,
    report_shards: int,
    jobs: int,
    regions: Optional[Path],
    gene_padding: Optional[int] = None,
    min_alternate_count: int = 2,
    min_alternate_fraction: float = 0.05,
    haplotype_length: int = 3,
    min_qual: float = 20.0,
    min_depth: Optional[int] = None,
    min_gq: Optional[float] = None,
    freebayes_max_region_length: Optional[int] = None,
    freebayes_stall_minutes: Optional[float] = None,
    annotation: str,
    profile: str,
    segregation_profile: str | None = None,
    genome: Optional[str],
    annovar_dir: Optional[Path],
    annovar_db: Optional[Path],
    annovar_buildver: Optional[str],
    annovar_protocol: Optional[str],
    annovar_operation: Optional[str],
    snpeff_jar: Optional[Path],
    snpeff_config: Optional[Path],
    snpeff_genome: Optional[str],
    partition: Optional[str],
    stage_memories: Mapping[str, str] | None = None,
    mem: Optional[str] = None,
    mark_duplicates: bool,
    remove_duplicates: bool,
    alignment_profile: str = "dna",
    atac_threads: Optional[int] = None,
    motif_scanner_backend: str = "auto",
    alpha: Optional[str] = None,
    rna_statistic: str = "t",
    rna_factors: tuple[str, ...] = (),
    # Paired-t's pairing column and lm's covariate.
    rna_pair_by: Optional[str] = None,
    rna_covariate: Optional[str] = None,
    rna_covariate_feature: Optional[str] = None,
    rna_strandedness: int = 0,
    rna_pseudocount: float = 1.0,
    rna_block_size: int = 5000,
    rna_min_expression: str = "5.0",
    rna_min_expression_difference: str = "5.0",
    rna_refine_responders: Optional[int] = None,
    rds: Optional[Path] = None,
    replicate_column: Optional[str] = None,
    max_concurrent_strata: int = 0,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    stratify_skip: Optional[str] = None,
    min_cells_per_stratum: int = 0,
    rscript: Optional[str] = None,
    data_workdir_path: Optional[Path] = None,
    singlecell_differential_workdir: Optional[Path] = None,
    singlecell_motif_workdir: Optional[Path] = None,
    rna_assay: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
    singlecell_rna_workdir: Optional[Path] = None,
    singlecell_rna_differential_workdir: Optional[Path] = None,
    remove_samples: tuple[str, ...] = (),
    trajectory_root: Optional[str] = None,
    trajectory_bins: int = 5,
    trajectory_fit_cells: Optional[str] = None,
    trajectory_dims: Optional[int] = None,
    trajectory_method: Optional[str] = None,
    trajectory_weld: Optional[List[str]] = None,
    trajectory_max_routes: Optional[int] = None,
    trajectory_model: Optional[str] = None,
    trajectory_lineages: Optional[str] = None,
    state: Optional[dict[str, Any]] = None,
    write_inputs: bool = True,
    ancestry: Optional[str] = None,
) -> tuple[Path, list[str], str, str]:
    """Return workdir, command, message, selected stage.

    ``write_inputs=False`` builds the same command and writes no
    file a stage reads: for a dry run, and for a command built only to be
    compared with the one a stage was last given.
    """
    selected_stage = plan.next_stage if plan is not None else None
    if state and isinstance(state.get("pipeline"), dict):
        # A state that says the pipeline is finished decides too. The
        # dispatch plan only names the stage after alignment; it cannot see
        # that the stages after that one are done.
        selected_stage = state["pipeline"].get("next_stage") or (
            "complete" if state["pipeline"].get("complete") else selected_stage
        )
    if mem is not None and not stage_memories:
        # For direct callers; the pipeline CLI has no pipeline-wide --mem option.
        stage_memories = {selected_stage: mem}
    # Where the RNA stage folders go -- the state reader's rule.
    stage_root = project_stage_root(dedup_workdir, align_workdir)

    if selected_stage == "fastq_qc":
        out = fastq_qc_workdir or infer_related_workdir(samples, "fastq_qc")
        cmd = [*worker_argv(), "fastq", "qc", "--samples", str(samples), "--out", str(out), "--backend", backend, "--threads", str(fastp_threads), "--jobs", str(jobs)]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "fastq_qc"))
        # Fused: the trimming happens in the alignment task, so this stage
        # writes its manifest (naming the RAW reads) and nothing else.
        _add_bool(cmd, "--defer-trimming", fuse_stages)
        _add_opt(cmd, "--adapter-fasta", fastp_adapter_fasta)
        # BOTH forms, because the default is True: _add_bool emits nothing when the value is false, so an
        # opt-out must be written explicitly or the stage command would default it back on.
        cmd.append("--cut-right" if fastp_cut_right else "--no-cut-right")
        _add_opt(cmd, "--cut-right-window-size", fastp_cut_right_window_size)
        _add_opt(cmd, "--cut-right-mean-quality", fastp_cut_right_mean_quality)
        _add_opt(cmd, "--length-required", fastp_length_required)
        _add_opt(cmd, "--extra-fastp-options", fastp_extra)
        return out, cmd, "Run FASTQ QC/trimming. Re-run the same paralome command to resume after completion.", selected_stage

    if selected_stage == "alignment":
        from paralome.align.bwa import normalize_aligner

        use_bwa = normalize_aligner(aligner) == "bwa"
        if use_bwa and reference is None:
            raise ValueError("--aligner bwa aligns against the genome's FASTA: give --reference or --genome")
        if not use_bwa and star_genome is None:
            raise ValueError("--star-genome is required when the next stage is alignment")
        out = align_workdir or infer_related_workdir(samples, "star_align")
        # ALWAYS THE FASTQ MANIFEST, never the dispatch plan's resolved manifest (which resolves to the
        # aligned BAM manifest as soon as an alignment is on disk, and would hand an alignment that has to
        # run again its own output). The trimmed manifest of a finished read QC, else the sheet.
        fastq_state = state.get("fastq_qc", {}) if state else {}
        input_manifest = _file_path(fastq_state.get("trimmed_manifest")) if fastq_state.get("complete") else None
        if not input_manifest:
            if plan is not None and getattr(plan, "resolved_manifest_role", "") != "aligned_bam_manifest":
                input_manifest = plan.resolved_manifest
            else:
                input_manifest = samples
        cmd = [
            *worker_argv(), "align", "bwa" if use_bwa else "star",
            "--samples", str(input_manifest),
            "--reference", str(reference if use_bwa else star_genome),
            "--out", str(out),
            "--backend", backend,
            "--threads", str(star_threads),
            "--jobs", str(jobs),
        ]
        if alignment_requires_gtf(alignment_profile) and gtf is None:
            raise ValueError("RNA alignment requires a configured GTF; pass --gtf or configure it for --genome")
        from paralome.align.star import UNSPLICED_MATES_GAP
        from paralome.pipeline.profiles import PipelineProfile, normalize_pipeline_profile

        mates_gap = None if alignment_requires_gtf(alignment_profile) else UNSPLICED_MATES_GAP[
            "atac" if normalize_pipeline_profile(profile) is PipelineProfile.ATAC else "dna"]
        if not use_bwa:
            if alignment_requires_gtf(alignment_profile):
                _add_opt(cmd, "--gtf", gtf)
            else:
                # DNA and ATAC: STAR in DNA mode (no splicing, no GTF). RNA keeps its parameters.
                _add_opt(cmd, "--unspliced-mates-gap", mates_gap)
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "alignment"))
        if use_bwa:
            _add_opt(cmd, "--bwa-extra", bwa_extra)
            # Bwa-mem2's pairs filtered to what STAR, on this run's settings, would have written
            # (align.bwa.FILTER_AWK).
            _add_opt(cmd, "--mismatch-fraction", star_out_filter_mismatch_nover_read_lmax)
            _add_opt(cmd, "--match-fraction", star_out_filter_match_nmin_over_lread)
            _add_opt(cmd, "--multimap-nmax", star_out_filter_multimap_nmax)
            _add_opt(cmd, "--mates-gap", mates_gap)
        else:
            _add_opt(cmd, "--star-read-files-command", star_read_files_command)
            _add_opt(cmd, "--star-out-reads-unmapped", star_out_reads_unmapped)
            _add_opt(cmd, "--star-out-filter-multimap-nmax", star_out_filter_multimap_nmax)
            _add_opt(cmd, "--star-out-filter-mismatch-nover-read-lmax", star_out_filter_mismatch_nover_read_lmax)
            _add_opt(cmd, "--star-out-filter-mismatch-nmax", star_out_filter_mismatch_nmax)
            _add_opt(cmd, "--star-out-filter-score-min-over-lread", star_out_filter_score_min_over_lread)
            _add_opt(cmd, "--star-out-filter-match-nmin-over-lread", star_out_filter_match_nmin_over_lread)
            _add_bool(cmd, "--loose", star_loose)
            _add_opt(cmd, "--star-extra", star_extra)
        if fuse_stages:
            # fastp runs in this task, streaming into STAR through FIFOs; its
            # reports go to the FASTQ QC stage's own directories, where every
            # reader of them already looks.
            _add_opt(cmd, "--trim-in-task",
                     str(fastq_qc_workdir or infer_related_workdir(samples, "fastq_qc")))
            # The run's fastp settings reach the fused task too (it trimmed with the defaults).
            _add_opt(cmd, "--trim-adapter-fasta", fastp_adapter_fasta)
            cmd.append("--trim-cut-right" if fastp_cut_right else "--no-trim-cut-right")
            _add_opt(cmd, "--trim-cut-right-window-size", fastp_cut_right_window_size)
            _add_opt(cmd, "--trim-cut-right-mean-quality", fastp_cut_right_mean_quality)
            _add_opt(cmd, "--trim-length-required", fastp_length_required)
            _add_opt(cmd, "--trim-extra", fastp_extra)
            _add_opt(cmd, "--fused-buffer", fused_buffer)
            # STAR's unsorted output goes straight into duplicate marking, so no aligned BAM is written and
            # STAR does not sort at all. Where the marked BAM goes depends on whether this assay HAS a
            # deduplication stage to own it: DNA and ATAC do, RNA does not. Asked of the STAGE GRAPH, which
            # is the actual question (not, say, whether the alignment needs a GTF).
            own_dedup_stage = "deduplication" in resolve_pipeline_profile(
                profile, segregation_profile
            ).stage_graph
            if not own_dedup_stage:
                _add_bool(cmd, "--mark-in-place", True)
            elif mark_duplicates:
                _add_opt(cmd, "--fuse-dedup",
                         str(dedup_workdir or infer_related_workdir(samples, "dedup")))
                # The overlap of a pair's mates soft-clipped after duplicate marking.
                _add_bool(cmd, "--clip-overlap", resolve_pipeline_profile(profile, segregation_profile).is_variant)
        return out, cmd, ("Run STAR alignment, trimming and duplicate marking in one task per sample."
                          if fuse_stages else
                          "Run STAR alignment. Re-run the same paralome command to resume after completion."), selected_stage


    if selected_stage == "rna_quantification":
        if gtf is None:
            raise ValueError("--gtf is required for RNA FPKM quantification")
        alignment = state.get("alignment", {}) if state else {}
        bam_sheet = samples if _is_bam_sheet(alignment) else None
        aligned_manifest = bam_sheet or _file_path(alignment.get("aligned_manifest")) or plan.resolved_manifest
        out = (stage_root / "rna_quantification") if stage_root else Path("paralome.rna_quantification")
        cmd = [
            *worker_argv(), "rna", "quantify-fpkm",
            "--samples", str(aligned_manifest),
            "--gtf", str(gtf),
            "--out", str(out),
            "--threads", str(star_threads),
            "--jobs", str(jobs),
            "--strandedness", str(rna_strandedness),
            "--pseudocount", str(rna_pseudocount),
            "--backend", backend,
        ]
        cmd.append("--single-end" if _rna_library_layout(align_workdir, bam_sheet) == "single-end" else "--paired-end")
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "rna_quantification"))
        return out, cmd, "Run checkpointed annotation-aware native FPKM quantification.", selected_stage

    if selected_stage == "differential_expression":
        quant_state = state.get("rna_quantification", {}) if state else {}
        expression = _file_path(quant_state.get("gene_log2_fpkm"))
        if not expression:
            raise ValueError("Could not resolve RNA log2-FPKM expression matrix")
        out = (stage_root / "differential_expression") if stage_root else Path("paralome.differential_expression")
        # One value each (the command line refuses a list before anything runs).
        rna_min_expression_values = _parse_threshold_list(str(rna_min_expression), option_name="--rna-min-expression")
        rna_min_expression_difference_values = _parse_threshold_list(str(rna_min_expression_difference), option_name="--rna-min-expression-difference")
        single_run_alpha = alpha.split(",")[0].strip() if alpha else alpha
        cmd = [
            *worker_argv(), "rna", "differential",
            "--expression", expression,
            "--samples", str(samples),
            "--out", str(out),
            "--statistic", rna_statistic,
            "--block-size", str(rna_block_size),
            "--threads", str(max(1, atac_threads or 1)),
            "--jobs", str(jobs),
            "--backend", backend,
            "--pseudocount", str(rna_pseudocount),
            "--min-expression", str(rna_min_expression_values[0]),
            "--min-expression-difference", str(rna_min_expression_difference_values[0]),
        ]
        for factor_name in rna_factors:
            _add_opt(cmd, "--factor", factor_name)
        # Paired-t's pairing column and lm's covariate, relayed
        # wherever --factor is. A statistic the stage can compute but was
        # never told the column for is a refusal the user cannot act on.
        _add_opt(cmd, "--pair-by", rna_pair_by)
        _add_opt(cmd, "--covariate", rna_covariate)
        _add_opt(cmd, "--covariate-feature", rna_covariate_feature)
        _add_opt(cmd, "--refine-responders", rna_refine_responders)
        _add_opt(cmd, "--alpha", single_run_alpha)
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "differential_expression"))
        return out, cmd, "Run checkpointed FPKM-based DEG statistics and reporting.", selected_stage

    if selected_stage == "deduplication":
        if not mark_duplicates:
            raise ValueError("Duplicate marking is required for this workflow stage and cannot be switched off for this profile.")
        out = dedup_workdir or infer_related_workdir(samples, "dedup")
        # ADOPT ONLY WHAT THE FUSED ALIGNMENT ACTUALLY WROTE. A sheet of BAMs skips alignment entirely, so
        # there is nothing fused to adopt and the duplicates have never been marked; adopting there would
        # write a plan and a summary over BAMs nothing had marked.
        aligned_here = not _is_bam_sheet(state.get("alignment", {}) if state else {})
        if fuse_stages and aligned_here:
            # The marking already happened, inside the alignment task. This
            # stage writes its plan, summary, checkpoint and manifests from the
            # BAMs that are there; nothing is re-marked, and a missing BAM is
            # reported as an incomplete stage rather than quietly redone.
            fastq_out = fastq_qc_workdir or infer_related_workdir(samples, "fastq_qc")
            cmd = [
                *worker_argv(), "dedup", "adopt",
                "--workdir", str(out),
                "--samples", str(samples),
                "--alignment-workdir", str(align_workdir or infer_related_workdir(samples, "star_align")),
                "--fastq-qc-workdir", str(fastq_out),
            ]
            _add_bool(cmd, "--remove-duplicates", remove_duplicates)
            return out, cmd, ("Adopt the duplicate-marked BAMs the fused alignment task wrote, "
                              "and summarize FASTQ QC from its fastp reports."), selected_stage
        alignment = state.get("alignment", {}) if state else {}
        aligned_manifest = (
            (samples if _is_bam_sheet(alignment) else None)
            or _file_path(alignment.get("aligned_manifest"))
            or plan.resolved_manifest
        )
        cmd = [
            *worker_argv(), "dedup", "mark",
            "--samples", str(aligned_manifest),
            "--out", str(out),
            "--backend", backend,
            "--threads", str(dedup_threads),
            "--jobs", str(jobs),
            "--manifest-profile", resolve_pipeline_profile(profile, segregation_profile).manifest_profile.value,
        ]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_bool(cmd, "--remove-duplicates", remove_duplicates)
        # The overlap of a pair's mates soft-clipped after duplicate marking.
        _add_bool(cmd, "--clip-overlap", resolve_pipeline_profile(profile, segregation_profile).is_variant)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "deduplication"))
        return out, cmd, "Run duplicate marking with samtools markdup.", selected_stage

    if selected_stage == "macs_peak_calling":
        if reference is None:
            raise ValueError("--reference is required to derive MACS genome size")
        dedup = state.get("deduplication", {}) if state else {}
        sample_manifest = _file_path(dedup.get("dedup_manifest")) or plan.resolved_manifest
        out = (dedup_workdir.parent / "macs") if dedup_workdir else Path("paralome.macs")
        cmd = [*worker_argv(), "macs", "call", "--samples", str(sample_manifest), "--out", str(out), "--genome-size", "auto", "--reference", str(reference), "--backend", backend, "--jobs", str(jobs)]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "macs_peak_calling"))
        return out, cmd, "Run per-sample MACS3 peak calling.", selected_stage

    if selected_stage == "peak_merge":
        macs_state = state.get("macs_peak_calling", {}) if state else {}
        peak_manifest = _file_path(macs_state.get("peak_manifest"))
        if not peak_manifest:
            raise ValueError("Could not resolve MACS peak manifest")
        out = (dedup_workdir.parent / "merged_peaks") if dedup_workdir else Path("paralome.merged_peaks")
        # Only the samples still active in the sheet. A '#'-commented
        # sample's peaks were merged into the atlas (MACS keeps its per-sample
        # peaks; they are simply not used).
        peak_manifest = str(_active_sample_manifest(
            Path(peak_manifest), Path(samples) if samples else None, out,
            write=write_inputs, filename="peak_merge_samples.csv",
        )[0])
        cmd = [
            *worker_argv(), "atac", "merge-peaks",
            "--peak-manifest", peak_manifest,
            "--out", str(out),
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
        ]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "peak_merge"))
        return out, cmd, "Run checkpointed sample-parallel peak merge and its cohort finalizer.", selected_stage

    if selected_stage == "peak_quantification":
        if gtf is None:
            raise ValueError("--gtf is required for ATAC peak annotation")
        merge_state = state.get("peak_merge", {}) if state else {}
        atlas = _file_path(merge_state.get("cohort_peaks"))
        dedup = state.get("deduplication", {}) if state else {}
        bam_manifest = _file_path(dedup.get("dedup_manifest"))
        if not atlas or not bam_manifest:
            raise ValueError("Could not resolve cohort peak atlas and deduplicated BAM manifest")
        out = (dedup_workdir.parent / "peak_quantification") if dedup_workdir else Path("paralome.peak_quantification")
        # Only the samples still active in the sheet, with its groups.
        bam_manifest = str(_active_sample_manifest(
            Path(bam_manifest), Path(samples) if samples else None, out,
            write=write_inputs, filename="quantification_samples.csv",
        )[0])
        cmd = [
            *worker_argv(), "atac", "quantify-peaks",
            "--atlas", atlas,
            "--samples", bam_manifest,
            "--gtf", str(gtf),
            "--out", str(out),
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
        ]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "peak_quantification"))
        return out, cmd, "Run checkpointed sample-parallel peak quantification and deterministic matrix finalization.", selected_stage

    if selected_stage == "differential_accessibility":
        quant_state = state.get("peak_quantification", {}) if state else {}
        cpm = _file_path(quant_state.get("cpm"))
        if not cpm:
            raise ValueError("Could not resolve peak CPM matrix")
        dedup = state.get("deduplication", {}) if state else {}
        bam_manifest = _file_path(dedup.get("dedup_manifest"))
        if not bam_manifest:
            raise ValueError("Could not resolve deduplicated BAM manifest for differential accessibility")
        # `differential_accessibility` must pass the dedup-stage's bam-augmented manifest
        # (sample_id/condition/.../bam), the same one `peak_quantification` uses above -- not the
        # pipeline-level `--samples` sheet, which has no `bam` column, so every candidate-gene locus view
        # would have no coverage to draw.
        #
        # Dedup ran once for all original samples, so its manifest does not reflect samples
        # `#`-commented out of the top-level sheet later to exclude an outlier from a differential-only
        # rerun -- restrict it to whichever samples are still active in the top-level sheet.
        differential_out = (dedup_workdir.parent / "differential_accessibility") if dedup_workdir else Path("paralome.differential_accessibility")
        bam_manifest, manifest_rows = _active_sample_manifest(
            Path(bam_manifest), Path(samples) if samples else None, differential_out,
            write=write_inputs, factor_columns=_factor_columns(rna_factors),
        )
        bam_manifest = str(bam_manifest)
        out = differential_out
        cmd = [
            *worker_argv(), "atac", "differential",
            "--cpm", cpm,
            "--samples", str(bam_manifest),
            "--out", str(out),
            # Required by the stage, so always relayed.
            "--statistic", rna_statistic,
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
        ]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "differential_accessibility"))
        # The same --statistic/--factor bulk RNA takes.
        # The option is one option for every pipeline that tests two groups --
        # the whole point of the shared stats module -- so the pipeline passes
        # it here as it passes it to the RNA differential.
        for factor_name in rna_factors:
            cmd.extend(["--factor", str(factor_name)])
        # Paired-t's pairing column and lm's covariate, relayed
        # wherever --factor is. A statistic the stage can compute but was
        # never told the column for is a refusal the user cannot act on.
        _add_opt(cmd, "--pair-by", rna_pair_by)
        _add_opt(cmd, "--covariate", rna_covariate)
        _add_opt(cmd, "--covariate-feature", rna_covariate_feature)
        _add_opt(cmd, "--alpha", alpha)
        return out, cmd, "Run checkpointed peak-block differential statistics and global correction/reporting.", selected_stage

    # The ATAC stages are treated as the other pipelines' are, and each split is available to
    # --stop-at: motif_analysis is two real stages. --skip-motif-pairs is not forwarded into the CLI
    # command: state.py's compute_pipeline_state marks motif_analysis_pairs "optional" and skips
    # selecting it when project.skip_motif_pairs is set, the same way --skip-motif works for the whole
    # motif section.
    if selected_stage in ("motif_analysis_singles", "motif_analysis_pairs"):
        quant_state = state.get("peak_quantification", {}) if state else {}
        diff_state = state.get("differential_accessibility", {}) if state else {}
        cpm = _file_path(quant_state.get("cpm"))
        # The motif stage reads the differential's checkpoint and
        # runs one enrichment per statistic folder and comparison.
        differential_checkpoint = _file_path(diff_state.get("checkpoint"))
        if not cpm or not differential_checkpoint or not diff_state.get("statistics"):
            raise ValueError("Could not resolve peak CPM matrix and differential statistics for motif analysis")
        # Both stages share one --out workdir -- that's where --stage
        # singles leaves the inputs/motif_singles_state.pkl checkpoint that
        # --stage pairs then resumes from.
        out = (dedup_workdir.parent / "motif_analysis") if dedup_workdir else Path("paralome.motif_analysis")
        # The motif stages get the same manifest as the
        # differential (sample_id, condition, bam; active samples only), not
        # the raw sheet: the sheet has no bam column, so candidate-gene views
        # drew flat-zero coverage, and an id alias or BOM in it stopped the
        # stage.
        dedup = state.get("deduplication", {}) if state else {}
        dedup_manifest = _file_path(dedup.get("dedup_manifest"))
        if not dedup_manifest:
            raise ValueError("Could not resolve deduplicated BAM manifest for motif analysis")
        motif_manifest, _manifest_rows = _active_sample_manifest(
            Path(dedup_manifest), Path(samples) if samples else None, out,
            write=write_inputs, filename="motif_samples.csv", factor_columns=_factor_columns(rna_factors),
        )
        run_stage = "singles" if selected_stage == "motif_analysis_singles" else "pairs"
        cmd = [
            *worker_argv(), "atac", "motifs",
            "--cpm", cpm,
            "--samples", str(motif_manifest),
            "--out", str(out),
            "--differential-checkpoint", differential_checkpoint,
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
            "--scanner-backend", motif_scanner_backend,
            "--stage", run_stage,
        ]
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--reference", reference)
        _add_opt(cmd, "--alpha", alpha)
        _add_bool(cmd, "--submit", submit)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, selected_stage))
        message = (
            "Run checkpointed motif preparation, parallel scanning, and single-motif outputs "
            "(stops before pair co-occurrence)."
            if run_stage == "singles"
            else "Run motif-pair co-occurrence analysis from the completed singles checkpoint."
        )
        return out, cmd, message, selected_stage

    if selected_stage == "singlecell_atac":
        if rds is None:
            raise ValueError("--rds is required for snATAC (single-cell ATAC) profiles")
        if not replicate_column:
            raise ValueError("--replicate-column is required for snATAC (single-cell ATAC) profiles")
        # scRNA and snATAC take the same parameters, with the same defaults and the same optional ones;
        # the groups are --factor columns of the pseudobulk manifest. Peak annotation is the bulk ATAC
        # one, and like every pipeline this one asks for --genome: `atac singlecell-samples` resolves its
        # own GTF from --genome via the genome catalog motif analysis uses (resolve_genome_resources --
        # see singlecell.py's run_singlecell_atac), so this dispatch forwards the --genome name, as the
        # motif and annotation stages below do. Not required here (same as those stages) -- a run without
        # --genome gets bare, unannotated peaks.
        out = data_workdir_path or Path("paralome.singlecell_atac")
        cmd = [
            *worker_argv(), "atac", "singlecell-samples",
            "--rds", rds_argument(rds),
            "--out", str(out),
            "--replicate-column", replicate_column,
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
        ]
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--gtf", gtf)
        _add_opt(cmd, "--stratify-by", stratify_by)
        _add_opt(cmd, "--rscript", rscript)
        for sample_id in remove_samples:
            cmd += ["--remove", sample_id]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "singlecell_atac"))
        # A folder of objects is extracted --jobs objects at a time.
        _add_opt(cmd, "--jobs", jobs)
        return out, cmd, "Aggregate the Seurat/Signac single-cell ATAC object into bulk-style pseudobulk matrices.", selected_stage

    if selected_stage == "singlecell_differential":
        # `atac singlecell-differential`: every --statistic at the pseudobulk
        # level, or with :cell the cells as N (--replicate-column names the
        # cell-metadata column whose value is a cell's sample).
        pb_state = state.get("singlecell_atac", {}) if state else {}
        cpm = _file_path(pb_state.get("cpm"))
        manifest = _file_path(pb_state.get("sample_manifest"))
        if not cpm or not manifest:
            raise ValueError("Could not resolve single-cell pseudobulk CPM matrix/sample manifest for differential accessibility")
        pb_workdir = pb_state.get("workdir") or (str(data_workdir_path) if data_workdir_path else None)
        if not pb_workdir:
            raise ValueError("Could not resolve single-cell shared data workdir (needed for the Seurat extraction directory)")
        extraction_dir = layout_extraction_dir(pb_workdir)
        # --stratify-by is optional here, matching scRNA (singlecell_stats guards every use of it).
        # --stratify-by always fans out, on all pipelines: omitting --stratify-value makes the
        # differential stage run every stratum into its own subdirectory; passing it narrows to one,
        # which is what an array task does.
        out = singlecell_differential_workdir or Path("paralome.singlecell_differential")
        cmd = [
            *worker_argv(), "atac", "singlecell-differential",
            "--cpm", cpm,
            "--samples", manifest,
            "--extraction", str(extraction_dir),
            "--out", str(out),
            "--statistic", rna_statistic,
            # The ANOVA's factors, relayed to the single-cell
            # differential exactly as they are to the two bulk ones. Without
            # these two lines --statistic anova reached the stage with nothing
            # to test and the portal could not run a factorial design.
            *[token for name in rna_factors for token in ("--factor", str(name))],
            *(["--pair-by", str(rna_pair_by)] if rna_pair_by else []),
            *(["--covariate", str(rna_covariate)] if rna_covariate else []),
            *(["--covariate-feature", str(rna_covariate_feature)] if rna_covariate_feature else []),
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
        ]
        # The stratum array's %K throttle. Absent means all at once.
        _add_opt(cmd, "--max-concurrent-strata", max_concurrent_strata or None)
        # Relayed CONDITIONALLY, as it is optional, mirroring the scRNA branch: in the literal list it
        # would render as the four-character string "None" in the generated sbatch.
        if stratify_by:
            _add_opt(cmd, "--stratify-by", stratify_by)
            _add_opt(cmd, "--stratify-value", stratify_value)
            _add_opt(cmd, "--stratify-skip", stratify_skip)
        _add_opt(cmd, "--min-cells-per-stratum", min_cells_per_stratum or None)
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "singlecell_differential"))
        _add_opt(cmd, "--alpha", alpha)
        return out, cmd, "Run checkpointed single-cell differential-accessibility statistics (one folder per --statistic).", selected_stage

    if selected_stage in ("singlecell_motif_analysis_singles", "singlecell_motif_analysis_pairs"):
        # The motif stage reads the differential's CHECKPOINT and runs one enrichment per stratum, per
        # statistic folder, per comparison. Nothing here picks a table.
        pb_state = state.get("singlecell_atac", {}) if state else {}
        diff_state = state.get("singlecell_differential", {}) if state else {}
        cpm = _file_path(pb_state.get("cpm"))
        manifest = _file_path(pb_state.get("sample_manifest"))
        if not cpm or not manifest:
            raise ValueError("Could not resolve single-cell pseudobulk CPM matrix/sample manifest for motif analysis")
        differential_checkpoint = _file_path(diff_state.get("checkpoint")) or (
            str(stage_checkpoint_path(Path(singlecell_differential_workdir), "singlecell_differential"))
            if singlecell_differential_workdir else None
        )
        if not differential_checkpoint or (state and not diff_state.get("statistics")):
            raise ValueError(
                "Could not resolve the single-cell differential's results for motif analysis: its "
                "checkpoint records no statistics folders (re-run singlecell_differential)"
            )
        out = singlecell_motif_workdir or Path("paralome.singlecell_motif_analysis")
        run_stage = "singles" if selected_stage == "singlecell_motif_analysis_singles" else "pairs"
        cmd = [
            *worker_argv(), "atac", "motifs",
            "--cpm", cpm,
            "--samples", manifest,
            "--out", str(out),
            "--differential-checkpoint", differential_checkpoint,
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
            "--scanner-backend", motif_scanner_backend,
            "--stage", run_stage,
        ]
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--reference", reference)
        _add_opt(cmd, "--alpha", alpha)
        _add_bool(cmd, "--submit", submit)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, selected_stage))
        message = (
            "Run checkpointed single-cell motif preparation, parallel scanning, and single-motif "
            "outputs (stops before pair co-occurrence)."
            if run_stage == "singles"
            else "Run single-cell motif-pair co-occurrence analysis from the completed singles checkpoint."
        )
        return out, cmd, message, selected_stage

    if selected_stage == "singlecell_rna":
        # snRNA: hybrid of snATAC up through Diffs -- mirrors the
        # singlecell_atac branch above, driving `paralome rna
        # singlecell-samples` instead. The RNA and the ATAC can come from
        # the same export: when both profiles point --rds/
        # --out at the same combined Seurat object and data folder,
        # whichever stage runs first performs the one shared
        # singlecell_extraction/, the other reuses it via checkpoint.
        if rds is None:
            raise ValueError("--rds is required for scRNA (single-cell RNA) profiles")
        if not replicate_column:
            raise ValueError("--replicate-column is required for scRNA (single-cell RNA) profiles")
        out = singlecell_rna_workdir or data_workdir_path or Path("paralome.singlecell_rna")
        cmd = [
            *worker_argv(), "rna", "singlecell-samples",
            "--rds", rds_argument(rds),
            "--out", str(out),
            "--replicate-column", replicate_column,
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
        ]
        _add_opt(cmd, "--rna-assay", rna_assay)
        _add_opt(cmd, "--gene-symbol-column", gene_symbol_column)
        # The stage REQUIRES a genome, so it is relayed here: without it `paralome scRNA` would build a
        # command its own sub-command rejects.
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--gtf", gtf)
        _add_opt(cmd, "--stratify-by", stratify_by)
        _add_opt(cmd, "--rscript", rscript)
        for sample_id in remove_samples:
            cmd += ["--remove", sample_id]
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "singlecell_rna"))
        # A folder of objects is extracted --jobs objects at a time.
        _add_opt(cmd, "--jobs", jobs)
        # The trajectories run as a subfunction of the extraction, serially after it: the lineages are
        # fitted in this stage's job, after the extraction and the pseudobulk, into the run's trajectories
        # folder, and written into the run's cell export; the trajectories stage reads the fit.
        # --trajectory-fit-cells changed refits only the fit. The extraction writes the object's cell
        # export into the run's omicview folder only, not into <parent>/omicview as well.
        from paralome.pipeline.state import omicview_dir

        exports = omicview_dir(stage_root, data_workdir_path or out, True)
        _add_opt(cmd, "--cell-export-dir", str(exports) if exports else None)
        if trajectory_root:
            cmd += ["--trajectory-root", str(trajectory_root), "--trajectory-fit-dir", str(_trajectory_fit_dir(stage_root))]
            _add_opt(cmd, "--trajectory-fit-cells", trajectory_fit_cells)
            _add_opt(cmd, "--trajectory-method", trajectory_method)
            for weld in trajectory_weld or ():
                _add_opt(cmd, "--trajectory-weld", weld)
            _add_opt(cmd, "--trajectory-max-routes",
                     str(int(trajectory_max_routes)) if trajectory_max_routes is not None else None)
            _add_opt(cmd, "--trajectory-dims", str(int(trajectory_dims)) if trajectory_dims is not None else None)
            _add_opt(cmd, "--trajectory-export-dir", str(exports) if exports else None)
        return out, cmd, "Aggregate the Seurat single-cell RNA object into bulk-style pseudobulk matrices.", selected_stage

    if selected_stage == "singlecell_rna_differential":
        pb_state = state.get("singlecell_rna", {}) if state else {}
        cpm = _file_path(pb_state.get("cpm"))
        manifest = _file_path(pb_state.get("sample_manifest"))
        if not cpm or not manifest:
            raise ValueError("Could not resolve single-cell RNA pseudobulk CPM matrix/sample manifest for differential expression")
        pb_workdir = pb_state.get("workdir") or (
            str(singlecell_rna_workdir or data_workdir_path) if (singlecell_rna_workdir or data_workdir_path) else None
        )
        if not pb_workdir:
            raise ValueError("Could not resolve single-cell RNA shared data workdir (needed for the Seurat extraction directory)")
        extraction_dir = layout_extraction_dir(pb_workdir)
        out = singlecell_rna_differential_workdir or Path("paralome.singlecell_rna_differential")
        cmd = [
            *worker_argv(), "rna", "singlecell-differential",
            "--cpm", cpm,
            "--samples", manifest,
            "--extraction", str(extraction_dir),
            "--out", str(out),
            "--statistic", rna_statistic,
            # The ANOVA's factors, relayed to the single-cell
            # differential exactly as they are to the two bulk ones. Without
            # these two lines --statistic anova reached the stage with nothing
            # to test and the portal could not run a factorial design.
            *[token for name in rna_factors for token in ("--factor", str(name))],
            *(["--pair-by", str(rna_pair_by)] if rna_pair_by else []),
            *(["--covariate", str(rna_covariate)] if rna_covariate else []),
            *(["--covariate-feature", str(rna_covariate_feature)] if rna_covariate_feature else []),
            "--backend", backend,
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            "--jobs", str(jobs),
        ]
        # The stratum array's %K throttle. Absent means all at once.
        _add_opt(cmd, "--max-concurrent-strata", max_concurrent_strata or None)
        if stratify_by:
            _add_opt(cmd, "--stratify-by", stratify_by)
            _add_opt(cmd, "--stratify-value", stratify_value)
            _add_opt(cmd, "--stratify-skip", stratify_skip)
            # This line was also emitted unconditionally just below,
            # so a stratified run carried --min-cells-per-stratum twice. Typer
            # takes the last, so it was harmless -- but it is the tell that an
            # anchored insert landed in two places, and the duplicate is the
            # kind of thing that stops being harmless the moment the flag gains
            # a list type or an accumulate action.
            _add_opt(cmd, "--min-cells-per-stratum", min_cells_per_stratum or None)
        _add_bool(cmd, "--submit", submit)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "singlecell_rna_differential"))
        # ScRNA takes --alpha as snATAC does (the stage refuses the --alpha-metric/--alpha-value pair).
        _add_opt(cmd, "--alpha", alpha)
        return out, cmd, "Run checkpointed single-cell RNA differential-expression statistics (one folder per --statistic).", selected_stage

    if selected_stage == "variants":
        if reference is None:
            raise ValueError("--reference is required when the next stage is variants")
        out = variant_workdir or Path("paralome.variant_run")
        dedup = state.get("deduplication", {}) if state else {}
        sample_manifest = _file_path(dedup.get("dedup_manifest")) or plan.resolved_manifest
        if _file_path(dedup.get("dedup_manifest")):
            # Joint calling of the samples still active in the sheet.
            # A '#'-commented sample was still called with the others (its BAM
            # stays in the dedup manifest); FreeBayes checks itself, so the
            # changed BAM list calls again.
            sample_manifest = str(_active_sample_manifest(
                Path(sample_manifest), Path(samples) if samples else None, Path(out),
                write=write_inputs, filename="variant_samples.csv",
            )[0])
        cmd = [
            *worker_argv(), "variants", "freebayes",
            "--samples", str(sample_manifest),
            "--reference", str(reference),
            "--out", str(out),
            "--backend", backend,
            "--threads", str(freebayes_threads),
            # --jobs is how many SAMPLES run at
            # once. Calling is split by genome region, not by sample, so it is
            # not passed here; the stage sizes its own pieces from --threads.
            "--annotation", "none",
            "--profile", "none",
        ]
        _add_opt(cmd, "--regions", regions)
        _add_opt(cmd, "--gtf", gtf)
        # Genes can be padded wide to take in putative regulatory
        # variants. Gene-scheduled calling only; a
        # --regions file is used as given.
        if regions is None:
            _add_opt(cmd, "--gene-padding", gene_padding)
        # The calling thresholds the user set, written out; no mapping- or base-quality cut
        # (one would throw out every read STAR places in two equally good places).
        _add_opt(cmd, "--min-alternate-count", min_alternate_count)
        _add_opt(cmd, "--min-alternate-fraction", f"{float(min_alternate_fraction):g}")
        _add_opt(cmd, "--haplotype-length", haplotype_length)
        _add_opt(cmd, "--min-qual", f"{float(min_qual):g}")
        _add_opt(cmd, "--min-mapping-quality", 0)
        _add_opt(cmd, "--min-base-quality", 0)
        # The region cap and the stall watchdog, when the run or its
        # configuration sets them. How the calls are run, not what is called:
        # left out of the stage's request record (_REQUEST_IGNORED_BY_STAGE).
        _add_opt(cmd, "--max-region-length", freebayes_max_region_length)
        if freebayes_stall_minutes is not None:
            _add_opt(cmd, "--stall-minutes", f"{float(freebayes_stall_minutes):g}")
        _add_bool(cmd, "--submit", submit)
        _add_opt(cmd, "--partition", partition)
        # The configured memory for variant calling (called variant_calling in the configuration, while
        # the stage is called variants; both names are read).
        _add_opt(
            cmd, "--mem",
            _stage_mem(stage_memories, "variants")
            or _stage_mem(stage_memories, "variant_calling")
            or mem,
        )
        return out, cmd, "Run FreeBayes variant calling from duplicate-marked BAM manifest.", selected_stage

    if selected_stage == "annotation":
        variants = state.get("variants", {}) if state else {}
        input_vcf = _file_path(variants.get("final_output"))
        if not input_vcf:
            raise ValueError("Could not resolve final variant VCF for annotation")
        engine = (annotation or "annovar").lower().strip()
        out = annotation_workdir or Path("paralome.annotation")
        variant_source_workdir = variant_workdir or Path(str(variants.get("workdir"))) if variants.get("workdir") else variant_workdir
        cmd = [
            *worker_argv(), "annotate",
            "--engine", engine,
            "--vcf", input_vcf,
            "--out", str(out),
            "--backend", backend,
            # Was str(jobs) -- --jobs 8 gave annotation 8 CPUs and
            # --jobs 0 gave it ONE, every shard queued behind the last.
            "--threads", str(max(1, int(freebayes_threads or 1))),
        ]
        _add_opt(cmd, "--variant-workdir", variant_source_workdir)
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--annovar-buildver", annovar_buildver)
        _add_opt(cmd, "--annovar-dir", annovar_dir)
        _add_opt(cmd, "--annovar-db", annovar_db)
        _add_opt(cmd, "--annovar-protocol", annovar_protocol)
        _add_opt(cmd, "--annovar-operation", annovar_operation)
        _add_opt(cmd, "--snpeff-jar", snpeff_jar)
        _add_opt(cmd, "--snpeff-config", snpeff_config)
        _add_opt(cmd, "--snpeff-genome", snpeff_genome)
        _add_bool(cmd, "--submit", submit)
        _add_opt(cmd, "--partition", partition)
        _add_opt(cmd, "--mem", _stage_mem(stage_memories, "annotation"))
        return out, cmd, f"Run {engine} annotation engine chain on variant shards.", selected_stage

    def _project_prefix(stage_out) -> str:
        """The family/project ID, resolved HERE and passed to the worker.

        Every one of these commands accepts a prefix flag. Whatever the
        child's environment does to relative-path resolution, the supervisor
        already knows which project this is, so deriving it once here and
        passing it explicitly removes the child's environment from the question
        entirely -- and an explicit --prefix takes precedence over the one each
        worker would derive from its own --out.
        """
        return derive_file_prefix(Path(stage_out).resolve())

    if selected_stage == "gene_burden":
        # The probands against gnomAD -- see paralome.burden.stage.
        annotation_state = state.get("annotation", {}) if state else {}
        input_vcf = _file_path(annotation_state.get("output_vcf"))
        if not input_vcf:
            raise ValueError("Could not resolve annotated VCF for gene burden")
        out = (stage_root / "gene_burden") if stage_root else Path("paralome.gene_burden")
        burden_reference, species = genome_burden_inputs(genome)
        cmd = [
            *worker_argv(), "burden", "run",
            "--vcf", input_vcf,
            "--samples", str(samples),
            "--out", str(out),
            "--prefix", _project_prefix(out),
            "--ancestry", (ancestry or "auto"),
        ]
        # Without a reference the stage records itself skipped and names the
        # command that builds one; --genome is passed so it can name it.
        _add_opt(cmd, "--reference", burden_reference)
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--species", species)
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--alpha", alpha.split(",")[0].strip() if alpha else None)
        # The scan reads contigs in parallel; the job asks for these CPUs.
        _add_opt(cmd, "--threads", str(min(GENE_BURDEN_MAX_THREADS, max(1, int(freebayes_threads or 1)))))
        return out, cmd, "Run gene burden of the probands against gnomAD.", selected_stage

    def _zips(stage_name: str) -> Optional[str]:
        # Where this stage's zips go -- the run's omicview folder (paralome.pipeline.state.
        # omicview_dir), a subfolder per stage.
        from paralome.pipeline.state import omicview_dir

        single = bool(profile) and resolve_pipeline_profile(profile, segregation_profile).pipeline_profile.value in ("snATAC", "scRNA")
        pb = ((state.get("singlecell_atac") or {}).get("workdir") or (state.get("singlecell_rna") or {}).get("workdir")) if state else None
        root = omicview_dir(stage_root, data_workdir_path or pb, single)
        return str(root / stage_name) if root else None

    def _former_folder() -> Optional[str]:
        # The stage's folder under its retired name, renamed in place by the stage.
        return str(stage_root / "browser_bundle") if stage_root else None

    def _genotype_output_command():
        # The variant browser's bundle -- see paralome.browser.stage. The genotype_output stage.
        segregation_state = state.get("segregation", {}) if state else {}
        input_tsv = _file_path(segregation_state.get("segregation_tsv"))
        if not input_tsv:
            raise ValueError("Could not resolve the segregation table for the genotype output")
        out = (stage_root / "genotype_output") if stage_root else Path("paralome.genotype_output")
        cmd = [
            *worker_argv(), "output", "genotype",
            "--segregation", input_tsv,
            "--out", str(out),
            "--prefix", _project_prefix(out),
        ]
        _add_opt(cmd, "--zip-dir", _zips("genotype_output"))
        _add_opt(cmd, "--former-folder", _former_folder())
        _add_opt(cmd, "--qc-root", str(stage_root) if stage_root else None)
        burden_dir = ((stage_root / "gene_burden") if stage_root else Path("paralome.gene_burden")) / "results"
        if burden_dir.is_dir():
            _add_opt(cmd, "--burden", str(burden_dir))
        filter_dir = Path(filter_workdir or "paralome.filter") / "results"
        if filter_dir.is_dir():
            _add_opt(cmd, "--filter", str(filter_dir))
        if segregation_profile:
            from paralome.filters.profiles import normalize_profile_name

            _add_opt(cmd, "--filter-profile", normalize_profile_name(segregation_profile))
        _add_bool(cmd, "--overwrite", overwrite)
        _add_opt(cmd, "--threads", str(min(BROWSER_BUNDLE_MAX_THREADS, max(1, int(freebayes_threads or 1)))))
        return out, cmd, "Build the run's genotype output for OmicView's variant browser.", selected_stage

    if selected_stage == "genotype_output":
        return _genotype_output_command()

    if selected_stage == "trajectories":
        # The object's lineages (Slingshot, the whole object) and the run's own statistics
        # along and between them -- see paralome.singlecell.trajectories. Reads the extraction the run's pseudobulk
        # made and that pseudobulk's manifest; the same replicate column, --remove, genome and statistic.
        if not trajectory_root:
            from paralome.errors import ParalomeError

            raise ParalomeError(121, "The trajectories stage needs --trajectory-root <column>:<value>, the cluster "
                                     "the lineages start from (or --no-trajectories to leave the stage out)")
        pb_state = state.get("singlecell_rna", {}) if state else {}
        manifest = _file_path(pb_state.get("sample_manifest"))
        pb_workdir = pb_state.get("workdir") or (
            str(singlecell_rna_workdir or data_workdir_path) if (singlecell_rna_workdir or data_workdir_path) else None
        )
        if not manifest or not pb_workdir:
            raise ValueError("Could not resolve the single-cell RNA pseudobulk (its sample manifest and extraction) "
                             "for the trajectories")
        out = (stage_root / "trajectories") if stage_root else Path("paralome.trajectories")
        cmd = [
            *worker_argv(), "trajectories", "run",
            "--extraction", str(layout_extraction_dir(pb_workdir)),
            "--samples", manifest,
            "--out", str(out),
            "--prefix", _project_prefix(out),
            "--modality", "scRNA",
            "--root", str(trajectory_root),
            "--bins", str(int(trajectory_bins)),
            *(["--fit-cells", str(trajectory_fit_cells)] if trajectory_fit_cells else []),
            *(["--method", str(trajectory_method)] if trajectory_method else []),
            *(["--dims", str(int(trajectory_dims))] if trajectory_dims is not None else []),
            # The gene model along pseudotime, and the lineages the statistics follow.
            *(["--model", str(trajectory_model)] if trajectory_model else []),
            *(["--lineages", str(trajectory_lineages)] if trajectory_lineages else []),
            "--replicate-column", str(replicate_column),
            "--statistic", rna_statistic,
            *[token for name in rna_factors for token in ("--factor", str(name))],
            *(["--pair-by", str(rna_pair_by)] if rna_pair_by else []),
            *(["--covariate", str(rna_covariate)] if rna_covariate else []),
            *(["--covariate-feature", str(rna_covariate_feature)] if rna_covariate_feature else []),
            "--threads", str(atac_threads if atac_threads is not None else dedup_threads),
            # The fit singlecell_rna made.
            "--fit", str(_trajectory_fit_dir(stage_root)),
        ]
        for sample_id in remove_samples:
            cmd += ["--remove", sample_id]
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--gtf", gtf)
        _add_opt(cmd, "--alpha", alpha)
        _add_opt(cmd, "--rscript", rscript)
        _add_opt(cmd, "--zip-dir", _zips("trajectories"))
        _add_bool(cmd, "--overwrite", overwrite)
        return out, cmd, "Run the statistics along and between the object's lineages.", selected_stage

    if selected_stage == "statistics_output":
        # Every arm's export for OmicView. DNA's is the variant browser's bundle
        # (paralome.browser.stage, <prefix>_genotype_output.zip); every other arm's is
        # paralome.browser.arm, its statistics (<prefix>_statistics_output.zip). The statistics_output stage.
        arm = resolve_pipeline_profile(profile, segregation_profile).pipeline_profile.value
        out = (stage_root / "statistics_output") if stage_root else Path("paralome.statistics_output")
        cmd = [*worker_argv(), "output", "statistics", "--arm", arm, "--out", str(out), "--prefix", _project_prefix(out)]
        _add_opt(cmd, "--zip-dir", _zips("statistics_output"))
        _add_opt(cmd, "--former-folder", _former_folder())
        if arm == "RNA":
            _add_opt(cmd, "--results", str((stage_root or Path(".")) / "differential_expression" / "results"))
            _add_opt(cmd, "--samples-table", str(samples) if samples else None)
            # From the FASTQ: the QC-to-alignment summaries go in too.
            _add_opt(cmd, "--qc-root", str(stage_root) if stage_root else None)
        elif arm == "scRNA":
            pb_state = state.get("singlecell_rna", {}) if state else {}
            _add_opt(cmd, "--results", str(singlecell_rna_differential_workdir) if singlecell_rna_differential_workdir else None)
            _add_opt(cmd, "--samples-table", _file_path(pb_state.get("sample_manifest")))
        elif arm == "snATAC":
            _add_opt(cmd, "--results", str(singlecell_differential_workdir) if singlecell_differential_workdir else None)
            if singlecell_motif_workdir:
                _add_opt(cmd, "--motifs", str(Path(singlecell_motif_workdir) / "results"))
            _add_opt(cmd, "--samples-table", _file_path((state.get("singlecell_atac", {}) if state else {}).get("sample_manifest")))
        elif arm == "ATAC":
            root = stage_root or Path(".")
            _add_opt(cmd, "--results", str(root / "differential_accessibility" / "results"))
            _add_opt(cmd, "--motifs", str(root / "motif_analysis" / "results"))
            _add_opt(cmd, "--samples-table", str(samples) if samples else None)
            _add_opt(cmd, "--qc-root", str(stage_root) if stage_root else None)
        _add_bool(cmd, "--overwrite", overwrite)
        threads_for = atac_threads if atac_threads is not None else dedup_threads
        _add_opt(cmd, "--threads", str(max(1, int(threads_for or 1))))
        # <prefix>_statistics_output.zip, the statistics only; the ATAC tracks are the
        # genome_tracks stage's again.
        return out, cmd, "Build the statistics output for OmicView.", selected_stage

    if selected_stage == "genome_tracks":
        # The genome browser's tracks -- see paralome.atac.genome_tracks. Bulk ATAC reads the deduplicated
        # BAMs; snATAC the fragment files its Signac object names. Writes <prefix>_genome_tracks.zip; tracks
        # an older browser bundle holds are taken over (--bundled-tracks).
        tracks_graph = resolve_pipeline_profile(profile, segregation_profile).stage_graph if profile else ()
        if "variants" in tracks_graph:
            # DNA: the reads FreeBayes used at every called site, <prefix>_DNA_reads.zip -- paralome.variants.genome_tracks.
            variants_state = state.get("variants", {}) if state else {}
            vcf = _file_path(variants_state.get("final_output"))
            dedup = state.get("deduplication", {}) if state else {}
            dedup_manifest = _file_path(dedup.get("dedup_manifest"))
            if not vcf or not dedup_manifest:
                raise ValueError("Could not resolve the variant calling's VCF and the BAM manifest for the reads at the sites")
            out = (stage_root / "genome_tracks") if stage_root else Path("paralome.genome_tracks")
            cmd = [
                *worker_argv(), "output", "dna-genome-tracks",
                "--vcf", vcf,
                "--samples", dedup_manifest,
                "--out", str(out),
                "--prefix", _project_prefix(out),
                "--threads", str(max(1, int(dedup_threads or 1))),
            ]
            _add_opt(cmd, "--genome", genome)
            _add_opt(cmd, "--reference", reference)
            _add_opt(cmd, "--gtf", gtf)
            _add_opt(cmd, "--zip-dir", _zips("genome_tracks"))
            _add_bool(cmd, "--overwrite", overwrite)
            return out, cmd, "Slice the reads FreeBayes used at every called site, for the variant panel.", selected_stage
        # The stage folder that may hold an older layout's tracks: statistics_output, else the former
        # browser_bundle folder.
        bundled = None
        if stage_root:
            bundled = stage_root / "statistics_output"
            if not bundled.is_dir() and (stage_root / "browser_bundle").is_dir():
                bundled = stage_root / "browser_bundle"
        if "singlecell_differential" in tracks_graph:
            pb_state = state.get("singlecell_atac", {}) if state else {}
            diff_state = state.get("singlecell_differential", {}) if state else {}
            manifest = _file_path(pb_state.get("sample_manifest"))
            differential_checkpoint = _file_path(diff_state.get("checkpoint")) or (
                str(stage_checkpoint_path(Path(singlecell_differential_workdir), "singlecell_differential"))
                if singlecell_differential_workdir else None
            )
            pb_workdir = pb_state.get("workdir") or (str(data_workdir_path) if data_workdir_path else None)
            if not manifest or not differential_checkpoint or not pb_workdir:
                raise ValueError("Could not resolve the pseudobulk manifest, extraction and differential for the genome tracks")
            if rds is None or not replicate_column:
                raise ValueError("--rds and --replicate-column are required for the snATAC genome tracks")
            # <project>/genome_tracks, as bulk ATAC's, beside browser_bundle and
            # functional_evidence (it was <project>/atac/genome_tracks, whose tracks are moved here: --former-tracks).
            out = (stage_root / "genome_tracks") if stage_root else Path("paralome.genome_tracks")
            cmd = [
                *worker_argv(), "atac", "singlecell-genome-tracks",
                "--differential-checkpoint", differential_checkpoint,
                "--samples", manifest,
                "--extraction", str(layout_extraction_dir(pb_workdir)),
                "--rds", rds_argument(rds),
                "--replicate-column", replicate_column,
                "--out", str(out),
                "--prefix", _project_prefix(out),
            ]
            _add_opt(cmd, "--stratify-by", stratify_by)
            _add_opt(cmd, "--genome", genome)
            _add_opt(cmd, "--gtf", gtf)
            _add_opt(cmd, "--reference", reference)
            _add_opt(cmd, "--rscript", rscript)
            _add_opt(cmd, "--bundled-tracks", str(bundled) if bundled else None)
            if singlecell_differential_workdir:
                _add_opt(cmd, "--former-tracks", str(Path(singlecell_differential_workdir) / "genome_tracks"))
            _add_opt(cmd, "--zip-dir", _zips("genome_tracks"))
            _add_bool(cmd, "--overwrite", overwrite)
            return out, cmd, "Write the genome browser's tracks: the viable peaks and one profile per pseudobulk sample.", selected_stage
        quant_state = state.get("peak_quantification", {}) if state else {}
        diff_state = state.get("differential_accessibility", {}) if state else {}
        cpm = _file_path(quant_state.get("cpm"))
        differential_checkpoint = _file_path(diff_state.get("checkpoint"))
        dedup = state.get("deduplication", {}) if state else {}
        dedup_manifest = _file_path(dedup.get("dedup_manifest"))
        if not cpm or not differential_checkpoint or not dedup_manifest:
            raise ValueError("Could not resolve the peak quantification, differential and BAM manifest for the genome tracks")
        from paralome.atac.statistics import _resolve_coverage_summary

        coverage_summary = _resolve_coverage_summary(Path(cpm))
        if coverage_summary is None:
            raise ValueError("Could not resolve peak quantification's summary (each sample's eligible fragments)")
        out = (stage_root / "genome_tracks") if stage_root else Path("paralome.genome_tracks")
        tracks_manifest, _rows = _active_sample_manifest(
            Path(dedup_manifest), Path(samples) if samples else None, out,
            write=write_inputs, filename="genome_tracks_samples.csv", factor_columns=_factor_columns(rna_factors),
        )
        cmd = [
            *worker_argv(), "atac", "genome-tracks",
            "--differential-checkpoint", differential_checkpoint,
            "--samples", str(tracks_manifest),
            "--coverage-summary", str(coverage_summary),
            "--out", str(out),
            "--prefix", _project_prefix(out),
            "--threads", str(max(1, int((atac_threads if atac_threads is not None else dedup_threads) or 1))),
        ]
        _add_opt(cmd, "--genome", genome)
        _add_opt(cmd, "--gtf", gtf)
        _add_opt(cmd, "--bundled-tracks", str(bundled) if bundled else None)
        _add_opt(cmd, "--zip-dir", _zips("genome_tracks"))
        _add_bool(cmd, "--overwrite", overwrite)
        return out, cmd, "Write the genome browser's tracks: the viable peaks and one profile per sample.", selected_stage

    if selected_stage == "functional_evidence":
        # Evidome on the differential statistics -- see paralome.evidence.
        #
        # Every pipeline reaches this stage, and each one hands
        # Evidome its OWN differential table. Which key holds it is the only
        # thing that differs, so the stage is resolved by asking the stages
        # that could have produced one, in graph order, rather than by a
        # chain of profile tests -- a profile added later gets this for free
        # as soon as its differential records a table.
        evidence_graph = (
            resolve_pipeline_profile(profile, segregation_profile).stage_graph
            if profile else ()
        )
        # A differential stage hands over its CHECKPOINT, and Evidome runs once per statistic folder (per
        # stratum) it records: each statistic gets its own end result. DNA's gene burden is one table.
        source_option = source = None
        tableless = None    # a differential that ran but wrote no Evidome table
        for stage_name, key, option in (
            ("gene_burden", "evidome_input", "--statistics"),                    # DNA
            ("differential_expression", "checkpoint", "--differential-checkpoint"),  # bulk RNA
            ("differential_accessibility", "checkpoint", "--differential-checkpoint"),  # bulk ATAC
            ("singlecell_differential", "checkpoint", "--differential-checkpoint"),     # snATAC
            ("singlecell_rna_differential", "checkpoint", "--differential-checkpoint"),  # scRNA
        ):
            if evidence_graph and stage_name not in evidence_graph:
                continue
            stage_state = state.get(stage_name, {}) if state else {}
            if option == "--differential-checkpoint" and not stage_state.get("evidome_inputs"):
                # snATAC without a peak annotation has no gene to hand Evidome.
                # Its checkpoint still goes to the stage, which records the
                # skip and why, rather than this stage failing to resolve.
                if tableless is None and _file_path(stage_state.get(key)):
                    tableless = _file_path(stage_state.get(key))
                continue
            source = _file_path(stage_state.get(key))
            if source:
                source_option = option
                break
        if not source and tableless:
            source, source_option = tableless, "--differential-checkpoint"
        if not source:
            raise ValueError(
                "Could not resolve the differential statistics for functional evidence from "
                f"this project's stages ({', '.join(evidence_graph)}). Evidome reads the tables "
                "its own pipeline's differential wrote."
            )
        out = (stage_root / "functional_evidence") if stage_root else Path("paralome.functional_evidence")
        # The modality after the prefix, so every arm's Evidome output keeps its own name:
        # <prefix>_<arm>[_<stratum>]_<statistic>_evidome_output.zip.
        evidence_prefix = f"{_project_prefix(out)}_{resolve_pipeline_profile(profile, segregation_profile).pipeline_profile.value}"
        cmd = [
            *worker_argv(), "evidence", "run",
            str(source_option), str(source),
            "--out", str(out),
            "--prefix", evidence_prefix,
            # Evidome submits its own chain only when this run actually submits jobs.
            "--backend", "slurm" if backend == "slurm" and submit else "local",
        ]
        _add_opt(cmd, "--partition", partition)
        # Each finished archive is moved into the run's omicview folder.
        _add_opt(cmd, "--zip-dir", _zips("functional_evidence"))
        # The run's --alpha (its first, when several are given): the p Evidome selects genes by.
        _add_opt(cmd, "--alpha", alpha.split(",")[0].strip() if alpha else None)
        # And the trajectories' statistics in every segment, one Evidome run each.
        trajectories_state = state.get("trajectories", {}) if state else {}
        if "trajectories" in evidence_graph and trajectories_state.get("expression_checkpoint"):
            _add_opt(cmd, "--trajectory-checkpoint", str(trajectories_state["expression_checkpoint"]))
        return out, cmd, "Run Evidome: pathway, gene-set and upstream-regulator evidence.", selected_stage

    if selected_stage == "segregation":
        annotation_state = state.get("annotation", {}) if state else {}
        input_vcf = _file_path(annotation_state.get("output_vcf"))
        if not input_vcf:
            raise ValueError("Could not resolve annotated VCF for segregation")
        out = segregation_workdir or Path("paralome.segregation")
        # SEGREGATION TAKES --threads AND --shards and splits the VCF by region (segregation/runner.py), so
        # both are passed: the same budget numbers reporting is given -- the run's, not a stage's -- and
        # write_stage_job_script asks SLURM for min(threads, shards) CPUs on the strength of them.
        cmd = [
            *worker_argv(), "segregate", "run",
            "--vcf", input_vcf, "--samples", str(samples), "--out", str(out),
            "--threads", str(max(1, report_threads)),
            "--shards", str(max(1, report_shards)),
        ]
        # Sharding by the regions FreeBayes called against keeps shard
        # boundaries on the boundaries the variants were produced with, so no
        # shard splits a locus that was called as one.
        region_map = _resolve_region_map(regions, variant_workdir)
        _add_opt(cmd, "--regions", region_map)
        if region_map is not None:
            _add_opt(cmd, "--regions-convention", "freebayes")
        _add_opt(cmd, "--prefix", _project_prefix(out))
        _add_bool(cmd, "--overwrite", overwrite)
        return out, cmd, "Run family segregation analysis.", selected_stage

    if selected_stage == "filtering":
        segregation_state = state.get("segregation", {}) if state else {}
        input_tsv = _file_path(segregation_state.get("segregation_tsv"))
        if not input_tsv:
            raise ValueError("Could not resolve segregation TSV for filtering")
        out = filter_workdir or Path("paralome.filter")
        filter_profile = segregation_profile or "family-segregating"
        # As with segregation, filtering shards and was never told to.
        cmd = [
            *worker_argv(), "filter", "run",
            "--segregation", input_tsv, "--out", str(out), "--profile", filter_profile,
            "--threads", str(max(1, report_threads)),
            "--shards", str(max(1, report_shards)),
        ]
        _add_opt(cmd, "--scope", _filter_profile_scope(filter_profile))
        _add_opt(cmd, "--prefix", _project_prefix(out))
        # The profile's quality rules follow the run's own settings.
        _add_opt(cmd, "--min-alternate-count", min_alternate_count)
        _add_opt(cmd, "--min-alternate-fraction", f"{float(min_alternate_fraction):g}")
        _add_opt(cmd, "--min-qual", f"{float(min_qual):g}")
        _add_opt(cmd, "--min-depth", min_depth)
        _add_opt(cmd, "--min-gq", min_gq)
        _add_bool(cmd, "--overwrite", overwrite)
        return out, cmd, "Run final variant filtering.", selected_stage

    out = (
        report_workdir or filter_workdir or segregation_workdir or annotation_workdir
        or variant_workdir or dedup_workdir or align_workdir or fastq_qc_workdir
        or singlecell_motif_workdir or singlecell_differential_workdir or data_workdir_path
        or singlecell_rna_differential_workdir or singlecell_rna_workdir
        or (samples.parent if samples else Path("."))
    )
    return Path(out), [], "Pipeline appears complete or no runnable next stage was identified.", selected_stage or "complete"


def run_pipeline_next_stage(
    *,
    fuse_stages: bool = DEFAULT_FUSE_STAGES,
    # What each of a fused task's two read streams may buffer
    # between fastp and STAR. See paralome.buffer.
    fused_buffer: str = "8G",
    samples: Optional[Union[str, Path]] = None,
    reference: str | Optional[Path] = None,
    star_genome: str | Optional[Path] = None,
    gtf: str | Optional[Path] = None,
    fastq_qc_workdir: str | Optional[Path] = None,
    align_workdir: str | Optional[Path] = None,
    dedup_workdir: str | Optional[Path] = None,
    variant_workdir: str | Optional[Path] = None,
    annotation_workdir: str | Optional[Path] = None,
    segregation_workdir: str | Optional[Path] = None,
    filter_workdir: str | Optional[Path] = None,
    report_workdir: str | Optional[Path] = None,
    backend: str = "slurm",
    submit: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    fastp_threads: int = 4,
    star_threads: int = 16,
    fastp_adapter_fasta: str | Optional[Path] = None,
    fastp_cut_right: bool = True,  # Fastq.fastp.DEFAULT_CUT_RIGHT, as the CLI passes
    fastp_cut_right_window_size: Optional[int] = None,
    fastp_cut_right_mean_quality: Optional[int] = None,
    fastp_length_required: Optional[int] = None,
    fastp_extra: Optional[str] = None,
    star_extra: Optional[str] = None,
    aligner: str = "star",
    bwa_extra: Optional[str] = None,
    star_read_files_command: str = "zcat",
    star_out_reads_unmapped: str = "None",
    star_out_filter_multimap_nmax: int = 2,
    star_out_filter_mismatch_nover_read_lmax: float = 0.04,
    star_out_filter_mismatch_nmax: int = 999,
    star_out_filter_score_min_over_lread: float = 0.66,
    star_out_filter_match_nmin_over_lread: float = 0.66,
    star_loose: bool = False,
    dedup_threads: int = 8,
    atac_threads: int = 8,
    motif_scanner_backend: str = "auto",
    alpha: Optional[str] = None,
    rna_statistic: str = "t",
    rna_factors: tuple[str, ...] = (),
    # Paired-t's pairing column and lm's covariate.
    rna_pair_by: Optional[str] = None,
    rna_covariate: Optional[str] = None,
    rna_covariate_feature: Optional[str] = None,
    rna_strandedness: int = 0,
    rna_pseudocount: float = 1.0,
    rna_block_size: int = 5000,
    rna_min_expression: str = "5.0",
    rna_min_expression_difference: str = "5.0",
    rna_refine_responders: Optional[int] = None,
    freebayes_threads: int = 4,
    report_threads: int = 4,
    report_shards: int = 1,
    jobs: int = 0,
    regions: str | Optional[Path] = None,
    gene_padding: Optional[int] = None,
    min_alternate_count: int = 2,
    min_alternate_fraction: float = 0.05,
    haplotype_length: int = 3,
    min_qual: float = 20.0,
    min_depth: Optional[int] = None,
    min_gq: Optional[float] = None,
    freebayes_max_region_length: Optional[int] = None,
    freebayes_stall_minutes: Optional[float] = None,
    annotation: str = "annovar",
    profile: str = "DNA",
    segregation_profile: Optional[str] = None,
    genome: Optional[str] = None,
    annovar_dir: str | Optional[Path] = None,
    annovar_db: str | Optional[Path] = None,
    annovar_buildver: Optional[str] = None,
    annovar_protocol: Optional[str] = None,
    annovar_operation: Optional[str] = None,
    snpeff_jar: str | Optional[Path] = None,
    snpeff_config: str | Optional[Path] = None,
    snpeff_genome: Optional[str] = None,
    partition: Optional[str] = None,
    stage_memories: Mapping[str, str] | None = None,
    mark_duplicates: bool = True,
    remove_duplicates: bool = False,
    rds: str | Optional[Path] = None,
    replicate_column: Optional[str] = None,
    max_concurrent_strata: int = 0,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    stratify_skip: Optional[str] = None,
    min_cells_per_stratum: int = 0,
    rscript: Optional[str] = None,
    data_workdir_path: str | Optional[Path] = None,
    singlecell_differential_workdir: str | Optional[Path] = None,
    singlecell_motif_workdir: str | Optional[Path] = None,
    rna_assay: Optional[str] = None,
    gene_symbol_column: Optional[str] = None,
    singlecell_rna_workdir: str | Optional[Path] = None,
    singlecell_rna_differential_workdir: str | Optional[Path] = None,
    remove_samples: tuple[str, ...] = (),
    skip_motif: bool = False,
    skip_motif_pairs: bool = False,
    skip_evidence: bool = False,
    no_genome_track: bool = False,
    no_trajectories: bool = False,
    trajectory_root: Optional[str] = None,
    trajectory_bins: int = 5,
    trajectory_fit_cells: Optional[str] = None,
    trajectory_dims: Optional[int] = None,
    trajectory_method: Optional[str] = None,
    trajectory_weld: Optional[List[str]] = None,
    trajectory_max_routes: Optional[int] = None,
    trajectory_model: Optional[str] = None,
    trajectory_lineages: Optional[str] = None,
    stop_at: Optional[str] = None,
    stale_stages: tuple[str, ...] = (),
    _only_build_stage: Optional[str] = None,
    # Run this stage, whatever the state says is next -- a stage the supervisor submits alongside the
    # next one, because it reads only finished stages (profiles.STAGE_READS).
    _run_stage: Optional[str] = None,
    ancestry: Optional[str] = None,
) -> PipelineRunResult:
    samples_path = Path(samples) if samples else None
    resolved_profile = resolve_pipeline_profile(profile, segregation_profile)
    profile = resolved_profile.pipeline_profile.value
    segregation_profile = resolved_profile.segregation_profile
    # Evidome submits its own jobs and `evidence run` takes no
    # memory, so a memory given for functional_evidence was accepted and
    # never used. Refused before any stage, where it would have been dropped.
    graph = list(resolved_profile.stage_graph)
    # --skip-evidence takes the stage out of the graph this run
    # walks, so nothing downstream waits on it and the state does not report a
    # stage that was never meant to run.
    if skip_evidence:
        graph = [stage for stage in graph if stage != "functional_evidence"]
    # --no-genome-track, the same for the genome tracks.
    if no_genome_track:
        graph = [stage for stage in graph if stage != "genome_tracks"]
    # --no-trajectories.
    if no_trajectories:
        graph = [stage for stage in graph if stage != "trajectories"]
    stop_raw = str(stop_at or "").strip().lower()
    evidence_runs = "functional_evidence" in graph and not (
        stop_raw in graph and graph.index(stop_raw) < graph.index("functional_evidence")
    )
    if evidence_runs and backend == "slurm" and submit and _stage_mem(stage_memories, "functional_evidence"):
        raise ParalomeError(
            111,
            f"A memory of {_stage_mem(stage_memories, 'functional_evidence')} was given for functional_evidence "
            "(--stage-mem or stage_resources.functional_evidence.mem), but Evidome submits its own SLURM jobs "
            "and sets their memory itself: the value would not be used. Nothing was started.",
            fix="remove the functional_evidence memory from --stage-mem and stage_resources, then run again",
        )
    plan = None
    if samples_path is not None:
        plan = build_dispatch_plan(
            samples=samples_path,
            fastq_qc_workdir=fastq_qc_workdir,
            align_workdir=align_workdir,
            variant_workdir=variant_workdir,
            manifest_profile=resolved_profile.manifest_profile.value,
            stage_graph=resolved_profile.stage_graph,
            # A sheet whose design is in --factor columns has no
            # `condition`, and was refused here before read QC could start.
            factors=tuple(rna_factors),
        )
    state = build_project_state(
        samples=samples_path,
        fastq_qc_workdir=fastq_qc_workdir,
        align_workdir=align_workdir,
        dedup_workdir=dedup_workdir,
        variant_workdir=variant_workdir,
        annotation_workdir=annotation_workdir,
        segregation_workdir=segregation_workdir,
        filter_workdir=filter_workdir,
        report_workdir=report_workdir,
        rds=rds,
        data_workdir_path=data_workdir_path,
        singlecell_differential_workdir=singlecell_differential_workdir,
        singlecell_motif_workdir=singlecell_motif_workdir,
        singlecell_rna_workdir=singlecell_rna_workdir,
        singlecell_rna_differential_workdir=singlecell_rna_differential_workdir,
        profile=profile,
        segregation_profile=segregation_profile,
        skip_motif=skip_motif,
        skip_motif_pairs=skip_motif_pairs,
        # As the supervisor's state reads it (the trajectory fit is singlecell_rna's).
        no_trajectories=no_trajectories,
        trajectory_root=None if no_trajectories else trajectory_root, trajectory_fit_cells=trajectory_fit_cells,
        trajectory_dims=trajectory_dims, trajectory_method=trajectory_method,
        trajectory_weld=tuple(trajectory_weld or ()), trajectory_max_routes=trajectory_max_routes,
        stop_at=stop_at,
        stale_stages=tuple(stale_stages),
        factors=tuple(rna_factors),
    )
    if _only_build_stage or _run_stage:
        # See build_stage_command. The command for a named stage,
        # whatever the state says is next.
        state = {**state, "pipeline": {**state.get("pipeline", {}), "next_stage": _only_build_stage or _run_stage}}
    # --overwrite reaches a stage only when it was typed on the pipeline command. The supervisor
    # re-issues only the stages that check their own result, and refuses for the others.

    workdir, cmd, message, selected_stage = build_next_stage_command(
        samples=samples_path,
        plan=plan,
        fuse_stages=fuse_stages,
        fused_buffer=fused_buffer,
        rds=Path(rds) if rds else None,
        replicate_column=replicate_column,
        max_concurrent_strata=max_concurrent_strata,
        stratify_by=stratify_by,
        stratify_value=stratify_value,
        # Forwarded to build_next_stage_command with the other stratification options, so the generated
        # sbatch carries the values given rather than the defaults.
        stratify_skip=stratify_skip,
        min_cells_per_stratum=min_cells_per_stratum,
        rscript=rscript,
        data_workdir_path=Path(data_workdir_path) if data_workdir_path else None,
        singlecell_differential_workdir=Path(singlecell_differential_workdir) if singlecell_differential_workdir else None,
        singlecell_motif_workdir=Path(singlecell_motif_workdir) if singlecell_motif_workdir else None,
        rna_assay=rna_assay,
        gene_symbol_column=gene_symbol_column,
        singlecell_rna_workdir=Path(singlecell_rna_workdir) if singlecell_rna_workdir else None,
        singlecell_rna_differential_workdir=Path(singlecell_rna_differential_workdir) if singlecell_rna_differential_workdir else None,
        remove_samples=tuple(remove_samples),
        # Singlecell_rna fits the lineages when the run has them.
        trajectory_root=None if no_trajectories else trajectory_root,
        trajectory_bins=trajectory_bins,
        trajectory_fit_cells=trajectory_fit_cells,
        trajectory_dims=trajectory_dims,
        trajectory_method=trajectory_method,
        trajectory_weld=trajectory_weld,
        trajectory_max_routes=trajectory_max_routes,
        trajectory_model=trajectory_model,
        trajectory_lineages=trajectory_lineages,
        reference=Path(reference) if reference else None,
        star_genome=Path(star_genome) if star_genome else None,
        gtf=Path(gtf) if gtf else None,
        fastq_qc_workdir=Path(fastq_qc_workdir) if fastq_qc_workdir else None,
        align_workdir=Path(align_workdir) if align_workdir else None,
        dedup_workdir=Path(dedup_workdir) if dedup_workdir else None,
        variant_workdir=Path(variant_workdir) if variant_workdir else None,
        annotation_workdir=Path(annotation_workdir) if annotation_workdir else None,
        segregation_workdir=Path(segregation_workdir) if segregation_workdir else None,
        filter_workdir=Path(filter_workdir) if filter_workdir else None,
        report_workdir=Path(report_workdir) if report_workdir else None,
        backend=backend,
        submit=submit,
        overwrite=overwrite,
        fastp_threads=fastp_threads,
        star_threads=star_threads,
        fastp_adapter_fasta=Path(fastp_adapter_fasta) if fastp_adapter_fasta else None,
        fastp_cut_right=fastp_cut_right,
        fastp_cut_right_window_size=fastp_cut_right_window_size,
        fastp_cut_right_mean_quality=fastp_cut_right_mean_quality,
        fastp_length_required=fastp_length_required,
        fastp_extra=fastp_extra,
        star_extra=star_extra,
        aligner=aligner,
        bwa_extra=bwa_extra,
        star_read_files_command=star_read_files_command,
        star_out_reads_unmapped=star_out_reads_unmapped,
        star_out_filter_multimap_nmax=star_out_filter_multimap_nmax,
        star_out_filter_mismatch_nover_read_lmax=star_out_filter_mismatch_nover_read_lmax,
        star_out_filter_mismatch_nmax=star_out_filter_mismatch_nmax,
        star_out_filter_score_min_over_lread=star_out_filter_score_min_over_lread,
        star_out_filter_match_nmin_over_lread=star_out_filter_match_nmin_over_lread,
        star_loose=star_loose,
        alignment_profile=resolved_profile.alignment_profile.value,
        dedup_threads=dedup_threads,
        atac_threads=atac_threads,
        motif_scanner_backend=motif_scanner_backend,
        alpha=alpha,
        rna_statistic=rna_statistic,
        rna_factors=tuple(rna_factors),
        rna_pair_by=rna_pair_by,
        rna_covariate=rna_covariate,
        rna_covariate_feature=rna_covariate_feature,
        rna_strandedness=rna_strandedness,
        rna_pseudocount=rna_pseudocount,
        rna_block_size=rna_block_size,
        rna_min_expression=rna_min_expression,
        rna_min_expression_difference=rna_min_expression_difference,
        rna_refine_responders=rna_refine_responders,
        freebayes_threads=freebayes_threads,
        report_threads=report_threads,
        report_shards=report_shards,
        jobs=jobs,
        regions=Path(regions) if regions else None,
        gene_padding=gene_padding,
        min_alternate_count=min_alternate_count,
        min_alternate_fraction=min_alternate_fraction,
        haplotype_length=haplotype_length,
        min_qual=min_qual,
        min_depth=min_depth,
        min_gq=min_gq,
        freebayes_max_region_length=freebayes_max_region_length,
        freebayes_stall_minutes=freebayes_stall_minutes,
        annotation=annotation,
        profile=profile,
        segregation_profile=segregation_profile,
        genome=genome,
        annovar_dir=Path(annovar_dir) if annovar_dir else None,
        annovar_db=Path(annovar_db) if annovar_db else None,
        annovar_buildver=annovar_buildver,
        annovar_protocol=annovar_protocol,
        annovar_operation=annovar_operation,
        snpeff_jar=Path(snpeff_jar) if snpeff_jar else None,
        snpeff_config=Path(snpeff_config) if snpeff_config else None,
        snpeff_genome=snpeff_genome,
        partition=partition,
        stage_memories=stage_memories,
        mark_duplicates=mark_duplicates,
        remove_duplicates=remove_duplicates,
        state=state,
        # A dry run, and a command built only to be compared, write
        # nothing.
        write_inputs=not (dry_run or _only_build_stage),
        ancestry=ancestry,
    )

    result = PipelineRunResult(
        workdir=Path(workdir),
        next_stage=selected_stage,
        command=cmd,
        executed=False,
        returncode=None,
        dispatch_plan=plan.to_dict() if plan is not None else {},
        message=message,
    )

    if _only_build_stage:
        return result

    provenance = result.to_dict()
    provenance["profile"] = resolved_profile.to_dict()
    provenance["project_state"] = state.get("pipeline", {})
    # Not under a dry run, which writes nothing.
    if cmd and not dry_run:
        write_json(Path(workdir) / "inputs" / "pipeline_run_plan.json", provenance)

    if cmd and not dry_run:
        pipeline_name = resolved_profile.pipeline_profile.value
        with _pipeline_in_environment(pipeline_name):
            if backend == "slurm" and submit and selected_stage in STAGES_SUBMITTED_AS_ONE_JOB:
                # Its own job, not the supervisor's process. The command
                # recorded (result.command) is still the stage command itself.
                returncode, submitted_job_ids = _submit_stage_as_one_job(
                    selected_stage, cmd, workdir,
                    mem=_stage_mem(stage_memories, selected_stage), partition=partition,
                    pipeline=pipeline_name,
                )
            elif backend == "slurm" and submit:
                returncode, submitted_job_ids = _run_stage_command(cmd, own_session=True)
            else:
                returncode, submitted_job_ids = _run_stage_command(cmd)
        result.executed = True
        result.returncode = returncode
        result.submitted_job_ids = submitted_job_ids
        provenance = result.to_dict()
        provenance["profile"] = resolved_profile.to_dict()
        provenance["project_state"] = state.get("pipeline", {})
        write_json(Path(workdir) / "inputs" / "pipeline_run_plan.json", provenance)
        # Refresh project state in target workdir after each synchronous command.
        try:
            refreshed = build_project_state(
                samples=samples_path,
                fastq_qc_workdir=fastq_qc_workdir,
                align_workdir=align_workdir,
                dedup_workdir=dedup_workdir,
                variant_workdir=variant_workdir,
                annotation_workdir=annotation_workdir,
                segregation_workdir=segregation_workdir,
                filter_workdir=filter_workdir,
                report_workdir=report_workdir,
                rds=rds,
                data_workdir_path=data_workdir_path,
                singlecell_differential_workdir=singlecell_differential_workdir,
                singlecell_motif_workdir=singlecell_motif_workdir,
                singlecell_rna_workdir=singlecell_rna_workdir,
                singlecell_rna_differential_workdir=singlecell_rna_differential_workdir,
                profile=profile,
                segregation_profile=segregation_profile,
                skip_motif=skip_motif,
                # As for the state built before the stage, so the
                # next stage named here is the one the run will take.
                skip_motif_pairs=skip_motif_pairs,
                no_trajectories=no_trajectories,
                trajectory_root=None if no_trajectories else trajectory_root, trajectory_fit_cells=trajectory_fit_cells,
                trajectory_dims=trajectory_dims, trajectory_method=trajectory_method,
                trajectory_weld=tuple(trajectory_weld or ()), trajectory_max_routes=trajectory_max_routes,
                stop_at=stop_at,
                run_plan=provenance,
                stale_stages=tuple(name for name in stale_stages if name.partition(".")[0] != selected_stage),
                factors=tuple(rna_factors),
            )
            write_project_state(Path(workdir), refreshed)
        except Exception as exc:
            # Said, not swallowed; the stage's own result stands.
            print(f"Warning: could not refresh {Path(workdir)}/project_state.json: {exc}", file=sys.stderr)
        if returncode != 0:
            raise StageCommandError(
                returncode, cmd, workdir=Path(workdir), submitted_job_ids=submitted_job_ids,
            )

    return result


# ---------------------------------------------------------------------------
# What a stage was asked for, in a form that can be compared later
# ---------------------------------------------------------------------------

#: Stages that decide for themselves, from their own checkpoint, whether a
#: finished result answers the request and the inputs they are now given, and
#: compute again (replacing only their own files) when it does not. The
#: supervisor may issue one of these again over its earlier result, without
#: --overwrite. Every other stage, issued again, would keep what it has: the
#: supervisor refuses instead and names the folder (PLM-0308).
#:
#: The two single-cell aggregations are among them. Each compares
#: the plan it is given with the one in its checkpoint and aggregates again,
#: in place, from the extraction it keeps (atac/singlecell.py,
#: rna/singlecell.py; on SLURM the one job runs that same check). Their folder
#: is the study's shared data folder, which is nobody's to be told to remove.
SELF_CHECKING_STAGES = frozenset({
    "rna_quantification", "differential_expression", "functional_evidence",
    "singlecell_atac", "singlecell_rna",
    # Compares its VCF, probands, reference and options with its
    # checkpoint; keeps the result or computes again (paralome.burden.stage).
    "gene_burden",
    # Again: the same for the genome tracks' zip (paralome.atac.genome_tracks).
    "genome_tracks",
    # Compares its inputs (files' size and time) and options with its checkpoint; keeps the zip or builds
    # it again (paralome.browser.stage). Genotype_output and statistics_output.
    "genotype_output", "statistics_output",
    # Compares its request (root, bins, statistic, the extraction's files) with its checkpoint;
    # keeps the zip or fits and tests again (paralome.singlecell.trajectories).
    "trajectories",
    # FreeBayes fingerprints its chunks, options and BAMs (path,
    # size, time) in inputs/freebayes_chunk_plan.json. Issued again with the
    # same plan it calls only the chunks without a shard, and nothing at all
    # when every shard is there and merged under that plan; with a different
    # plan it calls everything afresh, after moving the old shards and merged
    # VCF to superseded/ -- nothing is deleted (paralome.variants.freebayes).
    "variants",
    # The four motif stages. `atac motifs` has no --overwrite and keeps no result on its own account: its
    # scan shards are reused only under a plan id hashed from the peaks, the manifest, the genome, the
    # motif library and the thresholds; a --stage pairs hand-off is resumed only when its identity record
    # matches; and the finalizer withdraws the old checkpoint, computes every table again and removes
    # result files the run did not produce (atac/motifs.py). So a motif stage whose request changed, or
    # whose checkpoint pipeline.state reads as out of date, is issued again in place, with the reason
    # said, instead of being refused with PLM-0308 until its folder is removed.
    "motif_analysis_singles", "motif_analysis_pairs",
    "singlecell_motif_analysis_singles", "singlecell_motif_analysis_pairs",
})
#: Of those, the one that computes afresh on every run: functional evidence
#: sets the earlier archive aside before it starts Evidome (paralome.evidence),
#: so an archive that is there afterwards was written by that run.
STAGES_THAT_ALWAYS_RECOMPUTE = frozenset({"functional_evidence"})

#: The one command that submits jobs without being given --submit: Evidome
#: submits its own chain whenever its backend is SLURM.
_SUBMITS_WITHOUT_SUBMIT = ("evidence", "run")


def only_writes_scripts(command: list[str]) -> bool:
    """Whether this stage command writes SLURM scripts and runs nothing.

    Read from the command that was built, not from the run's options.
    Segregation, filtering and reporting have no backend and always run where
    they are started, so a run on the SLURM backend without --submit does run
    them, and their requests are recorded.
    """
    tokens = [str(token) for token in command]
    prefix = [str(token) for token in worker_argv()]
    if tokens[:len(prefix)] == prefix:
        tokens = tokens[len(prefix):]
    if tuple(tokens[:2]) == _SUBMITS_WITHOUT_SUBMIT or "--submit" in tokens:
        return False
    return "--backend" in tokens[:-1] and tokens[tokens.index("--backend") + 1] == "slurm"


#: Stages whose command is handed the sample sheet (or, for RNA, whose result
#: is a table across the cohort): their request includes which samples are
#: switched on and the columns a stage reads. The per-sample stages do not, so
#: commenting a sample out never re-trims or re-aligns the others.
COHORT_STAGES = frozenset({
    "rna_quantification", "differential_expression", "functional_evidence",
    "differential_accessibility", "motif_analysis_singles", "motif_analysis_pairs",
    "segregation",
    # Its probands are the sheet's affected samples.
    "gene_burden",
    # Bulk ATAC's tracks draw one profile per active sample, labelled with its value of the first --factor;
    # the statistics_output results carry every sample's sheet columns (DNA's genotype_output reads the
    # cohort's segregation table).
    "statistics_output", "genotype_output",
    "genome_tracks",
})

#: Stages that pool the active samples but read no design column
#: (joint calling, the peak atlas, the count matrix). They are handed a manifest written inside the
#: project (inputs/*_samples.csv), whose path alone is the same after a sample is commented out, so
#: their request records which samples are switched on -- not their groups, so a relabel does not
#: re-run FreeBayes or rebuild the atlas.
POOLED_SAMPLE_STAGES = frozenset({"variants", "peak_merge", "peak_quantification"})

# Options that say how or where a stage runs, not what it computes.
_REQUEST_IGNORED_OPTIONS = frozenset({
    "--backend", "--threads", "--jobs", "--mem", "--partition",
    "--block-size", "--shards", "--max-concurrent-strata", "--rscript",
    # Names of output files, derived from the project folder's name: renaming
    # the project is not a new request.
    "--prefix", "--workbook-prefix",
    # Where an older layout's tracks may be taken over from, not what the tracks are made of;
    # the same for snATAC's former folder, a stage's former folder, and where its zips go.
    "--bundled-tracks", "--former-tracks", "--former-folder", "--zip-dir",
})
# Every option the commands above emit WITHOUT a value. Anything else takes
# one, which is how a value that itself starts with "--" (--star-extra) is
# told from an option. A test holds this to the commands built above.
_REQUEST_SWITCHES = frozenset({
    "--submit", "--overwrite", "--cut-right", "--no-cut-right", "--loose",
    # The fused task's fastp sliding window, and the DNA pipeline's overlap clipping.
    "--trim-cut-right", "--no-trim-cut-right", "--clip-overlap",
    "--single-end", "--paired-end",
    "--remove-duplicates",
    # Reporting's.
    "--no-workbook",
    # The fused stages' switch. --trim-in-task and --fuse-dedup
    # take a path, so they are options rather than switches and the reader
    # already handles them.
    "--defer-trimming",
    # The RNA fusion's switch. The alignment marks duplicates in
    # its own tree when the profile has no deduplication stage to hand them
    # to, so the option carries no value.
    "--mark-in-place",
})
_REQUEST_IGNORED_SWITCHES = frozenset({"--submit", "--overwrite"})
_REQUEST_PATH_OPTIONS = frozenset({
    "--samples", "--out", "--reference", "--gtf", "--adapter-fasta", "--expression",
    "--peak-manifest", "--atlas", "--cpm", "--differential-statistics",
    "--differential-checkpoint", "--rds", "--qc-root",
    "--extraction", "--regions", "--vcf", "--variant-workdir", "--annovar-dir",
    "--annovar-db", "--snpeff-jar", "--snpeff-config", "--statistics", "--segregation",
    # Every arm's browser bundle (browser arm).
    "--results", "--motifs", "--samples-table", "--coverage-summary",
})
# Sample ids to leave out: a set, so the order they were typed in is not a change.
_REQUEST_UNORDERED_OPTIONS = frozenset({"--remove"})
SAMPLE_SHEET_MARK = "<sample sheet>"
PROJECT_MARK = "<project>"
#: How a request record is written. Raised whenever the way a
#: command is reduced to a record changes (what is left out, how a path is
#: spelt). A record with another number, or none, was written under other
#: rules: the supervisor takes it over as it stands and does not compare it.
REQUEST_RECORD_FORMAT = 3
#: Options that say how ONE stage runs, though they decide
#: what another computes. Reporting's --regions only deals the rows into
#: shards (reports/candidates.py _split_report_input); the report is the same.
#: Kept in reporting's request it read as a change the first time the branch
#: found FreeBayes' region file, and a finished report was refused (PLM-0308).
_REQUEST_IGNORED_BY_STAGE: dict[str, frozenset] = {
    # How long a region one FreeBayes call is given, and when a
    # call is taken for stalled. They change how the calls are run, at region
    # edges and stalls, not what the stage is asked to call; a finished
    # variant calling is not refused for them. Both are recorded in its
    # results/freebayes_summary.json (with the chunk plan they made).
    "variants": frozenset({"--max-region-length", "--stall-minutes"}),
    # --regions-convention says how that map is read; the same.
    "reporting": frozenset({"--regions", "--regions-convention",
                            # Whether the TSVs are also drawn as
                            # workbooks; the report is the same.
                            "--no-workbook"}),
    # Where the extraction writes the cell export (the run's omicview folder) -- not what is extracted
    # or aggregated; a finished singlecell_rna is not changed by it.
    "singlecell_rna": frozenset({"--cell-export-dir"}),
}
# The columns of the sample sheet a stage reads, besides the sample id and the
# columns named by --factor: the group of each sample and, for the family
# stages, the pedigree. A notes column, or any other, is nobody's input.
_SHEET_COLUMNS_STAGES_READ = frozenset({
    "condition", "group", "status", "affected", "relationship", "relation",
    "family_id", "family", "pedigree", "sex", "gender", "mother_id", "father_id",
})
#: Columns only some stages read. gene_burden reads each proband's
#: `ancestry`; a change there is a new request for it and for no other stage.
_SHEET_COLUMNS_ONE_STAGE_READS: dict[str, frozenset] = {
    "gene_burden": frozenset({"ancestry"}),
}


def _request_path(value: str, project: Optional[Path], samples: Optional[Path]) -> str:
    """One spelling for a path, however it was typed.

    Symbolic links and relative paths are resolved. A path inside the project
    is written relative to it, so a project that was moved or reached through
    another mount point compares equal. The sample sheet is written as a mark:
    which file it is does not matter, and what is in it is compared separately
    for the stages that depend on it.

    A path OUTSIDE the project (reference, annotation, index, candidate list)
    is written as what it is, not where it is: a file by name and size, a
    folder by name, so moving the reference is not a new request for every
    stage that uses it. One that is not there cannot be compared and raises;
    the caller then leaves the stage alone.
    """
    real = os.path.realpath(str(value))
    if samples is not None and real == os.path.realpath(str(samples)):
        return SAMPLE_SHEET_MARK
    if project is not None:
        root = os.path.realpath(str(project))
        if real == root:
            return PROJECT_MARK
        if real.startswith(root + os.sep):
            return PROJECT_MARK + "/" + os.path.relpath(real, root).replace(os.sep, "/")
    name = os.path.basename(real) or real
    if os.path.isdir(real):
        return name
    try:
        return f"{name} ({os.path.getsize(real)} bytes)"
    except OSError:
        raise ValueError(f"{value} is not there to compare") from None


def normalise_stage_command(
    command: list[str],
    *,
    project: Optional[Path] = None,
    samples: Optional[Path] = None,
) -> dict[str, Any]:
    """A stage command reduced to what decides its result.

    Returns ``{"command": [...], "options": {option: [values]}}``: the worker
    sub-command, and every option that changes what is computed. The program
    path, and the options for where and how wide the stage runs, are left out.
    """
    tokens = [str(token) for token in command]
    prefix = [str(token) for token in worker_argv()]
    # Where the worker is installed is not part of the request. Both sides of a
    # comparison are built by this installation, so the prefix is always this.
    if tokens[:len(prefix)] == prefix:
        tokens = tokens[len(prefix):]
    words: list[str] = []
    while tokens and not tokens[0].startswith("--"):
        words.append(tokens.pop(0))
    options: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        option = tokens[index]
        if option in _REQUEST_SWITCHES or index + 1 >= len(tokens):
            index += 1
            if option not in _REQUEST_IGNORED_SWITCHES:
                options.setdefault(option, [])
            continue
        value = tokens[index + 1]
        index += 2
        if option in _REQUEST_IGNORED_OPTIONS:
            continue
        if option in _REQUEST_PATH_OPTIONS:
            value = _request_path(value, project, samples)
        options.setdefault(option, []).append(value)
    for option in _REQUEST_UNORDERED_OPTIONS & set(options):
        options[option] = sorted(options[option])
    return {"command": words, "options": {name: options[name] for name in sorted(options)}}


def sample_sheet_meaning(
    samples: Optional[Union[str, Path]],
    remove_samples: tuple[str, ...] = (),
    factors: tuple[str, ...] = (),
    stage: Optional[str] = None,
) -> Optional[dict[str, dict[str, str]]]:
    """What the sample sheet says to a stage, without where its files are.

    The samples that are switched on (comment lines and --remove taken out),
    each with the columns a stage reads: its group, the pedigree columns, and
    the columns named by --factor. ``None`` when there is no sheet or it
    cannot be read; the run reports an unreadable sheet itself.

    This was every column that is not a file path, so a word typed
    into a notes column ran the cohort stages again.
    """
    if not samples:
        return None
    from paralome.fastq.manifest import SAMPLE_ALIASES

    try:
        fieldnames, rows = read_manifest_rows(Path(samples))
    except Exception:
        return None
    lowered = {str(name).strip().lower(): name for name in fieldnames if isinstance(name, str)}
    id_column = next((lowered[alias] for alias in SAMPLE_ALIASES if alias in lowered), None)
    if id_column is None:
        return None
    read_columns = (
        _SHEET_COLUMNS_STAGES_READ | _SHEET_COLUMNS_ONE_STAGE_READS.get(str(stage), frozenset())
        | {str(name).strip().lower() for name in factors}
    )
    removed = {str(name).strip() for name in remove_samples}
    meaning: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = str(row.get(id_column) or "").strip()
        if not sample_id or sample_id in removed:
            continue
        meaning[sample_id] = {
            str(name).strip(): str(value or "").strip()
            for name, value in row.items()
            if isinstance(name, str) and name != id_column and name.strip().lower() in read_columns
        }
    return meaning


#: The stages that read each sample's input files straight from
#: the sheet: read QC its FASTQs, duplicate marking the BAMs of a BAM sheet.
#: Their request records hold each sample's input files by name and size.
SHEET_INPUT_STAGES = frozenset({"fastq_qc", "deduplication"})


def sample_input_identity(samples: Optional[Union[str, Path]]) -> Optional[dict[str, list[str]]]:
    """Each active sample's input files, as "name (N bytes)" ("name (missing)"), or None if unreadable.

    In the per-sample stages' request record, so pointing a sample at different
    FASTQs or a different BAM is a different request. Name and size, not the
    path: a project copied or reached under another mount is the same request.
    """
    if not samples:
        return None
    from paralome.fastq.manifest import read_sample_inputs

    try:
        rows = read_sample_inputs(Path(samples))
    except Exception:
        return None
    identity: dict[str, list[str]] = {}
    for row in rows:
        files = [value for value in (row.bam, row.fastq_r1, row.fastq_r2) if value]
        described = []
        for value in files:
            path = Path(value)
            try:
                described.append(f"{path.name} ({path.stat().st_size} bytes)")
            except OSError:
                described.append(f"{path.name} (missing)")
        identity[row.sample_id] = described
    return identity


def stage_request_record(
    stage: str,
    command: list[str],
    *,
    project: Optional[Path] = None,
    samples: Optional[Union[str, Path]] = None,
    remove_samples: tuple[str, ...] = (),
) -> dict[str, Any]:
    """What ``stage`` was asked to compute: a fingerprint and the readable form.

    Two runs of the same request give the same record, whatever the working
    directory, the thread count or the backend.
    """
    record = normalise_stage_command(command, project=project, samples=Path(samples) if samples else None)
    for option in _REQUEST_IGNORED_BY_STAGE.get(stage, frozenset()):
        record["options"].pop(option, None)
    if stage in SHEET_INPUT_STAGES:
        inputs = sample_input_identity(samples)
        if inputs is not None:
            record["inputs"] = inputs
    if stage in POOLED_SAMPLE_STAGES:
        meaning = sample_sheet_meaning(samples, tuple(remove_samples), stage=stage)
        if meaning is not None:
            record["sample_set"] = sorted(meaning)
    if stage in COHORT_STAGES:
        record["samples"] = sample_sheet_meaning(
            samples, tuple(remove_samples), factors=tuple(record["options"].get("--factor") or ()),
            stage=stage,
        )
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return {
        "fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "format": REQUEST_RECORD_FORMAT,
        **record,
    }


_NAME_AND_SIZE = re.compile(r"^(?P<name>.*) \((?P<size>\d+) bytes\)$")


def _same_file_written_two_ways(one: str, other: str, project: Optional[Path]) -> bool:
    """Whether "<project>/.../NAME" and "NAME (N bytes)" can be the same file.

    A stage's checkpoint holds the absolute paths of its outputs.
    In a copy of a project, or in one reached under another mount point, the
    next stage's input was therefore a path into the other tree: written by
    name and size, and compared with the record's path inside the project, an
    untouched project read as changed. The two are the same request when the
    names agree, and the sizes where the file is in this project to measure.
    """
    inside, outside = (one, other) if one.startswith(PROJECT_MARK + "/") else (other, one)
    if not inside.startswith(PROJECT_MARK + "/") or outside.startswith(PROJECT_MARK):
        return False
    described = _NAME_AND_SIZE.match(outside)
    name = described.group("name") if described else outside
    if inside.rsplit("/", 1)[-1] != name:
        return False
    if described and project is not None:
        try:
            return os.path.getsize(Path(project) / inside[len(PROJECT_MARK) + 1:]) == int(described.group("size"))
        except OSError:
            return True
    return True


def _same_request_values(option: str, old: Any, new: Any, project: Optional[Path]) -> bool:
    if old == new:
        return True
    if option not in _REQUEST_PATH_OPTIONS or old is None or new is None or len(old) != len(new):
        return False
    return all(
        one == other or _same_file_written_two_ways(str(one), str(other), project)
        for one, other in zip(old, new)
    )


def describe_request_change(
    before: Mapping[str, Any], after: Mapping[str, Any], project: Optional[Path] = None,
) -> str:
    """What differs between two request records, for the person reading the console."""
    changes: list[str] = []
    if list(before.get("command") or []) != list(after.get("command") or []):
        changes.append(
            "command " + " ".join(before.get("command") or ["?"]) + " -> " + " ".join(after.get("command") or ["?"])
        )
    old_options = _in_current_spelling(before.get("options")) if isinstance(before.get("options"), Mapping) else {}
    new_options = after.get("options") if isinstance(after.get("options"), Mapping) else {}
    old_options, new_options = _before_dna_mode(old_options, new_options)
    for name in sorted(set(old_options) | set(new_options)):
        old, new = old_options.get(name), new_options.get(name)
        if _same_request_values(name, old, new, project):
            continue
        if old is None:
            changes.append((name + " " + " ".join(new)).strip() + " (added)")
        elif new is None:
            changes.append((name + " " + " ".join(old)).strip() + " (no longer given)")
        else:
            changes.append(f"{name} {' '.join(old) or '.'} -> {' '.join(new) or '.'}")
    old_samples, new_samples = before.get("samples"), after.get("samples")
    if isinstance(old_samples, Mapping) and isinstance(new_samples, Mapping) and old_samples != new_samples:
        gone = sorted(set(old_samples) - set(new_samples))
        added = sorted(set(new_samples) - set(old_samples))
        if gone:
            changes.append("samples no longer in the sheet: " + ", ".join(gone))
        if added:
            changes.append("samples added to the sheet: " + ", ".join(added))
        for sample_id in sorted(set(old_samples) & set(new_samples)):
            old_row, new_row = old_samples[sample_id] or {}, new_samples[sample_id] or {}
            for column in sorted(set(old_row) | set(new_row)):
                if old_row.get(column) != new_row.get(column):
                    changes.append(
                        f"sample {sample_id} {column} {old_row.get(column) or '.'} -> {new_row.get(column) or '.'}"
                    )
    old_set, new_set = before.get("sample_set"), after.get("sample_set")
    if isinstance(old_set, list) and isinstance(new_set, list) and sorted(old_set) != sorted(new_set):
        gone, added = sorted(set(old_set) - set(new_set)), sorted(set(new_set) - set(old_set))
        if gone:
            changes.append("samples no longer pooled: " + ", ".join(gone))
        if added:
            changes.append("samples added to the pool: " + ", ".join(added))
    old_inputs, new_inputs = before.get("inputs"), after.get("inputs")
    if isinstance(old_inputs, Mapping) and isinstance(new_inputs, Mapping):
        for sample_id in sorted(set(old_inputs) & set(new_inputs)):
            if list(old_inputs[sample_id] or []) != list(new_inputs[sample_id] or []):
                changes.append(
                    f"sample {sample_id} input {', '.join(old_inputs[sample_id] or ['.'])} -> "
                    f"{', '.join(new_inputs[sample_id] or ['.'])}"
                )
    shown = changes[:6] + ([f"and {len(changes) - 6} more"] if len(changes) > 6 else [])
    return "; ".join(shown) or "the request differs"


def _in_current_spelling(options: Mapping[str, Any]) -> dict[str, Any]:
    """A recorded request's options as current Paralome spells them.

    ``--group-by <replicate>`` or ``--group-by
    <replicate>,<stratum>`` (the --stratify-by column) made the same samples
    ``--replicate-column <replicate>`` makes, so it is read as that. Any other
    --group-by grouped by something --replicate-column cannot say, and is left
    as it was, a difference.
    """
    options = dict(options or {})
    grouping = options.get("--group-by")
    if not grouping or "--replicate-column" in options:
        return options
    columns = [column.strip() for column in str(grouping[0]).split(",") if column.strip()]
    stratum = (options.get("--stratify-by") or [None])[0]
    if columns and columns[1:] in ([], [stratum]):
        options.pop("--group-by")
        options["--replicate-column"] = [columns[0]]
    return options


def _before_dna_mode(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """An alignment finished before STAR ran in DNA mode (DNA, ATAC) is not a changed request.

    Its record has the GTF and no --unspliced-mates-gap; the BAMs it made are kept as they are, and only a
    new alignment (the stage deleted, or another project) gets DNA mode.
    """
    old, new = dict(old), dict(new)
    if "--unspliced-mates-gap" in new and "--unspliced-mates-gap" not in old:
        new.pop("--unspliced-mates-gap")
        old.pop("--gtf", None)
    # Nor is one made before Paralome switched STAR's absolute mismatch cap off (align.star._SETTLED_KEYS):
    # the cap is fixed by Paralome, not a request.
    old.pop("--star-out-filter-mismatch-nmax", None)
    new.pop("--star-out-filter-mismatch-nmax", None)
    # Nor one marked before Paralome clipped mate overlaps: its BAMs are kept; a new alignment clips.
    if "--clip-overlap" in new and "--clip-overlap" not in old:
        new.pop("--clip-overlap")
    # Nor a fused alignment made before Paralome handed it the run's fastp settings (it trimmed with the
    # defaults): its BAMs are kept.
    if "--trim-in-task" in new and not any(k.startswith("--trim-cut-right") or k.startswith("--no-trim-cut-right")
                                           for k in old):
        for key in ("--trim-cut-right", "--no-trim-cut-right", "--trim-cut-right-window-size",
                    "--trim-cut-right-mean-quality", "--trim-adapter-fasta", "--trim-length-required", "--trim-extra"):
            if key not in old:
                new.pop(key, None)
    return old, new


def request_records_differ(
    before: Mapping[str, Any], after: Mapping[str, Any], project: Optional[Path] = None,
) -> bool:
    """Whether two request records ask for different results.

    The sample part is compared only when both records could read the sheet.
    ``project`` is where a file written inside the project can be measured.
    """
    if list(before.get("command") or []) != list(after.get("command") or []):
        return True
    old_options, new_options = _in_current_spelling(before.get("options") or {}), dict(after.get("options") or {})
    old_options, new_options = _before_dna_mode(old_options, new_options)
    if any(
        not _same_request_values(name, old_options.get(name), new_options.get(name), project)
        for name in set(old_options) | set(new_options)
    ):
        return True
    old_samples, new_samples = before.get("samples"), after.get("samples")
    if isinstance(old_samples, Mapping) and isinstance(new_samples, Mapping):
        if dict(old_samples) != dict(new_samples):
            return True
    # The pooled stages' sample set (only when both records hold one:
    # a record from has none, and is not stale for that).
    old_set, new_set = before.get("sample_set"), after.get("sample_set")
    if isinstance(old_set, list) and isinstance(new_set, list) and sorted(old_set) != sorted(new_set):
        return True
    # A sample's input files, compared for the samples in both records only (one added or
    # taken out is the per-sample hold's business), and only when both records hold them (a record from
    # has none, and is not stale for that).
    old_inputs, new_inputs = before.get("inputs"), after.get("inputs")
    if isinstance(old_inputs, Mapping) and isinstance(new_inputs, Mapping):
        return any(list(old_inputs[s] or []) != list(new_inputs[s] or [])
                   for s in set(old_inputs) & set(new_inputs))
    return False


def build_stage_command(stage: str, **run_kwargs: Any) -> tuple[Path, list[str]]:
    """The command this request gives ``stage``, built and not run.

    Takes what ``run_pipeline_next_stage`` takes. Nothing is executed and no
    plan file is written. Raises what building the command raises -- a stage
    whose inputs are not there yet cannot be built.
    """
    result = run_pipeline_next_stage(**{**run_kwargs, "dry_run": True, "_only_build_stage": stage})
    return result.workdir, list(result.command)
