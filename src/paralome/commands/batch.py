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
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import typer

from paralome.binaries import human_argv, worker_command
from paralome.csvio import skip_comment_lines
from paralome.pipeline.profiles import normalize_pipeline_profile


batch_app = typer.Typer(help="Run multiple complete Paralome projects with family-level admission control")

# Two apps, so the audience split is structural: `batch_app` carries `run` and mounts on `paralome`;
# `batch_worker_app` carries `worker` (a command whose only caller is a generated sbatch body) and
# mounts on `paralome-worker`. Both keep the group name `batch`, so the generated
# `<worker> batch worker --plan ...` argv reads the same.
batch_worker_app = typer.Typer(help="Batch array-task internals. Invoked by generated SLURM scripts.")

_RESERVED_PIPELINE_OPTIONS = {
    "--samples",
    "--project",
    "--backend",
    "--submit",
    "--dry-run",
}


@dataclass(frozen=True)
class BatchFamily:
    project: Path
    samples: Path
    pipeline_args: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": str(self.project),
            "samples": str(self.samples),
            "pipeline_args": list(self.pipeline_args),
        }


@dataclass(frozen=True)
class BatchPlan:
    name: str
    family_list: Path
    workdir: Path
    max_families: int
    commands: Tuple[Tuple[str, ...], ...]
    script: Path
    logs_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "family_list": str(self.family_list),
            "workdir": str(self.workdir),
            "max_families": self.max_families,
            "commands": [list(command) for command in self.commands],
            "script": str(self.script),
            "logs_dir": str(self.logs_dir),
        }


def _first_nonempty(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_extra_args(args: Sequence[str], *, source: str) -> None:
    for token in args:
        option = token.split("=", 1)[0]
        if option in _RESERVED_PIPELINE_OPTIONS:
            raise ValueError(
                f"{source} may not override reserved pipeline option {option}; "
                "use the batch family list and batch options instead"
            )


def read_family_list(path: Path) -> Tuple[BatchFamily, ...]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Family list does not exist: {path}")

    # Utf-8-sig drops the byte-order mark Excel writes.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        # A '#' row is switched off, as in every other sheet.
        reader = csv.DictReader(skip_comment_lines(handle))
        if not reader.fieldnames:
            raise ValueError(f"Family list has no header: {path}")
        rows = list(reader)

    families: List[BatchFamily] = []
    seen_projects = set()
    base = path.parent

    for index, row in enumerate(rows, start=2):
        project_value = _first_nonempty(row, ("project", "family", "project_dir"))
        samples_value = _first_nonempty(row, ("samples", "manifest", "sample_sheet", "samplesheet"))
        if not project_value or not samples_value:
            raise ValueError(
                f"Invalid family list row {index}: expected project/family and samples/manifest"
            )

        project = _resolve_path(project_value, base=base)
        samples = _resolve_path(samples_value, base=base)
        if project in seen_projects:
            raise ValueError(f"Duplicate project in family list row {index}: {project}")
        if not samples.is_file():
            raise ValueError(f"Sample manifest does not exist for row {index}: {samples}")

        row_args_text = (row.get("pipeline_args") or "").strip()
        row_args = tuple(shlex.split(row_args_text)) if row_args_text else ()
        _validate_extra_args(row_args, source=f"family list row {index} pipeline_args")
        families.append(BatchFamily(project=project, samples=samples, pipeline_args=row_args))
        seen_projects.add(project)

    if not families:
        raise ValueError(f"Family list contains no projects: {path}")
    return tuple(families)


def _paralome_argv(explicit: Optional[Path]) -> List[str]:
    """The argv prefix each per-family command runs.

    This resolves argv[0] of the per-family ANALYSIS command, which is a human
    command, so it returns the human binary. The worker binary is what runs the
    array task, and that is resolved separately in ``render_batch_script``. The
    emitted command can therefore be copied out of ``batch_plan.json`` and run
    by hand.

    Returns the full argv PREFIX, not a single token: the fallback form is
    ``[sys.executable, "-m", "paralome.cli"]``, three tokens, and flattening
    that into one string would produce a command whose argv[0] is not an
    executable. ``batch worker`` hands these straight to ``subprocess.run``.
    """
    if explicit is not None:
        return [str(explicit.expanduser().resolve())]
    return human_argv()


def _option_values(command: Sequence[str]) -> Dict[str, str]:
    """The value each option of a family command ends up with.

    Both spellings, `--genome X` and `--genome=X`; the last one given wins, as
    it does when the command is run, so a family's own pipeline_args replace
    what `batch run` gave every family.
    """
    values: Dict[str, str] = {}
    for index, token in enumerate(command):
        if not token.startswith("--"):
            continue
        if "=" in token:
            option, value = token.split("=", 1)
            values[option] = value
        elif index + 1 < len(command) and not command[index + 1].startswith("--"):
            values[token] = command[index + 1]
        else:
            values.setdefault(token, "")
    return values


def _reaches_annotation(stop_at: Optional[str]) -> bool:
    """Whether a DNA family command with this --stop-at runs annotation.

    This tested for --stop-at-stage, which no command has, so a
    family stopped before annotation was refused over its ANNOVAR lists. The
    stop stage is resolved as the analysis resolves it; a value it cannot
    resolve is left to the family command and the lists are still checked.
    """
    from paralome.pipeline.profiles import resolve_pipeline_profile, resolve_stage_alias

    if not stop_at:
        return True
    graph = resolve_pipeline_profile("DNA").stage_graph
    try:
        stop_stage = resolve_stage_alias(stop_at, graph)
    except ValueError:
        return True
    return "annotation" in graph[: graph.index(stop_stage) + 1]


def _require_a_genome_in(
    command: Sequence[str], *, analysis: str, project: Path,
    pipeline_config: Optional[Path] = None,
) -> None:
    """Every family command names a genome that exists, or the plan is not built.

    There is no default genome, so a family command without --genome
    would be refused by the analysis -- inside its array task, after the batch
    had reported the array submitted. Refused here instead.

    The name is resolved too, with the function the analysis resolves it with:
    a mistyped --genome built the plan, was reported submitted, and each family
    task then died with PLM-0401. And a DNA command without --genome is judged
    by the analysis's own rule for a run given every path as an option.
    """
    from paralome.commands.pipeline import (
        _annotation_engines, _configured_setting, _load_pipeline_config,
        _refuse_an_operation_from_elsewhere, _require_a_genome, what_explicit_paths_lack,
    )
    from paralome.errors import ParalomeError
    from paralome.pipeline.project import cfg_get

    given = _option_values(command)
    engines: List[str] = []
    # Beside a --genome, a --pipeline-config that is not there is left for the
    # family command to refuse, as it was.
    if analysis == "DNA" and (
        not given.get("--genome") or pipeline_config is None or Path(pipeline_config).is_file()
    ):
        typed = given.get("--annotation")
        engines = _annotation_engines(
            typed or str(cfg_get(_load_pipeline_config(pipeline_config), "annotation", "engine", "annovar")),
            source="--annotation" if typed else "annotation.engine in the configuration",
        )
        # One ANNOVAR list typed or configured without the other is
        # refused by the analysis; here that is before anything is submitted.
        # A run stopped before annotation is not asked.
        if "annovar" in engines and _reaches_annotation(given.get("--stop-at")):
            _refuse_an_operation_from_elsewhere(
                pipeline_config=pipeline_config,
                typed={
                    "annovar_protocol": given.get("--annovar-protocol"),
                    "annovar_operation": given.get("--annovar-operation"),
                },
            )
    if given.get("--genome"):
        _require_a_genome(analysis=analysis, genome=given["--genome"], explicit_paths_cover_it=False)
        return
    lacking: List[str] = []
    if analysis == "DNA" and {"--star-genome", "--reference"} <= set(given):
        # The protocol and operation lists are the option or the
        # configured value, as the analysis reads them. Without them the plan
        # was built and submitted, and each family task stopped with PLM-0117.
        def listed(option: str, key: str) -> Optional[str]:
            return given.get(option) or _configured_setting("annotation", key, pipeline_config)[0]

        lacking = what_explicit_paths_lack(
            annotation_engines=engines,
            star_genome=given.get("--star-genome"), reference=given.get("--reference"),
            annovar_db=given.get("--annovar-db"), annovar_buildver=given.get("--annovar-buildver"),
            annovar_protocol=listed("--annovar-protocol", "annovar_protocol"),
            annovar_operation=listed("--annovar-operation", "annovar_operation"),
            snpeff_jar=given.get("--snpeff-jar"), snpeff_config=given.get("--snpeff-config"),
            snpeff_genome=given.get("--snpeff-genome"),
        )
        if not lacking:
            return

    try:
        from paralome.config.user import load_config

        names = sorted(str(name) for name in (load_config(initialize=False).get("genomes") or {}))
    except ParalomeError:
        # A key nothing reads is refused by name, here as everywhere.
        raise
    except Exception:
        names = []
    raise ValueError(
        f"--genome is required: the {analysis} run for {project} names no genome, and "
        "Paralome never chooses one. Pass --genome to `batch run`, or put it in that "
        "family's pipeline_args"
        + (" (a DNA family may instead give every path: --star-genome, --reference, and "
           "--annovar-db with --annovar-buildver, --annovar-protocol and "
           "--annovar-operation, or --snpeff-jar with --snpeff-config "
           "and --snpeff-genome)." if analysis == "DNA" else ".")
        + (f" This command gives genome paths but not {', '.join(lacking)}."
           + (" The two ANNOVAR lists may instead be annotation.annovar_protocol and "
              "annotation.annovar_operation in your configuration."
              if any(o in lacking for o in ("--annovar-protocol", "--annovar-operation")) else "")
           if lacking else "")
        + (f" Configured genomes: {', '.join(names)}" if names else "")
    )


def build_batch_plan(
    *,
    family_list: Path,
    out: Path,
    max_families: int,
    threads: int,
    jobs: int,
    profile: str,
    genome: Optional[str],
    pipeline_config: Optional[Path],
    partition: Optional[str],
    poll_seconds: int,
    pipeline_args: Sequence[str],
    name: str,
    paralome_executable: Optional[Path] = None,
) -> BatchPlan:
    if max_families < 1:
        raise ValueError("max_families must be at least 1")
    if threads < 1:
        raise ValueError("threads must be at least 1")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if poll_seconds < 1:
        raise ValueError("poll_seconds must be at least 1")

    global_args = tuple(pipeline_args)
    _validate_extra_args(global_args, source="--pipeline-args")
    families = read_family_list(family_list)
    out = out.expanduser().resolve()
    logs_dir = out / "logs"
    script = out / "scripts" / "family_array.sbatch"
    argv_prefix = _paralome_argv(paralome_executable)

    # THE GROUP IS THE VERB. A family run is `paralome scRNA --samples ...`, exactly what a person
    # would type: the profile is the command name, not a --profile option. Validating it here means a
    # bad profile fails when the plan is built, on the login node, rather than 300 array tasks later.
    analysis = normalize_pipeline_profile(profile).value
    # Refuse a profile whose command lacks an option every family command carries
    # (scRNA and snATAC take no --samples), before the plan is written or submitted.
    from paralome.pipeline.assay_options import options_for

    emitted = ["samples", "project", "backend", "submit", "threads", "jobs", "poll_seconds"]
    emitted += ["genome"] * bool(genome) + ["pipeline_config"] * (pipeline_config is not None)
    emitted += ["partition"] * bool(partition)
    unsupported = [name for name in emitted if name not in options_for(analysis)]
    if unsupported:
        raise ValueError(
            f"batch cannot run the {analysis} profile: `{analysis}` has no "
            + ", ".join("--" + name.replace("_", "-") for name in unsupported)
            + ", which every family command passes. Use DNA, RNA or ATAC, or run each family's "
            f"{analysis} command yourself"
        )

    commands: List[Tuple[str, ...]] = []
    for family in families:
        command: List[str] = [
            *argv_prefix,
            analysis,
            "--samples",
            str(family.samples),
            "--project",
            str(family.project),
            "--backend",
            "slurm",
            "--submit",
            "--threads",
            str(threads),
            "--jobs",
            str(jobs),
            "--poll-seconds",
            str(poll_seconds),
        ]
        if genome:
            command.extend(("--genome", genome))
        if pipeline_config is not None:
            command.extend(("--pipeline-config", str(pipeline_config.expanduser().resolve())))
        if partition:
            command.extend(("--partition", partition))
        # No --mem: memory is per stage (--stage-mem). Per-stage memory goes through --pipeline-args.
        command.extend(global_args)
        command.extend(family.pipeline_args)
        _require_a_genome_in(
            command, analysis=analysis, project=family.project,
            pipeline_config=pipeline_config,
        )
        commands.append(tuple(command))

    return BatchPlan(
        name=name,
        family_list=family_list.expanduser().resolve(),
        workdir=out,
        max_families=max_families,
        commands=tuple(commands),
        script=script,
        logs_dir=logs_dir,
    )


def write_batch_plan(plan: BatchPlan) -> Path:
    plan.workdir.mkdir(parents=True, exist_ok=True)
    plan.logs_dir.mkdir(parents=True, exist_ok=True)
    plan.script.parent.mkdir(parents=True, exist_ok=True)
    path = plan.workdir / "batch_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_batch_script(
    plan: BatchPlan,
    *,
    plan_path: Path,
    partition: Optional[str],
    supervisor_mem: str,
) -> str:
    # SLURM's #SBATCH directive parser tokenizes each line on whitespace like a shell command line, so
    # a value containing a literal space (a family/plan name or logs directory derived from a name like
    # "CD14 Mono") would silently corrupt the directive -- no error, no job ID, no log file. Quote
    # every value that isn't already known-safe (as pipeline/slurm.py's SlurmOptions.header() does).
    def q(value: object) -> str:
        return shlex.quote(str(value))

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={q(plan.name)}",
        f"#SBATCH --array=1-{len(plan.commands)}%{plan.max_families}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={q(supervisor_mem)}",
        f"#SBATCH --output={q(str(plan.logs_dir) + '/family_%A_%a.out')}",
        f"#SBATCH --error={q(str(plan.logs_dir) + '/family_%A_%a.err')}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={q(partition)}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f"PLAN={shlex.quote(str(plan_path))}",
            'INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"',
            "",
            # The worker binary through the resolver, so the job does not depend on the compute
            # node's PATH.
            f'exec {worker_command()} batch worker --plan "$PLAN" --index "$INDEX"',
            "",
        ]
    )
    return "\n".join(lines)


def _submit_script(script: Path) -> str:
    completed = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        check=True,
        text=True,
        capture_output=True,
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse SLURM job ID from sbatch output: {completed.stdout!r}")
    return job_id


@batch_app.command("run")
def batch_run_command(
    family_list: Path = typer.Option(..., "--family-list", help="CSV with project/family and samples/manifest columns"),
    out: Path = typer.Option(Path("paralome_batch"), "--out", help="Batch planning, script, and log directory"),
    max_families: int = typer.Option(3, "--max-families", min=1, help="Maximum complete family pipelines admitted concurrently"),
    threads: int = typer.Option(30, "--threads", min=1, help="Per-family pipeline CPU budget"),
    jobs: int = typer.Option(30, "--jobs", min=1, help="Per-family stage concurrency/array throttle"),
    profile: str = typer.Option("DNA", "--profile", help="Pipeline profile passed to every family"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Configured genome passed to every family. Required, unless every family names its own in pipeline_args"),
    pipeline_config: Optional[Path] = typer.Option(None, "--pipeline-config", help="Pipeline defaults JSON passed to every family"),
    partition: Optional[str] = typer.Option(None, "--partition", help="SLURM partition for family supervisors and child stages"),
    supervisor_mem: str = typer.Option("1G", "--supervisor-mem", help="Memory reserved by each outer family supervisor"),
    poll_seconds: int = typer.Option(60, "--poll-seconds", min=1, help="Supervisor poll interval within each family"),
    pipeline_args: Optional[str] = typer.Option(None, "--pipeline-args", help="Additional common pipeline-run arguments as one quoted string, e.g. '--stage-mem alignment=40G'"),
    name: str = typer.Option("paralome_family", "--name", help="SLURM job name for the family admission array"),
    submit: bool = typer.Option(True, "--submit/--no-submit", help="Submit the family array after generating it"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate and print the plan without submitting"),
) -> None:
    """Run many complete family pipelines while admitting only N at a time."""
    try:
        extra_args = tuple(shlex.split(pipeline_args)) if pipeline_args else ()
        plan = build_batch_plan(
            family_list=family_list,
            out=out,
            max_families=max_families,
            threads=threads,
            jobs=jobs,
            profile=profile,
            genome=genome,
            pipeline_config=pipeline_config,
            partition=partition,
            poll_seconds=poll_seconds,
            pipeline_args=extra_args,
            name=name,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    plan_path = write_batch_plan(plan)
    plan.script.write_text(
        render_batch_script(
            plan,
            plan_path=plan_path,
            partition=partition,
            supervisor_mem=supervisor_mem,
        ),
        encoding="utf-8",
    )
    plan.script.chmod(0o755)

    print("Paralome family batch")
    print("======================")
    print(f"Families             : {len(plan.commands)}")
    print(f"Maximum active       : {plan.max_families}")
    print(f"Per-family threads   : {threads}")
    print(f"Per-family jobs      : {jobs}")
    print(f"Plan                 : {plan_path}")
    print(f"Script               : {plan.script}")
    print(f"Array                : 1-{len(plan.commands)}%{plan.max_families}")

    if dry_run or not submit:
        print("Submitted             : False")
        return

    job_id = _submit_script(plan.script)
    print("Submitted             : True")
    print(f"SLURM job             : {job_id}")


@batch_worker_app.command("worker")
def batch_worker_command(
    plan: Path = typer.Option(..., "--plan"),
    index: Optional[int] = typer.Option(None, "--index", min=1),
) -> None:
    """Execute one family command from a persisted batch plan."""
    data = json.loads(plan.expanduser().resolve().read_text(encoding="utf-8"))
    commands = data.get("commands")
    if not isinstance(commands, list) or not commands:
        raise typer.BadParameter(f"Batch plan contains no commands: {plan}")
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        if not raw.isdigit():
            raise typer.BadParameter("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(raw)
    if index < 1 or index > len(commands):
        raise typer.BadParameter(f"Batch task index {index} is outside 1-{len(commands)}")

    command = commands[index - 1]
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
        raise typer.BadParameter(f"Invalid command at batch index {index}")

    print(f"Batch family index   : {index}", flush=True)
    print("Command              : " + " ".join(shlex.quote(token) for token in command), flush=True)
    completed = subprocess.run(command)
    raise typer.Exit(code=completed.returncode)
