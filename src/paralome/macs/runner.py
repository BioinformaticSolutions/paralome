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
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from paralome.binaries import worker_argv
from paralome.checkpoints import stage_checkpoint_path
from paralome.fastq.manifest import read_sample_inputs
from paralome.pipeline.naming import slurm_job_name
from paralome.pipeline.profiles import ManifestProfile
from paralome.pipeline.slurm import SlurmOptions, placeable_cpus, tmp_directive, write_sbatch
from paralome.scratch import job_scratch_bash, job_scratch_dir

SCHEMA_PLAN = "paralome.macs.plan.v1"
SCHEMA_CHECKPOINT = "paralome.macs.checkpoint.v1"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def find_macs3() -> Tuple[Optional[str], Optional[str]]:
    path = shutil.which("macs3")
    return ("macs3", path) if path else (None, None)


def find_macs() -> Tuple[Optional[str], Optional[str]]:
    for exe in ("macs3", "macs2"):
        path = shutil.which(exe)
        if path:
            return exe, path
    return None, None


def macs_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def check_macs() -> bool:
    exe, path = find_macs()
    if exe is None:
        print("MACS not found.")
        print("Install MACS3 before running Paralome ATAC peak calling.")
        return False
    print("MACS found")
    print("Executable : %s" % exe)
    print("Path       : %s" % path)
    print("Version    : %s" % macs_version(exe))
    if exe != "macs3":
        print("Warning    : peak calling requires macs3; macs2 is detected only for compatibility checks")
    return True


@dataclass(frozen=True)
class MacsSample:
    sample_id: str
    group: str
    bam: str
    output_dir: str
    narrowpeak: str
    summits: str
    log: str


#: MACS3 is given the BAM without the reads samtools markdup flagged
#: as duplicates (flag 1024). Duplicate marking only flags them by default, and
#: MACS3's BAMPE reader does not drop flag 1024 (it drops unmapped, secondary,
#: QC-fail and supplementary reads), so with --keep-dup all every PCR duplicate
#: was piled up: PCR-jackpot peaks entered the atlas, and calling disagreed with
#: peak quantification, which excludes duplicates. --keep-dup all is then right:
#: what is left has no PCR duplicates.
DUPLICATE_FLAG = 1024
DUPLICATES = "excluded: samtools view -F 1024 before MACS3"


def nodup_bam(sample: MacsSample) -> Path:
    """The duplicate-free BAM MACS3 reads; removed once MACS3 has run."""
    return Path(sample.output_dir) / f"{sample.sample_id}.nodup.bam"


def peaks_marker(sample: MacsSample) -> Path:
    """Written beside the peaks once MACS3 has run on the duplicate-free BAM.

    Peaks without it were called with the PCR duplicates and are called again.
    """
    return Path(sample.output_dir) / f"{sample.sample_id}.paralome_macs.json"


def ensure_dirs(workdir: Path) -> None:
    for name in ("inputs", "peaks", "logs", "checkpoints", "results"):
        (workdir / name).mkdir(parents=True, exist_ok=True)


def load_samples(samples_csv: Union[str, Path], workdir: Path) -> List[MacsSample]:
    records = read_sample_inputs(
        Path(samples_csv),
        manifest_profile=ManifestProfile.GENERAL,
        validate_schema=True,
    )
    samples: List[MacsSample] = []
    for record in records:
        if not record.bam:
            raise ValueError("Sample %s has an empty BAM path" % record.sample_id)
        bam = Path(record.bam)
        if not bam.exists():
            raise FileNotFoundError("BAM for sample %s not found: %s" % (record.sample_id, bam))
        condition = record.condition if record.condition not in ("", ".") else record.group
        condition = condition if condition not in ("", ".") else "unknown"
        sample_dir = workdir / "peaks" / record.sample_id
        prefix = sample_dir / record.sample_id
        samples.append(MacsSample(
            sample_id=record.sample_id,
            group=condition,
            bam=str(bam),
            output_dir=str(sample_dir),
            narrowpeak=str(prefix) + "_peaks.narrowPeak",
            summits=str(prefix) + "_summits.bed",
            log=str(workdir / "logs" / (record.sample_id + ".macs3.log")),
        ))
    return samples


#: MACS asked for no memory at all, so SLURM gave each task a whole
#: node's worth on a CR_CPU_MEMORY cluster and the array ran one sample at a
#: time. MACS3 itself is modest; this covers it and the samtools copy beside it.
#: Every other stage already had a default (fastp 8G, dedup 16G, STAR 32G).
DEFAULT_MEM = "8G"
#: The duplicate-free copy is what takes the time, not callpeak: one thread
#: writing a fully compressed BAM took 27-35 minutes on a 46 M fragment sample
#: while MACS3 then read it in 9. These threads go to `samtools view`; MACS3 is
#: single-threaded and unaffected.
DEFAULT_THREADS = 4
#: The scratch space guard is given this multiple of the source BAM, or it
#: would let a job start on a disk that cannot hold what it is about to write.
#: The copy is written with `samtools view -1` (fast compression). Measured on
#: coordinate-sorted paired-end BAMs at samtools' default level, 50 and 150 bp
#: reads, binned (NovaSeq) and full-range qualities: the -1 copy is 0.83-1.04x
#: the source with a quarter of the reads flagged duplicate and 1.11-1.38x with
#: none dropped (binned qualities, the worst case). Two covers the worst case
#: with room to spare.
COPY_BAM_FACTOR = 2


def build_callpeak_command(sample: MacsSample, genome_size: str, qvalue: float,
                           keep_dup: str = "all", extra_options: Optional[str] = None,
                           executable: str = "macs3", treatment: Optional[str] = None) -> List[str]:
    """The MACS3 call; ``treatment`` is the duplicate-free BAM (nodup_bam) a run gives it."""
    command = [
        executable, "callpeak", "--treatment", treatment or sample.bam,
        "--format", "BAMPE", "--name", sample.sample_id,
        "--outdir", sample.output_dir, "--gsize", str(genome_size),
        "--qvalue", str(qvalue), "--call-summits", "--keep-dup", str(keep_dup),
    ]
    if extra_options:
        command.extend(shlex.split(extra_options))
    return command


def macs_parameters(genome_size: str, qvalue: float, keep_dup: str = "all",
                    extra_options: Optional[str] = None) -> Dict[str, Any]:
    """The MACS3 settings recorded in each sample's marker and compared before peaks are reused."""
    return {"genome_size": str(genome_size), "qvalue": float(qvalue), "keep_dup": str(keep_dup),
            "extra_options": shlex.split(extra_options or "")}


def parameter_mismatch(sample: MacsSample, parameters: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why the sample's peaks were not called with ``parameters`` (None when they were, or not asked).

    Peaks whose marker records no parameters do not count as matching.
    """
    if parameters is None:
        return None
    try:
        recorded = json.loads(peaks_marker(sample).read_text()).get("parameters")
    except (OSError, ValueError, AttributeError):
        return "its marker is unreadable"
    if not isinstance(recorded, dict):
        return "the peaks record no MACS3 parameters (called by an older Paralome)"
    changed = [f"{key} {recorded.get(key)!r} -> {value!r}" for key, value in parameters.items()
               if recorded.get(key) != value]
    return ("called with " + ", ".join(changed)) if changed else None


def outputs_complete(sample: MacsSample, parameters: Optional[Dict[str, Any]] = None) -> bool:
    return peaks_marker(sample).is_file() and all(
        Path(path).exists() and Path(path).stat().st_size > 0 for path in (sample.narrowpeak, sample.summits)
    ) and parameter_mismatch(sample, parameters) is None


def _incomplete_reason(sample: MacsSample, parameters: Optional[Dict[str, Any]] = None) -> str:
    """Why a sample's peaks are not complete (named in the errors)."""
    if not peaks_marker(sample).is_file():
        return f"samtools or MACS3 did not finish; see {sample.log}"
    mismatch = parameter_mismatch(sample, parameters)
    if mismatch:
        return f"MACS3 settings differ from the plan: {mismatch}; run `macs call` again"
    narrowpeak = Path(sample.narrowpeak)
    if narrowpeak.is_file() and narrowpeak.stat().st_size == 0:
        return "MACS3 ran and called zero peaks"
    return "MACS3 outputs missing or empty"


def _drop_stale_nodup(sample: MacsSample) -> None:
    """A killed task can leave its duplicate-free BAM beside the peaks.

    Removed only once the sample's marker says MACS3 has finished.
    """
    if peaks_marker(sample).is_file():
        nodup_bam(sample).unlink(missing_ok=True)


def build_dedup_command(sample: MacsSample, treatment: str, samtools: str = "samtools",
                        threads: int = DEFAULT_THREADS) -> List[str]:
    """The samtools step that writes the duplicate-free BAM MACS3 reads.

    Threaded, and ``-1`` (fast compression), not ``-u`` (none) and not the
    default level: the file is read once and deleted, so compression is
    almost all of its cost, while an uncompressed copy is up to six times the
    source BAM on scratch. Level 1 keeps it about the size of the source (see
    COPY_BAM_FACTOR) at a fraction of the default level's time. MACS3 reads
    either the same way.
    """
    return [samtools, "view", "-@", str(max(1, int(threads))), "-1",
            "-F", str(DUPLICATE_FLAG), "-o", treatment, sample.bam]


def _size(path: Union[str, Path]) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _read_checkpoint(workdir: Path) -> Dict[str, Any]:
    """The stage checkpoint as written, or {} when absent or unreadable."""
    try:
        data = json.loads(stage_checkpoint_path(workdir, "macs_peak_calling").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(path: Path, samples: Sequence[MacsSample],
                    parameters: Optional[Dict[str, Any]] = None) -> None:
    fields = ["sample_id", "condition", "group", "bam", "narrowpeak", "summits", "macs_log", "status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "sample_id": sample.sample_id, "condition": sample.group, "group": sample.group, "bam": sample.bam,
                "narrowpeak": sample.narrowpeak, "summits": sample.summits,
                "macs_log": sample.log, "status": "complete" if outputs_complete(sample, parameters) else "incomplete",
            })


def run_macs_stage(samples_csv: Union[str, Path], workdir: Union[str, Path],
                   genome_size: str = "hs", qvalue: float = 0.01,
                   keep_dup: str = "all", extra_options: Optional[str] = None,
                   overwrite: bool = False, dry_run: bool = False, backend: str = "local",
                   submit: bool = False, jobs: int = 0, partition: Optional[str] = None,
                   mem: str = DEFAULT_MEM, threads: int = DEFAULT_THREADS) -> Dict[str, Any]:
    workdir = Path(workdir)
    # A dry run writes nothing, not even the folders.
    if not dry_run:
        ensure_dirs(workdir)
    samples = load_samples(samples_csv, workdir)
    exe, exe_path = find_macs3()
    if not dry_run and exe is None:
        raise RuntimeError("Required executable not found in PATH: macs3")
    executable = exe or "macs3"
    samtools = shutil.which("samtools")
    if not dry_run and samtools is None:
        raise RuntimeError("Required executable not found in PATH: samtools (MACS3 is given the BAM without duplicates)")
    samtools = samtools or "samtools"
    parameters = macs_parameters(genome_size, qvalue, keep_dup, extra_options)
    commands = {
        sample.sample_id: build_callpeak_command(sample, genome_size, qvalue, keep_dup, extra_options, executable,
                                                 treatment=str(nodup_bam(sample)))
        for sample in samples
    }
    plan = {
        "schema": SCHEMA_PLAN, "stage": "macs_peak_calling", "created": now_iso(),
        "workdir": str(workdir), "samples_manifest": str(samples_csv), "samples": len(samples),
        "macs3_path": exe_path, "macs3_version": macs_version(executable) if exe else None,
        "genome_size": genome_size, "qvalue": qvalue, "keep_dup": keep_dup, "duplicates": DUPLICATES,
        "extra_options": extra_options, "commands": commands,
        "parameters": parameters,
        # The samtools step that runs before each MACS3 call.
        "dedup_commands": {
            sample.sample_id: build_dedup_command(sample, str(nodup_bam(sample)), samtools) for sample in samples
        },
        "sample_outputs": {s.sample_id: asdict(s) for s in samples},
    }
    if dry_run:
        return {**plan, "dry_run": True}
    _write_json(workdir / "inputs" / "macs_plan.json", plan)

    if backend == "slurm":
        # #SBATCH lines are not shell-parsed, so paths there are
        # written unquoted (as write_sbatch does) and whitespace is refused.
        if any(ch.isspace() for ch in str(workdir)):
            raise ValueError(
                f"The MACS output folder contains whitespace, which SLURM cannot take in "
                f"#SBATCH --output/--error: {workdir}. Use a folder path without spaces."
            )
        manifest_tsv = workdir / "inputs" / "macs_samples.tsv"
        with manifest_tsv.open("w") as handle:
            handle.write("sample_id\tbam\toutdir\tlog\n")
            for sample in samples:
                handle.write(f"{sample.sample_id}\t{sample.bam}\t{sample.output_dir}\t{sample.log}\n")
        concurrency = min(len(samples), jobs or len(samples))
        script = workdir / "inputs" / "macs_array.sbatch"
        # Every SLURM job is named for its project and stage, so the array
        # and the finalize job can be told apart from other projects' jobs.
        array_job_name = slurm_job_name(workdir, "macs")
        finalize_job_name = slurm_job_name(workdir, "macs_finalize")
        lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name={array_job_name}", f"#SBATCH --cpus-per-task={placeable_cpus(threads, partition, job_name=array_job_name)}", f"#SBATCH --array=1-{len(samples)}%{concurrency}", f"#SBATCH --output={workdir}/logs/macs_%A_%a.out", f"#SBATCH --error={workdir}/logs/macs_%A_%a.err"]
        if partition: lines.append(f"#SBATCH --partition={partition}")
        lines.append(f"#SBATCH --mem={mem}")
        # The duplicate-free copy goes to node scratch; SLURM is
        # asked for the largest sample's share of it (see paralome.scratch).
        tmp_line = tmp_directive(max((_size(sample.bam) for sample in samples), default=0) * COPY_BAM_FACTOR,
                                 partition)
        if tmp_line:
            lines.append(tmp_line.rstrip("\n"))
        lines += ["set -euo pipefail", f"M={shlex.quote(str(manifest_tsv))}", "LINE=$(awk -v n=\"$SLURM_ARRAY_TASK_ID\" 'NR==n+1 {print}' \"$M\")", "SAMPLE=$(echo \"$LINE\" | cut -f1)", "BAM=$(echo \"$LINE\" | cut -f2)", "OUTDIR=$(echo \"$LINE\" | cut -f3)", "LOG=$(echo \"$LINE\" | cut -f4)", "mkdir -p \"$OUTDIR\"", job_scratch_bash('"$OUTDIR"', "macs-nodup", var="NODUP_DIR", need_bytes='$(( $(stat -c %s "$BAM" 2>/dev/null || echo 0) * COPY_FACTOR ))'.replace("COPY_FACTOR", str(COPY_BAM_FACTOR))), "NODUP=\"$NODUP_DIR/$SAMPLE.nodup.bam\"", "rm -f \"$OUTDIR/$SAMPLE.paralome_macs.json\"", "paralome_on_exit rm -f -- \"$NODUP\"", f"samtools view -@ {max(1, int(threads))} -1 -F {DUPLICATE_FLAG} -o \"$NODUP\" \"$BAM\" >\"$LOG\" 2>&1", f"macs3 callpeak --treatment \"$NODUP\" --format BAMPE --name \"$SAMPLE\" --outdir \"$OUTDIR\" --gsize {shlex.quote(str(genome_size))} --qvalue {qvalue} --call-summits --keep-dup {shlex.quote(str(keep_dup))} {extra_options or ''} >>\"$LOG\" 2>&1", "printf '%s\\n' " + shlex.quote(json.dumps({"duplicates": DUPLICATES, "parameters": parameters})) + " > \"$OUTDIR/$SAMPLE.paralome_macs.json\""]
        script.write_text("\n".join(lines) + "\n")
        checkpoint = {"schema": SCHEMA_CHECKPOINT, "stage": "macs_peak_calling", "updated": now_iso(), "complete": False, "submitted": submit, "samples": len(samples), "peak_manifest": str(workdir / "results" / "macs_peaks.csv")}
        # Writing scripts without --submit runs nothing, so a
        # finished stage whose peaks match this sheet and these settings keeps
        # its complete checkpoint; it read incomplete and was called again.
        previous = _read_checkpoint(workdir)
        keep_finished = (
            not submit and previous.get("complete") is True and previous.get("samples") == len(samples)
            and all(outputs_complete(sample, parameters) for sample in samples)
        )
        if keep_finished:
            checkpoint = previous
        else:
            _write_json(stage_checkpoint_path(workdir, "macs_peak_calling"), checkpoint)
        # The finalizer runs the worker, with logs, a partition and
        # memory like every other stage's. It called `paralome macs finalize`;
        # `macs` exists only on the worker, so the job died with "No such
        # command 'macs'" every time and bulk ATAC stopped at this stage.
        final = write_sbatch(
            workdir / "inputs" / "macs_finalize.sbatch",
            SlurmOptions(
                job_name=finalize_job_name,
                cpus_per_task=1,
                mem="2G",
                partition=partition,
                output=workdir / "logs" / "macs_finalize_%j.out",
                error=workdir / "logs" / "macs_finalize_%j.err",
            ),
            shlex.join([*worker_argv(), "macs", "finalize", "--samples", str(samples_csv), "--out", str(workdir)]),
        )
        if submit:
            proc = subprocess.run(["sbatch", "--parsable", str(script)], text=True, capture_output=True, check=True)
            job_id = proc.stdout.strip().split(";")[0]
            # Named and recorded before the finalizer is submitted, so a refused finalizer never leaves
            # the array queued and unnamed. stderr, which the supervisor reads too; stdout carries this
            # stage's JSON.
            print(f"Submitted MACS3 array job: {job_id}", file=sys.stderr, flush=True)
            checkpoint["submitted_job_ids"] = [job_id]
            _write_json(stage_checkpoint_path(workdir, "macs_peak_calling"), checkpoint)
            dep = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{job_id}", str(final)], text=True, capture_output=True, check=True).stdout.strip().split(";")[0]
            print(f"Submitted MACS3 job: {job_id},{dep}")
            checkpoint["submitted_job_ids"] = [job_id, dep]
            _write_json(stage_checkpoint_path(workdir, "macs_peak_calling"), checkpoint)
        return checkpoint

    failures: List[str] = []
    reused: List[str] = []
    recalled: Dict[str, str] = {}
    if not dry_run:
        for sample in samples:
            Path(sample.output_dir).mkdir(parents=True, exist_ok=True)
            if outputs_complete(sample) and not overwrite:
                # Peaks are reused only when called with the settings asked for now.
                mismatch = parameter_mismatch(sample, parameters)
                if mismatch is None:
                    _drop_stale_nodup(sample)
                    reused.append(sample.sample_id)
                    continue
                recalled[sample.sample_id] = mismatch
                print(f"MACS3 peaks for {sample.sample_id} were {mismatch}; calling them again",
                      file=sys.stderr, flush=True)
            log_path = Path(sample.log)
            peaks_marker(sample).unlink(missing_ok=True)
            # The duplicate-free BAM on this node's scratch when the site has one (removed with it);
            # otherwise beside the peaks, removed once MACS3 has run.
            try:
                need = os.path.getsize(sample.bam) * COPY_BAM_FACTOR
            except OSError:
                need = 0
            with job_scratch_dir(Path(sample.output_dir), "macs-nodup", need_bytes=need) as nodup_dir:
                nodup = nodup_dir / nodup_bam(sample).name
                command = build_callpeak_command(sample, genome_size, qvalue, keep_dup, extra_options, executable,
                                                 treatment=str(nodup))
                try:
                    with log_path.open("w") as log_handle:
                        proc = subprocess.run(
                            build_dedup_command(sample, str(nodup), samtools),
                            stdout=log_handle, stderr=subprocess.STDOUT, text=True, check=False,
                        )
                        if proc.returncode == 0:
                            proc = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                                                  check=False)
                finally:
                    nodup.unlink(missing_ok=True)
            if proc.returncode == 0:
                _write_json(peaks_marker(sample), {"duplicates": DUPLICATES, "parameters": parameters})
                _drop_stale_nodup(sample)
            if proc.returncode != 0 or not outputs_complete(sample, parameters):
                failures.append(sample.sample_id)

    manifest = workdir / "results" / "macs_peaks.csv"
    _write_manifest(manifest, samples, parameters)
    complete_count = sum(outputs_complete(sample, parameters) for sample in samples)
    complete = bool(samples) and complete_count == len(samples) and not failures
    checkpoint = {
        "schema": SCHEMA_CHECKPOINT, "stage": "macs_peak_calling", "updated": now_iso(),
        "complete": complete, "dry_run": dry_run, "samples": len(samples),
        "completed_samples": complete_count, "failed_samples": failures,
        "reused_samples": reused, "recalled_samples": recalled, "peak_manifest": str(manifest),
    }
    _write_json(stage_checkpoint_path(workdir, "macs_peak_calling"), checkpoint)
    if failures:
        # Each sample with its reason; zero peaks is not a MACS3 failure.
        raise RuntimeError("MACS3 peaks incomplete for sample(s): %s" % ", ".join(
            f"{sample.sample_id} ({_incomplete_reason(sample, parameters)})" for sample in samples
            if sample.sample_id in failures
        ))
    return checkpoint


def finalize_macs_stage(samples_csv: Union[str, Path], workdir: Union[str, Path]) -> Dict[str, Any]:
    workdir = Path(workdir)
    samples = load_samples(samples_csv, workdir)
    manifest = workdir / "results" / "macs_peaks.csv"
    # Peaks must have been called with the settings the plan records.
    try:
        parameters = json.loads((workdir / "inputs" / "macs_plan.json").read_text()).get("parameters")
    except (OSError, ValueError, AttributeError):
        parameters = None
    for sample in samples:
        _drop_stale_nodup(sample)
    _write_manifest(manifest, samples, parameters)
    completed = sum(outputs_complete(sample, parameters) for sample in samples)
    checkpoint = {"schema": SCHEMA_CHECKPOINT, "stage": "macs_peak_calling", "updated": now_iso(), "complete": bool(samples) and completed == len(samples), "samples": len(samples), "completed_samples": completed, "peak_manifest": str(manifest)}
    _write_json(stage_checkpoint_path(workdir, "macs_peak_calling"), checkpoint)
    if not checkpoint["complete"]:
        # Names each incomplete sample and why.
        incomplete = [
            f"{sample.sample_id} ({_incomplete_reason(sample, parameters)})" for sample in samples
            if not outputs_complete(sample, parameters)
        ]
        raise RuntimeError(
            f"MACS3 outputs incomplete: {completed}/{len(samples)}; " + (", ".join(incomplete) or "no samples")
        )
    return checkpoint
