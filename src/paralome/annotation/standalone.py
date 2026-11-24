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
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paralome.checkpoints import stage_checkpoint_path
from paralome.annotation.annovar import (
    SAME_ALLELE_JOIN_AWK, AnnovarConfig, info_id_rename_bash, normalize_for_annotation_bash,
)
from paralome.annotation.snpeff import SnpEffConfig
from paralome.pipeline.naming import slurm_job_name
from paralome.pipeline.resources import plan_sample_parallel


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")




@dataclass(frozen=True)
class AnnotationRunResult:
    engine: str
    backend: str
    workdir: Path
    input_vcf: Path
    output_vcf: Path
    script: Path
    submitted: bool = False
    job_id: Optional[str] = None
    complete: bool = False
    plan: Optional[Path] = None
    checkpoint: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "backend": self.backend,
            "workdir": str(self.workdir),
            "input_vcf": str(self.input_vcf),
            "output_vcf": str(self.output_vcf),
            "script": str(self.script),
            "submitted": self.submitted,
            "job_id": self.job_id,
            "complete": self.complete,
            "plan": str(self.plan) if self.plan else None,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
        }


def _run_prefix(workdir: Path, prefix: Optional[str] = None) -> str:
    if prefix and prefix.strip():
        return prefix.strip()
    if workdir.name in {"annotation", "annovar"}:
        return "annotation"
    return workdir.name


def _ensure_dirs(workdir: Path) -> None:
    for name in ("inputs", "logs", "scripts", "results", "temp", "checkpoints"):
        (workdir / name).mkdir(parents=True, exist_ok=True)


def _require_tool(name: str) -> str:
    """The tool to invoke, from what setup recorded -- not from PATH.

    See binaries.resolve_tool. Returns the resolved absolute path so
    the caller can invoke THAT rather than the bare name it asked about; a
    recorded path that no longer resolves raises PLM-0410 instead of silently
    falling back to whatever this account's PATH happens to find.
    """
    from paralome.binaries import resolve_tool

    exe = resolve_tool(name)
    if exe is None:
        raise FileNotFoundError(f"Required executable not found (recorded path or PATH): {name}")
    return exe


def recorded_tools_bash(*names: str) -> str:
    """Bash functions that pin each tool to the path setup recorded.

    The checks before submission use the recorded path (binaries.tool_path);
    the generated scripts called bare names, so a job ran PATH's binary, or
    stopped at `require_tool` when the recorded one was not on PATH. A tool
    with nothing recorded is left to PATH, as before.
    """
    from paralome.binaries import tool_path

    lines = []
    for name in names:
        exe = tool_path(name)
        if not exe:
            continue
        q = shlex.quote(exe)
        lines.append(f"if [[ ! -x {q} ]]; then echo {shlex.quote(f'ERROR: {name} recorded at {exe} is not executable on this node')} >&2; exit 1; fi")
        lines.append(f"{name}() {{ {q} \"$@\"; }}")
    return "\n".join(lines)


_PINNED_TOOLS = ("bcftools", "bgzip", "tabix", "java", "perl")


def _absolute(path: Optional[Path]) -> Optional[Path]:
    """Every path written into a job script is absolute.

    A job then finds its files wherever SLURM starts it. os.path.abspath, not
    resolve(): the folder names the user gave are kept, symlinks and all.
    """
    if path is None:
        return None
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _info_id_map_path(workdir: Path, engine: str) -> Path:
    """Where the INFO ID renames of one annotation run are gathered."""
    return workdir / "temp" / f"{engine}_info_id_renames.tsv"


def _checkpoint_python(
    workdir: Path,
    engine: str,
    output_vcf: Path,
    input_vcf: Path,
    info_id_map: Optional[Path] = None,
    duplicates_file: Optional[Path] = None,
    joined_file: Optional[Path] = None,
) -> str:
    # Stage_checkpoint_path is imported INSIDE the heredoc: this block runs in its own python3 and
    # never sees the module's import.
    # The split the VCF was made with (annovar.SPLIT_METHOD), when
    # ANNOVAR is in the chain -- only then are the calls split. With it the
    # pipeline can tell an annotation made before a change to the split from
    # one made after.
    from paralome.annotation.annovar import SPLIT_METHOD

    split_method = SPLIT_METHOD if "annovar" in str(engine).split("_") else None
    # The submitting install's interpreter, not PATH's python3
    # (which may not have paralome), as align/star.py does.
    return f'''{shlex.quote(sys.executable)} - <<'PYCHECKPOINT'
import json
from datetime import datetime
from pathlib import Path
from paralome.checkpoints import stage_checkpoint_path
workdir = Path({str(workdir)!r})
checkpoint = stage_checkpoint_path(workdir, f"annotation_{engine}")
info_id_map = {str(info_id_map) if info_id_map else None!r}
duplicates_file = {str(duplicates_file) if duplicates_file else None!r}
joined_file = {str(joined_file) if joined_file else None!r}
duplicates = None
if duplicates_file and Path(duplicates_file).exists():
    text = Path(duplicates_file).read_text().strip()
    duplicates = int(text) if text.isdigit() else None
joined = apart = None
if joined_file and Path(joined_file).exists():
    words = Path(joined_file).read_text().split()
    if len(words) == 2 and all(w.isdigit() for w in words):
        joined, apart = int(words[0]), int(words[1])
renames = {{}}
if info_id_map and Path(info_id_map).exists():
    for line in Path(info_id_map).read_text().splitlines():
        old, _, new = line.partition("\t")
        if old and new:
            renames[old] = new
state = {{
  "stage": "annotation",
  "engine": {engine!r},
  "complete": True,
  "started": None,
  "finished": datetime.now().isoformat(timespec="seconds"),
  "input_vcf": {str(input_vcf)!r},
  "output_vcf": {str(output_vcf)!r},
  "info_id_renames": renames,
  "split_method": {split_method!r},
}}
if duplicates_file:
    # Exact duplicate records (CHROM, POS, REF, ALT and the sample columns) the merge of the shards
    # dropped -- see _stream_merge_bash.
    state["duplicate_records_removed"] = duplicates
if joined_file:
    # Records of one allele whose sample columns differed, joined
    # into one record (joined) or left as two because their FORMAT differed
    # (kept apart) -- see _stream_merge_bash.
    state["same_allele_records_joined"] = joined
    state["same_allele_records_kept_apart"] = apart
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text(json.dumps(state, indent=2, sort_keys=True) + "\\n")
project = workdir / "project_state.json"
if project.exists():
    data = json.loads(project.read_text())
else:
    data = {{}}
data.setdefault("annotation", {{}})[{engine!r}] = state
data["updated"] = datetime.now().isoformat(timespec="seconds")
project.write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n")
PYCHECKPOINT'''


def _common_script_header() -> str:
    return '''#!/usr/bin/env bash
set -euo pipefail

# Paralome annotation shards are one-CPU tasks. Some tools and
# libraries may otherwise discover all node CPUs and create helper
# threads, which can violate SLURM cpus-per-task limits during high
# concurrency arrays.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required executable not found in PATH: $1" >&2
    exit 1
  fi
}
''' + recorded_tools_bash(*_PINNED_TOOLS) + '''
require_tool bgzip
require_tool tabix
'''


def build_snpeff_script(
    *,
    input_vcf: Path,
    output_vcf: Path,
    workdir: Path,
    config: SnpEffConfig,
) -> str:
    raw = workdir / "temp" / (output_vcf.name.removesuffix(".gz") if output_vcf.name.endswith(".gz") else output_vcf.name + ".raw.vcf")
    return "\n".join([
        _common_script_header(),
        "require_tool java",
        f"mkdir -p {shlex.quote(str(raw.parent))} {shlex.quote(str(output_vcf.parent))}",
        f"rm -f {shlex.quote(str(raw))} {shlex.quote(str(output_vcf))} {shlex.quote(str(output_vcf))}.tbi",
        # Quote each message as one shell word (paths may hold '(' or '&').
        f"echo {shlex.quote('Running SnpEff: ' + str(input_vcf))}",
        f"{shlex.quote(config.java)} -XX:ActiveProcessorCount=1 -Djava.util.concurrent.ForkJoinPool.common.parallelism=1 -Xmx{shlex.quote(config.memory)} \\",
        f"  -jar {shlex.quote(str(config.jar))} \\",
        f"  -c {shlex.quote(str(config.config))} \\",
        f"  {shlex.quote(config.genome)} \\",
        f"  {shlex.quote(str(input_vcf))} > {shlex.quote(str(raw))}",
        f"if [[ ! -s {shlex.quote(str(raw))} ]]; then echo 'ERROR: SnpEff produced empty output' >&2; exit 1; fi",
        f"bgzip -f {shlex.quote(str(raw))}",
        f"mv -f {shlex.quote(str(raw) + '.gz')} {shlex.quote(str(output_vcf))}",
        f"tabix -f -p vcf {shlex.quote(str(output_vcf))}",
        f"echo {shlex.quote('Wrote ' + str(output_vcf))}",
        _checkpoint_python(workdir, "snpeff", output_vcf, input_vcf),
        "",
    ])


from paralome.pipeline.slurm import tmp_directive  # noqa: E402
from paralome.scratch import job_scratch_bash, uses_scratch  # noqa: E402

#: The scratch need of a whole-VCF ANNOVAR job, as a multiple of its (compressed) input.
ANNOVAR_SPLIT_FACTOR = 30
#: The scratch need of one shard's split: the split, its sorted copy and the sort's spill.
SHARD_SPLIT_FACTOR = 3


def _size(path: Optional[Path]) -> int:
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def build_annovar_script(
    *,
    input_vcf: Path,
    output_vcf: Path,
    workdir: Path,
    config: AnnovarConfig,
) -> str:
    temp_input = workdir / "temp" / "annovar_input.vcf"
    prefix = workdir / "temp" / "annovar" / output_vcf.name.replace(".vcf.gz", "").replace(".vcf", "")
    annovar_vcf = Path(str(prefix) + f".{config.buildver}_multianno.vcf")
    id_map = _info_id_map_path(workdir, "annovar")
    input_prep = [
        "require_tool perl",
        "require_tool bcftools",
        f"mkdir -p {shlex.quote(str(temp_input.parent))} {shlex.quote(str(prefix.parent))} {shlex.quote(str(output_vcf.parent))}",
        # The split input and its sort spill on this node's scratch when the site has one.
        # A compressed VCF is about a tenth of its text; the split, sorted copy and spill are three of those.
        job_scratch_bash(shlex.quote(str(temp_input.parent)), "annotation-split", var="SPLIT_DIR",
                         need_bytes=f"$(stat -c %s {shlex.quote(str(input_vcf))} 2>/dev/null || echo 0) * {ANNOVAR_SPLIT_FACTOR}"),
        normalize_for_annotation_bash(
            shlex.quote(str(input_vcf)), '"$SPLIT_DIR/annovar_input.vcf"', config.reference_fasta,
            '"$SPLIT_DIR/annovar_sort"',
        ),
    ]
    return "\n".join([
        _common_script_header(),
        *input_prep,
        f"rm -f {shlex.quote(str(annovar_vcf))} {shlex.quote(str(output_vcf))} {shlex.quote(str(output_vcf))}.tbi",
        f"echo {shlex.quote('Running ANNOVAR: ' + str(input_vcf))}",
        f"perl {shlex.quote(str(config.table_annovar))} \\",
        "  \"$SPLIT_DIR/annovar_input.vcf\" \\",
        f"  {shlex.quote(str(config.database_dir))} \\",
        f"  -buildver {shlex.quote(config.buildver)} \\",
        f"  -out {shlex.quote(str(prefix))} \\",
        "  -remove \\",
        f"  -protocol {shlex.quote(config.protocol)} \\",
        f"  -operation {shlex.quote(config.operation)} \\",
        f"  -nastring {shlex.quote(config.nastring)} \\",
        "  -vcfinput",
        f"if [[ ! -s {shlex.quote(str(annovar_vcf))} ]]; then echo {shlex.quote('ERROR: ANNOVAR did not create ' + str(annovar_vcf))} >&2; exit 1; fi",
        info_id_rename_bash(shlex.quote(str(annovar_vcf)), shlex.quote(str(id_map))),
        f"bgzip -f {shlex.quote(str(annovar_vcf))}",
        f"mv -f {shlex.quote(str(annovar_vcf) + '.gz')} {shlex.quote(str(output_vcf))}",
        f"tabix -f -p vcf {shlex.quote(str(output_vcf))}",
        f"echo {shlex.quote('Wrote ' + str(output_vcf))}",
        _checkpoint_python(workdir, "annovar", output_vcf, input_vcf, id_map),
        "",
    ])


def _resolve_shard_count(threads: int = 1, jobs: int = 0) -> tuple[int, int]:
    # Return (shards, concurrency) for annotation sharding.
    # ANNOVAR and SnpEff are treated as one-CPU workers.
    total_threads = max(1, int(threads or 1))
    shards = max(1, int(jobs or total_threads))
    plan = plan_sample_parallel(
        total_threads=total_threads,
        sample_count=shards,
        max_threads_per_sample=1,
        jobs=jobs,
    )
    return shards, plan.concurrent_jobs


def _shard_prep_bash(input_vcf: Path, shard_dir: Path, shards: int) -> str:
    return "\n".join([
        f"INPUT_VCF={shlex.quote(str(input_vcf))}",
        f"SHARD_DIR={shlex.quote(str(shard_dir))}",
        f"SHARDS={int(shards)}",
        "SHARD_ID=${SLURM_ARRAY_TASK_ID:-${PARALOME_SHARD_ID:-1}}",
        "mkdir -p \"$SHARD_DIR/input\" \"$SHARD_DIR/output\"",
        "SHARD_INPUT=$(printf '%s/input/shard_%04d.vcf' \"$SHARD_DIR\" \"$SHARD_ID\")",
        "TOTAL_RECORDS=$(bcftools view -H \"$INPUT_VCF\" | wc -l | tr -d ' ')",
        "if [[ \"$TOTAL_RECORDS\" -lt 1 ]]; then echo 'ERROR: input VCF has no records' >&2; exit 1; fi",
        "PER_SHARD=$(( (TOTAL_RECORDS + SHARDS - 1) / SHARDS ))",
        "START=$(( (SHARD_ID - 1) * PER_SHARD + 1 ))",
        "END=$(( SHARD_ID * PER_SHARD ))",
        "echo \"Preparing annotation shard ${SHARD_ID}/${SHARDS}: records ${START}-${END} of ${TOTAL_RECORDS}\"",
        "bcftools view -h \"$INPUT_VCF\" > \"$SHARD_INPUT\"",
        "bcftools view -H \"$INPUT_VCF\" | awk -v s=\"$START\" -v e=\"$END\" 'NR >= s && NR <= e {print}' >> \"$SHARD_INPUT\"",
        # A shard with no records is written header-only below
        # (see _empty_shard_bash). This exited 0 with no output, and the merge
        # then stopped on the missing shard.
        "SHARD_RECORDS=$(grep -vc '^#' \"$SHARD_INPUT\" || true)",
    ])


def _existing_shard_prep_bash(input_shard_dir: Path, output_shard_dir: Path) -> str:
    return "\n".join([
        f"SHARD_INPUT_DIR={shlex.quote(str(input_shard_dir))}",
        f"SHARD_DIR={shlex.quote(str(output_shard_dir))}",
        "SHARD_ID=${SLURM_ARRAY_TASK_ID:-${PARALOME_SHARD_ID:-1}}",
        "mkdir -p \"$SHARD_DIR/output\"",
        "SHARD_INPUT=\"$SHARD_INPUT_DIR/shard_${SHARD_ID}.vcf\"",
        "if [[ ! -s \"$SHARD_INPUT\" ]]; then",
        "  SHARD_INPUT=$(printf '%s/shard_%04d.vcf' \"$SHARD_INPUT_DIR\" \"$SHARD_ID\")",
        "fi",
        "if [[ ! -s \"$SHARD_INPUT\" ]]; then",
        "  SHARD_INPUT=\"$SHARD_INPUT_DIR/shard_${SHARD_ID}.vcf.gz\"",
        "fi",
        "if [[ ! -s \"$SHARD_INPUT\" ]]; then",
        "  SHARD_INPUT=$(printf '%s/shard_%04d.vcf.gz' \"$SHARD_INPUT_DIR\" \"$SHARD_ID\")",
        "fi",
        "if [[ ! -s \"$SHARD_INPUT\" ]]; then echo \"ERROR: source shard VCF not found for shard $SHARD_ID in $SHARD_INPUT_DIR\" >&2; exit 1; fi",
        # A SHARD WITH NO RECORDS IS NOT AN ERROR. A FreeBayes chunk
        # whose regions hold no variant writes a header-only shard, and capping
        # region length makes that more likely. It is passed through
        # header-only (see _empty_shard_bash) and the merge takes its header
        # from a shard that has records.
        # Counted to the end rather than cut off with `head -1`: in an
        # assignment under pipefail, the SIGPIPE head gives bcftools would
        # end the job, and a failure of bcftools itself must still end it.
        "SHARD_RECORDS=$(bcftools view -H \"$SHARD_INPUT\" | awk 'END { print NR }')",
        "echo \"Annotating existing VCF shard $SHARD_ID: $SHARD_INPUT\"",
    ])


def _normalized_shard_bash(reference: Optional[Path]) -> str:
    """Bash that replaces SHARD_INPUT by its split form under $SHARD_DIR/normalized.

    Written under the annotation's own shard folder, never beside the source
    shard (which may be the variant stage's). SHARD_RECORDS is counted again:
    a shard can gain records (split calls) or, rarely, lose all of them (only
    '*' alleles), and the header-only path below must see that.
    """
    from paralome.scratch import job_scratch_bash

    # The split shard and its sort spill on this node's scratch when the site has one.
    return "\n".join([
        # About 3x the shard: the split, its sorted copy and the sort's spill.
        job_scratch_bash('"$SHARD_DIR/normalized"', "annotation-split", var="SPLIT_DIR",
                         need_bytes=f"$(stat -c %s \"$SHARD_INPUT\" 2>/dev/null || echo 0) * {SHARD_SPLIT_FACTOR}"),
        "SHARD_NORMALIZED=$(printf '%s/shard_%04d.vcf' \"$SPLIT_DIR\" \"$SHARD_ID\")",
        normalize_for_annotation_bash('"$SHARD_INPUT"', '"$SHARD_NORMALIZED"', reference,
                                      '"$SPLIT_DIR/sort_$SHARD_ID"'),
        "SHARD_INPUT=\"$SHARD_NORMALIZED\"",
        "SHARD_RECORDS=$(bcftools view -H \"$SHARD_INPUT\" | awk 'END { print NR }')",
    ])


def _empty_shard_bash(id_map: Optional[str] = None) -> str:
    """Bash that writes a record-less shard straight through, header-only.

    Expects SHARD_RECORDS, SHARD_INPUT, SHARD_ID and SHARD_GZ to be set. No
    engine is run on it: there is nothing to annotate, and ANNOVAR's behaviour
    on an empty input is not something to depend on.
    """
    lines = [
        "if [[ \"$SHARD_RECORDS\" -lt 1 ]]; then",
        "  echo \"Shard $SHARD_ID has no variant records; writing it header-only: $SHARD_GZ\"",
        "  rm -f \"$SHARD_GZ\" \"${SHARD_GZ}.tbi\"",
        "  bcftools view -h \"$SHARD_INPUT\" | bgzip -c > \"$SHARD_GZ\"",
        "  tabix -f -p vcf \"$SHARD_GZ\"",
    ]
    if id_map:
        lines.append(f"  : > {id_map}")
    lines += ["  exit 0", "fi"]
    return "\n".join(lines)


def _merge_header_bash(first_shard_expr: str) -> str:
    """Bash that sets HEADER_FROM to the first listed shard holding a record.

    An empty shard is passed through without annotation, so its
    header lacks the annotation INFO lines; the merged header is taken from a
    shard that was annotated. Expects SHARD_PATHS (an array) to be set.
    """
    # Counted to the end in an assignment, as _existing_shard_prep_bash
    # does, not `| head -1` inside a condition: there a bcftools failure on a
    # corrupt shard read as "no records" and the shard was passed over without
    # a word. In an assignment under pipefail it stops the merge.
    return "\n".join([
        "HEADER_FROM=\"\"",
        "for SHARD in \"${SHARD_PATHS[@]}\"; do",
        "  SHARD_RECORD_COUNT=$(bcftools view -H \"$SHARD\" | awk 'END { print NR }')",
        "  if [[ \"$SHARD_RECORD_COUNT\" -gt 0 ]]; then HEADER_FROM=\"$SHARD\"; break; fi",
        "done",
        f"if [[ -z \"$HEADER_FROM\" ]]; then HEADER_FROM={first_shard_expr}; fi",
    ])


def _duplicates_path(workdir: Path, label: str) -> Path:
    return workdir / "temp" / f"{label}_merge_duplicates.txt"


def _joined_path(workdir: Path, label: str) -> Path:
    """Where the merge counts the same-allele records it joined and left apart."""
    return workdir / "temp" / f"{label}_merge_joined.txt"


#: CPUs for the merge job, and so for bgzip. The merge of a
#: 40-proband cohort wrote its 21 GB output at about 100 MB a minute on one
#: thread, which was most of a six-hour annotation stage. The sort ahead of it
#: is still serial; what this buys is the write.
MERGE_THREADS = 4


def _stream_merge_bash(duplicates: Path, joined: Path) -> list[str]:
    """Bash that merges SHARD_PATHS into OUTPUT_VCF in one stream.

    Header of HEADER_FROM, every shard's records | bcftools sort, spilling to
    SORT_TMP in the run's own folder (never /tmp) | the exact-duplicate filter
    | the same-allele join | bgzip. No uncompressed copy is written.

    THE DUPLICATE FILTER. With --variant-workdir the shards annotated are
    FreeBayes's own, which the FreeBayes merge's duplicate removal never
    reached, so the same rule is applied here: a record with the CHROM, POS,
    REF, ALT and sample columns of one already kept is dropped, and the number
    dropped goes into the annotation checkpoint. The count of records kept is
    checked too: a merge with none is an error.

    THE SAMPLE COLUMNS ARE PART OF THE RULE. Since the split for annotation,
    two FreeBayes records (different haplotypes, different carriers) can give
    the same single change, and when the two land in different shards, or
    their FORMAT differs, the join inside the shard does not see them (e.g.
    chr1:11 ACG>TCG in S1 and AC>TA in S2 give two A>T records). A record of an
    allele already kept is dropped only when its sample columns are the same;
    otherwise it is joined with the kept one, genotype by haplotype, by the
    join the split uses (annovar.SAME_ALLELE_JOIN_AWK), and records whose
    FORMAT differs stay as two lines. The counts of records joined and left
    apart go into the checkpoint beside the duplicates dropped.
    """
    return [
        f"DUPLICATES={shlex.quote(str(duplicates))}",
        f"JOINED={shlex.quote(str(joined))}",
        "KEPT_COUNT=\"$DUPLICATES.kept\"",
        # The sort's spill files go on this node's scratch when the site has one, rather than
        # shared storage (a 40-proband merge spills about 18 GB). The guard is given the shards'
        # own total, which is what the sort has to hold.
        job_scratch_bash('"$(dirname "$DUPLICATES")/sort_tmp"', "annotation-merge",
                         var="SORT_TMP",
                         need_bytes='$(du -scb "${SHARD_PATHS[@]}" 2>/dev/null | tail -1 | cut -f1)'),
        "mkdir -p \"$SORT_TMP\"",
        "rm -f \"$OUTPUT_VCF.partial\" \"$DUPLICATES\" \"$KEPT_COUNT\" \"$JOINED\"",
        "{",
        "  bcftools view -h \"$HEADER_FROM\"",
        "  for SHARD in \"${SHARD_PATHS[@]}\"; do",
        "    bcftools view -H \"$SHARD\"",
        "  done",
        "} | bcftools sort -T \"$SORT_TMP/bcftools.\" -O v - \\",
        "  | awk -F '\\t' -v DUPS=\"$DUPLICATES\" -v KEPT=\"$KEPT_COUNT\" '",
        "      /^#/ { print; next }",
        "      {",
        "        if ($1 != chrom || $2 != pos) { split(\"\", seen); chrom = $1; pos = $2 }",
        "        samples = \"\"",
        "        for (i = 9; i <= NF; i++) samples = samples \"\\t\" $i",
        "        key = $4 \"\\t\" $5 samples",
        "        if (key in seen) { dropped++; next }",
        "        seen[key] = 1; kept++",
        "        print",
        "      }",
        "      END { print dropped + 0 > DUPS; print kept + 0 > KEPT }",
        "    ' \\",
        f"  | awk -v JOINED=\"$JOINED\" {shlex.quote(SAME_ALLELE_JOIN_AWK)} \\",
        f"  | bgzip -c -@ {MERGE_THREADS} > \"$OUTPUT_VCF.partial\"",
        "if [[ \"$(cat \"$KEPT_COUNT\")\" -lt 1 ]]; then rm -f \"$OUTPUT_VCF.partial\"; echo \"ERROR: merged annotation VCF has no records\" >&2; exit 1; fi",
        "echo \"Duplicate records removed in the merge: $(cat \"$DUPLICATES\")\"",
        "read -r JOINED_COUNT APART_COUNT < \"$JOINED\"",
        "echo \"Records of one allele with different carriers joined in the merge: $JOINED_COUNT (left as two records, FORMAT differs: $APART_COUNT)\"",
        "if [[ \"$APART_COUNT\" -gt 0 ]]; then echo \"WARNING: $APART_COUNT record(s) share CHROM, POS, REF and ALT with another but a different FORMAT; both were kept, and gene burden reads the first only\" >&2; fi",
        "mv -f \"$OUTPUT_VCF.partial\" \"$OUTPUT_VCF\"",
        "rm -f \"$KEPT_COUNT\"",
        # Under scratch the directory goes with the job; this is for the
        # fallback, where it sits beside the output.
        "rmdir \"$SORT_TMP\" 2>/dev/null || true",
    ]


def _gather_id_maps_bash(id_map: Path) -> list[str]:
    """Bash that gathers every shard's INFO ID renames into one file."""
    return [
        f"ID_MAP={shlex.quote(str(id_map))}",
        ": > \"$ID_MAP\"",
        "for MAP in \"$SHARD_DIR\"/output/shard_*.info_id_renames.tsv; do",
        "  if [[ -s \"$MAP\" ]]; then cat \"$MAP\" >> \"$ID_MAP.all\"; fi",
        "done",
        "if [[ -s \"$ID_MAP.all\" ]]; then sort -u \"$ID_MAP.all\" > \"$ID_MAP\"; fi",
        "rm -f \"$ID_MAP.all\"",
    ]


def _shard_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.endswith(".vcf.gz"):
        name = name[:-7]
    elif name.endswith(".vcf"):
        name = name[:-4]
    try:
        return (int(name.rsplit("_", 1)[1]), path.name)
    except Exception:
        return (10**9, path.name)


def _discover_shard_vcfs(shard_dir: Path) -> list[Path]:
    shard_dir = Path(shard_dir)
    if not shard_dir.exists() or not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard VCF directory not found: {shard_dir}")
    shards = sorted(shard_dir.glob("shard_*.vcf"), key=_shard_sort_key)
    if not shards:
        shards = sorted(shard_dir.glob("shard_*.vcf.gz"), key=_shard_sort_key)
    if not shards:
        raise FileNotFoundError(f"No shard_*.vcf or shard_*.vcf.gz files found in {shard_dir}")
    return shards


def _shard_number(path: Path) -> Optional[int]:
    number = _shard_sort_key(path)[0]
    return None if number == 10**9 else number


def _planned_freebayes_shards(variant_workdir: Path) -> tuple[Optional[int], Optional[str]]:
    """(the number of shards FreeBayes planned for this workdir, where it said so).

    The chunk plan (inputs/freebayes_chunk_plan.json)
    first; the summary the merge writes (results/freebayes_summary.json)
    when there is no plan. (None, None) when neither names a count.
    """
    plan_path = Path(variant_workdir) / "inputs" / "freebayes_chunk_plan.json"
    summary_path = Path(variant_workdir) / "results" / "freebayes_summary.json"
    for path, key in ((plan_path, "chunks"), (summary_path, "shards")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, int) and value > 0:
            return value, str(path)
    return None, None


def _check_freebayes_shards(variant_workdir: Path, shard_dir: Path, shards: list[Path]) -> None:
    """Refuse a FreeBayes workdir whose calling is not finished.

    FreeBayes writes each shard through a .tmp and moves it into place when
    its task ends, so with chunks still running or failed the high-numbered
    shards are simply absent, and annotating what is there would give a merged
    VCF missing part of the genome. Every shard FreeBayes planned must be
    there, numbered 1..N without a gap or an extra, and each must be one a
    task finished (variants.freebayes.shard_is_finished: it has its #CHROM
    header line). Without a chunk plan or summary the count cannot be checked
    and the run says so; the shards found are still each checked for a header.
    """
    from paralome.variants.freebayes import shard_is_finished

    planned, source = _planned_freebayes_shards(variant_workdir)
    numbers = [_shard_number(path) for path in shards]
    if planned is None:
        print(f"NOTE: {variant_workdir} has no inputs/freebayes_chunk_plan.json or "
              f"results/freebayes_summary.json; cannot check that its {len(shards)} shard(s) are every chunk "
              f"FreeBayes planned. Annotating the shards found in {shard_dir}.")
    else:
        expected = list(range(1, planned + 1))
        if numbers != expected:
            missing = sorted(set(expected) - set(n for n in numbers if n is not None))
            extra = sorted(set(n for n in numbers if n is not None) - set(expected))
            unnumbered = [path.name for path, n in zip(shards, numbers) if n is None]
            details = []
            if missing:
                details.append(f"missing shard(s): {_ranges(missing)}")
            if extra:
                details.append(f"shard(s) beyond the plan: {_ranges(extra)}")
            if unnumbered:
                details.append(f"not numbered shard_N: {', '.join(unnumbered)}")
            raise RuntimeError(
                f"FreeBayes has not finished calling in {variant_workdir}: {source} plans {planned} chunk(s), "
                f"{shard_dir} holds {len(shards)} shard VCF(s) ({'; '.join(details)}). FreeBayes writes a "
                f"shard only when its task finishes, so a missing shard is a chunk still running or failed. "
                f"Let the variants stage finish (paralome status, or `paralome variants` on that workdir), "
                f"then annotate; or pass --shard-vcf-dir to annotate a folder of shards as it is."
            )
    unfinished = [path.name for path in shards if path.suffix == ".vcf" and not shard_is_finished(path)]
    if unfinished:
        shown = ", ".join(unfinished[:10]) + (f" and {len(unfinished) - 10} more" if len(unfinished) > 10 else "")
        raise RuntimeError(
            f"{len(unfinished)} shard VCF(s) in {shard_dir} have no #CHROM header line, so no FreeBayes task "
            f"finished them: {shown}. Let the variants stage finish before annotating."
        )


def _ranges(numbers: list[int]) -> str:
    """1-3, 7, 9-12 for [1,2,3,7,9,10,11,12]."""
    out: list[str] = []
    start = prev = None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


def _resolve_source_shards(variant_workdir: Optional[Path], shard_vcf_dir: Optional[Path]) -> tuple[Optional[Path], list[Path]]:
    from_variant_workdir = shard_vcf_dir is None and variant_workdir is not None
    if from_variant_workdir:
        shard_vcf_dir = Path(variant_workdir) / "shards"
    if shard_vcf_dir is None:
        return None, []
    shard_vcf_dir = _absolute(Path(shard_vcf_dir))
    shards = _discover_shard_vcfs(shard_vcf_dir)
    if from_variant_workdir:
        # FreeBayes's own shards are taken only when FreeBayes is done with them.
        _check_freebayes_shards(_absolute(Path(variant_workdir)), shard_vcf_dir, shards)
    return shard_vcf_dir, shards


def build_snpeff_shard_script(
    *,
    input_vcf: Path,
    workdir: Path,
    config: SnpEffConfig,
    shards: int,
    source_shard_dir: Optional[Path] = None,
) -> str:
    shard_dir = workdir / "temp" / "snpeff_shards"
    prep = _existing_shard_prep_bash(source_shard_dir, shard_dir) if source_shard_dir else _shard_prep_bash(input_vcf, shard_dir, shards)
    return "\n".join([
        _common_script_header(),
        "require_tool java",
        "require_tool bcftools",
        prep,
        "SHARD_OUT=$(printf '%s/output/shard_%04d.snpeff.vcf' \"$SHARD_DIR\" \"$SHARD_ID\")",
        "SHARD_GZ=\"${SHARD_OUT}.gz\"",
        _empty_shard_bash(),
        "rm -f \"$SHARD_OUT\" \"$SHARD_GZ\" \"${SHARD_GZ}.tbi\"",
        f"{shlex.quote(config.java)} -XX:ActiveProcessorCount=1 -Djava.util.concurrent.ForkJoinPool.common.parallelism=1 -Xmx{shlex.quote(config.memory)} \\",
        f"  -jar {shlex.quote(str(config.jar))} \\",
        f"  -c {shlex.quote(str(config.config))} \\",
        f"  {shlex.quote(config.genome)} \\",
        "  \"$SHARD_INPUT\" > \"$SHARD_OUT\"",
        "if [[ ! -s \"$SHARD_OUT\" ]]; then echo \"ERROR: SnpEff produced empty output for shard $SHARD_ID\" >&2; exit 1; fi",
        "bgzip -f \"$SHARD_OUT\"",
        "tabix -f -p vcf \"$SHARD_GZ\"",
        "echo \"Wrote $SHARD_GZ\"",
        "",
    ])


def build_annovar_shard_script(
    *,
    input_vcf: Path,
    workdir: Path,
    config: AnnovarConfig,
    shards: int,
    source_shard_dir: Optional[Path] = None,
) -> str:
    shard_dir = workdir / "temp" / "annovar_shards"
    prep = _existing_shard_prep_bash(source_shard_dir, shard_dir) if source_shard_dir else _shard_prep_bash(input_vcf, shard_dir, shards)
    return "\n".join([
        _common_script_header(),
        "require_tool perl",
        "require_tool bcftools",
        prep,
        _normalized_shard_bash(config.reference_fasta),
        "SHARD_PREFIX=$(printf '%s/output/shard_%04d/annovar' \"$SHARD_DIR\" \"$SHARD_ID\")",
        "mkdir -p \"$(dirname \"$SHARD_PREFIX\")\"",
        f"SHARD_ANNOVAR_VCF=\"${{SHARD_PREFIX}}.{config.buildver}_multianno.vcf\"",
        "SHARD_GZ=$(printf '%s/output/shard_%04d.annovar.vcf.gz' \"$SHARD_DIR\" \"$SHARD_ID\")",
        "SHARD_ID_MAP=$(printf '%s/output/shard_%04d.info_id_renames.tsv' \"$SHARD_DIR\" \"$SHARD_ID\")",
        _empty_shard_bash('"$SHARD_ID_MAP"'),
        "rm -f \"$SHARD_ANNOVAR_VCF\" \"$SHARD_GZ\" \"${SHARD_GZ}.tbi\"",
        f"perl {shlex.quote(str(config.table_annovar))} \\",
        "  \"$SHARD_INPUT\" \\",
        f"  {shlex.quote(str(config.database_dir))} \\",
        f"  -buildver {shlex.quote(config.buildver)} \\",
        "  -out \"$SHARD_PREFIX\" \\",
        "  -remove \\",
        f"  -protocol {shlex.quote(config.protocol)} \\",
        f"  -operation {shlex.quote(config.operation)} \\",
        f"  -nastring {shlex.quote(config.nastring)} \\",
        "  -vcfinput",
        "if [[ ! -s \"$SHARD_ANNOVAR_VCF\" ]]; then echo \"ERROR: ANNOVAR produced no VCF for shard $SHARD_ID\" >&2; exit 1; fi",
        info_id_rename_bash('"$SHARD_ANNOVAR_VCF"', '"$SHARD_ID_MAP"'),
        "bgzip -f \"$SHARD_ANNOVAR_VCF\"",
        "mv -f \"${SHARD_ANNOVAR_VCF}.gz\" \"$SHARD_GZ\"",
        "tabix -f -p vcf \"$SHARD_GZ\"",
        "echo \"Wrote $SHARD_GZ\"",
        "",
    ])


def build_shard_merge_script(
    *,
    engine: str,
    input_vcf: Path,
    output_vcf: Path,
    workdir: Path,
    shards: int,
) -> str:
    shard_dir = workdir / "temp" / f"{engine}_shards"
    suffix = "snpeff" if engine == "snpeff" else "annovar"
    duplicates = _duplicates_path(workdir, engine)
    joined = _joined_path(workdir, engine)
    id_map = _info_id_map_path(workdir, engine) if engine == "annovar" else None
    return "\n".join([
        _common_script_header(),
        "require_tool bcftools",
        f"SHARD_DIR={shlex.quote(str(shard_dir))}",
        f"OUTPUT_VCF={shlex.quote(str(output_vcf))}",
        f"SHARDS={int(shards)}",
        f"SUFFIX={suffix}",
        f"mkdir -p {shlex.quote(str(workdir / 'temp'))} \"$(dirname \"$OUTPUT_VCF\")\"",
        "FIRST=$(printf '%s/output/shard_%04d.%s.vcf.gz' \"$SHARD_DIR\" 1 \"$SUFFIX\")",
        "if [[ ! -s \"$FIRST\" ]]; then echo \"ERROR: first shard output missing: $FIRST\" >&2; exit 1; fi",
        "rm -f \"$OUTPUT_VCF\" \"${OUTPUT_VCF}.tbi\"",
        "SHARD_PATHS=()",
        "for i in $(seq 1 \"$SHARDS\"); do",
        "  SHARD=$(printf '%s/output/shard_%04d.%s.vcf.gz' \"$SHARD_DIR\" \"$i\" \"$SUFFIX\")",
        "  if [[ ! -s \"$SHARD\" ]]; then echo \"ERROR: shard output missing: $SHARD\" >&2; exit 1; fi",
        "  SHARD_PATHS+=(\"$SHARD\")",
        "done",
        _merge_header_bash('"$FIRST"'),
        *(_gather_id_maps_bash(id_map) if id_map else []),
        *_stream_merge_bash(duplicates, joined),
        "tabix -f -p vcf \"$OUTPUT_VCF\"",
        "if [[ ! -s \"$OUTPUT_VCF\" ]]; then echo \"ERROR: sorted annotation VCF missing: $OUTPUT_VCF\" >&2; exit 1; fi",
        "if [[ ! -s \"${OUTPUT_VCF}.tbi\" ]]; then echo \"ERROR: annotation VCF index missing: ${OUTPUT_VCF}.tbi\" >&2; exit 1; fi",
        "echo \"Wrote $OUTPUT_VCF and ${OUTPUT_VCF}.tbi\"",
        _checkpoint_python(workdir, engine, output_vcf, input_vcf, id_map, duplicates, joined),
        "",
    ])


def _write_sharded_slurm_wrappers(
    *,
    workdir: Path,
    engine: str,
    shard_script: Path,
    merge_script: Path,
    mem: str,
    partition: Optional[str],
    shards: int,
    concurrency: int,
    shard_tmp_bytes: int = 0,
    merge_tmp_bytes: int = 0,
) -> tuple[Path, Path]:
    """The array and merge sbatch wrappers.

    ``shard_tmp_bytes`` / ``merge_tmp_bytes`` are what one shard
    task / the merge writes to node scratch (0 when its script uses none),
    asked of SLURM as --tmp (see paralome.pipeline.slurm.tmp_request).
    """
    array_sbatch = workdir / "scripts" / f"annotate_{engine}_array.sbatch"
    merge_sbatch = workdir / "scripts" / f"annotate_{engine}_merge.sbatch"
    partition_line = f"#SBATCH --partition={shlex.quote(str(partition))}\n" if partition else ""
    workdir_q = shlex.quote(str(workdir))
    shard_tmp_line = tmp_directive(shard_tmp_bytes, partition)
    merge_tmp_line = tmp_directive(merge_tmp_bytes, partition)
    array_sbatch.write_text(f'''#!/usr/bin/env bash
{partition_line}#SBATCH --job-name={slurm_job_name(workdir, f"annotation_{engine}")}
#SBATCH --array=1-{shards}%{concurrency}
#SBATCH --cpus-per-task=1
#SBATCH --mem={shlex.quote(str(mem))}
{shard_tmp_line}#SBATCH --output={workdir_q}/logs/annotate_{engine}_%A_%a.out
#SBATCH --error={workdir_q}/logs/annotate_{engine}_%A_%a.err

set -euo pipefail
bash {shlex.quote(str(shard_script))}
''')
    merge_sbatch.write_text(f'''#!/usr/bin/env bash
{partition_line}#SBATCH --job-name={slurm_job_name(workdir, f"annotation_{engine}_merge")}
#SBATCH --cpus-per-task=1
#SBATCH --mem={shlex.quote(str(mem))}
{merge_tmp_line}#SBATCH --output={workdir_q}/logs/annotate_{engine}_merge_%j.out
#SBATCH --error={workdir_q}/logs/annotate_{engine}_merge_%j.err

set -euo pipefail
bash {shlex.quote(str(merge_script))}
''')
    array_sbatch.chmod(0o755)
    merge_sbatch.chmod(0o755)
    return array_sbatch, merge_sbatch


def run_sharded_annotation_script(
    *,
    engine: str,
    shard_script_body: str,
    merge_script_body: str,
    input_vcf: Path,
    output_vcf: Path,
    workdir: Path,
    backend: str = "local",
    submit: bool = False,
    dry_run: bool = False,
    partition: Optional[str] = None,
    mem: str = "16G",
    threads: int = 1,
    jobs: int = 0,
    source_shard_dir: Optional[Path] = None,
    source_shards: Optional[list[Path]] = None,
) -> AnnotationRunResult:
    backend = backend.lower().strip()
    workdir = _absolute(Path(workdir))
    if source_shards:
        shards = len(source_shards)
        concurrency = plan_sample_parallel(
            total_threads=threads,
            sample_count=shards,
            max_threads_per_sample=1,
            jobs=jobs,
        ).concurrent_jobs
    else:
        shards, concurrency = _resolve_shard_count(threads=threads, jobs=jobs)
    # What the shard tasks and the merge write to node scratch, for
    # --tmp. A shard task splits its shard (the largest source shard, else an
    # even share of the input); the merge's sort holds the annotated shards,
    # which are not written yet: the input they annotate stands in for them
    # (annotation only adds to a record), and the job's own space guard and
    # the scratch ledger see the real figure when it runs.
    if source_shards:
        shard_estimate = max((_size(p) for p in source_shards), default=0)
        merge_estimate = sum(_size(p) for p in source_shards)
    else:
        merge_estimate = _size(input_vcf)
        shard_estimate = -(-merge_estimate // max(1, int(shards)))
    shard_tmp_bytes = shard_estimate * SHARD_SPLIT_FACTOR if uses_scratch(shard_script_body) else 0
    merge_tmp_bytes = merge_estimate if uses_scratch(merge_script_body) else 0
    workdir = Path(workdir)
    _ensure_dirs(workdir)
    shard_script = workdir / "scripts" / f"annotate_{engine}_shard.sh"
    merge_script = workdir / "scripts" / f"merge_{engine}_shards.sh"
    shard_script.write_text(shard_script_body)
    merge_script.write_text(merge_script_body)
    shard_script.chmod(0o755)
    merge_script.chmod(0o755)
    plan = write_annotation_plan(
        workdir=workdir,
        engine=engine,
        input_vcf=input_vcf,
        output_vcf=output_vcf,
        backend=backend,
        script=shard_script,
        extra={
            "submit": submit,
            "dry_run": dry_run,
            "mem": mem,
            "partition": partition,
            "sharded": True,
            "shards": shards,
            "concurrency": concurrency,
            "threads": threads,
            "jobs": jobs,
            "source_shard_dir": str(source_shard_dir) if source_shard_dir else None,
            "source_shards": [str(p) for p in source_shards] if source_shards else [],
            "merge_script": str(merge_script),
        },
    )
    checkpoint = stage_checkpoint_path(workdir, f"annotation_{engine}")
    if not dry_run:
        begin_annotation_checkpoint(workdir, engine, input_vcf, output_vcf, plan)

    if dry_run:
        if backend == "slurm":
            array_sbatch, _ = _write_sharded_slurm_wrappers(
                workdir=workdir,
                engine=engine,
                shard_script=shard_script,
                merge_script=merge_script,
                mem=mem,
                partition=partition,
                shards=shards,
                concurrency=concurrency,
                shard_tmp_bytes=shard_tmp_bytes,
                merge_tmp_bytes=merge_tmp_bytes,
            )
            return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, array_sbatch, submitted=False, complete=False, plan=plan, checkpoint=checkpoint)
        return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, shard_script, submitted=False, complete=False, plan=plan, checkpoint=checkpoint)

    if backend == "local":
        log_out = workdir / "logs" / f"annotate_{engine}.out"
        log_err = workdir / "logs" / f"annotate_{engine}.err"
        with log_out.open("w") as out, log_err.open("w") as err:
            for shard_id in range(1, shards + 1):
                # In this environment, as an sbatch job gets it. This
                # ran each shard through `bash -lc`, a login shell, whose
                # profile can put a different PATH -- and different tools --
                # in front of the shard than the run itself has.
                # SLURM_ARRAY_TASK_ID is dropped: the script prefers it to
                # PARALOME_SHARD_ID, and a local run started inside a SLURM job
                # would otherwise annotate that job's task number every time.
                env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
                env["PARALOME_SHARD_ID"] = str(shard_id)
                subprocess.run(["bash", str(shard_script)], env=env, check=True, stdout=out, stderr=err)
            subprocess.run(["bash", str(merge_script)], check=True, stdout=out, stderr=err)
        return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, merge_script, submitted=False, complete=output_vcf.exists(), plan=plan, checkpoint=checkpoint)

    if backend == "slurm":
        array_sbatch, merge_sbatch = _write_sharded_slurm_wrappers(
            workdir=workdir,
            engine=engine,
            shard_script=shard_script,
            merge_script=merge_script,
            mem=mem,
            partition=partition,
            shards=shards,
            concurrency=concurrency,
            shard_tmp_bytes=shard_tmp_bytes,
            merge_tmp_bytes=merge_tmp_bytes,
        )
        job_id = None
        submitted = False
        if submit:
            _require_tool("sbatch")
            array_proc = subprocess.run(["sbatch", "--parsable", str(array_sbatch)], check=True, text=True, capture_output=True)
            array_job = array_proc.stdout.strip().split(";")[0]
            merge_proc = subprocess.run(["sbatch", "--parsable", f"--dependency=afterok:{array_job}", str(merge_sbatch)], check=True, text=True, capture_output=True)
            merge_job = merge_proc.stdout.strip().split(";")[0]
            job_id = f"{array_job},{merge_job}"
            submitted = True
        return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, array_sbatch, submitted=submitted, job_id=job_id, complete=False, plan=plan, checkpoint=checkpoint)

    raise ValueError(f"Unsupported annotation backend: {backend}")


def begin_annotation_checkpoint(
    workdir: Path,
    engine: str,
    input_vcf: Path,
    output_vcf: Path,
    plan: Path,
) -> Path:
    """Record that this engine's annotation is being made again, before anything runs.

    Every annotation script deletes the old output VCF first (rm -f) and
    writes checkpoints/annotation_<engine>.json only at its end, so the
    checkpoint is written complete:false with this run's input and plan before
    the scripts are written, for the local and the SLURM backend alike; the
    script's own checkpoint replaces it when the run finishes. Between the two,
    or when the job fails, the pipeline state (pipeline/state.py
    summarize_annotation) therefore never reads complete over a missing VCF.

    Other engines' checkpoints in the same workdir are left as they are --
    their VCFs still exist -- but each is marked ``superseded_by`` this
    engine, and the state reader follows the newest plan to the checkpoint
    that is current rather than the first name in alphabetical order.
    """
    now = now_iso()
    checkpoint = stage_checkpoint_path(workdir, f"annotation_{engine}")
    previous: dict[str, Any] = {}
    try:
        loaded = json.loads(checkpoint.read_text())
        previous = loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        pass
    state = {
        "stage": "annotation",
        "engine": engine,
        "complete": False,
        "started": now,
        "finished": None,
        "input_vcf": str(input_vcf),
        "output_vcf": str(output_vcf),
        "plan": str(plan),
        # What stood here, so a failed run can be told from a first run.
        "previous_finished": previous.get("finished"),
        "previous_output_vcf": previous.get("output_vcf"),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    for other in sorted(checkpoint.parent.glob("annotation_*.json")):
        if other == checkpoint:
            continue
        try:
            data = json.loads(other.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("superseded_by") == engine:
            continue
        data["superseded_by"] = engine
        data["superseded_at"] = now
        other.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    project = workdir / "project_state.json"
    try:
        data = json.loads(project.read_text()) if project.exists() else {}
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("annotation", {})[engine] = state
    data["updated"] = now
    project.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return checkpoint


def write_annotation_plan(
    *,
    workdir: Path,
    engine: str,
    input_vcf: Path,
    output_vcf: Path,
    backend: str,
    script: Path,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    plan = {
        "stage": "annotation",
        "engine": engine,
        "backend": backend,
        "input_vcf": str(input_vcf),
        "output_vcf": str(output_vcf),
        "script": str(script),
        "created": now_iso(),
    }
    if extra:
        plan.update(extra)
    path = workdir / "inputs" / f"annotation_{engine}_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def _write_slurm_wrapper(
    *,
    workdir: Path,
    engine: str,
    script: Path,
    mem: str,
    partition: Optional[str],
    tmp_bytes: int = 0,
) -> Path:
    sbatch = workdir / "scripts" / f"annotate_{engine}.sbatch"
    partition_line = f"#SBATCH --partition={shlex.quote(str(partition))}\n" if partition else ""
    workdir_q = shlex.quote(str(workdir))
    # Node scratch the job writes, asked of SLURM (see tmp_request).
    tmp_line = tmp_directive(tmp_bytes, partition)
    sbatch.write_text(f'''#!/usr/bin/env bash
{partition_line}#SBATCH --job-name={slurm_job_name(workdir, f"annotation_{engine}")}
#SBATCH --cpus-per-task={MERGE_THREADS}
#SBATCH --mem={shlex.quote(str(mem))}
{tmp_line}#SBATCH --output={workdir_q}/logs/annotate_{engine}_%j.out
#SBATCH --error={workdir_q}/logs/annotate_{engine}_%j.err

set -euo pipefail
bash {shlex.quote(str(script))}
''')
    sbatch.chmod(0o755)
    return sbatch


def run_annotation_script(
    *,
    engine: str,
    script_body: str,
    input_vcf: Path,
    output_vcf: Path,
    workdir: Path,
    backend: str = "local",
    submit: bool = False,
    dry_run: bool = False,
    partition: Optional[str] = None,
    mem: str = "16G",
    threads: int = 1,
    jobs: int = 0,
) -> AnnotationRunResult:
    backend = backend.lower().strip()
    workdir = _absolute(Path(workdir))
    _ensure_dirs(workdir)
    script = workdir / "scripts" / f"annotate_{engine}.sh"
    script.write_text(script_body)
    script.chmod(0o755)
    plan = write_annotation_plan(
        workdir=workdir,
        engine=engine,
        input_vcf=input_vcf,
        output_vcf=output_vcf,
        backend=backend,
        script=script,
        extra={"submit": submit, "dry_run": dry_run, "mem": mem, "partition": partition},
    )
    checkpoint = stage_checkpoint_path(workdir, f"annotation_{engine}")
    if not dry_run:
        begin_annotation_checkpoint(workdir, engine, input_vcf, output_vcf, plan)

    if dry_run:
        return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, script, submitted=False, complete=False, plan=plan, checkpoint=checkpoint)

    if backend == "local":
        log_out = workdir / "logs" / f"annotate_{engine}.out"
        log_err = workdir / "logs" / f"annotate_{engine}.err"
        with log_out.open("w") as out, log_err.open("w") as err:
            subprocess.run(["bash", str(script)], check=True, stdout=out, stderr=err)
        return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, script, submitted=False, complete=output_vcf.exists(), plan=plan, checkpoint=checkpoint)

    if backend == "slurm":
        # Only the whole-VCF ANNOVAR job writes to scratch.
        tmp_bytes = _size(input_vcf) * ANNOVAR_SPLIT_FACTOR if uses_scratch(script_body) else 0
        sbatch = _write_slurm_wrapper(workdir=workdir, engine=engine, script=script, mem=mem, partition=partition,
                                      tmp_bytes=tmp_bytes)
        job_id = None
        submitted = False
        if submit:
            _require_tool("sbatch")
            proc = subprocess.run(["sbatch", "--parsable", str(sbatch)], check=True, text=True, capture_output=True)
            job_id = proc.stdout.strip().split(";")[0]
            submitted = True
        return AnnotationRunResult(engine, backend, workdir, input_vcf, output_vcf, sbatch, submitted=submitted, job_id=job_id, complete=False, plan=plan, checkpoint=checkpoint)

    raise ValueError(f"Unsupported annotation backend: {backend}")



def _engine_chain_label(engines: list[str]) -> str:
    cleaned = [e.strip().lower() for e in engines if e and e.strip()]
    if not cleaned:
        raise ValueError("Annotation engine chain cannot be empty")
    return "_".join(cleaned)


def build_annotation_chain_shard_script(
    *,
    input_vcf: Path,
    workdir: Path,
    engines: list[str],
    annovar_config: Optional[AnnovarConfig],
    snpeff_config: Optional[SnpEffConfig],
    shards: int,
    source_shard_dir: Optional[Path] = None,
) -> str:
    """Build a per-shard script that runs an ordered annotation engine chain.

    Existing FreeBayes shard VCFs are consumed directly when source_shard_dir is
    provided. Otherwise the merged input VCF is split by record count as a
    fallback for non-FreeBayes inputs.
    """
    label = _engine_chain_label(engines)
    shard_dir = workdir / "temp" / f"{label}_shards"
    prep = _existing_shard_prep_bash(source_shard_dir, shard_dir) if source_shard_dir else _shard_prep_bash(input_vcf, shard_dir, shards)
    lines = [
        _common_script_header(),
        "require_tool bcftools",
    ]
    if "annovar" in engines:
        if annovar_config is None:
            raise ValueError("ANNOVAR config is required when annovar is in the annotation engine chain")
        lines.append("require_tool perl")
    if "snpeff" in engines:
        if snpeff_config is None:
            raise ValueError("SnpEff config is required when snpeff is in the annotation engine chain")
        lines.append("require_tool java")
    lines.append(prep)
    if "annovar" in engines:
        # Split before the whole chain, so every engine annotates the same records.
        lines.append(_normalized_shard_bash(annovar_config.reference_fasta))
    lines.extend([
        f"CHAIN_LABEL={shlex.quote(label)}",
        "CHAIN_DIR=$(printf '%s/output/shard_%04d/chain' \"$SHARD_DIR\" \"$SHARD_ID\")",
        "mkdir -p \"$CHAIN_DIR\"",
        "CURRENT_VCF=\"$SHARD_INPUT\"",
        f"SHARD_GZ=$(printf '%s/output/shard_%04d.{label}.vcf.gz' \"$SHARD_DIR\" \"$SHARD_ID\")",
        "SHARD_ID_MAP=$(printf '%s/output/shard_%04d.info_id_renames.tsv' \"$SHARD_DIR\" \"$SHARD_ID\")",
        _empty_shard_bash('"$SHARD_ID_MAP"'),
        "echo \"Running annotation engine chain ${CHAIN_LABEL} on shard $SHARD_ID\"",
    ])

    for engine in engines:
        if engine == "annovar":
            cfg = annovar_config
            assert cfg is not None
            lines.extend([
                "ANNOVAR_PREFIX=\"$CHAIN_DIR/annovar\"",
                f"ANNOVAR_VCF=\"${{ANNOVAR_PREFIX}}.{cfg.buildver}_multianno.vcf\"",
                "rm -f \"$ANNOVAR_VCF\" \"${ANNOVAR_VCF}.gz\"",
                f"perl {shlex.quote(str(cfg.table_annovar))} \\",
                "  \"$CURRENT_VCF\" \\",
                f"  {shlex.quote(str(cfg.database_dir))} \\",
                f"  -buildver {shlex.quote(cfg.buildver)} \\",
                "  -out \"$ANNOVAR_PREFIX\" \\",
                "  -remove \\",
                f"  -protocol {shlex.quote(cfg.protocol)} \\",
                f"  -operation {shlex.quote(cfg.operation)} \\",
                f"  -nastring {shlex.quote(cfg.nastring)} \\",
                "  -vcfinput",
                "if [[ ! -s \"$ANNOVAR_VCF\" ]]; then echo \"ERROR: ANNOVAR produced no VCF for shard $SHARD_ID\" >&2; exit 1; fi",
                info_id_rename_bash('"$ANNOVAR_VCF"', '"$SHARD_ID_MAP"'),
                "CURRENT_VCF=\"$ANNOVAR_VCF\"",
            ])
        elif engine == "snpeff":
            cfg = snpeff_config
            assert cfg is not None
            lines.extend([
                "SNPEFF_VCF=\"$CHAIN_DIR/snpeff.vcf\"",
                "rm -f \"$SNPEFF_VCF\" \"${SNPEFF_VCF}.gz\"",
                f"{shlex.quote(cfg.java)} -XX:ActiveProcessorCount=1 -Djava.util.concurrent.ForkJoinPool.common.parallelism=1 -Xmx{shlex.quote(cfg.memory)} \\",
                f"  -jar {shlex.quote(str(cfg.jar))} \\",
                f"  -c {shlex.quote(str(cfg.config))} \\",
                f"  {shlex.quote(cfg.genome)} \\",
                "  \"$CURRENT_VCF\" > \"$SNPEFF_VCF\"",
                "if [[ ! -s \"$SNPEFF_VCF\" ]]; then echo \"ERROR: SnpEff produced empty output for shard $SHARD_ID\" >&2; exit 1; fi",
                "CURRENT_VCF=\"$SNPEFF_VCF\"",
            ])
        else:
            raise ValueError(f"Unsupported annotation engine in chain: {engine}")

    lines.extend([
        "rm -f \"$SHARD_GZ\" \"${SHARD_GZ}.tbi\"",
        "bgzip -c \"$CURRENT_VCF\" > \"$SHARD_GZ\"",
        "tabix -f -p vcf \"$SHARD_GZ\"",
        "echo \"Wrote $SHARD_GZ\"",
        "",
    ])
    return "\n".join(lines)


def build_annotation_chain_merge_script(
    *,
    engines: list[str],
    input_vcf: Path,
    output_vcf: Path,
    workdir: Path,
    shards: int,
) -> str:
    label = _engine_chain_label(engines)
    shard_dir = workdir / "temp" / f"{label}_shards"
    duplicates = _duplicates_path(workdir, label)
    joined = _joined_path(workdir, label)
    id_map = _info_id_map_path(workdir, label) if "annovar" in engines else None
    return "\n".join([
        _common_script_header(),
        "require_tool bcftools",
        f"SHARD_DIR={shlex.quote(str(shard_dir))}",
        f"OUTPUT_VCF={shlex.quote(str(output_vcf))}",
        f"SHARDS={int(shards)}",
        f"CHAIN_LABEL={shlex.quote(label)}",
        f"mkdir -p {shlex.quote(str(workdir / 'temp'))} \"$(dirname \"$OUTPUT_VCF\")\"",
        "FIRST=$(printf '%s/output/shard_%04d.%s.vcf.gz' \"$SHARD_DIR\" 1 \"$CHAIN_LABEL\")",
        "if [[ ! -s \"$FIRST\" ]]; then echo \"ERROR: first annotated shard output missing: $FIRST\" >&2; exit 1; fi",
        "rm -f \"$OUTPUT_VCF\" \"${OUTPUT_VCF}.tbi\"",
        "SHARD_PATHS=()",
        "for i in $(seq 1 \"$SHARDS\"); do",
        "  SHARD=$(printf '%s/output/shard_%04d.%s.vcf.gz' \"$SHARD_DIR\" \"$i\" \"$CHAIN_LABEL\")",
        "  if [[ ! -s \"$SHARD\" ]]; then echo \"ERROR: annotated shard output missing: $SHARD\" >&2; exit 1; fi",
        "  SHARD_PATHS+=(\"$SHARD\")",
        "done",
        _merge_header_bash('"$FIRST"'),
        *(_gather_id_maps_bash(id_map) if id_map else []),
        *_stream_merge_bash(duplicates, joined),
        "tabix -f -p vcf \"$OUTPUT_VCF\"",
        "if [[ ! -s \"$OUTPUT_VCF\" ]]; then echo \"ERROR: sorted annotation VCF missing: $OUTPUT_VCF\" >&2; exit 1; fi",
        "if [[ ! -s \"${OUTPUT_VCF}.tbi\" ]]; then echo \"ERROR: annotation VCF index missing: ${OUTPUT_VCF}.tbi\" >&2; exit 1; fi",
        "echo \"Wrote $OUTPUT_VCF and ${OUTPUT_VCF}.tbi\"",
        _checkpoint_python(workdir, label, output_vcf, input_vcf, id_map, duplicates, joined),
        "",
    ])


def run_annotation_engine_chain(
    *,
    engines: list[str],
    vcf: Path,
    workdir: Path,
    annovar_config: Optional[AnnovarConfig] = None,
    snpeff_config: Optional[SnpEffConfig] = None,
    prefix: Optional[str] = None,
    backend: str = "local",
    submit: bool = False,
    dry_run: bool = False,
    partition: Optional[str] = None,
    mem: str = "16G",
    threads: int = 1,
    jobs: int = 0,
    variant_workdir: Optional[Path] = None,
    shard_vcf_dir: Optional[Path] = None,
) -> AnnotationRunResult:
    """Run one ordered annotation chain over each shard, then merge once."""
    clean_engines = [e.strip().lower() for e in engines if e and e.strip()]
    if not clean_engines:
        raise ValueError("At least one annotation engine is required")
    if len(clean_engines) == 1:
        engine = clean_engines[0]
        if engine == "annovar":
            if annovar_config is None:
                raise ValueError("ANNOVAR config is required")
            return run_annovar_annotation(
                vcf=vcf,
                workdir=workdir,
                annovar_dir=annovar_config.annovar_dir,
                annovar_db=annovar_config.database_dir,
                buildver=annovar_config.buildver,
                protocol=annovar_config.protocol,
                operation=annovar_config.operation,
                reference_fasta=annovar_config.reference_fasta,
                prefix=prefix,
                backend=backend,
                submit=submit,
                dry_run=dry_run,
                partition=partition,
                mem=mem,
                threads=threads,
                jobs=jobs,
                variant_workdir=variant_workdir,
                shard_vcf_dir=shard_vcf_dir,
            )
        if engine == "snpeff":
            if snpeff_config is None:
                raise ValueError("SnpEff config is required")
            return run_snpeff_annotation(
                vcf=vcf,
                workdir=workdir,
                jar=snpeff_config.jar,
                config=snpeff_config.config,
                genome=snpeff_config.genome,
                memory=snpeff_config.memory,
                prefix=prefix,
                backend=backend,
                submit=submit,
                dry_run=dry_run,
                partition=partition,
                mem=mem,
                threads=threads,
                jobs=jobs,
                variant_workdir=variant_workdir,
                shard_vcf_dir=shard_vcf_dir,
            )
        raise ValueError(f"Unsupported annotation engine: {engine}")

    label = _engine_chain_label(clean_engines)
    vcf = _absolute(Path(vcf))
    workdir = _absolute(Path(workdir))
    if annovar_config is not None:
        annovar_config = replace(
            annovar_config,
            annovar_dir=_absolute(Path(annovar_config.annovar_dir)),
            database_dir=_absolute(Path(annovar_config.database_dir)),
            reference_fasta=_absolute(annovar_config.reference_fasta),
        )
    if snpeff_config is not None:
        snpeff_config = replace(
            snpeff_config,
            jar=_absolute(Path(snpeff_config.jar)),
            config=_absolute(Path(snpeff_config.config)),
        )
    _ensure_dirs(workdir)
    out = workdir / "results" / f"{_run_prefix(workdir, prefix)}.{label}.annotated.vcf.gz"
    source_shard_dir, source_shards = _resolve_source_shards(variant_workdir, shard_vcf_dir)
    shards, _ = _resolve_shard_count(threads=threads, jobs=jobs)
    effective_shards = len(source_shards) if source_shards else shards
    shard_body = build_annotation_chain_shard_script(
        input_vcf=vcf,
        workdir=workdir,
        engines=clean_engines,
        annovar_config=annovar_config,
        snpeff_config=snpeff_config,
        shards=effective_shards,
        source_shard_dir=source_shard_dir,
    )
    merge_body = build_annotation_chain_merge_script(
        engines=clean_engines,
        input_vcf=vcf,
        output_vcf=out,
        workdir=workdir,
        shards=effective_shards,
    )
    return run_sharded_annotation_script(
        engine=label,
        shard_script_body=shard_body,
        merge_script_body=merge_body,
        input_vcf=vcf,
        output_vcf=out,
        workdir=workdir,
        backend=backend,
        submit=submit,
        dry_run=dry_run,
        partition=partition,
        mem=mem,
        threads=threads,
        jobs=jobs,
        source_shard_dir=source_shard_dir,
        source_shards=source_shards,
    )

def run_snpeff_annotation(
    *,
    vcf: Path,
    workdir: Path,
    jar: Path,
    config: Path,
    genome: str,  # Required; it defaulted to the human GRCh38.99
    memory: str = "8g",
    prefix: Optional[str] = None,
    backend: str = "local",
    submit: bool = False,
    dry_run: bool = False,
    partition: Optional[str] = None,
    mem: str = "16G",
    threads: int = 1,
    jobs: int = 0,
    variant_workdir: Optional[Path] = None,
    shard_vcf_dir: Optional[Path] = None,
) -> AnnotationRunResult:
    vcf = _absolute(Path(vcf))
    workdir = _absolute(Path(workdir))
    _ensure_dirs(workdir)
    cfg = SnpEffConfig(jar=_absolute(Path(jar)), config=_absolute(Path(config)), genome=genome, memory=memory)
    cfg.validate()
    out = workdir / "results" / f"{_run_prefix(workdir, prefix)}.snpeff.annotated.vcf.gz"
    source_shard_dir, source_shards = _resolve_source_shards(variant_workdir, shard_vcf_dir)
    shards, _ = _resolve_shard_count(threads=threads, jobs=jobs)
    if source_shards or shards > 1:
        effective_shards = len(source_shards) if source_shards else shards
        shard_body = build_snpeff_shard_script(input_vcf=vcf, workdir=workdir, config=cfg, shards=effective_shards, source_shard_dir=source_shard_dir)
        merge_body = build_shard_merge_script(engine="snpeff", input_vcf=vcf, output_vcf=out, workdir=workdir, shards=effective_shards)
        return run_sharded_annotation_script(engine="snpeff", shard_script_body=shard_body, merge_script_body=merge_body, input_vcf=vcf, output_vcf=out, workdir=workdir, backend=backend, submit=submit, dry_run=dry_run, partition=partition, mem=mem, threads=threads, jobs=jobs, source_shard_dir=source_shard_dir, source_shards=source_shards)
    body = build_snpeff_script(input_vcf=vcf, output_vcf=out, workdir=workdir, config=cfg)
    return run_annotation_script(engine="snpeff", script_body=body, input_vcf=vcf, output_vcf=out, workdir=workdir, backend=backend, submit=submit, dry_run=dry_run, partition=partition, mem=mem)


def run_annovar_annotation(
    *,
    vcf: Path,
    workdir: Path,
    annovar_dir: Path,
    annovar_db: Path,
    # Required; they defaulted to hg38 and a human protocol list.
    buildver: str,
    protocol: str,
    operation: str,
    reference_fasta: Optional[Path] = None,
    prefix: Optional[str] = None,
    backend: str = "local",
    submit: bool = False,
    dry_run: bool = False,
    partition: Optional[str] = None,
    mem: str = "16G",
    threads: int = 1,
    jobs: int = 0,
    variant_workdir: Optional[Path] = None,
    shard_vcf_dir: Optional[Path] = None,
) -> AnnotationRunResult:
    vcf = _absolute(Path(vcf))
    workdir = _absolute(Path(workdir))
    _ensure_dirs(workdir)
    cfg = AnnovarConfig(annovar_dir=_absolute(Path(annovar_dir)), database_dir=_absolute(Path(annovar_db)), buildver=buildver, protocol=protocol, operation=operation,
                        reference_fasta=_absolute(Path(reference_fasta)) if reference_fasta else None)
    cfg.validate()
    out = workdir / "results" / f"{_run_prefix(workdir, prefix)}.annovar.annotated.vcf.gz"
    source_shard_dir, source_shards = _resolve_source_shards(variant_workdir, shard_vcf_dir)
    shards, _ = _resolve_shard_count(threads=threads, jobs=jobs)
    if source_shards or shards > 1:
        effective_shards = len(source_shards) if source_shards else shards
        shard_body = build_annovar_shard_script(input_vcf=vcf, workdir=workdir, config=cfg, shards=effective_shards, source_shard_dir=source_shard_dir)
        merge_body = build_shard_merge_script(engine="annovar", input_vcf=vcf, output_vcf=out, workdir=workdir, shards=effective_shards)
        return run_sharded_annotation_script(engine="annovar", shard_script_body=shard_body, merge_script_body=merge_body, input_vcf=vcf, output_vcf=out, workdir=workdir, backend=backend, submit=submit, dry_run=dry_run, partition=partition, mem=mem, threads=threads, jobs=jobs, source_shard_dir=source_shard_dir, source_shards=source_shards)
    body = build_annovar_script(input_vcf=vcf, output_vcf=out, workdir=workdir, config=cfg)
    return run_annotation_script(engine="annovar", script_body=body, input_vcf=vcf, output_vcf=out, workdir=workdir, backend=backend, submit=submit, dry_run=dry_run, partition=partition, mem=mem)
