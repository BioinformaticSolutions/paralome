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

"""BWA (bwa-mem2) beside STAR: ``--aligner bwa``.

The alignment stage is the same stage whichever aligner runs: the same task script
(paralome.align.star.write_star_slurm, which runs locally too), the same
``bam/<sample>.sorted.bam`` or fused duplicate-marking chain, the same checkpoint. Only the aligner's
command differs, and what this module gives it:

* the index: bwa-mem2's files for the genome's FASTA, under ``<FASTA folder>/bwa-mem2/<FASTA name>.*``.
  Built on the first BWA run: here before the tasks run locally, or as one SLURM
  job the alignment array waits on (afterok). Built into a folder of its own and moved into place, so two
  runs never read a half-written index.
* the command: ``bwa-mem2 mem`` with the read group STAR is given (ID/SM/LB/PU the sample, PL ILLUMINA),
  its SAM piped into ``samtools sort`` (or into duplicate marking when that runs in the task).
* the mapping metrics: ``samtools flagstat`` and the reads at MAPQ >= 1, written beside the task's logs and
  read into the same per-sample table STAR's Log.final.out fills (paralome.align.metrics).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from paralome.binaries import resolve_tool

ALIGNERS = ("star", "bwa")
BWA_TOOL = "bwa-mem2"
#: The files ``bwa-mem2 index -p PREFIX`` writes.
INDEX_SUFFIXES = (".0123", ".amb", ".ann", ".bwt.2bit.64", ".pac")
#: Each BWA task writes these beside its logs: the metrics table reads them (align.metrics).
FLAGSTAT_SUFFIX = ".bwa.flagstat.json"
MAPQ1_SUFFIX = ".bwa.mapq1.txt"
#: What the pair filter kept and dropped, and why (FILTER_AWK): the mapping table reads it.
FILTER_SUFFIX = ".bwa.filter.tsv"

#: bwa-mem2 has no setting for any of STAR's output filters, so its SAM goes through this before duplicate marking. A read
#: pair (all records of one name; bwa-mem2 writes them together) is kept only where STAR, run as Paralome runs
#: it, would have written it, on the run's own STAR settings:
#:   both mates mapped, a proper pair (0x2), and no farther apart than the DNA-mode mate gap (``gap``);
#:   mismatches (NM less inserted and deleted bases), both mates together, <= ``frac`` of their read length;
#:   aligned (M/=/X) bases, both mates together, >= ``matchf`` of their read length;
#:   at most ``nmax`` places for the pair: a mate whose XS is more than 1 below its AS has one place (STAR's
#:   --outFilterMultimapScoreRange 1); otherwise its places are the primary and the XA entries no worse
#:   (NM), and with no XA (bwa-mem2 lists 5 or fewer) more than 5. A pair's places are the combinations
#:   on one chromosome, on opposite strands, within the gap; a unique mate fixes the pair.
#: Kept pairs pass unchanged (placement and MAPQ are bwa-mem2's); the counts go to ``logf``. Plain awk
#: (no gawk-only functions), one pass, one pair held at a time.
FILTER_AWK = r'''function bit(f, b) { return int(f / b) % 2 }
function flush(    i, f, n, prim, m1, m2, keep, why) {
    if (ng == 0) return
    n = 0
    for (i = 1; i <= ng; i++) { f = flag[i]; if (bit(f, 256) == 0 && bit(f, 2048) == 0) { n++; prim[n] = i } }
    total++
    why = verdict(prim, n)
    if (why == "") { kept++; for (i = 1; i <= ng; i++) print rec[i] }
    else dropped[why]++
    ng = 0
}
function tag(line, name,    k, t, nf, fld) {
    nf = split(line, fld, "\t")
    for (k = 12; k <= nf; k++) if (substr(fld[k], 1, 5) == name ":") return substr(fld[k], 6)
    return ""
}
function cig(c, ops,    s, num, ch, k, out) {
    out = 0; num = ""
    for (k = 1; k <= length(c); k++) {
        ch = substr(c, k, 1)
        if (ch ~ /[0-9]/) num = num ch
        else { if (index(ops, ch)) out += num + 0; num = "" }
    }
    return out
}
function places(i, P,    as, xs, nm, xa, a, k, e, cnt, fld) {
    # P[1..cnt] = "chr\tstrand\tpos" of the equally good places of record i; returns cnt, or 99 for "many".
    split(rec[i], fld, "\t")
    cnt = 1; P[1] = fld[3] "\t" (bit(flag[i], 16) ? "-" : "+") "\t" fld[4]
    as = tag(rec[i], "AS:i"); xs = tag(rec[i], "XS:i"); nm = tag(rec[i], "NM:i") + 0
    if (xs == "" || xs + 0 < as - 1) return 1
    xa = tag(rec[i], "XA:Z")
    if (xa == "") return 99
    n = split(xa, a, ";")
    for (k = 1; k <= n; k++) {
        if (a[k] == "") continue
        split(a[k], e, ",")
        if (e[4] + 0 <= nm) { cnt++; P[cnt] = e[1] "\t" substr(e[2], 1, 1) "\t" substr(e[2], 2) }
    }
    return cnt
}
function verdict(prim, n,    i, j, f, f1, f2, l1, l2, seq, mm, mt, tl, g, c1, c2, P1, P2, a, b, joint, x, y, fld) {
    if (n == 0) return "unmapped"
    for (i = 1; i <= n; i++) if (bit(flag[prim[i]], 4)) return "unmapped"
    f1 = flag[prim[1]]
    if (bit(f1, 1)) {
        if (n < 2) return "unmapped"
        if (!bit(f1, 2) || !bit(flag[prim[2]], 2)) return "improper_pair"
    }
    mm = 0; mt = 0; seq = 0
    for (i = 1; i <= n; i++) {
        split(rec[prim[i]], fld, "\t")
        seq += length(fld[10])
        mm += tag(rec[prim[i]], "NM:i") - cig(fld[6], "ID")
        mt += cig(fld[6], "M=X")
        if (i == 1) tl = fld[9]
    }
    if (gap > 0 && n == 2) {
        tl = tl < 0 ? -tl : tl
        if (tl - seq > gap) return "mates_too_far"
    }
    if (mm > frac * seq) return "too_many_mismatches"
    if (mt < matchf * seq) return "too_short"
    c1 = places(prim[1], P1)
    if (n < 2) return (c1 > nmax ? "too_many_places" : "")
    c2 = places(prim[2], P2)
    if (c1 == 1 || c2 == 1) return ""
    if (c1 == 99 && c2 == 99) return "too_many_places"
    if (c1 == 99 || c2 == 99) return ""
    joint = 0
    for (x = 1; x <= c1; x++) for (y = 1; y <= c2; y++) {
        split(P1[x], a, "\t"); split(P2[y], b, "\t")
        if (a[1] != b[1] || a[2] == b[2]) continue
        d = a[3] - b[3]; if (d < 0) d = -d
        if (d <= (gap > 0 ? gap : 1000) + seq) joint++
    }
    return (joint > nmax ? "too_many_places" : "")
}
/^@/ { print; next }
{
    split($0, q, "\t")
    if (q[1] != name) { flush(); name = q[1] }
    ng++; rec[ng] = $0; flag[ng] = q[2] + 0
}
END {
    flush()
    printf "pairs_in\t%d\nkept\t%d\n", total, kept > logf
    split("unmapped improper_pair mates_too_far too_many_mismatches too_short too_many_places", r, " ")
    for (i = 1; i <= 6; i++) printf "%s\t%d\n", r[i], dropped[r[i]] + 0 > logf
}'''


def normalize_aligner(value: Optional[str]) -> str:
    """``star`` or ``bwa`` (any case); anything else is refused by name."""
    name = str(value or "star").strip().lower()
    if name not in ALIGNERS:
        raise ValueError(f"--aligner must be star or bwa, got {value!r}")
    return name


def index_prefix(fasta: Path) -> Path:
    """Where the genome's bwa-mem2 index lives: ``<FASTA folder>/bwa-mem2/<FASTA name>``."""
    fasta = Path(fasta)
    return fasta.parent / "bwa-mem2" / fasta.name


def index_ready(fasta: Path) -> bool:
    prefix = index_prefix(fasta)
    return all(Path(str(prefix) + suffix).is_file() and Path(str(prefix) + suffix).stat().st_size
               for suffix in INDEX_SUFFIXES)


def index_bytes(fasta: Path) -> Optional[int]:
    """The index on disk (what bwa-mem2 holds in memory while aligning), or None before it is built."""
    if not index_ready(fasta):
        return None
    prefix = index_prefix(fasta)
    return sum(Path(str(prefix) + suffix).stat().st_size for suffix in INDEX_SUFFIXES)


#: Before the first BWA run builds it: the bwa-mem2 index (.0123, .bwt.2bit.64 and the rest) comes to about
#: this many times the FASTA's bytes (GRCh38: 3.1 GB of FASTA, about 19 GB of index).
INDEX_PER_FASTA_BYTE = 6


def memory_index(fasta: Path) -> tuple[int, str]:
    """What the task's memory is sized from: the index bwa-mem2 holds while it aligns."""
    built = index_bytes(fasta)
    if built is not None:
        return built, "bwa-mem2 index on disk"
    return (INDEX_PER_FASTA_BYTE * Path(fasta).stat().st_size,
            f"bwa-mem2 index not built yet: {INDEX_PER_FASTA_BYTE} x the FASTA")


def require_fasta(fasta: Path) -> Path:
    fasta = Path(fasta)
    if not fasta.is_file():
        raise FileNotFoundError(
            f"--aligner bwa aligns against the genome's FASTA, and {fasta} is not a file. Give the FASTA as "
            "--reference (the pipeline passes the --genome entry's reference)")
    return Path(os.path.abspath(fasta))


def index_build_bash(fasta: Path, *, bwa_exe: Optional[str] = None) -> str:
    """Bash that builds the index into a folder of its own and moves it into place (skips a finished one)."""
    fasta = Path(fasta)
    prefix = index_prefix(fasta)
    exe = shlex.quote(bwa_exe or resolve_tool(BWA_TOOL) or BWA_TOOL)
    ready = " && ".join(f"-s {shlex.quote(str(prefix) + s)}" for s in INDEX_SUFFIXES)
    return "\n".join([
        f"if [[ {ready} ]]; then",
        f"  echo \"bwa-mem2 index already built: {prefix}\"",
        "else",
        f"  mkdir -p {shlex.quote(str(prefix.parent))}",
        f"  BUILDING={shlex.quote(str(prefix.parent))}/.building_$$",
        "  rm -rf \"$BUILDING\" && mkdir -p \"$BUILDING\"",
        f"  {exe} index -p \"$BUILDING\"/{shlex.quote(fasta.name)} {shlex.quote(str(fasta))}",
        *[f"  mv -f \"$BUILDING\"/{shlex.quote(fasta.name + s)} {shlex.quote(str(prefix) + s)}" for s in INDEX_SUFFIXES],
        "  rm -rf \"$BUILDING\"",
        f"  echo \"bwa-mem2 index built: {prefix}\"",
        "fi",
    ])


def build_index_here(fasta: Path, *, echo=print) -> Path:
    """Build the index on this machine when it is not there (the local backend's first BWA run)."""
    fasta = require_fasta(fasta)
    prefix = index_prefix(fasta)
    if index_ready(fasta):
        return prefix
    exe = resolve_tool(BWA_TOOL) or shutil.which(BWA_TOOL)
    if not exe:
        raise FileNotFoundError(f"{BWA_TOOL} is not installed or not on PATH; it builds the index --aligner bwa needs")
    echo(f"[paralome] alignment: building the bwa-mem2 index for {fasta} (first BWA run on this genome)")
    subprocess.run(["bash", "-c", "set -euo pipefail\n" + index_build_bash(fasta, bwa_exe=exe)], check=True)
    return prefix


def filter_settings(*, mismatch_fraction: float, match_fraction: float, multimap_nmax: int,
                    mates_gap: Optional[int]) -> dict:
    """The pair filter's thresholds: the run's STAR settings (FILTER_AWK)."""
    return {"frac": float(mismatch_fraction), "matchf": float(match_fraction), "nmax": int(multimap_nmax),
            "gap": int(mates_gap or 0)}


def result_arguments(*, extra_args: str = "", sorted_output: bool = True, pair_filter: Optional[dict] = None) -> list[str]:
    """What decides a BWA BAM besides the reads and the genome: recorded with each sample (its request)."""
    args = [BWA_TOOL, "mem", *(extra_args.strip().split() if extra_args.strip() else [])]
    if pair_filter:
        args += ["|", "pair filter", *(f"{k}={pair_filter[k]}" for k in sorted(pair_filter))]
    return args + (["|", "samtools", "sort"] if sorted_output else ["|", "duplicate marking"])


def filter_line(pair_filter: dict, counts: str = '"$WORKDIR/logs/${SAMPLE_ID}' + FILTER_SUFFIX + '"') -> str:
    """The pipe stage that runs FILTER_AWK with ``pair_filter``'s thresholds."""
    variables = " ".join(f"-v {k}={pair_filter[k]}" for k in ("frac", "matchf", "nmax", "gap"))
    return f"  | awk {variables} -v logf={counts} {shlex.quote(FILTER_AWK)} \\"


def align_lines(*, bwa_exe: str, extra_args: str, read_command: Optional[str], fused_chain: Optional[list[str]],
                sorted_bam: str, pair_filter: Optional[dict] = None) -> list[str]:
    """The task's alignment: bwa-mem2 mem on ``READS`` (through ``read_command`` when they are the fastp
    FIFOs), into samtools sort (``sorted_bam``) or, given ``fused_chain``, into duplicate marking."""
    extra = " ".join(shlex.quote(x) for x in extra_args.split()) if extra_args.strip() else ""
    head = (f"{bwa_exe} mem -t \"$THREADS\" "
            "-R \"@RG\\tID:${SAMPLE_ID}\\tSM:${SAMPLE_ID}\\tPL:ILLUMINA\\tLB:${SAMPLE_ID}\\tPU:${SAMPLE_ID}\" "
            + (extra + " " if extra else "") + "\"$BWA_INDEX\"")
    if fused_chain:
        tail = list(fused_chain)
    else:
        tail = [f'  | "$SAMTOOLS" sort -@ "$THREADS" -m "${{SORT_MEM_MB}}M" -T "$STAR_SCRATCH/${{SAMPLE_ID}}.possort" '
                f'-o {sorted_bam} -']
    if pair_filter:
        tail = [filter_line(pair_filter), *tail]
    lines = ["", "# bwa-mem2 (--aligner bwa): its SAM goes straight on, never to a file."]
    if not fused_chain:
        lines += ['SORT_MEM_MB=$(( STAR_SORT_RAM / 1048576 / THREADS ))',
                  # samtools takes each thread's share at once (STAR sorts lazily up to its limit): at most
                  # 1 GB a thread, so the sort never asks for more than the task was given.
                  'if (( SORT_MEM_MB > 1024 )); then SORT_MEM_MB=1024; fi',
                  'if (( SORT_MEM_MB < 256 )); then SORT_MEM_MB=256; fi']
    if read_command:
        # The trimmed reads arrive in FIFOs; bwa-mem2 reads the mates through the buffer STAR is given
        # (stream.fused_read_files_command), so neither of fastp's writers blocks on the other.
        lines += ['if [[ ${#READS[@]} -eq 2 ]]; then',
                  f'{head} <({read_command} "${{READS[0]}}") <({read_command} "${{READS[1]}}") \\', *tail,
                  "else",
                  f'{head} <({read_command} "${{READS[0]}}") \\', *tail,
                  "fi"]
    else:
        lines += [f'{head} "${{READS[@]}}" \\', *tail]
    return lines


def metrics_lines() -> list[str]:
    """After the BAM is in place: the numbers the mapping table reads (align.metrics)."""
    return [
        f'"$SAMTOOLS" flagstat -O json "$BAM" > "$WORKDIR/logs/${{SAMPLE_ID}}{FLAGSTAT_SUFFIX}"',
        f'"$SAMTOOLS" view -c -q 1 -F 0x904 "$BAM" > "$WORKDIR/logs/${{SAMPLE_ID}}{MAPQ1_SUFFIX}"',
    ]


def index_job_lines(fasta: Path) -> str:
    """The body of the one SLURM job that builds a missing index before the alignment array."""
    return "set -euo pipefail\n" + index_build_bash(fasta)


def threads_env() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
