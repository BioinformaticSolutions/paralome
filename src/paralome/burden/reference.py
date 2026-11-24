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

"""The per-gene gnomAD expected table, built once per genome.

The gene_burden stage compares how many probands carry a qualifying allele in a
gene with how many gnomAD says to expect. The expectation is a property of the
genome and its ANNOVAR databases, not of any cohort, so it is built here, once,
at genome setup, and every DNA project on that genome reads it.

WHAT IT IS. For every Ensembl gene x tier x model:

    q_pop = sum of gnomAD AF_pop over the alleles in the gene that qualify for
            the tier (burden/qualify.py) and pass the model's frequency gate,
            which is judged on AF_grpmax (dominant < 0.001, recessive < 0.01)

for pop in all (gnomAD's overall AF), afr, amr, asj, eas, fin, mid, nfe, sas.
An allele with no AF for a population adds nothing to that population's sum.
Only genes with at least one qualifying allele get a row.

The second power sums as well,

    s2_pop   = sum of AF_pop^2              (one per population, and 'all')
    s2_a_b   = sum of AF_a x AF_b           (one per pair of populations)

over the same alleles. The recessive carrier probability needs them
(burden/stats.py: P(2 or more qualifying alleles) = 1 - exp(-(2q+s2))(1+2q+2s2)),
and an admixed proband's s2 is a weighted sum over population PAIRS, which is
why the cross sums are kept. A table without them is still read (the stage
falls back to the conservative Poisson form and says so).

RE-RUNS. The plan is kept in work/plan.json with a signature: each
gnomAD table's path, size and modification time, the chunk size, the
qualifying thresholds, the ANNOVAR inputs (table_annovar.pl, the ensGene and
dbNSFP tables, buildver, protocols) and the partial format. Run again with the
same signature, the plan -- and its plan_id -- is reused, every chunk whose
partial is already there is kept, and only the missing chunks are computed
(locally, or as the SLURM array's task list). Anything in the signature
changed: a new plan_id, and every chunk is computed again; so too with
--rebuild once the earlier plan's build has finished (its provenance names
the plan_id), while an interrupted --rebuild resumes. The SLURM array has
no concurrency limit unless --max-parallel N is given (the
command gives 48 unless told otherwise -- 0 for none -- and the array runs
at --nice=100, behind a project's own jobs).

HOW. The ANNOVAR table ``<buildver>_gnomad41_exome.txt`` (tab separated,
header ``#Chr Start End Ref Alt gnomad41_exome_AF ...``, one row per allele in
ANNOVAR avinput coordinates) is cut into byte ranges -- the chunk count is
known before anything is read, so a SLURM array can be sized up front. Each
chunk keeps the rows that could qualify for either model (AF_grpmax < 0.01 or
absent), writes them as avinput with the AF columns carried as extra columns,
annotates them with

    table_annovar.pl <chunk> <humandb> -buildver B -out X -remove
        -protocol ensGene,<dbnsfp> -operation g,f -nastring . -otherinfo

(the dbNSFP table is the one the genome's annovar_protocol names) and reduces
the result to per-gene partial sums. The finalize step adds the partials, takes
gene names and spans from the genome's GTF (gene lines; the ensGene annotation
gives no coordinates), and writes the table and its provenance atomically.

THE REGULATORY TIER needs gnomAD GENOME frequencies. When
``<buildver>_gnomad41_genome.txt`` is in the database directory it is processed
the same way (protocol ensGene only, gene assignment as in qualify.py) into a
separate table; otherwise the provenance records that it is absent and why.

Outputs, in ``<genome folder>/burden/``:
    gnomad41_exome.<buildver>.burden_reference.tsv.gz
    gnomad41_genome.<buildver>.burden_reference.tsv.gz   (regulatory; optional)
    burden_reference.json                                (provenance)
    work/                                                (plan, scripts, logs, partials)
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import shlex
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from paralome.binaries import HUMAN_SCRIPT
from paralome.burden import qualify as Q

SCHEMA = "paralome.burden_reference.v1"
PLAN_SCHEMA = "paralome.burden_reference_plan.v1"
PARTIAL_SCHEMA = "paralome.burden_reference_partial.v2"   # + s2 sums

REFERENCE_DIRNAME = "burden"
PROVENANCE_NAME = "burden_reference.json"
WORK_DIRNAME = "work"

#: Chunks are cut by bytes, so their number is known before the table is read.
DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024
#: SLURM's default MaxArraySize is 1001; chunks grow rather than exceed it.
MAX_CHUNKS = 1000
#: 6G (was 16G). An ANNOVAR chunk peaks near 2.1 GB; 16G a task let only 31 of
#: them share a 512 GB node with 29 CPUs idle.
DEFAULT_MEM = "6G"
#: Array tasks at once unless --max-parallel says otherwise (0: no limit), and the array's
#: --nice: the build waits behind a project's own jobs rather than starving them.
DEFAULT_MAX_PARALLEL = 48
ARRAY_NICE = 100

#: Rows that cannot qualify under either model are dropped before ANNOVAR.
PREFILTER_MAX_AF = max(Q.MODEL_MAX_AF.values())

#: Per source: the tiers it feeds, and the ANNOVAR protocols it needs beyond ensGene.
SOURCE_TIERS = {"exome": Q.CODING_TIERS, "genome": (Q.REGULATORY_TIER,)}

TABLE_COLUMNS = ("#Chr", "Start", "End", "Ref", "Alt")

def _s2_keys() -> Tuple[str, ...]:
    """'all', each population, then each pair a_b (population order): stats.s2_key's spellings."""
    pops = Q.ALL_POPULATIONS
    pairs = tuple(f"{a}_{b}" for i, a in enumerate(pops) for b in pops[i + 1:])
    return ("all",) + pops + pairs


S2_KEYS = _s2_keys()

OUTPUT_COLUMNS = (
    "gene_id", "gene_name", "chrom", "start", "end", "tier", "model",
    "n_qualifying_alleles",
) + tuple(f"q_{pop}" for pop in Q.POPULATIONS_AND_ALL) + tuple(f"s2_{key}" for key in S2_KEYS)


class BurdenReferenceError(RuntimeError):
    """Raised by the planning checks; the command layer numbers it."""

    def __init__(self, code: int, detail: str, fix: Optional[str] = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.fix = fix


# ------------------------------------------------------------------- naming

def gnomad_table(annovar_db: Path, buildver: str, source: str) -> Path:
    return Path(annovar_db) / f"{buildver}_{Q.GNOMAD_SOURCES[source]}.txt"


def reference_filename(source: str, buildver: str) -> str:
    return f"{Q.GNOMAD_SOURCES[source]}.{buildver}.burden_reference.tsv.gz"


def reference_dir(genome_dir: Path) -> Path:
    return Path(genome_dir) / REFERENCE_DIRNAME


def reference_path(genome_dir: Path, buildver: str, source: str = "exome") -> Path:
    return reference_dir(genome_dir) / reference_filename(source, buildver)


def build_command(genome: str, backend: str = "slurm") -> str:
    """The command a person types to build it. Printed wherever it is missing."""
    tail = " --backend slurm --submit" if backend == "slurm" else ""
    # The name of THIS install's command.
    return f"{HUMAN_SCRIPT} setup genome burden-reference {shlex.quote(genome)}{tail}"


def dbnsfp_protocol(protocols: Optional[str]) -> Optional[str]:
    """The first dbNSFP protocol in a comma-separated ANNOVAR protocol list."""
    for item in str(protocols or "").split(","):
        if item.strip().lower().startswith("dbnsfp"):
            return item.strip()
    return None


_GENOME_FOLDER_KEYS = (
    "reference_fasta", "gtf", "star_index", "annovar_dir", "annovar_db",
    "snpeff_jar", "snpeff_config",
)


def genome_folder_of(entry: Mapping[str, Any]) -> Optional[Path]:
    """The folder a registered genome lives in: the deepest common ancestor of
    the paths its entry names (a genome folder holds its FASTA, GTF, index and
    ANNOVAR together; setup/genome_scan.py). None when nothing narrower than a
    filesystem root is shared, which means the entry is not one folder."""
    paths = []
    for key in _GENOME_FOLDER_KEYS:
        value = entry.get(key)
        if not value or not os.path.isabs(str(value)):
            continue
        path = Path(str(value)).expanduser()
        paths.append(str(path if path.is_dir() else path.parent))
    if not paths:
        return None
    common = Path(os.path.commonpath(paths))
    if len(common.parts) < 3:  # "/" or "/storage" names no genome
        return None
    return common


def gnomad_tables_present(annovar_db: Optional[str], buildver: Optional[str]) -> Dict[str, Path]:
    """{source: table} for the gnomAD tables in an ANNOVAR database directory."""
    if not annovar_db or not buildver:
        return {}
    found = {}
    for source in Q.GNOMAD_SOURCES:
        table = gnomad_table(Path(annovar_db), buildver, source)
        if table.is_file():
            found[source] = table
    return found


# -------------------------------------------------------------- table reading

def read_table_header(table: Path) -> List[str]:
    with open(table, "r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().rstrip("\r\n")
    return first.split("\t")


def required_af_columns(source: str) -> List[str]:
    return [Q.af_key(source, pop) for pop in Q.POPULATIONS_AND_ALL + ("grpmax",)]


def check_table_header(table: Path, source: str) -> Dict[str, int]:
    """Column name -> index of the five avinput columns and the AFs used.

    Columns are found by NAME, never by position: ANNOVAR's gnomAD tables
    carry more AF columns (raw, XX, XY, remaining, faf95, ...) than are read.
    """
    header = read_table_header(table)
    if [h.strip() for h in header[:5]] != list(TABLE_COLUMNS):
        raise BurdenReferenceError(
            106,
            f"{table} does not start with the ANNOVAR header "
            f"{' '.join(TABLE_COLUMNS)}; it starts {' '.join(header[:5])!r}",
            fix="Use the gnomAD table ANNOVAR's annotate_variation.pl -downdb installed",
        )
    index = {name.strip(): i for i, name in enumerate(header)}
    missing = [col for col in required_af_columns(source) if col not in index]
    if missing:
        raise BurdenReferenceError(
            106,
            f"{table} lacks the column(s) {', '.join(missing)}",
            fix="Use the gnomAD 4.1 table ANNOVAR's annotate_variation.pl -downdb installed",
        )
    wanted = list(TABLE_COLUMNS) + required_af_columns(source)
    return {name: index[name] for name in wanted}


def byte_chunks(size: int, chunk_bytes: int) -> List[Tuple[int, int]]:
    """[start, end) byte ranges covering ``size``; a row belongs to the range
    its first byte falls in (see iter_rows_in_range)."""
    if size <= 0:
        return [(0, 0)]
    count = max(1, math.ceil(size / chunk_bytes))
    step = math.ceil(size / count)
    return [(start, min(size, start + step)) for start in range(0, size, step)]


def iter_rows_in_range(table: Path, start: int, end: int) -> Iterator[List[str]]:
    """The data rows whose first byte lies in [start, end)."""
    with open(table, "rb") as handle:
        if start > 0:
            # Land at the start of the first row beginning at or after start.
            handle.seek(start - 1)
            handle.readline()
        while True:
            position = handle.tell()
            if position >= end:
                return
            raw = handle.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            yield line.split("\t")


# -------------------------------------------------------------------- plan

def _finished_plan_id(provenance: Path) -> Optional[str]:
    """The plan_id of the build whose provenance is at ``provenance``, or None (none, unreadable, or written recorded it -- treated as finished by another plan)."""
    try:
        value = json.loads(provenance.read_text(encoding="utf-8")).get("plan_id")
    except (OSError, ValueError, AttributeError):
        return None
    return str(value) if value else None


def plan_reference(
    *,
    genome: str,
    entry: Mapping[str, Any],
    genome_dir: Optional[Path] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """Check the genome can build a reference, and say how it will be built.

    Refuses (BurdenReferenceError) rather than building something partial: no
    ANNOVAR keys, no gnomAD exome table, no dbNSFP in the protocol list or its
    table missing, no ensGene table, or a reference already there without
    ``rebuild``.
    """
    annovar_dir = entry.get("annovar_dir")
    annovar_db = entry.get("annovar_db")
    buildver = entry.get("annovar_buildver")
    absent = [k for k, v in (("annovar_dir", annovar_dir), ("annovar_db", annovar_db),
                             ("annovar_buildver", buildver)) if not v]
    if absent:
        raise BurdenReferenceError(
            402, f"genome {genome} has no {', '.join(absent)} recorded",
            fix=f"Record them (`{HUMAN_SCRIPT} setup doctor {genome}` shows the entry)",
        )
    annovar_dir, annovar_db = Path(str(annovar_dir)), Path(str(annovar_db))
    script = annovar_dir / "table_annovar.pl"
    if not script.is_file():
        raise BurdenReferenceError(402, f"{script} is not there (annovar_dir of genome {genome})")

    exome = gnomad_table(annovar_db, buildver, "exome")
    if not exome.is_file():
        raise BurdenReferenceError(
            402,
            f"genome {genome} has no gnomAD exome table: {exome} is not there",
            fix=f"Install it with annotate_variation.pl -buildver {buildver} "
                f"-downdb -webfrom annovar gnomad41_exome {annovar_db}",
        )
    dbnsfp = dbnsfp_protocol(entry.get("annovar_protocol"))
    if dbnsfp is None:
        raise BurdenReferenceError(
            402,
            f"genome {genome}'s annovar_protocol ({entry.get('annovar_protocol') or 'none'}) "
            "names no dbNSFP table; the ptv_missense tier needs REVEL_score and "
            "AlphaMissense_pred from it",
            fix="Add the dbNSFP protocol (e.g. dbnsfp47a) to the genome's annovar_protocol",
        )
    for protocol in ("ensGene", dbnsfp):
        needed = annovar_db / f"{buildver}_{protocol}.txt"
        if not needed.is_file():
            raise BurdenReferenceError(
                402,
                f"genome {genome}: {needed} is not there, and the reference "
                f"is annotated with {protocol}",
            )

    folder = Path(genome_dir) if genome_dir else genome_folder_of(entry)
    if folder is None:
        raise BurdenReferenceError(
            402,
            f"genome {genome}'s paths share no genome folder to write burden/ into",
        )
    out_dir = reference_dir(folder)
    final = reference_path(folder, buildver, "exome")
    if final.exists() and not rebuild:
        raise BurdenReferenceError(
            308,
            f"{final} already exists",
            fix=f"Pass --rebuild to build it again: {build_command(genome, 'local')} --rebuild",
        )

    sources: Dict[str, Any] = {}
    genome_table = gnomad_table(annovar_db, buildver, "genome")
    for source, table in (("exome", exome), ("genome", genome_table)):
        if source == "genome" and not table.is_file():
            continue
        columns = check_table_header(table, source)
        stat = table.stat()
        sources[source] = {
            "table": str(table.resolve()),
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "mtime_ns": stat.st_mtime_ns,
            "columns": columns,
            "protocol": "ensGene" if source == "genome" else f"ensGene,{dbnsfp}",
            "operation": "g" if source == "genome" else "g,f",
            "output": str((out_dir / reference_filename(source, buildver)).resolve()),
        }

    total = sum(info["size"] for info in sources.values())
    # Each source rounds its last chunk up, so leave one chunk of room per source.
    chunk_bytes = max(int(chunk_bytes), math.ceil(total / (MAX_CHUNKS - len(sources))) if total else 1, 1)
    chunks = []
    for source, info in sources.items():
        for start, end in byte_chunks(info["size"], chunk_bytes):
            chunks.append({"index": len(chunks) + 1, "source": source, "start": start, "end": end})

    work = (out_dir / WORK_DIRNAME).resolve()
    gtf = entry.get("gtf")
    signature = plan_signature(sources=sources, chunk_bytes=chunk_bytes, annovar_dir=annovar_dir,
                               annovar_db=annovar_db, buildver=str(buildver), dbnsfp=dbnsfp)
    earlier = _earlier_plan(work / "plan.json")
    reused = bool(earlier) and earlier.get("signature") == signature and bool(earlier.get("plan_id"))
    if reused and rebuild and _finished_plan_id(out_dir / PROVENANCE_NAME) == str(earlier["plan_id"]):
        # --rebuild of a FINISHED build starts a new plan: it is how a change the signature cannot see
        # (ANNOVAR's own scripts replaced in place) is picked up. A rebuild that was itself interrupted
        # keeps its plan, so running it again computes only the chunks it had not finished.
        reused = False
    plan = {
        "schema": PLAN_SCHEMA,
        "genome": genome,
        "genome_dir": str(folder.resolve()),
        "out_dir": str(out_dir.resolve()),
        "work_dir": str(work),
        "provenance": str((out_dir / PROVENANCE_NAME).resolve()),
        "annovar_dir": str(annovar_dir.resolve()),
        "annovar_db": str(annovar_db.resolve()),
        "buildver": str(buildver),
        "dbnsfp": dbnsfp,
        "gtf": str(Path(str(gtf)).resolve()) if gtf else None,
        "gene_mapping": Q.GENE_MAPPING,
        "chunk_bytes": chunk_bytes,
        "sources": sources,
        "regulatory_absent_reason": (
            None if "genome" in sources
            else f"regulatory reference not built: needs gnomad41_genome ({genome_table} is not there; "
                 f"{HUMAN_SCRIPT} setup genome annovar-table {genome} gnomad41_genome puts it there)"
        ),
        "chunks": chunks,
        "rebuild": bool(rebuild),
        # A partial belongs to the plan that cut its chunk; one left by an
        # earlier plan (other chunk sizes) is never summed into this one.
        # The same signature keeps the earlier plan_id, so its finished chunks count.
        "plan_id": (str(earlier["plan_id"]) if reused
                    else f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"),
        "signature": signature,
        "reused_plan": reused,
    }
    return plan


def _file_stamp(path: Path) -> Dict[str, Any]:
    try:
        stat = Path(path).stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def plan_signature(*, sources: Mapping[str, Any], chunk_bytes: int, annovar_dir: Path, annovar_db: Path,
                   buildver: str, dbnsfp: str) -> Dict[str, Any]:
    """What a finished chunk's partial depends on. Equal signatures: the partials can be reused."""
    from paralome.version import __version__

    return {
        "sources": {name: {"table": info["table"], "size": info["size"], "mtime_ns": info["mtime_ns"],
                           "columns": info["columns"], "protocol": info["protocol"],
                           "operation": info["operation"]}
                    for name, info in sorted(sources.items())},
        "chunk_bytes": int(chunk_bytes),
        "thresholds": {
            "dominant_max_af": Q.DOMINANT_MAX_AF, "recessive_max_af": Q.RECESSIVE_MAX_AF,
            "prefilter_max_af": PREFILTER_MAX_AF, "revel_min": Q.REVEL_MIN,
            "alphamissense_pathogenic": Q.ALPHAMISSENSE_PATHOGENIC,
            "splice_max_distance": Q.SPLICE_MAX_DISTANCE, "regulatory_max_distance": Q.REGULATORY_MAX_DISTANCE,
        },
        "annovar": {
            "buildver": buildver, "dbnsfp": dbnsfp,
            "table_annovar": _file_stamp(Path(annovar_dir) / "table_annovar.pl"),
            # How a gene field's names become Ensembl ids (partials keyed the old way are not reused).
            "gene_mapping": Q.GENE_MAPPING,
            "ensGene": _file_stamp(Path(annovar_db) / f"{buildver}_ensGene.txt"),
            "dbnsfp_table": _file_stamp(Path(annovar_db) / f"{buildver}_{dbnsfp}.txt"),
        },
        "partial_schema": PARTIAL_SCHEMA,
        # The qualifying rules live in code; another Paralome version may judge an allele differently.
        "paralome_version": __version__,
    }


def _earlier_plan(path: Path) -> Dict[str, Any]:
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return plan if isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA else {}


def missing_chunks(plan: Mapping[str, Any]) -> List[int]:
    """Chunk indexes with no finished partial from this plan: what a run still has to compute."""
    return [int(c["index"]) for c in plan["chunks"] if not _partial_done(plan, int(c["index"]))]


def array_spec(indexes: Sequence[int], max_parallel: Optional[int] = None) -> str:
    """SLURM --array for these task ids: ranges joined by commas ('1-3,7,9'), '%N' only when asked."""
    ids = sorted(set(int(i) for i in indexes))
    parts: List[str] = []
    start = previous = None
    for i in ids:
        if start is None:
            start = previous = i
        elif i == previous + 1:
            previous = i
        else:
            parts.append(f"{start}-{previous}" if previous != start else str(start))
            start = previous = i
    if start is not None:
        parts.append(f"{start}-{previous}" if previous != start else str(start))
    spec = ",".join(parts)
    if max_parallel is not None and int(max_parallel) > 0:
        spec += f"%{int(max_parallel)}"
    return spec


def write_plan(plan: Mapping[str, Any]) -> Path:
    from paralome.checkpoints import atomic_write_text

    path = Path(plan["work_dir"]) / "plan.json"
    atomic_write_text(path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def read_plan(path: Path) -> Dict[str, Any]:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise BurdenReferenceError(106, f"{path} is not a burden reference plan")
    return plan


def partial_path(plan: Mapping[str, Any], index: int) -> Path:
    return Path(plan["work_dir"]) / "partials" / f"chunk_{int(index):04d}.json"


# ------------------------------------------------------------ one chunk

def annovar_command(plan: Mapping[str, Any], source: str, avinput: Path, out_prefix: Path) -> List[str]:
    info = plan["sources"][source]
    return [
        "perl", str(Path(plan["annovar_dir"]) / "table_annovar.pl"),
        str(avinput), str(plan["annovar_db"]),
        "-buildver", str(plan["buildver"]),
        "-out", str(out_prefix),
        "-remove",
        "-protocol", info["protocol"],
        "-operation", info["operation"],
        "-nastring", ".",
        "-otherinfo",
    ]


def multianno_path(out_prefix: Path, buildver: str) -> Path:
    return Path(f"{out_prefix}.{buildver}_multianno.txt")


def write_chunk_avinput(plan: Mapping[str, Any], chunk: Mapping[str, Any], avinput: Path) -> Dict[str, int]:
    """Rows of the chunk that could qualify, as avinput + the AFs as extra columns."""
    source = chunk["source"]
    info = plan["sources"][source]
    columns = info["columns"]
    af_names = required_af_columns(source)
    grpmax_index = columns[Q.af_key(source, "grpmax")]
    order = [columns[name] for name in TABLE_COLUMNS] + [columns[name] for name in af_names]
    read = kept = 0
    avinput.parent.mkdir(parents=True, exist_ok=True)
    with open(avinput, "w", encoding="utf-8") as out:
        for row in iter_rows_in_range(Path(info["table"]), chunk["start"], chunk["end"]):
            read += 1
            grpmax = Q.parse_float(row[grpmax_index]) if grpmax_index < len(row) else None
            if grpmax is not None and grpmax >= PREFILTER_MAX_AF:
                continue
            values = [row[i] if i < len(row) else "." for i in order]
            out.write("\t".join(values) + "\n")
            kept += 1
    return {"rows_read": read, "rows_kept": kept}


def iter_multianno(path: Path, otherinfo_names: Sequence[str]) -> Iterator[Dict[str, str]]:
    """Rows of a table_annovar ``*_multianno.txt`` as {column: value}.

    With -otherinfo the extra avinput columns follow the annotation columns.
    Current ANNOVAR names them Otherinfo1..N; older releases print one
    'Otherinfo' header over all of them. Either way they start at the first
    header column named Otherinfo*, and are mapped back to ``otherinfo_names``
    by position.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        start = next((i for i, name in enumerate(header) if name.startswith("Otherinfo")), len(header))
        names = header[:start]
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            row = dict(zip(names, fields[:start]))
            row.update(zip(otherinfo_names, fields[start:]))
            yield row


def aggregate_rows(rows: Iterable[Mapping[str, str]], source: str,
                   symbol_ids: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Partial sums: {"gene|tier|model": {"n": int, "q": {pop: float}}}.

    ``symbol_ids``: gene name -> Ensembl id (qualify.gene_symbol_ids). A gene name
    that is neither an id nor in the map is counted in ``unmapped_names`` (alleles per name).
    """
    tiers = SOURCE_TIERS[source]
    pops = Q.POPULATIONS_AND_ALL
    sums: Dict[str, Dict[str, Any]] = {}
    unmapped: Dict[str, int] = {}
    alleles = 0
    for row in rows:
        alleles += 1
        call = Q.classify_allele(row, symbol_ids=symbol_ids)
        for name in Q.unmapped_gene_names(row.get(Q.KEYS["gene"]), symbol_ids):
            unmapped[name] = unmapped.get(name, 0) + 1
        freqs = Q.allele_frequencies(row, source)
        for tier in tiers:
            genes = call.genes_for(tier)
            if not genes:
                continue
            for model in Q.MODELS:
                if not Q.passes_frequency(model, freqs["grpmax"]):
                    continue
                for gene in genes:
                    cell = sums.setdefault(f"{gene}|{tier}|{model}",
                                           {"n": 0, "q": {pop: 0.0 for pop in pops},
                                            "s2": {key: 0.0 for key in S2_KEYS}})
                    cell["n"] += 1
                    for pop in pops:
                        af = freqs[pop]
                        if af is not None:
                            cell["q"][pop] += af
                    for key, (a, b) in _S2_PAIRS.items():
                        fa, fb = freqs[a], freqs[b]
                        if fa is not None and fb is not None:
                            cell["s2"][key] += fa * fb
    return {"alleles_annotated": alleles, "sums": sums, "unmapped_names": unmapped}


#: s2 key -> the two populations whose AFs multiply ('nfe' -> nfe x nfe, 'afr_nfe' -> afr x nfe).
_S2_PAIRS = {key: ((key, key) if "_" not in key else tuple(key.split("_", 1))) for key in S2_KEYS}


Runner = Callable[..., Any]

SYMBOL_MAP_NAME = "gene_symbols.json"


def _symbol_map_file(plan: Mapping[str, Any]) -> Path:
    return Path(plan["work_dir"]) / SYMBOL_MAP_NAME


def plan_symbol_ids(plan: Mapping[str, Any]) -> Dict[str, str]:
    """The plan's gene name -> Ensembl id map, worked out once and kept in its work folder.

    From the genome's GTF and ANNOVAR's ``<buildver>_ensGene.txt`` (qualify.gene_symbol_ids). The
    first chunk task to need it writes it (atomically; two writing at once write the same thing).
    """
    from paralome.checkpoints import atomic_write_text

    path = _symbol_map_file(plan)
    try:
        cached = json.loads(path.read_text())
        if cached.get("plan_id") == plan.get("plan_id"):
            return dict(cached["map"])
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    mapping, stats = Q.gene_symbol_ids(plan.get("gtf"), Q.ensgene_table(plan.get("annovar_db"), plan.get("buildver")))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps({"plan_id": plan.get("plan_id"), "stats": stats, "map": mapping},
                                       sort_keys=True) + "\n")
    return mapping


def plan_symbol_ids_stats(plan: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return dict(json.loads(_symbol_map_file(plan).read_text()).get("stats") or {})
    except (OSError, ValueError, AttributeError):
        plan_symbol_ids(plan)
        try:
            return dict(json.loads(_symbol_map_file(plan).read_text()).get("stats") or {})
        except (OSError, ValueError, AttributeError):
            return {}


def chunk_scratch_need(chunk: Mapping[str, Any]) -> int:
    """What a chunk task writes to node scratch: its byte range of the gnomAD table, four times over."""
    return (int(chunk["end"]) - int(chunk["start"])) * 4


def run_chunk(plan: Mapping[str, Any], index: int, *, runner: Runner = subprocess.run) -> Path:
    """Filter, annotate and reduce one chunk; write its partial atomically.

    The chunk's own avinput and ANNOVAR output are removed once the partial is
    written: they are this task's scratch, not results, and across a gnomAD
    table they add up to the table's size again.
    """

    from paralome.scratch import job_scratch_dir

    chunk = next(c for c in plan["chunks"] if int(c["index"]) == int(index))
    shared = Path(plan["work_dir"]) / "chunks" / f"chunk_{int(index):04d}"
    if shared.exists():
        shutil.rmtree(shared)  # this task's own scratch from an earlier attempt
    # The chunk's avinput and ANNOVAR output (together the size of the gnomAD table) on this node's
    # scratch when the site has one; removed with it.
    # The chunk's avinput is about its byte range; ANNOVAR's output adds the annotation columns (~3x).
    need = chunk_scratch_need(chunk)
    with job_scratch_dir(shared, f"burden-chunk-{int(index):04d}", need_bytes=need) as scratch:
        return _run_chunk_in(plan, chunk, index, scratch, runner)


def _run_chunk_in(plan: Mapping[str, Any], chunk: Mapping[str, Any], index: int, scratch: Path,
                  runner: Runner) -> Path:
    from paralome.checkpoints import atomic_write_text

    source = chunk["source"]
    scratch.mkdir(parents=True, exist_ok=True)
    avinput = scratch / "input.avinput"
    counts = write_chunk_avinput(plan, chunk, avinput)
    result: Dict[str, Any] = {"alleles_annotated": 0, "sums": {}}
    if counts["rows_kept"]:
        prefix = scratch / "annovar"
        cmd = annovar_command(plan, source, avinput, prefix)
        log = Path(plan["work_dir"]) / "logs" / f"annovar_chunk_{int(index):04d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as handle:
            completed = runner(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        if getattr(completed, "returncode", 0) != 0:
            raise BurdenReferenceError(
                304, f"table_annovar.pl failed on chunk {index} (exit {completed.returncode}); see {log}",
            )
        output = multianno_path(prefix, plan["buildver"])
        if not output.is_file():
            raise BurdenReferenceError(
                304, f"table_annovar.pl wrote no {output.name} for chunk {index}; see {log}",
            )
        result = aggregate_rows(iter_multianno(output, required_af_columns(source)), source,
                                symbol_ids=plan_symbol_ids(plan))
    payload = {
        "schema": PARTIAL_SCHEMA,
        "plan_id": plan.get("plan_id"),
        "index": int(index),
        "source": source,
        "start": chunk["start"],
        "end": chunk["end"],
        **counts,
        **result,
        "complete": True,
    }
    target = partial_path(plan, index)
    atomic_write_text(target, json.dumps(payload, sort_keys=True) + "\n")
    shutil.rmtree(scratch)
    return target


# ---------------------------------------------------------------- finalize

def _partial_done(plan: Mapping[str, Any], index: int) -> bool:
    path = partial_path(plan, index)
    if not path.is_file():
        return False
    try:
        partial = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (bool(partial.get("complete")) and partial.get("plan_id") == plan.get("plan_id")
            and partial.get("schema") == PARTIAL_SCHEMA)


def read_gtf_genes(gtf: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """{ENSG: {gene_name, chrom, start, end}} from the GTF's gene lines."""
    genes: Dict[str, Dict[str, Any]] = {}
    if not gtf or not Path(gtf).is_file():
        return genes
    opener = gzip.open if str(gtf).endswith(".gz") else open
    with opener(gtf, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = {}
            for item in parts[8].split(";"):
                item = item.strip()
                if not item or " " not in item:
                    continue
                key, value = item.split(" ", 1)
                attrs[key] = value.strip().strip('"')
            ids = Q.ensembl_gene_ids(attrs.get("gene_id", ""))
            if not ids:
                continue
            genes[ids[0]] = {
                "gene_name": attrs.get("gene_name", ""),
                "chrom": parts[0],
                "start": int(parts[3]),
                "end": int(parts[4]),
            }
    return genes


def _fmt(value: float) -> str:
    return format(value, ".10g")


def _write_table(path: Path, rows: List[List[str]]) -> None:
    from paralome.checkpoints import temporary_beside

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = temporary_beside(path)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                text = io.TextIOWrapper(zipped, encoding="utf-8", newline="\n")
                text.write("\t".join(OUTPUT_COLUMNS) + "\n")
                for row in rows:
                    text.write("\t".join(row) + "\n")
                text.flush()
                text.detach()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize(plan: Mapping[str, Any], *, version: Optional[str] = None) -> Dict[str, Any]:
    """Add the partials, write each source's table and the provenance json.

    Refuses when any chunk's partial is missing: a reference summed over part
    of gnomAD would understate every expectation without saying so.
    """
    from paralome.checkpoints import atomic_write_text

    if version is None:
        from paralome.version import __version__ as version

    missing = [c["index"] for c in plan["chunks"] if not _partial_done(plan, c["index"])]
    if missing:
        raise BurdenReferenceError(
            304,
            f"{len(missing)} of {len(plan['chunks'])} chunk(s) have no result "
            f"(chunk {', '.join(str(i) for i in missing[:20])}{' ...' if len(missing) > 20 else ''}); "
            f"logs in {Path(plan['work_dir']) / 'logs'}",
            fix="Run the same command again; the chunks already finished are kept and only these are computed",
        )
    genes = read_gtf_genes(plan.get("gtf"))
    unmapped_names: Dict[str, int] = {}
    per_source: Dict[str, Dict[str, Any]] = {}
    for source in plan["sources"]:
        per_source[source] = {"sums": {}, "rows_read": 0, "rows_kept": 0, "alleles_annotated": 0}
    for chunk in plan["chunks"]:
        partial = json.loads(partial_path(plan, chunk["index"]).read_text())
        target = per_source[chunk["source"]]
        for key in ("rows_read", "rows_kept", "alleles_annotated"):
            target[key] += int(partial.get(key, 0))
        for name, count in (partial.get("unmapped_names") or {}).items():
            unmapped_names[name] = unmapped_names.get(name, 0) + int(count)
        for key, cell in partial["sums"].items():
            into = target["sums"].setdefault(key, {"n": 0, "q": {p: 0.0 for p in Q.POPULATIONS_AND_ALL},
                                                   "s2": {k: 0.0 for k in S2_KEYS}})
            into["n"] += int(cell["n"])
            for pop, value in cell["q"].items():
                into["q"][pop] = into["q"].get(pop, 0.0) + float(value)
            for k, value in (cell.get("s2") or {}).items():
                into["s2"][k] = into["s2"].get(k, 0.0) + float(value)

    outputs: Dict[str, Any] = {}
    for source, info in plan["sources"].items():
        rows = []
        for key in sorted(per_source[source]["sums"]):
            gene, tier, model = key.split("|")
            cell = per_source[source]["sums"][key]
            place = genes.get(gene)
            if place is None:
                place = {"gene_name": "", "chrom": "", "start": "", "end": ""}
            rows.append([
                gene, place["gene_name"], str(place["chrom"]), str(place["start"]),
                str(place["end"]), tier, model, str(cell["n"]),
                *(_fmt(cell["q"][pop]) for pop in Q.POPULATIONS_AND_ALL),
                *(_fmt(cell["s2"][key]) for key in S2_KEYS),
            ])
        output = Path(info["output"])
        _write_table(output, rows)
        outputs[source] = {
            "output": str(output),
            "tiers": list(SOURCE_TIERS[source]),
            "rows": len(rows),
            "genes": len({r[0] for r in rows}),
            "genes_without_gtf_coordinates": len({r[0] for r in rows if not r[2]}),
            "table": info["table"],
            "table_size": info["size"],
            "table_mtime": info["mtime"],
            "protocol": info["protocol"],
            "operation": info["operation"],
            "rows_read": per_source[source]["rows_read"],
            "rows_kept": per_source[source]["rows_kept"],
            "alleles_annotated": per_source[source]["alleles_annotated"],
        }

    provenance = {
        "schema": SCHEMA,
        "genome": plan["genome"],
        "buildver": plan["buildver"],
        "built": datetime.now().isoformat(timespec="seconds"),
        "paralome_version": version,
        "burden_reference": outputs["exome"]["output"],
        "sources": outputs,
        "regulatory": (
            {"status": "built", "output": outputs["genome"]["output"]}
            if "genome" in outputs
            else {"status": "absent", "reason": plan.get("regulatory_absent_reason")}
        ),
        "annovar": {
            "annovar_dir": plan["annovar_dir"],
            "annovar_db": plan["annovar_db"],
            "dbnsfp": plan["dbnsfp"],
            "command": "table_annovar.pl <chunk.avinput> <annovar_db> -buildver "
                       f"{plan['buildver']} -out <x> -remove -protocol <protocol> "
                       "-operation <operation> -nastring . -otherinfo",
        },
        "gene_coordinates": {
            "source": "gtf gene lines" if plan.get("gtf") else "none (no GTF recorded)",
            "gtf": plan.get("gtf"),
        },
        "gene_key": "Ensembl gene id from Gene.ensGene, version removed; a gene named by symbol mapped to its id",
        # Gene_burden refuses a reference without this (one built without mapping gene symbols).
        "gene_mapping": {
            "method": Q.GENE_MAPPING,
            "map": plan_symbol_ids_stats(plan),
            "unmapped_gene_names": len(unmapped_names),
            "alleles_with_unmapped_names": sum(unmapped_names.values()),
            "most_frequent_unmapped": dict(sorted(unmapped_names.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
        },
        "thresholds": {
            "dominant_max_af_grpmax": Q.DOMINANT_MAX_AF,
            "recessive_max_af_grpmax": Q.RECESSIVE_MAX_AF,
            "prefilter_max_af_grpmax": PREFILTER_MAX_AF,
            "revel_min": Q.REVEL_MIN,
            "alphamissense_pathogenic": Q.ALPHAMISSENSE_PATHOGENIC,
            "splice_max_distance": Q.SPLICE_MAX_DISTANCE,
            "regulatory_max_distance": Q.REGULATORY_MAX_DISTANCE,
        },
        "tiers": list(Q.TIERS),
        "models": list(Q.MODELS),
        "populations": list(Q.POPULATIONS_AND_ALL),
        "chunks": len(plan["chunks"]),
        "plan": str(Path(plan["work_dir"]) / "plan.json"),
        "plan_id": plan.get("plan_id"),
    }
    atomic_write_text(Path(plan["provenance"]), json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


# --------------------------------------------------------------- execution

def run_local(plan: Mapping[str, Any], *, threads: int = 1, runner: Runner = subprocess.run,
              echo: Callable[[str], None] = print) -> Dict[str, Any]:
    """Every chunk here (``threads`` at a time), then finalize."""
    indexes = missing_chunks(plan)
    kept = len(plan["chunks"]) - len(indexes)
    echo(f"Building the burden reference for {plan['genome']}: {len(indexes)} chunk(s) to compute, "
         f"{max(1, threads)} at a time" + (f"; {kept} finished earlier with this plan, kept" if kept else ""))
    failed: List[Tuple[int, BurdenReferenceError]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        futures = [(index, pool.submit(run_chunk, plan, index, runner=runner)) for index in indexes]
        for index, future in futures:
            try:
                echo(f"  done {Path(future.result()).name}")
            except BurdenReferenceError as exc:
                # The other chunks go on; a re-run computes only what is still missing
                echo(f"  chunk {index} failed: {exc.detail}")
                failed.append((index, exc))
    if failed:
        first = failed[0][1]
        raise BurdenReferenceError(
            first.code,
            f"{len(failed)} of {len(plan['chunks'])} chunk(s) failed (chunk "
            f"{', '.join(str(i) for i, _ in failed[:20])}{' ...' if len(failed) > 20 else ''}); first: {first.detail}",
            fix="Run the same command again; the chunks that finished are kept and only these are computed",
        )
    return finalize(plan)


def write_slurm_scripts(
    plan: Mapping[str, Any], plan_path: Path, *, partition: Optional[str] = None,
    max_parallel: Optional[int] = None, mem: str = DEFAULT_MEM, worker: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[Path]]:
    """An array over the chunks still to compute, and a finalize job, with absolute paths only.

    The array lists only the chunks without a finished partial of
    this plan (None when there are none: only finalize is left), and has no
    concurrency limit unless ``max_parallel`` is given -- SLURM's own limits
    and the partition decide how many run at once.
    """
    from paralome.binaries import shell_join, worker_argv
    from paralome.pipeline.naming import slurm_job_name
    from paralome.pipeline.slurm import SlurmOptions, tmp_request, write_sbatch

    work = Path(plan["work_dir"])
    scripts = work / "scripts"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker_cmd = shell_join(list(worker) if worker is not None else worker_argv())
    plan_q = shlex.quote(str(Path(plan_path).resolve()))
    todo = missing_chunks(plan)
    chunk_script: Optional[Path] = None
    if todo:
        chunk_script = write_sbatch(
            scripts / "burden_reference_array.sbatch",
            SlurmOptions(
                job_name=slurm_job_name(work, "burden_reference"), mem=mem,
                partition=partition, array=array_spec(todo, max_parallel),
                # The largest chunk's scratch, asked of SLURM (see paralome.scratch).
                tmp=tmp_request(max((chunk_scratch_need(c) for c in plan["chunks"]
                                     if int(c["index"]) in {int(i) for i in todo}), default=0), partition),
                extra_directives=(f"#SBATCH --nice={ARRAY_NICE}",),
                output=logs / "burden_reference_%A_%a.out",
                error=logs / "burden_reference_%A_%a.err",
            ),
            f"{worker_cmd} genome burden-chunk --plan {plan_q} --chunk \"$SLURM_ARRAY_TASK_ID\"\n",
        )
    final_script = write_sbatch(
        scripts / "burden_reference_finalize.sbatch",
        SlurmOptions(
            job_name=slurm_job_name(work, "burden_reference_finalize"), mem="8G",
            partition=partition,
            output=logs / "burden_reference_finalize_%j.out",
            error=logs / "burden_reference_finalize_%j.err",
        ),
        f"{worker_cmd} genome burden-finalize --plan {plan_q}\n",
    )
    return {"array": chunk_script, "finalize": final_script}


def submit_slurm(scripts: Mapping[str, Path], *, echo: Callable[[str], None] = print,
                 submit: Callable[..., str] = None) -> Dict[str, str]:
    """sbatch the array, then the finalize job after it; print each id as it returns."""
    from paralome.pipeline import slurm

    submit = submit or slurm.sbatch
    if scripts.get("array") is None:
        # Every chunk finished earlier with this plan; only the sum is left.
        final_id = submit(scripts["finalize"])
        echo(f"Every chunk was finished earlier; submitted burden reference finalize: job {final_id}")
        return {"array": None, "finalize": final_id}
    array_id = submit(scripts["array"])
    echo(f"Submitted burden reference chunks: job {array_id}")
    final_id = submit(scripts["finalize"], dependency=slurm.afterok(array_id))
    echo(f"Submitted burden reference finalize: job {final_id} (after {array_id})")
    return {"array": array_id, "finalize": final_id}
