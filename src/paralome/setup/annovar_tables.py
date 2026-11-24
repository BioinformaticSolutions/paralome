# Paralome -- a reproducible, checkpointed framework for omics analysis.
# Copyright (C) 2026 Kenneth L. Jones
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License, version 3, as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# A separate commercial licence is available; see NOTICE.

"""Add an ANNOVAR table to a registered genome.

`paralome setup genome annovar-table NAME TABLE` downloads ``TABLE`` into the
genome's ANNOVAR database with ANNOVAR's own downloader::

    perl <annovar_dir>/annotate_variation.pl -buildver <buildver> -downdb \
         -webfrom annovar <TABLE> <annovar_db>

checks that ``<buildver>_<TABLE>.txt`` and its ``.idx`` are there (a
gene-based table: its ``Mrna.fa`` instead of an index; a region table: the
``.txt`` alone), and adds
the table to the genome entry's ``annovar_protocol`` (with ``f``, or the
``--operation`` given, in ``annovar_operation``), so every later annotation
of that genome queries it. A table already in the directory is not downloaded
again; it is only added to the protocol. A table already in the protocol is
left where it is.

WHY: a genome with gnomad41_exome only gives most whole-genome variants --
intronic, intergenic, regulatory -- no population frequency at all, so the
rare-variant filter cannot tell a common non-coding variant from a rare one,
and the gene-burden regulatory tier cannot run. Every reader in Paralome takes the
gnomad41_genome frequencies when the VCF carries them (population_af.py,
segregation/table.py, burden); this puts the table in place and into the
genome's protocol without editing the configuration by hand.

A RESULT ANNOTATED BEFORE THE TABLE WAS ADDED IS NOT REDONE ON ITS OWN. The
supervisor notices a changed annotation request only on a project whose
annotation recorded its request; an older project's annotation has to be
removed by hand, and the stages after it with it (PLM-0308 names them).
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

#: ANNOVAR's downloader, beside table_annovar.pl in the install.
DOWNLOADER = "annotate_variation.pl"
#: A table name as ANNOVAR spells them: letters, digits, '_', '.', '-'.
_TABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
OPERATIONS = ("f", "g", "r")
#: ANNOVAR region databases (operation r); installed without an .idx.
REGION_TABLES = frozenset({
    "cytoBand", "genomicSuperDups", "wgRna", "tfbsConsSites", "gwasCatalog", "dgvMerged",
    "targetScanS", "evofold", "phastConsElements46way", "phastConsElements100way",
})
#: A 1000 Genomes protocol (1000g2015aug_all), and the bare download name (1000g2015aug).
_THOUSAND_G_PROTOCOL = re.compile(r"^1000g(\d{4})([a-z]{3})_([a-z]+)$")
_THOUSAND_G_DOWNLOAD = re.compile(r"^1000g\d{4}[a-z]{3}$")
_MONTHS = {name: f"{index:02d}" for index, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}


def protocol_file_name(buildver: str, protocol: str) -> str:
    """The file ANNOVAR reads for ``protocol``: ``<build>_<protocol>.txt``, except 1000g.

    ANNOVAR reads the protocol 1000g2015aug_all from
    ``<build>_ALL.sites.2015_08.txt`` (population upper-cased, month as a number).
    """
    match = _THOUSAND_G_PROTOCOL.match(str(protocol))
    if match and match.group(2) in _MONTHS:
        year, month, population = match.groups()
        return f"{buildver}_{population.upper()}.sites.{year}_{_MONTHS[month]}.txt"
    return f"{buildver}_{protocol}.txt"


def infer_operation(annovar_db: Path, buildver: str, table: str) -> str:
    """G when ANNOVAR's gene-based companion <build>_<table>Mrna.fa is there, r for a known
    region database, f otherwise."""
    if table in REGION_TABLES:
        return "r"
    mrna = Path(annovar_db) / f"{buildver}_{table}Mrna.fa"
    return "g" if mrna.is_file() and mrna.stat().st_size > 0 else "f"


class AnnovarTableError(Exception):
    """A refusal with a Paralome error number (raised as ParalomeError by the command)."""

    def __init__(self, code: int, detail: str, fix: Optional[str] = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.fix = fix


def split_list(value: Any) -> List[str]:
    """'a,b , c' -> ['a', 'b', 'c']; a list is taken as it is."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def table_files(annovar_db: Path, buildver: str, table: str) -> tuple[Path, Path]:
    """(<db>/<the file ANNOVAR reads for table>, its .idx)."""
    txt = Path(annovar_db) / protocol_file_name(buildver, table)
    return txt, Path(str(txt) + ".idx")


def _non_empty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def table_present(annovar_db: Path, buildver: str, table: str, operation: Optional[str] = None) -> bool:
    """Whether the table is in the database with what ANNOVAR needs beside it, all non-empty.

    Operation f (filter): the .txt and its index. The index is required:
    ANNOVAR without it reads the whole table for every query, which for
    gnomad41_genome makes an annotation run take days. Operation g
    (gene-based): the .txt and its <table>Mrna.fa; operation r (region): the
    .txt alone -- ANNOVAR installs neither with an index. Without an
    operation, it is inferred (infer_operation).
    """
    txt, idx = table_files(annovar_db, buildver, table)
    if operation is None:
        operation = infer_operation(annovar_db, buildver, table)
    if not _non_empty(txt):
        return False
    if operation == "g":
        return _non_empty(Path(annovar_db) / f"{buildver}_{table}Mrna.fa")
    if operation == "r":
        return True
    return _non_empty(idx)


def check_table_name(table: str) -> str:
    table = str(table).strip()
    if not _TABLE_NAME.match(table):
        raise AnnovarTableError(108, f"{table!r} is not an ANNOVAR table name",
                                fix="Give the name ANNOVAR lists it under, e.g. gnomad41_genome")
    return table


def check_not_bare_1000g(table: str) -> None:
    """Refuse 1000g2015aug (a download name, several files) as a protocol (PLM-0108)."""
    if _THOUSAND_G_DOWNLOAD.match(str(table)):
        raise AnnovarTableError(
            108, f"{table} is ANNOVAR's download name for several population files, not a protocol",
            fix=(f"Download it by hand (annotate_variation.pl -buildver <build> -downdb -webfrom annovar {table} "
                 f"<annovar_db>), then add the population protocol, e.g. {table}_all"),
        )


def genome_annovar(entry: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """(annovar_dir, annovar_db, buildver) of a genome entry, each checked; PLM-0402 if not usable."""
    missing = [key for key in ("annovar_dir", "annovar_db", "annovar_buildver") if not entry.get(key)]
    if missing:
        raise AnnovarTableError(
            402, "the genome entry has no " + ", ".join(missing),
            fix="Register ANNOVAR for this genome first (paralome setup genome add), then add tables to it",
        )
    annovar_dir = Path(str(entry["annovar_dir"]))
    annovar_db = Path(str(entry["annovar_db"]))
    if not (annovar_dir / DOWNLOADER).is_file():
        raise AnnovarTableError(402, f"{annovar_dir / DOWNLOADER} is not there",
                                fix="Check annovar_dir in the genome entry")
    if not annovar_db.is_dir():
        raise AnnovarTableError(402, f"the ANNOVAR database directory {annovar_db} is not there",
                                fix="Check annovar_db in the genome entry")
    return annovar_dir, annovar_db, str(entry["annovar_buildver"])


def with_protocol(entry: Mapping[str, Any], table: str, operation: str = "f") -> Dict[str, Any]:
    """The entry with ``table`` at the end of its protocol and ``operation`` at the end of its operations.

    Unchanged when the table is already in the protocol. The two lists are
    paired entry by entry by ANNOVAR, so lists of different lengths are
    refused rather than extended (PLM-0402): which operation belongs to which
    protocol could not be known.
    """
    if operation not in OPERATIONS:
        raise AnnovarTableError(108, f"--operation {operation!r}: expected one of {', '.join(OPERATIONS)}")
    check_not_bare_1000g(table)
    protocols = split_list(entry.get("annovar_protocol"))
    operations = split_list(entry.get("annovar_operation"))
    if len(protocols) != len(operations):
        raise AnnovarTableError(
            402,
            f"the genome entry's annovar_protocol has {len(protocols)} entries and annovar_operation "
            f"{len(operations)}; ANNOVAR pairs them one to one",
            fix="Correct the two lists in the genome entry (paralome setup doctor shows them), then run again",
        )
    if table in protocols:
        return dict(entry)
    updated = dict(entry)
    updated["annovar_protocol"] = ",".join(protocols + [table])
    updated["annovar_operation"] = ",".join(operations + [operation])
    return updated


def download_command(annovar_dir: Path, annovar_db: Path, buildver: str, table: str) -> List[str]:
    perl = shutil.which("perl") or "perl"
    return [perl, str(annovar_dir / DOWNLOADER), "-buildver", buildver, "-downdb", "-webfrom", "annovar",
            table, str(annovar_db)]


def free_bytes(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def fetch_table(
    entry: Mapping[str, Any],
    table: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
    echo: Callable[[str], None] = print,
    log: Optional[Path] = None,
    operation: Optional[str] = None,
) -> bool:
    """Download ``table`` into the entry's ANNOVAR database unless it is there. True if it was downloaded.

    The download is ANNOVAR's: it fetches the .txt.gz and .idx.gz and unpacks
    them in place. Success is the files being there afterwards, not only an
    exit code of 0 (the downloader reports some failures as warnings and exits
    0) -- PLM-0304 otherwise, naming the log.
    """
    table = check_table_name(table)
    check_not_bare_1000g(table)
    annovar_dir, annovar_db, buildver = genome_annovar(entry)
    txt, idx = table_files(annovar_db, buildver, table)
    if table_present(annovar_db, buildver, table, operation):
        echo(f"{txt.name} is already in {annovar_db}; not downloaded again")
        return False
    if _THOUSAND_G_PROTOCOL.match(table):
        # ANNOVAR downloads the 1000g populations together, under another name.
        download = table.rsplit("_", 1)[0]
        raise AnnovarTableError(
            304, f"{txt.name} (protocol {table}) is not in {annovar_db}; {table} cannot be downloaded by that name",
            fix=(f"Download {download} by hand (annotate_variation.pl -buildver {buildver} -downdb -webfrom annovar "
                 f"{download} {annovar_db}), then run this again"),
        )
    cmd = download_command(annovar_dir, annovar_db, buildver, table)
    free = free_bytes(annovar_db)
    if free is not None:
        echo(f"Free space in {annovar_db}: {free / 1e9:,.0f} GB")
    echo("Downloading: " + " ".join(shlex.quote(part) for part in cmd))
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as handle:
            completed = runner(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    else:
        completed = runner(cmd)
    where = f"; see {log}" if log is not None else ""
    if getattr(completed, "returncode", 0) != 0:
        raise AnnovarTableError(304, f"{DOWNLOADER} exited {completed.returncode} downloading {table}{where}",
                                fix="Read the log; a partial download is replaced by the next attempt")
    if not table_present(annovar_db, buildver, table, operation):
        raise AnnovarTableError(
            304, f"{DOWNLOADER} finished but {txt.name} and {idx.name} (for a gene-based table, its "
                 f"Mrna.fa) are not both in {annovar_db}{where}",
            fix=f"Check that ANNOVAR offers {table} for {buildver} (annotate_variation.pl -downdb avdblist)",
        )
    echo(f"Downloaded {txt} ({txt.stat().st_size / 1e9:,.1f} GB)")
    return True


def slurm_script(
    genome: str, table: str, operation: str, annovar_db: Path, *, partition: Optional[str] = None,
    mem: str = "8G", time: Optional[str] = None, worker: Optional[Sequence[str]] = None,
) -> Path:
    """One job that downloads the table and records it; written beside the database's other jobs."""
    from paralome.binaries import shell_join, worker_argv
    from paralome.pipeline.slurm import SlurmOptions, write_sbatch

    work = Path(annovar_db) / "paralome_jobs"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker_cmd = shell_join(list(worker) if worker is not None else worker_argv())
    options = SlurmOptions(
        job_name=f"annovar_{table}"[:64], mem=mem, partition=partition,
        output=logs / f"annovar_table_{table}_%j.out", error=logs / f"annovar_table_{table}_%j.err",
        # A whole directive line (#SBATCH --time=...).
        extra_directives=(f"#SBATCH --time={time}",) if time else (),
    )
    return write_sbatch(
        work / f"annovar_table_{table}.sbatch", options,
        f"{worker_cmd} genome annovar-table --genome {shlex.quote(genome)} --table {shlex.quote(table)} "
        f"--operation {shlex.quote(operation)}\n",
    )
