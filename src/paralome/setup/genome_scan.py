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

"""Read a genome folder and work out what is in it.

snpEff, ANNOVAR and the rest live in the genome's folder: all are genome
version specific and all need to be matched by the user. The USER decides
which snpEff database pairs with which GTF release; putting them in one
directory makes that decision physical instead of remembered, and this
module's job is only to read back what they decided.

THIS IS NOT A SEARCH OF THE FILESYSTEM. Paralome cannot know where to look for
references, so it never guesses a root. A folder the user names on the command line is not a
guess. Nothing here is written without being shown first.

NOTHING IS MATCHED BY NAME IF A SIGNATURE WILL DO, because real layouts defeat
name matching twice over:

    Human_GRCh38_81/Human_GRCh38/          <- index named for the assembly
    Macaca_mulata_10.113/Macaca_mulata_10.113/   <- index named for its PARENT

A scanner looking for a directory called "STAR" finds neither. One looking for
SA + Genome + SAindex finds both, and would find them under any name anybody
ever chooses.

THE FASTA AND GTF COME FROM STAR, not from a glob. genomeParameters.txt records
the build command, so the index itself names the two files it was made from.
That removes the ambiguity a glob cannot resolve -- a folder holding two GTF
releases has one correct answer and a glob has no way to know it -- and it
doubles as verification: a FASTA whose name disagrees with what the index was
built from is a genuine defect, and silently aligning against one assembly
while annotating with another is exactly the failure nobody catches by eye.
"""

from __future__ import annotations

import os

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paralome.setup.annovar_tables import REGION_TABLES

#: A STAR index is these three files in one directory. Present under every
#: STAR version that Paralome supports, and present under no other tool.
STAR_SIGNATURE = ("SA", "Genome", "SAindex")

#: Written by STAR at genomeGenerate, holding the full build command.
STAR_PARAMETERS = "genomeParameters.txt"

#: The script every ANNOVAR install has and nothing else does.
ANNOVAR_MARKER = "table_annovar.pl"

#: ANNOVAR database files are `<buildver>_<protocol>.txt`. The companions are
#: skipped: .idx is an index, *Mrna.fa is sequence for a protocol already
#: counted through its .txt.
_ANNOVAR_DB = re.compile(r"^(?P<build>[A-Za-z0-9]+)_(?P<protocol>.+)\.txt$")

#: ANNOVAR's operation letters. g is gene-based, f is filter-based, r is
#: region-based. Derivable from the protocol name only as a guess, which is why
#: it is shown for confirmation and never written silently.
_GENE_BASED = ("refgene", "ensgene", "knowngene", "wgencode", "genericgene")

#: ANNOVAR ships sample databases so its tutorial runs out of the box. They are not annotation sources
#: (hg19_example_db_generic, hg19_example_db_gff3) and are never proposed as protocols.
_EXAMPLE_DATABASES = ("example_db", "example_gene", "_example")

#: Assembly name to ANNOVAR/UCSC build. THE GENOME ALREADY KNOWS WHICH BUILD IT
#: IS, and reading it off the assembly is a fact where counting files is a
#: popularity contest: an ANNOVAR humandb commonly holds hg19_* and hg38_* tables
#: side by side, in equal numbers, and a GRCh38 genome annotated with hg19 tables
#: would be wrong at every coordinate while looking completely ordinary.
ASSEMBLY_TO_BUILD = {
    "grch38": "hg38", "hg38": "hg38",
    "grch37": "hg19", "hg19": "hg19",
    "grcm39": "mm39", "mm39": "mm39",
    "grcm38": "mm10", "mm10": "mm10",
    # Keys are lower case (the lookup lowers the assembly).
    "mmul_10": "rheMac10", "rhemac10": "rheMac10",
    "rnor_6.0": "rn6", "mratbn7.2": "rn7",
    "canfam3.1": "canFam3", "susscrofa11.1": "susScr11",
}


def build_for_assembly(assembly: Optional[str]) -> Optional[str]:
    """The ANNOVAR build name for an assembly, when the pair is known.

    Returns None rather than guessing for anything unlisted -- a sheep or a
    ferret has no UCSC build, and inventing one would be worse than asking.
    """
    if not assembly:
        return None
    key = str(assembly).strip().lower()
    # A RefSeq/GenBank patch suffix ("GRCh38.p14") names the same
    # assembly; it did not map, and the build fell back to a file count.
    key = re.sub(r"\.p\d+$", "", key)
    return ASSEMBLY_TO_BUILD.get(key)

#: Directories whose contents are examples and test data, not resources.
_IGNORED_DIRECTORIES = {"example", "examples", "galaxy", "scripts", "exec"}

FASTA_SUFFIXES = (".fa", ".fasta", ".fna")
GTF_SUFFIXES = (".gtf", ".gff", ".gff3")


@dataclass
class Finding:
    """One resource, where it was found, and how confident that is.

    ``source`` matters more than it looks. A FASTA named by
    genomeParameters.txt is a fact recorded by the tool that consumed it; a
    FASTA found by globbing for *.fa is a guess that happens to be the only
    candidate. Both end up in the same config key, and the person confirming
    deserves to know which they are looking at.
    """

    key: str
    value: str
    source: str
    confident: bool = True
    note: str = ""


@dataclass
class ScanResult:
    folder: Path
    name: str
    findings: Dict[str, Finding] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    def value(self, key: str) -> Optional[str]:
        found = self.findings.get(key)
        return found.value if found else None

    def as_entry(self) -> Dict[str, str]:
        return {key: found.value for key, found in self.findings.items()}


def _looks_like_star_index(directory: Path) -> bool:
    try:
        return all((directory / name).is_file() for name in STAR_SIGNATURE)
    except OSError:
        return False


def unreadable_directories(folder: Path, depth: int = 3) -> List[Path]:
    """Directories in this genome folder that this account cannot open.

    A STAR index built by another account with a private umask is invisible to
    the scan: stat() on its files fails with "permission denied", which the
    signature test cannot tell from "not there". Seen on the first rat genome:
    an index built as root, reported as "no STAR index here" and registered
    for ATAC only. Listing the unreadable folders turns that into the real
    reason.
    """
    found: List[Path] = []
    frontier = [folder]
    for _level in range(depth):
        nxt: List[Path] = []
        for directory in frontier:
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    found.append(child)
                    continue
                if not os.access(child, os.R_OK | os.X_OK):
                    found.append(child)
                else:
                    nxt.append(child)
        frontier = nxt
    return found


def find_star_index(folder: Path) -> Optional[Path]:
    """The STAR index inside this genome folder, by signature not by name.

    Searched breadth-first so a shallow index wins over a deeper one -- a
    genome folder holding an index and a stray backup of one should resolve to
    the obvious candidate rather than to whichever os.walk reached first.
    """
    candidates: List[Path] = []
    if _looks_like_star_index(folder):
        return folder
    for depth in range(1, 4):
        pattern = "/".join(["*"] * depth)
        for child in sorted(folder.glob(pattern)):
            try:
                if child.is_dir() and _looks_like_star_index(child):
                    candidates.append(child)
            except OSError:
                continue
        if candidates:
            return candidates[0]
    return None


def star_fasta_files(index: Path) -> List[str]:
    """Every FASTA basename on the index's genomeFastaFiles line."""
    try:
        text = (index / STAR_PARAMETERS).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        parts = line.split(None, 1)
        if not line.startswith("###") and len(parts) == 2 and parts[0] == "genomeFastaFiles":
            return [Path(value).name for value in parts[1].split()]
    return []


def read_star_sources(index: Path) -> Tuple[Optional[str], Optional[str]]:
    """The FASTA and GTF basenames the index was built from.

    Indexes often record them relative -- `./Macaca_mulatta.Mmul_10.113.gtf`
    -- because that is how they were typed, and STAR stores the command
    verbatim. The working directory of that build is not recorded and is long
    gone, so the DIRECTORY is unrecoverable and the BASENAME is exact. The
    basename is what this returns, and it is enough: the file is somewhere in
    the genome folder, and an exact filename beats any glob.

    An index built without --sjdbGTFfile records no annotation, which is a
    legitimate state rather than an error -- the caller falls back to a glob.
    """
    parameters = index / STAR_PARAMETERS
    fasta = gtf = None
    try:
        text = parameters.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    for line in text.splitlines():
        # Key and value are whitespace separated, and the file also carries a
        # `### STAR  --runMode ...` comment line holding the same paths. Only
        # the key/value lines are read, so the comment cannot win.
        if line.startswith("###"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0], parts[1].strip()
        if key == "genomeFastaFiles" and value:
            fasta = Path(value.split()[0]).name
        elif key == "sjdbGTFfile" and value and value != "-":
            gtf = Path(value).name
    return fasta, gtf


def _find_by_name(folder: Path, basename: str) -> Optional[Path]:
    """Somewhere in the folder, a file with exactly this name."""
    direct = folder / basename
    if direct.is_file():
        return direct
    for match in sorted(folder.rglob(basename)):
        if match.is_file():
            return match
    return None


def _first_with_suffix(folder: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    found: List[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _IGNORED_DIRECTORIES for part in path.parts):
            continue
        name = path.name.lower()
        if name.endswith(".gz"):
            name = name[:-3]
        if any(name.endswith(suffix) for suffix in suffixes):
            found.append(path)
    return found


def find_annovar(folder: Path, assembly: Optional[str] = None) -> Dict[str, Finding]:
    """ANNOVAR's install, database, build version and protocol list.

    The build version is the awkward one. An ANNOVAR humandb accumulates
    databases and never discards them (hg19_*, hg38_* and GRCh37_* side by
    side), so reading a prefix off one filename would be a coin toss. The
    build comes from the assembly (ASSEMBLY_TO_BUILD); the alternatives are
    recorded as a problem, because picking silently between hg19 and hg38 is a
    way to annotate a study against the wrong assembly.
    """
    findings: Dict[str, Finding] = {}
    marker = None
    for candidate in sorted(folder.rglob(ANNOVAR_MARKER)):
        if candidate.is_file():
            marker = candidate
            break
    if marker is None:
        return findings

    install = marker.parent
    findings["annovar_dir"] = Finding(
        "annovar_dir", str(install), f"found {ANNOVAR_MARKER}",
    )

    # The database directory is whichever holds the most `<build>_<name>.txt`
    # files. Conventionally humandb, but ANNOVAR does not require that name and
    # a non-human install will not use it.
    by_directory: Dict[Path, List[Tuple[str, str]]] = {}
    for path in sorted(install.rglob("*.txt")):
        if not path.is_file() or path.parent.name in _IGNORED_DIRECTORIES:
            continue
        match = _ANNOVAR_DB.match(path.name)
        if not match:
            continue
        protocol = match.group("protocol")
        if protocol.endswith("Mrna"):
            continue
        # ANNOVAR's own tutorial data. Proposing these as protocols would put
        # sample databases into a real annotation run.
        if any(token in protocol.lower() for token in _EXAMPLE_DATABASES):
            continue
        by_directory.setdefault(path.parent, []).append(
            (match.group("build"), protocol)
        )
    if not by_directory:
        return findings

    # THE ASSEMBLY DECIDES, not the file count. See ASSEMBLY_TO_BUILD.
    expected = build_for_assembly(assembly)
    # The directory too, so humandb/ beside mousedb/ is not chosen by its file count for a mouse genome.
    holding = {d: sum(1 for b, _ in e if b == expected) for d, e in by_directory.items()} if expected else {}
    if holding and max(holding.values()) > 0:
        database = max(by_directory, key=lambda d: (holding[d], len(by_directory[d])))
    else:
        database = max(by_directory, key=lambda d: len(by_directory[d]))
    entries = by_directory[database]
    findings["annovar_db"] = Finding("annovar_db", str(database), "database files")

    builds: Dict[str, List[str]] = {}
    for build, protocol in entries:
        builds.setdefault(build, []).append(protocol)
    if expected and expected in builds:
        chosen = expected
        source = f"{assembly} uses {chosen}"
        confident = True
        note = (
            "" if len(builds) == 1
            else "ignored, wrong assembly: " + ", ".join(
                f"{b} ({len(p)})" for b, p in sorted(builds.items()) if b != chosen
            )
        )
    elif expected:
        # The assembly is known and its databases are NOT here. Saying so is
        # the whole value: whatever else is in the directory is for a different
        # genome, and using it would be the error this exists to prevent.
        chosen = expected
        source = (f"{assembly} needs {chosen}, and no {chosen}_* database is here "
                  "(in any directory of this ANNOVAR install)")
        confident = False
        everywhere: Dict[str, int] = {}
        for listed in by_directory.values():
            for b, _ in listed:
                everywhere[b] = everywhere.get(b, 0) + 1
        note = "present instead: " + ", ".join(
            f"{b} ({n})" for b, n in sorted(everywhere.items())
        )
    elif len(builds) == 1:
        chosen = next(iter(builds))
        source = f"{len(builds[chosen])} databases named {chosen}_*"
        confident = True
        note = ""
    else:
        # NO BUILD IS PICKED BY COUNT. With the assembly unknown (a GENCODE, UCSC or RefSeq-named FASTA) and
        # several builds present, nothing is recorded; `genome add` looks again once the assembly is
        # confirmed, and an annotation without a build refuses and says so.
        findings["annovar_buildver"] = Finding(
            "annovar_buildver", "", "the assembly is not known yet", confident=False,
            note="several builds present (" + ", ".join(
                f"{b} ({len(p)})" for b, p in sorted(builds.items())
            ) + "): give the assembly (--assembly) or --annovar-buildver",
        )
        return findings
    findings["annovar_buildver"] = Finding(
        "annovar_buildver", chosen, source, confident=confident, note=note,
    )
    if chosen not in builds:
        return findings

    # Gene-based protocols first, which is ANNOVAR's own convention -- refGeneWithVer then the filter
    # databases. Alphabetical would bury the gene annotation in the middle for no reason, and the two lists
    # have to stay index-aligned, so the order is decided once here rather than in two places.
    def _is_gene_based(protocol: str) -> bool:
        return any(token in protocol.lower() for token in _GENE_BASED)

    protocols = sorted(set(_annovar_protocols(builds[chosen])), key=lambda p: (not _is_gene_based(p), p))
    operations = ["g" if _is_gene_based(p) else "r" if p in REGION_TABLES else "f" for p in protocols]
    findings["annovar_protocol"] = Finding(
        "annovar_protocol", ",".join(protocols),
        f"every {chosen} database in that directory",
        confident=False,
        note="this is what is AVAILABLE -- drop any you do not want queried",
    )
    findings["annovar_operation"] = Finding(
        "annovar_operation", ",".join(operations), "inferred from protocol names",
        confident=False,
        note="g is gene-based, r region-based, f filter-based -- check before accepting",
    )
    return findings


#: Versioned families of which only the newest release is proposed
#: (two ClinVar releases would both write CLNSIG). Each pattern gives the family
#: and a sortable release.
_VERSIONED_FAMILIES = (
    (re.compile(r"^clinvar_(\d{8})$"), lambda m: ("clinvar", (int(m.group(1)),))),
    (re.compile(r"^dbnsfp(\d+)([a-z]?)$"), lambda m: ("dbnsfp", (int(m.group(1)), m.group(2)))),
    (re.compile(r"^avsnp(\d+)$"), lambda m: ("avsnp", (int(m.group(1)),))),
    (re.compile(r"^gnomad(\d*)_(exome|genome)$"),
     lambda m: ("gnomad_" + m.group(2), (int(m.group(1) or 0),))),
    (re.compile(r"^1000g(\d{4})([a-z]{3})_([a-z]+)$"),
     lambda m: ("1000g_" + m.group(3), (int(m.group(1)), _MONTH_NUMBER.get(m.group(2), 0)))),
)
_MONTH_NUMBER = {name: index for index, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_THOUSAND_G_FILE = re.compile(r"^([A-Z]+)\.sites\.(\d{4})_(\d{2})$")


def _annovar_protocols(found: List[str]) -> List[str]:
    """The protocols proposed for the database names found.

    ANNOVAR's companion files (<table>Version) are not protocols; 1000 Genomes
    files (<POP>.sites.2015_08) are queried as 1000g2015aug_<pop>; of a
    versioned family (clinvar_*, dbnsfp*, avsnp*, gnomad*_exome/_genome,
    1000g*_<pop>) only the newest release is kept.
    """
    months = {number: name for name, number in _MONTH_NUMBER.items()}
    mapped: List[str] = []
    for protocol in found:
        if protocol.endswith("Version"):
            continue
        thousand = _THOUSAND_G_FILE.match(protocol)
        if thousand:
            pop, year, month = thousand.groups()
            if int(month) not in months:
                continue
            protocol = f"1000g{year}{months[int(month)]}_{pop.lower()}"
        mapped.append(protocol)
    newest: Dict[str, Tuple[Any, str]] = {}
    kept: List[str] = []
    for protocol in mapped:
        for pattern, family_of in _VERSIONED_FAMILIES:
            match = pattern.match(protocol)
            if match:
                family, release = family_of(match)
                if family not in newest or release > newest[family][0]:
                    newest[family] = (release, protocol)
                break
        else:
            kept.append(protocol)
    return kept + [protocol for _release, protocol in newest.values()]


def find_snpeff(folder: Path, assembly: Optional[str] = None,
                problems: Optional[List[str]] = None) -> Dict[str, Finding]:
    """snpEff's jar, config, and the database name from its data directory."""
    findings: Dict[str, Finding] = {}
    for jar in sorted(folder.rglob("snpEff.jar")):
        if jar.is_file():
            findings["snpeff_jar"] = Finding("snpeff_jar", str(jar), "snpEff.jar")
            break
    for config in sorted(folder.rglob("snpEff.config")):
        if config.is_file():
            findings["snpeff_config"] = Finding(
                "snpeff_config", str(config), "snpEff.config",
            )
            break
    if "snpeff_jar" not in findings:
        return findings

    # snpEff names its databases by assembly and annotation release -- GRCh38.99
    # -- and the directory under data/ IS that name. It is not a file path, so
    # nothing else can supply it.
    data = findings["snpeff_jar"]
    # Data/genomes is where `snpEff build` keeps reference FASTAs, not a database. A built one
    # (snpEffectPredictor.bin) matching the assembly is taken only when it is the one candidate, else nothing.
    found_dirs = [
        child
        for child in sorted((Path(data.value).parent / "data").glob("*"))
        if child.is_dir() and child.name != "genomes"
    ]
    databases = [child.name for child in found_dirs]
    if len(databases) == 1:
        findings["snpeff_genome"] = Finding(
            "snpeff_genome", databases[0], "the only database under data/",
        )
    elif databases:
        narrowed = [c for c in found_dirs if (c / "snpEffectPredictor.bin").is_file()] or found_dirs
        if assembly:
            wanted = assembly.lower()
            narrowed = [c for c in narrowed
                        if c.name.lower() == wanted or c.name.lower().startswith(wanted + ".")]
        else:
            narrowed = []
        if len(narrowed) == 1:
            findings["snpeff_genome"] = Finding(
                "snpeff_genome", narrowed[0].name, f"the one of several under data/ for {assembly}",
                confident=False,
                note="also present: " + ", ".join(d for d in databases if d != narrowed[0].name),
            )
        elif problems is not None:
            problems.append(
                f"{len(databases)} snpEff databases under data/ and the scan cannot tell which is for "
                f"{assembly or 'this assembly'}: " + ", ".join(databases)
                + " -- give snpeff_genome by hand"
            )
    return findings


def _why_not_a_directory(folder: Path) -> str:
    """Which of the three ways a path fails to be a readable directory.

    A typo, a file passed where a folder was meant, and a permissions problem
    all reach the same `is_dir() is False`, and all three want a different
    response from the person reading the message. Naming the nearest parent
    that DOES exist turns "no such directory" into "you are one character
    away", which is the common case.
    """
    try:
        if folder.exists():
            return f"{folder} is a file, not a directory"
    except OSError:
        return (
            f"{folder} cannot be reached -- a directory above it is not "
            f"readable by you"
        )
    existing = folder.parent
    while not existing.is_dir() and existing != existing.parent:
        existing = existing.parent
    if existing == folder.parent:
        siblings = sorted(
            child.name for child in existing.iterdir() if child.is_dir()
        )[:8]
        listing = ("; found: " + ", ".join(siblings)) if siblings else ""
        return f"no {folder.name} in {existing}{listing}"
    return f"{folder} does not exist -- the deepest directory that does is {existing}"


def scan_genome_folder(folder: Path) -> ScanResult:
    """Everything this folder offers, with provenance, written nowhere."""
    folder = Path(folder).expanduser().resolve()
    result = ScanResult(folder=folder, name=folder.name)
    if not folder.is_dir():
        # Say WHICH of the three it is. "is not a directory" covers
        # a path that does not exist, a path that is a file, and a path whose
        # parent you cannot traverse -- and those need completely different
        # responses: a typo, a wrong argument, or a chmod by someone else.
        # Path.is_dir() returns False for all three, which is why this has to
        # be worked out rather than reported from that one call.
        result.problems.append(_why_not_a_directory(folder))
        return result

    index = find_star_index(folder)
    fasta_name = gtf_name = None
    if index is not None:
        result.findings["star_index"] = Finding(
            "star_index", str(index),
            f"has {', '.join(STAR_SIGNATURE)}",
        )
        fasta_name, gtf_name = read_star_sources(index)
        # An index built from several FASTAs (assembly + transgene)
        # matches none of them alone; the first was recorded as if it did.
        several = star_fasta_files(index)
        if len(several) > 1:
            fasta_name = None
            result.problems.append(
                f"the STAR index was built from {len(several)} FASTA files ({', '.join(several)}); "
                "no single one of them is this genome's sequence -- concatenate them in that order "
                "and give the combined file as reference_fasta"
            )
    else:
        blocked = unreadable_directories(folder)
        if blocked:
            names = ", ".join(str(path) for path in blocked[:5])
            result.problems.append(
                f"no STAR index could be checked: this account cannot read {names} "
                f"(permission denied), so an index there is invisible. Make it readable by "
                f"the genome group, e.g. `chgrp -R <group> <folder>; chmod -R g+rX <folder>`, "
                f"then run `paralome setup genome add` again"
            )
        else:
            result.problems.append(
                "no STAR index here -- looked for a directory holding "
                + ", ".join(STAR_SIGNATURE)
            )

    for key, recorded, suffixes, label in (
        ("reference_fasta", fasta_name, FASTA_SUFFIXES, "FASTA"),
        ("gtf", gtf_name, GTF_SUFFIXES, "GTF"),
    ):
        if key == "reference_fasta" and index is not None and len(star_fasta_files(index)) > 1:
            continue
        if recorded:
            match = _find_by_name(folder, recorded)
            if match is not None:
                result.findings[key] = Finding(
                    key, str(match), f"named by the STAR index ({recorded})",
                )
                continue
            result.problems.append(
                f"the STAR index was built from {recorded}, which is not in "
                f"this folder"
            )
        candidates = _first_with_suffix(folder, suffixes)
        if len(candidates) == 1:
            result.findings[key] = Finding(
                key, str(candidates[0]), f"the only {label} here",
                confident=recorded is None,
                note="" if recorded is None else "does NOT match the STAR index",
            )
        elif len(candidates) > 1:
            result.problems.append(
                f"{len(candidates)} {label} files here and the STAR index does "
                f"not say which: " + ", ".join(c.name for c in candidates)
            )

    result.findings.update(
        find_annovar(folder, identity_from_fasta(
            result.value('reference_fasta')).get('assembly'))
    )
    result.findings.update(find_snpeff(
        folder, identity_from_fasta(result.value("reference_fasta")).get("assembly"), result.problems))
    result.findings.update(find_burden_reference(folder, result.value("annovar_buildver")))

    result.findings.update(find_motifs(folder))
    result.findings.update(find_uniprot_table(folder, result.problems))
    return result


#: Ensembl's UniProt cross-references, as its FTP names them.
UNIPROT_TABLE_SUFFIXES = (".uniprot.tsv", ".uniprot.tsv.gz")


def find_uniprot_table(folder: Path, problems: List[str]) -> Dict[str, Finding]:
    """Ensembl's UniProt table for this genome, in the folder itself.

    Registered with the genome, from the same Ensembl release as its GTF, so a protein named by no gene symbol
    (Olink's Abeta40, pTau217) is found by its UniProt accession. Optional: one file ending .uniprot.tsv or
    .uniprot.tsv.gz at the folder's top; two are refused, as two GTFs are."""
    try:
        found = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.name.lower().endswith(UNIPROT_TABLE_SUFFIXES))
    except OSError:
        return {}
    if len(found) > 1:
        problems.append(f"{len(found)} UniProt tables here and nothing says which: "
                        + ", ".join(p.name for p in found))
        return {}
    if not found:
        return {}
    return {"uniprot_table": Finding("uniprot_table", str(found[0]), "Ensembl's UniProt cross-references")}


def find_burden_reference(folder: Path, buildver: Optional[str]) -> Dict[str, Finding]:
    """A gnomAD burden reference already built in this folder.

    Read back like everything else here, so registering a folder again keeps
    the reference `setup genome burden-reference` built into it. Only the
    file for this folder's ANNOVAR build counts.
    """
    if not buildver:
        return {}
    from paralome.burden.reference import reference_path

    path = reference_path(folder, buildver, "exome")
    if not path.is_file():
        return {}
    return {
        "burden_reference": Finding(
            "burden_reference", str(path), "built by setup genome burden-reference",
        )
    }


#: A motif database, by any of the names it actually arrives under.  JASPAR's
#: own downloads end `_meme.txt`; a file exported by hand is usually `.meme`.
_MOTIF_SUFFIXES = (".meme", ".meme.txt", "_meme.txt")


def _is_motif_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(s) for s in _MOTIF_SUFFIXES)


def _jaspar_cache(motifs: Path) -> Optional[Path]:
    """The newest release in a JASPAR cache tree, if this is one.

    Mirrors the layout atac/motifs.py creates: motifs/jaspar/<release>/<file>.
    Releases are compared numerically so 2026 beats 2024, and a directory that
    is not a release number is skipped rather than guessed at.
    """
    root = motifs / "jaspar"
    if not root.is_dir():
        return None
    best: Optional[Tuple[int, Path]] = None
    for release in root.iterdir():
        if not release.is_dir() or not release.name.isdigit():
            continue
        files = [p for p in sorted(release.iterdir()) if _is_motif_file(p)]
        if not files:
            continue
        rank = int(release.name)
        if best is None or rank > best[0]:
            best = (rank, files[0])
    return best[1] if best else None


def find_motifs(folder: Path) -> Dict[str, Finding]:
    """A motif database in this genome folder, or shared beside it.

    The one resource that is legitimately NOT per-genome. JASPAR CORE
    vertebrates is the same file for human, macaque, mouse and sheep -- the
    collection is taxonomic, not per-assembly -- so one copy is kept for the
    whole repository, in the parent genomes folder.

    Searched inside the genome folder FIRST, so a genome that needs its own --
    a plant, an insect, a fungus, where the vertebrate set is simply wrong --
    overrides the shared one by having its own.

    ONE LEVEL UP AND NO FURTHER. The parent is the repository root, and that
    boundary is the whole reason this is safe to do: walking further would
    eventually reach a home directory or a shared storage root and start finding motif
    files belonging to unrelated work. A rule that stops at a known place is a
    convention; one that keeps climbing is a guess.
    """
    findings: Dict[str, Finding] = {}
    own = [p for p in sorted(folder.rglob("*")) if _is_motif_file(p)]
    if len(own) == 1:
        findings["motif_database"] = Finding(
            "motif_database", str(own[0]), "in this genome folder",
        )
        return findings
    if len(own) > 1:
        findings["motif_database"] = Finding(
            "motif_database", str(own[0]), "first of several in this folder",
            confident=False,
            note="also here: " + ", ".join(m.name for m in own[1:]),
        )
        return findings

    parent = folder.parent

    # THE CACHE PARALOME ITSELF WRITES. atac/motifs.py downloads to
    #
    #     <config>/motifs/jaspar/<release>/JASPAR<release>_..._meme.txt
    #
    # and that tree, copied into the repository, is a shared copy everyone can use. It is recognised as
    # a SPECIFIC SHAPE (the downloader writes `_meme.txt`, three directories deep) rather than by walking
    # deeper: the generic rule still stops at the repository root. Knowing one layout by name is a
    # convention; descending until something matches is a guess.
    cached = _jaspar_cache(parent / "motifs")
    if cached is not None:
        findings["motif_database"] = Finding(
            "motif_database", str(cached),
            f"JASPAR cache shared across {parent.name}",
        )
        return findings

    shared: List[Path] = []
    for candidate in sorted(parent.glob("*")) + sorted(parent.glob("*/*")):
        if not _is_motif_file(candidate):
            continue
        # Anything under a sibling GENOME folder belongs to that genome, not to
        # the repository, so it is not shared and must not be borrowed.
        if folder.name not in candidate.parts:
            if candidate.parent == parent or candidate.parent.parent == parent:
                # BOTH SIDES LOWERCASED.  This compared a
                # lowercased directory name against an un-lowercased
                # parent.name, so a repository whose name contains a capital
                # never matched itself (e.g. "Shared_genomes").  A motif file
                # sitting directly in such a repository root was therefore
                # invisible.
                if candidate.parent.name.lower() in ("motifs", parent.name.lower()):
                    shared.append(candidate)
    if len(shared) == 1:
        findings["motif_database"] = Finding(
            "motif_database", str(shared[0]),
            f"shared across {parent.name}",
        )
    elif len(shared) > 1:
        findings["motif_database"] = Finding(
            "motif_database", str(shared[0]),
            f"first of several shared in {parent.name}",
            confident=False,
            note="also there: " + ", ".join(m.name for m in shared[1:]),
        )
    return findings


#: Ensembl names every reference the same way: Genus_species.Assembly.rest.
#: `Macaca_mulatta.Mmul_10.dna.toplevel.fa` -> Macaca mulatta / Mmul_10, and
#: `Homo_sapiens.GRCh38.dna.toplevel.fa` -> Homo sapiens / GRCh38.
_ENSEMBL_FASTA = re.compile(
    r"^(?P<genus>[A-Z][a-z]+)_(?P<species>[a-z_]+?)\.(?P<assembly>[^.]+(?:\.[0-9]+)?)\.dna"
)


#: GCF_016772045.2_ARS-UI_Ramb_v3.0_genomic.fna -> ARS-UI_Ramb_v3.0
_REFSEQ_FASTA = re.compile(r"^GC[AF]_\d+\.\d+_(?P<assembly>.+?)_genomic\b")


def identity_from_fasta(fasta: Optional[str]) -> Dict[str, str]:
    """Species and assembly, read off an Ensembl reference filename.

    Ensembl's convention is Genus_species.Assembly.dna.*, so a folder scan
    fills the identity fields as well as the resource keys. Nothing is
    invented: a filename that does not match the convention yields nothing,
    and the walkthrough asks.
    """
    if not fasta:
        return {}
    name = Path(fasta).name
    match = _ENSEMBL_FASTA.match(name)
    if match:
        species = f"{match.group('genus')} {match.group('species').replace('_', ' ')}"
        return {"species": species, "assembly": match.group("assembly")}

    # RefSeq, which is what a newly sequenced species arrives as --
    # GCF_016772045.2_ARS-UI_Ramb_v3.0_genomic.fna. The assembly name is in
    # there; the species is not, at all. Returning the half that IS present
    # beats returning nothing, because the walkthrough then asks for one field
    # instead of two, and the one it asks for is the one no filename holds.
    refseq = _REFSEQ_FASTA.match(name)
    if refseq:
        return {"assembly": refseq.group("assembly")}
    return {}


#: Annotation sources, as recorded. UNKNOWN is a real answer, not a failure:
#: see annotation_identity.
ENSEMBL = "ensembl"
GENCODE = "gencode"
REFSEQ = "refseq"
CUSTOM = "custom"
UNKNOWN = "unknown"
ANNOTATION_SOURCES = (ENSEMBL, GENCODE, REFSEQ, CUSTOM, UNKNOWN)

#: Genus_species. -- the leading token of every Ensembl filename.
_ENSEMBL_NAME = re.compile(r"^[A-Z][a-z]+_[a-z_]+\.")
#: gencode.v44.annotation.gtf, gencode.vM33.annotation.gtf (mouse carries an M).
_GENCODE_NAME = re.compile(r"^gencode\.(?P<release>v[A-Z]?\d+)\.", re.IGNORECASE)
#: GCF_016772045.2_ARS-UI_Ramb_v3.0_genomic.gtf
_REFSEQ_NAME = re.compile(r"^GC[AF]_\d+\.\d+_")

#: Suffixes to strip before looking for the release component.
_ANNOTATION_SUFFIXES = (".gz", ".gtf", ".gff3", ".gff")


def _stem(name: str) -> str:
    """The filename with its annotation suffixes removed, innermost last."""
    lowered = name
    changed = True
    while changed:
        changed = False
        for suffix in _ANNOTATION_SUFFIXES:
            if lowered.lower().endswith(suffix):
                lowered = lowered[: -len(suffix)]
                changed = True
    return lowered


def annotation_identity(gtf: Optional[str]) -> Dict[str, str]:
    """Which annotation project this GTF came from, and which release of it.

    The release is the last purely numeric component of the name, so a
    legitimate Ensembl file such as
    `Homo_sapiens.GRCh38.113.chr_patch_hapl_scaff.gtf` is read too.

    THE SOURCE IS WHAT MAKES THE NUMBER MEAN ANYTHING. Ensembl 113 and GENCODE
    v44 are both current and neither is "newer"; ordering them by integer would
    be nonsense wearing the costume of arithmetic. So the source is recorded
    alongside the release and two genomes are only ever compared within one.

    UNKNOWN IS A REAL ANSWER. A GTF whose provenance cannot be read returns
    UNKNOWN rather than a guess, and a genome recorded as UNKNOWN takes no part
    in supersession -- said out loud at registration, so that nobody mistakes
    "Paralome is not offering you anything" for "nothing newer exists". That
    distinction is the whole reason this is asked rather than inferred.
    """
    if not gtf:
        return {"annotation_source": UNKNOWN, "annotation_release": UNKNOWN}

    name = Path(gtf).name
    stem = _stem(name)

    gencode = _GENCODE_NAME.match(name)
    if gencode:
        return {
            "annotation_source": GENCODE,
            "annotation_release": gencode.group("release"),
        }

    if _REFSEQ_NAME.match(name):
        # The accession's own version (GCF_...2) is the ASSEMBLY's, not the
        # annotation's, and RefSeq annotation releases do not appear in the
        # filename at all. Claiming one here would be inventing it.
        return {"annotation_source": REFSEQ, "annotation_release": UNKNOWN}

    if _ENSEMBL_NAME.match(name):
        # The last purely numeric dot-component. Robust to an assembly that contains a dot (Sscrofa11.1)
        # and to trailing content after the release (chr_patch_hapl_scaff, abinitio).
        digits = [part for part in stem.split(".") if part.isdigit() and len(part) <= 3]
        if digits:
            return {"annotation_source": ENSEMBL, "annotation_release": digits[-1]}
        return {"annotation_source": ENSEMBL, "annotation_release": UNKNOWN}

    return {"annotation_source": UNKNOWN, "annotation_release": UNKNOWN}


def annotation_release(gtf: Optional[str]) -> Optional[str]:
    """The release alone, or None when it is not known. Kept for callers."""
    found = annotation_identity(gtf).get("annotation_release")
    return None if found in (None, UNKNOWN) else found


def analyses_for(result: ScanResult) -> List[str]:
    """Which analyses this folder is equipped for, from what it actually holds.

    A folder with ANNOVAR and snpEff is meant for the exome path; one with
    neither is not, and asking its owner about protocol lists would be the
    walkthrough inventing a requirement again.
    """
    analyses: List[str] = []
    if result.value("star_index"):
        analyses += ["RNA", "scRNA"]
    if result.value("reference_fasta"):
        analyses += ["ATAC", "snATAC"]
    if result.value("annovar_dir") and result.value("snpeff_jar"):
        analyses.append("DNA")
    return analyses
