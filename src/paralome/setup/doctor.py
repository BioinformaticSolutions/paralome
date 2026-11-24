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
import platform
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from paralome.config.schema import ConfigValidationError
from paralome.config.user import config_path, initialize_config, load_config, save_config
from paralome.setup.annovar_tables import protocol_file_name
from paralome.pipeline.resource_catalog import (
    ResourceResolutionError,
    resolve_genome_resources,
    select_genome_config,
)

CORE_EXECUTABLES = (
    "fastp",
    "STAR",
    "samtools",
    "bcftools",
    "freebayes",
    "bgzip",
    "tabix",
    "java",
)

SLURM_EXECUTABLES = ("sbatch", "squeue", "sacct")

#: Tools a stage uses when present and records as skipped when not.
#: Evidome (the functional_evidence stage) is the first: recorded by `setup`
#: so it is resolved by absolute path like STAR, reported by `doctor`, never
#: counted against the core score -- an analysis is complete without it.
OPTIONAL_EXECUTABLES = ("evidome",
                        # The UCSC writers of the ATAC genome tracks; the preflight of a run that
                        # reaches genome_tracks requires them, nothing else does.
                        "bedGraphToBigWig", "bedToBigBed")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "status": self.status, "detail": self.detail}


def _which(name: str) -> Optional[str]:
    # Absolute, but NOT through symlinks. A tool installed as a link
    # can be a program that chooses what to run by the name it was called as:
    # Debian and Ubuntu's STAR (`apt install rna-star`) is /usr/bin/STAR ->
    # simd-dispatch, which picks the build for this CPU from that name, and
    # called by its own name exits 127. Recorded resolved, every alignment
    # failed on a stock Ubuntu (and WSL2) machine. The path PATH found is what
    # the shell would run.
    found = shutil.which(name)
    return os.path.abspath(found) if found else None


def _version(command: list[str], timeout: int = 5) -> Optional[str]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except Exception:
        return None
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    return text[0].strip() if text else None


def discover_executables() -> dict[str, dict[str, Any]]:
    """Discover common Paralome external executables from PATH."""
    found: dict[str, dict[str, Any]] = {}
    for name in CORE_EXECUTABLES + SLURM_EXECUTABLES + OPTIONAL_EXECUTABLES:
        path = _which(name)
        entry: dict[str, Any] = {"path": path, "found": bool(path)}
        if path:
            if name == "java":
                entry["version"] = _version([path, "-version"])
            elif name in {"samtools", "bcftools"}:
                entry["version"] = _version([path, "--version"])
            elif name == "fastp":
                entry["version"] = _version([path, "--version"])
            elif name == "freebayes":
                entry["version"] = _version([path, "--version"])
            elif name in {"STAR", "evidome"}:
                entry["version"] = _version([path, "--version"])
        found[name] = entry
    return found


def detect_backend(executables: dict[str, dict[str, Any]] | None = None) -> str:
    executables = executables or discover_executables()
    return "slurm" if executables.get("sbatch", {}).get("found") else "local"


def initialize_setup(*, force: bool = False) -> dict[str, Any]:
    """Initialize/update user config with discovered executables and backend.

    ``--force`` NEVER DESTROYS THE CONFIG: genomes and profiles are never
    touched, with or without it.

    Executables are re-detected on EVERY run, with or without the flag. The
    backend is re-detected only when the recorded one cannot run here (slurm
    without sbatch) or with --force; a backend this account chose is kept.
    What --force adds is the one thing that could not otherwise happen:
    re-apply the shipped default VALUES over ones this account has changed --
    threads, jobs, annotation, the filter profile, the RNA gates.
    """
    from paralome.config.schema import default_config

    # Never forced: creates the file if it is absent, keeps it if it is not.
    initialize_config()
    config = load_config(initialize=True)
    executables = discover_executables()
    config["executables"] = executables
    defaults = config.setdefault("defaults", {})
    if force:
        from paralome.config.user import record_as_shipped

        shipped = default_config()
        defaults.update(shipped.get("defaults", {}))
        for section in ("rna", "stage_resources"):
            if section in shipped:
                config.setdefault(section, {}).update(shipped[section])
        # Re-applied defaults are Paralome's, and follow it again.
        record_as_shipped(config, ("defaults", "rna", "stage_resources"))
    recorded = defaults.get("backend")
    usable = recorded == "local" or (recorded == "slurm" and executables.get("sbatch", {}).get("found"))
    if force or not usable:
        defaults["backend"] = detect_backend(executables)
    save_config(config)
    return config


def _check_path(name: str, value: Optional[str], *, kind: str) -> dict[str, Any]:
    """Present AND readable by whoever is asking.

    Existence needs only traversal of the parent directories, while opening
    the file needs the read bit on the file itself -- a reference FASTA at
    mode 600 in a colleague's home exists for everybody and can be opened by
    its owner alone, and the run would fail on a compute node, the most
    expensive place to learn it.

    Reported as its own status rather than folded into "missing", because the
    two need different fixes: a chmod by the file's owner, against rebuilding
    or re-registering a path that is genuinely not there.
    """
    if kind not in ("file", "directory"):
        raise ValueError(f"Unsupported path check kind: {kind}")
    if not value:
        return CheckResult(name, False, "missing", "not configured").to_dict()

    from paralome.setup.genomes import resource_state

    path = Path(value).expanduser()
    state = resource_state(value, kind)
    detail = str(path)
    if state == "unreadable":
        detail = f"{path}  (exists, but you cannot read it)"
    return CheckResult(name, state == "ok", state, detail).to_dict()


def _check_star_index_carries_annotation(
    value: Optional[str], registered_gtf: Optional[str] = None,
) -> dict[str, Any]:
    """A STAR index has to have been built with its GTF.

    STAR is always built with the GTF included; that is part of what QC of a
    genome build checks. An index generated without --sjdbGTFfile carries no
    splice junctions and cannot align a junction-spanning read properly.

    It is also the other half of the alignment rule: `align.star` withholds
    --sjdbGTFfile from an index that already has junctions (passing it makes
    STAR insert them a second time). That only holds if "built with the GTF"
    is enforced where the genome is registered, which is here.

    genomeParameters.txt is STAR's own record of the build. Its absence is
    reported separately from an index built without annotation: one is an
    index this cannot judge, the other is an index that is judged and wanting.

    AND IT IS COMPARED AGAINST THE REGISTERED GTF. If they disagree, the
    index's splice junctions came from one annotation while the union-exon
    gene model that counts fragments came from another: every stage runs,
    every file is present, and the two halves of the answer are built against
    different releases. Nothing else in the system would notice -- both paths
    are individually valid.

    Compared by basename. The directory STAR was run from is not recorded and
    is usually long gone (see genome_scan._star_index_sources on the same
    point), so the filename is the only part of the build record that can be
    matched at all.
    """
    if not value:
        return CheckResult(
            "star_index_annotation", False, "missing", "not configured",
        ).to_dict()

    parameters = Path(value).expanduser() / "genomeParameters.txt"
    try:
        text = parameters.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return CheckResult(
            "star_index_annotation", True, "unknown",
            f"{parameters} cannot be read; STAR's build record is unavailable",
        ).to_dict()

    for line in text.splitlines():
        if line.startswith("###"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == "sjdbGTFfile":
            named = parts[1].strip()
            if named and named != "-":
                built_with = Path(named).name
                registered = Path(registered_gtf).name if registered_gtf else None
                if registered and registered != built_with:
                    return CheckResult(
                        "star_index_annotation", False, "annotation_mismatch",
                        f"index was built with {built_with}, but this genome "
                        f"registers {registered}; the splice junctions and the "
                        f"gene model come from different annotations",
                    ).to_dict()
                return CheckResult(
                    "star_index_annotation", True, "ok", f"built with {built_with}",
                ).to_dict()
            break

    return CheckResult(
        "star_index_annotation", False, "no_annotation",
        "this index was generated without --sjdbGTFfile, so it holds no splice "
        "junctions; rebuild it with the GTF",
    ).to_dict()


def _check_optional_path(
    name: str, value: Optional[str], *, kind: str, fallback: str,
) -> dict[str, Any]:
    """A path that is allowed to be absent, because absence means something.

    ``_check_path`` treats an unset value as a failure, which is
    right for a reference FASTA and wrong for the two motif keys: skipping
    ``motif_database`` selects the JASPAR CORE vertebrates set, and skipping
    ``motif_species`` asks ``infer_motif_species`` to read the genome name.
    Both are the documented, correct answer for most vertebrates, so a genome
    relying on either is not defective and must not be called INVALID for it.

    What must still be checked is a value the user DID supply. Somebody who
    named a motif file meant that file; if it is gone, the fallback is not a
    silent substitute for it, it is a different answer to the same question.
    """
    if not value:
        return CheckResult(name, True, "default", fallback).to_dict()
    return _check_path(name, value, kind=kind)


def _check_burden_reference(genome: str, value: Optional[str]) -> dict[str, Any]:
    """The gnomAD burden reference -- built, not registered.

    Not built is not a fault: a DNA run skips the gene_burden stage and says
    why, and everything else on the genome works. So it reads like an unset
    motif database, with the command that builds it. A reference that WAS
    recorded and is gone is a fault, like any other recorded path.
    """
    if not value:
        from paralome.burden.reference import build_command

        return CheckResult(
            "burden_reference", True, "not built",
            f"not built -- gene_burden is skipped until `{build_command(genome)}`",
        ).to_dict()
    return _check_path("burden_reference", value, kind="file")


def _split_csv(value: Optional[str]) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def genome_doctor_report(genome: str, *, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Resolve, display, and validate one configured genome without modifying config."""
    cfg = config if config is not None else load_config(initialize=False)

    # SPLIT, so the two failures do not wear the same word. select_genome_config failing means NO SUCH
    # GENOME -- nothing was validated, so there is nothing to call invalid -- while resolve_genome_resources
    # failing means a genome that exists and whose files do not.
    #
    # `unknown` is what the CLI reads to print the list of names that DO exist, which is the only useful
    # thing to say to somebody who named one that does not.
    unknown = False
    try:
        genome_name, genome_config = select_genome_config(genome, config=cfg)
    except ResourceResolutionError as exc:
        unknown, error = True, str(exc)
    else:
        try:
            resolved = resolve_genome_resources(
                genome=genome_name, config=cfg, enforce_withdrawal=False
            )
        except ResourceResolutionError as exc:
            error = str(exc)
        else:
            error = ""
    if unknown or error:
        return {
            "requested": genome,
            "name": None if unknown else genome_name,
            "valid": False,
            "unknown": unknown,
            "error": error,
            "identity": {},
            "withdrawn": None,
            "analyses": [],
            "sections_in_scope": {},
            "resources": {},
            "checks": [],
        }

    identity = {
        "name": str(genome_config.get("name") or genome_name),
        "species": str(genome_config.get("species") or ""),
        "assembly": str(genome_config.get("assembly") or ""),
        # The annotation is versioned independently of the assembly,
        # so a report that names only the assembly describes half the genome.
        "annotation_source": str(genome_config.get("annotation_source") or ""),
        "annotation_release": str(genome_config.get("annotation_release") or ""),
        "modified": bool(genome_config.get("modified")),
        "modified_note": str(genome_config.get("modified_note") or ""),
    }
    resource_values = {
        name: resource.to_dict()
        for name, resource in sorted(resolved.resources.items())
    }
    value = resolved.value
    checks: list[dict[str, Any]] = [
        _check_path("reference_fasta", value("reference_fasta"), kind="file"),
        _check_path("gtf", value("gtf"), kind="file"),
        _check_path("star_index", value("star_index"), kind="directory"),
        _check_star_index_carries_annotation(value("star_index"), value("gtf")),
        # THE ONE PATH THAT LEAVES THE GENOME FOLDER. find_motifs searches the genome folder and then
        # <parent>/motifs, so motif_database is the only resource that legitimately points outside the folder
        # it belongs to -- which makes it the first thing any move of the repository breaks. It is checked
        # here, in the doctor's "motifs" section.
        #
        # motif_database is in PATH_RESOURCES too, so a stale value also stops a run at
        # resolve_genome_resources; a doctor exists so that the failure happens here rather than at the
        # front of a queued ATAC run.
        _check_optional_path(
            "motif_database", value("motif_database"), kind="file",
            fallback="JASPAR CORE vertebrates, fetched and cached on first use",
        ),
        CheckResult(
            "motif_species",
            True,
            "ok" if value("motif_species") else "default",
            value("motif_species") or "inferred from the genome name",
        ).to_dict(),
        _check_path("annovar_dir", value("annovar_dir"), kind="directory"),
        _check_path("annovar_db", value("annovar_db"), kind="directory"),
        _check_path(
            "annovar_table_annovar",
            str(resolved.path("annovar_dir") / "table_annovar.pl") if resolved.path("annovar_dir") else None,
            kind="file",
        ),
        _check_path("snpeff_jar", value("snpeff_jar"), kind="file"),
        _check_path("snpeff_config", value("snpeff_config"), kind="file"),
        _check_burden_reference(str(genome_name), value("burden_reference")),
    ]

    buildver = value("annovar_buildver")
    protocols = _split_csv(value("annovar_protocol"))
    operations = _split_csv(value("annovar_operation"))
    checks.append(
        CheckResult(
            "annovar_buildver",
            bool(buildver),
            "ok" if buildver else "missing",
            buildver or "not configured",
        ).to_dict()
    )
    checks.append(
        CheckResult(
            "annovar_protocol_operation_count",
            bool(protocols) and len(protocols) == len(operations),
            "ok" if bool(protocols) and len(protocols) == len(operations) else "invalid",
            f"protocols={len(protocols)} operations={len(operations)}",
        ).to_dict()
    )

    annovar_db = resolved.path("annovar_db")
    protocol_checks: list[dict[str, Any]] = []
    for index, protocol in enumerate(protocols):
        operation = operations[index] if index < len(operations) else ""
        # The file ANNOVAR reads (1000g2015aug_all -> <build>_ALL.sites.2015_08.txt).
        expected = annovar_db / protocol_file_name(buildver, protocol) if annovar_db and buildver else None
        compressed = Path(f"{expected}.gz") if expected else None
        actual = expected if expected and expected.is_file() else compressed
        ok = bool(actual and actual.is_file())
        protocol_checks.append(
            {
                "protocol": protocol,
                "operation": operation,
                "path": str(actual or expected) if (actual or expected) else "",
                "ok": ok,
                "status": "ok" if ok else "missing",
            }
        )

    snpeff_genome = value("snpeff_genome")
    checks.append(
        CheckResult(
            "snpeff_genome",
            bool(snpeff_genome),
            "ok" if snpeff_genome else "missing",
            snpeff_genome or "not configured",
        ).to_dict()
    )

    # JUDGE THE GENOME ON WHAT IT IS FOR. A genome registered for RNA, scRNA, snATAC and ATAC must not read
    # INVALID because its ANNOVAR and snpEff lines say "not configured": a verdict that says INVALID for a
    # genome that is entirely correct trains people to ignore it.
    #
    # sections_in_scope decides which sections apply, from the analyses the genome was registered for,
    # falling back to which keys it carries -- so genomes registered before that field existed are still
    # judged properly.
    from paralome.setup.genomes import DOCTOR_SECTIONS, sections_in_scope

    scope = sections_in_scope(genome_config)
    section_of = {
        key: section
        for section, (_, keys) in DOCTOR_SECTIONS.items()
        for key in keys
    }
    # annovar_table_annovar is derived from annovar_dir rather than configured,
    # so it is not in any section's key list and has to be placed by hand.
    section_of["annovar_table_annovar"] = "annovar"
    section_of["annovar_protocol_operation_count"] = "annovar"
    # Derived from star_index rather than configured, so it needs
    # placing by hand for the same reason. It belongs to "star", which means a
    # DNA genome with no STAR index is not failed for an index it never had.
    section_of["star_index_annotation"] = "star"

    for check in checks:
        section = section_of.get(str(check.get("name")), "reference")
        check["section"] = section
        check["applies"] = bool(scope.get(section, True))
        if not check["applies"] and not check.get("ok"):
            # Not a failure and not a pass: this genome was never meant to
            # carry it. Said plainly rather than left looking like an absence.
            check["status"] = "not applicable"

    applicable = [check for check in checks if check.get("applies")]
    valid = all(bool(check.get("ok")) for check in applicable) and (
        all(bool(check.get("ok")) for check in protocol_checks)
        if scope.get("annovar") else True
    )

    # Said during setup doctor, for all errors and warnings. A fatally withdrawn genome is INVALID no matter
    # how many of its files are present -- every path can resolve and every byte be readable while the
    # index was still built against the wrong FASTA, which is the case the flag exists to describe. A
    # superseded one stays valid, because nothing about it is wrong.
    from paralome.setup.withdrawal import FATAL, effective_withdrawal, replacements_for

    withdrawn = effective_withdrawal(str(genome_name), genome_config)
    if withdrawn is not None:
        withdrawn = dict(withdrawn)
        withdrawn["replacements"] = [
            {**item._asdict(), "headline": item.headline}
            for item in replacements_for(
                str(genome_name), genome_config, cfg.get("genomes", {}) or {}
            )
        ]
        if withdrawn.get("severity") == FATAL:
            valid = False

    # EVERY reason, not the first one (a genome withdrawn AND missing its FASTA, GTF and STAR index must
    # say both). A verdict that names one cause invites fixing that cause and expecting to be done.
    reasons: list[str] = []
    for check in applicable:
        if not check.get("ok"):
            reasons.append(f"{check.get('name')}: {check.get('detail') or check.get('status')}")
    if scope.get("annovar"):
        for item in protocol_checks:
            if not item.get("ok"):
                reasons.append(
                    f"annovar {item.get('protocol')}: {item.get('status')} "
                    f"({item.get('path') or 'no path'})"
                )
    if withdrawn is not None and withdrawn.get("severity") == FATAL:
        reasons.insert(
            0,
            f"withdrawn (fatal) -- {withdrawn.get('reason') or 'no reason given'}",
        )

    return {
        "requested": genome,
        "name": genome_name,
        "valid": valid,
        "reasons": reasons,
        "error": "",
        "identity": identity,
        "withdrawn": withdrawn,
        "config_file": str(config_path()),
        "analyses": list(genome_config.get("analyses") or []),
        "sections_in_scope": {k: bool(v) for k, v in scope.items()},
        "resources": resource_values,
        "checks": checks,
        "annovar_protocols": protocol_checks,
    }


def _shared_directory_report(path, expected_mode: int, expected_group: str) -> dict:
    """One shared directory: is it there, and is it the mode it has to be.

    Nothing has ever checked these, so a repository whose setgid bit
    was lost to a `chmod -R` looks completely normal until the second person
    adds a genome and nobody can read it. The mode IS the contract here, in a
    way that is true of almost nothing else Paralome touches.
    """
    import grp
    import os
    import stat

    entry = {
        "path": str(path) if path else "",
        "exists": False,
        "mode": "",
        "expected_mode": format(expected_mode, "04o"),
        "group": "",
        "expected_group": expected_group,
        "writable_by_you": False,
        "ok": False,
    }
    if not path:
        return entry
    try:
        info = os.stat(path)
    except OSError:
        return entry
    entry["exists"] = True
    entry["mode"] = format(stat.S_IMODE(info.st_mode), "04o")
    try:
        entry["group"] = grp.getgrgid(info.st_gid).gr_name
    except (KeyError, OverflowError):
        entry["group"] = str(info.st_gid)
    entry["writable_by_you"] = os.access(path, os.W_OK | os.X_OK)
    entry["ok"] = (
        stat.S_IMODE(info.st_mode) == expected_mode
        and (expected_group == "" or entry["group"] == expected_group)
    )
    return entry


def shared_setup_report() -> dict:
    """What this machine has decided, and whether the filesystem agrees."""
    from paralome.setup.genomes import (
        SHARED_DIR_MODE, config_for_a_report, default_site_registry_path, genome_repository,
        site_registry_path,
    )
    from paralome.setup.site import (
        has_site_config, load_site_config, reporting_on_the_site_file,
        site_config_path, site_config_problem, site_registry,
    )

    # Doctor reports a refused configuration, and a refused site
    # file; it does not stop on either. The site file's fault is `problem`.
    with reporting_on_the_site_file():
        site = load_site_config()
        config = config_for_a_report()
        repository = genome_repository(config)
        report = {
            "configured": has_site_config(),
            "config_file": str(site_config_path()),
            "problem": site_config_problem(),
            "shared_root": str(site.get("shared_root") or ""),
            "repository": _shared_directory_report(repository, 0o3775, ""),
            # Group root is expected only of the default registry; one a user
            # chose (PARALOME_SITE_REGISTRY or the user config) belongs to whoever made it.
            "registry": _shared_directory_report(
                site_registry_path(config), SHARED_DIR_MODE,
                "root" if site_registry() is not None
                or site_registry_path(config) == default_site_registry_path() else ""),
        }
    # Membership, not writability, is what decides it.
    from paralome.setup.site import shared_group_membership

    report["membership"] = shared_group_membership()
    return report


def doctor_report(*, initialize: bool = False, genome: Optional[str] = None) -> dict[str, Any]:
    """Return an installation/configuration report and optional genome diagnostics."""
    from paralome.setup.site import reporting_on_the_site_file

    # Beside a site file that is refused, this still reports.
    with reporting_on_the_site_file():
        return _doctor_report(initialize=initialize, genome=genome)


def _doctor_report(*, initialize: bool, genome: Optional[str]) -> dict[str, Any]:
    cfg_path = config_path()
    cfg: dict[str, Any] = {}
    config_ok = False
    config_error = ""
    if cfg_path.exists():
        try:
            cfg = load_config(initialize=False)
            config_ok = True
        except (ConfigValidationError, FileNotFoundError, json.JSONDecodeError) as exc:
            config_error = str(exc)
    elif initialize:
        initialize_config()
        cfg = load_config(initialize=False)
        config_ok = True
    else:
        config_error = "Configuration has not been initialized. Run 'paralome setup'."

    discovered = discover_executables()
    configured_execs = cfg.get("executables", {}) if isinstance(cfg.get("executables"), dict) else {}
    executables: dict[str, dict[str, Any]] = {}
    for name in CORE_EXECUTABLES + SLURM_EXECUTABLES + OPTIONAL_EXECUTABLES:
        configured = configured_execs.get(name, {}) if isinstance(configured_execs.get(name), dict) else {}
        configured_path = configured.get("path")
        discovered_path = discovered.get(name, {}).get("path")
        path = configured_path or discovered_path
        exists = bool(path and Path(path).exists())
        # THE RECORDING IS AUTHORITATIVE. binaries.resolve_tool is what every stage calls, and it prefers
        # the recorded path; a recorded path that no longer resolves raises PLM-0410 rather than falling
        # back. So a divergence here is a statement that PATH would find something else and Paralome is
        # deliberately not using it (see print_doctor_report).
        diverged = bool(
            configured_path and discovered_path and configured_path != discovered_path
        )
        executables[name] = {
            "required_for_core": name in CORE_EXECUTABLES,
            "configured_path": configured_path,
            "discovered_path": discovered_path,
            "path": path,
            "found": exists,
            "diverged": diverged,
            "version": configured.get("version") or discovered.get(name, {}).get("version"),
        }

    slurm_found = bool(executables.get("sbatch", {}).get("found"))
    backend = cfg.get("defaults", {}).get("backend") if config_ok else detect_backend(discovered)
    # What kind of machine Paralome has judged this to be, and why.
    # INSTALLATION.md section 1 sends administrators here to read it.
    from paralome.setup.deployment import explain_deployment

    deployment, deployment_source = explain_deployment(cfg if config_ok else None)
    genomes = cfg.get("genomes", {}) if isinstance(cfg.get("genomes"), dict) else {}
    genome_summaries = []
    if config_ok:
        for name in sorted(str(item) for item in genomes):
            item = genome_doctor_report(name, config=cfg)
            genome_summaries.append(
                {
                    "name": name,
                    "species": item.get("identity", {}).get("species", ""),
                    "assembly": item.get("identity", {}).get("assembly", ""),
                    "valid": bool(item.get("valid")),
                }
            )

    # ALL of the core tools, not any of them: one tool found must not score the whole section (a run
    # would get through setup and genome resolution and fail at alignment). SLURM tools are judged
    # separately, by the scheduler section, because a local-backend machine legitimately has none of them.
    missing_core = [
        name for name in CORE_EXECUTABLES
        if not executables.get(name, {}).get("found")
    ]
    shared_setup = shared_setup_report()
    # A site file that is refused refuses every run on the machine.
    # So does an account outside the shared root's group; every run
    # stops on it in the preflight, so doctor cannot call the machine ready.
    membership = shared_setup.get("membership")
    group_ok = membership is None or bool(membership.get("member"))
    site_ok = not shared_setup.get("problem") and group_ok
    sections = {
        "bootstrap": True,
        "configuration": config_ok and site_ok,
        "executables": not missing_core,
        "scheduler": bool(slurm_found or backend == "local"),
        "genomes": bool(genomes),
    }
    score_items = ["bootstrap", "configuration", "executables", "scheduler"]
    score = int(round(100 * sum(1 for k in score_items if sections[k]) / len(score_items)))

    report = {
        "schema": "paralome.doctor.report.v1",
        "python": {
            "version": platform.python_version(),
            "executable": shutil.which("python3") or shutil.which("python") or "",
        },
        "config": {"path": str(cfg_path), "exists": cfg_path.exists(), "ok": config_ok, "error": config_error},
        "executables": executables,
        "scheduler": {"backend": backend, "slurm_found": slurm_found},
        "deployment": {"kind": deployment, "source": deployment_source},
        "genomes": {"count": len(genomes), "names": sorted(genomes), "summaries": genome_summaries},
        "shared_setup": shared_setup,
        "sections": sections,
        "readiness_percent": score,
        "ready_for_setup": True,
        # A machine missing a core tool is not ready, whatever else is in place.
        "missing_core_executables": missing_core,
        "ready_for_pipeline": config_ok and site_ok and bool(genomes) and not missing_core,
        "in_shared_group": group_ok,
    }
    if genome is not None:
        report["genome"] = genome_doctor_report(genome, config=cfg) if config_ok else {
            "requested": genome,
            "name": None,
            "valid": False,
            # Not `unknown`: the genome may well exist -- there is simply no
            # configuration to look it up in, and the message says so.
            "unknown": False,
            "error": config_error or "Configuration is unavailable",
            "identity": {},
            "resources": {},
            "checks": [],
        }
    return report


def _mark(ok: bool) -> str:
    return "OK" if ok else "--"


def print_setup_result(config: dict[str, Any]) -> None:
    cfg_path = config_path()
    executables = config.get("executables", {}) if isinstance(config.get("executables"), dict) else {}
    found = sum(1 for v in executables.values() if isinstance(v, dict) and v.get("found"))
    total = len(executables)
    print("Paralome setup")
    print("===============")
    print(f"Config path          : {cfg_path}")
    print(f"Default backend      : {config.get('defaults', {}).get('backend')}")
    print(f"Executables found    : {found}/{total}")
    # Say what survived: this is the command people run when something already looks wrong.
    genomes = config.get("genomes", {}) if isinstance(config.get("genomes"), dict) else {}
    print(f"Genomes kept         : {len(genomes)}")
    print("")
    print("Next recommended step")
    print("---------------------")
    print("paralome setup doctor")


def _resource_value(genome_report: dict[str, Any], name: str) -> str:
    resource = genome_report.get("resources", {}).get(name, {})
    return str(resource.get("value") or "not configured")


def print_genome_doctor_report(genome_report: dict[str, Any]) -> None:
    # A name nobody registered gets no heading and no verdict. The
    # heading said "Genome Validation" and the verdict said INVALID, and neither
    # was true -- there was no genome, so nothing was validated and nothing was
    # found wanting. The CLI prints the list of real names after this.
    if genome_report.get("unknown"):
        print(f"No genome is registered here under the name "
              f"{str(genome_report.get('requested'))!r}.")
        print("")
        return

    print("Genome Validation")
    print("=================")
    print("")
    if genome_report.get("error"):
        print(f"Requested genome     : {genome_report.get('requested')}")
        print("Status               : INVALID")
        print(f"Message              : {genome_report.get('error')}")
        return

    # Printed before the identity, because if this genome is retired it is the
    # only line that matters and the rest is detail about a thing not to use.
    withdrawn = genome_report.get("withdrawn")
    if withdrawn:
        from paralome.setup.withdrawal import describe

        print(describe(str(genome_report.get("name")), withdrawn))
        for item in withdrawn.get("replacements") or []:
            print(f"{item.get('headline'):21}: {item.get('name')} ({item.get('detail')})")
            if item.get("caution"):
                print(f"{'':21}  {item['caution']}")
        print("")

    identity = genome_report.get("identity", {})
    print(f"Genome key           : {genome_report.get('name')}")
    print(f"Canonical name       : {identity.get('name') or '.'}")
    print(f"Species              : {identity.get('species') or '.'}")
    print(f"Assembly             : {identity.get('assembly') or '.'}")
    annotation = " ".join(
        part for part in (
            identity.get("annotation_source"), identity.get("annotation_release")
        ) if part
    )
    print(f"Annotation           : {annotation or '.'}")
    if identity.get("modified"):
        note = identity.get("modified_note") or ""
        print(f"Modified             : YES{f'  ({note})' if note else ''}")
    print(f"Config file          : {genome_report.get('config_file')}")
    analyses = genome_report.get("analyses") or []
    if analyses:
        print(f"Registered for       : {', '.join(analyses)}")
    scope = genome_report.get("sections_in_scope") or {}
    print("")
    print("Reference")
    print("---------")
    for name, label in (
        ("reference_fasta", "FASTA"),
        ("gtf", "GTF"),
        ("star_index", "STAR index"),
    ):
        check = next((item for item in genome_report.get("checks", []) if item.get("name") == name), {})
        if check and check.get("applies") is False:
            continue
        print(f"{label:20s}: {_mark(bool(check.get('ok')))} {_resource_value(genome_report, name)}")

    # Printed, not merely computed: STAR is always built with the GTF included, and this is the line that
    # says whether the build passed that check.
    annotation = next(
        (item for item in genome_report.get("checks", [])
         if item.get("name") == "star_index_annotation"),
        {},
    )
    if annotation and annotation.get("applies") is not False:
        print(f"{'  built with GTF':20s}: {_mark(bool(annotation.get('ok')))} "
              f"{annotation.get('detail') or ''}")

    # The section that existed everywhere except on the page.
    if scope.get("motifs", True):
        print("")
        print("Motifs")
        print("------")
        for name, label in (
            ("motif_database", "Database"),
            ("motif_species", "Species"),
        ):
            check = next(
                (item for item in genome_report.get("checks", []) if item.get("name") == name),
                {},
            )
            # A key left unset is the documented fallback, so it is marked
            # neither OK nor failed -- printing "OK" for a file nobody named
            # would claim a check that did not happen.
            if check.get("status") == "default":
                print(f"{label:20s}: -- {check.get('detail')}")
            else:
                print(f"{label:20s}: {_mark(bool(check.get('ok')))} {check.get('detail')}")

    # A section this genome is not for is named and skipped, not printed as five "not configured" lines
    # that invite fixing something that was never wrong.
    if scope.get("annovar", True):
        print("")
        print("ANNOVAR")
        print("-------")
        for name, label in (
            ("annovar_dir", "Installation"),
            ("annovar_db", "Database"),
            ("annovar_buildver", "Build"),
            ("annovar_protocol", "Protocols"),
            ("annovar_operation", "Operations"),
        ):
            print(f"{label:20s}: {_resource_value(genome_report, name)}")
        for item in genome_report.get("annovar_protocols", []):
            operation = item.get("operation") or "?"
            print(f"  {item.get('protocol'):18s} ({operation}): {_mark(bool(item.get('ok')))} {item.get('path')}")
        count_check = next(
            (item for item in genome_report.get("checks", []) if item.get("name") == "annovar_protocol_operation_count"),
            {},
        )
        print(f"Protocol alignment   : {_mark(bool(count_check.get('ok')))} {count_check.get('detail', '')}")

    if scope.get("snpeff", True):
        print("")
        print("SnpEff")
        print("------")
        for name, label in (
            ("snpeff_jar", "JAR"),
            ("snpeff_config", "Config"),
            ("snpeff_genome", "Genome"),
        ):
            check = next((item for item in genome_report.get("checks", []) if item.get("name") == name), {})
            print(f"{label:20s}: {_mark(bool(check.get('ok')))} {_resource_value(genome_report, name)}")

    # The gene-burden reference, under the DNA sections.
    if scope.get("burden", False):
        print("")
        print("Gene burden")
        print("-----------")
        check = next(
            (item for item in genome_report.get("checks", []) if item.get("name") == "burden_reference"),
            {},
        )
        if check.get("status") == "not built":
            print(f"{'Reference':20s}: -- {check.get('detail')}")
        else:
            print(f"{'Reference':20s}: {_mark(bool(check.get('ok')))} {check.get('detail')}")

    skipped = [name for name in ("annovar", "snpeff", "star", "motifs")
               if not scope.get(name, True)]
    if skipped:
        print("")
        print("Not checked")
        print("-----------")
        print(f"{'Sections':20s}: {', '.join(skipped)}")
        print(f"{'':20s}  this genome is not registered for them")
    print("")
    print("Result")
    print("------")
    print(f"Status               : {'VALID' if genome_report.get('valid') else 'INVALID'}")
    # The verdict has to carry its own reason. A genome whose every file
    # checked OK and which then reads INVALID sends people looking through the
    # sections above for the failure that is not there.
    #
    # And it has to carry ALL of them -- see genome_doctor_report.
    # A genome can be both withdrawn and missing every file it names, and
    # reporting only the withdrawal makes lifting it look like the fix.
    reasons = list(genome_report.get("reasons") or [])
    if not reasons and withdrawn:
        # A superseded genome is still VALID, so it has no entry in `reasons`;
        # say so here rather than leaving the warning off the report entirely.
        reasons = [
            f"withdrawn ({withdrawn.get('severity')}) -- "
            f"{withdrawn.get('reason') or 'no reason given'}"
        ]
    for index, reason in enumerate(reasons):
        label = "Reason" if index == 0 else ""
        print(f"{label:20s} {':' if index == 0 else ' '} {reason}")


def print_doctor_report(report: dict[str, Any]) -> None:
    if "genome" in report:
        print_genome_doctor_report(report["genome"])
        return

    print("Paralome Doctor")
    print("================")
    print("")
    print("Bootstrap")
    print("---------")
    py = report.get("python", {})
    print(f"Python               : OK {py.get('version')}")
    print("")
    print("Configuration")
    print("-------------")
    cfg = report.get("config", {})
    print(f"Config               : {_mark(bool(cfg.get('ok')))} {cfg.get('path')}")
    if cfg.get("error"):
        print(f"Message              : {cfg.get('error')}")
    site_problem = report.get("shared_setup", {}).get("problem")
    if site_problem:
        # The machine's own file, which no ordinary account can
        # edit. Every run is refused for it, so it is said here.
        print(f"Site file            : {_mark(False)} {report['shared_setup'].get('config_file')}")
        print(f"Message              : {site_problem}")
        print(f"{'':20s}  Every run on this machine is refused until an administrator corrects it.")
    print("")
    print("Executables")
    print("-----------")
    for name, entry in report.get("executables", {}).items():
        flag = _mark(bool(entry.get("found")))
        # This printed a bare "." for a tool that was not found,
        # which reads as a path rather than an absence.
        path = entry.get("path") or "not found on PATH"
        print(f"{name:20s}: {flag} {path}")
        if entry.get("diverged"):
            # The runner invokes the recorded path, so the arrow points at it. Both lines are printed,
            # because a divergence is worth seeing either way -- it usually means a tool was upgraded into a
            # different directory since setup last ran.
            print(f"{'':20s}  RECORDED {entry.get('configured_path')}  <- this is what runs")
            print(f"{'':20s}  ON PATH  {entry.get('discovered_path')}")
            print(f"{'':20s}  `paralome setup` re-records what is on PATH now")
    print("")
    print("Scheduler")
    print("---------")
    sched = report.get("scheduler", {})
    print(f"Backend              : {sched.get('backend')}")
    print(f"SLURM detected       : {sched.get('slurm_found')}")
    deployment = report.get("deployment", {})
    if deployment:
        print(f"Deployment           : {deployment.get('kind')} ({deployment.get('source')})")
        print(f"{'':20s}  set defaults.deployment in your config to overrule it")
    print("")
    print("Genomes")
    print("-------")
    genomes = report.get("genomes", {})
    print(f"Configured genomes   : {genomes.get('count', 0)}")
    summaries = genomes.get("summaries", [])
    if summaries:
        for item in summaries:
            identity = " ".join(part for part in (item.get("species"), item.get("assembly")) if part)
            print(f"{item.get('name'):20s}: {_mark(bool(item.get('valid')))} {identity}")
        print("")
        print("Detailed validation  : paralome setup doctor <genome>")
    elif genomes.get("names"):
        print(f"Genome names         : {', '.join(genomes.get('names'))}")
    shared = report.get("shared_setup", {})
    if shared.get("configured") or shared.get("repository", {}).get("exists"):
        print("")
        print("Shared setup")
        print("------------")
        if shared.get("shared_root"):
            print(f"{'Shared root':20s}: {shared.get('shared_root')}")
        for label, key in (("Genome repository", "repository"), ("Registry", "registry")):
            item = shared.get(key, {})
            if not item.get("path"):
                continue
            print(f"{label:20s}: {_mark(bool(item.get('ok')))} {item.get('path')}")
            if not item.get("exists"):
                print(f"{'':20s}  recorded, but not present on this machine")
                continue
            if not item.get("ok"):
                # The mode is the contract. Say what it is and what it must be,
                # rather than leaving somebody to decode 3775 from a manual.
                print(f"{'':20s}  mode {item.get('mode')}, expected "
                      f"{item.get('expected_mode')}"
                      + (f"; group {item.get('group')}, expected "
                         f"{item.get('expected_group')}"
                         if item.get("expected_group")
                         and item.get("group") != item.get("expected_group") else ""))
            if key == "repository" and not item.get("writable_by_you"):
                print(f"{'':20s}  you cannot add genomes here")
        membership = shared.get("membership")
        if membership is not None:
            print(f"{'Your account':20s}: {_mark(bool(membership.get('member')))} "
                  f"group {membership.get('group')}"
                  + ("" if membership.get("member") else " -- not a member"))
            if not membership.get("member"):
                from paralome.setup.site import not_in_shared_group_text

                print(f"{'':20s}  {not_in_shared_group_text(membership)}")

    print("")
    print("Readiness")
    print("---------")
    print(f"Setup readiness      : {report.get('readiness_percent')}%")
    absent = list(report.get("missing_core_executables") or [])
    if report.get("ready_for_pipeline"):
        print("Status               : Ready for configured pipeline use.")
    elif site_problem:
        print("Status               : Not ready -- this machine's site file is refused (see Configuration).")
    elif not report.get("in_shared_group", True):
        print("Status               : Not ready -- this account is not in the shared genome repository's group (see Shared setup).")
    elif absent:
        # Named, because "not ready" without the list sends someone back
        # through the Executables section counting dashes.
        print(f"Missing core tools   : {', '.join(absent)}")
        print("Status               : Not ready -- a core tool is not on PATH.")
    else:
        print("Status               : Bootstrap complete; configure a genome before pipeline use.")
    print("")
    print("Next recommended step")
    print("---------------------")
    if not cfg.get("ok"):
        print("paralome setup")
    elif site_problem:
        print(f"Ask this machine's administrator to correct {report['shared_setup'].get('config_file')}")
    elif not report.get("in_shared_group", True):
        group = (report["shared_setup"].get("membership") or {}).get("group")
        print(f"Ask this machine's administrator to add this account to group {group}, then log in again")
    elif absent and (report.get("genomes") or {}).get("count"):
        # A missing core tool is fixed by installing it, not by adding a genome.
        print(f"Install {', '.join(absent)} (or put it on PATH), then: paralome setup")
    elif not report.get("ready_for_pipeline"):
        if absent:
            print(f"Install {', '.join(absent)} (or put it on PATH), then: paralome setup")
        # Genome configuration is a subcommand: `paralome setup genome`.
        from paralome.setup.genomes import config_for_a_report, genome_repository

        repository = genome_repository(config_for_a_report())
        if repository is not None:
            print(f"paralome setup genome add {repository}/<folder>")
        else:
            print("paralome setup genome add <folder>   # register a genome")
        print("paralome setup genome list           # what this machine has")
    else:
        print("paralome <ANALYSIS> ...   # re-run the same command to resume")
