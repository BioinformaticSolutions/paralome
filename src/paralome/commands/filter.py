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

from pathlib import Path
from typing import Optional

import typer

from paralome.filters.profiles import get_profile, list_profiles
from paralome.filters.table import filter_segregation_table
from paralome.filtering.runner import (
    counts_label,
    print_filter_stage_result,
    print_filter_verify,
    run_filter_stage,
    verify_filter_stage,
)

filter_app = typer.Typer(help="Filter variants using configured analysis profiles")


def _default_out_prefix(segregation: Path, profile: str, out_dir: Optional[Path]) -> Path:
    """Derive a stable output prefix for legacy filter-only reruns."""
    base = segregation.name
    suffixes = [".segregation.tsv", ".tsv"]
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    profile_slug = (profile or "none").strip().lower().replace("_", "-")
    root = out_dir if out_dir is not None else segregation.parent
    return root / f"{base}.{profile_slug}"


def list_profiles_fn(filter_config: Optional[Path] = None) -> list[str]:
    return list_profiles(filter_config)


def run_filter_command(
    *,
    segregation: Path,
    out_prefix: Optional[Path],
    out_dir: Optional[Path],
    profile: str,
    filter_config: Optional[Path],
    list_profiles: bool = False,
) -> dict[str, object] | None:
    """Legacy thin wrapper around paralome.filters.table.filter_segregation_table.

    No command registers it; it is kept for imports and tests. The stage is
    ``paralome-worker filter run`` (``run_filter_stage``); This
    said ``paralome filter run``, which does not exist.
    """
    if list_profiles:
        for name in list_profiles_fn(filter_config):
            prof = get_profile(name, filter_config)
            desc = f" - {prof.description}" if prof.description else ""
            typer.echo(f"{name}{desc}")
        return None

    segregation = Path(segregation)
    if not segregation.exists():
        raise typer.BadParameter(f"Segregation table not found: {segregation}")

    if out_prefix is not None and out_dir is not None:
        raise typer.BadParameter("Use either --out-prefix or --out-dir, not both")

    resolved_prefix = Path(out_prefix) if out_prefix else _default_out_prefix(segregation, profile, out_dir)
    get_profile(profile, filter_config)

    summary = filter_segregation_table(
        segregation_tsv=segregation,
        out_prefix=resolved_prefix,
        profile_name=profile,
        filter_config=filter_config,
    )

    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    typer.echo(f"Filter profile: {profile}")
    typer.echo(f"Input: {segregation}")
    typer.echo(f"Output prefix: {resolved_prefix}")
    typer.echo(
        "Counts: "
        f"total={counts.get('total', 0)} "
        f"candidates={counts.get('candidates', 0)} "
        f"segregating={counts.get('segregating', 0)} "
        f"non_segregating={counts.get('non_segregating', 0)} "
        f"uninformative={counts.get('uninformative', 0)} "
        f"other={counts.get('other', 0)}"
    )
    typer.echo(f"Summary: {summary['outputs']['summary']}")
    return summary


@filter_app.command("run")
def filter_run_command(
    list_profiles_option: bool = typer.Option(False, "--list-profiles", help="List available profiles and exit"),
    segregation: Optional[Path] = typer.Option(None, "--segregation", help="Existing Paralome segregation TSV"),
    out: Optional[Path] = typer.Option(None, "--out", help="Filtering work directory"),
    profile: str = typer.Option("family-segregating", "--profile", help="Primary filter profile name"),
    profiles: Optional[str] = typer.Option(None, "--profiles", help="Comma-separated independent filter profiles"),
    scope: str = typer.Option("segregation_pass", "--scope", help="Filtering scope: segregation_pass, segregation_partial, segregation_fail, segregation_uninformative, all, or comma-separated scopes"),
    filter_config: Optional[Path] = typer.Option(None, "--filter-config", help="Optional filter profile config, a JSON file shaped like the packaged filter_profiles.json (JSON only; an earlier version)"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Output file prefix; default is workdir name"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite an existing completed filtering stage"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write plan/checkpoint but do not execute filtering"),
    threads: int = typer.Option(1, "--threads", help="Total local worker budget for shard execution"),
    shards: int = typer.Option(1, "--shards", help="Number of filtering shards to generate"),
    regions: Optional[Path] = typer.Option(None, "--regions", help="Optional regions.txt/BED-style region map for genomic shard assignment"),
    # See pipeline.region_shards REGION_CONVENTIONS.
    min_alternate_count: Optional[int] = typer.Option(None, "--min-alternate-count", help="With the other run settings: every germline profile's variant reads per genotype (the pipeline's --min-alternate-count)"),
    min_alternate_fraction: Optional[float] = typer.Option(None, "--min-alternate-fraction", help="Variant-read fraction a het or hom genotype needs (the pipeline's --min-alternate-fraction)"),
    min_qual: Optional[float] = typer.Option(None, "--min-qual", help="Site QUAL (the pipeline's --min-qual)"),
    min_depth: Optional[int] = typer.Option(None, "--min-depth", help="Genotype depth; no cut unless given"),
    min_gq: Optional[float] = typer.Option(None, "--min-gq", help="Genotype quality; no cut unless given"),
    regions_convention: str = typer.Option("one-based", "--regions-convention", help="How a chr:start-end line of --regions is read: one-based (1-based, both ends included; a hand-written list) or freebayes (0-based, end-exclusive: a FreeBayes region file). BED lines are 0-based either way."),
):
    """Run filtering as a checkpointed workflow stage."""
    if list_profiles_option:
        for name in list_profiles_fn(filter_config):
            prof = get_profile(name, filter_config)
            desc = f" - {prof.description}" if prof.description else ""
            typer.echo(f"{name}{desc}")
        return

    if segregation is None:
        raise typer.BadParameter("--segregation is required unless --list-profiles is used")
    if out is None:
        raise typer.BadParameter("--out is required unless --list-profiles is used")

    if min_alternate_count is not None or min_alternate_fraction is not None or min_qual is not None:
        # The profiles' quality thresholds come from the run's settings, not packaged numbers
        # (filters.config.with_run_settings); the config this run used is written beside its inputs.
        import json

        from paralome.filters.config import load_filter_config, with_run_settings

        if None in (min_alternate_count, min_alternate_fraction, min_qual):
            raise typer.BadParameter("--min-alternate-count, --min-alternate-fraction and --min-qual go together")
        settings = with_run_settings(
            load_filter_config(filter_config), min_alternate_count=min_alternate_count,
            min_alternate_fraction=min_alternate_fraction, min_qual=min_qual, min_depth=min_depth, min_gq=min_gq,
        )
        run_config = Path(out) / "inputs" / "filter_profiles.run.json"
        run_config.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(settings, indent=2, sort_keys=True) + "\n"
        if not run_config.is_file() or run_config.read_text() != text:
            run_config.write_text(text)
        filter_config = run_config
    elif min_depth is not None or min_gq is not None:
        raise typer.BadParameter("--min-depth and --min-gq go with --min-alternate-count, --min-alternate-fraction and --min-qual")

    result = run_filter_stage(
        segregation=segregation,
        workdir=out,
        profile=profile,
        profiles=profiles,
        scope=scope,
        filter_config=filter_config,
        prefix=prefix,
        overwrite=overwrite,
        dry_run=dry_run,
        threads=threads,
        shards=shards,
        regions=regions,
        region_convention=regions_convention,
    )
    print_filter_stage_result(result)


@filter_app.command("verify")
def filter_verify_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="Filtering work directory"),
):
    """Verify a filtering workflow stage."""

    report = verify_filter_stage(workdir)
    print_filter_verify(report)


@filter_app.command("status")
def filter_status_command(
    workdir: Path = typer.Option(..., "--workdir", "--out", help="Filtering work directory"),
):
    """Print concise filtering stage status."""
    report = verify_filter_stage(workdir)
    counts = report.get("counts") or {}
    print("Filtering status")
    print("================")
    print(f"Workdir             : {workdir}")
    print(f"Profiles            : {','.join(report.get('profiles') or [str(report.get('profile') or '.')])}")
    print(f"Scope               : {','.join(report.get('scope') or []) or '.'}")
    print(f"Complete            : {report.get('complete')}")
    # Each profile reads the whole table, so the totals are printed per profile (an aggregate over
    # profiles would count every variant once per profile).
    profiles = report.get("profile_outputs") or {}
    for name, prof in sorted(profiles.items()):
        per = prof.get("counts", {}) if isinstance(prof, dict) else {}
        print(f"{name:<20}: total={per.get('total', 0)} pass={per.get('pass', 0)}")
    label = counts_label(counts)
    print(f"Pass variants       : {counts.get('pass', counts.get('candidates', 0))}{label}")
    print(f"Total variants      : {counts.get('total', 0)}{label}")
