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
from pathlib import Path
from typing import Any, Optional
from paralome.errors import ParalomeError


def default_filter_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "filter_profiles.json"


def load_filter_config(config_path: str | Optional[Path] = None) -> dict[str, Any]:
    """The filter profile config: a JSON object (the packaged one when no path is given).

    JSON only, said so: a .yaml/.yml path and a file that is not JSON are
    refused with the file named and what to do.
    """
    path = Path(config_path) if config_path else default_filter_config_path()

    if not path.exists():
        raise ParalomeError(104, f"Filter config not found: {path}")

    fix = ("Write the filter profile config as JSON (the packaged "
           f"{default_filter_config_path().name} is the model); YAML is not read.")
    if path.suffix.lower() in (".yaml", ".yml"):
        raise ParalomeError(106, f"Filter config {path} is a YAML file, and filter configs are read as JSON only.",
                            fix=fix)
    try:
        with path.open() as fh:
            data = json.load(fh)
    except ValueError as exc:
        raise ParalomeError(106, f"Filter config {path} is not valid JSON: {exc}", fix=fix) from None

    if not isinstance(data, dict):
        raise ParalomeError(106, f"Filter config {path} must contain a JSON object at the top level, "
                                 f"not {type(data).__name__}.", fix=fix)

    return data


def config_sha256(config_path: str | Optional[Path] = None) -> str:
    import hashlib

    path = Path(config_path) if config_path else default_filter_config_path()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# No copy of the filter profiles is written into workdirs: filtering reads the packaged config
# unless --filter-config names another. To use other profiles, pass the file with --filter-config
# (filter run) -- the plan and summaries record which config, by path and SHA-256, was applied.



#: The quality thresholds the run's own settings replace (filtering
#: changes with user and project, so what the user set is what applies). The packaged numbers for these are not used.
RUN_SETTINGS_REPLACE = {
    "site": ("min_qual", "min_site_depth", "min_site_alt_reads"),
    "genotype": ("min_dp", "min_gq", "min_alt_reads", "min_alt_fraction_het", "min_alt_fraction_hom"),
}


def with_run_settings(
    config: dict[str, Any],
    *,
    min_alternate_count: int,
    min_alternate_fraction: float,
    min_qual: float,
    min_depth: Optional[int] = None,
    min_gq: Optional[float] = None,
) -> dict[str, Any]:
    """The filter config with every germline profile's quality thresholds set from the run.

    Variant reads per genotype = --min-alternate-count; het and hom variant-read fraction =
    --min-alternate-fraction; site QUAL = --min-qual; genotype depth and GQ only when --min-depth / --min-gq
    are given; no site depth or site variant-read cut. A null threshold is not applied. Profiles keep their
    question (mode, classes, rarity, segregation score, scope) and their other site rules. The somatic
    profiles, which filtering refuses, are left as they are.
    """
    import copy

    out = copy.deepcopy(config)
    profiles = out.get("profiles", out)
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("mode") == "somatic":
            continue
        quality = profile.setdefault("quality", {})
        site = quality.setdefault("site", {})
        genotype = quality.setdefault("genotype", {})
        site.update({"min_qual": float(min_qual), "min_site_depth": None, "min_site_alt_reads": None})
        genotype.update({
            "min_alt_reads": int(min_alternate_count),
            "min_alt_fraction_het": float(min_alternate_fraction),
            "min_alt_fraction_hom": float(min_alternate_fraction),
            "min_dp": None if min_depth is None else int(min_depth),
            "min_gq": None if min_gq is None else float(min_gq),
        })
    return out
