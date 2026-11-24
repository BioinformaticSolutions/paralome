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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import load_filter_config


SEGREGATING_CLASSES = {
    "affected_only",
    "shared_affected",
    "affected_enriched",
    "possible_dominant",
    "possible_recessive",
    "possible_de_novo",
}

NON_SEGREGATING_CLASSES = {"non_segregating"}
UNINFORMATIVE_CLASSES = {"uninformative", "parental_shared"}


@dataclass(frozen=True)
class AnalysisProfile:
    name: str
    mode: str = "unknown"
    rarity: str = "unknown"
    description: str = ""
    thresholds: dict[str, Any] = field(default_factory=dict)
    candidate_classes: set[str] = field(default_factory=set)
    segregating_classes: set[str] = field(default_factory=lambda: set(SEGREGATING_CLASSES))
    non_segregating_classes: set[str] = field(default_factory=lambda: set(NON_SEGREGATING_CLASSES))
    uninformative_classes: set[str] = field(default_factory=lambda: set(UNINFORMATIVE_CLASSES))
    min_segregation_score: float = 0.0
    require_affected_alt: bool = True
    allow_unaffected_alt: bool = True
    require_control_absence: bool = False
    quality: dict[str, Any] = field(default_factory=dict)
    population: dict[str, Any] = field(default_factory=dict)
    annotation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def min_qual(self) -> Optional[float]:
        return _nested(self.quality, "site", "min_qual")



def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_profile_name(name: Optional[str]) -> str:
    if not name:
        return "none"
    return name.strip().lower().replace("_", "-")


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()}
    return {str(x).strip().lower() for x in value}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _profile_from_dict(name: str, data: dict[str, Any]) -> AnalysisProfile:
    thresholds = dict(data.get("thresholds") or {})
    if "min_segregation_score" in data:
        thresholds.setdefault("min_segregation_score", data.get("min_segregation_score"))
    quality = dict(data.get("quality") or {})
    population = dict(data.get("population") or {})
    annotation = dict(data.get("annotation") or {})

    return AnalysisProfile(
        name=normalize_profile_name(name),
        mode=str(data.get("mode", "unknown")),
        rarity=str(data.get("rarity", "unknown")),
        description=str(data.get("description", "")),
        thresholds=thresholds,
        candidate_classes=_as_set(data.get("candidate_classes")),
        segregating_classes=_as_set(data.get("segregating_classes")) or set(SEGREGATING_CLASSES),
        non_segregating_classes=_as_set(data.get("non_segregating_classes")) or set(NON_SEGREGATING_CLASSES),
        uninformative_classes=_as_set(data.get("uninformative_classes")) or set(UNINFORMATIVE_CLASSES),
        min_segregation_score=_as_float(data.get("min_segregation_score", thresholds.get("min_segregation_score", 0.0))),
        require_affected_alt=bool(data.get("require_affected_alt", True)),
        allow_unaffected_alt=bool(data.get("allow_unaffected_alt", True)),
        require_control_absence=bool(data.get("require_control_absence", False)),
        quality=quality,
        population=population,
        annotation=annotation,
        raw=dict(data),
    )


def load_profiles(config_path: str | Optional[Path] = None) -> dict[str, AnalysisProfile]:
    cfg = load_filter_config(config_path)
    raw_profiles = cfg.get("profiles", cfg)
    profiles = {
        normalize_profile_name(name): _profile_from_dict(name, dict(data or {}))
        for name, data in raw_profiles.items()
    }
    profiles.setdefault("none", _profile_from_dict("none", {"description": "Do not filter.", "candidate_classes": []}))
    return profiles


def list_profiles(config_path: str | Optional[Path] = None) -> list[str]:
    return sorted(load_profiles(config_path))


def get_profile(name: Optional[str], config_path: str | Optional[Path] = None) -> AnalysisProfile:
    key = normalize_profile_name(name)
    profiles = load_profiles(config_path)
    if key not in profiles:
        raise ValueError(f"Unknown analysis profile '{name}'. Available profiles: {', '.join(sorted(profiles))}")
    return profiles[key]


def profile_scope(name: Optional[str], config_path: str | Optional[Path] = None) -> Optional[str]:
    """The segregation buckets a profile filters when the caller names none.

    The filter stage reads only ``segregation_pass`` unless told
    otherwise, and a variant carried by one of forty unrelated probands is
    ``affected_enriched`` -- the partial bucket -- so under the default it was
    never looked at. A profile written for such a cohort says so here
    (``"scope"`` in filter_profiles.json) and the pipeline passes it on.
    ``None`` means the stage default.
    """
    raw = get_profile(name, config_path).raw.get("scope")
    if not raw:
        return None
    if isinstance(raw, (list, tuple)):
        return ",".join(str(item) for item in raw)
    return str(raw)
