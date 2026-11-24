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

"""The differential stages: the core statistics on mass-spec values.

Nothing here is a statistic. The comparison is built as the core's
``PseudobulkComparison`` (samples are the design; log2(x + 1) as for RNA),
planned, fitted, corrected and written by ``singlecell.stats_core
.compute_statistics`` and ``engine.fit_plan`` -- T, paired-T, LM and ANOVA,
arms, factors, covariates and the result folders exactly as every other
pipeline has them.

What is particular to mass spec is only which samples a feature was measured
in. In the ``na`` arm a missing value stays missing, so a feature is tested on the samples that
measured it: the features are grouped by the set of samples that measured
them, and each group is fitted by the core engine with the same plan made on
those samples (``fit_by_measured_samples``). A statistic those samples cannot
support (an arm left with one sample, an ANOVA cell left empty) leaves the
feature untested in it, as a stratum with too few samples does. In the
``half_min`` arm every testable feature is measured everywhere, so it is one
group and the ordinary fit.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np


def build_comparison(
    keys: Sequence[str],
    annotation: Sequence[Mapping[str, object]],
    linear: np.ndarray,
    rows: Sequence[Mapping[str, str]],
    *,
    tested_values: Optional[np.ndarray] = None,
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    all_rows: Optional[Sequence[Mapping[str, str]]] = None,
):
    """The core comparison for these samples (``rows``, one per column of ``linear``).

    ``tested_values`` replaces log2(linear + 1) as the values tested (the
    phosphopeptide-relative-to-protein measure); otherwise it is computed here
    exactly as the core computes it for CPM.
    """
    from paralome.singlecell.stats_core import PSEUDOBULK_PSEUDOCOUNT, PseudobulkComparison

    rows = [dict(row) for row in rows]
    with np.errstate(invalid="ignore"):
        logged = np.log2(linear + PSEUDOBULK_PSEUDOCOUNT) if tested_values is None else tested_values
    return PseudobulkComparison(
        feature_ids=list(keys), annotation=[dict(a) for a in annotation],
        sample_ids=[row["sample_id"] for row in rows], sample_rows=rows,
        n_cells=np.ones(len(rows)), cpm=linear, log2_cpm=logged,
    )


def fit_by_measured_samples(
    comparison, request, *, eligible: np.ndarray, alpha: float,
    factors: Sequence[str] = (), pair_by: Optional[str] = None,
    covariate: Optional[str] = None, captured: Optional[List[object]] = None,
    block_by: Optional[str] = None,
) -> Callable[[object], List[object]]:
    """``fit_values`` for ``compute_statistics``: the plan fitted per set of measured samples."""
    from paralome.engine import fit_plan
    from paralome.singlecell.stats_core import plan_statistics
    from paralome.stats import StatisticRequest

    def fit(plan) -> List[object]:
        y = np.asarray(comparison.log2_cpm, dtype=float)
        raw = np.asarray(comparison.cpm, dtype=float)
        n_features, n_samples = y.shape
        # The full shape of every result, untested everywhere: the core engine
        # on nothing eligible.
        template = fit_plan(np.zeros_like(y), plan, raw=None, eligible=np.zeros(n_features, bool), alpha=alpha)
        slots = {}
        for result in template:
            result.columns = {name: np.array(values, copy=True) for name, values in result.columns.items()}
            slots[(result.folder, result.name)] = result
        measured = np.isfinite(y)
        candidates = np.where(eligible & measured.any(axis=1))[0]
        patterns: Dict[bytes, List[int]] = {}
        for i in candidates:
            patterns.setdefault(measured[i].tobytes(), []).append(i)
        for pattern_bytes, members in patterns.items():
            present = np.frombuffer(pattern_bytes, dtype=bool)
            idx = np.where(present)[0]
            if len(idx) < 2:
                continue
            members_arr = np.asarray(members)
            if len(idx) == n_samples:
                sub_plans = [plan]
            else:
                rows = [comparison.sample_rows[i] for i in idx]
                sub_plans = []
                for statistic_plan in plan.statistics:
                    try:
                        sub_plans.append(plan_statistics(
                            rows, StatisticRequest((statistic_plan.statistic,), statistic_plan.data_level),
                            factors=factors,
                            pair_by=pair_by if statistic_plan.statistic == "paired-t" else None,
                            covariate=covariate if statistic_plan.statistic == "lm" else None,
                            block_by=block_by if statistic_plan.statistic in ("anova", "lm") else None,
                        ))
                    except Exception:  # noqa: BLE001 - the design cannot be made on these samples
                        continue
            for sub_plan in sub_plans:
                fitted = fit_plan(
                    y[np.ix_(members_arr, idx)], sub_plan, raw=raw[np.ix_(members_arr, idx)],
                    eligible=np.ones(len(members_arr), bool), alpha=alpha,
                )
                for result in fitted:
                    slot = slots.get((result.folder, result.name))
                    if slot is None:
                        continue
                    for name, values in result.columns.items():
                        if name in slot.columns:
                            target = slot.columns[name]
                            values = np.asarray(values)
                            if target.dtype != values.dtype and target.dtype != object:
                                target = target.astype(np.result_type(target.dtype, values.dtype))
                                slot.columns[name] = target
                            target[members_arr] = values
        if captured is not None:
            captured.extend(template)
        return template

    return fit
