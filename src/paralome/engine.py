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

"""The differential engine: one plan, fitted on one matrix, the same way everywhere.

Bulk RNA (log2 FPKM), bulk ATAC (log2 CPM) and the pseudobulk
single-cell pipelines (log2 CPM per pseudobulk sample) all hand this module a
features x samples matrix and a ``comparisons.DifferentialPlan``, and get back
one ``ComparisonResult`` per comparison, with the same columns whichever
pipeline asked. Every statistic is computed by ``paralome.stats``; nothing
here is a statistic of its own, only the routing of the plan's sample groups
into the shared functions.

The cell-level fits (``:cell``) answer the same plan with cells as the
observations and return the same ``ComparisonResult`` -- see
``paralome.atac.singlecell_stats``.

Block-friendly: ``fit_plan`` on a slice of features returns the raw (uncorrected)
columns for that slice; ``concatenate`` joins slices in order; ``finalize``
corrects each comparison's p-values over the features it tested (BH, per
comparison and per ANOVA term) and marks what is significant at the run's
--alpha. A parallel stage fits blocks and finalizes once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from paralome.comparisons import (
    KIND_CONTRAST,
    KIND_FTEST,
    KIND_SLOPE,
    Comparison,
    DifferentialPlan,
    StatisticPlan,
)
from paralome.stats import (
    anova_table,
    factorial_from_sample_sums,
    one_way_anova,
    t_two_sided_p,
    benjamini_hochberg,
    constant_rows,
    contrast_table,
    group_moments,
    minimum_detectable_d,
    observed_power_array,
    ols_covariate_block_fit,
    ols_covariate_fit,
    paired_t_from_pairs,
    student_t_from_moments,
)

#: The columns every directional result carries, in the order they are written.
CONTRAST_COLUMNS = (
    "tested", "log2_fold_change", "control_mean", "treatment_mean",
    "raw_control_mean", "raw_treatment_mean", "statistic", "df", "p_value", "q_value",
    "n_control", "n_treatment", "cohens_d", "sd_log2", "mde_log2fc", "observed_power",
)

#: An F-test's columns. ``log2_fold_change`` is its direction (the signed
#: estimate of its own contrast, or the interaction coefficient), so a
#: consumer that classifies up/down -- motif enrichment -- reads every result
#: the same way.
FTEST_COLUMNS = (
    "tested", "log2_fold_change", "F", "df_term", "df_error",
    "p_value", "q_value",
)

#: The one-way ANOVA's F-test also carries its linear trend on the factor's
#: values and the departure from that trend. Each p is corrected
#: (BH) over the features tested, like every other p.
TREND_COLUMNS = (
    "trend_slope", "trend_slope_se", "trend_F", "trend_p", "trend_q",
    "lack_of_fit_F", "lack_of_fit_df", "lack_of_fit_p", "lack_of_fit_q",
)
#: An extra p-value a result may carry, and the q-value finalize writes for it.
EXTRA_P_VALUES = (("trend_p", "trend_q"), ("lack_of_fit_p", "lack_of_fit_q"))

#: Statistic-specific columns, after the shared ones.
EXTRA_COLUMNS = {
    "paired-t": ("n_pairs", "mean_within_pair_difference", "sd_within_pair_difference"),
    "lm": ("slope", "slope_se", "intercept", "pearson_r", "r_squared", "n_regressed"),
    "anova": ("standard_error", *TREND_COLUMNS),
}


@dataclass
class ComparisonResult:
    """One comparison's per-feature results, as named columns of equal length."""

    statistic: str
    folder: str
    comparison: Comparison
    columns: Dict[str, np.ndarray]
    #: Per-comparison facts for headers and the power report: replication unit,
    #: n per arm, df, the design's balance, the covariate.
    facts: Dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.comparison.name

    @property
    def kind(self) -> str:
        return self.comparison.kind

    def column_order(self) -> List[str]:
        base = list(FTEST_COLUMNS if self.kind == KIND_FTEST else CONTRAST_COLUMNS)
        extras = [c for c in EXTRA_COLUMNS.get(self.statistic, ()) if c in self.columns]
        rest = [c for c in self.columns if c not in base and c not in extras]
        return [c for c in base if c in self.columns] + extras + rest

    def __len__(self) -> int:
        first = next(iter(self.columns.values()), None)
        return 0 if first is None else len(first)


def _full(n: int, value: float = np.nan) -> np.ndarray:
    return np.full(n, value, dtype=float)


def _eligible_mask(eligible: Optional[np.ndarray], n: int) -> np.ndarray:
    return np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)


def _raw_means(raw: Optional[np.ndarray], index: Sequence[int], n: int) -> np.ndarray:
    if raw is None or not len(index):
        return _full(n)
    return np.asarray(raw, dtype=float)[:, list(index)].mean(axis=1)


def _finish(columns: Dict[str, np.ndarray], eligible: np.ndarray) -> Dict[str, np.ndarray]:
    """Untested means untested everywhere: no p, no statistic, flagged."""
    p = columns["p_value"]
    tested = np.isfinite(p) & eligible
    columns["p_value"] = np.where(tested, p, np.nan)
    # A feature that was not tested has no statistic, effect size or power --
    # a 0 there would read as "tested, no effect". Its means stay: they are
    # what the expression was.
    for name in ("statistic", "F", "cohens_d", "observed_power", "mde_log2fc", "standard_error",
                 "trend_slope", "trend_slope_se", "trend_F", "trend_p", "lack_of_fit_F",
                 "lack_of_fit_p"):
        if name in columns:
            columns[name] = np.where(tested, np.asarray(columns[name], dtype=float), np.nan)
    columns["tested"] = tested
    return columns


def fit_two_groups(
    y: np.ndarray, raw: Optional[np.ndarray], comparison: Comparison, eligible: np.ndarray,
    alpha: float,
) -> Dict[str, np.ndarray]:
    """Student's t: the two groups of ``comparison``."""
    n = y.shape[0]
    c_mean, c_var, n_c = group_moments(y, comparison.control_index)
    t_mean, t_var, n_t = group_moments(y, comparison.treatment_index)
    t_stat, df, p = student_t_from_moments(c_mean, c_var, n_c, t_mean, t_var, n_t)
    flat = constant_rows(y[:, list(comparison.samples())])
    t_stat = np.where(flat, np.nan, t_stat)
    p = np.where(flat, np.nan, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = ((n_c - 1) * c_var + (n_t - 1) * t_var) / (n_c + n_t - 2)
        sd = np.sqrt(np.maximum(pooled, 0.0))
        d = np.where(sd > 0, (t_mean - c_mean) / sd, np.where(np.isfinite(sd), 0.0, np.nan))
    mde_d = minimum_detectable_d(n_c, n_t, alpha=alpha)
    return _finish({
        "log2_fold_change": t_mean - c_mean,
        "control_mean": c_mean, "treatment_mean": t_mean,
        "raw_control_mean": _raw_means(raw, comparison.control_index, n),
        "raw_treatment_mean": _raw_means(raw, comparison.treatment_index, n),
        "statistic": t_stat, "df": _full(n, df), "p_value": p,
        "n_control": _full(n, n_c), "n_treatment": _full(n, n_t),
        "cohens_d": d, "sd_log2": sd, "mde_log2fc": mde_d * sd,
        "observed_power": observed_power_array(d, n_c, n_t, alpha),
    }, eligible)


def fit_paired(
    y: np.ndarray, raw: Optional[np.ndarray], comparison: Comparison, eligible: np.ndarray,
    alpha: float,
) -> Dict[str, np.ndarray]:
    """paired-t: one-sample t on each pair's treatment-minus-control difference."""
    n = y.shape[0]
    control_index = [c for c, _t, _p in comparison.pairs]
    treatment_index = [t for _c, t, _p in comparison.pairs]
    differences = y[:, treatment_index] - y[:, control_index]
    t_stat, df, p = paired_t_from_pairs(differences)
    flat = constant_rows(differences)
    t_stat = np.where(flat, np.nan, t_stat)
    p = np.where(flat, np.nan, p)
    mean_difference = differences.mean(axis=1)
    sd_difference = differences.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(sd_difference > 0, mean_difference / sd_difference, np.nan)
    n_pairs = len(comparison.pairs)
    mde_d = minimum_detectable_d(n_pairs, n_pairs, alpha=alpha)
    return _finish({
        "log2_fold_change": mean_difference,
        "control_mean": y[:, control_index].mean(axis=1),
        "treatment_mean": y[:, treatment_index].mean(axis=1),
        "raw_control_mean": _raw_means(raw, control_index, n),
        "raw_treatment_mean": _raw_means(raw, treatment_index, n),
        "statistic": t_stat, "df": _full(n, df), "p_value": p,
        "n_control": _full(n, n_pairs), "n_treatment": _full(n, n_pairs),
        "cohens_d": d, "sd_log2": sd_difference, "mde_log2fc": mde_d * sd_difference,
        "observed_power": observed_power_array(d, n_pairs, n_pairs, alpha),
        "n_pairs": _full(n, n_pairs),
        "mean_within_pair_difference": mean_difference,
        "sd_within_pair_difference": sd_difference,
    }, eligible)


def fit_slope(
    y: np.ndarray, raw: Optional[np.ndarray], plan: StatisticPlan, comparison: Comparison,
    eligible: np.ndarray,
) -> Dict[str, np.ndarray]:
    """lm: the regression on the covariate over ``comparison``'s samples.

    The slope is the result (log2 per unit of the covariate, which is what
    downstream reads as its log2 fold change). The two arms' means, where the
    run has arms, ride along for orientation only.
    """
    n = y.shape[0]
    members = list(comparison.regressed_index)
    x = np.asarray(plan.covariate_values, dtype=float)[members]
    sub = y[:, members]
    if plan.block_values is not None:      # --block-by
        fit = ols_covariate_block_fit(sub, x, [plan.block_values[i] for i in members])
    else:
        fit = ols_covariate_fit(sub, x)
    flat = constant_rows(sub)
    p = np.where(flat, np.nan, fit["p_value"])
    t_stat = np.where(flat, np.nan, fit["t_statistic"])

    def arm_mean(index: Sequence[int]) -> np.ndarray:
        return y[:, list(index)].mean(axis=1) if len(index) else _full(n)

    return _finish({
        "log2_fold_change": fit["slope"],
        "control_mean": arm_mean(comparison.control_index),
        "treatment_mean": arm_mean(comparison.treatment_index),
        "raw_control_mean": _raw_means(raw, comparison.control_index, n),
        "raw_treatment_mean": _raw_means(raw, comparison.treatment_index, n),
        "statistic": t_stat, "df": _full(n, fit["df"]), "p_value": p,
        "n_control": _full(n, len(comparison.control_index)),
        "n_treatment": _full(n, len(comparison.treatment_index)),
        "cohens_d": _full(n), "sd_log2": sub.std(axis=1, ddof=1),
        "mde_log2fc": _full(n), "observed_power": _full(n),
        "slope": fit["slope"], "slope_se": fit["slope_se"], "intercept": fit["intercept"],
        "pearson_r": fit["pearson_r"], "r_squared": fit["r_squared"],
        "n_regressed": _full(n, len(members)),
    }, eligible)


def fit_anova(
    y: np.ndarray, raw: Optional[np.ndarray], plan: StatisticPlan, eligible: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """The factorial ANOVA: an F-test per term and every contrast, every feature, one fit."""
    n = y.shape[0]
    design = np.asarray(plan.design, dtype=float)
    terms = anova_table(y, design, plan.term_columns)
    contrasts = [c for c in plan.comparisons if c.kind == KIND_CONTRAST]
    estimates, df, beta = contrast_table(y, design, [(c.name, c.weights) for c in contrasts])
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for comparison in plan.comparisons:
        if comparison.kind == KIND_FTEST:
            term = terms[str(comparison.term)]
            out[comparison.name] = _finish({
                "log2_fold_change": np.where(np.isfinite(term["p"]),
                                             _direction(comparison, estimates, beta, n), np.nan),
                "F": term["F"], "df_term": _full(n, term["df_term"]),
                "df_error": _full(n, term["df_error"]), "p_value": term["p"],
            }, eligible)
        else:
            values = estimates[comparison.name]
            c_mean = y[:, list(comparison.control_index)].mean(axis=1)
            t_mean = y[:, list(comparison.treatment_index)].mean(axis=1)
            out[comparison.name] = _finish({
                "log2_fold_change": values["estimate"],
                "control_mean": c_mean, "treatment_mean": t_mean,
                "raw_control_mean": _raw_means(raw, comparison.control_index, n),
                "raw_treatment_mean": _raw_means(raw, comparison.treatment_index, n),
                "statistic": values["t"], "df": _full(n, df), "p_value": values["p"],
                "n_control": _full(n, len(comparison.control_index)),
                "n_treatment": _full(n, len(comparison.treatment_index)),
                "cohens_d": _full(n), "sd_log2": _full(n), "mde_log2fc": _full(n),
                "observed_power": _full(n),
                "standard_error": values["se"],
            }, eligible)
    return out


def fit_anova_cells(
    sums: np.ndarray, squares: np.ndarray, counts: np.ndarray, raw_sums: Optional[np.ndarray],
    plan: StatisticPlan, eligible: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """The factorial ANOVA with the cells as N (anova:cell, one sample per combination; ).

    ``stats.factorial_from_sample_sums`` on each sample's sum over its cells;
    the columns ``fit_anova`` writes, the means over the cells of each side's
    samples.
    """
    n = sums.shape[0]
    contrasts = [c for c in plan.comparisons if c.kind == KIND_CONTRAST]
    terms, estimates, df, beta = factorial_from_sample_sums(
        sums, squares, counts, np.asarray(plan.design, dtype=float), plan.term_columns,
        [(c.name, c.weights) for c in contrasts],
    )

    def mean_over(values: Optional[np.ndarray], index: Sequence[int]) -> np.ndarray:
        cells = float(np.asarray(counts, dtype=float)[list(index)].sum())
        if values is None or not cells:
            return _full(n)
        return np.asarray(values, dtype=float)[:, list(index)].sum(axis=1) / cells

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for comparison in plan.comparisons:
        if comparison.kind == KIND_FTEST:
            term = terms[str(comparison.term)]
            out[comparison.name] = _finish({
                "log2_fold_change": np.where(np.isfinite(term["p"]),
                                             _direction(comparison, estimates, beta, n), np.nan),
                "F": term["F"], "df_term": _full(n, term["df_term"]),
                "df_error": _full(n, term["df_error"]), "p_value": term["p"],
            }, eligible)
        else:
            values = estimates[comparison.name]
            out[comparison.name] = _finish({
                "log2_fold_change": values["estimate"],
                "control_mean": mean_over(sums, comparison.control_index),
                "treatment_mean": mean_over(sums, comparison.treatment_index),
                "raw_control_mean": mean_over(raw_sums, comparison.control_index),
                "raw_treatment_mean": mean_over(raw_sums, comparison.treatment_index),
                "statistic": values["t"], "df": _full(n, df), "p_value": values["p"],
                "n_control": _full(n, len(comparison.control_index)),
                "n_treatment": _full(n, len(comparison.treatment_index)),
                "cohens_d": _full(n), "sd_log2": _full(n), "mde_log2fc": _full(n),
                "observed_power": _full(n),
                "standard_error": values["se"],
                "n_control_cells": _full(n, float(np.asarray(counts)[list(comparison.control_index)].sum())),
                "n_treatment_cells": _full(n, float(np.asarray(counts)[list(comparison.treatment_index)].sum())),
            }, eligible)
    return out


def _direction(comparison: Comparison, estimates: Mapping[str, Mapping[str, np.ndarray]],
               beta: np.ndarray, n: int) -> np.ndarray:
    """An F-test's direction, as the plan says where it comes from.

    A main effect of a two-level factor: its main-effect contrast (the same
    test, the same model). A one-column interaction: its coefficient. A term
    of several columns has no single direction -- each of its levels has its
    own, in its contrasts -- so its log2_fold_change is left empty.
    """
    source = comparison.direction
    if not source:
        return _full(n)
    if source.startswith("coef:"):
        return np.asarray(beta[:, int(source.split(":", 1)[1])], dtype=float)
    if source in estimates:
        return np.asarray(estimates[source]["estimate"], dtype=float)
    return _full(n)


def fit_one_way(
    y: np.ndarray, raw: Optional[np.ndarray], plan: StatisticPlan, eligible: np.ndarray,
    groups: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """The one-way ANOVA: its F-test (with trend and lack of fit on numeric levels) and its contrasts.

    ``groups`` are each level's observation columns of ``y``: the plan's
    samples by default, the cells of those samples at ``:cell``.
    """
    n = y.shape[0]
    groups = [list(g) for g in (groups if groups is not None else plan.level_members)]
    fit = one_way_anova(
        y, groups, None if plan.level_values is None else np.asarray(plan.level_values, dtype=float),
    )
    means = fit["means"]
    df_within = float(fit["df_within"])
    out: Dict[str, Dict[str, np.ndarray]] = {}
    position = {level: j for j, level in enumerate(plan.levels)}
    for comparison in plan.comparisons:
        if comparison.kind == KIND_FTEST:
            columns = {
                # A directed F: its direction and value are the trend's,
                # log2 per unit of the factor -- the same quantity lm reports.
                # Levels that are not numbers have no trend, so the F has no
                # direction and carries no trend columns; each level's
                # direction is its contrast's.
                "log2_fold_change": (np.where(np.isfinite(fit["p"]), fit["trend_slope"], np.nan)
                                     if plan.level_values is not None else _full(n)),
                "F": fit["F"], "df_term": _full(n, float(fit["df_between"])),
                "df_error": _full(n, df_within), "p_value": fit["p"],
            }
            if plan.level_values is not None:
                columns.update({
                    "trend_slope": fit["trend_slope"], "trend_slope_se": fit["trend_slope_se"],
                    "trend_F": fit["trend_F"], "trend_p": fit["trend_p"],
                    "lack_of_fit_F": fit["lack_of_fit_F"],
                    "lack_of_fit_df": _full(n, float(len(plan.levels) - 2)),
                    "lack_of_fit_p": fit["lack_of_fit_p"],
                })
            out[comparison.name] = _finish(columns, eligible)
            continue
        a, b = position[str(comparison.control)], position[str(comparison.treatment)]
        n_a, n_b = len(groups[a]), len(groups[b])
        estimate = means[:, b] - means[:, a]
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(fit["mse"] * (1.0 / n_a + 1.0 / n_b))
            t_value = np.where(se > 0, estimate / se, np.nan)
        out[comparison.name] = _finish({
            "log2_fold_change": estimate,
            "control_mean": means[:, a], "treatment_mean": means[:, b],
            "raw_control_mean": _raw_means(raw, groups[a], n),
            "raw_treatment_mean": _raw_means(raw, groups[b], n),
            "statistic": t_value, "df": _full(n, df_within),
            "p_value": t_two_sided_p(t_value, df_within),
            "n_control": _full(n, n_a), "n_treatment": _full(n, n_b),
            "cohens_d": _full(n), "sd_log2": _full(n), "mde_log2fc": _full(n),
            "observed_power": _full(n), "standard_error": se,
        }, eligible)
    return out


def comparison_facts(plan: StatisticPlan, comparison: Comparison, replication_unit: str) -> Dict[str, object]:
    facts: Dict[str, object] = {
        "replication_unit": replication_unit,
        "description": comparison.description,
        "role": comparison.role,
    }
    if comparison.kind == KIND_SLOPE:
        facts.update(covariate=plan.covariate_label, covariate_kind=plan.covariate_kind)
    if plan.statistic == "paired-t":
        facts["pair_by"] = plan.pair_by
        facts["pair_ids"] = [p for _c, _t, p in comparison.pairs]
    if plan.block_by:                      #
        facts["block_by"] = plan.block_by
    if plan.statistic == "anova" and plan.one_way:
        facts["one_way"] = True
        facts["balanced_design"] = bool(plan.balanced)
        facts["factor"] = plan.factor
        facts["levels"] = list(plan.levels)
        facts["samples_per_level"] = {level: len(m) for level, m in zip(plan.levels, plan.level_members)}
    elif plan.statistic == "anova":
        facts["balanced_design"] = bool(plan.balanced)
        facts["terms"] = dict(plan.term_safe_names)
        if plan.cells_as_n:
            facts["cells_as_n"] = True
    return facts


def fit_plan(
    y: np.ndarray,
    plan: DifferentialPlan,
    *,
    raw: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
    min_mean: Optional[float] = None,
    alpha: float = 0.05,
    replication_unit: str = "samples",
) -> List[ComparisonResult]:
    """Fit every comparison of every statistic in ``plan`` on ``y`` (features x plan.rows).

    ``raw`` is the same matrix on its linear scale (FPKM, CPM) when the
    pipeline reports linear means beside the log ones; ``eligible`` marks the
    features any further pipeline rule lets through to a test, and
    ``min_mean`` is the expression floor on ``raw``, judged per comparison.
    Returns uncorrected results: call ``finalize`` on the whole feature set.
    """
    matrix = np.asarray(y, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(plan.rows):
        raise ValueError(
            f"the matrix has {matrix.shape[1] if matrix.ndim == 2 else '?'} sample columns and the "
            f"plan {len(plan.rows)} samples"
        )
    base = _eligible_mask(eligible, matrix.shape[0])
    floor_values = raw if raw is not None else None

    def mask_for(samples: Sequence[int]) -> np.ndarray:
        """The expression floor, judged on the samples THIS comparison reads.

        One rule for every pipeline -- a feature is tested in a comparison
        when its mean (on the linear scale) over that comparison's samples
        reaches --min-mean-cpm -- so a pseudobulk result reads exactly as a
        bulk one.
        """
        if min_mean is None or floor_values is None or not len(samples):
            return base
        means = np.asarray(floor_values, dtype=float)[:, list(samples)].mean(axis=1)
        return base & (means >= float(min_mean))

    results: List[ComparisonResult] = []
    for statistic_plan in plan.statistics:
        if statistic_plan.statistic == "anova":
            mask = mask_for(range(matrix.shape[1]))
            fitted = (fit_one_way(matrix, raw, statistic_plan, mask) if statistic_plan.one_way
                      else fit_anova(matrix, raw, statistic_plan, mask))
            for comparison in statistic_plan.comparisons:
                results.append(ComparisonResult(
                    statistic="anova", folder=statistic_plan.folder, comparison=comparison,
                    columns=fitted[comparison.name],
                    facts=comparison_facts(statistic_plan, comparison, replication_unit),
                ))
            continue
        for comparison in statistic_plan.comparisons:
            mask = mask_for(comparison.samples())
            if statistic_plan.statistic == "t":
                columns = fit_two_groups(matrix, raw, comparison, mask, alpha)
            elif statistic_plan.statistic == "paired-t":
                columns = fit_paired(matrix, raw, comparison, mask, alpha)
            elif statistic_plan.statistic == "lm":
                columns = fit_slope(matrix, raw, statistic_plan, comparison, mask)
            else:  # pragma: no cover
                raise ValueError(statistic_plan.statistic)
            results.append(ComparisonResult(
                statistic=statistic_plan.statistic, folder=statistic_plan.folder,
                comparison=comparison, columns=columns,
                facts=comparison_facts(statistic_plan, comparison, replication_unit),
            ))
    return results


def concatenate(blocks: Sequence[Sequence[ComparisonResult]]) -> List[ComparisonResult]:
    """Join per-block results (same plan, consecutive feature slices) in order."""
    blocks = [list(b) for b in blocks if b]
    if not blocks:
        return []
    joined: List[ComparisonResult] = []
    for position, first in enumerate(blocks[0]):
        parts = [block[position] for block in blocks]
        columns = {
            name: np.concatenate([np.asarray(part.columns[name]) for part in parts])
            for name in first.columns
        }
        joined.append(ComparisonResult(
            statistic=first.statistic, folder=first.folder, comparison=first.comparison,
            columns=columns, facts=dict(first.facts),
        ))
    return joined


def finalize(
    results: Sequence[ComparisonResult],
    *,
    alpha_metric: str = "p",
    alpha_value: float = 0.05,
) -> List[ComparisonResult]:
    """BH per comparison over the features it tested, and significance at --alpha.

    The q-value is corrected within each comparison (each ANOVA term is its own
    comparison), never across them -- a statistic's answer does not depend on
    how many other comparisons the run happened to make.
    """
    metric = str(alpha_metric).lower()
    if metric not in ("p", "q"):
        raise ValueError(f"alpha metric must be p or q, got {alpha_metric!r}")
    for result in results:
        p = np.asarray(result.columns["p_value"], dtype=float)
        tested = np.asarray(result.columns["tested"], dtype=bool)
        q = benjamini_hochberg(np.where(tested, p, np.nan))
        result.columns["q_value"] = q
        for p_name, q_name in EXTRA_P_VALUES:
            if p_name in result.columns:
                extra = np.asarray(result.columns[p_name], dtype=float)
                result.columns[q_name] = benjamini_hochberg(np.where(tested, extra, np.nan))
        # No call of significance is written: p and q are the result. --alpha only sets the cut a later stage
        # (motif enrichment, the phospho pairing test) selects by.
        result.facts["alpha_metric"] = metric
        result.facts["alpha_value"] = float(alpha_value)
        result.facts["tested"] = int(tested.sum())
    return list(results)


def by_folder(results: Sequence[ComparisonResult]) -> Dict[str, List[ComparisonResult]]:
    """Results grouped by the statistic folder they are written to, in plan order."""
    grouped: Dict[str, List[ComparisonResult]] = {}
    for result in results:
        grouped.setdefault(result.folder, []).append(result)
    return grouped


def statistic_of_folder(results: Mapping[str, Sequence[ComparisonResult]], folder: str) -> str:
    return str(results[folder][0].statistic) if results.get(folder) else folder
