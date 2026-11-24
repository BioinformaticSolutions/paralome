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

"""ms_normalize: the numbers made ready for the core statistics.

Nothing here is a switch: a user cannot be assumed to know the differences,
so the software tests, decides and records what it decided.

1. LOG SCALE IS DETECTED, NOT ASSUMED. The header first (exporters that write
   logs say so), then the values: a negative value only exists on a log scale,
   and no log2 intensity reaches 100. A table that cannot be told (all values
   under 16 with a silent header: log10, log2 or small linear numbers alike)
   is refused with PLM-0229. A logged table is taken back to linear, so every
   table then follows the one path below.
2. EVERY SAMPLE IS SCALED to equal totals, CPM-like, whether TMT or not. The
   totals are taken over the features measured in every sample, so a sample
   is not penalised for what it missed. A sample already in line gets a
   factor of about 1.
3. A PAIRED PHOSPHO TABLE. Each sample's loading offset in each
   table is the median, over features measured everywhere, of its log2 value
   less the feature's mean; the group means are removed so treatment effects
   do not count as loading; the two tables' offsets are correlated across
   samples, and the p value is from shuffling samples within groups. Paired
   (r > 0, p < PAIRED_P): the phospho table takes the protein table's factors.
   Otherwise it is scaled on its own totals.
4. THE MISSING VALUES. A missing value is half the feature's lowest measured
   value (``half_min``). Checked on the Olink arm (23 comparisons: the assays
   measured in every sample, their lowest 10-40% made missing, each refitted
   with the model the run used): half_min kept the full data's direction in
   91% of 3,304 tests, leaving values missing in 74%, and its p value was
   closer in 71%. This code is shared by proteomics and its phospho table,
   which take the same rule.
5. WHAT IS TESTED: a feature
   measured in at least half the samples of at least one design group.

log2(x + 1) is applied to the scaled values, as for RNA, by the core
(``singlecell.stats_core``), not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paralome.qc_stats import within_group_permutation_correlation

#: Below this maximum, with a silent header, a table cannot be told apart
#: (log10, log2 or small linear numbers).
AMBIGUOUS_MAXIMUM = 16.0
#: No log2 intensity reaches this.
LINEAR_MINIMUM_MAXIMUM = 100.0
#: Fewer features measured in every sample than this, and the totals are
#: taken over every measured value instead.
MIN_COMPLETE_FEATURES = 50
#: The paired-loading test: within-group shuffles, and the p value below
#: which a positive correlation means paired.
PAIRING_PERMUTATIONS = 5000
PAIRED_P = 0.05
PAIRING_SEED = 20261024

#: Half_min alone (see 4. above); its name is in no folder or file name.


@dataclass
class ScaleDecision:
    scale: str                    # linear | log2 | log10
    reason: str


def decide_scale(values: np.ndarray, log_header: Optional[bool], header_text: str = "") -> ScaleDecision:
    """Linear, log2 or log10 -- or PLM-0229 when the table cannot be told."""
    from paralome.errors import ParalomeError

    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ParalomeError(229, "the table holds no measured value at all")
    negative = bool((finite < 0).any())
    maximum = float(finite.max())
    if log_header:
        base = "log10" if "log10" in header_text.lower() else "log2"
        if maximum >= LINEAR_MINIMUM_MAXIMUM:
            raise ParalomeError(229, f"the header says {base} but the values reach {maximum:g}, "
                                     "which no logged intensity does")
        return ScaleDecision(base, f"the header says {base}")
    if negative and maximum >= LINEAR_MINIMUM_MAXIMUM:
        raise ParalomeError(229, f"the values are negative (log-like) and reach {maximum:g} (linear-like)")
    if maximum >= LINEAR_MINIMUM_MAXIMUM:
        return ScaleDecision("linear", f"values reach {maximum:g}, which no log2 intensity does, and none is negative")
    if maximum >= AMBIGUOUS_MAXIMUM:
        return ScaleDecision("log2", f"values stay under {LINEAR_MINIMUM_MAXIMUM:g} (largest {maximum:g}); "
                                     "that is a log2 intensity range")
    raise ParalomeError(229, f"every value is under {AMBIGUOUS_MAXIMUM:g} (largest {maximum:g}) and the header "
                             "does not say whether they are logged: log10, log2 and small linear numbers all "
                             "look like this")


def to_linear(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "linear":
        return values.astype(float)
    base = 10.0 if scale == "log10" else 2.0
    with np.errstate(over="ignore"):
        return np.power(base, values.astype(float))


def scaling_factors(values: np.ndarray) -> Tuple[np.ndarray, str]:
    """Per-sample factors to equal totals, and what the totals were taken over."""
    complete = np.isfinite(values).all(axis=1)
    if complete.sum() >= MIN_COMPLETE_FEATURES:
        totals = values[complete].sum(axis=0)
        basis = f"the {int(complete.sum()):,} features measured in every sample"
    else:
        totals = np.nansum(values, axis=0)
        basis = (f"every measured value (only {int(complete.sum())} features are measured in every "
                 f"sample, fewer than {MIN_COMPLETE_FEATURES})")
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(totals > 0, totals.mean() / totals, np.nan)
    return factors, basis


def loading_offsets(values: np.ndarray) -> np.ndarray:
    """Each sample's log2 offset: median over complete features of (log2 value - feature mean)."""
    complete = np.isfinite(values).all(axis=1)
    logged = np.log2(values[complete] + 1.0)
    if not logged.shape[0]:
        return np.full(values.shape[1], np.nan)
    return np.median(logged - logged.mean(axis=1, keepdims=True), axis=0)


def design_groups(rows: Sequence[Dict[str, str]], skip: Sequence[str] = ("column", "sample_id")) -> List[str]:
    """Each sample's design group: its values of every metadata column that is
    shared by more than one sample (a column unique to each sample -- an ID, a
    replicate number -- names no group)."""
    columns = [c for c in (rows[0] if rows else {}) if c not in skip]
    shared = [c for c in columns if len({row.get(c, "") for row in rows}) < len(rows)]
    return ["|".join(f"{c}={row.get(c, '')}" for c in shared) for row in rows]


@dataclass
class PairingTest:
    r: Optional[float]
    p: Optional[float]
    paired: bool
    reason: str
    protein_offsets: List[float] = field(default_factory=list)
    phospho_offsets: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"r": self.r, "p": self.p, "paired": self.paired, "reason": self.reason,
                "protein_offsets_log2": self.protein_offsets, "phospho_offsets_log2": self.phospho_offsets,
                "permutations": PAIRING_PERMUTATIONS}


def pairing_test(protein: np.ndarray, phospho: np.ndarray, groups: Sequence[str],
                 alpha: float = PAIRED_P) -> PairingTest:
    """Do the two tables' sample loadings move together? (columns = the same samples)."""
    x = loading_offsets(protein)
    y = loading_offsets(phospho)
    record = dict(protein_offsets=[float(v) for v in x], phospho_offsets=[float(v) for v in y])
    tested = within_group_permutation_correlation(x, y, groups, permutations=PAIRING_PERMUTATIONS,
                                                  seed=PAIRING_SEED)
    if tested is None:
        return PairingTest(None, None, False, "no sample varies within its group, so loading cannot be "
                           "compared; the tables are treated as independent", **record)
    r, p = tested
    paired = r > 0 and p < alpha                 # --alpha's cutoff (default p < 0.05)
    reason = (f"loading offsets correlate across samples (r = {r:.2f}, p = {p:.4f}): paired, so the "
              "phospho table takes the protein table's factors" if paired else
              f"loading offsets do not correlate (r = {r:.2f}, p = {p:.4f}): independent, so the phospho "
              "table is scaled on its own totals")
    return PairingTest(r, p, paired, reason, **record)


def half_minimum_fill(values: np.ndarray) -> np.ndarray:
    """Each missing value set to half its feature's lowest measured value (rows with none stay missing)."""
    filled = values.copy()
    with np.errstate(invalid="ignore"):
        floor = np.nanmin(np.where(np.isfinite(values), values, np.inf), axis=1)
    floor = np.where(np.isfinite(floor), floor / 2.0, np.nan)
    missing = ~np.isfinite(filled)
    filled[missing] = np.broadcast_to(floor[:, None], filled.shape)[missing]
    return filled


def testable(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    """Measured in at least half the samples of at least one design group."""
    groups = np.asarray(list(groups))
    measured = np.isfinite(values)
    keep = np.zeros(values.shape[0], dtype=bool)
    for g in np.unique(groups):
        idx = groups == g
        keep |= measured[:, idx].sum(axis=1) >= math.ceil(idx.sum() / 2)
    return keep
