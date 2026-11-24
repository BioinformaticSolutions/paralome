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

"""Statistics primitives shared by every modality.

Every pipeline corrects p-values through this one module, so the arithmetic
cannot drift apart between them. In particular Benjamini-Hochberg is computed
over the tested features only: a pipeline that substituted p = 1.0 for every
untested feature and corrected across the whole array would inflate the
denominator and hand back a confident-looking q = 1.0 for features that were
never tested.

This module is modality-free on purpose. Bulk ATAC, snATAC, bulk RNA, scRNA and
motif enrichment all correct p-values, so the correction cannot live in any of
their packages without one of them importing another's internals.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from statsmodels.stats.multitest import multipletests

from paralome.errors import ParalomeError


def benjamini_hochberg(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """BH-FDR over the TESTED entries only; non-finite in, non-finite out.

    Correction is applied across the finite subset, so ``n`` is the number of
    features actually tested rather than the length of the array. Entries that
    were never tested stay NaN and are rendered as MISSING_VALUE downstream,
    which keeps "never tested" distinguishable from "tested, not significant".
    """
    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan)
    mask = np.isfinite(array)
    if mask.any():
        output[mask] = multipletests(array[mask], method="fdr_bh")[1]
    return output



def signed_times_fold_change(log2_delta: float) -> float:
    """A log2 difference expressed as SIGNED "times".

    It is NOT the division of logs: it is 2^(log2-log2), which yields "times",
    where FC 2 = 2x.

        +2.0  ->  two-fold UP        -2.0  ->  two-fold DOWN
         1.0  ->  unchanged

    WHY 2**delta AND NOT A RATIO OF RAW MEANS. Every Paralome test -- bulk and
    single-cell -- is computed on log2 values, so the quantity the p-value
    describes is a difference of MEAN LOGS. Reporting `mean(raw_treatment) /
    mean(raw_control)` instead answers a different question and can disagree
    violently with its own p-value (one outlier in four samples, or a near-zero
    control mean, moves it by an order of magnitude). 2**delta is the ratio of
    GEOMETRIC means, which is what was tested. Working from
    log2(x + pseudocount), the operand is always finite, so no column ever
    mixes ratios with differences.

    THE RECIPROCAL FORM is the convention: a halving reads -2.0, not 0.5, so
    that magnitudes are comparable up and down and `abs()` is a symmetric
    threshold. The scale is therefore discontinuous -- no value lies strictly
    between -1 and +1 -- and 1.0, not 0.0, means "no change", because under
    "times" a fold change of 0 would say the gene vanished.

    WHY "TIMES" AT ALL, given that the models work in log2 throughout. A log
    fold change of 9.45 or 0.00029 is hard for a person to read; -104.3x is
    understood instinctively, without explanation.

    The audience for this column is a person reading a table, not the model.
    log2_fold_change is retained unchanged in the machine/ tables and is what
    motif_partners and the peak classifier consume, so nothing downstream reads
    the human-facing number.

    THE PSEUDOCOUNT DOES NOT CANCEL, and that is accepted rather than overlooked.
    The stored matrix is log2(x + pseudocount), so an exact doubling of 10 -> 20
    reports 1.909, not 2.000; at 500 -> 1000 it is 1.998. The compression is
    worst for features near the pseudocount, which is also where the test is
    least trustworthy. Removing it would make the reported effect disagree with
    its own p-value. Every way of calculating a fold change is wrong somewhere;
    this tradeoff is the accepted one.
    """
    if not math.isfinite(log2_delta):
        return float("nan")
    ratio = 2.0 ** log2_delta
    if not math.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return ratio if ratio >= 1.0 else -1.0 / ratio


# ---------------------------------------------------------------------------
# Effect size and power
#
# Power is calculated for both statistical strategies so their effect can be
# shown side by side. The power calculation from the naive cell t-test is
# upwardly biased by the pseudo-N, and that is the point: a reviewer can see the
# effect on power of doing it correctly and doing it pseudo-optimistically.
#
# WHY HERE. The power function existed TWICE -- atac/statistics.py and
# rna/differential.py -- with identical arithmetic and different spelling
# (`math.nan` vs `float("nan")`). The RNA copy's docstring even said "matching
# paralome.atac.statistics", which is an acknowledged fork. Benjamini-Hochberg
# was in the same state and needed a fix when two of its four copies turned
# out to be doing genuinely different arithmetic. Adding single-cell as a THIRD
# consumer without consolidating first would have been the same mistake with
# the same ending.
# ---------------------------------------------------------------------------


def cohens_d_from_moments(
    mean_control: float, var_control: float, n_control: float,
    mean_treatment: float, var_treatment: float, n_treatment: float,
) -> float:
    """Standardised mean difference from moments already computed.

    Takes moments rather than arrays because every caller has already computed
    them to run its own test -- per sample for t, per cell for t_cell.
    Recomputing from the raw data would
    risk using a different variance than the test used, which is precisely the
    disagreement this whole exercise is meant to expose rather than commit.

    Positive means higher in treatment, matching log2_fold_change everywhere.
    """
    if not all(math.isfinite(v) for v in
               (mean_control, var_control, mean_treatment, var_treatment)):
        return float("nan")
    if n_control < 2 or n_treatment < 2:
        return float("nan")
    pooled = (
        (n_control - 1) * var_control + (n_treatment - 1) * var_treatment
    ) / (n_control + n_treatment - 2)
    if pooled <= 0:
        return 0.0
    return float((mean_treatment - mean_control) / math.sqrt(pooled))


def pooled_sd_from_moments(
    var_control: float, n_control: float, var_treatment: float, n_treatment: float
) -> float:
    """The denominator cohens_d_from_moments divides by.

    Reported alongside the effect size because it is what converts a Cohen's d
    into a fold change, and it is PER FEATURE -- a stable housekeeping gene and
    a sparse one have wildly different SDs, so a single study-wide "detectable
    fold change" would be true only of the median gene.
    """
    if not all(math.isfinite(v) for v in (var_control, var_treatment)):
        return float("nan")
    if n_control < 2 or n_treatment < 2:
        return float("nan")
    pooled = (
        (n_control - 1) * var_control + (n_treatment - 1) * var_treatment
    ) / (n_control + n_treatment - 2)
    return float(math.sqrt(pooled)) if pooled > 0 else 0.0


def minimum_detectable_d(
    n_control: float, n_treatment: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """The smallest Cohen's d this DESIGN could detect at `power`.

    Depends only on n, alpha and the target power -- never on what was observed.
    That is the whole difference from post-hoc power, which is a deterministic
    function of the p-value and tells a reader nothing the p-value did not
    (Hoenig & Heisey 2001, "The Abuse of Power"). A reviewer asking "was this
    adequately powered" is asking this question, not that one.

    One number per design, which is why the per-feature detectable FOLD CHANGE
    is this multiplied by each feature's own pooled SD.
    """
    if n_control < 2 or n_treatment < 2:
        return float("nan")
    from scipy import stats as _st

    df = n_control + n_treatment - 2
    t_alpha = float(_st.t.ppf(1 - alpha / 2.0, df))
    t_beta = float(_st.t.ppf(power, df))
    return (t_alpha + t_beta) * math.sqrt(1.0 / n_control + 1.0 / n_treatment)


#: log2 fold change above which 2**x stops fitting in a float64. Past roughly
#: 1024 the exponentiation overflows to inf, and inf is not valid JSON -- a
#: report carrying it either fails to serialise or writes a bare `Infinity` that
#: strict parsers reject.
MAX_LOG2_FOLD_CHANGE = 1000.0


def fold_change_from_log2(log2_value: float) -> Optional[float]:
    """2**x as a fold change, or None when x is too large to mean anything.

    It lives HERE rather than beside its caller so that fold change has one
    implementation, not a private helper that looks too small to share.

    A detectable effect of 1200 log2 units is not a number to show a reader --
    it says the feature could not be tested, which None says better and without
    overflowing. A near-singular anova_cell fit returns an enormous standard
    error and the MDE arithmetic carries it faithfully all the way to the
    report.
    """
    value = float(log2_value)
    if not math.isfinite(value) or abs(value) > MAX_LOG2_FOLD_CHANGE:
        return None
    return float(2 ** value)


def sd_from_standard_error(
    standard_error: float, n_control: float, n_treatment: float
) -> float:
    """The per-feature SD implied by a model's own standard error.

    ``pooled_sd_from_moments`` is the route in for a test that
    computed two group variances. A model that fits a coefficient and reports
    its standard error -- anova_cell's random-intercept fit -- has no pair of
    group variances to hand back, so the SD is recovered from the SE instead,
    by inverting the relation the SE was built from:

        SE**2 = sd**2 * (1/n_control + 1/n_treatment)

    THE n HERE IS SAMPLES, NOT CELLS. The fit uses every cell but prices its
    uncertainty in samples through the random intercept, so inverting with the
    sample n returns the sample-level SD its inference assumed, and
    ``minimum_detectable_d(n_c, n_t) * sd`` collapses to the textbook SE-based
    minimum detectable effect, ``(t_alpha + t_beta) * SE``.
    """
    if not math.isfinite(standard_error) or standard_error < 0:
        return float("nan")
    if n_control < 2 or n_treatment < 2:
        return float("nan")
    scale = math.sqrt(1.0 / n_control + 1.0 / n_treatment)
    if scale <= 0:
        return float("nan")
    return float(standard_error / scale)


def observed_power(
    effect_size: float, n_control: float, n_treatment: float, alpha: float = 0.05
) -> float:
    """Power to detect the OBSERVED effect. Post-hoc, and labelled as such.

    This is reported for both replication units deliberately, so a
    reviewer can see that the pseudo-N inflation was measured rather than
    overlooked. It is named `observed_power` and not `power` for the same
    reason: on its own it is a restatement of the p-value, and the column name
    should say so before anyone builds an argument on it.
    """
    if not math.isfinite(effect_size) or n_control < 2 or n_treatment < 2:
        return float("nan")

    # scipy's non-central t directly, NOT statsmodels' TTestIndPower.
    #
    # statsmodels returns nan for large effect sizes at small n -- measured:
    # d=5.0 gives 0.99992 and d=50 gives 1.0, but d=13.67 at n=4 gives nan. A
    # tight 2-fold effect across four donors lands squarely in that hole, so
    # the column would have been empty for exactly the genes a reviewer looks
    # at first. student_power below uses this same formula, so the two cannot
    # disagree.
    from scipy import stats as _st

    df = n_control + n_treatment - 2
    ncp = abs(effect_size) * math.sqrt(
        (n_control * n_treatment) / (n_control + n_treatment)
    )
    critical = float(_st.t.ppf(1 - alpha / 2.0, df))
    value = float(_st.nct.sf(critical, df, ncp) + _st.nct.cdf(-critical, df, ncp))

    # scipy's non-central t returns nan for large non-centrality at small df --
    # measured: fine at ncp=7.1 and ncp=70.7, nan at ncp=19.3 with df=6. That is
    # a numerical hole, not a real discontinuity, and a tight 2-fold effect
    # across four donors lands in it. The normal approximation is at its most
    # accurate in exactly that regime (the effect is many standard errors from
    # the critical value), so it fills the hole rather than papering over it.
    if not math.isfinite(value):
        value = float(_st.norm.sf(critical - ncp) + _st.norm.cdf(-critical - ncp))
    if not math.isfinite(value):
        return float("nan")
    return min(1.0, max(0.0, value))


#: Values this close together (relative to their size) are one
#: value. The matrix is written to ten significant digits, so a real
#: difference is at least a hundred times larger than this.
_CONSTANT_TOLERANCE = 1e-12


def _is_constant(values: np.ndarray) -> bool:
    """Whether every value is the same number, so there is no variance to test.

    The mean of n copies of log2(0 + pseudocount) is not always that
    number again: it can differ in the last bit, and the variance computed
    from it is then about 1e-32 instead of 0. With the default pseudocount the
    value is log2(1) = 0 and the arithmetic is exact, which is the only reason
    an unexpressed gene read as untested. With any other pseudocount it got a
    statistic made of rounding error, with p-values down to 1e-6.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False
    scale = max(1.0, float(np.max(np.abs(array))))
    return float(np.ptp(array)) <= _CONSTANT_TOLERANCE * scale


def _ordered_levels(rows: Sequence[Mapping[str, str]], field: str) -> List[str]:
    levels: List[str] = []
    for row in rows:
        value = str(row.get(field) or "").strip()
        if value and value not in levels:
            levels.append(value)
    return levels


def _dedupe_factor_prefix(component: str, remainder: str) -> str:
    """Drop ``component`` from the front of a generated name when
    ``remainder`` already restates it (e.g. component ``"DRUG"``, remainder
    starting with the level value ``"DRUG20"``) -- otherwise the two stutter
    together (``"DRUG_DRUG20_vs_Control"``) instead of reading cleanly
    (``"DRUG20_vs_Control"``).

    Descriptive level values are kept (0/1-coded columns would make names
    *less* readable -- "DRUG_1_vs_0" says less than "DRUG20_vs_Control"), and
    the factor name is just not re-stated in front of them when it's already
    implied. A factor whose level names don't already restate it (e.g.
    factor "Diet", level "HighFat") still gets the factor prefix, since
    "HighFat_vs_LowFat" alone wouldn't say what varies.
    """
    component_key = "".join(ch for ch in component if ch.isalnum()).lower()
    remainder_key = "".join(ch for ch in remainder if ch.isalnum()).lower()
    if component_key and remainder_key.startswith(component_key):
        return remainder
    return f"{component}_{remainder}"


#: The --factor help, the same words on every command that takes it.
FACTOR_HELP = (
    "A sample-sheet column and its control level, COLUMN:CONTROL (e.g. Diet:CD). Repeatable, once "
    "per factor. Every other level of the column is compared with the control level: t and "
    "paired-t each level against it (with more factors, within each combination of the "
    "others); anova with one factor is the one-way ANOVA, with two or more the factorial ANOVA "
    "with every interaction, and its contrasts are each level against the control. Levels may "
    "be words or numbers (Sex:F, Day:1). The control level is required: it gives every result "
    "its direction. With --stratify-by, several control levels may be listed "
    "(Treatment:2mo,dcKO_CTL): each stratum takes the one it holds, and a stratum holding two "
    "is refused"
)


#: One wording of --stratify-by for every pipeline.
STRATIFY_HELP = (
    "Column(s) to analyse separately per value, e.g. cell_type, or Experiment,cell_type for every "
    "combination. A cell may list several values (Exp1,Exp1b): the sample is in each. A sample "
    "with no value is in no stratum"
)


#: --block-by, the repeated-measures term of anova and lm.
BLOCK_HELP = (
    "Sample-sheet column naming the block each sample belongs to (e.g. subject, for repeated "
    "measures). anova and lm add it to the model as a term that is fitted but not tested, so "
    "each factor and covariate is tested within blocks. A factor that is constant within every "
    "block cannot be tested this way and is refused. t and paired-t do not use it"
)


STATISTIC_HELP = (
    "Required: the differential statistics, comma-separated, any combination of t, paired-t, lm "
    "and anova, e.g. --statistic t or --statistic t,anova. There is no default. Each writes its own folder of results, named by the "
    "flag (t, paired_t, lm, anova). t and paired-t compare each level of a --factor with its "
    "control level (paired-t pairs the samples by --pair-by); lm regresses on --covariate or "
    "--covariate-feature, and within each level of a --factor if one is given; anova fits one "
    "--factor (one-way) or two or more (the factorial, with every interaction). Every pipeline "
    "fits the same models"
)


class FactorSpecError(ParalomeError, ValueError):
    """PLM-0113: a --factor without its control level, or with one the column lacks.

    A ValueError too, so a caller that validates a request early catches it as
    it always did (comparisons.DesignError is built the same way).
    """

    def __init__(self, detail: str = "", **context: object) -> None:
        super().__init__(113, detail, **context)


class TwoControlsError(FactorSpecError):
    """PLM-0113, two of a --factor's listed control levels in one stratum -- one would be a treatment.

    Refused for the whole run, not skipped as a stratum the design cannot test: a listed control is never
    used as a treatment."""


def parse_factor(spec: str) -> Tuple[str, str]:
    """``Diet:CD`` -> ``("Diet", "CD")``: a --factor's column and its control level.

    A factor without a designated control is an error: all tests are
    directional, because of the contrasts. The control is said on the command
    line, so nothing about a result depends on how the sheet is sorted.
    """
    text = str(spec).strip()
    column, colon, baseline = text.partition(":")
    column, baseline = column.strip(), baseline.strip()
    if not colon or not column or not baseline:
        name = column or "Diet"
        raise FactorSpecError(
            f"--factor {text!r} does not say its control level. Write the column, a colon "
            f"and the level every other level is compared with: --factor {name}:<control level> "
            "(for example --factor Diet:CD). Every comparison is made against that level, so "
            "it has to be named"
        )
    return column, baseline


def control_choices(control: Optional[str]) -> List[str]:
    """``"2mo,dcKO_CTL"`` -> ``["2mo", "dcKO_CTL"]``: the control levels a --factor lists."""
    return [part.strip() for part in str(control or "").split(",") if part.strip()]


def factor_columns(specs: Sequence[str]) -> List[str]:
    """The sample-sheet columns a list of --factor values names, in order."""
    return [parse_factor(spec)[0] for spec in specs]


def level_numbers(levels: Sequence[str]) -> Optional[List[float]]:
    """Each level as a number, or None when any level is not one."""
    numbers: List[float] = []
    for level in levels:
        try:
            value = float(str(level).strip())
        except ValueError:
            return None
        if not np.isfinite(value):
            return None
        numbers.append(value)
    return numbers


def baseline_first_levels(rows: Sequence[Mapping[str, str]], factor: str, baseline: str) -> List[str]:
    """A factor's levels, its control level first, as every comparison reads them.

    The others follow by value when every level is a number (day 1, 4, 8, 9),
    else in the order the sample sheet first gives them. Only the control level
    decides a direction; the order of the rest decides nothing but the order
    results are listed in. A control level the column does not hold is refused
    with the levels it does.
    """
    levels = _ordered_levels(rows, factor)
    # Controls are transient, so a list of controls works, presuming a control in the list is not used as a
    # treatment in a similar design -- --factor Treatment:2mo,dcKO_CTL,OE_CTL: in the samples in scope (a
    # stratum), the one listed level present is the control. None present, or two (one would be a
    # treatment), is refused.
    choices = control_choices(baseline)
    present = [choice for choice in choices if choice in levels]
    if len(choices) > 1 and len(present) == 1:
        baseline = present[0]
    elif len(choices) > 1:
        raise (TwoControlsError if present else FactorSpecError)(
            f"--factor {factor}:{','.join(choices)}: "
            + (f"none of the listed control levels is among the samples in scope ({factor} has: "
               f"{', '.join(levels) or 'none'})" if not present else
               f"{' and '.join(present)} are both among the samples in scope, so one of them would be a "
               "treatment. Each stratum must hold exactly one of the listed control levels")
        )
    if baseline not in levels:
        raise FactorSpecError(
            f"--factor {factor}:{baseline}: {factor} has no level {baseline!r} among the "
            f"samples in scope. Its levels are: {', '.join(levels) or 'none'}"
        )
    others = [level for level in levels if level != baseline]
    numbers = level_numbers(others)
    if numbers is not None:
        others = [level for _value, level in sorted(zip(numbers, others))]
    return [baseline] + others


def design_matrix(
    rows: Sequence[Mapping[str, str]],
    factors: Sequence[str],
    levels_by_factor: Mapping[str, Sequence[str]],
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """Treatment-coded design: Intercept, a 0/1 column per non-control level, and their products.

    The interaction is ALWAYS in a two-factor design. Any number of levels per factor, each factor's control
    level the one --factor names (``levels_by_factor[f][0]``). A level's column
    is ``factor[level_vs_control]``; an interaction column
    ``A:B[a_level:b_level]``. Returns the design, the column names, and the
    columns of each TERM -- ``A``, ``B`` and ``A:B`` -- which is what an F-test
    tests: a factor of k levels is k - 1 columns and one F with k - 1 df.

    Any number of factors, the full factorial.  Every interaction of every order is in the model: the terms are
    the main effects, then every pair, every triple ... up to all the factors
    together (``A:B:C[a:b:c]``), each in the order the --factor options gave
    the factors. With two factors this is exactly the two-way design it was.
    """
    cols = [np.ones(len(rows))]
    names = ["Intercept"]
    terms: Dict[str, List[int]] = {}
    indicator: Dict[Tuple[str, str], np.ndarray] = {}
    for factor in factors:
        levels = list(levels_by_factor[factor])
        baseline = levels[0]
        terms[factor] = []
        for level in levels[1:]:
            vector = np.asarray(
                [1.0 if str(row.get(factor) or "").strip() == level else 0.0 for row in rows]
            )
            indicator[(factor, level)] = vector
            terms[factor].append(len(cols))
            cols.append(vector)
            names.append(f"{factor}[{level}_vs_{baseline}]")
    for order in range(2, len(factors) + 1):
        for members in itertools.combinations(factors, order):
            interaction = ":".join(members)
            terms[interaction] = []
            for combination in itertools.product(*(list(levels_by_factor[f])[1:] for f in members)):
                column = np.ones(len(rows))
                for factor, level in zip(members, combination):
                    column = column * indicator[(factor, level)]
                terms[interaction].append(len(cols))
                cols.append(column)
                names.append(f"{interaction}[{':'.join(combination)}]")
    return np.column_stack(cols), names, terms


def is_balanced(rows: Sequence[Mapping[str, str]], factors: Sequence[str]) -> bool:
    """Equal n in every cell of the factorial?

    Matters because Type I, II and III sums of squares all coincide on a
    balanced design and diverge on an unbalanced one. Saying which was used is
    part of reporting the result. Counted on the samples' levels,
    so a factor of any number of levels is counted the same way.
    """
    cells: Dict[Tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(str(row.get(factor) or "").strip() for factor in factors)
        cells[key] = cells.get(key, 0) + 1
    return len(set(cells.values())) <= 1


def model_error_df(design: np.ndarray, n_samples: int) -> int:
    """Residual degrees of freedom: N - rank(X).

    Exposed so ``rna/parallel.py`` can report it in the contrast header without
    importing numpy -- that module has no numpy import, and adding one just to
    call matrix_rank is how a NameError reaches a SLURM job. (It nearly did:
    the unimported-names test caught `np` in a function body here.)
    """
    return int(n_samples) - int(np.linalg.matrix_rank(design))


def _residual_sum_of_squares(y: np.ndarray, design: np.ndarray) -> Tuple[float, int]:
    """RSS and the model's rank, for one nested fit."""
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(np.dot(residual, residual)), int(np.linalg.matrix_rank(design))


def term_reference_columns(term: str, term_columns: Mapping[str, Sequence[int]], width: int) -> List[int]:
    """The model a term is tested within (Type II): every term but those that contain it.

    At any number of factors. A term is tested against the model
    holding every term that does not contain it -- for ``A`` with ``B`` and
    ``C``, the model without ``A:B``, ``A:C`` and ``A:B:C``; for the highest
    interaction, the full model. With two factors: each main effect within
    ``A + B``, the interaction within the full model, as before.
    """
    members = set(term.split(":"))
    containing = {
        i for name, cols in term_columns.items()
        if set(name.split(":")) > members for i in cols
    }
    return [i for i in range(width) if i not in containing]



def _anova_terms(
    y: np.ndarray,
    design: np.ndarray,
    term_columns: Mapping[str, Sequence[int]],
) -> Dict[str, Tuple[float, float]]:
    """A factorial (or one-way) ANOVA's F-tests for one feature: term -> (F, p).

    A t-test produces one result. A two-way ANOVA produces three: the main
    effect of each factor and their interaction, each tested as the whole term
    -- all of its columns at once, k - 1 df for a factor of k levels.

    WHY THIS IS NOT A TEST OF COEFFICIENTS. Under treatment coding with an
    interaction in the model a coefficient is not a main effect: it is that
    level's effect at the CONTROL level of the other factor. A main effect is
    the term's whole contribution across the other factor's levels, which is
    what the sums-of-squares comparison below measures.

    TYPE II, ALWAYS. Each main effect is tested by comparing two models that
    BOTH exclude the interaction: SS(A | B) = RSS(B) - RSS(A, B). The
    interaction is tested against both main effects. At more factors every
    term is tested within the model of every term that does not contain it
    (``term_reference_columns``). On a balanced design Type
    II coincides with I and III. NOT TYPE III: with 0/1 coding the interaction
    columns still carry the main effects, so dropping a main effect while
    keeping them tests nothing meaningful (checked against statsmodels'
    ``anova_lm(typ=2)``).

    The error term is the FULL model's for every test: df = N - rank(X).
    """
    width = design.shape[1]
    full_rss, full_rank = _residual_sum_of_squares(y, design)
    df_error = len(y) - full_rank
    if df_error <= 0 or _is_constant(y):
        return {name: (float("nan"), float("nan")) for name in term_columns}
    mse = full_rss / df_error
    out: Dict[str, Tuple[float, float]] = {}
    for name, columns in term_columns.items():
        reference_columns = term_reference_columns(name, term_columns, width)
        reduced_columns = [i for i in reference_columns if i not in set(columns)]
        reference_rss, reference_rank = _residual_sum_of_squares(y, design[:, reference_columns])
        reduced_rss, reduced_rank = _residual_sum_of_squares(y, design[:, reduced_columns])
        df_term = max(reference_rank - reduced_rank, 1)
        extra = max(reduced_rss - reference_rss, 0.0)
        if mse <= 0:
            out[name] = (float("nan"), float("nan"))
            continue
        f_value = (extra / df_term) / mse
        from scipy import stats as _st

        out[name] = (float(f_value), float(_st.f.sf(f_value, df_term, df_error)))
    return out


# ---------------------------------------------------------------------------
# What a run asked for: WHICH statistics, and on WHAT
#
# Every pipeline that produces differential statistics takes the same
# --statistic, from the same list, parsed here.
#
# The grammar is `<statistic>[:<data level>]`, comma between them: the colon
# belongs to the statistic it follows. Pseudobulk is a data action, not a
# statistic, so it does not belong in the comma list beside the tests.
#
#     --statistic t                       one statistic (required; no default)
#     --statistic t,anova                 two statistics, two sets of results
#     --statistic t,paired-t,lm,anova     every statistic, side by side
#     --statistic t:cell,anova:cell       the cells as N, for each
#     --statistic anova,anova:cell        the ANOVA on the pseudobulk samples and on the cells
#
# ANY COMBINATION. Running multiple statistics tests what changing the
# statistic does to the answer, so each reality needs its own end result: each
# statistic named is computed on its own and writes its own folder of results,
# named by the flag that asked for it (t, paired_t, lm, anova; t_cell ... with
# :cell), all the way through motif enrichment and Evidome.
# ---------------------------------------------------------------------------

#: Statistics that compare two groups of samples. t and paired-t are two-group
#: tests; lm regresses on a covariate and is listed here because, like them, it
#: is fitted per comparison rather than as a factorial model.
PAIRWISE_STATISTICS = ("t", "paired-t", "lm")

#: The factorial model. An ANOVA is used when there ARE interactions (without them a t-test would do), so the
#: interaction is always in the model, and the ANOVA needs both factors.
FACTORIAL_STATISTICS = ("anova",)

ALL_STATISTICS = PAIRWISE_STATISTICS + FACTORIAL_STATISTICS

#: What the statistics run on.
#:
#: Pseudobulk IS the normal case, so it is the
#: default and has no term: bare `--statistic t` is the pseudobulk test.
#: `:cell` is the one trigger there is -- "they are very different theoretical
#: calculations and they need to be maintained apart so no one ever gets
#: confused at which column is which": each level writes its own folders
#: (t/, t_cell/). `:cell` applies to the statistic it follows, so
#: one run may hold both levels, each in its own folders.
DATA_LEVEL_PSEUDOBULK = "pseudobulk"
DATA_LEVEL_CELL = "cell"

#: The levels a user may name. Pseudobulk is not among them -- it is what you
#: get by saying nothing.
DATA_LEVELS = (DATA_LEVEL_CELL,)


def statistic_folder(statistic: str, data_level: Optional[str] = None) -> str:
    """The name a statistic's results are filed under: the flag that asked for it.. ``paired-t`` becomes ``paired_t`` only because a hyphen is not kept
    in file names; ``:cell`` becomes the ``_cell`` suffix. Pseudobulk has no
    term in the flag, so it has none here either and its folders read exactly
    like bulk's.
    """
    token = str(statistic).strip().lower().replace("-", "_")
    return f"{token}_cell" if data_level == DATA_LEVEL_CELL else token


class StatisticRequest(NamedTuple):
    """One parsed --statistic: the statistics asked for, in the order given,
    and the data level (None where the pipeline has no choice).

    Each statistic has its own level. ``levels`` is empty when every
    statistic is at ``data_level`` (the common case, and every request built
    in code); a request mixing levels gives each statistic's level there, in
    the order of ``statistics``, and ``data_level`` is None. ``by_level()``
    splits it into requests of one level each, which is how it is computed.
    """

    statistics: Tuple[str, ...]
    data_level: Optional[str] = None
    levels: Tuple[Optional[str], ...] = ()

    @property
    def tokens(self) -> Tuple[str, ...]:
        """Every statistic this request runs, in the order asked."""
        return tuple(self.statistics)

    @property
    def pairs(self) -> Tuple[Tuple[str, Optional[str]], ...]:
        """(statistic, data level) of each statistic, in the order asked."""
        levels = self.levels or (self.data_level,) * len(self.statistics)
        return tuple(zip(self.statistics, levels))

    @property
    def mixed(self) -> bool:
        return len({level for _, level in self.pairs}) > 1

    @property
    def has_cell(self) -> bool:
        """Some statistic here runs on the cells."""
        return any(level == DATA_LEVEL_CELL for _, level in self.pairs)

    def by_level(self) -> Tuple["StatisticRequest", ...]:
        """One request per data level, each with its statistics in the order asked."""
        order: List[Optional[str]] = []
        for _, level in self.pairs:
            if level not in order:
                order.append(level)
        return tuple(StatisticRequest(tuple(s for s, l in self.pairs if l == level), level) for level in order)

    @classmethod
    def of_pairs(cls, pairs: Sequence[Tuple[str, Optional[str]]]) -> "StatisticRequest":
        """The request for these (statistic, data level) pairs: one level, or each its own."""
        pairs = tuple(pairs)
        levels = tuple(level for _, level in pairs)
        statistics = tuple(statistic for statistic, _ in pairs)
        if len(set(levels)) <= 1:
            return cls(statistics, levels[0] if levels else None)
        return cls(statistics, None, levels)

    def level_records(self) -> object:
        """The data level as a checkpoint records it: the one level, or each statistic's."""
        return self.data_level if not self.mixed else {statistic_folder(s, l): l for s, l in self.pairs}

    @property
    def anova(self) -> bool:
        return "anova" in self.statistics

    @property
    def pairwise(self) -> Tuple[str, ...]:
        return tuple(s for s in self.statistics if s in PAIRWISE_STATISTICS)

    def names_the(self, statistic: str) -> bool:
        return statistic in self.statistics

    def folders(self) -> Tuple[str, ...]:
        return tuple(statistic_folder(s, level) for s, level in self.pairs)


# ---------------------------------------------------------------------------
# paired-t and lm
#
# Every statistic is available for every differential analysis in every
# pipeline. Written HERE, once, for the same reason the factorial fit is:
# pipelines calling one function cannot drift apart, and a `slope` column means
# the same quantity whichever of them produced it.
# ---------------------------------------------------------------------------


def paired_t_from_pairs(differences: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """One-sample t on the WITHIN-PAIR differences: ``(t, df, p)``.

    ``differences`` is features x pairs, each column one pair's
    treatment-minus-control log2 CPM. The pairing is the whole point: a
    subject's own baseline removes every between-subject difference from the
    error term, so the test is on the mean difference against zero with
    ``df = pairs - 1`` -- NOT the pooled two-sample df, which is what makes
    this a different answer from ``t`` rather than a relabelling of it.

    Sign convention is the pipeline's throughout: treatment minus control.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2:
        raise ValueError("differences must be features x pairs")
    n_pairs = values.shape[1]
    if n_pairs < 2:
        raise ValueError(
            f"paired-t needs at least two pairs; this comparison has {n_pairs}. "
            "A single pair has no within-pair variance and so no df."
        )
    # scipy.stats is imported inside the function, as everywhere in this module.
    from scipy import stats as _st

    df = float(n_pairs - 1)
    mean = values.mean(axis=1)
    # Unbiased (n-1) variance, matching every other variance in this module.
    variance = values.var(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.sqrt(variance / n_pairs)
        t_stat = np.where(standard_error > 0, mean / standard_error, np.nan)
    p_value = np.full(t_stat.shape, np.nan)
    finite = np.isfinite(t_stat)
    if finite.any():
        p_value[finite] = 2.0 * _st.t.sf(np.abs(t_stat[finite]), df)
    return t_stat, df, p_value


def resolve_pairs(
    rows: Sequence[Mapping[str, str]],
    labels: Mapping[str, str],
    pair_by: str,
    control: str,
    treatment: str,
) -> Tuple[List[str], List[int], List[int]]:
    """Which samples pair, by the ``--pair-by`` column: ``(pair ids, control index, treatment index)``.

    Indices are positions in ``rows``. A pair must have exactly one sample in
    each arm; a pair-by value with two samples in one arm, or with only one
    arm present, is REFUSED BY NAME rather than dropped, because a paired test
    that silently discards half its pairs reports a df the design does not
    have.
    """
    if pair_by not in (rows[0] if rows else {}):
        raise ValueError(
            f"--pair-by {pair_by!r}: no such column. Available: "
            f"{', '.join(sorted(rows[0])) if rows else 'none'}"
        )
    by_pair: Dict[str, Dict[str, List[int]]] = {}
    for index, row in enumerate(rows):
        arm = labels.get(str(row.get("sample_id")))
        if arm not in (control, treatment):
            continue
        key = str(row.get(pair_by) or "").strip()
        if not key:
            raise ValueError(
                f"Sample {row.get('sample_id')!r} has no value in --pair-by "
                f"column {pair_by!r}; every tested sample must say which pair "
                f"it belongs to."
            )
        by_pair.setdefault(key, {control: [], treatment: []})[arm].append(index)

    pair_ids: List[str] = []
    control_index: List[int] = []
    treatment_index: List[int] = []
    broken: List[str] = []
    for key in sorted(by_pair):
        arms = by_pair[key]
        if len(arms[control]) == 1 and len(arms[treatment]) == 1:
            pair_ids.append(key)
            control_index.append(arms[control][0])
            treatment_index.append(arms[treatment][0])
        else:
            broken.append(
                f"{key} ({len(arms[control])} {control}, {len(arms[treatment])} {treatment})"
            )
    if broken:
        raise ValueError(
            f"--pair-by {pair_by!r}: these values are not one-to-one between "
            f"the two arms: {'; '.join(broken)}. A paired test needs exactly "
            f"one {control} and one {treatment} sample per pair."
        )
    if not pair_ids:
        raise ValueError(
            f"--pair-by {pair_by!r} matched no complete pair across "
            f"{control} and {treatment}."
        )
    return pair_ids, control_index, treatment_index


def ols_covariate_fit(
    y: np.ndarray,
    x: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-feature OLS of expression on one continuous covariate.

    ``y`` is features x samples (log2 CPM), ``x`` one value per sample. Returns
    ``slope``, ``intercept``, ``slope_se``, ``t_statistic``, ``p_value``,
    ``pearson_r`` and ``r_squared``, each one value per feature, plus the
    scalar ``df``.

    This is what ``--statistic lm`` computes: ``--covariate`` makes ``x`` a
    sample-sheet column (a demographic or analytic covariate to regress
    against) and ``--covariate-feature`` makes it another feature's values
    across the same samples ("what correlates to SOX2 expression"). Same fit
    either way; only where x comes from differs.

    EVERY SAMPLE IN SCOPE, not two levels. A regression on a covariate is
    not a two-group contrast, so restricting it to two levels of a factor would
    throw away the range of x that gives the slope its precision.
    """
    y_matrix = np.asarray(y, dtype=float)
    x_vector = np.asarray(x, dtype=float)
    if y_matrix.ndim != 2:
        raise ValueError("y must be features x samples")
    if x_vector.shape[0] != y_matrix.shape[1]:
        raise ValueError(
            f"the covariate has {x_vector.shape[0]} values and the matrix has "
            f"{y_matrix.shape[1]} samples"
        )
    n = y_matrix.shape[1]
    if n < 3:
        raise ValueError(
            f"lm needs at least three samples (two parameters plus one residual "
            f"df); this comparison has {n}."
        )
    if _is_constant(x_vector):
        raise ValueError(
            "the covariate is the same value in every sample, so there is "
            "nothing to regress against."
        )
    from scipy import stats as _st

    df = float(n - 2)
    x_centered = x_vector - x_vector.mean()
    sxx = float((x_centered ** 2).sum())
    y_mean = y_matrix.mean(axis=1)
    slope = (y_matrix - y_mean[:, None]) @ x_centered / sxx
    intercept = y_mean - slope * x_vector.mean()
    fitted = intercept[:, None] + slope[:, None] * x_vector[None, :]
    residual_ss = ((y_matrix - fitted) ** 2).sum(axis=1)
    total_ss = ((y_matrix - y_mean[:, None]) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_square_error = residual_ss / df
        slope_se = np.sqrt(mean_square_error / sxx)
        t_stat = np.where(slope_se > 0, slope / slope_se, np.nan)
        r_squared = np.where(total_ss > 0, 1.0 - residual_ss / total_ss, np.nan)
    # Signed correlation: r_squared alone cannot say which direction.
    pearson_r = np.sign(slope) * np.sqrt(np.clip(r_squared, 0.0, 1.0))
    p_value = np.full(t_stat.shape, np.nan)
    finite = np.isfinite(t_stat)
    if finite.any():
        p_value[finite] = 2.0 * _st.t.sf(np.abs(t_stat[finite]), df)
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "t_statistic": t_stat,
        "p_value": p_value,
        "pearson_r": pearson_r,
        "r_squared": r_squared,
        "df": df,
    }


def ols_covariate_block_fit(
    y: np.ndarray,
    x: np.ndarray,
    blocks: Sequence[str],
) -> Dict[str, np.ndarray]:
    """``ols_covariate_fit`` with a block term (--block-by).

    The model is intercept + covariate + one column per block level after the first; the slope is the
    covariate's coefficient in it, tested on that model's error, df = N - rank. A block whose samples share one
    covariate value carries no information on the slope and simply absorbs its own mean. ``pearson_r`` and
    ``r_squared`` are the partial ones: of y and x after each is taken off the blocks. Returns what
    ``ols_covariate_fit`` returns.
    """
    from scipy import stats as _st

    y_matrix = np.asarray(y, dtype=float)
    x_vector = np.asarray(x, dtype=float)
    labels = [str(b) for b in blocks]
    levels = list(dict.fromkeys(labels))
    block_design = np.column_stack([np.ones(len(labels))] + [
        np.asarray([1.0 if b == level else 0.0 for b in labels]) for level in levels[1:]])
    design = np.column_stack([block_design, x_vector])
    rank = int(np.linalg.matrix_rank(design))
    if rank <= int(np.linalg.matrix_rank(block_design)):
        raise ValueError(
            "the covariate does not change within any block, so the blocks already hold it and there is "
            "no slope left to estimate"
        )
    df = float(len(labels) - rank)
    if df < 1:
        raise ValueError(f"lm blocked: {len(labels)} samples and {rank} parameters leave no error degrees of freedom")
    pinv = np.linalg.pinv(design)
    beta = y_matrix @ pinv.T
    residual = y_matrix - beta @ design.T
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.einsum("ij,ij->i", residual, residual) / df
        xtx_inverse = np.linalg.pinv(design.T @ design)
        slope = beta[:, -1]
        slope_se = np.sqrt(np.maximum(mse * float(xtx_inverse[-1, -1]), 0.0))
        t_stat = np.where(slope_se > 0, slope / slope_se, np.nan)
    maker = np.eye(len(labels)) - block_design @ np.linalg.pinv(block_design)
    y_within = y_matrix @ maker.T
    x_within = maker @ x_vector
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx = float(x_within @ x_within)
        syy = np.einsum("ij,ij->i", y_within, y_within)
        pearson_r = np.where(syy > 0, (y_within @ x_within) / np.sqrt(sxx * syy), np.nan)
    p_value = np.full(t_stat.shape, np.nan)
    finite = np.isfinite(t_stat)
    if finite.any():
        p_value[finite] = 2.0 * _st.t.sf(np.abs(t_stat[finite]), df)
    return {
        "slope": slope, "intercept": beta[:, 0], "slope_se": slope_se, "t_statistic": t_stat,
        "p_value": p_value, "pearson_r": pearson_r, "r_squared": pearson_r ** 2, "df": df,
    }


def numeric_covariate(
    rows: Sequence[Mapping[str, str]],
    column: str,
) -> np.ndarray:
    """Read one sample-sheet column as the lm's x, refusing a non-numeric value by name."""
    if not rows or column not in rows[0]:
        raise ValueError(
            f"--covariate {column!r}: no such column. Available: "
            f"{', '.join(sorted(rows[0])) if rows else 'none'}"
        )
    values: List[float] = []
    bad: List[str] = []
    for row in rows:
        raw = str(row.get(column) or "").strip()
        try:
            values.append(float(raw))
        except ValueError:
            bad.append(f"{row.get('sample_id')}={raw!r}")
    if bad:
        raise ValueError(
            f"--covariate {column!r} must be numeric to regress against; "
            f"could not read {', '.join(bad[:6])}"
            + (" ..." if len(bad) > 6 else "")
            + ". A two-level category is a --factor for the ANOVA, not a covariate."
        )
    return np.asarray(values, dtype=float)


def parse_statistic_request(
    value: str,
    *,
    factors: Sequence[str] = (),
    data_levels: Sequence[str] = (),
    default_data_level: Optional[str] = None,
    covariate: Optional[str] = None,
    pair_by: Optional[str] = None,
    validate_companions: bool = True,
) -> StatisticRequest:
    """Read a --statistic string. The one place any pipeline reads one.

    ``data_levels`` is what this pipeline will accept after the colon; an
    empty sequence means it has no data-level choice and the suffix is
    refused. Every refusal names the flag that is missing or wrong, because
    the alternative is a run that quietly tests something else.

    ``validate_companions=False`` is for re-reading a string already validated
    and recorded (a checkpoint, a plan), where the run's --covariate and
    --pair-by are not to hand and are not needed.
    """
    text = str(value)
    # Each comma-separated entry is <statistic>[:<data level>], the level its own.
    requested: List[str] = []
    levels: List[Optional[str]] = []
    for entry in text.split(","):
        statistic_text, colon, level_text = entry.partition(":")
        statistic = statistic_text.strip().lower()
        level = level_text.strip().lower() or None
        if not statistic:
            if colon:
                raise ValueError(f"--statistic {text!r}: a data level needs its statistic before the colon, e.g. t:cell")
            continue
        if level is not None and not data_levels:
            raise ValueError(
                f"--statistic does not take a data level here ({statistic}:{level}): this "
                "pipeline has one kind of observation. The colon form is for the single-cell "
                "pipelines, e.g. --statistic t:cell"
            )
        if level is not None and level not in data_levels:
            raise ValueError(
                f"unknown data level {level!r} ({statistic}:{level}); the only one is "
                f"{', '.join(data_levels)} (--statistic {statistic}:{data_levels[0]}). "
                "Saying nothing is the default, so there is no term for it."
            )
        if level is None and data_levels:
            level = default_data_level
        requested.append(statistic)
        levels.append(level)
    if not requested:
        raise ValueError(f"--statistic requires at least one of: {', '.join(ALL_STATISTICS)}")

    unknown = [token for token in requested if token not in ALL_STATISTICS]
    if unknown:
        raise ValueError(
            f"unknown statistic(s) {', '.join(sorted(set(unknown)))}; "
            f"choose from {', '.join(ALL_STATISTICS)}"
        )
    named = [statistic_folder(s, l) for s, l in zip(requested, levels)]
    repeated = sorted({
        f"{s}:{l}" if l == DATA_LEVEL_CELL else s
        for s, l, n in zip(requested, levels, named) if named.count(n) > 1
    })
    if repeated:
        raise ValueError(
            f"--statistic names {', '.join(repeated)} more than once; each statistic "
            "at each data level is computed once and writes one set of results"
        )
    statistics = tuple(requested)
    one_level = len(set(levels)) <= 1
    level = levels[0] if one_level else None
    made = StatisticRequest(statistics, level) if one_level else StatisticRequest(statistics, None, tuple(levels))
    if not validate_companions:
        return made

    for spec in factors:
        parse_factor(spec)
    if "anova" in statistics and not factors:
        raise ValueError(
            "--statistic anova needs at least one --factor. One is the one-way ANOVA (a "
            "column of three or more levels: --factor Day:1); two or more are the factorial "
            "ANOVA with every interaction (--factor Diet:CD --factor Metformin:VEH)"
        )
    grouped = [s for s in statistics if s in ("t", "paired-t")]
    if grouped and not factors:
        raise ValueError(
            f"--statistic {','.join(grouped)} compares groups, and the groups are a --factor: "
            "name the sample-sheet column and its control level, e.g. --factor Diet:CD"
        )
    if "lm" in statistics and not covariate:
        raise ValueError(
            "--statistic lm needs something to regress against: pass "
            "--covariate <sample-sheet column> or --covariate-feature <gene>"
        )
    if "paired-t" in statistics and not pair_by:
        raise ValueError(
            "--statistic paired-t needs to know which samples pair: pass "
            "--pair-by <sample-sheet column>"
        )
    if pair_by and "paired-t" not in statistics:
        raise ValueError(
            f"--pair-by {pair_by} is only read by --statistic paired-t, which this "
            f"run does not ask for (--statistic {text}). Add paired-t to the list, "
            "or leave out --pair-by"
        )
    if covariate and "lm" not in statistics:
        raise ValueError(
            f"--covariate/--covariate-feature {covariate} is only read by --statistic "
            f"lm, which this run does not ask for (--statistic {text}). Add lm to the "
            "list, or leave the covariate out"
        )
    return made


def student_t_from_moments(
    control_mean, control_variance, control_n,
    treatment_mean, treatment_variance, treatment_n,
):
    """Paralome's two-sample t-test. The only one. Returns (t, df, p).

    Pooled variance, df = n_control + n_treatment - 2, and the
    difference taken as treatment minus control so the sign matches
    log2_fold_change everywhere.

    WHY THERE IS ONLY ONE, AND WHY IT IS HERE. Every pipeline that tests two
    groups calls this function: one test, applied identically everywhere, in
    the shared module rather than written again per pipeline.

    Student's t is the simplest case of the family
    the pipelines offer -- it, the ANOVA F and a linear-model fit give the
    identical p-value on a two-arm, single-factor design (F = t squared). The
    log transform earlier in every pipeline is there to reach approximate
    normality so that this statistic is the one available.
    """
    from scipy import stats as _st

    control_mean = np.asarray(control_mean, dtype=float)
    treatment_mean = np.asarray(treatment_mean, dtype=float)
    control_variance = np.asarray(control_variance, dtype=float)
    treatment_variance = np.asarray(treatment_variance, dtype=float)
    n_control = float(control_n)
    n_treatment = float(treatment_n)

    df = n_control + n_treatment - 2.0
    if df <= 0 or n_control < 2 or n_treatment < 2:
        nan = np.full(np.broadcast(control_mean, treatment_mean).shape, np.nan)
        return nan, float("nan"), nan.copy()

    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_variance = (
            (n_control - 1.0) * control_variance + (n_treatment - 1.0) * treatment_variance
        ) / df
        standard_error = np.sqrt(pooled_variance * (1.0 / n_control + 1.0 / n_treatment))
        t_statistic = (treatment_mean - control_mean) / standard_error
    p_value = np.full(np.shape(t_statistic), np.nan)
    finite = np.isfinite(t_statistic)
    if np.any(finite):
        p_value[finite] = 2.0 * _st.t.sf(np.abs(t_statistic[finite]), df)
    return t_statistic, df, p_value


def student_power(
    mean_difference: float, variance1: float, variance2: float,
    n1: float, n2: float, alpha: float,
) -> float:
    """Analytic power for the pooled-variance t-test.

    Non-central t, with nan-filling for large non-centrality at small df; the
    variance is pooled and the df fixed at n1 + n2 - 2.
    """
    from scipy import stats as _st

    if not all(math.isfinite(x) for x in (mean_difference, variance1, variance2)):
        return float("nan")
    if n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2.0
    pooled = ((n1 - 1.0) * variance1 + (n2 - 1.0) * variance2) / df
    se2 = pooled * (1.0 / n1 + 1.0 / n2)
    if not math.isfinite(se2) or se2 <= 0:
        return float("nan")
    ncp = abs(mean_difference) / math.sqrt(se2)
    critical = float(_st.t.ppf(1 - alpha / 2.0, df))
    upper = float(_st.nct.sf(critical, df, ncp))
    lower = float(_st.nct.cdf(-critical, df, ncp))
    # Scipy's non-central t returns nan for a large non-centrality at a small df, which is the strongest
    # effects in the table; they are filled rather than left blank and dropped from every summary.
    if not math.isfinite(upper):
        upper = _noncentral_t_tail(critical, df, ncp)
    if not math.isfinite(lower):
        lower = _noncentral_t_tail(critical, df, -ncp)
    value = upper + lower
    if not math.isfinite(value):
        return float("nan")
    return min(1.0, max(0.0, value))



def _noncentral_t_tail(critical: float, df: float, ncp: float) -> float:
    """P(T > critical) for a non-central t, by integrating over its chi-square.

    T = (Z + ncp) / S with Z standard normal and S = sqrt(V / df), V
    chi-square(df), so the tail is the mean over S of Phi(ncp - critical * S).
    By symmetry P(T < -critical) is this with -ncp.

    Integrated over S itself, with the place where Phi steps from 1
    to 0 (S = ncp / critical) and the bulk of S's density given to quad as
    breakpoints. The first version integrated over V's probability scale,
    where that step is a sliver quad could miss: df=1, ncp=50 came out 1.0
    where the answer is 0.99991253.
    """
    from scipy import integrate as _integrate
    from scipy import stats as _st

    if not (math.isfinite(critical) and math.isfinite(df) and math.isfinite(ncp)) or df <= 0:
        return float("nan")
    if critical == 0:
        return float(_st.norm.cdf(ncp))

    half = df / 2.0
    log_constant = math.log(2.0) + half * math.log(half) - math.lgamma(half)

    def integrand(s: float) -> float:
        if s <= 0:
            return 0.0
        log_density = log_constant + (df - 1.0) * math.log(s) - half * s * s
        return math.exp(log_density) * 0.5 * math.erfc((critical * s - ncp) / math.sqrt(2.0))

    # Beyond `top` S has less than 1e-18 of its probability.
    top = math.sqrt(float(_st.chi2.isf(1e-18, df)) / df)
    spread = 1.0 / math.sqrt(2.0 * df)
    step = ncp / critical
    width = 1.0 / abs(critical)
    marks = {1.0 + k * spread for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8)}
    marks.update(step + k * width for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8))
    points = sorted(mark for mark in marks if 0.0 < mark < top)
    edges = [0.0, *points, top]
    value = 0.0
    for low, high in zip(edges, edges[1:]):
        piece, _error = _integrate.quad(integrand, low, high, limit=200, epsabs=1e-14, epsrel=1e-12)
        value += piece
    return min(1.0, max(0.0, float(value)))


# ---------------------------------------------------------------------------
# The vectorised fits: every feature of a matrix at once
#
# The statistics above were written for one feature at a time and
# called from a per-gene loop in one pipeline and from vectorised code in
# another. These fit the whole matrix (features x samples) in one pass -- same
# arithmetic, checked against the per-feature functions they replace -- so the
# shared differential engine (paralome.engine) can serve bulk and pseudobulk
# data of any size from one implementation.
# ---------------------------------------------------------------------------


def constant_rows(y: np.ndarray) -> np.ndarray:
    """Per feature: is every value the same number? (``_is_constant``, row-wise.)"""
    matrix = np.asarray(y, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0] if matrix.ndim else 0, dtype=bool)
    finite = np.all(np.isfinite(matrix), axis=1)
    scale = np.maximum(1.0, np.max(np.abs(np.where(np.isfinite(matrix), matrix, 0.0)), axis=1))
    spread = np.ptp(np.where(np.isfinite(matrix), matrix, 0.0), axis=1)
    return finite & (spread <= _CONSTANT_TOLERANCE * scale)


def group_moments(y: np.ndarray, index: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Mean and unbiased (n-1) variance of each feature over the samples in ``index``."""
    matrix = np.asarray(y, dtype=float)[:, list(index)]
    n = matrix.shape[1]
    mean = matrix.mean(axis=1) if n else np.full(matrix.shape[0], np.nan)
    variance = matrix.var(axis=1, ddof=1) if n >= 2 else np.full(matrix.shape[0], np.nan)
    return mean, variance, n


def _residual_maker(design: np.ndarray) -> Tuple[np.ndarray, int]:
    """M = I - X X^+ (projects onto the residual space) and rank(X)."""
    x = np.asarray(design, dtype=float)
    hat = x @ np.linalg.pinv(x)
    return np.eye(x.shape[0]) - hat, int(np.linalg.matrix_rank(x))


def anova_table(
    y: np.ndarray, design: np.ndarray, term_columns: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, object]]:
    """``_anova_terms`` for every feature at once: term -> {F, p, df_term, df_error}.

    Type II exactly as ``_anova_terms``, each term tested as all of its
    columns; every F over the FULL model's error. A feature with one value in
    every sample has no variance to partition and gets NaN.
    """
    from scipy import stats as _st

    matrix = np.asarray(y, dtype=float)
    width = design.shape[1]
    full_maker, full_rank = _residual_maker(design)
    df_error = matrix.shape[1] - full_rank
    full_residual = matrix @ full_maker.T
    full_rss = np.einsum("ij,ij->i", full_residual, full_residual)
    flat = constant_rows(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.where(flat, np.nan, full_rss / df_error) if df_error > 0 else np.full(matrix.shape[0], np.nan)
    out: Dict[str, Dict[str, object]] = {}
    for name, columns in term_columns.items():
        reference_columns = term_reference_columns(name, term_columns, width)
        reduced_columns = [i for i in reference_columns if i not in set(columns)]
        reference_maker, reference_rank = _residual_maker(design[:, reference_columns])
        reduced_maker, reduced_rank = _residual_maker(design[:, reduced_columns])
        reference_residual = matrix @ reference_maker.T
        reduced_residual = matrix @ reduced_maker.T
        reference_rss = np.einsum("ij,ij->i", reference_residual, reference_residual)
        reduced_rss = np.einsum("ij,ij->i", reduced_residual, reduced_residual)
        df_term = max(reference_rank - reduced_rank, 1)
        extra = np.maximum(reduced_rss - reference_rss, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_value = np.where(mse > 0, (extra / df_term) / mse, np.nan)
        p_value = np.full(f_value.shape, np.nan)
        finite = np.isfinite(f_value)
        if finite.any() and df_error > 0:
            p_value[finite] = _st.f.sf(f_value[finite], df_term, df_error)
        out[name] = {"F": f_value, "p": p_value, "df_term": float(df_term), "df_error": float(df_error)}
    return out


def contrast_table(
    y: np.ndarray,
    design: np.ndarray,
    contrasts: Sequence[Tuple[str, np.ndarray]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, np.ndarray]:
    """Every contrast of the factorial fit, for every feature: estimate, SE, t, p.

    Each estimate is ``w . beta`` and its error ``w (X'X)^-1 w' * MSE`` from the
    FULL model, so the fold change and the p-value of a contrast come from one
    vector and cannot describe different comparisons. The df is the model's,
    N - rank(X). Returns ``(by name, df, beta)``; beta is features x columns,
    so the interaction coefficient (the direction of the interaction F) is
    ``beta[:, 3]`` in a 2x2.
    """
    from scipy import stats as _st

    matrix = np.asarray(y, dtype=float)
    x = np.asarray(design, dtype=float)
    pinv = np.linalg.pinv(x)
    beta = matrix @ pinv.T
    residual = matrix - beta @ x.T
    rank = int(np.linalg.matrix_rank(x))
    df = float(matrix.shape[1] - rank)
    flat = constant_rows(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.einsum("ij,ij->i", residual, residual) / df if df > 0 else np.full(matrix.shape[0], np.nan)
    xtx_inverse = np.linalg.pinv(x.T @ x)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, weights in contrasts:
        w = np.asarray(weights, dtype=float)
        estimate = beta @ w
        variance = float(w @ xtx_inverse @ w) * mse
        se = np.sqrt(np.maximum(variance, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_value = np.where(se > 0, estimate / se, np.nan)
        t_value = np.where(flat, np.nan, t_value)
        estimate = np.where(flat, 0.0, estimate)
        p_value = np.full(t_value.shape, np.nan)
        finite = np.isfinite(t_value)
        if finite.any() and df > 0:
            p_value[finite] = 2.0 * _st.t.sf(np.abs(t_value[finite]), df)
        out[name] = {"estimate": estimate, "se": np.where(flat, np.nan, se), "t": t_value, "p": p_value}
    return out, df, beta


def factorial_from_sample_sums(
    sums: np.ndarray,
    squares: np.ndarray,
    counts: np.ndarray,
    design: np.ndarray,
    term_columns: Mapping[str, Sequence[int]],
    contrasts: Sequence[Tuple[str, np.ndarray]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, np.ndarray]], float, np.ndarray]:
    """The factorial ANOVA with the CELLS as the observations, from per-sample sums.

    Every cell of a sample has its sample's design row,
    so the least-squares fit over the cells needs only, per feature, each
    sample's sum over its cells (``sums``, features x samples), the sum of
    squares over every cell (``squares``) and each sample's cell count
    (``counts``): X'X = sum_s n_s x_s x_s', X'y = sum_s x_s S_s, RSS = y'y -
    b'X'y. The same model as ``anova_table`` and ``contrast_table`` with each
    cell a row -- Type II F for every term within the model of the terms that
    do not contain it (``term_reference_columns``), every F and contrast over
    the full model's error, df = cells - rank -- without a cells x cells
    matrix. Returns ``(terms, contrasts, df, beta)`` in their shapes.
    """
    from scipy import stats as _st

    sums = np.atleast_2d(np.asarray(sums, dtype=float))
    squares = np.asarray(squares, dtype=float)
    counts = np.asarray(counts, dtype=float)
    x = np.asarray(design, dtype=float)
    n_cells = float(counts.sum())
    width = x.shape[1]

    def fit(columns: Sequence[int]):
        part = x[:, list(columns)]
        gram = part.T @ (part * counts[:, None])
        inverse = np.linalg.pinv(gram)
        moment = sums @ part
        beta = moment @ inverse
        rss = np.maximum(squares - np.einsum("ij,ij->i", beta, moment), 0.0)
        return beta, rss, int(np.linalg.matrix_rank(gram)), inverse

    beta, full_rss, full_rank, full_inverse = fit(range(width))
    df_error = n_cells - full_rank
    total = squares - sums.sum(axis=1) ** 2 / max(n_cells, 1.0)
    flat = total <= 1e-12 * np.maximum(squares, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.where(flat, np.nan, full_rss / df_error) if df_error > 0 else np.full(sums.shape[0], np.nan)

    terms: Dict[str, Dict[str, object]] = {}
    fitted: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {tuple(range(width)): (full_rss, full_rank)}
    for name, columns in term_columns.items():
        reference = tuple(term_reference_columns(name, term_columns, width))
        reduced = tuple(i for i in reference if i not in set(columns))
        for key in (reference, reduced):
            if key not in fitted:
                _b, rss, rank, _i = fit(key)
                fitted[key] = (rss, rank)
        reference_rss, reference_rank = fitted[reference]
        reduced_rss, reduced_rank = fitted[reduced]
        df_term = max(reference_rank - reduced_rank, 1)
        extra = np.maximum(reduced_rss - reference_rss, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_value = np.where(mse > 0, (extra / df_term) / mse, np.nan)
        p_value = np.full(f_value.shape, np.nan)
        finite = np.isfinite(f_value)
        if finite.any() and df_error > 0:
            p_value[finite] = _st.f.sf(f_value[finite], df_term, df_error)
        terms[name] = {"F": f_value, "p": p_value, "df_term": float(df_term), "df_error": float(df_error)}

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, weights in contrasts:
        w = np.asarray(weights, dtype=float)
        estimate = beta @ w
        se = np.sqrt(np.maximum(float(w @ full_inverse @ w) * mse, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_value = np.where(se > 0, estimate / se, np.nan)
        t_value = np.where(flat, np.nan, t_value)
        p_value = np.full(t_value.shape, np.nan)
        finite = np.isfinite(t_value)
        if finite.any() and df_error > 0:
            p_value[finite] = 2.0 * _st.t.sf(np.abs(t_value[finite]), df_error)
        out[name] = {"estimate": np.where(flat, 0.0, estimate), "se": np.where(flat, np.nan, se),
                     "t": t_value, "p": p_value}
    return terms, out, float(df_error), beta


def observed_power_array(effect_size: np.ndarray, n_control: float, n_treatment: float,
                         alpha: float = 0.05) -> np.ndarray:
    """``observed_power`` for every feature: the same non-central t, element-wise."""
    effect = np.asarray(effect_size, dtype=float)
    out = np.full(effect.shape, np.nan)
    if n_control < 2 or n_treatment < 2:
        return out
    from scipy import stats as _st

    df = n_control + n_treatment - 2
    critical = float(_st.t.ppf(1 - alpha / 2.0, df))
    finite = np.isfinite(effect)
    if not finite.any():
        return out
    ncp = np.abs(effect[finite]) * math.sqrt((n_control * n_treatment) / (n_control + n_treatment))
    with np.errstate(all="ignore"):
        value = _st.nct.sf(critical, df, ncp) + _st.nct.cdf(-critical, df, ncp)
        normal = _st.norm.sf(critical - ncp) + _st.norm.cdf(-critical - ncp)
    value = np.where(np.isfinite(value), value, normal)
    out[finite] = np.clip(value, 0.0, 1.0)
    return out


def n_per_group_for_power(
    effect_size: float, alpha: float = 0.05, power: float = 0.80, limit: int = 100000,
) -> float:
    """The smallest equal n per group at which ``observed_power`` reaches ``power``.

    The power summary asked statsmodels' TTestIndPower for this --
    a second power implementation beside ``observed_power``, which returns nan
    for large effects at small n (see ``observed_power``). Solved here with the
    one non-central t the rest of the package uses. NaN when no n up to
    ``limit`` gets there (an effect near zero, normal for a genome-wide median).
    """
    if not math.isfinite(effect_size) or effect_size == 0:
        return float("nan")
    if observed_power(effect_size, limit, limit, alpha) < power:
        return float("nan")
    low, high = 2, limit
    if observed_power(effect_size, low, low, alpha) >= power:
        return float(low)
    while high - low > 1:
        middle = (low + high) // 2
        if observed_power(effect_size, middle, middle, alpha) >= power:
            high = middle
        else:
            low = middle
    return float(high)


# ---------------------------------------------------------------------------
# The random-intercept fit: anova_cell
# ---------------------------------------------------------------------------


def t_two_sided_p(t_value, df: float) -> np.ndarray:
    """Two-sided p for t statistics at ``df``; NaN where t is not finite or df <= 0."""
    from scipy import stats as _st

    values = np.asarray(t_value, dtype=float)
    p = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.any() and df > 0:
        p[finite] = 2.0 * _st.t.sf(np.abs(values[finite]), df)
    return p


def f_upper_p(f_value, df_term: float, df_error: float) -> np.ndarray:
    """Upper-tail p for F statistics on (df_term, df_error); NaN where F is not finite or a df <= 0."""
    from scipy import stats as _st

    values = np.asarray(f_value, dtype=float)
    p = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.any() and df_term > 0 and df_error > 0:
        p[finite] = _st.f.sf(values[finite], df_term, df_error)
    return p


def fisher_exact_2x2(
    a: int, b: int, c: int, d: int, *, alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Fisher's exact test on [[a, b], [c, d]]: (sample odds ratio, p).

    The motif enrichment (singles, two-sided) and the motif-pair
    co-occurrence (one-sided, "greater") both call this; the test has one home
    with the rest of the package's statistics.
    """
    from scipy import stats as _st

    odds, p = _st.fisher_exact([[int(a), int(b)], [int(c), int(d)]], alternative=alternative)
    return float(odds), float(p)


#: The two-sided 95% normal quantile.
Z_95 = 1.959963984540054


def inverse_variance_pooled(
    estimates: Sequence[float], variances: Sequence[float],
) -> Optional[Dict[str, float]]:
    """Inverse-variance (fixed-effect) pool of independent estimates, with a Wald z test.

    Estimates whose variance is not positive carry no weight. Returns None
    when nothing carries weight; otherwise the pooled estimate, its standard
    error, the 95% interval, z and the two-sided normal p.
    """
    numerator = denominator = 0.0
    used = 0
    for estimate, variance in zip(estimates, variances):
        if not variance > 0:
            continue
        weight = 1.0 / float(variance)
        numerator += weight * float(estimate)
        denominator += weight
        used += 1
    if used == 0 or denominator <= 0:
        return None
    estimate = numerator / denominator
    stderr = math.sqrt(1.0 / denominator)
    z = estimate / stderr if stderr > 0 else 0.0
    return {
        "estimate": estimate, "stderr": stderr,
        "ci_low": estimate - Z_95 * stderr, "ci_high": estimate + Z_95 * stderr,
        "z": z, "p": math.erfc(abs(z) / math.sqrt(2.0)), "used": used,
    }


def t_interval_multiplier(confidence: float, df: float) -> float:
    """The two-sided Student t quantile for a ``confidence`` interval at ``df`` (NaN when df <= 0)."""
    if not df > 0:
        return float("nan")
    from scipy import stats as _st

    return float(_st.t.ppf(0.5 + float(confidence) / 2.0, df))


def spearman_matrix(rows_by_column: np.ndarray) -> np.ndarray:
    """Rows x rows Spearman correlation of a rows x columns matrix.

    scipy returns a scalar when there are two rows; that is returned as the
    2 x 2 matrix it stands for (an identity there would put the two rows'
    correlation at 0).
    """
    from scipy import stats as _st

    values = np.asarray(rows_by_column, dtype=float)
    n = values.shape[0]
    correlation = _st.spearmanr(values, axis=1).statistic
    if np.isscalar(correlation) or np.ndim(correlation) == 0:
        value = float(correlation)
        if n == 2:
            return np.array([[1.0, value], [value, 1.0]], dtype=float)
        return np.eye(n, dtype=float)
    return np.asarray(correlation, dtype=float)


# ---------------------------------------------------------------------------
# The one-way ANOVA
# ---------------------------------------------------------------------------

def one_way_anova(
    y: np.ndarray, groups: Sequence[Sequence[int]], level_values: Optional[Sequence[float]],
) -> Dict[str, np.ndarray]:
    """One factor of k >= 3 levels: the omnibus F, and for numeric levels its linear trend and the rest.

    ``y`` is features x observations (samples, or cells at ``:cell``);
    ``groups`` the observation positions of each level, in level order;
    ``level_values`` each level's number (the trend reads the levels' real
    spacing). Vectorised over features.

    Returns per feature the level means (features x k) and, with ``df_within``
    = N - k as the error for every test:

    * ``F``/``p``, df k - 1: the omnibus test of the level means;
    * ``trend_slope`` (log2 per unit of the level values), ``trend_slope_se``,
      ``trend_F``/``trend_p``, df 1: the straight line on the level values;
    * ``lack_of_fit_F``/``lack_of_fit_p``, df k - 2: what the level means show
      beyond that line.

    The two parts' sums of squares add up to the omnibus one. A feature with
    one value everywhere, or a fit with no error df, is NaN throughout.
    """
    values = np.asarray(y, dtype=float)
    groups = [np.asarray(list(g), dtype=int) for g in groups]
    k = len(groups)
    # Levels that are not numbers (sex, strain) have no spacing, so
    # no trend: the omnibus F alone, and the trend columns NaN.
    numeric = level_values is not None
    x = np.asarray(list(level_values), dtype=float) if numeric else np.arange(k, dtype=float)
    if k < 3 or x.size != k:
        raise ValueError("a one-way ANOVA here needs three or more levels")
    n_level = np.array([g.size for g in groups], dtype=float)
    n_total = float(n_level.sum())
    n_features = values.shape[0]
    means = np.column_stack([
        values[:, g].mean(axis=1) if g.size else np.full(n_features, np.nan) for g in groups
    ])
    columns = np.concatenate(groups) if groups else np.zeros(0, dtype=int)
    grand = values[:, columns].mean(axis=1)
    ss_between = ((means - grand[:, None]) ** 2 * n_level[None, :]).sum(axis=1)
    ss_within = np.zeros(n_features)
    for j, g in enumerate(groups):
        if g.size:
            ss_within += ((values[:, g] - means[:, [j]]) ** 2).sum(axis=1)
    df_between = float(k - 1)
    df_within = float(n_total - k)

    x_bar = float((n_level * x).sum() / n_total)
    sxx = float((n_level * (x - x_bar) ** 2).sum())
    slope = ((means - grand[:, None]) * (n_level * (x - x_bar))[None, :]).sum(axis=1) / sxx
    ss_linear = slope ** 2 * sxx
    ss_lack = np.maximum(ss_between - ss_linear, 0.0)

    out: Dict[str, np.ndarray] = {"means": means, "n_level": n_level, "df_between": np.float64(df_between),
                                  "df_within": np.float64(df_within)}
    nan = np.full(n_features, np.nan)
    if df_within < 1 or not np.all(n_level > 0):
        out.update(F=nan.copy(), p=nan.copy(), mse=nan.copy(), trend_slope=nan.copy(),
                   trend_slope_se=nan.copy(), trend_F=nan.copy(), trend_p=nan.copy(),
                   lack_of_fit_F=nan.copy(), lack_of_fit_p=nan.copy())
        return out
    from scipy import stats as _st

    mse = ss_within / df_within
    flat = constant_rows(values[:, columns])
    with np.errstate(invalid="ignore", divide="ignore"):
        f_value = (ss_between / df_between) / mse
        trend_f = ss_linear / mse
        lack_f = (ss_lack / (k - 2)) / mse
        slope_se = np.sqrt(mse / sxx)

    def tail(f: np.ndarray, df1: float) -> np.ndarray:
        p = np.full(n_features, np.nan)
        ok = np.isfinite(f) & ~flat
        p[ok] = _st.f.sf(f[ok], df1, df_within)
        return p

    out.update(
        mse=np.where(flat, np.nan, mse),
        F=np.where(flat, np.nan, f_value), p=tail(f_value, df_between),
        trend_slope=np.where(flat, np.nan, slope), trend_slope_se=np.where(flat, np.nan, slope_se),
        trend_F=np.where(flat, np.nan, trend_f), trend_p=tail(trend_f, 1.0),
        lack_of_fit_F=np.where(flat, np.nan, lack_f), lack_of_fit_p=tail(lack_f, float(k - 2)),
    )
    if not numeric:
        for key in ("trend_slope", "trend_slope_se", "trend_F", "trend_p", "lack_of_fit_F", "lack_of_fit_p"):
            out[key] = nan.copy()
    return out


# ---------------------------------------------------------------------------
# A model of many observations, from its cross-products
#
# each gene along a lineage's pseudotime,
# the run's full factorial crossed with a natural spline of pseudotime, fitted with the cells as observations. A
# lineage holds tens of thousands of cells, so the N x N residual makers anova_table uses are out of reach; every
# nested model's fit is read from X'X, X'y and y'y instead, which hold the same least squares.
# ---------------------------------------------------------------------------


def natural_spline_basis(x: Sequence[float] | np.ndarray, df: int) -> np.ndarray:
    """A natural cubic spline of ``x``: ``df`` columns that, with an intercept, span what R's ``splines::ns(x, df)``
    spans -- boundary knots at x's range, ``df - 1`` interior knots at its quantiles (R's type 7). ``df`` 1 is the
    straight line. Knots that coincide (many tied values) are one knot, so fewer columns come back.

    The columns are the truncated-power form (Hastie, Tibshirani and Friedman, eq. 5.4-5.5) on x scaled to 0-1, not
    R's B-spline form: the same space, so every F test of them is the same."""
    values = np.asarray(x, dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    if not high > low:
        return np.zeros((values.size, 0))
    z = (values - low) / (high - low)
    interior = np.quantile(z, np.linspace(0.0, 1.0, int(df) + 1)[1:-1]) if int(df) > 1 else np.array([])
    knots = np.unique(np.concatenate([[0.0], interior, [1.0]]))
    last = knots[-1]

    def d(k: int) -> np.ndarray:
        return (np.clip(z - knots[k], 0.0, None) ** 3 - np.clip(z - last, 0.0, None) ** 3) / (last - knots[k])

    columns = [z] + [d(k) - d(len(knots) - 2) for k in range(len(knots) - 2)]
    return np.column_stack(columns)


def crossproduct_anova(
    yty: np.ndarray, xty: np.ndarray, gram: np.ndarray, n_observations: int,
    term_columns: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[str, object]]:
    """``anova_table``'s Type II F tests, every feature at once, from the model's cross-products.

    ``yty``: each feature's sum of squares (features); ``xty``: X'y (features x columns); ``gram``: X'X. Each term
    is tested within the model of every term that does not contain it (``term_reference_columns``), as all of its
    columns at once, over the full model's error, df N - rank(X). A nested model's residual sum of squares is
    y'y - b' (X_s'X_s)^+ b with b = X_s'y; the columns are scaled to unit length first and a direction counts toward
    the rank when its eigenvalue is above 1e-9 of the largest. A term the data cannot separate from the others (df
    0), and a feature with no residual, are NaN.
    """
    from scipy import stats as _st

    yty = np.asarray(yty, dtype=float)
    xty = np.atleast_2d(np.asarray(xty, dtype=float))
    gram = np.asarray(gram, dtype=float)
    width = gram.shape[0]
    scale = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    scale[scale == 0] = 1.0
    g = gram / np.outer(scale, scale)
    b = xty / scale
    fits: Dict[Tuple[int, ...], Tuple[np.ndarray, int]] = {}

    def explained(columns: Sequence[int]) -> Tuple[np.ndarray, int]:
        key = tuple(sorted(int(c) for c in columns))
        if key not in fits:
            if not key:
                fits[key] = (np.zeros(yty.shape), 0)
            else:
                values, vectors = np.linalg.eigh(g[np.ix_(key, key)])
                keep = values > 1e-9 * max(float(values.max()), 1e-300)
                along = b[:, list(key)] @ vectors[:, keep]
                fits[key] = (np.einsum("ij,ij->i", along / values[keep], along), int(keep.sum()))
        return fits[key]

    full, full_rank = explained(range(width))
    df_error = int(n_observations) - full_rank
    residual = np.clip(yty - full, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.where(residual > 1e-12 * np.maximum(yty, 1e-300), residual / df_error, np.nan) \
            if df_error > 0 else np.full(yty.shape, np.nan)
    out: Dict[str, Dict[str, object]] = {}
    for name, columns in term_columns.items():
        reference = term_reference_columns(name, term_columns, width)
        reduced = [i for i in reference if i not in set(columns)]
        reference_fit, reference_rank = explained(reference)
        reduced_fit, reduced_rank = explained(reduced)
        df_term = reference_rank - reduced_rank
        f_value = np.full(yty.shape, np.nan)
        p_value = np.full(yty.shape, np.nan)
        if df_term > 0 and df_error > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                f_value = (np.clip(reference_fit - reduced_fit, 0.0, None) / df_term) / mse
            finite = np.isfinite(f_value)
            if finite.any():
                p_value[finite] = _st.f.sf(f_value[finite], df_term, df_error)
        out[name] = {"F": f_value, "p": p_value, "df_term": float(df_term), "df_error": float(df_error)}
    return out
