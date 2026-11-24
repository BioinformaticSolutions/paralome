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

"""What a differential run compares: one plan, every pipeline.

Every pipeline asks this one design layer: given the samples in scope and what
was asked for, which comparisons does each statistic make? The engine
(``paralome.engine``) then fits them on whatever matrix the pipeline holds --
bulk FPKM, bulk CPM, pseudobulk CPM -- and the cell-level fits answer the same
plan with cells as the observations. So one design gives the same set of
results whichever pipeline runs it.

The names are the ones Evidome reads (there: names are matched
exactly). A main-effect F-test is ``<factor>_<alt>_vs_<ref>`` (or
``<alt>_vs_<ref>`` when the level already says the factor), its contrast
``<alt>_vs_<ref>_main_effect``, the interaction ``<A>_x_<B>``, the simple
effects ``<alt>_vs_<ref>_within_<level>_contrast``. Evidome takes each F-test's
direction from those contrasts by name, so the names are part of the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paralome.errors import ParalomeError
from paralome.stats import (
    DATA_LEVEL_CELL,
    StatisticRequest,
    _dedupe_factor_prefix,
    baseline_first_levels,
    design_matrix,
    is_balanced,
    level_numbers,
    model_error_df,
    numeric_covariate,
    parse_factor,
    resolve_pairs,
    statistic_folder,
)

#: The three kinds of result a comparison produces. A contrast compares two
#: groups and has a direction (t, paired-t, an ANOVA contrast); a slope is a
#: regression on a covariate and has a direction (lm); an F-test is an ANOVA
#: term, whose direction is taken from its own contrasts.
KIND_CONTRAST = "contrast"
KIND_SLOPE = "slope"
KIND_FTEST = "ftest"


# ---------------------------------------------------------------------------
# Names. Moved verbatim from rna/differential.py; the
# reasons for each rule are recorded there in the history and kept short here.
# ---------------------------------------------------------------------------


def safe_name(value: str) -> str:
    """A name as it appears in a file or column: letters and digits; ':' -> '_x_'."""
    value = str(value).replace(":", "_x_")
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def levels_alone_are_ambiguous(
    factors: Sequence[str], levels_by_factor: Mapping[str, Sequence[str]],
) -> bool:
    """Whether two factors would give the SAME comparison name from their levels.

    Two factors coded with the same levels (yes/no and yes/no) would give one
    name to two different tests, and the second written would replace the
    first. For those designs, and only those, the factor name goes back into
    the name. With more factors, any two coded alike.
    """
    if len(factors) < 2:
        return False
    level_sets = [
        frozenset(safe_name(str(level)).casefold() for level in levels_by_factor.get(factor, ()))
        for factor in factors
    ]
    return any(
        first and first == second
        for position, first in enumerate(level_sets) for second in level_sets[position + 1:]
    )


def level_is_shared(
    factor: str, level: str, levels_by_factor: Mapping[str, Sequence[str]],
) -> bool:
    """Whether ``level`` of ``factor`` is also a level of another factor."""
    wanted = safe_name(str(level)).casefold()
    for other, levels in levels_by_factor.items():
        if other == factor:
            continue
        if any(safe_name(str(candidate)).casefold() == wanted for candidate in levels):
            return True
    return False


def level_label(factor: str, level: str, levels_by_factor: Mapping[str, Sequence[str]]) -> str:
    """A level as it appears in a name: ``day4`` for a numeric level, the level itself otherwise.

    A bare number is no name (``4_vs_1``), so a factor whose levels are all
    numbers carries its name into every level (``day4_vs_day1``).
    """
    if level_numbers(list(levels_by_factor.get(factor, ()))) is not None:
        return f"{factor}{level}"
    return str(level)


def contrast_stem(
    factor: str, alternate: str, levels_by_factor: Mapping[str, Sequence[str]], ambiguous: bool,
) -> str:
    """``ALT_vs_CONTROL`` for one level of ``factor``; the factor first when levels alone are ambiguous."""
    control = levels_by_factor[factor][0]
    stem = (f"{level_label(factor, alternate, levels_by_factor)}_vs_"
            f"{level_label(factor, control, levels_by_factor)}")
    if ambiguous and level_numbers(list(levels_by_factor[factor])) is None:
        return f"{factor}_{stem}"
    return stem


def within_label(
    factor: str, level: str, ambiguous: bool,
    levels_by_factor: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """The environment half of a name: the level, with its factor when ambiguous, shared or numeric."""
    if levels_by_factor is not None and level_numbers(list(levels_by_factor.get(factor, ()))) is not None:
        return f"{factor}{level}"
    if ambiguous or (levels_by_factor is not None and level_is_shared(factor, level, levels_by_factor)):
        return f"{factor}_{level}"
    return str(level)


def term_name(factor: str, levels_by_factor: Mapping[str, Sequence[str]], ambiguous: bool) -> str:
    """A main-effect F-test's name.

    A factor of two levels keeps the name Evidome reads (``MET_vs_VEH``, the
    factor dropped when the level restates it). A factor of more levels has
    no single pair to name, so its F-test is ``<factor>_F``.
    """
    levels = list(levels_by_factor[factor])
    if len(levels) == 2:
        stem = contrast_stem(factor, levels[1], levels_by_factor, ambiguous)
        return safe_name(stem if ambiguous else _dedupe_factor_prefix(factor, stem))
    return safe_name(f"{factor}_F")


def _samples_at(
    rows: Sequence[Mapping[str, str]], factor: str, level: str,
    within: str = "", within_level: str = "",
) -> List[int]:
    return [
        index for index, row in enumerate(rows)
        if str(row.get(factor) or "").strip() == level
        and (not within or str(row.get(within) or "").strip() == within_level)
    ]


def _samples_where(rows: Sequence[Mapping[str, str]], levels: Mapping[str, str]) -> List[int]:
    """The rows holding every factor -> level of ``levels``."""
    return [
        index for index, row in enumerate(rows)
        if all(str(row.get(factor) or "").strip() == level for factor, level in levels.items())
    ]


def _environments(
    others: Sequence[str], levels_by_factor: Mapping[str, Sequence[str]], *, every_subset: bool,
) -> List[Tuple[Tuple[str, str], ...]]:
    """The settings of the other factors a comparison is made within: ((factor, level), ...).

    ``every_subset`` (the ANOVA's contrasts): every subset of the other factors
    held at every combination of its levels -- the empty one first (the main
    effect, averaged over all of them), then one factor at a time, two ...,
    to all of them (the simple effects). Otherwise (t and paired-t, which have
    no model to average over): every combination of all the other factors.
    """
    import itertools

    sizes = range(len(others) + 1) if every_subset else [len(others)]
    out: List[Tuple[Tuple[str, str], ...]] = []
    for size in sizes:
        for held in itertools.combinations(others, size):
            for combination in itertools.product(*(list(levels_by_factor[f]) for f in held)):
                out.append(tuple(zip(held, combination)))
    return out


def _within_name(
    within: Sequence[Tuple[str, str]], ambiguous: bool, levels_by_factor: Mapping[str, Sequence[str]],
) -> str:
    return "_".join(within_label(factor, level, ambiguous, levels_by_factor) for factor, level in within)


def _spelled(within: Sequence[Tuple[str, str]]) -> str:
    return ", ".join(f"{factor} {level}" for factor, level in within)


def _listed(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def factorial_slices(
    rows: Sequence[Mapping[str, str]],
    factors: Sequence[str],
    levels_by_factor: Mapping[str, Sequence[str]],
) -> List[Tuple[str, str, str, str, List[int], List[int], str]]:
    """The two-group comparisons a factor design holds.

    (name, factor, control level, other level, control rows, other rows,
    the other factors' levels, spelled, or ""). Every level
    against the control level --factor names -- "Day:1 is correct. That would
    give 1 vs 4, 1 vs 9". A pair that does not include the control (B vs D) is
    a different run, with D named as the control. Two or more factors: each
    factor's levels against its control within each combination of the
    others' levels.
    """
    ambiguous = levels_alone_are_ambiguous(factors, levels_by_factor)
    slices: List[Tuple[str, str, str, str, List[int], List[int], str]] = []
    if len(factors) == 1:
        only = factors[0]
        control, *others = list(levels_by_factor[only])
        for level in others:
            slices.append((
                safe_name(contrast_stem(only, level, levels_by_factor, ambiguous)), only, control, level,
                _samples_at(rows, only, control), _samples_at(rows, only, level), "",
            ))
        return slices
    # Two or more factors: each factor's levels against its control within
    # each combination of the others' levels (at any number of
    # factors; the order with two is the one it always was).
    for tested in reversed(list(factors)):
        others = [f for f in factors if f != tested]
        control, *levels = list(levels_by_factor[tested])
        for level in levels:
            for within in _environments(others, levels_by_factor, every_subset=False):
                name = safe_name(
                    f"{contrast_stem(tested, level, levels_by_factor, ambiguous)}"
                    f"_within_{_within_name(within, ambiguous, levels_by_factor)}"
                )
                held = dict(within)
                slices.append((
                    name, tested, control, level,
                    _samples_where(rows, {**held, tested: control}),
                    _samples_where(rows, {**held, tested: level}),
                    _spelled(within),
                ))
    return slices


def contrast_vectors(
    factors: Sequence[str],
    levels_by_factor: Mapping[str, Sequence[str]],
    term_names: Sequence[str],
) -> List[Tuple[str, str, str, str, Tuple[Tuple[str, str], ...], np.ndarray]]:
    """The factorial ANOVA's contrasts: (name, factor, control, level, within, weights).

    ``within`` is the other factors held fixed, ((factor, level), ...), empty
    for a main effect. For every non-control level of each factor: its main
    effect (averaged over every other factor's levels) and its effect within
    each setting of every subset of the other factors, averaged over the rest
    -- with two factors, the main effect and the simple effect within each
    level of the other, as always. The contrasts are what go on to pathway
    analysis, and the simple effects are not optional: they are how a
    significant interaction is read, at any number of factors. Every main
    effect, interaction and contrast is included; the output is large by
    design.

    Each weight vector is the model's mean of the level minus that of the
    control, both at the held setting, averaged with equal weight over every
    combination of the factors not held: a difference of rows of the design
    (``stats.design_matrix`` of the combination). In a 2x2 that is
    ``A_i + 0.5 AB`` for the main effect and ``A_i + AB_ij`` within b_j. The
    ``_contrast`` suffix on a within-effect keeps it apart from the t-test
    slice that compares the same cells.
    """
    import itertools

    if len(factors) < 2:
        return []
    ambiguous = levels_alone_are_ambiguous(factors, levels_by_factor)
    columns = list(factors)

    def row(levels: Mapping[str, str]) -> np.ndarray:
        design, names, _terms = design_matrix([dict(levels)], columns, levels_by_factor)
        assert list(names) == list(term_names)
        return design[0]

    out: List[Tuple[str, str, str, str, Tuple[Tuple[str, str], ...], np.ndarray]] = []
    for tested in columns:
        others = [f for f in columns if f != tested]
        control, *levels = list(levels_by_factor[tested])
        for level in levels:
            stem = contrast_stem(tested, level, levels_by_factor, ambiguous)
            for within in _environments(others, levels_by_factor, every_subset=True):
                held = dict(within)
                free = [f for f in others if f not in held]
                settings = list(itertools.product(*(list(levels_by_factor[f]) for f in free)))
                weights = np.zeros(len(term_names))
                for setting in settings:
                    at = {**held, **dict(zip(free, setting))}
                    weights += row({**at, tested: level}) - row({**at, tested: control})
                weights /= len(settings)
                name = (f"{stem}_main_effect" if not within else
                        f"{stem}_within_{_within_name(within, ambiguous, levels_by_factor)}_contrast")
                out.append((safe_name(name), tested, control, level, tuple(within), weights))
    return out


def refuse_names_that_collide(named: Sequence[Tuple]) -> None:
    """Refuse when two different tests would be written under one name.

    An entry is ``(name, what)`` or ``(name, what, samples)``. Compared without
    regard to case: Excel and the file systems of macOS and Windows
    do not tell "ko_vs_wt" from "KO_vs_wt". The same two groups of the same
    samples under one name is one test asked for twice, not a collision.
    """
    seen: Dict[str, str] = {}
    compared: Dict[str, object] = {}
    spelled: Dict[str, str] = {}
    for name, what, *rest in named:
        samples = rest[0] if rest else None
        key = str(name).casefold()
        if key in seen and seen[key] != what:
            if samples is not None and compared.get(key) == samples:
                continue
            same = (
                f"{name!r}" if spelled[key] == name
                else f"{spelled[key]!r} and {name!r}, which differ only by case"
            )
            raise ValueError(
                f"{seen[key]} and {what} would both be written as {same}, and "
                "the second would replace the first (Excel and some file systems do "
                "not tell names apart by case). Rename a level in the sample sheet so "
                "the two names differ"
            )
        seen[key] = what
        compared[key] = samples
        spelled[key] = str(name)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """One comparison a statistic makes: what it is called, and which samples it reads."""

    name: str
    kind: str
    #: The control level and the level compared with it; None for a
    #: regression over all samples or an F-test.
    control: Optional[str]
    treatment: Optional[str]
    #: One line, written into the result files' headers.
    description: str
    #: Positions (in the plan's sample rows) of the two groups -- the samples a
    #: two-group test compares, or the cells of the design a contrast's means
    #: are taken from. Empty for an F-test.
    control_index: Tuple[int, ...] = ()
    treatment_index: Tuple[int, ...] = ()
    #: paired-t: aligned pairs (control position, treatment position, pair id).
    pairs: Tuple[Tuple[int, int, str], ...] = ()
    #: lm: the samples regressed, positions in the plan's rows.
    regressed_index: Tuple[int, ...] = ()
    #: ANOVA: the contrast's weights over the design columns, or the term an
    #: F-test tests.
    weights: Optional[np.ndarray] = None
    term: Optional[str] = None
    #: "main" (the lm fit over every sample), "slice", "term" or "contrast".
    role: str = "main"
    #: The factor a contrast or slice compares the levels of.
    factor: str = ""
    #: An F-test's direction: the contrast whose estimate gives it, "coef:<column>"
    #: for a one-column interaction, "trend" for a numeric one-way ANOVA, or
    #: None where a term of several columns has no single direction -- its
    #: contrasts carry the directions.
    direction: Optional[str] = None

    def samples(self) -> Tuple[int, ...]:
        """Every sample position this comparison's statistic is computed from."""
        if self.kind == KIND_SLOPE:
            return tuple(self.regressed_index)
        if self.pairs:
            return tuple(c for c, _t, _p in self.pairs) + tuple(t for _c, t, _p in self.pairs)
        return tuple(self.control_index) + tuple(self.treatment_index)


@dataclass
class StatisticPlan:
    """Everything one statistic computes: its comparisons, and what they share."""

    statistic: str
    data_level: Optional[str]
    comparisons: List[Comparison]
    #: The factorial ANOVA: the design (samples x columns), its column names,
    #: each term's columns and the safe name of each term's F-test.
    design: Optional[np.ndarray] = None
    term_names: Optional[List[str]] = None
    term_columns: Dict[str, List[int]] = field(default_factory=dict)
    term_safe_names: Dict[str, str] = field(default_factory=dict)
    balanced: Optional[bool] = None
    #: lm: the covariate's value per sample (positions of the plan's rows), and
    #: what it is.
    covariate_values: Optional[np.ndarray] = None
    covariate_label: str = ""
    covariate_kind: str = ""
    #: paired-t: the --pair-by column.
    pair_by: str = ""
    #: --block-by, the column whose levels (subjects) are a term of anova's and lm's
    #: model that is never tested: its columns in ``design`` (anova) and each sample's level (lm).
    block_by: str = ""
    block_columns: List[int] = field(default_factory=list)
    block_values: Optional[List[str]] = None
    #: The one-way ANOVA: one factor of three or more levels -- the factor, its
    #: levels (control first), each level's number when every level is one
    #: (None otherwise: no trend), and the plan-row positions of each level.
    one_way: bool = False
    #: The factorial anova:cell: the cells are the observations, replicates
    #: pooled.
    cells_as_n: bool = False
    factor: str = ""
    levels: List[str] = field(default_factory=list)
    level_values: Optional[np.ndarray] = None
    level_members: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def folder(self) -> str:
        return statistic_folder(self.statistic, self.data_level)


@dataclass
class DifferentialPlan:
    """Every statistic a run asked for, over one set of samples."""

    rows: List[Dict[str, str]]
    sample_ids: List[str]
    statistics: List[StatisticPlan]
    #: The --factor columns, and each one's control level.
    factors: List[str]
    levels_by_factor: Dict[str, List[str]]
    data_level: Optional[str] = None

    @property
    def baselines(self) -> Dict[str, str]:
        return {factor: levels[0] for factor, levels in self.levels_by_factor.items() if levels}

    def statistic(self, name: str) -> StatisticPlan:
        for plan in self.statistics:
            if plan.statistic == name:
                return plan
        raise KeyError(name)

    def as_record(self) -> Dict[str, object]:
        """What a checkpoint records about the plan: JSON-safe."""
        return {
            "factors": [f"{factor}:{self.baselines.get(factor, '')}" for factor in self.factors],
            "levels_by_factor": {k: list(v) for k, v in self.levels_by_factor.items()},
            "data_level": self.data_level,
            "statistics": {
                plan.folder: {
                    "statistic": plan.statistic,
                    "data_level": plan.data_level,
                    "comparisons": [
                        {"name": c.name, "kind": c.kind, "role": c.role,
                         "control": c.control, "treatment": c.treatment,
                         "description": c.description}
                        for c in plan.comparisons
                    ],
                }
                for plan in self.statistics
            },
        }


def _check_factors(rows: Sequence[Mapping[str, str]], factors: Sequence[str]) -> None:
    if len(set(factors)) != len(factors):
        repeated = next(factor for factor in factors if list(factors).count(factor) > 1)
        raise ValueError(
            f"--factor {repeated} was given twice; name two different columns "
            "of the sample sheet, or give it once"
        )
    for factor in factors:
        if not any(factor in row for row in rows):
            raise ValueError(
                f"--factor {factor}: the sample sheet has no column named {factor!r}"
            )
        for row in rows:
            if not str(row.get(factor) or "").strip():
                raise ValueError(
                    f"sample {row.get('sample_id')} has no value for factor {factor!r}. Fill "
                    "the cell, or leave the sample out of the sheet"
                )


def parse_factors(specs: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """The --factor values: the columns in order, and each column's control level."""
    columns: List[str] = []
    baselines: Dict[str, str] = {}
    for spec in specs:
        column, baseline = parse_factor(spec)
        columns.append(column)
        baselines.setdefault(column, baseline)
    return columns, baselines


def factor_levels(rows: Sequence[Mapping[str, str]], factor: str, baseline: str) -> List[str]:
    """A factor's levels, its control level first (``stats.baseline_first_levels``)."""
    levels = baseline_first_levels(rows, factor, baseline)
    if len(levels) < 2:
        raise ValueError(
            f"--factor {factor}:{baseline} has one level across the samples in this comparison "
            f"({', '.join(levels)}); a factor needs at least two"
        )
    return levels


def check_factor_validity(
    rows: Sequence[Mapping[str, str]],
    factors: Sequence[str],
    group_of: Optional[Mapping[int, str]] = None,
) -> None:
    """A factor must vary between samples, and be one value within a sample's group.

    ``factors`` are column names. ``group_of`` maps a row position to the group
    key it was aggregated from (single-cell pseudobulk); where a group's rows
    disagree on the factor, the factor is refused by name.
    """
    for factor in factors:
        if group_of:
            by_group: Dict[str, set] = {}
            for index, row in enumerate(rows):
                key = group_of.get(index)
                if key is None:
                    continue
                by_group.setdefault(key, set()).add(str(row.get(factor) or "").strip())
            mixed = sorted(key for key, values in by_group.items() if len(values) > 1)
            if mixed:
                raise FactorNotConstantError(
                    f"--factor {factor} is not one value within each sample: "
                    f"{', '.join(mixed[:5])}{' ...' if len(mixed) > 5 else ''} hold more than one. "
                    "A factor describes a sample, so every cell of a sample must carry the same "
                    "value; fix the cell metadata, or group by the column that the factor follows"
                )
    _check_factors(rows, factors)


def recorded_factors(record: object) -> List[str]:
    """The --factor values (``column:control``) a differential checkpoint records, or [].

    Bulk RNA writes them at the top, bulk ATAC and the single-cell runs inside
    their plan, and a stratified index inside each stratum's plan; the first
    list found is the run's. A checkpoint path or its parsed JSON.
    """
    import json as _json

    if isinstance(record, (str, Path)):
        try:
            record = _json.loads(Path(record).read_text())
        except (OSError, ValueError):
            return []
    pending = [record]
    while pending:
        item = pending.pop(0)
        if isinstance(item, Mapping):
            value = item.get("factors")
            if isinstance(value, (list, tuple)) and value and all(":" in str(v) for v in value):
                return [str(v) for v in value]
            pending.extend(v for v in item.values() if isinstance(v, (Mapping, list)))
        elif isinstance(item, list):
            pending.extend(v for v in item if isinstance(v, (Mapping, list)))
    return []


def display_group(factors: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """The column and control level a plot labels and orders samples by: the first --factor's.

    (None, None) when the run has no factor.
    """
    if not factors:
        return None, None
    column, control = parse_factor(str(factors[0]))
    return column, control


class FactorNotConstantError(ParalomeError, ValueError):
    """PLM-0203: a --factor holds more than one value within a pseudobulk sample."""

    def __init__(self, detail: str = "", **context: object) -> None:
        super().__init__(203, detail, **context)


class DesignError(ParalomeError, ValueError):
    """PLM-0221: the requested statistics do not fit these samples.

    A ValueError too, so a caller that validates a request early (the
    commands, before a stage is queued) catches it as it always did, and a
    ParalomeError, so one raised mid-run reaches the console as a numbered
    usage error rather than PLM-0001's internal-error report.
    """

    def __init__(self, detail: str = "", **context: object) -> None:
        super().__init__(221, detail, **context)


def plan_differential(
    rows: Sequence[Mapping[str, str]],
    *,
    request: StatisticRequest,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    covariate_feature_values: Optional[np.ndarray] = None,
    group_of: Optional[Mapping[int, str]] = None,
    block_by: Optional[str] = None,
) -> "DifferentialPlan":
    """Every comparison the requested statistics make of these samples (see _plan_differential).

    ``factors`` are --factor values as typed, ``column:control``. A request the
    samples cannot support raises DesignError (PLM-0221).
    """
    try:
        return _plan_differential(
            rows, request=request, factors=factors, pair_by=pair_by, covariate=covariate,
            covariate_feature=covariate_feature, covariate_feature_values=covariate_feature_values,
            group_of=group_of, block_by=block_by,
        )
    except ParalomeError:
        raise
    except ValueError as error:
        raise DesignError(str(error)) from error


def _plan_one_way(
    plan: "StatisticPlan", rows: Sequence[Mapping[str, str]], factor: str,
    levels: Sequence[str], levels_by_factor: Mapping[str, Sequence[str]], request: StatisticRequest,
) -> None:
    """The one-way ANOVA: one factor of three or more levels.

    The omnibus F-test; for numeric levels also its linear trend on the
    levels' values and the departure from it (lack of fit), the trend giving
    the F its direction. Its contrasts: each level against the control level
    --factor names (before, each level against the one before it).
    At ``:cell`` the cells are the observations.
    """
    levels = list(levels)
    if len(levels) < 3:
        raise ValueError(
            f"--statistic anova with one --factor fits a one-way ANOVA, which needs three or "
            f"more levels; {factor} has two ({', '.join(levels)}). Two levels are the "
            "two-group comparison: use --statistic t"
        )
    numbers = level_numbers(levels)
    members = [tuple(_samples_at(rows, factor, level)) for level in levels]
    n_levels = len(levels)
    if request.data_level != DATA_LEVEL_CELL and len(rows) - n_levels < 1:
        raise ValueError(
            f"the one-way ANOVA of {factor} has {len(rows)} sample(s) in {n_levels} levels, which "
            "leaves no degrees of freedom for the error: no F or p can be computed. Add a "
            "second sample to at least one level, or fit it with the cells as the observations "
            "(--statistic anova:cell)"
        )
    plan.one_way = True
    plan.factor = factor
    plan.levels = levels
    plan.level_values = None if numbers is None else np.asarray(numbers, dtype=float)
    plan.level_members = members
    plan.balanced = len({len(m) for m in members}) == 1
    spelled = ", ".join(levels)
    plan.comparisons.append(Comparison(
        name=safe_name(f"{factor}_one_way"), kind=KIND_FTEST, control=levels[0], treatment=None,
        description=(f"one-way ANOVA F-test across {factor} ({spelled})"
                     + (f", with its linear trend on {factor} and the departure from that trend "
                        "(lack of fit); direction from the trend" if numbers is not None
                        else "; the direction of each level is its contrast's")),
        term=factor, role="term", factor=factor,
        direction="trend" if numbers is not None else None,
    ))
    control_index = members[0]
    for level, alt_index in zip(levels[1:], members[1:]):
        stem = contrast_stem(factor, level, levels_by_factor, False)
        plan.comparisons.append(Comparison(
            name=safe_name(f"{stem}_contrast"),
            kind=KIND_CONTRAST, control=levels[0], treatment=level,
            description=(f"ANOVA contrast, {factor} {level} versus {factor} {levels[0]} "
                         "(pooled-error estimate from the one-way model)"),
            control_index=tuple(control_index), treatment_index=tuple(alt_index), role="contrast",
            factor=factor,
        ))


def _plan_differential(
    rows: Sequence[Mapping[str, str]],
    *,
    request: StatisticRequest,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    covariate_feature_values: Optional[np.ndarray] = None,
    group_of: Optional[Mapping[int, str]] = None,
    block_by: Optional[str] = None,
) -> DifferentialPlan:
    """Decide every comparison every requested statistic makes, and refuse what cannot be fitted.

    ``block_by``: a sample-sheet
    column -- the subject of a repeated-measures design -- whose levels are a term of the model of ``anova`` and
    ``lm`` that is fitted but never tested, so each subject's samples are compared with each other. ``t`` and
    ``paired-t`` do not take it (paired-t pairs by --pair-by); a request holding neither anova nor lm is refused.

    ``rows`` are the samples in scope (every sample of the run, after any
    stratum narrowing). The groups are the --factor columns' levels, each
    compared with the control level its --factor names:
    there is no other way to say what a group is.

    Per statistic:

    * ``t`` and ``paired-t``: each level of a factor against its control
      level; with more factors, within each combination of the others' levels. paired-t pairs
      the samples of each comparison by --pair-by.
    * ``lm``: the regression over every sample, and within each level of each
      --factor.
    * ``anova``: one factor, the one-way ANOVA; two or more, the factorial
      ANOVA with every interaction in the model: an
      F-test per term and the contrasts.
    """
    rows = [dict(row) for row in rows]
    columns, baselines = parse_factors(factors)
    statistics = list(request.statistics)
    if not rows:
        raise ValueError("no samples are in scope for this comparison")
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    two_group = [s for s in statistics if s in ("t", "paired-t")]
    if (two_group or "anova" in statistics) and not columns:
        raise ValueError(
            f"--statistic {','.join(two_group or ['anova'])} compares groups, and the groups are a "
            "--factor: name the sample-sheet column and its control level, e.g. --factor Diet:CD"
        )

    if columns:
        check_factor_validity(rows, columns, group_of)
    levels_by_factor: Dict[str, List[str]] = {f: factor_levels(rows, f, baselines[f]) for f in columns}
    for factor, levels in levels_by_factor.items():
        for position, reference in enumerate(levels):
            for alternate in levels[position + 1:]:
                if (safe_name(reference).casefold() == safe_name(alternate).casefold()
                        or not safe_name(reference) or not safe_name(alternate)):
                    raise ValueError(
                        f"levels {reference!r} and {alternate!r} of factor {factor!r} are written "
                        f"the same way ({safe_name(reference)!r}) in file and column names, where "
                        "only letters and digits are kept and case is not told apart. Rename one "
                        "level in the sample sheet"
                    )
    if len(columns) >= 2:
        import itertools

        for combination in itertools.product(*(levels_by_factor[f] for f in columns)):
            if not _samples_where(rows, dict(zip(columns, combination))):
                spelled = " with ".join(f"{f}={level}" for f, level in zip(columns, combination))
                raise ValueError(
                    f"no sample has {spelled}. A design of {len(columns)} factors needs samples in "
                    "every combination of their levels; add them, or leave a --factor out"
                )

    def fewest_samples(statistic: str) -> int:
        """Samples a group of a two-group comparison needs.

        Two, except t with the cells as the observations: t:cell's N is the
        cells (hundreds to thousands), so a group of one sample is testable
        there, and the run records that its sample is confounded with its
        group.
        """
        return 1 if (statistic == "t" and request.data_level == DATA_LEVEL_CELL) else 2

    blocks: Optional[List[str]] = None
    if block_by:
        if not any(s in ("anova", "lm") for s in statistics):
            raise ValueError(
                f"--block-by {block_by} is a term of the anova and lm models; --statistic "
                f"{','.join(statistics)} has neither (paired-t pairs by --pair-by)"
            )
        if request.data_level == DATA_LEVEL_CELL:
            raise ValueError(f"--block-by {block_by} blocks samples; with :cell the cells are the observations")
        missing = [sample_ids[i] for i, row in enumerate(rows) if not str(row.get(block_by) or "").strip()]
        if missing:
            raise ValueError(
                f"--block-by {block_by}: {len(missing)} sample(s) have no value in that column "
                f"({', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''})"
            )
        blocks = [str(row.get(block_by)).strip() for row in rows]
        if len(set(blocks)) == len(blocks):
            raise ValueError(
                f"--block-by {block_by}: every sample has its own {block_by}, so no two samples share a "
                "block and the term removes nothing but the error"
            )

    def add_blocks(design: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """The block's columns, one per level after the first, appended to ``design``."""
        if blocks is None:
            return design, []
        levels = list(dict.fromkeys(blocks))
        extra = [np.asarray([1.0 if b == level else 0.0 for b in blocks]) for level in levels[1:]]
        start = design.shape[1]
        return np.column_stack([design, *extra]), list(range(start, start + len(extra)))

    slices = factorial_slices(rows, columns, levels_by_factor) if columns else []
    plans: List[StatisticPlan] = []
    named: List[Tuple] = []
    ambiguous = levels_alone_are_ambiguous(columns, levels_by_factor)

    for statistic in statistics:
        comparisons: List[Comparison] = []
        plan = StatisticPlan(statistic=statistic, data_level=request.data_level, comparisons=comparisons)
        if statistic in ("t", "paired-t"):
            for name, factor, reference, alternate, ci, ti, within in slices:
                what = f"{factor} {alternate} versus {factor} {reference}" + (f", {within} only" if within else "")
                if min(len(ci), len(ti)) < fewest_samples(statistic):
                    raise ValueError(
                        f"{name} ({what}) has {len(ci)} and {len(ti)} sample(s); each group "
                        "needs at least two"
                        + (" (t:cell, at least one)" if fewest_samples(statistic) == 1 else "")
                    )
                pairs: Tuple[Tuple[int, int, str], ...] = ()
                if statistic == "paired-t":
                    subset = [rows[i] for i in ci + ti]
                    labels = {str(rows[i]["sample_id"]): reference for i in ci}
                    labels.update({str(rows[i]["sample_id"]): alternate for i in ti})
                    pair_ids, c_local, t_local = resolve_pairs(
                        subset, labels, str(pair_by), reference, alternate,
                    )
                    positions = list(ci) + list(ti)
                    pairs = tuple(
                        (positions[c], positions[t], pid)
                        for pid, c, t in zip(pair_ids, c_local, t_local)
                    )
                    if len(pairs) < 2:
                        raise ValueError(
                            f"paired-t: {name} ({what}) has {len(pairs)} complete pair(s) by "
                            f"--pair-by {pair_by}; a paired test needs at least two"
                        )
                comparisons.append(Comparison(
                    name=name, kind=KIND_CONTRAST, control=reference, treatment=alternate,
                    description=what, control_index=tuple(ci), treatment_index=tuple(ti),
                    pairs=pairs, role="slice", factor=factor,
                ))
                named.append((name, f"{statistic} {what}", (tuple(ci), tuple(ti))))
        elif statistic == "lm":
            label = str(covariate or covariate_feature)
            if covariate:
                values = numeric_covariate(rows, str(covariate))
                plan.covariate_kind = "sample_column"
            else:
                if covariate_feature_values is None:
                    raise ValueError(
                        f"--covariate-feature {covariate_feature!r} was not found in this "
                        "matrix under any of its identifier columns"
                    )
                values = np.asarray(covariate_feature_values, dtype=float)
                plan.covariate_kind = "feature"
            plan.covariate_values = values
            plan.covariate_label = label
            if blocks is not None:
                plan.block_by, plan.block_values = str(block_by), list(blocks)
            stem = safe_name(f"{label}_slope")
            subsets: List[Tuple[str, List[int], str, str]] = [
                (stem, list(range(len(rows))), "main", f"regression on {label}, every sample"),
            ]
            for factor in columns:
                # A factor that is the --covariate column itself holds one
                # covariate value per level: no regression within a level.
                if covariate and factor == covariate:
                    continue
                for level in levels_by_factor[factor]:
                    subsets.append((
                        safe_name(f"{stem}_within_{within_label(factor, level, ambiguous, levels_by_factor)}"),
                        _samples_at(rows, factor, level), "slice",
                        f"regression on {label}, {factor} {level} only",
                    ))
            for name, members, role, what in subsets:
                x = values[members]
                if len(members) < 3:
                    raise ValueError(
                        f"lm: {name} ({what}) has {len(members)} sample(s); a regression needs "
                        "at least three (two parameters and one residual df)"
                    )
                if np.ptp(x) == 0:
                    raise ValueError(
                        f"lm: {name} ({what}): {label} is the same value in every sample, so "
                        "there is nothing to regress against"
                    )
                comparisons.append(Comparison(
                    name=name, kind=KIND_SLOPE, control=None, treatment=None,
                    description=what, regressed_index=tuple(members), role=role,
                ))
                named.append((name, f"lm {what}"))
        elif statistic == "anova" and len(columns) == 1 and blocks is not None:
            # One factor within subjects: the factorial fit (one term) with the block's columns, so the
            # F and each level's contrast come from the model that holds the subjects.
            factor = columns[0]
            levels = levels_by_factor[factor]
            design, names, term_columns = design_matrix(rows, columns, levels_by_factor)
            design, block_columns = add_blocks(design)
            _refuse_terms_inside_blocks(design, term_columns, block_columns, str(block_by), len(rows))
            plan.design, plan.term_names, plan.term_columns = design, names, term_columns
            plan.block_by, plan.block_columns = str(block_by), block_columns
            plan.balanced = is_balanced(rows, columns)
            safe = safe_name(f"{factor}_one_way")
            plan.term_safe_names[factor] = safe
            comparisons.append(Comparison(
                name=safe, kind=KIND_FTEST, control=levels[0], treatment=None,
                description=(f"ANOVA F-test across {factor} ({', '.join(levels)}), blocked by {block_by}"),
                term=factor, role="term", factor=factor,
                direction=f"coef:{term_columns[factor][0]}" if len(levels) == 2 else None,
            ))
            named.append((safe, f"anova {factor} blocked by {block_by}"))
            for position, level in enumerate(levels[1:]):
                weights = np.zeros(design.shape[1])
                weights[term_columns[factor][position]] = 1.0
                name = safe_name(f"{contrast_stem(factor, level, levels_by_factor, False)}_contrast")
                comparisons.append(Comparison(
                    name=name, kind=KIND_CONTRAST, control=levels[0], treatment=level,
                    description=(f"ANOVA contrast, {factor} {level} versus {factor} {levels[0]}, within "
                                 f"{block_by} (estimate from the model blocked by {block_by})"),
                    control_index=tuple(_samples_at(rows, factor, levels[0])),
                    treatment_index=tuple(_samples_at(rows, factor, level)),
                    weights=weights, role="contrast", factor=factor,
                ))
                named.append((name, f"anova {factor} {level} versus {levels[0]} blocked"))
        elif statistic == "anova" and len(columns) == 1:
            _plan_one_way(plan, rows, columns[0], levels_by_factor[columns[0]], levels_by_factor, request)
            for comparison in plan.comparisons:
                named.append((comparison.name, f"anova {comparison.description}"))
        elif statistic == "anova":
            design, names, term_columns = design_matrix(rows, columns, levels_by_factor)
            width = design.shape[1]
            design, block_columns = add_blocks(design)
            if block_columns:
                _refuse_terms_inside_blocks(design, term_columns, block_columns, str(block_by), len(rows))
                plan.block_by, plan.block_columns = str(block_by), block_columns
            # There are only two models for Seurat data, pseudobulk and cell. At :cell the cells are the
            # observations, replicates or not: when there are replicates, their cells are pooled as they
            # are in Seurat.
            plan.cells_as_n = request.data_level == DATA_LEVEL_CELL
            if not plan.cells_as_n and model_error_df(design, len(rows)) < 1:
                raise ValueError(
                    f"the ANOVA of {_listed(columns)} has {len(rows)} sample(s) and "
                    f"{design.shape[1]} terms to estimate, which leaves no degrees of freedom "
                    "for the error: no F or p can be computed. Add a second sample to at least "
                    "one combination of levels, or fit it with the cells as the observations "
                    "(--statistic anova:cell)"
                )
            plan.design = design
            plan.term_names = names
            plan.term_columns = term_columns
            plan.balanced = is_balanced(rows, columns)
            contrasts = contrast_vectors(columns, levels_by_factor, names)
            if block_columns:     # the block's columns carry no weight in any contrast
                contrasts = [(n, f, c, l, w, np.concatenate([v, np.zeros(design.shape[1] - width)]))
                             for n, f, c, l, w, v in contrasts]
            main_effect_of = {
                factor: name for name, factor, _c, _l, within, _w in contrasts
                if not within and name.endswith("_main_effect")
            }
            for term, term_cols in term_columns.items():
                if ":" not in term:
                    levels = levels_by_factor[term]
                    safe = term_name(term, levels_by_factor, ambiguous)
                    what = (f"ANOVA F-test, main effect of {term} ({', '.join(levels[1:])} versus "
                            f"{levels[0]})")
                    direction = main_effect_of.get(term) if len(levels) == 2 else None
                else:
                    safe = safe_name(term)
                    what = f"ANOVA F-test, interaction of {_listed(term.split(':'))}"
                    direction = f"coef:{term_cols[0]}" if len(term_cols) == 1 else None
                plan.term_safe_names[term] = safe
                comparisons.append(Comparison(
                    name=safe, kind=KIND_FTEST, control=None, treatment=None,
                    description=what, term=term, role="term", direction=direction,
                    factor=term if ":" not in term else "",
                ))
                named.append((safe, f"anova {what}"))
            for name, factor, reference, alternate, within, weights in contrasts:
                held = dict(within)
                free = [f for f in columns if f != factor and f not in held]
                ref_idx = _samples_where(rows, {**held, factor: reference})
                alt_idx = _samples_where(rows, {**held, factor: alternate})
                description = f"ANOVA contrast, {factor} {alternate} versus {factor} {reference}"
                if within:
                    description += f" within {_spelled(within)}"
                if free:
                    description += f"{',' if within else ''} averaged over {_listed(free)}"
                model = "two-way" if len(columns) == 2 else f"{len(columns)}-factor"
                comparisons.append(Comparison(
                    name=name, kind=KIND_CONTRAST, control=reference, treatment=alternate,
                    description=description + f" (pooled-error estimate from the full {model} model)",
                    control_index=tuple(ref_idx), treatment_index=tuple(alt_idx),
                    weights=weights, role="contrast", factor=factor,
                ))
                named.append((name, f"anova {description}"))
        else:  # pragma: no cover - parse_statistic_request refuses it first
            raise ValueError(f"unknown statistic {statistic!r}")
        if statistic == "paired-t":
            plan.pair_by = str(pair_by)
        plans.append(plan)

    # Within ONE statistic a name must be one test. Across statistics the same
    # name is expected -- t and paired-t both make KO_vs_WT -- and they are
    # written to different folders.
    for plan in plans:
        refuse_names_that_collide([
            entry for entry in named
            if entry[1].split(" ", 1)[0] == plan.statistic
        ])

    return DifferentialPlan(
        rows=rows, sample_ids=sample_ids, statistics=plans,
        factors=columns, levels_by_factor=levels_by_factor,
        data_level=request.data_level,
    )


def _refuse_terms_inside_blocks(design: np.ndarray, term_columns: Mapping[str, Sequence[int]],
                                block_columns: Sequence[int], block_by: str, n_samples: int) -> None:
    """A term the blocks already hold -- a factor whose level never changes within a block (a between-subject
    factor, blocked by subject) -- cannot be tested beside them: refused by name, not reported as no effect."""
    width = design.shape[1]
    full_rank = int(np.linalg.matrix_rank(design))
    for term, columns in term_columns.items():
        without = [i for i in range(width) if i not in set(columns)]
        if int(np.linalg.matrix_rank(design[:, without])) == full_rank:
            raise ValueError(
                f"--block-by {block_by}: {term} does not change within any {block_by} (each {block_by} "
                f"has one level of it), so the {block_by} term already holds it and it cannot be tested. "
                f"Leave out --block-by, or leave out that --factor"
            )
    if n_samples - full_rank < 1:
        raise ValueError(
            f"--block-by {block_by}: the model with every {block_by} has {full_rank} parameters for "
            f"{n_samples} samples, which leaves no degrees of freedom for the error"
        )


def is_cell_level(plan: DifferentialPlan) -> bool:
    return plan.data_level == DATA_LEVEL_CELL
