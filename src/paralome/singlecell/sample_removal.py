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

"""Resolving ``--remove`` values to pseudobulk samples.

Dropping samples (two bad donors seen in the PCAs, say) is by whatever their
column header is, not by hand-transcribing every composite pseudobulk
sample_id (``<donor>__<cell_type>__<condition>``) a donor has.

So a ``--remove`` value is resolved against, in order:

1. the sample-ID column -- ``--replicate-column`` when the caller has one
   (``pipeline run`` does), otherwise the first ``--group-by`` column;
2. failing that, an exact composite ``sample_id``.

Values are comma-split, matching ``--group-by``: ``--remove
'donor_A_POS,donor_B_POS'`` is two values.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


ARM_LABEL_SEPARATOR = ":"


class ComparisonArm:
    """One ``--color-by`` group: a label plus its member values.

    A bare comma form would make ``LOW,MED,HIGH`` mean "arm LOW containing MED
    and HIGH", which reads as three members and fails silently inside an array
    script, so the label has an explicit separator: ``POS:LOW,MED,HIGH``.

    Accepted forms:
      ``HIGH``                       label HIGH, one member
      ``LOW,MED,HIGH``               three members, label auto-derived LOW_MED_HIGH
      ``POS:LOW,MED,HIGH``           label POS, three members
      ``POS:donor_B_POS,donor_C_POS``   label POS, two donors
    """

    __slots__ = ("label", "members", "raw")

    def __init__(self, label: str, members: Sequence[str], raw: str) -> None:
        self.label = label
        self.members = list(members)
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ComparisonArm(label={self.label!r}, members={self.members!r})"

    def as_dict(self) -> Dict[str, object]:
        """Recorded in stage results/checkpoints so the pooling stays visible."""
        return {"label": self.label, "members": list(self.members), "spec": self.raw}


def parse_comparison_arm(spec: str, *, flag: str) -> ComparisonArm:
    """Parse one ``LABEL:members`` value into a labelled group."""
    text = str(spec).strip()
    if not text:
        raise ValueError(f"{flag} must not be empty")

    if ARM_LABEL_SEPARATOR in text:
        label, _, member_text = text.partition(ARM_LABEL_SEPARATOR)
        label = label.strip()
        if not label:
            raise ValueError(f"{flag} {spec!r} has an empty label before {ARM_LABEL_SEPARATOR!r}")
        members = split_remove_values([member_text])
        if not members:
            raise ValueError(f"{flag} {spec!r} names label {label!r} but no members after it")
    else:
        members = split_remove_values([text])
        if not members:
            raise ValueError(f"{flag} {spec!r} names no values")
        # One value keeps its own name, exactly as.
        label = members[0] if len(members) == 1 else "_".join(members)

    return ComparisonArm(label=label, members=members, raw=text)


def resolve_arm_samples(
    arm: ComparisonArm,
    rows: Sequence[Mapping[str, object]],
    *,
    group_by: Sequence[str],
    replicate_column: Optional[str] = None,
    sample_id_key: str = "sample_id",
    absent_members: Optional[List[str]] = None,
) -> Set[str]:
    """Which pseudobulk sample_ids belong to this arm.

    A member matches, in order: a value of the sample-ID column
    (``--replicate-column``, else the first ``--group-by`` column); or a full
    composite ``sample_id``. One group may mix them.

    ``absent_members`` switches a member that matches nothing from an ERROR to
    a reported ABSENCE, and is passed only when ``rows`` has been narrowed to a
    single stratum: the absent member is dropped, reported, and the stratum
    runs.

    The distinction is real. Over the whole manifest, a member matching nothing
    is a typo and must fail loudly. Within ONE stratum it is the routine
    consequence of ``--min-cells-per-group``: a donor that contributed 4 CD16
    Mono cells, below the floor of 10, has no row in the CD16 Mono slice at all
    -- while existing perfectly well in the other strata. And
    strata.run_stratified_fan_out resolves the arms against the FULL manifest
    to decide which strata to skip, so the per-stratum run must accept what
    that check accepted. Absences are appended here and surfaced by the caller
    so the n behind each stratum stays visible.
    """
    id_column = resolve_sample_id_column(group_by, replicate_column)
    matched: Set[str] = set()
    unknown: List[str] = []

    for member in arm.members:
        hits: Set[str] = set()
        if id_column:
            hits = {
                str(row[sample_id_key]) for row in rows
                if str(row.get(id_column, "")) == member
            }
        if not hits:
            hits = {
                str(row[sample_id_key]) for row in rows
                if str(row[sample_id_key]) == member
            }
        if not hits:
            unknown.append(member)
            continue
        matched |= hits

    if unknown:
        if absent_members is not None:
            absent_members.extend(unknown)
        else:
            # The message lists the spellings that work (a sample-ID value, or a full pseudobulk sample_id of
            # this grouping), not only the one that was wrong: donor__celltype__condition from a three-column
            # --group-by, donor__celltype from a two-column one, and the bare donor, which resolves per stratum
            # inside a fan-out, are all plausible.
            identifiers = sorted({str(row.get(id_column, "")) for row in rows}) if id_column else []
            lines = [
                f"Arm {arm.label!r} member(s) matched nothing: {', '.join(unknown)}.",
                "Accepted spellings, in the order they are tried:",
            ]
            if identifiers:
                lines.append(
                    f"  1. a value of {id_column!r}: {', '.join(identifiers[:12])}"
                    + (f" (+{len(identifiers) - 12} more)" if len(identifiers) > 12 else "")
                )
            lines.append("  2. a full pseudobulk sample_id, e.g. "
                         + ", ".join(sorted(str(r[sample_id_key]) for r in rows)[:3]))
            # The near-miss that actually happened: a value that is a PREFIX of
            # real ids, i.e. the right sample named at the wrong granularity.
            for bad in unknown:
                prefixes = sorted({
                    value for value in identifiers
                    if value and bad.startswith(value) and value != bad
                })
                if prefixes:
                    lines.append(
                        f"Did you mean {prefixes[0]!r}? {bad!r} starts with it, "
                        f"then adds more. Inside a --stratify-by fan-out the "
                        f"{id_column!r} value is usually what you want: it resolves "
                        f"to that sample within EVERY stratum, where a full "
                        f"sample_id names one stratum and matches nothing in the rest."
                    )
            raise ValueError("\n".join(lines))
    return matched


COLOR_GROUP_COLUMN = "color_group"
UNASSIGNED_LABEL = "unassigned"


def resolve_color_groups(
    color_by: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    group_by: Sequence[str] = (),
    replicate_column: Optional[str] = None,
    sample_id_key: str = "sample_id",
) -> Optional[Dict[str, object]]:
    """Synthesize an ad-hoc coloring column from ``LABEL:members`` specs.

    Subgroups of donors found from a PCA are a finding that by definition does
    not exist in the Seurat object, so there is no column to colour by. Rather
    than a sample-annotation file format, ``--color-by`` takes the delimited
    expression (``POS:LOW,MED,HIGH``), with no new option.

    So ``--color-by`` is repeatable, and:
      - a bare value is a COLUMN NAME;
      - any value containing ``:`` defines a group, and all such values
        together build one temporary ``color_group`` column.

    Returns None when the caller passed a plain column name (nothing to
    synthesize). Mixing the two forms is rejected rather than guessed at.

    Members resolve, in this order: a value of the sample-ID column, or a full
    composite ``sample_id``. Samples in no group become ``unassigned`` rather
    than being dropped -- a donor left out by accident should be visible in the
    plot, not silently absent from it.
    """
    specs = [str(value).strip() for value in color_by if str(value).strip()]
    if not specs:
        return None

    with_label = [spec for spec in specs if ARM_LABEL_SEPARATOR in spec]
    if not with_label:
        return None
    if len(with_label) != len(specs):
        plain = [spec for spec in specs if ARM_LABEL_SEPARATOR not in spec]
        # Say how to express what was meant, not only what is forbidden: the flag is repeatable, not
        # space-separated, and once any value is LABEL:members they all must be -- and a label may equal its
        # own member (HIGH:HIGH is legal).
        rewrite = " \\\n  ".join(
            [f"--color-by {spec!r}" for spec in with_label]
            + [f"--color-by {value}{ARM_LABEL_SEPARATOR}{value}" for value in plain]
        )
        raise ValueError(
            "--color-by mixes a column name with ad-hoc group definitions: "
            f"{', '.join(plain)} vs {', '.join(with_label)}.\n"
            "A bare value names a metadata COLUMN, so it cannot sit beside "
            "LABEL:members groups. Members are sample-ID values or full "
            "pseudobulk sample_ids; when the bare values are sample IDs, a label "
            "may be the same word as its member:\n\n"
            f"  {rewrite}\n\n"
            "To colour by the values of a column (a treatment, say), pass that "
            "column's name on its own. Remember --color-by "
            "is repeatable rather than space-separated: '--color-by A B C' "
            "gives the flag only 'A'."
        )

    id_column = resolve_sample_id_column(group_by, replicate_column)
    assignments: Dict[str, str] = {}
    groups: List[Dict[str, object]] = []
    overlaps: List[str] = []

    for spec in with_label:
        arm = parse_comparison_arm(spec, flag="--color-by")
        members = resolve_arm_samples(
            arm, rows,
            group_by=group_by, replicate_column=replicate_column,
            sample_id_key=sample_id_key,
        )
        for sample_id in sorted(members):
            if sample_id in assignments and assignments[sample_id] != arm.label:
                overlaps.append(f"{sample_id} ({assignments[sample_id]} and {arm.label})")
            assignments[sample_id] = arm.label
        groups.append({"label": arm.label, "members": arm.members, "n_samples": len(members)})

    if overlaps:
        raise ValueError(
            "--color-by group(s) claim the same sample twice: " + "; ".join(sorted(overlaps))
        )

    unassigned = [
        str(row[sample_id_key]) for row in rows if str(row[sample_id_key]) not in assignments
    ]
    return {
        "column": COLOR_GROUP_COLUMN,
        "assignments": assignments,
        "groups": groups,
        "unassigned": sorted(unassigned),
        "sample_id_column": id_column,
    }


def resolve_sample_id_column(
    group_by: Sequence[str], replicate_column: Optional[str] = None
) -> Optional[str]:
    """Which metadata column identifies a biological sample.

    ``--replicate-column`` is primary, because a caller that
    has one has said outright which column that is. The first ``--group-by``
    column is the fallback convention for the standalone commands, which have
    no such flag.
    """
    if replicate_column:
        return replicate_column
    return group_by[0] if group_by else None


def split_remove_values(remove: Optional[Iterable[str]]) -> List[str]:
    """Flatten and comma-split the repeatable ``--remove`` option.

    ``--remove a,b --remove c`` and ``--remove a --remove b --remove c`` are
    equivalent. Consistent with ``--group-by``, which has always comma-split.
    """
    values: List[str] = []
    for entry in remove or []:
        for piece in str(entry).split(","):
            cleaned = piece.strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return values


def resolve_removals(
    remove: Optional[Iterable[str]],
    rows: Sequence[Mapping[str, object]],
    *,
    group_by: Sequence[str],
    replicate_column: Optional[str] = None,
    sample_id_key: str = "sample_id",
) -> Tuple[Set[str], Dict[str, object]]:
    """Expand ``--remove`` values into the concrete ``sample_id``s they name.

    ``rows`` is the pseudobulk sample manifest -- either the in-memory
    ``manifest_rows`` during aggregation, or the DataFrame records read back
    from ``sample_manifest.csv`` at PCA time.

    Returns ``(sample_ids_to_drop, report)``. Raises ValueError naming the
    sample-ID column's distinct values -- NOT every composite sample_id, which
    is what made the previous error unreadable at 194 samples.
    """
    requested = split_remove_values(remove)
    if not requested:
        return set(), {"requested": [], "matched": {}, "sample_id_column": None}

    id_column = resolve_sample_id_column(group_by, replicate_column)
    all_sample_ids = {str(row[sample_id_key]) for row in rows}
    column_values: Set[str] = set()
    if id_column and rows and id_column in rows[0]:
        column_values = {str(row.get(id_column)) for row in rows}

    matched: Dict[str, List[str]] = {}
    to_drop: Set[str] = set()
    unknown: List[str] = []
    for value in requested:
        if value in column_values:
            hits = sorted(
                str(row[sample_id_key]) for row in rows if str(row.get(id_column)) == value
            )
        elif value in all_sample_ids:
            # Back-compat: a full composite sample_id still resolves to itself.
            hits = [value]
        else:
            unknown.append(value)
            continue
        matched[value] = hits
        to_drop.update(hits)

    if unknown:
        if column_values:
            available = ", ".join(sorted(column_values))
            hint = (
                f"Values of the sample-ID column {id_column!r} are: {available}. "
                "A full pseudobulk sample_id also works."
            )
        else:
            available = ", ".join(sorted(all_sample_ids))
            hint = f"Available pseudobulk sample_id(s): {available}"
        raise ValueError(
            f"--remove value(s) not found: {', '.join(sorted(unknown))}. {hint}"
        )

    if len(to_drop) == len(all_sample_ids):
        raise ValueError("--remove excluded every pseudobulk sample; nothing left to analyze")

    return to_drop, {
        "requested": requested,
        "matched": {key: value for key, value in sorted(matched.items())},
        "sample_id_column": id_column,
    }
