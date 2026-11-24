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

"""Did alpha and the expression filter represent this data honestly?

An unbiased QC of the results: it illustrates whether the alpha
and filtering strategies produced an accurate representation of the data.

THIS IS AN AUDIT, AND GOVERNS IT. It runs after the differential
checkpoint is written, contributes nothing to any downstream stage, and cannot
fail, delay or block the run it is checking. Every entry point here swallows
its own errors: a QC that breaks an analysis is worse than no QC at all.

IT REPORTS; IT NEVER ACTS. Outlying animals are named with what excluding them
would do, and nothing else happens: report only, never change it. Excluding
an animal is a judgement the analyst makes by editing the sample sheet, not
something a diagnostic does on their behalf.

WHY THE PERMUTATION NULL IS NOT OPTIONAL
----------------------------------------
Every headline statistic here -- the p-value histogram, the significant-gene
count, the responder score, the leave-one-out gain -- was computed during this
feature's design WITHOUT a null, and every one of them produced a confident,
wrong answer on a real data set:

  "~331 genes are genuinely non-null"     -> permutation p = 0.22
  "the diet effect is confined to VEH"    -> both strata inside the null
  "47 genes reversed, 100% opposite"      -> shared-cell artifact; baseline 67%
  "metformin retains 32% of the response" -> winner's curse; symmetric at 0.91
  "this test is a broken thermometer"     -> permutation p = 0.91, mid-null

The cause is the same every time. Genes are co-regulated, so 4,472 of them
carry about as much independent evidence as 133 -- the permutation spread of
the significant fraction is nearly six times wider than independence predicts.
A statistic read against a binomial expectation is therefore read against an
interval several times too narrow, and lands "significant" on noise.

So the null is computed by shuffling the group labels, which respects whatever
correlation structure the data actually has because it never touches the genes.
Anything this module reports as notable has cleared that null. Nothing is
reported as notable on the strength of a gene count alone.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Every statistic this audit computes -- the permutation
# nulls, pi0, the effective number of tests, the responder score, the
# leave-one-out gain, the array-quality rule -- is in paralome.qc_stats, with
# every other QC statistic. This module reads the run, chooses the comparisons
# and writes the pages.
from paralome.qc_stats import (  # noqa: E402
    DEFAULT_ALPHA,
    PERMUTATIONS,
    PERMUTATIONS_PER_ANIMAL,
    TIME_BUDGET_SECONDS,
    array_quality,
    audit_comparison,
    responder_scores,
    significant,
    student_t_p,
    variance_runaway,
)

#: A comparison needs enough animals for a leave-one-out to mean anything.
MIN_PER_GROUP = 3

METADATA_FIELDS = ("gene_id", "gene_name", "chromosome", "start", "end",
                   "strand", "length_bp")


# ---------------------------------------------------------------------------
# Reading the run back
# ---------------------------------------------------------------------------

def factor_specs(factors: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """The run's --factor values as columns and control levels.

    The audit orients every comparison the way the run does: against the
    control level --factor names, never by which level the sheet lists first.
    """
    from paralome.comparisons import parse_factors

    return parse_factors([f for f in factors if f])


def level_order_for(groups: Mapping[str, Tuple[str, ...]], columns: Sequence[str],
                    baselines: Mapping[str, str]) -> List[List[str]]:
    """Each factor's levels, its control level first, then as the sheet gives them."""
    order: List[List[str]] = []
    for position, column in enumerate(columns):
        seen: List[str] = []
        for key in groups.values():
            if key[position] not in seen:
                seen.append(key[position])
        from paralome.stats import control_choices

        # A --factor listing several controls orders by the first of them present.
        baseline = next((c for c in control_choices(baselines.get(column)) if c in seen), None)
        order.append(([baseline] if baseline in seen else []) + [lvl for lvl in seen if lvl != baseline])
    return order


def read_sample_groups(samples_path: Path, factors: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """sample_id -> its cell, as a tuple of factor levels."""
    from paralome.csvio import skip_comment_lines

    groups: Dict[str, Tuple[str, ...]] = {}
    # Utf-8-sig and comment lines, as the run itself reads this sheet, and the run's own alias resolution
    # (a sheet headed "sample" or "Sample_ID"), so the audit sees the samples the run tested.
    from paralome.rna.differential import _resolve_sheet_aliases

    with Path(samples_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            for row in csv.DictReader(skip_comment_lines(handle))
        ]
    for row in _resolve_sheet_aliases(rows):
        sample = (row.get("sample_id") or "").strip()
        if not sample:
            continue
        key = tuple(str(row.get(f) or "").strip() for f in factors)
        if all(key):
            groups[sample] = key
    return groups


def run_statistics_tables(differential_dir: Path) -> Tuple[Dict[str, object], List[Path]]:
    """The finished result's checkpoint, and the machine table of every comparison it wrote.

    The result is one folder per statistic, each comparison with its
    machine table (``results/<folder>/machine/..._statistics_rna.csv``), listed
    in the checkpoint's ``statistics`` manifests. Raises ``ValueError`` with the
    reason when there is no finished result of that layout to read.
    """
    import json as _json

    from paralome.checkpoints import stage_checkpoint_path

    path = stage_checkpoint_path(Path(differential_dir), "differential_expression")
    try:
        checkpoint = _json.loads(path.read_text())
    except (OSError, ValueError):
        raise ValueError(f"no differential checkpoint at {path}") from None
    statistics = checkpoint.get("statistics") if isinstance(checkpoint, Mapping) else None
    tables: List[Path] = []
    if isinstance(statistics, Mapping):
        for manifest in statistics.values():
            comparisons = manifest.get("comparisons") if isinstance(manifest, Mapping) else None
            for entry in (comparisons or {}).values():
                if isinstance(entry, Mapping) and entry.get("statistics"):
                    tables.append(Path(str(entry["statistics"])))
    if not tables:
        raise ValueError(f"the checkpoint {path} lists no statistics tables")
    missing = [table for table in tables if not table.is_file()]
    if missing:
        raise ValueError(f"no statistics table at {missing[0]}")
    return dict(checkpoint), tables


def read_expression_columns(
    differential_dir: Path,
    sample_ids: Sequence[str],
    expression_matrix: Optional[Path] = None,
) -> Tuple[List[str], np.ndarray]:
    """The log2 expression (samples x genes) of the genes the result tested.

    The genes, in order, are those of the result's machine tables
    (every comparison's table holds every gene of the run); their values are
    the log2(FPKM + pseudocount) the tests ran on, read from the run's
    ``--expression`` (``expression_matrix``, else the matrix the checkpoint
    records). Reading the matrix does not change it. Samples the matrix does
    not hold are left out; ``ValueError`` says why nothing can be read.
    """
    checkpoint, tables = run_statistics_tables(Path(differential_dir))
    with tables[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        genes = [str(row.get("gene_id") or "") for row in reader]
    source = expression_matrix or checkpoint.get("expression_matrix")
    if not source or not Path(str(source)).is_file():
        raise ValueError(f"the expression matrix of the result is not available ({source})")
    with Path(str(source)).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        present = [s for s in sample_ids if s in (reader.fieldnames or [])]
        if not present:
            return [], np.empty((0, 0))
        by_gene: Dict[str, List[float]] = {}
        for row in reader:
            values = []
            for sample in present:
                try:
                    values.append(float(row[sample]))
                except (TypeError, ValueError):
                    values.append(float("nan"))
            by_gene.setdefault(str(row.get("gene_id") or ""), values)
    absent = [gene for gene in genes if gene not in by_gene]
    if absent:
        raise ValueError(
            f"{len(absent)} gene(s) of the result are not in {source} (e.g. {absent[0]}); "
            "the matrix is not the one the result was computed from"
        )
    matrix = np.asarray([by_gene[gene] for gene in genes], dtype=float).T
    return present, matrix


def _cell_label(key) -> str:
    """A cell's levels as text, for a reader. Never parsed back."""
    return key if isinstance(key, str) else "/".join(str(level) for level in key)


# ---------------------------------------------------------------------------
# The pages
# ---------------------------------------------------------------------------

READING_GUIDE = [
    ("A  Where the filter cuts",
     "You want the red line on the shoulder of the distribution, not inside the tall spike "
     "and not out in the flat tail. Inside the spike you are keeping noise; far right and you "
     "are shearing off expressed genes."),
    ("B  Variance runaway check",
     "Read right to left. If the curve CLIMBS as expression falls, tiny means are dragging "
     "their variance down with them and manufacturing significance -- the filter is protecting "
     "you. Flat or falling at the left means that pathology is not operating here and the "
     "filter is costing genes for nothing."),
    ("C  Power vs n",
     "Find your n. The solid line is the smallest fold change you could have detected; "
     "anything smaller was invisible whatever the pipeline did. The dashed line is your actual "
     "power. Below 0.8, a negative result says 'could not see', not 'not there'."),
    ("D  p-value histogram",
     "A real effect pushes bars ABOVE the grey band at the left. Bars inside the band across "
     "the whole width mean nothing is distinguishable from no effect. A histogram that RISES "
     "left to right is a warning about the test, not about biology."),
    ("E  Does alpha mean what it says?",
     "The grey band is what chance alone produces on THIS data. Your line inside it means "
     "alpha is honest and p<0.05 means what it claims. Outside it -- especially below -- means "
     "the threshold is not delivering the risk you are paying for."),
    ("F  Filter stability",
     "Look only near your red line. High and flat means the exact cut-off is not driving your "
     "gene list. A dip means the threshold is load-bearing and no single value is defensible "
     "on its own."),
    ("G  Is the result outside the null?",
     "The most important panel. If the red line sits inside the grey pile, this run found "
     "nothing -- a real answer, not a failure. Only a red line clear of the pile is evidence."),
    ("H  How much evidence is really there?",
     "The red bar is how many genuinely independent tests you have. Co-regulated genes do not "
     "each count as separate evidence. When it sits far below the blue, any conclusion phrased "
     "as 'N genes changed' overstates what the data supports."),
    ("I  Array quality",
     "Bars should be tight and similar. A short flagged bar is an animal whose profile does not "
     "resemble its cell-mates -- suspect the sample, library or sequencing before the biology."),
    ("J  Responder projection",
     "Each dot is a treated animal, scored by how far the treatment moved the group, coloured "
     "by responsiveness. Dots in the grey band did not move. A dot below the red cutoff moved "
     "the wrong way entirely -- usually dose or handling, not biology."),
    ("K  Leave-one-out influence",
     "How much the result changes if you delete the single most inconvenient animal. This "
     "almost always looks dramatic, which is why the grey interval is there: it shows what "
     "deleting an ARBITRARY animal already does. A bar inside it is small-n instability, "
     "not an outlier effect."),
]

RULE_OF_THUMB = (
    "Panels G and H come first. If G says the result is inside the null, nothing in D, E or F "
    "can rescue it and no filter or alpha change will either -- the honest next step is more "
    "animals, not more analysis. Only once G clears the null do the calibration and filter "
    "panels decide which gene list to report. J and K are read last and separately: an animal "
    "is worth excluding only if it clears its own permutation cutoff, and this audit will "
    "never exclude one for you."
)


def write_qc_interactive(
    out_dir: Path,
    *,
    expression: np.ndarray,
    cells: Mapping[str, Sequence[int]],
    sample_ids: Sequence[str],
    audits: Mapping[str, Dict[str, object]],
) -> Dict[str, str]:
    """Panels I and J as interactive Plotly pages, with sample IDs on hover.

    For looking at high responders versus low responders. A static panel can say THAT an animal is unusual; only hover
    can say WHICH.

    These are the two PER-ANIMAL panels and only those. The rest of the
    ensemble is about genes, where thousands of hover targets would be noise
    rather than information.

    ``include_plotlyjs="cdn"`` matches pca_2d.html and sample_outlier_qc_2d.html,
    which this sits beside. It keeps the files small at the cost of needing a
    network when opened.

    Returns {name: path} for whatever it managed. Never raises: this is part of
    an audit, and cannot take a finished run down with it.
    """
    written: Dict[str, str] = {}
    try:
        from paralome import plotting as px
    except Exception:  # pragma: no cover - plotly absent
        return written

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(figure, path: Path) -> None:
        """Write via a temporary, so an interrupted audit never leaves a
        half-written page that looks like a finished one."""
        import os

        temporary = path.with_suffix(path.suffix + ".tmp")
        figure.write_html(str(temporary), include_plotlyjs="cdn")
        os.replace(temporary, path)
        written[path.stem] = str(path)

    # ---- J: responder projection -------------------------------------------
    try:
        rows: List[Dict[str, object]] = []
        for label, audit in audits.items():
            cutoff = float(audit["responder_cutoff"])
            for index, score in audit["responder_z"].items():
                rows.append({
                    "sample": sample_ids[index],
                    "comparison": label,
                    "score": round(float(score), 3),
                    "cutoff": round(cutoff, 2) if np.isfinite(cutoff) else None,
                    "cell": next((_cell_label(k) for k, v in cells.items() if index in v), "?"),
                    "status": ("outlier" if np.isfinite(cutoff) and score <= cutoff
                               else "no response" if score < 1.0 else "responder"),
                })
        if rows:
            figure = px.scatter(
                x=[r["comparison"] for r in rows],
                y=[r["score"] for r in rows],
                color=[r["score"] for r in rows],
                symbol=[r["status"] for r in rows],
                color_continuous_scale="RdYlGn", range_color=(-6, 3),
                labels={"x": "", "y": "SD above control mean, on the treatment axis",
                        "color": "responsiveness"},
                hover_name=[r["sample"] for r in rows],
                hover_data={"cell": [r["cell"] for r in rows],
                            "status": [r["status"] for r in rows],
                            "cutoff": [r["cutoff"] for r in rows]},
                title="Responder projection -- hover for the animal",
            )
            figure.update_traces(marker=dict(size=13, line=dict(width=1, color="#555")))
            # The control band and each comparison's own permutation cutoff, so
            # the page carries the same reference lines as the printed panel.
            figure.add_hrect(y0=-1, y1=1, fillcolor="#999", opacity=.22, line_width=0)
            for position, (label, audit) in enumerate(audits.items()):
                cutoff = float(audit["responder_cutoff"])
                if np.isfinite(cutoff):
                    figure.add_shape(type="line", x0=position - .4, x1=position + .4,
                                     y0=cutoff, y1=cutoff,
                                     line=dict(color="#d03b3b", width=2, dash="dash"))
            _save(figure, out_dir / "qc_responders.html")
    except Exception:  # pragma: no cover
        pass

    # ---- I: array quality -------------------------------------------------
    try:
        quality = array_quality(expression, cells)
        order = list(np.argsort(quality["median_r"]))
        figure = px.bar(
            x=[float(quality["median_r"][i]) for i in order],
            y=[sample_ids[i] for i in order],
            color=[float(quality["robust_z"][i]) for i in order],
            color_continuous_scale="RdYlBu", range_color=(-5, 2),
            orientation="h",
            labels={"x": "median correlation to cell-mates", "y": "",
                    "color": "outlierness (MAD z)"},
            hover_data={"cell": [next((_cell_label(k) for k, v in cells.items() if i in v), "?")
                                 for i in order]},
            title="Array quality -- hover for the animal",
        )
        figure.add_vline(x=float(quality["cutoff_r"]), line_dash="dash",
                         line_color="#d03b3b",
                         annotation_text="3.5 MAD", annotation_position="top")
        figure.update_layout(height=max(420, 22 * len(order)))
        _save(figure, out_dir / "qc_array_quality.html")
    except Exception:  # pragma: no cover
        pass

    return written


def write_qc_pdf(
    out_pdf: Path,
    *,
    expression: np.ndarray,
    cells: Mapping[str, Sequence[int]],
    sample_ids: Sequence[str],
    audits: Mapping[str, Dict[str, object]],
    gate: float,
    alpha: float,
    genes_total: int,
    genes_gated: int,
    project: str = "",
    pseudocount: float = 1.0,
    alpha_metric: str = "p",
    unusable: Optional[Mapping[str, str]] = None,
    primary_pair: Optional[Tuple[str, str]] = None,
) -> Optional[str]:
    """Three pages. Returns the path, or None if plotting is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:  # pragma: no cover - headless environments without mpl
        return None
    import textwrap

    from paralome.stats import minimum_detectable_d, observed_power

    def ttl(ax, title, subtitle=None):
        ax.set_title(title, fontsize=9.5, loc="left", pad=14)
        if subtitle:
            ax.annotate(subtitle, xy=(0, 1.0), xytext=(0, 4), xycoords="axes fraction",
                        textcoords="offset points", fontsize=7.2, color="#666", va="bottom")

    names = list(audits)
    primary = audits[names[0]]
    cell_means = np.vstack([expression[list(v)].mean(axis=0) for v in cells.values()])
    linear = 2 ** cell_means.max(axis=0) - pseudocount
    metric = "q" if str(alpha_metric).lower() == "q" else "p"

    figures = []
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(
        f"Paralome QC ensemble  ·  {project}  ·  gate {gate:g} FPKM, alpha "
        f"{alpha if metric == 'p' else f'q<{alpha}'}  ·  "
        f"{genes_gated:,} of {genes_total:,} genes retained", fontsize=11, y=0.985)
    gs = fig.add_gridspec(3, 3, hspace=0.58, wspace=0.30,
                          left=.06, right=.97, top=.92, bottom=.07)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(np.log2(linear + 1), bins=90, color="#2a78d6", alpha=.85)
    ax.axvline(np.log2(gate + 1), color="#d03b3b", ls="--", lw=1.3)
    ax.text(np.log2(gate + 1) + .2, ax.get_ylim()[1] * .88, f"{gate:g} FPKM",
            color="#d03b3b", fontsize=8)
    ax.set_xlabel("log2(FPKM+1)", fontsize=8); ax.set_ylabel("genes", fontsize=8)
    ttl(ax, "A  Where the filter cuts",
        f"{100*genes_gated/max(1,genes_total):.0f}% retained; look for the antimode")

    ax = fig.add_subplot(gs[0, 1])
    sd = np.vstack([expression[list(v)].std(axis=0, ddof=1) for v in cells.values()]).mean(axis=0)
    order = np.argsort(linear); step = max(1, len(order) // 60)
    bx = [np.median(linear[order[i:i+step]]) for i in range(0, len(order), step)]
    by = [np.median(sd[order[i:i+step]]) for i in range(0, len(order), step)]
    ax.plot(bx, by, color="#199e70", lw=1.4); ax.set_xscale("log")
    ax.axvline(gate, color="#d03b3b", ls="--", lw=1.3)
    ax.set_xlabel("expression (FPKM, log)", fontsize=8); ax.set_ylabel("median SD", fontsize=8)
    ttl(ax, "B  Variance runaway check", "SD climbing at the left = the gate is earning its keep")

    ax = fig.add_subplot(gs[0, 2])
    gated = np.where(linear >= gate)[0]
    pooled_sd = float(np.median(sd[gated])) if gated.size else float(np.median(sd))
    ns = np.arange(2, 26); mdes = []; power = []
    # Panels C and F use the primary audited pair (cells differing
    # in one factor), not the first and last cell, which differ in every factor.
    if primary_pair is not None:
        a_idx = cells[primary_pair[0]]; b_idx = cells[primary_pair[1]]
    else:
        a_idx = list(cells.values())[0]; b_idx = list(cells.values())[-1]
    med_eff = float(np.median(np.abs(expression[list(b_idx)][:, gated].mean(axis=0)
                                     - expression[list(a_idx)][:, gated].mean(axis=0))))
    # The package's power arithmetic (paralome.stats), where this
    # panel computed its own non-central t.
    for n in ns:
        mdes.append(minimum_detectable_d(n, n, alpha=alpha) * pooled_sd)
        effect = med_eff / pooled_sd if pooled_sd else 0.0
        power.append(observed_power(effect, n, n, alpha))
    ax.plot(ns, mdes, color="#2a78d6", lw=1.6)
    ax.axvline(min(len(v) for v in cells.values()), color="#d03b3b", ls="--", lw=1.3)
    ax.set_xlabel("n per group", fontsize=8); ax.set_ylabel("log2 fold change", fontsize=8)
    twin = ax.twinx(); twin.plot(ns, power, color="#eda100", lw=1.6, ls="--")
    twin.set_ylim(0, 1); twin.axhline(.8, color="#eda100", ls=":", lw=1)
    twin.set_ylabel("power at median effect", fontsize=8, color="#c98500")
    ttl(ax, "C  Power vs n", "solid: detectable effect   dashed: power at your median effect")

    centres = np.arange(20) * .05 + .025
    ax = fig.add_subplot(gs[1, 0])
    nh = primary["null_histograms"]
    ax.fill_between(centres, np.percentile(nh, 2.5, axis=0), np.percentile(nh, 97.5, axis=0),
                    color="#bbb", alpha=.55, step="mid", label="permutation null 95%")
    ax.bar(centres, primary["observed_histogram"], width=.048, color="#eb6834")
    ax.set_xlabel("p-value", fontsize=8); ax.set_ylabel("genes", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ttl(ax, "D  p-value histogram", "bars inside the grey band = indistinguishable from no effect")

    ax = fig.add_subplot(gs[1, 1])
    grid = primary["grid"]; nc_ = primary["null_calibration"]
    ax.fill_between(grid, np.percentile(nc_, 2.5, axis=0), np.percentile(nc_, 97.5, axis=0),
                    color="#bbb", alpha=.55)
    ax.plot(grid, primary["observed_calibration"], "o-", color="#eb6834", lw=1.5, ms=3.5)
    ax.axhline(1, color="#555", ls="--", lw=1.2); ax.set_xscale("log")
    ax.set_xlabel("nominal alpha", fontsize=8); ax.set_ylabel("delivered / promised", fontsize=8)
    ttl(ax, "E  Does alpha mean what it says?", "grey = what chance alone produces on THIS data")

    ax = fig.add_subplot(gs[1, 2])
    nf = primary["null_fractions"] * 100
    ax.hist(nf, bins=40, color="#888", alpha=.8)
    ax.axvline(primary["observed_fraction"] * 100, color="#d03b3b", lw=2)
    ax.axvline(alpha * 100, color="#2a78d6", ls="--", lw=1.2)
    ax.text(.97, .88, f"observed {primary['observed_fraction']*100:.2f}%\n"
            f"= {primary['fraction_percentile']:.0f}th pct of null",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#d03b3b")
    ax.set_xlabel(f"% genes {metric}<{alpha} under permuted labels", fontsize=8)
    ax.set_ylabel("permutations", fontsize=8)
    ttl(ax, "G  Is the result outside the null?",
        "red = your data. If it sits in the pile, there is nothing.")

    ax = fig.add_subplot(gs[2, 0])
    ths = [0, .5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    pfull = student_t_p(expression[list(a_idx)], expression[list(b_idx)])
    tested = np.where(np.isfinite(pfull))[0]
    called = np.zeros(pfull.shape, dtype=bool)
    called[tested] = significant(pfull[tested], alpha, metric)
    prev = None; jac = []
    for t in ths:
        keep = set(np.where((linear >= t) & called)[0])
        jac.append(float("nan") if prev is None
                   else len(keep & prev) / max(1, len(keep | prev)))
        prev = keep
    ax.plot(ths, jac, "o-", color="#6250d6", lw=1.5, ms=3.5)
    ax.axvline(gate, color="#d03b3b", ls="--", lw=1.3); ax.set_ylim(0, 1.05)
    ax.set_xlabel("min_expression (FPKM)", fontsize=8)
    ax.set_ylabel("Jaccard vs previous", fontsize=8)
    ttl(ax, "F  Filter stability", "near 1.0 = the exact cut-off is not driving your gene list")

    ax = fig.add_subplot(gs[2, 1])
    neff = primary["effective_tests"]
    ax.bar(["assumed\n(independent)", "actual\n(correlated)"], [primary["genes"], neff],
           color=["#2a78d6", "#d03b3b"]); ax.set_yscale("log")
    ax.set_ylabel("effective independent tests", fontsize=8)
    for i, v in enumerate([primary["genes"], neff]):
        if np.isfinite(v):
            ax.text(i, v * 1.3, f"{v:,.0f}", ha="center", fontsize=8.5)
    ax.set_ylim(50, primary["genes"] * 3)
    widen = (math.sqrt(float(np.var(primary["null_fractions"])))
             / math.sqrt(alpha * (1 - alpha) / max(1, primary["genes"])))
    ttl(ax, "H  How much evidence is really there?",
        f"spread {widen:.1f}x wider than independence predicts")

    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    i05 = list(grid).index(0.05) if 0.05 in list(grid) else 4
    cal_lo = float(np.percentile(nc_[:, i05], 2.5)); cal_hi = float(np.percentile(nc_[:, i05], 97.5))
    cal_ok = cal_lo <= primary["observed_calibration"][i05] <= cal_hi
    near = [j for t, j in zip(ths, jac) if abs(t - gate) <= 1.5 and j == j]
    lines = [
        ("Alpha calibration",
         f"{primary['observed_calibration'][i05]:.2f}x at {alpha}, null band {cal_lo:.2f}-{cal_hi:.2f}\n"
         + ("within the null - alpha is honest" if cal_ok else "OUTSIDE the null - suspect")),
        ("Filter position", f"Jaccard {min(near):.2f} near {gate:g} FPKM" if near else "n/a"),
        ("Variance runaway", {None: "n/a", False: "not present", True: "present at low expression"}[
            variance_runaway(bx, by, gate)]),
        ("Effective tests", f"~{neff:,.0f} of {primary['genes']:,} genes"),
        ("Verdict", "NO effect distinguishable from null" if primary["inside_null"]
         else "signal exceeds the null"),
    ]
    ax.text(0, 1.0, "Read-out", fontsize=10, va="top"); y = .90
    for label, body in lines:
        ax.text(0, y, label, fontsize=8, color="#777", va="top"); y -= .055
        for ln in body.split("\n"):
            ax.text(0, y, ln, fontsize=8.5, va="top",
                    color="#d03b3b" if (label == "Verdict" and primary["inside_null"]) else "#111")
            y -= .058
        y -= .035
    figures.append(fig)

    # ------------------------------- page 2 -------------------------------
    quality = array_quality(expression, cells)
    f2 = plt.figure(figsize=(11.7, 8.3))
    f2.suptitle("Paralome QC ensemble — page 2: per-animal outliers and non-responders",
                fontsize=11, y=0.985)
    g2 = f2.add_gridspec(2, 3, hspace=0.55, wspace=0.28,
                         left=.07, right=.97, top=.90, bottom=.09)

    ax = f2.add_subplot(g2[0, 0])
    order = np.argsort(quality["median_r"])
    cmap_i = plt.cm.RdYlBu; norm_i = mcolors.Normalize(vmin=-5, vmax=2)
    ax.barh(range(len(order)), quality["median_r"][order],
            color=[cmap_i(norm_i(quality["robust_z"][i])) for i in order])
    ax.axvline(quality["cutoff_r"], color="#d03b3b", ls="--", lw=1.3)
    ax.text(quality["cutoff_r"], len(order) + .5, "3.5 MAD", color="#d03b3b",
            fontsize=7, ha="center", va="bottom")
    ax.set_yticks([])
    ax.set_xlim(min(float(np.min(quality["median_r"])), quality["cutoff_r"]) - .003,
                float(np.max(quality["median_r"])) + .002)
    for k, i in enumerate(order[:3]):
        ax.annotate(sample_ids[i][-12:], (quality["median_r"][i], k), xytext=(-6, 0),
                    textcoords="offset points", fontsize=6.5, ha="right", va="center", color="#444")
    cb = f2.colorbar(plt.cm.ScalarMappable(norm=norm_i, cmap=cmap_i), ax=ax, pad=.02, fraction=.045)
    cb.set_label("outlierness (MAD z)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("median r to cell-mates", fontsize=8); ax.set_ylabel("animals (sorted)", fontsize=8)
    ttl(ax, "I  Array quality",
        f"range {np.min(quality['median_r']):.3f}-{np.max(quality['median_r']):.3f}; "
        + (f"{len(quality['flagged'])} beyond 3.5 MAD" if quality["flagged"] else "none beyond 3.5 MAD"))

    ax = f2.add_subplot(g2[0, 1:])
    cmap_j = plt.cm.RdYlGn; norm_j = mcolors.Normalize(vmin=-6, vmax=3)
    for k, name in enumerate(names):
        a = audits[name]; z = a["responder_z"]; cut = a["responder_cutoff"]
        ax.fill_between([k - .36, k + .36], -1, 1, color="#bbb", alpha=.32, zorder=0)
        if np.isfinite(cut):
            ax.plot([k - .36, k + .36], [cut, cut], color="#d03b3b", ls="--", lw=1.4, zorder=2)
            ax.annotate(f"cut {cut:.1f}", (k + .36, cut), xytext=(3, -9),
                        textcoords="offset points", fontsize=6.5, color="#d03b3b")
        if not z:
            continue
        jitter = np.linspace(-.13, .13, len(z))
        for (idx, value), dx in zip(sorted(z.items(), key=lambda t: t[1]), jitter):
            bad = value <= cut
            ax.scatter(k + dx, value, s=62 if bad else 30, color=cmap_j(norm_j(value)),
                       zorder=4, edgecolor="#8b1a1a" if bad else "none", linewidth=1.2)
            if bad:
                ax.annotate(sample_ids[idx][-12:], (k + dx, value), xytext=(10, -3),
                            textcoords="offset points", fontsize=7.5, color="#8b1a1a")
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("SD above control mean, on the treatment axis", fontsize=8)
    cb = f2.colorbar(plt.cm.ScalarMappable(norm=norm_j, cmap=cmap_j), ax=ax, pad=.015, fraction=.035)
    cb.set_label("responsiveness", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ttl(ax, "J  Responder projection",
        "grey band = control range; red dashes = permutation cutoff for a true outlier")

    ax = f2.add_subplot(g2[1, 0:2])
    xs = np.arange(len(names))
    ax.bar(xs - .18, [audits[n]["loo_gain"] for n in names], width=.36,
           color="#6250d6", label="observed best gain")
    med = [float(np.median(audits[n]["loo_null"])) for n in names]
    lo = [m - float(np.percentile(audits[n]["loo_null"], 2.5)) for m, n in zip(med, names)]
    hi = [float(np.percentile(audits[n]["loo_null"], 97.5)) - m for m, n in zip(med, names)]
    ax.errorbar(xs + .18, med, yerr=[lo, hi], fmt="o", color="#888", capsize=4,
                label="null: drop an arbitrary animal")
    for k, n in enumerate(names):
        ax.text(k, max(audits[n]["loo_gain"], med[k] + hi[k]) + 6,
                f"p={audits[n]['loo_p']:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(f"extra genes at {metric}<{alpha}", fontsize=8)
    ax.legend(fontsize=6.8, loc="upper right")
    ttl(ax, "K  Leave-one-out influence", "a bar inside the grey interval is NOT an outlier effect")

    ax = f2.add_subplot(g2[1, 2]); ax.axis("off")
    ax.text(0, 1.0, "Flagged animals", fontsize=10, va="top"); y = .89
    any_flag = False
    for name in names:
        a = audits[name]
        for idx in a["flagged"]:
            any_flag = True
            ax.text(0, y, sample_ids[idx], fontsize=8.5, color="#d03b3b", va="top"); y -= .058
            ax.text(0, y, f"{name} · {a['responder_z'][idx]:+.2f} SD · perm p={a['responder_p']:.3f}",
                    fontsize=7.8, color="#444", va="top"); y -= .085
    for idx in quality["flagged"]:
        any_flag = True
        ax.text(0, y, sample_ids[idx], fontsize=8.5, color="#d03b3b", va="top"); y -= .058
        ax.text(0, y, f"array quality · {quality['robust_z'][idx]:+.1f} MAD",
                fontsize=7.8, color="#444", va="top"); y -= .085
    for sample_id in sorted(unusable or {}):
        any_flag = True
        ax.text(0, y, sample_id, fontsize=8.5, color="#d03b3b", va="top"); y -= .058
        ax.text(0, y, "no expression measured · left out of every panel",
                fontsize=7.8, color="#444", va="top"); y -= .085
    if not any_flag:
        ax.text(0, y, "none beyond the permutation null", fontsize=8.5, va="top"); y -= .09
    ax.text(0, y - .03, "Influence read-out", fontsize=9, va="top"); y -= .11
    for name in names:
        a = audits[name]
        ax.text(0, y, f"{name}: {a['loo_base']} genes, best drop {a['loo_gain']:+d} "
                f"(p={a['loo_p']:.2f})", fontsize=7.6,
                color="#111" if a["loo_p"] < .05 else "#666", va="top"); y -= .062
    ax.text(0, y - .03, "This audit never excludes an animal.", fontsize=7.6,
            color="#777", va="top")
    figures.append(f2)

    # ------------------------------- page 3 -------------------------------
    f3 = plt.figure(figsize=(11.7, 8.3))
    f3.suptitle("Paralome QC ensemble — page 3: what to read from each panel",
                fontsize=11, y=.975)

    def column(ax, items, tail=None):
        yy = 1.0
        for title, body in items:
            ax.text(0, yy, title, fontsize=9, va="top", color="#111"); yy -= 0.042
            for ln in textwrap.wrap(body, 62):
                ax.text(0, yy, ln, fontsize=8, va="top", color="#444"); yy -= 0.033
            yy -= 0.022
        if tail:
            ax.text(0, yy - 0.012, tail[0], fontsize=9, va="top", color="#111"); yy -= 0.056
            for ln in textwrap.wrap(tail[1], 62):
                ax.text(0, yy, ln, fontsize=8, va="top", color="#444"); yy -= 0.033

    left = f3.add_axes([0.055, 0.05, 0.42, 0.88]); left.axis("off")
    right = f3.add_axes([0.535, 0.05, 0.42, 0.88]); right.axis("off")
    column(left, READING_GUIDE[:6])
    column(right, READING_GUIDE[6:], ("Rule of thumb", RULE_OF_THUMB))
    figures.append(f3)

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)
    return str(out_pdf)


# ---------------------------------------------------------------------------
# The entry point the pipeline calls
# ---------------------------------------------------------------------------

def _audited_cells(
    matrix: np.ndarray,
    sample_ids: Sequence[str],
    groups: Mapping[str, Tuple[str, ...]],
) -> Tuple[np.ndarray, List[str], Dict[Tuple[str, ...], List[int]], Dict[str, str], Dict[str, str]]:
    """The animals the audit can speak about, and the ones it cannot.

    * A sample with no expression at all (every gene the same value) has no
      correlation with anything, and would turn the array-quality panel into
      NaN. It is a finding about that sample, returned as ``unusable``, and
      the audit goes on without it.
    * A cell with fewer than MIN_PER_GROUP animals is not compared, and its
      animals would otherwise have a correlation of zero to "cell-mates" they
      do not have and be flagged as bad arrays. They are returned as
      ``not_audited`` and taken out of the matrix.

    Returns (matrix, sample_ids, cells, unusable, not_audited), the matrix and
    the cells' indexes restricted to the animals that are left.
    """
    unusable: Dict[str, str] = {}
    not_audited: Dict[str, str] = {}
    cells: Dict[Tuple[str, ...], List[int]] = {}
    for index, sample in enumerate(sample_ids):
        if matrix.shape[1] and float(np.ptp(matrix[index])) == 0.0:
            unusable[sample] = (
                "every gene has the same value: no expression was measured for this "
                "sample, and it is left out of the audit"
            )
            continue
        cells.setdefault(tuple(groups[sample]), []).append(index)
    for key in [key for key, members in cells.items() if len(members) < MIN_PER_GROUP]:
        for index in cells.pop(key):
            not_audited[sample_ids[index]] = (
                f"its cell {_cell_label(key)} has fewer than {MIN_PER_GROUP} usable animals"
            )
    kept = sorted(index for members in cells.values() for index in members)
    position = {old: new for new, old in enumerate(kept)}
    return (
        matrix[kept],
        [sample_ids[index] for index in kept],
        {key: [position[index] for index in members] for key, members in cells.items()},
        unusable,
        not_audited,
    )


def _pairwise_cells(
    cells: Mapping[Tuple[str, ...], Sequence[int]],
    level_order: Optional[Sequence[Sequence[str]]] = None,
) -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
    """Cell pairs differing in exactly one factor -- the simple effects.

    A 2x2 gives four. Pairs differing in BOTH factors are skipped: nothing in
    the differential output tests them, and reporting a comparison the run did
    not make would be inventing a result inside an audit.

    ORIENTED THE WAY THE RUN ORIENTS IT: against the control level, so every
    comparison in the report reads in the same direction.

    A cell is its TUPLE of levels, so a level that holds a "/" itself
    (``+/+``, ``KO/het``) is one level, not two.
    """
    rank: List[Dict[str, int]] = []
    if level_order:
        rank = [{level: i for i, level in enumerate(levels)} for levels in level_order]

    keys = sorted(cells)
    pairs = []
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            a, b = tuple(left), tuple(right)
            if len(a) != len(b):
                continue
            differing = [k for k in range(len(a)) if a[k] != b[k]]
            if len(differing) != 1:
                continue
            axis = differing[0]
            control, treated = left, right
            if axis < len(rank) and rank[axis]:
                if rank[axis].get(b[axis], 0) < rank[axis].get(a[axis], 0):
                    control, treated = right, left
            ca, cb = tuple(control), tuple(treated)
            # As the run compares -- each level against the control
            # level only, never two non-control levels with each other.
            if axis < len(rank) and rank[axis] and rank[axis].get(ca[axis], 0) != 0:
                continue
            held = "/".join(v for k, v in enumerate(ca) if k != axis)
            label = f"{cb[axis]} vs {ca[axis]}"
            pairs.append((f"{label} in {held}" if held else label, control, treated))
    return pairs


def run_qc_ensemble(
    differential_dir: Path,
    samples_path: Path,
    *,
    factors: Sequence[str],
    min_expression: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    permutations: int = PERMUTATIONS,
    animal_permutations: int = PERMUTATIONS_PER_ANIMAL,
    time_budget: float = TIME_BUDGET_SECONDS,
    seed: int = 0,
    alpha_metric: str = "p",
    pseudocount: float = 1.0,
    expression_matrix: Optional[Path] = None,
) -> Dict[str, object]:
    """Audit a finished differential run. NEVER RAISES.

    ``alpha``/``alpha_metric`` and ``pseudocount`` are the run's own --alpha
    (cutoff and P or Q) and --pseudocount; raw p < 0.05 and a pseudocount of 1
    are the defaults. Its comparisons are Student's t between cells, whatever
    the run's --statistic: it audits the alpha and the gate, on the cells of
    the design.

    The genes are the ones the run's machine tables hold, their
    values read from ``expression_matrix`` (else the matrix the checkpoint
    records) -- see ``read_expression_columns``.

    This is an audit, not a result. It is called after the checkpoint
    is written and its return value gates nothing. A failure here -- no
    matplotlib, a sample sheet that moved, a design too small to permute -- must
    leave the analysis exactly as it was, so every path out of this function is
    a dictionary, never an exception.
    """
    report: Dict[str, object] = {"complete": False, "reason": None, "pdf": None}
    out_dir = Path(differential_dir) / "results" / "qc"

    def _record(payload: Dict[str, object]) -> Dict[str, object]:
        """Leave the reason somewhere a human will actually find it.

        results/qc/ is created before the audit decides whether to proceed, so
        every early return and every crash leaves a record of why. A component
        whose contract is "fails quietly" must still be able to say that it
        did.
        """
        try:
            import json as _json
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "qc_report.json").write_text(_json.dumps(
                {k: v for k, v in payload.items() if k != "effective_tests"},
                indent=2, default=str) + "\n")
        except Exception:
            pass
        if not payload.get("complete"):
            print(f"  QC ensemble skipped: {payload.get('reason')}", flush=True)
        return payload

    try:
        differential_dir = Path(differential_dir)
        try:
            run_statistics_tables(differential_dir)
        except ValueError as error:
            report["reason"] = str(error)
            return _record(report)
        if not Path(samples_path).exists():
            report["reason"] = f"sample sheet not found: {samples_path}"
            return _record(report)
        columns, baselines = factor_specs(factors)
        if not columns:
            report["reason"] = "no factors: nothing to audit"
            return _record(report)

        groups = read_sample_groups(samples_path, columns)
        # The control level --factor names comes first, as in the run.
        level_order = level_order_for(groups, columns, baselines)
        if not groups:
            report["reason"] = "sample sheet carries none of the run's factors"
            return _record(report)
        try:
            sample_ids, matrix = read_expression_columns(
                differential_dir, list(groups), expression_matrix)
        except ValueError as error:
            report["reason"] = str(error)
            return _record(report)
        if matrix.size == 0 or len(sample_ids) < 4:
            report["reason"] = "per-sample expression columns not present"
            return _record(report)
        good = ~np.isnan(matrix).any(axis=0)
        matrix = matrix[:, good]
        genes_total = int(matrix.shape[1])

        matrix, sample_ids, cells, unusable, not_audited = _audited_cells(
            matrix, sample_ids, groups)
        report["unusable_samples"] = unusable
        report["samples_not_audited"] = not_audited
        for sample, why in unusable.items():
            print(f"  QC ensemble: {sample}: {why}", flush=True)
        if len(cells) < 2:
            report["reason"] = f"fewer than two cells with {MIN_PER_GROUP}+ animals"
            return _record(report)

        cell_means = np.vstack([matrix[v].mean(axis=0) for v in cells.values()])
        linear = 2 ** cell_means.max(axis=0) - float(pseudocount)
        gated = np.where(linear >= min_expression)[0]
        if gated.size < 50:
            report["reason"] = f"only {gated.size} genes pass min_expression={min_expression:g}"
            return _record(report)
        gated_matrix = matrix[:, gated]

        pairs = _pairwise_cells(cells, level_order)
        share = max(10.0, float(time_budget) / max(1, len(pairs)))
        audits: Dict[str, Dict[str, object]] = {}
        for offset, (label, left, right) in enumerate(pairs):
            audits[label] = audit_comparison(
                gated_matrix, cells[left], cells[right], alpha=alpha,
                permutations=permutations, animal_permutations=animal_permutations,
                time_budget=share, seed=seed + offset,
                alpha_metric=alpha_metric,
            )
        if not audits:
            report["reason"] = "no cell pair differs in exactly one factor"
            return _record(report)

        pdf = write_qc_pdf(
            out_dir / "paralome_qc_ensemble.pdf",
            expression=matrix, cells=cells, sample_ids=sample_ids, audits=audits,
            gate=min_expression, alpha=alpha, genes_total=genes_total,
            genes_gated=int(gated.size), project=differential_dir.name,
            pseudocount=float(pseudocount),
            alpha_metric=alpha_metric, unusable=unusable,
            primary_pair=(pairs[0][1], pairs[0][2]),
        )
        interactive = write_qc_interactive(
            out_dir, expression=matrix, cells=cells,
            sample_ids=sample_ids, audits=audits,
        )
        summary = out_dir / "qc_summary.tsv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w") as handle:
            handle.write("comparison\tn_control\tn_treated\tgenes\tsignificant\t"
                         "expected_by_chance\tnull_percentile\tinside_null\t"
                         "effective_tests\tloo_best_gain\tloo_p\tflagged_animals\n")
            for label, a in audits.items():
                flagged = ";".join(sample_ids[i] for i in a["flagged"]) or "."
                handle.write(
                    f"{label}\t{a['n_control']}\t{a['n_treated']}\t{a['genes']}\t"
                    f"{a['significant']}\t{a['expected_by_chance']:.1f}\t"
                    f"{a['fraction_percentile']:.1f}\t{a['inside_null']}\t"
                    f"{a['effective_tests']:.0f}\t{a['loo_gain']}\t{a['loo_p']:.3f}\t"
                    f"{flagged}\n")

        report.update({
            "complete": True, "pdf": pdf, "summary": str(summary),
            "interactive": interactive,
            "alpha": alpha, "min_expression": min_expression,
            "alpha_metric": "q" if str(alpha_metric).lower() == "q" else "p",
            "statistic": "t",
            "pseudocount": float(pseudocount),
            "comparisons": list(audits),
            "flagged_animals": sorted({sample_ids[i] for a in audits.values()
                                       for i in a["flagged"]}),
            "all_inside_null": all(a["inside_null"] for a in audits.values()),
            "effective_tests": {k: float(a["effective_tests"]) for k, a in audits.items()},
        })
        # The audit keeps its OWN record. It deliberately does not go into differential_expression.json:
        # an audit contributes nothing to the result, and a checkpoint is the result's contract.
        return _record(report)
    except Exception as exc:  # pragma: no cover - the whole point is not raising
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return _record(report)


# ---------------------------------------------------------------------------
# Refining the population -- picking, not just reporting
# ---------------------------------------------------------------------------

def _safe_dir_name(value: str) -> str:
    """Filesystem-safe token for a refined-run subdirectory name."""
    kept = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return kept.strip("_") or "comparison"


def _file_stamps(folder: Path) -> Dict[str, Tuple[int, int]]:
    """Every file under ``folder`` with its modification time and size."""
    stamps: Dict[str, Tuple[int, int]] = {}
    if folder.is_dir():
        for path in folder.rglob("*"):
            if path.is_file():
                stat = path.stat()
                stamps[str(path)] = (stat.st_mtime_ns, stat.st_size)
    return stamps


def _files_written(folder: Path, before: Mapping[str, Tuple[int, int]]) -> List[str]:
    """The files under ``folder`` that are new or rewritten since ``before``.

    What a refined run wrote, so the differential checkpoint can
    list it and a recompute can remove it. A file that was already there and
    was not touched is not the run's, and is not listed.
    """
    return sorted(
        path for path, stamp in _file_stamps(folder).items() if before.get(path) != stamp
    )


def _refine_one_comparison(
    *, label: str, left: Tuple[str, ...], right: Tuple[str, ...],
    cells: Mapping[Tuple[str, ...], Sequence[int]], gated_matrix: np.ndarray,
    sample_ids: Sequence[str], n: int, out_root: Path,
    expression_matrix: Path, statistic: str,
    min_expression: float, min_expression_difference: float, pseudocount: float,
    alpha_metric: Optional[str] = None, alpha_value: Optional[float] = None,
    directory_name: Optional[str] = None,
    factors: Sequence[str] = (),
    sheet_rows: Optional[Mapping[str, Mapping[str, str]]] = None,
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Dict[str, object]:
    """One comparison's worth of ``run_responder_refine``: pick the animals,
    write their reduced sample sheet, and hand it to the real
    ``differential_expression`` -- the production plan, engine and writer,
    not the audit's own quick internal test.

    The reduced sheet keeps every column of the run's sheet for the
    animals picked (``sheet_rows``), with the tested factor's column holding the
    level each side stands for, so the run's --pair-by and --covariate columns are there;
    the run's --pair-by, --covariate and --covariate-feature are passed on, and
    the factor this comparison varies is its --factor (the other factor is one
    level in the trimmed population, and a factor needs two).
    """
    from paralome.rna.differential import differential_expression

    control_idx, treated_idx = list(cells[left]), list(cells[right])
    scores = responder_scores(gated_matrix, control_idx, treated_idx)
    control_scored = [i for i in control_idx if i in scores]
    treated_scored = [i for i in treated_idx if i in scores]
    if len(control_scored) < 2 or len(treated_scored) < 2:
        return {"complete": False,
                "reason": "too few animals survive leave-one-out for this comparison"}

    # The SAME axis and scores panel J plots and hovers over -- a control
    # close to its own group's mean here is a typical control, not just any
    # animal that happened to be labelled control.
    centre = float(np.mean([scores[i] for i in control_scored]))
    ranked_control = sorted(control_scored, key=lambda i: abs(scores[i] - centre))
    ranked_treated = sorted(treated_scored, key=lambda i: scores[i], reverse=True)
    take = min(int(n), len(ranked_control), len(ranked_treated))
    if take < 2:
        return {"complete": False,
                "reason": f"only {take} animal(s) available on one side after ranking"}
    selected_control = ranked_control[:take]
    selected_treated = ranked_treated[:take]

    a, b = tuple(left), tuple(right)
    axis = next((k for k in range(len(a)) if a[k] != b[k]), None)
    if axis is None:
        return {"complete": False, "reason": "comparison cells do not differ"}
    control_level, treated_level = a[axis], b[axis]

    comparison_dir = out_root / (directory_name or _safe_dir_name(label))
    already_there = _file_stamps(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    filtered_sheet = comparison_dir / "refined_samples.csv"
    if axis >= len(factors) or not factors[axis]:
        return {"complete": False, "reason": "the comparison's factor is not known"}
    tested_column = str(factors[axis])
    picked = [(i, control_level) for i in selected_control] + [(i, treated_level) for i in selected_treated]
    # The sheet keeps the run's columns; the factor this comparison
    # varies holds each animal's level, and is passed with its control level.
    sheet = [
        {**dict((sheet_rows or {}).get(sample_ids[i]) or {}), "sample_id": sample_ids[i], tested_column: level}
        for i, level in picked
    ]
    columns: List[str] = ["sample_id", tested_column]
    for row in sheet:
        columns.extend(key for key in row if key not in columns)
    with filtered_sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sheet)
    tested_factor = [f"{tested_column}:{control_level}"]

        # The run's own --alpha, so the refined files are filtered and named as the full-cohort result is.
        # Left out, differential_expression's own default applies.
    run_alpha: Dict[str, object] = {}
    if alpha_metric is not None:
        run_alpha["alpha_metric"] = alpha_metric
    if alpha_value is not None:
        run_alpha["alpha_value"] = float(alpha_value)

    try:
        result = differential_expression(
            expression_matrix, filtered_sheet, comparison_dir,
            statistic=statistic, factors=tested_factor,
            pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
            overwrite=True, min_expression=min_expression,
            min_expression_difference=min_expression_difference,
            pseudocount=pseudocount, run_qc=False,
            # The refined-top-N files say so -- "MET_vs_VEH_in_CD_top_XX_...",
            # XX being the actual N, distinguishing a trimmed-population file
            # from what the same label would name on the full cohort. `take`
            # (not the requested `n`) because a run can fall short of the
            # requested N per side (see the `take < 2` guard above).
            file_prefix=_safe_dir_name(f"{label}_top_{take}"),
            **run_alpha,
        )
    except Exception as exc:
        return {"complete": False, "reason": f"{type(exc).__name__}: {exc}",
                "files": _files_written(comparison_dir, already_there)}

    audit_alpha: Dict[str, object] = {}
    if alpha_metric is not None:
        audit_alpha["alpha_metric"] = alpha_metric
    if alpha_value is not None:
        audit_alpha["alpha"] = float(alpha_value)
    qc_report = run_qc_ensemble(
        comparison_dir, filtered_sheet, factors=tested_factor,
        min_expression=min_expression, expression_matrix=expression_matrix,
        pseudocount=pseudocount, **audit_alpha,
    )

    return {
        "complete": bool(result.get("complete", True)),
        "control_level": control_level, "treated_level": treated_level,
        "requested_n": int(n), "used_n": take,
        "control_samples": [sample_ids[i] for i in selected_control],
        "treated_samples": [sample_ids[i] for i in selected_treated],
        "control_scores": {sample_ids[i]: round(float(scores[i]), 3) for i in selected_control},
        "treated_scores": {sample_ids[i]: round(float(scores[i]), 3) for i in selected_treated},
        "output_dir": str(comparison_dir),
        "statistic": statistic,
        "statistics": {
            folder: manifest.get("directory")
            for folder, manifest in (result.get("statistics") or {}).items()
            if isinstance(manifest, Mapping)
        },
        "qc": qc_report,
        "files": _files_written(comparison_dir, already_there),
    }


def run_responder_refine(
    differential_dir: Path,
    samples_path: Path,
    expression_matrix: Path,
    *,
    factors: Sequence[str],
    n: int,
    statistic: str = "t",
    min_expression: float = 0.0,
    min_expression_difference: float = 0.0,
    pseudocount: float = 1.0,
    alpha_metric: Optional[str] = None,
    alpha_value: Optional[float] = None,
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Dict[str, object]:
    """Re-run the real differential test on a trimmed population, per comparison.

    Every statistic of the run's --statistic that compares two
    groups (t, paired-t, lm; the ANOVA needs both factors, which a trimmed
    pair of cells does not have -- a run given only anova refines with t),
    with the run's --pair-by, --covariate and --covariate-feature.

    ``alpha_metric``/``alpha_value`` are the run's --alpha ("p" or "q", and the
    cutoff). None leaves the refined runs at differential_expression's default.

    For choosing which of ~10 responders per group are worth an expensive
    follow-up assay -- not "not a non-responder" but the best of the best, on
    both sides.
    For each comparison the audit already defines, this takes the N treated
    animals that responded most (the highest leave-one-out score on the same
    axis panel J plots and hovers over) and the N control animals closest to
    the control mean on that axis (the most typical controls, not just any N)
    -- then hands that reduced sample list to the real
    ``differential_expression``, never the audit's own quick internal test.

    This is a SEPARATE, clearly-labelled result beside the full-cohort one,
    under results/qc/refined_top<N>/: nothing here touches, gates, or is read
    by the original checkpoint. Unlike ``run_qc_ensemble`` this is asked for
    by name, so on failure it says why rather than swallowing it -- but one
    comparison failing must not take the others down with it.
    """
    report: Dict[str, object] = {"complete": False, "reason": None,
                                  "requested_n": int(n), "comparisons": {}}
    try:
        differential_dir = Path(differential_dir)
        try:
            run_statistics_tables(differential_dir)
        except ValueError as error:
            report["reason"] = str(error)
            return report
        if not Path(samples_path).exists():
            report["reason"] = f"sample sheet not found: {samples_path}"
            return report
        active_factors, baselines = factor_specs(factors)
        if not active_factors:
            report["reason"] = "no factors: nothing to refine"
            return report
        if int(n) < 2:
            report["reason"] = "--refine-responders needs at least 2 animals per side"
            return report

        groups = read_sample_groups(samples_path, active_factors)
        level_order = level_order_for(groups, active_factors, baselines)
        if not groups:
            report["reason"] = "sample sheet carries none of the run's factors"
            return report
        try:
            sample_ids, matrix = read_expression_columns(
                differential_dir, list(groups), Path(expression_matrix))
        except ValueError as error:
            report["reason"] = str(error)
            return report
        if matrix.size == 0 or len(sample_ids) < 4:
            report["reason"] = "per-sample expression columns not present"
            return report
        good = ~np.isnan(matrix).any(axis=0)
        matrix = matrix[:, good]

        cells: Dict[Tuple[str, ...], List[int]] = {}
        for index, sample in enumerate(sample_ids):
            cells.setdefault(tuple(groups[sample]), []).append(index)
        cells = {k: v for k, v in cells.items() if len(v) >= MIN_PER_GROUP}
        if len(cells) < 2:
            report["reason"] = f"fewer than two cells with {MIN_PER_GROUP}+ animals"
            return report

        # Same gate run_qc_ensemble applies, kept in step deliberately: this
        # refined re-run and the audit panel the animals are read off
        # must agree on which genes define the axis.
        cell_means = np.vstack([matrix[v].mean(axis=0) for v in cells.values()])
        linear = 2 ** cell_means.max(axis=0) - float(pseudocount)
        gated = np.where(linear >= min_expression)[0]
        if gated.size < 50:
            report["reason"] = f"only {gated.size} genes pass min_expression={min_expression:g}"
            return report
        gated_matrix = matrix[:, gated]

        pairs = _pairwise_cells(cells, level_order)
        if not pairs:
            report["reason"] = "no cell pair differs in exactly one factor"
            return report

        from paralome.rna.differential import _read_samples
        from paralome.stats import parse_statistic_request

        requested = parse_statistic_request(statistic, validate_companions=False).statistics
        # ANOVA needs a factorial design; a refine run compares two cells, so
        # only the statistics of the run that compare two groups apply here. A
        # run given only "anova" has none, so this defaults to "t".
        tokens = [token for token in requested if token != "anova"] or ["t"]
        refine_statistic = ",".join(tokens)
        sheet_rows = {str(row["sample_id"]).strip(): row for row in _read_samples(Path(samples_path))}

        out_root = differential_dir / "results" / "qc" / f"refined_top{int(n)}"
        comparisons: Dict[str, object] = {}
        # "+/- vs +/+" and "-/- vs +/+" both make the folder name
        # "vs", and the second comparison overwrote the first. A name already
        # taken gets a number.
        taken: Dict[str, int] = {}
        for label, left, right in pairs:
            directory_name = _safe_dir_name(label)
            taken[directory_name] = taken.get(directory_name, 0) + 1
            if taken[directory_name] > 1:
                directory_name = f"{directory_name}_{taken[directory_name]}"
            comparisons[label] = _refine_one_comparison(
                label=label, left=left, right=right, cells=cells,
                gated_matrix=gated_matrix, sample_ids=sample_ids, n=int(n),
                out_root=out_root, expression_matrix=Path(expression_matrix),
                statistic=refine_statistic,
                min_expression=min_expression,
                min_expression_difference=min_expression_difference,
                pseudocount=pseudocount,
                alpha_metric=alpha_metric, alpha_value=alpha_value,
                directory_name=directory_name,
                factors=active_factors, sheet_rows=sheet_rows,
                pair_by=pair_by if "paired-t" in tokens else None,
                covariate=covariate if "lm" in tokens else None,
                covariate_feature=covariate_feature if "lm" in tokens else None,
            )

        # Every file this refine wrote, its own record included, for
        # the differential checkpoint -- see differential.record_refined_outputs.
        record = out_root / "refine_report.json"
        written = [str(record)]
        for comparison in comparisons.values():
            written.extend(comparison.get("files") or ())
        report.update({
            "complete": True, "reason": None,
            "output_dir": str(out_root), "comparisons": comparisons,
            "files": sorted(written),
        })
        try:
            import json as _json
            out_root.mkdir(parents=True, exist_ok=True)
            record.write_text(
                _json.dumps(report, indent=2, default=str) + "\n")
        except Exception:
            pass
        return report
    except Exception as exc:  # pragma: no cover - asked for by name, but still must not raise
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return report
