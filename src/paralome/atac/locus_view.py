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

from paralome.atac.quantification import fragment_interval
import math
import re
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter


_ATTR_RE = re.compile(r'([A-Za-z0-9_]+)\s+"([^"]*)"')
#: An Ensembl gene id with its version: ENSG00000157764.14, ENSMUSG00000000001.4.
_ENSEMBL_GENE_VERSION = re.compile(r"^(ENS[A-Z]*G\d+)\.\d+$", re.IGNORECASE)


def _norm(value: object) -> str:
    """A gene id or name as compared: casefolded, an Ensembl id without its version.

    Only an Ensembl gene id loses its version; a name is compared whole, so a
    clone-based name (AL627309.1) or a C. elegans name (F55A12.3) matches only
    itself, not every gene sharing its stem.
    """
    text = str(value or "").strip()
    match = _ENSEMBL_GENE_VERSION.match(text)
    if match:
        text = match.group(1)
    return text.casefold()


class AmbiguousGeneQuery(ValueError):
    """A candidate query that names more than one gene in the GTF."""


def _stars(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number < 1e-4:
        return "****"
    if number < 1e-3:
        return "***"
    if number < 1e-2:
        return "**"
    if number < 5e-2:
        return "*"
    return ""


def _significance_labels(p_value: object, q_value: object) -> Optional[Tuple[str, str]]:
    """Return display labels, suppressing only a doubly nonsignificant result."""
    p_label = _stars(p_value) or "ns"
    q_label = _stars(q_value) or "ns"
    if p_label == "ns" and q_label == "ns":
        return None
    return p_label, q_label


def _parse_attrs(text: str) -> Dict[str, str]:
    return {key: value for key, value in _ATTR_RE.findall(text)}


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _find_gene_model(gtf_path: Path, query: str) -> Optional[Dict[str, object]]:
    """Return a gene record and a representative exon-level transcript model.

    Exons are grouped by transcript before rendering. Merging exons across every
    isoform can erase true introns, so the representative transcript is selected
    by exon count, then transcript span. Genuine single-exon genes remain a
    single box with no artificial intron connector.
    """
    target = _norm(query)
    records: List[Dict[str, object]] = []
    with Path(gtf_path).open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in {"gene", "transcript", "exon"}:
                continue
            attrs = _parse_attrs(fields[8])
            gene_id = attrs.get("gene_id", "")
            gene_name = attrs.get("gene_name", gene_id)
            if target not in {_norm(gene_id), _norm(gene_name)}:
                continue
            records.append(
                {
                    "feature": fields[2],
                    "chrom": fields[0],
                    "start": int(fields[3]) - 1,
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "transcript_id": attrs.get("transcript_id", ""),
                }
            )
    if not records:
        return None
    # One gene, or a refusal. A name or id present at two loci (a
    # GENCODE gene and its _PAR_Y copy, the same name on two chromosomes) was
    # pooled into one model spanning both, with the chromosome of the first.
    by_gene: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        by_gene.setdefault(str(record["gene_id"]), []).append(record)
    if len(by_gene) > 1:
        loci = []
        for gene_id, group in by_gene.items():
            loci.append(f"{gene_id} ({group[0]['gene_name']}) at {group[0]['chrom']}:"
                        f"{min(int(r['start']) for r in group) + 1}-{max(int(r['end']) for r in group)}")
        raise AmbiguousGeneQuery(
            f"candidate gene {query!r} names {len(by_gene)} genes in {gtf_path}: " + "; ".join(loci)
            + ". Name the one you mean by its gene_id."
        )
    gene_records = [record for record in records if record["feature"] == "gene"]
    anchor = gene_records[0] if gene_records else records[0]
    start = min(int(record["start"]) for record in records)
    end = max(int(record["end"]) for record in records)

    exon_groups: Dict[str, List[Tuple[int, int]]] = {}
    for record in records:
        if record["feature"] != "exon":
            continue
        transcript_id = str(record.get("transcript_id", "")) or "__unassigned__"
        exon_groups.setdefault(transcript_id, []).append(
            (int(record["start"]), int(record["end"]))
        )
    candidate_models = [_merge_intervals(intervals) for intervals in exon_groups.values()]
    candidate_models = [model for model in candidate_models if model]
    exons: List[Tuple[int, int]] = []
    if candidate_models:
        exons = max(
            candidate_models,
            key=lambda model: (
                len(model),
                model[-1][1] - model[0][0],
                sum(end_ - start_ for start_, end_ in model),
            ),
        )
    return {
        "chrom": anchor["chrom"],
        "start": start,
        "end": end,
        "strand": anchor["strand"],
        "gene_id": anchor["gene_id"],
        "gene_name": anchor["gene_name"],
        "exons": exons,
        "tss": start if anchor["strand"] == "+" else end - 1,
    }


def _matches_candidate(row: Mapping[str, object], query: str) -> bool:
    target = _norm(query)
    for field in (
        "candidate_query",
        "nearest_gene_id",
        "nearest_gene_name",
        "tss_gene_id",
        "tss_gene_name",
        "tss_gene_ids",
        "tss_gene_names",
    ):
        for token in str(row.get(field, "")).replace(";", ",").split(","):
            if _norm(token) == target:
                return True
    return False


def _coverage(
    bam: Path,
    chrom: str,
    start: int,
    end: int,
    scale_factor: float = 1.0,
    bins: int = 900,
    fragment_padding: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired-fragment coverage, optionally scaled to CPM.

    The same first-in-pair and exclusion flags used by peak quantification are
    applied. The query is padded so fragments whose first mate lies just outside
    the plotted window are still represented, then clipped to the locus.
    """
    length = max(1, end - start)
    delta = np.zeros(length + 1, dtype=float)
    query_start = max(0, start - max(0, int(fragment_padding)))
    query_end = end + max(0, int(fragment_padding))
    region = f"{chrom}:{query_start + 1}-{query_end}"
    # The samtools setup recorded, as quantification is given
    # (binaries.resolve_tool), not whatever this account's PATH finds first.
    from paralome.binaries import resolve_tool

    samtools = resolve_tool("samtools")
    if samtools is None:
        raise FileNotFoundError("Required executable not found (recorded path or PATH): samtools")
    proc = subprocess.Popen(
        [samtools, "view", "-f", "66", "-F", "3852", str(bam), region],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9:
            continue
        # The fragment as quantification counts it (fragment_interval).
        interval = fragment_interval(fields)
        if interval is None:
            continue
        fragment_start, fragment_end = interval
        clipped_start = max(start, fragment_start)
        clipped_end = min(end, fragment_end)
        if clipped_end <= clipped_start:
            continue
        left = clipped_start - start
        right = clipped_end - start
        delta[left] += scale_factor
        delta[right] -= scale_factor
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"samtools view failed for {bam}: exit code {rc}")
    values = np.cumsum(delta[:-1])
    if length <= bins:
        return np.arange(start, end), values
    edges = np.linspace(0, length, bins + 1, dtype=int)
    y = np.array(
        [
            values[edges[i] : edges[i + 1]].mean() if edges[i + 1] > edges[i] else 0.0
            for i in range(bins)
        ]
    )
    x = start + (edges[:-1] + edges[1:]) / 2
    return x, y


def _sniff_delimiter(path: Path) -> str:
    """The summary is comma-delimited, but an older tab-delimited copy can still be on disk -- sniff
    instead of assuming either.
    """
    with open(path, "r", newline="") as handle:
        first_line = handle.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def load_coverage_scaling(summary_path: Optional[Path]) -> Dict[str, float]:
    """Load per-sample CPM factors from peak quantification summary.

    Factors are one million divided by all eligible nuclear fragments, matching
    the fragment filters used by peak quantification. A sample with a missing
    or zero denominator gets no factor (render_candidate_gene_views
    refuses to draw it beside scaled traces).
    """
    factors: Dict[str, float] = {}
    if summary_path is None or not Path(summary_path).exists():
        return factors
    with Path(summary_path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter=_sniff_delimiter(Path(summary_path))):
            sample_id = str(row.get("sample_id", "")).strip()
            try:
                eligible = float(row.get("eligible_fragments", 0) or 0)
            except (TypeError, ValueError):
                eligible = 0.0
            if sample_id and eligible > 0:
                factors[sample_id] = 1_000_000.0 / eligible
    return factors


def _parse_window_spec(value: str) -> Tuple[int, float]:
    """Parse ``upstream,multiplier`` and validate a positive fixed-width window."""
    pieces = [piece.strip() for piece in str(value).split(",")]
    if len(pieces) != 2:
        raise ValueError("viewer window spec must be 'upstream,multiplier', for example 10000,2")
    try:
        upstream = int(pieces[0])
        multiplier = float(pieces[1])
    except ValueError as exc:
        raise ValueError("viewer window spec must contain numeric upstream and multiplier values") from exc
    if upstream < 1:
        raise ValueError("viewer upstream distance must be at least 1 bp")
    if multiplier < 1:
        raise ValueError("viewer window multiplier must be at least 1")
    return upstream, multiplier


def _window_from_tss(gene: Mapping[str, object], window_spec: str) -> Tuple[int, int]:
    upstream, multiplier = _parse_window_spec(window_spec)
    total_width = max(1, int(round(upstream * multiplier)))
    downstream = max(0, total_width - upstream)
    tss = int(gene["tss"])
    if str(gene.get("strand", "+")) == "+":
        start = tss - upstream
        end = tss + downstream
    else:
        start = tss - downstream
        end = tss + upstream
    return max(0, start), max(max(0, start) + 1, end)


def order_samples(
    samples: Sequence[Mapping[str, object]], group_column: Optional[str], control: Optional[str],
) -> List[Mapping[str, object]]:
    """The traces' order: the control level of the first --factor first, then by level, then sample_id."""

    def key(row: Mapping[str, object]) -> Tuple[int, str, str]:
        value = str(row.get(group_column, "") or "") if group_column else ""
        from paralome.stats import control_choices

        # A --factor may list several control levels.
        return (0 if value in control_choices(control) else 1, value, str(row.get("sample_id", "")))

    return sorted(samples, key=key)


def render_candidate_gene_views(
    rows: Sequence[Mapping[str, object]],
    candidate_queries: Sequence[str],
    gtf_path: Path,
    output_dir: Path,
    samples: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    significance_from: str,
    flank: int = 25000,
    window_spec: Optional[str] = None,
    min_y: float = 5.0,
    coverage_scaling: Optional[Mapping[str, float]] = None,
    motif_labels_by_peak: Optional[Mapping[str, Sequence[str]]] = None,
    motifs_per_peak: int = 5,
    group_column: Optional[str] = None,
    control: Optional[str] = None,
) -> List[Path]:
    """Render portrait, fixed-window candidate-gene ATAC locus views.

    The traces are ordered by ``group_column`` -- the run's first
    --factor -- with its ``control`` level first, then by sample_id; with no
    factor, by sample_id alone.

    ``rows`` are peaks (chrom, start, end, peak_id and the gene annotation)
    carrying the generic ``p_value`` and ``q_value`` of ONE comparison;
    ``significance_from`` names that comparison (e.g. ``t / KO_vs_WT``) and is
    written into each view's title, so a reader knows which test the stars are.

    ``flank`` is used only when ``window_spec`` is omitted;
    ``window_spec='upstream,multiplier'`` with multiplier 2 centers the TSS,
    larger multipliers allocate additional space downstream in transcriptional
    orientation.
    """
    if min_y <= 0:
        raise ValueError("viewer minimum Y-axis height must be positive")
    if not str(significance_from).strip():
        raise ValueError("significance_from must name the comparison whose p/q values the views show")
    p_field, q_field = "p_value", "q_value"
    if window_spec is None:
        window_spec = f"{max(1, int(flank))},2"
    _parse_window_spec(window_spec)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    ordered_samples = order_samples(samples or [], group_column, control)
    scaling = dict(coverage_scaling or {})
    motif_labels = {str(key): list(value) for key, value in dict(motif_labels_by_peak or {}).items()}

    # EVERY TRACE IS A REAL ONE, AND ALL ON ONE SCALE. A sample whose 'bam' is empty or does not
    # exist, or that is absent from the peak-counting summary (so has no CPM factor), is refused
    # here, before anything is drawn, rather than drawn as zeros or as raw depth on a CPM axis.
    # When no sample has a factor the traces are raw fragments and the axis says so.
    without_bam = []
    for sample in ordered_samples:
        bam_text = str(sample.get("bam", "") or "")
        if not bam_text or not Path(bam_text).exists():
            without_bam.append(f"{sample.get('sample_id', '?')} ({bam_text or 'no bam column'})")
    if without_bam:
        raise ValueError(
            f"candidate gene views need every sample's BAM; {len(without_bam)} sample(s) have none: "
            + ", ".join(without_bam)
            + ". Give --samples a sheet with a bam column naming the deduplicated BAMs "
            "(the pipeline passes the deduplication stage's manifest)."
        )
    sample_ids = [str(sample.get("sample_id", "")) for sample in ordered_samples]
    unscaled = [sample_id for sample_id in sample_ids if sample_id not in scaling]
    if unscaled and len(unscaled) != len(sample_ids):
        raise ValueError(
            f"candidate gene views cannot mix CPM-scaled and raw traces: {len(unscaled)} of {len(sample_ids)} "
            f"sample(s) have no scaling factor in the peak-counting summary ({', '.join(unscaled)}). "
            "Run peak quantification for every sample in the comparison, or pass the summary that has them."
        )
    scaled = not unscaled
    y_label = "Fragment coverage (CPM)" if scaled else "Fragment coverage (raw fragments)"
    if ordered_samples and not scaled:
        print("WARNING: no per-sample scaling factors (peak_counting_summary not found or empty); "
              "candidate gene views show raw fragment coverage, and the axis says so.")

    for query in candidate_queries:
        try:
            gene = _find_gene_model(Path(gtf_path), query)
        except AmbiguousGeneQuery as exc:
            # Loudly, and without a view: a window drawn around the wrong gene
            # reads as a result. The other candidates are still drawn.
            print(f"WARNING: no locus view for {query!r}: {exc}")
            continue
        matched = [row for row in rows if _matches_candidate(row, query)]
        if gene is None and not matched:
            continue
        if gene is not None:
            chrom = str(gene["chrom"])
            region_start, region_end = _window_from_tss(gene, window_spec)
        else:
            chrom = str(matched[0].get("chrom", ""))
            center = int(
                round(
                    np.mean(
                        [
                            (int(float(row["start"])) + int(float(row["end"]))) / 2
                            for row in matched
                        ]
                    )
                )
            )
            upstream, multiplier = _parse_window_spec(window_spec)
            width = max(1, int(round(upstream * multiplier)))
            region_start = max(0, center - upstream)
            region_end = region_start + width

        peaks = [
            row
            for row in rows
            if str(row.get("chrom", "")) == chrom
            and int(float(row.get("end", 0))) > region_start
            and int(float(row.get("start", 0))) < region_end
        ]
        peaks.sort(key=lambda row: (int(float(row.get("start", 0))), int(float(row.get("end", 0)))))

        # Load all sample coverages first so every trace shares one locus-wide Y maximum.
        coverage_rows: List[Tuple[Mapping[str, object], np.ndarray, np.ndarray]] = []
        observed_max = 0.0
        for sample in ordered_samples:
            x, y = _coverage(
                Path(str(sample.get("bam", ""))), chrom, region_start, region_end,
                scale_factor=float(scaling[str(sample.get("sample_id", ""))]) if scaled else 1.0,
            )
            if y.size:
                observed_max = max(observed_max, float(np.nanmax(y)))
            coverage_rows.append((sample, x, y))
        y_max = max(float(min_y), observed_max * 1.05)

        n_cov = len(coverage_rows)
        # Portrait dimensions: the coverage section scales with sample number, while
        # gene and peak metadata remain fixed-height across every candidate locus.
        # Include a dedicated spacer row below the final coverage trace. This
        # protects the genomic X-axis labels from the gene-name annotation while
        # preserving the established gene/peak/significance track positions.
        fig_height = max(10.8, 1.34 * n_cov + 4.5)
        fig = plt.figure(figsize=(8.5, fig_height))
        peak_track_height = 0.98 if motif_labels else 0.78
        grid = fig.add_gridspec(
            n_cov + 3,
            1,
            height_ratios=([1.0] * n_cov) + [0.24, 0.54, peak_track_height],
            hspace=0.13,
        )
        cov_axes = [fig.add_subplot(grid[index, 0]) for index in range(n_cov)]
        if cov_axes:
            gene_ax = fig.add_subplot(grid[n_cov + 1, 0], sharex=cov_axes[0])
            peak_ax = fig.add_subplot(grid[n_cov + 2, 0], sharex=cov_axes[0])
        else:
            gene_ax = fig.add_subplot(grid[1, 0])
            peak_ax = fig.add_subplot(grid[2, 0], sharex=gene_ax)
        all_axes = cov_axes + [gene_ax, peak_ax]
        for axis in all_axes:
            axis.set_xlim(region_start, region_end)
        # Keep the bottom annotation tracks behind the trace axes so their text
        # cannot obscure the genomic axis on the final coverage panel.
        gene_ax.set_zorder(1)
        peak_ax.set_zorder(1)
        for axis in cov_axes:
            axis.set_zorder(3)
            axis.patch.set_facecolor("white")
            axis.patch.set_alpha(1.0)

        for index, (axis, (sample, x, y)) in enumerate(zip(cov_axes, coverage_rows)):
            sample_id = str(sample.get("sample_id", ""))
            axis.fill_between(x, y, 0, alpha=0.72, linewidth=0.45)
            axis.set_ylim(0, y_max)
            axis.set_ylabel(y_label, fontsize=7, labelpad=7)
            axis.text(
                0.012, 0.94, sample_id,
                transform=axis.transAxes,
                ha="left", va="top",
                fontsize=10, fontweight="bold",
            )
            axis.tick_params(axis="y", labelsize=7)
            if index != len(cov_axes) - 1:
                axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                axis.spines["bottom"].set_visible(True)
            else:
                axis.spines["bottom"].set_visible(True)
                axis.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize=8, pad=3)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                axis.xaxis.set_major_formatter(formatter)
                axis.set_xlabel(f"{chrom}:{region_start:,}-{region_end:,}", fontsize=8, labelpad=6)

        # Exon-level gene model, placed between coverage and peaks.
        gene_ax.set_ylim(0, 1)
        gene_ax.set_yticks([])
        gene_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        gene_ax.text(
            -0.055, 0.50, "Gene", transform=gene_ax.transAxes,
            ha="right", va="center", fontsize=8,
        )
        for spine in gene_ax.spines.values():
            spine.set_visible(False)
        if gene is not None:
            clipped_start = max(region_start, int(gene["start"]))
            clipped_end = min(region_end, int(gene["end"]))
            if clipped_end > clipped_start:
                gene_ax.plot(
                    [clipped_start, clipped_end], [0.50, 0.50],
                    color="0.68", linewidth=0.75, zorder=1,
                )
                # Dense repeated arrows make transcriptional direction obvious even
                # across long introns and wide fixed windows.
                visible_fraction = (clipped_end - clipped_start) / max(1, region_end - region_start)
                arrow_count = max(10, min(48, int(round(visible_fraction * 36))))
                arrow_positions = np.linspace(clipped_start, clipped_end, arrow_count + 2)[1:-1]
                direction = 1 if str(gene.get("strand", "+")) == "+" else -1
                arrow_span = max(1, int((region_end - region_start) * 0.008))
                for position in arrow_positions:
                    gene_ax.annotate(
                        "",
                        xy=(position + direction * arrow_span, 0.50),
                        xytext=(position, 0.50),
                        arrowprops=dict(arrowstyle="-|>", color="0.62", lw=0.55),
                        zorder=2,
                    )
            for exon_start, exon_end in gene.get("exons", []):
                start = max(region_start, int(exon_start))
                end = min(region_end, int(exon_end))
                if end > start:
                    gene_ax.add_patch(Rectangle((start, 0.34), end - start, 0.32, color="steelblue", zorder=3))
            # Fall back to a compact gene body if the GTF has no exon records.
            if not gene.get("exons") and clipped_end > clipped_start:
                gene_ax.add_patch(Rectangle((clipped_start, 0.39), clipped_end - clipped_start, 0.22, color="steelblue", zorder=3))
            tss = int(gene["tss"])
            tss_width = max(1, int((region_end - region_start) * 0.004))
            gene_ax.add_patch(
                Rectangle((tss - tss_width / 2, 0.27), tss_width, 0.46, color="darkorange", zorder=4)
            )
            gene_ax.text(tss, 0.20, "TSS", ha="center", va="top", fontsize=7, color="darkorange")
            gene_ax.text(
                (region_start + region_end) / 2,
                0.86,
                str(gene["gene_name"]),
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

        # Fixed-height metadata track with collision-aware label lanes.
        peak_ax.set_ylim(0, 1)
        peak_ax.set_yticks([])
        peak_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for spine in peak_ax.spines.values():
            spine.set_visible(False)
        peak_ax.text(
            -0.055, 0.86 if motif_labels else 0.83, "Peaks", transform=peak_ax.transAxes,
            ha="right", va="center", fontsize=8,
        )
        if motif_labels:
            peak_ax.text(
                -0.055, 0.44, "Motifs", transform=peak_ax.transAxes,
                ha="right", va="center", fontsize=8,
            )
            peak_ax.text(
                -0.055, 0.16, "Significance", transform=peak_ax.transAxes,
                ha="right", va="center", fontsize=8,
            )
        else:
            peak_ax.text(
                -0.055, 0.29, "Significance", transform=peak_ax.transAxes,
                ha="right", va="center", fontsize=8,
            )
        label_size = 6
        significance_size = 6
        region_width = max(1, region_end - region_start)
        # Approximate rendered label width in genomic units. Adjacent labels are
        # assigned alternating lanes when their text boxes would overlap.
        lane_ends = [-float("inf"), -float("inf"), -float("inf")]
        annotated_peaks = []
        for row in peaks:
            start = max(region_start, int(float(row.get("start", 0))))
            end = min(region_end, int(float(row.get("end", 0))))
            if end <= start:
                continue
            center = (start + end) / 2
            peak_id = str(row.get("peak_id", ""))
            significance = _significance_labels(row.get(p_field), row.get(q_field))
            p_star, q_star = significance if significance is not None else ("", "")
            sig_text = f"(P: {p_star} | Q: {q_star})" if significance is not None else ""
            motif_text = "; ".join(motif_labels.get(peak_id, [])[: max(1, motifs_per_peak)])
            longest_chars = max(len(peak_id), len(sig_text), len(motif_text), 1)
            half_width = region_width * min(0.18, 0.0062 * longest_chars) / 2
            lane = min(
                range(len(lane_ends)),
                key=lambda idx: (center - half_width < lane_ends[idx], lane_ends[idx]),
            )
            # Prefer the first non-colliding lane; otherwise use the lane ending earliest.
            for candidate, lane_end in enumerate(lane_ends):
                if center - half_width >= lane_end:
                    lane = candidate
                    break
            lane_ends[lane] = center + half_width
            annotated_peaks.append((row, start, end, center, peak_id, motif_text, p_star, q_star, lane))

        if motif_labels:
            name_y = [0.72, 0.64, 0.56]
            motif_y = [0.44, 0.37, 0.30]
            sig_y = [0.18, 0.11, 0.04]
            bar_y = 0.84
        else:
            name_y = [0.68, 0.58, 0.48]
            motif_y = [0.0, 0.0, 0.0]
            sig_y = [0.31, 0.21, 0.11]
            bar_y = 0.79
        for row, start, end, center, peak_id, motif_text, p_star, q_star, lane in annotated_peaks:
            peak_ax.add_patch(Rectangle((start, bar_y), max(1, end - start), 0.08, color="steelblue"))
            if peak_id:
                peak_ax.text(center, name_y[lane], peak_id, ha="center", va="top", fontsize=label_size)
            if motif_text:
                peak_ax.text(center, motif_y[lane], motif_text, ha="center", va="center", fontsize=5.5, color="0.25")
            if p_star or q_star:
                # A packed offset box guarantees visible spacing around the pipe
                # while preserving independent P/Q colors in vector PDF output.
                packed = HPacker(
                    children=[
                        TextArea(
                            f"(P: {p_star}",
                            textprops={"fontsize": significance_size, "color": "purple"},
                        ),
                        TextArea(
                            " | ",
                            textprops={"fontsize": significance_size, "color": "black"},
                        ),
                        TextArea(
                            f"Q: {q_star})",
                            textprops={"fontsize": significance_size, "color": "teal"},
                        ),
                    ],
                    align="center",
                    pad=0,
                    sep=2,
                )
                peak_ax.add_artist(
                    AnnotationBbox(
                        packed,
                        (center, sig_y[lane]),
                        xycoords="data",
                        box_alignment=(0.5, 0.5),
                        frameon=False,
                        pad=0,
                    )
                )

        fig.text(
            0.5,
            0.018,
            "* <0.05   ** <0.01   *** <0.001   **** <0.0001   |   P = raw P-value; Q = BH-adjusted Q-value",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        fig.suptitle(
            f"{query}: significance from {significance_from}", fontsize=9, y=0.995,
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query).strip("_") or "candidate_gene"
        out_path = output_dir / f"{safe}_locus.pdf"
        fig.subplots_adjust(bottom=0.07, top=0.97, left=0.21, right=0.97)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written
