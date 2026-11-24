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

"""Every QC statistic Paralome computes, in one place.

``paralome.stats`` holds the tests that make results (t, ANOVA, lm, BH ...).
This module holds the statistics that check samples and results: the PCA, the
sample-sample correlation, the clustering, the robust outlier rule, the group
ellipses and the permutation nulls. Every arm reaches them here and nowhere else.

WHAT IS ONE
-----------
* ONE PCA (``pca``): centred SVD, every component signed so its largest loading
  is positive. ``scale=True`` standardises each feature first (the outlier QC's
  choice); otherwise unscaled. Exact, never randomized.
* ONE sample-sample correlation (``sample_correlation``), Pearson or Spearman,
  with or without centring each feature first; ONE pair correlation
  (``pearson``, ``spearman``).
* ONE clustering on correlation (``average_linkage``) and its cut (``cut_tree``).
* ONE robust outlier rule (``robust_centre_scale``): median and scaled MAD, the
  Iglewicz-Hoaglin modified z, with their mean-absolute-deviation fallback.
* ONE covariance ellipse (``covariance_axes``). How large it is drawn stays a
  per-plot choice, made by the caller: the pseudobulk QC's 95% Hotelling region
  of the mean (``confidence_ellipse_radius``), the global PCA's fixed 1.75 SD.

WHAT STAYS PER ARM, AS PARAMETERS
---------------------------------
The thresholds (the outlier QC's 90% variance, distance ratio 2 and Spearman
0.80; the pseudobulk 3.5 MAD and half-median detection), whether features are
standardised, centred or ranked, and the permutation counts and seeds. Each is
an argument here, set by the caller, so the same arithmetic serves every arm.

No module outside ``paralome.stats`` and this one computes a PCA, a
correlation, a covariance, a clustering or a permutation itself.
"""

from __future__ import annotations

import time as _time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def orient(scores: np.ndarray, loadings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Give every component a deterministic sign.

    Signed so the largest-magnitude loading is positive. Any rule works as long
    as it is a function of the data alone; this one is stable under row
    permutation, which ``argmax`` on ties is not, because the comparison is on
    the signed value at a position chosen by magnitude. (It is also the rule
    sklearn's PCA used, so its sample views keep their orientation.)
    """
    for k in range(loadings.shape[0]):
        position = int(np.argmax(np.abs(loadings[k])))
        if loadings[k, position] < 0:
            loadings[k] *= -1.0
            scores[:, k] *= -1.0
    return scores, loadings


def pca(values: np.ndarray, components: int, *, scale: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """THE PCA: ``(scores, explained, loadings)`` of ``values``, samples x features.

    Centred, by exact SVD, oriented (``orient``). ``explained`` is each
    component's share of the total variance. ``scale`` standardises each feature
    to unit variance first (a constant feature becomes 0); the outlier QC
    standardises, every other view does not -- on log2 values scaling promotes
    the low-expression noise.
    """
    values = np.asarray(values, dtype=float)
    if scale:
        means = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        centred = np.divide(values - means, std, out=np.zeros_like(values), where=std > 0)
    else:
        centred = values - values.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    total = float((s ** 2).sum())
    explained = (s ** 2) / total if total > 0 else np.zeros_like(s)
    scores = u * s
    scores, vt = orient(scores, vt)
    return scores[:, :components], explained[:components], vt[:components]


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def sample_correlation(values: np.ndarray, method: str = "pearson", *,
                       center_features: bool = False) -> np.ndarray:
    """Sample x sample correlation across features. ``values`` is samples x features.

    ``method`` is ``pearson`` or ``spearman`` (each sample's values ranked,
    ties averaged). ``center_features`` subtracts each feature's mean across
    samples first (after ranking), so the correlation asks whether samples
    deviate from the average profile alike rather than whether they share it --
    the global PCA heatmap's choice; uncentred, every log-expression
    sample correlates at 0.93-1.00 with every other.
    """
    values = np.asarray(values, dtype=float)
    if method == "spearman":
        from scipy.stats import rankdata

        if not center_features:
            from paralome.stats import spearman_matrix

            return spearman_matrix(values)
        values = rankdata(values, axis=1)
    if center_features:
        values = values - values.mean(axis=0, keepdims=True)
    return np.corrcoef(values)


def correlation_distance(correlation: np.ndarray) -> np.ndarray:
    """``sqrt(2 (1 - r))``: the Euclidean distance between standardised profiles."""
    return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.asarray(correlation, dtype=float))))


def pearson(a: Sequence[float], b: Sequence[float], *, min_n: int = 3,
            undefined: Optional[float] = 0.0) -> Optional[float]:
    """Pearson r of two vectors; ``undefined`` when fewer than ``min_n`` or either is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < min_n or np.std(a) == 0 or np.std(b) == 0:
        return undefined
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else undefined


def spearman(a: Sequence[float], b: Sequence[float], *, min_n: int = 3,
             undefined: Optional[float] = None) -> Optional[float]:
    """Spearman's rho (Pearson on average ranks); ``undefined`` as for ``pearson``."""
    from scipy.stats import rankdata

    if len(a) < min_n:
        return undefined
    return pearson(rankdata(np.asarray(a, dtype=float)), rankdata(np.asarray(b, dtype=float)),
                   min_n=min_n, undefined=undefined)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def average_linkage(correlation: np.ndarray, *, min_n: int = 3) -> Tuple[List[int], Optional[np.ndarray]]:
    """Leaf order AND the linkage that produced it, from average linkage on 1 - r.

    The heatmap ordering, its dendrograms and any cut must all describe the SAME
    tree; recomputing the linkage per consumer is how a plot ends up whose drawn
    tree disagrees with its own row order. Computed once, passed everywhere.

    Returns ``(order, None)`` when there are fewer than ``min_n`` rows, and
    callers draw without dendrograms. Average linkage is fixed. Undefined correlations (a constant row) are treated
    as uncorrelated, distance 1, for ordering only.
    """
    n = correlation.shape[0]
    if n < min_n:
        return list(range(n)), None
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    return [int(index) for index in leaves_list(linkage_matrix)], linkage_matrix


def cut_tree(linkage_matrix: np.ndarray, k: int) -> np.ndarray:
    """Cluster numbers (1..k) from cutting a linkage into at most ``k`` groups."""
    from scipy.cluster.hierarchy import fcluster

    return fcluster(linkage_matrix, t=int(k), criterion="maxclust")


def kmeans(points: np.ndarray, k: int, seed: int = 0, iterations: int = 300) -> np.ndarray:
    """Lloyd's algorithm, seeded deterministically. numpy only.

    Deliberately not sklearn: this must run identically wherever the QC runs,
    and a fixed seed with a fixed initialisation makes the cluster labels
    reproducible across invocations. A QC whose clusters renumber themselves
    between runs is unreadable.
    """
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(len(points), k, replace=False)]
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        labels = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1).argmin(1)
        moved = np.array([
            points[labels == j].mean(0) if (labels == j).any() else centres[j]
            for j in range(k)
        ])
        if np.allclose(moved, centres):
            break
        centres = moved
    # Relabel largest-first so the numbering is stable and meaningful.
    order = np.argsort([-(labels == j).sum() for j in range(k)])
    remap = {int(old): new for new, old in enumerate(order)}
    return np.array([remap[int(v)] for v in labels], dtype=int)


def silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette width. 0 means no cluster structure at all."""
    n = len(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    scores = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = distance[i, same].mean()
        b = min(distance[i, labels == g].mean() for g in unique if g != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def cluster_samples(
    scores: np.ndarray,
    distance: np.ndarray,
    *,
    max_clusters: int = 6,
    seed: int = 0,
) -> Dict[str, object]:
    """Find structure with no labels: k chosen by mean silhouette.

    Returns the labels, the chosen k, its silhouette, and the silhouette of
    every k considered -- because "k=2 scored 0.16 and k=3 scored 0.15" is a
    materially different statement from "there are 2 clusters", and only the
    first is true.

    THE NULL IS NOT OPTIONAL. A silhouette has no absolute scale: k-means
    returns k clusters from noise just as willingly as from structure, and
    0.16 sounds like something. The caller compares it against
    ``silhouette_null``. Without that, this function would report clusters in
    pure noise and never say so.
    """
    n = len(scores)
    result: Dict[str, object] = {"labels": np.zeros(n, dtype=int), "k": 1,
                                 "silhouette": 0.0, "by_k": {}}
    if n < 6:
        result["note"] = f"only {n} samples; not clustered (needs 6)"
        return result
    best = None
    for k in range(2, min(max_clusters, n - 1) + 1):
        labels = kmeans(scores, k, seed=seed)
        if len(np.unique(labels)) < k:
            continue
        width = silhouette(distance, labels)
        result["by_k"][k] = round(width, 4)  # type: ignore[index]
        if best is None or width > best[0]:
            best = (width, k, labels)
    if best is None:
        return result
    result["silhouette"], result["k"], result["labels"] = float(best[0]), int(best[1]), best[2]
    return result


def silhouette_null(values: np.ndarray, components: int, k: int, *, permutations: int,
                    seed: int = 0) -> Tuple[float, float]:
    """Mean and SD of the silhouette of ``k`` clusters on column-permuted copies of ``values``.

    Each feature is shuffled across samples independently: the same marginals,
    no sample structure. Clustered exactly as ``cluster_samples`` clusters the
    real data (``pca`` scores, correlation distance), so the two are comparable.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        shuffled = np.array([rng.permutation(row) for row in values.T]).T
        null_scores, _, _ = pca(shuffled, components)
        null_distance = correlation_distance(sample_correlation(shuffled))
        null.append(silhouette(null_distance, kmeans(null_scores, int(k), seed=seed)))
    return float(np.mean(null)), float(np.std(null))


# ---------------------------------------------------------------------------
# The robust outlier rule
# ---------------------------------------------------------------------------

#: The Iglewicz-Hoaglin modified z-score cutoff.
MODIFIED_Z_CUTOFF = 3.5


def robust_centre_scale(values: Sequence[float]) -> Tuple[float, float]:
    """``(median, scale)``: the scale is the MAD x 1.4826, the SD on normal data.

    MEDIAN AND MAD, NOT MEAN AND SD, and this is not a stylistic preference.
    Given correlations [0.90, 0.91, 0.89, 0.92, 0.30], the 0.30 inflates the SD
    to 0.27, which puts a mean - 2 SD threshold at 0.24, BELOW the outlier: the
    sample masks itself, and with several bad samples each one raises the bar
    protecting the others. Median and a scaled MAD have a 50% breakdown point.

    When more than half the values sit exactly at the median the MAD is 0 and
    every threshold built on it collapses; Iglewicz-Hoaglin's documented
    fallback is the mean absolute deviation x 1.253314.
    """
    values = np.asarray(values, dtype=float)
    centre = float(np.median(values))
    deviation = np.abs(values - centre)
    scale = float(np.median(deviation)) * 1.4826
    if scale == 0.0:
        scale = float(np.mean(deviation)) * 1.253314
    return centre, scale


def modified_z(values: Sequence[float]) -> np.ndarray:
    """The Iglewicz-Hoaglin modified z of each value (0 where the scale is 0)."""
    values = np.asarray(values, dtype=float)
    centre, scale = robust_centre_scale(values)
    if not scale > 0:
        return np.zeros_like(values)
    return (values - centre) / scale


def flag_samples(
    mean_correlation: np.ndarray,
    detection: np.ndarray,
    *,
    correlation_mad: float = MODIFIED_Z_CUTOFF,
    detection_ratio: float = 0.5,
    min_samples: int = 3,
) -> Tuple[np.ndarray, List[str]]:
    """The pseudobulk QC's flags: low mean correlation, or low detection.

    Returns a boolean mask and one reason string per sample (empty when not
    flagged). A sample is flagged when its mean correlation to the others is
    more than ``correlation_mad`` scaled MADs below the cohort median
    (``robust_centre_scale``), or its detection rate is below
    ``detection_ratio`` x the cohort median. Both are relative to the cohort,
    so the same rule behaves correctly on a modality where every sample is fine
    and on one where the small strata are noise.
    """
    n = len(mean_correlation)
    reasons = [""] * n
    if n < min_samples:
        return np.zeros(n, dtype=bool), reasons

    corr_median, mad = robust_centre_scale(mean_correlation)
    corr_floor = corr_median - correlation_mad * mad if mad > 0 else -np.inf
    detect_floor = float(np.median(detection)) * detection_ratio

    for i in range(n):
        why: List[str] = []
        if mean_correlation[i] < corr_floor:
            why.append(
                f"mean r to others {mean_correlation[i]:.3f} < {corr_floor:.3f} "
                f"({correlation_mad:g} MAD below the cohort median {corr_median:.3f})"
            )
        if detection[i] < detect_floor:
            why.append(
                f"detection {detection[i]*100:.1f}% < {detect_floor*100:.1f}% "
                f"({detection_ratio:g}x the cohort median)"
            )
        reasons[i] = "; ".join(why)
    flagged = np.array([bool(r) for r in reasons], dtype=bool)
    return flagged, reasons


# ---------------------------------------------------------------------------
# Group ellipses
# ---------------------------------------------------------------------------

def covariance_axes(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(centre, eigenvalues, eigenvectors)`` of a group's covariance, largest axis first.

    THE ellipse: every group ellipse and disc Paralome draws is these axes,
    scaled by a radius the plot chooses. None when the covariance is not
    finite.
    """
    points = np.asarray(points, dtype=float)
    covariance = np.cov(points.T, ddof=1)
    if not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return points.mean(axis=0), eigenvalues[order], eigenvectors[:, order]


def f_quantile_2(df2: float, p: float = 0.95) -> float:
    """Exact ``F_p(2, df2)``. Closed form, so scipy is not needed.

    The F distribution with **2** numerator degrees of freedom has an
    invertible CDF: ``P(F <= x) = 1 - (1 + 2x/df2)^(-df2/2)``. Every ellipse
    here is 2-dimensional, so df1 is always 2 and this is exact rather than an
    approximation. Checked against tabulated values in the tests --
    F(2,3)=9.5521, F(2,6)=5.1433, F(2,10)=4.1028.
    """
    if df2 <= 0:
        return float("nan")
    return df2 / 2.0 * ((1.0 - p) ** (-2.0 / df2) - 1.0)


def confidence_ellipse_radius(n: int, p: float = 0.95) -> float:
    """Hotelling T^2 radius, in SD units, for a ``p`` confidence region of the MEAN.

    The 95% CI ellipses of the pseudobulk QC, which is for initial exploration.

    **This radius depends on n, and that is not a bug -- it is what a
    confidence interval is.** At n=5 it is 2.26 SD; at n=8, 1.22; at n=120,
    0.23. Two clusters of different size therefore get differently-sized
    ellipses from identical spread.

    The global PCA deliberately does NOT do this. It draws a fixed 1.75 SD so
    the radius cannot move with n, for visual grouping with no inference
    attached. The two plots answer different questions; both draw the same
    ``covariance_axes``.
    """
    if n < 3:
        return float("nan")
    return float(np.sqrt(2.0 * (n - 1) / (n * (n - 2)) * f_quantile_2(n - 2, p)))


def ellipse_for(points: np.ndarray, p: float = 0.95) -> Optional[Dict[str, float]]:
    """Centre, semi-axes and rotation of the ``p`` confidence ellipse for a cluster's mean.

    ``covariance_axes`` gives the axes; the radius comes from
    ``confidence_ellipse_radius`` and so scales with n. Returns None when
    fewer than three points, because a covariance from two is a line.
    """
    if len(points) < 3:
        return None
    axes = covariance_axes(points)
    if axes is None:
        return None
    centre, values, vectors = axes
    k = confidence_ellipse_radius(len(points), p)
    return {
        "cx": float(centre[0]), "cy": float(centre[1]),
        "rx": float(k * np.sqrt(max(values[0], 0.0))),
        "ry": float(k * np.sqrt(max(values[1], 0.0))),
        "angle": float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))),
        "n": int(len(points)), "k_sd": float(k),
    }


# ---------------------------------------------------------------------------
# The sample outlier QC (bulk RNA, bulk ATAC; the single-cell and proteomics
# sample views): distance to the group centroid, and correlation to the group
# ---------------------------------------------------------------------------

#: The outlier QC's thresholds.
VARIANCE_KEPT = 0.90
DISTANCE_RATIO_FLAG = 2.0
CORRELATION_FLAG = 0.80


def sample_outlier_metrics(
    sample_by_feature: np.ndarray,
    groups: Sequence[str],
    *,
    variance_kept: float = VARIANCE_KEPT,
    distance_ratio_flag: float = DISTANCE_RATIO_FLAG,
    correlation_flag: float = CORRELATION_FLAG,
) -> Tuple[np.ndarray, np.ndarray, int, List[Dict[str, object]]]:
    """``(scores, explained, components_used, per-sample metrics)`` of the outlier QC.

    The standardised PCA (``pca(scale=True)``) over the components that reach
    ``variance_kept`` of the variance, each weighted by the square root of its
    share; a sample's distance to its group's centroid (the group without it)
    against its peers' own spread; and its median Spearman correlation with
    them. Both flags: OUTLIER_CANDIDATE; one: WARN; neither: PASS.
    """
    values = np.asarray(sample_by_feature, dtype=float)
    groups = list(groups)
    n = values.shape[0]
    scores, explained, _ = pca(values, min(values.shape), scale=True)
    keep = max(1, min(n - 1, int(np.searchsorted(np.cumsum(explained), variance_kept) + 1)))
    weighted = scores[:, :keep] * np.sqrt(explained[:keep] + 1e-12)
    correlation = sample_correlation(values, "spearman")

    metrics: List[Dict[str, object]] = []
    for index, group in enumerate(groups):
        peers = [i for i, value in enumerate(groups) if value == group and i != index]
        if peers:
            centroid = weighted[peers].mean(axis=0)
            distance = float(np.linalg.norm(weighted[index] - centroid))
            peer_spread = float(np.mean([np.linalg.norm(weighted[peer] - centroid) for peer in peers]))
            # With ONE peer the centroid is that peer and its spread is
            # exactly 0, so the ratio was distance / 1e-12 and every sample of a
            # two-per-group design was flagged. A spread cannot be measured
            # from one point; the ratio is unknown and the correlation decides.
            distance_ratio = distance / max(peer_spread, 1e-12) if len(peers) >= 2 else float("nan")
            peer_correlations = correlation[index, peers]
            median_correlation = (float(np.nanmedian(peer_correlations))
                                  if np.isfinite(peer_correlations).any() else float("nan"))
        else:
            distance = distance_ratio = median_correlation = float("nan")
        pca_flag = bool(np.isfinite(distance_ratio) and distance_ratio >= distance_ratio_flag)
        corr_flag = bool(np.isfinite(median_correlation) and median_correlation < correlation_flag)
        metrics.append({
            "distance": distance, "distance_ratio": distance_ratio,
            "median_correlation": median_correlation, "pca_flag": pca_flag, "correlation_flag": corr_flag,
            "status": ("OUTLIER_CANDIDATE" if pca_flag and corr_flag
                       else "WARN" if pca_flag or corr_flag else "PASS"),
        })
    return scores, explained, keep, metrics


# ---------------------------------------------------------------------------
# Paired tables (proteomics): do two tables' sample loadings move together?
# ---------------------------------------------------------------------------

def within_group_permutation_correlation(
    x: Sequence[float], y: Sequence[float], groups: Sequence[str], *, permutations: int, seed: int,
) -> Optional[Tuple[float, float]]:
    """``(r, p)``: Pearson r of ``x`` and ``y`` centred within each group, and its one-sided
    permutation p (``y`` shuffled within groups), ``(hits + 1) / (permutations + 1)``.

    None when nothing varies within its group, so there is nothing to correlate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(list(groups))
    xc, yc = x.copy(), y.copy()
    for g in np.unique(groups):
        idx = groups == g
        xc[idx] -= x[idx].mean()
        yc[idx] -= y[idx].mean()
    if not (np.isfinite(xc).all() and np.isfinite(yc).all()) or np.std(xc) == 0 or np.std(yc) == 0:
        return None
    r = float(np.corrcoef(xc, yc)[0, 1])
    rng = np.random.default_rng(seed)
    members = [np.where(groups == g)[0] for g in np.unique(groups)]
    hits = 0
    for _ in range(permutations):
        shuffled = yc.copy()
        for idx in members:
            shuffled[idx] = yc[rng.permutation(idx)]
        if np.corrcoef(xc, shuffled)[0, 1] >= r:
            hits += 1
    return r, (hits + 1) / (permutations + 1)


# ---------------------------------------------------------------------------
# The differential audit (bulk RNA's qc_ensemble): every statistic against a
# permutation null of the group labels
# ---------------------------------------------------------------------------

#: Fixes reviewer-facing significance at the raw p-value. The QC audits
#: the alpha the run actually used rather than proposing a different one.
DEFAULT_ALPHA = 0.05

#: Enough to put a 2.5th percentile somewhere trustworthy without making an
#: audit the slowest thing in the stage. 500 shuffles over ~4,500 genes is a
#: few seconds; the per-animal nulls use fewer because each shuffle costs a
#: full leave-one-out pass.
PERMUTATIONS = 500
PERMUTATIONS_PER_ANIMAL = 200

#: An audit must never be the long pole in the stage it audits. Permutations
#: stop at whichever comes first: the target count, or this budget. A short run
#: widens its own intervals honestly and says how many shuffles it managed.
TIME_BUDGET_SECONDS = 180.0
MIN_PERMUTATIONS = 40


def student_t_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Student's t p-values, one per gene, from ``paralome.stats``.

    ``a`` and ``b`` are samples x genes. The package's one t-test
    (``stats.student_t_from_moments`` on ``stats.group_moments``).
    """
    from paralome.stats import group_moments, student_t_from_moments

    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    genes = np.concatenate([first, second], axis=0).T
    n_first = first.shape[0]
    mean_a, var_a, n_a = group_moments(genes, range(n_first))
    mean_b, var_b, n_b = group_moments(genes, range(n_first, genes.shape[1]))
    _t, _df, p = student_t_from_moments(mean_a, var_a, n_a, mean_b, var_b, n_b)
    return np.asarray(p, dtype=float)


def significant(p_values: np.ndarray, alpha: float, alpha_metric: str = "p") -> np.ndarray:
    """Which of these (finite) p-values count as significant the way the run counts.

    By the run's own --alpha: for a Q cutoff the p-values are BH-adjusted
    first, over the genes the audit tests.
    """
    if str(alpha_metric).lower() == "q" and p_values.size:
        from paralome.stats import benjamini_hochberg

        return np.asarray(benjamini_hochberg(p_values)) < alpha
    return p_values < alpha


def significant_fraction(matrix: np.ndarray, left: Sequence[int],
                         right: Sequence[int], alpha: float, *,
                         alpha_metric: str = "p") -> float:
    p = student_t_p(matrix[list(left)], matrix[list(right)])
    p = p[np.isfinite(p)]
    return float(np.mean(significant(p, alpha, alpha_metric))) if p.size else float("nan")


def significant_count(matrix: np.ndarray, left: Sequence[int],
                      right: Sequence[int], alpha: float, *,
                      alpha_metric: str = "p") -> int:
    """The number of finite p-values passing alpha (not a fraction rescaled by the total gene count, which
    would overstate it when genes are untestable)."""
    p = student_t_p(matrix[list(left)], matrix[list(right)])
    p = p[np.isfinite(p)]
    return int(np.sum(significant(p, alpha, alpha_metric))) if p.size else 0


def pi_zero(p_values: np.ndarray, lam: float = 0.5) -> float:
    """Storey's estimate of the proportion of true nulls.

    Reported ONLY beside its permutation interval. On correlated genes this
    statistic is far less certain than its own arithmetic suggests -- the null
    interval on one real data set ran from 0.83 to 1.00, so an observed 0.93 means
    nothing on its own.
    """
    finite = p_values[np.isfinite(p_values)]
    if finite.size == 0:
        return float("nan")
    return float(min(1.0, np.mean(finite > lam) / (1.0 - lam)))


def effective_independent_tests(null_fractions: np.ndarray, alpha: float) -> float:
    """How many genuinely independent tests the gene set is worth.

    Under independence the significant fraction has variance alpha(1-alpha)/n.
    Inverting the OBSERVED permutation variance gives the n that would have
    produced it. The gap between this and the gene count is the single number
    that explains why gene-count reasoning misleads here.
    """
    variance = float(np.var(null_fractions))
    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    return alpha * (1.0 - alpha) / variance


def responder_scores(matrix: np.ndarray, control: Sequence[int],
                     treated: Sequence[int]) -> Dict[int, float]:
    """Project each animal onto the treatment axis, LEAVE-ONE-OUT.

    The axis is rebuilt without the animal being scored, so a non-responder
    cannot flatten the very axis it is measured against. Without that, a strong
    non-responder drags the group centroid toward the controls and then scores
    as unremarkable -- it hides itself.
    """
    scores: Dict[int, float] = {}
    for index in list(control) + list(treated):
        others_control = [i for i in control if i != index]
        others_treated = [i for i in treated if i != index]
        if len(others_control) < 2 or len(others_treated) < 2:
            continue
        axis = matrix[others_treated].mean(axis=0) - matrix[others_control].mean(axis=0)
        norm = np.linalg.norm(axis)
        if norm == 0:
            continue
        axis = axis / norm
        midpoint = (matrix[others_control].mean(axis=0)
                    + matrix[others_treated].mean(axis=0)) / 2.0
        scores[index] = float((matrix[index] - midpoint) @ axis)
    return scores


def responder_z(matrix, control, treated) -> Tuple[Dict[int, float], float]:
    """Each treated animal's responder score as a z against the controls' scores, and the lowest."""
    scores = responder_scores(matrix, control, treated)
    reference = [scores[i] for i in control if i in scores]
    if len(reference) < 2:
        return {}, float("nan")
    spread = float(np.std(reference)) or 1e-9
    centre = float(np.mean(reference))
    z = {i: (scores[i] - centre) / spread for i in treated if i in scores}
    return z, (min(z.values()) if z else float("nan"))


def leave_one_out_gain(matrix, control, treated, alpha, *,
                       alpha_metric: str = "p") -> Tuple[int, int, Optional[int]]:
    """Genes at alpha, and the best single removal. Returns (base, gain, who)."""
    how = {"alpha_metric": alpha_metric}
    base = significant_count(matrix, control, treated, alpha, **how)
    best, who = 0, None
    for index in list(control) + list(treated):
        left = [i for i in control if i != index]
        right = [i for i in treated if i != index]
        if len(left) < 2 or len(right) < 2:
            continue
        count = significant_count(matrix, left, right, alpha, **how)
        if count - base > best:
            best, who = count - base, index
    return base, best, who


def audit_comparison(
    matrix: np.ndarray,
    control: Sequence[int],
    treated: Sequence[int],
    *,
    alpha: float = DEFAULT_ALPHA,
    permutations: int = PERMUTATIONS,
    animal_permutations: int = PERMUTATIONS_PER_ANIMAL,
    time_budget: float = TIME_BUDGET_SECONDS,
    seed: int = 0,
    alpha_metric: str = "p",
) -> Dict[str, object]:
    """Everything the differential audit knows about one comparison, each against its null.

    The null shuffles the group labels, which respects whatever correlation
    structure the genes have because it never touches them. ``alpha_metric``
    ("p" or "q") is the run's own.
    """
    control, treated = list(control), list(treated)
    pooled = np.array(control + treated)
    split = len(control)
    rng = np.random.default_rng(seed)

    by_p = str(alpha_metric).lower() != "q"
    observed_p = student_t_p(matrix[control], matrix[treated])
    observed_p = observed_p[np.isfinite(observed_p)]
    observed_fraction = (float(np.mean(significant(observed_p, alpha, alpha_metric)))
                         if observed_p.size else float("nan"))
    grid = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

    deadline = _time.monotonic() + max(1.0, time_budget) * 0.6

    null_fractions, null_histograms, null_calibration, null_pi0 = [], [], [], []
    for attempt in range(max(1, permutations)):
        if attempt >= MIN_PERMUTATIONS and _time.monotonic() > deadline:
            break
        order = rng.permutation(len(pooled))
        left = [pooled[i] for i in order[:split]]
        right = [pooled[i] for i in order[split:]]
        p = student_t_p(matrix[left], matrix[right])
        p = p[np.isfinite(p)]
        if p.size == 0:
            continue
        null_fractions.append(float(np.mean(significant(p, alpha, alpha_metric))))
        null_histograms.append(np.histogram(p, bins=20, range=(0, 1))[0])
        null_calibration.append([float(np.mean(p < g) / g) for g in grid])
        null_pi0.append(pi_zero(p))
    null_fractions = np.asarray(null_fractions)
    null_histograms = np.asarray(null_histograms)
    null_calibration = np.asarray(null_calibration)
    null_pi0 = np.asarray(null_pi0)

    z_scores, observed_min_z = responder_z(matrix, control, treated)
    base, gain, culprit = leave_one_out_gain(
        matrix, control, treated, alpha, alpha_metric=alpha_metric)

    deadline = _time.monotonic() + max(1.0, time_budget) * 0.4
    null_min_z, null_gain = [], []
    for attempt in range(max(1, animal_permutations)):
        if attempt >= MIN_PERMUTATIONS and _time.monotonic() > deadline:
            break
        order = rng.permutation(len(pooled))
        left = [pooled[i] for i in order[:split]]
        right = [pooled[i] for i in order[split:]]
        _z, minimum = responder_z(matrix, left, right)
        if np.isfinite(minimum):
            null_min_z.append(minimum)
        null_gain.append(leave_one_out_gain(
            matrix, left, right, alpha, alpha_metric=alpha_metric)[1])
    null_min_z = np.asarray(null_min_z)
    null_gain = np.asarray(null_gain)

    cutoff = float(np.percentile(null_min_z, 2.5)) if null_min_z.size else float("-inf")
    return {
        "alpha": alpha,
        "alpha_metric": "p" if by_p else "q",
        "statistic": "t",
        "permutations_used": int(null_fractions.size),
        "animal_permutations_used": int(null_gain.size),
        "n_control": len(control), "n_treated": len(treated),
        "genes": int(observed_p.size),
        "observed_p": observed_p,
        "observed_fraction": observed_fraction,
        "significant": int(round(observed_fraction * observed_p.size)),
        # alpha x genes is what chance gives a raw-p cutoff; a q cutoff has no
        # such figure, and the permutation null beside it is the reference.
        "expected_by_chance": alpha * observed_p.size if by_p else float("nan"),
        "pi0": pi_zero(observed_p),
        "pi0_null": null_pi0,
        "grid": grid,
        "observed_calibration": [float(np.mean(observed_p < g) / g) for g in grid],
        "null_calibration": null_calibration,
        "null_histograms": null_histograms,
        "observed_histogram": np.histogram(observed_p, bins=20, range=(0, 1))[0],
        "null_fractions": null_fractions,
        "fraction_percentile": (float(np.mean(null_fractions < observed_fraction) * 100)
                                if null_fractions.size else float("nan")),
        "inside_null": bool(
            null_fractions.size
            and np.percentile(null_fractions, 2.5) <= observed_fraction
            <= np.percentile(null_fractions, 97.5)
        ),
        "effective_tests": (effective_independent_tests(null_fractions, alpha)
                            if by_p else float("nan")),
        "responder_z": z_scores,
        "responder_cutoff": cutoff,
        "flagged": sorted(i for i, z in z_scores.items() if z <= cutoff),
        "responder_p": (float(np.mean(null_min_z <= observed_min_z))
                        if null_min_z.size and np.isfinite(observed_min_z) else float("nan")),
        "loo_base": base, "loo_gain": gain, "loo_culprit": culprit,
        "loo_null": null_gain,
        "loo_p": (float(np.mean(null_gain >= gain)) if null_gain.size else float("nan")),
    }


def array_quality(matrix: np.ndarray, cells: Mapping[str, Sequence[int]]) -> Dict[str, object]:
    """Median correlation of each animal to its own cell-mates, and the robust rule on it.

    Catches a bad library or a swapped sample, which is a DIFFERENT failure
    from a non-responder: a non-responder has a perfectly healthy array and
    simply did not move. Flagged below ``MODIFIED_Z_CUTOFF`` scaled MADs
    under the median (``robust_centre_scale``), the pseudobulk QC's rule.
    """
    count = matrix.shape[0]
    correlation = sample_correlation(matrix)
    median_r = np.zeros(count)
    for members in cells.values():
        for index in members:
            others = [i for i in members if i != index]
            median_r[index] = float(np.median(correlation[index, others])) if others else np.nan
    centre, scale = robust_centre_scale(median_r)
    robust_z = (median_r - centre) / scale if scale > 0 else np.zeros(count)
    return {
        "median_r": median_r,
        "robust_z": robust_z,
        "cutoff_r": centre - MODIFIED_Z_CUTOFF * scale,
        "flagged": [int(i) for i in np.where(robust_z < -MODIFIED_Z_CUTOFF)[0]],
    }


def variance_runaway(expression_bins: Sequence[float], sd_bins: Sequence[float],
                     gate: float) -> Optional[bool]:
    """Whether SD climbs as expression falls below the gate (panel B's question).

    The read-out tested ``first bin < 1.5 x the largest bin``, which
    is true of every curve, so it could only ever say "not present". The low
    end is the bins under the gate (the lowest quarter when nothing is gated);
    the runaway is present when its highest SD is more than 1.5 times the
    median SD of the rest. None when there is nothing to compare.
    """
    pairs = [(x, y) for x, y in zip(expression_bins, sd_bins) if np.isfinite(x) and np.isfinite(y)]
    low = [y for x, y in pairs if x < gate]
    high = [y for x, y in pairs if x >= gate]
    if not low or not high:
        ordered = [y for _x, y in sorted(pairs)]
        cut = len(ordered) // 4
        low, high = ordered[:cut], ordered[cut:]
    if not low or not high:
        return None
    reference = float(np.median(high))
    return bool(reference > 0 and max(low) > 1.5 * reference)
