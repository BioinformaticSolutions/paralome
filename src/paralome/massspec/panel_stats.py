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

"""The panel statistics: the elastic net, its cross-validation, chance by shuffled labels and the
bootstrap -- the statistics module of ``paralome.massspec.panel`` (which holds the design's comparisons and the tables
it writes), as ``paralome.burden.stats`` is gene burden's. What it computes is described there.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np




ALPHA = 0.5
N_LAMBDA = 20
REPEATS = 10
PERMUTATIONS = 20
PERMUTATION_REPEATS = 5   # each shuffling's cross-validation: fewer repeats than the real labels' -- its mean varies
                          # more, so the p-value can only come out larger (conservative), never smaller
INNER_FOLDS = 3           # the cross-validation that chooses the penalty inside each training part
BOOTSTRAPS = 100
CV_TOL = 1e-5            # the cross-validation, permutation and bootstrap fits (held-out predictions move < 1e-3)
MAX_FOLDS = 5
MIN_PER_GROUP = 3
SEED = 20261120

TWO_GROUP, MULTICLASS, REGRESSION = "two_group", "multiclass", "regression"


# ---------------------------------------------------------------------------
# the elastic net
# ---------------------------------------------------------------------------

def _loss_grad(family: str, A: np.ndarray, y: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean loss and its gradient in B (columns: one per output) for the design A (with its intercept column)."""
    n = A.shape[0]
    eta = A @ B
    if family == REGRESSION:
        r = eta[:, 0] - y
        return float(r @ r) / (2 * n), (A.T @ r)[:, None] / n
    if family == TWO_GROUP:
        z = eta[:, 0]
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        return loss, (A.T @ (p - y))[:, None] / n
    eta = eta - eta.max(axis=1, keepdims=True)
    e = np.exp(eta)
    P = e / e.sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(np.maximum(P[np.arange(n), y], 1e-300))))
    Y = np.zeros_like(P)
    Y[np.arange(n), y] = 1.0
    return loss, A.T @ (P - Y) / n


def _prox(B: np.ndarray, step: float, lam: float, penalised: np.ndarray) -> np.ndarray:
    out = B.copy()
    if lam > 0:
        z = B[penalised]
        out[penalised] = np.sign(z) * np.maximum(np.abs(z) - step * lam * ALPHA, 0.0) / (1.0 + step * lam * (1 - ALPHA))
    return out


def _grad_fn(family: str, A: np.ndarray, y: np.ndarray):
    """The gradient of the mean loss in B, as a closure over the design (no loss computed: FISTA needs none)."""
    n = A.shape[0]
    At = A.T / n
    if family == REGRESSION:
        return lambda B: At @ ((A @ B)[:, 0] - y)[:, None]
    if family == TWO_GROUP:
        return lambda B: At @ (1.0 / (1.0 + np.exp(-(A @ B)[:, 0])) - y)[:, None]
    Y = np.zeros((n, int(y.max()) + 1 if len(y) else 1))
    Y[np.arange(n), y] = 1.0

    def g(B):
        eta = A @ B
        eta -= eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return At @ (e / e.sum(axis=1, keepdims=True) - Y[:, :B.shape[1]])
    return g


def _fista(family: str, A: np.ndarray, y: np.ndarray, B0: np.ndarray, lam: float, penalised: np.ndarray,
           lipschitz: float, tol: float = 1e-7, max_iter: int = 5000, grad=None) -> np.ndarray:
    step = 1.0 / lipschitz
    grad = grad or _grad_fn(family, A, y)
    shrink = step * lam * ALPHA
    scale = 1.0 / (1.0 + step * lam * (1 - ALPHA))
    pen = penalised
    B, Z, t = B0.copy(), B0.copy(), 1.0
    for _ in range(max_iter):
        B_new = Z - step * grad(Z)
        if lam > 0:
            z = B_new[pen]
            B_new[pen] = np.sign(z) * np.maximum(np.abs(z) - shrink, 0.0) * scale
        diff = B_new - B
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        Z = B_new + ((t - 1) / t_new) * diff
        if np.vdot(diff, Z - B_new) > 0:                    # adaptive restart: the momentum turned against the descent
            Z, t_new = B_new.copy(), 1.0
        if np.abs(diff).max() <= tol * (1.0 + np.abs(B_new).max()):
            return B_new
        B, t = B_new, t_new
    return B


class Fitted:
    """A path of elastic-net fits on one standardised design."""

    def __init__(self, family: str, X: np.ndarray, Z: np.ndarray, y: np.ndarray, n_classes: int,
                 lambdas: Optional[np.ndarray] = None, tol: float = 1e-7):
        self.family, self.n_classes = family, n_classes
        self.mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.zmean = Z.mean(axis=0) if Z.shape[1] else np.zeros(0)
        A = self.design(X, Z)
        p_pen = X.shape[1]
        self.penalised = np.zeros(A.shape[1], dtype=bool)
        self.penalised[1 + Z.shape[1]:] = True
        k = 1 if family in (TWO_GROUP, REGRESSION) else n_classes
        self.y = y.astype(float) if family != MULTICLASS else y.astype(int)
        if family == REGRESSION:
            self.ymean = float(np.mean(y))
            self.y = self.y - self.ymean
        else:
            self.ymean = 0.0
        smax = float(np.linalg.norm(A, 2)) ** 2 / A.shape[0]
        self.L = smax * (1.0 if family == REGRESSION else 0.25 if family == TWO_GROUP else 0.5) + 1e-12
        B = np.zeros((A.shape[1], k))
        grad = _grad_fn(family, A, self.y)
        B = _fista(family, A, self.y, B, 1e30, self.penalised, self.L, tol=tol, grad=grad)   # the unpenalised part alone
        if lambdas is None:
            _, g = _loss_grad(family, A, self.y, B)
            lam_max = float(np.max(np.abs(g[self.penalised]))) / ALPHA if p_pen else 1.0
            ratio = 0.01 if A.shape[0] > p_pen else 0.05
            lambdas = lam_max * np.logspace(0, math.log10(ratio), N_LAMBDA) if lam_max > 0 else np.full(N_LAMBDA, 1e-6)
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.coefs = []
        for lam in self.lambdas:
            B = _fista(family, A, self.y, B, float(lam), self.penalised, self.L, tol=tol, grad=grad)
            self.coefs.append(B.copy())
        self.p_pen, self.nz = p_pen, Z.shape[1]

    def design(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        cols = [np.ones((X.shape[0], 1))]
        if Z.shape[1]:
            cols.append(Z - self.zmean)
        cols.append((X - self.mean) / self.sd)
        return np.hstack(cols)

    def predict(self, X: np.ndarray, Z: np.ndarray, i: int) -> np.ndarray:
        """Two groups: P(level); several: class probabilities; a covariate: its predicted value."""
        eta = self.design(X, Z) @ self.coefs[i]
        if self.family == REGRESSION:
            return eta[:, 0] + self.ymean
        if self.family == TWO_GROUP:
            return 1.0 / (1.0 + np.exp(-eta[:, 0]))
        eta = eta - eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(axis=1, keepdims=True)

    def weights(self, i: int) -> np.ndarray:
        """The features' weights (features x outputs), per standard deviation."""
        return self.coefs[i][1 + self.nz:]


def _deviance(family: str, truth: np.ndarray, pred: np.ndarray) -> float:
    if family == REGRESSION:
        return float(np.mean((truth - pred) ** 2))
    if family == TWO_GROUP:
        p = np.clip(pred, 1e-12, 1 - 1e-12)
        return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))
    return float(-np.mean(np.log(np.clip(pred[np.arange(len(truth)), truth], 1e-12, 1.0))))


# ---------------------------------------------------------------------------
# folds, scores
# ---------------------------------------------------------------------------

def _folds(labels: np.ndarray, blocks: Optional[np.ndarray], k: int, rng: np.random.Generator,
           family: str) -> np.ndarray:
    """A fold for each sample: groups (or the covariate's order) spread over the folds; a block kept whole."""
    units = np.arange(len(labels)) if blocks is None else np.unique(blocks, return_inverse=True)[1]
    n_units = int(units.max()) + 1
    if family == REGRESSION:
        key = np.asarray([np.mean(labels[units == u]) for u in range(n_units)])
        order = np.argsort(key + rng.uniform(0, 1e-9, n_units))
        fold_of_unit = np.empty(n_units, dtype=int)
        for start in range(0, n_units, k):                      # each run of k neighbours spread over the folds
            chunk = order[start:start + k]
            fold_of_unit[chunk] = rng.permutation(k)[:len(chunk)]
    else:
        # a block goes with the group it holds most of (its rarest when tied), so every group reaches every fold
        counts = np.bincount(labels.astype(int))
        unit_label = np.asarray([min(np.unique(labels[units == u]).astype(int), key=lambda c: (-np.sum(labels[units == u] == c), counts[c]))
                                 for u in range(n_units)])
        fold_of_unit = np.empty(n_units, dtype=int)
        offset = 0
        for c in np.unique(unit_label):
            members = rng.permutation(np.flatnonzero(unit_label == c))
            fold_of_unit[members] = (np.arange(len(members)) + offset) % k
            offset += len(members)
    return fold_of_unit[units]


def _auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC of a two-group truth (1 = positive), ties counted half."""
    from scipy.stats import rankdata

    pos, neg = truth == 1, truth == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if not n1 or not n0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _score(family: str, truth: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    if family == REGRESSION:
        ss = float(np.sum((truth - truth.mean()) ** 2))
        return 1.0 - float(np.sum((truth - pred) ** 2)) / ss if ss > 0 else float("nan")
    if family == TWO_GROUP:
        return _auc(truth, pred)
    aucs = [_auc((truth == c).astype(int), pred[:, c]) for c in range(n_classes) if (truth == c).any()]
    return float(np.nanmean(aucs)) if aucs else float("nan")


def _choose_lambda(family: str, X, Z, y, n_classes, blocks, k, rng, tol: float = CV_TOL) -> Tuple[Fitted, int]:
    """The path on these samples, and the penalty its own k-fold cross-validation chooses (least deviance)."""
    full = Fitted(family, X, Z, y, n_classes, tol=tol)
    folds = _folds(y, blocks, k, rng, family)
    dev = np.zeros(len(full.lambdas))
    for f in range(k):
        test = folds == f
        train = ~test
        if not test.any() or (family != REGRESSION and len(np.unique(y[train])) < (2 if family == TWO_GROUP else n_classes)):
            continue
        part = Fitted(family, X[train], Z[train], y[train], n_classes, lambdas=full.lambdas, tol=CV_TOL)
        for i in range(len(full.lambdas)):
            dev[i] += _deviance(family, y[test], part.predict(X[test], Z[test], i)) * test.sum()
    return full, int(np.argmin(dev))


def _single(family: str, x: np.ndarray, Z: np.ndarray, y: np.ndarray, n_classes: int, train: np.ndarray,
            test: np.ndarray) -> np.ndarray:
    """One feature's held-out prediction, from the training part alone (the factors held taken off it first)."""
    xt, xs = x[train], x[test]
    if Z.shape[1]:
        D = np.column_stack([np.ones(train.sum()), Z[train]])
        beta, *_ = np.linalg.lstsq(D, xt, rcond=None)
        xt = xt - D @ beta
        xs = xs - np.column_stack([np.ones(test.sum()), Z[test]]) @ beta
    if family == REGRESSION:
        slope = np.polyfit(xt, y[train], 1) if np.ptp(xt) > 0 else np.asarray([0.0, np.mean(y[train])])
        return np.polyval(slope, xs)
    if family == TWO_GROUP:
        direction = 1.0 if np.mean(xt[y[train] == 1]) >= np.mean(xt[y[train] == 0]) else -1.0
        return direction * xs
    centres = np.asarray([np.mean(xt[y[train] == c]) if (y[train] == c).any() else np.inf for c in range(n_classes)])
    score = -np.abs(xs[:, None] - centres[None, :])
    e = np.exp(score - score.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# one panel
# ---------------------------------------------------------------------------

def _centre_in_blocks(X: np.ndarray, y: np.ndarray, blocks: Optional[np.ndarray], family: str):
    if blocks is None:
        return X, y
    X = X.copy()
    y = y.astype(float).copy() if family == REGRESSION else y
    for b in np.unique(blocks):
        on = blocks == b
        X[on] -= X[on].mean(axis=0)
        if family == REGRESSION:
            y[on] -= y[on].mean()
    return X, y


def _train_score(family: str, x: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """A feature's own score on the training part (to choose the best single feature inside each fold)."""
    if family == REGRESSION:
        return float(np.corrcoef(x, y)[0, 1] ** 2) if np.ptp(x) > 0 and np.ptp(y) > 0 else 0.0
    if family == TWO_GROUP:
        a = _auc(y, x)
        return max(a, 1 - a)
    centres = np.asarray([np.mean(x[y == c]) if (y == c).any() else np.inf for c in range(n_classes)])
    score = -np.abs(x[:, None] - centres[None, :])
    return _score(family, y, score, n_classes)


def _cv(family, X, Z, y, blocks, n_classes, k, repeats, rng, singles: bool = True):
    """``repeats`` repeats of k-fold cross-validation: the panel's held-out score in each, each feature's own, and
    the best single feature's (chosen in each training part by its training score, scored on the held-out part)."""
    panel_scores, single_scores, best_scores = [], [], []
    for _ in range(repeats):
        folds = _folds(y, blocks, k, rng, family)
        pred = np.zeros((len(y), n_classes)) if family == MULTICLASS else np.zeros(len(y))
        single = np.zeros((X.shape[1],) + pred.shape) if singles else None
        best_pred = np.zeros_like(pred)
        for f in range(k):
            test, train = folds == f, folds != f
            if not test.any():
                continue
            inner_blocks = blocks[train] if blocks is not None else None
            k_inner = max(2, min(INNER_FOLDS, _k(family, y[train], inner_blocks, n_classes)))
            fitted, best = _choose_lambda(family, X[train], Z[train], y[train], n_classes, inner_blocks, k_inner, rng)
            pred[test] = fitted.predict(X[test], Z[test], best)
            chosen = int(np.argmax([_train_score(family, X[train, j], y[train], n_classes) for j in range(X.shape[1])]))
            best_pred[test] = _single(family, X[:, chosen], Z, y, n_classes, train, test)
            if singles:
                for j in range(X.shape[1]):
                    single[j][test] = _single(family, X[:, j], Z, y, n_classes, train, test)
        panel_scores.append(_score(family, y, pred, n_classes))
        best_scores.append(_score(family, y, best_pred, n_classes))
        if singles:
            single_scores.append([_score(family, y, single[j], n_classes) for j in range(X.shape[1])])
    return np.asarray(panel_scores), np.asarray(single_scores), np.asarray(best_scores)


def _permute(y: np.ndarray, blocks: Optional[np.ndarray], rng) -> np.ndarray:
    """The labels shuffled -- within each block when blocked, so a subject keeps its own set of labels."""
    if blocks is None:
        return rng.permutation(y)
    out = y.copy()
    for b in np.unique(blocks):
        on = np.flatnonzero(blocks == b)
        out[on] = y[rng.permutation(on)]
    return out


# ---------------------------------------------------------------------------
# the work, in independent pieces (run across worker processes)
# ---------------------------------------------------------------------------
# Every cross-validation repeat, every repeat of every shuffling and every bootstrap resample is its own piece,
# with its own random stream spawned from the seed: the same data give the same numbers whatever the number of
# workers, run in any order.

def _repeat_piece(family, Xc, Z, yc, blocks, n_classes, k, stream):
    """One repeat of the real labels' cross-validation: the panel's, each feature's and the best single's score."""
    p, s, b = _cv(family, Xc, Z, yc, blocks, n_classes, k, 1, np.random.default_rng(stream))
    return float(p[0]), s[0], float(b[0])


def _shuffle_piece(family, X, Z, y_raw, blocks, n_classes, k, shuffle_stream, stream):
    """One repeat of one shuffling's cross-validation: the panel's and the best single's score. The shuffling is
    drawn from its own stream, so every repeat of it shuffles the labels the same way."""
    yp = _permute(y_raw, blocks, np.random.default_rng(shuffle_stream))
    Xp, ypc = _centre_in_blocks(X, yp, blocks, family)
    p, _s, b = _cv(family, Xp, Z, ypc, blocks, n_classes, k, 1, np.random.default_rng(stream), singles=False)
    return float(p[0]), float(b[0])


def _bootstrap_piece(family, X, Z, y_raw, blocks, n_classes, lam, streams):
    """Bootstrap resamples fitted at the full data's penalty: each one's weights, or None when a group is lost."""
    out = []
    for stream in streams:
        idx = _resample(y_raw, blocks, np.random.default_rng(stream), family)
        Xb, yb = _centre_in_blocks(X[idx], y_raw[idx], blocks[idx] if blocks is not None else None, family)
        if family != REGRESSION and len(np.unique(yb)) < (2 if family == TWO_GROUP else n_classes):
            out.append(None)
            continue
        out.append(Fitted(family, Xb, Z[idx], yb, n_classes, lambdas=lam, tol=CV_TOL).weights(len(lam) - 1))
    return out


def _run_pieces(pool, function, argument_lists):
    """Each piece's result, in order: across ``pool`` (a concurrent.futures executor) when given, else here."""
    if pool is None:
        return [function(*arguments) for arguments in argument_lists]
    futures = [pool.submit(function, *arguments) for arguments in argument_lists]
    return [future.result() for future in futures]


BOOTSTRAP_PIECE = 10      # resamples per piece (each is quick)


def fit_panel(family: str, X: np.ndarray, y: np.ndarray, *, Z: Optional[np.ndarray] = None,
              blocks: Optional[np.ndarray] = None, n_classes: int = 2, repeats: Optional[int] = None,
              permutations: Optional[int] = None, permutation_repeats: Optional[int] = None,
              bootstraps: Optional[int] = None, seed: int = SEED, pool=None) -> Dict[str, object]:
    """Fit one panel: X samples x features (log2), y the groups (0..k-1) or the covariate.

    The cross-validated scores (panel, each feature, the best single feature chosen inside each fold), the same
    cross-validation on ``permutations`` shufflings of the labels (what this design and these samples give with no
    relationship at all), and the bootstrap's selection counts. The caller has already checked the smallest group.
    ``pool``: an executor the pieces run across (``panel_pool``); the numbers are the same without one."""
    repeats = REPEATS if repeats is None else repeats
    permutations = PERMUTATIONS if permutations is None else permutations
    permutation_repeats = PERMUTATION_REPEATS if permutation_repeats is None else permutation_repeats
    bootstraps = BOOTSTRAPS if bootstraps is None else bootstraps
    X = np.asarray(X, dtype=float)
    Z = np.zeros((X.shape[0], 0)) if Z is None else np.asarray(Z, dtype=float)
    y_raw = np.asarray(y)
    if blocks is not None:
        blocks = np.asarray(blocks)
    root = np.random.SeedSequence(seed)
    real_streams, shuffle_roots, boot_streams, (full_stream,) = (
        root.spawn(repeats), root.spawn(permutations), root.spawn(bootstraps), root.spawn(1))
    k = _k(family, y_raw, blocks, n_classes)
    Xc, yc = _centre_in_blocks(X, y_raw, blocks, family)

    real = _run_pieces(pool, _repeat_piece, [(family, Xc, Z, yc, blocks, n_classes, k, s) for s in real_streams])
    shuffle_args = []
    for shuffle_root in shuffle_roots:
        own, *repeat_streams = shuffle_root.spawn(1 + permutation_repeats)
        shuffle_args += [(family, X, Z, y_raw, blocks, n_classes, k, own, s) for s in repeat_streams]
    shuffled = _run_pieces(pool, _shuffle_piece, shuffle_args)
    panel_scores = np.asarray([r[0] for r in real])
    single_scores = np.asarray([r[1] for r in real])
    best_scores = np.asarray([r[2] for r in real])
    perm_panel = [float(np.nanmean([shuffled[i * permutation_repeats + r][0] for r in range(permutation_repeats)]))
                  for i in range(permutations)]
    perm_best = [float(np.nanmean([shuffled[i * permutation_repeats + r][1] for r in range(permutation_repeats)]))
                 for i in range(permutations)]

    # stability: the penalty the full data choose, refitted on resamples
    full, best = _choose_lambda(family, Xc, Z, yc, n_classes, blocks, min(k, INNER_FOLDS + 2),
                                np.random.default_rng(full_stream), tol=1e-7)
    outputs = 1 if family != MULTICLASS else n_classes
    lam = full.lambdas[: best + 1]
    chunks = [boot_streams[i:i + BOOTSTRAP_PIECE] for i in range(0, bootstraps, BOOTSTRAP_PIECE)]
    boots = [w for piece in _run_pieces(pool, _bootstrap_piece,
                                         [(family, X, Z, y_raw, blocks, n_classes, lam, c) for c in chunks])
             for w in piece]
    chosen = np.zeros(X.shape[1])
    positive = np.zeros((X.shape[1], outputs))
    weights = np.full((bootstraps, X.shape[1], outputs), np.nan)
    fitted_boots = 0
    for b, w in enumerate(boots):
        if w is None:
            continue
        weights[b] = w
        chosen += np.any(w != 0, axis=1)
        positive += (w > 0)
        fitted_boots += 1
    obs = float(np.nanmean(panel_scores))
    obs_best = float(np.nanmean(best_scores))
    return {
        "family": family, "k": k, "repeats": repeats, "permutations": permutations, "bootstraps": fitted_boots,
        "panel_scores": panel_scores, "single_scores": single_scores, "best_scores": best_scores,
        "perm_panel": np.asarray(perm_panel), "perm_best": np.asarray(perm_best),
        "panel_p": (1 + sum(v >= obs for v in perm_panel)) / (1 + permutations) if permutations else float("nan"),
        "best_p": (1 + sum(v >= obs_best for v in perm_best)) / (1 + permutations) if permutations else float("nan"),
        "selected": chosen / max(fitted_boots, 1), "positive": positive / max(fitted_boots, 1), "weights": weights,
        "lambda": float(full.lambdas[best]), "full_weights": full.weights(best),
        "n_selected_full": int(np.sum(np.any(full.weights(best) != 0, axis=1))),
    }


# ---------------------------------------------------------------------------
# the workers
# ---------------------------------------------------------------------------

#: The maths libraries' own thread pools, held to one thread in each worker: the panel's matrices are a few dozen
#: rows, the parallel work is the pieces, and a pool per worker would put cores x workers threads on the node.
_ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def available_cpus() -> int:
    """The CPUs this process may run on; inside a SLURM job, no more than its allocation (SLURM_CPUS_PER_TASK)."""
    import os

    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    allocated = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocated.isdigit() and int(allocated) > 0:
        cpus = min(cpus, int(allocated))
    return max(1, cpus)


class panel_pool:
    """``with panel_pool(threads) as pool``: ``threads`` worker processes (0: ``available_cpus()``), each started
    fresh with its maths libraries at one thread; None (run here) for one."""

    def __init__(self, threads: int = 0):
        self.workers = available_cpus() if not threads else max(1, int(threads))
        self.executor = None
        self.before = {}

    def __enter__(self):
        if self.workers <= 1:
            return None
        import multiprocessing
        import os
        from concurrent.futures import ProcessPoolExecutor

        # Held for the pool's life: a worker may be started at any submit, and reads them when it starts.
        self.before = {name: os.environ.get(name) for name in _ONE_THREAD}
        os.environ.update({name: "1" for name in _ONE_THREAD})
        self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
        return self.executor

    def __exit__(self, *exc):
        import os

        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        for name, value in self.before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


def _k(family, y, blocks, n_classes) -> int:
    return min(MAX_FOLDS, smallest_group(family, y, blocks, n_classes))


def smallest_group(family: str, y: np.ndarray, blocks: Optional[np.ndarray], n_classes: int) -> int:
    """The smallest group's count -- in blocks when blocked; for a covariate, the samples (blocks)."""
    if family == REGRESSION:
        return len(np.unique(blocks)) if blocks is not None else len(y)
    counts = []
    for c in range(n_classes if family == MULTICLASS else 2):
        on = y == c
        counts.append(len(np.unique(blocks[on])) if blocks is not None else int(on.sum()))
    return min(counts) if counts else 0


def _resample(y, blocks, rng, family) -> np.ndarray:
    if blocks is not None:
        units = np.unique(blocks)
        pick = rng.choice(units, size=len(units), replace=True)
        return np.concatenate([np.flatnonzero(blocks == u) for u in pick])
    if family == REGRESSION:
        return rng.choice(len(y), size=len(y), replace=True)
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=int((y == c).sum()), replace=True)
                           for c in np.unique(y)])
