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

"""Each proband's genetic ancestry, as a mixture of gnomAD populations.

The expected number of carriers of a gene depends on who the
probands are: a PTV common in one population and absent from another changes
the expectation for every proband from the first. When the sheet does not say,
the mixture is estimated from the cohort's own common coding SNPs.

THE MODEL (supervised, as in frappe/ADMIXTURE with the allele frequencies
fixed). Population k has allele frequency f[l,k] at SNP l (gnomAD's
per-population exome AF, clamped to [AF_CLAMP, 1-AF_CLAMP]). A proband with
mixture weights w carries the alternate allele on each of its two copies with
probability p[l] = sum_k w[k] f[l,k], independently (Hardy-Weinberg), so its
genotype g[l] in {0,1,2} has likelihood Binomial(2, p[l]). The weights are the
maximum-likelihood point on the simplex, found by EM with the population of
origin of every allele copy as the hidden variable:

    E: an alternate copy came from k with probability w[k] f[l,k] / p[l],
       a reference copy with w[k] (1-f[l,k]) / (1-p[l]);
    M: w[k] = the mean of those probabilities over all observed copies.

Each step cannot lower the likelihood, the weights stay on the simplex, and
the problem is concave in w, so the fixed point is the maximum.

SNPs are treated as independent; nearby SNPs in linkage disequilibrium make
the likelihood overconfident but leave the point estimate sound. The SNPs are
taken from the cohort's own calls -- no external panel -- so the estimate needs
no data beyond the annotated VCF.

``confidence`` is the largest weight: 1.0 is a proband wholly from one
population; an admixed proband has a lower value and its expectation is a
genuine mixture, which is the point of using weights rather than a label.

SEX. The burden test needs each proband's sex on chrX and chrY
outside the pseudo-autosomal regions (a male has one copy there). The sheet's
``sex`` column is used when it says male or female. Otherwise sex is inferred
from the same common coding SNPs, on chrX outside the PARs, with the
inbreeding coefficient PLINK's check-sex uses:

    F = 1 - (observed heterozygous calls) / (expected, sum 2p(1-p))

over the SNPs called in that proband (p = gnomAD's overall AF). A male has
one X, so a genuine heterozygous call there is impossible and F is near 1; a
female's F is near 0. F >= ``SEX_F_MALE`` (0.8) is male, F <= ``SEX_F_FEMALE``
(0.2) female; in between, or with fewer than ``MIN_SEX_SNPS`` (50) called
SNPs, the sex is unknown, and the proband is left out of the chrX and chrY
genes rather than guessed. The ancestry table reports what was used and where
it came from (sheet / inferred / unknown) with the SNP count and F.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from paralome.burden.qualify import ALL_POPULATIONS

AF_CLAMP = 1e-3
EM_MAX_ITERATIONS = 2000
EM_TOLERANCE = 1e-8
#: Fewer informative SNPs than this and no estimate is made: the proband's
#: expectation then uses gnomAD's overall frequencies, and the table says so.
MIN_SNPS = 200

AUTO = "auto"
OVERALL = "all"


@dataclass(frozen=True)
class AncestryEstimate:
    sample_id: str
    method: str                       # estimated | given (option) | given (sheet) | overall
    weights: Optional[Mapping[str, float]]   # None: use gnomAD's overall AF
    assigned: str
    confidence: Optional[float]
    n_snps: int
    log_likelihood: Optional[float]
    note: str = ""


def fit_mixture(dosages: np.ndarray, afs: np.ndarray) -> tuple[np.ndarray, float, int]:
    """EM for one proband. ``dosages`` (L,) in {0,1,2}; ``afs`` (L,K).

    Returns (weights (K,), log-likelihood, iterations).
    """
    f = np.clip(np.asarray(afs, dtype=float), AF_CLAMP, 1.0 - AF_CLAMP)
    g = np.asarray(dosages, dtype=float)
    n_snps, k = f.shape
    if n_snps == 0:
        raise ValueError("no SNPs to estimate ancestry from")
    w = np.full(k, 1.0 / k)
    alt = g
    refc = 2.0 - g
    previous = -np.inf
    log_likelihood = previous
    iteration = 0
    for iteration in range(1, EM_MAX_ITERATIONS + 1):
        p = f @ w
        alt_share = (f * w) / p[:, None]
        ref_share = ((1.0 - f) * w) / (1.0 - p)[:, None]
        w = (alt[:, None] * alt_share + refc[:, None] * ref_share).sum(axis=0) / (2.0 * n_snps)
        w = np.clip(w, 0.0, None)
        w = w / w.sum()
        p = f @ w
        log_likelihood = float(np.sum(alt * np.log(p) + refc * np.log(1.0 - p)))
        if abs(log_likelihood - previous) < EM_TOLERANCE:
            break
        previous = log_likelihood
    return w, log_likelihood, iteration


def estimate_ancestry(
    probands: Sequence[str],
    dosages: Sequence[Sequence[int]],
    afs: Sequence[Sequence[float]],
    *,
    option: str = AUTO,
    sheet: Optional[Mapping[str, str]] = None,
) -> list[AncestryEstimate]:
    """One estimate per proband.

    ``option`` is --ancestry: 'auto' estimates, a population name gives every
    proband that population. A non-empty ``sheet`` value (the 'ancestry'
    column) overrides the option for its sample; 'auto' there asks for the
    estimate. Values are checked by ``check_ancestry_value`` before this runs.
    """
    sheet = dict(sheet or {})
    matrix = np.asarray(dosages, dtype=float).reshape(len(dosages), len(probands)) if len(dosages) else np.zeros((0, len(probands)))
    freq = np.asarray(afs, dtype=float).reshape(len(afs), len(ALL_POPULATIONS)) if len(afs) else np.zeros((0, len(ALL_POPULATIONS)))
    out: list[AncestryEstimate] = []
    for column, sample in enumerate(probands):
        wanted = (sheet.get(sample) or "").strip().lower() or option
        origin = "given (sheet)" if sheet.get(sample, "").strip() else "given (option)"
        observed = matrix[:, column] >= 0 if matrix.size else np.zeros(0, dtype=bool)
        n_snps = int(observed.sum())
        if wanted != AUTO:
            weights = {pop: (1.0 if pop == wanted else 0.0) for pop in ALL_POPULATIONS}
            out.append(AncestryEstimate(sample, origin, weights, wanted, 1.0, n_snps, None))
            continue
        if n_snps < MIN_SNPS:
            out.append(AncestryEstimate(
                sample, "overall", None, OVERALL, None, n_snps, None,
                note=f"{n_snps} common coding SNPs called (fewer than {MIN_SNPS}); gnomAD's overall frequencies used",
            ))
            continue
        w, log_likelihood, _iterations = fit_mixture(matrix[observed, column], freq[observed])
        weights = {pop: float(value) for pop, value in zip(ALL_POPULATIONS, w)}
        best = max(weights, key=lambda pop: weights[pop])
        out.append(AncestryEstimate(sample, "estimated", weights, best, float(weights[best]), n_snps, log_likelihood))
    return out


def check_ancestry_value(value: str) -> Optional[str]:
    """None when ``value`` is 'auto' or a gnomAD population; otherwise why not."""
    text = (value or "").strip().lower()
    if text in (AUTO,) + ALL_POPULATIONS:
        return None
    return f"{value!r} is not 'auto' or one of the gnomAD populations {', '.join(ALL_POPULATIONS)}"


# ------------------------------------------------------------------- sex

MIN_SEX_SNPS = 50
SEX_F_MALE = 0.8
SEX_F_FEMALE = 0.2


@dataclass(frozen=True)
class SexCall:
    sample_id: str
    sex: str                      # male | female | unknown
    source: str                   # sheet | inferred | unknown
    n_snps: int                   # chrX non-PAR SNPs called in the proband
    f: Optional[float]            # 1 - observed / expected heterozygosity on them
    note: str = ""


def infer_sex(dosages: Sequence[int], afs: Sequence[float]) -> tuple[str, Optional[float], int]:
    """(male | female | unknown, F, SNPs used) from chrX non-PAR genotypes. See the module docstring."""
    g = np.asarray(dosages, dtype=float).reshape(-1)
    p = np.asarray(afs, dtype=float).reshape(-1)
    called = g >= 0
    n = int(called.sum())
    if n < MIN_SEX_SNPS:
        return "unknown", None, n
    expected = float(np.sum(2.0 * p[called] * (1.0 - p[called])))
    if expected <= 0:
        return "unknown", None, n
    observed = float(np.sum(g[called] == 1))
    f = 1.0 - observed / expected
    if f >= SEX_F_MALE:
        return "male", f, n
    if f <= SEX_F_FEMALE:
        return "female", f, n
    return "unknown", f, n


def assign_sexes(probands: Sequence[str], sheet: Mapping[str, str], dosages: Sequence[Sequence[int]],
                 afs: Sequence[float]) -> list[SexCall]:
    """One SexCall per proband: the sheet's male/female, else inferred from chrX (or unknown)."""
    matrix = (np.asarray(dosages, dtype=float).reshape(len(dosages), len(probands))
              if len(dosages) else np.zeros((0, len(probands))))
    out = []
    for column, sample in enumerate(probands):
        inferred, f, n = infer_sex(matrix[:, column], afs) if matrix.size else ("unknown", None, 0)
        given = (sheet or {}).get(sample, "unknown")
        if given in ("male", "female"):
            note = "" if inferred in ("unknown", given) else f"chrX heterozygosity suggests {inferred} (F={f:.2f})"
            out.append(SexCall(sample, given, "sheet", n, f, note))
        elif inferred != "unknown":
            out.append(SexCall(sample, inferred, "inferred", n, f))
        else:
            why = (f"{n} chrX SNPs called (fewer than {MIN_SEX_SNPS})" if f is None
                   else f"F={f:.2f} is between {SEX_F_FEMALE} and {SEX_F_MALE}")
            out.append(SexCall(sample, "unknown", "unknown", n, f,
                               f"sex not given and not inferable ({why}); left out of chrX/chrY non-PAR genes"))
    return out


ANCESTRY_COLUMNS = ("sample_id", "method", "assigned", "confidence", "n_snps", "log_likelihood",
                    *(f"w_{pop}" for pop in ALL_POPULATIONS), "sex", "sex_source", "chrx_snps", "chrx_het_f",
                    "note")


def ancestry_rows(estimates: Sequence[AncestryEstimate],
                  sexes: Optional[Sequence[SexCall]] = None) -> list[dict[str, str]]:
    by_sample = {call.sample_id: call for call in (sexes or ())}
    rows = []
    for est in estimates:
        row = {
            "sample_id": est.sample_id, "method": est.method, "assigned": est.assigned,
            "confidence": "." if est.confidence is None else f"{est.confidence:.4f}",
            "n_snps": str(est.n_snps),
            "log_likelihood": "." if est.log_likelihood is None else f"{est.log_likelihood:.3f}",
            "note": est.note or ".",
        }
        for pop in ALL_POPULATIONS:
            row[f"w_{pop}"] = "." if est.weights is None else f"{est.weights[pop]:.4f}"
        call = by_sample.get(est.sample_id)
        row.update({
            "sex": call.sex if call else ".", "sex_source": call.source if call else ".",
            "chrx_snps": str(call.n_snps) if call else ".",
            "chrx_het_f": "." if call is None or call.f is None else f"{call.f:.3f}",
        })
        if call and call.note:
            row["note"] = call.note if row["note"] == "." else f"{row['note']}; {call.note}"
        rows.append(row)
    return rows
