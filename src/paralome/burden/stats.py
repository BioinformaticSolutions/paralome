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

"""The burden test: observed carrier probands against gnomAD's expectation.

For gene g, tier t and model m:

q
    the sum of gnomAD AF over the gene's qualifying alleles that pass the
    model's frequency gate -- one q per gnomAD population, from the reference
    table built at genome setup. A proband's own q is those mixed by its
    ancestry weights, sum_k w_k q_k (gnomAD's overall q when it has none).
    Below ``Q_FLOOR`` it is raised to ``Q_FLOOR``: a gene with no qualifying
    allele in gnomAD is not certain to have none, and with q = 0 a single
    carrier would have p = 0 exactly. The floor is one allele in gnomAD v4.1's
    exome allele number (2 x 730,947 individuals) -- "not seen" at gnomAD's
    resolution. Stated in the summary.

carrier probability
    dominant 1 - (1-q)^2 (at least one qualifying allele on two copies).

    recessive: the cohort counts a proband who is homozygous for a qualifying
    allele OR carries two or more qualifying heterozygous variants in the
    gene, in any phase (cis pairs included: phase is unknown, so they cannot
    be told apart). Under Hardy-Weinberg each of the two haplotypes carries
    qualifying allele i independently with probability f_i, so the number N
    of qualifying alleles a proband carries is a sum of 2m independent
    Bernoulli(f_i) (every allele twice), and the probability that matches
    the observed definition is P(N >= 2) -- not q^2, which counts only the
    in-trans pairs and the homozygotes and is about HALF of it (for many
    rare alleles P(N >= 2) ~ 2q^2 - s2). Exactly,

        P(N = 0) = prod (1-f_i)^2 = exp(2 sum log(1-f_i))
        P(N = 1) = P(N = 0) x 2 sum f_i / (1-f_i)

    and with the first two power sums q = sum f_i and s2 = sum f_i^2 (the
    reference table carries both, per population and per pair of
    populations) the expansion to second order is

        P(N >= 2) = 1 - exp(-(2q + s2)) (1 + 2q + 2 s2)

    It is exact for every term in q and s2; what it leaves out are terms in
    s3 = sum f_i^3 and higher. Under the recessive gate (every f < 0.01)
    leaving them out makes the expectation larger (the test conservative;
    not so for large f, e.g. two alleles at 0.5): for one allele of frequency
    f it gives f^2 + (4/3) f^3 against the exact f^2, i.e. at most ~1.3%
    high at the recessive gate's AF 0.01 (checked against brute force and
    Monte Carlo). A reference built
    before the s2 columns existed has only q: then the Poisson form
    1 - exp(-2q)(1 + 2q) is used -- s2 = 0, which is again never smaller than
    the truth -- and the summary says so.

    An admixed proband's allele frequencies are the mixtures sum_k w_k f_ik,
    so its q is sum_k w_k q_k (linear) and its s2 is
    sum_k sum_l w_k w_l S_kl with S_kl = sum_i f_ik f_il -- why the reference
    keeps the cross sums s2_<a>_<b> as well as s2_<pop>.

    Hemizygous copies: a male outside the pseudo-autosomal regions
    of X and Y carries one copy, so under BOTH models he counts when he carries
    one qualifying allele and his carrier probability is q (capped at 1). A
    female on chrX is diploid and is scored as above; on chrY she is not in
    the test at all (not in the denominator). A proband whose sex is neither
    given nor inferable is left out of the chrX/chrY non-PAR genes.

p
    the exact one-sided upper tail P(X >= observed) of the Poisson-binomial
    distribution of the number of carriers, X = sum of independent
    Bernoulli(P_i), one per proband with its own probability.

q-value
    Benjamini-Hochberg over the genes tested within one (model, tier).

log2 O/E
    log2((observed + 0.5) / (expected + 0.5)), expected = sum_i P_i. The 0.5
    keeps genes with nothing observed or nothing expected finite.

genomic inflation (lambda)
    median of the chi-square (1 df) quantiles of p, over the genes whose
    expected count is at least ``LAMBDA_MIN_EXPECTED``, divided by 0.4549.
    With 40 probands nearly every gene expects far less than one carrier, its
    p is 1 or a few discrete values, and lambda computed over all genes would
    sit near 0 and mean nothing. Even over the genes kept, an exact test of a
    discrete count is conservative and lambda is biased DOWNWARDS; it is a
    sanity check, not a correction factor, and is not applied to anything.
    Fewer than ``LAMBDA_MIN_GENES`` such genes: not estimated.

THE REGULATORY TIER IS AN ALLELE-COUNT TEST
    Summed over every rare non-coding allele in a gene and its 10 kb flanks,
    q is often above 1, so "the probability a proband carries one" (1-(1-q)^2)
    is 1 for most genes and the carrier test above says nothing. The coding
    tiers keep it; the regulatory tier counts alleles instead:

    observed
        the qualifying alleles the probands carry in the gene: a
        heterozygote counts 1, a homozygote 2, summed over probands and
        variants. Qualifying as for the coding tiers (qualify.py's regulatory
        classes), gated on gnomAD GENOME AF_grpmax by the model's threshold;
        "dominant" and "recessive" differ only in that gate (< 0.001,
        < 0.01) -- both count alleles, neither counts carrier status.
    expected
        sum over probands of 2 x q_i, q_i the proband's ancestry-weighted sum
        of gnomAD genome AF over the gene's qualifying alleles (2Nq for one
        ancestry mix). 1 x q_i for a male outside the PARs (whose
        1/1 call is one allele), and nothing for a proband not in the test
        (a female on chrY, unknown sex on chrX/chrY) -- copies_in_test. q is not capped at 1: it is an expected number of
        alleles per haplotype, not a probability. Below ``Q_FLOOR_GENOME`` --
        one allele in gnomAD v4.1's genome allele number (2 x 76,215) -- it is
        raised to it, as Q_FLOOR is for the exome tiers.
    p
        Poisson upper tail P(X >= observed), X ~ Poisson(expected). Alleles
        are treated as independent draws; linkage between nearby rare
        alleles is ignored, which can only make p too small, so a regulatory
        hit is a lead to check, not a finding.
    log2 O/E and q-values
        as above: log2((observed + 0.5) / (expected + 0.5)); BH within the
        (model, regulatory) comparison.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, NamedTuple, Optional, Sequence

GNOMAD41_EXOME_INDIVIDUALS = 730_947
Q_FLOOR = 1.0 / (2 * GNOMAD41_EXOME_INDIVIDUALS)
#: GnomAD v4.1 genomes, for the regulatory tier's floor.
GNOMAD41_GENOME_INDIVIDUALS = 76_215
Q_FLOOR_GENOME = 1.0 / (2 * GNOMAD41_GENOME_INDIVIDUALS)
PSEUDOCOUNT = 0.5
LAMBDA_MIN_EXPECTED = 0.5
LAMBDA_MIN_GENES = 20
CHI2_1DF_MEDIAN = 0.454936423119572


def recessive_probability(q: float, s2: Optional[float] = None) -> float:
    """P(a diploid proband carries >= 2 qualifying alleles), from q and s2. See the module docstring.

    ``s2`` None: the Poisson form 1 - exp(-2q)(1 + 2q). Computed as
    P(Poisson(x) >= 2) - s2 exp(-x), x = 2q + s2, so a tiny probability keeps
    its precision (the direct form subtracts two numbers within 1e-12 of 1).
    """
    from scipy.special import gammainc

    q = max(float(q), 0.0)
    s2 = 0.0 if s2 is None else min(max(float(s2), 0.0), q * q)
    x = 2.0 * q + s2
    value = float(gammainc(2.0, x)) - s2 * math.exp(-x)
    return min(max(value, 0.0), 1.0)


def carrier_probability(model: str, q: float, s2: Optional[float] = None, *, hemizygous: bool = False) -> float:
    """The probability one proband counts as a carrier. See the module docstring.

    ``hemizygous``: one copy (a male outside the PARs): q under both models.
    """
    if model not in ("dominant", "recessive"):
        raise ValueError(f"unknown burden model {model!r}")
    q = max(float(q), 0.0)
    if hemizygous:
        return min(q, 1.0)
    if model == "dominant":
        q = min(q, 1.0)
        return 1.0 - (1.0 - q) ** 2
    return recessive_probability(q, s2)


def s2_key(first: str, second: str) -> str:
    """The reference column suffix for sum_i f_first,i f_second,i: 'nfe' or 'afr_nfe' (population order)."""
    if first == second:
        return first
    from paralome.burden.qualify import ALL_POPULATIONS

    a, b = sorted((first, second), key=ALL_POPULATIONS.index)
    return f"{a}_{b}"


def mixed_s2(s2_by_key: Optional[Mapping[str, float]], weights: Optional[Mapping[str, float]]) -> Optional[float]:
    """sum_k sum_l w_k w_l S_kl, or gnomAD's overall s2 ('all') with no weights; None when not known."""
    if s2_by_key is None:
        return None
    if weights is None:
        value = s2_by_key.get("all")
        return None if value is None else float(value)
    total = 0.0
    pops = [pop for pop, w in weights.items() if w]
    for a in pops:
        for b in pops:
            total += float(weights[a]) * float(weights[b]) * float(s2_by_key.get(s2_key(a, b)) or 0.0)
    return total


def mixed_q(q_by_population: Mapping[str, float], weights: Optional[Mapping[str, float]],
            floor: float = Q_FLOOR) -> float:
    """sum_k w_k q_k; gnomAD's overall q ('all') when there are no weights; never below ``floor``."""
    if weights is None:
        value = float(q_by_population.get("all") or 0.0)
    else:
        value = sum(float(w) * float(q_by_population.get(pop) or 0.0) for pop, w in weights.items())
    return max(value, floor)


def mixed_q_and_s2(q_by_population: Mapping[str, float], s2_by_key: Optional[Mapping[str, float]],
                   weights: Optional[Mapping[str, float]], floor: float = Q_FLOOR) -> tuple[float, Optional[float]]:
    """(q, s2) for one proband. Where q is raised to the floor, s2 is the floor's own square:
    the floor stands for one allele not seen at gnomAD's resolution."""
    raw = mixed_q(q_by_population, weights, 0.0)
    s2 = mixed_s2(s2_by_key, weights)
    if raw < floor:
        return floor, (None if s2_by_key is None else floor * floor)
    if s2 is not None:
        s2 = min(max(s2, 0.0), raw * raw)
    return raw, s2


def poisson_binomial_pmf(probabilities: Sequence[float]) -> list[float]:
    """P(X = k), k = 0..n, for X a sum of independent Bernoulli(p_i). Exact DP, O(n^2)."""
    dist = [1.0]
    for p in probabilities:
        p = min(max(float(p), 0.0), 1.0)
        nxt = [0.0] * (len(dist) + 1)
        for k, mass in enumerate(dist):
            nxt[k] += mass * (1.0 - p)
            nxt[k + 1] += mass * p
        dist = nxt
    return dist


def poisson_binomial_upper_tail(probabilities: Sequence[float], observed: int) -> float:
    """P(X >= observed). Summed from the upper terms, so a tiny tail keeps its precision."""
    if observed <= 0:
        return 1.0
    dist = poisson_binomial_pmf(probabilities)
    if observed >= len(dist):
        return 0.0
    return min(1.0, math.fsum(dist[observed:]))


def upper_tails(dist: Sequence[float]) -> list[float]:
    """[P(X >= k) for k = 0..n+1] from a pmf, each summed from its own upper terms."""
    tails = [0.0] * (len(dist) + 1)
    running: list[float] = []
    for k in range(len(dist) - 1, -1, -1):
        running.append(dist[k])
        tails[k] = min(1.0, math.fsum(running))
    tails[0] = 1.0
    return tails


#: The default --alpha: Evidome's ORA selects a gene when its raw p < the run's --alpha (evidome.enrich.Settings.alpha, the
#: crosswalk's --alpha default). The null selection probability written for Evidome is computed at the same
#: value; a cross-repo test checks the two agree.
EVIDOME_ORA_ALPHA = 0.05


def null_selection_probability(probabilities: Sequence[float], alpha: float = EVIDOME_ORA_ALPHA,
                               dist: Optional[Sequence[float]] = None) -> float:
    """P(the gene's p < alpha) when its carriers follow the null (Poisson-binomial of ``probabilities``).

    The p of an observed count k is P(X >= k), which falls as k rises; the gene
    is selected when k reaches k* = the smallest k with P(X >= k) < alpha, so
    the probability is exactly P(X >= k*) (0 when no count can reach alpha).
    It differs from gene to gene by orders of magnitude -- a gene with many
    common qualifying alleles can reach p < alpha, one at the floor almost never
    -- which is the detectability Evidome's weighted ORA null corrects for.
    """
    dist = poisson_binomial_pmf(probabilities) if dist is None else dist
    for tail in upper_tails(dist):
        if tail < alpha:
            return tail
    return 0.0


def log2_observed_over_expected(observed: float, expected: float) -> float:
    return math.log2((observed + PSEUDOCOUNT) / (expected + PSEUDOCOUNT))


def genomic_inflation(pvalues: Iterable[float], expected: Iterable[float]) -> tuple[Optional[float], int]:
    """(lambda or None, genes used). See the module docstring for which genes and why."""
    from scipy.stats import chi2

    kept = [p for p, e in zip(pvalues, expected) if e >= LAMBDA_MIN_EXPECTED]
    if len(kept) < LAMBDA_MIN_GENES:
        return None, len(kept)
    statistics = sorted(float(chi2.isf(min(max(p, 1e-300), 1.0), 1)) for p in kept)
    middle = len(statistics) // 2
    median = statistics[middle] if len(statistics) % 2 else 0.5 * (statistics[middle - 1] + statistics[middle])
    return median / CHI2_1DF_MEDIAN, len(kept)


class GeneScore(NamedTuple):
    observed: int
    expected: float
    log2_obs_exp: float
    p: float
    n_tested: int
    null_selection: float


#: gene_region() values: where a male has one copy.
HEMIZYGOUS_X = "X"
HEMIZYGOUS_Y = "Y"


def genomic_control(pvalues: Iterable[float], inflation: Optional[float]) -> tuple[list[float], bool]:
    """(p corrected by lambda, applied?). Applied only when lambda > 1: chi2_1(isf(p)) / lambda -> sf."""
    from scipy.stats import chi2

    if inflation is None or not inflation > 1.0:
        return [float(p) for p in pvalues], False
    return [float(chi2.sf(chi2.isf(min(max(float(p), 1e-300), 1.0), 1) / inflation, 1)) for p in pvalues], True


def score_genes(
    model: str,
    universe: Mapping[str, Mapping[str, float]],
    carriers: Mapping[str, Sequence[str]],
    weights: Sequence[Optional[Mapping[str, float]]],
    *,
    second: Optional[Mapping[str, Optional[Mapping[str, float]]]] = None,
    sexes: Optional[Sequence[str]] = None,
    regions: Optional[Mapping[str, Optional[str]]] = None,
    alpha: float = EVIDOME_ORA_ALPHA,
) -> dict[str, GeneScore]:
    """gene -> GeneScore for one (model, tier).

    ``universe``: gene -> {population or 'all' -> q} (zeros for a gene seen
    only in the cohort). ``weights``: one mixture per proband, in proband order.
    ``second``: gene -> {s2 key -> value} (stats.s2_key), None for a reference
    without s2 (the Poisson form is used). ``sexes``: 'male' / 'female' /
    'unknown' per proband, in proband order; ``regions``: gene -> 'X' or 'Y'
    for a gene outside the pseudo-autosomal regions (gene_region), else None.
    Without ``sexes`` every proband is diploid everywhere.
    ``carriers``: gene -> the probands who count (cohort.CohortScan.carriers
    applies the same sex rules); only probands in the test are counted.
    """
    out: dict[str, GeneScore] = {}
    sex_aware = sexes is not None and regions is not None
    sexes = list(sexes) if sexes is not None else ["female"] * len(weights)
    for gene, q_by_population in universe.items():
        region = (regions or {}).get(gene)
        # A gene the reference has no row for (seen only in the cohort) has s2 = 0 there: its q is the floor,
        # and mixed_q_and_s2 gives the floor's s2.
        s2_by_key = (second.get(gene) or {}) if second is not None else None
        probabilities = []
        in_test = []
        for index, (w, sex) in enumerate(zip(weights, sexes)):
            if not sex_aware:
                region = None
            elif region is not None:
                if sex not in ("male", "female") or (region == HEMIZYGOUS_Y and sex == "female"):
                    continue
            hemizygous = region is not None and sex == "male"
            q, s2 = mixed_q_and_s2(q_by_population, s2_by_key, w)
            probabilities.append(carrier_probability(model, q, s2, hemizygous=hemizygous))
            in_test.append(index)
        observed = len(carriers.get(gene, ()))
        expected = math.fsum(probabilities)
        dist = poisson_binomial_pmf(probabilities)
        p = 1.0 if observed <= 0 else (0.0 if observed >= len(dist) else min(1.0, math.fsum(dist[observed:])))
        out[gene] = GeneScore(observed, expected, log2_observed_over_expected(observed, expected), p,
                              len(in_test), null_selection_probability(probabilities, alpha, dist))
    return out


# ----------------------------------------------------------- sex and copies

#: GRCh38 pseudo-autosomal regions, 1-based inclusive (Genome Reference Consortium). Outside them a male has
#: one X and one Y.
PAR_GRCH38 = {
    "X": ((10_001, 2_781_479), (155_701_383, 156_030_895)),
    "Y": ((10_001, 2_781_479), (56_887_903, 57_217_415)),
}
#: The builds whose PAR coordinates are known here (ANNOVAR buildver or assembly names).
PAR_BUILDS = {"hg38": PAR_GRCH38, "grch38": PAR_GRCH38}


def sex_chromosome(chrom: str) -> Optional[str]:
    name = str(chrom or "")
    if name.lower().startswith("chr"):
        name = name[3:]
    name = name.upper()
    return name if name in ("X", "Y") else None


def gene_region(chrom: str, start: Optional[int], end: Optional[int],
                par: Mapping[str, Sequence[tuple[int, int]]] = PAR_GRCH38) -> Optional[str]:
    """'X' or 'Y' for a gene where a male has one copy; None for autosomes and the PARs.

    A gene that overlaps a PAR at all is treated as pseudo-autosomal (diploid in
    both sexes); a gene on X or Y without coordinates is outside them.
    """
    which = sex_chromosome(chrom)
    if which is None:
        return None
    if start is not None and end is not None:
        low, high = min(int(start), int(end)), max(int(start), int(end))
        for par_start, par_end in par.get(which, ()):
            if low <= par_end and high >= par_start:
                return None
    return which


# ------------------------------------------------- the regulatory tier

def poisson_upper_tail(mean: float, observed: int) -> float:
    """P(X >= observed) for X ~ Poisson(mean)."""
    from scipy.stats import poisson

    if observed <= 0:
        return 1.0
    return min(1.0, float(poisson.sf(int(observed) - 1, max(float(mean), 0.0))))


def allele_count(hits_by_proband: Mapping[str, Sequence], caps: Optional[Mapping[str, int]] = None) -> int:
    """Qualifying alleles carried: per proband, each variant once, het 1 and hom 2.

    ``caps``: proband -> the most copies one variant can add (1 for a
    male outside the PARs, whose 1/1 call is one copy); 2 when not given.
    """
    total = 0
    for proband, hits in hits_by_proband.items():
        cap = 2 if caps is None else int(caps.get(proband, 2))
        copies: dict = {}
        for hit in hits:
            copies[hit.variant] = max(copies.get(hit.variant, 0), min(int(hit.copies), cap))
        total += sum(copies.values())
    return total


def copies_in_test(sex: str, region: Optional[str]) -> int:
    """Copies a proband has of a gene: 2 (diploid), 1 (a male outside the PARs), 0 (not in the test:
    a female on Y, or unknown sex on X or Y)."""
    if region is None:
        return 2
    if sex == "male":
        return 1
    return 2 if (sex == "female" and region == HEMIZYGOUS_X) else 0


def score_allele_counts(
    universe: Mapping[str, Mapping[str, float]],
    observed_alleles: Mapping[str, int],
    weights: Sequence[Optional[Mapping[str, float]]],
    *,
    floor: float = Q_FLOOR_GENOME,
    sexes: Optional[Sequence[str]] = None,
    regions: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, tuple[int, float, float, float]]:
    """gene -> (observed alleles, expected alleles, log2 O/E, p): the regulatory tier's test.

    expected = sum over probands of c_i * q_i (q_i mixed by that proband's
    ancestry weights, floored, never capped; c_i its copies of the gene: 2, or
    1 for a male outside the PARs, 0 for a proband not in the test -- see
    copies_in_test); p = Poisson P(X >= observed). See the module docstring.
    """
    out = {}
    sexes = list(sexes) if sexes is not None else None
    for gene, q_by_population in universe.items():
        region = (regions or {}).get(gene) if sexes is not None else None
        terms = []
        for index, w in enumerate(weights):
            copies = 2 if sexes is None else copies_in_test(sexes[index], region)
            if copies:
                terms.append(copies * mixed_q(q_by_population, w, floor))
        expected = math.fsum(terms)
        observed = int(observed_alleles.get(gene, 0))
        p = poisson_upper_tail(expected, observed)
        out[gene] = (observed, expected, log2_observed_over_expected(observed, expected), p)
    return out
