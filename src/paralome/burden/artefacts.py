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

"""Alleles too common in the cohort for their gnomAD frequency.

A caller over-calls some alleles in nearly every sample -- homopolymer indels,
reads of a pseudogene mapped onto its gene -- and gnomAD, called another way,
has them rare or not at all. Counted as they stand they are the top of every
burden table (GK in 39 of 40 probands against ~0 expected). The test
itself has to deal with them, so before any gene is scored each qualifying
allele is checked on its own:

    the probands carrying it, k of the n probands, against gnomAD's
    prediction: each proband carries it with probability c = 1-(1-f)^2, f its
    AF_grpmax (gnomAD's genome table for the regulatory tier, the exome table
    otherwise); an allele gnomAD does not have is given the gene floor (one
    allele in gnomAD's allele number, stats.Q_FLOOR / Q_FLOOR_GENOME). The
    count is binomial; the allele is seen at all (k >= 1), so the chance is
    taken among alleles seen: P(X >= k | X >= 1).

Below ``OVERREPRESENTED_P`` the allele is dropped from every gene and tier it
counted in -- from the carriers (observed) and, where gnomAD has it, from the
gene's expectation (its per-population AFs out of q, their products out of
s2) -- so both sides are counted on the same alleles. AF_grpmax is the highest
population's frequency, so an allele common in one population is not called
too common. At 1e-6 an allele gnomAD does not have is dropped from 3 carriers
in 40 probands (P ~ 5e-10) and kept at 2 (P ~ 3e-5); one at AF 3e-5 is kept at
2 carriers (P ~ 1e-3); GK A376T (1.9e-5, 39 of 40) is dropped.

Nothing is hidden: each gene table lists the dropped alleles it had
(``overrepresented_alleles``: variant(carriers)) and how many carriers rested
on them alone (``carriers_from_overrepresented_alleles``; with ``n_probands``,
the count before the drop), and the summary lists every dropped allele with its
carriers, frequency and P.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from paralome.burden import qualify
from paralome.burden import stats as burden_stats

#: Below this chance of its carrier count, an allele is too common for gnomAD and dropped.
OVERREPRESENTED_P_DEFAULT = 1e-6
OVERREPRESENTED_P = OVERREPRESENTED_P_DEFAULT


def source_of(tier: str) -> str:
    """The gnomAD table a tier's frequencies come from."""
    return "genome" if tier == qualify.REGULATORY_TIER else "exome"


def floor_of(source: str) -> float:
    return burden_stats.Q_FLOOR_GENOME if source == "genome" else burden_stats.Q_FLOOR


def carrier_tail(n: int, f: float, k: int) -> float:
    """P(X >= k | X >= 1) for X ~ Binomial(n, 1 - (1 - f)^2): the chance of k or more carriers among n,
    for an allele seen at all. Summed term by term in logs (no 1 - cdf: the tails are tiny)."""
    if n <= 0 or k <= 1:
        return 1.0
    c = min(max(1.0 - (1.0 - min(max(f, 0.0), 1.0)) ** 2, 1e-300), 1.0 - 1e-12)
    lc, l1 = math.log(c), math.log1p(-c)

    def pmf(j: int) -> float:
        return math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lc + (n - j) * l1)

    seen = sum(pmf(j) for j in range(1, n + 1))
    tail = sum(pmf(j) for j in range(min(k, n + 1), n + 1))
    return min(1.0, tail / seen) if seen > 0 else 1.0


@dataclass
class Allele:
    variant: str
    source: str
    probands: set = field(default_factory=set)
    grpmax: Optional[float] = None
    absent: bool = False
    afs: Optional[tuple] = None
    p: float = 1.0

    @property
    def frequency(self) -> float:
        return self.grpmax if self.grpmax and self.grpmax > 0 else floor_of(self.source)

    def summary(self) -> dict:
        return {"variant": self.variant, "gnomad": self.source, "carriers": len(self.probands),
                "af_grpmax": self.grpmax, "absent_from_gnomad": self.absent, "p": self.p}


def alleles_of(hits_tree: dict) -> dict[tuple[str, str], Allele]:
    """(source, variant) -> the allele, with every proband any tier or model counted it for."""
    out: dict[tuple[str, str], Allele] = {}
    for model, by_tier in hits_tree.items():
        for tier, by_gene in by_tier.items():
            source = source_of(tier)
            for by_proband in by_gene.values():
                for proband, hits in by_proband.items():
                    for hit in hits:
                        allele = out.get((source, hit.variant))
                        if allele is None:
                            allele = out[(source, hit.variant)] = Allele(hit.variant, source, set(), hit.grpmax,
                                                                         hit.absent, getattr(hit, "afs", None))
                        allele.probands.add(proband)
    return out


def overrepresented(hits_tree: dict, n_probands: int, *, threshold: Optional[float] = None
                    ) -> dict[tuple[str, str], Allele]:
    """The alleles whose carrier count is below ``threshold`` (default ``OVERREPRESENTED_P``, read when called)
    given gnomAD (module docstring)."""
    threshold = OVERREPRESENTED_P if threshold is None else threshold
    out = {}
    for key, allele in alleles_of(hits_tree).items():
        allele.p = carrier_tail(n_probands, allele.frequency, len(allele.probands))
        if allele.p < threshold:
            out[key] = allele
    return out


def without(hits_tree: dict, dropped: Iterable[tuple[str, str]]) -> dict:
    """A copy of the hits tree with the dropped alleles taken out (probands and genes left empty removed)."""
    gone = set(dropped)
    out: dict = {}
    for model, by_tier in hits_tree.items():
        for tier, by_gene in by_tier.items():
            source = source_of(tier)
            for gene, by_proband in by_gene.items():
                for proband, hits in by_proband.items():
                    kept = [h for h in hits if (source, h.variant) not in gone]
                    if kept:
                        out.setdefault(model, {}).setdefault(tier, {}).setdefault(gene, {})[proband] = kept
    return out


def expectation_less(row: dict, alleles: Iterable[Allele]) -> dict:
    """A reference row (q per population, s2 per key) with these alleles' gnomAD frequencies taken out.

    Only alleles gnomAD has (with per-population AFs) are in the reference sums; an absent allele
    added nothing to them. Sums are kept at or above zero."""
    present = [a for a in alleles if not a.absent and a.afs]
    if not present:
        return row
    q = dict(row["q"])
    for allele in present:
        for pop, af in zip(qualify.POPULATIONS_AND_ALL, allele.afs):
            if af:
                q[pop] = max(0.0, q.get(pop, 0.0) - af)
    s2 = row.get("s2")
    if s2 is not None:
        s2 = dict(s2)
        for allele in present:
            af = dict(zip(qualify.POPULATIONS_AND_ALL, allele.afs))
            for key in s2:
                a, b = (key, key) if "_" not in key else key.split("_", 1)
                fa, fb = af.get(a) or 0.0, af.get(b) or 0.0
                if fa and fb:
                    s2[key] = max(0.0, s2[key] - fa * fb)
    return {**row, "q": q, "s2": s2}
