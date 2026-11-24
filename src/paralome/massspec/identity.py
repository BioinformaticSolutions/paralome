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

"""protein_identity: each protein (or phosphopeptide's protein) to an Ensembl gene.

 Every identifier the row carries is used, through the genome's
own GTF and the core index (``gene_identity.build_gtf_index``), in this order:

    1. an Ensembl gene ID anywhere in the row's identifiers
    2. the gene symbol(s): the exporter's gene column, every member of the
       protein group, the ``GN=`` of the description
    3. the same symbols matched without regard to case (rat ``Gbe1`` against a
       GTF that writes ``GBE1``), only when that is unambiguous
    4. the UniProt entry name's gene part (``GBE1_RAT`` -> ``GBE1``)

Every assignment records which evidence decided it. A protein group whose
members point at different genes is a CONFLICT: it gets no gene, and is
listed, never resolved by guessing. A symbol the GTF gives to two genes is
ambiguous and likewise gets none. UniProt accessions are carried into the
results for every row; mapping an accession with no usable symbol needs
Ensembl's UniProt cross-reference for the genome's release, which this does
not yet read.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence, Tuple

from paralome.report_schema import MISSING_VALUE

_ENSEMBL_ANY = re.compile(r"\bENS[A-Z]*G\d{6,}(?:\.\d+)?\b")


def _symbols(feature: Mapping[str, str]) -> List[str]:
    found: List[str] = []
    for key in ("gene_symbol", "gene_symbols"):
        for part in re.split(r"[;,]", feature.get(key, "") or ""):
            part = part.strip()
            if part and part not in found:
                found.append(part)
    for match in re.findall(r"\bGN=(\S+)", feature.get("description", "") or ""):
        if match not in found:
            found.append(match)
    return found


def _entry_genes(feature: Mapping[str, str]) -> List[str]:
    out = []
    for part in re.split(r"[;,]", feature.get("entry_name", "") or ""):
        part = part.strip()
        if "_" in part:
            out.append(part.rsplit("_", 1)[0])
    return out


def _accessions(feature: Mapping[str, str]) -> List[str]:
    found: List[str] = []
    for key in ("accession", "accessions"):
        for part in re.split(r"[;,]", feature.get(key, "") or ""):
            part = part.strip().split("-")[0]
            if part and part not in found:
                found.append(part)
    return found


def resolve_proteins(features: Sequence[Mapping[str, str]], gtf,
                     uniprot_table=None) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """``(per feature {gene_id, gene_name, identity_evidence, identity_note}, report)``.

    ``uniprot_table`` (the genome's Ensembl UniProt table, ``gene_identity.read_uniprot_table``) adds a
    last step, the feature's UniProt accession, for a name that is no gene symbol (Olink's Abeta40: P05067, APP).
    An accession on several genes takes the primary assembly's when the others are on alternate haplotypes."""
    from paralome.gene_identity import gtf_gene_index, prefer_the_primary_assembly, read_uniprot_table, strip_version

    by_symbol, by_id, contig_of = gtf_gene_index(gtf)
    by_accession = read_uniprot_table(uniprot_table) if uniprot_table else {}

    def accession_gene(accession: str):
        genes = [g for g in by_accession.get(accession, []) if g in by_id]
        if not genes:
            return None
        chosen = prefer_the_primary_assembly(genes, contig_of)
        return chosen[0] if len(chosen) == 1 else ""

    folded: Dict[str, str] = {}
    for symbol, gene in by_symbol.items():
        key = symbol.lower()
        if key in folded and folded[key] != gene:
            folded[key] = ""
        else:
            folded.setdefault(key, gene)
    rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    conflicts: List[str] = []
    for feature in features:
        text = " ".join(str(v) for v in feature.values())
        ensembl = [strip_version(m) for m in _ENSEMBL_ANY.findall(text)]
        ensembl = [g for g in dict.fromkeys(ensembl) if g in by_id]
        evidence, genes = "", []
        if ensembl:
            evidence, genes = "Ensembl gene ID in the export", ensembl
        else:
            symbols = _symbols(feature)
            for label, lookup, names in (
                ("gene symbol", lambda s: by_symbol.get(s), symbols),
                ("gene symbol, case-insensitive", lambda s: folded.get(s.lower()), symbols),
                ("UniProt entry name", lambda s: folded.get(s.lower()), _entry_genes(feature)),
                ("UniProt accession, Ensembl's table", accession_gene, _accessions(feature) if by_accession else []),
            ):
                if evidence and not label.startswith("UniProt accession"):
                    continue          # ambiguous by name: only the accession can still settle it
                hits = [lookup(s) for s in names]
                if any(h == "" for h in hits) and not any(hits):
                    evidence = evidence or f"{label}: ambiguous in the GTF"
                    continue
                hits = [h for h in dict.fromkeys(hits) if h]
                if hits:
                    evidence, genes = label, hits
                    break
        if len(genes) == 1:
            gene = genes[0]
            rows.append({"gene_id": gene, "gene_name": by_id.get(gene, "") or gene,
                         "identity_evidence": evidence, "identity_note": ""})
            counts[evidence] = counts.get(evidence, 0) + 1
        elif len(genes) > 1:
            note = "protein group spans genes " + ", ".join(f"{by_id.get(g) or g} ({g})" for g in genes)
            conflicts.append(f"{feature.get('accession', '')}: {note}")
            rows.append({"gene_id": MISSING_VALUE, "gene_name": _symbols(feature)[0] if _symbols(feature) else "",
                         "identity_evidence": "conflict", "identity_note": note})
            counts["conflict"] = counts.get("conflict", 0) + 1
        else:
            reason = evidence or "no identifier the GTF knows"
            rows.append({"gene_id": MISSING_VALUE, "gene_name": _symbols(feature)[0] if _symbols(feature) else "",
                         "identity_evidence": "none", "identity_note": reason})
            counts["none"] = counts.get("none", 0) + 1
    total = len(rows)
    mapped = sum(1 for row in rows if row["gene_id"] != MISSING_VALUE)
    report = {
        "gtf": str(gtf), "uniprot_table": str(uniprot_table) if uniprot_table else None, "total": total,
        "mapped": mapped,
        "fraction": round(mapped / total, 6) if total else 0.0,
        "by_evidence": counts, "conflict_examples": conflicts[:25],
    }
    return rows, report
