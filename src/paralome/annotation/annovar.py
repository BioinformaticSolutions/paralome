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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re
import shlex


# INFO IDS THAT ARE NOT VCF IDS. table_annovar.pl -vcfinput turns
# every output column into an INFO key, and dbNSFP's columns include names VCF
# 4.2 does not allow as an ID (^[A-Za-z_][0-9A-Za-z_.]*$): GERP++_RS,
# M-CAP_score, fathmm-MKL_coding_score, Eigen-raw_coding, LIST-S2_score,
# H1-hESC_fitCons_score. htslib-based readers warn on or refuse them. After
# ANNOVAR writes its VCF, '-' becomes '_' and '+' becomes 'p' in the ##INFO
# lines and in the INFO column, so GERP++_RS reads GERPpp_RS and M-CAP_score
# reads M_CAP_score. No key Paralome reads contains either character. Only
# those two characters are rewritten: an ID that is invalid for another reason
# (a leading digit, as in 1000g2015aug_all) is left as ANNOVAR wrote it.
VALID_VCF_ID = re.compile(r"^[A-Za-z_][0-9A-Za-z_.]*$")


def renamed_info_id(key: str) -> str:
    """The INFO ID Paralome writes for ANNOVAR's ``key``."""
    return key.replace("-", "_").replace("+", "p")


# One pass over an uncompressed VCF. Header ##INFO IDs are renamed, and a
# renamed ID that lands on an ID already in the header stops the job rather
# than merging two different columns under one name. Each rename is written
# once to MAP as "old<TAB>new". POSIX awk only (mawk and gawk both run it).
_INFO_ID_RENAME_AWK = r'''
function fix(k,   n) { n = k; gsub(/-/, "_", n); gsub(/\+/, "p", n); return n }
BEGIN { FS = "\t"; OFS = "\t"; bad = 0 }
/^##INFO=<ID=/ {
    rest = substr($0, 12)
    i = index(rest, ",")
    if (i == 0) { i = index(rest, ">") }
    id = (i > 0) ? substr(rest, 1, i - 1) : rest
    nid = fix(id)
    if ((nid in source) && source[nid] != id) {
        printf "ERROR: INFO IDs %s and %s would both be named %s once '-' and '+' are replaced; nothing was renamed\n", source[nid], id, nid > "/dev/stderr"
        bad = 1
        exit 2
    }
    source[nid] = id
    if (nid != id) {
        print id "\t" nid > MAP
        $0 = "##INFO=<ID=" nid substr(rest, length(id) + 1)
    }
    print
    next
}
/^#/ { print; next }
{
    if ($8 ~ /[-+]/) {
        n = split($8, kv, ";")
        out = ""
        for (x = 1; x <= n; x++) {
            item = kv[x]
            j = index(item, "=")
            key = (j > 0) ? substr(item, 1, j - 1) : item
            if (key ~ /[-+]/) {
                item = fix(key) ((j > 0) ? substr(item, j) : "")
            }
            out = (x == 1) ? item : out ";" item
        }
        $8 = out
    }
    print
}
END { if (bad) exit 2 }
'''


def info_id_rename_bash(vcf: str, map_file: str) -> str:
    """Bash that renames invalid INFO IDs in the uncompressed VCF at ``vcf``.

    ``vcf`` and ``map_file`` are bash words, already quoted, so a caller can
    pass a variable such as ``"$ANNOVAR_VCF"``. The VCF is rewritten through a
    temporary file beside it and replaced only when awk succeeded; the map file
    holds one ``old<TAB>new`` line per renamed ID (empty when none was).
    """
    program = shlex.quote(_INFO_ID_RENAME_AWK)
    return "\n".join([
        "# Rename INFO IDs that are not valid VCF IDs ('-' -> '_', '+' -> 'p').",
        f": > {map_file}",
        f"awk -v MAP={map_file} {program} {vcf} > {vcf}.ids_renamed",
        f"mv -f {vcf}.ids_renamed {vcf}",
    ])


# SPLIT BEFORE ANNOVAR. FreeBayes (haplotype length 3) writes a
# change it sees inside a short haplotype with the bases around it, e.g.
# 10:100517905 CCT>CCC, which is T>C at 100517907. ANNOVAR looks gnomAD up by
# the exact position, REF and ALT, finds nothing for CCT>CCC, and would leave
# the allele with no frequency ("absent from gnomAD", so rare), which floods
# the gene-burden tables. Every ANNOVAR input is put in gnomAD's own form first
# (normalize_for_annotation_bash):
#
#     bcftools norm -m -any [-f <reference> -c w]      every record
#     bcftools norm --atomize                          MNPs only
#
# -m -any splits multi-allelic records into one record per ALT; -f left-aligns
# indels against the genome's FASTA (-c w: a REF that does not match is warned
# about, not fatal). Only MNPs (REF and ALT the same length, longer than one
# base) are then atomized into their SNVs; complex alleles (an SNV and an indel
# sharing a base) reach ANNOVAR whole, because atomizing them gives every
# carrier the '*' overlap allele. The "*" (overlapping deletion)
# records are dropped: there is nothing to annotate. The result is sorted
# again, since a split change can land after the next record's position, and
# records of one position with the same REF and ALT are joined (
# _JOIN_SAME_ALLELE_AWK below).
#
# THE ORIGIN TAG. Two changes split out of one FreeBayes allele are on one
# haplotype. Read as two separate heterozygous variants they would make a
# "possible compound heterozygote" of every proband with a multi-base call in
# a gene. Before splitting, each allele of a record that is not a plain
# single-base change is tagged INFO/PARALOME_ORIGIN=<POS>_<ALT number>_<ALT
# count> (Number=A, so the split gives each ALT its own value and --atomize
# copies it to each piece); gene burden treats alleles with one origin as one
# event per haplotype.
ORIGIN_TAG = "PARALOME_ORIGIN"
#: First bcftools release with norm --atomize.
ATOMIZE_MIN_BCFTOOLS = "1.13"

_ORIGIN_AWK = r"""
BEGIN { FS = "\t"; OFS = "\t"; had = 0 }
/^##INFO=<ID=PARALOME_ORIGIN,/ { had = 1; print; next }
/^##/ { print; next }
/^#CHROM/ {
    if (!had) print "##INFO=<ID=PARALOME_ORIGIN,Number=A,Type=String,Description=\"<POS>_<ALT number>_<ALT count> of the record this allele came from before Paralome split it for annotation (multi-allelic records split, MNPs atomized)\">"
    print
    next
}
{
    n = split($5, alts, ",")
    multi = (n > 1 || length($4) > 1)
    for (i = 1; i <= n && !multi; i++) if (length(alts[i]) > 1) multi = 1
    if (multi && index(";" $8, ";PARALOME_ORIGIN=") == 0) {
        tag = "PARALOME_ORIGIN="
        for (i = 1; i <= n; i++) tag = tag ((i > 1) ? "," : "") $2 "_" i "_" n
        $8 = ($8 == "." || $8 == "") ? tag : $8 ";" tag
    }
    print
}
"""


#: After the split, records of one position with the same REF and ALT
#: are one allele and are joined, their samples' genotypes combined haplotype by
#: haplotype (an ALT copy on either record is a copy). Two ALTs of one
#: FreeBayes call that share a base change (ACG>TCA,TCG: A>T twice) come out of
#: the split as two A>T records with different samples' genotypes; every later
#: duplicate filter (the annotation merge, the burden scan) kept the first and
#: dropped the second, losing its carriers and turning a homozygote of the
#: shared change into a heterozygote. Records whose FORMAT differs are left as
#: they are. The first record's INFO, QUAL and FILTER are kept; a sample's other
#: FORMAT fields come from the record that gave it its ALT copy.
#: Also run by the merge of the annotated shards (standalone
#: _stream_merge_bash), where the two records can come from different shards.
#: With ``-v JOINED=<file>`` it writes "<records joined> <records left apart>"
#: there at the end; without it, nothing. "Left apart": a record whose CHROM,
#: POS, REF and ALT matched a kept one but whose FORMAT (or GT ploidy) differed.
SAME_ALLELE_JOIN_AWK = r"""
BEGIN { FS = "\t"; OFS = "\t"; n = 0; site = ""; joined = 0; apart = 0 }
function flush(   i) {
    for (i = 1; i <= n; i++) if (!gone[i]) print rec[i]
    n = 0; delete rec; delete key; delete gone
}
function has_alt(gt,   a, k, j) {
    k = split(gt, a, /[\/|]/)
    for (j = 1; j <= k; j++) if (a[j] != "0" && a[j] != ".") return 1
    return 0
}
function join_gt(g1, g2,   a, b, k, m, j, sep, out, x, y) {
    sep = (index(g1, "|") > 0) ? "|" : "/"
    k = split(g1, a, /[\/|]/); m = split(g2, b, /[\/|]/)
    if (k != m) return ""
    out = ""
    for (j = 1; j <= k; j++) {
        x = a[j]; y = b[j]
        if (x == "1" || y == "1") v = "1"
        else if (x == "." || y == ".") v = "."
        else v = "0"
        out = out ((j > 1) ? sep : "") v
    }
    return out
}
function join(i, line,   f1, f2, keys, gi, kf, j, s1, s2, joined, c) {
    split(rec[i], f1, "\t"); split(line, f2, "\t")
    if (f1[9] != f2[9]) return 0
    kf = split(f1[9], keys, ":"); gi = 0
    for (j = 1; j <= kf; j++) if (keys[j] == "GT") gi = j
    if (!gi) return 0
    for (c = 10; c in f1; c++) {
        split(f1[c], s1, ":"); split(f2[c], s2, ":")
        joined = join_gt(s1[gi], s2[gi])
        if (joined == "") return 0
        if (!has_alt(s1[gi]) && has_alt(s2[gi])) { f1[c] = f2[c]; split(f1[c], s1, ":") }
        s1[gi] = joined
        f1[c] = s1[1]
        for (j = 2; j in s1; j++) f1[c] = f1[c] ":" s1[j]
    }
    out = f1[1]
    for (c = 2; c in f1; c++) out = out "\t" f1[c]
    rec[i] = out
    return 1
}
/^#/ { print; next }
{
    here = $1 SUBSEP $2
    if (here != site) { flush(); site = here }
    k = $4 SUBSEP $5
    matched = 0
    for (i = 1; i <= n; i++) if (key[i] == k && !gone[i]) { matched = 1; if (join(i, $0)) { joined++; next } }
    if (matched) apart++
    n++; rec[n] = $0; key[n] = k
}
END { flush(); if (JOINED != "") print joined, apart > JOINED }
"""
#: The former name, for anything that imports it.
_JOIN_SAME_ALLELE_AWK = SAME_ALLELE_JOIN_AWK


#: Which split for annotation (normalize_for_annotation_bash) made an
#: annotated VCF, recorded in the annotation checkpoint: v1 everything
#: atomized, v2 MNPs alone atomized, v3 pieces of one allele joined, v4 the
#: merge of the annotated shards joins same-allele records of different
#: carriers (standalone _stream_merge_bash). A finished annotation without
#: it, or with an older one, is read by the pipeline state (pipeline/state.py
#: summarize_annotation) as to be done again. Change it whenever the split,
#: or the merge after it, changes which records and carriers the annotated
#: VCF holds.
SPLIT_METHOD = "paralome.annotation_split.v4"


def normalize_for_annotation_bash(src: str, dst: str, reference: Optional[Path], tmp: str) -> str:
    """Bash that writes ``src`` to ``dst`` (uncompressed VCF) split into gnomAD's representation.

    ``src``, ``dst`` and ``tmp`` (a directory for bcftools sort to spill into)
    are bash words, already quoted. Stops the job, with the bcftools version
    found, when bcftools has no ``norm --atomize`` (1.13 and later have it).
    Without a reference FASTA the split is still made; indels are not
    left-aligned, and the log says so.
    """
    # Split first, then atomize only the MNPs (REF and ALT the same
    # length, longer than one base: their pieces are SNVs at separate
    # positions). Atomizing a complex allele whose SNV and indel pieces share a
    # base (CA>T, AC>TCC) makes bcftools give every carrier the '*' overlap
    # allele, so complex alleles reach ANNOVAR whole (as a block substitution).
    norm = "bcftools norm -m -any"
    if reference is not None:
        norm += f" -f {shlex.quote(str(reference))} -c w"
    mnp = "strlen(REF)==strlen(ALT) && strlen(REF)>1"
    lines = [
        "# Split multi-allelic and MNP calls before ANNOVAR (gnomAD's representation):",
        "# bcftools norm -m -any --atomize, the atomize step on MNPs only ; see annotation/annovar.py.",
        "if ! (bcftools norm -h 2>&1 || true) | grep -q -- '--atomize'; then",
        "  echo \"ERROR: $(bcftools --version 2>/dev/null | sed -n 1p) has no 'norm --atomize'; Paralome needs "
        f"bcftools {ATOMIZE_MIN_BCFTOOLS} or later to split complex calls before annotation\" >&2",
        "  exit 1",
        "fi",
    ]
    if reference is None:
        lines.append("echo 'NOTE: no reference FASTA for this genome; calls are split but indels are not left-aligned'")
    lines += [
        f"mkdir -p {tmp} \"$(dirname {dst})\"",
        f"NORM_IN=$(bcftools view -H {src} | awk 'END {{ print NR }}')",
        f"bcftools view {src} \\",
        f"  | awk {shlex.quote(_ORIGIN_AWK)} \\",
        f"  | {norm} -O b -o {tmp}/split.bcf -",
        f"bcftools view -i {shlex.quote(mnp)} {tmp}/split.bcf -O u \\",
        f"  | bcftools norm --atomize -O b -o {tmp}/split_mnp.bcf -",
        f"bcftools view -e {shlex.quote(mnp)} {tmp}/split.bcf -O b -o {tmp}/split_rest.bcf",
        # Joined as text (one header, both bodies), then sorted. `bcftools concat` refuses two files
        # that both hold a chromosome once one of them has moved on to the next ("not contiguous").
        f"( bcftools view -h {tmp}/split_rest.bcf; bcftools view -H {tmp}/split_rest.bcf; "
        f"bcftools view -H {tmp}/split_mnp.bcf ) \\",
        "  | bcftools view -e 'ALT=\"*\"' -O u - \\",
        f"  | bcftools sort -T {tmp}/bcftools. -O v - \\",
        f"  | awk {shlex.quote(SAME_ALLELE_JOIN_AWK)} > {dst}",
        f"rm -f {tmp}/split.bcf {tmp}/split_mnp.bcf {tmp}/split_rest.bcf",
        f"NORM_OUT=$(bcftools view -H {dst} | awk 'END {{ print NR }}')",
        "echo \"Split for annotation: $NORM_IN records in, $NORM_OUT out\"",
    ]
    return "\n".join(lines)


def rename_info_ids_text(text: str) -> tuple[str, dict[str, str]]:
    """The same rename as the awk program, in Python, for one VCF's text.

    Used by tests to check the shell program against an independent reading,
    and available to any caller holding a small VCF in memory.
    """
    renames: dict[str, str] = {}
    source: dict[str, str] = {}
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        newline = line[len(body):]
        if body.startswith("##INFO=<ID="):
            rest = body[len("##INFO=<ID="):]
            cut = rest.find(",")
            if cut < 0:
                cut = rest.find(">")
            key = rest[:cut] if cut >= 0 else rest
            new = renamed_info_id(key)
            if source.get(new, key) != key:
                raise ValueError(
                    f"INFO IDs {source[new]} and {key} would both be named {new} once '-' and '+' are replaced"
                )
            source[new] = key
            if new != key:
                renames[key] = new
                body = "##INFO=<ID=" + new + rest[len(key):]
        elif body and not body.startswith("#"):
            fields = body.split("\t")
            if len(fields) > 7:
                items = []
                for item in fields[7].split(";"):
                    key, eq, value = item.partition("=")
                    items.append(renamed_info_id(key) + eq + value)
                fields[7] = ";".join(items)
                body = "\t".join(fields)
        out.append(body + newline)
    return "".join(out), renames


@dataclass(frozen=True)
class AnnovarConfig:
    annovar_dir: Path
    database_dir: Path
    # No build and no protocol of Paralome's own: a caller that leaves them out would otherwise
    # annotate any genome as human.
    buildver: str
    protocol: str
    operation: str
    nastring: str = "."
    #: The genome's FASTA, for left-aligning indels before ANNOVAR (see ORIGIN_TAG above).
    reference_fasta: Optional[Path] = None

    @property
    def table_annovar(self) -> Path:
        return Path(self.annovar_dir) / "table_annovar.pl"

    def protocols(self) -> list[str]:
        return [x.strip() for x in self.protocol.split(",") if x.strip()]

    def operations(self) -> list[str]:
        return [x.strip() for x in self.operation.split(",") if x.strip()]

    def validate(self) -> None:
        for name in ("buildver", "protocol", "operation"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(
                    f"ANNOVAR {name} is empty: pass --annovar-{name}, or record annovar_{name} "
                    "in the genome entry. Paralome never assumes a genome build."
                )
        if not self.table_annovar.exists():
            raise FileNotFoundError(f"table_annovar.pl not found: {self.table_annovar}")
        if not Path(self.database_dir).exists():
            raise FileNotFoundError(f"ANNOVAR database directory not found: {self.database_dir}")
        if self.reference_fasta is not None:
            for path in (Path(self.reference_fasta), Path(str(self.reference_fasta) + ".fai")):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"reference FASTA for splitting calls before ANNOVAR not found: {path} "
                        "(reference_fasta in the genome entry, or --reference-fasta; it needs its .fai index)"
                    )
        protocols = self.protocols()
        operations = self.operations()
        if len(protocols) != len(operations):
            raise ValueError(
                f"ANNOVAR protocol/operation count mismatch: {len(protocols)} protocol(s), {len(operations)} operation(s)"
            )
        missing = [
            protocol
            for protocol in protocols
            if not list(Path(self.database_dir).glob(f"{self.buildver}_{protocol}*"))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing ANNOVAR database files for genome build {self.buildver}: "
                f"{', '.join(missing)} in {self.database_dir}. "
                "Check --genome or override --annovar-protocol explicitly."
            )

    # The ANNOVAR invocation lives in annotation/standalone.py, after the split; nothing here runs
    # table_annovar on a VCF that has not been split.
