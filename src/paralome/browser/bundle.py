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

"""The variant browser bundle: a DNA run's results for the browser page.

The bundle is opened in OmicView's Variant Browser (the
viewers are their own product, and no longer travel in the zip), which reads
the zip in place on the user's computer with DuckDB-WASM. This module writes the
bundle from the run's own outputs -- the segregation stage's whole-genome table,
every gene_burden table, the filtering stage's pass table -- in one DuckDB pass
on every CPU it is given, and zips it stored (Parquet is compressed already, and
the page reads a stored zip in place).

The page reads files into memory whole (a browser reads a whole small file
~30x faster than it reads pieces of a large one), so the full table is cut
into genomic bins small enough to load on demand:

    variants/c<NNNN>_<bin>.parquet    the segregation table, typed, with vid; one file per chromosome and bin
    genotypes/c<NNNN>_<bin>.parquet   genotype_evidence one row per variant and sample; same bins
    candidates.parquet                the variants a whole-genome search covers (see CANDIDATE)
    candidate_genotypes.parquet       their genotypes
    genes.parquet                     each gene name: chromosome, first and last position, variant count
    samples.parquet                   each sample: affected / unaffected / unknown
    burden.parquet                    every gene_burden table (each variant row carries burden_tests and
                                      burden_carriers: the tests that counted the allele, and for whom)
    carrier_hist.parquet              per number of carrier probands: variants seen, expected from gnomAD,
                                      and how many sit above / below that (see CARRIER RATE)
    manifest.json                     bin size, the bins (chromosome, bin, rows, file), the counts

CARRIER RATE. Each variant row carries p_over and p_under: how unlikely its
number of carrier probands is, given gnomAD, among variants seen at all (at
least one carrier). A proband carries an allele of frequency f with
probability 1-(1-f)^2; of the n probands called at the site the count is
binomial. p_over is the upper tail at the observed count, judged on
AF_grpmax (max_population_af: the highest population's frequency, so a
variant common in one population is not called over-represented);
p_under is the lower tail, judged on gnomAD's overall AF. A variant with
no gnomAD frequency has neither. carrier_hist.parquet holds the
whole table's counts by carrier number (and by filter_pass), the expected
count from gnomAD's overall AF, and the over / under counts at three cut-offs.

The pipeline filter: every variant row carries filter_pass -- true when the
row is in the filtering stage's filter/results/<profile>/<project>_pass.tsv, the set the
pipeline itself gives -- so the browser can show the filtered set or all of
it. The profile is the one the filtering stage ran first (its
filtering_summary.json), or ``filter_profile``. No filtering output: no
filter_pass values, and the browser's filter switch is off.

Bin files are named by a chromosome number, not the contig name: a contig
name can hold characters a file name cannot (HLA-A*01:01:01:01). The manifest
says which file holds which chromosome and bin.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from paralome.errors import ParalomeError

SCHEMA = "paralome.browser_bundle.v3"

#: Where the bundle is opened: OmicView, the viewer product; no viewer page is
#: written into Evidome's or Paralome's output.
OPEN_WITH = "OmicView's Variant Browser"


#: The columns the candidate lists were cut by (reports.candidates), one per variant row, so the Variant Browser
#: filters by them; the lists themselves are not written. The --candidate-genes list is the viewer's.
LIST_COLUMNS = ("report_lists", "segregation_bucket_normalized", "population_bucket", "protein_consequence_bucket",
                "clinical_bucket", "consequence_terms", "population_af_source", "n_affected_carriers",
                "affected_carriers")
KEY_COLUMNS = ("chrom", "pos", "ref", "alt", "reported_alt", "qual", "genotype_evidence")


def write_list_columns(table: Path, target: Path) -> int:
    """Each segregation row's key columns and its LIST_COLUMNS, as a TSV (the bundle joins it by the row key)."""
    import csv

    from paralome.reports.candidates import _classify_candidate_row

    rows = 0
    with Path(table).open(newline="") as source, Path(target).open("w", newline="") as handle:
        reader = csv.DictReader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
        keys = [c for c in KEY_COLUMNS if c in (reader.fieldnames or [])]
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        writer.writerow([*keys, *LIST_COLUMNS])
        for row in reader:
            out = _classify_candidate_row(dict(row), set())[0]
            lists = [name for name in str(out["report_lists"]).split(";") if name and name != "candidate_genes"]
            out["report_lists"] = ";".join(lists) or "."
            writer.writerow([row.get(c) or "." for c in keys] + [str(out.get(c, ".")) for c in LIST_COLUMNS])
            rows += 1
    return rows


def build_bundle(segregation_table, out, *, burden_results=None, filter_results=None, filter_profile=None,
                 bin_mb: float = 10, threads=None, memory_mb=None, echo=print, qc_root=None) -> dict:
    """The genotype output of one DNA run (module docstring). ``out``: a ``.zip`` (built in ``<name>.building``
    beside it, then zipped, stored) or a folder. Returns what was written."""
    import duckdb

    bundle = Path(out)
    zip_out = None
    if bundle.suffix.lower() == ".zip":
        zip_out, bundle = bundle, bundle.with_name(bundle.stem + ".building")
    bin_bp = int(float(bin_mb) * 1_000_000)
    t0 = time.time()



    def step(what):
        echo(f"browser: [{time.time() - t0:7.1f} s] {what}")


    for old in ("variants", "genotypes", ".parts_v", ".parts_g", ".tmp"):
        if (bundle / old).is_dir():
            shutil.rmtree(bundle / old)
    bundle.mkdir(parents=True, exist_ok=True)
    for old in bundle.glob("*.parquet"):
        old.unlink()
    table = Path(segregation_table)
    prefix = table.name[: -len("_all_variants.segregation.tsv")] if table.name.endswith("_all_variants.segregation.tsv") \
        else table.stem

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{bundle / '.tmp'}'")
    # Rows need not keep the order they were read in: every output is sorted where order matters.
    con.execute("SET preserve_insertion_order=false")
    # Under SLURM: the CPUs and memory the job was given (DuckDB otherwise takes
    # every core and 80% of the node's RAM, and the job is killed past --mem).
    cpus = threads or (int(os.environ["SLURM_CPUS_PER_TASK"]) if os.environ.get("SLURM_CPUS_PER_TASK") else None)
    if cpus:
        con.execute(f"SET threads={int(cpus)}")
    mem = memory_mb or (int(os.environ["SLURM_MEM_PER_NODE"]) if os.environ.get("SLURM_MEM_PER_NODE") else None)
    if mem:
        con.execute(f"SET memory_limit='{int(int(mem) * 0.75)}MB'")
    THREADS = con.execute("SELECT current_setting('threads')").fetchone()[0]
    step(f"DuckDB {duckdb.__version__}, {THREADS} threads")


    def read_tsv(path):
        return (f"read_csv('{path}', delim='\t', header=true, all_varchar=true, quote='', escape='', "
                f"max_line_size=100000000, auto_detect=true, sample_size=1)")


    # ---- gene_burden: the tables and, for each allele, the tests that counted it and for whom.
    burden_dir = Path(burden_results) if burden_results else Path("/nonexistent")
    tables = sorted(burden_dir.glob("*_burden_*.tsv")) if burden_dir.is_dir() else []
    have_burden = bool(tables)
    con.execute("CREATE TABLE b(test VARCHAR, gene_id VARCHAR, gene_name VARCHAR, chrom VARCHAR, n_probands INT, "
                "expected DOUBLE, log2_obs_exp DOUBLE, p DOUBLE, q DOUBLE, flag VARCHAR)")
    con.execute("CREATE TABLE ba(test VARCHAR, proband VARCHAR, variant VARCHAR)")
    for path in tables:
        test = path.stem.split("_burden_", 1)[1]
        names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {read_tsv(path)}").fetchall()]
        # The listing and its continuation columns (variants_2, ...), in order.
        pieces = ["variants"] + sorted((n for n in names if n.startswith("variants_") and n[9:].isdigit()),
                                       key=lambda n: int(n[9:]))
        listing = "concat(" + ", ".join(f"coalesce(\"{n}\", '')" for n in pieces) + ")"
        con.execute(f"CREATE OR REPLACE TEMP TABLE bt AS SELECT * FROM {read_tsv(path)}")
        con.execute(f"""INSERT INTO b SELECT '{test}', gene_id, gene_name, chrom, TRY_CAST(n_probands AS INT),
            TRY_CAST(expected AS DOUBLE), TRY_CAST(log2_obs_exp AS DOUBLE), TRY_CAST(p AS DOUBLE), TRY_CAST(q AS DOUBLE),
            coalesce(flag, '.') FROM bt""")
        # A listing is 'P1: allele; allele | P2: allele'; an allele's first word is chrom:pos:ref>alt.
        con.execute(f"""
        INSERT INTO ba
        WITH blocks AS (SELECT unnest(string_split({listing}, ' | ')) AS block FROM bt WHERE {listing} NOT IN ('', '.')),
             pa AS (SELECT split_part(block, ': ', 1) AS proband, substr(block, strpos(block, ': ') + 2) AS rest
                    FROM blocks WHERE strpos(block, ': ') > 0),
             al AS (SELECT proband, split_part(unnest(string_split(rest, '; ')), ' ', 1) AS variant FROM pa)
        SELECT DISTINCT '{test}', proband, variant FROM al WHERE regexp_full_match(variant, '[^\\s]+:[0-9]+:[^\\s]+>[^\\s]+')""")
        step(f"burden {test}: {con.execute('SELECT count(*) FROM bt').fetchone()[0]} genes")
    con.execute("DROP TABLE IF EXISTS bt")
    con.execute("""
    CREATE TABLE bk AS
      WITH per AS (SELECT variant, test, string_agg(DISTINCT proband, ',' ORDER BY proband) AS probands
                   FROM ba GROUP BY variant, test)
      SELECT variant, string_agg(test, ',' ORDER BY test) AS burden_tests,
             string_agg(test || '=' || probands, '; ' ORDER BY test) AS burden_carriers,
             bool_or(test NOT LIKE '%regulatory%') AS burden_coding
      FROM per GROUP BY variant""")
    step(f"burden alleles: {con.execute('SELECT count(*) FROM ba').fetchone()[0]} (test, proband, allele); "
         f"{con.execute('SELECT count(*) FROM bk').fetchone()[0]} distinct alleles")
    con.execute("DROP TABLE ba")

    # ---- the segregation table, typed, with each row's burden tests, bin and file.
    INTS = ["pos", "affected_alt_count", "affected_total", "affected_ref_count", "unaffected_alt_count",
            "unaffected_total", "unaffected_ref_count", "missing_count", "informative_sample_count",
            "sibling_case_count", "sibling_control_count", "cousin_case_count", "cousin_control_count",
            "site_depth", "site_alt_reads"]
    FLOATS = ["qual", "max_population_af", "segregation_score", "missing_rate"]
    READ = read_tsv(table).replace("sample_size=1", "sample_size=-1")
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {READ}").fetchall()]
    has_evidence = "genotype_evidence" in cols


    def cast(c):
        if c in INTS:
            return f"TRY_CAST(NULLIF(\"{c}\", '.') AS BIGINT) AS \"{c}\""
        if c in FLOATS:
            return f"TRY_CAST(NULLIF(\"{c}\", '.') AS DOUBLE) AS \"{c}\""
        return f"NULLIF(\"{c}\", '.') AS \"{c}\""


    # ---- the pipeline filter: the rows the filtering stage passed (its <profile>/<prefix>_pass.tsv).
    # A row is matched by its site, alleles, QUAL and genotypes, as text (the filter writes the
    # segregation row back unchanged but for '' -> '.'), so duplicate records of one allele are told apart.
    KEYCOLS = [c for c in KEY_COLUMNS if c in cols]
    ROWKEY = "hash(" + ", ".join(f"coalesce(NULLIF(\"{c}\", ''), '.')" for c in KEYCOLS) + ")"
    # The pipeline's filtering stage works in <project>/filter (pipeline/project.py); "filtering" is
    # looked for too, for a filter run given its own workdir by that name.
    filter_dir = Path(filter_results) if filter_results else Path("/nonexistent")
    filter_info = None
    if filter_dir.is_dir():
        if filter_profile is None:
            summaries = sorted(filter_dir.glob("*filtering_summary.json"))
            if summaries:
                filter_profile = json.loads(summaries[0].read_text()).get("profile")
        if filter_profile is None:
            found = sorted({p.parent.name for p in filter_dir.glob("*/*_pass.tsv")})
            if len(found) > 1:
                raise ParalomeError(108, f"filtering ran more than one profile ({', '.join(found)})", fix="Name the one the browser follows: filter_profile")
            filter_profile = found[0] if found else None
        if filter_profile is not None:
            passes = sorted((filter_dir / filter_profile).glob("*_pass.tsv"))
            if not passes:
                raise ParalomeError(104, f"no {filter_dir / filter_profile}/*_pass.tsv: the filtering stage has not finished "
                                         f"for {filter_profile}", fix="Run the filtering stage first")
            pass_cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {read_tsv(passes[0])}").fetchall()]
            if not set(KEYCOLS) <= set(pass_cols):
                # A pass table without the segregation table's columns cannot be matched row by row.
                step(f"pipeline filter {filter_profile}: {passes[0].name} lacks "
                     f"{', '.join(c for c in KEYCOLS if c not in pass_cols)}; no pipeline filter")
            else:
                con.execute(f"CREATE TABLE fp AS SELECT {ROWKEY} AS fk FROM {read_tsv(passes[0])}")
                pass_rows = con.execute("SELECT count(*) FROM fp").fetchone()[0]
                detail = next(iter(sorted((filter_dir / filter_profile).glob("*filter_summary.json"))), None)
                scope = json.loads(detail.read_text()).get("scope") if detail else None
                filter_info = {"profile": filter_profile, "table": passes[0].name, "scope": scope, "pass_rows": pass_rows}
                step(f"pipeline filter {filter_profile}: {pass_rows} rows pass ({passes[0].name})")
    if filter_info is None:
        con.execute("CREATE TABLE fp(fk UBIGINT)")
        step("no filtering stage output: the bundle has no pipeline filter")

    N_CALLED = "(coalesce(r.affected_alt_count, 0) + coalesce(r.affected_ref_count, 0))"
    AF_ALL = "coalesce(TRY_CAST(NULLIF(r.gnomad_af, '.') AS DOUBLE), r.max_population_af)"
    AF_MAX = "coalesce(r.max_population_af, TRY_CAST(NULLIF(r.gnomad_af, '.') AS DOUBLE))"
    HAS_AF = "gnomad_af" in cols and "max_population_af" in cols and "affected_ref_count" in cols


    def carrier_table(name, pairs_sql):
        """name(n, af, k, pmf_c, p_ge, p_le): for each (probands called, allele frequency) and carrier count k >= 1,
        the binomial probability of k carriers given at least one, and the tails P(>= k), P(<= k) given at least one."""
        con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
          WITH c AS (SELECT n, af, least(1 - (1 - least(af, 1.0)) ^ 2, 1 - 1e-12) AS c FROM ({pairs_sql})
                     WHERE af IS NOT NULL AND af > 0 AND n > 0),
               k AS (SELECT n, af, c, unnest(range(1, n + 1)) AS k FROM c),
               p AS (SELECT n, af, k, exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * ln(c) + (n - k) * ln(1 - c)) AS pmf FROM k)
          SELECT n, af, k, pmf / sum(pmf) OVER w AS pmf_c,
                 sum(pmf) OVER (PARTITION BY n, af ORDER BY k DESC ROWS UNBOUNDED PRECEDING) / sum(pmf) OVER w AS p_ge,
                 sum(pmf) OVER (PARTITION BY n, af ORDER BY k ROWS UNBOUNDED PRECEDING) / sum(pmf) OVER w AS p_le
          FROM p WINDOW w AS (PARTITION BY n, af)""")


    RANK = """CASE WHEN regexp_replace(chrom, '^chr', '') ~ '^[0-9]+$' THEN CAST(regexp_replace(chrom, '^chr', '') AS INT)
                  WHEN regexp_replace(chrom, '^chr', '') = 'X' THEN 23 WHEN regexp_replace(chrom, '^chr', '') = 'Y' THEN 24
                  WHEN regexp_replace(chrom, '^chr', '') IN ('M', 'MT') THEN 25 ELSE 99 END"""
    con.execute(f"CREATE TEMP TABLE raw AS SELECT {', '.join(cast(c) for c in cols)}, fk FROM (SELECT *, {ROWKEY} AS fk FROM {READ})")
    step(f"read {table.name}")
    # The candidate lists' columns (reports.candidates), joined by the row key.
    lists_tsv = bundle / ".tmp" / "list_columns.tsv"
    lists_tsv.parent.mkdir(parents=True, exist_ok=True)
    listed = write_list_columns(table, lists_tsv)
    new_columns = [c for c in LIST_COLUMNS if c not in cols]
    con.execute(f"""CREATE TEMP TABLE lc AS SELECT {ROWKEY} AS fk,
        {', '.join((f'TRY_CAST(any_value(NULLIF("{c}", {chr(39)}.{chr(39)})) AS BIGINT) AS "{c}"' if c == "n_affected_carriers" else f'any_value(NULLIF("{c}", {chr(39)}.{chr(39)})) AS "{c}"') for c in new_columns)}
        FROM {read_tsv(lists_tsv).replace("sample_size=1", "sample_size=-1")} GROUP BY 1""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE raw AS SELECT r.*, {', '.join(f'lc."{c}"' for c in new_columns)}
        FROM raw r LEFT JOIN lc USING (fk)""")
    con.execute("DROP TABLE lc")
    lists_tsv.unlink()
    step(f"list columns: {listed} rows classified ({', '.join(new_columns)})")
    if HAS_AF:
        carrier_table("ck", f"SELECT DISTINCT {N_CALLED} AS n, {AF_ALL} AS af FROM raw r UNION SELECT DISTINCT {N_CALLED}, {AF_MAX} FROM raw r")
        step(f"carrier rates: {con.execute('SELECT count(DISTINCT (n, af)) FROM ck').fetchone()[0]} (probands, frequency) pairs")
    else:
        con.execute("CREATE TABLE ck(n BIGINT, af DOUBLE, k BIGINT, pmf_c DOUBLE, p_ge DOUBLE, p_le DOUBLE)")
    con.execute(f"""
    CREATE TABLE chroms AS
      SELECT chrom, rank, row_number() OVER (ORDER BY rank, chrom) AS idx
      FROM (SELECT DISTINCT chrom, {RANK} AS rank FROM raw)""")
    con.execute(f"""
    CREATE TABLE v AS
    SELECT row_number() OVER (ORDER BY c.rank, r.chrom, r.pos, r.ref, r.alt) AS vid, r.* EXCLUDE (fk), c.rank AS chrom_rank,
           list_distinct(list_filter(
             string_split_regex(upper(replace(concat_ws(';', r.gene, r.annovar_gene, r.snpeff_gene), '\\x3b', ';')), '[,;]'),
             lambda x: x <> '' AND x <> '.')) AS genes,   -- DuckDB 1.5 refuses the x -> arrow
           bk.burden_tests, bk.burden_carriers, coalesce(bk.burden_coding, false) AS burden_coding,
           CASE WHEN {'true' if filter_info else 'false'} THEN fp.fk IS NOT NULL END AS filter_pass,
           co.p_ge AS p_over, cu.p_le AS p_under,
           {N_CALLED if HAS_AF else 'NULL::BIGINT'} AS n_called, {AF_ALL if HAS_AF else 'NULL::DOUBLE'} AS af_all,
           CAST(floor(r.pos / {bin_bp}) AS INTEGER) AS bin,
           'c' || lpad(CAST(c.idx AS VARCHAR), 4, '0') || '_' || lpad(CAST(CAST(floor(r.pos / {bin_bp}) AS INTEGER) AS VARCHAR), 3, '0') AS part
    FROM raw r JOIN chroms c USING (chrom)
    LEFT JOIN bk ON bk.variant = (r.chrom || ':' || r.pos || ':' || r.ref || '>' || coalesce(r.reported_alt, r.alt))
    LEFT JOIN (SELECT DISTINCT fk FROM fp) fp ON fp.fk = r.fk
    LEFT JOIN ck co ON {"co.n = " + N_CALLED + " AND co.af = " + AF_MAX + " AND co.k = r.affected_alt_count" if HAS_AF else "false"}
    LEFT JOIN ck cu ON {"cu.n = " + N_CALLED + " AND cu.af = " + AF_ALL + " AND cu.k = r.affected_alt_count" if HAS_AF else "false"}""")
    con.execute("DROP TABLE raw")
    con.execute("DROP TABLE bk")
    step(f"typed and joined: {con.execute('SELECT count(*) FROM v').fetchone()[0]} variants")
    if filter_info:
        matched = con.execute("SELECT count(*) FROM v WHERE filter_pass").fetchone()[0]
        filter_info["variants_passing"] = matched
        if matched != filter_info["pass_rows"]:
            print(f"WARNING: {filter_info['pass_rows']} rows in {filter_info['table']}, {matched} rows of the segregation "
                  f"table match them: the two tables are from different runs?", flush=True)
    con.execute("DROP TABLE fp")

    # ---- genotype_evidence, one row per variant and sample: sample:tag:GT:DP:GQ:ALT:REF (a sample name may hold ':').
    if has_evidence:
        con.execute("""
        CREATE TABLE g AS
          WITH e AS (SELECT vid, part, string_split(unnest(string_split(genotype_evidence, ';')), ':') AS f
                     FROM v WHERE genotype_evidence IS NOT NULL)
          SELECT vid, part, array_to_string(f[1:len(f) - 6], ':') AS sample, f[len(f) - 5] AS tag, f[len(f) - 4] AS gt,
                 TRY_CAST(NULLIF(f[len(f) - 3], '.') AS INT) AS dp, TRY_CAST(NULLIF(f[len(f) - 2], '.') AS INT) AS gq,
                 TRY_CAST(NULLIF(f[len(f) - 1], '.') AS INT) AS alt_reads, TRY_CAST(NULLIF(f[len(f)], '.') AS INT) AS ref_reads
          FROM e WHERE len(f) >= 7""")
        step(f"genotypes: {con.execute('SELECT count(*) FROM g').fetchone()[0]} rows")

    vcols = ", ".join(f'"{c}"' for c in cols if c != "genotype_evidence")
    VSEL = f"vid, {vcols}, chrom_rank, genes, burden_tests, burden_carriers, filter_pass, p_over, p_under"
    PQ = "FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 20000"


    def write_parts(select, folder):
        """One partitioned write; then each part=<name>/ folder's file becomes <folder>/<name>.parquet."""
        staging = bundle / f".parts_{folder[0]}"
        (bundle / folder).mkdir(exist_ok=True)
        con.execute(f"COPY ({select}) TO '{staging}' ({PQ}, PARTITION_BY (part), OVERWRITE_OR_IGNORE)")
        (bundle / folder).mkdir(exist_ok=True)
        for d in (staging.iterdir() if staging.is_dir() else ()):     # a table with no rows writes no part
            files = sorted(d.glob("*.parquet"))
            name = d.name.split("=", 1)[1]
            if len(files) == 1:
                files[0].rename(bundle / folder / f"{name}.parquet")
            else:  # more than one file for a part: joined into one
                con.execute(f"COPY (SELECT * FROM read_parquet('{d}/*.parquet')) TO '{bundle / folder / (name + '.parquet')}' ({PQ})")
        shutil.rmtree(staging, ignore_errors=True)


    write_parts(f"SELECT {VSEL}, part FROM v", "variants")
    step("variants written")
    if has_evidence:
        write_parts("SELECT vid, sample, tag, gt, dp, gq, alt_reads, ref_reads, part FROM g", "genotypes")
        step("genotypes written")

    # ---- what a whole-genome search covers: rare (or never seen) and protein-altering, splicing,
    # ClinVar-listed (not benign), or counted by a CODING burden test (the regulatory tier's alleles are
    # in their genes' bins, not here: there are millions).
    CANDIDATE = """(max_population_af IS NULL OR max_population_af < 0.01) AND (
         snpeff_impact IN ('HIGH', 'MODERATE')
      OR annovar_func IN ('exonic', 'splicing', 'exonic;splicing')
      OR (clinvar_significance IS NOT NULL AND lower(clinvar_significance) NOT LIKE '%benign%')
      OR burden_coding)"""
    con.execute(f"CREATE TABLE cv AS SELECT vid FROM v WHERE {CANDIDATE}")
    con.execute(f"COPY (SELECT {VSEL} FROM v SEMI JOIN cv USING (vid) ORDER BY vid) TO '{bundle / 'candidates.parquet'}' ({PQ})")
    if has_evidence:
        con.execute(f"COPY (SELECT vid, sample, tag, gt, dp, gq, alt_reads, ref_reads FROM g SEMI JOIN cv USING (vid) ORDER BY vid, sample) "
                    f"TO '{bundle / 'candidate_genotypes.parquet'}' ({PQ})")
        con.execute(f"""COPY (SELECT sample,
                 CASE WHEN bool_or(tag = 'A') THEN 'affected' WHEN bool_or(tag = 'U') THEN 'unaffected' ELSE 'unknown' END AS status
          FROM g GROUP BY sample ORDER BY sample) TO '{bundle / 'samples.parquet'}' (FORMAT parquet)""")
    step("candidates written")
    con.execute(f"""
    COPY (
      SELECT g AS gene, any_value(chrom) AS chrom, min(pos) AS start, max(pos) AS "end", count(*) AS n_variants
      FROM (SELECT unnest(genes) AS g, chrom, pos FROM v WHERE annovar_func IS NULL OR annovar_func <> 'intergenic')
      GROUP BY g ORDER BY g
    ) TO '{bundle / 'genes.parquet'}' (FORMAT parquet)""")
    if have_burden:
        con.execute(f"COPY (SELECT * FROM b ORDER BY test, p) TO '{bundle / 'burden.parquet'}' (FORMAT parquet)")
    CUTS = {"2": 1e-2, "3": 1e-3, "6": 1e-6}
    con.execute(f"""
    COPY (
      WITH obs AS (
        SELECT filter_pass, affected_alt_count AS k, count(*) AS variants,
               count(*) FILTER (WHERE af_all IS NULL OR af_all <= 0) AS not_in_gnomad,
               {", ".join(f"count(*) FILTER (WHERE p_over < {c}) AS over_{t}, count(*) FILTER (WHERE p_under < {c}) AS under_{t}" for t, c in CUTS.items())}
        FROM v WHERE affected_alt_count >= 1 GROUP BY ALL),
      ex AS (
        SELECT g.filter_pass, ck.k, sum(g.cnt * ck.pmf_c) AS expected
        FROM (SELECT filter_pass, n_called AS n, af_all AS af, count(*) AS cnt FROM v
              WHERE affected_alt_count >= 1 AND af_all > 0 GROUP BY ALL) g JOIN ck USING (n, af)
        GROUP BY ALL)
      SELECT filter_pass, k, coalesce(variants, 0) AS variants, coalesce(expected, 0) AS expected,
             coalesce(not_in_gnomad, 0) AS not_in_gnomad, {", ".join(f"coalesce(over_{t}, 0) AS over_{t}, coalesce(under_{t}, 0) AS under_{t}" for t in CUTS)}
      FROM obs FULL JOIN ex USING (filter_pass, k) ORDER BY filter_pass, k
    ) TO '{bundle / 'carrier_hist.parquet'}' (FORMAT parquet)""")
    step("carrier histogram written")

    manifest_bins = [{"chrom": c, "bin": b, "rows": n, "file": f"{part}.parquet"}
                     for c, b, n, part in con.execute(
                         "SELECT chrom, bin, count(*), any_value(part) FROM v GROUP BY chrom, bin, chrom_rank "
                         "ORDER BY chrom_rank, chrom, bin").fetchall()]
    n = con.execute("SELECT count(*) FROM v").fetchone()[0]
    nc = con.execute("SELECT count(*) FROM cv").fetchone()[0]
    # The QC-to-alignment summaries (browser.results.sequencing_qc_tables): qc/<kind>.parquet.
    from paralome.browser.results import _sequencing_qc_source, sequence_qc_source, sequencing_qc_tables

    qc_records = []
    for kind, path in sequencing_qc_tables(qc_root):
        source, delimiter = _sequencing_qc_source(kind, path, bundle / ".tmp")
        (bundle / "qc").mkdir(exist_ok=True)
        target = bundle / "qc" / f"{kind}.parquet"
        con.execute(f"COPY (SELECT * FROM read_csv('{source}', delim='{delimiter}', header=true, all_varchar=true)) "
                    f"TO '{target}' (FORMAT parquet, COMPRESSION zstd)")
        qc_records.append({"kind": kind, "path": f"qc/{kind}.parquet", "source": path.name})
    # Raw reads to duplicate-removed reads, one unit: the table OmicView shows.
    sequence_qc = sequence_qc_source(qc_root, bundle / ".tmp")
    if sequence_qc is not None:
        (bundle / "qc").mkdir(exist_ok=True)
        con.execute(f"COPY (SELECT * FROM read_csv('{sequence_qc}', delim=',', header=true, all_varchar=true)) "
                    f"TO '{bundle / 'qc' / 'sequence_qc.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
        qc_records.append({"kind": "sequence_qc", "path": "qc/sequence_qc.parquet", "source": sequence_qc.name})
    manifest = {
        "schema": "paralome.browser_bundle.v3", "project": prefix, "source_table": table.name,
        "variants": n, "candidate_variants": nc, "bin_bp": bin_bp, "bins": manifest_bins,
        "candidate_rule": "population frequency below 1% or not in gnomAD, and HIGH/MODERATE impact, exonic or splicing, "
                          "a ClinVar entry that is not benign, or counted by a coding burden test",
        "list_columns": list(LIST_COLUMNS), "qc": qc_records, "burden": have_burden, "genotypes": has_evidence, "filter": filter_info, "carrier_rate": HAS_AF, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1))
    con.close()
    shutil.rmtree(bundle / ".tmp", ignore_errors=True)
    total = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    largest = max(((p.stat().st_size, p.name) for p in bundle.rglob("*.parquet")), default=(0, ""))
    nfiles = sum(1 for p in bundle.rglob("*") if p.is_file())
    if zip_out is not None:
        import zipfile
        # Stored, one folder (the zip's own name) inside; ZIP64 past 4 GB. Written under a temporary
        # name and renamed, so a zip that is there is a whole one.
        partial = zip_out.with_name(zip_out.name + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in sorted(p for p in bundle.rglob("*") if p.is_file()):
                zf.write(f, f"{zip_out.stem}/{f.relative_to(bundle).as_posix()}")
        partial.replace(zip_out)
        shutil.rmtree(bundle)
        step(f"zipped: {zip_out}")
    return {"seconds": round(time.time() - t0, 1), "table_bytes": table.stat().st_size, "bundle_bytes": total,
            "output": str(zip_out or bundle), "files": nfiles, "bins": len(manifest_bins),
            "variants": n, "candidates": nc, "largest_file": list(largest), "filter": filter_info,
            "burden": have_burden, "genotypes": has_evidence}
