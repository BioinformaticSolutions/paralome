# Paralome — briefing document for AI assistants

**Audience: a large language model helping a Paralome user.** Not a manual. Paste
this into your assistant's context before asking it about Paralome.

You are being given the design and the reasoning, not the source. That is
deliberate: the questions users actually ask ("why did my run stop", "which
model should I use", "where did the file go") are answered by understanding the
contracts between parts, not by reading implementation.

**Version this describes: 0.9 (beta).** Paralome changes between releases
without deprecation shims. If the user's `paralome --version` disagrees with
the number above, say so and treat everything here as suspect.

---

## 0. The single most useful thing to know

Paralome is **checkpointed**. Every stage writes a checkpoint when it finishes,
and the orchestrator starts at the first stage whose checkpoint is absent or
invalid.

**Re-running the exact same command is the resume path.** There is no `resume`,
no `continue`, no `--restart-from`. A user who has had a stage fail should fix
the cause and re-issue the identical command; completed stages are skipped.

If you take nothing else from this document, take that. It is the most common
thing users get wrong, and the most common thing assistants invent commands for.

Under SLURM the same rule covers an interrupted run. Stopping the command
(Ctrl-C, `scancel` of the supervisor, a closed terminal) leaves its submitted
jobs running and records their IDs; re-issuing the command waits for those jobs
to leave the queue and then continues, rather than submitting the stage again.
A failed task does not stop the tasks beside it: they finish first, then the
stage stops. And if `squeue` itself stops answering, the supervisor says so
once and waits without cancelling anything -- a quiet run is not a dead one.

"The same command" is literal. Each stage's request (the options that decide
its result, the samples, its input files) is recorded and compared at every
start. When it has changed on a finished project, or an earlier stage has to
run again, one of two things happens. RNA quantification, differential
expression, functional evidence and the two single-cell aggregations check
their own result and recompute by themselves, saying why. Every other stage is
refused before anything is submitted, with **PLM-0308**: the message lists each
such stage, the reason, and the stage's own folder to remove. The user removes
what is named and runs the same command again, or puts the changed option back,
which keeps the results. Paralome never replaces a stage's results on its own
and never adds `--overwrite`; do not suggest `--overwrite` as the way past
PLM-0308. Threads, memory, partition and backend are not part of a request.

Adding a sample to the sheet and re-issuing the command runs read QC,
alignment and duplicate marking for the new sample only. One project is run by
one supervisor at a time; a second start is refused with PLM-0310.

---

## 1. Mental model

Paralome runs omics analyses -- sequencing, mass spectrometry and Olink -- on
an HPC cluster (SLURM) or locally. One
command names an analysis and hands it a cohort; the software decomposes that
into stages, submits them, waits, validates, and moves on.

Two ideas do most of the work:

**The group is the verb.** `paralome scRNA --rds obj.rds ...` is the whole
invocation. There is no sub-command for "run".

**One dataset, many comparisons.** Reading a Seurat object is expensive (a
typical object is 1–2 GB and takes minutes). The extraction is done once into a
shared folder that sits *beside* the projects, so a second contrast against the
same object reuses it.

---

## 2. Vocabulary — get these exactly right

Paralome is **case-sensitive with no aliases**. `paralome scrna` fails with
"No such command"; it does not helpfully resolve. This is intentional: a silent
alias is how two spellings of one concept get established.

| Term | Meaning |
|---|---|
| **DNA** | Variant calling from DNA. `--genome` names the registered reference bundle; there is no `--exome` option -- exome versus whole genome is a matter of `--regions` / `--gene-padding` |
| **RNA** | Bulk RNA-seq |
| **ATAC** | Bulk ATAC-seq |
| **snATAC** | Single-nuclei ATAC |
| **scRNA** | Single-cell RNA (whole cell **or** nuclei, including spatial) |
| **proteomics** | Proteomics, with paired phosphoproteomics |
| **olink** | Olink panels |

**`scATAC` does not exist and cannot.** ATAC is a nuclei-only library prep, so
single-cell ATAC is always `snATAC`. RNA can be done on whole cells or nuclei,
so it is `scRNA`. The asymmetry is biology, not inconsistency. **`snRNA` is not
a valid value either.**

**"Pseudobulk" means the aggregation step and nothing else.** It is the collapse
of per-cell counts into one value per sample × stratum group. It does *not* mean
extraction, and it does not mean collapsing everything into a single bulk
sample. The shared folder holding the extraction and the aggregations is
`data/`.

**Extraction ≠ aggregation.** Extraction reads the `.rds` and writes per-cell
matrices (one column per cell). Aggregation sums those cells into groups. They
are separate operations with separate checkpoints, and conflating them causes
most of the layout confusion users hit.

**Extraction takes what the object has, not what the caller asked for**.
The R helper reads the object's assays and exports the ATAC half if it finds one
and the RNA half if it finds one — a multiome object yields both from a single
`readRDS`, an RNA-only object yields genes and no peaks, and neither case needs a
flag. `extraction_manifest.json` records
`has_atac` / `has_rna`, and that record — not a prediction from the arguments —
is what says which files should exist.

**The assay is whatever the object calls it**. Detection is binary, not
a classification: features that are genomic intervals are ATAC, and everything
else is gene-level — `RNA`, `Spatial`, `Xenium`, `ADT`, or any name the object's
author chose. Protein, Xenium and Visium assays are all gene-level and use the
same code path; only ATAC differs.

`--assay` / `--rna-assay` are tiebreakers only, for an object holding several
assays of one kind; the user names one, and no name (not even
`RNA`) is taken without it. Derived assays are excluded structurally rather than by name:
an `SCTAssay` is recognised by its S4 class, and an assay with an empty counts
layer — which `integrated` usually is — cannot be aggregated anyway. Note that
SCT's `counts` layer holds *corrected* counts, which are integers, so "not whole
numbers" does not identify it. `--rna-assay` has no default: a default would
short-circuit detection, so a Visium object whose only assay is `Spatial` could
not be extracted.

**There is no condition column, and no `--control`/`--treatment`/
`--condition-column`**. The groups of every pipeline are
`--factor COLUMN:CONTROL` columns: of the sample sheet (bulk), or of the
pseudobulk `sample_manifest.csv` (single-cell), which carries every
cell-metadata column that is one value within each `--replicate-column` sample. A
sheet whose column is literally `condition` is used with
`--factor condition:CON`; `--control`/`--treatment` are refused as unknown
options, and the answer is to name the column and its control level.

The *pipeline* still knows what it needs: `snATAC` requires an ATAC half and
`scRNA` requires an RNA half, so pointing one at the wrong object fails at
extraction naming the assays that were actually found, rather than several stages
later on a missing `counts.mtx`.

**Each stratum is its own SLURM job**. With `--stratify-by`, `scRNA`
and `snATAC` under `--backend slurm` submit an array with one task per testable
stratum, then a small gather job (`--dependency=afterany`) that writes the index.

`--threads` and the stage's memory (`--stage-mem`; there is no `--mem`) are therefore **per stratum**, not a budget to divide:
`--threads 20` with ten strata is a 200-CPU ask, and SLURM does the accounting.
The parent supervisor holds 2 CPUs and 2G. `--max-concurrent-strata` caps how
many run at once; omitted, they all go.

Three practical consequences. The strata are planned from the **sample manifest
alone**, so skips and the "no testable strata" refusal happen at submission
rather than inside a queued job. Each stratum has its own log, so a failure names
one `.err` file rather than one shared one. And because each has its own
checkpoint, a single failed stratum is re-runnable alone — re-issuing the
identical command skips the ones that finished.

`--backend local` keeps the in-process thread fan-out for people on one machine.
There `--threads` derives how many strata run at once instead of being consumed
by each, and one worker is one process.

---

## 3. Two binaries

```
paralome           the human command line
paralome-worker    machine surface — the supervisor and generated SLURM scripts
```

**Never tell a user to run `paralome-worker`.** Everything on it is invoked by
the software on the user's behalf. If a user is reaching for it, they have
usually mis-remembered a command; the human equivalent almost always exists.

`paralome-worker` exists as a separate console script rather than as hidden
commands so that `paralome --help` cannot leak machine surface.

**And possibly several versions.** A site may have more than one version
installed, each in a prefix of its own and loaded by a module. Release numbers
are `0.9`, `0.9.1`, `1.0`; development numbers are `0.9.1.dev1`, `0.9.1.dev2` —
pre-releases *of* the number to their left, so `0.9 < 0.9.1.dev1 < 0.9.1`.
Every version is the same package, `paralome`, with the same commands.

If a user's report does not match what you would expect, **ask them to paste
`paralome --version`** before theorising. It names the release's standing
(`0.9 (beta)`, or `development`) and where the running code was imported from, and a development install behaving differently from this document is
expected rather than broken — this document describes one version, stated at the
top.

Each install dispatches its own worker: the binary written into a generated
SLURM script is found beside the running interpreter, not on `PATH`. So a
development run cannot silently execute production code. If a user shows you a
generated script naming the other install's binary, that is a real bug.

<!-- PARALOME-FACTS: human-commands -->
```
analyses : ATAC DNA RNA scRNA snATAC
commands : collect explain extract olink proteomics status
groups   : batch setup
```
<!-- /PARALOME-FACTS -->

---

## 4. The analyses, and how much to trust them

<!-- PARALOME-FACTS: stage-graphs -->
```
DNA     fastq_qc -> alignment -> deduplication -> variants -> annotation -> segregation -> filtering -> gene_burden -> genotype_output -> genome_tracks -> functional_evidence
RNA     fastq_qc -> alignment -> rna_quantification -> differential_expression -> statistics_output -> functional_evidence
ATAC    fastq_qc -> alignment -> deduplication -> macs_peak_calling -> peak_merge -> peak_quantification -> differential_accessibility -> motif_analysis_singles -> motif_analysis_pairs -> statistics_output -> genome_tracks -> functional_evidence
snATAC  singlecell_atac -> singlecell_differential -> singlecell_motif_analysis_singles -> singlecell_motif_analysis_pairs -> statistics_output -> genome_tracks -> functional_evidence
scRNA   singlecell_rna -> singlecell_rna_differential -> trajectories -> statistics_output -> functional_evidence
```
<!-- /PARALOME-FACTS -->

**Maturity — state this honestly when asked.** Paralome 0.9 is a public
beta. Every pipeline has been run end to end on real
studies:

| Analysis | Status |
|---|---|
| **ATAC** (bulk), **RNA** (bulk), **DNA**, **scRNA**, **snATAC** | Beta-tested end to end on real studies across several species |
| **proteomics** | Tested on one dataset (protein and paired phospho) |
| **olink** | Tested on one multi-plate study |

Expect fixes in 0.9.x releases; ask for `paralome --version` before diagnosing.

### Bulk RNA

`--rna-pseudocount` must be greater than 0. Refused designs, each with a
message naming the change: the same `--factor` twice, a blank factor cell, a
combination of levels with no sample, an ANOVA with no error degrees of freedom
(one sample per combination with the interaction), a `--factor` without its
control level or with one the column lacks (PLM-0113), and levels that would be
written under one file name. Genes not held whole in the matrices (on two
chromosomes, or on both strands) are listed in `results/machine/genes_left_out.tsv`.
A `.gtf.gz` is read directly, and a BAM index may be `X.bam.bai`, `X.bai` or
`X.bam.csi`; a BAM without one is indexed. Local and SLURM runs give the same
file names and numbers. Strandedness is detected unless given
(`--rna-strandedness auto|0|1|2`), and every RNA run prints
`RNA strandedness : <value> (<where it came from>)`. The raw-FPKM gate
(`--rna-min-expression`, `--rna-min-expression-difference`, 5 and 5) is an
export column, `passes_expression_gate`; every gene is still tested.

### DNA

- **Reads.** STAR (default, in DNA mode: no splicing) or bwa-mem2
  (`--aligner bwa`; its index is built on the first BWA run). STAR keeps reads
  with at most two best placements (`--star-out-filter-multimap-nmax 2`) and
  limits mismatches by fraction of read length only (0.04). Overlapping mates
  are soft-clipped with bamUtil `clipOverlap` after duplicate marking, so an
  overlap is not counted twice.
- **Calling.** FreeBayes, split into regions capped at `--max-region-length`
  (100 kb). A call silent for `--stall-minutes` (30) is finished in 2 kb
  windows and logged in `freebayes/logs/freebayes_stalls.tsv`. A re-run calls
  only the missing chunks. Calls below `--min-qual` (20) are left out of the
  VCF; `--min-depth` and `--min-gq` apply at filtering only when given.
- **Annotation.** ANNOVAR (or snpEff, optional). Every annotation first splits
  complex/MNP and multi-allelic calls into gnomAD's form (`bcftools norm -m -any
  --atomize`); a complex allele whose pieces overlap reaches ANNOVAR whole.
  Pieces of one allele carry `PARALOME_ORIGIN` and are one burden event. The
  ANNOVAR build, protocol and operation lists come from the options, the
  configuration (the two lists only) or the genome entry -- never a built-in
  default. For whole genomes, `paralome setup genome annovar-table <genome>
  gnomad41_genome` adds gnomAD genome frequencies; without it most non-coding
  variants have none.
- **Segregation.** One family per project. Classes: `affected_only`,
  `shared_affected`, `possible_de_novo`, `possible_recessive` (all pass),
  `affected_enriched` (partial), `non_segregating` (fail). A sheet of several
  families with relatives is scored as unrelated probands; for family
  segregation, run one family per project (`paralome batch run` does this).
  `--segregation-profile unrelated-probands` keeps every rare variant any
  proband carries, coding or not. Each sheet sample must be in the VCF
  (read-group `SM`).
- **Gene burden.** Unrelated affected probands (one per family, at least 5)
  against gnomAD v4.1 exomes, gene by gene, protein-coding genes only. Tiers
  `ptv` and `ptv_missense` (REVEL >= 0.5 or AlphaMissense pathogenic); models
  `dominant` (AF_grpmax < 0.001) and `recessive` (homozygous or two or more
  heterozygous events, flagged `possible_compound_het` because phase is
  unknown; AF < 0.01). Exact Poisson-binomial on carriers, each proband's rate
  from their estimated gnomAD ancestry mixture (`--ancestry auto`, one
  population, or an `ancestry` sheet column); males hemizygous on X/Y outside
  the PARs; BH q per model and tier. An allele far too common in the cohort for
  its gnomAD frequency is dropped and listed (`overrepresented_alleles`).
  `absent_recurrent` flags a gene resting on an allele missing from gnomAD that
  three or more probands carry: suspect calling, not biology. The regulatory
  tier needs `gnomad41_genome` and reports genomic-control `p_gc`/`q_gc`. Needs
  `paralome setup genome burden-reference <genome>` once per genome; without it
  the stage is *skipped*, not failed, and so is DNA functional_evidence. Evidome
  reads the four coding tests as `test=burden`.
  `paralome DNA --project P --extract-damaging-variants A,B` (or
  `--extract-all-variants`) writes one gene's alleles to
  `gene_burden/extracted/` and runs no pipeline.
- **Output.** `genotype_output` writes `<prefix>_DNA_genotype_output.zip` (the
  segregation table, gene burden and the filter's pass list as Parquet) for
  OmicView's Variants viewer; the candidate lists are columns of each variant's
  row (`report_lists` and the buckets), not files. `genome_tracks` writes
  `<prefix>_DNA_reads.zip`, the reads FreeBayes used at every called site, for
  each variant's panel in that viewer.

### ATAC and snATAC

Bulk ATAC counts no peaks on the mitochondrial genome. MACS3 calls peaks on
each BAM without its duplicate-flagged reads. snATAC reads peaks and counts
from the curated Signac object; a single over-wide peak is skipped and
counted, never fatal. Motif enrichment (Fisher's exact) runs per comparison of
every statistic folder, on peak sets cut at `--alpha`. `genome_tracks` writes
`<prefix>_<ATAC|snATAC>_genome_tracks.zip`: the viable peaks (the union of the
peaks any comparison tested) as bigBed, a bigBed per comparison (log2FC, p, q;
no call of significance), one bigWig profile per sample (bulk: per BAM;
snATAC: per pseudobulk sample) of fragment coverage inside the viable peaks
only, 25 bp bins, CPM, and the GTF's gene models. snATAC reads the fragment
files the Signac object names (PLM-0231 if one is missing); the UCSC tools
bedGraphToBigWig and bedToBigBed write the files (PLM-0416 if missing).
`--no-genome-track` leaves the stage out.

### scRNA and snATAC inputs

Both start from a fully curated Seurat (scRNA, including spatial/Visium) or
Signac (snATAC) object with its metadata attached; running the pipeline
command extracts it, and `paralome extract` is the alternative that extracts
once and writes the QC tables first. `--rds` takes one object (`*.rds`/`*.qs2`)
or a folder written with a trailing `/`: each object is extracted on its own
into `data/extraction/rds_members/` (on SLURM one job per object), barcodes are
prefixed `<file>_`, every object must carry the `--replicate-column` /
`--stratify-by` columns and the same genes (by Ensembl ID) or peaks, with no
file taken as the reference (PLM-0222 to -0226). With several gene-level
assays and no `--rna-assay`, extraction stops with PLM-0217.

Every extraction of an object writes
`<rds name>_cell_expression_embeddings.zip`: an .h5ad (X = log2(CPM + 1), every
cell, every gene with a count; obs = meta.data; obsm = each 2-D/3-D reduction
and the spatial x, y; the tissue images in `uns['spatial']` as scanpy reads
them) and a manifest, for plotting in OmicView's Cells viewer -- not for
re-analysis. Needs h5py (PLM-0415).

### scRNA trajectories

The `trajectories` stage fits the object's lineages on the whole object (every
condition together) and tests how they change with the run's design.
`--trajectory-root <column>:<value>` is required whenever a run reaches the
stage (PLM-0121) and is never guessed; `--no-trajectories` leaves the stage out,
and the same command run again with the root adds it.

- **The fit** runs as `singlecell_rna`'s last step, into `<run>/trajectories/fit/`.
  `--trajectory-method monocle3` (default: one principal graph learned in the
  object's UMAP, every root-to-leaf path a lineage; `fit/lineages.tsv` says how
  each leaf was derived) or `slingshot` (curves through the clusters' centres).
  `--trajectory-fit-cells` takes the method's settings as key=value pairs.
  `--trajectory-weld` (Monocle3, repeatable) joins tips of the learned graph,
  named as in `fit/tips.tsv`: `L39=L42` tip to tip, `L30=L33=L35` each to a new
  middle node, `L39~L42` L39's tip to the nearest node of L42's own branch,
  `'L39->L42'` L39's tip to L42's with L42's tip staying an end. Changing only
  the welds keeps the learned graph. `--trajectory-max-routes` (200) caps the
  routes welds may make.
- **The lineages the statistics follow**, `--trajectory-lineages`: `fate`
  (default; one per end state), `exit` (one per way out of the root's cluster
  and end state) or `route` (every route).
- **Expression along lineages.** Per lineage, each cell's log2(CPM+1) as the
  full factorial of the `--factor` columns crossed with a natural spline of
  pseudotime (df 3), every term Type II: each factorial term, `pseudotime`,
  and `<term>:pseudotime` (the factors change how the gene moves along the
  lineage). `--trajectory-model ls` (default; least squares, F) or `monocle3`
  (negative binomial, likelihood-ratio tests); `ls:df=4` sets the spline's df.
- **Composition.** Each lineage cut into `--trajectory-bins` (5) segments of
  equal cells by pseudotime; per replicate, its share of the lineage's cells in
  each segment and of its fitted cells on each branch, tested with the run's
  `--statistic` at the replicate level.
- **Output.** `<parent>/omicview/<run>/trajectories/<prefix>_scRNA_trajectory_output.zip`
  (fit/, lineages.tsv, expression/, composition/, manifest.json), and the
  lineages in the run's cell export (`pseudotime`, `pseudotime_<lineage>`,
  `trajectory_lineages` in obs; curves in `uns["paralome_trajectories"]`).
  snATAC has no trajectories stage.

### Proteomics and Olink

`paralome proteomics` reads an exporter's protein table (Proteome Discoverer,
MaxQuant, FragPipe, DIA-NN) and optionally the paired phosphopeptide table,
with a sample sheet (column, sample_id, metadata). Zeros are missing; log scale
is detected (PLM-0229); every sample is scaled to equal totals, then
log2(x + 1); a phospho table takes the protein table's factors only when the
two tables' loadings correlate. A missing value is set to half the feature's
minimum (`half_min`), the one way, with no switch. Import and QC are in
`<parent>/data/proteomics/`, results in `--project`.

With `--phospho`, three questions, each named in the files: `protein_change`
(each protein, own error), `phospho_change` (each phosphopeptide form,
unadjusted, own error) and `phospho_vs_protein_change` (per phosphopeptide
whose protein was measured: the measure x condition interaction of a 2x2
ANOVA, pooled error, log2FC = phospho change - protein change; filed under
`anova`; runs beside t or paired-t). When reporting, read the three together: a
phosphopeptide that changed while its protein did not is the phospho-specific
signal; one that moved with its protein may reflect protein abundance. Evidome
gets all three as blocks of one table.

`paralome olink --plate <export> [--plate ...] --samples <sheet>` reads Olink's
long plate export (by Olink's column names, PLM-0240). Value: QuantifiedValue
(pg/mL), log2(x + 1); BelowLOD or no concentration is missing, then half_min.
Every plate after the first is bridged to it per assay by the median ratio of
the shared SAMPLE_CONTROL wells (PLM-0241, `bridging.tsv`), but a question whose
samples are all on one plate is tested on that plate's own values. One sheet
for every question: PlateID, SampleID, sample_id, design columns; replicate
wells averaged. An assay named by no gene symbol (Abeta40/42, pTau217) is found
by its UniProt accession in the genome's registered UniProt table (APP, MAPT),
with `measured_form` kept beside it.

Both: `--block-by <column>` adds an untested term to anova and lm (repeated
measures; a factor constant within blocks is refused, PLM-0221).
`--statistic ...,panel` fits an elastic-net panel per comparison
(cross-validated AUC or R^2 against label-shuffling chance, the best single
assay, bootstrap selection frequencies), in `<stratum>/panel/` and the
statistics zip. Both may run inside a SLURM job; `--backend slurm` submits the
panels as jobs.

### Shared behaviour worth knowing

- `--stratify-by` takes several columns (every combination, folder `a__b`); a
  cell may list several values comma-separated (the sample is in each stratum,
  summed once); a blank is in no stratum. `--factor COLUMN:c1,c2` lists
  controls, each stratum taking the one it holds (none: skipped with reason;
  two: refused, PLM-0113).
- Fused stages (the default) stream fastp, alignment and duplicate marking per
  sample, on SLURM and locally; `--separate-stages` writes every intermediate.
  A fused alignment works out its own memory from the STAR index on disk.
- No SLURM job asks for more than three quarters of the largest node.
- On a shared machine Evidome writes into `<shared_root>/evidome/` during a
  run, so the account running a pipeline must be in the shared root's group;
  the preflight checks it (PLM-0117).
- A recorded tool whose path is gone is looked for on PATH in the preflight
  and re-recorded there; found nowhere, it is a PLM-0117 fault.
- A genome's folder carries the genome's name; `setup genome add` renames a
  folder called anything else and reports it.
- No Excel workbooks anywhere (the old workbook options are refused, PLM-0120).
  The ANOVA contrast note a workbook once carried is worth saying yourself: an
  ANOVA contrast is a model contrast with the model's pooled error, not a
  two-sample t-test.

**Every zip is named for what is inside it, and every zip goes in one folder,
`omicview/`, by analysis.** Bulk: `<project>/omicview/<stage>/`. Single-cell:
`<parent>/omicview/<run>/<stage>/`, `<parent>` being the folder holding `data/`
and `<run>` the run folder's path under it, mirrored; the object's cell export
sits in `<parent>/omicview/` itself. The stages' working files stay where they
were; only the zips move.

| Zip | Stage | Opens in OmicView's |
|---|---|---|
| `<prefix>_<modality>_statistics_output.zip` | `statistics_output` (RNA, scRNA, ATAC, snATAC, proteomics, Olink) | Results viewer |
| `<prefix>_DNA_genotype_output.zip` | `genotype_output` (DNA) | Variants viewer |
| `<prefix>_DNA_reads.zip` | `genome_tracks` (DNA) | Variants viewer |
| `<prefix>_<ATAC\|snATAC>_genome_tracks.zip` | `genome_tracks` | Genome viewer |
| `<prefix>_scRNA_trajectory_output.zip` | `trajectories` | (for an agent) |
| `<rds>_cell_expression_embeddings.zip` | the extraction | Cells viewer |
| `<prefix>_<modality>[_<stratum>]_<statistic>_evidome_output.zip` | `functional_evidence` | Evidence viewer |

No viewer page is written into any zip; the viewers are their own product,
OmicView, and read every zip on the user's computer (nothing uploaded).

Note the single-cell graphs have **no alignment or FASTQ stages**. They begin
from an already-processed Seurat/Signac object. Paralome reads the raw counts
layer (`--slot counts`) and normalises them itself: CPM per pseudobulk sample
after aggregation, and, for the `:cell` statistics, CPM per cell over every
extracted feature of that cell, then log2(CPM + `--pseudocount`). Seurat's
own normalised layers are not used.

---

## 5. On-disk contract

```
<parent>/
    data/                          shared across every comparison of one .rds
        extraction/                per-cell matrices read out of the object
            counts.plmsp           ATAC: peaks x cells (binary sparse)
            rna_counts.mtx         RNA: genes x cells
            cells.tsv              cell barcodes, column order of the matrices
            cell_metadata.tsv      every metadata column, unfiltered
            peaks.bed              ATAC only
            checkpoints/singlecell_extract.json
        rna/                       scRNA aggregation output
            raw_counts.csv         summed counts per sample x stratum group
            cpm.csv                counts per million, sample level
            log2_cpm.csv           what the statistics actually run on
            sample_manifest.csv    one row per pseudobulk sample
            cell_counts_per_sample.csv
            checkpoints/singlecell_rna.json
        atac/                      snATAC aggregation output, same shape
        qc/<timestamp>_<modality>/ QC tables and plots per run
    <PROJECT>/                     one contrast, e.g. treated_vs_control
        project_state.json         the pipeline's view of what is done
        checkpoints/pipeline_supervisor.json
        rna/                       scRNA differential results
            t/  anova/  t_cell/ ...  one folder per statistic (section 8)
        atac/                      snATAC differential results, the same folders
            motif_analysis/        snATAC motif results, results/<folder>/
        trajectories/              (scRNA) the lineages and the statistics along them
        statistics_output/         the statistics zip's working folder
        functional_evidence/       one Evidome run per statistic folder
    omicview/                      every zip, by analysis (section 4)
```

Bulk analyses keep their stages side by side in the project:
`differential_expression/results/<folder>/` (RNA),
`differential_accessibility/results/<folder>/` and
`motif_analysis/results/<folder>/` (ATAC), `functional_evidence/<folder>/`.

Every statistic folder holds the same files (`<stamp>` is `rna` or
`atac_peaks`; every file names its statistic, and so does Evidome's bundle):
`<prefix>_<folder>_evidome_input_<stamp>.csv` (the Evidome table);
`machine/<prefix>_<folder>_<comparison>_statistics_<stamp>.csv` with every
feature and column at full precision (`tested`, `log2_fold_change`, `p_value`,
`q_value` ..., then each sample's value as `s:<sample>`);
`machine/<prefix>_<folder>_sample_matrix_<stamp>.csv` (the PCA input); and
`machine/<prefix>_<folder>_manifest_<stamp>.json`. There is no human-readable
report, volcano, power file, significance column or workbook: OmicView draws
those from the statistics zip with the user's own thresholds.

**`data/` is a sibling of the project, not a child.** That is what allows a
second contrast to reuse the extraction. The default is
`<project-parent>/data`, so `--project scRNA/treated_vs_control` gives `scRNA/data`.
A run first looks upward from its `--project` for the
nearest `data/` already holding an extraction of the same `--rds`, so analyses
nested at any depth under one parent project share the parent's
`<parent>/data` -- which is where `paralome extract --project <parent>` puts it.

**Both the pipeline and the manual stage command accept `--project` and derive
the same folder.** If a user passes `--out` to the manual stage pointing
somewhere else, the pipeline will not find the extraction and will redo it,
which costs a full re-read of the object.

**Authoritative files:** the checkpoints decide what is complete.
`project_state.json` is a rendered view. Deleting a checkpoint forces that stage
to re-run; deleting `project_state.json` does not.

---

## 6. The stratified fan-out — the part users find surprising

`--stratify-by predicted.id` runs **one complete differential analysis per
stratum** (per cell type). Omit `--stratify-value` and it fans out over every
value present. Each stratum gets its own checkpoint, and its result files
carry the stratum in their prefix.

Consequences worth explaining to a user:

- A cohort with 14 cell types produces 14 result sets, not one.
- Strata are **skipped** when an arm has too few samples. The skip is recorded
  with a reason; it is not a failure.
- `--min-cells-per-group` (default 10) drops individual sample × stratum
  combinations during aggregation. A sample can therefore be present in some
  strata and absent from others. An arm member absent from one
  stratum is **dropped from that stratum with a printed note**, not an error —
  but an arm reduced to zero members still fails.
- With `--backend slurm` each stratum is its own array task (section 2);
  with `--backend local` they share the machine's threads.

---

## 7. Groups: `--factor COLUMN:CONTROL`

`--factor Diet:CD --factor Metformin:VEH`: the stated level is the control
state. A factor without a designated control is an error, because every test
is directional. So:

- the control level is named, never inferred from the order of the sheet's
  rows;
- every comparison is a level against its factor's control and nothing else:
  `--factor Day:1` gives `Day4_vs_Day1`, `Day9_vs_Day1`. B vs D is a second
  run with `--factor X:D`;
- levels may be words or numbers (Sex, ethnicity, dose, day). The outcome is
  always the feature's expression/accessibility/abundance; the factor only
  groups samples. A numeric factor additionally gets the one-way ANOVA's
  linear trend; a word factor does not, because its levels have no values;
- any number of `--factor`, any number of levels each.

`--replicate-column` names the sample-ID column `--remove` matches against.

**`--replicate-column` defines the pseudobulk samples.** It names the one cell-metadata column
whose value is a cell's biological sample; the samples are its values, one per
`--stratify-by` value when that is given. It is required on every single-cell
run, pseudobulk and `:cell` alike. There is no `--group-by`: naming it stops
with PLM-0119. A `--factor` column
is never part of the grouping -- it must be one value per sample (PLM-0203). For
a design with one library per factor combination, the library column is the
`--replicate-column`.

---

## 8. The statistics, and how they are asked for

**One option, one module.** `--statistic` is taken by every pipeline that
produces differential statistics (bulk RNA, bulk ATAC, snATAC, scRNA,
proteomics, Olink; DNA's gene burden is its own test), and every statistic is
computed by `paralome.stats`, routed by `paralome.engine` -- not a copy per
pipeline, so one column name never has two different tests behind it.

The QC statistics are one module as well, `paralome.qc_stats`: every PCA (exact centred SVD, deterministically signed), the
sample-sample correlation, average-linkage clustering, k-means and silhouette,
the robust outlier rule (median and 1.4826 x MAD, 3.5), the group ellipses'
covariance axes, the sample outlier QC, bulk RNA's differential audit and the
proteomics pairing test. Per-arm differences (thresholds, whether features are
standardised or centred, ellipse radius) are parameters, not copies.

The grammar is `<statistics>[:cell]`: any combination of `t`, `paired-t`, `lm`
and `anova`, comma-separated, each once; a colon before the data level.

| Request | What runs | Results folder |
|---|---|---|
| `t` | Student's t, df = n1 + n2 - 2 | `t/` |
| `paired-t` + `--pair-by <column>` | one-sample t on each pair's level - control difference, df = pairs - 1 | `paired_t/` |
| `lm` + `--covariate <numeric column>` or `--covariate-feature <feature>` | regression slope of each feature on the covariate | `lm/` |
| `anova` + one `--factor` of 3+ levels | the one-way ANOVA: 1 F-test (with its linear trend and lack of fit when the levels are numbers) + each level against the control | `anova/` |
| `anova` + two `--factor` | the two-way ANOVA with interaction: 3 F-tests + each factor's main-effect and within-level contrasts against its control | `anova/` |
| `anova` + three or more `--factor` | the factorial ANOVA, every interaction: an F-test per term (`A_x_B_x_C` ...) + each factor's main effect and its effect within every combination of every subset of the other factors (`<level>_vs_<control>_within_<levels>_contrast`); four two-level factors give 15 F-tests and 108 contrasts | `anova/` |
| any of these `:cell` | the same, with cells as the observations | `t_cell/`, `paired_t_cell/`, `lm_cell/`, `anova_cell/` |

`:cell` is the only suffix and belongs to the statistic it follows:
`anova,anova:cell` runs the ANOVA on the pseudobulk samples and on the cells,
in `anova/` and `anova_cell/`. Pseudobulk is the normal case and is what saying
nothing means; there is no term for it. The colon is for the single-cell
pipelines; bulk RNA and bulk ATAC refuse it. An option no listed statistic
reads (`--pair-by` without `paired-t`, a covariate without `lm`) is
refused, as are
`--covariate` and `--covariate-feature` together. Pseudobulk is a **data
action**, not a statistic, which is why it is not in the list of tests.

**Which comparisons, on which samples** (decided once by
`paralome.comparisons.plan_differential`, the same for every pipeline):

- `t`, `paired-t`: one `--factor` -- each level against the control,
  `<level>_vs_<control>` (numeric levels carry the factor's name:
  `Day4_vs_Day1`); two -- each factor's levels against its control within
  each level of the other (`MET_vs_VEH_within_CD`). Each group needs two
  samples (`t:cell` one: its N is cells, and the run records that the sample
  is confounded with its group). paired-t needs exactly one sample per level
  per `--pair-by` value and at least two pairs. A `t`, `paired-t` or `anova`
  with no `--factor` is refused.
- `lm`: every sample in scope (`<covariate>_slope`) and within each level of
  each `--factor` (not of a factor that is the covariate itself).
  `log2_fold_change` is the slope (log2 per unit of the covariate). At least
  three samples. Needs no factor.
- `anova` with one `--factor` (any level type): the
  one-way ANOVA of three or more levels (two levels: refused, use t). One
  F-test `<factor>_one_way`; for numeric levels it carries `trend_slope`,
  `trend_p`/`trend_q` (the straight line on the levels' values, the line `lm`
  fits) and `lack_of_fit_p`/`lack_of_fit_q`, its direction the trend's; for
  word levels it has no trend and no direction. And
  `<level>_vs_<control>_contrast` for each other level, with the model's
  pooled error. At pseudobulk it needs an error df.
- `anova` with two `--factor`: every sample in scope, any number of levels
  each, the interaction always in the model (it is what an ANOVA is used for;
  without interactions a t-test is). Treatment coding with each factor's
  named control as reference; Type II F-tests, each term's columns tested
  together (k levels = k - 1 df), matching statsmodels `anova_lm(typ=2)`.
  Three F-tests -- a two-level main effect named `<level>_vs_<control>`, a
  multi-level one `<factor>_F` (no direction), the interaction `<A>_x_<B>` --
  and, from the same fit, each factor's `<level>_vs_<control>_main_effect` and
  `<level>_vs_<control>_within_<other level>_contrast`. A 2x2 gives nine
  results; a 3x2 gives thirteen.
- The cell level: `t_cell` is t on the cells; `paired_t_cell` takes, per pair,
  the level sample's mean per-cell value minus the control-level sample's and
  tests those across pairs; `lm_cell` regresses with cells as N; the one-way
  `anova_cell` fits the cells as N (no random intercept; one sample per level
  computes and is recorded as confounded, like `t_cell`), and its
  `trend_slope` equals `lm_cell`'s slope on the same cells; the factorial
  `anova_cell` (two or more factors) is least squares on the cells, a
  sample's cells pooled with its replicates' (the two models for single-cell
  data are pseudobulk and cell), the same F-tests
  and contrasts as `anova` on df = cells - model columns; a combination held
  by one sample is recorded as confounded, like `t_cell`. Every cell statistic
  is vectorised and never subset; there is no random-intercept model.

**Each statistic is its own result.** Every statistic writes its own folder
with the same files and the same generic column names (`p_value`, `q_value`,
`log2_fold_change`); the folder says which statistic. A pseudobulk folder
reads exactly like a bulk one. Two statistics are never combined in one
table. Each is BH-corrected per comparison (each ANOVA F-test its own), never
across comparisons or statistics, and motif enrichment and Evidome run once
per folder. On single-cell runs `statistic_comparison.json` compares
statistics that made the same comparison (correlation of fold changes, rank
correlation of p-values; no call of significance).

**The expression floor.** A feature is tested in a comparison when its mean
CPM over that comparison's samples reaches `--min-mean-cpm` (default 1.0) --
every sample, for an ANOVA; judged on pseudobulk CPM at the cell level too.
Bulk RNA has no CPM floor; its `--rna-min-expression` /
`--rna-min-expression-difference` gate (raw FPKM, 5 and 5 by default) is the
`passes_expression_gate` column of each machine table for a two-arm
comparison -- every gene is still tested and in every table.

**If q-values are blank:** BH correction is applied over the features actually
tested; anything not tested has no p or q rather than a confident q = 1.0.
Blank means "never tested", which is different from "tested and
unremarkable". This distinction is deliberate.

**No call of significance anywhere.** Paralome writes p and q and no
`significant` column; the user sets thresholds in OmicView. `--alpha` is only
the cut a next stage selects by (motif peak sets, the phospho pairing test,
Evidome's gene selection).

---

## 9. Statistical caution to pass on

Two things a user's assistant should raise if relevant, because they are easy to
get wrong and Paralome will not stop you:

**Circularity.** If subgroups were defined from a PCA of the same matrix, then
testing those subgroups against each other on that matrix is circular — the
features driving the separation come back significant by construction. On a
pure-noise simulation (9 × 10,000) this produced 16 features at R² > 0.8. This
is a methods-section caveat, not a bug.

**`--alpha P0.05` is not `--alpha Q0.05`.** The spec is a metric letter plus a
cutoff. `P` (the default, `P0.05`) is the unadjusted p-value; `Q` is
FDR-adjusted. It changes which genes Evidome selects and which peaks motif
enrichment tests, so the difference matters.

---

## 10. Errors: what they mean and what to do

| Message | Cause | Fix |
|---|---|---|
| `No such command 'X'` | Names are case-sensitive; or the command moved | Check spelling exactly. `pipeline run` no longer exists — use `paralome <ANALYSIS>` |
| `Unknown pipeline profile 'snRNA'` | `snRNA` was never valid | Use `scRNA` |
| `No such option: --control` (or `--treatment`, `--condition-column`) | Not an option: groups are `--factor` columns | `--factor COLUMN:CONTROL`, e.g. `--factor condition:CON` |
| `PLM-0113 --factor is malformed` | A `--factor` without `:CONTROL`, or a control level the column does not hold | Write `--factor Diet:CD`; the message lists the column's levels |
| `PLM-0221 ... compares the levels of a --factor` | `t`, `paired-t` or `anova` asked for with no `--factor` | Add `--factor COLUMN:CONTROL` |
| `--rds is required for scRNA profiles` | Single-cell analyses take an object, not a manifest | Pass `--rds`, not `--samples` |
| `--replicate-column is required` | Aggregation needs to know what a sample is | e.g. `--replicate-column donor` (and `--stratify-by cell_type` to split by cell type) |
| `PLM-0119 --group-by is retired` | Not an option | `--replicate-column <sample column>`; a second grouping is `--stratify-by` |
| `PLM-0230 --stratify-by produced no testable strata` | Every stratum was skipped for thin groups | Check the factor's levels per stratum and `--min-cells-per-group` |
| `Sample manifest is missing n_cells` | Manifest is not a Paralome pseudobulk manifest | Point at `data/<modality>/sample_manifest.csv` |
| `SLURM stage failed: <id>:FAILED:1:0` | The stage's own job died | **Read the job's `.err` file** — the supervisor only reports that it failed |
| `--alpha must look like Q0.05 or P0.10` | Missing the metric letter | `--alpha P0.05`, not `--alpha 0.05` |
| `PLM-0101 ... --genome is required` | There is no default genome: not the only one configured, not `defaults.genome`, not GRCh38 | Add `--genome <name>`; the message lists the configured names |
| `PLM-0103 ... --submit is for the SLURM backend` | The backend is local (`--backend local`, or `defaults.backend` in the configuration). Every run prints its backend and what chose it | `--backend slurm` to submit, or drop `--submit` to run here |
| `PLM-0308 A stage that has to run again would keep its old result` | An option, the sheet or an input changed on a finished project, for a stage that does not recompute by itself | Remove what the message names and run the same command; or put the option back |
| `PLM-0309 This stage was started with a different request` | Read QC or alignment resumed with other options or inputs than its finished samples were made with | Repeat the earlier options, or remove the stage folder to redo every sample |
| `PLM-0310 Another supervisor is already running on this project` | The same command is still running elsewhere (nohup, a lost terminal) | Wait for it or stop that process; there is normally no lock to clear |
| `PLM-0311 Jobs from an earlier submission may still be running` | The stage was submitted before from this folder | Wait for those jobs or `scancel` them, then run again |
| `PLM-0411 A configuration file has a key that nothing reads` | A mistyped or retired key in `~/.paralome/config.json`, a `--pipeline-config` file or `site.json`. It stops every run until fixed | Correct or remove the key; `paralome setup doctor` reports it. Invalid JSON is PLM-0412 |
| `<file> is an RData workspace saved by save() (magic 'RDX3')` | `save(object, ...)` was used instead of `saveRDS(object, ...)`. Same size, same extension; R itself only says `unknown input format` | Re-save it: `e <- new.env(); nm <- load(f, envir=e); saveRDS(e[[nm[1]]], f2)` |
| `<file> has no atac assay, which this command requires. Assays found: RNA, integrated` | An RNA-only object given to `snATAC` (or vice versa) | Use the matching analysis group, or pass `--assay` if the assay is present under an unusual name |
| `Found neither an ATAC nor an RNA assay to extract` | Nothing in the object is recognisable as either | Name them: `--assay` and/or `--rna-assay` |
| `PLM-0216 A named assay is not in the object` | `--assay`/`--rna-assay` named one that isn't there | Usually: drop the flag and let detection run |
| `PLM-0217 More than one assay of the same kind` | e.g. RNA + ADT, both gene-level | Name the one you want |
| `<dir> is not writable ... owner / mode ... running as ...` | The project tree was created by another user (commonly root) | `chown -R` it to the running user, as the message spells out |
| `PLM-0206 The samples in scope matched no cells` | The manifest and the extraction are not from the same object | Use the `sample_manifest.csv` written from this extraction |

**When a SLURM stage fails, the supervisor's traceback is not the error.** It
reports that a job failed. The actual cause is in the job's own `.err` file, and PLM-0301 prints its
path. For bulk stages it is under the stage's folder in the project; for
single-cell stages it is under the shared `data/` folder beside the projects,
not inside the project. Ask the user for the path the message printed rather
than guessing one.

Always ask the user for that file before speculating.

---

## 11. Things Paralome deliberately refuses

An assistant trained on general bioinformatics will suggest several of these.
Do not.

- **No edgeR, no DESeq2.** Not anywhere, not as an option. What `--statistic`
  offers, plus CPM, is the entire statistical surface for differential testing.
- **No deprecated options or aliases, ever.** Old spellings are removed, not
  kept working. If something used to be called X and is now Y, X fails loudly.
- **No normalisation of single-cell input.** Data arrives processed and
  normalised. Paralome computes CPM at the sample level after aggregation; that
  is the only normalisation it owns.
- **No `run`, `continue`, or `--profile` on the analyses.** The analysis
  name is the command. `status --profile` narrows a survey of finished work,
  and `batch run` takes it to choose the analysis for every family; neither
  overrides a command name.
- **No lower-case analysis names.** `scrna` is not `scRNA`.
- **No `--gtf` on the analyses.** `--genome` resolves it: a hand-typed GTF
  could disagree with the genome the run recorded, with nothing to say which
  one was used. The standalone stage commands on the human binary
  (`scRNA samples`, `snATAC samples`, `RNA strandedness`) take `--gtf` for the
  file they work on directly.
- **No default genome.** `--genome` is required on every analysis and on
  `batch`. Nothing is chosen for the user, and there is no built-in build:
  the ANNOVAR build, protocol list and operation list come from the options,
  the configuration (the two lists only) or the genome entry, both lists from
  the same place.
- **No results replaced without being asked.** The supervisor adds no
  `--overwrite`. Typed on read QC or alignment, `--overwrite` deletes nothing
  at submission: it marks each sample to be replaced, and each task replaces
  its own.
- **No ignored configuration keys.** An unknown key is refused (PLM-0411).
- **No `--mem` on `batch`.** Per-stage memory goes through
  `--pipeline-args '--stage-mem alignment=40G'`.
- **No `--name` when registering a genome.** The name is derived from the
  genome as `<Species>.<Assembly>.<source-initial><release>`.
- **No separate checker command.** `--dry-run` prints the
  dispatch plan and every run preflights whether asked or not.
- **No utilities inside a SLURM job.** Only the analyses (olink and proteomics
  included), `batch` and `extract` are submittable; anything else stops with
  PLM-0116.

---

## 12. Useful diagnostic commands

```
paralome --version                     which build is installed
paralome <ASSAY> --help                what that analysis takes, grouped by stage
paralome status                        every project below here, one line each
paralome status --profile scRNA        only the scRNA ones (case-sensitive)
paralome status --project P            one project, every stage
paralome status --project P --detail   ...with outputs, metrics and paths
paralome <ASSAY> ... --dry-run         the plan plus a full preflight, submits nothing
paralome setup doctor                  environment and dependency check
paralome setup genome                  genomes registered on this machine
paralome extract --rds obj.rds --project P --genome G   extraction + qc/ in P/data
paralome explain 117                   what a numbered error means
```

**`paralome <ASSAY> --help` is the first thing to reach for.** Each analysis
lists its own options, under one heading per stage it runs.

`paralome extract` deliberately takes **no** grouping flags. Choosing
`--replicate-column` / `--factor` / `--stratify-by` is what the QC tables it
writes are *for*; requiring them up front would invert the workflow. It
requires `--genome` (it attaches an Ensembl ID to every gene, once). Its
`--project` is the parent project: the extraction goes in `<project>/data`, and
every analysis whose `--project` is inside that folder, at any depth, reuses
it. Then read `<project>/data/qc/metadata_values.csv`
(`column`, `value`, `n_cells`, `pct_of_cells`): it holds the exact spelling
of every value a `--factor` control level must match, and
`metadata_columns.csv` says which columns are numeric (a `--covariate`) or
groupable. A `--factor`, `--pair-by` or `--covariate` column must be one value
within every `--replicate-column` sample: aggregation carries only such columns into
the pseudobulk `sample_manifest.csv`.

---

## 13. Known limitations — do not present these as user error

- Different groupings pointed at one shared data folder **overwrite each
  other's** aggregation outputs. Use a separate data folder per grouping.
- snATAC motif **pairs** have never been run; only `--stage singles` has been
  exercised.

---

## 14. How to answer well

1. **Ask for the `.err` file** before diagnosing a SLURM failure. The
   supervisor traceback names the job, not the cause.
2. **Ask for `paralome --version`, and on a cluster ask for it via `srun`.**
   This document describes the version stated at its top, and the software
   breaks compatibility between releases. `/usr/local` may be node-local, in
   which case the login node and the compute nodes can be running different
   versions.
3. **Ask what is on disk.** `ls` of the data folder and the project usually
   answers "why did it re-run" immediately.
4. **Prefer re-running the same command** over inventing recovery commands.
   Checkpoints make it safe and it is the designed path. If the user changed
   an option since the project finished, say what to expect: the RNA and
   aggregation stages recompute, and any other stage is named by PLM-0308.
5. **Say when you do not know.** Paralome is a beta and has surfaces this
   document does not cover. A confident wrong answer about which samples entered a test is worse than no
   answer.
