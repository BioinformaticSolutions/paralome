# Paralome — user manual

**Version this describes: 0.9 (beta).** Paralome changes between releases
without deprecation shims. If `paralome --version` disagrees with the number
above, treat this document as suspect.

---

## 0. Which version am I running?

There is one package, `paralome`, and its version number says where it stands:

| | version numbers | what it is |
|---|---|---|
| **release** | `0.9`, `0.9.1`, `1.0` | stable; does not move under you |
| **development** | `0.9.1.dev1`, `0.9.1.dev2`, … | the next release, in progress |

A `.dev` number is a pre-release **of the version to its left**: `0.9.1.dev1`
comes after `0.9` and before `0.9.1`. Promotion is dropping the suffix.
`paralome --version` prints the release's standing beside its number:
`0.9 (beta)`.

**Installing.** `INSTALLATION.md` is the authoritative procedure, for
the machine and for each account. The short form, for the machine:

```
sudo python3 -m pip install paralome
```

Python 3.10 or newer. No `--no-deps`: the wheel declares what it
needs (typer, rich, numpy and the rest), and pip installs them with it. `--force-reinstall` is for replacing a
copy that is already there (see PLM-0405); it is not part of a first install.
Do not install with `--user` (PLM-0408).

**Several versions on one machine.** Install each version into a prefix of its
own and reach it through a module (or a `PATH` entry) for that version:

```
sudo python3.14 -m pip install --ignore-installed --prefix=/opt/paralome/<version> paralome-<version>-py3-none-any.whl
```

`--ignore-installed` puts every dependency into the prefix. Without it pip
leaves out what that interpreter can already import from elsewhere, and the
module, which puts only its own prefix on the path, then stops at the first
import (`No module named 'typer'`).

DuckDB is held below 2: DuckDB 2.0.0's build stops with
`Illegal instruction` on a CPU with AVX but not AVX2, which stops the output
stages there. `grep -o -w avx2 /proc/cpuinfo` says
whether a machine has it.

Run pip with the interpreter the version is to use -- any `python3.X` of 3.10
or newer; on RHEL 9 plain `python3` is 3.9 and is refused. That
interpreter is written into the first line of `<prefix>/bin/paralome`, so the
version runs on it whatever `python3` means in the user's shell.

The version's module puts `<prefix>/bin` first on `PATH`, and both
`<prefix>/lib/python3.X/site-packages` and `<prefix>/lib64/python3.X/site-packages`
on `PYTHONPATH` (`3.X` the interpreter above). pip installs the dependencies
into the same prefix, so each version carries its own; on RHEL and its
relatives the compiled ones (numpy, pandas, matplotlib, pysam, duckdb, h5py,
MOODS) go to `lib64/`, and a module without that line stops at the first
import with `No module named 'matplotlib'`. Switching is `module load` of
another version.

```
#%Module1.0
set prefix /opt/paralome/0.9
prepend-path PATH       $prefix/bin
prepend-path PYTHONPATH $prefix/lib/python3.14/site-packages
prepend-path PYTHONPATH $prefix/lib64/python3.14/site-packages
```

A modulefile copied from an older version must have its prefix and Python
version changed, not only its name: `paralome --version` names the prefix the
code was imported from and the interpreter running it, so check it after
every new module.

Which version a bare `module load paralome` gives is the default, set in the
modulefile folder's `.version` and moved by hand once a new version has been
checked -- not automatically to the newest:

```
sudo tee <modulefiles>/paralome/.version >/dev/null <<'EOF'
#%Module1.0
set ModulesVersion "<version>"
EOF
module avail -d paralome          # the default
```

Never install two versions into the **same** site-packages. They are the same
package, so installing the second silently removes the first — pip reports
success, because that is what it was asked to do:

```
Found existing installation: paralome 0.1
  Uninstalling paralome-0.1:
    Successfully uninstalled paralome-0.1
```

**They share one configuration.** `~/.paralome/config.json` is the same for
every version, so a genome registered once is visible to all of them.

**A run stays inside its own install.** When a run submits its stages to
SLURM, it writes the path of the `paralome-worker` beside the `paralome` that
started it into every generated script, so every job runs the version that
submitted it, whatever else is on `PATH`. Submit with the version's module
loaded: SLURM passes the environment, `PYTHONPATH` included, to the jobs.

`paralome --version` answers the question directly, and it is worth asking
before you interpret any result:

```
$ paralome --version
Paralome version 0.9 (beta)
  code imported from /opt/paralome/0.9/lib/python3.14/site-packages/paralome
  running /usr/bin/python3.14

Copyright (C) 2026 Kenneth L. Jones
Licensed under the GNU Affero General Public License, version 3.
This program comes with ABSOLUTELY NO WARRANTY.
Full terms: <https://www.gnu.org/licenses/agpl-3.0.html>
A separate commercial licence is available; see NOTICE.
```

---

## 1. The one thing to know first

Paralome is **checkpointed**. Every stage writes a checkpoint when it finishes,
and a run starts at the first stage whose checkpoint is missing or invalid.

**Re-running the same command is how you resume.** There is no `resume`, no
`continue`, no `--restart-from`. If a stage fails, fix the cause and re-issue
the identical command; the completed stages are skipped.

The identical command is the point. A finished stage is kept only while it
still answers what the command asks for. What happens when an option, the
sample sheet or an input file has changed since the stage finished is in
section 3, under "When an option is changed on a finished project".

---

## 2. Running an analysis

The analysis name is the verb. There is no `run` subcommand.

```
paralome scRNA  --rds obj.rds --project ko_vs_wt --replicate-column sample \
                --statistic t --factor genotype:wt \
                --genome Mus_musculus.GRCm39.e112 --backend slurm --submit
```

**The groups are columns, and each names its control level**.
`--factor genotype:wt` says: the samples are grouped by their `genotype`
value, and every other value is compared with `wt`. The column can hold
words (`Sex:F`, `Diet:CD`) or numbers (`Day:1`); every pipeline -- RNA, ATAC,
snATAC, scRNA and proteomics -- reads it the same way and fits the same
models. There is no `condition` column, no `--control`, no `--treatment` and
no `--condition-column`; a column named `condition` is just a column, and
`--factor condition:CON` uses it.

The control level is required. A `--factor` without one is refused
(**PLM-0113**): every comparison is a direction, level minus control, and
nothing about it may depend on the order the rows of a sheet happen to be in.
Each level is compared with the control and only with it: `--factor Day:1`
gives `Day4_vs_Day1` and `Day9_vs_Day1`. To compare two of the other levels,
run again with one of them as the control. Two or more `--factor` options
make a factorial design (any number of levels each, every interaction in the
model). There is no limit on the number of factors: the samples set
it, since every combination of levels needs a sample.

All of this is checked before anything is queued: a `--factor` without its
control, a `--statistic t`, `paired-t` or `anova` with no `--factor` (unless
`--stop-at` ends the run before its differential), and, on a bulk sheet, a
column the sheet lacks or a control level its active samples do not hold.

You do not have to remember that genome name. `paralome setup genome` prints
every name registered here, and the names are derived from the genome itself
rather than typed by whoever registered it, so the one you see is the one to
paste.

**`--genome` is required, and there is no default.** Every analysis refuses to
start without it (**PLM-0101**) and lists the genomes configured here. Paralome
does not fall back to the only genome you have, to `defaults.genome` in your
configuration, or to human GRCh38. A `defaults.genome` in the configuration is
not an error and is not read; the refusal says so. A name that is not configured is refused with
**PLM-0401**. The one run that needs no `--genome` is a DNA run given every
path as an option (`--star-genome`, `--reference` and the annotation options);
the refusal lists the ones still missing.

| Name | Data |
|---|---|
| `DNA` | Variant calling |
| `RNA` | Bulk RNA-seq |
| `ATAC` | Bulk ATAC-seq |
| `snATAC` | Single-nuclei ATAC |
| `scRNA` | Single-cell RNA (cells or nuclei, including spatial) |
| `proteomics` | Proteomics, with paired phosphoproteomics |
| `olink` | Olink panels |

**Every pipeline ends with Evidome** (the `functional_evidence` stage), run on
that pipeline's own differential results -- the gene-burden table for DNA;
for the others, once per statistic folder the differential wrote (section 7).

Evidome skips itself, and says so, when it or its data for your species is not
installed (its error codes EVD-0013, -0014, -0019 and -0025); installing it later makes the stage run on the next command rather
than leaving it finished-by-omission. `--skip-evidence` is the deliberate off
switch, for when it *is* installed and you do not want it -- the stage then
leaves the graph for that run, and the project still reads complete.

On a shared machine Evidome's reference data live in `<shared_root>/evidome/`,
which belongs to the shared root's group. The first run that needs a
registered genome's index, or a species not yet downloaded, writes it there,
so the account running the pipeline must be in that group. Every run checks
that first, with its other missing pieces, and an account
outside the group of the shared genome repository (`<shared_root>/genomes`,
which the Evidome reference shares) stops there (PLM-0117), before anything is
queued; `paralome setup doctor` reports it under Shared setup and does not call
the machine ready. An administrator adds the account (`usermod -aG <group>
<account>`), and the account logs in again.

**Trimming, alignment and duplicate marking run as one task per sample by
default**, streaming between them: the trimmed FASTQs and the
aligned BAM are never written, and fastp's QC reports are kept. The
intermediates a staged run writes get deleted by hand afterwards anyway, and
on a 40-genome run they multiplied 2.2 TB of input several times over. A task
that dies re-trims that one sample, which costs less than carrying the files.

**On the local backend too**. The task is the same script a SLURM
array runs, run here once per sample, as many at once as `--threads` allows.

`--separate-stages` writes every intermediate instead. Use it when the QC or
the mapping is itself what you need to look at.

**`paralome <NAME> --help` lists what that analysis takes,** grouped by what
each option is for — its inputs, how it is run, and then one heading per stage
it actually runs. Each analysis shows only its own: `paralome DNA --help` has a
Variant annotation heading and no single-cell sampling; `paralome scRNA --help`
has neither ANNOVAR nor STAR, because scRNA runs neither.

Names are **case-sensitive and have no aliases**. `paralome scrna` fails rather
than quietly resolving — a silent alias is how two spellings of one thing become
established. `scATAC` does not exist, because ATAC is a nuclei-only prep;
`snRNA` is not valid either.

### The menu has two halves

`paralome --help` separates them, because most of the list is not a pipeline:

**Pipelines** — `ATAC`, `DNA`, `olink`, `proteomics`, `RNA`, `scRNA`,
`snATAC`. Choosing what to do means choosing one of these.

**Utilities** — `batch`, `collect`, `explain`, `extract`, `setup`, `status`.
Everything that supports a run without being one. `batch` is here rather than
above: it runs pipelines, it is not one.

Only the pipelines, `batch` and `extract` can be submitted to SLURM. A utility run
inside a job stops with **PLM-0116** rather than doing configuration work on a
compute node where nobody will see the output.

`batch run` takes `--genome` like the analyses it starts, and refuses a family
whose command names none. It has no `--mem`: memory is set per stage, and
reaches the families through `--pipeline-args`, as in
`--pipeline-args '--stage-mem alignment=40G'`. (`--supervisor-mem` is the small
job that watches each family, not the analysis.)

### Which backend a run uses, and `--submit`

Every analysis prints its backend and where that came from:

```
Backend             : slurm (defaults.backend in /home/you/.paralome/config.json)
```

The order is `--backend`, then `resources.backend` in a `--pipeline-config`
file or in your configuration, then `defaults.backend` in your configuration,
then `slurm`. `defaults.backend` applies to the pipelines. A stage command
run by itself (`paralome scRNA differential`, for one) keeps its own
`--backend` default, which is `local`.

`--submit` is for the SLURM backend. With a local backend it is refused
(**PLM-0103**), and the message says what made the backend local: `setup`
records `local` on a machine where `sbatch` is not on PATH, and a run from
that account with `--submit` would otherwise trim and align on the login node.
Pass `--backend slurm` to submit, or drop `--submit` to run here.

The SLURM backend without `--submit` writes the stage's scripts and submits
nothing. The run says so, names the folder, and stops there; the same command
with `--submit` submits them.

### Memory for one stage

`--stage-mem alignment=40G` sets the memory of one stage and may be repeated.
A value typed on an analysis reaches that stage. The stage names are the ones `paralome status --project`
prints, and a name that is not a stage is refused.

Alignment needs no `--stage-mem` when its stages are fused. A task
that marks duplicates on STAR's output (`--fuse-stages` for ATAC, ChIP and DNA,
and marking in place for RNA) asks for the memory its pieces hold at once:

- the genome's STAR index as it is on disk (`Genome`, `SA`, `SAindex`);
- STAR's working memory, 2G and 256M a thread;
- the samtools name sort, at samtools' own 768M a thread, plus 15%;
- the read buffers, `--fused-buffer` each (two for paired reads), and 2G for
  fastp when trimming is fused too.

After STAR exits, the name sort hands its reads to the position sort and
markdup; the request covers whichever of the two phases is larger, adds 15%,
rounds up to whole gigabytes and is never under 32G. The alignment summary
(`results/star_alignment_summary.json`, `fused_memory_plan`) lists each piece.
An unfused alignment still asks for 32G. `--stage-mem alignment=` replaces the
calculated amount; the fused buffers are then added to it.

When trimming is fused, fastp writes the trimmed reads into FIFOs on the node's
scratch and STAR reads them. A fastp that ends for any reason releases its
FIFOs, so STAR ends on the reads it was given and never waits on a FIFO nobody
will write; the task fails with `ERROR: fastp failed for <sample> (exit <n>)`,
and the signal when one ended it, in its `.err` log.

### The configuration is checked, key by key

A key that nothing reads is refused, with **PLM-0411**, in all three places a
configuration can come from: `~/.paralome/config.json`, a `--pipeline-config`
file, and the machine's `site.json`. The message names the file, the key, the
nearest valid spelling and the keys that are valid there. A mistyped key --
`stage_resources.alignmnt`, `defaults.thraeds` -- is never loaded and ignored,
which would run on the default it was written to replace.

Because every command reads the configuration, one unknown key stops every
run until it is corrected or removed. `paralome setup doctor` reports the key
without running anything. A file that is not valid JSON is **PLM-0412**;
`python3 -m json.tool FILE` shows where it breaks.

`annotation.annovar_buildver` and `annotation.snpeff_genome` are refused the
same way: a build belongs to a genome, so it is recorded in the genome entry.

### DNA: unrelated probands, and variants outside genes

A run of independent affected people with no relatives is one project, each
person their own family (`family_id` = `sample_id`, `relationship` = proband,
`affected` = affected). Filter it with `--segregation-profile unrelated-probands`:
every rare variant (gnomAD <= 1%) that any proband carries, coding or not,
ClinVar benign excluded. The default `family-segregating` needs a variant in
half the affected samples and would keep almost nothing.

On genome data, calling is scheduled from the genome's GTF genes plus
`--gene-padding` bases on each side (default 100). Pad wide -- thousands of
bases -- to call promoter and other flanking regulatory variants;
`unrelated-probands` has no coding requirement, so they are reported. A
`--regions` file (one `chr:start-end` per line) is used as given instead.

```
paralome DNA --samples probands.csv --project Probands_DNA --genome <name> \
             --gene-padding 5000 --segregation-profile unrelated-probands \
             --backend slurm --submit
```

### The sample sheet decides

Every stage that pools samples takes them from the sheet as it is now: a
sample `#`-commented out is left out of FreeBayes joint calling, the ATAC peak
merge, peak quantification and the differential, and a group corrected in the
sheet reaches the ATAC differential. The sample id column may be called
`sample_id`, `sample`, `id` or `name`, in any case. Pointing a sample at
different FASTQs or a different BAM is noticed: the run stops and names the
sample and the files. A sheet that gives some samples as BAMs and others as
FASTQs is refused; a project takes one or the other.

A relabelled group -- a sample's value in a `--factor` column changed in the
sheet -- makes a finished ATAC differential stale, and it is tested again.
Joint calling, the peak merge and peak quantification are run again when the
set of active samples changes (not for a relabel, which they do not use).
Bulk ATAC does not count peaks on the mitochondrial genome (`chrM`, `MT`), so
they are not part of any sample's CPM library size. MACS3 calls peaks on each
BAM without the reads duplicate marking flagged (`samtools view -F 1024`, a
temporary BAM removed afterwards), as peak quantification counts them, so PCR
duplicates are not piled up. Each sheet sample must be a
sample of the VCF: a BAM names its VCF sample by its read-group `SM` tag, and
segregation stops, naming the samples, when the two disagree.

### DNA: families -- how a variant is classed

Segregation scores one family per project. Each variant gets a class:
`affected_only`, `shared_affected` and `possible_de_novo` (the proband
carries it and both parents are called reference) pass; `possible_recessive`
(the proband and every affected relative homozygous, no
unaffected relative homozygous, carrier parents allowed) passes too;
`affected_enriched` is partial; `non_segregating` fails. The default
`family-segregating` profile keeps all the passing classes.

A sheet holding several families that each have more than one sample is
scored as unrelated probands: one affected sample per family, its proband where one is
named, relatives left out, and the console says so. For family segregation,
run each family as its own project (`paralome batch run` does this).

The filter profiles' thresholds are applied in three kinds (per
genotype); `filter_summary.json` says what each was tested on:

- **Site** (`min_site_depth`, `min_site_alt_reads`, `max_missing_rate`), per
  variant, on the segregation table's `site_depth`, `site_alt_reads` and
  `missing_rate`. A row whose column is `.` (the VCF gives no INFO DP or AO)
  is not failed on that threshold, and `quality_not_tested` counts such rows.
- **Genotype** (`min_dp`, `min_gq`, `min_alt_reads`, `min_ref_reads`,
  `min_alt_fraction_het`, `min_alt_fraction_hom`), per genotype, on the
  table's `genotype_evidence`: at least `family.min_affected_carriers` (1 when
  unset) of the affected carriers -- every carrier when the sheet names no
  affected sample -- must pass them; a number the VCF does not give fails. In
  a pedigree, every called relative not carrying the ALT must pass `min_dp`
  and `min_gq`. GQ is the VCF's, or worked out from PL/GL where the caller
  wrote none.
- **Family** (`min_total_carriers`, `min_sequenced_samples`,
  `require_proband`), per variant, on the table's carrier and called-sample
  counts.

There is no sheet-wide `min_sample_dp` or `min_sample_gq`: the worst sample of
the whole sheet would let one unrelated person's weak reference call fail every
carrier's variant. A profile that sets a
threshold no test reads, or a tumor/normal allele fraction (Paralome has no
such sample roles), is refused by name; so is a family profile on a table
scored as pooled probands. A table written by an older segregation is refused
by the filter: run segregation again. A profile's allowed impacts read
ANNOVAR's consequence words when snpEff did not run (as the report does).

### DNA: where the ANNOVAR build and tables come from

Paralome has no genome build and no protocol list of its own, so a genome is
never annotated as `hg38` against the human tables by default.

- **The build** (`annovar_buildver`, and `snpeff_genome` for snpEff) comes from
  `--annovar-buildver` / `--snpeff-genome`, or from the genome entry. Not from
  a configuration file.
- **The protocol and operation lists** come from `--annovar-protocol` and
  `--annovar-operation`, then `annotation.annovar_protocol` and
  `annotation.annovar_operation` in a configuration file, then the genome
  entry.

ANNOVAR pairs the two lists entry by entry, so both are taken from one place:
one typed without the other, one in a configuration file without the other, or
lists of different lengths are refused before anything runs. The run prints
each value with where it came from. A value that nothing supplies stops the
run at the preflight (**PLM-0117**), which names the key to record in the
genome entry and the option that gives it for one run. The standalone
`annotate` stage follows the same rule, with the options and the genome entry
as its two places.

### DNA: gene burden against gnomAD

A cohort of affected people with no unaffected relatives has
no controls of its own, so the DNA arm compares them with gnomAD, gene by gene.
The `gene_burden` stage runs after filtering, and its result goes on
to Evidome (`functional_evidence`), as the RNA arm's differential tables do.

**Once per genome**, build the gnomAD side. It annotates every gnomAD v4.1 exome
allele with ANNOVAR and adds up, per gene, how common the qualifying alleles
are in each population:

```
paralome setup genome burden-reference <genome> --backend slurm --submit
```

It is cut into chunks: a SLURM array that runs 48 chunks at a time
(`--max-parallel N` to change it, 0 for no limit), at a low priority
(`--nice=100`) so a project's own jobs go first, with 6G a chunk (`--mem`). Run again after a failure, it computes only the
chunks that have no result. An existing reference is never replaced without
`--rebuild`. The result is recorded in the genome entry as `burden_reference`,
and `setup doctor` reports it. A genome without one still runs: `gene_burden`
is skipped with the reason and the command that builds it.

Genes are keyed by Ensembl gene id. ANNOVAR's hg38 `ensGene` names most genes
by symbol (`TTN`) and only the unnamed ones by id; a symbol is
mapped to its id through the genome's GTF (by the transcripts ANNOVAR lists
for it, else the GTF's `gene_name`), and a symbol that would map to two genes
is left out and counted in `burden_reference.json`. A reference built without
this mapping is refused by `gene_burden` and must be built again with
`--rebuild`.

**Whole genomes: add gnomAD genomes first.** gnomAD's exome table holds
almost none of a genome's intronic, intergenic and regulatory variants, so on
a whole-genome run most variants have no population frequency and the
rare-variant filter cannot tell common from rare outside exons. Once per
genome:

```
paralome setup genome annovar-table <genome> gnomad41_genome
```

This downloads the table with ANNOVAR's own downloader (a large download: run
it on the login node under `nohup` or `screen`, or with `--backend slurm
--submit` where compute nodes reach the internet). It checks the table and
its index arrived, then adds the table to the genome's ANNOVAR protocol, so
every later annotation of that genome queries it. A table already there is
not downloaded again. Then build the burden reference, with `--rebuild` if it
was built before, so it has the regulatory table too. A project annotated
before keeps its annotation. Move its annotation folder aside, with the
segregation, filtering and report folders after it, to annotate it again.

**Complex calls are split before annotation**. FreeBayes writes a
change it sees inside a short haplotype with the bases around it: `CCT>CCC` at
one position is really `T>C` two bases on. gnomAD lists the single change, so
ANNOVAR finds no frequency for the FreeBayes form, and the allele would count
as absent from gnomAD and so as rare. Every annotation therefore first puts each call into gnomAD's form with `bcftools norm` (1.13 or later),
left-aligned against the genome's FASTA: multi-allelic records are split into
one record per allele, and an MNP (REF and ALT the same length, `CCT>CCC`) is
atomized into its single changes. **A complex allele whose pieces overlap**
(`CA>T`, `AC>TCC`; REF and ALT of different lengths) **reaches ANNOVAR whole**
(atomizing those makes bcftools give every carrier the `*` overlap
allele, and the real pieces come out 0/0). gnomAD may list such a change in
another form, so a complex allele can lack a gnomAD frequency and be counted
as absent from gnomAD -- `absent_recurrent` in gene burden -- when the
representation, not the population, is what missed. Weigh a complex indel's
burden flag with that in mind. Pieces of one called allele keep a tag
(`PARALOME_ORIGIN`), so gene burden counts them as one event, never as a
possible compound heterozygote. The FASTA is the genome entry's
`reference_fasta` (`--reference-fasta` on `annotate` to give another). The
split that made an annotation is recorded in its checkpoint; an annotation
made with an older split is done again by the next run.

**Protein-coding genes only**. Every tier tests only the genes the
genome's GTF calls `protein_coding` (from the genome entry's `gtf`, or
`--gtf`). A gene the GTF does not list is left out too. The summary says how
many genes were left out of each test.

**What qualifies.** Two tiers, both from the ANNOVAR annotation the DNA arm
already writes:

- `ptv`: stop-gain, start-loss, frameshift, and splice-site variants within
  2 bases of the exon.
- `ptv_missense`: those plus a missense with REVEL >= 0.5 or AlphaMissense
  "likely pathogenic".

**Two models:**

- `dominant`: one qualifying allele, gnomAD grpmax AF < 0.001.
- `recessive`: homozygous, or two or more qualifying heterozygous events in the
  gene (`possible_compound_het`; phase is not known), AF < 0.01. Overlapping
  alleles are one event. Two different ALTs at one site, a `1/2` call, are two.
  Pieces split from one FreeBayes call are one event, unless they are two
  different ALTs at one position.

A genotype counts only with QUAL >= 30, DP >= 10, GQ >= 20 and, for a
heterozygote, an allele balance of 0.25-0.75. Genes are matched on Ensembl gene
ID on both sides.

The coding tiers take an allele's frequency from gnomAD's exome table. An
allele outside the exome capture is not in that table; its
frequency comes from the genome table, and it counts as absent from gnomAD
only when neither table has it.

**Alleles too common for gnomAD are dropped**. A caller over-calls
some alleles in nearly every sample -- homopolymer indels, reads of a
pseudogene mapped onto its gene -- that gnomAD has as rare or not at all.
Before any gene is scored, each qualifying allele's number of carrier
probands is compared with what its gnomAD frequency (AF_grpmax, so an allele
common in one population is not caught) predicts. When the chance of that many
carriers or more is below 1e-6, the allele is taken out of the carriers and,
where gnomAD has it, out of the gene's expected count too.

**The test.** Each proband's chance of carrying a qualifying genotype is worked
out from gnomAD in that proband's population. The count of carriers is compared
with that by an exact Poisson-binomial test, with Benjamini-Hochberg q-values
within each model and tier. On chrX and chrY outside the pseudoautosomal
regions, a male counts with one allele, and a proband of unknown sex is left out
of that gene's test.

**Ancestry and sex.** `--ancestry auto` (the default) estimates each proband's
mixture of the gnomAD populations (afr, amr, asj, eas, fin, mid, nfe, sas) from
the cohort's common coding SNPs. `--ancestry nfe` (or any one population) uses
that one for everyone. An `ancestry` column in the sample sheet sets it per
sample. Sex comes from a `sex` column, or it is inferred from chrX
heterozygosity. Both are written to `<prefix>_ancestry.tsv`.

**One per family.** Relatives are not independent draws, so only one affected
person per family is used: the `proband`, or the first listed. The others are
recorded with the reason. At least 5 probands are needed.

**Results**, in `gene_burden/results/`:

- `<prefix>_burden_<model>_<tier>.tsv`: one row per gene with the observed
  carriers, `n_tested`, the expected count, `log2_obs_exp` (with 0.5 added to
  both), p and q.
- `flag`: `absent_recurrent` when a carrier rests only on an allele that gnomAD
  does not have and that three or more probands carry. That is often an
  artefact of the calling rather than a finding.
- `overrepresented_alleles`: the gene's alleles dropped as too common for
  gnomAD, each with its number of carriers; `carriers_from_overrepresented_alleles`:
  the carriers who counted only through them (add it to `n_probands` for the
  count before the drop). The summary lists every allele dropped.
- `_manhattan.png/.pdf` (genes by position, -log10 p) and `_qq.png/.pdf` (with
  the genomic inflation lambda) for each table. The y axis stops at
  max(12, 2.5 x the genome-wide line); genes above it are drawn at the top as
  triangles and counted in the caption. Up to eight genes below 0.05 are
  named, and a name that would cover another is left off.
- `<prefix>_evidome_input.tsv`: the four coding tests for Evidome.
- `variants`: the alleles behind the gene, per proband. A listing longer than
  an Excel cell holds (32,767 characters) runs on into `variants_2`,
  `variants_3`, ..., cut between two alleles, never inside one; join them in
  order for the whole listing.

**The variants of a few genes.** Once gene_burden is complete:

```
paralome DNA --project <project> --extract-damaging-variants SCN1A,ENSG00000144285
paralome DNA --project <project> --extract-all-variants SCN1A,ENSG00000144285
```

Genes are names or Ensembl ids. This runs no pipeline: no stage is checked,
run or made stale, and nothing else is needed on the command line. One file
per gene, in `gene_burden/extracted/`:

- `<prefix>_<GENE>_damaging_variants.tsv`: the alleles gene_burden counted for
  the gene, one row per allele and proband, with `zygosity`, `consequence`,
  `AF_grpmax`, `REVEL`, `AlphaMissense`, `absent_from_gnomAD` and
  `burden_tests` (every model and tier it counted in).
- `<prefix>_<GENE>_all_variants.tsv`: every row of the segregation table (the
  annotated VCF as a table, with every sample's genotype) whose gene columns
  name the gene, written as it is.

A gene found nowhere, or with no allele counted, is named and gets no file.

**Regulatory variants.** UTR, upstream, downstream and intronic variants, and
intergenic ones within 10 kb of a gene (called when `--gene-padding` reaches
them), are tested separately as an allele count, with no damage score, because
none exists for them. This needs the gnomAD v4.1 GENOME table (`gnomad41_genome`) in the
genome's ANNOVAR database and in the annotation. Without it the tier is
recorded as not run, with the reason. It is never mixed into the coding tiers
and never sent to Evidome. On a whole genome this test is inflated, so its
tables add `p_gc` and `q_gc`: p corrected by the tier's lambda (genomic
control, applied only when lambda is above 1). Its Manhattan plot is drawn
from `p_gc` and says so; the QQ plot shows the uncorrected p with lambda.

**Speed.** With `--threads N` (the pipeline gives it the run's threads, at
most 16), the annotated VCF is read one chromosome per process through its
tabix index, and the log says as each chromosome finishes. The job asks for
N CPUs and 64G.

Not done yet: phasing, which would turn `possible_compound_het` into a call.

### The zips a run writes, and where they go

Every zip is named for what is inside it, with the modality after the prefix
where more than one modality writes it, and every zip goes in one
folder, `omicview/`, organised by analysis, so there is nothing to
collect: open OmicView on that folder's files.

```
<project>/omicview/                     a bulk run (RNA, ATAC, DNA, proteomics)
    statistics_output/  genotype_output/  genome_tracks/  functional_evidence/

<parent>/omicview/                      single-cell: <parent> holds data/
    <run>/                              each run's folder, its path under <parent> mirrored
        <rds>_cell_expression_embeddings.zip    the run's cell export (scRNA)
        trajectories/  statistics_output/  genome_tracks/  functional_evidence/
```

Each stage keeps its own working files where they were (`<project>/<stage>/`);
only the zips move. A single-cell run nested deeper under `<parent>`
(`<parent>/cohort_a/t_vs_c`) mirrors that path (`<parent>/omicview/cohort_a/t_vs_c/`);
a run outside `<parent>` uses its folder's name.

| Zip | Holds | Written by | Opens in OmicView's |
|---|---|---|---|
| `<prefix>_<modality>_statistics_output.zip` (RNA, scRNA, ATAC, snATAC, proteomics) | the statistics (below) | `statistics_output` | Results viewer |
| `<prefix>_DNA_genotype_output.zip` | the variant calls, gene burden and the filter's pass list (below) | `genotype_output` (DNA) | Variants viewer |
| `<prefix>_<ATAC\|snATAC>_genome_tracks.zip` | the genome browser tracks (below) | `genome_tracks` | Genome viewer |
| `<prefix>_DNA_reads.zip` | the reads FreeBayes used at every called site (below) | `genome_tracks` (DNA) | Variants viewer, each variant's panel |
| `<prefix>_scRNA_trajectory_output.zip` | the lineages and the statistics along and between them (section 2, Single-cell trajectories) | `trajectories` (scRNA) | -- (OmicView's viewer follows; an agent reads it) |
| `<rds>_cell_expression_embeddings.zip` | a single-cell object's cells (section 5); a scRNA run's is in its own omicview folder, the only one written, with the lineages when the run fits them | the extraction (the run's copy: `singlecell_rna`'s trajectory fit) | Cells viewer |
| `<prefix>_<modality>[_<stratum>]_<statistic>_evidome_output.zip` | Evidome's evidence | `functional_evidence` | Evidence viewer |

The output stage is named for its zip: `statistics_output` (RNA, scRNA, ATAC,
snATAC, proteomics, Olink) or `genotype_output` (DNA). Every pipeline has it
before Evidome (`functional_evidence`); DNA, ATAC and snATAC have
`genome_tracks` between them. `statistics_output` writes
the statistics zip:

**The results** are every comparison of every statistic folder (and stratum)
the differential wrote, read from the folders' manifests: each comparison's
machine table at full precision, with each sample's value as the report gives
it; the sample PCA and outlier tables; the sample sheet (on single-cell, the
pseudobulk manifest; on proteomics, the sheet), so OmicView colours samples by
any of its columns; and on ATAC and snATAC each comparison's motif enrichment.
They are Parquet files under `results/`, described by `results/manifest.json`.
OmicView's Results viewer lists the comparisons, their features with filters, a
volcano, each sample's value of a feature by group, the sample PCA and the motif
tables; every list exports as TSV.

The stage checks itself: issued again, it keeps the zip until one of its inputs
changes. `--stop-at statistics_output` runs everything but the genome tracks
and Evidome. A single-cell object's cell export is not in the statistics zip:
it is its own zip in `<parent>/omicview/` (section 5).

### DNA: the variant browser

Every DNA run builds a bundle of its results for the variant browser, after
gene burden (the `genotype_output` stage):

```
<project>/omicview/genotype_output/<prefix>_DNA_genotype_output.zip
```

The zip holds the whole-genome segregation table, cut into genomic bins, every
gene-burden table and the filtering stage's pass list, as Parquet files. It is
stored, not compressed: Parquet is compressed already, and the browser reads a
stored zip in place.

To browse: open OmicView (<https://github.com/BioinformaticSolutions/omicview>), hosted or
downloaded, click **Variants** and choose the zip. Nothing is
uploaded: the page reads the zip on that computer. The zip holds only the
results; the viewer is OmicView's. Search a gene or a region, or leave the search empty for the whole
genome's candidate variants; the filters, the burden tests, a pop-up per
variant with every carrier's genotype, and **Export TSV** on every list.

- **Pipeline filter**: on, only the variants the run's filter profile passed
  (its `filter/results/<profile>/<prefix>_pass.tsv`); off, every variant called.
- **Carrier counts**: how many probands carry each variant, against the number
  gnomAD predicts, and how many variants have more or fewer carriers than
  predicted; a filter lists them.

The stage uses the run's threads (at most 32) and 64G as one job. Issued again
it keeps the zip until one of its inputs changes. Without gene burden (no
burden reference) or filtering, it builds the bundle without them and says so.
It needs the `duckdb` Python package, a Paralome dependency.

The variant lists a review starts from -- variants in a candidate gene,
segregating rare damaging, non-segregating rare damaging, ClinVar P/LP, VUS or
conflicting, possible de novo, and on a cohort rare damaging variants by
affected carriers -- are not written as files: each variant's row carries the
columns they are cut by (`report_lists`, the segregation, population, protein
consequence and clinical buckets, `n_affected_carriers`, `affected_carriers`),
and the Variants viewer filters by them. "Rare" is rare, ultrarare or not in
gnomAD; "damaging" is a high or moderate protein consequence. The candidate-gene
list is the viewer's.

**The reads behind each call.** DNA's `genome_tracks` stage, after
`genotype_output`, slices every sample's BAM at the sites the VCF holds, as
FreeBayes read them (primary, not duplicate, not QC-failed), with the genome's
sequence around each site and the gene models, into
`<project>/omicview/genome_tracks/<prefix>_DNA_reads.zip`. OmicView's Variants
viewer shows each variant's reads from it. `--no-genome-track` leaves it out of
a run.

### ATAC and snATAC: genome browser tracks

Bulk ATAC and snATAC write their peaks and per-sample profiles for a genome
browser once the differential has run, which is when the viable peaks are
known: the `genome_tracks` stage, after `statistics_output` and before Evidome,
working in `<project>/genome_tracks/` and writing

```
<project>/omicview/genome_tracks/<prefix>_ATAC_genome_tracks.zip              (bulk)
<parent>/omicview/<run>/genome_tracks/<prefix>_snATAC_genome_tracks.zip       (snATAC)
```

They are the slow part of the export (every BAM or fragment file is read).
`--no-genome-track` leaves the stage out of a run, so it goes from the
statistics zip straight to Evidome.

The BAMs are not exported. The tracks are:

- **The viable peaks** (`peaks/<prefix>_viable_peaks.bb`): the union of the
  peaks tested in any comparison of any statistic (and, on snATAC, any
  stratum). A peak is tested when it reaches `--min-mean-cpm` over that
  comparison's samples.
- **One peak track per comparison** (`peaks/<stratum>__<folder>__<comparison>.bb`):
  the peaks it tested, coloured by direction (higher red, lower blue, no
  direction grey), each with its log2 fold change, p-value and q-value; the
  tracks carry no call of significance.
- **One profile per sample** (`coverage/<sample>.bw`): fragment coverage
  inside the viable peaks in 25 bp bins, in CPM (covering fragments per base
  per million eligible fragments of the sample). Outside the viable peaks a
  profile has no data at all, not zero. On snATAC the samples are the
  pseudobulk samples `--replicate-column` made, after `--remove`. Each profile is
  labelled with its value of the run's first `--factor`.
- **Gene models** (`annotation/genes.bb`) from the run's GTF, and
  `chrom.sizes`; `manifest.json` lists every track and the strongest peaks to
  start from.

Bulk ATAC reads its fragments from the deduplicated BAMs as peak
quantification counts them, and scales each sample by one million over its
eligible fragments. snATAC reads the fragment
files the Signac object names: the object holds each file's path, not the
fragments, so the file must be where that path says (`fragments.tsv.gz`
beside its `.tbi`). A file that is not there stops the stage on PLM-0231,
listing every path the object holds; fix it in R with
`Fragments(obj)[[i]] <- UpdatePath(Fragments(obj)[[i]], new.path = "...")`
and save the object. A pseudobulk sample's eligible fragments are all its
cells' fragments off the mitochondrial genome.

The bigWig and bigBed files are written with the UCSC tools
`bedGraphToBigWig` and `bedToBigBed` (`conda install -c bioconda
ucsc-bedgraphtobigwig ucsc-bedtobigbed`, then `paralome setup`); an ATAC run
that reaches the genome tracks without them is refused before it starts
(PLM-0416). The zip is stored, not compressed. The stage runs as one job (bulk:
the run's threads, one BAM per process). The tracks keep their own checkpoint
in the stage folder, so a change to the results alone rebuilds the zip without
reading the BAMs or fragment files again. OmicView's Genome viewer (igv.js)
draws them from the zip in place: every sample's profile coloured by its group,
the viable peaks, each comparison's peaks and the gene models.

### DNA: FreeBayes regions that stall

FreeBayes can stop making progress on one region for days. Two things keep
that from holding up a run:

- **`--max-region-length`** (default 100,000 bases; 0 = no cap). A longer region
  is cut into pieces that meet end to start, so no one call is given a whole
  large gene with its padding.
- **`--stall-minutes`** (default 30; 0 = off). A call that writes nothing for
  that long is stopped, with everything it started. Its region is finished in
  2 kb windows from the last whole record it wrote. A window still running
  after its time limit is run once more with no limit. Each stall is a row in
  `freebayes/logs/freebayes_stalls.tsv`.

Run again, variant calling submits only the chunks that have no shard. A changed
option plans afresh and moves the earlier shards to `superseded/` rather than
deleting them. Every path in the job scripts is absolute, so `sbatch` works from
any folder. Region files are FreeBayes's own 0-based, end-exclusive
coordinates, and so is a `--regions` file given to `paralome DNA`: it is handed
to FreeBayes as it is. Segregation and filtering shard their work
by those same regions and are told by the pipeline to read them that way;
there is nothing to set. BED lines are 0-based either way.

### DNA: a whole genome

- **"absent from gnomAD exomes".** Most non-coding variants of a genome have no
  gnomAD exome frequency at all, and that is what the population column says.
- **One allele per row.** A row of a multi-allelic site reports one ALT, in
  `reported_alt`: the one the most affected samples carry, then the rarest,
  then the most damaging. Its consequence, gene and frequency are that allele's.
- **SLURM.** Under `--backend slurm --submit`, segregation, filtering and gene
  burden each run as a SLURM job of their own, not inside the supervisor on one
  CPU.

### Samples at once and CPUs: --jobs and --threads

`--threads` is the CPUs a stage may use in total. `--jobs` is how many samples
run at once in the stages that work one sample at a time -- read QC,
alignment, duplicate marking -- and each running sample gets `--threads`
divided by `--jobs` (STAR at most 40, fastp at most 15). Leave `--jobs` out to
run every sample at once, up to `--threads`.

Variant calling and annotation are split by genome region, not by sample, so
`--jobs` does not apply to them: they keep `--threads` one-CPU pieces running,
and Paralome cuts the regions into about eight pieces per CPU so a few long
genes do not leave CPUs idle. Lower `--jobs` to fit memory -- STAR needs
~32 GB per human sample -- without slowing the stages that do not need it:

```
paralome DNA ... --threads 160 --jobs 8
```

aligns 8 samples at a time on 20 CPUs each, then calls and annotates on 160.

**No one job asks for more than three quarters of the largest node.**
A job that runs as one piece -- a finalizer, a pseudobulk, a
merge -- is given `--threads` CPUs up to that ceiling, and the script says so
when it brings a request down. A job asking for a whole node waits for that
node to be empty, and with a one-CPU pipeline supervisor always running
somewhere, it can wait for ever. The ceiling comes from `sinfo`; without it,
requests are left as asked.

### Collecting files into one folder

`paralome collect` copies every file under `--project` whose name contains
`--filter` into one flat folder (`--out`, default `<project>_collected`). It
plans first -- which files to copy, which are already there, which need a
dated name because a different file already has theirs -- and then copies
`--jobs` files at a time (default 4):

```
paralome collect --project Study --filter fastq.gz --out /fast/Study_fastq --jobs 4 --dry-run
paralome collect --project Study --filter fastq.gz --out /fast/Study_fastq --jobs 4
```

The result is the same whatever `--jobs` is; only the time changes. The disks
set the ceiling, not the CPUs: on fast storage 4 to 8 at once helps, and more
than the disks can serve only shares the same throughput. For a large haul,
hand the copies to SLURM -- one array task per file, `--jobs` at a time, and a
last job that checks every file arrived at its full size:

```
paralome collect --project Study --filter fastq.gz --out /fast/Study_fastq --jobs 4 --backend slurm --submit
```

It prints the job numbers and returns at once. The plan, scripts and logs are
in `.paralome_collect/` inside the destination; the finalize log ends with the
summary and names any file that did not arrive. Running the same command again
copies only what is missing. `--slurm-node` pins the copies to one node, and
without `--submit` the plan and scripts are written and nothing runs.

### Before committing to a run

```
paralome extract --rds obj.rds --project study --genome <name>
                                            # metadata columns and their values
paralome status                             # every project below here
paralome setup doctor                       # is this machine set up
paralome setup where                        # where this machine keeps genomes and the registry
paralome setup genome                       # genomes you have, and what this machine has
paralome explain 104                        # what an error code means
```

**Add `--dry-run` to any analysis.** It prints the command it would submit and
runs the full preflight — opening the sample sheet and checking every FASTQ it
names, resolving the genome and confirming its files are on disk, checking that
no recorded tool has moved. It submits nothing and writes nothing.

Every real run does those checks too, whether you ask or not, and refuses with
**PLM-0117** listing *all* the problems at once rather than the first. The point
is to fail on the login node where you are standing, in seconds, instead of on a
compute node one poll cycle later.

There is no separate checker command: a checker that lives beside the thing it
checks is a second implementation of it. The validation is the preflight; the
dispatch plan is what `--dry-run` prints.

### Single-cell: extract first, then choose the comparison

Every option that names a group -- `--factor` and its control level,
`--pair-by`, `--covariate`, `--replicate-column`, `--stratify-by` -- is matched
exactly against the object's metadata. Read the metadata before typing them.
Run the extraction once, on its own:

```
paralome extract --rds obj.rds --project study --genome <name>
```

`--genome` is required: the extraction attaches an Ensembl ID to every gene
from that genome's GTF, once, and every later comparison inherits it.

`--project` is the parent project: the folder every analysis of this object
sits under. The extraction goes in `study/data`, and every analysis whose
`--project` is anywhere inside `study` -- `study/A_vs_B`,
`study/secondary/A_vs_B` -- uses it instead of reading the object again. An
analysis looks upward from its own `--project` for the nearest `data/` holding
an extraction of the same `--rds`; one made from another object is passed
over. With none above it, an analysis makes its data folder beside its
`--project` (`<project-parent>/data`). `--out` names a data folder verbatim.

It prints every metadata column, how many distinct values each has and
whether it can group samples, and writes three tables into `study/data/qc/`:

- `metadata_columns.csv` -- one row per column: `column`, `dtype` (numeric or
  character), `n_distinct`, `grouping_suitable` (yes for a character column
  with at most 50 values), `n_missing`, and `min`/`median`/`max` for a numeric
  column.
- `metadata_values.csv` -- one row per value of each groupable column:
  `column`, `value`, `n_cells`, `pct_of_cells`. **This is where the exact
  spelling of a control level is.** Copy it from here.
- `metadata_summary.csv` -- the two above in one file.

Then choose:

- `--replicate-column` -- the one column whose value is a cell's biological
  sample (library, animal, donor). Its values are the pseudobulk samples, one
  per `--stratify-by` value when you stratify. Required on every single-cell
  run, pseudobulk and `:cell` alike. There is no `--group-by` (refused by
  name, PLM-0119).
- `--factor COLUMN:CONTROL` -- a column and its control level, spelled as
  `metadata_values.csv` spells it; `--pair-by` -- the column that says which
  samples pair; `--covariate` -- a numeric column (`metadata_columns.csv`
  gives its range).

A factor, a pairing column or a covariate must describe a **sample**: one
value for every cell of each `--replicate-column` sample. Aggregation carries every
such column into the pseudobulk `sample_manifest.csv`, and leaves out a
column that varies within a group (its report names it), so a column like
that cannot be used. Cell type varies within a sample; it is a stratum (`--stratify-by`),
not a factor.

### Single-cell trajectories (scRNA)

Every scRNA run has trajectories: the object's lineages, fitted on the whole
object (every cell of every condition together), and the run's own statistics
along and between them, so an agent reading the output can see how the cells
move from the root and how treatment changes that.

**The method**. `--trajectory-method monocle3` (the default) or
`slingshot`:

- **Monocle3** learns one principal graph through the cells in the object's
  UMAP (`learn_graph`, without closing loops, so the graph is a tree). The
  root is the graph node the most cells of the `--trajectory-root` cluster are
  nearest. Every path from the root to a leaf of the graph is a lineage; a cell
  is on every lineage its nearest graph node lies on (the shared trunk is on all
  of them), with Monocle3's one pseudotime -- its distance along the graph from
  the root -- on each. `lineages.tsv` says how each leaf was derived: the
  clusters along the path (each node's cluster is that of the most of the cells
  nearest it; a cluster held by a single node at a junction is left out), the
  leaf's node and cluster, the nodes from the path's last branch point to its
  leaf (`branch_nodes`) and the cells on that lineage alone
  (`cells_on_it_alone`): a lineage of few nodes and few cells of its own is a
  spur of the graph more than a fate. Each lineage is drawn as its path through
  the graph's nodes. Cells in a part of the UMAP the root's part of the graph
  does not reach have no pseudotime. Its settings, in
  `--trajectory-fit-cells`: `reduction=<name>` (the 2D/3D embedding the graph
  is learned in; by default the object's UMAP), `ncenter=<n>` (the graph's
  nodes), `minimal_branch_len=<n>` (the shortest branch kept; Monocle3's
  default 10), `use_partition=y|n` (a graph in each partition Monocle3 finds,
  its default, or one across every cell). By default Monocle3 learns a
  separate graph in each partition and never joins them, so parts of the UMAP
  outside the root's partition get no lineage; `use_partition=n` joins them.
  For weaker connections (learn_graph's own, by Monocle3's names):
  `close_loop=y|n` (join ends of the graph that come close, e.g. a cell-cycle
  arc meeting its start; Paralome's default `n` keeps a tree; with loops a
  lineage is the shortest path from the root to its leaf),
  `geodesic_distance_ratio=<x>` and `euclidean_distance_ratio=<x>` (with
  `close_loop=y`, how far apart two ends may be, along the graph and in the
  UMAP, and still be joined; Monocle3's defaults 1/3 and 1; larger joins
  more), `orthogonal_proj_tip=y|n` (extend each leaf out to the cells beyond
  it; default `n`), `prune_graph=y|n` (trim short branches; default `y`),
  `l1_gamma=<x>` (L1.gamma: how hard the graph is pulled short and smooth;
  default 0.5; lower lets it stretch further) and `l1_sigma=<x>` (L1.sigma: how
  local each node's pull on the cells is; default 0.01; larger reaches across
  gaps). More nodes (`ncenter`) also carry the graph into thin arms.
  `monocle3.json` records every setting given (`learn_graph_control`).

  **Welds**. A tip of the learned graph is often not a fate but
  a branch the graph did not tether. `--trajectory-weld` (repeatable) joins
  tips, named as the learned graph's lineages are (`fit/tips.tsv`, which keeps
  that numbering whatever is welded):

  | Weld | Joins |
  |---|---|
  | `L39=L42` | the two tips directly |
  | `L30=L33=L35` | each tip to a new node at their middle |
  | `L39~L42` | L39's tip to the nearest node of L42's own branch (from its last branch point to its tip) |
  | `'L39->L42'` | L39's tip to L42's, L42's tip staying an end: L39 stopped short of the end L42 reached (quote it: the shell reads `>` as a redirect) |

  A route moves away from the root along the learned graph. `~` and `->` welds
  are crossed one way only, from the first tip; an `=` weld either way, and a
  route that crosses it may run on back down the other tip's own branch (that
  branch was an untethered corridor, not a fate's).

  A weld goes into the graph at its length in the UMAP, and the cells are
  ordered from the root again. A weld makes a loop: there is then more than
  one way from the root to what lies beyond it (an RPC can leave by more than
  one exit), and each way is a lineage. The lineages are every route from the
  root to each end -- a leaf left once the welds are made, or a point of a loop
  where its two ways meet (every neighbour nearer the root), named for the
  welded tip whose branch it is on (else `M1`, `M2` ...). A route keeps its
  end's tip name: the route the learned graph had is the name alone (`L44`),
  any other a letter after it (`L44b`, `L44c` ... by length). On each route a
  cell's pseudotime is Monocle3's moved by how much longer that route is to its
  graph node than the shortest way; the `pseudotime` column of the cell export
  stays the shortest. `lineages.tsv` gives each route's `route_length` and the
  `welds` it passes through; `mst.tsv` marks the weld edges. Welds that make
  more routes than `--trajectory-max-routes` (default 200) are
  refused, the error giving how many routes cross each weld (PLM-0127):
  the busiest are the ones to make one-way or join in one weld. When only the welds change, the
  learned graph (`fit/graph.rds`) is kept and only the ordering and the
  lineages are made again (so too when only `--trajectory-max-routes` changes). Whether the fit is to run again is read from the
  fit's own record (`trajectories/fit/checkpoints`): a run asking for other
  welds, method or `--trajectory-fit-cells` than it was made with fits again,
  whatever the stage's request record says; any other setting learns the graph, and numbers its
  tips, again.

  ```
  --trajectory-fit-cells use_partition=n,ncenter=1000 \
      --trajectory-weld L39=L42 --trajectory-weld L30~L33 --trajectory-weld 'L12->L15'
  ```
- **Slingshot** threads curves through the clusters' centres, in a reduction
  of many dimensions (below).

Each method's settings are refused with the other, and `--trajectory-dims` is
Slingshot's (PLM-0123, PLM-0125). Changing the method refits only the fit, as
any other fit setting does.

```
paralome scRNA ... --trajectory-root tissue_strata:0-RPC \
    --trajectory-fit-cells minimal_branch_len=20,use_partition=n --stop-at singlecell_rna
```

**Where they run**. The lineages are fitted by `singlecell_rna`, as
its last step, right after the extraction and the pseudobulk, and written into
the run's cell export at once, for OmicView's Cells viewer. The statistics are
the `trajectories` stage, which reads only the fit: the supervisor submits it
together with `singlecell_rna_differential`, and `statistics_output` together
with Evidome, because each pair reads only finished stages (scRNA's stages are
run by what they read, not by their place in the list; the other arms still run
one stage after another). Fitted again (another `--trajectory-fit-cells` or
`--trajectory-dims`, another `--trajectory-method`), only the fit runs again, and only what reads it -- the
trajectory statistics and Evidome -- is stale; the extraction, the pseudobulk,
the differential and `statistics_output` are kept. To try several fits:

```
paralome scRNA ... --trajectory-root celltype:RPC --trajectory-method slingshot --trajectory-fit-cells 10:random --stop-at singlecell_rna
paralome scRNA ... --trajectory-root celltype:RPC --trajectory-method slingshot --trajectory-fit-cells 100:random --stop-at singlecell_rna
```

each refitting in minutes and replacing the lineages in the cell export.
`--stop-at` keeps the stage named and what it reads: `--stop-at trajectories`
runs `singlecell_rna` and the trajectory statistics, not the differential.

```
paralome scRNA --project study/A_vs_B --rds obj.rds --replicate-column donor \
    --statistic anova --factor diet:CD --factor drug:VEH --genome <name> \
    --trajectory-root celltype:RPC
```

`--trajectory-root <column>:<value>` is required whenever a run reaches the
stage: the cell-metadata column whose clusters the lineages are fitted on, and
the cluster they start from, spelled as `metadata_values.csv` spells it. It is
never guessed. A run without it is refused before anything is queued
(PLM-0121). To look at the UMAP first, run with `--no-trajectories`, then run
the same command again with `--trajectory-root`: the finished stages are kept
and only the trajectories run.

What the stage does:

- **The lineages.** Monocle3's root-to-leaf paths (above), or Slingshot on the object's own reduction: an integrated one
  (a name with harmony, then integrated) when the object has one, else `pca`;
  only one of more than three dimensions. The clusters are the root's column.
  Each cell gets a pseudotime and a weight on every lineage and the lineage it
  is on (`L1,L2` before the lineages part), and the curves are drawn in every
  2D or 3D embedding the object holds (its UMAP).
- **The lineages the statistics follow**. Welds can make
  hundreds of routes, so `--trajectory-lineages` groups them: `fate` (the
  default) is one lineage per end state, every route ending in the same
  cluster together, named for it (`Rod`); `exit` is one per way out of the
  root's cluster and end state (`X_to_Rod`: the routes that leave the root
  into cluster X and end in Rod); `route` is every route the fit made (`L44`,
  `L44b` ...). A cell is on a grouped lineage when it is on any of its routes,
  its pseudotime there its earliest on them. `lineages.tsv` in the zip lists
  each lineage's routes, exit and end.
- **Expression along the lineages**. Each gene along each
  lineage's pseudotime, by the run's `--factor`s: each cell's log2(CPM+1) on
  the lineage modelled as the full factorial of the factors (every factor,
  every interaction, each from its control level) crossed with a natural
  spline of the cell's pseudotime (3 df). Every term is tested Type II, as the
  run's ANOVA tests its terms: each factorial term (the factors' effect on the
  gene's level along the lineage), `pseudotime` (the gene changes along the
  lineage) and each factorial term crossed with it (`drug:pseudotime`: the
  factors change how the gene moves along the lineage). A gene is tested on a
  lineage when its mean CPM over the replicates (each replicate's cells on
  the lineage pooled) is at least `--min-mean-cpm`. `--trajectory-model`
  chooses the engine:

  ```
  --trajectory-model ls            least squares on the cells (the default): F, df, p, q
  --trajectory-model monocle3      Monocle3's fit_models: LLR, df, p, q
  --trajectory-model ls:df=4       the spline's df (default 3; 1 is a straight line)
  ```

  `ls` fits every gene of a lineage in one solve. `monocle3` fits each model
  the tests need with Monocle3's `fit_models` (negative binomial on the
  counts, the cell's size factor its offset) and compares each pair as
  Monocle3's `compare_models` does (twice the difference of the
  log-likelihoods, on the difference of their residual df); Monocle3's default
  quasi-Poisson has no log-likelihood to compare. The q values are BH over the
  genes tested, per lineage and term. One table per lineage,
  `<prefix>_trajectory_<lineage>_pseudotime_<ls|monocle3>.tsv` (every gene,
  `tested` saying which were), and
  `<prefix>_trajectory_pseudotime_<ls|monocle3>_summary.tsv`: per lineage and
  term its df, the genes tested and how many have p at or below `--alpha`. The
  run's `--statistic` is not run in each segment of a lineage: the main
  design's strata ask that question.
- **Composition.** Each lineage's cells are cut into `--trajectory-bins`
  segments (default 5) of equal numbers of cells by pseudotime, `bin1` nearest
  the root (a cell before the lineages part is in a segment of each). Per
  replicate, the share of its lineage's cells in each segment (`Rod_bin3`:
  cells moving along pseudotime) and the share of its fitted cells on each
  branch (`branch_Rod`: cells moving between lineages), tested with the run's
  `--statistic` at the replicate level (a share is a replicate's, so
  `anova:cell` is `anova` there). With `lm` on a gene
  (`--covariate-feature`), the shares are regressed on that gene's
  expression in each replicate: every cell of the replicate, summed.

When the object is already extracted, the root's column and value are checked
against its cells before anything is queued (PLM-0234). Everything is in

```
<parent>/omicview/<run>/trajectories/<prefix>_scRNA_trajectory_output.zip
```

`fit/` (cells, the routes, the graph, the curves), `lineages.tsv` (the
lineages the statistics follow), `expression/` (the gene tables),
`composition/` (each statistic folder, as section 7 describes them, and
`segments.tsv`) and `manifest.json`, which says what each part is. The stage
works in `<run>/trajectories/` and checks itself: issued again, it keeps the
zip until the root, the bins, the fit's cells or dimensions, the statistic,
`--trajectory-model`, `--trajectory-lineages`, the extraction or the run's
samples change, or the lineages are fitted again. The
fit is in `<run>/trajectories/fit/`, with either method's record (`monocle3.json` or `slingshot.json`)
and log; its record, `trajectory_fit`, says what it was made from.

Monocle3 is installed from GitHub, in the R the extraction uses (`--rscript`):
`remotes::install_github("cole-trapnell-lab/monocle3")`, after its
dependencies (leidenbase, sf, terra and BPCells among them). Slingshot is the
Bioconductor package `slingshot`: `BiocManager::install("slingshot")`. Paralome
does not install either. The preflight checks that the run's method loads
before anything is submitted (PLM-0418, PLM-0417). The statistics run as one job (64G); the fit runs in
`singlecell_rna`'s job. snATAC has no trajectories stage.

### Proteomics, and paired phosphoproteomics

```
paralome proteomics --proteins proteins.csv [--phospho phospho.csv] --samples sheet.csv \
    --project Study/Drug_vs_Ctrl --genome <name> \
    --statistic t --factor treatment:Ctrl
```

The search stays in the exporter's software; Paralome starts at its table.
Read: Proteome Discoverer protein and phosphopeptide exports, MaxQuant
`proteinGroups.txt`, FragPipe `combined_protein.tsv`, DIA-NN
`report.pg_matrix.tsv`. A table none of them recognises is refused
(**PLM-0227**), never read by guessing.

The sample sheet has one row per sample: `column` (the abundance column's
header, or the sample name inside it, e.g. `Ctrl_1` for
`Abundances (TMT 126, Ctrl_1)`), `sample_id`, then any metadata columns,
which `--factor`, `--stratify-by`, `--pair-by` and `--covariate` name. Every abundance column must be in it (**PLM-0228**). A
paired protein and phospho export share one sheet.

What the run decides, and writes down, without being asked:

- **Zeros are missing.** Exporters write 0 for "not measured".
- **Logged or not** is read from the header, then the values (negatives
  exist only on a log scale; no log2 intensity reaches 100). A table that
  cannot be told is refused (**PLM-0229**).
- **Every sample is scaled** to equal totals, CPM-like, over the features
  measured in every sample; log2(x + 1) follows, as for RNA.
- **A phospho table is paired with its protein table only when their
  loadings move together.** Each sample's loading offset is measured in both
  tables, group means removed, and correlated across samples (p from
  shuffling samples within groups). Paired: the phospho table takes the
  protein table's factors. Otherwise it is scaled on its own totals.
- **A missing value is set to half the feature's lowest measured value**
  (`half_min`), the one way missing values are set; there is no switch. On an
  Olink study, with the lowest 10-40% of complete assays made missing,
  half_min kept the full data's direction in 91% of 3,304 refits, against 74%
  when they were left missing.
- **A feature is tested** when it is measured in at least half the samples of
  at least one design group.
- **Proteins get their Ensembl gene** from the genome's GTF: an Ensembl ID in
  the row, else its gene symbols (gene column, protein-group members, `GN=`),
  matched exactly, then case-insensitively, then the UniProt entry name. A
  group spanning genes, or a symbol the GTF gives two genes, gets none and is
  listed.

A phosphopeptide is one feature per exported form (its sites annotated, not
combined). With `--phospho` the arm answers three questions, each
named in its files:

- **`protein_change`** -- did the protein change? Each protein, its own
  statistic and its own error.
- **`phospho_change`** -- did the phosphopeptide change? Each phosphopeptide,
  on its own abundance (not adjusted for its protein), its own error.
- **`phospho_vs_protein_change`** -- did the phosphopeptide change differently
  from its protein? For each phosphopeptide whose protein was measured, a
  two-way ANOVA of measure (phospho, protein) by the comparison's two
  levels over its samples, each sample counted once per measure; only its
  interaction is reported, with one error pooled over the four cells. Its
  log2 fold change is (phospho level - control) - (protein level -
  control). Always named `anova`; it runs beside `t` and `paired-t`, for
  each of their comparisons (every level of the `--factor` against its
  control), and is skipped, saying so, for `anova` or `lm`.

Read the three together: a phosphopeptide that changed while its protein did
not is the phospho-specific signal; one that moved with its protein may reflect
protein abundance. OmicView reads them together from the statistics zip, and
Evidome gets all three as one table. Every result name ends with its question,
e.g.
`<prefix>_t_<comparison>_phospho_change.csv`,
`<prefix>_anova_<comparison>_phospho_vs_protein_change.csv`.

Where things go. The import, scaling and QC go in the parent project's
`data/proteomics/` (the rule of the single-cell extraction): `qc/samples.tsv`
(each sample's total, factor, loading offset, missing count),
`qc/missing_by_intensity.tsv`, `qc/on_off_features.tsv` (measured in one
group, absent from another). The statistics go in `--project`:
`protein_differential/` and `phospho_differential/`, each holding statistic
folders as every other pipeline writes (`phospho_vs_protein_change` in
`anova/`). Evidome runs once per statistic, on one table
holding every question as its own block, under `evidence/`:
`<prefix>[_<stratum>]_<statistic>_proteomics_evidome_output.zip`
(with no `--phospho`, on the protein results alone). Before Evidome, the
statistics output (`<project>/omicview/statistics_output/<prefix>_proteomics_statistics_output.zip`)
puts the results of every question in one zip for OmicView's Results
viewer. `--statistic`, `--alpha`, arms and strata are the shared
ones (section 6). The command runs its stages itself rather than through the
stage supervisor; it may run inside a SLURM job of your own.

### Olink panels

`paralome olink` reads Olink's own plate exports -- the long format NPX
Signature writes, one row per sample and assay, as CSV, .xlsx or .parquet --
by Olink's column names (PlateID, SampleID, SampleType, Assay, UniProt,
OlinkID, QuantifiedValue, BelowLOD, Unit; PLM-0240). Give each plate once:

```
paralome olink --plate batch1_plate1.3.csv --plate batch2_plate_1.xlsx --plate batch2_plate_2.xlsx \
  --samples olink_sheet.csv --project Study/time_course --genome human_grch38 \
  --statistic anova,lm --factor time:0 --covariate time --block-by subject --stratify-by experiment
```

* **The value** is QuantifiedValue (pg/mL), tested on log2(x + 1) like every
  other Paralome arm. NPX is not used.
* **Below the limit of detection.** A value Olink flags `BelowLOD` (TRUE/1), or
  gives no concentration for, is missing, and set to half the assay's lowest
  measured value (`half_min`), as in proteomics.
* **Plate bridging.** Every plate after the first `--plate` is brought onto the
  first plate's scale, assay by assay, by the median ratio of the sample
  controls (SAMPLE_CONTROL wells, matched by SampleID) both plates ran, each
  control counted where both its values are measured. A plate sharing no
  control with the first is refused (PLM-0241). `bridging.tsv` lists every
  factor and the controls used.
* **Bridged only where a question spans plates**. A question --
  the run, or each stratum of `--stratify-by` -- whose samples' wells are all
  on one plate is tested on that plate's own values
  (`unbridged_wells.tsv`; the run says `a question on <plate> alone is tested
  on its own values`). Within a plate the bridging factor is the same for every
  sample, so it changes no sample's rank. An assay no shared control measured
  on both plates has no factor: a question within one plate keeps it, a
  question across plates has it missing on the plates that needed one.
  `sample_plates.tsv` gives each sample's plates.
* **The import** (read, checked, bridged) is in the parent project's
  `data/olink/` and is shared by every analysis of those plates.
* **The sample sheet** (PLM-0242/0243) is one sheet for every question of the
  plates: `PlateID` and `SampleID` name the well as the export does,
  `sample_id` the sample it is, then the design's columns. Wells given the same
  `sample_id` are averaged (their design columns must agree); control wells may
  be listed and are passed over. A sample with no value in a column the run
  uses (`--factor`, `--block-by`, `--covariate`, `--pair-by`) is not in that run
  (`samples_not_in_this_run.tsv`), and an experiment column with
  `--stratify-by` chooses each question's samples (below).
* **Results** are the proteomics arm's: `protein_differential/`,
  one `statistics_output.zip` for OmicView, and Evidome. Each assay is given its
  Ensembl gene by its symbol. An assay named by no gene symbol (Abeta40, Abeta42,
  pTau217) is given its gene by its UniProt accession when the genome has
  Ensembl's UniProt table registered (below): Abeta40 and Abeta42 are APP,
  pTau217 is MAPT, and `measured_form` in `features.tsv` and in the results
  keeps the assay's own name, the form of the protein it measures. A symbol
  whose other genes are all on alternate haplotypes or patches (the MHC's
  HLA-DRA, AGER, MOG in an Ensembl GRCh38 GTF) is the primary assembly's gene,
  in every pipeline.

### Panels: `--statistic ...,panel` (Olink and proteomics)

`panel` named beside the statistic whose design it follows (`--statistic t,panel`,
`anova,panel`, `lm,panel`) fits, for each stratum, an elastic-net panel to every
comparison of that design a panel can be fitted to, on the values its statistics
are tested on (log2, half_min):

* each factor's level against its control (a main effect; the other factors held
  as unpenalised columns), and with two or more factors each such pair within each
  combination of the others (a simple effect);
* each factor of three or more levels, all its levels at once (multiclass);
* `--covariate`: a regression panel predicting it.
* Interaction contrasts are listed as not fitted: they are a difference of
  differences, not a set of samples to tell apart.

With `--block-by` or `--pair-by` each assay is centred within its block (the
subject's own level taken out, as the blocked ANOVA does) and every split keeps a
block whole. A comparison whose smallest group has fewer than 3 samples (blocks)
is not fitted, with the reason: cross-validation needs every group in every fold.

The model is an elastic net (mix 0.5): logistic, multinomial or Gaussian, each assay
standardised, the penalty chosen inside each training part by its own 3-fold
cross-validation. Written to `<stratum>/panel/` and into the statistics zip
(`panel/`):

* `*_panel_performance_*.tsv`, one row per comparison: the held-out AUC (several
  groups: the mean of each against the rest) or R² over 10 repeats of k-fold
  cross-validation (k = up to 5, the smallest group's count), its 2.5-97.5%
  range; **chance** for this design and these samples, the same cross-validation on
  20 shufflings of the labels (within blocks), and the panel's p-value against them;
  the **best single assay** (chosen inside each training part, scored held out) with
  its own chance and p-value, and the panel minus it; how many assays the full data
  select; folds, repeats, shufflings, resamples, the penalty and the seed.
* `*_panel_markers_*.tsv`, one row per assay and comparison: how often 100
  bootstrap resamples select it, its usual sign and how consistent that is, its
  median weight per standard deviation (two groups: positive is higher in the
  level; a covariate: rises with it; several groups: one weight per group), and its
  own held-out score.

Cross-validation on small samples can score below 0.5 with no relationship at all;
`chance_mean` is the level a panel is read against. A panel is about a minute of CPU per
comparison for 40 samples and 40 assays. Its cross-validation repeats, shufflings and resamples
run across `--threads` worker processes (default 0: every CPU the process may use,
no more than `SLURM_CPUS_PER_TASK` inside a job), each with the maths libraries at one thread; each
piece has its own random stream from the seed, so the numbers are the same for any `--threads`.
`paralome olink` and `paralome proteomics` may run inside a SLURM job, where the
panels take the job's CPUs.

With `--backend slurm` the command fits no panel itself: each stratum's panels are
submitted as a SLURM job (`paralome-worker panels run`, its plan in `<stratum>/panel/panel_plan.json`)
asking for `--threads` CPUs, or with no `--threads` the `resources.threads` of your config (8 as
shipped), and one job after them builds the statistics zip again with `panel/` in it. The command
prints the job numbers and ends; run it on the login node as usual. With `--backend local` the panels
run in the command, across its CPUs.

### Repeated measures: `--block-by`

`--block-by subject` adds the column's levels to the model of `anova` and `lm`
as a term that is fitted and not tested, so every factor and covariate is
tested within blocks (a one-way ANOVA becomes `y ~ time + subject`; an lm
`y ~ time + subject`). A factor that is one value within every block (a
between-subject factor such as sex, blocked by subject) cannot be tested this
way and is refused (PLM-0221), as is a model with no error left. `t` and
`paired-t` do not use it (`paired-t` pairs by `--pair-by`); a request with
neither `anova` nor `lm` is refused. On `paralome olink` and
`paralome proteomics`.

### Experiments in one sheet: `--stratify-by` columns, lists, and controls

One sample sheet (or one object's metadata) can hold several experiments.

* `--stratify-by Experiment,tissue_strata` names several columns; a stratum is
  one value of each, every combination present, in its own folder
  (`Exp1__Retina`).
* A cell may list several values, comma-separated (`Exp1,Exp1b`; quote it in a
  CSV): the sample is in each of those strata. A sample is still one sample --
  its cells are summed once -- and is only reused by each stratum it is in.
* A sample with no value in a stratum column is in no stratum and is not tested.
* `--factor Treatment:2mo,dcKO_CTL,OE_CTL` lists the controls of the
  experiments: each stratum takes the one listed level it holds. A stratum
  holding none is skipped with its reason, like any stratum the design cannot
  test; a stratum holding two of them is refused (PLM-0113), because one of them
  would be a treatment. Give that comparison its own experiment instead.
* Every stratum runs the same `--statistic`.

So scripts that differ only by experiment become one:

```
paralome scRNA --project Experiments --rds merged.rds --replicate-column sampleID \
  --stratify-by Experiment,tissue_strata --statistic t \
  --factor Treatment:2mo,dcKO_CTL,OE_CTL --genome mouse_grcm39 ...
```

A comparison whose control is a treatment of another experiment in the same
run (6mo vs 9mo beside 2mo vs 6mo) is a run of its own.

### If someone else set this machine up

On a shared server or a cluster you will probably find genomes already there,
built by a colleague. You do not need a copy of the files and you do not need
to be told where they are:

```
paralome setup genome list         # what this machine holds
paralome setup genome sync         # take the ones you want
```

`sync` copies the *entry* — the record of where the files are — into your own
configuration. The files themselves stay where they are and are read in place,
so adopting a 200 GB genome costs you nothing and takes a second.

Two consequences worth knowing. Your copy of the entry is yours: removing a
genome you adopted affects nobody else, and Paralome will not offer it back at
the next sync. And because the entry is a copy rather than a reference, its
owner can later mark the genome as superseded or unusable and that judgement
still reaches you — your doctor and your runs read it out of the machine's
registry each time, which is why a genome you have used for months can start
warning you one morning.

Run `paralome setup` once before all this, not just `sync`. Sync will create a
configuration if you have none, but only `setup` discovers which tools this
machine has and which scheduler it runs.

If nothing is listed and you expected something, the person who administers the
machine has `docs/INSTALLATION.md`.

### Registering a genome

`--genome` names a bundle you have registered, not a path and not an assembly
name in the abstract:

```
paralome setup genome              # what is available on this machine
paralome setup genome add PATH     # read a genome folder and register it
paralome setup genome list         # the same listing, said explicitly
paralome setup genome sync         # exchange genomes with the machine registry
paralome setup genome remove NAME  # unregister one; files are never touched
paralome setup genome withdraw NAME --reason "..."   # retire it, with a reason
paralome setup genome restore NAME                   # un-retire it
```

**`add` is the only way in.** A genome is registered by naming the folder it
lives in; there is no second command that takes the paths one at a time, and no
`--name`. Both of those existed and both were ways past the
rules `add` enforces — that a genome lives in the shared repository, and that
its name is derived from what it is rather than typed by whoever registered it.

`--species`, `--assembly`, `--annotation-source`, `--annotation-release` and
`--modified` / `--stock` override what the scan read, so a setup script states
the identity without a terminal. With `--yes` it never prompts at all.

**The name comes from the genome, not from whoever is typing.** It is derived
as `<Species>.<Assembly>.<source-initial><release>`:

```
Homo_sapiens.GRCh38.e113      Ensembl 113
Mus_musculus.GRCm39.e112      Ensembl 112
Homo_sapiens.GRCh38.g44       GENCODE v44
```

That is why `--name` is gone. A name someone types is a name only that person
can guess, and the whole point of a shared registry is that nobody has to: the
name alone says whether two results are comparable. A genome renamed onto its
derived name keeps answering to the old one — `setup genome rename` records
former names, so commands and scripts written against the old spelling go on
working.

**Point Paralome at the folder:**

```
paralome setup genome add /shared/paralome/genomes/Homo_sapiens.GRCh38.e81
```

Everything is read from the folder and shown before anything is written. The
STAR index is found by its `SA`/`Genome`/`SAindex` files rather than by its
name, so it can be called anything. The reference and annotation come from
`genomeParameters.txt` — the index records the two files it was built from, so
a folder holding two GTF releases has no ambiguity. ANNOVAR is found by
`table_annovar.pl`, its build version and protocol list read off the database
filenames; snpEff by its jar, and the database name from the directory under
`data/`.

**Motifs are shared across the repository.** JASPAR CORE vertebrates is the
same file for human, macaque, mouse and sheep, so one copy in the repository
serves them all. Paralome looks inside the genome folder first — a plant or an
insect overrides the vertebrate set by having its own — then in the repository
root, then in a `motifs/` folder beside the genomes, including the
`motifs/jaspar/<release>/` layout Paralome's own downloader creates. Files
ending `.meme`, `.meme.txt` or `_meme.txt` all count.

**Ensembl's UniProt table** (optional). A file in the genome folder
ending `.uniprot.tsv` or `.uniprot.tsv.gz` is registered as `uniprot_table`:
Ensembl's UniProt cross-references, from the same release as the GTF
(ftp.ensembl.org/pub/release-81/tsv/homo_sapiens/Homo_sapiens.GRCh38.81.uniprot.tsv.gz
for Ensembl 81). The Olink arm finds an assay named by no gene symbol by its
UniProt accession in it. Put the file in the folder and run `setup genome add`
on the folder again to register it; two such files are refused (PLM-0245 when
the file is not Ensembl's table).

Without one, ATAC falls back to downloading JASPAR CORE vertebrates into each
user's own `~/.paralome/motifs/`. That works, but on a shared machine it means
every account fetches a private copy of the same file — one in the repository
avoids that, and avoids needing outbound network on a compute node.

These are what a DNA run annotates with, so a genome registered without an
ANNOVAR build or protocol list cannot be annotated until they are recorded or
given as options; nothing is assumed in their place.

A `?` marks anything Paralome inferred rather than read — an ANNOVAR operation
letter guessed from a protocol name, or a build version chosen because
`humandb` held three. Check those before accepting.

The layout; registration renames the folder to the genome's name if it is
called anything else:

```
Homo_sapiens.GRCh38.e81/
    Homo_sapiens.GRCh38.dna.toplevel.fa
    Homo_sapiens.GRCh38.dna.toplevel.fa.fai
    Homo_sapiens.GRCh38.81.gtf
    STAR/
    annovar/
    snpEff/
    motifs/
```

The folder carries the genome's name -- species, assembly and annotation
release -- so two builds of one assembly can coexist and the name a listing
shows is the folder on disk. When the folder cannot be renamed (the name is
already taken, or the account may not rename it), nothing is registered and the
name to give it is printed with the `mv` and `genome add` to run. A motif file shared by every vertebrate can sit in the parent
folder instead; a genome with its own — a plant, an insect — overrides it.

**Which analyses a genome is for** is read from what the folder holds — a
folder with no STAR index is not registered for RNA — and `--analyses`
overrides it. That is what decides which sections `setup doctor` judges it on,
so an ATAC/RNA genome is never called invalid for having no ANNOVAR database.

With no terminal it refuses rather than prompts: pass `--yes`, or supply the
identity flags. A prompt in a batch job does not fail, it waits until the wall
clock runs out.

### Where genomes live, and who can see them

**On a machine an administrator has set up, there is one answer: the shared
repository.** `paralome setup site --root /storage/paralome` creates it once
and records where it is, so every account on the machine finds it — including
accounts created later, who are told nothing and have to look nothing up. It
makes two sibling directories:

```
<root>/genomes     the genome files themselves
<root>/registry    who has published what
```

**And once that exists, registering a genome outside it is refused** with
**PLM-0409**. It is a refusal rather than a warning because otherwise people
would take the easy route and register their own copy that nobody else could
use, and a site would end up with a dozen human and a dozen mouse genomes. A machine that has not been
set up this way is unaffected — a workstation or a laptop registers genomes
wherever they are.

Everyone in the group may add a genome; nobody may delete or rename anyone
else's. That is the filesystem doing it, not Paralome: the directories are
`3775` and `1777`, which is the `/tmp` arrangement, and the sticky bit is what
restricts removal to the owner of each entry.

**The shared registry** is the record layer above the files — a drop-in
directory where each account writes a single file of its own. Nobody reads
anybody's home directory and nobody needs permission to. On a cluster this is
the only channel, and Paralome treats it as the documented one everywhere.

**Peer discovery** is the opportunistic bonus. Paralome reads the account list
from `passwd`, looks for `~/.paralome/config.json` in each home, and shows you
the genomes other people have registered — attributed to the account they came
from:

```
Registered by other accounts on this machine (2 in 1: alice):
  Macaca_mulatta.Mmul_10.e112     5 of 5 files present   [alice]
  Homo_sapiens.GRCh38.e113        6 of 6 files present   [alice]
```

`paralome setup genome sync` adopts any of them into your own configuration
without retyping a path.

This only works where home directories happen to be readable by other accounts,
which is normal on a single lab's machine and normal*ly not* on a managed
cluster. Paralome uses it where it works and never asks anyone to arrange it.

**Paralome tailors its advice to the machine.** When a genome is not visible to
others, what it suggests depends on what kind of system this is — inferred from
whether a batch scheduler is present and how many accounts exist, and
overridable by recording `deployment` under `defaults` in your configuration:

| | what Paralome says |
|---|---|
| `workstation` | nothing to arrange — there is nobody else here |
| `shared` | publish to the registry; or, since the other accounts are colleagues, you *may* make your own home traversable |
| `cluster` | publish to the registry. Opening a home is never suggested |

On a cluster the other accounts are strangers, and a site that keeps home
directories private means it. That is a policy question for an administrator,
not a permissions trick, so Paralome does not raise it.

**Nothing is ever written outside your own home.** Discovery is read-only, and
the filesystem decides what is visible: registering a genome makes your
`~/.paralome` directory and config readable by others — never writable.

Permissions compose down the *whole* path, so Paralome checks every directory
between `/` and your config, and names the ones that block it:

```
Others cannot see this genome yet.
  Permissions compose down the whole path, and these block it:
      /shared/home  (not yours to change)
      /shared/home/alice  (yours)

  You can open yours with:
      chmod o+x /shared/home/alice
```

`o+x` grants traversal, not listing: others can reach a path they already know
and still cannot enumerate your files. Paralome will not run it for you — those
directories are not its to widen.

Where accounts share a parent directory — `/home/projects/<name>`, which is how
RStudio Server installations are usually laid out — that parent is a single
choke point. One directory at `0750` hides *every* account beneath it from
every other, and only an administrator can open it.

`INSTALLATION.md` covers setting a machine up: `setup site` decides these
locations once for everybody.

Nothing about genomes ships in the wheel. A genome entry is a set of absolute
paths, which mean nothing on another machine.

`paralome setup genome sync` reconciles the two in both directions: it offers
your genomes to the machine registry, offers the machine's to you, and where the
same name holds different files it shows which keys differ and asks whether you
want the machine's version or want to keep yours under a new name. It never
resolves that silently.

### Scratch: where a job's temporary files go

A job's temporary files -- the annotation split and its sort, each
ANNOVAR shard's working files, the burden reference's per-chunk ANNOVAR runs,
MACS3's duplicate-free BAM -- can go to scratch on the node that runs the job
rather than beside the results on shared storage, the slowest disk a cluster
has. Results never go to scratch.

The administrator decides once, on the login node:

```
paralome setup site --scratch /scratch            # show what this machine has
sudo paralome setup site --scratch /scratch --apply
```

`paralome setup site` on its own, and `--scratch` with a value before
`--apply`, list the usual places (`/scratch`, `/lscratch`, `/local/scratch`,
`/local_scratch`, `/scratch/local`, `/tmp`) that exist here, with filesystem
type, whether it is local or network, and free space, and suggest one.
`--scratch` takes an absolute path, `TMPDIR` (use the private directory many
SLURM sites give each job) or `none`. A user can override the site's choice
for one shell with `PARALOME_SCRATCH` (same values). Unset, nothing changes:
temporary files stay where they always were.

On the node, each job makes its own directory,
`<scratch>/paralome-<analysis>-<user>-<job>-<task>-<what>.XXXXXX` (for example
`paralome-RNA-you-48213-3-star.a8F2kQ`), and removes it when it
ends: normally, on an error, or when SLURM stops it with SIGTERM (`scancel`, a
time limit). A job killed outright (SIGKILL, a node that goes down) runs no
clean-up, and its directory stays on that node until someone removes it. A
stage command run by itself, outside an analysis, leaves out `<analysis>-`.

A job uses scratch only when the directory exists on that node, is writable,
and has room for the job's estimate of what it will write with a tenth of the
disk kept free besides. Otherwise it says so in its log and uses the shared
location: a node without the drive is slower, never broken.

Jobs that start together on one node do not each see the same free space:
every job that writes to scratch asks SLURM for that space
(`#SBATCH --tmp`), so on nodes that define `TmpDisk` the scheduler does the
accounting; where no node in the partition defines it, no `--tmp` is asked.
Everywhere, each job also writes what it expects to use into a ledger in the
scratch root (`.paralome-scratch-reservations`, under a file lock), and the
room check subtracts what the other running jobs have reserved and not yet
written. An entry whose job is gone is dropped by the next job that reads the
ledger. Where the lock cannot be had, the check works on free space alone and
says so.

MACS3's duplicate-free copy of each BAM is written with `samtools view -1`
(fast compression), about the size of the source BAM rather than up to six
times it uncompressed.

### Retiring a genome

Adoption copies an entry rather than referencing it, which is what makes the
crowd-sourced master list work without a shared file. It also means that
removing your own entry tells nobody: everyone who adopted it keeps a working
copy, and a genome you have since discovered was built wrong goes on being used
by four other people.

Withdrawal is the answer. It leaves the entry in place and attaches a reason to
it, because a deleted record cannot carry a message:

```
paralome setup genome withdraw Homo_sapiens.GRCh38.e81 \
    --reason "GRCh38 e113 is built and indexed"
```

Two severities, and the difference matters:

**`--superseded`** (the default) warns and proceeds. This is the ordinary case:
hg19 after hg38. Nothing about the genome is wrong, years of correct results
came off it, and somebody mid-analysis should not have it pulled out from under
them — they should simply not start new work on it. `setup doctor` still reports
it VALID, and names the reason.

**`--fatal`** refuses. Use it when the results themselves are wrong: an index
built against the wrong FASTA, a GTF that does not match the assembly, a
truncated download. Every future run stops with **PLM-0407**, including the
owner's own — a genome that produces wrong answers does so regardless of who
runs it. `setup doctor` reports it INVALID even when every file is present and
readable, which is exactly the case the flag exists to describe.

`paralome setup genome restore NAME` undoes either one.

**A fatally withdrawn entry cannot be removed casually.** `remove` refuses it,
because that entry is the only thing warning everyone who adopted the genome —
their copies carry no marker and read your reason out of the shared registry
each time. Deleting it would silently make a genome you called dangerous valid
again for all of them. `--force` is there for when you are sure nobody adopted
it. A superseded entry is not blocked: work against it is still correct, so
tidying it away harms nobody.

**How an adopter finds out.** Their copy carries no marker — the owner wrote it
afterwards — so Paralome follows whichever channel the copy came through.

The **shared registry** is checked first, because it works however private
people's home directories are. When the owner withdraws a genome, the entry is
republished, so the registry's copy carries the reason and anyone who adopted
from it sees that on their next `setup doctor`, `setup genome list` or run.

Failing that, an entry adopted through **passwd discovery** records the account
it came from, and that account's config is read directly.

On a machine where users cannot read each other's homes — the usual RStudio
Server arrangement — the registry is the only channel, which is why it is the
one checked first.

The owner's word wins either way: holding an older copy does not un-withdraw
somebody else's genome. A registry entry is only trusted when it names the same
files, compared by definition rather than by name, so two people registering
unrelated references under one string cannot block each other.

### What Paralome needs to know about a genome

Registration asks five things beyond the file paths. They exist for one
purpose: so Paralome can tell you when something newer is available, and be
right about what "newer" means.

```
Species             Homo sapiens
Assembly            GRCh38            the coordinate system
Annotation source   ensembl           ensembl / gencode / refseq / custom
Annotation release  113               or v44, for GENCODE
Modified            no                have you altered the FASTA or the GTF?
```

The folder scan pre-fills what it can read from the filenames and you confirm
or correct it. **`unknown` is a real answer** — recorded, not left blank. A
genome with an unknown annotation takes no part in supersession, and Paralome
says so at registration, so that you never mistake "nothing was offered" for
"nothing newer exists".

**Why the source is asked and not just the number.** Ensembl 113 and GENCODE
v44 are both current, and neither is newer than the other. Ordering them by
integer would be nonsense wearing the costume of arithmetic, so two genomes are
only ever compared within one annotation project.

**Why you are asked at all.** Inferring all of this from filenames fails on
legitimate files (`Homo_sapiens.GRCh38.113.chr_patch_hapl_scaff.gtf`), and
fails *silently*: an offer that never appears looks exactly like there being
nothing to offer. Asking turns a silent gap into an answered question.

**Modified genomes.** If you have added transgenes — GFP, mCherry — or filtered
biotypes, say so. A modified bundle is compared only to itself: stock GRCh38 is
not a replacement for a GFP-bearing GRCh38, because swapping it in would drop
the transgene and the reads over it would simply stop mapping. Paralome will
still tell you a newer release exists, but never as a straight swap. The
modification is shown wherever the genome appears, including in `setup doctor`,
so that anyone reading a gene table containing GFP can find out why in one
command.

**The replacement is computed, not typed,** and there are two kinds:

**Drop-in replacement** — same species, same assembly, same annotation source,
higher release. The coordinates are identical and only the gene models moved,
so you can re-run and compare the results to the old ones.

**Newer assembly** — same species, a later assembly. GRCh37 to GRCh38 is the
usual case. This is offered, because it is what someone on hg19 needs to hear,
but never in the same words: it is a migration, not an upgrade. Every
coordinate changes and old results cannot be compared to new ones without
lifting over. Paralome will not order two assemblies that merely look similar —
`hg38` and `GRCh38` are the same thing under two conventions, and it declines
to guess that rather than risk moving your coordinates.

---

## 2a. How reads are trimmed before anything is aligned

Every assay that starts from FASTQ -- DNA, RNA and ATAC -- runs fastp first,
and it trims to these defaults:

| What | Value | fastp flag |
|---|---|---|
| Sliding-window quality trim, 5' to 3' | window 4, mean quality 15 | `--cut_right` |
| Minimum length after trimming | 50 | `--length_required` |
| Adapters | whatever `setup site --adapter-fasta` recorded | `--adapter_fasta` |

These are trimmomatic's `SLIDINGWINDOW:4:15 MINLEN:50 ILLUMINACLIP:...`,
because that is the trimming a working RNA-seq practice uses and fastp's own
defaults are not it.

**Why the minimum length matters more than it looks.** fastp defaults
`--length_required` to 15. A 20bp read survives QC, is counted in your
survival percentage, goes to STAR, fails to align, and is recorded as
"unmapped: too short" -- a counter in a file nobody opens. Two pipelines can
report the same high-90s survival rate and hand the aligner completely
different data.

**Why `--cut-right` and not `--cut-tail`.** They run in opposite directions.
`--cut_tail` scans 3' to 5' and stops at the first window that passes, nibbling
the tail. `--cut_right` scans 5' to 3' and, at the first window that fails,
discards that window and everything after it. Trimmomatic's `SLIDINGWINDOW` is
the second, so `--cut-right` is what reproduces standard trimming.

**Recording your adapters once.** An adapter FASTA is a fact about your
sequencing kit and your machine, so it belongs in the site config rather than
in every script:

```
sudo paralome setup site --root /shared/paralome \
    --adapter-fasta /shared/paralome/adapters/TruSeq_PE.fa --apply
```

Every project on that machine then trims against it. Without one, fastp guesses
the adapter from overrepresented sequence at the read ends -- which usually
works and is never stated anywhere.

**Changing it for one run.** `--length-required`, `--cut-right-window-size`,
`--cut-right-mean-quality` and `--adapter-fasta` all take a value on any assay
command. `--no-cut-right` turns the window trim off entirely.

## 2b. The STAR output filters, and `--loose`

Three parameters decide whether a read ends up in your BAM. Paralome sets all
three explicitly on every run, so the generated sbatch records which thresholds
that run enforced.

| Option | Default | What it does |
|---|---|---|
| `--star-out-filter-mismatch-nover-read-lmax` | 0.06 RNA, 0.04 DNA/ATAC | mismatches as a FRACTION of read length |
| `--star-out-filter-score-min-over-lread` | 0.66 | minimum alignment score, as a fraction of read length |
| `--star-out-filter-match-nmin-over-lread` | 0.66 | minimum matched bases, same |

**The mismatch limit is the fraction alone.** STAR's absolute mismatch cap
(`--outFilterMismatchNmax`) is switched off, so the fraction of read length
(of both mates together, for paired reads) is the only mismatch limit: 0.04 for
DNA and ATAC, 0.06 for RNA, where mapping across splice junctions is more
error-prone.

**The last two are what produce "unmapped: too short".** That phrase does not
mean the read was short. It means no alignment scored above
`ScoreMinOverLread`, or matched more than `MatchNminOverLread` of the read.
A read failing the mismatch filters usually lands in this bucket too, rather
than in "too many mismatches".

### `--loose`

Opens all three at once: mismatch fraction 0.3, score and match minima 0.3. For a divergent reference or a deliberately cheap library.

```
paralome RNA --project P --samples s.csv --genome G --statistic t --factor group:CON --loose
```

**It makes alignments worse, not more numerous.** On a 92bp sheep library at
~5-8% per-base divergence, `--loose` moved unique mapping from 11.3% to 29.2%
-- while average aligned length fell from 87.1 bases to 65.5. Those are partial
alignments. Use it to find out whether your filters are what is limiting a run,
not to make a run look better.

Anything you set explicitly still wins: `--loose
--star-out-filter-mismatch-nover-read-lmax 0.08` gives you 0.08 and the loose
values for the other two. A run made with `--loose` records the fact, and
`status` says so under "Aligned with:", so a loose result cannot be mistaken
for a normal one later.

## 2c. Strandedness, detected rather than guessed

If you don't pass `--rna-strandedness`, Paralome works it out from the data. The
principle: a stranded library assigned with the wrong convention sends its reads
to the strand the gene isn't on, they land nowhere, and aggregate signal
collapses. The correct convention maximises the reads that reach a gene at all.

How the panel is chosen,:

1. Take every gene with **no opposite-strand overlapping neighbour** -- so the
   rare antisense pairs can't confound the verdict.
2. Count a wide candidate pool (~1,000 per strand) **once, unstranded**.
3. Keep the **top 50 plus-strand and top 50 minus-strand by that count**.
4. Count modes 1 and 2 over those 100 genes and compare recovery against the
   unstranded ceiling.

Step 2-3 is what stops the panel filling with genes this tissue doesn't
transcribe. A blind random draw would, in a tissue where half the annotation is
silent, leave half the panel contributing nothing and the verdict noisy on
shallow libraries.

**Deliberately not a housekeeping gene list.** That was considered and rejected:
it needs `gene_name`, and Ensembl annotations for non-model organisms carry a
large fraction of genes with only an accession, so a symbol-matched panel can come
back near-empty on exactly the genome that needs it. It also isn't strand-balanced
and would need curating per species. Ranking by what the library actually expressed
gets the same benefit and stays species-agnostic.

**Detection is the default.** `--rna-strandedness` accepts
`auto` (the default), `0`, `1` or `2`.

The verdict is one of three, and it is printed with its reasoning:

- **stranded** -- one mode recovers >= 70% and the other <= 30%
- **unstranded** -- neither exceeds 65% and they're within 1.5x of each other
- **inconclusive** -- resolves to **0 (unstranded)**, loudly

That last one is a deliberate policy, and the reason is that the costs are not
symmetric. Counting unstranded keeps every read and adds antisense noise.
Guessing a direction *wrong* sends reads to the strand the gene isn't on, where
they match nothing, and the signal is simply gone. So an undetectable library
falls back to the recoverable failure -- but the reason is always printed and
recorded, because "inconclusive" covers both a genuinely non-directional
library and a panel that couldn't see enough reads, and those deserve to stay
distinguishable long after they resolve to the same number.

**Every RNA run says which strandedness it uses and where that came from**
Beside the backend:

```
RNA strandedness    : auto (Paralome's default, as written in /home/you/.paralome/config.json)
```

**A default Paralome wrote into your configuration follows the default**
`setup` writes every default into `~/.paralome/config.json` so it
can be seen and edited there, and records which values it wrote under
`paralome_defaults`. A value still equal to what Paralome wrote takes the new
default when a later version changes it, and the run says so once; a value
you changed is yours and is never moved. `setup` wrote
`rna.strandedness: 0`; a file from then that still holds it is changed to
`"auto"`, with a notice, the first time it is read. To keep a value equal to
the default whatever later versions do, there is nothing to set: the default
is what the run uses until you change it.

Preview it without running a quantification:

```
paralome RNA strandedness --samples aligned_samples.csv --gtf genes.gtf --single-end
```

## 2d. Bulk RNA: what is refused, and what is left out

**The pseudocount must be greater than 0.** `--rna-pseudocount` (default 1.0)
is added to both group means before the fold change is taken; 0 or less is
refused. A gene that is constant across the samples is not tested, at any
pseudocount.

**Designs that are refused, each with a message that says what to change:**

- the same `--factor` column twice;
- a `--factor` without its control level (`--factor Diet`), or with a control
  level the column does not hold (**PLM-0113**; the message lists the levels);
- a sample with a blank cell in a factor column (fill it, or leave the sample
  out by starting its line with `#`);
- a design of two or more factors with no sample in some combination of their
  levels (the message names it);
- a pseudobulk ANOVA with as many terms as samples, such as one sample per
  combination: every interaction is in the model, so that leaves no degrees of
  freedom for the error. `--statistic anova:cell` fits that design with the
  cells as the observations;
- two levels that are written the same way in a file name
  (`A+` and `A-` both become `A`), and two requested comparisons that would be
  written under one name;
- `--covariate` and `--covariate-feature` together: `lm` regresses on one of
  them;
- `--refine-responders` without a `--factor`: the refined runs are made per
  comparison the QC audit defines from the factors, so there is nothing to
  refine;
- a differential pseudocount other than the one its matrix was quantified
  with, which the quantification checkpoint records. `--rna-pseudocount`
  reaches both stages, so a pipeline run never differs; the check is there
  because the raw FPKM means, the per-sample values and the expression gate
  are back-transformed with it;
- a GTF whose chromosome names match none of a BAM's (`chr1` against `1`):
  every fragment would miss every gene and the matrix would be all zeros.
  Refused at preflight, before any job is created.

**Which level is the control.** The one `--factor` names: `--factor
Metformin:VEH` makes `MET_vs_VEH`, MET over VEH, however the sheet's rows are
ordered.

**The sample sheet** may start with a byte-order mark (Excel writes one), and
may head its sample column `sample`, `id` or `name`, as the pipeline's own
sheet reader accepts them. Factor columns are named exactly as
`--factor` gives them (case counts); no column is required besides the sample
and its FASTQs or BAM.

**Coordinates in the matrices.** The `start` column of `gene_fpkm.tsv`,
`gene_log2_fpkm.tsv`, `gene_fragment_counts.tsv` and `genes_left_out.tsv` is
0-based and `end` is the last base, as a BED file puts them: a GTF gene at
1001-2000 reads `1000  2000`. `length_bp` is the union-exon length.

**The FPKM library size** is the number of fragments the counter considered,
under its own rules: in a paired-end run, orphan reads (a mate unmapped or
missing) and the pairs it discards are left out of the per-million denominator
as they are out of the gene counts.

**What a recompute removes.** Besides the files its checkpoint lists, a
recomputed differential result removes the QC audit's files under
`results/qc` (`paralome_qc_ensemble.pdf`, `qc_summary.tsv`, the two HTML
pages and `qc_report.json`), since it describes the replaced result, and
what the expression-gate sweep of a version recorded
writing under a statistic folder's `fpkm_filter_sweep/`. A result folder that was copied or moved
is recognised by its recorded paths and made again in its new place, and
the copy's old files are removed as the new ones are written.

**`--refine-responders` is recorded as done only when it was.** When the
refine declines or a comparison fails, the run prints why
(`--refine-responders 3 did not run: ...`), the checkpoint keeps
`refine_responders: 0` with the reason under `refine_responders_failed`, and
asking again tries again. On SLURM, where trying again means
running the whole stage, the same N on an unchanged result is not resubmitted:
the stored reason is printed, and `--overwrite` or a different N tries again.

`--statistic`, `--factor`, `--pair-by`, `--covariate` and
`--covariate-feature` are offered, and mean the same, on every pipeline that
produces differential statistics -- bulk RNA, bulk ATAC, `scRNA` and `snATAC`
(section 6). Only the single-cell two accept `:cell`; the bulk two have one
kind of observation and refuse it. A comma-separated `--alpha` list is swept
by RNA only.

**Genes the matrices do not hold whole** are listed in
`results/machine/genes_left_out.tsv`, written by every quantification (a
header alone when there are none). A gene annotated on two chromosomes is
counted on its primary one, and the row says which and where. A gene annotated
on both strands of one chromosome is left out of every matrix. The run prints
one line with the counts and the first few IDs.

**Inputs.** A gzip-compressed GTF (`.gtf.gz`, as Ensembl ships it) is read as
it is. A BAM's index may be `X.bam.bai`, `X.bai` or `X.bam.csi`. A sheet of
BAMs starts the run after alignment.

**Results are named and numbered the same on both backends.** The SLURM and
local paths of differential expression write the same file names, labels and
values; levels that look like numbers keep their names.

## 3. When something goes wrong

Failures print a number, a line of cause, a line of instruction, and where the
details went:

```
PLM-0104  An input file does not exist
  --rds 'cohort_integrated_rds.rds' does not exist.
```

Two lines: the number, and what went wrong. Everything else is behind the
number. `paralome explain 104` gives the long version, and Appendix A has the
same text. The suggested fix, the traceback and the diagnostic context — which
directory was searched, which similarly-named files are in it, which columns the
object actually has — are written to the log file under the project's `logs/`.

**When a SLURM stage fails, the supervisor's message is not the error.** It
reports only that a job failed, and names the job's own `.err` file. That file
is the error. Note that single-cell stages run out of the **shared data folder**
beside your projects, not inside the project directory — which is why the log
path is printed rather than left to be guessed at.

### While a run is waiting on SLURM

The command you started stays alive and watches the jobs it submitted. What it
does when something other than the analysis goes wrong:

- **One task fails.** It says so at once (`Failed so far: ...`) and lets the
  tasks still running finish. The stage stops, with PLM-0301, when nothing in it
  is running. Finished tasks are kept; the same command again repeats only what
  failed.
- **SLURM stops answering** (`squeue` fails or hangs). It prints one line saying
  so and waits, for as long as it takes. It cancels nothing and no deadline
  runs while the state of the queue is unknown. If a run seems hung, that line
  is the reason; the fix is on the cluster, not in Paralome.
- **A job is held.** It names the job and the hold reason once and keeps
  waiting. `scontrol release <job>` lets it run; `scancel <job>` gives it up.
- **A node fails or a job is preempted, and SLURM requeues it.** The requeued
  job is waited for like any other.
- **You stop the command** (Ctrl-C, `scancel` or `kill` of the supervisor, a
  closed terminal). The submitted jobs are left running. The message names
  them, and they are recorded in `checkpoints/pipeline_supervisor.json`.
- **You run the same command again while those jobs are still going.** It waits
  for them to leave the queue before it submits anything, then continues from
  what they produced. It does not submit the stage a second time.
- **Job numbers are printed as each `sbatch` returns.** If a later submission
  of the same stage is refused, the jobs already queued are on the screen and
  in the checkpoint.

### One run per project

One project is run by one supervisor at a time. Starting the same command
while another is still running on that project is refused with **PLM-0310**,
which names the host, the process and the start time of the one that holds
it. The usual cause is a run started under `nohup`, or in a terminal since
lost, and typed again. The run that is going needs no help; `paralome status
--project <path>` shows where it has got to. There is normally no lock to
clear: it is held by the running process and released when that ends.

### When a finished result is reused, and when it is not

Running the same command again reuses what is finished. What each stage was
asked for is recorded when its command is issued -- the options that decide
its result, the sample IDs and the sheet columns it reads, the outside files
it is given by name and size -- and compared at every start. A stage is
**stale** when its request has changed, or when a stage before it has to run
again. `--dry-run` shows the same verdicts and writes nothing.

The backend, threads, `--jobs`, memory and partition are not part of a
request. Changing them never makes a result stale.

### When an option is changed on a finished project

What happens depends on the stage.

**Seven stages check themselves and recompute on their own**: RNA
quantification, differential expression, functional evidence, the two
single-cell aggregations, variant calling and gene burden.
The run issues them again as they are. Each compares
its options and its inputs with what its checkpoint recorded, computes again
only if they differ, says why, and replaces only the files it recorded
writing. So a changed `--factor` or control level, a changed `--statistic`,
a sample commented out of the sheet, a regenerated matrix, another annotation
or a replaced BAM is simply recomputed. A result that carries no record of its
inputs is made again once, the next time its stage is issued.

**Every other stage is refused, before anything is submitted**, with
**PLM-0308**. Started again over its own output, such a stage would find that
output and keep it, and the run would report the old result as the new one.
Paralome does not remove or replace a stage's results on its own account, and
it adds `--overwrite` to nothing. The one message lists every such stage, the
reason for each (which option changed, from what to what, or which stage it is
behind) and what to remove:

```
PLM-0308  A stage that has to run again would keep its old result
  These stages hold results that no longer answer this run, and Paralome
    does not replace a stage's results on its own account:
  alignment: remove /work/P/star -- its request changed: <option, from what to what>
  deduplication: remove /work/P/dedup -- alignment has to run again
  Remove the folders named, then run the same command again.
  Or put the changed option back as it was (alignment): the results on
    disk answer that request and are kept.
  Nothing was submitted.
```

There are two ways on:

- **Remove what is named and run the same command again.** Those stages are
  then computed afresh, and the stages after them follow.
- **Put the option back as it was.** The results on disk answer that request,
  nothing was touched, and the run carries on with them. The message says when
  this applies; it does not for a stage that is stale for another reason.

A folder is named for removal only when it is that stage's own and inside the
project. A stage folder that is a link inside the project to somewhere else
(`<project>/star` linked to scratch) is the stage's own: it is named by its
path in the project, and its contents are what has to go; the link may stay.

A stage folder given outside the project (`--align-workdir` elsewhere) is
named and said to lie outside the project, where Paralome cannot know what
else there is. The message says what of the stage has to be removed from it:
for the alignment the per-sample BAMs and their indexes and its records
(`checkpoints/alignment.json`, `results/star_alignment_*.json`); for any other
stage everything in the folder. Removing a stage's report files alone is not
enough -- the alignment would find its BAMs and keep them -- and the run is
refused until what the message names is gone.

Where a folder is not named for another reason -- the project folder itself,
one shared with an earlier stage, one that holds an input of this run -- the
message says why and lists the stage's own files instead.

### Adding a sample, or taking one out

Read QC, alignment and duplicate marking work one sample at a time, and each
is finished only when every sample now in the sheet is done. **Add a sample
to the sheet and run the same command**: those stages run for the new sample
only, and the finished samples are kept. The stages after them see a changed
sample list, so they follow the rule above: the RNA stages recompute, and a
stage that does not check itself is named by PLM-0308.

Commenting a sample out with `#` changes nothing for the per-sample stages;
its files stay where they are. The stages that read the whole cohort see one
sample fewer.

### `--overwrite`, and resuming a stage with a different request

Read QC and alignment record, for each sample, the request it was made with:
the options that decide the result and the input files by path and size.

**Resuming with a different request is refused** (**PLM-0309**). A stage
resumes by running only the samples that are not finished; resumed under
other options it would hold samples made two ways and report them all under
the new ones. The message lists each finished sample and what differs. Repeat
the earlier options to finish the stage as it was started, or remove the
stage's folder to redo every sample.

**`--overwrite` on read QC or alignment deletes nothing when you type it.** It
marks every sample "to be replaced", and each task replaces its own sample's
files when it runs. A run stopped part-way therefore still has the samples it
had not reached, and the stage is not complete until every marked sample has
been made again. `--overwrite` reaches a stage only when you type it.

Two related refusals:

- **PLM-0311** -- the stage was submitted before from this folder and those
  jobs may still be running. If they are, they are doing the work; wait for
  them, or `scancel` them to start over. If SLURM does not answer, Paralome
  asks again for ten minutes and then refuses without submitting or
  cancelling anything.
- **PLM-0312** -- a BAM that `align verify` rejected is still in place.
  Alignment run again would take it as finished, so remove the BAMs named and
  their indexes. A sample with no mapped reads would only come back empty:
  leave it out of the sheet with `#`.

### Two factors coded with the same levels

Factorial results are named from the levels: `MET_vs_VEH_within_CD`. When
both factors use the same two levels (`yes`/`no` and `yes`/`no`, `1`/`0` and
`1`/`0`, in either order) that name cannot say which factor it tests, so for
those designs the factor is added to both halves:
`drug_yes_vs_no_within_diet_no`. When the two factors share one level -- a
`Control` reference in both columns -- the shared level alone cannot say
which factor is held, so it carries its factor and the rest stays bare:
`Drug_vs_Control_within_diet_Control`, `Drug_vs_Control_within_HighFat`.
Two names that differ
only by case (`ko` and `KO`) are refused as one name: Excel and some file
systems do not tell them apart.

---

## 4. Finding out what you already ran

`paralome status` with no arguments walks down from where you are standing,
reads every `project_state.json` it finds, and prints one line each:

```
Projects under /shared/home/alice/work

  PROJECT                   PROFILE   UPDATED           NEXT STAGE
  Study_scRNA/CON_vs_TRT    scRNA     2026-09-19 11:04  complete
  Study_snATAC/CON_vs_TRT   snATAC    2026-09-18 22:17  singlecell_motif_analysis_singles
  Study_DNA                 DNA       2026-09-14 08:41  annotation

  3 project(s), 2 unfinished.

  Detail for one:  paralome status --project <path>
```

**What it prints is what you can paste.** The PROJECT column is a path relative
to where you ran it, not a bare directory name — two different projects can
easily share a last name, and `--project CON_vs_WSD` would then find neither.

`--profile scRNA` narrows it. Profiles are case-sensitive, like the commands:
`scRNA`, not `scrna`. If nothing matches, the listing says which profiles *are*
present rather than leaving you to guess.

`--project <path>` deep-dives one project and reports every stage. `--detail`
adds outputs, metrics and file paths.

A project appears here once it has run at least once, because that is when
`project_state.json` is written. There is no depth limit and the scan is
read-only, so an interrupted one leaves nothing behind.

**You should not have to retype a run to ask about it.** `status` reads the
state file, which holds the entire answer, written after every step; it needs
only `--project`.

---

## 5. Where files go

```
<parent>/
    data/                     shared by every comparison of one .rds
        extraction/           per-cell matrices read out of the object
            cell_metadata.tsv every metadata column, unfiltered
        rna/  atac/           aggregated matrices and the sample manifest
        qc/                   metadata_columns/values/summary.csv (extract)
        logs/                 the stage logs a failure will point you at
    <comparison>/             one directory per contrast
        rna/  atac/           one folder per statistic: t/, anova/, t_cell/ ...
            motif_analysis/   (atac) motif enrichment, results/<folder>/
        statistics_output/    the export for OmicView
        genome_tracks/        (atac) the genome browser tracks
        trajectories/         (rna) the lineages and the statistics along them
        functional_evidence/  one Evidome run per statistic folder
    omicview/                 every zip, by analysis
        <rds>_cell_expression_embeddings.zip
        <comparison>/trajectories/ statistics_output/ genome_tracks/ functional_evidence/
```

Bulk analyses keep their stages side by side in the project:
`differential_expression/results/<folder>/` (RNA),
`differential_accessibility/results/<folder>/` and
`motif_analysis/results/<folder>/` (ATAC),
`statistics_output/` or `genotype_output/` (every pipeline) and
`functional_evidence/<folder>/`, with every
zip in `omicview/`. What a statistic folder holds is in section
7.

Reading a Seurat object is expensive, so extraction happens once into `data/`
and every comparison against that object reuses it. Move a project without its
`data/` sibling and the extraction is lost.

**Note that `data/` sits in the project's parent, beside the project.** `--project A/B` puts it
at `A/data`, so everything under `A/` shares one extraction. Two analyses under
*different* parents each extract the same object separately.

### Making extraction faster

The extraction log ends with a breakdown, so you can see where the time goes
rather than guess:

```
[paralome] extraction timing
  R startup + packages              7.2s   10.7%
  read object                      31.9s   47.3%
  GetAssayData (RNA)                5.9s    8.7%
  write cell_metadata.tsv          14.0s   20.7%
  write rna_counts (binary)         6.3s    9.3%
  -- wall clock                    67.5s
  still unmeasured                  2.2s    3.3%
```

Percentages are of **wall clock**, not of the measured subset, so
`still unmeasured` is a real number rather than always zero.

Reading the object is almost always the bulk of it, and `readRDS` **cannot** be
made faster — an RDS is a sequential byte stream, so deserialisation is
single-threaded by construction, and R's gzip inflate is too.

`--rds` also accepts a **`.qs2`** file, which is the same object stored with
zstd and read on many cores:

```r
library(qs2)
qs_save(readRDS("object.rds"), "object.qs2", nthreads = 16)
```

On a 19,070 × 724,502 object that took the read from 164s to 37s. Threads come
from `SLURM_CPUS_PER_TASK`, so a job with `--cpus-per-task 16` uses sixteen
without being told.

**`qs2` has to be installed where the job runs, not where you submit from.**
Installing it on the login node is the obvious thing and is not enough — the
extraction runs on a compute node. If they don't share an R library, put `qs2`
somewhere both can see and point `R_LIBS` at it. The failure names the R that
looked, so you can tell the two cases apart:

```
PLM-0404  A required R package is not installed
  object.qs2 needs the qs2 package, and this R cannot see it. R.home:
    /usr/lib64/R  libPaths: /usr/lib64/R/library  If qs2 is installed
    elsewhere, this job is running under a different R, a different user,
    or on a node that does not have it.
```

**Do not simply uncompress the `.rds`.** It is the obvious idea and it is
usually wrong: on the same object an uncompressed read took *longer* (320s vs
164s) while using less CPU, because the larger file made storage the
constraint. gzip was buying bandwidth, not costing time.

### One object, or a folder of objects

`--rds` takes a single object or a folder of objects. The way the
path is written decides which, and the disk must agree:

| Written as | Meaning | Refused if |
|---|---|---|
| ends in `.rds` (or `.qs2`), any case | a single object, as always | it is not a file |
| ends in `/` (`slides/`, even `my.rds/`) | a folder of objects | it is not a folder |
| anything else | -- | always (PLM-0118) |

Tab completion adds the `/` to a folder, so the folder form comes naturally at
the prompt:

```
paralome scRNA --rds hippocampus_slides/ --project HPC --genome <name> \
  --replicate-column sample --statistic t --factor ...
```

Use a folder when there are more samples than Seurat can reasonably merge into
one object. Paralome never analyses the merged object anyway -- extraction turns
an object into counts, gene names, barcodes and metadata on disk, and every
later stage reads only those -- so it combines the per-sample objects itself,
one in memory at a time:

- **What is read.** Every file ending in `.rds` (any case) directly in the
  folder. Subfolders are not read; every other file is left alone.
- **Barcodes.** Each is prefixed with its file's name and `_`: `HPC_03.rds`
  gives `HPC_03_<barcode>`. Nothing needs doing in Seurat.
- **Each object is extracted on its own** into `data/extraction/rds_members/`,
  with its own checkpoint. A rerun after fixing one file re-extracts only that
  file.
- **Every file is checked before anything is combined, and none is the
  reference.** Every object must carry the columns `--replicate-column` and
  `--stratify-by` name; every one that does not is listed (PLM-0224). The
  objects must have the same genes, compared by Ensembl ID and, for a gene the
  GTF gives none, by its name; if they do not, every group of identical gene
  lists is reported with its files, the genes it has that another group lacks
  and the reverse, and its match rate against the `--genome` GTF, with the full
  lists in `data/extraction/rds_folder_gene_groups.tsv` (PLM-0225). Paralome
  does not pick a winner. Peaks are held to the same rule for the ATAC half
  (PLM-0226).
- **Metadata columns not in every object** are left out of the combined table,
  and listed.
- **`--rna-assay`**, when given, applies to every file.
- **Seurat's variable-feature statistics** describe one object each, so the
  combined gene table does not carry them, and `--top-variable-features` has
  no ranking to take from a folder.
- **On SLURM, one job per object**. `paralome extract --backend
  slurm --submit` extracts a folder as an array, one task per object still to
  extract, `--jobs` at a time (0, the default: all at once), then one job that
  checks and combines them -- the same `extract`, run on a node, finding every
  object done. That job waits for the whole array, so an object whose task
  failed is extracted again there and its error is the one you see. Objects
  already extracted are not submitted. A run (`scRNA`, `snATAC`) given a
  folder with `--backend slurm` does the same before its samples stage, at its
  own `--jobs`.

```
paralome extract --rds RDS_files/ --project Visium_analyses --genome <name> \
  --backend slurm --submit --jobs 4
```

### The cell export: one object's cells for an AI agent

Every extraction also writes
`<rds name>_cell_expression_embeddings.zip` into `<parent>/omicview/`, beside
`data/`, or, in a scRNA pipeline run, into the run's own omicview folder
(`<parent>/omicview/<run>/`) and nowhere else. An object with an ATAC half alone
gets one too, without genes: its cells' metadata and embeddings, which
OmicView's Cells viewer colours by any column. A folder of objects gives one ZIP
per object: each `.rds` is its own dataset, whatever samples or slides it holds. The ZIP holds two files:

- `<rds name>_cell_expression_embeddings.h5ad` -- AnnData: `X` is every cell by
  every gene with a count in at least one cell, as log2(CPM + 1) with CPM over
  every gene of the cell (sparse); `obs` is every `meta.data` column, indexed by
  barcode; `var` is the gene's row name, Ensembl ID and symbol; `obsm` holds each
  two- or three-dimensional reduction the object carries (`X_umap`, a 3-D UMAP,
  t-SNE) -- the file is for plotting, not re-analysis, so PCA, Harmony and other
  many-dimensional reductions are left out -- and, for a spatial
  object, `spatial` (x, y, full-resolution pixels) with the image in `obs`;
  `uns["spatial"]` holds each tissue image the object carries --
  whatever H&E Load10X_Spatial put in the RDS, lowres or hires -- with its
  scale factors, in the layout scanpy reads, so
  `sc.pl.spatial(a, color="<gene>", library_id="<image>")` draws the gene over
  the tissue. The image is only inside the .h5ad, not a separate file.
- `<rds name>_cell_expression_embeddings_manifest.json` -- what the file holds,
  in words an agent can read before it opens it.

**The trajectories.** The scRNA trajectory fit (`singlecell_rna`'s last step)
writes the lineages into the run's export as soon as the fit finishes. The run's
export, in its omicview folder with its other outputs
(`<parent>/omicview/<run>/<rds>_cell_expression_embeddings.zip`), is copied from
the object's (`<parent>/omicview/`, left as it is) or written from the
extraction, never re-extracting the object. The stage is finished only when the
export carries its fit. `obs` gains `pseudotime` (the cell's pseudotime averaged
over its lineages, weighted by its weights), `pseudotime_<lineage>` (blank off
that lineage) and `trajectory_lineages` (the lineages the cell is on);
`uns["paralome_trajectories"]` holds each lineage's clusters (`lineages`,
`lineage_clusters`), the root, and each lineage's curve in each embedding it was
drawn in (`curves/<embedding>`: `lineage`, `coordinates`). The manifest's
`trajectories` entry says the same. The trajectory zip's statistics are for an
agent, not the viewer.

**Drawing it.** Open OmicView's Cells viewer
(<https://github.com/BioinformaticSolutions/omicview>) and choose the ZIP; nothing
needs unzipping, and the file is read on that computer, never uploaded. It draws
genes or any metadata column over every tissue section on the H&E and on the
2-D or 3-D UMAP, with the lineages' curves. What the viewer does is OmicView's,
and its own documentation describes it.

`X` is stored gene by gene (CSC), so one gene's cells are one slice
of the file.

Python reads it with `anndata.read_h5ad` (scanpy draws UMAP and spatial gene
overlays from it directly); R with anndata or zellkonverter. Writing it needs
`h5py` in the Python that runs Paralome (**PLM-0415**).

It is written when an extraction is written. A reused extraction rebuilds it
only when it is missing, older than the extraction, or written by another
version of Paralome -- the export is rebuilt from the extraction;
the object is never read again for it. An extraction that lacks reductions,
coordinates or the tissue image is NOT re-extracted to get them: its ZIP holds
what the extraction has and its manifest says what is missing. To have them,
move that `extraction/` folder aside and run again.

---

## 6. Vocabulary worth getting right

**Extraction is not aggregation.** Extraction reads the `.rds` and writes
per-cell matrices. Aggregation ("pseudobulk") sums those cells into groups.
Separate operations, separate checkpoints.

**Extraction takes what the object has.** Both halves of a multiome come out of
one read; an RNA-only object yields genes and no peaks. `--assay` and
`--rna-assay` exist only to break ties. It is the *analysis* that has a
requirement: `scRNA` needs an RNA half, `snATAC` needs an ATAC half.

**`--replicate-column` defines what a sample is**. It names the
one column whose value is a cell's biological sample; the cells of each sample
are summed into one pseudobulk sample, one per `--stratify-by` value when that
is given. Nothing else groups the cells: the `--factor` columns describe the
samples, they do not split them. A run with one library per factor combination
names the library column. With `:cell` the cells are the N and the samples only
carry each cell's levels, so the F and p do not depend on how the cells are
grouped -- but the samples still decide which features pass the expression
floor, which cells a small or `--remove`d sample takes with it, and the QC
views, so the column is required there too.

**The groups are `--factor` columns of the pseudobulk manifest.** Aggregation
carries into `sample_manifest.csv` every cell-metadata column that is one value
within each sample, so a column that describes the sample (its
treatment, sex, day) is there to be named; `--factor treatment:CTL` compares
each other treatment with CTL. A column that varies within a sample is left
out, and cannot be a factor.

**`--statistic` names the tests, and the data they run on.** One option, the
same on every pipeline that produces differential statistics -- bulk RNA, bulk
ATAC, `scRNA`, `snATAC`, proteomics and Olink. Comma between statistics; a colon before the data
level, where there is a choice:

```
--statistic t                          one statistic
--statistic t,anova                    two statistics, two folders of results
--statistic t,paired-t,lm,anova        all four
--statistic t:cell                     single-cell, the cells as the observations
--statistic t:cell,anova:cell          two statistics, both on the cells
--statistic anova,anova:cell           the ANOVA on the pseudobulk samples and on the cells
```

Any combination of `t`, `paired-t`, `lm` and `anova`, each named once, in any
order. Anything else is refused, naming the four. **`--statistic` is required
and has no default**: a run that reaches its
differential without one is refused before anything is queued (**PLM-0101**).
A run that `--stop-at` ends before the differential needs none.

**`:cell` is the only suffix there is, and it belongs to the statistic it
follows.** Pseudobulk is the
normal case, so it is what you get by saying nothing; there is no term for it.
One run may hold both levels -- `anova,anova:cell` -- and each statistic at each
level writes its own folder (`anova/`, `anova_cell/`): the pseudobulk and cell
calculations are different things, and keeping them in separate folders is
what stops anyone having to work out which column came from which.
`t,anova:cell` runs t on the pseudobulk samples and the ANOVA on the cells;
write `t:cell,anova:cell` for both on the cells. A pipeline
with one kind of observation -- bulk RNA, bulk ATAC -- refuses a colon rather
than ignoring it. Pseudobulk is a data action, not a statistic, which is why
it is on the other side of the colon and not in the list of tests.

Each statistic has the companions it reads, and an option no statistic in the
list reads is refused rather than ignored:

| Statistic | Needs | Refused without |
|---|---|---|
| `t` | one or two `--factor COLUMN:CONTROL` | a factor |
| `paired-t` | the same, and `--pair-by <column>` | `--pair-by` |
| `lm` | `--covariate <numeric column>` or `--covariate-feature <feature>` -- one of them; a `--factor` adds within-level fits | both, and both together |
| `anova` | one `--factor` of three or more levels, or two `--factor` | a factor |

`--pair-by` without `paired-t` and a covariate without `lm` are each refused
with the change to make.

**Each statistic is its own result.** Every statistic named writes its own
folder, named by the flag that asked for it: `t`, `paired_t`, `lm`, `anova`,
and with `:cell` `t_cell`, `paired_t_cell`, `lm_cell`, `anova_cell`. Two
statistics are never combined in one table. The columns have the same generic
names in every folder (`p_value`, `q_value`, `log2_fold_change`); the folder
and the file name say which statistic and which comparison. A pseudobulk
folder reads exactly like a bulk one. Each statistic's q-values are corrected
on their own, and motif enrichment and Evidome run once per folder, so
changing the statistic shows what that change does to the final answer.

On the single-cell pipelines, `statistic_comparison.json` (beside the folders)
sets side by side each comparison made by more than one statistic -- `t` and
`paired_t` both make `KO_vs_WT`: how many features both tested, the
correlation of their log2 fold changes, and the rank correlation of their
p-values. No call of significance.

### What each statistic computes, and on which samples

Which comparisons a run makes is decided once, before anything is fitted, by
`paralome.comparisons.plan_differential`, and every pipeline asks it. The
samples in scope are every sample of the sheet (bulk; a `#`-commented sample is
left out) or every pseudobulk sample of the stratum (single-cell, after
`--remove` and `--min-cells-per-group`). Every comparison is a level of a
`--factor` against that factor's control level, and nothing else: there is no
comparison of two non-control levels (run again with one of them as the
control), and the control is never chosen from the order of the sheet.

- **`t`** -- Student's t on the log values (log2 FPKM for bulk RNA, log2 CPM
  otherwise), level minus control, df = n1 + n2 - 2. It makes:
  - one `--factor`: each other level against the control,
    `<level>_vs_<control>` (`A_vs_VEH`, `B_vs_VEH`). Levels that are numbers
    carry the factor's name: `--factor Day:1` makes `Day4_vs_Day1`,
    `Day9_vs_Day1`;
  - two `--factor`: each factor's levels against its control within each
    level of the other, `MET_vs_VEH_within_CD`, `MET_vs_VEH_within_WD`,
    `WD_vs_CD_within_VEH`, `WD_vs_CD_within_MET`. A factor of three levels
    gives two such comparisons within each level of the other.

  Each group needs at least two samples.
- **`paired-t`** -- the same comparisons, each comparison's samples paired by
  `--pair-by`: a value of that column must have exactly one control-level and
  one other-level sample, or the run stops naming it, and a comparison needs
  at least two complete pairs. What is tested is each pair's level-minus-
  control difference, df = pairs - 1. Extra columns: `n_pairs`,
  `mean_within_pair_difference`, `sd_within_pair_difference`.
- **`lm`** -- the regression of each feature on `--covariate` (a numeric column
  of the sheet; a non-numeric value is refused by sample) or on
  `--covariate-feature` (another feature's values in the same samples, found
  by its ID or any annotation value, so a gene symbol works). Fitted over
  every sample in scope (`<covariate>_slope`) and again within each level of
  each `--factor` (`<covariate>_slope_within_<level>`; `..._within_day4` for a
  many-level factor), except a `--factor` that is the `--covariate` column
  itself, which holds one covariate value per level. Each needs at least
  three samples and a covariate that varies. Its `log2_fold_change` is the
  slope: log2 change per unit of the covariate. Extra columns: `slope`,
  `slope_se`, `intercept`, `pearson_r`, `r_squared`, `n_regressed`.
- **`anova`** with one `--factor` -- the one-way ANOVA. The factor
  has three or more levels, words or numbers; a factor of two levels is the
  two-group comparison and is refused with "use --statistic t". Fitted on
  every sample in scope, with treatment coding (the control level is the
  intercept), it reports:
  - one F-test, `<factor>_one_way`, df = levels - 1 over the model's error
    (samples - levels). When the levels are numbers it also carries the
    straight line on their values and what the level means show beyond it:
    `trend_slope` (log2 per unit of the factor, the same line
    `lm --covariate <factor>` fits), `trend_slope_se`, `trend_F`, `trend_p`,
    `trend_q` (df 1) and `lack_of_fit_F`, `lack_of_fit_df`, `lack_of_fit_p`,
    `lack_of_fit_q` (df levels - 2), both over the same error; the two add up
    to the omnibus F, and its `log2_fold_change`, and so its direction, is the
    trend. In the Evidome table the trend is a directed block of its own,
    `<factor>_linear_trend`, beside the F. Levels that are words have no
    values to put a line through, so their F carries no trend and no
    direction;
  - each other level against the control, `<level>_vs_<control>_contrast`, a
    contrast with the model's pooled error -- a model contrast, not a
    two-sample t-test.

  At pseudobulk it needs at least one error degree of freedom: one sample per
  level is refused, and points to `anova:cell`.
- **`anova`** with two `--factor` -- the two-way ANOVA, each factor of two or
  more levels, fitted on every sample in scope. The interaction is always in
  the model: an ANOVA is what is used when there are interactions, and
  without them a t-test is. Treatment coding with each factor's control level
  as its reference, Type II sums of squares with each term's columns tested
  together (a factor of k levels is one F on k - 1 df), every F over the full
  model's error. Every ANOVA, in every pipeline, reports:
  - three F-tests: each main effect and the interaction (`<A>_x_<B>`). A
    two-level factor's main effect is named `<level>_vs_<control>` (with its
    factor in front when the level does not already say it); a factor of more
    levels has no single direction, and its main effect is `<factor>_F`;
  - the contrasts from the same fit, every one a level against its factor's
    control: each factor's main effects, `<level>_vs_<control>_main_effect`,
    averaged over the other factor, and its simple effects within each level
    of the other, `<level>_vs_<control>_within_<other level>_contrast`. A 2x2
    has six; a 3x2 has ten. A contrast's fold change and its p-value
    come from one contrast vector and the full model's error, so they always
    describe the same comparison. It is a model contrast, not a two-sample
    t-test.

  Every combination of the two factors needs a sample, and the design needs at
  least one error degree of freedom.
- **`anova`** with three or more `--factor` -- the factorial ANOVA: every
  main effect, every interaction and every contrast. The same model and the same coding at any
  number of factors, every interaction of every order in it:
  - an F-test per term: each main effect, every interaction of two factors,
    of three, ... to all of them, named `<A>_x_<B>_x_<C>`. Type II: each term
    within the model of every term that does not contain it, every F over the
    full model's error;
  - the contrasts, every one a level against its factor's control: the main
    effect (averaged over every other factor), the effect within each level
    of each other factor, within each combination of two of them, ... and
    within each combination of all of them (the simple effects), each
    averaged over the factors it does not name:
    `<level>_vs_<control>_within_<level>_<level>_contrast`, the held levels in
    the order the `--factor` options gave their factors. For each non-control
    level of a factor that is the product over the other factors of (1 + their
    number of levels) contrasts: 27 per factor for four two-level factors, so
    15 F-tests and 108 contrasts in all.

  The design needs a sample in every combination of levels and, at pseudobulk,
  at least one error degree of freedom.

`--factor` is given once per factor, always as `COLUMN:CONTROL`. No blank cell,
and (on single-cell data) one value within each pseudobulk sample
(**PLM-0203**).

### The cell level (`:cell`)

`:cell` fits the same comparisons as the pseudobulk run would, with the cells
as the observations: the cells of each comparison's samples, matched to their
pseudobulk sample through the sample manifest. Each sample must match exactly
the `n_cells` the manifest records for it, or the run stops rather than test a
different cohort from the pseudobulk one. A cell's value is
log2(CPM + `--pseudocount`), its CPM taken over every feature of that cell.

- **`t_cell`** -- Student's t on the two groups' cells. Its N is the cells, so
  a group of one sample is allowed: the run prints, and records in the
  checkpoint's `warnings`, that the sample is then confounded with its group.
- **`paired_t_cell`** -- per pair, the mean per-cell value of the level
  sample's cells minus that of the control-level sample's cells; those per-pair
  differences are tested across the pairs, as `paired-t` tests them. Its
  replicate is the pair.
- **`lm_cell`** -- the regression with the cells as N: each cell carries its
  sample's `--covariate` value, or its own value of `--covariate-feature`.
- **`anova_cell`** with one `--factor` -- the one-way ANOVA with the cells as
  N: the same F-test (with its trend and lack of
  fit when the levels are numbers), and the same contrasts against the
  control, each level's cells its
  observations, df = cells - levels. No random intercept: with one sample per
  level the sample and the level are the same column. So, like `t_cell`, a
  level of one sample computes, and the run records in the checkpoint's
  `warnings` that the sample is confounded with its level. Run beside
  `lm:cell` (`--statistic anova,lm:cell --factor day --covariate day`), its
  `trend_slope` is lm_cell's `slope` on the same cells; only the error each is
  tested against differs.
- **`anova_cell`** with two or more `--factor` -- the factorial ANOVA with the
  cells as N, a sample's cells pooled with its replicates', as Seurat pools
  them (the two models for single-cell data are pseudobulk and cell, and the
  user chooses): least squares over every cell, the same F-tests and
  contrasts as `anova` (Type II F per term, every contrast from the full
  model), on df = cells - model columns. A combination of levels held by one
  sample computes, and the run records in the checkpoint's `warnings` that
  such a sample is confounded with its combination, as `t_cell` does. There
  is no random intercept per sample, and no `--peak-subset-limit`.

Every `:cell` statistic is computed for every feature at once and never
subset.
`--naive-cell-t-peak-chunk-size` only bounds how many features' per-cell
counts are held in memory at a time; it changes no number.

### The expression floor

A feature is tested in a comparison when its mean CPM **over that comparison's
samples** reaches `--min-mean-cpm` (default 1.0): the two arms' samples for `t`
and `paired-t`, the regressed samples for `lm`, every sample for the ANOVA.
The `:cell` statistics are judged on the same pseudobulk CPM, so a feature is
tested at the cell level where it is tested at the pseudobulk level. The
option is on `paralome scRNA differential` and `paralome snATAC differential`;
bulk ATAC runs at the default.

A feature that does not reach the floor, or that has one value in every
sample of the comparison, is **untested**: `tested` is false, and it has no
p, q, statistic or effect size. Its means stay, because they are what
the expression was. The q-value is corrected over the tested features only,
so a blank q means "never tested", not "tested and unremarkable".

Bulk RNA has no CPM floor. Its expression gate is an export rule, not a test
(section 7).

### What runs where, and where it comes from

One module computes the statistics for every pipeline: `paralome.stats`. Not a
copy per pipeline, so one column name never has two different tests behind it.
`paralome.engine` routes each comparison's samples into it, for bulk
and pseudobulk data alike; the `:cell` fits are in
`paralome.atac.singlecell_stats` for both single-cell pipelines.

| Statistic | Subroutine |
|---|---|
| `t`, `t_cell` | `engine.fit_two_groups` -> `stats.student_t_from_moments` |
| `paired-t`, `paired_t_cell` | `engine.fit_paired` -> `stats.paired_t_from_pairs` (pairs from `stats.resolve_pairs`) |
| `lm`, `lm_cell` | `engine.fit_slope` -> `stats.ols_covariate_fit` |
| `anova` | `engine.fit_anova` -> `stats.anova_table`, `stats.contrast_table` |
| `anova_cell` | `engine.fit_one_way` (one factor), `engine.fit_anova_cells` -> `stats.factorial_from_sample_sums` (two or more) |
| DNA gene burden | `burden.stats`: an exact Poisson-binomial tail |

Every statistic, at both data levels for the single-cell pipelines, from every
differential pipeline. Multiple-testing correction is Benjamini-Hochberg
from `stats.benjamini_hochberg`, everywhere, gene burden included: per
comparison, each ANOVA F-test its own (`engine.finalize`), never across
comparisons or statistics.

`t` is **Student's t**, the simplest case of the family, not a different
test: on a two-arm,
single-factor design the t, the ANOVA F and a linear-model fit give the
identical p-value, with F = t squared.

Motif enrichment is the one stage of the ATAC pipelines that is not one of
these: it is Fisher's exact, on counts, and takes no `--statistic` of its own.
It runs once per comparison of every statistic folder (section 7).

### The QC statistics: one module too

The statistics that check samples and results, rather than make them, are
`paralome.qc_stats`, for every pipeline. Where one method serves every data
type it is the same arithmetic everywhere; where a data type needs another
rule, the rule is a parameter the pipeline passes, not a copy.

| QC | Arms | Subroutine |
|---|---|---|
| PCA (the outlier QC, the pseudobulk QC, the single-cell PCA) | all | `qc_stats.pca`: exact centred SVD, each component signed so its largest loading is positive; the outlier QC standardises each feature first, the others do not |
| Sample outlier QC (distance to the group centroid, median Spearman to the group) | bulk RNA, bulk ATAC, the single-cell and proteomics sample views | `qc_stats.sample_outlier_metrics` (90% of the variance, ratio 2, Spearman 0.80) |
| Sample-sample correlation | all | `qc_stats.sample_correlation` (Pearson or Spearman; the single-cell PCA heatmap centres each feature first) |
| Clustering | single-cell PCA heatmap, bulk RNA expression heatmap, pseudobulk QC | `qc_stats.average_linkage` (average linkage on 1 - r), `cut_tree`; `kmeans`, `silhouette` and its permutation null |
| Outlier rule | pseudobulk QC flags, bulk RNA's array quality | `qc_stats.robust_centre_scale`: median and 1.4826 x MAD, 3.5 (Iglewicz-Hoaglin) |
| Group ellipses | pseudobulk QC (95% region of the mean), single-cell PCA (1.75 SD) | `qc_stats.covariance_axes`, at the radius each plot chooses |
| Differential audit (permutation nulls, responder, leave-one-out) | bulk RNA | `qc_stats.audit_comparison` and its parts |
| Protein/phospho pairing | proteomics | `qc_stats.within_group_permutation_correlation` |

A test fails the build when any other module computes a PCA, a correlation, a
covariance, a clustering or a permutation itself.

### Effect size, power and `--alpha`

Paralome makes no call of significance and draws no power figures: OmicView
draws power, volcanoes and every significance view from the statistics zip,
with the user's own thresholds. Per feature, the machine tables of `t` and
`paired-t` (and `t_cell`, `paired_t_cell`) carry `cohens_d`, `sd_log2`,
`mde_log2fc` (the smallest log2 fold change detectable at 80% power) and
`observed_power`. That last one is **post-hoc power** and is named so
deliberately: power at the observed effect is a monotone function of the
p-value and carries no information the p-value did not. A slope or an F-test
has no Cohen's d, so `lm` and the ANOVA carry none.

**`--alpha` is the cut a next stage selects by**, never a call written into a
result: the motif enrichment's up and down peak sets, the phospho pairing
test, and Evidome's gene selection. It carries its metric: `P0.05` (the
default) cuts on the raw p-value, `Q0.10` on the FDR-adjusted q-value. The
letter is required, because a bare `0.05` does not say which was meant.

---

## 7. Reading the results

### What a statistic folder holds

Every pipeline writes its differential results through one writer
(`paralome.results`), so every statistic folder -- `t/`, `paired_t/`, `lm/`,
`anova/`, `t_cell/` ... -- has the same files. `<comparison>` is the
comparison's name (`KO_vs_WT`, `MET_vs_VEH_within_CD`, `age_slope`,
`diet_x_drug`, `MET_vs_VEH_main_effect`); `<prefix>` is the project's name
(with the stratum, on a stratified run); `<stamp>` is `rna` (bulk RNA, scRNA)
or `atac_peaks` (bulk ATAC, snATAC):

```
<prefix>_<folder>_evidome_input_<stamp>.csv                       this statistic's Evidome table
machine/<prefix>_<folder>_<comparison>_statistics_<stamp>.csv     every feature, every column, full precision
machine/<prefix>_<folder>_sample_matrix_<stamp>.csv               every feature x sample: the PCA input
machine/<prefix>_<folder>_manifest_<stamp>.json                   every file the folder holds
```

Every file names its statistic (`<folder>`: `t`, `paired_t`, `anova`, `t_cell`
...), so a file taken out of its folder still says which test made it. So does
Evidome's bundle, `<prefix>_<modality>[_<stratum>]_<folder>_evidome_output.zip`.
There is no human-readable report, volcano, power file or workbook: OmicView
draws them all from the statistics zip.

**The machine tables** carry every feature and every column at full precision,
under generic names. A directional result: `tested`, `log2_fold_change`,
`control_mean` and `treatment_mean` (log scale), `raw_control_mean` and
`raw_treatment_mean` (linear), `statistic`, `df`, `p_value`, `q_value`,
`n_control`, `n_treatment`, the effect-size columns above, then the statistic's
own: `n_pairs`, `mean_within_pair_difference`, `sd_within_pair_difference`
(paired-t); `slope`, `slope_se`, `intercept`, `pearson_r`, `r_squared`,
`n_regressed` (lm); `standard_error` (an ANOVA contrast; `anova_cell` adds the
cell and sample counts). An F-test: `tested`, `log2_fold_change` (its
direction, where it has one), `F`, `df_term`, `df_error`, `p_value`, `q_value`,
and for a one-way ANOVA on numeric levels its trend and lack-of-fit columns.
The feature's identity and annotation columns come first; after the statistics,
each sample the comparison read, control group first (every sample, for an
F-test), as `s:<sample>`. A missing value is blank.

**The Evidome table** (`<prefix>_<folder>_evidome_input_<stamp>.csv`) is
everything the statistic produced, one row per gene and one block of columns
per comparison, unfiltered. An ANOVA's table holds its F-tests and its
contrasts together, so Evidome can give each F-test its direction from its own
contrasts. A peak becomes its gene -- the gene whose TSS it overlaps, else its
nearest gene -- and the lowest-p peak stands for its gene in each comparison.
An `lm` slope is written as the log2 fold change. It starts with
`# generated by paralome <version>` comment lines above the header: pandas
takes `comment='#'`.

**The manifest** lists every file the folder holds, and the stage's checkpoint
carries the same under `statistics`. A run removes what the previous run of the
same prefix wrote there and this one did not, and says so, so every file in a
folder belongs to the run its checkpoint describes. A statistic dropped from
`--statistic` has its folder's files removed by the next run.

### Bulk RNA's expression gate: an export rule, not a test

`--rna-min-expression` and `--rna-min-expression-difference` (raw FPKM;
`rna.min_expression` and `rna.min_expression_difference` in the configuration,
5.0 and 5.0 out of the box). A gene of a two-arm comparison passes the gate
only when the raw FPKM mean of the level or of its control
reaches the first and the two raw means differ by at least the second. Every
gene is still tested and stays in every CSV, and the machine table says per
gene whether it passed (`passes_expression_gate`). A comparison with no two
groups -- an F-test, an `lm` slope -- is not gated. Both
at 0 is no gate. Each takes one value.

### Motif enrichment: one per statistic folder and comparison

On ATAC and snATAC the motif scan is done once, and the enrichment (Fisher's
exact) is run for each comparison of each statistic folder, and each stratum.
Each comparison's peaks are classified from its machine table by `--alpha`
(default p < 0.05): untested peaks are left out; a peak under the cut with a
positive `log2_fold_change` is up, negative is down; the rest are neither.
Every comparison an ANOVA made is classified: its contrasts, and its F-tests by
the direction their `log2_fold_change` carries (a two-level main effect's from
its main-effect contrast, a 2x2 interaction's from its coefficient); an F-test
of a term with more than one column has no direction and is left out. An `lm`
comparison is classified by the sign of its slope.

The results go to `motif_analysis/results/<folder>/` (bulk:
`<project>/motif_analysis/`; snATAC: `<project>/atac/motif_analysis/`), each
comparison's tables in `results/<folder>/machine/`. The stage reads the
differential's checkpoint, which says what every comparison compared, so
`paralome snATAC motifs` given `--differential-checkpoint` refuses
`--differential-statistics` beside it. Motif partners keep each statistic
folder's tables apart the same way.

### Evidome: one run per statistic folder

`functional_evidence` runs Evidome once per statistic folder (and stratum), on
that folder's Evidome table, into
`functional_evidence/<folder>/<prefix>[_<stratum>]_<folder>/`, with a log each. The
stage is finished when every run's archive is there; OmicView's Evidence
viewer opens the archive. DNA's gene burden is one table and one run. Each
finished archive is moved into the project's `omicview/functional_evidence/`.
When Evidome runs as SLURM jobs, one more small job waits for them and does the
move, so `paralome olink` and `paralome proteomics`, which end once Evidome is
submitted, get their archives moved too. Its log is
`functional_evidence/.../logs/place_archives_<job>.out`;
`paralome-worker evidence place --out <stage folder> --zip-dir <omicview
folder>` does the same move by hand.

### `log2_fold_change`

What `log2_fold_change` is depends on the statistic -- the column name is the
same, the quantity is not:

| folder | `log2_fold_change` |
|---|---|
| `t` | mean log2 treatment − mean log2 control: the ratio of *geometric* means |
| `paired_t` | the mean of the pairs' treatment − control differences |
| `lm` | the slope: log2 change per unit of the covariate |
| `anova` | a contrast's estimate; an F-test's direction |
| `t_cell`, `paired_t_cell`, `lm_cell`, `anova_cell` | the same, from per-cell values |

In each case it is the effect the p-value beside it describes, on the same log
values the test used. `raw_control_mean` and `raw_treatment_mean` are on the
linear scale: raw FPKM for bulk RNA, CPM for bulk ATAC and the single-cell
pipelines (pseudobulk CPM per sample; for a `:cell` statistic the means are
over cells). They are arithmetic means, so their ratio is not the fold change,
which comes from the log values.

**One caveat, stated because it is real.** The matrix is stored as
`log2(x + pseudocount)` and the pseudocount does not cancel, so an exact
doubling from 10 → 20 FPKM gives a log2 fold change of 0.933, not 1. At
500 → 1000 it is 0.999. The compression is worst for features near the
pseudocount -- which is also where the test is least trustworthy. Removing it
would put the reported effect out of step with its own p-value.

**One caveat on `t_cell`.** Its fold change is computed from per-*cell* mean
log2 CPM rather than per-sample, which makes it dominated by the fraction of
cells expressing the gene rather than by expression level. It can be several
times the pseudobulk value for the same gene and is not the same quantity. It
exists as a pseudoreplication baseline, not for reporting effect sizes.

---

## Appendix A — Error codes

Every numbered failure Paralome can report. The console prints the code,
what went wrong and the specifics; the fix and the rest are here, in
the run's log, and from `paralome explain`.

`paralome explain 104` prints any entry below without leaving the terminal.

Numbers are grouped in blocks of a hundred. The block says where the failure
lives: what you typed, what your data is, how it ran, or how the machine is
set up. New codes append to a block and never renumber an existing one.

### General

#### PLM-0001 — Unexpected internal error

**Fix:** Send <project>/logs/paralome.err (else ./paralome.err) to the maintainer

Paralome hit a failure it has no number for. That is a gap in this catalogue, not necessarily a bug in your command -- but the software cannot tell you which, so it stops rather than guess.

The full traceback is in the log file; the console does not name it. The log is <folder>/logs/paralome.err when the command was given --project or --out and that folder exists; otherwise it is paralome.err in the directory the command was run from. The traceback is not printed to the terminal because it is diagnostic material for whoever maintains the software, and burying the one actionable line in eighty lines of stack frames is how the previous behaviour hid it.

If your command looks correct, this is worth reporting: every reproducible failure should eventually have its own number and its own instruction.

### What the command said (0100-0199)

#### PLM-0101 — A required option was not given

**Fix:** Add the option(s) named in the message and re-run

Each analysis has a small set of options it cannot proceed without. For scRNA and snATAC those are --rds and --replicate-column, plus --project for a full pipeline run; --factor COLUMN:CONTROL is needed by every statistic that compares levels (t, paired-t and anova).

--genome belongs to that set for every analysis that reads a genome. Paralome never chooses one for you, and a genome named in the configuration is not used as a default: the wrong genome gives results that look fine. The message lists the genomes configured.

Options NOT in that set are genuinely optional, including --stratify-by. If you were told one of those is required, that is a bug -- report it.

#### PLM-0102 — That option belongs to a different analysis

**Fix:** Single-cell analyses take --rds; bulk analyses take --samples

Bulk analyses (RNA, ATAC, DNA) start from FASTQ files and are handed a sample manifest with --samples. Single-cell analyses (scRNA, snATAC) start from an already-processed Seurat/Signac object and are handed it with --rds.

Passing both, or passing the wrong one, is refused rather than silently ignored: an ignored option looks like it took effect.

#### PLM-0103 — Two options were given that contradict each other

**Fix:** Pass one or the other, not both; the message names the pair

Two of the options given cannot both take effect, and neither is guessed at. The message names the pair. The cases:

--stratify-value names the strata to RUN and --stratify-skip the strata to EXCLUDE. Given both there is no defensible interpretation. Omitting both runs every stratum, which is usually what you want.

--submit with --backend local. Submission is what the SLURM backend does; a local run has nothing to submit. Drop --submit, or run with --backend slurm.

On the genome build commands, a SLURM-only option (--partition, --time, --max-parallel) without --backend slurm, or --threads, which is this machine's thread count, with it. Accepted, the option would do nothing, which looks like it took effect. The message says which option and what it needs.

#### PLM-0104 — An input file does not exist

**Fix:** Check the path and spelling; the log names the directory searched

The file named by --rds, --samples or --reference is not on disk at the path given. Paths are resolved relative to the directory you ran the command from, not to --project.

This is checked before anything is submitted, so a missing file is found here rather than by a compute node a poll cycle later.

Renaming an object and not updating the script is the single most common cause, so when a similarly-named file sits in the same directory it is recorded in the log and offered as the suggested fix.

#### PLM-0105 — An input file exists but cannot be read

**Fix:** Check permissions; the log reports the owner, the mode and your uid

The file is present but the user running Paralome cannot open it. On a shared cluster this is usually a file created by another account, or one inside a directory whose execute bit is not set for you.

The log records the file's owner, its mode, and the account you are actually running as, because 'permission denied' without those three facts cannot be acted on.

#### PLM-0106 — The input file is not the format it claims to be

**Fix:** Convert it; `paralome explain 106` prints the exact command

A .rds extension does not make a file an RDS. The two common cases:

save() instead of saveRDS(). This writes an RData WORKSPACE -- a bag of named objects -- not a single serialised object. Same extension, similar size, and R itself reports only 'unknown input format'. Paralome detects it by magic bytes (RDX2/RDX3/RDA2). The conversion, for a workspace holding one object:

    Rscript -e 'e <- new.env(); load("obj.rds", envir = e); saveRDS(get(ls(e)[1], envir = e), "obj_fixed.rds")'

A workspace holding several objects needs the right one named in place of ls(e)[1]; `ls(e)` lists them.

SaveH5Seurat() instead of saveRDS(). An HDF5 container. Detected the same way. In R, with SeuratDisk loaded: `saveRDS(LoadH5Seurat("obj.h5seurat"), "obj.rds")`.

#### PLM-0107 — The output directory cannot be written to

**Fix:** chown or chmod the directory; the console prints the exact command

Paralome checks the project tree, or the nearest existing parent of it, before starting.

The usual cause on a shared system is a directory tree created by root -- for instance when an administrator sets up a new user's folders and does not hand ownership over. The symptom without this check is brutal: the run fails, and it fails while trying to write its error into the directory it could not create, so there is no log to read and no message to see.

This is one of two errors allowed more than two console lines: the owner, the mode, the uid actually running and the chown that fixes it are printed, because a permissions error you have to go and diagnose is a permissions error you will get wrong.

#### PLM-0108 — That value is not one of the accepted choices

**Fix:** Use one of the values listed on the console

Options such as --statistic, --backend and --scanner-backend accept a fixed set of values. The full set is printed, and when your value is close to a real one it is suggested.

Paralome is case-sensitive throughout and has no aliases, by design: a silent alias is how two spellings of one concept become established.

#### PLM-0109 — --alpha is missing its metric letter

**Fix:** Write Q0.05 or P0.05, not 0.05

The significance threshold carries WHICH statistic it applies to. Q0.05 cuts on the FDR-adjusted q-value; P0.05 cuts on the raw p-value.

The letter is mandatory because the two are very different decisions and a bare 0.05 does not say which was intended. P is a deliberate choice with consequences, not a default to arrive at by omission.

#### PLM-0110 — A memory value is not in a form SLURM understands

**Fix:** Use values such as 4G, 32G or 32000M

Memory is passed to SLURM as a number with a unit suffix. A bare number, or a unit SLURM does not recognise, is rejected here rather than at submission time, where the failure would be a queue rejection with no indication of which stage's setting was malformed.

#### PLM-0111 — --stage-mem names a stage that does not exist

**Fix:** Check the stage name against the analysis's stage list

--stage-mem takes STAGE=MEM. The stage must be one in the stage graph for the analysis you are running; the valid names are printed.

A misspelled stage name would otherwise be accepted and silently do nothing, and the stage would run with its default allocation -- which presents much later as an out-of-memory kill.

#### PLM-0112 — A numeric option is outside its allowed range

**Fix:** Use a value within the range printed on the console

--threads and --jobs must be at least 1. --alpha must be strictly between 0 and 1. Cell and sample floors must not be negative.

Zero threads is the notable one: it is accepted by most tools as 'auto' and by Paralome as an error, because on a shared cluster the number of CPUs a job may use is a scheduling decision, not something to be inferred from the machine it lands on.

#### PLM-0113 — --factor is malformed

**Fix:** Use --factor COLUMN:CONTROL, e.g. --factor Diet:CD

A --factor names a sample-sheet column and its control level, joined by a colon: --factor Diet:CD, --factor Day:1, --factor Sex:F. Every other level of the column is compared with the control level, and that is what gives every result its direction, so a --factor without a control level is refused rather than given one.

The control level must be a value the column holds; the message lists the levels it does hold. To compare two non-control levels with each other, run again with one of them as the control.

#### PLM-0114 — --rna-strandedness must be 0, 1, 2 or auto

**Fix:** 0 unstranded, 1 first-strand, 2 second-strand, auto to detect it

The numbers mirror the featureCounts convention. Getting the direction wrong silently costs most of the reads, which looks like a biological result rather than a configuration error.

`auto` is the default. It checks a small panel of genes against the largest BAM and prints what it decided and why. A library whose direction cannot be told is counted as unstranded (0), and that is printed and recorded beside the run: counting unstranded keeps every read, where a wrong direction loses them.

#### PLM-0115 — This command needs a terminal, and this is not one

**Fix:** Re-run it from a login shell, or pass the flag named in the message

The command was about to ask you something -- which assay to use, whether to register, whether to withdraw for everybody -- and nothing is attached that can answer.

This happens under SLURM, in a cron job, or with input redirected from /dev/null. In all three a prompt would hang the job rather than fail it, or be answered with an end-of-file that aborts the command half-way through something it had already started.

NOTHING WAS WRITTEN. Every one of these checks runs before the first write, so there is no half-finished state to clean up and re-running is always safe.

The console line names the flag that drives this particular command without asking -- --yes for a registration, a rename or a fatal withdrawal, --force for a removal, --dry-run for a sync. They are per command because the alternatives differ: a registration can be driven entirely by flags, a sync cannot, because it is a conversation about specific genomes by design.

Interactivity is judged on BOTH streams, not on stdin alone. A job with input from /dev/null has a perfectly readable stdin that answers every prompt with EOF, and treating that as interactive is how the abort-half-way case arose.

#### PLM-0116 — This command is not meant to run inside a SLURM job

**Fix:** Run it on the login node; the analyses, batch and extract are submittable

The rule is that only the genomics pipelines, batch, and extract need to be submittable. Everything else -- setup, the genome registry, collect, the reports -- is something a person does at a prompt and reads the answer to.

THE ONE THAT MADE THIS NECESSARY is `paralome setup`. Your configuration lives in ~/.paralome/config.json, on shared storage, and every node sees the same file. The tool paths recorded in it are what every run actually invokes. So `paralome setup` inside a job records THAT COMPUTE NODE's paths into the file the login node reads -- silently repointing every future run, yours and everyone else's using that account, at one node's binaries. If they differ you get PLM-0410 everywhere; if they do not differ you get no error and no record that anything moved.

The rest are refused for a smaller reason: they ask questions, write to a shared registry, or print a report nobody is watching. A job is not where any of that belongs.

`batch` IS submittable, and it invokes the human command for each family -- which is why the analyses are allowed rather than the worker binary alone. `extract` is submittable too: reading a large object is work for a compute node, and it asks no questions.

Detected from SLURM_JOB_ID. Running the same command on the login node does exactly what you meant.

#### PLM-0117 — This run is not ready to start

**Fix:** Fix everything listed above, then re-run; nothing was submitted

Checked on the login node before anything was queued: that every sample is in place and not empty, and the logistics -- the genomes, the programs -- are there.

EVERY PROBLEM IS LISTED, not the first one, as in the genome doctor: a verdict that names one cause invites fixing that cause and expecting to be done. A preflight that stopped at the first bad FASTQ would cost a fix-and-resubmit cycle for every bad row in the sheet.

WHAT IS CHECKED. The sample sheet is OPENED, not merely found: every FASTQ or BAM it names has to exist and be non-empty, sample ids have to be unique. --genome has to resolve and the files it names have to be on disk. Any tool whose path `paralome setup` recorded has to still be where it was recorded, or be found on this account's PATH, where it is then recorded. On a machine with a shared root, checked first of all: the account has to be in the group of the shared genome repository, which an administrator fixes with usermod -aG <group> <account> and a fresh login.

WHAT IS NOT. Whether a tool is missing altogether is `paralome setup doctor`'s question, and it depends on which stages your profile runs -- freebayes absent is fatal to DNA and irrelevant to scRNA. A stale RECORDING is different: it breaks whichever profile uses that tool, and it is PLM-0410 arriving here instead of at the stage that reaches for it.

NOTHING WAS SUBMITTED AND NOTHING WAS WRITTEN. The whole point is that this runs before the first sbatch, so there is no queue slot to release and no half-finished project to clean up.

#### PLM-0118 — --rds names neither a .rds object nor a folder

**Fix:** Write one object as name.rds (or .qs2), or a folder of objects with a trailing /

--rds takes a single Seurat object or a folder of objects. The way the path is written decides which, and the disk must agree: a name ending in .rds (or .qs2), in any case, is one object and must be a file; a name ending in / is a folder and must be a directory (even my.rds/); anything else is refused, because a guess here decides what the whole run analyses.

Tab completion adds the / to a directory, so the folder form comes naturally at the prompt. From a folder every file ending in .rds is read; subfolders are not, and every other file is left alone.

#### PLM-0119 — --group-by is retired

**Fix:** Name the sample column with --replicate-column

The user names one column, --replicate-column: the column whose value is a cell's biological sample (library, animal, donor). The pseudobulk samples are its values, one per --stratify-by value when that is given -- the samples --group-by <replicate>,<stratum> made.

The --factor columns are not part of the grouping. Each must be one value per sample; a column that is not is refused (PLM-0203). --replicate-column is required on every single-cell run, pseudobulk and :cell alike.

#### PLM-0120 — The Excel workbooks are retired

**Fix:** Leave out the workbook option; the TSVs and CSVs hold every row

Paralome writes no .xlsx files: not the statistic folders' results workbook, not the motif workbook and not the DNA report's client or bucket workbooks. OmicView reads the run's results (the statistics output) and its variant browser reads the segregation table; every row a workbook held is in the CSVs and TSVs beside it.

So --workbook/--no-workbook, --bucket-workbooks and the older --max-workbook-rows and --bucketed-workbooks are refused by name rather than ignored.

#### PLM-0121 — The trajectories need a root, given as <column>:<value>

**Fix:** Add --trajectory-root <column>:<value> (e.g. celltype:RPC), or --no-trajectories

The trajectories stage (scRNA) fits the object's lineages with Slingshot, starting from the cluster the run names. Which cluster is the start is the analyst's knowledge, so it is never guessed: a run that reaches the stage names it, or leaves the stage out with --no-trajectories and adds it later by running again with --trajectory-root once the UMAP has been seen. The column is the cell-metadata column whose clusters the lineages are fitted on (celltype, seurat_clusters ...); the value is one of them.

#### PLM-0122 — --trajectory-bins must be 2 or more

**Fix:** Give --trajectory-bins 2 or more (the default is 5)

Each lineage is cut into this many segments of equal numbers of its cells by pseudotime, and each replicate's share of the lineage's cells in each segment is compared by the run's statistics (the composition only). One segment would be the whole lineage.

#### PLM-0123 — --trajectory-fit-cells is not a setting it knows

**Fix:** Give e.g. 100:random, or cells=1000,dispersion=random,stretch=0,extend=n

The lineages' curves are fitted on n cells of each cluster (every cell of a cluster smaller than n), drawn at random (seeded, so a run repeats) or the n nearest the cluster's centre in the reduction the fit uses; every cell is then placed on the fitted curves (slingshot's predict). Without the option every cell is fitted. slingshot.json records the cells per cluster, how they were chosen and how many were fitted. As key=value pairs separated by commas -- cells=<n>, dispersion=random|centroid (random when cells is given alone), stretch=<s> (0 or more: how far each curve runs past its last cluster; Slingshot's default 2), extend=y|n|pc1 (how a curve's ends are drawn; default y), reduction=<name> (the reduction the curves are fitted in, as the extraction named it, e.g. umap; by default the object's integrated one, else pca). Each key once; dispersion needs cells. Df=<n> (more than 1: the smoothing spline's degrees of freedom for every curve; 2 is close to straight; by default smooth.spline chooses them) and shrink=<0-1> (how far the lineages' shared start is pulled together; Slingshot's default 1). under --trajectory-method monocle3 (the default) the keys are Monocle3's -- reduction=<name> (the 2D/3D UMAP the graph is learned in), ncenter=<n>, minimal_branch_len=<n>, use_partition=y|n; also close_loop=y|n, geodesic_distance_ratio=<x>, euclidean_distance_ratio=<x>, orthogonal_proj_tip=y|n, prune_graph=y|n, l1_gamma=<x>, l1_sigma=<x> (x more than 0) -- and Slingshot's are refused, as Monocle3's are under slingshot.

#### PLM-0124 — --trajectory-dims is outside the reduction's dimensions

**Fix:** Give 2 up to the number of dimensions the error names

The lineages are fitted on the first n dimensions of the reduction the stage chose (an integrated one, else pca; snATAC's lsi without its first). Every distance the fit measures -- between the clusters' centres, and from each cell to each curve -- is measured in those dimensions. Without the option all of them are used.

#### PLM-0125 — The trajectory method is unknown, or the option is not its

**Fix:** Give --trajectory-method monocle3 or slingshot; leave --trajectory-dims out with monocle3

The lineages are fitted by Monocle3 (the default: one principal graph learned through the cells in the object's UMAP, each path from the root to a leaf of the graph a lineage) or by Slingshot (curves through the clusters' centres, fitted in a reduction of many dimensions). --trajectory-dims chooses those dimensions, so it is Slingshot's; Monocle3 learns its graph in the UMAP.

#### PLM-0126 — The object has no UMAP for Monocle3 to learn its graph in

**Fix:** Name one with --trajectory-fit-cells reduction=<name>, or --trajectory-method slingshot

Monocle3 learns its principal graph in a 2D or 3D UMAP: the object's (a 2D/3D reduction with umap in its name, or its only 2D/3D one), or the one reduction=<name> names. The error lists the object's 2D/3D embeddings.

#### PLM-0127 — A --trajectory-weld cannot be made

**Fix:** Name tips: L<a>=L<b>, L<a>=L<b>=L<c>, L<a>~L<b> or 'L<a>->L<b>'

--trajectory-weld joins tips of Monocle3's learned graph, named as its lineages are in <run>/trajectories/fit/tips.tsv: L39=L42 joins the two tips, L30=L33=L35 joins each to a new node at their middle, L39~L42 joins L39's tip to the nearest point of L42's own branch, 'L39->L42' joins L39's tip to L42's and L42's stays an end (quoted: unquoted, the shell reads > as a redirect). A tip the graph does not have is refused (the error says how many it has). The lineages are every route from the root to each end; welds that make more than 200 are refused: weld fewer tips, or with ~.

#### PLM-0128 — --trajectory-model is not an engine and its df

**Fix:** Give ls or monocle3, the spline's df after a colon: ls:df=4

Each gene along each lineage's pseudotime is modelled as the run's full factorial crossed with a natural spline of pseudotime: by least squares on the cells' log2(CPM+1) (ls, the default) or by Monocle3's fit_models on the counts (monocle3). df is the spline's degrees of freedom, 3 unless given; 1 is a straight line.

#### PLM-0129 — --trajectory-lineages is not fate, exit or route

**Fix:** Give --trajectory-lineages fate, exit or route

The trajectory statistics follow one lineage per end state (fate, the default: every route the fit made that ends in the same cluster), one per way out of the root's cluster and end state (exit), or every route (route).

### The request versus the data (0200-0299)

#### PLM-0201 — A named metadata column is not in the object

**Fix:** Use one of the columns listed; `paralome extract` prints them all

--replicate-column, --stratify-by and the --factor columns all name columns of the Seurat object's cell metadata. The column you asked for is not among them.

The console lists every column that IS present, wrapped rather than truncated -- a list whose answer might be in the elided part is not worth printing. This is the one error allowed more than two lines, because the alternative is opening cell_metadata.tsv by hand.

When an extraction already exists for this object, this is checked before submission. On a first run the columns are not knowable until the object has been read, so it is checked inside the job, at the earliest point the answer exists -- before any aggregation work.

`paralome extract` prints every metadata column with its distinct values, which is the intended way to choose these before committing to a run.

#### PLM-0202 — --replicate-column does not name one column

**Fix:** Name the one column whose value is a cell's sample, e.g. --replicate-column orig.ident

--replicate-column defines what a pseudobulk sample IS: cells sharing its value are summed into one (one per --stratify-by value when that is given). It names exactly one column; a second grouping is --stratify-by.

#### PLM-0203 — A --factor is not one value within a pseudobulk sample

**Fix:** Check that --replicate-column names the sample column, or fix the metadata

A factor describes a sample, so every cell aggregated into one pseudobulk sample must carry the same value of it -- otherwise the sample sits in two levels at once and no comparison of the levels means anything.

The usual cause is a --replicate-column that is not the sample column -- a cell type, a cluster or a batch that holds cells of both control and treated samples -- or metadata that gives one sample's cells two levels of the factor.

#### PLM-0204 — --stratify-by is not constant within a pseudobulk group

**Fix:** Check --replicate-column and --stratify-by

A stratum is analysed separately, so each pseudobulk sample must belong to exactly one. The --stratify-by column is part of every sample's grouping, so this is reached only when the metadata itself is inconsistent.

#### PLM-0205 — Named samples matched nothing

**Fix:** The message lists the values that do exist

A sample named on the command line or in the sample sheet is not in the data it has to be found in -- a VCF's samples, a manifest's sample-ID column or its full sample_ids.

Most often a typo, or a value from a different column than the one you had in mind. The message prints what is there, so the right spelling is in front of you.

#### PLM-0206 — The samples in scope matched no cells

**Fix:** Use the sample_manifest.csv written from this extraction

The cell-level statistics find each pseudobulk sample's cells by the metadata values the sample was aggregated from. None were found here: the manifest's samples are drawn from values the cell metadata does not hold, which means the manifest and the extraction are not from the same object.

#### PLM-0207 — --remove excluded every sample

**Fix:** Remove fewer samples

Every pseudobulk sample was excluded, leaving nothing to compare.

--remove matches a sample by its sample-ID value or its full pseudobulk sample_id -- both are tried, so a sample-ID value removes that sample from every cell type.

The console lists what was removed. Naming full sample_ids is the unambiguous form.

#### PLM-0208 — A level of a --factor has no samples

**Fix:** Check the --factor control level and the column's values against the sample manifest

A comparison needs samples on both sides. One level resolved to nothing, which means either the value is not in the column or every sample holding it was excluded by --remove or by a cell-count floor.

#### PLM-0209 — Every group fell below the minimum cell count

**Fix:** Lower --min-cells-per-group, or stratify less finely

Groups with too few cells are dropped, because a pseudobulk value computed from a handful of cells is mostly noise. Here every group was dropped.

A grouping that is too fine is the usual cause: --stratify-by multiplies the number of groups and divides the cells among them.

#### PLM-0210 — The sample manifest is missing a required column

**Fix:** Point at data/<modality>/sample_manifest.csv from a completed run

The file given to --samples is not a Paralome pseudobulk manifest, or it predates the current layout. The expected columns are printed alongside the ones the file has.

The manifest a differential stage wants is the one written by the aggregation stage, at data/<modality>/sample_manifest.csv. Pointing at a hand-made sample sheet, or at the bulk manifest handed to --samples on a FASTQ-based analysis, produces this: they are different files with different columns and the same intuitive name.

#### PLM-0211 — The extraction's cell files disagree with each other

**Fix:** Re-run the extraction with --overwrite

cells.tsv and cell_metadata.tsv must list the same barcodes in the same order; they index the same count matrix columns.

Disagreement means the extraction directory holds files from two different runs -- usually an interrupted extraction, or two runs writing to one directory. Nothing is salvageable by hand; re-extract.

#### PLM-0212 — The extraction's peak files disagree with each other

**Fix:** Re-run the extraction with --overwrite

peaks.bed and peaks_order.tsv must agree on the peak order, since both describe the rows of counts.mtx. The redundancy is deliberate: two files stating the same order means a truncated or interleaved write can be caught instead of silently shifting every peak's counts.

Same cause and same fix as PLM-0211 -- the directory holds files from two different runs. Re-extract rather than repairing by hand.

#### PLM-0213 — The object has no assay of the kind this analysis needs

**Fix:** Use the matching analysis, or name the assay with --assay / --rna-assay

snATAC requires an ATAC half; scRNA requires an RNA half. The object has the other one, or neither.

Extraction itself takes whatever the object has -- both halves of a multiome come out of a single read. It is the ANALYSIS that has a requirement, and it says so here rather than failing several stages later on a missing counts file.

If the assay is present under an unusual name, --assay and --rna-assay name it explicitly. Note that `integrated` and `SCT` are never auto-selected: they are derived values, not counts.

#### PLM-0214 — No ATAC or RNA assay could be found in the object

**Fix:** Name the assays with --assay and/or --rna-assay

Nothing in the object was recognisable as either half. An ATAC assay is identified by being a Signac ChromatinAssay, by being named ATAC/peaks/macs2, or by having genomic-interval rownames. An RNA assay is identified by being named RNA.

The assays that WERE found are listed. Naming one explicitly overrides the detection entirely.

#### PLM-0215 — A multiome object's two halves disagree about their cells

**Fix:** Re-save the object with both assays over the same cells

The ATAC and RNA assays hold different barcodes, or the same barcodes in different orders. Both are indexed by one shared cells.tsv, so Paralome refuses rather than picking one.

This matters more than it looks: silently trusting one assay's order would misalign every barcode-keyed join downstream, and nothing later would raise. The result would be complete, plausible and wrong.

#### PLM-0216 — A named assay is not in the object

**Fix:** Use one of the assays listed, or omit the flag and let it be detected

--assay or --rna-assay named an assay this object does not have. The assays it does have are listed.

Usually the flag is unnecessary. Extraction detects assays structurally: features that are genomic intervals are ATAC, and everything else is gene-level -- RNA, Spatial, Xenium, ADT or whatever the object's author called it. The flags exist only to break a tie between several assays of the same kind.

#### PLM-0217 — More than one assay of the same kind

**Fix:** Name the one you want with --assay or --rna-assay

The object holds several assays that are all ATAC, or several that are all gene-level -- an RNA+ADT object, for instance. Both are legitimate; neither can be chosen for you.

No name settles it, not even RNA: picking one silently is how the wrong matrix gets analysed with nothing in the output saying so.

The candidates are listed. Note that derived assays are already excluded: an SCTAssay is recognised by its class, and an assay with an empty counts layer -- which is what `integrated` usually is -- cannot be aggregated and is skipped.

#### PLM-0218 — Every peak is wider than the motif scan allows

**Fix:** Raise --max-peak-width above the widest peak named in the message

Motif scanning fetches and scans the sequence under every peak, so its cost is proportional to total base pairs, not to peak count. --max-peak-width bounds that: a peak wider than it is SKIPPED, counted, and reported, and the scan proceeds on the rest.

This error means every peak was skipped, so there is nothing left to scan -- which is a different situation from an empty matrix, and the only one where --max-peak-width is the answer.

A single over-wide peak never aborts the analysis: it is skipped and counted. (On the snATAC path the peaks come from the user's Signac object, so rebuilding an atlas is not an action there.)

#### PLM-0219 — Too many peaks are on contigs the reference FASTA lacks

**Fix:** Register the genome the peaks were called on, or rename their contigs to the FASTA's

The motif scan fetches each peak's sequence from the reference FASTA by contig name, trying the chr-prefixed and unprefixed spellings and the mitochondrial aliases. A peak whose contig is still not found is skipped, counted and reported, like a peak wider than --max-peak-width: single-cell peak sets routinely carry a few unplaced scaffolds (KI270728.1, GL000194.1) that a UCSC-style FASTA names differently (chrUn_KI270728v1), and one such peak must not abort the whole stage.

Above one percent of the peak set, though, the missing names are not a few scaffolds but a naming convention: the peaks were called on a genome other than the one the motif stage was given. Scanning the rest would describe a different peak set from the one the differential tested, so the stage refuses and names the contigs. Skipped peaks are outside the enrichment universe: they are peaks that were not scanned, not peaks without a motif.

#### PLM-0220 — The differential table does not name its comparison

**Fix:** Pass --differential-checkpoint, or a table with control_condition/treatment_condition

Given a differential CHECKPOINT (what the pipeline passes), the motif stage runs one enrichment per statistic folder and comparison the checkpoint records, each named after the comparison as the differential named it; a second table (--differential-statistics) beside it is refused.

Given ONE table (--differential-statistics), there is one comparison and the results are named after it, from the table's own control_condition/treatment_condition columns. A table without them is refused rather than have its comparison named from the sample manifest's order, which reads 'up' backwards when the manifest lists the treated level first.

#### PLM-0221 — The requested statistics do not fit these samples

**Fix:** The message names the column, level or sample; check --factor, --pair-by and --covariate

Each statistic in --statistic is planned against the sample sheet before anything is fitted: anova needs every --factor column present -- one factor of three or more levels (the one-way ANOVA), or two or more factors with samples in every combination of their levels -- and at least one error degree of freedom at pseudobulk; a factor of more than two levels must be numeric, so recode one that is not; paired-t needs each --pair-by pair to hold one sample in each arm; lm needs a numeric --covariate that varies; and every arm needs enough samples for the statistic's degrees of freedom. A request the samples cannot support is refused here, with the reason, rather than fitted into results that would not mean what their names say.

Nothing was written. Correct the option or the sample sheet and run the same command again.

#### PLM-0222 — The --rds folder holds no .rds object

**Fix:** Point --rds at the folder that holds the objects, written with a trailing /

A folder given to --rds is read for files ending in .rds, in any case, directly inside it. Subfolders are not read, and other files are left alone, so a folder of objects one level down, or of .qs2 files, holds none.

#### PLM-0223 — Two objects in the --rds folder give the same barcode prefix

**Fix:** Rename one of the files named in the message

Barcodes repeat between samples, so each cell's barcode is prefixed with its file's name without .rds: HPC_03.rds gives HPC_03_<barcode>. Two files that differ only in the case of .rds (x.rds, x.RDS) would give the same prefix, and cells from different objects could then share a name.

#### PLM-0224 — Objects in the --rds folder lack a metadata column

**Fix:** Add the column to each object named, or change --replicate-column/--stratify-by

The command line is the authority: every object in the folder must carry the columns --replicate-column and --stratify-by name. Each file is checked on its own, every failing file is listed with the column(s) it lacks, and nothing is stacked or analysed. No file is treated as the reference, because the first one read could be the broken one.

Columns that only some objects carry, and that the command does not name, are left out of the combined metadata and listed.

#### PLM-0225 — Objects in the --rds folder do not have the same genes

**Fix:** Decide which file is wrong from the groups listed and fix or remove it

The objects are grouped by identical gene list, and every group is reported: its files, its number of genes, the genes it has that another group lacks and the reverse (the first few on screen, every one in rds_folder_gene_groups.tsv in the extraction folder), and its match rate against the --genome GTF, as evidence rather than a ruling. Paralome never picks a winner.

Genes are compared by Ensembl ID, and a gene the GTF gave none by its feature name. Matching by symbol silently pairs or drops genes renamed between annotation releases (Qk/Qki, Sept7/Septin7).

Each object's extraction is kept, so once the file is fixed a rerun re-extracts only that file.

#### PLM-0226 — Objects in the --rds folder do not have the same peaks

**Fix:** Decide which file is wrong from the groups listed and fix or remove it

As PLM-0225, for the ATAC half: the objects are grouped by identical peak set (chromosome, start, end), every group is reported, and the full lists are in rds_folder_peak_groups.tsv in the extraction folder. Peaks called separately per object differ, and counts on different peaks cannot be stacked into one matrix.

#### PLM-0227 — The mass-spec export is not a layout Paralome reads

**Fix:** Give the exporter's own table as exported: proteins, or phosphopeptides for --phospho

Each exporter's table is recognised from its header: Proteome Discoverer protein and phosphopeptide exports, MaxQuant proteinGroups.txt, FragPipe combined_protein.tsv and DIA-NN pg_matrix.tsv. A table none of them claims is refused rather than read by guessing which columns are samples. The message names the file and the first columns found. A protein table given as --phospho (or the reverse) is refused the same way.

#### PLM-0228 — The sample sheet does not match the export

**Fix:** Make the sheet list every abundance column once, with a sample_id each

The sample sheet has one row per sample: `column` (the abundance column's header, or the sample name inside it) and `sample_id`, then any metadata columns. Every abundance column of the export must be in the sheet and every row must name one. Every fault is listed at once, with the sample names the export holds. An option naming a sheet column (--factor, --stratify-by, --pair-by, --covariate) that the sheet does not have is refused here too.

#### PLM-0229 — Whether the abundances are already logged cannot be told

**Fix:** Export the values with a header that says log2 (or log10), or export them linear

Paralome applies log2(x + 1) to linear values, so a logged table must be recognised. The header is read first; then the values: a negative value exists only on a log scale and no log2 intensity reaches 100. A table whose values all stay under 16 with a silent header could be log10, log2 or small linear numbers, and is refused rather than assumed. So is a table that contradicts itself (negative values that also reach 100, or a log header over values that reach 100).

#### PLM-0230 — No stratum can be tested

**Fix:** Read each stratum's reason and change what it names

With --stratify-by, every stratum is tested on its own, and one that cannot be is skipped with its reason: an arm with too few samples, samples under --min-cells-per-group, or a column the statistic names (--covariate, --factor, --pair-by) that the sample manifest does not have. When every stratum is skipped there is nothing to test, and the message lists each one with its reason.

#### PLM-0231 — A fragment file the Signac object names cannot be found

**Fix:** Fix the path in R with Signac's UpdatePath and save the object, or move the file there

The snATAC genome tracks (the genome_tracks stage) draw each pseudobulk sample's profile from the fragment files the Signac object names. The object holds only each file's path and which cell is which barcode in it; the fragments stay in the file (fragments.tsv.gz, beside its .tbi index), and Paralome reads it from that path. A path written where the object was built, on another machine or before the data moved, is not there.

The message lists every fragment file the object holds, found or missing. In R: Fragments(obj)[[i]] <- UpdatePath(Fragments(obj)[[i]], new.path = '<real path>') for each missing one, then saveRDS(obj, ...) and run again. An object that names no fragment file at all (its ATAC assay was built from a counts matrix alone) has nothing to draw profiles from.

#### PLM-0232 — The differential tested no peak

**Fix:** Look at each comparison's tested count in the differential's results

The genome tracks (the genome_tracks stage) export the viable peaks: the union of the peaks tested in any comparison, where a peak is tested when it reaches the expression floor (--min-mean-cpm) over that comparison's samples. A differential that tested none leaves nothing to export; the floor, or the samples each comparison reads, is the place to look.

#### PLM-0233 — The fragments are not sorted by contig

**Fix:** Sort the BAM by coordinate (or the fragment file) and index it

The genome tracks (the genome_tracks stage) read a sample's fragments one contig at a time, so it holds only that contig's coverage. A coordinate-sorted BAM and a tabix-indexed fragment file both come contig by contig; a contig that comes again after another means the file is not sorted.

#### PLM-0234 — The trajectory root is not in the object's cell metadata

**Fix:** Name a column the object has and one of its values (the error lists them)

--trajectory-root <column>:<value> is read against the extraction's cell_metadata.tsv: every metadata column the object holds. The column must be there and at least one cell must have the value, spelled as the object spells it.

#### PLM-0235 — The object holds no reduction the lineages can be fitted on

**Fix:** Run PCA (scRNA) on the object, or an integration, save it, and run again

The lineages are fitted on the object's own reduction: an integrated one (a name with harmony or integrated) when it has one, else pca. Only a reduction of more than three dimensions is used; a UMAP or t-SNE is for drawing the curves, not fitting them. The extraction writes every reduction the object holds (extraction/reductions/).

#### PLM-0236 — No trajectory replicate is a sample of the run's pseudobulk

**Fix:** Check --replicate-column and --remove: they decide the run's samples

The composition of each replicate along and between the lineages is compared with the run's own statistics, so each replicate must be one of the run's pseudobulk samples (its manifest gives the replicate's factor levels).

#### PLM-0237 — The trajectories stage does not run on this modality yet

**Fix:** Run it on an scRNA analysis, or add --no-trajectories

Paralome builds the trajectories for scRNA objects: the genes along each lineage are the scRNA extraction's, and the composition is tested by the scRNA differential. An snATAC run has no trajectories stage.

#### PLM-0238 — The trajectories were given no statistic to run

**Fix:** Pass the run's --statistic

The trajectories' composition is tested by the run's own --statistic, at the replicate level (the genes along each lineage by --trajectory-model). With none there is nothing to run; the pipeline always passes the run's, so this is met only by running the worker command by hand.

#### PLM-0239 — The --covariate-feature gene is not one gene of the object

**Fix:** Name the gene as the object's features name it (its symbol or its Ensembl ID)

With lm on a gene (--covariate-feature), the trajectories' composition (each replicate's shares of cells along and between the lineages) is regressed on that gene's expression in each replicate, every cell of the replicate summed. The gene must be exactly one of the object's genes.

#### PLM-0240 — The --plate file is not an Olink plate export

**Fix:** Give each plate as Olink exported it (NPX Signature long format), once per --plate

paralome olink reads Olink's long export by its column names: PlateID, SampleID, SampleType, Assay, UniProt, OlinkID, QuantifiedValue, BelowLOD and Unit, one row per sample and assay. CSV (comma, semicolon or tab), .xlsx and .parquet are read. A PlateID may be in only one --plate file.

#### PLM-0241 — A plate shares no sample control with the first plate

**Fix:** Put first the plate that shares controls with every other, or give the plates that do

Every plate after the first --plate is brought onto the first plate's scale, assay by assay, by the median ratio of the SAMPLE_CONTROL wells both plates ran (matched by SampleID). A plate with none in common with the first cannot be bridged to it.

#### PLM-0242 — The Olink sample sheet cannot be read

**Fix:** Give a CSV or tab-separated sheet with PlateID, SampleID and sample_id, one row per well

Each row is one well of a --plate (PlateID and SampleID as the export gives them) and the sample it is (sample_id), then the design's columns. A well is listed once; wells given the same sample_id are averaged into one sample.

#### PLM-0243 — The Olink sample sheet does not match the plates

**Fix:** List only SAMPLE wells of the given plates; give averaged wells the same design values

The sheet's wells must be in the plates, and be study samples (the SAMPLE_CONTROL wells bridge the plates and are not tested). Wells averaged into one sample_id must agree on every design column. The columns the options name (--factor, --pair-by, --block-by, --covariate, --stratify-by) must be the sheet's.

#### PLM-0244 — There is no trajectory fit for the trajectories stage

**Fix:** Run the pipeline again: singlecell_rna fits the lineages before the trajectories stage

The singlecell_rna stage fits the lineages, right after its extraction and pseudobulk, into <run>/trajectories/fit (an older run's is <run>/trajectories/slingshot), and writes them into the run's cell export; the trajectories stage then runs the statistics on that fit. Run by hand, the worker command names the fit's folder with --fit.

#### PLM-0245 — The genome's uniprot_table is not Ensembl's UniProt table

**Fix:** Put Ensembl's <Species>.<assembly>.<release>.uniprot.tsv.gz in the genome folder

Ensembl publishes each release's UniProt cross-references at ftp.ensembl.org/pub/release-<n>/tsv/<species>/ (e.g. Homo_sapiens.GRCh38.81.uniprot.tsv.gz); Paralome reads its gene_stable_id and xref columns to find a protein named by no gene symbol (an Olink assay such as Abeta40) by its UniProt accession. `paralome setup genome add <folder>` registers the one file in the genome folder ending .uniprot.tsv or .uniprot.tsv.gz; use the release of the genome's GTF.

### Execution (0300-0399)

#### PLM-0301 — A SLURM stage failed

**Fix:** Read the .err file named in the message -- that is the real error

The supervisor reports only that a job failed. The cause is in the job's own .err file, whose path is printed.

The stage's logs live under the stage's working directory, which for single-cell work is the SHARED data folder beside the projects, not inside the project directory -- looking in the project and finding no logs/ folder is expected, and is why the path is now printed rather than left to be deduced.

A failure is announced when it happens (`Failed so far: ...`), but the stage does not stop there. The tasks still running are left to finish and this error is raised once nothing in the stage is running, so one failed piece of a large array does not cost the rest. What finished is checkpointed: fix the cause and re-issue the identical command, and only the failed pieces run again.

#### PLM-0302 — A submitted job never appeared in the queue

**Fix:** Check sinfo and the partition name

The job was submitted and accepted, but neither squeue nor sacct ever reported it existing.

Usually a partition that does not exist, or one your account cannot submit to -- sbatch accepts the job and the scheduler discards it. `sinfo` lists the partitions and their states.

Paralome waits a grace period before concluding this, so a job that is merely slow to appear is not mistaken for one that never will.

#### PLM-0303 — SLURM accounting is not answering

**Fix:** Check that slurmdbd is running; re-run to resume

sacct stopped responding for longer than the grace period, so the supervisor cannot tell whether jobs finished. It stops rather than assume either way.

Work already completed is checkpointed. Re-running the same command resumes from the first incomplete stage.

#### PLM-0304 — A stage exited with an error

**Fix:** Read the stage output above; the log has the traceback

Running locally rather than under SLURM, the stage runs as a subprocess and its own output is already on your terminal, above this message. That output is the authoritative error; this line only reports that the stage did not succeed.

Work completed before the failure is checkpointed. Fix the cause and re-run the identical command to resume.

#### PLM-0305 — One or more strata produced no result

**Fix:** Read the .err for the named strata; the count is in the message

The fan-out runs each stratum independently and collects failures rather than stopping at the first, so one bad stratum cannot hide the others.

Under --backend slurm each stratum is its own array task with its own log, and the gather job runs on `afterany` -- so it runs even when some tasks failed, and its job is to name which. Seventeen of twenty succeeding is a partial result on disk, not a lost run: fix the three and re-issue the identical command.

When ALL of them fail identically, the cause is the request, not the data: a column name, an arm member, a model that cannot run on this design. The distinct errors are summarised rather than repeated once per stratum.

#### PLM-0306 — The log directory cannot be written to

**Fix:** chown or chmod the directory; the console prints the exact command

Paralome could not create or write its log directory. This is reported on the console rather than logged, for obvious reasons.

A run whose logger cannot write is a run whose failures will be invisible, so this is fatal rather than a warning.

#### PLM-0307 — The stage has more tasks than one SLURM job array can hold

**Fix:** Lower the number of pieces, or ask the administrator to raise MaxArraySize

SLURM limits the task numbers a job array may use. The highest allowed number is MaxArraySize minus one (`scontrol show config | grep MaxArraySize`), and Paralome numbers tasks from 1, so one array holds MaxArraySize minus one tasks.

The stage named in the message would need a task number above that. Nothing was submitted.

Stages that choose their own number of pieces stay inside the limit on their own. This error means the pieces are fixed by the data -- one per sample, or one per block of peaks -- and there are more of them than the scheduler allows in one array.

#### PLM-0308 — A stage that has to run again would keep its old result

**Fix:** Remove what the message names, then run the same command again

Something earlier in the run changed -- a stage before this one has to run again, or the options this stage is given are no longer the ones it ran with -- so the results the stages named hold no longer answer this run. The message gives the reason for each: the stage it is behind, or the option that changed, from what to what.

They cannot simply be started again. Each of them, started over a folder that already holds its output, finds that output and keeps it: the run would finish and report the old result as the new one. The message says 'started and did not finish' for a stage that was interrupted; what it would keep is then the part it had written.

PARALOME DOES NOT REMOVE OR REPLACE RESULTS ON ITS OWN ACCOUNT. It adds no --overwrite to any stage; --overwrite reaches a stage only when you type it. The folders are yours, and what is in them may be the only copy of days of work. Every stage that is affected is named at once, with its folder and the reason, so the command is run again once and not once per stage. A folder shared with another stage says so: that stage will run again too.

A FOLDER IS NAMED FOR REMOVAL ONLY IF IT IS THAT STAGE'S OWN, INSIDE THE PROJECT. A folder outside the project, the project folder itself, a folder that also holds the results of an earlier stage, and a folder that holds an input of this run are never named for removal: the single-cell aggregation, for one, writes into the study's data folder, which holds the extraction too.

A stage folder that is a link inside the project to somewhere else (<project>/star linked to scratch) is the stage's own. It is named by its path in the project, and what is in it is what has to go; the link may stay.

A STAGE FOLDER GIVEN OUTSIDE THE PROJECT (--align-workdir elsewhere) is named, and said to lie outside the project: Paralome cannot know what else is in it. The message says what of the stage has to be removed from it before the same command is run again. For the alignment that is the per-sample BAMs and their indexes (bam/*.sorted.bam and .bai) and its records (checkpoints/alignment.json, results/star_alignment_*.json). For any other stage it is everything in the folder. Removing a stage's report files alone does not do: the alignment would find its BAMs, align nothing and report them as the new result. The run is refused until what the message names is gone, and `status` goes by the same test.

For a folder that is not named for another reason (the project folder itself, one shared with an earlier stage, one holding an input) the message says why and lists the stage's own files in it; for the alignment, again its BAMs, indexes and records.

TWO WAYS ON. Remove what is named and run the same command again: those stages are then computed afresh. Or, where the reason is an option that changed, put the option back as it was: the results on disk answer that request, nothing was touched, and the run carries on with them. `--dry-run` shows the same refusal without starting anything.

RNA quantification, differential expression, functional evidence, the two single-cell aggregations, variant calling and gene burden are never listed. Each checks its own result against what it is now given and computes again, replacing only its own files, when the result no longer answers; the run issues those again itself. Nothing was submitted.

#### PLM-0309 — This stage was started with a different request

**Fix:** Remove the stage folder to redo it, or repeat the earlier options

The folder already holds finished samples, and they were made with other options, or from other input files, than this command gives. The message lists each such sample and what differs.

A stage resumes by running only the samples that are not finished. Resumed under a different request, it would hold samples made two ways and report all of them under the new one, and nothing in the output would say which sample was made how.

TWO WAYS ON. To finish the stage as it was started, repeat the earlier options. To redo every sample with the new ones, remove the stage folder and run again. Nothing is replaced without being asked, and nothing was started.

Threads, memory and partition are not part of the request: they do not change a result, and changing them is not refused.

#### PLM-0310 — Another supervisor is already running on this project

**Fix:** Wait for that run, or stop it, then run the same command again

One project is run by one supervisor at a time. The message names the host, the process and the start time of the one that holds it.

The usual cause is a run started under nohup or in a terminal that was since lost, and the same command typed again. Two supervisors on one project both wait for the first one's jobs and then both submit the next stage, into the same folder.

The supervisor that is running needs no help: it submits each stage when the one before it finishes. `paralome status --project` shows where it has got to. To take the project over, stop that process on the host named, then run the same command again; it resumes from what is finished.

THERE IS NORMALLY NO STALE LOCK TO CLEAR. The lock is held by the running process and released by the kernel when it ends, however it ends. The lock file stays in checkpoints/ and does not need removing.

The one exception is a network filesystem, which can keep a lock for a process that is gone. If the host named is down, or the process named no longer exists on it, and the lock is still held, the lock file named in the message may be removed; then run the same command again. Check the host and the process first: removing the file under a supervisor that is running lets a second one start beside it.

Nothing was submitted.

#### PLM-0311 — Jobs from an earlier submission may still be running

**Fix:** Wait for them to finish, or scancel them, then run again

This stage was submitted before from the same folder, and the jobs recorded for it are still in the queue -- or SLURM could not be asked whether they are, or this machine has no squeue to ask. The message says which, and lists the job ids.

Submitted again beside them, every unfinished sample would be run twice at once, two jobs writing the same files.

If the jobs are running, they are doing the work: wait for them, and run the command again only if something is then missing. To start over, scancel them first.

WHEN SLURM DOES NOT ANSWER, nothing is submitted and nothing is cancelled. Not knowing whether work is running is no reason to duplicate it and no reason to stop it. squeue is asked again for ten minutes before this is reported; run the same command again when it answers.

WHEN SQUEUE IS NOT ON THIS MACHINE, asking again cannot help, so the refusal comes at once and nothing is started. Run the command on the cluster, where squeue can be asked. Or, if you know those jobs have ended, remove the record of them that the message names and run again.

#### PLM-0312 — A BAM that failed verification is still in place

**Fix:** Remove the BAMs named and their .bai; a sample with no reads: '#' it out of the sheet

`align verify` read the BAMs named in the message and rejected them; the reason is given beside each.

Such a BAM is present and indexed, so alignment run again would take the sample as finished, and verification would reject it again, for as long as the file is there.

It is not deleted for you: it is your output, and you may want to look at it first. Remove the BAMs named and their .bai files and run the same command again; only those samples are aligned. Removing the whole alignment folder aligns every sample again.

A SAMPLE THAT ALIGNED NO READS is the exception, and the message lists such samples apart ("Aligned no reads"). Removing its BAM and aligning again gives the same empty BAM: an empty negative control, or FASTQ from another species. To go on without the sample, take it out of the sample sheet by putting '#' at the start of its line, and run again. If its FASTQ is the wrong file, correct that, then remove that sample's BAM and .bai and run again.

Nothing was started.

#### PLM-0313 — Slingshot failed

**Fix:** Read the slingshot.log the error names in the trajectories folder

The trajectory fit runs slingshot_trajectories.R in the R --rscript names. Its output, and the R error when it stops, is in <run>/trajectories/fit/slingshot.log (an older run's is in <run>/trajectories/slingshot/).

#### PLM-0314 — Monocle3 failed

**Fix:** Read the monocle3.log the error names in the trajectories folder

The trajectory fit runs monocle3_trajectories.R in the R --rscript names. Its output, and the R error when it stops, is in <run>/trajectories/fit/monocle3.log. A graph whose part holding the root has no leaf has no lineage to follow: try use_partition=n, or a smaller minimal_branch_len.

#### PLM-0315 — Monocle3's fit_models failed on the trajectory gene models

**Fix:** Read monocle3_gene_models.log in the trajectories expression folder

--trajectory-model monocle3 fits each lineage's gene models with Monocle3's fit_models, in monocle3_gene_models.R in the R --rscript names. Its output, and the R error when it stops, is in <run>/trajectories/expression/monocle3_work/monocle3_gene_models.log. --trajectory-model ls fits the same models by least squares.

### Environment (0400-0499)

#### PLM-0401 — That genome is not configured

**Fix:** paralome setup genome lists what exists; setup genome add <folder> registers one

--genome names a genome bundle registered in your Paralome configuration, not a path and not an assembly name in the abstract.

`paralome setup genome` lists the genomes that exist, and `paralome setup genome add <folder>` registers one.

Genome configuration is per-user, so a bundle registered by one account is not visible to another until it is adopted. If a colleague can use a genome you cannot, run `paralome setup genome sync` -- it takes anything registered on this machine into your own configuration without retyping a path.

If this account cannot read part of the machine's registry or genome repository, the message says Permission denied and names each path, rather than calling the genome absent. Ask the administrator to set the permissions it names (usually: add the account to the shared repository's group, then log in again).

#### PLM-0402 — A configured genome's files cannot be read

**Fix:** Check the paths in paralome setup config, and their permissions

The bundle is registered but a file it points at is missing or unreadable -- a moved reference, a deleted GTF, or a path pointing into another user's home directory.

The console names the specific file, not just the bundle.

#### PLM-0403 — Rscript was not found

**Fix:** Load the R module, or pass --rscript with the full path

Single-cell extraction shells out to R to read the Seurat object. Rscript is not on PATH.

On a cluster this usually means an unloaded module -- and note that the compute node's environment is not your login shell's, so a module loaded interactively is not necessarily loaded inside the job.

#### PLM-0404 — A required R package is not installed

**Fix:** Install the package named into the R that Rscript actually resolves

Seurat and Matrix are required to read the object; Signac is required for ChromatinAssay objects.

Check WHICH R is being used. A package installed in an interactive session may not be visible to the Rscript a compute node resolves.

#### PLM-0405 — A file that ships with Paralome is missing

**Fix:** Reinstall the wheel with --force-reinstall

A bundled resource -- the extraction R script, a motif database -- is absent from the installed package. The install is damaged or partial.

Because Paralome versions below 1.0 sort under the old 1.0.x builds, pip may consider an older install newer and decline to replace it. --force-reinstall is required, not merely advisable.

#### PLM-0406 — Gene annotation needs a registered genome

**Fix:** Pass --genome; setup genome list shows what is registered here

Gene identity is resolved once, at extraction, and stored -- so that every downstream comparison agrees on what a gene is.

Without a GTF the run would produce a gene table with no stable identifiers, which is worse than not running: two comparisons of the same object could then disagree about which gene a row refers to.

There is no --gtf. It would be an override on a value --genome already resolves, so a hand-typed GTF could disagree with the genome the run recorded, with nothing to say which one was used -- and the registered genome is what makes two runs comparable.

#### PLM-0407 — That genome has been withdrawn as unusable

**Fix:** Adopt the replacement, or ask its owner why it was withdrawn

The account that registered this genome marked it FATALLY flawed, which means results produced with it are wrong rather than merely dated -- an index built against the wrong reference, a GTF that does not match the assembly, a truncated download.

The run stops rather than warning, and it stops for the owner too. A fatally flawed genome produces wrong answers regardless of who runs it, and a warning would be read once and then scrolled past.

This is NOT what a superseded genome does. A genome retired because something newer exists -- hg19 after hg38 -- warns and proceeds, because work already under way against it is still correct. Only a genome whose results cannot be trusted reaches this code.

`paralome setup genome list` shows the reason its owner gave.

#### PLM-0408 — Paralome is installed privately, in your home directory

**Fix:** Ask an administrator to install it for the machine, then remove this one

The copy you are running was installed with `pip install --user`, into your own home directory rather than for the machine.

Nothing is broken. The objection is that it would work -- privately. A private install cannot publish to the machine's registry and cannot see what anyone else has published, so the reference genomes it uses are ones you build yourself, beside identical copies somebody has already built. That is the duplication the shared registry exists to end, and it is invisible from both sides: nobody else can see your genomes, and you cannot see theirs.

It is also a version trap. Your own packages come before the system ones on Python's import path, so this copy is loaded even when the system command is invoked by its full path, and `paralome --version` reports through the same shadowed path. Two accounts on one machine can run different software while both answer to `paralome`.

Removing it costs nothing. Your configuration and any genomes you registered live in ~/.paralome and are independent of where the code is installed, so they survive:

    python3 -m pip uninstall paralome
    which -a paralome

Paralome is meant to be installed once for a machine, by whoever administers it, so that everyone on it shares one version and one set of reference genomes. If you do not have that access, the install is the administrator's to make.

A virtualenv is not affected -- this is about `--user` specifically, not about whether the install needed root.

#### PLM-0409 — That genome is not in this machine's shared repository

**Fix:** Put the folder in the repository the message names, then register it there

This machine has a shared genome repository, and a genome registered anywhere else can only be used by the account that registered it.

This refuses rather than warns because otherwise people take the easy route and register their own, which nobody else can use, and the machine ends up with a dozen human and a dozen mouse genomes.

That is the failure the shared repository exists to prevent, and it is not a failure anyone notices at the time -- a genome in your home directory works perfectly for you. It shows up months later as four people holding four copies of GRCh38 and four different answers to which annotation release their results were against.

Move the folder into the repository and register it from there. If it is large, move it rather than copying -- the repository is group writable for exactly this, and the sticky bit means only you can remove your own genome afterwards.

If you cannot write to the repository, you are not in its group yet; an administrator adds you, and you log out and back in.

A machine with no shared repository is unaffected: a workstation or a laptop registers genomes from wherever they are, as before.

#### PLM-0410 — A tool was recorded at a path that no longer exists

**Fix:** Run `paralome setup` to re-record where the tools are now

`paralome setup` records the absolute path of every external tool it finds, and the runner invokes those recorded paths rather than looking each tool up on PATH again at run time. The path is the one PATH finds, not where a symlink there points: Ubuntu's STAR (`apt install rna-star`) is a link to a program that chooses its build by the name it is called as, and recorded through the link it would not run.

The path recorded for the tool named above is no longer there. That usually means the tool was upgraded, moved, or removed since setup last ran.

WHY THIS REFUSES INSTEAD OF FALLING BACK TO PATH. The fallback is silent and the consequence is not. Two accounts with different PATHs -- which is exactly what happened at one site, where a local bin directory held newer builds of STAR and freebayes than /usr/bin -- would run different binaries for the same command and get different numbers, with nothing anywhere recording that they had. A stale recorded path is a configuration that needs fixing once; a silent substitution is a result nobody can reproduce.

It is the same judgement made for genome files: a path that does not resolve stops the run at the start rather than surfacing later as a number that looks fine.

`paralome setup doctor` shows every recorded tool and whether it is still there.

The preflight first looks for a moved tool on this account's PATH and, when it finds it, records it there, says so, and the run uses it -- an administrator moving a tool is not something a user can know about. This error is what remains: a tool found neither where it was recorded nor on PATH.

#### PLM-0411 — A configuration file has a key that nothing reads

**Fix:** Correct or remove the key the message names; it lists the valid ones

The file named in the message -- your configuration, a file given with --pipeline-config, or this machine's site.json -- holds a key no part of Paralome reads. Almost always it is a misspelling, and the nearest real key is suggested.

The rule is that a setting that does not match fails, because such things ultimately bite the user: a mistyped key would otherwise load without complaint and do nothing, and the run would go ahead on the default the key was written to replace.

Every command that reads the file refuses, not only the one the key was meant for: a setting that is silently ignored anywhere is the defect. `paralome setup doctor` and `paralome setup where` still report beside such a file, and name the key.

YOUR OWN CONFIGURATION is yours to correct: edit the file, or remove it with `paralome setup reset`.

SITE.JSON belongs to the machine's administrator, and you cannot correct it. Until they do, every run by every account on the machine is refused; nothing you can set overrides it. `paralome setup site` still shows the file and marks the key, for whoever does the editing. There is no reset for it.

Names you choose yourself -- genomes, profiles, recorded tools -- are not checked; the keys inside each of them are.

Two keys are refused although they were once read: annotation.annovar_buildver and annotation.snpeff_genome. A genome's build comes from the genome entry named by --genome, or from --annovar-buildver / --snpeff-genome, and a DNA run says so rather than load a build it will not use.

#### PLM-0412 — A configuration file is not valid JSON

**Fix:** Correct the file the message names; `python3 -m json.tool FILE` shows where it breaks

The file named in the message -- your configuration, a file given with --pipeline-config, or this machine's site.json -- could not be read as JSON, or holds JSON that is not an object (a list, a string, a number). The message gives the line and column where reading stopped.

The usual cause is an edit by hand: a comma after the last entry, a missing quote, a comment. JSON allows none of them. `python3 -m json.tool FILE` prints the same position and nothing else.

Such a file is never read as an empty one: read as empty, the run would go ahead without any of the settings in it, which is the silent fallback Paralome refuses everywhere.

YOUR OWN CONFIGURATION and a --pipeline-config file are yours to correct. `paralome setup reset` replaces your configuration with a new one if it is past repair.

SITE.JSON belongs to the machine's administrator. Until they correct it, every run by every account on the machine is refused.

This is not PLM-0106, which is about an input data file -- an .rds that is not an RDS. PLM-0411 is the same file with valid JSON and a key nothing reads.

#### PLM-0413 — A configuration file holds a value that is not allowed

**Fix:** Correct the value in the file the message names; it says which and what is allowed

Your configuration file is valid JSON and every key in it is one Paralome reads, but a value is not one it accepts: a config_version other than the one this build writes, a required section missing or not an object, an rna.strandedness other than 0, 1, 2 or auto, or a negative rna.min_expression. The message names the file and the value.

Every command that reads the file refuses, including the RNA expression gate: treating such a file as if it did not exist would fall back to the built-in 5 FPKM thresholds, so a configured rna.min_expression of 1 beside a misspelt key would silently become 5, a gate the user never set. That is the silent fallback Paralome refuses everywhere.

YOUR OWN CONFIGURATION is yours to correct: edit the file, or replace it with `paralome setup reset`. `paralome setup doctor` reports beside such a file and names the value.

PLM-0411 is the same file with a key nothing reads; PLM-0412 is a file that is not JSON.

#### PLM-0414 — The scheduler refused to accept a job

**Fix:** The message carries sbatch's own sentence; run the printed command to see it in full

sbatch exited non-zero, so nothing was queued. The useful part is the scheduler's own reason, which this message carries verbatim.

WHY THIS EXISTS. Paralome submitted with check=True and reported the CalledProcessError, whose text is the command and the exit status and nothing else -- so a refused submission reached the user as 'returned non-zero exit status 1' with the reason captured and discarded. It happened twice before this code existed, and both times cost a re-run by hand to recover a sentence Paralome already had.

Common reasons: a dependency on a job the scheduler no longer knows (the job it depended on had already left the queue), an invalid partition or account, a memory or CPU request above what any node offers, or a log directory that does not exist.

Nothing was queued by the refused call, so the stage is not half submitted. Whatever was submitted BEFORE it is still queued and still running; fix the cause and re-issue the identical command, which reuses the completed work.

#### PLM-0415 — A required Python package is not installed

**Fix:** Install the package named into the Python that runs Paralome

The cell export (<rds>_cell_expression_embeddings.zip, written with every single-cell extraction) writes an .h5ad file with h5py.

Paralome is installed with --no-deps into a prefix of its own, so the packages it uses come from the Python that runs it. Install h5py there: python3 -m pip install h5py. A compute node resolves the same Python only when the module loads it, so install it where the jobs run.

#### PLM-0416 — bedGraphToBigWig or bedToBigBed is not installed

**Fix:** conda install -c bioconda ucsc-bedgraphtobigwig ucsc-bedtobigbed, then paralome setup

The genome tracks (the genome_tracks stage) are written with the UCSC tools: the bigWig profiles with bedGraphToBigWig, the peak and gene tracks with bedToBigBed. Paralome does not ship them; install them where the jobs run, and `paralome setup` records their paths as it does every other tool's. UCSC's licence lets its command-line tools be used freely, commercial use included (all but BLAT and LiftOver).

#### PLM-0417 — slingshot is not installed in the R the stage runs

**Fix:** In that R: BiocManager::install("slingshot"), or add --no-trajectories

The trajectories stage fits the lineages with the Bioconductor package slingshot, in the R that runs the extraction (--rscript, default Rscript). Paralome does not install R packages; install it there, where the jobs run. The preflight checks for it before anything is submitted -- and,, for DelayedMatrixStats, which slingshot calls while it fits.

#### PLM-0418 — Monocle3 is not installed in the R the trajectory fit runs

**Fix:** remotes::install_github("cole-trapnell-lab/monocle3") in that R

The default --trajectory-method fits the lineages with Monocle3, in the R that runs the extraction (--rscript, default Rscript). Monocle3 is installed from GitHub, after its dependencies (leidenbase, sf, terra, BPCells among them). Paralome does not install R packages; install it there, where the jobs run. The preflight checks that it loads before anything is submitted.
