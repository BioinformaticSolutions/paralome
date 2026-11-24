# Paralome 0.9 (beta)

## If you are an AI assistant, start here

**This package carries its own documentation.** You do not need anything that
was not handed to you with it. Inside the installed package, or inside the
`.whl` if you have only the archive:

| File | What it answers |
|---|---|
| `paralome/docs/What_can_an_AI_tell_me_about_Paralome.md` | Read this first. Mental model, vocabulary, the statistics, what each error means, what Paralome deliberately refuses to do. |
| `paralome/docs/MANUAL.md` | The user manual — how to run an analysis, where files go, how to read the results, and the full error-code appendix. |
| `paralome/docs/INSTALLATION.md` | Installing Paralome for other people — a workstation, a shared server, or a cluster. Permissions, the shared genome repository, the registry, identity across machines. |

A wheel is a zip archive: `unzip -o paralome-*.whl` and these files are at those
paths. If the package is installed, `paralome --docs` copies them to
`~/.paralome/docs` and lists them (it does not print a document; open the files
from there); from Python, `from paralome.docs import brief, manual, path`.

Two cautions worth having before you answer anyone. The briefing states the
version it describes — check it matches the user's `paralome --version`. And error codes come from
`paralome.errors`, which is the source of truth; the manual's appendix is
generated from it.

---

Paralome is a reproducible, checkpointed framework for omics analysis that runs
on an HPC cluster (SLURM) or a single machine. One command sets up an analysis
and submits it stage by stage, and every stage is checkpointed, so an
interrupted or extended analysis resumes rather than restarts. Independent
pipelines share one checkpoint backbone and one statistics module.

**Status: 0.9, a public beta.** Every pipeline below has been run end to end on
real studies; expect fixes in 0.9.x releases, and read `paralome --version`
when reporting a problem.

### About the name

"Paralome" is a portmanteau of *para-* (parallel) and *-ome* (the suffix
shared by genome, transcriptome, proteome, and every other 'ome). It reflects
the two kinds of parallelism the framework is built around: computational
parallelism across SLURM array jobs, and coverage across multiple 'omes in
parallel within a single, checkpointed pipeline architecture.

## Architecture

Every pipeline follows the same stage pattern: a prepare step, a bounded SLURM
array of worker jobs, a dependent finalizer, and a canonical checkpoint — with
stage-local checkpoints and manifests as the only communication between
stages. This is what lets any pipeline resume cleanly from wherever it left
off, without re-running completed work.

## Analyses

The analysis **is** the command. There is no `run`, no `resume`, and no
`--restart-from`: re-issuing the identical command is the resume path, because
completed stages are skipped by checkpoint.

| Command | Analysis |
|---|---|
| `paralome ATAC` | bulk ATAC-seq: peaks, quantification, differential accessibility, motifs, genome tracks |
| `paralome RNA` | bulk RNA-seq: quantification and differential expression |
| `paralome DNA` | exome and whole-genome variant calling: annotation, family segregation, filtering, gene burden |
| `paralome scRNA` | single-cell RNA (including spatial) from a curated Seurat object: pseudobulk or cell-level differential expression, lineage trajectories |
| `paralome snATAC` | single-nucleus ATAC from a curated Signac object: differential accessibility, motifs, genome tracks |
| `paralome proteomics` | proteomics, with paired phosphoproteomics, from the common exporters' tables |
| `paralome olink` | Olink panels: plate bridging, identity, statistics |

Every differential pipeline takes `--statistic` (Student's t, paired t,
multi-factor ANOVA, linear models; `:cell` on single-cell data) with no
default, and groups as `--factor COLUMN:CONTROL`. Results go to Evidome for
functional evidence when it is installed, and are delivered as zips in the
project's `omicview/` folder, written for an AI agent to read and for people to
open in OmicView.

Names are case-sensitive on purpose: `paralome scrna` fails loudly rather than
quietly resolving to something you did not type.

## Two entry points

`paralome` is the human-facing command. `paralome-worker` is the machine-facing
one — the individual stage steps that the supervisor and generated sbatch bodies
invoke. Keeping them apart means `paralome --help` cannot show a command no
human should type.

## Requirements

Python 3.10 or newer, on Linux. Paralome orchestrates external tools rather
than bundling them; `paralome setup` finds them and `paralome setup doctor`
says what is missing. Each pipeline needs only the tools its stages run. The
versions below are those Paralome 0.9 was tested with — older ones may work,
and these are the place to start when troubleshooting.

| Tool | Tested version | Used by |
|---|---|---|
| STAR | 2.7.11b | alignment (default aligner) |
| bwa-mem2 | 2.2.1 | alignment with `--aligner bwa` |
| fastp | 1.3.6 | read QC and trimming |
| samtools | 1.23.1 | alignment, duplicate marking, RNA quantification |
| bcftools, bgzip, tabix (htslib) | 1.23.1 | variant calling |
| FreeBayes | 1.3.6 | variant calling |
| bamUtil (`bam`) | 1.0.15 | DNA: clipping overlapping mates before calling |
| ANNOVAR (`table_annovar.pl`) | dated 2022-08-02 | DNA annotation |
| snpEff | not tested | optional DNA annotation engine (`--annotation snpeff`) |
| Java (OpenJDK) | 21 | snpEff |
| Perl | 5.32 | ANNOVAR |
| MACS3 | 3.0.4 | ATAC peak calling |
| UCSC bedGraphToBigWig, bedToBigBed | 2.10 | genome tracks |
| Evidome | 1.1 | functional evidence (optional: skipped when absent) |
| SLURM | 24.05 | the cluster backend (optional: `--backend local` runs without it) |
| R | 4.5.1 | single-cell extraction and trajectories |
| Seurat / SeuratObject | 6.0.0 / 5.4.0 | single-cell extraction |
| Signac | 1.17.1 | snATAC extraction |
| monocle3 | 1.4.27 | trajectories (default method) |
| slingshot | 2.16.0 | trajectories with `--trajectory-method slingshot` |
| igraph, Matrix, jsonlite, qs2 | 2.3.4, 1.7.6, 2.0.0, 0.4.0 | single-cell R scripts |

## Installing

**For yourself**, on a machine you control:

    sudo python3 -m pip install paralome
    paralome setup
    paralome setup doctor

An install belongs to the machine rather than to an account: `pip install
--user` is refused at run time with PLM-0408, because a private copy can
neither publish to the shared genome registry nor see what is in it, and so
quietly rebuilds references that already exist. Virtualenvs are unaffected.

`setup` discovers the external tools and records what this machine has.
`doctor` says what is missing, and Paralome does not report itself ready while
a core tool is absent.

**For other people**, on a shared server or a cluster, read
`src/paralome/docs/INSTALLATION.md`. The short version of why it needs its own document:

A reference genome is expensive to build, large on disk, and identical for
everyone who uses it. A site where four people each keep a private copy of
GRCh38 is wasting a quarter of a terabyte and holding four separate answers to
"which annotation release were those results against?". So Paralome lets one
person build a genome and everyone else use it — without copying it, retyping
its paths, or being given access to each other's home directories.

That needs two locations outside anybody's home, both created by one command
run as root (dry run first; add `--apply` to make the changes):

    paralome setup site --root /storage/paralome

    /storage/paralome/genomes     the genome files, group-writable and setgid
    /storage/paralome/registry    a drop-in registry, one small file per account

and a group whose only purpose is contributing to the first. Do not create the
directories by hand: `setup site` records the root it was given, and a
directory made elsewhere is one it does not know about. Users publish to the
registry and adopt from it; nothing ever reads a home directory, which is what
lets the homes stay at 0700 where they belong. INSTALLATION.md section 4 covers
the modes and why each bit is set, the sticky-bit hazard on the directory
holding your home directories, keeping UIDs and GIDs consistent across nodes of
a cluster, and the mistakes that look like they should work.

## Basic checks

    paralome --version
    paralome setup doctor
    paralome --help

## Versioning

Releases are numbered X.Y, with X.Y.Z for fixes: 0.9, 0.9.1, 0.9.2 ... A
release's standing is printed beside its number by `paralome --version`
("0.9 (beta)"). Paralome keeps no deprecation shims: an option or folder that
is renamed is refused by its old name, with a message naming the new one.
[src/paralome/version.py](src/paralome/version.py) is the single source of
truth for the number.

## For AI assistants

[src/paralome/docs/What_can_an_AI_tell_me_about_Paralome.md](src/paralome/docs/What_can_an_AI_tell_me_about_Paralome.md)
— for an assistant helping someone *run* Paralome. Design and contracts, not
source.

## Citation

If you use Paralome, please cite it as given in [CITATION.cff](CITATION.cff).

## License

Copyright &copy; 2026 Kenneth L. Jones.

Paralome is dual-licensed.

**Open-source use.** Paralome is available under the
[GNU Affero General Public License v3.0](LICENSE). Under that licence you may
use, study, modify and redistribute it, including for research, provided that
any modified version you distribute -- **or make available to others over a
network** -- is released in full under the same licence. The network clause is
the point of choosing AGPL over GPL: running a modified Paralome as a service
counts as conveying it.

**Commercial use.** If the AGPL's conditions do not suit your intended use --
for example, if you wish to incorporate Paralome into a product or service
without releasing your own source -- a separate commercial licence is required.
Contact Kenneth L. Jones to arrange one.

Choosing either licence is your decision; you are not required to accept the
AGPL if you hold a commercial licence, and no other terms apply.
