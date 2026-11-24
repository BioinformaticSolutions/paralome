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

"""Numbered errors: one line on the console, the long version in the manual.

Every user-defined parameter has a specific error for each way it can be
wrong, with a suggestion of what to do. The command line shows an error number,
a short description and the fix; the details go to the log; the number can be
looked up in the manual for a longer description.

THREE AUDIENCES, THREE LENGTHS, and the whole point is that they are separate:

  ``summary``     one line. What went wrong. Printed on the console.
  ``fix``         one line. What to do about it. Printed on the console.
  ``explanation`` paragraphs. Why it happens and what it means. Manual only.

Everything else -- the traceback, the locals, the full paths -- goes to the log
file and nowhere else.

NUMBERING. Plain integers in blocks of a hundred, so a number can be read aloud
and still says where it lives:

  0001-0099   general; 0001 is the catch-all for anything unnumbered
  0100-0199   what the user typed -- CLI parameters, before any work starts
  0200-0299   the request versus the data -- valid syntax, wrong for this object
  0300-0399   execution -- SLURM, fan-out, exit codes
  0400-0499   environment -- genomes, R, the install itself

Appending to a block never renumbers another. A failure that prints no code
becomes a new entry in whichever block it belongs to.

THE REGISTRY IS THE SOURCE OF TRUTH. ``docs/MANUAL.md``'s appendix is generated
from it, so a code cannot exist in one and not the other.
"""

from __future__ import annotations

import os
import pickle
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

__all__ = [
    "ErrorSpec",
    "ParalomeError",
    "REGISTRY",
    "format_code",
    "lookup",
    "render_console",
    "write_details",
    "install_error_handler",
]


@dataclass(frozen=True)
class ErrorSpec:
    """One numbered failure mode.

    ``summary`` and ``fix`` are single lines because they are printed together
    on a terminal, immediately after something went wrong, to a person who wants
    to get back to work. Neither may end in a full stop-free run-on; keep them
    short enough to read at a glance. The test suite enforces the length.
    """

    code: int
    summary: str
    fix: str
    explanation: str


def format_code(code: int) -> str:
    """``104`` -> ``PLM-0104``. The spelling used everywhere it is displayed."""
    return f"PLM-{code:04d}"


# ---------------------------------------------------------------------------
# The registry.
#
# Written as a flat literal rather than assembled, so the whole catalogue can be
# read in one pass and a reviewer can see every message the software is capable
# of printing without running it.
# ---------------------------------------------------------------------------

_SPECS: Sequence[ErrorSpec] = (
    # --- 0001-0099  general -------------------------------------------------
    ErrorSpec(
        1,
        "Unexpected internal error",
        "Send <project>/logs/paralome.err (else ./paralome.err) to the maintainer",
        "Paralome hit a failure it has no number for. That is a gap in this "
        "catalogue, not necessarily a bug in your command -- but the software "
        "cannot tell you which, so it stops rather than guess.\n\n"
        "The full traceback is in the log file; the console does not name "
        "it. The log is "
        "<folder>/logs/paralome.err when the command was given --project or "
        "--out and that folder exists; otherwise it is paralome.err in the "
        "directory the command was run from. The traceback is not printed to "
        "the terminal because it is diagnostic material for whoever maintains "
        "the software, and burying the one actionable line in eighty lines of "
        "stack frames is how the previous behaviour hid it.\n\n"
        "If your command looks correct, this is worth reporting: every "
        "reproducible failure should eventually have its own number and its own "
        "instruction.",
    ),
    # --- 0100-0199  what the user typed -------------------------------------
    ErrorSpec(
        101,
        "A required option was not given",
        "Add the option(s) named in the message and re-run",
        "Each analysis has a small set of options it cannot proceed without. "
        "For scRNA and snATAC those are --rds and --replicate-column, plus --project "
        "for a full pipeline run; --factor COLUMN:CONTROL is needed by every "
        "statistic that compares levels (t, paired-t and anova).\n\n"
        "--genome belongs to that set for every analysis that reads a genome. "
        "Paralome never chooses one for you, and a genome named in the "
        "configuration is not used as a default: the wrong genome gives "
        "results that look fine. The message lists the genomes configured.\n\n"
        "Options NOT in that set are genuinely optional, including "
        "--stratify-by. If you were told one of those is "
        "required, that is a bug -- report it.",
    ),
    ErrorSpec(
        102,
        "That option belongs to a different analysis",
        "Single-cell analyses take --rds; bulk analyses take --samples",
        "Bulk analyses (RNA, ATAC, DNA) start from FASTQ files and are handed a "
        "sample manifest with --samples. Single-cell analyses (scRNA, snATAC) "
        "start from an already-processed Seurat/Signac object and are handed it "
        "with --rds.\n\n"
        "Passing both, or passing the wrong one, is refused rather than "
        "silently ignored: an ignored option looks like it took effect.",
    ),
    ErrorSpec(
        103,
        "Two options were given that contradict each other",
        "Pass one or the other, not both; the message names the pair",
        "Two of the options given cannot both take effect, and neither is "
        "guessed at. The message names the pair. The cases:\n\n"
        "--stratify-value names the strata to RUN and --stratify-skip the "
        "strata to EXCLUDE. Given both there is no defensible interpretation. "
        "Omitting both runs every stratum, which is usually what you want.\n\n"
        "--submit with --backend local. Submission is what the SLURM backend "
        "does; a local run has nothing to submit. Drop --submit, or run with "
        "--backend slurm.\n\n"
        "On the genome build commands, a SLURM-only option (--partition, "
        "--time, --max-parallel) without --backend slurm, or --threads, which "
        "is this machine's thread count, with it. Accepted, the option would "
        "do nothing, which looks like it took effect. The message says which "
        "option and what it needs.",
    ),
    ErrorSpec(
        104,
        "An input file does not exist",
        "Check the path and spelling; the log names the directory searched",
        "The file named by --rds, --samples or --reference is not on "
        "disk at the path given. Paths are resolved relative to the directory "
        "you ran the command from, not to --project.\n\n"
        "This is checked before anything is submitted, so a missing file is "
        "found here rather than by a compute node a poll cycle later.\n\n"
        "Renaming an object and not updating the script is the single most "
        "common cause, so when a similarly-named file sits in the same "
        "directory it is recorded in the log and offered as the suggested fix.",
    ),
    ErrorSpec(
        105,
        "An input file exists but cannot be read",
        "Check permissions; the log reports the owner, the mode and your uid",
        "The file is present but the user running Paralome cannot open it. On a "
        "shared cluster this is usually a file created by another account, or "
        "one inside a directory whose execute bit is not set for you.\n\n"
        "The log records the file's owner, its mode, and the account you are "
        "actually running as, because 'permission denied' without those three "
        "facts cannot be acted on.",
    ),
    ErrorSpec(
        106,
        "The input file is not the format it claims to be",
        "Convert it; `paralome explain 106` prints the exact command",
        "A .rds extension does not make a file an RDS. The two common cases:\n\n"
        "save() instead of saveRDS(). This writes an RData WORKSPACE -- a bag of "
        "named objects -- not a single serialised object. Same extension, "
        "similar size, and R itself reports only 'unknown input format'. "
        "Paralome detects it by magic bytes (RDX2/RDX3/RDA2). The conversion, "
        "for a workspace holding one object:\n\n"
        "    Rscript -e 'e <- new.env(); load(\"obj.rds\", envir = e); "
        "saveRDS(get(ls(e)[1], envir = e), \"obj_fixed.rds\")'\n\n"
        "A workspace holding several objects needs the right one named in "
        "place of ls(e)[1]; `ls(e)` lists them.\n\n"
        "SaveH5Seurat() instead of saveRDS(). An HDF5 container. Detected the "
        "same way. In R, with SeuratDisk loaded: "
        "`saveRDS(LoadH5Seurat(\"obj.h5seurat\"), \"obj.rds\")`.",
    ),
    ErrorSpec(
        107,
        "The output directory cannot be written to",
        "chown or chmod the directory; the console prints the exact command",
        "Paralome checks the project tree, or the nearest existing parent of "
        "it, before starting.\n\n"
        "The usual cause on a shared system is a directory tree created by root "
        "-- for instance when an administrator sets up a new user's folders and "
        "does not hand ownership over. The symptom without this check is "
        "brutal: the run fails, and it fails while trying to write its error "
        "into the directory it could not create, so there is no log to read and "
        "no message to see.\n\n"
        "This is one of two errors allowed more than two console lines: the "
        "owner, the mode, the uid actually running and the chown that fixes it "
        "are printed, because a permissions error you have to go and diagnose "
        "is a permissions error you will get wrong.",
    ),
    ErrorSpec(
        108,
        "That value is not one of the accepted choices",
        "Use one of the values listed on the console",
        "Options such as --statistic, --backend and --scanner-backend accept a "
        "fixed set of values. The full set is printed, and when your value is "
        "close to a real one it is suggested.\n\n"
        "Paralome is case-sensitive throughout and has no aliases, by design: a "
        "silent alias is how two spellings of one concept become established.",
    ),
    ErrorSpec(
        109,
        "--alpha is missing its metric letter",
        "Write Q0.05 or P0.05, not 0.05",
        "The significance threshold carries WHICH statistic it applies to. "
        "Q0.05 cuts on the FDR-adjusted q-value; P0.05 cuts on the raw p-value.\n\n"
        "The letter is mandatory because the two are very different decisions "
        "and a bare 0.05 does not say which was intended. P is a deliberate "
        "choice with consequences, not a default to arrive at by omission.",
    ),
    ErrorSpec(
        110,
        "A memory value is not in a form SLURM understands",
        "Use values such as 4G, 32G or 32000M",
        "Memory is passed to SLURM as a number with a unit suffix. A bare "
        "number, or a unit SLURM does not recognise, is rejected here rather "
        "than at submission time, where the failure would be a queue rejection "
        "with no indication of which stage's setting was malformed.",
    ),
    ErrorSpec(
        111,
        "--stage-mem names a stage that does not exist",
        "Check the stage name against the analysis's stage list",
        "--stage-mem takes STAGE=MEM. The stage must be one in the stage graph "
        "for the analysis you are running; the valid names are printed.\n\n"
        "A misspelled stage name would otherwise be accepted and silently do "
        "nothing, and the stage would run with its default allocation -- which "
        "presents much later as an out-of-memory kill.",
    ),
    ErrorSpec(
        112,
        "A numeric option is outside its allowed range",
        "Use a value within the range printed on the console",
        "--threads and --jobs must be at least 1. --alpha must be strictly "
        "between 0 and 1. Cell and sample floors must not be negative.\n\n"
        "Zero threads is the notable one: it is accepted by most tools as "
        "'auto' and by Paralome as an error, because on a shared cluster the "
        "number of CPUs a job may use is a scheduling decision, not something "
        "to be inferred from the machine it lands on.",
    ),
    ErrorSpec(
        113,
        "--factor is malformed",
        "Use --factor COLUMN:CONTROL, e.g. --factor Diet:CD",
        "A --factor names a sample-sheet column and its control level, joined by "
        "a colon: --factor Diet:CD, --factor Day:1, --factor Sex:F. Every other "
        "level of the column is compared with the control level, and that is "
        "what gives every result its direction, so a --factor without a control "
        "level is refused rather than given one.\n\n"
        "The control level must be a value the column holds; the message lists "
        "the levels it does hold. To compare two non-control levels with each "
        "other, run again with one of them as the control.",
    ),
    ErrorSpec(
        114,
        "--rna-strandedness must be 0, 1, 2 or auto",
        "0 unstranded, 1 first-strand, 2 second-strand, auto to detect it",
        "The numbers mirror the featureCounts convention. Getting the direction "
        "wrong silently costs most of the reads, which looks like a biological "
        "result rather than a configuration error.\n\n"
        "`auto` is the default. It checks a small panel of genes against the "
        "largest BAM and prints what it decided and why. A library whose "
        "direction cannot be told is counted as unstranded (0), and that is "
        "printed and recorded beside the run: counting unstranded keeps every "
        "read, where a wrong direction loses them.",
    ),
    ErrorSpec(
        115,
        "This command needs a terminal, and this is not one",
        "Re-run it from a login shell, or pass the flag named in the message",
        "The command was about to ask you something -- which assay to use, "
        "whether to register, whether to withdraw for everybody -- and nothing "
        "is attached that can answer.\n\n"
        "This happens under SLURM, in a cron job, or with input redirected from "
        "/dev/null. In all three a prompt would hang the job rather than fail "
        "it, or be answered with an end-of-file that aborts the command "
        "half-way through something it had already started.\n\n"
        "NOTHING WAS WRITTEN. Every one of these checks runs before the first "
        "write, so there is no half-finished state to clean up and re-running "
        "is always safe.\n\n"
        "The console line names the flag that drives this particular command "
        "without asking -- --yes for a registration, a rename or a fatal "
        "withdrawal, --force for a removal, --dry-run for a sync. They are per "
        "command because the alternatives differ: a registration can be driven "
        "entirely by flags, a sync cannot, because it is a conversation about "
        "specific genomes by design.\n\n"
        "Interactivity is judged on BOTH streams, not on stdin alone. A job "
        "with input from /dev/null has a perfectly readable stdin that answers "
        "every prompt with EOF, and treating that as interactive is how the "
        "abort-half-way case arose.",
    ),
    ErrorSpec(
        116,
        "This command is not meant to run inside a SLURM job",
        "Run it on the login node; the analyses, batch and extract are submittable",
        "The rule is that only the genomics pipelines, batch, and extract need "
        "to be submittable. Everything else -- setup, the genome registry, "
        "collect, the reports -- is something a person does at a prompt and "
        "reads the answer to.\n\n"
        "THE ONE THAT MADE THIS NECESSARY is `paralome setup`. Your "
        "configuration lives in ~/.paralome/config.json, on shared storage, and "
        "every node sees the same file. The tool paths recorded "
        "in it are what every run actually invokes. So `paralome setup` inside "
        "a job records THAT COMPUTE NODE's paths into the file the login node "
        "reads -- silently repointing every future run, yours and everyone "
        "else's using that account, at one node's binaries. If they differ you "
        "get PLM-0410 everywhere; if they do not differ you get no error and no "
        "record that anything moved.\n\n"
        "The rest are refused for a smaller reason: they ask questions, write "
        "to a shared registry, or print a report nobody is watching. A job is "
        "not where any of that belongs.\n\n"
        "`batch` IS submittable, and it invokes the human command for each "
        "family -- which is why the analyses are allowed rather than the worker "
        "binary alone. `extract` is submittable too: reading a large object "
        "is work for a compute node, and it asks no questions.\n\n"
        "Detected from SLURM_JOB_ID. Running the same command on the login node "
        "does exactly what you meant.",
    ),
    ErrorSpec(
        117,
        "This run is not ready to start",
        "Fix everything listed above, then re-run; nothing was submitted",
        "Checked on the login node before anything was queued: that every "
        "sample is in place and not empty, and the logistics -- the genomes, "
        "the programs -- are there.\n\n"
        "EVERY PROBLEM IS LISTED, not the first one, as in the genome doctor: "
        "a verdict that names one "
        "cause invites fixing that cause and expecting to be done. A preflight "
        "that stopped at the first bad FASTQ would cost a fix-and-resubmit "
        "cycle for every bad row in the sheet.\n\n"
        "WHAT IS CHECKED. The sample sheet is OPENED, not merely found: every "
        "FASTQ or BAM it names has to exist and be non-empty, sample ids have "
        "to be unique. --genome has to resolve and the files it names have to "
        "be on disk. Any tool whose path `paralome setup` recorded has to still "
        "be where it was recorded, or be found on this account's PATH, where it "
        "is then recorded. On a machine with a shared root, checked "
        "first of all: the account has to be in the group of the "
        "shared genome repository, which an administrator fixes with usermod -aG <group> <account> "
        "and a fresh login.\n\n"
        "WHAT IS NOT. Whether a tool is missing altogether is `paralome setup "
        "doctor`'s question, and it depends on which stages your profile runs "
        "-- freebayes absent is fatal to DNA and irrelevant to scRNA. A stale "
        "RECORDING is different: it breaks whichever profile uses that tool, "
        "and it is PLM-0410 arriving here instead of at the stage that reaches "
        "for it.\n\n"
        "NOTHING WAS SUBMITTED AND NOTHING WAS WRITTEN. The whole point is that "
        "this runs before the first sbatch, so there is no queue slot to "
        "release and no half-finished project to clean up.",
    ),
    ErrorSpec(
        118,
        "--rds names neither a .rds object nor a folder",
        "Write one object as name.rds (or .qs2), or a folder of objects with a trailing /",
        "--rds takes a single Seurat object or a folder of objects. "
        "The way the path is written decides which, and the disk must agree: a "
        "name ending in .rds (or .qs2), in any case, is one object and must be a file; a "
        "name ending in / is a folder and must be a directory (even my.rds/); "
        "anything else is refused, because a guess here decides what the whole "
        "run analyses.\n\n"
        "Tab completion adds the / to a directory, so the folder form comes "
        "naturally at the prompt. From a folder every file ending in .rds is "
        "read; subfolders are not, and every other file is left alone.",
    ),
    ErrorSpec(
        119,
        "--group-by is retired",
        "Name the sample column with --replicate-column",
        "The user names one column, "
        "--replicate-column: the column whose value is a cell's biological "
        "sample (library, animal, donor). The pseudobulk samples are its values, "
        "one per --stratify-by value when that is given -- the samples "
        "--group-by <replicate>,<stratum> made.\n\n"
        "The --factor columns are not part of the grouping. Each must be one "
        "value per sample; a column that is not is refused (PLM-0203). "
        "--replicate-column is required on every single-cell run, pseudobulk "
        "and :cell alike.",
    ),
    ErrorSpec(
        120,
        "The Excel workbooks are retired",
        "Leave out the workbook option; the TSVs and CSVs hold every row",
        "Paralome writes no .xlsx files: not the statistic "
        "folders' results workbook, not the motif workbook and not the DNA "
        "report's client or bucket workbooks. OmicView reads the run's results "
        "(the statistics output) and its variant browser reads the segregation "
        "table; every row a workbook held is in the CSVs and TSVs beside it.\n\n"
        "So --workbook/--no-workbook, --bucket-workbooks and the older "
        "--max-workbook-rows and --bucketed-workbooks are refused by name rather "
        "than ignored.",
    ),
    # --- 0200-0299  the request versus the data -----------------------------
    ErrorSpec(
        201,
        "A named metadata column is not in the object",
        "Use one of the columns listed; `paralome extract` prints them all",
        "--replicate-column, --stratify-by and the --factor columns "
        "all name columns of the Seurat object's cell metadata. The column you "
        "asked for is not among them.\n\n"
        "The console lists every column that IS present, wrapped rather than "
        "truncated -- a list whose answer might be in the elided part is not "
        "worth printing. This is the one error allowed more than two lines, "
        "because the alternative is opening cell_metadata.tsv by hand.\n\n"
        "When an extraction already exists for this object, this is checked "
        "before submission. On a first run the columns are not knowable until "
        "the object has been read, so it is checked inside the job, at the "
        "earliest point the answer exists -- before any aggregation work.\n\n"
        "`paralome extract` prints every metadata column with its "
        "distinct values, which is the intended way to choose these before "
        "committing to a run.",
    ),
    ErrorSpec(
        202,
        "--replicate-column does not name one column",
        "Name the one column whose value is a cell's sample, e.g. --replicate-column orig.ident",
        "--replicate-column defines what a pseudobulk sample IS: cells sharing "
        "its value are summed into one (one per --stratify-by value when that is "
        "given). It names exactly one column; a second grouping is --stratify-by.",
    ),
    ErrorSpec(
        203,
        "A --factor is not one value within a pseudobulk sample",
        "Check that --replicate-column names the sample column, or fix the metadata",
        "A factor describes a sample, so every cell aggregated into one "
        "pseudobulk sample must carry the same value of it -- otherwise the "
        "sample sits in two levels at once and no comparison of the levels "
        "means anything.\n\n"
        "The usual cause is a --replicate-column that is not the sample "
        "column -- a cell type, a cluster or a batch that holds cells of both "
        "control and treated samples -- or metadata that gives one sample's "
        "cells two levels of the factor.",
    ),
    ErrorSpec(
        204,
        "--stratify-by is not constant within a pseudobulk group",
        "Check --replicate-column and --stratify-by",
        "A stratum is analysed separately, so each pseudobulk sample must "
        "belong to exactly one. The --stratify-by column is "
        "part of every sample's grouping, so this is reached only when the "
        "metadata itself is inconsistent.",
    ),
    ErrorSpec(
        205,
        "Named samples matched nothing",
        "The message lists the values that do exist",
        "A sample named on the command line or in the sample sheet is not in "
        "the data it has to be found in -- a VCF's samples, a manifest's "
        "sample-ID column or its full sample_ids.\n\n"
        "Most often a typo, or a value from a different column than the one "
        "you had in mind. The message prints what is there, so the right "
        "spelling is in front of you.",
    ),
    ErrorSpec(
        206,
        "The samples in scope matched no cells",
        "Use the sample_manifest.csv written from this extraction",
        "The cell-level statistics find each pseudobulk sample's cells by the "
        "metadata values the sample was aggregated from. None were found here: "
        "the manifest's samples are drawn from values the cell metadata does "
        "not hold, which means the manifest and the extraction are not from the "
        "same object.",
    ),
    ErrorSpec(
        207,
        "--remove excluded every sample",
        "Remove fewer samples",
        "Every pseudobulk sample was excluded, leaving nothing to compare.\n\n"
        "--remove matches a sample by its sample-ID value or its full pseudobulk "
        "sample_id -- both are tried, so a sample-ID value removes that sample "
        "from every cell type.\n\n"
        "The console lists what was removed. Naming full sample_ids is the "
        "unambiguous form.",
    ),
    ErrorSpec(
        208,
        "A level of a --factor has no samples",
        "Check the --factor control level and the column's values against the sample manifest",
        "A comparison needs samples on both sides. One level resolved to nothing, "
        "which means either the value is not in the column or every sample "
        "holding it was excluded by --remove or by a cell-count floor.",
    ),
    ErrorSpec(
        209,
        "Every group fell below the minimum cell count",
        "Lower --min-cells-per-group, or stratify less finely",
        "Groups with too few cells are dropped, because a pseudobulk value "
        "computed from a handful of cells is mostly noise. Here every group was "
        "dropped.\n\n"
        "A grouping that is too fine is the usual cause: --stratify-by "
        "multiplies the number of groups and divides the cells among them.",
    ),
    ErrorSpec(
        210,
        "The sample manifest is missing a required column",
        "Point at data/<modality>/sample_manifest.csv from a completed run",
        "The file given to --samples is not a Paralome pseudobulk manifest, or "
        "it predates the current layout. The expected columns are printed "
        "alongside the ones the file has.\n\n"
        "The manifest a differential stage wants is the one written by the "
        "aggregation stage, at data/<modality>/sample_manifest.csv. Pointing at "
        "a hand-made sample sheet, or at the bulk manifest handed to --samples "
        "on a FASTQ-based analysis, produces this: they are different files "
        "with different columns and the same intuitive name.",
    ),
    ErrorSpec(
        211,
        "The extraction's cell files disagree with each other",
        "Re-run the extraction with --overwrite",
        "cells.tsv and cell_metadata.tsv must list the same barcodes in the "
        "same order; they index the same count matrix columns.\n\n"
        "Disagreement means the extraction directory holds files from two "
        "different runs -- usually an interrupted extraction, or two runs "
        "writing to one directory. Nothing is salvageable by hand; re-extract.",
    ),
    ErrorSpec(
        212,
        "The extraction's peak files disagree with each other",
        "Re-run the extraction with --overwrite",
        "peaks.bed and peaks_order.tsv must agree on the peak order, since both "
        "describe the rows of counts.mtx. The redundancy is deliberate: two "
        "files stating the same order means a truncated or interleaved write "
        "can be caught instead of silently shifting every peak's counts.\n\n"
        "Same cause and same fix as PLM-0211 -- the directory holds files from "
        "two different runs. Re-extract rather than repairing by hand.",
    ),
    ErrorSpec(
        213,
        "The object has no assay of the kind this analysis needs",
        "Use the matching analysis, or name the assay with --assay / --rna-assay",
        "snATAC requires an ATAC half; scRNA requires an RNA half. The object "
        "has the other one, or neither.\n\n"
        "Extraction itself takes whatever the object has -- both halves of a "
        "multiome come out of a single read. It is the ANALYSIS that has a "
        "requirement, and it says so here rather than failing several stages "
        "later on a missing counts file.\n\n"
        "If the assay is present under an unusual name, --assay and "
        "--rna-assay name it explicitly. Note that `integrated` and `SCT` are "
        "never auto-selected: they are derived values, not counts.",
    ),
    ErrorSpec(
        214,
        "No ATAC or RNA assay could be found in the object",
        "Name the assays with --assay and/or --rna-assay",
        "Nothing in the object was recognisable as either half. An ATAC assay "
        "is identified by being a Signac ChromatinAssay, by being named "
        "ATAC/peaks/macs2, or by having genomic-interval rownames. An RNA assay "
        "is identified by being named RNA.\n\n"
        "The assays that WERE found are listed. Naming one explicitly overrides "
        "the detection entirely.",
    ),
    ErrorSpec(
        215,
        "A multiome object's two halves disagree about their cells",
        "Re-save the object with both assays over the same cells",
        "The ATAC and RNA assays hold different barcodes, or the same barcodes "
        "in different orders. Both are indexed by one shared cells.tsv, so "
        "Paralome refuses rather than picking one.\n\n"
        "This matters more than it looks: silently trusting one assay's order "
        "would misalign every barcode-keyed join downstream, and nothing later "
        "would raise. The result would be complete, plausible and wrong.",
    ),
    ErrorSpec(
        216,
        "A named assay is not in the object",
        "Use one of the assays listed, or omit the flag and let it be detected",
        "--assay or --rna-assay named an assay this object does not have. The "
        "assays it does have are listed.\n\n"
        "Usually the flag is unnecessary. Extraction detects assays "
        "structurally: features that are genomic intervals are ATAC, and "
        "everything else is gene-level -- RNA, Spatial, Xenium, ADT or whatever "
        "the object's author called it. The flags exist only to break a tie "
        "between several assays of the same kind.",
    ),
    ErrorSpec(
        217,
        "More than one assay of the same kind",
        "Name the one you want with --assay or --rna-assay",
        "The object holds several assays that are all ATAC, or several that are "
        "all gene-level -- an RNA+ADT object, for instance. Both are legitimate; "
        "neither can be chosen for you.\n\n"
        "No name settles it, not even RNA: picking one "
        "silently is how the wrong matrix gets analysed with nothing in the "
        "output saying so.\n\n"
        "The candidates are listed. Note that derived assays are already "
        "excluded: an SCTAssay is recognised by its class, and an assay with an "
        "empty counts layer -- which is what `integrated` usually is -- cannot "
        "be aggregated and is skipped.",
    ),
    ErrorSpec(
        218,
        "Every peak is wider than the motif scan allows",
        "Raise --max-peak-width above the widest peak named in the message",
        "Motif scanning fetches and scans the sequence under every peak, so its "
        "cost is proportional to total base pairs, not to peak count. "
        "--max-peak-width bounds that: a peak wider than it is SKIPPED, counted, "
        "and reported, and the scan proceeds on the rest.\n\n"
        "This error means every peak was skipped, so there is nothing left to "
        "scan -- which is a different situation from an empty matrix, and the "
        "only one where --max-peak-width is the answer.\n\n"
        "A single over-wide peak never aborts the analysis: it is skipped and "
        "counted. (On the snATAC path the peaks come from the user's Signac "
        "object, so rebuilding an atlas is not an action there.)",
    ),
    ErrorSpec(
        219,
        "Too many peaks are on contigs the reference FASTA lacks",
        "Register the genome the peaks were called on, or rename their contigs to the FASTA's",
        "The motif scan fetches each peak's sequence from the reference FASTA by "
        "contig name, trying the chr-prefixed and unprefixed spellings and the "
        "mitochondrial aliases. A peak whose contig is still not found is skipped, "
        "counted and reported, like a peak wider than --max-peak-width: "
        "single-cell peak sets routinely carry a few unplaced scaffolds "
        "(KI270728.1, GL000194.1) that a UCSC-style FASTA names differently "
        "(chrUn_KI270728v1), and one such peak must not abort the whole stage.\n\n"
        "Above one percent of the peak set, though, the missing names are not a "
        "few scaffolds but a naming convention: the peaks were called on a genome "
        "other than the one the motif stage was given. Scanning the rest would "
        "describe a different peak set from the one the differential tested, so "
        "the stage refuses and names the contigs. Skipped peaks are outside the "
        "enrichment universe: they are peaks that were not scanned, not peaks "
        "without a motif.",
    ),
    ErrorSpec(
        220,
        "The differential table does not name its comparison",
        "Pass --differential-checkpoint, or a table with control_condition/treatment_condition",
        "Given a differential CHECKPOINT (what the pipeline passes), the motif "
        "stage runs one enrichment per statistic folder and comparison the "
        "checkpoint records, each named after the comparison as the "
        "differential named it; a second table (--differential-statistics) "
        "beside it is refused.\n\n"
        "Given ONE table (--differential-statistics), there is one comparison "
        "and the results are named after it, from the table's own "
        "control_condition/treatment_condition columns. A table without them "
        "is refused rather than have its comparison named from the sample "
        "manifest's order, which reads 'up' backwards when the manifest lists "
        "the treated level first.",
    ),
    ErrorSpec(
        221,
        "The requested statistics do not fit these samples",
        "The message names the column, level or sample; check --factor, --pair-by and --covariate",
        "Each statistic in --statistic is planned against the sample sheet "
        "before anything is fitted: anova needs every --factor column present -- "
        "one factor of three or more levels (the one-way ANOVA), or two or more "
        "factors with samples in every combination of their levels -- "
        "and at least one error degree of freedom at pseudobulk; a factor of "
        "more than two levels must be numeric, so recode one that is not; "
        "paired-t needs each --pair-by pair to hold one sample in each arm; lm "
        "needs a numeric --covariate that varies; and every arm needs enough "
        "samples for the statistic's degrees of freedom. A request the samples "
        "cannot support is refused here, with the reason, rather than fitted "
        "into results that would not mean what their names say.\n\n"
        "Nothing was written. Correct the option or the sample sheet and run "
        "the same command again.",
    ),
    ErrorSpec(
        222,
        "The --rds folder holds no .rds object",
        "Point --rds at the folder that holds the objects, written with a trailing /",
        "A folder given to --rds is read for files ending in .rds, in any case, "
        "directly inside it. Subfolders are not read, and other files are left "
        "alone, so a folder of objects one level down, or of .qs2 files, holds "
        "none.",
    ),
    ErrorSpec(
        223,
        "Two objects in the --rds folder give the same barcode prefix",
        "Rename one of the files named in the message",
        "Barcodes repeat between samples, so each cell's barcode is prefixed "
        "with its file's name without .rds: HPC_03.rds gives HPC_03_<barcode>. "
        "Two files that differ only in the case of .rds (x.rds, x.RDS) would "
        "give the same prefix, and cells from different objects could then "
        "share a name.",
    ),
    ErrorSpec(
        224,
        "Objects in the --rds folder lack a metadata column",
        "Add the column to each object named, or change --replicate-column/--stratify-by",
        "The command line is the authority: every object in the folder must "
        "carry the columns --replicate-column and --stratify-by name. Each file is "
        "checked on its own, every failing file is listed with the column(s) "
        "it lacks, and nothing is stacked or analysed. No file is treated as "
        "the reference, because the first one read could be the broken one.\n\n"
        "Columns that only some objects carry, and that the command does not "
        "name, are left out of the combined metadata and listed.",
    ),
    ErrorSpec(
        225,
        "Objects in the --rds folder do not have the same genes",
        "Decide which file is wrong from the groups listed and fix or remove it",
        "The objects are grouped by identical gene list, and every group is "
        "reported: its files, its number of genes, the genes it has that "
        "another group lacks and the reverse (the first few on screen, every "
        "one in rds_folder_gene_groups.tsv in the extraction folder), and its "
        "match rate against the --genome GTF, as evidence rather than a "
        "ruling. Paralome never picks a winner.\n\n"
        "Genes are compared by Ensembl ID, and a gene the GTF gave none by its "
        "feature name. Matching by symbol silently pairs or drops genes renamed "
        "between annotation releases (Qk/Qki, Sept7/Septin7).\n\n"
        "Each object's extraction is kept, so once the file is fixed a rerun "
        "re-extracts only that file.",
    ),
    ErrorSpec(
        226,
        "Objects in the --rds folder do not have the same peaks",
        "Decide which file is wrong from the groups listed and fix or remove it",
        "As PLM-0225, for the ATAC half: the objects are grouped by identical "
        "peak set (chromosome, start, end), every group is reported, and the "
        "full lists are in rds_folder_peak_groups.tsv in the extraction folder. "
        "Peaks called separately per object differ, and counts on different "
        "peaks cannot be stacked into one matrix.",
    ),
    ErrorSpec(
        227,
        "The mass-spec export is not a layout Paralome reads",
        "Give the exporter's own table as exported: proteins, or phosphopeptides for --phospho",
        "Each exporter's table is recognised from its header: Proteome "
        "Discoverer protein and phosphopeptide exports, MaxQuant "
        "proteinGroups.txt, FragPipe combined_protein.tsv and DIA-NN "
        "pg_matrix.tsv. A table none of them claims is refused rather than "
        "read by guessing which columns are samples. The message names the "
        "file and the first columns found. A protein table given as --phospho "
        "(or the reverse) is refused the same way.",
    ),
    ErrorSpec(
        228,
        "The sample sheet does not match the export",
        "Make the sheet list every abundance column once, with a sample_id each",
        "The sample sheet has one row per sample: `column` (the abundance "
        "column's header, or the sample name inside it) and `sample_id`, then "
        "any metadata columns. Every abundance column of the export must be in "
        "the sheet and every row must name one. Every fault is listed at once, "
        "with the sample names the export holds. An option naming a sheet "
        "column (--factor, --stratify-by, --pair-by, "
        "--covariate) that the sheet does not have is refused here too.",
    ),
    ErrorSpec(
        229,
        "Whether the abundances are already logged cannot be told",
        "Export the values with a header that says log2 (or log10), or export them linear",
        "Paralome applies log2(x + 1) to linear values, so a logged table must "
        "be recognised. The header is read first; then the values: a negative "
        "value exists only on a log scale and no log2 intensity reaches 100. A "
        "table whose values all stay under 16 with a silent header could be "
        "log10, log2 or small linear numbers, and is refused rather than "
        "assumed. So is a table that contradicts itself (negative values that "
        "also reach 100, or a log header over values that reach 100).",
    ),
    ErrorSpec(
        230,
        "No stratum can be tested",
        "Read each stratum's reason and change what it names",
        "With --stratify-by, every stratum is tested on its own, and one that "
        "cannot be is skipped with its reason: an arm with too few samples, "
        "samples under --min-cells-per-group, or a column the statistic names "
        "(--covariate, --factor, --pair-by) that the sample manifest does not "
        "have. When every stratum is skipped there is nothing to test, and the "
        "message lists each one with its reason."
    ),
    ErrorSpec(
        231,
        "A fragment file the Signac object names cannot be found",
        "Fix the path in R with Signac's UpdatePath and save the object, or move the file there",
        "The snATAC genome tracks (the genome_tracks stage) draw each pseudobulk sample's "
        "profile from the fragment files the Signac object names. The object holds "
        "only each file's path and which cell is which barcode in it; the fragments "
        "stay in the file (fragments.tsv.gz, beside its .tbi index), and Paralome "
        "reads it from that path. A path written where the object was built, on "
        "another machine or before the data moved, is not there.\n\n"
        "The message lists every fragment file the object holds, found or missing. "
        "In R: Fragments(obj)[[i]] <- UpdatePath(Fragments(obj)[[i]], "
        "new.path = '<real path>') for each missing one, then saveRDS(obj, ...) "
        "and run again. An object that names no fragment file at all (its ATAC "
        "assay was built from a counts matrix alone) has nothing to draw "
        "profiles from."
    ),
    ErrorSpec(
        232,
        "The differential tested no peak",
        "Look at each comparison's tested count in the differential's results",
        "The genome tracks (the genome_tracks stage) export the viable peaks: the union of the "
        "peaks tested in any comparison, where a peak is tested when it reaches "
        "the expression floor (--min-mean-cpm) over that comparison's samples. A "
        "differential that tested none leaves nothing to export; the floor, or "
        "the samples each comparison reads, is the place to look."
    ),
    ErrorSpec(
        233,
        "The fragments are not sorted by contig",
        "Sort the BAM by coordinate (or the fragment file) and index it",
        "The genome tracks (the genome_tracks stage) read a sample's fragments one contig at a "
        "time, so it holds only that contig's coverage. A coordinate-sorted BAM "
        "and a tabix-indexed fragment file both come contig by contig; a contig "
        "that comes again after another means the file is not sorted."
    ),
    # --- 0300-0399  execution ------------------------------------------------
    ErrorSpec(
        301,
        "A SLURM stage failed",
        "Read the .err file named in the message -- that is the real error",
        "The supervisor reports only that a job failed. The cause is in the "
        "job's own .err file, whose path is printed.\n\n"
        "The stage's logs live under the stage's working directory, which for "
        "single-cell work is the SHARED data folder beside the projects, not "
        "inside the project directory -- looking in the project and finding no "
        "logs/ folder is expected, and is why the path is now printed rather "
        "than left to be deduced.\n\n"
        "A failure is announced when it happens (`Failed so far: ...`), but the "
        "stage does not stop there. The tasks still running are left to finish "
        "and this error is raised once nothing in the stage is running, so one "
        "failed piece of a large array does not cost the rest. What finished is "
        "checkpointed: fix the cause and re-issue the identical command, and "
        "only the failed pieces run again.",
    ),
    ErrorSpec(
        302,
        "A submitted job never appeared in the queue",
        "Check sinfo and the partition name",
        "The job was submitted and accepted, but neither squeue nor sacct ever "
        "reported it existing.\n\n"
        "Usually a partition that does not exist, or one your account cannot "
        "submit to -- sbatch accepts the job and the scheduler discards it. "
        "`sinfo` lists the partitions and their states.\n\n"
        "Paralome waits a grace period before concluding this, so a job that is "
        "merely slow to appear is not mistaken for one that never will.",
    ),
    ErrorSpec(
        303,
        "SLURM accounting is not answering",
        "Check that slurmdbd is running; re-run to resume",
        "sacct stopped responding for longer than the grace period, so the "
        "supervisor cannot tell whether jobs finished. It stops rather than "
        "assume either way.\n\n"
        "Work already completed is checkpointed. Re-running the same command "
        "resumes from the first incomplete stage.",
    ),
    ErrorSpec(
        304,
        "A stage exited with an error",
        "Read the stage output above; the log has the traceback",
        "Running locally rather than under SLURM, the stage runs as a "
        "subprocess and its own output is already on your terminal, above this "
        "message. That output is the authoritative error; this line only "
        "reports that the stage did not succeed.\n\n"
        "Work completed before the failure is checkpointed. Fix the cause and "
        "re-run the identical command to resume.",
    ),
    ErrorSpec(
        305,
        "One or more strata produced no result",
        "Read the .err for the named strata; the count is in the message",
        "The fan-out runs each stratum independently and collects failures "
        "rather than stopping at the first, so one bad stratum cannot hide the "
        "others.\n\n"
        "Under --backend slurm each stratum is its own array task with its own "
        "log, and the gather job runs on `afterany` -- so it runs even when "
        "some tasks failed, and its job is to name which. Seventeen of twenty "
        "succeeding is a partial result on disk, not a lost run: fix the three "
        "and re-issue the identical command.\n\n"
        "When ALL of them fail identically, the cause is the request, not the "
        "data: a column name, an arm member, a model that cannot run on this "
        "design. The distinct errors are summarised rather than repeated once "
        "per stratum.",
    ),
    ErrorSpec(
        306,
        "The log directory cannot be written to",
        "chown or chmod the directory; the console prints the exact command",
        "Paralome could not create or write its log directory. This is reported "
        "on the console rather than logged, for obvious reasons.\n\n"
        "A run whose logger cannot write is a run whose failures will be "
        "invisible, so this is fatal rather than a warning.",
    ),
    ErrorSpec(
        307,
        "The stage has more tasks than one SLURM job array can hold",
        "Lower the number of pieces, or ask the administrator to raise MaxArraySize",
        "SLURM limits the task numbers a job array may use. The highest "
        "allowed number is MaxArraySize minus one (`scontrol show config | "
        "grep MaxArraySize`), and Paralome numbers tasks from 1, so one array "
        "holds MaxArraySize minus one tasks.\n\n"
        "The stage named in the message would need a task number above that. "
        "Nothing was submitted.\n\n"
        "Stages that choose their own number of pieces stay inside the limit "
        "on their own. This error means the pieces are fixed by the data -- "
        "one per sample, or one per block of peaks -- and there are more of "
        "them than the scheduler allows in one array.",
    ),
    ErrorSpec(
        308,
        "A stage that has to run again would keep its old result",
        "Remove what the message names, then run the same command again",
        "Something earlier in the run changed -- a stage before this one has to "
        "run again, or the options this stage is given are no longer the ones "
        "it ran with -- so the results the stages named hold no longer answer "
        "this run. The message gives the reason for each: the stage it is "
        "behind, or the option that changed, from what to what.\n\n"
        "They cannot simply be started again. Each of them, started over a "
        "folder that already holds its output, finds that output and keeps "
        "it: the run would finish and report the old result as the new one. "
        "The message says 'started and did not finish' for a stage that was "
        "interrupted; what it would keep is then the part it had written.\n\n"
        "PARALOME DOES NOT REMOVE OR REPLACE RESULTS ON ITS OWN ACCOUNT. It "
        "adds no --overwrite to any stage; --overwrite reaches a stage only "
        "when you type it. The folders are yours, and what is in them may be "
        "the only copy of days of work. Every stage that is affected is named "
        "at once, with its folder and the reason, so the command is run again "
        "once and not once per stage. A folder shared with another stage says "
        "so: that stage will run again too.\n\n"
        "A FOLDER IS NAMED FOR REMOVAL ONLY IF IT IS THAT STAGE'S OWN, INSIDE "
        "THE PROJECT. A folder outside the project, the project folder itself, "
        "a folder that also holds the results of an earlier stage, and a "
        "folder that holds an input of this run are never named for removal: "
        "the single-cell aggregation, for one, writes into the study's data "
        "folder, which holds the extraction too.\n\n"
        "A stage folder that is a link inside the project to somewhere else "
        "(<project>/star linked to scratch) is the stage's own. It is named "
        "by its path in the project, and what is in it is what has to go; "
        "the link may stay.\n\n"
        "A STAGE FOLDER GIVEN OUTSIDE THE PROJECT (--align-workdir elsewhere) "
        "is named, and said to lie outside the project: Paralome cannot know "
        "what else is in it. The message says what of the stage has to be "
        "removed from it before the same command is run again. For the "
        "alignment that is the per-sample BAMs and their indexes "
        "(bam/*.sorted.bam and .bai) and its records "
        "(checkpoints/alignment.json, results/star_alignment_*.json). For any "
        "other stage it is everything in the folder. Removing a stage's "
        "report files alone does not do: the alignment would find its BAMs, "
        "align nothing and report them as the new result. The run is refused "
        "until what the message names is gone, and `status` goes by the same "
        "test.\n\n"
        "For a folder that is not named for another reason (the project "
        "folder itself, one shared with an earlier stage, one holding an "
        "input) the message says why and lists the stage's own files in it; "
        "for the alignment, again its BAMs, indexes and records.\n\n"
        "TWO WAYS ON. Remove what is named and run the same command "
        "again: those stages are then computed afresh. Or, where the reason "
        "is an option that changed, put the option back as it was: the "
        "results on disk answer that request, nothing was touched, and the "
        "run carries on with them. `--dry-run` shows the same refusal without "
        "starting anything.\n\n"
        "RNA quantification, differential expression, functional evidence, "
        "the two single-cell aggregations, variant calling and gene burden "
        "are never listed. Each checks its own result against what it is now "
        "given and computes again, replacing only its own files, when the "
        "result no longer answers; the run issues those again itself. Nothing "
        "was submitted.",
    ),
    ErrorSpec(
        309,
        "This stage was started with a different request",
        "Remove the stage folder to redo it, or repeat the earlier options",
        "The folder already holds finished samples, and they were made with "
        "other options, or from other input files, than this command gives. "
        "The message lists each such sample and what differs.\n\n"
        "A stage resumes by running only the samples that are not finished. "
        "Resumed under a different request, it would hold samples made two "
        "ways and report all of them under the new one, and nothing in the "
        "output would say which sample was made how.\n\n"
        "TWO WAYS ON. To finish the stage as it was started, repeat the "
        "earlier options. To redo every sample with the new ones, remove the "
        "stage folder and run again. Nothing is replaced without being asked, "
        "and nothing was started.\n\n"
        "Threads, memory and partition are not part of the request: they do "
        "not change a result, and changing them is not refused.",
    ),
    ErrorSpec(
        310,
        "Another supervisor is already running on this project",
        "Wait for that run, or stop it, then run the same command again",
        "One project is run by one supervisor at a time. The message names the "
        "host, the process and the start time of the one that holds it.\n\n"
        "The usual cause is a run started under nohup or in a terminal that "
        "was since lost, and the same command typed again. Two supervisors on "
        "one project both wait for the first one's jobs and then both submit "
        "the next stage, into the same folder.\n\n"
        "The supervisor that is running needs no help: it submits each stage "
        "when the one before it finishes. `paralome status --project` shows "
        "where it has got to. To take the project over, stop that process on "
        "the host named, then run the same command again; it resumes from "
        "what is finished.\n\n"
        "THERE IS NORMALLY NO STALE LOCK TO CLEAR. The lock is held by the "
        "running process and released by the kernel when it ends, however it "
        "ends. The lock file stays in checkpoints/ and does not need "
        "removing.\n\n"
        "The one exception is a network filesystem, which can keep a lock for "
        "a process that is gone. If the host named is down, or the process "
        "named no longer exists on it, and the lock is still held, the lock "
        "file named in the message may be removed; then run the same command "
        "again. Check the host and the process first: removing the file "
        "under a supervisor that is running lets a second one start beside "
        "it.\n\n"
        "Nothing was submitted.",
    ),
    ErrorSpec(
        311,
        "Jobs from an earlier submission may still be running",
        "Wait for them to finish, or scancel them, then run again",
        "This stage was submitted before from the same folder, and the jobs "
        "recorded for it are still in the queue -- or SLURM could not be "
        "asked whether they are, or this machine has no squeue to ask. The "
        "message says which, and lists the job ids.\n\n"
        "Submitted again beside them, every unfinished sample would be run "
        "twice at once, two jobs writing the same files.\n\n"
        "If the jobs are running, they are doing the work: wait for them, and "
        "run the command again only if something is then missing. To start "
        "over, scancel them first.\n\n"
        "WHEN SLURM DOES NOT ANSWER, nothing is submitted and nothing is "
        "cancelled. Not knowing whether work is running is no reason to "
        "duplicate it and no reason to stop it. squeue is asked again for ten "
        "minutes before this is reported; run the same command again when it "
        "answers.\n\n"
        "WHEN SQUEUE IS NOT ON THIS MACHINE, asking again cannot help, so the "
        "refusal comes at once and nothing is started. Run the command on "
        "the cluster, where squeue can be asked. Or, if you know those jobs "
        "have ended, remove the record of them that the message names and "
        "run again.",
    ),
    ErrorSpec(
        312,
        "A BAM that failed verification is still in place",
        "Remove the BAMs named and their .bai; a sample with no reads: '#' it out of the sheet",
        "`align verify` read the BAMs named in the message and rejected them; "
        "the reason is given beside each.\n\n"
        "Such a BAM is present and indexed, so alignment run again would take "
        "the sample as finished, and verification would reject it again, for "
        "as long as the file is there.\n\n"
        "It is not deleted for you: it is your output, and you may want to "
        "look at it first. Remove the BAMs named and their .bai files and run "
        "the same command again; only those samples are aligned. Removing the "
        "whole alignment folder aligns every sample again.\n\n"
        "A SAMPLE THAT ALIGNED NO READS is the exception, and the message "
        "lists such samples apart (\"Aligned no reads\"). Removing its BAM "
        "and aligning again gives the same empty BAM: an empty negative "
        "control, or FASTQ from another species. To go on without the "
        "sample, take it out of the sample sheet by putting '#' at the start "
        "of its line, and run again. If its FASTQ is the wrong file, correct "
        "that, then remove that sample's BAM and .bai and run again.\n\n"
        "Nothing was started.",
    ),
    # --- 0400-0499  environment ----------------------------------------------
    ErrorSpec(
        401,
        "That genome is not configured",
        "paralome setup genome lists what exists; setup genome add <folder> registers one",
        "--genome names a genome bundle registered in your Paralome "
        "configuration, not a path and not an assembly name in the abstract.\n\n"
        "`paralome setup genome` lists the genomes that exist, and "
        "`paralome setup genome add <folder>` registers one.\n\n"
        "Genome configuration is per-user, so a bundle registered by one "
        "account is not visible to another until it is adopted. If a colleague "
        "can use a genome you cannot, run `paralome setup genome sync` -- it "
        "takes anything registered on this machine into your own "
        "configuration without retyping a path.\n\n"
        "If this account cannot read part of the machine's registry "
        "or genome repository, the message says Permission denied and names "
        "each path, rather than calling the genome absent. Ask the "
        "administrator to set the permissions it names (usually: add the "
        "account to the shared repository's group, then log in again).",
    ),
    ErrorSpec(
        402,
        "A configured genome's files cannot be read",
        "Check the paths in paralome setup config, and their permissions",
        "The bundle is registered but a file it points at is missing or "
        "unreadable -- a moved reference, a deleted GTF, or a path pointing "
        "into another user's home directory.\n\n"
        "The console names the specific file, not just the bundle.",
    ),
    ErrorSpec(
        403,
        "Rscript was not found",
        "Load the R module, or pass --rscript with the full path",
        "Single-cell extraction shells out to R to read the Seurat object. "
        "Rscript is not on PATH.\n\n"
        "On a cluster this usually means an unloaded module -- and note that "
        "the compute node's environment is not your login shell's, so a module "
        "loaded interactively is not necessarily loaded inside the job.",
    ),
    ErrorSpec(
        404,
        "A required R package is not installed",
        "Install the package named into the R that Rscript actually resolves",
        "Seurat and Matrix are required to read the object; Signac is required "
        "for ChromatinAssay objects.\n\n"
        "Check WHICH R is being used. A package installed in an interactive "
        "session may not be visible to the Rscript a compute node resolves.",
    ),
    ErrorSpec(
        405,
        "A file that ships with Paralome is missing",
        "Reinstall the wheel with --force-reinstall",
        "A bundled resource -- the extraction R script, a motif database -- is "
        "absent from the installed package. The install is damaged or partial.\n\n"
        "Because Paralome versions below 1.0 sort under the old 1.0.x builds, "
        "pip may consider an older install newer and decline to replace it. "
        "--force-reinstall is required, not merely advisable.",
    ),
    ErrorSpec(
        406,
        "Gene annotation needs a registered genome",
        "Pass --genome; setup genome list shows what is registered here",
        "Gene identity is resolved once, at extraction, and stored -- so that "
        "every downstream comparison agrees on what a gene is.\n\n"
        "Without a GTF the run would produce a gene table with no stable "
        "identifiers, which is worse than not running: two comparisons of the "
        "same object could then disagree about which gene a row refers to.\n\n"
        "There is no --gtf. It would be an override on a "
        "value --genome already resolves, so a hand-typed GTF could disagree "
        "with the genome the run recorded, with nothing to say which one was "
        "used -- and the registered genome is what makes two runs comparable.",
    ),
    ErrorSpec(
        407,
        "That genome has been withdrawn as unusable",
        "Adopt the replacement, or ask its owner why it was withdrawn",
        "The account that registered this genome marked it FATALLY flawed, "
        "which means results produced with it are wrong rather than merely "
        "dated -- an index built against the wrong reference, a GTF that does "
        "not match the assembly, a truncated download.\n\n"
        "The run stops rather than warning, and it stops for the owner too. A "
        "fatally flawed genome produces wrong answers regardless of who runs "
        "it, and a warning would be read once and then scrolled past.\n\n"
        "This is NOT what a superseded genome does. A genome retired because "
        "something newer exists -- hg19 after hg38 -- warns and proceeds, "
        "because work already under way against it is still correct. Only a "
        "genome whose results cannot be trusted reaches this code.\n\n"
        "`paralome setup genome list` shows the reason its owner gave.",
    ),
    ErrorSpec(
        408,
        "Paralome is installed privately, in your home directory",
        "Ask an administrator to install it for the machine, then remove this one",
        "The copy you are running was installed with `pip install --user`, "
        "into your own home directory rather than for the machine.\n\n"
        "Nothing is broken. The objection is that it would work -- privately. "
        "A private install cannot publish to the machine's registry and cannot "
        "see what anyone else has published, so the reference genomes it uses "
        "are ones you build yourself, beside identical copies somebody has "
        "already built. That is the duplication the shared registry exists to "
        "end, and it is invisible from both sides: nobody else can see your "
        "genomes, and you cannot see theirs.\n\n"
        "It is also a version trap. Your own packages come before the system "
        "ones on Python's import path, so this copy is loaded even when the "
        "system command is invoked by its full path, and `paralome --version` "
        "reports through the same shadowed path. Two accounts on one machine "
        "can run different software while both answer to `paralome`.\n\n"
        "Removing it costs nothing. Your configuration and any genomes you "
        "registered live in ~/.paralome and are independent of where the code "
        "is installed, so they survive:\n\n"
        "    python3 -m pip uninstall paralome\n"
        "    which -a paralome\n\n"
        "Paralome is meant to be installed once for a machine, by whoever "
        "administers it, so that everyone on it shares one version and one set "
        "of reference genomes. If you do not have that access, the install is "
        "the administrator's to make.\n\n"
        "A virtualenv is not affected -- this is about `--user` specifically, "
        "not about whether the install needed root.",
    ),
    ErrorSpec(
        410,
        "A tool was recorded at a path that no longer exists",
        "Run `paralome setup` to re-record where the tools are now",
        "`paralome setup` records the absolute path of every external tool it "
        "finds, and the runner invokes those recorded paths "
        "rather than looking each tool up on PATH again at run time. The path "
        "is the one PATH finds, not where a symlink there points: "
        "Ubuntu's STAR (`apt install rna-star`) is a link to a program that "
        "chooses its build by the name it is called as, and recorded through "
        "the link it would not run.\n\n"
        "The path recorded for the tool named above is no longer there. That "
        "usually means the tool was upgraded, moved, or removed since setup "
        "last ran.\n\n"
        "WHY THIS REFUSES INSTEAD OF FALLING BACK TO PATH. The fallback is "
        "silent and the consequence is not. Two accounts with different PATHs "
        "-- which is exactly what happened at one site, where a local bin "
        "directory held newer builds of STAR and freebayes than /usr/bin -- "
        "would run different binaries for "
        "the same command and get different numbers, with nothing anywhere "
        "recording that they had. A stale recorded path is a configuration "
        "that needs fixing once; a silent substitution is a result nobody can "
        "reproduce.\n\n"
        "It is the same judgement made for genome files: a path "
        "that does not resolve stops the run at the start rather than "
        "surfacing later as a number that looks fine.\n\n"
        "`paralome setup doctor` shows every recorded tool and whether it is "
        "still there.\n\n"
        "The preflight first looks for a moved tool on this "
        "account's PATH and, when it finds it, records it there, says so, and "
        "the run uses it -- an administrator moving a tool is not something a "
        "user can know about. This error is what remains: a tool found neither "
        "where it was recorded nor on PATH.",
    ),
    ErrorSpec(
        409,
        "That genome is not in this machine's shared repository",
        "Put the folder in the repository the message names, then register it there",
        "This machine has a shared genome repository, and a genome registered "
        "anywhere else can only be used by the account that registered it.\n\n"
        "This refuses rather than warns because otherwise people take the easy "
        "route and register their own, which nobody else can use, and the "
        "machine ends up with a dozen human and a dozen mouse genomes.\n\n"
        "That is the failure the shared repository exists to prevent, and it is "
        "not a failure anyone notices at the time -- a genome in your home "
        "directory works perfectly for you. It shows up months later as four "
        "people holding four copies of GRCh38 and four different answers to "
        "which annotation release their results were against.\n\n"
        "Move the folder into the repository and register it from there. If it "
        "is large, move it rather than copying -- the repository is group "
        "writable for exactly this, and the sticky bit means only you can "
        "remove your own genome afterwards.\n\n"
        "If you cannot write to the repository, you are not in its group yet; "
        "an administrator adds you, and you log out and back in.\n\n"
        "A machine with no shared repository is unaffected: a workstation or a "
        "laptop registers genomes from wherever they are, as before.",
    ),
    ErrorSpec(
        411,
        "A configuration file has a key that nothing reads",
        "Correct or remove the key the message names; it lists the valid ones",
        "The file named in the message -- your configuration, a file given "
        "with --pipeline-config, or this machine's site.json -- holds a key "
        "no part of Paralome reads. Almost always it is a misspelling, and "
        "the nearest real key is suggested.\n\n"
        "The rule is that a setting that does not match fails, because such "
        "things ultimately bite the user: a mistyped key would otherwise load "
        "without complaint and do nothing, and the run would go ahead on the "
        "default the key was written to replace.\n\n"
        "Every command that reads the file refuses, not only the one the key "
        "was meant for: a setting that is silently ignored anywhere is the "
        "defect. `paralome setup doctor` and `paralome setup where` still "
        "report beside such a file, and name the key.\n\n"
        "YOUR OWN CONFIGURATION is yours to correct: edit the file, or remove "
        "it with `paralome setup reset`.\n\n"
        "SITE.JSON belongs to the machine's administrator, and you cannot "
        "correct it. Until they do, every run by every account on the machine "
        "is refused; nothing you can set overrides it. `paralome setup site` "
        "still shows the file and marks the key, for whoever does the "
        "editing. There is no reset for it.\n\n"
        "Names you choose yourself -- genomes, profiles, recorded tools -- are "
        "not checked; the keys inside each of them are.\n\n"
        "Two keys are refused although they were once read: "
        "annotation.annovar_buildver and annotation.snpeff_genome. A genome's "
        "build comes from the genome entry named by --genome, or from "
        "--annovar-buildver / --snpeff-genome, and a DNA run says so rather "
        "than load a build it will not use.",
    ),
    ErrorSpec(
        412,
        "A configuration file is not valid JSON",
        "Correct the file the message names; `python3 -m json.tool FILE` shows where it breaks",
        "The file named in the message -- your configuration, a file given "
        "with --pipeline-config, or this machine's site.json -- could not be "
        "read as JSON, or holds JSON that is not an object (a list, a "
        "string, a number). The message gives the line and column where "
        "reading stopped.\n\n"
        "The usual cause is an edit by hand: a comma after the last entry, a "
        "missing quote, a comment. JSON allows none of them. "
        "`python3 -m json.tool FILE` prints the same position and nothing "
        "else.\n\n"
        "Such a file is never read as an empty one: read as empty, the run "
        "would go ahead without any of the settings in it, which is the "
        "silent fallback Paralome refuses everywhere.\n\n"
        "YOUR OWN CONFIGURATION and a --pipeline-config file are yours to "
        "correct. `paralome setup reset` replaces your configuration with a "
        "new one if it is past repair.\n\n"
        "SITE.JSON belongs to the machine's administrator. Until they correct "
        "it, every run by every account on the machine is refused.\n\n"
        "This is not PLM-0106, which is about an input data file -- an .rds "
        "that is not an RDS. PLM-0411 is the same file with valid JSON and a "
        "key nothing reads.",
    ),
    ErrorSpec(
        413,
        "A configuration file holds a value that is not allowed",
        "Correct the value in the file the message names; it says which and what is allowed",
        "Your configuration file is valid JSON and every key in it is one "
        "Paralome reads, but a value is not one it accepts: a config_version "
        "other than the one this build writes, a required section missing or "
        "not an object, an rna.strandedness other than 0, 1, 2 or auto, or a "
        "negative rna.min_expression. The message names the file and the "
        "value.\n\n"
        "Every command that reads the file refuses, including the RNA "
        "expression gate: treating such a file as if it did not exist would "
        "fall back to the built-in 5 FPKM thresholds, so a configured "
        "rna.min_expression of 1 beside a misspelt key would silently become "
        "5, a gate the user never set. That is the silent fallback Paralome "
        "refuses everywhere.\n\n"
        "YOUR OWN CONFIGURATION is yours to correct: edit the file, or replace "
        "it with `paralome setup reset`. `paralome setup doctor` reports "
        "beside such a file and names the value.\n\n"
        "PLM-0411 is the same file with a key nothing reads; PLM-0412 is a "
        "file that is not JSON.",
    ),
    ErrorSpec(
        414,
        "The scheduler refused to accept a job",
        "The message carries sbatch's own sentence; run the printed command to see it in full",
        "sbatch exited non-zero, so nothing was queued. The useful part is the "
        "scheduler's own reason, which this message carries verbatim.\n\n"
        "WHY THIS EXISTS. Paralome submitted with check=True and reported the "
        "CalledProcessError, whose text is the command and the exit status and "
        "nothing else -- so a refused submission reached the user as 'returned "
        "non-zero exit status 1' with the reason captured and discarded. It "
        "happened twice before this code existed, and both times cost a "
        "re-run by hand to recover a sentence Paralome already had.\n\n"
        "Common reasons: a dependency on a job the scheduler no longer knows "
        "(the job it depended on had already left the queue), an invalid "
        "partition or account, a memory or CPU request above what any node "
        "offers, or a log directory that does not exist.\n\n"
        "Nothing was queued by the refused call, so the stage is not half "
        "submitted. Whatever was submitted BEFORE it is still queued and still "
        "running; fix the cause and re-issue the identical command, which "
        "reuses the completed work."
    ),
    ErrorSpec(
        415,
        "A required Python package is not installed",
        "Install the package named into the Python that runs Paralome",
        "The cell export (<rds>_cell_expression_embeddings.zip, written with every "
        "single-cell extraction) writes an .h5ad file with h5py.\n\n"
        "Paralome is installed with --no-deps into a prefix of its own, so the "
        "packages it uses come from the Python that runs it. Install h5py there: "
        "python3 -m pip install h5py. A compute node resolves the same Python only "
        "when the module loads it, so install it where the jobs run."
    ),
    ErrorSpec(
        416,
        "bedGraphToBigWig or bedToBigBed is not installed",
        "conda install -c bioconda ucsc-bedgraphtobigwig ucsc-bedtobigbed, then paralome setup",
        "The genome tracks (the genome_tracks stage) are written with the "
        "UCSC tools: the bigWig profiles with bedGraphToBigWig, the peak and gene tracks with "
        "bedToBigBed. Paralome does not ship them; install them where the jobs "
        "run, and `paralome setup` records their paths as it does every other "
        "tool's. UCSC's licence lets its command-line tools be used freely, "
        "commercial use included (all but BLAT and LiftOver)."
    ),
    # The trajectories stage.
    ErrorSpec(
        121,
        "The trajectories need a root, given as <column>:<value>",
        "Add --trajectory-root <column>:<value> (e.g. celltype:RPC), or --no-trajectories",
        "The trajectories stage (scRNA) fits the object's lineages with Slingshot, starting from the "
        "cluster the run names. Which cluster is the start is the analyst's knowledge, so it is never guessed: "
        "a run that reaches the stage names it, or leaves the stage out with --no-trajectories and adds it later "
        "by running again with --trajectory-root once the UMAP has been seen. The column is the cell-metadata "
        "column whose clusters the lineages are fitted on (celltype, seurat_clusters ...); the value is one of them."
    ),
    ErrorSpec(
        122,
        "--trajectory-bins must be 2 or more",
        "Give --trajectory-bins 2 or more (the default is 5)",
        "Each lineage is cut into this many segments of equal numbers of its cells by pseudotime, and each "
        "replicate's share of the lineage's cells in each segment is compared by the run's statistics ("
        "the composition only). One segment would be the whole lineage."
    ),
    # The trajectories stage's gene model and the lineages it follows.
    ErrorSpec(
        128,
        "--trajectory-model is not an engine and its df",
        "Give ls or monocle3, the spline's df after a colon: ls:df=4",
        "Each gene along each lineage's pseudotime is modelled as the run's full factorial crossed with a natural "
        "spline of pseudotime: by least squares on the cells' log2(CPM+1) (ls, the default) or by Monocle3's "
        "fit_models on the counts (monocle3). df is the spline's degrees of freedom, 3 unless given; 1 is a "
        "straight line."
    ),
    ErrorSpec(
        129,
        "--trajectory-lineages is not fate, exit or route",
        "Give --trajectory-lineages fate, exit or route",
        "The trajectory statistics follow one lineage per end state (fate, the default: every route the fit made "
        "that ends in the same cluster), one per way out of the root's cluster and end state (exit), or every route "
        "(route)."
    ),
    # The fit's cells and dimensions.
    ErrorSpec(
        123,
        "--trajectory-fit-cells is not a setting it knows",
        "Give e.g. 100:random, or cells=1000,dispersion=random,stretch=0,extend=n",
        "The lineages' curves are fitted on n cells of each cluster (every cell of a cluster smaller than n), "
        "drawn at random (seeded, so a run repeats) or the n nearest the cluster's centre in the reduction the fit "
        "uses; every cell is then placed on the fitted curves (slingshot's predict). Without the option every cell "
        "is fitted. slingshot.json records the cells per cluster, how they were chosen and how many were fitted. "
        "As key=value pairs separated by commas -- cells=<n>, dispersion=random|centroid (random when "
        "cells is given alone), stretch=<s> (0 or more: how far each curve runs past its last cluster; Slingshot's "
        "default 2), extend=y|n|pc1 (how a curve's ends are drawn; default y), reduction=<name> (the reduction the "
        "curves are fitted in, as the extraction named it, e.g. umap; by default the object's integrated one, else "
        "pca). Each key once; dispersion needs cells. Df=<n> (more than 1: the smoothing spline's degrees "
        "of freedom for every curve; 2 is close to straight; by default smooth.spline chooses them) and "
        "shrink=<0-1> (how far the lineages' shared start is pulled together; Slingshot's default 1). "
        "under --trajectory-method monocle3 (the default) the keys are Monocle3's -- reduction=<name> (the 2D/3D "
        "UMAP the graph is learned in), ncenter=<n>, minimal_branch_len=<n>, use_partition=y|n; also "
        "close_loop=y|n, geodesic_distance_ratio=<x>, euclidean_distance_ratio=<x>, orthogonal_proj_tip=y|n, "
        "prune_graph=y|n, l1_gamma=<x>, l1_sigma=<x> (x more than 0) -- and Slingshot's are refused, as Monocle3's "
        "are under slingshot."
    ),
    # Monocle3 beside Slingshot.
    ErrorSpec(
        125,
        "The trajectory method is unknown, or the option is not its",
        "Give --trajectory-method monocle3 or slingshot; leave --trajectory-dims out with monocle3",
        "The lineages are fitted by Monocle3 (the default: one principal graph learned through the cells in the "
        "object's UMAP, each path from the root to a leaf of the graph a lineage) or by Slingshot (curves through "
        "the clusters' centres, fitted in a reduction of many dimensions). --trajectory-dims chooses those "
        "dimensions, so it is Slingshot's; Monocle3 learns its graph in the UMAP."
    ),
    ErrorSpec(
        127,
        "A --trajectory-weld cannot be made",
        "Name tips: L<a>=L<b>, L<a>=L<b>=L<c>, L<a>~L<b> or 'L<a>->L<b>'",
        "--trajectory-weld joins tips of Monocle3's learned graph, named as its lineages are in "
        "<run>/trajectories/fit/tips.tsv: L39=L42 joins the two tips, L30=L33=L35 joins each to a new node at "
        "their middle, L39~L42 joins L39's tip to the nearest point of L42's own branch, 'L39->L42' joins L39's tip "
        "to L42's and L42's stays an end (quoted: unquoted, the shell reads > as a redirect). A tip the graph does not "
        "have is refused (the error says how many it has). The lineages are every route from the root to each "
        "end; welds that make more than 200 are refused: weld fewer tips, or with ~."
    ),
    ErrorSpec(
        126,
        "The object has no UMAP for Monocle3 to learn its graph in",
        "Name one with --trajectory-fit-cells reduction=<name>, or --trajectory-method slingshot",
        "Monocle3 learns its principal graph in a 2D or 3D UMAP: the object's (a 2D/3D reduction with umap in its "
        "name, or its only 2D/3D one), or the one reduction=<name> names. The error lists the object's 2D/3D "
        "embeddings."
    ),
    ErrorSpec(
        124,
        "--trajectory-dims is outside the reduction's dimensions",
        "Give 2 up to the number of dimensions the error names",
        "The lineages are fitted on the first n dimensions of the reduction the stage chose (an integrated one, else "
        "pca; snATAC's lsi without its first). Every distance the fit measures -- between the clusters' centres, and "
        "from each cell to each curve -- is measured in those dimensions. Without the option all of them are used."
    ),
    ErrorSpec(
        234,
        "The trajectory root is not in the object's cell metadata",
        "Name a column the object has and one of its values (the error lists them)",
        "--trajectory-root <column>:<value> is read against the extraction's cell_metadata.tsv: every metadata "
        "column the object holds. The column must be there and at least one cell must have the value, spelled as "
        "the object spells it."
    ),
    ErrorSpec(
        235,
        "The object holds no reduction the lineages can be fitted on",
        "Run PCA (scRNA) on the object, or an integration, save it, and run again",
        "The lineages are fitted on the object's own reduction: an integrated one (a name with harmony or "
        "integrated) when it has one, else pca. Only a reduction of more than three dimensions is used; a UMAP or "
        "t-SNE is for drawing the curves, not fitting them. The extraction writes every reduction the object holds "
        "(extraction/reductions/)."
    ),
    ErrorSpec(
        236,
        "No trajectory replicate is a sample of the run's pseudobulk",
        "Check --replicate-column and --remove: they decide the run's samples",
        "The composition of each replicate along and between the lineages is compared with the run's own "
        "statistics, so each replicate must be one of the run's pseudobulk samples (its manifest gives the "
        "replicate's factor levels)."
    ),
    ErrorSpec(
        237,
        "The trajectories stage does not run on this modality yet",
        "Run it on an scRNA analysis, or add --no-trajectories",
        "Paralome builds the trajectories for scRNA objects: the genes along each lineage are the scRNA "
        "extraction's, and the composition is tested by the scRNA differential. An snATAC run has no "
        "trajectories stage."
    ),
    ErrorSpec(
        238,
        "The trajectories were given no statistic to run",
        "Pass the run's --statistic",
        "The trajectories' composition is tested by the run's own --statistic, at the replicate level ("
        "the genes along each lineage by --trajectory-model). With none there is nothing to run; the pipeline "
        "always passes the run's, so this is met only by running the worker command by hand."
    ),
    ErrorSpec(
        239,
        "The --covariate-feature gene is not one gene of the object",
        "Name the gene as the object's features name it (its symbol or its Ensembl ID)",
        "With lm on a gene (--covariate-feature), the trajectories' composition (each replicate's shares of cells "
        "along and between the lineages) is regressed on that gene's expression in each replicate, every cell of "
        "the replicate summed. The gene must be exactly one of the object's genes."
    ),
    ErrorSpec(
        240,
        "The --plate file is not an Olink plate export",
        "Give each plate as Olink exported it (NPX Signature long format), once per --plate",
        "paralome olink reads Olink's long export by its column names: PlateID, SampleID, SampleType, "
        "Assay, UniProt, OlinkID, QuantifiedValue, BelowLOD and Unit, one row per sample and assay. CSV (comma, "
        "semicolon or tab), .xlsx and .parquet are read. A PlateID may be in only one --plate file."
    ),
    ErrorSpec(
        241,
        "A plate shares no sample control with the first plate",
        "Put first the plate that shares controls with every other, or give the plates that do",
        "Every plate after the first --plate is brought onto the first plate's scale, assay by assay, by the median "
        "ratio of the SAMPLE_CONTROL wells both plates ran (matched by SampleID). A plate with none in common with "
        "the first cannot be bridged to it."
    ),
    ErrorSpec(
        242,
        "The Olink sample sheet cannot be read",
        "Give a CSV or tab-separated sheet with PlateID, SampleID and sample_id, one row per well",
        "Each row is one well of a --plate (PlateID and SampleID as the export gives them) and the sample it is "
        "(sample_id), then the design's columns. A well is listed once; wells given the same sample_id are "
        "averaged into one sample."
    ),
    ErrorSpec(
        243,
        "The Olink sample sheet does not match the plates",
        "List only SAMPLE wells of the given plates; give averaged wells the same design values",
        "The sheet's wells must be in the plates, and be study samples (the SAMPLE_CONTROL wells bridge the plates "
        "and are not tested). Wells averaged into one sample_id must agree on every design column. The columns the "
        "options name (--factor, --pair-by, --block-by, --covariate, --stratify-by) must be the sheet's."
    ),
    ErrorSpec(
        244,
        "There is no trajectory fit for the trajectories stage",
        "Run the pipeline again: singlecell_rna fits the lineages before the trajectories stage",
        "The singlecell_rna stage fits the lineages, right after its extraction and pseudobulk, "
        "into <run>/trajectories/fit (an older run's is <run>/trajectories/slingshot), and writes them into "
        "the run's cell export; the trajectories stage then runs the statistics on that fit. Run by hand, the "
        "worker command names the fit's folder with --fit."
    ),
    # The genome's UniProt table.
    ErrorSpec(
        245,
        "The genome's uniprot_table is not Ensembl's UniProt table",
        "Put Ensembl's <Species>.<assembly>.<release>.uniprot.tsv.gz in the genome folder",
        "Ensembl publishes each release's UniProt cross-references at ftp.ensembl.org/pub/release-<n>/tsv/"
        "<species>/ (e.g. Homo_sapiens.GRCh38.81.uniprot.tsv.gz); Paralome reads its gene_stable_id and xref "
        "columns to find a protein named by no gene symbol (an Olink assay such as Abeta40) by its UniProt "
        "accession. `paralome setup genome add <folder>` registers the one file in the genome folder ending "
        ".uniprot.tsv or .uniprot.tsv.gz; use the release of the genome's GTF."
    ),
    ErrorSpec(
        313,
        "Slingshot failed",
        "Read the slingshot.log the error names in the trajectories folder",
        "The trajectory fit runs slingshot_trajectories.R in the R --rscript names. Its output, and the R error "
        "when it stops, is in <run>/trajectories/fit/slingshot.log (an older run's is in "
        "<run>/trajectories/slingshot/)."
    ),
    ErrorSpec(
        314,
        "Monocle3 failed",
        "Read the monocle3.log the error names in the trajectories folder",
        "The trajectory fit runs monocle3_trajectories.R in the R --rscript names. Its output, and the "
        "R error when it stops, is in <run>/trajectories/fit/monocle3.log. A graph whose part holding the root has "
        "no leaf has no lineage to follow: try use_partition=n, or a smaller minimal_branch_len."
    ),
    ErrorSpec(
        315,
        "Monocle3's fit_models failed on the trajectory gene models",
        "Read monocle3_gene_models.log in the trajectories expression folder",
        "--trajectory-model monocle3 fits each lineage's gene models with Monocle3's fit_models, in "
        "monocle3_gene_models.R in the R --rscript names. Its output, and the R error when it stops, is in "
        "<run>/trajectories/expression/monocle3_work/monocle3_gene_models.log. --trajectory-model ls fits the "
        "same models by least squares."
    ),
    ErrorSpec(
        417,
        "slingshot is not installed in the R the stage runs",
        "In that R: BiocManager::install(\"slingshot\"), or add --no-trajectories",
        "The trajectories stage fits the lineages with the Bioconductor package slingshot, in the R that "
        "runs the extraction (--rscript, default Rscript). Paralome does not install R packages; install it there, "
        "where the jobs run. The preflight checks for it before anything is submitted -- and,, for "
        "DelayedMatrixStats, which slingshot calls while it fits."
    ),
    ErrorSpec(
        418,
        "Monocle3 is not installed in the R the trajectory fit runs",
        "remotes::install_github(\"cole-trapnell-lab/monocle3\") in that R",
        "The default --trajectory-method fits the lineages with Monocle3, in the R that runs the "
        "extraction (--rscript, default Rscript). Monocle3 is installed from GitHub, after its dependencies "
        "(leidenbase, sf, terra, BPCells among them). Paralome does not install R packages; install it there, where "
        "the jobs run. The preflight checks that it loads before anything is submitted."
    ),
)

REGISTRY: Mapping[int, ErrorSpec] = {spec.code: spec for spec in _SPECS}

if len(REGISTRY) != len(_SPECS):  # pragma: no cover - defended by tests too
    raise RuntimeError("Duplicate error code in the Paralome error registry")


def lookup(code: Union[int, str]) -> Optional[ErrorSpec]:
    """Find a spec by number, accepting ``104``, ``"104"`` or ``"PLM-0104"``."""
    if isinstance(code, int):
        return REGISTRY.get(code)
    text = str(code).strip().upper()
    for prefix in ("PLM-", "PLM"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.lstrip("-").lstrip("0") or "0"
    try:
        return REGISTRY.get(int(text))
    except ValueError:
        return None


class ParalomeError(RuntimeError):
    """A failure with a number, a one-line cause, and a one-line instruction.

    ``detail`` is the specific, contextual sentence -- which file, which column,
    which value -- and IS printed, because a code without the specifics is a
    riddle. ``fix`` overrides the registry's generic instruction when the
    situation allows a better one (naming the similarly-spelled file that does
    exist, say). ``context`` is arbitrary diagnostic material for the log only.

    WHY IT SUBCLASSES RuntimeError. Numbering a failure replaces a `raise
    RuntimeError(...)` that callers may already be catching (the supervisor's
    SLURM failures are the ones most likely to be wrapped in someone's script).
    Deriving from RuntimeError keeps `except RuntimeError` working, while the
    user gets two lines and a number instead of a traceback.
    """

    def __init__(
        self,
        code: int,
        detail: str = "",
        *,
        fix: Optional[str] = None,
        **context: object,
    ) -> None:
        spec = REGISTRY.get(code)
        if spec is None:  # pragma: no cover - a coding error, caught by tests
            raise KeyError(f"No such Paralome error code: {code}")
        self.code = code
        self.spec = spec
        self.detail = detail
        self.fix = fix or spec.fix
        self.context = context
        super().__init__(f"{format_code(code)} {spec.summary}"
                         + (f": {detail}" if detail else ""))

    def __reduce__(self):
        # The default rebuilds as cls(*args), where args is the
        # message string, so a refusal raised in a process-pool worker failed
        # to unpickle and the parent lost the number. Rebuilt from state, so
        # subclasses with their own __init__ come back intact too.
        state = dict(self.__dict__)
        try:
            pickle.dumps(state.get("context", {}))
        except Exception:
            state["context"] = {k: repr(v) for k, v in state.get("context", {}).items()}
        return (_rebuild_paralome_error, (type(self), self.args, state))


def _rebuild_paralome_error(cls, args, state):
    error = cls.__new__(cls, *args)
    error.args = args
    error.__dict__.update(state)
    return error


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def render_console(error: ParalomeError, log_path: Optional[Path] = None) -> str:
    """The whole terminal output. Two lines:

        PLM-0104  An input file does not exist
          --rds 'cohort_integrated_rds.rds' does not exist.

    The number and what went wrong. Nothing else. The fix, the log path, the
    near-miss suggestion and the traceback are all in the log and behind
    ``paralome explain`` -- reachable from the number, which is the point of
    having numbers.

    ``log_path`` is accepted and deliberately not printed: callers still pass it
    so the detail is written.
    """
    lines = [f"{format_code(error.code)}  {error.spec.summary}"]
    for line in str(error.detail).splitlines():
        if not line.strip():
            continue
        # A detail line may legitimately be long -- PLM-0201 carries every
        # metadata column the object has, which is the thing that stops the user
        # opening cell_metadata.tsv by hand. Wrapped, not truncated: a list whose
        # answer might be in the part that was cut is not worth printing.
        lines.extend(textwrap.wrap(
            line.strip(), width=76, initial_indent="  ", subsequent_indent="    ",
            break_long_words=False, break_on_hyphens=False,
        ) or [f"  {line.strip()}"])
    return "\n".join(lines)


def write_details(
    error: BaseException,
    log_path: Path,
    *,
    argv: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Append the long version to the log. Never raises.

    A failure to record a failure must not replace it -- the original error is
    what the user needs, and losing it to a permissions problem in the log
    directory would turn a diagnosable run into a silent one.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + "=" * 72 + "\n")
            if argv is not None:
                handle.write("command: " + " ".join(str(a) for a in argv) + "\n")
            handle.write(f"cwd: {os.getcwd()}\n")
            if isinstance(error, ParalomeError):
                handle.write(f"{format_code(error.code)} {error.spec.summary}\n")
                if error.detail:
                    handle.write(f"{error.detail}\n")
                handle.write(f"fix: {error.fix}\n")
                handle.write(f"explain: paralome explain {error.code}\n")
                for key, value in error.context.items():
                    handle.write(f"  {key}: {value!r}\n")
            handle.write("".join(traceback.format_exception(
                type(error), error, error.__traceback__
            )))
        return log_path
    except OSError:
        return None


def _default_log_path(argv: Sequence[str]) -> Path:
    """Where details go when the failing command did not nominate a directory.

    Prefers a --project or --out named on the command line, so the record lands
    beside the work it describes; falls back to the current directory.

    Only a folder that is already there: a run refused before it started must
    not leave a project folder behind.
    """
    for flag in ("--project", "--out"):
        if flag in argv:
            position = list(argv).index(flag) + 1
            if position < len(argv) and Path(argv[position]).is_dir():
                return Path(argv[position]) / "logs" / "paralome.err"
    return Path("paralome.err")


def install_error_handler(app_callable):
    """Wrap a Typer app so failures print short and log long.

    Returns a zero-argument callable suitable for a console-script entry point.

    Typer's own pretty tracebacks are disabled at the app level (see cli.py and
    worker.py); this is the other half, and it is what turns an unnumbered
    exception into PLM-0001 rather than a stack dump.

    Takes no program name on purpose: the human binary's name is not spelled
    as a string literal outside the modules whose job is naming it.
    """

    def main() -> None:
        argv = list(sys.argv)
        # A dry run writes nothing, a refusal's log included. The
        # refusal is on the console in full. An unnumbered failure is a defect
        # and its traceback is still kept, in the current directory.
        dry_run = "--dry-run" in argv
        try:
            app_callable()
        except ParalomeError as error:
            written = None if dry_run else write_details(error, _default_log_path(argv), argv=argv)
            print(render_console(error, written), file=sys.stderr)
            raise SystemExit(1)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as error:  # noqa: BLE001 - the catch-all is the point
            wrapped = ParalomeError(1, f"{type(error).__name__}: {error}")
            wrapped.__traceback__ = error.__traceback__
            log_path = Path("paralome.err") if dry_run else _default_log_path(argv)
            written = write_details(error, log_path, argv=argv)
            print(render_console(wrapped, written), file=sys.stderr)
            raise SystemExit(1)

    return main


# ---------------------------------------------------------------------------
# The manual's appendix, generated
# ---------------------------------------------------------------------------

_BLOCK_TITLES = {
    0: "General",
    1: "What the command said (0100-0199)",
    2: "The request versus the data (0200-0299)",
    3: "Execution (0300-0399)",
    4: "Environment (0400-0499)",
}


def render_appendix() -> str:
    """The error appendix, from the registry.

    The long descriptions live in an appendix at the back of the manual, and
    they are GENERATED from the registry, so the manual cannot describe an
    error differently from the code.
    """
    lines = [
        "## Appendix A — Error codes",
        "",
        "Every numbered failure Paralome can report. The console prints the code,",
        "what went wrong and the specifics; the fix and the rest are here, in",
        "the run's log, and from `paralome explain`.",
        "",
        "`paralome explain 104` prints any entry below without leaving the terminal.",
        "",
        "Numbers are grouped in blocks of a hundred. The block says where the failure",
        "lives: what you typed, what your data is, how it ran, or how the machine is",
        "set up. New codes append to a block and never renumber an existing one.",
        "",
    ]
    for block in sorted({code // 100 for code in REGISTRY}):
        lines.append(f"### {_BLOCK_TITLES.get(block, f'{block}xx')}")
        lines.append("")
        for code in sorted(c for c in REGISTRY if c // 100 == block):
            spec = REGISTRY[code]
            lines.append(f"#### {format_code(code)} — {spec.summary}")
            lines.append("")
            lines.append(f"**Fix:** {spec.fix}")
            lines.append("")
            lines.extend(spec.explanation.split("\n"))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
