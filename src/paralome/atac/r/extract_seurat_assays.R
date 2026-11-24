#!/usr/bin/env Rscript
#
# extract_seurat_assays.R
#
# It does not pseudobulk anything: it writes the counts matrix with ONE
# COLUMN PER CELL (tens of thousands of them in a typical object). The
# collapse to one value per sample x stratum group happens afterwards, in
# Python, in singlecell/pseudobulk_core.py. In this repository `pseudobulk`
# names the aggregation and nothing else.
#
# Internal helper shipped with Paralome (see paralome.atac.singlecell). Users
# do not invoke this directly -- the single-cell stage shells out to it
# automatically. Its ONLY job is to read a single Seurat (Signac) object and
# dump exactly the flat files Paralome's Python side needs, in a
# deterministic, machine-friendly shape.
# All grouping/aggregation/statistics logic stays on the Python side, so it
# stays testable and versionable without requiring R for every iteration.
#
# Usage:
#   Rscript extract_seurat_assays.R \
#       --rds /path/to/object.rds \
#       --out /path/to/output_dir \
#       [--assay ATAC] [--slot counts] [--rna-assay RNA] [--rna-slot counts]
#
# Neither half is mandatory and neither is requested: the script reads the
# object's assays and exports whichever of ATAC and RNA it finds. --assay and
# --rna-assay exist only to break ties. The ATAC outputs (peaks.bed,
# peaks_order.tsv, peaks_features.tsv, counts.plmsp) and the RNA outputs
# (rna_counts.plmsp, rna_genes.tsv, rna_genes_order.tsv) are therefore both
# conditional; extraction_manifest.json's has_atac/has_rna record what was
# actually written.
#
# Outputs written into --out:
#   peaks.bed           BED4 (chrom, start, end, peak_id), one row per peak,
#                       in the same row order as counts.plmsp.
#   peaks_order.tsv     Single-column peak_id list confirming counts.plmsp row
#                       order (defensive redundancy with peaks.bed; Python
#                       validates the two agree before trusting either).
#   counts.plmsp        Binary sparse peak-by-cell counts matrix. See
#                       plm_write_sparse below for the layout. Python reads it
#                       with paralome.atac.singlecell._read_plmsp, and reads a
#                       MatrixMarket counts.mtx when an extraction has one.
#   cells.tsv           Single-column barcode list in counts.plmsp column order.
#   cell_metadata.tsv   Full meta.data (every column, unfiltered) plus a
#                       leading `barcode` column in the same row order as
#                       cells.tsv.
#   extraction_manifest.json
#                       Small provenance record (assay used, cell/peak
#                       counts, package versions) for the Python side to
#                       fold into its own stage checkpoint.

suppressWarnings(suppressMessages({
  library(methods)
}))

parse_args <- function(argv) {
  # RNA and ATAC are exported in the same run. --rna-assay/--rna-slot/
  # --gene-symbol-column are all optional; --rna-slot defaults to --slot.
  opts <- list(rds = NULL, out = NULL, assay = NULL, slot = "counts",
               rna_assay = NULL, rna_slot = NULL, gene_symbol_column = NULL)
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--rds") { opts$rds <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--out") { opts$out <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--assay") { opts$assay <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--slot") { opts$slot <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--rna-assay") { opts$rna_assay <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--rna-slot") { opts$rna_slot <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--gene-symbol-column") { opts$gene_symbol_column <- argv[[i + 1]]; i <- i + 2 }
    else stop(paste0("Unrecognized argument: ", key))
  }
  if (is.null(opts$rds) || is.null(opts$out)) {
    stop("--rds and --out are required")
  }
  if (is.null(opts$rna_slot)) opts$rna_slot <- opts$slot
  opts
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))

if (!requireNamespace("Seurat", quietly = TRUE)) {
  stop("The Seurat package is required to read the .rds object but is not installed.")
}
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("The Matrix package is required to export the counts matrix but is not installed.")
}

dir.create(opts$out, recursive = TRUE, showWarnings = FALSE)

# --- Timing -------------------------------------------------------
#
# Extraction can take far longer than the statistics. Every expensive step is
# timed and the breakdown is printed at the end, so a slow extraction reports
# its own profile.
#: Elapsed at the top of the script. proc.time() counts from R process start,
#: so this IS R's own startup plus loading the packages above -- a measured line
#: rather than part of an unexplained remainder.
.PLM_STARTUP <- proc.time()[["elapsed"]]

.PLM_TIMINGS <- list()
plm_time <- function(label, expr) {
  started <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - started
  .PLM_TIMINGS[[label]] <<- (if (is.null(.PLM_TIMINGS[[label]])) 0 else .PLM_TIMINGS[[label]]) + elapsed
  cat(sprintf("  [time] %-28s %8.1fs\n", label, elapsed))
  # invisible(), or a bare `plm_time("write ...", write.table(...))` at top
  # level auto-prints its value (a stray "NULL" in the log).
  # Assignment is unaffected: `x <- plm_time(...)` still binds the value.
  invisible(value)
}

# --- Binary sparse writer -----------------------------------------
#
# Matrix::writeMM formats every nonzero as ASCII, single-threaded; an RNA
# matrix can be close to 1 GB of text, and parsing it back is just as slow.
#
# So the text file existed only to carry data from R to Python, and both ends
# paid to encode and decode it. This writes the CSC arrays as raw little-endian
# binary instead, which numpy reads with np.frombuffer at memory-bandwidth
# speed. No formatting, no parsing, and the .npz conversion becomes unnecessary
# because the file is already in the form scipy wants.
#
# LAYOUT (paralome sparse v1). A 64-byte header, then three arrays:
#   magic     8 bytes    "PLMSPR01"
#   nrow      float64    features
#   ncol      float64    cells
#   nnz       float64    stored values
#   indptr_n  float64    always ncol + 1; written so the reader can verify
#   reserved  24 bytes   zero
#   indptr    float64 x (ncol + 1)
#   indices   int32   x nnz        row index of each stored value
#   data      float64 x nnz
#
# THE COUNTS ARE FLOAT64, NOT INTEGERS, on purpose. R has no 64-bit integer
# type -- writeBin(size = 8) on an integer is an error, and faking it as two
# int32s would be a lie the reader has to un-tell. float64 represents every
# integer below 2^53 exactly, which covers any nnz a Seurat object can reach,
# so the round trip is lossless without pretending.
#
# int32 for the row indices matches scipy's own default and holds 2.1e9 rows.
# A dgCMatrix carries exactly these three arrays (@p, @i, @x), so writing is a
# dump of what is already in memory, not a conversion.
plm_write_sparse <- function(m, path) {
  m <- as(m, "CsparseMatrix")
  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  writeBin(charToRaw("PLMSPR01"), con)
  writeBin(
    c(as.double(nrow(m)), as.double(ncol(m)),
      as.double(length(m@x)), as.double(length(m@p))),
    con, size = 8, endian = "little"
  )
  writeBin(raw(24), con)
  writeBin(as.double(m@p), con, size = 8, endian = "little")
  writeBin(as.integer(m@i), con, size = 4, endian = "little")
  writeBin(as.double(m@x), con, size = 8, endian = "little")
  invisible(path)
}

# --- Reading the object -------------------------------------------
#
# readRDS is single-threaded and cannot be otherwise: an RDS is a sequential
# byte stream with back-references, so deserialisation walks it in order, and
# R's gzip inflate is single-threaded C (one example: 164s on one core of a
# 20-core node).
#
# qs2 stores the same object graph with zstd and multithreads both halves. On
# the same object: 37s, about 4x faster.
#
# THREADS COME FROM SLURM, NOT FROM A FLAG. The extraction already runs inside
# a job with --cpus-per-task, so the allocation is the honest answer and needs
# no plumbing: a flag that does not exist cannot be parsed and then reach
# nothing.
plm_threads <- function() {
  slurm <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "")))
  if (!is.na(slurm) && slurm >= 1) return(slurm)
  detected <- tryCatch(parallel::detectCores(logical = FALSE),
                       error = function(e) 1L)
  if (is.na(detected) || detected < 1) return(1L)
  # Only the FALLBACK is capped. Outside SLURM this may be a shared login node,
  # where detectCores() reports the whole machine and taking it would be rude.
  min(detected, 8L)
}

# A structured failure the Python side turns into a numbered error, in the same
# shape as assay_error.json. Without it, R's stop() message reaches
# the user as PLM-0001 "Unexpected internal error", which says they hit a bug
# rather than something they can fix.
plm_fail <- function(code, message, fix = NA) {
  payload <- paste0(
    '{"code": ', code,
    ', "message": "', gsub('"', "'", gsub("\n", " ", message), fixed = TRUE), '"',
    ', "fix": ', if (is.na(fix)) "null" else paste0('"', gsub('"', "'", fix, fixed = TRUE), '"'),
    '}'
  )
  try(writeLines(payload, file.path(opts$out, "extraction_error.json")), silent = TRUE)
  stop(message, call. = FALSE)
}

plm_read_object <- function(path) {
  if (grepl("\\.qs2?$", path, ignore.case = TRUE)) {
    if (!requireNamespace("qs2", quietly = TRUE)) {
      # NAME THE R THAT LOOKED, not just the package that is missing. The
      # extraction may run under SLURM on another node or as another user, so
      # which R, and which library paths it searched, is the diagnosis.
      plm_fail(
        404,
        paste0(
          basename(path), " needs the qs2 package, and this R cannot see it.\n",
          "  R.home:    ", R.home(), "\n",
          "  Rscript:   ", file.path(R.home("bin"), "Rscript"), "\n",
          "  libPaths:  ", paste(.libPaths(), collapse = ", "), "\n",
          "If qs2 is installed elsewhere, this job is running under a different ",
          "R, a different user, or on a node that does not have it."
        ),
        fix = "install qs2 for THIS R, or point --rds at the .rds"
      )
    }
    nthreads <- plm_threads()
    cat(sprintf("  reading with qs2 on %d thread(s)\n", nthreads))
    return(qs2::qs_read(path, nthreads = nthreads))
  }
  readRDS(path)
}

object <- plm_time("read object", plm_read_object(opts$rds))
if (!inherits(object, "Seurat")) {
  # Numbered errors via plm_fail rather than R's stop().
  plm_fail(106, "--rds does not contain a Seurat object",
           fix = "point --rds at a Seurat object written with saveRDS() or qs2")
}

# --- Discover which assays are actually present ------------------------------
# ATAC does not always come with RNA, nor RNA with ATAC: the assays
# are read and whatever is there is extracted.
#
# No flag declares what the file contains: the object already knows, and a
# flag the caller has to set correctly is a flag that will eventually be set
# wrongly. Both halves are discovered here and each is exported if and only if
# it is found. --assay / --rna-assay remain, purely to break ties.
#
# PEAK-SHAPED ROWNAMES are the structural discriminator when names do not
# settle it. A ChromatinAssay is unambiguous, but plenty of objects hold ATAC
# counts in a plain Assay; those have rownames like "chr1-9776-10668", which no
# gene-level assay has. This is the same regex the peak-coordinate fallback
# below already relies on, so an assay that passes here is one whose
# coordinates can actually be parsed.
available_assays <- Seurat::Assays(object)
if (length(available_assays) == 0) {
  stop("The Seurat object has no assays")
}

PEAK_ID_RE <- "^([^:-]+)[:-](\\d+)-(\\d+)$"

# DERIVED ASSAYS are detected structurally, not by name.
#
# Kept a list -- c("integrated", "SCT") -- which is the same mistake as
# requiring an assay be named "RNA": it works until someone names theirs
# something else. Two structural tells cover both cases and generalise:
#
#   * an SCTAssay is its own S4 class, carrying SCTModel.list. Note that its
#     `counts` layer holds CORRECTED counts, which ARE integers -- so "counts
#     are not whole numbers" does NOT catch SCT, and the class check is the
#     only reliable test.
#   * an `integrated` assay typically has an EMPTY counts layer: the correction
#     lives in `data`. An assay with no counts cannot be aggregated anyway.
is_derived <- function(name) {
  assay <- object[[name]]
  if (inherits(assay, "SCTAssay")) return(TRUE)
  counts <- tryCatch(
    SeuratObject::LayerData(assay, layer = "counts"),
    error = function(e) tryCatch(assay@counts, error = function(e2) NULL)
  )
  is.null(counts) || nrow(counts) == 0 || ncol(counts) == 0
}

is_chromatin <- function(name) inherits(object[[name]], "ChromatinAssay")

looks_like_peaks <- function(name) {
  ids <- tryCatch(rownames(object[[name]]), error = function(e) character(0))
  if (length(ids) == 0) return(FALSE)
  probe <- head(ids, 100)
  sum(grepl(PEAK_ID_RE, probe)) >= 0.9 * length(probe)
}

# Written when a resolution fails, so the Python side can raise a NUMBERED
# error naming the assays instead of surfacing R's stderr as PLM-0001. R stops
# after writing it; nothing downstream reads a partial extraction.
stop_with_assays <- function(code, message, requested = NA) {
  payload <- paste0(
    '{"code": ', code,
    ', "message": "', gsub('"', "'", message, fixed = TRUE), '"',
    ', "requested": ', if (is.na(requested)) "null" else paste0('"', requested, '"'),
    ', "available_assays": [',
    paste0('"', available_assays, '"', collapse = ", "), ']}'
  )
  writeLines(payload, file.path(opts$out, "assay_error.json"))
  stop(message, call. = FALSE)
}

detect_atac_assay <- function(requested, available) {
  if (!is.null(requested)) {
    if (!(requested %in% available)) {
      stop_with_assays(216, paste0(
        "--assay '", requested, "' is not in this object"
      ), requested)
    }
    return(requested)
  }
  # A Signac ChromatinAssay says so outright.
  chromatin <- Filter(is_chromatin, available)
  if (length(chromatin) == 1) return(chromatin[[1]])
  if (length(chromatin) > 1) {
    stop_with_assays(217, paste0(
      "More than one ATAC assay (", paste(chromatin, collapse = ", "),
      "); name one with --assay"
    ))
  }
  matched <- intersect(c("ATAC", "peaks", "macs2"), available)
  if (length(matched) == 1) return(matched[[1]])
  if (length(matched) > 1) {
    stop_with_assays(217, paste0(
      "More than one ATAC-named assay (", paste(matched, collapse = ", "),
      "); name one with --assay"
    ))
  }
  # Last resort: an assay whose rownames are genomic intervals.
  shaped <- Filter(looks_like_peaks, Filter(function(n) !is_derived(n), available))
  if (length(shaped) == 1) return(shaped[[1]])
  # Not found is not an error here. It means this object has no ATAC half,
  # which is an ordinary thing for an object to be.
  NULL
}

detect_gene_level_assay <- function(requested, available, atac_name) {
  # RNA, Spatial, Xenium, protein and Visium are all gene-level
  # analyses, under whatever name the object's author chose; only ATAC differs.
  #
  # So there is nothing to classify. The question is binary -- are the features
  # genomic intervals, or are they labels? -- and the interval test above already
  # answers it. Everything that is not ATAC is gene-level, whatever it is called.
  #
  # No assay name is preferred, and --rna-assay has no default: a default would
  # reach R as an explicit request and short-circuit detection, so a Visium
  # object whose only assay is "Spatial" could not be extracted.
  if (!is.null(requested)) {
    if (!(requested %in% available)) {
      stop_with_assays(216, paste0(
        "--rna-assay '", requested, "' is not in this object"
      ), requested)
    }
    return(requested)
  }
  candidates <- setdiff(available, atac_name)
  candidates <- Filter(function(name) !is_chromatin(name), candidates)
  candidates <- Filter(function(name) !looks_like_peaks(name), candidates)
  candidates <- Filter(function(name) !is_derived(name), candidates)

  if (length(candidates) == 1) return(candidates[[1]])
  if (length(candidates) == 0) return(NULL)

  # Several measured, gene-level assays -- an RNA+ADT object, say. The user
  # has to choose, because picking one silently is how the wrong matrix gets
  # analysed and nothing ever says so.
  #
  # A name (RNA, Spatial, Xenium, originalexp) never breaks the tie: an object
  # with assays RNA, integrated and RNA_shared -- the last holding only the
  # genes two references counted under the same name -- would otherwise be
  # extracted from RNA with no message. Several candidates is always 217.
  stop_with_assays(217, paste0(
    "More than one gene-level assay (", paste(candidates, collapse = ", "),
    "); name one with --rna-assay"
  ))
}

assay_name <- detect_atac_assay(opts$assay, available_assays)
rna_assay_name <- detect_gene_level_assay(opts$rna_assay, available_assays, assay_name)

if (is.null(assay_name) && is.null(rna_assay_name)) {
  stop_with_assays(214, "Nothing in this object could be extracted")
}
cat(sprintf(
  "  Assays present: %s -> extracting ATAC=%s, RNA=%s\n",
  paste(available_assays, collapse = ", "),
  if (is.null(assay_name)) "none" else assay_name,
  if (is.null(rna_assay_name)) "none" else rna_assay_name
))

counts <- NULL
if (!is.null(assay_name)) {

assay_obj <- object[[assay_name]]

counts <- plm_time("GetAssayData (ATAC)",
                   Seurat::GetAssayData(object, assay = assay_name, layer = opts$slot))
if (is.null(counts) || ncol(counts) == 0 || nrow(counts) == 0) {
  plm_fail(213, paste0("Assay '", assay_name, "' slot '", opts$slot, "' is empty"),
           fix = "name a layer that holds counts with --slot")
}

# --- Resolve peak coordinates ------------------------------------------------
peak_ids <- rownames(counts)
peak_regions <- NULL
if (requireNamespace("Signac", quietly = TRUE) && inherits(assay_obj, "ChromatinAssay")) {
  gr <- tryCatch(Signac::granges(assay_obj), error = function(e) NULL)
  if (!is.null(gr)) {
    peak_regions <- data.frame(
      chrom = as.character(GenomicRanges::seqnames(gr)),
      start = GenomicRanges::start(gr) - 1L,  # GRanges is 1-based; BED is 0-based
      end = GenomicRanges::end(gr),
      peak_id = peak_ids,
      stringsAsFactors = FALSE
    )
  }
}
if (is.null(peak_regions)) {
  # Fall back to parsing "chrom-start-end" or "chrom:start-end" style rownames.
  parsed <- regmatches(
    peak_ids,
    regexec("^([^:-]+)[:-](\\d+)-(\\d+)$", peak_ids)
  )
  if (any(lengths(parsed) != 4)) {
    plm_fail(213, paste0(
      "Could not resolve peak coordinates: assay '", assay_name, "' is not a Signac ",
      "ChromatinAssay and its peak names do not match 'chrom-start-end' / 'chrom:start-end'."
    ), fix = "name the ATAC assay with --assay, or convert it to a ChromatinAssay")
  }
  # The rowname start is written as given. Signac-style 'chrom-start-end'
  # names usually carry a 1-based start, so these can differ by 1 bp from the
  # ChromatinAssay path above, which converts to 0-based BED.
  peak_regions <- data.frame(
    chrom = vapply(parsed, `[`, character(1), 2),
    start = as.integer(vapply(parsed, `[`, character(1), 3)),
    end = as.integer(vapply(parsed, `[`, character(1), 4)),
    peak_id = peak_ids,
    stringsAsFactors = FALSE
  )
}

# --- Write peaks.bed / peaks_order.tsv --------------------------------------
write.table(
  peak_regions[, c("chrom", "start", "end", "peak_id")],
  file.path(opts$out, "peaks.bed"),
  sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
)
writeLines(peak_ids, file.path(opts$out, "peaks_order.tsv"))

# --- Peak feature statistics ------------------------------------
# The ATAC counterpart of the RNA HVF export below. peaks.bed is a strict
# 4-column BED that read_atlas() parses positionally, so these go in a
# SEPARATE sidecar file rather than as extra BED columns.
#
# Signac's FindTopFeatures populates meta.features with `count` and
# `percentile` and sets VariableFeatures accordingly -- note this ranks peaks
# by total accessibility, NOT by a variance-stabilized statistic like RNA's
# vst. The two are deliberately kept distinguishable downstream rather than
# both being called "variable features" as if they meant the same thing.
peaks_meta <- tryCatch(assay_obj@meta.features, error = function(e) NULL)
peaks_frame <- data.frame(peak_id = peak_ids, stringsAsFactors = FALSE)
if (!is.null(peaks_meta) && nrow(peaks_meta) > 0) {
  peaks_matched <- peaks_meta[match(peak_ids, rownames(peaks_meta)), , drop = FALSE]
  for (column in colnames(peaks_matched)) {
    values <- peaks_matched[[column]]
    if (is.numeric(values) || is.logical(values)) {
      peaks_frame[[gsub(".", "_", column, fixed = TRUE)]] <- values
    }
  }
}
atac_variable_features <- tryCatch(
  Seurat::VariableFeatures(object, assay = assay_name),
  error = function(e) character(0)
)
peaks_frame$variable_feature <- peak_ids %in% atac_variable_features
write.table(
  peaks_frame, file.path(opts$out, "peaks_features.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = ""
)
cat(sprintf(
  "  Peak feature stats: %d columns; Signac VariableFeatures: %d of %d peaks flagged\n",
  ncol(peaks_frame) - 1L, length(atac_variable_features), length(peak_ids)
))

# --- Write counts.plmsp / cells.tsv -----------------------------------------
plm_time("write counts (binary)",
         plm_write_sparse(counts, file.path(opts$out, "counts.plmsp")))
cell_barcodes <- colnames(counts)
writeLines(cell_barcodes, file.path(opts$out, "cells.tsv"))

} else {
  # No ATAC half. cells come from the RNA assay instead; cells.tsv and
  # cell_metadata.tsv are still written either way, because both sides read
  # them -- they are shared, not ATAC-specific.
  cat("  No ATAC assay: skipping peaks.bed, peaks_order.tsv, peaks_features.tsv, counts.plmsp\n")
  # Colnames WITHOUT materialising the counts.
  #
  # On a Seurat v5 object with split layers -- which is what `merge()`
  # produces -- GetAssayData joins the layers and builds the whole matrix, so
  # calling it only to read colnames() assembled the object twice.
  # colnames() on the assay reads the cell names directly.
  cell_barcodes <- colnames(object[[rna_assay_name]])
  if (is.null(cell_barcodes) || length(cell_barcodes) == 0) {
    plm_fail(213, paste0("RNA assay '", rna_assay_name, "' is empty; nothing to extract."),
             fix = "name a gene-level assay that holds cells with --rna-assay")
  }
  writeLines(cell_barcodes, file.path(opts$out, "cells.tsv"))
}

# --- Write cell_metadata.tsv (every meta.data column, unfiltered) ----------
metadata <- object@meta.data
metadata_out <- plm_time("build metadata frame", {
  frame <- cbind(barcode = rownames(metadata), metadata)
  frame[cell_barcodes, , drop = FALSE]
})
# Timed: hundreds of thousands of rows through write.table, which formats a
# row at a time. data.table::fwrite is 20-50x faster -- but it
# formats doubles differently, and Python groups cells by the STRINGS in this
# file, so that swap needs the two outputs diffed before it can be trusted.
plm_time("write cell_metadata.tsv", write.table(
  metadata_out,
  file.path(opts$out, "cell_metadata.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = ""
))

# --- Reductions and tissue coordinates --------------------------
# For the per-object cell export (singlecell/cell_export.py): every reduction
# the object carries, and each image's tissue coordinates, keyed by barcode in
# cells.tsv order. Nothing here is required by the analysis; an object without
# reductions or images writes none and the extraction is unchanged.
reduction_names <- character(0)
reduction_list <- tryCatch(names(object@reductions), error = function(e) character(0))
for (reduction in reduction_list) {
  embedding <- tryCatch(Seurat::Embeddings(object, reduction = reduction), error = function(e) NULL)
  if (is.null(embedding) || nrow(embedding) == 0 || ncol(embedding) == 0) next
  embedding <- embedding[match(cell_barcodes, rownames(embedding)), , drop = FALSE]
  # A cell the reduction does not cover is an NA row with an NA row name, and
  # data.frame() refuses NA row names: the barcode column is the key.
  rownames(embedding) <- NULL
  dir.create(file.path(opts$out, "reductions"), showWarnings = FALSE)
  plm_time("write reductions", write.table(
    data.frame(barcode = cell_barcodes, embedding, check.names = FALSE),
    file.path(opts$out, "reductions", paste0(reduction, ".tsv")),
    sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = ""
  ))
  reduction_names <- c(reduction_names, reduction)
}
# The dims each Seurat command that takes them was run with (RunUMAP, FindNeighbors ...), from the
# object's command log, so the trajectories fit in the dimensions the object's UMAP was built from. An object
# without a command log writes none.
command_rows <- list()
for (command_name in tryCatch(names(object@commands), error = function(e) character(0))) {
  command <- tryCatch(object@commands[[command_name]], error = function(e) NULL)
  params <- tryCatch(command@params, error = function(e) NULL)
  if (is.null(params) || is.null(params$dims)) next
  used <- suppressWarnings(as.integer(params$dims))
  if (!length(used) || anyNA(used)) next
  command_rows[[length(command_rows) + 1]] <- data.frame(
    command = command_name,
    # The command's own name is <function>.<assay>.<reduction> (RunUMAP.RNA.pca); the function is its first part.
    function_name = sub("\\..*$", "", tryCatch(as.character(command@name), error = function(e) command_name)),
    reduction = if (!is.null(params$reduction)) as.character(params$reduction)[1] else "",
    dims = paste(used, collapse = ","),
    stringsAsFactors = FALSE
  )
}
if (length(command_rows)) {
  write.table(do.call(rbind, command_rows), file.path(opts$out, "seurat_commands.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
}
image_names <- character(0)
image_list <- tryCatch(Seurat::Images(object), error = function(e) character(0))
coordinate_frames <- list()
for (image in image_list) {
  # Scale = NULL, the full-resolution pixels the scale factors
  # below are relative to. Seurat 4's Visium default is the lowres scale;
  # Seurat 5's is already NULL. A method without the argument gets the call
  # without it.
  coordinates <- tryCatch(Seurat::GetTissueCoordinates(object, image = image, scale = NULL),
                          error = function(e) tryCatch(Seurat::GetTissueCoordinates(object, image = image),
                                                       error = function(e) NULL))
  if (is.null(coordinates) || nrow(coordinates) == 0) next
  barcodes <- if ("cell" %in% colnames(coordinates)) as.character(coordinates$cell) else rownames(coordinates)
  # Seurat 5 (VisiumV2, SlideSeq, FOV) names them x/y; Seurat 4 Visium
  # imagerow/imagecol, where the column is x and the row is y.
  if (all(c("x", "y") %in% colnames(coordinates))) {
    x <- coordinates$x; y <- coordinates$y
  } else if (all(c("imagerow", "imagecol") %in% colnames(coordinates))) {
    x <- coordinates$imagecol; y <- coordinates$imagerow
  } else next
  coordinate_frames[[image]] <- data.frame(barcode = barcodes, image = image,
                                           x = as.numeric(x), y = as.numeric(y))
  image_names <- c(image_names, image)
}
# The tissue image itself: whatever the object holds -- the
# lowres H&E Load10X_Spatial reads, or the hires one when it was loaded --
# written as raw 8-bit RGB, row by row (height x width x 3), with its size and
# the slide's scale factors beside it. No PNG package is needed to write it.
image_files <- character(0)
for (image in image_names) {
  slot_image <- tryCatch(object@images[[image]]@image, error = function(e) NULL)
  if (is.null(slot_image) || length(dim(slot_image)) != 3 || dim(slot_image)[3] < 3) next
  pixels <- slot_image[, , 1:3, drop = FALSE]
  if (max(pixels, na.rm = TRUE) <= 1) pixels <- pixels * 255
  pixels[is.na(pixels)] <- 0
  dir.create(file.path(opts$out, "images"), showWarnings = FALSE)
  connection <- file(file.path(opts$out, "images", paste0(image, ".rgb")), "wb")
  writeBin(as.integer(round(as.vector(aperm(pixels, c(3, 2, 1))))), connection, size = 1)
  close(connection)
  factors <- tryCatch(Seurat::ScaleFactors(object@images[[image]]), error = function(e) NULL)
  factor_values <- if (is.null(factors)) list() else unclass(factors)
  writeLines(c(paste0("height\t", dim(pixels)[1]), paste0("width\t", dim(pixels)[2]),
               vapply(names(factor_values), function(name)
                 paste0(name, "\t", format(as.numeric(factor_values[[name]]), digits = 15)), character(1))),
             file.path(opts$out, "images", paste0(image, ".tsv")))
  image_files <- c(image_files, image)
}
if (length(coordinate_frames) > 0) {
  plm_time("write spatial_coordinates", write.table(
    do.call(rbind, coordinate_frames), file.path(opts$out, "spatial_coordinates.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = ""
  ))
}
cat(sprintf("  Reductions: %s; images with coordinates: %s\n",
            if (length(reduction_names)) paste(reduction_names, collapse = ", ") else "none",
            if (length(image_names)) paste(image_names, collapse = ", ") else "none"))

# --- RNA export (optional/additive) -------------------------------
# RNA and ATAC are exported in the same run, from the SAME already-loaded
# `object` (no second readRDS). Genes have no genomic coordinates to resolve
# the way peaks do -- the primary ID is rownames(rna_counts) (often Ensembl),
# with an optional common ID; no GTF/chrom/start/end here, unlike the ATAC
# peak annotation above.
#
# The trigger is discovery, not --rna-assay. If the object has an RNA
# half it is exported, whether or not anyone asked -- one readRDS serves both
# pipelines, and a downstream stage that does not want genes simply does not
# read them.
n_genes <- 0
if (!is.null(rna_assay_name)) {
  rna_counts <- plm_time("GetAssayData (RNA)",
                         Seurat::GetAssayData(object, assay = rna_assay_name, layer = opts$rna_slot))
  if (is.null(rna_counts) || ncol(rna_counts) == 0 || nrow(rna_counts) == 0) {
    plm_fail(213, paste0("RNA assay '", rna_assay_name, "' slot '", opts$rna_slot, "' is empty"),
             fix = "name a layer that holds counts with --rna-slot")
  }
  # cells.tsv is written once and indexes BOTH count matrices. If the two halves
  # of a multiome object disagree about their cells, every barcode-keyed join
  # downstream is silently wrong, so refuse rather than pick one.
  if (!identical(colnames(rna_counts), cell_barcodes)) {
    plm_fail(215, paste0(
      "Assay '", rna_assay_name, "' has ", ncol(rna_counts), " cells but cells.tsv has ",
      length(cell_barcodes), " from '", if (is.null(assay_name)) "RNA" else assay_name,
      "'; they must be the same barcodes in the same order to share one cells.tsv."
    ))
  }
  gene_ids <- rownames(rna_counts)
  common_ids <- rep(NA_character_, length(gene_ids))
  if (!is.null(opts$gene_symbol_column)) {
    rna_assay_obj <- object[[rna_assay_name]]
    meta_features <- tryCatch(rna_assay_obj@meta.features, error = function(e) NULL)
    if (!is.null(meta_features) && opts$gene_symbol_column %in% colnames(meta_features)) {
      common_ids <- as.character(meta_features[gene_ids, opts$gene_symbol_column])
    }
  }
  # `feature_name` is the assay ROWNAME, and it is the row's identity: it is
  # what indexes the counts matrix, what rna_genes_order.tsv records, and it is
  # unique by construction. `gene_id` and `common_id` are ANNOTATION about that
  # row -- Paralome resolves them against the GTF after this script exits, and
  # either may end up "." when the GTF cannot say. Keeping the three separate is
  # the whole point: an unresolvable annotation must never be able to become a
  # key. Writing the same values into feature_name and gene_id here is not
  # redundancy, it is the pre-annotation state of a column that is about to mean
  # something different.
  genes_frame <- data.frame(
    feature_name = gene_ids,
    gene_id = gene_ids,
    common_id = common_ids,
    stringsAsFactors = FALSE
  )

  # --- Seurat's own variance-stabilized feature statistics --------
  # Seurat's VariableFeatures and HVFInfo are already in the object. The
  # variance.standardized RANKING is exported here and the top-N CUTOFF is
  # applied later at PCA time -- changing N never requires re-extraction.
  #
  # HVFInfo covers EVERY gene, not just the selected ones, so genes with
  # variance == 0 can be left out of the PCA.
  #
  # All of this is best-effort: an assay that never had FindVariableFeatures run
  # on it simply yields no columns, and the extraction proceeds unchanged.
  hvf <- tryCatch(
    Seurat::HVFInfo(object, assay = rna_assay_name),
    error = function(e) NULL
  )
  if (!is.null(hvf) && nrow(hvf) > 0) {
    hvf_matched <- hvf[match(gene_ids, rownames(hvf)), , drop = FALSE]
    # Seurat's vst method names these mean/variance/variance.expected/
    # variance.standardized; the older mean.var.plot method reports dispersion
    # instead. Carry whichever exist, normalizing "." to "_" for the TSV.
    for (column in colnames(hvf_matched)) {
      genes_frame[[gsub(".", "_", column, fixed = TRUE)]] <- as.numeric(hvf_matched[[column]])
    }
    cat(sprintf("  HVF metrics: %s\n", paste(colnames(hvf_matched), collapse = ", ")))
  } else {
    cat("  HVF metrics: none (FindVariableFeatures was never run on this assay)\n")
  }

  variable_features <- tryCatch(
    Seurat::VariableFeatures(object, assay = rna_assay_name),
    error = function(e) character(0)
  )
  genes_frame$variable_feature <- gene_ids %in% variable_features
  cat(sprintf(
    "  Seurat VariableFeatures: %d of %d genes flagged\n",
    length(variable_features), length(gene_ids)
  ))

  write.table(
    genes_frame, file.path(opts$out, "rna_genes.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, na = ""
  )
  writeLines(gene_ids, file.path(opts$out, "rna_genes_order.tsv"))
  plm_time("write rna_counts (binary)",
           plm_write_sparse(rna_counts, file.path(opts$out, "rna_counts.plmsp")))
  n_genes <- nrow(rna_counts)
  cat(sprintf(
    "Extracted RNA assay '%s': %d genes x %d cells -> %s\n",
    rna_assay_name, nrow(rna_counts), ncol(rna_counts), opts$out
  ))
}

# --- Provenance manifest -----------------------------------------------------
# `has_atac` / `has_rna` are what the Python side reads to learn which outputs
# this run actually produced. It can no longer predict them from its own
# arguments -- that is the point of discovery -- so the manifest is the record.
manifest <- list(
  schema = "paralome.atac.singlecell_extraction_manifest.v1",
  has_atac = !is.null(assay_name),
  has_rna = !is.null(rna_assay_name),
  assay = if (is.null(assay_name)) NA else assay_name,
  slot = opts$slot,
  n_cells = length(cell_barcodes),
  n_peaks = if (is.null(counts)) 0L else nrow(counts),
  rna_assay = if (is.null(rna_assay_name)) NA else rna_assay_name,
  rna_slot = if (is.null(rna_assay_name)) NA else opts$rna_slot,
  n_genes = n_genes,
  gene_symbol_column = if (is.null(opts$gene_symbol_column)) NA else opts$gene_symbol_column,
  seurat_version = as.character(utils::packageVersion("Seurat")),
  signac_version = tryCatch(as.character(utils::packageVersion("Signac")), error = function(e) NA),
  metadata_columns = colnames(metadata),
  reductions = I(reduction_names),
  images = I(image_names),
  tissue_images = I(image_files)
)
jsonlite_available <- requireNamespace("jsonlite", quietly = TRUE)
if (jsonlite_available) {
  writeLines(jsonlite::toJSON(manifest, auto_unbox = TRUE, pretty = TRUE),
             file.path(opts$out, "extraction_manifest.json"))
} else {
  # Minimal hand-rolled JSON fallback so this script has no hard dependency
  # on jsonlite; Python re-derives everything authoritative anyway.
  scalar_fields <- manifest[setdiff(names(manifest), c("metadata_columns", "reductions", "images", "tissue_images"))]
  # NA as null and logicals as true/false, not as strings.
  json_scalar <- function(value) {
    if (length(value) == 0 || is.na(value)) return("null")
    if (is.logical(value)) return(if (value) "true" else "false")
    if (is.numeric(value)) return(as.character(value))
    paste0('"', gsub('"', "'", as.character(value), fixed = TRUE), '"')
  }
  kv <- paste0(
    '"', names(scalar_fields), '": ',
    vapply(scalar_fields, json_scalar, character(1)),
    collapse = ", "
  )
  cols <- paste0('"', manifest$metadata_columns, '"', collapse = ", ")
  json_list <- function(values) paste0("[", paste0('"', values, '"', collapse = ", "), "]")
  if (length(manifest$reductions) == 0) reductions_json <- "[]" else reductions_json <- json_list(manifest$reductions)
  if (length(manifest$images) == 0) images_json <- "[]" else images_json <- json_list(manifest$images)
  if (length(manifest$tissue_images) == 0) tissue_json <- "[]" else tissue_json <- json_list(manifest$tissue_images)
  writeLines(paste0("{", kv, ", \"metadata_columns\": [", cols, "], \"reductions\": ",
                    reductions_json, ", \"images\": ", images_json,
                    ", \"tissue_images\": ", tissue_json, "}"),
             file.path(opts$out, "extraction_manifest.json"))
}

if (!is.null(assay_name)) {
  cat(sprintf(
    "Extracted assay '%s': %d peaks x %d cells -> %s\n",
    assay_name, nrow(counts), ncol(counts), opts$out
  ))
}

# --- Where the time went ------------------------------------------
#
# Printed unconditionally, so a slow extraction reports its own profile into the
# stage log and nobody has to reproduce it to find out which step was expensive.
if (length(.PLM_TIMINGS) > 0) {
  wall <- proc.time()[["elapsed"]]
  measured <- sum(unlist(.PLM_TIMINGS))
  cat("\n[paralome] extraction timing\n")
  cat(sprintf("  %-28s %8.1fs  %5.1f%%\n", "R startup + packages",
              .PLM_STARTUP, if (wall > 0) 100 * .PLM_STARTUP / wall else 0))
  for (label in names(.PLM_TIMINGS)) {
    seconds <- .PLM_TIMINGS[[label]]
    cat(sprintf("  %-28s %8.1fs  %5.1f%%\n", label, seconds,
                if (wall > 0) 100 * seconds / wall else 0))
  }
  # Percentages are OF WALL TIME, not of the measured subset. Of
  # the measured subset they always sum to 100 and so cannot show how much is
  # still unaccounted for -- which is exactly the number worth watching.
  cat(sprintf("  %-28s %8.1fs\n", "-- wall clock", wall))
  cat(sprintf("  %-28s %8.1fs  %5.1f%%\n", "still unmeasured",
              max(0, wall - measured - .PLM_STARTUP),
              if (wall > 0) 100 * max(0, wall - measured - .PLM_STARTUP) / wall else 0))
}
