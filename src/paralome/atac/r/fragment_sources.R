#!/usr/bin/env Rscript
#
# fragment_sources.R
#
# Internal helper shipped with Paralome (see paralome.atac.genome_tracks). The snATAC genome_tracks
# stage runs it once per Seurat/Signac object to learn which fragment files the object's ATAC assay reads, and
# which cell is which barcode in each. The fragments themselves are never in the object: a Signac Fragment
# holds the file's path and a named vector of cells (names: the cells as the object calls them; values: their
# barcodes in the file), and Signac reads the file from that path when it needs it.
#
# Usage:
#   Rscript fragment_sources.R --rds /path/to/object.rds --out /path/to/fragment_sources.tsv [--assay ATAC]
#
# Writes a tab-separated table with a header: path, cell, barcode -- one row per cell per fragment file. A
# Fragment with no cell vector stands for every cell of the object under its own name. With no --assay, every
# ChromatinAssay of the object is read; a file named by more than one assay is listed once per cell.

suppressWarnings(suppressMessages({
  library(methods)
}))

parse_args <- function(argv) {
  opts <- list(rds = NULL, out = NULL, assay = NULL)
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--rds") { opts$rds <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--out") { opts$out <- argv[[i + 1]]; i <- i + 2 }
    else if (key == "--assay") { opts$assay <- argv[[i + 1]]; i <- i + 2 }
    else stop(paste0("Unrecognized argument: ", key))
  }
  if (is.null(opts$rds) || is.null(opts$out)) stop("--rds and --out are required")
  opts
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
if (!requireNamespace("Signac", quietly = TRUE)) {
  stop("The R package Signac is not installed; it is needed to read the object's fragment files.")
}
suppressWarnings(suppressMessages({
  requireNamespace("SeuratObject", quietly = TRUE)
}))

object <- readRDS(opts$rds)
assay_names <- names(object@assays)
if (!is.null(opts$assay) && opts$assay %in% assay_names) {
  chosen <- opts$assay
} else {
  chosen <- assay_names[vapply(assay_names, function(name) inherits(object[[name]], "ChromatinAssay"), logical(1))]
}

object_cells <- colnames(object)
rows <- list()
for (name in chosen) {
  assay <- object[[name]]
  if (!inherits(assay, "ChromatinAssay")) next
  for (fragment in Signac::Fragments(assay)) {
    path <- methods::slot(fragment, "path")
    cells <- methods::slot(fragment, "cells")
    if (is.null(cells) || length(cells) == 0) {
      cells <- stats::setNames(object_cells, object_cells)
    }
    rows[[length(rows) + 1]] <- data.frame(
      path = rep(path, length(cells)),
      cell = names(cells),
      barcode = unname(as.character(cells)),
      stringsAsFactors = FALSE
    )
  }
}

table <- if (length(rows)) unique(do.call(rbind, rows)) else
  data.frame(path = character(0), cell = character(0), barcode = character(0))
utils::write.table(table, file = opts$out, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
