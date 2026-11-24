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

# The lineages of one object, fitted by Slingshot on the whole object (every condition
# together) from the reduction and cluster column paralome.singlecell.trajectories chose, rooted at the cluster the
# run named (--trajectory-root <column>:<value>). Reads the extraction's TSVs, never the .rds.
#
#   Rscript slingshot_trajectories.R --reduction <tsv> --labels <tsv> --root <cluster> --out <dir>
#           [--drop-first] [--columns <i,j,...>] [--fit-cells <n> --fit-by random|centroid]
#           [--stretch <s>] [--extend y|n|pc1] [--df <n>] [--shrink <0-1>] [--embedding <name>=<tsv>]...
#
# --reduction: barcode + one column per dimension (extraction/reductions/<name>.tsv)
# --labels:    barcode + cluster (the chosen cell-metadata column)
# --drop-first: leave out the first dimension (LSI's, which follows sequencing depth)
# --columns:   fit on these dimensions of the reduction (1-based, as the object numbers them; --drop-first is then
#              not applied); default every one
# --fit-cells: the lineages from every cell; their curves fitted on <n> cells of each cluster (every cell of a smaller
#              one), chosen --fit-by random (a seeded draw) or centroid (the <n> nearest the cluster's centre); every
#              cell is then placed on the fitted curves with slingshot's predict(). Default: every cell is fitted
#
# --stretch, --extend: getCurves' own (how far a curve runs past its last cluster, Slingshot's default 2; how its
#              ends are drawn, default y); not given, Slingshot's defaults
# --df:        the smoothing spline's degrees of freedom for every curve, in the fit and in each embedding; not given,
#              smooth.spline chooses them by cross-validation
# --shrink:    getCurves' shrink, 0-1 (how far the lineages' shared start is pulled together; default 1)
# --embedding: a 2D or 3D embedding the curves are drawn in (embedCurves, on the fitted cells), as many as given
#
# Writes in --out: cells.tsv (barcode, pseudotime_<L>, weight_<L> for every lineage, branch), lineages.tsv
# (lineage, clusters along it), mst.tsv (the cluster tree's edges), curve_<embedding>.tsv (lineage, point, the
# curve's coordinates in that embedding) and slingshot.json (versions and what was fitted).

suppressPackageStartupMessages({
  library(methods)
})

args <- commandArgs(trailingOnly = TRUE)
opts <- list(embedding = character(0), drop_first = FALSE, columns = integer(0), fit_cells = 0L, fit_by = "",
             stretch = NA_real_, extend = "", df = NA_real_, shrink = NA_real_)
i <- 1
while (i <= length(args)) {
  key <- args[[i]]
  if (key == "--drop-first") { opts$drop_first <- TRUE; i <- i + 1; next }
  if (i == length(args)) stop(sprintf("%s needs a value", key))
  value <- args[[i + 1]]
  if (key == "--reduction") opts$reduction <- value
  else if (key == "--labels") opts$labels <- value
  else if (key == "--root") opts$root <- value
  else if (key == "--out") opts$out <- value
  else if (key == "--embedding") opts$embedding <- c(opts$embedding, value)
  else if (key == "--columns") opts$columns <- as.integer(strsplit(value, ",", fixed = TRUE)[[1]])
  else if (key == "--fit-cells") opts$fit_cells <- as.integer(value)
  else if (key == "--fit-by") opts$fit_by <- value
  else if (key == "--stretch") opts$stretch <- as.numeric(value)
  else if (key == "--extend") opts$extend <- value
  else if (key == "--df") opts$df <- as.numeric(value)
  else if (key == "--shrink") opts$shrink <- as.numeric(value)
  else stop(sprintf("unknown option %s", key))
  i <- i + 2
}
for (needed in c("reduction", "labels", "root", "out")) {
  if (is.null(opts[[needed]])) stop(sprintf("--%s is required", needed))
}
if (!requireNamespace("slingshot", quietly = TRUE)) {
  cat("PARALOME_MISSING_R_PACKAGE slingshot\n")
  quit(status = 3)
}
dir.create(opts$out, showWarnings = FALSE, recursive = TRUE)

read_table <- function(path) {
  read.delim(path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE,
             colClasses = c(barcode = "character"), na.strings = c("", "NA"), quote = "")
}

reduction <- read_table(opts$reduction)
labels <- read.delim(opts$labels, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE,
                     colClasses = "character", na.strings = character(0), quote = "")
dims <- as.matrix(reduction[, setdiff(colnames(reduction), "barcode"), drop = FALSE])
rownames(dims) <- reduction$barcode
if (length(opts$columns)) {
  if (anyNA(opts$columns) || max(opts$columns) > ncol(dims) || min(opts$columns) < 1L)
    stop(sprintf("--columns: the reduction has %d dimensions", ncol(dims)))
  dims <- dims[, opts$columns, drop = FALSE]
} else if (opts$drop_first) {
  dims <- dims[, -1, drop = FALSE]
}
cluster <- labels$cluster[match(rownames(dims), labels$barcode)]
keep <- stats::complete.cases(dims) & !is.na(cluster) & cluster != ""
dims <- dims[keep, , drop = FALSE]
cluster <- cluster[keep]
if (!(opts$root %in% cluster)) {
  cat(sprintf("PARALOME_ROOT_NOT_FOUND %s\n", opts$root))
  quit(status = 4)
}
cat(sprintf("  Slingshot: %d cells, %d dimensions, %d clusters, root %s\n",
            nrow(dims), ncol(dims), length(unique(cluster)), opts$root))
flush.console()

set.seed(1)
fitted <- seq_len(nrow(dims))
if (opts$fit_cells > 0L) {
  # Each cluster's cells, or <n> of them: a seeded draw, or the <n> nearest the cluster's centre.
  fitted <- sort(unlist(lapply(split(seq_len(nrow(dims)), cluster), function(index) {
    if (length(index) <= opts$fit_cells) return(index)
    if (opts$fit_by == "centroid") {
      centre <- colMeans(dims[index, , drop = FALSE])
      distance <- rowSums(sweep(dims[index, , drop = FALSE], 2, centre)^2)
      return(index[order(distance)[seq_len(opts$fit_cells)]])
    }
    sample(index, opts$fit_cells)
  }), use.names = FALSE))
  cat(sprintf("  Slingshot: curves fitted on %d cells (%d per cluster, %s); every cell is then placed on them\n",
              length(fitted), opts$fit_cells, opts$fit_by))
  flush.console()
}
# "stretch=0, extend=n" -- getCurves' own settings, when given.
curve_settings <- list()
if (!is.na(opts$stretch)) curve_settings$stretch <- opts$stretch
if (nzchar(opts$extend)) curve_settings$extend <- opts$extend
# The spline's degrees of freedom (passed on to
# smooth.spline) and the shrinking of the shared start. embedCurves does not take df from the fit, so both are
# given to it again below.
if (!is.na(opts$df)) curve_settings$df <- opts$df
if (!is.na(opts$shrink)) curve_settings$shrink <- opts$shrink
embed_settings <- curve_settings[intersect(names(curve_settings), c("df", "shrink"))]
if (length(curve_settings)) {
  cat(sprintf("  Slingshot: curves with %s\n",
              paste(sprintf("%s = %s", names(curve_settings), unlist(curve_settings)), collapse = ", ")))
  flush.console()
}
started <- proc.time()[["elapsed"]]
if (length(fitted) == nrow(dims)) {
  fit <- do.call(slingshot::slingshot, c(list(dims, clusterLabels = cluster, start.clus = opts$root), curve_settings))
} else {
  # The lineages (the tree of the clusters' centres) from every cell; the curves from the chosen ones.
  lineages_of_all <- slingshot::getLineages(dims, clusterLabels = cluster, start.clus = opts$root)
  cat(sprintf("  Slingshot: lineages from every cell in %.0f s\n", proc.time()[["elapsed"]] - started))
  flush.console()
  started <- proc.time()[["elapsed"]]
  fit <- do.call(slingshot::getCurves, c(list(lineages_of_all[fitted, ]), curve_settings))
}
cat(sprintf("  Slingshot: curves fitted in %.0f s\n", proc.time()[["elapsed"]] - started))
flush.console()
# The curves as fitted, before every cell is placed -- the curves are drawn in each embedding from
# these, on the fitted cells (resolution" toggle (num cells)").
fit_on_fitted <- fit
if (length(fitted) < nrow(dims)) {
  started <- proc.time()[["elapsed"]]
  fit <- slingshot::predict(fit, dims)
  cat(sprintf("  Slingshot: %d cells placed on the curves in %.0f s\n", nrow(dims), proc.time()[["elapsed"]] - started))
  flush.console()
}
pseudotime <- slingshot::slingPseudotime(fit, na = TRUE)
weights <- slingshot::slingCurveWeights(fit)
branch <- as.character(slingshot::slingBranchID(fit))
lineages <- slingshot::slingLineages(fit)
lineage_names <- paste0("L", seq_along(lineages))
colnames(pseudotime) <- lineage_names
colnames(weights) <- lineage_names
# slingBranchID names lineages by their index ("1,2"); written with the L names used everywhere else.
branch <- vapply(strsplit(branch, ",", fixed = TRUE),
                 function(parts) paste0("L", parts, collapse = ","), character(1))
branch[is.na(slingshot::slingBranchID(fit))] <- ""

cells <- data.frame(barcode = rownames(dims), check.names = FALSE, stringsAsFactors = FALSE)
for (name in lineage_names) {
  cells[[paste0("pseudotime_", name)]] <- pseudotime[, name]
  cells[[paste0("weight_", name)]] <- weights[, name]
}
cells$branch <- branch
write.table(cells, file.path(opts$out, "cells.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "")

write.table(data.frame(lineage = lineage_names,
                       clusters = vapply(lineages, function(path) paste(path, collapse = " -> "), character(1))),
            file.path(opts$out, "lineages.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

edges <- data.frame(from = character(0), to = character(0))
mst <- tryCatch(slingshot::slingMST(fit), error = function(e) NULL)
if (!is.null(mst) && requireNamespace("igraph", quietly = TRUE)) {
  ends <- igraph::as_edgelist(mst)
  ends <- ends[!(ends[, 1] %in% c("Origin") | ends[, 2] %in% c("Origin")), , drop = FALSE]
  edges <- data.frame(from = ends[, 1], to = ends[, 2])
}
write.table(edges, file.path(opts$out, "mst.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

drawn <- character(0)
for (spec in opts$embedding) {
  name <- sub("=.*$", "", spec)
  path <- sub("^[^=]*=", "", spec)
  embedding <- read_table(path)
  coordinates <- as.matrix(embedding[, setdiff(colnames(embedding), "barcode"), drop = FALSE])
  rownames(coordinates) <- embedding$barcode
  # Slingshot's embedCurves on the cells the curves were fitted on (every cell without
  # --fit-cells): with all 99K cells it refitted every curve through every cell, minutes to hours per embedding.
  coordinates <- coordinates[rownames(dims)[fitted], , drop = FALSE]
  if (any(!stats::complete.cases(coordinates))) {
    cat(sprintf("  Curves not drawn in %s: it does not place every fitted cell\n", name))
    next
  }
  started <- proc.time()[["elapsed"]]
  embedded <- tryCatch(do.call(slingshot::embedCurves, c(list(fit_on_fitted, coordinates), embed_settings)), error = function(e) {
    cat(sprintf("  Curves not drawn in %s: %s\n", name, conditionMessage(e))); NULL })
  if (is.null(embedded)) next
  cat(sprintf("  Slingshot: curves drawn in %s from %d cells in %.0f s\n", name, length(fitted),
              proc.time()[["elapsed"]] - started))
  flush.console()
  curves <- slingshot::slingCurves(embedded)
  frames <- lapply(seq_along(curves), function(index) {
    curve <- curves[[index]]
    # Each curve through the cells on its lineage only.
    # Every fitted cell has a point on every curve; a cell off the lineage (weight 0) is placed at the curve's
    # stretched ends, which drew each curve far past its cells. A curve of Slingshot's approximated points (not
    # one per cell) is drawn as it is.
    order <- curve$ord
    if (nrow(curve$s) == length(curve$w)) order <- order[curve$w[order] > 0]
    points <- curve$s[order, , drop = FALSE]
    frame <- data.frame(lineage = lineage_names[index], point = seq_len(nrow(points)), points, check.names = FALSE)
    colnames(frame)[-(1:2)] <- colnames(coordinates)
    frame
  })
  write.table(do.call(rbind, frames), file.path(opts$out, paste0("curve_", name, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
  drawn <- c(drawn, name)
}

json_string <- function(x) paste0('"', gsub('"', '\\\\"', gsub("\\\\", "\\\\\\\\", x)), '"')
writeLines(c(
  "{",
  sprintf('  "slingshot_version": %s,', json_string(as.character(utils::packageVersion("slingshot")))),
  sprintf('  "r_version": %s,', json_string(R.version.string)),
  sprintf('  "cells": %d,', nrow(dims)),
  sprintf('  "dimensions": %d,', ncol(dims)),
  sprintf('  "fitted_cells": %d,', length(fitted)),
  sprintf('  "fit_cells_per_cluster": %s,', if (opts$fit_cells > 0L) as.character(opts$fit_cells) else "null"),
  sprintf('  "fit_by": %s,', if (opts$fit_cells > 0L) json_string(opts$fit_by) else "null"),
  '  "seed": 1,',
  sprintf('  "root": %s,', json_string(opts$root)),
  sprintf('  "lineages": %d,', length(lineages)),
  sprintf('  "curves_drawn_in": [%s],', paste(vapply(drawn, json_string, character(1)), collapse = ", ")),
  sprintf('  "curves_drawn_from_cells": %d', length(fitted)),
  "}"
), file.path(opts$out, "slingshot.json"))
cat(sprintf("  Slingshot: %d lineage(s): %s\n", length(lineages),
            paste(sprintf("%s %s", lineage_names, vapply(lineages, paste, character(1), collapse = ">")), collapse = "; ")))
