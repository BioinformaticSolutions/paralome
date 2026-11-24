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

# The trajectories stage's gene models by Monocle3 (--trajectory-model monocle3; least squares, ls, is the
# default). Paralome builds each lineage's model -- the run's full factorial crossed with
# a natural spline of pseudotime, the same columns the least-squares engine fits -- and the nested models each term
# is tested by; this script fits each of them with Monocle3's own fit_models (negative binomial, the cell's size
# factor its offset) and compares each pair as Monocle3's compare_models does: LLR = 2 |logLik(full) -
# logLik(reduced)|, df = their residual df's difference, p from the chi-squared. Negative binomial, not Monocle3's
# default quasi-Poisson: a quasi-Poisson fit has no log-likelihood, so compare_models would give it no p.
#
#   Rscript monocle3_gene_models.R --counts <rna_counts.mtx> --cores <n> --lineage <folder> [--lineage <folder> ...]
#
# Each lineage folder holds design.tsv (column: the cell's column of the counts, 1-based; size_factor; c1..cn the
# model's columns but the intercept), genes.txt (the counts' rows tested, 1-based), models.tsv (model, its columns
# separated by commas; none is the intercept alone) and tests.tsv (term, full, reduced). Writes results.tsv (gene_row,
# term, LLR, df, p_value) there, and prints PARALOME_LINEAGE_SECONDS <folder> <seconds> for each.

args <- commandArgs(trailingOnly = TRUE)
opts <- list(lineage = character(0), cores = 1L)
i <- 1
while (i <= length(args)) {
  key <- args[i]
  value <- if (i < length(args)) args[i + 1] else NA
  if (key == "--counts") opts$counts <- value
  else if (key == "--cores") opts$cores <- as.integer(value)
  else if (key == "--lineage") opts$lineage <- c(opts$lineage, value)
  else stop(sprintf("unknown option %s", key))
  i <- i + 2
}
if (is.null(opts$counts) || !length(opts$lineage)) stop("--counts and at least one --lineage are needed")

suppressPackageStartupMessages({
  library(Matrix)
  library(monocle3)
})
cat(sprintf("  Monocle3 %s gene models: reading %s\n", as.character(utils::packageVersion("monocle3")), opts$counts))
counts <- methods::as(Matrix::readMM(opts$counts), "CsparseMatrix")
cat(sprintf("  %d genes x %d cells\n", nrow(counts), ncol(counts)))

for (folder in opts$lineage) {
  started <- Sys.time()
  design <- utils::read.delim(file.path(folder, "design.tsv"), check.names = FALSE)
  genes <- scan(file.path(folder, "genes.txt"), what = integer(), quiet = TRUE)
  models <- utils::read.delim(file.path(folder, "models.tsv"), colClasses = "character", na.strings = character(0))
  tests <- utils::read.delim(file.path(folder, "tests.tsv"), colClasses = "character")
  cell_names <- paste0("cell", design$column)
  gene_names <- paste0("g", genes)
  expression <- counts[genes, design$column, drop = FALSE]
  dimnames(expression) <- list(gene_names, cell_names)
  cell_data <- design[, setdiff(colnames(design), "column"), drop = FALSE]
  rownames(cell_data) <- cell_names
  cds <- suppressWarnings(monocle3::new_cell_data_set(
    expression, cell_metadata = cell_data,
    gene_metadata = data.frame(gene_short_name = gene_names, row.names = gene_names)))
  # The size factors are the whole library's (every gene of the cell), as the least-squares engine's CPM.
  SummarizedExperiment::colData(cds)$Size_Factor <- design$size_factor
  cat(sprintf("  %s: %d cells, %d genes, %d models, %d tests\n", folder, nrow(design), length(genes),
              nrow(models), nrow(tests)))
  fits <- list()
  for (k in seq_len(nrow(models))) {
    columns <- strsplit(models$columns[k], ",", fixed = TRUE)[[1]]
    columns <- columns[nzchar(columns)]
    formula <- if (length(columns)) paste("~", paste(columns, collapse = " + ")) else "~ 1"
    fitted <- monocle3::fit_models(cds, model_formula_str = formula, expression_family = "negbinomial",
                                   cores = opts$cores, verbose = FALSE)
    glance <- monocle3::evaluate_fits(fitted)
    fits[[models$model[k]]] <- data.frame(gene = as.character(glance$gene_id), logLik = glance$logLik,
                                          df_residual = glance$df_residual, stringsAsFactors = FALSE)
    rm(fitted, glance)
    invisible(gc())
    cat(sprintf("    model %s (%d columns) fitted\n", models$model[k], length(columns)))
  }
  rows <- list()
  for (k in seq_len(nrow(tests))) {
    full <- fits[[tests$full[k]]]
    reduced <- fits[[tests$reduced[k]]]
    joined <- merge(full, reduced, by = "gene", suffixes = c(".full", ".reduced"), all = TRUE)
    llr <- 2 * abs(joined$logLik.full - joined$logLik.reduced)
    dfs <- round(abs(joined$df_residual.full - joined$df_residual.reduced))
    p <- ifelse(dfs > 0, stats::pchisq(llr, dfs, lower.tail = FALSE), NA_real_)
    rows[[k]] <- data.frame(gene_row = as.integer(sub("^g", "", joined$gene)), term = tests$term[k], LLR = llr,
                            df = dfs, p_value = p)
  }
  results <- do.call(rbind, rows)
  utils::write.table(results, file.path(folder, "results.tsv"), sep = "\t", quote = FALSE, row.names = FALSE,
                     na = "")
  seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  cat(sprintf("PARALOME_LINEAGE_SECONDS %s %.1f\n", folder, seconds))
}
