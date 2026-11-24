# Paralome the object's lineages with Monocle3 -- one principal graph learned through the cells in the
# object's UMAP, the cells ordered from the root the run names (--trajectory-root <column>:<value>), and every path
# from that root to a leaf of the graph taken as a lineage, with how each leaf was derived recorded. Reads the
# extraction's TSVs, never the .rds.
#
#   Rscript monocle3_trajectories.R --reduction <tsv> --name <embedding> --labels <tsv> --root <cluster> --out <dir>
#           [--ncenter <n>] [--minimal-branch-len <n>] [--use-partition y|n] [--close-loop y|n]
#           [--geodesic-distance-ratio <x>] [--euclidean-distance-ratio <x>] [--orthogonal-proj-tip y|n]
#           [--prune-graph y|n] [--l1-gamma <x>] [--l1-sigma <x>] [--reuse-graph y|n] [--weld <weld>]...
#           [--max-routes <n>]
#
# --reduction: barcode + the 2D or 3D UMAP the graph is learned in (extraction/reductions/<name>.tsv); --name its name
# --labels:    barcode + cluster (the --trajectory-root column)
# --ncenter, --minimal-branch-len: learn_graph's own (the graph's number of nodes; how long a branch must be to be
#              kept); not given, Monocle3's defaults
# --use-partition: learn one graph per partition Monocle3 finds (y, its default) or one across every cell (n)
# Learn_graph's own:
# --close-loop: join ends of the graph that come close (y), or keep it a tree (n, Paralome's default)
# --geodesic-distance-ratio, --euclidean-distance-ratio: with --close-loop y, how far apart two ends may be (along the
#              graph, in the UMAP) and still be joined (Monocle3's defaults 1/3 and 1)
# --orthogonal-proj-tip: extend each leaf out to the cells beyond it (y), or not (n, Monocle3's default)
# --prune-graph: trim short branches (y, Monocle3's default) or keep them (n)
# --l1-gamma, --l1-sigma: L1.gamma (how hard the graph is pulled short and smooth; default 0.5) and L1.sigma (how
#              local each node's pull on the cells is; default 0.01)
#
# --reuse-graph: read the graph learned before (graph.rds in --out) rather than learn it again (the
#              caller asks when only the welds changed)
# --weld:      join tips of the learned graph, named as its lineages are (L1, L2 ... in tips.tsv): L39=L42 tip to
#              tip; L30=L33=L35 each tip to a new node at their middle; L39~L42 L39's tip to the nearest node of L42's
#              own branch (from its last branch point to its tip); L39->L42 L39's tip to L42's, L42's tip staying an
#              end (L39 stopped short of it). Repeatable
#
# The lineages are every route from the root to each leaf left once the welds are made (in a tree, one per leaf):
# a route keeps the name of the tip it ends at; the route the learned graph had is the name alone, any other a
# letter after it (L44b, L44c ... by length). A cell's pseudotime on a route is Monocle3's, moved by how much longer
# the route is to the cell's graph node than the shortest way there. More routes than --max-routes (default 200)
# stops the fit, saying how many routes cross each weld.
#
# Writes in --out, as the Slingshot script does: cells.tsv (barcode, pseudotime_<L> and weight_<L> for every lineage:
# the cell's pseudotime on each lineage its graph node lies on, weight 1, blank and 0 off it; branch: those
# lineages), lineages.tsv (lineage, the clusters along it, its leaf: the graph node and its cluster), mst.tsv (the
# principal graph's edges, each node's cluster), curve_<name>.tsv (lineage, point, the path's nodes in the UMAP) and
# monocle3.json (versions and what was learned).

suppressPackageStartupMessages({
  library(methods)
})

args <- commandArgs(trailingOnly = TRUE)
opts <- list(ncenter = NA_integer_, minimal_branch_len = NA_real_, use_partition = "", close_loop = "n",
             geodesic_distance_ratio = NA_real_, euclidean_distance_ratio = NA_real_, orthogonal_proj_tip = "",
             prune_graph = "", l1_gamma = NA_real_, l1_sigma = NA_real_, reuse_graph = "", weld = character(0),
             max_routes = NA_integer_)
i <- 1
while (i <= length(args)) {
  key <- args[[i]]
  if (i == length(args)) stop(sprintf("%s needs a value", key))
  value <- args[[i + 1]]
  if (key == "--reduction") opts$reduction <- value
  else if (key == "--name") opts$name <- value
  else if (key == "--labels") opts$labels <- value
  else if (key == "--root") opts$root <- value
  else if (key == "--out") opts$out <- value
  else if (key == "--ncenter") opts$ncenter <- as.integer(value)
  else if (key == "--minimal-branch-len") opts$minimal_branch_len <- as.numeric(value)
  else if (key == "--use-partition") opts$use_partition <- value
  else if (key == "--close-loop") opts$close_loop <- value
  else if (key == "--geodesic-distance-ratio") opts$geodesic_distance_ratio <- as.numeric(value)
  else if (key == "--euclidean-distance-ratio") opts$euclidean_distance_ratio <- as.numeric(value)
  else if (key == "--orthogonal-proj-tip") opts$orthogonal_proj_tip <- value
  else if (key == "--prune-graph") opts$prune_graph <- value
  else if (key == "--l1-gamma") opts$l1_gamma <- as.numeric(value)
  else if (key == "--l1-sigma") opts$l1_sigma <- as.numeric(value)
  else if (key == "--reuse-graph") opts$reuse_graph <- value
  else if (key == "--weld") opts$weld <- c(opts$weld, value)
  else if (key == "--max-routes") opts$max_routes <- as.integer(value)
  else stop(sprintf("unknown option %s", key))
  i <- i + 2
}
for (needed in c("reduction", "name", "labels", "root", "out")) {
  if (is.null(opts[[needed]])) stop(sprintf("--%s is required", needed))
}
for (package in c("monocle3", "igraph")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    cat(sprintf("PARALOME_MISSING_R_PACKAGE %s\n", package))
    quit(status = 3)
  }
}
suppressPackageStartupMessages(library(monocle3))
dir.create(opts$out, showWarnings = FALSE, recursive = TRUE)

read_table <- function(path) {
  read.delim(path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE,
             colClasses = c(barcode = "character"), na.strings = c("", "NA"), quote = "")
}

reduction <- read_table(opts$reduction)
labels <- read.delim(opts$labels, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE,
                     colClasses = "character", na.strings = character(0), quote = "")
coordinates <- as.matrix(reduction[, setdiff(colnames(reduction), "barcode"), drop = FALSE])
rownames(coordinates) <- reduction$barcode
cluster <- labels$cluster[match(rownames(coordinates), labels$barcode)]
keep <- stats::complete.cases(coordinates) & !is.na(cluster) & cluster != ""
coordinates <- coordinates[keep, , drop = FALSE]
cluster <- cluster[keep]
if (!(opts$root %in% cluster)) {
  cat(sprintf("PARALOME_ROOT_NOT_FOUND %s\n", opts$root))
  quit(status = 4)
}
cat(sprintf("  Monocle3: %d cells, %s (%d dimensions), %d clusters, root %s\n",
            nrow(coordinates), opts$name, ncol(coordinates), length(unique(cluster)), opts$root))
flush.console()

# >>> the learned graph: what makes graph.rds
# The graph is learned from the UMAP alone; the cell_data_set carries one placeholder gene, which nothing reads.
expression <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                   dims = c(1L, nrow(coordinates)),
                                   dimnames = list("paralome_placeholder", rownames(coordinates)))
cds <- suppressWarnings(suppressMessages(monocle3::new_cell_data_set(
  expression,
  # Two columns: Monocle3's loop closing (connect_tips) subsets this table by row and fails on a one-column one.
  cell_metadata = data.frame(cluster = cluster, barcode = rownames(coordinates), row.names = rownames(coordinates)),
  gene_metadata = data.frame(gene_short_name = "paralome_placeholder", row.names = "paralome_placeholder"))))
colnames(coordinates) <- paste0("UMAP_", seq_len(ncol(coordinates)))
SingleCellExperiment::reducedDims(cds)[["UMAP"]] <- coordinates

# The learned graph is kept (graph.rds) so welds can be changed without learning it again: with
# --reuse-graph y and graph.rds there, it is read back rather than learned (the caller asks only when nothing but the
# welds changed).
graph_file <- file.path(opts$out, "graph.rds")
reused <- identical(opts$reuse_graph, "y") && file.exists(graph_file)
control <- list()
if (!is.na(opts$ncenter)) control$ncenter <- opts$ncenter
if (!is.na(opts$minimal_branch_len)) control$minimal_branch_len <- opts$minimal_branch_len
if (!is.na(opts$geodesic_distance_ratio)) control$geodesic_distance_ratio <- opts$geodesic_distance_ratio
if (!is.na(opts$euclidean_distance_ratio)) control$euclidean_distance_ratio <- opts$euclidean_distance_ratio
if (nzchar(opts$orthogonal_proj_tip)) control$orthogonal_proj_tip <- identical(opts$orthogonal_proj_tip, "y")
if (nzchar(opts$prune_graph)) control$prune_graph <- identical(opts$prune_graph, "y")
if (!is.na(opts$l1_gamma)) control$L1.gamma <- opts$l1_gamma
if (!is.na(opts$l1_sigma)) control$L1.sigma <- opts$l1_sigma
use_partition <- !identical(opts$use_partition, "n")
close_loop <- identical(opts$close_loop, "y")
if (reused) {
  cds <- readRDS(graph_file)
  if (!identical(colnames(cds), rownames(coordinates))) {
    cat("PARALOME_GRAPH_STALE\n")
    quit(status = 8)
  }
  cat("  Monocle3: the graph learned before is kept (only the welds changed)\n")
} else {
  started <- proc.time()[["elapsed"]]
  cds <- monocle3::cluster_cells(cds, reduction_method = "UMAP", random_seed = 1)
  cat(sprintf("  Monocle3: %d partition(s) of the UMAP in %.0f s\n",
              length(unique(monocle3::partitions(cds, reduction_method = "UMAP"))),
              proc.time()[["elapsed"]] - started))
  settings <- c(sprintf("use_partition = %s", use_partition), sprintf("close_loop = %s", close_loop),
                if (length(control)) sprintf("%s = %s", names(control), vapply(control, format, character(1))))
  cat(sprintf("  Monocle3: learn_graph with %s\n", paste(settings, collapse = ", ")))
  flush.console()
  set.seed(1)
  started <- proc.time()[["elapsed"]]
  cds <- monocle3::learn_graph(cds, use_partition = use_partition, close_loop = close_loop,
                               learn_graph_control = if (length(control)) control else NULL, verbose = FALSE)
  saveRDS(cds, graph_file)
  cat(sprintf("  Monocle3: a graph of %d nodes in %.0f s\n",
              igraph::vcount(monocle3::principal_graph(cds)[["UMAP"]]), proc.time()[["elapsed"]] - started))
}
flush.console()
# <<< the learned graph (Paralome keeps a learned graph while the code above this mark is the same)
partitions <- monocle3::partitions(cds, reduction_method = "UMAP")
graph <- monocle3::principal_graph(cds)[["UMAP"]]
aux <- cds@principal_graph_aux[["UMAP"]]
nodes <- aux$dp_mst                                    # dimensions x nodes, columns Y_1 ...
closest <- paste0("Y_", aux$pr_graph_cell_proj_closest_vertex[rownames(coordinates), 1])
node_distance <- function(a, b) sqrt(sum((nodes[, a] - nodes[, b])^2))
ends <- igraph::as_edgelist(graph)
igraph::E(graph)$weight <- vapply(seq_len(nrow(ends)), function(k) node_distance(ends[k, 1], ends[k, 2]), numeric(1))
igraph::E(graph)$weld <- ""

# The root: the graph node the most of the root cluster's cells are nearest.
root_node <- names(which.max(table(closest[cluster == opts$root])))
# Each node's cluster: the cluster of the most of the cells nearest it (none when no cell is).
node_cluster <- unlist(tapply(cluster, closest, function(x) names(which.max(table(x)))))
# The clusters a path passes through: each node's cluster in order, a cluster held by a single node between two
# others left out (a node at a junction takes whichever neighbour's cells are the more), repeats joined.
clusters_along <- function(path) {
  along <- as.character(unname(node_cluster[path]))
  along <- along[!is.na(along)]
  if (!length(along)) return(character(0))
  runs <- rle(along)
  inner <- seq_along(runs$lengths) > 1 & seq_along(runs$lengths) < length(runs$lengths)
  values <- runs$values[!(inner & runs$lengths < 2)]
  values[c(TRUE, values[-1] != values[-length(values)])]
}
describe <- function(path) paste(clusters_along(path), collapse = " -> ")

# The tips a weld names, whichever its kind.
weld_tips <- function(spec) strsplit(spec, "->|=|~")[[1]]

# The tips of the learned graph: every path from the root to a leaf (a node of one edge) of the root's part of it,
# named L1, L2 ... in the order of the clusters they pass. These names are what --trajectory-weld names, and a
# lineage keeps the name of the tip it ends at whatever is welded.
tips_of <- function(g) {
  parts <- igraph::components(g)$membership
  in_part <- names(parts)[parts == parts[[root_node]]]
  degree <- igraph::degree(g)
  leaves <- setdiff(in_part[degree[in_part] == 1], root_node)
  paths <- lapply(igraph::shortest_paths(g, from = root_node, to = leaves)$vpath, names)
  described <- vapply(paths, describe, character(1))
  order_ <- order(described, leaves)
  list(leaves = leaves[order_], paths = paths[order_], degree = degree)
}
learned <- tips_of(graph)
if (!length(learned$leaves)) {
  cat("PARALOME_NO_LEAVES\n")
  quit(status = 5)
}
tip_names <- paste0("L", seq_along(learned$leaves))
names(learned$paths) <- tip_names
tip_node <- setNames(learned$leaves, tip_names)
# Each tip's own branch: its path from the last branch point to the tip.
own_branch <- lapply(learned$paths, function(path) {
  forks <- which(learned$degree[path] > 2)
  path[(if (length(forks)) max(forks) else 1L):length(path)]
})
write.table(data.frame(
  lineage = tip_names, leaf_node = learned$leaves,
  leaf_cluster = vapply(learned$paths, function(p) { cl <- clusters_along(p); if (length(cl)) cl[length(cl)] else "" },
                        character(1)),
  clusters = vapply(learned$paths, describe, character(1)),
  branch_nodes = vapply(own_branch, length, integer(1)) - 1L,
  welded = vapply(tip_names, function(name) {
    hits <- opts$weld[vapply(opts$weld, function(spec) name %in% weld_tips(spec), logical(1))]
    paste(hits, collapse = ", ")
  }, character(1)),
  check.names = FALSE), file.path(opts$out, "tips.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# A learned leaf may be an uncommitted node rather than a fate: welds tether such tips. A=B joins the two tips; A=B=C... joins them all to a new node at
# their middle; A~B joins A's tip to the nearest node of B's own branch. Each weld goes into the principal graph and
# into the cell-by-cell graph Monocle3 orders the cells on, at its length in the UMAP.
cell_graph <- aux$pr_graph_cell_proj_tree
# Each node's distance from the root in the learned graph: a route moves away from the root along it.
learned_distance <- igraph::distances(graph, v = root_node)[1, ]
welds <- list()
# 
# -- A->B joins A's tip to B's, and B's tip stays an end (no meeting point is made on either branch).
kept_ends <- character(0)
quiet_nodes <- character(0)
add_edge <- function(u, v, spec, one_way = FALSE) {
  w <- node_distance(u, v)
  graph <<- igraph::add_edges(graph, c(u, v), attr = list(weight = w, weld = spec))
  cell_graph <<- igraph::add_edges(cell_graph, c(u, v), attr = list(weight = w))
  welds[[length(welds) + 1]] <<- list(spec = spec, from = u, to = v, length = w, one_way = one_way)
}
for (spec in opts$weld) {
  tilde <- grepl("~", spec, fixed = TRUE)
  # (an A->B weld holds no = or ~; "->" is read before "=" below)
  extend <- grepl("->", spec, fixed = TRUE)
  named <- weld_tips(spec)
  unknown <- setdiff(named, tip_names)
  if (length(unknown)) {
    cat(sprintf("PARALOME_WELD_UNKNOWN %s %s %d\n", spec, unknown[1], length(tip_names)))
    quit(status = 6)
  }
  if (extend) {
    add_edge(tip_node[[named[1]]], tip_node[[named[2]]], spec, one_way = TRUE)
    kept_ends <- c(kept_ends, tip_node[[named[2]]])
    quiet_nodes <- c(quiet_nodes, own_branch[[named[1]]], own_branch[[named[2]]])
  } else if (tilde) {
    branch <- own_branch[[named[2]]]
    reach <- vapply(branch, function(node) node_distance(tip_node[[named[1]]], node), numeric(1))
    add_edge(tip_node[[named[1]]], branch[which.min(reach)], spec, one_way = TRUE)
  } else if (length(named) == 2) {
    add_edge(tip_node[[named[1]]], tip_node[[named[2]]], spec)
  } else {
    hub <- paste0("W_", length(welds) + 1)
    nodes <- cbind(nodes, rowMeans(nodes[, tip_node[named], drop = FALSE]))
    colnames(nodes)[ncol(nodes)] <- hub
    graph <- igraph::add_vertices(graph, 1, name = hub)
    cell_graph <- igraph::add_vertices(cell_graph, 1, name = hub)
    for (name in named) add_edge(tip_node[[name]], hub, spec)
  }
}
if (length(welds)) {
  cat(sprintf("  Monocle3: %s\n", paste(vapply(welds, function(w) sprintf(
    "weld %s: %s (%s) to %s (%s), %.2f long", w$spec, w$from, node_cluster[w$from], w$to,
    if (is.na(node_cluster[w$to])) "the junction" else node_cluster[w$to], w$length), character(1)),
    collapse = "; ")))
  monocle3::principal_graph(cds)[["UMAP"]] <- graph
  cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_tree <- cell_graph
}

cds <- monocle3::order_cells(cds, reduction_method = "UMAP", root_pr_nodes = root_node)
pseudotime <- monocle3::pseudotime(cds, reduction_method = "UMAP")[rownames(coordinates)]
pseudotime[!is.finite(pseudotime)] <- NA

# The lineages ("there are multiple ways to leave the RPC state"): every route from the root to each
# leaf left once the welds are made -- one route for each leaf of a tree; a welded tip is a connection, not a leaf.
# A route keeps its leaf's tip name; the route the learned graph itself had is that name alone, any other a letter
# after it (L44b, L44c ... by length).
parts <- igraph::components(graph)$membership
in_part <- names(parts)[parts == parts[[root_node]]]
degree <- igraph::degree(graph)
leaves <- setdiff(in_part[degree[in_part] == 1], root_node)
extra <- setdiff(leaves, learned$leaves)                # leaves of a part a weld joined to the root's
leaf_name <- c(setNames(tip_names, learned$leaves),
               setNames(paste0("L", length(tip_names) + seq_along(extra)), extra))
# Where two ways around a loop meet: a node (not a leaf, not the root) every neighbour of which is nearer the root
# -- the cells there are reached from both sides. Each is an end like a leaf (with every tip welded into loops there
# would be none otherwise), named for the welded tip whose own branch it is on, or the first tip of the weld whose
# junction it is, else M1, M2 ...
shortest <- igraph::distances(graph, v = root_node)[1, ]
welded_tips <- unique(unlist(lapply(opts$weld[!grepl("->", opts$weld, fixed = TRUE)], function(spec)
  weld_tips(spec)[1:(if (grepl("~", spec, fixed = TRUE)) 1L else length(weld_tips(spec)))])))
meets <- Filter(function(node) {
  if (node == root_node || degree[[node]] < 2 || !is.finite(shortest[[node]])) return(FALSE)
  if (node %in% c(quiet_nodes, kept_ends)) return(FALSE)
  near <- names(igraph::neighbors(graph, node))
  all(shortest[near] < shortest[[node]])
}, in_part)
unnamed <- 0L
for (node in meets) {
  owner <- welded_tips[vapply(welded_tips, function(tip) node %in% own_branch[[tip]], logical(1))]
  hub_of <- Filter(function(w) w$to == node && startsWith(node, "W_"), welds)
  name <- if (length(owner)) owner[1] else if (length(hub_of)) strsplit(hub_of[[1]]$spec, "=", fixed = TRUE)[[1]][1]
          else { unnamed <- unnamed + 1L; paste0("M", unnamed) }
  leaf_name[[node]] <- name
}
leaves <- unique(c(leaves, kept_ends[kept_ends %in% in_part], meets))
route_length <- function(path) sum(vapply(seq_len(length(path) - 1L),
                                          function(k) node_distance(path[k], path[k + 1L]), numeric(1)))
# Which way a route may go: along the learned graph only away from the root; across an A->B or A~B
# weld only from A; across an = weld either way, and on from the tip it reaches back down that tip's own branch (the
# branch was an untethered corridor, not a fate's).
weld_key <- function(u, v) paste(u, v, sep = "\r")
crossing <- list()
for (w in welds) {
  crossing[[weld_key(w$from, w$to)]] <- !w$one_way    # TRUE: may then run back down the tip's branch
  if (!w$one_way) crossing[[weld_key(w$to, w$from)]] <- TRUE
}
branch_of_tip <- setNames(own_branch, tip_node[names(own_branch)])
goes_forward <- function(path) {
  back <- character(0)
  for (k in seq_len(length(path) - 1L)) {
    u <- path[k]; v <- path[k + 1L]
    key <- weld_key(u, v)
    if (!is.null(crossing[[key]])) {
      back <- if (isTRUE(crossing[[key]]) && !is.null(branch_of_tip[[v]])) branch_of_tip[[v]] else character(0)
      next
    }
    if (any(vapply(welds, function(w) (w$from == u && w$to == v) || (w$from == v && w$to == u), logical(1)))) {
      return(FALSE)                                    # a one-way weld crossed the wrong way
    }
    if (isTRUE(learned_distance[[v]] > learned_distance[[u]])) {
      back <- character(0)
    } else if (!(u %in% back && v %in% back)) {
      return(FALSE)
    }
  }
  TRUE
}
# --max-routes. Over it, the fit stops and says,
# for each weld, how many of the routes cross it.
max_routes <- if (is.na(opts$max_routes)) 200L else as.integer(opts$max_routes)
ordered_ends <- leaves[order(as.integer(sub("^[LM]", "", leaf_name[leaves])) + 1e6 * startsWith(leaf_name[leaves], "M"))]
found_by_end <- lapply(ordered_ends, function(leaf)
  Filter(goes_forward, lapply(igraph::all_simple_paths(graph, from = root_node, to = leaf), names)))
total <- sum(lengths(found_by_end))
if (total > max_routes) {
  weld_of <- igraph::E(graph)$weld
  crossing_count <- table(factor(unlist(lapply(unlist(found_by_end, recursive = FALSE), function(path) {
    ids <- igraph::get.edge.ids(graph, as.vector(rbind(path[-length(path)], path[-1])))
    unique(weld_of[ids][nzchar(weld_of[ids])])
  })), levels = unique(opts$weld)))
  cat(sprintf("PARALOME_TOO_MANY_ROUTES %d %d\n", total, max_routes))
  for (spec in names(crossing_count)) cat(sprintf("PARALOME_ROUTES_ACROSS %s %d\n", spec, crossing_count[[spec]]))
  quit(status = 7)
}
routes <- list()
for (index in seq_along(ordered_ends)) {
  leaf <- ordered_ends[index]
  found <- found_by_end[[index]]
  if (!length(found)) next
  lengths <- vapply(found, route_length, numeric(1))
  name <- leaf_name[[leaf]]
  tree <- if (leaf %in% learned$leaves) learned$paths[[name]] else NULL
  is_tree <- vapply(found, function(p) identical(p, tree), logical(1))
  order_ <- order(!is_tree, lengths)
  suffixes <- c("", letters[-1], paste0("_", 27:(27L + length(found))))
  for (k in seq_along(order_)) {
    path <- found[[order_[k]]]
    routes[[length(routes) + 1]] <- list(name = paste0(name, suffixes[k]), path = path, leaf = leaf,
                                         length = lengths[order_[k]])
  }
}
if (!length(routes)) {
  cat("PARALOME_NO_LEAVES\n")
  quit(status = 5)
}
lineage_names <- vapply(routes, `[[`, character(1), "name")
# A cell's pseudotime on a route: Monocle3's, moved by how much longer the route is to the cell's node than the
# shortest way there -- on the shortest way it is Monocle3's own.
cells <- data.frame(barcode = rownames(coordinates), check.names = FALSE, stringsAsFactors = FALSE)
on_lineage <- matrix(FALSE, nrow(coordinates), length(routes), dimnames = list(NULL, lineage_names))
for (index in seq_along(routes)) {
  path <- routes[[index]]$path
  along <- cumsum(c(0, vapply(seq_len(length(path) - 1L), function(k) node_distance(path[k], path[k + 1L]),
                              numeric(1))))
  names(along) <- path
  on <- closest %in% path & !is.na(pseudotime)
  on_lineage[, index] <- on
  name <- lineage_names[index]
  cells[[paste0("pseudotime_", name)]] <- ifelse(on, pseudotime + along[closest] - shortest[closest], NA)
  cells[[paste0("weight_", name)]] <- as.numeric(on)
}
cells$branch <- apply(on_lineage, 1, function(on) paste(lineage_names[on], collapse = ","))
write.table(cells, file.path(opts$out, "cells.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "")

weld_of_edge <- igraph::E(graph)$weld
route_welds <- function(path) {
  ids <- igraph::get.edge.ids(graph, as.vector(rbind(path[-length(path)], path[-1])))
  paste(unique(weld_of_edge[ids][nzchar(weld_of_edge[ids])]), collapse = ", ")
}
lineages <- data.frame(
  lineage = lineage_names,
  clusters = vapply(routes, function(r) describe(r$path), character(1)),
  leaf_node = vapply(routes, `[[`, character(1), "leaf"),
  leaf_cluster = vapply(routes, function(r) { cl <- clusters_along(r$path); if (length(cl)) cl[length(cl)] else "" },
                        character(1)),
  graph_nodes = vapply(routes, function(r) length(r$path), integer(1)),
  # How the leaf came about: the nodes from the route's last branch point to its leaf, and the cells on this
  # lineage alone (a short side branch of few cells is a spur of the graph more than a fate).
  branch_nodes = vapply(routes, function(r) {
    forks <- which(degree[r$path] > 2)
    length(r$path) - (if (length(forks)) max(forks) else 1L)
  }, integer(1)),
  route_length = round(vapply(routes, `[[`, numeric(1), "length"), 4),
  welds = vapply(routes, function(r) route_welds(r$path), character(1)),
  cells = colSums(on_lineage),
  cells_on_it_alone = vapply(seq_along(routes), function(index) sum(on_lineage[, index] & rowSums(on_lineage) == 1),
                             integer(1)),
  check.names = FALSE)
write.table(lineages, file.path(opts$out, "lineages.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

ends <- igraph::as_edgelist(graph)
write.table(data.frame(from = ends[, 1], to = ends[, 2],
                       from_cluster = unname(node_cluster[ends[, 1]]), to_cluster = unname(node_cluster[ends[, 2]]),
                       weld = weld_of_edge),
            file.path(opts$out, "mst.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "")

# Each lineage drawn as its route's nodes in the UMAP, named as the extraction named its columns.
frames <- lapply(seq_along(routes), function(index) {
  points <- t(nodes[, routes[[index]]$path, drop = FALSE])
  frame <- data.frame(lineage = lineage_names[index], point = seq_len(nrow(points)), points, check.names = FALSE)
  colnames(frame)[-(1:2)] <- setdiff(colnames(reduction), "barcode")
  frame
})
write.table(do.call(rbind, frames), file.path(opts$out, paste0("curve_", opts$name, ".tsv")),
            sep = "\t", quote = FALSE, row.names = FALSE)

json_string <- function(x) paste0('"', gsub('"', '\\\\"', gsub("\\\\", "\\\\\\\\", x)), '"')
placed <- sum(!is.na(pseudotime))
writeLines(c(
  "{",
  sprintf('  "monocle3_version": %s,', json_string(as.character(utils::packageVersion("monocle3")))),
  sprintf('  "r_version": %s,', json_string(R.version.string)),
  sprintf('  "cells": %d,', nrow(coordinates)),
  sprintf('  "cells_placed": %d,', placed),
  sprintf('  "embedding": %s,', json_string(opts$name)),
  sprintf('  "dimensions": %d,', ncol(coordinates)),
  sprintf('  "partitions": %d,', length(unique(partitions))),
  sprintf('  "use_partition": %s,', if (use_partition) "true" else "false"),
  sprintf('  "ncenter": %s,', if (is.na(opts$ncenter)) "null" else as.character(opts$ncenter)),
  sprintf('  "minimal_branch_len": %s,', if (is.na(opts$minimal_branch_len)) "null" else format(opts$minimal_branch_len)),
  sprintf('  "close_loop": %s,', if (close_loop) "true" else "false"),
  sprintf('  "learn_graph_control": {%s},', paste(sprintf('"%s": %s', names(control), vapply(control, function(v) {
    if (is.logical(v)) (if (v) "true" else "false") else format(v)
  }, character(1))), collapse = ", ")),
  '  "seed": 1,',
  sprintf('  "graph_reused": %s,', if (reused) "true" else "false"),
  sprintf('  "root": %s,', json_string(opts$root)),
  sprintf('  "root_node": %s,', json_string(root_node)),
  sprintf('  "graph_nodes": %d,', igraph::vcount(graph)),
  sprintf('  "tips": %d,', length(tip_names)),
  sprintf('  "welds": [%s],', paste(vapply(welds, function(w) sprintf(
    '{"weld": %s, "from": %s, "to": %s, "length": %s}', json_string(w$spec), json_string(w$from), json_string(w$to),
    format(round(w$length, 4))), character(1)), collapse = ", ")),
  sprintf('  "lineages": %d,', length(routes)),
  sprintf('  "curves_drawn_in": [%s]', json_string(opts$name)),
  "}"
), file.path(opts$out, "monocle3.json"))
if (placed < nrow(coordinates)) {
  cat(sprintf("  Monocle3: %d cells are in a part of the graph the root is not in, and have no pseudotime\n",
              nrow(coordinates) - placed))
}
cat(sprintf("  Monocle3: %d tips; %d lineage(s), root node %s: %s\n", length(tip_names), length(routes), root_node,
            paste(sprintf("%s %s", lineage_names, gsub(" -> ", ">", lineages$clusters)), collapse = "; ")))
