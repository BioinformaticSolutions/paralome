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

"""--statistic panel: the features that together tell a design's groups apart, or predict its covariate.

Named beside the statistic whose design it follows (``--statistic
anova,panel``), it fits a panel to every comparison of that design a panel can
be fitted to, with an elastic net, for the Olink and proteomics arms. It
answers whether the features together tell more than each single feature
alone, as evidence for the AI agent that reads Paralome's statistics.

THE COMPARISONS (``panel_comparisons``), from --factor, --covariate and --block-by / --pair-by:

  * each factor's level against its control (a main effect), on the samples at those two levels, the other factors
    held as unpenalised columns;
  * with two or more factors, each such pair within each combination of the other factors (a simple effect);
  * each factor of three or more levels, all its levels at once (multiclass), the other factors held;
  * --covariate: a regression panel predicting it from the features.
  Interaction contrasts are not a set of samples to tell apart; they are listed as not fitted, with the reason.

THE MODEL: one elastic net (mix ``ALPHA``) -- logistic for two groups, multinomial for several, Gaussian for a
covariate -- on the features' log2 values, each standardised on the samples it is fitted to, an unpenalised
intercept and unpenalised columns for the factors held; fitted by FISTA (accelerated proximal gradient) along a
path of ``N_LAMBDA`` penalties, warm-started. With --block-by or --pair-by each feature is centred within its block,
so a subject's own level is taken out (as the blocked ANOVA does), and every split keeps a block whole.

The model, its validation and chance are computed in ``paralome.massspec.panel_stats``.

WHAT IT WRITES, per comparison:

  * accuracy on samples the model did not see: ``REPEATS`` repeats of k-fold cross-validation (k = the smaller of
    ``MAX_FOLDS`` and the smallest group's count, in blocks when blocked), the penalty chosen inside each training
    part by its own ``INNER_FOLDS``-fold cross-validation; AUC (two groups; several: the mean of each group against
    the rest) or R^2 (a covariate) of the held-out predictions of each repeat -- their mean and 2.5th / 97.5th
    percentiles;
  * chance, for this design and these samples: the same cross-validation on ``PERMUTATIONS`` shufflings of the
    labels (within each block when blocked), ``PERMUTATION_REPEATS`` repeats each -- their mean, 95th percentile,
    and the panel's p-value, (1 + shufflings scoring at least the panel) / (1 + shufflings). Cross-validation on
    small samples can score below chance with no relationship at all (a held-out subject tends to go against the
    direction the rest learned); this is the level a panel is read against;
  * the best single feature, chosen inside each training part by its training score and scored on the held-out
    part, with its own chance level and p-value -- the panel set against it answers "more than each alone";
  * each feature alone, cross-validated on the same splits: its held-out AUC or R^2;
  * stability: ``BOOTSTRAPS`` resamples (stratified by group; of whole blocks when blocked) fitted at the penalty
    the full data chose -- how often each feature is selected, how often with its usual sign, its median weight
    (per standard deviation; for two groups positive is higher in the level, for a covariate it rises with it, for
    several groups one weight per group).

A comparison whose smallest group has fewer than ``MIN_PER_GROUP`` samples (blocks) is not fitted: cross-validation
needs every group in every fold. Fixed seed: the same data give the same numbers.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paralome.massspec import panel_stats as S
from paralome.massspec.panel_stats import MULTICLASS, REGRESSION, TWO_GROUP, Fitted, fit_panel  # noqa: F401

FOLDER = "panel"

# ---------------------------------------------------------------------------
# the design's comparisons
# ---------------------------------------------------------------------------

def panel_comparisons(rows: Sequence[Mapping[str, str]], factors: Sequence[str], covariate: Optional[str]
                      ) -> List[Dict[str, object]]:
    """Every panel this design allows, and the ones it does not, each with what it holds.

    Each: name, family, samples (positions in ``rows``), labels (0 = control or 0..k-1), classes (names),
    held (the factors held as unpenalised columns), within (a simple effect's levels), or ``not_fitted``."""
    from paralome.stats import parse_factor

    specs = [parse_factor(f) for f in factors]
    names = [c for c, _ in specs]
    levels: Dict[str, List[str]] = {}
    for column, control in specs:
        seen = sorted({str(r.get(column, "")).strip() for r in rows} - {""}, key=_natural)
        levels[column] = [control] + [v for v in seen if v != control]
    out: List[Dict[str, object]] = []
    value = lambda r, c: str(r.get(c, "")).strip()          # noqa: E731
    for column, control in specs:
        others = [c for c in names if c != column]
        for level in levels[column][1:]:
            idx = [i for i, r in enumerate(rows) if value(r, column) in (control, level)]
            out.append({"name": f"{level}_vs_{control}" + ("_main_effect" if others else ""),
                        "family": TWO_GROUP, "factor": column, "samples": idx,
                        "labels": [1 if value(rows[i], column) == level else 0 for i in idx],
                        "classes": [control, level], "held": others, "within": {}})
            for combo in itertools.product(*(levels[o] for o in others)) if others else []:
                where = dict(zip(others, combo))
                idx = [i for i, r in enumerate(rows) if value(r, column) in (control, level)
                       and all(value(r, o) == v for o, v in where.items())]
                within = "_".join(f"{o}{v}" for o, v in where.items())
                out.append({"name": f"{level}_vs_{control}_within_{_slug(within)}", "family": TWO_GROUP,
                            "factor": column, "samples": idx,
                            "labels": [1 if value(rows[i], column) == level else 0 for i in idx],
                            "classes": [control, level], "held": [], "within": where})
        if len(levels[column]) >= 3:
            idx = [i for i, r in enumerate(rows) if value(r, column) in levels[column]]
            out.append({"name": f"{column}_all_levels", "family": MULTICLASS, "factor": column, "samples": idx,
                        "labels": [levels[column].index(value(rows[i], column)) for i in idx],
                        "classes": list(levels[column]), "held": others, "within": {}})
    for order in range(2, len(names) + 1):
        for members in itertools.combinations(names, order):
            out.append({"name": ":".join(members), "family": "interaction", "not_fitted":
                        "an interaction contrast is a difference of differences, not a set of samples to tell apart"})
    if covariate:
        idx, vals = [], []
        for i, r in enumerate(rows):
            try:
                v = float(value(r, covariate))
            except ValueError:
                continue
            if math.isfinite(v):
                idx.append(i)
                vals.append(v)
        out.append({"name": f"{covariate}", "family": REGRESSION, "factor": covariate, "samples": idx,
                    "labels": vals, "classes": [], "held": [], "within": {}})
    return out


def _natural(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _held_columns(rows: Sequence[Mapping[str, str]], idx: Sequence[int], held: Sequence[str]) -> np.ndarray:
    cols = []
    for column in held:
        vals = [str(rows[i].get(column, "")).strip() for i in idx]
        for level in sorted(set(vals), key=_natural)[1:]:
            col = np.asarray([1.0 if v == level else 0.0 for v in vals])
            if 0 < col.sum() < len(col):
                cols.append(col)
    return np.column_stack(cols) if cols else np.zeros((len(idx), 0))


# ---------------------------------------------------------------------------
# a stratum's panels, written
# ---------------------------------------------------------------------------

PERFORMANCE_FIELDS = ["comparison", "kind", "status", "reason", "groups", "n_samples", "blocked_by", "held",
                      "metric", "panel_mean", "panel_low", "panel_high", "chance_mean", "chance_95",
                      "panel_p_value", "best_single_mean", "best_single_chance_mean", "best_single_p_value",
                      "panel_minus_best_single", "best_single_feature", "best_single_gene", "features_tested",
                      "features_selected", "folds", "repeats", "permutations", "permutation_repeats",
                      "bootstraps", "alpha", "lambda", "seed"]
MARKER_FIELDS = ["comparison", "feature", "gene_name", "measured_form", "selection_frequency", "usual_sign",
                 "sign_consistency", "median_weight", "single_mean", "single_low", "single_high"]


def run_panels(log2_values: np.ndarray, keys: Sequence[str], annotation: Sequence[Mapping[str, str]],
               rows: Sequence[Mapping[str, str]], *, factors: Sequence[str], covariate: Optional[str],
               blocks_by: Optional[str], eligible: np.ndarray, folder, prefix: str, stamp: str, version: str,
               echo=print, threads: int = 0) -> Dict[str, object]:
    """Every panel of one stratum's design: ``log2_values`` features x the stratum's samples (``rows``).

    ``threads`` worker processes (--threads; 0, every CPU this process may use) share each panel's
    cross-validation repeats, shufflings and resamples."""
    from pathlib import Path

    from paralome.massspec.stages import _write_rows

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    performance: List[Dict[str, object]] = []
    markers: List[Dict[str, object]] = []
    with S.panel_pool(threads) as pool:
        for comp in panel_comparisons(rows, factors, covariate):
            base = {"comparison": comp["name"], "kind": comp["family"], "alpha": S.ALPHA, "seed": S.SEED,
                    "blocked_by": blocks_by or ""}
            if comp.get("not_fitted"):
                performance.append({**base, "status": "not_fitted", "reason": comp["not_fitted"]})
                continue
            idx = list(comp["samples"])
            family = str(comp["family"])
            y = np.asarray(comp["labels"], dtype=float if family == REGRESSION else int)
            n_classes = len(comp["classes"]) if family == MULTICLASS else 2
            blocks = (np.asarray([str(rows[i].get(blocks_by, "")).strip() for i in idx]) if blocks_by else None)
            sub = log2_values[:, idx]
            finite = np.all(np.isfinite(sub), axis=1)
            usable = eligible & finite & (np.where(finite[:, None], sub, 0.0).std(axis=1) > 0)
            groups = (";".join(f"{c}:{int(np.sum(y == i))}" for i, c in enumerate(comp["classes"]))
                      if family != REGRESSION else f"{comp['factor']}:{len(idx)} samples, {np.min(y) if len(y) else ''}"
                      f"-{np.max(y) if len(y) else ''}")
            base.update(groups=groups, n_samples=len(idx), held=",".join(comp["held"]) or
                        ",".join(f"{k}={v}" for k, v in comp["within"].items()),
                        metric="R2" if family == REGRESSION else "AUC", features_tested=int(usable.sum()))
            smallest = S.smallest_group(family, y, blocks, n_classes) if len(idx) else 0
            if smallest < S.MIN_PER_GROUP or usable.sum() < 1:
                why = (f"its smallest group has {smallest} {'blocks' if blocks is not None else 'samples'}; a panel "
                       f"needs at least {S.MIN_PER_GROUP} in every group (each fold of the cross-validation needs every group)"
                       if smallest < S.MIN_PER_GROUP else "no feature is measured in all of its samples")
                performance.append({**base, "status": "not_fitted", "reason": why})
                continue
            Z = _held_columns(rows, idx, comp["held"])
            result = S.fit_panel(family, sub[usable].T, y, Z=Z, blocks=blocks, n_classes=n_classes,
                                  pool=pool)
            ps = result["panel_scores"]
            ss = result["single_scores"]                                   # repeats x features
            bs = result["best_scores"]
            single_mean = np.nanmean(ss, axis=0)
            best = int(np.nanargmax(single_mean)) if np.isfinite(single_mean).any() else 0
            used = np.flatnonzero(usable)
            perm = result["perm_panel"]
            performance.append({
                **base, "status": "fitted", "reason": "",
                "panel_mean": _f(np.nanmean(ps)), "panel_low": _f(np.nanpercentile(ps, 2.5)),
                "panel_high": _f(np.nanpercentile(ps, 97.5)),
                "chance_mean": _f(np.nanmean(perm)) if len(perm) else "",
                "chance_95": _f(np.nanpercentile(perm, 95)) if len(perm) else "",
                "panel_p_value": _f(result["panel_p"]),
                "best_single_mean": _f(np.nanmean(bs)),
                "best_single_chance_mean": _f(np.nanmean(result["perm_best"])) if len(result["perm_best"]) else "",
                "best_single_p_value": _f(result["best_p"]),
                "panel_minus_best_single": _f(np.nanmean(ps) - np.nanmean(bs)),
                "best_single_feature": keys[used[best]], "best_single_gene": annotation[used[best]].get("gene_name", ""),
                "features_selected": result["n_selected_full"], "folds": result["k"], "repeats": result["repeats"],
                "permutations": result["permutations"], "permutation_repeats": S.PERMUTATION_REPEATS,
                "bootstraps": result["bootstraps"], "lambda": _f(result["lambda"]),
            })
            W = result["weights"]                                           # bootstraps x features x outputs
            for j, f_index in enumerate(used):
                ann = annotation[f_index]
                if family == MULTICLASS:
                    med = np.nanmedian(W[:, j, :], axis=0)
                    sign = ";".join(f"{c}:{_f(m)}" for c, m in zip(comp["classes"], med))
                    usual, consistency = "", ""
                else:
                    med = float(np.nanmedian(W[:, j, 0]))
                    pos = float(result["positive"][j, 0])
                    sel = float(result["selected"][j])
                    usual = "+" if pos >= sel - pos else "-"
                    consistency = _f(max(pos, sel - pos) / sel) if sel > 0 else ""
                    sign = _f(med)
                markers.append({
                    "comparison": comp["name"], "feature": keys[f_index], "gene_name": ann.get("gene_name", ""),
                    "measured_form": ann.get("measured_form", ""), "selection_frequency": _f(result["selected"][j]),
                    "usual_sign": usual, "sign_consistency": consistency, "median_weight": sign,
                    "single_mean": _f(single_mean[j]), "single_low": _f(np.nanpercentile(ss[:, j], 2.5)),
                    "single_high": _f(np.nanpercentile(ss[:, j], 97.5)),
                })
            echo(f"[paralome] panel {comp['name']}: {base['metric']} {_f(np.nanmean(ps))} "
                 f"({_f(np.nanpercentile(ps, 2.5))}-{_f(np.nanpercentile(ps, 97.5))}), chance "
                 f"{_f(np.nanmean(perm)) if len(perm) else '-'}, p {_f(result['panel_p'])}; best single assay "
                 f"{_f(np.nanmean(bs))}")
    markers.sort(key=lambda r: (r["comparison"], -float(r["selection_frequency"] or 0)))
    perf_path = folder / f"{prefix}_panel_performance_{stamp}.tsv"
    mark_path = folder / f"{prefix}_panel_markers_{stamp}.tsv"
    _write_rows(perf_path, performance, PERFORMANCE_FIELDS)
    _write_rows(mark_path, markers, MARKER_FIELDS)
    return {"version": version, "performance": str(perf_path), "markers": str(mark_path),
            "fitted": sum(1 for r in performance if r["status"] == "fitted"),
            "not_fitted": sum(1 for r in performance if r["status"] != "fitted")}


def _f(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(v) else f"{v:.4g}"


def split_statistic(statistic: str) -> Tuple[str, bool]:
    """``--statistic`` without ``panel``, and whether panel was named."""
    tokens = [t.strip() for t in str(statistic).split(",") if t.strip()]
    others = [t for t in tokens if t.lower() != FOLDER]
    return ",".join(others), len(others) != len(tokens)


# ---------------------------------------------------------------------------
# With --backend slurm, the panels are a SLURM job of their own
# ---------------------------------------------------------------------------
# `paralome olink` runs on the login node and submits its own work, as every Paralome pipeline does.
# With --backend slurm each stratum's panels are written as a plan, run by `paralome-worker panels run`
# in a job Paralome submits -- --threads N CPUs, or with no --threads the configured resources.threads
# (8 as shipped), the pipelines' own standard -- and a job after them builds the statistics zip again
# with the panels in it. With --backend local they run in the command.

PLAN_NAME = "panel_plan.json"
VALUES_NAME = "panel_plan_values.npz"


def write_panel_plan(log2_values: np.ndarray, keys: Sequence[str], annotation: Sequence[Mapping[str, str]],
                     rows: Sequence[Mapping[str, str]], *, factors: Sequence[str], covariate: Optional[str],
                     blocks_by: Optional[str], eligible: np.ndarray, folder, prefix: str, stamp: str,
                     version: str) -> str:
    """Everything ``run_panels`` reads, in ``<folder>/panel_plan.json`` (+ the values, .npz): the plan a panel job runs."""
    import json
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / VALUES_NAME, log2_values=np.asarray(log2_values, dtype=float),
             eligible=np.asarray(eligible, dtype=bool))
    plan = {"keys": list(keys), "annotation": [dict(a) for a in annotation], "rows": [dict(r) for r in rows],
            "factors": list(factors), "covariate": covariate, "blocks_by": blocks_by, "folder": str(folder.resolve()),
            "prefix": prefix, "stamp": stamp, "version": version, "values": VALUES_NAME}
    path = folder / PLAN_NAME
    path.write_text(json.dumps(plan, indent=1) + "\n")
    return str(path.resolve())


def run_panel_plan(plan_path, threads: int = 0, echo=print) -> Dict[str, object]:
    """`paralome-worker panels run`: the panels of one written plan."""
    import json
    from pathlib import Path

    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    values = np.load(plan_path.parent / plan["values"])
    return run_panels(values["log2_values"], plan["keys"], plan["annotation"], plan["rows"], factors=plan["factors"],
                      covariate=plan["covariate"], blocks_by=plan["blocks_by"], eligible=values["eligible"],
                      folder=plan["folder"], prefix=plan["prefix"], stamp=plan["stamp"], version=plan["version"],
                      echo=echo, threads=threads)


def configured_threads() -> int:
    """The configured resources.threads (the pipelines' --threads default; 8 as shipped)."""
    try:
        from paralome.config.user import load_config
        from paralome.pipeline.project import cfg_get

        return max(1, int(cfg_get(load_config(), "resources", "threads", 8)))
    except Exception:
        return 8


def submit_panels(plans: Sequence[str], *, workdir, rebuild_argv: Sequence[str], threads: int = 0,
                  partition: Optional[str] = None, submit=None, echo=print) -> Dict[str, object]:
    """One SLURM job per plan (``threads`` CPUs; 0, the configured resources.threads), then one job after them all (afterany)
    running ``rebuild_argv`` -- the statistics zip built again, the panels in it. A refused submission is said and
    recorded; the statistics already written stand."""
    import shlex
    from pathlib import Path

    from paralome.binaries import worker_command
    from paralome.pipeline.slurm import SlurmOptions, afterany, sbatch, write_sbatch

    workdir = Path(workdir)
    logs, scripts = workdir / "logs", workdir / "scripts"
    submit = submit or sbatch
    cpus = int(threads) if threads else configured_threads()
    jobs: List[str] = []
    out: Dict[str, object] = {"plans": list(plans), "jobs": jobs, "cpus": cpus}
    try:
        for i, plan in enumerate(plans):
            options = SlurmOptions(job_name="paralome_panels", cpus_per_task=cpus, mem="16G", partition=partition,
                                   output=logs / f"panels_{i}_%j.out", error=logs / f"panels_{i}_%j.err")
            script = write_sbatch(scripts / f"panels_{i}.sbatch", options,
                                  f"{worker_command()} panels run --plan {shlex.quote(str(plan))} --threads {cpus}\n")
            jobs.append(str(submit(script)))
            echo(f"[paralome] panels: job {jobs[-1]} ({out['cpus']} CPUs) runs {plan}")
        rebuild = write_sbatch(scripts / "panels_statistics_output.sbatch",
                               SlurmOptions(job_name="paralome_panels_output", cpus_per_task=1, mem="4G",
                                            partition=partition, output=logs / "panels_output_%j.out",
                                            error=logs / "panels_output_%j.err"),
                               " ".join(shlex.quote(str(a)) for a in rebuild_argv) + "\n")
        out["rebuild_job"] = str(submit(rebuild, dependency=":".join([afterany(jobs[0]), *jobs[1:]])))
        echo(f"[paralome] panels: job {out['rebuild_job']} puts them in the statistics zip when they end")
        out["submitted"] = True
    except Exception as exc:
        echo(f"[paralome] panels: not submitted -- {exc}")
        out.update(submitted=False, reason=str(exc))
    return out


def submit_arm_panels(plans: Sequence[str], *, arm: str, project, prefix: str, results_roots: Sequence,
                      samples_table, zip_dir, threads: int = 0, partition: Optional[str] = None, submit=None,
                      echo=print) -> Dict[str, object]:
    """The panel jobs of one `paralome olink` / `paralome proteomics` run, and the statistics zip built again after
    them (`paralome-worker output statistics ... --overwrite`, the arguments the command's own build used)."""
    from pathlib import Path

    from paralome.binaries import worker_argv
    from paralome.browser.arm import STAGE as OUTPUT_STAGE

    workdir = Path(project) / OUTPUT_STAGE
    rebuild = [*worker_argv(), "output", "statistics", "--arm", arm, "--out", str(Path(workdir).resolve()),
               "--prefix", prefix]
    for root in results_roots:
        rebuild += ["--results", str(Path(root).resolve())]
    rebuild += ["--samples-table", str(Path(samples_table).resolve()), "--zip-dir", str(Path(zip_dir).resolve()),
                "--overwrite"]
    return submit_panels(plans, workdir=Path(project) / "panel_jobs", rebuild_argv=rebuild, threads=threads,
                         partition=partition, submit=submit, echo=echo)
