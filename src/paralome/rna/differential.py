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

"""Bulk RNA differential expression: the request, the fit, and every file it writes.

The statistics are the shared engine's (``paralome.engine`` over the
plan ``paralome.comparisons.plan_differential`` makes), fitted on the
log2(FPKM + pseudocount) matrix with the back-transformed FPKM beside it, and
every result is written by ``paralome.results`` into one folder per statistic
under ``<out>/results/<folder>/`` (``t``, ``paired_t``, ``lm``, ``anova``) --
the same names and columns bulk ATAC and the pseudobulk pipelines write.

What stays here is what is bulk RNA's own: reading the sheet and the matrix,
the pseudocount checks, the optional raw-FPKM expression gate on the
significant sheets, the stage-level views (PCA, sample-outlier QC,
the expression heatmap), the checkpoint and its staleness rules, the QC audit
and --refine-responders. What 'rna fpkm-filter-sweep' wrote (in an older
result) is cleared when a result is replaced.

ONE WRITER, TWO CALLERS. The local run and the SLURM finalize both
hand their fitted (uncorrected) results to ``write_differential_result``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from paralome.checkpoints import stage_checkpoint_path
from paralome.comparisons import KIND_FTEST, DifferentialPlan, plan_differential
from paralome.csvio import skip_comment_lines
from paralome.engine import ComparisonResult, concatenate, finalize, fit_plan
from paralome.pipeline.jsonio import (
    checkpoint_stale_reason, write_json,
    input_fingerprint, inputs_changed_reason, invalidate_checkpoint,
    version_stale_reason,
)
from paralome.stats import StatisticRequest, parse_statistic_request
from paralome.version import __version__


METADATA_FIELDS = ["gene_id", "gene_name", "chromosome", "start", "end", "strand", "length_bp"]

#: The sheet headers the pipeline's manifest reader accepts for the sample, matched without regard to
#: case, so this stage accepts every sheet the preflight accepts. The groups are --factor columns,
#: named as the sheet names them; there is no condition column to find.
SAMPLE_ID_ALIASES = ("sample_id", "sample", "id", "name")


def _resolve_sheet_aliases(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Give every row an exact ``sample_id`` key, from the first alias the header carries.

    Only a key that is MISSING is added: a sheet that already says
    ``sample_id`` is returned exactly as read.
    """
    if not rows:
        return rows
    lowered = {str(key).strip().lower(): key for key in rows[0] if key is not None}
    sample_key = next((lowered[a] for a in SAMPLE_ID_ALIASES if a in lowered), None)
    if sample_key is None or sample_key == "sample_id":
        return rows
    resolved = []
    for row in rows:
        row = dict(row)
        row.setdefault("sample_id", row.get(sample_key))
        resolved.append(row)
    return resolved


def _read_samples(path: Path) -> List[Dict[str, str]]:
    # Utf-8-sig, as rna/fpkm.py and fastq.manifest read a sheet. A
    # sheet saved by Excel starts with a byte-order mark, which made the first
    # header "﻿sample_id" and the sheet fail with "requires sample_id".
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        # Cells are stripped HERE, once, so every reader of these rows sees the same value ("ctrl " is
        # ctrl everywhere).
        rows = [
            {key: (value.strip() if isinstance(value, str) else value) for key, value in r.items()}
            for r in csv.DictReader(skip_comment_lines(handle))
        ]
    rows = _resolve_sheet_aliases(rows)
    if not rows or "sample_id" not in rows[0]:
        raise ValueError(
            "RNA metadata requires a sample_id column (or one of its aliases: "
            + ", ".join(SAMPLE_ID_ALIASES[1:]) + ")"
        )
    ids = [str(r.get("sample_id") or "").strip() for r in rows]
    if any(not value for value in ids):
        raise ValueError("RNA metadata contains an empty sample_id")
    if len(ids) != len(set(ids)):
        raise ValueError("RNA metadata contains duplicate sample_id values")
    return rows


def _read_matrix(
    path: Path,
    sample_ids: Sequence[str],
) -> Tuple[List[Dict[str, str]], np.ndarray]:
    meta: List[Dict[str, str]] = []
    values: List[List[float]] = []
    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [s for s in sample_ids if s not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Expression matrix missing sample(s): " + ",".join(missing))
        for row in reader:
            meta.append({key: row.get(key, "") for key in METADATA_FIELDS})
            values.append([float(row[s]) for s in sample_ids])
    if not meta:
        raise ValueError(f"Expression matrix contains no genes: {path}")
    return meta, np.asarray(values, dtype=float)


def _gene_name_coverage(meta: Sequence[Mapping[str, str]]) -> float:
    """Fraction of genes whose ``gene_name`` differs from their ``gene_id``.

    gene_name/Common_name capture can only reflect
    what's actually present in ``--expression``'s TSV: a matrix whose GTF
    carried no ``gene_name`` attribute, or one built by an older Paralome
    without the column, repeats the Ensembl ID in every row (surfaced
    explicitly here, rather than left for a user to notice only by eyeballing
    the Common_name column, after exactly this symptom was seen on a real
    run). See ``GENE_NAME_WARNING``.
    """
    if not meta:
        return 0.0
    hits = sum(
        1
        for row in meta
        if str(row.get("gene_name") or "").strip()
        and str(row.get("gene_name")).strip() != str(row.get("gene_id") or "").strip()
    )
    return hits / len(meta)


#: The one sentence both execution paths put in the checkpoint when no gene has a symbol. It gives both
#: diagnoses: a GTF with no gene_name attribute cannot be fixed by regenerating the matrix.
GENE_NAME_WARNING = (
    "No gene symbols found in --expression (gene_name equals gene_id for every "
    "gene), so every Common_name repeats the Ensembl gene_id. Either the GTF "
    "carries no gene_name attribute, or the matrix was built by an older "
    "Paralome that did not capture symbols -- rebuild it with 'rna quantify-fpkm' "
    "from a GTF that names its genes to get real gene symbols."
)


def gene_name_warning_for(coverage: float) -> Optional[str]:
    """The notice for a matrix with no symbols at all, else ``None``."""
    return GENE_NAME_WARNING if coverage == 0.0 else None



def _raw_fpkm(log2_value: np.ndarray, pseudocount: float) -> np.ndarray:
    """Back-transform log2(FPKM + pseudocount) values to raw linear FPKM.

    Paralome's expression matrix is log2(FPKM + pseudocount) by convention
    (see ``paralome.rna.fpkm``); the CSV report schema works in raw FPKM throughout
    (Control_mean/Treatment_mean/Fold_change/per-sample columns). Clipped to
    0.0 to absorb floating-point noise around a true raw value of 0.
    """
    return np.clip(np.exp2(log2_value) - pseudocount, 0.0, None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The request
#
# The design layer that lived here (the slices, the ANOVA's terms
# and contrasts, every name and the collision check) is paralome.comparisons,
# which every pipeline asks. What is left is what only bulk RNA refuses.
# ---------------------------------------------------------------------------

#: The stage's checkpoint schema. v8: one folder of results per
#: statistic, written by paralome.results; a checkpoint of an earlier schema
#: is made again whatever else it records.
CHECKPOINT_SCHEMA = "paralome.rna.differential_expression.v9"


def check_request_options(
    *,
    statistic: str,
    factors: Sequence[str],
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    refine_responders: Optional[int] = None,
) -> StatisticRequest:
    """The part of a request that can be judged without the sample sheet.

    --statistic is read by ``stats.parse_statistic_request``, the one reader
    every pipeline uses; bulk RNA has no data level, so ``t:cell`` is refused
    there. The refusals below are bulk RNA's own.
    """
    # The refine defines its comparisons from the factors, as its
    # help says. Asked for without any, it declined after the result was
    # written, said nothing, and the checkpoint recorded it as done.
    if refine_responders and not factors:
        raise ValueError(
            f"--refine-responders {int(refine_responders)} needs the comparisons the QC "
            "audit defines from --factor, and no --factor was given. Add the factor "
            "column(s) of the sample sheet and their control levels (--factor <column>:<control>), "
            "or leave out "
            "--refine-responders"
        )
    # Both together are refused: --covariate-feature would otherwise be ignored.
    if covariate and covariate_feature:
        raise ValueError(
            f"--covariate {covariate} and --covariate-feature {covariate_feature} were both "
            "given; lm regresses on one of them. Give one"
        )
    return parse_statistic_request(
        statistic, factors=tuple(factors), covariate=covariate or covariate_feature,
        pair_by=pair_by,
    )


def covariate_feature_values(
    meta: Sequence[Mapping[str, str]], matrix: np.ndarray, wanted: str,
) -> np.ndarray:
    """--covariate-feature's log2 expression in every sample: the first gene any of
    whose identifier columns (gene_id, gene_name, ...) is exactly ``wanted``."""
    for index, row in enumerate(meta):
        if any(str(value) == str(wanted) for value in dict(row).values()):
            return np.asarray(matrix[index], dtype=float)
    raise ValueError(
        f"--covariate-feature {wanted!r} is not in this expression matrix under any "
        "of its identifier columns."
    )


def plan_bulk_differential(
    sample_rows: Sequence[Mapping[str, str]],
    *,
    statistic: str,
    factors: Sequence[str],
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    covariate_feature_values: Optional[np.ndarray] = None,
    refine_responders: Optional[int] = None,
) -> DifferentialPlan:
    """Every comparison this run makes, or the refusal, before a gene is read.

    ONE implementation, called by ``differential_expression``, the SLURM plan,
    every block and the finalize: each of them fits and names exactly
    the comparisons that were checked. Every sample of the sheet is in scope;
    the groups are the --factor columns' levels, each against the control
    level its --factor names.

    ``covariate_feature_values`` is --covariate-feature's expression, found in
    the matrix by ``covariate_feature_values``. Before the matrix is read (the
    request is checked first, so a mistyped option does not cost a finished
    result) a stand-in that varies is used, and the plan is made again with
    the real values once the matrix is in hand.
    """
    request = check_request_options(
        statistic=statistic, factors=factors,
        pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
        refine_responders=refine_responders,
    )
    rows = [dict(row) for row in sample_rows]
    values = covariate_feature_values
    if covariate_feature and not covariate and values is None:
        values = np.arange(len(rows), dtype=float)
    return plan_differential(
        rows, request=request,
        factors=list(factors), pair_by=pair_by, covariate=covariate,
        covariate_feature=covariate_feature if not covariate else None,
        covariate_feature_values=values,
    )


def statistic_tokens(statistic: str) -> Tuple[str, ...]:
    """The statistics a --statistic string names, sorted: what a result is compared by."""
    return tuple(sorted(parse_statistic_request(str(statistic), validate_companions=False).statistics))


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def power_alpha(alpha_metric: str, alpha_value: float) -> float:
    """The alpha the per-gene power columns are computed at.

    --alpha's cutoff when it is a raw-p cutoff, else 0.05 -- the rule
    ``results.write_statistic_folder`` applies to the power summary, so the
    columns and the summary beside them agree.
    """
    return float(alpha_value) if str(alpha_metric).lower() == "p" else 0.05


def fit_blocks(
    matrix: np.ndarray,
    raw: np.ndarray,
    plan: DifferentialPlan,
    *,
    start: int,
    stop: int,
    block_size: int,
    alpha: float,
) -> List[ComparisonResult]:
    """The engine's fit of genes ``start:stop``, one block of ``block_size`` at a time.

    The local run fits its blocks here, and each SLURM block task fits its
    slice here, so the two paths do the same arithmetic on the same rows.
    Uncorrected: ``engine.finalize`` runs once, over every gene.
    """
    blocks = []
    for first in range(int(start), int(stop), max(1, int(block_size))):
        last = min(first + max(1, int(block_size)), int(stop))
        blocks.append(fit_plan(
            matrix[first:last], plan, raw=raw[first:last], alpha=alpha,
            replication_unit="samples",
        ))
    return concatenate(blocks)


def expression_gate(
    raw_control_mean: np.ndarray,
    raw_treatment_mean: np.ndarray,
    *,
    min_expression: float,
    min_expression_difference: float,
) -> np.ndarray:
    """optional low-expression/small-effect gate, in raw FPKM.

    A gene passes when at least one arm's raw mean reaches ``min_expression``
    AND the two raw means differ by at least ``min_expression_difference`` --
    the rule this gate has applied since it was ported from the earlier
    RNA-seq script (``control_mean >= 5`` / ``abs(control_mean -
    treatment_mean) >= 5``). Both at 0 is no gate. A gene with no raw mean
    (an arm with no samples) does not pass an active gate.
    """
    control = np.asarray(raw_control_mean, dtype=float)
    treatment = np.asarray(raw_treatment_mean, dtype=float)
    if float(min_expression) <= 0.0 and float(min_expression_difference) <= 0.0:
        return np.ones(control.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        return (
            (np.fmax(control, treatment) >= float(min_expression))
            & (np.abs(treatment - control) >= float(min_expression_difference))
        )


def has_arms(result: ComparisonResult) -> bool:
    """Whether a result has two groups whose raw means the gate can read."""
    comparison = result.comparison
    return (
        result.kind != KIND_FTEST
        and bool(comparison.control_index) and bool(comparison.treatment_index)
    )


def apply_expression_gate(
    results: Sequence[ComparisonResult],
    *,
    min_expression: float,
    min_expression_difference: float,
) -> None:
    """Set ``passes_expression_gate`` on every result (``results.EXPORT_GATE_COLUMN``).

    A contrast or a slope with two arms is gated on its own two raw means --
    the samples THAT comparison compares, so a within-CD contrast is gated on
    CD expression. An F-test, or a regression with no arms, has no two means
    to gate on and passes.
    """
    from paralome.results import EXPORT_GATE_COLUMN

    for result in results:
        size = len(result)
        if has_arms(result):
            gate = expression_gate(
                result.columns["raw_control_mean"], result.columns["raw_treatment_mean"],
                min_expression=min_expression, min_expression_difference=min_expression_difference,
            )
        else:
            gate = np.ones(size, dtype=bool)
        result.columns[EXPORT_GATE_COLUMN] = gate


def run_qc_beside_the_result(
    differential_dir: Path,
    samples_path: Path,
    *,
    factors: Sequence[str],
    min_expression: float = 0.0,
    refine_responders: Optional[int] = None,
    expression_matrix: Optional[Path] = None,
    statistic: str = "t",
    min_expression_difference: float = 0.0,
    pseudocount: float = 1.0,
    alpha_metric: Optional[str] = None,
    alpha_value: Optional[float] = None,
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Dict[str, object]:
    """The QC audit, called from BOTH execution paths, after the checkpoint.

    AN AUDIT IS NOT A RESULT, and must never gate one. So this runs only once
    ``differential_expression.json`` is already on disk, and
    ``run_qc_ensemble`` is written never to raise -- a QC that takes a finished
    analysis down with it is worse than no QC at all. The result dictionary is
    returned for the checkpoint's information and is read by nothing.

    ONE IMPLEMENTATION, TWO CALLERS, so the audit cannot differ between the
    monolithic and parallel paths.

    ``refine_responders`` is a second, OPTIONAL step after the audit: re-run
    the real pairwise test on a trimmed, best-responders/most-typical-controls
    population per comparison (see ``run_responder_refine``). It is not itself
    an audit -- it is asked for by name -- so its own failure is recorded but
    kept from erasing an otherwise-successful audit report.

    ``alpha_metric``/``alpha_value``, ``statistic`` and ``pseudocount`` are
    the run's own, and reach both steps. None leaves each step at its own
    default.

    And the run's --statistic as given, --pair-by, --covariate and
    --covariate-feature reach the refine, which re-runs every statistic the run
    asked for that compares two groups.
    """
    # No QC ensemble is drawn here: the viewer shows it from the statistics output, with the user's
    # toggles. Older audit files are removed so none describes this result.
    remove_qc_audit_outputs(Path(differential_dir))
    report: Dict[str, object] = {}

    if refine_responders and expression_matrix is not None:
        from paralome.rna.qc_ensemble import run_responder_refine
        try:
            report["refined_responders"] = run_responder_refine(
                Path(differential_dir), Path(samples_path), Path(expression_matrix),
                factors=list(factors), n=int(refine_responders), statistic=statistic,
                min_expression=float(min_expression),
                min_expression_difference=float(min_expression_difference),
                pseudocount=float(pseudocount),
                alpha_metric=alpha_metric, alpha_value=alpha_value,
                pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
            )
        except Exception as exc:  # pragma: no cover - belt as well as braces
            report["refined_responders"] = {
                "complete": False, "reason": f"{type(exc).__name__}: {exc}"}
        # SAID, as run_qc_ensemble says its own reason. The refine
        # was asked for by name, and a decline, an exception or a comparison
        # that failed left no message anywhere: the result read as refined.
        for line in refine_outcome_lines(report["refined_responders"], int(refine_responders)):
            print(line, file=sys.stderr, flush=True)
    return report


#: What run_qc_ensemble and write_qc_interactive write directly under results/qc/, by name. These are
#: this stage's own outputs (the audit is called from both execution paths after the checkpoint), named
#: here so a recompute can remove them.
QC_AUDIT_FILES = (
    "paralome_qc_ensemble.pdf", "qc_summary.tsv", "qc_array_quality.html",
    "qc_responders.html", "qc_report.json",
)


def remove_qc_audit_outputs(output_dir: Path) -> List[str]:
    """Remove the previous audit's files under ``results/qc`` (not refined_top*)."""
    qc = Path(output_dir) / "results" / "qc"
    removed: List[str] = []
    for name in QC_AUDIT_FILES:
        path = qc / name
        if path.is_file():
            try:
                path.unlink()
            except OSError as error:
                print(
                    f"[paralome] differential_expression: could not remove {path} from the "
                    f"audit being replaced ({error}); left in place.", file=sys.stderr,
                )
                continue
            removed.append(str(path))
    return removed


def refine_complete(refined: object) -> bool:
    """Whether a --refine-responders report says every comparison was refined."""
    if not isinstance(refined, Mapping) or not refined.get("complete"):
        return False
    comparisons = refined.get("comparisons")
    if isinstance(comparisons, Mapping):
        return all(
            isinstance(outcome, Mapping) and bool(outcome.get("complete", True))
            for outcome in comparisons.values()
        )
    return True


def refine_outcome_lines(refined: object, requested: int) -> List[str]:
    """The sentences that say what --refine-responders did, or why it did not."""
    lines: List[str] = []
    if not isinstance(refined, Mapping):
        return [f"  --refine-responders {requested} did not run: no report"]
    if not refined.get("complete"):
        lines.append(
            f"  --refine-responders {requested} did not run: {refined.get('reason') or 'no reason given'}"
        )
        return lines
    comparisons = refined.get("comparisons")
    if isinstance(comparisons, Mapping):
        for label, outcome in comparisons.items():
            if isinstance(outcome, Mapping) and not outcome.get("complete", True):
                lines.append(
                    f"  --refine-responders {requested}: {label} was not refined: "
                    f"{outcome.get('reason') or 'no reason given'}"
                )
    return lines

def check_pseudocount(pseudocount: float) -> None:
    """0 was accepted here while quantification refuses it. The
    matrix is log2(FPKM + pseudocount), which has no value at 0 for a gene
    with no reads, so no matrix this stage can read was built with 0."""
    if not pseudocount > 0:
        raise ValueError(
            f"--pseudocount must be greater than 0 (got {pseudocount!r}). Give the "
            "pseudocount --expression was built with ('rna quantify-fpkm' uses 1.0 "
            "unless told otherwise)"
        )


def quantification_pseudocount_for(expression_matrix: Union[str, Path]) -> Optional[float]:
    """The pseudocount ``--expression`` was built with, when it can be known.

    A matrix written by ``rna quantify-fpkm`` sits at
    ``<out>/results/machine/<matrix>.tsv`` beside the checkpoint that records
    the pseudocount it was built with (``<out>/checkpoints/
    rna_fpkm_quantification.json``). Returns ``None`` for a matrix from
    anywhere else, or a checkpoint that does not say.
    """
    matrix = Path(expression_matrix).expanduser()
    try:
        matrix = matrix.resolve()
        out = matrix.parents[2]
    except (IndexError, OSError):
        return None
    checkpoint_path = stage_checkpoint_path(out, "rna_fpkm_quantification")
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("complete"):
        return None
    recorded_matrix = checkpoint.get("gene_log2_fpkm")
    if recorded_matrix:
        try:
            if Path(str(recorded_matrix)).resolve() != matrix:
                return None     # another file of that tree, not the matrix it records
        except OSError:
            return None
    value = checkpoint.get("pseudocount")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def check_pseudocount_matches_matrix(expression_matrix: Union[str, Path], pseudocount: float) -> None:
    """Refuse a --pseudocount other than the one the matrix was built with.

    The back-transform 2^x - pseudocount behind Control_mean,
    Treatment_mean, the per-sample columns and the expression gate used this
    stage's own --pseudocount (default 1.0) whatever the quantification had
    used: after 'rna quantify-fpkm --pseudocount 0.5' and a default
    'rna differential', every raw mean was 0.5 FPKM low and the gate and the
    pathway lists shifted with it, with nothing said. The quantification
    checkpoint records the value; it is read here on both backends.
    """
    recorded = quantification_pseudocount_for(expression_matrix)
    if recorded is None or math.isclose(recorded, float(pseudocount), rel_tol=1e-9, abs_tol=1e-12):
        return
    raise ValueError(
        f"--pseudocount {float(pseudocount):g} does not match the pseudocount "
        f"{recorded:g} that {expression_matrix} was built with (recorded in its "
        "rna quantify-fpkm checkpoint). The raw FPKM means, per-sample values and the "
        f"expression gate are back-transformed with it: pass --pseudocount {recorded:g}, "
        "or quantify again with the value you want"
    )

def differential_request_fingerprint(
    expression_matrix: Union[str, Path], sample_rows: Sequence[Mapping[str, str]],
) -> str:
    """What was ASKED for: the sheet's rows, and the matrix as a path, not a file.

    As rna/fpkm.py's _request_fingerprint. Unlike the input
    fingerprint this needs no matrix to exist, so it tells "the matrix of this
    result has since been deleted" (same request) from "--expression now names
    something else" (a different one).
    """
    return input_fingerprint(content={
        "expression": os.path.abspath(os.path.expanduser(str(expression_matrix))),
        "samples": [dict(row) for row in sample_rows],
    })


#: Why a finished result with no input fingerprint is made again.
NO_FINGERPRINT_REASON = (
    "this result was made by an older Paralome that did not record what it was made from"
)


def say_kept_without_matrix(expression_matrix: Union[str, Path]) -> None:
    """Name the matrix of a result that is kept although the matrix is gone.

    The sheet is still compared then (its rows are in the request
    fingerprint); the matrix cannot be, and that was not said.
    """
    if not Path(expression_matrix).expanduser().exists():
        print(
            "[paralome] differential_expression: the finished result is kept although its "
            f"input file no longer exists: {expression_matrix}",
            file=sys.stderr,
        )


def _folder_files(statistics: object) -> List[str]:
    """Every file the per-statistic manifests of a checkpoint list, their own manifests too."""
    files: List[str] = []
    if not isinstance(statistics, Mapping):
        return files
    for manifest in statistics.values():
        if not isinstance(manifest, Mapping):
            continue
        files.extend(str(path) for path in manifest.get("files") or () if path)
        if manifest.get("manifest"):
            files.append(str(manifest["manifest"]))
    return files


def _recorded_outputs(previous: Mapping[str, object]) -> List[str]:
    """The result files a finished checkpoint vouches for: what must still exist."""
    files: List[str] = []
    statistics = previous.get("statistics")
    if isinstance(statistics, Mapping):
        for manifest in statistics.values():
            if not isinstance(manifest, Mapping):
                continue
            for key in ("workbook", "evidome_input"):
                if manifest.get(key):
                    files.append(str(manifest[key]))
            comparisons = manifest.get("comparisons")
            if isinstance(comparisons, Mapping):
                for entry in comparisons.values():
                    if isinstance(entry, Mapping):
                        files.extend(
                            str(entry[key]) for key in ("report", "statistics")
                            if entry.get(key)
                        )
    return files


def differential_stale_reason(
    previous: Mapping[str, object],
    *,
    statistic: str,
    factors: Sequence[str],
    pair_by: Optional[str],
    covariate: Optional[str],
    covariate_feature: Optional[str],
    min_expression: float,
    min_expression_difference: float,
    pseudocount: float,
    alpha_metric: str,
    alpha_value: float,
    fingerprint: str,
    request_fingerprint: str = "",
    output_dir: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Why a finished result does not answer this request, or ``None``.

    The decision ``differential_expression`` makes, as a
    function, so the SLURM plan makes the same one.

    A result of an earlier schema (one table for every statistic)
    is made again; --statistic is compared as the statistics it names, in any
    order, and --pair-by, --covariate and --covariate-feature are compared,
    since each changes what paired-t and lm compute.
    """
    # Version checked FIRST: the version only bumps on a real behaviour
    # change, which is what makes a mismatch a reliable staleness signal --
    # see jsonio.version_stale_reason.
    stale_reason = version_stale_reason(previous)
    if stale_reason is None and previous.get("schema") != CHECKPOINT_SCHEMA:
        stale_reason = (
            f"it was written in the layout {previous.get('schema')!r}, and this version "
            f"writes {CHECKPOINT_SCHEMA!r} (one folder of results per statistic)"
        )
    if stale_reason is None:
        try:
            recorded_statistics = statistic_tokens(str(previous.get("statistic") or ""))
        except ValueError:
            recorded_statistics = (str(previous.get("statistic")),)
        wanted = statistic_tokens(statistic)
        if recorded_statistics != wanted:
            stale_reason = (
                f"statistic changed from {previous.get('statistic')!r} to {statistic!r}"
            )
    if stale_reason is None:
        stale_reason = checkpoint_stale_reason(
            previous,
            factors=list(factors),
            pair_by=pair_by,
            covariate=covariate,
            covariate_feature=covariate_feature,
            min_expression=min_expression,
            min_expression_difference=min_expression_difference,
            pseudocount=pseudocount,
            alpha_metric=alpha_metric,
            alpha_value=alpha_value,
        )
    # A result that records no fingerprint cannot vouch for its inputs.
    if stale_reason is None and not previous.get("input_fingerprint"):
        stale_reason = NO_FINGERPRINT_REASON
    if stale_reason is None and fingerprint:
        stale_reason = inputs_changed_reason(previous, fingerprint)
    # Only a request identical to the recorded one may keep a result
    # whose matrix has gone -- see differential_request_fingerprint.
    recorded_request = previous.get("request_fingerprint")
    if (stale_reason is None and not fingerprint and recorded_request
            and recorded_request != request_fingerprint):
        stale_reason = (
            "the sample sheet or --expression no longer names what this "
            "result was computed from"
        )
    # A checkpoint saying complete beside a deleted result is as
    # misleading as one beside a wrongly named file.
    if stale_reason is None:
        gone = []
        gone.extend(path for path in _recorded_outputs(previous) if not Path(path).exists())
        if gone:
            stale_reason = (
                f"{len(gone)} recorded output(s) no longer exist, "
                f"e.g. {gone[0]}"
            )
    # A checkpoint whose outputs lie in another folder is a COPY of
    # a result, not this folder's.
    if stale_reason is None and output_dir is not None:
        elsewhere = recorded_results_root(previous)
        here = (Path(output_dir).resolve() / "results")
        if elsewhere is not None and elsewhere != here:
            stale_reason = (
                f"this checkpoint records its outputs under {elsewhere}, not under "
                f"{here}: the folder was copied or moved, so the result is made again here"
            )
    return stale_reason


def recorded_results_root(previous: Mapping[str, object]) -> Optional[Path]:
    """The ``results`` folder a checkpoint's outputs were written under, or None.

    Read from a statistic folder's manifest (``<out>/results/<folder>``), or, for an older result, its
    sample-outlier or ``all_genes_statistics`` table under results/machine.
    """
    statistics = previous.get("statistics")
    if isinstance(statistics, Mapping):
        for manifest in statistics.values():
            directory = manifest.get("directory") if isinstance(manifest, Mapping) else None
            if directory:
                try:
                    return Path(str(directory)).resolve().parent
                except OSError:
                    return None
    for key in ("sample_outlier_metrics", "all_genes_statistics"):
        recorded = previous.get(key)
        if recorded and isinstance(recorded, str):
            try:
                return Path(recorded).resolve().parents[1]
            except (IndexError, OSError):
                return None
    return None


def _rerooted(value: str, old_results: Optional[Path], results: Path) -> Path:
    """``value`` as a path under this folder's results, if it was recorded under another's.

    Checkpoint paths are absolute, so in a folder that was copied or moved
    they would otherwise resolve outside the current results folder and
    nothing would be removed.
    """
    path = Path(value)
    if old_results is None or old_results == results:
        return path
    try:
        return results / path.resolve().relative_to(old_results)
    except (ValueError, OSError):
        return path


#: What a result of the earlier layout (schema v5-v7, one table for every
#: statistic) recorded, so that the first run of this version removes it
#: instead of leaving it beside the folders that replace it.
_EARLIER_LAYOUT_KEYS = (
    "all_genes_statistics", "all_genes_clean_statistics", "workbook", "volcano",
)


def remove_previous_outputs(previous: Mapping[str, object], output_dir: Path) -> List[str]:
    """Remove the files a replaced result recorded, before the new ones are written.

    Only files the old checkpoint lists, and only under this stage's
    own results folder, are removed -- nothing this stage did not write. That
    is every file of every statistic folder it wrote (a folder this run does
    not write again is not left behind), its stage-level views, the refine's
    and the audit's files, and what 'rna fpkm-filter-sweep' wrote from it.
    """
    results = (Path(output_dir) / "results").resolve()
    old_results = recorded_results_root(previous)
    recorded: List[object] = [previous.get(key) for key in (previous.get("output_sha256") or {})]
    recorded.extend(_folder_files(previous.get("statistics")))
    recorded.extend(previous.get(key) for key in _EARLIER_LAYOUT_KEYS)
    for key in ("exports", "pathway_exports"):
        group = previous.get(key)
        if isinstance(group, Mapping):
            recorded.extend(group.values())
    term_exports = previous.get("term_exports")
    if isinstance(term_exports, Mapping):
        for group in term_exports.values():
            if isinstance(group, Mapping):
                recorded.extend(group.values())
    # And what --refine-responders wrote for the replaced request, under results/qc/refined_top<N>/; and
    # the audit's files.
    for key in ("refined_outputs", "qc_outputs"):
        listed = previous.get(key)
        if isinstance(listed, (list, tuple)):
            recorded.extend(listed)
    removed: List[str] = remove_qc_audit_outputs(Path(output_dir))
    # And what 'rna fpkm-filter-sweep' wrote FROM this result, from
    # its own record in each statistic folder (and the earlier single folder).
    sweep_dirs = [results / "pathway_analysis_inputs" / "fpkm_filter_sweep"]
    statistics = previous.get("statistics")
    if isinstance(statistics, Mapping):
        sweep_dirs.extend(results / str(folder) / SWEEP_DIRNAME for folder in statistics)
    for sweep_dir in sweep_dirs:
        removed.extend(remove_previous_sweep_outputs(sweep_dir))
    for value in recorded:
        if not value or not isinstance(value, str):
            continue
        path = _rerooted(value, old_results, results)
        try:
            inside = path.resolve().relative_to(results)
        except ValueError:
            continue
        if SWEEP_DIRNAME in inside.parts:
            continue        # removed above, from the sweep's own record
        if path.is_file():
            try:
                path.unlink()
            except OSError as error:
                print(
                    f"[paralome] differential_expression: could not remove {path} "
                    f"from the result being replaced ({error}); left in place.",
                    file=sys.stderr,
                )
                continue
            removed.append(str(path))
    # A folder this stage wrote is its own to remove, once nothing is left in it: a statistic folder and its
    # machine/, an older layout's term folders, and a refined run's folders.
    for value in sorted(removed, key=len, reverse=True):
        for parent in Path(value).resolve().parents:
            if parent == results or results not in parent.parents:
                break
            if parent.is_dir() and not any(parent.iterdir()):
                _remove_empty_folder(parent)
    return removed


def _remove_empty_folder(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass        # an empty folder left behind is harmless


def previous_sweep_files(sweep_dir: Path) -> List[Path]:
    """The files an earlier 'rna fpkm-filter-sweep' in ``sweep_dir`` says it wrote.

    From its checkpoint's ``files`` list (written since this
    version), else from the ``file`` column of its summary table, plus the
    summary and the checkpoint themselves. Nothing the sweep did not record
    is named: a user's own file in that folder is not this stage's to remove.
    """
    sweep_dir = Path(sweep_dir)
    checkpoint_path = sweep_dir / "fpkm_filter_sweep.json"
    summary_path = sweep_dir / "fpkm_filter_sweep_summary.tsv"
    files: List[Path] = []
    listed: List[str] = []
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if isinstance(checkpoint, Mapping):
            recorded = checkpoint.get("files")
            if isinstance(recorded, (list, tuple)):
                listed = [str(item) for item in recorded if item]
            summary = checkpoint.get("summary")
            if summary:
                summary_path = Path(str(summary))
    except (OSError, ValueError):
        pass
    if not listed and summary_path.is_file():
        try:
            with summary_path.open(newline="") as handle:
                for row in csv.DictReader(skip_comment_lines(handle), delimiter="\t"):
                    if row.get("file"):
                        listed.append(str(row["file"]))
        except (OSError, ValueError, csv.Error):
            listed = []
    for item in listed:
        path = Path(item)
        # Recorded absolute; re-rooted by name into THIS sweep folder, so a
        # copied or moved result is cleared as well.
        files.append(sweep_dir / path.name if path.parent != sweep_dir else path)
    for own in (summary_path, checkpoint_path):
        if own.is_file():
            files.append(own)
    return files


def remove_previous_sweep_outputs(sweep_dir: Path) -> List[str]:
    """Remove what an earlier sweep in ``sweep_dir`` recorded writing."""
    removed: List[str] = []
    sweep_dir = Path(sweep_dir)
    for path in previous_sweep_files(sweep_dir):
        try:
            inside = path.resolve().relative_to(sweep_dir.resolve())
        except (ValueError, OSError):
            continue
        if len(inside.parts) != 1 or not path.is_file():
            continue
        try:
            path.unlink()
        except OSError as error:
            print(
                f"[paralome] fpkm_filter_sweep: could not remove {path} from the sweep "
                f"being replaced ({error}); left in place.", file=sys.stderr,
            )
            continue
        removed.append(str(path))
    if sweep_dir.is_dir() and not any(sweep_dir.iterdir()):
        _remove_empty_folder(sweep_dir)
    return removed


def record_refined_outputs(
    checkpoint_path: Path, checkpoint: Mapping[str, object], report: Mapping[str, object],
    requested: Optional[int] = None,
) -> Dict[str, object]:
    """Add what --refine-responders did, and the audit's files, to the checkpoint.

    The refine runs after the checkpoint, so it is added afterwards, and a
    recompute can then remove the replaced request's refined folders
    (remove_previous_outputs only removes what a checkpoint lists).

    ``refine_responders`` is recorded as DONE (the N asked for) only when
    every comparison was refined. A refine that did not complete leaves 0
    there and its reason in ``refine_responders_failed``, so asking again
    tries again. The audit's own files (``qc_outputs``) are listed for the
    same reason the refine's are.

    ``refine_responders_attempted`` records the N that was TRIED, whatever
    came of it. A refine that declines on this result (no cell pair differs in
    one factor, say) declines every time, so the SLURM plan treats an
    unchanged result whose attempted N equals the request as answered, prints
    the stored reason, and retries only on --overwrite or a different N.
    """
    refined = report.get("refined_responders") if isinstance(report, Mapping) else None
    files = refined.get("files") if isinstance(refined, Mapping) else None
    updated = dict(checkpoint)
    changed = False
    if files:
        updated["refined_outputs"] = sorted(
            {*(str(path) for path in checkpoint.get("refined_outputs") or ()), *(str(path) for path in files)}
        )
        changed = True
    if requested:
        updated["refine_responders_attempted"] = int(requested)
        if refine_complete(refined):
            updated["refine_responders"] = int(requested)
            updated.pop("refine_responders_failed", None)
        else:
            updated["refine_responders"] = 0
            lines = refine_outcome_lines(refined, int(requested))
            updated["refine_responders_failed"] = "; ".join(line.strip() for line in lines) or (
                f"--refine-responders {int(requested)} did not complete"
            )
        changed = True
    audit_files = qc_audit_files_of(report)
    if audit_files:
        updated["qc_outputs"] = sorted(
            {*(str(path) for path in checkpoint.get("qc_outputs") or ()), *audit_files}
        )
        changed = True
    if not changed:
        return updated
    write_json(Path(checkpoint_path), updated, atomic=True)
    return updated


def qc_audit_files_of(report: object) -> List[str]:
    """The files run_qc_ensemble's report says it wrote."""
    if not isinstance(report, Mapping):
        return []
    files: List[str] = []
    for key in ("pdf", "summary"):
        value = report.get(key)
        if value:
            files.append(str(value))
    interactive = report.get("interactive")
    if isinstance(interactive, Mapping):
        files.extend(str(value) for value in interactive.values() if value)
    return files


@contextmanager
def _atomic_path(path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_tsv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    fields: Sequence[str],
    *,
    comment_lines: Sequence[str] = (),
    delimiter: str = "\t",
) -> None:
    """Write a delimited table, optionally preceded by ``#``-prefixed comments.

    ``comment_lines`` is opt-in (default empty, so every existing
    caller is byte-for-byte unchanged) -- used so far only by the new
    pathway-analysis exports, to stamp the generating package/version
    directly on the file itself where it will not interfere -- a leading comment line
    a spreadsheet tool or `head -1` ignores/skips, rather than a new column
    that would shift every downstream field).

    ``delimiter`` likewise defaults to the historical tab, so the machine-side
    tables under ``machine/`` are unchanged. The pathway exports pass ",".
    Everything is supposed to be CSV -- a parameter here rather than a second writer, because a copy of this
    function differing in one character is how the duplication in this codebase
    starts every time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_path(path) as temporary:
        with temporary.open("w", newline="") as handle:
            for comment in comment_lines:
                handle.write(f"# {comment}\n")
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                delimiter=delimiter,
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)


def _resolve_project_root_from_expression_matrix(
    expression_matrix: Union[str, Path],
) -> Optional[Path]:
    """Best-effort resolution of the Paralome project root from the RNA
    expression matrix path.

    Assumes the standard sibling stage-workdir layout:
    ``<project>/rna_quantification/results/machine/<matrix>.tsv``, so the
    project root is four levels up. Returns None (rather than raising) if the
    matrix was supplied from outside that layout, since sequencing QC
    cross-referencing is a best-effort convenience, not a hard requirement.
    """
    try:
        return Path(expression_matrix).resolve().parents[3]
    except IndexError:
        return None





#: Where 'rna fpkm-filter-sweep' wrote, inside each statistic's folder, until
#: Removed it. Still cleared when a result is replaced.
SWEEP_DIRNAME = "fpkm_filter_sweep"


def _features(meta: Sequence[Mapping[str, str]]):
    """The genes as ``results.Features``: keyed by gene_id, with the matrix's other columns."""
    from paralome.results import Features

    gene_ids = [str(row.get("gene_id") or "") for row in meta]
    return Features(
        key=gene_ids, key_column="gene_id", gene_id=gene_ids,
        gene_name=[str(row.get("gene_name") or "") for row in meta],
        annotation=[dict(row) for row in meta],
    )


def write_differential_result(
    raw_results: Sequence[ComparisonResult],
    plan: DifferentialPlan,
    *,
    out: Path,
    expression_matrix: Path,
    samples_path: Path,
    sample_rows: Sequence[Mapping[str, str]],
    meta: Sequence[Mapping[str, str]],
    matrix: np.ndarray,
    statistic: str,
    factors: Sequence[str],
    pair_by: Optional[str],
    covariate: Optional[str],
    covariate_feature: Optional[str],
    min_expression: float,
    min_expression_difference: float,
    pseudocount: float,
    alpha_metric: str,
    alpha_value: float,
    file_prefix: Optional[str],
    fingerprint: str,
    request_fingerprint: str,
    superseded: Optional[Mapping[str, object]],
    refine_responders: Optional[int],
    run_qc: bool,
    block_count: int,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Correct, gate, write, checkpoint, audit. THE ONE WRITER.

    ``raw_results`` are the engine's uncorrected results over every gene, in
    the matrix's order -- from the local run's blocks or the SLURM blocks,
    concatenated. Everything from here on is the same for both paths.
    """
    from paralome.atac.naming import RNA_MODALITY, derive_file_prefix
    from paralome.results import Samples, write_all_statistics

    out = Path(out).resolve()
    results_dir = out / "results"
    (results_dir / "machine").mkdir(parents=True, exist_ok=True)
    final_checkpoint = stage_checkpoint_path(out, "differential_expression")
    prefix = file_prefix or derive_file_prefix(out)
    sample_ids = [str(row["sample_id"]).strip() for row in sample_rows]

    results = finalize(list(raw_results), alpha_metric=alpha_metric, alpha_value=alpha_value)
    apply_expression_gate(
        results, min_expression=min_expression, min_expression_difference=min_expression_difference,
    )

    # The result this one replaces goes just before the new one is
    # written -- see remove_previous_outputs.
    if superseded is not None:
        remove_previous_outputs(superseded, out)

    raw_matrix = _raw_fpkm(matrix, pseudocount)
    # the gene-name warning, in the checkpoint and on stderr (the
    # workbook that also opened on it is gone).
    gene_name_coverage = _gene_name_coverage(meta)
    gene_name_warning = gene_name_warning_for(gene_name_coverage)
    statistics = write_all_statistics(
        results,
        root=results_dir,
        file_prefix=prefix,
        stamp=RNA_MODALITY,
        features=_features(meta),
        samples=Samples(ids=sample_ids, values=raw_matrix, unit="raw FPKM"),
        version=__version__,
        prefer_raw_means=True,
        raw_fpkm=True,
        alpha_metric=alpha_metric,
        alpha_value=alpha_value,
        stage="differential_expression",
    )

    if gene_name_warning:
        print(f"[paralome] differential_expression: {gene_name_warning}", file=sys.stderr)

    # Sample QC, PCA and the heatmap are the viewer's, drawn from each statistic folder's sample matrix.
    checkpoint: Dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "differential_expression",
        "complete": True,
        # WHICH PARALOME WROTE THIS.
        "paralome_version": __version__,
        "input_fingerprint": fingerprint,
        "request_fingerprint": request_fingerprint,
        # 0 until the refine below has run and completed -- see
        # record_refined_outputs.
        "refine_responders": 0,
        # The contract every reader of a differential stage reads:
        # the --statistic string as given, the plan, and one manifest per
        # statistic folder.
        "statistic": str(statistic),
        "plan": plan.as_record(),
        "statistics": statistics,
        "file_prefix": prefix,
        "expression_matrix": str(Path(expression_matrix).resolve()),
        "samples": str(Path(samples_path).resolve()),
        "factors": list(factors),
        "factor_levels": {k: list(v) for k, v in plan.levels_by_factor.items()},
        "pair_by": pair_by,
        "covariate": covariate,
        "covariate_feature": covariate_feature,
        "pseudocount": float(pseudocount),
        "sample_count": len(sample_rows),
        "genes_tested": len(meta),
        "min_expression": min_expression,
        "min_expression_difference": min_expression_difference,
        "alpha_metric": alpha_metric,
        "alpha_value": alpha_value,
        "expression_gate_rule": (
            "a gene of a two-arm "
            f"comparison passes only when an arm's raw FPKM mean is >= {min_expression:g} and the "
            f"two raw means differ by >= {min_expression_difference:g} "
            "(passes_expression_gate in each machine table); every table keeps every gene"
        ),
        "gene_name_coverage": gene_name_coverage,
        "gene_name_warning": gene_name_warning,
        "block_count": int(block_count),
        **dict(extra or {}),
    }
    write_json(final_checkpoint, checkpoint, atomic=True)
    # After the checkpoint, and NOT into it. The audit writes its own
    # record under results/qc/.
    if run_qc:
        report = run_qc_beside_the_result(
            out, samples_path, factors=factors,
            min_expression=min_expression,
            refine_responders=refine_responders,
            expression_matrix=expression_matrix,
            statistic=statistic,
            min_expression_difference=min_expression_difference,
            pseudocount=pseudocount,
            alpha_metric=alpha_metric, alpha_value=alpha_value,
            pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
        )
        checkpoint = record_refined_outputs(
            final_checkpoint, checkpoint, report, requested=int(refine_responders or 0),
        )
    return checkpoint


# ---------------------------------------------------------------------------
# 'rna fpkm-filter-sweep'
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _finished_result(
    previous: Dict[str, object],
    checkpoint_path: Path,
    out: Path,
    samples_path: Path,
    expression_matrix: Path,
    *,
    refine: int,
    factors: Sequence[str],
    statistic: str,
    min_expression: float,
    min_expression_difference: float,
    pseudocount: float,
    alpha_metric: Optional[str] = None,
    alpha_value: Optional[float] = None,
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
) -> Dict[str, object]:
    """Hand back a result that is already complete, refining it first if asked.

    --refine-responders is a step AFTER the result, so a finished
    result is no reason to skip it. It is run here once per value and the
    checkpoint remembers which value it was run with.
    """
    say_kept_without_matrix(expression_matrix)
    if refine and previous.get("refine_responders") != refine:
        report = run_qc_beside_the_result(out, samples_path, factors=factors,
                                          min_expression=min_expression,
                                          refine_responders=refine,
                                          expression_matrix=expression_matrix,
                                          statistic=statistic,
                                          min_expression_difference=min_expression_difference,
                                          pseudocount=pseudocount,
                                          alpha_metric=alpha_metric, alpha_value=alpha_value,
                                          pair_by=pair_by, covariate=covariate,
                                          covariate_feature=covariate_feature)
        # Recorded as done only if it was -- see record_refined_outputs.
        previous = record_refined_outputs(checkpoint_path, previous, report, requested=refine)
    return previous


def differential_expression(
    expression_matrix: Path,
    samples_path: Path,
    output_dir: Path,
    *,
    statistic: str = "t",
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    block_size: int = 5000,
    overwrite: bool = False,
    min_expression: float = 0.0,
    min_expression_difference: float = 0.0,
    pseudocount: float = 1.0,
    run_qc: bool = True,
    refine_responders: Optional[int] = None,
    file_prefix: Optional[str] = None,
    alpha_metric: str = "p",
    alpha_value: float = 0.05,
) -> Dict[str, object]:
    """Bulk RNA differential expression, run here: every statistic --statistic names.

    ``file_prefix`` names the result files (default: the --out folder's name,
    as ``atac.naming.derive_file_prefix`` reads it); the refine passes its own.
    """
    check_pseudocount(pseudocount)
    check_request_options(
        statistic=statistic, factors=factors,
        pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
        refine_responders=refine_responders if run_qc else None,
    )
    # The pseudocount the matrix was built with, when its
    # quantification checkpoint is beside it -- see check_pseudocount_matches_matrix.
    check_pseudocount_matches_matrix(expression_matrix, pseudocount)
    options = dict(
        statistic=statistic, factors=factors,
        pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
    )
    # The whole request is checked against the sample sheet HERE,
    # before a folder is made and before an existing result is withdrawn -- a
    # mistyped --factor must not cost the finished result it was aimed at.
    planned = False
    try:
        sample_rows: Optional[List[Dict[str, str]]] = _read_samples(samples_path)
    except (OSError, ValueError):
        sample_rows = None
    if sample_rows is not None:
        plan_bulk_differential(sample_rows, **options)
        planned = True

    out = Path(output_dir).resolve()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    final_checkpoint = stage_checkpoint_path(out, "differential_expression")
    # What this result is computed FROM -- see jsonio.input_fingerprint.
    try:
        if sample_rows is None:
            raise ValueError("the sample sheet could not be read")
        fingerprint = input_fingerprint(hashed=[expression_matrix], content=sample_rows)
    except (OSError, ValueError):
        fingerprint = ""        # unreadable inputs are reported where they are read, below
    request_fingerprint = (
        differential_request_fingerprint(expression_matrix, sample_rows)
        if sample_rows is not None else ""
    )
    superseded: Optional[Dict[str, object]] = None
    withdraw: Optional[Tuple[Dict[str, object], str]] = None
    if final_checkpoint.exists():
        previous = json.loads(final_checkpoint.read_text())
        superseded = previous
        if previous.get("complete"):
            stale_reason = "--overwrite" if overwrite else differential_stale_reason(
                previous,
                min_expression=min_expression,
                min_expression_difference=min_expression_difference,
                pseudocount=pseudocount,
                alpha_metric=alpha_metric,
                alpha_value=alpha_value,
                fingerprint=fingerprint,
                request_fingerprint=request_fingerprint,
                output_dir=out,
                **options,
            )
            if stale_reason is None:
                return _finished_result(
                    previous, final_checkpoint, out, samples_path, expression_matrix,
                    refine=bool(run_qc) and int(refine_responders or 0),
                    factors=factors, statistic=statistic, min_expression=min_expression,
                    min_expression_difference=min_expression_difference, pseudocount=pseudocount,
                    alpha_metric=alpha_metric, alpha_value=alpha_value,
                    pair_by=pair_by, covariate=covariate, covariate_feature=covariate_feature,
                )
            withdraw = (previous, stale_reason)

    if sample_rows is None:
        sample_rows = _read_samples(samples_path)
    if not planned:
        plan_bulk_differential(sample_rows, **options)
    sample_ids = [str(row["sample_id"]).strip() for row in sample_rows]
    meta, matrix = _read_matrix(expression_matrix, sample_ids)
    feature_values = (
        covariate_feature_values(meta, matrix, covariate_feature)
        if covariate_feature and not covariate else None
    )
    plan = plan_bulk_differential(sample_rows, covariate_feature_values=feature_values, **options)
    if withdraw is not None:
        if withdraw[1] != "--overwrite":
            print(
                f"[paralome] differential_expression: {withdraw[1]}; "
                "recomputing instead of reusing the existing checkpoint.",
                file=sys.stderr,
            )
        invalidate_checkpoint(final_checkpoint, *withdraw)

    raw = _raw_fpkm(matrix, pseudocount)
    raw_results = fit_blocks(
        matrix, raw, plan, start=0, stop=len(meta), block_size=block_size,
        alpha=power_alpha(alpha_metric, alpha_value),
    )
    return write_differential_result(
        raw_results, plan,
        out=out, expression_matrix=Path(expression_matrix), samples_path=Path(samples_path),
        sample_rows=sample_rows, meta=meta, matrix=matrix,
        min_expression=min_expression, min_expression_difference=min_expression_difference,
        pseudocount=pseudocount, alpha_metric=alpha_metric, alpha_value=alpha_value,
        file_prefix=file_prefix, fingerprint=fingerprint, request_fingerprint=request_fingerprint,
        superseded=superseded, refine_responders=refine_responders, run_qc=run_qc,
        block_count=math.ceil(len(meta) / max(1, int(block_size))),
        **options,
    )
