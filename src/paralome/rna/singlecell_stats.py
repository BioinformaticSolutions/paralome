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

from __future__ import annotations

"""Single-cell RNA differential expression: the scRNA driver.

The twin of ``paralome.atac.singlecell_stats.run_singlecell_differential``
for genes, over the same shared body (``singlecell.stats_core.compute_statistics``)
and the same cell-level fits (``atac.singlecell_stats.fit_cell_plan``). Each statistic --statistic names writes its own folder through
``paralome.results`` -- the writer bulk RNA uses -- so a pseudobulk folder reads
exactly as a bulk one. What differs from snATAC is only the feature identity
(the key is the feature name; gene_id and common_id are the gene the
annotation placed it on) and that there is no PCA here.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

from paralome.atac.naming import (
    RNA_MODALITY,
    assert_results_are_stamped,
    derive_file_prefix,
    naming_signature,
    run_scoped_prefix,
    stage_banner,
)
from paralome.checkpoints import stage_checkpoint_path
from paralome.rna.singlecell import load_singlecell_rna_inputs
from paralome.singlecell.strata import (
    keep_for_submission,
    resolve_one_stratum,
    run_stratified_fan_out,
    split_stratify_values,
    stamp_submission,
    stratum_slugs,
    withdraw_checkpoint,
)
from paralome.singlecell.stats_core import (
    DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE,
    SC_DIFF_METHOD,
    compute_statistics,
    gene_features,
    input_file_identity,
    load_pseudobulk_comparison,
    parse_request,
    stamped_files,
    statistics_on_disk,
)
from paralome.stats import DATA_LEVEL_CELL

# The KEY is feature_name; gene_id and common_id are annotation carried beside
# it. gene_id is "." for every gene the annotation could not place, and a key
# that repeats does not raise, it merges rows -- so the roles stay split.
FEATURE_ID_COLUMN = "feature_name"
CORE_ANNOTATION_COLUMNS = ("gene_id", "common_id")

#: Every key in this stage's checkpoint that maps to the files it wrote (see
#: the ATAC driver's note).
_CHECKPOINT_OUTPUT_KEYS = ("statistics",)


#: (path, size, mtime_ns) -> sha256, so a fan-out over twenty strata hashes the
#: pseudobulk matrices once rather than once per stratum.
_SHA_MEMO: Dict[tuple, str] = {}


def _file_sha256(path: Path) -> str:
    import hashlib

    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA_MEMO:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        _SHA_MEMO[key] = digest.hexdigest()
    return _SHA_MEMO[key]


def refuse_unvouched_pseudobulk(
    cpm_matrix: Union[str, Path], sample_manifest: Union[str, Path],
) -> None:
    """Refuse a CPM matrix / manifest its own pseudobulk checkpoint does not vouch for.

    This command reads rna/cpm.csv and sample_manifest.csv directly. The
    aggregation marks its checkpoint incomplete before it rewrites anything
    and records the SHA-256 of each file it finishes; this holds the
    differential to that record, so a truncated cpm.csv, or a cpm.csv and a
    manifest from two different plans, is refused.

    Matrices with no pseudobulk checkpoint beside them (supplied from
    elsewhere) are taken as given -- there is nothing to hold them to.
    """
    cpm = Path(cpm_matrix).expanduser().resolve()
    manifest = Path(sample_manifest).expanduser().resolve()
    checkpoint = stage_checkpoint_path(cpm.parent.parent, "singlecell_rna")
    if not checkpoint.is_file():
        return
    try:
        recorded = json.loads(checkpoint.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"The pseudobulk checkpoint beside {cpm} cannot be read ({error}), so "
            f"nothing says {cpm.name} is a finished aggregation.\n"
            f"Re-run `paralome scRNA samples` against the same project (add "
            f"--overwrite if it reports the stage complete)."
        ) from error
    try:
        governs = Path(str(recorded.get("cpm", ""))).expanduser().resolve() == cpm
    except OSError:
        governs = False
    if not governs:
        return
    if not recorded.get("complete"):
        reason = recorded.get("invalidated") or "it did not finish"
        raise RuntimeError(
            f"{cpm} is not a finished pseudobulk aggregation: its checkpoint "
            f"({checkpoint}) is marked incomplete ({reason}). The matrices may be "
            f"part old, part new, or cut short.\n"
            f"Re-run `paralome scRNA samples` (or the pipeline run that made it) to "
            f"finish the aggregation, then run this again."
        )
    shas = recorded.get("output_sha256")
    if not isinstance(shas, dict):
        return
    checks = [("cpm", cpm)]
    try:
        if Path(str(recorded.get("sample_manifest", ""))).expanduser().resolve() == manifest:
            checks.append(("sample_manifest", manifest))
    except OSError:
        pass
    for key, path in checks:
        expected = shas.get(key)
        if not expected:
            continue
        actual = _file_sha256(path) if path.is_file() else "missing"
        if actual != expected:
            raise RuntimeError(
                f"{path} is not the file its pseudobulk aggregation wrote: its "
                f"checkpoint ({checkpoint}) records a different SHA-256. It was "
                f"changed afterwards, or a re-aggregation was interrupted.\n"
                f"Re-run `paralome scRNA samples --overwrite` against the same "
                f"project, then run this again."
            )


def run_singlecell_rna_differential(
    cpm_matrix: Union[str, Path],
    sample_manifest: Union[str, Path],
    extraction_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    statistic: str = "t",
    stratify_by: Optional[str] = None,
    stratify_value: Optional[str] = None,
    stratify_skip: Optional[Sequence[str]] = None,
    # The CLI passes it unconditionally; the callee's signature must take it.
    min_cells_per_stratum: int = 0,
    factors: Sequence[str] = (),
    pair_by: Optional[str] = None,
    covariate: Optional[str] = None,
    covariate_feature: Optional[str] = None,
    naive_cell_t_peak_chunk_size: int = DEFAULT_NAIVE_CELL_T_PEAK_CHUNK_SIZE,
    alpha: float = 0.05,
    alpha_metric: str = "p",
    pseudocount: float = 1.0,
    min_mean_cpm: float = 1.0,
    overwrite: bool = False,
    jobs: int = 1,
    file_prefix: Optional[str] = None,
    load_inputs: Optional[Callable[[], object]] = None,
) -> Dict[str, object]:
    """Run every statistic --statistic asks for on one gene-level comparison.

    ``load_inputs``: the cells a ``:cell`` statistic is fitted on, when they are not the
    extraction's as it stands -- the trajectories stage's, each cell once per lineage it is on, with its segment.

    ``cpm_matrix``/``sample_manifest`` are the RNA pseudobulk outputs;
    ``extraction_dir`` the shared extraction (read only for ``:cell``). Each
    statistic writes its own folder under ``out_dir``.
    """
    alpha_metric = alpha_metric.strip().lower()
    if alpha_metric not in {"p", "q"}:
        raise ValueError("alpha_metric must be 'p' or 'q'")
    from paralome.singlecell.stats_core import refuse_a_factor_aggregation_dropped

    refuse_a_factor_aggregation_dropped(sample_manifest, extraction_dir, factors)
    request = parse_request(
        statistic, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature,
    )
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A SLURM array task runs one stratum with --stratify-value and has
    # no flag for file_prefix; derived through the SAME run_scoped_prefix the
    # in-process fan-out uses. The same value is the checkpoint variant.
    _single = split_stratify_values(stratify_value) if stratify_by else []
    if file_prefix is None and len(_single) == 1:
        file_prefix = run_scoped_prefix(
            derive_file_prefix(output_dir), None,
            stratum_slugs(_single)[_single[0]],
        )
    checkpoint_variant = file_prefix
    file_prefix = file_prefix or derive_file_prefix(output_dir)
    checkpoint_path = stage_checkpoint_path(
        output_dir, "singlecell_rna_differential", variant=checkpoint_variant,
    )
    # One write per line, so strata fanned out in threads do not run their lines together.
    print(stage_banner("singlecell_rna_differential", file_prefix, output_dir) + "\n", end="", flush=True)

    # Before anything is computed from them, on every path.
    refuse_unvouched_pseudobulk(cpm_matrix, sample_manifest)

    stratify_values = split_stratify_values(stratify_value)
    if stratify_by and len(stratify_values) != 1:
        return run_stratified_fan_out(
            stage="singlecell_rna_differential",
            file_prefix=file_prefix,
            sample_manifest=sample_manifest, output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            stratify_by=stratify_by, overwrite=overwrite,
            stratify_only=stratify_values or None, stratify_skip=stratify_skip,
            min_cells_per_stratum=min_cells_per_stratum,
            statistic=statistic, factors=factors, pair_by=pair_by,
            covariate=covariate, covariate_feature=covariate_feature,
            # The CPU budget for the whole fan-out.
            jobs=jobs,
            # The second implementation of this signature's contract: every parameter above must be passed on
            # here.
            run_one=lambda value, slug, results_dir, _per_stratum_jobs: run_singlecell_rna_differential(
                cpm_matrix, sample_manifest, extraction_dir, results_dir,
                statistic=statistic,
                stratify_by=stratify_by, stratify_value=value,
                factors=factors, pair_by=pair_by, covariate=covariate,
                covariate_feature=covariate_feature,
                naive_cell_t_peak_chunk_size=naive_cell_t_peak_chunk_size,
                alpha=alpha, alpha_metric=alpha_metric,
                pseudocount=pseudocount, min_mean_cpm=min_mean_cpm,
                # The per-stratum share of the CPU budget.
                jobs=_per_stratum_jobs,
                # <comparison>_<stratum>, so files collected from many
                # comparisons into one folder stay apart.
                overwrite=overwrite, file_prefix=run_scoped_prefix(file_prefix, None, slug),
                load_inputs=load_inputs,
            ),
        )

    # The slug spelling becomes the raw metadata spelling ('CD8 TEM' vs CD8_TEM).
    if stratify_by and stratify_values:
        stratify_value = resolve_one_stratum(sample_manifest, stratify_by, stratify_values[0])

    plan_id_payload = {
        "cpm_matrix": str(Path(cpm_matrix).expanduser().resolve()),
        "sample_manifest": str(Path(sample_manifest).expanduser().resolve()),
        "extraction_dir": str(Path(extraction_dir).expanduser().resolve()),
        # What the files hold, not only where they are.
        "inputs_identity": [input_file_identity(cpm_matrix), input_file_identity(sample_manifest)],
        "stratify_by": stratify_by, "stratify_value": stratify_value,
        "statistic": statistic, "factors": list(factors), "pair_by": pair_by,
        "covariate": covariate, "covariate_feature": covariate_feature,
        "alpha": alpha, "alpha_metric": alpha_metric,
        "pseudocount": pseudocount if request.has_cell else None,
        "min_mean_cpm": min_mean_cpm,
        # The prefix and the naming convention name every
        # output file, so a different one is a different result.
        "file_prefix": file_prefix,
        "naming": naming_signature(file_prefix=file_prefix, modality=RNA_MODALITY),
        # Which computation produced the result.
        "method": SC_DIFF_METHOD,
    }
    import hashlib
    plan_id = hashlib.sha256(json.dumps(plan_id_payload, sort_keys=True, default=str).encode()).hexdigest()

    existing: Optional[Dict[str, object]] = None
    if checkpoint_path.is_file():
        try:
            existing = json.loads(checkpoint_path.read_text())
        except (OSError, ValueError):
            existing = None
    if not overwrite and existing is not None:
        if (
            existing.get("complete")
            and existing.get("plan_id") == plan_id
            and all(key in existing for key in _CHECKPOINT_OUTPUT_KEYS)
            and statistics_on_disk(existing.get("statistics"))
        ):
            # A reusing array task still records this submission.
            return keep_for_submission(existing, checkpoint_path, _write_checkpoint)
    # Withdrawn first so a kill part way cannot leave it vouching
    # for a mixture of two runs.
    withdraw_checkpoint(checkpoint_path, "being recomputed", _write_checkpoint)

    comparison = load_pseudobulk_comparison(
        cpm_matrix, sample_manifest,
        stratify_by=stratify_by, stratify_value=stratify_value,
        feature_id_column=FEATURE_ID_COLUMN,
    )
    from paralome.atac.singlecell_stats import make_cell_fitter
    from paralome.version import __version__

    cell_report: Dict[str, object] = {}
    fit_cells = make_cell_fitter(
        load_inputs=load_inputs or (lambda: load_singlecell_rna_inputs(extraction_dir)),
        comparison=comparison, sample_manifest=sample_manifest,
        stratify_by=stratify_by, stratify_value=stratify_value,
        extraction_feature_ids=lambda inputs: [gene.feature_name for gene in inputs.genes],
        min_mean_cpm=min_mean_cpm, alpha=alpha, pseudocount=pseudocount,
        peak_chunk_size=naive_cell_t_peak_chunk_size,
        jobs=jobs, covariate_feature=covariate_feature,
        report=cell_report,
    )
    computed = compute_statistics(
        comparison, request,
        statistic_text=statistic, factors=factors, pair_by=pair_by,
        covariate=covariate, covariate_feature=covariate_feature,
        min_mean_cpm=min_mean_cpm, alpha_metric=alpha_metric, alpha_value=alpha,
        output_dir=output_dir, file_prefix=file_prefix, stamp=RNA_MODALITY,
        features=gene_features(comparison, FEATURE_ID_COLUMN, *CORE_ANNOTATION_COLUMNS),
        version=__version__, stage="scRNA differential",
        one_stratum_of_many=bool(stratify_by and stratify_value is not None),
        fit_cells=fit_cells, jobs=jobs,
    )

    result: Dict[str, object] = {
        "schema": "paralome.rna.singlecell_differential.v2",
        "stage": "singlecell_rna_differential",
        "complete": True,
        "plan_id": plan_id,
        # Read by pipeline.state, so a pre-bump result is not complete.
        "method": SC_DIFF_METHOD,
        "stratify_by": stratify_by,
        "stratify_value": stratify_value,
        **computed,
        "cell_level": cell_report if request.has_cell else None,
    }
    # Both halves of one multiome comparison derive the same
    # file_prefix, so every declared file must carry the modality stamp.
    assert_results_are_stamped(
        {"statistics": stamped_files(computed["statistics"]),
         "statistic_comparison": computed.get("statistic_comparison")},
        modality=RNA_MODALITY, stage="singlecell_rna_differential",
    )
    # Stamped with the array submission it was computed for.
    _write_checkpoint(checkpoint_path, stamp_submission(result))
    result["checkpoint"] = str(checkpoint_path)
    return result


def _write_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    """The leaf checkpoint writer, shared by a fresh run, a reuse re-stamp and
    a withdrawal so all three produce the same file shape."""
    from paralome.pipeline.jsonio import write_json

    write_json(Path(path), payload, atomic=True, default=str)
