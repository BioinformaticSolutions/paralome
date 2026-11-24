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

"""Shared output-file naming helper for ATAC differential/motif modules.

Output files must carry a comparison-unique name. Without one, every
module writes the same file names for every comparison, so only the folder
distinguishes them and files must be renamed by hand once collected.

Every --out directory is a unique "project ID" per comparison by convention,
so the file prefix is derived automatically from that directory's own
basename -- no CLI flag, nothing extra to remember to pass. (Deriving names
from the comparison's levels would collide across two runs of the same
comparison, such as a rerun after a QC fix.)

Every module that writes comparison-scoped output calls
``derive_file_prefix(out_dir)`` once and prepends it (via `prefixed_name`/
`prefixed_path`) to every file it writes -- never to directory names.
Checkpoint files are deliberately NOT prefixed: their paths are load-bearing
constants other modules (state.py's summarizers, the supervisor) expect at
fixed locations.
"""
# The docstring comes first. It sat after the __future__ import, which
# made it a bare expression and left naming.__doc__ empty.
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Optional, Union

_UNSAFE_PREFIX_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


# A pipeline-driven stage's --out is always `<project>/<fixed conventional stage name>` (e.g.
# orchestrator.py's `project / "singlecell_differential"`), so the uniqueness the user chose lives
# in the *project* directory name one level up, not in the stage's own subdirectory. Only a
# standalone --out (`atac differential --out NEG_vs_LOW`) has the unique name as its own basename.
# Walk up past a recognized generic stage dirname so both cases resolve to the name the user picked.
_GENERIC_STAGE_DIRNAMES = frozenset({
    "rna_quantification",
    "differential_expression",
    "macs",
    "merged_peaks",
    "peak_quantification",
    "differential_accessibility",
    "motif_analysis",
    "singlecell_atac",
    "singlecell_differential",
    "singlecell_motif_analysis",
    "pseudobulk",
    "data",
    "extraction",
    "singlecell_extraction",
    "motifs",
    "variant_run",
    "annotation",
    "segregation",
    "filter",
    "candidate_report",
    "singlecell_rna",
    "singlecell_rna_differential",
    "functional_evidence",
    # The DNA gene-burden stage folder names the stage, not the project.
    "gene_burden",
    # The variant browser bundle stage's folder (renamed in place on the next run).
    "browser_bundle",
    # Its folders, and the omicview folder every zip goes in.
    "statistics_output", "genotype_output", "omicview",
    # The genome tracks stage's folder (<project>/genome_tracks; snATAC's <project>/atac/genome_tracks
    #).
    "genome_tracks",
    # The trajectories stage's folder.
    "trajectories",
    # The QC layer, which the PCA's exploratory aggregation nests under.
    "qc",
})

# The modality-layer directory names of the folder layout (project/atac/..., project/rna/...) -- see derive_file_prefix.
_MODALITY_DIRNAMES = frozenset({"atac", "rna"})


def derive_file_prefix(out_dir: Union[str, Path]) -> str:
    """The file-naming prefix for a comparison: the meaningful basename
    nearest to its --out directory.

    E.g. standalone ``--out results/NEG_vs_LOW`` -> prefix ``"NEG_vs_LOW"``,
    so ``t_results.xlsx`` becomes ``NEG_vs_LOW_t_results.xlsx``. But a
    pipeline-driven ``--out`` like ``.../celltype_treated_vs_control/singlecell_differential``
    has a fixed, generic conventional stage-directory name as its own
    basename -- walk up to the project directory ("celltype_treated_vs_control")
    instead, which is where the actual per-comparison uniqueness lives.
    Sanitized the same way sample IDs are (``singlecell.sanitize_sample_id``'s
    charset) so any directory name is safe to reuse as a filename prefix.
    """
    resolved = Path(out_dir).expanduser().resolve()
    path = resolved
    # Climbs ITERATIVELY past every generic layer (as project_identifier() does), so
    # <project>/data/rna resolves to <project>, not "data".
    #
    # Bounded by path depth: it stops at the filesystem root, since the root's
    # own name is "". A project directory named exactly like a stage layer
    # ("data", "rna") is therefore climbed past and takes its PARENT's name.
    generic = _GENERIC_STAGE_DIRNAMES | _MODALITY_DIRNAMES
    while path.name in generic and path.parent.name:
        path = path.parent
    name = path.name
    cleaned = _UNSAFE_PREFIX_CHARS.sub("_", name).strip("_")
    return cleaned or "output"


def prefixed_name(name: str, file_prefix: Optional[str]) -> str:
    """Prepend ``file_prefix`` (if given) to a bare filename.

    E.g. ``prefixed_name("t_results.xlsx", "NEG_vs_LOW")`` ->
    ``"NEG_vs_LOW_t_results.xlsx"``. No-op when ``file_prefix`` is falsy.
    """
    if not file_prefix:
        return name
    return f"{file_prefix}_{name}"


def prefixed_path(path: Union[str, Path], file_prefix: Optional[str]) -> Path:
    """Prepend ``file_prefix`` to a path's filename, leaving its directory untouched."""
    path = Path(path)
    if not file_prefix:
        return path
    return path.with_name(prefixed_name(path.name, file_prefix))


#: The stamp that appears at the END of a result filename. Values, not free
#: strings, so a typo cannot invent a new one.
#:
#: These name the RESULT FAMILY, which for ATAC is finer than the
#: modality: there are three different kinds of ATAC differential result, and
#: each is stamped accordingly. The snATAC graph produces three families of comparable
#: per-stratum results --
#:
#:     singlecell_differential            peak accessibility  -> _atac_peaks
#:     singlecell_motif_analysis_singles  motif enrichment    -> _atac_motifs
#:     singlecell_motif_analysis_pairs    motif-pair          -> _atac_motif_pairs
#:
#: -- and all three derive the SAME file_prefix, exactly as RNA and ATAC did
#:. One `_atac` stamp for all three is the same collision one
#: level down: three files named <project>_<stratum>_<thing>_atac.csv, from
#: three different tests, in one folder.
#:
#: RNA stays a single token because the scRNA graph stops at diffs -- there is
#: only one family to distinguish.
#:
#: The `atac` root is kept as a prefix of each so the modality is still legible
#: at a glance and an `*_atac*` glob still finds every ATAC result.
RNA_MODALITY = "rna"
ATAC_PEAK_MODALITY = "atac_peaks"
ATAC_MOTIF_MODALITY = "atac_motifs"
ATAC_MOTIF_PAIR_MODALITY = "atac_motif_pairs"

#: Retained so a caller that genuinely means "ATAC, family unspecified" still
#: reads clearly. Nothing that writes a per-stratum result should use it -- the
#: three families above exist for that.
ATAC_MODALITY = "atac"


def modality_stamped_name(name: str, modality: Optional[str]) -> str:
    """Append ``_<modality>`` to a filename, before its extension.

    ``("power_summary.csv", "atac")`` -> ``"power_summary_atac.csv"``.

    WHY A STAMP AT ALL. The RNA and ATAC halves of one multiome comparison
    derive the SAME ``file_prefix``: ``derive_file_prefix`` climbs past
    ``singlecell_differential`` AND past ``rna``/``atac``, so both land on
    ``treated_vs_control``, and collecting both into one folder would
    overwrite half the results.

    A trailing stamp is applied per-file rather than carried in the prefix, so
    a writer could forget it -- which is why ``result_path`` below is the only
    sanctioned way to build one of these names.
    """
    if not modality:
        return name
    path = Path(name)
    return f"{path.stem}_{modality}{path.suffix}"


def result_path(
    directory: Union[str, Path],
    name: str,
    *,
    file_prefix: Optional[str],
    modality: Optional[str] = None,
) -> Path:
    """THE way to name a single-cell differential result file.

    ``<directory>/<file_prefix>_<name stem>_<modality><ext>``, e.g.
    ``treated_vs_control_celltypeA_pseudobulk_t_atac.csv``.

    One funnel for both modalities. ``file_prefix`` has usually already been
    through ``run_scoped_prefix`` and so carries the stratum; this adds the
    modality and nothing else. Writers call this rather than composing
    ``prefixed_path`` themselves, so the RNA and ATAC sides cannot drift in
    naming the way they drifted in layout -- which is the whole reason this
    function exists rather than two conventions that happen to agree today.
    """
    return Path(directory) / modality_stamped_name(
        prefixed_name(name, file_prefix), modality,
    )


#: A token no real output name contains, run through the naming functions to
#: fingerprint them. Stem and suffix are both present so a change to EITHER
#: half of the convention (leading prefix, trailing stamp) moves the result.
_NAMING_PROBE = "probe.ext"


def naming_signature(*, file_prefix: Optional[str], modality: Optional[str] = None) -> str:
    """A fingerprint of HOW this stage names files, for the checkpoint hash.

    A stage's ``plan_id`` records the INPUTS that determine its results, and
    reuses the checkpoint when they are unchanged. A change to the naming
    CONVENTION (a stamp moving, a separator changing, a modality added or
    dropped) changes no input, so without this the stage would return its old
    checkpoint instead of writing files under the new names.

    So the hash covers the convention, not just its inputs. This returns the
    name the current code would give a probe file -- e.g.
    ``treated_vs_control_celltypeA_probe_atac.ext`` -- by calling the real
    ``result_path``. Anything that changes how a name is built changes this
    string and busts the checkpoint by construction.

    Deliberately computed rather than a hand-maintained version constant: a
    constant is one more thing to remember to bump.
    """
    return result_path(".", _NAMING_PROBE, file_prefix=file_prefix, modality=modality).name


# --------------------------------------------------------------------------
# The guard rails.
#
# When the prefix is falsy the three functions above return the BARE name and
# carry on. The rule these encode: a stage may not claim success unless the
# files it just wrote actually carry the prefix it claims to have used. Not the
# plan, not the checkpoint, not a helper's return value -- the directory listing.
# --------------------------------------------------------------------------


def require_file_prefix(file_prefix: Optional[str], *, stage: str) -> str:
    """Return ``file_prefix``, or raise if it is missing or unusable.

    ``prefixed_name(name, None)`` silently returns the bare name, so
    a prefix that failed to resolve produced correct-looking code, a clean
    exit, and unprefixed files. Making that an error at the point of
    resolution means a stage cannot get as far as writing.

    ``"output"`` is rejected too: that is ``derive_file_prefix``'s own
    last-resort return when a path has no usable basename, so seeing it here
    means derivation failed rather than found something. A file called
    ``output_pass.tsv`` is exactly the confidently-wrong name the segregation
    docstring warns is worse than an unprefixed one.
    """
    cleaned = (file_prefix or "").strip()
    if not cleaned or cleaned == "output":
        raise ValueError(
            f"{stage}: could not resolve a project/family ID for output filenames "
            f"(got {file_prefix!r}). Pass one explicitly with --prefix, e.g. "
            f"--prefix FAMILY01. Refusing to write unprefixed files: every one of "
            f"them collides with the same file from every other family the "
            f"moment they are collected."
        )
    return cleaned


# Files that are DELIBERATELY not prefixed, checked by name at the top level
# of a stage's results/ directory. Keep this list short and justified -- it is
# the only way a bare name can legitimately survive the self-check.
_UNPREFIXED_BY_DESIGN = frozenset({
    # A provenance record that never leaves the workdir. Renaming it would churn
    # the resume path for no benefit; see segregation/runner.build_segregation_plan.
    "provenance.json",
})

# Directories under results/ whose CONTENTS are intermediates, deleted or
# ignored after the merge and never collected. The directory names themselves
# are not prefixed either -- directories already disambiguate by position,
# which is the distinction this whole module turns on.
_INTERMEDIATE_DIRNAMES = frozenset({
    "shard_fragments",
})


def unprefixed_outputs(
    results_dir: Union[str, Path],
    file_prefix: str,
    *,
    recurse: bool = True,
) -> list[Path]:
    """Every file under ``results_dir`` that does NOT carry ``file_prefix``.

    The listing-based check. Walks the stage's own results
    directory and returns what is wrongly named, so the caller can fail with
    the actual paths rather than a boolean.

    Deliberately checks the EXACT expected prefix, not "does it have a
    prefix". A glob for ``*_pass.tsv`` matches ``segregation_pass.tsv`` and
    ``scen_pass.tsv`` just as happily as ``FAMILY01_pass.tsv``.
    """
    results = Path(results_dir)
    if not results.exists():
        return []
    expected = f"{file_prefix}_"
    offenders: list[Path] = []
    for path in sorted(results.rglob("*") if recurse else results.iterdir()):
        if not path.is_file():
            continue
        if any(part in _INTERMEDIATE_DIRNAMES for part in path.relative_to(results).parts[:-1]):
            continue
        if path.name in _UNPREFIXED_BY_DESIGN:
            continue
        if not path.name.startswith(expected):
            offenders.append(path)
    return offenders


def snapshot_results(results_dir: Union[str, Path]) -> frozenset[str]:
    """The files already present before a stage runs, as relative paths.

    Taken at stage start so the post-write check can tell "this run
    wrote a badly named file" (a bug, fail the stage) from "a previous run
    under an older convention left one here" (history, not this run's fault).

    Without this distinction the check fails every re-run on a tree that has
    ever been run before, which is most real trees.
    """
    results = Path(results_dir)
    if not results.exists():
        return frozenset()
    return frozenset(
        str(p.relative_to(results)) for p in results.rglob("*") if p.is_file()
    )


def assert_outputs_are_prefixed(
    results_dir: Union[str, Path],
    file_prefix: str,
    *,
    stage: str,
    preexisting: Optional[frozenset[str]] = None,
) -> None:
    """Fail the stage if anything IT WROTE is not named ``<prefix>_*``.

    And this is the whole point of the release. Called after the
    writers finish and before the stage reports success, so the stage cannot
    exit 0 having written ``pass.tsv``.

    This is the "run the thing and look at what lands on disk" rule, moved
    out of the test suite and into the product, so it runs on real data, on
    every family, every time -- not only against a toy VCF in the tests.

    ``preexisting`` (from ``snapshot_results`` at stage start) scopes the
    failure to files this run created. Older wrongly-named files are reported
    as a warning instead: they are stale copies, they are not what the stage
    just produced, and failing on them would make every re-run on an
    already-used workdir impossible. They are still worth saying out loud,
    because a directory holding both ``pass.tsv`` and ``FAMILY01_pass.tsv`` is
    how somebody eventually emails the wrong one.
    """
    results = Path(results_dir)
    offenders = unprefixed_outputs(results_dir, file_prefix)
    if not offenders:
        return

    known = preexisting or frozenset()
    written_now = [p for p in offenders if str(p.relative_to(results)) not in known]
    leftovers = [p for p in offenders if str(p.relative_to(results)) in known]

    if leftovers:
        print(
            f"[paralome] {stage}: WARNING -- {len(leftovers)} file(s) in {results} "
            f"predate this run and do not carry the project ID {file_prefix!r}, "
            f"e.g. {', '.join(p.name for p in leftovers[:3])}. They are stale "
            f"copies from an older naming convention; the current outputs are "
            f"the {file_prefix}_* files beside them. Delete them when convenient.",
            flush=True,
        )

    if not written_now:
        return

    listing = "\n".join(f"    {p.relative_to(results)}" for p in written_now[:20])
    more = f"\n    ... and {len(written_now) - 20} more" if len(written_now) > 20 else ""
    raise RuntimeError(
        f"{stage}: wrote {len(written_now)} file(s) that do not carry the project "
        f"ID {file_prefix!r}:\n{listing}{more}\n"
        f"  Expected every file this stage writes into {results} to be named "
        f"{file_prefix}_*.\n"
        f"  The stage is being failed rather than reporting success, because an "
        f"unprefixed result is indistinguishable from every other family's copy "
        f"of the same file once collected."
    )


def assert_results_are_stamped(
    outputs: object,
    *,
    modality: str,
    stage: str,
) -> None:
    """Fail the stage if anything it just wrote lacks its ``_<modality>`` stamp.

    ``assert_outputs_are_prefixed`` above is the listing-based check:
    it walks a stage's ``results/`` directory. That works for the variant
    pipeline, whose results directory holds nothing but results. A single-cell
    differential ``--out`` also holds ``checkpoints/``, ``logs/``, ``scripts/``
    and per-stratum subdirectories, so a listing walk there would flag files
    the stage never claimed to name and would be switched off within a day --
    which is worse than not having it.

    So this checks the paths the stage DECLARES it wrote, taken from the same
    dict it is about to return and record in its checkpoint. That has no
    false-positive surface at all: every entry is by definition something the
    stage takes credit for. The trade is that a file written but never declared
    escapes the check -- acceptable, because an undeclared output is invisible
    to the collect tooling anyway.

    Accepts the nested ``{model: {"table": path}}`` / ``{name: path}`` /
    ``[path]`` shapes the two report writers actually return, rather than
    demanding callers flatten first -- a guard that is awkward to call is a
    guard that does not get called.

    EXTENSIONLESS VALUES ARE SKIPPED, and that is load-bearing rather than
    laziness. Both report dicts carry ``results_dir`` -- a directory, which no
    naming convention stamps -- alongside the files. A first cut of this guard
    walked it too and would have failed every ATAC run on its own output
    directory. A guard whose first live encounter is a false alarm on correct
    data is worse than no guard: it gets switched off, and then it is not there
    for the real one. Every result file this stage writes has an extension, so
    requiring one costs nothing and removes the whole class.
    """
    offenders: list[str] = []

    def _visit(value: object) -> None:
        if value is None or isinstance(value, (bool, int, float)):
            return
        if isinstance(value, (str, Path)):
            candidate = Path(str(value))
            name = candidate.name
            if not name or not candidate.suffix:
                return
            if candidate.is_dir():
                return
            if candidate.stem.endswith(f"_{modality}"):
                return
            offenders.append(str(value))
            return
        if isinstance(value, dict):
            for item in value.values():
                _visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                _visit(item)

    _visit(outputs)
    if not offenders:
        return

    listing = "\n".join(f"    {path}" for path in offenders[:20])
    more = f"\n    ... and {len(offenders) - 20} more" if len(offenders) > 20 else ""
    raise RuntimeError(
        f"{stage}: wrote {len(offenders)} result file(s) with no {modality!r} "
        f"modality stamp:\n{listing}{more}\n"
        f"  Every result filename must end in _{modality} before its extension "
        f"(build it with naming.result_path).\n"
        f"  The stage is being failed rather than reporting success, because the "
        f"RNA and ATAC halves of one comparison derive the SAME file_prefix -- an "
        f"unstamped file silently overwrites its counterpart the moment the two "
        f"modalities are collected into one folder."
    )


def stage_banner(stage: str, file_prefix: str, workdir: Union[str, Path]) -> str:
    """One line identifying what is actually executing, printed at stage start.

    Without this, a run against a stale installed version cannot
    be told apart from the output. A stage that
    announces its version, its resolved prefix and its workdir turns "which
    code is this and what will it name things" into something a log answers
    by itself, instead of a round-trip.
    """
    from paralome.version import __version__

    return (
        f"[paralome {__version__}] {stage}: prefix={file_prefix} "
        f"workdir={Path(workdir)}"
    )


_RUN_STAMP = re.compile(r"^\d{4}-(\d{2})-(\d{2})T(\d{2})(\d{2})")


def short_run_stamp(run_id: Optional[str]) -> str:
    """``MMDD_HHMM`` from a QC ``run_id``, or ``""`` if it doesn't look like one.

    ``run_id`` is ``<timestamp>_<modality>`` with the timestamp in
    ``%Y-%m-%dT%H%M%S`` form, e.g. ``2026-08-02T074500_rna`` -> ``0802_0745``.
    Seconds and the year are dropped deliberately: the point is to tell a
    morning run from an afternoon one at a glance in a filename, not to be a
    globally unique key -- the run DIRECTORY already carries the full
    second-resolution stamp and its collision suffix. Two runs started inside
    the same minute still produce the same file names; they remain
    distinguishable by folder.

    Returns "" rather than raising on an unrecognised run_id, so a caller that
    is handed something unexpected degrades to the plain naming instead of
    failing a plot that is otherwise fine.
    """
    if not run_id:
        return ""
    match = _RUN_STAMP.match(str(run_id))
    if not match:
        return ""
    month, day, hour, minute = match.groups()
    return f"{month}{day}_{hour}{minute}"


def run_scoped_prefix(
    file_prefix: Optional[str],
    run_id: Optional[str] = None,
    stratum: Optional[str] = None,
) -> str:
    """``<project>_<MMDD_HHMM>_<stratum>``, skipping any part that is absent.

    So a QC file still says which run and which cell type it belongs to once
    it leaves its folder (with ``--stratify-by`` every cell type would
    otherwise write the same names). Composed from the parts that already
    exist rather than a new CLI option: the project ID already carries that
    information.
    """
    parts = [part for part in (file_prefix, short_run_stamp(run_id), stratum) if part]
    return "_".join(parts)


def resolve_prefixed_file(directory: Union[str, Path], bare_name: str) -> Optional[Path]:
    """Find a file that may or may not have had a file prefix applied.

    Cross-module readers (e.g. the differential and motif analysis finding the
    peak-counting stage's ``peak_counting_summary.csv``) can't know another module's derived
    ``file_prefix`` in advance -- it's derived from that OTHER module's own
    ``--out`` directory basename, not the reader's. The candidates are the
    exact unprefixed name (runs from before file prefixes, or a directory whose
    basename happens to equal ``bare_name`` already) and every
    ``"*_<bare_name>"`` match in ``directory``; the most recently modified
    one is returned, and when more than one candidate exists that is said on
    stderr, because a folder holding two tables of the same kind is how the
    wrong one gets read.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = [path for path in [directory / bare_name, *directory.glob(f"*_{bare_name}")] if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        print(
            f"[paralome] {directory} holds {len(candidates)} files named like {bare_name} "
            f"({', '.join(path.name for path in candidates)}); reading the newest, {candidates[0].name}. "
            f"The others were left by an earlier run.",
            file=sys.stderr, flush=True,
        )
    return candidates[0]


def sample_file_token(sample_id: str) -> str:
    """A filesystem-safe, collision-resistant token for a sample id.

    One rule for ``atac/peak_merge.py`` and ``atac/quantification.py`` -- the
    two stages that write one file per sample and must therefore agree,
    exactly, on what that file is called.

    Ids that are already safe pass through unchanged, so existing per-sample
    files keep their names; anything else is normalised and suffixed with a
    sha256 prefix so two different ids cannot collapse onto one filename.
    """
    if sample_id in {"", ".", ".."}:
        raise ValueError("sample_id must not be empty, '.' or '..'")
    if "/" not in sample_id and "\\" not in sample_id and re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id):
        return sample_id
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._-") or "sample"
    return f"{normalized}_{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:10]}"
