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

"""Motif-partner shift: how one motif's co-occurring partners change across comparisons.

A secondary analysis, in the sense `pca` is secondary to a differential:
it runs after the primary ATAC analyses are finished and evaluated, reads their
written outputs, and belongs to no stage graph. Nothing schedules it; the user
asks for it once there is something worth asking about.

WHAT IT ANSWERS
---------------
Whether a shift in one motif's potential binding partners between successive
states of a study is real. Every comparison in a project scans the SAME cohort
peak atlas with the SAME motif library, so each pair's co-occurring peak set is
fixed -- identical across comparisons. A motif cannot move and a peak cannot
widen between comparisons, because there is one of each. The only quantity
that varies is which of those fixed peaks landed in a given comparison's
significant bucket.

So a "partner shift" is, necessarily and only, a change in which peaks are
significant, and it is testable: this module reports what FRACTION of each
pair's fixed peak set a comparison captured, against the rate at which that
comparison opened peaks generally. Indiscriminate opening gives a ratio near 1;
a ratio well above it in a compartment whose overall opening barely moved is
biology rather than an artefact of one comparison simply having more
significant peaks.

WHY IT SPANS COMPARISONS
------------------------
Every other stage works inside one project directory. This one cannot: the
states being compared are separate runs (`StateA_vs_CTL`, `StateB_vs_CTL`,
`StateC_vs_CTL`), each with its own complete motif analysis, sitting side by
side under one root. `--project` names that root -- the same sense the bulk
pipeline uses it in, the directory its sub-analyses are created under -- and
`--comparisons` names the contained comparisons, in order.

WHAT IT GUARDS AGAINST
----------------------
1. UNCORRECTED p. The truncated `motif_pair_p_0.05` convenience files are cut
   on raw p across ~285,000 tested pairs. At that threshold most strata sit at
   or below the null expectation, so lists drawn from it turn over almost
   completely between states -- and near-total turnover is exactly what noise
   produces. This module defaults to q and requires an explicit opt-out.

2. FOREGROUND-SIZE CONFOUNDING. Comparisons differ several-fold in how many
   peaks they call significant. More foreground
   means more power, so "more partners here" is not by itself a finding. The
   selectivity ratio is scale-free and the foreground size is written into
   every output row so the confound is visible rather than implicit.

3. WINNER'S CURSE. A pair selected because it is significant in state B has an
   inflated effect size in B by construction. The wide table therefore carries
   each pair's effect size in EVERY state, not only the one that selected it,
   so the non-selected states can be read directly.

RECURRENCE IS EVIDENCE, NOT A THRESHOLD
---------------------------------------
How many strata a pair clears is the strongest signal here -- independent noise
clears three of nine about 0.8% of the time, while a real effect driven by
shared biology clears most of them. But it is REPORTED, never filtered on.

A minimum recurrence is not a valid question. A project fans out
over anywhere from one stratum to a dozen, and there is no minimal number of
them -- it depends on the project; a fixed minimum would make a project with
one or two strata match nothing and report zero pairs without saying why.

So a pair is present in a comparison when it clears the gate in at least one of
that comparison's strata, and every row carries the per-comparison numbers
and `<comparison>_strata_total` alongside their ratio. Nine of nine and one of
nine are both reported, both visible, and the reader decides which is worth
anything -- which is the only place that decision can correctly be made, since
it depends on how many strata the project has.
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import json
import math
import re
import statistics
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from paralome.atac.naming import (
    ATAC_MOTIF_PAIR_MODALITY,
    derive_file_prefix,
    result_path,
    stage_banner,
)
from paralome.checkpoints import stage_checkpoint_path
from paralome.pipeline.jsonio import write_json
from paralome.stats import benjamini_hochberg, inverse_variance_pooled

#: The per-stratum pair tables this analysis consumes, written by `atac motifs`.
_PAIR_TABLE_GLOB = "*motif_pair_statistics*.csv"

#: Where a comparison keeps its motif results, relative to the comparison root.
#: Both layouts are tried: the snATAC pipeline nests under `atac/`, a bulk
#: standalone run does not.
_MOTIF_RESULT_DIRS = (
    Path("atac") / "motif_analysis" / "results",
    Path("motif_analysis") / "results",
)

#: Columns every pair table must carry. Named explicitly so a schema drift
#: fails loudly here rather than producing a silently empty result.
_REQUIRED_COLUMNS = (
    "pair_id", "motif_a_name", "motif_b_name",
    "cooccurring_peak_count", "global_peaks",
)

#: Both directions, always. Accessibility, like expression, can be gained or
#: lost; a differential has two directions by nature, so offering a choice
#: would only let someone read half a result. The up_* and down_* blocks are in
#: the same row of the same file, so reading both costs no I/O.
DIRECTIONS = ("up", "down")

#: Direction-specific columns, formatted with each direction.
_DIRECTION_COLUMNS = (
    "{d}_foreground_peaks", "{d}_foreground_hits", "{d}_specificity_ratio",
    "{d}_fisher_p", "{d}_fisher_q", "{d}_fc_median", "{d}_fc_bucket_median",
)


#: The method version of this analysis's outputs. It is part of
#: every output's fingerprint (see _OUTPUT_MANIFEST), so --no-overwrite never
#: reuses a table an earlier method wrote. v3: a change_q/change_p of exactly
#: 0 is significant (it was read as 1); the gene annotation reads one
#: differential table per stratum and applies the statistics' top-N
#: co-occurrence rule; the locus export ranks distinct motifs, not hit rows.
METHOD = "paralome.atac.motif_partner_shift.v3"

#: Per-output completion record, in --out/checkpoints. An output is
#: reused under --no-overwrite only when this records it as finished, with the
#: same size, by the same method, inputs and settings.
_OUTPUT_MANIFEST = "motif_partner_outputs.json"


def _file_identity(path: Optional[Path]) -> List[object]:
    if path is None:
        return []
    try:
        st = Path(path).stat()
    except OSError:
        return [str(path), None, None]
    return [str(path), int(st.st_size), int(st.st_mtime_ns)]


def _fingerprint(payload: Mapping[str, object]) -> str:
    import hashlib

    rendered = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


#: How many changed pairs get detailed annotation -- loci AND genes -- unless
#: the user says otherwise. The default is something small that does
#: not take long to run; the user then decides how much to add -- a bit more,
#: or all of it. Annotating every pair that clears the cut can run to tens of
#: millions of gene rows, over 10 GB, and close to 100 GB of RAM, so the run
#: says what it left out and how to get more, and --top-pairs 0 means all.
_DEFAULT_TOP_PAIRS = 25

#: How long a phase may stay silent before it repeats itself. A
#: stage that runs for a long time repeats its last line with a fresh time
#: stamp, so the user can know it is still running, just taking a long time.
#: Five minutes: over a nine-hour phase that is ~110 lines, readable in a
#: SLURM log.
_HEARTBEAT_SECONDS = 300.0


def _read_progress(handle, path: Path, rows_seen: int, elapsed: float) -> str:
    """" ~52% of ~27,100,000 rows, ~4.1 min left" from the byte position.

    A projection of how many rows there should be, so the user can
    know where they are in the progress.

    The row count is not knowable without reading the file, which is the thing
    taking the time -- but the SIZE is known, and the bytes consumed so far
    give both the fraction done and, divided into the rows seen, a projected
    total. It is an estimate and reads as one: rows vary in length, so the
    projection drifts a little as it goes and lands on the truth.

    Returns "" rather than guessing when the position is unavailable -- a
    silent omission beats a confident wrong number.
    """
    try:
        position = handle.tell()
        total_bytes = path.stat().st_size
    except (OSError, ValueError, AttributeError):
        return ""
    if position <= 0 or total_bytes <= 0 or rows_seen <= 0:
        return ""
    fraction = min(1.0, position / total_bytes)
    if fraction <= 0:
        return ""
    projected = int(rows_seen / fraction)
    remaining = (elapsed / fraction) - elapsed if fraction < 1.0 else 0.0
    if remaining >= 90:
        left = f", ~{remaining / 60:.0f} min left"
    elif remaining >= 1:
        left = f", ~{remaining:.0f}s left"
    else:
        left = ""
    return f" (~{fraction * 100:.0f}% of ~{projected:,} rows{left})"


class _Heartbeat:
    """Repeats the last progress line when a phase runs long and quiet.

    Deliberately NOT a new progress metric. A percentage needs a denominator
    the loop often does not have, and inventing one produces the estimate that
    is wrong rather than the fact that is useful. Re-emitting the last real
    line with a fresh elapsed time says exactly one thing -- still alive, still
    here -- which is the thing being asked.

    ``tick()`` is a float comparison, but it is called from loops running to
    hundreds of millions of iterations, so callers should tick every few
    thousand rather than every row.
    """

    def __init__(self, started: float, interval: float = _HEARTBEAT_SECONDS) -> None:
        self._started = started
        self._interval = interval
        self._message = ""
        self._last = time.monotonic()

    def say(self, message: str) -> None:
        self._message = message
        self._last = time.monotonic()
        print(f"{message} ({self._last - self._started:.1f}s)", flush=True)

    def tick(self) -> None:
        if not self._message:
            return
        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now
        print(
            f"{self._message} -- still running ({now - self._started:.1f}s)",
            flush=True,
        )


@contextmanager
def _output(path: Path, skip: bool):
    """The real file, or a sink when --no-overwrite says it already exists.

    Writing to os.devnull rather than branching at every call site keeps the
    building and the writing of each table in one place -- the rows are still
    computed, because later tables are derived from them, only the bytes are
    discarded.
    """
    if skip:
        with open(os.devnull, "w", newline="") as sink:
            yield sink
    else:
        # Written to <name>.partial and renamed on success, so a
        # killed run cannot leave a truncated table under the final name.
        with _atomic_open(path) as handle:
            yield handle


class MotifPartnerError(RuntimeError):
    """Input the user can fix: a missing comparison, an unreadable table."""


@dataclass
class PairObservation:
    """One pair's numbers in one stratum of one comparison."""

    stratum: str
    specificity: float
    fisher_p: float
    fisher_q: float
    foreground_hits: int
    foreground_peaks: int
    cooccurring_peaks: int
    global_peaks: int
    fc_median: float
    fc_bucket_median: float

    @property
    def capture_fraction(self) -> float:
        """Fraction of this pair's FIXED co-occurring peak set that this
        comparison called significant. The numerator varies by comparison;
        the denominator cannot."""
        return self.foreground_hits / self.cooccurring_peaks if self.cooccurring_peaks else 0.0

    @property
    def background_fraction(self) -> float:
        """Fraction of ALL peaks this comparison called significant -- the
        rate against which capture has to be judged."""
        return self.foreground_peaks / self.global_peaks if self.global_peaks else 0.0

    @property
    def selectivity(self) -> float:
        """capture / background. 1.0 means this pair's peaks opened at exactly
        the rate peaks opened generally, i.e. nothing specific happened."""
        base = self.background_fraction
        return (self.capture_fraction / base) if base > 0 else 0.0


@dataclass
class PairRecord:
    """One pair, accumulated across every stratum of every comparison."""

    pair_id: str
    motif_a: str
    motif_b: str
    #: A pair is tracked once PER DIRECTION. The same pair can be
    #: acquired among peaks gaining accessibility and lost among those losing
    #: it -- two different claims about two disjoint peak buckets. Merging them
    #: into one row would produce a classification that is true of neither.
    direction: str = "up"
    cooccurring_peaks: int = 0
    observations: Dict[str, List[PairObservation]] = field(default_factory=dict)
    significant: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, comparison: str, obs: PairObservation, is_significant: bool) -> None:
        self.observations.setdefault(comparison, []).append(obs)
        self.significant.setdefault(comparison, [])
        if is_significant:
            self.significant[comparison].append(obs.stratum)

    def recurrence(self, comparison: str) -> int:
        return len(self.significant.get(comparison, ()))

    def _median(self, comparison: str, attr: str) -> Optional[float]:
        values = [getattr(o, attr) for o in self.observations.get(comparison, ())]
        values = [v for v in values if v == v]  # drop NaN
        return statistics.median(values) if values else None

    def median_specificity(self, comparison: str) -> Optional[float]:
        return self._median(comparison, "specificity")

    def median_selectivity(self, comparison: str) -> Optional[float]:
        return self._median(comparison, "selectivity")

    def median_capture(self, comparison: str) -> Optional[float]:
        return self._median(comparison, "capture_fraction")

    def median_q(self, comparison: str) -> Optional[float]:
        return self._median(comparison, "fisher_q")

    def median_p(self, comparison: str) -> Optional[float]:
        """P is written alongside q everywhere.

        It was missing entirely, which meant a run gated on --alpha P0.05
        reported q values while the cut had used p. Carrying both adjacent also
        makes the gap between them visible on sight -- the gap that took most
        of a day to notice when only one was on the page.
        """
        return self._median(comparison, "fisher_p")

    def median_fc_excess(self, comparison: str) -> Optional[float]:
        """fc_median minus its bucket baseline.

        The raw `fc_median` is nearly constant across pairs, because
        it is dominated by the comparison's overall differential rather than by
        anything about the pair -- every top pair of a real run landed in a
        narrow band against an identical bucket baseline. The excess over that
        baseline is the part that varies, so it is what gets reported.
        """
        values = [
            o.fc_median - o.fc_bucket_median
            for o in self.observations.get(comparison, ())
            if o.fc_median == o.fc_median and o.fc_bucket_median == o.fc_bucket_median
        ]
        return statistics.median(values) if values else None


def compile_anchor(anchor: Optional[str]) -> Optional[re.Pattern]:
    """Anchor spec -> a case-insensitive pattern, or None for unguided mode.

    Naming the putative partner to identify its alternate partners is the
    primary use; an unguided analysis over every pair is the other. So the
    anchor is an optional filter over every pair rather than a separate code
    path -- unguided mode is this function returning None and nothing else
    changing.

    A bare family name like ``IRF`` matches IRF1..IRF9 and composites such as
    ``STAT2::IRF9``, which is the common case. Anything containing regex
    metacharacters is honoured as a regex, so ``^IRF3$`` pins one member.

    The text is ALSO matched literally.
    Every anchor was compiled as a regex and nothing else, so a JASPAR name
    with parentheses -- ``IRF4(var.2)`` -- made a group of ``var.2`` and never
    matched the literal name, and an unbalanced one crashed with re.error. A
    name without metacharacters is unchanged by this; one with them matches
    as a regex or as itself. A spec that is not a valid regex is refused with
    the reason, as a MotifPartnerError.
    """
    if anchor is None:
        return None
    text = str(anchor).strip()
    if not text:
        return None
    literal = re.escape(text)
    if literal == text:
        return re.compile(text, re.IGNORECASE)
    try:
        return re.compile(f"(?:{literal})|(?:{text})", re.IGNORECASE)
    except re.error as exc:
        raise MotifPartnerError(
            f"--anchor {text!r} is not a valid regular expression ({exc}). A motif name is matched "
            "as written; to use a pattern, escape its metacharacters or write a valid one."
        ) from exc


def _pair_matches_anchor(row: Mapping[str, str], pattern: Optional[re.Pattern]) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(row.get("motif_a_name", "")) or
                pattern.search(row.get("motif_b_name", "")))


def partner_of(row: Mapping[str, str], pattern: Optional[re.Pattern]) -> Optional[str]:
    """The non-anchor member of a pair, or None when both or neither match."""
    if pattern is None:
        return None
    a, b = row.get("motif_a_name", ""), row.get("motif_b_name", "")
    a_hit, b_hit = bool(pattern.search(a)), bool(pattern.search(b))
    if a_hit and not b_hit:
        return b
    if b_hit and not a_hit:
        return a
    return None


def find_motif_instances(
    project: Union[str, Path], comparisons: Sequence[str],
) -> Optional[Path]:
    """The motif_instances table to read loci from, found without being told.

    The table is found without user intervention. Every comparison
    writes one into its own motif results directory, and because they all
    scan the same atlas with the same library the files are byte-identical --
    verified by md5 across the comparisons of a real project. So which one is
    used cannot matter, and asking would only create a way to point at the
    wrong project's file by mistake. That premise is checked, not
    assumed (_refuse_mixed_atlases).
    """
    root = Path(project).expanduser().resolve()
    for comparison in comparisons:
        for rel in _MOTIF_RESULT_DIRS:
            results_dir = root / comparison / rel
            if not results_dir.is_dir():
                continue
            found = sorted(results_dir.glob("*motif_instances*.csv"))
            if found:
                return found[0]
    return None


def _stratum_from_filename(path: Path, comparison: str) -> str:
    """The stratum a pair table's name carries: the text between the
    comparison prefix the motif finalizer put there via ``result_path`` and
    the table suffix, so the two stay in step by construction."""
    stem = path.name
    for token in (f"{comparison}_", "_motif_pair_statistics"):
        stem = stem.replace(token, "|", 1) if token in stem else stem
    parts = [p for p in stem.split("|") if p]
    stratum = parts[0].strip("_") if parts else path.stem
    return stratum or "(all)"


#: A fork of the analysis: (statistic folder, the differential comparison the
#: tables were classified on). ("", "") is a motif run that records neither --
#: an explicit single table from, or a run found by the glob.
Fork = Tuple[str, str]


def _recorded_pair_tables(
    comparison: str, comparison_dir: Path, results_dir: Path,
) -> Tuple[Optional[Dict[Fork, Dict[str, Path]]], Optional[Path]]:
    """The pair tables the motif checkpoint records as this comparison's, or None.

    The checkpoint says which tables are current, so a stratum the motif stage
    dropped on a later run, or a finalizer still running or dead partway,
    cannot contribute a stale or partial table. One that says the pairs are
    not finished -- a --stage singles hand-off, a run in progress or one that
    died (its record is withdrawn when the run starts) -- is refused. One that
    records ``pair_statistics`` paths names the tables, and a table on disk
    that it does not record is reported and left unread. A checkpoint that
    records neither (a synthetic or very old one) decides nothing, and so does
    a comparison without one: the caller falls back to the glob and says so.

    Returned per fork -- ``{(folder, differential comparison):
    {stratum: table}}`` -- from the folder, comparison and stratum each entry
    records, since the motif stage writes one enrichment per comparison of
    every statistic folder. An older entry that records neither is the fork
    ("", ""), its stratum read from the file name.
    """
    checkpoint = _motif_checkpoint(comparison_dir)
    if checkpoint is None:
        return None, None
    try:
        payload = json.loads(checkpoint.read_text())
    except (OSError, ValueError) as exc:
        raise MotifPartnerError(f"{comparison}: cannot read motif checkpoint {checkpoint}: {exc}") from exc
    if "pairs_complete" in payload and not payload.get("pairs_complete"):
        withdrawn = payload.get("withdrawn") or {}
        why = (
            f"it was withdrawn ({withdrawn.get('reason')}): the motif run that replaced these "
            f"results had not finished when this was read"
            if withdrawn else
            "the motif-pair analysis is not finished for this comparison (--stage singles, "
            "--skip-motif-pairs, or a run that did not complete)"
        )
        raise MotifPartnerError(
            f"{comparison}: {checkpoint} says pairs_complete is false -- {why}. "
            "Finish `atac motifs` for this comparison before comparing it with others."
        )
    recorded: Dict[Fork, Dict[str, Path]] = {}
    for entry in dict(payload.get("comparisons") or {}).values():
        if not isinstance(entry, Mapping) or not entry.get("pair_statistics"):
            continue
        table = _relocate(Path(str(entry["pair_statistics"])), comparison_dir)
        if table is None:
            raise MotifPartnerError(
                f"{comparison}: the motif checkpoint records the pair table "
                f"{entry['pair_statistics']}, which no longer exists (and no single file of "
                f"that name is under {comparison_dir}). Re-run `atac motifs` for this comparison."
            )
        if "folder" in entry:
            fork: Fork = (str(entry.get("folder") or ""), str(entry.get("comparison") or ""))
            stratum = str(entry.get("stratum") or "") or "(all)"
        else:
            # Keyed as the filename route keys it, so two comparisons resolved
            # by different routes still share their strata.
            fork, stratum = ("", ""), _stratum_from_filename(table, comparison)
        recorded.setdefault(fork, {})[stratum] = table
    if not recorded:
        return None, checkpoint
    read = {table.resolve() for tables in recorded.values() for table in tables.values()}
    unrecorded = [
        path for path in sorted(results_dir.rglob(_PAIR_TABLE_GLOB))
        if path.resolve() not in read
    ]
    if unrecorded:
        print(
            f"[paralome] {comparison}: {len(unrecorded)} pair table(s) under {results_dir} are not "
            f"recorded by the motif checkpoint {checkpoint} and are not read -- they were left by an "
            f"earlier motif run: " + ", ".join(path.name for path in unrecorded[:6])
            + (", ..." if len(unrecorded) > 6 else ""),
            file=sys.stderr, flush=True,
        )
    return recorded, checkpoint


def discover_pair_tables(
    project: Union[str, Path], comparisons: Sequence[str],
) -> Dict[str, Dict[Fork, Dict[str, Path]]]:
    """Locate every pair table of each named comparison, per statistical fork.

    Returns ``{comparison: {(folder, differential comparison): {stratum:
    path}}}``. From the motif checkpoint where it records them
    (_recorded_pair_tables), which also refuses a comparison whose pairs are
    not finished; only a comparison without that record falls back to the
    filename glob directly under results/ (the fork ("", "")), and says so.
    """
    root = Path(project).expanduser().resolve()
    if not root.is_dir():
        raise MotifPartnerError(f"--project is not a directory: {root}")

    found: Dict[str, Dict[Fork, Dict[str, Path]]] = {}
    for comparison in comparisons:
        comparison_dir = root / comparison
        if not comparison_dir.is_dir():
            available = sorted(p.name for p in root.iterdir() if p.is_dir())
            raise MotifPartnerError(
                f"comparison {comparison!r} not found under {root}. "
                f"Available: {', '.join(available) or '(none)'}"
            )
        results_dir = next(
            (comparison_dir / rel for rel in _MOTIF_RESULT_DIRS if (comparison_dir / rel).is_dir()),
            None,
        )
        if results_dir is None:
            raise MotifPartnerError(
                f"{comparison}: no motif results directory. Looked for "
                + " and ".join(str(comparison_dir / rel) for rel in _MOTIF_RESULT_DIRS)
                + ". Has `atac motifs` finished for this comparison?"
            )
        recorded, checkpoint = _recorded_pair_tables(comparison, comparison_dir, results_dir)
        if recorded is not None:
            found[comparison] = recorded
            continue
        tables: Dict[str, Path] = {}
        for path in sorted(results_dir.glob(_PAIR_TABLE_GLOB)):
            tables[_stratum_from_filename(path, comparison)] = path
        if not tables:
            raise MotifPartnerError(
                f"{comparison}: no {_PAIR_TABLE_GLOB} under {results_dir}. "
                "Motif pair analysis may have been skipped (--skip-motif-pairs) "
                "or stopped at --stage singles."
            )
        print(
            f"[paralome] {comparison}: "
            + (f"the motif checkpoint {checkpoint} records no pair tables" if checkpoint
               else "no motif checkpoint under " + str(comparison_dir))
            + f", so every {_PAIR_TABLE_GLOB} under {results_dir} is taken as current "
            f"({len(tables)} table(s)); whether they all belong to one finished motif run cannot be checked.",
            file=sys.stderr, flush=True,
        )
        found[comparison] = {("", ""): tables}
    return found


def partner_forks(
    discovered: Mapping[str, Mapping[Fork, Mapping[str, Path]]], comparisons: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Split the discovered tables into the forks the analysis runs once each.

    Every statistic is its own result ("each statistical
    reality needs its own forked set of files"), so tables of different
    statistic folders are never pooled as strata of one comparison, and
    neither are two comparisons of one folder (an ANOVA's contrasts and
    F-tests). Per folder present in every --comparisons run:

    - each run has ONE comparison in it (a run per state, the layout this
      analysis was built for): one fork, its members being each run's
      comparison whatever it is named;
    - otherwise one fork per comparison name that every run has, so the
      progression is read across the same comparison of each run.

    A folder or comparison not in every run cannot be followed across them;
    it is returned as a note, not run. Each fork carries ``key`` ("<folder>",
    "<folder>/<comparison>", or "" for the unfoldered layout), ``folder``,
    ``label`` (the comparison named in its file prefix, or ""), ``members``
    ({run: differential comparison}) and ``tables`` ({run: {stratum: path}}).
    """
    folders: List[str] = []
    for comparison in comparisons:
        for folder, _name in discovered.get(comparison, {}):
            if folder not in folders:
                folders.append(folder)
    forks: List[Dict[str, object]] = []
    skipped: List[str] = []
    for folder in folders:
        names = {
            comparison: [name for (f, name) in discovered.get(comparison, {}) if f == folder]
            for comparison in comparisons
        }
        missing = [comparison for comparison, found in names.items() if not found]
        where = f"statistic folder {folder!r}" if folder else "the tables directly under results/"
        if missing:
            skipped.append(
                f"{where} is not in every comparison (absent from {', '.join(missing)}), so it cannot "
                f"be followed across them; not analysed"
            )
            continue
        if all(len(found) == 1 for found in names.values()):
            members = {comparison: found[0] for comparison, found in names.items()}
            distinct = set(members.values())
            label = next(iter(distinct)) if len(distinct) == 1 else ""
            groups = [(label, members)]
        else:
            groups = []
            common = [name for name in names[comparisons[0]] if all(name in found for found in names.values())]
            for name in common:
                groups.append((name, {comparison: name for comparison in comparisons}))
            for comparison, found in names.items():
                for name in found:
                    if name not in common:
                        skipped.append(
                            f"{where}: comparison {name!r} of {comparison} is not in every run, so it "
                            f"cannot be followed across them; not analysed"
                        )
        for label, members in groups:
            key = folder + (f"/{label}" if folder and label else "")
            forks.append({
                "key": key,
                "folder": folder,
                "label": label,
                "members": dict(members),
                "tables": {
                    comparison: dict(discovered[comparison][(folder, members[comparison])])
                    for comparison in comparisons
                },
            })
    if not forks:
        raise MotifPartnerError(
            "No statistic folder's motif-pair tables are present in every one of "
            + ", ".join(comparisons) + ", so there is nothing to follow across them: "
            + "; ".join(skipped)
        )
    return forks, skipped


def _validate_columns(
    path: Path, fieldnames: Optional[Sequence[str]], directions: Sequence[str],
) -> None:
    present = set(fieldnames or ())
    needed = set(_REQUIRED_COLUMNS)
    for direction in directions:
        needed |= {c.format(d=direction) for c in _DIRECTION_COLUMNS}
    missing = sorted(needed - present)
    if missing:
        raise MotifPartnerError(
            f"{path.name} is missing column(s): {', '.join(missing)}. "
            "This analysis reads the FULL motif_pair_statistics table, not the "
            "truncated motif_pair_<alpha> convenience file."
        )


def _as_float(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


#: Enrichment means the pair's peaks opened FASTER than peaks generally. At
#: exactly 1.0 they opened at the same rate; below it they are depleted. So
#: this is the definition of the word, not a threshold anyone should be asked
#: to choose -- a minimum hit count or minimum specificity is not
#: something a human should be asked to set. Any value above 1.0 would be someone
#: quietly deciding how large an effect counts, which is the reader's judgment
#: and belongs in the reported selectivity column, not in a filter.
_ENRICHED_ABOVE = 1.0


def _open_indexed(path: Path) -> Tuple[object, object, Dict[str, int]]:
    """A csv.reader plus a name->position map.

    csv.reader with integer indices rather than DictReader: measured at 0.54s
    versus 0.88s on a real 285,572-row table, because DictReader builds a fresh
    dict for every row and this analysis wants nineteen of forty-seven columns.
    """
    handle = path.open(newline="")
    reader = csv.reader(handle)
    header = next(reader)
    return handle, reader, {name: index for index, name in enumerate(header)}


def _check_columns(path: Path, index: Mapping[str, int]) -> None:
    _validate_columns(path, list(index), DIRECTIONS)


def _anchor_hit(row: Sequence[str], idx: Mapping[str, int], pattern: Optional[re.Pattern]) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(row[idx["motif_a_name"]]) or pattern.search(row[idx["motif_b_name"]]))


def _scan_worker(task) -> set:
    """PASS ONE: which pair_ids ever clear the gate in this file.

    Returns a set of ids and nothing else. That tiny payload is the whole point
    of splitting the work in two: it makes the file-level parallelism below
    almost free, where shipping 571,144 observation objects back from each
    worker would have cost more in pickling than the parallelism saved.
    """
    path, alpha, alpha_metric, pattern = task
    passing: set = set()
    matched: set = set()
    handle, reader, idx = _open_indexed(path)
    try:
        _check_columns(path, idx)
        pid = idx["pair_id"]
        fields = [(idx[f"{d}_specificity_ratio"], idx[f"{d}_fisher_{alpha_metric}"]) for d in DIRECTIONS]
        for row in reader:
            if not _anchor_hit(row, idx, pattern):
                continue
            matched.add(row[pid])
            if row[pid] in passing:
                continue
            for spec_i, val_i in fields:
                try:
                    if float(row[spec_i]) > _ENRICHED_ABOVE and float(row[val_i]) < alpha:
                        passing.add(row[pid])
                        break
                except (TypeError, ValueError):
                    continue
    finally:
        handle.close()
    return passing, matched


def _collect_worker(task):
    """PASS TWO: the numbers, for the surviving pairs only.

    Skips ~98% of rows on the pair_id lookup before touching any float, which
    is what makes the second read cheaper than the first despite parsing the
    same bytes. Returns plain tuples so the payload pickles cheaply.
    """
    path, comparison, stratum, pattern, wanted, alpha, alpha_metric = task
    out: List[tuple] = []
    handle, reader, idx = _open_indexed(path)
    try:
        pid = idx["pair_id"]
        a_i, b_i = idx["motif_a_name"], idx["motif_b_name"]
        cooc_i, glob_i = idx["cooccurring_peak_count"], idx["global_peaks"]
        per_direction = [
            (d, idx[f"{d}_specificity_ratio"], idx[f"{d}_fisher_p"], idx[f"{d}_fisher_q"],
             idx[f"{d}_foreground_hits"], idx[f"{d}_foreground_peaks"],
             idx[f"{d}_fc_median"], idx[f"{d}_fc_bucket_median"])
            for d in DIRECTIONS
        ]
        for row in reader:
            pair_id = row[pid]
            if pair_id not in wanted:
                continue
            if not _anchor_hit(row, idx, pattern):
                continue
            cooc, glob = _as_int(row[cooc_i]), _as_int(row[glob_i])
            for d, s_i, p_i, q_i, h_i, n_i, f_i, fb_i in per_direction:
                out.append((
                    pair_id, row[a_i], row[b_i], d,
                    _as_float(row[s_i]), _as_float(row[p_i]), _as_float(row[q_i]),
                    _as_int(row[h_i]), _as_int(row[n_i]), cooc, glob,
                    _as_float(row[f_i]), _as_float(row[fb_i]),
                ))
    finally:
        handle.close()
    return comparison, stratum, out


def passes_gate(obs: PairObservation, *, alpha: float, alpha_metric: str) -> bool:
    """Whether one observation counts as significant.

    Two conditions, neither of them tunable: the pair is enriched rather than
    depleted, and its test clears alpha.

    There is no minimum-hits floor. Fisher's exact is exact, so a small-count
    result is genuinely unlikely under the null. What small counts risk is
    irreproducibility, and that is what the per-stratum recurrence count
    measures and reports: a three-peak pair recurring in nine strata of nine
    means something; the same pair in one of nine does not.
    """
    if not (obs.specificity > _ENRICHED_ABOVE):
        return False
    value = obs.fisher_q if alpha_metric == "q" else obs.fisher_p
    return value == value and value < alpha


def classify_progression(
    flags: Sequence[bool], comparisons: Sequence[str],
) -> str:
    """Label a pair's presence/absence pattern across the ORDERED comparisons.

    The list order is the progression: the first entry is the baseline and each
    later entry a successive state. That ordering is what makes "acquired" and
    "lost" meaningful rather than merely "different somewhere", and it is why
    the CLI treats --comparisons as a sequence rather than a set.
    """
    if not any(flags):
        return "absent"
    if all(flags):
        return "persistent in all"
    first = flags.index(True)
    last = len(flags) - 1 - list(reversed(flags)).index(True)
    contiguous = all(flags[first:last + 1])
    if contiguous and last == len(flags) - 1:
        if first == 0:
            return "persistent in all"
        return f"acquired at {comparisons[first]}, retained"
    if contiguous and first == 0:
        return f"lost after {comparisons[last]}"
    if contiguous:
        if first == last:
            return f"{comparisons[first]} only"
        return f"{comparisons[first]}..{comparisons[last]} only"
    present = [c for c, f in zip(comparisons, flags) if f]
    return "intermittent (" + ", ".join(present) + ")"


def contrast_specs(comparisons: Sequence[str]) -> List[Tuple[str, str]]:
    """Every ordered pair of comparisons, earlier-listed first.

    A study may know its progression (A > B > C), but that is one
    case; so every pair is tested rather than only
    consecutive ones -- the list order is used solely to orient each contrast,
    so the sign of the change reads as "later minus earlier", and it does not
    have to be a real progression for the statistics to mean anything.
    """
    return [
        (comparisons[i], comparisons[j])
        for i in range(len(comparisons))
        for j in range(i + 1, len(comparisons))
    ]


def _log_selectivity(obs: PairObservation) -> Optional[Tuple[float, float]]:
    """log selectivity for one stratum, and its variance.

    selectivity = (k/|S|) / (n/N), so its log is log(k/|S|) - log(n/N). |S| --
    the pair's co-occurring peak count -- is IDENTICAL across comparisons,
    because they share one atlas and one motif scan. That is what makes this a
    clean two-sample problem on a common denominator rather than a comparison
    of two differently-defined quantities.

    Variance is the binomial delta method on k/|S|. The number of significant
    peaks n is a design quantity in the tens of thousands and is treated as
    known; its sampling variance is negligible beside k's. A zero count gets
    the Haldane-Anscombe half, which is what keeps a pair that went from 0 to
    40 peaks from producing an infinite change.
    """
    k, total = obs.foreground_hits, obs.cooccurring_peaks
    n, atlas = obs.foreground_peaks, obs.global_peaks
    if total <= 0 or atlas <= 0 or n <= 0:
        return None
    adjusted = float(k) if k > 0 else 0.5
    if adjusted >= total:
        adjusted = total - 0.5
    if adjusted <= 0:
        return None
    value = math.log(adjusted / total) - math.log(n / atlas)
    variance = (total - adjusted) / (total * adjusted)
    return value, variance


def conservative_effect(delta: float, ci_low: float, ci_high: float) -> float:
    """The smallest change the data still supports -- the CI bound nearest zero.

    Sorting by the largest change biases in a completely different
    way from sorting by the strongest statistics, and both biases are real:

      rank by change_q   -> favours LARGE |S|. Power scales with the peak set,
                            so a pair with thousands of co-occurring peaks
                            clears q on a trivial effect.
      rank by |delta|    -> favours SMALL |S|. Six peaks going 0->4 is a huge
                            log ratio with an enormous interval on it. On real
                            data the median |S| of the top 200 differed by
                            more than an order of magnitude between the two
                            rankings, so they describe near-disjoint studies.

    The confidence bound is not a third arbitrary choice; it is the quantity
    that penalises both. A big effect measured imprecisely has a bound near
    zero. A tiny effect measured precisely has a bound near its own small
    value. Only a substantial effect measured well scores high.

    Floored at zero: an interval straddling zero supports no change at all,
    and its sign is not evidence of direction. Note the interval is the plain
    95% one while the cut uses BH-adjusted q, so a few significant rows can
    still bound to zero -- that is the interval being honest about a single
    comparison, not a contradiction.
    """
    if delta != delta or ci_low != ci_low or ci_high != ci_high:
        return 0.0
    bound = ci_low if delta > 0 else -ci_high
    return bound if bound > 0 else 0.0


def contrast_between(
    record: "PairRecord", earlier: str, later: str,
) -> Optional[Dict[str, object]]:
    """Does this pair's enrichment DIFFER between two comparisons?

    The question the pairs analysis cannot answer: this is a comparison of
    comparisons. Everything upstream tests one comparison against its own
    background; nothing tests one comparison against another, and reading a
    change off two separate significance calls is the difference-between-
    significant-and-non-significant fallacy -- q=0.049 here and q=0.051 there
    is not a change.

    Computed per stratum and combined by inverse-variance weighting, so strata
    where the pair has more co-occurring peaks carry more weight. That also
    makes recurrence do statistical work rather than being a column to eyeball.
    """
    left = {o.stratum: o for o in record.observations.get(earlier, ())}
    right = {o.stratum: o for o in record.observations.get(later, ())}
    differences: List[float] = []
    variances: List[float] = []
    for stratum in sorted(set(left) & set(right)):
        a, b = _log_selectivity(left[stratum]), _log_selectivity(right[stratum])
        if a is None or b is None:
            continue
        differences.append(b[0] - a[0])
        variances.append(a[1] + b[1])
    pooled = inverse_variance_pooled(differences, variances)
    if pooled is None:
        return None
    return {
        "delta_log_selectivity": pooled["estimate"],
        "stderr": pooled["stderr"],
        "ci_low": pooled["ci_low"],
        "ci_high": pooled["ci_high"],
        "z": pooled["z"],
        "p": pooled["p"],
        "strata_compared": int(pooled["used"]),
    }


def change_field(alpha_metric: str) -> str:
    """Which change column --alpha selects.

    Change_p < alpha should hold for --alpha P. It did not -- the cut was
    hardcoded to change_q while its filename was built from the chosen metric,
    so --alpha P0.05 produced a file called p_0.05 that had been cut on q. That
    is the same lie as a table whose name does not say what made it, which this
    module already refuses to ship elsewhere.
    """
    return "change_p" if str(alpha_metric).strip().lower() == "p" else "change_q"


def change_value(row: Mapping[str, object], column: str) -> Optional[float]:
    """A contrast row's change_p / change_q as a float, or None when absent.

    Only a blank or None is missing. A zero is the most significant value
    there is (the normal tail returns 0.0 once |z| is above about 38, which BH
    keeps as q=0), so ``float(x or 1.0)`` would read the strongest changes as
    p=1.
    """
    value = row.get(column)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def change_is_significant(row: Mapping[str, object], column: str, alpha: float) -> bool:
    """Whether a contrast row's change clears alpha. One test for every reader
    -- the cut, the shift class and the gene annotation -- so they cannot
    disagree about which rows changed."""
    value = change_value(row, column)
    return value is not None and value < alpha


def _refuse_mixed_atlases(
    project: Union[str, Path], comparisons: Sequence[str],
    global_peaks_seen: Mapping[int, Sequence[str]], cooccurrence_disagreements: Sequence[str],
) -> None:
    """Refuse when the comparisons were not scanned over one atlas with one library.

    Selectivity contrasts divide a pair's capture in one comparison by its
    capture in another over the SAME co-occurring peak set, and the loci come
    from one comparison's motif_instances. A comparison whose atlas was
    re-merged after a sample was dropped, or scanned with another motif library
    or --max-motifs-per-peak, has a different |S| and global_peaks: the
    contrasts would still print, meaning nothing. Three checks, each of which
    names what disagrees: global_peaks across every table read; each pair's
    cooccurring_peak_count across tables; and, where motif checkpoints exist,
    the library, the scan's peak count and --max-motifs-per-peak they record.
    """
    problems: List[str] = []
    if len(global_peaks_seen) > 1:
        problems.append(
            "global_peaks differs between tables: " + "; ".join(
                f"{glob} in {', '.join(where[:4])}" + (", ..." if len(where) > 4 else "")
                for glob, where in sorted(global_peaks_seen.items())
            )
        )
    if cooccurrence_disagreements:
        problems.append("cooccurring_peak_count differs between tables: " + "; ".join(cooccurrence_disagreements))
    root = Path(project).expanduser().resolve()
    recorded: Dict[str, Dict[str, object]] = {}
    for comparison in comparisons:
        checkpoint = _motif_checkpoint(root / comparison)
        if checkpoint is None:
            continue
        try:
            payload = json.loads(checkpoint.read_text())
        except (OSError, ValueError):
            continue
        recorded[comparison] = {
            key: payload.get(key)
            for key in ("motif_database", "motifs_scanned", "peaks_scanned", "max_motifs_per_peak", "motif_p_threshold")
            if payload.get(key) is not None
        }
    for key in ("motif_database", "motifs_scanned", "peaks_scanned", "max_motifs_per_peak", "motif_p_threshold"):
        values = {c: v[key] for c, v in recorded.items() if key in v}
        if len(set(map(str, values.values()))) > 1:
            problems.append(
                f"the motif checkpoints disagree on {key}: " + "; ".join(f"{c}: {v}" for c, v in sorted(values.items()))
            )
    if problems:
        raise MotifPartnerError(
            "These comparisons were not scanned over one peak atlas with one motif scan, so their "
            "pair statistics cannot be contrasted: " + ". ".join(problems) + ". Re-run `atac motifs` "
            "for every comparison from the same atlas, motif library and settings."
        )


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or value != value:
        return ""
    return f"{value:.{digits}g}"


def run_motif_partner_shift(
    project: Union[str, Path],
    comparisons: Sequence[str],
    output_dir: Union[str, Path],
    *,
    anchor: Optional[str] = None,
    alpha: float = 0.05,
    alpha_metric: str = "p",
    threads: int = 1,
    #: 0 means every pair that passed; a positive value narrows that to the
    #: top N.
    top_pairs: int = _DEFAULT_TOP_PAIRS,
    loci_max_rows: int = 50_000_000,
    overwrite: bool = True,
) -> Dict[str, object]:
    """Compare one motif's partner repertoire across ordered comparisons."""
    started = time.monotonic()
    comparisons = [str(c).strip() for c in comparisons if str(c).strip()]
    if len(comparisons) < 2:
        raise MotifPartnerError(
            "--comparisons needs at least two comparisons to compare; "
            f"got {comparisons or '(none)'}"
        )
    if len(set(comparisons)) != len(comparisons):
        raise MotifPartnerError(f"--comparisons contains duplicates: {comparisons}")
    directions = list(DIRECTIONS)
    alpha_metric = str(alpha_metric).strip().lower()
    if alpha_metric not in {"p", "q"}:
        raise MotifPartnerError("alpha metric must be 'p' or 'q'")

    out = Path(output_dir).expanduser().resolve()
    results = out / "results"
    checkpoints = out / "checkpoints"
    for path in (results, checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    file_prefix = derive_file_prefix(out)
    print(stage_banner("motif_partner_shift", file_prefix, out), flush=True)

    # --no-overwrite, so a re-run does not spend hours writing the same files over.
    #
    # It is deliberately LOUD, because the result is then a mixture of this
    # run's outputs and an earlier run's: reused paths are printed and recorded
    # in the checkpoint. Every output is written atomically and recorded in
    # checkpoints/motif_partner_outputs.json with a fingerprint of the method,
    # the inputs (path, size, mtime) and the settings that produced it (the
    # file names carry none of --anchor, --alpha or --top-pairs). A file is
    # reused only when that record matches; anything else -- a table truncated
    # by a killed run included -- is recomputed, and says why.
    reused: List[str] = []
    manifest_path = checkpoints / _OUTPUT_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    except (OSError, ValueError):
        manifest = {}
    finished: Dict[str, Dict[str, object]] = dict(manifest.get("outputs") or {})

    def _key(path: Path) -> str:
        """The manifest key: the path under results/, so the same file name in
        two statistic folders (t and paired_t name their comparisons alike) is
        two records."""
        try:
            return str(Path(path).resolve().relative_to(results))
        except ValueError:
            return Path(path).name

    def _save_manifest() -> None:
        write_json(manifest_path, {"method": METHOD, "outputs": finished},
                   atomic=True, stamp=True)

    def _forget(path: Path) -> None:
        if finished.pop(_key(path), None) is not None:
            _save_manifest()

    def _skip(path: Path, fingerprint: str) -> bool:
        if overwrite or not path.exists():
            _forget(path)
            return False
        entry = finished.get(_key(path)) or {}
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        if entry.get("fingerprint") == fingerprint and entry.get("size") == size:
            reused.append(str(path))
            print(f"  reusing (exists, --no-overwrite): {path.name}", flush=True)
            return True
        print(
            f"  recomputing (exists, but not recorded as a finished output of these "
            f"inputs and settings -- an interrupted or different run): {path.name}",
            flush=True,
        )
        _forget(path)
        return False

    def _record(path: Path, fingerprint: str, **meta: object) -> None:
        finished[_key(path)] = {
            "fingerprint": fingerprint, "size": path.stat().st_size,
            "method": METHOD, **meta,
        }
        _save_manifest()

    def _set_aside(path: Path) -> None:
        """An output this run did not produce, left by an earlier run. Renamed
        rather than left under the live name, where it would read as this
        run's answer."""
        _forget(path)
        if path.exists():
            stale = path.with_name(path.name + ".stale")
            os.replace(path, stale)
            print(
                f"[paralome] {path.name} was left by an earlier run and this run did "
                f"not produce it; moved to {stale.name} so it is not read as current.",
                file=sys.stderr, flush=True,
            )

    pattern = compile_anchor(anchor)
    discovered = discover_pair_tables(project, comparisons)
    forks, forks_skipped = partner_forks(discovered, comparisons)
    for note in forks_skipped:
        print(f"[paralome] motif partner shift: {note}", file=sys.stderr, flush=True)

    def _fork(fork: Mapping[str, object]) -> Dict[str, object]:
        """The whole analysis over one statistical fork's tables, into its folder."""
        tables = fork["tables"]
        results = fork["results"]
        results.mkdir(parents=True, exist_ok=True)
        file_prefix = fork["file_prefix"]
        print(
            f"Motif partner shift: statistic fork {fork['key'] or '(results/)'} -> {results}",
            flush=True,
        )
        base_settings: Dict[str, object] = {
            "method": METHOD,
            "fork": fork["key"],
            "fork_members": dict(fork["members"]),
            "project": str(Path(project).expanduser().resolve()),
            "comparisons": list(comparisons),
            "anchor": anchor or "",
            "alpha": float(alpha),
            "alpha_metric": alpha_metric,
            "pair_tables": {
                c: {s: _file_identity(path) for s, path in sorted(tables[c].items())}
                for c in comparisons
            },
        }
        base_fp = _fingerprint(base_settings)

        # Two passes, both parallel over files. Pass one keeps nothing but the ids
        # that ever clear the gate; pass two builds objects for those only, so ~98%
        # of rows are rejected on a set lookup before any float is parsed. The
        # small payloads are what make the parallelism worth having -- returning
        # 571,144 observation objects per file would cost more in pickling than the
        # workers save. Measured on a real project: 196s serial for the single-pass
        # version, against a two-pass scan that skips most of the second read.
        strata_seen: Dict[str, List[str]] = {}
        jobs: List[Tuple[str, str, Path]] = []
        for comparison in comparisons:
            for stratum, path in sorted(tables[comparison].items()):
                strata_seen.setdefault(comparison, []).append(stratum)
                jobs.append((comparison, stratum, path))

        workers = max(1, int(threads))
        print(
            f"Motif partner shift: {len(jobs)} table(s) across {len(comparisons)} comparison(s), "
            f"{workers} worker(s)",
            flush=True,
        )

        def _run(worker, tasks):
            if workers == 1 or len(tasks) == 1:
                return [worker(task) for task in tasks]
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=ctx) as pool:
                return list(pool.map(worker, tasks))

        wanted_ids: set = set()
        matched_ids: set = set()
        for passing, matched in _run(
            _scan_worker, [(path, alpha, alpha_metric, pattern) for _c, _s, path in jobs]
        ):
            wanted_ids |= passing
            matched_ids |= matched

        # Two different empty results, and conflating them is how a typo becomes a
        # finding. An anchor that matched no pair at all is a mistake the user can
        # fix; an anchor that matched thousands of pairs, none of which cleared the
        # cut, is a real answer and gets written out as one.
        if not matched_ids:
            raise MotifPartnerError(
                f"no motif pairs matched anchor {anchor!r}. Check the spelling against the "
                "motif_a_name/motif_b_name columns of a pair table -- matching is "
                "case-insensitive, and a bare family name like IRF matches IRF1..IRF9 "
                "and composites such as STAT2::IRF9."
            )
        if not wanted_ids:
            print(
                f"[paralome] {len(matched_ids)} pair(s) matched the anchor but none cleared "
                f"{alpha_metric}<{alpha:g} in any stratum of any comparison. Writing empty "
                "result tables rather than failing -- that is an answer, not an error.",
                file=sys.stderr, flush=True,
            )
        print(
            f"Motif partner shift: {len(wanted_ids)} pair(s) clear {alpha_metric}<{alpha:g} somewhere "
            f"({time.monotonic() - started:.1f}s)",
            flush=True,
        )

        records: Dict[Tuple[str, str], PairRecord] = {}
        # The premise of every number below
        # -- one atlas, one scan, so |S| and global_peaks are the same in every
        # comparison -- is checked instead of assumed. What was seen, so a
        # violation can be named.
        global_peaks_seen: Dict[int, List[str]] = {}
        cooccurrence_disagreements: List[str] = []
        if wanted_ids:
            collected = _run(_collect_worker, [
                (path, comparison, stratum, pattern, wanted_ids, alpha, alpha_metric)
                for comparison, stratum, path in jobs
            ])
            for comparison, stratum, rows in collected:
                for (pair_id, motif_a, motif_b, direction, spec, fp, fq,
                     hits, fg_peaks, cooc, glob, fcm, fcb) in rows:
                    obs = PairObservation(
                        stratum=stratum, specificity=spec, fisher_p=fp, fisher_q=fq,
                        foreground_hits=hits, foreground_peaks=fg_peaks,
                        cooccurring_peaks=cooc, global_peaks=glob,
                        fc_median=fcm, fc_bucket_median=fcb,
                    )
                    where = f"{comparison}/{stratum}"
                    seen_in = global_peaks_seen.setdefault(glob, [])
                    if where not in seen_in:
                        seen_in.append(where)
                    key = (pair_id, direction)
                    record = records.get(key)
                    if record is None:
                        record = records[key] = PairRecord(
                            pair_id=pair_id, motif_a=motif_a, motif_b=motif_b,
                            direction=direction, cooccurring_peaks=cooc,
                        )
                    elif cooc != record.cooccurring_peaks and len(cooccurrence_disagreements) < 6:
                        cooccurrence_disagreements.append(
                            f"{pair_id} has {record.cooccurring_peaks} co-occurring peaks in one table and "
                            f"{cooc} in {where}"
                        )
                    record.add(
                        comparison, obs,
                        passes_gate(obs, alpha=alpha, alpha_metric=alpha_metric),
                    )
        _refuse_mixed_atlases(project, comparisons, global_peaks_seen, cooccurrence_disagreements)
        print(
            f"Motif partner shift: collected ({time.monotonic() - started:.1f}s)",
            flush=True,
        )

        # --- long form: one row per pair per comparison -----------------------
        summary_path = result_path(
            results, "motif_partner_statistics.csv",
            file_prefix=file_prefix, modality=ATAC_MOTIF_PAIR_MODALITY,
        )
        summary_fields = [
            "pair_id", "direction", "motif_a_name", "motif_b_name", "partner", "comparison",
            "cooccurring_peaks", "strata_total", "strata_significant",
            "median_specificity", "median_selectivity", "median_capture_fraction",
            "median_background_fraction", "median_fc_excess",
            "median_fisher_p", "median_fisher_q",
            "significant_strata",
        ]
        # Same convention as `atac motifs`: one full long-form table, and one cut with the metric
        # and alpha in the filename, so the file says which threshold produced it.
        cut_rows: List[Dict[str, object]] = []
        # No motif_partner_statistics.csv: its rows are assembled for the shift table below.
        with contextlib.nullcontext(io.StringIO()) as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
            writer.writeheader()
            for record in records.values():
                partner = partner_of(
                    {"motif_a_name": record.motif_a, "motif_b_name": record.motif_b}, pattern,
                )
                for comparison in comparisons:
                    obs = record.observations.get(comparison, ())
                    if not obs:
                        continue
                    payload = {
                        "pair_id": record.pair_id,
                        "direction": record.direction,
                        "motif_a_name": record.motif_a,
                        "motif_b_name": record.motif_b,
                        "partner": partner or "",
                        "comparison": comparison,
                        "cooccurring_peaks": record.cooccurring_peaks,
                        "strata_total": len(obs),
                        "strata_significant": record.recurrence(comparison),
                        "median_specificity": _fmt(record.median_specificity(comparison)),
                        "median_selectivity": _fmt(record.median_selectivity(comparison)),
                        "median_capture_fraction": _fmt(record.median_capture(comparison)),
                        "median_background_fraction": _fmt(
                            statistics.median([o.background_fraction for o in obs])
                        ),
                        "median_fc_excess": _fmt(record.median_fc_excess(comparison)),
                        "median_fisher_p": _fmt(record.median_p(comparison)),
                        "median_fisher_q": _fmt(record.median_q(comparison)),
                        "significant_strata": ";".join(sorted(record.significant.get(comparison, ()))),
                    }
                    writer.writerow(payload)
                    if record.recurrence(comparison):
                        cut_rows.append(payload)

        # --- the comparison OF comparisons ------------------------------------
        # Everything above re-reports what the pairs analysis already computed.
        # This is the part it cannot: whether a pair's enrichment actually differs
        # between two comparisons, tested directly rather than inferred from two
        # separate significance calls.
        contrasts = contrast_specs(comparisons)
        contrast_rows: List[Dict[str, object]] = []
        pending: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        for record in records.values():
            partner = partner_of(
                {"motif_a_name": record.motif_a, "motif_b_name": record.motif_b}, pattern,
            )
            for earlier, later in contrasts:
                result = contrast_between(record, earlier, later)
                if result is None:
                    continue
                row = {
                    "pair_id": record.pair_id, "direction": record.direction,
                    "motif_a_name": record.motif_a, "motif_b_name": record.motif_b,
                    "partner": partner or "",
                    "control_state": earlier,
                    "treatment_state": later,
                    "cooccurring_peaks": record.cooccurring_peaks,
                    "strata_compared": result["strata_compared"],
                    "control_selectivity": _fmt(record.median_selectivity(earlier)),
                    "treatment_selectivity": _fmt(record.median_selectivity(later)),
                    "delta_log_selectivity": result["delta_log_selectivity"],
                    "stderr": result["stderr"],
                    "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                    "z": result["z"], "change_p": result["p"],
                    "effect_lower_bound": conservative_effect(
                        result["delta_log_selectivity"], result["ci_low"], result["ci_high"],
                    ),
                }
                contrast_rows.append(row)
                # BH within (contrast, direction), matching how the upstream
                # corrects within one stratum and direction rather than across
                # everything at once.
                pending.setdefault(
                    (row["control_state"], row["treatment_state"], record.direction), [],
                ).append(row)

        for group in pending.values():
            qvalues = benjamini_hochberg([r["change_p"] for r in group])
            for row, qvalue in zip(group, qvalues):
                row["change_q"] = float(qvalue) if qvalue == qvalue else ""

        contrast_path = result_path(
            results, "motif_partner_contrasts.csv",
            file_prefix=file_prefix, modality=ATAC_MOTIF_PAIR_MODALITY,
        )
        # The statistics table has a column of comparison; a comparison
        # of comparisons needs another column, and the relative positions of the
        # comparisons in those two columns give the control state and the
        # treatment state. A PAIR of columns rather than a concatenated name keeps
        # the label readable when the comparisons themselves contain "_vs_".
        # Named *_state rather than bare control/treatment because within
        # StateA_vs_CTL the control is already CTL; these name the comparisons
        # being contrasted, one level up.
        contrast_fields = ["pair_id", "direction", "motif_a_name", "motif_b_name", "partner",
                           "control_state", "treatment_state",
                           "cooccurring_peaks", "strata_compared",
                           
                           "control_selectivity", "treatment_selectivity",
                           "delta_log_selectivity", "stderr", "ci_low", "ci_high",
                           "effect_lower_bound", "z", "change_p", "change_q"]
        contrast_rows.sort(key=lambda r: (r.get("change_q") if r.get("change_q") != "" else 1.0,
                                          r["change_p"]))
        contrast_path_skip = _skip(contrast_path, base_fp)
        with _output(contrast_path, contrast_path_skip) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(contrast_fields)
            for row in contrast_rows:
                writer.writerow([_fmt(row[f]) if isinstance(row.get(f), float) else row.get(f, "")
                                 for f in contrast_fields])

        # The contrasts table is shipped whole (a list the user sees that would change with alpha goes
        # to the viewer uncut). Only the loci and genes detail files, too large uncut, keep the --alpha cut.
        change_column = change_field(alpha_metric)
        cut_contrasts = [row for row in contrast_rows
                         if change_is_significant(row, change_column, alpha)]

        # --- wide form: one row per pair, with the progression classification ---
        shift_path = result_path(
            results, "motif_partner_shift.csv",
            file_prefix=file_prefix, modality=ATAC_MOTIF_PAIR_MODALITY,
        )
        shift_fields = ["pair_id", "direction", "motif_a_name", "motif_b_name", "partner",
                        "cooccurring_peaks"]
        for comparison in comparisons:
            shift_fields += [
                f"{comparison}_strata_total",
                f"{comparison}_specificity",
                f"{comparison}_selectivity",
                f"{comparison}_capture_fraction",
                f"{comparison}_background_fraction",
                f"{comparison}_fisher_p",
                f"{comparison}_fisher_q",
            ]

        # Every pair, uncut: no class, pattern or count of strata below --alpha (the viewer applies the user's
        # threshold to the per-comparison p and q). Ordered by the pair's smallest median Fisher p.
        def _best_p(record: PairRecord) -> float:
            values = [record.median_p(c) for c in comparisons]
            finite = [v for v in values if v is not None and math.isfinite(v)]
            return min(finite) if finite else 1.0

        ranked = sorted(records.values(), key=_best_p)

        shift_path_skip = _skip(shift_path, base_fp)
        with _output(shift_path, shift_path_skip) as handle:
            writer = csv.DictWriter(handle, fieldnames=shift_fields, lineterminator="\n")
            writer.writeheader()
            for record in ranked:
                partner = partner_of(
                    {"motif_a_name": record.motif_a, "motif_b_name": record.motif_b}, pattern,
                )
                row = {
                    "pair_id": record.pair_id,
                    "direction": record.direction,
                    "motif_a_name": record.motif_a,
                    "motif_b_name": record.motif_b,
                    "partner": partner or "",
                    "cooccurring_peaks": record.cooccurring_peaks,
                }
                for comparison in comparisons:
                    obs = record.observations.get(comparison, ())
                    # Every comparison's numbers are written for every pair, not
                    # only the comparison that made it significant. A pair is
                    # selected BECAUSE it was significant somewhere, which inflates
                    # its effect size there; the other columns are the control that
                    # makes that visible instead of hidden.
                    total = len(strata_seen.get(comparison, ()))
                    row[f"{comparison}_strata_total"] = total
                    row[f"{comparison}_specificity"] = _fmt(record.median_specificity(comparison))
                    row[f"{comparison}_selectivity"] = _fmt(record.median_selectivity(comparison))
                    row[f"{comparison}_capture_fraction"] = _fmt(record.median_capture(comparison))
                    row[f"{comparison}_background_fraction"] = _fmt(
                        statistics.median([o.background_fraction for o in obs]) if obs else None
                    )
                    row[f"{comparison}_fisher_p"] = _fmt(record.median_p(comparison))
                    row[f"{comparison}_fisher_q"] = _fmt(record.median_q(comparison))
                writer.writerow(row)

        # The cheap tables are recorded as finished only once all of
        # them are on disk; a run killed in between simply recomputes them.
        for written_path, was_skipped in (
            (contrast_path, contrast_path_skip), (shift_path, shift_path_skip),
        ):
            if not was_skipped:
                _record(written_path, base_fp)

        # --- optional peak-level loci for the top pairs -----------------------
        # Loci follow the same restriction as the gene annotation --
        # pairs whose CHANGE is significant. A pair that did not change
        # has the same co-occurring peaks in every comparison, so its loci say
        # nothing about the contrast. ONE pair set serves both detail files, so
        # --loci-top-n narrows the genes too.
        #
        # Ranked by effect size, not by input order: this chooses which pairs
        # are worth detailing, and significance is a poor rank for that when |S| in
        # the thousands reaches q<0.05 on a modest effect.
        effect: Dict[str, float] = {}
        for row in cut_contrasts:
            magnitude = float(row["effect_lower_bound"])
            if magnitude > effect.get(row["pair_id"], -1.0):
                effect[row["pair_id"]] = magnitude
        ordered_pairs = sorted(effect, key=lambda pid: -effect[pid])
        total_changed = len(ordered_pairs)
        if top_pairs > 0 and total_changed > top_pairs:
            ordered_pairs = ordered_pairs[:top_pairs]
            print(
                f"[paralome] detailing the top {len(ordered_pairs)} of {total_changed:,} "
                f"changed pair(s) by effect_lower_bound -- the CI bound nearest zero, "
                f"which ranks neither by raw effect nor by significance alone. "
                f"For both the locus export and the gene annotation. "
                f"Pass --top-pairs N for more, or "
                f"--top-pairs 0 for all of them -- all was tens of millions of gene rows "
                f"and over 10 GB on a three-state project.",
                file=sys.stderr, flush=True,
            )
        changed_ids = set(ordered_pairs)
        loci_path: Optional[Path] = None
        loci_pairs = 0
        instances = find_motif_instances(project, comparisons)
        planned_loci = result_path(
            results, "motif_partner_loci.csv",
            file_prefix=file_prefix, modality=ATAC_MOTIF_PAIR_MODALITY,
        )
        planned_genes = result_path(
            results, "motif_partner_genes.csv",
            file_prefix=file_prefix, modality=ATAC_MOTIF_PAIR_MODALITY,
        )
        # One top-N rule for BOTH detail files. The gene annotation used
        # none, so it described co-occurrences the statistics never counted.
        top_n, top_n_recorded = (
            resolve_max_motifs_per_peak(project, comparisons) if instances is not None
            else (_DEFAULT_MAX_MOTIFS_PER_PEAK, False)
        )
        if instances is not None and changed_ids and not top_n_recorded:
            print(
                f"[paralome] no max_motifs_per_peak in any motif checkpoint (runs "
                f"before 0.0.153 did not record it); assuming the default "
                f"{top_n}. If that run used another value the locus export and gene "
                f"annotation will not match its cooccurring_peak_count.",
                file=sys.stderr, flush=True,
            )
        detail_settings = dict(
            base_settings, top_pairs=int(top_pairs), max_motifs_per_peak=int(top_n),
            motif_instances=_file_identity(instances),
        )
        # The locus export and the gene annotation are the only expensive writers,
        # so for them --no-overwrite skips the COMPUTATION, not just the output.
        if changed_ids and instances is not None:
            loci_fp = _fingerprint(dict(detail_settings, output="loci",
                                        loci_max_rows=int(loci_max_rows)))
            if _skip(planned_loci, loci_fp):
                loci_path = planned_loci
                loci_pairs = int((finished.get(_key(planned_loci)) or {}).get("loci_pairs_exported", 0))
            else:
                # Deduplicated by pair, NOT by (pair, direction). A pair's co-occurring
                # peaks are a fixed property of the atlas -- the same set whichever
                # direction it was reported in -- so a pair passing in both would
                # otherwise emit every locus twice.
                # 0 means ALL, in both modes, with no hidden override. Narrowing is
                # available; it is not applied on the user's behalf.
                wanted: List[PairRecord] = []
                seen_pairs: set = set()
                for record in ranked:
                    if record.pair_id in seen_pairs or record.pair_id not in changed_ids:
                        continue
                    seen_pairs.add(record.pair_id)
                    wanted.append(record)
                loci_pairs = _write_loci(
                    instances, wanted, planned_loci,
                    max_motifs_per_peak=top_n, max_rows=loci_max_rows, started=started,
                )
                if loci_pairs < 0:
                    loci_pairs = 0
                    _set_aside(planned_loci)
                else:
                    loci_path = planned_loci
                    _record(planned_loci, loci_fp, loci_pairs_exported=loci_pairs)
        else:
            if instances is None:
                print(
                    "[paralome] no motif_instances table found under any comparison; "
                    "skipping the locus export. Everything else is unaffected.",
                    file=sys.stderr, flush=True,
                )
            _set_aside(planned_loci)

        gene_path: Optional[Path] = None
        gene_rows = 0
        gene_sources: Dict[str, Dict[str, Tuple[str, Path]]] = {}
        gene_source_resolution: Dict[str, str] = {}
        gene_annotation_skipped = ""
        if instances is not None and cut_contrasts:
            # Resolved before the reuse check: which differential tables were read
            # is part of what a finished gene table was computed from.
            gene_sources, gene_source_resolution = resolve_peak_statistics(
                project, comparisons, folder=fork["folder"], members=fork["members"],
            )
        if instances is not None and cut_contrasts and not any(gene_sources.values()):
            # See resolve_peak_statistics. Nothing to classify against, so no
            # gene table is written, and that is said.
            gene_annotation_skipped = (
                "no comparison has a differential table to classify peaks against: "
                + "; ".join(f"{c}: {gene_source_resolution.get(c)}" for c in comparisons)
            )
            print(
                f"[paralome] skipping the gene annotation -- {gene_annotation_skipped}. The locus export "
                "and every other table are unaffected; re-run `atac motifs` so each comparison's "
                "checkpoint records the differential table it classified against.",
                file=sys.stderr, flush=True,
            )
            _set_aside(planned_genes)
        elif instances is not None and cut_contrasts:
            genes_fp = _fingerprint(dict(
                detail_settings, output="genes",
                peak_statistics={
                    c: {k: [label, *_file_identity(path)] for k, (label, path) in sorted(v.items())}
                    for c, v in sorted(gene_sources.items())
                },
            ))
            gene_path = planned_genes
            if _skip(planned_genes, genes_fp):
                gene_rows = int((finished.get(_key(planned_genes)) or {}).get("gene_annotation_rows", 0))
            else:
                gene_rows = _write_gene_annotation(
                    project, comparisons, contrast_rows, instances, planned_genes,
                    alpha=alpha, alpha_metric=alpha_metric,
                    only_pairs=changed_ids, started=started,
                    max_motifs_per_peak=top_n, peak_sources=gene_sources,
                )
                _record(planned_genes, genes_fp, gene_annotation_rows=gene_rows)
        else:
            _set_aside(planned_genes)

        result: Dict[str, object] = {
            # One statistical fork's record; the run-level keys (schema,
            # completion, reuse) are on the checkpoint that holds every fork.
            "method": METHOD,
            "project": str(Path(project).expanduser().resolve()),
            "comparisons": list(comparisons),
            "comparison_order_is_progression": True,
            "anchor": anchor or "",
            "anchor_mode": "anchored" if pattern is not None else "unguided",
            "directions_analysed": list(directions),
            "alpha": alpha,
            "alpha_metric": alpha_metric,
            "threads": int(threads),
            "strata_per_comparison": {c: sorted(v) for c, v in strata_seen.items()},
            "pairs_examined": len(records),
            "pairs_reported": len(ranked),
            "motif_partner_contrasts": str(contrast_path),
            "motif_partner_genes": str(gene_path) if gene_path else "",
            "gene_annotation_rows": gene_rows,
            "gene_annotation_skipped": gene_annotation_skipped,
            "contrasts_tested": [{"control_state": a, "treatment_state": b}
                                 for a, b in contrasts],
            # The gate (passes_gate), the scan and the contrast cut all use a
            # strict comparison; this said "<=".
            "detail_cut": f"loci and genes: pairs whose change has {alpha_metric} < {alpha:g} (--alpha)",
            "change_significance_column": change_field(alpha_metric),
            "motif_partner_shift": str(shift_path),
            "motif_partner_loci": str(loci_path) if loci_path else "",
            "motif_instances_used": str(instances) if instances else "",
            "top_pairs": int(top_pairs),
            "changed_pairs_total": total_changed,
            "pairs_detailed": len(changed_ids),
            "loci_pairs_exported": loci_pairs,
            "max_motifs_per_peak": int(top_n),
            "max_motifs_per_peak_recorded": bool(top_n_recorded),
            "gene_annotation_peak_statistics": {
                c: {label: str(path) for _k, (label, path) in sorted(v.items())}
                for c, v in sorted(gene_sources.items())
            },
            "gene_annotation_source_resolution": gene_source_resolution,
        }
        result["fork"] = fork["key"]
        result["statistic_folder"] = fork["folder"]
        result["differential_comparison"] = fork["label"]
        result["differential_comparisons"] = dict(fork["members"])
        return result

    fork_results: Dict[str, Dict[str, object]] = {}
    for fork in forks:
        # results/<folder>/ and <prefix>_<comparison>_..., as `atac motifs`
        # lays out the tables this reads; the unfoldered layout stays flat.
        fork["results"] = results / str(fork["folder"]) if fork["folder"] else results
        fork["file_prefix"] = (
            "_".join(part for part in (file_prefix, str(fork["label"])) if part)
            if fork["folder"] else file_prefix
        )
        fork_results[str(fork["key"])] = _fork(fork)
    result = {
        "schema": "paralome.atac.motif_partner_shift.v4",
        "method": METHOD,
        "stage": "motif_partner_shift",
        "complete": True,
        "interpretation": (
            "Comparisons share one peak atlas and one motif scan, so each pair's "
            "co-occurring peak set is identical across them. A difference between "
            "comparisons is a difference in WHICH of those fixed peaks became "
            "significant -- never a motif moving or a peak widening. Selectivity "
            "(capture / background) is the scale-free reading; a value near 1 means "
            "the pair's peaks opened at the same rate as peaks generally."
        ),
        "project": str(Path(project).expanduser().resolve()),
        "comparisons": list(comparisons),
        "anchor": anchor or "",
        "alpha": alpha,
        "alpha_metric": alpha_metric,
        # One fork per statistic folder (and, where the runs carry
        # several comparisons of it, per comparison): each statistic is its
        # own result, never pooled with another's tables.
        "forks": fork_results,
        "forks_skipped": list(forks_skipped),
        "output_manifest": str(manifest_path),
        "overwrite": bool(overwrite),
        "reused_outputs": reused,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    # The tables for OmicView, the only thing a client receives: <prefix>_motif_partners_output.zip.
    result["output_zip"] = write_motif_partner_zip(fork_results, out, file_prefix, comparisons=list(comparisons),
                                                   anchor=anchor or "", alpha=alpha, alpha_metric=alpha_metric)
    checkpoint_path = checkpoints / "motif_partner_shift.json"
    write_json(checkpoint_path, result, atomic=True, stamp=True)
    for key, fork_result in fork_results.items():
        print(
            f"Motif partner shift [{key or 'results/'}]: {fork_result['pairs_reported']} "
            f"pair(s) of {fork_result['pairs_examined']} examined",
            flush=True,
        )
    print(f"Motif partner shift: done ({time.monotonic() - started:.1f}s)", flush=True)
    return result


#: The motif-partner zip's manifest kind and schema (OmicView's Motif partners view).
MOTIF_PARTNER_ZIP_KIND = "paralome.motif_partners"
MOTIF_PARTNER_ZIP_SCHEMA = 1
_ZIP_TABLES = (("shift", "motif_partner_shift"), ("contrasts", "motif_partner_contrasts"),
               ("genes", "motif_partner_genes"), ("loci", "motif_partner_loci"))


def write_motif_partner_zip(fork_results: Mapping[str, Mapping[str, object]], out: Path, prefix: str, *,
                            comparisons: Sequence[str], anchor: str, alpha: float, alpha_metric: str) -> Optional[str]:
    """Every fork's tables as Parquet, with a manifest, in ``<out>/<prefix>_motif_partners_output.zip`` (stored)."""
    import shutil
    import zipfile

    from paralome.browser.results import _Parquet
    from paralome.version import __version__

    staging = Path(out) / "results" / ".motif_partners_zip"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    forks = []
    parquet = _Parquet(1)
    try:
        for index, (key, fork) in enumerate(sorted(fork_results.items()), start=1):
            record: Dict[str, object] = {"fork": key or "", "pairs_reported": fork.get("pairs_reported"),
                                         "pairs_examined": fork.get("pairs_examined")}
            for part, name in _ZIP_TABLES:
                source = str(fork.get(name) or "")
                if source and Path(source).is_file() and Path(source).stat().st_size:
                    target = f"tables/{index:03d}_{part}.parquet"
                    record[part] = {"path": target, "rows": parquet.write(Path(source), staging / target)}
            forks.append(record)
    finally:
        parquet.close()
    manifest = {"kind": MOTIF_PARTNER_ZIP_KIND, "schema": MOTIF_PARTNER_ZIP_SCHEMA, "program": "Paralome",
                "program_version": __version__, "project": prefix, "comparisons": list(comparisons),
                "anchor": anchor, "detail_cut": f"loci and genes: pairs whose change has {alpha_metric} < {alpha:g}",
                "forks": forks}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    target = Path(out) / f"{prefix}_motif_partners_output.zip"
    partial = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging).as_posix())
    partial.replace(target)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Motif partner shift: {target}", flush=True)
    return str(target)


#: Fallback when a motif run predates the checkpoint recording it. Matches the
#: `atac motifs` default; a run that used another value cannot be detected,
#: which is why it is announced rather than assumed silently.
_DEFAULT_MAX_MOTIFS_PER_PEAK = 25


def resolve_max_motifs_per_peak(
    project: Union[str, Path], comparisons: Sequence[str],
) -> Tuple[int, bool]:
    """The top-N rule the pair statistics were built with, and whether it was found.

    `cooccurring_peak_count` -- the |S| behind every selectivity number
    -- counts only peaks where BOTH motifs are among the strongest N in that
    peak. `motif_instances` holds every hit with no such restriction, so a raw
    join reports co-occurrences the statistics never counted. Reading the value
    the run actually used is the only way for a locus export to describe the
    same peak set its q-values do.
    """
    root = Path(project).expanduser().resolve()
    for comparison in comparisons:
        for path in sorted((root / comparison).rglob("motif_analysis.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            value = payload.get("max_motifs_per_peak")
            if value:
                return int(value), True
    return _DEFAULT_MAX_MOTIFS_PER_PEAK, False


def _stratum_key(label: str) -> str:
    """A stratum label reduced to the form both naming routes agree on.

    The motif checkpoint records the stratum as the differential
    named it ("B intermediate"), while a filename carries a sanitised form
    ("B_intermediate"). Keyed on the raw text, a comparison resolved one way
    and another resolved the other way would share no stratum, and every
    gained/lost count would silently be zero.
    """
    return re.sub(r"[^0-9a-z]+", "_", str(label or "").lower()).strip("_") or "(all)"


def _motif_checkpoint(comparison_dir: Path) -> Optional[Path]:
    """The motif_analysis.json beside the results directory the pair tables
    were read from -- the run whose enrichment this analysis re-reads."""
    for rel in _MOTIF_RESULT_DIRS:
        candidate = stage_checkpoint_path(comparison_dir / rel.parent, "motif_analysis")
        if candidate.is_file():
            return candidate
    return None


def _relocate(recorded: Path, comparison_dir: Path) -> Optional[Path]:
    """A recorded path, or the one file of the same name under the comparison
    when the project has been moved since the motif run wrote it."""
    if recorded.is_file():
        return recorded
    matches = sorted(comparison_dir.rglob(recorded.name))
    return matches[0] if len(matches) == 1 else None


def resolve_peak_statistics(
    project: Union[str, Path], comparisons: Sequence[str],
    *, folder: str = "", members: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, Tuple[str, Path]]], Dict[str, str]]:
    """{comparison: {stratum_key: (label, table)}} -- ONE differential table per
    stratum, the one the motif enrichment classified against.

    The motif checkpoint records, per enrichment, the exact table it read
    (``differential_statistics``), and that is what is used, for one
    statistical fork -- the entries of statistic ``folder`` whose differential
    comparison is ``members[comparison]``. An entry that records no folder
    belongs to the fork ("", "") (``folder`` "" and no ``members``). A run
    whose checkpoint records no table has none: anything ambiguous is refused
    rather than guessed.

    Returns the per-comparison sources and how each comparison was resolved.
    """
    root = Path(project).expanduser().resolve()
    sources: Dict[str, Dict[str, Tuple[str, Path]]] = {}
    how: Dict[str, str] = {}
    for comparison in comparisons:
        comparison_dir = root / comparison
        recorded: Dict[str, Tuple[str, Path]] = {}
        checkpoint = _motif_checkpoint(comparison_dir)
        if checkpoint is not None:
            try:
                payload = json.loads(checkpoint.read_text())
            except (OSError, ValueError) as exc:
                raise MotifPartnerError(
                    f"{comparison}: cannot read motif checkpoint {checkpoint}: {exc}"
                ) from exc
            wanted = (members or {}).get(comparison)
            for entry in dict(payload.get("comparisons") or {}).values():
                if not isinstance(entry, Mapping) or not entry.get("differential_statistics"):
                    continue
                if str(entry.get("folder") or "") != str(folder or ""):
                    continue
                if wanted is not None and "folder" in entry and str(entry.get("comparison") or "") != wanted:
                    continue
                label = str(entry.get("stratum") or "")
                table = _relocate(Path(str(entry["differential_statistics"])), comparison_dir)
                if table is None:
                    raise MotifPartnerError(
                        f"{comparison}: the motif run classified stratum {label or '(all)'!r} "
                        f"against {entry['differential_statistics']}, which no longer exists "
                        f"(and no single file of that name is under {comparison_dir}). The "
                        "gene annotation must read the same table the enrichment did; "
                        "restore it or re-run `atac motifs` for this comparison."
                    )
                key = _stratum_key(label)
                if key in recorded and recorded[key][1] != table:
                    raise MotifPartnerError(
                        f"{comparison}: motif checkpoint {checkpoint} records two different "
                        f"differential tables for stratum {label or '(all)'!r}"
                        + (f" of {folder}/{wanted}" if folder else "") + "."
                    )
                recorded[key] = (label or "(all)", table)
        sources[comparison] = recorded
        if recorded:
            how[comparison] = f"motif checkpoint {checkpoint}"
        elif checkpoint is None:
            how[comparison] = f"no table: no motif checkpoint under {comparison_dir}"
        else:
            how[comparison] = (
                f"no table: motif checkpoint {checkpoint} records no differential_statistics"
                + (f" for {folder}" + (f"/{wanted}" if wanted else "") if folder else "")
            )
    # Some comparisons with tables and some without is refused: the gained/lost counts would be
    # made against nothing on one side. None at all is said loudly, and the caller skips the gene
    # annotation rather than write an empty table that reads as an answer.
    without = [c for c in comparisons if not sources.get(c)]
    if without and len(without) < len(comparisons):
        raise MotifPartnerError(
            "No differential table to classify peaks against for " + ", ".join(without)
            + " (" + "; ".join(how[c] for c in without) + "), while "
            + ", ".join(c for c in comparisons if sources.get(c)) + " have one. The gene "
            "annotation reads the machine statistics table the motif enrichment classified "
            "against; re-run `atac motifs` for the comparisons without one so their checkpoints record it."
        )
    return sources, how


def _peak_buckets(
    sources: Mapping[str, Mapping[str, Tuple[str, Path]]], wanted_peaks: set,
    beat: Optional["_Heartbeat"] = None, beat_started: float = 0.0, *, alpha: float = 0.05, alpha_metric: str = "p",
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """(comparison, stratum_key) -> peak_id -> "up" / "down", for wanted peaks only.

    Restricted to the peaks that matter before anything is read, because the
    per-stratum tables carry the whole atlas -- 170,109 rows each, 33 of them.
    Keeping only the co-occurring peaks of the pairs that actually changed
    turns that from gigabytes into a bounded lookup.

    Classification matches ``atac motifs`` exactly: tested and significant and
    positive is "up", tested and significant and negative is "down", everything
    else is absent from the map. Reproduced rather than imported because
    motifs.py works in peak INDICES against its own peak_rows ordering, and
    this needs peak IDs.

    Reads exactly the tables ``resolve_peak_statistics`` chose --
    one per stratum -- instead of every model table it could find.
    """
    buckets: Dict[Tuple[str, str], Dict[str, str]] = {}
    for comparison, strata in sources.items():
        for key, (label, path) in sorted(strata.items()):
            found: Dict[str, str] = {}
            try:
                handle, reader, idx = _open_indexed(path)
            except (OSError, StopIteration) as exc:
                raise MotifPartnerError(
                    f"{comparison}/{label}: cannot read peak statistics {path}: {exc}"
                ) from exc
            rows_seen = 0
            try:
                value_column = "q_value" if alpha_metric == "q" else "p_value"
                needed = {"peak_id", "tested", value_column, "log2_fold_change"}
                if not needed <= set(idx):
                    # An unreadable table is an error, not a stratum where nothing changed: it is the
                    # table the enrichment classified on. `tested` is required, as
                    # motifs._classify_peaks_for_enrichment requires it.
                    raise MotifPartnerError(
                        f"{path.name} is missing column(s): "
                        + ", ".join(sorted(needed - set(idx)))
                        + ". The gene annotation classifies peaks exactly as the motif "
                        "enrichment did, from the differential's machine statistics table."
                    )
                pid, sig = idx["peak_id"], idx[value_column]
                lfc = idx["log2_fold_change"]
                tested = idx["tested"]
                for row in reader:
                    rows_seen += 1
                    if beat is not None and rows_seen % 500_000 == 0:
                        beat.say(
                            f"Motif partner shift: peak statistics, {comparison}/{label}, "
                            f"{rows_seen:,} row(s)"
                            + _read_progress(handle, path, rows_seen,
                                             time.monotonic() - beat_started)
                        )
                    peak = row[pid]
                    if peak not in wanted_peaks:
                        continue
                    if row[tested].strip().lower() not in ("true", "1", "yes"):
                        continue
                    try:                                   # at --alpha, as atac motifs selects
                        if not float(row[sig]) < alpha:
                            continue
                    except ValueError:
                        continue
                    try:
                        value = float(row[lfc])
                    except (TypeError, ValueError):
                        continue
                    found[peak] = "up" if value > 0 else "down"
            finally:
                handle.close()
            buckets[(comparison, key)] = found
    return buckets


@dataclass
class _TopMotifSelection:
    """The co-occurrence rule the pair statistics used, applied to motif_instances.

    by_motif: motif name -> peaks where that motif is among the peak's top N.
    coords:   peak -> (chrom, start, end, nearest_gene_id, nearest_gene_name,
              tss_gene_id, tss_gene_name), for peaks holding a wanted motif.
    spans:    motif name -> peak -> (start, end, strand) of that motif's BEST
              hit in the peak, the hit motifs.py measures distance from.
    """

    by_motif: Dict[str, set]
    coords: Dict[str, tuple]
    spans: Dict[str, Dict[str, tuple]]


def select_top_motifs(
    instances_path: Path,
    wanted_motifs: set,
    *,
    max_motifs_per_peak: int,
    started: float,
    beat: Optional["_Heartbeat"] = None,
) -> _TopMotifSelection:
    """Per peak, the strongest max(2, N) DISTINCT motifs -- motifs.py's rule.

    Shared by the locus export and the gene annotation.

    motifs._build_pair_features ranks each peak's motifs by their best p (one
    entry per distinct motif, ties in motif order) and forms pairs only from
    the strongest max(2, N). That defines cooccurring_peak_count, the |S| behind
    every change_q here, so both readers use exactly this rule: distinct
    motifs, not hit rows (a motif with several hits in a peak takes one slot),
    and each motif's STRONGEST hit's span, so centre_distance_bp is measured
    from the right site.

    Bounded memory: each peak keeps at most N motifs. A motif evicted from a
    full peak can never re-enter with a worse p (every kept motif already
    beats it and only improves), and re-enters correctly with a better one, so
    the result equals sorting every distinct motif in the peak.
    """
    keep = max(2, int(max_motifs_per_peak))
    tops: Dict[str, Dict[object, Tuple[float, int]]] = {}
    worst: Dict[str, Tuple[float, int]] = {}
    names_of: Dict[str, Dict[object, str]] = {}
    rank: Dict[object, int] = {}
    coords: Dict[str, tuple] = {}
    best_span: Dict[str, Dict[str, tuple]] = {}

    handle, reader, idx = _open_indexed(instances_path)
    try:
        needed = {"peak_id", "motif_name", "motif_p"}
        missing = sorted(needed - set(idx))
        if missing:
            raise MotifPartnerError(
                f"{instances_path.name} is missing column(s): {', '.join(missing)}"
            )
        peak_i, name_i, p_i = idx["peak_id"], idx["motif_name"], idx["motif_p"]
        id_i = idx.get("motif_id")
        score_i = idx.get("motif_score")
        place_i = [idx.get(c) for c in ("chrom", "peak_start", "peak_end",
                                        "nearest_gene_id", "nearest_gene_name",
                                        "tss_gene_id", "tss_gene_name")]
        span_i = [idx.get(c) for c in ("motif_start", "motif_end", "strand")]

        def _get(row, i):
            return row[i] if i is not None and i < len(row) else ""

        seen = 0
        for row in reader:
            seen += 1
            if seen % 2_000_000 == 0 and beat is not None:
                beat.say(
                    f"Motif partner shift: motif instances, {seen:,} row(s) read"
                    + _read_progress(handle, instances_path, seen,
                                     time.monotonic() - started)
                )
            try:
                pvalue = float(row[p_i])
            except (TypeError, ValueError, IndexError):
                continue
            if pvalue != pvalue:
                continue
            peak, name = row[peak_i], row[name_i]
            # Identity is the motif id when the table carries it, so two
            # library entries sharing a name hold two slots as they do in
            # motifs.py; the name is what pairs are reported under.
            identity = _get(row, id_i) or name
            order = rank.get(identity)
            if order is None:
                order = rank[identity] = len(rank)
            key = (pvalue, order)
            kept = tops.get(peak)
            if kept is None:
                tops[peak] = {identity: key}
                worst[peak] = key
                names_of[peak] = {identity: name}
            else:
                current = kept.get(identity)
                if current is not None:
                    if key < current:
                        kept[identity] = key
                        if current == worst[peak]:
                            worst[peak] = max(kept.values())
                elif len(kept) < keep:
                    kept[identity] = key
                    names_of[peak][identity] = name
                    if key > worst[peak]:
                        worst[peak] = key
                elif key < worst[peak]:
                    evict = worst[peak]
                    for other, other_key in kept.items():
                        if other_key == evict:
                            del kept[other]
                            del names_of[peak][other]
                            break
                    kept[identity] = key
                    names_of[peak][identity] = name
                    worst[peak] = max(kept.values())
            if name in wanted_motifs:
                try:
                    score = float(_get(row, score_i) or "-inf")
                except ValueError:
                    score = float("-inf")
                spans = best_span.setdefault(name, {})
                previous = spans.get(peak)
                if previous is None or pvalue < previous[0] or (
                    pvalue == previous[0] and score > previous[1]
                ):
                    spans[peak] = (pvalue, score, _get(row, span_i[0]),
                                   _get(row, span_i[1]), _get(row, span_i[2]))
                if peak not in coords:
                    coords[peak] = tuple(_get(row, i) for i in place_i)
    finally:
        handle.close()

    by_motif: Dict[str, set] = {}
    for peak, kept in tops.items():
        labels = names_of[peak]
        for identity in kept:
            name = labels[identity]
            if name in wanted_motifs:
                by_motif.setdefault(name, set()).add(peak)
    spans_out = {
        name: {peak: value[2:] for peak, value in peaks.items()}
        for name, peaks in best_span.items()
    }
    return _TopMotifSelection(by_motif=by_motif, coords=coords, spans=spans_out)


@contextmanager
def _atomic_open(path: Path):
    """Write to <name>.partial and rename into place only on success.

    A file under the final name always means the write completed, so a run
    killed mid-write (SLURM wall time, or an OOM) cannot leave a truncated file
    that --no-overwrite would reuse as finished.
    """
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("w", newline="") as handle:
            yield handle
        os.replace(partial, path)
    except BaseException:
        try:
            partial.unlink()
        except OSError:
            pass
        raise


def _write_gene_annotation(
    project: Union[str, Path],
    comparisons: Sequence[str],
    contrast_rows: Sequence[Mapping[str, object]],
    instances_path: Path,
    destination: Path,
    *,
    alpha: float,
    alpha_metric: str,
    only_pairs: set,
    started: float,
    max_motifs_per_peak: int = _DEFAULT_MAX_MOTIFS_PER_PEAK,
    peak_sources: Optional[Mapping[str, Mapping[str, Tuple[str, Path]]]] = None,
) -> int:
    """Which genes sit at the peaks a changed pair gained or lost.

    Restricted to pairs whose change is significant, and within those, to the
    peaks that actually changed state -- co-occurring peaks that are in the
    treatment comparison's significant bucket and not the control's, or the
    reverse. A pair's full co-occurring set is the same in every comparison
    (one atlas, one scan), so listing all of it would describe the pair's
    territory rather than the change; the gained/lost subset is the part the
    contrast is actually about.

    Both gene assignments are carried. `tss_gene_*` is a promoter overlap and
    is blank for distal peaks; `nearest_gene_*` always has a value but is a
    proximity guess on a distal peak. They disagree often enough that reporting
    one would be choosing an answer on the reader's behalf.

    "co-occurring" means what it means in the statistics -- both motifs among
    the peak's top max(2, N) -- via select_top_motifs, and each stratum is
    counted once, against the one differential table its enrichment used
    (resolve_peak_statistics).
    """
    column = change_field(alpha_metric)
    changed_pairs = {}
    for row in contrast_rows:
        if row["pair_id"] not in only_pairs:
            continue
        if not change_is_significant(row, column, alpha):
            continue
        changed_pairs.setdefault(row["pair_id"], []).append(row)
    if not changed_pairs:
        # Still write the header-only file, so an empty answer is
        # on disk as one rather than as a missing file.
        with _atomic_open(destination) as out:
            csv.writer(out, lineterminator="\n").writerow(_GENE_FIELDS)
        return 0

    beat = _Heartbeat(started)
    beat.say(f"Motif partner shift: annotating genes for {len(changed_pairs)} changed pair(s)")

    wanted_motifs: set = set()
    for rows in changed_pairs.values():
        wanted_motifs.add(rows[0]["motif_a_name"])
        wanted_motifs.add(rows[0]["motif_b_name"])

    selection = select_top_motifs(
        instances_path, wanted_motifs,
        max_motifs_per_peak=max_motifs_per_peak, started=started, beat=beat,
    )

    cooccurring: Dict[str, Dict[str, tuple]] = {}
    every_peak: set = set()
    for pair_id, rows in changed_pairs.items():
        a = selection.by_motif.get(rows[0]["motif_a_name"]) or set()
        b = selection.by_motif.get(rows[0]["motif_b_name"]) or set()
        shared = a & b
        if shared:
            cooccurring[pair_id] = {peak: selection.coords.get(peak, ("",) * 7)
                                    for peak in shared}
            every_peak |= shared

    if peak_sources is None:
        peak_sources, _how = resolve_peak_statistics(project, comparisons)
    buckets = (_peak_buckets(peak_sources, every_peak, beat, started, alpha=alpha, alpha_metric=alpha_metric)
               if every_peak else {})
    labels: Dict[str, str] = {}
    strata_of: Dict[str, set] = {}
    for comparison, strata in peak_sources.items():
        strata_of[comparison] = set(strata)
        for key, (label, _path) in strata.items():
            labels.setdefault(key, label)

    # One row per (pair, contrast, GENE), which is what was asked for.
    #
    # This was written one row per (pair, contrast, stratum, peak) -- a dozen
    # strata and ~1,000 co-occurring peaks per pair on top of thousands of
    # changed pairs and three contrasts, which is a 23 GB file. The peak counts belong in columns,
    # not in rows: peak-level coordinates are already in the locus export, and
    # nobody filters a 23 GB table in a spreadsheet.
    # Aggregated PER PAIR and flushed, not accumulated.
    #
    # The collapse to one row per gene traded a 23 GB file for an in-memory
    # dict, and on a real project that dict reached 80 GB -- roughly 44 million
    # (pair, contrast, gene) entries, each a dict with two sets. But the key
    # begins with pair_id, so no gene ever merges across pairs: one pair can be
    # aggregated, written and discarded before the next is touched. That holds
    # a few hundred entries instead of tens of millions.
    #
    # The cost is that rows are grouped by pair rather than globally sorted by
    # magnitude. Within a pair they are still strongest-first, and a global
    # sort on a file this size belongs in pandas rather than in RAM here.
    written = 0
    with _atomic_open(destination) as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(_GENE_FIELDS)
        done = 0
        for pair_id, rows in changed_pairs.items():
            done += 1
            if done % 200 == 0:
                beat.say(
                    f"Motif partner shift: genes, {done:,}/{len(changed_pairs):,} pair(s), "
                    f"{written:,} row(s)"
                )
            peaks = cooccurring.get(pair_id)
            if not peaks:
                continue
            aggregated: Dict[tuple, Dict[str, object]] = {}
            for row in rows:
                control, treatment = row["control_state"], row["treatment_state"]
                direction = row["direction"]
                # Only strata BOTH comparisons have, as in
                # contrast_between. A stratum missing from the treatment used
                # to read as an empty bucket, so every peak significant in the
                # control was counted as "lost".
                shared_strata = strata_of.get(control, set()) & strata_of.get(treatment, set())
                for stratum in sorted(shared_strata):
                    before = buckets.get((control, stratum)) or {}
                    after = buckets.get((treatment, stratum)) or {}
                    beat.tick()
                    for peak, info in peaks.items():
                        was = before.get(peak) == direction
                        now = after.get(peak) == direction
                        if was == now:
                            continue
                        key = (direction, control, treatment, info[3], info[4])
                        entry = aggregated.get(key)
                        if entry is None:
                            entry = aggregated[key] = {
                                "motif_a": row["motif_a_name"],
                                "motif_b": row["motif_b_name"],
                                "partner": row.get("partner", ""),
                                "gained": 0, "lost": 0, "strata": set(), "tss": set(),
                            }
                        entry["gained" if now else "lost"] += 1
                        entry["strata"].add(labels.get(stratum, stratum))
                        if info[6]:
                            entry["tss"].add(info[6])
            for key, entry in sorted(
                aggregated.items(),
                key=lambda item: -(item[1]["gained"] + item[1]["lost"]),
            ):
                direction, control, treatment, gene_id, gene_name = key
                writer.writerow([
                    pair_id, direction, entry["motif_a"], entry["motif_b"],
                    entry["partner"], control, treatment, gene_id, gene_name,
                    ";".join(sorted(entry["tss"])),
                    entry["gained"], entry["lost"],
                    len(entry["strata"]), ";".join(sorted(entry["strata"])),
                ])
                written += 1
    return written


_GENE_FIELDS = ["pair_id", "direction", "motif_a_name", "motif_b_name", "partner",
                "control_state", "treatment_state",
                "nearest_gene_id", "nearest_gene_name", "tss_gene_names",
                "peaks_gained", "peaks_lost", "strata_with_change", "strata"]


def _write_loci(
    instances_path: Path,
    records: Sequence[PairRecord],
    destination: Path,
    *,
    max_motifs_per_peak: int,
    max_rows: int,
    started: float,
) -> int:
    """Emit the genomic locations where each top pair's two motifs co-occur.

    This is the file that answers "which loci, and near which genes". The
    motif scan writes every hit's `chrom`, `peak_start`, `peak_end`, its own
    span within the peak, and the nearest/TSS gene -- one row per motif per
    peak. Joining that table to itself on `peak_id` recovers, for a given pair,
    every peak where both motifs are present.

    Bounded to the top N pairs on purpose. The instances table is the largest
    thing the motif stage writes, and an unbounded join over every significant
    pair produces a file nobody opens.
    """
    wanted_motifs = set()
    for record in records:
        wanted_motifs.add(record.motif_a)
        wanted_motifs.add(record.motif_b)

    beat = _Heartbeat(started)
    beat.say(
        f"Motif partner shift: reading motif instances for {len(records)} pair(s), "
        f"top {max(2, max_motifs_per_peak)} motif(s) per peak"
    )

    # The SAME top-N-by-p selection the pair statistics used.
    #
    # Without it this join answers a different question than the analysis it
    # annotates. `cooccurring_peak_count` counts only peaks where both motifs
    # are among the strongest N in that peak; motif_instances holds every hit.
    # Joining it raw gave a 291 GB file listing co-occurrences the statistics
    # never counted -- ~40x too many rows on a real project, and rows that did
    # not correspond to the |S| behind their own q-values.
    #
    # Through select_top_motifs, shared with the gene annotation,
    # which ranks DISTINCT motifs by their best hit as motifs.py does rather
    # than ranking hit rows.
    selection = select_top_motifs(
        instances_path, wanted_motifs,
        max_motifs_per_peak=max_motifs_per_peak, started=started, beat=beat,
    )
    by_motif, coords, spans = selection.by_motif, selection.coords, selection.spans

    # Size the output before writing it. A ten-hour write discovered only by
    # looking at the filesystem is not an acceptable way to learn this.
    pair_peaks: List[Tuple[PairRecord, set]] = []
    estimated = 0
    for record in records:
        a = by_motif.get(record.motif_a)
        b = by_motif.get(record.motif_b)
        if not a or not b:
            continue
        shared = a & b
        if shared:
            pair_peaks.append((record, shared))
            estimated += len(shared)
    print(
        f"Motif partner shift: locus export is {estimated:,} row(s), "
        f"~{estimated * 150 / 1e9:.2f} GB ({time.monotonic() - started:.1f}s)",
        flush=True,
    )
    if max_rows and estimated > max_rows:
        print(
            f"[paralome] REFUSING to write {estimated:,} locus rows (~"
            f"{estimated * 150 / 1e9:.1f} GB), which exceeds --loci-max-rows "
            f"{max_rows:,}. Narrow it with --top-pairs N, tighten --alpha, or "
            f"raise --loci-max-rows if you genuinely want a file this size. "
            f"Everything else has already been written.",
            file=sys.stderr, flush=True,
        )
        return -1

    fields = [
        "pair_id", "motif_a_name", "motif_b_name", "peak_id", "chrom",
        "peak_start", "peak_end", "motif_a_start", "motif_a_end", "motif_a_strand",
        "motif_b_start", "motif_b_end", "motif_b_strand", "centre_distance_bp",
        "nearest_gene_name",
    ]
    written = 0
    with _atomic_open(destination) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        emitted = 0
        for record, shared in pair_peaks:
            emitted += 1
            if emitted % 200 == 0:
                beat.say(
                    f"Motif partner shift: loci, {emitted:,}/{len(pair_peaks):,} pair(s), "
                    f"{written:,} row(s)"
                )
            beat.tick()
            spans_a = spans.get(record.motif_a) or {}
            spans_b = spans.get(record.motif_b) or {}
            for peak_id in sorted(shared):
                place = coords.get(peak_id, ("",) * 7)
                a = spans_a.get(peak_id, ("", "", ""))
                b = spans_b.get(peak_id, ("", "", ""))
                try:
                    centre_a = (float(a[0]) + float(a[1])) / 2
                    centre_b = (float(b[0]) + float(b[1])) / 2
                    distance = f"{abs(centre_b - centre_a):.1f}"
                except (TypeError, ValueError):
                    distance = ""
                writer.writerow({
                    "pair_id": record.pair_id,
                    "motif_a_name": record.motif_a,
                    "motif_b_name": record.motif_b,
                    "peak_id": peak_id,
                    "chrom": place[0], "peak_start": place[1], "peak_end": place[2],
                    "motif_a_start": a[0], "motif_a_end": a[1], "motif_a_strand": a[2],
                    "motif_b_start": b[0], "motif_b_end": b[1], "motif_b_strand": b[2],
                    "centre_distance_bp": distance,
                    "nearest_gene_name": place[4] or place[6],
                })
                written += 1
    return written
