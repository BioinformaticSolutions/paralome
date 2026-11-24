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

import shutil
from pathlib import Path

from paralome.align.verify import StarVerifyReport, print_star_verify_report, verify_star_alignment


def run_align_verify(workdir: Path) -> StarVerifyReport:
    """Verify a STAR alignment stage; the body of ``align verify``.

    ``--json-only`` prints ``json.dumps(report, indent=2, sort_keys=True)``,
    the same contract as ``dedup verify``.
    """
    report = verify_star_alignment(workdir, write_json=True)
    _preserve_star_logs(Path(workdir))
    _collect_metrics(Path(workdir))
    print_star_verify_report(report)
    return report


def _preserve_star_logs(workdir: Path) -> None:
    """Copy each sample's Log.final.out out of temp/ and into logs/.

    It is the record of how a run actually went, and temp/ is the one place
    whose name says it is disposable. Copied rather than moved: STAR owns
    temp/ and a later re-run of one sample should not find its own scratch
    rearranged underneath it.
    """
    source, target = workdir / "temp", workdir / "logs"
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for log in sorted(source.glob("*.Log.final.out")):
        try:
            shutil.copy2(log, target / log.name)
        except OSError:
            # One unreadable log must not fail a stage that aligned every read
            # it was given. The metrics collector reports it as missing.
            continue


def _collect_metrics(workdir: Path) -> None:
    """Write alignment_metrics.tsv/json -- the per-sample mapping table.

    ``paralome.align.metrics.collect_star_metrics`` parses Log.final.out into
    input reads, unique and multimapped counts and percentages, and the
    unmapped breakdown. Two consumers read the result:
    ``pipeline.state.summarize_alignment`` (the JSON; the mapping rate in
    `status --detail`) and ``pipeline.sequencing_qc`` (the TSV).

    Failing here is never fatal: the alignment itself succeeded, and a report
    that cannot be written must not undo it.
    """
    try:
        from paralome.align.metrics import collect_star_metrics

        collect_star_metrics(workdir, write_outputs=True)
    except Exception as exc:  # pragma: no cover - reported, never raised
        print(f"  note: alignment metrics could not be collected: {exc}")
