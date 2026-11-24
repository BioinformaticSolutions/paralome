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

"""What every differential result shares: how a missing value is written, and removing a previous run's
files this run did not write. (The report layout, power and pathway exports that lived here are gone: OmicView
draws them from the statistics output.)"""

from __future__ import annotations

from typing import List



#: Written wherever a value is genuinely absent or uncalculable. A real 0.0 is
#: never rendered this way -- only an actually-missing number.
MISSING_VALUE = "."



def remove_unwritten_results(candidates, written, *, stage: str = "single-cell differential") -> List[str]:
    """Remove each of ``candidates`` that exists and is not in ``written``.

    The single-cell results folder is flat and shared by every run of a
    comparison. A writer that has nothing to write -- no gene at p < 0.05 for
    a model, a model no longer requested, no effect size for the power
    summary, a figure not drawn -- simply does not write, so the PREVIOUS
    run's file would stay under exactly the name this run would have used
    (a significant-gene list naming genes as significant beside an all-genes
    table that says they are not).

    ``candidates`` are exact names this writer produces for this file prefix
    and modality stamp, so another stratum's, modality's or command's files
    cannot match. Removing what this run did not reproduce is what makes the
    folder say what the run found; the rule bulk RNA's differential follows
    for "the result being replaced".
    """
    import sys
    from pathlib import Path

    keep = {str(Path(str(path)).resolve()) for path in written if path}
    removed: List[str] = []
    for candidate in candidates:
        path = Path(str(candidate))
        if str(path.resolve()) in keep or not path.is_file():
            continue
        try:
            path.unlink()
        except OSError as error:
            # A file that cannot be removed is said, and loudly: it carries the
            # current run's name and not the current run's answer.
            print(
                f"[paralome] {stage}: could not remove {path}, left by an earlier run "
                f"and not reproduced by this one ({error}). Delete it by hand: it "
                f"does not reflect this result.",
                file=sys.stderr, flush=True,
            )
            continue
        print(
            f"[paralome] {stage}: removed {path.name} -- an earlier run's result "
            f"that this run did not reproduce.",
            file=sys.stderr, flush=True,
        )
        removed.append(str(path))
    return removed
