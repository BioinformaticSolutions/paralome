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

"""What kind of machine this is, and therefore what advice is appropriate.

A genome other accounts cannot see could be fixed with ``chmod o+x ~`` on a
lab workstation; on a university cluster that is an instruction to violate
site policy, and often one the user cannot carry out. So the advice is
platform specific: a closed system (single user or multiple local) versus a
fully distributed HPC system.

THE TWO CHANNELS, and which one survives contact with a real cluster:

    PEER DISCOVERY reads ~/.paralome/config.json out of other accounts' home
    directories, found through passwd. It needs those homes to be traversable
    by strangers, which a managed multi-tenant system will not allow and should
    not. Excellent on a single lab's machine, where everyone already can; not a
    mechanism to build a deployment on.

    THE SHARED REGISTRY is a drop-in directory where each account writes ONE
    file of its own. Nobody reads anybody's home; nobody needs permission to;
    the only shared resource is a directory an administrator creates once. This
    is what works everywhere, and it is what Paralome leads with.

So the registry is the documented mechanism and peer discovery is an
opportunistic bonus -- still used where it happens to work, never advised into
existence, and never the thing a user is told to open their home for.

WHY THIS IS INFERRED RATHER THAN ASKED. A question at setup would be answered
once, by whoever installed it, and then be wrong for everybody who came later.
The signals are unambiguous enough to read directly, and the value is recorded
so a site that disagrees can simply overrule it.
"""

from __future__ import annotations

import shutil
from typing import Any, Mapping, Optional

#: Recorded under ``defaults`` so a site can overrule the inference once.
DEPLOYMENT_KEY = "deployment"

#: One person, their own machine. Nobody to share with, nothing to arrange.
WORKSTATION = "workstation"

#: Several accounts, one administrative domain -- a lab server, a benchtop
#: machine with logins for the group. Colleagues are colleagues, and opening a
#: home to them is a decision the owner is entitled to make.
SHARED = "shared"

#: A managed multi-tenant system, where the other accounts are strangers and
#: home directories are private by policy.
CLUSTER = "cluster"

DEPLOYMENTS = (WORKSTATION, SHARED, CLUSTER)

#: Below this many human accounts, a machine with no scheduler is one person's.
#: Two is the interesting boundary: a single admin account alongside one user
#: is still, in practice, a workstation.
SHARED_ACCOUNT_THRESHOLD = 3


def infer_deployment(config: Optional[Mapping[str, Any]] = None) -> str:
    """Read the machine rather than ask about it.

    A BATCH SCHEDULER IS THE STRONGEST SIGNAL. Nobody installs SLURM on their
    laptop. Its presence means shared, scheduled, managed infrastructure with
    an administrator and a policy -- which is exactly the situation in which
    telling somebody to widen their home directory is wrong. Paralome already
    detects this for ``defaults.backend``, so the fact is on hand.

    Failing that, the number of human accounts separates a lab server from a
    single-user machine. Counting is done through the same passwd walk
    discovery uses, so a machine where enumeration is disabled -- itself a sign
    of a managed directory service -- reports few accounts and is treated as a
    workstation. That errs toward saying less rather than toward giving advice
    about colleagues who may not exist.
    """
    # One decision procedure, in explain_deployment, so the doctor's
    # line and the advice given at genome registration cannot disagree.
    return explain_deployment(config)[0]


def explain_deployment(config: Optional[Mapping[str, Any]] = None) -> tuple[str, str]:
    """The deployment and, in one phrase, why Paralome decided it.

    `paralome setup doctor` prints the same value ``infer_deployment`` uses
    together with its source, so an administrator who disagrees knows which
    signal to overrule (``defaults.deployment``).
    """
    recorded = deployment_of(config)
    if recorded is not None:
        return recorded, "recorded in defaults.deployment"

    backend = ""
    if config:
        defaults = config.get("defaults")
        if isinstance(defaults, Mapping):
            backend = str(defaults.get("backend") or "")
    if backend == "slurm":
        return CLUSTER, "inferred: defaults.backend is slurm"
    if shutil.which("sbatch"):
        return CLUSTER, "inferred: sbatch is on PATH"

    accounts = _human_accounts()
    if accounts >= SHARED_ACCOUNT_THRESHOLD:
        return SHARED, f"inferred: {accounts} human accounts, no scheduler"
    return WORKSTATION, f"inferred: {accounts} human account(s), no scheduler"


def deployment_of(config: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The recorded value, when a site has set one and it is one we know."""
    if not config:
        return None
    defaults = config.get("defaults")
    if not isinstance(defaults, Mapping):
        return None
    found = str(defaults.get(DEPLOYMENT_KEY) or "").strip().lower()
    return found if found in DEPLOYMENTS else None


def _human_accounts() -> int:
    """How many real people have accounts here.

    Uses the same UID floor as discovery, so the two agree about who counts.
    """
    try:
        import pwd
    except ImportError:  # pragma: no cover - Windows
        return 1
    from paralome.setup.genomes import FIRST_HUMAN_UID

    try:
        accounts = pwd.getpwall()
    except Exception:  # pragma: no cover - no passwd database
        return 1
    return sum(
        1 for entry in accounts
        if entry.pw_uid >= FIRST_HUMAN_UID and entry.pw_uid < 65534
    )


def may_open_own_home(deployment: str) -> bool:
    """Whether widening your home is a reasonable thing to suggest.

    False on a cluster. Not because the command would fail -- it usually
    succeeds -- but because succeeding is the problem: it makes one account
    readable to every stranger with a login, against a policy the user did not
    write and may be disciplined for breaking. Paralome is not in a position to
    weigh that, so it does not raise the option.
    """
    return deployment != CLUSTER


def sharing_advice(deployment: str, registry: Any) -> str:
    """What to do about a genome nobody else can see, for this kind of machine.

    One place, so the registration report, the doctor and the manual cannot
    drift apart about it.
    """
    if deployment == WORKSTATION:
        return (
            "This machine has no other users, so there is nobody for this "
            "genome to be visible to.\n"
            "Nothing needs arranging."
        )

    registry_line = (
        f"Publish it to the shared registry, which is {registry}.\n"
        "Every account writes one file of its own there; nobody reads anybody's\n"
        "home directory, and no home needs to be opened for it to work."
    )

    if deployment == CLUSTER:
        return (
            "On a shared cluster the registry is how genomes are published.\n"
            "\n"
            f"{registry_line}\n"
            "\n"
            "Paralome will not suggest opening your home directory here. Other\n"
            "accounts on a managed system are strangers, and a site that keeps\n"
            "homes private means it -- that is a policy question for your\n"
            "administrator, not a permissions trick."
        )

    return (
        f"{registry_line}\n"
        "\n"
        "On a machine where the other accounts are colleagues, you can also let\n"
        "them read your configuration directly by making your home traversable.\n"
        "That grants traversal, not listing: others reach a path they already\n"
        "know and still cannot enumerate your files. It is your home and your\n"
        "decision -- Paralome will not do it for you."
    )
