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

"""Showing the user configuration. One command.

`setup` is the parent of all utilities involved in setting up and maintaining
the software; this shows what the configuration holds and where it lives.
Creating a config is `paralome setup` (which also detects the external tools
and the scheduler), and `paralome setup reset` is the way to remove a config:
it says what it is about to destroy before doing it.
"""

import json

import typer

from paralome.config.schema import ConfigValidationError
from paralome.config.user import describe_config_location, load_config


def config_command(
) -> None:
    """Print this account's Paralome configuration, and where it lives."""
    from paralome.errors import ParalomeError

    try:
        data = load_config(initialize=False)
    except ParalomeError:
        # A key nothing reads is refused by name, not reported as a missing file.
        raise
    except (FileNotFoundError, ConfigValidationError) as exc:
        raise typer.BadParameter(
            f"{exc}\nRun `paralome setup` to create one."
        ) from exc

    loc = describe_config_location()
    print("Paralome configuration")
    print("=======================")
    print(f"Config path          : {loc['config_path']}")
    print(f"Config dir           : {loc['config_dir']}")
    print(f"Environment override : {loc['env_var']}")
    print("")
    print(json.dumps(data, indent=2, sort_keys=True))
