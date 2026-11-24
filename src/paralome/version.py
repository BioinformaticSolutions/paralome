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

"""The single source of truth for the Paralome version.

Releases are numbered X.Y, and X.Y.Z for fixes; a trailing .0 is dropped (0.9, 0.9.1, 1.0, 1.0.1, 1.1).
Development builds are PEP 440 pre-releases of the next release (X.Y.devN), so a development build always sorts
above the installed release and below the one it is heading for. Every build carries its own number; the wheel
builder refuses to build over an existing wheel of the same version.

``RELEASE_LABEL`` is the release's standing, printed after the number by ``paralome --version`` and on the
documents ("0.9 (beta)"); empty for a release that carries no label.

Every version installs into a prefix of its own (pip install --prefix=<root>/<version>) and is reached through a
module for that version, so versions stay resident side by side and which one is the default is which module is
the default. Where they are installed is not recorded here: the package finds itself from sys.executable, so it can
be relocated and installed more than once. ~/.paralome/config.json is shared by every version.
"""

from __future__ import annotations

__version__ = "0.9"
RELEASE_LABEL = "beta"
