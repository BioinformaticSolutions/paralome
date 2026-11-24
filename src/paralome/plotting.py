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

"""Thread-safe plotly-express construction.

WHY THIS MODULE EXISTS. ``strata.fan_out`` runs the strata in a
``ThreadPoolExecutor``, and a stratum's report may call ``px.scatter``.
``px.apply_default_cascade`` reads plotly's PROCESS-GLOBAL default template:

    if trace_data_list := getattr(args["template"].data, trace_type):
        trace_specific_colors = [
            trace_data.marker.color
            for trace_data in trace_data_list
            if hasattr(trace_data, "marker") and hasattr(trace_data.marker, "color")
        ]

The default template's ``scatter`` entry carries only ``fillpattern`` -- no
``marker``. Touching ``.marker`` on it LAZILY MATERIALISES that child on the
shared template object. Two threads doing so at once leave the parent's
children tuple inconsistent with the child's back-reference, and ``_index_is``
then cannot find the child it was just handed: ``ValueError: Invalid value``
in ``apply_default_cascade``, in one stratum of many, a different one each run.

THE FIX. One lock, one funnel. Figure construction is milliseconds against
minutes of statistics per stratum, so serialising it costs nothing measurable
and removes the whole class rather than one call. Callers use these wrappers
instead of importing plotly.express directly.
"""

from __future__ import annotations

import threading
from typing import Any

import plotly.express as _px

#: Guards every plotly-express entry point. Re-entrant because a wrapper may
#: one day call another, and a deadlock in the plotting layer would be a
#: strictly worse failure than the race it replaced.
_PLOTLY_LOCK = threading.RLock()


def scatter(*args: Any, **kwargs: Any):
    """``plotly.express.scatter``, serialised across threads."""
    with _PLOTLY_LOCK:
        return _px.scatter(*args, **kwargs)


def scatter_3d(*args: Any, **kwargs: Any):
    """``plotly.express.scatter_3d``, serialised across threads."""
    with _PLOTLY_LOCK:
        return _px.scatter_3d(*args, **kwargs)


def imshow(*args: Any, **kwargs: Any):
    """``plotly.express.imshow``, serialised across threads."""
    with _PLOTLY_LOCK:
        return _px.imshow(*args, **kwargs)


def line(*args: Any, **kwargs: Any):
    """``plotly.express.line``, serialised across threads."""
    with _PLOTLY_LOCK:
        return _px.line(*args, **kwargs)


def bar(*args: Any, **kwargs: Any):
    """``plotly.express.bar``, serialised across threads."""
    with _PLOTLY_LOCK:
        return _px.bar(*args, **kwargs)


def histogram(*args: Any, **kwargs: Any):
    """``plotly.express.histogram``, serialised across threads."""
    with _PLOTLY_LOCK:
        return _px.histogram(*args, **kwargs)


#: The same lock, for code that builds figures with
#: plotly.graph_objects (which reads the same process-global template).
LOCK = _PLOTLY_LOCK


#: Plotly's qualitative palettes, re-exported so callers reading colour
#: sequences do not need to import plotly.express for that alone -- which
#: would defeat the "no direct px import" rule the test enforces.
colors = _px.colors
