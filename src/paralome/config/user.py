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

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

from paralome.config.schema import (
    ConfigValidationError, MalformedConfigError, default_config, validate_config,
)
from paralome.errors import ParalomeError

ENV_CONFIG_DIR = "PARALOME_CONFIG_DIR"
DEFAULT_CONFIG_DIRNAME = ".paralome"
CONFIG_FILENAME = "config.json"

#: The shipped defaults Paralome itself wrote into the file, by
#: "section.key", so that a value nobody changed follows the default when the
#: default changes. Every default is written into the file so it can be seen
#: and edited there (_backfill_missing_defaults), and without this record a
#: copied default could not be told from a value the user chose.
DEFAULTS_RECORD_KEY = "paralome_defaults"

#: Defaults a key shipped with before its current one. A configuration written
#: before the record existed, holding one of these, holds a copied default and
#: not a choice -- for rna.strandedness, 0 was the only value there was until
#: detection existed.
EARLIER_DEFAULTS: dict[str, tuple[Any, ...]] = {
    "rna.strandedness": (0,),
}


class InvalidConfigValueError(ParalomeError, ConfigValidationError):
    """A configuration file whose JSON and keys are fine but a value is not.

    Validate_config raised a bare ConfigValidationError for a wrong
    config_version, a missing section, an rna.strandedness other than
    0/1/2/auto or a negative threshold; it named neither a number nor the
    file, and resolve_rna_expression_gate took it for a missing file and fell
    back to the built-in 5/5 gate. A ParalomeError as well, so every command
    reports it under PLM-0413 with the file's name; still a
    ConfigValidationError for the code that catches those.
    """

    def __init__(self, detail: str) -> None:
        ParalomeError.__init__(
            self, 413, detail,
            fix="Correct the value in the file the message names",
        )


def config_dir() -> Path:
    """Return the active Paralome user configuration directory.

    Precedence:
      1. PARALOME_CONFIG_DIR environment variable
      2. ~/.paralome
    """
    override = os.environ.get(ENV_CONFIG_DIR)
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return (Path.home() / DEFAULT_CONFIG_DIRNAME).resolve()


def config_path() -> Path:
    return config_dir() / CONFIG_FILENAME


def _default_items(defaults: dict[str, Any]) -> dict[str, Any]:
    """The defaults one level deep, as {"section.key": value}: what the file is given."""
    items: dict[str, Any] = {}
    for section, block in defaults.items():
        if isinstance(block, dict):
            for key, value in block.items():
                items[f"{section}.{key}"] = value
    return items


def fresh_config() -> dict[str, Any]:
    """The configuration a new file is written with: the defaults, recorded as such."""
    config = default_config()
    config[DEFAULTS_RECORD_KEY] = copy.deepcopy(_default_items(config))
    return config


def record_as_shipped(config: dict[str, Any], sections: tuple[str, ...]) -> None:
    """Mark every key of ``sections`` as holding the shipped default (setup --force)."""
    record = config.setdefault(DEFAULTS_RECORD_KEY, {})
    for dotted, value in _default_items(default_config()).items():
        if dotted.split(".", 1)[0] in sections:
            record[dotted] = copy.deepcopy(value)


def holds_shipped_default(config: dict[str, Any], dotted: str) -> bool:
    """True when the file's value for "section.key" is the default Paralome wrote there."""
    section, key = dotted.split(".", 1)
    record = config.get(DEFAULTS_RECORD_KEY)
    block = config.get(section)
    return (isinstance(record, dict) and dotted in record and isinstance(block, dict)
            and key in block and block[key] == record[dotted])


def initialize_config(*, force: bool = False) -> Path:
    """Create the user config file if needed and return its path.

    Existing config is preserved unless force=True.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    save_config(fresh_config(), path=path)
    return path


def _follow_changed_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    """Bring every value Paralome wrote and nobody changed up to the current default.

    Kept in ``DEFAULTS_RECORD_KEY``: a key whose value still equals
    the default recorded for it takes the current default when that differs;
    a key whose value differs from its record was changed by the user, and
    leaves the record, so it is never touched again. A file written before the
    record existed is given one: a value equal to the current default is
    recorded as it, and a value equal to an EARLIER_DEFAULTS entry is a copied
    old default and takes the current one. Returns one notice per value
    changed.
    """
    legacy = not isinstance(config.get(DEFAULTS_RECORD_KEY), dict)
    record = {} if legacy else config[DEFAULTS_RECORD_KEY]
    notices: list[str] = []
    for dotted, value in _default_items(defaults).items():
        section, key = dotted.split(".", 1)
        block = config.get(section)
        if not isinstance(block, dict) or key not in block:
            continue                       # added, and recorded, by the backfill
        current = block[key]
        if dotted in record:
            if current != record[dotted]:
                del record[dotted]          # the user's value from here on
            elif current != value:
                block[key] = copy.deepcopy(value)
                record[dotted] = copy.deepcopy(value)
                notices.append(
                    f"{dotted} was {current!r}, the default an earlier Paralome wrote into this "
                    f"file; the default is now {value!r}, so it now reads {value!r}"
                )
        elif legacy:
            if current == value:
                record[dotted] = copy.deepcopy(value)
            elif any(current == old and type(current) is type(old)
                     for old in EARLIER_DEFAULTS.get(dotted, ())):
                block[key] = copy.deepcopy(value)
                record[dotted] = copy.deepcopy(value)
                notices.append(
                    f"{dotted} was {current!r}, the default an earlier Paralome wrote into this "
                    f"file; the default is now {value!r}, so it now reads {value!r}"
                )
    config[DEFAULTS_RECORD_KEY] = record
    return notices


def _backfill_missing_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> bool:
    """Fill in keys ``defaults`` has that ``config`` is missing, one level of
    nesting deep for dict-valued sections (e.g. ``rna``). Never overwrites a
    key already present, even if its value differs from the default.
    Returns True if anything was added.

    So every key Paralome reads is visible and editable in the one place a
    user would look for it, including keys added after the file was created.
    Applied generically, for every section, whenever a config is loaded.
    """
    changed = False
    record = config.get(DEFAULTS_RECORD_KEY)
    for key, value in defaults.items():
        if key not in config:
            config[key] = copy.deepcopy(value)
            changed = True
            if isinstance(value, dict) and isinstance(record, dict):
                for sub_key, sub_value in value.items():
                    record[f"{key}.{sub_key}"] = copy.deepcopy(sub_value)
        elif isinstance(value, dict) and isinstance(config.get(key), dict):
            for sub_key, sub_value in value.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = copy.deepcopy(sub_value)
                    if isinstance(record, dict):
                        # A value Paralome wrote, so it follows the default.
                        record[f"{key}.{sub_key}"] = copy.deepcopy(sub_value)
                    changed = True
    return changed


def _validate_naming_the_file(data: dict[str, Any], path: Path) -> None:
    """validate_config, with a bare validation failure numbered and naming the file."""
    try:
        validate_config(data, source=str(path))
    except ParalomeError:
        raise
    except ConfigValidationError as exc:
        # See InvalidConfigValueError.
        raise InvalidConfigValueError(
            f"{path}: {exc}. That file is this account's Paralome configuration."
        ) from exc


def load_config(*, initialize: bool = False) -> dict[str, Any]:
    """Load and minimally validate user config.

    If initialize=True, create a default config if missing.
    """
    path = config_path()
    if not path.exists():
        if initialize:
            initialize_config()
        else:
            raise FileNotFoundError(
                f"Paralome config not found: {path}. Run 'paralome setup'."
            )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        # Under a number, naming the file. See MalformedConfigError.
        raise MalformedConfigError(
            f"{path} is not valid JSON: {exc}. That file is this account's "
            "Paralome configuration."
        ) from exc
    if not isinstance(data, dict):
        raise MalformedConfigError(
            f"{path} holds JSON that is not an object ({type(data).__name__}). "
            "That file is this account's Paralome configuration."
        )
    _validate_naming_the_file(data, path)
    record_before = copy.deepcopy(data.get(DEFAULTS_RECORD_KEY))
    notices = _follow_changed_defaults(data, default_config())
    backfilled = _backfill_missing_defaults(data, default_config())
    for notice in notices:
        print(f"[paralome] {path}: {notice}. Set it in that file to keep a value of your own.",
              file=sys.stderr, flush=True)
    if backfilled or notices or data.get(DEFAULTS_RECORD_KEY) != record_before:
        _validate_naming_the_file(data, path)
        # READING THE CONFIG MUST NOT BE ABLE TO FAIL ON A WRITE.
        #
        # The backfill is a courtesy -- it makes new keys visible in the file --
        # and the values are already in `data`. But this runs in every process
        # that reads the config, including array tasks starting together, and
        # the account may not be allowed to write the file at all; a failed
        # write must not make a caller see "no config" (tools from PATH instead
        # of the recorded paths, stage settings dropped).
        try:
            save_config(data, path=path)
        except OSError:
            pass
    return data


def save_config(config: dict[str, Any], *, path: Optional[Path] = None) -> Path:
    """Validate and write a config JSON file."""
    out = path or config_path()
    validate_config(config, source=str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    # A temporary file of its own, so two processes saving at the same moment cannot share one and
    # trip over each other's rename.
    fd, temporary_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=str(out.parent))
    tmp = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        # mkstemp creates the file private to the owner; keep the permissions
        # the config already had, or the usual ones for a new file.
        try:
            mode = out.stat().st_mode & 0o777
        except OSError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out


def resolve_rna_expression_gate(
    min_expression: Optional[float], min_expression_difference: Optional[float]
) -> Tuple[float, float]:
    """Resolve the RNA low-expression/small-effect gate's two thresholds.

    A CLI caller that didn't pass an explicit value (``None``)
    falls back to the user's configured ``rna.min_expression``/
    ``rna.min_expression_difference`` (5.0/5.0 out of the box -- see
    ``paralome.config.schema.DEFAULT_CONFIG``), so the gate is on by
    default everywhere rather than requiring every project to discover and
    set it manually. A caller that *does* pass a value (including an
    explicit ``0.0`` to disable the gate) always wins over config. Missing
    or not-yet-initialized config (a fresh install that never ran
    ``paralome setup``) falls back to the same built-in defaults
    rather than raising, since this is a convenience default, not a
    required setting.
    """
    if min_expression is not None and min_expression_difference is not None:
        return float(min_expression), float(min_expression_difference)
    try:
        config = load_config(initialize=False)
    except ParalomeError:
        # A mistyped key, or a file that is not JSON, is refused
        # here as everywhere. Falling back to the built-in gate would be the
        # silent ignoring the refusal ends.
        # And so is a value that does not validate (PLM-0413). This
        # caught ConfigValidationError beside FileNotFoundError, so a config
        # with rna.min_expression 1.0 and a misspelt rna.strandedness was
        # replaced, whole, by default_config(): the gate ran at 5 FPKM and
        # nothing said so. Only a file that is not there means "no setting".
        raise
    except FileNotFoundError:
        config = default_config()
    rna_config = config.get("rna") or {}
    resolved_min_expression = (
        float(min_expression)
        if min_expression is not None
        else float(rna_config.get("min_expression", 5.0))
    )
    resolved_min_expression_difference = (
        float(min_expression_difference)
        if min_expression_difference is not None
        else float(rna_config.get("min_expression_difference", 5.0))
    )
    return resolved_min_expression, resolved_min_expression_difference


def describe_config_location() -> dict[str, Any]:
    path = config_path()
    return {
        "env_var": ENV_CONFIG_DIR,
        "config_dir": str(config_dir()),
        "config_path": str(path),
        "exists": path.exists(),
    }
