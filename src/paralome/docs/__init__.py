"""Documentation that ships inside the wheel.

WHY THESE LIVE UNDER ``src/``.

The wheel is the unit that gets handed to someone -- a collaborator, a reviewer,
an AI assistant. ``pyproject.toml`` sets ``package-dir = {"" = "src"}`` and
finds packages only there, so the user-facing documents live here and the
answer to "how do I do this" and "why is this broken" travels with the
software.

Here: the user manual, and the briefing written to be pasted into an AI
assistant. Both answer questions about *using* Paralome.

REACHING THESE FILES.

    from paralome.docs import brief, manual, path

``path()`` returns a real directory on disk, which is what an assistant with
file access actually wants. The module-level readers exist so the text can be
had without knowing the filenames.
"""

from __future__ import annotations

import pathlib

__all__ = [
    "path", "names", "read", "manual", "brief", "installation",
    "MANUAL", "BRIEF", "INSTALLATION",
]

#: Filenames, not paths. The briefing keeps its long, self-describing name on
#: purpose: someone who unzips the wheel and reads a file listing should be able
#: to tell what it is without opening it.
MANUAL = "MANUAL.md"
BRIEF = "What_can_an_AI_tell_me_about_Paralome.md"
#: 0.3: the third audience. The manual answers "how do I run this"; the
#: briefing answers "what is this"; this one answers "how do I set this up for
#: other people", which is a different reader with different assumptions and
#: root on the machine.
INSTALLATION = "INSTALLATION.md"


def path() -> pathlib.Path:
    """The directory holding the shipped documents.

    Uses ``__file__`` rather than ``importlib.resources`` because every
    consumer of this -- a person, an assistant with a filesystem, ``paralome
    docs --path`` -- wants a real path it can open, and Paralome is never
    installed from a zipimport or an egg where that would fail.
    """
    return pathlib.Path(__file__).resolve().parent


def names() -> list[str]:
    """Every shipped document, sorted, filenames only."""
    return sorted(p.name for p in path().glob("*.md"))


def read(name: str) -> str:
    """One document by filename. Raises FileNotFoundError with the options."""
    target = path() / name
    if not target.is_file():
        raise FileNotFoundError(
            f"no shipped document named {name!r}; have: {', '.join(names())}"
        )
    return target.read_text(encoding="utf-8")


def manual() -> str:
    """The user manual."""
    return read(MANUAL)


def brief() -> str:
    """The briefing intended to be given to an AI assistant."""
    return read(BRIEF)


def installation() -> str:
    """The guide for whoever administers the machine."""
    return read(INSTALLATION)


def user_copy_path() -> pathlib.Path:
    """Where this account keeps its readable copy of the documents.

    Beside the config, in ``~/.paralome/docs/``. 

    He is right on both counts. The shipped copy lives under the install
    prefix -- somewhere inside site-packages, a path no user should have to
    learn and none of them would think to look in. It is readable, because
    everything in site-packages is, but "readable" and "findable" are not the
    same property, and only the second one helps anybody.
    """
    from paralome.config.user import config_dir

    return config_dir() / "docs"


def refresh_user_copy() -> tuple[pathlib.Path, list[str]]:
    """Copy every shipped document into the user's own directory, overwriting.

    REFRESHED ON EVERY CALL, not written once at setup: a copy made once at
    install time goes stale the moment the wheel is upgraded, and a user would
    read the manual for a version they are not running. The documents are
    small, so rewriting them whenever somebody asks costs nothing measurable.

    Returns the directory and the filenames placed in it.
    """
    import shutil

    destination = user_copy_path()
    destination.mkdir(parents=True, exist_ok=True)
    placed: list[str] = []
    for filename in names():
        shutil.copyfile(path() / filename, destination / filename)
        placed.append(filename)
    return destination, placed
