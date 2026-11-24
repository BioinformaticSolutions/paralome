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

"""One zip archive, its files compressed on every CPU at once.

Python's zipfile compresses one member after another in the calling process, which for an archive of tens
of GB takes over an hour on one CPU. Here each file is deflated in a worker process (zlib, the level zipfile
uses) and the main process appends the compressed members in order and writes the central directory. The
archive is a standard zip in zip64 form (every entry carries its zip64 sizes and offset, and the archive its
zip64 end records), so any size and any number of entries; Python's zipfile, unzip and every OS's archive
tool read it.
"""

from __future__ import annotations

import os
import struct
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

LEVEL = 6                       # zlib's default, what zipfile.ZIP_DEFLATED writes
_VERSION = 45                   # zip64
_FLAGS = 0x0800                 # names in UTF-8
_DEFLATED, _STORED = 8, 0
_U32 = 0xFFFFFFFF


def _dos_time(mtime: float) -> Tuple[int, int]:
    t = time.localtime(mtime)
    year = max(1980, min(2107, t.tm_year))
    return ((t.tm_hour << 11) | (t.tm_min << 5) | (t.tm_sec // 2),
            ((year - 1980) << 9) | (t.tm_mon << 5) | t.tm_mday)


def _deflate(data: bytes, level: int) -> Tuple[bytes, int]:
    packer = zlib.compressobj(level, zlib.DEFLATED, -15)
    return packer.compress(data) + packer.flush(), zlib.crc32(data) & _U32


def _compress_file(task: Tuple[str, int]) -> Tuple[bytes, int, int, float]:
    """(deflated bytes, crc32, size, mtime) of one file -- in a worker."""
    path, level = task
    with open(path, "rb") as handle:
        data = handle.read()
    packed, crc = _deflate(data, level)
    return packed, crc, len(data), os.stat(path).st_mtime


class _Writer:
    def __init__(self, handle):
        self.handle = handle
        self.offset = 0
        self.central: List[bytes] = []

    def add(self, name: str, packed: bytes, crc: int, size: int, mtime: float, method: int = _DEFLATED,
            mode: int = 0o100644) -> None:
        encoded = name.encode("utf-8")
        dostime, dosdate = _dos_time(mtime)
        local_extra = struct.pack("<HHQQ", 0x0001, 16, size, len(packed))
        header = struct.pack("<IHHHHHIIIHH", 0x04034B50, _VERSION, _FLAGS, method, dostime, dosdate, crc,
                             _U32, _U32, len(encoded), len(local_extra))
        offset = self.offset
        self.handle.write(header + encoded + local_extra)
        self.handle.write(packed)
        self.offset += len(header) + len(encoded) + len(local_extra) + len(packed)
        central_extra = struct.pack("<HHQQQ", 0x0001, 24, size, len(packed), offset)
        self.central.append(struct.pack(
            "<IHHHHHHIIIHHHHHII", 0x02014B50, (3 << 8) | _VERSION, _VERSION, _FLAGS, method, dostime, dosdate,
            crc, _U32, _U32, len(encoded), len(central_extra), 0, 0, 0, (mode & 0xFFFF) << 16, _U32,
        ) + encoded + central_extra)

    def close(self) -> None:
        start = self.offset
        for entry in self.central:
            self.handle.write(entry)
        size = sum(len(entry) for entry in self.central)
        count = len(self.central)
        zip64_end = start + size
        self.handle.write(struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, (3 << 8) | _VERSION, _VERSION, 0, 0,
                                      count, count, size, start))
        self.handle.write(struct.pack("<IIQI", 0x07064B50, 0, zip64_end, 1))
        self.handle.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, min(count, 0xFFFF), min(count, 0xFFFF),
                                      min(size, _U32), _U32, 0))


def write_zip(zip_path: Union[str, Path], members: Sequence[Tuple[Union[str, Path], str]], *,
              texts: Iterable[Tuple[str, str]] = (), jobs: int = 1, level: int = LEVEL) -> Path:
    """Write ``zip_path`` (through ``<name>.partial``, renamed when complete): ``texts`` first (name, text), then
    every (file, name in the archive) of ``members`` in order, each deflated -- on ``jobs`` processes."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".partial")
    jobs = max(1, int(jobs or 1))
    tasks = [(str(path), level) for path, _name in members]
    names = [name for _path, name in members]
    with open(partial, "wb") as handle:
        writer = _Writer(handle)
        for name, text in texts:
            data = text.encode("utf-8")
            packed, crc = _deflate(data, level)
            writer.add(name, packed, crc, len(data), time.time())
        if jobs == 1 or len(tasks) <= 1:
            for (name, task) in zip(names, tasks):
                writer.add(name, *_compress_file(task))
        else:
            import multiprocessing

            window = jobs * 4                  # compressed members held at once, in order
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("fork")) as pool:
                pending = []
                position = 0
                while position < len(tasks) or pending:
                    while position < len(tasks) and len(pending) < window:
                        pending.append((names[position], pool.submit(_compress_file, tasks[position])))
                        position += 1
                    name, future = pending.pop(0)
                    writer.add(name, *future.result())
        writer.close()
    os.replace(partial, zip_path)
    return zip_path
