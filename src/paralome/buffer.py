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

"""A wide buffer between two processes, because a kernel pipe is 64 KB.

The fused alignment task has fastp writing both mates into FIFOs
and STAR reading both. STAR consumes mates in lockstep -- it will not take
more of R1 until it has the matching R2 -- and fastp's two output writers are
independent threads that run ahead of each other by however much their internal
pack rings allow. When that skew exceeds the pipe's 64 KB, the two deadlock:

    cat R1   pipe_write   blocked writing INTO STAR, so it stops draining R1
    cat R2   pipe_read    blocked waiting FOR fastp
    fastp    futex_wait   blocked filling R1, so it never writes that R2

A kernel pipe cannot be the fix. `F_SETPIPE_SZ` is capped by
`/proc/sys/fs/pipe-max-size` (1 MB by default), the pages are unswappable
kernel memory, and bash cannot set it at all. So the buffer has to be a process
in the stream holding ordinary user memory -- which is what this is, and why it
ships with Paralome rather than being `mbuffer`: a dependency the pipeline
cannot run without should not be something a cluster may not have.

    STAR --readFilesCommand "paralome-buffer --bytes 8G"

fastp then never blocks until 8 GB of skew has accumulated on one mate, which
is well clear of fastp's own run-ahead. Two threads: one reads the file as fast
as it is written, one writes stdout as fast as it is consumed, with a bounded
queue between them. os.read and os.write release the GIL, so this sustains well
over a gigabyte a second -- several times what fastp produces.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading

#: Read and write in blocks this size. Large enough that the per-call overhead
#: is irrelevant, small enough that the queue's granularity stays useful.
BLOCK_BYTES = 4 * 1024 * 1024

#: What a fused task gives each of its two streams. Chosen to clear fastp's
#: own run-ahead (its per-output pack ring, on the order of a gigabyte) with
#: room to spare, not to be the largest number the node could hold.
DEFAULT_BUFFER_BYTES = 8 * 1024 ** 3

_SUFFIXES = {"": 1, "B": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def parse_bytes(text: str) -> int:
    """'8G' -> 8589934592. Plain digits are bytes."""
    value = str(text).strip().upper().rstrip("IB") if str(text).strip().upper().endswith("IB") else str(text).strip().upper()
    if not value:
        raise ValueError("no size given")
    unit = value[-1] if value[-1] in _SUFFIXES and not value[-1].isdigit() else ""
    number = value[: len(value) - len(unit)] if unit else value
    try:
        scale = _SUFFIXES[unit]
        size = int(float(number) * scale)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"not a size: {text!r} (expected e.g. 8G, 512M, 1048576)") from exc
    if size < BLOCK_BYTES:
        raise ValueError(f"a buffer below {BLOCK_BYTES // 1024 ** 2} MB is not worth the processes: {text!r}")
    return size


def pump(path: str, buffer_bytes: int, out_fd: int = 1, block: int = BLOCK_BYTES) -> int:
    """Read ``path`` into a bounded queue; write it to ``out_fd``. Returns bytes written.

    The queue holds whole blocks, so its depth is the buffer size divided by the
    block size -- at 8 GB and 4 MB blocks, 2048 blocks. Memory is only what is
    actually in flight: a reader that cannot run ahead costs nothing.
    """
    depth = max(1, buffer_bytes // block)
    blocks: queue.Queue = queue.Queue(maxsize=depth)
    trouble: list[BaseException] = []

    def read_in() -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                while True:
                    chunk = os.read(fd, block)
                    if not chunk:
                        break
                    blocks.put(chunk)
            finally:
                os.close(fd)
        except BaseException as exc:  # noqa: BLE001 -- reported through the queue
            trouble.append(exc)
        finally:
            # None is end of stream, whether the read finished or failed. Without
            # it the writer would wait for ever on a reader that has gone -- the
            # very failure this program exists to prevent.
            blocks.put(None)

    reader = threading.Thread(target=read_in, name="paralome-buffer-read", daemon=True)
    reader.start()
    written = 0
    while True:
        chunk = blocks.get()
        if chunk is None:
            break
        view = memoryview(chunk)
        while view:
            # A short write is normal on a pipe whose reader is slow.
            sent = os.write(out_fd, view)
            view = view[sent:]
            written += sent
    reader.join()
    if trouble:
        raise trouble[0]
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paralome-buffer",
        description="Copy a file to standard output through a large in-memory buffer. "
                    "Used as STAR's --readFilesCommand in a fused alignment task, where a "
                    "64 KB kernel pipe is too narrow for the skew between fastp's two outputs.",
    )
    parser.add_argument("path", help="file (or FIFO) to read")
    parser.add_argument("--bytes", default=str(DEFAULT_BUFFER_BYTES),
                        help="buffer size, e.g. 8G (default: 8G)")
    args = parser.parse_args(argv)
    try:
        size = parse_bytes(args.bytes)
    except ValueError as exc:
        print(f"paralome-buffer: {exc}", file=sys.stderr)
        return 2
    try:
        pump(args.path, size)
    except BrokenPipeError:
        # The consumer went away. Not this program's error to report, and not a
        # reason to print a traceback into a job log.
        return 141
    except OSError as exc:
        print(f"paralome-buffer: {args.path}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover -- exercised through the console script
    raise SystemExit(main())
