# -*- coding: utf-8 -*-
"""A write lock for the shared summary files.

Two summary files are rewritten by more than one finishing routine, and those routines can run
at the same time on one machine. Every tool that rewrites them takes this lock first, so that
'archive the old version, then write the new one' is never interleaved.

The lock is an empty file created with O_EXCL and exists only while it is held. A lock left
behind by a crash is treated as stale after thirty minutes; both tools finish in minutes.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "out" / "summary" / ".summary_write.lock"


@contextmanager
def summary_write_lock(owner: str, timeout_s: float = 4 * 3600, stale_s: float = 1800):
    t0, warned = time.time(), False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{owner} pid={os.getpid()} at={time.strftime('%H:%M:%S')}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - LOCK.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_s:
                print(f"[lock] stale lock ({age / 60:.0f} min) -> taking over", flush=True)
                try:
                    LOCK.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"[lock] timeout waiting for {LOCK.name}")
            if not warned:
                try:
                    holder = LOCK.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    holder = "?"
                print(f"[lock] waiting for {LOCK.name} held by: {holder}", flush=True)
                warned = True
            time.sleep(5)
    try:
        yield
    finally:
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass
