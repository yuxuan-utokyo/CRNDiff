# -*- coding: utf-8 -*-
"""write lock for the shared summary file.

`215_purity_v8_rescore.json` and `215_paper_numbers_v8.json` is rewritten by the wrap-up steps of several tickets
(the overnight orchestrator of 217 and the CFGen ingestion of 219) may run concurrently on the same machine. Rewriting these two files
tools all take this lock first, so archive-old then write-new is never interleaved.

the lock is an empty file created with O_EXCL that exists only while held; a lock left by a crashed holder is considered stale after 30 minutes
broken (both tools finish within minutes).
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
