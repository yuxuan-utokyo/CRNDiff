# -*- coding: utf-8 -*-
"""Check that a fresh clone of this repository actually works, without any data download.

It runs every entry point in `scripts/reproduce/` -- the ten paper tables and the five figures
that need no atlas counts -- and reports what each one produced. It also re-derives the toy
figure extract and compares its arrays against the copy shipped in `results/toy/`, which is the
one check here that compares against a published artefact rather than only asserting that a
script ran.

Nothing under `results/` is written. Everything lands in `out/`.

    python scripts/run_smoke_test.py            # tables and figures (about 2 minutes)
    python scripts/run_smoke_test.py --tables   # tables only (about 20 seconds)
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce._array_compare import compare_npz
REPRO = ROOT / "scripts" / "reproduce"

TABLES = sorted(REPRO.glob("table_*.py"))
FIGURES = [REPRO / n for n in ("build_toy_figure_data.py",
                               "figure_01_toy_delivered.py",
                               "figure_toy_dataset.py",
                               "figure_toy_chain_and_resampling.py",
                               "figure_horizon_criterion.py")]
SHIPPED_FIGDATA = ROOT / "results" / "toy" / "figdata.npz"
REBUILT_FIGDATA = ROOT / "out" / "toy" / "figdata.npz"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def environment() -> None:
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    for mod in ("numpy", "scipy", "matplotlib", "pandas"):
        try:
            m = __import__(mod)
            print(f"{mod:12s}{getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{mod:12s}MISSING -- pip install -r requirements.txt")
            raise SystemExit(2)
    if sys.version_info < (3, 10):
        raise SystemExit("this package needs Python 3.10 or newer")


def run(script: Path) -> tuple[bool, float, str]:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()
    return r.returncode == 0, time.time() - t0, (tail[-1] if tail else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="skip the figures")
    a = ap.parse_args()

    print("=" * 78)
    environment()
    print("=" * 78)

    scripts = list(TABLES) + ([] if a.tables else FIGURES)
    failed = []
    for s in scripts:
        ok, dt, last = run(s)
        print(f"{'PASS' if ok else 'FAIL'}  {s.name:44s}{dt:6.1f}s")
        if not ok:
            failed.append(s.name)
            print(f"      {last}")

    print("-" * 78)
    out_tables = sorted((ROOT / "out" / "tables").glob("*"))
    print(f"tables written to out/tables/   ({len(out_tables)} files)")
    if not a.tables:
        figs = sorted((ROOT / "out" / "figures").glob("*.pdf"))
        print(f"figures written to out/figures/ ({len(figs)} pdf)")
        ok, detail = compare_npz(SHIPPED_FIGDATA, REBUILT_FIGDATA)
        print(f"toy figure extract rebuilt: {detail}")
        if not ok:
            failed.append("figdata.npz array comparison")

    print("-" * 78)
    if failed:
        print(f"[smoke] {len(failed)} of {len(scripts)} checks failed: {', '.join(failed)}")
        return 1
    print(f"[smoke] all {len(scripts)} entry points reproduced from the files in this repository")
    return 0


if __name__ == "__main__":
    sys.exit(main())
