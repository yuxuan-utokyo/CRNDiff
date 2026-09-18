# -*- coding: utf-8 -*-
"""One table: every input this package needs, present or absent, with its expected path.

It checks the frozen files' sha256, the product-tilt tables, the discriminators, the generator
checkpoint, the data tree, the rebuilt gene panel, and the assets known to be missing, then
prints a present/absent table with expected paths and hashes. This script fixes nothing.

Exit code: non-zero on any MISMATCH, because a file whose sha256 disagrees with
PROVENANCE.json invalidates whatever is computed from it. A MISSING input is the normal state
of a fresh clone -- the large arrays, checkpoints and count matrices are not redistributed --
so it is reported but does not fail the run; pass --strict to require everything.

Three verdicts:
    OK        present, and its sha256 matches where an authoritative value exists
    MISMATCH  present but the sha256 disagrees with PROVENANCE.json: stop and investigate
    MISSING   absent; assets/README.md says how to rebuild it

    python scripts/check_assets.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                          # noqa: E402
from crndiff.io import sha256_file                       # noqa: E402

PROV = ROOT / "PROVENANCE.json"

# Inputs that are known NOT to have come across in the migration. They are listed here so the
# table says WHERE they should be and WHAT depends on them, instead of a downstream script
# failing with a bare FileNotFoundError.
KNOWN_MISSING = [
    {"what": "residual_classifier.joblib (source of the portable npz) -- for the "
             "MIGRATED bare-type dirs only",
     "expect": C.RESIDUAL / "<Endothelial|Myeloid|Neuronal|UNCOND>" /
               "residual_classifier.joblib",
     "needed_by": "the frozen source-hash check in fk_hvg_v2.main",
     "note": "old tree: HVG2K/universality_fk/out/disc_*/residual_classifier.joblib. The "
             "portable npz records its own source_sha256, which is reported instead; the "
             "comparison is marked UNAVAILABLE in those run jsons rather than skipped. "
             "UPDATED 2026-09-07: this no longer covers every dir. INTERSECT_Endothelial_"
             "g128 (migrated with its joblib on 2026-09-07) and the three UNTILTED_* dirs "
             "(fitted in this tree) DO carry the joblib, so their runs record "
             "source_hash_check=performed. check_assets enumerates the dirs, so the table "
             "below shows the per-dir truth."},
    {"what": "20,000-cell proposal pool (Baseline/OURS/results)",
     "expect": C.LEGACY_ROOT / "Baseline" / "OURS" / "results" / "<pool>.npy",
     "needed_by": "nothing in this package today",
     "note": "6.3 GB, deliberately not inherited; still on the desktop backup (PROVENANCE.json "
             "not_inherited). NOTE this is a DIFFERENT artefact from the tau=0 base pool topped "
             "up on 2026-09-06, which IS present (see the release-assets block)."},
]

# Topped up 2026-09-06 by ruling 164 section 3 (all bitwise-verified; PROVENANCE.json
# asset_topup_20260906). They were MISSING in receipt 105 and are OK now.
RELEASE_ASSETS = [
    {"what": "canonical purity label sets (master_purity_v7.json)", "path": C.MASTER_PURITY_V7,
     "sha256": "8ea88aee14297c046d62ff9681380c9389334bd05b5775b342e3241453808796",
     "needed_by": "hvg/metrics.py canonical_purity -- every published purity is on this scale"},
    {"what": "bare cell-type tilt tables (tables_kz)", "path": C.TABLES_KZ, "sha256": None,
     "needed_by": "Stage 1 / 2 / 5 arms, whose requests are bare cell types. Stages 3, 4 and 6 request INTERSECT_* / MIX_*, whose tables live in tables_ref"},
    {"what": "CellTypist eval script + archived label histograms", "path": C.CELLTYPIST,
     "sha256": None, "needed_by": "gate G5 (the purity ruler must be the archived one)"},
    {"what": "tau=0 UNTILTED base pool (20,000 cells)", "path": C.BASE_POOL,
     "sha256": C.BASE_POOL_SHA256,
     "needed_by": "the negative pool a proposal_untilted discriminator is fitted against "
                  "(the residual_only ablation), and the pilot row of gate G5's reference arm"},
]


def row(status: str, what: str, path, extra: str = "") -> dict:
    return {"status": status, "what": what, "path": str(path), "detail": extra}


def check_file(what: str, path: Path, want_sha: str | None = None,
               hint: str = "") -> dict:
    p = Path(path)
    if not p.exists():
        return row("MISSING", what, p, hint)
    if want_sha is None:
        size = p.stat().st_size if p.is_file() else None
        return row("OK", what, p, f"{size} B" if size is not None else "directory")
    got = sha256_file(p)
    if got != want_sha:
        return row("MISMATCH", what, p, f"got {got} want {want_sha}")
    return row("OK", what, p, f"sha256 {got[:16]}...")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="where to write the report (default out/gates/)")
    ap.add_argument("--strict", action="store_true",
                    help="also exit non-zero when an input is MISSING, not only on a MISMATCH")
    a = ap.parse_args(argv)
    rows: list[dict] = []
    prov = json.loads(PROV.read_text(encoding="utf-8")) if PROV.exists() else {}

    # ---- 1. the eight frozen function-level copies ----
    for f in prov.get("files", []):
        rows.append(check_file(f"frozen: {f['dst']}", ROOT / f["dst"], f["sha256_dst"]))

    # ---- 2. the single edited file of the whole migration ----
    edit = prov.get("inheritance_X_to_HVG", {}).get("only_edit")
    if edit:
        rows.append(check_file("migration's ONLY text edit: legacy config_hvg2k.py ROOT",
                               ROOT / edit["file"], edit["sha256_after"]))
        rows.append(check_file("pristine pre-edit copy kept in _frozen",
                               C.FROZEN / "config_hvg2k.py", edit["sha256_before"]))

    # ---- 3. the rebuilt panel61 (gate H0 is what actually validates it) ----
    rb = prov.get("inheritance_X_to_HVG", {}).get("reconstructed_panel61", {})
    for name, sha in (rb.get("sha256") or {}).items():
        rows.append(check_file(f"panel61 (rebuilt, H0): {name}", C.PANEL61_CODE / name, sha))
    rows.append(check_file("panel61 data dir", C.PANEL61 / "_shared" / "data"))

    # ---- 4. the frozen generator ----
    rows.append(check_file("frozen generator ckpt", C.GENERATOR_CKPT, C.GENERATOR_CKPT_SHA256))
    rows.append(check_file("generator ckpt sidecar json",
                           C.GENERATOR / "pilot_run1c_s20260625.json"))
    rows.append(check_file("LA prior logpi (unconditional proposal)", C.LOGPI_LA))
    rows.append(check_file("legacy copy of the generator ckpt",
                           C.LEGACY_ROOT / "Baseline" / "OURS" / "models"
                           / "pilot_run1c_s20260625.pt", C.GENERATOR_CKPT_SHA256))

    # ---- 5. the data tree ----
    rows.append(check_file("legacy data tree (1.7 GB counts + labels)", C.LEGACY_DATA / "all"))
    rows.append(check_file("legacy data meta.json", C.LEGACY_DATA / "all" / "meta.json"))
    rows.append(check_file("frozen sampling code dir", C.BASE_CODE))

    # ---- 6. product tilt tables ----
    if C.TABLES.exists():
        tables = sorted(C.TABLES.glob("logc_*.npy"))
        rows.append(row("OK", f"product tilt tables ({len(tables)} present)", C.TABLES,
                        ", ".join(p.stem.replace("logc_", "") for p in tables)))
        for p in tables:
            rows.append(check_file(f"  tilt table {p.stem}", p))
    else:
        rows.append(row("MISSING", "product tilt tables", C.TABLES, "assets/tables/tables_ref"))

    # ---- 7. discriminators ----
    if C.CLEAN_REWARD.exists():
        for p in sorted(C.CLEAN_REWARD.glob("*_clean_logistic.npz")):
            rows.append(check_file(f"clean reward {p.stem.split('_clean')[0]}", p))
    else:
        rows.append(row("MISSING", "clean reward dir", C.CLEAN_REWARD))
    if C.RESIDUAL.exists():
        for d in sorted(p for p in C.RESIDUAL.iterdir() if p.is_dir()):
            paths = C.residual_paths(d)
            have = [k for k in ("portable", "log", "split", "joblib") if paths[k].exists()]
            miss = [k for k in ("portable", "log", "split", "joblib") if not paths[k].exists()]
            rows.append(row("OK" if "portable" in have else "MISSING",
                            f"residual reward dir {d.name}", d,
                            f"have {have}" + (f"; missing {miss}" if miss else "")))
    else:
        rows.append(row("MISSING", "residual reward dir", C.RESIDUAL))

    # ---- 8. the four inputs ruling 164 section 3 topped up ----
    for r in RELEASE_ASSETS:
        rows.append(check_file(f"release asset: {r['what']}", r["path"], r["sha256"],
                               f"needed by: {r['needed_by']}"))
    kz = sorted(C.TABLES_KZ.glob("logc_*.npy")) if C.TABLES_KZ.exists() else []
    rows.append(row("OK" if len(kz) == 11 else "MISSING",
                    f"  tables_kz cell-type tables ({len(kz)}/11)", C.TABLES_KZ,
                    ", ".join(p.stem.replace("logc_", "") for p in kz)))
    hist = sorted(C.CELLTYPIST.glob("celltypist_*.json")) if C.CELLTYPIST.exists() else []
    rows.append(row("OK" if hist else "MISSING",
                    f"  archived CellTypist label histograms ({len(hist)})", C.CELLTYPIST,
                    "gate G5 reproduces one of these key-by-key"))

    # ---- 9. the known-missing list (kept: ruling 164 section 3 says do not delete these) ----
    for m in KNOWN_MISSING:
        p = Path(m["expect"])
        status = "OK" if ("<" not in p.name and p.exists()) else "MISSING"
        rows.append(row(status, m["what"], p, f"needed by: {m['needed_by']} | {m['note']}"))

    n_bad = sum(1 for r in rows if r["status"] != "OK")
    width = max(len(r["what"]) for r in rows) + 2
    print(f"{'STATUS':10s}{'WHAT':{width}s}PATH / DETAIL")
    print("-" * (10 + width + 40))
    for r in rows:
        print(f"{r['status']:10s}{r['what']:{width}s}{r['path']}")
        if r["detail"]:
            print(f"{'':10s}{'':{width}s}  {r['detail']}")
    print("-" * (10 + width + 40))
    print(f"[assets] {len(rows)} checks, {n_bad} not OK "
          f"({sum(1 for r in rows if r['status'] == 'MISSING')} MISSING, "
          f"{sum(1 for r in rows if r['status'] == 'MISMATCH')} MISMATCH)")
    dest = Path(a.json) if a.json else (C.GATES / "ASSETS.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"rows": rows, "n_not_ok": n_bad}, ensure_ascii=False, indent=1)
                    + "\n", encoding="utf-8")
    print(f"[assets] -> {dest}")
    n_mismatch = sum(1 for r in rows if r["status"] == "MISMATCH")
    if n_mismatch:
        return 1
    if a.strict and n_bad:
        return 1
    if n_bad:
        print("[assets] the missing entries above are inputs this repository does not "
              "redistribute; assets/README.md says how to obtain or rebuild each one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
