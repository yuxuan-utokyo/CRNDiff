# -*- coding: utf-8 -*-
"""R18 dry check: snapshot the frozen roots before R18 and prove nothing under them changed after it.

    python rebuttal\\r18\\frozen_sha.py snapshot --out rebuttal\\out\\R18\\frozen_sha_before.json
    python rebuttal\\r18\\frozen_sha.py snapshot --out rebuttal\\out\\R18\\frozen_sha_after.json
    python rebuttal\\r18\\frozen_sha.py compare  --before ...before.json --after ...after.json --diff ...diff.json

`compare` exits 1 on ANY added / removed / changed file or directory (and 0 only if identical).

What is recorded per file (efficiency rule, see README_R18.md):
  * every file: relative path, size, mtime_ns;
  * files <= --full-max-mb (default 200 MB): full sha256 as well;
  * files  > --full-max-mb: size + mtime_ns only (these are a handful of large .npy/.h5ad/.pt; any write by
    a normal process changes mtime_ns, and a size+mtime collision would need a deliberate timestamp reset).
  * every directory: its relative path (so a new __pycache__/ or results/ dir is caught even if empty).
A file counts as changed if size, mtime_ns or (when recorded on both sides) sha256 differ.

Roots: the reviewer's frozen set (assets/ incl. assets/legacy_root/** and RUN_LOG.md, out/runs/, out/summary/,
the CGE baseline dirs R18 reads) plus the read-only inputs R18 opens outside them (marked `input`), so a
stray write into any of them is caught too. A root that does not exist is recorded as missing (not an error).
Never deletes, never moves anything; an existing --out file is renamed to <name>.<mtime>.json first.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
import argparse                             # noqa: E402
import hashlib                              # noqa: E402
import json                                 # noqa: E402
import time                                 # noqa: E402
from pathlib import Path                    # noqa: E402

HVG = Path(__file__).resolve().parents[2]
CGE = HVG.parents[1] / "Conditional generation experiments" / "HVG2K"

ROOTS = [   # (label, path, kind)
    ("assets", HVG / "assets", "frozen"),                       # includes assets/legacy_root/** and its RUN_LOG.md
    ("out_runs", HVG / "out" / "runs", "frozen"),
    ("out_summary", HVG / "out" / "summary", "frozen"),
    ("cge_baseline_scvi", CGE / "baseline_scvi", "frozen"),
    ("cge_baseline_scanvi", CGE / "baseline_scanvi", "frozen"),
    ("cge_baseline_cfgen", CGE / "baseline_cfgen", "frozen"),   # includes upstream/ and project_folder/
    ("cge_code_overnight_cfgen", CGE / "code" / "overnight" / "cfgen", "frozen"),
    # read-only inputs outside the frozen set
    ("out_gates_basepool_ref", HVG / "out" / "gates" / "200_basepool_ref", "input"),
    ("out_machine2_mdlm_cond", HVG / "out" / "machine2" / "MDLM_cond", "input"),
    ("out_machine2_mdlm_blackout_ckpt", HVG / "out" / "machine2" / "MDLM_Blackout" / "ckpt", "input"),
    ("out_mdlm_cond_216", HVG / "out" / "mdlm_cond_216", "input"),
    ("rebuttal_out_R12_hvg2k", HVG / "rebuttal" / "out" / "R12" / "hvg2k", "input"),
    ("rebuttal_out_R10_pool_root", HVG / "rebuttal" / "out" / "R10" / "pool_root", "input"),
    ("rebuttal_r12_src", HVG / "rebuttal" / "r12_src", "input"),
    ("hvg_pkg", HVG / "hvg", "input"),
    ("experiments_pkg", HVG / "experiments", "input"),
    ("tools_pkg", HVG / "tools", "input"),
]


def sha256(p: Path, bs: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(full_max_mb: float) -> dict:
    lim = int(full_max_mb * 2 ** 20)
    out = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "hvg": str(HVG), "cge": str(CGE),
           "full_sha_max_bytes": lim, "roots": {}}
    t_all = time.time()
    for label, root, kind in ROOTS:
        t0 = time.time()
        rec = {"path": str(root), "kind": kind, "exists": root.is_dir(), "files": {}, "dirs": []}
        n_full = n_meta = 0
        nbytes = 0
        if root.is_dir():
            for dp, dns, fns in os.walk(root, followlinks=False):
                dns.sort()
                rel_d = os.path.relpath(dp, root).replace("\\", "/")
                rec["dirs"].append(rel_d)
                for fn in sorted(fns):
                    p = Path(dp) / fn
                    rel = (rel_d + "/" + fn) if rel_d != "." else fn
                    try:
                        st = p.stat()
                    except OSError as e:
                        rec["files"][rel] = {"error": repr(e)}
                        continue
                    e = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                    if st.st_size <= lim:
                        e["sha256"] = sha256(p)
                        n_full += 1
                        nbytes += st.st_size
                    else:
                        n_meta += 1
                    rec["files"][rel] = e
        rec["n_files_full_sha"] = n_full
        rec["n_files_size_mtime_only"] = n_meta
        rec["bytes_hashed"] = nbytes
        rec["seconds"] = time.time() - t0
        out["roots"][label] = rec
        print(f"[frozen_sha] {label:<34} {kind:<6} files {n_full + n_meta:>6} (sha {n_full}, size+mtime {n_meta}) "
              f"{nbytes / 1e9:7.2f} GB hashed  {rec['seconds']:6.1f} s", flush=True)
    out["seconds_total"] = time.time() - t_all
    return out


def compare(before: dict, after: dict) -> dict:
    diff = {"added": [], "removed": [], "changed": [], "dirs_added": [], "dirs_removed": [], "roots_mismatch": []}
    for label in sorted(set(before["roots"]) | set(after["roots"])):
        b, a = before["roots"].get(label), after["roots"].get(label)
        if b is None or a is None or b["exists"] != a["exists"]:
            diff["roots_mismatch"].append(label)
            continue
        bf, af = b["files"], a["files"]
        for k in sorted(set(af) - set(bf)):
            diff["added"].append(f"{label}/{k}")
        for k in sorted(set(bf) - set(af)):
            diff["removed"].append(f"{label}/{k}")
        for k in sorted(set(bf) & set(af)):
            x, y = bf[k], af[k]
            why = [f for f in ("size", "mtime_ns", "error") if x.get(f) != y.get(f)]
            if "sha256" in x and "sha256" in y and x["sha256"] != y["sha256"]:
                why.append("sha256")
            if why:
                diff["changed"].append({"file": f"{label}/{k}", "fields": why, "before": x, "after": y})
        bd, ad = set(b["dirs"]), set(a["dirs"])
        diff["dirs_added"] += [f"{label}/{k}" for k in sorted(ad - bd)]
        diff["dirs_removed"] += [f"{label}/{k}" for k in sorted(bd - ad)]
    diff["n_differences"] = sum(len(v) for k, v in diff.items() if isinstance(v, list))
    diff["identical"] = diff["n_differences"] == 0
    return diff


def _write_new(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():                                     # never delete: rename the previous one aside
        p.rename(p.with_name(f"{p.stem}.{time.strftime('%Y%m%d_%H%M%S', time.localtime(p.stat().st_mtime))}{p.suffix}"))
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out", required=True)
    s.add_argument("--full-max-mb", type=float, default=200.0)
    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--diff", required=True)
    a = ap.parse_args()
    if a.cmd == "snapshot":
        out = Path(a.out).resolve()
        if not str(out).lower().startswith(str((HVG / "rebuttal" / "out" / "R18").resolve()).lower()):
            raise SystemExit(f"snapshot must be written under rebuttal/out/R18: {out}")
        snap = snapshot(a.full_max_mb)
        _write_new(out, snap)
        print(f"[frozen_sha] snapshot -> {out}  ({snap['seconds_total']:.0f} s)", flush=True)
        return 0
    before = json.loads(Path(a.before).read_text(encoding="utf-8"))
    after = json.loads(Path(a.after).read_text(encoding="utf-8"))
    if before.get("full_sha_max_bytes") != after.get("full_sha_max_bytes"):
        print("[frozen_sha] WARNING: snapshots used different --full-max-mb", flush=True)
    diff = compare(before, after)
    diff.update({"before": str(Path(a.before).resolve()), "after": str(Path(a.after).resolve()),
                 "written_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _write_new(Path(a.diff).resolve(), diff)
    if diff["identical"]:
        print("[frozen_sha] PASS: no file or directory under any watched root changed", flush=True)
        return 0
    print(f"[frozen_sha] FAIL: {diff['n_differences']} difference(s)", flush=True)
    for k in ("roots_mismatch", "added", "removed", "dirs_added", "dirs_removed"):
        for x in diff[k][:20]:
            print(f"  {k}: {x}", flush=True)
    for x in diff["changed"][:20]:
        print(f"  changed: {x['file']} {x['fields']}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
