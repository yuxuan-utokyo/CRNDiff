# -*- coding: utf-8 -*-
"""R14 Fano data for the Mac. CPU only, 1 thread, read-only on every source; writes only into
_data_exchange/r14_fano/. Resumable item by item; between items it stops (exit 3) once line 1 must be free for MDLM
("R13D STEP1 DONE" logged and "R13D MDLM10 exit" not yet), and is relaunched after MDLM.
  (1) slices: the 9 Table 9 / 200 unconditional pools (OURS, scVI, scANVI x seeds 20260931/32/33), the FIRST 20,000 rows,
      as int16 (asserted integral and within int16), scipy CSR in 4 row blocks of 5,000, save_npz(compressed=True),
      <method>_s<seed>_rows<lo>-<hi>.npz; SOURCES.tsv (pool path, its sha256, the pool_sha256 in 200_uncond_fidelity.json, match).
  (2) copies, byte for byte: make_F8_dispersion_baselines.py, tools/t200_uncond_fidelity.py (the board's
      "score_unconditional_fidelity"), tools/t200_pool_composition.py ("score_pool_composition"), 200_uncond_fidelity.json,
      202_tableB1_full.json.
  (3) per-cell CellTypist predicted_labels for the 9 slices and the 20,000-cell val reference (seed 20261200,
      np.sort(default_rng(seed).choice(N_val, 20000, replace=False)) as in t200_uncond_fidelity.py), with
      analysis.score_purity.annotate (the call t200_pool_composition.py makes: normalize_total(1e4)+log1p,
      Healthy_Adult_Heart.pkl, majority_voting=False). The per-cell labels do not depend on the v7/v8 label-set mapping.
      CSV: row,label (row = row in the slice, or the original val index for the reference).
  SHA256SUMS (2 columns, LF, no BOM) is rewritten over every file in the folder after each item.
    python -m rebuttal.r14_fano_prep
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import csv, hashlib, json, shutil, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import sparse

X = Path(r"work\exchange\r14_fano")
LOG = ROOT / "rebuttal" / "logs"
POOLS = ROOT / "out" / "uncond_pools"
J200 = ROOT / "out" / "summary" / "200_uncond_fidelity.json"
J202 = ROOT / "out" / "summary" / "202_tableB1_full.json"
F8 = ROOT.parents[1] / "Conditional generation experiments" / "HVG2K" / "viz" / "make_F8_dispersion_baselines.py"
COPIES = [F8, ROOT / "tools" / "t200_uncond_fidelity.py", ROOT / "tools" / "t200_pool_composition.py", J200, J202]
METHODS = [("OURS", "ours_uncond_seed{s}_n160000.npy"), ("scVI", "scvi_uncond_seed{s}_n100000.npy"), ("scANVI", "scanvi_uncond_seed{s}_n100000.npy")]
SEEDS = [20260931, 20260932, 20260933]
N, BLK = 20000, 5000
VAL_SUB_SEED = 20261200


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def pause():
    s2 = (LOG / "gpu_slot2.log").read_text(encoding="utf-8", errors="ignore")
    s1 = (LOG / "gpu_line.log").read_text(encoding="utf-8", errors="ignore")
    if "R13D STEP1 DONE" in s2 and "R13D MDLM10 exit" not in s1:
        print("[pause] line 1 is due for MDLM; stopping between items", flush=True); write_sums(); raise SystemExit(3)


def write_sums():
    rows = [f"{sha(p)}  {p.name}\n" for p in sorted(X.iterdir()) if p.is_file() and p.name != "SHA256SUMS" and not p.name.endswith(".part")]
    (X / "SHA256SUMS").write_bytes("".join(rows).encode("utf-8"))


def json_sha(name):
    d = json.loads(J200.read_text(encoding="utf-8"))
    for r in d.get("rows", []):
        if str(r.get("pool", "")).replace("\\", "/").endswith(name):
            return r.get("pool_sha256")
    return None


def atomic_save_npz(path, m):
    tmp = path.with_name(path.name + ".part.npz"); sparse.save_npz(tmp, m, compressed=True); os.replace(tmp, path)


def main():
    X.mkdir(parents=True, exist_ok=True)
    src = X / "SOURCES.tsv"
    if not src.exists():
        src.write_bytes("method\tseed\tpool_path\tpool_sha256\tsha256_in_200_uncond_fidelity\tmatch\tslice\n".encode("utf-8"))
    done_src = src.read_text(encoding="utf-8")
    # (1) slices
    for m, pat in METHODS:
        for s in SEEDS:
            p = POOLS / pat.format(s=s); last = X / f"{m}_s{s}_rows{N - BLK}-{N}.npz"
            if last.exists() and f"{m}\t{s}\t" in done_src:
                continue
            pause()
            x = np.load(p, mmap_mode="r")[:N]
            xa = np.asarray(x)
            if not np.array_equal(xa, np.rint(xa)) or xa.min() < 0 or xa.max() > np.iinfo(np.int16).max:
                raise SystemExit(f"{p.name}: first {N} rows are not non-negative integers within int16")
            xi = xa.astype(np.int16)
            for lo in range(0, N, BLK):
                atomic_save_npz(X / f"{m}_s{s}_rows{lo}-{lo + BLK}.npz", sparse.csr_matrix(xi[lo:lo + BLK]))
            h = sha(p); hj = json_sha(p.name)
            with open(src, "ab") as f:
                f.write(f"{m}\t{s}\t{p.relative_to(ROOT).as_posix()}\t{h}\t{hj}\t{h == hj}\tfirst {N} rows, int16, CSR, 4 blocks of {BLK}\n".encode("utf-8"))
            done_src = src.read_text(encoding="utf-8")
            print(f"[slice] {m} s{s} dtype {x.dtype} max {int(xa.max())} sha {h[:12]} json {str(hj)[:12]} match {h == hj}", flush=True)
            write_sums()
    # (2) copies
    for p in COPIES:
        t = X / p.name
        if not t.exists():
            shutil.copy2(p, t)
            print(f"[copy] {p.name}", flush=True)
    write_sums()
    # (3) CellTypist labels
    from analysis.score_purity import annotate
    from hvg import config as C
    genes = json.loads((C.LEGACY_DATA / "all" / "meta.json").read_text(encoding="utf-8"))["genes"]
    jobs = [(f"{m}_s{s}", POOLS / pat.format(s=s), None) for m, pat in METHODS for s in SEEDS] + [("val_ref_seed20261200", C.LEGACY_DATA / "all" / "val.npy", "val")]
    for name, p, kind in jobs:
        out = X / f"{name}_celltypist_labels.csv"
        if out.exists():
            continue
        pause()
        x = np.load(p, mmap_mode="r")
        if kind == "val":
            idx = np.sort(np.random.default_rng(VAL_SUB_SEED).choice(x.shape[0], N, replace=False))
        else:
            idx = np.arange(N)
        lab = annotate(np.asarray(x[idx], dtype=np.float64), genes)
        tmp = out.with_name(out.name + ".part")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["row", "label"]); w.writerows(zip(idx.tolist(), lab.tolist()))
        os.replace(tmp, out)
        u, c = np.unique(lab, return_counts=True)
        print(f"[labels] {name} n={len(lab)} top {sorted(zip(c.tolist(), u.tolist()))[-3:]}", flush=True)
        write_sums()
    print("[done] R14 FANO PREP DONE", flush=True)


if __name__ == "__main__":
    main()
