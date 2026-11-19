# -*- coding: utf-8 -*-
"""R14 scoring for the 3-generator table (vice 10:2x: Windows scores with the s25 R14 scripts; val first, test read once; no coverage).
Frozen scorers, imported unedited (sha-asserted here):
  analysis/score_intersect.py      86da6ce1...  (W1 mean/median, sliced W1 = jSWD, UAP1+ fraction; build_reference; archived-floor check)
  analysis/score_e21_intersect.py  e3693a2a...  (MMD2 / PCC / SCC)
  analysis/score_purity.py         9b1cb6a0...  (CellTypist purity; split-free, run as is via its CLI)
Modes:
  --split val:  (1) score_intersect.main(["--experiment", E, "--out", ...]) UNCHANGED (incl. the bit-exact archived-val-floor check);
                (2) score_e21_intersect.main with only its run root (out/runs/intersect_ladder -> out/runs/<E>), its arm list (OUR_ARMS ->
                    the arms present under E) and its baseline routes (ROUTES -> none) swapped in-process.
  --split test: the SAME code paths with build_reference swapped for a copy whose only change is the file pair
                val.npy/val_celltype.npy -> test.npy/test_celltype.npy (same rule: Endothelial & gene128 > 0, same SPLIT_SEED halves,
                same SUB_SEED/PROJ_SEED, matched n = min(halves, arm sizes)). The archived floor is a val constant, so on test the floor is
                computed (half B vs half A) and reported, not compared; e21's matched_n (read from the val panel json) is replaced by the
                test matched n. The test read covers s25 (intersect_ladder a1) and gen27/30 (rebuttal_R14g) in ONE process, and every test
                file read is logged with sha256 in rebuttal/out/R14/test_read_log.json.
    python -u rebuttal/r14_score.py --split val  --experiment rebuttal_R14g
    python -u rebuttal/r14_score.py --split test
"""
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HVG = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HVG))
from hvg import config as C
SHA = {"analysis/score_intersect.py": "86da6ce1b56ad967cdd0e858c3fa11a3b899f58267a678ac34e3b634928892f8",
       "analysis/score_e21_intersect.py": "e3693a2a2f883ad5a7a96a34d4fa029c496734e2c8f689c2eb97c75958483cc4"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for k, v in SHA.items():
    assert sha(HVG / k) == v, f"{k} changed"
import analysis.score_intersect as SI
OUT = HVG / "rebuttal" / "out" / "R14"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=("val", "test"), required=True); ap.add_argument("--experiment", default="rebuttal_R14g")
a = ap.parse_args()


class _Runs:   # C.RUNS stand-in for score_e21_intersect: RUNS / "intersect_ladder" -> RUNS / <experiment>; everything else unchanged
    def __init__(self, exp): self.exp = exp
    def __truediv__(self, x): return C.RUNS / (self.exp if x == "intersect_ladder" else x)


class _CShim:
    def __init__(self, exp): self._r = _Runs(exp)
    def __getattr__(self, k): return self._r if k == "RUNS" else getattr(C, k)


def e21_run(exp, out, matched_n=None):
    import analysis.score_e21_intersect as E21
    arms = sorted(p.name for p in (C.RUNS / exp).iterdir() if p.is_dir() and list(p.glob("*.json")))
    E21.OUR_ARMS = [(arm, f"{exp}/{arm}") for arm in arms]; E21.ROUTES = []; E21.C = _CShim(exp)
    if matched_n is not None:
        pa = OUT / f"_panel_matched_n_{exp}.json"; pa.write_text(json.dumps({"reference": {"matched_n": int(matched_n)}}), encoding="utf-8")
        E21.PANEL_A = pa
    E21.main(["--out", str(out)])


if a.split == "val":
    SI.main(["--experiment", a.experiment, "--out", str(OUT / f"intersect_{a.experiment}_val.json")])
    e21_run(a.experiment, OUT / f"e21_intersect_{a.experiment}_val.json")
    sys.exit(0)

# ---- test: one read, s25 a1 + gen27/30 -------------------------------------------------------------------------------------------
D = C.LEGACY_DATA / "all"; files = {f: D / f for f in ("test.npy", "test_celltype.npy")}
log = {"what": "R14 test read: s25 intersect_ladder/a1 + rebuttal_R14g a1_g20260627/a1_g20260630, W1/jSWD/UAP1+ "
               "fraction + MMD2/PCC/SCC; reference = test Endothelial & UAP1+ (gene128 > 0)", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
       "files": {k: {"path": str(v), "sha256": sha(v)} for k, v in files.items()}, "script_sha256": sha(Path(__file__))}


def build_reference_test(arm_sizes):
    """score_intersect.build_reference with ONLY the file pair changed to the test split."""
    Xv = np.load(files["test.npy"], mmap_mode="r")
    yv = np.load(files["test_celltype.npy"], allow_pickle=True).astype(str)
    sel = np.flatnonzero((yv == "Endothelial") & (np.asarray(Xv)[:, SI.GENE] > 0))
    rs = np.random.default_rng(SI.SPLIT_SEED); perm = rs.permutation(len(sel)); half = len(sel) // 2
    idx_A, idx_B = np.sort(sel[perm[:half]]), np.sort(sel[perm[half:2 * half]])
    n = int(min([len(idx_A), len(idx_B)] + list(arm_sizes)))
    rng = np.random.default_rng(SI.SUB_SEED)
    ref = np.asarray(Xv[idx_A[np.sort(rng.choice(len(idx_A), n, replace=False))]], dtype=np.float64)
    flo = np.asarray(Xv[idx_B[np.sort(rng.choice(len(idx_B), n, replace=False))]], dtype=np.float64)
    return {"ref": ref, "flo": flo, "n": n, "n_total": int(len(sel)), "half_sizes": [int(len(idx_A)), int(len(idx_B))],
            "data": str(files["test.npy"].resolve())}


SI.build_reference = build_reference_test
SI.ARCHIVED_FLOOR = None   # the identity check is val-specific; the test path below never calls it
M, sw1 = SI.load_frozen()
from hvg.io import load_runs
rows, sizes = [], []
for exp, arms in (("intersect_ladder", ("a1",)), (a.experiment, None)):
    for r in load_runs(exp):
        if r.get("request") != SI.REQUEST or (arms and r.get("arm") not in arms): continue
        p = Path(r["_path"]).with_suffix(".npy"); rows.append((exp, r, p)); sizes.append(int(np.load(p, mmap_mode="r").shape[0]))
R = build_reference_test(sizes); ref, flo, n = R["ref"], R["flo"], R["n"]
log["rows_read"] = {"test Endothelial & UAP1+": R["n_total"], "halves": R["half_sizes"], "matched_n": n}
print(f"[ref TEST] Endothelial & UAP1+ n = {R['n_total']} halves {R['half_sizes']} matched n {n}", flush=True)
proj = M.make_projections(ref.shape[1], n_proj=SI.N_PROJ, seed=SI.PROJ_SEED)
floor = {**SI.score(flo, ref, proj, M, sw1), "source": "test.npy, disjoint half of the reference", "uap1_pos_frac": 1.0}
table = json.loads((C.TABLES / f"logc_{SI.REQUEST}.json").read_text(encoding="utf-8")); real_rate = table["provenance"]["detection_rate_in_type"]
out_rows = []
for exp, r, p in rows:
    arr = np.load(p, mmap_mode="r"); rng = np.random.default_rng(SI.SUB_SEED)
    s = np.sort(rng.choice(arr.shape[0], n, replace=False)); g = np.asarray(arr[s], dtype=np.float64); full = np.asarray(arr[:, SI.GENE])
    row = {"experiment": exp, "stem": r["stem"], "arm": r.get("arm"), "seed": r.get("seed"), "checkpoint": Path(str(r.get("checkpoint"))).name,
           "uap1_pos_frac_delivered": float((full > 0).mean()), "uap1_pos_frac_in_sample": float((g[:, SI.GENE] > 0).mean()),
           "uap1_pos_frac_real_in_type": real_rate, "enrichment_over_real": float((full > 0).mean() / real_rate), **SI.score(g, ref, proj, M, sw1)}
    row["mW1_over_floor"] = row["marginal_w1_mean"] / floor["marginal_w1_mean"]; row["jSWD_over_floor"] = row["sliced_w1"] / floor["sliced_w1"]
    out_rows.append(row); print(f"[score TEST] {exp}/{row['arm']} seed {row['seed']} mW1 {row['marginal_w1_mean']:.4f} jSWD {row['sliced_w1']:.4f}", flush=True)
(OUT / "intersect_R14_test.json").write_text(json.dumps({"split": "test", "reference": {k: R[k] for k in ("n", "n_total", "half_sizes", "data")},
    "floor": floor, "rows": out_rows, "script_sha256": log["script_sha256"]}, indent=1), encoding="utf-8")
for exp in ("intersect_ladder", a.experiment):
    import analysis.score_e21_intersect as E21
    E21.build_reference = build_reference_test   # e21 imported build_reference by name
    e21_run(exp, OUT / f"e21_intersect_{exp}_test.json", matched_n=n)
log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
(OUT / "test_read_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
print("[test read logged]", OUT / "test_read_log.json", flush=True)
