# -*- coding: utf-8 -*-
"""R12 CPU window item 4: 11-class replacement TSTR (tab:hvg_tstr, 212_tstr11.json) for the CountsDiff arm.
Same route as the CFGen row (tools/t212_tstr11.py --single-arm, prereg section 5): analysis/e33_tstr11.py is imported and NOT changed,
only its module constants are overridden:
  T11.MAN_DIR = rebuttal/out/R12/tstr11/manifest   (copy of out/e33_212/manifest/E30plus_manifest.json with the 3 CFGen entries
                                                    re-pointed to the CountsDiff final arrays; the slot name CFGen is only a carrier)
  T11.WORK    = rebuttal/out/R12/tstr11/single     T11.SUMMARY = rebuttal/out/R12/tstr11 (its fit_score diff file lands here,
                                                    not in out/summary)          T11.ARMS_5 = ["CFGen"]
Test set = all 45,145 val cells; 8 non-target classes from real train; test.npy is not read. 1 thread (affinity 0x4).
    cd Experiments/HVG && python -m rebuttal.r12_cd_tstr --build-manifest ; --seed S (each of MS.SEEDS) ; --aggregate
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np   # noqa: E402  (BLAS pinned to 1 thread before e33_tstr11 resets the env to 2)
import argparse, hashlib, json, statistics as st, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvg import config as C                       # noqa: E402
from analysis import e33_tstr11 as T11            # noqa: E402
from analysis import e33_multiseed as MS          # noqa: E402
W = ROOT / "rebuttal" / "out" / "R12" / f"tstr11_k128_s{os.environ['CD_TRAIN_SEED']}"; MAN_DIR = W / "manifest"; MAN = MAN_DIR / "E30plus_manifest.json"   # k128 copy
SRC_MAN = C.OUT / "e33_212" / "manifest" / "E30plus_manifest.json"
TS = os.environ["CD_TRAIN_SEED"]   # k128 copy: CountsDiff K128 g1.0, train seed from env, one output dir per train seed
CD = ROOT / "out" / "runs" / f"rebuttal_CD_K128_s{TS}"
CD_FILES = {"Endothelial": CD / "Endothelial_K128_g1p0_n10057" / f"CountsDiff_Endothelial_K128_g1p0_n10057_train{TS}_samp20261061_20261062.npy",
            "Myeloid": CD / "Myeloid_K128_g1p0_n2500" / f"CountsDiff_Myeloid_K128_g1p0_n2500_train{TS}_samp20261061.npy",
            "Neuronal": CD / "Neuronal_K128_g1p0_n2500" / f"CountsDiff_Neuronal_K128_g1p0_n2500_train{TS}_samp20261061.npy"}
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def build_manifest():
    man = json.loads(SRC_MAN.read_text(encoding="utf-8")); n = 0
    for it in man["items"]:
        if it.get("method") == "CFGen" and it.get("target") in CD_FILES:
            t = it["target"]; it["_r12_was"] = it["npy"]; it["npy"] = str(CD_FILES[t]); it["seed"] = 20261061
            it["_r12_note"] = "CFGen slot re-pointed to the CountsDiff final array (carrier only)"; n += 1
    if n != 3: raise SystemExit(f"expected 3 CFGen entries, re-pointed {n}")
    MAN_DIR.mkdir(parents=True, exist_ok=True); MAN.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[manifest] {MAN} sha {sha(MAN)[:16]} (source {SRC_MAN} unchanged, sha {sha(SRC_MAN)[:16]})"); return 0
def aggregate():
    rows = {s: json.loads((W / "single" / f"raw_11class_seed_{s}.json").read_text(encoding="utf-8")) for s in MS.SEEDS}
    def agg(arm, key, t=None):
        v = [r["arms"][arm][key] if t is None else r["arms"][arm][key][t] for r in rows.values()]
        return {"mean": st.mean(v), "sd": st.stdev(v), "per_seed": v}
    out = {"arm": "CountsDiff (carried in the CFGen slot)", "eval_seeds": list(MS.SEEDS), "sd": "ddof=1 over 5 eval seeds",
           "test": "all 45,145 val cells, 11 classes", "arms": {}}
    for arm, name in (("REAL (ceiling)", "REAL (ceiling)"), ("CFGen", "CountsDiff")):
        out["arms"][name] = {"macro_F1_11class": agg(arm, "macro_F1"),
                             **{f"F1_{t}": agg(arm, "per_type_F1", t) for t in CD_FILES},
                             **{f"recall_{t}": agg(arm, "per_type_recall", t) for t in CD_FILES},
                             **{f"precision_{t}": agg(arm, "per_type_precision", t) for t in CD_FILES}}
    out["arm_arrays"] = rows[MS.SEEDS[0]]["arm_arrays"]; out["walltime_s"] = sum(r["walltime_s"] for r in rows.values())
    (W / "r12_countsdiff_tstr11.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for n, a in out["arms"].items():
        print(n, {k: f"{v['mean']:.4f}±{v['sd']:.4f}" for k, v in a.items() if k.startswith(("macro", "F1"))})
    return 0
ap = argparse.ArgumentParser(); ap.add_argument("--build-manifest", action="store_true"); ap.add_argument("--seed", type=int); ap.add_argument("--aggregate", action="store_true")
a = ap.parse_args()
if a.build_manifest: raise SystemExit(build_manifest())
if a.aggregate: raise SystemExit(aggregate())
T11.MAN_DIR = MAN_DIR; T11.WORK = W / "single"; T11.SUMMARY = W; T11.ARMS_5 = ["CFGen"]; T11.N_THREADS = 1
print(f"[override] MAN_DIR {T11.MAN_DIR}  WORK {T11.WORK}  SUMMARY {T11.SUMMARY}  ARMS_5 {T11.ARMS_5}", flush=True)
raise SystemExit(T11.run("11class", a.seed))
