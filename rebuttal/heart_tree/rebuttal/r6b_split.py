# -*- coding: utf-8 -*-
"""R6b: rule in rebuttal/out/R6/criterion_R6b.md. Val halves; select tau on A by W1; score on B.
    cd Experiments/HVG && python -m rebuttal.r6b_split
"""
import importlib.util, json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "8"
from pathlib import Path
import numpy as np, torch
sys.dont_write_bytecode = True
from rebuttal.w2_runs import ROOT, RUNS, TARGETS, M as MM, N_OUT, real_cells
torch.set_num_threads(8)
from analysis.score_e21 import frozen
E = frozen()
spec = importlib.util.spec_from_file_location("_m8", ROOT / "analysis/_frozen/metrics_exp08.py"); m8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m8)
OUT = ROOT / "rebuttal/out/R6"; TAUS = ["0p2", "0p4", "1p0"]; VAL = {"0p2": 0.2, "0p4": 0.4, "1p0": 1.0}; SEEDS = (20260987, 20260988, 20260989)
DEP = {"Endothelial": "0p4", "Myeloid": "0p2", "Neuronal": "0p2"}
pp = json.loads((ROOT / "out/summary/215_paper_numbers_v8.json").read_text(encoding="utf-8"))
V8 = {(r["request"], float(r["row"])): r["mean"] for r in pp["rows"] if r["table"] == "tab:hvg_tau" and r["metric"] == "purity_abs"}


def path(t, tau, s):
    stem = f"{t}_residual_hvgtwist_fk_tiltevery_trigany_K32_M{MM[t]}_N{N_OUT[t]}_alpha1p0_J16_int8_th0p5_tau{tau}_cc256_odds_seed{s}.npy"
    return (RUNS / "table1_ours" / f"{t}_a1" / stem) if tau == DEP[t] else (RUNS / "tau_sweep" / f"{t}_tau{tau}" / stem)


def w1_mean(a, b):
    n = min(len(a), len(b)); return float(np.abs(np.sort(a[:n], 0) - np.sort(b[:n], 0)).mean())


def scores(gen, ref_raw):
    ref, _ = E.cp10k_log1p(ref_raw[:2000]); g, _ = E.cp10k_log1p(gen[:2000])
    rt, gt = torch.from_numpy(ref), torch.from_numpy(g)
    sig = E.exact_bandwidth(rt); krr = E.kernel_mean(rt, rt, sig); mmd, _ = E.mmd2_biased(rt, gt, sig, krr)
    fr, fg = E.cp10k_log1p(ref_raw)[0].mean(0), E.cp10k_log1p(gen)[0].mean(0)
    return float(mmd), float(np.corrcoef(fr, fg)[0, 1])


res = {}
for t in ["Neuronal", "Myeloid", "Endothelial"]:
    xv = real_cells("val", t); g = {tau: [np.asarray(np.load(path(t, tau, s), mmap_mode="r"), dtype=np.float64) for s in SEEDS] for tau in TAUS}
    a0 = g[DEP[t]][0][:300, 5]; assert abs(m8.w1_1d(a0, xv[:300, 5]) - np.abs(np.sort(a0) - np.sort(xv[:300, 5])).mean()) < 1e-12
    rows = []
    for r in range(1, 21):
        rng = np.random.default_rng(20261107 + r); p = rng.permutation(len(xv)); h = len(p) // 2; A, B = xv[p[:h]], xv[p[h:]]
        pick = lambda G, n: G[rng.choice(len(G), n, replace=False)]
        wA = {tau: float(np.mean([w1_mean(pick(G, len(A)), A) for G in g[tau]])) for tau in TAUS}
        sel = min(TAUS, key=lambda x: (wA[x], VAL[x]))
        wB = {tau: float(np.mean([w1_mean(pick(G, len(B)), B) for G in g[tau]])) for tau in TAUS}
        out = {"r": r, "selected": sel, "W1_A": wA, "W1_B": wB}
        for tag, tau in (("selected", sel), ("deployed", DEP[t])):
            ms = [scores(pick(G, min(len(B), len(G))), B) for G in g[tau]]
            out[f"{tag}_mmd2_B"] = float(np.mean([x[0] for x in ms])); out[f"{tag}_pcc_B"] = float(np.mean([x[1] for x in ms]))
            out[f"{tag}_v8"] = V8[(t, VAL[tau])]
        rows.append(out); print(t, r, sel, {k: round(v, 4) for k, v in wB.items()}, round(out["selected_mmd2_B"], 4), flush=True)
    res[t] = {"n_half": len(xv) // 2, "deployed": DEP[t], "rows": rows}
    (OUT / "r6b_numbers.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("[done]", flush=True)
