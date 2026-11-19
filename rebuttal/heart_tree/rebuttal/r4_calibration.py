# -*- coding: utf-8 -*-
"""R4 (z7bw: odds with no upper clip and no calibration check). Analysis only; sampler untouched.

For each residual classifier: reliability and ECE on its own held-out split (proposal_val vs target_val),
odds distribution on proposal_candidate, and the share of total FK weight held by the top 1% of odds
(alpha = 1, as deployed). Held-out class balance differs from the fit balance, so reliability is also
reported after the standard prior-shift correction of the logit. Target cells are from the train split
(as in the classifier's own split). Test split not read.

    cd Experiments/HVG && python -m rebuttal.r4_calibration
"""
import glob, json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = os.environ.get("R4_THREADS", "8")   # board 11-08 12:09: 2 threads in the CPU windows
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hvg.sampler import PortableHGBT
from rebuttal.w2_runs import DATA, TARGETS, COL
OUT = ROOT / "rebuttal/out/R4"; OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "assets/legacy_root/Baseline/OURS/results"
R10POOL = ROOT / "rebuttal/out/R10/pool_root/Baseline/OURS/results"
R4POOL = ROOT / "rebuttal/out/R4/pool_root/Baseline/OURS/results"   # board 11-08 17:36: Endothelial/Myeloid 20260625 pools regenerated here
BINS = np.linspace(0, 1, 16)


def norm(x):
    z = np.asarray(x, dtype=np.float32); lib = z.sum(1, dtype=np.float64)
    return np.log1p(z * (1e4 / np.maximum(lib, 1.0)).astype(np.float32)[:, None])


def ece(p, y):
    idx = np.clip(np.digitize(p, BINS) - 1, 0, 14); rows = []; e = 0.0
    for b in range(15):
        m = idx == b
        if m.any():
            rows.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())]); e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e), rows


def proposal_of(t, gen):
    if gen == 20260625:
        h = glob.glob(str(R10POOL / f"*CONDFULL_{t}_tables_kz_n20000_g0p4_noa4.npy")) or glob.glob(str(R4POOL / f"*CONDFULL_{t}_tables_kz_n20000_g0p4_noa4.npy"))
    else:
        h = glob.glob(str(RES / f"linspace_T_O_K32_spilot_run214_s{gen}_samp20260902_*CONDFULL_{t}_tables_kz_s{gen}_n20000_g0p4_noa4.npy"))
    return Path(h[0]) if h else None


Xtr = np.load(DATA / "train.npy", mmap_mode="r")
res = {}; rel = {}
for t in TARGETS:
    for gen, name in ((20260625, t), (20260627, f"{t}_s20260627"), (20260630, f"{t}_s20260630")):
        d = ROOT / "assets/models/residual" / name
        q_path = proposal_of(t, gen); key = f"{t}|{gen}"
        if q_path is None:
            res[key] = {"status": "proposal pool not on disk"}; print(key, "no proposal", flush=True); continue
        z = np.load(d / "split_indices.npz"); log = json.loads((d / "residual_log.json").read_text(encoding="utf-8"))
        clf = PortableHGBT(d / "residual_classifier_portable.npz"); f = z["classifier_features"]
        q = np.load(q_path, mmap_mode="r")
        px = lambda a: clf.predict_probability(norm(a)[:, f].astype(np.float64))
        pp = px(np.asarray(Xtr[np.sort(z["target_val"])])); pn = px(np.asarray(q[np.sort(z["proposal_val"])]))
        pc = px(np.asarray(q[np.sort(z["proposal_candidate"])]))
        p = np.r_[pp, pn]; y = np.r_[np.ones(len(pp)), np.zeros(len(pn))]
        n_fp, n_fn = len(z["target_fit"]), len(z["proposal_fit"]); pif, piv = n_fp / (n_fp + n_fn), len(pp) / len(p)
        lg = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1)) + np.log(piv / (1 - piv)) - np.log(pif / (1 - pif))
        p2 = 1 / (1 + np.exp(-lg))
        e1, r1 = ece(p, y); e2, r2 = ece(p2, y)
        o = pc / np.clip(1 - pc, 1e-12, None); os_ = np.sort(o)[::-1]; k = max(1, int(round(0.01 * len(o))))
        auc = float((np.argsort(np.argsort(p))[y == 1].mean() - (len(pp) - 1) / 2) / len(pn))
        res[key] = {"status": "ok", "n_val_pos": len(pp), "n_val_neg": len(pn), "fit_prior": pif, "val_prior": piv, "auc_recomputed": auc, "auc_logged": log["classifier"]["heldout_roc_auc"],
                    "ece_raw": e1, "ece_prior_corrected": e2, "odds_median": float(np.median(o)), "odds_p99": float(np.percentile(o, 99)), "odds_max": float(o.max()),
                    "top1pct_weight_share": float(os_[:k].sum() / o.sum()), "top10_weight_share": float(os_[:10].sum() / o.sum()), "ess_frac_candidate": float(o.sum() ** 2 / (o ** 2).sum() / len(o)),
                    "n_candidate": len(o), "proposal": q_path.name}
        rel[key] = r2
        print(key, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in res[key].items() if k_ not in ("proposal", "status")}, flush=True)
(OUT / "r4_numbers.json").write_text(json.dumps({"res": res, "reliability_prior_corrected": rel}, indent=1), encoding="utf-8")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.5})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True); GC = {20260625: "#2a78d6", 20260627: "#eb6834", 20260630: "#1baf7a"}
for a, t in zip(ax, TARGETS):
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 1.5)))
    for gen, c in GC.items():
        k = f"{t}|{gen}"
        if k in rel:
            v = np.array(rel[k]); a.plot(v[:, 0], v[:, 1], "-o", ms=2.5, lw=1, color=c, label=f"train {gen}")
    a.set_title(t, fontsize=8, pad=3); a.set_xlabel("Predicted probability (prior corrected)", color=MUTED, fontsize=7); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.spines[["top", "right"]].set_visible(False); a.grid(lw=0.3, color=GRID); a.set_axisbelow(True)
ax[0].set_ylabel("Observed target fraction", color=MUTED); h, l = ax[-1].get_legend_handles_labels()
for a in ax:
    h2, l2 = a.get_legend_handles_labels()
    if len(h2) > len(h): h, l = h2, l2
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02)); fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
fig.savefig(OUT / "R4_reliability.png", dpi=300)
