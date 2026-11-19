#!/usr/bin/env python
r"""R39: where the off-target generated cells go. No test read:
  - every (method, seed) array of the v3 manifest conditioned on the target is labelled with the frozen purity LR (219e9bce, prediction as
    score_with_npz.py); per method the share of the 5 most frequent predicted labels (by the method's 3-seed mean) + the rest;
  - those labels' frequency in the real TRAIN set;
  - the 5 cell types whose train centroid is nearest to the target's train centroid in the train-fit 50-D PCA (067dc34d), Euclidean.
    python r39_offtarget.py OUT.md OUT.json
"""
import hashlib, json, sys
import numpy as np
W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
sys.path.insert(0, f"{W}/scripts/r13_score"); import r13_score as R
NPZ = f"{W}/runs/r16/purity_lr/purity_lr_hlca.npz"; assert sha(NPZ) == "219e9bce"; z = np.load(NPZ)
pred = lambda x: np.concatenate([z["classes"][(R.log_cp10k(x[i:i + 5000]) @ z["coef"].T + z["intercept"]).argmax(1)] for i in range(0, len(x), 5000)])
man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json")); METH = ["CRNDiff", "CFGen", "CountsDiff", "scVI", "scANVI"]; CL = ["plasma cell", "ionocyte"]
XT = np.load(f"{W}/data/r16_hlca_fit/train.npy", mmap_mode="r"); YT = np.load(f"{W}/data/r16_hlca_fit/train_celltype.npy", allow_pickle=True).astype(str)
P = np.load(f"{W}/runs/r29_30/hlca_pca50_train.npz"); assert sha(f"{W}/runs/r29_30/hlca_pca50_train.npz") == "067dc34d"
def cp(c): x = np.asarray(c, dtype=np.float64); lib = x.sum(1, keepdims=True); return np.log1p(x / np.maximum(lib, 1) * 1e4)
pca50 = lambda c: (cp(c).astype(np.float32) - P["mean"]) @ P["components"].T
types = sorted(set(YT)); cen = {}
for t in types:
    idx = np.flatnonzero(YT == t); s = np.zeros(50);
    for i in range(0, len(idx), 20000): s += pca50(XT[idx[i:i + 20000]]).sum(0)
    cen[t] = s / len(idx)
freq = {t: float((YT == t).mean()) for t in types}
res = {"targets": {}}; L = ["# R39 baseline where off-target cells go (HLCA, no test reads)", "",
     "**Question to answer.** Baselines have low purity on rare classes; do the non-target cells they generate fall on cell types close to the target and abundant in the training set?", "",
     "**Method.** The same frozen logistic-regression classifier used for purity labels the 2,500 generated cells with condition = target for each method and seed (generated cells only, no test reads); for each method the top 5 labels by mean share over 3 seeds are kept and the rest merged."
     "label frequencies in the training set come from real train cells. Nearness is the Euclidean distance between class centroids in a 50-dim PCA fitted on the training set (real train cells, no test reads).", ""]
for t in CL:
    d = {k: float(np.linalg.norm(cen[k] - cen[t])) for k in types if k != t}; near = sorted(d, key=d.get)[:5]
    res["targets"][t] = {"nearest5": {k: d[k] for k in near}, "methods": {}}
    L += [f"**{t}**(training-set frequency {freq[t]:.5f})", "", "50 the 5 types whose class centroids are nearest in the PCA of dimension" + ",".join(f"{k}(distance {d[k]:.2f}, training-set frequency {freq[k]:.4f})" for k in near) + ".", "",
          "| method | top 5 predicted labels (mean share over 3 seeds; training-set frequency in parentheses; ★ marks one of the 5 nearest types) | rest |", "|---|---|---|"]
    for m in METH:
        es = [e for e in man if e["method"] == m and e["target"] == t]; assert len(es) == 3
        sh = []
        for e in es:
            p = pred(np.load(e["path"])); u, c = np.unique(p, return_counts=True); sh.append(dict(zip(u, c / len(p))))
        labs = sorted({k for s in sh for k in s}, key=lambda k: -np.mean([s.get(k, 0) for s in sh])); top = labs[:5]
        mean = {k: float(np.mean([s.get(k, 0) for s in sh])) for k in labs}
        res["targets"][t]["methods"][m] = {"top5": {k: mean[k] for k in top}, "rest": float(1 - sum(mean[k] for k in top)), "target_share": mean.get(t, 0.0)}
        L.append(f"| {m} | " + ";".join(f"{'**' + k + '**' if k == t else k}{' ★' if k in near else ''} {mean[k]:.3f}({freq[k]:.4f})" for k in top) + f" | {1 - sum(mean[k] for k in top):.3f} |")
    off = {m: [k for k in res["targets"][t]["methods"][m]["top5"] if k != t] for m in METH}
    nn = {m: sum(k in near for k in off[m]) for m in METH}; hi = {m: sum(freq[k] > 0.01 for k in off[m]) for m in METH}
    res["targets"][t]["off_in_nearest5"] = nn; res["targets"][t]["off_freq_gt_1pct"] = hi
    L += ["", "- among non-target labels in the top 5 labels of each method: number belonging to the 5 nearest types / number with training frequency > 1%:" + ";".join(f"{m} {nn[m]} / {hi[m]}({len(off[m])} in total)" for m in METH) + ".", ""]
L += ["> [!info]- reproducibility information", f"> - frozen classifier runs/r16/purity_lr/purity_lr_hlca.npz (219e9bce); PCA basis runs/r29_30/hlca_pca50_train.npz (067dc34d); generated arrays = runs/r24_26_3seed/manifest.json ({sha(W + '/runs/r24_26_3seed/manifest.json')}).",
      f"> - this document:`scripts/r39/r39_offtarget.py`({sha(__file__)})."]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); json.dump(res, open(sys.argv[2], "w"), indent=1); print("\n".join(L[5:]))
