#!/usr/bin/env python
r"""HLCA purity ruler check: CellTypist's official lung model instead of our 50-class LR (the author 19:58, scaled to plasma cell
and ionocyte, the author 21:25). VAL ONLY, no test. Describes; changes no selection.
Model: Human_Lung_Atlas.pkl v2 (~/.celltypist/data/models, sha256 6b073053..., downloaded 11-10 with Q's approval), celltypist 1.7.1.
Input: our 2,000 HVGs (Ensembl -> symbol via genes_cfgen_hvg2000.json 2dd94483), log1p(counts / total over the 2,000 genes x 1e4).
  main   "zero-filled": an AnnData over ALL 5,017 model features; the model features we do not have are 0;
  native "shared":      celltypist's own handling, the classifier restricted to the shared features (its default when genes are missing).
Labels: celltypist predicted_labels (best match, no majority voting) on every val cell. Label set per target, v7-style: the model labels whose
majority true class among val cells is the target; v8-style: v7 plus labels sharing the label stem (first word) -- printed so it can be read.
Ceiling = share of all real val cells of the target predicted into its set; purity = share of generated cells predicted into the set.
Rows: CRNDiff seed 1 tau grid (alpha 1, tau 0.2..1.0), CFGen (samp 20260987), CountsDiff K128, scANVI, scVI -- the val arrays of R16.
Rules A/B/C (definitions of scripts/r17/r17_rules.py) on the CRNDiff tau grid with the val W1, MMD², coverage, 1-NN of R16 (unchanged) and
purity from (i) the LR (check: must give the current picks 0.2 / 0.4) and (ii) CellTypist; picks compared with plasma 0.2, ionocyte 0.4.
    python r16_celltypist_check.py OUT.json
"""
import hashlib, json, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "8"
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import celltypist
from celltypist import models
W = "./work"; out = sys.argv[1]; R = f"{W}/runs/r16"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
MP = os.path.join(models.models_path, "Human_Lung_Atlas.pkl"); assert sha(MP).startswith("6b073053"), "model sha"
model = models.Model.load(MP); feats = list(model.features)
G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json")); assert G["genes"] == json.load(open(f"{W}/data/r16_hlca/genes.json"))["genes"]
sym = list(G["feature_name"]); shared = [s for s in sym if s in set(feats)]
TG = {"plasma cell": "plasmacell", "ionocyte": "ionocyte"}
def logcp(x):
    x = np.asarray(x, dtype=np.float64); lib = x.sum(1, keepdims=True); lib[lib == 0] = 1; return np.log1p(x / lib * 1e4).astype(np.float32)
def predict(x, mode):
    Z = logcp(x)
    if mode == "zero":
        col = {s: i for i, s in enumerate(feats)}; M = np.zeros((len(Z), len(feats)), dtype=np.float32)
        idx = [(j, col[s]) for j, s in enumerate(sym) if s in col]; M[:, [c for _, c in idx]] = Z[:, [j for j, _ in idx]]
        A = ad.AnnData(sp.csr_matrix(M), var=pd.DataFrame(index=feats))
    else:
        A = ad.AnnData(sp.csr_matrix(Z), var=pd.DataFrame(index=sym))
    A.obs_names = [f"c{i}" for i in range(len(Z))]
    return np.asarray(celltypist.annotate(A, model=model, majority_voting=False).predicted_labels["predicted_labels"].astype(str))
xv = np.load(f"{W}/data/r16_hlca_score/val.npy", mmap_mode="r"); yv = np.load(f"{W}/data/r16_hlca_score/val_celltype.npy", allow_pickle=True).astype(str)
res = {"model": {"file": MP, "sha256": sha(MP), "description": model.description, "n_features": len(feats), "n_labels": len(model.cell_types)},
       "genes": {"ours": 2000, "shared_with_model": len(shared), "model_features_zero_filled": len(feats) - len(shared)}, "modes": {}, "script_sha256": sha(__file__)}
# generated arrays (val manifests of R16)
gen = []
for t, k in TG.items():
    for e in json.load(open(f"{R}/tau_val/manifest_{k}.json")):
        c = json.load(open(f"{R}/tau_val/select_{k}.json"))["cells"].get(e["arm"])
        if c and c["alpha"] == 1.0: gen.append({"method": "CRNDiff", "arm": e["arm"], "tau": c["tau"], "target": t, "path": e["path"]})
    gen += [{"method": "CountsDiff", "arm": e["arm"], "target": t, "path": e["path"]} for e in json.load(open(f"{R}/cd_val/manifest_{k}.json")) if e["arm"] == "K128_g1p0"]
    for mf in ("base_cfgen/manifest.json", "base/manifest.json"):
        gen += [{"method": e["method"], "arm": e.get("arm"), "target": t, "path": e["path"]} for e in json.load(open(f"{R}/{mf}")) if e["target"] == t]
for mode in ("zero", "shared"):
    pv = predict(xv, mode); D = {"labelsets": {}, "ceiling": {}, "purity": [], "val_label_hist": {}}
    df = pd.DataFrame({"pred": pv, "true": yv}); maj = df.groupby("pred")["true"].agg(lambda s: s.value_counts().index[0])
    for t in TG:
        v7 = sorted(maj[maj == t].index); stems = {l.split()[0].lower() for l in v7}
        v8 = sorted(set(v7) | {l for l in model.cell_types if l.split()[0].lower() in stems})
        h = df[df.true == t]["pred"].value_counts(); D["val_label_hist"][t] = {k2: int(v) for k2, v in h.items()}
        D["labelsets"][t] = {"v7": v7, "v8": v8}
        for s_ in ("v7", "v8"): D["ceiling"][f"{t}/{s_}"] = float(np.isin(df[df.true == t]["pred"], D["labelsets"][t][s_]).mean()) if len(D["labelsets"][t][s_]) else None
    for g in gen:
        p = predict(np.load(g["path"], mmap_mode="r"), mode)
        D["purity"].append({**g, "n": len(p), **{s_: float(np.isin(p, D["labelsets"][g["target"]][s_]).mean()) if D["labelsets"][g["target"]][s_] else None for s_ in ("v7", "v8")},
                            "top_labels": dict(pd.Series(p).value_counts().head(4))})
    res["modes"][mode] = D; print(mode, "labelsets", D["labelsets"], "ceiling", D["ceiling"], flush=True)
# rules A/B/C (r17_rules.py definitions)
def rules(t, k, purity_by_tau, ref_pur):
    S = json.load(open(f"{R}/tau_val/select_{k}.json"))["cells"]; F = json.load(open(f"{R}/tau_val/scores_{k}_val.json"))["floors"][t]
    ref = {"W1": F["W1"]["W1"], "mmd2": F["e21"]["mmd2"], "purity": ref_pur, "coverage": F["two_sample_heldout_basis"]["coverage"]["mean"]}
    taus = sorted([c["tau"] for c in S.values() if c["alpha"] == 1.0]); cell = {c["tau"]: c for c in S.values() if c["alpha"] == 1.0}
    sc = {x: {"W1": ref["W1"] / cell[x]["W1"], "mmd2": ref["mmd2"] / cell[x]["mmd2"], "purity": purity_by_tau[x] / ref["purity"],
              "coverage": cell[x]["coverage"] / ref["coverage"], "nn1_acc": (1 - cell[x]["nn1_acc"]) / 0.5} for x in taus}
    cap = {x: {m: min(1.0, v) for m, v in sc[x].items()} for x in taus}
    A = next((x for x in taus if purity_by_tau[x] >= 0.9 * ref_pur), None)
    B = min(taus, key=lambda x: (np.sqrt(sum((1 - v) ** 2 for v in cap[x].values())), x))
    C = max(taus, key=lambda x: (np.exp(np.mean([np.log(max(v, 1e-12)) for v in cap[x].values()])), -x))
    return {"A": A, "B": B, "C": C, "purity_by_tau": purity_by_tau, "ref_purity": ref_pur}
PICK = {"plasma cell": 0.2, "ionocyte": 0.4}; res["rules"] = {}
for t, k in TG.items():
    S = json.load(open(f"{R}/tau_val/select_{k}.json"))
    lr = {c["tau"]: c["purity_logreg"] for c in S["cells"].values() if c["alpha"] == 1.0}
    res["rules"][f"{t}/LR"] = rules(t, k, lr, S["ceiling"])
    for mode in ("zero", "shared"):
        for s_ in ("v7", "v8"):
            ct = {r["tau"]: r[s_] for r in res["modes"][mode]["purity"] if r["method"] == "CRNDiff" and r["target"] == t}
            ceil = res["modes"][mode]["ceiling"][f"{t}/{s_}"]
            if ceil: res["rules"][f"{t}/CT-{mode}-{s_}"] = rules(t, k, ct, ceil)
    for key, v in res["rules"].items():
        if key.startswith(t): v["matches_current_pick"] = {r: (v[r] == PICK[t]) for r in "ABC"}
for k2, v in res["rules"].items(): print(k2, {r: v[r] for r in "ABC"}, flush=True)
json.dump(res, open(out, "w"), indent=1, default=str)
