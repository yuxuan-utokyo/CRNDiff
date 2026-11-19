#!/usr/bin/env python
r"""R34: marker co-expression + marker heatmap on HLCA test. Gene lists = R24's present markers (unchanged):
plasma cell = the non-IG markers present in the HVG panel (JCHAIN, MZB1, SDC1, TNFRSF17; JCHAIN is one of the four), ionocyte = the 5 present
(FOXI1, ASCL3, BSND, ATP6V1G3, CLNK). Co-expression: fraction of cells expressing (count > 0) >= 2 and >= 3 of the list. Rows: every
(method, seed) of the v3 manifest (generated cells only) and the real TEST target / non-target cells (one test read, logged).
Heatmap: R24's stored expressing fractions (runs/r24_26_3seed/test/r24_26_test.json, already read; no re-read), methods = mean over seeds.
    python r34_coexpr.py val|test OUT.json [FIG_DIR]      (FIG_DIR only with test)
"""
import hashlib, json, os, sys, time
import numpy as np
SPL, OUT = sys.argv[1], sys.argv[2]; FIG = sys.argv[3] if len(sys.argv) > 3 else None; assert SPL in ("val", "test") and not os.path.exists(OUT)
W = "./work"; sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
R24 = json.load(open(f"{W}/runs/r24_26_3seed/test/r24_26_test.json"))["R24"]
G = json.load(open(f"{W}/data/r16_hlca_meta/genes_cfgen_hvg2000.json")); assert G["genes"] == json.load(open(f"{W}/data/r16_hlca_test_view/meta.json"))["genes"]
SYM = list(G["feature_name"])   # as r24_26_3seed.py
GL = {"plasma cell": [g for g in R24["plasma cell"]["present"] if not g.startswith("IG")], "ionocyte": R24["ionocyte"]["present"]}
assert GL["plasma cell"] == ["JCHAIN", "MZB1", "SDC1", "TNFRSF17"] and len(GL["ionocyte"]) == 5
ED = f"{W}/data/r16_hlca_score" if SPL == "val" else f"{W}/data/r16_hlca_test_view"
X = np.load(f"{ED}/val.npy", mmap_mode="r"); Y = np.load(f"{ED}/val_celltype.npy", allow_pickle=True).astype(str)
man = json.load(open(f"{W}/runs/r24_26_3seed/manifest.json"))
def co(x, gi):
    k = (np.asarray(x[:, gi]) > 0).sum(1); return {"n": int(len(k)), "ge2": float((k >= 2).mean()), "ge3": float((k >= 3).mean()), "mean_n_expressed": float(k.mean())}
res = {"split": SPL, "genes": GL, "script_sha256": sha(__file__), "manifest_sha256": sha(f"{W}/runs/r24_26_3seed/manifest.json"), "real": {}, "runs": []}
for t, genes in GL.items():
    gi = [SYM.index(g) for g in genes]; it = np.flatnonzero(Y == t); io = np.flatnonzero(Y != t)
    res["real"][t] = {"target": co(X[it], gi)}
    ko = np.concatenate([(np.asarray(X[io[i:i + 20000]][:, gi]) > 0).sum(1) for i in range(0, len(io), 20000)])
    res["real"][t]["non_target"] = {"n": int(len(ko)), "ge2": float((ko >= 2).mean()), "ge3": float((ko >= 3).mean()), "mean_n_expressed": float(ko.mean())}
    for e in [e for e in man if e["target"] == t]:
        res["runs"].append({"method": e["method"], "seed": e["seed"], "target": t, **co(np.load(e["path"], mmap_mode="r"), gi)})
json.dump(res, open(OUT, "w"), indent=1)
print({t: {k: round(v["ge2"], 3) for k, v in r.items()} for t, r in res["real"].items()})
if SPL == "test":
    with open(f"{W}/runs/r16/TEST_READ_LOG.tsv", "a") as f:
        f.write(f"{time.strftime('%m-%d_%H:%M:%S')}\tHLCA R34 marker co-expression test (real target + non-target; r34_coexpr.py {sha(__file__)[:8]})\t{OUT.replace(W + '/', '')}\t{sha(OUT)[:8]}\n")
    # heatmap from R24's stored fractions
    sys.path.insert(0, f"{W}/scripts/figs/fig7_src"); import style as st
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    st.apply_rc(); cmap = LinearSegmentedColormap.from_list("w2v", ["#ffffff", "#f6c9a8", "#D55E00", "#6b2f00"])
    METH = ["CRNDiff", "CFGen", "CountsDiff", "scVI", "scANVI"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [len(R24["plasma cell"]["present"]), len(R24["ionocyte"]["present"])]})
    fig.subplots_adjust(left=.16, right=.90, top=.86, bottom=.22, wspace=.12); hm = {}
    for ax, t in zip(axes, ["plasma cell", "ionocyte"]):
        rows = R24[t]["rows"]; lab = ["Real target cells"] + METH + ["Real other cells"]
        M = [rows["real eval target"]["frac"]] + [np.mean([rows[k]["frac"] for k in rows if k.split("|")[0] == m], 0).tolist() for m in METH] + [rows["real eval non-target"]["frac"]]
        M = np.array(M); hm[t] = {"rows": lab, "genes": R24[t]["present"], "values": M.tolist(), "n_seeds_per_method": {m: sum(k.split("|")[0] == m for k in rows) for m in METH}}
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > .6 else "black")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(R24[t]["present"], rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(lab if t == "plasma cell" else [], fontsize=7); ax.set_title(t, fontsize=8, pad=3)
        ax.axhline(.5, color="black", lw=.6); ax.axhline(len(lab) - 1.5, color="black", lw=.6)
    cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.02); cb.set_label("Fraction of cells expressing (count > 0)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    for e in ("pdf", "png"): fig.savefig(f"{FIG}/hlca_marker_heatmap_test.{e}", dpi=400 if e == "png" else 300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    json.dump({"heatmap": hm, "source": "runs/r24_26_3seed/test/r24_26_test.json " + sha(f"{W}/runs/r24_26_3seed/test/r24_26_test.json")[:8], "script_sha256": sha(__file__),
               "outputs": {e: sha(f"{FIG}/hlca_marker_heatmap_test.{e}") for e in ("pdf", "png")}}, open(f"{FIG}/hlca_marker_heatmap_test.json", "w"), indent=1)
