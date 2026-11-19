# -*- coding: utf-8 -*-
"""celltypist_eval.py — uses **CellTypist** to score the purity of conditional generation, to align with published numbers.

why change the classifier
--------------
scDiffusion reports **CellTypist classification accuracy: 0.93 averaged over all types** (0.98 for real cells).
our local joblib classifier gives a different scale; the two numbers cannot be put side by side.
replacing it with the same public pre-trained classifier makes our numbers **directly comparable** with the published 0.93,
no need to rerun other models.

★ all three rows must be reported together (this is the key point of the whole script)
------------------------------------------
our data have only **2000 HVGs**, while the CellTypist model was trained on the full gene set.
missing genes lower accuracy across the board: our numbers are depressed by this handicap, others are not.

  so **every time** run"real val cells"that row**: it bears exactly the same handicap,
  so the ratio ours / real cells cancels the handicap; only that can be compared with
  scDiffusion quantity compared in 0.93 / 0.98. **Reporting only absolute values is misleading.**

three rows: no-op (unconditional generation) / conditional generation / real val cells.

definition
----
* CellTypist requires input **normalize_total(1e4) + log1p** (as in its training),
  this differs from our local joblib classifier, which takes raw counts; do not mix the two.
* `majority_voting=False`: per-cell verdicts are required. Majority voting smooths over neighbourhoods and inflates purity.

★ subtype labels must be **aggregated back to coarse classes** (tripped over on 2026-08-16)
------------------------------------------------
`Healthy_Adult_Heart.pkl` it predicts subtypes: endothelial is split into EC2_cap / EC3_cap / EC5_art /
EC6_ven ... the first version of the script took"the most frequent label among real cells"as the target, picked EC5_art,
so even real cells score only 0.254: **that is not purity, it is the share of one subtype.**

the aggregation rule** is learned from real cells, not guessed from prefixes: for each CellTypist label L, look in val at
assign label L to the coarse class that is most common among real cells with label L. The mapping is thus data-driven and auditable,
and the script prints the mapping.

★ weakly supported labels must be reported separately (tripped over a second time on 2026-08-16)
------------------------------------------------
majority voting is noise on labels with **very few samples**:`Adip4` assigned to only a few dozen real cells; most
is assigned to Endothelial, so it is counted in the target set and gives every row a free share.
so this version:
* print the **number of supporting cells** and the **majority share** of each target label;
* support < `--min-support` or majority share < `--min-major` labels are marked **weak**;
* purity **reports two numbers**:`purity`(with weak labels) and `purity_strict`(weak labels removed).
  if the two numbers differ a lot, the conclusion rests on a noisy mapping and cannot be used.
* the **full label distribution** of every row is stored in json, so later audits need no rerun.
* model names available `--list-models` lookup; heart-related ones first (the dataset is the Heart Cell Atlas).

how to run
----
    python celltypist_eval.py --list-models
    python celltypist_eval.py --type Endothelial --arms a.npy,b.npy --model <name>

red line: read-only, no existing output changed; CPU only.
"""
import argparse, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XR = Path(__file__).resolve().parents[3] / "assets" / "legacy_root"


def to_adata(X, genes):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.asarray(X, dtype=np.float32))
    a.var_names = list(genes)
    a.var_names_make_unique()
    # CellTypist input definition: normalize each cell to 1e4 then log1p (as in its training)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-root", default=str(XR))
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--model", default="Healthy_Adult_Heart.pkl")
    ap.add_argument("--type", default=None, help="target cell type (spelling in CellTypist may differ)")
    ap.add_argument("--arms", default=None, help="conditional generation outputs, comma-separated")
    ap.add_argument("--pilot", default=None)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n-map", type=int, default=20000,
                    help="how many real val cells to use for learning the mapping subtype label -> coarse class")
    ap.add_argument("--min-support", type=int, default=50,
                    help="a subtype label must be assigned to at least this many real cells for the mapping to be trusted")
    ap.add_argument("--min-major", type=float, default=0.5,
                    help="the majority coarse class must take at least this share for the mapping to be trusted")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("-o", "--out", default="celltypist")
    # ★ 2026-08-17: suffix added to the json file name. Previously every run wrote celltypist_<type>.json,
    #   the single arm run later **overwrites** the earlier one (the γ=0.85 run overwrote 0.6910 of γ=0.4, and the books did not balance).
    #   only changes file names, never touches any numerical path.
    ap.add_argument("--tag", default="",
                    help="appended to the json file name: celltypist_<type><tag>.json, so repeated runs do not overwrite each other")
    a = ap.parse_args()

    import celltypist
    from celltypist import models

    if a.list_models:
        models.download_models(force_update=False)
        print(models.models_description().to_string())
        return

    R = Path(a.x_root); D = R / "_shared" / "data" / "all"
    OUT = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    genes = json.loads((D / "meta.json").read_text(encoding="utf-8"))["genes"]
    print(f"[genes] {len(genes)} HVGs, first 5:{genes[:5]}")

    models.download_models(model=[a.model], force_update=False)
    mdl = models.Model.load(model=a.model)
    print(f"[model] {a.model}  can predict {len(mdl.cell_types)} classes")
    ov = len(set(genes) & set(mdl.features))
    print(f"[genes] intersection with the model features {ov}/{len(genes)}"
          f"(the model has {len(mdl.features)} features in total)"
          f"  ← only 2000 HVGs; missing genes depress every row, relying on'real cells'cancels that row")

    dists = {}

    def annotate(X, tag):
        res = celltypist.annotate(to_adata(X, genes), model=a.model,
                                  majority_voting=False)
        lab = res.predicted_labels["predicted_labels"].to_numpy().astype(str)
        u, c = np.unique(lab, return_counts=True)
        top = sorted(zip(c / c.sum(), u), reverse=True)[:4]
        print(f"  [{tag}] first few classes:" + "  ".join(f"{n} {p:.3f}" for p, n in top))
        dists[tag] = {str(k): int(v) for k, v in zip(u, c)}   # store the full distribution for review
        return lab

    jobs = []
    pilot = Path(a.pilot) if a.pilot else (
        R / "Baseline" / "OURS" / "results" /
        "linspace_T_O_K32_spilot_run1c_s20260625_samp20260902_batched1_chunk256_F2_K32_base_noa4.npy")
    jobs.append(("no-op (unconditional)", pilot))
    for f in (a.arms or "").split(","):
        if f.strip():
            jobs.append((Path(f).name[:60], Path(f.strip())))

    Xv = np.load(D / "val.npy", mmap_mode="r")
    yv = np.load(D / "val_celltype.npy", allow_pickle=True)

    out = {"model": a.model, "n_genes_overlap": int(ov), "target": a.type, "rows": []}
    labels = {}
    for tag, f in jobs:
        X = np.load(f, mmap_mode="r")
        i = np.sort(rng.choice(X.shape[0], size=min(a.n, X.shape[0]), replace=False))
        labels[tag] = annotate(np.asarray(X[i], dtype=np.float64), tag)

    if a.type:
        # ---- first run once on **all real val cells** to learn the mapping subtype label -> coarse class ----
        m = np.sort(rng.choice(Xv.shape[0], size=min(a.n_map, Xv.shape[0]),
                               replace=False))
        lab_map = annotate(np.asarray(Xv[m], dtype=np.float64), "real val (for building the mapping)")
        coarse = yv[m].astype(str)
        fine2coarse, support = {}, {}
        for L in np.unique(lab_map):
            sel = lab_map == L
            u, c = np.unique(coarse[sel], return_counts=True)
            k = int(np.argmax(c))
            fine2coarse[str(L)] = str(u[k])                  # majority assignment
            support[str(L)] = {"n": int(sel.sum()),
                               "major_frac": float(c[k] / c.sum())}
        tgt_labels = sorted(L for L, cc in fine2coarse.items() if cc == a.type)
        if not tgt_labels:
            raise SystemExit(f"no CellTypist label has {a.type} as its majority assignment; stop and inspect the mapping")
        strong = [L for L in tgt_labels
                  if support[L]["n"] >= a.min_support
                  and support[L]["major_frac"] >= a.min_major]
        weak = [L for L in tgt_labels if L not in strong]
        print(f"\n[mapping] '{a.type}' corresponds to {len(tgt_labels)} CellTypist labels")
        print(f"  {'label':<20}{'supporting cells':>10}{'majority share':>10}   verdict")
        for L in tgt_labels:
            s = support[L]
            print(f"  {L:<20}{s['n']:>10d}{s['major_frac']:>10.3f}   "
                  f"{'strong' if L in strong else 'weak (will be removed by strict)'}")
        if not strong:
            raise SystemExit("no strongly supported target label; the mapping is not trustworthy, stop and inspect")
        out["fine_to_coarse"] = fine2coarse
        out["label_support"] = support
        out["target_labels"] = tgt_labels
        out["target_labels_strong"] = strong
        out["target_labels_weak"] = weak

        ii = np.flatnonzero(yv == a.type)
        s = np.sort(rng.choice(ii, size=min(a.n, len(ii)), replace=False))
        labels["real val cells"] = annotate(np.asarray(Xv[s], dtype=np.float64),
                                          "real val cells")
        # of rare types"real cells"that row may have only a few dozen cells (Mesothelial val has only 71),
        # the standard error of purity is above 0.03. **n must be reported together with the number**, otherwise the ceiling looks like a hard number.
        print(f"\n{'population':<44}{'n':>7}{'purity (including weak)':>12}{'±s.e.':>9}{'purity(strict)':>14}")
        for tag in labels:
            p = float(np.isin(labels[tag], tgt_labels).mean())
            ps = float(np.isin(labels[tag], strong).mean())
            n = int(len(labels[tag]))
            se = float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))
            out["rows"].append({"arm": tag, "n": n, "purity": p,
                                "purity_se": se, "purity_strict": ps})
            print(f"{tag:<44}{n:>7d}{p:>12.4f}{se:>9.4f}{ps:>14.4f}")
        base = out["rows"][0]; ceil = out["rows"][-1]
        d = max(abs(r["purity"] - r["purity_strict"]) for r in out["rows"])
        print(f"\n[weak-label sensitivity] max difference between the two columns {d:.4f}"
              + ("   ← the conclusion does not depend on the weak mapping" if d < 0.01 else
                 "   ← differs noticeably; the weak mapping contributes to the share; use the strict column"))
        print(f"\nreference: scDiffusion reports a CellTypist mean of 0.93; its real cells 0.98"
              f"({0.93/0.98:.0%} of the ceiling)")
        for r in out["rows"][1:-1]:
            print(f"  {r['arm'][:44]:<44} share of the cap "
                  f"{r['purity']/max(ceil['purity'],1e-9):.0%}"
                  f" / strict {r['purity_strict']/max(ceil['purity_strict'],1e-9):.0%}"
                  f"(no-op {base['purity']/max(ceil['purity'],1e-9):.0%})")
    out["label_dists"] = dists
    (OUT / f"celltypist_{a.type or 'all'}{a.tag}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
