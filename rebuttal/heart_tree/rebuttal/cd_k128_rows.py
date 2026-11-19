# -*- coding: utf-8 -*-
"""CountsDiff HVG2K K128 main-table rows (tab:hvg_single_type_main), 3 train seeds: purity (v8, abs, as the table's purity column),
W1, MMD^2 (mmd2_rbf_biased), PCC from out/summary/rebuttal_CD_K128_s<seed>_{scored,v8}.json (score_by_arm 042130be + cd_rescore_v8
03215983). mean +- sd (ddof 1). Reads json only.  python -m rebuttal.cd_k128_rows
"""
import json, statistics as st
from pathlib import Path
S = Path(__file__).resolve().parents[1] / "out" / "summary"; SEEDS = ["20260625", "20260626", "20260627"]; T = ["Endothelial", "Myeloid", "Neuronal"]
def rows(blob):
    if isinstance(blob, dict):
        if "request" in blob: yield blob
        for v in blob.values(): yield from rows(v)
    elif isinstance(blob, list):
        for v in blob: yield from rows(v)
per = {t: {k: [] for k in ("purity_v8", "W1", "mmd2", "pcc")} for t in T}; prov = {}
for s in SEEDS:
    sc = json.loads((S / f"rebuttal_CD_K128_s{s}_scored.json").read_text(encoding="utf-8"))
    v8 = json.loads((S / f"rebuttal_CD_K128_s{s}_v8.json").read_text(encoding="utf-8"))
    for t in T:
        r = [x for x in rows(sc) if x.get("request") == t and "W1" in x and "mmd2_rbf_biased" in x and "train" + s in str(x.get("stem", ""))]
        p = [x for k, x in v8["rows"].items() if x.get("request") == t and "purity_v8" in x and "train" + s in k]   # v8 rows are keyed by stem
        assert len(r) == 1 and len(p) == 1, (s, t, len(r), len(p))
        per[t]["W1"].append(r[0]["W1"]); per[t]["mmd2"].append(r[0]["mmd2_rbf_biased"]); per[t]["pcc"].append(r[0]["pcc"])
        per[t]["purity_v8"].append(p[0]["purity_v8"]); prov[f"{t}|{s}"] = r[0]["stem"]
out = {t: {k: {"mean": st.mean(v), "sd": st.stdev(v), "per_seed": v} for k, v in d.items()} for t, d in per.items()}
(S / "rebuttal_CD_K128_rows.json").write_text(json.dumps({"seeds": SEEDS, "rows": out, "stems": prov}, indent=1), encoding="utf-8")
for t in T:
    o = out[t]; print(f"{t:12s} purity_v8 {o['purity_v8']['mean']:.4f}+-{o['purity_v8']['sd']:.4f}  W1 {o['W1']['mean']:.4f}+-{o['W1']['sd']:.4f}  "
                      f"MMD2 {o['mmd2']['mean']:.5f}+-{o['mmd2']['sd']:.5f}  PCC {o['pcc']['mean']:.4f}+-{o['pcc']['sd']:.4f}  "
                      f"per-seed W1 {[round(x, 5) for x in o['W1']['per_seed']]} MMD2 {[round(x, 6) for x in o['mmd2']['per_seed']]}")
