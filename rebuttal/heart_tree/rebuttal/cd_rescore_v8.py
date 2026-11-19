# -*- coding: utf-8 -*-
"""v8 lineage purity for a scored rebuttal json, the tools.t215_rescore_v8 way: the frozen
hvg.metrics.purity_from_hist on each row's stored label_dist_whole with the master_purity_v8_lineage label set.
No CellTypist rerun. Gate G1: the v7 recompute must equal the row's purity_whole_set exactly; G2: v8 >= v7.
    python -m rebuttal.cd_rescore_v8 out/summary/rebuttal_CD_select_scored.json
"""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hvg import config as C
from hvg.metrics import purity_from_hist
src = ROOT / sys.argv[1]; j = json.loads(src.read_text(encoding="utf-8"))
lab = {v: {t: sorted(x["fixed_labels"]) for t, x in json.loads((C.ASSETS / "tables" / f).read_text(encoding="utf-8"))["types"].items()}
       for v, f in (("v7", "master_purity_v7.json"), ("v8", "master_purity_v8_lineage.json"))}
for t in ("Endothelial", "Myeloid", "Neuronal"):
    print(f"[labels] {t}: v7 {lab['v7'][t]} | v8 {lab['v8'][t]} | same {lab['v7'][t] == lab['v8'][t]}")
rows = {}
for r in j["rows"]:
    t = r["request"]; h = r["label_dist_whole"]
    p7, _ = purity_from_hist(h, set(lab["v7"][t])); p8, _ = purity_from_hist(h, set(lab["v8"][t]))
    assert p7 == r["purity_whole_set"], (r["arm"], p7, r["purity_whole_set"]); assert p8 >= p7
    # 11-09 17:4x: rows keyed by the row's stem (unique per run), not by arm, which silently kept only the last
    # seed when one arm had several. Scoring unchanged; "arm" kept in the value.
    k = r.get("stem") or r["arm"]
    assert k not in rows, ("duplicate row key", k)
    rows[k] = {"arm": r["arm"], "request": t, "purity_v7": p7, "purity_v8": p8, "n": sum(h.values())}
    print(f"{r['arm']:<24} v7 {p7:.4f} v8 {p8:.4f}")
out = ROOT / "out/summary" / (src.stem.replace("_scored", "") + "_v8.json")
out.write_text(json.dumps({"source": str(src), "method": "tools.t215_rescore_v8 way: purity_from_hist on label_dist_whole", "labels": lab, "gates": "G1 v7 recompute == purity_whole_set (all rows), G2 v8 >= v7", "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print("[out]", out)
