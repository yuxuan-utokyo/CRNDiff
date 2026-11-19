# -*- coding: utf-8 -*-
"""Board 11-08 21:3x: scVI / scANVI frozen-ruler val numbers for their Table 1 runs (w2_runs.TABLE1), from EXISTING records only
(no rescoring), next to rebuttal_R12_side.json:
  v7, W1, W1/floor, sliced W1, MMD^2 (rbf, biased), PCC, SCC, floors = out/summary/214g_<arm>_s<train>_samp<samp>_scored.json (frozen scorer)
  v8 = out/summary/215_purity_v8_rescore.json (label histogram of the same scored file re-read under v8; same file + stem)
    python -m rebuttal.r12_side_baselines
"""
import hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from rebuttal import w2_runs as W
S = ROOT / "out/summary"
X = Path(r"work\exchange\runs\table1_ours")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v8 = {(r["file"], r["stem"]): r for r in json.loads((S / "215_purity_v8_rescore.json").read_text(encoding="utf-8"))["rows"]}
out = {"split": "val (frozen rulers; test not read)", "rescored": False,
       "sources": {"v7, W1, MMD2, PCC, floors": "out/summary/214g_<arm>_s<train>_samp<samp>_scored.json",
                   "v8": "out/summary/215_purity_v8_rescore.json (same file + stem)"}, "targets": {}}
for t in ("Endothelial", "Myeloid", "Neuronal"):
    out["targets"][t] = {}
    for arm, low in (("scVI", "scvi"), ("scANVI", "scanvi")):
        rows = []
        for tr, s in W.TABLE1[arm]:
            fn = f"214g_{low}_s{tr}_samp{s}_scored.json"
            r = [q for q in json.loads((S / fn).read_text(encoding="utf-8"))["rows"] if q["request"] == t]
            assert len(r) == 1, (fn, t, len(r)); r = r[0]; e = v8[(fn, r["stem"])]
            assert abs(e["purity_v7"] - r["purity_whole_set"]) < 1e-12, (fn, t)
            rows.append({"train": tr, "samp": s, "stem": r["stem"], "source_file": fn, "n_delivered": r["n_delivered"],
                         "purity_v8": e["purity_v8"], "purity_v7": r["purity_whole_set"],
                         **{k: r.get(k) for k in ("W1", "W1_over_floor", "sliced_W1", "mmd2_rbf_biased", "pcc", "scc",
                                                  "floor_W1", "floor_sliced_W1", "floor_mmd2", "floor_pcc", "floor_scc", "W1_matched_n", "e21_matched_n")}})
        out["targets"][t][arm] = rows
p = ROOT / "rebuttal/out/summary/rebuttal_R12_side_baselines.json"; p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
shutil.copyfile(p, X / p.name)
with open(X / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as f: f.write(f"{sha(X / p.name)}  {p.name}\n")
for t, d in out["targets"].items():
    for a, rr in d.items():
        print(t, a, [(r["purity_v8"], r["purity_v7"], round(r["W1"], 4), round(r["W1_over_floor"], 2), round(r["mmd2_rbf_biased"], 5), round(r["pcc"], 4)) for r in rr])
print("[out]", p, sha(p))
