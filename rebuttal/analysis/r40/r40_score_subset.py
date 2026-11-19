#!/usr/bin/env python
"""R40: run r13_score.py unchanged with ONLY the --subset step of score_with_npz.py (ed519051), copied verbatim, and no purity (the frozen
purity classifier needs all 2,000 genes, so it cannot run on the no-IG view; purity is not an R40 metric). This keeps the Table 3 target list,
row layout and subset files, which the frozen W1 code depends on. Gate: on the full-gene view it must reproduce Table 3 bit for bit.
Usage: r40_score_subset.py [--subset IDX.npz ...] <r13_score.py arguments without --purity>
"""
import hashlib, json, os, sys
from pathlib import Path
W = os.path.expanduser("~/research/crndiff-rebuttal"); sys.path.insert(0, f"{W}/scripts/r13_score")
import numpy as np
import r13_score as R
assert "--purity" not in sys.argv
SUBSETS = []
while "--subset" in sys.argv:
    i = sys.argv.index("--subset"); SUBSETS.append(sys.argv[i + 1]); del sys.argv[i:i + 2]
SUB_REC = [{"file": f, "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()} for f in SUBSETS]
_init = R.Scorer.__init__
def init(self, a):
    _init(self, a)
    for f, r in zip(SUBSETS, SUB_REC):   # verbatim from score_with_npz.py (after the ceilings there; no ceilings here)
        s = np.load(f, allow_pickle=False); t = str(s["target"]); idx = s["idx"]
        assert t in a.targets and len(np.unique(idx)) == len(idx) and (self.yv[idx] == t).all(), f
        out_mask = (self.yv == t); out_mask[idx] = False
        self.yv = self.yv.copy(); self.yv[out_mask] = "__outside_subset__"
        self.W.N_MATCH[t] = int(len(idx)); r.update({"target": t, "n": int(len(idx)), "n_class": int(s["n_class"]), "split": str(s["split"])})
        print(f"[subset] {t}: {len(idx)} of {int(s['n_class'])} cells ({f}, sha {r['sha256'][:16]})", flush=True)
R.Scorer.__init__ = init
rc = R.main()
out = Path(sys.argv[sys.argv.index("--out") + 1]); res = json.loads(out.read_text(encoding="utf-8"))
if SUBSETS: res["subsets"] = SUB_REC
res["wrapper"] = {"file": "scripts/r40/r40_score_subset.py", "sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
out.write_text(json.dumps(res, indent=1), encoding="utf-8"); sys.exit(rc)
