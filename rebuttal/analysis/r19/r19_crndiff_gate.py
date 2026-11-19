#!/usr/bin/env python
"""R19 CRNDiff val gate: the Mac r1r2_metrics.py output on VAL
(original val bases) for the 9 CRNDiff runs of one target must reproduce Windows' windows_full_r123 r1_numbers.json / r2_numbers.json OURS
rows (matched by train, samp). Pass: every numeric field within 1e-12 (absolute). Prints counts and writes a json verdict.
    python r19_crndiff_gate.py MAC_VAL.json T OUT.json
"""
import json, sys
import numpy as np
W = "./work"; mac_f, T, out = sys.argv[1:4]
M = json.load(open(mac_f)); assert M["split"] == "val", M["split"]
w1 = [r for r in json.load(open(f"{W}/from_windows/windows_full_r123/r1_numbers.json"))["rows"] if r["arm"] == "OURS"]
w2 = [r for r in json.load(open(f"{W}/from_windows/windows_full_r123/r2_numbers.json"))[T]["runs"] if r["arm"] == "OURS"]
TOL = 1e-12; n_ok = n_all = 0; bad = []
def cmp(tag, a, b):
    global n_ok, n_all
    for k, v in b.items():
        if k in ("arm", "train", "samp", "space", "n") or not isinstance(v, (int, float, list)): continue
        x = np.asarray(a[k], float); y = np.asarray(v, float); n_all += 1
        if x.shape == y.shape and np.all(np.abs(x - y) <= TOL): n_ok += 1
        else: bad.append((tag, k, float(np.max(np.abs(x - y))) if x.shape == y.shape else "shape"))
runs = [r for r in M["runs"] if r["target"] == T]; assert len(runs) == 9, len(runs)
for r in runs:
    key = (int(r["train"]), int(r["seed"]))
    b2 = [x for x in w2 if (x["train"], x["samp"]) == key]; assert len(b2) == 1, ("R2 rows", key, len(b2))
    cmp(f"R2 {key}", r["r2"], b2[0])
    if T == "Neuronal":
        for sp, a in r["r1"].items():
            b1 = [x for x in w1 if (x["train"], x["samp"], x["space"]) == (*key, sp)]; assert len(b1) == 1, ("R1 rows", key, sp, len(b1))
            cmp(f"R1 {key} {sp}", a, b1[0])
res = {"target": T, "fields_compared": n_all, "fields_within_1e-12": n_ok, "bad": bad, "PASS": not bad and n_all > 0}
json.dump(res, open(out, "w"), indent=1); print(T, f"{n_ok}/{n_all} fields within {TOL}", "PASS" if res["PASS"] else f"FAIL {bad[:5]}")
