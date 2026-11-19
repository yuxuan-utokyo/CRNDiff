#!/usr/bin/env python
"""R19 CRNDiff runs of one HVG2K target (the 9 deployed runs = generators 20260625/27/30 x samp 20260987/88/89, as in Windows'
windows_full_r123 r1/r2/r3_numbers OURS rows; Table 1 = w2_runs.TABLE1 OURS (25/89, 27/87, 30/89)). Every array is checked against its
json: generator seed in the checkpoint name, samp seed, request = target, tau = the target's deployed tau (r6_numbers.json).
Local sources (sha-verified copies): from_windows/r19_arrays/crndiff_table1_ours/<T>_a1 (gen 25), from_windows/r19_arrays/r4_deployed_compare
(Neuronal 27/87, 30/89), from_windows/r19_crndiff/table1_ours_g<gen>/<T>_a1 (rest, delivered 11-11).
    python r19_crndiff_manifest.py T OUT_ALL.json OUT_TABLE1.json      exits 3 and lists what is missing if any of the 9 is absent
"""
import glob, json, re, sys
W = "./work"; T, out_all, out_t1 = sys.argv[1:4]
DEP = {"Endothelial": 0.4, "Myeloid": 0.2, "Neuronal": 0.2}
r6 = json.load(open(f"{W}/from_windows/r17_hvg2k_tau/r6_numbers.json")); assert abs(r6[T]["deployed"] - DEP[T]) < 1e-12, (T, r6[T]["deployed"])
TABLE1 = [(20260625, 20260989), (20260627, 20260987), (20260630, 20260989)]   # main_table_cov_1nn.py / w2_runs.TABLE1["OURS"], verbatim
found = {}
for f in sorted(glob.glob(f"{W}/from_windows/r19_arrays/crndiff_table1_ours/{T}_a1/*.npy") + glob.glob(f"{W}/from_windows/r19_arrays/r4_deployed_compare/*/{T}_*.npy")
                + glob.glob(f"{W}/from_windows/r19_crndiff/table1_ours_g*/{T}_a1/*.npy")):
    j = json.load(open(f[:-4] + ".json")); g = int(re.search(r"_s(2026\d{4})\.pt", str(j["checkpoint"]).replace("\\", "/")).group(1))
    assert j["seed"] == int(f.rsplit("_seed", 1)[1][:8]) and abs(j["tau"] - DEP[T]) < 1e-12 and j.get("request", T) == T, (f, j["seed"], j["tau"], j.get("request"))
    k = (g, j["seed"])
    if k in found: assert open(found[k], "rb").read() == open(f, "rb").read(), ("two different files for", k); continue
    found[k] = f
want = [(g, s) for g in (20260625, 20260627, 20260630) for s in (20260987, 20260988, 20260989)]
miss = [k for k in want if k not in found]
if miss: print("MISSING", T, miss); sys.exit(3)
man = [{"method": "OURS", "arm": "OURS", "train": g, "seed": s, "target": T, "path": found[(g, s)]} for g, s in want]
json.dump(man, open(out_all, "w"), indent=1)
json.dump([e for e in man if (e["train"], e["seed"]) in TABLE1], open(out_t1, "w"), indent=1)
print(T, "9 runs ok; Table 1", TABLE1)
