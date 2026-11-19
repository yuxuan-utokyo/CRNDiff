#!/usr/bin/env python
"""Manifest for r24_26_3seed.py: the 3 seeds per method of the HLCA test table v3 (R16_test_ruleB_v3_s25s27s28.md), read from the same
scoring manifests (no path typed by hand). CRNDiff = generators s20260625 / 27 / 28 at the
rule-B tau (plasma cell 0.2, ionocyte 0.4); CFGen = 1 public checkpoint x samp 20260987/88/89; CountsDiff K128 = train 20260625/26/27;
scVI = train 20261050/53/54; scANVI = train 20261060/63/64. Seed label = generator / train seed (CFGen: sampling seed).
    python r24_26_manifest_3seed.py OUT.json
"""
import hashlib, json, sys
W = "./work"; R = f"{W}/runs/r16"; out = sys.argv[1]
TAU = {"plasma cell": "tau0p2", "ionocyte": "tau0p4"}; CL = list(TAU)
ld = lambda f: json.load(open(f"{R}/{f}"))
A = ld("ruleA_test/manifest_test.json") + ld("cd_k128_test/manifest_test.json")
m = []
def add(method, seed, t, path): m.append({"method": method, "seed": str(seed), "target": t, "path": path})
for t in CL:
    one = lambda meth, arm=None: [e for e in A if e["method"] == meth and e["target"] == t and (arm is None or e["arm"] == arm)]
    e = one("CRNDiff", TAU[t]); assert len(e) == 1; add("CRNDiff", "gen20260625", t, e[0]["path"])
    for g in ("27", "28"):
        e = [x for x in ld(f"crn_s{g}/manifest.json") if x["target"] == t]; assert len(e) == 1 and e[0]["arm"] == TAU[t], e; add("CRNDiff", f"gen202606{g}", t, e[0]["path"])
    e = one("CFGen"); assert len(e) == 1; add("CFGen", "samp20260987", t, e[0]["path"])
    for s in ("20260988", "20260989"):
        e = [x for x in ld(f"base_cfgen_s{s}/manifest.json") if x["target"] == t]; assert len(e) == 1; add("CFGen", f"samp{s}", t, e[0]["path"])
    e = {x["path"] for x in one("CountsDiff", "K128_g1p0")}; assert len(e) == 1; add("CountsDiff", "train20260625", t, e.pop())
    for d, s in (("cd_s26_k128", "20260626"), ("cd_s27_k128", "20260627")):
        e = [x for x in ld(f"{d}/manifest.json") if x["target"] == t]; assert len(e) == 1; add("CountsDiff", f"train{s}", t, e[0]["path"])
    for meth, key, seeds in (("scVI", "scvi", ("20261053", "20261054")), ("scANVI", "scanvi", ("20261063", "20261064"))):
        e = one(meth); assert len(e) == 1; add(meth, f"train{e[0]['seed']}", t, e[0]["path"])
        for s in seeds:
            e = [x for x in ld(f"base_{key}_s{s}/manifest.json") if x["target"] == t]; assert len(e) == 1; add(meth, f"train{s}", t, e[0]["path"])
assert len(m) == 30 and len({x["path"] for x in m}) == 30, len(m)
assert not any("20260630" in x["path"] and x["method"] == "CRNDiff" for x in m)
json.dump(m, open(out, "w"), indent=1)
print("manifest", len(m), "entries; sha", hashlib.sha256(open(out, "rb").read()).hexdigest()[:8])
for x in m: print(" ", x["method"], x["seed"], x["target"], x["path"].replace(W + "/", "")[-70:])
