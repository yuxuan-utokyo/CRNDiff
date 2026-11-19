#!/usr/bin/env python
r"""Lung pericyte (third HLCA target): manifest of the delivered pericyte arrays, 3 seeds per method, each verified against its transit
dir's SHA256SUMS (hash lines only) and copied to from_windows/r16_peri/. CRNDiff = generators 20260625 / 27 / 28 at tau 0.2 (rule B on val,
11-11 05:0x); each run json's portable_classifier_source_sha256 must equal the Mac fit record of that generator's pericyte disc.
CFGen = public checkpoint x samp 20260987/88/89; CountsDiff K128 = train 20260625/26/27; scVI = train 20261050/53/54; scANVI = 20261060/63/64.
    python r16_peri_score.py OUT_MANIFEST.json [--require-all]     (exit 3 if --require-all and not all 15 are present)
"""
import glob, hashlib, json, os, re, shutil, sys
W = "./work"; out = sys.argv[1]; need_all = "--require-all" in sys.argv
X = os.path.expanduser("./work/exchange")
L = f"{W}/from_windows/r16_peri"; T = "lung pericyte"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
def listed(d, name):
    s = f"{d}/SHA256SUMS"
    if not os.path.exists(s): return None
    for line in open(s):
        m = re.match(r"^([0-9a-f]{64}) [ *](.+)$", line.rstrip("\n"))
        if m and m.group(2) == name: return m.group(1)
    return None
def take(d, name, sub):
    h = listed(d, name); f = f"{d}/{name}"
    if not h or not os.path.exists(f) or sha(f) != h: return None
    os.makedirs(f"{L}/{sub}", exist_ok=True); g = f"{L}/{sub}/{name}"
    if not os.path.exists(g) or sha(g) != h: shutil.copy2(f, g)
    assert sha(g) == h; return g, h
man = []; missing = []
# CRNDiff
for g in ("20260625", "20260627", "20260628"):
    jl = json.load(open(f"{W}/runs/r16_disc/s{g}/lungpericyte/fit_record.json"))["joblib_sha256"] if os.path.exists(f"{W}/runs/r16_disc/s{g}/lungpericyte/fit_record.json") else None
    if g == "20260625":
        loc = glob.glob(f"{W}/from_windows/rebuttal_R16/lungpericyte_tau0p2/*.npy")   # grid run, verified when the tau grid was scored
        got = (loc[0], sha(loc[0])) if len(loc) == 1 else None
    else:
        d = f"{X}/runs/rebuttal_R16/lungpericyte_tau0p2_gen{g}"; ns = [os.path.basename(p) for p in glob.glob(f"{d}/*.npy")]
        got = take(d, ns[0], f"CRNDiff_gen{g}") if len(ns) == 1 else None
        if got: take(d, ns[0][:-4] + ".json", f"CRNDiff_gen{g}")
    if not got: missing.append(f"CRNDiff gen{g}"); continue
    j = json.load(open(got[0][:-4] + ".json"))
    assert abs(j["tau"] - 0.2) < 1e-9 and j["alpha"] == 1.0 and g in str(j["checkpoint"]), (g, j["tau"], j["checkpoint"])
    assert jl and j["portable_classifier_source_sha256"] == jl, (g, "disc differs from the Mac fit", j["portable_classifier_source_sha256"], jl)
    man.append({"method": "CRNDiff", "arm": "tau0p2", "seed": int(j["seed"]), "train": int(g), "target": T, "path": got[0], "sha256": got[1]})
# CFGen
for s in ("20260987", "20260988", "20260989"):
    got = take(f"{X}/r16_baselines/cfgen_lungpericyte", f"cfgen_cond_lungpericyte_n2500_seed{s}.npy", "CFGen")
    if got: man.append({"method": "CFGen", "arm": "cond", "seed": int(s), "train": "public", "target": T, "path": got[0], "sha256": got[1]})
    else: missing.append(f"CFGen samp{s}")
# CountsDiff
for tr in ("20260625", "20260626", "20260627"):
    d = f"{X}/r16_cd/hlca_s{tr}"; got = take(d, "lungpericyte_K128_g1p0.npy", f"CountsDiff_s{tr}")
    if got:
        jj = take(d, "lungpericyte_K128_g1p0.json", f"CountsDiff_s{tr}"); j = json.load(open(jj[0]))
        assert j["K"] == 128 and j["guidance"] == 1.0 and tr in str(j.get("ckpt", "")), (tr, j)
        man.append({"method": "CountsDiff", "arm": "K128_g1p0", "seed": int(j["seed"]), "train": int(tr), "target": T, "path": got[0], "sha256": got[1]})
    else: missing.append(f"CountsDiff train{tr}")
# scVI / scANVI
for meth, key, rows in (("scVI", "scvi", (("20261050", "20260987"), ("20261053", "20260988"), ("20261054", "20260989"))),
                        ("scANVI", "scanvi", (("20261060", "20260987"), ("20261063", "20260988"), ("20261064", "20260989")))):
    for tr, s in rows:
        got = take(f"{X}/r16_baselines/{key}_s{tr}/counts", f"{key}_cond_lungpericyte_n2500_seed{s}.npy", f"{meth}_s{tr}")
        if got: man.append({"method": meth, "arm": "cond", "seed": int(s), "train": int(tr), "target": T, "path": got[0], "sha256": got[1]})
        else: missing.append(f"{meth} train{tr}")
json.dump(man, open(out, "w"), indent=1)
print(len(man), "of 15 present; missing:", missing or "none")
if need_all and missing: sys.exit(3)
