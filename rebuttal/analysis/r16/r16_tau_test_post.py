#!/usr/bin/env python
"""R16: build the the run log post for one class's CRNDiff test read (computed from the json, never by eye). Asserts the class's test
floor is bit-identical to the CountsDiff and scVI/scANVI test reads (same reference and floor cells), and lists the already-read
baseline rows next to it (not re-read). Writes the post to OUT.md.
    python r16_tau_test_post.py TOKEN "STATUS time" OUT.md      e.g. classicalmonocyte "07:33" /path/post.md
"""
import hashlib, json, sys
W = "./work"
tok, when, out = sys.argv[1:4]
pc = f"{W}/runs/r16/tau_test/scores_{tok}_test.json"; C = json.load(open(pc)); assert len(C["runs"]) == 1
r0 = C["runs"][0]; t = r0["target"]; _ta = r0["arm"][3:].split("_a")   # arms "tau0p6" or, after step 2, "tau0p6_a2"
tau = float(_ta[0].replace("p", ".")); alpha = float(_ta[1]) if len(_ta) > 1 else 1.0
D = json.load(open(f"{W}/runs/r16/cd_test/scores_test.json")); Bs = json.load(open(f"{W}/runs/r16/base/scores_test.json"))
assert C["floors"][t] == D["floors"][t] == Bs["floors"][t], "test floor differs from the baseline reads"
cdk = [r for r in D["runs"] if r["target"] == t][0]
rows = [(f"CRNDiff τ{tau:g} α{alpha:g}", r0), (f"CountsDiff {cdk['arm'].split('_')[0]}", cdk)] + [(r["method"], r) for r in Bs["runs"] if r["target"] == t]
sub = {"type II pneumocyte": "test subset abee9a18 (2,500/12,435)", "classical monocyte": "test subset 012142ac (2,500/3,561)"}.get(t, f"all {r0['matched_n']:,} test cells")
L = [f"Mac → R16 CRNDiff {t} test (read once, only τ {tau:g}, α {alpha:g}; after STATUS {when})",
     f"- {sub}; scorer ed519051, purity 219e9bce, fixed class order. Results runs/r16/tau_test/scores_{tok}_test.json ({hashlib.sha256(open(pc, 'rb').read()).hexdigest()[:8]}).",
     f"- the noise floor is bit-identical to the test reads of CountsDiff and scVI/scANVI; W1 noise floor {C['floors'][t]['W1']['W1']:.4f}, ceiling {r0['purity_logreg_val_ceiling']:.4f}(all test cells of the class).",
     "", "| method | W1 | MMD² | PCC | 1-NN | precision | recall | coverage | purity |", "|---|---|---|---|---|---|---|---|---|"]
for n, r in rows:
    L.append(f"| {n} | {r['W1']:.4f} | {r['mmd2']:.5f} | {r['pcc']:.4f} | {r['nn1_acc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['coverage']:.3f} | {r['purity_logreg']:.4f} |")
L += ["", "- 1 training seed(s), 1 sampling seed; baseline rows are numbers read earlier, shown side by side only, not re-read."]
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
