import json, numpy as np
m=json.load(open("meta.json")); c=json.load(open("data_card.json"))
print("gene_hash", m["gene_hash"], "n_genes", m["n_genes"], "dtype", m["dtype"], "| val", m["val_rule"]["used"], m["val_rule"]["tries"])
print("test_donor_overlap", c["test_donor_overlap"])
print("Vg_cells", m["Vg_cells"]); print("Vg>1:", sum(v>1 for v in m["Vg"]), "max Vg", max(m["Vg"]))
for s in ("train","val","test"):
    x=np.asarray(np.load(f"{s}.npy", mmap_mode="r")); sp=c["splits"][s]
    print(s, x.shape, x.dtype, "min", int(x.min()), "max", int(x.max()), "donors", sp["n_donors"], "zero_frac", round(sp["zero_frac"],4), "lib median", sp["counts_per_cell"]["median"])
tr=set(open("train_obs_names.txt").read().split()); va=set(open("val_obs_names.txt").read().split()); te=set(open("test_obs_names.txt").read().split())
print("obs overlaps tr&va", len(tr&va), "tr&te", len(tr&te), "va&te", len(va&te), "| n", len(tr), len(va), len(te))
T=["type II pneumocyte","classical monocyte","multi-ciliated epithelial cell","plasma cell","ionocyte"]
cnt={s:{r["cell_type"]:r["n"] for r in c["splits"][s]["per_type"]} for s in ("train","val","test")}
rows=sorted(cnt["test"], key=lambda k:-cnt["test"][k])
with open("../per_class_counts.tsv","w") as f:
    f.write("cell_type\ttrain\tval\ttest\tincluded_test_ge_100\ttarget\n")
    for k in rows: f.write("\t".join([k, str(cnt["train"].get(k,0)), str(cnt["val"].get(k,0)), str(cnt["test"][k]), str(cnt["test"][k]>=100), str(k in T)]) + "\n")
for k in T: print("TARGET", k, cnt["train"][k], cnt["val"][k], cnt["test"][k])
print("min val among included:", min((cnt["val"][k],k) for k in rows if cnt["test"][k]>=100))
