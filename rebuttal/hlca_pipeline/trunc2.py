import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import sys, numpy as np
D = sys.argv[1]; T = sys.argv[2].split("|") if len(sys.argv) > 2 and sys.argv[2] else []; splits = sys.argv[3].split(",") if len(sys.argv) > 3 else ["train"]
C = [128, 256, 512, 1024, 2048, 4096]
for sp in splits:
    X = np.load(f"{D}/{sp}.npy", mmap_mode="r"); y = np.load(f"{D}/{sp}_celltype.npy", allow_pickle=True).astype(str)
    groups = [("ALL", None)] + [(t, np.flatnonzero(y == t)) for t in T]
    acc = {g: [0.0, np.zeros(len(C)), np.zeros(len(C))] for g, _ in groups}
    for lo in range(0, X.shape[0], 20000):
        b = np.asarray(X[lo:lo+20000], dtype=np.int64); yy = y[lo:lo+20000]
        for g, _ in groups:
            bb = b if g == "ALL" else b[yy == g]
            a = acc[g]; a[0] += bb.sum()
            for i, c in enumerate(C):
                a[1][i] += np.clip(bb - c, 0, None).sum(); a[2][i] += bb[bb > c].sum()
    for g, _ in groups:
        a = acc[g]
        print(sp, g, "| excess (clip loss):", {c: f"{a[1][i]/a[0]*100:.3f}%" for i, c in enumerate(C)}, "| mass of entries > c:", {c: f"{a[2][i]/a[0]*100:.3f}%" for i, c in enumerate(C)}, flush=True)
