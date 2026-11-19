# Truncated count mass on the scale the frozen generator sees: train is np.clip(train, 0, NMAX) (model_hvg2k.py line 154),
# raw counts; truncated mass = sum(max(x - NMAX, 0)) / sum(x). Overall and per target class. Read-only.
import sys, numpy as np
D = sys.argv[1]; targets = sys.argv[2].split("|") if len(sys.argv) > 2 and sys.argv[2] else []
X = np.load(f"{D}/train.npy", mmap_mode="r"); y = np.load(f"{D}/train_celltype.npy", allow_pickle=True).astype(str)
C = [256, 512, 1024, 2048, 4096]; tot = 0; over = dict.fromkeys(C, 0); tt = dict.fromkeys(targets, 0); to = {t: dict.fromkeys(C, 0) for t in targets}
for i in range(0, X.shape[0], 50000):
    b = np.asarray(X[i:i + 50000], dtype=np.int64); yb = y[i:i + 50000]; tot += b.sum()
    for c in C: over[c] += np.clip(b - c, 0, None).sum()
    for t in targets:
        m = yb == t
        if m.any():
            bt = b[m]; tt[t] += bt.sum()
            for c in C: to[t][c] += np.clip(bt - c, 0, None).sum()
print("ALL", {c: f"{over[c] / tot * 100:.3f}%" for c in C})
for t in targets: print(t, {c: f"{to[t][c] / tt[t] * 100:.3f}%" for c in C})
