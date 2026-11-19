"""R16 purity ruler on HLCA: the R13d recipe of
scripts/r13d/freeze_purity.py unchanged except the data. LogisticRegression(C=1, max_iter=300) on ALL train cells (train file minus
the val cells, 421,159 cells, all 50 classes), log-CP10K over the 2,000 HVGs, fitted ONCE at 3 BLAS threads and saved; every R16
purity and ceiling is then predicted from the saved coefficients, never refitted.
Ceiling of a target class = share of ALL its val cells predicted as that class (pre-registration item 7).
Prediction = classes[argmax(log_cp10k(x) @ coef.T + intercept)], checked equal to clf.predict on all val cells.
Reads train and val only (never test).
    freeze_purity_hlca.py DATA_DIR OUT_DIR [MAX_ITER]
DATA_DIR holds the arrays from load_split() in r16_data.py (Windows' export, sha-checked before this runs).
v2: v1 (60736d1a) hit max_iter 300 unconverged; the ruler is refitted with ONLY max_iter raised (argument,
2,000), nothing else changed. Also written: the max |gradient| of sklearn's lbfgs objective at the saved coefficients
(mean multinomial log-loss + ||coef||^2 / (2 C n); lbfgs stops when this is <= tol 1e-4) and the objective value, for both fits.
"""
import hashlib, json, os, sys, time, threadpoolctl
import numpy as np, sklearn, joblib
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r16_common import TARGETS
from r16_data import load_split, split_files

D, O = sys.argv[1], sys.argv[2]
MAX_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 300
os.makedirs(O, exist_ok=False)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def log_cp10k(x):            # same as r13_score.log_cp10k / freeze_purity.py, applied to dense blocks
    x = np.asarray(x, dtype=np.float64)
    lib = x.sum(1, keepdims=True)
    return np.log1p(x / np.maximum(lib, 1) * 1e4).astype(np.float32)


def log_cp10k_blocks(X, b=20000):
    return np.concatenate([log_cp10k(X[i:i + b].toarray() if hasattr(X, "toarray") else X[i:i + b]) for i in range(0, X.shape[0], b)])


threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
assert all(v == "3" for v in threads.values()), threads
Xt, yt, gt = load_split(D, "train"); Xv, yv, gv = load_split(D, "val")
assert np.array_equal(gt, gv) and len(gt) == 2000, "train/val genes differ or not 2,000"
assert Xt.shape[0] == 421159 and Xv.shape[0] == 46796, (Xt.shape, Xv.shape)
Zt = log_cp10k_blocks(Xt); Zv = log_cp10k_blocks(Xv)
t0 = time.time()
clf = LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(Zt, yt)
fit_s = time.time() - t0


def objective_grad(W, b, Z, y, classes, C=1.0, blk=50000):
    """sklearn lbfgs objective for multinomial LR: mean log-loss + ||W||^2 / (2 C n); returns (value, max |gradient|)."""
    n = len(Z); yi = np.searchsorted(classes, y); assert (classes[yi] == y).all()
    gW = np.zeros_like(W, dtype=np.float64); gb = np.zeros(len(b)); loss = 0.0
    for i in range(0, n, blk):
        z = Z[i:i + blk].astype(np.float64); s = z @ W.T + b; s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True); r = np.arange(len(z))
        loss -= np.log(p[r, yi[i:i + blk]]).sum(); p[r, yi[i:i + blk]] -= 1.0
        gW += p.T @ z; gb += p.sum(0)
    val = loss / n + (W ** 2).sum() / (2 * C * n)
    g = np.concatenate([(gW / n + W / (C * n)).ravel(), gb / n])
    return float(val), float(np.abs(g).max())


obj, gmax = objective_grad(clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64), Zt, yt, clf.classes_.astype(str))
prev = None
PREV = os.path.expanduser("~/research/crndiff-rebuttal/_archive_r16/purity_lr_maxiter300_06c0c711/purity_lr_hlca.npz")
if os.path.exists(PREV) and MAX_ITER != 300:
    zp = np.load(PREV); assert np.array_equal(zp["classes"], clf.classes_.astype(str))
    prev = objective_grad(zp["coef"].astype(np.float64), zp["intercept"].astype(np.float64), Zt, yt, zp["classes"])
NPZ = f"{O}/purity_lr_hlca.npz"
np.savez(NPZ, coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_.astype(str), genes=np.asarray(gt, dtype=str))
joblib.dump(clf, f"{O}/purity_lr_hlca.joblib")
z = np.load(NPZ)
pv = np.concatenate([z["classes"][(Zv[i:i + 8192] @ z["coef"].T + z["intercept"]).argmax(1)] for i in range(0, len(Zv), 8192)])
assert (pv == clf.predict(Zv)).all(), "saved coefficients do not reproduce clf.predict"
ceil = {}
for c in TARGETS:
    m = yv == c; wrong = pv[m][pv[m] != c]
    ceil[c] = {"n_val": int(m.sum()), "n_correct": int((pv[m] == c).sum()), "ceiling": float((pv[m] == c).mean()),
               "misassigned_to": {k: int(n) for k, n in zip(*np.unique(wrong, return_counts=True))}}
rec = {"model": f"LogisticRegression(C=1, max_iter={MAX_ITER}), all HLCA train cells (train file minus val), log-CP10K over the 2,000 HVGs "
                "(R13d freeze_purity.py recipe; only max_iter may differ), 3 BLAS threads",
       "max_iter": MAX_ITER, "objective": obj, "max_abs_gradient": gmax, "lbfgs_tol": clf.tol,
       "previous_fit_maxiter300": None if prev is None else {"npz": PREV, "objective": prev[0], "max_abs_gradient": prev[1]},
       "threads": threads, "blas": threadpoolctl.threadpool_info(), "n_iter": clf.n_iter_.tolist(), "converged": bool(clf.n_iter_.max() < MAX_ITER),
       "fit_seconds": fit_s, "n_train": int(Xt.shape[0]), "n_classes": int(len(clf.classes_)), "n_val": int(Xv.shape[0]),
       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__},
       "data_sha256": {os.path.basename(p): sha(p) for p in split_files(D, "train") + split_files(D, "val")},
       "npz_sha256": sha(NPZ), "joblib_sha256": sha(f"{O}/purity_lr_hlca.joblib"),
       "val_accuracy_all_classes": float((pv == yv).mean()), "ceilings": ceil,
       "check": "argmax of saved coefficients == clf.predict on all val cells",
       "script_sha256": sha(__file__)}
json.dump(rec, open(f"{O}/freeze_record.json", "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("max_iter", "n_iter", "converged", "objective", "max_abs_gradient", "previous_fit_maxiter300", "fit_seconds",
                                      "versions", "npz_sha256", "val_accuracy_all_classes")}))
for c, r in ceil.items():
    print(f"{c:32s} {r['n_correct']}/{r['n_val']}  ceiling {r['ceiling']:.4f}  {r['misassigned_to']}")
