# -*- coding: utf-8 -*-
"""Train-on-synthetic, test-on-real under ELEVEN-CLASS REPLACEMENT.

The archived module is not edited; its sha256 is checked at the start and at the end. The
preprocessing, the data paths, the type list, the per-type training size and the seed are all
taken from that module's own objects rather than rewritten.

The classifier is word for word the archived `fit_score`, with ONE difference: the worker count
is pinned to one, to respect a hard CPU budget. This file extracts that source and diffs it
against its own implementation, writing the diff out; it continues only if that is the only
difference. `fit_score` is a closure inside `block_a` and cannot be imported, so it has to be
copied, which is exactly why the diff exists. The worker count has no effect on the lbfgs
multinomial path anyway, and the three-class self-check below verifies that rather than assuming it.

Two modes:

* `--mode 3class` runs the OLD configuration (three classes, the same test set and the same
    sampling order) and compares against the archived block. This is the script's self-check and
    it must agree to 1e-9, otherwise the eleven-class run does not start.
* `--mode 11class` is the main run: the three requested classes are supplied by generated cells
    only, the other eight by real training cells, and the test set is every held-out real cell.

One process per seed, serial, low priority, two threads.

    python -m analysis.e33_tstr11 --mode 3class  --seed 20260833
    python -m analysis.e33_tstr11 --mode 11class --seed 20260833
"""
from __future__ import annotations

# the thread count must be pinned BEFORE numpy is imported (hard CPU budget)
import os                                                            # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import argparse                                                      # noqa: E402
import difflib                                                       # noqa: E402
import inspect                                                       # noqa: E402
import json                                                          # noqa: E402
import re                                                            # noqa: E402
import sys                                                           # noqa: E402
import textwrap                                                      # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402
try:
    from sklearn.linear_model import LogisticRegression                  # noqa: E402
except ModuleNotFoundError as _exc:                                      # noqa: E402
    raise SystemExit(
        "this entry point needs scikit-learn, which the offline reproduction does not "
        "install. See README section 7; in short: pip install scikit-learn joblib") from _exc
from sklearn.metrics import confusion_matrix, f1_score               # noqa: E402
from sklearn.metrics import precision_score, recall_score            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                          # noqa: E402
from evaluation import downstream_multiseed as MS                             # noqa: E402
from evaluation import downstream_deployed_arm as DA                          # noqa: E402

WORK = C.OUT / "e33_211"
SUMMARY = C.OUT / "summary"
ARMS_5 = ["OURS", "scVI", "scANVI", "CFGen", "MDLM"]
ARMS_4 = ["OURS", "scVI", "scANVI", "CFGen"]
MDLM_SEED = 20260931
MAN_DIR = C.OUT / "e33_209" / "manifest"        # the manifest copy that includes the MDLM entries
N_THREADS = 2


def low_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return {"nice": "BELOW_NORMAL_PRIORITY_CLASS", "pid": p.pid}
    except Exception as e:                                           # noqa: BLE001
        return {"nice": f"not set ({type(e).__name__}: {e})"}


def cpu_now():
    try:
        import psutil
        return {"cpu_percent": psutil.cpu_percent(interval=0.4),
                "n_logical": psutil.cpu_count(),
                "proc_cpu_percent": psutil.Process().cpu_percent(interval=None),
                "rss_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}
    except Exception:                                                # noqa: BLE001
        return {}


# --------------------------------------------------------------- fit_score --
# what follows is a verbatim copy of the archived fit_score, with the worker count pinned to one
# and the class list taken as an argument, which is the only place three becomes eleven
def fit_score(train_sets, labels, Xte, yte, label, cp10k_log1p, seed):
    Xtr = np.concatenate([cp10k_log1p(m) for m in train_sets])
    ytr = np.concatenate([[t] * len(m) for t, m in zip(labels, train_sets)])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=2000, class_weight="balanced",
                             random_state=seed, n_jobs=1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    cls = list(clf.classes_)
    return {"arm": label,
            "n_train_per_type": [int(len(m)) for m in train_sets],
            "per_type_F1": {t: float(f) for t, f in
                            zip(clf.classes_,
                                f1_score(yte, pred, average=None, labels=clf.classes_))},
            "macro_F1": float(f1_score(yte, pred, average="macro")),
            "confusion_matrix": confusion_matrix(yte, pred, labels=cls).tolist(),
            "confusion_labels": cls,
            # diagnostics for the eleven-class mode only; the three-class self-check does not compare these
            "per_type_precision": {t: float(v) for t, v in zip(
                cls, precision_score(yte, pred, average=None, labels=cls,
                                     zero_division=0))},
            "per_type_recall": {t: float(v) for t, v in zip(
                cls, recall_score(yte, pred, average=None, labels=cls,
                                  zero_division=0))},
            "_pred": pred}


BODY_FIT_SCORE_EXPECTED_DIFF = ["n_jobs=-1", "n_jobs=1"]


def fit_score_diff(m) -> dict:
    """Extract the archived fit_score and diff it against the few lines implemented here."""
    src = inspect.getsource(m.block_a)
    mm = re.search(r"( {4}def fit_score.*?)\n\n {4}res = ", src, re.S)
    if not mm:
        raise SystemExit("could not locate fit_score inside the body's block_a")
    body_src = textwrap.dedent(mm.group(1))
    mine = textwrap.dedent(inspect.getsource(fit_score))
    # compare only the classifier, the fit and the predict lines; this run reports more fields
    def core(s):
        return [l.strip() for l in s.splitlines()
                if re.search(r"LogisticRegression|penalty=|max_iter=|random_state=|"
                             r"clf\.fit|clf\.predict|n_jobs", l)]
    a, b = core(body_src), core(mine)
    d = list(difflib.unified_diff(a, b, fromfile="body fit_score",
                                  tofile="e33_tstr11.fit_score", lineterm="", n=1))
    changed = [l[1:].strip() for l in d if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    only_n_jobs = all(("n_jobs=-1" in c or "n_jobs=1" in c) for c in changed) and changed
    p = SUMMARY / "211_fit_score_diff.txt"
    p.write_text("\n".join(d) + "\n\n"
                 f"# core lines compared: body {len(a)}, here {len(b)}\n"
                 f"# changed lines: {changed}\n"
                 f"# only difference is n_jobs: {bool(only_n_jobs)}\n"
                 "# n_jobs is consulted by sklearn only for liblinear / one-vs-rest; on the\n"
                 "# lbfgs multinomial path it takes no part in the computation. Stage 2\n"
                 "# checks that empirically rather than taking it on faith.\n",
                 encoding="utf-8")
    return {"diff_file": str(p), "changed_lines": changed,
            "only_difference_is_n_jobs": bool(only_n_jobs),
            "body_core_lines": a, "here_core_lines": b}


# ------------------------------------------------------------------- data --
def deliveries(m, arms):
    """The delivered arrays of the five arms: the same batch the differential-expression table uses."""
    was = m.PLUS
    m.PLUS = MAN_DIR                       # the manifest copy containing the MDLM entries, built earlier and unmodified
    by = DA.manifest_local(m)
    m.PLUS = was
    by, swap = DA.swap_ours(by, DA.DEPLOYED)
    out = {}
    for arm in arms:
        for t in m.TYPES:
            its = by.get((arm, t))
            if not its:
                raise SystemExit(f"no delivered array for {arm}/{t}")
            out[(arm, t)] = Path(its[0]["npy"])
    return out, swap


def run(mode: str, seed: int) -> int:
    t0 = time.time()
    prio = low_priority()
    WORK.mkdir(parents=True, exist_ok=True)
    m, prov = MS.load_body(WORK / "body")
    body_sha = MS.sha256_file(MS.SRC)
    fsd = fit_score_diff(m)
    if not fsd["only_difference_is_n_jobs"]:
        raise SystemExit(f"fit_score differs by more than n_jobs: {fsd['changed_lines']}")
    print(f"[sha] body {body_sha}", flush=True)
    print(f"[prio] {prio}   threads {N_THREADS}", flush=True)
    print(f"[fit_score] only difference vs the body: {fsd['changed_lines']}", flush=True)

    arms = ARMS_4 if mode == "3class" else ARMS_5
    paths, swap = deliveries(m, arms)

    Xtr_all = np.load(m.DATA / "train.npy", mmap_mode="r")
    ytr_all = np.load(m.DATA / "train_celltype.npy", allow_pickle=True).astype(str)
    Xv = np.load(m.DATA / "val.npy", mmap_mode="r")
    yv = np.load(m.DATA / "val_celltype.npy", allow_pickle=True).astype(str)

    if mode == "3class":
        labels = list(m.TYPES)
        te_i, te_y = [], []
        for t in labels:                                   # verbatim from the archived module
            ii = np.flatnonzero(yv == t)
            te_i.append(np.sort(ii))
            te_y += [t] * len(ii)
        te_i = np.concatenate(te_i)
        yte = np.array(te_y)
    else:
        labels = sorted(set(ytr_all))                      # all eleven atlas classes, in lexicographic order
        if len(labels) != 11:
            raise SystemExit(f"expected 11 atlas classes, found {len(labels)}: {labels}")
        te_i = np.arange(len(yv))                          # every held-out cell
        yte = yv.copy()
    Xte = m.cp10k_log1p(np.asarray(Xv[te_i]))
    print(f"[test] {mode}: {Xte.shape[0]} cells, {len(labels)} classes", flush=True)

    rng = np.random.default_rng(seed)
    res = {"ticket": 211, "mode": mode, "seed": seed, "labels": labels,
           "n_classes": len(labels), "n_test": int(Xte.shape[0]),
           "n_test_per_class": {t: int((yte == t).sum()) for t in labels},
           "n_per_type_train": m.N_PER_TYPE_TRAIN,
           "classifier_random_state": m.SEED,
           "arm_swap": swap, "arms": {}, "cpu": []}

    others = [t for t in labels if t not in m.TYPES]

    def real_sets_for(lbls):
        """Real training cells, up to the per-type size, all of them if fewer. Same as the archived ceiling."""
        out = []
        for t in lbls:
            ii = np.flatnonzero(ytr_all == t)
            n = min(m.N_PER_TYPE_TRAIN, ii.size)
            out.append(np.asarray(Xtr_all[np.sort(rng.permutation(ii)[:n])]))
        return out

    def gen_sets_for(arm):
        out = []
        for t in m.TYPES:                                  # verbatim from the archived module
            a = np.load(paths[(arm, t)], mmap_mode="r")
            n = min(m.N_PER_TYPE_TRAIN, a.shape[0])
            out.append(np.asarray(a[np.sort(rng.permutation(a.shape[0])[:n])]))
        return out

    def score(sets, lbls, name):
        res["cpu"].append({"arm": name, "when": "before_fit", **cpu_now()})
        t1 = time.time()
        r = fit_score(sets, lbls, Xte, yte, name, m.cp10k_log1p, m.SEED)
        r["fit_walltime_s"] = time.time() - t1
        res["cpu"].append({"arm": name, "when": "after_fit", **cpu_now()})
        pred = r.pop("_pred")
        if mode == "11class":
            r["confusion_for_targets"] = {
                t: {k: int(v) for k, v in sorted(
                    zip(*np.unique(pred[yte == t], return_counts=True)),
                    key=lambda kv: -kv[1])} for t in m.TYPES}
        res["arms"][name] = r
        print(f"  {name:<14} macro-F1 {r['macro_F1']:.4f}  "
              + "  ".join(f"{t[:4]} {r['per_type_F1'][t]:.4f}" for t in m.TYPES)
              + f"   ({r['fit_walltime_s']:.0f}s)", flush=True)
        return r

    # the ceiling is drawn first, in the archived consumption order
    if mode == "3class":
        score(real_sets_for(labels), labels, "REAL (ceiling)")
    else:
        score(real_sets_for(labels), labels, "REAL (ceiling)")

    for arm in arms:
        if mode == "3class":
            sets, lbls = gen_sets_for(arm), labels
        else:
            # the three requested classes use generated cells and the other eight real training cells
            sets = gen_sets_for(arm) + real_sets_for(others)
            lbls = list(m.TYPES) + others
        score(sets, lbls, arm)

    res["fit_score_diff"] = fsd
    res["body"] = prov
    res["body_sha256"] = body_sha
    res["sha256_after_run"] = MS.sha256_file(MS.SRC)
    if res["sha256_after_run"] != body_sha:
        raise SystemExit("body changed during the run -- STOP")
    res["arm_arrays"] = {f"{a}|{t}": {"path": str(p), "sha256": MS.sha256_file(p)}
                         for (a, t), p in paths.items()}
    res["cpu_threads"] = N_THREADS
    res["priority"] = prio
    res["walltime_s"] = time.time() - t0
    res["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    q = WORK / f"raw_{mode}_seed_{seed}.json"
    q.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n",
                 encoding="utf-8")
    print(f"[out] {q}   ({res['walltime_s']:.0f}s total)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("3class", "11class"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    if a.seed not in MS.SEEDS:
        raise SystemExit(f"--seed must be one of {MS.SEEDS}")
    return run(a.mode, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
