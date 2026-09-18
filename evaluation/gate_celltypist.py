# -*- coding: utf-8 -*-
"""Gate G5 (A 类): the CellTypist purity ruler must be the SAME ruler the archive used.
用途：闸门 G5（裁定 164 §3）。CellTypist 不是丢失的冻结件，是 pip 包 `celltypist` +
下载到 `~/.celltypist/` 的 `Healthy_Adult_Heart.pkl`。归档 json **只记了模型文件名**，
没记版本也没记 sha256 —— 模型换个版本，标签直方图就变，新 purity 就落到另一把尺子上，
和论文表里留用的那些行不可比。

判据：用**当前机器上**的 celltypist + 模型，重打一条**已归档**的臂，
标签直方图必须与归档 **逐 key 相同**（key 与 count 都相同，不许四舍五入、不许只比 top-k）。

选哪一条：`celltypist_Myeloid_ceiling_canonical.json`。理由是它的 job 列表最短——
只有默认的 pilot（就是 §3 补迁进来的那个 tau=0 base 池），所以它同时验证
**模型 + val 数据 + base 池**三样，而且 rng 的消费序列最短、最不容易走样。
它的三份直方图（空操作 / 真实 val 建映射用 / 真实 val 细胞）全部逐 key 比。

不过 ⇒ **停下贴数字**，报给 Q。**禁止**去找旧版本模型换着试
（与 H0(b) 同一类禁令：那是拿哈希当超参搜索）。

    python -m evaluation.gate_celltypist          -> out/gates/CELLTYPIST.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                          # noqa: E402
from crndiff.io import sha256_file                       # noqa: E402

# the archived arm this gate reproduces, and the command that produced it
REFERENCE = C.CELLTYPIST / "celltypist_Myeloid_ceiling_canonical.json"
EVAL = C.CELLTYPIST / "celltypist_eval.py"
ARGS = ["--type", "Myeloid", "--n", "2500"]          # its rows are n=2500 / n=2302 (Myeloid val)
MODEL_PKL = (Path(os.path.expanduser("~")) / ".celltypist" / "data" / "models"
             / "Healthy_Adult_Heart.pkl")


def model_identity() -> dict:
    out = {"model_file": str(MODEL_PKL), "model_present": MODEL_PKL.exists()}
    try:
        import celltypist
        out["celltypist_version"] = str(celltypist.__version__)
    except Exception as e:
        out["celltypist_version"] = None
        out["celltypist_import_error"] = repr(e)
        return out
    if MODEL_PKL.exists():
        out["model_sha256"] = sha256_file(MODEL_PKL)
        out["model_bytes"] = MODEL_PKL.stat().st_size
        try:
            import celltypist as ct
            m = ct.models.Model.load(str(MODEL_PKL))
            d = getattr(m, "description", {}) or {}
            out["model_date"] = str(d.get("date"))
            out["model_version"] = str(d.get("version"))
            out["model_n_celltypes"] = d.get("number_celltypes")
            out["model_source"] = d.get("source")
        except Exception as e:
            out["model_description_error"] = repr(e)
    return out


def compare_hists(ref: dict, new: dict) -> dict:
    """Key-by-key comparison of two label histograms. No rounding, no top-k."""
    rk, nk = set(ref), set(new)
    diffs = {k: {"archived": ref.get(k), "recomputed": new.get(k)}
             for k in sorted(rk | nk) if ref.get(k) != new.get(k)}
    return {"n_labels_archived": len(rk), "n_labels_recomputed": len(nk),
            "n_cells_archived": sum(ref.values()), "n_cells_recomputed": sum(new.values()),
            "keys_only_archived": sorted(rk - nk), "keys_only_recomputed": sorted(nk - rk),
            "n_differing_keys": len(diffs), "differing": diffs,
            "identical": not diffs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", default=None, help="keep the recomputed json here (debugging)")
    a = ap.parse_args(argv)
    report = {"gate": "G5", "reference": str(REFERENCE),
              "why_this_arm": "shortest job list (pilot = the migrated tau=0 base pool), so it "
                              "checks model + val data + base pool at once",
              "identity": model_identity()}
    C.require(REFERENCE, "archived CellTypist arm to reproduce")
    C.require(EVAL, "celltypist_eval.py")
    C.require(C.BASE_POOL, "tau=0 base pool (the pilot row of the reference arm)")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if not report["identity"].get("model_present"):
        report.update({"passed": False,
                       "error": f"Healthy_Adult_Heart.pkl not found at {MODEL_PKL}; "
                                "G5 cannot run. Do NOT substitute another model."})
        return finish(report)

    with tempfile.TemporaryDirectory(prefix="hvg_g5_") as td:
        cmd = [sys.executable, str(EVAL), "--x-root", str(C.LEGACY_ROOT),
               *ARGS, "--tag", "_G5_repro", "-o", td]
        report["command"] = [str(x) for x in cmd[1:]]
        print("[G5] " + " ".join(str(x) for x in cmd[1:]))
        rc = subprocess.run(cmd, cwd=str(C.CELLTYPIST),
                            env=dict(os.environ, PYTHONIOENCODING="utf-8",
                                     PYTHONUTF8="1")).returncode
        if rc != 0:
            report.update({"passed": False, "error": f"celltypist_eval.py failed (rc={rc})"})
            return finish(report)
        hits = sorted(Path(td).glob("celltypist_Myeloid_G5_repro.json"))
        if len(hits) != 1:
            report.update({"passed": False,
                           "error": f"expected one recomputed json, found {len(hits)}"})
            return finish(report)
        new = json.loads(hits[0].read_text(encoding="utf-8"))
        if a.keep:
            Path(a.keep).write_text(json.dumps(new, ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    rd, nd = ref.get("label_dists", {}), new.get("label_dists", {})
    report["rows"] = {}
    for tag in sorted(set(rd) | set(nd)):
        if tag not in rd or tag not in nd:
            report["rows"][tag] = {"identical": False,
                                   "error": "row present on only one side",
                                   "in_archive": tag in rd, "in_recompute": tag in nd}
            continue
        report["rows"][tag] = compare_hists(rd[tag], nd[tag])
    # the scalar readings should follow from the histograms, but compare them too
    report["scalars"] = {
        "n_genes_overlap": {"archived": ref.get("n_genes_overlap"),
                            "recomputed": new.get("n_genes_overlap"),
                            "same": ref.get("n_genes_overlap") == new.get("n_genes_overlap")},
        "target_labels": {"archived": ref.get("target_labels"),
                          "recomputed": new.get("target_labels"),
                          "same": ref.get("target_labels") == new.get("target_labels")},
        "target_labels_strong": {
            "archived": ref.get("target_labels_strong"),
            "recomputed": new.get("target_labels_strong"),
            "same": ref.get("target_labels_strong") == new.get("target_labels_strong")},
        "purity_rows": {"archived": [(r["arm"], r["n"], r["purity"]) for r in ref.get("rows", [])],
                        "recomputed": [(r["arm"], r["n"], r["purity"])
                                       for r in new.get("rows", [])]},
    }
    report["scalars"]["purity_rows"]["same"] = (
        report["scalars"]["purity_rows"]["archived"]
        == report["scalars"]["purity_rows"]["recomputed"])
    report["passed"] = bool(report["rows"]
                            and all(v.get("identical") for v in report["rows"].values()))
    return finish(report)


def finish(report: dict) -> int:
    C.GATES.mkdir(parents=True, exist_ok=True)
    p = C.GATES / "CELLTYPIST.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ok = bool(report.get("passed"))
    for tag, v in (report.get("rows") or {}).items():
        print(f"[G5] {tag:28s} identical={v.get('identical')} "
              f"labels {v.get('n_labels_archived')}/{v.get('n_labels_recomputed')} "
              f"cells {v.get('n_cells_archived')}/{v.get('n_cells_recomputed')} "
              f"differing={v.get('n_differing_keys')}")
    i = report.get("identity", {})
    print(f"[G5] celltypist {i.get('celltypist_version')} | model sha "
          f"{str(i.get('model_sha256'))[:16]} | date {i.get('model_date')} "
          f"| version {i.get('model_version')}")
    if not ok and report.get("error"):
        print("[G5] error:", report["error"])
    print("[G5]", "PASSED" if ok else "FAILED", "->", p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
