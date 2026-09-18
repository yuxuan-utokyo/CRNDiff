# -*- coding: utf-8 -*-
"""The gates that must pass before any HVG experiment grid is queued (task book section B2).
Purpose: acceptance gates G1 to G4.

    G1  telescoping: on a run with no intermediate resampling, |delivered log weight minus
    G2  bitwise reproduction: `--sampler plugin --reward-mode probability` reproduces an archived
                      frozen run, and the sha256 of the samples must match. This is the core port-correctness
    G2b determinism: the same stem run twice, into separate output directories, gives bitwise
    G3  unconditional quality: one unconditional run agrees with the archived unconditional run.
    G4  discriminator: the migrated frozen discriminator is used as is and its held-out metric is

A failure exits non-zero. G1, G2 and G2b failing means a bug; G3 or G4 failing is a result.

    python -m analysis.gates            -> out/gates/GATES.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crndiff import config as C                                  # noqa: E402
from crndiff.io import load_runs, sha256_array, sha256_file      # noqa: E402

TOL = 1.0e-10          # frozen G1 threshold; deliberately NOT a CLI knob
G4_AUROC = 0.90        # frozen G4 threshold (held-out, experiment targets)

# --- G2's archived reference -------------------------------------------------------------
# The ONLY frozen fk_hvg_v2 run with intermediate_scale = 1.0, i.e. the one configuration the
# single-parameter port can reproduce without the knob section B1 deletes. Chosen mechanically:
# `ls *_fkv2_*_sc1p0_*.json` in the old tree returns exactly this one file.
OLD_TREE = C.ARCHIVE
G2_REFERENCE = {
    "stem": ("Myeloid_residual_fkv2_fk_tiltevery_trigscheduled_K32_M5000_N2500"
             "_alpha2p0_sc1p0_int8_seed20260987"),
    "npy": (OLD_TREE / "results" / "runs" /
            "Myeloid_residual_fkv2_fk_tiltevery_trigscheduled_K32_M5000_N2500"
            "_alpha2p0_sc1p0_int8_seed20260987.npy"),
    "json": (OLD_TREE / "results" / "runs" /
             "Myeloid_residual_fkv2_fk_tiltevery_trigscheduled_K32_M5000_N2500"
             "_alpha2p0_sc1p0_int8_seed20260987.json"),
    "why": "the only archived fkv2 run with intermediate_scale = 1.0 (see section B1)",
}
# --- G3's archived reference: the frozen unconditional (product-tilted) arm ---------------
G3_REFERENCE = {
    "npy": (OLD_TREE / "results" / "runs" /
            "linspace_T_O_K32_spilot_run1c_s20260625_samp20261043_batched1_chunk256"
            "_UNI_DONOR_D1_n10000_g0p4_noa4.npy"),
    "json": (OLD_TREE / "results" / "runs" /
             "sample_log_K032_spilot_run1c_s20260625_chunk256_batched1"
             "_UNI_DONOR_D1_n10000_g0p4_noa4.json"),
    "sha256": "97d9309da99ae0c6c54e3d5d731bd55b9a5a1f427452d07d2463b09b3ef8cec7",
}
# The reproduction gate H0 writes here (the frozen sampler's own output directory).
H0_OUTPUT = (C.LEGACY_ROOT / "Baseline" / "OURS" / "results" /
             "linspace_T_O_K32_spilot_run1c_s20260625_samp20261043_batched1_chunk256"
             "_UNI_DONOR_D1_n10000_g0p4_noa4.npy")

FIELDS_G2 = ("n_intermediate_resamples", "resample_steps", "unique_lineage_fraction",
             "unique_cell_fraction", "ess_pre_final_fraction", "ess_min_fraction")


def by_role(experiment: str = "gate") -> dict:
    out: dict[str, list[dict]] = {}
    for r in load_runs(experiment):
        out.setdefault(r.get("role"), []).append(r)
    return out


def gate_telescope(runs: dict) -> dict:
    hits = runs.get("gate_telescope", [])
    if len(hits) != 1:
        return {"passed": False, "error": f"expected exactly one gate_telescope run, "
                                          f"found {len(hits)}"}
    r = hits[0]
    err = r.get("telescope_max_abs_error")
    ok = err is not None and err < TOL
    return {"stem": r["stem"], "sampler": r.get("sampler"), "reward_mode": r.get("reward_mode"),
            "telescope_max_abs_error": err, "threshold": TOL,
            "n_intermediate_resamples": r.get("n_intermediate_resamples"),
            "passed": bool(ok)}


def gate_port_fidelity(runs: dict) -> dict:
    hits = runs.get("gate_port", [])
    if len(hits) != 1:
        return {"passed": False, "error": f"expected exactly one gate_port run, found {len(hits)}",
                "reference": G2_REFERENCE["stem"]}
    r = hits[0]
    ref_npy = G2_REFERENCE["npy"]
    out = {"stem": r["stem"], "reference_stem": G2_REFERENCE["stem"],
           "reference_npy": str(ref_npy), "reference_choice": G2_REFERENCE["why"],
           "sampler": r.get("sampler"), "reward_mode": r.get("reward_mode")}
    if not ref_npy.exists():
        out.update({"passed": False, "error": f"archived reference not found: {ref_npy}"})
        return out
    ref = np.load(ref_npy, allow_pickle=False)
    new = np.load(Path(r["_path"]).with_suffix(".npy"), allow_pickle=False)
    out["sha256_reference"] = sha256_array(ref)
    out["sha256_reproduced"] = sha256_array(new)
    out["shapes"] = [list(ref.shape), list(new.shape)]
    out["bit_identical"] = bool(ref.shape == new.shape and np.array_equal(ref, new))
    if G2_REFERENCE["json"].exists():
        rj = json.loads(G2_REFERENCE["json"].read_text(encoding="utf-8"))
        out["fields"] = {k: {"archived": rj.get(k), "reproduced": r.get(k),
                             "same": rj.get(k) == r.get(k)} for k in FIELDS_G2}
    out["passed"] = bool(out["bit_identical"])
    return out


def gate_determinism(runs: dict) -> dict:
    """G2b: re-execute the gate_port configuration in a SECOND process writing to an
    independent HVG_OUT, and require the delivered cells to be bit-identical."""
    hits = runs.get("gate_port", [])
    if len(hits) != 1:
        return {"passed": False, "error": "G2b needs the single gate_port run"}
    r = hits[0]
    argv = r.get("_rerun_argv")
    if argv is None:
        return {"passed": False,
                "error": "the gate_port run json carries no _rerun_argv; re-queue it with "
                         "experiments/manifests.py so the exact command line is recorded"}
    with tempfile.TemporaryDirectory(prefix="hvg_g2b_") as td:
        rc = subprocess.run([sys.executable, *argv], cwd=str(ROOT),
                            env=dict(os.environ, HVG_OUT=td)).returncode
        if rc != 0:
            return {"passed": False, "error": f"the second execution failed (rc={rc})"}
        hits2 = sorted(Path(td).rglob(f"{r['stem']}.npy"))
        if len(hits2) != 1:
            return {"passed": False, "error": f"second execution produced {len(hits2)} outputs"}
        a = np.load(Path(r["_path"]).with_suffix(".npy"), allow_pickle=False)
        b = np.load(hits2[0], allow_pickle=False)
    return {"stem": r["stem"], "sha256_first": sha256_array(a), "sha256_second": sha256_array(b),
            "passed": bool(a.shape == b.shape and np.array_equal(a, b))}


def gate_unconditional() -> dict:
    """G3: the unconditional (product-tilted, no classifier) arm reproduced here against the
    frozen archive. This is gate H0's artefact -- the same run answers both questions, so no
    extra sampling is spent: H0 asks whether the rebuilt panel61 is the original, G3 asks
    whether the unconditional quality matches what the old tree reported."""
    ref_npy, ref_json = G3_REFERENCE["npy"], G3_REFERENCE["json"]
    out = {"reference_npy": str(ref_npy), "reference_sha256_recorded": G3_REFERENCE["sha256"],
           "reproduction": str(H0_OUTPUT)}
    if not ref_npy.exists():
        return {**out, "passed": False, "error": f"archived reference not found: {ref_npy}"}
    out["reference_sha256_measured"] = sha256_file(ref_npy)
    if out["reference_sha256_measured"] != G3_REFERENCE["sha256"]:
        return {**out, "passed": False,
                "error": "the archived reference itself does not match the sha256 recorded in "
                         "receipt 036 -- stop and report, do not proceed"}
    if ref_json.exists():
        rj = json.loads(ref_json.read_text(encoding="utf-8"))
        cell = next(iter(rj["cells"].values()))
        out["archived_statistics"] = {k: cell[k] for k in ("mean", "max", "zerofrac")}
    if not H0_OUTPUT.exists():
        return {**out, "passed": False,
                "error": "no reproduction present -- run gate H0 first "
                         "(tools/run_h0 or the archived sample_hvg2k_noa4 command)"}
    out["reproduction_sha256"] = sha256_file(H0_OUTPUT)
    gen = np.load(H0_OUTPUT, allow_pickle=False)
    out["reproduction_statistics"] = {"mean": float(gen.mean()), "max": int(gen.max()),
                                      "zerofrac": float(np.mean(gen == 0))}
    out["bit_identical"] = bool(out["reproduction_sha256"] == out["reference_sha256_measured"])
    out["passed"] = bool(out["bit_identical"])
    return out


def gate_discriminators() -> dict:
    """G4: RECORD the held-out metrics of the migrated frozen discriminators. Nothing is
    refitted here; a missing log is reported, never imputed.

    SCOPE (ruling 164 section 1): the statistic is over the discriminators a run can actually
    `load`. A directory with no `residual_classifier_portable.npz` holds no classifier at all --
    no arm can load it -- so it is excluded, with the reason recorded. That criterion has
    nothing to do with the AUROC: it excludes the same directory whatever the number says.

        For the record, all three points:
            * the 0.90 threshold is NOT pre-registered. The gate as written says only 'record, do not
                refit'; 0.90 was carried over from the toy package and has not been changed since.
            * this ruling was made AFTER seeing the value, and does not pretend otherwise. It stands
                on two facts that do not depend on the number: the module does not exist, and the
            * the counts and the full tables stay in the gates json so anyone can recompute them.
    """
    rows = {}
    for d in sorted(p for p in C.RESIDUAL.iterdir() if p.is_dir()) if C.RESIDUAL.exists() else []:
        log = C.residual_paths(d)["log"]
        if not log.exists():
            rows[f"residual/{d.name}"] = {"error": f"missing {log}", "loadable": False,
                                          "excluded_reason": "no residual_log.json"}
            continue
        blob = json.loads(log.read_text(encoding="utf-8")).get("classifier", {})
        present = C.residual_paths(d)["portable"].exists()
        rows[f"residual/{d.name}"] = {
            "heldout_roc_auc": blob.get("heldout_roc_auc"),
            "gate_passed_at_fit_time": blob.get("gate_passed"),
            "portable_present": present,
            "loadable": present,
            "excluded_reason": (None if present else
                                "no residual_classifier_portable.npz: the classifier body does "
                                "not exist, so no arm can load it (and its own residual_log "
                                f"records gate_passed={blob.get('gate_passed')!r} at fit time)")}
    for p in sorted(C.CLEAN_REWARD.glob("*_clean_logistic.json")) if C.CLEAN_REWARD.exists() else []:
        npz = p.with_suffix(".npz")
        rows[f"clean/{p.stem.split('_clean')[0]}"] = {
            "roc_auc": json.loads(p.read_text(encoding="utf-8")).get("validation", {}).get("roc_auc"),
            "gate_passed_at_fit_time": json.loads(
                p.read_text(encoding="utf-8")).get("validation", {}).get("gate_passed"),
            "portable_present": npz.exists(), "loadable": npz.exists(),
            "excluded_reason": (None if npz.exists() else
                                "no *_clean_logistic.npz: the classifier body does not exist")}
    auc_of = (lambda v: v.get("heldout_roc_auc") if v.get("heldout_roc_auc") is not None
              else v.get("roc_auc"))
    recorded = {k: auc_of(v) for k, v in rows.items() if auc_of(v) is not None}
    in_scope = {k: auc_of(v) for k, v in rows.items()
                if v.get("loadable") and auc_of(v) is not None}
    excluded = {k: {"heldout_roc_auc": auc_of(v), "reason": v.get("excluded_reason")}
                for k, v in rows.items() if not v.get("loadable")}
    return {"all": rows,                       # the full 9-row table, kept for recomputation
            "recorded": rows,                  # backwards-compatible alias
            "scope": "discriminators a run can actually load "
                     "(classifier body present) -- ruling 164 section 1",
            "excluded": excluded,
            "threshold_reference": G4_AUROC,
            "threshold_is_preregistered": False,
            "threshold_note": "0.90 came from Toy2D's G4; task book section B2 says "
                              "'record, do not retrain' and states no threshold. Unchanged here.",
            "n_recorded": len(recorded),
            "n_loadable": len(in_scope),
            "n_at_or_above_threshold_in_scope": sum(1 for v in in_scope.values() if v >= G4_AUROC),
            "n_at_or_above_threshold_all_9": sum(1 for v in recorded.values() if v >= G4_AUROC),
            "retrained": False,
            "passed": bool(in_scope and all(v >= G4_AUROC for v in in_scope.values()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-g2b", action="store_true",
                    help="G2b re-executes a full run; skip only when explicitly asked")
    a = ap.parse_args(argv)
    runs = by_role("gate")
    report = {"tol": TOL,
              "telescope": gate_telescope(runs),
              "port_fidelity": gate_port_fidelity(runs),
              "determinism": ({"passed": None, "skipped": True} if a.skip_g2b
                              else gate_determinism(runs)),
              "unconditional": gate_unconditional(),
              "discriminators": gate_discriminators()}
    considered = [report["telescope"], report["port_fidelity"], report["unconditional"],
                  report["discriminators"]]
    if not a.skip_g2b:
        considered.append(report["determinism"])
    report["passed"] = bool(all(bool(v.get("passed")) for v in considered))
    C.GATES.mkdir(parents=True, exist_ok=True)
    (C.GATES / "GATES.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("[gates]", "PASSED" if report["passed"] else "FAILED", "->", C.GATES / "GATES.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
