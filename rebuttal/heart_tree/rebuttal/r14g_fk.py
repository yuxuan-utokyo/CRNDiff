# -*- coding: utf-8 -*-
"""R14 a1 (tilt+FK) for generators 20260627 / 20260630. Takes the s25 a1 run's own _rerun_argv
(out/runs/intersect_ladder/a1/<stem>_seed20260987.json) and changes ONLY:
  --experiment intersect_ladder -> rebuttal_R14g ; --arm a1 -> a1_g<GEN> ; --seed -> the HVG2K Table 1 pairing (627 x 20260987, 630 x 20260989)
  + --ckpt  assets/legacy_root/Baseline/OURS/models/pilot_run214_s<GEN>.pt          (s25: the frozen default generator)
  + --logpi rebuttal/out/R14g/s<GEN>/tables_ref/logc_INTERSECT_Endothelial_g128.npy  (s25: tables_ref default for the request)
  + --residual-dir rebuttal/out/R14g/s<GEN>/disc_INTERSECT_g128                      (s25: assets residual INTERSECT_Endothelial_g128)
as the main-table gen27/30 runs passed --ckpt/--logpi/--residual-dir. After the run: the run json's portable_classifier_source_sha256
must equal the sha256 of that disc's residual_classifier.joblib, else exit 2.
    python -u rebuttal/r14g_fk.py --gen 20260627
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
HVG = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True, choices=("20260627", "20260630")); a = ap.parse_args()
SEED = {"20260627": "20260987", "20260630": "20260989"}[a.gen]
src = sorted((HVG / "out" / "runs" / "intersect_ladder" / "a1").glob("*seed20260987.json"))[0]
argv = list(json.loads(src.read_text(encoding="utf-8"))["_rerun_argv"])
def setv(flag, val):
    i = argv.index(flag); argv[i + 1] = val
setv("--experiment", "rebuttal_R14g"); setv("--arm", f"a1_g{a.gen}"); setv("--seed", SEED)
O = HVG / "rebuttal" / "out" / "R14g" / f"s{a.gen}"; disc = O / "disc_INTERSECT_g128"
setv("--residual-dir", str(disc))
argv += ["--ckpt", str(HVG / "assets" / "legacy_root" / "Baseline" / "OURS" / "models" / f"pilot_run214_s{a.gen}.pt"),
         "--logpi", str(O / "tables_ref" / "logc_INTERSECT_Endothelial_g128.npy")]
print("argv:", " ".join(argv), flush=True)
t0 = time.time(); rc = subprocess.run([sys.executable, "-u", *argv], cwd=HVG).returncode
print(f"run_one exit {rc} wall {time.time() - t0:.0f}s", flush=True)
if rc != 0: sys.exit(rc)
hits = sorted((HVG / "out" / "runs" / "rebuttal_R14g" / f"a1_g{a.gen}").glob(f"*seed{SEED}.json"))
assert len(hits) == 1, hits
j = json.loads(hits[0].read_text(encoding="utf-8"))
want = hashlib.sha256((disc / "residual_classifier.joblib").read_bytes()).hexdigest()
got = j.get("portable_classifier_source_sha256")
print(json.dumps({"run_json": str(hits[0]), "portable_classifier_source_sha256": got, "disc_joblib_sha256": want, "match": got == want,
                  "walltime_s": j.get("walltime_s"), "peak_alloc_gb": j.get("peak_alloc_gb"), "checkpoint_sha256": j.get("checkpoint_sha256")}), flush=True)
sys.exit(0 if got == want else 2)
