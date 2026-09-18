# -*- coding: utf-8 -*-
"""One line = one run. Every argument that moves a number is explicit; nothing is inferred.
Purpose: the command line for one run. Writes a json (diagnostics and label-free readings) and
an npy of the delivered cells to `out/runs/<experiment>/<arm>/`. The assembly order follows the
frozen `frozen/fk_hvg_v2.py::main` word for word: tf32 off, torch.manual_seed(seed), the
deduplication bank and its bitwise gate, torch_gen = seed + 1. Only three things differ:
`--reward-mode {odds,probability}`.

    python -m experiments.run_one --experiment smoke --arm twist --request DONOR_D1 \
        --reward-model residual --residual-dir UNCOND --alpha 1.0 --seed 20260987
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Direct script execution otherwise exposes experiments/queue.py as stdlib queue.
if __package__ in (None, ""):
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

from crndiff import config as C                                          # noqa: E402
from crndiff.io import (paths_for, run_stem, sha256_array, sha256_file,   # noqa: E402
                    write_run)
from crndiff.metrics import distribution_readings                        # noqa: E402
from crndiff.io import celltypist_identity                                # noqa: E402
from crndiff.multicluster_sampler import (MixtureReward, build_mixture_tilt,   # noqa: E402
                                          make_stratified_resample)
from crndiff.sampler import (CleanLogisticReward, HVGEngine, PortableHGBT,   # noqa: E402
                         make_twist_generator, run_fk, run_tilt_only,
                         systematic_resample)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arm", required=True,
                    help="descriptive configuration name; runs land in out/runs/<experiment>/<arm>/")
    ap.add_argument("--role", default="experiment")
    ap.add_argument("--ticket", type=int, default=None)
    ap.add_argument("--request", required=True,
                    help="the product tilt table to steer the proposal: "
                         "assets/tables/tables_ref/logc_<request>.npy")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sampler", choices=("plugin", "twist", "tilt_only"), default="twist",
                    help="twist = the single-parameter potential (section B1); "
                         "plugin = the frozen intermediate potential, kept for gate G2; "
                         "tilt_only = the tilted chain alone, no reward / no resampling "
                         "(the ablation arm, and the stand-in for alpha=0 -- ruling 164 sec.2)")
    ap.add_argument("--tilt", choices=("on", "off"), default="on",
                    help="WHETHER the product tilt is applied at all. Distinct from "
                         "--tilt-mode, which only says WHEN. 'off' is the residual_only "
                         "ablation and needs a discriminator fitted against the UNTILTED "
                         "proposal (--residual-dir UNTILTED_*).")
    ap.add_argument("--reward-mode", choices=("odds", "probability"), default="odds",
                    help="odds = the paper's weight; probability = port fidelity (gate G2)")
    ap.add_argument("--reward-model", choices=("clean", "residual", "mixture"),
                    default="residual")
    ap.add_argument("--reward-target", default=None,
                    help="which clean reward to use (default: --request)")
    ap.add_argument("--residual-dir", default=None,
                    help="residual reward directory; bare names resolve under "
                         "assets/models/residual/ (it names a PRODUCT, not a prereg label)")
    # NOT `required`: alpha is not a knob of the tilt_only arm (that arm evaluates no reward, so
    # `argv_for` omits it and `run_stem` keeps it out of the stem). main() below requires it for
    # every other sampler, so a genuinely missing --alpha still fails loudly.
    ap.add_argument("--alpha", type=float, default=None, help="the single model parameter")
    ap.add_argument("--J", type=int, default=C.DEFAULT_J)
    ap.add_argument("--K", type=int, default=C.DEFAULT_K)
    ap.add_argument("--tau", type=float, default=C.DEFAULT_TAU)
    ap.add_argument("--tilt-mode", choices=("last", "every"), default="every")
    ap.add_argument("--trigger", choices=("scheduled", "any"), default="scheduled")
    ap.add_argument("--min-resample-gap", type=int, default=C.DEFAULT_MIN_RESAMPLE_GAP)
    ap.add_argument("--n-particles", type=int, default=C.DEFAULT_M)
    ap.add_argument("--n-out", type=int, default=C.DEFAULT_N_OUT)
    ap.add_argument("--cell-chunk", type=int, default=C.DEFAULT_CELL_CHUNK)
    ap.add_argument("--ess-frac", type=float, default=C.DEFAULT_ESS_FRAC)
    ap.add_argument("--resample-interval", type=int, default=C.DEFAULT_INTERVAL)
    ap.add_argument("--reward-floor", type=float, default=C.DEFAULT_REWARD_FLOOR)
    ap.add_argument("--no-intermediate-resampling", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--logpi", default=None, help="override the tilt table path")
    # ---- ticket 170: multi-component (Table 8) ----
    ap.add_argument("--types", default=None,
                    help="comma-separated component cell types; switches on the mixture "
                         "reward and the per-gene mixture tilt (ticket 170)")
    ap.add_argument("--weights", default=None,
                    help="comma-separated component weights, must sum to 1 and must agree "
                         "with the request table's own provenance.w_A")
    ap.add_argument("--stratify", default=None,
                    help="comma-separated per-component quota targets; turns the arm into "
                         "quota+FK (D2). Omit for the equal-weight arm.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # ticket 182: observability only. Records the per-step ancestry snapshot the sampler
    # already maintains, and REDIRECTS the products to out/runs/ancestry_trace/<arm>/ so the
    # archived table1_ours / ablation_table10 directories are never written to. Default off,
    # so no existing command line changes behaviour.
    ap.add_argument("--ancestry-trace", action="store_true",
                    help="write a per-step ancestry npz and redirect the run to "
                         "out/runs/ancestry_trace/<arm>/")
    return ap


def resolve_reward(a) -> tuple[object, np.ndarray, float, dict]:
    """Load the reward model exactly the way the frozen main() does, and report what was used."""
    if a.reward_model == "clean":
        target = a.reward_target or a.request
        npz, log = C.clean_reward_paths(target)
        C.require(npz, "clean reward classifier")
        C.require(log, "clean reward log")
        meta = json.loads(Path(log).read_text(encoding="utf-8"))
        if not meta["validation"]["gate_passed"]:
            raise SystemExit("clean reward validation gate did not pass")
        clf = CleanLogisticReward(npz)
        return clf, clf.features, meta["validation"]["roc_auc"], {
            "clean_reward_model": str(Path(npz).resolve()),
            "clean_reward_target": target,
            "clean_reward_uses_generated_cells_for_fit": False}
    if a.residual_dir is None:
        raise SystemExit("--reward-model residual needs --residual-dir "
                         "(a directory under assets/models/residual/, or an absolute path)")
    p = C.residual_paths(a.residual_dir)
    C.require(p["portable"], "portable residual classifier")
    C.require(p["log"], "residual reward log")
    C.require(p["split"], "residual split indices")
    meta = json.loads(p["log"].read_text(encoding="utf-8"))
    if not meta["classifier"]["gate_passed"]:
        raise SystemExit("held-out residual classifier AUC gate did not pass")
    split = np.load(p["split"], allow_pickle=False)
    clf = PortableHGBT(p["portable"])
    if clf.identity_max_abs_error > 1.0e-12:
        raise SystemExit("portable classifier identity gate did not pass")
    # The frozen main() also re-derives the joblib's sha256 and compares it with the portable
    # file's recorded source_sha256. That joblib was not migrated, so the comparison is
    # REPORTED as unavailable here (with the hash the portable file itself records) instead of
    # being silently skipped -- see tools/check_assets.py.
    audit = {"residual_dir": str(p["dir"].resolve()),
             "residual_classifier_portable": str(p["portable"].resolve()),
             "portable_classifier_identity_max_abs_error": clf.identity_max_abs_error,
             "portable_classifier_source_sha256": clf.source_sha256,
             "residual_classifier_joblib": (str(p["joblib"].resolve())
                                            if p["joblib"].exists() else None),
             "residual_classifier_joblib_sha256": (sha256_file(p["joblib"])
                                                   if p["joblib"].exists() else None),
             "source_hash_check": ("performed" if p["joblib"].exists()
                                   else "UNAVAILABLE: residual_classifier.joblib was not migrated")}
    if p["joblib"].exists() and clf.source_sha256 != audit["residual_classifier_joblib_sha256"]:
        raise SystemExit("portable classifier source hash does not match frozen joblib")
    return clf, split["classifier_features"], meta["classifier"]["heldout_roc_auc"], audit


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.n_out <= 0 or a.n_out > a.n_particles:
        raise SystemExit("--n-out must be in 1..n-particles")
    if a.cell_chunk <= 0 or a.resample_interval <= 0:
        raise SystemExit("--cell-chunk and --resample-interval must be positive")
    if a.min_resample_gap < 1:
        raise SystemExit("--min-resample-gap must be at least 1")
    tilt_on = (a.tilt == "on")
    if a.sampler == "tilt_only":
        if not tilt_on:
            raise SystemExit("--sampler tilt_only with --tilt off is the untilted chain with "
                             "no potential at all: it delivers the unconditional prior. "
                             "If that is what you want, say so explicitly in a manifest.")
    elif a.alpha is None:
        raise SystemExit(f"--alpha is required for --sampler {a.sampler} "
                         "(it is only optional for tilt_only, which evaluates no reward)")
    elif not a.alpha > 0:
        # ruling 164 section 2
        raise SystemExit("alpha=0 is exactly the tilt-only arm (psi == 1); "
                         "run --sampler tilt_only instead")

    logpi_path = Path(a.logpi) if a.logpi else C.tilt_table(a.request)
    ckpt = Path(a.ckpt) if a.ckpt else C.GENERATOR_CKPT

    # ---- ticket 170: the multi-component request --------------------------------------
    # Everything here is READ from the request's own table json; nothing is typed in. The
    # proposal for this arm is NOT that .npy -- it is the per-gene mixture of the components'
    # tempered tilted marginals, built below from the per-type tables and then run at tau=1.0.
    mix_types = mix_weights = mix_targets = None
    component_tau = None
    mixture = None
    if a.types:
        if a.reward_model != "mixture":
            raise SystemExit("--types needs --reward-model mixture")
        mix_types = [t.strip() for t in a.types.split(",") if t.strip()]
        if not a.weights:
            raise SystemExit("--types needs --weights")
        mix_weights = [float(w) for w in a.weights.split(",")]
        if len(mix_types) != len(mix_weights):
            raise SystemExit("--types and --weights must have the same length")
        prov = json.loads(C.tilt_table(a.request).with_suffix(".json")
                          .read_text(encoding="utf-8"))
        pv = prov["provenance"]
        if [pv["A"], pv["B"]] != mix_types:
            raise SystemExit(f"--types {mix_types} does not match {a.request}'s own "
                             f"provenance {[pv['A'], pv['B']]}; refusing to run a pair the "
                             "request was not built for")
        want = [float(pv["w_A"]), 1.0 - float(pv["w_A"])]
        if max(abs(x - y) for x, y in zip(mix_weights, want)) > 1e-12:
            raise SystemExit(f"--weights {mix_weights} does not match the request's own "
                             f"w_A ({want}); refusing")
        m = re.search(r"--tau\s+([0-9.]+)", str(prov.get("note", "")))
        if not m:
            raise SystemExit(f"{a.request}'s table json records no `--tau` in its note; "
                             "the component temperature must be read, not guessed")
        component_tau = float(m.group(1))
        if abs(component_tau - a.tau) > 1e-12:
            raise SystemExit(f"--tau {a.tau} but {a.request}'s own note says --tau "
                             f"{component_tau}; refusing to temper the components at a "
                             "temperature the request did not register")
        if a.stratify:
            mix_targets = [float(v) for v in a.stratify.split(",")]
            if len(mix_targets) != len(mix_types):
                raise SystemExit("--stratify needs one target per component")
            if abs(sum(mix_targets) - 1.0) > 1e-9:
                raise SystemExit("--stratify targets must sum to 1")
    elif a.reward_model == "mixture":
        raise SystemExit("--reward-model mixture needs --types and --weights")

    args = dict(request=a.request, reward_model=a.reward_model, sampler=a.sampler,
                K=a.K, n_particles=a.n_particles, n_out=a.n_out, alpha=a.alpha, J=a.J,
                resample_interval=a.resample_interval, ess_frac=a.ess_frac,
                # in mixture mode the ENGINE runs at tau=1.0 (the temperature is already inside
                # each component), so that is what goes in the stem, with the component
                # temperature as its own token
                tau=(1.0 if mix_types else a.tau),
                component_tau=component_tau, stratified=bool(mix_targets),
                cell_chunk=a.cell_chunk, reward_mode=a.reward_mode, seed=a.seed,
                tilt_mode=a.tilt_mode, trigger=a.trigger, tilt_on=tilt_on,
                intermediate_resampling=not a.no_intermediate_resampling)
    stem = run_stem(args)
    if a.dry_run:
        print(stem)
        return 0
    # ---- ticket 167-AMEND: the registered manifest is the single source of truth ----
    # `experiments/queue.py` reads its manifest ONCE at start, so a plan change (seeds dropped,
    # a stage cancelled) cannot reach a queue that is already running. Checking here -- in the
    # freshly-spawned child, before any model or RNG is touched -- lets the plan change take
    # effect without killing the in-flight run or restarting the driver.
    # Only applies when the experiment HAS a manifest; ad-hoc runs (smoke, gate reruns with no
    # registered grid) are unaffected. Refusing early costs a couple of seconds, never a sample.
    man = C.OUT / "manifests" / f"{a.experiment}.json"
    if man.exists():
        try:
            registered = {r.get("stem") for r in json.loads(man.read_text(encoding="utf-8"))}
        except Exception as e:                       # a broken manifest must not silently pass
            raise SystemExit(f"cannot read the registered grid {man}: {e!r}")
        if stem not in registered:
            print(f"[cancelled] {stem}\n"
                  f"            not in the registered grid {man} "
                  f"({len(registered)} entries) -- refusing to run a de-registered cell")
            return 0
    # ticket 182: the existence check must look where this run will actually be WRITTEN.
    # Against the archived directory it would always fire (the archived run is what we are
    # reproducing) and the trace could never be produced.
    pj, pn = paths_for(stem, "ancestry_trace" if a.ancestry_trace else a.experiment, a.arm)
    if pj.exists():
        print(f"[skip] exists: {pj}")
        return 0
    if pn.exists():
        raise SystemExit(f"orphan npy without json (crashed run?): {pn} -- delete it first")

    C.require(ckpt, "frozen generator checkpoint")
    if mix_types is None:
        C.require(logpi_path, "product tilt table",
                  f"expected assets/tables/tables_kz/logc_{a.request}.npy (bare cell type) or "
                  f"assets/tables/tables_ref/logc_{a.request}.npy (DONOR / INTERSECT / MIX)")
    else:
        # the request's own .npy is NOT the proposal for this arm; only its json was read above
        for t in mix_types:
            C.require(C.TABLES_KZ / f"logc_{t}.npy", f"{t} component tilt table")
    if a.sampler == "tilt_only":
        clf, features, reward_auc, reward_audit = None, None, None, {
            "reward_model": None,
            "note": "tilt_only evaluates no reward; no discriminator is loaded"}
    elif mix_types is not None:
        mixture = MixtureReward(mix_types, mix_weights, a.reward_floor)
        clf, features, reward_auc = None, None, None
        reward_audit = {"reward_model": "mixture", **mixture.identity()}
        print("[reward] mixture: " + ", ".join(
            f"{t} w={w:.3f} auc={mixture.aucs[t]:.6f}"
            for t, w in zip(mix_types, mix_weights)), flush=True)
    else:
        clf, features, reward_auc, reward_audit = resolve_reward(a)

    import torch
    C.add_base_code_to_path()
    import sample_hvg2k_noa4 as BASE            # noqa: E402  (the frozen sampling code)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()

    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != C.GENERATOR_CKPT_SHA256 and a.ckpt is None:
        raise SystemExit(f"frozen generator checkpoint hash changed: {ckpt_sha} != "
                         f"{C.GENERATOR_CKPT_SHA256}")
    data, meta = BASE.M.load_data(("train",))
    train = data["train"]
    derived = BASE.M.derive(train)
    dim, nmax = train.shape[1], BASE.CFG.NMAX
    net = BASE.M.build_net(dim, derived["iscale"]).to(BASE.DEV)
    net.load_state_dict(torch.load(ckpt, map_location=BASE.DEV, weights_only=False))
    net.eval()
    horizon = float(BASE.M.horizon())
    tgrid, nfe = BASE.M.build_grid(a.K, horizon)
    logfact = BASE.VO._logfact_arr(nmax)

    started = time.time()
    bank = BASE.DedupBank(derived["V"], tgrid, nmax,
                          lambda tau, v, nm: BASE.VO.kmat(tau, v, nm, logfact),
                          lambda v, nm: BASE.VO.pois_prior(v, nm, logfact), BASE.DEV)
    gate = bank.verify(lambda tau, v, nm: BASE.VO.kmat(tau, v, nm, logfact), derived["V"])
    if not gate["bit_exact"]:
        raise SystemExit("deduplicated kernel bank bit-exact gate failed")
    tilt_audit = {}
    if mix_types is None:
        logpi = np.load(logpi_path, allow_pickle=False)
        if logpi.shape != (dim, nmax + 1):
            raise SystemExit(f"logpi shape {logpi.shape} != {(dim, nmax + 1)}")
        engine_tau = a.tau
    else:
        # ticket 170: per-gene mixture of the components' TEMPERED tilted marginals, built by
        # the ported `build_mixture_tilt`; the engine then runs at tau=1.0 because the
        # temperature is already inside each component.
        comp_paths = [C.TABLES_KZ / f"logc_{t}.npy" for t in mix_types]
        logc_list = []
        for cp in comp_paths:
            arr = np.load(cp, allow_pickle=False)
            if arr.shape != (dim, nmax + 1):
                raise SystemExit(f"{cp.name} shape {arr.shape} != {(dim, nmax + 1)}")
            logc_list.append(arr)
        prior = bank.prior.cpu().numpy().astype(np.float64)
        logpi, _log_qmix = build_mixture_tilt(prior, logc_list, mix_weights, component_tau)
        engine_tau = 1.0
        tilt_audit = {
            "tilt_table_kind": "built: per-gene mixture of tempered tilted marginals "
                               "(hvg/multicluster.build_mixture_tilt, ported verbatim)",
            "component_tilt_tables": [str(cp.resolve()) for cp in comp_paths],
            "component_tilt_sha256": [sha256_file(cp) for cp in comp_paths],
            "component_tau": component_tau,
            "component_tau_source": str(C.tilt_table(a.request).with_suffix(".json"))
                                    + " :: note (`--tau`)",
            "engine_tau": 1.0,
            "mixture_tilt_sha256": sha256_array(logpi),
            "mixture_tilt_min": float(logpi.min()), "mixture_tilt_max": float(logpi.max()),
            "request_table_not_used_as_proposal": str(logpi_path.resolve()),
        }
        print(f"[tilt]   mixture table: min {logpi.min():.3f} max {logpi.max():.3f}; "
              f"engine runs at tau=1.0 (tau={component_tau} already applied per component)",
              flush=True)

    torch_gen = torch.Generator(device=BASE.DEV).manual_seed(a.seed + 1)
    twist_gen = make_twist_generator(BASE.DEV, a.seed) if a.sampler == "twist" else None
    engine = HVGEngine(BASE, net, bank, logpi, engine_tau, clf, features, a.cell_chunk,
                       a.reward_floor, torch_gen, a.tilt_mode, sampler=a.sampler,
                       J=a.J, alpha=(a.alpha if a.sampler != "tilt_only" else 1.0),
                       twist_gen=twist_gen, reward_mode=a.reward_mode, tilt_on=tilt_on,
                       mixture=mixture)
    strat_log: list = []
    resample_fn = None
    if mix_targets is not None:
        resample_fn = make_stratified_resample(engine, mix_targets, strat_log,
                                               systematic_resample)
        print(f"[strat]  quota resampling ON, targets {mix_targets}; the in-chain resamples "
              "AND the final N-of-M selection both go through it", flush=True)
    trace = {} if a.ancestry_trace else None
    if a.sampler == "tilt_only":
        if trace is not None:
            raise SystemExit("--ancestry-trace is only wired for run_fk; the tilt_only arm "
                             "never resamples, so its ancestry is the identity by "
                             "construction (ticket 182 s3)")
        out, diag = run_tilt_only(engine, a.n_particles, a.n_out, a.seed, verbose=not a.quiet)
    else:
        out, diag = run_fk(engine, a.n_particles, a.n_out, a.seed, a.alpha, a.ess_frac,
                           a.resample_interval, not a.no_intermediate_resampling,
                           a.trigger, a.min_resample_gap, verbose=not a.quiet,
                           resample_fn=resample_fn, trace=trace)
    if tilt_audit:
        diag.update(tilt_audit)
    if mix_targets is not None:
        diag["stratified_quota"] = {"targets": mix_targets, "n_events": len(strat_log),
                                    "events": strat_log}
    diag.update(distribution_readings(out))
    diag.update({
        "experiment": a.experiment, "arm": a.arm, "role": a.role, "ticket": a.ticket,
        "stem": stem, "request": a.request, "seed": a.seed,
        "reward_model_kind": a.reward_model, "K": a.K, "NFE": nfe, "tau": a.tau,
        "tilt_mode": a.tilt_mode,
        "product_tilt_application": ("every_step" if a.tilt_mode == "every"
                                     else "last_reverse_step_only"),
        "horizon_used": horizon, "reward_floor": a.reward_floor,
        "cell_chunk": a.cell_chunk, "no_a4": True, "batched_bridge": True,
        "generator_retrained": False, "reward_model_retrained": False,
        "checkpoint": str(Path(ckpt).resolve()), "checkpoint_sha256": ckpt_sha,
        "tilt_on": tilt_on,
        # ---- ruling 164 section 4, debt 1: WHICH table was used, recorded in OUR json ----
        # `logpi` and `tilt_table` are the same file; both names are written because the frozen
        # code calls it logpi and the paper calls it the product tilt table.
        # ticket 170: in mixture mode the proposal is a BUILT table, so these four fields
        # describe it (a derived-array sha256 and a `built:` path), not the request's .npy --
        # recording the request file here would name a table the run never used.
        "logpi": (str(Path(logpi_path).resolve()) if mix_types is None
                  else "built: hvg.multicluster.build_mixture_tilt"),
        "logpi_path": (str(Path(logpi_path).resolve()) if mix_types is None
                       else "built: hvg.multicluster.build_mixture_tilt"),
        "logpi_sha256": (sha256_file(logpi_path) if mix_types is None
                         else sha256_array(logpi)),
        "tilt_table_path": (str(Path(logpi_path).resolve()) if mix_types is None
                            else "built: hvg.multicluster.build_mixture_tilt"),
        "tilt_table_sha256": (sha256_file(logpi_path) if mix_types is None
                              else sha256_array(logpi)),
        # ---- ruling 164 section 4, debt 2: which purity ruler this run is on (gate G5) ----
        **celltypist_identity(),
        "reward_validation_auc": reward_auc, "kernel_bank_gate": gate,
        "dtype_network": "float32", "dtype_kernel_bank": "float64", "tf32": False,
        "device": str(BASE.DEV),
        "twist_seed": (a.seed + C.TWIST_SEED_OFFSET) if a.sampler == "twist" else None,
        "transition_seed": a.seed + 1,
        "gene_hash": meta["gene_hash"],
        # the exact command line, so a gate can re-execute this run without guessing (G2b)
        "_rerun_argv": ["-m", "experiments.run_one"] + list(sys.argv[1:]),
        "walltime_s": time.time() - started,
        "peak_alloc_gb": (float(torch.cuda.max_memory_allocated() / 2 ** 30)
                          if torch.cuda.is_available() else 0.0),
        **reward_audit,
    })
    write_experiment, write_arm = a.experiment, a.arm
    if trace is not None:
        # never write into the archived directories the paper tables read from
        write_experiment = "ancestry_trace"
        diag.update({"experiment": write_experiment, "source_experiment": a.experiment,
                     "ancestry_trace": True,
                     "ancestry_trace_note": "observation-only rerun of the archived run; the "
                                            "delivered array must be bit-identical to it"})
    pj = write_run(stem, write_experiment, write_arm, out, diag)
    if trace is not None:
        anc = np.stack(trace["anc"]).astype(np.int32)
        npz = pj.with_name(f"{stem}_ancestry.npz")
        np.savez_compressed(
            npz, anc=anc,
            unique_per_step=np.asarray(trace["unique_per_step"], dtype=np.int64),
            step=np.asarray(trace["step"], dtype=np.int64),
            t=np.asarray(trace["t"], dtype=np.float64),
            delivered_idx=trace["delivered_idx"],
            resample_steps=np.asarray(diag["resample_steps"], dtype=np.int64))
        print(f"[anc] {npz}  anc{anc.shape} delivered_idx{trace['delivered_idx'].shape}")
    print(json.dumps({k: diag[k] for k in (
        "stem", "sampler", "reward_mode", "n_intermediate_resamples", "resample_steps",
        "ess_min_fraction", "ess_pre_final_fraction", "unique_lineage_fraction",
        "unique_cell_fraction", "telescope_max_abs_error", "mean", "max", "zero_fraction",
        "walltime_s")}, indent=1))
    print(f"[out] {pj}")
    if a.no_intermediate_resampling:
        err = diag["telescope_max_abs_error"]
        if err is None or err >= 1.0e-10:
            raise SystemExit("telescope gate FAILED")   # >= : aligned with analysis/gates.py
    return 0


if __name__ == "__main__":
    sys.exit(main())
