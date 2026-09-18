"""Inference-time Feynman--Kac steering for the frozen HVG CRN sampler (v2).

WHAT THIS FILE IS
-----------------
A verbatim copy of `fk_hvg.py` plus exactly two behavioural knobs.
`fk_hvg.py` is the reproduction path of the results already reported in
`FK_RESULTS_20260817.md` and is therefore never edited.

WHY v2 EXISTS (the defect it repairs)
-------------------------------------
`fk_hvg.py` applies the product tilt `exp(tau * log pi)` only at the LAST
reverse step -- its own JSON says `last_reverse_step_only` -- yet the report
claimed the proposal matched "the verified production proposal".  It does
not.  The production CONDFULL gamma=0.4 arm is produced by
`condition/scan_gamma.py`, whose sampler command never passes `--last-only`,
so `sample_hvg2k_noa4.py` multiplies `p_net` by `pi^tau` and renormalises at
EVERY reverse step.  `--last-only` is the PPC control recipe, and
`Experiments/X/_claude/out/fd_basecal.json` shows it is close to a
distributional no-op (FD 4.4419 vs 4.4648 for no tilt at all).  The v1 FK arm
was therefore "unconditional proposal + classifier potential", not a
controlled comparison against the reported "Product proposal" row.

The tilt is an EXACT reparameterisation, not an approximation:
    p_used  ~  p_net * pi^tau,   p_net = softmax(logits)
            =  softmax(logits + tau * log pi)
(see the LA/PPC comment block in `sample_hvg2k_noa4.py`).

CHANGE 1 -- `--tilt-mode {last,every}` (default `every`)
    `every` multiplies by `exp(tau*log pi)` and renormalises at every reverse
    step, i.e. the production semantics.  `last` reproduces `fk_hvg.py`
    exactly and is kept as the proposal ablation.

CHANGE 2 -- `--trigger {scheduled,any}` (default `scheduled`)
    `scheduled` = the original behaviour: ESS is only *acted on* at steps
    where `(step+1) % resample_interval == 0`.  `any` checks ESS at every
    reverse step and resamples whenever it drops below the threshold, subject
    to `--min-resample-gap` steps of hysteresis (anti-chatter).  This exists
    to treat the lineage collapse seen in the v1 main runs (unique_lineage
    0.185 Endothelial / 0.060 Myeloid).

PITFALLS ALREADY PAID FOR
-------------------------
* Two-pass vs fused pass does NOT change the RNG stream.  The reward-only
  pass (`do_transition=False`) draws nothing from the torch generator, so
  `pass_all(...,True,False)` then `pass_all(...,False,True)` produces the same
  transition samples as the fused `pass_all(...,True,True)`.  That is why
  `--trigger any` -- which must two-pass every step in order to be able to
  resample between reward and transition -- stays comparable; it only doubles
  the generator forward passes.
* The reward is accumulated at EVERY step under both triggers.  Only the
  *resampling* is gated.  Do not "fix" this by gating the potential too: the
  telescoping terminal correction depends on the running difference.
* `sample_hvg2k_noa4.py` records that per-step PPC raised cross-gene
  correlation error by ~17% on the unconditional arm.  Do not be surprised if
  `every` trades correlation for first moment; report it, do not tune it away.
* Output filenames must be unique or the run refuses to write; every knob
  that moves a number is therefore in the stem.

This file is an independent wrapper: it does not modify or retrain the CRN
generator, and it does not edit the verified baseline sampling path.  dtype
caliber is inherited bit-for-bit: network float32, kernel bank float64, tf32
off.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
COND = HERE.parent
CRNDIFF = HERE.parents[3]
X_ROOT = CRNDIFF / "Experiments" / "X"
BASE_CODE = X_ROOT / "Baseline" / "OURS" / "code"
sys.path.insert(0, str(BASE_CODE))

import sample_hvg2k_noa4 as BASE  # noqa: E402


DEFAULT_CKPT = X_ROOT / "Baseline" / "OURS" / "models" / "pilot_run1c_s20260625.pt"
# These are only fallbacks for `--alpha` when it is not passed explicitly; every
# arm in this round passes `--alpha` from a blind pre-registered rule, so the
# values here never enter a reported number.  Mesothelial is added so the
# coverage sweep can address it; its `tables_kz/logc_Mesothelial.npy` and clean
# reward both exist.
DEFAULT_ALPHA = {"Endothelial": 1.0, "Myeloid": 0.95, "Neuronal": 0.5,
                 "Mesothelial": 1.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized(logw: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.asarray(logw, dtype=np.float64)
    z = z - float(z.max())
    w = np.exp(z)
    w /= w.sum()
    return w, 1.0 / float(np.square(w).sum())


def systematic_resample(rng: np.random.Generator, weight: np.ndarray,
                        n_out: int | None = None) -> np.ndarray:
    n = len(weight) if n_out is None else int(n_out)
    cdf = np.cumsum(np.asarray(weight, dtype=np.float64))
    cdf[-1] = 1.0
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    return np.searchsorted(cdf, positions, side="right").clip(0, len(weight) - 1)


class PortableHGBT:
    """Inference-only binary HistGradientBoosting tree ensemble.

    PITFALL PAID FOR (2026-08-18, cost one crashed run).  `fk_hvg.py` traverses
    by slicing the structured `nodes` array and then reading field VIEWS
    (`node["feature_idx"]` etc.).  In this environment that form intermittently
    corrupts memory and kills the process with an access violation
    (0xC0000005 / SIGSEGV) part-way through a long run -- the B-2 grid's first
    arm died at reverse step ~5 after 80 s, and the same reader segfaulted
    repeatedly in `diag_reward_axis.py`.  It is load-dependent, so the P3 arms
    got through by luck.

    The fix is to materialise every field as a plain contiguous array once at
    load time and index those.  It is arithmetically the same traversal, and it
    is verified **bitwise identical** to `fk_hvg.py`'s implementation on both
    frozen classifiers (`max abs diff = 0.0`, `np.array_equal` True), so the
    already-reported P3 numbers stand unchanged.  `fk_hvg.py` itself is a
    frozen reproduction path and is NOT edited.
    """

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        nodes = z["nodes"]
        self.is_leaf = np.ascontiguousarray(nodes["is_leaf"])
        self.feature_idx = np.ascontiguousarray(nodes["feature_idx"]).astype(np.intp)
        self.num_threshold = np.ascontiguousarray(nodes["num_threshold"]).astype(np.float64)
        self.missing_left = np.ascontiguousarray(nodes["missing_go_to_left"])
        self.left = np.ascontiguousarray(nodes["left"]).astype(np.int64)
        self.right = np.ascontiguousarray(nodes["right"]).astype(np.int64)
        self.leaf_value = np.ascontiguousarray(nodes["value"]).astype(np.float64)
        self.offsets = np.ascontiguousarray(z["offsets"]).astype(np.int64)
        self.baseline = float(z["baseline"][0])
        self.n_features = int(z["n_features"][0])
        self.source_sha256 = str(z["source_sha256"][0])
        self.identity_max_abs_error = float(z["identity_max_abs_error"][0])
        del nodes, z

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.n_features:
            raise ValueError(f"classifier expects {self.n_features} features, got {x.shape[1]}")
        raw = np.full(len(x), self.baseline, dtype=np.float64)
        rows = np.arange(len(x))
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            s, e = int(start), int(end)
            index = np.zeros(len(x), dtype=np.int64)          # node id within tree
            for _ in range(e - s):
                gi = index + s                                 # global node id
                active = self.is_leaf[gi] == 0
                if not np.any(active):
                    break
                rr = rows[active]
                ga = gi[active]
                value = x[rr, self.feature_idx[ga]]
                go_left = np.where(
                    np.isnan(value), self.missing_left[ga] != 0,
                    value <= self.num_threshold[ga],
                )
                index[active] = np.where(go_left, self.left[ga], self.right[ga])
            else:
                raise RuntimeError("portable tree traversal did not reach leaves")
            raw += self.leaf_value[index + s]
        return np.where(
            raw >= 0, 1.0 / (1.0 + np.exp(-raw)),
            np.exp(raw) / (1.0 + np.exp(raw)),
        )


class CleanLogisticReward:
    """Portable smooth condition classifier fitted only on labeled real cells."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.features = z["features"].astype(np.int64)
        self.mean = z["scaler_mean"].astype(np.float64)
        self.scale = z["scaler_scale"].astype(np.float64)
        self.coef = z["coef"].astype(np.float64)
        self.intercept = float(z["intercept"][0])
        self.n_features = len(self.features)

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.n_features:
            raise ValueError(f"classifier expects {self.n_features} features, got {x.shape[1]}")
        raw = ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.coef
        raw += self.intercept
        return np.where(
            raw >= 0, 1.0 / (1.0 + np.exp(-raw)),
            np.exp(raw) / (1.0 + np.exp(raw)),
        )


def classifier_log_reward(clf, features: np.ndarray, x: np.ndarray,
                          floor: float) -> np.ndarray:
    """Bounded clean reward log P(target-like | x), with fit-time normalization."""
    z = np.asarray(x, dtype=np.float32)
    lib = z.sum(axis=1, dtype=np.float64)
    scale = (1.0e4 / np.maximum(lib, 1.0)).astype(np.float32)
    xf = np.log1p(z[:, features] * scale[:, None]).astype(np.float32, copy=False)
    p = np.asarray(clf.predict_probability(xf), dtype=np.float64)
    return np.log(np.clip(p, floor, 1.0))


class HVGFK:
    def __init__(self, net, bank, logpi: np.ndarray, tau: float, clf,
                 features: np.ndarray, cell_chunk: int, reward_floor: float,
                 torch_gen: torch.Generator, tilt_mode: str = "every"):
        if tilt_mode not in ("last", "every"):
            raise ValueError("tilt_mode must be 'last' or 'every'")
        self.tilt_mode = tilt_mode
        self.net = net
        self.bank = bank
        self.dev = bank.device
        self.logpi = torch.as_tensor(
            np.exp(float(tau) * np.asarray(logpi, dtype=np.float64)),
            dtype=torch.float32,
            device=self.dev,
        )
        self.clf = clf
        self.features = np.asarray(features, dtype=np.int64)
        self.cell_chunk = int(cell_chunk)
        self.reward_floor = float(reward_floor)
        self.torch_gen = torch_gen
        self.cgrid = torch.arange(bank.prior.shape[1], dtype=torch.float32,
                                  device=self.dev)

    def _posterior(self, state: torch.Tensor, step: int) -> torch.Tensor:
        b = state.shape[0]
        t = float(self.bank.tgrid[step])
        p = self.net.post(
            state.to(torch.float32), torch.full((b,), t, device=self.dev)
        )
        # Product tilt.  `every` == the verified production proposal
        # (`sample_hvg2k_noa4.py` without `--last-only`): multiply by pi^tau
        # and renormalise at every reverse step.  `last` == the v1 behaviour,
        # kept only as the proposal ablation.  No A4 truncation either way.
        if self.tilt_mode == "every" or step == len(self.bank.tgrid) - 2:
            p = p * self.logpi[None]
            p = p / p.sum(-1, keepdim=True).clamp_min(1.0e-30)
        s = p.sum(-1, keepdim=True)
        return torch.where(s > 0, p / torch.clamp_min(s, 1.0e-30), p)

    def _transition(self, p: torch.Tensor, state: torch.Tensor,
                    step: int) -> torch.Tensor:
        b, dim, c = p.shape
        n0 = BASE._cat(p.reshape(b * dim, c), self.torch_gen).view(b, dim)
        del p
        knext = self.bank.Kgrid[step + 1]
        kdelt = self.bank.Kdelta_T[step]
        vi_next = BASE._slot_v_idx(knext, self.dev)[None, :].expand(b, dim)
        vi_delt = BASE._slot_v_idx(kdelt, self.dev)[None, :].expand(b, dim)
        row = knext.U[vi_next, n0]
        row.mul_(kdelt.U[vi_delt, state.clamp(0, c - 1)])
        out = BASE._cat(row.reshape(b * dim, c), self.torch_gen).view(b, dim)
        del row, n0
        return out

    @torch.inference_mode()
    def pass_all(self, state: torch.Tensor, step: int, need_reward: bool,
                 do_transition: bool) -> tuple[np.ndarray | None, torch.Tensor | None]:
        n = len(state)
        reward = np.empty(n, dtype=np.float64) if need_reward else None
        new_state = torch.empty_like(state) if do_transition else None
        for lo in range(0, n, self.cell_chunk):
            hi = min(lo + self.cell_chunk, n)
            chunk = state[lo:hi]
            p = self._posterior(chunk, step)
            if need_reward:
                xhat = torch.sum(p * self.cgrid[None, None, :], dim=-1)
                reward[lo:hi] = classifier_log_reward(
                    self.clf, self.features, xhat.cpu().numpy(), self.reward_floor
                )
                del xhat
            if do_transition:
                new_state[lo:hi] = self._transition(p, chunk, step)
            else:
                del p
        return reward, new_state

    def final_reward(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float64)
        for lo in range(0, len(x), self.cell_chunk):
            hi = min(lo + self.cell_chunk, len(x))
            out[lo:hi] = classifier_log_reward(
                self.clf, self.features, x[lo:hi], self.reward_floor
            )
        return out


def initial_stationary(bank, n: int, rng: np.random.Generator) -> torch.Tensor:
    prior = bank.prior.cpu().numpy()
    cdf = np.cumsum(prior, axis=1)
    dim = bank.dim
    x = np.empty((n, dim), dtype=np.int64)
    for d in range(dim):
        x[:, d] = np.searchsorted(
            cdf[d], rng.random(n) * cdf[d, -1], side="right"
        ).clip(0, prior.shape[1] - 1)
    return torch.as_tensor(x, dtype=torch.long, device=bank.device)


def run_fk(engine: HVGFK, n_particles: int, n_out: int, seed: int, alpha: float,
           intermediate_scale: float, ess_frac: float, resample_interval: int,
           enable_intermediate_resampling: bool, trigger: str = "scheduled",
           min_resample_gap: int = 2) -> tuple[np.ndarray, dict]:
    if trigger not in ("scheduled", "any"):
        raise ValueError("trigger must be 'scheduled' or 'any'")
    rng = np.random.default_rng(seed)
    state = initial_stationary(engine.bank, n_particles, rng)
    lineage = np.arange(n_particles, dtype=np.int64)
    logw = np.zeros(n_particles, dtype=np.float64)
    prev_value = np.zeros(n_particles, dtype=np.float64)
    ess_trace: list[dict] = []
    resample_steps: list[int] = []
    generator_evals = 0
    reward_evals = 0
    k_steps = len(engine.bank.tgrid) - 1

    last_resample_step = -10 ** 9
    for step in range(k_steps):
        need_reward = step > 0
        # `scheduled` decides only whether this step gets the two-pass
        # structure that allows a resample between reward and transition.
        # The potential itself is accumulated at every step in both branches.
        if trigger == "any":
            scheduled = need_reward and enable_intermediate_resampling
        else:
            scheduled = (
                need_reward
                and enable_intermediate_resampling
                and (step + 1) % resample_interval == 0
            )
        if scheduled:
            reward, _ = engine.pass_all(state, step, True, False)
            reward_evals += 1
            current_value = alpha * intermediate_scale * reward
            logw += current_value - prev_value
            weight, ess = normalized(logw)
            ess_trace.append({
                "step": step,
                "t": float(engine.bank.tgrid[step]),
                "ess_fraction": ess / n_particles,
                "scheduled": True,
            })
            gap_ok = (step - last_resample_step) >= min_resample_gap
            if ess < ess_frac * n_particles and gap_ok:
                idx = systematic_resample(rng, weight)
                index_t = torch.as_tensor(idx, dtype=torch.long, device=engine.dev)
                state = state.index_select(0, index_t)
                current_value = current_value[idx]
                lineage = lineage[idx]
                logw.fill(0.0)
                resample_steps.append(step)
                last_resample_step = step
            prev_value = current_value
            _, new_state = engine.pass_all(state, step, False, True)
            generator_evals += 2
        else:
            reward, new_state = engine.pass_all(state, step, need_reward, True)
            generator_evals += 1
            if need_reward:
                reward_evals += 1
                current_value = alpha * intermediate_scale * reward
                logw += current_value - prev_value
                weight, ess = normalized(logw)
                ess_trace.append({
                    "step": step,
                    "t": float(engine.bank.tgrid[step]),
                    "ess_fraction": ess / n_particles,
                    "scheduled": False,
                })
                prev_value = current_value
        del state
        state = new_state
        if step == 0 or (step + 1) % 4 == 0 or step + 1 == k_steps:
            msg = f"[FK] reverse {step + 1:02d}/{k_steps}"
            if ess_trace:
                msg += f"  ESS/N={ess_trace[-1]['ess_fraction']:.4f}"
            if resample_steps and resample_steps[-1] == step:
                msg += "  RESAMPLED"
            print(msg, flush=True)

    x0 = state.cpu().numpy().astype(np.int16, copy=False)
    del state
    final_reward = engine.final_reward(x0)
    logw += alpha * final_reward - prev_value
    final_weight, final_ess = normalized(logw)
    ess_trace.append({"step": k_steps, "t": 0.0,
                      "ess_fraction": final_ess / n_particles,
                      "scheduled": False})
    telescope_error = None
    if not resample_steps:
        telescope_error = float(np.max(np.abs(logw - alpha * final_reward)))

    idx = systematic_resample(rng, final_weight, n_out)
    out = x0[idx]
    lineage = lineage[idx]
    diag = {
        "method": "sequential_Feynman_Kac_difference_potential",
        "proposal": "frozen_product_tilted_CRN_reverse_kernel",
        "reward": "frozen_clean_classifier_log_probability",
        "resample_trigger": trigger,
        "min_resample_gap": int(min_resample_gap),
        "n_particles": n_particles,
        "n_out": n_out,
        "alpha": alpha,
        "intermediate_scale": intermediate_scale,
        "ess_threshold_fraction": ess_frac,
        "resample_interval": resample_interval,
        "intermediate_resampling": enable_intermediate_resampling,
        "n_intermediate_resamples": len(resample_steps),
        "resample_steps": resample_steps,
        "ess_trace": ess_trace,
        "ess_min_fraction": float(min(v["ess_fraction"] for v in ess_trace)),
        "ess_pre_final_fraction": float(final_ess / n_particles),
        "unique_lineage_fraction": float(np.unique(lineage).size / n_out),
        "unique_cell_fraction": float(np.unique(out, axis=0).shape[0] / n_out),
        "telescope_max_abs_error": telescope_error,
        "generator_forward_passes_per_particle": generator_evals,
        "reward_evaluations_per_particle": reward_evals + 1,
    }
    return out, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=tuple(DEFAULT_ALPHA), default="Endothelial")
    ap.add_argument("--reward-model", choices=("clean", "residual"), default="clean")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--logpi", default=None)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--n-particles", type=int, default=5000)
    ap.add_argument("--n-out", type=int, default=2500)
    ap.add_argument("--cell-chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--intermediate-scale", type=float, default=0.5)
    ap.add_argument("--ess-frac", type=float, default=0.5)
    ap.add_argument("--resample-interval", type=int, default=8)
    ap.add_argument("--reward-floor", type=float, default=1.0e-8)
    ap.add_argument("--no-intermediate-resampling", action="store_true")
    ap.add_argument("--tilt-mode", choices=("last", "every"), default="every",
                    help="every = production semantics (pi^tau at every "
                         "reverse step); last = fk_hvg.py v1 behaviour")
    ap.add_argument("--trigger", choices=("scheduled", "any"),
                    default="scheduled",
                    help="scheduled = act on ESS only at interval "
                         "checkpoints; any = act at any step below threshold")
    ap.add_argument("--min-resample-gap", type=int, default=2,
                    help="minimum reverse steps between two resamples "
                         "(anti-chatter; only bites for --trigger any)")
    # Added 2026-08-18 pm.  Lets a residual discriminator refitted against a
    # DIFFERENT proposal be used without touching the frozen out_<target>/
    # directory.  Default reproduces the previous behaviour exactly, so every
    # earlier invocation is unaffected.
    # Added 2026-09-05 (ticket 159 / E52).  The reverse chain has always
    # started at `BASE.M.horizon()` = CFG.T_O.  E52 asks what is lost by
    # stopping there instead of at a conservatively long horizon, which needs
    # a checkpoint TRAINED on the longer interval and a grid built on it.
    # `default=None` keeps the deployed path character-for-character identical;
    # the value, when given, must equal the checkpoint's own training T_MAX so
    # that a model trained on [F_BASE, T] is never extrapolated past T.
    ap.add_argument("--horizon", type=float, default=None,
                    help="reverse-chain horizon T. Default None = "
                         "BASE.M.horizon() (the deployed T_O path, unchanged). "
                         "When given it must match the checkpoint json's "
                         "T_MAX to within 1e-6.")
    ap.add_argument("--residual-dir", default=None,
                    help="directory holding residual_classifier{,_portable}.* , "
                         "residual_log.json and split_indices.npz "
                         "(default: out_<target>)")
    a = ap.parse_args()

    if a.n_out <= 0 or a.n_out > a.n_particles:
        raise SystemExit("--n-out must be in 1..n-particles")
    if a.cell_chunk <= 0 or a.resample_interval <= 0:
        raise SystemExit("--cell-chunk and --resample-interval must be positive")
    if a.min_resample_gap < 1:
        raise SystemExit("--min-resample-gap must be at least 1")
    alpha = DEFAULT_ALPHA[a.target] if a.alpha is None else float(a.alpha)
    ckpt = Path(a.ckpt)
    logpi_path = Path(a.logpi) if a.logpi else COND / "tables_kz" / f"logc_{a.target}.npy"
    model_dir = Path(a.residual_dir) if a.residual_dir else HERE / f"out_{a.target}"
    if a.reward_model == "residual":
        split_path = model_dir / "split_indices.npz"
        clf_source_path = model_dir / "residual_classifier.joblib"
        clf_path = model_dir / "residual_classifier_portable.npz"
        reward_log_path = model_dir / "residual_log.json"
        required_reward = (split_path, clf_source_path, clf_path, reward_log_path)
    else:
        clf_path = HERE / "clean_reward" / f"{a.target}_clean_logistic.npz"
        reward_log_path = HERE / "clean_reward" / f"{a.target}_clean_logistic.json"
        required_reward = (clf_path, reward_log_path)
    for path in (ckpt, logpi_path, *required_reward):
        if not path.exists():
            raise SystemExit(f"missing required frozen input: {path}")

    reward_meta = json.loads(reward_log_path.read_text(encoding="utf-8"))
    if a.reward_model == "residual":
        if not reward_meta["classifier"]["gate_passed"]:
            raise SystemExit("held-out residual classifier AUC gate did not pass")
        split = np.load(split_path, allow_pickle=False)
        features = split["classifier_features"]
        clf = PortableHGBT(clf_path)
        if clf.source_sha256 != sha256_file(clf_source_path):
            raise SystemExit("portable classifier source hash does not match frozen joblib")
        if clf.identity_max_abs_error > 1.0e-12:
            raise SystemExit("portable classifier identity gate did not pass")
        reward_auc = reward_meta["classifier"]["heldout_roc_auc"]
        reward_audit = {
            "residual_dir": str(model_dir.resolve()),
            "residual_classifier_source": str(clf_source_path.resolve()),
            "residual_classifier_portable": str(clf_path.resolve()),
            "portable_classifier_identity_max_abs_error": clf.identity_max_abs_error,
        }
    else:
        if not reward_meta["validation"]["gate_passed"]:
            raise SystemExit("clean reward validation gate did not pass")
        clf = CleanLogisticReward(clf_path)
        features = clf.features
        reward_auc = reward_meta["validation"]["roc_auc"]
        reward_audit = {
            "clean_reward_model": str(clf_path.resolve()),
            "clean_reward_uses_generated_cells_for_fit": False,
        }

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()

    data, meta = BASE.M.load_data(("train",))
    train = data["train"]
    derived = BASE.M.derive(train)
    dim, nmax = train.shape[1], BASE.CFG.NMAX
    net = BASE.M.build_net(dim, derived["iscale"]).to(BASE.DEV)
    net.load_state_dict(torch.load(ckpt, map_location=BASE.DEV, weights_only=False))
    net.eval()
    horizon_default = float(BASE.M.horizon())
    if a.horizon is None:
        horizon_used, ckpt_t_max = horizon_default, None
    else:
        horizon_used = float(a.horizon)
        # the checkpoint's sidecar json is the only record of the interval the
        # network was actually trained on; `T` in that file is CFG.T_O and is
        # NOT the training bound (train_pilot.py writes both).
        name = ckpt.name
        meta_stem = name[:-3] if name.endswith(".pt") else ckpt.stem
        ckpt_meta = ckpt.parent / f"{meta_stem}.json"
        if not ckpt_meta.exists():
            raise SystemExit(f"--horizon given but no checkpoint json: {ckpt_meta}")
        blob = json.loads(ckpt_meta.read_text(encoding="utf-8"))
        if "T_MAX" not in blob:
            raise SystemExit(f"checkpoint json has no T_MAX field: {ckpt_meta}")
        ckpt_t_max = float(blob["T_MAX"])
        if abs(horizon_used - ckpt_t_max) > 1.0e-6:
            raise SystemExit(
                f"--horizon {horizon_used!r} does not match the checkpoint's "
                f"training T_MAX {ckpt_t_max!r} ({ckpt_meta}); sampling a "
                f"model outside the interval it was trained on is "
                f"extrapolation, not a horizon ablation")
    tgrid, nfe = BASE.M.build_grid(a.K, horizon_used)
    logfact = BASE.VO._logfact_arr(nmax)
    started = time.time()
    bank = BASE.DedupBank(
        derived["V"], tgrid, nmax,
        lambda tau, v, nm: BASE.VO.kmat(tau, v, nm, logfact),
        lambda v, nm: BASE.VO.pois_prior(v, nm, logfact), BASE.DEV,
    )
    gate = bank.verify(
        lambda tau, v, nm: BASE.VO.kmat(tau, v, nm, logfact), derived["V"]
    )
    if not gate["bit_exact"]:
        raise SystemExit("deduplicated kernel bank bit-exact gate failed")
    logpi = np.load(logpi_path, allow_pickle=False)
    if logpi.shape != (dim, nmax + 1):
        raise SystemExit(f"logpi shape {logpi.shape} != {(dim, nmax + 1)}")

    torch_gen = torch.Generator(device=BASE.DEV).manual_seed(a.seed + 1)
    engine = HVGFK(
        net, bank, logpi, a.tau, clf, features, a.cell_chunk,
        a.reward_floor, torch_gen, a.tilt_mode,
    )
    out, diag = run_fk(
        engine, a.n_particles, a.n_out, a.seed, alpha,
        a.intermediate_scale, a.ess_frac, a.resample_interval,
        not a.no_intermediate_resampling, a.trigger, a.min_resample_gap,
    )
    diag.update({
        "target": a.target,
        "reward_model_kind": a.reward_model,
        "seed": a.seed,
        "K": a.K,
        "horizon_used": horizon_used,
        "horizon_arg": a.horizon,
        "horizon_default_T_O": horizon_default,
        "ckpt_T_MAX": ckpt_t_max,
        "NFE": nfe,
        "tau": a.tau,
        "tilt_mode": a.tilt_mode,
        "product_tilt_application": (
            "every_step" if a.tilt_mode == "every" else "last_reverse_step_only"
        ),
        "production_proposal_equivalent": bool(a.tilt_mode == "every"),
        "resample_interval_arg": a.resample_interval,
        "no_a4": True,
        "batched_bridge": True,
        "cell_chunk": a.cell_chunk,
        "generator_retrained": False,
        "reward_model_retrained": False,
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": sha256_file(ckpt),
        "logpi": str(logpi_path.resolve()),
        "reward_validation_auc": reward_auc,
        "kernel_bank_gate": gate,
        "dtype_network": "float32",
        "dtype_kernel_bank": "float64",
        "tf32": False,
        "walltime_s": time.time() - started,
        "peak_alloc_gb": (
            float(torch.cuda.max_memory_allocated() / 2 ** 30)
            if torch.cuda.is_available() else 0.0
        ),
        "output_mean": float(out.mean()),
        "output_max": int(out.max()),
        "output_zero_fraction": float(np.mean(out == 0)),
        "gene_hash": meta["gene_hash"],
        **reward_audit,
    })

    mode = (
        "endpoint_gate" if a.no_intermediate_resampling
        else ("fk" if diag["n_intermediate_resamples"] > 0 else "fk_no_trigger")
    )
    sc = str(a.intermediate_scale).replace(".", "p")
    al = str(alpha).replace(".", "p")
    # Every knob that moves a number goes into the stem: tilt mode, trigger
    # and interval are all part of the caliber, and no v1 artefact may ever
    # be hit by a v2 name.
    ht = "" if a.horizon is None else "_T" + f"{horizon_used:.4f}".replace(".", "p")
    stem = (
        f"{a.target}_{a.reward_model}_fkv2{ht}_{mode}_tilt{a.tilt_mode}_"
        f"trig{a.trigger}_K{a.K}_M{a.n_particles}_N{a.n_out}_"
        f"alpha{al}_sc{sc}_int{a.resample_interval}_seed{a.seed}"
    )
    sample_path = HERE / f"{stem}.npy"
    report_path = HERE / f"{stem}.json"
    if sample_path.exists() or report_path.exists():
        raise SystemExit(f"refusing to overwrite existing FK output: {stem}")
    np.save(sample_path, out.astype(np.int16, copy=False))
    report_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: diag[k] for k in (
        "target", "tilt_mode", "product_tilt_application", "resample_trigger",
        "n_particles", "n_out", "n_intermediate_resamples",
        "resample_steps", "ess_min_fraction", "ess_pre_final_fraction",
        "unique_lineage_fraction", "unique_cell_fraction",
        "telescope_max_abs_error", "walltime_s", "peak_alloc_gb",
    )}, indent=2), flush=True)
    print(f"[out] {sample_path}", flush=True)
    print(f"[out] {report_path}", flush=True)

    if a.no_intermediate_resampling:
        err = diag["telescope_max_abs_error"]
        if err is None or err > 1.0e-10:
            raise SystemExit("FK terminal telescope gate failed")

    del engine, bank, net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
