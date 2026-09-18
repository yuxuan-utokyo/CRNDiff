# -*- coding: utf-8 -*-
"""Loading the 2-D data, the frozen network, the tilt tables, the discriminators and the kernel chain.
Purpose: load the frozen 2D dataset, network, tilt tables and discriminators, and build and
cache the reverse-chain kernel matrices. Ported at function level from the 10-dimensional toy
package (the caliber, chain, product-x0 draw and model loading are unchanged; S=2, C=64, paths
moved into this repository). Added here: the request mechanism (requests come from a frozen
json), three families of discriminator, and the TabularOdds wrapper, whose rho is the ratio of
the closed-form request pmf to the histogram of a large tilted pool, wrapped as
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import (ASSETS, CF_MULT, DATA, EPS_T, F_BASE, K_REV, MODELS, REFERENCES,
                     build_net, draw_rows, feats, kmat, softmax_np, tgrid, trunc_pois_pmf)

DATA_STEM = "data2d_ring8_sym_v2"   # the configuration the paper uses (frozen)


def load_data():
    """Returns (npz dict, cfg dict). The dataset and its pmf tables are FROZEN inputs."""
    path = DATA / f"{DATA_STEM}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- the frozen 2-D dataset must sit in data/")
    data = np.load(path, allow_pickle=False)
    cfg = json.loads((DATA / f"{DATA_STEM}.json").read_text(encoding="utf-8"))
    return data, cfg


def load_pmf():
    """Closed-form component pmf tables [10, C, C] + the mixture pmf [C, C]."""
    z = np.load(DATA / f"{DATA_STEM}_pmf.npz", allow_pickle=False)
    return z["components"], z["mixture"]


def caliber(data, cfg):
    x = data["X_train"].astype(np.int64)
    mu = x.mean(0)
    var = x.var(0)
    kappa2 = float((var - mu).max())
    if kappa2 <= EPS_T:
        raise RuntimeError(f"non-positive spectral horizon: kappa2={kappa2:g}")
    terminal = 0.5 * math.log(kappa2 / EPS_T)
    ts = tgrid(terminal, K_REV, F_BASE)
    return mu, kappa2, terminal, ts, CF_MULT * int(cfg["C"])


def model_path_for(path: str | None) -> Path:
    """Resolve the checkpoint WITHOUT importing torch. Explicit path wins (bare names resolve
    under assets/models, '.pt' appended); with no path there must be EXACTLY ONE non-smoke
    net2d_v1_*.pt under assets/models -- never an mtime choice."""
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
        q = MODELS / (str(path) if str(path).endswith(".pt") else f"{path}.pt")
        if q.exists():
            return q
        raise SystemExit(f"model checkpoint not found: {path}")
    candidates = sorted(p for p in MODELS.glob("net2d_v1_*.pt") if "smoke" not in p.name)
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one non-smoke net2d_v1_*.pt under {MODELS}, "
                         f"found {len(candidates)} -- pass --model explicitly")
    return candidates[0]


def load_model(path: str | None, device: str):
    import torch

    model_path = model_path_for(path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    net = build_net(cfg["S"], cfg["C"], cfg["hidden"], seed=0, onehot=cfg.get("onehot", 0))
    net.load_state_dict(checkpoint["ema"])
    net.to(device).eval()
    return net, cfg, model_path


class Chain2D:
    """Precomputed Kcum / Kstep on the extended grid CF (Kcum rows only up to the data support
    C-1) plus the semigroup assertion. Cached on disk under assets/models by caliber
    (mu, ts, C, CF); the assertion is re-run on a cache hit."""

    def __init__(self, mu, ts, C, CF):
        self.C, self.CF, self.ts = C, CF, ts
        S = len(mu)
        K = len(ts) - 1
        key = f"{hash((tuple(np.round(mu, 9)), tuple(np.round(ts, 9)), C, CF)) & 0xffffffff:08x}"
        cache = MODELS / f"chain_C{C}_CF{CF}_K{K}_{key}.npz"
        if cache.exists():
            z = np.load(cache)
            self.Kcum, self.Kstep = z["Kcum"], z["Kstep"]
        else:
            self.Kcum = np.empty((K + 1, S, C, CF))
            self.Kstep = np.empty((K + 1, S, CF, CF))
            eyeC = np.eye(CF)[:C]
            for g in range(S):
                self.Kcum[0, g] = eyeC
                self.Kstep[0, g] = np.eye(CF)
                for k in range(1, K + 1):
                    self.Kstep[k, g] = kmat(float(ts[k] - ts[k - 1]), float(mu[g]), CF)
                    self.Kcum[k, g] = kmat(float(ts[k]), float(mu[g]), CF)[:C]
            MODELS.mkdir(parents=True, exist_ok=True)
            np.savez(cache, Kcum=self.Kcum, Kstep=self.Kstep)
        worst = 0.0
        for g in range(S):
            for k in range(1, K + 1):
                err = float(np.abs(self.Kcum[k - 1, g] @ self.Kstep[k, g] - self.Kcum[k, g]).max())
                worst = max(worst, err)
        if worst > 1e-9:
            raise SystemExit(f"chain semigroup consistency FAILED: {worst:.2e}")
        self.semigroup_err = worst
        self.stat = np.stack([trunc_pois_pmf(float(m), CF) for m in mu])
        self.cache_path = cache


def product_x0_draw(rng, ltil):
    """Draw n0 dim by dim (g = 0..S-1). Shared by every path; the anchor of bitwise determinism."""
    N, S, C = ltil.shape
    n0 = np.empty((N, S), dtype=np.int64)
    for g in range(S):
        n0[:, g] = draw_rows(rng, softmax_np(ltil[:, g, :]))
    return n0


# --------------------------------------------------------------------- requests
def request_names(cfg) -> list[str]:
    return list(cfg["requests"].keys())


def request_weight_vector(cfg, request: str) -> np.ndarray:
    """The request's composition as a vector over the 10 components (json order).
    The pseudo-request 'unconditional' is the data law itself (D priors) -- used by gate G3."""
    comps = list(cfg["components"])
    w = np.zeros(len(comps), dtype=np.float64)
    if request == "unconditional":
        return np.asarray(cfg["priors"], dtype=np.float64)
    req = cfg["requests"][request]
    for name, weight in zip(req["components"], req["weights"]):
        w[comps.index(name)] = float(weight)
    return w


def theta_for(request: str) -> np.ndarray:
    """The fitted tilt table [S, C] for a request; zeros for 'unconditional' / tilt-off."""
    if request == "unconditional":
        raise SystemExit("theta_for('unconditional') is undefined -- use zeros (tilt off)")
    p = MODELS / "tilts_v1.npz"
    if not p.exists():
        raise SystemExit(f"{p} not found -- run tools/fit_tilts.py first")
    z = np.load(p, allow_pickle=False)
    names = [str(s) for s in z["requests"]]
    if request not in names:
        raise SystemExit(f"no tilt table for request {request!r} in {p.name} (have {names})")
    return z["theta"][names.index(request)]


# ----------------------------------------------------------------- discriminators
class TabularOdds:
    """The 'tabular' reward family: rho(x) = closed-form request pmf / smoothed histogram of a
    large TILTED proposal pool (an approximate oracle for the deployed arm). Wrapped as a
    classifier-like object with predict_proba[:, 1] = rho / (1 + rho), so the sampler's
    odds-mode log_reward returns exactly log rho (within the +/- log((1-floor)/floor) guard).
    Not a fitted classifier; no training data beyond the pool histogram."""

    def __init__(self, log_rho_table: np.ndarray):
        self.log_rho_table = np.asarray(log_rho_table, dtype=np.float64)

    def predict_proba(self, x) -> np.ndarray:
        xi = np.asarray(x).astype(np.int64)
        lr = self.log_rho_table[xi[:, 0], xi[:, 1]]
        # saturate before the sigmoid: table minima reach ~-677 (out-of-support cells) and a
        # sparser pool could pass exp's overflow point (~709); the sampler floors the result
        # at +/-log((1-floor)/floor) ~ 18.4 anyway, so this cannot change any output.
        # NOTE the suppression side of that floor DOES engage for out-of-support states (by
        # design of reward_floor); only the amplification side (log_rho_max ~ 6) is clip-free.
        lr = np.clip(lr, -700.0, 700.0)
        p1 = 1.0 / (1.0 + np.exp(-lr))                # sigmoid(log rho) = rho / (1 + rho)
        return np.stack([1.0 - p1, p1], axis=1)


def reward_model_for(discriminators, request: str, family: str):
    """Every arm must use the reward fitted against the proposal it actually runs:
      'proposal_residual'  gbt fitted against the request's TILTED chain pool (seed 777001)
      'proposal_untilted'  gbt fitted against the UNTILTED chain pool (seed 777002, shared)
      'tabular'            TabularOdds (approximate oracle; tilt-on arms only)."""
    entry = discriminators["requests"][request]
    if family == "proposal_residual":
        return entry["proposal_residual"]["gbt"]
    if family == "proposal_untilted":
        return entry["proposal_untilted"]["gbt"]
    if family == "tabular":
        return TabularOdds(entry["tabular"]["log_rho_table"])
    raise ValueError(family)


def load_everything(device: str, model: str | None = None):
    """One call that returns everything a run needs. Returns a dict."""
    data, cfg = load_data()
    mu, kappa2, terminal, ts, CF = caliber(data, cfg)
    net, model_cfg, model_path = load_model(model, device)
    chain = Chain2D(mu, ts, int(cfg["C"]), CF)
    pmf_components, pmf_mixture = load_pmf()
    disc_path = MODELS / "discriminators_v1.joblib"
    discriminators = None
    if disc_path.exists():
        import joblib
        discriminators = joblib.load(disc_path)
    return dict(data=data, cfg=cfg, mu=mu, kappa2=kappa2, terminal=terminal, ts=ts, CF=CF,
                net=net, model_cfg=model_cfg, model_path=model_path, chain=chain,
                pmf_components=pmf_components, pmf_mixture=pmf_mixture,
                discriminators=discriminators)
