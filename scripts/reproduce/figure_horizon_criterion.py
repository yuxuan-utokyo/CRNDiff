# -*- coding: utf-8 -*-
"""The horizon criterion and its consequence, in one dimension, by exact Bayes. No training, no
sampling.

Two panels:

(a) the criterion: the measured excess Fano against t, the Poisson null floor, and the horizon
    where the two meet. The only inputs are the two quantities measured at t = 0, and the
    horizon is COMPUTED from them rather than read off the curve.
(b) the consequence: the endpoint marginals of reverse chains started at a quarter, one and one
    and a half times that horizon, drawn as steps because counts are discrete and nothing is
    smoothed.

The two panels share their margins and are emitted at identical size, so neither may be scaled
by the document: the font sizes are the ones that appear on paper.

    python scripts/reproduce/figure_horizon_criterion.py
"""
import io, json, sys, pathlib
import numpy as np, matplotlib
from PIL import Image
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from horizon import reverse_kernel_core as C

OUT = pathlib.Path(__file__).resolve().parents[2] / "out" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NMAX, W, LO, HI, SEED = 400, 0.3, 4.0, 40.0, 20260909
N_DATA, K_STEPS, B_NULL, R_MEAS = 20000, 8, 600, 80
rng = np.random.default_rng(SEED)

q0 = C.q0_mixture(NMAX, W, LO, HI); k = np.arange(NMAX+1)
mu = float(k @ q0); pi = C.stationary(NMAX, mu)
sigma1 = float(((k**2 @ q0) - mu**2 - mu) / mu)

def sig_hat(x):
    m = x.mean(); return (x.var(ddof=1) - m) / m

null = np.array([abs(sig_hat(rng.poisson(mu, N_DATA))) for _ in range(B_NULL)])
sig_noise = float(np.percentile(null, 95))
null_lo, null_hi = np.percentile(null, [5, 95])
T_O = 0.5*np.log(sigma1/sig_noise)
print(f"sigma1={sigma1:.4f}  sigma_noise={sig_noise:.5f}  T_O={T_O:.4f}")

LAD=[0.25,1.0,1.5]
t_meas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 3.3, 3.8, 4.4, 5.0, 5.6])
med,q25,q75 = [],[],[]
for t in t_meas:
    qt = q0 if t == 0 else q0 @ C.kernel(NMAX, mu, float(t))
    v = np.array([abs(sig_hat(rng.choice(k, N_DATA, p=qt))) for _ in range(R_MEAS)])
    med.append(np.median(v)); q25.append(np.percentile(v,25)); q75.append(np.percentile(v,75))
med,q25,q75 = map(np.array,(med,q25,q75))

laws={f: C.reverse_chain(q0,mu,f*T_O,K_STEPS,pi) for f in LAD}
tvs={f: C.tv(laws[f],q0) for f in LAD}
for f in LAD: print(f"  T={f:>4}xT_O  TV={tvs[f]:.4e}")

INK,GRID="#22222a","#d7d4dc"; BLU,RED="#1f5fbf","#c0392b"
ORANGE="#D55E00"   # the delivery colour used throughout the paper, matching the atlas and toy figures
STYLE_COL = {0.25: "#7b3fa0", 1.00: ORANGE, 1.50: "#00767a"}
Y_TOP, Y_BOT = sigma1*3.2, sig_noise*0.055
DASH = (3.0, 1.5)          # all dashed lines share one pattern; only the colour differs

# delivery sizes
# the font sizes are the actual points that appear on paper, so each target width gets its
# own version. The figure must NEVER be scaled by the document, which would shrink 9 pt to 4.
# panels: both stacked, the criterion alone, or the distributions alone
# split into two separate figures with identical size and identical margins, so they line
# no titles and no panel letters inside the figure: identity is left entirely to the caption
GEOM = dict(W=2.54, H=2.05,               # 0.46\textwidth = 182.84 pt = 2.539 in
            M=dict(left=0.255, right=0.980, top=0.972, bottom=0.208),
            base=7.0, tick=6.8, lab=7.6, ttl=7.8, leg=6.4,
            lw=0.72, dot=2.9, big=4.8, curve=(1.5, 1.2))
PRESET = {
  "t1d_criterion_a": dict(panels="a", ylim_b=0.074, **GEOM),
  # the legend labels carry no prefix: with one the legend box spans far enough to cover the
  # first curve's peak, and without it the box clears the data and the upper limit can stay
  "t1d_criterion_b": dict(panels="b", ylim_b=0.074, short=True, **GEOM),
  # the side-by-side version puts both panels in one pdf, to be used with a figure and two
  # minipages, the figure on the left and the caption on the right, sized to the left one
  # the upper limit is raised for the side-by-side version: the column is narrow, the boxed
  # legend spans further, and without raising it the legend would cover the first peak.
  # Widening the left block instead would leave the caption too narrow to use.
  "t1d_criterion_pair": dict(W=3.53, H=2.35, panels="ab", layout="h",
      short=True, ylim_b=0.092, labels=True,
      M=dict(left=0.105, right=0.995, top=0.935, bottom=0.155, wspace=0.42),
      base=6.4, tick=6.2, lab=7.0, ttl=7.2, leg=6.0,
      lw=0.62, dot=2.6, big=4.2, curve=(1.4, 1.1)),
}

def render(stem, P):
    plt.rcParams.update({"font.size":P["base"], "axes.linewidth":0.95*P["lw"],
                         "xtick.labelsize":P["tick"], "ytick.labelsize":P["tick"],
                         "axes.labelsize":P["lab"],
                         "xtick.major.width":0.95*P["lw"], "ytick.major.width":0.95*P["lw"],
                         "xtick.minor.width":0.70*P["lw"], "ytick.minor.width":0.70*P["lw"],
                         "xtick.major.size":3.5*P["lw"]+1.2, "ytick.major.size":3.5*P["lw"]+1.2})
    LEG = dict(frameon=True, facecolor="white", edgecolor="#b9b6bf", framealpha=1.0,
               fontsize=P["leg"], loc="upper right", bbox_to_anchor=(0.990, 0.990),
               borderpad=0.45, labelspacing=0.34, handletextpad=0.45)
    want = P["panels"]; two = want == "ab"
    horiz = P.get("layout") == "h"
    fig, axs = plt.subplots(*( (1, len(want)) if horiz else (len(want), 1) ),
                            figsize=(P["W"], P["H"]), squeeze=False)
    axs = axs.reshape(-1, 1) if horiz else axs
    fig.subplots_adjust(**P["M"])
    A = dict(zip(want, axs[:, 0]))
    a, b = A.get("a"), A.get("b")

    if a is not None:                       # the criterion panel
        # no shading beyond the horizon: measured, it pulled the ink-weighted centre of the panel
        # noticeably to the right, and removing it brings the two panels level. The horizon is
        a.axhline(sig_noise, color=INK, lw=1.1*P["lw"], ls="--", zorder=4)
        a.plot([T_O, T_O], [Y_BOT, sig_noise], ls=(0,(3.4,1.9)), lw=1.25*P["lw"],
               color=ORANGE, zorder=5, solid_capstyle="butt")   # stops at the marker, no overshoot
        a.plot([T_O],[sig_noise], "o", ms=P["big"], mfc=ORANGE, mec=INK,
               mew=1.25*P["lw"], zorder=9)
        a.errorbar(t_meas, med, yerr=[med-q25, q75-med], fmt="o", ms=P["dot"],
                   lw=0, elinewidth=1.2*P["lw"], capsize=2.3*P["lw"], color=RED, zorder=6)
        a.set_yscale("log"); a.set_xlim(-0.30, 6.5); a.set_ylim(Y_BOT, Y_TOP)
        a.set_yticks([1e-3, sig_noise, 1e-1, 1e0, 1e1])
        a.set_yticklabels([r"$10^{-3}$", r"$\sigma_{\mathrm{noise}}$",
                           r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
        a.set_xticks([0,1,2,T_O,4,5,6])
        a.set_xticklabels([r"$0$",r"$1$",r"$2$",r"$T_O$",r"$4$",r"$5$",r"$6$"])
        for lb, tk in zip(a.get_xticklabels(), a.get_xticks()):
            if abs(tk-T_O) < 1e-9:
                lb.set_color(ORANGE); lb.set_fontweight("bold")
        a.set_xlabel("Noising time $t$"); a.set_ylabel(r"Excess Fano $|\widehat{\Sigma}_t|$")   # a longer label, which weights the left side
        if P.get("labels"): a.set_title("(a)", loc="left", fontsize=P["ttl"], pad=3)
        a.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.8)
        if P.get("legend", True): a.legend(handles=[
            Line2D([0],[0], ls="none", marker="o", ms=P["dot"]*1.07, color=RED,
                   label=r"$|\widehat{\Sigma}_t|$"),
            Line2D([0],[0], ls="none", marker="o", ms=P["big"]*0.86, mfc=ORANGE, mec=INK,
                   mew=1.2*P["lw"], label=r"$T_O$"),
            Line2D([0],[0], color=INK, lw=1.1*P["lw"], ls="--",
                   label=r"$\sigma_{\mathrm{noise}}$")],
            handlelength=1.3, **LEG)

    if b is not None:                       # the consequence panel
        b.fill_between(k, 0, q0, step="mid", facecolor="#b9b6bf", edgecolor="none",
                       lw=0, alpha=0.65, label=r"$q_0$", zorder=1)
        STYLE = {0.25: dict(color=STYLE_COL[0.25], z=6, off=0.0),
                 1.00: dict(color=STYLE_COL[1.00], z=3, off=None),   # the horizon itself: a solid line
                 1.50: dict(color=STYLE_COL[1.50], z=4, off=1.5)}    # dashed, offset by half a period so the two dashed curves stay distinguishable
        for f in LAD:
            st = STYLE[f]
            ln, = b.plot(k, laws[f],
                         lw=(P["curve"][0] if st["off"] is None else P["curve"][1]),
                         drawstyle="steps-mid", color=st["color"], zorder=st["z"],
                         label=((r"$T_O$" if f == 1.0 else rf"${f:g}\,T_O$") if P.get("short")
                                else (r"$T=T_O$" if f == 1.0 else rf"$T={f:g}\,T_O$")),
                         solid_capstyle="butt", dash_capstyle="butt")
            if st["off"] is not None:
                ln.set_dashes(DASH); ln.set_dash_capstyle("butt")
                ln.set_linestyle((st["off"], DASH))
        b.set_xlim(0, 70); b.set_ylim(0, P["ylim_b"])
        b.set_xticks([0,20,40,60]); b.set_xticklabels([r"$0$",r"$20$",r"$40$",r"$60$"])
        b.set_yticks([0,0.02,0.04,0.06])
        b.set_yticklabels([r"$0.00$",r"$0.02$",r"$0.04$",r"$0.06$"])
        b.set_xlabel("Count $n$"); b.set_ylabel("Probability")
        if P.get("labels"): b.set_title("(b)", loc="left", fontsize=P["ttl"], pad=3)
        b.grid(True, color=GRID, lw=0.4, alpha=0.8)
        if P.get("legend", True):
            b.legend(handlelength=(1.4 if P.get("short") else 1.7), **LEG)

    for axx in axs[:, 0]:
        axx.set_axisbelow(True)
        axx.spines["top"].set_visible(False); axx.spines["right"].set_visible(False)
        for sp in axx.spines.values(): sp.set_color("#8b8794")
        if axx.get_legend() is not None:
            axx.get_legend().get_frame().set_linewidth(0.7*P["lw"])
            axx.get_legend().set_zorder(20)

    # tighten the margins until the ink fills the page, leaving only a small pad, so that the
    # rendered width equals the caption width and the two align. The page size is unchanged,
    PAD = 0.5/72.0                      # half a point of safety margin
    for _ in range(3):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        W, H = fig.get_size_inches(); sp = fig.subplotpars
        fig.subplots_adjust(left   = sp.left   - (bb.x0 - PAD)/W,
                            right  = sp.right  + ((W - bb.x1) - PAD)/W,
                            bottom = sp.bottom - (bb.y0 - PAD)/H,
                            top    = sp.top    + ((H - bb.y1) - PAD)/H)
    # the tight bounding box measures text boxes rather than glyph ink and is a few points
    # short, so the margins are tightened once more against a REAL RENDERED BITMAP, measured in
    def ink_margins_pt(fig, dpi=200):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        arr = np.array(Image.open(buf).convert("L")) < 250
        ys, xs = np.where(arr); hpx, wpx = arr.shape
        W, H = fig.get_size_inches()
        return (xs.min()/wpx*W*72, (wpx-1-xs.max())/wpx*W*72,
                ys.min()/hpx*H*72, (hpx-1-ys.max())/hpx*H*72)   # left, right, top, bottom

    for _ in range(4):
        l_, r_, t_, b_ = ink_margins_pt(fig)
        W, H = fig.get_size_inches(); sp = fig.subplotpars
        fig.subplots_adjust(left   = sp.left   - (l_-PAD*72)/72/W,
                            right  = sp.right  + (r_-PAD*72)/72/W,
                            top    = sp.top    + (t_-PAD*72)/72/H,
                            bottom = sp.bottom - (b_-PAD*72)/72/H)
    l_, r_, t_, b_ = ink_margins_pt(fig)
    print(f"  [{stem}] ink margins pt: left {l_:.2f} right {r_:.2f} top {t_:.2f} bottom {b_:.2f}")

    # self-check: convert each legend box back into data coordinates and confirm it does not
    # no tight bounding box on save: that would crop to the content and the width would no
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for w, axx in A.items():
        if axx.get_legend() is None:
            print(f"  [{stem}] ({w}) no legend; colours are given in the caption"); continue
        bb = axx.get_legend().get_window_extent(r)
        (x0,y0),(x1,y1) = axx.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
        if w == "b":
            m = (k >= x0) & (k <= x1)
            mx = max([q0[m].max()] + [laws[f][m].max() for f in LAD]) if m.any() else 0.0
        else:
            m = (t_meas >= x0) & (t_meas <= x1)
            mx = float(q75[m].max()) if m.any() else 0.0
        print(f"  [{stem}] ({w}) legend x=[{x0:.4g},{x1:.4g}] lower edge {y0:.4g} "
              f"highest in range {mx:.4g}  " + ("clear" if y0 > mx else "** OVERLAPS **"))
    for e in ("pdf","png"):
        fig.savefig(OUT / f"{stem}.{e}", dpi=400, facecolor="white")
    plt.close(fig)

for stem, P in PRESET.items():
    render(stem, P)

json.dump({"criterion":"draft Eq. 27-31, S=1; only sigma_1 (t=0) and sigma_noise are inputs",
           "sigma1":sigma1,"sigma_noise_p95":sig_noise,"T_O":T_O,"N":N_DATA,
           "B_null":B_NULL,"R_meas":R_MEAS,"t":t_meas.tolist(),
           "sigma_hat_median":med.tolist(),
           "ladder":{str(f):{"T":f*T_O,"tv_to_q0":tvs[f]} for f in LAD}},
          open(OUT / "t1d_criterion.json","w"), indent=1)
print(f"[out] {OUT} / " + "  ".join(f"{s}.{{pdf,png}}" for s in PRESET) + "  t1d_criterion.json")
