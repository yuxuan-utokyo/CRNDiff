# -*- coding: utf-8 -*-
"""from `sample_hvg2k_la.py` **generated verbatim** `sample_hvg2k_noa4.py`, only adds one `--no-a4` switch.

why a script instead of editing by hand:`sample_hvg2k_la.py` is itself derived this way from `sample_hvg2k.py`
generated (see its file header). Every replacement asserts **exactly one hit**, so"nothing changed apart from this switch"
is machine-verifiable, not dependent on human review. Red line 6.1:`sample_hvg2k.py` / `model_hvg2k.py` /
`bank_dedup.py` / `vendor_ours*.py` / `sample_hvg2k_la.py` is left byte-identical.

not given `--no-a4` the whole numerical path is identical to `sample_hvg2k_la.py` **bit-identical**
(`if not _NO_A4:` true -> execute the original masked_fill line; the subsequent normalization is kept as is).

    python gen_noa4.py            # write sample_hvg2k_noa4.py and print the hit count of every replacement
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "sample_hvg2k_la.py"
DST = SRC.with_name("sample_hvg2k_noa4.py")

REPL = [
    # ---- 1. file header ----
    ('"""hvg2k route OURS sampling + **logit adjustment (LA)**. T3 layer.',
     '''"""hvg2k · OURS sampling + LA/PPC + **switchable A4**. From `gen_noa4.py` from
`sample_hvg2k_la.py` generated verbatim; the only change is a switch added to the A4 line.

    # no A4 + PPC (the arm added in this round)
    python sample_hvg2k_noa4.py --ckpt <ckpt> --no-a4 --tau 0.5 \\
        --logpi <ppc_logc>.npy --last-only --ks 32 --n-gen 20000 --cell-chunk 500

    # no A4 + bare model (same-K control; **this branch gives the number that proves A4 was doing work**)
    python sample_hvg2k_noa4.py --ckpt <ckpt> --no-a4 --tau 0 --ks 32 ...

## why this ablation

A4(`p.masked_fill(n > d_g, 0)`) is a **hard cut at sampling time**; it prescribes a priori that the model must not
"never seen in the training set count"puts mass. Its original empirical basis was the D=10 multi-gene toy
**bare model **without A4 has 3.909 times the total mass**, but that conclusion **predates PPC**. In the same table,
Once reweighting is on, R_mass without A4 returns to 1.142 (LA-0.5) and 0.960 (LA-1.0+WEIGHT-0.5).
2D even more decisive on the toy: A4 on/off is bit-identical, because the model already puts only 1e-13 above d_g.

so the real question is: **after PPC suppresses it as measured, is the A4 prior rule still needed.**
needs four arms ({A4 on/off} × {PPC on/off}) to answer; missing"no A4 + bare model"that branch,
"A4 redundant"is just"A4 never used".

## definition

`--no-a4` automatically goes to `--tag` appended after `_noa4`; output file names must include this segment and cannot clash with
clashes with an existing output name (red line: never overwrite an existing .npy).`no_a4` written to both the result json and RUN_LOG.
`n_gen` / `cell_chunk` / `batched` / seed are part of the definition and **must match the compared arm verbatim**.

------------------------------------------------------------------------
the following is `sample_hvg2k_la.py` original notes (kept verbatim):

hvg2k route OURS sampling + **logit adjustment (LA)**. T3 layer.''', 1),

    # ---- 2. global switch ----
    ("@torch.no_grad()\ndef sample_chunk(",
     "_NO_A4 = False        # overridden by main() via --no-a4; when False the numerical path is bit-identical to the la version\n\n\n"
     "@torch.no_grad()\ndef sample_chunk(", 1),

    # ---- 3. A4 add a switch to that line ----
    ('''        p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                          > dmax_t[None, :, None], 0.0)          # A4 posterior-support truncation''',
     '''        # ==== A4 posterior-support truncation (the only change in this file: a switch was added)==========
        # `--no-a4` **this line is not executed**; the model may keep its own output on n > d_g.
        # the subsequent normalization is kept as is (without A4, p is already normalized, so this step is the identity),
        # this way `_NO_A4 = False` numerical path identical to `sample_hvg2k_la.py` bit-identical.
        if not _NO_A4:
            p = p.masked_fill(torch.arange(p.shape[2], device=dev)[None, None, :]
                              > dmax_t[None, :, None], 0.0)
        # ======================================================================''', 1),

    # ---- 4. command-line arguments ----
    ('''    ap.add_argument("--last-only", action="store_true",''',
     '''    ap.add_argument("--no-a4", action="store_true",
                    help="★ turn off the A4 support cut (no longer hard-zero n > d_g). Automatically goes to --tag "
                         "_noa4 appended, so outputs cannot clash with existing files. How to read:\\n"
                         "  bare model much worse + PPC arm essentially unchanged => A4 is redundant and can be removed\\n"
                         "  PPC arm also gets worse                    => A4 is irreplaceable, keep it")
    ap.add_argument("--last-only", action="store_true",''', 1),

    # ---- 5. main() sets the global + appends tag ----
    ('''    global _LA_TAU, _LA_PI_TAU, _LA_LAST_ONLY
    _LA_TAU = float(a.tau)''',
     '''    global _LA_TAU, _LA_PI_TAU, _LA_LAST_ONLY, _NO_A4
    _NO_A4 = bool(a.no_a4)
    if _NO_A4:
        a.tag = (a.tag or "") + "_noa4"      # output names must include it, to avoid clashing with existing arms
        rl("  [A4] **is closed**: the posterior is no longer truncated at n <= d_g; tag automatically appended _noa4")
    else:
        rl("  [A4] on (numerical path bit-identical to sample_hvg2k_la.py)")
    _LA_TAU = float(a.tau)''', 1),

    # ---- 6. the result json records this bit ----
    ('''    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,''',
     '''    out = {"ckpt": Path(a.ckpt).name, "T": T, "NMAX": nmax, "dim": dim,
           "no_a4": bool(a.no_a4), "tau": float(a.tau), "last_only": bool(a.last_only),''', 1),

    # ---- 7. the log line at launch includes the A4 state ----
    ('''    rl(f"### hvg2k sampling starts ckpt={Path(a.ckpt).name} K={a.ks} samp={a.samps} "''',
     '''    rl(f"### hvg2k sampling starts [A4={'off' if a.no_a4 else 'on'}] "
       f"ckpt={Path(a.ckpt).name} K={a.ks} samp={a.samps} "''', 1),
]


def main():
    s = SRC.read_text(encoding="utf-8")
    for i, (old, new, want) in enumerate(REPL, 1):
        got = s.count(old)
        assert got == want, f"replacement {i} matched {got} times, expected {want}: refusing to generate"
        s = s.replace(old, new, want)
        print(f"  replacement {i}: matched {got} times  OK")
    if DST.exists():
        print(f"[warn] {DST.name} exists, overwriting (it is generated, not hand-written code)")
    DST.write_text(s, encoding="utf-8")
    import ast
    ast.parse(s)
    print(f"\nwrote {DST}  (syntax check passed)")
    print("self-check: without --no-a4 it should be bit-identical to sample_hvg2k_la.py: "
          "differing lines should cover only the 7 places above")


if __name__ == "__main__":
    main()
