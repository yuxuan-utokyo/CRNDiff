# -*- coding: utf-8 -*-
"""R14c NFE 64 / 128 delivery, the r14_fano format (rebuttal/r14_fano_prep.py): the the GPU server sampler outputs
(copied back unchanged to rebuttal/out/R14c/, n = 20,224), FIRST 20,000 rows, asserted non-negative integers within int16, scipy CSR in
4 row blocks of 5,000, save_npz(compressed=True), OURS_NFE<K>_s20260931_rows<lo>-<hi>.npz; SOURCES.tsv (source npy + sha256, command,
ckpt and sampler sha, walltime and peak_alloc_gb from the sampler's own sample_log json); SHA256SUMS (2 columns, LF, no BOM).
CPU only, 1 thread, read-only on the sources.
    python -m rebuttal.r14c_deliver
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "rebuttal" / "out" / "R14c"
X = Path(r"work\exchange\r14c_nfe")
N, BLK = 20000, 5000
CKPT_SHA = "2b07bf5ee19b342473f8222b75b9be8d791378b325681e39036bce38b052fe77"
SAMPLER_SHA = "6cc170883d793c120229719bd3c7aa087de0382231351b849e7309459930263a"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    X.mkdir(parents=True, exist_ok=True)
    rows = ["K\tsource_npy\tsource_sha256\tn_source\tslice\tcommand (the GPU server, from the HVG root)\tckpt_sha256\tsampler_sha256\twalltime_s\tpeak_alloc_gb\tdevice"]
    for k in (64, 128):
        src = SRC / f"linspace_T_O_K{k}_spilot_run1c_s20260625_samp20260931_batched1_chunk256_R14c_K{k}_noa4.npy"
        x = np.load(src)
        assert x.shape == (20224, 2000), x.shape
        xa = np.asarray(x[:N])
        assert xa.min() >= 0 and xa.max() <= np.iinfo(np.int16).max and np.array_equal(xa, np.rint(xa))
        xi = xa.astype(np.int16)
        for lo in range(0, N, BLK):
            p = X / f"OURS_NFE{k}_s20260931_rows{lo}-{lo + BLK}.npz"; tmp = p.with_name(p.name + ".part.npz")
            sparse.save_npz(tmp, sparse.csr_matrix(xi[lo:lo + BLK]), compressed=True); os.replace(tmp, p)
        logs = sorted(SRC.glob(f"sample_log_K{k:03d}_*R14c_K{k}*.json"))
        assert len(logs) == 1, logs
        c = next(iter(json.loads(logs[0].read_text(encoding="utf-8"))["cells"].values()))
        cmd = (f"python -u assets/legacy_root/Baseline/OURS/code/sample_hvg2k_noa4.py --ckpt assets/models/generator/pilot_run1c_s20260625.pt "
               f"--ks {k} --samps 20260931 --n-gen 20224 --cell-chunk 256 --batched --no-a4 --tau 0.0 --tag _R14c_K{k}")
        rows.append(f"{k}\t{src.name}\t{sha(src)}\t20224\tfirst {N} rows, int16, CSR, 4 blocks of {BLK}\t{cmd}\t{CKPT_SHA}\t{SAMPLER_SHA}"
                    f"\t{c['walltime_s']:.1f}\t{c['peak_alloc_gb']:.3f}\tthe GPU server GPU 0, NVIDIA RTX 6000 Ada Generation")
        print(f"K{k}: dtype {x.dtype} max {int(xa.max())} zero {float((xa == 0).mean()):.4f} walltime {c['walltime_s']:.1f}s peak {c['peak_alloc_gb']:.3f} GB", flush=True)
    (X / "SOURCES.tsv").write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    (X / "SHA256SUMS").write_bytes("".join(f"{sha(p)}  {p.name}\n" for p in sorted(X.iterdir())
                                           if p.is_file() and p.name != "SHA256SUMS").encode("utf-8"))
    print("[done] R14C DELIVERED", flush=True)


if __name__ == "__main__":
    main()
