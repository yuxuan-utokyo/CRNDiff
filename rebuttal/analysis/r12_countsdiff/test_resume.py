"""A run stopped and resumed must produce the same weights as an uninterrupted run (CPU, deterministic)."""
import subprocess, sys, tempfile
from pathlib import Path
import torch
data, here = sys.argv[1], Path(__file__).parent / "countsdiff2k.py"
with tempfile.TemporaryDirectory() as td:
    a, b = Path(td) / "a", Path(td) / "b"
    base = [sys.executable, str(here), "train", "--data", data, "--seed", "7", "--steps", "4", "--batch", "4", "--eval-every", "2", "--device", "cpu", "--val-cells", "8"]
    subprocess.run(base + ["--out", str(a)], check=True, capture_output=True)
    subprocess.run(base + ["--out", str(b), "--stop-after", "2"], check=True, capture_output=True)
    r = subprocess.run(base + ["--out", str(b)], check=True, capture_output=True, text=True)
    assert "resumed at step 2" in r.stdout, r.stdout
    ca, cb = torch.load(a / "s7.last.pt", weights_only=False), torch.load(b / "s7.last.pt", weights_only=False)
    for key in ("model", "ema"):
        for k in ca[key]:
            assert torch.equal(ca[key][k], cb[key][k]), (key, k)
    assert ca["curve"] == cb["curve"], (ca["curve"], cb["curve"])
    print("resume test passed: weights, EMA and curves are identical", ca["curve"]["val"])
