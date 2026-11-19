"""Test harness only (R16 baseline-wrapper check, 11-10): seed torch's global generator once, then run a script as __main__.
Used to compare r13_baselines_trainlib.py and r16_baselines_trainlib.py under the same global random state, because the generate
scripts' px.sample() draws from that generator and torch seeds it randomly per process.
    python run_with_torch_seed.py SEED SCRIPT.py [script args...]
"""
import runpy, sys
import torch
seed, script = int(sys.argv[1]), sys.argv[2]
torch.manual_seed(seed)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
