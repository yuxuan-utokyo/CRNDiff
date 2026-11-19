# R21: dump a Lightning checkpoint's non-tensor metadata (epoch, global_step, hyper_parameters, callbacks) to json. Read-only, CPU.
import json, sys, torch
def clean(o, depth=0):
    if type(o).__name__ in ("DictConfig", "ListConfig"):
        from omegaconf import OmegaConf; return clean(OmegaConf.to_container(o, resolve=False), depth)
    if depth > 6: return repr(o)[:200]
    if isinstance(o, torch.Tensor): return f"<tensor {tuple(o.shape)}>"
    if isinstance(o, dict): return {str(k): clean(v, depth + 1) for k, v in o.items() if k not in ("state_dict", "optimizer_states")}
    if isinstance(o, (list, tuple)): return [clean(v, depth + 1) for v in o][:50]
    if isinstance(o, (int, float, str, bool)) or o is None: return o
    d = getattr(o, "__dict__", None)
    return {"__class__": type(o).__name__, **({k: clean(v, depth + 1) for k, v in d.items()} if isinstance(d, dict) else {"repr": repr(o)[:300]})}
out = {}
for p in sys.argv[2:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    out[p] = {"keys": sorted(ck.keys()), "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
              "hyper_parameters": clean(ck.get("hyper_parameters", {})), "callbacks": clean(ck.get("callbacks", {})),
              "lr_schedulers": clean(ck.get("lr_schedulers")), "optimizer_param_groups": [clean({k: v for k, v in g.items() if k != "params"}) for s in ck.get("optimizer_states", []) for g in s.get("param_groups", [])]}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str); print("ok", list(out))
