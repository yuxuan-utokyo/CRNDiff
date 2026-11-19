# Count labels in HVG2K test_celltype.npy only (vice 16:11/16:20: labels only; test.npy is never opened; no metric). 1 thread.
import json, sys
from collections import Counter
import numpy as np
p = sys.argv[1]
y = np.load(p, allow_pickle=True).astype(str)
c = Counter(y.tolist())
print(json.dumps({"file": p, "n": int(y.size), "n_labels": len(c), "counts": dict(c.most_common())}, ensure_ascii=False, indent=1))
