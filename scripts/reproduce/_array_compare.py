"""Compare figure extracts without depending on ZIP bytes or platform libm rounding."""
from pathlib import Path

import numpy as np


def compare_npz(expected: Path, actual: Path) -> tuple[bool, str]:
    """Require identical keys/shapes/dtypes and exact discrete data.

    Floating arrays allow eight machine epsilons of relative rounding, with no
    absolute floor. This accommodates exp/reduction differences across platforms;
    it does not relax the hash checks protecting frozen source files or inputs.
    """
    if not expected.is_file() or not actual.is_file():
        return False, "missing figure extract"
    with np.load(expected, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            return False, "array keys differ"
        rounded = []
        for key in left.files:
            a, b = left[key], right[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                return False, f"{key}: shape or dtype differs"
            if np.array_equal(a, b):
                continue
            if np.issubdtype(a.dtype, np.floating):
                rtol = 8 * np.finfo(a.dtype).eps
                if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a, b, rtol=rtol, atol=0):
                    rounded.append(key)
                    continue
            return False, f"{key}: data differs"
    if rounded:
        return True, f"matching data ({len(rounded)} floating arrays differ within 8 machine epsilons)"
    return True, "all arrays exactly equal"
