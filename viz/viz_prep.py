# viz/viz_prep.py
import numpy as np
from utils.tree_arrays import *
import matplotlib as mpl
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from config import SEED

STATE_LOOKUP = globals().get("INT_TO_STATE", {})
ARCH_LOOKUP  = globals().get("INT_TO_ARCHETYPE", {})
VAL_LOOKUP   = globals().get("INT_TO_VALUE", {})

SLIDE_FIGSIZE = (13.33, 7.5)   # 16:9 inches for PowerPoint
SLIDE_DPI = 300                # crisp for projection/print
PAD_INCHES = 0.12              # padding to avoid cropped labels


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mpl.rcParams.update({
    "figure.figsize": SLIDE_FIGSIZE,
    "figure.dpi": 150,
    "savefig.dpi": SLIDE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": PAD_INCHES,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    # viewer-friendly: avoid scientific notation/offsets for deck clarity
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": False,
})



def _to_1d_list(x):
    if x is None:
        return []
    # bring from CuPy if needed
    if hasattr(x, "get"):
        x = x.get()
    arr = np.asarray(x)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.reshape(-1).tolist()



def build_viz_inputs(metrics: dict) -> dict:
    """
    Normalize everything for viz: CPU lists, 1-D, no CuPy, plus precomputed cumrevs.
    """
    keys = [
        "energy","net_revenue","rolling_churn",
        "mean_health","mean_engagement",
        "alive_users","churned_users",
        "content_attention","content_novelty","content_response",
        "annotations"
    ]
    out = {k: _to_1d_list(metrics.get(k)) for k in keys}
    # precompute cumulative revenue for downstream plots/CSV/animations
    nv = np.asarray(out["net_revenue"], dtype=float)
    out["cumrev"] = nv.cumsum().tolist() if nv.size else []
    return out

def counts_to_state_dicts(counts_array):
    out = {}
    for b, arr in enumerate(counts_array):
        if arr is None or len(arr) == 0:
            continue
        out[b] = {INT_TO_STATE[i]: int(c) for i, c in enumerate(arr) if c > 0}
    return out

def counts_to_named_dicts(counts_array):
    out = {}
    for b, arr in enumerate(counts_array):
        if arr is None or len(arr) == 0:
            continue
        out[b] = {INT_TO_ACTION[i]: int(c) for i, c in enumerate(arr) if c > 0}
    return out

def _to_scalar(val):
    """Convert CuPy/NumPy scalars or arrays to Python scalar for hashing/lookup."""
    try:
        import cupy as cp
        if isinstance(val, cp.ndarray):
            return val.item()  # extract single element
    except ImportError:
        pass

    import numpy as np
    if isinstance(val, np.ndarray):
        return val.item()
    if isinstance(val, (np.generic,)):
        return val.item()

    return val  # already a Python type



def get_value_counts(user_states):
    """Count value tiers (integers)."""
    counts = {}

    # If user_states is dict-like, extract the "value" field from each state dict
    if hasattr(user_states, "values"):  
        iterable = (state.get("value") for state in user_states.values())
    else:  
        # Otherwise assume it's already an array of integers
        iterable = user_states  

    for v in iterable:
        val = _to_scalar(v)  # ensure numpy/cupy scalars → python int
        counts[val] = counts.get(val, 0) + 1
    return counts



def get_archetype_counts(archetypes_array):
    """
    Count archetype integers in the array and return a dict with labels.
    Works with CuPy/Numpy arrays or lists of ints.
    """
    # If cupy, bring to host
    if hasattr(archetypes_array, "get"):  # CuPy
        archetypes_array = archetypes_array.get()

    counts = {}
    unique, freqs = np.unique(archetypes_array, return_counts=True)
    for u, f in zip(unique, freqs):
        label = ARCH_LOOKUP.get(int(u), str(u))  # int → label
        counts[label] = int(f)
    return counts

