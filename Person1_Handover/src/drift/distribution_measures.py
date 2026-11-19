"""Three distribution measures: Kolmogorov-Smirnov (KS), Wasserstein Distance, and Jensen-Shannon (JS) Divergence.

Quantifies distributional differences between a reference window representing accepted
normal behaviour and a recent window representing current operational behaviour.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any, Union
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def compute_ks_statistic(ref: np.ndarray, recent: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic between reference and recent samples."""
    ref_clean = np.asarray(ref).flatten()
    recent_clean = np.asarray(recent).flatten()
    ref_clean = ref_clean[np.isfinite(ref_clean)]
    recent_clean = recent_clean[np.isfinite(recent_clean)]

    if len(ref_clean) < 2 or len(recent_clean) < 2:
        return 0.0

    res = stats.ks_2samp(ref_clean, recent_clean)
    return float(res.statistic)


def compute_wasserstein_distance(ref: np.ndarray, recent: np.ndarray) -> float:
    """Compute the 1-Wasserstein (Earth Mover's) distance between reference and recent samples."""
    ref_clean = np.asarray(ref).flatten()
    recent_clean = np.asarray(recent).flatten()
    ref_clean = ref_clean[np.isfinite(ref_clean)]
    recent_clean = recent_clean[np.isfinite(recent_clean)]

    if len(ref_clean) < 2 or len(recent_clean) < 2:
        return 0.0

    return float(stats.wasserstein_distance(ref_clean, recent_clean))


def compute_js_divergence(ref: np.ndarray, recent: np.ndarray, bins: int = 40) -> float:
    """Compute Jensen-Shannon divergence between empirical distributions using common histogram bins."""
    ref_clean = np.asarray(ref).flatten()
    recent_clean = np.asarray(recent).flatten()
    ref_clean = ref_clean[np.isfinite(ref_clean)]
    recent_clean = recent_clean[np.isfinite(recent_clean)]

    if len(ref_clean) < 2 or len(recent_clean) < 2:
        return 0.0

    # Common range across both samples
    val_min = min(float(ref_clean.min()), float(recent_clean.min()))
    val_max = max(float(ref_clean.max()), float(recent_clean.max()))

    if np.isclose(val_min, val_max):
        return 0.0

    bin_edges = np.linspace(val_min, val_max, bins + 1)
    p, _ = np.histogram(ref_clean, bins=bin_edges, density=False)
    q, _ = np.histogram(recent_clean, bins=bin_edges, density=False)

    # Laplace smoothing to avoid zero probabilities
    p = (p + 1e-4) / (p.sum() + 1e-4 * bins)
    q = (q + 1e-4) / (q.sum() + 1e-4 * bins)

    # JS distance squared is the Jensen-Shannon divergence
    js_dist = jensenshannon(p, q, base=2.0)
    js_div = float(js_dist ** 2) if np.isfinite(js_dist) else 0.0
    return min(1.0, max(0.0, js_div))


def compute_all_distribution_measures(
    ref: np.ndarray, recent: np.ndarray
) -> Dict[str, float]:
    """Compute KS, Wasserstein, and JS divergence across 1-D arrays or multi-channel arrays."""
    ref_arr = np.asarray(ref)
    rec_arr = np.asarray(recent)

    if ref_arr.ndim == 1:
        ks = compute_ks_statistic(ref_arr, rec_arr)
        w = compute_wasserstein_distance(ref_arr, rec_arr)
        js = compute_js_divergence(ref_arr, rec_arr)
        return {"ks": ks, "wasserstein": w, "js": js}

    # Multi-channel: compute per channel and average
    num_channels = ref_arr.shape[1]
    ks_list = []
    w_list = []
    js_list = []

    for c in range(num_channels):
        r_c = ref_arr[:, c]
        q_c = rec_arr[:, c]
        ks_list.append(compute_ks_statistic(r_c, q_c))
        w_list.append(compute_wasserstein_distance(r_c, q_c))
        js_list.append(compute_js_divergence(r_c, q_c))

    return {
        "ks": float(np.mean(ks_list)),
        "wasserstein": float(np.mean(w_list)),
        "js": float(np.mean(js_list)),
        "ks_per_channel": ks_list,
        "wasserstein_per_channel": w_list,
        "js_per_channel": js_list,
    }

