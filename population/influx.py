"""
influx.py
---------
Population inflow model for ChurnLab : Equilibria.

This module computes the expected *influx rate* of new users per batch as a
function of:
  • Aggregate population health & engagement
  • Archetype-specific fatigue (as an effective "friction")
  • System energy usage (infra/prod cost proxy)
  • Seasonality / cyclic effects
  • Asymptotic capacity limits

Public API
----------
- compute_user_influx_rate(user_states, energy, batch_idx) -> float
    CPU-side convenience wrapper that accepts a dict-like user snapshot.
    Suitable for baselines or non-GPU branches.

- compute_user_influx_rate_gpu(health, engagement, archetypes, energy, batch_idx) -> float
    GPU-vectorized path for fast branches. Expects CuPy arrays.

Both functions share the same semantics and return a Python float in [0, 1].
"""

from __future__ import annotations

from typing import Dict, Mapping, Any

import numpy as np
import cupy as cp

from utils.tree_arrays import ARCH_ARRAY, ARCH_ATTRS  # archetype attributes (single source of truth)
from config import (
    HEALTH_THRESH,
    INFLUX_RATE,
    INFLUX_PERIOD,
    INFLUX_AMPLITUDE,
    MAX_USERS,
)

__all__ = ["compute_user_influx_rate", "compute_user_influx_rate_gpu"]


# --------------------------------------------------------------------------- #
# CPU-friendly wrapper (dict-based)
# --------------------------------------------------------------------------- #
def compute_user_influx_rate(user_states: Mapping[Any, Dict[str, float]],
                             energy: float,
                             batch_idx: int) -> float:
    """
    Compute expected influx rate using a dict-like population snapshot.

    Parameters
    ----------
    user_states : Mapping[uid, dict]
        Each value dict must contain:
          - "archetype": int (index into ARCH_ARRAY)
          - "user_health": float in [0,1]
          - "engagement":  float in [0,1]
        (Other keys are ignored.)
    energy : float
        Per-batch energy / cost proxy (higher → lower influx).
    batch_idx : int
        Current batch index (time; drives cyclic effects).

    Returns
    -------
    float
        Influx rate in [0, 1] (fraction of current alive users to add).

    Notes
    -----
    This wrapper mirrors the GPU function for convenience and testing.
    For performance at scale, prefer `compute_user_influx_rate_gpu`.
    """
    if not user_states:
        return 0.0

    # Aggregate metrics
    healths, engages, fat_mults = [], [], []

    for s in user_states.values():
        a_idx = int(s["archetype"])
        # fatigue multiplier from canonical archetype table
        fat_mult = float(ARCH_ARRAY[a_idx, ARCH_ATTRS["fatigue_mult"]])
        fat_mults.append(max(0.01, fat_mult))

        healths.append(float(s["user_health"]))
        engages.append(float(s["engagement"]))

    mean_h = float(np.mean(healths))
    mean_e = float(np.mean(engages))
    mean_f = float(np.mean(np.array(engages) * np.array(fat_mults)))

    # Health proxy (nonlinear): happy pops push more organic growth
    raw_health = (1.2 * mean_h + 0.4 * mean_e) - (0.25 * mean_f)
    raw_health = float(np.clip(raw_health, 0.0, 1.0))
    if mean_h < HEALTH_THRESH:
        raw_health *= 0.75

    # Logistic squashing around 0.5 for stability
    health_factor = 1.0 / (1.0 + np.exp(-10.0 * (raw_health - 0.5)))

    # Energy penalty (normalize by population size)
    n_users = max(1, len(user_states))
    energy_penalty = 1.0 / (1.0 + (energy / n_users) * 0.10)

    # Base influx
    influx_rate = INFLUX_RATE * health_factor * energy_penalty

    # Cyclic seasonality
    cycle = np.sin(2 * np.pi * (batch_idx / INFLUX_PERIOD))
    influx_rate *= max(0.0, 1.0 + INFLUX_AMPLITUDE * cycle)

    # Capacity constraint (asymptotic)
    size_factor = np.exp(-n_users / MAX_USERS)
    influx_rate *= size_factor

    return max(0.0, float(influx_rate))


# --------------------------------------------------------------------------- #
# GPU-vectorized version (preferred for large-N)
# --------------------------------------------------------------------------- #
def compute_user_influx_rate_gpu(health: cp.ndarray,
                                 engagement: cp.ndarray,
                                 archetypes: cp.ndarray,
                                 energy: float,
                                 batch_idx: int) -> float:
    """
    Compute expected influx rate using GPU vectors (CuPy).

    Parameters
    ----------
    health : cp.ndarray, shape (N,)
        User health values in [0,1].
    engagement : cp.ndarray, shape (N,)
        User engagement values in [0,1].
    archetypes : cp.ndarray, shape (N,)
        Archetype indices (int) per user; used to pull fatigue multipliers.
    energy : float
        Per-batch energy / cost proxy (higher → lower influx).
    batch_idx : int
        Current batch index (time; drives cyclic effects).

    Returns
    -------
    float
        Influx rate in [0, 1] (fraction of current alive users to add).
    """
    if health.size == 0:
        return 0.0

    # Archetype-specific fatigue multiplier (lower bounded)
    fat_mult = ARCH_ARRAY[archetypes, ARCH_ATTRS["fatigue_mult"]]
    fat_mult = cp.maximum(fat_mult, 0.01)

    mean_h = cp.mean(health)
    mean_e = cp.mean(engagement)
    mean_f = cp.mean(engagement * fat_mult)

    # Nonlinear health proxy (slightly more convex on happiness)
    raw_health = 1.8 * (mean_h ** 1.5) + 0.4 * mean_e - 0.2 * mean_f
    raw_health = cp.where(mean_h < HEALTH_THRESH, raw_health * 0.75, raw_health)

    # Smooth logistic (tanh form) to keep gradients stable
    health_factor = 0.5 + 0.5 * cp.tanh(3.0 * (raw_health - 0.5))

    # Energy penalty normalized by population size
    n_users = float(health.size)
    energy_penalty = 1.0 / (1.0 + (float(energy) / (n_users + 1.0)) * 0.13)

    influx_rate = INFLUX_RATE * health_factor * energy_penalty

    # Seasonality / cyclic component
    cycle = cp.sin(2 * cp.pi * (batch_idx / INFLUX_PERIOD))
    influx_rate = influx_rate * cp.maximum(0.0, 1.0 + INFLUX_AMPLITUDE * cycle)

    # Capacity limit (smooth logistic taper near 90% of MAX_USERS)
    size_factor = 1.0 / (1.0 + cp.exp((n_users - 0.9 * MAX_USERS) / (0.05 * MAX_USERS)))

    influx_rate = influx_rate * size_factor
    return float(influx_rate.item())
