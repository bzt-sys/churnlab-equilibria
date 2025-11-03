"""
row_generator.py
----------------
Row synthesis utilities for ChurnLab : Equilibria.

Responsibilities
  • Initialize per-user mechanism state (CPU/GPU)
  • Generate observed event rows per batch with realistic timing and noise
  • Keep schema stable for controllers and downstream viz

Notes
  - Public API preserved:
      simulate_absence_pressure(user_health, rng)
      generate_mechanism_row(uid, ts, rng)
      generate_mechanism_rows_gpu(uids, ts, rng)
      generate_rows_for_user(uid, ts, mech_row, rng)
  - All randomness is explicit via numpy/cupy RNGs (no hidden globals).
  - All archetype/state/event constants come from utils.tree_arrays (single source of truth).
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Dict, Iterable, Optional

import cupy as cp
import numpy as np
import pandas as pd
from numpy.random import default_rng

from utils.tree_arrays import (
    # Archetypes
    ARCH_ARRAY,
    ARCH_ATTRS,
    # States & multipliers
    STATE_TO_INT,
    STATE_MULTIPLIERS,
    # Value tiers
    TIER_PROBS,
    VALUE_TIERS,
    # Events
    EVENT_TYPES,
    SEVERITY_LEVELS,
    EVENT_PROBS_BY_STATE,
    EVENT_SEVERITY_MAP,
    INT_TO_EVENT_TYPE,
    EVENT_TYPE_SCORES,
)
from config import *

# --- cached GPU copies for fast vector draws ---
_TIER_PROBS_CP   = cp.asarray(TIER_PROBS, dtype=cp.float32)
_VALUE_TIERS_CP  = cp.asarray(VALUE_TIERS, dtype=cp.float32)

_COOLDOWN_COL    = ARCH_ATTRS["cooldown"]
_ROWMEAN_COL     = ARCH_ATTRS.get("row_mean", None)      # expected present in archetype table
_VOLATILITY_COL  = ARCH_ATTRS.get("volatility", None)    # expected present in archetype table


# --------------------------------------------------------------------------- #
# Activity / appearance model
# --------------------------------------------------------------------------- #
def simulate_absence_pressure(user_health: float, rng: Optional[np.random.Generator] = None) -> bool:
    """
    Stochastic appearance model: whether a user generates rows this batch.

    Heuristic:
      - Very healthy users almost always appear; unhealthy users appear rarely.
    """
    # Keep previous semantics (thresholds match legacy logic); use numpy RNG when provided
    if rng is None:
        # Preserve legacy behavior that used `random.random()` when no rng passed.
        roll = random.random
    else:
        roll = lambda: float(rng.random())

    if user_health >= 0.8:
        return True
    elif user_health >= 0.5:
        return roll() < 0.95
    elif user_health >= 0.2:
        return roll() < 0.8
    else:
        return roll() < 0.4


# --------------------------------------------------------------------------- #
# Mechanism initializers
# --------------------------------------------------------------------------- #
def generate_mechanism_row(uid: int, ts, rng: Optional[np.random.Generator] = None) -> Dict:
    """
    CPU initializer for a single user's mechanism row.

    Uses canonical archetype attributes from ARCH_ARRAY/ARCH_ATTRS.
    """
    rng = rng or default_rng()
    num_arch = int(ARCH_ARRAY.shape[0])

    arch_idx  = int(rng.integers(0, num_arch))
    cooldown  = int(ARCH_ARRAY[arch_idx, _COOLDOWN_COL])

    # Value tier via categorical
    tier_idx  = int(rng.choice(len(TIER_PROBS), p=TIER_PROBS))
    value_tier = float(VALUE_TIERS[tier_idx])

    return {
        "uid": uid,
        "timestamp": ts,
        "archetype": arch_idx,
        "user_health": float(rng.uniform(0.6, 1.0)),
        "engagement": 0.0,
        "cooldown": cooldown,
        "state": STATE_TO_INT["stable"],
        "decay": 0.0,
        "recovered": 0,
        "value": value_tier,
        "strategy_factor": 1.0,
        "burnout_counter": 0,
    }


def generate_mechanism_rows_gpu(uids: Iterable[int], ts, rng: Optional[cp.random.RandomState] = None) -> Dict[str, cp.ndarray]:
    """
    Vectorized GPU initializer for a batch of users.

    Parameters
    ----------
    uids : iterable[int] or cp.ndarray[int64]
    ts   : timestamp (unused per user; kept for API symmetry)
    rng  : cupy RandomState/Generator (optional)

    Returns
    -------
    dict[str, cp.ndarray]
      Keys align with PopulationBranch column names.
    """
    rng = rng or cp.random.RandomState()  # reproducibility comes from caller seeding
    uids_cp = cp.asarray(uids, dtype=cp.int64)
    N = int(uids_cp.size)

    num_arch = int(ARCH_ARRAY.shape[0])
    arch_ids = rng.randint(0, num_arch, size=N, dtype=cp.int32)

    cooldowns  = ARCH_ARRAY[arch_ids, _COOLDOWN_COL].astype(cp.int16)
    health     = rng.uniform(0.6, 1.0, size=N).astype(cp.float32)
    engagement = cp.zeros(N, dtype=cp.float32)
    state      = cp.full(N, STATE_TO_INT["stable"], dtype=cp.int8)

    # Value tier via categorical draw
    tier_idx = rng.choice(cp.arange(_TIER_PROBS_CP.size), size=N, p=_TIER_PROBS_CP)
    value    = _VALUE_TIERS_CP[tier_idx].astype(cp.float32)

    return {
        "uid": uids_cp,
        "archetype": arch_ids.astype(cp.int8),
        "cooldown": cooldowns,
        "user_health": health,
        "engagement": engagement,
        "state": state,
        "value": value,
        # defaults matching schema used elsewhere
        "strategy_factor": cp.ones(N, dtype=cp.float32),
        "burnout_counter": cp.zeros(N, dtype=cp.int16),
    }


# --------------------------------------------------------------------------- #
# Observed row synthesis (controller input)
# --------------------------------------------------------------------------- #
def generate_rows_for_user(uid: int, ts, mech_row: Dict, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """
    Build an observed row dataframe for a user at timestamp `ts`.

    Parameters
    ----------
    uid : int
        User identifier.
    ts : datetime-like
        Batch timestamp; used to space intra-batch events realistically.
    mech_row : dict
        Mechanism row (source of truth) with keys:
          archetype:int, user_health:float, engagement:float,
          cooldown:int, state:int, value:float, recovered:bool (optional)
    rng : numpy.random.Generator, optional
        RNG for all draws. If None, a default RNG is created.

    Returns
    -------
    pandas.DataFrame
        Columns: uid, timestamp, event_type, event_severity, session_id,
                 session_position, engagement_score, cooldown, value,
                 rolling_activity, recovered
        (May be empty if the user does not appear this batch.)
    """
    rng = rng or default_rng()

    # --- 1) Optional absence (skip if user "doesn't show") ---
    if not simulate_absence_pressure(float(mech_row["user_health"]), rng=rng):
        return pd.DataFrame()

    # --- 2) Compute noisy row count from archetype & state multipliers ---
    arch_idx    = int(mech_row["archetype"])
    volatility  = float(ARCH_ARRAY[arch_idx, _VOLATILITY_COL]) if _VOLATILITY_COL is not None else 0.15
    row_mean    = float(ARCH_ARRAY[arch_idx, _ROWMEAN_COL])    if _ROWMEAN_COL is not None else 8.0

    fatigue_damp    = max(0.0, 1.0 - float(mech_row["engagement"]))
    activity_factor = float(np.mean(mech_row.get("activity", [1.0])))
    cooldown_factor = 1.0 - min(1.0, 1.0 / (int(mech_row["cooldown"]) + 1))
    state_mult      = float(STATE_MULTIPLIERS.get(mech_row["state"], {}).get(mech_row["state"], 1.0))

    base_count = (
        row_mean
        * float(mech_row["user_health"])
        * fatigue_damp
        * activity_factor
        * state_mult
        * cooldown_factor
    )

    noisy_count = int(np.clip(rng.normal(loc=base_count, scale=base_count * volatility), 0, None))

    # --- 3) Timestamp realism (intra-batch spacing by health) ---
    if noisy_count == 0:
        return pd.DataFrame()

    if mech_row["user_health"] >= 0.8:
        timestamps = [ts + timedelta(minutes=i) for i in range(noisy_count)]
    elif mech_row["user_health"] >= 0.5:
        timestamps = sorted([ts + timedelta(minutes=int(rng.integers(0, 30))) for _ in range(noisy_count)])
    elif mech_row["user_health"] >= 0.2:
        timestamps = sorted([ts + timedelta(minutes=int(rng.integers(0, 60))) for _ in range(noisy_count)])
    else:
        timestamps = sorted([ts + timedelta(minutes=int(rng.integers(0, 180))) for _ in range(noisy_count)])

    # --- 4) Event draws (types → severities → engagement scores) ---
    state_idx    = int(mech_row["state"])
    event_probs  = EVENT_PROBS_BY_STATE.get(state_idx, [0.2, 0.4, 0.3, 0.05, 0.05])

    event_types = rng.choice(len(EVENT_TYPES), p=event_probs, size=noisy_count)
    event_sev   = np.array([
        rng.choice(len(SEVERITY_LEVELS), p=EVENT_SEVERITY_MAP[INT_TO_EVENT_TYPE[etype]])
        for etype in event_types
    ])
    engagement_scores = [EVENT_TYPE_SCORES.get(INT_TO_EVENT_TYPE[etype], 0) for etype in event_types]

    # --- 5) Observed dataframe ---
    observed_df = pd.DataFrame({
        "uid": [uid] * noisy_count,
        "timestamp": timestamps,
        "event_type": event_types,
        "event_severity": event_sev,
        "session_id": [f"{uid}_{getattr(ts, 'date', lambda: ts)()}"] * noisy_count,
        "session_position": list(range(noisy_count)),
        "engagement_score": engagement_scores,
        "cooldown": [int(mech_row["cooldown"])] * noisy_count,
        "value": [float(mech_row["value"])] * noisy_count,
        "rolling_activity": [activity_factor] * noisy_count,
        "recovered": [bool(mech_row.get("recovered", False))] * noisy_count,
    })

    return observed_df
