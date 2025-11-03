"""
PopulationBranch
----------------
GPU-accelerated population engine for ChurnLab : Equilibria.

Responsibilities
  • Maintain per-user state on GPU (CuPy arrays)
  • Vectorized add/remove of users (influx / churn)
  • Apply actions + content-package effects each step
  • Track per-batch KPIs (cost-adjusted value, churn, health, state mix)
  • Emit per-user observations for controllers to act on

Notes
  - Public API preserved:
      reserve_uids(k), add_users(uids, ts), remove_users(mask),
      generate_batch_rows(ts) -> pd.DataFrame, step(interventions, batch_idx, influx=True, events=None)
  - Minimal control flow; no extra exceptions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import math
import numpy as np
import cupy as cp
import pandas as pd

from config import *
from utils.tree_arrays import *   # states, actions, archetypes, events, content pkgs
from population.row_generator import generate_mechanism_rows_gpu, generate_rows_for_user
from population.influx import compute_user_influx_rate_gpu


class PopulationBranch:
    """
    A single simulated "world branch" (e.g., baseline vs challenger) with identical
    initial conditions and shared exogenous event streams for fair comparisons.

    Parameters
    ----------
    name : str
        Branch label for charts/logs.
    model : Any, optional
        Attached controller (e.g., Virgil). May be None for heuristic baselines.

    Attributes
    ----------
    cp_rng : cp.random.RandomState
        GPU RNG seeded from config.SEED (stable multi-seed runs).
    active_mask : cp.ndarray[bool]
        Marks active (alive) slots.
    metrics : dict
        Per-batch KPIs and counts (numpy / CPU for serialization).
    """

    def __init__(self, name: str, model=None):
        self.name = name
        self.model = model

        # --- RNGs (single initialization) ---
        self.np_rng = np.random.default_rng(SEED)
        self.cp_rng = cp.random.RandomState(SEED)

        # --- Capacity, indexing, IDs ---
        self.max_users = MAX_USERS
        self.n_users = NUM_USERS
        self.user_ids = cp.full(MAX_USERS, -1, dtype=cp.int64)  # reserved; not all branches need it

        # --- GPU state arrays (pre-allocated) ---
        self.user_states     = cp.full(MAX_USERS, -1, dtype=cp.int64)   # stores uid (historical naming kept)
        self.health          = cp.zeros(MAX_USERS, dtype=cp.float32)
        self.engagement      = cp.zeros(MAX_USERS, dtype=cp.float32)
        self.fatigue         = cp.zeros(MAX_USERS, dtype=cp.float32)
        self.cooldown        = cp.zeros(MAX_USERS, dtype=cp.int16)
        self.strategy_factor = cp.ones(MAX_USERS, dtype=cp.float32)
        self.state           = cp.full(MAX_USERS, -1, dtype=cp.int8)
        self.archetypes      = cp.empty(MAX_USERS, dtype=cp.int8)
        self.value           = cp.zeros(MAX_USERS, dtype=cp.float32)

        self.active_mask     = cp.zeros(MAX_USERS, dtype=cp.bool_)  # start all free
        self.burnout_counter = cp.zeros(MAX_USERS, dtype=cp.int16)
        self.churned_mask    = cp.zeros(MAX_USERS, dtype=cp.bool_)
        self.active_packages = cp.empty((0, 6), dtype=cp.float32)   # [att, nov, resp, dur, remaining, decay]

        # --- Seed initial population in one vectorized call ---
        init_uids = np.arange(NUM_USERS, dtype=np.int64)
        rows = generate_mechanism_rows_gpu(init_uids, datetime.now(), rng=self.cp_rng)
        idxs = slice(0, NUM_USERS)

        self.user_states[idxs]     = rows["uid"]
        self.health[idxs]          = rows["user_health"]
        self.engagement[idxs]      = rows["engagement"]
        self.cooldown[idxs]        = rows["cooldown"]
        self.strategy_factor[idxs] = rows["strategy_factor"]
        self.state[idxs]           = rows["state"]
        self.archetypes[idxs]      = rows["archetype"]
        self.value[idxs]           = rows["value"]
        self.active_mask[idxs]     = True

        self.next_uid = int(init_uids[-1]) + 1

        # --- Per-batch metrics (CPU-backed for easy serialization) ---
        self.metrics = {
            "energy": [0.0] * TOTAL_BATCHES,
            "arr": [0.0] * TOTAL_BATCHES,
            "infra_costs": [0.0] * TOTAL_BATCHES,
            "net_revenue": [0.0] * TOTAL_BATCHES,
            "mean_engagement": [0.0] * TOTAL_BATCHES,
            "mean_health": [0.0] * TOTAL_BATCHES,
            "rolling_churn": [0.0] * TOTAL_BATCHES,
            "penalties": [0.0] * TOTAL_BATCHES,
            "comebacks": [0] * TOTAL_BATCHES,
            "alive_users": [0] * TOTAL_BATCHES,
            "churned_users": [0] * TOTAL_BATCHES,
            "churn_rate": [0.0] * TOTAL_BATCHES,
            "content_attention": [0.0] * TOTAL_BATCHES,
            "content_novelty": [0.0] * TOTAL_BATCHES,
            "content_response": [0.0] * TOTAL_BATCHES,
            "package_list": {},  # kept for compatibility
            "strategy_counts": [np.zeros(len(ACTION_TO_INT), dtype=np.int32) for _ in range(TOTAL_BATCHES)],
            "state_counts": [np.zeros(NUM_STATES, dtype=np.int32) for _ in range(TOTAL_BATCHES)],
        }

        # convenient matrix view if your viz expects it
        self.strategy_counts = np.zeros((TOTAL_BATCHES, len(ACTION_TO_INT)), dtype=np.int32)

        # cache for “comebacks” metric
        self.prev_user_health = cp.ones_like(self.health)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def reserve_uids(self, k: int) -> List[int]:
        """Return `k` fresh, never-reused UIDs and advance the counter."""
        if k <= 0:
            return []
        start = self.next_uid
        self.next_uid += k
        return list(range(start, start + k))

    def add_users(self, uids: Iterable[int], ts: datetime) -> None:
        """Vectorized add: fill free slots with new rows for given UIDs."""
        free_idxs_host = cp.where(~self.active_mask)[0].get()
        if free_idxs_host.size == 0:
            return

        uids = list(uids)
        num_to_add = min(len(uids), free_idxs_host.size)
        if num_to_add == 0:
            return

        rows = generate_mechanism_rows_gpu(uids[:num_to_add], ts, rng=self.cp_rng)
        idxs = free_idxs_host[:num_to_add]

        self.user_states[idxs]     = rows["uid"]
        self.health[idxs]          = rows["user_health"]
        self.engagement[idxs]      = rows["engagement"]
        self.cooldown[idxs]        = rows["cooldown"]
        self.strategy_factor[idxs] = rows["strategy_factor"]
        self.state[idxs]           = rows["state"]
        self.archetypes[idxs]      = rows["archetype"]
        self.value[idxs]           = rows["value"]
        self.active_mask[idxs]     = True
        self.churned_mask[idxs]    = False
        self.burnout_counter[idxs] = 0

    def remove_users(self, remove_mask: cp.ndarray) -> None:
        """
        Vectorized remove: marks slots free and resets per-slot state.

        Parameters
        ----------
        remove_mask : cp.ndarray[bool] of shape (MAX_USERS,)
            True where user should be removed.
        """
        idxs = cp.where(remove_mask & self.active_mask)[0]
        if idxs.size == 0:
            return

        # mark inactive
        self.active_mask[idxs]  = False
        self.churned_mask[idxs] = True

        # clear state (makes free slots obvious)
        self.health[idxs]          = 0.0
        self.engagement[idxs]      = 0.0
        self.cooldown[idxs]        = 0
        self.strategy_factor[idxs] = 1.0
        self.state[idxs]           = 0
        self.archetypes[idxs]      = 0
        self.value[idxs]           = 0.0
        self.burnout_counter[idxs] = 0
        self.user_ids[idxs]        = -1

    def alive_uids(self) -> cp.ndarray:
        """Return GPU array of UIDs for active users."""
        return self.user_states[self.active_mask]

    def get_metrics(self):
        """Return (metrics_dict, strategy_counts_matrix) for compatibility."""
        return self.metrics, self.strategy_counts

    def generate_batch_rows(self, ts: datetime) -> pd.DataFrame:
        """
        Build per-user observation rows on CPU for the controller.

        Returns
        -------
        DataFrame with columns:
          uid, timestamp, archetype, user_health, engagement,
          cooldown, state, value, recovered
        """
        obs_dfs: List[pd.DataFrame] = []
        active_idxs = cp.where(self.active_mask)[0].get()

        for idx in active_idxs:
            # pull scalars to host; keep schema stable
            uid = int(self.user_states[idx].get())
            row = {
                "uid": uid,
                "timestamp": ts,
                "archetype": int(self.archetypes[idx].get()),
                "user_health": float(self.health[idx].get()),
                "engagement": float(self.engagement[idx].get()),
                "cooldown": int(self.cooldown[idx].get()),
                "state": int(self.state[idx].get()),
                "value": float(self.value[idx].get()),
                "recovered": False,
            }
            df = generate_rows_for_user(uid, ts, row, rng=self.np_rng)
            if not df.empty:
                obs_dfs.append(df)

        return pd.concat(obs_dfs, ignore_index=True) if obs_dfs else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Core simulation step
    # ------------------------------------------------------------------ #

    def step(self, interventions: Dict[int, str], batch_idx: int, influx: bool = True, events=None) -> None:
        """
        Advance the world by one batch.

        Parameters
        ----------
        interventions : dict[int, str]
            Map of uid -> strategy/action name.
        batch_idx : int
            Current batch index (0..TOTAL_BATCHES-1).
        influx : bool, default True
            Whether to allow population growth this step.
        events : optional
            Shared exogenous content-package array for fair head-to-head runs.
        """
        # === Content packages (GPU) ===
        if (batch_idx % CONTENT_EVENT_PERIOD) == 0:
            if events is not None:
                pkgs = cp.asarray(events, dtype=self.active_packages.dtype)
                if pkgs.size > 0:
                    self.active_packages = pkgs if self.active_packages.size == 0 else cp.vstack([self.active_packages, pkgs])
                self.metrics["content_attention"][batch_idx] = float(pkgs[0, 0].item())
                self.metrics["content_novelty"][batch_idx]   = float(pkgs[0, 1].item())
                self.metrics["content_response"][batch_idx]  = float(pkgs[0, 2].item())
            else:
                new_pkgs = sample_content_packages_gpu(num_pkgs=1, rng=self.cp_rng)
                self.active_packages = new_pkgs if self.active_packages.size == 0 else cp.vstack([self.active_packages, new_pkgs])
                self.metrics["content_attention"][batch_idx] = float(new_pkgs[0, 0].item())
                self.metrics["content_novelty"][batch_idx]   = float(new_pkgs[0, 1].item())
                self.metrics["content_response"][batch_idx]  = float(new_pkgs[0, 2].item())

        # === Influx (weekly cadence; capacity-aware) ===
        if influx and (batch_idx % (BATCHES_PER_DAY * 7) == 0) and int(self.active_mask.sum().item()) < MAX_USERS:
            alive = self.active_mask
            influx_rate = compute_user_influx_rate_gpu(
                self.health[alive],
                self.engagement[alive],
                self.archetypes[alive],
                self.metrics["energy"][batch_idx],
                batch_idx,
            )
            alive_count = int(alive.sum().item())
            num_influx = int(influx_rate * alive_count)

            free_idxs = cp.where(~self.active_mask)[0].get()
            if num_influx > 0 and free_idxs.size > 0:
                num_influx = min(num_influx, free_idxs.size)
                new_uids = self.reserve_uids(num_influx)
                self.add_users(new_uids, datetime.now())

        # === Early exit if no active users ===
        alive = self.active_mask
        idxs = cp.where(alive)[0]
        if idxs.size == 0:
            return

        # === Actions (host→device) ===
        uid_host = self.user_states[idxs].get()
        act_host = [ACTION_TO_INT.get(interventions.get(int(uid), "observe"), ACTION_TO_INT["observe"]) for uid in uid_host]
        actions = cp.asarray(act_host, dtype=cp.int32)

        actions_host = actions.get()
        self.metrics["strategy_counts"][batch_idx] = np.bincount(actions_host, minlength=len(ACTION_TO_INT))

        # === Content effects (GPU) ===
        self.health[idxs], self.engagement[idxs], self.fatigue[idxs], self.strategy_factor[idxs] = apply_content_effects_gpu(
            self.health[idxs], self.engagement[idxs], self.fatigue[idxs], self.strategy_factor[idxs],
            self.archetypes[idxs], self.active_packages
        )

        # === Action & state effects ===
        N = idxs.size
        decay_shift  = sample_effect_array("decay_shift", actions, rng=self.cp_rng)
        engage_shift = sample_effect_array("engagement_shift", actions, rng=self.cp_rng)

        rows = STATE_ARRAYS[self.state[idxs].get()]  # (N, …)
        decay_mult  = cp.asarray(rows[:, -2])
        engage_mult = cp.asarray(rows[:, -1])

        arch_idx     = self.archetypes[idxs]
        fatigue_mult = ARCH_ARRAY[arch_idx, ARCH_ATTRS["fatigue_mult"]]
        health_mult  = ARCH_ARRAY[arch_idx, ARCH_ATTRS["user_health_mult"]]

        # Optional creep for selected actions
        obs_id, boost_id, esc_id = ACTION_TO_INT["observe"], ACTION_TO_INT["boost"], ACTION_TO_INT["escalate"]
        creep_mask = (actions == obs_id) | (actions == boost_id) | (actions == esc_id)
        creep_shift = cp.zeros(N, dtype=cp.float32)
        if int(creep_mask.sum().item()) > 0:
            creep_shift[creep_mask] = sample_effect_array("creep", actions[creep_mask], rng=self.cp_rng)

        decay_effect = (decay_shift * decay_mult) + (self.engagement[idxs] * (engage_mult)) + creep_shift

        # Engagement effect with sign preserved
        sign = cp.sign(engage_shift)
        magnitude = cp.abs(engage_shift) * engage_mult * self.strategy_factor[idxs]
        engage_effect = sign * magnitude

        # Update health (decay then recovery)
        new_health = self.health[idxs] - decay_effect * (1 - self.health[idxs])
        new_health = cp.clip(new_health, 0.0, 1.0)

        # Engagement dynamics
        cycle_factor = 1.0 + FATIGUE_CYCLE_AMPLITUDE * math.sin(2 * math.pi * batch_idx / FATIGUE_CYCLE_PERIOD)
        engage_mask = (fatigue_mult >= 0.01) & (fatigue_mult <= 0.05)
        raw_engage_update = engage_effect * fatigue_mult * cycle_factor
        engage_update = cp.where(engage_mask, cp.abs(raw_engage_update) + 1.0, raw_engage_update)
        new_engage = cp.minimum(1.0, self.engagement[idxs] + engage_update)

        # Health recovery peaks around OPTIMAL_ENGAGE
        engage_gain_mult = cp.exp(-((new_engage - OPTIMAL_ENGAGE) ** 2) / (2 * ENGAGE_WIDTH ** 2))
        engage_gain_mult = MAX_GAIN_MULT * engage_gain_mult

        # Burnout & recovery
        burn_inc = (new_engage > ENGAGE_THRESH).astype(cp.int16) * 2
        burn_dec = (new_engage < ENGAGE_FLOOR).astype(cp.int16)
        self.burnout_counter[idxs] = cp.maximum(0, self.burnout_counter[idxs] + burn_inc - burn_dec)

        health_recovery = health_mult * engage_gain_mult
        health_recovery /= (1.0 + cp.log1p(self.burnout_counter[idxs]).astype(cp.float32))
        new_health = cp.minimum(1.0, new_health + health_recovery)

        # Small stochastic buff
        buff_mask = self.cp_rng.rand(N) < 0.03
        new_health = cp.where(buff_mask, cp.minimum(new_health * 1.25, 1.0), new_health)

        # Comebacks metric (was <0.4 → >0.6)
        comeback_mask = (self.prev_user_health[idxs] < 0.4) & (new_health > 0.6)
        self.metrics["comebacks"][batch_idx] += int(comeback_mask.sum().item())
        self.prev_user_health[idxs] = new_health

        # Write back
        self.health[idxs]     = new_health
        self.engagement[idxs] = new_engage

        # Next state sampling (GPU)
        next_states = sample_next_states(
            self.state[idxs],
            self.health[idxs],
            self.fatigue[idxs],
            decay_effect,
            new_engage,
            self.burnout_counter[idxs],
            rng=self.cp_rng,
        )
        self.state[idxs] = next_states

        # State-counts snapshot (CPU for charts)
        active_states = self.state[self.active_mask]
        counts_gpu = cp.bincount(active_states, minlength=NUM_STATES)
        self.metrics["state_counts"][batch_idx] = counts_gpu.get()

        # Remove near-dead users; free slots
        self.remove_users((self.health < 0.01) & self.active_mask)

        # Energy & ARR (per batch)
        batch_energy = cp.sum(ACTION_COST[actions]) / float(BATCHES_PER_DAY * 30)
        self.metrics["energy"][batch_idx] += float(batch_energy.item())
        self.metrics["arr"][batch_idx]    += float(cp.sum(self.value[idxs]).item())

        # Live counts, means, rolling churn
        alive_count   = int(cp.sum(self.active_mask).item())
        churned_count = int(cp.sum(self.churned_mask).item())
        self.metrics["alive_users"][batch_idx]   = alive_count
        self.metrics["churned_users"][batch_idx] = churned_count
        self.metrics["churn_rate"][batch_idx]    = churned_count / max(1, alive_count)

        if alive_count > 0:
            self.metrics["mean_health"][batch_idx]    = float(cp.mean(self.health[self.active_mask]).item())
            self.metrics["mean_engagement"][batch_idx]= float(cp.mean(self.engagement[self.active_mask]).item())

        if batch_idx >= 1:
            start = max(0, batch_idx - ROLLING_WINDOW)
            churned_delta = self.metrics["churned_users"][batch_idx] - self.metrics["churned_users"][start]
            alive_start   = self.metrics["alive_users"][start]
            self.metrics["rolling_churn"][batch_idx] = churned_delta / max(1, alive_start)
