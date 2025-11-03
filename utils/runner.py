"""
ChurnLab : Equilibria — Benchmark Runner
----------------------------------------

Purpose
  Run a head-to-head benchmark between a *challenger* controller (e.g. Virgil)
  and a *baseline* controller under identical event streams, then emit plots and
  summary figures used in the one-pager / paper.

Design notes
  - Minimal surface area: one function `run_batch_loop(...)`.
  - Clear docstrings & type hints; no defensive error scaffolding.
  - Behavior preserved from the original file.

Inputs
  - `challenger`, `baseline` are PopulationBranch-backed agents that expose:
      .generate_batch_rows(ts) -> pd.DataFrame
      .step(interventions: dict, batch_idx: int, influx: bool, events: Any) -> None
      .metrics (dict of arrays)
      .archetypes, .user_states
      .cp_rng (for GPU event sampling)
      .model.run(df, uid_col, time_col) -> {uid: {"strategy": str}}
  - `config` provides TOTAL_BATCHES, BATCHES_PER_DAY, SEED, CONTENT_EVENT_PERIOD.

Outputs
  - Charts: saved to ./output/ via viz_tools.
  - Prints a single "Generating Charts" notice at end.

Usage
  from runner import run_batch_loop
  run_batch_loop(challenger, baseline, config, same_events=True)
"""

from datetime import datetime, timedelta
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from population.PopulationBranch import *
from typing import Any, Dict
from utils.tree_arrays import *
from utils.game_tree import *
from config import *

from strategy.baseline_heuristics_v1 import compute_v1_actions
from strategy.baseline_trend_follower import compute_trend_follow_actions
from strategy.baseline_arr_forgetful import compute_forgetful_actions
from strategy.baseline_threshold_optimizer import compute_thresh_opt_actions
from strategy.challenger import Challenger
from viz.viz_tools import (
    generate_pitch_charts,
    plot_archetype_distributions,
    get_archetype_counts,
    plot_value_tier_distributions, 
    get_value_counts,
    to_cpu_array
)
from viz.viz_prep import *


def run_batch_loop(challenger, baseline, config, same_events=True):
    """
    Run the benchmark loop and export visualizations.

    Parameters
    ----------
    challenger : Challenger
        The adaptive controller (e.g., Virgil) wrapped around a PopulationBranch.
        Must expose .model.run(df, uid_col, time_col) -> {uid: {"strategy": str}}.
    baseline : Any
        The heuristic controller wrapped around a PopulationBranch.
        Must expose .generate_batch_rows(ts), .step(...), .metrics, .archetypes, .user_states.
    config : Any
        Config namespace with:
          - TOTAL_BATCHES: int
          - BATCHES_PER_DAY: int
          - SEED: int
          - CONTENT_EVENT_PERIOD: int
    same_events : bool, default True
        If True, both branches consume the same exogenous event/shock packages
        on “drop” steps (every CONTENT_EVENT_PERIOD). This ensures fair,
        head-to-head comparison.

    Notes
    -----
    - Time advances in fixed "batches" within a day. Each batch produces observations,
      runs each policy, applies actions, and advances the population.
    - At the end, figures are generated via viz_tools.
    """
    
     # Reset challenger model state for a clean run.
    challenger.model.reset()

    # Set up timing for batch windows.
    batch_minutes = 24 * 60 // config.BATCHES_PER_DAY
    start_ts = datetime.now()

    # Snapshot initial distributions for before/after plots.
    challenger_init_arche = get_archetype_counts(challenger.archetypes)
    baseline_init_arche = get_archetype_counts(baseline.archetypes)

    challenger_init_vals = get_value_counts(challenger.user_states)
    baseline_init_vals = get_value_counts(baseline.user_states)

    # --- Main loop ---
    for batch in tqdm(range(config.TOTAL_BATCHES)):
        ts = start_ts + timedelta(minutes=batch * batch_minutes)

        # On content “drop” steps, optionally synchronize exogenous packages.
        shared_pkgs = None
        is_drop = (batch % CONTENT_EVENT_PERIOD) == 0
        if same_events and is_drop:
            # shape: (K, 6) with [att, nov, resp, dur, remaining, decay]
            shared_pkgs = sample_content_packages_gpu(num_pkgs=1, rng=challenger.cp_rng)

        # === Challenger: observe → act ===
        chal_obs = challenger.generate_batch_rows(ts)
        if chal_obs.empty:
            continue

        chal_result = challenger.model.run(
            df=chal_obs,
            uid_col="uid",
            time_col="timestamp",
        )
        actions_challenger: Dict[str, str] = {uid: v["strategy"] for uid, v in chal_result.items()}

        # === Baseline: observe → act ===
        base_obs = baseline.generate_batch_rows(ts)
        if base_obs.empty:
            continue

        actions_baseline: Dict[str, str] = compute_thresh_opt_actions(base_obs)

        # === Apply actions and advance world ===
        challenger.step(
            interventions=actions_challenger,
            batch_idx=batch,
            influx=True,
            events=shared_pkgs,
        )
        baseline.step(
            interventions=actions_baseline,
            batch_idx=batch,
            influx=True,
            events=shared_pkgs,
        )

    # --- Post-run aggregation & visualization ---

    # Final distributions for before/after views.
    challenger_final_arche = get_archetype_counts(challenger.archetypes)
    baseline_final_arche = get_archetype_counts(baseline.archetypes)

    challenger_final_vals = get_value_counts(challenger.user_states)
    baseline_final_vals = get_value_counts(baseline.user_states)

    # Strategy/state counts (convert to named dicts for viz)
    chal_strat_counts = counts_to_named_dicts(challenger.metrics["strategy_counts"])
    base_strat_counts = counts_to_named_dicts(baseline.metrics["strategy_counts"])

    chal_state_counts = counts_to_state_dicts(challenger.metrics["state_counts"])
    base_state_counts = counts_to_state_dicts(baseline.metrics["state_counts"])

    # Metrics to CPU, then into viz-ready tensors
    chal_metrics = {k: to_cpu_array(v) for k, v in challenger.metrics.items()}
    base_metrics = {k: to_cpu_array(v) for k, v in baseline.metrics.items()}
    chal_inputs = build_viz_inputs(chal_metrics)
    base_inputs = build_viz_inputs(base_metrics)

    # Charts
    os.makedirs("output", exist_ok=True)
    print("Generating charts → ./output")

    generate_pitch_charts(
        chal_inputs,
        base_inputs,
        challenger_strat_counts=chal_strat_counts,
        base_strat_counts=base_strat_counts,
        challenger_state_counts=chal_state_counts,
        base_state_counts=base_state_counts,
        seed=SEED,
    )

    plot_archetype_distributions(
        challenger_init_arche,
        challenger_final_arche,
        baseline_init_arche,
        baseline_final_arche,
        save_path=os.path.join("output", f"chart_archetypes_seed{SEED}.png"),
    )

    plot_value_tier_distributions(
        challenger_init_vals,
        challenger_final_vals,
        baseline_init_vals,
        baseline_final_vals,
        save_path=os.path.join("output", f"chart_value_tiers_seed{SEED}.png"),
    )
        