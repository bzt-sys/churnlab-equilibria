# baseline_heuristics.py
import numpy as np
import pandas as pd
from utils.game_tree import *
from config import *

# Per-action “energy/cost” (tweak freely to shape behavior)
ACTION_COST = COSTLY_ACTIONS

# Value weighting from the *observed* column `value` (be robust to str/int)
VALUE_WEIGHTS = {
    "low": 1.0, "mid": 1.3, "high": 1.7,
    1: 1.0, 2: 1.15, 3: 1.3, 4: 1.5, 5: 1.7,
}

class HeuristicBaseline:
    """
    Quantile-adaptive, cost-aware, lightly stateful baseline operating *only* on observed data.
    - Remembers per-user last engagement and last action (no mechanism peeking).
    - Uses batch quantiles so thresholds adapt to drift/difficulty.
    - Trades off expected benefit vs action cost; small epsilon exploration.
    """
    def __init__(self, epsilon=0.05, lambda_cost=0.15):
        self.epsilon = float(epsilon)
        self.lambda_cost = float(lambda_cost)
        self.user_memory = {}  # uid -> {"last_eng": float, "last_action": str}

    # ---------- helpers ----------
    def _value_weight(self, v):
        if v in VALUE_WEIGHTS:
            return VALUE_WEIGHTS[v]
        try:
            return VALUE_WEIGHTS.get(int(v), 1.0)
        except Exception:
            return 1.0

    def _high_frac(self, s):
        # fraction of "high" severity in observed `event_severity`
        return np.mean(pd.Series(s).astype(str).str.lower().eq("high"))

    def _trend(self, user_df: pd.DataFrame):
        # slope of engagement vs session_position within the observed batch rows
        x = user_df["session_position"].to_numpy()
        y = user_df["engagement_score"].to_numpy()
        if x.size < 2 or np.all(x == x[0]):
            return 0.0
        # polyfit is robust for small n; returns slope in engagement/position
        return float(np.polyfit(x, y, 1)[0])

    def _per_uid_metrics(self, obs_df: pd.DataFrame):
        # Aggregate strictly from observed columns
        grp = obs_df.groupby("uid").agg(
            eng_mean=("engagement_score", "mean"),
            sev_high=("event_severity", self._high_frac),
            sessions=("session_id", "nunique"),
            cooldown=("cooldown", "mean"),
            activity=("rolling_activity", "mean"),
            value=("value", "last"),
            recovered=("recovered", "last"),
        ).reset_index()

        # Batch quantiles (adaptive thresholds)
        q = {
            "eng_q25": grp["eng_mean"].quantile(0.25),
            "eng_q75": grp["eng_mean"].quantile(0.75),
            "sev_q75": grp["sev_high"].quantile(0.75),
            "sess_q25": grp["sessions"].quantile(0.25),
        }
        return grp.set_index("uid"), q

    # ---------- main ----------
    def decide(self, obs_df: pd.DataFrame, rng=None):
        if obs_df is None or obs_df.empty:
            return {}
        rng = rng or np.random.default_rng()

        per_uid, q = self._per_uid_metrics(obs_df)
        actions = {}

        for uid, user_df in obs_df.groupby("uid"):
            row = per_uid.loc[uid]
            eng = float(row.eng_mean)
            sev_high = float(row.sev_high)
            sess = int(row.sessions)
            cooldown = float(row.cooldown) if pd.notna(row.cooldown) else 0.0
            activity = float(row.activity) if pd.notna(row.activity) else 0.0
            recovered = bool(row.recovered) if pd.notna(row.recovered) else False
            trend = self._trend(user_df)
            val_w = self._value_weight(row.value)

            # Risk signal from *observables* only (all bounded & normalized)
            risk = (
                (q["eng_q75"] - eng) / max(1e-6, q["eng_q75"]) * 0.40 +
                (sev_high) / max(1e-6, q["sev_q75"] or 1e-6) * 0.30 +
                (-min(trend, 0.0)) * 0.20 +
                (1.0 if cooldown > 0.6 else 0.0) * 0.10 +
                (1.0 if sess <= q["sess_q25"] else 0.0) * 0.10
            )
            risk = float(np.clip(risk, 0.0, 3.0))

            # Expected benefit templates per action (purely from observed signals)
            benefit = {
                "observe":   (0.05 if eng >= q["eng_q75"] else 0.0) - 0.10 * risk,
                "monitor":   0.05 + 0.10 * max(0.0, eng - q["eng_q25"]),
                "nudge":     0.20 * max(0.0, q["sess_q25"] - sess) + 0.10 * max(0.0, q["eng_q75"] - eng),
                "reinforce": 0.25 * max(0.0, trend) + (0.10 if eng >= q["eng_q75"] else 0.0),
                "support":   0.30 * risk + (0.10 if sev_high >= (q["sev_q75"] or 0)/2 else 0.0),
                "redirect":  0.20 * (sev_high >= (q["sev_q75"] or 0)/2) + 0.15 * (trend < 0.0),
                "boost":     0.30 * (eng >= q["eng_q25"]) * (trend >= 0.0) * (activity >= 0.5),
                "suppress":  0.35 * (sev_high > (q["sev_q75"] or 0)) + 0.10 * (trend < 0.0),
                "escalate":  0.60 * (risk > 0.9) + 0.20 * (sev_high > (q["sev_q75"] or 0)),
            }

            # Optional small bonus if the user appears "recovered"
            if recovered:
                benefit["reinforce"] = benefit.get("reinforce", 0.0) + 0.05
                benefit["observe"] = benefit.get("observe", 0.0) + 0.02

            utilities = {
                a: val_w * benefit[a] - self.lambda_cost * ACTION_COST[a]
                for a in ACTIONS if a in benefit
            }

            # Memory bias: if last action seemed to coincide with higher eng, prefer to repeat
            mem = self.user_memory.get(uid)
            if mem and "last_eng" in mem and "last_action" in mem:
                delta = eng - float(mem["last_eng"])
                if delta > 0.05:
                    la = mem["last_action"]
                    utilities[la] = utilities.get(la, 0) + 0.10 * val_w

            # ε-greedy
            if rng.random() < self.epsilon:
                action = rng.choice(ACTIONS)
            else:
                action = max(utilities.items(), key=lambda kv: kv[1])[0]

            actions[uid] = action
            self.user_memory[uid] = {"last_eng": eng, "last_action": action}

        return actions

# Module-level policy so state persists across batches without touching runner.py
_policy = HeuristicBaseline()

def compute_baseline_actions(obs_df: pd.DataFrame, rng=rng):
    """Public entrypoint used by runner; consumes *observed* dataframe only."""
    return _policy.decide(obs_df, rng=rng)
