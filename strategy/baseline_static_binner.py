import numpy as np
import random
from config import *


class BaselineStaticBinner:
    """
    Baseline Profile: Static Risk Binner (Low+ Complexity)
    ------------------------------------------------------
    Computes a simple composite risk score for each user,
    bins into static thresholds, and assigns actions accordingly.
    No adaptation, no memory, no cost-awareness.
    """

    def __init__(self, rng=rng):
        self.rng = rng or np.random.default_rng()

    def compute_static_binner_actions(self, obs_df, rng=None):
        if obs_df.empty:
            return {}

        actions = {}

        for uid, user_df in obs_df.groupby("uid"):
            session_count = user_df["session_id"].nunique()
            avg_engagement = user_df["engagement_score"].mean()
            severity_score = np.mean(user_df["event_severity"] == "high")

            # Composite risk score (0 → safe, 1 → risky)
            risk_score = (1 - avg_engagement) * 0.5 + severity_score * 0.5

            # Decision buckets (static thresholds)
            if risk_score > 0.7:
                action = self.rng.choice(["escalate", "suppress"])
            elif risk_score > 0.4:
                action = self.rng.choice(["nudge", "reinforce", "support"])
            elif risk_score > 0.2:
                action = self.rng.choice(["redirect", "boost"])
            else:
                action = self.rng.choice(["observe", "monitor"])

            actions[uid] = action

        return actions

_policy = BaselineStaticBinner()

def compute_stat_binner_actions(obs_df, rng=rng):
    """Public entrypoint used by runner; consumes *observed* dataframe only."""
    return _policy.compute_static_binner_actions(obs_df, rng=rng)