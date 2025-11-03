import numpy as np
from config import *
from utils.game_tree import VALUE_TIERS

class BaselineARRAware:
    """
    Baseline Profile: ARR-Aware but Forgetful (Medium+ Complexity)
    --------------------------------------------------------------
    Prioritizes interventions based on user ARR/value tier.
    Still ignores infra/energy costs and has no memory of past actions.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def compute_actions(self, obs_df, rng=None):
        if obs_df.empty:
            return {}

        actions = {}

        for uid, user_df in obs_df.groupby("uid"):
            session_count = user_df["session_id"].nunique()
            avg_engagement = user_df["engagement_score"].mean()
            high_severity_frac = np.mean(user_df["event_severity"] == "high")

            # Value awareness
            value_tier = user_df["value"].iloc[-1] if "value" in user_df else "low"
            if value_tier not in VALUE_TIERS:
                value_tier = "low"

            # Simple tier weighting
            if value_tier == "enterprise":
                weight = 1.0
            elif value_tier == "mid":
                weight = 0.7
            else:  # low tier
                weight = 0.4

            # Risk score modulated by value weight
            risk_score = (1 - avg_engagement) * 0.6 + high_severity_frac * 0.4
            weighted_risk = risk_score * weight

            # Decision rules (forgetful: no cross-batch memory)
            if weighted_risk > 0.6:
                action = self.rng.choice(["escalate", "boost"])
            elif weighted_risk > 0.3:
                action = self.rng.choice(["reinforce", "redirect", "support"])
            elif session_count <= 1:
                action = self.rng.choice(["nudge", "support"])
            else:
                action = self.rng.choice(["observe", "remind"])

            actions[uid] = action

        return actions

_policy = BaselineARRAware()

def compute_forgetful_actions(obs_df, rng=None):
    """Public entrypoint used by runner; consumes *observed* dataframe only."""
    return _policy.compute_actions(obs_df, rng=rng)