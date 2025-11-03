import numpy as np
from config import *

class BaselineThresholdOptimizer:
    """
    Baseline Profile: Threshold Optimizer (Medium– Complexity)
    ----------------------------------------------------------
    Uses batch-level quantiles for engagement and severity to
    adapt thresholds dynamically. Still rule-based, no memory,
    no cost-awareness, no ARR awareness.
    """

    def __init__(self, rng=None, high_q=0.25, low_q=0.75):
        """
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator for action sampling.
        high_q : float
            Quantile cutoff for high-risk (default 0.25 → bottom quartile).
        low_q : float
            Quantile cutoff for low-risk (default 0.75 → top quartile).
        """
        self.rng = rng or np.random.default_rng()
        self.high_q = high_q
        self.low_q = low_q

    def compute_baseline_actions(self, obs_df, rng=None):
        if obs_df.empty:
            return {}

        actions = {}

        # Compute quantiles for engagement & severity
        engagement_scores = obs_df.groupby("uid")["engagement_score"].mean()
        severity_scores = (
            obs_df.groupby("uid")["event_severity"].apply(lambda x: np.mean(x == "high"))
        )

        q_low = engagement_scores.quantile(self.high_q)   # low engagement cutoff
        q_high = engagement_scores.quantile(self.low_q)   # high engagement cutoff
        sev_high = severity_scores.quantile(self.low_q)   # "high severity" cutoff

        for uid, user_df in obs_df.groupby("uid"):
            avg_engagement = engagement_scores[uid]
            severity_frac = severity_scores[uid]
            session_count = user_df["session_id"].nunique()

            # Decision rules with adaptive thresholds
            if avg_engagement <= q_low or severity_frac >= sev_high:
                action = self.rng.choice(["escalate", "suppress"])
            elif session_count <= 1:
                action = self.rng.choice(["nudge", "reinforce", "support"])
            elif avg_engagement <= q_high:
                action = self.rng.choice(["redirect", "boost"])
            else:
                action = self.rng.choice(["observe", "monitor"])

            actions[uid] = action

        return actions

_policy = BaselineThresholdOptimizer()

def compute_thresh_opt_actions(obs_df, rng=None):
    """Public entrypoint used by runner; consumes *observed* dataframe only."""
    return _policy.compute_baseline_actions(obs_df, rng=rng)