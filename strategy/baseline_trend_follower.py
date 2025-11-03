import numpy as np
from config import *

class BaselineTrendFollower:
    """
    Baseline Profile: Cost-Blind Trend Follower (Medium Complexity)
    ---------------------------------------------------------------
    Reacts aggressively to engagement/severity trends (slopes).
    Ignores infra/energy costs, overspends on heavy actions when
    negative trends are detected.
    """

    def __init__(self, rng=None, window=3):
        """
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator for action sampling.
        window : int
            Lookback window (number of events) for trend calculation.
        """
        self.rng = rng or np.random.default_rng()
        self.window = window

    def compute_actions(self, obs_df, rng=None):
        if obs_df.empty:
            return {}

        actions = {}

        for uid, user_df in obs_df.groupby("uid"):
            session_count = user_df["session_id"].nunique()
            engagements = user_df["engagement_score"].values

            # Simple slope over last N points (linear regression slope)
            if len(engagements) >= 2:
                x = np.arange(len(engagements[-self.window:]))
                y = engagements[-self.window:]
                slope = np.polyfit(x, y, 1)[0]  # slope
            else:
                slope = 0.0

            avg_engagement = user_df["engagement_score"].mean()
            high_severity_frac = np.mean(user_df["event_severity"] == "high")

            # Decision rules
            if slope < -0.05 or high_severity_frac > 0.5:
                # Negative trend or high severity → overreact
                action = self.rng.choice(["boost", "escalate"])
            elif avg_engagement < 0.3:
                action = self.rng.choice(["nudge", "reinforce", "support"])
            elif session_count <= 1:
                action = self.rng.choice(["redirect", "support"])
            else:
                # Stable/positive → still interventionist (cost-blind)
                action = self.rng.choice(["monitor", "redirect", "boost"])

            actions[uid] = action

        return actions


_policy = BaselineTrendFollower()

def compute_trend_follow_actions(obs_df, rng=None):
    """Public entrypoint used by runner; consumes *observed* dataframe only."""
    return _policy.compute_actions(obs_df, rng=rng)