import numpy as np
import random
from utils.game_tree import *
from config import *

rng = np_rng


class BaselineHeuristicV1:
    """
    Naïve heuristic baseline strategy.
    Buckets users into coarse groups and selects random actions from those sets.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def compute__baseline_actions(self, obs_df, rng=None):
        """
        Compute baseline actions given observed user data.

        Parameters
        ----------
        obs_df : pd.DataFrame
            Observed dataset for the current batch.

        Returns
        -------
        dict
            {uid: chosen_action}
        """
        if obs_df.empty:
            return {}

        actions = {}

        # Group by user
        for uid, user_df in obs_df.groupby("uid"):
            # --- Heuristic features ---
            session_count = user_df["session_id"].nunique()
            avg_engagement = user_df["engagement_score"].mean()
            high_severity_frac = np.mean(user_df["event_severity"] == "high")

            # --- Decision rules ---
            if avg_engagement < 0.3 or high_severity_frac > 0.5:
                # High risk bucket
                action = self.rng.choice(["escalate", "suppress"])
            elif session_count <= 1:
                # Low presence bucket
                action = self.rng.choice(["nudge", "reinforce", "support"])
            elif avg_engagement < 0.6:
                # Moderate bucket
                action = self.rng.choice(["redirect", "boost"])
            else:
                # Stable bucket
                action = self.rng.choice(["observe", "remind"])

            actions[uid] = action

        return actions

_policy = BaselineHeuristicV1()

def compute_v1_actions(obs_df,  rng=rng):
    """Public entrypoint used by runner; consumes *observed* dataframe only."""
    return _policy.compute__baseline_actions(obs_df, rng=rng)