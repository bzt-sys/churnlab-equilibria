"""
challenger.py
-------------
Pluggable wrapper for retention intelligence systems.

Overview
--------
This module defines the `Challenger` interface used by ChurnLab : Equilibria
to evaluate and benchmark *adaptive retention systems* under controlled,
repeatable simulation.

The Challenger acts as an integration bridge between the simulator
and any external retention intelligence model — including, but not limited to,
Virgil, the reference implementation used for testing.

By wrapping each model with a consistent interface (`.reset()` and `.run()`),
the simulator can compare systems fairly across identical user populations,
environmental factors, and cost structures.

Purpose
-------
• Establish a standard way to embed arbitrary retention systems
  (RL agents, heuristics, supervised predictors, etc.) into the simulation loop.
• Demonstrate how an intelligent controller interprets population data,
  assigns strategies, and adapts over time.
• Provide a clear reference for extending the system with new retention AIs.

Example: Adapting Virgil
------------------------
Virgil, the baseline adaptive retention controller, was adapted by providing
a concrete implementation of `.reset()` and `.run(df, uid_col, time_col, **kwargs)`.
The Challenger forwards simulator batches to Virgil, along with the
semantically ordered `ActionGroups`, ensuring reproducible behavior.

To replace Virgil with another system (e.g., a custom RL agent):

```python
from churnlab.challenger import Challenger, ActionGroups
from my_agent.controller import MyRetentionModel

my_agent = MyRetentionModel.load("weights/latest.pt")
wrapper = Challenger(model=my_agent, groups=ActionGroups())"""


import sys
import os
from strategy.virgil.virgil_controller import challenger_instance

MOST = ["observe"]
HIGH = ["monitor", "nudge"]
MODERATE = ["reinforce", "support", "redirect"]
LOW = ["boost", "escalate"]
LEAST = ["suppress"]

class Challenger:
    def __init__(self):
        self.model = challenger_instance

    def reset(self):
        return self.model.reset()
    
    def run(self, df, uid_col=None, time_col=None):
        return self.model.run(
            df,
            uid_col=uid_col,
            time_col=time_col,
            mode="infer",
            most=MOST,
            high=HIGH,
            moderate=MODERATE,
            low=LOW,
            least=LEAST,
        )
