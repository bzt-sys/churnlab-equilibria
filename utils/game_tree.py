import random
from config import *
from utils.tree_arrays import *
# --- Game Tree State Machine ---
# cascade_states.py

# 4-Level Cascade Game Tree
# Normalized to lowercase naming conventions

PROBABILISTIC_RULES = {
      # --- Lifeline boosts ---
    ("d4", 'lifeline_1'): [("d3", 0.6), ("d4", 0.35), ("stable", 0.05)],  
        # at churn edge, support usually helps, but not always
    
    ("e4", "lifeline_2"):   [("e2", 0.5), ("e3", 0.3), ("e4", 0.2)],  
        # at peak escalation, boost *often* pulls them down, but not guaranteed
    # --- Degrade branch ---
    ("d1", "landmine_1"): [("stable", 0.4), ("d1", 0.4), ("d2", 0.2)],
    ("d2", "landmine_2"):   [("d1", 0.5), ("d2", 0.3), ("d3", 0.2)],
    ("d3", "landmine_3"):     [("d4", 0.6), ("d3", 0.3), ("d2", 0.1)],  # misplaced boost
    ("d3", "landmine_4"):   [("d2", 0.4), ("d3", 0.5), ("d4", 0.1)],  # neglect at low point risky

    # --- Escalate branch ---
    ("e1", "erratic_1"):     [("e2", 0.5), ("e1", 0.3), ("stable", 0.2)], 
    ("e2", "erratic_2"):   [("e1", 0.5), ("e2", 0.4), ("e3", 0.1)],   # ignoring at mid-escalation
    ("e2", "erratic_3"):     [("e3", 0.5), ("e2", 0.4), ("stable", 0.1)],
    ("e3", "erratic_4"):  [("d1", 0.6), ("e2", 0.2), ("e3", 0.2)],   # suppression backfires hard
    ("e3", "erratic_5"):   [("e1", 0.5), ("e2", 0.3), ("e3", 0.2)],
}



GAME_TREE = {
    "stable": {
        "transitions": {
            "degrade": {"next": "d1", "chance": 0.2},   # placeholder
            "escalate": {"next": "e1", "chance": 0.2},  # placeholder
            "stay": {"next": "stable", "chance": 0.6},  # placeholder
        }
    },

    # --- Degrade Branch ---
    "d1": {
        "transitions": {
            "degrade": {"next": "d2", "chance": 0.37},
            "recover": {"next": "stable", "chance": 0.28},
            "stay": {"next": "d1", "chance": 0.35},
        }
    },
    "d2": {
        "transitions": {
            "degrade": {"next": "d3", "chance": 0.4},
            "recover": {"next": "d1", "chance": 0.2},
            "stay": {"next": "d2", "chance": 0.3},
        }
    },
    "d3": {
        "transitions": {
            "degrade": {"next": "d4", "chance": 0.45},
            "recover": {"next": "d2", "chance": 0.25},
            "stay": {"next": "d3", "chance": 0.3},
        }
    },
    "d4": {
        "transitions": {
            "recover": {"next": "d3", "chance": 0.2},
            "stay": {"next": "d4", "chance": 0.45},
            "stabilize": {"next": "stable", "chance": 0.05},
            "degrade": {"next": "d5", "chance": 0.30}
            # no further degrade — presence decay itself handles churn
        }
    },

      "d5": {
        "transitions": {
            "degrade": {"next": "d6", "chance": 0.45},     # strong pull deeper
            "recover": {"next": "d4", "chance": 0.1},     # slim chance of stepping back
            "stay": {"next": "d5", "chance": 0.45},
        }
    },
    "d6": {
        "transitions": {
            "recover": {"next": "d5", "chance": 0.05},      # rare partial recovery
            "stay": {"next": "d6", "chance": 0.95}
        }
    },

    # --- Escalate Branch ---
    "e1": {
        "transitions": {
            "escalate": {"next": "e2", "chance": 0.33},
            "stabilize": {"next": "stable", "chance": 0.33},
            "stay": {"next": "e1", "chance": 0.34},
        }
    },
    "e2": {
        "transitions": {
            "escalate": {"next": "e3", "chance": 0.3},
            "stabilize": {"next": "stable", "chance": 0.37},
            "stay": {"next": "e2", "chance": 0.33},
        }
    },
    "e3": {
        "transitions": {
            "escalate": {"next": "e4", "chance": 0.2},
            "stabilize": {"next": "stable", "chance": 0.3},
            "stay": {"next": "e3", "chance": 0.45},
            "degrade": {"next": "d1", "chance": 0.05}
        }
    },
    "e4": {
        "transitions": {
            "stabilize": {"next": "stable", "chance": 0.5},
            "stay": {"next": "e4", "chance": 0.25},
            "degrade": {"next": "d2", "chance": 0.25}
            # no further escalate — diminishing returns cap
        }
    },
}

STATE_MULTIPLIERS = {
    "stable":   {"decay_mult": -0.008, "engage_mult": +0.01},

    "d1":       {"decay_mult": -0.05, "engage_mult":  0.02},
    "d2":       {"decay_mult": -0.08, "engage_mult": -0.05},
    "d3":       {"decay_mult": -0.13, "engage_mult": -0.06},
    "d4":       {"decay_mult": -0.21, "engage_mult": -0.07},
    "d5":       {"decay_mult": -0.34, "engage_mult": -0.17},
    "d6":       {"decay_mult": -0.55, "engage_mult": -0.25},

    "e1":       {"decay_mult":  0.00, "engage_mult": +0.02},
    "e2":       {"decay_mult": -0.06, "engage_mult": +0.03},
    "e3":       {"decay_mult": -0.09, "engage_mult": +0.04},
    "e4":       {"decay_mult": -0.13, "engage_mult": +0.06},
}



# Array of effects, same order as ACTIONS above
ACTION_EFFECTS_ARRAY = [
    # observe
    {
        "decay_shift": {"mean": +0.07, "std": 0.02, "outlier_chance": 0.01, "outlier_scale": 2.0},
        "engagement_shift": {"mean": -0.095, "std": 0.03, "outlier_chance": 0.01, "outlier_scale": 1.5},
        "creep": {"mean": 0.025, "std": 0.008, "outlier_chance": 0.01, "outlier_scale": 1.2},
    },
    # monitor
    {
        "decay_shift": {"mean": -0.025, "std": 0.01, "outlier_chance": 0.02, "outlier_scale": 1.5},
        "engagement_shift": {"mean": +0.05, "std": 0.02, "outlier_chance": 0.01, "outlier_scale": 2.0},
    },
    # nudge
    {
        "decay_shift": {"mean": -0.05, "std": 0.02, "outlier_chance": 0.03, "outlier_scale": 2.0},
        "engagement_shift": {"mean": +0.15, "std": 0.05, "outlier_chance": 0.02, "outlier_scale": 1.8},
    },
    # reinforce
    {
        "decay_shift": {"mean": -0.125, "std": 0.03, "outlier_chance": 0.02, "outlier_scale": 1.5},
        "engagement_shift": {"mean": +0.30, "std": 0.07, "outlier_chance": 0.01, "outlier_scale": 2.0},
    },
    # support
    {
        "decay_shift": {"mean": -0.185, "std": 0.05, "outlier_chance": 0.03, "outlier_scale": 2.0},
        "engagement_shift": {"mean": +0.20, "std": 0.08, "outlier_chance": 0.02, "outlier_scale": 2.5},
    },
    # redirect
    {
        "decay_shift": {"mean": -0.20, "std": 0.06, "outlier_chance": 0.04, "outlier_scale": 2.2},
        "engagement_shift": {"mean": +0.17, "std": 0.05, "outlier_chance": 0.03, "outlier_scale": 2.0},
    },
    # boost
    {
        "decay_shift": {"mean": -0.12, "std": 0.06, "outlier_chance": 0.05, "outlier_scale": 3.0},
        "engagement_shift": {"mean": +0.16, "std": 0.9, "outlier_chance": 0.05, "outlier_scale": 2.75},
        "creep": {"mean": 0.05, "std": 0.03, "outlier_chance": 0.01, "outlier_scale": 1.5},
    },
    # escalate
    {
        "decay_shift": {"mean": +0.075, "std": 0.08, "outlier_chance": 0.07, "outlier_scale": 4.0},
        "engagement_shift": {"mean": +0.25, "std": 0.15, "outlier_chance": 0.07, "outlier_scale": 3.0},
        "creep": {"mean": 0.05, "std": 0.03, "outlier_chance": 0.01, "outlier_scale": 1.5},
    },
    # suppress
    {
        "decay_shift": {"mean": +0.175, "std": 0.06, "outlier_chance": 0.04, "outlier_scale": 2.5},
        "engagement_shift": {"mean": -0.075, "std": 0.04, "outlier_chance": 0.02, "outlier_scale": 2.0},
    },
]

# --- User Archetype Definitions ---
ARCHETYPES = {
 'Steady Performer': {
     'user_health_mult': 1.02,
     'fatigue_mult': 0.07,
     'row_mean': 19,
     'volatility': 0.07,
     'cooldown': 8,
     "attention_sens": 0.4,
    "novelty_sens": 0.3,
    "response_sens": 0.5,
    "flat_decay": 0.02
   
 },
 'At-Risk Minimalist': {
    'user_health_mult': 0.61,
    'fatigue_mult': 0.10,
    'row_mean': 18,
    'volatility': 0.10,
    'cooldown': 8,
    "attention_sens": 0.8,
    "novelty_sens": 0.7,
    "response_sens": 0.9,
    "flat_decay": 0.10

 },
 'Quiet Churner': {
     'user_health_mult': 0.78,
     'fatigue_mult': 0.12,
     'row_mean': 16,
     'volatility': 0.09,
     'cooldown': 9,
     "attention_sens": 0.3,
    "novelty_sens": 0.2,
    "response_sens": 0.8,
    "flat_decay": 0.03

     
 },
 'Erratic Veteran': {
     'user_health_mult': 0.62,
     'fatigue_mult': 0.10,
     'row_mean': 19,
     'volatility': 0.10,
     'cooldown': 8,
     "attention_sens": 0.9,
    "novelty_sens": 0.8,
    "response_sens": 0.9,
    "flat_decay": 0.08

 },
 'Inconsistent Contributor': {
     'user_health_mult': 0.65,
     'fatigue_mult': 0.10,
     'row_mean': 17,
     'volatility': 0.10,
     'cooldown': 9,
     "flat_decay": 0.06
  
 },
 'Engaged Opportunist': {
     'user_health_mult': 0.75,
     'fatigue_mult': 0.09,
     'row_mean': 19,
     'volatility': 0.09,
     'cooldown': 8,
     "attention_sens": 0.7,
    "novelty_sens": 0.8,
    "response_sens": 0.6,
    "flat_decay": 0.04

    
 },
 'Stable High Value': {
     'user_health_mult': 1.05,
     'fatigue_mult': 0.07,
     'row_mean': 18,
     'volatility': 0.07,
     'cooldown': 8,
     "attention_sens": 0.5,
    "novelty_sens": 0.3,
    "response_sens": 0.4,
    "flat_decay": 0.02

    
 },
 'Recoverable Dropoff': {
     'user_health_mult': 1.00,
     'fatigue_mult': 0.08,
     'row_mean': 17,
     'volatility': 0.08,
     'cooldown': 9,
    "attention_sens": 0.7,
    "novelty_sens": 0.5,
    "response_sens": 0.8,
    "flat_decay": 0.06

 }
}

ARCHETYPE_TO_INT = {
    'Steady Performer': 0,
    'At-Risk Minimalist': 1,
    'Quiet Churner': 2,
    'Erratic Veteran': 3,
    'Inconsistent Contributor': 4,
    'Engaged Opportunist': 5,
    'Stable High Value': 6,
    'Recoverable Dropoff': 7,
}

INT_TO_ARCHETYPE = {v: k for k, v in ARCHETYPE_TO_INT.items()}


# Event type definitions
EVENT_TYPES = ["login", "browse", "interact", "purchase", "idle"]
EVENT_TYPE_TO_INT = {ev: i for i, ev in enumerate(EVENT_TYPES)}
INT_TO_EVENT_TYPE = {i: ev for ev, i in EVENT_TYPE_TO_INT.items()}

# Event severity definitions
SEVERITY_LEVELS = ["low", "medium", "high"]
SEVERITY_TO_INT = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
INT_TO_SEVERITY = {i: s for s, i in SEVERITY_TO_INT.items()}




COSTLY_ACTIONS = {
    "observe":   0.01,   # ultra-light telemetry, almost free
    "monitor":   0.05,   # continuous background tracking
    "nudge":     0.10,   # low-touch prod ping
    "reinforce": 0.25,   # lightweight reactivation
    "support":   0.30,   # helpdesk/autoresolve
    "redirect":  0.40,   # routing, attention-shifting
    "suppress":  0.50,   # costly throttle / moderation
    "boost":     1.20,   # compute-heavy promotion/inference
    "escalate":  1.50    # all-hands escalation, most expensive
}



# 200= basic, 720=pro, 3000=enterprise
VALUE_TIERS = [200, 720, 3000]
TIER_PROBS = [0.7, 0.25, 0.05]


ACTIONS = ["observe", "monitor", "nudge","reinforce", "support", "redirect", "boost", "escalate", "suppress"]
# === Strategies for baseline and Proprietary Challenger Model===
MOST = ["observe"]
HIGH = ["monitor", "nudge"]
MODERATE = ["reinforce", "support", "redirect"]
LOW = ["boost", "escalate"]
LEAST = ["suppress"]


def sample_effect(effect_def):
    """
    Turn an effect definition dict into a float sample.
    Handles mean/std plus optional outliers.
    """
    if isinstance(effect_def, (int, float)):
        return float(effect_def)

    mean = effect_def.get("mean", 0.0)
    std = effect_def.get("std", 0.0)
    outlier_chance = effect_def.get("outlier_chance", 0.0)
    outlier_mult = effect_def.get("outlier_mult", 1.0)

    # sample from normal distribution
    val = random.gauss(mean, std)

    # rare outlier spike
    if random.random() < outlier_chance:
        val *= outlier_mult if val >= 0 else -outlier_mult

    return val
