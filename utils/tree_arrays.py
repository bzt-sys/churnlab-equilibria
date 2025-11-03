# tree_arrays.py

"""
tree_arrays.py
--------------
GPU-ready constants and vectorized samplers for ChurnLab : Equilibria:
- States & transitions (GAME_TREE → STATE_ARRAYS)
- Actions & effect distributions (ACTION_EFFECTS → *_MEAN/STD/… arrays)
- Archetype attributes (ARCH_ARRAY)
- Event types/severities & samplers
- Content packages & population-scaled effects
"""

from config import *
import numpy as np
import cupy as cp

# --- State Indexing ---
STATES = [
    "stable",
    "d1", "d2", "d3", "d4", "d5", "d6",   # degrade branch
    "e1", "e2", "e3", "e4"                # escalate branch
]

STATE_TO_INT = {s: i for i, s in enumerate(STATES)}
INT_TO_STATE = {i: s for i, s in enumerate(STATES)}
N_STATES = len(STATE_TO_INT.keys())

# Dense numpy array version for GPU compatibility
STATE_ARRAY = np.array(STATES)  # mostly for reverse lookup in viz
  # ensure GPU resident



######################### Probabilistic Rules (lifelines, landmines, etc.) ###################
# Shape: [num_states, num_actions, num_states]
# These rules change every batch, meant to introduce strategic result chaos, to obfuscate results from actions

PROBABILISTIC_RULES = {
      # --- Lifeline boosts ---
    ("d4", 'lifeline_1'): [("d3", 0.6), ("d2", 0.35), ("stable", 0.05)],  
        # at churn edge, support usually helps, but not always
    
    ("e4", "lifeline_2"):   [("e1", 0.6), ("e3", 0.3), ("e4", 0.1)],  
        # at peak escalation, boost *often* pulls them down, but not guaranteed
    # --- Degrade branch ---
    ("stable", "landmine_1"): [("stable", 0.2), ("d1", 0.6), ("d2", 0.2)],
    ("d2", "landmine_2"):   [("d1", 0.4), ("d2", 0.55), ("d3", 0.05)],
    ("d3", "landmine_3"):     [("d4", 0.6), ("d3", 0.3), ("d2", 0.1)], 
    ("d3", "landmine_4"):   [("d4", 0.4), ("d5", 0.5), ("d6", 0.1)], 

    # --- Escalate branch ---
    ("e1", "erratic_1"):     [("e2", 0.5), ("e1", 0.3), ("stable", 0.2)], 
    ("e2", "erratic_2"):   [("e1", 0.5), ("e2", 0.4), ("e3", 0.1)],  
    ("e2", "erratic_3"):     [("e3", 0.5), ("e2", 0.4), ("stable", 0.1)],
    ("e3", "erratic_4"):  [("d1", 0.6), ("e2", 0.2), ("stable", 0.2)],  
    ("e3", "erratic_5"):   [("e1", 0.5), ("e2", 0.3), ("e3", 0.2)],
}



# Build the pseudo-action index automatically from the rules to avoid mismatches
PROB_ACTIONS = sorted({a for (_, a) in PROBABILISTIC_RULES.keys()})
PROB_ACTION_TO_INT = {a: i for i, a in enumerate(PROB_ACTIONS)}
INT_TO_PROB_ACTION = {i: a for a, i in PROB_ACTION_TO_INT.items()}
NUM_PROB_ACTIONS = len(PROB_ACTIONS)

# Number of branches is the max number of outcomes in any rule
MAX_BRANCHES = max(len(outcomes) for outcomes in PROBABILISTIC_RULES.values())

# Allocate tensors with the correct shapes:
#   axis0: states, axis1: pseudo-actions, axis2: outcome-branch
PROB_RULES  = cp.zeros((N_STATES, NUM_PROB_ACTIONS, MAX_BRANCHES), dtype=cp.float32)
PROB_STATES = cp.full( (N_STATES, NUM_PROB_ACTIONS, MAX_BRANCHES), -1, dtype=cp.int32)

# Populate
for (state_name, action_name), outcomes in PROBABILISTIC_RULES.items():
    s = STATE_TO_INT[state_name]
    a = PROB_ACTION_TO_INT[action_name]
    for j, (next_state_name, prob) in enumerate(outcomes):
        PROB_RULES[s, a, j]  = prob
        PROB_STATES[s, a, j] = STATE_TO_INT[next_state_name]


def sample_probabilistic_transitions(current_states, current_actions, rng=None):
    """
    Vectorized sampler for probabilistic rules (lifelines, landmines, erratic boosts).

    Args:
        current_states (cp.ndarray): shape (N,) ints of state indices
        current_actions (cp.ndarray): shape (N,) ints of action indices
        rng: cp.random (default), or np.random for CPU fallback

    Returns:
        cp.ndarray: shape (N,) of next state indices
    """
    rng = rng or cp.random

    # Ensure GPU arrays
    current_states = cp.asarray(current_states, dtype=cp.int32)
    current_actions = cp.asarray(current_actions, dtype=cp.int32)
    N = current_states.shape[0]

    # Slice the distribution rows for each (state, action)
    probs = PROB_RULES[current_states, current_actions]  # shape (N, NUM_STATES)

    # If no rule defined (all zeros), stay in same state
    row_sums = probs.sum(axis=1, keepdims=True)
    no_rule_mask = row_sums.squeeze() == 0

    # Normalize probabilities (avoid division by zero)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums

    # Sample next states
    rolls = rng.rand(N)
    cumsum_probs = cp.cumsum(probs, axis=1)
    next_states = cp.argmax(rolls[:, None] <= cumsum_probs, axis=1).astype(cp.int32)

    # Force "stay in same state" if no rules existed
    next_states[no_rule_mask] = current_states[no_rule_mask]

    return next_states



##################### Action Effects ##################
# --- Action Indexing ---
ACTIONS = [
    "observe", "remind",
    "nudge", "reinforce",
    "support", "redirect", "boost",
    "escalate", "suppress"
]

COSTLY_ACTIONS = {
    "observe":   0.01,   # ultra-light telemetry, almost free
    "remind":   0.07,   # continuous background mini-nudges
    "nudge":     0.15,   # low-touch prod ping
    "reinforce": 0.35,   # lightweight reactivation
    "support":   0.82,   # helpdesk/autoresolve
    "redirect":  0.65,   # routing, attention-shifting
    "suppress":  0.82,   # costly throttle / moderation
    "boost":     1.35,   # compute-heavy promotion/inference
    "escalate":  1.75    # all-hands escalation, most expensive
}

NUM_ACTIONS = len(ACTIONS)
ACTION_ARRAY = np.array(ACTIONS)


ACTION_TO_INT = {
    "observe": 0,
    "remind": 1,
    "nudge": 2,
    "reinforce": 3,
    "support": 4,
    "redirect": 5,
    "boost": 6,
    "escalate": 7,
    "suppress": 8,
}
INT_TO_ACTION = {v: k for k, v in ACTION_TO_INT.items()}

ACTION_EFFECTS = {
    # Passive actions
    "observe": {
        "decay_shift": {"mean": 0.03, "std": 0.009, "outlier_chance": 0.01, "outlier_scale": 1.2},
        "engagement_shift": {"mean": -0.03, "std": 0.02, "outlier_chance": 0.01, "outlier_scale": 1.5},
        'creep': {"mean": 0.025, "std": 0.008, "outlier_chance": 0.01, "outlier_scale": 1.2}
    # Light nudges
    },
    "remind": {
        "decay_shift": {"mean": 0.037, "std": 0.05, "outlier_chance": 0.02, "outlier_scale": 1.5},
        "engagement_shift": {"mean": 0.045, "std": 0.02, "outlier_chance": 0.01, "outlier_scale": 2.0},
    },

    "nudge": {
        "decay_shift": {"mean": 0.035, "std": 0.02, "outlier_chance": 0.03, "outlier_scale": 1.4},
        "engagement_shift": {"mean": +0.06, "std": 0.04, "outlier_chance": 0.02, "outlier_scale": 1.8},
    },
    "reinforce": {
        "decay_shift": {"mean": 0.08, "std": 0.03, "outlier_chance": 0.02, "outlier_scale": 1.5},
        "engagement_shift": {"mean": +0.16, "std": 0.03, "outlier_chance": 0.01, "outlier_scale": 2.0},
    },

    # Supportive (higher cost, bigger swings)
    "support": {
        "decay_shift": {"mean": 0.06, "std": 0.03, "outlier_chance": 0.03, "outlier_scale": 1.3},
        "engagement_shift": {"mean": +0.20, "std": 0.08, "outlier_chance": 0.02, "outlier_scale": 2.5},
    },
    "redirect": {
        "decay_shift": {"mean": 0.8, "std": 0.03, "outlier_chance": 0.04, "outlier_scale": 2.2},
        "engagement_shift": {"mean": +0.14, "std": 0.05, "outlier_chance": 0.03, "outlier_scale": 2.0},
    },
    "boost": {
        "decay_shift": {"mean": 0.10, "std": 0.5, "outlier_chance": 0.02, "outlier_scale": 3.0},
        "engagement_shift": {"mean": +0.16, "std": 0.9, "outlier_chance": 0.02, "outlier_scale": 2.75},
        'creep': {"mean": 0.05, "std": 0.05, "outlier_chance": 0.01, "outlier_scale": 1.5}
    },

    # Heavy interventions
    "escalate": {
        "decay_shift": {"mean": +0.12, "std": 0.055, "outlier_chance": 0.015, "outlier_scale": 4.0},
        "engagement_shift": {"mean": +0.22, "std": 0.08, "outlier_chance": 0.07, "outlier_scale": 3.0},
        'creep': {"mean": 0.05, "std": 0.03, "outlier_chance": 0.01, "outlier_scale": 1.5}
    },
    "suppress": {
        "decay_shift": {"mean": +0.12, "std": 0.06, "outlier_chance": 0.04, "outlier_scale": 2.5},
        "engagement_shift": {"mean": -0.075, "std": 0.04, "outlier_chance": 0.02, "outlier_scale": 2.0},
    }
}


# Number of actions
NUM_ACTIONS = len(ACTION_TO_INT)

# Initialize arrays for decay and engagement effects
DECAY_MEAN   = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
DECAY_STD    = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
DECAY_OUT_P  = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
DECAY_OUT_S  = cp.zeros(NUM_ACTIONS, dtype=cp.float32)

ENGAGE_MEAN  = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
ENGAGE_STD   = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
ENGAGE_OUT_P = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
ENGAGE_OUT_S = cp.zeros(NUM_ACTIONS, dtype=cp.float32)

# Optional creep effects (only for some actions)
CREEP_MEAN   = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
CREEP_STD    = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
CREEP_OUT_P  = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
CREEP_OUT_S  = cp.zeros(NUM_ACTIONS, dtype=cp.float32)

ACTION_COST = cp.zeros(NUM_ACTIONS, dtype=cp.float32)
# Fill arrays based on original dict
for action_name, params in ACTION_EFFECTS.items():
    idx = ACTION_TO_INT[action_name]

    if "decay_shift" in params:
        DECAY_MEAN[idx]   = params["decay_shift"]["mean"]
        DECAY_STD[idx]    = params["decay_shift"]["std"]
        DECAY_OUT_P[idx]  = params["decay_shift"]["outlier_chance"]
        DECAY_OUT_S[idx]  = params["decay_shift"]["outlier_scale"]

    if "engagement_shift" in params:
        ENGAGE_MEAN[idx]  = params["engagement_shift"]["mean"]
        ENGAGE_STD[idx]   = params["engagement_shift"]["std"]
        ENGAGE_OUT_P[idx] = params["engagement_shift"]["outlier_chance"]
        ENGAGE_OUT_S[idx] = params["engagement_shift"]["outlier_scale"]

    if "creep" in params:
        CREEP_MEAN[idx]   = params["creep"]["mean"]
        CREEP_STD[idx]    = params["creep"]["std"]
        CREEP_OUT_P[idx]  = params["creep"]["outlier_chance"]
        CREEP_OUT_S[idx]  = params["creep"]["outlier_scale"]

    if action_name in COSTLY_ACTIONS:
        ACTION_COST[idx] = COSTLY_ACTIONS[action_name]

def sample_effect_array(effect_type: str, actions, rng=None):
    """
    Vectorized effect sampler for each user action.

    effect_type ∈ {"decay_shift","engagement_shift","creep"}
    actions     : cp.ndarray[int] or array-like, shape (N,)
    returns     : cp.ndarray[float32], shape (N,)
    """
    actions = cp.asarray(actions, dtype=cp.int32)
    N = actions.shape[0]

    if effect_type == "decay_shift":
        mean = DECAY_MEAN[actions]
        std  = DECAY_STD[actions]
        pout = DECAY_OUT_P[actions]
        sout = DECAY_OUT_S[actions]
    elif effect_type == "engagement_shift":
        mean = ENGAGE_MEAN[actions]
        std  = ENGAGE_STD[actions]
        pout = ENGAGE_OUT_P[actions]
        sout = ENGAGE_OUT_S[actions]
    elif effect_type == "creep":
        mean = CREEP_MEAN[actions]
        std  = CREEP_STD[actions]
        pout = CREEP_OUT_P[actions]
        sout = CREEP_OUT_S[actions]
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")

    # Important: when loc/scale are (N,), do NOT pass size=...; CuPy returns (N,)
    base = rng.normal(loc=mean, scale=std).astype(cp.float32)

    # Outliers per user
    mask = rng.rand(N) < pout
    # multiply only where mask is True (elementwise scale)
    base = cp.where(mask, base * sout, base)
    return base


#################################### Archetypes #####################################

# Column indices for archetype attributes
ARCH_ATTRS = {
    "user_health_mult": 0,
    "fatigue_mult": 1,
    "row_mean": 2,
    "volatility": 3,
    "cooldown": 4,
    "attention_sens": 5,
    "novelty_sens": 6,
    "response_sens": 7,
    "flat_decay": 8,
}
NUM_ARCH_ATTRS = len(ARCH_ATTRS)
NUM_ARCHETYPES = 8

# Build on host (NumPy), then move to GPU in one go (CuPy)
import numpy as _np  # local import to avoid shadowing main np/cp usage

ARCH_ARRAY = cp.asarray(_np.array([
    #  user_health  fatigue  row_mean  volat  cooldown  att_s  nov_s  resp_s  flat_decay
    [ 1.02,         0.038,    19.0,     0.07,  8.0,      0.26,  0.33,  0.42,   0.08 ],  # 0 Steady Performer
    [ 0.91,         0.91,    18.0,     0.10,  8.0,      0.35,  0.58,  0.65,   0.1 ],  # 1 At-Risk Minimalist
    [ 0.9,         0.89,    16.0,     0.09,  9.0,      0.15,  0.20,  0.62,   0.12 ],  # 2 Quiet Churner
    [ 0.92,         0.02,    19.0,     0.10,  8.0,      0.45,  0.80,  0.68,   0.1 ],  # 3 Erratic Veteran
    [ 0.95,         0.04,    17.0,     0.10,  9.0,      0.27,  0.75,  0.02,   0.11 ],  # 4 Inconsistent Contributor
    [ 0.93,         0.92,    19.0,     0.09,  8.0,      0.35,  0.80,  0.40,   0.09 ],  # 5 Engaged Opportunist
    [ 1.05,         0.99,    18.0,     0.07,  8.0,      0.25,  0.30,  0.20,   0.07 ],  # 6 Stable High Value
    [ 1.00,         0.04,    17.0,     0.08,  9.0,      0.35,  0.50,  0.60,   0.101 ],  # 7 Recoverable Dropoff
], dtype=_np.float32))

# Access pattern example:
# row_mean for archetype 3 ("Erratic Veteran"):
# row_mean = ARCH_ARRAY[3, ARCH_ATTRS["row_mean"]

#######################################   Transition Properties   #########################################

# --- State Index Maps ---
STATE_TO_INT = {
    "stable": 0,
    "d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5, "d6": 6,
    "e1": 7, "e2": 8, "e3": 9, "e4": 10
}
INT_TO_STATE = {v: k for k, v in STATE_TO_INT.items()}

NUM_STATES = len(STATE_TO_INT)
MAX_TRANSITIONS = 4  # max fanout in GAME_TREE
NUM_MULTIPLIERS = 2  # decay_mult, engage_mult


GAME_TREE = {
    "stable": {
        "transitions": {
            "degrade": {"next": "d1", "chance": 0.03},   # placeholder
            "escalate": {"next": "e1", "chance": 0.27},  # placeholder
            "stay": {"next": "stable", "chance": 0.7},  # placeholder
        }
    },

    # --- Degrade Branch ---
    "d1": {
        "transitions": {
            "degrade": {"next": "d2", "chance": 0.35},
            "recover": {"next": "stable", "chance": 0.25},
            "stay": {"next": "d1", "chance": 0.4},
        }
    },
    "d2": {
        "transitions": {
            "degrade": {"next": "d3", "chance": 0.375},
            "recover": {"next": "d1", "chance": 0.2},
            "stay": {"next": "d2", "chance": 0.325},
        }
    },
    "d3": {
        "transitions": {
            "degrade": {"next": "d4", "chance": 0.45},
            "recover": {"next": "d2", "chance": 0.18},
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
            "degrade": {"next": "d6", "chance": 0.40},     # strong pull deeper
            "recover": {"next": "d4", "chance": 0.18},     # slim chance of stepping back
            "stay": {"next": "d5", "chance": 0.42},
        }
    },
    "d6": {
        "transitions": {
            "recover": {"next": "d5", "chance": 0.01},      # rare partial recovery
            "stay": {"next": "d6", "chance": 0.99}
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
            "stabilize": {"next": "stable", "chance": 0.4},
            "stay": {"next": "e2", "chance": 0.33},
        }
    },
    "e3": {
        "transitions": {
            "escalate": {"next": "e4", "chance": 0.2},
            "stabilize": {"next": "stable", "chance": 0.45},
            "stay": {"next": "e3", "chance": 0.25},
            "degrade": {"next": "d1", "chance": 0.1}
        }
    },
    "e4": {
        "transitions": {
            "stabilize": {"next": "stable", "chance": 0.5},
            "stay": {"next": "e4", "chance": 0.275},
            "degrade": {"next": "d2", "chance": 0.225}
            # no further escalate — diminishing returns cap
        }
    },
}

STATE_MULTIPLIERS = {
    "stable":   {"decay_mult": 1.0, "engage_mult": 1.0},

    "d1":       {"decay_mult": 1.05, "engage_mult": 1.0},
    "d2":       {"decay_mult": 1.08, "engage_mult": 0.99},
    "d3":       {"decay_mult": 1.13, "engage_mult": 0.97},
    "d4":       {"decay_mult": 1.15, "engage_mult": 0.96},
    "d5":       {"decay_mult": 1.34, "engage_mult": 0.94},
    "d6":       {"decay_mult": 1.55, "engage_mult": 0.93},

    "e1":       {"decay_mult":  1.03, "engage_mult": 1.02},
    "e2":       {"decay_mult": 1.07, "engage_mult": 1.03},
    "e3":       {"decay_mult": 1.11, "engage_mult": 1.04},
    "e4":       {"decay_mult": 1.21, "engage_mult": 1.11},
}

# --- State class helpers (GPU constants) ---
DEGRADE_STATE_IDS = cp.asarray([
    STATE_TO_INT["d1"], STATE_TO_INT["d2"], STATE_TO_INT["d3"],
    STATE_TO_INT["d4"], STATE_TO_INT["d5"], STATE_TO_INT["d6"]
], dtype=cp.int32)

ESCALATE_STATE_IDS = cp.asarray([
    STATE_TO_INT["e1"], STATE_TO_INT["e2"], STATE_TO_INT["e3"], STATE_TO_INT["e4"]
], dtype=cp.int32)

STABLE_STATE_ID = cp.int32(STATE_TO_INT["stable"])


# --- Build unified array ---
STATE_ARRAYS = np.full((NUM_STATES, MAX_TRANSITIONS * 2 + NUM_MULTIPLIERS), -1.0, dtype=np.float32)

for state_name, state_idx in STATE_TO_INT.items():
    # --- Transitions ---
    transitions = GAME_TREE[state_name]["transitions"]
    next_states = []
    probs = []

    for label, transition in transitions.items():
        next_states.append(STATE_TO_INT[transition["next"]])
        probs.append(transition["chance"])

    # pad if fewer than MAX_TRANSITIONS
    while len(next_states) < MAX_TRANSITIONS:
        next_states.append(-1)
        probs.append(0.0)

    # --- Multipliers ---
    mults = STATE_MULTIPLIERS.get(state_name, {"decay_mult": 0.0, "engage_mult": 0.0})
    decay_mult = mults.get("decay_mult", 0.0)
    engage_mult = mults.get("engage_mult", 0.0)

    # --- Write into array ---
    STATE_ARRAYS[state_idx, 0:MAX_TRANSITIONS] = next_states
    STATE_ARRAYS[state_idx, MAX_TRANSITIONS:MAX_TRANSITIONS*2] = probs
    STATE_ARRAYS[state_idx, -2] = decay_mult
    STATE_ARRAYS[state_idx, -1] = engage_mult

# --- Example use ---
if __name__ == "__main__":
    # Example: look at d4 row
    idx = STATE_TO_INT["d4"]
    row = STATE_ARRAYS[idx]
    print(f"d4 row = {row}")
    print("Next states:", row[0:MAX_TRANSITIONS])
    print("Probs:", row[MAX_TRANSITIONS:MAX_TRANSITIONS*2])
    print("Decay:", row[-2], "Engage:", row[-1])

STATE_ARRAYS = cp.asarray(STATE_ARRAYS, dtype=cp.float32)
# once, in tree_arrays.py
STATE_CLASS = cp.zeros(NUM_STATES, dtype=cp.int8)  # 0 = other
STATE_CLASS[DEGRADE_STATE_IDS]  = 1
STATE_CLASS[ESCALATE_STATE_IDS] = 2
STATE_CLASS[STABLE_STATE_ID]    = 3

# assumes STATE_ARRAYS, STATE_TO_INT, INT_TO_STATE already built
MAX_TRANSITIONS = 4

def sample_next_states(curr_states, healths, fatigues, decay_shifts, engage_shifts, burnout_counter, rng=None):
    """
    Vectorized sampler for next states using STATE_ARRAYS, with burnout fragility.

    Args:
        curr_states     (cp.ndarray[int]): current state indices, shape (N,)
        healths         (cp.ndarray[float]): health values, shape (N,)
        fatigues        (cp.ndarray[float]): fatigue values, shape (N,)
        decay_shifts    (cp.ndarray[float]): decay shift values, shape (N,)
        engage_shifts   (cp.ndarray[float]): engagement shift values, shape (N,)
        burnout_counter (cp.ndarray[int]): burnout accumulation per user, shape (N,)
        rng             (cp.random.Generator): optional CuPy RNG

    Returns:
        cp.ndarray[int]: sampled next state indices, shape (N,)
    """
    rng = rng or cp.random
    curr_states     = cp.asarray(curr_states,     dtype=cp.int32)
    healths         = cp.asarray(healths,         dtype=cp.float32)
    fatigues        = cp.asarray(fatigues,        dtype=cp.float32)
    decay_shifts    = cp.asarray(decay_shifts,    dtype=cp.float32)
    engage_shifts   = cp.asarray(engage_shifts,   dtype=cp.float32)
    burnout_counter = cp.asarray(burnout_counter, dtype=cp.float32)
    N = curr_states.shape[0]

    # gather rows for all current states
    rows = STATE_ARRAYS[curr_states]  # shape (N, MAX_TRANSITIONS*2+2)

    next_states = rows[:, 0:MAX_TRANSITIONS].astype(cp.int32)      # (N, 4)
    base_probs  = rows[:, MAX_TRANSITIONS:MAX_TRANSITIONS*2]       # (N, 4)
    decay_mult  = rows[:, -2]
    engage_mult = rows[:, -1]

    # --- classify transitions ---
    cls = STATE_CLASS[next_states]
    degrade_mask   = (next_states >= 0) & (cls == 1)
    escalate_mask  = (next_states >= 0) & (cls == 2)
    stay_mask      = (next_states == curr_states[:, None])
    stabilize_mask = (cls == 3)

    # start with base
    probs = base_probs.copy()

    # apply health & fatigue biases
    probs = cp.where(degrade_mask, probs * (1.0 - cp.clip(healths[:, None] + decay_shifts[:, None], 0, 1)), probs)
    probs = cp.where(escalate_mask, probs * cp.clip(healths[:, None] + decay_shifts[:, None], 0, 1), probs)
    probs = cp.where(degrade_mask | escalate_mask, probs * cp.clip(fatigues[:, None] + engage_shifts[:, None], 0, 1), probs)
    probs = cp.where(stay_mask | stabilize_mask, probs * (1.0 - cp.clip(fatigues[:, None] + engage_shifts[:, None], 0, 1)), probs)

    # apply engage_mult
        # apply engage_mult
    probs *= engage_mult[:, None]

    # --- Burnout fragility: tilt odds away from stability ---
    # burnout_factor grows linearly with burnout_counter (gentle ramp)
    burnout_factor = 1.0 + (burnout_counter / 10.0)

    # make degrading more likely as burnout rises
    probs = cp.where(degrade_mask, probs * burnout_factor[:, None], probs)

    # make staying stable less likely as burnout rises
    probs = cp.where(stay_mask, probs / burnout_factor[:, None], probs)

    # normalize
    probs_sum = cp.sum(probs, axis=1, keepdims=True)
    probs = cp.where(probs_sum > 0, probs / probs_sum, 0)

    # categorical sample per row
    cumprobs = cp.cumsum(probs, axis=1)
    rolls = rng.rand(N)[:, None]
    choices = cp.argmax(rolls <= cumprobs, axis=1)

    next_state = cp.take_along_axis(next_states, choices[:, None], axis=1).squeeze()
    next_state = cp.where(next_state >= 0, next_state, curr_states)

    return next_state


################ Events ################################


# Event type definitions
EVENT_TYPES = ["login", "browse", "interact", "purchase", "idle"]
EVENT_TO_INT = {ev: i for i, ev in enumerate(EVENT_TYPES)}
INT_TO_EVENT = {i: ev for ev, i in EVENT_TO_INT.items()}

# Event severity definitions
SEVERITY_LEVELS = ["low", "medium", "high"]
SEVERITY_TO_INT = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
INT_TO_SEVERITY = {i: s for s, i in SEVERITY_TO_INT.items()}

EVENT_PROBS_BY_STATE = {
        "stable":     [0.2, 0.4, 0.3, 0.05, 0.05],
        "d1":    [0.1, 0.3, 0.4, 0.1,  0.1],
        "e1":    [0.05, 0.25, 0.4, 0.1, 0.2],
        "e2": [0.3, 0.3, 0.2, 0.1,  0.1],
        "d2":  [0.1, 0.2, 0.3, 0.05, 0.35]
    }

EVENT_TYPE_SCORES = {
    "click": 1,
    "scroll": 0.5,
    "video_play": 2,
    "purchase": 5,
    "bounce": -1
}

EVENT_SEVERITY_MAP = {
    "login":     [0.7, 0.25, 0.05],  # mostly low
    "browse":    [0.5, 0.4, 0.1],    # mostly low/medium
    "interact":  [0.2, 0.6, 0.2],    # medium dominant
    "purchase":  [0.1, 0.3, 0.6],    # skew high
    "idle":      [0.8, 0.15, 0.05],  # almost always low
}



NUM_EVENTS = len(EVENT_TYPES)
NUM_SEVERITIES = len(SEVERITY_LEVELS)

EVENT_ARRAY = np.array(EVENT_TYPES)
SEVERITY_ARRAY = np.array(SEVERITY_LEVELS)

EVENT_PROBS = np.zeros((N_STATES, NUM_EVENTS), dtype=np.float32)
for state_name, probs in EVENT_PROBS_BY_STATE.items():
    s = STATE_TO_INT[state_name]
    EVENT_PROBS[s, :] = probs

EVENT_SCORES = np.zeros(NUM_EVENTS, dtype=np.float32)
for ev, score in EVENT_TYPE_SCORES.items():
    if ev in EVENT_TO_INT:
        EVENT_SCORES[EVENT_TO_INT[ev]] = score

def sample_event_types(states, rng=None):
    """
    Sample event type indices given current user states.
    states: cp.ndarray[int], shape (N,)
    Returns: cp.ndarray[int], shape (N,)
    """
    N = states.shape[0]
    probs = cp.asarray(EVENT_PROBS[states])  # (N, NUM_EVENTS)
    rolls = rng.rand(N)[:,None]
    cumprobs = cp.cumsum(probs, axis=1)
    return cp.argmax(rolls <= cumprobs, axis=1)

############ Content Packages ##############################################

PACKAGE_TEMPLATES = {
    "viral_meme": {
        "attention_range": (0.23, 0.38),
        "novelty_range": (0.1, 0.38),   # slightly higher
        "response_range": (-0.3, 0.7),  # more headroom
    },
    "product_launch": {
        "attention_range": (0.16, 0.31), # small bump
        "novelty_range": (0.10, 0.37),   # more novelty
        "response_range": (-0.13, 1.2),  # allow strong virality
    },
    "bad_press": {
        "attention_range": (0.1, 0.32),  # keep moderate
        "novelty_range": (0.1, 0.42),
        "response_range": (-0.95, -0.5), # unchanged, pure backlash
    },
    "seasonal_event": {
        "attention_range": (0.12, .35),
        "novelty_range": (0.13, 0.32),
        "response_range": (-.55, .65),   # mild bump on positive
    },
}
# Strategy-factor shaping
STRAT_POS_MIN = cp.float32(1.01)
STRAT_POS_MAX = cp.float32(1.33)
STRAT_NEG_MIN = cp.float32(0.6)  # closer to 0 = stronger decay polarization
STRAT_NEG_MAX = cp.float32(0.75)

# Expected raw input range before mapping (roughly what your current nudges produce)
STRAT_INPUT_MIN = cp.float32(0.70)
STRAT_INPUT_MAX = cp.float32(1.30)

# Mean reversion toward 1.0 each tick (0.0 = instant reset, 1.0 = no pull)
STRAT_MEAN_REVERT = cp.float32(0.90)

# How content totals feed the strategy factor (tune to taste)
STRAT_W_ATT = cp.float32(0.80)
STRAT_W_NOV = cp.float32(0.05)
STRAT_W_RESP = cp.float32(0.15)



############### === Content Package (Exogenous Events) === ############
# Periodicity of content events in batches
CONTENT_EVENT_PERIOD = BATCHES_PER_DAY * 30   # every month, new content drop

# Strength-duration tradeoff scaler (higher → strong packages decay faster)
CONTENT_DURATION_SCALER = 0.7


CONTENT_PACKAGES = list(PACKAGE_TEMPLATES.keys())
PKG_TO_INT = {p: i for i, p in enumerate(CONTENT_PACKAGES)}
INT_TO_PKG = {i: p for p, i in PKG_TO_INT.items()}
NUM_PACKAGES = len(CONTENT_PACKAGES)

# Column order: [att_min, att_max, nov_min, nov_max, resp_min, resp_max]
PKG_RANGES = np.zeros((NUM_PACKAGES, 6), dtype=np.float32)

for name, idx in PKG_TO_INT.items():
    att_min, att_max = PACKAGE_TEMPLATES[name]["attention_range"]
    nov_min, nov_max = PACKAGE_TEMPLATES[name]["novelty_range"]
    resp_min, resp_max = PACKAGE_TEMPLATES[name]["response_range"]
    PKG_RANGES[idx] = [att_min, att_max, nov_min, nov_max, resp_min, resp_max]

PKG_RANGES = cp.asarray(PKG_RANGES)

def sample_content_packages_gpu(num_pkgs=1, rng=None):
    """
    Sample exogenous content packages on GPU.

    Returns
    -------
    cp.ndarray shape (num_pkgs, 6):
      [attention, novelty, response, duration, remaining, decay_rate]
    """
    pkg_ids = rng.randint(0, NUM_PACKAGES, size=(num_pkgs,))
    ranges = PKG_RANGES[pkg_ids]  # (N,6)
    att = rng.uniform(ranges[:,0], ranges[:,1])
    nov = rng.uniform(ranges[:,2], ranges[:,3])
    resp = rng.uniform(ranges[:,4], ranges[:,5])
    resp = cp.where(resp > 0, resp + 1, resp - 1)
    base_strength = cp.abs(att) + cp.abs(nov) * resp
    dur = cp.maximum(5, (1.0 / (cp.abs(base_strength)+1e-6)) * 10 * CONTENT_DURATION_SCALER).astype(cp.int32)
    decay = 1.0 / dur
    return cp.stack([att, nov, resp, dur.astype(cp.float32), dur.astype(cp.float32), decay], axis=1)

def apply_content_effects_gpu(health, engagement, fatigue, strategy_factor, archetypes, active_packages):
    """
    Apply active packages to population tensors (vectorized; GPU).

    active_packages : (K,6) or None
      columns: [attention, novelty, response, duration, remaining, decay_rate]

    Returns
    -------
    (health, engagement, fatigue, strategy_factor)
    """
    if active_packages is None or active_packages.shape[0] == 0:
        return health, engagement, fatigue, strategy_factor

    att, nov, resp, duration, remaining, decay = cp.split(active_packages, 6, axis=1)
    scale = remaining.flatten() / cp.maximum(duration.flatten(), 1)

     # --- Recency weighting (most recent = 1.0, nth = 1/n)
    order = cp.argsort(-remaining.flatten())  # descending by recency
    ranks = cp.arange(1, active_packages.shape[0] + 1, dtype=cp.float32)
    weights = 1.0 / ranks
    weights = weights / cp.sum(weights)  # normalize
    w = cp.zeros_like(weights)
    w[order] = weights  # map back to package order

    total_att = cp.sum(att.flatten() * scale * w)
    total_nov = cp.sum(nov.flatten() * scale * w)
    total_resp = cp.sum(resp.flatten() * scale * w)


    # archetype multipliers
    att_mult  = ARCH_ARRAY[archetypes, ARCH_ATTRS["attention_sens"]]
    nov_mult  = ARCH_ARRAY[archetypes, ARCH_ATTRS["novelty_sens"]]
    resp_mult = ARCH_ARRAY[archetypes, ARCH_ATTRS["response_sens"]]

    # --- Population scaling (small softened, large amplified)
    pop_size = health.size
    pop_scale = 1.0 - cp.exp(-pop_size / (MAX_USERS / 2))
    # starts ~0.5 for tiny pops, grows logarithmically with size
     # --- Apply health
    health = cp.clip(
        health
        + pop_scale * (
            (total_att * att_mult * 0.2)
            + (total_nov * nov_mult * 0.15)
            + cp.where(total_resp >= 0,
                       total_resp * resp_mult * 0.1,
                       total_resp * resp_mult * 0.2)
        ),
        0.0, 1.0
    )

    # --- Apply fatigue
    fatigue = cp.clip(
        fatigue
        + pop_scale * (
            (total_nov * nov_mult * 0.2)
            + cp.where(total_resp < 0,
                       cp.abs(total_resp) * resp_mult * 0.1,
                       -total_resp * resp_mult * 0.1)
        ),
        0.0, MAX_FATIGUE
    )

    # --- Apply strategy factor (population-scaled nudge)
    # --- Strategy factor: mean-revert toward 1.0, then map into signed bands by content signal
    # Mean reversion keeps it from camping at the extremes
    strategy_factor = 1.0 + STRAT_MEAN_REVERT * (strategy_factor - 1.0)

    # Content-driven signal (positive → try harder for engagement boost; negative → defensive decay bias)
    sf_signal = (STRAT_W_ATT * total_att) + (STRAT_W_NOV * total_nov) + (STRAT_W_RESP * total_resp)

    # Population-scaled nudge from content
    sf_raw = strategy_factor + pop_scale * sf_signal

    # Normalize sf_raw to [0,1] across an expected input span
    t = (sf_raw - STRAT_INPUT_MIN) / (STRAT_INPUT_MAX - STRAT_INPUT_MIN)
    t = cp.clip(t, 0.0, 1.0).astype(cp.float32)

    # Decide polarity from the signal itself (ties go neutral)
    pol = cp.sign(sf_signal)

    # Positive band mapping: low t → POS_MIN, high t → POS_MAX
    sf_pos = STRAT_POS_MIN + t * (STRAT_POS_MAX - STRAT_POS_MIN)

    # Negative band mapping (more negative for higher t):
    # magnitude goes from |NEG_MAX| (0.70) up to |NEG_MIN| (1.30), then we negate it
    mag_min = cp.abs(STRAT_NEG_MIN)  # 1.30
    mag_max = cp.abs(STRAT_NEG_MAX)  # 0.70
    sf_neg = -(mag_max + t * (mag_min - mag_max))

    # Assemble signed strategy factor; neutral (pol==0) sits at 1.0
    strategy_factor = cp.where(pol > 0, sf_pos,
                    cp.where(pol < 0, sf_neg, cp.float32(1.0)))


    return health, engagement, fatigue, strategy_factor

