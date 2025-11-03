import random
import math
import copy
import numpy as np
import cupy as cp
from collections import Counter
# Constants

'''Disclaimer
This simulation is a research project into benchmarking adaptive AI in retention spaces via layered interlocking complexity.
It does not reflect your specific user footprint and excludes proprietary modeling layers used in real-world deployments.

ChurnLab: Equilibria is designed to stress-test retention strategies under aggressive behavioral decay assumptions.
For a tailored simulation that uncovers hidden strategic levers in your user base, contact our behavioral intelligence lab at divinecomedy.io

“ChurnLab: Equilibria includes user activity events such as views, clicks, and shares. These can be useful indicators in heuristic-driven systems. 
Strategy designers are encouraged to explore event-sensitive patterns to improve retention outcomes.”'''

SEED = 420
np_rng = np.random.default_rng(SEED) # older CuPy
NUM_USERS = 100
MAX_USERS = 350
BATCHES_PER_DAY = 4
DAYS = 365
TOTAL_BATCHES = DAYS * BATCHES_PER_DAY
 # tuneable — try 0.002 to 0.004 to start

# Energy to resource cost parallels
USD_PER_KWH = 0.1747

BUDGET_SCALER = 0.20

# Penalizes overuse of heavy actions to stem abuse
FATIGUE_DECAY = 0.5
MAX_FATIGUE = 10

MAX_ROWS_PER_USER = 50
FLAT_USER_HEALTH_DECAY = 0.013
ROLLING_WINDOW = BATCHES_PER_DAY * 10

INFLUX_PERIOD = round((BATCHES_PER_DAY * DAYS) / 2)
INFLUX_AMPLITUDE = 0.33
INFLUX_RATE = 10 / 52.0 / BATCHES_PER_DAY

# User rejunevation and health regain to hem decay toward stability
REVIVE_RATE = 0.035
REVIVE_THRESHOLD = 0.45

OPTIMAL_ENGAGE = 0.45       # sweet spot where gains peak
ENGAGE_WIDTH   = 0.2        # tolerance band around the peak
MAX_GAIN_MULT  = 1.8


ENGAGE_THRESH = 0.75
HEALTH_THRESH = 0.5
ENGAGE_FLOOR = 0.45

# Fatigue oscillations induce user based non-action related change in behavior over time frames
FATIGUE_CYCLE_PERIOD = BATCHES_PER_DAY * 15  # number of batches per cycle
FATIGUE_CYCLE_AMPLITUDE = 0.33  # strength of oscillation






