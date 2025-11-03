# ChurnLab : Equilibria

A GPU-accelerated simulation for population churn dynamics with side-by-side controller benchmarking (e.g., *Virgil* vs baselines). The repo includes a fast population engine, synthetic row generators, an influx model, a benchmark runner, and slide-ready visualization utilities.

## Highlights
- **GPU-first, CPU-friendly**: vectorized with CuPy, with CPU fallbacks where sensible.
- **Controller-agnostic**: plug in any agent; baseline heuristics included in repo structure.
- **Reproducible**: single-seed setup and deterministic vector draws where possible.
- **Publication-ready**: utilities emit charts sized for pitch decks and papers.

## Install

```bash
# clone your repo first, then inside it:
pip install -e .
```

### Optional: GPU acceleration
Install **CuPy** appropriate to your CUDA/ROCm stack. Example:
```bash
# CUDA 12.x wheels (see CuPy docs for other variants)
pip install cupy-cuda12x
```

## Quickstart

```python
from runner import run_batch_loop
from population.PopulationBranch import PopulationBranch
from strategy.challenger import Challenger
from config import *

# Instantiate branches (baseline strategy module referenced by runner)
challenger = PopulationBranch(name="virgil", model=Challenger())
baseline = PopulationBranch(name="baseline", model=None)

run_batch_loop(challenger, baseline, config=__import__("config"), same_events=True)
```

Outputs (charts) land in `./output/`.

> Note: If you don’t have a GPU, the simulation will still run; large-N runs are just slower.

## Module Map (what you’ll find in this repo)

- **Population engine**: `population/PopulationBranch.py` — GPU arrays for per-user state, vectorized add/remove, step loop, metrics collection.
- **Influx model**: `population/influx.py` — population inflow rate as a function of health, engagement, archetype fatigue, energy cost, seasonality, and capacity.
- **Row synthesis**: `population/row_generator.py` — initializes mechanism state and generates observed per-user rows with realistic noise.
- **Controller wrapper**: `strategy/challenger.py` — thin shim to call a Challenger model (e.g., Virgil) with curated action sets.
- **State machine / rules**: `utils/game_tree.py` and `utils/tree_arrays.py` — canonical states, actions, probabilistic rules, archetype attributes, and vector samplers.
- **Benchmark runner**: `runner.py` — single entry point `run_batch_loop(...)` for challenger vs baseline head-to-head.
- **Visualization**: `viz/viz_tools.py` — slide-ready figures, deck-friendly formatting helpers.

## Configuration
Set simulation constants in `config.py` (e.g., `TOTAL_BATCHES`, `BATCHES_PER_DAY`, `MAX_USERS`, seeds).

## Reproducibility
- Single `SEED` controls NumPy/CuPy RNGs in key components.
- Seasonality and exogenous content packages can be synchronized across branches for fair comparisons.

## Citing
If you use this work, please cite the repository. A minimal `CITATION.cff` is included.

## License
MIT — see `LICENSE`.
