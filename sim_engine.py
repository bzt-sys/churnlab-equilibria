import argparse
import sys
from types import SimpleNamespace
import cupy as cp
from config import *
from strategy.challenger import Challenger
from population.PopulationBranch import PopulationBranch
from utils.runner import run_batch_loop
from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChurnLab Simulation Engine")

    parser.add_argument("--days", type=int, default=DAYS,
                        help="Total number of days to simulate (default: from config.py)")
    parser.add_argument("--enable-influx", action="store_true",
                        help="Enable new user influx over time")
    parser.add_argument("--disable-influx", action="store_false", dest="enable_influx",
                        help="Disable influx (default)")
    parser.set_defaults(enable_influx=False)

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num-users", type=int, default=NUM_USERS,
                        help="Initial number of users (default from config)")
    parser.add_argument("--max-users", type=int, default=MAX_USERS,
                        help="Maximum user cap during influx (default from config)")
    parser.add_argument("--batches-per-day", type=int, default=BATCHES_PER_DAY,
                        help="How many intervention windows per day (default from config)")

    return parser.parse_args()


def update_config_from_args(args):
    # Build a runtime config that mirrors global structure
    runtime_config = SimpleNamespace(**globals())

    runtime_config.DAYS = args.days
    runtime_config.NUM_USERS = args.num_users
    runtime_config.MAX_USERS = args.max_users
    runtime_config.BATCHES_PER_DAY = args.batches_per_day
    runtime_config.TOTAL_BATCHES = args.days * args.batches_per_day

    # Re-seed RNG globally
    global rng
    from numpy.random import default_rng
    rng = default_rng(args.seed)

    return runtime_config


def run_sim():
    args = parse_args()
    config = update_config_from_args(args)

    print(f"Launching ChurnLab simulation...")
    print(f"• Days: {config.DAYS}")
    print(f"• Batches per day: {config.BATCHES_PER_DAY}")
    print(f"• Influx enabled: {args.enable_influx}")
    print(f"• Seed: {SEED}")
    print(f"• Initial Users: {config.NUM_USERS}")
    print(f"• Max Users: {config.MAX_USERS}")
    print(f"{'-'*40}")

    challenger = PopulationBranch(name="challenger", model=Challenger())
    baseline = PopulationBranch(name="baseline")
    
    
    run_batch_loop(challenger, baseline, config=config)


if __name__ == "__main__":
    run_sim()
