"""Train and score one collocation-density-study run. HTCondor entry point.

Usage: python run_one.py --arm <unweighted|causal> --T <horizon> --density <per_unit> --seed <seed> [--smoke]
e.g.:  python run_one.py --arm causal --T 14 --density 80 --seed 1
"""
import argparse

import training

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["unweighted", "causal"])
    p.add_argument("--T", required=True, type=float, dest="horizon")
    p.add_argument("--density", required=True, type=float)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--smoke", action="store_true",
                   help="shrunk budgets, writes to results/smoke/, for "
                        "pipeline validation only")
    args = p.parse_args()

    training.run_one(args.arm, args.horizon, args.density, args.seed,
                     smoke=args.smoke)
