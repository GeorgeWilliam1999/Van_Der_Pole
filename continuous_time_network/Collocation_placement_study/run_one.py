"""Train and score one collocation-placement-study run. HTCondor entry point.

Usage: python run_one.py --arm <unweighted|causal> --T <horizon> --placement <latin|uniform|anchored> --seed <seed> [--smoke]
e.g.:  python run_one.py --arm causal --T 27 --placement anchored --seed 1
"""
import argparse

import training

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["unweighted", "causal"])
    p.add_argument("--T", required=True, type=float, dest="horizon")
    p.add_argument("--placement", required=True,
                   choices=["latin", "uniform", "anchored"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--smoke", action="store_true",
                   help="shrunk budgets, writes to results/smoke/, for "
                        "pipeline validation only")
    args = p.parse_args()

    training.run_one(args.arm, args.horizon, args.placement, args.seed,
                     smoke=args.smoke)
