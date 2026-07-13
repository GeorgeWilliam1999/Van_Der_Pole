"""Train and score one capacity-study run. HTCondor entry point.

Usage: python run_one.py <variant> <horizon> <depth> <width> <start_idx> [--smoke]
e.g.:  python run_one.py causal 27 6 128 3
"""
import sys

import capacity

if __name__ == "__main__":
    variant, horizon, depth, width, start_idx = sys.argv[1:6]
    capacity.run_one(variant, float(horizon), int(depth), int(width),
                     int(start_idx), smoke="--smoke" in sys.argv)
