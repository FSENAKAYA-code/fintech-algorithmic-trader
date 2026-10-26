"""Run a chronological return-direction demo on generated toy data."""

import argparse

import numpy as np

from fintech.temporal import evaluate_return_direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", help="optional local market CSV containing a target column")
    parser.add_argument("--target", default="direction")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 600)
    X = np.column_stack([np.roll(returns, lag) for lag in (1, 2, 5)])[5:]
    y = (returns[5:] > 0).astype(int)
    if args.data_csv:
        import pandas as pd

        frame = pd.read_csv(args.data_csv)
        if args.target not in frame:
            raise ValueError(f"target column {args.target!r} is missing")
        if "timestamp" in frame:
            frame = frame.sort_values("timestamp")
        y = frame.pop(args.target).to_numpy()
        X = frame.select_dtypes(include="number").to_numpy(dtype=float)
    print(evaluate_return_direction(X, y))


if __name__ == "__main__":
    main()
