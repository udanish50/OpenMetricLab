from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .classification import evaluate_classification
from .regression import evaluate_regression


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openmetriclab",
        description="Truth-vs-prediction evaluation for regression and classification CSV files.",
    )
    parser.add_argument("csv")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--prob-prefix", default="prob_")
    args = parser.parse_args()

    with Path(args.csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    y_true = [row[args.actual] for row in rows]
    y_pred = [row[args.predicted] for row in rows]
    if args.task == "regression":
        result = evaluate_regression(
            [float(value) for value in y_true],
            [float(value) for value in y_pred],
        )
    else:
        probability_columns = [
            column for column in rows[0] if column.startswith(args.prob_prefix)
        ] if rows else []
        probabilities = (
            [[float(row[column]) for column in probability_columns] for row in rows]
            if probability_columns
            else None
        )
        classes = (
            [column[len(args.prob_prefix) :] for column in probability_columns]
            if probability_columns
            else None
        )
        result = evaluate_classification(y_true, y_pred, probabilities, classes)

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
