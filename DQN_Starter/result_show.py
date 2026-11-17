"""Plot the CSV training log and greedy validation returns."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    with (args.run / "training.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["mean_100_return"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([int(row["step"]) for row in rows],
            [float(row["mean_100_return"]) for row in rows], label="Training (exploration enabled)")
    evaluations_path = args.run / "evaluations.json"
    if evaluations_path.exists():
        evaluations = json.loads(evaluations_path.read_text(encoding="utf-8"))
        ax.plot([row["step"] for row in evaluations], [row["mean_return"] for row in evaluations],
                marker="o", label="Greedy validation (5 fixed seeds by default)")
    ax.set(xlabel="Environment steps", ylabel="Episode return", title="Image-only DQN Pendulum")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.run / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(args.run / "learning_curve.png")


if __name__ == "__main__":
    main()
