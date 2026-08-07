"""Aggregate LIBERO eval shards and log the success-rate table to wandb.

Usage: python benchmarks/libero/aggregate_results.py /work/roboleon1295/vlash-eval/results
"""
import json
import pathlib
import sys

import wandb

PAPER = {0: 0.968, 1: 0.972, 3: 0.946}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DELAYS = [0, 1, 3]


def main(results_dir: str) -> None:
    root = pathlib.Path(results_dir)
    table = {}

    # Validate that each JSON has sum(trials)==500
    for delay in DELAYS:
        for suite in SUITES:
            filepath = root / f"{suite}_d{delay}.json"
            data = json.loads(filepath.read_text())
            total_trials = sum(
                task_data.get("trials", 0)
                for task_data in data.get("per_task", {}).values()
            )
            if total_trials != 500:
                sys.exit(
                    f"ERROR: {filepath.name} has sum(trials)={total_trials}, expected 500"
                )

    for delay in DELAYS:
        rates = []
        for suite in SUITES:
            data = json.loads((root / f"{suite}_d{delay}.json").read_text())
            table[f"{suite}/d{delay}"] = data["success_rate"]
            rates.append(data["success_rate"])
        table[f"avg/d{delay}"] = sum(rates) / len(rates)

    run = wandb.init(project="vlash-libero", name="libero-success-rates", job_type="eval")
    wandb.log(table)
    print(f"{'':>16}" + "".join(f"  delay={d}" for d in DELAYS))
    for suite in SUITES:
        print(f"{suite:>16}" + "".join(f"  {table[f'{suite}/d{d}']:.3f}" for d in DELAYS))
    print(f"{'AVG (ours)':>16}" + "".join(f"  {table[f'avg/d{d}']:.3f}" for d in DELAYS))
    print(f"{'paper':>16}" + "".join(f"  {PAPER[d]:.3f}" for d in DELAYS))
    run.finish()


if __name__ == "__main__":
    main(sys.argv[1])
