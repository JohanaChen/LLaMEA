# plot_results.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "Results"
# mutation_ratios = [0.0, 0.3, 0.7, 1.0]
mutation_ratios = [0.0, 0.3]
num_runs = 1

def plot_all_runs():
    plt.figure(figsize=(12, 8))

    for i, ratio in enumerate(mutation_ratios):
        plt.subplot(2, 2, i+1)
        plt.title(f"Mutation ratio = {ratio}")

        for run in range(num_runs):
            csv_path = os.path.join(RESULTS_DIR, f"ratio_{ratio}_run_{run}.csv")
            df = pd.read_csv(csv_path)
            plt.plot(df["generation"], df["fitness"], label=f"run {run}")

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_averaged_curves():
    plt.figure(figsize=(8, 6))

    for ratio in mutation_ratios:
        runs = []

        for run in range(num_runs):
            df = pd.read_csv(f"{RESULTS_DIR}/ratio_{ratio}_run_{run}.csv")
            runs.append(df["fitness"].values)

        runs = np.array(runs)
        mean_curve = runs.mean(axis=0)
        std_curve = runs.std(axis=0)

        gens = np.arange(len(mean_curve))

        plt.plot(gens, mean_curve, label=f"ratio={ratio}")
        plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Average Fitness Evolution Across Mutation Ratios")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_all_runs()
    plot_averaged_curves()
