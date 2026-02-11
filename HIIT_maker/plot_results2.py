# plot_results.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "Results/Experiment_2"
MUTATION_RATIOS = [0.0, 0.3, 0.7, 1.0]
NUM_RUNS = 3

def plot_mutation_ratio_comparison():
    plt.figure(figsize=(10, 6))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(MUTATION_RATIOS)))

    for ratio, color in zip(MUTATION_RATIOS, colors):
        runs = []

        for run in range(NUM_RUNS):
            csv_path = os.path.join(
                RESULTS_DIR, f"gpt4o_mini_ratio_{ratio}_run_{run}.csv"
            )
            df = pd.read_csv(csv_path)

            fitness = df["fitness"].values
            runs.append(fitness)

            # Individual runs (light)
            plt.plot(
                df["generation"],
                fitness,
                color=color,
                alpha=0.25,
                linewidth=1
            )

        runs = np.array(runs)
        mean_curve = runs.mean(axis=0)
        gens = np.arange(len(mean_curve))

        # Mean curve (bold)
        plt.plot(
            gens,
            mean_curve,
            color=color,
            linewidth=3,
            label=f"mutation ratio = {ratio}"
        )

    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.title("Fitness Score Evolution Across Mutation Ratios")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mutation_ratio_comparison()


