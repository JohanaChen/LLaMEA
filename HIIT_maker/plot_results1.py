import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "Results/Experiment_1"
NUM_RUNS = 6

LLM_NAMES = [
    "gpt4o_mini",
    "deepseek_chat",
    "gemini_flash",
    "llama3_ollama",
]

def plot_llm_comparison():
    plt.figure(figsize=(10, 6))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(LLM_NAMES)))

    for llm_name, color in zip(LLM_NAMES, colors):
        runs = []

        for run in range(NUM_RUNS):
            csv_path = os.path.join(
                RESULTS_DIR, f"{llm_name}_ratio_0.2_run_{run}.csv"
            )
            df = pd.read_csv(csv_path)

            fitness = df["fitness"].values
            runs.append(fitness)

            # Plot individual runs (light)
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

        # Plot mean curve (strong)
        plt.plot(
            gens,
            mean_curve,
            color=color,
            linewidth=3,
            label=llm_name
        )

    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.title("Fitness Score Evolution Across Different LLMs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_llm_comparison()
