import logging
import json
import argparse
from datetime import datetime
from jinja2 import Template
from dotenv import load_dotenv
import google.generativeai as genai
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HIIT_maker.utils.io_utils import save_json_result
from llamea import LLaMEA, Gemini_LLM
from evaluation.ABtest import ABtest
from evaluation.LLMpredict import LLMpredict
# from evaluation.LLMchoose import LLMchoose
from evaluation.Hybrid import Hybrid
from utils.state_utils import reset_choice_state
from utils.llm_factory import make_llm

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

LLM_CONFIGS = {
    # "gemini_flash": {
    #     "provider": "gemini",
    #     "model": "gemini-flash-latest", #GEMINI 3 FLASH
    # },
    # "gpt4o_mini": {
    #     "provider": "openai",
    #     "model": "gpt-4o-mini",
    # },
    "llama3_ollama": {
        "provider": "ollama",
        "model": "llama3",
    },
    # "deepseek_chat": {
    #     "provider": "deepseek",
    #     "model": "deepseek-chat",
    # },
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger("evaluate_HIIT")

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run HIIT optimization with a chosen evaluator.")
    parser.add_argument(
        "--evaluator",
        type=str,
        default="LLMpredict",
        choices=["ABtest", "LLMpredict", "LLMchoose", "Hybrid"],
        help="Which evaluator to use (default: LLMpredict)"
    )
    args = parser.parse_args()

    # Load prompt
    task_prompt = load_prompt("prompts/hiit_prompt.jinja2")

    mutation_ratios = [0.2]
    num_runs = 1

    EXPERIMENT_NAME = "Experiment_1"
    experiment_dir = os.path.join("Results", EXPERIMENT_NAME)
    os.makedirs(experiment_dir, exist_ok=True)

    for model_tag, llm_cfg in LLM_CONFIGS.items():
        for ratio in mutation_ratios:
            for run in range(num_runs):

                log.info(f"Model={model_tag}, ratio={ratio}, run={run}")

                # Reset global state
                reset_choice_state()

                # Create fresh LLM
                llm = make_llm(llm_cfg)

                # Create evaluator (depends on LLM!)
                if args.evaluator == "ABtest":
                    evaluator = ABtest(logger=log)

                elif args.evaluator == "LLMpredict":
                    evaluator = LLMpredict(llm, logger=log)

                elif args.evaluator == "Hybrid":
                    evaluator = Hybrid(llm, llm_eval=llm, logger=log)

                else:
                    raise ValueError(f"Unknown evaluator: {args.evaluator}")

                # Create ES
                es = LLaMEA(
                    f=evaluator.evaluate_HIIT,
                    n_parents=1,
                    n_offspring=1,
                    llm=llm,
                    task_prompt=task_prompt,
                    experiment_name=f"hiit_{model_tag}",
                    elitism=True,
                    HPO=False,
                    budget=30,
                    max_workers=1,
                    mutation_ratio=ratio,
                    adaptive_mutation=False,
                    minimization=False,
                )

                result = es.run()

                # Save fitness history
                csv_path = os.path.join(
                    experiment_dir,
                    f"{model_tag}_ratio_{ratio}_run_{run}.csv"
                )
                es.save_fitness_history(csv_path)

                # Save final best
                if hasattr(result, "data"):
                    run_dir = os.path.abspath(es.logger.dirname)
                    final_path = os.path.join(run_dir, "final_best.json")

                    with open(final_path, "w") as f:
                        json.dump(result.data, f, indent=2)

                    log.info(f"Saved final best to: {final_path}")

