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
from llamea import LLaMEA, Gemini_LLM, OpenAI_LLM
from evaluation.ABtest import ABtest
from evaluation.LLMpredict import LLMpredict
# from evaluation.LLMchoose import LLMchoose
from evaluation.Hybrid import Hybrid
from utils.state_utils import reset_choice_state

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger("evaluate_HIIT")

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run HIIT optimization with a chosen evaluator.")
    parser.add_argument(
        "--evaluator",
        type=str,
        default="ABtest",
        choices=["ABtest", "LLMpredict", "LLMchoose", "Hybrid"],  # list all available evaluators
        help="Which evaluator to use (default: ABtest)"
    )
    args = parser.parse_args()

    # LLM setup
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Configure the OPENAI API
    # genai.configure(api_key=api_key)
    llm = OpenAI_LLM(api_key, "gpt-4o-mini")

    # Load prompt
    task_prompt = load_prompt("prompts/hiit_prompt.jinja2")

    # Choose evaluator
    if args.evaluator == "ABtest":
        evaluator = ABtest(logger=log)
    elif args.evaluator == "LLMpredict":
        evaluator = LLMpredict(llm, logger=log)
    # elif args.evaluator == "LLMchoose":
    #     evaluator = LLMchoose(logger=log)
    elif args.evaluator == "Hybrid":
        evaluator = Hybrid(llm, llm_eval=llm, logger=log)
    else:
        raise ValueError(f"Unknown evaluator: {args.evaluator}")

    mutation_ratios = [0.7]
    num_runs = 1

    for ratio in mutation_ratios:
        for run in range(num_runs):
            print(f"Running experiment: ratio={ratio}, run={run}")
            # A 1+1 strategy
            reset_choice_state()
            es = LLaMEA(
                f=evaluator.evaluate_HIIT,
                n_parents=1,
                n_offspring=1,
                llm=llm,
                task_prompt=task_prompt,
                experiment_name="hiit1",
                elitism=True,
                HPO=False,
                budget=10,
                max_workers=1,
                mutation_ratio=ratio,        # added mutation ratio
                adaptive_mutation=False,
            )
            result = es.run()
            # Save fitness history
            es.save_fitness_history(f"Results/ratio_{ratio}_run_{run}.csv")

            if hasattr(result, "data"):
                run_dir = os.path.abspath(es.logger.dirname)
                final_path = os.path.join(run_dir, "final_best.json")

                with open(final_path, "w") as f:
                    json.dump(result.data, f, indent=2)

                print(f"Saved final best to: {final_path}")

