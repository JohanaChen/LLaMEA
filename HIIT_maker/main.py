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
        choices=["ABtest", "LLMpredict", "LLMchoose"],  # list all available evaluators
        help="Which evaluator to use (default: ABtest)"
    )
    args = parser.parse_args()

    # LLM setup
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    llm = Gemini_LLM(api_key, "gemini-flash-latest")

    # Load prompt
    task_prompt = load_prompt("prompts/hiit_prompt.jinja2")

    # Choose evaluator
    if args.evaluator == "ABtest":
        evaluator = ABtest(logger=log)
    # elif args.evaluator == "LLMchoose":
    #     evaluator = LLMchoose(logger=log)
    elif args.evaluator == "LLMpredict":
        evaluator = LLMpredict(llm, logger=log)
    else:
        raise ValueError(f"Unknown evaluator: {args.evaluator}")

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
    )
    result = es.run()
    if hasattr(result, "data"):
        save_json_result(result.data, name_prefix="hiit_best")
