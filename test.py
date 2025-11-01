import os
import json
import numpy as np
# from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA, Gemini_LLM, OpenAI_LLM, Ollama_LLM
import logging


# Setup global logger
log = logging.getLogger("evaluate_HIIT")
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    # Execution code starts here

    # GEMINI
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-1.5-flash"
    experiment_name = "pop1-5"
    llm = Gemini_LLM(api_key, ai_model)

    # OPENAI
    # api_key = os.getenv("OPENAI_API_KEY")
    # ai_model = "gpt-3.5-turbo" 
    # experiment_name = "hiit1"
    # llm = OpenAI_LLM(api_key, ai_model)

    # Ollama
    # llm = Ollama_LLM(model="llama3")

    def _coerce_json(x):
        """Return a dict from either a dict or a JSON string/bytes (handles code fences)."""
        if isinstance(x, dict):
            return x
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", "ignore")
        if isinstance(x, str):
            s = x.strip()
            # strip ```json ... ``` fences if present
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
            return json.loads(s)
        raise TypeError(f"Unsupported solution.data type: {type(x)}")

    def _log(logger, msg, level="info", also_print=True):
        """
        Send `msg` to whatever logging interface we were given.
        Falls back gracefully if the object doesn't look like a stdlib logger.
        """
        try:
            if logger is None:
                if also_print:
                    print(msg)
                return

            # 1) stdlib-like: has .info/.debug/.warning etc.
            fn = getattr(logger, level, None)
            if callable(fn):
                fn(msg)
                if also_print is True and level.lower() in ("error", "warning"):
                    print(msg)
                return

            # 2) has .log(...) (maybe stdlib signature or custom)
            if hasattr(logger, "log"):
                try:
                    lvl = getattr(logging, level.upper(), logging.INFO)
                    logger.log(lvl, msg)  # stdlib signature (level, msg)
                except TypeError:
                    logger.log(msg)       # custom signature (msg)
                if also_print:
                    print(msg)
                return

            # 3) file-like .write(...)
            if hasattr(logger, "write"):
                logger.write(str(msg) + "\n")
                if also_print:
                    print(msg)
                return

            # 4) callable object
            if callable(logger):
                logger(msg)
                if also_print:
                    print(msg)
                return

            # 5) last resort
            if also_print:
                print(msg)
        except Exception:
            # Never let logging crash evaluation
            try:
                if also_print:
                    print(msg)
            except Exception:
                pass

    def evaluate_HIIT(solution, logger=None):
        feedback = ""
        program = []

        try:
            # Parse JSON input
            data = _coerce_json(solution.data)

            # Collect exercises from JSON structure
            if "warm_up" in data:
                program += data["warm_up"]
            if "main_set" in data and "exercises" in data["main_set"]:
                program += data["main_set"]["exercises"]
            if "cool_down" in data:
                program += data["cool_down"]

            if not program:
                raise ValueError("No exercises found in JSON input.")

            # Calculate metrics
            total_work = sum(ex["duration"] for ex in program)
            total_rest = sum(ex["rest"] for ex in program)
            total_duration = total_work + total_rest
            num_exercises = len(program)
            unique_exercises = len(set(ex["name"] for ex in program))
            density = min(10, total_work / total_rest * 5) if total_rest > 0 else 10

            # Compute fitness score
            fitness = 0
            fitness += density
            fitness += unique_exercises * 5
            fitness -= sum(1 for ex in program if ex["rest"] > 30) * 10


            feedback = (
                f"Workout evaluated: {num_exercises} exercises, "
                f"total duration {total_duration}s, "
                f"density {density:.2f}, fitness {fitness:.2f}"
            )

        except json.JSONDecodeError as e:
            fitness = -np.inf
            feedback = f"Invalid JSON: {e} | Raw text: {solution.data}"
        except Exception as e:
            fitness = -np.inf
            feedback = f"Error during evaluation: {e} | Raw text: {solution.data}"

        _log(logger, feedback)

        solution.set_scores(fitness=fitness, feedback=feedback)
        return solution
    
    task_prompt = """
    Design a high-intensity interval training (HIIT) program as valid JSON.
    The program should be 10-40 minutes long in total.

    Each exercise of the program must include:
    - The name of the exercise (string)
    - The duration of the exercise (in seconds, integer)
    - The rest period after the exercise (in seconds, integer)
    Include warm-up and cool-down phases and avoid repeating the same exercise too often.
    The core training should include a balance of cardio, strength, and flexibility exercises.

    Ensure variety in terms of range of:
    - Types of exercises
    - Training structures (e.g. EMOM, AMRAP, E2MOM, Death-by, Tabata, etc.)
    - Number of repetitions
    - Load and intensity (implicitly controlled via duration and rest).
    Each generated program doesn't have to contain the same set of exercises, different structures of programs are allowed, such as linear sequence, repeated circuits, or block cycles.
    Make the program suitable for intermediate-level individuals with moderate fitness.
    Annotate exercises with a tag like `(type: cardio)` or `(intensity: medium)` to indicate the nature and difficulty of the exercise.

    The output should be pure JSON and easy-to-read list of exercises, for example:
    {
    "warm_up": [
        { "name": "Jumping Jacks", "duration": 30, "rest": 10, "type": "cardio" },
        { "name": "Arm Circles", "duration": 20, "rest": 10, "type": "mobility" }
    ],
    "main_set": {
        "type": "EMOM",
        "rounds": 5,
        "exercises": [
        { "name": "Burpees", "duration": 40, "rest": 20, "type": "cardio", "intensity": "high" },
        { "name": "Push-ups", "duration": 30, "rest": 30, "type": "strength", "intensity": "medium" },
        { "name": "Jump Squats", "duration": 30, "rest": 20, "type": "strength/cardio" }
        ]
    },
    "cool_down": [
        { "name": "Forward Fold", "duration": 30, "rest": 10, "type": "flexibility" },
        { "name": "Child's Pose", "duration": 30, "rest": 10, "type": "recovery" }
    ]
    }

    """


    for experiment_i in [1]:
        # A 1+1 strategy
        es = LLaMEA(
            f=evaluate_HIIT,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name="hiit1",
            elitism=True,
            HPO=False,
            budget=10,
        )
        print(es.run())
