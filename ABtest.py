import os
import json
import numpy as np
import random
# from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA, Gemini_LLM, OpenAI_LLM, Ollama_LLM
import logging
from HIIT_maker.utils.state import _CHOICE_STATE


# Setup global logger
log = logging.getLogger("evaluate_HIIT")
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    # Execution code starts here

    # GEMINI
    api_key = os.getenv("GEMINI_API_KEY")
    # ai_model = "gemini-1.5-flash"
    ai_model = "gemini-flash-latest"
    experiment_name = "pop1-5"
    llm = Gemini_LLM(api_key, ai_model)

    # OPENAI
    # api_key = os.getenv("OPENAI_API_KEY")
    # ai_model = "gpt-3.5-turbo" 
    # experiment_name = "hiit1"
    # llm = OpenAI_LLM(api_key, ai_model)

    # Ollama
    # llm = Ollama_LLM(model="llama3")

    # ---- Global state for A/B comparisons across calls ----
    # Pure choice-based state (no Elo)
    # _CHOICE_STATE = {
    #     "incumbent_id": None,      # current champion's id (A)
    #     "incumbent_render": None,  # pretty text for A
    #     "incumbent_json": None,    # parsed JSON for A
    #     "fitness_level": 1.0,      # baseline to compare against
    #     "printed_help": False,     # show instructions once
    #     "last_feedback": None,     # user's feedback
    #     "incumbent_feedback": None,
    # }

    def _solution_id(solution):
        if getattr(solution, "id", None) is not None:
            return str(solution.id)
        d = solution.data
        try:
            if isinstance(d, (str, bytes, bytearray)):
                s = d if isinstance(d, str) else d.decode("utf-8", "ignore")
            else:
                # stable string for dicts/lists
                s = json.dumps(d, sort_keys=True)
        except Exception:
            s = repr(d)
        return f"s{abs(hash(s)) % 10**12}"

    def _flatten_program(data):
        """Collect all exercises into a single list for display/metrics."""
        program = []
        if "warm_up" in data and isinstance(data["warm_up"], list):
            program += data["warm_up"]
        if "main_set" in data and isinstance(data["main_set"], dict) and "exercises" in data["main_set"]:
            program += data["main_set"]["exercises"]
        if "cool_down" in data and isinstance(data["cool_down"], list):
            program += data["cool_down"]
        return program

    def _render_program_markdown(data):
        """Pretty text for the terminal (no external libs)."""
        try:
            wu = len(data.get("warm_up", [])) if isinstance(data.get("warm_up", []), list) else 0
            ms = len(data.get("main_set", {}).get("exercises", [])) if isinstance(data.get("main_set", {}), dict) else 0
            cd = len(data.get("cool_down", [])) if isinstance(data.get("cool_down", []), list) else 0
        except Exception:
            wu, ms, cd = 0, 0, 0

        blocks = []
        if wu:
            blocks.append("Warm-up")
        if ms:
            blocks.append(f"Main set ({data.get('main_set', {}).get('type', 'structured')})")
        if cd:
            blocks.append("Cool-down")
        header = f"{' · '.join(blocks) or 'HIIT Program'}"

        program = _flatten_program(data)
        total_work = sum(int(ex.get("duration", 0)) for ex in program)
        total_rest = sum(int(ex.get("rest", 0)) for ex in program)
        total = total_work + total_rest

        lines = [f"{header} — ~{total}s total (work {total_work}s / rest {total_rest}s)", ""]
        for i, ex in enumerate(program, 1):
            name = ex.get("name", "Exercise")
            dur = ex.get("duration", 0)
            rest = ex.get("rest", 0)
            etype = ex.get("type", "")
            intensity = ex.get("intensity", "")
            tag = " · ".join([t for t in [etype, intensity] if t])
            tag = f"  ·  {tag}" if tag else ""
            lines.append(f"{i:>2}. {name}: {dur}s + {rest}s rest{tag}")
        return "\n".join(lines) if lines else "(empty program)"
    
    def reset_choice_state():
        _CHOICE_STATE.update({
            "incumbent_id": None,
            "incumbent_render": None,
            "incumbent_json": None,
            "fitness_level": 1.0,
            "printed_help": False,
            "last_feedback": None,     
            "incumbent_feedback": None,
        })

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
        
        # Parse JSON
        try: 
            data = _coerce_json(solution.data)
        except Exception as e:
            fitness = -np.inf
            feedback = f"Invalid JSON: {e} | Raw text: {solution.data[:200]}..."
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution

        # Collect exercises
        program = _flatten_program(data)
        if not program:
            fitness = -np.inf
            feedback = "No exercises found in JSON input."
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution
        
        rid = _solution_id(solution)
        rendered = _render_program_markdown(data) # Create readable string for the terminal printout

        S = _CHOICE_STATE # Define the global choice-state

        # First ever program becomes the incumbent (A)
        if S["incumbent_id"] is None:
            S["incumbent_id"] = rid
            S["incumbent_render"] = rendered
            S["incumbent_json"] = data
            fitness = S["fitness_level"]
            feedback = "Initialized incumbent (A)."
            if not S["printed_help"]:
                print("\n=== HIIT Preference Picker (terminal) ===")
                print("Each new candidate (B) is compared to the incumbent (A).")
                print("Type A or B (T = tie → random winner).\n")
                S["printed_help"] = True
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution

        # A vs B
        print("\n-------------------------------------------")
        print("Incumbent [A]\n" + S["incumbent_render"])
        print("\nCandidate [B]\n" + rendered)
        print("\nType A / B / T (tie → random): ")

        # Read user choice
        while True:
            choice = input("> ").strip().upper()
            if choice in {"A", "B", "T"}: 
                break # Valid answer, then exit the loop
            print("Please type A, B, or T.") # Otherwise, ask again.

        if choice == "T":
            choice = random.choice(["A", "B"])
        
        if choice == "B":
            fitness = S["fitness_level"]+1.0 # Make the candidate strictly better
            feedback = "User chose B → candidate survives."
            # Update B as the new incumbent
            S["incumbent_id"] = rid
            S["incumbent_render"] = rendered
            S["incumbent_json"] = data
            S["fitness_level"] = fitness
            chosen_incumbent = solution

        else:
            fitness = S["fitness_level"]-1.0 # Make the candidate strictly worse
            feedback = "User chose A → incumbent survives."
            chosen_incumbent = None

        user_feedback = input("Any feedback for improving the next workout? (press Enter to skip): ").strip()
        if user_feedback:
            # save it somewhere
            S["last_feedback"] = user_feedback
            S["incumbent_feedback"] = user_feedback
            if chosen_incumbent is not None: # Choice B
                chosen_incumbent.last_feedback = user_feedback
        else: 
            S["last_feedback"] = None
            S["incumbent_feedback"] = None
            if chosen_incumbent is not None: # Choice B
                chosen_incumbent.last_feedback = None


        _log(logger, feedback)
        solution.set_scores(fitness=fitness, feedback=feedback)
        print("=== DEBUG evaluate_HIIT ===")
        print("System feedback:", feedback)
        print("Candidate (this solution) user feedback:", getattr(solution, "last_feedback", None))
        print("S['last_feedback']:", S.get("last_feedback"))
        print("S['incumbent_feedback']:", S.get("incumbent_feedback"))
        print("===========================")
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
        # Reset _CHOICE_STATE
        reset_choice_state()

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