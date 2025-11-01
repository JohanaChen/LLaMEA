import numpy as np
import random
import json
import os
from utils.json_tools import _coerce_json
from utils.logging_utils import _log
from utils.state_utils import reset_choice_state
from utils.state import _CHOICE_STATE
from HIIT_maker.utils.io_utils import save_json_result

class ABtest:
    """ Class to manage user A/B testing between two solutions. """
    def __init__(self, logger=None):
        self.logger = logger

    def _solution_id(self, solution):
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

    def _flatten_program(self, data):
        """Collect all exercises into a single list for display/metrics."""
        program = []
        if "warm_up" in data and isinstance(data["warm_up"], list):
            program += data["warm_up"]
        if "main_set" in data and isinstance(data["main_set"], dict) and "exercises" in data["main_set"]:
            program += data["main_set"]["exercises"]
        if "cool_down" in data and isinstance(data["cool_down"], list):
            program += data["cool_down"]
        return program
    
    def _render_program_markdown(self, data):
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

        program = self._flatten_program(data)
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
    
    def evaluate_HIIT(self, solution, logger=None):
        
        # Parse JSON
        try: 
            data = _coerce_json(solution.data)
        except Exception as e:
            fitness = -np.inf
            feedback = f"Invalid JSON: {e} | Raw text: {solution.data[:200]}..."
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution

        # Save parsed JSON
        try:
            save_json_result(data, name_prefix="hiit_candidate")
        except Exception as e:
            _log(logger, f"Warning: could not save JSON file ({e})", level="warning")

        # Collect exercises
        program = self._flatten_program(data)
        if not program:
            fitness = -np.inf
            feedback = "No exercises found in JSON input."
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution
        
        rid = self._solution_id(solution)
        rendered = self._render_program_markdown(data) # Create readable string for the terminal printout

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
                print("Type STOP if you are satisfied with the current incumbent.\n")
                S["printed_help"] = True
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution

        # A vs B
        print("\n-------------------------------------------")
        print("Incumbent [A]\n" + S["incumbent_render"])
        print("\nCandidate [B]\n" + rendered)
        print("\nType A / B / T (tie → random) or STOP (to finish): ")

        # Read user choice
        while True:
            choice = input("> ").strip().upper()
            if choice in {"A", "B", "T", "STOP"}: 
                break # Valid answer, then exit the loop
            print("Please type A, B, T, or STOP.") # Otherwise, ask again.

        if choice == "STOP":
            save_json_result(S["incumbent_json"], name_prefix="final_result")
            feedback = "User chose to stop — saved incumbent as final program."
            fitness = S["fitness_level"]
            solution.set_scores(fitness=fitness, feedback=feedback)
            raise KeyboardInterrupt

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
        return solution
    