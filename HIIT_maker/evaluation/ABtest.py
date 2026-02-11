import numpy as np
import random
import json
import os
import sys
from utils.json_tools import _coerce_json
from utils.logging_utils import _log
from utils.state_utils import reset_choice_state
import utils.state as state 
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
    
    def _sec_to_min(self, sec):
        return round(sec / 60, 1)

    def _render_program_markdown(self, data):
        """Pretty text for the terminal (no external libs). Shows sections + main_set structure/rounds."""
        def _safe_list(x):
            return x if isinstance(x, list) else []

        def _safe_dict(x):
            return x if isinstance(x, dict) else {}

        def _fmt_exercise(i, ex):
            name = ex.get("name", "Exercise")
            dur = ex.get("duration", 0)
            rest = ex.get("rest", 0)
            etype = ex.get("type", "")
            intensity = ex.get("intensity", "")
            tag = " · ".join([t for t in [etype, intensity] if t])
            tag = f"  ·  {tag}" if tag else ""
            return f"{i:>2}. {name}: {dur}s + {rest}s rest{tag}"

        warm_up = _safe_list(data.get("warm_up"))
        main_set = _safe_dict(data.get("main_set"))
        main_exercises = _safe_list(main_set.get("exercises"))
        cool_down = _safe_list(data.get("cool_down"))

        structure = main_set.get("structure") or main_set.get("type") or "Structured"
        rounds = main_set.get("rounds")

        wu_work = sum(int(ex.get("duration", 0)) for ex in warm_up)
        wu_rest = sum(int(ex.get("rest", 0)) for ex in warm_up)

        ms_work_one = sum(int(ex.get("duration", 0)) for ex in main_exercises)
        ms_rest_one = sum(int(ex.get("rest", 0)) for ex in main_exercises)

        cd_work = sum(int(ex.get("duration", 0)) for ex in cool_down)
        cd_rest = sum(int(ex.get("rest", 0)) for ex in cool_down)

        # Apply rounds to main set
        ms_work = ms_work_one * rounds
        ms_rest = ms_rest_one * rounds

        total_work = wu_work + ms_work + cd_work
        total_rest = wu_rest + ms_rest + cd_rest
        total = total_work + total_rest

        # Convert to minutes
        total_work_m = self._sec_to_min(total_work)
        total_rest_m = self._sec_to_min(total_rest)
        total_m = self._sec_to_min(total)

        # Header
        round_str = f" | Rounds: {rounds}" if rounds is not None else ""
        header = f"{structure}{round_str} — ~{total_m}min total (work {total_work_m}min / rest {total_rest_m}min)"

        lines = [header, ""]

        idx = 1

        # Warm-up section
        if warm_up:
            lines.append("=== Warm-up ===")
            for ex in warm_up:
                lines.append(_fmt_exercise(idx, ex))
                idx += 1
            lines.append("")

        # Main set section
        if main_exercises:
            lines.append("=== Main set ===")
            # Print main set meta clearly
            meta = f"Structure: {structure}"
            if rounds is not None:
                meta += f" | Rounds: {rounds}"
            lines.append(meta)
            lines.append("")
            for ex in main_exercises:
                lines.append(_fmt_exercise(idx, ex))
                idx += 1
            lines.append("")

        # Cool-down section
        if cool_down:
            lines.append("=== Cool-down ===")
            for ex in cool_down:
                lines.append(_fmt_exercise(idx, ex))
                idx += 1

        return "\n".join(lines) if len(lines) > 1 else "(empty program)"

    
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

        S = state._CHOICE_STATE 

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

        # Re-render incumbent A from its saved JSON
        try:
            incumbent_render = self._render_program_markdown(S["incumbent_json"])
        except Exception:
            incumbent_render = json.dumps(S["incumbent_json"], indent=2)
        # A vs B
        print("\n-------------------------------------------")
        print("Incumbent [A]\n" + incumbent_render)
        print("\n-------------------------------------------")
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
            _log(logger, feedback)
            sys.exit(0)

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
            S["feedback_memory"].append(user_feedback)
            S["last_feedback"] = user_feedback
            S["incumbent_feedback"] = user_feedback
            if chosen_incumbent is not None: # Choice B
                chosen_incumbent.last_feedback = user_feedback

        _log(logger, feedback)
        solution.set_scores(fitness=fitness, feedback=feedback)
        return solution
    