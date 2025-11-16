import numpy as np
import random
import json
import os
import math
import re
import logging
from typing import Dict, Any, List
from utils.json_tools import _coerce_json
from utils.logging_utils import _log
from utils.state_utils import reset_choice_state
from utils.state import _CHOICE_STATE as S
from HIIT_maker.utils.io_utils import save_json_result
from HIIT_maker.utils.llm_utils import _llm_complete
from HIIT_maker.utils.schema_utils import HIIT_RESPONSE_SCHEMA

class LLMpredict:
    """ Class to manage LLM predictions on HR and power for a given HIIT program. """
    def __init__(self, llm_eval = None, logger=None):
        self.logger = logger
        self.llm_eval = llm_eval
        self.SCHEMA = HIIT_RESPONSE_SCHEMA
    
    def make_prompt(self, program_text: str) -> str:
         return (
            "You are a sports physiologist. Predict an average healthy women's (around 20-30 years) HEART RATE (bpm) "
            "and POWER OUTPUT (watts) over the given HIIT workout.\n\n"
            "Guidelines:\n"
            "- HR rises during work, falls during rest (~20-60s lag). Range: 60-200 bpm.\n"
            "- Power spikes during work, near zero during rest. Range: 0-600 W.\n"
            "- Keep start/end timestamps consistent.\n\n"
            f"JSON Schema:\n{self.SCHEMA}\n\n"
            f"HIIT Program Description:\n\"\"\"\n{program_text.strip()}\n\"\"\"\n\n"
            "Output STRICT JSON, no markdown, no backticks, no text, no single quotes. Do not include explanations, code fences, or text before/after."
            "Begin your response with '{' and end with '}'."
        )
    
    def _render_program_text(self, program):
        """
        Program is a flat list of exercise dicts with at least: name, duration, rest.
        Produces a concise, deterministic description for the evaluation LLM.
        """
        lines = []
        t = 0
        for i, ex in enumerate(program, 1):
            name = ex.get("name", f"ex_{i}")
            dur = int(ex.get("duration", 0))
            rest = int(ex.get("rest", 0))
            lines.append(f"{i:02d}. {name}: work {dur}s, rest {rest}s")
            t += dur + rest
        return f"TOTAL_DURATION_SEC={t}\n" + "\n".join(lines)

    def predict_hr_power(self, llm_eval, prompt: str, max_retries: int = 1, logger=None):
        """Predict both heart rate and power output for the HIIT program."""
        raw = _llm_complete(llm_eval, prompt, temperature=0.2, max_tokens=9000)  # bump tokens a bit
        try:
            s, e = raw.find("{"), raw.rfind("}")
            candidate = raw[s:e+1] if s != -1 and e != -1 else raw
            data = _coerce_json(candidate)
            required_keys = ["summary", "per_interval_hr", "per_interval_power"]
            if not all(k in data for k in required_keys):
                raise ValueError("Missing required keys in predicted_hr_power JSON")
            if not isinstance(data, dict):
                raise ValueError(f"Evaluation LLM did not return a valid JSON object.\nSnippet:\n{raw[:500]}")
            return data
        
        except Exception:
            if max_retries <= 0:
                raise ValueError(f"Evaluation LLM did not return valid JSON.\nSnippet:\n{raw[:800]}")

        # Attempt 2 — have the model repair its own output
        repair_prompt = (
            "The following was intended to be STRICT JSON but is invalid or incomplete. "
            "Return a corrected JSON ONLY (no code fences, no prose), matching the schema I gave earlier.\n\n"
            "INVALID_JSON:\n" + raw[:4000]
        )
        raw2 = _llm_complete(llm_eval, repair_prompt, temperature=0.0, max_tokens=9000)
        s, e = raw2.find("{"), raw2.rfind("}")
        candidate = raw2[s:e+1] if s != -1 and e != -1 else raw2
        data = _coerce_json(candidate)
        required_keys = ["summary", "per_interval_hr", "per_interval_power"]
        
        if not isinstance(data, dict) or not all(k in data for k in required_keys):
            raise ValueError(f"Evaluation LLM did not return valid JSON after repair.\nSnippet:\n{raw2[:800]}")
        return data
    
    def fitness(self, pred: Dict[str, Any], hr_max: int = 190, weights: Dict[str, float] = None) -> float:
        """
        Compute overall HIIT effectiveness score using physiological,
        biomechanical, and engagement proxies.
        Now adapted for input structures where predicted heart rate data
        is under pred["predicted_hr"]["per_interval_hr"].
        """

        # --- Default weights (sum = 1.0) ---
        default_weights = {
            "hr": 0.30,
            "power": 0.25,
            "work": 0.20,
            "intensity": 0.15,
            "variety": 0.10,
        }

        # Use user-provided weights if available
        if weights is None:
            weights = default_weights
        else:
            # Fill in missing keys from defaults
            for k, v in default_weights.items():
                weights.setdefault(k, v)
                
        # --- Check the sum of weight is 1 ---
        total_w = sum(weights.values())
        if not math.isclose(total_w, 1.0, rel_tol=1e-3):
            raise ValueError(f"Sum of weights must be 1.0, got {total_w:.3f}")

        # --- Helper functions ---
        def clamp(x, lo=0.0, hi=1.0):
            return max(lo, min(hi, float(x)))

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def tanh(x):
            return math.tanh(x)

        # --- Gather exercise data ---
        exercises: List[Dict[str, Any]] = []
        exercises += pred.get("warm_up", [])
        exercises += pred.get("cool_down", [])
        main = pred.get("main_set", {})
        # Handle both flat and block-based structures
        if isinstance(main.get("exercises"), list):
            exercises += main["exercises"]

        elif isinstance(main.get("blocks"), list):
            for block in main["blocks"]:
                # Handle "exercises" lists inside blocks
                if isinstance(block.get("exercises"), list):
                    exercises += block["exercises"]
                # Handle "sequence" lists inside blocks (some use this key)
                seq = block.get("sequence", [])
                if isinstance(seq, list):
                    exercises += seq

        elif "finisher_block" in main and isinstance(main["finisher_block"].get("exercises"), list):
            exercises += main["finisher_block"]["exercises"]


        if not exercises:
            return 0.0
        
        # --- Gather HR and power prediction data ---
        hr_predictions = {}
        if "predicted_hr" in pred and isinstance(pred["predicted_hr"], dict):
            for interval in pred["predicted_hr"].get("per_interval_hr", []):
                name = str(interval.get("name", "")).lower().strip()
                hr_predictions[name] = {
                    "avg_hr": interval.get("avg_hr", 0),
                    "peak_hr": interval.get("peak_hr", 0)
                }
        power_predictions = {}
        if "predicted_power" in pred and isinstance(pred["predicted_power"], dict):
            for interval in pred["predicted_power"].get("per_interval_power", []):
                name = str(interval.get("name", "")).lower().strip()
                power_predictions[name] = {
                    "avg_power": interval.get("avg_power", 0.0),
                    "peak_power": interval.get("peak_power", 0.0)
                }

        # --- Derived features ---
        work_ratios, intensities, hr_ratios, power_ratios = [], [], [], []
        types = set()

        for ex in exercises:
            duration = float(ex.get("duration", 0))
            rest = float(ex.get("rest", 1))
            ex_type = str(ex.get("type", "")).lower()
            intensity_label = str(ex.get("intensity", "medium")).lower()
            name = str(ex.get("name", "")).lower().strip()

            # Retrieve matching HR and power data (if available)
            hr_data = hr_predictions.get(name, {})
            power_data = power_predictions.get(name, {})

            pred_hr = float(hr_data.get("avg_hr") or 0.0)
            pred_power = float(power_data.get("avg_power") or 0.0)

            hr_ratio = pred_hr / hr_max if pred_hr > 0 else 0.0
            power_ratio = pred_power / 500.0 if pred_power > 0 else 0.0  
            hr_ratios.append(hr_ratio)
            power_ratios.append(power_ratio)  

            # Work-to-rest ratio (biomechanical density)
            work_ratio = duration / (duration + rest) if (duration + rest) > 0 else 0.0
            work_ratios.append(work_ratio)

            # Intensity label to numeric (physiological load)
            if intensity_label == "high":
                i_val = 0.9
            elif intensity_label == "medium":
                i_val = 0.7
            elif intensity_label == "low":
                i_val = 0.5
            else:
                i_val = 0.6
            intensities.append(i_val)

            # Track exercise types for variety
            if "/" in ex_type:
                parts = [p.strip() for p in ex_type.split("/")]
                types.update(parts)
            else:
                types.add(ex_type.strip())

        # --- Aggregate ---
        avg_hr_ratio = sum(hr_ratios) / len(hr_ratios) if hr_ratios else 0.0
        avg_power_ratio = sum(power_ratios) / len(power_ratios) if power_ratios else 0.0
        avg_intensity = sum(intensities) / len(intensities)
        avg_work_ratio = sum(work_ratios) / len(work_ratios)

        # Variety proxy (psychological engagement)
        all_types = {"cardio", "strength", "core", "mobility", "stability", "flexibility", "recovery"}
        variety = len(types & all_types) / len(all_types)

        # --- Normalization ---
        z_hr = clamp(avg_hr_ratio)
        z_power = clamp(avg_power_ratio)
        z_intensity = clamp(avg_intensity)
        z_work = clamp(avg_work_ratio)
        z_variety = clamp(variety)

        # --- Weighted hybrid formula ---
        u = (
            weights["hr"] * tanh(z_hr) +
            weights["power"] * tanh(z_power) +
            weights["work"] * z_work +
            weights["intensity"] * z_intensity +
            weights["variety"] * z_variety
        )

        score = sigmoid(5 * (u - 0.5))
        return clamp(score)


    # def _infer_hr_max(self, pred, default=200):
    #     txt = (pred.get("assumptions") or "") + " " + str(pred.get("summary", {}))
    #     m = re.search(r'(\d{2,3})\s*(?:bpm|HRmax)', txt, flags=re.I)
    #     if m:
    #         val = int(m.group(1))
    #         if 150 <= val <= 220:
    #             return val
    #     return default
    
    def _infer_hr_max(self, pred, default=200):
        if not isinstance(pred, dict):
            return default
        txt = (pred.get("assumptions") or "") + " " + str(pred.get("summary", {}))
        m = re.search(r'(\d{2,3})\s*(?:bpm|HRmax)', txt, flags=re.I)
        if m:
            val = int(m.group(1))
            if 150 <= val <= 220:
                return val
        return default


    def evaluate_HIIT(self, solution, logger=None, llm_eval=None):
        if llm_eval is None:
            llm_eval = self.llm_eval

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

            if llm_eval is None:
                try: 
                    llm_eval = globals().get("llm")
                except KeyError:
                    raise RuntimeError("llm_eval is None and no global 'llm' found.")
                
            program_text = self._render_program_text(program)
            prompt = self.make_prompt(program_text)
            predictions = self.predict_hr_power(llm_eval, prompt)
            if predictions is None:
                raise ValueError("❌ LLM returned None instead of predictions.")

            # Extract heart rate and power predictions
            predicted_hr = predictions["summary"].get("avg_hr", None)
            predicted_power = predictions["summary"].get("avg_power", None)

            hr_max = self._infer_hr_max(predictions, default=200)

            total_work = sum(int(ex.get("duration", 0)) for ex in program)
            total_rest = sum(int(ex.get("rest", 0)) for ex in program)
            total_duration = total_work + total_rest
            num_exercises = len(program)

            # --- Write predictions back into JSON and persist on solution ---
            data["predicted_hr"] = {
                "summary": predictions.get("summary", {}),
                "per_interval_hr": predictions.get("per_interval_hr", [])
            }

            data["predicted_power"] = {
                "per_interval_power": predictions.get("per_interval_power", [])
            }

            # --- Compute fitness ---
            fitness_score = self.fitness(data, hr_max)

            data["fitness"] = fitness_score
            # Store back to solution.data as a JSON string
            # solution.data = json.dumps(data, ensure_ascii=False, indent=2)
            solution.data = data
            save_json_result(data, name_prefix="hiit_candidate_with_predictions")   

            feedback = (
                f"Evaluated with HR/Power prediction: {num_exercises} exercises, "
                f"total duration {total_duration}s, "
                f"fitness {fitness_score:.3f}"
            )

        except json.JSONDecodeError as e:
            fitness_score = -np.inf
            feedback = f"Invalid JSON: {e} | Raw text: {solution.data}"
        except Exception as e:
            fitness_score = -np.inf
            feedback = f"Error during evaluation: {e} | Raw text: {solution.data}"

        _log(logger, feedback)

        solution.set_scores(fitness=fitness_score, feedback=feedback)

        # Update glocal choice
        try :
            if S["incumbent_id"] is None or fitness_score > S["fitness_score"]:
                S["incumbent_id"] = getattr(solution, "id", "unkown")
                S["incumbent_json"] = getattr(solution, "data", {})
                S["incumbent_render"] = getattr(solution, "description", "")
                S["fitness_score"] = fitness_score
        except Exception as e:
            if logger:
                logger.warning(f"[Hybrid] Could not update incumbent after LLMpredict: {e}")

        return solution

    def get_top_candidates(self, population, n=2):
        """Return the top-n individuals from the current population."""
        if not population:
            raise ValueError("Empty population passed to get_top_candidates().")
        sorted_pop = sorted(
            population,
            key=lambda ind: getattr(ind, "fitness", float("-inf")),
            reverse=True
        )
        return sorted_pop[:n]
    
    def integrate_user_feedback(self, feedback):
        """Optionally store or use user feedback."""
        if not feedback:
            return
        if self.logger:
            self.logger.info(f"Integrating user feedback: {feedback}")
        self.last_feedback = feedback

    
