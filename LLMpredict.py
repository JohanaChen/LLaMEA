import os
import json
import numpy as np
# from ioh import get_problem, logger
import re
import inspect
import math
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA, Gemini_LLM, OpenAI_LLM, Ollama_LLM
from typing import Dict, Any
from functools import partial
import logging
from typing import Dict, Any, List


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

    # add default information about the user
    SCHEMA = """
    Output STRICT JSON only:
    {
    "per_interval_hr": [
        {"name": "string", "start_sec": int, "end_sec": int, "avg_hr": int, "peak_hr": int}
    ],
    "per_interval_power": [
        {"name": "string", "start_sec": int, "end_sec": int, "avg_power": float, "peak_power": float}
    ],
    "summary": {
        "avg_hr": int,
        "max_hr": int,
        "avg_power": float,
        "max_power": float,
        "time_in_zones": {"z1": int, "z2": int, "z3": int, "z4": int, "z5": int},
        "zone_cutoffs": {"z1": float, "z2": float, "z3": float, "z4": float, "z5": float}
    },
    "assumptions": "string"
    }
    Rules:
    - Return ONLY the JSON object (no code fences, no extra text).
    - Do NOT include timeline_hr.
    - HR in bpm (integers). Power in watts (floats).
    - HR rises on work, falls on rest with ~20-60s lag; plausible 4-210 bpm.
    - Power rises sharply on work and drops near zero on rest; plausible 0-600 W.
    - Sum(time_in_zones) must equal total workout duration (seconds).
    """

    def make_prompt(program_text: str) -> str:
        return (
            "You are a professional exercise physicologist. Predict the user's heart rate and power output responses throughout the HIIT program. \n"
            + SCHEMA 
            + "\nPROGRAM:\n\"\"\"\n" + program_text.strip() + "\n\"\"\"\n"
            "Return JSON now."
        )
    
    def predict_hr_power(llm_eval, prompt: str, max_retries: int = 1):
        """Predict both heart rate and power output for the HIIT program."""
        raw = _llm_complete(llm_eval, prompt, temperature=0.2, max_tokens=1500)  # bump tokens a bit
        try:
            s, e = raw.find("{"), raw.rfind("}")
            candidate = raw[s:e+1] if s != -1 and e != -1 else raw
            data = _coerce_json(candidate)

            required_keys = ["summary", "per_interval_hr", "per_interval_power"]
            if not all(k in data for k in required_keys):
                raise ValueError("Missing required keys in predicted_hr_power JSON")
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
        raw2 = _llm_complete(llm_eval, repair_prompt, temperature=0.0, max_tokens=1500)
        s, e = raw2.find("{"), raw2.rfind("}")
        candidate = raw2[s:e+1] if s != -1 and e != -1 else raw2
        data = _coerce_json(candidate)

        required_keys = ["summary", "per_interval_hr", "per_interval_power"]
        if not all(k in data for k in required_keys):
            raise ValueError(f"Evaluation LLM did not return valid JSON.\nSnippet:\n{raw2[:800]}")
        
        return data

                
    # def fitness_hr(pred: Dict[str, Any], hr_max: int) -> float:
    #     s = pred["summary"]
    #     tz = s.get("time in zones", {})
    #     total = sum(tz.values()) or 1 # or 1 prevents divisions by 0 if the dict is empty.
    #     reward = tz.get("z3",0) + tz.get("z4",0) # Reward for the amount of time spent in the moderate-hard zones (z3 and z4).
    #     penalty = 2.0 * tz.get("z5", 0) # Penalty for the anount of time spent in zone 5.
    #     max_over = max(0, s.get("max_hr", 0) - int(0.98*hr_max))
    #     penalty = penalty + 5.0 * max_over # Penalize for pushing beyond safe limits. 
    #     raw = (reward-penalty)/total
    #     return max(0, min(1, 0.5+0.5*raw)) # Score goes between 0 and 1.

    def fitness(pred: Dict[str, Any], hr_max: int = 190, weights: Dict[str, float] = None) -> float:
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
        exercises += main.get("exercises", []) if isinstance(main.get("exercises"), list) else []

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

        
    
    def _infer_hr_max(pred, default=200):
        txt = (pred.get("assumptions") or "") + " " + str(pred.get("summary", {}))
        m = re.search(r'(\d{2,3})\s*(?:bpm|HRmax)', txt, flags=re.I)
        if m:
            val = int(m.group(1))
            if 150 <= val <= 220:
                return val
        return default

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
    
    def _render_program_text(program):
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

    def _extract_text_from_response(resp):
        # Gemini SDK common
        if hasattr(resp, "text") and resp.text:
            return resp.text
        cand = getattr(resp, "candidates", None)
        if cand:
            try:
                parts = getattr(getattr(cand[0], "content", None), "parts", [])
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if texts:
                    return "\n".join(texts)
            except Exception:
                pass
        # OpenAI-like
        ch = getattr(resp, "choices", None)
        if ch:
            try:
                msg = getattr(ch[0], "message", None)
                if msg and getattr(msg, "content", None):
                    return msg.content
                if hasattr(ch[0], "text"):
                    return ch[0].text
            except Exception:
                pass
        # Dict-like
        if isinstance(resp, dict):
            for k in ("text","response","output","content"):
                if isinstance(resp.get(k), str):
                    return resp[k]
            if "choices" in resp and resp["choices"]:
                if "message" in resp["choices"][0]:
                    return resp["choices"][0]["message"].get("content","")
                if "text" in resp["choices"][0]:
                    return resp["choices"][0]["text"]
        raise RuntimeError("Cannot extract text from response object")

    def _maybe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError:
            try:
                return fn(*args)  # try positional only
            except TypeError:
                return fn(kwargs) # some wrappers expect a single payload

    def _find_gemini_model(obj):
        # look for a GenerativeModel nested inside common attributes
        for name in ("model","_model","client","_client","gem","gm"):
            m = getattr(obj, name, None)
            if hasattr(m, "generate_content") and callable(m.generate_content):
                return m
        # brute-force scan of attributes
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                m = getattr(obj, name)
            except Exception:
                continue
            if hasattr(m, "generate_content") and callable(m.generate_content):
                return m
        return None

    def _llm_complete(llm, prompt: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        """Accepts many client shapes and returns plain text."""
        # 1) Your wrapper with .complete(prompt, …)
        if hasattr(llm, "complete") and callable(llm.complete):
            return llm.complete(prompt, temperature=temperature, max_tokens=max_tokens)

        # 2) Raw Gemini model OR wrapper holding one
        model = None
        if hasattr(llm, "generate_content") and callable(llm.generate_content):
            model = llm
        else:
            model = _find_gemini_model(llm)

        if model is not None:
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature,
                                        "max_output_tokens": max_tokens,
                                        # "response_mime_type": "application/json",
                                        },
                )
            except TypeError:
                resp = model.generate_content(prompt)
            return _extract_text_from_response(resp)

        # 3) OpenAI v1 chat client shape
        if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
            create = llm.chat.completions.create
            resp = _maybe_call(create,
                model=getattr(llm, "model", None) or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens
            )
            return _extract_text_from_response(resp)

        # 4) Generic .generate(...) or .predict(...) or callable
        for meth in ("generate","predict","invoke","ask","run","__call__"):
            if hasattr(llm, meth) and callable(getattr(llm, meth)):
                fn = getattr(llm, meth)
                sig = None
                try:
                    sig = inspect.signature(fn)
                except Exception:
                    pass
                if sig and "prompt" in sig.parameters:
                    resp = _maybe_call(fn, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
                else:
                    resp = _maybe_call(fn, prompt)
                # If it already returned a string, great; else extract text
                return resp if isinstance(resp, str) else _extract_text_from_response(resp)

        raise AttributeError(f"Unsupported LLM client: no known completion method (type={type(llm).__name__})")

    def evaluate_HIIT(solution, logger=None, llm_eval=None):
        feedback = ""
        program = []
        # llm_eval = ctx.get("llm_eval") or ctx.get("llm")
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
                
            program_text = _render_program_text(program)
            prompt = make_prompt(program_text)
            predictions = predict_hr_power(llm_eval, prompt)

            # Extract heart rate and power predictions
            predicted_hr = predictions["summary"].get("avg_hr", None)
            predicted_power = predictions["summary"].get("avg_power", None)

            hr_max = _infer_hr_max(predicted_hr, default=200)
            fitness = fitness(data, hr_max)
            # fitness = fitness_hr(predicted_hr, hr_max=200)

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
            data["fitness"] = fitness
            # Store back to solution.data as a JSON string
            # solution.data = json.dumps(data, ensure_ascii=False, indent=2)
            solution.data = data

            feedback = (
                f"Evaluated with HR/Power prediction: {num_exercises} exercises, "
                f"total duration {total_duration}s, "
                f"fitness {fitness:.3f}"
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
        eval_fn = partial(evaluate_HIIT, llm_eval=llm)

        es = LLaMEA(
            # f=evaluate_HIIT,
            f=eval_fn,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name="hiit1",
            elitism=True,
            HPO=False,
            budget=10,
        )
        # print(es.run())

        result = es.run()

        # Save the final surviving HIIT program
        final_program = result.data if hasattr(result, "data") else result

        # Ensure it's a dictionary (not string)
        if isinstance(final_program, str):
            try:
                final_program = json.loads(final_program)
            except json.JSONDecodeError:
                print("Warning: final program is not valid JSON, saving as raw text.")

        # Define output folder (assuming same folder as your existing outputs)
        output_dir = os.path.join(os.getcwd(), "outputs")  # adjust if your path differs
        os.makedirs(output_dir, exist_ok=True)

        # Define file name and full path
        final_path = os.path.join(output_dir, "final_choice.json")

        # Save JSON
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_program, f, ensure_ascii=False, indent=2)

        print(f"✅ Final surviving HIIT program saved to: {final_path}")
