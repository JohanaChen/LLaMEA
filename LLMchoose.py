import os
import json
import numpy as np
import random
# from ioh import get_problem, logger
import re
import inspect
import math
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
    
    _CHOICE_STATE = {}

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

    # def _coerce_json(x):
    #     """Return a dict from either a dict or a JSON string/bytes (handles code fences)."""
    #     if isinstance(x, dict):
    #         return x
    #     if isinstance(x, (bytes, bytearray)):
    #         x = x.decode("utf-8", "ignore")
    #     if isinstance(x, str):
    #         s = x.strip()
    #         # strip ```json ... ``` fences if present
    #         if s.startswith("```"):
    #             s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
    #         return json.loads(s)
    #     raise TypeError(f"Unsupported solution.data type: {type(x)}")
    def _coerce_json(x):
        """Return a dict from either a dict or a JSON string (handles code fences and stray text)."""
        if isinstance(x, dict):
            return x
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", "ignore")
        if isinstance(x, str):
            s = x.strip()
            # Strip markdown fences if present
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
            # Extract the first {...} JSON block even if there's extra text around it
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                s = m.group(0)
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
                                        "response_mime_type": "application/json",
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

    def llm_decide(llm, incumbent, candidate):
        judge_prompt = f"""
        You are a professional trainer. Your job is to compare two HIIT (High-Intensity Interval Training) programs, labeled A (incumbent) and B (candidate). 

        Program A:
        {incumbent}

        Program B:
        {candidate}

        Task: 
        Choose the better program based on the following criteria:
        1. Variety: A good program should include a mix of different exercises to target various muscle groups and avoid monotony.
        2. Balance: The program should balance cardio, strength, and flexibility exercises.
        3. Structure: Look for well-defined structures like circuits, EMOM (Every Minute on the Minute), AMRAP (As Many Rounds As Possible), etc.
        4. Duration and Rest: Ensure the total duration is between 10-40 minutes, with appropriate work and rest intervals.
        5. Suitability: The program should be suitable for intermediate-level individuals with moderate fitness.
        Respond in JSON format:
            {{
                "choice": "A" or "B", 
                "explanation": "Brief explanation of your choice."
            }}
        """
        response = _llm_complete(llm, judge_prompt, temperature=0.2, max_tokens=1500)
        # if hasattr(response, "text"):
        #     response = response.text

        try:
            data = _coerce_json(response)
            choice = data.get("choice", "A")
            explanation = data.get("explanation", "")
        except Exception:
            choice =  "A"
            explanation = f"Failed to parse LLM response: {response}"
        return choice, explanation

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
                print("Each new candidate (B) is compared to the incumbent (A) by the LLM judge.\n")
                S["printed_help"] = True
            _log(logger, feedback)
            solution.set_scores(fitness=fitness, feedback=feedback)
            return solution

        # A vs B
        print("\n-------------------------------------------")
        print("Incumbent [A]\n" + S["incumbent_render"])
        print("\nCandidate [B]\n" + rendered)

        choice, explanation = llm_decide(llm, S["incumbent_render"], rendered)
        print(f"\nLLM judge chose: {choice}")
        print(f"Reason: {explanation}")

        if choice == "B":
            fitness = S["fitness_level"] + 1.0
            feedback = f"LLM chose candidate (B). Reason: {explanation}"
            # Update
            S["incumbent_id"] = rid
            S["incumbent_render"] = rendered
            S["incumbent_json"] = data
            S["fitness_level"] = fitness
            S["last_feedback"] = explanation
            S["incumbent_feedback"] = explanation
            solution.last_feedback = explanation
            chosen_incumbent = solution
        else:
            fitness = S["fitness_level"] - 1.0
            feedback = f"LLM chose incumbent (A). Reason: {explanation}"
            S["last_feedback"] = explanation
            S["incumbent_feedback"] = explanation
            chosen_incumbent = None  # incumbent remains unchanged

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
        