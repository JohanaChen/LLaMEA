import numpy as np
import random
import json
import os
from utils.json_tools import _coerce_json
from utils.logging_utils import _log
from utils.state_utils import reset_choice_state
from utils.state import _CHOICE_STATE
from HIIT_maker.utils.io_utils import save_json_result
from HIIT_maker.utils.llm_utils import extract_text_from_response

class LLMpredict:
    """ Class to manage LLM predictions on HR and power for a given HIIT program. """
    def __init__(self, logger=None):
        self.logger = logger
        self.llm_eval = llm_eval
        self.SCHEMA = self._schema_text()

    def _schema_text(self):
        return """
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
    
    def make_prompt(self, program_text: str) -> str:
        return (
            "You are a professional exercise physicologist. Predict the user's heart rate and power output responses throughout the HIIT program. \n"
            + self.SCHEMA 
            + "\nPROGRAM:\n\"\"\"\n" + program_text.strip() + "\n\"\"\"\n"
            "Return JSON now."
        )
    
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
    

        
    
