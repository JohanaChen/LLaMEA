# HIIT_maker/utils/io_utils.py
import os
import json
from datetime import datetime

def save_json_result(data, name_prefix="hiit_program"):
    """Save a JSON object to the Results folder with a timestamped filename."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "Results")
    os.makedirs(os.path.abspath(results_dir), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name_prefix}_{timestamp}.json"
    filepath = os.path.abspath(os.path.join(results_dir, filename))

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved result to {filepath}")
    return filepath
