import json, re

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
