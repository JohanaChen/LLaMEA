import json, re

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
    """Return a dict from either a dict or a JSON string/bytes (handles code fences and partial JSON)."""
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", "ignore")
    if isinstance(x, str):
        s = x.strip()
        # Remove Markdown code fences if present
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
        
        # Try normal JSON parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Try to truncate after the last closing brace to salvage partial JSON
            end = s.rfind("}")
            if end != -1:
                try:
                    return json.loads(s[:end+1])
                except Exception:
                    pass
        # If nothing works, raise clean error
        raise ValueError(f"Invalid JSON or incomplete generation:\n{s[:300]}...")
    raise TypeError(f"Unsupported type for JSON coercion: {type(x)}")
