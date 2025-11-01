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