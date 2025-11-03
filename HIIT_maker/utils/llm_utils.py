import inspect

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
                                    "max_output_tokens": max_tokens},
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
            