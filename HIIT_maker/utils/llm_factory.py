from dotenv import load_dotenv
import os
import google.generativeai as genai
from llamea.llm import Gemini_LLM
from llamea.llm import OpenAI_LLM
from llamea.llm import Ollama_LLM
from llamea.llm import DeepSeek_LLM


def make_llm(llm_cfg: dict):
    """
    llm_cfg example:
    {
        "provider": "gemini",
        "model": "gemini-flash-latest"
    }
    """
    provider = llm_cfg["provider"]
    model = llm_cfg["model"]

    load_dotenv()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)
        return Gemini_LLM(api_key, model)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        return OpenAI_LLM(api_key=api_key, model=model)
    
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")

        return DeepSeek_LLM(api_key=api_key, model=model)

    elif provider == "ollama":
        # Ollama runs locally → no API key
        return Ollama_LLM(model=model)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


