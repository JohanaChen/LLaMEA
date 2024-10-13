"""LLM manager to connect to different types of models.
"""
import google.generativeai as genai
import openai
import transformers
import torch


class LLMmanager:
    """LLM manager, currently only supports ChatGPT models."""

    def __init__(self, api_key, model="gpt-4-turbo"):
        """Initialize the LLM manager with an api key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, 
                Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
                Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
                CodeLlama-7b-Instruct-hf, CodeLlama-13b-Instruct-hf,
                CodeLlama-34b-Instruct-hf, CodeLlama-70b-Instruct-hf,
        """
        self.api_key = api_key
        self.model = model
        if "gpt" in self.model:
            self.client = openai.OpenAI(api_key=api_key)
        if "gemini" in self.model:
            genai.configure(api_key=api_key)
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            self.client = genai.GenerativeModel(
                model_name=self.model,  # "gemini-1.5-flash",
                generation_config=generation_config,
                # safety_settings = Adjust safety settings
                # See https://ai.google.dev/gemini-api/docs/safety-settings
                system_instruction="You are a computer scientist and excellent Python programmer.",
            )
        if "Llama" in self.model:
            model_id = f"meta-llama/{self.model}"
            self.client = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            self.llama_messages = [
                {
                    "role": "system", "content":
                    "You are a computer scientist and excellent Python programmer."
                },
            ]

    def chat(self, session_messages):
        if "gpt" in self.model:
            response = self.client.chat.completions.create(
                model=self.model, messages=session_messages, temperature=0.8
            )
            return response.choices[0].message.content
        elif "gemini" in self.model:
            history = []
            last = session_messages.pop()
            for msg in session_messages:
                history.append(
                    {
                        "role": msg["role"],
                        "parts": [
                            msg["content"],
                        ],
                    }
                )
            chat_session = self.client.start_chat(history=history)
            response = chat_session.send_message(last["content"])
            return response.text
        elif "Llama" in self.model:
            history = []
            for msg in session_messages:
                history.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )
            response = self.client(history, max_new_tokens=8192)
            return response[0]["generated_text"][-1]['content']
