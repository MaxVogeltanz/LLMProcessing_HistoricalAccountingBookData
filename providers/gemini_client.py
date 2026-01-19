# Author: Max Vogeltanz, University of Graz, 2026

# Script for logic specific to API provided by Gemini (Google).
# Gets called in __init__.py in the same folder

import os
from google import genai
from google.genai import types
from .base import GenResult, Usage

class GeminiClient:
    def __init__(self, *, api_key_env: str = "GEMINI_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not found.")
        self.client = genai.Client(api_key=api_key)
        self.api_key_env = api_key_env  # optional (for logging)

    def generate(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float):
        resp = self.client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        text = (resp.text or "").strip()

        usage_meta = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
        )
        return GenResult(text=text, usage=usage)

