# Author: Max Vogeltanz, University of Graz, 2026

# Script for logic specific to API provided by Open AI.
# Gets called in __init__.py in the same folder

import os
from openai import OpenAI
from .base import GenResult, Usage

class OpenAIClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float):
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return GenResult(text=text, usage=usage)

