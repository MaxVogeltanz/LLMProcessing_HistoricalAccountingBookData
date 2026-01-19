# Author: Max Vogeltanz, University of Graz, 2026

# Script for logic specific to API provided by Mistral.
# Gets called in __init__.py in the same folder

import os
from mistralai import Mistral
from .base import GenResult, Usage

class MistralClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found.")
        self.client = Mistral(api_key=api_key)

    def generate(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float):
        resp = self.client.chat.complete(
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
