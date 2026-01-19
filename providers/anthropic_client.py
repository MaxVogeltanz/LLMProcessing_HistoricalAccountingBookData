# Author: Max Vogeltanz, University of Graz, 2026

# Script for logic specific to API provided by Anthropic (Claude).
# Gets called in __init__.py in the same folder


import os
from anthropic import Anthropic
from .base import GenResult, Usage

class AnthropicClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found.")
        self.client = Anthropic(api_key=api_key)

    def generate(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float):
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip() if resp.content else ""
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        )
        return GenResult(text=text, usage=usage)
