# Author: Max Vogeltanz, University of Graz, 2026

# Script to initialize different providers for LLM API Processing in project "Aldersbach Digital"


from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .mistral_client import MistralClient
from .gemini_client import GeminiClient

def make_client(provider: str):
    p = provider.lower()
    if p == "anthropic":
        return AnthropicClient()
    if p == "openai":
        return OpenAIClient()
    if p == "mistral":
        return MistralClient()
    if p == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown provider: {provider}")
