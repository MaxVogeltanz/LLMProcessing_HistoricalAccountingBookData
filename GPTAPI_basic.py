# Simple script to use Open AI API for basic prompting and output in console.

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()
api_key_present = os.getenv("OPENAI_API_KEY") is not None
print(f"OPENAI_API_KEY loaded: {api_key_present}")

# Initialize the OpenAI client
client = OpenAI()

# Define model, prompts and further parameters
response = client.chat.completions.create(
    model="gpt-4o", # current model
    temperature=0, 
    # 0 means each output is as deterministic as possible
    # note that some models like gpt-5-nano do not accept temperature setting at all
    messages=[
        {"role": "system", "content": "You are a helpful assistant"}, # system prompt can be left out altogether depending on your use case
        {"role": "user", "content": "Hello"} # your prompt
    ],
)
print(response.choices[0].message.content)
