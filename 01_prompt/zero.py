#Zero shot Prompting
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

#Zero Shot Prompting: Directly giving the instruction to the model

SYSTEM_PROMPT = "You are a mathematic teacher. just explai me in simple only and only mathematic no other anything."

response = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[
        {   "role": "system", "content": SYSTEM_PROMPT },

        {
            "role": "user", "content": "can you explain me whole square of a + b"
        }
    ]
)

print(response.choices[0].message.content)