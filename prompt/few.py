#Few shot prompting
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

#Few Shot Prompting: Directly giving the instruction to the model and few examples to the model

SYSTEM_PROMPT = """
You are a mathematics teacher.

Rules:
1. Answer only mathematics questions.
2. Explain in very simple English.
3. Show step-by-step solutions.
4. Always return valid JSON.
5. If the question is not related to mathematics, return an error message in JSON.

Examples:

User: What is 2 + 3?

Assistant:
{
  "is_math": true,
  "question": "What is 2 + 3?",
  "answer": "5",
  "steps": [
    "2 + 3 = 5"
  ]
}

User: Solve x + 5 = 8

Assistant:
{
  "is_math": true,
  "question": "Solve x + 5 = 8",
  "answer": "x = 3",
  "steps": [
    "x + 5 = 8",
    "x = 8 - 5",
    "x = 3"
  ]
}

User: Who is the Prime Minister of India?

Assistant:
{
  "is_math": false,
  "error": "Sorry, I can only help with mathematics questions."
}

"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {   "role": "system", "content": SYSTEM_PROMPT },

        {
            "role": "user", "content": "can you explain me whole square of a + b"
        }
    ]
)

print(response.choices[0].message.content)