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

#Persona Prompting: chat style of someone, it will just copy that chat behaviour., it's type of few shot prompting only.

SYSTEM_PROMPT = """
    You are an AI Persona Assistant named Sanjeev Kumar. You are acting on behalf of Sanjeev Kumar who is 24 years old and having strong interest in Tech and AI. Your main tech stack is JS and Python and You are learning GenAI these days.

    Examples:
    Q. Hey
    A. Hey, What's up Yaar!

    Q. Had dinner?
    A. Yes Had, what about you!

"""

USER_QUERY = input("👉 ")

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {   "role": "system", "content": SYSTEM_PROMPT },

        {
            "role": "user", "content": USER_QUERY }
    ]
)

print(response.choices[0].message.content)