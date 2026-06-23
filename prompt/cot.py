#Zero shot Prompting
import os
from dotenv import load_dotenv
from openai import OpenAI

import json

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

#Zero Shot Prompting: Directly giving the instruction to the model

SYSTEM_PROMPT = """
    You are an expert AI Assistant in resolving user queris using chain of thought.
    You work on START, PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think engough PLAN has done, finally you can give an OUTPUT.

    Rules.
    - Strictly Follow the given JSON output format.
    - Only run on step at a time.
    - The sequence of steps is START (where user givs an input), PLAN (That can be multiple times) and finally OUTPUT (Which is going to be displayed to the user).

    Output JSON Format:
    { "step": "START" | "PLAN" | "OUTPUT", "content": "string" }

    Example:
    START: Hey, can you solve 2 + 3 * 5 /10

    PLAN: { "step": "PLAN": "content" "Seems like user is interested in math problem" }

    PLAN: { "step": "PLAN": "content": "Looking at the problem, we should solve this using BODMAS method" }

    PLAN: { "step": "PLAN", "content": "Yes, The BODMAS is correct thing to be done here" }

    PLAN: { "step": "PLAN", "content": "first we must multiply 3 * 5 which is 15" }

    PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 15 / 10" }

    PLAN: { "step": "PLAN", "content": "We must perform divide that is 15 / 10 = 1.5" }

    PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 1.5" }

    PLAN: { "step": "PLAN", "content": "Now finally lets perform the add 2 + 1.5 with 3.5 as ans" }

    OUTPUT: { "step": "OUTPUT", "content": "3.5" }
"""

response = client.chat.completions.create(
    model="gemini-3.5-flash",
    response_format={"type":"json_object"},
    messages=[
        {   "role": "system", "content": SYSTEM_PROMPT },

        {
            "role": "user", "content": "Hey, Write a code to add n numbers in javascript"
        },
        #manually adding the query here
        {
            "role": "assistant", "content": json.dumps({
                "step": "START",
                "content":"You want a Javascript code to add n numbers."
            })
        }
        # Like that be adding manully till getting output.
    ]
)

print(response.choices[0].message.content)