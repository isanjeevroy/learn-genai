import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"

    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    
    return "Something went wrong"

available_tools = {
    "get_weather": get_weather
}

SYSTEM_PROMPT = """
    You are an expert AI Assistant in resolving user queris using chain of thought.
    You work on START, PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think engough PLAN has done, finally you can give an OUTPUT.
    You can also call a tool if required from the list of available tools.
    for every tool call wait for the observe the step which is the output from called tool.

    Rules.
    - Strictly Follow the given JSON output format.
    - Only run on step at a time.
    - The sequence of steps is START (where user givs an input), PLAN (That can be multiple times) and finally OUTPUT (Which is going to be displayed to the user).

    Output JSON Format:
    { "step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string", "tool": "string", "input":"string" }

    Available Tools:
    - get_weather(city: str): Takes city name as an input string and return the weather info about the city.

    Example 1:
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


    Example 2:
    START: What is the weather of Hyderabad?
    PLAN: { "step": "PLAN": "content" "Seems like user is interested in getting weather of Hyderabad in India" }
    PLAN: { "step": "PLAN": "content": "Lets see if we have any available tool from the list of available tools" }
    PLAN: { "step": "PLAN", "content": "Great, we have get_weather tool avilable for this query." }
    PLAN: { "step": "PLAN", "content": "I need to call get_weatherr tool for Hyderabad as input for city" }
    PLAN: { "step": "TOOL", "tool": "get_weather", "input": "Hyderabad" }
    PLAN: { "step": "OBSERVE", "tool": "get_weather", "output": "The temp of Hyderabad i cloudy with 23" }
    PLAN: { "step": "PLAN", "content": "Great, I got the weather info about Hyderabad" }
    OUTPUT: { "step": "OUTPUT", "content": "The current weather in Hyderabad is 23 C with some cloudy." }

"""

print("\n\n\n")

class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="The optional string content for the step")
    tool: Optional[str] = Field(None, description="The ID of the tool to call.")
    input: Optional[str] = Field(None, description="The input params for the tool")

message_history = [
    {   "role": "system", "content": SYSTEM_PROMPT },
]

while True:
    user_query = input("👉 ")
    message_history.append({ "role": "user", "content": user_query })

    while True:
        response = client.chat.completions.parse(
            model="gemini-3.5-flash",
            response_format=MyOutputFormat,
            messages= message_history
        )

        raw_result = response.choices[0].message.content
        message_history.append({ "role": "assistant", "content": raw_result })

        parsed_result = response.choices[0].message.parsed

        if parsed_result.step == "START":
            print("🔥", parsed_result.content)
            continue

        if parsed_result.step == "TOOL":
            tool_to_call = parsed_result.tool
            tool_input = parsed_result.input
            print(f"🛠️: {tool_to_call} ({tool_input})")

            tool_response = available_tools[tool_to_call](tool_input)
            print(f"🛠️: {tool_to_call} ({tool_input}) = {tool_response}")
            message_history.append({"role":"developer","content":json.dumps({"step":"OBSERVE","tool":tool_to_call,"input":tool_input, "output":tool_response})})
            continue

        if parsed_result.step == "PLAN":
            print("🧠", parsed_result.content)
            continue

        if parsed_result.step == "OUTPUT":
            print("🖥️", parsed_result.content)
            break

    print("\n\n\n")
