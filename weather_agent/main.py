import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

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

def main():
    user_query = input("> ")
    response = client.chat.completions.create(
        model="gemini-3.5-flash",
        messages=[
            { "role": "user", "content": user_query }
        ]
    )

    print(f"Response : {response.choices[0].message.content}")

main()

