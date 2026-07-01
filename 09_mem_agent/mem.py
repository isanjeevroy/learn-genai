#Install SDK
from mem0 import MemoryClient
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import os

load_dotenv(Path(__file__).parent.parent / ".env")

openai_client = OpenAI()

#Initialize the client
MEM0_API_KEY = os.environ.get("MEM0_API_KEY")
client = MemoryClient(api_key=MEM0_API_KEY)

while True:
    user_query = input("> ")

    if user_query.lower() == "exit":
        break

    # Search memories
    search_memory = client.search(
        query=user_query,
        filters={
            "user_id": "sanjeev"
        }
    )

    memories = "\n".join(
    memory["memory"] for memory in search_memory["results"]
    )

    SYSTEM_PROMPT = f"""
        Here is the context about the user: 
        {memories}
    """

    response = openai_client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role":"system","content": SYSTEM_PROMPT},
        { "role": "user", "content": user_query }
        ]
    )

    ai_response = response.choices[0].message.content

    print("AI:", ai_response)

    client.add(
        user_id="sanjeev",
        messages=[
            { "role":"user", "content": user_query },
            { "role":"assistant", "content": ai_response }
        ]
    )

    print("Memory has saved..")