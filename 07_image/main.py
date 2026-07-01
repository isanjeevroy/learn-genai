from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI()

response = client.chat.complentions.create(
    model="gpt-4.1",
    messages=[
        {
            "role":"user",
            "content":[
                {
                    "type":"text", "text":" Generate a caption for this image"
                },
                {
                    "type":"image_url",
                    "image_url":"https://res.cloudinary.com/dbab3wklc/image/upload/v1758818141/Profile%20Pictures/black-formal-photo-resized_sg0nzv.png"
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)