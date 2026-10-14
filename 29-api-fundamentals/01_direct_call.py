import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful customer support assistant."
        },{
            "role": "user",
            "content": "Explain what an ordering tracking number is in one sentence",
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)