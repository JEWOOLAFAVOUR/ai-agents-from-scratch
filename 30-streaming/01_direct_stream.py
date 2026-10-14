import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

stream = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful agent.",
        },{
            "role": "user",
            "content": "Explain how an ai agent uses tool in simple english."
        }
    ],
    temperature=0,
    stream=True
)

for chunk in stream:
    text = chunk.choices[0].delta.content

    if text:
        print(text, end="", flush=True)

print("nothing")