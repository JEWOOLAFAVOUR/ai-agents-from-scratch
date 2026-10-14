import json
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
            "content": """"
            Classify customer support messages.
Return only a JSON object with these fields:
- category: shipping, payment, account, or other
- priority: low, medium, or high
- needs_human: boolean
Do not include markdown or explanations.
            """
        },
        {
            "role": "user",
            "content": "I paid for my order five days ago and it still hasn't shipped."
        }
    ],
    temperature = 0
)

content = response.choices[0].message.content

print("Raw content:", content)

try:
    result = json.loads(content)
    print("category", result["category"])
    print("priority", result["priority"])
    print("needs_human", result["needs_human"])

except (json.JSONDecodeError, TypeError, KeyError) as error:
    print("Could not parse the response as JSON:", error)