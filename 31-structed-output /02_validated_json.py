import os
from typing import Literal


from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

class SupportClassification(BaseModel):
    category: Literal["shipping", "payment", "account", "other"]
    priority: Literal["low", "medium", "high"]
    needs_human: bool

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "Classify the customer's support message. "
                "Choose the category and priority that best fit."
            ),
            # This will show error but if you want it real it'll use the bellow 
            #    "content": """Classify customer support messages.
            #     Return only a JSON object with these fields:
            #     - category: shipping, payment, account, or other
            #     - priority: low, medium, or high
            #     - needs_human: boolean
            #     Do not include markdown or explanations."""
        },
        {
            "role": "user",
            "content": (
                "I paid five days ago. My order hasn't shipped, "
                "and support has ignored two messages."
            )
        },
    ],
    temperature=0
)

content = response.choices[0].message.content

try:
    result = SupportClassification.model_validate_json(content)
    print(result.model_dump())
    print("Catgory ", result.category)
    print("Priority ", result.priority)

except(ValidationError, ValueError, TypeError) as error:
    print("Invalid model output", error)