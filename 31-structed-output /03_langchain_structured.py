import os 
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq

load_dotenv()

class SupportClassification(BaseModel):
    category: Literal["shipping", "payment", "account", "other"]
    priority: Literal["low", "medium", "high"]
    needs_human: bool

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

structured_model = model.with_structured_output(
    SupportClassification,
)

result = structured_model.invoke(
    "I paid five days ago, my order hasn't shipped, "
    "and support has ignored two messages."
)

if result.needs_human:
    print("Send this case to human agent.")
elif result.category == "shipping":
    print("Start the shipping support workflow")

print(result)
print("Category: ", result.category)
print("Priority: ", result.priority)
print("Needs human: ", result.needs_human)