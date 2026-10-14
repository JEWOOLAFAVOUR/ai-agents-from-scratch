import os 

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

response = model.invoke([
    SystemMessage(
        content="You are a helpful customer support assistant."
    ),
    HumanMessage(
        content="Explain what an order tracking number means in one sentence."
    )
])

print(response.content)
print("Usage: ", response.usage_metadata)