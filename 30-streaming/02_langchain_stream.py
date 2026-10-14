import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

messages = [
    SystemMessage(
        content="You are a helpful customer support assistant."
    ),
    HumanMessage(
        content="Explain how an AI agent uses tools in simple English."
    )
]

for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()