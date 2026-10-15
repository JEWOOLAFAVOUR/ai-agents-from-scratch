import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
# from lang

load_dotenv()

def create_model(provider: str):
    if provider == "groq":
        return ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ["OPENAI_MODEL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def ask_model(model, question: str) -> str:
    messages = [
        SystemMessage(
            content="You are a concise, helpful assistant."
        ),
        HumanMessage(content=question),
    ]

    response = model.invoke(messages)
    return response.content


provider = os.getenv("LLM_PROVIDER", "groq")
model = create_model(provider)

answer = ask_model(
    model,
    "Explain the difference between an LLM and an AI agent.",
)

print(answer)