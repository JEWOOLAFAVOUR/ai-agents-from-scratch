import os
import json 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-120b"

# transaction dummy data 
TRANSACTIONS = [
    {"month": "September", "category": "food", "amount": 45000},
    {"month": "September", "category": "transport", "amount": 18000},
    {"month": "September", "category": "food", "amount": 62000},
    {"month": "September", "category": "rent", "amount": 80000},
    {"month": "September", "category": "utilities", "amount": 12000},
    {"month": "August", "category": "food", "amount": 40000},
]

# 1. Tool for getting the transactions 
def get_transactions(month: str) -> dict:
    records = [
        transaction 
        for transaction in TRANSACTIONS
        if transaction["month"].lower() == month.lower()
    ]

    return {
        "month": month,
        "transactions": records,
        "count": len(records)
    }

# 2. Calculate spending 

def summarize_spending(month: str) -> dict:
    records = [
        transaction
        for transaction in TRANSACTIONS
        if transaction["month"].lower() == month.lower()
    ]

    totals= {}

    for transaction in records:
        category = transaction["category"]
        totals[category] = (
            totals.get(category, 0) + transaction["amount"]
        )

    if not totals:
        return {
            "month": month,
            "category_totals": {},
            "total_spending": 0,
            "largest_category": None,
            "food_total": 0,
        }

    largest_category = max(totals, key=totals.get)

    return {
        "month": month,
        "category_totals": totals,
        "total_spending": sum(totals.values()),
        "largest_category": largest_category,
        "food_total": totals.get("food", 0),
    }

# describing tool 
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "Retrieve transaction records for a month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "string"}
                },
                "required": ["month"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_spending",
            "description": (
                "Calculate spending totals by category for a month. "
                "Returns total spending, largest category, and food total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "string"}
                },
                "required": ["month"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

# Allow only these two function to execute 
TOOL_FUNCTIONS = {
    "get_transaction": get_transactions,
    "summarize_spending": summarize_spending,
}


def run_agent(question: str, max_steps: int = 6) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analysis agent. "
                "Use tools to retrieve financial data and calculate totals. "
                "Never invent transactions or financial figures. "
                "Use the tool results as evidence. "
                "Amounts are in Nigerian naira. "
                "Give a concise answer based on the available data. "
                "If the requested month has no records, say so."
            ),
        },
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        print(f"\n--- STEP {step + 1}: MODEL ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )

        message = response.choices[0].message

        # Preserve the assistant message, including any tool calls.
        messages.append(message.model_dump(exclude_none=True))

        # No tool calls means the model has produced its final response.
        if not message.tool_calls:
            return message.content or "The model returned no text."

        for tool_call in message.tool_calls:
            name = tool_call.function.name

            print("Requested tool:", name)

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                function = TOOL_FUNCTIONS.get(name)

                if function is None:
                    result = {"error": "Unknown tool."}
                else:
                    result = function(**arguments)

            except (json.JSONDecodeError, TypeError, KeyError) as error:
                result = {"error": f"Invalid tool request: {error}"}

            print("Observed result:", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return "Stopped: maximum agent steps reached."

if __name__ == "__main__":
    answer = run_agent(
        "Analyze my September spending. "
        "What was my biggest expense category, and did I spend "
        "more than 100000 naira on food?"
    )

    print("\n--- FINAL ANSWER ---")
    print(answer)

    answer2 = run_agent("get me th total spending in january return as json", 10)
    print(answer2)