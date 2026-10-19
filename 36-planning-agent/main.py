import os 
import json 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"

TRANSACTIONS = [
    {"category": "food", "amount": 107000},
    {"category": "rent", "amount": 80000},
    {"category": "transport", "amount": 18000},
    {"category": "utilities", "amount": 12000},
]

def make_plan(goal: str) -> list:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a task planner. Return ONLY a JSON object "
                    "with a 'tasks' array. Choose tasks only from: "
                    "get_transactions, calculate_totals, write_report. "
                    "Use that order. Each task must have 'name' and "
                    "'description'. Do not include any other task names."
                ),
            },
            {"role": "user", "content": goal}
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    data = json.loads(response.choices[0].message.content or "{}")
    return data["tasks"]

def get_transactions(state: dict) -> dict:
    return {"transactions": TRANSACTIONS}


def calculate_totals(state: dict) -> dict:
    transactions = state["transactions"]
    totals = {}

    for item in transactions:
        category = item["category"]
        totals[category] = totals.get(category, 0) + item["amount"]

    return {
        "category_totals": totals,
        "total_spending": sum(totals.values()),
    }



def write_report(state: dict) -> dict:
    facts = {
        "category_totals": state["category_totals"],
        "total_spending": state["total_spending"],
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a concise report using only the supplied facts. "
                    "Do not invent financial figures."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(facts),
            },
        ],
        temperature=0,
    )

    return {"report": response.choices[0].message.content or ""}



HANDLERS = {
    "get_transactions": get_transactions,
    "calculate_totals": calculate_totals,
    "write_report": write_report,
}


def run_planner(goal: str) -> dict:
    plan = make_plan(goal)

    # Shared state stores inputs, outputs, and task status.
    state = {"goal": goal, "completed_tasks": []}

    print("\nPLAN:")
    print(json.dumps(plan, indent=2))

    for task in plan:
        name = task.get("name")

        # Never execute arbitrary function names from the model.
        handler = HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"Unsupported task: {name}")

        print(f"\nExecuting: {name}")
        result = handler(state)

        # Store the output for later tasks.
        state.update(result)
        state["completed_tasks"].append(name)

    return state


if __name__ == "__main__":
    result = run_planner(
        "Analyze the transactions, calculate category totals, "
        "and write a short financial report."
    )

    print("\nFINAL REPORT:")
    print(result.get("report", "No report was generated."))
    print("\nCOMPLETED:")
    print(result["completed_tasks"])
