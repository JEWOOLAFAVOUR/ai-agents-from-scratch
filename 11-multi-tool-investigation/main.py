from openai import OpenAI
from dotenv import load_dotenv
import json
import uuid
import time
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

MODEL = "openai/gpt-oss-20b"

MAX_ITERATIONS=10
MAX_TOOL_CALLS=15
MAX_RUNTIME_SECONDS=30

LOGS = [
    {
        "timestamp": "10:01:12",
        "service": "transaction-service",
        "message": "GET /transactions"
    },
    {
        "timestamp": "10:01:14",
        "service": "transaction-service",
        "message": "Database connection timeout after 2 seconds"
    },
    {
        "timestamp": "10:01:14",
        "service": "transaction-service",
        "message": "GET /transactions returned HTTP 500"
    },
]

CODE = {
    "transactions": """
async function getTransactions() {
    const db = await database.connect({
        timeout: 2000
    });

    return db.query(
        "SELECT * FROM transactions"
    );
}
""",

    "database": """
Database connections are created with a
2-second timeout.
"""
}

DOCS = {
    "transactions": """
Transaction Service Documentation

The transaction service depends on the primary database.

Database connection timeout should be at least
10 seconds because large transaction queries may
take several seconds to complete.

If the database connection exceeds the timeout,
the service should retry before returning an error.
"""
}

# This part are the tools 

def search_logs(query: str):
    query = query.lower()

    results = []

    for log in LOGS:
        text = log["message"].lower()

        if any(word in text for word in query.split()):
            results.append(log)

    if not results:
        return "No matching logs found."

    return results


def search_code(query: str):
    query = query.lower()

    results = []

    for name, code in CODE.items():
        if (
            query in name.lower()
            or any(word in code.lower() for word in query.split())
        ):
            results.append({
                "file": name,
                "code": code
            })

    if not results:
        return "No matching code found."

    return results


def search_docs(query: str):
    query = query.lower()

    results = []

    for name, document in DOCS.items():
        if (
            query in name.lower()
            or any(word in document.lower() for word in query.split())
        ):
            results.append({
                "document": name,
                "content": document
            })

    if not results:
        return "No matching documentation found."

    return results


def calculator(a: float, b: float, operation: str):
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")

        return a / b

    raise ValueError(
        f"Unknown operation: {operation}"
    )


# tool schema so the agents can know how the tools look like 

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": (
                "Search application logs for evidence about "
                "errors, requests, failures, or system behavior."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the logs."
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search the codebase to understand how a feature "
                "or endpoint is implemented."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What part of the code to investigate."
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search technical documentation to understand "
                "the expected or documented behavior of the system."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What requirement to investigate."
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ]
                    }
                },
                "required": [
                    "a",
                    "b",
                    "operation"
                ]
            }
        }
    }
]


# tool execution - consolidate all tools into a place executor

def execute_tool(tool_call):
    tool_name = tool_call.function.name

    args = json.loads(tool_call.function.arguments)

    if tool_name == "search_logs":
        return search_logs(args["query"])

    if tool_name == "search_code":
        return search_code(args["query"])

    if tool_name == "search_docs":
        return search_docs(
            args["query"]
        )

    if tool_name == "calculator":
        return calculator(
            args["a"],
            args["b"],
            args["operation"]
        )

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# Tracing for logging events

def log_event(event, **data):
    print({
        "event":event,
        "timestamp": time.time(),
        **data,
    })

# Agents Function 

def run_agent(user_question: str):

    run_id = str(uuid.uuid4())

    state = {
        "question": user_question,
        "evidence": [],
        "tools_used": [],
        "iteration": 0,

    }

    messages = [
        {
            "role": "system",
            "content": """
                You are a software investigation agent.

                Your job is to investigate technical problems using
                the available tools.

                Do not jump to conclusions from one piece of evidence.

                Gather evidence from logs, code, and documentation
                when useful.

                Your final answer must contain:

                1. Finding
                2. Evidence
                3. Recommended next step

                Only make claims supported by the evidence you collected.

                If evidence is insufficient, say so instead of inventing
                an explanation.
                """
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    start_time = time.monotonic()

    tool_calls_count = 0

    log_event("run_start", run_id=run_id)

    # agent loop 
    for iteration in range(MAX_ITERATIONS):

        state["iterations"] = iteration + 1

        elapsed = time.monotonic() - start_time

        if elapsed > MAX_RUNTIME_SECONDS:

            log_event(
                "run_stop",
                run_id=run_id,
                reason="runtime_limit"
            )

            return (
                "Investigation stopped because "
                "the maximum runtime was exceeded."
            )

        log_event(
            "iteration_start",
            run_id=run_id,
            iteration=iteration + 1
        )

        model_start = time.monotonic()

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        model_latency = (
            time.monotonic() - model_start
        )

        log_event(
            "model_call",
            run_id=run_id,
            iteration=iteration + 1,
            latency=round(model_latency, 3)
        )

        message = response.choices[0].message

        # The assistant's decision becomes part of history.
        messages.append(message)

        # Final answer 
        if not message.tool_calls:

            log_event(
                "run_stop",
                run_id=run_id,
                reason="final_answer",
                iterations=iteration + 1
            )

            return message.content

        # tool calls 
        for tool_call in message.tool_calls:

            tool_calls_count += 1

            if tool_calls_count > MAX_TOOL_CALLS:

                log_event(
                    "run_stop",
                    run_id=run_id,
                    reason="tool_call_limit"
                )

                return (
                    "Investigation stopped because "
                    "the maximum number of tool calls was exceeded."
                )

            tool_name = tool_call.function.name

            state["tools_used"].append(
                tool_name
            )

            log_event(
                "tool_call",
                run_id=run_id,
                tool=tool_name
            )

            # execute tools 
            tool_start = time.monotonic()

            try:

                result = execute_tool(
                    tool_call
                )

                tool_status = "success"

            except Exception as error:

                result = f"Tool failed: {error}"

                tool_status = "error"

            tool_latency = (
                time.monotonic() - tool_start
            )

            # Store evidence in state
            state["evidence"].append({
                "tool": tool_name,
                "result": result
            })

            log_event(
                "tool_result",
                run_id=run_id,
                tool=tool_name,
                status=tool_status,
                latency=round(tool_latency, 3)
            )

            # send result back to model

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

            # Iteration limit 
    log_event(
        "run_stop",
        run_id=run_id,
        reason="iteration_limit"
    )

    return (
        "Investigation stopped because "
        "the maximum number of iterations was reached."
    )

# Run a test 
# question = """
# Why is the /transactions endpoint returning HTTP 500 errors?

# Investigate the problem carefully.

# Check the logs, implementation, and documented requirements
# before giving your conclusion.
# """

question = """
Search the codebase and explain how the /transactions endpoint is implemented.
"""

answer = run_agent(question)

print("\n")
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(answer)