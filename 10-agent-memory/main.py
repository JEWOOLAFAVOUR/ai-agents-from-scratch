from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import uuid

load_dotenv()

MAX_ITERATION = 10
MAX_RUNTIME_SECONDS = 30

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL = "openrouter/free"


# Memory

messages = []

messages.append({
    "role": "user",
    "content": "My name is Favour"
})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages
)

messages.append({
    "role": "assistant",
    "content": response.choices[0].message.content
})

print("Agentic response:", response.choices[0].message.content)


messages.append({
    "role": "user",
    "content": "What is my name?"
})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages
)

messages.append({
    "role": "assistant",
    "content": response.choices[0].message.content
})

print("Agentic response:", response.choices[0].message.content)


print("\n-- history --")

for message in messages:
    print("role:", message["role"])
    print("content:", message["content"])


# Calculator agent, this is another agent on it's own different from the chat messages and history keep

def calculator(a: float, b: float, operation: str) -> float:

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
        f"Unknown operation {operation}"
    )


calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide"
                    ],
                    "description": "Arithmetic operation"
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

# TOOL EXECUTION

def execute_tool(tool_call):

    if tool_call.function.name != "calculator":
        raise ValueError(
            f"Unknown tool {tool_call.function.name}"
        )

    args = json.loads(
        tool_call.function.arguments
    )

    return calculator(
        args["a"],
        args["b"],
        args["operation"]
    )


# trace function

def log_event(event, **data):
    print({
        "event": event,
        "timestamp": time.time(),
        **data,
    })



# Agent Loop & memory

agent_messages = []


def run_agent(user_message):

    run_id = str(uuid.uuid4())

    log_event(
        "RUN_START",
        run_id=run_id
    )

    start_time = time.monotonic()

    agent_messages.append({
        "role": "user",
        "content": user_message
    })

    for iteration in range(MAX_ITERATION):

        log_event(
            "ITERATION_START",
            run_id=run_id,
            iteration=iteration
        )

        # Check total runtime
        elapsed = time.monotonic() - start_time

        if elapsed > MAX_RUNTIME_SECONDS:
            log_event(
                "RUN_STOP",
                run_id=run_id,
                reason="max_runtime"
            )

            raise TimeoutError(
                "Agent exceeded maximum runtime"
            )

        # CALL MODEL

        log_event(
            "MODEL_CALL",
            run_id=run_id,
            iteration=iteration
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=agent_messages,
            tools=[calculator_tool]
        )

        assistant_message = response.choices[0].message

        # MODEL ANSWERED DIRECTLY FOR TASKS THAT DOESN"T REQUIRE TOOL

        if not assistant_message.tool_calls:

            agent_messages.append({
                "role": "assistant",
                "content": assistant_message.content
            })

            log_event(
                "RUN_STOP",
                run_id=run_id,
                reason="final_answer"
            )

            return assistant_message.content

        # MODEL REQUESTED TOOL

        agent_messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        })

        # EXECUTE TOOLS

        for tool_call in assistant_message.tool_calls:

            log_event(
                "TOOL_CALL",
                run_id=run_id,
                iteration=iteration,
                tool=tool_call.function.name
            )

            tool_start = time.monotonic()

            log_event(
                "TOOL_START",
                run_id=run_id,
                tool=tool_call.function.name
            )

            result = execute_tool(tool_call)

            tool_duration = time.monotonic() - tool_start

            log_event(
                "TOOL_SUCCESS",
                run_id=run_id,
                tool=tool_call.function.name,
                duration_ms=round(tool_duration * 1000, 2)
            )

            agent_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({
                    "result": result
                })
            })

    # LOOP FINISHED

    log_event(
        "RUN_STOP",
        run_id=run_id,
        reason="max_iterations"
    )

    raise RuntimeError(
        "Agent exceeded maximum iterations"
    )


# Running a Test
# -------------------------

answer = run_agent(
    "What is 2000 + 950?"
)

print("Agent:", answer)


answer = run_agent(
    "Now subtract 25000 from that."
)

print("Agent:", answer)

answer = run_agent(
    "What is the capital of Nigeria"
)

print("Agent:", answer)
