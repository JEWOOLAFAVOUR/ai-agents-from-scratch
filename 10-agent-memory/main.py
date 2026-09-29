from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

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


# Agent Loop

agent_messages = []


def run_agent(user_message):

    agent_messages.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=agent_messages,
        tools=[calculator_tool]
    )

    assistant_message = response.choices[0].message

    # Model answered normally without using a tool
    if not assistant_message.tool_calls:

        agent_messages.append({
            "role": "assistant",
            "content": assistant_message.content
        })

        return assistant_message.content

    # Model requested a tool
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

    # Execute every tool the model requested
    for tool_call in assistant_message.tool_calls:

        result = execute_tool(tool_call)

        agent_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({
                "result": result
            })
        })

    # Send the tool result back to the model
    response = client.chat.completions.create(
        model=MODEL,
        messages=agent_messages,
        tools=[calculator_tool]
    )

    final_message = response.choices[0].message

    agent_messages.append({
        "role": "assistant",
        "content": final_message.content
    })

    return final_message.content



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
