from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

def calculator(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")

        return a / b 

    raise ValueError(f"Unknown operation: {operation}")

calculator_tool = {
    "name": "calculator",
    "description": "Perform basic arithmetic operations",
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
                "description": "The arithmetic operation."
            }
        },
        "required": ["a", "b", "operation"]
    }
}

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="What is 2400 + 7500?",
    config={
        "tools": [
            {
                "function_declarations": [calculator_tool]
            }
        ]
    }
)

for part in response.candidates[0].content.parts:
    
    if not part.function_call:
        continue

    function_call = part.function_call

    if function_call.name == "calculator":
        args = function_call.args

        result = calculator(
            args["a"],
            args["b"],
            args["operation"]
        )

    print("Tool Result: ", result)

def validate_calculator_args(args):

    if "a" not in args:
        raise ValueError("Missing argument: a")

    if "b" not in args:
        raise ValueError("Missing argument: b")

    if "operation" not in args:
        raise ValueError("Missing argument: operation")

    if not isinstance(args["a"], (int, float)):
        raise ValueError("a must be a number")

    if not isinstance(args["b"], (int, float)):
        raise ValueError("b must be a number")

    allowed_operations = {
        "add",
        "subtract",
        "multiply",
        "divide"
    }

    if args["operation"] not in allowed_operations:
        raise ValueError("Invalid operation")

def execute_tool(function_call): 

    tool_name = function_call.name
    args = function_call.args

    if tool_name != "calculator":
        raise ValueError(
            f"Unknown tool: ", tool_name
        )
    
    validate_calculator_args(args)

    return calculator(
        args["a"],
        args["b"],
        args["operation"]
    )

result = execute_tool(function_call)

print("TOOL RESULT:")
print(result)