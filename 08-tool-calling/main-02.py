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
    contents="What is 2500 + 7500?",
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