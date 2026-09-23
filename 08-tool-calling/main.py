from google import genai
import os
from dotenv import load_dotenv

load_dotenv() # this helps read the key from env

client = genai.Client() 

accounts = {
    "123": {
        "owner": "John",
        "balance": 250000
    },
    "456": {
        "owner": "Sarah",
        "balance": 780000
    }
}


get_balance_tool = { 
    "name": "get_balance",
    "description": "Get the current account balance",
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "description": "The account ID",
            }
        },
        "required": ["account_id"]
    }
}

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="what is the balance of account 456",
    config= {
        "tools": [
            {
                "function_declarations": [get_balance_tool]
            }
        ]
    }
)

for part in response.candidates[0].content.parts:

    if part.function_call:
        print("Model want to call", part.function_call.name)

        print("Arguments", part.function_call.args)
        
