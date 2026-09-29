from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

messages = [
    {
        "role": "user",
        "parts":[
            {"text": "My name is Favour."}
        ]
    },
     {
        "role": "model",
        "parts": [
            {"text": "Nice to meet you, Favour!"}
        ]
    },
    {
        "role": "user",
        "parts": [
            {"text": "I am learning agentic AI."}
        ]
    },
]

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=messages
)

# new_messages = []

# while True:

#     user_input = input("You: ")

#     new_messages.append({
#         "role": "user",
#         "parts": [
#             {"text": user_input}
#         ]
#     })

#     response = client.models.generate_content(
#         model="gemini-3.8-flash",
#         contents=new_messages,
#     )

#     print("Agent: ", response.text)

#     new_messages.append({
#         "role": "model",
#         "parts": [
#             {"text": response.text}
#         ]
#     })


print(response.text)