import os 
import time 
import random

from dotenv import load_dotenv 
from openai import (OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=30.0,
    max_retries=0
)

RETRYABLE_ERROR = (
    APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
)

def ask_model(prompt: str, max_attempts: int = 4) -> str:
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            return response.choices[0].message.content or ""

        except RETRYABLE_ERROR:
            if attempt == max_attempts - 1:
                raise

            # 1, 2, 4, 8 seconds, with a little randomness.
            delay = min(2 ** attempt, 8) + random.uniform(0, 0.5)

            print(
                f"Temporary failure. Attempt {attempt + 1} "
                f"of {max_attempts}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


answer = ask_model("Explain the difference between ai agents and llm")

print(answer)