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

FINANCIAL_FACTS = """
September financial data:
- Food: NGN 107,000
- Rent: NGN 80,000
- Transport: NGN 18,000
- Utilities: NGN 12,000
- Total spending: NGN 217,000

These figures are already calculated and verified.
Do not change the figures or invent additional transactions.
We do not know the person's income, savings target,
household size, or whether food spending is excessive
for their circumstances.
"""

def ask_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful financial analysis assistant. "
                    "Use only the supplied facts. "
                    "Separate facts from assumptions. "
                    "Do not claim that an expense is excessive "
                    "without enough context."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def explore_candidates() -> str:
    print("\n=== 1. TREE OF THOUGHTS: EXPLORE ===")

    return ask_model(
        f"""
{FINANCIAL_FACTS}

We need to help the person understand their spending
and identify practical ways to save money.

Generate exactly three DISTINCT analysis approaches.

For each approach, state:
1. Its main question.
2. What evidence it would use.
3. Its potential benefit.
4. Its main limitation.

Consider different approaches, such as category analysis,
budget planning, and identifying missing information.

Do not invent personal circumstances.
Do not write the final financial report yet.
"""
    )

def evaluate_candidates(candidates: str) -> str:
    print("\n=== 2. TREE OF THOUGHTS: EVALUATE ===")

    return ask_model(
        f"""
{FINANCIAL_FACTS}

Evaluate these three candidate approaches:

{candidates}

Score each approach from 1 to 5 on:
- Evidence accuracy
- Practical usefulness
- Awareness of missing information
- Risk of unsupported assumptions

Explain the scores briefly.
Then select the best approach for this situation.

Return the selected approach and explain why.
Do not alter the financial figures.
"""
    )


def create_draft(selected_approach: str) -> str:
    print("\n=== 3. CREATE FIRST DRAFT ===")

    return ask_model(
        f"""
{FINANCIAL_FACTS}

Use this selected approach:

{selected_approach}

Write a short financial analysis with:
- The most important spending facts.
- One or two practical next steps.
- What additional information would improve the advice.

Be specific but do not assume the person has an income
or that any category is automatically too expensive.
"""
    )


def critique_draft(draft: str) -> str:
    print("\n=== 4. SELF-REFINE: CRITIQUE ===")

    return ask_model(
        f"""
{FINANCIAL_FACTS}

Critique this report:

{draft}

Check it against these criteria:
1. Are the figures consistent with the supplied facts?
2. Are facts separated from assumptions?
3. Are the recommendations actionable?
4. Does it acknowledge important missing information?
5. Is anything misleading, vague, or unnecessarily repetitive?

List specific problems and improvements.
If a criterion is already satisfied, say so.
Do not rewrite the entire report yet.
"""
    )


def revise_draft(draft: str, critique: str) -> str:
    print("\n=== 5. SELF-REFINE: REVISE ===")

    return ask_model(
        f"""
{FINANCIAL_FACTS}

Here is the first draft:

{draft}

Here is the critique:

{critique}

Produce an improved final report.
Correct genuine problems, preserve accurate information,
and avoid adding unsupported claims.

Keep it concise and practical.
Do not invent facts to satisfy the critique.
"""
    )


def main():
    candidates = explore_candidates()
    print(candidates)

    evaluation = evaluate_candidates(candidates)
    print(evaluation)

    draft = create_draft(evaluation)
    print("\nFIRST DRAFT:\n", draft)

    critique = critique_draft(draft)
    print("\nCRITIQUE:\n", critique)

    final_report = revise_draft(draft, critique)
    print("\n=== FINAL REPORT ===\n")
    print(final_report)


if __name__ == "__main__":
    main()