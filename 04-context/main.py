import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

messages = [
    "You are a helpful accounting assistant.",
    "Find all overdue invoices for ACME.",
    "Invoice 001: Customer John, Amount ₦200,000, Status Paid.",
    "Invoice 002: Customer Sarah, Amount ₦500,000, Status Overdue.",
    "Invoice 003: Customer Mike, Amount ₦150,000, Status Paid.",
]

running_context = []

total_processed = 0

for i, message in enumerate(messages, start=1):
    running_context.append(message)

    context_text = "\n".join(running_context)

    context_tokens = len(
        encoding.encode(context_text)
    )

    total_processed += context_tokens

    print(f"TURN {i}")
    print(f"Current context tokens: {context_tokens}")
    print(f"Cumulative processed tokens: {total_processed}")
    print("-" * 50)