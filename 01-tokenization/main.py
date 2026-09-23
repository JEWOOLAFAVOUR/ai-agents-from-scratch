import tiktoken

text = "Hello, how are you doing today?"

encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode(text)

print("Original text", text)

print("Tokenized text", tokens)

print("number of tokens", len(tokens))

for token in tokens:
    print(token, "->", repr(encoding.decode([token])))


# import tiktoken


# encoding = tiktoken.get_encoding("cl100k_base")


# examples = [
#     "hello",
#     "Hello, how are you?",
#     "antidisestablishmentarianism",
#     "I am building an AI agent.",
#     "function calculateTransactionBalance() { return balance; }",
#     "₦1,500,000",
# ]


# for text in examples:
#     tokens = encoding.encode(text)

#     print("=" * 60)
#     print("TEXT:")
#     print(text)

#     print("\nTOKENS:")
#     for token in tokens:
#         print(f"{token} -> {repr(encoding.decode([token]))}")

#     print("\nTOKEN COUNT:")
#     print(len(tokens))