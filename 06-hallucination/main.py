documents = [
    {
        "title": "Refund Policy",
        "content": "Customers can request a refund within 30 days"
    },
    {
        "title": "Shipping Policy",
        "content": "Standard shipping takes 3 to 5 business days."
    },
    {
        "title": "Support Policy",
        "content": "Support is available Monday to Friday."
    }
]

question = "How long do customers have a request to refund?"

print("Question: ", question)

print("\nAvailable evidence:")

for document in documents:
    print(f"\n[{document['title']}]")
    print(document["content"])
