import os 
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool, tool 

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# LLM

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature = 0
)

# FAKE CUSTOMER DATA 


ORDERS = {
    "ORD-1001": {
        "customer_id": "CUS-001",
        "status": "shipped",
        "item": "MacBook Air",
        "total": 850000,
    },
    "ORD-1002": {
        "customer_id": "CUS-002",
        "status": "delivered",
        "item": "iPhone 16",
        "total": 1200000,
    },
}


CUSTOMERS = {
    "CUS-001": {
        "name": "Favour",
        "email": "favour@example.com",
        "tier": "premium",
    },
    "CUS-002": {
        "name": "Daniel",
        "email": "daniel@example.com",
        "tier": "standard",
    },
}


# Tools 

@tool
def get_order(order_id: str) -> dict:
    """Look up an order using its order ID."""

    order = ORDERS.get(order_id)

    if not order:
        return {"error": f"Order with ID {order_id} not found."}

    return order

@tool
def get_customer(customer_id: str) -> dict:
    """Look up a customer using their customer ID."""

    customer = CUSTOMERS.get(customer_id)

    if not customer:
        return {"error": f"Customer with ID {customer_id} not found."}

    return customer

tools = [get_order, get_customer]

llm_with_tools = llm.bind_tools(tools)

# State

class State(TypedDict):
    messages: list


# assistant node 

def assistance(state: State):
    print("\n--- ASSISTANT ---")

    system_message = SystemMessage(
        content="""
You are a helpful customer support agent.

Your job is to help customers with:
- orders
- delivery status
- customer account information

Rules:
- Use tools when you need information.
- Never invent order information.
- Never invent customer information.
- If a tool returns an error, explain the problem clearly.
- Be concise and friendly.
"""
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": state["messages"] + [response]
    }

#  tool node

tool_node = ToolNode(tools)

# router 
def route(state: State):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"

# build the graph 

graph = StateGraph(State)

graph.add_node("assistant", assistance)
graph.add_node("tools", tool_node)

graph.add_edge(START, "assistant")

graph.add_conditional_edges("assistant", route, {"tools": "tools", "end": END})

app = graph.compile()

# run 

result = app.invoke({
    "messages": [
        HumanMessage(
            content="Where is my order ORD-1001?"
        )
    ]
})


# output 

print("=" * 10)

print("result ", result["messages"][-1].content)
print("result full ", result)
