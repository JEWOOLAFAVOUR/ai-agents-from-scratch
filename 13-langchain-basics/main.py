from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# defining the state
class State(TypedDict):
    question: str
    needs_tool: bool
    tool_used: bool
    answer: str



# model node

def model(state: State):
    print("Model")


    if "weather" in state["question"].lower() and not state["tool_used"]:
        return {
            "needs_tool": True
        }

    # Otherwise, we have enough information to finish the answer.
    return {
        "needs_tool": False,
        "answer": state["answer"]
    }

# tool node 

def tool(state: State):
    print("Tool")

    # Pretend this came from a real
    # weather API.
    return {
        "tool_used": True,
        "needs_tool": False,
        "answer": "The weather is sunny"
    }

# Router

def route(state: State):
    if state["needs_tool"]:
        return "tool"

    return "end"


# -------------------------
# 5. Build the graph
# -------------------------

graph = StateGraph(State)

# Nodes
graph.add_node("model", model)
graph.add_node("tool", tool)

# START → model
graph.add_edge(START, "model")

# model → tool OR END
graph.add_conditional_edges(
    "model",
    route,
    {
        "tool": "tool",
        "end": END,
    }
)

# tool → model
graph.add_edge("tool", "model")

# compile the graph

app = graph.compile()

# Run the graph with an initial state

result = app.invoke({
    "question": "What is the weather like today?",
    "needs_tool": False,
    "tool_used": False,
    "answer": "",
})



print("result:", result)

# from typing import TypedDict

# from langgraph.graph import StateGraph, START, END

# class State(TypedDict):
#     name: str

# def greet(state: State):
#     return {
#         "name": f"Hello {state['name']}!"
#     }

# graph = StateGraph(State)

# graph.add_node("greet", greet)

# graph.add_edge(START, "greet")
# graph.add_edge("greet", END)

# app = graph.compile()

# result = app.invoke({"name": "Favour"})

# print(result)