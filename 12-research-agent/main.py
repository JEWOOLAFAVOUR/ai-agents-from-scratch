import os
import json
import time
import uuid

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

MODEL = "openai/gpt-oss-20b"

MAX_ITERATIONS = 9
MAX_TOOL_CALLS = 12
MAX_RUNTIME_SECONDS = 300

MAX_SEARCH_RESULTS = 4
MAX_PAGE_CHARS = 4000

MAX_SEARCH_CALLS = 3
MAX_PAGE_READS = 4


# research state
state = {
    "question": "",
    "sources_found": [],
    "sources_read": [],
    "evidence": [],
    "tools_used": [],
    "iterations": 0,
    "search_calls": 0,
    "page_reads": 0
}


# web search function
def search_web(query: str):
    """
    Search the real web and return a small set of results.
    """

    print(f"\n[WEB SEARCH] {query}")

    results = []

    try:
        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=MAX_SEARCH_RESULTS
            )

            for result in search_results:

                results.append({
                    "title": result.get("title"),
                    "url": result.get("href"),
                    "snippet": result.get("body")
                })

    except Exception as error:

        return {
            "error": f"Web search failed: {error}"
        }

    return results


# read web page
def read_web_page(url: str):
    """
    Download a webpage and extract readable text.
    """

    print(f"\n[READ PAGE] {url}")

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/154 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside"
        ]):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        text = text[:MAX_PAGE_CHARS]

        return {
            "url": url,
            "title": soup.title.get_text(strip=True)
            if soup.title
            else url,
            "content": text
        }

    except Exception as error:

        return {
            "url": url,
            "error": f"Failed to read page: {error}"
        }


# calculator function
def calculator(
    a: float,
    b: float,
    operation: str
):
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":

        if b == 0:
            raise ValueError(
                "Cannot divide by zero"
            )

        return a / b

    raise ValueError(
        f"Unknown operation: {operation}"
    )


# tool definition or schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": """
Search the real web for information relevant to the
research question.

Use this when you need to discover sources or find
additional evidence.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused web search query."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "read_web_page",
        "description": """
Read a specific webpage from a URL returned by
search_web.

IMPORTANT:
This tool accepts ONLY a URL.

Always call this tool with exactly this argument format:

{
    "url": "https://example.com/page"
}

Do NOT use cursor.
Do NOT use loc.
Do NOT use offset.
Do NOT use page number.
Do NOT use search result index.

If you want to read a search result, first take the
actual URL from the search_web result and pass that URL
to this tool.
""",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The complete URL of the webpage "
                        "to read, copied from a search_web "
                        "result."
                    )
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }
    }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform basic arithmetic calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ]
                    }
                },
                "required": ["a", "b", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_research",
            "description": """
Stop researching and move to the final answer.

Use this when you have enough evidence to answer the
research question.

Do not call this simply because one source was found.
Use it when the important claims have enough evidence,
important contradictions have been checked, and
additional searching is unlikely to materially improve
the answer.
""",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# tracing function for logs
def log_event(event, **data):
    print({
        "event": event,
        "timestamp": time.time(),
        **data
    })


# final answer generation
def finalize_research(messages, run_id):

    log_event(
        "research_finalizing",
        run_id=run_id
    )

    final_start = time.monotonic()

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[]
    )

    final_latency = time.monotonic() - final_start

    log_event(
        "final_model_call",
        run_id=run_id,
        latency=round(final_latency, 3)
    )

    return final_response.choices[0].message.content


# research agent
def run_research_agent(research_question: str):

    run_id = str(uuid.uuid4())

    state["question"] = research_question
    state["sources_found"] = []
    state["sources_read"] = []
    state["evidence"] = []
    state["tools_used"] = []
    state["iterations"] = 0
    state["search_calls"] = 0
    state["page_reads"] = 0

    messages = [
        {
            "role": "system",
            "content": """
You are a web research agent.

Your job is to investigate a research question using
real web sources before producing an answer.

Research process:

1. Understand the question.
2. Perform 1-2 focused searches.
3. Select the strongest and most relevant sources.
4. Read 2-4 useful sources.
5. Compare the evidence.
6. Check for important gaps or contradictions.
7. If necessary, perform one additional focused search.
8. Once there is enough evidence, call finish_research.

Important rules:

- Do not invent facts.
- Do not treat search-result snippets as strong evidence
  when the actual source can be read.
- Prefer primary and authoritative sources when available.
- Distinguish facts from opinions.
- If credible sources disagree, explain the disagreement.
- If evidence is insufficient, explicitly say so.
- Do not pretend certainty when the evidence does not
  support it.
- Do not search simply because more sources exist.
- Use the evidence already collected when the research
  budget is exhausted.

Research lifecycle:

RESEARCHING
    ↓
ENOUGH EVIDENCE
    ↓
finish_research
    ↓
FINALIZING
    ↓
FINAL ANSWER

You must eventually call finish_research when you have
enough evidence.

Do not continue researching indefinitely.

A reasonable research run usually consists of:
- 1-2 searches
- 2-4 page reads
- then finish_research

If the evidence is sufficient, call finish_research
immediately.

If a tool reports that its budget is exhausted, do not
try to call that tool again. Use the evidence already
collected and call finish_research.

If the system tells you that research time is almost
exhausted, do not perform additional tool calls.
Call finish_research immediately.

Your final response must contain:

Research conclusion

Evidence
- Important evidence discovered during research.

Sources
- Important sources used during the research.

Limitations
- Important things the available evidence does not establish.

Do not reveal private chain-of-thought.
Provide concise evidence and conclusions instead.

The application will separately generate the canonical
source list from the webpages actually read.
"""
        },
        {
            "role": "user",
            "content": research_question
        }
    ]

    start_time = time.monotonic()

    tool_calls_count = 0

    log_event(
        "research_start",
        run_id=run_id
    )

    for iteration in range(MAX_ITERATIONS):

        state["iterations"] = iteration + 1

        elapsed = time.monotonic() - start_time

        # hard runtime limit
        if elapsed > MAX_RUNTIME_SECONDS:

            log_event(
                "research_stop",
                run_id=run_id,
                reason="runtime_limit"
            )

            # IMPORTANT:
            # Do not throw away the evidence collected.
            # Try to finalize using what we already have.
            return finalize_research(
                messages,
                run_id
            )

        log_event(
            "iteration_start",
            run_id=run_id,
            iteration=iteration + 1
        )

        # warn the model before the hard runtime limit
        remaining_time = (
            MAX_RUNTIME_SECONDS - elapsed
        )

        if remaining_time <= 60:

            messages.append({
                "role": "system",
                "content": """
Research time is almost exhausted.

Do not perform additional searches or page reads.

Use the evidence already collected.

Call finish_research now.
"""
            })

            log_event(
                "research_warning",
                run_id=run_id,
                reason="runtime_almost_exhausted",
                remaining_seconds=round(
                    remaining_time,
                    2
                )
            )

        model_start = time.monotonic()

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        model_latency = time.monotonic() - model_start

        log_event(
            "model_call",
            run_id=run_id,
            iteration=iteration + 1,
            latency=round(model_latency, 3)
        )

        message = response.choices[0].message

        messages.append(message)

        # model directly produced an answer
        if not message.tool_calls:

            log_event(
                "research_stop",
                run_id=run_id,
                reason="final_answer"
            )

            return message.content

        for tool_call in message.tool_calls:

            tool_calls_count += 1

            if tool_calls_count > MAX_TOOL_CALLS:

                log_event(
                    "research_stop",
                    run_id=run_id,
                    reason="tool_call_limit"
                )

                # Instead of returning a generic stop message,
                # finalize using the evidence already collected.
                return finalize_research(
                    messages,
                    run_id
                )

            tool_name = tool_call.function.name

            state["tools_used"].append(
                tool_name
            )

            log_event(
                "tool_call",
                run_id=run_id,
                tool=tool_name
            )

            # -------------------------------------------------
            # FINISH RESEARCH
            # -------------------------------------------------

            if tool_name == "finish_research":

                log_event(
                    "research_stop",
                    run_id=run_id,
                    reason="finish_research"
                )

                return finalize_research(
                    messages,
                    run_id
                )

            # -------------------------------------------------
            # PARSE ARGUMENTS
            # -------------------------------------------------

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except Exception as error:

                result = {
                    "error": (
                        f"Invalid tool arguments: {error}"
                    )
                }

                log_event(
                    "tool_result",
                    run_id=run_id,
                    tool=tool_name,
                    status="error",
                    latency=0
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

                continue

            # -------------------------------------------------
            # TOOL EXECUTION
            # -------------------------------------------------

            tool_start = time.monotonic()

            try:

                # ---------------------------------------------
                # SEARCH WEB
                # ---------------------------------------------

                if tool_name == "search_web":

                    state["search_calls"] += 1

                    if (
                        state["search_calls"]
                        > MAX_SEARCH_CALLS
                    ):

                        result = {
                            "error": (
                                "Web search budget exhausted."
                            ),
                            "instruction": (
                                "Do not search again. "
                                "Use the evidence already "
                                "collected and call "
                                "finish_research."
                            )
                        }

                    else:

                        result = search_web(
                            arguments["query"]
                        )

                        if isinstance(result, list):

                            for source in result:

                                state[
                                    "sources_found"
                                ].append(source)

                # ---------------------------------------------
                # READ WEB PAGE
                # ---------------------------------------------

                elif tool_name == "read_web_page":

                    state["page_reads"] += 1

                    if (
                        state["page_reads"]
                        > MAX_PAGE_READS
                    ):

                        result = {
                            "error": (
                                "Page-read budget exhausted."
                            ),
                            "instruction": (
                                "Do not read another page. "
                                "Use the evidence already "
                                "collected and call "
                                "finish_research."
                            )
                        }

                    else:

                        result = read_web_page(
                            arguments["url"]
                        )

                        if (
                            isinstance(result, dict)
                            and "content" in result
                        ):

                            state[
                                "sources_read"
                            ].append(result)

                            state[
                                "evidence"
                            ].append({
                                "url": result.get("url"),
                                "title": result.get("title"),
                                "evidence": result.get(
                                    "content"
                                )
                            })

                # ---------------------------------------------
                # CALCULATOR
                # ---------------------------------------------

                elif tool_name == "calculator":

                    result = calculator(
                        arguments["a"],
                        arguments["b"],
                        arguments["operation"]
                    )

                # ---------------------------------------------
                # UNKNOWN TOOL
                # ---------------------------------------------

                else:

                    raise ValueError(
                        f"Unknown tool: {tool_name}"
                    )

                status = "success"

            except Exception as error:

                result = {
                    "error": f"Tool failed: {error}"
                }

                status = "error"

            tool_latency = (
                time.monotonic()
                - tool_start
            )

            log_event(
                "tool_result",
                run_id=run_id,
                tool=tool_name,
                status=status,
                latency=round(
                    tool_latency,
                    3
                )
            )

            # -------------------------------------------------
            # RETURN TOOL RESULT TO MODEL
            # -------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result,
                    default=str
                )
            })

    # ---------------------------------------------------------
    # ITERATION LIMIT
    # ---------------------------------------------------------

    log_event(
        "research_stop",
        run_id=run_id,
        reason="iteration_limit"
    )

    # Do not throw away evidence just because the
    # iteration limit was reached.
    return finalize_research(
        messages,
        run_id
    )


# -------------------------------------------------------------
# TEST AGENT
# -------------------------------------------------------------

question = """
Research whether PostgreSQL is suitable for
high-volume financial transaction systems.

I want evidence from authoritative or technically
credible sources.

Compare the evidence and tell me:

1. What PostgreSQL is good at.
2. Where it may have limitations.
3. Whether it is commonly suitable for financial
   transaction workloads.
4. What the available evidence does NOT establish.
"""

answer = run_research_agent(question)

print("\n")
print("=" * 70)
print("FINAL RESEARCH ANSWER")
print("=" * 70)

print(answer)


# -------------------------------------------------------------
# CONSOLIDATED SOURCES
# -------------------------------------------------------------

def get_consolidated_sources():

    sources = []
    seen = set()

    for source in state["sources_read"]:

        url = source.get("url")

        if not url:
            continue

        if url in seen:
            continue

        seen.add(url)

        sources.append({
            "title": source.get("title"),
            "url": url
        })

    return sources


sources = get_consolidated_sources()

print("\n")
print("=" * 70)
print("SOURCES USED")
print("=" * 70)

for index, source in enumerate(
    sources,
    start=1
):

    print(
        f"{index}. {source['title']}\n"
        f"   {source['url']}\n"
    )


# -------------------------------------------------------------
# RESEARCH STATE
# -------------------------------------------------------------

print("\n")
print("=" * 70)
print("RESEARCH STATE")
print("=" * 70)

print(
    json.dumps(
        {
            "iterations": state["iterations"],
            "tools_used": state["tools_used"],
            "search_calls": state["search_calls"],
            "page_reads": state["page_reads"],
            "sources_found": len(
                state["sources_found"]
            ),
            "sources_read": len(
                state["sources_read"]
            ),
            "evidence_items": len(
                state["evidence"]
            )
        },
        indent=2
    )
)




# import os 
# import json 
# import time
# import uuid 

# import requests
# from bs4 import BeautifulSoup
# from ddgs import DDGS 
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ["GROQ_API_KEY"]
# )

# MODEL = "openai/gpt-oss-120b"

# MAX_ITERATIONS = 9
# MAX_TOOL_CALLS = 15
# MAX_RUNTIME_SECONDS = 120

# MAX_SEARCH_RESULTS = 5
# MAX_PAGE_CHARS = 8000

# MAX_SEARCH_CALLS = 4
# MAX_PAGE_READS = 4


# # research state 
# state = {
#     "question": "",
#     "sources_found": [],
#     "sources_read": [],
#     "evidence": [],
#     "tools_used": [],
#     "iterations": 0,
#     "searched_queries": []
# }

# # web search function 
# def search_web(query: str):
#     """
#     Search the real web and return a small set of results.
#     """

#     print(f"\n[WEB SEARCH] {query}")

#     results = []

#     try:
#         with DDGS() as ddgs:

#             search_results = ddgs.text(
#                 query,
#                 max_results=MAX_SEARCH_RESULTS
#             )

#             for result in search_results:

#                 results.append({
#                     "title": result.get("title"),
#                     "url": result.get("href"),
#                     "snippet": result.get("body")
#                 })

#     except Exception as error:

#         return {
#             "error": f"Web search failed: {error}"
#         }

#     return results

# # read web page

# def read_web_page(url: str):
#     """
#     Download a webpage and extract readable text.
#     """

#     print(f"\n[READ PAGE] {url}")

#     try:

#         response = requests.get(
#             url,
#             timeout=15,
#             headers={
#                 "User-Agent": (
#                     "Mozilla/5.0 "
#                     "(Macintosh; Intel Mac OS X 10_15_7) "
#                     "AppleWebKit/537.36 "
#                     "(KHTML, like Gecko) "
#                     "Chrome/154 Safari/537.36"
#                 )
#             }
#         )

#         response.raise_for_status()

#         soup = BeautifulSoup(
#             response.text,
#             "html.parser"
#         )

#         # Remove things that are usually not useful
#         # for research.
#         for element in soup([
#             "script",
#             "style",
#             "nav",
#             "footer",
#             "header",
#             "aside"
#         ]):
#             element.decompose()

#         text = soup.get_text(
#             separator=" ",
#             strip=True
#         )

#         # Prevent enormous pages from consuming
#         # the entire context window.
#         text = text[:MAX_PAGE_CHARS]

#         return {
#             "url": url,
#             "title": soup.title.get_text(strip=True)
#             if soup.title
#             else url,
#             "content": text
#         }

#     except Exception as error:

#         return {
#             "url": url,
#             "error": f"Failed to read page: {error}"
#         }

# # calculator function

# def calculator(
#     a: float,
#     b: float,
#     operation: str
# ):
#     if operation == "add":
#         return a + b

#     if operation == "subtract":
#         return a - b

#     if operation == "multiply":
#         return a * b

#     if operation == "divide":

#         if b == 0:
#             raise ValueError(
#                 "Cannot divide by zero"
#             )

#         return a / b

#     raise ValueError(
#         f"Unknown operation: {operation}"
#     )

# # tool definition or schema 
# tools = [

#     {
#         "type": "function",
#         "function": {

#             "name": "search_web",

#             "description": """
# Search the real web for information relevant to the
# research question.

# Use this when you need to discover sources or find
# additional evidence.
# """,

#             "parameters": {
#                 "type": "object",

#                 "properties": {

#                     "query": {
#                         "type": "string",
#                         "description": (
#                             "A focused web search query."
#                         )
#                     }

#                 },

#                 "required": ["query"]
#             }
#         }
#     },


#     {
#         "type": "function",
#         "function": {

#             "name": "read_web_page",

#             "description": """
# Read a specific webpage from a URL returned by
# search_web.

# Use this when a search result looks relevant and
# you need more detailed evidence from the source.
# """,

#             "parameters": {
#                 "type": "object",

#                 "properties": {

#                     "url": {
#                         "type": "string",
#                         "description": (
#                             "The URL of the webpage to read."
#                         )
#                     }

#                 },

#                 "required": ["url"]
#             }
#         }
#     },


#     {
#         "type": "function",
#         "function": {

#             "name": "calculator",

#             "description": (
#                 "Perform basic arithmetic calculations."
#             ),

#             "parameters": {

#                 "type": "object",

#                 "properties": {

#                     "a": {
#                         "type": "number"
#                     },

#                     "b": {
#                         "type": "number"
#                     },

#                     "operation": {
#                         "type": "string",
#                         "enum": [
#                             "add",
#                             "subtract",
#                             "multiply",
#                             "divide"
#                         ]
#                     }
#                 },

#                 "required": [
#                     "a",
#                     "b",
#                     "operation"
#                 ]
#             }
#         }
#     }
# ]


# # tool execution 
# def execute_tool(tool_call):

#     tool_name = tool_call.function.name

#     arguments = json.loads(
#         tool_call.function.arguments
#     )

#     if tool_name == "search_web":

#         query = arguments["query"]

#         state["searched_queries"].append(query)

#         return search_web(
#             arguments["query"]
#         )

#     if tool_name == "read_web_page":

#         return read_web_page(
#             arguments["url"]
#         )

#     if tool_name == "calculator":

#         return calculator(
#             arguments["a"],
#             arguments["b"],
#             arguments["operation"]
#         )

#     raise ValueError(
#         f"Unknown tool: {tool_name}"
#     )

# # tracing function for logs 
# def log_event(
#     event,
#     **data
# ):

#     print({
#         "event": event,
#         "timestamp": time.time(),
#         **data
#     })


# # research agent 

# def run_research_agent(research_question: str):

#     run_id = str(uuid.uuid4())

#     state["question"] = research_question

#     state["sources_found"] = []
#     state["sources_read"] = []
#     state["evidence"] = []
#     state["tools_used"] = []
#     state["iterations"] = 0
#     state["searched_queries"] = []

#     messages = [

#     {
#         "role": "system",

#         "content": """
# You are a web research agent.

# Your job is to investigate a research question using
# real web sources before producing an answer.

# Research process:

# 1. Understand the question.
# 2. Perform 1-2 focused searches.
# 3. Select the strongest and most relevant sources.
# 4. Read 2-4 useful sources.
# 5. Compare the evidence.
# 6. Check for important gaps or contradictions.
# 7. If necessary, perform one additional focused search.
# 8. Once there is enough evidence, stop researching and
#    provide the final answer.

# Important rules:

# - Do not invent facts.
# - Do not treat search-result snippets as strong evidence
#   when the actual source can be read.
# - Prefer primary and authoritative sources when available.
# - Distinguish facts from opinions.
# - If credible sources disagree, explain the disagreement.
# - If evidence is insufficient, explicitly say so.
# - Do not pretend certainty when the evidence does not
#   support it.
# - Do not search simply because more sources exist.
# - Use the evidence already collected when the research
#   budget is exhausted.

# You have a strict research budget.

# You should normally finish within 4-6 tool calls.

# Your final response must contain:

# Research conclusion

# Evidence
# - Important evidence discovered during research.

# Sources
# - Important sources used during the research.

# Limitations
# - Important things the available evidence does not establish.

# Do not reveal private chain-of-thought.
# Provide concise evidence and conclusions instead.
# """
#     },

#     {
#         "role": "user",
#         "content": research_question
#     }
# ]

#     start_time = time.monotonic()

#     tool_calls_count = 0
#     search_calls = 0
#     page_reads = 0

#     log_event(
#         "research_start",
#         run_id=run_id
#     )


#     for iteration in range(MAX_ITERATIONS):

#         state["iterations"] = iteration + 1

       
#         # Runtime limit

#         elapsed = (
#             time.monotonic()
#             - start_time
#         )

#         if elapsed > MAX_RUNTIME_SECONDS:

#             log_event(
#                 "research_stop",
#                 run_id=run_id,
#                 reason="runtime_limit"
#             )

#             return (
#                 "Research stopped because "
#                 "the maximum runtime was exceeded."
#             )


#         log_event(
#             "iteration_start",
#             run_id=run_id,
#             iteration=iteration + 1
#         )

#         # Model Call

#         model_start = time.monotonic()

#         response = client.chat.completions.create(

#             model=MODEL,

#             messages=messages,

#             tools=tools,

#             tool_choice="auto"
#         )

#         model_latency = (time.monotonic() - model_start)

#         log_event(
#             "model_call",
#             run_id=run_id,
#             iteration=iteration + 1,
#             latency=round(
#                 model_latency,
#                 3
#             )
#         )


#         message = (
#             response.choices[0].message
#         )


#         # IMPORTANT:
#         # Store the assistant's tool decision.
#         messages.append(message)


#         # Final answer for those that doesn't require tool


#         if not message.tool_calls:

#             log_event(
#                 "research_stop",
#                 run_id=run_id,
#                 reason="final_answer"
#             )

#             return message.content


        
#         # Execute Tool
        

#         for tool_call in message.tool_calls:

#             tool_calls_count += 1

#             if (tool_calls_count > MAX_TOOL_CALLS):

#                 log_event(
#                     "research_stop",
#                     run_id=run_id,
#                     reason="tool_call_limit"
#                 )

#                 return (
#                     "Research stopped because "
#                     "the maximum number of tool calls "
#                     "was exceeded."
#                 )


#             tool_name = (
#                 tool_call.function.name
#             )

#             # Tool-specific research budgets

#             if tool_name == "search_web":

#                 if search_calls >= MAX_SEARCH_CALLS:

#                     result = {
#                         "error": "Web search budget exhausted.",
#                         "instruction": (
#                             "Do not search again. "
#                             "Use the evidence already collected "
#                             "and provide the final answer."
#                         )
#                     }

#                     messages.append({
#                         "role": "tool",
#                         "tool_call_id": tool_call.id,
#                         "content": json.dumps(result)
#                     })

#                     continue

#                 search_calls += 1


#             if tool_name == "read_web_page":

#                 if page_reads >= MAX_PAGE_READS:

#                     result = {
#                         "error": "Page reading budget exhausted.",
#                         "instruction": (
#                             "Do not read another page. "
#                             "Use the evidence already collected "
#                             "and provide the final answer."
#                         )
#                     }

#                     messages.append({
#                         "role": "tool",
#                         "tool_call_id": tool_call.id,
#                         "content": json.dumps(result)
#                     })

#                     continue


#                 page_reads += 1

#             state["tools_used"].append(
#                 tool_name
#             )


#             log_event(
#                 "tool_call",
#                 run_id=run_id,
#                 tool=tool_name
#             )



#             # Execute

#             tool_start = time.monotonic()


#             try:

#                 result = execute_tool(
#                     tool_call
#                 )

#                 status = "success"


#             except Exception as error:

#                 result = (
#                     f"Tool failed: {error}"
#                 )

#                 status = "error"


#             tool_latency = (
#                 time.monotonic()
#                 - tool_start
#             )


#             # Store Research Information

#             if tool_name == "search_web":

#                 if isinstance(
#                     result,
#                     list
#                 ):

#                     for source in result:

#                         state[
#                             "sources_found"
#                         ].append(source)


#             elif tool_name == "read_web_page":

#                 if isinstance(
#                     result,
#                     dict
#                 ):

#                     if "content" in result:

#                         state[
#                             "sources_read"
#                         ].append(result)

#                         state[
#                             "evidence"
#                         ].append({

#                             "url": result.get(
#                                 "url"
#                             ),

#                             "title": result.get(
#                                 "title"
#                             ),

#                             "evidence": result.get(
#                                 "content"
#                             )
#                         })


#             log_event(
#                 "tool_result",
#                 run_id=run_id,
#                 tool=tool_name,
#                 status=status,
#                 latency=round(
#                     tool_latency,
#                     3
#                 )
#             )


#             # Send Tool Result back to Model

#             messages.append({

#                 "role": "tool",

#                 "tool_call_id": (
#                     tool_call.id
#                 ),

#                 "content": json.dumps(
#                     result,
#                     default=str
#                 )
#             })


#     # ITERATION LIMIT

#     log_event(
#         "research_stop",
#         run_id=run_id,
#         reason="iteration_limit"
#     )

#     return (
#         "Research stopped because "
#         "the maximum number of iterations "
#         "was reached."
#     )


# # test agent 

# question = """
# Research whether PostgreSQL is suitable for
# high-volume financial transaction systems.

# I want evidence from authoritative or technically
# credible sources.

# Compare the evidence and tell me:

# 1. What PostgreSQL is good at.
# 2. Where it may have limitations.
# 3. Whether it is commonly suitable for financial
#    transaction workloads.
# 4. What the available evidence does NOT establish.
# """

# answer = run_research_agent(
#     question
# )


# print("\n")
# print("=" * 70)
# print("FINAL RESEARCH ANSWER")
# print("=" * 70)

# print(answer)


# print("\n")
# print("=" * 70)
# print("RESEARCH STATE")
# print("=" * 70)

# print(
#     json.dumps(
#         {
#             "iterations": state["iterations"],
#             "tools_used": state["tools_used"],
#             "sources_found": len(
#                 state["sources_found"]
#             ),
#             "sources_read": len(
#                 state["sources_read"]
#             ),
#             "evidence_items": len(
#                 state["evidence"]
#             )
#         },
#         indent=2
#     )
# )


# def get_consolidated_sources():

#     sources = []

#     seen = set()

#     for source in state["sources_read"]:

#         url = source.get("url")

#         if not url:
#             continue

#         if url in seen:
#             continue

#         seen.add(url)

#         sources.append({
#             "title": source.get("title"),
#             "url": url
#         })

#     return sources

# sources = get_consolidated_sources()

# print("\n")
# print("=" * 70)
# print("SOURCES USED")
# print("=" * 70)

# for index, source in enumerate(sources, start=1):

#     print(
#         f"{index}. {source['title']}\n"
#         f"   {source['url']}\n"
#     )