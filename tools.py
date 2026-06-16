"""Built-in tools for the trace_use tool-calling agent.

Provides: calculator, python_exec, wikipedia_search.
Each tool is safe to call and returns a string result.
`TOOL_DEFINITIONS` is the Anthropic API tool spec; `dispatch()` routes calls.
"""
from __future__ import annotations

import io
import json
import math
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

# ── Anthropic tool definitions ────────────────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Supports standard arithmetic, math functions (sqrt, log, sin, cos, etc.), "
            "and Python numeric syntax. Use this for any numeric computation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '2 ** 10' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "python_exec",
        "description": (
            "Execute a block of Python code and return its printed output. "
            "Use for data processing, list operations, string manipulation, "
            "or any logic too complex for a single math expression. "
            "Print the result you want to see."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to output results.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for a topic and return a summary. "
            "Use for factual questions about people, places, events, concepts, "
            "history, science, or anything requiring external knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or entity to search for on Wikipedia.",
                }
            },
            "required": ["query"],
        },
    },
]

_SAFE_MATH = {
    k: v for k, v in vars(math).items() if not k.startswith("_")
}
_SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max,
                   "sum": sum, "int": int, "float": float, "len": len})


def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def python_exec(code: str) -> str:
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(code, {"__builtins__": __builtins__,
                        "math": math, "json": json})
        out = buf_out.getvalue().strip()
        err = buf_err.getvalue().strip()
        if out:
            return out
        if err:
            return f"stderr: {err}"
        return "(no output)"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def wikipedia_search(query: str) -> str:
    import ssl
    slug = urllib.parse.quote(query.strip().replace(" ", "_"))
    url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    req  = urllib.request.Request(url, headers={"User-Agent": "trace_use/1.0"})
    # macOS Python often lacks the system cert bundle — use unverified context for local use
    ctx  = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=8, context=ctx) as r:
            data = json.loads(r.read())
        extract = data.get("extract", "")
        if not extract:
            return f"No Wikipedia article found for '{query}'."
        return extract[:2000]
    except urllib.error.HTTPError as e:
        return f"No Wikipedia article found for '{query}'." if e.code == 404 else f"HTTP error {e.code}"
    except Exception as e:
        return f"Search error: {e}"


def dispatch(tool_name: str, tool_input: dict) -> str:
    if tool_name == "calculator":
        return calculator(tool_input["expression"])
    if tool_name == "python_exec":
        return python_exec(tool_input["code"])
    if tool_name == "wikipedia_search":
        return wikipedia_search(tool_input["query"])
    return f"Unknown tool: {tool_name}"
