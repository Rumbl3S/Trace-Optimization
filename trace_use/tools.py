"""Built-in tools for the trace_use tool-calling agent.

Provides: calculator, python_exec, write_file, wikipedia_search.
Each tool is safe to call and returns a string result.
`TOOL_DEFINITIONS` is the Anthropic API tool spec; `dispatch()` routes calls.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import urllib.parse
import urllib.request

# Shared workspace — files written here persist across turns and sessions.
# Set TRACE_USE_WORKSPACE env var to override.
_WORKSPACE = pathlib.Path(
    os.environ.get("TRACE_USE_WORKSPACE", pathlib.Path.home() / ".trace_use" / "workspace")
)


def get_workspace() -> pathlib.Path:
    _WORKSPACE.mkdir(parents=True, exist_ok=True)
    return _WORKSPACE

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
        "name": "write_file",
        "description": (
            "Write content to a file in the shared workspace (~/.trace_use/workspace/). "
            "Use this to create .py files, data files, or any project files that should "
            "persist across tool calls. After writing, use python_exec to run and test them. "
            "The workspace is added to sys.path so written modules can be imported."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path within the workspace, e.g. 'app.py' or 'utils/helpers.py'",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file.",
                },
            },
            "required": ["path", "content"],
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


def python_exec(code: str, timeout: int = 30) -> str:
    import sys
    import subprocess
    ws = str(get_workspace())
    # Run in a fresh subprocess so it can be killed on timeout and modules are never stale
    setup = f"import sys; sys.path.insert(0, {ws!r})\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", setup + code],
            capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if out and err:
            return f"{out}\nstderr: {err}"
        if out:
            return out
        if err:
            return f"stderr: {err}"
        return "(no output)"
    except subprocess.TimeoutExpired:
        return f"TimeoutError: python_exec exceeded {timeout}s"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def write_file(path: str, content: str) -> str:
    ws = get_workspace()
    # Prevent path traversal
    target = (ws / path).resolve()
    if not str(target).startswith(str(ws.resolve())):
        return "Error: path must be within the workspace directory."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written: {target}  ({len(content)} chars)"


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
        expr = tool_input.get("expression") or next(iter(tool_input.values()), "")
        return calculator(expr)
    if tool_name == "python_exec":
        code = tool_input.get("code") or tool_input.get("script") or next(iter(tool_input.values()), "")
        return python_exec(code)
    if tool_name == "write_file":
        path    = tool_input.get("path", "")
        content = tool_input.get("content", "")
        return write_file(path, content)
    if tool_name == "wikipedia_search":
        query = tool_input.get("query") or next(iter(tool_input.values()), "")
        return wikipedia_search(query)
    return f"Unknown tool: {tool_name}"
