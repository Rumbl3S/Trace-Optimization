#!/usr/bin/env python3
"""eval_dev_learning.py — Cold-start motif learning benchmark for developer tasks.

Tests whether BrainAgent learns reusable failure motifs from real LLM failures
on realistic developer tasks and prevents repeated failures in the same family.

No seed_motifs(). No hardcoded motifs. No mutation injection.
Haiku runs at configurable temperature (default 0.3) so it makes realistic mistakes.

Usage:
  python eval/eval_dev_learning.py --brain-cold --temperature 0.3
  python eval/eval_dev_learning.py --no-brain --temperature 0.3

10 families × 7 tasks (1 discovery + 4 recurrence + 2 near-miss) = 70 tasks.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from trace_use.brain import BrainAgent
from trace_use.agents import build_embedder
from trace_use.config import BrainConfig

# Auto-detect provider: prefer OpenAI when OPENAI_API_KEY is set; fall back to Anthropic.
# Override with --provider anthropic|openai on the CLI.
_HAS_OPENAI    = bool(os.environ.get("OPENAI_API_KEY"))
_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
_PROVIDER    = "openai" if _HAS_OPENAI else "anthropic"
_MODEL       = "gpt-4o-mini" if _PROVIDER == "openai" else "claude-haiku-4-5-20251001"
# Judge needs stronger instruction-following than the tool agent.
# gpt-4o-mini returns paraphrases instead of verbatim quotes → grounding check rejects them.
_JUDGE_MODEL = "gpt-4o" if _PROVIDER == "openai" else "claude-haiku-4-5-20251001"

OUT = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)


def _retry_call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential-backoff retry on 429/5xx."""
    last = None
    for attempt in range(5):
        try:
            return fn(*args, **kwargs)
        except Exception as _e:
            last = _e
            code = (getattr(getattr(_e, "response", None), "status_code", None)
                    or getattr(_e, "status_code", None))
            if code is not None and (code == 429 or code >= 500):
                wait = min(4 ** attempt, 60)
                print(f"  [RATE LIMIT {code}] backing off {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise
    raise last


# ── Provider-aware tool agent ─────────────────────────────────────────────────

def _make_tool_agent(temperature: float = 0.3, max_turns: int = 5, max_tokens: int = 4096,
                     provider: str = _PROVIDER, model: str = _MODEL):
    """tool_agent variant with configurable temperature, supporting Anthropic and OpenAI."""
    from trace_use.tools import TOOL_DEFINITIONS, dispatch
    import json as _json

    anthropic_tool_defs = [t for t in TOOL_DEFINITIONS if t["name"] == "python_exec"]
    # OpenAI tool format wraps input_schema as "parameters" inside a "function" object.
    openai_tool_defs = [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"],
        }} for t in anthropic_tool_defs
    ]

    _client_holder: list = [None]

    def _get_client():
        if _client_holder[0] is None:
            if provider == "openai":
                from openai import OpenAI
                _client_holder[0] = OpenAI()
            else:
                import anthropic
                _client_holder[0] = anthropic.Anthropic()
        return _client_holder[0]

    def _run_turn_anthropic(client, messages):
        r = _retry_call(
            client.messages.create,
            model=model, max_tokens=max_tokens, temperature=temperature,
            tools=anthropic_tool_defs, messages=messages,
        )
        text_blocks, tool_calls_raw = [], []
        for block in r.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls_raw.append(("anthropic", block))
        tokens = r.usage.input_tokens + r.usage.output_tokens
        done = r.stop_reason == "end_turn"
        return text_blocks, tool_calls_raw, tokens, done, r.content

    def _run_turn_openai(client, messages):
        r = _retry_call(
            client.chat.completions.create,
            model=model, max_tokens=max_tokens, temperature=temperature,
            tools=openai_tool_defs, messages=messages,
        )
        choice = r.choices[0]
        msg = choice.message
        text_blocks = [msg.content] if msg.content else []
        tool_calls_raw = [("openai", tc) for tc in (msg.tool_calls or [])]
        tokens = r.usage.prompt_tokens + r.usage.completion_tokens
        done = choice.finish_reason == "stop"
        return text_blocks, tool_calls_raw, tokens, done, msg

    def agent(prompt: str):
        client = _get_client()
        messages = [{"role": "user", "content": prompt}]
        trace_parts: list[str] = []
        total_tokens = 0
        monitor = getattr(agent, "monitor", None)

        for _ in range(max_turns):
            if provider == "openai":
                text_blocks, tool_calls_raw, tok, done, raw_msg = _run_turn_openai(client, messages)
            else:
                text_blocks, tool_calls_raw, tok, done, raw_msg = _run_turn_anthropic(client, messages)
            total_tokens += tok

            for txt in text_blocks:
                trace_parts.append(txt)
                if monitor:
                    monitor.push(txt)

            tool_result_msgs = []
            anthropic_results = []
            for kind, tc in tool_calls_raw:
                if kind == "openai":
                    name = tc.function.name
                    try:
                        input_dict = _json.loads(tc.function.arguments)
                    except Exception:
                        input_dict = {}
                    tc_id = tc.id
                else:  # anthropic
                    name, input_dict, tc_id = tc.name, tc.input, tc.id

                pre = None
                if monitor and hasattr(monitor, "before_tool_call"):
                    pre = monitor.before_tool_call(name, input_dict)
                result = pre if pre is not None else dispatch(name, input_dict)
                if monitor and hasattr(monitor, "on_tool_call"):
                    mod = monitor.on_tool_call(name, input_dict, result)
                    if mod is not None:
                        result = mod

                chunk = f"[tool:{name}({_json.dumps(input_dict)})] → {result[:500]}"
                trace_parts.append(chunk)
                if monitor:
                    monitor.push(chunk)

                if kind == "openai":
                    tool_result_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": result[:4000]})
                else:
                    anthropic_results.append({"type": "tool_result", "tool_use_id": tc_id, "content": result[:4000]})

            if done or not tool_calls_raw:
                break
            if monitor and hasattr(monitor, "pulse"):
                monitor.pulse()
            if monitor and getattr(monitor, "should_bail", False):
                trace_parts.append("[EARLY_EXIT]")
                break

            if provider == "openai":
                # OpenAI: append assistant msg (with tool_calls), then tool result msgs
                assistant_dict: dict = {"role": "assistant", "content": raw_msg.content}
                if raw_msg.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for _, tc in tool_calls_raw
                    ]
                messages.append(assistant_dict)
                messages.extend(tool_result_msgs)
            else:
                messages.append({"role": "assistant", "content": raw_msg})
                messages.append({"role": "user", "content": anthropic_results})

        return "\n".join(trace_parts), total_tokens

    agent.monitor = None
    return agent


# ── Code extraction + execution ───────────────────────────────────────────────

def _extract_last_code(trace: str) -> str:
    last_code = ""
    idx = 0
    while True:
        start = trace.find("[tool:python_exec(", idx)
        if start < 0:
            break
        brace = trace.find("{", start)
        if brace < 0:
            break
        depth = 0
        end = brace
        for i, ch in enumerate(trace[brace:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = brace + i + 1
                    break
        try:
            d = json.loads(trace[brace:end])
            code = d.get("code", "")
            if code:
                last_code = code
        except Exception:
            pass
        idx = end
    if not last_code:
        blocks = re.findall(r"```python\n(.*?)```", trace, re.DOTALL)
        if blocks:
            last_code = blocks[-1]
    return last_code


def _exec_ns(code: str) -> "dict | str":
    ns: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<bench>", "exec"), ns)
        return ns
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# ── Task dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    name: str
    family: str
    kind: str   # "discovery" | "recurrence" | "near_miss"
    prompt: str
    verify: Callable[[dict], tuple[bool, str]]

    def run_verify(self, code: str) -> tuple[bool, str]:
        if not code:
            return False, "no code extracted from trace"
        ns = _exec_ns(code)
        if isinstance(ns, str):
            return False, f"exec error: {ns[:200]}"
        try:
            return self.verify(ns)
        except AssertionError as e:
            return False, str(e)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


@dataclass
class TaskResult:
    task: Task
    passed: bool
    fired: bool
    detail: str
    tokens: int
    time_s: float
    motif_learned: bool = False
    fire_detail: dict | None = None


# ── Task definitions ──────────────────────────────────────────────────────────

def _make_tasks() -> list[Task]:
    tasks: list[Task] = []

    # ── Family 1: nested_key_direct_index ────────────────────────────────────
    # Bug: record["key"] raises KeyError when key missing. Fix: record.get("key")

    def _v_extract_emails(ns):
        fn = ns.get("extract_emails")
        if not fn:
            return False, "extract_emails not defined"
        try:
            r = fn([{"name": "Alice", "email": "a@b.com"}, {"name": "Bob"}, {"email": None}])
            if r != ["a@b.com", None, None]:
                return False, f"basic case failed: got {r}"
            if fn([]) != []:
                return False, "empty list failed"
            if fn([{"name": "x"}, {"id": 1}]) != [None, None]:
                return False, "all-missing case failed"
            return True, ""
        except KeyError as e:
            return False, f"KeyError on missing key {e} — use .get()"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_emails", "nested_key", "discovery",
        "Write `extract_emails(records: list[dict]) -> list[str | None]`. "
        "We have a list of user dicts from our CRM export. Pull out the email address for each user. "
        "Call python_exec to implement it.",
        _v_extract_emails))

    def _v_extract_phones(ns):
        fn = ns.get("extract_phones")
        if not fn:
            return False, "extract_phones not defined"
        try:
            r = fn([{"phone": "555-1234"}, {"name": "Bob"}, {"phone": None}])
            if r != ["555-1234", None, None]:
                return False, f"got {r}"
            if fn([{"id": 1}, {"id": 2}]) != [None, None]:
                return False, "missing-key case failed"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e} — use .get()"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_phones", "nested_key", "recurrence",
        "Write `extract_phones(records: list[dict]) -> list[str | None]`. "
        "We have contact dicts from our address book sync. Pull the phone number for each contact. "
        "Call python_exec to implement it.",
        _v_extract_phones))

    def _v_extract_tags(ns):
        fn = ns.get("extract_tags")
        if not fn:
            return False, "extract_tags not defined"
        try:
            r = fn([{"tags": ["a", "b"]}, {"name": "x"}, {"tags": None}])
            if r != [["a", "b"], [], []]:
                return False, f"got {r}"
            if fn([{"id": 1}]) != [[]]:
                return False, "missing key → should return [[]]"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e} — use .get()"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_tags", "nested_key", "recurrence",
        "Write `extract_tags(records: list[dict]) -> list[list]`. "
        "We have post dicts from our CMS API. Get the tags for each post. "
        "Call python_exec to implement it.",
        _v_extract_tags))

    def _v_extract_scores(ns):
        fn = ns.get("extract_scores")
        if not fn:
            return False, "extract_scores not defined"
        try:
            r = fn([{"score": 95}, {"name": "Bob"}, {"score": None}])
            if r != [95, 0, 0]:
                return False, f"got {r}"
            if fn([{"id": 1}, {"id": 2}]) != [0, 0]:
                return False, "missing keys should return 0"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e} — use .get()"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_scores", "nested_key", "recurrence",
        "Write `extract_scores(records: list[dict]) -> list[int]`. "
        "We have game records from the leaderboard API. Pull the score for each player (default 0). "
        "Call python_exec to implement it.",
        _v_extract_scores))

    def _v_extract_city(ns):
        fn = ns.get("extract_city")
        if not fn:
            return False, "extract_city not defined"
        try:
            r = fn([{"address": {"city": "NYC"}}, {"name": "Bob"}, {"address": None}, {"address": {}}])
            if r != ["NYC", None, None, None]:
                return False, f"got {r}"
            return True, ""
        except (KeyError, TypeError) as e:
            return False, f"{type(e).__name__} {e} — use .get() at each level"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_city", "nested_key", "recurrence",
        "Write `extract_city(records: list[dict]) -> list[str | None]`. "
        "We have user profile dicts. Pull the city from each user's address. "
        "Call python_exec to implement it.",
        _v_extract_city))

    # Near-miss: 'id' is guaranteed present — direct indexing IS correct
    def _v_extract_ids(ns):
        fn = ns.get("extract_ids")
        if not fn:
            return False, "extract_ids not defined"
        try:
            r = fn([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
            if r != [1, 2]:
                return False, f"got {r}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_ids", "nested_key", "near_miss",
        "Write `extract_ids(records: list[dict]) -> list[int]`. "
        "Every record is guaranteed to have an 'id' key (it is always present). "
        "Return the list of id values. "
        "Call python_exec to implement and test.",
        _v_extract_ids))

    # Near-miss: using dict.get() is correct and task says so explicitly
    def _v_get_defaults(ns):
        fn = ns.get("get_with_defaults")
        if not fn:
            return False, "get_with_defaults not defined"
        try:
            r = fn({"x": 5}, "x", 0)
            if r != 5:
                return False, f"present key: got {r}"
            r2 = fn({}, "x", 42)
            if r2 != 42:
                return False, f"missing key: got {r2}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_with_defaults", "nested_key", "near_miss",
        "Write `get_with_defaults(d: dict, key: str, default) -> any`. "
        "Return d[key] if present, otherwise return default. Use dict.get(). "
        "Call python_exec to implement and test.",
        _v_get_defaults))

    # ── Family 2: shared_class_state ──────────────────────────────────────────
    # Bug: class-level mutable defaults (e.g. _items = []) shared across instances

    def _v_history(ns):
        cls = ns.get("History")
        if not cls:
            return False, "History class not defined"
        try:
            h1 = cls()
            h2 = cls()
            h1.append("a")
            items2 = h2.items()
            if items2 != []:
                return False, f"h2 should be empty after h1.append, got {items2} — shared state bug"
            h2.append("b")
            if h1.items() != ["a"]:
                return False, f"h1.items() should be ['a'], got {h1.items()}"
            if h2.items() != ["b"]:
                return False, f"h2.items() should be ['b'], got {h2.items()}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("history_class", "shared_state", "discovery",
        "Write a `History` class with an `append(item)` method and an `items()` method "
        "that returns everything appended so far. "
        "Call python_exec to implement it.",
        _v_history))

    def _v_counter(ns):
        cls = ns.get("EventCounter")
        if not cls:
            return False, "EventCounter class not defined"
        try:
            c1 = cls()
            c2 = cls()
            c1.record("click")
            c1.record("click")
            c2.record("view")
            if c1.count("click") != 2:
                return False, f"c1.count('click') should be 2, got {c1.count('click')}"
            if c2.count("click") != 0:
                return False, f"c2 should not see c1's clicks, got {c2.count('click')} — shared state"
            if c2.count("view") != 1:
                return False, f"c2.count('view') should be 1, got {c2.count('view')}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("event_counter_class", "shared_state", "recurrence",
        "Write an `EventCounter` class with a `record(event_name)` method and a `count(event_name) -> int` method. "
        "Call python_exec to implement it.",
        _v_counter))

    def _v_cache(ns):
        cls = ns.get("SimpleCache")
        if not cls:
            return False, "SimpleCache class not defined"
        try:
            c1 = cls()
            c2 = cls()
            c1.put("key", "val")
            if c2.get("key") is not None:
                return False, f"c2 should not see c1's data, got {c2.get('key')} — shared state"
            if c1.get("key") != "val":
                return False, f"c1.get('key') should be 'val', got {c1.get('key')}"
            if c1.get("missing") is not None:
                return False, f"missing key should return None"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("simple_cache_class", "shared_state", "recurrence",
        "Write a `SimpleCache` class with `put(key, value)` and `get(key)` methods. "
        "Call python_exec to implement it.",
        _v_cache))

    def _v_budget(ns):
        cls = ns.get("BudgetTracker")
        if not cls:
            return False, "BudgetTracker class not defined"
        try:
            b1 = cls(100)
            b2 = cls(200)
            b1.spend(30)
            if b2.remaining() == 70:
                return False, f"b2 should be 200, not 70 — shared expenses list"
            if b1.remaining() != 70:
                return False, f"b1.remaining() should be 70, got {b1.remaining()}"
            if b2.remaining() != 200:
                return False, f"b2.remaining() should be 200, got {b2.remaining()}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("budget_tracker_class", "shared_state", "recurrence",
        "Write a `BudgetTracker` class. Init takes a budget limit. `spend(amount)` records an expense. `remaining()` returns how much is left. "
        "Call python_exec to implement it.",
        _v_budget))

    def _v_queue(ns):
        cls = ns.get("TaskQueue")
        if not cls:
            return False, "TaskQueue class not defined"
        try:
            q1 = cls()
            q2 = cls()
            q1.push("task_a")
            if q2.pop() is not None:
                return False, f"q2 should be empty, got {q2.pop()} — shared state"
            item = q1.pop()
            if item != "task_a":
                return False, f"q1.pop() should return 'task_a', got {item}"
            if q1.pop() is not None:
                return False, "empty queue should return None"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("task_queue_class", "shared_state", "recurrence",
        "Write a `TaskQueue` class with `push(task)` and `pop()` methods (FIFO, returns None when empty). "
        "Call python_exec to implement it.",
        _v_queue))

    # Near-miss: only instance vars used — no class-level state bug possible
    def _v_point(ns):
        cls = ns.get("Point")
        if not cls:
            return False, "Point class not defined"
        try:
            p1 = cls(1, 2)
            p2 = cls(3, 4)
            if p1.x != 1 or p1.y != 2:
                return False, f"p1 coords wrong: ({p1.x},{p1.y})"
            if p2.x != 3 or p2.y != 4:
                return False, f"p2 coords wrong: ({p2.x},{p2.y})"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("point_class", "shared_state", "near_miss",
        "Write a `Point` class with x and y coordinates:\n"
        "  - __init__(x: float, y: float)\n"
        "  - attributes self.x and self.y\n"
        "This is a simple value class. Call python_exec to implement and test.",
        _v_point))

    def _v_counter_simple(ns):
        cls = ns.get("Counter")
        if not cls:
            return False, "Counter class not defined"
        try:
            c = cls()
            c.increment()
            c.increment()
            if c.value() != 2:
                return False, f"value should be 2, got {c.value()}"
            c2 = cls()
            if c2.value() != 0:
                return False, f"new instance should start at 0, got {c2.value()}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("simple_counter_class", "shared_state", "near_miss",
        "Write a `Counter` class with:\n"
        "  - increment(): adds 1 to the count\n"
        "  - value() -> int: returns current count (starts at 0)\n"
        "Call python_exec to implement and test. Each instance starts at 0.",
        _v_counter_simple))

    # ── Family 3: api_key_assumption ─────────────────────────────────────────
    # Bug: assumes response["data"] — but may be "items", "results", etc.

    def _v_extract_items(ns):
        fn = ns.get("extract_items")
        if not fn:
            return False, "extract_items not defined"
        try:
            if fn({"data": [1, 2, 3]}) != [1, 2, 3]:
                return False, "data key failed"
            if fn({"items": [4, 5]}) != [4, 5]:
                return False, "items key failed — only checked 'data'"
            if fn({"results": [6]}) != [6]:
                return False, "results key failed"
            if fn({}) != []:
                return False, "empty response should return []"
            try:
                fn({"error": "not found"})
                return False, "should raise ValueError on error"
            except ValueError:
                pass
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e} — check multiple possible keys"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("extract_items", "api_key", "discovery",
        "Write `extract_items(response: dict) -> list`. "
        "We call a third-party API and get back a dict. Pull out the list of items from it. "
        "Call python_exec to implement it.",
        _v_extract_items))

    def _v_next_cursor(ns):
        fn = ns.get("get_next_cursor")
        if not fn:
            return False, "get_next_cursor not defined"
        try:
            if fn({"next_cursor": "abc"}) != "abc":
                return False, "next_cursor failed"
            if fn({"cursor": "def"}) != "def":
                return False, "cursor key failed — only checked 'next_cursor'"
            if fn({"next_page": "ghi"}) != "ghi":
                return False, "next_page key failed"
            if fn({}) is not None:
                return False, "no cursor should return None"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_next_cursor", "api_key", "recurrence",
        "Write `get_next_cursor(response: dict) -> str | None`. "
        "Our paginated API returns a cursor somewhere in the response. Pull it out. "
        "Call python_exec to implement it.",
        _v_next_cursor))

    def _v_total_count(ns):
        fn = ns.get("get_total_count")
        if not fn:
            return False, "get_total_count not defined"
        try:
            if fn({"total": 100}) != 100:
                return False, "total key failed"
            if fn({"count": 50}) != 50:
                return False, "count key failed"
            if fn({"n": 25}) != 25:
                return False, "n key failed"
            if fn({}) != 0:
                return False, "missing should return 0"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_total_count", "api_key", "recurrence",
        "Write `get_total_count(response: dict) -> int`. "
        "Pull the total item count from an API response. Return 0 if not present. "
        "Call python_exec to implement it.",
        _v_total_count))

    def _v_normalize_user(ns):
        fn = ns.get("normalize_user")
        if not fn:
            return False, "normalize_user not defined"
        try:
            r1 = fn({"user_id": 1, "name": "Alice"})
            if r1.get("id") != 1:
                return False, f"user_id→id mapping failed: {r1}"
            r2 = fn({"id": 2, "name": "Bob"})
            if r2.get("id") != 2:
                return False, f"id passthrough failed: {r2}"
            r3 = fn({"userId": 3, "name": "Carol"})
            if r3.get("id") != 3:
                return False, f"userId→id mapping failed: {r3}"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("normalize_user", "api_key", "recurrence",
        "Write `normalize_user(user: dict) -> dict`. "
        "We integrate with three different user APIs that each use a different field name for the user ID. Normalize to a standard dict with an 'id' field. "
        "Call python_exec to implement it.",
        _v_normalize_user))

    def _v_merge_lists(ns):
        fn = ns.get("merge_response_lists")
        if not fn:
            return False, "merge_response_lists not defined"
        try:
            r1 = fn({"items": [1, 2], "extra": [3]})
            if set(r1) != {1, 2, 3}:
                return False, f"both lists failed: {r1}"
            r2 = fn({"items": [4, 5]})
            if r2 != [4, 5]:
                return False, f"only items key: {r2}"
            r3 = fn({"data": [6], "records": [7]})
            if set(r3) != {6, 7}:
                return False, f"data+records failed: {r3}"
            if fn({}) != []:
                return False, "empty should return []"
            return True, ""
        except KeyError as e:
            return False, f"KeyError {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("merge_response_lists", "api_key", "recurrence",
        "Write `merge_response_lists(response: dict) -> list`. "
        "Some of our API endpoints return items under different keys. Collect everything into one flat list. "
        "Call python_exec to implement it.",
        _v_merge_lists))

    # Near-miss: 'data' key is always present — direct access is fine
    def _v_always_data(ns):
        fn = ns.get("get_data_list")
        if not fn:
            return False, "get_data_list not defined"
        try:
            r = fn({"data": [1, 2, 3], "meta": {}})
            if r != [1, 2, 3]:
                return False, f"got {r}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_data_list", "api_key", "near_miss",
        "Write `get_data_list(response: dict) -> list`. "
        "The response always has a 'data' key containing a list — this is guaranteed by the API spec. "
        "Return response['data']. "
        "Call python_exec to implement and test.",
        _v_always_data))

    def _v_status(ns):
        fn = ns.get("get_status")
        if not fn:
            return False, "get_status not defined"
        try:
            r = fn({"status": "ok", "code": 200})
            if r != "ok":
                return False, f"got {r}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_status", "api_key", "near_miss",
        "Write `get_status(response: dict) -> str`. "
        "Every response has a 'status' key. Return response['status']. "
        "Call python_exec to implement and test.",
        _v_status))

    # ── Family 4: validation_stops_early ─────────────────────────────────────
    # Bug: return on first error instead of collecting all errors

    def _v_validate_password(ns):
        fn = ns.get("validate_password")
        if not fn:
            return False, "validate_password not defined"
        try:
            if fn("Good1Pass") != []:
                return False, f"valid password got errors: {fn('Good1Pass')}"
            errors_short_no_upper = fn("abc1")
            if len(errors_short_no_upper) < 2:
                return False, f"expected ≥2 errors for 'abc1', got {errors_short_no_upper} — stops at first?"
            errors_all_bad = fn("abc")
            if len(errors_all_bad) < 3:
                return False, f"expected 3 errors for 'abc', got {errors_all_bad} — stops at first?"
            if fn("ABCDEFGH1") != []:
                return False, "ABCDEFGH1 should be valid"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("validate_password", "validation_all_errors", "discovery",
        "Write `validate_password(password: str) -> list[str]`. "
        "Password must be at least 8 chars, have a digit, and have an uppercase letter. "
        "Return a list of error messages for any rules that fail. "
        "Call python_exec to implement it.",
        _v_validate_password))

    def _v_validate_email(ns):
        fn = ns.get("validate_email")
        if not fn:
            return False, "validate_email not defined"
        try:
            if fn("user@example.com") != []:
                return False, f"valid email got errors"
            errors = fn("no-at-sign")
            if len(errors) < 1:
                return False, "missing @ should be an error"
            errors_all = fn("no at no dot")
            if len(errors_all) < 2:
                return False, f"'no at no dot' should have ≥2 errors (no @, has space), got {errors_all}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("validate_email", "validation_all_errors", "recurrence",
        "Write `validate_email(email: str) -> list[str]`. "
        "Return a list of ALL validation failures from these rules:\n"
        "  1. must contain exactly one '@'\n"
        "  2. domain part (after @) must contain a '.'\n"
        "  3. must not contain spaces\n"
        "Return [] if valid. Collect ALL failures, not just the first. "
        "Call python_exec to implement and test.",
        _v_validate_email))

    def _v_validate_username(ns):
        fn = ns.get("validate_username")
        if not fn:
            return False, "validate_username not defined"
        try:
            if fn("alice_123") != []:
                return False, "valid username got errors"
            errors = fn("a!")
            # Should catch: too short (< 3) AND invalid char
            if len(errors) < 2:
                return False, f"'a!' should have ≥2 errors (length + invalid char), got {errors}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("validate_username", "validation_all_errors", "recurrence",
        "Write `validate_username(username: str) -> list[str]`. "
        "Username rules: 3-20 chars, only letters/digits/underscores, cannot start with a digit. "
        "Return any violations as a list of strings. "
        "Call python_exec to implement it.",
        _v_validate_username))

    def _v_validate_config(ns):
        fn = ns.get("validate_config")
        if not fn:
            return False, "validate_config not defined"
        try:
            if fn({"host": "localhost", "port": 8080, "timeout": 30}) != []:
                return False, "valid config got errors"
            errors = fn({"host": "x", "port": -1})
            # Missing timeout AND port out of range
            if len(errors) < 2:
                return False, f"expected ≥2 errors (missing timeout, bad port), got {errors}"
            errors_empty = fn({})
            if len(errors_empty) < 3:
                return False, f"empty dict should have 3 errors, got {errors_empty}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("validate_config", "validation_all_errors", "recurrence",
        "Write `validate_config(cfg: dict) -> list[str]`. "
        "Validate a server config: host (non-empty string), port (1-65535), timeout (positive). "
        "Return any violations found. "
        "Call python_exec to implement it.",
        _v_validate_config))

    def _v_validate_sku(ns):
        fn = ns.get("validate_sku")
        if not fn:
            return False, "validate_sku not defined"
        try:
            if fn("PROD-001-XL") != []:
                return False, f"valid SKU got errors: {fn('PROD-001-XL')}"
            # Too short, no hyphen, lowercase
            errors = fn("ab")
            if len(errors) < 2:
                return False, f"'ab' should have ≥2 errors, got {errors}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("validate_sku", "validation_all_errors", "recurrence",
        "Write `validate_sku(sku: str) -> list[str]`. "
        "SKU rules: min 6 chars, must include a hyphen, must be uppercase. "
        "Return any violations. "
        "Call python_exec to implement it.",
        _v_validate_sku))

    # Near-miss: explicitly return FIRST error — early return IS correct
    def _v_first_error(ns):
        fn = ns.get("first_error")
        if not fn:
            return False, "first_error not defined"
        try:
            if fn("Good1Pass") is not None:
                return False, "valid should return None"
            err = fn("abc")
            if err is None:
                return False, "invalid should return an error string"
            if not isinstance(err, str):
                return False, f"should return a string, got {type(err)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("first_error", "validation_all_errors", "near_miss",
        "Write `first_error(password: str) -> str | None`. "
        "Return the FIRST failing rule message, or None if valid. "
        "Stop checking after the first failure (early return is correct here). "
        "Rules: length >= 8, contains a digit, contains uppercase. "
        "Call python_exec to implement and test.",
        _v_first_error))

    def _v_is_valid(ns):
        fn = ns.get("is_valid_email")
        if not fn:
            return False, "is_valid_email not defined"
        try:
            if fn("user@example.com") is not True:
                return False, "valid email should return True"
            if fn("bad") is not False:
                return False, "invalid email should return False"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("is_valid_email", "validation_all_errors", "near_miss",
        "Write `is_valid_email(email: str) -> bool`. "
        "Return True if valid, False if not. This is a boolean check — only one answer. "
        "An email is valid if it contains '@' and a '.' in the domain. "
        "Call python_exec to implement and test.",
        _v_is_valid))

    # ── Family 5: off_by_one_1indexed ────────────────────────────────────────
    # Bug: uses 0-indexed page number when task specifies 1-indexed

    def _v_get_page(ns):
        fn = ns.get("get_page")
        if not fn:
            return False, "get_page not defined"
        try:
            items = list(range(10))
            r1 = fn(items, 3, 1)
            if r1 != [0, 1, 2]:
                return False, f"page 1 wrong: {r1} (0-indexed bug returns {items[3:6]}?)"
            r2 = fn(items, 3, 2)
            if r2 != [3, 4, 5]:
                return False, f"page 2 wrong: {r2}"
            r4 = fn(items, 3, 4)
            if r4 != [9]:
                return False, f"partial last page wrong: {r4}"
            r5 = fn(items, 3, 5)
            if r5 != []:
                return False, f"out of range should be [], got {r5}"
            if fn([], 3, 1) != []:
                return False, "empty list failed"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_page", "off_by_one", "discovery",
        "Write `get_page(items: list, page_size: int, page_number: int) -> list`. "
        "Our API uses page numbers starting at 1. Return the items for that page. "
        "Call python_exec to implement it.",
        _v_get_page))

    def _v_nth_item(ns):
        fn = ns.get("nth_item")
        if not fn:
            return False, "nth_item not defined"
        try:
            items = ["a", "b", "c", "d"]
            if fn(items, 1) != "a":
                return False, f"1st item should be 'a', got {fn(items, 1)} (0-indexed bug?)"
            if fn(items, 2) != "b":
                return False, f"2nd item should be 'b', got {fn(items, 2)}"
            if fn(items, 4) != "d":
                return False, f"4th item should be 'd', got {fn(items, 4)}"
            if fn(items, 5) is not None:
                return False, f"out of range should return None"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("nth_item", "off_by_one", "recurrence",
        "Write `nth_item(items: list, n: int) -> any`. "
        "Get the nth item from a list. Our users count from 1, not 0. Return None if out of range. "
        "Call python_exec to implement it.",
        _v_nth_item))

    def _v_rank(ns):
        fn = ns.get("rank_of")
        if not fn:
            return False, "rank_of not defined"
        try:
            scores = [10, 50, 30, 50, 20]
            # Sorted desc: 50,50,30,20,10. Value 10 is rank 5.
            if fn(scores, 10) != 5:
                return False, f"rank of 10 should be 5 (1-indexed), got {fn(scores, 10)}"
            if fn(scores, 50) != 1:
                return False, f"rank of 50 (highest) should be 1, got {fn(scores, 50)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("rank_of", "off_by_one", "recurrence",
        "Write `rank_of(scores: list[int], value: int) -> int`. "
        "Return what rank a given score has in a leaderboard (top score = #1). "
        "Call python_exec to implement it.",
        _v_rank))

    def _v_chunk(ns):
        fn = ns.get("chunk_number")
        if not fn:
            return False, "chunk_number not defined"
        try:
            # chunk 1 = items 0..2, chunk 2 = items 3..5, chunk 3 = items 6..8
            items = list(range(9))
            r1 = fn(items, 3, 1)
            if r1 != [0, 1, 2]:
                return False, f"chunk 1 wrong: {r1}"
            r2 = fn(items, 3, 2)
            if r2 != [3, 4, 5]:
                return False, f"chunk 2 wrong: {r2}"
            r3 = fn(items, 3, 3)
            if r3 != [6, 7, 8]:
                return False, f"chunk 3 wrong: {r3}"
            r4 = fn(items, 3, 4)
            if r4 != []:
                return False, f"chunk 4 (out of range) should be [], got {r4}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("chunk_number", "off_by_one", "recurrence",
        "Write `chunk_number(items: list, chunk_size: int, chunk_num: int) -> list`. "
        "Return a specific batch from a list. Batches start at 1. Return [] if out of range. "
        "Call python_exec to implement it.",
        _v_chunk))

    def _v_skip_n(ns):
        fn = ns.get("skip_n")
        if not fn:
            return False, "skip_n not defined"
        try:
            items = list(range(10))
            # skip_n(items, 1) skips the 1st item → [1,2,...,9]
            r1 = fn(items, 1)
            if r1 != list(range(1, 10)):
                return False, f"skip 1 item wrong: {r1}"
            # skip_n(items, 3) skips first 3 → [3,4,...,9]
            r3 = fn(items, 3)
            if r3 != list(range(3, 10)):
                return False, f"skip 3 items wrong: {r3}"
            if fn(items, 0) != items:
                return False, "skip 0 should return all"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("skip_n", "off_by_one", "recurrence",
        "Write `skip_n(items: list, n: int) -> list`. "
        "Skip the first n items from a list and return the rest. "
        "Call python_exec to implement it.",
        _v_skip_n))

    # Near-miss: explicitly 0-indexed — no off-by-one bug
    def _v_zero_indexed(ns):
        fn = ns.get("get_item_at")
        if not fn:
            return False, "get_item_at not defined"
        try:
            items = ["a", "b", "c"]
            if fn(items, 0) != "a":
                return False, f"index 0 should be 'a', got {fn(items, 0)}"
            if fn(items, 2) != "c":
                return False, f"index 2 should be 'c'"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("get_item_at", "off_by_one", "near_miss",
        "Write `get_item_at(items: list, index: int) -> any`. "
        "Return items[index] using standard 0-based Python indexing. "
        "index=0 returns the first element. Return None if index is out of range. "
        "Call python_exec to implement and test.",
        _v_zero_indexed))

    def _v_count_items(ns):
        fn = ns.get("count_items")
        if not fn:
            return False, "count_items not defined"
        try:
            if fn([1, 2, 3]) != 3:
                return False, "wrong count"
            if fn([]) != 0:
                return False, "empty list wrong"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("count_items", "off_by_one", "near_miss",
        "Write `count_items(items: list) -> int`. Return the number of items in the list. "
        "Call python_exec to implement and test.",
        _v_count_items))

    # ── Family 6: unit_scale ─────────────────────────────────────────────────
    # Bug: multiplies when should divide, or uses wrong factor

    def _v_ms_to_sec(ns):
        fn = ns.get("ms_to_seconds")
        if not fn:
            return False, "ms_to_seconds not defined"
        try:
            if fn(1000) != 1.0:
                return False, f"1000ms should be 1.0s, got {fn(1000)}"
            if fn(500) != 0.5:
                return False, f"500ms → 0.5s, got {fn(500)}"
            if fn(1) != 0.001:
                return False, f"1ms → 0.001s, got {fn(1)}"
            if fn(0) != 0.0:
                return False, "0ms → 0.0s"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("ms_to_seconds", "unit_scale", "discovery",
        "Write `ms_to_seconds(milliseconds: float) -> float`. "
        "Our logging system records durations in milliseconds. Convert to seconds. "
        "Call python_exec to implement it.",
        _v_ms_to_sec))

    def _v_cents_to_dollars(ns):
        fn = ns.get("cents_to_dollars")
        if not fn:
            return False, "cents_to_dollars not defined"
        try:
            if abs(fn(100) - 1.0) > 1e-9:
                return False, f"100 cents → $1.00, got {fn(100)}"
            if abs(fn(50) - 0.5) > 1e-9:
                return False, f"50 cents → $0.50, got {fn(50)}"
            if abs(fn(1) - 0.01) > 1e-9:
                return False, f"1 cent → $0.01, got {fn(1)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("cents_to_dollars", "unit_scale", "recurrence",
        "Write `cents_to_dollars(cents: int) -> float`. "
        "Our payment system stores amounts in cents. Convert to dollars for display. "
        "Call python_exec to implement it.",
        _v_cents_to_dollars))

    def _v_kb_to_bytes(ns):
        fn = ns.get("kb_to_bytes")
        if not fn:
            return False, "kb_to_bytes not defined"
        try:
            if fn(1) != 1024:
                return False, f"1 KB should be 1024 bytes, got {fn(1)}"
            if fn(2) != 2048:
                return False, f"2 KB → 2048 bytes, got {fn(2)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("kb_to_bytes", "unit_scale", "recurrence",
        "Write `kb_to_bytes(kb: float) -> float`. "
        "Convert kilobytes to bytes for our file size calculator. "
        "Call python_exec to implement it.",
        _v_kb_to_bytes))

    def _v_pct_to_dec(ns):
        fn = ns.get("pct_to_decimal")
        if not fn:
            return False, "pct_to_decimal not defined"
        try:
            if abs(fn(100) - 1.0) > 1e-9:
                return False, f"100% → 1.0, got {fn(100)}"
            if abs(fn(5) - 0.05) > 1e-9:
                return False, f"5% → 0.05, got {fn(5)}"
            if abs(fn(50) - 0.5) > 1e-9:
                return False, f"50% → 0.5, got {fn(50)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("pct_to_decimal", "unit_scale", "recurrence",
        "Write `pct_to_decimal(pct: float) -> float`. "
        "Our config stores rates as whole numbers like 5 for 5%. Convert to decimal for calculations. "
        "Call python_exec to implement it.",
        _v_pct_to_dec))

    def _v_bytes_to_mb(ns):
        fn = ns.get("bytes_to_mb")
        if not fn:
            return False, "bytes_to_mb not defined"
        try:
            expected = 1024 * 1024
            if abs(fn(expected) - 1.0) > 1e-6:
                return False, f"1MB bytes → 1.0 MB, got {fn(expected)}"
            if abs(fn(512 * 1024) - 0.5) > 1e-6:
                return False, f"512KB → 0.5 MB, got {fn(512*1024)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("bytes_to_mb", "unit_scale", "recurrence",
        "Write `bytes_to_mb(bytes_val: int) -> float`. "
        "Convert a file size in bytes to megabytes for our storage dashboard. "
        "Call python_exec to implement it.",
        _v_bytes_to_mb))

    # Near-miss: multiply IS correct
    def _v_sec_to_ms(ns):
        fn = ns.get("seconds_to_ms")
        if not fn:
            return False, "seconds_to_ms not defined"
        try:
            if fn(1.0) != 1000:
                return False, f"1.0s → 1000ms, got {fn(1.0)}"
            if fn(0.5) != 500:
                return False, f"0.5s → 500ms, got {fn(0.5)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("seconds_to_ms", "unit_scale", "near_miss",
        "Write `seconds_to_ms(seconds: float) -> float`. "
        "Convert seconds to milliseconds. Multiply by 1000. "
        "Call python_exec to implement and test: seconds_to_ms(1.0) = 1000.",
        _v_sec_to_ms))

    def _v_mb_to_bytes(ns):
        fn = ns.get("mb_to_bytes")
        if not fn:
            return False, "mb_to_bytes not defined"
        try:
            if fn(1) != 1024 * 1024:
                return False, f"1 MB → {1024*1024} bytes, got {fn(1)}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("mb_to_bytes", "unit_scale", "near_miss",
        "Write `mb_to_bytes(mb: float) -> float`. "
        "Convert megabytes to bytes. Multiply by 1024*1024. "
        "Call python_exec to implement and test.",
        _v_mb_to_bytes))

    # ── Family 7: missing_secondary_sort ─────────────────────────────────────
    # Bug: sorts by primary key only, no tiebreak → unstable on ties

    def _v_rank_users(ns):
        fn = ns.get("rank_users")
        if not fn:
            return False, "rank_users not defined"
        try:
            users = [
                {"name": "Charlie", "score": 10},
                {"name": "Alice", "score": 10},
                {"name": "Bob", "score": 20},
            ]
            result = fn(users)
            if result[0]["name"] != "Bob":
                return False, f"first should be Bob (score 20), got {result[0]['name']}"
            if result[1]["name"] != "Alice":
                return False, f"second should be Alice (tie, alphabetical), got {result[1]['name']}"
            if result[2]["name"] != "Charlie":
                return False, f"third should be Charlie, got {result[2]['name']}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("rank_users", "secondary_sort", "discovery",
        "Write `rank_users(users: list[dict]) -> list[dict]`. "
        "Each user has 'name' and 'score'. Sort highest score first; "
        "if scores are equal, sort alphabetically by name. "
        "Call python_exec to implement it.",
        _v_rank_users))

    def _v_sort_products(ns):
        fn = ns.get("sort_products")
        if not fn:
            return False, "sort_products not defined"
        try:
            products = [
                {"name": "Widget", "price": 10},
                {"name": "Gadget", "price": 10},
                {"name": "Donut", "price": 5},
            ]
            result = fn(products)
            if result[0]["name"] != "Donut":
                return False, f"cheapest first, got {result[0]['name']}"
            if result[1]["name"] != "Gadget":
                return False, f"tie on price 10: Gadget < Widget alphabetically, got {result[1]['name']}"
            if result[2]["name"] != "Widget":
                return False, f"last should be Widget, got {result[2]['name']}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("sort_products", "secondary_sort", "recurrence",
        "Write `sort_products(products: list[dict]) -> list[dict]`. "
        "Each product has 'name' and 'price'. Sort cheapest first; "
        "for equal prices, sort alphabetically by name. "
        "Call python_exec to implement it.",
        _v_sort_products))

    def _v_sort_events(ns):
        fn = ns.get("sort_events")
        if not fn:
            return False, "sort_events not defined"
        try:
            events = [
                {"id": 3, "date": "2024-01-01"},
                {"id": 1, "date": "2024-01-01"},
                {"id": 2, "date": "2024-01-02"},
            ]
            result = fn(events)
            if result[0]["id"] != 1:
                return False, f"same date tie: id 1 < 3, got id {result[0]['id']}"
            if result[1]["id"] != 3:
                return False, f"id 3 should be second, got {result[1]['id']}"
            if result[2]["id"] != 2:
                return False, f"later date last, got {result[2]['id']}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("sort_events", "secondary_sort", "recurrence",
        "Write `sort_events(events: list[dict]) -> list[dict]`. "
        "Each event has an 'id' and a 'date'. Sort earliest date first; "
        "for equal dates, sort by id ascending. "
        "Call python_exec to implement it.",
        _v_sort_events))

    def _v_sort_transactions(ns):
        fn = ns.get("sort_transactions")
        if not fn:
            return False, "sort_transactions not defined"
        try:
            txns = [
                {"amount": 100, "ts": "2024-01-02"},
                {"amount": 200, "ts": "2024-01-01"},
                {"amount": 100, "ts": "2024-01-01"},
            ]
            result = fn(txns)
            if result[0]["amount"] != 200:
                return False, f"highest amount first: expected 200, got {result[0]['amount']}"
            if result[1]["ts"] != "2024-01-01":
                return False, f"tie on 100: earlier ts first, got {result[1]['ts']}"
            if result[2]["ts"] != "2024-01-02":
                return False, f"last should be ts 2024-01-02, got {result[2]['ts']}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("sort_transactions", "secondary_sort", "recurrence",
        "Write `sort_transactions(txns: list[dict]) -> list[dict]`. "
        "Each transaction has 'amount' and 'ts'. Sort by amount descending; "
        "for equal amounts, sort by timestamp ascending. "
        "Call python_exec to implement it.",
        _v_sort_transactions))

    def _v_rank_leaderboard(ns):
        fn = ns.get("rank_leaderboard")
        if not fn:
            return False, "rank_leaderboard not defined"
        try:
            players = [
                {"name": "Zoe", "score": 50, "time": 30},
                {"name": "Amy", "score": 50, "time": 25},
                {"name": "Max", "score": 80, "time": 20},
            ]
            result = fn(players)
            if result[0]["name"] != "Max":
                return False, f"Max has highest score, got {result[0]['name']}"
            if result[1]["name"] != "Amy":
                return False, f"tie on 50: Amy faster (25 < 30), got {result[1]['name']}"
            if result[2]["name"] != "Zoe":
                return False, f"Zoe last in tie, got {result[2]['name']}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("rank_leaderboard", "secondary_sort", "recurrence",
        "Write `rank_leaderboard(players: list[dict]) -> list[dict]`. "
        "Each player has 'name', 'score', and 'time'. Rank by score descending (higher is better); "
        "for equal scores, rank by time ascending (faster is better). "
        "Call python_exec to implement it.",
        _v_rank_leaderboard))

    # Near-miss: unique primary key — no ties possible
    def _v_sort_unique(ns):
        fn = ns.get("sort_by_id")
        if not fn:
            return False, "sort_by_id not defined"
        try:
            items = [{"id": 3}, {"id": 1}, {"id": 2}]
            result = fn(items)
            if [r["id"] for r in result] != [1, 2, 3]:
                return False, f"sort by unique id failed: {[r['id'] for r in result]}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("sort_by_id", "secondary_sort", "near_miss",
        "Write `sort_by_id(items: list[dict]) -> list[dict]`. "
        "Sort by 'id' ascending. IDs are unique integers — no ties possible. "
        "Call python_exec to implement and test.",
        _v_sort_unique))

    def _v_sort_by_name_only(ns):
        fn = ns.get("sort_by_name")
        if not fn:
            return False, "sort_by_name not defined"
        try:
            items = [{"name": "C"}, {"name": "A"}, {"name": "B"}]
            result = fn(items)
            if [r["name"] for r in result] != ["A", "B", "C"]:
                return False, f"got {[r['name'] for r in result]}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("sort_by_name", "secondary_sort", "near_miss",
        "Write `sort_by_name(items: list[dict]) -> list[dict]`. "
        "Sort by 'name' alphabetically ascending. Order among ties doesn't matter. "
        "Call python_exec to implement and test.",
        _v_sort_by_name_only))

    # ── Family 8: retry_error_classification ─────────────────────────────────
    # Bug: retries 4xx (client errors) instead of only 5xx (server errors)

    def _v_retry(ns):
        fn = ns.get("retry_request")
        if not fn:
            return False, "retry_request not defined"
        try:
            class FakeError(Exception):
                def __init__(self, code):
                    self.status_code = code
                    super().__init__(str(code))

            # 400 must NOT be retried
            calls400 = [0]
            def raises_400():
                calls400[0] += 1
                raise FakeError(400)
            try:
                fn(raises_400, max_retries=3)
            except Exception:
                pass
            if calls400[0] != 1:
                return False, f"400 should not retry: expected 1 call, got {calls400[0]}"

            # 500 SHOULD be retried
            calls500 = [0]
            def raises_500():
                calls500[0] += 1
                if calls500[0] < 3:
                    raise FakeError(500)
                return "ok"
            result = fn(raises_500, max_retries=3)
            if result != "ok":
                return False, f"expected 'ok' after retries, got {result}"
            if calls500[0] != 3:
                return False, f"500 should retry: expected 3 calls, got {calls500[0]}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("retry_request", "retry_classification", "discovery",
        "Write `retry_request(fn, max_retries: int = 3)`. "
        "Our HTTP client sometimes fails. Call fn() and retry up to max_retries times if it throws. "
        "Return the result. "
        "Call python_exec to implement it.",
        _v_retry))

    def _v_retry_type(ns):
        fn = ns.get("retry_on_type")
        if not fn:
            return False, "retry_on_type not defined"
        try:
            calls_value = [0]
            def raises_value():
                calls_value[0] += 1
                raise ValueError("bad")

            calls_runtime = [0]
            def raises_runtime():
                calls_runtime[0] += 1
                if calls_runtime[0] < 2:
                    raise RuntimeError("temporary")
                return "done"

            # Should NOT retry ValueError
            try:
                fn(raises_value, retry_on=(RuntimeError,), max_retries=3)
            except ValueError:
                pass
            if calls_value[0] != 1:
                return False, f"ValueError should not retry, got {calls_value[0]} calls"

            # Should retry RuntimeError
            result = fn(raises_runtime, retry_on=(RuntimeError,), max_retries=3)
            if result != "done":
                return False, f"expected 'done', got {result}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("retry_on_type", "retry_classification", "recurrence",
        "Write `retry_on_type(fn, retry_on: tuple, max_retries: int = 3)`. "
        "Call fn() and retry up to max_retries times. Only retry exceptions listed in retry_on. "
        "Call python_exec to implement it.",
        _v_retry_type))

    def _v_retry_count(ns):
        fn = ns.get("retry_with_count")
        if not fn:
            return False, "retry_with_count not defined"
        try:
            calls = [0]
            def always_fail():
                calls[0] += 1
                raise RuntimeError("fail")

            try:
                fn(always_fail, max_retries=2)
            except RuntimeError:
                pass

            # max_retries=2 means: 1 initial try + 2 retries = 3 total calls
            if calls[0] != 3:
                return False, f"max_retries=2 → 3 total calls expected, got {calls[0]}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("retry_with_count", "retry_classification", "recurrence",
        "Write `retry_with_count(fn, max_retries: int = 3)`. "
        "Call fn() and retry on RuntimeError. max_retries is the number of additional attempts after the first. "
        "Call python_exec to implement it.",
        _v_retry_count))

    def _v_no_retry_4xx(ns):
        fn = ns.get("safe_request")
        if not fn:
            return False, "safe_request not defined"
        try:
            class HttpError(Exception):
                def __init__(self, code):
                    self.status_code = code
                    super().__init__(str(code))

            calls = [0]
            def bad_request():
                calls[0] += 1
                raise HttpError(404)

            try:
                fn(bad_request)
            except Exception:
                pass

            if calls[0] != 1:
                return False, f"404 should not retry, got {calls[0]} calls"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("safe_request", "retry_classification", "recurrence",
        "Write `safe_request(fn, max_retries: int = 3)`. "
        "Our service occasionally has transient server errors. Retry server errors but not client errors. "
        "Exceptions have a .status_code attribute. "
        "Call python_exec to implement it.",
        _v_no_retry_4xx))

    def _v_idempotent_retry(ns):
        fn = ns.get("idempotent_retry")
        if not fn:
            return False, "idempotent_retry not defined"
        try:
            class HttpError(Exception):
                def __init__(self, code):
                    self.status_code = code

            calls_post = [0]
            def post_request():
                calls_post[0] += 1
                raise HttpError(500)

            try:
                fn(post_request, method="POST", max_retries=3)
            except Exception:
                pass

            # POST is not idempotent — should not retry
            if calls_post[0] != 1:
                return False, f"POST should not be retried, got {calls_post[0]} calls"

            calls_get = [0]
            def get_request():
                calls_get[0] += 1
                if calls_get[0] < 2:
                    raise HttpError(500)
                return "data"

            result = fn(get_request, method="GET", max_retries=3)
            if result != "data":
                return False, f"GET should retry, got {result}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("idempotent_retry", "retry_classification", "recurrence",
        "Write `idempotent_retry(fn, method: str = 'GET', max_retries: int = 3)`. "
        "Only retry if method is 'GET' or 'HEAD' (idempotent). "
        "Never retry POST, PUT, or DELETE. "
        "Still only retry on 5xx status codes. "
        "Call python_exec to implement and test.",
        _v_idempotent_retry))

    # Near-miss: retry ALL errors — no classification needed
    def _v_retry_all(ns):
        fn = ns.get("retry_all")
        if not fn:
            return False, "retry_all not defined"
        try:
            calls = [0]
            def unstable():
                calls[0] += 1
                if calls[0] < 3:
                    raise Exception("transient")
                return "ok"
            result = fn(unstable, max_retries=3)
            if result != "ok":
                return False, f"expected 'ok', got {result}"
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("retry_all", "retry_classification", "near_miss",
        "Write `retry_all(fn, max_retries: int = 3)`. "
        "Retry on ANY exception up to max_retries times. No error classification needed. "
        "Return the result on success. "
        "Call python_exec to implement and test.",
        _v_retry_all))

    def _v_try_once(ns):
        fn = ns.get("try_once")
        if not fn:
            return False, "try_once not defined"
        try:
            def ok():
                return 42
            if fn(ok) != 42:
                return False, "should return 42"
            def fail():
                raise ValueError("err")
            try:
                fn(fail)
                return False, "should propagate exception"
            except ValueError:
                pass
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    tasks.append(Task("try_once", "retry_classification", "near_miss",
        "Write `try_once(fn)`. Call fn() exactly once and return the result. "
        "Let any exception propagate. No retry logic needed. "
        "Call python_exec to implement and test.",
        _v_try_once))

    return tasks


# ── Task ordering ─────────────────────────────────────────────────────────────

FAMILY_ORDER = [
    "nested_key",
    "shared_state",
    "api_key",
    "validation_all_errors",
    "off_by_one",
    "unit_scale",
    "secondary_sort",
    "retry_classification",
]


def build_task_order(all_tasks: list[Task]) -> list[Task]:
    """All discoveries first, then interleaved recurrences, then near-misses."""
    discoveries = [t for t in all_tasks if t.kind == "discovery"]
    recurrences = [t for t in all_tasks if t.kind == "recurrence"]
    near_misses = [t for t in all_tasks if t.kind == "near_miss"]

    # Sort discoveries by family order
    fam_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}
    discoveries.sort(key=lambda t: fam_idx.get(t.family, 99))

    # Round-robin interleave recurrences across families
    by_family: dict[str, list[Task]] = {}
    for t in recurrences:
        by_family.setdefault(t.family, []).append(t)

    interleaved: list[Task] = []
    max_recs = max((len(v) for v in by_family.values()), default=0)
    for i in range(max_recs):
        for fam in FAMILY_ORDER:
            fam_tasks = by_family.get(fam, [])
            if i < len(fam_tasks):
                interleaved.append(fam_tasks[i])

    return discoveries + interleaved + near_misses


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(use_brain: bool, temperature: float, max_tasks: int | None = None) -> list[TaskResult]:
    all_tasks = _make_tasks()
    ordered = build_task_order(all_tasks)
    n_families = len(set(t.family for t in all_tasks))

    # Fail fast if no API key is available
    if not _HAS_OPENAI and not _HAS_ANTHROPIC:
        print("ERROR: No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        sys.exit(1)

    print("\n" + "═" * 72)
    print("  eval_dev_learning — Cold-start learning benchmark")
    mode = "brain-cold" if use_brain else "no-brain"
    print(f"  {len(ordered)} tasks | {n_families} families | mode={mode} | temp={temperature}")
    print(f"  provider={_PROVIDER} | agent={_MODEL} | judge={_JUDGE_MODEL}")
    print("═" * 72)

    embedder = build_embedder()
    brain: BrainAgent | None = None
    if use_brain:
        brain_cfg = BrainConfig(
            provider      = _PROVIDER,
            judge_model   = _JUDGE_MODEL,   # gpt-4o: better verbatim-quote compliance
            extract_model = _JUDGE_MODEL,   # gpt-4o: better structured JSON extraction
        )
        brain = BrainAgent(embedder, k=3, threshold=0.50, config=brain_cfg)
        # Cold start: NO seed_motifs() call

    agent = _make_tool_agent(temperature=temperature, max_turns=5)
    if brain:
        agent.monitor = brain

    # Track which families had motifs extracted (from stdout capture)
    family_motifs_extracted: dict[str, bool] = {}

    results: list[TaskResult] = []
    for i, task in enumerate(ordered):
        if max_tasks is not None and i >= max_tasks:
            break
        if brain:
            brain.set_task(i, task=task.prompt[:300])
            brain.reset()

        t0 = time.time()
        try:
            trace, tokens = agent(task.prompt)
        except Exception as e:
            trace = f"[agent error: {e}]"
            tokens = 0
        elapsed = time.time() - t0

        fired = brain.last_fire is not None if brain else False
        fire_detail = dict(brain.last_fire) if fired and brain else None

        code = _extract_last_code(trace)
        passed, detail = task.run_verify(code)

        motif_learned = False
        if brain:
            label = 1 if passed else 0
            reason = detail[:200] if not passed else ""
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                brain.store(trace, label, reason)
            store_out = buf.getvalue()
            if "[BRAIN MOTIF LEARNED]" in store_out:
                motif_learned = True
                family_motifs_extracted[task.family] = True
                print(f"  [MOTIF EXTRACTED] {store_out.strip()[:120]}")

        status = "PASS" if passed else "FAIL"
        fire_tag = " [FIRED]" if fired else ""
        detail_s = detail[:80] if detail else "ok"
        print(f"[{i+1:02d}/{len(ordered)}] {task.name:<38} {task.family:<22} | {status}{fire_tag} | {elapsed:.1f}s | {detail_s}")

        results.append(TaskResult(
            task=task, passed=passed, fired=fired, detail=detail,
            tokens=tokens, time_s=elapsed, motif_learned=motif_learned,
            fire_detail=fire_detail,
        ))
        time.sleep(0.5)   # small inter-task pause to stay inside rate limits

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_and_print_metrics(results: list[TaskResult]) -> dict:
    by_kind: dict[str, list[TaskResult]] = {"discovery": [], "recurrence": [], "near_miss": []}
    for r in results:
        by_kind[r.task.kind].append(r)

    discoveries = by_kind["discovery"]
    recurrences = by_kind["recurrence"]
    near_misses = by_kind["near_miss"]

    # Families where discovery failed
    disc_failed_families = {r.task.family for r in discoveries if not r.passed}
    # Families where a motif was actually extracted
    motif_extracted_families = {r.task.family for r in results if r.motif_learned}
    # "Learnable" = discovery failed AND motif was extracted
    learnable_families = disc_failed_families & motif_extracted_families

    n_disc_fail = len(disc_failed_families)
    first_fail_rate = n_disc_fail / max(len(discoveries), 1)

    # motif extraction rate = families with motif / families with discovery failure
    motif_extr_rate = len(motif_extracted_families) / max(len(disc_failed_families), 1) \
        if disc_failed_families else 0.0

    # repeat_prevention_rate: recurrence tasks in LEARNABLE families where brain fired AND passed
    learnable_recs = [r for r in recurrences if r.task.family in learnable_families]
    prevented = sum(1 for r in learnable_recs if r.fired and r.passed)
    repeat_prev_rate = prevented / max(len(learnable_recs), 1)

    # false_positive_rate: near-miss where brain fired
    nm_fp = sum(1 for r in near_misses if r.fired)
    fp_rate = nm_fp / max(len(near_misses), 1)

    fires_on_pass = sum(1 for r in recurrences if r.fired and r.passed)
    missed = sum(1 for r in learnable_recs if not r.fired and not r.passed)
    total_tokens = sum(r.tokens for r in results)
    total_time = sum(r.time_s for r in results)

    print("\n" + "═" * 72)
    print("  PER-FAMILY TABLE")
    print(f"  {'Family':<24} {'DiscFail':<10} {'MotifLrn':<10} {'RecFire':<10} {'RecPass':<10} {'NM-FP'}")
    print("  " + "-" * 66)
    for fam in FAMILY_ORDER:
        disc = [r for r in discoveries if r.task.family == fam]
        rec = [r for r in recurrences if r.task.family == fam]
        nm = [r for r in near_misses if r.task.family == fam]
        df = sum(1 for r in disc if not r.passed)
        ml = fam in motif_extracted_families
        rf = sum(1 for r in rec if r.fired)
        rp = sum(1 for r in rec if r.passed)
        nf = sum(1 for r in nm if r.fired)
        print(f"  {fam:<24} {df}/{len(disc):<9} {'YES' if ml else 'no':<10} {rf}/{len(rec):<9} {rp}/{len(rec):<9} {nf}/{len(nm)}")

    print("\n  OVERALL METRICS")
    print(f"  total tasks              : {len(results)}")
    print(f"  total pass rate          : {sum(1 for r in results if r.passed)/len(results):.1%}")
    print(f"  first_fail_rate          : {first_fail_rate:.1%}  ({n_disc_fail}/{len(discoveries)} discovery tasks failed)")
    print(f"  learnable families       : {len(learnable_families)} (disc failed + motif extracted)")
    print(f"  motif_extraction_rate    : {motif_extr_rate:.1%}  ({len(motif_extracted_families)}/{len(disc_failed_families)} failed families got motif)")
    print(f"  repeat_prevention_rate   : {repeat_prev_rate:.1%}  ({prevented}/{len(learnable_recs)} learnable recurrences — brain fired+passed)")
    print(f"  false_positive_rate      : {fp_rate:.1%}  ({nm_fp}/{len(near_misses)} near-miss tasks — brain fired incorrectly)")
    print(f"  fires_on_pass_count      : {fires_on_pass}")
    print(f"  missed_repeat_failures   : {missed}")
    print(f"  total_tokens             : {total_tokens:,}")
    print(f"  tokens_per_task          : {total_tokens/len(results):.0f}")
    print(f"  total_time               : {total_time:.0f}s")

    # Print an example of a learned motif (if any)
    motif_fires = [r for r in results if r.fired and r.fire_detail]
    if motif_fires:
        print("\n  EXAMPLE FIRE:")
        ex = motif_fires[0]
        print(f"    Task: {ex.task.name} ({ex.task.family})")
        if ex.fire_detail:
            evs = ex.fire_detail.get("evidence", [])
            if evs:
                print(f"    Evidence: {evs[0]}")

    # Find a missed failure example
    missed_ex = next((r for r in learnable_recs if not r.fired and not r.passed), None)
    if missed_ex:
        print(f"\n  EXAMPLE MISSED FAILURE:")
        print(f"    Task: {missed_ex.task.name} — {missed_ex.detail[:80]}")

    # Find a near-miss that correctly did not fire
    nm_correct = next((r for r in near_misses if not r.fired), None)
    if nm_correct:
        print(f"\n  NEAR-MISS (correctly no fire):")
        print(f"    Task: {nm_correct.task.name} — {nm_correct.task.family}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent_model": _MODEL,
        "judge_model": _JUDGE_MODEL,
        "mode": "brain-cold",
        "temperature": "configured",
        "n_tasks": len(results),
        "metrics": {
            "total_pass_rate": round(sum(1 for r in results if r.passed) / len(results), 4),
            "first_fail_rate": round(first_fail_rate, 4),
            "learnable_families": len(learnable_families),
            "motif_extraction_rate": round(motif_extr_rate, 4),
            "repeat_prevention_rate": round(repeat_prev_rate, 4),
            "false_positive_rate": round(fp_rate, 4),
            "fires_on_pass_count": fires_on_pass,
            "missed_repeat_failures": missed,
            "total_tokens": total_tokens,
            "tokens_per_task": round(total_tokens / len(results), 1),
            "total_time_s": round(total_time, 1),
        },
        "per_family": {
            fam: {
                "disc_fail": sum(1 for r in discoveries if r.task.family == fam and not r.passed),
                "motif_learned": fam in motif_extracted_families,
                "rec_fired": sum(1 for r in recurrences if r.task.family == fam and r.fired),
                "rec_pass": sum(1 for r in recurrences if r.task.family == fam and r.passed),
                "rec_total": sum(1 for r in recurrences if r.task.family == fam),
                "nm_fp": sum(1 for r in near_misses if r.task.family == fam and r.fired),
                "nm_total": sum(1 for r in near_misses if r.task.family == fam),
            }
            for fam in FAMILY_ORDER
        },
        "tasks": [
            {
                "name": r.task.name,
                "family": r.task.family,
                "kind": r.task.kind,
                "passed": r.passed,
                "fired": r.fired,
                "motif_learned": r.motif_learned,
                "detail": r.detail[:200],
                "tokens": r.tokens,
                "time_s": round(r.time_s, 1),
            }
            for r in results
        ],
    }
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start motif learning benchmark for developer tasks")
    parser.add_argument("--no-brain", action="store_true", help="Run without BrainAgent (baseline)")
    parser.add_argument("--brain-cold", action="store_true", help="Run with cold-start BrainAgent (default)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Agent temperature (default 0.3)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Stop after N tasks (for debugging)")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                        help="LLM provider override (default: auto-detect from env)")
    args = parser.parse_args()

    # Allow CLI to override auto-detected provider
    if args.provider:
        global _PROVIDER, _MODEL, _JUDGE_MODEL
        _PROVIDER    = args.provider
        _MODEL       = "gpt-4o-mini" if _PROVIDER == "openai" else "claude-haiku-4-5-20251001"
        _JUDGE_MODEL = "gpt-4o"      if _PROVIDER == "openai" else "claude-haiku-4-5-20251001"

    use_brain = not args.no_brain
    temperature = args.temperature

    results = run_benchmark(use_brain=use_brain, temperature=temperature, max_tasks=args.max_tasks)
    summary = compute_and_print_metrics(results)

    out_path = OUT / "eval_dev_learning.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("═" * 72)


if __name__ == "__main__":
    main()
