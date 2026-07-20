"""Self-contained agent + embedder for the evals — no external project deps.

`haiku` / `opus` are temperature-0 Anthropic agents (`prompt -> (text, tokens)`);
`_build_openai()` returns an OpenAI `text-embedding-3-small` embedder
(`texts -> L2-normalised vectors`). Clients are created lazily on first use, so importing
this module never needs keys — only running an eval does. Keys load from the environment
or a nearby `.env`.

`BackgroundMonitor` runs in a shadow thread alongside the main agent. It receives
partial trace chunks via `push()`, embeds them asynchronously, and sets a bail flag
the moment the trajectory resembles past failures — before the main agent finishes.

`streaming_agent(model)` uses the Anthropic streaming API and checks the bail flag
after every chunk, enabling mid-generation early exit for plain text agents.

`monitored_agent(base_agent, monitor)` attaches a BackgroundMonitor to any agent.
"""
from __future__ import annotations

import os
import threading
import time

_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
_SONNET_MODEL = "claude-sonnet-4-6"
_OPUS_MODEL   = "claude-opus-4-8"
_EMBED_MODEL  = "text-embedding-3-small"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        for p in (".env", os.path.join("..", ".env"), os.path.join("..", "..", ".env")):
            load_dotenv(p)
    except Exception:                                       # noqa: BLE001
        pass


_load_env()
_client = None
_openai_chat_client = None


def _anthropic_call(model: str, prompt: str, max_tokens: int = 512):
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    # temperature=0 pins sampling where the model allows it; Opus 4 rejects the
    # parameter outright (400), so omit it there. (Haiku 4.5 accepts it.)
    supports_temp = "opus-4" not in model
    kwargs = {"temperature": 0} if supports_temp else {}
    last = None
    for attempt in range(5):
        try:
            r = _client.messages.create(model=model, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}],
                                        **kwargs)
            return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens
        except Exception as e:                              # noqa: BLE001
            last = e
            code = getattr(getattr(e, 'response', None), 'status_code', None) or \
                   getattr(e, 'status_code', None)
            # Retry on rate-limit (429), overload (529), and any other 5xx server error.
            # Give up immediately on 4xx client errors (bad request, auth, etc.).
            if code is not None and (code == 429 or code >= 500):
                wait = min(4 ** attempt, 60)   # 1s, 4s, 16s, 60s, 60s
                time.sleep(wait)
            else:
                break
    raise last


def _openai_call(model: str, prompt: str, max_tokens: int = 512):
    """OpenAI text-only call (no tools). Returns (text, total_tokens)."""
    global _openai_chat_client
    if _openai_chat_client is None:
        from openai import OpenAI
        _openai_chat_client = OpenAI()
    last = None
    for attempt in range(5):
        try:
            r = _openai_chat_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text   = r.choices[0].message.content or ""
            tokens = r.usage.prompt_tokens + r.usage.completion_tokens
            return text, tokens
        except Exception as e:                              # noqa: BLE001
            last = e
            code = getattr(getattr(e, 'response', None), 'status_code', None) or \
                   getattr(e, 'status_code', None)
            if code is not None and (code == 429 or code >= 500):
                wait = min(4 ** attempt, 60)
                time.sleep(wait)
            else:
                break
    raise last


def _llm_call(provider: str, model: str, prompt: str, max_tokens: int = 512):
    """Route to Anthropic or OpenAI based on provider string."""
    if provider == "openai":
        return _openai_call(model, prompt, max_tokens)
    return _anthropic_call(model, prompt, max_tokens)


def haiku(prompt: str):
    """Fast, cheap temperature-0 agent. Returns (text, total_tokens)."""
    return _anthropic_call(_HAIKU_MODEL, prompt)


def opus(prompt: str):
    """High-accuracy temperature-0 agent. Returns (text, total_tokens)."""
    return _anthropic_call(_OPUS_MODEL, prompt, max_tokens=1024)


class BackgroundMonitor:
    """Shadow thread that watches the accumulating trace and fires a bail flag
    the moment the trajectory resembles past failures.

    Usage::

        monitor = BackgroundMonitor(forecaster)
        agent   = monitored_agent(haiku, monitor)

        with monitor:                          # starts the watcher thread
            result = run_task(task, agent=agent, monitor=monitor, ...)
            # if the monitor fires mid-run, the agent exits early and the
            # pipeline fires a self-critique retry automatically

    The monitor resets between components (via reset()) so each sub-question
    gets a clean slate. It never blocks the main agent — the embedding call
    happens in the background thread, not on the critical path.
    """

    def __init__(self, forecaster, check_interval: float = 1.0, min_chars: int = 300):
        """
        Args:
            forecaster:      The shared Forecaster instance.
            check_interval:  Seconds between kNN checks (each is an embedding API call).
            min_chars:       Minimum trace length before the first check fires — avoids
                             flagging on fragments too short to embed meaningfully.
        """
        self.forecaster     = forecaster
        self.check_interval = check_interval
        self.min_chars      = min_chars

        self._buffer:     str   = ""
        self._lock               = threading.Lock()
        self._bail_event         = threading.Event()
        self._done_event         = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_p_fail: float | None = None   # last score computed, for display

    # ── main-thread interface ─────────────────────────────────────────────────

    def push(self, text: str) -> None:
        """Append a chunk of trace text. Called from the main agent thread."""
        with self._lock:
            self._buffer += text

    @property
    def should_bail(self) -> bool:
        """True when the monitor has decided this trajectory is failing."""
        return self._bail_event.is_set()

    def reset(self) -> None:
        """Clear buffer and bail flag before a new component attempt."""
        with self._lock:
            self._buffer = ""
        self._bail_event.clear()
        self.last_p_fail = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "BackgroundMonitor":
        self._done_event.clear()
        self._bail_event.clear()
        with self._lock:
            self._buffer = ""
        self._thread = threading.Thread(target=self._loop, daemon=True, name="trace-monitor")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._done_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def __enter__(self) -> "BackgroundMonitor":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._done_event.is_set():
            time.sleep(self.check_interval)

            with self._lock:
                trace = self._buffer

            # Need enough text to embed meaningfully, and enough stored history
            # for the kNN to have signal.
            if len(trace) < self.min_chars:
                continue
            if len(self.forecaster._vecs) < 2:
                continue

            p_fail = self.forecaster.predict_fail(trace)
            self.last_p_fail = p_fail

            if self.forecaster.should_intervene(trace):
                self._bail_event.set()
                return   # job done — bail flag is set, stop looping


def tool_agent(tools: list | None = None, max_turns: int = 4,
               model: str | None = None, max_tokens: int = 4096):
    """ReAct tool-calling agent. Returns a callable (prompt -> (trace, tokens)).

    `tools` is a subset of ['calculator', 'python_exec', 'wikipedia_search'].
    Defaults to all three. The full trajectory — reasoning, tool calls, and
    tool results — is returned as the trace so the Forecaster can embed it.

    `model` overrides the default (_HAIKU_MODEL). Pass _SONNET_MODEL or any
    Anthropic model ID to trade cost for capability on harder tasks.
    """
    from .tools import TOOL_DEFINITIONS, dispatch

    selected = tools or ["calculator", "python_exec", "wikipedia_search"]
    tool_defs = [t for t in TOOL_DEFINITIONS if t["name"] in selected]
    _model = model or _HAIKU_MODEL

    def agent(prompt: str):
        global _client
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic()

        messages    = [{"role": "user", "content": prompt}]
        trace_parts: list[str] = []
        total_tokens = 0
        monitor: BackgroundMonitor | None = getattr(agent, 'monitor', None)

        for _ in range(max_turns):
            # 4096 so the model can emit a complete implementation inside a single
            # tool call — at 1024 a large code block is truncated mid-JSON, the
            # tool input arrives empty, nothing runs, and the component fails.
            r = _client.messages.create(
                model=_model, max_tokens=max_tokens, temperature=0,
                tools=tool_defs if tool_defs else [],
                messages=messages,
            )
            total_tokens += r.usage.input_tokens + r.usage.output_tokens

            # Collect text and tool calls; push every text block to the monitor
            # so the background thread can embed the accumulating trace.
            tool_results = []
            for block in r.content:
                if block.type == "text":
                    trace_parts.append(block.text)
                    if monitor:
                        monitor.push(block.text)
                elif block.type == "tool_use":
                    # Pre-execution hook: brain checks reasoning trajectory and
                    # proposed code BEFORE the tool runs. If it returns a string,
                    # that string replaces execution — the tool is not called and
                    # the agent sees the warning as the result.
                    pre_result = None
                    if monitor and hasattr(monitor, "before_tool_call"):
                        pre_result = monitor.before_tool_call(block.name, block.input)

                    if pre_result is not None:
                        result = pre_result   # brain blocked execution
                    else:
                        result = dispatch(block.name, block.input)

                    # Post-execution hook: brain sees code + result, can inject
                    # a reactive warning (e.g. execution error matches stored failure).
                    if monitor and hasattr(monitor, "on_tool_call"):
                        _modified = monitor.on_tool_call(block.name, block.input, result)
                        if _modified is not None:
                            result = _modified

                    # json.dumps guarantees parseable extraction (no brace-count
                    # issues when code itself contains { } characters).
                    import json as _json
                    chunk  = f"[tool:{block.name}({_json.dumps(block.input)})] → {result[:500]}"
                    trace_parts.append(chunk)
                    if monitor:
                        monitor.push(chunk)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:4000],
                    })

            if r.stop_reason == "end_turn" or not tool_results:
                break

            # After each tool turn: pulse the monitor for immediate check.
            # This catches failure patterns right after a tool result arrives
            # rather than waiting for the timed background loop.
            if monitor and hasattr(monitor, 'pulse'):
                monitor.pulse()

            # Between-turn checkpoint: if the background monitor has already
            # decided this trajectory is failing, exit now rather than burning
            # another LLM turn. The outer pipeline sees [EARLY_EXIT] and fires
            # the self-critique retry.
            if monitor and monitor.should_bail:
                trace_parts.append(
                    f"[EARLY_EXIT: monitor P(fail)={monitor.last_p_fail:.2f} "
                    "— trajectory matches past failures]"
                )
                break

            messages.append({"role": "assistant", "content": r.content})
            messages.append({"role": "user", "content": tool_results})

        return "\n".join(trace_parts), total_tokens

    agent.monitor = None   # attached by monitored_agent() or run_task()
    return agent


def streaming_agent(model: str | None = None):
    """Text agent using the Anthropic streaming API.

    Behaves identically to haiku/opus in normal use. When a BackgroundMonitor
    is attached (via monitored_agent), it checks the bail flag after every
    streamed chunk and closes the connection immediately on detection —
    enabling mid-generation early exit without waiting for the full response.

    Example::

        monitor = BackgroundMonitor(forecaster)
        agent   = monitored_agent(streaming_agent(), monitor)
    """
    _model = model or _HAIKU_MODEL
    supports_temp = "opus-4" not in _model

    def agent(prompt: str):
        global _client
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic()

        monitor: BackgroundMonitor | None = getattr(agent, 'monitor', None)
        kwargs = {"temperature": 0} if supports_temp else {}

        accumulated  = ""
        total_tokens = 0
        bailed       = False

        with _client.messages.stream(
            model=_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            for chunk in stream.text_stream:
                accumulated += chunk
                if monitor:
                    monitor.push(chunk)
                    if monitor.should_bail:
                        bailed = True
                        break   # closes the stream, stops billing for remaining tokens

        if bailed:
            accumulated += (
                f"\n[EARLY_EXIT: monitor P(fail)={monitor.last_p_fail:.2f} "
                "— trajectory matches past failures]"
            )
        else:
            try:
                msg = stream.get_final_message()
                total_tokens = msg.usage.input_tokens + msg.usage.output_tokens
            except Exception:
                pass

        return accumulated, total_tokens

    agent.monitor = None
    return agent


def monitored_agent(base_agent, monitor: BackgroundMonitor):
    """Attach a BackgroundMonitor to any agent.

    - For tool_agent and streaming_agent: sets agent.monitor directly.
    - For plain haiku / opus callables: wraps them in a streaming_agent so
      the monitor can trigger mid-generation bail.

    The returned agent is the same object (or a new streaming wrapper) with
    monitor wired in. Pass it to run_task as the agent argument::

        monitor = BackgroundMonitor(forecaster)
        agent   = monitored_agent(haiku, monitor)

        with monitor:
            result = run_task(task, agent=agent, monitor=monitor, forecaster=fc)
    """
    if hasattr(base_agent, 'monitor'):
        # Already a monitor-aware agent (tool_agent or streaming_agent)
        base_agent.monitor = monitor
        return base_agent

    # Plain callable (haiku, opus, or a CoT lambda) — wrap in streaming_agent
    # so the monitor can bail mid-generation.
    name = getattr(base_agent, '__name__', '')
    model_map = {'haiku': _HAIKU_MODEL, 'opus': _OPUS_MODEL}
    _model = model_map.get(name, _HAIKU_MODEL)

    wrapped = streaming_agent(_model)
    wrapped.monitor = monitor

    # Preserve the original callable as a fallback docstring hint
    wrapped.__wrapped__ = base_agent
    return wrapped


def _build_local_embedder():
    """sentence-transformers local embedder — free, no API key, ~10ms/chunk on CPU.

    Uses all-MiniLM-L6-v2 (80MB, 384-dim). Downloaded once on first call.
    Returns L2-normalised float32 vectors, same interface as _build_openai().
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        texts = [t[:8000] for t in texts]
        vecs = _model.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")
    return embed


def _build_openai():
    """OpenAI text-embedding-3-small embedder. Requires OPENAI_API_KEY.

    Returns L2-normalised float32 vectors. Use build_embedder() instead to
    prefer the free local embedder and fall back here only if needed.
    """
    from openai import OpenAI
    import numpy as np
    client = OpenAI()

    def embed(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), 256):
            batch = [t[:8000] for t in texts[i:i + 256]]
            r = client.embeddings.create(model=_EMBED_MODEL, input=batch)
            out.extend(d.embedding for d in r.data)
        v = np.asarray(out, dtype="float32")
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return embed


def build_embedder():
    """Return the best available embedder.

    Prefers OpenAI text-embedding-3-small when OPENAI_API_KEY is set (avoids
    loading a local model and its GPU-cache disk writes on macOS).
    Falls back to sentence-transformers (local, no API key needed) otherwise.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return _build_openai()
    try:
        return _build_local_embedder()
    except ImportError:
        return _build_openai()
