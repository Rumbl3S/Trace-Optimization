"""
trace_use live demo — run this directly:

    python3 demo.py
    python3 demo.py "your own task here"
"""
import sys
sys.path.insert(0, ".")

from agents import haiku, opus, tool_agent, _build_openai
from pipeline import run_task, Forecaster, self_judge

embedder = _build_openai()
agent    = tool_agent(["wikipedia_search", "calculator", "python_exec"])
fc       = Forecaster(embedder, k=5)

task = (sys.argv[1] if len(sys.argv) > 1 else
        "Who directed The Dark Knight, when was it released, "
        "what was its box office gross, and what is that number divided by its budget?")

# run 1 — seed the store
run_task(
    task="What is the population of Tokyo, and what percentage of Japan's total population does that represent?",
    agent=agent,
    verifier=self_judge(judge_agent=opus),
    forecaster=fc,
    display=True,
)

# run 2 — the main task, now with forecaster signal
result = run_task(
    task=task,
    agent=agent,
    verifier=self_judge(judge_agent=opus),
    forecaster=fc,
    display=True,
)

# explain any flagged components
flagged = [c for c in result.components if c.p_fail and c.p_fail >= 0.4]
if flagged:
    print("\n── Why these were flagged ──")
    for c in flagged:
        print(f"\n  [{c.question[:60]}]  P(fail)={c.p_fail:.2f}")
        for n in fc.explain(c.trace, k=2):
            print(f"    sim={n['similarity']:.3f}  [{n['outcome']}]  {n['excerpt'][:80]}...")
