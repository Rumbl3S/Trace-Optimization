"""trace_use — forecast agent failure from execution traces.

After `pip install -e /path/to/this-repo`, import the modules directly:

    from brain    import BrainAgent
    from agents   import build_embedder, tool_agent, haiku, opus
    from pipeline import Forecaster, run_task, self_judge, code_judge

Quickstart — wrap any tool agent with the brain (3 lines):

    from brain  import BrainAgent
    from agents import build_embedder, tool_agent

    brain         = BrainAgent(build_embedder(), threshold=0.30)
    agent         = tool_agent(["python_exec"], model="claude-haiku-4-5-20251001")
    agent.monitor = brain

    for i, (prompt, check_fn) in enumerate(my_tasks):
        brain.set_task(i, probe_fn=my_probe)
        brain.reset()
        trace, tokens = agent(prompt)
        passed = check_fn(trace)
        brain.store(trace, int(passed))

Forecaster-only (offline, no probes):

    from pipeline import Forecaster, run_task, self_judge
    from agents   import haiku, opus, build_embedder

    fc     = Forecaster(build_embedder())
    result = run_task(
        task       = "What is the GDP of France?",
        agent      = haiku,
        verifier   = self_judge(judge_agent=opus),
        forecaster = fc,
        retry      = True,
    )
    print(result.summary())
"""
