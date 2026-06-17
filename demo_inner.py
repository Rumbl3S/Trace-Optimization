"""demo_inner.py — show the full inner loop: attempt → critique → retry.

Prints:
  1. First attempt (full trace)
  2. P(fail) and whether intervention fires
  3. WHAT NOT TO DO (quoted bad step, first sentence of retry)
  4. The improved retry attempt
  5. P(fail) after retry + pass/fail verdict
"""
from __future__ import annotations
import textwrap
from dotenv import load_dotenv
load_dotenv()

from agents   import haiku, _build_openai
from pipeline import Forecaster, decompose, attempt, self_judge, _RETRY, _text

embedder = _build_openai()

SEP  = "─" * 72
BOLD = "\033[1m"; DIM = "\033[2m"; RED = "\033[31m"; GRN = "\033[32m"
YEL  = "\033[33m"; CYN = "\033[36m"; MAG = "\033[35m"; RST = "\033[0m"

def wrap(text: str, indent: int = 2, width: int = 78) -> str:
    prefix = " " * indent
    return "\n".join(
        textwrap.fill(ln, width, initial_indent=prefix, subsequent_indent=prefix)
        if ln.strip() else ""
        for ln in text.splitlines()
    )

def split_critique(retry_trace: str) -> tuple[str, str]:
    """First sentence = quoted bad step + what not to do; rest = new attempt."""
    idx = retry_trace.find(". ")
    if idx != -1:
        return retry_trace[:idx+1].strip(), retry_trace[idx+2:].strip()
    return "", retry_trace.strip()

# ── task (hard enough to generate real failures) ──────────────────────────────
TASK = (
    "Write a Python function `flatten(lst)` that recursively flattens a "
    "nested list of arbitrary depth into a single flat list. "
    "Then demonstrate it on [[1,[2,3]],[4,[5,[6]]]] and verify the result "
    "equals [1,2,3,4,5,6]."
)

# ── pre-seed: 4 fails + 4 passes so adaptive_threshold kicks in ───────────────
SEED_FAILS = [
    "I'll implement flatten. def flatten(lst): result=[]\n"
    "for item in lst: result.append(item)\nreturn result\n"
    "ANSWER: flatten([[1,[2]],[3]]) = [[1,[2]],[3]] — it does not recurse.",

    "To flatten I'll use: def flatten(lst): return [x for x in lst]\n"
    "This just copies the top level. ANSWER: Returns outer list unchanged.",

    "def flatten(lst):\n  if isinstance(lst, list): return lst\n"
    "  return [lst]\nANSWER: Returns the list as-is without recursing.",

    "flatten([[1,2],[3,[4]]]) — I'll iterate and extend.\n"
    "result=[]; for item in lst: result.extend(item)\n"
    "ANSWER: Crashes on integers because int is not iterable.",
]
SEED_PASS = [
    "def flatten(lst):\n  result=[]\n  for item in lst:\n"
    "    if isinstance(item,list): result.extend(flatten(item))\n"
    "    else: result.append(item)\n  return result\n"
    "ANSWER: Correct recursive implementation.",

    "Merge sort splits list in half, recursively sorts, then merges.\n"
    "ANSWER: Time complexity O(n log n).",

    "Binary search: lo=0, hi=n-1, mid=(lo+hi)//2. Compare arr[mid] to target.\n"
    "ANSWER: O(log n) search.",

    "To count words: split on whitespace, lowercase, strip punctuation.\n"
    "ANSWER: Use collections.Counter on processed tokens.",
]

print(f"\n{SEP}")
print(f"{BOLD}Seeding forecaster…{RST}")
fc = Forecaster(embedder, k=4, pca_dim=0)
fc.fit(SEED_FAILS + SEED_PASS, [0,0,0,0,1,1,1,1])
print(f"  {len(fc._labels)} traces  |  fail_rate={1-sum(fc._labels)/len(fc._labels):.0%}"
      f"  |  adaptive_threshold={fc.adaptive_threshold:.2f}")

verifier = self_judge(haiku)

print(f"\n{SEP}")
print(f"{BOLD}TASK:{RST} {TASK}")
print(SEP)

sub_qs = decompose(TASK, haiku, cap=5)
print(f"\nDecomposed into {len(sub_qs)} sub-questions:")
for i, q in enumerate(sub_qs, 1):
    print(f"  {i}. {q}")

# context = full task description (so agent knows what it's implementing)
CTX = f"Task context:\n{TASK}"

for i, q in enumerate(sub_qs, 1):
    print(f"\n\n{'═'*72}")
    print(f"{BOLD} [{i}/{len(sub_qs)}]  {q}{RST}")
    print(f"{'═'*72}")

    trace = attempt(q, CTX, haiku)
    p_fail = fc.predict_fail(trace)
    t      = fc.adaptive_threshold

    print(f"\n{DIM}{'─'*30} FIRST ATTEMPT {'─'*27}{RST}")
    print(wrap(trace))

    flag = f"{YEL}▲ INTERVENTION  (P(fail)={p_fail:.3f} ≥ threshold={t:.3f}){RST}"
    skip = f"{DIM}  no intervention (P(fail)={p_fail:.3f} < threshold={t:.3f}){RST}"
    print(f"\n  {flag if p_fail >= t else skip}")

    if p_fail >= t:
        retry_trace = _text(haiku(_RETRY.format(ctx=CTX, q=q, prev=trace[-2000:])))
        critique, new_attempt = split_critique(retry_trace)

        print(f"\n  {RED}{BOLD}⚑  WHAT NOT TO DO  (quoted from failed attempt):{RST}")
        print(wrap(critique, indent=4))

        print(f"\n{DIM}{'─'*30} RETRY ATTEMPT {'─'*27}{RST}")
        print(wrap(new_attempt))

        p_fail2 = fc.predict_fail(retry_trace)
        label   = int(verifier(q, retry_trace) >= 0.5)
        verdict = f"{GRN}✓ RECOVERED{RST}" if label == 1 else f"{RED}✗ still failing{RST}"
        print(f"\n  {CYN}P(fail) after retry: {p_fail2:.3f}{RST}  →  {verdict}")
        fc.add(retry_trace, label)
    else:
        label = int(verifier(q, trace) >= 0.5)
        verdict = f"{GRN}✓ pass{RST}" if label == 1 else f"{RED}✗ fail{RST}"
        print(f"  {verdict}")
        fc.add(trace, label)

print(f"\n\n{SEP}")
print(f"{BOLD}Session complete.{RST}  Store: {len(fc._labels)} traces  "
      f"|  adaptive_threshold now: {fc.adaptive_threshold:.2f}")
