"""eval/eval_fires.py — 30-task brain fire-rate evaluation.

Mixed domains: algorithms, math puzzles, geography, history, science, logic.
For each task the harness runs the model once; if it fails it feeds back exactly
what went wrong and runs once more (the retry result becomes the final score, but
the FIRST-attempt trace + label is what gets stored in the brain).

Task types
----------
"code"  — tool_agent with python_exec; code extracted and executed locally
"text"  — plain haiku call; response string checked directly

Run:
    python eval/eval_fires.py
"""
from __future__ import annotations

import ast as _ast
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agents import build_embedder, haiku, tool_agent
from brain import BrainAgent
from eval.viz_brain import BrainViz

OUT       = _ROOT / "eval" / "results"
OUT.mkdir(exist_ok=True)

HAIKU     = "claude-haiku-4-5-20251001"
MAX_TURNS = 8
# Tells haiku its function must be self-contained (no eval/compile internals).
_NC       = "Your implementation must be self-contained — do not call Python's built-in eval() or compile() inside the function."


# ─── code helpers ─────────────────────────────────────────────────────────────

def _run(code: str, ns: dict) -> str | None:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<fires>", "exec"), ns)
        return None
    except Exception as e:
        return str(e)


def _check_code(code: str | None, check_fn) -> tuple[bool, str]:
    if not code:
        return False, "no code was executed — write the function and call python_exec to run it"
    ns: dict = {}
    err = _run(code, ns)
    if err:
        return False, f"compile/runtime error: {err[:120]}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ok = bool(check_fn(ns))
        return ok, "" if ok else "implementation failed tests"
    except Exception as e:
        return False, f"test error: {e}"


def _check_text(response: str, check_fn) -> tuple[bool, str]:
    try:
        ok = bool(check_fn(response))
        return ok, "" if ok else "answer incorrect"
    except Exception as e:
        return False, f"check error: {e}"


def _extract_code(trace: str, want_fn: str | None = None) -> str | None:
    candidates: list[str] = []

    # python_exec({...}) tool calls — handle both JSON (new) and repr (old) format
    import json as _json
    search = 0
    while True:
        start = trace.find("python_exec({", search)
        if start == -1:
            break
        brace = start + len("python_exec(")
        depth = i = brace
        while i < len(trace):
            if trace[i] == '{':   depth += 1
            elif trace[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        raw = trace[brace:i + 1]
        search = i + 1
        code = None
        for loader in (_json.loads, _ast.literal_eval):
            try:
                d = loader(raw)
                if isinstance(d, dict) and "code" in d:
                    code = d["code"]
                    break
            except Exception:
                pass
        if code:
            candidates.append(code)

    # Markdown fenced blocks (with or without language tag)
    for m in re.finditer(r"```(?:python)?\s*(.*?)```", trace, re.DOTALL):
        candidates.append(m.group(1).strip())

    # Bare def blocks — haiku sometimes writes code without fences
    if want_fn:
        for m in re.finditer(
            rf"(def {re.escape(want_fn)}\s*\(.*?)(?=\ndef |\nclass |\Z)",
            trace,
            re.DOTALL,
        ):
            candidates.append(m.group(1).strip())

    if not candidates:
        return None

    def score(c: str) -> int:
        try:
            compile(c, "<s>", "exec")
        except SyntaxError:
            return -1
        if want_fn and f"def {want_fn}" in c:
            return 2
        if "def " in c:
            return 1
        return 0

    best = max(candidates, key=score)
    return best if score(best) >= 0 else None


# ─── retry feedback ────────────────────────────────────────────────────────────

def _retry_prompt(task: dict, original_prompt: str, detail: str) -> str:
    """Build a second-attempt prompt that includes specific feedback."""
    hint = task.get("retry_hint", "")
    return (
        f"{original_prompt}\n\n"
        f"Your previous attempt was incorrect: {detail}.\n"
        + (f"{hint}\n" if hint else "")
        + "Please try again."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PROBE FUNCTIONS  (code tasks — haiku's first python_exec output is tested)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Luhn ──────────────────────────────────────────────────────────────────────

def _probe_luhn(ns):
    fn = ns.get("luhn_check") or ns.get("luhn") or ns.get("check_luhn")
    if not fn:
        return ["luhn_check not defined"]
    fails = []
    try:
        r = fn("18")
        if r is not True:
            fails.append(
                f"luhn('18')={r!r}, expected True. "
                "FIX: double every SECOND digit from the RIGHTMOST position, not the left; "
                "'18'→keep 8, double 1→2; sum=10→valid"
            )
            return fails
        if fn("17") is not False:
            fails.append(f"luhn('17')={fn('17')!r}, expected False")
        if fn("79927398713") is not True:
            fails.append("luhn('79927398713') should be True (Wikipedia example)")
        if fn("26") is not True:
            fails.append(
                f"luhn('26')={fn('26')!r}, expected True. "
                "FIX: when doubled digit > 9, subtract 9; '26'→keep 6, 2×2=4; sum=10"
            )
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_luhn(ns):
    fn = ns.get("luhn_check") or ns.get("luhn") or ns.get("check_luhn")
    if not fn: return False
    try:
        return all(fn(s) is e for s, e in [
            ("18", True), ("17", False), ("26", True),
            ("79927398713", True), ("79927398714", False),
            ("4111111111111111", True), ("1234567890123456", False),
        ])
    except Exception: return False


# ── Game of Life ───────────────────────────────────────────────────────────────

def _probe_game_of_life(ns):
    fn = ns.get("game_of_life") or ns.get("gameOfLife") or ns.get("step")
    if not fn: return ["game_of_life not defined"]
    fails = []
    try:
        block = [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]
        result = fn([r[:] for r in block], 1)
        if result != block:
            fails.append(
                f"block still-life changed after 1 step: got {result}. "
                "FIX: build a NEW grid for each step — do NOT modify the board in-place; "
                "compute all cells from the original state before writing any results"
            )
            return fails
        horiz = [[0,0,0],[1,1,1],[0,0,0]]
        vert  = [[0,1,0],[0,1,0],[0,1,0]]
        if fn([r[:] for r in horiz], 1) != vert:
            fails.append(
                "horizontal blinker did not become vertical after 1 step. "
                "Rule: live cell survives with 2–3 neighbors; dead cell born with exactly 3"
            )
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_game_of_life(ns):
    fn = ns.get("game_of_life") or ns.get("gameOfLife") or ns.get("step")
    if not fn: return False
    try:
        block = [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]
        if fn([r[:] for r in block], 1) != block: return False
        horiz = [[0,0,0],[1,1,1],[0,0,0]]
        vert  = [[0,1,0],[0,1,0],[0,1,0]]
        if fn([r[:] for r in horiz], 1) != vert: return False
        if fn([r[:] for r in horiz], 2) != horiz: return False
        return True
    except Exception: return False


# ── Day of week ────────────────────────────────────────────────────────────────

def _probe_day_of_week(ns):
    fn = ns.get("day_of_week") or ns.get("dayOfWeek") or ns.get("weekday")
    if not fn: return ["day_of_week not defined"]
    fails = []
    try:
        r = fn(2024, 1, 1)
        if r != "Monday":
            fails.append(
                f"day_of_week(2024,1,1)={r!r}, expected 'Monday'. "
                "FIX: use Tomohiko Sakamoto's algorithm — "
                "t=[0,3,2,5,0,3,5,1,4,6,2,4]; if month<3: year-=1; "
                "return DAYS[(year+year//4-year//100+year//400+t[month-1]+day)%7] "
                "where DAYS[0]='Sunday'"
            )
            return fails
        for y,m,d,exp in [(1970,1,1,"Thursday"),(2000,1,1,"Saturday"),(1999,12,31,"Friday")]:
            r = fn(y,m,d)
            if r != exp:
                fails.append(f"day_of_week({y},{m},{d})={r!r}, expected {exp!r}")
                break
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_day_of_week(ns):
    fn = ns.get("day_of_week") or ns.get("dayOfWeek") or ns.get("weekday")
    if not fn: return False
    try:
        return all(fn(y,m,d)==e for y,m,d,e in [
            (2024,1,1,"Monday"),(1970,1,1,"Thursday"),(2000,1,1,"Saturday"),
            (2023,12,25,"Monday"),(1999,12,31,"Friday"),(2024,2,29,"Thursday"),
        ])
    except Exception: return False


# ── Decode string ──────────────────────────────────────────────────────────────

def _probe_decode_string(ns):
    fn = ns.get("decode_string") or ns.get("decodeString") or ns.get("decode")
    if not fn: return ["decode_string not defined"]
    fails = []
    try:
        r = fn("3[a2[c]]")
        if r != "accaccacc":
            fails.append(
                f"decode('3[a2[c]]')={r!r}, expected 'accaccacc'. "
                "FIX: use a STACK — push (current_string, repeat_count) on '['; "
                "on ']' pop, multiply the inner result by count and append to outer"
            )
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_decode_string(ns):
    fn = ns.get("decode_string") or ns.get("decodeString") or ns.get("decode")
    if not fn: return False
    try:
        return all(fn(s)==e for s,e in [
            ("3[a]2[bc]","aaabcbc"),("3[a2[c]]","accaccacc"),
            ("2[abc]3[cd]ef","abcabccdcdcdef"),("abc","abc"),
            ("10[a]","aaaaaaaaaa"),("2[3[a]b]","aaabaaab"),
        ])
    except Exception: return False


# ── Soundex ────────────────────────────────────────────────────────────────────

def _probe_soundex(ns):
    fn = ns.get("soundex")
    if not fn: return ["soundex not defined"]
    fails = []
    try:
        r = fn("Ashcraft")
        if r != "A261":
            fails.append(
                f"soundex('Ashcraft')={r!r}, expected 'A261'. "
                "FIX: H and W between coded letters are IGNORED (not treated as separators); "
                "table: BFPV=1 CGJKQSXYZ=2 DT=3 L=4 MN=5 R=6; "
                "remove adjacent same codes; pad/truncate to 4 chars"
            )
            return fails
        for name, exp in [("Robert","R163"),("Rupert","R163")]:
            r = fn(name)
            if r != exp:
                fails.append(f"soundex('{name}')={r!r}, expected {exp!r}")
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_soundex(ns):
    fn = ns.get("soundex")
    if not fn: return False
    try:
        return all(fn(n)==e for n,e in [
            ("Robert","R163"),("Rupert","R163"),("Ashcraft","A261"),
            ("Euler","E460"),("Ellery","E460"),("Gauss","G200"),("Thompson","T512"),
        ])
    except Exception: return False


# ── Look-and-say ──────────────────────────────────────────────────────────────

def _probe_look_and_say(ns):
    fn = ns.get("look_and_say") or ns.get("lookAndSay")
    if not fn: return ["look_and_say not defined"]
    fails = []
    try:
        for n,e in [(1,"1"),(2,"11"),(3,"21"),(4,"1211"),(5,"111221")]:
            r = fn(n)
            if str(r) != e:
                fails.append(f"look_and_say({n})={r!r}, expected {e!r}")
                return fails
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_look_and_say(ns):
    fn = ns.get("look_and_say") or ns.get("lookAndSay")
    if not fn: return False
    try:
        return all(str(fn(i+1))==e for i,e in enumerate(
            ["1","11","21","1211","111221","312211","13112221"]))
    except Exception: return False


# ── Fraction add ──────────────────────────────────────────────────────────────

def _probe_fraction_add(ns):
    fn = ns.get("fraction_add") or ns.get("fractionAdd") or ns.get("add_fractions")
    if not fn: return ["fraction_add not defined"]
    fails = []
    try:
        r = fn("1/3","1/6")
        if r != "1/2":
            fails.append(f"fraction_add('1/3','1/6')={r!r}, expected '1/2'")
            return fails
        r2 = fn("-1/2","1/3")
        if r2 != "-1/6":
            fails.append(f"fraction_add('-1/2','1/3')={r2!r}, expected '-1/6'")
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_fraction_add(ns):
    fn = ns.get("fraction_add") or ns.get("fractionAdd") or ns.get("add_fractions")
    if not fn: return False
    try:
        return all(fn(a,b)==e for a,b,e in [
            ("1/3","1/6","1/2"),("1/2","1/3","5/6"),
            ("-1/2","1/3","-1/6"),("1/1","1/1","2/1"),
            ("-1/3","-1/6","-1/2"),
        ])
    except Exception: return False


# ── Knight moves ──────────────────────────────────────────────────────────────

def _probe_knight(ns):
    fn = ns.get("knight_min_moves") or ns.get("knightMinMoves") or ns.get("min_knight_moves")
    if not fn: return ["knight_min_moves not defined"]
    fails = []
    try:
        if fn((0,0),(0,0)) != 0:
            fails.append("knight((0,0),(0,0)) should be 0")
            return fails
        r = fn((0,0),(1,1))
        if r != 4:
            fails.append(
                f"knight((0,0),(1,1))={r}, expected 4. "
                "FIX: use BFS on infinite board; (0,0)→(1,1) requires 4 moves due to parity"
            )
            return fails
        if fn((0,0),(1,2)) != 1:
            fails.append(f"knight((0,0),(1,2))={fn((0,0),(1,2))}, expected 1")
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_knight(ns):
    fn = ns.get("knight_min_moves") or ns.get("knightMinMoves") or ns.get("min_knight_moves")
    if not fn: return False
    try:
        return all(fn(s,t)==e for s,t,e in [
            ((0,0),(0,0),0),((0,0),(1,2),1),((0,0),(2,1),1),
            ((0,0),(1,1),4),((0,0),(2,2),4),((0,0),(3,3),2),((0,0),(0,1),3),
        ])
    except Exception: return False


# ── Haversine ──────────────────────────────────────────────────────────────────

def _probe_haversine(ns):
    fn = ns.get("haversine") or ns.get("haversine_distance")
    if not fn: return ["haversine not defined"]
    fails = []
    try:
        if abs(fn(0.0,0.0,0.0,0.0)) > 0.01:
            fails.append("haversine(0,0,0,0) should be 0")
            return fails
        r = fn(51.5074,-0.1278,48.8566,2.3522)
        if abs(r - 341) > 15:
            fails.append(
                f"haversine(London,Paris)={r:.1f} km, expected ~341. "
                "FIX: convert degrees to RADIANS with math.radians() before calling sin/cos; "
                "formula: a=sin²(Δlat/2)+cos(lat1)·cos(lat2)·sin²(Δlon/2); d=2·6371·asin(√a)"
            )
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_haversine(ns):
    fn = ns.get("haversine") or ns.get("haversine_distance")
    if not fn: return False
    try:
        if abs(fn(0.0,0.0,0.0,0.0)) > 0.01: return False
        if abs(fn(51.5074,-0.1278,48.8566,2.3522) - 341) > 15: return False
        if abs(fn(40.7128,-74.0060,34.0522,-118.2437) - 3940) > 60: return False
        return True
    except Exception: return False


# ── Pig Latin ──────────────────────────────────────────────────────────────────

def _probe_pig_latin(ns):
    fn = ns.get("pig_latin") or ns.get("pigLatin") or ns.get("to_pig_latin")
    if not fn: return ["pig_latin not defined"]
    fails = []
    try:
        r = fn("string")
        if r != "ingstray":
            fails.append(
                f"pig_latin('string')={r!r}, expected 'ingstray'. "
                "FIX: move ALL leading consonants (not just the first) to end; "
                "'string'→consonants='str', rest='ing', result='ing'+'str'+'ay'"
            )
            return fails
        if fn("apple") != "appleyay":
            fails.append(f"pig_latin('apple')={fn('apple')!r}, expected 'appleyay' (vowel start: add 'yay')")
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_pig_latin(ns):
    fn = ns.get("pig_latin") or ns.get("pigLatin") or ns.get("to_pig_latin")
    if not fn: return False
    try:
        return all(fn(w)==e for w,e in [
            ("pig","igpay"),("apple","appleyay"),("the","ethay"),
            ("string","ingstray"),("glove","oveglay"),("school","oolschay"),
        ])
    except Exception: return False


# ── Roman to int ──────────────────────────────────────────────────────────────

def _probe_roman(ns):
    fn = ns.get("roman_to_int") or ns.get("romanToInt")
    if not fn: return ["roman_to_int not defined"]
    fails = []
    try:
        for s,e in [("IV",4),("IX",9),("XL",40),("MCMXCIV",1994)]:
            r = fn(s)
            if r != e:
                fails.append(
                    f"roman_to_int('{s}')={r}, expected {e}. "
                    "FIX: subtraction rule — if a smaller value precedes larger, subtract it"
                )
                return fails
    except Exception as e:
        fails.append(f"error: {e}")
    return fails

def _check_roman(ns):
    fn = ns.get("roman_to_int") or ns.get("romanToInt")
    if not fn: return False
    try:
        return all(fn(s)==e for s,e in [
            ("III",3),("IV",4),("IX",9),("LVIII",58),
            ("MCMXCIV",1994),("XL",40),("XC",90),("CD",400),("CM",900),
        ])
    except Exception: return False


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT TASK CHECK FUNCTIONS
#  check_fn(response: str) -> bool
# ═══════════════════════════════════════════════════════════════════════════════

def _check_bat_ball(resp: str) -> bool:
    # Correct: 5 cents. Common wrong answer: 10 cents.
    r = resp.lower()
    return bool(re.search(r'\b5\s*(?:cents?|¢)\b|\b0\.05\b|\bfive\s+cents?\b', r))

def _check_monty_hall(resp: str) -> bool:
    # Should say switch and give 2/3 probability
    r = resp.lower()
    has_switch = bool(re.search(r'\bswitch\b|\bchange\b|\bswap\b', r))
    has_prob   = bool(re.search(r'2/3|two.thirds?|66\.?6?%|0\.6+6?\b', r))
    return has_switch and has_prob

def _check_landlocked(resp: str) -> bool:
    # Largest landlocked country by area = Kazakhstan
    return "kazakhstan" in resp.lower()

def _check_time_zones(resp: str) -> bool:
    # Country with most time zones = France (13, due to overseas territories)
    return "france" in resp.lower()

def _check_saturn_moons(resp: str) -> bool:
    # As of 2024, Saturn has more moons than Jupiter
    return "saturn" in resp.lower()

def _check_women_vote(resp: str) -> bool:
    # New Zealand, 1893
    r = resp.lower()
    return "new zealand" in r

def _check_octopus_hearts(resp: str) -> bool:
    # 3 hearts (2 branchial + 1 systemic)
    r = resp.lower()
    return bool(re.search(r'\bthree\b|\b3\b', r))

def _check_coin_independence(resp: str) -> bool:
    # After 10 heads, P(heads on 11th) is still exactly 1/2 — independent events
    r = resp.lower()
    return bool(re.search(r'\b(?:0\.5|1/2|50\s*%|fifty\s*percent|one.half)\b', r))

def _check_periodic_jq(resp: str) -> bool:
    # J and Q are the only letters not appearing in any element symbol
    r = resp.lower()
    # Must mention both J and Q (Q is less known; haiku often forgets it)
    has_j = bool(re.search(r'\bj\b', r))
    has_q = bool(re.search(r'\bq\b', r))
    return has_j and has_q

def _check_blood_type(resp: str) -> bool:
    # Most common worldwide is O+ (O positive)
    r = resp.lower()
    return bool(re.search(r'o\s*\+|o\s*pos', r))

def _check_coriolis_myth(resp: str) -> bool:
    # The Coriolis effect does NOT determine drain swirl direction in sinks — it's a myth.
    # Correct answer: direction is random / determined by basin shape, not hemisphere.
    r = resp.lower()
    myth_keywords = ["random", "no consistent", "no definite", "not determine",
                     "myth", "negligible", "basin", "shape of", "doesn't determine",
                     "does not determine", "too small", "not caused"]
    return any(k in r for k in myth_keywords)

def _check_most_sides(resp: str) -> bool:
    # How many sides does a myriagon have? 10,000
    r = resp.lower()
    return bool(re.search(r'10[,\s]?000|ten thousand', r))

def _check_light_speed(resp: str) -> bool:
    # Speed of light in vacuum: 299,792,458 m/s (exact, by definition)
    r = resp.replace(",", "").replace(" ", "")
    return "299792458" in r

def _check_elements_order(resp: str) -> bool:
    # What comes after Oganesson (118) on the periodic table? Nothing — it's the last.
    r = resp.lower()
    return any(k in r for k in [
        "nothing", "last", "no element", "doesn't exist", "does not exist",
        "end of", "heaviest", "undiscovered", "not yet", "currently"
    ])

def _check_prime_count_under_10(resp: str) -> bool:
    # Primes under 10: 2, 3, 5, 7 → 4 primes
    r = resp.lower()
    return bool(re.search(r'\bfour\b|\b4\b', r))

def _check_earth_layers(resp: str) -> bool:
    # Earth's thickest layer is the mantle (not the core or crust)
    return "mantle" in resp.lower()

def _check_great_wall_visible(resp: str) -> bool:
    # The Great Wall of China is NOT visible from space with the naked eye — a myth
    r = resp.lower()
    return any(k in r for k in [
        "not visible", "cannot be seen", "can't be seen", "myth",
        "false", "not true", "no, it", "not actually"
    ])

def _check_water_formula(resp: str) -> bool:
    # Water is H2O — also accept Unicode subscript H₂O
    import unicodedata
    r = unicodedata.normalize("NFKC", resp).lower()
    return bool(re.search(r'h\s*2\s*o', r))


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK LIST — 30 tasks: 9 hard code · 6 medium code · 15 text/reasoning
# ═══════════════════════════════════════════════════════════════════════════════

TASKS = [
    # ── HARD CODE ─────────────────────────────────────────────────────────────
    {"name": "Luhn checksum",
     "type": "code", "domain": "algorithm", "want_fn": "luhn_check",
     "probe": _probe_luhn, "check": _check_luhn,
     "prompt": (
         f"Write `luhn_check(number: str) -> bool` — credit card Luhn validation. {_NC} "
         f"Use python_exec to test your implementation."
     )},
    {"name": "Conway's Game of Life",
     "type": "code", "domain": "algorithm", "want_fn": "game_of_life",
     "probe": _probe_game_of_life, "check": _check_game_of_life,
     "prompt": (
         f"Write `game_of_life(board: list[list[int]], steps: int) -> list[list[int]]` — "
         f"simulate Conway's Game of Life for the given steps. Return the new board; "
         f"do not modify the input. {_NC} Use python_exec to test your implementation."
     )},
    {"name": "day of week (no datetime)",
     "type": "code", "domain": "algorithm", "want_fn": "day_of_week",
     "probe": _probe_day_of_week, "check": _check_day_of_week,
     "prompt": (
         f"Write `day_of_week(year: int, month: int, day: int) -> str` — "
         f"return the weekday name ('Monday' etc.). Do NOT import datetime or calendar. "
         f"Use Zeller's congruence or Tomohiko Sakamoto's algorithm. {_NC} "
         f"Use python_exec to test your implementation."
     ),
     "retry_hint": "Verify your formula handles January/February correctly (year adjustment needed)."},
    {"name": "decode string (nested brackets)",
     "type": "code", "domain": "algorithm", "want_fn": "decode_string",
     "probe": _probe_decode_string, "check": _check_decode_string,
     "prompt": (
         f"Write `decode_string(s: str) -> str`. "
         f"Rule: k[string] repeats string k times; brackets may be arbitrarily nested. {_NC} "
         f"Use python_exec to test your implementation."
     )},
    {"name": "Soundex algorithm",
     "type": "code", "domain": "algorithm", "want_fn": "soundex",
     "probe": _probe_soundex, "check": _check_soundex,
     "prompt": (
         f"Write `soundex(name: str) -> str` — American Soundex phonetic code. "
         f"Keep first letter; encode B/F/P/V=1, C/G/J/K/Q/S/X/Z=2, D/T=3, L=4, M/N=5, R=6; "
         f"H and W are ignored; remove vowels; drop adjacent same codes; pad/truncate to 4 chars. "
         f"{_NC} Use python_exec to test your implementation."
     ),
     "retry_hint": "Pay close attention to the H/W rule: they are silently dropped, not treated as separators."},
    {"name": "look-and-say sequence",
     "type": "code", "domain": "algorithm", "want_fn": "look_and_say",
     "probe": _probe_look_and_say, "check": _check_look_and_say,
     "prompt": (
         f"Write `look_and_say(n: int) -> str` — nth term of the look-and-say sequence "
         f"(1-indexed; term 1 = '1'). {_NC} Use python_exec to test your implementation."
     )},
    {"name": "fraction addition",
     "type": "code", "domain": "algorithm", "want_fn": "fraction_add",
     "probe": _probe_fraction_add, "check": _check_fraction_add,
     "prompt": (
         f"Write `fraction_add(a: str, b: str) -> str` — add two fractions given as "
         f"strings like '1/3'. Return the simplified result, e.g. '1/2'. Handle negatives. "
         f"{_NC} Use python_exec to test your implementation."
     )},
    {"name": "knight minimum moves",
     "type": "code", "domain": "algorithm", "want_fn": "knight_min_moves",
     "probe": _probe_knight, "check": _check_knight,
     "prompt": (
         f"Write `knight_min_moves(source: tuple, target: tuple) -> int` — minimum chess "
         f"knight moves from source to target on an infinite board. {_NC} "
         f"Use python_exec to test your implementation."
     ),
     "retry_hint": "Note: (0,0)→(1,1) requires 4 moves, not 2 — the geometry forces a detour."},
    {"name": "Haversine distance",
     "type": "code", "domain": "algorithm", "want_fn": "haversine",
     "probe": _probe_haversine, "check": _check_haversine,
     "prompt": (
         f"Write `haversine(lat1, lon1, lat2, lon2) -> float` — great-circle distance "
         f"in kilometres between two GPS coordinates. R=6371 km. "
         f"{_NC} Use python_exec to test your implementation."
     ),
     "retry_hint": "Remember to convert degrees to radians before passing to sin/cos."},
    # ── MEDIUM CODE ───────────────────────────────────────────────────────────
    {"name": "Pig Latin",
     "type": "code", "domain": "algorithm", "want_fn": "pig_latin",
     "probe": _probe_pig_latin, "check": _check_pig_latin,
     "prompt": (
         f"Write `pig_latin(word: str) -> str` — convert one lowercase word to Pig Latin. "
         f"Vowel start → append 'yay'. Consonant start → move ALL leading consonants to end, "
         f"then append 'ay'. {_NC} Use python_exec to test your implementation."
     )},
    {"name": "Roman numerals to int",
     "type": "code", "domain": "algorithm", "want_fn": "roman_to_int",
     "probe": _probe_roman, "check": _check_roman,
     "prompt": (
         f"Write `roman_to_int(s: str) -> int` — parse a Roman numeral string. "
         f"Handle subtractive notation (IV=4, IX=9, XL=40, XC=90, CD=400, CM=900). "
         f"{_NC} Use python_exec to test your implementation."
     )},
    # ── MATH / LOGIC PUZZLES ──────────────────────────────────────────────────
    {"name": "bat and ball problem",
     "type": "text", "domain": "math_logic",
     "check": _check_bat_ball,
     "prompt": (
         "A bat and a ball cost $1.10 in total. "
         "The bat costs exactly $1.00 more than the ball. "
         "How much does the ball cost? Give your answer in cents (just the number)."
     ),
     "retry_hint": (
         "Set up the equations: let ball = x cents. "
         "Bat = x + 100. Together: x + (x+100) = 110."
     )},
    {"name": "Monty Hall problem",
     "type": "text", "domain": "math_logic",
     "check": _check_monty_hall,
     "prompt": (
         "You are on a game show with three doors. Behind one is a car, behind the others are goats. "
         "You pick door 1. The host (who knows what's behind each door) opens door 3 to reveal a goat. "
         "Should you switch to door 2? What is the probability of winning if you switch? "
         "Give a clear yes/no and the exact probability as a fraction."
     ),
     "retry_hint": (
         "Think about it this way: 2/3 of the time you originally picked a goat. "
         "When that happens, the host is forced to open the other goat door, "
         "and switching wins. Calculate the probability of winning by switching."
     )},
    {"name": "independent coin flips",
     "type": "text", "domain": "math_logic",
     "check": _check_coin_independence,
     "prompt": (
         "You flip a fair coin 10 times and get heads every single time. "
         "What is the exact probability of getting heads on the 11th flip? "
         "Give your answer as a decimal or fraction."
     ),
     "retry_hint": "Each coin flip is an independent event — past outcomes do not affect future flips."},
    {"name": "primes under 10",
     "type": "text", "domain": "math_logic",
     "check": _check_prime_count_under_10,
     "prompt": (
         "How many prime numbers are there that are strictly less than 10? "
         "List them and give the count."
     )},
    {"name": "myriagon sides",
     "type": "text", "domain": "math_logic",
     "check": _check_most_sides,
     "prompt": (
         "A myriagon is a polygon. How many sides does it have? "
         "Give the exact number."
     ),
     "retry_hint": "The prefix 'myria-' means ten thousand."},
    # ── GEOGRAPHY / HISTORY ───────────────────────────────────────────────────
    {"name": "largest landlocked country",
     "type": "text", "domain": "geography",
     "check": _check_landlocked,
     "prompt": (
         "What is the largest country in the world by land area that is completely "
         "landlocked (has no coastline)? Name the country."
     ),
     "retry_hint": "Think Central Asia — it is not Mongolia."},
    {"name": "most time zones",
     "type": "text", "domain": "geography",
     "check": _check_time_zones,
     "prompt": (
         "Which country spans the most time zones in the world? "
         "Consider all territories and overseas departments."
     ),
     "retry_hint": "Consider countries with scattered overseas territories far from their mainland."},
    {"name": "planet with most moons (2024)",
     "type": "text", "domain": "science",
     "check": _check_saturn_moons,
     "prompt": (
         "As of 2024, which planet in our solar system has the greatest number of "
         "confirmed natural moons? Name the planet."
     ),
     "retry_hint": "The answer changed in recent years — check whether Jupiter or Saturn holds the record now."},
    {"name": "first national women's vote",
     "type": "text", "domain": "history",
     "check": _check_women_vote,
     "prompt": (
         "Which country was the first to grant women the right to vote at the national level? "
         "Name the country and the year."
     ),
     "retry_hint": "Look to the Southern Hemisphere in the late 19th century."},
    # ── SCIENCE / NATURE ─────────────────────────────────────────────────────
    {"name": "octopus hearts",
     "type": "text", "domain": "science",
     "check": _check_octopus_hearts,
     "prompt": (
         "How many hearts does an octopus have? "
         "Give the number and briefly describe their roles."
     ),
     "retry_hint": "An octopus has more than two hearts — one pumps blood to the body, two to the gills."},
    {"name": "Earth's thickest layer",
     "type": "text", "domain": "science",
     "check": _check_earth_layers,
     "prompt": (
         "What is the thickest layer of the Earth (by depth/volume)? "
         "Choose from: inner core, outer core, mantle, crust."
     )},
    {"name": "Great Wall from space (myth)",
     "type": "text", "domain": "science",
     "check": _check_great_wall_visible,
     "prompt": (
         "Is the Great Wall of China visible from space with the naked eye? "
         "Give a clear yes or no, and explain the scientific reason."
     ),
     "retry_hint": "Consider the wall's width (about 5–9 metres) relative to the resolution of the human eye at orbital altitude."},
    {"name": "Coriolis drain myth",
     "type": "text", "domain": "science",
     "check": _check_coriolis_myth,
     "prompt": (
         "Does water drain clockwise in the Northern Hemisphere and counterclockwise "
         "in the Southern Hemisphere due to the Coriolis effect? "
         "Explain your answer precisely."
     ),
     "retry_hint": (
         "The Coriolis force is real but extremely weak at the scale of a bathtub — "
         "the basin shape and initial water motion dominate."
     )},
    {"name": "letters not on periodic table",
     "type": "text", "domain": "science",
     "check": _check_periodic_jq,
     "prompt": (
         "Which letters of the alphabet do NOT appear in any element symbol on the "
         "periodic table? List all of them."
     ),
     "retry_hint": "There are exactly two such letters. Think carefully — 'Q' is one, what is the other?"},
    {"name": "element after Oganesson",
     "type": "text", "domain": "science",
     "check": _check_elements_order,
     "prompt": (
         "Oganesson (Og, element 118) is the last element on the current periodic table. "
         "What element comes after it? Give its name and atomic number."
     ),
     "retry_hint": "Oganesson is the current heaviest confirmed element; no element beyond it has been officially named."},
    # ── MIXED KNOWLEDGE ──────────────────────────────────────────────────────
    {"name": "speed of light exact",
     "type": "text", "domain": "science",
     "check": _check_light_speed,
     "prompt": (
         "What is the exact speed of light in a vacuum in metres per second? "
         "Give the precise integer value (it is defined by international convention)."
     )},
    {"name": "most common blood type",
     "type": "text", "domain": "science",
     "check": _check_blood_type,
     "prompt": (
         "What is the most common blood type worldwide? "
         "Give the ABO type and Rh factor (e.g. A+)."
     ),
     "retry_hint": "The most common type globally is O positive (O+), not A or B."},
    {"name": "chemical formula of water",
     "type": "text", "domain": "science",
     "check": _check_water_formula,
     "prompt": (
         "What is the chemical formula for water? "
         "Write it using standard notation."
     )},
    {"name": "right angle triangle — 3-4-5",
     "type": "text", "domain": "math_logic",
     "check": lambda r: bool(re.search(r'\byes\b|\bright angle\b|\bis\b.*right|right.*triangle', r.lower()))
                        and "5" in r,
     "prompt": (
         "A triangle has sides of length 3, 4, and 5. Is it a right-angled triangle? "
         "Verify using the Pythagorean theorem and state which angle is 90°."
     )},
]

assert len(TASKS) == 30, f"Expected 30 tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval() -> tuple[list[dict], list[int]]:
    print("\n" + "═" * 72)
    print("  eval_fires — 30 tasks across algorithms, math, geography,")
    print("  history, science, and logic. Retry loop feeds back failures.")
    print("  Brain stores first-attempt traces. Model: haiku")
    print("═" * 72)

    embedder    = build_embedder()
    brain       = BrainAgent(embedder, threshold=0.35, k=5)
    code_agent  = tool_agent(["python_exec"], max_turns=MAX_TURNS,
                             model=HAIKU, max_tokens=3000)
    code_agent.monitor = brain
    viz = BrainViz()

    results:     list[dict] = []
    fire_counts: list[int]  = []

    for i, task in enumerate(TASKS):
        n = i + 1
        ttype = task["type"]
        print(f"\n  {n:>2}/30 [{task['domain']:12}] {task['name']}")

        # Reset brain for new task
        brain.set_task(i, probe_fn=task.get("probe") if ttype == "code" else None)
        brain.reset()

        t0    = time.time()
        agent = code_agent if ttype == "code" else haiku
        result = agent(task["prompt"])
        elapsed = time.time() - t0
        trace, tok1 = result if isinstance(result, tuple) else (str(result), 0)

        # ── first-attempt check ──
        if ttype == "code":
            code   = _extract_code(trace, task.get("want_fn"))
            passed, detail = _check_code(code, task["check"])
        else:
            code   = None
            passed, detail = _check_text(trace, task["check"])

        first_passed = passed
        fires = brain._code_interventions if ttype == "code" else 0

        # ── store first attempt in brain (always) ──
        if ttype == "code" and code:
            brain.store_code(code, int(passed), metadata=detail if not passed else "")
        brain.store(trace, int(passed), metadata=detail if not passed else "")

        # ── retry with feedback if failed ──
        tok2 = 0
        if not passed:
            retry_p = _retry_prompt(task, task["prompt"], detail)
            brain.reset()
            r2 = agent(retry_p)
            t2, tok2 = r2 if isinstance(r2, tuple) else (str(r2), 0)
            elapsed += time.time() - t0 - elapsed  # running total

            if ttype == "code":
                code2 = _extract_code(t2, task.get("want_fn"))
                passed, detail2 = _check_code(code2, task["check"])
            else:
                passed, detail2 = _check_text(t2, task["check"])

            if passed:
                detail = f"fixed on retry ({detail})"

        total_tok = tok1 + tok2
        fire_tag  = f"  [⚡×{fires}]" if fires else ""
        status    = "PASS" if passed else "FAIL"
        retry_tag = "  [retry]" if not first_passed else ""
        print(f"       {status}{retry_tag}  {total_tok:>6,} tok  {time.time()-t0:.0f}s"
              f"{fire_tag}"
              + (f"  {detail[:55]}" if detail and not passed else ""))

        results.append({
            "task": n, "name": task["name"], "domain": task["domain"],
            "type": ttype,
            "first_passed": first_passed,
            "passed": passed,
            "tokens": total_tok, "elapsed": round(time.time() - t0, 1),
            "fires": fires, "detail": detail,
        })
        fire_counts.append(fires)

        viz.update(brain, results, fire_counts)
        viz.save(OUT / "brain_overview.png")

        if n % 10 == 0:
            n_pass1  = sum(1 for r in results if r["first_passed"])
            n_pass   = sum(1 for r in results if r["passed"])
            n_fired  = sum(1 for c in fire_counts if c > 0)
            print(f"\n  ── {n}/30: first-attempt {n_pass1}/{n}  "
                  f"final {n_pass}/{n}  fire {n_fired/n:.0%} ──")

    _report(results, fire_counts)
    with open(OUT / "fires_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved brain_overview.png + fires_run.json")
    return results, fire_counts


def _report(results, fire_counts):
    n  = len(results)
    p1 = sum(1 for r in results if r["first_passed"])
    pf = sum(1 for r in results if r["passed"])
    nf = sum(1 for c in fire_counts if c > 0)
    fr = nf / n

    by_domain: dict[str, list] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)

    print("\n" + "═" * 72)
    print("  FINAL RESULTS")
    print("─" * 72)
    print(f"  First-attempt accuracy : {p1}/{n}  ({p1/n:.0%})")
    print(f"  Final accuracy         : {pf}/{n}  ({pf/n:.0%})  (includes retries)")
    print(f"  Brain fire rate        : {nf}/{n}  ({fr:.0%})")
    print()
    for domain, rs in sorted(by_domain.items()):
        p  = sum(1 for r in rs if r["passed"])
        p1d = sum(1 for r in rs if r["first_passed"])
        fs = sum(1 for r in rs if r["fires"] > 0)
        print(f"  {domain:14} {p1d:>2}/{len(rs)} first  {p:>2}/{len(rs)} final  "
              f"  {fs:>2}/{len(rs)} fired")
    print("─" * 72)
    print(f"  Avg tokens : {sum(r['tokens'] for r in results)/n:,.0f} / task")
    print(f"  Total time : {sum(r['elapsed'] for r in results):.0f}s")
    print("═" * 72)


if __name__ == "__main__":
    run_eval()
