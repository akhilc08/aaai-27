"""LLM agents for the TextArena pilot (v2).

Arms (same model, same action prompt; only the extra block differs):
  baseline : 1 call/turn  -> act
  cot2     : 2 calls/turn -> free-form analysis, then act conditioned on the analysis
  belief   : 2 calls/turn -> update JSON opponent-belief block, then act conditioned on it (method 3a)
Interface: agent.decide(obs, legal) and agent.retry(obs, legal, error) -> (action_text, info)
"""
import json, os, re, threading, time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Users/sickle/Coding/context-research/.env")
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

MODELS = ["qwen/qwen3-30b-a3b-instruct-2507", "deepseek/deepseek-chat", "z-ai/glm-4.5-air"]
TEMPERATURE = 0.3
STOP_AT_USD = 3.83  # project cap: $3.85 cumulative in runs/usage.jsonl (backstop; run.py gates earlier)

RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)
USAGE_LOG = RUNS / "usage.jsonl"
_lock = threading.Lock()
_spent = {"usd": 0.0}
if USAGE_LOG.exists():  # resume: count prior spend
    _spent["usd"] = sum(json.loads(l).get("cost") or 0 for l in USAGE_LOG.read_text().splitlines())


class BudgetExceeded(RuntimeError):
    pass


def spent_usd() -> float:
    return _spent["usd"]


def _log_usage(rec: dict):
    with _lock:
        _spent["usd"] += rec.get("cost") or 0.0
        with USAGE_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def chat(system: str, user: str, meta: dict, max_tokens: int = 700) -> tuple[str, dict]:
    """One chat call with model fallback + retries. Returns (text, usage record)."""
    if _spent["usd"] > STOP_AT_USD:
        raise BudgetExceeded(f"spent ${_spent['usd']:.3f}")
    last_err = None
    for model in MODELS:
        for attempt in range(3):
            try:
                r = _client.chat.completions.create(
                    model=model, temperature=TEMPERATURE, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    extra_body={"usage": {"include": True}})
                text = (r.choices[0].message.content or "").strip()
                u = r.usage.model_dump() if r.usage else {}
                rec = {"ts": time.time(), "model": model, **meta, "prompt_tokens": u.get("prompt_tokens", 0),
                       "completion_tokens": u.get("completion_tokens", 0), "cost": u.get("cost", 0.0)}
                _log_usage(rec)
                if text:
                    return text, rec
                last_err = RuntimeError("empty completion")
            except BudgetExceeded:
                raise
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"all models failed: {last_err}")


BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def extract_action(text: str) -> str:
    m = BRACKET_RE.findall(text)
    return m[-1] if m else ""


def _add(tok: list, rec: dict):
    tok[0] += rec["prompt_tokens"]; tok[1] += rec["completion_tokens"]


ACT_SYSTEM = (
    "You are a competitive game player. Make sure you read the game instructions carefully, "
    "and always follow the required format.\n"
    "You may think briefly (under 120 words, no square brackets in your thinking) before acting. "
    "End your response with your chosen action in square brackets on its own final line, exactly in the "
    "format the game instructions require, e.g. [Bid: 3, 4] or [Call] or [check] or [4]."
)


def act_prompt(obs: str, legal: str, block: str = "") -> str:
    # the CURRENT DECISION line always comes last so no extra block can make the agent answer a stale situation
    return (f"{obs}\n\n{block}" + f"=== CURRENT DECISION ===\n{legal}\n\n"
            "Now think briefly, then end with your action in square brackets.")


class _Base:
    arm = "?"

    def __init__(self, meta: dict):
        self.meta = {**meta, "arm": self.arm}
        self.block = ""  # extra context kept for retries

    def _act(self, obs, legal, error=None) -> tuple[str, dict]:
        user = act_prompt(obs, legal, self.block)
        if error:
            user += f"\n\nYOUR PREVIOUS ACTION WAS ILLEGAL: {error}\nChoose a LEGAL action."
        return chat(ACT_SYSTEM, user, {**self.meta, "call": "act"}, max_tokens=600)

    def retry(self, obs, legal, error):
        text, rec = self._act(obs, legal, error)
        tok = [0, 0]; _add(tok, rec)
        return extract_action(text), {"raw": text, "tokens": tok}


class BaselineAgent(_Base):
    arm = "baseline"

    def decide(self, obs, legal):
        text, rec = self._act(obs, legal)
        tok = [0, 0]; _add(tok, rec)
        return extract_action(text), {"raw": text, "tokens": tok, "belief": None}


THINK_SYSTEM = (
    "You are the analysis module of a competitive game player. Read the game instructions and history carefully "
    "and reason step by step about the current situation and the best next move (under 200 words). "
    "Do NOT output a final bracketed action; another module will act on your analysis."
)


class CoT2Agent(_Base):
    arm = "cot2"

    def decide(self, obs, legal):
        tok = [0, 0]
        thought, r1 = chat(THINK_SYSTEM, f"{obs}\n\n=== CURRENT DECISION ===\n{legal}", {**self.meta, "call": "think"})
        _add(tok, r1)
        thought = BRACKET_RE.sub(lambda m: m.group(0)[1:-1], thought)  # no stray bracketed tokens
        self.block = f"=== YOUR ANALYSIS OF THIS TURN ===\n{thought}\n\n"
        text, r2 = self._act(obs, legal); _add(tok, r2)
        return extract_action(text), {"raw": text, "tokens": tok, "belief": None, "thought": thought}


EMPTY_BELIEF = {"opponent_likely_hidden_info": "unknown", "opponent_strategy_guess": "unknown",
                "opponent_bluff_history": [], "my_plan": ""}

BELIEF_SYSTEM = (
    "You are the opponent-modeling module of a competitive game player. You maintain an OPPONENT BELIEF BLOCK "
    "across turns. Given the game transcript so far, the current decision, and your previous belief block, output an "
    "UPDATED belief block as a single fenced JSON object and nothing else:\n"
    "```json\n{\"opponent_likely_hidden_info\": \"...\", \"opponent_strategy_guess\": \"...\", "
    "\"opponent_bluff_history\": [\"...\"], \"my_plan\": \"...\"}\n```\n"
    "- opponent_likely_hidden_info: best estimate of the opponent's CURRENT private information (this round's cards/dice), with reasons. "
    "Private info is re-dealt each round (card decks are reshuffled every round), so past rounds' cards/dice say NOTHING "
    "about the current hand; never reason by elimination from previously seen cards. Infer only from THIS round's actions "
    "and the opponent's behavioral tendencies.\n"
    "- opponent_strategy_guess: how the opponent tends to play (aggressive/passive/bluffs often/honest/patterns).\n"
    "- opponent_bluff_history: short list of the opponent's past claims/bets and whether they were revealed true or false.\n"
    "- my_plan: plan for the CURRENT decision and rest of this round only, consistent with the legal actions. Keep fields concise."
)

ROUND_MARKERS = ("New round", "### Starting round")


class BeliefAgent(_Base):
    arm = "belief"

    def __init__(self, meta):
        super().__init__(meta)
        self.belief = json.loads(json.dumps(EMPTY_BELIEF))
        self.rounds_seen = 0

    def decide(self, obs, legal):
        tok = [0, 0]
        rounds = sum(obs.count(m) for m in ROUND_MARKERS)
        if rounds != self.rounds_seen:  # new round: reset plan and per-round hidden-info guess
            self.rounds_seen = rounds
            self.belief["my_plan"] = ""
            self.belief["opponent_likely_hidden_info"] = "unknown (new round, re-dealt)"
        user1 = (f"{obs}\n\n=== CURRENT DECISION ===\n{legal}\n\n=== YOUR PREVIOUS BELIEF BLOCK (update it) ===\n"
                 f"```json\n{json.dumps(self.belief, indent=1)}\n```\n")
        text1, r1 = chat(BELIEF_SYSTEM, user1, {**self.meta, "call": "belief"}); _add(tok, r1)
        m = JSON_RE.search(text1) or re.search(r"(\{.*\})", text1, re.S)
        parsed = False
        if m:
            try:
                b = json.loads(m.group(1))
                if isinstance(b, dict):
                    self.belief = {k: b.get(k, self.belief.get(k)) for k in EMPTY_BELIEF}; parsed = True
            except json.JSONDecodeError:
                pass
        self.block = ("=== YOUR OPPONENT BELIEF BLOCK (maintained across turns; condition your decision on it) ===\n"
                      f"```json\n{json.dumps(self.belief, indent=1)}\n```\n\n")
        text, r2 = self._act(obs, legal); _add(tok, r2)
        return extract_action(text), {"raw": text, "tokens": tok, "belief": dict(self.belief), "belief_parsed": parsed}


ARMS = {"baseline": BaselineAgent, "cot2": CoT2Agent, "belief": BeliefAgent}
