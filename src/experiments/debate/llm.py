"""OpenRouter client wrapper with per-request usage logging and a hard budget stop."""
import json, os, threading, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Users/sickle/Coding/context-research/.env")
RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)
USAGE = RUNS / "usage.jsonl"
BUDGET_USD = 1.50
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120)
_lock = threading.Lock()


def spent() -> float:
    if not USAGE.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in USAGE.read_text().splitlines() if l.strip())


class BudgetExceeded(RuntimeError):
    pass


def chat(model: str, messages: list, temperature: float, meta: dict, max_tokens: int = 4000) -> dict:
    """Returns {'text', 'prompt_tokens', 'completion_tokens', 'cost'}; logs usage line. Retries transient errors."""
    if spent() > BUDGET_USD:
        raise BudgetExceeded(f"spent ${spent():.3f} > ${BUDGET_USD}")
    last = None
    for attempt in range(5):
        try:
            r = _client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                extra_body={"reasoning": {"enabled": False}, "usage": {"include": True}},
            )
            text = r.choices[0].message.content or ""
            u = r.usage
            rec = {"ts": time.time(), "model": model, "prompt_tokens": u.prompt_tokens,
                   "completion_tokens": u.completion_tokens, "cost": float(getattr(u, "cost", 0.0) or 0.0), **meta}
            with _lock:
                with USAGE.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            return {"text": text, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "cost": rec["cost"]}
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after retries: {last}")
