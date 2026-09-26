"""OpenRouter client for the pilot experiments: tool-call support, usage logging, hard budget stop."""
import json, os, threading, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Users/sickle/Coding/context-research/.env")
RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)
USAGE = RUNS / "usage.jsonl"
BUDGET_USD = 9.5
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=120)
_lock = threading.Lock()

MODELS = ["qwen/qwen3-30b-a3b-instruct-2507", "openai/gpt-4.1-mini", "google/gemini-2.5-flash"]


def spent() -> float:
    if not USAGE.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in USAGE.read_text().splitlines() if l.strip())


class BudgetExceeded(RuntimeError):
    pass


def chat(model: str, messages: list, meta: dict, tools: list | None = None, temperature: float = 0.0,
         max_tokens: int = 1500) -> dict:
    """Returns {'text', 'tool_calls': [{'name','args'}], 'cost'}; logs a usage line. Retries transient errors."""
    if spent() > BUDGET_USD:
        raise BudgetExceeded(f"spent ${spent():.3f} > ${BUDGET_USD}")
    kw = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
              extra_body={"reasoning": {"enabled": False}, "usage": {"include": True}})
    if tools:
        kw["tools"] = tools
    last = None
    for attempt in range(5):
        try:
            r = _client.chat.completions.create(**kw)
            m = r.choices[0].message
            calls = []
            for tc in m.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                calls.append({"name": tc.function.name, "args": args})
            u = r.usage
            rec = {"ts": time.time(), "model": model, "prompt_tokens": u.prompt_tokens,
                   "completion_tokens": u.completion_tokens, "cost": float(getattr(u, "cost", 0.0) or 0.0), **meta}
            with _lock:
                with USAGE.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            return {"text": m.content or "", "tool_calls": calls, "cost": rec["cost"], "provider": getattr(r, "provider", None)}
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after retries: {last}")
