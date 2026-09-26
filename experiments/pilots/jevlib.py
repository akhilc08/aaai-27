"""Shared helpers for the idea-validation pilots: Jev typed decisions and cheap LLM calls
through OpenRouter, with retries, thread-safe spend logging, and a parallel map."""
import json, os, ssl, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import certifi

_SSL = ssl.create_default_context(cafile=certifi.where())
ENV_FILE = "/Users/sickle/Coding/context-research/.env"
HERE = os.path.dirname(os.path.abspath(__file__))
SPEND_LOG = os.path.join(HERE, "spend.jsonl")
JEV = "typesafe/jev-1.13"
CHEAP = "qwen/qwen3-30b-a3b-instruct-2507"  # cheap LLM baseline used across pilots
_lock = threading.Lock()


def _load_env():
    if "OPENROUTER_API_KEY" in os.environ:
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
KEY = os.environ["OPENROUTER_API_KEY"]


def _post(url, body, timeout=120, retries=6):
    data = json.dumps(body).encode()
    for i in range(retries):
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, context=_SSL, timeout=timeout))
        except urllib.error.HTTPError as e:
            msg = e.read()[:400]
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 ** i)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except Exception:
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            raise


def _log(tag, model, cost):
    with _lock, open(SPEND_LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), "tag": tag, "model": model, "cost": cost}) + "\n")


GLOBAL_CAP = 24.50  # hard stop on total pilot spend across all processes (user budget is $25)
_cache = {"t": 0.0, "total": 0.0}


def _check_cap():
    now = time.time()
    if now - _cache["t"] > 20:
        _cache["total"] = spend("")
        _cache["t"] = now
    if _cache["total"] >= GLOBAL_CAP:
        raise RuntimeError(f"GLOBAL SPEND CAP reached (${_cache['total']:.2f} >= ${GLOBAL_CAP}). Stop making API calls and write up what you have.")


def jev(state, questions, tag="jev"):
    """state: str or JSON-able dict. questions: {name: {type: noul|choice|score, instructions, criteria?}}.
    choice criteria = {option: description}; score criteria = [ordered labels].
    Returns the answers dict, e.g. answers[name]["noul"] or ["probabilities"]."""
    _check_cap()
    r = _post("https://openrouter.ai/api/alpha/decisions",
              {"model": JEV, "state": state, "questions": questions})
    _log(tag, JEV, (r.get("usage") or {}).get("cost", 0))
    return r["answers"]


def llm(messages, model=CHEAP, max_tokens=300, temperature=0.0, tag="llm", **extra):
    """Returns (text, usage). usage includes cost when OpenRouter reports it."""
    _check_cap()
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "usage": {"include": True}, **extra}
    r = _post("https://openrouter.ai/api/v1/chat/completions", body)
    u = r.get("usage") or {}
    _log(tag, model, u.get("cost", 0))
    return (r["choices"][0]["message"].get("content") or ""), u


def pmap(fn, items, workers=8):
    """Parallel map that returns results in order; exceptions are returned as values."""
    def safe(x):
        try:
            return fn(x)
        except Exception as e:
            return e
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(safe, items))


def spend(tag_prefix=""):
    if not os.path.exists(SPEND_LOG):
        return 0.0
    total = 0.0
    for l in open(SPEND_LOG):
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d["tag"].startswith(tag_prefix):
            total += d.get("cost") or 0
    return total
