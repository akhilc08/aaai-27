"""Shared helpers: ALFWorld TextWorld env, OpenRouter LLM client, usage logging."""
import os, json, time, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("ALFWORLD_DATA", str(ROOT / "data"))
RUNS = ROOT / "runs"; RUNS.mkdir(exist_ok=True)

from dotenv import load_dotenv
load_dotenv("/Users/sickle/Coding/context-research/.env")

import yaml, textworld, textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos
from openai import OpenAI

# ---------------- config ----------------
PRIMARY_MODEL = os.environ.get("AGENT_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
FALLBACK_MODELS = [m for m in os.environ.get("AGENT_FALLBACKS", "deepseek/deepseek-chat,z-ai/glm-4.5-air").split(",") if m and m != PRIMARY_MODEL]
# OpenRouter list prices ($/M tokens) for cost estimation
PRICES = {
    "qwen/qwen3-30b-a3b-instruct-2507": (0.10, 0.30),
    "deepseek/deepseek-chat": (0.32, 0.89),
    "z-ai/glm-4.5-air": (0.13, 0.85),
}
BUDGET_USD = 9.50  # hard cumulative cap for this folder (sum of cost in runs/usage.jsonl)
MAX_STEPS = 30
N_GAMES = 25
SEED = 0

PREFIXES = {
    "pick_and_place": "put", "pick_clean_then_place": "clean", "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool", "look_at_obj": "examine", "pick_two_obj": "puttwo",
}
def task_type(gamefile: str) -> str:
    for k, v in PREFIXES.items():
        if k in gamefile: return v
    return "put"

# ---------------- env ----------------
def load_config():
    cfg = yaml.safe_load(open(ROOT / "configs" / "base_config.yaml"))
    return cfg

def select_games(n=N_GAMES, seed=SEED):
    cfg = load_config()
    env = AlfredTWEnv(cfg, train_eval="eval_out_of_distribution")
    files = sorted(env.game_files)
    assert len(files) == 134, len(files)
    rng = random.Random(seed)
    if n >= len(files): return rng.sample(files, len(files))
    return random.Random(seed + 1).sample(files, len(files))[:n] if os.environ.get("PREFIX_ORDER") else rng.sample(files, n)

def make_env(gamefile: str):
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games([gamefile], request_infos, batch_size=1,
                                          asynchronous=False, max_episode_steps=MAX_STEPS + 5,
                                          wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(env_id)

def clean_obs(obs: str) -> str:
    # strip the ASCII banner textworld prints at reset
    if "-= Welcome" in obs:
        obs = obs[obs.find("Looking quickly"):] if "Looking quickly" in obs else obs
        obs = "You are in the middle of a room. " + obs
    return obs.strip()

# ---------------- LLM ----------------
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
USAGE_LOG = RUNS / "usage.jsonl"

class BudgetExceeded(RuntimeError): pass

def total_spend() -> float:
    if not USAGE_LOG.exists(): return 0.0
    return sum(json.loads(l)["cost"] for l in USAGE_LOG.read_text().splitlines() if l.strip())

def llm(messages, tag="", max_tokens=300, temperature=0.0, stop=None):
    """Chat completion with fallback, usage logging, and budget guard. Returns (text, usage)."""
    if total_spend() > BUDGET_USD:
        raise BudgetExceeded(f"spend {total_spend():.3f} > {BUDGET_USD}")
    last_err = None
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        for attempt in range(3):
            try:
                r = _client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                    temperature=temperature, stop=stop, seed=SEED)
                text = (r.choices[0].message.content or "").strip()
                u = r.usage
                pin, pout = PRICES.get(model, (1.0, 1.0))
                cost = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6
                rec = {"t": time.time(), "tag": tag, "model": model, "prompt_tokens": u.prompt_tokens,
                       "completion_tokens": u.completion_tokens, "cost": cost}
                with open(USAGE_LOG, "a") as f: f.write(json.dumps(rec) + "\n")
                return text, rec
            except Exception as e:  # noqa
                last_err = e; time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"all models failed: {last_err}")
