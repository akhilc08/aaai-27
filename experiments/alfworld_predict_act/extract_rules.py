"""Arm C step 1: from arm A failures, ask the model for one reusable rule per failure, keyed by task type -> runs/rules.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

eps = [json.loads(l) for l in (RUNS / "arm_A.jsonl").read_text().splitlines() if l.strip()]
rules = {}
for e in eps:
    if e["success"]: continue
    traj = "\n".join(f"> {t['agent']}\n{t['obs']}" for t in e["transcript"])[-6000:]
    task = e["transcript"][0]["agent"] if e["transcript"] else ""
    msgs = [{"role": "user", "content":
             f"An agent failed a household task of type '{e['task_type']}' in a text game (ALFWorld). Here is the end of its trajectory:\n\n{traj}\n\n"
             "Write ONE short, reusable rule (max 25 words) that would help an agent avoid this failure on OTHER tasks of the same type. "
             "It must be general (do not mention specific object numbers or this room). Output only the rule."}]
    txt, u = llm(msgs, tag="rule", max_tokens=60)
    rules.setdefault(e["task_type"], []).append(txt.strip().strip('"'))
    print(e["task_type"], "->", txt.strip())
(RUNS / "rules.json").write_text(json.dumps(rules, indent=2))
print(json.dumps(rules, indent=2))
