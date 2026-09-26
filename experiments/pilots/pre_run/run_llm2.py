"""Stronger LLM baseline (gpt-4.1-mini): zero-shot probability + the same 10 questions as JSON probabilities."""
import json, os, re, sys, time
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots"); import jevlib as J
from run import I, state, MQ, TAG, HERE
M = "openai/gpt-4.1-mini"
OUT = f"{HERE}/raw_llm2.jsonl"
done = {json.loads(l)["instance_id"] for l in open(OUT)} if os.path.exists(OUT) else set()
def one(r):
    s = state(r); head = f"Repository: {r['repo']}\n\nGitHub issue:\n{s['github_issue']}\n\n"; o = {"instance_id": r["instance_id"]}
    t0 = time.time()
    t, u = J.llm([{"role": "user", "content": head + "An autonomous AI coding agent will be given this issue and the repository and must write a patch that makes the hidden tests pass. "
        "What is the probability (0-100) that the agent resolves the issue? Reply with only a number."}], model=M, max_tokens=8, tag=TAG + "_llm2_zs")
    o["t_zs"] = time.time() - t0; m = re.search(r"\d+(\.\d+)?", t); o["zs"] = float(m.group()) / 100 if m else None
    qs = "\n".join(f'"{k}": {v["instructions"]}' for k, v in MQ.items())
    t0 = time.time()
    t, u = J.llm([{"role": "user", "content": head + "For each question below give the probability (0-100) that the answer is yes. Reply with only a JSON object mapping each key to a number.\n" + qs}],
                 model=M, max_tokens=200, tag=TAG + "_llm2_mq", response_format={"type": "json_object"})
    o["t_mq"] = time.time() - t0
    try: d = json.loads(re.search(r"\{.*\}", t, re.S).group()); o["mq"] = {k: float(d.get(k, 50)) / 100 for k in MQ}
    except Exception: o["mq"] = None
    return o
R = J.pmap(one, [r for r in I if r["instance_id"] not in done], workers=10)
with open(OUT, "a") as f:
    for r in R:
        if isinstance(r, dict): f.write(json.dumps(r) + "\n")
        else: print("ERR", r)
print("spend", J.spend(TAG + "_llm2"))
