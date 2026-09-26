import json, re, time
from common import *
S = json.load(open("data/sample1500.json"))[:150]
def one(x):
    opts = options(x)
    prompt = (render(x) + "\n\nPredict the OPPONENT's action this turn. It either uses one of its active Pokemon's moves "
              "(possibly one not yet revealed) or switches. Reply with ONLY a JSON object: "
              '{"p_switch": <probability 0-1 the opponent switches>, "probs": {<option>: <probability>, ...}} where the options are exactly: '
              + json.dumps(opts) + ' plus "OTHER_MOVE" for a not-yet-revealed move. Probabilities should sum to 1.')
    t = time.time()
    txt, u = J.llm([{"role": "user", "content": prompt}], max_tokens=250, tag=TAG)
    dt = time.time() - t
    m = re.search(r"\{.*\}", txt, re.S)
    try: d = json.loads(m.group(0))
    except Exception: d = {}
    return {"battle": x["battle"], "turn": x["turn"], "me": x["me"], "label": x["label"], "lat": dt,
            "cost": u.get("cost", 0), "p_sw": d.get("p_switch"), "probs": d.get("probs"), "raw": txt}
R = J.pmap(one, S, workers=8)
with open("llm_out.jsonl", "w") as f:
    for r in R:
        if isinstance(r, Exception): print("ERR", r); continue
        f.write(json.dumps(r) + "\n")
print("spend", J.spend(TAG), "parsed", sum(1 for r in R if not isinstance(r, Exception) and r["p_sw"] is not None))
