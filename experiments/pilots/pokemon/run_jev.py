import json, random, time
from common import *
S = json.load(open("data/all_samples.json"))
random.Random(1).shuffle(S); S = S[:1500]
Q_FEAT = {
 "f_disadv": "Is the opponent's active Pokemon at a type disadvantage against your active Pokemon (your active Pokemon's likely attacks hit it super-effectively, or it cannot hurt yours)?",
 "f_koed": "Is the opponent's active Pokemon at risk of being knocked out by your active Pokemon this turn?",
 "f_opphits": "Does the opponent's active Pokemon threaten your active Pokemon with a strong or super-effective attack?",
 "f_setup": "Does the opponent's active Pokemon have a setup, boosting, or status move it is likely to use?",
 "f_faster": "Is the opponent's active Pokemon likely faster than your active Pokemon?",
 "f_goodswitch": "Does the opponent likely have a benched Pokemon that matches up well against your active Pokemon?",
 "f_sack": "Is the opponent's active Pokemon low value now (low HP or job done), so the opponent would keep it in rather than preserve it?",
}
def one(x):
    crit = {key(m): f"The opponent's active Pokemon uses the move {m}" for m in x["revealed_moves"]}
    crit["switch"] = "The opponent switches its active Pokemon out for a teammate"
    q = {"sw": {"type": "noul", "instructions": "Will the opponent switch out their active Pokemon this turn instead of using a move?"}}
    q.update({k: {"type": "noul", "instructions": v} for k, v in Q_FEAT.items()})
    if len(crit) >= 2: q["act"] = {"type": "choice", "instructions": "Which action will the opponent take this turn?", "criteria": crit}
    t = time.time(); a = J.jev(render(x), q, tag=TAG); dt = time.time() - t
    return {"battle": x["battle"], "turn": x["turn"], "me": x["me"], "label": x["label"], "lat": dt,
            "p_sw": a["sw"]["noul"], **{k: a[k]["noul"] for k in Q_FEAT},
            "act_probs": a.get("act", {}).get("probabilities")}
t0 = time.time()
R = J.pmap(one, S, workers=10)
with open("jev_out.jsonl", "w") as f:
    for x, r in zip(S, R):
        if isinstance(r, Exception): print("ERR", r); continue
        f.write(json.dumps(r) + "\n")
json.dump(S, open("data/sample1500.json", "w"))
print("done", time.time() - t0, "spend", J.spend(TAG))
