import json, random
import wm
J = wm.J
R = [json.loads(l) for l in open("stage1_transitions.jsonl")]
jp = J.pmap(wm.jev_predict, R, workers=12)
sub = set(random.Random(1).sample(range(len(R)), 300))
lp = J.pmap(lambda i: wm.llm_predict(R[i]) if i in sub else None, range(len(R)), workers=12)
with open("stage1_preds.jsonl", "w") as f:
    for r, a, b in zip(R, jp, lp):
        f.write(json.dumps({"game": r["game"], "t": r["t"], "action": r["action"], "src": r["src"], "ep_won": r["ep_won"],
            "in_adm": r["action"] in r["adm"], "y": wm.labels(r),
            "jev": None if isinstance(a, Exception) else a, "llm": None if isinstance(b, Exception) or b is None else b}) + "\n")
print("jev errs", sum(isinstance(a, Exception) for a in jp), [str(a) for a in jp if isinstance(a, Exception)][:2])
print("llm errs", sum(isinstance(b, Exception) for b in lp))
print("spend", J.spend(wm.A.TAG))
