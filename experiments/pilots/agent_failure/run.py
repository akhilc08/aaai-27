"""Query Jev (8 noul questions, one call) and the cheap LLM judge on every (trajectory, k) prefix."""
import json, re, time, os
from common import *
DS = sys.argv[1] if len(sys.argv) > 1 else "trajs"; T = load(DS)
items = [(i, k) for i, t in enumerate(T) for k in KS if len(t["steps"]) > k]
print(len(items), "prefixes")
def one(ik):
    i, k = ik; t = T[i]; s = render(t, k)
    t0 = time.time(); a = J.jev(s, jevq(s), tag=TAG); tj = time.time() - t0
    t0 = time.time()
    txt, u = J.llm([{"role": "user", "content": s + "\n\nThe above is the beginning of a software agent's attempt to fix a GitHub issue. "
                     "Will this agent run end with a successful fix that passes the hidden tests? Reply with only a probability 0-100."}],
                   max_tokens=10, tag=TAG)
    tl = time.time() - t0
    m = re.search(r"\d+(\.\d+)?", txt)
    return {"i": i, "k": k, "instance_id": t["instance_id"], "repo": t["repo"], "y": t["y"], "n_steps": len(t["steps"]), "loo_rate": t.get("loo_rate"),
            "chars": len(s), "jev": {q: a[q]["noul"] for q in Q}, "llm": float(m.group()) / 100 if m else None,
            "llm_raw": txt, "lat_jev": tj, "lat_llm": tl, "cost_llm": u.get("cost", 0), **heur(t, k)}
res = J.pmap(one, items, workers=12)
ok = [r for r in res if isinstance(r, dict)]
print(len(ok), "ok;", [str(r)[:200] for r in res if not isinstance(r, dict)][:3])
with open("raw.jsonl" if DS == "trajs" else f"raw_{DS}.jsonl", "w") as f:
    for r in ok: f.write(json.dumps(r) + "\n")
print("spend", J.spend(TAG))
