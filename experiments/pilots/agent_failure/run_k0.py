"""Issue-only (k=0) control: how much of the signal is just issue difficulty?"""
import json, re, time
from common import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
import numpy as np
T = load("trajs_natural")
def one(t):
    s = render(t, 0)
    a = J.jev(s, {"succ": {"type": "noul", "instructions": "An autonomous software agent will attempt to fix this GitHub issue. Will it produce a correct fix that passes the hidden tests?"},
                  **{q: v for q, v in jevq(s).items() if q in ("located",)}}, tag=TAG)
    txt, u = J.llm([{"role": "user", "content": s + "\n\nAn autonomous software agent (Llama-70B SWE-agent) will attempt to fix this GitHub issue. "
                     "Will it produce a successful fix that passes the hidden tests? Reply with only a probability 0-100."}], max_tokens=10, tag=TAG)
    m = re.search(r"\d+(\.\d+)?", txt)
    return {"instance_id": t["instance_id"], "y": t["y"], "jev": a["succ"]["noul"], "llm": float(m.group()) / 100 if m else 0.5}
R = [r for r in J.pmap(one, T, workers=12) if isinstance(r, dict)]
with open("raw_k0_natural.jsonl", "w") as f:
    for r in R: f.write(json.dumps(r) + "\n")
y = np.array([r["y"] for r in R])
out = {"n": len(R), "jev_issue_only": roc_auc_score(y, [r["jev"] for r in R]), "llm_issue_only": roc_auc_score(y, [r["llm"] for r in R]),
       "issue_chars(-)": roc_auc_score(y, [-len(t["issue"]) for t in T[:len(R)]])}
print(out); json.dump(out, open("auroc_k0_natural.json", "w"), indent=1); print("spend", J.spend(TAG))
