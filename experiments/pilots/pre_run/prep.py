"""Build issues.jsonl (SWE-bench Verified) and labels.json ({system: {iid: 0/1}}, plus per-instance cost when present)."""
import json, glob, os
from datasets import load_dataset
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
iss = [{k: r[k] for k in ("instance_id", "repo", "problem_statement", "hints_text", "difficulty", "created_at")} for r in ds]
ids = [r["instance_id"] for r in iss]
with open(f"{D}/issues.jsonl", "w") as f:
    for r in iss: f.write(json.dumps(r) + "\n")
L, C = {}, {}
for fn in sorted(glob.glob(f"{D}/res/*.json")):
    name = os.path.basename(fn).replace(".pid.json", "").replace(".json", "")
    d = json.load(open(fn))
    if "resolved" in d and isinstance(d["resolved"], list):
        s = set(d["resolved"]); L[name] = {i: int(i in s) for i in ids}
    else:
        L[name] = {i: int(bool(d.get(i, {}).get("resolved"))) for i in ids}
        C[name] = {i: d.get(i, {}).get("cost") for i in ids}
json.dump({"labels": L, "cost": C}, open(f"{D}/labels.json", "w"))
for k, v in sorted(L.items(), key=lambda kv: sum(kv[1].values())): print(f"{k:60s} {sum(v.values())/500:.3f}")
