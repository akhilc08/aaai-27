"""Sample balanced success/fail pairs per instance (llama-70b runs) and build leakage-free step lists."""
import json, random, pyarrow.parquet as pq, pandas as pd
random.seed(0)
df = pd.concat([pq.read_table(f"data/t{i}.parquet").to_pandas() for i in ("00", "03", "06", "09")])
df = df[df.model_name == "swe-agent-llama-70b"]
df["target"] = df.target.astype(str) == "True"
out = []
for iid, g in df.groupby("instance_id"):
    pos, neg = g[g.target], g[~g.target]
    if len(pos) == 0 or len(neg) == 0:
        continue
    rate = g.target.mean()
    for r in (pos.sample(1, random_state=1).iloc[0], neg.sample(1, random_state=1).iloc[0]):
        traj = list(r.trajectory)
        issue = next(m["text"] for m in traj if m["role"] == "user")
        steps, i = [], 0
        msgs = [m for m in traj if m["role"] in ("ai", "user")][1:]  # drop issue message
        for j, m in enumerate(msgs):
            if m["role"] == "ai":
                obs = msgs[j + 1]["text"] if j + 1 < len(msgs) and msgs[j + 1]["role"] == "user" else ""
                steps.append({"act": m["text"], "obs": obs})
        out.append({"instance_id": iid, "repo": iid.rsplit("-", 1)[0], "y": int(r.target),
                    "n_runs": len(g), "inst_rate": rate, "exit_status": r.exit_status,
                    "issue": issue, "steps": steps})
random.shuffle(out)
print(len(out), "trajs", len({o['instance_id'] for o in out}), "instances", len({o['repo'] for o in out}), "repos")
import numpy as np
L = np.array([len(o["steps"]) for o in out]); y = np.array([o["y"] for o in out])
print("steps pos median", np.median(L[y == 1]), "neg median", np.median(L[y == 0]), "frac<=10", (L <= 10).mean())
with open("data/trajs.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
