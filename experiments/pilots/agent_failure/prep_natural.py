"""Natural-distribution sample: one random llama-70b run per instance (no balancing), 400 instances."""
import json, random, pyarrow.parquet as pq, pandas as pd
df = pd.concat([pq.read_table(f"data/t{i}.parquet").to_pandas() for i in ("00", "03", "06", "09")])
df = df[df.model_name == "swe-agent-llama-70b"]; df["target"] = df.target.astype(str) == "True"
out = []
for iid, g in df.groupby("instance_id"):
    r = g.sample(1, random_state=2).iloc[0]
    others = g.drop(r.name) if False else g[g.index != r.name]
    traj = list(r.trajectory); issue = next(m["text"] for m in traj if m["role"] == "user")
    msgs = [m for m in traj if m["role"] in ("ai", "user")][1:]
    steps = [{"act": m["text"], "obs": msgs[j+1]["text"] if j+1 < len(msgs) and msgs[j+1]["role"] == "user" else ""}
             for j, m in enumerate(msgs) if m["role"] == "ai"]
    loo = (g.target.sum() - r.target) / max(len(g) - 1, 1)
    out.append({"instance_id": iid, "repo": iid.rsplit("-", 1)[0], "y": int(r.target), "loo_rate": loo,
                "issue": issue, "steps": steps})
random.seed(0); random.shuffle(out); out = out[:500]
print(len(out), "pos rate", sum(o["y"] for o in out) / len(out), "repos", len({o["repo"] for o in out}))
with open("data/trajs_natural.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
