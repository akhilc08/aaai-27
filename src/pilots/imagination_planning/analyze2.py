import json, collections, numpy as np
R = [json.loads(l) for l in open("stage2_episodes.jsonl")]
S = [json.loads(l) for l in open("../spend.jsonl") if json.loads(l)["tag"] == "imagination_planning"]
avg = {m: np.mean([s["cost"] or 0 for s in S if s["model"] == m]) for m in {s["model"] for s in S}}
cj, cl = avg["typesafe/jev-1.13"], avg["qwen/qwen3-30b-a3b-instruct-2507"]
print(f"avg $/call: jev {cj:.6f}, llm {cl:.6f}")
print("| arm | n games | success | mean steps | LLM calls/game | Jev calls/game | surprise notes/game | est $/game |\n|---|---|---|---|---|---|---|---|")
by = collections.defaultdict(list)
for r in R: by[r["arm"]].append(r)
for a, rs in sorted(by.items()):
    print(f"| {a} | {len(rs)} | {sum(r['won'] for r in rs)}/{len(rs)} | {np.mean([r['steps'] for r in rs]):.1f} | {np.mean([r['llm_calls'] for r in rs]):.1f} | "
          f"{np.mean([r['jev_calls'] for r in rs]):.1f} | {np.mean([r['surprises'] for r in rs]):.1f} | {np.mean([r['llm_calls']*cl + r['jev_calls']*cj for r in rs]):.4f} |")
W = {(r["game"], r["arm"]): r["won"] for r in R}; G = sorted({r["game"] for r in R})
for x, y in (("B", "A"), ("C", "A"), ("C", "B"), ("D", "A"), ("B", "D"), ("C", "D")):
    print(f"paired {x} vs {y}: {x}-only wins {sum(W[g,x] and not W[g,y] for g in G)}, {y}-only wins {sum(W[g,y] and not W[g,x] for g in G)}")
nh = {a: np.mean([np.mean([s['obs'] == 'Nothing happens.' for s in r['log']]) for r in rs]) for a, rs in by.items()}
print("Nothing-happens rate per arm:", {k: round(v, 3) for k, v in sorted(nh.items())})
rep = {a: np.mean([len(s:=[x['action'] for x in r['log']]) - len(set(s)) for r in rs]) for a, rs in by.items()}
print("repeated actions/game:", {k: round(v, 1) for k, v in sorted(rep.items())})
