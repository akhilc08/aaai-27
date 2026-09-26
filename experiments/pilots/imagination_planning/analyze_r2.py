import json, glob, collections, numpy as np
from scipy.stats import binomtest
R = [json.loads(l) for f in sorted(glob.glob("round2_seed*.jsonl")) for l in open(f)]
R = [r for r in R if "won" in r]
S = [json.loads(l) for l in open("../spend.jsonl") if json.loads(l)["tag"] == "imagination_planning"]
cj = np.mean([s["cost"] or 0 for s in S if s["model"].startswith("typesafe")])
cl = np.mean([s["cost"] or 0 for s in S[-3000:] if s["model"].startswith("qwen")])
key = lambda r: (r["game"], r["seed"])
W = {(key(r), r["arm"]): r["won"] for r in R}
full = [k for k in {key(r) for r in R} if all((k, a) in W for a in "DBCV")]
print(f"complete game-seed units: {len(full)}; errors/skips excluded")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
print("| arm | success | 95% CI | steps | override rate | surprise notes/ep | LLM calls | Jev calls | est $/ep | wall s/ep |\n|---|---|---|---|---|---|---|---|---|---|")
for a in "DBCV":
    rs = [r for r in R if r["arm"] == a and key(r) in set(full)]; k = sum(r["won"] for r in rs); n = len(rs); lo, hi = wilson(k, n)
    st = sum(r["steps"] for r in rs)
    print(f"| {a} | {k}/{n} = {k/n:.3f} | [{lo:.3f}, {hi:.3f}] | {st/n:.1f} | {sum(r['overrides'] for r in rs)/st:.3f} | {np.mean([r['surprises'] for r in rs]):.2f} | "
          f"{np.mean([r['llm_calls'] for r in rs]):.1f} | {np.mean([r['jev_calls'] for r in rs]):.1f} | {np.mean([r['llm_calls']*cl + r['jev_calls']*cj for r in rs]):.4f} | {np.mean([r['wall'] for r in rs]):.0f} |")
for x, y in (("B", "D"), ("C", "B"), ("B", "V"), ("V", "D"), ("C", "D")):
    b = sum(W[k, x] and not W[k, y] for k in full); c = sum(W[k, y] and not W[k, x] for k in full)
    print(f"{x} vs {y}: {x}-only {b}, {y}-only {c}, exact McNemar p={binomtest(b, b + c).pvalue if b + c else 1:.3f}")
print(f"avg $/call jev {cj:.6f} llm {cl:.6f}")
