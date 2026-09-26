"""Round 3: pooled McNemar + cluster-aware (per-game) tests; J and R arms vs round-2 seed-0 D/B."""
import json, glob, collections, numpy as np
from scipy.stats import binomtest, wilcoxon
def load(files): return [r for f in files for l in open(f) for r in [json.loads(l)] if "won" in r]
def compare(R, x, y, label):
    W = {(r["game"], r["seed"], r["arm"]): r["won"] for r in R}
    units = sorted({(g, s) for g, s, a in W if (g, s, x) in W and (g, s, y) in W})
    b = sum(W[u + (x,)] and not W[u + (y,)] for u in units); c = sum(W[u + (y,)] and not W[u + (x,)] for u in units)
    per = collections.defaultdict(list)
    for g, s in units: per[g].append(W[(g, s, x)] - W[(g, s, y)])
    d = np.array([np.mean(v) for v in per.values()])
    rng = np.random.default_rng(0); boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    px = np.mean([W[u + (x,)] for u in units]); py = np.mean([W[u + (y,)] for u in units])
    wp = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() else 1
    print(f"| {label} | {x} {px:.3f} vs {y} {py:.3f} | {len(units)} units / {len(d)} games | {b}:{c}, p={binomtest(b, b + c).pvalue:.4f} | "
          f"{d.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}] | {wp:.4f} |")
print("| comparison | success | n | McNemar (x-only:y-only) | per-game diff, cluster-bootstrap 95% CI | Wilcoxon per-game p |\n|---|---|---|---|---|---|")
small = load(sorted(glob.glob("round2_seed[0-9].jsonl")))
compare(small, "B", "D", "qwen3-30b, 3 seeds (seed 2 partial)")
compare(small, "B", "V", "qwen3-30b")
compare(small, "C", "B", "qwen3-30b")
strong = load(["round2_strong_all.jsonl"] + sorted(glob.glob("round3_strong_seed*.jsonl")))
compare(strong, "B", "D", "qwen3-235b, all seeds")
jr = load(["round3_JR.jsonl"]) + [r for r in load(["round2_seed0.jsonl"])]
compare(jr, "J", "D", "Jev alone vs 30b D (seed 0)")
compare(jr, "J", "B", "Jev alone vs 30b B (seed 0)")
compare(jr, "R", "D", "rule vs 30b D (seed 0)")
compare(jr, "B", "R", "30b B vs rule (seed 0)")
for a in "JR":
    rs = [r for r in jr if r["arm"] == a]; print(f"{a}: {sum(r['won'] for r in rs)}/{len(rs)}, steps {np.mean([r['steps'] for r in rs]):.1f}, "
        f"override {sum(r['overrides'] for r in rs)/sum(r['steps'] for r in rs):.3f}, jev calls/ep {np.mean([r['jev_calls'] for r in rs]):.1f}, llm calls/ep {np.mean([r['llm_calls'] for r in rs]):.1f}")
for s in sorted({r["seed"] for r in strong}):
    for a in "DB":
        rs = [r for r in strong if r["seed"] == s and r["arm"] == a]; print(f"strong seed {s} {a}: {sum(r['won'] for r in rs)}/{len(rs)}")
