import json, re, collections, numpy as np
from sklearn.metrics import roc_auc_score
P = [json.loads(l) for l in open("stage1_preds.jsonl")]
Q = ("effect", "reveal", "empty")
def verb(p): return p["action"].split()[0] if p["action"] else ""
def vrec(p):
    m = re.match(r"(go to|open) ([a-z]+)", p["action"]); return (m.group(1) + m.group(2)) if m else verb(p)
def logo_prior(q, key):  # leave-one-game-out base rate by key, backoff to global
    out = []
    for p in P:
        o = [x["y"][q] for x in P if x["game"] != p["game"] and key(x) == key(p)]
        g = [x["y"][q] for x in P if x["game"] != p["game"]]
        out.append((sum(o) + np.mean(g)) / (len(o) + 1))
    return np.array(out)
def ece(y, p, b=10):
    y, p = np.array(y), np.array(p); idx = np.minimum((p * b).astype(int), b - 1)
    return sum(abs(y[idx == i].mean() - p[idx == i].mean()) * (idx == i).sum() for i in range(b) if (idx == i).any()) / len(y)
def row(name, y, p):
    y, p = np.array(y), np.clip(np.array(p, float), 1e-3, 1 - 1e-3)
    return f"| {name} | {len(y)} | {y.mean():.2f} | {roc_auc_score(y, p):.3f} | {np.mean((p - y) ** 2):.3f} | {ece(y, p):.3f} |"
lines = []
for q in Q:
    y = [p["y"][q] for p in P]
    lines.append(f"\n### {q}\n| predictor | n | base rate | AUROC | Brier | ECE |\n|---|---|---|---|---|---|")
    lines.append(row("Jev (all)", y, [p["jev"][q] for p in P]))
    lines.append(row("prior: verb (LOGO)", y, logo_prior(q, verb)))
    lines.append(row("prior: verb+receptacle (LOGO)", y, logo_prior(q, vrec)))
    if q == "effect": lines.append(row("rule: action in admissible list", y, [0.98 if p["in_adm"] else 0.02 for p in P]))
    S = [p for p in P if p["llm"]]
    ys = [p["y"][q] for p in S]
    lines.append(row("Jev (LLM subsample)", ys, [p["jev"][q] for p in S]))
    lines.append(row("qwen3-30b LLM (subsample)", ys, [p["llm"][q] for p in S]))
# reveal restricted to go/open actions (the hidden-state case)
G = [p for p in P if verb(p) in ("go", "open")]
lines.append("\n### reveal, only go/open actions (hidden-state prediction)\n| predictor | n | base rate | AUROC | Brier | ECE |\n|---|---|---|---|---|---|")
lines.append(row("Jev", [p["y"]["reveal"] for p in G], [p["jev"]["reveal"] for p in G]))
Gs = [p for p in G if p["llm"]]
lines.append(row("Jev (LLM subsample)", [p["y"]["reveal"] for p in Gs], [p["jev"]["reveal"] for p in Gs]))
lines.append(row("qwen3-30b (subsample)", [p["y"]["reveal"] for p in Gs], [p["llm"]["reveal"] for p in Gs]))
# reliability bins for Jev effect & reveal
for q in ("effect", "reveal"):
    y = np.array([p["y"][q] for p in P]); pr = np.array([p["jev"][q] for p in P]); idx = np.minimum((pr * 5).astype(int), 4)
    lines.append(f"\nJev reliability [{q}] (bin: n, mean pred, freq): " + "; ".join(
        f"{i/5:.1f}-{(i+1)/5:.1f}: {(idx==i).sum()}, {pr[idx==i].mean():.2f}, {y[idx==i].mean():.2f}" for i in range(5) if (idx == i).any()))
# surprise vs episode failure
eps = collections.defaultdict(list)
for p in P: eps[p["game"]].append(p)
def surprise(p, src="jev"):
    return np.mean([-np.log(np.clip(p[src][q] if p["y"][q] else 1 - p[src][q], 1e-3, 1)) for q in Q])
for K in (5, 10, 25):
    fail = [0 if v[0]["ep_won"] else 1 for v in eps.values()]
    s = [np.mean([surprise(p) for p in sorted(v, key=lambda p: p["t"])[:K]]) for v in eps.values()]
    nh = [np.mean([p["y"]["effect"] == 0 for p in sorted(v, key=lambda p: p["t"])[:K]]) for v in eps.values()]
    lines.append(f"\nSurprise->failure, first {K} steps (n_eps={len(fail)}, fail={sum(fail)}): AUROC Jev mean surprise={roc_auc_score(fail, s):.3f}; "
                 f"AUROC raw 'Nothing happens' rate={roc_auc_score(fail, nh):.3f}")
open("stage1_analysis.txt", "w").write("\n".join(lines)); print("\n".join(lines))
