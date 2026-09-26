import json, sys, numpy as np, collections
DS = sys.argv[1] if len(sys.argv) > 1 else "trajs"; SUF = "" if DS == "trajs" else "_" + DS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common import Q, KS
R = [json.loads(l) for l in open(f"raw{SUF}.jsonl")]
HF = ["n_err_obs", "n_edits", "n_repeat", "chars_obs", "reproduce"]
rng = np.random.default_rng(0)

def cv(X, y, g):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p

def boot(y, s, inst, B=500):
    u = np.unique(inst); idx = {v: np.where(inst == v)[0] for v in u}; out = []
    for _ in range(B):
        ii = np.concatenate([idx[v] for v in rng.choice(u, len(u))])
        if len(set(y[ii])) == 2: out.append(roc_auc_score(y[ii], s[ii]))
    return np.percentile(out, [2.5, 97.5])

def pairacc(s, inst, y):
    d = collections.defaultdict(dict)
    for a, b, c in zip(inst, y, s): d[a][b] = c
    v = [(1.0 if x[1] > x[0] else 0.5 if x[1] == x[0] else 0.0) for x in d.values() if len(x) == 2]
    return np.mean(v), len(v)

table = {}
for k in KS:
    rs = [r for r in R if r["k"] == k]
    y = np.array([r["y"] for r in rs]); g = np.array([r["repo"] for r in rs]); inst = np.array([r["instance_id"] for r in rs])
    H = np.array([[r[f] for f in HF] for r in rs], float)
    JQ = np.array([[r["jev"][q] for q in Q] for r in rs])
    llm = np.array([r["llm"] if r["llm"] is not None else 0.5 for r in rs])
    M = {"heur: -errors_so_far": -H[:, 0], "heur: logistic(5 feats)": cv(H, y, g),
         "LEAKY ref: final traj length (-)": -np.array([r["n_steps"] for r in rs], float),
         "LLM judge (qwen3-30b)": llm, "Jev zero-shot": JQ[:, 0],
         "Jev 8q + logistic": cv(JQ, y, g), "Jev 8q + heur logistic": cv(np.hstack([JQ, H]), y, g)}
    if rs[0].get("loo_rate") is not None: M["ORACLE ref: other-runs success rate"] = np.array([r["loo_rate"] for r in rs])
    row = {"n": len(y), "pos_rate": y.mean(), "instances": len(set(inst)), "repos": len(set(g))}
    for name, s in M.items():
        lo, hi = boot(y, s, inst); pa, npair = pairacc(s, inst, y)
        row[name] = {"auroc": roc_auc_score(y, s), "ci": [lo, hi], "pair_acc": pa, "n_pairs": npair}
    row["single_q_auroc"] = {q: roc_auc_score(y, JQ[:, j]) for j, q in enumerate(Q)}
    table[k] = row
json.dump(table, open(f"auroc_by_k{SUF}.json", "w"), indent=1)
lat_j = np.array([r["lat_jev"] for r in R]); lat_l = np.array([r["lat_llm"] for r in R])
print("latency median jev %.2fs llm %.2fs" % (np.median(lat_j), np.median(lat_l)))
print("mean prompt chars", np.mean([r["chars"] for r in R]))
for k, row in table.items():
    print(f"\nk={k} n={row['n']} pos={row['pos_rate']:.2f} inst={row['instances']} repos={row['repos']}")
    for name, v in row.items():
        if isinstance(v, dict) and "auroc" in v:
            print(f"  {name:34s} AUROC {v['auroc']:.3f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]  pair-acc {v['pair_acc']:.3f} (n={v['n_pairs']})")
    print("  single qs:", {q: round(a, 3) for q, a in row["single_q_auroc"].items()})
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = [n for n in table[1] if isinstance(table[1][n], dict) and "auroc" in table[1][n]]
    for n in names: plt.plot(KS, [table[k][n]["auroc"] for k in KS], marker="o", label=n)
    plt.axhline(0.5, color="gray", ls=":"); plt.xlabel("k (agent steps seen)"); plt.ylabel("AUROC"); plt.legend(fontsize=7); plt.savefig(f"auroc_vs_k{SUF}.png", dpi=120)
except ImportError:
    pass
