import json, numpy as np, collections as C
from sklearn.metrics import roc_auc_score as A
from sklearn.linear_model import LogisticRegression
R = [json.loads(l) for l in open("features.jsonl")]
y = np.array([r["y"] for r in R]); sc = np.array([r["scenario"] for r in R]); bbf = np.array([r["backbone"] == "qwen" for r in R], float)
F = ["lost", "topic", "prohib", "cond", "know_refuse", "generic"]
lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
X = np.c_[lg(np.array([[r[f] for f in F] for r in R])), bbf]
jev0 = np.array([r["jev0_lost"] for r in R]); kw = np.array([not r["rule_kw_present"] for r in R], float)
llm = np.array([r["llm_lost"] for r in R])
def thr_for_recall(s, yy, rec):
    t = np.sort(s[yy == 1])[::-1]; return t[int(np.ceil(rec * len(t))) - 1]
def loso(Xf, rec=None, raw=None):
    """returns held-out scores and (optionally) gates with thresholds chosen on training scenarios"""
    s = np.zeros(len(y)); g = {r_: np.zeros(len(y), bool) for r_ in (rec or [])}
    for sid in np.unique(sc):
        te, tr = sc == sid, sc != sid
        if raw is None:
            m = LogisticRegression(max_iter=2000).fit(Xf[tr], y[tr]); s[te] = m.predict_proba(Xf[te])[:, 1]; st = m.predict_proba(Xf[tr])[:, 1]
        else:
            s[te] = raw[te]; st = raw[tr]
        for r_ in g: g[r_][te] = s[te] >= thr_for_recall(st, y[tr], r_)
    return s, g
REC = [0.8, 0.9, 0.95]
head, hg = loso(X, REC); _, zg = loso(None, REC, raw=jev0)
# ---------- part 3: held-out wording
P = {d["row"]: d for d in map(json.loads, open("rephrase_features.jsonl"))}
Xn = np.c_[lg(np.array([[P[i][f] for f in F] for i in range(len(R))])), bbf]
hn = np.zeros(len(y))
for sid in np.unique(sc):
    te, tr = sc == sid, sc != sid
    hn[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(Xn[te])[:, 1]  # trained on ORIGINAL wording
print("== held-out wording (n=%d)" % len(y))
print(" jev 'lost' single q: orig %.3f  new %.3f" % (A(y, X[:, 0]), A(y, Xn[:, 0])))
print(" jev head LOSO, trained orig wording: tested orig %.3f  tested new %.3f" % (A(y, head), A(y, hn)))
# ---------- part 2: continuous judges
J_ = [json.loads(l) for l in open("judge.jsonl")]
rows = sorted({d["row"] for d in J_}); ys = y[rows]
print("== continuous judges on stratified n=%d, base rate %.3f" % (len(rows), ys.mean()))
for jn in ["qwen3-30b", "gpt-4.1-mini"]:
    for pn in ["lost", "viol"]:
        d = {x["row"]: x for x in J_ if x["judge"] == jn and x["prompt"] == pn}
        s = np.array([d[i]["p"] if d[i]["p"] is not None else 0.5 for i in rows])
        print(f" {jn:13s} {pn}: AUROC {A(ys, s):.3f}  parsefail {sum(d[i]['p'] is None for i in rows)}  $/1k {1000*np.mean([d[i]['cost'] for i in rows]):.3f}  lat {np.mean([d[i]['lat'] for i in rows]):.2f}s  distinct {len(set(s))}")
for nm, s in [("keyword", kw), ("llm binary", llm), ("jev zero-shot", jev0), ("jev head LOSO", head)]:
    print(f" {nm:13s}: AUROC {A(ys, s[rows]):.3f}")
# ---------- part 1: intervention
try:
    I = [json.loads(l) for l in open("intervene.jsonl")]
except FileNotFoundError:
    raise SystemExit("no intervene.jsonl yet")
D = C.defaultdict(dict)
for d in I: D[(d["row"], d["rep"])][d["pin"]] = d
keys = [k for k, v in D.items() if 0 in v and 1 in v and v[0]["decision"] != "UNKNOWN" and v[1]["decision"] != "UNKNOWN"]
print("== intervention: %d (row,rep) pairs usable of %d; UNKNOWN dropped" % (len(keys), len(D)))
ri = np.array([k[0] for k in keys]); p0 = np.array([D[k][0]["decision"] == "COMPLY" for k in keys], float)
p1 = np.array([D[k][1]["decision"] == "COMPLY" for k in keys], float)
dt = np.array([D[k][1]["prompt_tokens"] - D[k][0]["prompt_tokens"] for k in keys], float)
rng = np.random.default_rng(0)
gates = {"a. plain": np.zeros(len(y), bool), "b. always-pin": np.ones(len(y), bool), "c. keyword-gated": kw.astype(bool),
         "e. LLM self-judge-gated": llm.astype(bool), "d0. Jev zero-shot @0.5": jev0 >= 0.5}
for r_ in REC: gates[f"d0. Jev zero-shot @train-recall{r_}"] = zg[r_]
for r_ in REC: gates[f"d. Jev head LOSO @train-recall{r_}"] = hg[r_]
gates["random gate @ Jev-head-0.9 rate"] = rng.random(len(y)) < hg[0.9].mean()
gates["(upper bnd) gate = original-run label"] = y.astype(bool)
out = []
for nm, g in gates.items():
    gi = g[ri]; v = np.where(gi, p1, p0)
    by = {b: v[bbf[ri] == (b == "qwen")].mean() for b in ["qwen", "minimax"]}
    line = dict(cond=nm, viol=round(v.mean(), 3), viol_qwen=round(by["qwen"], 3), viol_minimax=round(by["minimax"], 3),
                reinsert=round(gi.mean(), 3), extra_tok=round((gi * dt).mean(), 1))
    out.append(line); print(line)
print("pinned-prompt extra tokens mean %.1f; plain viol %.3f pinned %.3f" % (dt.mean(), p0.mean(), p1.mean()))
json.dump(out, open("metrics_r2_intervention.json", "w"), indent=1)
