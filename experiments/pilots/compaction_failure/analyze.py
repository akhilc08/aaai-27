import json, numpy as np, collections as C
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
R = [json.loads(l) for l in open("features.jsonl")]
y = np.array([r["y"] for r in R]); sc = np.array([r["scenario"] for r in R]); bb = np.array([r["backbone"] for r in R])
F = ["lost", "topic", "prohib", "cond", "know_refuse", "generic"]
def acc(s, y):
    return max(((s >= t) == y).mean() for t in np.unique(np.r_[s, -1, 2]))
X = np.array([[r[f] for f in F] for r in R]); lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
Xl = lg(X); kw = np.array([0. if r["rule_kw_present"] else 1. for r in R]); bbf = (bb == "qwen").astype(float)
def head(Xf, cv):
    m = LogisticRegression(C=1.0, max_iter=2000)
    if cv == "5fold": return cross_val_predict(m, Xf, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return cross_val_predict(m, Xf, y, cv=LeaveOneGroupOut(), groups=sc, method="predict_proba")[:, 1]
S = {"keyword (rule kw missing)": kw,
     "LLM self-judgment (qwen3-30b, 'lost')": np.array([r["llm_lost"] for r in R]),
     "Jev zero-shot P(lost)": np.array([r["jev0_lost"] for r in R])}
for cv in ["5fold", "LOSO"]:
    S[f"Jev 6q + LR [{cv}]"] = head(Xl, cv)
    S[f"Jev 6q + backbone + LR [{cv}]"] = head(np.c_[Xl, bbf], cv)
    S[f"keyword + backbone + LR [{cv}]"] = head(np.c_[kw, bbf], cv)
    S[f"keyword + Jev 6q + backbone + LR [{cv}]"] = head(np.c_[kw, Xl, bbf], cv)
S["backbone only"] = bbf
print(f"n={len(y)} base rate violation={y.mean():.3f}")
out = []
for k, s in S.items():
    per = {b: round(roc_auc_score(y[bb == b], s[bb == b]), 3) for b in ["qwen", "minimax"]}
    line = dict(method=k, auroc=round(roc_auc_score(y, s), 3), acc=round(acc(s, y), 3), auroc_qwen=per["qwen"], auroc_minimax=per["minimax"],
                flag_rate=round(float((s >= 0.5).mean()), 3))
    out.append(line); print(line)
# single Jev features
for i, f in enumerate(F): print(f, round(roc_auc_score(y, X[:, i]), 3))
print("LLM flags 'lost':", np.mean([r["llm_lost"] for r in R]), " kw missing:", kw.mean())
# per-scenario AUROC for zero-shot and keyword
for s_ in sorted(set(sc)):
    m = sc == s_; print(s_, "y", round(y[m].mean(), 2), "kw", round(roc_auc_score(y[m], kw[m]), 3), "jev0", round(roc_auc_score(y[m], S["Jev zero-shot P(lost)"][m]), 3), "llm", round(roc_auc_score(y[m], S["LLM self-judgment (qwen3-30b, 'lost')"][m]), 3))
# within kw-present subset: does Jev add info?
for v in [0, 1]:
    m = kw == v; print("kw_missing=", v, "n", m.sum(), "y", round(y[m].mean(), 2), "jev0 auc", round(roc_auc_score(y[m], S["Jev zero-shot P(lost)"][m]), 3), "jev head5 auc", round(roc_auc_score(y[m], S["Jev 6q + backbone + LR [LOSO]"][m]), 3))
lat = {k: np.mean([r[k] for r in R]) for k in ["llm_lat", "jev0_lat", "jev_lat"]}
cost = json.load(open("/dev/stdin")) if False else None
print("latency", lat, "llm cost/1k", 1000 * np.mean([r["llm_cost"] for r in R]))
json.dump(out, open("metrics.json", "w"), indent=1)
