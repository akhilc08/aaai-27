import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from common import key
S = json.load(open("data/sample1500.json"))
JV = {(r["battle"], r["turn"], r["me"]): r for r in map(json.loads, open("jev_out.jsonl"))}
LL = {(r["battle"], r["turn"], r["me"]): r for r in map(json.loads, open("llm_out.jsonl"))}
S = [x for x in S if (x["battle"], x["turn"], x["me"]) in JV]
K = lambda x: (x["battle"], x["turn"], x["me"])
y = np.array([x["label"][0] == "switch" for x in S], int); g = np.array([x["battle"] for x in S])
FQ = ["f_disadv", "f_koed", "f_opphits", "f_setup", "f_faster", "f_goodswitch", "f_sack"]
def simple(x):
    sn, me, op = x["snap"], x["me"], x["opp"]; oa = sn["st"][op][sn["active"][op]]; ma = sn["st"][me][sn["active"][me]]
    return [oa["hp"] / 100, ma["hp"] / 100, oa["status"] is not None, sum(sn["boosts"][op].values()), sum(sn["boosts"][me].values()),
            x["opp_bench_alive"], 6 - len(sn["st"][op]), len(x["revealed_moves"]), min(sn["turn"], 40) / 40,
            x["prev_opp_action"] == "switch", x["prev_opp_action"] is None, len(sn["sidecond"][op]) > 0]
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
Xs = np.array([simple(x) for x in S], float)
Xj0 = lg(np.array([[JV[K(x)]["p_sw"]] for x in S]))
Xjf = lg(np.array([[JV[K(x)][k] for k in FQ] for x in S]))
def cv(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p
def cvconst():
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xs, y, g): p[te] = y[tr].mean()
    return p
prev = (Xs[:, 9] > 0).astype(float)
def cvrepeat():
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xs, y, g):
        for v in (0, 1): p[te][prev[te] == v]; p[te[prev[te] == v]] = y[tr][prev[tr] == v].mean()
    return p
P = {"majority (base rate)": cvconst(), "repeat last action type": cvrepeat(), "simple feats LR": cv(Xs),
     "Jev zero-shot (raw)": np.array([JV[K(x)]["p_sw"] for x in S]), "Jev zero-shot + Platt": cv(Xj0),
     "Jev 8 nouls + simple feats LR": cv(np.hstack([Xs, Xj0, Xjf])), "Jev 8 nouls LR": cv(np.hstack([Xj0, Xjf]))}
def met(p, yy):
    return dict(acc=((p > 0.5) == yy).mean(), ll=log_loss(yy, np.clip(p, 1e-3, 1 - 1e-3), labels=[0, 1]), brier=brier_score_loss(yy, p),
                auc=roc_auc_score(yy, p) if len(set(p)) > 1 else 0.5)
out = ["## (a) switch vs move", f"n={len(y)} turns, {len(set(g))} battles, switch base rate {y.mean():.3f}", "",
       "| method | n | acc | log-loss | Brier | AUROC |", "|---|---|---|---|---|---|"]
for k, p in P.items(): m = met(p, y); out.append(f"| {k} | {len(y)} | {m['acc']:.3f} | {m['ll']:.3f} | {m['brier']:.3f} | {m['auc']:.3f} |")
li = [i for i, x in enumerate(S) if K(x) in LL and LL[K(x)]["p_sw"] is not None]
yl = y[li]; out.append(f"\nLLM subset (n={len(li)}, switch rate {yl.mean():.3f}):\n\n| method | acc | log-loss | Brier | AUROC |\n|---|---|---|---|---|")
pl = np.array([float(LL[K(S[i])]["p_sw"]) for i in li])
for k, p in list(P.items()) + [("cheap LLM (qwen3-30b-a3b)", None)]:
    pp = pl if p is None else p[li]; m = met(pp, yl); out.append(f"| {k} | {m['acc']:.3f} | {m['ll']:.3f} | {m['brier']:.3f} | {m['auc']:.3f} |")
# (b)
head = P["Jev 8 nouls + simple feats LR"]
E = [i for i, x in enumerate(S) if len(x["revealed_moves"]) >= 1 and (x["label"][0] == "switch" or x["label"][1] in x["revealed_moves"])]
cover_all = len(E) / len(S); cover_mv = sum(S[i]["label"][0] == "move" for i in E) / max(1, sum(y == 0))
def bmet(dists, idx):
    acc, ll, br = [], [], []
    for d, i in zip(dists, idx):
        x = S[i]; opts = [key(m) for m in x["revealed_moves"]] + ["switch"]
        v = np.array([max(float(d.get(o, 0) or 0), 0) for o in opts]); v = v / v.sum() if v.sum() > 0 else np.ones(len(opts)) / len(opts)
        v = np.clip(v, 1e-3, 1); v = v / v.sum()
        t = "switch" if x["label"][0] == "switch" else key(x["label"][1]); ti = opts.index(t); oh = np.eye(len(opts))[ti]
        acc.append(v.argmax() == ti); ll.append(-np.log(v[ti])); br.append(((v - oh) ** 2).sum())
    return np.mean(acc), np.mean(ll), np.mean(br)
def dists(idx, kind):
    D = []
    for i in idx:
        x = S[i]; ms = [key(m) for m in x["revealed_moves"]]; opts = ms + ["switch"]
        if kind == "uniform": D.append({o: 1 for o in opts})
        elif kind == "repeat":
            last = key(x["snap"]["st"][x["opp"]][x["snap"]["active"][x["opp"]]].get("last") or x["revealed_moves"][0])
            D.append({o: (0.7 if o == last else 0.3 / (len(opts) - 1)) for o in opts})
        elif kind == "jev": D.append(JV[K(x)]["act_probs"] or {})
        elif kind == "jevhead":
            jp = JV[K(x)]["act_probs"] or {}; mm = np.array([jp.get(m, 0) for m in ms]) + 1e-6; mm = mm / mm.sum() * (1 - head[i])
            D.append({**dict(zip(ms, mm)), "switch": head[i]})
        elif kind == "llm":
            lp = LL[K(x)]["probs"] or {}; D.append({key(k): v for k, v in lp.items()})
    return D
out += ["", "## (b) which option among opponent's revealed moves + switch",
        f"eligible turns n={len(E)} ({cover_all:.1%} of all turns; covers {cover_mv:.1%} of move turns + all switch turns); "
        f"switch share in eligible {np.mean([S[i]['label'][0]=='switch' for i in E]):.3f}; mean #options {np.mean([len(S[i]['revealed_moves'])+1 for i in E]):.2f}", "",
        "| method | n | acc | NLL | Brier |", "|---|---|---|---|---|"]
for k, kind in [("uniform", "uniform"), ("repeat last move (0.7 mass)", "repeat"), ("Jev zero-shot choice", "jev"), ("Jev choice + head switch prob", "jevhead")]:
    a, l, b = bmet(dists(E, kind), E); out.append(f"| {k} | {len(E)} | {a:.3f} | {l:.3f} | {b:.3f} |")
El = [i for i in E if i in set(li) and LL[K(S[i])]["probs"]]
out.append(f"\nLLM subset (n={len(El)}):\n\n| method | acc | NLL | Brier |\n|---|---|---|---|")
for k, kind in [("uniform", "uniform"), ("repeat last move", "repeat"), ("Jev zero-shot choice", "jev"), ("Jev choice + head", "jevhead"), ("cheap LLM", "llm")]:
    a, l, b = bmet(dists(El, kind), El); out.append(f"| {k} | {a:.3f} | {l:.3f} | {b:.3f} |")
jl = [r["lat"] for r in JV.values()]; ll_ = [r["lat"] for r in LL.values()]; lc = [r["cost"] or 0 for r in LL.values()]
out += ["", "## cost / latency", f"Jev: median latency {np.median(jl):.2f}s (one call with 9 questions), $/1k predictions from spend log below",
        f"LLM: median latency {np.median(ll_):.2f}s, $/1k = {1000*np.mean(lc):.4f}"]
print("\n".join(out))
