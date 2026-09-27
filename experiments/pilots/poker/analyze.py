"""Baselines, Jev zero-shot, learned heads (5-fold CV) and LLM comparison -> metrics.json."""
import json, re, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
P = "/Users/sickle/Coding/aaai-27/experiments/pilots/poker/"
C = ["fold", "check", "call", "bet", "raise"]
RV = {r: i for i, r in enumerate("23456789TJQKA", 2)}
POS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

def norm(a, r):  # map action to the legal class set
    if a in ("all", "all-in", "allin", "shove"): a = "raise"
    if not r["pre"] and a == "raise" and not r["facing"]: a = "bet"
    if a == "bet" and (r["facing"] or r["pre"]): a = "raise"
    return a

rows = [json.loads(l) for l in open(P + "data/spots.jsonl")]
jev = {j["id"]: j for j in map(json.loads, open(P + "jev.jsonl"))}
for r in rows: r["y"] = norm(r["label"], r)
y = np.array([C.index(r["y"]) for r in rows])
mask = np.array([[c in [norm(l, r) for l in r["legal"]] for c in C] for r in rows], float)
assert all(mask[i, y[i]] for i in range(len(rows))), "label outside legal set"

def textfeat(r):
    h, b = r["hand"], r["board"]; hr = sorted([RV[c[0]] for c in h], reverse=True); br = [RV[c[0]] for c in b]
    suits = Counter(c[1] for c in h + b); hs = {c[1] for c in h}
    fl = max([suits[s] for s in hs]); allr = set(hr + br) | ({1} if 14 in hr + br else set())
    straight = max(sum(1 for k in range(s, s + 5) if k in allr) for s in range(1, 11))
    nums = [float(x) for x in re.findall(r"(?:bet|raise) ([\d.]+)", r["cur_line"])]
    last = nums[-1] if nums and r["facing"] else 0.0
    first = r.get(r["street"] + "_line", "").split(",")[0] if r["street"] != "preflop" else ""
    f = [hr[0], hr[1], hr[0] == hr[1], len(hs) == 1, hr[0] - hr[1],
         sum(x in br for x in hr), bool(br) and hr[0] == hr[1] and hr[1] > max(br), bool(br) and max(br) in hr,
         fl, fl == 4, fl >= 5, straight, straight == 4, straight >= 5, len(br) - len(set(br)), max(Counter(c[1] for c in b).values()) if b else 0,
         r["facing"], np.log1p(r["pot"]), last / r["pot"], r["preflop_line"].count("raise") + r["preflop_line"].count("all in"),
         r["preflop_line"].count("call"), r["cur_line"].count("raise"), not first.startswith(r["pos"]) and r["street"] != "preflop"]
    f += [r["street"] == s for s in ("preflop", "flop", "turn", "river")] + [r["pos"] == p for p in POS]
    return [float(v) for v in f]

NQ = ["strong", "made", "best", "draw", "ip", "wet", "aggr", "odds", "bluff"]
def jevprobs(r):
    p = np.zeros(5)
    for k, v in jev[r["id"]]["probs"].items(): p[C.index(norm(k, r))] += v
    return p
JP = np.array([jevprobs(r) for r in rows])
XT = np.array([textfeat(r) for r in rows])
XJ = np.array([[jev[r["id"]][k] for k in NQ] for r in rows] + [])
XJ = np.hstack([XJ, JP, mask])  # Jev answers + zero-shot probs + legal mask (from text parse)

def metrics(P_, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    Pm = P_[idx] * mask[idx]; Pm = Pm / Pm.sum(1, keepdims=True)
    pred = Pm.argmax(1); yy = y[idx]
    ll = -np.mean(np.log(np.clip(Pm[np.arange(len(idx)), yy], 1e-3, 1)))
    return {"acc": round(float((pred == yy).mean()), 3), "macroF1": round(float(f1_score(yy, pred, average="macro")), 3), "logloss": round(float(ll), 3)}

def cv(X, seed=0):
    out = np.zeros((len(y), 5)); skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], y[tr])
        out[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
    return out + 1e-9

# majority within legal set (fit on all = optimistic)
maj = np.zeros((len(y), 5))
for i, r in enumerate(rows):
    key = (r["pre"], r["facing"]); same = [y[j] for j, s in enumerate(rows) if (s["pre"], s["facing"]) == key]
    maj[i] = np.bincount(same, minlength=5) / len(same)
rule = np.array([[0, 1, 0.5, 0, 0] if not r["facing"] else [0, 0, 1, 0, 0] for r in rows], float) + 1e-3
methods = {"global majority": np.tile(np.bincount(y, minlength=5) / len(y), (len(y), 1)),
           "majority | legal set": maj, "rule: check else call": rule,
           "Jev zero-shot choice": JP + 1e-6,
           "LR text features": cv(XT), "LR Jev features": cv(XJ), "LR text + Jev": cv(np.hstack([XT, XJ]))}
pre = np.array([r["pre"] for r in rows]); ids = np.array([r["id"] for r in rows])
res = {}
for k, M in methods.items():
    res[k] = {"all": metrics(M), "pre": metrics(M, np.where(pre)[0]), "post": metrics(M, np.where(~pre)[0])}

# LLM on its 200-spot subset; compare all methods on the same subset
L = {l["id"]: l for l in map(json.loads, open(P + "llm.jsonl"))}
sub = np.array([i for i, r in enumerate(rows) if r["id"] in L])
lp, em, parsed = np.zeros((len(y), 5)), [], 0
for i in sub:
    r = rows[i]; t = L[r["id"]]["text"].lower().replace("all in", "all-in").strip()
    w = re.findall(r"fold|check|call|bet|raise|all-in", t)
    a = norm(w[0], r) if w else None; parsed += a is not None
    num = re.findall(r"[\d.]+\d|\d", t); sz = float(num[0]) if num else None
    if a in C and mask[i, C.index(a)]: lp[i, C.index(a)] = 1
    else: lp[i] = mask[i] / mask[i].sum()  # unparseable/illegal -> uniform over legal (counts as partial)
    ok = a == r["y"] and (r["size"] is None or (sz is not None and abs(sz - r["size"]) < 0.51))
    em.append(ok)
lp = lp * 0.98 + 0.02 * mask / mask.sum(1, keepdims=True)
res["subset200"] = {k: metrics(M, sub) for k, M in methods.items()}
res["subset200"]["cheap LLM (qwen3-30b-a3b), dataset prompt"] = metrics(lp, sub)
res["llm_exact_match"] = round(float(np.mean(em)), 3); res["llm_parsed"] = int(parsed)
res["llm_EM_pre"] = round(float(np.mean([e for e, i in zip(em, sub) if pre[i]])), 3)
res["llm_EM_post"] = round(float(np.mean([e for e, i in zip(em, sub) if not pre[i]])), 3)
res["class_dist"] = {C[k]: int(v) for k, v in enumerate(np.bincount(y, minlength=5))}
res["class_dist_pre"] = {C[k]: int(v) for k, v in enumerate(np.bincount(y[pre], minlength=5))}
res["latency"] = {"jev_mean_s": round(float(np.mean([j["latency"] for j in jev.values()])), 3),
                  "llm_mean_s": round(float(np.mean([l["latency"] for l in L.values()])), 3)}
spend = [json.loads(l) for l in open(P + "../spend.jsonl")]
for t, n in (("poker-jev", len(jev)), ("poker-llm", len(L))):
    res["latency"][t + "_usd_per_1k"] = round(sum(s["cost"] or 0 for s in spend if s["tag"] == t) / n * 1000, 4)
json.dump(res, open(P + "metrics.json", "w"), indent=1)
print(json.dumps(res, indent=1))
