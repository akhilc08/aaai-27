"""Seed variance of CV heads + text-feature LR trained on 40k PokerBench train spots (no Jev) -> extra.json."""
import json, random, numpy as np, sys
sys.argv = ["x"]; import analyze as A
from parse import parse
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
out = {}
XTJ = np.hstack([A.XT, A.XJ]); XP = np.hstack([A.JP, A.mask])
for name, X in (("text", A.XT), ("jev", A.XJ), ("text+jev", XTJ), ("jev choice probs only (calibrated)", XP)):
    out[name] = [A.metrics(A.cv(X, seed=s))["acc"] for s in range(5)]
    out[name + "_mean"] = round(float(np.mean(out[name])), 3)
ds = load_dataset("RZ412/PokerBench", cache_dir=A.P + "data")["train"]
random.seed(0); idx = random.sample(range(len(ds)), 40000)
tr = []
for i in idx:
    try:
        r = parse(ds[i]["instruction"], ds[i]["output"]); r["pre"] = "flop comes" not in ds[i]["instruction"]
        r["y"] = A.norm(r["label"], r)
        if r["y"] in A.C: tr.append(r)
    except Exception: pass
Xtr = np.array([A.textfeat(r) for r in tr]); ytr = np.array([A.C.index(r["y"]) for r in tr])
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
Pm = np.zeros((len(A.y), 5)); Pm[:, m.classes_] = m.predict_proba(A.XT); Pm += 1e-9
out["text LR trained on 40k train spots"] = A.metrics(Pm); out["n_train"] = len(tr)
json.dump(out, open(A.P + "extra.json", "w"), indent=1); print(json.dumps(out, indent=1))
