"""Non-transformer baselines: P(win), delta-line, opponent next action (with / without history).
usage: eval_base.py FMT  -> data/base_{FMT}.json, data/preds_base_{FMT}.pkl, ckpt/base_{FMT}.pkl"""
import json, os, pickle, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as Me

HERE = os.path.dirname(os.path.abspath(__file__))
FMT = sys.argv[1] if len(sys.argv) > 1 else "gen9randombattle"
DDIR = os.environ.get("INDIR", os.path.join(HERE, "data"))
SUF = os.environ.get("SUF", "")


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


def turn_buckets(rows, p, y):
    out = {}
    for name, lo, hi in (("t1-5", 0, 5), ("t6-15", 6, 15), ("t16+", 16, 999)):
        m = np.array([lo <= r["turn"] <= hi for r in rows])
        if m.sum() > 50:
            out[name] = Me.binary(np.asarray(p)[m], np.asarray(y)[m])
    return out


def pwin_like(d, key, numkey="num", hpkey="hp"):
    sp = {s: [r for r in d[key] if r["split"] == s] for s in ("train", "val", "test")}
    X = {s: np.array([r[numkey] for r in sp[s]], float) for s in sp}
    Y = {s: np.array([r["y"] for r in sp[s]]) for s in sp}
    H = {s: np.array([r[hpkey] for r in sp[s]]).reshape(-1, 1) for s in sp}
    res, preds, models = {}, {}, {}
    preds["base_rate"] = np.full(len(Y["test"]), Y["train"].mean())
    preds["hp_raw"] = H["test"].ravel()
    platt = LogisticRegression().fit(H["train"], Y["train"])
    preds["hp_platt"] = platt.predict_proba(H["test"])[:, 1]
    m = lr().fit(X["train"], Y["train"]); preds["logreg"] = m.predict_proba(X["test"])[:, 1]; models["logreg"] = m
    g = gbm().fit(X["train"], Y["train"]); preds["gbm"] = g.predict_proba(X["test"])[:, 1]; models["gbm"] = g
    models["hp_platt"] = platt
    for k, p in preds.items():
        res[k] = dict(Me.binary(p, Y["test"]), by_turn=turn_buckets(sp["test"], p, Y["test"]), rel=Me.reliability(p, Y["test"]))
    return res, preds, models, Y["test"]


def opp_task(d):
    rows = [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2]
    sp = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}

    def cand_X(rs, hist):
        Xs, ys, grp = [], [], []
        for i, r in enumerate(rs):
            for k, cf in enumerate(r["cf"]):
                x = list(cf) + list(r["num"]) + (list(r["hist"]) if hist else [])
                Xs.append(x); ys.append(int(k == r["y_idx"])); grp.append(i)
        return np.array(Xs, float), np.array(ys), np.array(grp)

    def pols_from(p, grp, n):
        out = [[] for _ in range(n)]
        for pi, g in zip(p, grp):
            out[g].append(pi)
        return [np.array(o) / max(sum(o), 1e-9) for o in out]

    test = sp["test"]
    yidx = [r["y_idx"] for r in test]
    ysw = np.array([r["y_sw"] for r in test])
    res, preds, models = {}, {}, {}
    preds["uniform"] = [np.ones(len(r["cands"])) / len(r["cands"]) for r in test]
    preds["heuristic"] = [np.array(r["heur"]) for r in test]
    # "majority": always the max-damage option, smoothed with the train-set top-1 hit rate
    hit = np.mean([int(np.argmax(np.array(r["cf"])[:, 2]) == r["y_idx"]) for r in sp["train"]])
    mj = []
    for r in test:
        k = int(np.argmax(np.array(r["cf"])[:, 2])); n = len(r["cands"])
        p = np.full(n, (1 - hit) / (n - 1)); p[k] = hit; mj.append(p)
    preds["maxdmg_majority"] = mj
    for hist in (False, True):
        Xtr, ytr, _ = cand_X(sp["train"], hist)
        Xte, _, gte = cand_X(test, hist)
        tag = "hist" if hist else "nohist"
        m = lr().fit(Xtr, ytr); preds[f"logreg_{tag}"] = pols_from(m.predict_proba(Xte)[:, 1], gte, len(test)); models[f"logreg_{tag}"] = m
        g = gbm().fit(Xtr, ytr); preds[f"gbm_{tag}"] = pols_from(g.predict_proba(Xte)[:, 1], gte, len(test)); models[f"gbm_{tag}"] = g
    for k, pols in preds.items():
        psw = np.array([sum(p[j] for j in range(len(p)) if r["cand_sw"][j]) for p, r in zip(pols, test)])
        res[k] = {"policy": Me.policy(pols, yidx), "switch": Me.binary(psw, ysw) if k != "uniform" else Me.binary(psw, ysw),
                  "by_opp": {c: Me.policy([p for p, r in zip(pols, test) if r["opp_cls"] == c], [r["y_idx"] for r in test if r["opp_cls"] == c])["acc"]
                             for c in sorted({r["opp_cls"] for r in test})}}
    res["_switch_base_rate_test"] = round(float(ysw.mean()), 4)
    res["_opp_cls_counts_test"] = {c: sum(r["opp_cls"] == c for r in test) for c in sorted({r["opp_cls"] for r in test})}
    return res, preds, models


def main():
    import memguard
    memguard.start(6000)
    d = pickle.load(open(os.path.join(DDIR, f"ds_{FMT}.pkl"), "rb"))
    out, allp, allm = {}, {}, {}
    out["pwin"], allp["pwin"], allm["pwin"], _ = pwin_like(d, "pwin")
    print("pwin", {k: (v["brier"], v["auroc"], v["ece"]) for k, v in out["pwin"].items()}, flush=True)
    out["delta"], allp["delta"], allm["delta"], _ = pwin_like(d, "delta", hpkey="hp_leaf")
    print("delta", {k: (v["brier"], v["auroc"], v["ece"]) for k, v in out["delta"].items()}, flush=True)
    out["opp"], allp["opp"], allm["opp"] = opp_task(d)
    for k, v in out["opp"].items():
        if not k.startswith("_"):
            print("opp", k, v["policy"], "switch", (v["switch"]["brier"], v["switch"]["auroc"]), v["by_opp"], flush=True)
    json.dump(out, open(os.path.join(DDIR, f"base_{FMT}.json"), "w"), indent=1, default=float)
    pickle.dump(allp, open(os.path.join(DDIR, f"preds_base_{FMT}.pkl"), "wb"))
    os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
    pickle.dump(allm, open(os.path.join(HERE, "ckpt", f"base_{FMT}{SUF}.pkl"), "wb"))


if __name__ == "__main__":
    main()
