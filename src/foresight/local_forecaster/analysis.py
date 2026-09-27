"""(4) opponent identification from history; (5) data scaling of the GBM opponent / P(win) models.
usage: analysis.py -> data/analysis_gen9.json"""
import json, os, pickle, random, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as Me
import eval_base as EB

HERE = os.path.dirname(os.path.abspath(__file__))
FMT = "gen9randombattle"


def main():
    import memguard
    memguard.start(6000)
    d = pickle.load(open(os.path.join(HERE, "data", f"ds_{FMT}.pkl"), "rb"))
    out = {}
    # ---- (4) which bot is the opponent, from the history available at turn k
    rows = [r for r in d["opp"] if r["y_idx"] >= 0]
    classes = sorted({r["opp_cls"] for r in rows})
    ident = {}
    for k in (2, 3, 4, 6, 8, 11):
        rk = [r for r in rows if r["turn"] == k]
        tr = [r for r in rk if r["split"] == "train"]; te = [r for r in rk if r["split"] != "train"]
        if len(te) < 50:
            continue
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(np.array([r["hist"] for r in tr]),
                                                                           [classes.index(r["opp_cls"]) for r in tr])
        acc = float(np.mean(g.predict(np.array([r["hist"] for r in te])) == np.array([classes.index(r["opp_cls"]) for r in te])))
        maj = max(np.mean([r["opp_cls"] == c for r in te]) for c in classes)
        ident[f"turn {k} (history of {k - 1} turns)"] = {"n_test": len(te), "acc": round(acc, 3), "majority": round(float(maj), 3)}
    out["opp_identification"] = {"classes": classes, "by_turn": ident}
    print(json.dumps(out["opp_identification"], indent=1), flush=True)
    # ---- (5) data scaling: GBM opp (with history) and GBM P(win) vs number of training battles
    battles = sorted({r["battle"] for r in d["pwin"] if r["split"] == "train"})
    rng = random.Random(0); rng.shuffle(battles)
    te_opp = [r for r in rows if r["split"] == "test" and len(r["cands"]) >= 2]
    te_pw = [r for r in d["pwin"] if r["split"] == "test"]
    scale = {}
    for nb in (30, 100, 300, 1000, len(battles)):
        keep = set(battles[:nb])
        tr_opp = [r for r in rows if r["split"] == "train" and r["battle"] in keep and len(r["cands"]) >= 2]
        Xs, ys = [], []
        for r in tr_opp:
            for j, cf in enumerate(r["cf"]):
                Xs.append(list(cf) + list(r["num"]) + list(r["hist"])); ys.append(int(j == r["y_idx"]))
        g = EB.gbm().fit(np.array(Xs, float), ys)
        Xt, grp = [], []
        for i, r in enumerate(te_opp):
            for cf in r["cf"]:
                Xt.append(list(cf) + list(r["num"]) + list(r["hist"])); grp.append(i)
        p = g.predict_proba(np.array(Xt, float))[:, 1]
        pols = [[] for _ in te_opp]
        for pi, gi in zip(p, grp):
            pols[gi].append(pi)
        pol = Me.policy([np.array(x) / sum(x) for x in pols], [r["y_idx"] for r in te_opp])
        tr_pw = [r for r in d["pwin"] if r["split"] == "train" and r["battle"] in keep]
        gp = EB.gbm().fit(np.array([r["num"] for r in tr_pw], float), [r["y"] for r in tr_pw])
        bp = Me.binary(gp.predict_proba(np.array([r["num"] for r in te_pw], float))[:, 1], [r["y"] for r in te_pw])
        scale[nb] = {"opp_rows": len(tr_opp), "opp_acc": pol["acc"], "opp_nll": pol["nll"], "pwin_rows": len(tr_pw),
                     "pwin_brier": bp["brier"], "pwin_auroc": bp["auroc"], "pwin_ece": bp["ece"]}
        print(nb, scale[nb], flush=True)
    out["scaling_gbm"] = scale
    json.dump(out, open(os.path.join(HERE, "data", "analysis_gen9.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
