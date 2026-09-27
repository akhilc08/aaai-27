"""Collect forecast metrics for all models on common held-out test subsets -> data/summary_{FMT}.json (+ printed tables)."""
import glob, json, os, pickle, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as Me

HERE = os.path.dirname(os.path.abspath(__file__))
FMT = sys.argv[1] if len(sys.argv) > 1 else "gen9randombattle"


def sub_idx(n, N, seed=0):
    return sorted(random.Random(seed).sample(range(n), min(N, n)))


def softmax(l, T=1.0):
    z = np.asarray(l, float) / T
    z = np.exp(z - z.max())
    return z / z.sum()


def main():
    d = pickle.load(open(os.path.join(HERE, "data", f"ds_{FMT}_slim.pkl"), "rb"))
    base = pickle.load(open(os.path.join(HERE, "data", f"preds_base_{FMT}.pkl"), "rb"))
    te = {"pwin": [r for r in d["pwin"] if r["split"] == "test"], "delta": [r for r in d["delta"] if r["split"] == "test"],
          "opp": [r for r in [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2] if r["split"] == "test"]}
    S = {t: sub_idx(len(te[t]), 4000) for t in te}
    out = {"pwin": {}, "delta": {}, "opp": {}}

    def add_bin(task, name, p, idx):
        y = np.array([te[task][i]["y"] for i in idx])
        rows = [te[task][i] for i in idx]
        early = np.array([r["turn"] <= 5 for r in rows])
        out[task][name] = dict(Me.binary(p, y), early_auroc=Me.binary(np.asarray(p)[early], y[early])["auroc"],
                               rel=Me.reliability(p, y))

    def add_pol(name, pols, idx):
        rows = [te["opp"][i] for i in idx]
        yi = [r["y_idx"] for r in rows]
        ysw = np.array([r["y_sw"] for r in rows])
        psw = np.array([sum(p[j] for j in range(len(p)) if r["cand_sw"][j]) for p, r in zip(pols, rows)])
        by = {}
        for c in sorted({r["opp_cls"] for r in rows}):
            m = [k for k, r in enumerate(rows) if r["opp_cls"] == c]
            by[c] = Me.policy([pols[k] for k in m], [yi[k] for k in m])["acc"]
        out["opp"][name] = dict(Me.policy(pols, yi), sw_brier=Me.binary(psw, ysw)["brier"], sw_auroc=Me.binary(psw, ysw)["auroc"],
                                sw_ece=Me.binary(psw, ysw)["ece"], by_opp=by)

    for task in ("pwin", "delta"):
        for k, p in base[task].items():
            add_bin(task, k, np.asarray(p)[S[task]], S[task])
    for k, pols in base["opp"].items():
        add_pol(k, [pols[i] for i in S["opp"]], S["opp"])
    # Laya runs
    for fn in sorted(glob.glob(os.path.join(HERE, "data", f"preds_laya_*_{FMT}.pkl"))):
        tag = os.path.basename(fn)[len("preds_laya_"):-len(f"_{FMT}.pkl")]
        x = pickle.load(open(fn, "rb"))
        if tag == "zeroshot":
            for task, v in x.items():
                if task == "pwin":
                    add_bin("pwin", "laya_zeroshot(T=1.98 shipped)", [softmax(l, 1.98)[1] for l in v["logits"]], v["idx"])
                else:
                    add_pol("laya_zeroshot" + ("" if True else ""), [softmax(l, 1.76) for l in v["logits"]], v["idx"])
            continue
        for (task, s), lg in x["preds"].items():
            if s != "test":
                continue
            T = x["temps"][task]
            idx = x["idx"][(task, s)]
            if task in ("pwin", "delta"):
                add_bin(task, f"laya_{tag}", [softmax(l, T)[1] for l in lg], idx)
            else:
                add_pol(f"laya_{tag}", [softmax(l, T) for l in lg], idx)
    for kind in ("lm", "mbert"):
        for fn in sorted(glob.glob(os.path.join(HERE, "data", f"preds_{kind}_*_{FMT}.pkl"))):
            tag = os.path.basename(fn)[len(f"preds_{kind}_"):-len(f"_{FMT}.pkl")]
            x = pickle.load(open(fn, "rb"))
            lg, idx, T = x["preds"]["test"], x["idx"]["test"], x["T"]
            if x["task"] == "pwin":
                add_bin("pwin", f"{kind}_{tag}", [softmax(l, T)[1] for l in lg], idx)
            else:
                add_pol(f"{kind}_{tag}", [softmax(l, T) for l in lg], idx)
    json.dump(out, open(os.path.join(HERE, "data", f"summary_{FMT}.json"), "w"), indent=1, default=float)
    for task in ("pwin", "delta"):
        print(f"\n## {task} (test subset n={len(S[task])} unless noted)\n| model | n | Brier | AUROC | AUROC t<=5 | ECE | logloss |\n|---|---|---|---|---|---|---|")
        for k, v in out[task].items():
            print(f"| {k} | {v['n']} | {v['brier']} | {v['auroc']} | {v['early_auroc']} | {v['ece']} | {v['logloss']} |")
    print(f"\n## opponent next action\n| model | n | acc | NLL | Brier | ECE(top1) | switch Brier | switch AUROC | acc by opponent |\n|---|---|---|---|---|---|---|---|---|")
    for k, v in out["opp"].items():
        print(f"| {k} | {v['n']} | {v['acc']} | {v['nll']} | {v['brier']} | {v['ece_top1']} | {v['sw_brier']} | {v['sw_auroc']} | "
              + ", ".join(f"{c[:6]} {a}" for c, a in v["by_opp"].items()) + " |")


if __name__ == "__main__":
    main()
