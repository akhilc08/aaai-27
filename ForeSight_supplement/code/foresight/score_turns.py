"""Score opponent models on the REAL opponent's actions logged during search battles (e.g. vs Abyssal).
usage: score_turns.py TURNS_PKL -> prints table, writes data/score_<name>.json"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as Mo
import evals as E
import metrics as Me
from build_ds import opp_label_idx

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1]
    R = []
    with open(path, "rb") as f:
        while True:
            try:
                R += pickle.load(f)
            except EOFError:
                break
    reqs, ys, hists, meta = [], [], [], []
    for r in R:
        s = pickle.loads(r["state"])
        if r["opp_act"] is None or s.active(1).hp <= 0 or s.active(1).unseen or s.active(0).hp <= 0:
            continue
        idx, is_sw, cands = opp_label_idx(s, list(r["opp_act"]))
        if idx == -1 and not is_sw:
            idx = next((k for k, a in enumerate(cands) if a[0] == "m" and s.active(1).moves[a[1]].id.startswith("unrevealed")), -1)
        if idx < 0 or len(cands) < 2:
            continue
        reqs.append((s, 1, cands, None)); ys.append(idx); hists.append(r["hist"])
        meta.append({"turn": r["turn"], "n": len(cands), "y_sw": int(cands[idx][0] == "s"), "cand_sw": [int(a[0] == "s") for a in cands]})
    print("scorable turns", len(ys), "of", len(R), "battles", len({r["battle"] for r in R}), flush=True)
    models = {}
    ck = os.environ.get("CKPTS")
    pairs = [(c, c) for c in ck.split(",")] if ck else [("laya_hist (gen9-trained)", "laya_opp_hist_gen9randombattle.pt"),
                                                         ("laya_hist (gen8-trained)", "laya_opp_hist_gen8randombattle.pt")]
    for tag, p in pairs:
        if os.path.exists(os.path.join(HERE, "ckpt", p)):
            models[tag] = E.LayaOpp(p, hist="nohist" not in p)
    out, allp = {}, {}
    for name, m in models.items():
        pols = []
        for i in range(0, len(reqs), 32):
            for j in range(i, min(i + 32, len(reqs))):
                pols += m([reqs[j]], hists[j])
        out[name] = Me.policy(pols, ys)
        allp[name] = pols
        print(name, out[name], flush=True)
        if name.startswith("laya"):
            E._CACHE.clear()
    fn = os.path.join(HERE, "data", "score_" + os.path.basename(path).replace(".pkl", "") + os.environ.get("SCORE_TAG", "") + ".json")
    json.dump(out, open(fn, "w"), indent=1, default=float)
    pickle.dump({"pols": allp, "ys": ys, "meta": meta}, open(fn.replace(".json", "_preds.pkl"), "wb"))


if __name__ == "__main__":
    main()
