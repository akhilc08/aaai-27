"""Item 0: reliability diagrams, accuracy by turn / #options / switch-vs-attack, top-k, for gen8 Laya
on (a) held-out gen8 bot test and (b) Abyssal's real moves. -> figures/*.png, data/night_analysis.json"""
import json, os, pickle, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as Me

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def softmax(l, T):
    z = np.asarray(l, float) / T
    z = np.exp(z - z.max())
    return z / z.sum()


def rel_bins(conf, cor, bins=10):
    conf, cor = np.asarray(conf), np.asarray(cor)
    out = []
    for i in range(bins):
        m = (conf >= i / bins) & ((conf < (i + 1) / bins) if i < bins - 1 else (conf <= 1))
        if m.sum():
            out.append((float(conf[m].mean()), float(cor[m].mean()), int(m.sum())))
    return out


def breakdown(pols, ys, meta):
    top1 = [float(np.argmax(p) == y) for p, y in zip(pols, ys)]
    rank = [int((np.asarray(p) > p[y]).sum()) for p, y in zip(pols, ys)]
    res = {"n": len(ys), "top1": np.mean(top1), "top2": np.mean([r < 2 for r in rank]), "top3": np.mean([r < 3 for r in rank]),
           "uniform_top1": np.mean([1 / m["n"] for m in meta])}
    bt = {}
    for name, lo, hi in (("turn 1-5", 1, 5), ("turn 6-15", 6, 15), ("turn 16-30", 16, 30), ("turn 31+", 31, 999)):
        k = [i for i, m in enumerate(meta) if lo <= m["turn"] <= hi]
        if k:
            bt[name] = {"n": len(k), "acc": np.mean([top1[i] for i in k])}
    res["by_turn"] = bt
    bn = {}
    for name, lo, hi in (("2 options", 2, 2), ("3 options", 3, 3), ("4 options", 4, 4), ("5-6 options", 5, 6), ("7+ options", 7, 99)):
        k = [i for i, m in enumerate(meta) if lo <= m["n"] <= hi]
        if k:
            bn[name] = {"n": len(k), "acc": np.mean([top1[i] for i in k]), "chance": np.mean([1 / meta[i]["n"] for i in k])}
    res["by_n_options"] = bn
    for lab, v in (("actual attack/move", 0), ("actual switch", 1)):
        k = [i for i, m in enumerate(meta) if m["y_sw"] == v]
        res[f"acc_{lab.replace(' ', '_').replace('/', '_')}"] = {"n": len(k), "acc": np.mean([top1[i] for i in k]) if k else None}
    psw = [sum(p[j] for j in range(len(p)) if m["cand_sw"][j]) for p, m in zip(pols, meta)]
    res["switch_forecast"] = Me.binary(psw, [m["y_sw"] for m in meta])
    res["policy"] = Me.policy(pols, ys)
    conf = [float(np.max(p)) for p in pols]
    res["reliability_top1"] = rel_bins(conf, top1)
    res["reliability_switch"] = rel_bins(psw, [m["y_sw"] for m in meta])
    return res


def main():
    out = {}
    # (a) gen8 held-out bot test
    d = pickle.load(open(os.path.join(HERE, "data", "ds_gen8randombattle_slim.pkl"), "rb"))
    te = [r for r in [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2] if r["split"] == "test"]
    x = pickle.load(open(os.path.join(HERE, "data", "preds_laya_opp_hist_gen8randombattle.pkl"), "rb"))
    idx, lg, T = x["idx"][("opp", "test")], x["preds"][("opp", "test")], x["temps"]["opp"]
    rows = [te[i] for i in idx]
    pols = [softmax(l, T) for l in lg]
    meta = [{"turn": r["turn"], "n": len(r["cands"]), "y_sw": r["y_sw"], "cand_sw": r["cand_sw"]} for r in rows]
    out["gen8_bot_test"] = breakdown(pols, [r["y_idx"] for r in rows], meta)
    # (b) Abyssal's real moves
    ap = os.path.join(HERE, "data", "score_turns_abyssal_eval_gen8_night_preds.pkl")
    if os.path.exists(ap):
        a = pickle.load(open(ap, "rb"))
        key = [k for k in a["pols"] if "gen8" in k][0]
        out["abyssal_real_moves"] = breakdown(a["pols"][key], a["ys"], a["meta"])
    json.dump(out, open(os.path.join(HERE, "data", "night_analysis.json"), "w"), indent=1, default=float)
    # figures
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, kind, title in ((axes[0], "reliability_top1", "Top-1 action confidence"), (axes[1], "reliability_switch", "P(opponent switches)")):
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        for name, col in (("gen8_bot_test", "#1f6fb2"), ("abyssal_real_moves", "#d0621c")):
            if name in out:
                b = out[name][kind]
                ax.plot([p for p, _, _ in b], [f for _, f, _ in b], marker="o", color=col,
                        label=f"{'held-out bot battles' if name == 'gen8_bot_test' else 'Abyssal real moves'} (n={out[name]['n']})")
        ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency"); ax.set_title(title)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Fine-tuned Laya (gen8) opponent-action forecasts: reliability")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability_laya_gen8.png"), dpi=160)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, col, off in (("gen8_bot_test", "#1f6fb2", -0.2), ("abyssal_real_moves", "#d0621c", 0.2)):
        if name not in out:
            continue
        bt = out[name]["by_turn"]; ks = list(bt)
        axes[0].bar(np.arange(len(ks)) + off, [bt[k]["acc"] for k in ks], width=0.4, color=col, label=name.replace("_", " "))
        axes[0].set_xticks(range(len(ks))); axes[0].set_xticklabels(ks, fontsize=8)
        bn = out[name]["by_n_options"]; ks2 = list(bn)
        axes[1].bar(np.arange(len(ks2)) + off, [bn[k]["acc"] for k in ks2], width=0.4, color=col)
        axes[1].scatter(np.arange(len(ks2)) + off, [bn[k]["chance"] for k in ks2], color="k", marker="_", s=120, zorder=3)
        axes[1].set_xticks(range(len(ks2))); axes[1].set_xticklabels(ks2, fontsize=8)
    axes[0].set_title("Top-1 accuracy by turn"); axes[1].set_title("Top-1 accuracy by # options (black = chance)")
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "accuracy_breakdown_laya_gen8.png"), dpi=160)
    for k, v in out.items():
        print(k, {kk: v[kk] for kk in ("n", "top1", "top2", "top3", "uniform_top1")}, v["by_turn"], v["by_n_options"],
              v["acc_actual_attack_move"], v["acc_actual_switch"], v["switch_forecast"], flush=True)


if __name__ == "__main__":
    main()
