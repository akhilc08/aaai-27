"""Regenerate the 'Overnight batch' section of RESULT.md from result files (safe to run any time)."""
import glob, json, os, pickle, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as Me
from report import wilson

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
START, END = "<!-- NIGHT:START -->", "<!-- NIGHT:END -->"

ARMS = [  # (arm name, description, item)
    ("layah8_hp_d2", "gen8 Laya opp + HP leaf, depth 2 (headline; first 100 from the day run)", "B2"),
    ("layah8_hp_d1", "same, depth 1", "B1"),
    ("layah8_hp_d3", "same, depth 3", "B1"),
    ("onestep_check", "PokéChamp's own OneStepPlayer vs Abyssal (protocol check; published 44%)", "B6"),
    ("unif_hp_d2", "uniform opponent model (forecaster ablation), depth 2", "B5"),
    ("layah8x3_hp_d2", "overconfident Laya (logits x3), depth 2", "B4"),
    ("layah8x3_hp_d1", "overconfident Laya (logits x3), depth 1", "B4"),
    ("layah8x3_hp_d3", "overconfident Laya (logits x3), depth 3", "B4"),
    ("layah8_wide_d2", "wider depth-2 search (opp mass .99, caps 4/3, chance .95x5/.85x3)", "B3"),
    ("layah8_alive_d2", "leaf = HP balance + 0.3 per Pokémon alive, depth 2", "L1"),
    ("layah8_pwinleaf_d2", "leaf = fine-tuned Laya P(win), depth 2", "L2"),
]
GTAGS = [  # (ckpt/preds tag, description, item)
    ("opp_hist", "gen8 Laya + history, 10k rows (reference)", "-"),
    ("opp_hist_n1k", "1k training rows", "G1"),
    ("opp_hist_n3k", "3k training rows", "G1"),
    ("opp_hist_n30k", "30k training rows", "G1"),
    ("opp_nohist", "no history, 10k rows", "G2"),
    ("ml_opp_hist", "laya-multilingual 322M, 10k rows", "G3"),
    ("opp_hist_seed1", "seed 1, 10k rows", "G4"),
    ("full_opp_hist", "full fine-tune (all layers), 10k rows", "G5"),
]


def softmax(l, T):
    z = np.asarray(l, float) / T
    z = np.exp(z - z.max())
    return z / z.sum()


def battles_table():
    res = defaultdict(lambda: {"n": 0, "w": 0, "lat": [], "lmax": 0, "nodes": [], "pq": []})
    for l in open(os.path.join(D, "arm_results.jsonl")):
        x = json.loads(l)
        if x["fmt"] != "gen8randombattle" or x["opponent"] != "ABYSSAL":
            continue
        a = res[x["arm"]]
        a["n"] += x["n"]; a["w"] += x["wins"]; a["lat"].append((x["lat_mean"], x["n"])); a["lmax"] = max(a["lmax"], x["lat_max"])
    # partial (in-progress / killed) arms from the per-battle log
    part = defaultdict(lambda: [0, 0])
    for l in open(os.path.join(D, "battles_log.jsonl")):
        x = json.loads(l)
        part[x["arm"]][0] += 1; part[x["arm"]][1] += x["won"] is True
    for arm, _, _ in ARMS:
        fn = os.path.join(D, f"dec_{arm}_ABYSSAL_gen8randombattle.jsonl")
        if os.path.exists(fn):
            for l in open(fn):
                if '"nodes"' in l:
                    x = json.loads(l)
                    res[arm]["nodes"].append(x["nodes"]); res[arm]["pq"].append(x["policy_q"])
    lines = ["| item | agent (all vs Abyssal, gen8, PokéChamp protocol) | n | win % | 95% CI | s/turn mean | s/turn max | nodes/turn | forecasts/turn |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, desc, item in ARMS:
        a = res.get(arm)
        n, w, tag = (a["n"], a["w"], "") if a and a["n"] else (0, 0, "")
        if n == 0 and part[arm][0]:
            n, w, tag = part[arm][0], part[arm][1], " (partial, in progress or stopped)"
        if n == 0:
            lines.append(f"| {item} | {desc} | 0 | — | — | — | — | — | — |")
            continue
        lo, hi = wilson(w, n)
        lat = (sum(m * k for m, k in a["lat"]) / max(sum(k for _, k in a["lat"]), 1)) if a and a["lat"] else float("nan")
        nd = np.mean(a["nodes"]) if a and a["nodes"] else float("nan")
        pq = np.mean(a["pq"]) if a and a["pq"] else float("nan")
        lines.append(f"| {item} | {desc}{tag} | {n} | {100 * w / n:.1f} | {100 * lo:.0f}-{100 * hi:.0f} | {lat:.2f} | "
                     f"{a['lmax'] if a else float('nan'):.1f} | {nd:.0f} | {pq:.0f} |")
    lines += ["| — | *published:* PokéChamp GPT-4o / Llama-3.1-8B / PokéLLMon / One-Step Lookahead | — | 70 / 64 / 56 / 44 | — | — | — | — | — |"]
    return "\n".join(lines)


def forecast_table():
    d = pickle.load(open(os.path.join(D, "ds_gen8randombattle_slim.pkl"), "rb"))
    te = [r for r in [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2] if r["split"] == "test"]
    lines = ["| item | gen8 Laya opponent model | gen8 bot test: acc / NLL / ECE (n) | Abyssal real moves: acc / NLL / ECE (n=1,963) | train time |",
             "|---|---|---|---|---|"]
    for tag, desc, item in GTAGS:
        fn = os.path.join(D, f"preds_laya_{tag}_gen8randombattle.pkl")
        left = "not run / failed"
        if os.path.exists(fn):
            x = pickle.load(open(fn, "rb"))
            if ("opp", "test") in x["preds"]:
                idx, lg, T = x["idx"][("opp", "test")], x["preds"][("opp", "test")], x["temps"]["opp"]
                m = Me.policy([softmax(l, T) for l in lg], [te[i]["y_idx"] for i in idx])
                left = f"{m['acc']:.3f} / {m['nll']:.3f} / {m['ece_top1']:.3f} ({m['n']})"
        sj = os.path.join(D, f"score_turns_abyssal_eval_gen8_{tag}.json")
        right = "—"
        if os.path.exists(sj):
            v = list(json.load(open(sj)).values())[0]
            right = f"{v['acc']:.3f} / {v['nll']:.3f} / {v['ece_top1']:.3f}"
        tt = "—"
        lg_ = os.path.join(HERE, "logs", f"night_G_{tag}.log")
        if os.path.exists(lg_):
            mm = re.findall(r"^done ([0-9.]+) min", open(lg_).read(), re.M)
            tt = f"{mm[-1]} min" if mm else "—"
        lines.append(f"| {item} | {desc} | {left} | {right} | {tt} |")
    return "\n".join(lines)


def pwin_line():
    fn = os.path.join(D, "preds_laya_pwin8_gen8randombattle.pkl")
    if not os.path.exists(fn):
        return "L2 P(win) model: not trained yet."
    d = pickle.load(open(os.path.join(D, "ds_gen8randombattle_slim.pkl"), "rb"))
    te = [r for r in d["pwin"] if r["split"] == "test"]
    x = pickle.load(open(fn, "rb"))
    idx, lg, T = x["idx"][("pwin", "test")], x["preds"][("pwin", "test")], x["temps"]["pwin"]
    p = [softmax(l, T)[1] for l in lg]
    b = Me.binary(p, [te[i]["y"] for i in idx])
    base = json.load(open(os.path.join(D, "base_gen8randombattle.json")))["pwin"]
    return (f"L2 fine-tuned Laya P(win) (gen8, 10k rows) on gen8 test (n={b['n']}): Brier {b['brier']}, AUROC {b['auroc']}, ECE {b['ece']:.3f}. "
            f"Reference on the full gen8 test set: base rate Brier {base['base_rate']['brier']}; raw HP balance Brier {base['hp_raw']['brier']}, AUROC {base['hp_raw']['auroc']}.")


def bench_lines():
    out = []
    for fn, lab in (("logs/bench_laya.log", "laya 421M (gen9 ckpt, day run)"), ("logs/night_bench_421.log", "laya 421M (gen8 ckpt)"),
                    ("logs/night_bench_ml.log", "laya-multilingual 322M")):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            ms = re.findall(r"batch (\d+): ([0-9.]+) ms", open(p).read())
            if ms:
                out.append(f"- {lab}: " + ", ".join(f"batch {b}: {m} ms" for b, m in ms))
    return "\n".join(out) or "- (latency benchmarks pending)"


def analysis_lines():
    p = os.path.join(D, "night_analysis.json")
    if not os.path.exists(p):
        return "(pending)"
    a = json.load(open(p))
    out = []
    for k, lab in (("gen8_bot_test", "held-out gen8 bot battles"), ("abyssal_real_moves", "Abyssal's real moves")):
        if k not in a:
            continue
        v = a[k]
        out.append(f"- **{lab}** (n={v['n']}): top-1 {v['top1']:.3f}, top-2 {v['top2']:.3f}, top-3 {v['top3']:.3f} (uniform top-1 {v['uniform_top1']:.3f}). "
                   "By turn: " + ", ".join(f"{t} {x['acc']:.2f} (n={x['n']})" for t, x in v["by_turn"].items()) + ". "
                   "By # options: " + ", ".join(f"{t} {x['acc']:.2f} vs chance {x['chance']:.2f}" for t, x in v["by_n_options"].items()) + ". "
                   f"Actual attacks {v['acc_actual_attack_move']['acc']:.2f} (n={v['acc_actual_attack_move']['n']}), "
                   f"actual switches {v['acc_actual_switch']['acc']:.2f} (n={v['acc_actual_switch']['n']}); "
                   f"P(switch) Brier {v['switch_forecast']['brier']}, AUROC {v['switch_forecast']['auroc']}.")
    return "\n".join(out)


def queue_log():
    p = os.path.join(HERE, "logs", "night_queue.log")
    return "\n".join("    " + l for l in open(p).read().splitlines()[-40:]) if os.path.exists(p) else "    (not started)"


def main():
    sec = f"""{START}
## Overnight batch (2026-09-26/27; one job at a time; auto-updated by `night_summary.py`)

All runs use the gen8-trained Laya opponent model unless stated. Settings: depth-2 expectimax, HP leaf, vs PokéChamp's AbyssalPlayer, gen8, dynamax off, 20-battle chunks. Win-rate CIs are 95% Wilson. "partial" means the arm is still running or was stopped; those counts come from the per-battle log.

### Battles
{battles_table()}

### Forecasting (gen8; same recipe as the day run: top 4 layers + head, RLCD loss, batch 4, temperature fitted on validation)
{forecast_table()}

{pwin_line()}

Latency, single process, ms per opponent forecast:
{bench_lines()}

### Item 0: error analysis of the gen8 Laya opponent model (figures in `figures/`)
{analysis_lines()}
- Laya almost never ranks "switch" first. Actual switches are rare among scorable turns (most switches go to unseen Pokémon and are unscorable), and top-1 accuracy on them is 0. The P(switch) probabilities are still informative (AUROC above).
- Figures: `figures/reliability_laya_gen8.png` (top-1 confidence and P(switch) reliability), `figures/accuracy_breakdown_laya_gen8.png`.

### Queue log (tail)
{queue_log()}
{END}"""
    p = os.path.join(HERE, "RESULT.md")
    txt = open(p).read()
    if START in txt:
        txt = txt[:txt.index(START)] + sec + txt[txt.index(END) + len(END):]
    else:
        txt = txt.rstrip() + "\n\n" + sec + "\n"
    open(p, "w").write(txt)


if __name__ == "__main__":
    main()
