"""Aggregate runs/arm*.json + runs/usage.jsonl into runs/results.json and print a summary table."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from data import load, correct

RUNS = Path(__file__).parent / "runs"
qs = {q["id"]: q for q in load()}
ARMNAME = {"1": "single_cot", "2": "self_consistency_5", "3": "debate_homog_3x2", "4": "triggered_debate_2a", "5": "debate_hetero_2b"}
DS = ["mmlu_pro", "gsm8k", "all"]


def mean(xs):
    xs = list(xs); return sum(xs) / len(xs) if xs else float("nan")


def stats(recs):
    return {"n": len(recs), "acc": mean(r["correct"] for r in recs),
            "avg_total_tokens": mean(r["prompt_tokens"] + r["completion_tokens"] for r in recs),
            "avg_cost_usd": mean(r["cost"] for r in recs), "avg_calls": mean(r["calls"] for r in recs)}


def subset(d, ds):
    return [r for k, r in d.items() if ds == "all" or qs[k]["dataset"] == ds]


results = {"arms": {}, "arm4": {}, "arm5": {}, "arm3": {}, "matched_cost": {}, "notes": []}
arms = {a: json.loads((RUNS / f"arm{a}.json").read_text()) for a in "12345" if (RUNS / f"arm{a}.json").exists()}
for a, d in arms.items():
    results["arms"][ARMNAME[a]] = {ds: stats(subset(d, ds)) for ds in DS}

# Arm 3 / 5: per-round majority accuracy + per-agent accuracy
for a in ("3", "5"):
    if a not in arms:
        continue
    d = arms[a]
    out = {}
    for ds in DS:
        recs = subset(d, ds)
        if not recs:
            continue
        nr = len(recs[0]["per_round_answers"])
        na = len(recs[0]["models"])
        out[ds] = {"majority_acc_by_round": [mean(r["per_round_majority_correct"][k] for r in recs) for k in range(nr)],
                   "agent_acc_by_round": {recs[0]["models"][i] + f"#{i}": [mean(r["per_round_correct"][k][i] for r in recs) for k in range(nr)] for i in range(na)},
                   "round0_all_agree_frac": mean(len(set(r["per_round_answers"][0])) == 1 for r in recs)}
    results["arm" + a] = out

# Arm 4: trigger fraction and subset accuracies; compare with arm 2 and arm 3 on same questions
if "4" in arms:
    d = arms["4"]
    out = {}
    for ds in DS:
        recs = {k: r for k, r in d.items() if ds == "all" or qs[k]["dataset"] == ds}
        trig = {k: r for k, r in recs.items() if r["triggered"]}
        untrig = {k: r for k, r in recs.items() if not r["triggered"]}
        o = {"n": len(recs), "triggered_frac": len(trig) / max(len(recs), 1),
             "acc_triggered": mean(r["correct"] for r in trig.values()),
             "acc_untriggered": mean(r["correct"] for r in untrig.values()),
             "sc_on_own_samples_acc_triggered": mean(r["sc_correct"] for r in trig.values()),
             "sc_on_own_samples_acc_all": mean(r["sc_correct"] for r in recs.values()),
             "avg_calls_triggered": mean(r["calls"] for r in trig.values()),
             "n_positions_when_triggered": Counter(len(r["positions"]) for r in trig.values())}
        for other in ("2", "3"):
            if other in arms:
                o[f"arm{other}_acc_on_triggered_subset"] = mean(arms[other][k]["correct"] for k in trig if k in arms[other])
                o[f"arm{other}_acc_on_untriggered_subset"] = mean(arms[other][k]["correct"] for k in untrig if k in arms[other])
        if "3" in arms:
            o["arm3_round1_majority_acc_on_triggered_subset"] = mean(arms["3"][k]["per_round_majority_correct"][1] for k in trig if k in arms["3"])
        out[ds] = o
    results["arm4"] = out

# Matched-cost view: accuracy vs avg tokens/cost for arm2, arm4, arm3 (after 1 round: 6 calls, and after 2 rounds: 9 calls)
if "3" in arms:
    for ds in DS:
        recs = subset(arms["3"], ds)
        # approximate cost of arm3 truncated at round 1 from usage log (rounds 0 and 1 only)
        results["matched_cost"][ds] = {"arm3_round1_majority_acc": mean(r["per_round_majority_correct"][1] for r in recs),
                                       "arm3_round2_majority_acc": mean(r["per_round_majority_correct"][2] for r in recs)}
usage = [json.loads(l) for l in (RUNS / "usage.jsonl").read_text().splitlines() if l.strip()]
by = defaultdict(lambda: {"cost": 0.0, "prompt": 0, "completion": 0, "calls": 0})
for u in usage:
    key = (u["arm"], u["dataset"], u.get("round", -1))
    by[key]["cost"] += u["cost"]; by[key]["prompt"] += u["prompt_tokens"]; by[key]["completion"] += u["completion_tokens"]; by[key]["calls"] += 1
for ds in ["mmlu_pro", "gsm8k"]:
    n = 60
    for a in ("3", "5"):
        if a in arms:
            r01 = [by[(a, ds, k)] for k in (0, 1)]
            results["matched_cost"].setdefault(ds, {})[f"arm{a}_through_round1_avg_cost"] = sum(x["cost"] for x in r01) / n
            results["matched_cost"][ds][f"arm{a}_through_round1_avg_tokens"] = sum(x["prompt"] + x["completion"] for x in r01) / n
results["total_cost_usd"] = sum(u["cost"] for u in usage)
results["cost_by_arm_usd"] = {a: sum(u["cost"] for u in usage if u["arm"] == a) for a in "12345"}
results["cost_by_model_usd"] = dict(Counter({m: 0 for m in set(u["model"] for u in usage)}) )
for u in usage:
    results["cost_by_model_usd"][u["model"]] += u["cost"]
results["notes"] = ["Grading: exact match on final 'ANSWER: X' line (fallback: last \\boxed{X} / 'answer is X'); unparsable => wrong.",
                    "Arm 4 reuses no arm-2 samples; it draws its own 5 samples at T=0.7 (same seed for questions, not for sampling).",
                    "Arm 3/5 rounds: round0 = independent answers, round1/round2 = revisions after seeing other agents.",
                    "Costs are OpenRouter-reported USD per request."]
(RUNS / "results.json").write_text(json.dumps(results, indent=1, default=str))

print(f"{'arm':<22}{'dataset':<10}{'acc':>7}{'tok/q':>9}{'$/q':>10}{'calls/q':>9}")
for a, v in results["arms"].items():
    for ds in DS:
        s = v[ds]
        print(f"{a:<22}{ds:<10}{s['acc']:7.3f}{s['avg_total_tokens']:9.0f}{s['avg_cost_usd']:10.5f}{s['avg_calls']:9.1f}")
print(f"\nTOTAL COST ${results['total_cost_usd']:.3f}   by arm: { {k: round(v,3) for k,v in results['cost_by_arm_usd'].items()} }")
print("\nARM4:", json.dumps(results["arm4"], indent=1, default=str))
print("\nARM3:", json.dumps(results["arm3"], indent=1))
print("\nARM5:", json.dumps(results["arm5"], indent=1))
print("\nMATCHED COST:", json.dumps(results["matched_cost"], indent=1))
