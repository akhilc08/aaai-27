"""134-game analysis: Wilson CI, McNemar (exact) vs A, per-task success, mismatch-before-failure -> runs/results_134.json"""
import json, math
from pathlib import Path
from math import comb
RUNS = Path(__file__).resolve().parent / "runs"
ARMS = ["A", "B", "B2", "Bnc"]

def load(a):
    return {json.loads(l)["game"]: json.loads(l) for l in (RUNS / f"arm_{a}.jsonl").read_text().splitlines() if l.strip()}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def analyze(arms, base):
    data = {a: load(a) for a in arms}
    games = sorted(set.intersection(*(set(d) for d in data.values())))
    res = {"n_games": len(games), "arms": {}}
    for a in arms:
        eps = [data[a][g] for g in games]; n = len(eps); k = sum(e["success"] for e in eps)
        s = dict(success=k, success_rate=round(k / n, 3), wilson95=wilson(k, n),
                 avg_steps=round(sum(e["steps"] for e in eps) / n, 2),
                 agent_calls=round(sum(e["llm_calls"] - e["judge_calls"] for e in eps) / n, 2),
                 judge_calls=round(sum(e["judge_calls"] for e in eps) / n, 2),
                 tokens=round(sum(e["prompt_tokens"] + e["completion_tokens"] for e in eps) / n),
                 cost=round(sum(e["cost"] for e in eps), 3), by_task={})
        for tt in sorted({e["task_type"] for e in eps}):
            sub = [e for e in eps if e["task_type"] == tt]
            s["by_task"][tt] = f"{sum(e['success'] for e in sub)}/{len(sub)}"
        if a != base:
            b_ = sum(data[a][g]["success"] and not data[base][g]["success"] for g in games)
            c_ = sum(data[base][g]["success"] and not data[a][g]["success"] for g in games)
            s["mcnemar_vs_" + base] = dict(arm_only=b_, base_only=c_, p_exact=round(mcnemar_exact(b_, c_), 4))
            s["replans"] = round(sum(e["replans"] for e in eps) / n, 2)
            s["mismatches"] = round(sum(e["mismatches"] for e in eps) / n, 2)
            s["cached_executed"] = round(sum(e["cached_executed"] for e in eps) / n, 2)
            late = lambda e: any(e["steps"] - 3 < m <= e["steps"] for m in e["mismatch_steps"])
            fails = [e for e in eps if not e["success"]]; succ = [e for e in eps if e["success"]]
            s["late_mismatch_fail"] = f"{sum(map(late, fails))}/{len(fails)}"
            s["late_mismatch_succ"] = f"{sum(map(late, succ))}/{len(succ)}"
            def rate(group):
                j = sum(1 for e in group for t in e["transcript"] if t.get("match") is not None and t.get("pred"))
                return round(sum(e["mismatches"] for e in group) / max(j, 1), 3)
            s["mismatch_rate_per_judged_step_fail"] = rate(fails); s["mismatch_rate_per_judged_step_succ"] = rate(succ)
            s["any_mismatch_fail"] = f"{sum(e['mismatches'] > 0 for e in fails)}/{len(fails)}"
            s["any_mismatch_succ"] = f"{sum(e['mismatches'] > 0 for e in succ)}/{len(succ)}"
            s["replan_fail"] = f"{sum(e['replans'] > 0 for e in fails)}/{len(fails)}"
            s["replan_succ"] = f"{sum(e['replans'] > 0 for e in succ)}/{len(succ)}"
        res["arms"][a] = s
    return res, games

out = {}
out["qwen3-30b-a3b-instruct-2507"], _ = analyze(ARMS, "A")
if (RUNS / "arm_A_ds.jsonl").exists() and (RUNS / "arm_B_ds.jsonl").exists():
    out["deepseek-chat"], ds_games = analyze(["A_ds", "B_ds"], "A_ds")
    # qwen A/B restricted to the same games, for a like-for-like comparison
    sub = {}
    for a in ["A", "B"]:
        d = load(a); eps = [d[g] for g in ds_games]
        sub[a] = f"{sum(e['success'] for e in eps)}/{len(eps)}"
    out["deepseek-chat"]["qwen_on_same_games"] = sub
out["total_spend_usd"] = round(sum(json.loads(l)["cost"] for l in (RUNS / "usage.jsonl").read_text().splitlines() if l.strip()), 3)
(RUNS / "results_134.json").write_text(json.dumps(out, indent=2))
for bk, r in out.items():
    if not isinstance(r, dict): continue
    print(f"== {bk} (n={r['n_games']})")
    for a, s in r["arms"].items():
        mc = next((v for k, v in s.items() if k.startswith("mcnemar")), None)
        print(f"{a:6s} {s['success']:3d} {s['success_rate']:.3f} CI{s['wilson95']} steps={s['avg_steps']} agent={s['agent_calls']} judge={s['judge_calls']} tok={s['tokens']} cost=${s['cost']} {mc or ''}")
        print("       ", s["by_task"])
        if "replans" in s:
            print("       ", {k: s[k] for k in s if k.startswith(("late_", "mismatch_rate", "any_mis", "replan", "cached"))})
print("total spend", out["total_spend_usd"])
