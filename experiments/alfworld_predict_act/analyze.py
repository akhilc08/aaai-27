"""Aggregate runs/arm_*.jsonl into runs/results.json and print a table."""
import sys, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS

def load(arm):
    p = RUNS / f"arm_{arm}.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def summarize(eps):
    n = len(eps)
    if not n: return None
    mean = lambda k: sum(e[k] for e in eps) / n
    s = dict(n=n, success_rate=mean("success"), avg_steps=mean("steps"), avg_llm_calls=mean("llm_calls"),
             avg_agent_calls=(sum(e["llm_calls"] - e["judge_calls"] for e in eps) / n),
             avg_tokens=(sum(e["prompt_tokens"] + e["completion_tokens"] for e in eps) / n),
             total_cost=sum(e["cost"] for e in eps), avg_invalid_actions=mean("invalid_actions"),
             by_task={})
    for tt in sorted({e["task_type"] for e in eps}):
        sub = [e for e in eps if e["task_type"] == tt]
        s["by_task"][tt] = f"{sum(e['success'] for e in sub)}/{len(sub)}"
    if eps[0]["arm"] == "B":
        s["avg_replans"] = mean("replans"); s["avg_mismatches"] = mean("mismatches")
        s["episodes_with_replan"] = sum(e["replans"] > 0 for e in eps)
        s["avg_cached_executed"] = mean("cached_executed"); s["avg_batches"] = mean("batches")
        judged = sum(len(t.get("pred", "")) > 0 for e in eps for t in e["transcript"] if "pred" in t)
        s["mismatch_rate_per_judged_step"] = sum(e["mismatches"] for e in eps) / max(judged, 1)
        # mismatch within last 3 env steps before episode end
        def late_mismatch(e): return any(e["steps"] - 3 < ms <= e["steps"] for ms in e["mismatch_steps"])
        fails = [e for e in eps if not e["success"]]; succ = [e for e in eps if e["success"]]
        s["late_mismatch_in_failures"] = f"{sum(map(late_mismatch, fails))}/{len(fails)}"
        s["late_mismatch_in_successes"] = f"{sum(map(late_mismatch, succ))}/{len(succ)}"
        s["failures_by_cause"] = dict(Counter("hit_step_cap" if e["steps"] >= 30 else "hit_llm_cap" for e in fails))
    return s

if __name__ == "__main__":
    res = {arm: summarize(load(arm)) for arm in "ABC"}
    res = {k: v for k, v in res.items() if v}
    (RUNS / "results.json").write_text(json.dumps(res, indent=2))
    print(f"{'arm':4s} {'n':>3s} {'succ':>6s} {'steps':>6s} {'calls':>6s} {'agent':>6s} {'tok/game':>9s} {'cost$':>7s}")
    for arm, s in res.items():
        print(f"{arm:4s} {s['n']:3d} {s['success_rate']:6.2f} {s['avg_steps']:6.1f} {s['avg_llm_calls']:6.1f} {s['avg_agent_calls']:6.1f} {s['avg_tokens']:9.0f} {s['total_cost']:7.3f}")
    for arm, s in res.items():
        print(arm, {k: v for k, v in s.items() if k not in ("n", "success_rate", "avg_steps", "avg_llm_calls", "avg_agent_calls", "avg_tokens", "total_cost")})
