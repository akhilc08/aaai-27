"""Calibration for exp2: find a transcript length / summary budget where the
1-round source-anchored arm is not already saturated at ~100% violation.
Conditional rules only, 1 rep. Writes results/calib_exp2.jsonl."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.RESULTS_DIR, "calib_exp2.jsonl")
GRID = [(T, B) for T in [32, 48] for B in [80, 160]]
KEY = ["total_turns", "budget", "regime", "rounds", "scenario", "backbone"]


def trial(T, B, regime, rounds, sid, bb):
    sc = C.SCENARIOS[sid]
    turns = [sc["policy"]] + C.POOL[:T - 1]
    s = C.run_session(bb, turns, rounds, regime, tail_k=2, budget_words=B)
    ctx = C.render_context(s["summary"], s["tail"]).lower()
    d, raw, u = C.agent_decide(bb, s["summary"], s["tail"], sc["trigger"])
    C.append_result(OUT, {"total_turns": T, "budget": B, "regime": regime, "rounds": rounds, "scenario": sid,
                          "backbone": bb, "decision": d, "rule_kw_present": C.COND_KW[sid].lower() in ctx,
                          "tokens_compaction": s["usage"]["prompt"] + s["usage"]["completion"], "summary": s["summary"]})


def main():
    done = C.done_keys(OUT, KEY)
    jobs = [(trial, (T, B, rg, rd, sid, bb)) for T, B in GRID for rg in ["recursive", "source"] for rd in [1, 4]
            for sid in C.VALID_IDS for bb in ["qwen", "minimax"] if (T, B, rg, rd, sid, bb) not in done]
    C.run_jobs(jobs, workers=12, label="calib")
    rows = C.load_results(OUT)
    print("\nviolation % (n=12 per cell: 6 scenarios x 2 backbones)")
    print(f"{'T':>3} {'B':>4} | {'rec r1':>7} {'rec r4':>7} | {'src r1':>7} {'src r4':>7} || kw present: rec r1  rec r4  src r1  src r4")
    for T, B in GRID:
        def cell(rg, rd, f=lambda r: r["decision"] == "COMPLY"):
            c = [r for r in rows if r["total_turns"] == T and r["budget"] == B and r["regime"] == rg and r["rounds"] == rd]
            return f"{100 * sum(f(r) for r in c) / len(c):5.0f}%" if c else "   -  "
        print(f"{T:>3} {B:>4} | {cell('recursive', 1):>7} {cell('recursive', 4):>7} | {cell('source', 1):>7} {cell('source', 4):>7} ||"
              f"  {cell('recursive', 1, lambda r: r['rule_kw_present'])}  {cell('recursive', 4, lambda r: r['rule_kw_present'])}  "
              f"{cell('source', 1, lambda r: r['rule_kw_present'])}  {cell('source', 4, lambda r: r['rule_kw_present'])}")


if __name__ == "__main__":
    main()
