"""Idea 2 + Idea 5: does recursive compaction ratchet away rules, and does the
conditional clause go first?

Design (v2, after calibration showed a fixed-length transcript saturates):
growing session, compaction every 16 turns, rule at turn 1. Rounds R in {1,2,3,4}
means a 16R-turn session compacted R times. Recursive: each round summarizes
(prior summary + new turns). Source-anchored: each round re-summarizes the raw
transcript so far. At matched R both see identical content, so the gap between
them isolates the recursion penalty (idea 2) from plain session length (idea 5).
Factors: regime x R x constraint type {cond, uncond} x summary budget {80,160}
x 6 scenarios x 2 backbones x REPS. Plus a no-compaction ceiling at 64 turns.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.RESULTS_DIR, "exp2_ratchet.jsonl")
REGIMES = ["recursive", "source"]
ROUNDS = [1, 2, 3, 4]
TURNS_PER_ROUND = 16
BUDGETS = [80, 160]
CTYPES = ["cond", "uncond"]
BACKBONES = ["qwen", "minimax"]
REPS = 3
KEY_FIELDS = ["regime", "rounds", "budget", "ctype", "scenario", "backbone", "rep"]


def transcript(sid, ctype, rounds):
    return [C.policy_for(sid, ctype)] + C.POOL[:TURNS_PER_ROUND * rounds - 1]


def trial(regime, rounds, budget, ctype, sid, backbone, rep):
    sc = C.SCENARIOS[sid]
    turns = transcript(sid, ctype, max(rounds, 1) if regime != "none" else max(ROUNDS))
    if regime == "none":
        decision, raw, u = C.decide_full_context(backbone, turns, sc["trigger"])
        row = {"regime": regime, "rounds": 0, "budget": 0, "ctype": ctype, "scenario": sid, "backbone": backbone, "rep": rep,
               "decision": decision, "rule_kw_present": True, "topic_kw_present": True,
               "tokens_compaction": 0, "tokens_decision": u["prompt"] + u["completion"], "summary": ""}
        C.append_result(OUT, row)
        return
    s = C.run_session(backbone, turns, rounds, regime, tail_k=2, budget_words=budget)
    ctx = C.render_context(s["summary"], s["tail"]).lower()
    decision, raw, u = C.agent_decide(backbone, s["summary"], s["tail"], sc["trigger"])
    row = {"regime": regime, "rounds": rounds, "budget": budget, "ctype": ctype, "scenario": sid, "backbone": backbone,
           "rep": rep, "total_turns": len(turns), "decision": decision, "tail": s["tail"],
           "rule_kw_present": C.rule_kw(sid, ctype).lower() in ctx,
           "topic_kw_present": C.TOPIC_KW[sid].lower() in ctx,
           "cond_clause_present": (C.COND_KW[sid].lower() in ctx) if ctype == "cond" else None,
           "tokens_compaction": s["usage"]["prompt"] + s["usage"]["completion"],
           "tokens_decision": u["prompt"] + u["completion"],
           "summary": s["summary"], "round_summaries": s["round_summaries"]}
    C.append_result(OUT, row)


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    done = C.done_keys(OUT, KEY_FIELDS)
    jobs = []
    for sid in C.VALID_IDS:
        for ctype in CTYPES:
            for bb in BACKBONES:
                for rep in range(2):
                    if ("none", 0, 0, ctype, sid, bb, rep) not in done:
                        jobs.append((trial, ("none", 0, 0, ctype, sid, bb, rep)))
                for regime in REGIMES:
                    for rounds in ROUNDS:
                        for budget in BUDGETS:
                            for rep in range(REPS):
                                if (regime, rounds, budget, ctype, sid, bb, rep) not in done:
                                    jobs.append((trial, (regime, rounds, budget, ctype, sid, bb, rep)))
    print(f"exp2: {len(done)} done, {len(jobs)} to run")
    C.run_jobs(jobs, workers=12, label="exp2")


if __name__ == "__main__":
    main()
