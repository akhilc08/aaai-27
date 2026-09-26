"""Idea 3 (minimal): ratchet vs rescue.

Take the most recursed condition from exp2 (4 rounds x 16 turns = 64-turn session) and the
source-anchored control, and ask whether damage can be rescued by repair passes
(k=4, no oracle) or only by retention (320-word budget). Conditional rules only,
one backbone, to keep it cheap. Prediction: repair helps when the clause is still
present (source regime) and does nothing when it was deleted (recursive regime).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.RESULTS_DIR, "exp3_ratchet_rescue.jsonl")
ROUNDS = 4
REGIMES = ["recursive", "source"]
ARMS = [("plain", 80, 0), ("repair", 80, 4), ("retention", 320, 0)]
BACKBONE = "qwen"
REPS = 4
KEY_FIELDS = ["regime", "arm", "scenario", "rep"]


def trial(regime, arm, budget, k, sid, rep):
    sc = C.SCENARIOS[sid]
    turns = [sc["policy"]] + C.POOL[:16 * ROUNDS - 1]
    s = C.run_session(BACKBONE, turns, ROUNDS, regime, tail_k=2, budget_words=budget)
    ctx = C.render_context(s["summary"], s["tail"]).lower()
    present = C.COND_KW[sid].lower() in ctx
    notes, ru = C.repair_passes(BACKBONE, s["summary"], s["tail"], k) if k else ([], {"prompt": 0, "completion": 0})
    decision, raw, du = C.agent_decide(BACKBONE, s["summary"], s["tail"], sc["trigger"], notes)
    row = {"regime": regime, "arm": arm, "budget": budget, "k": k, "scenario": sid, "backbone": BACKBONE, "rep": rep,
           "decision": decision, "raw": raw, "rule_kw_present": present,
           "repair_recovered_kw": (C.COND_KW[sid].lower() in " ".join(notes).lower()) if k else None,
           "tokens_compaction": s["usage"]["prompt"] + s["usage"]["completion"],
           "tokens_repair": ru["prompt"] + ru["completion"],
           "tokens_decision": du["prompt"] + du["completion"],
           "summary": s["summary"], "notes": notes}
    row["tokens_total"] = row["tokens_compaction"] + row["tokens_repair"] + row["tokens_decision"]
    C.append_result(OUT, row)


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    done = C.done_keys(OUT, KEY_FIELDS)
    jobs = []
    for regime in REGIMES:
        for arm, budget, k in ARMS:
            for sid in C.VALID_IDS:
                for rep in range(REPS):
                    if (regime, arm, sid, rep) not in done:
                        jobs.append((trial, (regime, arm, budget, k, sid, rep)))
    print(f"exp3: {len(done)} done, {len(jobs)} to run")
    C.run_jobs(jobs, workers=12, label="exp3")


if __name__ == "__main__":
    main()
