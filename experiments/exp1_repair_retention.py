"""Idea 1: repair-compute vs retention-compute at matched token cost.

Base cell mirrors the Watchpoint paper's harshest condition: recursive, 3 rounds
of 16 turns, rule at turn 1, 2-turn verbatim tail, 80-word summary budget.

Retention arm: summary budget in {80, 160, 320, 640} words, no repair.
Repair arm:    80-word budget, k in {1, 2, 4} generic no-oracle self-check passes
               whose notes are appended to the context before the agent acts.
Each trial records total tokens (compaction + repair + decision) and whether the
rule's clause keyword was still present in the compacted context *before* repair,
so failures can be split into retrieval-type (present) vs deletion-type (absent).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.RESULTS_DIR, "exp1_repair_retention.jsonl")
ROUNDS, TURNS_PER_ROUND = 3, 16
CONDITIONS = [("retention", 80, 0), ("retention", 160, 0), ("retention", 320, 0), ("retention", 640, 0),
              ("repair", 80, 1), ("repair", 80, 2), ("repair", 80, 4)]
BACKBONES = ["qwen", "minimax"]
REPS = 4
KEY_FIELDS = ["arm", "budget", "k", "scenario", "backbone", "rep"]


def transcript(sid):
    return [C.SCENARIOS[sid]["policy"]] + C.POOL[:ROUNDS * TURNS_PER_ROUND - 1]


def trial(arm, budget, k, sid, backbone, rep):
    sc = C.SCENARIOS[sid]
    s = C.run_session(backbone, transcript(sid), ROUNDS, "recursive", tail_k=2, budget_words=budget)
    ctx = C.render_context(s["summary"], s["tail"]).lower()
    present = C.COND_KW[sid].lower() in ctx
    notes, ru = C.repair_passes(backbone, s["summary"], s["tail"], k) if k else ([], {"prompt": 0, "completion": 0})
    decision, raw, du = C.agent_decide(backbone, s["summary"], s["tail"], sc["trigger"], notes)
    notes_text = " ".join(notes).lower()
    row = {"arm": arm, "budget": budget, "k": k, "scenario": sid, "backbone": backbone, "rep": rep,
           "decision": decision, "rule_kw_present": present,
           "topic_kw_present": C.TOPIC_KW[sid].lower() in ctx,
           "repair_recovered_kw": (C.COND_KW[sid].lower() in notes_text) if k else None,
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
    for arm, budget, k in CONDITIONS:
        for sid in C.VALID_IDS:
            for bb in BACKBONES:
                for rep in range(REPS):
                    if (arm, budget, k, sid, bb, rep) not in done:
                        jobs.append((trial, (arm, budget, k, sid, bb, rep)))
    print(f"exp1: {len(done)} done, {len(jobs)} to run")
    C.run_jobs(jobs, workers=12, label="exp1")


if __name__ == "__main__":
    main()
