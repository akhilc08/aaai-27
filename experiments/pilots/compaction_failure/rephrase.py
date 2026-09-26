"""Round 2: held-out rule wording. Paraphrase each policy once (cheap LLM), then re-ask Jev's 6 questions
with the paraphrased rule in the state (same summaries/tails)."""
import json, sys, os, time
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots")
import jevlib as J
from features import SC, context, JQ
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "compaction_failure_r2_rephrase"
PF = os.path.join(HERE, "rephrased_rules.json")
if not os.path.exists(PF):
    def para(pol):
        out, _ = J.llm([{"role": "user", "content": "Rewrite this workplace rule with completely different wording and sentence structure "
                         "but exactly the same meaning (keep any condition/exception). Do not start with 'POLICY' or 'Never'. "
                         "Output only the rewritten rule.\n\n" + pol}], max_tokens=120, temperature=0.7, tag=TAG)
        return out.strip()
    keys = [(s, c) for s in SC for c in (0, 1)]
    outs = J.pmap(lambda k: para(SC[k[0]][k[1]]), keys)
    json.dump({f"{s}|{'cond' if c == 0 else 'uncond'}": o for (s, c), o in zip(keys, outs)}, open(PF, "w"), indent=1)
NEW = json.load(open(PF))
R = [json.loads(l) for l in open(os.path.join(HERE, "data_snapshot.jsonl"))]
R = [r for r in R if r["regime"] != "none" and r["decision"] != "UNKNOWN"]
def one(i):
    r = R[i]
    state = {"original_rule_given_at_turn_1": NEW[f"{r['scenario']}|{r['ctype']}"],
             "agent_context_after_compaction": context(r), "new_request": SC[r["scenario"]][2]}
    a = J.jev(state, {k: {"type": "noul", "instructions": v} for k, v in JQ.items()}, tag=TAG)
    return {"row": i} | {k: a[k]["noul"] for k in JQ}
res = J.pmap(one, range(len(R)), workers=8)
bad = [x for x in res if isinstance(x, Exception)]; print("errors", len(bad), bad[:2])
with open(os.path.join(HERE, "rephrase_features.jsonl"), "w") as f:
    for x in res:
        if not isinstance(x, Exception): f.write(json.dumps(x) + "\n")
print("spend", J.spend(TAG))
