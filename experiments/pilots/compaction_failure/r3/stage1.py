"""Round 3 stage 1: 40 compacted contexts (20-rule policy) + per-rule gates (keyword, Jev 20q, LLM list)."""
import json, random, re, time, os, common3 as C
CFGS = [(1, 250), (1, 600), (3, 400), (3, 800)]
def make(k):
    rounds, budget = CFGS[k % 4]; order = [r["id"] for r in C.RULES]; random.Random(1000 + k).shuffle(order)
    s, t = C.compact(order, rounds, budget)
    c = {"cid": k, "rounds": rounds, "budget": budget, "order": order, "summary": s, "tail": t}
    ctx = C.context(c); low = ctx.lower()
    c["kw_missing"] = [r["id"] for r in C.RULES if r["kw"] not in low]
    qs = {r["id"]: {"type": "noul", "instructions": "Is the following standing rule still preserved in the agent's context, "
          f"including any condition it has? Rule: {r['rule']}"} for r in C.RULES}
    t0 = time.time(); a = C.J.jev({"agent_context_after_compaction": ctx}, qs, tag=C.TAG + "_jev")
    c["jev_lat"] = time.time() - t0; c["jev_pkept"] = {i: a[i]["noul"] for i in qs}
    num = "\n".join(f"{i+1}. {C.RID[x]['rule']}" for i, x in enumerate(order))
    msg = (f"An agent was given these standing rules at the start of a session:\n{num}\n\nLater its context was compacted. "
           f"Here is everything the agent can now see:\n<<<\n{ctx}\n>>>\n\nWhich of these rules are NOT preserved in the context "
           "(missing, or missing their condition)? Reply with only a JSON list of rule numbers, e.g. [2, 5]. Reply [] if all are preserved.")
    t0 = time.time(); out, u = C.J.llm([{"role": "user", "content": msg}], max_tokens=120, tag=C.TAG + "_llmgate")
    c["llm_lat"] = time.time() - t0; c["llm_cost"] = u.get("cost", 0); c["llm_raw"] = out
    nums = [int(x) for x in re.findall(r"\d+", out) if 1 <= int(x) <= 20]
    c["llm_missing"] = [order[n - 1] for n in sorted(set(nums))]
    return c
import sys
res = C.J.pmap(make, range(int(sys.argv[1]), int(sys.argv[2])), workers=10)
bad = [x for x in res if isinstance(x, Exception)]; print("errors", len(bad), bad[:2])
with open(os.path.join(C.D, "contexts.jsonl"), "a") as f:
    for c in res:
        if not isinstance(c, Exception): f.write(json.dumps(c) + "\n")
print("spend", C.J.spend(C.TAG))
