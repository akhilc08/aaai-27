"""Round 3 stage 2: decisions under each per-rule gate. Decisions cached by (context, trigger, pinned set, rep)."""
import json, os, numpy as np, common3 as C
CT = [json.loads(l) for l in open(os.path.join(C.D, "contexts.jsonl"))]
CTD = {c["cid"]: c for c in CT}
REAL = list(C.REAL); REPS = int(os.environ.get("REPS", 2)); CACHE_F = os.path.join(C.D, "decisions.jsonl")
cache = {}
if os.path.exists(CACHE_F):
    for d in map(json.loads, open(CACHE_F)): cache[(d["cid"], d["sid"], tuple(d["pinned"]), d["rep"])] = d
def run(sets):  # sets: {(cid,sid): pinned list in policy order}
    jobs = [(cid, sid, tuple(p), rep) for (cid, sid), p in sets.items() for rep in range(REPS) if (cid, sid, tuple(p), rep) not in cache]
    def one(j):
        cid, sid, p, rep = j; dec, pt = C.decide(CTD[cid], sid, list(p))
        return {"cid": cid, "sid": sid, "pinned": list(p), "rep": rep, "decision": dec, "prompt_tokens": pt}
    res = C.J.pmap(one, jobs, workers=12)
    with open(CACHE_F, "a") as f:
        for d in res:
            if isinstance(d, Exception): print("ERR", d); continue
            cache[(d["cid"], d["sid"], tuple(d["pinned"]), d["rep"])] = d; f.write(json.dumps(d) + "\n")
def gate_sets(fn):  # fn(context) -> set of rule ids to pin
    return {(c["cid"], s): [i for i in c["order"] if i in fn(c)] for c in CT for s in REAL}
thr_keep = lambda t: (lambda c: {i for i, p in c["jev_pkept"].items() if p < t})
G = {"plain": gate_sets(lambda c: set()), "always-all": gate_sets(lambda c: set(C.RID))}
run(G["plain"]); run(G["always-all"])
units = [(c["cid"], s) for c in CT for s in REAL]
viol = {u: np.mean([cache[(u[0], u[1], (), r)]["decision"] == "COMPLY" for r in range(REPS)]) for u in units}
pk = {(c["cid"], s): c["jev_pkept"][s] for c in CT for s in REAL}
# LOSO threshold: smallest t (fewest re-insertions) whose training-scenario recall of plain violations >= target
def tuned(target):
    thr = {}
    for s in REAL:
        tr = [u for u in units if u[1] != s]; w = np.array([viol[u] for u in tr]); p = np.array([pk[u] for u in tr])
        for t in np.arange(0.30, 1.001, 0.01):
            if (w * (p < t)).sum() >= target * w.sum(): thr[s] = round(t, 2); break
    return thr
G["keyword-per-rule"] = gate_sets(lambda c: set(c["kw_missing"]))
G["LLM-per-rule"] = gate_sets(lambda c: set(c["llm_missing"]))
G["Jev-per-rule @0.5"] = gate_sets(thr_keep(0.5))
THR = {}
for tgt in (0.8, 0.9):
    THR[tgt] = tuned(tgt)
    G[f"Jev-per-rule LOSO @recall{tgt}"] = {(c["cid"], s): [i for i in c["order"] if c["jev_pkept"][i] < THR[tgt][s]] for c in CT for s in REAL}
print("LOSO thresholds", THR)
for g in G.values(): run(g)
json.dump({"thr": {str(k): v for k, v in THR.items()}}, open(os.path.join(C.D, "thresholds.json"), "w"))
base_pt = {u: np.mean([cache[(u[0], u[1], (), r)]["prompt_tokens"] for r in range(REPS)]) for u in units}
out = []
for name, sets in G.items():
    v = np.mean([cache[(u[0], u[1], tuple(sets[u]), r)]["decision"] == "COMPLY" for u in units for r in range(REPS)])
    trig = np.mean([u[1] in sets[u] for u in units])
    nre = np.mean([len(sets[u]) for u in units])
    xt = np.mean([cache[(u[0], u[1], tuple(sets[u]), r)]["prompt_tokens"] - base_pt[u] for u in units for r in range(REPS)])
    row = dict(gate=name, viol=round(v, 3), triggered_rule_reinserted=round(trig, 3), rules_reinserted=round(nre, 2), extra_tokens=round(xt, 1))
    out.append(row); print(row)
json.dump(out, open(os.path.join(C.D, "metrics_r3.json"), "w"), indent=1)
unk = sum(d["decision"] == "UNKNOWN" for d in cache.values()); print("UNKNOWN decisions", unk, "of", len(cache))
print("spend r3", C.J.spend(C.TAG))
