"""Round 3 extras: matched-budget Jev gate, random gate, per-rule AUROC, spillover diagnostic."""
import json, os, random, numpy as np, common3 as C
from sklearn.metrics import roc_auc_score as A
import stage2 as S  # re-uses cache; re-running stage2 main is cheap (all cached)
CT, units, cache, REPS = S.CT, S.units, S.cache, S.REPS
allp = np.array([p for c in CT for p in c["jev_pkept"].values()])
t_m = np.quantile(allp, np.mean([len(c["kw_missing"]) for c in CT]) / 20)  # matches keyword gate's mean count
rng = random.Random(0)
G = {f"Jev-per-rule matched to keyword count (t={t_m:.2f})": S.gate_sets(S.thr_keep(t_m)),
     "random 14 rules": {(c["cid"], s): [i for i in c["order"] if i in set(rng.sample(c["order"], 14))] for c in CT for s in S.REAL}}
for g in G.values(): S.run(g)
for name, sets in G.items():
    v = np.mean([cache[(u[0], u[1], tuple(sets[u]), r)]["decision"] == "COMPLY" for u in units for r in range(REPS)])
    print(dict(gate=name, viol=round(v, 3), trig=round(np.mean([u[1] in sets[u] for u in units]), 3), n=round(np.mean([len(sets[u]) for u in units]), 2)))
y = np.array([S.viol[u] >= 0.5 for u in units]); ctx = {c["cid"]: c for c in CT}
print("units", len(units), "plain viol-majority rate", y.mean())
sc = {"keyword missing": [u[1] in ctx[u[0]]["kw_missing"] for u in units], "LLM says missing": [u[1] in ctx[u[0]]["llm_missing"] for u in units],
      "Jev 1-P(kept)": [1 - ctx[u[0]]["jev_pkept"][u[1]] for u in units]}
for k, s in sc.items(): print(f"AUROC triggered-rule {k}: {A(y, np.array(s, float)):.3f}")
# spillover: units where triggered rule NOT pinned, by whether other rules were pinned
for name in ["LLM-per-rule", "Jev-per-rule @0.5", "keyword-per-rule"]:
    sets = S.G[name]; nt = [u for u in units if u[1] not in sets[u]]
    vo = [cache[(u[0], u[1], tuple(sets[u]), r)]["decision"] == "COMPLY" for u in nt for r in range(REPS)]
    vp = [cache[(u[0], u[1], (), r)]["decision"] == "COMPLY" for u in nt for r in range(REPS)]
    print(f"spillover {name}: triggered rule not pinned (n={len(nt)}): viol with others pinned {np.mean(vo):.3f} vs plain {np.mean(vp):.3f}")
print("spend r3", C.J.spend(C.TAG))
