"""Round 2: all 134 valid_unseen games x seeds. D = LLM top-3, own #1; B = Jev picks; C = B + dense surprise replan;
V = same top-3, cheap LLM scores candidates with the same questions (one call per step)."""
import json, re, sys, time
import os
import alfenv, agent as A, wm
J = A.J
MODEL = os.environ.get("R2_MODEL", J.CHEAP)
CAP, BASE, CAPUSD, GLOBAL = 30, 6.311, 4.9, 24.3
PROP = A.SYS.split("Reply exactly")[0] + ("Propose the 3 best candidate next actions, copied exactly from the admissible list, best first. "
       'Reply with JSON only: {"think": "<one short sentence>", "actions": ["a1", "a2", "a3"]}')
PQ = "Text household game. Will the proposed next action make real progress toward completing the task (e.g. finding, taking, transforming or placing the needed object), rather than wasting a step?"
OUT = ("effect", "reveal", "empty")

def qs(tgt):
    q = wm.questions(tgt); q["progress"] = {"type": "noul", "instructions": PQ}; return q

def propose(obs, hist, adm, note, seed):
    t, u = J.llm([{"role": "system", "content": PROP}, {"role": "user", "content": A.render(obs, hist, adm, note)}],
                 max_tokens=150, temperature=0.7, seed=seed, model=MODEL, tag=A.TAG)
    try: c = json.loads(re.search(r"\{.*\}", t, re.S).group(0))["actions"]
    except Exception: c = [A.parse_action(t, adm)]
    c = [x.strip().lower() for x in c if isinstance(x, str)]
    ok = [x for x in dict.fromkeys(c) if x in adm]
    return (ok or c[:1] or ["look"])[:3]

def jev_score(st, tgt):
    a = J.jev(st, qs(tgt), tag=A.TAG); return {k: a[k]["noul"] for k in ("progress",) + OUT}

def llm_score(base, cands, tgt):
    q = qs(tgt); st = wm.state({**base, "action": "<see candidates>"}); st.pop("proposed_next_action")
    p = ("You are a world model. Predict outcomes of each candidate next action BEFORE it is executed. State:\n" + json.dumps(st, indent=1) +
         "\n\nCandidates:\n" + "\n".join(f"{i}: {c}" for i, c in enumerate(cands)) +
         "\n\nFor each candidate give probability 0-100 that the answer is yes to each question:\n" +
         "\n".join(f"{k}: {v['instructions']}" for k, v in q.items()) +
         '\nReply with JSON only: {"0": {"progress": p, "effect": p, "reveal": p, "empty": p}, "1": {...}, ...}')
    t, _ = J.llm([{"role": "user", "content": p}], max_tokens=60 * len(cands), tag=A.TAG)
    d = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
    return [{k: float(d[str(i)][k]) / 100 for k in ("progress",) + OUT} for i in range(len(cands))]

def run(args):
    gf, arm, seed = args
    if J.spend(A.TAG) > BASE + CAPUSD or J.spend("") > GLOBAL: return {"game": gf, "arm": arm, "seed": seed, "skipped": True}
    t0 = time.time(); tgt = A.target_of(gf)
    e = alfenv.Env(gf); obs, adm = e.reset(); hist, note, won, log, calls, jcalls, nsur, nover = [], None, False, [], 0, 0, 0, 0
    for t in range(CAP):
        if arm == "J":  # Jev alone: one call scoring every admissible action, no LLM proposer
            st = wm.state({"goal": e.goal, "init": obs, "hist": hist[-12:], "action": ""}); st.pop("proposed_next_action"); st["candidate_actions"] = adm
            ans = J.jev(st, {f"a{i}": {"type": "noul", "instructions": f'Text household game. If the agent executes the candidate action "{c}" next, will it make real progress toward completing the task?'}
                             for i, c in enumerate(adm)}, tag=A.TAG); jcalls += 1
            i = max(range(len(adm)), key=lambda i: ans[f"a{i}"]["noul"]); a = adm[i]
            o, won, adm2 = e.step(a); log.append({"t": t, "action": a, "obs": o, "p": ans[f"a{i}"]["noul"]}); hist.append((a, o)); adm = adm2
            if won: break
            continue
        cands = propose(obs, hist, adm, note if arm == "C" else None, seed * 1000 + t); calls += 1
        base = {"goal": e.goal, "target": tgt, "init": obs, "hist": hist[-12:]}; sc = None
        if arm in "BC":
            sc = J.pmap(lambda c: jev_score(wm.state({**base, "action": c}), tgt), cands, workers=3); jcalls += len(cands)
        elif arm == "V" and len(cands) > 1:
            try: sc = llm_score(base, cands, tgt)
            except Exception: sc = None
            calls += 1
        if sc:
            sc = [s if isinstance(s, dict) else {"progress": 0, "effect": .5, "reveal": .5, "empty": .5} for s in sc]
            i = max(range(len(cands)), key=lambda i: (sc[i]["progress"], -i))
        elif arm == "R":  # trivial rule: first LLM candidate not executed before and never answered 'Nothing happens.'
            past = {x for x, _ in hist}; nh = {x for x, y in hist if y == "Nothing happens."}
            i = next((j for j, c in enumerate(cands) if c not in past and c not in nh), 0)
        else: i = 0
        a, pred = cands[i], (sc[i] if sc else None); nover += i != 0
        o, won, adm = e.step(a); note = None
        if pred and arm == "C":
            y = wm.labels({"obs": o, "target": tgt}); bad = []
            for q in OUT:
                if (pred[q] if y[q] else 1 - pred[q]) < 0.3:
                    bad.append({"effect": ("the action would do nothing", "the action would work"),
                                "reveal": (f"no {tgt} would be found", f"a {tgt} would be found"),
                                "empty": ("the place would not be empty", "the place would be empty")}[q][1 - y[q]])
            if bad:
                note = f"Your last expectation was wrong: after '{a}' you expected that {' and '.join(bad)}, but observed: '{o[:150]}'. Reconsider the plan."; nsur += 1
        log.append({"t": t, "cands": cands, "chosen": i, "action": a, "obs": o, "pred": pred, "note": note})
        hist.append((a, o))
        if won: break
    return {"game": gf, "arm": arm, "seed": seed, "won": won, "steps": len(log), "llm_calls": calls, "jev_calls": jcalls,
            "surprises": nsur, "overrides": nover, "wall": time.time() - t0, "log": log}

def safe(x):
    try: return run(x)
    except Exception as ex: return {"game": x[0], "arm": x[1], "seed": x[2], "error": repr(ex)}

if __name__ == "__main__":
    import multiprocessing as mp
    seed = int(sys.argv[1]); G = alfenv.games(); arms = sys.argv[2] if len(sys.argv) > 2 else "DBCV"
    if len(sys.argv) > 3: import random; random.Random(7).shuffle(G); lo, _, hi = sys.argv[3].rpartition(':'); G = G[int(lo or 0):int(hi) if hi else None] if ':' in sys.argv[3] else G[:int(sys.argv[3])]
    jobs = [(gf, arm, seed) for gf in G for arm in arms]
    t0 = time.time()
    with mp.get_context("spawn").Pool(24) as p, open(sys.argv[4] if len(sys.argv) > 4 else f"round2_seed{seed}.jsonl", "w") as f:
        for r in p.imap_unordered(safe, jobs):
            f.write(json.dumps(r) + "\n"); f.flush()
    print("wall", time.time() - t0, "spend", J.spend(A.TAG))
