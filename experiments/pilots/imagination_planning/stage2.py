"""Stage 2: closed loop. A = ReAct baseline; B = LLM top-3 + Jev picks; C = B + surprise-triggered replan note; D = LLM top-3 admissible, take its first (no Jev)."""
import json, random, re, sys
import alfenv, agent as A, wm
J = A.J
CAP, NG = 30, 20
PROP = A.SYS.split("Reply exactly")[0] + ("Propose the 3 best candidate next actions, copied exactly from the admissible list, best first. "
       'Reply with JSON only: {"think": "<one short sentence>", "actions": ["a1", "a2", "a3"]}')

def propose(obs, hist, adm, note):
    t, u = J.llm([{"role": "system", "content": PROP}, {"role": "user", "content": A.render(obs, hist, adm, note)}], max_tokens=150, tag=A.TAG)
    try: c = json.loads(re.search(r"\{.*\}", t, re.S).group(0))["actions"]
    except Exception: c = [A.parse_action(t, adm)]
    c = [x.strip().lower() for x in c if isinstance(x, str)]
    ok = [x for x in dict.fromkeys(c) if x in adm]
    return (ok or c[:1] or ["look"])[:3]

def score(r):
    q = wm.questions(r["target"]); q = {"effect": q["effect"], "reveal": q["reveal"],
        "progress": {"type": "noul", "instructions": "Text household game. Will the proposed next action make real progress toward completing the task (e.g. finding, taking, transforming or placing the needed object), rather than wasting a step?"}}
    a = J.jev(wm.state(r), q, tag=A.TAG); return {k: a[k]["noul"] for k in q}

def run(args):
    gf, arm = args
    e = alfenv.Env(gf); obs, adm = e.reset(); hist, note, won, log, calls, jcalls, nsur = [], None, False, [], 0, 0, 0
    tgt = A.target_of(gf)
    for t in range(CAP):
        if arm == "A":
            a, _ = A.react_step(obs, hist, adm); calls += 1; pred = None
        elif arm == "D":
            a = propose(obs, hist, adm, None)[0]; calls += 1; pred = None
        else:
            cands = propose(obs, hist, adm, note if arm == "C" else None); calls += 1
            base = {"goal": e.goal, "target": tgt, "init": obs, "hist": hist[-12:]}
            sc = J.pmap(lambda c: score({**base, "action": c}), cands, workers=3); jcalls += len(cands)
            sc = [s if isinstance(s, dict) else {"progress": 0, "effect": .5, "reveal": .5} for s in sc]
            i = max(range(len(cands)), key=lambda i: (sc[i]["progress"], -i)); a, pred = cands[i], sc[i]
        o, won, adm = e.step(a); note = None
        if pred:
            y = wm.labels({"obs": o, "target": tgt})
            for q in ("effect", "reveal"):
                pa = pred[q] if y[q] else 1 - pred[q]
                if pa < 0.3:
                    exp = {"effect": "the action would work", "reveal": f"a {tgt} would be found"}[q]
                    exp = exp if y[q] == 0 else {"effect": "the action would do nothing", "reveal": f"no {tgt} would be found"}[q]
                    note = f"Your last expectation was wrong: after '{a}' you expected that {exp}, but observed: '{o[:150]}'. Reconsider the plan."; nsur += 1
        log.append({"t": t, "action": a, "obs": o, "pred": pred, "note": note})
        hist.append((a, o))
        if won: break
    return {"game": gf, "arm": arm, "won": won, "steps": len(log), "llm_calls": calls, "jev_calls": jcalls, "surprises": nsur, "log": log}

def safe(x):
    try: return run(x)
    except Exception as ex: return {"game": x[0], "arm": x[1], "error": repr(ex)}

if __name__ == "__main__":
    import multiprocessing as mp
    g = alfenv.games(); random.Random(0).shuffle(g); G = g[45:45 + NG]
    arms = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    jobs = [(gf, arm) for gf in G for arm in arms]
    with mp.get_context("spawn").Pool(10) as p: out = p.map(safe, jobs)
    with open("stage2_episodes.jsonl", "a") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print("spend", J.spend(A.TAG))
