"""Stage 1: collect transitions with an LLM agent (eps=0.2 random admissible)."""
import json, random, sys
import alfenv, agent as A
J = A.J
CAP, EPS = 25, 0.2

def run(gf):
    rng = random.Random(gf)
    e = alfenv.Env(gf); obs, adm = e.reset(); hist, rows, won = [], [], False
    for t in range(CAP):
        if rng.random() < EPS:
            a, src = rng.choice(adm), "random"
        else:
            a, _ = A.react_step(obs, hist, adm); src = "llm"
        o, won, nadm = e.step(a)
        rows.append({"game": gf, "t": t, "goal": e.goal, "target": A.target_of(gf), "init": obs,
                     "hist": list(hist[-12:]), "adm": adm, "action": a, "src": src, "obs": o, "won": won})
        hist.append((a, o)); adm = nadm
        if won: break
    for r in rows: r["ep_won"] = won
    return rows

def safe(gf):
    try: return run(gf)
    except Exception as ex: return ex

if __name__ == "__main__":
    g = alfenv.games(); random.Random(0).shuffle(g)
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(8) as p: out = p.map(safe, g[:45])
    with open("stage1_transitions.jsonl", "w") as f:
        for rs in out:
            if isinstance(rs, Exception): print("ERR", rs); continue
            for r in rs: f.write(json.dumps(r) + "\n")
    print("spend", J.spend(A.TAG))
