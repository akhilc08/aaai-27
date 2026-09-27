"""Probe: is Jev's weak P(win) due to missing Pokemon knowledge or to not reading the state?
(1) HP-only variants of real positions; (2) type-effectiveness quiz; (3) the same HP variants with names stripped."""
import copy, pickle, sys, json
sys.path.insert(0, "."); sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots")
import model as Mo, jevlib as J

S = [x[0] for x in pickle.load(open("data/c_states.pkl", "rb"))]

def variant(s, ours, theirs):
    v = copy.deepcopy(s)
    for side, hps in ((0, ours), (1, theirs)):
        for m, h in zip(v.sides[side], hps):
            m.hp = h
    return v

FULL = [1.0] * 6
AHEAD = ([1.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.3, 0.3])   # we 6 alive full, they 2 alive at 30%
BEHIND = ([0.0, 0.0, 0.0, 0.0, 0.3, 0.3], [1.0] * 6)
Q = lambda txt: {"type": "noul", "instructions": f"Position:\n{txt}\nWill we (Us) win this battle from this position?"}

def fix_active(v):
    # keep the active mon alive so the position is legal
    for side in (0, 1):
        a = v.active(side)
        if a.hp <= 0:
            a.hp = 0.3
    return v

rows = []
for i, s in enumerate(S):
    vs = {"orig": s, "ahead": fix_active(variant(s, *AHEAD)), "behind": fix_active(variant(s, *BEHIND))}
    qs = {k: Q(Mo.state_text(v)) for k, v in vs.items()}
    # names-stripped version: only counts and HP totals
    def plain(v):
        return (f"A two-player battle game. Each side has 6 units; a side loses when all its units reach 0 HP.\n"
                f"Us: {v.alive(0)}/6 units left, total HP {round(100*sum(m.hp for m in v.sides[0]))}% of 600%.\n"
                f"Them: {v.alive(1)}/6 units left, total HP {round(100*sum(m.hp for m in v.sides[1]))}% of 600%.")
    for k, v in vs.items():
        qs["plain_" + k] = {"type": "noul", "instructions": plain(v) + "\nWill we (Us) win this battle from this position?"}
    a = J.jev("Answer each question.", qs, tag="pk_probe")
    rows.append({k: a[k]["noul"] for k in qs})
    rows[-1]["txt_ahead"] = Mo.state_text(vs["ahead"])

TYPES = [("electric", "ground", "immune"), ("ground", "flying", "immune"), ("normal", "ghost", "immune"),
         ("fighting", "ghost", "immune"), ("psychic", "dark", "immune"), ("dragon", "fairy", "immune"),
         ("water", "fire", "super"), ("fire", "grass", "super"), ("grass", "water", "super"), ("ice", "dragon", "super"),
         ("electric", "water", "super"), ("fighting", "steel", "super"), ("fire", "water", "resisted"),
         ("grass", "fire", "resisted"), ("electric", "grass", "resisted"), ("normal", "rock", "resisted"),
         ("water", "normal", "neutral"), ("fire", "electric", "neutral"), ("psychic", "fire", "neutral"), ("ice", "water", "resisted")]
tq = {f"t{k}": {"type": "choice", "instructions": f"In Pokemon, how effective is a {a}-type attack against a pure {d}-type Pokemon?",
                "criteria": {"immune": "no effect (0x)", "resisted": "not very effective (0.5x)",
                             "neutral": "normal damage (1x)", "super": "super effective (2x)"}}
      for k, (a, d, _) in enumerate(TYPES)}
ta = J.jev("Pokemon type chart knowledge check.", tq, tag="pk_probe")
correct = sum(ta[f"t{k}"]["choice"] == ans for k, (_, _, ans) in enumerate(TYPES))
by = {}
for k, (a, d, ans) in enumerate(TYPES):
    by.setdefault(ans, []).append(ta[f"t{k}"]["choice"] == ans)

import statistics as st
out = {"n_positions": len(rows)}
for k in ["orig", "ahead", "behind", "plain_orig", "plain_ahead", "plain_behind"]:
    out[k] = round(st.mean(r[k] for r in rows), 3)
out["type_quiz"] = f"{correct}/{len(TYPES)}"
out["type_quiz_by_class"] = {c: f"{sum(v)}/{len(v)}" for c, v in by.items()}
out["type_answers"] = {f"{a}->{d}": ta[f"t{k}"]["choice"] for k, (a, d, _) in enumerate(TYPES)}
out["spend"] = round(J.spend("pk_probe"), 4)
json.dump({"summary": out, "rows": rows}, open("data/probe_knowledge.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("example 'ahead' state:\n" + rows[0]["txt_ahead"])
