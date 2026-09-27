"""Offline check: strategy.choose == search.search (+Ev HP leaf) on logged states, using the free heuristic opponent model."""
import sys, os, glob, pickle, importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import model as Mo, search as Se, evals as E
spec = importlib.util.spec_from_file_location("strategy", os.path.join(HERE, "autoresearch", "strategy.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
rows = []
f = open(os.path.join(HERE, "data", "turns_abyssal_eval_gen8.pkl"), "rb")
while len(rows) < 3000:
    try:
        rows += pickle.load(f)
    except EOFError:
        break
same = n = 0
for r in rows[:3000:10]:
    s = pickle.loads(r["state"])
    if s.terminal() is not None or s.active(0).hp <= 0:
        continue
    acts = Mo.actions(s, 0)
    if len(acts) < 2:
        continue
    k1, st1 = Se.search(s, acts, 2, E.Ev(E.HeurOpp(), E.HPLeaf()), "exp", False)
    k2, st2 = S.choose(s, acts, False, S.make_ev(E.HeurOpp()))
    n += 1; same += (k1 == k2) and st1["q"] == st2["q"]
print(f"identical choices and Q-values: {same}/{n}")
