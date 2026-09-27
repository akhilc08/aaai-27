"""ForeSight decision logic (the ONLY file the autoresearch loop edits).
Baseline = the headline configuration: depth-2 expectimax, gen8 Laya opponent forecasts (kept fixed), HP-balance leaf,
opp mass 0.9 (cap 3 root / 2 deep), chance pruning (0.9, 4) root / (0.75, 2) deep, our actions pruned to 2 at depth by
the free damage heuristic. Copied from search.py / evals.py with identical behaviour for this configuration.

Interface used by run_arm.py (STRATEGY=autoresearch/strategy.py):
  make_ev(laya_opp) -> evaluator with set_root(state, hist) and run(policy_reqs, leaf_states)
  choose(root_state, root_actions, forced, ev) -> (index into root_actions, stats dict)
"""
import threading
import numpy as np
import model as Mo
import search as Se   # only Node and HeurEval are reused (unchanged helpers)

# ---------------- search shape ----------------
DEPTH = 2
MODE = "exp"                      # "exp" expectimax, "min" minimax over kept opponent replies
OUR_K = 2                         # our actions kept at depth >= 1 (free heuristic prior)
OPP_MASS, OPP_CAP, OPP_CAP_DEEP = 0.9, 3, 2
CHANCE_ROOT, CHANCE_DEEP = (0.9, 4), (0.75, 2)


# ---------------- leaf evaluation: state -> value in [0, 1] (our POV) ----------------
def leaf_value(s, root=None):
    t = s.terminal()
    if t is not None:
        return t
    d = sum(m.hp for m in s.sides[0]) - sum(m.hp for m in s.sides[1])
    return 0.5 + d / 12.0


# ---------------- use of the opponent forecasts ----------------
def opp_post(probs, s, acts):
    """Post-process Laya's opponent-action distribution (identity in the baseline)."""
    return probs


class StratEv:
    def __init__(self, laya_opp):
        self.opp = laya_opp
        self.tl = threading.local()
        self.calls = 0

    def set_root(self, s, hist=None):
        self.tl.root, self.tl.hist = s, hist

    def run(self, policy_reqs, leaf_states, *a, **k):
        pols = []
        if policy_reqs:
            raw = self.opp(policy_reqs, getattr(self.tl, "hist", None))
            pols = [np.asarray(opp_post(np.asarray(p, float), s, acts), float) for p, (s, _, acts, _) in zip(raw, policy_reqs)]
            pols = [p / p.sum() for p in pols]
        root = getattr(self.tl, "root", None)
        vals = [4.0 * leaf_value(s, root) for s in leaf_states]
        return pols, vals


def make_ev(laya_opp):
    return StratEv(laya_opp)


# ---------------- search (copy of search.search, Jev-only branches removed) ----------------
def prune(dist, mass, cap):
    order = sorted(range(len(dist)), key=lambda k: -dist[k])
    kept, tot = [], 0.0
    for k in order:
        if len(kept) >= cap or (tot >= mass and kept):
            break
        kept.append(k)
        tot += dist[k]
    return [(k, dist[k] / tot) for k in kept], tot


def choose(root_state, root_actions, forced, ev):
    depth = DEPTH
    root = Se.Node(root_state, depth)
    root.forced = forced
    frontier, stats = [root], {"nodes": 0, "leaves": 0, "policy_q": 0, "levels": 0, "depth": depth}
    for level in range(depth):
        live = [n for n in frontier if n.s.terminal() is None]
        reqs, owners, heur_reqs, heur_owners = [], [], [], []
        for n in live:
            if n.forced:
                continue
            reqs.append((n.s, 1, Mo.actions(n.s, 1), None)); owners.append((n, 1))
            if level > 0 and len(Mo.actions(n.s, 0)) > OUR_K:
                heur_reqs.append((n.s, 0, Mo.actions(n.s, 0), None)); heur_owners.append((n, 0))
        pols, _ = ev.run(reqs, []) if reqs else ([], [])
        stats["policy_q"] += len(reqs)
        pri = {}
        for (n, side), p in zip(owners, pols):
            pri[(id(n), side)] = p
        hp_, _ = Se.HeurEval().run(heur_reqs, [])
        for (n, side), p in zip(heur_owners, hp_):
            pri[(id(n), side)] = p
        new = []
        for n in live:
            acts0 = root_actions if level == 0 else Mo.actions(n.s, 0)
            if level > 0 and (id(n), 0) in pri:
                p0 = pri[(id(n), 0)]
                acts0 = [acts0[k] for k in sorted(range(len(acts0)), key=lambda k: -p0[k])[:OUR_K]]
            n.ours = acts0
            if n.forced:
                n.opp = [(None, 1.0)]
            else:
                acts1 = Mo.actions(n.s, 1)
                kept, tot = prune(pri[(id(n), 1)], OPP_MASS, OPP_CAP if level == 0 else OPP_CAP_DEEP)
                n.opp = [(acts1[k], p) for k, p in kept]
            cm, cc = CHANCE_ROOT if level == 0 else CHANCE_DEEP
            for ai, a in enumerate(n.ours):
                for bi, (b, _) in enumerate(n.opp):
                    kids = []
                    for p, s2 in Mo.step(n.s, a, b, cm, cc):
                        c = Se.Node(s2, n.d - 1)
                        kids.append((p, c)); new.append(c)
                    n.kids[(ai, bi)] = kids
        stats["nodes"] += len(new)
        stats["levels"] = level + 1
        frontier = new
    leaves = [n for n in frontier if n.s.terminal() is None]
    uniq, idx = {}, []
    for n in leaves:
        key = Mo.state_text(n.s)
        if key not in uniq:
            uniq[key] = len(uniq)
        idx.append(uniq[key])
    ustates = [None] * len(uniq)
    for n, k in zip(leaves, idx):
        ustates[k] = n.s
    _, vals = ev.run([], ustates) if ustates else ([], [])
    for n, k in zip(leaves, idx):
        n.val = vals[k]
    stats["leaves"] = len(ustates)

    def backup(n):
        t = n.s.terminal()
        if t is not None:
            return 4.0 * t
        if n.val is not None and not n.kids:
            return n.val
        if not n.kids:
            return 4.0 * leaf_value(n.s)
        qs = []
        for ai in range(len(n.ours)):
            per_b = [(pb, sum(p * backup(c) for p, c in n.kids[(ai, bi)])) for bi, (_, pb) in enumerate(n.opp)]
            qs.append(sum(pb * v for pb, v in per_b) if MODE == "exp" else min(v for _, v in per_b))
        n.val = max(qs)
        if n is root:
            stats["q"] = [round(x, 3) for x in qs]
        return n.val

    backup(root)
    q = stats.get("q", [0])
    return max(range(len(q)), key=lambda k: q[k]), stats
