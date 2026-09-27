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
MM_LAMBDA = 0.0                   # with MODE "exp": value = (1-l) * expectation + l * worst kept reply
OUR_K = 2                         # our actions kept at depth >= 1 (free heuristic prior)
LAYA_LEVELS = 99                  # opponent forecasts from Laya at levels < this; free damage-softmax heuristic below
DEEP_EXTRA = 0                    # extra plies when the root has at most DEEP_IF_ACTS actions (0 = off)
DEEP_IF_ACTS = 0
DEEP_BUDGET = 30                   # >0: search one ply deeper than DEPTH, but only if that ply needs <= this many forecasts
SWITCH_COST = 0.02                 # subtracted from the root Q of our voluntary switches, in leaf units (x4 scale)
OPP_MASS, OPP_CAP, OPP_CAP_DEEP = 0.9, 3, 2
CHANCE_ROOT, CHANCE_DEEP = (0.9, 4), (0.75, 2)


# ---------------- leaf evaluation: state -> value in [0, 1] (our POV) ----------------
ALIVE_W = 0.0                     # bonus per Pokemon still alive (in units of one full HP bar)
STATUS_W = 0.0                    # scales the per-status effective-HP penalty below
STATUS_PEN = {"slp": 0.35, "frz": 0.45, "par": 0.25, "brn": 0.15, "tox": 0.2, "psn": 0.1}
BOOST_W = 0.0                     # value per positive offensive/speed boost stage of the active Pokemon (x its HP)
MATCH_W = 0.0                     # weight of the active-vs-active matchup term (model.matchup)


def mon_val(m):
    if m.hp <= 0:
        return 0.0
    return m.hp * (1.0 - STATUS_W * STATUS_PEN.get(m.status, 0.0)) + ALIVE_W


def boost_val(m):
    if m.hp <= 0 or not m.boosts:
        return 0.0
    b = sum(max(-2, min(3, m.boosts.get(k, 0))) for k in ("atk", "spa", "spe"))
    return b * m.hp


def leaf_value(s, root=None):
    t = s.terminal()
    if t is not None:
        return t
    d = sum(mon_val(m) for m in s.sides[0]) - sum(mon_val(m) for m in s.sides[1])
    if BOOST_W:
        d += BOOST_W * (boost_val(s.active(0)) - boost_val(s.active(1)))
    if MATCH_W:
        d += MATCH_W * (Mo.matchup(s, 0, s.act[0]) - Mo.matchup(s, 1, s.act[1]))
    return min(1.0, max(0.0, 0.5 + d / (12.0 * (1.0 + ALIVE_W))))


# ---------------- use of the opponent forecasts ----------------
OPP_TEMP = 0.3                    # extra temperature on Laya's opponent distribution (1 = as calibrated)
SW_FLOOR = 0.0                    # minimum total probability mass on opponent switches (spread evenly), 0 = off


def opp_post(probs, s, acts):
    """Post-process Laya's opponent-action distribution (identity in the baseline)."""
    p = probs
    if OPP_TEMP != 1.0:
        p = np.power(np.clip(p, 1e-9, 1), 1.0 / OPP_TEMP); p = p / p.sum()
    if SW_FLOOR:
        sw = np.array([a[0] == "s" for a in acts])
        if sw.any() and p[sw].sum() < SW_FLOOR:
            p = p.copy(); p[~sw] *= (1 - SW_FLOOR) / max(p[~sw].sum(), 1e-9); p[sw] = SW_FLOOR / sw.sum()
    return p


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
    depth = DEPTH + (DEEP_EXTRA if DEEP_EXTRA and len(root_actions) <= DEEP_IF_ACTS else 0) + (1 if DEEP_BUDGET else 0)
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
        if DEEP_BUDGET and level == depth - 1 and len(reqs) > DEEP_BUDGET:
            break                         # too many forecasts for the extra ply: frontier stays as depth-(depth-1) leaves
        if level < LAYA_LEVELS:
            pols, _ = ev.run(reqs, []) if reqs else ([], [])
        else:
            pols, _ = Se.HeurEval().run(reqs, []) if reqs else ([], [])
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
            if MODE == "exp":
                e = sum(pb * v for pb, v in per_b)
                qs.append(e if not MM_LAMBDA else (1 - MM_LAMBDA) * e + MM_LAMBDA * min(v for _, v in per_b))
            else:
                qs.append(min(v for _, v in per_b))
        if n is root:
            if SWITCH_COST and not n.forced:
                qs = [q - 4.0 * SWITCH_COST if a[0] == "s" else q for q, a in zip(qs, n.ours)]
            stats["q"] = [round(x, 3) for x in qs]
        n.val = max(qs)
        return n.val

    backup(root)
    q = stats.get("q", [0])
    return max(range(len(q)), key=lambda k: q[k]), stats
