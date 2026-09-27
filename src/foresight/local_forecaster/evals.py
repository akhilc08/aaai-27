"""Search evaluators: opponent-action models x leaf-value models. ev.run(policy_reqs, leaf_states) -> (pols, vals[0..4])."""
import math, os, pickle, threading
import numpy as np
import model as Mo
import fmt as F
import feats as X
import search as Se
from build_ds import cand_feats

HERE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()   # one GPU model shared by concurrent battles
_CACHE = {}


def base_models(fmt_):
    k = ("base", fmt_)
    if k not in _CACHE:
        _CACHE[k] = pickle.load(open(os.path.join(HERE, "ckpt", f"base_{fmt_}.pkl"), "rb"))
    return _CACHE[k]


def laya_agent(path):
    k = ("laya", path)
    if k not in _CACHE:
        import laya_util as U
        _CACHE[k] = U.load(os.path.join(HERE, "ckpt", path))
    return _CACHE[k]


# ---------------- leaf models: list[state] (+root) -> P(win) array ----------------
class HPLeaf:
    def __call__(self, states, root=None):
        return np.array([Mo.hp_value(s) / 4 for s in states])


class SkLeaf:
    def __init__(self, fmt_, name="logreg", delta=False):
        self.m = base_models(fmt_)["delta" if delta else "pwin"][name]
        self.delta = delta

    def __call__(self, states, root=None):
        if self.delta:
            Xs = [X.numeric(root) + X.delta_numeric(root, s) + X.numeric(s) for s in states]
        else:
            Xs = [X.numeric(s) for s in states]
        return self.m.predict_proba(np.array(Xs, float))[:, 1]


class LayaLeaf:
    def __init__(self, path, delta=False, sharpen=1.0):
        self.ag = laya_agent(path)
        self.delta, self.sharpen = delta, sharpen
        self.T = self.ag.ft_temp.get("delta" if delta else "pwin", 1.0)

    def __call__(self, states, root=None):
        import laya_util as U
        if self.delta:
            its = [U.encode(self.ag, X.state_text(root) + "\n" + X.delta_text(root, s), "noul", U.Q_DELTA, max_len=448) for s in states]
        else:
            its = [U.encode(self.ag, X.state_text(s), "noul", U.Q_PWIN) for s in states]
        with _LOCK:
            lg = U.predict_logits(self.ag, its, bs=32)
        z = np.array([(l[1] - l[0]) / self.T for l in lg]) * self.sharpen
        return 1 / (1 + np.exp(-z))


class Sharpen:
    """Deliberately overconfident version of a leaf: logit(p) * k."""
    def __init__(self, leaf, k):
        self.leaf, self.k = leaf, k

    def __call__(self, states, root=None):
        p = np.clip(self.leaf(states, root), 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-np.log(p / (1 - p)) * self.k))


class Hybrid:
    def __init__(self, leaf, lam):
        self.leaf, self.lam = leaf, lam

    def __call__(self, states, root=None):
        return np.array([Mo.hp_value(s) / 4 for s in states]) + self.lam * (self.leaf(states, root) - 0.5)


# ---------------- opponent models: list of (state, actions) -> list of prob arrays ----------------
class HeurOpp:
    def __call__(self, reqs, hist=None):
        return Se.HeurEval().run(reqs, [])[0]


class SkOpp:
    def __init__(self, fmt_, name="gbm_hist"):
        self.m = base_models(fmt_)["opp"][name]
        self.hist = name.endswith("_hist")

    def __call__(self, reqs, hist=None):
        if not reqs:
            return []
        heur = Se.HeurEval().run(reqs, [])[0]
        rows, grp = [], []
        hv = list(hist[0]) if (self.hist and hist) else ([0.0] * 10 if self.hist else [])
        for j, ((s, side, acts, _), h) in enumerate(zip(reqs, heur)):
            num = X.numeric(s)
            for cf in cand_feats(s, acts, h):
                rows.append(list(cf) + num + hv); grp.append(j)
        p = self.m.predict_proba(np.array(rows, float))[:, 1]
        out = [[] for _ in reqs]
        for pi, g in zip(p, grp):
            out[g].append(pi)
        return [np.array(o) / max(sum(o), 1e-9) for o in out]


class UniformOpp:
    def __call__(self, reqs, hist=None):
        return [np.ones(len(acts)) / len(acts) for _, _, acts, _ in reqs]


class LayaOpp:
    def __init__(self, path, hist=True, sharpen=1.0):
        self.ag, self.hist = laya_agent(path), hist
        self.T = self.ag.ft_temp.get("opp", 1.0) / sharpen

    def __call__(self, reqs, hist=None):
        import laya_util as U
        if not reqs:
            return []
        its = []
        for s, side, acts, _ in reqs:
            txt = X.state_text(s) + ("\n" + hist[1] if (self.hist and hist) else "")
            crit = {f"o{k}": F.act_label(s, 1, a) for k, a in enumerate(acts)}
            its.append(U.encode(self.ag, txt, "choice", U.Q_OPP, crit, max_len=448))
        with _LOCK:
            lg = U.predict_logits(self.ag, its, bs=16)
        out = []
        for l in lg:
            z = l / self.T; z = np.exp(z - z.max()); out.append(z / z.sum())
        return out


class Ev:
    """Combines an opponent model and a leaf model. Per-decision context (root, history) is thread-local."""
    def __init__(self, opp, leaf):
        self.opp, self.leaf, self.calls = opp, leaf, 0
        self.tl = threading.local()

    def set_root(self, s, hist=None):
        self.tl.root, self.tl.hist = s, hist

    def run(self, policy_reqs, leaf_states, *a, **k):
        pols = self.opp(policy_reqs, getattr(self.tl, "hist", None)) if policy_reqs else []
        vals = []
        if leaf_states:
            vals = list(4.0 * np.asarray(self.leaf(leaf_states, getattr(self.tl, "root", None)), float))
        return pols, vals


# ---------------- Foul Play / poke-engine position evaluator (ported) ----------------
# Port of pmariglia/poke-engine src/genx/evaluate.rs (MIT License, Copyright (c) pmariglia), the evaluator used by
# Foul Play. Only the evaluation constants/logic are ported (Foul Play itself is GPL-3.0 and is not used).
# Terms kept (tracked by model.py): alive bonus, HP fraction, status penalties (burn uses the physical-move rule and
# the special-attacker halving), active-Pokemon boost terms with the boost-multiplier table, per-Pokemon floor at 0
# before adding the alive bonus.
# Terms dropped (model.py does not track them): items (+10), abilities (Poison Heal/Guts/Marvel Scale/Quick Feet/
# Toxic Boost/Magic Guard), hazards (stealth rock, spikes, toxic spikes, sticky web), volatiles (substitute,
# confusion, leech seed), side conditions (reflect, light screen, aurora veil, safeguard, tailwind, healing wish),
# terastallization. Unseen opposing Pokemon count as alive at full HP (same convention as the HP leaf).
FP_BOOST_MULT = {6: 3.3, 5: 3.15, 4: 3.0, 3: 2.5, 2: 2.0, 1: 1.0, 0: 0.0, -1: -1.0, -2: -2.0, -3: -2.5, -4: -3.0, -5: -3.15, -6: -3.3}
FP_BOOST_W = {"atk": 30.0, "def": 15.0, "spa": 30.0, "spd": 15.0, "spe": 30.0}


def _fp_burn(m):
    mult = sum(1.0 for mv in m.moves if mv.cat == "P")
    if m.st.get("spa", 0) > m.st.get("atk", 0):
        mult /= 2.0
    return mult * -25.0


def _fp_pokemon(m):
    sc = 100.0 * m.hp
    sc += {"brn": _fp_burn(m) if m.status == "brn" else 0.0, "frz": -40.0, "slp": -25.0, "par": -25.0,
           "tox": -30.0, "psn": -10.0}.get(m.status, 0.0)
    return max(sc, 0.0) + 30.0


def fp_score(s):
    tot = 0.0
    for i, sign in ((0, 1.0), (1, -1.0)):
        for k, m in enumerate(s.sides[i]):
            if m.hp > 0:
                v = _fp_pokemon(m)
                if k == s.act[i]:
                    v += sum(FP_BOOST_MULT[max(-6, min(6, m.boosts.get(b, 0)))] * w for b, w in FP_BOOST_W.items())
                tot += sign * v
    return tot


class FPLeaf:
    """Foul Play evaluation mapped linearly onto the search's [0,4] scale (a full healthy Pokemon = 130 = 1/3 unit, as in the HP leaf)."""
    def __call__(self, states, root=None):
        return np.array([min(max(0.5 + fp_score(s) / 1560.0, 0.0025), 0.9975) for s in states])


class AliveLeaf:
    """HP balance plus a bonus of `bonus` x (one full-HP Pokemon) per Pokemon still alive; mapped to [0,1]."""
    def __init__(self, bonus=0.3):
        self.b = bonus

    def __call__(self, states, root=None):
        out = []
        for s in states:
            d = sum(m.hp for m in s.sides[0]) - sum(m.hp for m in s.sides[1]) + self.b * (s.alive(0) - s.alive(1))
            out.append(0.5 + d / (2 * 6 * (1 + self.b)))
        return np.array(out)
