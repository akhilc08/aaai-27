"""Level-by-level expectimax / minimax over the abstract model with batched evaluators.
Opponent nodes: evaluator gives P(opponent action); chance nodes: model.step probabilities.
Leaves scored in [0,4] by the evaluator. Every level issues ONE batched evaluator request."""
import json, math, re, sys, threading, time
sys.path.insert(0, "/Users/sickle/Coding/jev-context-research/pilots")
import jevlib as J
import model as Mo
import fmt as F

CTX = ("Pokemon Showdown gen9 random battle (singles). Each question describes a (possibly hypothetical) "
       "position from OUR side's point of view. HP percentages and damage numbers are estimates.")
SCALE = ["very bad (we will almost surely lose)", "bad", "even", "good", "very good (we will almost surely win)"]


_TL = threading.local()


class Node:
    __slots__ = ("s", "d", "ours", "opp", "kids", "val", "forced", "opp_mass")

    def __init__(self, s, d):
        self.s, self.d, self.ours, self.opp, self.kids, self.val, self.forced, self.opp_mass = s, d, [], [], {}, None, False, 1.0


# ---------------- evaluators ----------------
class JevEval:
    def __init__(self, tag, leaf="score", hp_leaf=False, fmt=False):
        self.tag, self.calls, self.leaf, self.hp_leaf, self.fmt = tag, 0, leaf, hp_leaf, fmt

    def _ask(self, qs):
        items = list(qs.items())
        chunks = [dict(items[i:i + 120]) for i in range(0, len(items), 120)]
        self.calls += len(chunks)
        ctx = F.CTX if self.fmt else CTX
        outs = J.pmap(lambda c: J.jev(ctx, c, tag=self.tag), chunks, workers=6)
        ans = {}
        for o in outs:
            if isinstance(o, Exception):
                raise o
            ans.update(o)
        return ans

    def run(self, policy_reqs, leaf_states, jev_leaf=False):
        """policy_reqs: list of (state, side, actions, question) ; leaf_states: list of states.
        Returns (list of prob lists, list of leaf values)."""
        if self.fmt:
            return self._run_fmt(policy_reqs, leaf_states, jev_leaf)
        qs = {}
        for j, (s, side, acts, _) in enumerate(policy_reqs):
            who = "the opponent (Them)" if side == 1 else "we (Us)"
            qs[f"p{j}"] = {"type": "choice",
                           "instructions": f"Position:\n{Mo.state_text(s)}\nWhich action will {who} choose this turn?"
                           if side == 1 else f"Position:\n{Mo.state_text(s)}\nWhich action is best for us this turn?",
                           "criteria": {f"o{k}": Mo.act_text(s, side, a) for k, a in enumerate(acts)}}
        hp_leaf = self.hp_leaf and not jev_leaf
        if hp_leaf:
            lv = [Mo.hp_value(x) for x in leaf_states]
            leaf_states = []
        for j, s in enumerate(leaf_states):
            if self.leaf == "pwin" or jev_leaf:
                qs[f"l{j}"] = {"type": "noul", "instructions": f"Position:\n{Mo.state_text(s)}\nWill we (Us) win this battle from this position?"}
                continue
            qs[f"l{j}"] = {"type": "score", "instructions": f"Position:\n{Mo.state_text(s)}\nHow good is this position for us?",
                           "criteria": SCALE}
        if not qs:
            return [], (lv if hp_leaf else [])
        ans = self._ask(qs)
        pols = []
        for j, (s, side, acts, _) in enumerate(policy_reqs):
            pr = ans[f"p{j}"].get("probabilities", {})
            v = [max(float(pr.get(f"o{k}", 0)), 1e-4) for k in range(len(acts))]
            z = sum(v)
            pols.append([x / z for x in v])
        if hp_leaf:
            return pols, lv
        if self.leaf == "pwin" or jev_leaf:
            vals = [4.0 * float(ans[f"l{j}"].get("noul", 0.5)) for j in range(len(leaf_states))]
        else:
            vals = [float(ans[f"l{j}"].get("score", 2)) for j in range(len(leaf_states))]
        return pols, vals


    def _run_fmt(self, policy_reqs, leaf_states, root_narrow=False):
        """Formatted-facts variant: policies and P(win) leaves asked on precomputed fact states.
        root_narrow: the LAST leaf state is the real root; also ask the narrow nouls on it (logged in self.last_narrow)."""
        qs = {}
        for j, (s, side, acts, _) in enumerate(policy_reqs):
            q = "Which action will the opponent (Them) choose this turn?" if side == 1 else "Which action is best for us this turn?"
            qs[f"p{j}"] = {"type": "choice", "instructions": f"Position facts:\n{F.fact_text(s)}\n{q}",
                           "criteria": {f"o{k}": F.act_label(s, side, a) for k, a in enumerate(acts)}}
        for j, s in enumerate(leaf_states):
            qs.update(F.leaf_questions(f"l{j}", s, narrow=root_narrow and j == len(leaf_states) - 1))
        if not qs:
            return [], []
        ans = self._ask(qs)
        pols = []
        for j, (s, side, acts, _) in enumerate(policy_reqs):
            pr = ans[f"p{j}"].get("probabilities", {})
            v = [max(float(pr.get(f"o{k}", 0)), 1e-4) for k in range(len(acts))]
            pols.append([x / sum(v) for x in v])
        vals = [4.0 * float(ans[f"l{j}"].get("noul", 0.5)) for j in range(len(leaf_states))]
        if root_narrow and leaf_states:
            j = len(leaf_states) - 1
            _TL.narrow = {k: round(float(ans[f"l{j}_{k}"].get("noul", 0.5)), 3) for k in F.NARROW}
        return pols, vals


class HybEval(JevEval):
    """X2 setup (Jev opponent policies on the old text, HP leaf) plus a small formatted-Jev P(win) correction:
    leaf = hp_value + 4*lam*(P(win) - 0.5), i.e. hp/4 + lam*(P - 0.5) on a [0,1] scale."""
    def __init__(self, tag, lam):
        super().__init__(tag, "score", hp_leaf=True)
        self.lam = lam

    def run(self, policy_reqs, leaf_states, jev_leaf=False):
        if policy_reqs or not leaf_states:
            return super().run(policy_reqs, leaf_states, jev_leaf)
        hp = [Mo.hp_value(x) for x in leaf_states]
        qs = {}
        for j, x in enumerate(leaf_states):
            qs.update(F.leaf_questions(f"l{j}", x))
        fmt, self.fmt = self.fmt, True
        try:
            ans = self._ask(qs)
        finally:
            self.fmt = fmt
        pw = [float(ans[f"l{j}"].get("noul", 0.5)) for j in range(len(leaf_states))]
        _TL.hp_alt = hp
        return [], [h + 4 * self.lam * (p - 0.5) for h, p in zip(hp, pw)]


class HeurEval:
    """No-API control: softmax over estimated damage for policies; HP-balance leaf value."""
    calls = 0

    def run(self, policy_reqs, leaf_states):
        pols = []
        for s, side, acts, _ in policy_reqs:
            sc = []
            for a in acts:
                if a[0] == "m":
                    mv = s.active(side).moves[a[1]]
                    sc.append(min(Mo.dmg(s.active(side), s.active(1 - side), mv), 1.0) * 4)
                else:
                    sc.append(Mo.matchup(s, side, a[1]) * 2)
            m = max(sc)
            e = [math.exp(x - m) for x in sc]
            pols.append([x / sum(e) for x in e])
        return pols, [Mo.hp_value(s) for s in leaf_states]


class LLMEval:
    """Cheap LLM verbalizes opponent-action probabilities and leaf scores (batched JSON)."""
    def __init__(self, tag, leaf="pwin"):
        self.tag, self.calls, self.leaf = tag, 0, leaf

    def _json(self, prompt, max_tokens):
        self.calls += 1
        txt, _ = J.llm([{"role": "user", "content": prompt}], max_tokens=max_tokens, tag=self.tag)
        m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
        return json.loads(m.group(0)) if m else None

    def run(self, policy_reqs, leaf_states):
        def pol(req):
            s, side, acts, _ = req
            opts = "\n".join(f"o{k}: {Mo.act_text(s, side, a)}" for k, a in enumerate(acts))
            who = "the opponent (Them) will choose" if side == 1 else "is best for us"
            p = (f"{CTX}\nPosition:\n{Mo.state_text(s)}\nOptions:\n{opts}\nGive the probability that each option {who} this turn. "
                 'Answer ONLY a JSON object like {"o0": 0.5, "o1": 0.3, ...} summing to 1.')
            try:
                d = self._json(p, 200) or {}
                v = [max(float(d.get(f"o{k}", 0)), 1e-3) for k in range(len(acts))]
            except Exception:
                v = [1.0] * len(acts)
            return [x / sum(v) for x in v]

        def leaves(chunk):
            body = "\n\n".join(f"#{k}:\n{Mo.state_text(s)}" for k, s in enumerate(chunk))
            p = (f"{CTX}\nFor each position, give the probability (0 to 1) that we (Us) win this battle from there."
                 f"\n\n{body}\n\nAnswer ONLY a JSON list of {len(chunk)} probabilities in order.")
            try:
                v = self._json(p, 20 + 8 * len(chunk))
                v = [4.0 * min(max(float(x), 0.0), 1.0) for x in v][:len(chunk)]
                return v + [2.0] * (len(chunk) - len(v))
            except Exception:
                return [2.0] * len(chunk)

        chunks = [leaf_states[i:i + 20] for i in range(0, len(leaf_states), 20)]
        res = J.pmap(lambda x: ("p", pol(x)) if isinstance(x, tuple) else ("l", leaves(x)), list(policy_reqs) + chunks, workers=8)
        pols = [r[1] for r in res[:len(policy_reqs)]]
        vals = [v for r in res[len(policy_reqs):] for v in r[1]]
        return pols, vals


# ---------------- search ----------------
def prune(dist, mass, cap):
    order = sorted(range(len(dist)), key=lambda k: -dist[k])
    kept, tot = [], 0.0
    for k in order:
        if len(kept) >= cap or (tot >= mass and kept):
            break
        kept.append(k)
        tot += dist[k]
    return [(k, dist[k] / tot) for k in kept], tot


def search(root_state, root_actions, depth, ev, mode="exp", forced=False, our_k=2, opp_mass=0.9, opp_cap=3, opp_cap_deep=2,
           chance_root=(0.9, 4), chance_deep=(0.75, 2)):
    """Returns (best root action index, stats dict). root_actions: abstract actions for us."""
    root = Node(root_state, depth)
    root.forced = forced
    frontier, stats = [root], {"nodes": 0, "leaves": 0, "policy_q": 0, "opp_mass": [], "chance_mass": [], "levels": 0}
    for level in range(depth):
        live = [n for n in frontier if n.s.terminal() is None]
        reqs, owners, heur_reqs, heur_owners = [], [], [], []
        for n in live:
            if n.forced:
                continue
            reqs.append((n.s, 1, Mo.actions(n.s, 1), None)); owners.append((n, 1))
            if level > 0 and len(Mo.actions(n.s, 0)) > our_k:
                # our own action pruning at deeper levels uses the free heuristic prior (not the evaluator)
                heur_reqs.append((n.s, 0, Mo.actions(n.s, 0), None)); heur_owners.append((n, 0))
        if level == 0 and getattr(ev, "hp_leaf", False) and root.s.terminal() is None:
            # calibration logging: Jev P(win) forecast of the real root position, asked in the same call
            pols, rv = ev.run(reqs, [root.s], jev_leaf=True)
            stats["root_value"] = round(rv[0] / 4, 4)
        else:
            pols, _ = ev.run(reqs, []) if reqs else ([], [])
        stats["calls"] = stats.get("calls", 0) + (1 if reqs else 0)
        stats["policy_q"] += len(reqs)
        pri = {}
        for (n, side), p in zip(owners, pols):
            pri[(id(n), side)] = p
        hp_, _ = HeurEval().run(heur_reqs, [])
        for (n, side), p in zip(heur_owners, hp_):
            pri[(id(n), side)] = p
        new = []
        for n in live:
            acts0 = root_actions if level == 0 else Mo.actions(n.s, 0)
            if level > 0 and (id(n), 0) in pri:
                p0 = pri[(id(n), 0)]
                acts0 = [acts0[k] for k in sorted(range(len(acts0)), key=lambda k: -p0[k])[:our_k]]
            n.ours = acts0
            if n.forced:
                n.opp = [(None, 1.0)]
            else:
                acts1 = Mo.actions(n.s, 1)
                kept, tot = prune(pri[(id(n), 1)], opp_mass, opp_cap if level == 0 else opp_cap_deep)
                n.opp = [(acts1[k], p) for k, p in kept]
                n.opp_mass = tot
                stats["opp_mass"].append(tot)
            cm, cc = chance_root if level == 0 else chance_deep
            for ai, a in enumerate(n.ours):
                for bi, (b, _) in enumerate(n.opp):
                    br = Mo.step(n.s, a, b, cm, cc)
                    stats["chance_mass"].append(Mo.step.last_mass)
                    kids = []
                    for p, s2 in br:
                        c = Node(s2, n.d - 1)
                        kids.append((p, c))
                        new.append(c)
                    n.kids[(ai, bi)] = kids
        stats["nodes"] += len(new)
        stats["levels"] = level + 1
        frontier = new
    # leaves
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
    root_extra = not isinstance(ev, HeurEval) and not getattr(ev, "hp_leaf", False) and root.s.terminal() is None
    if getattr(ev, "fmt", False) and (ustates or root_extra):
        _, vals = ev.run([], ustates + ([root.s] if root_extra else []), root_extra)
    else:
        _, vals = ev.run([], ustates + ([root.s] if root_extra else [])) if (ustates or root_extra) else ([], [])
    if root_extra:
        stats["root_value"] = round(vals[-1] / 4, 4)
        if getattr(ev, "fmt", False):
            stats["root_narrow"] = getattr(_TL, "narrow", None)
    stats["calls"] = stats.get("calls", 0) + (1 if (ustates or root_extra) else 0)
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
            return Mo.hp_value(n.s)
        best, qs = -1, []
        for ai in range(len(n.ours)):
            per_b = []
            for bi, (_, pb) in enumerate(n.opp):
                per_b.append((pb, sum(p * backup(c) for p, c in n.kids[(ai, bi)])))
            q = sum(pb * v for pb, v in per_b) if mode == "exp" else min(v for _, v in per_b)
            qs.append(q)
        n.val = max(qs)
        if n is root:
            stats["q"] = [round(x, 3) for x in qs]
        return n.val

    backup(root)
    q = stats.get("q", [0])
    alt = getattr(_TL, "hp_alt", None)
    if isinstance(ev, HybEval) and alt is not None and leaves:
        _TL.hp_alt = None
        for n, k in zip(leaves, idx):
            n.val = alt[k]
        backup(root)
        qa = stats["q"]
        stats["q"] = q
        stats["hp_only_choice_differs"] = max(range(len(qa)), key=lambda k: qa[k]) != max(range(len(q)), key=lambda k: q[k])
    stats["chance_mass"] = round(sum(stats["chance_mass"]) / max(len(stats["chance_mass"]), 1), 3)
    stats["opp_mass"] = round(sum(stats["opp_mass"]) / max(len(stats["opp_mass"]), 1), 3)
    return max(range(len(q)), key=lambda k: q[k]), stats
