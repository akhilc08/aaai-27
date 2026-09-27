"""Abstract multi-turn model of a gen9randombattle singles position, built from a poke-env Battle.
Supports stochastic one-turn resolution (accuracy, speed ties, KO-roll buckets, secondary status,
full paralysis) for expectimax search. Approximations: estimated randbats stats (85 EV/31 IV),
no abilities/items/weather/hazard damage/crits/tera/volatile effects; opponent unseen mons are
generic placeholders; unrevealed opponent moves are filled with 90 BP STAB attacks."""
from poke_env.battle import MoveCategory, PokemonType

BOOST = {i: (2 + i) / 2 if i >= 0 else 2 / (2 - i) for i in range(-6, 7)}
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]


class M:
    __slots__ = ("id", "type", "cat", "bp", "pri", "acc", "heal", "selfb", "tb", "status", "sec", "hits")

    def __init__(self, id, type, cat, bp, pri=0, acc=1.0, heal=0.0, selfb=None, tb=None, status="", sec=None, hits=1.0):
        self.id, self.type, self.cat, self.bp, self.pri, self.acc = id, type, cat, bp, pri, acc
        self.heal, self.selfb, self.tb, self.status, self.sec, self.hits = heal, selfb or {}, tb or {}, status, sec or [], hits


def conv_move(m):
    cat = {MoveCategory.PHYSICAL: "P", MoveCategory.SPECIAL: "S"}.get(m.category, "X")
    acc = m.accuracy if isinstance(m.accuracy, (int, float)) and 0 < m.accuracy <= 1 else 1.0
    sec = []
    for s in (m.secondary or []):
        if s.get("status") and s.get("chance"):
            sec.append((s["chance"] / 100, s["status"]))
    selfb = dict(m.self_boost or {})
    tb = {}
    if m.boosts:
        if m.target == "self":
            selfb.update(m.boosts)
        else:
            tb = dict(m.boosts)
    st = m.status.name.lower() if m.status else ""
    try:
        hits = m.expected_hits or 1
    except Exception:
        hits = 1
    return M(m.id, m.type, cat, m.base_power or 0, m.priority, acc, m.heal or 0.0, selfb, tb, st, sec, hits)


def est_stats(base, L):
    s = {k: (2 * base[k] + 31 + 21) * L // 100 + 5 for k in STATS}
    s["hp"] = (2 * base["hp"] + 31 + 21) * L // 100 + L + 10
    return s


class Mon:
    __slots__ = ("name", "types", "L", "st", "hp", "status", "boosts", "moves", "unseen")

    def copy(self):
        n = Mon.__new__(Mon)
        n.name, n.types, n.L, n.st, n.hp, n.status = self.name, self.types, self.L, self.st, self.hp, self.status
        n.boosts, n.moves, n.unseen = dict(self.boosts), self.moves, self.unseen
        return n


def conv_mon(p, ours):
    m = Mon()
    m.name, m.L, m.unseen = p.species, p.level or 80, False
    m.types = [t for t in p.types if t is not None]
    if ours and p.stats and all(p.stats.get(k) for k in STATS[1:]):
        m.st = dict(p.stats)
        m.st["hp"] = p.max_hp or est_stats(p.base_stats, m.L)["hp"]
    else:
        m.st = est_stats(p.base_stats, m.L)
    m.hp = 0.0 if p.fainted else p.current_hp_fraction
    m.status = p.status.name.lower() if p.status and not p.fainted else ""
    m.boosts = {k: v for k, v in p.boosts.items() if v and k in STATS}
    moves = [conv_move(x) for x in p.moves.values()]
    if not ours and len(moves) < 4:
        cat = "P" if p.base_stats["atk"] >= p.base_stats["spa"] else "S"
        for t in m.types:
            moves.append(M(f"unrevealed-{t.name.lower()}-stab", t, cat, 90))
    m.moves = moves
    return m


def placeholder():
    m = Mon()
    m.name, m.types, m.L, m.unseen = "unseen-pokemon", [], 84, True
    m.st = est_stats({k: 90 for k in STATS}, 84)
    m.hp, m.status, m.boosts = 1.0, "", {}
    m.moves = [M("unknown-attack", None, "P", 85)]
    return m


class S:
    """State: sides[0]=us, sides[1]=them; act[i] = active index; ev = short event log of this path."""
    __slots__ = ("sides", "act", "ev", "turn")

    def copy(self):
        n = S.__new__(S)
        n.sides = [[m.copy() for m in side] for side in self.sides]
        n.act, n.ev, n.turn = list(self.act), list(self.ev), self.turn
        return n

    def active(self, i):
        return self.sides[i][self.act[i]]

    def alive(self, i):
        return sum(m.hp > 0 for m in self.sides[i])

    def terminal(self):
        a, b = self.alive(0), self.alive(1)
        if b == 0:
            return 1.0
        if a == 0:
            return 0.0
        return None


def from_battle(b):
    s = S.__new__(S)
    us = list(b.team.values())
    them = list(b.opponent_team.values())
    s.sides = [[conv_mon(p, True) for p in us], [conv_mon(p, False) for p in them] + [placeholder() for _ in range(6 - len(them))]]
    s.act = [next((i for i, p in enumerate(us) if p.active), 0), next((i for i, p in enumerate(them) if p.active), 0)]
    s.ev, s.turn = [], b.turn
    return s


from poke_env.data import GenData
_CHART = GenData.from_gen(9).type_chart


def type_mult(t, defender):
    if t is None or not defender.types:
        return 1.0
    ts = defender.types[:2]
    return t.damage_multiplier(ts[0], ts[1] if len(ts) > 1 else None, type_chart=_CHART)


def speed(m):
    v = m.st["spe"] * BOOST[m.boosts.get("spe", 0)]
    return v * 0.5 if m.status == "par" else v


def dmg(att, dfn, mv):
    """Expected (mid-roll, on-hit) damage as fraction of defender max HP."""
    if mv.cat == "X" or not mv.bp:
        return 0.0
    a, d = ("atk", "def") if mv.cat == "P" else ("spa", "spd")
    A = att.st[a] * BOOST[att.boosts.get(a, 0)]
    D = dfn.st[d] * BOOST[dfn.boosts.get(d, 0)]
    base = ((2 * att.L / 5 + 2) * mv.bp * A / max(D, 1)) / 50 + 2
    stab = 1.5 if mv.type is not None and mv.type in att.types else 1.0
    burn = 0.5 if mv.cat == "P" and att.status == "brn" else 1.0
    return base * stab * type_mult(mv.type, dfn) * burn * 0.925 * mv.hits / dfn.st["hp"]


def actions(s, i):
    """Legal abstract actions for side i: ('m', move_idx) or ('s', mon_idx). Placeholders are not switchable."""
    out = []
    act = s.active(i)
    if act.hp > 0:
        out += [("m", k) for k in range(len(act.moves))]
    out += [("s", k) for k, m in enumerate(s.sides[i]) if k != s.act[i] and m.hp > 0 and not m.unseen]
    return out


def matchup(s, i, k):
    me, op = s.sides[i][k], s.active(1 - i)
    best = max([dmg(me, op, mv) for mv in me.moves] or [0])
    worst = max([dmg(op, me, mv) for mv in op.moves] or [0])
    return min(best, op.hp) - min(worst, me.hp) + 0.3 * me.hp


def replace_fainted(s):
    for i in (0, 1):
        if s.active(i).hp <= 0:
            cands = [k for k, m in enumerate(s.sides[i]) if m.hp > 0]
            if cands:
                s.act[i] = max(cands, key=lambda k: matchup(s, i, k))
                s.ev.append(f"{'we' if i == 0 else 'they'} send in {s.active(i).name}")


def _apply_move(branches, i, mv):
    """Apply side i's move to every branch; returns expanded branch list [(p, state)]."""
    out = []
    for p, s in branches:
        att, dfn = s.active(i), s.active(1 - i)
        who = "we" if i == 0 else "they"
        if att.hp <= 0:
            out.append((p, s))
            continue
        pre = []
        if att.status == "slp" or att.status == "frz":
            s.ev.append(f"{who}: {att.name} can't move ({att.status})")
            out.append((p, s))
            continue
        if att.status == "par":
            s2 = s.copy()
            s2.ev.append(f"{who}: {att.name} fully paralyzed")
            pre.append((p * 0.25, s2))
            p *= 0.75
        cur = [(p, s)]
        if mv.acc < 1:
            s2 = s.copy()
            s2.ev.append(f"{who}: {mv.id} missed")
            pre.append((p * (1 - mv.acc), s2))
            cur = [(p * mv.acc, s)]
        res = []
        for q, t in cur:
            att, dfn = t.active(i), t.active(1 - i)
            if mv.cat != "X" and mv.bp:
                d = dmg(att, dfn, mv)
                lo, hi = d * 0.85 / 0.925, d / 0.925
                if lo < dfn.hp <= hi:  # KO depends on the damage roll
                    pko = (hi - dfn.hp) / (hi - lo)
                    t2 = t.copy()
                    t2.active(1 - i).hp = 0.0
                    t2.ev.append(f"{who}: {mv.id} KOs {dfn.name} (high roll)")
                    dfn.hp = max(0.01, dfn.hp - lo)
                    t.ev.append(f"{who}: {mv.id} leaves {dfn.name} at {round(100 * dfn.hp)}% (low roll)")
                    res += [(q * pko, t2), (q * (1 - pko), t)]
                else:
                    dfn.hp = max(0.0, dfn.hp - d)
                    t.ev.append(f"{who}: {mv.id} ~{round(100 * d)}% to {dfn.name}" + (" (KO)" if dfn.hp <= 0 else ""))
                    res.append((q, t))
                # secondary status chance (only if target survives and has no status)
                nres = []
                for q2, t3 in res:
                    tgt = t3.active(1 - i)
                    sec = [x for x in mv.sec if tgt.hp > 0 and not tgt.status]
                    if sec:
                        ch, stt = sec[0]
                        t4 = t3.copy()
                        t4.active(1 - i).status = stt
                        t4.ev.append(f"{who}: {mv.id} inflicts {stt}")
                        nres += [(q2 * ch, t4), (q2 * (1 - ch), t3)]
                    else:
                        nres.append((q2, t3))
                res = nres
            else:
                bits = []
                if mv.heal:
                    att.hp = min(1.0, att.hp + mv.heal)
                    bits.append(f"heals {round(100 * mv.heal)}%")
                for k, v in mv.selfb.items():
                    att.boosts[k] = max(-6, min(6, att.boosts.get(k, 0) + v))
                if mv.selfb:
                    bits.append("boosts " + ",".join(f"{k}{v:+d}" for k, v in mv.selfb.items()))
                for k, v in mv.tb.items():
                    dfn.boosts[k] = max(-6, min(6, dfn.boosts.get(k, 0) + v))
                if mv.tb:
                    bits.append("target " + ",".join(f"{k}{v:+d}" for k, v in mv.tb.items()))
                if mv.status and not dfn.status and dfn.hp > 0:
                    dfn.status = mv.status
                    bits.append(f"inflicts {mv.status}")
                t.ev.append(f"{who}: {mv.id} ({', '.join(bits) or 'no modeled effect'})")
                res.append((q, t))
        out += pre + res
    return out


def step(s, a0, a1, mass=0.9, cap=4):
    """Resolve one turn. a0/a1 abstract actions (a1 may be None = no action). Returns [(p, state)] pruned
    to cover `mass` of probability (max `cap` branches), renormalized."""
    s = s.copy()
    s.ev = []
    s.turn += 1
    acts = {0: a0, 1: a1}
    for i in (0, 1):
        a = acts[i]
        if a and a[0] == "s":
            s.act[i] = a[1]
            s.ev.append(f"{'we' if i == 0 else 'they'} switch to {s.active(i).name}")
    movers = [i for i in (0, 1) if acts[i] and acts[i][0] == "m"]
    branches = [(1.0, s)]
    if movers:
        mvs = {i: s.active(i).moves[acts[i][1]] for i in movers}
        if len(movers) == 2:
            k0 = (mvs[0].pri, speed(s.active(0)))
            k1 = (mvs[1].pri, speed(s.active(1)))
            if k0 == k1:
                orders = [(0.5, [0, 1]), (0.5, [1, 0])]
            else:
                orders = [(1.0, [0, 1] if k0 > k1 else [1, 0])]
        else:
            orders = [(1.0, movers)]
        branches = []
        for po, order in orders:
            br = [(po, s if len(orders) == 1 else s.copy())]
            for i in order:
                br = _apply_move(br, i, mvs[i])
            branches += br
    # end of turn residual damage, faint replacement
    for p, t in branches:
        for i in (0, 1):
            m = t.active(i)
            if m.hp > 0 and m.status in ("brn", "psn", "tox"):
                m.hp = max(0.0, m.hp - (1 / 16 if m.status == "brn" else 1 / 8))
        replace_fainted(t)
    branches.sort(key=lambda x: -x[0])
    kept, tot = [], 0.0
    for p, t in branches:
        if len(kept) >= cap or (tot >= mass and kept):
            break
        kept.append((p, t))
        tot += p
    step.last_mass = tot
    return [(p / tot, t) for p, t in kept]


def act_text(s, i, a):
    me, op = s.active(i), s.active(1 - i)
    if a[0] == "s":
        m = s.sides[i][a[1]]
        return f"switch to {m.name} ({'/'.join(t.name.lower() for t in m.types)}, {round(100 * m.hp)}% HP)"
    mv = me.moves[a[1]]
    if mv.cat == "X" or not mv.bp:
        return f"use {mv.id} (status move)"
    d = dmg(me, op, mv)
    tn = mv.type.name.lower() if mv.type else "typeless"
    return f"use {mv.id} ({tn}, {mv.bp} BP, est. {round(100 * min(d, 1.5))}% of {op.name}'s HP)"


def mon_text(m, active=False):
    if m.unseen:
        return "unseen pokemon"
    s = f"{m.name} ({'/'.join(t.name.lower() for t in m.types)}) {round(100 * m.hp)}%"
    if m.hp <= 0:
        return f"{m.name} fainted"
    if m.status:
        s += f" {m.status}"
    if m.boosts:
        s += " " + ",".join(f"{k}{v:+d}" for k, v in m.boosts.items() if v)
    if active:
        s += " [" + ", ".join(mv.id for mv in m.moves) + "]"
    return s


def state_text(s):
    a0, a1 = s.active(0), s.active(1)
    unseen = sum(m.unseen for m in s.sides[1])
    faster = "we are faster" if speed(a0) > speed(a1) else ("speed tie" if speed(a0) == speed(a1) else "they are faster")
    lines = [
        f"Us ({s.alive(0)}/6 left): ACTIVE {mon_text(a0, True)}; bench: " +
        ", ".join(mon_text(m) for k, m in enumerate(s.sides[0]) if k != s.act[0]),
        f"Them ({s.alive(1)}/6 left, {unseen} unseen): ACTIVE {mon_text(a1, True)}; bench: " +
        (", ".join(mon_text(m) for k, m in enumerate(s.sides[1]) if k != s.act[1] and not m.unseen) or "none revealed"),
        f"Active speed: {faster}.",
        f"Total team HP (unseen opponents counted as full): us {round(100 * sum(m.hp for m in s.sides[0]))}%, "
        f"them {round(100 * sum(m.hp for m in s.sides[1]))}% (out of 600% each).",
    ]
    if s.ev:
        lines.append("Recent events: " + "; ".join(s.ev[-6:]))
    return "\n".join(lines)


def hp_value(s):
    """Hand-crafted evaluator in [0,4]: total HP balance."""
    t = s.terminal()
    if t is not None:
        return 4 * t
    d = sum(m.hp for m in s.sides[0]) - sum(m.hp for m in s.sides[1])
    return 2 + 2 * d / 6
