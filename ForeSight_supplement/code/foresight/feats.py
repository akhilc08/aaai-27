"""Numeric features and text renderings of an abstract state (our POV) for the trained forecasters."""
import math
import numpy as np
import model as Mo
import fmt as F

STATUSES = ("brn", "par", "slp", "frz", "psn", "tox")


def _side(s, i):
    ms = s.sides[i]
    alive = [m for m in ms if m.hp > 0]
    return [len(alive), sum(m.hp for m in ms), sum(1 for m in alive if m.status),
            sum(1 for m in ms if m.unseen and m.hp > 0)]


def numeric(s):
    """Fixed-length feature vector (floats)."""
    a0, a1 = _side(s, 0), _side(s, 1)
    us, them = s.active(0), s.active(1)
    f = a0 + a1 + [a0[0] - a1[0], a0[1] - a1[1]]
    f += [us.hp, them.hp, float(bool(us.status)), float(bool(them.status)),
          sum(us.boosts.values()), sum(them.boosts.values())]
    if us.hp > 0 and them.hp > 0:
        sp0, sp1 = Mo.speed(us), Mo.speed(them)
        m0, d0 = F.best_move(us, them)
        m1, d1 = F.best_move(them, us)
        k0, k1 = F.hits_to_ko(d0, them.hp), F.hits_to_ko(d1, us.hp)
        first = 1.0 if sp0 > sp1 else -1.0 if sp1 > sp0 else 0.0
        race = 1.0 if (k0 < k1 or (k0 == k1 and first > 0)) else -1.0 if (k1 < k0 or (k0 == k1 and first < 0)) else 0.0
        safe0 = sum(1 for k, m in enumerate(s.sides[0]) if k != s.act[0] and m.hp > 0 and F.best_move(them, m)[1] < 0.35)
        safe1 = sum(1 for k, m in enumerate(s.sides[1]) if k != s.act[1] and m.hp > 0 and not m.unseen and F.best_move(us, m)[1] < 0.35)
        f += [first, min(d0, 2.0), min(d1, 2.0), float(d0 >= them.hp), float(d1 >= us.hp),
              min(k0, 6), min(k1, 6), race, safe0, safe1, 1.0]
    else:
        f += [0.0] * 10 + [0.0]
    f += [min(s.turn, 60) / 60.0]
    return f


NUM_NAMES = ["alive0", "hp0", "status0", "unseen0", "alive1", "hp1", "status1", "unseen1", "alive_d", "hp_d",
             "act_hp0", "act_hp1", "act_st0", "act_st1", "boost0", "boost1", "first", "dmg0", "dmg1", "ko0", "ko1",
             "hko0", "hko1", "race", "safe0", "safe1", "both_active", "turn"]


def fact_text(s):
    """fmt.py facts WITHOUT the last_turn event line (real positions have no path events)."""
    ev, s.ev = s.ev, []
    try:
        return F.fact_text(s)
    finally:
        s.ev = ev


def roster_text(s):
    """Compact roster lines (species, HP, status) so the text model can use Pokemon identity too."""
    def mon(m):
        if m.unseen:
            return "unseen"
        if m.hp <= 0:
            return f"{m.name} fainted"
        return f"{m.name} {round(100 * m.hp)}%" + (f" {m.status}" if m.status else "")
    us = "; ".join(("*" if k == s.act[0] else "") + mon(m) for k, m in enumerate(s.sides[0]))
    th = "; ".join(("*" if k == s.act[1] else "") + mon(m) for k, m in enumerate(s.sides[1]))
    return f"our team: {us}\ntheir team: {th}"


def compact_facts(s):
    f = F.facts(s)
    u, h = f["units_alive"], f["total_hp_percent"]
    out = [f"units alive: us {u['us']}, them {u['them']}; total HP: us {h['us']}%, them {h['them']}% ({f['material']})"]
    if "note" in f:
        out.append(f["note"])
    a = f.get("actives")
    if a:
        def side(x):
            b = x["boosts"]
            return f"{x['hp_percent']}%" + (f" {x['status']}" if x["status"] != "none" else "") + (
                " boosts " + ",".join(f"{k}{v:+d}" for k, v in b.items()) if b != "none" else "")
        o, t = a["our_best_attack"], a["their_best_attack"]
        out.append(f"our active {side(a['our_active'])}; their active {side(a['their_active'])}; moves first: {a['moves_first']}")
        out.append(f"our best attack: {o['damage_percent_of_their_hp']}% {o['effectiveness']}, hits to KO {o['hits_to_knock_out']}"
                   f"{', KOs now' if o['knocks_out_this_turn'] else ''}; their best attack: {t['damage_percent_of_our_hp']}% "
                   f"{t['effectiveness']}, hits to KO {t['hits_to_knock_out']}{', KOs now' if t['knocks_out_this_turn'] else ''}")
        sf = f["safe_switches"]
        out.append(f"race: {a['one_on_one_race']}; safe switches: us {sf['our_bench_that_take_under_35pct_from_their_active']}, "
                   f"them {sf['their_known_bench_that_take_under_35pct_from_our_active']}")
    return "\n".join(out)


def state_text(s):
    ev, s.ev = s.ev, []
    try:
        return compact_facts(s) + "\n" + roster_text(s)
    finally:
        s.ev = ev


def delta_text(root, leaf):
    """What changed along a line from root to leaf (both our POV)."""
    out = []
    for i, who in ((0, "we"), (1, "they")):
        lost = sum(max(0.0, a.hp - b.hp) for a, b in zip(root.sides[i], leaf.sides[i]))
        gain = sum(max(0.0, b.hp - a.hp) for a, b in zip(root.sides[i], leaf.sides[i]))
        faint = sum(1 for a, b in zip(root.sides[i], leaf.sides[i]) if a.hp > 0 and b.hp <= 0)
        st = sum(1 for a, b in zip(root.sides[i], leaf.sides[i]) if not a.status and b.status and b.hp > 0)
        bo = sum(leaf.active(i).boosts.values()) - (sum(root.active(i).boosts.values()) if root.act[i] == leaf.act[i] else 0)
        sw = root.act[i] != leaf.act[i]
        out.append(f"{who}: lost {round(100 * lost)}% HP, healed {round(100 * gain)}%, {faint} fainted, "
                   f"{st} newly statused, active boosts change {bo:+d}, {'active changed' if sw else 'same active'}")
    return "changes over the line: " + "; ".join(out)


def delta_numeric(root, leaf):
    f = []
    for i in (0, 1):
        lost = sum(max(0.0, a.hp - b.hp) for a, b in zip(root.sides[i], leaf.sides[i]))
        gain = sum(max(0.0, b.hp - a.hp) for a, b in zip(root.sides[i], leaf.sides[i]))
        faint = sum(1 for a, b in zip(root.sides[i], leaf.sides[i]) if a.hp > 0 and b.hp <= 0)
        st = sum(1 for a, b in zip(root.sides[i], leaf.sides[i]) if not a.status and b.status and b.hp > 0)
        f += [lost, gain, faint, st]
    return f


def arr(rows):
    return np.asarray(rows, dtype=np.float32)
