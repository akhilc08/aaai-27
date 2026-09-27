"""Formatter: renders an abstract state as compact precomputed facts (no API).
Everything Jev would otherwise have to compose (damage, KO race, speed, effectiveness, safe switches,
material balance) is computed here by the simulator's damage model."""
import math
import model as Mo


def eff_label(mv, dfn):
    if mv.type is None or not dfn.types:
        return "neutral"
    x = Mo.type_mult(mv.type, dfn)
    return "immune" if x == 0 else "resisted" if x < 1 else "super-effective" if x > 1 else "neutral"


def best_move(att, dfn):
    mvs = [m for m in att.moves if m.cat != "X" and m.bp]
    if not mvs:
        return None, 0.0
    m = max(mvs, key=lambda m: Mo.dmg(att, dfn, m) * m.acc)
    return m, Mo.dmg(att, dfn, m) * m.acc


def hits_to_ko(d, hp):
    return math.inf if d <= 0 else max(1, math.ceil(hp / d - 1e-9))


def facts(s):
    us, them = s.active(0), s.active(1)
    a0, a1 = s.alive(0), s.alive(1)
    hp0 = round(100 * sum(m.hp for m in s.sides[0]))
    hp1 = round(100 * sum(m.hp for m in s.sides[1]))
    unseen = sum(m.unseen and m.hp > 0 for m in s.sides[1])
    f = {"units_alive": {"us": a0, "them": a1, "difference": a0 - a1},
         "total_hp_percent": {"us": hp0, "them": hp1, "difference": hp0 - hp1},
         "material": "we are ahead" if (hp0 - hp1) > 40 else "we are behind" if (hp1 - hp0) > 40 else "roughly even"}
    if unseen:
        f["note"] = f"{unseen} of their units not yet seen (counted at full HP)"
    if us.hp > 0 and them.hp > 0:
        sp0, sp1 = Mo.speed(us), Mo.speed(them)
        m0, d0 = best_move(us, them)
        m1, d1 = best_move(them, us)
        k0, k1 = hits_to_ko(d0, them.hp), hits_to_ko(d1, us.hp)
        first = "us" if sp0 > sp1 else "them" if sp1 > sp0 else "tie"
        if k0 < k1 or (k0 == k1 and first == "us"):
            race = "we knock out their active first"
        elif k1 < k0 or (k0 == k1 and first == "them"):
            race = "they knock out our active first"
        else:
            race = "even (speed tie)"
        f["actives"] = {
            "our_active": {"hp_percent": round(100 * us.hp), "status": us.status or "none",
                           "boosts": {k: v for k, v in us.boosts.items() if v} or "none"},
            "their_active": {"hp_percent": round(100 * them.hp), "status": them.status or "none",
                             "boosts": {k: v for k, v in them.boosts.items() if v} or "none"},
            "moves_first": first,
            "our_best_attack": {"damage_percent_of_their_hp": round(100 * d0),
                                "effectiveness": eff_label(m0, them) if m0 else "no attack",
                                "knocks_out_this_turn": d0 >= them.hp, "hits_to_knock_out": None if k0 == math.inf else k0},
            "their_best_attack": {"damage_percent_of_our_hp": round(100 * d1),
                                  "effectiveness": eff_label(m1, us) if m1 else "no attack",
                                  "knocks_out_this_turn": d1 >= us.hp, "hits_to_knock_out": None if k1 == math.inf else k1},
            "one_on_one_race": race,
        }
        # safe switches: bench mons that take < 35% from the opposing active's best attack
        safe0 = sum(1 for k, m in enumerate(s.sides[0]) if k != s.act[0] and m.hp > 0 and best_move(them, m)[1] < 0.35)
        safe1 = sum(1 for k, m in enumerate(s.sides[1]) if k != s.act[1] and m.hp > 0 and not m.unseen and best_move(us, m)[1] < 0.35)
        f["safe_switches"] = {"our_bench_that_take_under_35pct_from_their_active": safe0,
                              "their_known_bench_that_take_under_35pct_from_our_active": safe1}
    if s.ev:
        f["last_turn"] = "; ".join(s.ev[-4:])
    return f


def fact_text(s):
    f = facts(s)
    lines = []
    for k, v in f.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    lines.append(f"  {k2}: " + ", ".join(f"{a}={b}" for a, b in v2.items()))
                else:
                    lines.append(f"  {k2}: {v2}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


CTX = ("A turn-based two-player battle game (Pokemon singles). Each side has 6 units; a side loses when all its units "
       "reach 0 HP. Facts below were precomputed by a damage calculator, from OUR point of view.")
PWIN = "Will we (Us) win this battle from this position?"
NARROW = {
    "ahead": "Are we ahead on total remaining HP and units?",
    "race": "Will our active unit knock out their active unit before it knocks out ours?",
    "safe": "Do we have a safe switch into their active unit?",
    "threat": "Is our active unit in danger of being knocked out this turn?",
}


def leaf_questions(prefix, s, narrow=False):
    t = fact_text(s)
    q = {f"{prefix}": {"type": "noul", "instructions": f"Position facts:\n{t}\n{PWIN}"}}
    if narrow:
        for k, v in NARROW.items():
            q[f"{prefix}_{k}"] = {"type": "noul", "instructions": f"Position facts:\n{t}\n{v}"}
    return q


def act_label(s, side, a):
    """Action description with precomputed consequences."""
    me, op = s.active(side), s.active(1 - side)
    if a[0] == "s":
        m = s.sides[side][a[1]]
        taken = best_move(op, m)[1]
        return f"switch in a unit at {round(100 * m.hp)}% HP that takes ~{round(100 * taken)}% from the opposing active's best attack"
    mv = me.moves[a[1]]
    if mv.cat == "X" or not mv.bp:
        extra = []
        if mv.heal: extra.append(f"heals {round(100 * mv.heal)}%")
        if mv.selfb: extra.append("raises own " + ",".join(f"{k}{v:+d}" for k, v in mv.selfb.items()))
        if mv.status: extra.append(f"inflicts {mv.status}")
        if mv.tb: extra.append("lowers target " + ",".join(f"{k}{v:+d}" for k, v in mv.tb.items()))
        return f"non-damaging move {mv.id} ({'; '.join(extra) or 'utility'})"
    d = Mo.dmg(me, op, mv)
    return (f"attack {mv.id}: ~{round(100 * min(d, 1.5))}% of target HP, {eff_label(mv, op)}, accuracy {round(100 * mv.acc)}%"
            + (", knocks out" if d >= op.hp else ""))
