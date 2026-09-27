"""Live opponent-history features, parsed from the battle's message log (same features as build_ds)."""
from poke_env.data import to_id_str
import model as Mo
import fmt as F
from build_ds import hist_feats, hist_text


def actions_by_turn(battle):
    """{turn: {role: ('m', move_id) | ('s', species) | None}} for chosen (non-replacement) actions."""
    out, turn, fainted = {}, 0, set()
    for ev in battle._replay_data:
        if len(ev) < 2:
            continue
        k = ev[1]
        if k == "turn":
            turn = int(ev[2]); fainted = set()
            continue
        if turn == 0 or len(ev) < 3 or not ev[2][:2] in ("p1", "p2"):
            continue
        role = ev[2][:2]
        d = out.setdefault(turn, {})
        if k == "faint":
            fainted.add(role)
        if role in d or role in fainted:
            continue
        if k == "move":
            if any("[from]" in x and "lockedmove" not in x for x in ev[4:]):
                continue
            d[role] = ("m", to_id_str(ev[3]))
        elif k == "switch":
            d[role] = ("s", to_id_str(ev[3].split(",")[0]))
        elif k == "cant":
            d[role] = None
    return out


def live_hist(battle, snaps):
    """snaps: {turn: our abstract state at our non-forced decision}. Returns (hist_vec, hist_text) for the current turn."""
    me, op = battle.player_role, battle.opponent_role
    abt = actions_by_turn(battle)
    t_now = battle.turn
    prev = sorted(t for t in abt if t < t_now and op in abt[t])[-5:]
    hist = []
    for t in prev:
        a = abt[t][op]
        if a is None:
            hist.append(None); continue
        s = snaps.get(t)
        threatened, maxdmg, status_mv = None, None, False
        if s is not None and s.active(0).hp > 0 and s.active(1).hp > 0 and not s.active(1).unseen:
            threatened = F.best_move(s.active(0), s.active(1))[1] >= s.active(1).hp
            if a[0] == "m":
                mon = next((p for p in battle.opponent_team.values() if p.species == s.active(1).name), None)
                mv = Mo.conv_move(mon.moves[a[1]]) if mon is not None and a[1] in mon.moves else None
                if mv is not None:
                    status_mv = mv.cat == "X" or not mv.bp
                    seen = [m for m in s.active(1).moves if not m.id.startswith("unrevealed")] + [mv]
                    dm = [Mo.dmg(s.active(1), s.active(0), m) for m in seen]
                    maxdmg = Mo.dmg(s.active(1), s.active(0), mv) >= max(dm) - 1e-9 and not status_mv
        hist.append({"kind": a[0], "name": a[1], "threat": threatened, "maxdmg": maxdmg, "status": status_mv})
    ours = [abt[t][me][0] for t in sorted(t for t in abt if t < t_now and abt[t].get(me))][-3:]
    return hist_feats(hist, ours), hist_text(hist, ours)
