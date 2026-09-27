"""Convert 160 rated human gen9randombattle replays (pilots/pokemon/data/logs, read-only) into opponent-action rows
with the same features as the bot data (both perspectives). Own unrevealed team is missing (spectator view).
usage: human.py -> data/human_opp.pkl"""
import glob, logging, os, pickle, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poke_env.battle import Battle
import model as Mo
import fmt as F
import feats as X
import search as Se
import hist_live as HL
from build_ds import cand_feats, opp_label_idx

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = "/Users/sickle/Coding/aaai-27/experiments/pilots/pokemon/data/logs"  # Jev pokemon pilot, read-only
log = logging.getLogger("h"); log.setLevel(logging.ERROR)


def rows_for(path, role):
    lines = open(path).read().splitlines()
    names = {}
    for l in lines:
        p = l.split("|")
        if len(p) > 3 and p[1] == "player":
            names[p[2]] = p[3]
    tag = os.path.basename(path)[:-4]
    b = Battle(tag, names.get(role, "x"), log, gen=9)
    b.player_role = role
    snaps, pending, out = {}, [], []
    for l in lines:
        p = l.split("|")
        if len(p) < 2:
            continue
        try:
            b.parse_message(p)
        except Exception:
            continue
        if p[1] == "turn":
            try:
                s = Mo.from_battle(b)
            except Exception:
                continue
            if s.active(0).hp <= 0 or s.active(1).hp <= 0 or s.active(1).unseen:
                continue
            hist = HL.live_hist(b, snaps)
            snaps[b.turn] = s
            pending.append((b.turn, s, hist))
    abt = HL.actions_by_turn(b)
    opp = "p2" if role == "p1" else "p1"
    won = None
    for l in lines[::-1]:
        p = l.split("|")
        if len(p) > 2 and p[1] == "win":
            won = p[2] == names.get(role)
            break
    for t, s, hist in pending:
        a = abt.get(t, {}).get(opp)
        if a is None:
            continue
        idx, is_sw, cands = opp_label_idx(s, list(a))
        if idx == -1 and not is_sw:
            idx = next((k for k, c in enumerate(cands) if c[0] == "m" and s.active(1).moves[c[1]].id.startswith("unrevealed")), -1)
        if idx < 0 or len(cands) < 2:
            continue
        heur = Se.HeurEval().run([(s, 1, cands, None)], [])[0][0]
        out.append(dict(battle=tag, player=role, turn=t, y_idx=idx, y_sw=int(is_sw), cands=[F.act_label(s, 1, c) for c in cands],
                        cand_sw=[int(c[0] == "s") for c in cands], heur=heur, num=X.numeric(s), text=X.state_text(s),
                        hist=hist[0], htext=hist[1], cf=cand_feats(s, cands, heur), opp_cls="human", y=int(bool(won)),
                        split="A" if zlib.crc32(tag.encode()) % 2 == 0 else "B"))
    return out


def main():
    rows = []
    for fn in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        for role in ("p1", "p2"):
            try:
                rows += rows_for(fn, role)
            except Exception as e:
                print("fail", fn, role, repr(e)[:120])
    print("rows", len(rows), "battles", len({r["battle"] for r in rows}), "A", sum(r["split"] == "A" for r in rows))
    pickle.dump(rows, open(os.path.join(HERE, "data", "human_opp.pkl"), "wb"))


if __name__ == "__main__":
    main()
