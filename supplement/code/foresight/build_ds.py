"""Build datasets from logged battles: P(win) rows, opponent-next-action rows, delta (line) rows.
usage: build_ds.py FMT   -> data/ds_{FMT}.pkl"""
import glob, os, pickle, sys, zlib
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as Mo
import fmt as F
import feats as X
import search as Se

HERE = os.path.dirname(os.path.abspath(__file__))


def split_of(tag):
    h = zlib.crc32(tag.encode()) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def opp_label_idx(s, act):
    """Map the opponent's actual action onto our-view candidate list Mo.actions(s, 1). Returns (idx or -1, is_switch)."""
    cands = Mo.actions(s, 1)
    if act is None:
        return -1, None, cands
    kind, name = act
    them = s.sides[1]
    for k, a in enumerate(cands):
        if kind == "s" and a[0] == "s" and them[a[1]].name == name:
            return k, True, cands
        if kind == "m" and a[0] == "m" and s.active(1).moves[a[1]].id == name:
            return k, False, cands
    return -1, kind == "s", cands


def prev_info(r_me, r_opp):
    """Public summary of the opponent's action at an earlier turn, computed from OUR state at that turn."""
    a = r_opp["act"]
    if a is None:
        return None
    s = pickle.loads(r_me["state"]) if r_me is not None else None
    threatened, maxdmg, status_mv = None, None, False
    if s is not None and s.active(0).hp > 0 and s.active(1).hp > 0 and not s.active(1).unseen:
        d0 = F.best_move(s.active(0), s.active(1))[1]
        threatened = d0 >= s.active(1).hp
        if a[0] == "m":
            so = pickle.loads(r_opp["state"])
            mv = next((m for m in so.active(0).moves if m.id == a[1]), None)
            if mv is not None:
                status_mv = mv.cat == "X" or not mv.bp
                seen = [m for m in s.active(1).moves if not m.id.startswith("unrevealed")] + [mv]
                dm = [Mo.dmg(s.active(1), s.active(0), m) for m in seen]
                maxdmg = Mo.dmg(s.active(1), s.active(0), mv) >= max(dm) - 1e-9 and not status_mv
    return {"kind": a[0], "name": a[1], "threat": threatened, "maxdmg": maxdmg, "status": status_mv}


def hist_feats(hist, ours):
    """hist: list of prev_info dicts (oldest first, last 5); ours: our previous action kinds."""
    h = [x for x in hist if x]
    n = len(h)
    sw = [x["kind"] == "s" for x in h]
    thr = [x for x in h if x["threat"]]
    mv = [x for x in h if x["kind"] == "m" and x["maxdmg"] is not None]
    rep = 0
    for x in reversed(h[:-1]):
        if h and x["kind"] == "m" and x["name"] == h[-1]["name"]:
            rep += 1
        else:
            break
    return [n / 5, np.mean(sw) if n else 0.0, float(sw[-1]) if n else 0.0,
            np.mean([x["kind"] == "s" for x in thr]) if thr else -1.0, len(thr) / 5,
            np.mean([x["maxdmg"] for x in mv]) if mv else -1.0, np.mean([x["status"] for x in h]) if n else 0.0,
            rep / 4, float(ours[-1] == "s") if ours else 0.0, np.mean([o == "s" for o in ours]) if ours else 0.0]


HIST_NAMES = ["h_n", "h_sw", "h_last_sw", "h_sw_when_threat", "h_n_threat", "h_maxdmg", "h_status", "h_repeat", "our_last_sw", "our_sw"]


def hist_text(hist, ours):
    parts = []
    for x in hist:
        if not x:
            continue
        if x["kind"] == "s":
            parts.append(f"switched to {x['name']}" + (" while threatened with a KO" if x["threat"] else ""))
        else:
            tag = " (status move)" if x["status"] else " (their highest-damage option)" if x["maxdmg"] else " (not their highest-damage option)" if x["maxdmg"] is False else ""
            parts.append(f"used {x['name']}{tag}" + (" while threatened with a KO" if x["threat"] else ""))
    t = "their previous actions (oldest first): " + ("; ".join(parts) if parts else "none yet")
    t += "\nour previous actions: " + (", ".join("switch" if o == "s" else "attack" for o in ours) if ours else "none yet")
    return t


def cand_feats(s, cands, heur):
    """Per-candidate features for the opponent's options (our POV; side 1 acts on side 0)."""
    op, me = s.active(1), s.active(0)
    dms = [min(Mo.dmg(op, me, op.moves[a[1]]), 1.5) if a[0] == "m" else 0.0 for a in cands]
    mx = max(dms) if dms else 0.0
    out = []
    for k, a in enumerate(cands):
        if a[0] == "m":
            mv = op.moves[a[1]]
            out.append([0.0, heur[k], dms[k], float(dms[k] >= mx - 1e-9 and mx > 0), float(dms[k] * mv.acc >= me.hp),
                        Mo.type_mult(mv.type, me) if mv.type is not None else 1.0, float(mv.cat == "X" or not mv.bp), mv.acc,
                        mv.pri, float(mv.id.startswith("unrevealed")), 0.0, 0.0, 0.0])
        else:
            m = s.sides[1][a[1]]
            taken = F.best_move(me, m)[1]
            out.append([1.0, heur[k], 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, Mo.matchup(s, 1, a[1]), m.hp, min(taken, 1.5)])
    return out


CAND_NAMES = ["is_switch", "heur_p", "dmg", "is_max_dmg", "kos", "type_mult", "status_mv", "acc", "prio", "unrevealed",
              "sw_matchup", "sw_hp", "sw_taken"]


def main(fmt_):
    indir = os.environ.get("INDIR", os.path.join(HERE, "data"))
    files = sorted(glob.glob(os.path.join(indir, f"pos_{fmt_}_*.pkl")))
    rows = []
    for fn in files:
        pair = os.path.basename(fn).split("_")[2:4]
        with open(fn, "rb") as f:
            while True:
                try:
                    rs = pickle.load(f)
                except EOFError:
                    break
                for r in rs:
                    r["pair"] = "-".join(pair)
                rows += rs
    # index decisions by (battle, turn, forced, player)
    by = defaultdict(dict)
    for r in rows:
        by[(r["battle"], r["turn"], r["forced"])][r["player"]] = r
    per_player = defaultdict(dict)
    for r in rows:
        if not r["forced"]:
            per_player[(r["battle"], r["player"])][r["turn"]] = r
    out = {"pwin": [], "opp": [], "delta": []}
    for r in rows:
        if r["tie"]:
            continue
        s = pickle.loads(r["state"])
        if s.terminal() is not None:
            continue
        base = dict(battle=r["battle"], player=r["player"], turn=r["turn"], forced=r["forced"], y=int(r["won"]),
                    split=split_of(r["battle"]), pair=r["pair"], cls=r["cls"])
        num = X.numeric(s)
        out["pwin"].append(dict(base, num=num, text=X.state_text(s), hp=Mo.hp_value(s) / 4, **({"raw": r["raw"]} if "raw" in r else {})))
        if not r["forced"]:
            other = [v for p, v in by[(r["battle"], r["turn"], False)].items() if p != r["player"]]
            opp_name = other[0]["player"] if other else None
            prev_t = sorted(t for t in per_player[(r["battle"], opp_name)] if t < r["turn"])[-5:] if opp_name else []
            mine = per_player[(r["battle"], r["player"])]
            hist = [prev_info(mine.get(t), per_player[(r["battle"], opp_name)][t]) for t in prev_t]
            ours = [mine[t]["act"][0] for t in sorted(t for t in mine if t < r["turn"])[-3:] if mine[t]["act"]]
            if other and s.active(1).hp > 0 and not s.active(1).unseen:
                idx, is_sw, cands = opp_label_idx(s, other[0]["act"])
                if is_sw is not None:
                    labels = [F.act_label(s, 1, a) for a in cands]
                    if idx == -1 and not is_sw:
                        idx = next((k for k, a in enumerate(cands) if a[0] == "m" and s.active(1).moves[a[1]].id.startswith("unrevealed")), -1)
                    heur = Se.HeurEval().run([(s, 1, cands, None)], [])[0][0]
                    out["opp"].append(dict(base, num=num, text=X.state_text(s), cands=labels, y_idx=idx, y_sw=int(is_sw),
                                           cand_sw=[int(a[0] == "s") for a in cands], heur=heur,
                                           cand_raw=[("move " + s.active(1).moves[a[1]].id) if a[0] == "m" else ("switch to " + s.sides[1][a[1]].name) for a in cands],
                                           hist=r["live_hist"][0] if "live_hist" in r else hist_feats(hist, ours),
                                           htext=r["live_hist"][1] if "live_hist" in r else hist_text(hist, ours), cf=cand_feats(s, cands, heur),
                                           opp_cls=other[0]["cls"], **({"raw": r["raw"]} if "raw" in r else {})))
            nxt = per_player[(r["battle"], r["player"])].get(r["turn"] + 2)
            if nxt is not None:
                s2 = pickle.loads(nxt["state"])
                out["delta"].append(dict(base, num=num + X.delta_numeric(s, s2) + X.numeric(s2),
                                         text=X.state_text(s) + "\n" + X.delta_text(s, s2),
                                         hp_leaf=Mo.hp_value(s2) / 4))
    for k, v in out.items():
        sp = defaultdict(int)
        for x in v:
            sp[x["split"]] += 1
        print(k, len(v), dict(sp), "base rate", round(np.mean([x["y"] for x in v]), 3) if k != "opp" else "")
    print("battles", len({r["battle"] for r in rows}))
    with open(os.path.join(indir, f"ds_{fmt_}.pkl"), "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gen9randombattle")
