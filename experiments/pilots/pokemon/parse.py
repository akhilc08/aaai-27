"""Parse Showdown logs into (visible state, opponent action) samples, one per turn per perspective."""
import glob, json, random, re, copy
def side(ident): return ident[:2]
def name(ident): return ident.split(": ", 1)[1].strip() if ": " in ident else ident
def hp(s):
    s = s.split()[0]
    if s == "0" or "fnt" in s: return 0
    a, b = s.split("/"); return round(100 * int(a) / int(b))
def status(s):
    p = s.split(); return p[1] if len(p) > 1 and p[1] != "fnt" else None

def parse(path):
    lines = open(path).read().split("\n")
    ratings = {}
    st = {"p1": {}, "p2": {}}  # side -> nickname -> dict
    active = {"p1": None, "p2": None}
    boosts = {"p1": {}, "p2": {}}
    sidecond = {"p1": [], "p2": []}
    field = {"weather": None, "terrain": []}
    turn, events, hist = 0, [], []
    lastact = {"p1": None, "p2": None}
    blocks = []  # (turn, snapshot, lines of the turn)
    cur = None
    for ln in lines:
        p = ln.split("|")
        if len(p) < 2: continue
        t = p[1]
        if t == "player" and len(p) > 5 and p[5].isdigit(): ratings[p[2]] = int(p[5])
        if t == "turn":
            if cur: blocks.append(cur)
            turn = int(p[2])
            snap = dict(turn=turn, st=copy.deepcopy(st), active=dict(active), boosts=copy.deepcopy(boosts),
                        sidecond=copy.deepcopy(sidecond), field=copy.deepcopy(field), recent=list(hist[-3:]),
                        lastact=dict(lastact))
            cur = {"snap": snap, "lines": []}
            events = []
            continue
        if cur is not None: cur["lines"].append(p)
        if t in ("switch", "drag", "replace"):
            s, n = side(p[2]), name(p[2])
            d = st[s].setdefault(n, {"species": p[3].split(",")[0], "moves": [], "hp": 100, "status": None, "tera": None})
            if len(p) > 4: d["hp"], d["status"] = hp(p[4]), status(p[4])
            active[s] = n; boosts[s] = {}
            events.append(f"{s} {'switched to' if t!='drag' else 'was dragged to'} {d['species']}")
            if t == "switch" and cur: lastact[s] = lastact[s]  # set below at turn end
        elif t == "move":
            s, n = side(p[2]), name(p[2]); mv = p[3]
            tail = "|".join(p[4:])
            if "[from]" not in tail and s in st and n in st[s] and mv not in st[s][n]["moves"]:
                st[s][n]["moves"].append(mv)
            if "[from]" not in tail and n in st.get(s, {}): st[s][n]["last"] = mv
            events.append(f"{s} {st[s].get(n,{}).get('species',n)} used {mv}" + (" (missed)" if "[miss]" in tail else ""))
        elif t in ("-damage", "-heal", "-sethp") and len(p) > 3:
            s, n = side(p[2]), name(p[2])
            if n in st[s]:
                st[s][n]["hp"], st[s][n]["status"] = hp(p[3]), status(p[3]) or st[s][n]["status"]
        elif t == "faint":
            s, n = side(p[2]), name(p[2])
            if n in st[s]: st[s][n]["hp"] = 0
            events.append(f"{s} {st[s].get(n,{}).get('species',n)} fainted")
        elif t == "-status": st[side(p[2])].get(name(p[2]), {})["status"] = p[3]; events.append(f"{side(p[2])} {name(p[2])} got {p[3]}")
        elif t == "-curestatus": st[side(p[2])].get(name(p[2]), {})["status"] = None
        elif t in ("-boost", "-unboost"):
            s = side(p[2]); k = p[3]; v = int(p[4]) * (1 if t == "-boost" else -1)
            boosts[s][k] = boosts[s].get(k, 0) + v
        elif t in ("-clearallboost",): boosts = {"p1": {}, "p2": {}}
        elif t == "-terastallize": st[side(p[2])].get(name(p[2]), {})["tera"] = p[3]; events.append(f"{side(p[2])} {name(p[2])} terastallized {p[3]}")
        elif t == "-weather": field["weather"] = None if p[2] == "none" else p[2]
        elif t == "-fieldstart": field["terrain"].append(p[2].replace("move: ", ""))
        elif t == "-fieldend": field["terrain"] = [x for x in field["terrain"] if x != p[2].replace("move: ", "")]
        elif t == "-sidestart": sidecond[p[2][:2]].append(p[3].replace("move: ", ""))
        elif t == "-sideend":
            c = p[3].replace("move: ", ""); l = sidecond[p[2][:2]]
            if c in l: l.remove(c)
        elif t == "upkeep":
            hist.append(f"Turn {turn}: " + "; ".join(events)); events = []
        elif t == "win":
            if cur: blocks.append(cur); cur = None
    return ratings, blocks

def label(lines, opp):
    """First voluntary action of `opp` in a turn block: ('switch', species) or ('move', name) or None."""
    for p in lines:
        t = p[1]
        if t == "upkeep": return None
        if t in ("switch", "move", "cant") and p[2][:2] == opp:
            if t == "cant": return None
            if t == "switch": return ("switch", p[3].split(",")[0])
            if "[from]" in "|".join(p[4:]): return None
            return ("move", p[3])
    return None

def build():
    rng = random.Random(0); out = []
    for f in sorted(glob.glob("data/logs/*.log")):
        bid = f.split("/")[-1][:-4]
        ratings, blocks = parse(f)
        prev = {"p1": None, "p2": None}
        for b in blocks:
            for me, opp in (("p1", "p2"), ("p2", "p1")):
                lab = label(b["lines"], opp); sn = b["snap"]
                pa = prev[opp]; prev_next = lab[0] if lab else None
                if lab and sn["active"][opp] and sn["active"][me]:
                    oa = sn["st"][opp][sn["active"][opp]]
                    out.append(dict(battle=bid, turn=sn["turn"], me=me, opp=opp, rating=ratings.get(opp),
                                    snap=sn, label=lab, prev_opp_action=pa,
                                    revealed_moves=list(oa["moves"]),
                                    opp_bench_alive=sum(1 for k, v in sn["st"][opp].items() if k != sn["active"][opp] and v["hp"] > 0)))
                if lab: prev[opp] = lab[0]
    return out

if __name__ == "__main__":
    s = build()
    json.dump(s, open("data/all_samples.json", "w"))
    import collections
    print(len(s), collections.Counter(x["label"][0] for x in s))
    el = [x for x in s if x["label"][0] == "switch" or x["label"][1] in x["revealed_moves"]]
    print("eligible(b)", len(el), "of which move", sum(x["label"][0]=="move" for x in el))
