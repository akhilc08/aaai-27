import json, random, re, sys
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/src/pilots"); import jevlib as J
TAG = "pokemon"

def mon_str(d, active=False, boosts=None):
    s = f"{d['species']} {d['hp']}% HP" + (f", {d['status']}" if d['status'] else "") + (f", terastallized {d['tera']}" if d.get('tera') else "")
    if d['hp'] == 0: s = f"{d['species']} (fainted)"
    elif d['moves']: s += "; revealed moves: " + ", ".join(d['moves'])
    if active and boosts: s += "; stat boosts: " + ", ".join(f"{k} {v:+d}" for k, v in boosts.items() if v)
    return s

def render(x):
    sn, me, op = x["snap"], x["me"], x["opp"]
    L = [f"Pokemon Showdown Gen 9 Random Battle (singles, level-balanced random teams, no team preview). Turn {sn['turn']}.",
         "You are player " + me + ". Opponent is " + op + ", a human rated about " + str(x.get("rating") or "?") + "."]
    L.append("YOUR active: " + mon_str(sn["st"][me][sn["active"][me]], True, sn["boosts"][me]))
    L.append("OPPONENT active: " + mon_str(sn["st"][op][sn["active"][op]], True, sn["boosts"][op]))
    for who, lab in ((me, "Your"), (op, "Opponent's")):
        bench = [mon_str(v) for k, v in sn["st"][who].items() if k != sn["active"][who]]
        L.append(f"{lab} other revealed Pokemon (6 total per team; unrevealed ones unknown): " + ("; ".join(bench) or "none"))
    f = sn["field"]
    L.append(f"Field: weather {f['weather'] or 'none'}; terrain/rooms {', '.join(f['terrain']) or 'none'}; "
             f"hazards/screens on your side {', '.join(sn['sidecond'][me]) or 'none'}; on opponent's side {', '.join(sn['sidecond'][op]) or 'none'}.")
    rec = [r.replace(me + " ", "You: ").replace(op + " ", "Opp: ") for r in sn["recent"]]
    L.append("Recent turns: " + (" | ".join(rec) or "none (battle start)"))
    return "\n".join(L)

def options(x):
    return list(x["revealed_moves"]) + ["SWITCH"]

def key(m): return re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")
