"""Jev zero-shot choice over legal actions + 9 narrow noul questions, one call per spot -> jev.jsonl."""
import sys, json, time
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/src/pilots"); import jevlib as J
P = "/Users/sickle/Coding/aaai-27/src/pilots/poker/"
DESC = {"fold": "fold: give up the hand", "call": "call: match the current bet",
        "raise": "raise: put in more chips than the current bet (re-raise / open-raise / all in)",
        "check": "check: pass without betting", "bet": "bet: put chips in when no one has bet yet"}
NOUL = {
    "strong": "Is hero's hand very strong right now (a premium preflop hand, or two pair or better postflop)?",
    "made": "Does hero hold at least one pair (a pocket pair, or a hole card pairing the board)?",
    "best": "Is hero's hand likely to be the best hand at this moment?",
    "draw": "Does hero have a meaningful flush draw or straight draw that could improve?",
    "ip": "Is hero in position (acting after the opponent)?",
    "wet": "Is the board wet or coordinated (many straight/flush possibilities, or paired)?",
    "aggr": "Has the opponent shown strength with bets or raises in this hand?",
    "odds": "Is hero facing a bet with good pot odds for a call given hero's hand?",
    "bluff": "Is this a good spot for hero to bluff or semi-bluff?",
}

def state(r):
    s = r["instruction"].split("Here is a game summary:")[1].split("Decide on an action")[0].strip()
    return s + "\nLegal action types for hero: " + ", ".join(r["legal"]) + "."

def one(r):
    q = {"act": {"type": "choice", "instructions": "Which action is the game-theory-optimal (solver) play for hero now?",
                 "criteria": {k: DESC[k] for k in r["legal"]}}}
    q.update({k: {"type": "noul", "instructions": v} for k, v in NOUL.items()})
    t = time.time(); a = J.jev(state(r), q, tag="poker-jev"); dt = time.time() - t
    return {"id": r["id"], "probs": a["act"]["probabilities"], "choice": a["act"]["choice"],
            **{k: a[k]["noul"] for k in NOUL}, "latency": dt}

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(P + "data/spots.jsonl")]
    out = J.pmap(one, rows, workers=10)
    ok = [o for o in out if isinstance(o, dict)]
    print("ok", len(ok), "errors", [str(o)[:200] for o in out if not isinstance(o, dict)][:3])
    with open(P + "jev.jsonl", "w") as f:
        for o in ok: f.write(json.dumps(o) + "\n")
    print("spend", J.spend("poker-jev"))
