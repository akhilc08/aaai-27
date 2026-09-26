"""World-model queries: Jev (typed noul) and cheap-LLM baseline, same state and questions."""
import json, re
import agent as A
J = A.J

def state(r, k=8):
    return {"task": r["goal"], "room": r["init"].split("Your task")[0].replace("-= Welcome to TextWorld, ALFRED! =-", "").strip(),
            "recent_history": [{"action": a, "observation": o} for a, o in r["hist"][-k:]],
            "proposed_next_action": r["action"]}

def questions(target):
    return {
        "effect": {"type": "noul", "instructions": "Text household game. Will the proposed next action be accepted by the game and change or reveal something, rather than the game replying 'Nothing happens.'?"},
        "reveal": {"type": "noul", "instructions": f"Text household game. After the proposed next action, will the game's reply mention a {target} (for example '{target} 1' seen at the location, or picking it up)?"},
        "empty": {"type": "noul", "instructions": "Text household game. After the proposed next action, will the game's reply say the place contains nothing (e.g. 'you see nothing')?"},
    }

def jev_predict(r):
    a = J.jev(state(r), questions(r["target"]), tag=A.TAG)
    return {k: a[k]["noul"] for k in ("effect", "reveal", "empty")}

def llm_predict(r):
    qs = questions(r["target"])
    p = ("Predict the outcome of the proposed next action BEFORE seeing it. State:\n" + json.dumps(state(r), indent=1) +
         "\n\nFor each question give a probability 0-100 that the answer is yes.\n" +
         "\n".join(f"{k}: {v['instructions']}" for k, v in qs.items()) +
         '\nReply with JSON only: {"effect": p, "reveal": p, "empty": p}')
    t, _ = J.llm([{"role": "user", "content": p}], max_tokens=60, tag=A.TAG)
    d = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
    return {k: max(0, min(100, float(d[k]))) / 100 for k in ("effect", "reveal", "empty")}

def labels(r):
    o = r["obs"]
    return {"effect": int(o != "Nothing happens."), "reveal": int(bool(re.search(rf"\b{r['target']} \d", o))),
            "empty": int("see nothing" in o)}
