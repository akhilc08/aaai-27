"""Cheap LLM (qwen3-30b-a3b-instruct) with the dataset's own prompt on 200 spots (80 pre, 120 post) -> llm.jsonl."""
import sys, json, time, random
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/src/pilots"); import jevlib as J
P = "/Users/sickle/Coding/aaai-27/src/pilots/poker/"

def one(r):
    t = time.time()
    txt, u = J.llm([{"role": "user", "content": r["instruction"]}], max_tokens=20, tag="poker-llm")
    return {"id": r["id"], "text": txt, "cost": u.get("cost"), "latency": time.time() - t}

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(P + "data/spots.jsonl")]
    random.seed(1)
    sub = random.sample([r for r in rows if r["pre"]], 80) + random.sample([r for r in rows if not r["pre"]], 120)
    out = J.pmap(one, sub, workers=10)
    ok = [o for o in out if isinstance(o, dict)]
    print("ok", len(ok), [str(o)[:200] for o in out if not isinstance(o, dict)][:3])
    with open(P + "llm.jsonl", "w") as f:
        for o in ok: f.write(json.dumps(o) + "\n")
    print("spend", J.spend("poker"))
