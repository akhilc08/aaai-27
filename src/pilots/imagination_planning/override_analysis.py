"""Round 3: what does Jev prefer when it overrides the LLM's #1 (round-2 arms B and C, qwen3-30b)?"""
import json, glob, collections
R = [json.loads(l) for f in sorted(glob.glob("round2_seed[0-9].jsonl")) for l in open(f)]
R = [r for r in R if "log" in r and r["arm"] in "BC"]
verb = lambda a: a.split()[0] if a else ""
cat = collections.Counter(); n_over = n_steps = 0; vb = collections.Counter()
for r in R:
    done = set(); nh = set()
    for s in r["log"]:
        n_steps += 1; c = s["cands"]
        if s["chosen"] != 0:
            n_over += 1; a1, aj = c[0], c[s["chosen"]]
            vb[f"{verb(a1)} -> {verb(aj)}"] += 1
            if a1 in done and aj not in done: cat["LLM #1 repeats a past action, Jev picks a new one"] += 1
            if aj in done and a1 not in done: cat["Jev picks a repeat, LLM #1 was new"] += 1
            if verb(a1) == "go" and verb(aj) != "go": cat["LLM #1 'go', Jev picks object interaction"] += 1
            if verb(a1) != "go" and verb(aj) == "go": cat["LLM #1 interaction, Jev picks 'go'"] += 1
            if verb(a1) in ("examine", "look", "inventory") : cat["LLM #1 is examine/look/inventory (no-op-ish)"] += 1
        done.add(s["action"])
print(f"overrides {n_over}/{n_steps} steps = {n_over/n_steps:.3f}")
for k, v in cat.most_common(): print(f"  {k}: {v} ({v/n_over:.1%} of overrides)")
print("top verb transitions (LLM#1 -> Jev):", vb.most_common(8))

# go->go overrides: is Jev's destination more likely to contain the target? (contents from first visit in any round-2 episode)
import re
ALL = [json.loads(l) for f in sorted(glob.glob("round2_seed[0-9].jsonl")) for l in open(f)]
seen = collections.defaultdict(dict)
for r in ALL:
    for s in r.get("log", []):
        if s["action"].startswith("go to ") and s["t"] < 3 or s["action"].startswith("go to "):
            seen[r["game"]].setdefault(s["action"], s["obs"])
def has_t(game, a):
    o = seen[game].get(a); tgt = game.split("/")[-3].split("-")[1].lower()
    return None if o is None else bool(re.search(rf"\b{tgt} \d", o))
k = collections.Counter(); rtype = collections.Counter(); revisit = collections.Counter()
for r in R:
    visited = set()
    for s in r["log"]:
        c = s["cands"]
        if s["chosen"] != 0 and c[0].startswith("go to") and c[s["chosen"]].startswith("go to"):
            a, b = has_t(r["game"], c[0]), has_t(r["game"], c[s["chosen"]])
            if a is not None and b is not None: k[(a, b)] += 1
            revisit[(c[0] in visited, c[s["chosen"]] in visited)] += 1
            rtype[(re.sub(r" \d+$", "", c[0][6:]), re.sub(r" \d+$", "", c[s["chosen"]][6:]))] += 1
        if s["action"].startswith("go to"): visited.add(s["action"])
print("go->go overrides, (LLM#1 dest has target, Jev dest has target):", dict(k))
print("go->go overrides, (LLM#1 dest visited before, Jev dest visited before):", dict(revisit))
print("top receptacle-type swaps (LLM#1 -> Jev):", rtype.most_common(10))
