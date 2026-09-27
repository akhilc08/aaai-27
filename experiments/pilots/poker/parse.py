"""Parse PokerBench test instructions into structured spots; sample ~400 pre + ~600 post; write data/spots.jsonl."""
import re, json, random, glob, ast, csv
from datasets import load_dataset
D = "/Users/sickle/Coding/aaai-27/experiments/pilots/poker/data"
R = {"Two": "2", "Three": "3", "Four": "4", "Five": "5", "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9",
     "Ten": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}
RV = {r: i for i, r in enumerate("23456789TJQKA", 2)}

def card(s):
    r, _, suit = s.strip().split()
    return R[r] + suit[0].lower()

def parse(ins, out):
    d = {}
    d["pos"] = re.search(r"your position is (\w+)", ins).group(1)
    h = re.search(r"your holding is \[(.+?) and (.+?)\]", ins)
    d["hand"] = [card(h.group(1)), card(h.group(2))]
    d["pot"] = float(re.search(r"current pot size is ([\d.]+)", ins).group(1))
    pre = re.search(r"Before the flop, (.*?)\. Assume", ins).group(1)
    d["preflop_line"] = pre
    board, street, cur = [], "preflop", pre
    fm = re.search(r"The flop comes (.+?), (.+?), and (.+?), then (.*?)\.\n", ins + "\n")
    fm = re.search(r"The flop comes ([^,]+), ([^,]+), and ([^,]+?)(?:, then (.*?))?\.\n", ins)
    if fm:
        board = [card(fm.group(i)) for i in (1, 2, 3)]; street = "flop"; cur = fm.group(4) or ""
        d["flop_line"] = cur
    for st in ("turn", "river"):
        m = re.search(rf"The {st} comes ([^,]+?)(?:, then (.*?))?\.\n", ins)
        if m:
            board.append(card(m.group(1))); street = st; cur = m.group(2) or ""
            d[st + "_line"] = cur
    d["board"], d["street"], d["cur_line"] = board, street, cur
    # facing a bet on current street?
    if street == "preflop":
        facing = bool(re.search(r"raise|all in", cur)) or d["pos"] != "BB"
    else:
        acts = [a.strip() for a in re.split(r",\s*(?:and )?|\band\b", cur) if a.strip()]
        facing = False
        for a in acts:
            if re.search(r"\b(bet|raise|all in)\b", a): facing = not a.startswith(d["pos"])
            elif re.search(r"\bcall\b", a) and a.startswith(d["pos"]): facing = False
        # hero is IP/OOP only in positions; simple rule above
    d["facing"] = facing
    d["legal"] = ["fold", "call", "raise"] if facing else ["check", "bet" if street != "preflop" else "raise"]
    parts = out.strip().split()
    d["label"], d["size"], d["label_raw"] = parts[0], (float(parts[1]) if len(parts) > 1 else None), out.strip()
    return d

if __name__ == "__main__":
    ds = load_dataset("RZ412/PokerBench", cache_dir=D)["test"]
    rows = []
    for i, x in enumerate(ds):
        try:
            d = parse(x["instruction"], x["output"]); d["id"] = i; d["pre"] = i >= 10000; d["instruction"] = x["instruction"]
            rows.append(d)
        except Exception as e:
            print("fail", i, e)
    bad = [r for r in rows if r["label"] not in r["legal"]]
    print("parsed", len(rows), "label not in legal:", len(bad))
    from collections import Counter
    print(Counter((r["pre"], r["label"], tuple(r["legal"])) for r in bad).most_common(10))
    for r in bad[:3]: print(r["instruction"][300:], r["label_raw"])
    random.seed(0)
    pre = [r for r in rows if r["pre"]]; post = [r for r in rows if not r["pre"]]
    samp = random.sample(pre, 400) + random.sample(post, 600)
    with open(f"{D}/spots.jsonl", "w") as f:
        for r in samp: f.write(json.dumps(r) + "\n")
    print(Counter((r["pre"], r["label"]) for r in samp))
