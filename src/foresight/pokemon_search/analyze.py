"""Aggregate results.jsonl + decision logs + spend into a markdown table."""
import json, math, os, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots")
import jevlib as J

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


agg = defaultdict(lambda: [0, 0, []])
for l in open(os.path.join(D, "results.jsonl")):
    r = json.loads(l)
    k = (r["arm"], r["opp"], r["fmt"])
    agg[k][0] += r["wins"]; agg[k][1] += r["n"]; agg[k][2] += r["turns"]

print("| arm | opp | fmt | n | win% | 95% CI | mean lat (s) | p95 lat | max lat | >15s | leaves/turn | opp-policy q/turn | opp mass | chance mass | errors | $/battle |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (arm, opp, fmt), (w, n, turns) in sorted(agg.items()):
    f = os.path.join(D, f"dec_{arm}_{opp}_{fmt}.jsonl")
    lat = leaves = pq = om = cm = "-"
    p95 = mx = over = errs = "-"
    if os.path.exists(f):
        L = [json.loads(x) for x in open(f)]
        errs = sum("error" in x for x in L)
        L = [x for x in L if "error" not in x]
        if L:
            ls = sorted(x["lat"] for x in L)
            lat = f"{st.mean(ls):.2f}"; p95 = f"{ls[int(.95 * len(ls)) - 1]:.2f}"; mx = f"{ls[-1]:.2f}"
            over = sum(x > 15 for x in ls)
            leaves = f"{st.mean(x.get('leaves', 0) for x in L):.0f}"
            pq = f"{st.mean(x.get('policy_q', 0) for x in L):.0f}"
            oms = [x["opp_mass"] for x in L if "opp_mass" in x]
            cms = [x["chance_mass"] for x in L if "chance_mass" in x]
            om = f"{st.mean(oms):.2f}" if oms else "-"
            cm = f"{st.mean(cms):.2f}" if cms else "-"
    cost = J.spend(f"pksearch_{arm}_{opp}") / n if n else 0
    lo, hi = wilson(w, n)
    print(f"| {arm} | {opp} | {fmt} | {n} | {100 * w / n:.0f} | {100 * lo:.0f}-{100 * hi:.0f} | {lat} | {p95} | {mx} | {over} | {leaves} | {pq} | {om} | {cm} | {errs} | {cost:.4f} |")
print(f"\nTotal pksearch spend: ${J.spend('pksearch'):.3f}")


def auroc(p, y):
    pairs = sorted(zip(p, y))
    ranks, i = {}, 0
    r = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for k in range(i, j):
            r[k] = (i + j + 1) / 2
        i = j
    npos = sum(y)
    nneg = len(y) - npos
    if not npos or not nneg:
        return float("nan")
    return (sum(rk for rk, (_, yy) in zip(r, pairs) if yy) - npos * (npos + 1) / 2) / (npos * nneg)


print("\n## Jev P(win) forecast calibration (root positions of real battles; only arms where Jev P(win) was asked)")
won = {}
for l in open(os.path.join(D, "battles.jsonl")):
    x = json.loads(l)
    won[x["battle"]] = x["won"]
print("| arms | turns | battles | win base rate | mean forecast | Brier | Brier(base rate) | AUROC | AUROC turns<=5 |")
print("|---|---|---|---|---|---|---|---|---|")
allp, ally, allt = [], [], []
for arm in ["P1", "P2", "X1", "X2", "XM2", "X3"]:
    for f in os.listdir(D):
        if f.startswith(f"dec_{arm}_") and f.endswith(".jsonl"):
            for l in open(os.path.join(D, f)):
                x = json.loads(l)
                # HP-leaf arms logged a constant 0.5 before a bug fix (battles numbered < 3380); exclude those
                bad = arm.startswith("X") and int(x.get("battle", "x-0").rsplit("-", 1)[1]) < 3380
                if "root_value" in x and x.get("battle") in won and not bad:
                    allp.append(x["root_value"]); ally.append(int(won[x["battle"]])); allt.append((arm, x["turn"], x["battle"]))
for name, sel in [("P1,P2 (Jev-leaf arms)", {"P1", "P2"}), ("X arms after fix (HP-leaf arms)", {"X1", "X2", "XM2", "X3"}), ("all", None)]:
    idx = [i for i, t in enumerate(allt) if sel is None or t[0] in sel]
    if not idx:
        continue
    p = [allp[i] for i in idx]; y = [ally[i] for i in idx]
    e = [i for i in idx if allt[i][1] <= 5]
    base = sum(y) / len(y)
    br = sum((a - b) ** 2 for a, b in zip(p, y)) / len(p)
    print(f"| {name} | {len(p)} | {len(set(allt[i][2] for i in idx))} | {base:.2f} | {st.mean(p):.2f} | {br:.3f} | {base * (1 - base):.3f} | "
          f"{auroc(p, y):.3f} | {auroc([allp[i] for i in e], [ally[i] for i in e]):.3f} |")
print("\nReliability (all arms):")
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
    s = [(a, b) for a, b in zip(allp, ally) if lo <= a < hi]
    if s:
        print(f"- forecast {lo:.1f}-{min(hi, 1):.1f}: n={len(s)}, mean forecast {st.mean(a for a, _ in s):.2f}, actual win rate {st.mean(b for _, b in s):.2f}")
