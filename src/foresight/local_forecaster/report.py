"""Win-rate table from data/arm_results.jsonl (pooled by arm/opponent/format) with Wilson 95% CIs."""
import json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    agg = defaultdict(lambda: {"n": 0, "w": 0, "lat": [], "lmax": 0, "err": 0, "spec": None})
    for l in open(os.path.join(HERE, "data", "arm_results.jsonl")):
        x = json.loads(l)
        k = (x["arm"], x["opponent"], x["fmt"])
        a = agg[k]
        a["n"] += x["n"]; a["w"] += x["wins"]; a["lat"].append((x["lat_mean"], x["n"])); a["lmax"] = max(a["lmax"], x["lat_max"])
        a["err"] += x.get("errors", 0); a["spec"] = (x["opp_model"], x["leaf"], x["depth"])
    print("| arm | opp model | leaf | depth | opponent | format | n | win % | 95% CI | mean s/decision | max s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (arm, opp, fmt), a in sorted(agg.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        lo, hi = wilson(a["w"], a["n"])
        lat = sum(m * n for m, n in a["lat"]) / max(sum(n for _, n in a["lat"]), 1)
        om, lf, d = a["spec"]
        lf = lf.replace("_gen9randombattle.pt", "").replace(":", " ")
        om = om.replace("_gen9randombattle.pt", "").replace(":", " ")
        print(f"| {arm} | {om} | {lf} | {d} | {opp} | {fmt.replace('randombattle', '')} | {a['n']} | {100 * a['w'] / max(a['n'], 1):.1f} | "
              f"{100 * lo:.0f}-{100 * hi:.0f} | {lat:.2f} | {a['lmax']:.1f} |")


if __name__ == "__main__":
    main()
