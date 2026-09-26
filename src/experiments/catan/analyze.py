"""Merge LLM shards and compute Wilson CIs + two-proportion z-tests for n=100 runs."""
import glob, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_prop(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))  # two-sided p


def merge_llm():
    games, cost, calls, errs = [], 0.0, 0, 0
    for f in sorted(glob.glob(os.path.join(R, "shards", "llm_*.json"))):
        d = json.load(open(f))
        games += d["games"]
        cost += d["budget"]["llm_cost_usd"]
        calls += d["budget"]["llm_calls"]
        errs += d["budget"]["llm_errors"]
    games.sort(key=lambda g: g["seed"])
    wins = sum(g["test_player_won"] for g in games)
    out = {"condition": "llm", "opponent": "TAF", "model": "qwen/qwen3-30b-a3b-instruct-2507",
           "num_games": len(games), "wins": wins, "win_rate": wins / len(games),
           "budget": {"llm_cost_usd": round(cost, 6), "llm_calls": calls, "llm_errors": errs},
           "games": games}
    json.dump(out, open(os.path.join(R, "llm_n100.json"), "w"), indent=2)
    return out


def main():
    data = {"llm": merge_llm()}
    for c in ("notrade", "heuristic"):
        data[c] = json.load(open(os.path.join(R, f"{c}_n100.json")))
    rows = {}
    for c, d in data.items():
        g = d["games"]; n = len(g); k = sum(x["test_player_won"] for x in g)
        offers = sum(x["stats"].get("offers_made", 0) for x in g)
        conf = sum(x["stats"].get("trades_confirmed", 0) for x in g)
        lo, hi = wilson(k, n)
        rows[c] = (k, n)
        acc = f"{conf/offers:.1%}" if offers else "n/a"
        print(f"{c:9s} {k}/{n} = {k/n:.2f}  95% Wilson [{lo:.3f}, {hi:.3f}]  "
              f"offers/game {offers/n:.1f}  accepted {acc}  avg turns {sum(x['turns'] for x in g)/n:.0f}")
    for other in ("heuristic", "notrade"):
        z, p = two_prop(*rows["llm"], *rows[other])
        print(f"llm vs {other}: z={z:.2f}, p={p:.3f}")
    print("LLM cost USD:", data["llm"]["budget"])


if __name__ == "__main__":
    main()
