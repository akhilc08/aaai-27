"""Aggregate runs/results_raw.jsonl (v2) -> runs/results.json + printed table."""
import json, math
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).parent / "runs"


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def binom_two_sided(k, n):
    """Exact two-sided binomial test vs p=0.5."""
    if n == 0: return None
    pk = math.comb(n, k) / 2 ** n
    return min(1.0, sum(math.comb(n, i) / 2 ** n for i in range(n + 1) if math.comb(n, i) / 2 ** n <= pk * (1 + 1e-9)))


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2: return None, None
    m = sum(xs) / len(xs); sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = 1.96 * sd / math.sqrt(len(xs)); return round(m, 3), [round(m - h, 3), round(m + h, 3)]


def summarize(rs):
    n = len(rs); w = sum(r["outcome"] == "A_win" for r in rs); l = sum(r["outcome"] == "B_win" for r in rs)
    d = sum(r["outcome"] == "draw" for r in rs)
    turns = {s: sum(r["stats"][s]["turns"] for r in rs) for s in "AB"}
    ill = {s: sum(r["stats"][s]["illegal_first"] for r in rs) for s in "AB"}
    forced = {s: sum(r["stats"][s]["forced"] for r in rs) for s in "AB"}
    tok = {s: [round(sum(r["stats"][s]["tokens"][k] for r in rs) / max(n, 1)) for k in (0, 1)] for s in "AB"}
    pm, pci = mean_ci([r["payoff_A"] for r in rs])
    return {"n": n, "A_wins": w, "B_wins": l, "draws": d, "A_win_rate": round(w / n, 3) if n else None,
            "wilson95": wilson(w, n), "A_win_rate_decisive": round(w / (w + l), 3) if w + l else None,
            "wilson95_decisive": wilson(w, w + l), "binom_p_decisive_vs_0.5": binom_two_sided(w, w + l),
            "illegal_first_attempt_rate": {s: round(ill[s] / turns[s], 4) if turns[s] else None for s in "AB"},
            "forced_fallbacks": forced, "avg_tokens_per_game_[prompt,completion]": tok,
            "mean_payoff_A": pm, "payoff_A_95ci": pci,
            "env_invalid_terminations": sum(r["env_invalid_terminations"] for r in rs)}


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    recs = [json.loads(l) for l in (RUNS / (f"results_raw_{tag}.jsonl" if tag else "results_raw.jsonl")).read_text().splitlines()]
    by = defaultdict(list)
    for r in recs: by[(r["pairing"], r["game"])].append(r)
    out, lines = {}, []
    for (p, g), rs in sorted(by.items()):
        allg = summarize(rs); clean = summarize([r for r in rs if not r["any_forced"]])
        out[f"{p}|{g}"] = {"all": allg, "excluding_forced_games": clean}
        a = allg
        lines.append(f"{p:20s} {g:20s} n={a['n']:3d} A WR={a['A_win_rate']} {a['wilson95']} W/L/D={a['A_wins']}/{a['B_wins']}/{a['draws']} "
                     f"p={a['binom_p_decisive_vs_0.5']:.3g} payoff={a['mean_payoff_A']} {a['payoff_A_95ci']} "
                     f"illegal1st A/B={a['illegal_first_attempt_rate']['A']}/{a['illegal_first_attempt_rate']['B']} "
                     f"forced={a['forced_fallbacks']} tok A={a['avg_tokens_per_game_[prompt,completion]']['A']} B={a['avg_tokens_per_game_[prompt,completion]']['B']} "
                     f"| excl-forced n={clean['n']} WR={clean['A_win_rate']} p={clean['binom_p_decisive_vs_0.5']}")
    usage = [json.loads(l) for l in (RUNS / "usage.jsonl").read_text().splitlines()]
    out["_meta"] = {"total_cost_usd": round(sum(u.get("cost") or 0 for u in usage), 4), "n_requests": len(usage),
                    "models_used": sorted({u["model"] for u in usage})}
    if tag == "kuhn_fix":
        from cardcount import count_dir
        out["_card_counting"] = {"before_fix(runs/games)": count_dir(RUNS / "games", "belief_vs_cot2__KuhnPoker"),
                                 "after_fix(runs/games_kuhn_fix)": count_dir(RUNS / "games_kuhn_fix", "belief_vs_cot2__KuhnPoker")}
        print(out["_card_counting"])
    (RUNS / (f"results_{tag}.json" if tag else "results.json")).write_text(json.dumps(out, indent=1))
    print("\n".join(lines)); print(out["_meta"])


if __name__ == "__main__":
    main()
