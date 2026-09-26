"""Run the hybrid-vs-bots evaluation and write results JSON.

Usage:
  uv run python run_eval.py --condition llm --num 5
  uv run python run_eval.py --condition heuristic --num 5
  uv run python run_eval.py --condition notrade --num 5

Seat 0 is the test player; seats 1-3 are TradeAwareValueFunctionPlayer.
Seating order is randomized by catanatron per game (seeded).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from catanatron import Color, Game
from catanatron.players.value import ValueFunctionPlayer

from hybrid_player import (
    BUDGET,
    HARD_BUDGET_USD,
    HeuristicTrader,
    HybridLLMTrader,
    TradeAwareValueFunctionPlayer,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def make_test_player(condition: str, color: Color):
    if condition == "llm":
        return HybridLLMTrader(color)
    if condition == "heuristic":
        return HeuristicTrader(color)
    if condition == "notrade":
        return ValueFunctionPlayer(color)
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["llm", "heuristic", "notrade"], required=True)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--opponent", choices=["F", "TAF"], default="TAF",
                    help="F=vanilla ValueFunctionPlayer (rejects all trades), "
                         "TAF=trade-aware ValueFunctionPlayer")
    ap.add_argument("--out", default=None, help="output JSON path (default: results/<cond>_vs_<opp>_n<N>.json)")
    ap.add_argument("--budget", type=float, default=HARD_BUDGET_USD, help="per-process hard cap (USD)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS_DIR, f"{args.condition}_vs_{args.opponent}_n{args.num}.json")

    games = []
    wins = 0
    t0 = time.time()
    for i in range(args.num):
        seed = args.seed0 + i
        test = make_test_player(args.condition, COLORS[0])
        opp_cls = TradeAwareValueFunctionPlayer if args.opponent == "TAF" else ValueFunctionPlayer
        players = [test] + [opp_cls(c) for c in COLORS[1:]]
        game = Game(players, seed=seed)
        gt = time.time()
        winner = game.play()
        won = winner == COLORS[0]
        wins += int(won)
        rec = {
            "seed": seed,
            "winner": winner.value if winner else None,
            "test_player_won": won,
            "turns": game.state.num_turns,
            "seconds": round(time.time() - gt, 1),
            "test_vp": _vp(game, COLORS[0]),
            "stats": getattr(test, "stats", {}),
            "budget_after": BUDGET.summary(),
        }
        games.append(rec)
        print(json.dumps(rec), flush=True)
        if BUDGET.cost_usd >= args.budget:
            print("HARD BUDGET HIT, stopping", flush=True)
            break

    summary = {
        "condition": args.condition,
        "opponent": args.opponent,
        "model": HybridLLMTrader.Params().model if args.condition == "llm" else None,
        "num_games": len(games),
        "wins": wins,
        "win_rate": round(wins / max(1, len(games)), 3),
        "elapsed_seconds": round(time.time() - t0, 1),
        "budget": BUDGET.summary(),
        "games": games,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "games"}))
    print("wrote", out_path)


def _vp(game, color):
    from catanatron.state_functions import get_actual_victory_points

    return get_actual_victory_points(game.state, color)


if __name__ == "__main__":
    main()
