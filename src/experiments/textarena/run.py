"""v2 runner: pairings of {baseline, cot2, belief} on TextArena games, alternating seats.

Illegal actions are caught BEFORE env.step for every arm (same validator), re-asked up to 2 times with the reason,
then replaced by a deterministic legal fallback ('forced'), so invalid moves cannot decide a game.
Usage: uv run python run.py [--smoke]
"""
import json, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import textarena as ta

from agents import ARMS, BudgetExceeded, spent_usd
from legality import fallback, legal_line, unwrap, validate

RUNS = Path(__file__).parent / "runs"
GAMES_DIR = RUNS / "games"; GAMES_DIR.mkdir(parents=True, exist_ok=True)
import os
TAG = os.environ.get("RUN_TAG", "")  # e.g. kuhn_fix -> separate raw/games outputs
RAW = RUNS / (f"results_raw_{TAG}.jsonl" if TAG else "results_raw.jsonl")
if TAG:
    GAMES_DIR = RUNS / f"games_{TAG}"; GAMES_DIR.mkdir(parents=True, exist_ok=True)
_raw_lock = threading.Lock()

PAIRINGS = [("belief", "baseline"), ("belief", "cot2"), ("cot2", "baseline")]  # (A, B): report A's win rate
GAMES = {"LiarsDice-v0": 100, "KuhnPoker-v0-medium": 100, "TicTacToe-v0": 40}
MAX_STEPS, MAX_RETRIES, WORKERS = 200, 2, 16


def play(pair, game, idx):
    A, B = pair
    a_pid = idx % 2
    arm_of = {a_pid: A, 1 - a_pid: B}
    env = ta.make(game); env.reset(num_players=2, seed=1000 + idx)
    base = unwrap(env)
    agents = {p: ARMS[arm_of[p]]({"pairing": f"{A}_vs_{B}", "game": game, "game_idx": idx, "pid": p}) for p in (0, 1)}
    stats = {p: {"turns": 0, "illegal_first": 0, "retries": 0, "forced": 0, "tokens": [0, 0]} for p in (0, 1)}
    turns, done, steps = [], False, 0
    while not done and steps < MAX_STEPS:
        pid, obs = env.get_observation()
        legal = legal_line(game, base, pid)
        raw_action, info = agents[pid].decide(obs, legal)
        s = stats[pid]; s["turns"] += 1
        for k in (0, 1): s["tokens"][k] += info["tokens"][k]
        canon, err = validate(game, base, pid, raw_action)
        if err: s["illegal_first"] += 1
        tries = 0
        while err and tries < MAX_RETRIES:
            tries += 1; s["retries"] += 1
            raw_action, rinfo = agents[pid].retry(obs, legal, err)
            for k in (0, 1): s["tokens"][k] += rinfo["tokens"][k]
            canon, err = validate(game, base, pid, raw_action)
        forced = False
        if err:
            canon, forced = fallback(game, base, pid), True; s["forced"] += 1
        done, _ = env.step(action=canon); steps += 1
        turns.append({"step": steps, "pid": pid, "arm": arm_of[pid], "legal": legal, "obs_tail": obs[-600:],
                      "raw": info["raw"], "thought": info.get("thought"), "belief": info.get("belief"),
                      "action": canon, "retries": tries, "forced": forced})
    rewards, game_info = env.close()
    env_invalid = sum(1 for p in (0, 1) if game_info.get(p, {}).get("invalid_move"))
    if rewards is None: outcome = "timeout"
    elif rewards[0] == rewards[1]: outcome = "draw"
    else: outcome = "A_win" if rewards[a_pid] > rewards[1 - a_pid] else "B_win"
    payoff = None
    if game.startswith("KuhnPoker"):
        ch = base.state.game_state["player_chips"]; payoff = ch[a_pid] - ch[1 - a_pid]  # A's chip lead (antes symmetric)
    elif game.startswith("LiarsDice"):
        rd = base.state.game_state["remaining_dice"]; payoff = rd[a_pid] - rd[1 - a_pid]  # A's dice margin
    rec = {"pairing": f"{A}_vs_{B}", "A": A, "B": B, "game": game, "idx": idx, "a_pid": a_pid, "outcome": outcome,
           "payoff_A": payoff, "rewards": rewards, "steps": steps, "env_invalid_terminations": env_invalid,
           "any_forced": any(stats[p]["forced"] for p in (0, 1)),
           "stats": {"A": stats[a_pid], "B": stats[1 - a_pid]}, "reason": game_info.get(0, {}).get("reason"),
           "spent_usd_after": spent_usd()}
    (GAMES_DIR / f"{A}_vs_{B}__{game}_{idx}.json").write_text(json.dumps({**rec, "turns": turns}, indent=1))
    with _raw_lock, RAW.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# Budget-prioritized plan (cap $3.00 cumulative): cells in priority order with target game counts.
PRIORITY = [(("belief", "cot2"), "LiarsDice-v0", 100), (("belief", "cot2"), "KuhnPoker-v0-medium", 100),
            (("belief", "baseline"), "LiarsDice-v0", 100), (("belief", "baseline"), "KuhnPoker-v0-medium", 100),
            (("belief", "baseline"), "TicTacToe-v0", 40), (("belief", "cot2"), "TicTacToe-v0", 40),
            (("cot2", "baseline"), "LiarsDice-v0", 100), (("cot2", "baseline"), "KuhnPoker-v0-medium", 100),
            (("cot2", "baseline"), "TicTacToe-v0", 40)]
if TAG == "kuhn_fix":
    PRIORITY = [(("belief", "cot2"), "KuhnPoker-v0-medium", 100)]
GATE_USD, EST_PER_GAME = 3.78, 0.008  # don't start a game unless spent + in-flight estimate stays under the gate


def main():
    done_keys = set()
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            r = json.loads(l); done_keys.add((r["pairing"], r["game"], r["idx"]))
    jobs = []
    for pair, g, n in PRIORITY:  # interleave the two cells of a tier so both grow together
        jobs += [(pair, g, i) for i in range(n) if (f"{pair[0]}_vs_{pair[1]}", g, i) not in done_keys]
    tiers = {}
    for j in jobs: tiers.setdefault((j[0], "TicTacToe" in j[1]), []).append(j)
    ordered = []
    for key in dict.fromkeys((j[0], "TicTacToe" in j[1]) for j in jobs):
        t = tiers[key]; ordered += sorted(t, key=lambda j: (j[2], j[1]))
    print(f"{len(ordered)} candidate games; spent so far ${spent_usd():.3f}", flush=True)
    t0 = time.time(); running = {}; it = iter(ordered); stop = False
    with ThreadPoolExecutor(WORKERS) as ex:
        while True:
            while not stop and len(running) < WORKERS:
                if spent_usd() + (len(running) + 1) * EST_PER_GAME > GATE_USD:
                    stop = True; print(f"BUDGET GATE reached at ${spent_usd():.3f}; no new games", flush=True); break
                j = next(it, None)
                if j is None: stop = True; break
                running[ex.submit(play, *j)] = j
            if not running: break
            fut = next(as_completed(running)); p, g, i = running.pop(fut)
            try:
                r = fut.result()
                print(f"[{time.time()-t0:6.0f}s] {r['pairing']} {g}#{i} -> {r['outcome']} payoff={r['payoff_A']} "
                      f"forced={r['any_forced']} ${r['spent_usd_after']:.3f}", flush=True)
            except BudgetExceeded as e:
                print(f"BUDGET STOP: {e}", flush=True); stop = True
            except Exception:
                print(f"{p} {g}#{i} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"done. total spent ${spent_usd():.3f}", flush=True)


if __name__ == "__main__":
    main()
