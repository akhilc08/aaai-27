# autoresearch: ForeSight search and evaluation

Autonomous overnight research loop, modeled on karpathy/autoresearch. You improve how ForeSight turns Laya's opponent forecasts into moves. You edit one file, run a fixed-budget evaluation, keep what helps, discard what doesn't, and repeat until stopped.

## Setup (once)

1. Work in `local_forecaster/autoresearch/`. Copy the current decision logic into `strategy.py`: the leaf evaluation, the search parameters (depth, `our_k`, `opp_mass`, `opp_cap`, chance pruning), and any glue that `run_arm.py` needs. `run_arm.py` must be able to load `autoresearch/strategy.py` in place of the defaults, with no behavior change for the baseline.
2. Create `results.tsv` with the header `exp	dev_wins	dev_n	dev_rate	status	p95_turn_s	description`.
3. Run the baseline (the current headline configuration) as experiment 000.

## What you CAN change: only `strategy.py`

Everything in the decision logic is fair game:

- **Leaf evaluation.** Examples: alive-count bonus, active matchup or speed advantage, boosts and status terms, a blend with Laya's P(win) if a trained leaf exists.
- **Search shape.** Depth, adaptive depth when there are few options, action pruning (`our_k`), opponent pruning (`opp_mass`, `opp_cap`), chance pruning, iterative deepening within the time budget.
- **Use of the forecasts.** Temperature, mixing with a small switch prior (Laya never ranks switch first; its P(switch) AUROC is 0.87), and minimax/expectimax blends.
- **Move ordering, tie-breaking, and caching.**

## What you CANNOT change

- **The forecaster.** Keep `ckpt/laya_opp_hist_gen8randombattle.pt`, with no retraining. Other files: simulator internals (`model.py`), the evaluation harness, the Showdown server, and the opponent bots.
- **No hidden information.** Nothing the real game hides from our side, such as the opponent's unrevealed Pokémon, moves, or items.
- **No opponent-specific code.** Nothing that detects or special-cases a particular opponent bot.
- **Hard limits.**
  - Mean turn time at most 8 s and p95 at most 12 s (the game limit is 15 s).
  - Total memory under the 14 GB watchdog.
  - Zero API calls, no LLMs, no gradient boosting or logistic regression, no Foul Play code.
- **One job at a time.** Never run two evaluations concurrently.

## Choosing what to run (your call)

There is no fixed queue. You decide what to run next, one job at a time, based on what the results so far tell you. Two kinds of experiments are allowed.

1. **Strategy experiments.** Edit `strategy.py` and evaluate on the dev set, per the loop below. This is the main loop.
2. **Measurement experiments.** These are for the paper, and they fix a configuration and measure it. Candidate ideas, none mandatory:
   - depth 1 and depth 3 with Laya
   - more Abyssal games for the headline configuration
   - data scaling (retraining Laya on 1k, 3k, and 30k rows)
   - Laya without history
   - PokéChamp's OneStepPlayer vs Abyssal, to check that our protocol reproduces the published 44%
   - a uniform-opponent floor
   - an overconfident Laya (logits x3)
   - the smaller laya-multilingual model
   - a seed rerun
   - a full fine-tune

   Measurement runs against Abyssal are allowed, because they measure fixed configurations. Never use an Abyssal result to choose or tune a strategy; only dev results may drive keep/discard decisions.

Balance the two kinds of experiments by what most strengthens the abstract. Start with whatever you judge most informative. Log every job, whichever kind, in `notes.md`, including why you chose it.

## Evaluation (fixed budget, dev set only)

- **Dev opponent.** SimpleHeuristicsPlayer, gen8randombattle (no Dynamax), using the same format as the benchmark but a different opponent.
- **Screen.** 80 games, 4 concurrent battles, about 15 minutes, with per-battle logging.
- **Confirm.** If the screen beats the current best dev rate by at least 6 points, run 80 more games.
- **Keep** a change only if the pooled 160-game rate beats the best pooled rate by at least 3 points. Otherwise discard it and restore `strategy.py`.
- **Never use Abyssal inside the loop.** It is the held-out test.
- **Timeout.** If an evaluation exceeds 30 minutes, or p95 turn time exceeds 12 s, count it as a failure and discard.
- **Crashes.** Fix trivial bugs and rerun once. Otherwise log `crash` and move on.

After each experiment, append a row to `results.tsv` and save a snapshot as `snapshots/exp_NNN_strategy.py`. Keep a short `notes.md` with one line per experiment: the idea, the result, and what you learned.

## Simplicity criterion

All else equal, simpler wins. Drop gains of about 1 point that add complexity. Keep changes that simplify at equal performance.

## The loop

Loop until stopped:

1. Read `results.tsv` and `notes.md`. Choose the next idea, favoring those that attack the known bottlenecks: the shallow search, the HP-only leaf that cannot value setup moves or status, and switch handling.
2. Edit `strategy.py`.
3. Screen, confirm if warranted, and record the result.
4. Keep or discard.
5. Every 5 experiments, re-read the notes. Combine near-misses, and try at least one bolder idea.

NEVER STOP to ask the human. The human is asleep. If you run out of ideas, think harder: re-read `search.py` and `model.py` for new angles, and combine earlier partial wins.

## Morning deliverable

Stop starting new experiments when either of these comes first:

- 06:30 local time
- the night queue's remaining jobs need the machine

Then:

1. Run the best `strategy.py` against Abyssal (gen8, PokéChamp protocol, chunked runner) for 100 games as the single held-out test. Also rerun the baseline on the same 100-game protocol if time allows.
2. Write an "Autoresearch" section in `RESULT.md` covering the best configuration's diff against the baseline, the dev curve (best dev rate by experiment number), the Abyssal test result with its 95% CI, and the full `results.tsv`.
