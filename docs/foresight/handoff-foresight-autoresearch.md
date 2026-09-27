# Handoff: run the ForeSight autoresearch loop overnight

**For:** a fresh Claude Code session started in a new terminal.
**Goal:** run the autonomous research loop, following `autoresearch/program.md`, all night without stopping. At 06:30 local time, run the single held-out Abyssal test and write the morning report.

## 1. What this project is (short)

This is an AAAI-27 student-abstract project. The deadline is 2026-09-28 AoE.

**ForeSight.** A small, fine-tuned, non-generative "System One" decision model predicts the opponent's next action in Pokémon Showdown. The model is Laya, 421M parameters, open-source and similar to TypeSafe's Jev. A depth-2 expectimax search weights every imagined future by those forecasts and by chance, and scores the leaves by HP balance.

**The paper's thesis.** Calibrated small forecasters can bring engine-style lookahead to settings where engines can't be hand-built, such as negotiation.

**Read these instead of re-deriving anything:**

- `/Users/sickle/Coding/aaai-27/src/foresight/local_forecaster/autoresearch/program.md` holds the loop spec and the rules. It is authoritative; follow it.
- `/Users/sickle/Coding/aaai-27/src/foresight/local_forecaster/RESULT.md` holds all results so far, including the "Overnight batch" section.
- `/Users/sickle/Coding/aaai-27/docs/foresight/findings.md` holds the project history and the idea as the user defined it (sections 8 and 9).
- `/Users/sickle/Coding/aaai-27/paper/paper.tex` and `paper.pdf` are the current 2-page draft. Don't edit the paper tonight.

**Key numbers.** Win rates are against Abyssal, Gen 8 random battles without Dynamax, under PokéChamp's protocol:

| Forecaster | Win rate | Forecast accuracy on Abyssal's real moves |
| --- | --- | --- |
| Gen 9-trained Laya | 56.5% (113/200) | 73% |
| Gen 8-trained Laya | 56.5% (113/200) | 75% |

- Published comparisons: PokéChamp GPT-4o 70%, PokéChamp Llama-3.1-8B 64%, PokéLLMon 56%, One-Step 44%.
- Better forecasts did not raise the win rate. The working hypothesis is that the bottleneck is the shallow search and the HP-only leaf, not the forecaster. That is what the loop should attack.
- Known weak spots: the HP leaf can't value setup moves such as Swords Dance, status, or hazards. Laya never ranks "switch" first, although its P(switch) AUROC is 0.87. Search uses under a tenth of the 15 s turn budget.

## 2. Current machine state (as of about 21:00, 2026-09-26)

| Item | State |
| --- | --- |
| Showdown server on port 8001 | Running: `node pokemon-showdown start 8001 --no-security` (pid 46353 at handoff), from `local_forecaster/pokemon-showdown` |
| Memory watchdog | Running: `local_forecaster/scripts/mem_watchdog.sh` (pid 46053). Kills the largest experiment process if their total `phys_footprint` exceeds 14 GB. Log: `local_forecaster/logs/watchdog.log` |
| `autoresearch/strategy.py` | Copy of the headline decision logic. `autoresearch/check_equiv.py` verified identical choices and Q-values on 249/249 logged states |
| `autoresearch/eval.py` | Dev evaluation (see section 3) |
| `autoresearch/results.tsv` | Header only. The baseline (exp 000) has not run yet |
| `autoresearch/notes.md` | Empty template |
| Other jobs | None running. The old `night_queue.sh` is stopped; don't restart it |

If the server or watchdog died, restart them from `local_forecaster/`:

```bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
nohup node pokemon-showdown/pokemon-showdown start 8001 --no-security > logs/showdown.log 2>&1 &
nohup ./scripts/mem_watchdog.sh > /dev/null 2>&1 &
```

## 3. Commands

All commands run from `/Users/sickle/Coding/aaai-27/src/foresight/local_forecaster`.

**Dev screen or confirm.** This plays the current `strategy.py` vs SimpleHeuristicsPlayer, Gen 8 without Dynamax, 4 concurrent battles, with a 30-minute timeout. It prints one JSON line with wins, n, rate, mean and p95 turn seconds, errors, and timeout. Counts pool across calls that share an EXP_ID.

```bash
.venv/bin/python autoresearch/eval.py 000 80      # screen (80 games, ~15 min)
.venv/bin/python autoresearch/eval.py 000 80      # confirm: same EXP_ID pools to 160
```

**Equivalence check** after refactors that should not change behavior:

```bash
.venv/bin/python autoresearch/check_equiv.py
```

**Held-out Abyssal test.** Run this only at 06:30 and only on the final best strategy, never inside the loop. It uses the chunked runner, because PokéChamp's Abyssal code leaks memory; never run two Abyssal runners at once.

```bash
STRATEGY=$PWD/autoresearch/strategy.py FMT=gen8randombattle CHUNK=20 \
  .venv/bin/python run_arm.py ar_best_aby laya_hist:laya_opp_hist_gen8randombattle.pt hp 2 ABYSSAL 100 4 \
  > logs/ar_best_aby.log 2>&1
```

For the baseline on the same protocol, run the same command without `STRATEGY` and with arm name `ar_base_aby`. Per-battle results land in `data/battles_log.jsonl`, keyed by the arm name.

## 4. Rules and user preferences (hard)

- **Never stop to ask.** The user is asleep. Loop until 06:30, then run the held-out test and report.
- **One job at a time.** Never run two evaluations or training jobs concurrently. The user was unhappy when the CPU hit 100% and 96 °C. Keep the default 4 concurrent battles inside a single eval; don't raise it.
- **Memory under 14 GB.** Check with `footprint <pid>` (`phys_footprint`), not RSS, which undercounts MPS/GPU memory.
- **Zero API spend.**
- **Banned tonight:**
  - Qwen or any LLM
  - gradient boosting or logistic regression comparisons
  - anything Foul Play (its partial port hurt play badly and the user set it aside)
- **No tuning on Abyssal.** Keep/discard decisions use dev results only. Abyssal is the held-out test.
- **No hidden information and no opponent-specific code.** Nothing the real game hides from our side.
- **Turn-time limits.** Mean turn time at most 8 s and p95 at most 12 s (the game limit is 15 s).
- **Don't move folders.** The Abyssal runner depends on `../pokemon_search/.venv-pc`.
- **Don't edit the paper, commit to git, or finalize anything.** The user decides those.
- **The bar is proof of concept.** The user wants evidence the idea "somewhat works," reported honestly. Don't overclaim: report confidence intervals, and call within-noise differences within noise.

## 5. Known pitfalls

- **Abyssal runner memory leak.** Use `CHUNK=20` and one runner at a time. The chunked runner restarts the process every 20 games.
- **MPS out-of-memory in training.** Fixed with batch size 4 and `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 0.3 to 0.35. The loop shouldn't need training anyway.
- **Finding processes.** Job processes show up as `.venv/bin/python run_arm.py ...`, so find them with `pgrep -f run_arm.py`, not by matching on `python`.
- **Pooled counts.** `eval.py` counts all battles in `data/battles_log.jsonl` for arm `ar_<EXP_ID>`. Reuse an EXP_ID only for a screen-plus-confirm pair; use a fresh ID for every new idea.
- **Noise.** 80 games give about ±11 points. The program.md thresholds (screen at least +6, pooled at least +3 over best) exist for this; don't loosen them.
- **Snapshots.** After each experiment, snapshot `strategy.py` to `autoresearch/snapshots/exp_NNN_strategy.py`. On discard, restore the best snapshot.

## 6. Morning deliverable

1. Run the best `strategy.py` vs Abyssal for 100 games. If time allows, rerun the baseline on the same protocol.
2. Add an "Autoresearch" section to `local_forecaster/RESULT.md` covering:
   - the diff of the best strategy vs the baseline
   - the dev curve (best dev rate by experiment)
   - the Abyssal test result with its 95% CI
   - the full `results.tsv`
3. Write a short plain-English summary for the user. Lead with whether anything beat the baseline on dev, and by how much on the held-out Abyssal test.

## 7. Suggested skills

- **`loop`**, in self-paced mode. Invoke it with the loop prompt so the session keeps waking itself through the night instead of ending its turn. Pass the prompt in section 8 as the argument.
- **`superpowers:systematic-debugging`**, when an eval crashes or a strategy change breaks the runner. Read the log tail before guessing.
- **`superpowers:verification-before-completion`**, before writing the morning report. Re-derive every number from `results.tsv` and `data/battles_log.jsonl` rather than from memory.

## 8. Suggested kickoff prompt for the new terminal

```text
Read /Users/sickle/Coding/aaai-27/docs/foresight/handoff-foresight-autoresearch.md and then
/Users/sickle/Coding/aaai-27/src/foresight/local_forecaster/autoresearch/program.md.
Verify the Showdown server (port 8001) and the memory watchdog are running, then start the
autoresearch loop with the baseline dev screen (exp 000) and keep going all night per program.md.
Never stop to ask. At 06:30 run the held-out Abyssal test and write the morning report.
```
