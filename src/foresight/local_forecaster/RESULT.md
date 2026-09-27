# Laya (fine-tuned System One model) as the opponent forecaster inside Pokémon expectimax

Status: DONE (2026-09-26). The Showdown server on port 8001 is stopped. Zero API spend.

## Question
Can a fast, calibrated, non-generative System One model, fine-tuned on Pokémon outcomes, act as the forecaster inside an engine-style lookahead search? Here it predicts the opponent's next action and supplies the opponent-node probabilities of a depth-2 expectimax. How does the resulting agent compare with published LLM agents under PokéChamp's Abyssal protocol?

## Setup
- **Search.** Copied from `pilots/pokemon_search`: an approximate simulator and a depth-2 expectimax. Opponent nodes keep 90% probability mass (at most 3 replies at the root, 2 deeper), chance nodes are pruned, and the leaf is the HP balance. Laya supplies the opponent-node probabilities.
- **Opponent and protocol.** PokéChamp's real AbyssalPlayer in gen8randombattle with dynamax off, on our own local Showdown server. The Abyssal process is restarted every 20 battles because it leaks memory. We also played SimpleHeuristicsPlayer in gen9randombattle.
- **Training data (free).** Bot-vs-bot battles on the local server: 3,150 gen9 and 1,000 gen8 (SimpleHeuristics, MaxBasePower, Random, and an HP-leaf search bot). Every decision of both sides is logged. The label is the opponent's actual next action, mapped onto the candidate list the search uses (revealed moves plus placeholder "unrevealed STAB" moves, and switches to seen bench units). About 5% of turns (switches to unseen units) have no candidate and are dropped. Splits are by battle.
- **Opponent history.** Parsed live from the battle log, with the same code at train and play time. It covers the opponent's last 5 actions (move or switch, whether it was threatened with a KO, whether the move was its highest-damage option, status moves, repeats) and our last 3 action kinds, rendered as text.
- **Laya.** convaiinnovations/laya (421M, ModernBERT-large). **Partial fine-tune**, not the full model:
  - Top 4 of 28 encoder layers plus the typed decision head, 75M trainable parameters.
  - Laya's RLCD proper-scoring loss (log + spherical score), AdamW, bf16 autocast on MPS.
  - 10,000 training rows, one epoch (33-35 min). Temperature fitted on validation.
  - Input: precomputed position facts from `fmt.py`, compact rosters and the history text, asked as a typed `choice` question over the labelled candidates.
  - Two models: one trained on gen9 data and one on gen8 data.
  - Peak memory was about 9-10 GB in training and about 4 GB per battle process at inference.
- **Not run.** Full fine-tune, frozen-encoder head, laya-multilingual, raw-log inputs, Qwen/MLX LLM baselines and Kev. Kev (jaredpalmer/kev-*) is open-weight, Qwen-based LoRA adapters. These were dropped when the user restricted the scope to Laya plus published numbers.

## 1. Laya opponent-action forecasts (offline, held-out battles)

| Laya model | eval set | n | top-1 acc | NLL | ECE (top-1) |
|---|---|---|---|---|---|
| zero-shot (shipped temperature) | gen9 bot test | 2,000 | .434 | 1.148 | .026 |
| fine-tuned + history, gen9 | gen9 bot test | 4,000 | .718 | .702 | .015 |
| fine-tuned + history, gen8 | gen8 bot test | 3,754 | .796 | .542 | .018 |
| fine-tuned + history, gen9 | **Abyssal's actual moves** (100 logged battles) | 1,963 | .730 | .622 | .041 |
| fine-tuned + history, gen8 | **Abyssal's actual moves** (100 logged battles) | 1,963 | .754 | .582 | .027 |
| fine-tuned + history, gen9 | Abyssal's actual moves (68 other battles) | 1,282 | .753 | .581 | .024 |
| fine-tuned + history, gen8 | Abyssal's actual moves (68 other battles) | 1,282 | .775 | .540 | .034 |

- There are about 3.3 candidate actions per turn (uniform accuracy .33).
- Fine-tuning moves Laya from near chance to 72-80% top-1 with good calibration. It transfers to the real Abyssal bot at 73-78%.
- By opponent type (gen9 model): MaxBasePower .93, SimpleHeuristics .82, HP-leaf search bot .58, Random .41.
- Caveat: candidates are listed with revealed moves in order of first use. Part of the accuracy on repetitive bots may come from that ordering.
- Zero-shot Laya as a P(win) forecaster is worse than the base rate (Brier .377 vs .250, AUROC .59). This matches what zero-shot Jev showed earlier: without domain training, System One forecasts are not usable here.

## 2. Win rates: Laya-driven depth-2 expectimax vs published agents (95% Wilson CIs)

| agent | vs Abyssal, gen8 (PokéChamp protocol) | vs SimpleHeuristics, gen9 |
|---|---|---|
| **Laya (gen8-trained) opp model + HP leaf, depth 2** | **60.0% [50.2-69.1]** (n=100, subset) | — |
| **Laya (gen9-trained) opp model + HP leaf, depth 2** | **56.5% [49.6-63.2]** (n=200) | **60.5% [53.6-67.0]** (n=200) |
| Laya (gen9-trained) opp model + Foul Play leaf, depth 2 | 38.2% [27.6-50.1] (n=68, stopped early) | — |
| *published:* PokéChamp (GPT-4o) | 70% | — |
| *published:* PokéChamp (Llama-3.1-8B) | 64% | — |
| *published:* PokéLLMon | 56% | — |

- **gen9-trained Laya vs Abyssal:** the first 100 battles went 64-36 and the second 100 went 49-51.
- **gen8-trained Laya vs Abyssal (100 battles):** it is 2.4 points more accurate on Abyssal's moves than the gen9 model and won 3.5 points more. The direction is consistent, but the difference is within noise.
- **Foul Play leaf.** This is a Python port of poke-engine `src/genx/evaluate.rs` (MIT, attribution in `evals.py`); Foul Play's own GPL code is not used.
  - Kept: alive +30, 100·HP fraction, status penalties (burn uses the physical-move rule), active boost terms, and the per-Pokémon floor.
  - Dropped, because model.py doesn't track them: items, abilities, hazards, volatiles (substitute, confusion, leech seed), side conditions, tera.
  - It was mapped linearly so that one healthy Pokémon = 1/3 unit on the search's [0,4] scale.
  - It did clearly worse than the HP leaf. Likely reason: without hazards and volatiles, and with the boost weights on top of our approximate simulator, the leaf rewards simulated stat-boosting and status lines. The user stopped the arm at 68 battles.
- **Reference.** Our search with its hand-built damage-softmax opponent model (no Laya) scored 56.7% vs Abyssal (n=300) and 62.2% vs SH (n=400), with the same search and HP leaf. That is listed only as an internal reference.

## 3. Latency and search size (Apple M5, MPS)

| measurement | value |
|---|---|
| Laya opponent forecast, single process, batch 1 / 4 / 16 / 64 | 84 / 45 / 39 / 36 ms per forecast |
| forecasts that fit in a 15 s turn (batch 1 → 64) | ~180 → ~410 |
| equivalent tree nodes per 15 s (≈4.8 nodes per opponent forecast in this search) | ~850 → ~1,950 |
| used by the depth-2 search per turn | ~15 forecasts, ~73 nodes |
| turn time in play, 4 concurrent battles sharing the GPU (gen9 model vs Abyssal / vs SH / gen8 model vs Abyssal) | mean 2.7 / 3.1 / 4.1 s; p95 4.5 / 5.7 / 7.1 s; max 6.1 / 9.9 / 11.0 s |
| turns over the 15 s limit | 0 |
| errors | 0 in the Abyssal runs; 3 of about 5,500 decisions vs SH (a poke-env move-data KeyError, which falls back to a random move) |

A first Laya launch had a logging bug (float32 in JSON) that turned every move random. It was discarded and rerun, and none of those battles are counted.

## Verdict: MIXED
- **Forecasting: PROMISING.** Fine-tuned on 10k free bot-vs-bot positions, Laya predicts the real Abyssal bot's next action 73-78% of the time with low calibration error (ECE .02-.04). It runs at 36-84 ms per forecast, so hundreds of forecasts fit in a turn.
- **Playing strength: comparable, not better.** The Laya-driven depth-2 search reaches 56.5% (gen9 model, n=200) and 60.0% (gen8 model, n=100) vs Abyssal. That is on par with PokéLLMon (56%), within the CI of PokéChamp with Llama-3.1-8B (64%), and below PokéChamp with GPT-4o (70%).
- Better opponent forecasts did not measurably raise win rates at these sample sizes. Laya's win rates tie the same search driven by a hand-built heuristic opponent model.

## What the abstract could honestly claim
1. A 421M System One model fine-tuned on free self-generated bot battles becomes an accurate, calibrated opponent forecaster: 72-80% top-1 on held-out bots and 73-78% on PokéChamp's Abyssal bot, vs 43% zero-shot. It takes tens of milliseconds per forecast, with no API cost.
2. Used as the opponent model in a depth-2 expectimax, it yields an LLM-free agent at 56-60% vs Abyssal under PokéChamp's protocol. That is comparable to PokéLLMon (56%) and PokéChamp-Llama-8B (64%, within CI), using a laptop and roughly 3 s per turn. Frame it as "comparable at far lower cost", **not** "better". It is below PokéChamp-GPT-4o (70%).
3. Zero-shot System One forecasts are miscalibrated for this domain (P(win) Brier .38 vs .25 base rate). Domain fine-tuning is what makes them usable.
4. Do not claim that better forecasts translate into more wins: that link is not established here (n=100-200, CIs ±7-10 points). Do not claim the Foul Play evaluator helps: in our approximate simulator it hurt.

## Appendix: earlier non-Laya runs (context only; not part of the Laya comparison)
Before the user narrowed the scope, feature-based models were run as sanity baselines. Details are in `data/arm_results.jsonl`, `data/base_gen*.json` and `data/summary_gen9randombattle.json` (`report.py`, `collect.py`).
- **Trained P(win) leaves.** Logistic/GBM on 28 hand features are calibrated (ECE ≈ .02-.03) but rank positions barely better than the raw HP balance (AUROC .85 vs .85).
- **Leaf vs depth, and overconfident vs calibrated leaf.** Leaf and depth ablations vs SH (n=200 each) gave non-monotone depth curves. The deliberately overconfident leaf (logit×3) was 10 points worse than the calibrated one at depth 1, equal at depth 2 and 6 points worse at depth 3. None of these differences are significant.
- **Excluded latencies.** The depth-3 arms were paused by SIGSTOP for thermal reasons, so their latency figures are invalid.

## Files
- **Code:**
  - `gen_data.py` (logged bot battles), `build_ds.py` (datasets and history), `hist_live.py` (live history), `slim.py`.
  - `train_laya.py` (partial fine-tune, RLCD loss), `laya_util.py`.
  - `evals.py` (search evaluators; ported poke-engine evaluator `FPLeaf`), `bots.py`, `run_arm.py` (arms; chunked Abyssal with a memory watchdog).
  - `score_turns.py` (scoring on Abyssal's real moves), `scripts/bench_laya.py` (latency), `collect.py`, `report.py`.
- **Results:**
  - `data/arm_results.jsonl`, `data/battles_log.jsonl` (per-battle outcomes), `data/dec_*.jsonl` (per-decision latency and tree size).
  - `data/score_turns_*.json`, `data/summary_gen*.json`, `logs/bench_laya.log`.
- **Checkpoints:** `ckpt/laya_opp_hist_gen9randombattle.pt`, `ckpt/laya_opp_hist_gen8randombattle.pt` (trainable-parameter overlays).

<!-- NIGHT:START -->
## Overnight batch (2026-09-26/27; one job at a time; auto-updated by `night_summary.py`)

All runs use the gen8-trained Laya opponent model unless stated. Settings: depth-2 expectimax, HP leaf, vs PokéChamp's AbyssalPlayer, gen8, dynamax off, 20-battle chunks. Win-rate CIs are 95% Wilson. "partial" means the arm is still running or was stopped; those counts come from the per-battle log.

### Battles
| item | agent (all vs Abyssal, gen8, PokéChamp protocol) | n | win % | 95% CI | s/turn mean | s/turn max | nodes/turn | forecasts/turn |
|---|---|---|---|---|---|---|---|---|
| B2 | gen8 Laya opp + HP leaf, depth 2 (headline; first 100 from the day run) | 200 | 56.5 | 50-63 | 4.07 | 11.0 | 69 | 15 |
| B1 | same, depth 1 | 0 | — | — | — | — | — | — |
| B1 | same, depth 3 | 0 | — | — | — | — | — | — |
| B6 | PokéChamp's own OneStepPlayer vs Abyssal (protocol check; published 44%) | 0 | — | — | — | — | — | — |
| B5 | uniform opponent model (forecaster ablation), depth 2 | 0 | — | — | — | — | — | — |
| B4 | overconfident Laya (logits x3), depth 2 | 0 | — | — | — | — | — | — |
| B4 | overconfident Laya (logits x3), depth 1 | 0 | — | — | — | — | — | — |
| B4 | overconfident Laya (logits x3), depth 3 | 0 | — | — | — | — | — | — |
| B3 | wider depth-2 search (opp mass .99, caps 4/3, chance .95x5/.85x3) | 0 | — | — | — | — | — | — |
| L1 | leaf = HP balance + 0.3 per Pokémon alive, depth 2 | 0 | — | — | — | — | — | — |
| L2 | leaf = fine-tuned Laya P(win), depth 2 | 0 | — | — | — | — | — | — |
| — | *published:* PokéChamp GPT-4o / Llama-3.1-8B / PokéLLMon / One-Step Lookahead | — | 70 / 64 / 56 / 44 | — | — | — | — | — |

### Forecasting (gen8; same recipe as the day run: top 4 layers + head, RLCD loss, batch 4, temperature fitted on validation)
| item | gen8 Laya opponent model | gen8 bot test: acc / NLL / ECE (n) | Abyssal real moves: acc / NLL / ECE (n=1,963) | train time |
|---|---|---|---|---|
| - | gen8 Laya + history, 10k rows (reference) | 0.796 / 0.542 / 0.018 (3754) | 0.754 / 0.582 / 0.027 | — |
| G1 | 1k training rows | not run / failed | — | — |
| G1 | 3k training rows | not run / failed | — | — |
| G1 | 30k training rows | not run / failed | — | — |
| G2 | no history, 10k rows | not run / failed | — | — |
| G3 | laya-multilingual 322M, 10k rows | not run / failed | — | — |
| G4 | seed 1, 10k rows | not run / failed | — | — |
| G5 | full fine-tune (all layers), 10k rows | not run / failed | — | — |

L2 P(win) model: not trained yet.

Latency, single process, ms per opponent forecast:
- laya 421M (gen9 ckpt, day run): batch 1: 84.3 ms, batch 4: 45.2 ms, batch 16: 39.3 ms, batch 32: 37.5 ms, batch 64: 36.3 ms

### Item 0: error analysis of the gen8 Laya opponent model (figures in `figures/`)
- **held-out gen8 bot battles** (n=3754): top-1 0.796, top-2 0.941, top-3 0.982 (uniform top-1 0.340). By turn: turn 1-5 0.89 (n=689), turn 6-15 0.81 (n=1628), turn 16-30 0.75 (n=1076), turn 31+ 0.71 (n=361). By # options: 2 options 0.93 vs chance 0.50, 3 options 0.84 vs chance 0.33, 4 options 0.72 vs chance 0.25, 5-6 options 0.57 vs chance 0.19, 7+ options 0.54 vs chance 0.14. Actual attacks 0.82 (n=3634), actual switches 0.00 (n=120); P(switch) Brier 0.0284, AUROC 0.8747.
- **Abyssal's real moves** (n=1963): top-1 0.754, top-2 0.935, top-3 0.992 (uniform top-1 0.344). By turn: turn 1-5 0.81 (n=345), turn 6-15 0.75 (n=765), turn 16-30 0.76 (n=614), turn 31+ 0.69 (n=239). By # options: 2 options 0.86 vs chance 0.50, 3 options 0.75 vs chance 0.33, 4 options 0.62 vs chance 0.25, 5-6 options 0.76 vs chance 0.20. Actual attacks 0.76 (n=1953), actual switches 0.00 (n=10); P(switch) Brier 0.0056, AUROC 0.8571.
- Laya almost never ranks "switch" first. Actual switches are rare among scorable turns (most switches go to unseen Pokémon and are unscorable), and top-1 accuracy on them is 0. The P(switch) probabilities are still informative (AUROC above).
- Figures: `figures/reliability_laya_gen8.png` (top-1 confidence and P(switch) reliability), `figures/accuracy_breakdown_laya_gen8.png`.

### Queue log (tail)
    20:14 queue start
    20:14 start B2 layah8_hp_d2 n=100 
    20:38 queue stopped by user request; B2 (running) will be recorded when it finishes; no further jobs
    21:19 end B2 layah8_hp_d2 :: 53/100 this batch; pooled with the day run 113/200
<!-- NIGHT:END -->
