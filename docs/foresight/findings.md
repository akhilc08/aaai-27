# Jev Context Research: Ideas, Pilots, and Findings

As of 2026-09-26. Goal: an AAAI-27 student abstract (deadline 2026-09-28 AoE) built on TypeSafe's Jev: a new method on an existing benchmark that beats a baseline.

## 1. What Jev is and how to call it

- **Jev** is TypeSafe AI's "System One" model, released 2026-09-15. It does not generate text. You send a state plus typed questions, and it returns calibrated answers in one pass.
- **Question types:** `noul` (yes/no, returns P(yes)), `choice` (pick from options you define, returns a probability per option), `score` (position on an ordered scale).
- **Access:** OpenRouter, `POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`. The normal chat endpoint rejects it. The existing OpenRouter key works.
- **Measured here:** about 0.25 s per call. Cost is similar to a cheap LLM (qwen3-30b), not dramatically lower. Speed is the real advantage, 4 to 10x.
- **Cannot be fine-tuned.** The workaround used in every pilot: ask several narrow questions and train a small logistic regression "head" on the answers.
- **Laya** (github.com/NandhaKishorM/laya) is an open-source Jev clone (421M ModernBERT) that can be fine-tuned. It installs with pip and runs on the M5 Mac at about 60 ms per call. It scores near chance until fine-tuned. Not used in the pilots.

Example call (see `experiments/pilots/jevlib.py`):

```python
import jevlib as J
J.jev("You are in a kitchen. The fridge is closed. Action: open fridge.",
      {"opens": {"type": "noul", "instructions": "After the action, is the fridge open?"}},
      tag="demo")
```

## 2. Ideas and the user's reactions

| Reaction | Ideas |
| --- | --- |
| Loved | Predicting agent and compaction failures; future prediction with a trained model |
| Liked | Imagination planning; swarms without LLMs; Pokémon; poker; surprise-triggered replanning |
| Okay | Negotiation; knows-when-to-ask; fast/slow model routing |
| Lukewarm | Rehearse-then-commit and stop gate on tau-bench |
| Rejected | Chess; classification by question splitting; tool-call injection guard |
| Not yet discussed | Speculative agent steps; value function for text agents; compaction that predicts future relevance; predicting whether a big model will get it right; anticipating the user's next request |

**Clarification from the user after the pilots:** the Pokémon and poker ideas were meant as Jev *playing to win* using lookahead, not predicting a human's next move. The pilots tested prediction, so they do not answer that question (see section 5).

## 3. Pilot results

Each pilot has full details in `RESULT.md` in each pilot folder. Total API spend: $12.71.

| Pilot | Verdict | One-line finding |
| --- | --- | --- |
| imagination_planning | **Promising** | Jev picking among the LLM's top 3 actions in ALFWorld: 34.6% to 48.4% success |
| compaction_failure | Mixed | Strong single-rule violation predictor, but no use as a re-insertion gate; nothing works with 20 rules |
| agent_failure | Mixed | Early-run prediction is mostly issue difficulty, not run progress |
| pre_run | Mixed | Jev predicts SWE-bench issue difficulty better than human ratings; routing gain is only ~2 pts |
| poker | Mixed | Jev 61% vs cheap LLM 58.5% on PokerBench; below GPT-4's 65.5%; head adds little over card features |
| pokemon | Mixed | Zero-shot worse than chance on switch prediction; head beats LLM but barely beats simple features |
| swarm | Not promising | No positive herding; negative herding is uncorrected, unlike humans; not cheaper than LLM |

### 3a. Imagination planning (the winner)

**Setup.** ALFWorld text household, all 134 valid_unseen games, 30-step cap. A cheap LLM proposes its top 3 admissible actions. Jev answers yes/no questions per candidate (will something happen, will the target appear, is the place empty). The agent takes Jev's top pick.

| Agent | Success |
| --- | --- |
| qwen3-30b alone (top-1 of its own top 3) | 34.6% |
| qwen3-30b scoring its own options with the same questions | 41.2% |
| Simple rule: skip repeated or failed actions | 39.6% |
| Jev alone over all admissible actions, no LLM | 40.3% |
| **qwen3-30b + Jev picking** | **48.4%** |
| qwen3-235b alone | 49.3% |
| qwen3-235b + Jev picking | 54.5% |

- **Significance:** 30b + Jev vs 30b alone: +12.9 pts, game-level 95% CI [+6.7, +19.2], Wilcoxon p = .0001. Jev also beats the LLM-as-scorer control (p = .019) and the simple rule (p = .023).
- **Stronger model:** +5.2 pts, CI [-0.7, +11.2], p = .10. Positive in all 3 seeds but not significant.
- **Why it works:** Jev overrides the LLM on about 40% of steps. Its chosen location held the target 272 times vs 94 for the LLM's pick. It also avoids revisits.
- **What didn't work:** surprise-triggered replanning added nothing. Despite the name, only one step of lookahead was used.
- **Cost:** under $0.005 per game.

### 3b. Compaction failure

- **Round 1:** predicting a turn-1 rule violation from the compacted context. AUROC: keyword 0.747, LLM self-judgment 0.519, Jev zero-shot 0.830, Jev 6-question head 0.883 (0.869 leave-one-scenario-out). Holds with reworded rules. Stronger LLM judges (gpt-4.1-mini) scored 0.54 to 0.58.
- **Round 2:** re-inserting one short rule every time costs 35 tokens and drops violations from 59% to 0.6%, so gating is pointless.
- **Round 3:** 20-rule policies. No gate predicts which rule is violated (AUROC ~0.5 for all). A random gate works as well as any.

### 3c. Other pilots

- **agent_failure:** nebius/SWE-agent-trajectories, llama-70b runs. Jev 8-question head AUROC 0.84 at step 5, but only 0.54 when comparing a pass and a fail on the same issue.
- **pre_run:** SWE-bench Verified, 13 public systems. Jev 10-question head AUROC 0.78 for "solved by half the systems", vs human difficulty 0.71 and length 0.55. Weaker on frontier agents.
- **poker:** PokerBench, 1,000 test spots. A head trained on the proper train split scores only 62%.
- **pokemon:** 160 rated gen9randombattle replays. Which-move prediction: Jev + head 50%, repeat-last-move 46%, uniform 34%.
- **swarm:** Muchnik et al. herding setup with 150 personas and 60 comments.

## 4. Recommendation for the abstract

Write up the ALFWorld result as a method: a cheap calibrated System One model reranks an LLM agent's candidate actions. It fits the "new method beats baselines on a public benchmark" archetype.

**Gaps to close before submitting:**
- A verified published ReAct baseline on ALFWorld (not yet checked).
- Honest reporting of the smaller, non-significant gain with the stronger model.
- A few-shot ReAct baseline, since ours is zero-shot.

## 5. Untested: Jev as a game player with lookahead

Jev cannot look ahead by itself; each call judges one situation. Lookahead needs a search: a simulator generates future positions, Jev scores each one, and the search picks the move leading to the best future. This is PokéChamp's design with a slow LLM as the scorer. Jev's speed would let the same time budget cover far more positions.

**Proposed tests:**
- **Pokémon:** a local Showdown server, a 2-3 move search with Jev scoring positions, a few hundred battles against the standard heuristic bots. Compare to LLM scoring and to no search. Metric: win rate. About half a day of setup.
- **Poker:** a Jev-scoring bot in a heads-up engine against baseline bots. Metric: chips per hand.

**Warning sign:** Jev's zero-shot judgments of Pokémon positions were poor in the prediction pilot. Scoring "is this position good for me" is a different question and was not tested.

## 6. Files

Paths are relative to the `aaai-27` repo.

- `experiments/pilots/jevlib.py`: shared helper for Jev and cheap-LLM calls, with retries, spend logging, and a hard spend cap (`GLOBAL_CAP`).
- `experiments/pilots/spend.jsonl`: every API call's cost.
- `experiments/pilots/<name>/`: scripts, raw outputs (`*.jsonl`), analysis files, and `RESULT.md` for the forecasting pilots (imagination_planning, pokemon, poker).
- `src/foresight/`: the Pokémon lookahead code (`pokemon_search`, `local_forecaster`).
- `docs/aaai27-student-abstract-research.md`: AAAI-27 student abstract track rules and what gets accepted.
- Elsewhere: `compaction_failure` lives in the `jev-context-research` repo; `agent_failure`, `pre_run` and `swarm` live in `aaai-27-ideas/experiments/jev-pilots/`.

**API key:** `jevlib.py` reads the OpenRouter key from `/Users/sickle/Coding/context-research/.env`. No key is stored in this folder.

## 7. Novelty check: calibrated forecasting for deep lookahead (2026-09-26)

Web search only; not exhaustive. Verdict: the parts are known, the combination and the calibration test look open.

**Already done (must cite):**
- **LLM as probabilistic world model inside search:** LLM-MCTS (Zhao et al., NeurIPS 2023, arXiv 2305.14078) uses LLM commonsense priors over object locations inside MCTS for household tasks. This directly overlaps the ALFWorld multi-step plan.
- **LLM lookahead in Pokémon:** PokéChamp (arXiv 2503.04094) uses minimax, worst case over an unweighted opponent set.
- **Fast learned evaluator in game search:** DeepMind MAV (arXiv 2412.12119), AlphaZero-style value nets.
- **Opponent-model quality in MCTS:** Pommerman study (arXiv 2305.13206) shows accurate opponent models help and inaccurate ones hurt. It studies accuracy, not calibration.
- **Calibrated agent world models (2026):** Ask the World Before Acting (arXiv 2606.31422), Belief-Calibrated Optimization (arXiv 2609.01861), WorldEvolver (arXiv 2606.30639, filters low-confidence foresight on ALFWorld/ScienceWorld).
- **Jev inside agents:** pentest decision layer (arXiv 2609.28940), Jev-Mem memory control (arXiv 2609.23986), and the system1-agents repo (github.com/ThinkFlowLab/system1-agents), which runs Jev as the direct policy on ALFWorld and games (2048, Blackjack) and reports speed and cost at similar scores. None use Jev inside a lookahead search.

**Not found (the open part):**
- A calibrated non-generative model as the probability source and leaf evaluator for deep expectimax search.
- A controlled test of forecaster calibration x search depth x time budget. Sharpest hypothesis: errors from an overconfident forecaster compound with depth, so deeper search helps a calibrated forecaster and hurts a miscalibrated one.
- The same test across several domains.

**Risks:** reviewers may call it "a new model in an old recipe" unless the calibration x depth result is shown. The Jev literature is growing weekly (two arXiv papers this month), so the window is short.

## 8. The idea, as clarified by the user (2026-09-26)

- **Engine-style lookahead.** Choose each move by building full decision trees several turns deep: our moves, the opponent's likely replies, chance outcomes, and one tree per plausible version of hidden information. Pick the move with the best probability-weighted outcome.
- **Jev supplies the probabilities.** Opponent moves, hidden-world likelihoods, and leaf values. It is fast (bigger trees) and meant to be calibrated (trustworthy weights).
- **Framing: future prediction.** The paper is about forecasting how a situation unfolds. Target settings are ones where hand-built deterministic engines are hard, or where there is so much information that a transformer processes it better.
- **Goal.** Beat existing LLM agents (PokéChamp, PokéLLMon). Getting close to hand-built engines (Foul Play) is a bonus.
- **Leaf evaluation.**
  - **A:** Jev scores the position.
  - **B:** Jev forecasts P(win) from the leaf. **This is the chosen option.**
  - **C:** roll out to the end with Jev predicting every move. Keep in mind; test its token and latency cost.
- **Scope.** The abstract covers Pokémon only; other domains go in future work. In domains without a simulator, the likely design is that an LLM imagines possible next states and Jev weighs them. Candidate future domains: Diplomacy with negotiation, Avalon/Werewolf, negotiation.

## 9. Pokémon lookahead pilot result (2026-09-26)

Full details: `src/foresight/pokemon_search/RESULT.md`. Local Showdown server; real AbyssalPlayer from the PokéChamp repo, gen8 random battle, dynamax off. Expectimax over an approximate simulator, pruned to 90% probability mass, one batched Jev call per tree level. Hidden-team sampling was not built.

| Arm | vs SimpleHeuristics (gen9) | vs Abyssal (gen8) |
| --- | --- | --- |
| Jev, no search | 43% | 27% |
| Jev P(win) or score at leaves, depth 1 / 2 | 19-21% / 0-12% | not run |
| Jev opponent probabilities + HP-balance leaf, depth 1 / 2 / 3 | 58 / 66 / 64% | depth 2: 57% (n=300, CI 51-62) |
| Same, minimax at depth 2 | 59% | not run |
| Same search, free heuristic opponent model, depth 2 | 62% (68% minimax) | 61% (n=300, CI 55-66) |

Published vs Abyssal: PokéChamp GPT-4o 70%, Llama-3.1-8B 64%, PokéLLMon 56%.

- **Jev as leaf evaluator fails.** Its scores barely respond to simulated HP changes, and deeper search makes it worse.
- **The search works, Jev adds nothing.** Jev's opponent probabilities tie a free damage-based opponent model. Depth 2 beats depth 1; depth 3 is flat. Probability weighting does not beat minimax.
- **P(win) forecasting is decent late, weak early.** 23.7k turns: Brier 0.209 vs base rate 0.250, AUROC 0.74 overall, 0.56 in turns 1-5. Underconfident at the extremes.
- **Rollout leaves (C) are too slow:** 11 s and $0.043 per decision vs 0.86 s and $0.0003 for P(win).
- **Latency:** depth 2 takes 0.6 s per turn; nothing exceeded the 15 s limit.

Verdict: not promising for "Jev as a search evaluator wins games"; mixed for Jev's P(win) forecasting.

### 9b. Formatted-state round and knowledge probe

- **Knowledge probe** (`src/foresight/pokemon_search/probe_knowledge.py`): Jev scored 16/20 on a type-matchup quiz (immunities 6/6), so the failure is not pure lack of Pokémon knowledge. With full Pokémon text, a crushing lead raised P(win) only 0.52 to 0.66; with names stripped, 0.89.
- **Formatter** (`fmt.py`): positions rendered as precomputed facts (HP and units, who moves first, best-attack damage and KO, one-on-one race, safe switches). Offline P(win): orig 0.52, ahead 0.87, behind 0.13. The "are we ahead?" question alone gives 0.98 / 0.02.
- **Battles:** depth-2 expectimax with the formatted Jev P(win) leaf won 9/60 = 15% (CI 8-25) vs SimpleHeuristics, against 66% for the HP-balance leaf.
- **Real positions** (1,905 turns, base rate 0.17): every Jev forecast variant has worse Brier than the base rate (P(win) 0.197, "ahead" 0.163, base 0.143). AUROC 0.76-0.79.
- **Likely mechanism (untested):** the search maximizes over noisy leaf values, so it steers into positions Jev overestimates. The ranking is too coarse to separate sibling moves that differ by a few HP points.
- **Untested cheap follow-up:** HP-balance leaf plus a small Jev P(win) correction, about 30 battles.
- **Hybrid leaf result:** HP balance + 0.25 × (formatted Jev P(win) − 0.5) won 20/40 = 50% (CI 35-65) vs SimpleHeuristics, against 66% for the pure HP leaf. The Jev correction changed the chosen move in 30% of decisions, so its sibling-to-sibling noise is as large as the HP differences. $0.026 per battle.
