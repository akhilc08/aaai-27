# Pilot: Jev as a cheap world model for imagination planning + surprise replanning (ALFWorld)

**Question.** Can Jev predict text-env action outcomes before they happen (Stage 1), and does Jev-scored action selection + surprise-triggered replanning help a cheap LLM agent (Stage 2)?

**Setup.** ALFWorld 0.4 TextWorld env, valid_unseen (= eval_out_of_distribution, 134 games). Agent LLM = qwen3-30b-a3b-instruct, zero-shot ReAct-style prompt (no few-shot traces). Jev = typesafe/jev-1.13 `noul` questions on {task, room, last 8 steps, proposed action}.
Stage 1: 45 games, cap 25, LLM agent + 20% random admissible actions, giving 1051 transitions (9/45 episodes won). Labels come from the next observation. The LLM baseline was asked the same questions on a random 300-transition subsample. Priors are leave-one-game-out base rates.
Stage 2: 20 other games, cap 30. A = ReAct (free-form action). D = LLM proposes top-3 admissible, executes its #1 (no Jev). B = same top-3, Jev "progress" P(yes) picks. C = B + surprise note when P(actual outcome) < 0.3.

## Stage 1 (AUROC / ECE)
| target (base rate) | Jev | qwen3-30b (n=300) | best LOGO prior |
|---|---|---|---|
| effect vs "Nothing happens" (.72) | .90 / .18 | .60 / .41 | .78 (admissible-list rule: 1.00) |
| target object mentioned (.08) | .87 / .17 | .74 / .15 | .80 |
| target found on go/open only (.04, n=772) | .75 / .15 | .39 (n=221) | n/a |
| place is empty (.17) | .74 / .19 | .61 / .39 | .81 |

Jev ranks outcomes well but is miscalibrated: it is underconfident at the extremes and overconfident at 0.2-0.4. It needs Platt scaling. Surprise did **not** predict episode failure (AUROC .53-.56), and the raw "Nothing happens" rate did better (.58-.71).

## Stage 2
| arm | success | steps | LLM calls | Jev calls | $/game |
|---|---|---|---|---|---|
| A ReAct | 7/20 | 22.5 | 22.5 | 0 | .0017 |
| D top-3, no Jev | 7/20 | 22.5 | 22.5 | 0 | .0017 |
| B top-3 + Jev | 10/20 | 19.8 | 19.8 | 53 | .0034 |
| C B + surprise | 11/20 | 20.4 | 20.4 | 55 | .0035 |

Paired: C beat D on 4 games and lost on 0 (sign test p≈.125). Surprise notes fired only 0.3 times per game, so the replanning part was barely exercised. Total spend: $0.35.

**Verdict: MIXED.** Jev works as a cheap outcome ranker and reranking gives a small, noisy gain. Surprise-as-failure-signal and surprise-triggered replanning were not supported.

**A full experiment would need:** all 134 games × 3 or more seeds, calibrated Jev, logging of how often Jev overrides the LLM's first choice, a few-shot ReAct baseline, a denser surprise definition, and a same-cost LLM-verifier arm. Published ReAct ALFWorld numbers are not quoted because I did not verify them.

# Round 2: scaled Stage 2 (all 134 valid_unseen games)

**Setup.** The proposer (qwen3-30b) runs at temperature 0.7 with a per-step seed and a 30-step cap. Arms (all on identical game-seed units):
- **D:** the LLM's own #1 of its top-3 admissible actions.
- **B:** Jev P(progress) picks among the 3.
- **C:** B plus a denser surprise rule: a replan note fires if the executed action's P(observed outcome) < 0.3 on any of effect, reveal or empty.
- **V:** the same LLM scores the 3 candidates with Jev's exact questions (one call per step).

Seeds 0 and 1 are complete; seed 2 stopped at the budget cap after 101 games, giving 369 units. Script: `round2.py`. Raw data: `round2_seed*.jsonl`. Tables: `round2_analysis.txt`.

| arm | success (95% Wilson CI) | steps | override rate | surprise notes/ep | $/ep |
|---|---|---|---|---|---|
| D | .347 [.300, .397] | 23.0 | — | — | .0020 |
| B | **.485 [.435, .536]** | 21.3 | .398 | — | .0040 |
| C | .480 [.429, .531] | 21.4 | .395 | 1.14 | .0040 |
| V | .412 [.363, .463] | 22.7 | .279 | — | .0038 (2x wall time) |

Exact McNemar tests over game-seed units:
- **B vs D:** B won on 64 units, D on 13 (p<.001). Seed 0 alone: 25 vs 6, p=.001.
- **B vs V:** 53 vs 26 (p=.003).
- **V vs D:** 46 vs 22 (p=.005).
- **C vs B:** 25 vs 27 (p=.89).

Units from the same game are correlated across seeds, so the p-values are somewhat optimistic.

**Stronger backbone** (qwen3-235b-a22b-2507 proposer, 134 games, 1 seed): D .515, B .552. B won 13 units, D won 8 (p=.38). The gain shrinks from +14 points to +4 and is not significant.

**Recalibrated Jev:** not run. Choosing by argmax over P(progress) is unchanged by any monotone recalibration (temperature or Platt), so it would equal B exactly.

**Wall time:** about 28–30 min per 536-episode seed with 24 processes. Round cost: $5.96.

**Round 2 verdict: PROMISING, with a caveat.**
- Jev reranking reliably helps a weak agent (+14 points, about 2x cost, still under $0.005 per episode).
- Jev beats an LLM second opinion given the same information, so the gain is specific to Jev.
- Surprise-triggered replanning adds nothing.
- Against a stronger agent the effect is small and unresolved; it would need more seeds.

# Round 3: stronger backbone, cluster-aware tests, ablations

**Tests.** Pooled McNemar over game-seed units, plus two cluster-aware tests. For each game, the B minus D success difference is averaged over seeds. The mean of those per-game differences gets a 95% cluster (game) bootstrap CI, and they also go into a Wilcoxon signed-rank test. Scripts: `analyze_r3.py` and `override_analysis.py`. Tables: `round3_analysis.txt`.

| comparison | success | units / games | McNemar | per-game diff [bootstrap CI] | Wilcoxon p |
|---|---|---|---|---|---|
| 30b B vs D | .484 vs .346 | 370 / 134 | 64:13, p<.001 | +.129 [+.067, +.192] | .0001 |
| 30b B vs V | .485 vs .412 | 369 / 134 | 53:26, p=.003 | +.068 [+.014, +.123] | .019 |
| 30b C vs B | .478 vs .484 | 370 / 134 | 25:27, p=.89 | -.006 [-.046, +.034] | .96 |
| **235b B vs D (3 seeds)** | .545 vs .493 | 402 / 134 | 51:30, p=.026 | +.052 [-.007, +.112] | .10 |
| Jev alone (J) vs 30b D | .403 vs .343 | 134 | 30:22, p=.33 | +.060 [-.045, +.164] | .27 |
| J vs 30b B | .403 vs .485 | 134 | 23:34, p=.18 | -.082 [-.187, +.030] | .15 |
| trivial rule (R) vs 30b D | .396 vs .343 | 134 | 14:7, p=.19 | +.052 [-.015, +.119] | .13 |
| 30b B vs R | .485 vs .396 | 134 | 20:8, p=.036 | +.090 [+.015, +.164] | .023 |

235b per seed, B vs D: 74 vs 69, 70 vs 68, and 75 vs 61 wins out of 134 games.

**Arm definitions.**
- **J:** Jev alone scores every admissible action for "progress" in one multi-question call per step and takes the argmax. There is no LLM. It costs about 1 Jev call per step.
- **R:** takes the first of the LLM's top-3 candidates that has not been executed before and never returned "Nothing happens."; otherwise the LLM's #1.

**Override analysis** (round-2 arms B and C, 30b). Jev overrode the LLM's #1 on 6267 of 15822 steps (39.6%).
- **Choosing a different location:** 76% of overrides were go→go. Of these, Jev's destination actually held the target 272 times, versus 94 for the LLM's pick.
- **Receptacle ordering:** cabinet-to-cabinet swaps were the largest group (1268).
- **Avoiding revisits:** in 991 overrides the LLM's #1 repeated a past action and Jev chose a new one. The reverse happened 488 times.
- **Skipping no-op actions:** 469 overrides replaced examine, look or inventory.
- **Preferring 'go' over interacting:** 692 overrides went from an object interaction to a 'go' action; 187 went the other way.

The trivial rule recovers about 40% of the gain (+5 of +14 points), and B beats R significantly.

**Round 3 verdict: MIXED to PROMISING.**
- With the 30b agent the gain survives cluster-aware testing (+13 points, CI excludes 0).
- Part of the gain is Jev knowing where objects are likely to be, beyond simply avoiding repeats.
- With the stronger 235b agent the effect shrinks to +5 points and is borderline: pooled McNemar p=.026, but cluster-aware p=.10.
- Jev without an LLM proposer lands between the D and B arms, so the proposer still helps.
- Round 3 cost: $2.45.
