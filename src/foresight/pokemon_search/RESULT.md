# Pilot: Jev-scored expectimax lookahead for a Pokémon Showdown bot

**Question.** Jev (fast, calibrated, non-generative) gives the opponent-move probabilities and/or the leaf values inside a probability-weighted (expectimax) search. Does that win games? Does it beat no search, minimax, and heuristic bots? Does depth help?

**Setup.** Local Showdown server. The main format is gen9randombattle vs poke-env `SimpleHeuristicsPlayer` (SH). The published protocol is gen8randombattle vs the **real PokéChamp `AbyssalPlayer`** (run from their repo with dynamax disabled, "as for local battles"). Our own abstract simulator (`model.py`) uses the damage formula with estimated stats and branches on chance: accuracy, KO-roll buckets, secondary status, full paralysis, and speed ties. It ignores abilities, items, crits, tera, and hazard damage. Unseen opponent mons are placeholders, not sampled worlds. Pruning: all our root actions; the opponent's top-3 root actions (top-2 deeper), covering 90% of the probability mass; chance branches covering 90%/75% of the mass. Our own actions at deeper levels are pruned by a free heuristic prior. Every tree level is one batched Jev call, and the leaves are one more.

| arm | leaf value / opp. policy | vs SH gen9 (n, win%, 95% CI) | vs Abyssal gen8 (n, win%, CI) | mean latency (s) | $/battle |
|---|---|---|---|---|---|
| J0 no search (Jev choice) | – | 100, 43 (34-53) | 100, **27** (19-36) | 0.28 | 0.0007 |
| E1 / E2 expectimax | Jev 5-pt score / Jev | 100, 19 / 60, **0** | – | 0.5 / 0.9 | 0.007 / 0.027 |
| P1 / P2 expectimax | Jev P(win) / Jev | 100, 21 / 40, 12 | – | 0.5 / 1.0 | 0.008 / 0.030 |
| X1 / X2 / X3 expectimax | HP balance / **Jev** | 100, 58 / 200, **66** (59-72) / 150, 64 | 100, 53 / 300, **57** (51-62) | 0.3 / 0.6 / 1.1 | 0.001 / 0.008 / 0.038 |
| XM2 minimax | HP balance / Jev-chosen replies | 200, 59 (52-66) | – | 0.6 | 0.009 |
| H1 / H2 / H3 expectimax, no API | HP balance / damage softmax | 200, 54 / 200, 62 / 200, 60 | 103, 49 / 300, **61** (55-66) / 100, 51 | ≤0.15 | 0 |
| HM1 / HM2 minimax, no API | same | 200, 61 / 200, 68 | – / 200, 54 | ≤0.03 | 0 |

No decision in any arm exceeded 15 s. The maximum was 11.3 s, and the p95 was 1.6 s for X3. Errors fell back to a random move in under 0.1% of decisions.

**Findings.**
1. **Jev as the leaf evaluator is harmful.** Both the score leaf and the P(win) leaf lose to SH far more often than no search does, and depth makes it worse (E2 won 0/60). Jev's leaf values barely respond to the simulated HP changes; for example, it rated an attack the target is immune to highly.
2. **With an HP leaf, Jev opponent probabilities make a solid bot.** X2 wins 66% vs SH and 57% vs the real Abyssal. But it is **not better than the same search with a free damage-softmax opponent model**: H2 wins 62% / 61%, and all the CIs overlap. The whole gain comes from the simulator plus search, not from Jev.
3. **Depth:** d1 → d2 helps (58→66, 54→62). d2 → d3 is flat. **Expectimax vs minimax:** with Jev policies, 66 vs 59 (CIs overlap); with the heuristic policy, minimax 68 vs expectimax 62. No consistent advantage for probability weighting.
4. **Jev P(win) forecasts** on real positions: 23.7k turns from 840 battles. Brier .209 vs .250 for the base rate; AUROC .74 overall but only .56 at turns 1-5. It is roughly calibrated at 0.4-0.8 and underconfident at the extremes: a forecast of 0.84 won 94% of the time, and 0.16 won 1%. Note that the P1/P2 bots themselves won only 16% of their battles.
5. **Rollout leaf (C)**, tested on 6 states at depth 1: 11 s per decision on average (max 15.8 s), $0.043 per decision, about 2.7k Jev questions. The P(win) leaf (B) took 0.86 s and $0.0003. C sits right at the 15 s limit and costs about 150× more per decision.

**Context (published, not re-run).** PokéChamp, gen8 random battles vs Abyssal: 70% with GPT-4o, 64% with Llama-3.1-8B, and PokéLLMon 56%. Our Jev X2 scored 57% (51-62), and the free H2 scored 61% (55-66). Both are in PokéLLMon/8B territory, at 0.6 s per turn and under $0.01 per battle. Our N and exact settings may differ from theirs.

**Verdict: NOT PROMISING** for the claim that Jev as a search evaluator wins games. The search works, but Jev adds nothing measurable over a free heuristic, and Jev leaf values actively hurt. The forecasting side is **MIXED**: P(win) discriminates late in the game but not early.

**A full experiment needs:** a real simulator (the Showdown engine or PokéChamp's calculator); hidden-team sampling from randbats set data (skipped here for budget); a trained or learned leaf head on top of Jev features; matched-N runs against PokéChamp's exact Abyssal protocol; and ≥500 battles per arm to separate effects of 5 points.

Spend: $16.84 (tag pksearch). The LLM-forecaster arm was dropped per instructions after 2 smoke battles.

## Round: formatted state (tag pksearch_fmt, $1.30)

**Change.** `fmt.py` is plain code with no API calls. It turns each position into compact, precomputed facts: units alive, total HP% and the difference, a material label, active HP/status/boosts, who moves first, each side's best attack (damage %, effectiveness, KO this turn, hits to KO), who wins the one-on-one race, and safe-switch counts. It drops move-id lists and species names. Jev receives these facts for its opponent-policy choices (option texts carry the damage, effectiveness and KO consequences precomputed) and for the P(win) leaf. At the real root it is also asked 4 narrow nouls: ahead, race, safe, threat.

Example formatted state:
```
units_alive: us=6, them=6, difference=0
total_hp_percent: us=584, them=588, difference=-4
material: roughly even
note: 4 of their units not yet seen (counted at full HP)
actives:
  our_active: hp_percent=84, status=none, boosts=none
  their_active: hp_percent=88, status=none, boosts=none
  moves_first: us
  our_best_attack: damage_percent_of_their_hp=16, effectiveness=resisted, knocks_out_this_turn=False, hits_to_knock_out=6
  their_best_attack: damage_percent_of_our_hp=22, effectiveness=resisted, knocks_out_this_turn=False, hits_to_knock_out=4
  one_on_one_race: they knock out our active first
safe_switches: our_bench...=0, their_known_bench...=0
```

**Offline check** on the 9 saved states (`data/probe_fmt.json`). Mean P(win) by variant:

| format | orig | ahead | behind |
|---|---|---|---|
| old text | .52 | .66 | .14 |
| formatted facts | .52 | **.87** | **.13** |

The formatted version hits the target (about 0.9 ahead, 0.1 behind). The "ahead" noul gives .98 / .02 on the same variants.

**Battles.** F2 is depth-2 expectimax with formatted Jev policies and the formatted Jev P(win) leaf, vs SimpleHeuristics in gen9 random battles: **9/60 wins = 15% (CI 8-25)**, plus 0/2 smoke battles. For comparison: the old Jev-leaf depth-2 arms scored 0% (E2) and 12% (P2); the HP-balance leaf X2 scored 66%. Latency was 0.89 s mean, 4.0 s max, at about $0.021 per battle.

**Forecasts on real root positions** (1,905 turns, 62 battles, win base rate .17, base-rate Brier .143):

| forecast | Brier | AUROC |
|---|---|---|
| P(win) | .197 | .76 |
| "ahead" noul | .163 | .79 |
| mean(P(win), ahead, race) | .184 | .77 |
| race / safe / threat | .33 / .29 / .33 | .64 / .61 / .34 |

All of them are miscalibrated: every Brier score is worse than the base rate's.

**Verdict for this round: NOT PROMISING.** Formatting fixes the offline sensitivity test, and it ranks real positions somewhat better (AUROC .76 vs .74). But as a leaf inside search it still loses badly, at 15% vs 66% for a plain HP-balance leaf. Likely cause (not tested): search takes the max over many noisy leaf values, so it steers toward positions whose P(win) is over-estimated. Discrimination of about .76 is far too weak for picking among sibling moves that differ by a few HP points.

Note: I accidentally re-ran `probe_knowledge.py` once, through an import. It overwrote `data/probe_knowledge.json` with near-identical numbers (orig .53 / ahead .68 / behind .14) and cost $0.001 under the pk_probe tag. The Showdown server is stopped.

## Round: hybrid leaf (tag pksearch_hyb, $1.05)

**Setup.** Leaf = HP balance plus λ × (formatted Jev P(win) − 0.5), with λ = 0.25, on a [0,1] scale. Everything else is identical to X2: depth-2 expectimax, Jev opponent probabilities on the original text, same pruning. Each search also backs up the pure-HP leaf, so it records whether the Jev correction changed the chosen move; this costs nothing extra.

**Result vs SimpleHeuristics (gen9):** 20/40 wins = **50% (CI 35-65)**, vs 66% (CI 59-72, n=200) for X2 with the pure HP leaf. Latency was 0.92 s mean, 1.65 s p95, 3.4 s max. Cost was $0.026 per battle, against $0.008 for X2.

**How often Jev changed the move:** in 373 of 1,255 unforced decisions (30%). That is not a small tie-breaker: at λ = 0.25, Jev's P(win) spread across sibling leaves is as large as the HP differences between them.

**Verdict: NOT PROMISING.** The correction changes about a third of moves and the win rate drops 16 points; the CIs overlap slightly at this n. I did not run λ = 0.5: the result was not "at least as good", and $0.15 of budget remained. Server stopped.
