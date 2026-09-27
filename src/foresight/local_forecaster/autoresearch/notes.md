# autoresearch notes (one line per experiment)

Setup 2026-09-27 00:10: strategy.py refactored to expose knobs (leaf ALIVE_W/STATUS_W/BOOST_W/MATCH_W, MM_LAMBDA, OPP_TEMP, SW_FLOOR, LAYA_LEVELS, DEEP_EXTRA); defaults identical to baseline (check_equiv 249/249). Dev screen takes ~23 min, not 15.
Offline (free, gen8 bot-vs-bot positions, not Abyssal): leaf variants barely change outcome AUROC (HP .755; +alive/status/matchup .759). Leaf gains, if any, must come from changing search behaviour, not prediction quality.
Offline timing: turn time is almost all Laya forecasts (depth 2 ~19/turn); simulator cost ~0.02 s even at depth 3. Full-Laya depth 3 needs ~90 forecasts/turn (too slow); depth 3 with Laya at levels 0-1 and the free damage heuristic at level 2 keeps ~19.
- 000 baseline (headline config): 34/80 then 32/80 -> pooled 66/160 = 41.3%, p95 3.7 s. Best = 41.3%; screen bar for new ideas = 47.3%.
- 001 depth 3 (Laya levels 0-1, free heuristic opponent at level 2): 24/80 = 30.0%, p95 3.7 s. Discard (-11 pts). A crude opponent model at the extra ply hurts more than the depth helps.
- 002 richer leaf (alive 0.3, status penalties, matchup 0.3): 32/80 = 40.0%. Discard (within noise of baseline).
- 003 root switch cost 0.02 (proxy for unmodelled hazards and tempo): screen 38/80 = 47.5% (+6.2 over best), p95 3.5 s -> confirm running.
- 003 confirm 33/80 -> pooled 71/160 = 44.4% (+3.1 over 41.3%). KEEP (just clears the +3 rule; honestly within noise). New best 44.4%, screen bar 50.4%.
- 004 best + sharper Laya forecasts (temp 0.5): 39/80 = 48.8% (+4.4, below the +6 screen bar). Discard; near-miss to combine later.
- 005 best + minimax blend (lambda 0.3): 32/80 = 40.0%. Discard. Review after 5: trusting forecasts more (004 sharper) helped a bit; hedging against worst case (005) hurt -> forecasts carry real signal. Next: bolder sharpening.
- 006 best + sharper forecasts (temp 0.3): screen 41/80 = 51.3% (+6.9), p95 2.6 s (faster: sharper probs prune more replies) -> confirm.
- 006 confirm 38/80 -> pooled 79/160 = 49.4% (+5.0 over 44.4%). KEEP. New best 49.4% (vs baseline 41.3%: +8.1, two-proportion p~0.15). Screen bar now 55.4%.
- 007 best + adaptive depth 3 (budget 30 forecasts): timeout at 67 games, 32/67 = 47.8%, mean turn 3.9 s, depth 3 on 82% of turns. Fail/discard (no sign of gain either).
- 008 best + OUR_K 4 (wider own replies at depth 1, no extra forecasts): 37/80 = 46.3%. Discard.
- 009 temp 0.15 (sharper than best's 0.3): 35/80 = 43.8%. Discard; 0.3 looks like the sweet spot (1.0: 44.4 pooled, 0.5: 48.8, 0.3: 49.4 pooled, 0.15: 43.8).
- 010 switch cost 0.04 (vs best's 0.02): 32/80 = 40.0%. Discard.
- 04:42 DISCOVERY: without Laya calls the search takes ~5 ms/turn, so a free-opponent-model eval of 600-1500 games takes 1-3 min. Used for measurements and fast knob sweeps (still one eval at a time).
- M1u measurement, best strategy (006) with a UNIFORM opponent model: 170/400 = 42.5%. M2h, same with the free damage-softmax HEURISTIC opponent: 174/400 = 43.5%. Laya best: 79/160 = 49.4% -> Laya +6-7 pts over both on dev (z~1.5).
- Sweep a (heur opp, 600 games each; SE ~2 pts): depth 1 = 30.0%, depth 2 = 43.5%, depth 3 = 43.7% -> lookahead depth 1->2 is worth +13.5 pts; temp 1.0 = 39.2% vs 0.3 = 43.5% (sharpening helps with any forecaster); switch cost 0 / .01 / .02 / .04 = 45.8 / 46.3 / 43.5 / 37.8.
- Sweep c (heur opp, 1500 games each, SE of a difference ~1.8): base 44.2, +matchup 43.9, +status 45.1, switch .01 41.6, OUR_K 4 46.7, combo 46.5, wider opp pruning 45.2, wider chance 45.1. No leaf/search knob moves play by more than ~2.5 pts; the levers are depth and forecast quality/sharpness. Nothing adopted.
- Plan (pre-registered 05:30): M3 = 160 more dev games of best Laya config; then held-out Abyssal test of best (006) for 100 games, extended to 200 if time allows; then heur/uniform opponent models vs Abyssal with the same strategy (measurement); then Laya depth 1 on dev.
- M3best: 160 more dev games of best (006) hit the 30-min cap at 131: 52/131 = 39.7%. Pooled best config 131/291 = 45.0% (006 screen+confirm 79/160 + M3 52/131). Honest dev picture: best 45.0% (n=291) vs baseline 41.3% (n=160), +3.7, not significant; heur-opp 44.2% (n=1500); uniform-opp 42.5% (n=400). The 006 jump was partly winner's curse.
- HELD-OUT (Abyssal, gen8, PokeChamp protocol, chunked), final strategy = 006 (switch cost 0.02 + forecast temp 0.3): 55/100 = 55.0%, p95 turn 2.9 s, 0 errors, 22 min. Same as baseline 113/200 = 56.5%. Extending to 200 as pre-registered (arm ar_best_aby2).
- HELD-OUT second 100 (ar_best_aby2): 50/100. Held-out total 105/200 = 52.5% (Wilson 95% CI 45.6-59.3) vs baseline 113/200 = 56.5% (49.6-63.2). No held-out gain; within noise.
- MEASUREMENTS vs Abyssal (fixed final strategy 006; only the opponent model / depth varied; never used for selection):
  - heuristic (free damage-softmax) opponent model, depth 2: 121/200 + 463/800 = 584/1000 = 58.4% (~2 min per 200 games, no GPU)
  - uniform opponent model, depth 2: 555/1000 = 55.5%
  - Laya opponent model, depth 2: 105/200 = 52.5% (held-out test above)
  - depth 1 (same strategy): heuristic 424/1000 = 42.4%, uniform 422/1000 = 42.2% -> depth 2 adds +16.0 / +13.3 pts (z~7). Depth-1 ~ PokeChamp's published One-Step 44% (protocol sanity check).
  - depth 3, heuristic opponent model: 518/1000 = 51.8% (< depth 2's 58.4%; mean turn 0.03 s). Depth helps 1->2 but not 2->3 in this approximate simulator.
- 10:05 WRAP-UP: user ended the loop. Laya depth-1 extension hung after 19 games (Abyssal process crashed inside a battle: PokeChamp AttributeError in _stat_estimation); killed. Laya depth 1 pooled 40/119 = 33.6%. Offline: top-1 on Abyssal's real moves: Laya 0.754, heuristic 0.489, uniform 0.550 (the uniform figure reflects the candidate-order tie-break). Report written to RESULT.md (Autoresearch section).
