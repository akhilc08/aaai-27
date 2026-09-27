# Pilot: Jev on PokerBench (zero-shot and learned head)

**Question.** Can Jev (calibrated typed choices) match a cheap LLM at predicting the solver-optimal action on PokerBench? And does a learned head over narrow Jev questions add anything beyond parsed card and board features?

**Setup.** PokerBench test split (HF `RZ412/PokerBench`). Random sample of 400 preflop and 600 postflop spots (seed 0). Instructions were parsed into hand, board, street, pot and whether hero faces a bet, and legal actions were derived from that (fold/call/raise if facing a bet, otherwise check/bet or check/raise). The labels contain 98 postflop "raise" labels where hero is not facing a bet; these were treated as bet. Metric: action accuracy (AA), with predictions masked to legal actions. Jev ran on all 1000 spots, using one call per spot: a choice over legal actions plus 9 yes/no questions (hand strength, made hand, likely best, draw, position, wet board, villain aggression, pot odds, bluff spot). The heads are multinomial logistic regressions with 5-fold stratified CV, averaged over 5 seeds. The cheap LLM (qwen3-30b-a3b-instruct) got the dataset's own prompt on a 200-spot subset (80 preflop, 120 postflop).

| Method (n=1000 unless noted) | AA | AA pre / post | macro-F1 |
|---|---|---|---|
| Majority, legal-masked | 50.3 | 46.3 / 53.0 | .29 |
| Jev zero-shot choice | 61.1 | 68.5 / 56.2 | .57 |
| Cheap LLM (subset n=200) | 58.5 (EM 41.5) | EM 35 / 46 | .57 |
| Jev zero-shot (same 200) | 61.0 | | .59 |
| LR on Jev choice probs + legal mask | 70.3 | | |
| LR on text features | 71.0 | 81.5 / 65.8 | .64 |
| LR on Jev features | 71.6 | 81.5 / 65.8 | .65 |
| LR on text + Jev | 72.6 | 82.8 / 66.8 | .66 |
| Text LR trained on 40k train spots | 62.2 | | .54 |

Seed-to-seed spread for the CV heads is about ±1 pt. Paper context (full test set, few-shot): GPT-4 AA 65.5 / EM 53.6; fine-tuned Llama-3-8B AA 80.6 / EM 78.3.

Cost and latency: Jev 0.27 s/call and $0.031 per 1k (10 questions per call). LLM 2.15 s/call and $0.022 per 1k. Total pilot spend: $0.036.

**Verdict: MIXED.** Zero-shot Jev beats the cheap LLM (61 vs 58.5 AA) at 8x lower latency, but it does not reach GPT-4. The learned Jev head ties the hand-crafted features, and adding Jev to them gains only about +1.6 pt. Much of the jump from 61 to 70 is re-fitting to the test set's balanced class prior; a head trained on the train split scores only 62.

**A full experiment would need:** heads trained on the train split and evaluated on the full test set, with a prior correction; exact match (bet sizing) for Jev via a choice over the solver's bet-size menu; stronger LLM baselines; and more targeted questions for postflop spots.
