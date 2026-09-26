# Pilot: predicting a human Showdown opponent's next action (Jev vs cheap LLM)

**Question.** Can a cheap calibrated model (Jev), optionally with a small learned head, predict a human opponent's next action about as well as an LLM, so it could replace LLM opponent modeling in PokéChamp-style minimax search?

**Setup.** 160 rated gen9randombattle replays (opponent rating median 1989, min 1630) from the replay API. Each turn is viewed from both sides. The state is what that player can see from the log: both actives (HP%, status, tera, boosts, revealed moves), revealed benches, weather, terrain, hazards, and the last 3 turns. Our own unrevealed team is missing, which is a limitation. The label is the opponent's first voluntary action that turn. From 7,430 labeled turns, 1,500 were sampled (159 battles). Learned heads use 5-fold GroupKFold by battle. The cheap LLM (qwen3-30b-a3b) ran on 150 turns because of the budget. Total spend: $0.061.

**(a) Switch vs move** (n=1500, switch base rate 0.181)

| method | acc | log-loss | Brier | AUROC |
|---|---|---|---|---|
| base rate | .819 | .474 | .149 | .47 |
| repeat last action type | .819 | .470 | .148 | .52 |
| 12 simple features, LR | .822 | .422 | .134 | .732 |
| Jev zero-shot, raw | .796 | .552 | .182 | **.440** |
| Jev zero-shot + Platt | .819 | .470 | .148 | .551 |
| 8 Jev nouls + simple features, LR | .822 | **.417** | **.132** | **.745** |
| LLM (subset of 150) | .780 | .592 | .181 | **.392** |

On the same 150-turn subset, Jev + simple features got AUROC .744 and the LLM got .392. Jev nouls add +0.012 AUROC over simple features alone (95% battle-bootstrap CI: -0.000 to 0.026).

**(b) Which option among revealed moves + switch.** This covers 52% of turns and 45% of move turns. n=783, about 3.1 options per turn.

| method | acc | NLL |
|---|---|---|
| uniform | .337 | 1.074 |
| repeat last move | .462 | 1.175 |
| Jev zero-shot | .469 | 1.146 |
| Jev + head switch prob | **.499** | **1.063** |

On a 79-turn subset: LLM .418 / 1.307, Jev + head .468 / 1.120.

**Cost/latency.** Jev: 0.26 s median per call with 9 questions, $0.036 per 1k. LLM: 2.56 s, $0.049 per 1k.

**Verdict: MIXED.** Zero-shot, both Jev and the LLM score below chance on switch prediction; they predict "switch when low HP or threatened", which humans don't do. Jev + a learned head beats the LLM and is about 10x faster, but it barely beats simple features.

**Full experiment needs:** a stronger LLM baseline (PokéChamp's GPT-4o prompt), a full own-team view from a simulator, thousands of battles to fit a proper head, and an actual win-rate test inside minimax search.
