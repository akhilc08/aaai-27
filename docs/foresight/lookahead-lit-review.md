# Jev-in-the-loop lookahead: literature review (2026-09-26)

Scope: using Jev (fast, calibrated, non-generative) as the source of **probabilities** (opponent-action distributions, chance outcomes) and **leaf values** in a deep, probability-weighted (expectimax-style) search for Pokemon Showdown and HUNL poker. [UNVERIFIED] means I could not confirm it from a primary source during this session. Items tagged [known] are standard references cited from memory and not re-fetched.

## Q1. PokeChamp (Karten, Nguyen, Jin; arXiv 2503.04094, ICML 2025)

- **Search:** it is *minimax*, not expectimax. The objective is `argmax_a min_b E_x ... r(x_H)`: it takes the worst case over the opponent's actions and an expectation only over stochastic transitions [P1]. Depth is a truncated *k* steps with an LLM value at the leaves. **The paper does not report k, the branching factor, the number of LLM calls per turn, or any cost or latency** [P1].
- **What the LLM does:** (1) it samples player actions, which are merged with "a few candidate actions from our tools"; (2) it models the opponent, combining stat estimates from historical data with LLM prediction of "the most likely opponent actions" (no probabilities are used, only a candidate set); (3) it scores leaves as a value function from prompted factors (move effectiveness, Pokemon remaining, speed, and so on) [P1].
- **World model:** a local Showdown simulator plus a one-step damage-formula lookahead. Opponent spreads are inferred from a dataset of 3M+ games, 500k+ of them high-Elo [P1].
- **Time:** 150 s bank per player plus a 15 s increment per turn. "About one third of the games" PokeChamp lost came from exceeding the turn limit [P1]. **Latency is the binding constraint on depth.** This is our opening.
- **Protocol and results:** Gen 9 OU and Gen 8 Random Battles (with and without Dynamax), against PokeLLMon, Abyssal, One-Step, MaxPower and Random, with at least 25 games per pair [P1].
  - Gen 9 OU vs Abyssal: GPT-4o 84% (Elo 1268), Llama-3.1-8B 56% (1204), PokeLLMon/GPT-4o 40% (1020).
  - Gen 8 Random (no Dmax) vs Abyssal: GPT-4o 70%, Llama-8B 64%, PokeLLMon 56%.
  - Abstract: 76% vs PokeLLMon (GPT-4o), 64% with Llama-8B, and a projected ladder Elo of 1300-1500 [P1].
- **Code:** github.com/sethkarten/pokechamp. It runs against a local Showdown fork with `uv`. Backends are OpenRouter and Ollama. It ships `local_1v1.py` (e.g. `--opponent_name abyssal`) and `scripts/evaluation/evaluate_gen9ou.py` [P2]. The license is **MIT plus a clause prohibiting use in Japan**, which is non-standard [P2]. Swapping in Jev looks feasible because the backend is a pluggable LLM wrapper. I have not inspected the code, so [UNVERIFIED] how cleanly the three modules separate.

## Q2. Other Pokemon agents and benchmarks

- **PokeLLMon** (arXiv 2402.01118): GPT-4 on Gen 8 random battles. Win rates were 49% on the ladder (105 games) and 56% in invited games (50) [P3].
- **Metamon** (arXiv 2504.04395): offline RL on reconstructed human replays, Gen 1-4 OU. It wins about 85-95% against heuristics, reaches 64-80% GXE on the ladder (top 10%) over 400+ battles per generation, beats Foul Play in Gen 1-2 and ties it in Gen 3-4 [P4].
- **PokeAgent Challenge** (NeurIPS 2025; arXiv 2603.15563):
  - Battling track in Gen 1 OU and Gen 9 OU, plus a speedrunning track.
  - Baselines: heuristics, 30 Metamon checkpoints, and a PokeChamp LLM harness.
  - Metric: Bradley-Terry rating (with Glicko and GXE); 250 or more battles.
  - Default timers of 60-90 s "proved insufficient for LLM inference", so an Extended Timer mode exists.
  - Foul Play (MCTS) won Gen 9 OU, and 13 of 16 qualifiers extended the RL baselines. "Specialist RL and search methods outperform LLM approaches" [P5].
- **Foul Play** (github.com/pmariglia/foul-play): root-parallel MCTS/DUCT on the Rust poke-engine, with no rollouts and a hand-written evaluation. It spends about 7 s per move [P6b] and reaches 80% GXE in Gen 9 OU (peak 1879) and 88% GXE in Gen 9 Random Battle (peak 2341) [P6].
- **PokaiTrainer** (arXiv 2608.29197): Student-of-Games belief-state search for VGC, reaching a 1350-1400 Elo band [P7].
- **poke-env baselines:** Random, then MaxBasePower, then SimpleHeuristics, in increasing strength. SimpleHeuristics is much stronger than the other two, and simple RL agents typically beat it only 16-20% of the time [P8]. Absolute numbers depend on the format.
- **Best "published numbers" for 2 days:** **PokeChamp's Gen 8 Random Battle table vs Abyssal.** No team building is needed, the code is public, and it gives LLM-search numbers to compare against directly. PokeAgent is stronger as a benchmark but needs 250+ ladder games on their server.

## Q3. Poker

- **LLM agents:**
  - Suspicion-Agent (arXiv 2309.17277, COLM 2024) uses GPT-4 theory-of-mind prompting on Leduc [P9].
  - PokerBench (arXiv 2501.08328, AAAI 2025) is an 11k-spot SFT benchmark [P10].
  - PokerSkill (arXiv 2605.30094) combines rule-based skills with an LLM and uses no solver. Against GTO Wizard it scores -57±21 mbb/hand (GPT-5.5) and -80±29 (Opus 4.6). It "outperform[s] Slumbot" [P11].
  - SpinGPT reports +13.4±12.9 BB/100 vs Slumbot over 30k hands [P12, UNVERIFIED beyond the search snippet].
  - GTO Wizard Benchmark (arXiv 2603.23660) is a public HUNL API with AIVAT. Frontier LLMs "remain far below" the benchmark there [P13].
- **Opponents and tools:** the Slumbot public API (200bb reset per hand, roughly 7 s average per hand [P14]); OpenSpiel and RLCard for Leduc and limit games; GTO Wizard API.
- **Metric:** mbb/hand (or bb/100) with 95% CIs. Variance is high: tens of thousands of hands without AIVAT.
- **Feasibility for a 2-day pilot:** **poor as a headline.** Getting a reliable CI against Slumbot needs roughly 10^4 or more hands, and naive probability-weighted search in HUNL is not sound (see Q4b). Leduc Hold'em in OpenSpiel is feasible as a toy secondary.

## Q4. Fast value or probability sources inside search

**a) LLMs in search.**
- ToT (2305.10601), RAP (2305.14992), LATS (2310.04406) and LLM-MCTS (2305.14078) all use a generative LLM as the prior, value or world model [known].
- TS-LLM applies AlphaZero-style search to LLM decoding (2309.17179) [P15].
- MC-DML runs MCTS with LLM priors on Jericho text games (2504.16855) [P16].
- Strategist (2408.10635) has an LLM write value heuristics that low-level MCTS then uses, in GOPS and Avalon [P17].
- DeepMind's MAV (2412.12119, ICML 2025) is a trained language model that scores all actions and drives MCTS, reaching grandmaster-level chess [P18].
- Speculative Actions (2510.04371) uses a fast model to predict an agent's next actions for speed. It gets up to 55% accuracy and up to 20% lower latency [P19].

**b) Probability-weighted search.**
- Expectimax and \*-minimax (Ballard 1983) add chance nodes [known].
- Stochastic MuZero adds learned chance nodes and afterstates (ICLR 2022) [P20].
- AlphaZero and MuZero use learned priors [known].
- Opponent-model search replaces min nodes with a *modelled* opponent. PrOM search (Donkers et al. 2001) weights opponent types by probability and beats OM search and sometimes minimax in Bao, at high cost with depth [P21].
- An accurate opponent model helps MCTS in Pommerman (2305.13206) [P22].
- In imperfect-information games (poker; hidden sets in Pokemon), plain expectimax over a fixed opponent model is exploitable. Sound methods search over public belief states: DeepStack, Libratus, Pluribus, ReBeL (2007.13544) and Student of Games (2112.03178) [known].
- Our framing should therefore be **best response to a calibrated opponent model**, not equilibrium.

**c) Calibration.**
- LLM verbalized confidence is systematically overconfident. Consistency sampling helps but costs many calls (Xiong et al., 2306.13063, ICLR 2024) [P23].
- RLHF degrades token-probability calibration (GPT-4 report 2303.08774; Kadavath et al. 2207.05221) [known].
- Expectimax is sensitive to miscalibration because errors in chance and opponent weights multiply along each path. I found **no paper that isolates calibrated vs overconfident opponent models inside expectimax or MCTS as its main variable** [UNVERIFIED gap; the Pommerman and PrOM results are the nearest]. **That makes it a clean ablation for us.**

**d) Jev and Laya.**
- Jev: 70-500 ms per call, $0.042/Mtok input, output free. It is trained with "RLCD" for calibration. The launch post gives **no ECE or Brier numbers**. Its demos are a Doom bot and Wikiracing [P24][P25].
- Laya is an open 421M-parameter analogue with about 33 ms local calls [P26].
- **I found no published use of Jev or Laya inside tree search.**

## Q5. Small-model rerankers for agents; ALFWorld reference numbers

- Step-level Q-value models (Zhai et al., AAAI 2025) were evaluated on WebShop, HotpotQA and ALFWorld; Phi-3-mini improves by 103% on WebShop [P27].
- Best-of-Q (2601.22701): an offline Q-function reranks candidates from a frozen VLM on WebVoyager. Qwen2.5-VL-7B goes from 38.8% to 55.7% and GPT-4.1 from 82.4% to 88.8% [P28].
- Cross-environment DeBERTa rerankers (2606.02204) on ALFWorld, WebShop and ScienceWorld [P29].
- Agent Step Value (2607.04419) and text world models as verifiers (2606.09032) [search hits; UNVERIFIED details].
- **Differentiator for our ALFWorld result:** these rerankers are *trained* per domain. Jev is used zero-shot.
- **ReAct** (2210.03629) was run on 134 unseen ALFWorld validation games with PaLM-540B and task-specific prompts. Success was 71% for the best of 6 prompts, 48% for the worst trial, 45% for the best Act run and 37% for BUTLER. GPT-3 (text-davinci-002) reached 78.4% vs PaLM's 70.9% [P30].
- **Reflexion** (2303.11366) with GPT-3 on the same 134 tasks solved 130 of 134 (97%) over 12 trials [P31].
- Our 34.6% to 48.4% baseline is far below these, so we should report our model and split explicitly.

## Closest prior work and our differentiation

1. **PokeChamp** is the closest: an LLM supplies priors, the opponent model and leaf values in Showdown search. It differs from us in three ways:
   - it uses worst-case minimax over a small *unweighted* opponent set, while we weight all opponent actions and damage and chance outcomes by calibrated probabilities (expectimax);
   - it uses generative, uncalibrated LLM scores, while Jev returns calibrated typed probabilities;
   - it is timeout-limited, with an unreported and presumably shallow depth, while we can make many more node evaluations inside the 15 s increment.
2. **MAV (2412.12119)** is also a single-pass model that scores all actions for MCTS. But it is trained per game on game data, runs on perfect-information games, and does not claim calibration.
3. **Strategist / PrOM / Stochastic MuZero** cover probability-weighted search with learned or opponent models, but none uses an off-the-shelf language-conditioned decision model.
4. **Foul Play / PokaiTrainer** are hand-engineered and far stronger. We should not claim to beat them.

**Novel claim (defensible):** *an off-the-shelf, zero-shot, calibrated non-generative decision model as both the chance/opponent distribution and the leaf evaluator for expectimax. Calibration and latency, not raw accuracy, are what make deeper probability-weighted lookahead pay off.*

Must-run ablations:
- (i) Jev probabilities vs uniform vs LLM-verbalized probabilities in the same tree;
- (ii) expectimax vs minimax;
- (iii) depth vs wall-clock budget;
- (iv) temperature-distorted Jev, to test calibration causally.

## Recommended benchmark and baselines

- **Setup:** PokeChamp repo, local Showdown, **Gen 8 Random Battle (no Dynamax)**, 15 s per-turn budget. Keep the damage calculator and simulator. Replace the three LLM modules with Jev: a choice distribution over the opponent's legal moves, and a win-probability yes/no at leaves. Search depth 2-3 with expectimax.
- **Opponents:** Abyssal, poke-env SimpleHeuristics and MaxBasePower, plus PokeChamp run with Llama-3.1-8B (via Ollama) and a GPT-4o-class backbone under the *same* budget. Use at least 100 games per pair for ±10pp CIs.
- **Numbers to beat (vs Abyssal, Gen 8 Random, no Dmax):** PokeChamp GPT-4o 70%, Llama-8B 64%, PokeLLMon 56% [P1].
- **Report:** timeout rate and nodes evaluated per turn.
- **Poker:** optional Leduc toy in OpenSpiel only.

## References

- [P1] https://arxiv.org/abs/2503.04094 (html) · [P2] https://github.com/sethkarten/pokechamp (+/blob/main/LICENSE)
- [P3] https://arxiv.org/abs/2402.01118 · [P4] https://arxiv.org/abs/2504.04395 · [P5] https://arxiv.org/abs/2603.15563, https://pokeagent.github.io/
- [P6] https://pmariglia.github.io/posts/foul-play/ · [P6b] https://www.smogon.com/forums/threads/re-introducing-foul-play-a-competitive-pokemon-battle-bot.3767378/ · [P7] https://arxiv.org/abs/2608.29197
- [P8] https://poke-env.readthedocs.io/ ; https://github.com/Siddharth-Y26/PokeRL (search snippets)
- [P9] https://arxiv.org/abs/2309.17277 · [P10] https://arxiv.org/abs/2501.08328 · [P11] https://arxiv.org/abs/2605.30094 · [P12] https://link.springer.com/chapter/10.1007/978-3-032-23657-9_8 · [P13] https://arxiv.org/abs/2603.23660 · [P14] https://github.com/aipoker-bot/slumbot-adapter
- [P15] https://arxiv.org/abs/2309.17179 · [P16] https://arxiv.org/abs/2504.16855 · [P17] https://arxiv.org/abs/2408.10635 · [P18] https://arxiv.org/abs/2412.12119 · [P19] https://arxiv.org/abs/2510.04371
- [P20] https://openreview.net/pdf?id=X6D9bAHhBQ1 · [P21] https://www.sciencedirect.com/science/article/abs/pii/S0020025501001335 · [P22] https://arxiv.org/abs/2305.13206 · [P23] https://arxiv.org/abs/2306.13063
- [P24] https://typesafe.ai/blog/introducing-system-one-models-and-jev · [P25] https://docs.typesafe.ai/concepts/system-one · [P26] https://huggingface.co/convaiinnovations/laya
- [P27] https://ojs.aaai.org/index.php/AAAI/article/view/34924 · [P28] https://arxiv.org/abs/2601.22701 · [P29] https://arxiv.org/abs/2606.02204 · [P30] https://arxiv.org/abs/2210.03629 · [P31] https://arxiv.org/abs/2303.11366
- [known, not re-fetched] ToT 2305.10601; RAP 2305.14992; LATS 2310.04406; LLM-MCTS 2305.14078; AlphaZero 1712.01815; MuZero 1911.08265; ReBeL 2007.13544; Student of Games 2112.03178; DeepStack 1701.01724; Pluribus (Science 2019); Ballard *-minimax (AIJ 1983); GPT-4 report 2303.08774; Kadavath et al. 2207.05221.
