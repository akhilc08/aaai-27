# ForeSight: Small Calibrated Decision Models as Opponent Forecasters for Lookahead Game Play

Code and results for the AAAI-27 student abstract ([`paper/paper.pdf`](paper/paper.pdf)).

ForeSight fine-tunes a small non-generative "System One" decision model ([Laya](https://github.com/NandhaKishorM/laya), 421M parameters) to forecast an opponent's next action with calibrated probabilities. A depth-2 expectimax search then weights every imagined future by those forecasts and by chance. In Pokémon Showdown it runs on a laptop with no LLM and no API calls.

## Repository layout

```
paper/                  the 2-page abstract (paper.tex, refs.bib, AAAI style files)
src/
  foresight/
    local_forecaster/   ForeSight: data generation, Laya fine-tuning, search, battle runners
      RESULT.md         every result with 95% CIs (start here)
      autoresearch/     the overnight search-tuning loop and its logs
    pokemon_search/     the earlier Jev-based search that local_forecaster grew out of
  pilots/               smaller feasibility studies that motivated ForeSight
    imagination_planning/   ALFWorld: System One reranking of an LLM's top-3 actions
    pokemon/  poker/        zero-shot Jev opponent/action forecasting
    alfworld_predict_act/   ALFWorld: LLM predicts the observation before acting
    jevlib.py               shared helper for Jev / cheap-LLM API calls
docs/                   research notes (docs/foresight/findings.md holds the project history)
authorkit/              unmodified AAAI-27 author kit
```

## Where the paper's numbers come from

| Paper claim | Source |
| --- | --- |
| Table 1: forecasting accuracy, log-loss, calibration | `src/foresight/local_forecaster/RESULT.md` §1 (`train_laya.py`, `score_turns.py`) |
| Table 2: 56.5% vs Abyssal (n=200) | `RESULT.md` §2 and "Overnight batch" (`run_arm.py ... ABYSSAL`) |
| Speed (36–84 ms per forecast, 2.7–4.1 s per turn) | `RESULT.md` §3 (`scripts/bench_laya.py`) |
| Lookahead adds 13–20 points (depth 1 vs 2, n=1000) | `RESULT.md` "Autoresearch" section |
| ALFWorld 34.6% → 48.4% | `src/pilots/imagination_planning/round3_analysis.txt` (arm B vs D; setup in `RESULT.md` there) |
| Raw text masks a decisive lead (P(win) 0.66 vs 0.87) | `docs/foresight/findings.md` §9b (`src/foresight/pokemon_search/probe_knowledge.py`) |

Per-battle outcomes are logged to `data/battles_log.jsonl` and per-decision latency to `data/dec_*.jsonl`. Both are gitignored.

## Reproducing ForeSight

Tested on macOS with an Apple M5 Pro (MPS), Python 3.11 and Node 24. Key packages: torch 2.14, poke-env 0.16.1, laya 0.3.20, transformers 5.17, scikit-learn 1.9.

```bash
cd src/foresight/local_forecaster
git clone https://github.com/smogon/pokemon-showdown        # tested at a5df827
git clone https://github.com/sethkarten/pokechamp           # tested at 0f84c46; provides the Abyssal bot
node pokemon-showdown/pokemon-showdown start 8001 --no-security &

# 1. Generate bot-vs-bot battles and build datasets (see scripts/gen.sh for the full list)
FMT=gen8randombattle LIVEHIST=1 .venv/bin/python gen_data.py SH SH 500 10
.venv/bin/python build_ds.py gen8randombattle

# 2. Fine-tune Laya as the opponent forecaster (~35 min)
BS=4 FMT=gen8randombattle HIST=1 TASKS=opp NOZS=1 .venv/bin/python train_laya.py 4 0 0 10000 opp_hist

# 3. Play Abyssal under PokéChamp's protocol (gen8, no Dynamax, 20-battle chunks)
FMT=gen8randombattle CHUNK=20 .venv/bin/python run_arm.py layah8_hp_d2 \
    laya_hist:laya_opp_hist_gen8randombattle.pt hp 2 ABYSSAL 100 4
```

The Abyssal runner expects PokéChamp's dependencies in `../pokemon_search/.venv-pc`. PokéChamp's Abyssal process leaks memory, so run one Abyssal job at a time.

**Not in the repo:** fine-tuned checkpoints (`ckpt/`, 874 MB), generated battle data (`data*/`), the cloned PokéChamp and Showdown repos, and virtual environments. Scripts under `scripts/` contain absolute paths from the original machine; edit the `cd` line before running them.

## Pilots

The pilots called TypeSafe's Jev and cheap LLMs through OpenRouter via `src/pilots/jevlib.py`. Every call's cost is in `src/pilots/spend.jsonl`. API keys are read from a local `.env` at runtime and are never stored in the repo. ForeSight itself (`src/foresight/local_forecaster`) makes no API calls.

## License

The AAAI author kit in `authorkit/` and the style files in `paper/` belong to AAAI. The poke-engine evaluator port in `evals.py` (`FPLeaf`) follows pmariglia/poke-engine (MIT), with attribution in the file.
