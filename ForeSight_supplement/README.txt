ForeSight: Small Calibrated Decision Models as Opponent Forecasters for Lookahead Game Play
Supplementary material for the AAAI-27 student abstract. Author: Saiakhil Chilaka, Cornell University.
Full repository: https://github.com/akhilc08/aaai-27

START HERE: supplement.pdf (6 pages). It expands every claim in the abstract with full tables,
95% confidence intervals, figures, method and training details, negative results, and
reproduction information. Section 1 maps each number in the abstract to its source.

CONTENTS

supplement.pdf
    Extended results and details.

code/foresight/
    ForeSight source code (Python). Main entry points:
      gen_data.py       run logged bot-vs-bot battles on a local Showdown server (training data)
      build_ds.py       turn logged battles into forecasting datasets
      train_laya.py     partial fine-tune of Laya as the opponent forecaster
      model.py          approximate Pokemon simulator used by the search
      search.py, evals.py, bots.py   expectimax search, opponent models, battle bots
      run_arm.py        play N battles against an opponent (e.g. ABYSSAL, SH) and log results
      abyssal_runner.py runs PokeChamp's AbyssalPlayer in chunks (it leaks memory)
      score_turns.py    score forecasters on an opponent's real logged moves
    scripts/            the exact shell commands used for each run (absolute paths from the
                        original machine; edit the cd line before running)
    autoresearch/       the overnight search-tuning loop: program.md (rules), eval.py,
                        baseline_strategy.py (headline configuration), final_strategy.py
                        (tuned configuration used for the depth/opponent-model grid)

code/pilots/
    alfworld_rerank/    the ALFWorld pilot (System One model reranks an LLM's top-3 actions)
    jevlib.py           helper for the pilot's API calls (keys read from a local .env; none included)

results/foresight/
    RESULT.md               full results log with CIs, including the overnight runs
    battles_log.jsonl       one line per battle for every run: arm name, battle id, won, turns
    arm_results.jsonl       one line per run: settings, wins, n, latency (mean / p95 / max), errors
    summary_gen8/9*.json    forecasting metrics for every opponent model (accuracy, log-loss, ECE,
                            switch AUROC, per-opponent accuracy)
    score_turns_*.json      forecaster scores on Abyssal's real moves

    Arm names used in the abstract:
      layah_hp_d2 (ABYSSAL, SH)   Gen 9 forecaster, depth 2: 113/200 vs Abyssal, 121/200 vs SimpleHeuristics
                                  (these ran before per-battle logging; totals are in arm_results.jsonl)
      layah8_hp_d2 (ABYSSAL)      Gen 8 forecaster, depth 2, vs Abyssal: 113/200
      ar_heur_aby + ar_heur_aby_b depth 2, damage-softmax heuristic: 584/1000
      ar_heur_d1_aby / ar_heur_d3_aby   depth 1 / 3, heuristic: 424/1000, 518/1000
      ar_unif_aby / ar_unif_d1_aby      depth 2 / 1, uniform: 555/1000, 422/1000
      ar_best_aby + ar_best_aby2  tuned strategy, Laya, depth 2 (held-out test): 105/200
      ar_laya_d1_aby + ar_laya_d1_aby_b tuned strategy, Laya, depth 1: 40/119

results/autoresearch/
    results.tsv, notes.md   every tuning experiment, its dev result, and the keep/discard decision

results/alfworld_rerank/
    RESULT.md, round2_analysis.txt, round3_analysis.txt   ALFWorld pilot results
    (round3_analysis.txt holds the 34.6% -> 48.4% result)

NOT INCLUDED
    Fine-tuned checkpoints (874 MB) and generated battle data; train_laya.py and gen_data.py
    regenerate them. Pokemon Showdown (commit a5df827) and PokeChamp (commit 0f84c46) are
    external repositories cloned next to the code.

SOFTWARE
    Python 3.11, torch 2.14, poke-env 0.16.1, laya 0.3.20, transformers 5.17, scikit-learn 1.9,
    Node 24. Tested on an Apple M5 Pro MacBook (MPS). ForeSight makes no API calls.
