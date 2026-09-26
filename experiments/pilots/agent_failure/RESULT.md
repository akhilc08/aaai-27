# Pilot: predicting coding-agent failure from its first k steps

**Question.** Can Jev, reading the issue plus the first k agent steps, predict whether a SWE-agent run ends with a passing fix? Does a logistic head over 8 narrow Jev questions beat one zero-shot question, a cheap LLM judge, and simple heuristics?

**Data.** `nebius/SWE-agent-trajectories`, 4 of 12 parquet shards, llama-70b runs only. Labels come from `target`, the hidden-test result. A step is one agent action plus its observation, and the prefix shows the first k steps. A run is scored at k only if it has more than k steps, so it is still running and its submit step is never shown. Observations are cut to 450 chars, actions to 700 and the issue to 1500 (about 1k tokens).
- **Paired set** (controls for difficulty): 1 passing and 1 failing run per issue, for every issue with mixed outcomes. That gives 526 runs, 263 issues, 196 repos and a 50% pass rate.
- **Natural set**: 1 random run per issue. That gives 500 runs, 500 issues, 319 repos and a 10% pass rate.
- Heads use 5-fold cross-validation grouped by repo.

**AUROC** (natural / paired)

| k | n (nat) | errors heuristic | 5-feat heuristic | LLM judge (qwen3-30b) | Jev 1 question | Jev 8q head | Issue-only Jev | Total length (leaky ref) |
|---|---|---|---|---|---|---|---|---|
| 1 | 500 | .53/.49 | .53/.50 | .68/.53 | **.81**/.51 | .79/.50 | .79 | .77/.56 |
| 3 | 495 | .48/.52 | .48/.51 | .76/.50 | .76/.55 | **.81**/.53 | .79 | .78/.56 |
| 5 | 472 | .47/.51 | .57/.52 | .76/.56 | .73/.53 | **.84**/.54 | .78 | .80/.57 |
| 10 | 396 | .42/.51 | .59/.54 | .78/.58 | .73/**.62** | **.88**/.61 | .78 | .82/.59 |

At k=10 on the natural set, the issue-only score plus the 8 questions gives .895. The natural-set 95% confidence intervals are about ±0.05 to 0.08, because there are only about 30 to 50 passing runs.

**Cost and speed.** Jev with all 8 questions in one call costs $0.063 per 1,000 predictions at a median of 0.24s. The LLM judge costs $0.10 per 1,000 at 0.93s median. The pilot spent $0.67 in total.

**Verdict: MIXED.** Jev clearly beats the heuristics and the LLM judge, and the 8-question head helps as k grows. However, the early (k=1) signal is almost all issue difficulty: scoring the issue text alone gives .79. When the issue is held fixed, success is barely predictable (.50 to .62). The early-kill story is weak; predicting difficulty plus progress by k=5 to 10 is the defensible claim.

**What a full experiment needs.** All 12 shards (more passing runs), several agent models, and stronger per-run features: diffs so far, and whether the test or reproduction script now passes. A "cost saved at a fixed recall of passing runs" curve. Paired and natural evaluation reported side by side.

Files: prep*.py, run.py, run_k0.py, analyze.py, raw*.jsonl, auroc_*.json, auroc_vs_k*.png
