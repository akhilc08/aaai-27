# Compaction pilots for the AAAI-27 student abstract

Minimal runs of the top ideas from `thinking-system/orchestrator/runs/run3_export.json`.
Cheap models via OpenRouter: `qwen/qwen3-30b-a3b-instruct-2507` and `minimax/minimax-01`.
Scenarios and prompts are reused from the Watchpoint recency study in `context-research`.

**Read first:** `results/FINDINGS.md` (morning brief), then `results/summary.md` (all tables and stats) and `figures/*.png`.

| Script | Idea | What it tests |
|---|---|---|
| `exp2_ratchet.py` | 2 and 5 | Recursive vs source-anchored compaction at matched session length, conditional vs unconditional rules, 80 vs 160 word summaries |
| `exp1_repair_retention.py` | 1 | Generic self-check passes (k=1,2,4) vs bigger summary budgets (160 to 640 words), scored per token spent |
| `exp3_ratchet_rescue.py` | 3 | On 4-round sessions, can repair passes or a bigger budget rescue a damaged rule? |
| `exp4_nll_gap.py` | 4 | Label-free probe: how much more surprising the rule sentence is given the summary vs the full transcript, under a local Qwen2.5-1.5B |
| `judge_exp2.py` | verification | LLM judge labels how each exp2 summary states the rule: as a rule, as status/to-do, or absent |
| `exp5_framing.py`, `exp5b_framing_robust.py` | verification | Same rule inserted with controlled framing, no compaction, 5 model families |
| `regrade.py` | verification | Re-judges every agent reply so "comply after checking the condition" is not counted as a violation |
| `calib_exp2.py` | - | Calibration that showed a fixed-length transcript saturates, which is why exp2 uses a growing session |

Run everything (resumable, skips finished trials): `./run_all.sh`. Refresh tables only: `python analyze.py`.

The OpenRouter key is read from `context-research/.env` at runtime and never written here.
`results/aborted/` holds the first exp2 attempt (fixed 96-turn transcript), stopped because every cell was near 100% violation.
