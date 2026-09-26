# Overnight findings (2026-09-26)

Full tables and stats: `summary.md`. Figures: `../figures/`. All violation numbers below are **regraded**: an LLM judge (gpt-4.1-mini) read each agent reply and counted it as a violation only if the agent did the forbidden thing. The original COMPLY/REFUSE label wrongly counted "merge after confirming CI and approval" as a violation.

## Verdict per idea

| Idea | Result | Paper-worthy? |
|---|---|---|
| 2. Recursive compaction ratchet | **Falsified.** Recursive and source-anchored compaction are tied at every session length (p > 0.19). Loss is driven by session length: 26% violation at 16 turns, 72% at 64 turns (p = 4e-29). | As a negative control, yes |
| 5. Recursion depth, not frequency | **Falsified**, same data. Length, not recursion, explains loss. | Folds into 2 |
| 1. Repair vs retention | **Directionally confirmed, small n.** Bigger summaries are cheaper per point of violation than self-check passes. Self-checks help when the rule survived (82% to 29%) but barely when it was deleted (86% to 77%). | Yes, as a secondary figure |
| 3. Ratchet vs rescue | **Inconclusive.** Nothing to rescue because recursion causes no extra damage (idea 2). Repair beat retention here, opposite of prediction. n=24 per cell, one model. | No |
| 4. Reconstruction-NLL gap | **Works, but no better than a trivial baseline.** Local 1.5B model's gap predicts violation with AUC 0.79 and deletion with AUC 0.84, with no labels. A word-overlap count gets AUC 0.81. Blind to status/to-do reframing (AUC 0.63). | Weak |

## The unplanned finding (strongest result of the night)

**Summaries turn conditional rules into to-do items, and agents treat to-do items as optional.**

1. In 1,152 real compacted sessions, a judge labeled how each summary stated the rule. Violation: stated as a rule 40%, recast as status/to-do 77%, deleted 91% (stated vs status p = 4e-17). A reframed rule is almost as bad as a deleted one.
2. Conditional rules ("never merge without approval") get recast this way 4x as often as flat bans (21% vs 5% of summaries).
3. **Controlled test, no compaction** (exp5, exp5b): the same rule inserted in one sentence, in different framings.

| Model | Rule as an order | Conditional as recap | Conditional as to-do (2 wordings) | Flat ban as recap |
|---|---|---|---|---|
| Qwen3-30B | 0% | 25% / 42% | 61% / 92% | 0% |
| MiniMax-01 | 0% | 6% / 31% | 22% / 58% | 0% |
| GPT-4.1-mini | 0% | 0% / 39% | 42% / 78% | 0% |
| Gemini-2.5-Flash-Lite | 0% | 14% / 47% | 44% / 83% | 0% |

The order form is never violated in any of 432 trials. Flat bans survive recap framing perfectly. Llama-3.3-70B was also run but refuses half the time with no rule at all, so it can't show an effect.

**Candidate abstract title:** "Compaction Turns Rules into To-Dos: Framing, Not Deletion, Drives Constraint Violation in Summarized Agent Context." It matches the "empirical finding" archetype and has a clean mechanism, a controlled test, 4 model families, and a one-line fix to test next (tell the summarizer to keep rules in imperative form).

## Caveats to state honestly

- Synthetic filler transcripts; decisions are a structured proxy, not real tool execution.
- Some to-do wordings also soften content ("move credentials into the secrets manager" no longer says "never hardcode"). The first wording set is cleaner; the effect holds there too.
- The judge is itself an LLM; spot-checked, not human-validated.
- Even "stated as rule" summaries show 40% violation vs about 4% with the full transcript. Unexplained; worth a look.
- exp1 present/absent splits have n = 9 to 17 per cell.

## Not done

- The obvious fix experiment (imperative-preserving summary prompt) was not run, to stay under budget.
- Nothing committed to git.

## Cost

About $10 of OpenRouter credit tonight (account usage 7.78 to 17.74 USD). All API work stopped at the cap; the NLL scoring ran locally.
