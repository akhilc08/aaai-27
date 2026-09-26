# Pilot: predicting compaction-induced rule violations with Jev

**Question.** Can Jev (reading the compacted context, the original rule, and the trigger) predict whether the agent will violate a turn-1 rule better than LLM self-judgment and the keyword oracle? Does a learned head over several Jev questions help?

**Setup.** Snapshot of exp2_ratchet.jsonl (complete, 1,200 rows); regime=none excluded -> n=1,152 (0 UNKNOWN). Label y=1 if decision==COMPLY. Verified: every trigger asks for the forbidden action with the condition unmet (cond) or outright (uncond); the no-compaction ceiling complies 2/48. Base rate 0.570 (qwen 0.70, minimax 0.44; cond 0.63, uncond 0.51). Scenarios are balanced (192 each, y 0.43-0.69), and none dominates. LLM self-judge = qwen3-30b, binary yes/no, run on all rows. Jev head = LR over logits of 6 noul questions. Accuracy uses the best threshold chosen in-sample, the same way for every method. Files: features.py, analyze.py, features.jsonl, metrics.json.

| Method | AUROC | Acc | AUROC qwen / minimax |
|---|---|---|---|
| Keyword missing | 0.747 | 0.744 | 0.703 / 0.806 |
| LLM self-judgment | 0.519 | 0.570 | 0.474 / 0.570 |
| Jev zero-shot P(lost) | 0.830 | 0.786 | 0.827 / 0.873 |
| Jev 6q + LR, 5-fold | 0.883 | 0.819 | 0.855 / 0.908 |
| Jev 6q + LR, leave-one-scenario-out | 0.869 | 0.810 | 0.842 / 0.892 |
| Jev 6q + backbone + LR, LOSO | 0.878 | 0.816 | 0.839 / 0.889 |
| Keyword + backbone + LR, LOSO | 0.764 | 0.752 | |
| Keyword + Jev + backbone, LOSO | 0.876 | 0.817 | |

- LLM self-judgment flags only 6.4% of trials as rule lost, while 57% violate. This reproduces the prior finding.
- Jev zero-shot beats the keyword oracle: +0.083 AUROC, paired-bootstrap 95% CI [+0.060, +0.107]. It wins in 5 of 6 scenarios; force_push is the exception (0.83 vs 0.87).
- Jev adds signal beyond the keyword. On rows where the keyword is present (n=558, y=0.32), the Jev head's AUROC is 0.85. Adding the keyword to the Jev head adds nothing.
- Cost per 1k predictions: Jev $0.023 (1 question) / $0.031 (6 questions) vs LLM $0.026. The costs are about the same because the LLM answers in one token. Latency: Jev 0.27s vs LLM 1.9s. The claim "a fraction of the cost" is false; "7x faster, same cost" holds.
- Total spend: $0.093.

**Verdict: PROMISING.** Jev beats both the keyword oracle and self-judgment. The multi-question head adds about +0.04-0.05 AUROC and holds under leave-one-scenario-out.

**Caveats / full experiment needs.** Only 6 synthetic scenarios and filler transcripts. There are 2 backbones, and the agents are the same small models. A continuous (logprob) LLM baseline and a stronger LLM judge are needed. Also needed: held-out rule phrasings, real agent traces, and a test of whether acting on the flag (re-injecting the rule) actually cuts violations.

## Round 2 (intervention + robustness), spend $0.457 (tag total $0.549)

**1. Intervention** (intervene.py, analyze_r2.py, metrics_r2_intervention.json). All 1,152 compacted contexts x 2 reps, same backbone as the original run. Each context gets a fresh plain decision and a fresh pinned decision (rule re-inserted verbatim before the request). Each gated condition takes the pinned decision when its gate fires and the plain decision otherwise (counterfactual, paired). Jev-head and zero-shot thresholds are picked on the training scenarios (leave-one-scenario-out) to reach a target recall.

| Condition | Violation | Re-inserted | Extra tok |
|---|---|---|---|
| a. plain | 0.591 | 0% | 0 |
| b. always-pin | 0.006 | 100% | 34.6 |
| c. keyword-gated | 0.172 | 52% | 17.8 |
| e. LLM self-judge-gated | 0.545 | 6% | 2.4 |
| d0. Jev zero-shot @0.5 / @recall.8 / @recall.9 | 0.238 / 0.129 / 0.074 | 40 / 58 / 77% | 13.5 / 19.8 / 26.4 |
| d. Jev head LOSO @recall.8 / .9 / .95 | 0.158 / 0.089 / 0.065 | 51 / 62 / 69% | 17.6 / 21.8 / 24.0 |
| random gate @62% | 0.247 | 61% | 21.1 |
| perfect predictor of original-run label | 0.046 | 57% | 19.9 |

- Jev gates beat the keyword gate and the random gate, but they never match always-pin.
- At a matched ~51% re-insertion rate, the Jev head beats the keyword gate only slightly (0.158 vs 0.172).
- Always-pin costs only 35 tokens and nearly eliminates violations. Gating can save at most ~43% of those tokens, since even a perfect gate re-inserts 57% of the time, and it adds a predictor call.
- **For a single short rule, gating is NOT worth it.** It would only matter with many or long rules, where pinning them all is expensive (untested).

**2. Continuous LLM judges** (judge.py). Stratified n=192, 8 per scenario x ctype x backbone cell. AUROC for "P(rule lost) 0-100" / "P(agent complies) 0-100":
- qwen3-30b: 0.563 / 0.577, $0.03 per 1k predictions
- gpt-4.1-mini: 0.574 / 0.540, $0.12-0.14 per 1k predictions

On the same rows: keyword 0.752, Jev zero-shot 0.794, Jev head 0.868. The LLM judges collapse to "0" in 70-80% of rows, so a fair continuous baseline is still near chance.

**3. Held-out wording** (rephrase.py, rephrased_rules.json). Each rule was paraphrased once by qwen, and the paraphrase was fed to Jev on all 1,152 rows. The head was trained on the original wording (LOSO). AUROC held up:
- Jev single question: 0.822 -> 0.834
- Jev head: 0.878 -> 0.891

**Round 2 verdict: MIXED.** The predictor is robust (it holds under paraphrase and beats fair LLM judges), but gated re-insertion is not a win over always-pin in this setup. A full experiment needs a multi-rule or long-policy setting where pinning everything is costly.

## Round 3 (multi-rule policy, per-rule gating), qwen only, spend $2.13 (tag total ~$2.68)

**Setup** (r3/: common3.py, stage1-3.py, distractors.json, contexts.jsonl, decisions.jsonl, metrics_r3*.json, stage3_n320.log)
- **Policy:** the turn-1 policy has 20 rules: the 6 real cond rules plus 14 distractors generated once by qwen.
  - The distractor keywords were hand-assigned, because the generated ones were "unless"/"never".
  - One distractor (hardcoded config values) was hand-replaced because it overlapped hardcoded_creds.
  - Rule order is shuffled for each context.
- **Compaction:** recursive, over 96 turns, with (rounds, budget) in {(1,250), (1,600), (3,400), (3,800)}. Tighter settings (3-6 rounds, 80-160 words) lost almost every rule.
- **Contexts and triggers:** 320 compacted contexts x 6 triggers = 1,920 units, 1 agent decision per unit per gate (14,359 decisions in total).
  - The first 40 contexts were also run with 2 reps, and the conclusions matched (metrics_r3_first40_2reps.json).
- **Gates:** each gate picks which rules to pin, and pinned rules are appended verbatim.
  - Jev: one call with 20 parallel noul questions ("is rule i still preserved, including its condition?").
  - Jev threshold: chosen leave-one-trigger-scenario-out for a target recall of plain violations.

| Gate | Violation | Triggered rule pinned | Rules pinned | Extra tok |
|---|---|---|---|---|
| plain | 0.470 | 0% | 0 | 0 |
| always-all | 0.043 | 100% | 20 | 504 |
| keyword-per-rule | 0.096 | 75% | 14.5 | 366 |
| LLM-per-rule | 0.207 | 41% | 8.2 | 212 |
| Jev @0.5 | 0.233 | 27% | 7.3 | 203 |
| Jev LOSO @recall.8 / .9 | 0.077 / 0.064 | 72 / 82% | 17.5 / 18.6 | 447 / 471 |
| Jev, count matched to keyword (t=0.73) | 0.112 | 53% | 14.3 | ~360 |
| **random 14 rules** | **0.100** | 69% | 14.0 | ~355 |

- **No gate predicts violations.** AUROC for predicting the triggered rule's plain violation from its per-rule signal: keyword 0.488, LLM 0.501, Jev 0.529. All are at chance.
- **Spillover:** pinning any rules at all lowers violations of the unpinned triggered rule, from ~0.49 to ~0.31. The gates therefore "work" only by pinning many rules, and a random gate does as well.
- **Gate cost per 1k:** Jev $0.082 at 0.26s vs LLM $0.115 at 1.74s.
- **Minimax repeat skipped**, because the gate showed no signal.

**Round 3 verdict: NOT PROMISING.** With 20 rules, the summaries carry vague "policy-compliant / approval workflows" phrasing, and per-rule preservation (by Jev, LLM or keyword) does not predict which triggered rule gets violated. Round 1's strong single-rule result does not carry over to per-rule gating.
