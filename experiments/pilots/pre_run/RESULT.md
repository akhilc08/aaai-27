# Pilot: predicting agent success on SWE-bench Verified from the issue text alone

**Question.** Can a cheap calibrated predictor that sees only the issue say whether an agent will resolve it? Is it more than issue length or repo difficulty, and does it hold across agents? Is it useful for routing?

**Setup.** All 500 SWE-bench Verified issues. Resolved labels come from 13 public systems, ranging from 2.8% resolved (RAG-GPT4) to 75.6% (mini-SWE-agent with Claude 4.6 Opus). Inputs are the problem statement (cut to 5k chars) and the repo name, with no hints. The predictors are:
- Jev zero-shot (1 yes/no question).
- A Jev 10-question head (logistic regression, 5-fold CV, 3 seeds).
- A zero-shot probability from qwen3-30b and from gpt-4.1-mini, plus a gpt-4.1-mini head on the same 10 questions.
- Baselines: log length, code block present, traceback present, repo one-hot, and human difficulty buckets.
- Oracle: the size of the gold patch.

**AUROC.** The target is "resolved by at least half the systems" (39% positive).

| Predictor | AUROC |
|---|---|
| length | .55 |
| length+code+traceback+repo | .61 |
| human difficulty | .71 |
| qwen3-30b zero-shot | .62 |
| gpt-4.1-mini zero-shot / 10q head | .69 / .77 |
| **Jev zero-shot / 10q head** | **.75 / .78** |
| baseline + human difficulty | .73 |
| baseline + human difficulty + Jev | **.80** |
| gold patch size + baseline + difficulty (oracle) | .75 |
| oracle + Jev | .81 |

- **Beyond length, repo and difficulty.** Adding Jev raises AUROC by .07 over baseline plus human difficulty (bootstrap 95% CI .03 to .10).
- **Not a length proxy.** Jev correlates with length at only ρ=−.14, and within each length quartile its AUROC is .71 to .81.
- **Per system.** AUROC is .69 to .81 for weak and mid systems, but only .61 to .65 for frontier systems.
- **Transfer.** A head trained on one system's labels does as well on other systems as on its own (mean .706 vs .712). The predictor captures difficulty shared by all agents, not anything specific to one agent.

**Routing** (send issues from a weak agent to a strong one). At 30% of issues routed from gpt-5-nano to Opus 4.6, the resolved rate is .486 with the Jev head, .470 with random routing and .544 with an oracle. Routing the "hardest" issues first can do worse than random, because the strong agent fails those too. A policy that routes by expected gain (P(strong solves) − P(weak solves)) gives +2 to 3 points, and the length/repo baseline does about as well.

**Cost per 1,000 predictions.** Jev: $0.04, 0.24s median latency. qwen: $0.04, 1.2s. gpt-4.1-mini: $0.19 to $0.37, 0.65s. The pilot spent $0.34 in total.

**Caveats.** All the issues are public from 2023 or earlier, so contamination was not tested. Each routing simulation is a single run with no confidence intervals.

**Verdict: MIXED.** The difficulty predictor is solid, cheap and robust across agents. Its practical routing value is small, because what it predicts is difficulty shared by all agents, not which agent will succeed.

**What a full experiment needs.**
- Issues from after the training cutoff (SWE-bench-Live or SWE-rebench).
- Routing aware of cost and gain, with CIs across many weak/strong pairs.
- Skip-hopeless-issue curves.
- Agent-conditioned features.
