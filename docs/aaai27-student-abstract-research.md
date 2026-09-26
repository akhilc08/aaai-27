# AAAI-27 Student Abstract: Research Notes

As of 2026-09-25. Everything we learned about the track, what gets accepted, how our existing work stacks up, and candidate directions.

## 1. Track facts

The program is explicitly for early-stage work. AAAI's stated goal: "to provide a forum in which students can present and discuss their work during its early stages."

| Item | Rule |
| --- | --- |
| Submission deadline | **September 28, 2026 (AoE)** via [EasyChair](https://easychair.org/conferences/?conf=aaai27sa) |
| Notification | November 13, 2026 |
| Camera-ready | December 4, 2026 |
| Presentation | February 19, 2027, in person only; poster plus a 3-minute talk with one static slide |
| Eligibility | Undergraduate, Masters, or Doctoral student must be primary author and investigator |
| Advisors | Non-student advisors are acknowledged, not listed as co-authors |
| Limit | One submission per student as primary author; no authors added after the deadline |
| Dual submission | Allowed for work under review elsewhere (including AAAI-27 main track); not for already-published work |
| Supplement | Optional PDF or ZIP up to 20 MB. The call says it is "critical to reviewers, because of the brevity of the submissions" |
| Publication | Accepted abstracts get two pages in the official proceedings with a DOI |
| Anonymization | None. Author names and affiliations go on the submission |
| Contact | aaai27sachairs@aaai.org |

Sources: [AAAI-27 call](https://aaai.org/conference/aaai/aaai-27/student-abstract-and-poster-program-call-for-proposals/), [AAAI-26 call](https://aaai.org/conference/aaai/aaai-26/student-abstract-and-poster-program-call/).

## 2. Format rules

**At submission: two pages including references.** The call says papers over length "are subject to rejection without review." Fit references inside two pages for the Monday PDF.

**At camera-ready: a third page for references is allowed in practice.** In a random sample of 20 published AAAI-26 student abstracts, 16 are three pages, and page 3 holds only references and acknowledgments. This matches AAAI's general proceedings policy of a fixed content budget "plus additional pages solely for references, acknowledgements."

Template: AAAI two-column camera-ready style. The AAAI-27 author kit is unpacked at `authorkit/AuthorKit27/` (`aaai2027.sty`, `aaai2027.bst`, `CameraReady2027.tex`). Title gets a "(Student Abstract)" suffix at camera-ready.

Practical budget: about 1.7 pages of body text plus 4 to 10 references. Each reference costs about three lines.

## 3. What gets accepted

Volume is growing: 87 accepted at AAAI-25, 132 at AAAI-26. Full title lists with links are in `prior-student-abstracts.md`.

### Topic breakdown (AAAI-26, by title keyword)

| Area | Share |
| --- | --- |
| LLMs, agents, RAG, prompting, jailbreaks | ~30% |
| Vision, multimodal, diffusion | ~27% |
| Probing, analysis, interpretability, bias, causal | ~20% |
| Applied science and health | ~19% |
| RL, planning, control | ~12% |
| Graphs, GNNs | ~9% |
| Federated, privacy, security | ~6% |

LLM-related work is the largest bucket and grew year over year. Analysis-style papers are well represented.

### Two title archetypes

1. **"NAME: A method that beats baselines."** 41 of 132 AAAI-26 titles use the `ACRONYM:` pattern. Needs a benchmark, baselines, and a results table.
2. **"Empirical finding or probe."** Examples: "Language Models Do Not Embed Numbers Continuously," "Do LLMs Understand Chronology?," "Fine-Tuning Sample Order Matters in Propositional Logical QA," "When Reasoning Collapses: A Depth-Aware Probe," "How Reasoning Influences Intersectional Biases in VLMs." One sharp question, one controlled experiment, one clear answer. Best fit for a three-day timeline.

### Anatomy of an accepted abstract

From reading full PDFs (example saved as `example-aaai26-student-abstract.pdf`):

- Abstract, then a short intro with 3 to 5 citations
- One method section
- One or two results sections with **2 to 4 figures or tables**
- A "Conclusion and Future Work" paragraph that frankly says what is still to do
- 4 to 10 references
- About 1,000 to 1,300 words of body text
- A code or data link in the header helps

The Aurora encoder probing paper ran its entire analysis on 10 samples, said so, and was accepted. Preliminary results with honest limitations are the norm.

Negative and mixed findings were welcomed. "Comparative Analysis of Demonstration Selection Algorithms" (AAAI-25) found some sophisticated methods lose to random selection.

### What to optimize for

A concrete, falsifiable claim about LLM or agent behavior. One experiment runnable with API calls rather than training. A figure that tells the story by itself. An honest future-work paragraph. Novelty of the question matters more than size of the result.

## 4. Agents-specific analysis

Of 219 accepted abstracts, 65 are LLM-related and roughly 25 are genuinely about agents. Four recurring shapes:

1. **Audit an existing agent class.** "Towards Capable and Secure Autonomous Computer-Use Agents" ran Anthropic, OpenAI, and open-source agents on tasks and reported success rates as low as 28% and 100% unauthorized installs. "Obedience or Vigilance?" appended one malicious option E to MCQ questions. No new method, just a clean measurement.
2. **A small multi-agent game with one control knob.** "SIGN: Schema Induced Games for Naming" ran LLM agents in a naming game and showed schemas give 5.8x faster convention agreement.
3. **A simple add-on to an agent, with a percentage.** Negotiation agents plus explicit strategy tools: +16% utility. Self-PR planning and repair for code gen: +5% pass@1. ProRefine prompt-refinement loop: +3 to 37 points.
4. **Comparative evaluation.** Six demonstration-selection methods compared; some lose to random.

Shapes 1, 2, and 4 are doable with API calls in a weekend. Shape 3 needs a benchmark and baselines already in hand.

Other accepted agent papers for reference: Adaptive Coopetition (multi-agent reasoning with coarse verifier), BDI opponent modeling for negotiation, HARK agentic video retrieval, Memory-based advantage shaping for LLM-guided RL, SciDataMAS, RESPOND (LLM-driven disaster agents), PANDA (agent-based dialogue synthesis), AutoToS (AAAI-25, LLM-generated search components), QAagent (AAAI-25, multi-agent unit test generation), ERFSL (AAAI-25, LLM reward function search).

## 5. Assessment of existing candidate work

### context-research (Watchpoint Compaction)

**What it is.** Coding agents compact their context by summarizing old turns. This silently loses two things: a stated rule gets dropped, or a superseded goal gets resurrected. Existing fixes are bespoke: a wider verbatim window helps rules but not goals; a goal-anchoring prompt helps goals but has no rule analogue. Watchpoint Compaction tracks each rule or goal as a key-value pair, lets compaction run normally, then checks afterward whether the compacted context still reflects the current value and patches on a miss.

**The real finding.** How you build the check is the whole story. Self-judgment ("is X still true, yes or no") fires on 0 to 8% of trials where the baseline already violates 22 to 42%. Extraction plus keyword match (make the model restate what it believes, compare to tracked value) drives violations to 0%.

**Data in hand.** About 2,080 completed trials, two open-weight models (Qwen3-30B, MiniMax-01) via OpenRouter, synthetic transcripts, Fisher exact tests. Numbers recomputed from raw JSON and match the draft.

| Method | Violation, 1 compaction, early rule | 3 recursive compactions |
| --- | --- | --- |
| Plain | 42% | 83% |
| Grace buffer | 31% | not run |
| Naive self-judgment watchpoint | 39% | 50% |
| Always pin the rule | 0% | 0% |
| Extraction watchpoint | 0% | 6% |

Goal task (5 scenarios, 1 model, 180 trials): extraction watchpoint ties the specialized anchor prompt at every position.

**The honest twist.** On rules the check patches 97 to 100% of the time, so it buys nothing over always pinning. On goals it patches 50 to 85%, so it is genuinely selective. Self-verification earns its cost on disambiguation problems, not presence problems.

**Best 2-page slice.** Not "a general mechanism." Instead: self-judgment fails to detect compaction loss, extraction-and-match catches it, and it only earns its cost on ambiguous goals.

**Reviewer criticisms to expect.** Synthetic templated filler rather than real tool output. One of two models barely fails at baseline, so the effect is mostly Qwen. Hand-picked keywords per scenario look like engineering. Rule task saturates at 0% for both pin and watchpoint so it cannot rank them. Goal task is single-model, n=20, not significantly better than plain.

**Red flags to resolve.**
- Every citation was found by search and never independently read (the repo's own notes say so). All eight arXiv IDs must be verified.
- All experiment code, results, and the paper are uncommitted working-tree files from 2026-09-23. Commit them.
- Charlie Xue made two early commits. Decide on co-authorship.
- Draft says 864 constraint trials; raw files hold 972. Reconcile.
- Keep the local `.env` out of anything packaged.

**Minimum work for a clean slice.** 2 to 4 days, under $20 API. Second model on the goal task, per-model reporting, entailment check instead of keywords on goals, a token-cost column, restyle two figures, trim to ~8 verified references.

Key paths: `context-research/docs/paper/paper.tex`, `context-research/docs/potential_ideas/pilots/recency-study/`.

### thinking-system (ideation swarm)

**What it is.** Generate 6 retrieval-grounded ideas, pairwise single-call LLM judge plus Elo, pool trim, evolution operators (combine, ground-in-lit, analogical, deliberate divergence), meta-review, 3 rounds, embedding-diversity top-k. Scored with vendored AgentIdeaBench (arXiv 2609.07611): LIT8D rubric, 3 fixed critics, Semantic Scholar prior-art evidence.

**Data in hand.** 39-subfield sweep with paired stats. Final-3 average 7.56 (sd 0.32) vs published Claude Sonnet 5 Active 6.65; 37/39 wins, t=10.19. Elo vs critic score correlation r=0.31 (Spearman 0.20). A single-call baseline on one custom topic scored 8.14 vs final-3 at 8.33.

**Why the headline is currently indefensible.**
- Model confound. Generator ran on Fable 5, which is not on the leaderboard at all. Leaderboard ceiling is Claude Opus 5 at 7.14. Against that, the gain is +0.42 not +0.91, and still attributable to the generator model rather than the architecture.
- Length confound. Ideas average 673 words; benchmark baselines are 80 to 150 words.
- Schema mirrors the rubric (8-field template tracks scoring dimensions).
- Selection is the weak link (r=0.31).

**Best 2-page slice.** "Does pairwise-Elo selection pick what a critic rewards?" with the answer being mostly no. Fits the "sophisticated method barely beats random" archetype.

**Minimum work.** Rerun generator on Sonnet 5 via OpenRouter (`--backend openrouter` exists), enforce 150-word finals, run no-Elo and compute-matched best-of-N ablations, 20-pair human spot check. About $130 API, 2 to 4 days, ~12 min per subfield.

**Red flags.** Not pushed to origin. `.env` with live keys in repo (gitignored). "antiviral resistance" subfield fails deterministically. No README in either package.

### Head-to-head

| | Watchpoint | Elo selection |
| --- | --- | --- |
| Real data | 2,080 trials, 2 models | 39 subfields, 1,287 ideas, 1 model |
| Sharpest question | Self-judgment fails, extraction works | Elo selection barely beats random |
| Fixes before Monday | Verify citations, entailment check, 2nd model on goals | Re-sweep on Sonnet 5, 150-word finals, ablation arms |
| API cost | < $20 | ~$130 |
| Wall-clock risk | Low | Medium |

Watchpoint is the safer bet: least new compute, data already reproducible, story fits the track.

## 6. Fresh agent-domain ideas

Excluded by decision: CLI vs MCP tool surfaces (overlaps MotherDuck work); rule-adherence decay in long trajectories (a control axis of the compaction work; own pilot showed 0/20 violations with no compaction, and Governance Decay found 0% when the constraint survives in context).

All below are API-only probes runnable by Sunday. None appear among the 219 accepted abstracts.

1. **Ask or assume: clarification calibration.** 40 tasks with controlled ambiguity levels. Measure ask-rate vs pick-and-run per model, overlay wrong-assumption rate. Figure: ask-rate vs ambiguity level.
2. **Parallel tool-call dependency errors.** Tool sets where some calls depend on earlier results. Measure how often agents issue dependent calls in parallel with stale or guessed arguments. Figure: error rate vs dependency depth.
3. **Tool description sensitivity.** Same tools, paraphrased descriptions (terse, verbose, misleading, swapped names). Measure tool-selection accuracy. Figure: heatmap by model and variant.
4. **Distractor tool scaling.** Correct tool fixed, add 5/20/50/100 irrelevant tools. Plot accuracy and latency vs tool count.
5. **Overshoot after task completion.** Crisp completion criterion in a sandbox. Count actions after the criterion is met. Figure: overshoot distribution per model.
6. **Budget adherence.** Tell the agent it has N tool calls or tokens. Measure compliance vs phrasing and model.
7. **Sycophantic capitulation on verifiable results.** Correct tool answer, user says "that's wrong." Measure re-run, defer, or fabricate rates.
8. **Plan rigidity under contradicting evidence.** Mid-execution tool result contradicts a planning assumption. Measure revise vs ignore vs rationalize.
9. **Conflicting tool outputs.** Two tools disagree. Which does the agent trust, does it flag the conflict.
10. **Tool-failure recovery.** Inject transient vs permanent errors. Classify retry, work around, hallucinate, give up.

Top picks: 1 and 2 (one obvious figure, deterministic grading, unmistakably "agents"). Idea 5 is the most original.

### 6b. Subfields with a known benchmark plus room for an own method

Sorted by weekend feasibility. Cost assumes a mid-tier or open-weight model via OpenRouter.

**Tier 1: runnable tonight**

| Subfield | Benchmark | Why cheap | Method hook | Accepted analogue |
| --- | --- | --- | --- | --- |
| Tool-use agents | tau-bench (retail, airline), BFCL v3 | Pure API, simulated user, pass^k built in | Tool-result verification, policy-reminder injection, schema-constrained retry | Strategic Tool Negotiator |
| Text embodied agents | ALFWorld, ScienceWorld | Free Python env, ReAct/Reflexion baselines public | Memory variant, subgoal decomposition, failure-driven replanning | Memory-based advantage shaping |
| Classical planning | PlanBench / Blocksworld, Mystery Blocksworld | Tiny prompts, easy grading | LLM-written validator or successor function, verify-then-plan | AutoToS (AAAI-25) |
| Code agents | HumanEval, MBPP, LiveCodeBench | Deterministic tests | Repair loop, plan selection, test-first generation | Self-PR, QAagent |
| Multi-agent math | GSM8K, MATH-500, AIME 2025 | Exact-match grading | Debate topology, verifier weighting, budget-aware agent count | Adaptive Coopetition, ProRefine |
| Agent prompt injection | InjecAgent, AgentDojo | Cheap harness, ASR metric | Tool-output quarantine, provenance tagging, dual-model check | Always Refuse, Hex Injection |

**Tier 2: runnable with more setup or budget**

| Subfield | Benchmark | Notes | Method hook |
| --- | --- | --- | --- |
| Web agents (offline) | Mind2Web | Cached HTML, no browser | Element ranking, DOM pruning |
| Constraint planning | TravelPlanner, NATURAL PLAN | Tool-backed, moderate cost | Constraint scratchpad, solver handoff |
| Multi-agent games | TextArena, Werewolf/Avalon | Free envs, Elo or win-rate | Opponent modeling, communication schema |
| Long-term memory | LongMemEval, LoCoMo | Long contexts cost more; adjacent to compaction work | Memory retrieval or write policy |
| Program search | ARC-AGI-1 eval | Moderate cost, hard | Concept-guided search (ConceptSearch, AAAI-25) |
| GUI grounding | ScreenSpot, Mind2Web-multimodal | Screenshot + click target | Typed action schemas (typesafe-computer-use repo) |

**Not feasible by Monday:** WebArena, SWE-bench, OSWorld, AndroidWorld, MLE-bench (Docker fleets, days of runtime).

**Top three picks:** tau-bench + policy-adherence method; PlanBench + LLM-generated validators; AgentDojo + tool-output provenance defense.

## 6c. Pilot results (2026-09-26)

All runs used cheap Chinese models via OpenRouter. Total spend $12.54. Code and logs are in `src/experiments/`.

**ALFWorld predict-then-act (lead candidate).** All 134 unseen games, temperature 0, max 30 steps.

| Model | Method | Success | Agent calls/game | Tokens/game |
| --- | --- | --- | --- | --- |
| Qwen3-30B | ReAct | 63/134 (47%) | 40.2 | 110k |
| Qwen3-30B | Predict-then-act | 77/134 (57%) | 15.2 | 47k |
| Qwen3-30B | Predict, manipulation-only checks | 65/134 (49%) | 13.3 | 39k |
| Qwen3-30B | Predict, no batching | 54/134 (40%) | 23.2 | 62k |
| DeepSeek-chat | ReAct | 118/134 (88%) | 23.4 | 55k |
| DeepSeek-chat | Predict-then-act | 116/134 (87%) | 10.0 | 27k |

Claim: about 2.4x fewer agent calls and 2x fewer tokens at equal success, on two models. Qwen success gain is p=0.07 (McNemar), DeepSeek p=0.82. The ablation shows the saving comes from batching verified multi-action plans. Manipulation-only prediction mismatches separate failures from successes (0.74 vs 0.57 per judged step), so they work as a failure detector.

**TextArena opponent-belief scratchpad (secondary).** Qwen3-30B, 100 games per matchup. Against a compute-matched two-call reasoning baseline: Liar's Dice 62% win (p=0.02), Kuhn Poker with reshuffle fix 59% (p=0.09), tic-tac-toe control 28%.

**Multi-agent debate (null).** 120 questions. Every gap within 1 to 7 questions; mixed-model debate matched its strongest member.

**Catan LLM negotiator (null).** 100 four-player games per condition. LLM trader 37%, fixed-rule trader 41%, no trading 31%; no significant differences. No catanrl checkpoint exists locally.

## 7. Pre-submission checklist

- [ ] Pick the topic
- [ ] Verify every citation exists and says what is claimed
- [ ] Commit experiment code and results to git
- [ ] Confirm co-authorship (student must be primary; advisors go in acknowledgments)
- [ ] Draft in `authorkit/AuthorKit27/` template, two-column
- [ ] 2 to 4 figures or tables, restyled for two-column width
- [ ] Honest "Conclusion and Future Work"
- [ ] References fit inside 2 pages for submission
- [ ] Prepare supplement PDF or ZIP (code, extra tables)
- [ ] Register on EasyChair, submit before Sept 28 AoE
- [ ] Never include `.env` or API keys in anything packaged

## 8. Sources and local files

- [AAAI-26 proceedings, Vol 40 No 48](https://ojs.aaai.org/index.php/AAAI/issue/view/732) (132 student abstracts)
- [AAAI-25 proceedings, Vol 39 No 28](https://ojs.aaai.org/index.php/AAAI/issue/view/651) (87 student abstracts)
- [AAAI-27 SA call](https://aaai.org/conference/aaai/aaai-27/student-abstract-and-poster-program-call-for-proposals/)
- [AAAI-26 SA call](https://aaai.org/conference/aaai/aaai-26/student-abstract-and-poster-program-call/)
- [AAAI-26 SA program page](https://aaai.org/conference/aaai/aaai-26/aaai-26-student-abstract-and-poster-program/)
- [AAAI-26 paper publication page](https://aaai.org/conference/aaai/aaai-26/paper-publication-and-conference-attendance/) (extra-page policy)
- `docs/prior-student-abstracts.md`: all 219 accepted titles with links
- `docs/example-aaai26-student-abstract.pdf`: Aurora encoder probing paper, structural template
- `authorkit/AuthorKit27/`: LaTeX and Word templates
