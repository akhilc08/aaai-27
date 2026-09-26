# Pilot: do LLM-swarm herding effects need an LLM in every agent? (Jev vs cheap LLM)

**Question.** Does a Jev typed-choice agent (upvote/downvote/ignore, action sampled from calibrated probs) reproduce Muchnik, Aral & Taylor (Science 2013) herding the way an LLM agent does, and what does it cost?
Human result (from memory, not re-checked): a +1 vote raised the chance of an up-vote by ~32% and final mean rating by ~25%. A -1 vote raised down-votes, but other users corrected it, so final ratings were not lower.

**Setup.** 60 LLM-generated Reddit comments on 6 topics and 150 LLM-generated personas. Cheap LLM is qwen3-30b-a3b at temperature 1.0, one-word answer. Jev uses `choice` and samples the action (argmax also recorded).
(1) **Probe.** The same (persona, comment) pair is shown with score +1, 0, -1 and hidden. LLM: 150 pairs = 600 decisions; Jev: 300 pairs = 1,200 decisions.
(2) **Dynamic sim.** 20 comments per treatment, 2 random viewers per comment per round, scores update after each round. LLM: 8 rounds = 960 decisions; Jev: 25 rounds = 3,000 decisions, sampled and argmax. Arms are compared at the same 8 rounds. Bootstrap 95% CIs.

| Probe (paired) | LLM | Jev sampled | Jev probs |
|---|---|---|---|
| P(up) at +1 / 0 / -1 / hidden | .55/.51/.43/.50 | .43/.45/.27/.50 | .45/.46/.28/.52 |
| dUp(+1 vs 0) | +.03 [-.01,+.08] (+6.5% rel.) | -.02 [-.08,+.05] | -.019 [-.023,-.015] |
| dDown(-1 vs 0) | +.09 [+.04,+.15] | +.22 [+.15,+.29] | +.21 [+.20,+.22] |

| Dynamic, 8 rounds: mean final score (+1 / 0 / -1) | LLM | Jev sampled | Jev argmax |
|---|---|---|---|
| mean final | 5.5 / 6.2 / 2.1 | 2.3 / 5.6 / -5.4 | 5.6 / 7.7 / -4.4 |
| P(final>0) | .75/.85/.55 | .55/.80/.20 | .65/.80/.35 |
| final(+1)-final(0) | -0.7 [-5.8,4.3] | -3.4 [-7.8,1.3] | -2.1 [-8.3,4.1] |

Diversity: per-comment action entropy across viewers is LLM 0.97 bits, Jev sampled 1.16, Jev argmax 0.68.
Cost per 1k decisions: LLM $0.013 vs Jev $0.019, so the LLM:Jev cost ratio is about 0.7 (the cheap LLM with a short prompt is **cheaper**). Wall time at 12 threads: LLM ~155 ms/decision vs Jev ~25 ms, so Jev is about 6x faster. Total spend $0.16.

**Verdict: NOT PROMISING** (for the "cheap drop-in that reproduces herding" story).
- Neither arm shows positive herding. Both show negative herding.
- Jev's negative herding is about 2x the LLM's and is not corrected: -1 comments stay negative, which is the opposite of the human result.
- There is no cost advantage over a small LLM, only a latency advantage.

**A full experiment would need:**
- many more comments, or repeated worlds (final scores are dominated by comment quality)
- a real comment dataset
- a stronger or OASIS-style LLM baseline with longer prompts, where Jev's cost advantage could appear
- the OASIS feed and recommendation mechanics
- ≥5 seeds per arm
