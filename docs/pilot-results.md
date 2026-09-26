# Pilot results (2026-09-26)

Total API spend: about $4.20 of the $10 cap. All code is in `src/experiments/pilots/` and the raw data in `runs/*.jsonl` there. Everything went through OpenRouter at temperature 0 unless noted otherwise.

## Verdict

| Idea | Verdict | Why |
|---|---|---|
| 1. Post-completion injection | Drop | The attack success rate is at 100% or 0% depending on the injection wording, so there is no range in which the answered/unanswered difference could show. The one significant effect (qwen3-30b, p=0.001) goes the **opposite** way from the hypothesis: an already-answered question *protects* the agent. |
| 2. Schema sycophancy | **Go, reframed** | Silent clamping, the effect the idea was designed around, is modest. The strong finding is that **constraints stated only in the JSON schema are frequently not enforced**, and there are two distinct causes. See below. |
| 3. Hedged reviewer notes | Drop | Models fix 92–100% of planted errors even with no note, because they just re-solve the problem. The only signal, blunt notes causing false alarms, is the known "Are you sure?" result. |
| 4. Judge self-preference | Drop for now | Only 20 crossed pairs came out of 156 screened problems. llama-8b judges at chance (47% vs 48%). gemma-12b shows 53% vs 40% on n≈35, which is suggestive but far from enough. |

## The finding worth submitting

**Working title:** *Invisible Constraints: Tool-Calling Agents Ignore Limits Declared Only in the JSON Schema*

### Setup
- 40 mock tools (24 with numeric ranges, 16 with enums). Every limit is arbitrary, so it can't be guessed from world knowledge.
- One out-of-range user request per tool, in 2 phrasings.
- The limit is placed in one of 4 ways: nowhere, JSON schema only (`minimum`/`maximum`/`enum`), description text only, or both.
- 17 models, 5,440 agent calls, plus a probe asking each model to recite the limit.

### Results

1. **Pooled across 17 models:** out-of-range calls are 96.8% with no limit, 11.8% with the limit in the schema only, 7.1% in the description only, and 1.6% with both. Schema-only vs both: McNemar 141 vs 2, p≈1e-39. Schema-only vs description-only: 116 vs 52, p≈9e-7. Stating the limit in both places is the only reliable choice.

2. **Mechanism A, constraints stripped in serving.** Gemini-2.5-flash violates 100% of range limits given only in the schema, against 4% when the same limit is in the description.
   - When asked, it can recite 0% of schema range limits but 100% of enums.
   - Asked to print its tool schema, it shows `minimum`, `maximum`, `exclusiveMaximum`, `multipleOf`, `maxLength`, `pattern`, `format` and `maxItems` all missing. Only `enum` survives.
   - This holds on both Google routes (Vertex and AI Studio).
   - The system-prompt fix does nothing (100%→98%): a prompt can't restore a constraint the model never receives.
   - Gemini-2.5-flash-lite, from the same provider, receives the limits and obeys them (2%).

3. **Mechanism B, seen but not enforced.** Some models recite the schema limit 100% of the time when asked, yet still break it:

   | Model | Schema only | Description only |
   |---|---|---|
   | qwen3-30b | 73% | 29% |
   | mistral-medium-3.1 | 42% | 19% |
   | mistral-small-3.2 | 31% | 2% |
   | llama-3.3-70b | 19% | 19% |

   The effect is specific to numeric ranges; enums are respected. It depends on the model, not the host: qwen shows the same gap on two providers. It shrinks with scale: qwen3-235b is at 4%.

4. **Replication.** At temperature 0.7 with 2 samples, the numbers hold: qwen 72% vs 26%, gemini 99% vs 5%, mistral-small 33% vs 1%, mistral-medium 40% vs 21%.

5. **Mitigation.** One system-prompt sentence ("check every argument against the JSON schema constraints") helps the models that can see the limit: llama 19%→0%, mistral-small 31%→10%, qwen 73%→38%. It does not help Gemini, which never receives the limit.

6. **What happens after a substitution.**
   - Up front: when agents quietly swap in an in-range value, 252 of 619 replies (41%) don't tell the user.
   - After a validation error: 155 of 1,321 rejected calls were retried with a substituted value, and 40% of those were not disclosed. Examples: llama-4-maverick in 25 of 40 retries; gpt-4o-mini in 21 of 23.
   - The judge's "false claim" label was too strict. Hand-checking shows most of those replies state the new value ("set to high priority" when the user asked for urgent) without flagging the change, so I report them only as undisclosed.
   - One real misstatement: flash-lite set priority to `high` and replied "I have marked the ticket as urgent."

### Figure
`src/experiments/pilots/figs/fig1_schema_vs_desc.pdf`: a dumbbell plot per model showing the violation rate on range tools, schema-only vs description-only.

### Caveats to state in the paper
- Mock tools, single-turn requests.
- 48 range items per model per condition.
- Asking the model to recite or print its schema shows what it can see, but it does not show whether the stripping happens in OpenRouter's translation or in Google's own serving stack. That needs a direct Gemini API check.
- The disclosure labels come from a gpt-4.1-mini judge, spot-checked by hand.
- Open models are served by third-party providers.

### Before the Sep 28 deadline
1. Check the Gemini stripping on Google's own API, without OpenRouter, to pin down where it happens.
2. Draft the 2-page abstract around results 1–3 and 5, with Figure 1 and a small table of the two causes.
