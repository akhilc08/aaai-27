"""Idea 4: reconstruction-NLL gap as a reference-free compaction fidelity probe.

For every compacted context produced in exp2, score the original policy sentence
under a small frozen local LM (via mlx-lm) conditioned on (a) the full raw
transcript and (b) the compacted context. gap = NLL_summary - NLL_full, in nats
per token of the policy sentence. Then test whether gap predicts (i) the rule
keyword being absent from the summary and (ii) the agent actually violating.

Runs fully offline after the model download; no API calls.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

IN = os.path.join(C.RESULTS_DIR, "exp2_ratchet.jsonl")
OUT = os.path.join(C.RESULTS_DIR, "exp4_nll_gap.jsonl")
MODEL_ID = os.environ.get("NLL_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-bf16")
PREFIX = ("Below is an autonomous coding agent's session context.\n\n{context}\n\n"
          "The operator's standing policy for this session, restated verbatim:\n")


def load_model():
    from mlx_lm import load
    return load(MODEL_ID)


def nll_of_target(model, tokenizer, prefix, target):
    """Mean negative log-likelihood (nats/token) of `target` given `prefix`."""
    import mlx.core as mx
    p_ids = tokenizer.encode(prefix)
    t_ids = tokenizer.encode(target, add_special_tokens=False) if hasattr(tokenizer, "encode") else tokenizer.encode(target)
    ids = mx.array([p_ids + t_ids])
    logits = model(ids)[0]  # [T, V]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # token i is predicted by position i-1
    T = len(p_ids)
    tgt = mx.array(t_ids)
    pos = mx.arange(T - 1, T - 1 + len(t_ids))
    lp = logp[pos, tgt]
    mx.eval(lp)
    return float(-lp.mean()), len(t_ids)


def main():
    rows = [r for r in C.load_results(IN) if r["regime"] != "none" and "tail" in r]
    if not rows:
        print("exp4: no exp2 results yet; skipping")
        return
    KF = ["regime", "rounds", "budget", "ctype", "scenario", "backbone", "rep"]
    done = {tuple(r[k] for k in KF) for r in C.load_results(OUT)}
    todo = [r for r in rows if tuple(r[k] for k in KF) not in done]
    print(f"exp4: {len(done)} done, {len(todo)} to score with {MODEL_ID}")
    if not todo:
        return
    model, tok = load_model()
    full_cache = {}
    for i, r in enumerate(todo):
        sid, ctype = r["scenario"], r["ctype"]
        policy = C.policy_for(sid, ctype)
        turns = [policy] + C.POOL[:r["total_turns"] - 1]
        ck = (sid, ctype, r["total_turns"])
        if ck not in full_cache:
            full_cache[ck] = nll_of_target(model, tok, PREFIX.format(context=C.render(turns)), policy)[0]
        nll_full = full_cache[ck]
        ctx = C.render_context(r["summary"], r["tail"])
        nll_sum, ntok = nll_of_target(model, tok, PREFIX.format(context=ctx), policy)
        # control: NLL with no context at all (how surprising is the sentence on its own?)
        nll_none = nll_of_target(model, tok, PREFIX.format(context="(none)"), policy)[0]
        out = {k: r[k] for k in KF + ["decision", "rule_kw_present", "total_turns"]}
        out.update({"nll_full": nll_full, "nll_summary": nll_sum, "nll_none": nll_none,
                    "gap": nll_sum - nll_full, "target_tokens": ntok})
        C.append_result(OUT, out)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"exp4: {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
