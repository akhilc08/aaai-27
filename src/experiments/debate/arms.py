"""The five experimental arms. Each returns a per-question record dict (pred, calls, tokens, cost, details)."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from data import instruction, extract, correct
from llm import chat

QWEN = "qwen/qwen3-30b-a3b-instruct-2507"
HETERO = ["qwen/qwen3-30b-a3b-instruct-2507", "deepseek/deepseek-chat", "z-ai/glm-4.5-air"]
T_SAMPLE = 0.7


def _acc(calls):
    return {"calls": len(calls), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls), "cost": sum(c["cost"] for c in calls)}


def _solve(model, q, temperature, meta):
    msgs = [{"role": "user", "content": f"{q['prompt']}\n\n{instruction(q['kind'])}"}]
    r = chat(model, msgs, temperature, meta)
    return msgs + [{"role": "assistant", "content": r["text"]}], r


def _majority(answers, tiebreak=None):
    """Majority vote; None answers ignored; ties broken by `tiebreak` counter then first-seen order."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    cnt = Counter(valid)
    best = max(cnt.values())
    cands = [a for a in cnt if cnt[a] == best]
    if len(cands) > 1 and tiebreak:
        cands.sort(key=lambda a: -tiebreak.get(a, 0))
    return cands[0]


def _debate_msg(others, q):
    parts = "\n\n".join(f"--- Agent {i+1} solution ---\n{t}" for i, t in enumerate(others))
    return ("These are the solutions to the problem from other agents:\n\n" + parts +
            "\n\nUsing the reasoning from other agents as additional advice, examine your solution and theirs step by step, "
            "then give an updated answer. " + instruction(q["kind"]))


def arm1_single_cot(q, meta):
    _, r = _solve(QWEN, q, 0.0, meta)
    pred = extract(r["text"], q["kind"])
    return {"pred": pred, "correct": correct(pred, q["gold"], q["kind"]), **_acc([r]), "raw": [r["text"]]}


def arm2_self_consistency(q, meta, n=5):
    with ThreadPoolExecutor(n) as ex:
        outs = list(ex.map(lambda i: _solve(QWEN, q, T_SAMPLE, {**meta, "sample": i}), range(n)))
    texts = [r["text"] for _, r in outs]
    answers = [extract(t, q["kind"]) for t in texts]
    pred = _majority(answers)
    return {"pred": pred, "correct": correct(pred, q["gold"], q["kind"]), "samples": answers,
            "sample_correct": [correct(a, q["gold"], q["kind"]) for a in answers], **_acc([r for _, r in outs]), "raw": texts}


def _debate(models, q, meta, rounds=2):
    """Du et al. style: each agent keeps its own history; each round it sees others' latest responses and revises."""
    n = len(models)
    with ThreadPoolExecutor(n) as ex:
        init = list(ex.map(lambda i: _solve(models[i], q, T_SAMPLE, {**meta, "agent": i, "round": 0}), range(n)))
    hists = [h for h, _ in init]
    calls = [r for _, r in init]
    latest = [r["text"] for _, r in init]
    per_round = [[extract(t, q["kind"]) for t in latest]]
    for rd in range(1, rounds + 1):
        def step(i):
            others = [latest[j] for j in range(n) if j != i]
            msgs = hists[i] + [{"role": "user", "content": _debate_msg(others, q)}]
            r = chat(models[i], msgs, T_SAMPLE, {**meta, "agent": i, "round": rd})
            return msgs + [{"role": "assistant", "content": r["text"]}], r
        with ThreadPoolExecutor(n) as ex:
            outs = list(ex.map(step, range(n)))
        hists = [h for h, _ in outs]
        calls += [r for _, r in outs]
        latest = [r["text"] for _, r in outs]
        per_round.append([extract(t, q["kind"]) for t in latest])
    return per_round, calls


def arm3_homogeneous_debate(q, meta, models=None):
    models = models or [QWEN] * 3
    per_round, calls = _debate(models, q, meta)
    pred = _majority(per_round[-1])
    return {"pred": pred, "correct": correct(pred, q["gold"], q["kind"]), "models": models,
            "per_round_answers": per_round,
            "per_round_correct": [[correct(a, q["gold"], q["kind"]) for a in rd] for rd in per_round],
            "per_round_majority_correct": [correct(_majority(rd), q["gold"], q["kind"]) for rd in per_round],
            **_acc(calls)}


def arm5_heterogeneous_debate(q, meta):
    return arm3_homogeneous_debate(q, meta, models=HETERO)


def arm4_triggered_debate(q, meta, n=5, threshold=4):
    """5 independent samples; if the top answer has >= threshold votes stop, else one debate round among distinct positions."""
    with ThreadPoolExecutor(n) as ex:
        outs = list(ex.map(lambda i: _solve(QWEN, q, T_SAMPLE, {**meta, "sample": i}), range(n)))
    calls = [r for _, r in outs]
    answers = [extract(r["text"], q["kind"]) for _, r in outs]
    cnt = Counter(a for a in answers if a is not None)
    sc_pred = _majority(answers)
    top = max(cnt.values()) if cnt else 0
    rec = {"samples": answers, "sc_pred": sc_pred, "sc_correct": correct(sc_pred, q["gold"], q["kind"])}
    if top >= threshold or len(cnt) <= 1:
        rec.update({"triggered": False, "pred": sc_pred})
    else:
        # one representative (first sample) per distinct answer; unparsable samples are dropped
        reps = {}
        for (hist, _), a in zip(outs, answers):
            if a is not None and a not in reps:
                reps[a] = hist
        keys = list(reps)
        def step(k):
            others = [reps[o][-1]["content"] for o in keys if o != k]
            msgs = reps[k] + [{"role": "user", "content": _debate_msg(others, q)}]
            return chat(QWEN, msgs, T_SAMPLE, {**meta, "position": k, "round": 1})
        with ThreadPoolExecutor(len(keys)) as ex:
            rev = list(ex.map(step, keys))
        calls += rev
        revised = [extract(r["text"], q["kind"]) for r in rev]
        pred = _majority(revised, tiebreak=cnt)
        rec.update({"triggered": True, "positions": keys, "revised": revised, "pred": pred})
    rec["correct"] = correct(rec["pred"], q["gold"], q["kind"])
    rec.update(_acc(calls))
    return rec


ARMS = {"1": arm1_single_cot, "2": arm2_self_consistency, "3": arm3_homogeneous_debate,
        "4": arm4_triggered_debate, "5": arm5_heterogeneous_debate}
