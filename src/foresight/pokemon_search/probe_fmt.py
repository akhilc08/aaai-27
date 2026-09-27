"""Offline check: the HP-variant probe (orig / ahead / behind) on the saved states, using the formatted-facts state.
Same variants as probe_knowledge.py. Also records the narrow nouls."""
import copy, json, pickle, statistics as st, sys
sys.path.insert(0, "."); sys.path.insert(0, "/Users/sickle/Coding/aaai-27/src/pilots")
import jevlib as J, fmt as F

S = [x[0] for x in pickle.load(open("data/c_states.pkl", "rb"))]
AHEAD = ([1.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.3, 0.3])
BEHIND = ([0.0, 0.0, 0.0, 0.0, 0.3, 0.3], [1.0] * 6)


def variant(s, ours, theirs):
    v = copy.deepcopy(s)
    for side, hps in ((0, ours), (1, theirs)):
        for m, h in zip(v.sides[side], hps):
            m.hp = h
    for side in (0, 1):
        if v.active(side).hp <= 0:
            v.active(side).hp = 0.3
    return v


if __name__ == "__main__":
    qs = {}
    for i, s in enumerate(S):
        for k, v in (("orig", s), ("ahead", variant(s, *AHEAD)), ("behind", variant(s, *BEHIND))):
            qs.update(F.leaf_questions(f"s{i}_{k}", v, narrow=True))
    c0 = J.spend("pksearch_fmt_probe")
    a = J.jev(F.CTX, qs, tag="pksearch_fmt_probe")
    out = {"n_positions": len(S), "questions": len(qs)}
    for k in ("orig", "ahead", "behind"):
        out[k] = round(st.mean(a[f"s{i}_{k}"]["noul"] for i in range(len(S))), 3)
        for n in F.NARROW:
            out[f"{k}_{n}"] = round(st.mean(a[f"s{i}_{k}_{n}"]["noul"] for i in range(len(S))), 3)
    out["per_state_pwin"] = {k: [round(a[f"s{i}_{k}"]["noul"], 2) for i in range(len(S))] for k in ("orig", "ahead", "behind")}
    J._cache["t"] = 0
    out["cost"] = round(J.spend("pksearch_fmt_probe") - c0, 5)
    out["example_orig_state"] = F.fact_text(S[0])
    json.dump(out, open("data/probe_fmt.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "example_orig_state"}, indent=1))
    print(out["example_orig_state"])
