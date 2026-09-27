"""Generative LLM (MLX, local) as the opponent model: prompt with state + history + lettered options,
read P(option) from the next-token distribution over option letters (no sampling). Optional LoRA adapter.
Also a generate-then-parse variant (greedy letter, smoothed).
usage (offline eval): mlx_opp.py MODEL N_TEST TAG [ADAPTER_DIR]"""
import os, sys, time, json, pickle, random, threading
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LET = "ABCDEFGHIJKLMNOP"
_LOCK = threading.Lock()


def prompt_text(state_text, htext, cands):
    opts = "\n".join(f"{LET[k]}) {c}" for k, c in enumerate(cands))
    return ("You are an expert Pokemon Showdown player predicting your opponent. Singles random battle; "
            "facts below are from OUR point of view.\n" + state_text + ("\n" + htext if htext else "") +
            f"\nWhich action will the opponent choose this turn?\n{opts}\nAnswer with the letter only.")


class MLXLM:
    def __init__(self, model_id, adapter=None):
        from mlx_lm import load
        self.model, self.tok = load(model_id, adapter_path=adapter) if adapter else load(model_id)
        self.let_ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in LET]
        self.let_ids_sp = [self.tok.encode(" " + c, add_special_tokens=False)[0] for c in LET]

    def chat(self, p):
        return self.tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) + "Answer: "

    def option_logits(self, p, k):
        import mlx.core as mx
        ids = self.tok.encode(self.chat(p), add_special_tokens=False)
        with _LOCK:
            out = self.model(mx.array(ids)[None])[0, -1].astype(mx.float32)
            lg = np.array(out)
        a = lg[self.let_ids[:k]]; b = lg[self.let_ids_sp[:k]]
        return np.logaddexp(a, b), len(ids)

    def generate_letter(self, p, k):
        from mlx_lm import generate
        with _LOCK:
            txt = generate(self.model, self.tok, self.chat(p), max_tokens=3, verbose=False)
        txt = txt.strip().upper()
        return next((LET.index(ch) for ch in txt if ch in LET[:k]), None)


class MLXOpp:
    """Opponent model for evals.Ev: policies for opponent nodes from LLM option-letter probabilities."""
    def __init__(self, model_id, adapter=None, hist=True, T=1.0):
        import feats as X, fmt as F
        self.X, self.F = X, F
        self.lm, self.hist, self.T = MLXLM(model_id, adapter), hist, T

    def __call__(self, reqs, hist=None):
        out = []
        for s, side, acts, _ in reqs:
            cands = [self.F.act_label(s, 1, a) for a in acts][:len(LET)]
            p = prompt_text(self.X.state_text(s), hist[1] if (self.hist and hist) else "", cands)
            lg, _ = self.lm.option_logits(p, len(cands))
            z = np.exp((lg - lg.max()) / self.T)
            out.append(z / z.sum())
        return out


def main():
    sys.path.insert(0, HERE)
    import metrics as Me
    model_id, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    adapter = sys.argv[4] if len(sys.argv) > 4 else None
    d = pickle.load(open(os.path.join(HERE, "data", "ds_gen9randombattle_slim.pkl"), "rb"))
    rows = [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2]
    te = [r for r in rows if r["split"] == "test"]; va = [r for r in rows if r["split"] == "val"]
    idx = sorted(random.Random(0).sample(range(len(te)), 4000))[:n]   # prefix of the common 4000-row test subset
    vidx = sorted(random.Random(0).sample(range(len(va)), 2000))[:max(200, n // 4)]
    lm = MLXLM(model_id, adapter)
    res, t0, ntok = {}, time.time(), []
    for split, ids, src in (("val", vidx, va), ("test", idx, te)):
        lgs, gens = [], []
        for i in ids:
            r = src[i]
            p = prompt_text(r["text"], r["htext"], r["cands"])
            lg, nt = lm.option_logits(p, len(r["cands"])); lgs.append(lg); ntok.append(nt)
            if split == "test" and os.environ.get("GEN") and len(gens) < 300:
                gens.append(lm.generate_letter(p, len(r["cands"])))
        res[split] = {"idx": ids, "logits": lgs, "gen": gens}
        print(split, len(ids), round(time.time() - t0, 1), flush=True)
    per = (time.time() - t0) / (len(idx) + len(vidx))
    ys = [va[i]["y_idx"] for i in vidx]
    def nll(T):
        return -np.mean([np.log(max(np.exp(l / T - (l / T).max())[y] / np.exp(l / T - (l / T).max()).sum(), 1e-9))
                         for l, y in zip(res["val"]["logits"], ys)])
    best = min((nll(T), T) for T in np.arange(0.3, 8.01, 0.1))
    T = float(best[1])
    yt = [te[i]["y_idx"] for i in idx]
    sm = lambda l, t: (lambda z: z / z.sum())(np.exp((l - l.max()) / t))
    m_raw = Me.policy([sm(l, 1.0) for l in res["test"]["logits"]], yt)
    m_T = Me.policy([sm(l, T) for l in res["test"]["logits"]], yt)
    out = {"model": model_id, "adapter": adapter, "n": len(idx), "T": T, "raw": m_raw, "temp_scaled": m_T,
           "sec_per_forecast": round(per, 3), "mean_prompt_tokens": round(float(np.mean(ntok)), 1)}
    if res["test"]["gen"]:
        g = res["test"]["gen"]; yg = yt[:len(g)]
        acc = np.mean([gi == y for gi, y in zip(g, yg)]); parse = np.mean([gi is not None for gi in g])
        out["generate_then_parse"] = {"n": len(g), "acc": round(float(acc), 4), "parse_rate": round(float(parse), 3)}
    print(json.dumps(out), flush=True)
    pickle.dump(dict(out, res=res), open(os.path.join(HERE, "data", f"preds_mlx_{tag}.pkl"), "wb"))


if __name__ == "__main__":
    main()
