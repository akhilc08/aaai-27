"""Small open LLM scored by next-token log-probabilities over option letters (no generation); zero-shot and LoRA.
usage: train_lm.py TASK N_TRAIN TAG   (TASK: pwin | opp_hist | opp_nohist; N_TRAIN=0 -> zero-shot only)
env: LM (default Qwen/Qwen2.5-0.5B-Instruct)"""
import os, sys, time, json, pickle, random, resource
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
FMT = os.environ.get("FMT", "gen9randombattle")
LM = os.environ.get("LM", "Qwen/Qwen2.5-0.5B-Instruct")
LET = "ABCDEFGHIJKLMNOP"
dev = torch.device(os.environ.get("DEV", "mps"))


TEXT = os.environ.get("TEXT", "hand")
DDIR = os.environ.get("INDIR", os.path.join(HERE, "data"))


def raw_tail(raw, nchar=1500):
    head, _, body = raw.partition("\n")
    return head + "\n" + body[-nchar:]


def txt_of(r, task):
    hand = r["text"] + ("\n" + r["htext"] if task == "opp_hist" else "")
    if TEXT == "raw":
        return raw_tail(r["raw"])
    if TEXT == "both":
        return hand + "\n" + raw_tail(r["raw"], 900)
    return hand


def cands_of(r):
    return r["cand_raw"] if TEXT == "raw" else r["cands"]


def sub_idx(n, N, seed=0):
    return sorted(random.Random(seed).sample(range(n), min(N, n)))


def prompt(r, task):
    if task == "pwin":
        return (f"Pokemon singles battle. Position from OUR point of view:\n{txt_of(r, task)}\n"
                "Question: Will we win this battle from this position?\nA) no\nB) yes\nAnswer:"), 2, r["y"]
    opts = "\n".join(f"{LET[k]}) {c}" for k, c in enumerate(cands_of(r)))
    return (f"Pokemon singles battle. Position from OUR point of view:\n{txt_of(r, task)}\n"
            f"Question: Which action will the opponent choose this turn?\n{opts}\nAnswer:"), len(r["cands"]), r["y_idx"]


def main():
    task, ntr, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    d = pickle.load(open(os.path.join(DDIR, f"ds_{FMT}_slim.pkl"), "rb"))
    key = "pwin" if task == "pwin" else "opp"
    rows = d[key] if key == "pwin" else [r for r in d["opp"] if r["y_idx"] >= 0 and 2 <= len(r["cands"]) <= len(LET)]
    if key == "opp":
        rows = [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2]
    sp = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    tok = AutoTokenizer.from_pretrained(LM)
    model = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.bfloat16).to(dev)
    let_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in LET]

    def logits_for(batch_rows, grad=False):
        ps = [prompt(r, task) for r in batch_rows]
        enc = tok([p for p, _, _ in ps], return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=640).to(dev)
        out = model(**enc).logits[:, -1, :].float()
        res = []
        for j, (_, k, y) in enumerate(ps):
            res.append((out[j, let_ids[:k]], y))
        return res

    def predict(rs, bs=8):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(rs), bs):
                out += [l.cpu().numpy() for l, _ in logits_for(rs[i:i + bs])]
                if (i // bs) % 10 == 0:
                    torch.mps.empty_cache() if dev.type == "mps" else None
        return out

    t0 = time.time()
    if ntr > 0:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        tr = random.Random(0).sample(sp["train"], ntr)
        bs = 8
        model.train(); run = []
        for i in range(0, len(tr), bs):
            res = logits_for(tr[i:i + bs], grad=True)
            loss = sum(-torch.log_softmax(l, -1)[y] for l, y in res) / len(res)
            opt.zero_grad(); loss.backward(); opt.step(); run.append(loss.item())
            if (i // bs) % 50 == 0:
                torch.mps.empty_cache() if dev.type == "mps" else None
                print(json.dumps({"step": i // bs, "of": len(tr) // bs, "loss": round(float(np.mean(run[-50:])), 4),
                                  "min": round((time.time() - t0) / 60, 1)}), flush=True)
    preds, pidx = {}, {}
    for s, N in (("val", 1500), ("test", 4000 if key == "pwin" else 4000)):
        idx = sub_idx(len(sp[s]), 16 if os.environ.get("SMOKE") else N)
        pidx[s] = idx
        preds[s] = predict([sp[s][i] for i in idx])
        print("pred", s, len(idx), round((time.time() - t0) / 60, 1), flush=True)
    # temperature on val
    ys = [prompt(sp["val"][i], task)[2] for i in pidx["val"]]
    best = None
    for T in np.concatenate([np.arange(0.2, 1.0, 0.05), np.arange(1.0, 8.01, 0.25)]):
        nll = -np.mean([np.log(max(np.exp(l / T - (l / T).max())[y] / np.exp(l / T - (l / T).max()).sum(), 1e-9)) for l, y in zip(preds["val"], ys)])
        if best is None or nll < best[0]:
            best = (nll, float(T))
    pickle.dump({"preds": preds, "idx": pidx, "T": best[1], "task": task, "lm": LM, "ntr": ntr},
                open(os.path.join(DDIR, f"preds_lm_{tag}_{FMT}.pkl"), "wb"))
    print("done T", best[1], "min", round((time.time() - t0) / 60, 1), "maxrss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, flush=True)


if __name__ == "__main__":
    main()
