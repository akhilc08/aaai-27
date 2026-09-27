"""Plain ModernBERT-base classifier (no typed-question head): pwin = CLS->logit; opp = per-candidate cross-encoder + softmax.
usage: train_mbert.py TASK N_TRAIN TAG   (TASK: pwin | opp_hist | opp_nohist)"""
import os, sys, time, json, pickle, random, resource
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
FMT = os.environ.get("FMT", "gen9randombattle")
MB = os.environ.get("MB", "answerdotai/ModernBERT-base")
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


class Clf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MB, attn_implementation="sdpa")
        self.lin = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **enc):
        h = self.enc(**enc).last_hidden_state[:, 0]
        return self.lin(h).squeeze(-1).float()


def main():
    import memguard
    memguard.start(8000)
    task, ntr, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    d = pickle.load(open(os.path.join(DDIR, f"ds_{FMT}_slim.pkl"), "rb"))
    rows = d["pwin"] if task == "pwin" else [r for r in d["opp"] if r["y_idx"] >= 0 and len(r["cands"]) >= 2]
    sp = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    tok = AutoTokenizer.from_pretrained(MB)
    m = Clf().to(dev)
    m.enc.config.reference_compile = False

    def text(r):
        return txt_of(r, task)

    def fwd(batch):
        if task == "pwin":
            enc = tok([text(r) for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=448).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                return m(**enc), None
        a, b, grp = [], [], []
        for j, r in enumerate(batch):
            for c in cands_of(r):
                a.append(text(r)); b.append(c); grp.append(j)
        enc = tok(a, b, return_tensors="pt", padding=True, truncation="only_first", max_length=448).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            return m(**enc), grp

    def split_logits(z, grp, n):
        out = [[] for _ in range(n)]
        for i, g in enumerate(grp):
            out[g].append(z[i])
        return [torch.stack(o) for o in out]

    t0 = time.time()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.01)
    tr = random.Random(0).sample(sp["train"], ntr)
    bs = 16 if task == "pwin" else 4
    steps = len(tr) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 50) * max(0.05, 1 - i / max(steps, 1)))
    m.train(); run = []
    for i in range(steps):
        batch = tr[i * bs:(i + 1) * bs]
        z, grp = fwd(batch)
        if task == "pwin":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([float(r["y"]) for r in batch], device=dev))
        else:
            ls = split_logits(z, grp, len(batch))
            loss = sum(-torch.log_softmax(l, -1)[r["y_idx"]] for l, r in zip(ls, batch)) / len(batch)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        run.append(loss.item())
        if i % 100 == 0:
            torch.mps.empty_cache() if dev.type == "mps" else None
            print(json.dumps({"step": i, "of": steps, "loss": round(float(np.mean(run[-100:])), 4), "min": round((time.time() - t0) / 60, 1)}), flush=True)
    m.eval()
    preds, pidx = {}, {}
    with torch.no_grad():
        for s, N in (("val", 2000), ("test", 4000)):
            idx = sub_idx(len(sp[s]), 16 if os.environ.get("SMOKE") else N); pidx[s] = idx
            rs = [sp[s][i] for i in idx]; out = []
            for i in range(0, len(rs), 16 if task == "pwin" else 4):
                b = rs[i:i + (16 if task == "pwin" else 4)]
                z, grp = fwd(b)
                if task == "pwin":
                    out += [np.array([0.0, float(x)]) for x in z.cpu()]
                else:
                    out += [l.cpu().numpy() for l in split_logits(z.float(), grp, len(b))]
            preds[s] = out
            print("pred", s, len(idx), round((time.time() - t0) / 60, 1), flush=True)
    ys = [(sp["val"][i]["y"] if task == "pwin" else sp["val"][i]["y_idx"]) for i in pidx["val"]]
    best = None
    for T in np.arange(0.3, 4.01, 0.05):
        nll = -np.mean([np.log(max((np.exp(l / T - (l / T).max()) / np.exp(l / T - (l / T).max()).sum())[y], 1e-9)) for l, y in zip(preds["val"], ys)])
        if best is None or nll < best[0]:
            best = (nll, float(T))
    pickle.dump({"preds": preds, "idx": pidx, "T": best[1], "task": task, "ntr": ntr},
                open(os.path.join(DDIR, f"preds_mbert_{tag}_{FMT}.pkl"), "wb"))
    print("done T", best[1], "min", round((time.time() - t0) / 60, 1), "maxrss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, flush=True)


if __name__ == "__main__":
    main()
