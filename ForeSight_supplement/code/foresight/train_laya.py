"""Partial fine-tune of Laya (top-K encoder layers + decision head) with Laya's RLCD proper-scoring loss,
multi-task: P(win) noul, delta-line noul, opponent-action choice. Then temperature-fit on val, predict test.
usage: train_laya.py [K_LAYERS] [N_PWIN] [N_DELTA] [N_OPP] [TAG]"""
import os, sys, time, pickle, random, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laya_util as U
import numpy as np
import torch
from laya.common import proper_reward, QTYPES

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NP, ND, NO = [int(x) for x in sys.argv[2:5]] if len(sys.argv) > 4 else (20000, 8000, 8000)
TAG = sys.argv[5] if len(sys.argv) > 5 else "ft"
FMT = os.environ.get("FMT", "gen9randombattle")
HIST = bool(int(os.environ.get("HIST", "0")))
TASKS = os.environ.get("TASKS", "pwin,delta,opp").split(",")
SEED = int(os.environ.get("SEED", "0"))
random.seed(SEED); torch.manual_seed(SEED)


def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)


TEXT = os.environ.get("TEXT", "hand")
DDIR = os.environ.get("INDIR", os.path.join(HERE, "data"))


def raw_tail(raw, nchar=1500):
    head, _, body = raw.partition("\n")
    return head + "\n" + body[-nchar:]


def txt_of(r, task):
    hand = r["text"] + ("\n" + r["htext"] if (task == "opp" and HIST) else "")
    if TEXT == "raw":
        return raw_tail(r["raw"])
    if TEXT == "both":
        return hand + "\n" + raw_tail(r["raw"], 900)
    return hand


def cands_of(r):
    return r["cand_raw"] if TEXT == "raw" else r["cands"]


def sub_idx(n, N, seed=0):
    return sorted(random.Random(seed).sample(range(n), min(N, n)))


def items_for(ag, task, rows):
    out = []
    for r in rows:
        if task == "pwin":
            it = U.encode(ag, txt_of(r, task), "noul", U.Q_PWIN, max_len=448); it["target"] = [1 - r["y"], r["y"]]
        elif task == "delta":
            it = U.encode(ag, r["text"], "noul", U.Q_DELTA, max_len=448); it["target"] = [1 - r["y"], r["y"]]
        else:
            crit = {f"o{k}": c for k, c in enumerate(cands_of(r))}
            it = U.encode(ag, txt_of(r, task), "choice", U.Q_OPP, crit, max_len=448)
            t = [0.0] * len(r["cands"]); t[r["y_idx"]] = 1.0; it["target"] = t
        it["task"] = task
        out.append(it)
    return out


def main():
    import memguard
    memguard.start(10500)
    d = pickle.load(open(os.path.join(DDIR, f"ds_{FMT}_slim.pkl"), "rb"))
    for k in ("opp",):
        d[k] = [r for r in d[k] if r["y_idx"] >= 0 and len(r["cands"]) >= 2]
    sp = {t: {s: [r for r in d[t] if r["split"] == s] for s in ("train", "val", "test")} for t in d}
    ag = U.load(os.environ.get("BASE_MODEL", "convaiinnovations/laya"))
    m = ag.model
    # ---- zero-shot reference (pretrained Laya, shipped temperatures not applied: raw logits T=1 and shipped T)
    zs = {}
    for task, n in [] if os.environ.get("NOZS") else  ((("pwin", 3000), ("opp", 2000)) if not os.environ.get("SMOKE") else (("pwin", 32),)):
        idx = sub_idx(len(sp[task]["test"]), n)
        rows = [sp[task]["test"][i] for i in idx]
        lg = U.predict_logits(ag, items_for(ag, task, rows), bs=16)
        zs[task] = {"idx": idx, "logits": lg}
        print("zero-shot done", task, rss(), flush=True)
    if zs: pickle.dump(zs, open(os.path.join(DDIR, f"preds_laya_zeroshot{os.environ.get('ZSTAG', '')}_{FMT}.pkl"), "wb"))
    # ---- freeze
    for p in m.parameters():
        p.requires_grad = False
    enc = m.encoder
    if K < 0:  # full fine-tune: every parameter, gradient checkpointing on encoder and head
        for p in m.parameters():
            p.requires_grad = True
        try:
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception as e:
            print("no enc ckpt", e)
        m.head_checkpointing = True
        head_mods = [m.head, m.scorer, m.type_emb, m.act_head]
        head_ids = {id(p) for mod in head_mods for p in mod.parameters()}
        enc_p = [p for p in m.parameters() if id(p) not in head_ids]
        head_p = [p for p in m.parameters() if id(p) in head_ids]
    else:
        train_mods = list(enc.layers[-K:]) + [enc.final_norm, m.head, m.scorer, m.type_emb] if K > 0 else [m.head, m.scorer, m.type_emb]
        for mod in train_mods:
            for p in mod.parameters():
                p.requires_grad = True
        enc_p = [p for mod in train_mods[:K + 1] for p in mod.parameters()] if K > 0 else []
        head_p = [p for mod in train_mods[-3:] for p in mod.parameters()]
    ntr = sum(p.numel() for p in enc_p + head_p)
    print("trainable params", ntr, flush=True)
    enc_lr = float(os.environ.get("ENC_LR", "1e-5" if K < 0 else "2e-5"))
    opt = torch.optim.AdamW([{"params": enc_p, "lr": enc_lr}, {"params": head_p, "lr": 5e-5}], weight_decay=0.01)
    rng = random.Random(SEED)
    tr = []
    for task, n in (("pwin", NP), ("delta", ND), ("opp", NO)):
        if task in TASKS and n > 0:
            tr += items_for(ag, task, rng.sample(sp[task]["train"], n))
    rng.shuffle(tr)
    bs = int(os.environ.get('BS', 8)); steps = (len(tr) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 100) * max(0.05, 1 - i / steps))
    m.train()
    t0, run = time.time(), []
    for i in range(steps):
        batch = tr[i * bs:(i + 1) * bs]
        logits, mm = U.forward(ag, batch, train=True)
        k = logits.shape[1]
        tgt = torch.zeros(len(batch), k, device=logits.device)
        for j, it in enumerate(batch):
            tgt[j, :len(it["target"])] = torch.tensor(it["target"])
        q = torch.softmax(logits.masked_fill(~mm, -1e4), -1)
        qt = torch.tensor([it["qtype"] for it in batch], device=logits.device)
        loss = -proper_reward(q, tgt, qt, mm.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_p + head_p, 1.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if i % 10 == 0:
            torch.mps.empty_cache()
        if i % 50 == 0:
            print(json.dumps({"step": i, "of": steps, "loss": round(float(np.mean(run[-100:])), 4), "min": round((time.time() - t0) / 60, 1), "rss": rss(),
                              "mps_gb": round(torch.mps.current_allocated_memory() / 1e9, 2)}), flush=True)
    m.eval()
    # ---- predict val + test, fit temperatures on val
    preds, pidx = {}, {}
    for task in TASKS:
        for s in ("val", "test"):
            rows = sp[task][s]
            idx = sub_idx(len(rows), 2000 if s == "val" else 4000)
            if os.environ.get("SMOKE"):
                idx = idx[:64]
            pidx[(task, s)] = idx
            preds[(task, s)] = U.predict_logits(ag, items_for(ag, task, [rows[i] for i in idx]), bs=16)
            print("pred", task, s, len(idx), round((time.time() - t0) / 60, 1), flush=True)
    temps = {}
    for task in TASKS:
        rows = [sp[task]["val"][i] for i in pidx[(task, "val")]]
        best = None
        for T in np.arange(0.5, 3.01, 0.05):
            nll = 0.0
            for r, lg in zip(rows, preds[(task, "val")]):
                z = lg / T; z = z - z.max(); p = np.exp(z) / np.exp(z).sum()
                yi = r["y"] if task != "opp" else r["y_idx"]
                nll -= np.log(max(p[yi], 1e-9))
            if best is None or nll < best[0]:
                best = (nll, float(T))
        temps[task] = best[1]
    print("temps", temps, flush=True)
    os.makedirs(U.CKPT, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad and "act_head" not in n}
    torch.save({"params": sd, "temp": temps, "K": K, "base": os.environ.get("BASE_MODEL", "convaiinnovations/laya")}, os.path.join(U.CKPT, f"laya_{TAG}_{FMT}.pt"))
    pickle.dump({"preds": preds, "idx": pidx, "temps": temps, "hist": HIST}, open(os.path.join(DDIR, f"preds_laya_{TAG}_{FMT}.pkl"), "wb"))
    print("done", round((time.time() - t0) / 60, 1), "min, maxrss", rss(), flush=True)


if __name__ == "__main__":
    main()
