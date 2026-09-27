"""Laya helpers: load (optionally with fine-tuned overlay), encode typed questions, batched forward."""
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.2")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.15")
import numpy as np
import torch
from laya.common import collate_items

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
Q_PWIN = ("Pokemon singles battle; facts precomputed by a damage calculator from OUR point of view. "
          "Will we (Us) win this battle from this position?")
Q_DELTA = ("Pokemon singles battle; the root position facts are followed by what changed over the next two turns "
           "along one line of play. Will we (Us) win this battle if the line happens?")
Q_OPP = "Pokemon singles battle; facts from OUR point of view. Which action will the opponent (Them) choose this turn?"


def load(overlay=None, device="mps", half=False):
    import laya
    base = "convaiinnovations/laya"
    if overlay and not overlay.endswith(".pt"):
        base, overlay = overlay, None
    if overlay:
        base = torch.load(overlay, map_location="cpu").get("base", base)
    sub = None
    if base.endswith("/multilingual"):
        base, sub = base[: -len("/multilingual")], "multilingual"
    ag = laya.Agent(base, device=device, subfolder=sub)
    if overlay:
        sd = torch.load(overlay, map_location="cpu")
        miss = ag.model.load_state_dict(sd["params"], strict=False)
        ag.ft_temp = sd.get("temp", {})
    else:
        ag.ft_temp = {}
    ag.model.eval()
    return ag


def encode(ag, text, qtype, ins, crit=None, max_len=384):
    q = {"t": qtype, "ins": ins, "crit": crit}
    return ag._encode_state(text, ["q"], {"q": q}, max_len=max_len)[0]


def forward(ag, items, train=False):
    b = collate_items([items], ag.tok.pad_token_id)
    dev = ag.device
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16 if train else torch.float16, enabled=dev.type in ("mps", "cuda")):
        logits, _ = ag.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                             b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits.float(), b["marker_mask"].to(dev)


@torch.no_grad()
def predict_logits(ag, items, bs=32):
    out = []
    order = np.argsort([len(it["ids"]) for it in items])
    res = [None] * len(items)
    for i in range(0, len(items), bs):
        idx = order[i:i + bs]
        lg, mm = forward(ag, [items[j] for j in idx])
        lg = lg.masked_fill(~mm, -1e4).cpu().numpy()
        for r, j in enumerate(idx):
            res[j] = lg[r, :len(items[j]["markers"])]
        if ag.device.type == "mps" and (i // bs) % 4 == 0:
            torch.mps.empty_cache()
    return res


def noul_p(logits, T=1.0):
    return np.array([1 / (1 + np.exp(-(l[1] - l[0]) / T)) for l in logits])
