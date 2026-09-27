"""Forecast metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= i / bins) & ((p < (i + 1) / bins) if i < bins - 1 else (p <= 1))
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def reliability(p, y, bins=5):
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = []
    for i in range(bins):
        m = (p >= i / bins) & ((p < (i + 1) / bins) if i < bins - 1 else (p <= 1))
        if m.any():
            out.append((round(float(p[m].mean()), 2), round(float(y[m].mean()), 2), int(m.sum())))
    return out


def binary(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return {"n": len(y), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auroc": round(float(roc_auc_score(y, p)), 4) if 0 < y.mean() < 1 else None,
            "ece": round(ece(p, y), 4), "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)}


def policy(pols, yidx):
    """pols: list of prob arrays; yidx: chosen index."""
    acc, nll, br, conf, cor = [], [], [], [], []
    for p, y in zip(pols, yidx):
        p = np.clip(np.asarray(p, float), 1e-9, 1)
        p = p / p.sum()
        top = np.flatnonzero(p >= p.max() - 1e-12)  # ties: expected accuracy of a random tie-break
        a = float(y in top) / len(top)
        acc.append(a); nll.append(-np.log(p[y]))
        oh = np.zeros(len(p)); oh[y] = 1; br.append(float(((p - oh) ** 2).sum()))
        conf.append(p.max()); cor.append(a)
    return {"n": len(acc), "acc": round(float(np.mean(acc)), 4), "nll": round(float(np.mean(nll)), 4),
            "brier": round(float(np.mean(br)), 4), "ece_top1": round(ece(conf, cor), 4)}
