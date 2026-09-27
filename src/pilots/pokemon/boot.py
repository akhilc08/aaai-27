exec(open("analyze.py").read().split("def met(")[0])
rng = np.random.default_rng(0); B = sorted(set(g)); idx = {b: np.where(g == b)[0] for b in B}
a, b_ = P["Jev 8 nouls + simple feats LR"], P["simple feats LR"]; d = []
for _ in range(1000):
    ii = np.concatenate([idx[b] for b in rng.choice(B, len(B))]); d.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b_[ii]))
print("AUROC diff (Jev+simple - simple) 95% CI", np.percentile(d, [2.5, 50, 97.5]).round(3))
