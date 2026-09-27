exec(open("analyze.py").read().split("P = {")[0])
names = ["opp_hp","my_hp","opp_status","opp_boost","my_boost","opp_bench_alive","opp_unrevealed","n_moves","turn","prev_sw","prev_none","opp_hazards"]
for j, n in enumerate(names): print(n, round(roc_auc_score(y, Xs[:, j]), 3))
for j, n in enumerate(["p_sw"] + FQ): print(n, round(roc_auc_score(y, np.hstack([Xj0, Xjf])[:, j]), 3))
print(np.corrcoef(Xj0[:,0], Xs[:,0])[0,1], "corr jev p_sw vs opp hp")
