"""AUROC per system for each pre-run predictor, cross-system transfer, value beyond length/repo/human difficulty, routing sim."""
import json, os, re, numpy as np
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
I = [json.loads(l) for l in open(f"{HERE}/data/issues.jsonl")]
S = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{HERE}/raw_scores.jsonl")}
LB = json.load(open(f"{HERE}/data/labels.json")); L, C = LB["labels"], LB["cost"]
ids = [r["instance_id"] for r in I]; n = len(ids)
systems = sorted(L, key=lambda s: sum(L[s].values()))
Y = {s: np.array([L[s][i] for i in ids]) for s in systems}
rate = np.mean([Y[s] for s in systems], 0)
Y["SOLVE_RATE>=0.5"] = (rate >= 0.5).astype(int)
targets = systems + ["SOLVE_RATE>=0.5"]
short = lambda s: re.sub(r"^\d+_", "", s).replace("mini-v1.7.0_", "mini_").replace("mini-v1.16.0_", "mini_").replace("mini-v2.0.0_", "mini_").replace("mini-v1.0.0_", "mini_")[:28]
repos = np.array([r["repo"] for r in I]); ur = sorted(set(repos))
txt = [r["problem_statement"] for r in I]
loglen = np.log1p([len(t) for t in txt])
has_code = np.array([("```" in t) or bool(re.search(r"\n(    |\t)\S", t)) for t in txt], float)
has_tb = np.array([bool(re.search(r"Traceback|Error:|Exception", t)) for t in txt], float)
diff_map = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 3}
diff = np.array([diff_map[r["difficulty"]] for r in I], float)
repo_oh = np.array([[r == u for u in ur] for r in repos], float)
jz = np.array([S[i]["jev_zs"] for i in ids]); llm = np.array([S[i]["llm"] if S[i]["llm"] is not None else 0.5 for i in ids])
L2 = {json.loads(l)["instance_id"]: json.loads(l) for l in open(f"{HERE}/raw_llm2.jsonl")}
g41 = np.array([L2[i]["zs"] for i in ids]); G41Q = np.array([[L2[i]["mq"][q] for q in L2[ids[0]]["mq"]] for i in ids])
from datasets import load_dataset
_ds = {r["instance_id"]: r["patch"] for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
plines = np.log1p([sum(1 for l in _ds[i].splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")) for i in ids])
pfiles = np.array([_ds[i].count("diff --git") for i in ids], float)
qn = list(S[ids[0]]["jq"]); JQ = np.array([[S[i]["jq"][q] for q in qn] for i in ids])
def lg(p): p = np.clip(p, .01, .99); return np.log(p / (1 - p))
F = {"len": loglen[:, None], "base(len+code+tb+repo)": np.c_[loglen, has_code, has_tb, repo_oh],
     "jev_mq_head": lg(JQ), "gpt41mini_mq_head": lg(G41Q), "jev_mq+zs+base": np.c_[lg(JQ), lg(jz), loglen, has_code, has_tb, repo_oh],
     "human_diff": diff[:, None], "base+human_diff": np.c_[loglen, has_code, has_tb, repo_oh, diff],
     "base+human_diff+jev": np.c_[loglen, has_code, has_tb, repo_oh, diff, lg(JQ), lg(jz)],
     "ORACLE gold patch size+base+diff": np.c_[loglen, has_code, has_tb, repo_oh, diff, plines, pfiles],
     "ORACLE+jev": np.c_[loglen, has_code, has_tb, repo_oh, diff, plines, pfiles, lg(JQ), lg(jz)]}
def cv(X, y, groups=None, seed=0, ytrain=None):
    ytrain = y if ytrain is None else ytrain
    p = np.zeros(n)
    sp = GroupKFold(5).split(X, y, groups) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    for tr, te in sp:
        m = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - X[tr].mean(0)) / (X[tr].std(0) + 1e-9), ytrain[tr])
        p[te] = m.predict_proba((X[te] - X[tr].mean(0)) / (X[tr].std(0) + 1e-9))[:, 1]
    return p
def cvm(X, y, **k): return np.mean([cv(X, y, seed=s, **k) for s in range(3)], 0) if "groups" not in k else cv(X, y, **k)
res = {"n_issues": n, "systems": {s: float(Y[s].mean()) for s in systems}}
# 1. AUROC table
tab = {}
for t in targets:
    y = Y[t]; row = {"rate": y.mean(), "len(-)": AUC(y, -loglen), "LLM qwen3-30b": AUC(y, llm), "gpt-4.1-mini zs": AUC(y, g41), "Jev zero-shot": AUC(y, jz), "human_diff(-)": AUC(y, -diff)}
    for k, X in F.items():
        if k in ("len", "human_diff"): continue
        row[k] = AUC(y, cvm(X, y))
    row["jev_mq_head[repo-grouped CV]"] = AUC(y, cv(lg(JQ), y, groups=repos))
    row["base(len+code+tb)[repo-grouped]"] = AUC(y, cv(np.c_[loglen, has_code, has_tb], y, groups=repos))
    tab[short(t)] = row
res["auroc"] = tab
cols = list(next(iter(tab.values())))
print(f"{'target':30s}" + "".join(f"{c[:14]:>15s}" for c in cols))
for t, r in tab.items(): print(f"{t:30s}" + "".join(f"{r[c]:15.3f}" for c in cols))
# bootstrap CI for key comparison on SOLVE_RATE>=0.5
y = Y["SOLVE_RATE>=0.5"]; pb = cvm(F["base(len+code+tb+repo)"], y); pj = cvm(F["jev_mq+zs+base"], y); pd = cvm(F["base+human_diff"], y); pdj = cvm(F["base+human_diff+jev"], y)
rng = np.random.default_rng(0); B = []
for _ in range(1000):
    b = rng.integers(0, n, n)
    if y[b].min() == y[b].max(): continue
    B.append((AUC(y[b], pj[b]) - AUC(y[b], pb[b]), AUC(y[b], pdj[b]) - AUC(y[b], pd[b]), AUC(y[b], jz[b]) - AUC(y[b], -loglen[b])))
B = np.array(B); res["boot_delta_ci95"] = {k: [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))] for j, k in enumerate(["jev+base - base", "jev+base+diff - base+diff", "jev_zs - len"])}
print("bootstrap 95% CI of AUROC deltas (median-system target):", res["boot_delta_ci95"])
# 2. correlations
res["spearman"] = {"jev_zs~loglen": spearmanr(jz, loglen)[0], "jev_zs~human_diff": spearmanr(jz, diff)[0], "jev_zs~solve_rate": spearmanr(jz, rate)[0],
                   "llm~solve_rate": spearmanr(llm, rate)[0], "gpt41mini~solve_rate": spearmanr(g41, rate)[0], "jev_one_line~gold_patch_lines": spearmanr(JQ[:, 0], -plines)[0], "jev_multi_file~gold_files": spearmanr(JQ[:, 6], pfiles)[0], "gold_patch_lines~solve_rate": spearmanr(plines, rate)[0], "loglen~solve_rate": spearmanr(loglen, rate)[0], "human_diff~solve_rate": spearmanr(diff, rate)[0],
                   "jev_head_cv~solve_rate": spearmanr(cvm(lg(JQ), y), rate)[0]}
res["spearman"].update({f"q_{q}~solve_rate": spearmanr(JQ[:, j], rate)[0] for j, q in enumerate(qn)})
print({k: round(v, 3) for k, v in res["spearman"].items()})
# within-bucket AUROC (does Jev ZS add within human-difficulty bucket / length quartile)
wb = {}
for name, g in [("human_diff", diff), ("len_quartile", np.digitize(loglen, np.quantile(loglen, [.25, .5, .75])))]:
    for gv in np.unique(g):
        m = g == gv; yy = y[m]
        if 5 < yy.sum() < m.sum() - 5: wb[f"{name}={int(gv)} (n={m.sum()},pos={yy.mean():.2f})"] = {"jev_zs": AUC(yy, jz[m]), "llm": AUC(yy, llm[m]), "len(-)": AUC(yy, -loglen[m])}
res["within_bucket_median_target"] = wb
for k, v in wb.items(): print(k, {a: round(b, 3) for a, b in v.items()})
# 3. transfer: head on Jev MQ(+zs) trained on source-system labels, evaluated on target-system labels (same CV folds)
Xj = np.c_[lg(JQ), lg(jz)]
tr_sys = [s for s in systems if Y[s].mean() > 0.05]
T = {}
for a in tr_sys + ["SOLVE_RATE>=0.5"]:
    T[short(a)] = {}
    p = np.mean([cv(Xj, Y[a], seed=s) for s in range(3)], 0)
    for b in tr_sys: T[short(a)][short(b)] = AUC(Y[b], p)
res["transfer"] = T
print("\ntransfer (row=train labels, col=eval system), jev mq+zs head")
print(" " * 30 + "".join(f"{short(b)[:10]:>11s}" for b in tr_sys))
for a, r in T.items(): print(f"{a:30s}" + "".join(f"{v:11.3f}" for v in r.values()))
diag = np.mean([T[short(s)][short(s)] for s in tr_sys]); off = np.mean([T[short(a)][short(b)] for a in tr_sys for b in tr_sys if a != b])
res["transfer_summary"] = {"diag_mean": diag, "offdiag_mean": off, "median_target_row_mean": np.mean(list(T["SOLVE_RATE>=0.5"].values()))}
print(res["transfer_summary"])
# 4. routing: weak cheap system vs strong expensive system
def route_curve(weak, strong, score_hard):
    fr = np.linspace(0, 1, 11); order = np.argsort(-score_hard); out = []
    for f in fr:
        k = int(round(f * n)); m = np.zeros(n, bool); m[order[:k]] = True
        out.append(float(np.where(m, Y[strong], Y[weak]).mean()))
    return out
R = {}
for weak, strong in [("20250807_mini-v1.7.0_gpt-5-nano", "20260217_mini-v2.0.0_claude-4-6-opus"), ("20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct", "20250726_mini-v1.0.0_claude-sonnet-4-20250514")]:
    yw = Y[weak]; key = f"{short(weak)} -> {short(strong)}"
    preds = {"random": None, "oracle(weak fails)": 1 - yw + 1e-3 * rng.random(n), "len": loglen, "base": 1 - cvm(F["base(len+code+tb+repo)"], yw),
             "LLM qwen": -llm, "gpt-4.1-mini zs": -g41, "Jev zs": -jz, "Jev head(trained on weak)": 1 - cvm(Xj, yw), "Jev head(trained on OTHER systems' solve rate)": None,
             "human_diff": diff + 1e-3 * rng.random(n)}
    others = [s for s in systems if s not in (weak, strong)]
    yo = (np.mean([Y[s] for s in others], 0) >= 0.5).astype(int); preds["Jev head(trained on OTHER systems' solve rate)"] = 1 - cvm(Xj, yo)
    preds["Jev gain = P(strong)-P(weak) heads"] = cvm(Xj, Y[strong]) - cvm(Xj, yw)
    preds["gpt-4.1-mini-mq gain heads"] = cvm(lg(G41Q), Y[strong]) - cvm(lg(G41Q), yw)
    preds["base gain heads"] = cvm(F["base(len+code+tb+repo)"], Y[strong]) - cvm(F["base(len+code+tb+repo)"], yw)
    preds["oracle gain"] = Y[strong] - yw + 1e-3 * rng.random(n)
    R[key] = {}
    for nm, sc in preds.items():
        if sc is None: R[key][nm] = list(np.linspace(yw.mean(), Y[strong].mean(), 11))
        else: R[key][nm] = route_curve(weak, strong, sc)
    cw = np.array([C[weak][i] or 0 for i in ids]); cs = np.array([C[strong][i] or 0 for i in ids])
    R[key]["_mean_cost_per_issue"] = {"weak": float(cw.mean()), "strong": float(cs.mean())}
    print(f"\nrouting {key}  (cost/issue weak ${cw.mean():.3f}, strong ${cs.mean():.3f}); resolved rate at fraction routed 0,10..100%")
    for nm, c in R[key].items():
        if not nm.startswith("_"): print(f"  {nm:48s}" + " ".join(f"{v:.3f}" for v in c) + f"   AUC={np.trapezoid(c, dx=.1):.3f}")
res["routing"] = R
# 5. cost & latency
sc = [S[i] for i in ids]
from statistics import median
import sys; sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots"); import jevlib as J
res["cost_latency_per_1000"] = {"jev_zs_$": J.spend("pre_run_jev_zs") / n * 1000, "jev_mq10_$": J.spend("pre_run_jev_mq") / n * 1000, "llm_qwen_$": (J.spend("pre_run_llm") - J.spend("pre_run_llm2")) / n * 1000, "gpt41mini_zs_$": J.spend("pre_run_llm2_zs") / n * 1000, "gpt41mini_mq_$": J.spend("pre_run_llm2_mq") / n * 1000, "gpt41mini_median_s": median(L2[i]["t_zs"] for i in ids),
    "jev_zs_median_s": median(s["t_jev_zs"] for s in sc), "jev_mq_median_s": median(s["t_jev_mq"] for s in sc), "llm_median_s": median(s["t_llm"] for s in sc)}
print(res["cost_latency_per_1000"])
json.dump(res, open(f"{HERE}/results.json", "w"), indent=1, default=float)
