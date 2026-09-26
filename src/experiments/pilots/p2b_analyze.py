"""Tables for the schema-visibility scale-up: violation rate per model x condition, paired tests, probe recitation."""
import json, math, sys
from collections import defaultdict
from scipy.stats import binomtest
from llm import RUNS

tag = sys.argv[1] if len(sys.argv) > 1 else ""
R = [json.loads(l) for l in open(RUNS / f"p2b{tag}.jsonl")]
P = [json.loads(l) for l in open(RUNS / f"p2b_probe{tag}.jsonl")]
models = list(dict.fromkeys(r["model"] for r in R))
short = lambda m: m.split("/")[1][:24]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    """a, b: dict key->bool. Exact McNemar on discordant pairs."""
    keys = a.keys() & b.keys()
    n10 = sum(a[k] and not b[k] for k in keys)
    n01 = sum(b[k] and not a[k] for k in keys)
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, p


print(f"{'model':25s} {'kind':5s} | VIOLATE% none / schema / desc / both | schema-vs-desc b/c p | NO_CALL% sch/desc | SUBST sch/desc (silent) | ERR")
for kind in ("all", "range", "enum"):
    print(f"--- {kind}")
    for m in models:
        rs = [r for r in R if r["model"] == m and (kind == "all" or r.get("kind") == kind)]
        v = {c: {(r["tool"], r["phr"]): r["label"] == "VIOLATE" for r in rs if r["cond"] == c and r["label"] != "ERROR"} for c in ("none", "schema", "desc", "both")}
        pct = {c: 100 * sum(v[c].values()) / max(1, len(v[c])) for c in v}
        b, c_, p = mcnemar(v["schema"], v["desc"])
        nc = {c: 100 * sum(r["label"] == "NO_CALL" for r in rs if r["cond"] == c) / max(1, sum(r["cond"] == c for r in rs)) for c in ("schema", "desc")}
        sub = {c: [r for r in rs if r["cond"] == c and r["label"] == "SUBSTITUTE"] for c in ("schema", "desc")}
        sil = {c: sum(r.get("disclosed") is False for r in sub[c]) for c in sub}
        err = sum(r["label"] == "ERROR" for r in rs)
        print(f"{short(m):25s} {kind:5s} | {pct['none']:5.0f} {pct['schema']:6.0f} {pct['desc']:6.0f} {pct['both']:6.0f} "
              f"| {b:3d}/{c_:<3d} p={p:.1e} | {nc['schema']:4.0f} {nc['desc']:4.0f} | {len(sub['schema'])}({sil['schema']}) {len(sub['desc'])}({sil['desc']}) | {err}")

# pooled
v = defaultdict(dict)
for r in R:
    if r["label"] != "ERROR":
        v[r["cond"]][(r["model"], r["tool"], r["phr"])] = r["label"] == "VIOLATE"
for c in ("none", "schema", "desc", "both"):
    k, n = sum(v[c].values()), len(v[c])
    lo, hi = wilson(k, n)
    print(f"POOLED {c:6s} violate {k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]")
print("POOLED schema vs desc", mcnemar(v["schema"], v["desc"]), " schema vs both", mcnemar(v["schema"], v["both"]),
      " none vs schema", mcnemar(v["none"], v["schema"]))

subs = [r for r in R if r["label"] == "SUBSTITUTE"]
print(f"SUBSTITUTIONS total {len(subs)}, silent (judge) {sum(r.get("disclosed") is False for r in subs)}")

print("\nPROBE: % tools where model recites the constraint (schema-only) vs guess baseline (none)")
for m in models:
    ps = [p for p in P if p["model"] == m and p.get("recited") is not None]
    s = [p["recited"] for p in ps if p["cond"] == "schema"]
    n = [p["recited"] for p in ps if p["cond"] == "none"]
    sr = [p["recited"] for p in ps if p["cond"] == "schema" and p["kind"] == "range"]
    se = [p["recited"] for p in ps if p["cond"] == "schema" and p["kind"] == "enum"]
    print(f"{short(m):25s} schema {100*sum(s)/max(1,len(s)):5.0f}% (range {100*sum(sr)/max(1,len(sr)):4.0f}, enum {100*sum(se)/max(1,len(se)):4.0f})  none {100*sum(n)/max(1,len(n)):4.0f}%")

# the key cross: among tools where the model CAN recite the schema limit, how often does it still violate in schema-only?
print("\nSEEN-BUT-IGNORED: violate rate in schema-only, split by whether the model recited that tool's limit")
for m in models:
    rec = {p["tool"]: p["recited"] for p in P if p["model"] == m and p["cond"] == "schema" and p.get("recited") is not None}
    rs = [r for r in R if r["model"] == m and r["cond"] == "schema" and r["label"] != "ERROR" and r["tool"] in rec]
    seen = [r["label"] == "VIOLATE" for r in rs if rec[r["tool"]]]
    unseen = [r["label"] == "VIOLATE" for r in rs if not rec[r["tool"]]]
    print(f"{short(m):25s} recited: {100*sum(seen)/max(1,len(seen)):5.0f}% of {len(seen):3d} | not recited: {100*sum(unseen)/max(1,len(unseen)):5.0f}% of {len(unseen):3d}")
