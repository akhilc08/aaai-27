"""Figure 1: violation rate on range tools when the limit is only in the schema vs only in the description."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from llm import RUNS

R = [json.loads(l) for f in ("p2b.jsonl", "p2b_big.jsonl") for l in open(RUNS / f)]
R = [r for r in R if r.get("kind") == "range" and r["label"] != "ERROR"]
models = list(dict.fromkeys(r["model"] for r in R))
rate = lambda m, c: 100 * sum(r["label"] == "VIOLATE" for r in R if r["model"] == m and r["cond"] == c) / \
    sum(1 for r in R if r["model"] == m and r["cond"] == c)
rows = sorted(((m, rate(m, "schema"), rate(m, "desc")) for m in models), key=lambda x: x[1])
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED})
fig, ax = plt.subplots(figsize=(3.3, 3.1))
for i, (m, s, d) in enumerate(rows):
    ax.plot([d, s], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(d, i, s=22, color=BLUE, marker="o", zorder=3, edgecolors="white", linewidths=0.8)
    ax.scatter(s, i, s=26, color=ORANGE, marker="D", zorder=3, edgecolors="white", linewidths=0.8)
labels = [m.split("/")[1].replace("-instruct", "").replace("-a3b-2507", "").replace("-a22b-2507", "").replace("-0905", "")
          .replace("-24b", "") for m, _, _ in rows]
labels = [l + " †" if "gemini-2.5-flash" == l else l for l in labels]
ax.set_yticks(range(len(rows)), labels, color=INK)
ax.set_xlim(-3, 103)
ax.set_xlabel("Out-of-range calls on range-limited tools (%)", color=INK)
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.scatter([], [], color=ORANGE, marker="D", s=26, label="limit in JSON schema only")
ax.scatter([], [], color=BLUE, marker="o", s=22, label="limit in description only")
ax.legend(loc="lower right", frameon=False, handletextpad=0.2, fontsize=6.5)
fig.tight_layout()
fig.savefig("figs/fig1_schema_vs_desc.pdf")
fig.savefig("figs/fig1_schema_vs_desc.png", dpi=200)
print(rows)
