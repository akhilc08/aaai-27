"""Replication of the headline effect: range tools, schema-only vs desc-only, temperature 0.7, 2 samples,
and qwen3-30b pinned to a different provider than the main run."""
import json
from concurrent.futures import ThreadPoolExecutor
from llm import _client, RUNS, USAGE, _lock
from p2_tools import TOOLS
from p2b_scale import build_tool, classify, SYS
import time

ARMS = [("qwen/qwen3-30b-a3b-instruct-2507", None), ("qwen/qwen3-30b-a3b-instruct-2507", "deepinfra"),
        ("google/gemini-2.5-flash", None), ("mistralai/mistral-small-3.2-24b-instruct", None),
        ("mistralai/mistral-medium-3.1", None)]
RANGE = [i for i, t in enumerate(TOOLS) if t[4] == "range"]


def one(a):
    (m, prov), ti, cond, pi, s = a
    t = TOOLS[ti]
    eb = {"reasoning": {"enabled": False}, "usage": {"include": True}}
    if prov:
        eb["provider"] = {"order": [prov], "allow_fallbacks": False}
    for k in range(4):
        try:
            r = _client.chat.completions.create(model=m, temperature=0.7, max_tokens=400, extra_body=eb,
                    tools=[build_tool(t, cond)], messages=[{"role": "system", "content": SYS}, {"role": "user", "content": t[8][pi]}])
            msg = r.choices[0].message
            calls = [{"name": c.function.name, "args": json.loads(c.function.arguments or "{}")} for c in msg.tool_calls or []]
            with _lock, USAGE.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "model": m, "cost": float(getattr(r.usage, "cost", 0) or 0), "exp": "p2e"}) + "\n")
            return {"model": m, "prov": prov, "served": getattr(r, "provider", None), "tool": t[0], "cond": cond, "phr": pi,
                    "s": s, "label": classify(t, calls)[0]}
        except Exception as e:  # noqa: BLE001
            err = str(e)[:150]
            time.sleep(2)
    return {"model": m, "prov": prov, "tool": t[0], "cond": cond, "label": "ERROR", "err": err}


jobs = [(a, ti, c, pi, s) for a in ARMS for ti in RANGE for c in ("schema", "desc") for pi in range(2) for s in range(2)]
with ThreadPoolExecutor(16) as ex:
    R = list(ex.map(one, jobs))
with (RUNS / "p2e.jsonl").open("w") as f:
    for r in R:
        f.write(json.dumps(r) + "\n")
for a in ARMS:
    rs = [r for r in R if (r["model"], r["prov"]) == a]
    v = {c: [r["label"] == "VIOLATE" for r in rs if r["cond"] == c and r["label"] != "ERROR"] for c in ("schema", "desc")}
    print(f"{a[0].split('/')[1][:24]:25s} {str(a[1]):10s} served={set(r.get('served') for r in rs)} "
          f"schema {100*sum(v['schema'])/max(1,len(v['schema'])):5.1f}% (n={len(v['schema'])})  desc {100*sum(v['desc'])/max(1,len(v['desc'])):5.1f}% (n={len(v['desc'])}) err={sum(r['label']=='ERROR' for r in rs)}")
