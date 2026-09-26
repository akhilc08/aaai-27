"""Which JSON-schema constraint keywords reach the model? Ask each model to print its tool schema verbatim; check the
distinctive constraint value survives. One keyword per tool so there is no interference."""
import json
from concurrent.futures import ThreadPoolExecutor
from llm import chat, RUNS
from p2b_scale import MODELS as SMALL

BIG = ["openai/gpt-4.1", "openai/gpt-4o", "anthropic/claude-sonnet-4.5", "qwen/qwen3-235b-a22b-2507",
       "meta-llama/llama-4-maverick", "mistralai/mistral-medium-3.1", "moonshotai/kimi-k2-0905"]
# keyword -> (property schema, distinctive token that must appear in the printout)
KW = {
    "minimum/maximum": ({"type": "integer", "minimum": 37, "maximum": 83}, ["37", "83"]),
    "exclusiveMaximum": ({"type": "integer", "exclusiveMaximum": 91}, ["91"]),
    "multipleOf": ({"type": "integer", "multipleOf": 7}, ["multipleof"]),
    "enum": ({"type": "string", "enum": ["quokka", "narwhal"]}, ["quokka", "narwhal"]),
    "maxLength": ({"type": "string", "maxLength": 23}, ["23"]),
    "pattern": ({"type": "string", "pattern": "^ZX[0-9]{4}$"}, ["zx"]),
    "format": ({"type": "string", "format": "date"}, ["date"]),
    "maxItems": ({"type": "array", "items": {"type": "string"}, "maxItems": 4}, ["maxitems"]),
}
Q = "Do not call any tool. Print the full JSON schema of the configure tool's parameters exactly as you received it, including every keyword."


def one(args):
    m, kw = args
    prop, toks = KW[kw]
    tool = {"type": "function", "function": {"name": "configure", "description": "Configure the device.",
            "parameters": {"type": "object", "properties": {"value": prop}, "required": ["value"]}}}
    try:
        r = chat(m, [{"role": "user", "content": Q}], {"exp": "p2d", "kw": kw}, tools=[tool], max_tokens=300)
        t = r["text"].lower()
        return {"model": m, "kw": kw, "kept": all(x in t for x in toks), "empty": not t.strip(), "text": r["text"]}
    except RuntimeError as e:
        return {"model": m, "kw": kw, "kept": None, "err": str(e)[:200]}


if __name__ == "__main__":
    jobs = [(m, k) for m in SMALL + BIG for k in KW]
    with ThreadPoolExecutor(16) as ex:
        R = list(ex.map(one, jobs))
    with (RUNS / "p2d.jsonl").open("w") as f:
        for r in R:
            f.write(json.dumps(r) + "\n")
    print(f"{'model':26s} " + " ".join(f"{k[:9]:>9s}" for k in KW))
    for m in SMALL + BIG:
        row = {r["kw"]: r for r in R if r["model"] == m}
        cell = lambda r: "ERR" if r["kept"] is None else ("empty" if r.get("empty") else ("kept" if r["kept"] else "DROP"))
        print(f"{m.split('/')[1][:26]:26s} " + " ".join(f"{cell(row[k]):>9s}" for k in KW))
