"""Pilot 4: do pairwise judges endorse wrong responses that match their OWN wrong answer?

Stages (cached in runs/): screen judges on MATH-500 integer problems (levels 4-5), have a neutral
writer produce a correct solution and solutions ending at chosen wrong answers, then run a crossed
pairwise judging design with both A/B orders.
"""
import json, random, re
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from llm import chat, RUNS

JUDGES = ["meta-llama/llama-3.1-8b-instruct", "google/gemma-3-12b-it", "mistralai/mistral-small-3.2-24b-instruct"]
WRITER = "openai/gpt-4.1-mini"
SCREEN = RUNS / "p4_screen.jsonl"
WRITE = RUNS / "p4_write.jsonl"
OUT = RUNS / "p4.jsonl"
POOL = 16


def boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if not m:
        return None
    s = m[-1].replace(",", "").replace("$", "").strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def problems():
    d = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"id": x["unique_id"], "q": x["problem"], "gold": str(int(x["answer"]))}
            for x in d if x["answer"].strip().lstrip("-").isdigit() and x["level"] >= 4]


def screen(ps):
    if SCREEN.exists():
        return load(SCREEN)
    def one(args):
        p, j = args
        r = chat(j, [{"role": "user", "content": p["q"] + "\n\nSolve step by step. Put the final answer in \\boxed{}."}],
                 {"exp": "p4", "stage": "screen"}, max_tokens=2000)
        return {"id": p["id"], "judge": j, "pred": boxed(r["text"])}
    with ThreadPoolExecutor(POOL) as ex:
        rows = list(ex.map(one, [(p, j) for p in ps for j in JUDGES]))
    SCREEN.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def write_solution(p, target, correct):
    instr = ("Write a clear, step-by-step solution to this problem." if correct else
             f"Write a plausible, confident step-by-step solution to this problem whose final answer is {target}. "
             "Do not mention that it might be wrong; write it exactly as a normal solution.")
    r = chat(WRITER, [{"role": "user", "content": f"{p['q']}\n\n{instr} Keep it under 250 words and end with "
                       f"\\boxed{{{target}}}."}], {"exp": "p4", "stage": "write"}, temperature=0.3, max_tokens=900)
    return r["text"]


JUDGE_PROMPT = """Problem:
{q}

--- Response A ---
{a}

--- Response B ---
{b}

Exactly one response is correct. Which one? Think briefly, then end with "Verdict: A" or "Verdict: B"."""


def judge(j, p, correct_sol, wrong_sol, wrong_first):
    a, b = (wrong_sol, correct_sol) if wrong_first else (correct_sol, wrong_sol)
    r = chat(j, [{"role": "user", "content": JUDGE_PROMPT.format(q=p["q"], a=a, b=b)}],
             {"exp": "p4", "stage": "judge"}, max_tokens=1200)
    m = re.findall(r"Verdict:\s*\**\s*([AB])", r["text"])
    if not m:
        return None
    return (m[-1] == "A") == wrong_first  # True = picked the wrong response


def main():
    random.seed(0)
    ps = {p["id"]: p for p in problems()}
    rows = screen(list(ps.values()))
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["judge"]] = r["pred"]
    for j in JUDGES:
        acc = sum(by[i][j] == ps[i]["gold"] for i in by) / len(by)
        print(f"screen {j}: acc={acc:.2f} n={len(by)}")

    # wrong answers per problem: each judge's (parseable) wrong answer
    items = []  # (pid, owner_judge, wrong_answer)
    for pid, preds in by.items():
        g = ps[pid]["gold"]
        wrong = {j: a for j, a in preds.items() if a and a != g and re.fullmatch(r"-?\d+", a)}
        if not wrong:
            continue
        # foreign answer for a judge: another judge's distinct wrong answer, else gold +/- perturbation
        for j, a in wrong.items():
            others = [b for k, b in wrong.items() if k != j and b != a]
            foreign = others[0] if others else str(int(g) + random.choice([-2, -1, 1, 2]) * max(1, abs(int(g)) // 10))
            if foreign == a:
                foreign = str(int(foreign) + 1)
            items.append({"id": pid, "judge": j, "own": a, "foreign": foreign, "crossed": bool(others)})
    print(f"usable (judge,problem) items: {len(items)}, crossed: {sum(i['crossed'] for i in items)}")

    # writer: one correct + one per needed wrong answer per problem
    cache = {(w["id"], w["target"]): w["text"] for w in load(WRITE)}
    need = {(it["id"], ps[it["id"]]["gold"]) for it in items}
    need |= {(it["id"], it["own"]) for it in items} | {(it["id"], it["foreign"]) for it in items}
    todo = [k for k in need if k not in cache]
    with ThreadPoolExecutor(POOL) as ex:
        texts = list(ex.map(lambda k: write_solution(ps[k[0]], k[1], k[1] == ps[k[0]]["gold"]), todo))
    with WRITE.open("a") as f:
        for k, t in zip(todo, texts):
            cache[k] = t
            f.write(json.dumps({"id": k[0], "target": k[1], "text": t}) + "\n")

    jobs = []
    for it in items:
        p = ps[it["id"]]
        for cond in ("own", "foreign"):
            for wf in (True, False):
                jobs.append((it, cond, wf))

    def run(job):
        it, cond, wf = job
        p = ps[it["id"]]
        picked_wrong = judge(it["judge"], p, cache[(p["id"], p["gold"])], cache[(p["id"], it[cond])], wf)
        return {**it, "cond": cond, "wrong_first": wf, "picked_wrong": picked_wrong}
    with ThreadPoolExecutor(POOL) as ex:
        res = list(ex.map(run, jobs))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in res))

    for j in JUDGES:
        for subset, f in (("all", lambda r: True), ("crossed", lambda r: r["crossed"])):
            line = []
            for cond in ("own", "foreign"):
                v = [r["picked_wrong"] for r in res if r["judge"] == j and r["cond"] == cond and f(r) and r["picked_wrong"] is not None]
                line.append(f"{cond}={sum(v)/max(1,len(v)):.2f}(n={len(v)})")
            print(j, subset, *line)


if __name__ == "__main__":
    main()
