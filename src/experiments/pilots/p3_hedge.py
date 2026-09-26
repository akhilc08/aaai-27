"""Pilot 3: do hedged reviewer notes get ignored? GSM8K drafts, half with one planted error."""
import json, re, random, math
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
from datasets import load_dataset
from llm import chat, MODELS, RUNS

OUT = RUNS / "p3.jsonl"
N_EACH = 25
NOTES = {
    "none": None,
    "blunt": "Step {n} is wrong.",
    "neutral": "I think Step {n} may contain an error.",
    "hedged": "Step {n} is interesting, isn't it?",
    "polite": "Thanks, this looks great overall! Just a tiny thought, and I'm probably overthinking it, but maybe Step {n} could be worth one more glance?",
}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_num(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}"


def build():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(0)
    items = []
    for ex in ds:
        body, gold = ex["answer"].split("####")
        gold = to_num(gold.strip())
        lines = [re.sub(r"<<[^>]*>>", "", l).strip() for l in body.strip().split("\n") if l.strip()]
        if len(lines) < 3 or gold is None:
            continue
        idx = len(lines) // 2 - (1 if len(lines) % 2 == 0 else 0)  # a middle, non-final step
        m = list(re.finditer(r"=\s*\$?(-?\d[\d,]*\.?\d*)", lines[idx]))
        if not m:
            continue
        items.append((ex["question"], lines, gold, idx, m[-1]))
        if len(items) >= 2 * N_EACH:
            break
    drafts = []
    for i, (q, lines, gold, idx, m) in enumerate(items):
        err = i >= N_EACH
        final = gold
        lines = list(lines)
        if err:
            old = to_num(m.group(1))
            delta = rng.choice([2, 3, 4, 5, 10]) * (1 if old < 10 else max(1, round(old * 0.2)) // 2 or 1)
            new = old + delta
            s, e = m.span(1)
            lines[idx] = lines[idx][:s] + fmt(new) + lines[idx][e:]
            final = gold + delta if gold + delta != gold else gold + 1
        draft = "\n".join(f"Step {k+1}: {l}" for k, l in enumerate(lines)) + f"\nFinal answer: {fmt(final)}"
        drafts.append({"id": i, "err": err, "q": q, "gold": gold, "draft_ans": final, "step": idx + 1, "draft": draft})
    return drafts


def run_one(args):
    model, d, wording = args
    note = NOTES[wording]
    note_txt = note.format(n=d["step"]) if note else "(no reviewer note)"
    prompt = (f"You are finalizing a solution. Here is the problem, a draft solution, and a reviewer note.\n\n"
              f"PROBLEM:\n{d['q']}\n\nDRAFT:\n{d['draft']}\n\nREVIEWER NOTE:\n{note_txt}\n\n"
              f"Briefly decide what the final answer should be, then end with 'ANSWER: <number>'.")
    r = chat(model, [{"role": "user", "content": prompt}], {"exp": "p3", "wording": wording}, max_tokens=700)
    m = re.findall(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)", r["text"])
    ans = to_num(m[-1]) if m else None
    return {"model": model, "id": d["id"], "err": d["err"], "wording": wording, "gold": d["gold"],
            "draft_ans": d["draft_ans"], "ans": ans, "text": r["text"]}


def analyze(rows):
    z = NormalDist().inv_cdf
    adj = lambda k, n: (k + 0.5) / (n + 1)  # log-linear correction
    print(f"{'model':34}{'wording':9}{'fix%err':>8}{'keep%ok':>8}{'H':>6}{'FA':>6}{'dprime':>7}{'c':>7}")
    for model in MODELS:
        for w in NOTES:
            rs = [r for r in rows if r["model"] == model and r["wording"] == w]
            E = [r for r in rs if r["err"]]; C = [r for r in rs if not r["err"]]
            chg = lambda r: r["ans"] is None or abs(r["ans"] - r["draft_ans"]) > 1e-6
            ok = lambda r: r["ans"] is not None and abs(r["ans"] - r["gold"]) < 1e-6
            h, fa = sum(map(chg, E)), sum(map(chg, C))
            H, F = adj(h, len(E)), adj(fa, len(C))
            print(f"{model:34}{w:9}{100*sum(map(ok,E))/len(E):8.0f}{100*sum(map(ok,C))/len(C):8.0f}"
                  f"{h/len(E):6.2f}{fa/len(C):6.2f}{z(H)-z(F):7.2f}{-(z(H)+z(F))/2:7.2f}")


if __name__ == "__main__":
    drafts = build()
    jobs = [(m, d, w) for m in MODELS for d in drafts for w in NOTES]
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(run_one, jobs))
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    analyze(rows)
