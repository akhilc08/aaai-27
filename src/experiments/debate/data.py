"""Fixed-seed question sampling: 60 MMLU-Pro (stratified over 5 hard categories) + 60 GSM8K test."""
import json, random, re
from pathlib import Path
from datasets import load_dataset

SEED = 0
QFILE = Path(__file__).parent / "runs" / "questions.json"
MMLU_CATS = ["math", "physics", "chemistry", "engineering", "law"]
LETTERS = "ABCDEFGHIJ"


def build():
    rng = random.Random(SEED)
    qs = []
    m = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    per = 60 // len(MMLU_CATS)
    for cat in MMLU_CATS:
        idx = [i for i, c in enumerate(m["category"]) if c == cat]
        for i in rng.sample(idx, per):
            row = m[i]
            opts = "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(row["options"]))
            qs.append({"id": f"mmlu-{row['question_id']}", "dataset": "mmlu_pro", "category": cat,
                       "prompt": f"{row['question']}\n\nOptions:\n{opts}", "gold": row["answer"], "kind": "letter"})
    g = load_dataset("openai/gsm8k", "main", split="test")
    for i in rng.sample(range(len(g)), 60):
        row = g[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        qs.append({"id": f"gsm8k-{i}", "dataset": "gsm8k", "category": "gsm8k", "prompt": row["question"], "gold": gold, "kind": "number"})
    QFILE.write_text(json.dumps(qs, indent=1))
    return qs


def load():
    return json.loads(QFILE.read_text()) if QFILE.exists() else build()


def instruction(kind: str) -> str:
    fmt = "a single option letter" if kind == "letter" else "a single number (no units, no commas)"
    return f"Solve the problem. Think step by step but be concise. On the very last line write exactly 'ANSWER: X' where X is {fmt}."


def extract(text: str, kind: str):
    """Primary: last 'ANSWER: X' line. Fallback (same for all arms): last \\boxed{X} or 'answer is (X)'."""
    if kind == "letter":
        ms = (re.findall(r"ANSWER:\s*\(?([A-J])\)?", text, flags=re.I)
              or re.findall(r"\\boxed\{\(?([A-J])\)?\}", text)
              or re.findall(r"answer is\s*:?\s*\(?([A-J])\)?(?![A-Za-z])", text, flags=re.I))
        return ms[-1].upper() if ms else None
    ms = (re.findall(r"ANSWER:\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
          or re.findall(r"\\boxed\{\$?\s*(-?[\d,]*\.?\d+)", text)
          or re.findall(r"answer is\s*:?\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I))
    if not ms:
        return None
    s = ms[-1].replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def correct(pred, gold, kind) -> bool:
    if pred is None:
        return False
    if kind == "letter":
        return pred == gold
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold
