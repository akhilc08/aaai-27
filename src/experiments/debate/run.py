"""Usage: uv run python run.py --arm 1 [--limit N]. Writes runs/arm{N}.json (resumable per question)."""
import argparse, json, sys, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from arms import ARMS
from data import load
from llm import spent, BudgetExceeded

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()
qs = load()[: a.limit] if a.limit else load()
out = Path(__file__).parent / "runs" / f"arm{a.arm}.json"
done = json.loads(out.read_text()) if out.exists() else {}
todo = [q for q in qs if q["id"] not in done]
print(f"arm {a.arm}: {len(done)} done, {len(todo)} todo, spent so far ${spent():.3f}", flush=True)
fn = ARMS[a.arm]


def one(q):
    return q["id"], fn(q, {"arm": a.arm, "dataset": q["dataset"], "qid": q["id"]})


with ThreadPoolExecutor(a.workers) as ex:
    futs = [ex.submit(one, q) for q in todo]
    for i, f in enumerate(as_completed(futs)):
        try:
            qid, rec = f.result()
            done[qid] = rec
            out.write_text(json.dumps(done, indent=1))
            if i % 10 == 0:
                print(f"  {len(done)}/{len(qs)} spent=${spent():.3f}", flush=True)
        except BudgetExceeded as e:
            print("BUDGET STOP", e); sys.exit(2)
        except Exception:
            traceback.print_exc()
acc = sum(r["correct"] for r in done.values()) / max(len(done), 1)
print(f"arm {a.arm} done: n={len(done)} acc={acc:.3f} spent=${spent():.3f}")
