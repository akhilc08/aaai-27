"""Fixed-budget dev evaluation for the autoresearch loop: current autoresearch/strategy.py vs SimpleHeuristicsPlayer
(no dynamax), gen8randombattle, 4 concurrent battles, per-battle logging, 30-min timeout. Never Abyssal.
usage: eval.py EXP_ID N  -> prints one JSON line; pooled counts for arm ar_<EXP_ID> across calls."""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "laya_opp_hist_gen8randombattle.pt"


def main():
    exp, n = sys.argv[1], int(sys.argv[2])
    arm = f"ar_{exp}"
    env = dict(os.environ, STRATEGY=os.path.join(HERE, "autoresearch", "strategy.py"), FMT="gen8randombattle",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", EVAL_TIMEOUT_S="1800",
               PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.25", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.2")
    t0 = time.time()
    with open(os.path.join(HERE, "autoresearch", "logs", f"{arm}_{int(t0)}.log"), "w") as lf:
        rc = subprocess.call([os.path.join(HERE, ".venv", "bin", "python"), "run_arm.py", arm, f"laya_hist:{CKPT}", "hp", "2", "AB", str(n), "4"],
                             cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    w = m = 0
    for l in open(os.path.join(HERE, "data", "battles_log.jsonl")):
        x = json.loads(l)
        if x["arm"] == arm:
            m += 1; w += x["won"] is True
    lat = []
    fn = os.path.join(HERE, "data", f"dec_{arm}_AB_gen8randombattle.jsonl")
    errs = 0
    if os.path.exists(fn):
        for l in open(fn):
            x = json.loads(l)
            if "lat" in x and "nodes" in x:
                lat.append(x["lat"])
            if "error" in x:
                errs += 1
    out = {"exp": exp, "rc": rc, "wins": w, "n": m, "rate": round(w / max(m, 1), 4), "secs": round(time.time() - t0),
           "mean_turn_s": round(float(np.mean(lat)), 2) if lat else None, "p95_turn_s": round(float(np.percentile(lat, 95)), 2) if lat else None,
           "errors": errs, "timeout": rc == 4}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
