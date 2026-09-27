"""Isolated latency of the fine-tuned Laya opponent model (history text), per forecast, at several batch sizes."""
import sys, os, time, pickle, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import model as Mo, evals as E, search as Se

rows = []
with open(sorted(glob.glob('data_smoke/pos_*SH_MBP*.pkl'))[0], 'rb') as f:
    while True:
        try: rows += pickle.load(f)
        except EOFError: break
states = [pickle.loads(r['state']) for r in rows if not r['forced']]
states = [s for s in states if s.active(1).hp > 0 and s.active(0).hp > 0][:64]
reqs = [(s, 1, Mo.actions(s, 1), None) for s in states]
hist = ([0.0] * 10, "their previous actions (oldest first): used thunderbolt (their highest-damage option); used thunderbolt (their highest-damage option)\nour previous actions: attack, attack")
opp = E.LayaOpp(sys.argv[1] if len(sys.argv) > 1 else "laya_opp_hist_gen9randombattle.pt", hist=True)
opp(reqs[:4], hist)  # warm-up
for bs in (1, 4, 16, 32, 64):
    t0 = time.time(); n = 0
    for i in range(0, 64, bs):
        opp(reqs[i:i + bs], hist); n += len(reqs[i:i + bs])
    torch.mps.synchronize()
    dt = (time.time() - t0) / n
    print(f"batch {bs}: {1000 * dt:.1f} ms per opponent forecast -> {15 / dt:.0f} forecasts per 15 s", flush=True)
