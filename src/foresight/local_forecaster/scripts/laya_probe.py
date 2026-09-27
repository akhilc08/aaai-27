import sys, os, time, pickle, glob, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import laya_util as U, feats as X, torch
rows = []
with open(glob.glob('data_smoke/pos_*SH_MBP*.pkl')[0], 'rb') as f:
    while True:
        try: rows += pickle.load(f)
        except EOFError: break
ag = U.load(half=True)
texts = [X.state_text(pickle.loads(r['state'])) for r in rows[:128]]
its = [U.encode(ag, t, "noul", U.Q_PWIN) for t in texts]
print("tokens", sorted(len(i['ids']) for i in its)[::16])
for bs in (16, 32, 64):
    t0 = time.time(); lg = U.predict_logits(ag, its, bs=bs); torch.mps.synchronize(); print(bs, "sec/128", round(time.time()-t0, 2))
print("zero-shot p", U.noul_p(lg)[:10].round(3), [r['won'] for r in rows[:10]])
print("maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)
