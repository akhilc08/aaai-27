import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import numpy as np
import build_ds as B
rows = []
for fn in glob.glob('data_check/pos_*.pkl'):
    with open(fn, 'rb') as f:
        while True:
            try: rows += pickle.load(f)
            except EOFError: break
pp = defaultdict(dict)
for r in rows:
    if not r['forced']: pp[(r['battle'], r['player'])][r['turn']] = r
players = defaultdict(set)
for r in rows: players[r['battle']].add(r['player'])
diffs, n = [], 0
for r in rows:
    if r['forced'] or 'live_hist' not in r: continue
    opp = [p for p in players[r['battle']] if p != r['player']][0]
    prev_t = sorted(t for t in pp[(r['battle'], opp)] if t < r['turn'])[-5:]
    mine = pp[(r['battle'], r['player'])]
    hist = [B.prev_info(mine.get(t), pp[(r['battle'], opp)][t]) for t in prev_t]
    ours = [mine[t]['act'][0] for t in sorted(t for t in mine if t < r['turn'])[-3:] if mine[t]['act']]
    off = np.array(B.hist_feats(hist, ours)); live = np.array(r['live_hist'])
    diffs.append(np.abs(off - live)); n += 1
d = np.array(diffs)
print('rows', n, 'exact-match rate', float((d.max(1) < 1e-6).mean()))
print('mean abs diff per feature', dict(zip(B.HIST_NAMES, d.mean(0).round(3))))
k = 0
for r in rows:
    if r['forced'] or 'live_hist' not in r: continue
    opp = [p for p in players[r['battle']] if p != r['player']][0]
    prev_t = sorted(t for t in pp[(r['battle'], opp)] if t < r['turn'])[-5:]
    mine = pp[(r['battle'], r['player'])]
    hist = [B.prev_info(mine.get(t), pp[(r['battle'], opp)][t]) for t in prev_t]
    ours = [mine[t]['act'][0] for t in sorted(t for t in mine if t < r['turn'])[-3:] if mine[t]['act']]
    off = np.array(B.hist_feats(hist, ours)); live = np.array(r['live_hist'])
    if np.abs(off-live).max() > 1e-6 and k < 3:
        k += 1
        print(r['turn'], prev_t, [ (h['kind'],h['name'],h['threat']) if h else None for h in hist]); print(off.round(2)); print(live.round(2))
from collections import Counter
c = Counter(); tot = Counter()
for r in rows:
    if r['forced'] or 'live_hist' not in r: continue
    tot[r['cls']] += 1
    if r['turn'] > 2 and np.array(r['live_hist'])[0] == 0: c[(r['cls'])] += 1
print('empty live hist after turn 2 by cls', c, tot)
