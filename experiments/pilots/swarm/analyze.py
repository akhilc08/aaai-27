import json,numpy as np,collections,math
B=2000; rng=np.random.default_rng(0)
def ci(x): x=np.asarray(x,float); bs=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(B)]; return f"{x.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"
def ent(acts):
    c=collections.Counter(acts); n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
print("== PROBE (same persona+comment, shown score varied) ==")
for arm in ["llm","jev"]:
    rows=[json.loads(l) for l in open(f"probe_{arm}.jsonl")]
    by=collections.defaultdict(dict)
    for r in rows: by[r["pair"]][r["score"]]=r
    pairs=[d for d in by.values() if len(d)==4]
    keyname={1:"+1",0:"0",-1:"-1",None:"hidden"}
    variants=[("sampled","action")]+([("argmax","argmax"),("prob","probs")] if arm=="jev" else [])
    for vn,k in variants:
        def up(r): return r["probs"].get("upvote",0) if k=="probs" else float(r[k]=="upvote")
        def dn(r): return r["probs"].get("downvote",0) if k=="probs" else float(r[k]=="downvote")
        s=" ".join(f"P(up|{keyname[s]})={np.mean([up(d[s]) for d in pairs]):.3f}" for s in (1,0,-1,None))
        s2=" ".join(f"P(dn|{keyname[s]})={np.mean([dn(d[s]) for d in pairs]):.3f}" for s in (1,0,-1,None))
        print(f"{arm}/{vn} n_pairs={len(pairs)}\n  {s}\n  {s2}")
        print("  dUp(+1 vs 0):",ci([up(d[1])-up(d[0]) for d in pairs])," dUp(-1 vs 0):",ci([up(d[-1])-up(d[0]) for d in pairs]),
              " dDown(-1 vs 0):",ci([dn(d[-1])-dn(d[0]) for d in pairs])," dUp(0 vs hidden):",ci([up(d[0])-up(d[None]) for d in pairs]))
        b=np.mean([up(d[0]) for d in pairs]); print(f"  relative up-vote lift +1 vs 0: {np.mean([up(d[1]) for d in pairs])/b-1:+.1%}")
        if k!="probs": print("  action entropy (bits):",round(ent([d[s][k] for d in pairs for s in (1,0,-1,None)]),3),
                            " frac pairs whose action changes with score:",round(np.mean([len({d[s][k] for s in (1,0,-1)})>1 for d in pairs]),3))
print("\n== DYNAMIC SIM (60 comments, 20/treatment, 2 views/comment/round) ==")
for tag,R in [("llm_sample",8),("jev_sample",8),("jev_sample",25),("jev_argmax",8),("jev_argmax",25)]:
    rows=[json.loads(l) for l in open(f"dyn_{tag}.jsonl")]; T={int(k):v for k,v in json.load(open(f"dyn_{tag}_final.json"))["treat"].items()}
    sc={c:T[c] for c in T}
    for r in rows:
        if r["round"]<R: sc[r["c"]]+={"upvote":1,"downvote":-1,"ignore":0}[r["used"]]
    print(f"{tag} rounds={R} decisions={sum(r['round']<R for r in rows)}")
    for t in (1,0,-1):
        f=[sc[c] for c in T if T[c]==t]; org=[x-t for x in f]
        print(f"  treat {t:+d}: mean final {ci(f)}  organic(final-treat) {np.mean(org):+.2f}  P(final>0)={np.mean(np.array(f)>0):.2f}")
    f=lambda t:np.array([sc[c] for c in T if T[c]==t],float)
    d=[f(1)[rng.integers(0,20,20)].mean()-f(0)[rng.integers(0,20,20)].mean() for _ in range(B)]
    print(f"  final(+1)-final(0) = {f(1).mean()-f(0).mean():+.2f} [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]")
    r0=[r for r in rows if r["round"]==0]
    print("  round-0 up-rate by treat:",{t:round(np.mean([r["used"]=="upvote" for r in r0 if r["treat"]==t]),3) for t in (1,0,-1)},
          " overall up-rate by shown score sign:",{s:round(np.mean([r["used"]=="upvote" for r in rows if r['round']<R and np.sign(r["shown"])==s]),3) for s in (1,0,-1)})
    ents=[ent([r["used"] for r in rows if r["c"]==c and r["round"]<R]) for c in T]
    print("  mean per-comment action entropy across viewers (bits):",round(np.mean(ents),3), " overall action mix:",dict(collections.Counter(r["used"] for r in rows if r["round"]<R)))
