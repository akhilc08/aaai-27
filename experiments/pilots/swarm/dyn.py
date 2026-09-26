# dynamic Reddit-like sim: 60 comments, treatment +1/0/-1 initial vote, rounds of 2 views per comment
from common import *
arm=sys.argv[1]; R=int(sys.argv[2]); mode=sys.argv[3] if len(sys.argv)>3 else "sample"
rng=random.Random(7); T={}
for topic in sorted({c["topic"] for c in C}):
    ids=[c["id"] for c in C if c["topic"]==topic]; tr=[1,1,1,0,0,0,-1,-1,-1,[1,0,-1][len(T)%3]][:len(ids)]; rng.shuffle(tr)
    T.update(dict(zip(ids,tr)))
score={i:T[i] for i in T}; log=[]; t=time.time(); rr=random.Random(11)
for r in range(R):
    views=[(r,rr.randrange(len(P)),ci,score[ci]) for ci in score for _ in range(2)]
    out=J.pmap(lambda v:decide(arm,P[v[1]],C[v[2]],v[3],random.Random(hash(v))),views,workers=12)
    for v,o in zip(views,out):
        if not isinstance(o,dict): continue
        a=o["argmax"] if mode=="argmax" else o["action"]
        score[v[2]]+={"upvote":1,"downvote":-1,"ignore":0}[a]
        log.append({**o,"used":a,"round":r,"p":v[1],"c":v[2],"shown":v[3],"treat":T[v[2]]})
    print(r,time.time()-t,flush=True)
tag=f"{arm}_{mode}"
with open(f"dyn_{tag}.jsonl","w") as f:
    for o in log: f.write(json.dumps(o)+"\n")
json.dump({"treat":T,"final":score,"wall":time.time()-t,"n":len(log)},open(f"dyn_{tag}_final.json","w"))
print(tag,len(log),time.time()-t,J.spend("swarm-"+arm))
