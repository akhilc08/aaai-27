# within-pair counterfactual: same (persona, comment) shown with score +1, 0, -1, hidden
from common import *
arm=sys.argv[1]; n=int(sys.argv[2]); rng=random.Random(1)
pairs=[(rng.randrange(len(P)),rng.randrange(len(C))) for _ in range(n)]
items=[(k,pi,ci,s) for k,(pi,ci) in enumerate(pairs) for s in (1,0,-1,None)]
t=time.time()
out=J.pmap(lambda it:{**decide(arm,P[it[1]],C[it[2]],it[3],random.Random(hash(it))),"pair":it[0],"p":it[1],"c":it[2],"score":it[3]},items,workers=12)
dt=time.time()-t
with open(f"probe_{arm}.jsonl","w") as f:
    for o in out:
        if isinstance(o,dict): f.write(json.dumps(o)+"\n")
print(arm,len(items),"errors",sum(not isinstance(o,dict) for o in out),"wall",dt,"spend",J.spend("swarm-"+arm))
