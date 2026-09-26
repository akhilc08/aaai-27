import sys, random, json, common3 as C
rounds, budget, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
def one(k):
    order = [r["id"] for r in C.RULES]; random.Random(k).shuffle(order)
    s, t = C.compact(order, rounds, budget)
    ctx = C.context({"summary": s, "tail": t}).lower()
    return [r["kw"] in ctx for r in C.RULES[:6]], [r["kw"] in ctx for r in C.RULES[6:]], s
res = C.J.pmap(one, range(n))
for real, dis, s in res: print(sum(real), sum(dis), s[:200].replace("\n", " "))
print("spend", C.J.spend(C.TAG))
