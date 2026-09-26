import sys,json,re; sys.path.insert(0,"/Users/sickle/Coding/aaai-27/experiments/pilots"); import jevlib as J
topics=["r/politics: minimum wage increase","r/technology: AI replacing jobs","r/nba: best player of all time","r/cooking: pineapple on pizza","r/science: new study on coffee and health","r/gaming: microtransactions in games"]
def gen_c(t):
    txt,_=J.llm([{"role":"user","content":f"Write 10 realistic, varied Reddit comments (1-2 sentences each) replying to a post in {t}. Mix quality: some insightful, some snarky, some low-effort, some controversial, different opinions. Output one JSON array of strings only."}],max_tokens=1200,temperature=0.9,tag="swarm-gen")
    return [{"topic":t,"text":s} for s in json.loads(re.search(r"\[.*\]",txt,re.S).group(0))]
def gen_p(i):
    txt,_=J.llm([{"role":"user","content":f"Generate 25 diverse, realistic US Reddit user personas (batch {i}). Vary age 18-70, gender, occupation, interests, political lean (left/center/right), and temperament (e.g. contrarian, agreeable, lurker, grumpy, enthusiastic). Each one sentence. Output one JSON array of strings only."}],max_tokens=2500,temperature=1.0,tag="swarm-gen")
    return json.loads(re.search(r"\[.*\]",txt,re.S).group(0))
C=[c for cs in J.pmap(gen_c,topics) if isinstance(cs,list) for c in cs]
P=[p for ps in J.pmap(gen_p,range(9)) if isinstance(ps,list) for p in ps][:150]
for i,c in enumerate(C): c["id"]=i
json.dump(C,open("comments.json","w"),indent=0); json.dump(P,open("personas.json","w"),indent=0)
print(len(C),len(P)); print(C[:3]); print(P[:3]); print(J.spend("swarm"))
