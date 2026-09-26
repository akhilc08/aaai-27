import sys,json,random,re,time; sys.path.insert(0,"/Users/sickle/Coding/aaai-27/experiments/pilots"); import jevlib as J
C=json.load(open("/Users/sickle/Coding/aaai-27/experiments/pilots/swarm/comments.json"))
P=json.load(open("/Users/sickle/Coding/aaai-27/experiments/pilots/swarm/personas.json"))
ACTS=["upvote","downvote","ignore"]
def decide(arm,persona,comment,score,rng):
    """arm in llm, jev. score None = hidden. returns dict(action, probs)"""
    sc = "" if score is None else f"\nCurrent score: {score:+d} points"
    if arm=="llm":
        m=[{"role":"system","content":f"You are this Reddit user: {persona}"},
           {"role":"user","content":f"You are browsing {comment['topic'].split(':')[0]}. A comment on a post about {comment['topic'].split(': ')[1]}:\n\"{comment['text']}\"{sc}\nDo you upvote, downvote, or ignore it? Answer with one word: upvote, downvote, or ignore."}]
        txt,u=J.llm(m,max_tokens=4,temperature=1.0,tag="swarm-llm")
        t=txt.strip().lower(); a=next((x for x in ACTS if x in t),"ignore")
        return {"action":a,"raw":txt,"cost":u.get("cost",0)}
    st={"you_are":persona,"browsing":comment["topic"],"comment":comment["text"]}
    if score is not None: st["comment_current_score"]=score
    ans=J.jev(st,{"action":{"type":"choice","instructions":"As this Reddit user, what do you do with this comment?","criteria":{"upvote":"click upvote","downvote":"click downvote","ignore":"scroll past without voting"}}},tag="swarm-jev")
    p=ans["action"]["probabilities"]; w=[p.get(x,0) for x in ACTS]
    return {"action":rng.choices(ACTS,weights=w)[0],"argmax":ans["action"]["choice"],"probs":p}
