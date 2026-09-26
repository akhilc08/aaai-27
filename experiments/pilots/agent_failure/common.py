import json, re, sys
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots")
import jevlib as J
TAG = "agent_failure"
KS = [1, 3, 5, 10]

def cut(s, n):
    s = s or ""
    return s if len(s) <= n else s[: n * 2 // 3] + " ...[truncated]... " + s[-n // 3:]

def load(name="trajs"):
    return [json.loads(l) for l in open(f"data/{name}.jsonl")]

def render(t, k):
    """Issue + first k steps. Only called when len(steps) > k, so the final (submit) step is never shown."""
    assert len(t["steps"]) > k
    parts = ["ISSUE:\n" + cut(t["issue"].split("ISSUE:", 1)[-1].split("INSTRUCTIONS:")[0].strip(), 1500)]
    for i, s in enumerate(t["steps"][:k]):
        parts.append(f"--- STEP {i+1} ---\nAGENT: {cut(s['act'], 700)}\nOBSERVATION: {cut(s['obs'], 450)}")
    return "\n\n".join(parts)

ERR = re.compile(r"Traceback|Error|error:|SyntaxError|not found|No such file|command not found|failed", re.I)

def heur(t, k):
    st = t["steps"][:k]
    acts = [s["act"].strip().split("```")[-2].strip() if s["act"].count("```") >= 2 else "" for s in st]
    return {"n_err_obs": sum(bool(ERR.search(s["obs"] or "")) for s in st),
            "n_edits": sum(a.startswith("edit") for a in acts),
            "n_repeat": len(acts) - len(set(acts)),
            "chars_obs": sum(len(s["obs"] or "") for s in st),
            "reproduce": int(any("reproduce" in a for a in acts))}

Q = {
 "succ": "Will this software agent run ultimately end with a correct fix for the issue that passes the hidden tests?",
 "reproduced": "Has the agent successfully reproduced the bug described in the issue (observed the buggy behavior)?",
 "located": "Has the agent found and opened the source file where the bug most likely needs to be fixed?",
 "edited_src": "Has the agent already edited library source code (not just a reproduction script)?",
 "looping": "Is the agent stuck: repeating the same or near-identical commands, or going in circles?",
 "ignored_err": "Did the agent encounter errors (tracebacks, syntax errors, failed commands) that it has not resolved?",
 "env_trouble": "Is the agent struggling with the environment (missing modules, wrong paths, files not found, broken reproduction)?",
 "on_track": "Does the agent's current plan correctly understand the root cause of the issue?",
}
def jevq(state):
    return {k: {"type": "noul", "instructions": v} for k, v in Q.items()}
