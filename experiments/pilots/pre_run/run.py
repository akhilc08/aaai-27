"""Score each SWE-bench Verified issue (text only) with Jev zero-shot, Jev multi-question, and a cheap LLM."""
import json, os, re, sys, time
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots"); import jevlib as J
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "pre_run"
MAXC = 5000
I = [json.loads(l) for l in open(f"{HERE}/data/issues.jsonl")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else len(I)
OUT = f"{HERE}/raw_scores.jsonl"
done = {json.loads(l)["instance_id"] for l in open(OUT)} if os.path.exists(OUT) else set()

def state(r):
    t = r["problem_statement"]
    if len(t) > MAXC: t = t[:MAXC] + "\n...[truncated]"
    return {"repository": r["repo"], "github_issue": t}

ZS = {"succ": {"type": "noul", "instructions": "An autonomous AI coding agent will be given this GitHub issue and the repository, and must write a patch that makes the hidden tests pass. Will the agent successfully resolve the issue?"}}
MQ = {
 "one_line": "Is the correct fix likely a small change of one or a few lines?",
 "clear_expected": "Does the issue clearly specify the expected (correct) behavior?",
 "repro": "Does the issue include a minimal reproducible code example?",
 "traceback": "Does the issue include an error message or traceback pointing to where the bug is?",
 "location": "Does the issue name the specific function, class or file that needs to change?",
 "design": "Does fixing this require design decisions or choosing between several reasonable behaviors?",
 "multi_file": "Will the fix likely require changes across multiple files or modules?",
 "new_feature": "Is this a request for a new feature or API (rather than a bug fix)?",
 "fix_given": "Does the issue text itself propose or contain the fix (a patch, code change or clear suggestion)?",
 "ambiguous": "Is the issue vague or ambiguous, so that a reader could misunderstand what needs to be done?",
}
MQ = {k: {"type": "noul", "instructions": v} for k, v in MQ.items()}

def one(r):
    s = state(r); out = {"instance_id": r["instance_id"]}
    t0 = time.time(); a = J.jev(s, ZS, tag=TAG + "_jev_zs"); out["t_jev_zs"] = time.time() - t0
    out["jev_zs"] = a["succ"]["noul"]
    t0 = time.time(); a = J.jev(s, MQ, tag=TAG + "_jev_mq"); out["t_jev_mq"] = time.time() - t0
    out["jq"] = {k: a[k]["noul"] for k in MQ}
    t0 = time.time()
    txt, u = J.llm([{"role": "user", "content": f"Repository: {r['repo']}\n\nGitHub issue:\n{s['github_issue']}\n\n"
        "An autonomous AI coding agent will be given this issue and the repository and must write a patch that makes the hidden tests pass. "
        "What is the probability (0-100) that the agent resolves the issue? Reply with only a number."}], max_tokens=8, tag=TAG + "_llm")
    out["t_llm"] = time.time() - t0; out["llm_cost"] = u.get("cost")
    m = re.search(r"\d+(\.\d+)?", txt); out["llm"] = float(m.group()) / 100 if m else None
    return out

def main():
  todo = [r for r in I[:N] if r["instance_id"] not in done]
  R = J.pmap(one, todo, workers=10)
  with open(OUT, "a") as f:
    for r in R:
        if isinstance(r, dict): f.write(json.dumps(r) + "\n")
        else: print("ERR", r)
  print("n", sum(isinstance(r, dict) for r in R), "spend", J.spend(TAG), {k: J.spend(TAG + k) for k in ("_jev_zs", "_jev_mq", "_llm")})

if __name__ == "__main__": main()
