"""Run one arm (A=ReAct, B=predict-then-act, B2=B judging only plan-critical actions, Bnc=B2 without batching, C=ReAct+failure-library rules) over N unseen ALFWorld games.
Usage: uv run python run.py --arm A [--n 25] [--workers 6]
"""
import sys, json, re, argparse, threading, hashlib, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *

PROMPTS = json.load(open(ROOT / "prompts" / "alfworld_3prompts.json"))
_reg_lock = threading.Lock()
MAX_LLM_CALLS = 70
MAX_CONSEC_THINK = 3
PRED_ARMS = {"B", "B2", "Bnc"}
# plan-critical (manipulation) verbs; "move" is this TextWorld version's verb for put ("move X to Y")
CRIT_VERBS = {"take", "put", "move", "heat", "cool", "clean", "open", "close", "use", "slice"}
def is_critical(a): return a.split(" ", 1)[0].lower() in CRIT_VERBS
NOCACHE_NOTE = "\nIMPORTANT: in this setting write EXACTLY ONE action (with its prediction) per turn."

SYS_A = """Interact with a household to solve a task. You are given a room description, a task, and the history of your actions and observations.
At each turn output EXACTLY ONE line: either a thought of the form "think: ..." or a single action copied verbatim from the admissible commands list.
Do not output anything else. Two example episodes follow.

{examples}"""

SYS_B = """Interact with a household to solve a task. You are given a room description, a task, and the history of your actions and observations.
At each turn, continue the transcript for up to 3 actions. Write each action as a line "> <action>" followed on the next line by your PREDICTION of the observation the environment will return (one line). You may start with an optional "> think: ..." line whose observation is always "OK.".
Actions must be copied verbatim from the admissible commands list (the list only covers the current state; later actions in your batch are guesses). Write several actions only when you are confident how the environment will respond to each; write a single action when uncertain (e.g. when opening or searching for an object). Predictions must be concrete (which objects you expect to see, whether a container is closed, etc.).
Output nothing else. Two example episodes follow.

{examples}"""

SYS_C_RULES = """

Rules learned from previous failures on this task type (follow them):
{rules}"""

def examples_for(tt):
    return "\n\n".join(PROMPTS[f"react_{tt}_{i}"].strip() for i in (0, 1))

def fmt_history(init_obs, hist):
    s = init_obs + "\n"
    for a, o in hist:
        s += f"> {a}\n{o}\n"
    return s

def user_msg(init_obs, hist, admissible, extra=""):
    return (f"Here is the task.\n{fmt_history(init_obs, hist)}"
            f"{extra}Admissible commands: {json.dumps(admissible)}\n>")

def match_action(a, admissible):
    a = a.strip().lstrip("> ").strip()
    if a in admissible: return a
    low = {c.lower(): c for c in admissible}
    if a.lower() in low: return low[a.lower()]
    a2 = a.replace("in/on", "in/on").replace(" on ", " in/on ").replace(" in ", " in/on ")
    if a2 in admissible: return a2
    return a  # invalid -> env will say "Nothing happens."

def judge(pred, actual):
    """Tiny LLM check: does the actual observation match the prediction? Returns bool match."""
    msgs = [{"role": "user", "content":
             "An agent in a text household game predicted what it would observe after an action.\n"
             f"PREDICTED: {pred}\nACTUAL: {actual}\n"
             "Is the actual observation consistent with the prediction (same kind of outcome, e.g. the same container state and the objects the agent expected to find are present, or the same action success/failure)? "
             "Minor wording differences do not matter. Answer with one word: MATCH or MISMATCH."}]
    txt, u = llm(msgs, tag="judge", max_tokens=4)
    return ("MISMATCH" not in txt.upper()), u

def run_episode(arm, gamefile, rules=None):
    tt = task_type(gamefile)
    name = "g" + hashlib.md5(gamefile.encode()).hexdigest()[:8] + arm
    with _reg_lock:
        import textworld.gym
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games([gamefile], request_infos, batch_size=1, asynchronous=False,
                                              max_episode_steps=MAX_STEPS + 5, name=name,
                                              wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        env = textworld.gym.make(env_id)
    obs, info = env.reset()
    init_obs = clean_obs(obs[0]); admissible = list(info["admissible_commands"][0])
    sys_prompt = (SYS_B if arm in PRED_ARMS else SYS_A).format(examples=examples_for(tt))
    if arm == "Bnc": sys_prompt += NOCACHE_NOTE
    if arm == "C" and rules:
        sys_prompt += SYS_C_RULES.format(rules="\n".join(f"- {r}" for r in rules))
    hist = []; transcript = []
    stats = dict(game=gamefile, task_type=tt, arm=arm, success=False, steps=0, llm_calls=0, judge_calls=0,
                 prompt_tokens=0, completion_tokens=0, cost=0.0, replans=0, mismatches=0,
                 mismatch_steps=[], replan_steps=[], cached_executed=0, batches=0, invalid_actions=0)
    def acct(u, is_judge=False):
        stats["llm_calls"] += 1; stats["judge_calls"] += int(is_judge)
        stats["prompt_tokens"] += u["prompt_tokens"]; stats["completion_tokens"] += u["completion_tokens"]; stats["cost"] += u["cost"]
    def env_step(action):
        o, r, done, inf = env.step([action]); o = o[0].strip()
        stats["steps"] += 1
        if o == "Nothing happens.": stats["invalid_actions"] += 1
        return o, bool(inf["won"][0]), list(inf["admissible_commands"][0])

    consec_think = 0; consec_mismatch = 0; force_think = False; done = False
    while not done and stats["steps"] < MAX_STEPS and stats["llm_calls"] < MAX_LLM_CALLS:
        extra = ""
        if force_think:
            extra = "Your predictions have been wrong; re-plan from the current state. Start your output with a '> think:' line.\n"
            force_think = False
        if consec_think >= MAX_CONSEC_THINK:
            extra += "You must output an action now (no think).\n"
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg(init_obs, hist, admissible, extra)}]
        try:
            out, u = llm(msgs, tag=f"agent{arm}", max_tokens=40 if arm not in PRED_ARMS else 260, stop=None if arm in PRED_ARMS else ["\n"])
        except BudgetExceeded: raise
        acct(u)
        raw = [l.rstrip() for l in out.splitlines() if l.strip()]
        if not raw: raw = ["> look"]
        if arm not in PRED_ARMS:
            line = raw[0].strip().lstrip("> ").strip()
            if line.lower().startswith("think:"):
                hist.append((line, "OK.")); transcript.append({"agent": line, "obs": "OK."}); consec_think += 1; continue
            consec_think = 0
            a = match_action(line, admissible)
            o, won, admissible = env_step(a); hist.append((a, o)); transcript.append({"agent": a, "obs": o})
            if won: stats["success"] = True; done = True
            continue
        # ---- Arm B: parse "> action" / prediction pairs ----
        turns = []  # [action, prediction]
        if not any(l.lstrip().startswith(">") for l in raw):
            turns = [[raw[0].strip(), raw[1].strip() if len(raw) > 1 and raw[1].strip() not in admissible else ""]]
        else:
            for l in raw:
                if l.lstrip().startswith(">"):
                    turns.append([l.lstrip().lstrip("> ").strip(), ""])
                elif turns and not turns[-1][1]:
                    turns[-1][1] = l.strip()
        batch = []
        for a, p in turns:
            if a.lower().startswith("think:"):
                if batch: break  # think after actions -> stop batch there
                hist.append((a, "OK.")); transcript.append({"agent": a, "obs": "OK."}); consec_think += 1
            else:
                batch.append((a, p))
        if not batch: continue
        consec_think = 0
        batch = batch[:1] if arm == "Bnc" else batch[:3]
        stats["batches"] += 1
        for i, (a, pred) in enumerate(batch):
            a = match_action(a, admissible)
            o, won, admissible = env_step(a); hist.append((a, o))
            if i > 0: stats["cached_executed"] += 1
            gated = arm in ("B2", "Bnc")
            crit = is_critical(a) if gated else True
            m, ju = judge(pred, o) if (pred and crit) else (True, None)
            if ju: acct(ju, True)
            transcript.append({"agent": a, "pred": pred, "obs": o, "match": m if (pred and crit) else None, "critical": crit})
            if won: stats["success"] = True; done = True; break
            if stats["steps"] >= MAX_STEPS: break
            if not crit:
                if o == "Nothing happens.": break  # invalid non-critical action: drop rest of cached plan, no replan signal
                continue
            if not m:
                stats["mismatches"] += 1; stats["mismatch_steps"].append(stats["steps"]); consec_mismatch += 1
                if consec_mismatch >= 2:
                    stats["replans"] += 1; stats["replan_steps"].append(stats["steps"]); force_think = True; consec_mismatch = 0
                break  # discard rest of cached plan
            else:
                consec_mismatch = 0
    stats["transcript"] = transcript
    env.close()
    return stats

def work(arm, g, rules):
    try:
        return run_episode(arm, g, rules)
    except BudgetExceeded as e:
        print("BUDGET EXCEEDED", e); return None
    except Exception:
        traceback.print_exc(); return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--n", type=int, default=N_GAMES)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--rules", default=None); ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    games = select_games(args.n)
    rules = json.load(open(args.rules)) if args.rules else {}
    out_path = RUNS / f"arm_{args.arm}{args.suffix}.jsonl"
    done_games = set()
    if out_path.exists():
        done_games = {json.loads(l)["game"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [g for g in games if g not in done_games]
    print(f"arm {args.arm}: {len(todo)} games to run ({len(done_games)} already done); spend so far ${total_spend():.3f}")
    lock = threading.Lock(); t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, args.arm, g, rules.get(task_type(g), [])[:3]): g for g in todo}
        for f in as_completed(futs):
            s = f.result()
            if s is None: continue
            with lock, open(out_path, "a") as fh: fh.write(json.dumps(s) + "\n")
            print(f"[{time.time()-t0:5.0f}s] {s['task_type']:8s} ok={int(s['success'])} steps={s['steps']:2d} calls={s['llm_calls']:2d} "
                  f"mism={s['mismatches']} replans={s['replans']} cost=${s['cost']:.4f}  total=${total_spend():.3f}", flush=True)

if __name__ == "__main__":
    main()
