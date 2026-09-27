"""Run a search arm vs a baseline on our server (port 8001).
usage: run_arm.py NAME OPP_MODEL LEAF DEPTH OPPONENT N [CONC]
OPP_MODEL: heur | gbm_hist | gbm_nohist | logreg_hist | laya_hist:<ckpt> | laya_nohist:<ckpt>
LEAF: hp | logreg | gbm | logreg_delta | laya:<ckpt> | layadelta:<ckpt> | hyb:<lam>:<leaf> | sharp:<k>:<leaf>
OPPONENT: SH | MBP | RND | ABYSSAL"""
import asyncio, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bots as B
import evals as E

HERE = os.path.dirname(os.path.abspath(__file__))


def leaf_of(spec, fmt_):
    if spec == "hp":
        return E.HPLeaf()
    if spec == "fp":
        return E.FPLeaf()
    if spec.startswith("alive"):
        return E.AliveLeaf(float(spec[5:] or 0.3))
    if spec in ("logreg", "gbm"):
        return E.SkLeaf(fmt_, spec)
    if spec in ("logreg_delta", "gbm_delta"):
        return E.SkLeaf(fmt_, spec.split("_")[0], delta=True)
    if spec.startswith("laya:"):
        return E.LayaLeaf(spec[5:])
    if spec.startswith("layadelta:"):
        return E.LayaLeaf(spec[10:], delta=True)
    if spec.startswith("hyb:"):
        _, lam, rest = spec.split(":", 2)
        return E.Hybrid(leaf_of(rest, fmt_), float(lam))
    if spec.startswith("sharp:"):
        _, k, rest = spec.split(":", 2)
        return E.Sharpen(leaf_of(rest, fmt_), float(k))
    raise ValueError(spec)


def opp_of(spec, fmt_):
    if spec == "heur":
        return E.HeurOpp()
    if spec == "uniform":
        return E.UniformOpp()
    if spec.startswith("laya_"):
        kind, path = spec.split(":", 1)
        sh = 1.0
        if "_x" in kind:
            kind, sh = kind.split("_x")[0], float(kind.split("_x")[1])
        return E.LayaOpp(path, hist=kind == "laya_hist", sharpen=sh)
    return E.SkOpp(fmt_, spec)


async def main():
    import memguard
    memguard.start(6000 if ("laya" in " ".join(sys.argv)) else 1500)
    name, oppm, leaf, depth, opponent, n = sys.argv[1:7]
    depth, n = int(depth), int(n)
    conc = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    fmt_ = B.FMT
    mfmt = os.environ.get("MODEL_FMT", fmt_)
    strat = None
    if os.environ.get("STRATEGY"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("strategy", os.environ["STRATEGY"])
        strat = importlib.util.module_from_spec(spec); spec.loader.exec_module(strat)
        ev = strat.make_ev(opp_of(oppm, mfmt))
    else:
        ev = E.Ev(opp_of(oppm, mfmt), leaf_of(leaf, mfmt))
    r = random.randint(10000, 99999)
    log = os.path.join(HERE, "data", f"dec_{name}_{opponent}_{fmt_}.jsonl")
    me = f"{name[:10]}a{r}"
    a = B.SearchBot(name, depth, "exp", ev, log, **B.kw(me, conc))
    a.strat = strat
    if opponent == "ABYSSAL" or os.environ.get("SAVE_TURNS"):
        a.save_turns, a.turns_path = {}, os.path.join(HERE, "data", f"turns_{name}_{opponent}_{fmt_}.pkl")
    t0 = time.time()
    if os.environ.get("EVAL_TIMEOUT_S"):
        async def _deadline():
            await asyncio.sleep(float(os.environ["EVAL_TIMEOUT_S"]))
            print(json.dumps({"timeout": True, "finished": a.n_finished_battles, "wins": a.n_won_battles}), flush=True)
            os._exit(4)
        asyncio.create_task(_deadline())
    if opponent == "ABYSSAL":
        # PokeChamp's AbyssalPlayer leaks memory: fresh process every CHUNK battles, killed if footprint > 2.5 GB
        pc = os.path.join(HERE, "pokechamp")
        py = os.environ.get("PC_PY", os.path.join(HERE, "..", "pokemon_search", ".venv-pc", "bin", "python"))
        chunk, done_target = int(os.environ.get("CHUNK", "20")), n
        ci = 0
        while a.n_finished_battles < done_target and ci < 3 * (n // chunk + 2):
            k = min(chunk, done_target - a.n_finished_battles)
            abn = f"aby{r}c{ci}"; ci += 1
            before = a.n_finished_battles
            proc = await asyncio.create_subprocess_exec(py, os.path.join(HERE, "abyssal_runner.py"), me, str(k), fmt_, abn, cwd=pc)
            acc = asyncio.create_task(a.accept_challenges(abn, k))
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    fp = await asyncio.create_subprocess_exec("footprint", str(proc.pid), stdout=asyncio.subprocess.PIPE,
                                                              stderr=asyncio.subprocess.DEVNULL)
                    out, _ = await fp.communicate()
                    mb = 0
                    for line in out.decode().splitlines():
                        if "phys_footprint:" in line:
                            v = line.split(":")[1].strip().split()
                            mb = float(v[0]) * (1024 if v[1].startswith("GB") else 1)
                    if mb > 2500:
                        print(f"abyssal footprint {mb} MB > 2.5 GB, killing chunk", flush=True)
                        proc.kill()
            await asyncio.sleep(3)
            acc.cancel()
            print(json.dumps({"chunk": ci, "finished": a.n_finished_battles, "wins": a.n_won_battles, "added": a.n_finished_battles - before}), flush=True)
    else:
        b = B.BASE[opponent](**B.kw(f"{opponent}b{r}", conc))
        await a.battle_against(b, n_battles=n)
    lats = []
    for line in open(log):
        try:
            x = json.loads(line)
            if x.get("battle") in a.battles and "lat" in x:
                lats.append(x["lat"])
        except Exception:
            pass
    res = {"search_env": {k: os.environ[k] for k in ("OPP_MASS", "OPP_CAP", "CHANCE", "OUR_K") if os.environ.get(k)}, "arm": name, "opp_model": oppm, "leaf": leaf, "depth": depth, "opponent": opponent, "fmt": fmt_,
           "n": a.n_finished_battles, "wins": a.n_won_battles, "ties": a.n_tied_battles, "secs": round(time.time() - t0, 1),
           "lat_mean": round(sum(lats) / max(len(lats), 1), 3), "lat_p95": round(sorted(lats)[int(0.95 * len(lats))] if lats else 0, 3),
           "lat_max": max(lats) if lats else 0, "errors": sum(1 for l in open(log) if '"error"' in l)}
    with open(os.path.join(HERE, "data", "arm_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
