"""Generate labeled positions: run N battles between two bots, both logged.
usage: gen_data.py A B N [CONC]   (A,B in SH MBP RND H2 H1)"""
import asyncio, json, os, sys, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bots as B
import search as Se

HERE = os.path.dirname(os.path.abspath(__file__))


def mk(code, name, conc):
    if code in B.BASE:
        return B.logged(B.BASE[code])(**B.kw(name, conc))
    if code in ("H1", "H2"):
        return B.logged(B.SearchBot)(code, int(code[1]), "exp", Se.HeurEval(), os.path.join(HERE, "data", "gen_dec.jsonl"), **B.kw(name, conc))
    raise ValueError(code)


async def main():
    a_, b_, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    r = random.randint(10000, 99999)
    a, b = mk(a_, f"g{a_}a{r}", conc), mk(b_, f"g{b_}b{r}", conc)
    out = os.path.join(os.environ.get("OUTDIR", os.path.join(HERE, "data")), f"pos_{B.FMT}_{a_}_{b_}_{r}.pkl")
    t0, done, chunk = time.time(), 0, 50
    while done < n:
        k = min(chunk, n - done)
        await a.battle_against(b, n_battles=k)
        a.dump(out); b.dump(out)
        done += k
        print(json.dumps({"pair": f"{a_}-{b_}", "done": done, "a_wins": a.n_won_battles, "secs": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
