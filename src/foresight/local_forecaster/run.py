"""Run one arm vs one baseline on the local Showdown server. Logs decisions and results to data/.
usage: run.py ARM OPP N [CONC]
ARM: J0 | E1 E2 E3 (Jev expectimax depth d) | M1 M2 (Jev minimax) | L1 (LLM expectimax d1) | H1 (heuristic eval d1) | SH (SimpleHeuristics) | RND
OPP: SH | MBP"""
import asyncio, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poke_env import AccountConfiguration
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
import model as Mo
import search as Se

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
FMT = os.environ.get("FMT", "gen9randombattle")


def root_actions(battle, s):
    """Map poke-env legal orders to abstract actions on state s (root active moves replaced by available moves)."""
    acts, orders = [], []
    if battle.available_moves and not battle.force_switch:
        act = s.active(0)
        act.moves = [Mo.conv_move(m) for m in battle.available_moves]
        for k, m in enumerate(battle.available_moves):
            acts.append(("m", k)); orders.append(m)
    names = [m.name for m in s.sides[0]]
    for p in battle.available_switches:
        if p.species in names:
            acts.append(("s", names.index(p.species))); orders.append(p)
    return acts, orders


class SearchBot(Player):
    def __init__(self, arm, depth, mode, ev, log, **kw):
        super().__init__(**kw)
        self.arm, self.depth, self.mode, self.ev, self.log = arm, depth, mode, ev, log

    async def choose_move(self, battle):
        t0 = time.time()
        try:
            s = Mo.from_battle(battle)
            acts, orders = root_actions(battle, s)
            if not acts:
                return self.choose_random_move(battle)
            if len(acts) == 1:
                return self.create_order(orders[0])
            forced = bool(battle.force_switch) or not battle.available_moves
            calls0 = self.ev.calls
            if self.depth == 0:
                k, st = await asyncio.to_thread(self._oneshot, s, acts)
            else:
                k, st = await asyncio.to_thread(Se.search, s, acts, self.depth, self.ev, self.mode, forced)
            st.update(arm=self.arm, battle=battle.battle_tag, turn=battle.turn, lat=round(time.time() - t0, 3),
                      calls=st.get("calls", 1 if self.depth == 0 else 0), n_acts=len(acts), choice=str(orders[k]), forced=forced)
            st["acts"] = [Mo.act_text(s, 0, a) for a in acts] if os.environ.get("DBG") else None
            if not os.environ.get("DBG"): st.pop("q", None)
            with open(self.log, "a") as f:
                f.write(json.dumps(st) + "\n")
            return self.create_order(orders[k])
        except Exception as e:
            with open(self.log, "a") as f:
                f.write(json.dumps({"arm": self.arm, "error": repr(e)[:300], "lat": round(time.time() - t0, 3)}) + "\n")
            return self.choose_random_move(battle)

    def _oneshot(self, s, acts):
        """J0: one Jev choice over our legal actions at the root (no search)."""
        pols, _ = self.ev.run([(s, 0, acts, None)], [])
        p = pols[0]
        return max(range(len(acts)), key=lambda k: p[k]), {"nodes": 0, "leaves": 0, "policy_q": 1, "levels": 0}


def make(arm, name, log, conc):
    kw = dict(battle_format=FMT, max_concurrent_battles=conc,
              account_configuration=AccountConfiguration(name, None))
    tag = f"pksearch_{arm}_{os.environ.get('OPP', '')}"
    if arm == "SH":
        return SimpleHeuristicsPlayer(**kw)
    if arm == "AB":  # Abyssal proxy: PokeChamp's Abyssal = SimpleHeuristics logic; dynamax disabled (no-Dmax setting)
        class NoDmaxSH(SimpleHeuristicsPlayer):
            @staticmethod
            def _should_dynamax(battle, n_remaining_mons):
                return False
        return NoDmaxSH(**kw)
    if arm == "MBP":
        return MaxBasePowerPlayer(**kw)
    if arm == "RND":
        return RandomPlayer(**kw)
    if arm == "J0":
        return SearchBot(arm, 0, "exp", Se.JevEval(tag), log, **kw)
    if arm.startswith("Y"):  # hybrid leaf: Y25 = lambda 0.25, Y50 = lambda 0.5 (depth 2)
        return SearchBot(arm, 2, "exp", Se.HybEval(f"pksearch_hyb_{arm}", int(arm[1:]) / 100), log, **kw)
    if arm.startswith("F") and arm[1:].isdigit():  # formatted facts: Jev policies + Jev P(win) leaf
        return SearchBot(arm, int(arm[1]), "exp", Se.JevEval(f"pksearch_fmt_{arm}_{os.environ.get('OPP', '')}", "pwin", fmt=True), log, **kw)
    if arm[0] in "PN" and arm[1:].isdigit():  # P(win) leaf: P=expectimax, N=minimax
        return SearchBot(arm, int(arm[1]), "exp" if arm[0] == "P" else "min", Se.JevEval(tag, "pwin"), log, **kw)
    if arm.startswith("X"):  # Jev opponent policy + hand-crafted HP leaf; XM = minimax over the same Jev-selected replies
        mode = "min" if arm[1] == "M" else "exp"
        return SearchBot(arm, int(arm[-1]), mode, Se.JevEval(tag, "score", hp_leaf=True), log, **kw)
    if arm.startswith("HM"):  # heuristic prior + HP leaf, minimax
        return SearchBot(arm, int(arm[-1]), "min", Se.HeurEval(), log, **kw)
    if arm.startswith("LP"):
        return SearchBot(arm, int(arm[2]), "exp", Se.LLMEval(tag, "pwin"), log, **kw)
    if arm[0] in "EM":
        return SearchBot(arm, int(arm[1]), "exp" if arm[0] == "E" else "min", Se.JevEval(tag), log, **kw)
    if arm[0] == "L":
        return SearchBot(arm, int(arm[1]), "exp", Se.LLMEval(tag), log, **kw)
    if arm[0] == "H":
        return SearchBot(arm, int(arm[1]), "exp", Se.HeurEval(), log, **kw)
    raise ValueError(arm)


async def main():
    arm, opp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    os.environ["OPP"] = opp
    r = random.randint(1000, 9999)
    log = os.path.join(HERE, "data", f"dec_{arm}_{opp}_{FMT}.jsonl")
    t0 = time.time()
    if opp == "ABYSSAL":  # real PokeChamp AbyssalPlayer in a separate process (abyssal_runner.py) challenges us
        a = make(arm, f"{arm}a{r}", log, 8)
        me, abn = f"{arm}a{r}", f"abyssal{r}"
        pc = os.path.join(HERE, "pokechamp")
        proc = await asyncio.create_subprocess_exec(os.path.join(HERE, ".venv-pc/bin/python"), os.path.join(HERE, "abyssal_runner.py"),
                                                    me, str(n), FMT, abn, cwd=pc)
        await a.accept_challenges(abn, n)
        await proc.wait()
    else:
        a = make(arm, f"{arm}a{r}", log, conc)
        b = make(opp, f"{opp}b{r}", log, conc)
        await a.battle_against(b, n_battles=n)
    res = {"arm": arm, "opp": opp, "fmt": FMT, "n": a.n_finished_battles, "wins": a.n_won_battles,
           "ties": a.n_tied_battles, "secs": round(time.time() - t0, 1),
           "turns": [bt.turn for bt in a.battles.values()]}
    with open(os.path.join(HERE, "data", "battles.jsonl"), "a") as f:
        for tag_, bt in a.battles.items():
            f.write(json.dumps({"arm": arm, "opp": opp, "fmt": FMT, "battle": tag_, "won": bool(bt.won), "turns": bt.turn}) + "\n")
    with open(os.path.join(HERE, "data", "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "turns"}))


if __name__ == "__main__":
    asyncio.run(main())
