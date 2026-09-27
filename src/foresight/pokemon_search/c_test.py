"""Leaf option C cost/latency test: depth-1 expectimax where each leaf is valued by Jev-policy rollouts
to the end of the battle (both sides' actions sampled from Jev choice probabilities, lockstep batched:
one Jev call per simulated turn across all rollouts). Compared with option B (P(win) noul leaf) on the same states."""
import asyncio, json, os, pickle, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/sickle/Coding/jev-context-research/pilots")
import jevlib as J
import model as Mo
import search as Se
import run as R
from poke_env import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.join(HERE, "data", "c_states.pkl")


class Capture(R.SearchBot):
    caught = []

    async def choose_move(self, battle):
        if battle.turn in (3, 9, 15) and battle.available_moves and not battle.force_switch:
            s = Mo.from_battle(battle)
            acts, _ = R.root_actions(battle, s)
            if len(acts) > 1:
                Capture.caught.append((s, acts))
        return await super().choose_move(battle)


class RolloutEval(Se.JevEval):
    def __init__(self, tag, R_=3, cap=25):
        super().__init__(tag, "pwin")
        self.R, self.cap, self.questions, self.rounds = R_, cap, 0, 0

    def run(self, policy_reqs, leaf_states):
        if policy_reqs:
            self.questions += len(policy_reqs)
            return super().run(policy_reqs, [])
        rolls = [[k, s.copy(), None] for k, s in enumerate(leaf_states) for _ in range(self.R)]
        for t in range(self.cap):
            live = [r for r in rolls if r[1].terminal() is None]
            if not live:
                break
            reqs = []
            for r in live:
                reqs += [(r[1], 0, Mo.actions(r[1], 0), None), (r[1], 1, Mo.actions(r[1], 1), None)]
            self.questions += len(reqs)
            self.rounds += 1
            pols, _ = super().run(reqs, [])
            for j, r in enumerate(live):
                a0 = random.choices(Mo.actions(r[1], 0), pols[2 * j])[0]
                a1 = random.choices(Mo.actions(r[1], 1), pols[2 * j + 1])[0]
                br = Mo.step(r[1], a0, a1, 1.0, 99)
                r[1] = random.choices([b[1] for b in br], [b[0] for b in br])[0]
        vals = [0.0] * len(leaf_states)
        for k, s, _ in rolls:
            t = s.terminal()
            vals[k] += (4 * t if t is not None else Mo.hp_value(s)) / self.R
        self.questions += 0
        return [], vals


async def collect():
    a = Capture("H1", 1, "exp", Se.HeurEval(), os.path.join(HERE, "data", "c_capture.jsonl"),
                battle_format="gen9randombattle", account_configuration=AccountConfiguration(f"cap{random.randint(100,999)}", None))
    b = SimpleHeuristicsPlayer(battle_format="gen9randombattle", account_configuration=AccountConfiguration(f"cb{random.randint(100,999)}", None))
    await a.battle_against(b, n_battles=3)
    pickle.dump(Capture.caught, open(PK, "wb"))


if __name__ == "__main__":
    if not os.path.exists(PK):
        asyncio.run(collect())
    states = pickle.load(open(PK, "rb"))[:6]
    out = []
    for i, (s, acts) in enumerate(states):
        for name, ev in (("B_pwin", Se.JevEval("pksearch_Ctest_B", "pwin")), ("C_rollout", RolloutEval("pksearch_Ctest_C"))):
            c0 = J.spend(f"pksearch_Ctest_{name[0]}")
            t0 = time.time()
            k, st = Se.search(s, acts, 1, ev, "exp", False)
            J._cache["t"] = 0
            rec = {"state": i, "leaf": name, "secs": round(time.time() - t0, 2), "leaves": st["leaves"],
                   "cost": round(J.spend(f"pksearch_Ctest_{name[0]}") - c0, 5),
                   "jev_questions": getattr(ev, "questions", None), "rollout_rounds": getattr(ev, "rounds", None),
                   "choice": Mo.act_text(s, 0, acts[k])}
            print(json.dumps(rec), flush=True)
            out.append(rec)
    json.dump(out, open(os.path.join(HERE, "data", "c_test.json"), "w"), indent=1)
