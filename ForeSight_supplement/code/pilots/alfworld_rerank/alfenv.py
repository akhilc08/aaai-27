"""Minimal single-game ALFWorld text env (TextWorld backend, valid_unseen = eval_out_of_distribution)."""
import glob, os, re
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/json_2.1.1/valid_unseen")

def games():
    return sorted(glob.glob(DATA + "/*/*/game.tw-pddl"))

class Env:
    def __init__(self, gamefile):
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        self.env = textworld.start(gamefile, request_infos=infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    def reset(self):
        s = self.env.reset()
        obs = s["feedback"]
        self.goal = re.search(r"Your task is to: (.*)", obs).group(1).strip()
        return obs, self._adm(s)
    def _adm(self, s):
        return [a for a in s["admissible_commands"] if a not in ("look", "inventory", "help")] + ["look", "inventory"]
    def step(self, a):
        s, _, done = self.env.step(a)
        return s["feedback"], bool(s["won"]), self._adm(s)
