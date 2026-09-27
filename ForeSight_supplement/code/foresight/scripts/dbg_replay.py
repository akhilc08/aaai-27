import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bots as B, hist_live as HL
from poke_env.player import SimpleHeuristicsPlayer, MaxBasePowerPlayer
class P(SimpleHeuristicsPlayer):
    def choose_move(self, battle):
        if battle.turn in (3, 4) and not getattr(self, 'done', 0):
            print('turn', battle.turn, 'n replay', len(battle._replay_data), [e[1] for e in battle._replay_data[-40:] if len(e) > 1])
            print(HL.actions_by_turn(battle), battle.player_role, battle.opponent_role)
            if battle.turn == 4: self.done = 1
        return super().choose_move(battle)
async def main():
    a = P(**B.kw('dbgA1', 1)); b = MaxBasePowerPlayer(**B.kw('dbgB1', 1))
    await a.battle_against(b, n_battles=1)
asyncio.run(main())
