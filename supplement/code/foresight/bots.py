"""Players on our own Showdown server (port 8001): heuristic bots, logging wrappers, search bot."""
import asyncio, json, os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
import model as Mo
import search as Se

PORT = int(os.environ.get("PS_PORT", "8001"))
SERVER = ServerConfiguration(f"ws://localhost:{PORT}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
FMT = os.environ.get("FMT", "gen9randombattle")


def root_actions(battle, s):
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


def order_label(order):
    """Abstract label of a chosen order: ('m', move_id) or ('s', species)."""
    o = getattr(order, "order", order)
    if o is None or isinstance(o, str):
        return None
    if hasattr(o, "species"):
        return ["s", o.species]
    if hasattr(o, "id"):
        return ["m", o.id]
    return None


SKIP = {"t:", "inactive", "j", "l", "rule", "gametype", "gen", "tier", "rated", "title", "init", "teamsize", "start", "upkeep", "",
        "player", "request", "timer", "raw", "c", "badge", "teampreview", "clearpoke", "poke"}


def raw_text(battle, max_lines=80):
    """Raw Showdown protocol log so far (no hand-built features), noise lines dropped, POV marked."""
    out = [f"we are {battle.player_role}"]
    for ev in battle._replay_data:
        if len(ev) < 2 or ev[1] in SKIP:
            continue
        out.append("|".join(ev[1:]))
    return "\n".join(out[:1] + out[1:][-max_lines:])


class LogMixin:
    """Records (state, chosen action) at every decision; rows flushed with win label at battle end."""
    rows = None

    def _log_decision(self, battle, order):
        if self.rows is None:
            self.rows = {}
        try:
            s = Mo.from_battle(battle)
            forced = bool(battle.force_switch) or not battle.available_moves
            row = {"turn": battle.turn, "forced": forced, "act": order_label(order), "state": pickle.dumps(s)}
            if os.environ.get("LIVEHIST") and not forced:
                import hist_live as HL
                snaps = self.__dict__.setdefault("_snaps", {}).setdefault(battle.battle_tag, {})
                row["live_hist"] = HL.live_hist(battle, snaps)
                snaps[battle.turn] = s
            if os.environ.get("RAWLOG"):
                row["raw"] = raw_text(battle)
            self.rows.setdefault(battle.battle_tag, []).append(row)
        except Exception as e:
            pass

    def dump(self, path):
        with open(path, "ab") as f:
            for tag, rs in (self.rows or {}).items():
                b = self.battles.get(tag)
                if b is None or not b.finished:
                    continue
                for r in rs:
                    r.update(battle=tag, player=self.username, won=bool(b.won), tie=b.won is None, cls=type(self).__name__)
                pickle.dump(rs, f)
        self.rows = {}


def logged(cls):
    class L(LogMixin, cls):
        def choose_move(self, battle):
            o = super().choose_move(battle)
            if asyncio.iscoroutine(o):
                async def w():
                    r = await o
                    self._log_decision(battle, r)
                    return r
                return w()
            self._log_decision(battle, o)
            return o
    L.__name__ = cls.__name__
    return L


class SearchBot(Player):
    def __init__(self, arm, depth, mode, ev, log, **kw):
        super().__init__(**kw)
        self.arm, self.depth, self.mode, self.ev, self.log = arm, depth, mode, ev, log

    async def choose_move(self, battle):
        t0 = time.time()
        try:
            s = Mo.from_battle(battle)
            hist = None
            if hasattr(self.ev, "set_root"):
                import hist_live as HL
                snaps = self.__dict__.setdefault("_hsnaps", {}).setdefault(battle.battle_tag, {})
                hist = HL.live_hist(battle, snaps)
                if not (bool(battle.force_switch) or not battle.available_moves):
                    snaps[battle.turn] = Mo.from_battle(battle)
            if getattr(self, "save_turns", None) is not None and not (bool(battle.force_switch) or not battle.available_moves):
                self.save_turns.setdefault(battle.battle_tag, []).append(
                    {"turn": battle.turn, "state": pickle.dumps(Mo.from_battle(battle)), "hist": hist})
            acts, orders = root_actions(battle, s)
            if not acts:
                return self.choose_random_move(battle)
            if len(acts) == 1:
                return self.create_order(orders[0])
            forced = bool(battle.force_switch) or not battle.available_moves
            k, st = await asyncio.to_thread(self._search, s, acts, forced, hist)
            st.update(arm=self.arm, battle=battle.battle_tag, turn=battle.turn, lat=round(time.time() - t0, 3),
                      n_acts=len(acts), forced=forced)
            if self.log:
                try:
                    with open(self.log, "a") as f:
                        f.write(json.dumps(st, default=float) + "\n")
                except Exception:
                    pass
            return self.create_order(orders[k])
        except Exception as e:
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps({"arm": self.arm, "error": repr(e)[:300], "lat": round(time.time() - t0, 3)}) + "\n")
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "battles_log.jsonl"), "a") as f:
                f.write(json.dumps({"arm": self.arm, "user": self.username, "battle": battle.battle_tag, "won": battle.won,
                                    "turns": battle.turn, "fmt": battle.battle_tag.split("-")[1]}) + "\n")
        except Exception:
            pass
        if getattr(self, "save_turns", None) is not None and self.turns_path:
            import hist_live as HL
            try:
                abt = HL.actions_by_turn(battle)
                rows = self.save_turns.pop(battle.battle_tag, [])
                for r in rows:
                    r["opp_act"] = abt.get(r["turn"], {}).get(battle.opponent_role)
                    r["battle"] = battle.battle_tag; r["won"] = bool(battle.won)
                with open(self.turns_path, "ab") as f:
                    pickle.dump(rows, f)
            except Exception as e:
                print("save_turns error", repr(e)[:200], flush=True)

    def _search(self, s, acts, forced, hist=None):
        if hasattr(self.ev, "set_root"):
            self.ev.set_root(s, hist)
        if getattr(self, "strat", None) is not None:
            return self.strat.choose(s, acts, forced, self.ev)
        kw = {}
        if os.environ.get("OPP_MASS"):
            kw["opp_mass"] = float(os.environ["OPP_MASS"])
        if os.environ.get("OPP_CAP"):
            kw["opp_cap"], kw["opp_cap_deep"] = [int(x) for x in os.environ["OPP_CAP"].split(",")]
        if os.environ.get("CHANCE"):  # "mass_root,cap_root,mass_deep,cap_deep"
            a, b, c, d = os.environ["CHANCE"].split(",")
            kw["chance_root"], kw["chance_deep"] = (float(a), int(b)), (float(c), int(d))
        if os.environ.get("OUR_K"):
            kw["our_k"] = int(os.environ["OUR_K"])
        return Se.search(s, acts, self.depth, self.ev, self.mode, forced, **kw)


class NoDmaxSH(SimpleHeuristicsPlayer):
    @staticmethod
    def _should_dynamax(battle, n_remaining_mons):
        return False


BASE = {"SH": SimpleHeuristicsPlayer, "MBP": MaxBasePowerPlayer, "RND": RandomPlayer, "AB": NoDmaxSH}


def kw(name, conc):
    return dict(battle_format=FMT, max_concurrent_battles=conc, server_configuration=SERVER,
                account_configuration=AccountConfiguration(name, None))
