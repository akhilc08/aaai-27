"""Protocol check: PokeChamp's own OneStepPlayer vs its AbyssalPlayer (gen8randombattle, dynamax disabled on both),
on our port-8001 server, inside PokeChamp's poke-env fork (.venv-pc, heavy ML deps stubbed as in abyssal_runner.py).
Published (PokeChamp Table 4, Gen 8 no dynamax): One Step Lookahead 44% vs Abyssal.
usage: (cwd=pokechamp) ../../pokemon_search/.venv-pc/bin/python ../pc_pair_runner.py N TAG"""
import sys, os, json, asyncio, random, importlib.abc, importlib.machinery
from unittest.mock import MagicMock

STUB = ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'google', 'ollama', 'wandb', 'datasets', 'evaluate', 'mcp',
        'pyfiglet', 'fade', 'huggingface_hub', 'sklearn', 'scipy', 'matplotlib', 'tqdm', 'anthropic', 'peft', 'safetensors')


class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in STUB:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, Finder())
sys.path.insert(0, '.')
from poke_env.player.baselines import AbyssalPlayer, OneStepPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

HERE = os.path.dirname(os.path.abspath(__file__))


async def main(n, tag):
    srv = ServerConfiguration("localhost:8001", "https://play.pokemonshowdown.com/action.php?")
    r = random.randint(10000, 99999)
    fmt = "gen8randombattle"
    a = OneStepPlayer(battle_format=fmt, account_configuration=AccountConfiguration(f"onestep{r}", ''), server_configuration=srv)
    b = AbyssalPlayer(battle_format=fmt, account_configuration=AccountConfiguration(f"abyssal{r}", ''), server_configuration=srv,
                      max_concurrent_battles=1)
    a._dynamax_disable = True
    b._dynamax_disable = True
    await a.battle_against(b, n_battles=n)
    with open(os.path.join(HERE, "data", "battles_log.jsonl"), "a") as f:
        for t, bt in a.battles.items():
            f.write(json.dumps({"arm": tag, "user": a.username, "battle": t, "won": bt.won, "turns": bt.turn, "fmt": fmt}) + "\n")
    res = {"arm": tag, "opp_model": "pokechamp OneStepPlayer (their code)", "leaf": "-", "depth": 1, "opponent": "ABYSSAL", "fmt": fmt,
           "n": a.n_finished_battles, "wins": a.n_won_battles, "ties": a.n_tied_battles, "secs": 0, "lat_mean": 0, "lat_p95": 0, "lat_max": 0, "errors": 0}
    print(json.dumps(res), flush=True)
    with open(os.path.join(HERE, "data", "arm_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[1]), sys.argv[2]))
