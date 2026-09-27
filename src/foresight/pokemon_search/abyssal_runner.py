"""Runs PokeChamp's real AbyssalPlayer (their poke-env fork, heavy ML deps stubbed) and challenges our bot on the local server.
usage: (cwd=pokechamp) ../.venv-pc/bin/python ../abyssal_runner.py OUR_USERNAME N FORMAT MYNAME"""
import sys, importlib.abc, importlib.machinery, asyncio
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
from poke_env.player.baselines import AbyssalPlayer
from poke_env.player.player import Player
import inspect

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('import ok'); print([n for n in dir(Player) if 'challenge' in n]); print(inspect.signature(Player.__init__)); sys.exit()
    our, n, fmt, me = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    from poke_env.ps_client.account_configuration import AccountConfiguration

    async def main():
        p = AbyssalPlayer(battle_format=fmt, account_configuration=AccountConfiguration(me, ''), max_concurrent_battles=8)
        p._dynamax_disable = True  # "dynamax is disabled for local battles" (PokeChamp prompt_eval.py)
        await asyncio.sleep(3)
        await p.send_challenges(our, n_challenges=n)
        print(f"abyssal done: {p.n_won_battles}/{p.n_finished_battles}", flush=True)

    asyncio.run(main())
