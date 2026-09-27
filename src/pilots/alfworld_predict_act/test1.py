import sys; sys.path.insert(0, ".")
from run import *
games = select_games()
for arm in ["B2", "Bnc"]:
    s = run_episode(arm, games[1])
    print(arm, {k: v for k, v in s.items() if k not in ("transcript", "game")})
    for t in s["transcript"][:8]: print("   ", {k: (v[:90] if isinstance(v, str) else v) for k, v in t.items()})
