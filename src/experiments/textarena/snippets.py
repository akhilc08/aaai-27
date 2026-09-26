"""Print belief-arm turns whose reasoning explicitly cites the belief block (candidate transcript snippets)."""
import json, re, sys
from pathlib import Path

GAMES = Path(__file__).parent / "runs" / "games"
KEYS = re.compile(r"belief|bluff|history|pattern|tendenc|opponent (likely|probably|tends|seems|has been)|they (tend|likely|probably|have been|bluffed)", re.I)

def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    for f in sorted(GAMES.glob(f"{want}*.json")):
        d = json.loads(f.read_text())
        for t in d["turns"]:
            if t["arm"] != "belief" or not t["belief"] or not KEYS.search(t["raw"]): continue
            if not re.search(r"\[(call|fold|bet)", t["action"], re.I): continue
            print(f"=== {f.name} step {t['step']} pid {t['pid']} outcome={d['outcome']} action={t['action']}")
            print("OBS_TAIL:", t["obs_tail"][-260:].replace("\n", " | "))
            print("BELIEF:", json.dumps(t["belief"])[:600])
            print("REASONING:", t["raw"][-450:].strip(), "\n")

if __name__ == "__main__":
    main()
