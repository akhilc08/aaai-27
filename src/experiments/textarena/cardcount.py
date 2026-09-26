"""Count cross-round card-counting reasoning (invalid: Kuhn deck reshuffles each round) in Kuhn transcripts."""
import json, re
from pathlib import Path

PAT = re.compile(  # explicit elimination reasoning from previously seen cards (precision over recall)
    r"only (?:the )?[JQK]\b[^.]{0,25}(?:left|remain)"
    r"|\b[JQK]\b (?:was|were|has been|is) (?:already )?(?:used|played|dealt|gone|out)\b[^.]{0,20}(?:round|earlier|before)"
    r"|unaccounted|card distribution|remaining card|cards? (?:cycle|left in the deck)", re.I)


def count_dir(d: Path, prefix: str) -> dict:
    out = {}
    for arm in ("belief", "cot2"):
        turns = hits = 0
        for f in d.glob(f"{prefix}*.json"):
            for t in json.loads(f.read_text())["turns"]:
                if t["arm"] != arm: continue
                turns += 1
                text = " ".join([t.get("raw") or "", t.get("thought") or "", json.dumps(t.get("belief") or {})])
                hits += bool(PAT.search(text))
        out[arm] = {"turns": turns, "turns_with_card_counting": hits, "rate": round(hits / turns, 4) if turns else None}
    return out


if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    print("before", count_dir(base / "games", "belief_vs_cot2__KuhnPoker"))
