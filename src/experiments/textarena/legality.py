"""Per-game legal-action descriptions, validation, canonicalization, and fallback (identical for all arms)."""
import re
from collections import Counter

CARD = {0: "J", 1: "Q", 2: "K"}


def unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def legal_line(game: str, base, pid: int) -> str:
    gs = base.state.game_state
    if game.startswith("LiarsDice"):
        cb, dice = gs["current_bid"], gs["dice_rolls"][pid]
        rem = "; ".join(f"Player {p}: {n}" for p, n in gs["remaining_dice"].items())
        if cb["quantity"] == 0:
            return (f"Your dice: {dice}. Dice remaining: {rem}. There is NO current bid this round: you must open with "
                    f"[Bid: quantity, face] (face 1-6). [Call] is not allowed now.")
        return (f"Your dice: {dice}. Dice remaining: {rem}. Current bid: {cb['quantity']} of face {cb['face_value']} "
                f"(by Player {gs['last_bidder_id']}). Legal actions: [Call], or [Bid: q, f] with q >= {cb['quantity']}, "
                f"f >= {cb['face_value']}, f <= 6, and not identical to the current bid (you may NOT lower the face).")
    if game.startswith("KuhnPoker"):
        acts = ", ".join(f"[{k}]" for k in gs["current_legal_action_tree"].keys())
        ch = gs["player_chips"]
        return (f"Round {gs['current_round']}. Your card: {CARD[gs['player_cards'][pid]]}. "
                f"Chips so far: you {ch[pid]}, opponent {ch[1 - pid]}. Legal actions now: {acts}\n"
                "NOTE: the 3-card deck (J, Q, K) is reshuffled at the start of EVERY round. Cards seen in past rounds say "
                "NOTHING about the current hands: the opponent holds one of the two cards you do not hold, each equally likely a priori.")
    if game.startswith("TicTacToe"):
        b = gs["board"]
        return "Legal moves: " + ", ".join(f"[{r*3+c}]" for r in range(3) for c in range(3) if b[r][c] == "")
    raise ValueError(game)


def validate(game: str, base, pid: int, action: str):
    """Return (canonical_action, None) if legal else (None, reason)."""
    gs = base.state.game_state
    a = action.strip()
    if game.startswith("LiarsDice"):
        cb = gs["current_bid"]
        if re.fullmatch(r"\[\s*call\s*\]", a, re.I):
            return ("[Call]", None) if cb["quantity"] > 0 else (None, "There is no bid to call; you must open with a bid.")
        m = re.fullmatch(r"\[\s*(?:bid\s*:?\s*)?(\d+)\s*[,\s]\s*(\d+)\s*\]", a, re.I)
        if not m:
            return None, f"Unrecognized format {a!r}; use [Bid: quantity, face] or [Call]."
        q, f = int(m.group(1)), int(m.group(2))
        if not (1 <= f <= 6) or q < 1: return None, "Face must be 1-6 and quantity >= 1."
        if q < cb["quantity"] or f < cb["face_value"]: return None, f"Bid must not lower quantity ({cb['quantity']}) or face ({cb['face_value']})."
        if q == cb["quantity"] and f == cb["face_value"]: return None, "Bid is identical to the current bid."
        return f"[Bid: {q}, {f}]", None
    if game.startswith("KuhnPoker"):
        m = re.fullmatch(r"\[\s*(check|bet|call|fold)\s*\]", a, re.I)
        legal = list(gs["current_legal_action_tree"].keys())
        if not m or m.group(1).lower() not in legal:
            return None, f"{a!r} is not legal; legal actions are " + ", ".join(f"[{k}]" for k in legal)
        return f"[{m.group(1).lower()}]", None
    if game.startswith("TicTacToe"):
        m = re.fullmatch(r"\[\s*(\d)\s*\]", a)
        if not m: return None, f"Unrecognized format {a!r}; reply like [4]."
        c = int(m.group(1)); r, col = divmod(c, 3)
        if c > 8 or gs["board"][r][col] != "": return None, f"Cell {c} is not available."
        return f"[{c}]", None
    raise ValueError(game)


def fallback(game: str, base, pid: int) -> str:
    """Deterministic legal default used only after repeated illegal attempts (logged as 'forced')."""
    gs = base.state.game_state
    if game.startswith("LiarsDice"):
        if gs["current_bid"]["quantity"] > 0: return "[Call]"
        face = Counter(gs["dice_rolls"][pid]).most_common(1)[0][0]
        return f"[Bid: 1, {face}]"
    if game.startswith("KuhnPoker"):
        legal = gs["current_legal_action_tree"].keys()
        return "[check]" if "check" in legal else "[call]"
    b = gs["board"]
    return next(f"[{r*3+c}]" for r in range(3) for c in range(3) if b[r][c] == "")
