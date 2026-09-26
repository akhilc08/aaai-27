"""Hybrid Catan player: LLM negotiates domestic trades, heuristic policy builds.

Feasibility spike for AAAI-27 student abstract. Built on catanatron>=3.3.0
(git master), which exposes domestic trades via OFFER_TRADE / ACCEPT_TRADE /
REJECT_TRADE / CONFIRM_TRADE / CANCEL_TRADE.

Players defined here:
  HybridLLMTrader            - ValueFunctionPlayer for all build/placement
                               decisions; an LLM (via OpenRouter) decides
                               (a) whether to OFFER_TRADE once per turn and
                               (b) ACCEPT/REJECT incoming offers.
  TradeAwareValueFunctionPlayer - ValueFunctionPlayer that evaluates incoming
                               offers by simulating the resource swap. Needed
                               because vanilla ValueFunctionPlayer rejects
                               100% of offers (apply_accept_trade does not move
                               resources, so ACCEPT/REJECT tie and REJECT wins).
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    freqdeck_contains,
)
from catanatron.models.enums import RESOURCES, Action, ActionPrompt, ActionType
from catanatron.players.value import ValueFunctionPlayer, get_value_fn
from catanatron.state_functions import (
    get_actual_victory_points,
    get_player_freqdeck,
    get_visible_victory_points,
    player_freqdeck_add,
    player_freqdeck_subtract,
    player_has_rolled,
    player_key,
)

ENV_PATH = "/Users/sickle/Coding/context-research/.env"
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
FALLBACK_MODELS = ["deepseek/deepseek-chat", "z-ai/glm-4.5-air"]
HARD_BUDGET_USD = 3.0
SOFT_BUDGET_USD = 2.5  # stop making LLM calls past this; fall back to heuristic

_RES_LOWER = [r.lower() for r in RESOURCES]
_COSTS = {
    "road": ROAD_COST_FREQDECK,
    "settlement": SETTLEMENT_COST_FREQDECK,
    "city": CITY_COST_FREQDECK,
    "dev_card": DEVELOPMENT_CARD_COST_FREQDECK,
}


class LLMBudget:
    """Process-wide cost/usage tracker (thread-safe)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cost_usd = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors = 0
        self.disabled = False

    def add(self, usage) -> None:
        with self.lock:
            self.calls += 1
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            cost = getattr(usage, "cost", None)
            if cost is None and hasattr(usage, "model_dump"):
                cost = usage.model_dump().get("cost")
            self.cost_usd += float(cost or 0.0)
            if self.cost_usd >= SOFT_BUDGET_USD:
                self.disabled = True

    def summary(self) -> dict:
        return {
            "llm_calls": self.calls,
            "llm_cost_usd": round(self.cost_usd, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_errors": self.errors,
            "llm_disabled_by_budget": self.disabled,
        }


BUDGET = LLMBudget()
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv(ENV_PATH)
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not found in environment")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client


def llm_json(system: str, user: str, model: str, max_tokens: int = 160) -> Optional[dict]:
    """One chat call; returns parsed JSON object or None on any failure."""
    if BUDGET.disabled:
        return None
    client = get_client()
    for m in [model] + FALLBACK_MODELS:
        try:
            r = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"usage": {"include": True}},
            )
            BUDGET.add(r.usage)
            text = r.choices[0].message.content or ""
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001 - spike code; any failure -> fallback
            BUDGET.errors += 1
            continue
    return None


# --------------------------------------------------------------------------
# State summary for the LLM
# --------------------------------------------------------------------------


def _fd_str(fd) -> str:
    parts = [f"{n} {r}" for r, n in zip(_RES_LOWER, fd) if n > 0]
    return ", ".join(parts) if parts else "nothing"


def _missing(hand, cost):
    return [max(0, c - h) for h, c in zip(hand, cost)]


def summarize_for_offer(game, color) -> str:
    st = game.state
    hand = get_player_freqdeck(st, color)
    lines = [f"You are {color.value}. Your hand: {_fd_str(hand)} ({sum(hand)} cards)."]
    lines.append(
        f"Your victory points: {get_actual_victory_points(st, color)} (need {game.vps_to_win})."
    )
    for name, cost in _COSTS.items():
        miss = _missing(hand, cost)
        if sum(miss) == 0:
            lines.append(f"- can afford {name} now")
        else:
            lines.append(f"- {name}: missing {_fd_str(miss)}")
    lines.append("Opponents (visible VP, cards in hand):")
    for c in st.colors:
        if c == color:
            continue
        fd = get_player_freqdeck(st, c)
        lines.append(f"- {c.value}: {get_visible_victory_points(st, c)} VP, {sum(fd)} cards")
    return "\n".join(lines)


OFFER_SYSTEM = (
    "You are the trade negotiator for a Settlers of Catan bot. Another module "
    "handles all building. Your only job: decide whether to propose ONE "
    "player-to-player trade this turn that helps you complete a purchase "
    "(city > settlement > dev card > road) without giving away cards you need. "
    "Opponents accept only if the trade also helps them, so keep offers fair "
    "(1-for-1 or 2-for-1). Never offer resources you do not hold. Never ask for "
    "a resource you also offer. Respond with JSON only: "
    '{"offer": null} or {"offer": {"give": {"wood": 1}, "get": {"ore": 1}}}. '
    "Resource names: wood, brick, sheep, wheat, ore."
)

RESPOND_SYSTEM = (
    "You are the trade negotiator for a Settlers of Catan bot. An opponent "
    "proposed a trade. Accept only if it moves you closer to your next purchase "
    "(city > settlement > dev card > road) and does not hand a leader what they "
    "need. Respond with JSON only: {\"accept\": true} or {\"accept\": false}."
)


def parse_offer(obj, hand) -> Optional[tuple]:
    """Validate LLM offer JSON against the hand and catanatron's trade rules."""
    if not obj or not isinstance(obj.get("offer"), dict):
        return None
    give = [0] * 5
    get = [0] * 5
    for side, arr in (("give", give), ("get", get)):
        d = obj["offer"].get(side) or {}
        if not isinstance(d, dict):
            return None
        for k, v in d.items():
            k = str(k).lower().strip()
            if k not in _RES_LOWER:
                return None
            try:
                n = int(v)
            except (TypeError, ValueError):
                return None
            if n < 0:
                return None
            arr[_RES_LOWER.index(k)] += n
    if sum(give) == 0 or sum(get) == 0:
        return None
    if sum(give) > 3 or sum(get) > 3:
        return None
    if any(g > 0 and r > 0 for g, r in zip(give, get)):
        return None
    if not freqdeck_contains(hand, give):
        return None
    return tuple(give + get)


# --------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------


class TradeAwareValueFunctionPlayer(ValueFunctionPlayer):
    """ValueFunctionPlayer that accepts offers which raise its heuristic value."""

    LABEL = "Trade-aware Value Function"

    def decide(self, game, playable_actions):
        st = game.state
        if st.current_prompt == ActionPrompt.DECIDE_TRADE:
            if st.colors[st.current_turn_index] == self.color:
                return playable_actions[0]  # catanatron asks offerer too; reject own offer
            accept = next(
                (a for a in playable_actions if a.action_type == ActionType.ACCEPT_TRADE),
                None,
            )
            if accept is None:
                return playable_actions[0]
            value_fn = get_value_fn(self.value_fn_builder_name, self.params.weights)
            before = value_fn(game, self.color)
            sim = game.copy()
            offering = list(st.current_trade[:5])
            asking = list(st.current_trade[5:10])
            # We are the responder: we give `asking`, receive `offering`.
            player_freqdeck_subtract(sim.state, self.color, asking)
            player_freqdeck_add(sim.state, self.color, offering)
            after = value_fn(sim, self.color)
            return accept if after > before else playable_actions[0]
        return super().decide(game, playable_actions)


class HybridLLMTrader(ValueFunctionPlayer):
    """ValueFunctionPlayer for building; LLM for domestic-trade negotiation."""

    LABEL = "Hybrid LLM negotiator + value-function builder"

    @dataclass(frozen=True)
    class Params:
        value_fn: str = "base"
        epsilon: Optional[float] = None
        weights: Optional[dict] = None
        model: str = DEFAULT_MODEL
        max_offers_per_turn: int = 1

    def __init__(self, color, params=None):
        super().__init__(color, params)
        self.stats = {
            "offers_made": 0,
            "offers_accepted_by_someone": 0,
            "trades_confirmed": 0,
            "incoming_offers": 0,
            "incoming_accepted": 0,
            "llm_offer_null": 0,
            "llm_offer_invalid": 0,
        }
        self._offers_this_turn = 0
        self._turn_key = None

    def _turn(self, st):
        return (st.num_turns, st.current_turn_index)

    def decide(self, game, playable_actions):
        st = game.state
        prompt = st.current_prompt

        if prompt == ActionPrompt.DECIDE_TRADE:
            return self._respond(game, playable_actions)

        if prompt == ActionPrompt.DECIDE_ACCEPTEES:
            confirms = [
                a
                for a in playable_actions
                if a.action_type == ActionType.CONFIRM_TRADE and a.value[10] != self.color
            ]
            if confirms:
                self.stats["offers_accepted_by_someone"] += 1
                # Prefer trading with the player who has the fewest visible VPs.
                best = min(
                    confirms, key=lambda a: get_visible_victory_points(st, a.value[10])
                )
                self.stats["trades_confirmed"] += 1
                return best
            return playable_actions[0]

        if prompt == ActionPrompt.PLAY_TURN and player_has_rolled(st, self.color):
            key = self._turn(st)
            if key != self._turn_key:
                self._turn_key = key
                self._offers_this_turn = 0
            hand = get_player_freqdeck(st, self.color)
            if (
                self._offers_this_turn < self.params.max_offers_per_turn
                and sum(hand) >= 2
                and not BUDGET.disabled
            ):
                self._offers_this_turn += 1
                offer = self._propose(game, hand)
                if offer is not None:
                    self.stats["offers_made"] += 1
                    return Action(self.color, ActionType.OFFER_TRADE, offer)

        return super().decide(game, playable_actions)

    def _propose(self, game, hand):
        user = summarize_for_offer(game, self.color)
        obj = llm_json(OFFER_SYSTEM, user, self.params.model)
        if obj is None or obj.get("offer") is None:
            self.stats["llm_offer_null"] += 1
            return None
        offer = parse_offer(obj, hand)
        if offer is None:
            self.stats["llm_offer_invalid"] += 1
        return offer

    def _respond(self, game, playable_actions):
        st = game.state
        if st.colors[st.current_turn_index] == self.color:
            return playable_actions[0]  # own offer echoed back by engine; reject
        self.stats["incoming_offers"] += 1
        accept = next(
            (a for a in playable_actions if a.action_type == ActionType.ACCEPT_TRADE), None
        )
        if accept is None:
            return playable_actions[0]
        offering = st.current_trade[:5]
        asking = st.current_trade[5:10]
        proposer = st.colors[st.current_turn_index]
        user = (
            summarize_for_offer(game, self.color)
            + f"\n{proposer.value} offers you {_fd_str(offering)} in exchange for "
            f"your {_fd_str(asking)}. Accept?"
        )
        obj = llm_json(RESPOND_SYSTEM, user, self.params.model, max_tokens=40)
        if obj is None:
            # Fallback: heuristic (same rule as TradeAwareValueFunctionPlayer)
            value_fn = get_value_fn(self.value_fn_builder_name, self.params.weights)
            sim = game.copy()
            player_freqdeck_subtract(sim.state, self.color, list(asking))
            player_freqdeck_add(sim.state, self.color, list(offering))
            do_accept = value_fn(sim, self.color) > value_fn(game, self.color)
        else:
            do_accept = bool(obj.get("accept"))
        if do_accept:
            self.stats["incoming_accepted"] += 1
            return accept
        return playable_actions[0]


class HeuristicTrader(TradeAwareValueFunctionPlayer):
    """Ablation: same as HybridLLMTrader but offers are chosen by a fixed rule.

    Rule: pick the cheapest purchase that is exactly one resource short; offer
    one unit of the resource we hold the most of (that the purchase does not
    need) for the missing one.
    """

    LABEL = "Heuristic negotiator + value-function builder"

    def __init__(self, color, params=None):
        super().__init__(color, params)
        self.stats = {"offers_made": 0, "trades_confirmed": 0}
        self._turn_key = None
        self._offered = False

    def decide(self, game, playable_actions):
        st = game.state
        if st.current_prompt == ActionPrompt.DECIDE_ACCEPTEES:
            confirms = [
                a
                for a in playable_actions
                if a.action_type == ActionType.CONFIRM_TRADE and a.value[10] != self.color
            ]
            if confirms:
                self.stats["trades_confirmed"] += 1
                return confirms[0]
            return playable_actions[0]
        if st.current_prompt == ActionPrompt.PLAY_TURN and player_has_rolled(st, self.color):
            key = (st.num_turns, st.current_turn_index)
            if key != self._turn_key:
                self._turn_key, self._offered = key, False
            if not self._offered:
                self._offered = True
                hand = get_player_freqdeck(st, self.color)
                for name in ("city", "settlement", "dev_card", "road"):
                    cost = _COSTS[name]
                    miss = _missing(hand, cost)
                    if sum(miss) != 1:
                        continue
                    want = miss.index(1)
                    surplus = [(hand[i] - cost[i], i) for i in range(5) if i != want]
                    amt, give_i = max(surplus)
                    if amt <= 0:
                        continue
                    give = [0] * 5
                    get = [0] * 5
                    give[give_i] = 1
                    get[want] = 1
                    self.stats["offers_made"] += 1
                    return Action(self.color, ActionType.OFFER_TRADE, tuple(give + get))
        return super().decide(game, playable_actions)


__all__ = [
    "BUDGET",
    "HARD_BUDGET_USD",
    "HybridLLMTrader",
    "HeuristicTrader",
    "TradeAwareValueFunctionPlayer",
]
