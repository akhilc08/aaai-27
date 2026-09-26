"""Shared LLM ReAct-style agent pieces for ALFWorld."""
import os, re, sys
sys.path.insert(0, "/Users/sickle/Coding/aaai-27/experiments/pilots")
import jevlib as J
TAG = "imagination_planning"

SYS = ("You are an agent in a text household (ALFWorld). Achieve the task with the fewest steps. "
       "Useful facts: you can hold one object; 'take X from Y', 'put X in/on Y', 'open Y', 'go to Y'; "
       "heat with 'heat X with microwave 1', cool with 'cool X with fridge 1', clean with 'clean X with sinkbasin 1', "
       "'use desklamp 1' to look at something under light. Search likely places for objects and do not revisit empty places. "
       "Reply exactly as:\nThink: <one short sentence>\nAction: <one action>")

def target_of(gamefile):
    parts = gamefile.split("/")[-3].split("-")
    return parts[1].lower()

def render(init_obs, hist, adm, note=None, k=12):
    h = "\n".join(f"> {a}\n{o}" for a, o in hist[-k:])
    s = f"{init_obs.strip()}\n\nRecent steps (most recent last):\n{h or '(none)'}\n\nAdmissible actions: {', '.join(adm)}"
    if note:
        s += f"\n\nNOTE: {note}"
    return s

def parse_action(text, adm):
    m = re.findall(r"Action:\s*(.+)", text)
    a = (m[-1] if m else text.strip().split("\n")[-1]).strip().strip(".").strip().lower()
    return a

def react_step(init_obs, hist, adm, note=None):
    t, u = J.llm([{"role": "system", "content": SYS}, {"role": "user", "content": render(init_obs, hist, adm, note)}],
                 max_tokens=120, tag=TAG)
    return parse_action(t, adm), u.get("cost", 0) or 0
