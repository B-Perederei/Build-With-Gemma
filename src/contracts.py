"""
contracts.py — the single source of truth for the 4-stage pipeline.

ALL FOUR teammates import from this file. It defines the exact shape of the two
dicts that cross stage boundaries, so people can build in parallel without
blocking each other.

    Person A (streamer)  ->  Person B (detector)  ->  Person C (agent)  ->  Person D (display)
                                        |                        |
                                   CONTRACT A               CONTRACT B

HOW TO USE THIS FILE
--------------------
- If your upstream stage isn't ready, build against the EXAMPLE_* dict below.
  It has the exact same shape as the real thing, so swapping later changes nothing.
- Right before you EMIT a dict, call the matching validate_*() function. It fails
  loudly on a typo NOW instead of silently at hour 4.
- Do not add or rename keys without telling the whole team. This file is the
  agreement. If it changes, everyone needs to know.

RUN `python contracts.py` TO SELF-TEST (checks the examples pass their own validators).
"""

from __future__ import annotations
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Allowed value sets (single source of truth for the string enums)
# ---------------------------------------------------------------------------

SEVERITIES = {"low", "medium", "high"}
ACTIONS = {"monitor", "isolate_channel", "safe_mode", "escalate"}
METHODS = {"zscore", "lstm", "isoforest"}
POLICY_DECISIONS = {"AUTONOMOUS_ACT", "HOLD_LOW_CONFIDENCE", "QUEUE_FOR_GROUND"}

# Which actions the spacecraft is pre-cleared to take on its own.
# The policy engine (Person C) uses this. Kept here so it's visible to everyone.
AUTONOMOUS_OK = {"monitor", "isolate_channel"}
NEEDS_GROUND = {"safe_mode", "escalate"}
CONFIDENCE_GATE = 0.60  # below this, hold for human review


# ===========================================================================
# CONTRACT A — Detector (Person B)  ->  Gemma agent (Person C)
# Emitted ONLY when the detector flags an anomaly. Otherwise the detector
# returns None.
# ===========================================================================

EXAMPLE_DETECTION_EVENT = {
    "event_id": "evt_S1_5305",   # unique id, convention: evt_<channel>_<timestep>
    "channel": "S-1",            # telemetry channel id from the CSV
    "spacecraft": "SMAP",        # "SMAP" or "MSL"
    "timestep": 5305,            # int, index where the flag fired
    "flagged_range": [5300, 5741],  # [start, end] inclusive index range
    "score": 7.8,               # float, z-score peak OR lstm error peak
    "method": "zscore",         # "zscore" | "lstm" | "isoforest"
    "channel_prefix": "S",      # first letter of channel, used to look up protocol text
}

DETECTION_EVENT_KEYS = set(EXAMPLE_DETECTION_EVENT.keys())


def validate_detection_event(d: dict) -> dict:
    """Person B calls this right before emitting a detection event.

    Raises AssertionError with a clear message if the dict is malformed.
    Returns the dict unchanged if valid (so you can write `return validate(...)`).
    """
    missing = DETECTION_EVENT_KEYS - d.keys()
    extra = d.keys() - DETECTION_EVENT_KEYS
    assert not missing, f"[Contract A] detection event MISSING keys: {missing}"
    assert not extra, f"[Contract A] detection event has UNEXPECTED keys: {extra}"

    assert isinstance(d["event_id"], str), "[Contract A] event_id must be str"
    assert isinstance(d["channel"], str), "[Contract A] channel must be str"
    assert d["spacecraft"] in {"SMAP", "MSL"}, \
        f"[Contract A] spacecraft must be SMAP or MSL, got {d['spacecraft']!r}"
    assert isinstance(d["timestep"], int), "[Contract A] timestep must be int"

    rng = d["flagged_range"]
    assert (isinstance(rng, (list, tuple)) and len(rng) == 2
            and all(isinstance(x, int) for x in rng) and rng[0] <= rng[1]), \
        f"[Contract A] flagged_range must be [start, end] ints with start<=end, got {rng!r}"

    assert isinstance(d["score"], (int, float)), "[Contract A] score must be a number"
    assert d["method"] in METHODS, \
        f"[Contract A] method must be one of {METHODS}, got {d['method']!r}"
    assert isinstance(d["channel_prefix"], str) and len(d["channel_prefix"]) >= 1, \
        "[Contract A] channel_prefix must be a non-empty str"
    return d


def make_detection_event(channel: str, spacecraft: str, timestep: int,
                         flagged_range, score: float, method: str) -> dict:
    """Helper so Person B doesn't hand-build the dict (and can't typo a key).

    Auto-fills event_id and channel_prefix. Validates before returning.
    """
    event = {
        "event_id": f"evt_{channel.replace('-', '')}_{timestep}",
        "channel": channel,
        "spacecraft": spacecraft,
        "timestep": int(timestep),
        "flagged_range": [int(flagged_range[0]), int(flagged_range[1])],
        "score": float(score),
        "method": method,
        "channel_prefix": channel.split("-")[0],
    }
    return validate_detection_event(event)


# ===========================================================================
# CONTRACT B — Gemma agent (Person C)  ->  Display (Person D)
# Emitted once per detection event. Includes BOTH Gemma's reasoning AND the
# deterministic policy verdict, so there are only two contracts total.
# ===========================================================================

EXAMPLE_GEMMA_DECISION = {
    "event_id": "evt_S1_5305",      # MUST match the detection event it responds to
    # --- from Gemma (the LLM) ---
    "severity": "medium",           # "low" | "medium" | "high"
    "confidence": 0.74,             # float 0.0 - 1.0
    "rationale": "A sharp 7.8-sigma single-point deviation on sensor S-1, "
                 "consistent with a transient glitch rather than a sustained fault.",
    "recommended_action": "isolate_channel",  # what Gemma suggests
    # --- from the deterministic policy engine (NOT the LLM) ---
    "policy_decision": "AUTONOMOUS_ACT",   # AUTONOMOUS_ACT | HOLD_LOW_CONFIDENCE | QUEUE_FOR_GROUND
    "command": "isolate_channel",          # the action actually taken, or None if held
    "policy_reason": "action within bounded-autonomy set and confidence 0.74 >= 0.60",
}

GEMMA_DECISION_KEYS = set(EXAMPLE_GEMMA_DECISION.keys())


def validate_gemma_decision(d: dict) -> dict:
    """Person C calls this right before emitting a decision to Person D."""
    missing = GEMMA_DECISION_KEYS - d.keys()
    extra = d.keys() - GEMMA_DECISION_KEYS
    assert not missing, f"[Contract B] gemma decision MISSING keys: {missing}"
    assert not extra, f"[Contract B] gemma decision has UNEXPECTED keys: {extra}"

    assert isinstance(d["event_id"], str), "[Contract B] event_id must be str"
    assert d["severity"] in SEVERITIES, \
        f"[Contract B] severity must be one of {SEVERITIES}, got {d['severity']!r}"

    conf = d["confidence"]
    assert isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0, \
        f"[Contract B] confidence must be a float in [0,1], got {conf!r}"

    assert isinstance(d["rationale"], str) and d["rationale"].strip(), \
        "[Contract B] rationale must be a non-empty str"
    assert d["recommended_action"] in ACTIONS, \
        f"[Contract B] recommended_action must be one of {ACTIONS}, got {d['recommended_action']!r}"
    assert d["policy_decision"] in POLICY_DECISIONS, \
        f"[Contract B] policy_decision must be one of {POLICY_DECISIONS}, got {d['policy_decision']!r}"
    assert d["command"] is None or d["command"] in ACTIONS, \
        f"[Contract B] command must be None or one of {ACTIONS}, got {d['command']!r}"
    assert isinstance(d["policy_reason"], str), "[Contract B] policy_reason must be str"
    return d


def apply_policy(gemma_output: dict, event_id: str) -> dict:
    """The deterministic policy engine (Person C owns this).

    Takes the parsed LLM output (severity/confidence/rationale/recommended_action)
    and the event_id, applies the bounded-autonomy rules, and returns a full,
    validated Contract B dict.

    This never trusts the LLM blindly: an action only executes autonomously if it
    is pre-cleared AND confidence clears the gate.
    """
    action = gemma_output["recommended_action"]
    conf = float(gemma_output["confidence"])

    if conf < CONFIDENCE_GATE:
        decision, command = "HOLD_LOW_CONFIDENCE", None
        reason = f"confidence {conf:.2f} < {CONFIDENCE_GATE} gate; holding for review"
    elif action in AUTONOMOUS_OK:
        decision, command = "AUTONOMOUS_ACT", action
        reason = f"action '{action}' pre-cleared and confidence {conf:.2f} >= {CONFIDENCE_GATE}"
    else:  # action in NEEDS_GROUND
        decision, command = "QUEUE_FOR_GROUND", action
        reason = f"action '{action}' requires ground approval; queued for next window"

    decision_dict = {
        "event_id": event_id,
        "severity": gemma_output["severity"],
        "confidence": conf,
        "rationale": gemma_output["rationale"],
        "recommended_action": action,
        "policy_decision": decision,
        "command": command,
        "policy_reason": reason,
    }
    return validate_gemma_decision(decision_dict)


# ===========================================================================
# Shared protocol snippets (Person B looks these up by channel_prefix).
# Kept here so Person C can also reference them when building prompts.
# Expand or edit as a team — these are placeholders you can improve.
# ===========================================================================

PROTOCOL_SNIPPETS = {
    "S": ("Channel prefix S (attitude/sensor group): sharp single-point deviations "
          ">5 sigma are usually transient sensor glitches unless sustained across "
          ">50 timesteps, in which case escalate."),
    "P": ("Channel prefix P (power group): deviations often correlate with eclipse "
          "entry/exit and expected solar-array load changes. Sustained drops with no "
          "orbital explanation may indicate a real power fault."),
    "T": ("Channel prefix T (thermal group): gradual drifts are normal across an "
          "orbit; abrupt jumps may indicate a heater or radiator issue."),
    "R": ("Channel prefix R (radiation group): spikes over the poles or the South "
          "Atlantic Anomaly are expected environmental effects, not faults."),
    "E": ("Channel prefix E (general/engineering group): evaluate deviation magnitude "
          "and duration; short single-point spikes are typically transient."),
}

DEFAULT_SNIPPET = ("No specific protocol on file for this channel group. Assess the "
                   "deviation magnitude and duration; short single-point spikes are "
                   "usually transient, sustained deviations warrant escalation.")


def get_protocol_snippet(channel_prefix: str) -> str:
    """Return the protocol text for a channel prefix, or a safe default."""
    return PROTOCOL_SNIPPETS.get(channel_prefix, DEFAULT_SNIPPET)


# ===========================================================================
# Self-test — run `python contracts.py` to confirm the examples are valid.
# ===========================================================================

if __name__ == "__main__":
    validate_detection_event(EXAMPLE_DETECTION_EVENT)
    validate_gemma_decision(EXAMPLE_GEMMA_DECISION)

    # make_detection_event round-trip
    ev = make_detection_event("S-1", "SMAP", 5305, [5300, 5741], 7.8, "zscore")
    assert ev["event_id"] == "evt_S1_5305"
    assert ev["channel_prefix"] == "S"

    # apply_policy on a typical medium/high-confidence isolate -> AUTONOMOUS_ACT
    fake_llm = {
        "severity": "medium",
        "confidence": 0.74,
        "rationale": "transient glitch",
        "recommended_action": "isolate_channel",
    }
    dec = apply_policy(fake_llm, ev["event_id"])
    assert dec["policy_decision"] == "AUTONOMOUS_ACT"
    assert dec["command"] == "isolate_channel"

    # low confidence -> HOLD
    fake_llm_low = dict(fake_llm, confidence=0.4)
    assert apply_policy(fake_llm_low, ev["event_id"])["policy_decision"] == "HOLD_LOW_CONFIDENCE"

    # escalate action -> QUEUE_FOR_GROUND
    fake_llm_esc = dict(fake_llm, recommended_action="escalate", confidence=0.9)
    assert apply_policy(fake_llm_esc, ev["event_id"])["policy_decision"] == "QUEUE_FOR_GROUND"

    print("contracts.py self-test passed ✓")
    print("  Contract A example:", EXAMPLE_DETECTION_EVENT["event_id"])
    print("  Contract B example:", EXAMPLE_GEMMA_DECISION["policy_decision"])