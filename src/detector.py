from typing import Optional, Dict, Any
import numpy as np

from contracts import make_detection_event


def calculate_zscore(value: float, history: list[float]) -> float:
    """Calculates z-score of current value against historical window."""
    if not history or len(history) < 5:
        return 0.0
    mean = float(np.mean(history))
    std = float(np.std(history))
    if std == 0:
        return 0.0
    return abs(value - mean) / std


def detect(
    tick: Dict[str, Any],
    threshold: float = 2.5,
    channel: str = "S-1",
    spacecraft: str = "SMAP",
    method_name: str = "zscore",
    score_fn=calculate_zscore,
) -> Optional[Dict[str, Any]]:
    """Evaluates a telemetry tick for anomalies.

    Input tick: {"t": int, "value": float, "history": list[float]}
    Output: Contract A dict (DETECTION_EVENT) if score >= threshold, else None.

    To swap models (e.g., to LSTM), simply pass a custom score_fn(value, history) -> float.
    """
    t = tick["t"]
    val = tick["value"]
    history = tick.get("history", [])

    score = score_fn(val, history)

    if score >= threshold:
        return make_detection_event(
            channel=channel,
            spacecraft=spacecraft,
            timestep=t,
            flagged_range=[t - 5, t + 400],
            score=round(score, 2),
            method=method_name,
        )

    return None

# ===========================================================================
# Simple Self-Test
# ===========================================================================
if __name__ == "__main__":
    # 1. Test normal tick -> returns None
    normal_tick = {"t": 100, "value": 0.1, "history": [0.1, 0.09, 0.11, 0.1, 0.08]}
    assert detect(normal_tick) is None, "Normal tick should return None"

    # 2. Test anomaly spike tick -> returns Contract A dict
    anomaly_tick = {"t": 105, "value": 5.0, "history": [0.1, 0.09, 0.11, 0.1, 0.08]}
    event = detect(anomaly_tick, threshold=2.5)

    assert event is not None, "Anomaly tick should return detection event"
    print("Contract A Event Output:")
    print(event)
    print("\n✓ detector.py self-test passed!")
