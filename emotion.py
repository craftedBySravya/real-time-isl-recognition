# emotion.py
import math
from collections import Counter

def _dist(lm, i, j):
    # Simple helper to measure distance between two face landmarks
    dx = lm[i].x - lm[j].x
    dy = lm[i].y - lm[j].y
    return math.sqrt(dx * dx + dy * dy)


def classify_face_emotion(face_lms):
    """
    Basic facial-emotion check using mouth opening.
    This is not a full model — just a rough idea:
        - If the mouth opens more, we treat it as HAPPY.
        - Otherwise we treat it as NEUTRAL.
    """
    if face_lms is None:
        return None

    lm = face_lms.landmark

    try:
        # width between mouth corners
        mouth_w = _dist(lm, 61, 291)
        # vertical opening between upper/lower lip
        mouth_h = _dist(lm, 13, 14)
    except Exception:
        return None

    # avoid division issues
    if mouth_w < 1e-6:
        return None

    open_ratio = mouth_h / mouth_w

    # if the mouth is relatively open → count as smile
    if open_ratio > 0.22:
        return "HAPPY"

    # otherwise treat as neutral
    return "NEUTRAL"


def estimate_emotion(emotion_list):
    """
    Given multiple frame-wise emotions for one sentence,
    take the most common one. If nothing was detected,
    fall back to NEUTRAL.
    """
    if not emotion_list:
        return "NEUTRAL"

    filtered = [e for e in emotion_list if e is not None]
    if not filtered:
        return "NEUTRAL"

    most, _ = Counter(filtered).most_common(1)[0]
    return most

