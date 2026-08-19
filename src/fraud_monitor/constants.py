FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"

def risk_level(probability: float, alert_threshold: float) -> str:
    """Classify severity relative to the model's calibrated alert threshold."""
    ratio = probability / max(alert_threshold, 1e-12)
    if probability >= 0.85 or ratio >= 8:
        return "critical"
    if probability >= 0.60 or ratio >= 4:
        return "high"
    if ratio >= 2:
        return "moderate"
    return "watch"
