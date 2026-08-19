import numpy as np
import pandas as pd

from .constants import FEATURES


def fit_gaussian_nb(frame: pd.DataFrame) -> dict:
    x, y = frame[FEATURES].to_numpy(dtype=float), frame["Class"].to_numpy(dtype=int)
    classes = {}
    for cls in (0, 1):
        subset = x[y == cls]
        classes[str(cls)] = {"prior": float(len(subset) / len(x)), "mean": subset.mean(axis=0).tolist(), "var": (subset.var(axis=0) + 1e-6).tolist()}
    return {"type": "gaussian_nb", "features": FEATURES, "classes": classes}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


def fit_logistic(frame: pd.DataFrame, epochs: int = 240, seed: int = 42) -> dict:
    x, y = frame[FEATURES].to_numpy(dtype=float), frame["Class"].to_numpy(dtype=int)
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-8
    x = (x - mean) / scale
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng, weights, bias = np.random.default_rng(seed), np.zeros(x.shape[1]), 0.0
    for epoch in range(epochs):
        chosen_negative = rng.choice(negative, size=len(positive), replace=False)
        indices = np.concatenate([positive, chosen_negative]); rng.shuffle(indices)
        batch_x, batch_y = x[indices], y[indices]
        prediction = _sigmoid(batch_x @ weights + bias)
        error = prediction - batch_y
        rate = 0.08 / (1.0 + epoch * 0.015)
        weights -= rate * ((batch_x.T @ error) / len(indices) + 0.001 * weights)
        bias -= rate * float(error.mean())
    prior = float(y.mean())
    return {"type": "logistic_regression", "features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(), "bias": bias, "prior_adjustment": float(np.log(prior / (1 - prior)))}


def predict_probability(model: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[model["features"]].to_numpy(dtype=float)
    if model["type"] == "logistic_regression":
        standardized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
        return _sigmoid(standardized @ np.asarray(model["weights"]) + model["bias"] + model["prior_adjustment"])
    scores = []
    for cls in (0, 1):
        item = model["classes"][str(cls)]
        mean, var = np.asarray(item["mean"]), np.asarray(item["var"])
        scores.append(np.log(item["prior"]) - 0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var, axis=1))
    scores = np.vstack(scores).T; scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores[:, 1] / exp_scores.sum(axis=1)


def feature_evidence(model: dict, row: pd.Series, limit: int = 3) -> list[dict]:
    values = row[model["features"]].to_numpy(dtype=float)
    if model["type"] == "logistic_regression":
        contributions = ((values - np.asarray(model["mean"])) / np.asarray(model["scale"])) * np.asarray(model["weights"])
    else:
        normal, fraud = model["classes"]["0"], model["classes"]["1"]
        m0, v0 = np.asarray(normal["mean"]), np.asarray(normal["var"])
        m1, v1 = np.asarray(fraud["mean"]), np.asarray(fraud["var"])
        contributions = -0.5 * (np.log(v1 / v0) + ((values - m1) ** 2) / v1 - ((values - m0) ** 2) / v0)
    indices = np.argsort(-contributions)[:limit]
    return [{"feature": model["features"][int(i)], "contribution": round(float(contributions[i]), 3)} for i in indices]
