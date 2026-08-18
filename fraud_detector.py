import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def import_csv(csv_path: Path, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS transactions")
        con.execute("DROP TABLE IF EXISTS alerts")
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=25000):
            chunk.to_sql("transactions", con, if_exists="replace" if first else "append", index=False)
            first = False
        con.execute("CREATE INDEX idx_transactions_class ON transactions(Class)")
        con.execute("""CREATE TABLE alerts (
            transaction_id INTEGER PRIMARY KEY,
            fraud_probability REAL NOT NULL,
            predicted_fraud INTEGER NOT NULL,
            actual_class INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        count, frauds = con.execute("SELECT COUNT(*), SUM(Class) FROM transactions").fetchone()
    print(json.dumps({"imported": count, "frauds": frauds, "database": str(db_path)}, ensure_ascii=False))


def split(df: pd.DataFrame):
    # Chronological split prevents future transactions leaking into training.
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    return df.iloc[:cut], df.iloc[cut:]


def fit_gaussian_nb(train: pd.DataFrame) -> dict:
    x = train[FEATURES].to_numpy(dtype=float)
    y = train["Class"].to_numpy(dtype=int)
    model = {"features": FEATURES, "classes": {}}
    for cls in (0, 1):
        xc = x[y == cls]
        model["classes"][str(cls)] = {
            "prior": float(len(xc) / len(x)),
            "mean": xc.mean(axis=0).tolist(),
            "var": (xc.var(axis=0) + 1e-6).tolist(),
        }
    return model


def probabilities(model: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[model["features"]].to_numpy(dtype=float)
    scores = []
    for cls in (0, 1):
        item = model["classes"][str(cls)]
        mean, var = np.array(item["mean"]), np.array(item["var"])
        logp = np.log(item["prior"]) - 0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var, axis=1)
        scores.append(logp)
    scores = np.vstack(scores).T
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    return probs[:, 1] / probs.sum(axis=1)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def train(db_path: Path, model_path: Path) -> None:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY Time", con)
    train_df, test_df = split(df)
    model = fit_gaussian_nb(train_df)
    p = probabilities(model, test_df)
    y = test_df["Class"].to_numpy(dtype=int)
    # Select the lowest threshold reaching at least 80% recall, then maximize precision.
    candidates = [metrics(y, p, float(t)) for t in np.unique(np.quantile(p, np.linspace(0, 1, 1001)))]
    eligible = [m for m in candidates if m["recall"] >= 0.80]
    chosen = max(eligible or candidates, key=lambda m: (m["precision"], m["recall"]))
    model["threshold"] = chosen["threshold"]
    model["evaluation"] = chosen | {"test_rows": len(test_df), "test_frauds": int(y.sum())}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps(model["evaluation"], ensure_ascii=False))


def score(db_path: Path, model_path: Path) -> None:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("SELECT rowid AS transaction_id, * FROM transactions", con)
        p = probabilities(model, df)
        flagged = p >= model["threshold"]
        alerts = pd.DataFrame({
            "transaction_id": df.loc[flagged, "transaction_id"].astype(int),
            "fraud_probability": p[flagged],
            "predicted_fraud": 1,
            "actual_class": df.loc[flagged, "Class"].astype(int),
        })
        con.execute("DELETE FROM alerts")
        alerts.to_sql("alerts", con, if_exists="append", index=False)
    print(json.dumps({"alerts_generated": len(alerts), "threshold": model["threshold"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Detector de fraude com alertas em SQLite")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import"); imp.add_argument("csv", type=Path); imp.add_argument("db", type=Path)
    trn = sub.add_parser("train"); trn.add_argument("db", type=Path); trn.add_argument("model", type=Path)
    scr = sub.add_parser("score"); scr.add_argument("db", type=Path); scr.add_argument("model", type=Path)
    args = parser.parse_args()
    {"import": import_csv, "train": train, "score": score}[args.command](*([args.csv, args.db] if args.command == "import" else [args.db, args.model]))


if __name__ == "__main__":
    main()

