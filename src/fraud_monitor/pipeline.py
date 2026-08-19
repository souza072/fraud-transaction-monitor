import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import risk_level
from .database import create_alert_table, import_transactions
from .metrics import average_precision, choose_threshold, classification_metrics
from .models import feature_evidence, fit_gaussian_nb, fit_logistic, predict_probability


def chronological_split(frame: pd.DataFrame):
    ordered = frame.sort_values("Time").reset_index(drop=True)
    train_end, validation_end = int(len(ordered) * 0.60), int(len(ordered) * 0.80)
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def train_models(db_path: Path, model_path: Path, target_recall: float = 0.80) -> dict:
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query("SELECT * FROM transactions ORDER BY Time", connection)
    train, validation, test = chronological_split(frame)
    candidates, reports = [fit_gaussian_nb(train), fit_logistic(train)], []
    validation_y = validation["Class"].to_numpy(dtype=int)
    for candidate in candidates:
        probability = predict_probability(candidate, validation)
        reports.append({"name": candidate["type"], "threshold": choose_threshold(validation_y, probability, target_recall), "validation_pr_auc": average_precision(validation_y, probability)})
    best = max(range(len(candidates)), key=lambda i: (reports[i]["validation_pr_auc"], reports[i]["threshold"]["precision"]))
    selected, report = candidates[best], reports[best]
    threshold = report["threshold"]["threshold"]
    test_probability, test_y = predict_probability(selected, test), test["Class"].to_numpy(dtype=int)
    evaluation = classification_metrics(test_y, test_probability, threshold)
    evaluation.update({"pr_auc": average_precision(test_y, test_probability), "rows": len(test), "frauds": int(test_y.sum())})
    artifact = {"schema_version": 2, "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "selected_model": selected["type"], "threshold": threshold, "model": selected, "candidates": reports, "evaluation": evaluation, "split": {"train": len(train), "validation": len(validation), "test": len(test)}}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def score_transactions(db_path: Path, model_path: Path) -> dict:
    artifact = json.loads(model_path.read_text(encoding="utf-8")); model, threshold = artifact["model"], artifact["threshold"]
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query("SELECT rowid AS transaction_id, * FROM transactions", connection)
        probability = predict_probability(model, frame); flagged = np.flatnonzero(probability >= threshold)
        create_alert_table(connection); records = []
        for index in flagged:
            row = frame.iloc[int(index)]
            explanation = {"basis": "Class=1 no dataset" if int(row["Class"]) == 1 else "Alerta do modelo; Class=0 no dataset", "top_features": feature_evidence(model, row)}
            records.append((int(row["transaction_id"]), float(probability[index]), 1, risk_level(float(probability[index]), threshold), int(row["Class"]), "pending", json.dumps(explanation, ensure_ascii=False), artifact["model_version"]))
        connection.executemany("INSERT INTO alerts (transaction_id,fraud_probability,predicted_fraud,risk_level,actual_class,review_status,explanation,model_version) VALUES (?,?,?,?,?,?,?,?)", records)
    return {"alerts_generated": len(records), "threshold": threshold, "model": artifact["selected_model"]}


def import_csv(csv_path: Path, db_path: Path) -> dict:
    return import_transactions(csv_path, db_path)
