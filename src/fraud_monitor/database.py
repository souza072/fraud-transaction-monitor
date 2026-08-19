import sqlite3
from pathlib import Path

import pandas as pd


def import_transactions(csv_path: Path, db_path: Path) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS transactions")
        connection.execute("DROP TABLE IF EXISTS alerts")
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=25_000):
            chunk.to_sql("transactions", connection, if_exists="replace" if first else "append", index=False)
            first = False
        connection.execute("CREATE INDEX idx_transactions_class ON transactions(Class)")
        count, frauds = connection.execute("SELECT COUNT(*), SUM(Class) FROM transactions").fetchone()
    return {"imported": count, "frauds": frauds, "database": str(db_path)}


def create_alert_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS alerts")
    connection.execute("""CREATE TABLE alerts (
        transaction_id INTEGER PRIMARY KEY, fraud_probability REAL NOT NULL,
        predicted_fraud INTEGER NOT NULL, risk_level TEXT NOT NULL,
        actual_class INTEGER, review_status TEXT NOT NULL DEFAULT 'pending',
        explanation TEXT NOT NULL, model_version TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    connection.execute("CREATE INDEX idx_alerts_risk ON alerts(risk_level, fraud_probability DESC)")
