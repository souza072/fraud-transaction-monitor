import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fraud_monitor.pipeline import import_csv, score_transactions, train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud Transaction Monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import"); importer.add_argument("csv", type=Path); importer.add_argument("db", type=Path)
    trainer = commands.add_parser("train"); trainer.add_argument("db", type=Path); trainer.add_argument("model", type=Path); trainer.add_argument("--target-recall", type=float, default=0.80)
    scorer = commands.add_parser("score"); scorer.add_argument("db", type=Path); scorer.add_argument("model", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "import":
        result = import_csv(arguments.csv, arguments.db)
    elif arguments.command == "train":
        result = train_models(arguments.db, arguments.model, arguments.target_recall)
    else:
        result = score_transactions(arguments.db, arguments.model)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
