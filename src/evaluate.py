from joblib import load
import json
from pathlib import Path

from dvclive import Live

from sklearn.metrics import accuracy_score

from train import load_data


def main(repo_path):
    test_csv_path = repo_path / "data/prepared/test.csv"
    test_data, labels = load_data(test_csv_path)
    model = load(repo_path / "model/model.joblib")
    predictions = model.predict(test_data)
    accuracy = accuracy_score(labels, predictions)
    metrics = {"accuracy": accuracy}
    with Live(save_dvc_exp=True) as live:
        live.log_metric("accuracy", metrics['accuracy'])
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
