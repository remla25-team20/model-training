from pathlib import Path

import joblib
import dvc.api
import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

PARAMS = dvc.api.params_show("params.yaml")

def predict(model_dir: Path, interim_data_dir: Path, metrics_directory: Path):
    classifier = joblib.load(model_dir / "Sentiment_Analysis_Model.joblib")
    X_test = joblib.load(interim_data_dir / "X_test.joblib")
    y_test = joblib.load(interim_data_dir / "y_test.joblib")

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    with open(metrics_directory / "accuracy.json", "w", encoding="utf-8") as out:
        json.dump({"accuracy": accuracy}, out)

    f1_acc = f1_score(y_test, y_pred)
    print(f"F1 accuracy: {f1_acc * 100:.2f}%")
    with open(metrics_directory / "f1_accuracy.json", "w", encoding="utf-8") as out:
        json.dump({"f1 accuracy": f1_acc}, out)

    tpr = precision_score(y_test, y_pred)
    print(f"Precision: {tpr * 100:.2f}%")
    with open(metrics_directory / "precision.json", "w", encoding="utf-8") as out:
        json.dump({"precision": tpr}, out)

    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall * 100:.2f}%")
    with open(metrics_directory / "recall.json", "w", encoding="utf-8") as out:
        json.dump({"recall": recall}, out)


if __name__ == "__main__":
    predict(
        model_dir=Path(PARAMS['models_directory']),

        interim_data_dir=Path(PARAMS['interim_data_directory']),
        metrics_directory=Path(PARAMS['metrics_directory'])
    )
