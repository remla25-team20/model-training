import joblib
import dvc.api
import json

from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

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
    with open(metrics_directory / "accuracy.json", "w") as out:
        json.dump({"accuracy": accuracy}, out)

if __name__ == "__main__":
    predict(
        model_dir=Path(PARAMS['models_directory']),
        interim_data_dir=Path(PARAMS['interim_data_directory']),
        metrics_directory=Path(PARAMS['metrics_directory'])
        )
