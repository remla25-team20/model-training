import joblib
import dvc.api

from pathlib import Path
from sklearn.naive_bayes import GaussianNB

PARAMS = dvc.api.params_show("params.yaml")

def train(dataset_dir: Path, target_dir):
    X_train = joblib.load(dataset_dir / "X_train.joblib")
    y_train = joblib.load(dataset_dir / "y_train.joblib")

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, target_dir / "model.joblib")

if __name__ == "__main__":
    train(dataset_dir=Path(PARAMS['interim_data_directory']), target_dir=Path(PARAMS['models_directory']))