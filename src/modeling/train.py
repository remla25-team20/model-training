import joblib
import os
import dvc.api

from pathlib import Path
from sklearn.naive_bayes import GaussianNB

PARAMS = dvc.api.params_show("params.yaml")

def train(dataset_dir: Path, target_dir):
    X_train = joblib.load(dataset_dir / "X_train.joblib")
    y_train = joblib.load(dataset_dir / "y_train.joblib")

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    joblib.dump(classifier, target_dir / "Sentiment_Analysis_Model.joblib")

if __name__ == "__main__":
    train(dataset_dir=Path(PARAMS['interim_data_directory']), target_dir=Path(PARAMS['models_directory']))