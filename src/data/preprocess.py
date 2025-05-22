import joblib
import dvc.api

from pathlib import Path
from sklearn.model_selection import train_test_split

from lib_ml import preprocessing

PARAMS = dvc.api.params_show("params.yaml")

def preprocess(dataset: Path, target_dir):

    preprocessor, preprocessed_data = preprocessing.preprocess(dataset)
    X = preprocessed_data.toarray()
    y = preprocessing._load_data(dataset).iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PARAMS['test_size'],
        random_state=PARAMS['random_state']
        )

    joblib.dump(X_train, target_dir / "X_train.joblib")
    joblib.dump(X_test, target_dir / "X_test.joblib")
    joblib.dump(y_train, target_dir / "y_train.joblib")
    joblib.dump(y_test, target_dir / "y_test.joblib")

if __name__ == "__main__":
    preprocess(
        dataset=Path(PARAMS['data_directory']) / "restaurant-reviews/training-reviews.tsv",
        target_dir=Path(PARAMS['interim_data_directory'])
        )
