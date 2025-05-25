import joblib
import os
import dvc.api

from pathlib import Path
from sklearn.model_selection import train_test_split

from lib_ml import preprocessing

PARAMS = dvc.api.params_show("params.yaml")

def preprocess(dataset: Path, interim_data_dir: Path, models_dir: Path, test_size, random_state):

    preprocessor, preprocessed_data = preprocessing.preprocess(dataset)
    X = preprocessed_data.toarray()
    y = preprocessing._load_data(dataset).iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
        )

    if not os.path.exists(interim_data_dir):
        os.makedirs(interim_data_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump(X_train, interim_data_dir / "X_train.joblib")
    joblib.dump(X_test, interim_data_dir / "X_test.joblib")
    joblib.dump(y_train, interim_data_dir / "y_train.joblib")
    joblib.dump(y_test, interim_data_dir / "y_test.joblib")
    joblib.dump(preprocessor, models_dir / "Sentiment_Analysis_Preprocessor.joblib")

if __name__ == "__main__":
    preprocess(
        dataset=Path(PARAMS['data_directory']) / PARAMS['file_path_gdrive'],
        interim_data_dir=Path(PARAMS['interim_data_directory']),
        models_dir=Path(PARAMS['models_directory']),
        test_size=PARAMS['test_size'],
        random_state=PARAMS['random_state']
        )
