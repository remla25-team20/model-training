import pytest
import os
import csv
import dvc.api
import gdown
import joblib

from pathlib import Path
from data.preprocess import preprocess
import lib_ml.preprocessing
from modeling.train import train
from modeling.evaluation import predict

PARAMS = dvc.api.params_show("params.yaml")

@pytest.fixture
def init_train_test_data():
    if not os.path.exists(PARAMS["data_directory"]):
        os.makedirs(PARAMS["data_directory"])

    gdown.download_folder(id=PARAMS["data_directory_gdrive_id"],
                          output=PARAMS["data_directory"], quiet=False)

    preprocess(
        dataset=Path(PARAMS['data_directory']) / PARAMS['file_path_gdrive'],
        interim_data_dir=Path(PARAMS['interim_data_directory']),
        models_dir=Path(PARAMS['models_directory']),
        test_size=PARAMS['test_size'],
        random_state=PARAMS['random_state']
        )

class TestModelDevelopment:
    def test_model_seed_robustness(self, init_train_test_data):
        # Train and evaluate first model
        train(dataset_dir=Path(PARAMS['interim_data_directory']), target_dir=Path(PARAMS['models_directory']))
        tpr_model1, recall_model1, accuracy_model1, f1_acc_model1 = predict(
            model_dir=Path(PARAMS['models_directory']),
            interim_data_dir=Path(PARAMS['interim_data_directory']),
            metrics_directory=Path(PARAMS['metrics_directory'])
            )
        
        # Train and evaluate second model
        train(dataset_dir=Path(PARAMS['interim_data_directory']), target_dir=Path(PARAMS['models_directory']))
        tpr_model2, recall_model2, accuracy_model2, f1_acc_model2 = predict(
            model_dir=Path(PARAMS['models_directory']),
            interim_data_dir=Path(PARAMS['interim_data_directory']),
            metrics_directory=Path(PARAMS['metrics_directory'])
            )
        
        # Compare evaluation metrics that they are close on same data
        assert abs(tpr_model1 - tpr_model2) <= 0.03
        assert abs(recall_model1 - recall_model2) <= 0.03
        assert abs(accuracy_model1 - accuracy_model2) <= 0.03
        assert abs(f1_acc_model1 - f1_acc_model2) <= 0.03

    def test_preprocessing_encoding_adequacy(self):
        """
        The preprocessing should successfully encode a sufficiently large segment of the data
        :return: True iff at least 75% of non-stopwords are encoded, False otherwise
        """
        count_word = 0
        count_encoding = 0

        count_vectorizer = joblib.load(PARAMS["models_directory"] + "/Sentiment_Analysis_Preprocessor.joblib")

        with open(PARAMS["data_directory"], "r") as data:
            f = csv.reader(data, delimiter="\t")
            next(f)     # skip header
            for review, _ in f:
                raw_review = lib_ml.preprocessing._text_process(review)
                count_word += 1 + raw_review.count(" ")
                vectorized_review = count_vectorizer.transform(raw_review)
                count_encoding += sum(vectorized_review)
        print(count_word)
        print(count_encoding)
        assert count_encoding / count_word > 0.75
