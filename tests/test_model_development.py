import pytest
import os
import dvc.api
import gdown

from pathlib import Path
from data.preprocess import preprocess
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
