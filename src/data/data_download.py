"""
Download the dataset from Google Drive using gdown
"""

import gdown
import os
import dvc.api

PARAMS = dvc.api.params_show("params.yaml")

if __name__ == "__main__":
    if not os.path.exists(PARAMS["data_directory"]):
        os.makedirs(PARAMS["data_directory"])

    gdown.download_folder(id=PARAMS["data_directory_gdrive_id"],
                          output=PARAMS["data_directory"], quiet=False)
