"""
Download the dataset from Google Drive using gdown
"""

import gdown
import dvc.api

PARAMS = dvc.api.params_show("params.yaml")

if __name__ == "__main__":
    gdown.download_folder(id=PARAMS["data_directory_id"],
                          output=PARAMS["data_directory"], quiet=False)
