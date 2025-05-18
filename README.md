# model-training

## Usage

```python
import requests
from joblib import load
import os

os.makedirs("models", exist_ok=True)

model_url = "https://github.com/remla25-team20/model-training/releases/download/v0.1.4/Sentiment_Analysis_Model.joblib"
preprocessor_url = "https://github.com/remla25-team20/model-training/releases/download/v0.1.4/Sentiment_Analysis_Preprocessor.joblib"

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)

download_file(model_url, "models/Sentiment_Analysis_Model.joblib")
download_file(preprocessor_url, "models/Sentiment_Analysis_Preprocessor.joblib")

model = load("models/Sentiment_Analysis_Model.joblib")
preprocessor = load("models/Sentiment_Analysis_Preprocessor.joblib")
```
