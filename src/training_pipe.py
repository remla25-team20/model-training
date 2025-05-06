import pandas as pd
import joblib
import sys

from pathlib import Path
from lib_ml.preprocessing import _load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB


def train(dataset: pd.DataFrame, target, preprocessed_data):

    X = joblib.load(preprocessed_data).toarray()
    y = dataset.iloc[:, 1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, target)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main(target, train_data, preprocessed_data):
    dataset = _load_data(train_data)
    train(dataset=dataset, target=target, preprocessed_data=preprocessed_data)


if __name__ == "__main__":
    # handle keyword arguments
    cl_args = [str(arg) for arg in sys.argv[1:]]
    flags = [arg for arg in cl_args if '=' not in arg]
    kwargs = {
        kwarg[0].lower(): kwarg[1] for kwarg in [arg.split('=') for arg in cl_args if '=' in arg]
              }

    # select target file location, train data location and preprocessed data location
    target = kwargs.get('--target', 'models/C2_Classifier_Sentiment_Model.joblib')
    train_data = kwargs.get('--train-data', 'data/a1_RestaurantReviews_HistoricDump.tsv')
    preprocessed_data = kwargs.get('--preprocessed-data', 'output/preprocessed_data.joblib')

    main(target=target, train_data=train_data, preprocessed_data=preprocessed_data)
