import joblib
import sys

from pathlib import Path
from lib_ml import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB


def train(dataset: Path, target):

    preprocessor, preprocessed_data = preprocessing.preprocess(dataset)
    X = preprocessed_data.toarray()
    y = preprocessing._load_data(dataset).iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, target + "_Model.joblib")
    joblib.dump(preprocessor, target + "_Preprocessor.joblib")

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    cl_args = [str(arg) for arg in sys.argv[1:]]
    flags = [arg for arg in cl_args if '=' not in arg]
    kwargs = {
        kwarg[0].lower(): kwarg[1] for kwarg in [arg.split('=') for arg in cl_args if '=' in arg]
              }

    target = kwargs.get('--target', 'models/Sentiment_Analysis')
    dataset = kwargs.get('--dataset', 'data/a1_RestaurantReviews_HistoricDump.tsv')

    train(dataset, target)

