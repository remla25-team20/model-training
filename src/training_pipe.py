import pandas as pd
import joblib

from pathlib import Path
from lib_ml.preprocessing import _load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

def train(dataset: pd.DataFrame):

    X = joblib.load('output/preprocessed_data.joblib').toarray()
    y = dataset.iloc[:, 1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'models/C2_Classifier_Sentiment_Model.joblib')

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def main():
    dataset = _load_data("/home/nathan/Robotics/CS4295/model-training/a1_RestaurantReviews_HistoricDump.tsv")
    train(dataset)

if __name__ == "__main__":
    main()