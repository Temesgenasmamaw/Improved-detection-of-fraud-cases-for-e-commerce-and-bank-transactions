import os
import mlflow
import mlflow.sklearn
import mlflow.keras
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np
import seaborn as sns
import joblib  # alternatively use import pickle
import time
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Input
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


class ModelBuilding:
    def split_fraud_data(df):
        X = df.drop(
            [
                "Unnamed: 0",
                "user_id",
                "signup_time",
                "purchase_time",
                "device_id",
                "ip_address",
                "class",
            ],
            axis=1,
        )
        y = df["class"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def split_creditcard_data(df):
        X = df.drop(["Class"], axis=1)
        y = df["Class"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

   
    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(
            Input(shape=(input_shape, 1))
        )  # Expecting input shape (timesteps, features)
        model.add(LSTM(50))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model


    def build_cnn_model(input_shape):
        """
        Builds and compiles a CNN model.
        """
        model = Sequential()
        model.add(
            Input(shape=(input_shape, 1))
        )  # Expecting input shape (timesteps, features)
        model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train_model(X_train, X_test, y_train, y_test):
            # Reshape data for LSTM and CNN (expecting 3D input: samples, timesteps, features)
        X_train_reshaped = np.expand_dims(X_train, axis=-1)
        X_test_reshaped = np.expand_dims(X_test, axis=-1)
        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "LSTM": ModelBuilding.build_lstm_model(X_train.shape[1]),
            "CNN": ModelBuilding.build_cnn_model(X_train.shape[1])
        }

        # Train, evaluate, and select the best model
        best_model = None
        best_accuracy = 0

        # Store accuracy scores and classification reports for all models
        results = {}

        for name, model in classifiers.items():
            if name in ["LSTM", "CNN"]:
                # Train neural network models with Keras
                model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
                y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32")
            else:
                # Train traditional ML models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            results[name] = {
                "accuracy": accuracy,
                "classification_report": classification_rep,
            }

            print(f"{name} Accuracy: {accuracy}")
            print(f"{name} Classification Report:\n{classification_rep}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

            # Print the best model after the loop
        print(f"\nBest Model: {type(best_model).__name__} with Accuracy: {best_accuracy}")

        return best_model, results

    def find_best_model_using_gridsearchcv(X, y):
        algos = {
            "logistic_regression": {
                "model": LogisticRegression(solver="liblinear", multi_class="auto"),
                "params": {"C": [1, 5, 10]},
            },
            "svm": {
                "model": SVC(),
                "params": {
                    "C": [1, 5, 10, 20, 40, 50, 60, 100],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            },
            "random_forest": {
                "model": RandomForestClassifier(),
                "params": {"n_estimators": [1, 5, 10, 40]},
            },
            "decision_tree": {
                "model": DecisionTreeClassifier(),
                "params": {
                    "criterion": ["gini", "log_loss", "entropy"],
                    "splitter": ["best", "random"],
                },
            },
            "naive_bayes_gaussian": {"model": GaussianNB(), "params": {}},
            "knn": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            },
            "adaboost": {
                "model": AdaBoostClassifier(estimator=DecisionTreeClassifier()),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },
            },
        }
        scores = []
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        for algo_name, config in algos.items():
            gs = GridSearchCV(
                config["model"], config["params"], cv=cv, return_train_score=False
            )
            gs.fit(X, y)
            scores.append(
                {
                    "model": algo_name,
                    "best_score": gs.best_score_,
                    "best_params": gs.best_params_,
                }
            )

        return pd.DataFrame(scores, columns=["model", "best_score", "best_params"])

    # find_best_model_using_gridsearchcv(X_train_scaled, Y_train)
