#################################################################
from rich.progress import Progress
import time
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")
#################################################################


class ModelSelection:

    def __init__(self, models):
        self.models = models
        self.result = []
        self.metrics = []

    def fit_search(self, X_train, y_train):

        # Setting up the parameters.
        param_grid = {
            "RandomForest": [
                {
                    "n_estimators": [90, 150, 200],
                    "max_features": [2, 4, 6, 8, 10],
                    "min_samples_split": [2, 4, 6, 8],
                    "min_samples_leaf": [2, 4, 6, 8]
                },
                {
                    "bootstrap": [False],
                    "n_estimators": [90, 150, 250],
                    "max_features": [3, 6, 8],
                    "min_samples_split": [2, 4, 6],
                    "min_samples_leaf": [2, 4, 6]
                },
            ],
            "DecisionTree": [
                {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 4, 6, 8],
                    "min_samples_leaf": [2, 4, 6, 8]
                },
                {
                    "criterion": ["entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 4, 6, 8],
                    "min_samples_leaf": [2, 4, 6, 8]
                }
            ],
            "AdaBoost": {
                "n_estimators": [90, 150, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1]
            },
            "GradientBoost": {
                "n_estimators": [100, 150, 300],
                "learning_rate": [0.01, 0.1, 0.5, 0.75, 1],
                "max_depth": [None, 3, 6, 8],
                "min_samples_split": [3, 6, 8],
                "min_samples_leaf": [3, 6, 8],
                "subsample": [0.25, 0.5, 0.75, 1]
            },
            "ExtraTrees": [
                {
                    "n_estimators": [90, 150, 200],
                    "max_features": [2, 4, 6],
                    "min_samples_split": [2, 4, 6],
                    "min_samples_leaf": [2, 4, 6]
                },
                {
                    "bootstrap": [False],
                    "n_estimators": [90, 150],
                    "max_features": [3, 8],
                    "min_samples_split": [2, 8],
                    "min_samples_leaf": [2, 6]
                },
            ],
            "XGBoost": {
                "n_estimators": [100, 250, 500, 1000],
                "learning_rate": [0.5, 0.1, 0.05, 0.01, 0.001],
                "max_depth": [None, 3, 8],
            },
            "SVM": {
                "kernel": ["rbf"],
                "C": [0.1, 1],
                "gamma": [1, 0.1,]
            },
            "LogisticRegression": {

            },
            "NaiveBayes": {

            },
            "KNearestNeighbor": {
                "n_neighbors": [3, 5, 8, 9, 12],
                "p": [2, 4, 6, 8]
            }
        }

        # Running the gridsearch.
        with Progress() as progress:
            best_models = {}

            for names, model in self.models.items():
                task = progress.add_task(
                    f"[green]Running {names}:", total=100
                )

                grid_search = GridSearchCV(
                    model, param_grid[names], cv=5,
                    scoring="accuracy", n_jobs=-1
                )

                # Fitting the model.
                search_result = grid_search.fit(X_train, y_train)
                while not progress.finished:
                    progress.update(task, advance=1)
                    time.sleep(0.2)

                # Storing the grid search result for each model.
                best_models[names] = search_result

        return best_models

    def best_search_(self, best_models):

        for model_name, search_result in best_models.items():
            best_params = search_result.best_params_
            best_score = f"{search_result.best_score_:.3f}"

            self.result.append(
                {
                    "Model": model_name,
                    "Best Parameters": best_params,
                    "Best Score": best_score
                }
            )

        return pd.DataFrame(self.result)

    def search_metrics(self, X_train, y_train, X_test, y_test, best_models):

        for model_name, search_result in best_models.items():
            test_pred = search_result.predict(X_test)
            train_pred = search_result.predict(X_train)

            # Metrics...
            train_accuracy = f"{accuracy_score(y_train, train_pred):.2f}"
            test_accuracy = f"{accuracy_score(y_test, test_pred):.2f}"
            precision = f"{precision_score(y_test, test_pred):.2f}"
            recall = f"{recall_score(y_test, test_pred):.2f}"

            self.metrics.append(
                {
                    "Model": model_name,
                    "Train score": train_accuracy,
                    "Test score": test_accuracy,
                    "Precision": precision,
                    "Recall": recall
                }
            )

        return pd.DataFrame(self.metrics)



