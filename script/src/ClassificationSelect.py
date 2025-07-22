# Importing dependencies
import time
from rich.progress import Progress
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Model selection
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
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


########################################################################
# Loading the data
########################################################################
df = pd.read_csv("../data/processed/train.csv")
df.head()

col = [
    "income", "overage",
    "leftover", "house", "handset_price",
    "over_15mins_calls_per_month", "average_call_duration"
]


########################################################################
# Splitting the data
########################################################################
scaler = StandardScaler()
df[col] = scaler.fit_transform(df[col])

X = df.drop("leave", axis=1)
y = df["leave"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


########################################################################
# Model list
########################################################################
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            random_state=42,
            max_depth=1,
            min_samples_leaf=3,
            min_samples_split=3
        )
    ),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
    "XGBoost": XGBClassifier(),
    "SVM": SVC(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "NaiveBayes": GaussianNB(),
    "KNearestNeighbor": KNeighborsClassifier()
}


########################################################################
# Model Selection
########################################################################

class ModelSelection():

    def __init__(self, models, X1, y1, X2, y2):
        self.models = models
        self.X1 = X1
        self.y1 = y1
        self.X2 = X2
        self.y2 = y2

    def run_grid_search(self):

        result = []
        metrics = []

        # Setting up the parameters.
        param_grid = {
            "RandomForest": [
                {
                    "n_estimators": [90, 150, 200],
                    #     "max_features": [2, 4, 6, 8, 10],
                    #     "min_samples_split": [2, 4, 6, 8],
                    #     "min_samples_leaf": [2, 4, 6, 8]
                    # },
                    # {
                    #     "bootstrap": [False],
                    #     "n_estimators": [90, 150, 250],
                    #     "max_features": [3, 6, 8],
                    #     "min_samples_split": [2, 4, 6],
                    #     "min_samples_leaf": [2, 4, 6]
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
                # "n_estimators": [100, 150, 300],
                # "learning_rate": [0.01, 0.1, 0.5, 0.75, 1],
                # "max_depth": [None, 3, 6, 8],
                # "min_samples_split": [3, 6, 8],
                # "min_samples_leaf": [3, 6, 8],
                "subsample": [0.25, 0.5, 0.75, 1]
            },
            "ExtraTrees": [
                {
                    # "n_estimators": [90, 150, 200],
                    # "max_features": [2, 4, 6],
                    "min_samples_split": [2, 4, 6],
                    "min_samples_leaf": [2, 4, 6]
                },
                {
                    "bootstrap": [False],
                    # "n_estimators": [90, 150],
                    # "max_features": [3, 8],
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
                # "kernel": ["rbf"],
                "C": [0.1, 1],
                "gamma": [1, 0.1,]
            },
            "LogisticRegression": {

            },
            "NaiveBayes": {

            },
            "KNearestNeighbor": {
                "n_neighbors": [2, 3, 5, 6, 8],
                "p": [2, 4, 6, 8]
            }
        }

        # Running the gridsearch.
        with Progress() as progress:

            for names, model in models.items():
                task = progress.add_task(
                    f"[green]Running {names}:", total=100
                )

                grid_search = GridSearchCV(
                    model, param_grid[names], cv=5,
                    scoring="accuracy", n_jobs=-1
                )

                # Fitting the model.
                search_result = grid_search.fit(self.X1, self.y1)
                while not progress.finished:
                    progress.update(task, advance=1)
                    time.sleep(0.2)

                # GridSearch attributes.
                best_params = search_result.best_params_
                best_score = f"{search_result.best_score_:.3f}"

                result.append(
                    {
                        "Model": names,
                        "Best Parameters": best_params,
                        "Best Score": best_score
                    }
                )

                # Predicting with the best estimator.
                train_pred = grid_search.predict(self.X1)
                test_pred = grid_search.predict(self.X2)

                # Metrics...
                train_accuracy = f"{accuracy_score(self.y1, train_pred):.2f}"
                test_accuracy = f"{accuracy_score(self.y2, test_pred):.2f}"
                precision = f"{precision_score(self.y2, test_pred):.2f}"
                recall = f"{recall_score(self.y2, test_pred):.2f}"

                metrics.append(
                    {
                        "Model": names,
                        "Train score": train_accuracy,
                        "Test score": test_accuracy,
                        "Precision": precision,
                        "Recall": recall
                    }
                )

    # GridSearch results.
    def result(self, result):

        return pd.DataFrame(self.result)

    # Result to a dataframe.
    def metrics(self, metrics):
        return pd.DataFrame(self.metrics)


# # Confusion matrix...
# cm = confusion_matrix(y2, test_pred)

# sns.heatmap(
#     cm, cbar=False,
#     cmap="coolwarm", annot=True,
#     xticklabels=[True, False],
#     yticklabels=["True False"], fmt="d"
#     ); plt.title(f"{name}'s Confusion Matrix")
# plt.show()

# # AUC and ROC curve...
# if hasattr(grid_search.best_estimator_, "predict_proba"):
#     y_prob = grid_search.predict_proba(X2)[:, 1]
#     AUC = roc_auc_score(y2, y_prob)

# # Plotting ROC curve...
# fpr, tpr, _ = roc_curve(y2, y_prob)

# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, label=f"AUC: {AUC:.2f}")
# plt.plot([0, 1], [0, 1], "k--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.title(f"{name}'s ROC Curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="upper left")
# plt.tight_layout()
# plt.show()
