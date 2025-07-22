# LIBRARIES ----
import scipy.stats as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

pd.set_option('float_format', lambda x: '%.2f' % x)

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# MODELS ----
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# METRICS ----
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# DATA ----
data_path = r'C:\Users\USER\OneDrive\Desktop\dsFolder\MLPrac\Classification\vsScript\churn.csv'

data = pd.read_csv(data_path)
data.drop(columns=['Unnamed: 0'], inplace=True)
data.head()


# DATA UNDERSTANDING ----
data.info()
data.describe().T

data.duplicated().sum()


# -PART 01 -EXPLORATORY DATA ANALYSIS-
plt.figure()
plt.pie(data['leave'].value_counts(), labels=data['leave'].value_counts().keys(),
        autopct='%1.1f%%', textprops={'fontsize': 20, 'fontweight': 'black'})
plt.title('Leave Feature Distribution', size=15)
plt.tight_layout()
plt.show()


# VISUALIZING NUMERICAL FEATURES
data.select_dtypes(include='int').columns

int_cols = ['income', 'overage', 'leftover', 'house', 'handset_price',
            'over_15mins_calls_per_month', 'average_call_duration']

plt.figure(figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features', size=20)
for idx, column in enumerate(int_cols):
    plt.subplot(3, 3, idx+1)
    sns.histplot(x=column, hue='leave', data=data, kde=True)
    plt.title(f'{column} Distribution')

plt.tight_layout()
plt.show()


# VISUALIZING CATEGORICAL FEATURES
data.select_dtypes(include='object').columns

cat_cols = ['college', 'reported_satisfaction',
            'reported_usage_level', 'considering_change_of_plan']

plt.figure(figsize=(15, 10))
plt.suptitle('Distribution of Categorical Features', size=20)
for idx, column in enumerate(cat_cols):
    plt.subplot(2, 2, idx+1)
    sns.countplot(x=column, hue='leave', data=data)
    plt.title(f'{column} Distribution')

plt.tight_layout()
plt.show()


# PART 02 -FEATURE ENGINEERING-
# reported_satisfaction
data.select_dtypes(include=['object'])


def satisfaction(df):
    satisfaction = []
    for i in df['reported_satisfaction']:
        if i == 'very_unsat' or i == 'unsat':
            satisfaction.append('a_unsatisfied')
        else:
            satisfaction.append('satisfied')

    df['satisfaction'] = satisfaction


satisfaction(data)

# VISUALIZING ENGINEERED SATISFACTION----
order = sorted(data['satisfaction'].value_counts().keys().to_list())

plt.figure()
sns.countplot(x='satisfaction', hue='leave', data=data, order=order)
plt.title('Satisfaction Distribution')

plt.tight_layout()
plt.show()


# reported_usage_level----
def usage_level(df):
    usage_level = []
    for i in df['reported_usage_level']:
        if i == 'very_high' or i == 'high':
            usage_level.append('high')
        else:
            usage_level.append('a_little')

    df['usage_level'] = usage_level


usage_level(data)

# VISUALIZING ENGINEERED USAGE LEVEL----
order = sorted(data['usage_level'].value_counts().keys().to_list())

plt.figure()
sns.countplot(x='usage_level', hue='leave', data=data, order=order)
plt.title('Usage Level Distribution')

plt.tight_layout()
plt.show()


# considering_change_of_plan----
def change_of_plan(df):
    change_of_plan = []
    for i in df['considering_change_of_plan']:
        if i == 'no' or i == 'never_thought':
            change_of_plan.append('a_no')
        else:
            change_of_plan.append('considering')

    df['change_of_plan'] = change_of_plan


change_of_plan(data)

# VISUALIZING ENGINEERED CHANGE OF PLAN----
order = sorted(data['change_of_plan'].value_counts().keys().to_list())

plt.figure()
sns.countplot(x='change_of_plan', hue='leave',
              data=data, order=order)
plt.title('Change of Plan Distribution')

plt.tight_layout()
plt.show()


# DROPPING UNNECESSARY COLUMNS
data.head()
data.drop(columns=['reported_satisfaction', 'reported_usage_level',
                   'considering_change_of_plan'], inplace=True)


# PART 03 -ONE-HOT-ENCONDING-
data.columns

data['leave'] = pd.Categorical(data['leave'],
                               categories=['STAY', 'LEAVE'],
                               ordered=True)

data['college'] = [1 if yi == 'one' else 0 for yi in data['college']]

predictors = [
    'college', 'income', 'overage', 'leftover', 'house', 'handset_price',
    'over_15mins_calls_per_month', 'average_call_duration',
    'satisfaction', 'usage_level', 'change_of_plan'
]
outcome = 'leave'

# -NOTE: AVOID DUMMY VARIABLE TRAP-
X = pd.get_dummies(data[predictors],
                   drop_first=True,
                   dtype='int')
y = [1 if yi == 'STAY' else 0 for yi in data[outcome]]  # 1 -STAY; 0 -LEAVE

X

# PART 04 -DATA PREPARATION-
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42)

# SCALED DATA FOR SENSITIVE MODEL
# SENSITIVE MODEL:
# -KNeighborsClassifier
# -LogisticRegression
# -SVC
# -AdaBoostClassifier
# -GradientBoostingClassifier

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train1, X_temp1, y_train1, y_temp1 = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

X_val1, X_test1, y_val1, y_test1 = train_test_split(
    X_temp1, y_temp1, test_size=1/3, random_state=42)


# PART 05 -MODEL TRAINING-
# TRAINING A NON-SCALED DATA

training_score = []
val_score = []


def model_prediction(model):
    model.fit(X_train, y_train)
    X_train_pred = model.predict(X_train)
    X_val_pred = model.predict(X_val)
    train = accuracy_score(y_train, X_train_pred) * 100
    val = accuracy_score(y_val, X_val_pred) * 100
    training_score.append(train)
    val_score.append(val)
    # METRICS -----
    print(f'{model} Training Accuracy Score:', train)
    print(f'{model} Validation Set Accuracy Score:', val)
    print('\n------------------------------------')
    print(f'{model} Precision Score:', precision_score(y_val, X_val_pred))
    print(f'{model} Recall Score:', recall_score(y_val, X_val_pred))
    print(f'{model} F1 Score:', f1_score(y_val, X_val_pred))
    print('\n------------------------------------')
    # ROC AND AUC -----
    print(f'{model} ROC Curve and AUC:')
    X_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, X_proba)
    roc_df = pd.DataFrame({
        'recall': tpr,
        'specificity': fpr
    })

    plt.figure()
    roc_df.plot(x='specificity', y='recall',
                label='ROC curve(AUC = %0.2f)' % auc(fpr, tpr))
    plt.title(f'{model} ROC & AUC')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.plot([0, 1], [0, 1], ls='--')
    plt.xlabel('specificity')
    plt.ylabel('recall')

    plt.legend()
    plt.tight_layout()
    plt.show()


model_prediction(RandomForestClassifier())


# TRAINING A SCALED DATA

def scaled_prediction(model):
    model.fit(X_train1, y_train1)
    X_train_pred1 = model.predict(X_train1)
    X_val_pred1 = model.predict(X_val1)
    train = accuracy_score(y_train1, X_train_pred1) * 100
    val = accuracy_score(y_val1, X_val_pred1) * 100
    training_score.append(train)
    val_score.append(val)

    print(f'{model} Training Accuracy Score:', train)
    print(f'{model} Validation Set Accuracy Score:', val)
    print('\n------------------------------------')
    print(f'{model} Precision Score:', precision_score(y_val1, X_val_pred1))
    print(f'{model} Recall Score:', recall_score(y_val1, X_val_pred1))
    print(f'{model} F1 Score:', f1_score(y_val1, X_val_pred1))
    print('\n------------------------------------')

    print(f'{model} ROC Curve and AUC:')
    X_proba1 = model.predict_proba(X_val1)[:, 1]
    fpr1, tpr1, thresholds1 = roc_curve(y_val1, X_proba1)
    roc_df1 = pd.DataFrame({
        'recall': tpr1,
        'specificity': fpr1
    })
    # auc = auc(fpr, tpr)

    plt.figure()
    roc_df1.plot(x='specificity', y='recall',
                 label='ROC curve(AUC = %0.2f)' % auc(fpr1, tpr1))
    plt.title(f'{model} ROC & AUC')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.plot([0, 1], [0, 1], ls='--')
    plt.xlabel('specificity')
    plt.ylabel('recall')

    plt.legend()
    plt.tight_layout()
    plt.show()


scaled_prediction(GradientBoostingClassifier())


# PART 06 -  ALL MODEL PERFORMANCE COMPARISON
models = [
    'GaussianNB', 'XGBClassifier', 'RandomForestClassifier',
    'DecisionTreeClassifier', 'LightGBMClassifier', 'CatBoostClassifier',
    'KNeighborsClassifier', 'LogisticRegression', 'SVC',
    'AdaBoostClassifier', 'GradientBoostingClassifier'
]

# CREATING DATAFRAME FOR PERFORMANCE
perf = pd.DataFrame({
    'Algorithms': models,
    'Training Score': training_score,
    'Validation Score': val_score
})
perf


# PART 07 - PERFORMING CROSS VALIDATION
# NON-SCALED DATA

# INITIALIZE CLASSIFIER FOR DIFFERENT MODELS
models = {
    'NaiveBayes': GaussianNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'LGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=False),
}

# PERFORM CROSS VALIDATION FOR EACH MODEL
cv_score = {}
for name, clf in models.items():
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    cv_score[name] = scores

# CREATING DATAFRAME FOR SCORE COMPARISON
perf_scores = pd.DataFrame(cv_score).T
perf_scores['mean_score'] = perf_scores.mean(axis=1)
perf_scores

# SCALED DATA
# INITIALIZE CLASSIFIER FOR DIFFERENT MODELS
models1 = {
    'KNeighbors': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(probability=True),
    'AdaBoosT': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# PERFORM CROSS VALIDATION FOR EACH MODEL
cv_score1 = {}
for name, clf in models1.items():
    scores1 = cross_val_score(clf, X_train1, y_train1,
                              cv=10, scoring='accuracy')
    cv_score1[name] = scores1

# CREATING DATAFRAME FOR SCORE COMPARISON
scaled_perf_scores = pd.DataFrame(cv_score1).T
scaled_perf_scores['mean_score'] = scaled_perf_scores.mean(axis=1)
scaled_perf_scores


# PART 08 -PERFORMING HYPER-PARAMETER TUNING FOR BEST MODELS-
# BEST MODEL
# -XGBoost
# -RandomForest
# -LightGMB
# -CatBoost
# -GradientBoosting

# HYPER-PARAMETER TUNING FOR XGBoost----
model1 = XGBClassifier()

parameter1 = {
    'n_estimators': [50, 100, 150],
    'random_state': [0, 42, 50],
    'learning_rate': [0.1, 0.3, 0.5, 1.0]
}
grid_search1 = GridSearchCV(model1, parameter1, cv=5, n_jobs=-1)
grid_search1.fit(X_train, y_train)

print(f'XGBoost Parameters Best Score: {grid_search1.best_score_}') # 0.6997142857142857

best_params1 = grid_search1.best_params_
best_params1

# APPLYING THE BEST PARAMETERS TO XGBoost----
model1 = XGBClassifier(**best_params1)
model1.fit(X_train, y_train)

X_val_pred1 = model1.predict(X_val)
print(f'XGBoost Accuracy Score: {accuracy_score(y_val, X_val_pred1)}') # 0.68975


# HYPER-PARAMETER TUNING FOR RandomForest----
model2 = RandomForestClassifier()

parameter2 = {
    'n_estimators': [100, 300, 500, 550],
    'min_samples_split': [7, 8, 9],
    'max_depth': [10, 11, 12],
    'min_samples_leaf': [4, 5, 6]
}
grid_search2 = GridSearchCV(model2, parameter2, cv=5, n_jobs=-1)
grid_search2.fit(X_train, y_train)

print(f'RandomForest Parameters Best Score: {grid_search2.best_score_}') # 0.7062142857142858

best_params2 = grid_search2.best_params_
best_params2

# APPLYING THE BEST PARAMETERS TO RandomForest----
model2 = RandomForestClassifier(**best_params2)
model2.fit(X_train, y_train)

X_val_pred2 = model2.predict(X_val)
print(f'RandomForest Accuracy Score: {accuracy_score(y_val, X_val_pred2)}') # 0.6975


# HYPER-PARAMETER TUNING FOR LightGBM
model3 = LGBMClassifier()

parameter3 = {
    'n_estimators': [100, 300, 500, 600, 650],
    'learning_rate': [0.01, 0.02, 0.03],
    'random_state': [0, 42, 48, 50],
   'num_leaves': [16, 17, 18]
}
grid_search3 = GridSearchCV(model3, parameter3, cv=5, n_jobs=-1)
grid_search3.fit(X_train, y_train)

print(f'LightGBM Parameters Best Score: {grid_search3.best_score_}') #0.702

best_params3 = grid_search3.best_params_; best_params3

# APPLYING THE BEST PARAMETERS TO LightGBM----
model3 = LGBMClassifier(**best_params3)
model3.fit(X_train, y_train)

X_val_pred3 = model3.predict(X_val)
print(f'LightGBM Accuracy Score: {accuracy_score(y_val, X_val_pred3)}') # 0.69025

# HYPER-PARAMETER TUNING FOR CatBoost----
model4 = CatBoostClassifier(verbose=False)

parameter4 = {
    'learning_rate': [0.1, 0.3, 0.5, 0.6, 0.7],
    'random_state': [0, 42, 48, 50],
    'depth': [8, 9, 10],
    'iterations': [35, 40, 50]
}
grid_search4 = GridSearchCV(model4, parameter4, cv=5, n_jobs=-1)
grid_search4.fit(X_train, y_train)

print(f'CatBoost Parameters Best Score: {grid_search4.best_score_}') # 0.701

best_params4 = grid_search4.best_params_
best_params4

# APPLYING THE BEST PARAMETERS TO CatBoost----
model4 = CatBoostClassifier(**best_params4)
model4.fit(X_train, y_train)

X_val_pred4 = model4.predict(X_val)

print(f'CatBoost Accuracy Score: {accuracy_score(y_val, X_val_pred4)}') # 0.703


# HYPER-PARAMETER TUNING FOR GradientBoost----
model5 = GradientBoostingClassifier()

parameter5 = {
    'n_estimators': [100, 300, 500, 550],
    'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.6],
    'max_depth': [10, 11, 12]
}

grid_search5 = GridSearchCV(model5, parameter5, cv=5, n_jobs=-1)
grid_search5.fit(X_train1, y_train1)

print(f'GradientBoosting Accuracy Score: {grid_search5.best_score_}') # 0.6922857142857143

best_params5 = grid_search5.best_params_
best_params5

# APPLYING THE BEST PARAMETER TO GradientBoosting----
model5 = GradientBoostingClassifier(**best_params5)
model5.fit(X_train1, y_train1)

X_val_pred5 = model5.predict(X_val1)

print(f'GradientBoosting Accuracy Score: {accuracy_score(y_val, X_val_pred5)}') # 0.69075