# ===========================================================
# Importing libraries and the data.
# ===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.rcParams["figure.dpi"] == 300


# Specify the CSV file path.
PATH = r"../data/raw/Gender Inequality Index.csv"

try:
    df = pd.read_csv(PATH, encoding="latin1")
except UnicodeDecodeError:
    print("UnicodeDecodeError encountered, try different encoding!")

# Checking the data
df.head(10)

df.info()

# Renaming features names.
df = df.rename(
    columns={
        "HDI rank": "HDI Rank",
        "HUMAN DEVELOPMENT": "Human Development",
        "GII VALUE": "GII Value",
        "GII RANK": "GII Rank",
        "Maternal_mortality": "Maternal Mortality",
        "Adolescent_birth_rate": "Adolescent Birth Rate",
        "Seats_parliamentt(% held by women)": "Seats Parliament",
        "F_secondary_educ": "Secondary Education(f)",
        "M_secondary_educ": "Secondary Education(m)",
        "F_Labour_force": "Labour Force(f)",
        "M_Labour_force": "Labour Force(m)"
    }
)

# Dropping unnecessary columns.
df = df.drop("Unnamed: 12", axis=1)


# ===========================================================
# Cleaning the categorical data.
# ===========================================================
clean_df = df.copy()

cat_cols = ["Country", "Human Development"]

# Checking for missing values.
for cols in cat_cols:
    print(clean_df[cols].isna().sum())

# Fixing data formatting
for cols in cat_cols:
    print(clean_df[cols].str.replace("[a-zA-Z0-9 ]", "", regex=True)
          .value_counts().to_frame())
    print("\n", clean_df[clean_df[cols].str.contains(r"\(")][cols])

# Removing unwanted characters.
for cols in cat_cols:
    clean_df[cols] = clean_df[cols].str.replace(r"\s*\(.*?\)", "", regex=True)
    print(clean_df[cols].str.replace(
        "[a-zA-Z0-9 ]", "", regex=True).value_counts())

clean_df.loc[clean_df["Country"].str.contains("[-,ü'ô']")]

# Fixing data capitalization.
clean_df["Human Development"] = clean_df["Human Development"].astype(str)

clean_df["Human Development"] = clean_df["Human Development"].str.title()

clean_df["Human Development"].value_counts()


# ===========================================================
# Cleaning the numerical data.
# ===========================================================
clean_df2 = clean_df.copy()

# Subsetting columns.
obj_cols = list(clean_df2.select_dtypes("object").columns[2:])
len(obj_cols)

# Converting dtypes.
for cols in obj_cols:
    clean_df2[cols] = clean_df2[cols].astype(str)

# Detecting unwanted characters.
for cols in obj_cols:
    print(clean_df2[cols].str.replace("[a-zA-Z0-9. ]", "", regex=True)
          .value_counts().to_frame())

# Removing unwanted characters.
for cols in obj_cols:
    clean_df2[cols] = clean_df2[cols].str.replace("[,]", "", regex=True)


# Fixing data formatting for integer type.
# Replacing unwanted characters into null values.
int_cols = ["GII Rank", "Maternal Mortality"]

# Turning unwanted characters into NaN values.
for cols in int_cols:
    clean_df2[cols] = clean_df2[cols].replace("..", np.nan)


# Converting datatypes
# NOTE: "Int64" can handle missing values when "int" cannot.
for cols in int_cols:
    clean_df2[cols] = clean_df2[cols].astype("Int64")

clean_df2.info()


# Fixing data formatting for float type.
# Replacing unwanted characters into null values.
float_cols = ["GII Value", "Seats Parliament",
              "Secondary Education(f)", "Secondary Education(m)",
              "Labour Force(f)", "Labour Force(m)"]

# Turning unwanted characters into NaN values.
for cols in float_cols:
    clean_df2[cols] = clean_df2[cols].replace("..", np.nan)

# Converting datatypes
for cols in float_cols:
    clean_df2[cols] = clean_df2[cols].astype("float64")

clean_df2.info()


# ===========================================================
# Analyzing missing values.
# ===========================================================
fill_df = clean_df2.copy()

miss_df = fill_df.isna().sum().to_frame(name="Missing Values")
miss_df["Percentage"] = round(fill_df.isna().sum() / len(fill_df) * 100, 2)

# Plotting the distribution of numerical features
num_cols = list(fill_df.select_dtypes(include=[int, float]).columns[-8:])

plt.figure(figsize=(6, 12))
for i, cols in enumerate(num_cols):
    plt.subplot(4, 2, i+1)
    sns.histplot(data=fill_df, x=cols)
    plt.title(cols)
    plt.xlabel("")
    plt.ylabel("")
plt.tight_layout()
plt.show()

# Statistics for imputation (integer)
int_cols = ["GII Rank", "Maternal Mortality"]

int_mean_value = []
int_median_value = []
for cols in int_cols:
    int_mean_value.append(round(fill_df[cols].mean()))
    int_median_value.append(round(fill_df[cols].median()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (col, title) in enumerate(zip(int_cols, int_cols)):
    sns.histplot(data=fill_df, x=col, ax=axes[i])
    axes[i].axvline(x=int_mean_value[i], color='r', linestyle='--', label=f'Mean: {int_mean_value[i]:.2f}')
    axes[i].axvline(x=int_median_value[i], color='b', linestyle='--', label=f'Median: {int_median_value[i]:.2f}')
    axes[i].legend()
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(title)
plt.tight_layout()
plt.show()

"""NOTE: Since the result of using statistics (mean/median) to impute missing value
    introduces bias and potential error that may affect the analysis, dropping them
    would be a better option.
    """

# Statistics for imputation (float)
float_cols = list(fill_df.select_dtypes("float").drop(
    "Adolescent Birth Rate", axis=1).columns)

float_mean_value = []
float_median_value = []
for cols in float_cols:
    float_mean_value.append(fill_df[cols].mean())
    float_median_value.append(fill_df[cols].median())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, (col, title) in enumerate(zip(float_cols, float_cols)):
    row, col_idx = divmod(i, 3)
    sns.histplot(data=fill_df, x=col, ax=axes[row, col_idx])
    axes[row, col_idx].axvline(x=float_mean_value[i], color='r',
                               linestyle='--', label=f'Mean: {float_mean_value[i]:.2f}')
    axes[row, col_idx].axvline(x=float_median_value[i], color='b',
                               linestyle='--', label=f'Median: {float_median_value[i]:.2f}')
    axes[row, col_idx].legend()
    axes[row, col_idx].set_xlabel("")
    axes[row, col_idx].set_ylabel("")
    axes[row, col_idx].set_title(title)
plt.tight_layout()
plt.show()

# Missing values imputation using statistics.
# Analyzing the error for integer data types.
int_mean_result = []
for cols, mean in zip(int_cols, int_mean_value):
    int_mean_result.append(np.mean(np.abs(fill_df[cols] - mean)))

int_median_result = []
for cols, median in zip(int_cols, int_median_value):
    int_median_result.append(np.mean(np.abs(fill_df[cols] - median)))

int_error_df = pd.DataFrame(
    {"Columns": int_cols,
     "Mean Error": [round(mean, 2) for mean in int_mean_result],
     "Median Error": [round(median, 2) for median in int_median_result]}
)

"""Considering the results of the analysis, the median value introduces a lesser error,
    therefore, we will be using median for imputation.
    """
for cols, median in zip(int_cols, int_median_value):
    fill_df[cols] = fill_df[cols].fillna(median)

# Analyzing the error for float data types.
float_mean_result = []
for cols, mean in zip(float_cols, float_mean_value):
    float_mean_result.append(np.mean(np.abs(fill_df[cols] - mean)))

float_median_result = []
for cols, median in zip(float_cols, float_median_value):
    float_median_result.append(np.mean(np.abs(fill_df[cols] - median)))

float_error_df = pd.DataFrame(
    {"Column": float_cols,
     "Mean Error": [round(mean, 2) for mean in float_mean_result],
     "Median Error": [round(median, 2) for median in float_median_result]}
)

"""As the result of the analysis, the median value introduces a lesser error for all features.
    """
for cols, median in zip(float_cols, float_median_value):
    fill_df[cols] = fill_df[cols].fillna(median)

fill_df.head()

# Visualizing the result.
viz_df = fill_df.copy()

num_cols = list(viz_df.select_dtypes(include=["float", "int"])
                .drop("HDI Rank", axis=1).columns)
len(num_cols)

plt.figure(figsize=(15, 12))
for i, cols in enumerate(num_cols):
    plt.suptitle("Feature Distribution After Imputation", size=18)
    plt.subplot(3, 3, i+1)
    sns.histplot(data=viz_df, x=cols)
    plt.title(cols)
    plt.xlabel("")
    plt.ylabel("")
plt.tight_layout()
plt.show()

viz_df.to_csv("../data/processed/gender_ineq_data.csv", index=False)
