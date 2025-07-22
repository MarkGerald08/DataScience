# Introduction to Exploratory Data Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")
pd.set_option("display.max_columns", 200)


# ---------------------------------------------------
# Data loading
# ---------------------------------------------------
df = pd.read_csv("../data/raw/coaster_db.csv")


# ---------------------------------------------------
# Basic Data Understanding
# ---------------------------------------------------
"""Data Understanding Methods
    shape : this tells us the shape of the data that we loaded in.
    head() : this shows us the first 5 rows of the data.
    columns : to list all the columns.
        NOTE: In the pandas DataFrame, every column is a Series.
    dtype : this will show us each of the colums data type.
    describe() : this shows us some information in statistics about the numeric data.
    """
df.shape
df.head()
df.columns
df.dtypes
df.describe()


# ---------------------------------------------------
# Data Preparation
# ---------------------------------------------------
"""Data Preparation Steps
    1. Dropping irrelevant columns and rows
    2. Identifying duplicated columns
    3. Renaming Columns
    4. Feature Creation
    """
# Example of dropping the columns
# df.drop(["columns"], axis=1) # axis=1 refers to columns, 0 refers to rows.

# Subsetting the columns (this is other way to drop columns)
df = df[["coaster_name",
        # "Length", "Speed",
         "Location", "Status",
         # "Opening date",
         # "Type",
         "Manufacturer",
         # "Height restriction", "Model", "Height",
         # "Inversions", "Lift/launch system", "Cost", "Trains", "Park section",
         # "Duration", "Capacity", "G-force", "Designer", "Max vertical angle",
         # "Drop", "Soft opening date", "Fast Lane available", "Replaced",
         # "Track layout", "Fastrack available", "Soft opening date.1",
         # "Closing date", "Opened",
         # "Replaced by", "Website",
         # "Flash Pass Available", "Must transfer from wheelchair", "Theme",
         # "Single rider line available", "Restraint Style",
         # "Flash Pass available", "Acceleration", "Restraints", "Name",
         "year_introduced",
         "latitude", "longitude",
         "Type_Main",
         "opening_date_clean",
         # "speed1", "speed2", "speed1_value", "speed1_unit",
         "speed_mph",
         # "height_value", "height_unit",
         "height_ft",
         "Inversions_clean", "Gforce_clean"]].copy()

df.shape

# converting date columns into datetime data type.
df["opening_date_clean"] = pd.to_datetime(df["opening_date_clean"])

# Renaming a columns
df = df.rename(columns={"coaster_name": "Coaster_Name",
                    "year_introduced": "Year_Introduced",
                    "opening_date_clean": "Opening_Date_Clean",
                    "speed_mph": "Speed_mph",
                    "height_ft": "Height_ft",
                    "Inversions_clean": "Inversions",
                    "Gforce_clean": "Gforce"})
df.head()