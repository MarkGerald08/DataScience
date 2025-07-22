#####################################################
# Importing libraries
#####################################################
library(ggplot2)
library(readxl)
library(dplyr)
library(lubridate)
library(tidyr)
library(gridExtra)
library(patchwork)
library(tseries)
library(tidyverse)
library(zoo)
library(lubridate)
library(imputeTS)
library(TTR)
library(fpp)
library(fpp2)
library(Metrics)
library(forecast)
library(prophet)
library(xgboost)
library(anomalize)


#####################################################
# 1. Checking the dataset
#####################################################
head(dataset)

str(dataset)

summary(dataset)

# a. Converting into date format
dataset$date <- as.Date(dataset$date)

# b. Checking for missing values
colSums(is.na(dataset))

# c. Checking for duplicated values
sum(duplicated(dataset))


#####################################################
# 2. Data Exploration
#####################################################
ggplot(dataset) +
  geom_line(aes(date, sales)) +
  #geom_smooth(aes(date, sales), color = "blue", linewidth = 1.2) +
  labs(
    title = NULL,
    x = "Date",
    y = "Sales"
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text()
  )


#####################################################
# 3. Extracting features
#####################################################
# 3.1. Extracting date features
dataset$year <- format(dataset$date, "%Y")
dataset$month <- format(dataset$date, "%m")
dataset$week <- format(dataset$date, "%w")
dataset$day <- format(dataset$date, "%d")

# 3.2. Renaming feature.
dataset$month <- factor(
  dataset$month,
  levels = c("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"),
  labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
)

dataset$week <- factor(
  dataset$week,
  levels = c("1", "2", "3", "4", "5", "6", "0"),
  labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
)

head(dataset)

# 3.3. Visualizing the extracted features
# Year
ggplot(dataset, aes(x = as.factor(year), y = sales)) +
  geom_boxplot() +
  labs(title = NULL, x= NULL, y = "Sales") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# Month
ggplot(dataset) +
  geom_boxplot(aes(x = as.factor(month), y = sales)) +
  labs(title = NULL, x= NULL, y = "Sales") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# Week
ggplot(dataset, aes(x = week, y = sales)) +
  geom_boxplot() +
  labs(title = NULL, x= NULL, y = "Sales") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# Day
ggplot(dataset, aes(x = as.factor(day), y = sales)) +
  geom_boxplot() +
  labs(title = NULL, x= NULL, y = "Sales") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()


#####################################################
# 4. Cutting a Yearly data
#####################################################
# 4.1. Getting unique years in the dataset
unique_years <-sort(unique(dataset$year))

# 4.2. Split the years into 4
year_split <- split(unique_years, cut(seq_along(unique_years), 5, labels = FALSE))

# 4.3. Visualizing the data
ggplot(dataset[dataset$year %in% year_split[[1]], ]) +
  geom_line(aes(x = date, y = sales)) +
  labs(title = NULL, x="Date") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

ggplot(dataset[dataset$year %in% year_split[[2]], ]) +
  geom_line(aes(x = date, y = sales)) +
  labs(title = NULL, x="Date") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

ggplot(dataset[dataset$year %in% year_split[[3]], ]) +
  geom_line(aes(x = date, y = sales)) +
  labs(title = NULL, x="Date") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

ggplot(dataset[dataset$year %in% year_split[[4]], ]) +
  geom_line(aes(x = date, y = sales)) +
  labs(title = NULL, x="Date") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

ggplot(dataset[dataset$year %in% year_split[[5]], ]) +
  geom_line(aes(x = date, y = sales)) +
  labs(title = NULL, x="Date") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()


#####################################################
# 5. Visualizing the Distribution
#####################################################
ggplot(dataset) +
  geom_histogram(aes(no_outlier_sales), color = "white", fill = "black") +
  labs(
    title = NULL,
    x = "Sales",
    y = NULL
  ) + theme_minimal()


#####################################################
# 6. Outlier Detection
#####################################################
# 6.1. Visualizing boxplot
ggplot(dataset, aes(x = date, y = sales)) +
  geom_boxplot() +
  labs(
    title = NULL,
    x = "Year",
    y = "Sales"
  ) +
  theme_minimal()

#####################################################
# 6.2. IQR Method
Q1 <- quantile(dataset$sales, 0.25)
Q3 <- quantile(dataset$sales, 0.75)
IQR <- Q3 - Q1

# 6.3. Defining the threshold.
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# 6.4. Identifying the outlier.
iqr_outlier_df <- data.frame(
  date = dataset$date,
  year = dataset$year,
  sales = dataset$sales,
  outlier = ifelse(dataset$sales < lower_bound | dataset$sales > upper_bound, 1, 0),
  no_outlier_sales = ifelse(dataset$sales < lower_bound | dataset$sales > upper_bound, NA, dataset$sales)
)

# 6.5. Visualizing the marked outlier.
ggplot(iqr_outlier_df, aes(x = date, y = sales)) +
  geom_point(aes(color = factor(outlier))) +
  scale_color_manual(values = c("black", "red"), labels = c("Sales", "Outliers"),
                     name = NULL) +
  labs(
    title = NULL,
    x = "Date",
    y = NULL,
    color = "Series"
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.13, 0.95),
    legend.background = element_rect(color = "white"),
    legend.text = element_text(size = 15),
    legend.title = element_text(face = "bold", size = 12)
  ) +
  guides(color = guide_legend(ncol = 2))


dataset$no_outlier_sales <- ifelse(dataset$sales < lower_bound | dataset$sales > upper_bound, NA, dataset$sales)


# Visualizing no outliers

ggplot(dataset[dataset$year %in% year_split[[3]], ]) +
  geom_line(aes(x = date, y = no_outlier_sales)) +
  labs(title = NULL, x="Date", y = NULL) +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()



#####################################################
# Checking for missing values
colSums(is.na(dataset))

# Treating missing values with AR
dataset$no_outlier_sales <- na_kalman(dataset$no_outlier_sales, model = "auto.arima")

help("na_kalman")

# Visualizing a boxplot
ggplot(dataset) +
  geom_line(aes(date, no_outlier_sales)) +
  #geom_smooth(aes(date, sales), color = "blue") +
  labs(
    title = NULL,
    x = "Date",
    y = "Sales"
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text()
  )


# 3.3. Visualizing the extracted features
# Year
ggplot(dataset, aes(x = as.factor(year), y = no_outlier_sales)) +
  geom_boxplot() +
  labs(title = NULL, x="Year") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# Month
ggplot(dataset) +
  geom_boxplot(aes(x = as.factor(month), y = no_outlier_sales)) +
  labs(title = NULL, x="Month") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# Week
ggplot(dataset, aes(x = week, y = no_outlier_sales)) +
  geom_boxplot() +
  labs(title = NULL, x=NULL, y = NULL) +
  theme_minimal() + theme(plot.title = element_text(face = "bold", size = 12))

# Day
ggplot(dataset, aes(x = as.factor(day), y = no_outlier_sales)) +
  geom_boxplot() +
  labs(title = NULL, x="Day") +
  theme( plot.title =element_text(vjust=3, size=15, face = "bold") ) + theme_minimal()

