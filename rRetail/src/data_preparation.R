#####################################################
# 1. Data preparation
#####################################################
# 2.3. Preparing the time series object
train_ts <- ts(train_set$no_outlier_sales, frequency = 365, start = c(2013,1))
test_ts <- ts(test_set$no_outlier_sales, frequency = 365, start = c(2017,1))


#####################################################
# Preparing the external regressors
#####################################################
# External regressors for train data
train_rolling_mean <- train_set[, c("rolling_mean_7", "rolling_mean_30")]
train_rolling_std <- train_set[, c("rolling_std_7", "rolling_std_30")]
train_ewma <- train_set[, c("ewma_1", "ewma_2")]
is_weekend <- train_set[, c("is_weekend")]
train_fourier <- train_set[, c(
  "S1.365", "C1.365", "S2.365", "C2.365",
  "S3.365", "C3.365", "S4.365", "C4.365",
  "S5.365", "C5.365"
)]

#####################################################
train_regressor_1 <- cbind(train_rolling_mean, is_weekend)
train_regressor_2 <- cbind(train_rolling_std, is_weekend)
train_regressor_3 <- cbind(train_ewma, is_weekend)
train_regressor_4 <- cbind(train_fourier, is_weekend)
train_regressor_5 <- train_set[, c(
  "rolling_mean_7", "rolling_mean_30", "rolling_std_7", "rolling_std_30",
  "ewma_1", "ewma_2", "is_weekend", "S1.365", "C1.365", "S2.365", "C2.365",
  "S3.365", "C3.365", "S4.365", "C4.365", "S5.365", "C5.365"
)]


#####################################################
# External regressors for validation data
test_rolling_mean <- test_set[, c("rolling_mean_7", "rolling_mean_30")]
test_rolling_std <- test_set[, c("rolling_std_7", "rolling_std_30")]
test_ewma <- test_set[, c("ewma_1", "ewma_2")]
is_weekend <- test_set[, c("is_weekend")]
test_fourier <- test_set[, c(
  "S1.365", "C1.365", "S2.365", "C2.365",
  "S3.365", "C3.365", "S4.365", "C4.365",
  "S5.365", "C5.365"
)]

#####################################################
test_regressor_1 <- cbind(test_rolling_mean, is_weekend)
test_regressor_2 <- cbind(test_rolling_std, is_weekend)
test_regressor_3 <- cbind(test_ewma, is_weekend)
test_regressor_4 <- cbind(test_fourier, is_weekend)
test_regressor_5 <- test_set[, c(
  "rolling_mean_7", "rolling_mean_30", "rolling_std_7", "rolling_std_30",
  "ewma_1", "ewma_2", "is_weekend", "S1.365", "C1.365", "S2.365", "C2.365",
  "S3.365", "C3.365", "S4.365", "C4.365", "S5.365", "C5.365"
)]