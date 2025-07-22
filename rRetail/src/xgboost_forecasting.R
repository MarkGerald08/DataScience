#####################################################
# 1. Checking the data
#####################################################
head(test_set)

str(test_set)

summary(test_set)


#####################################################
# 2. Forecasting with XGBoost
#####################################################
library(caret)

# Setting up the parameters
param_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1, 0.3),
  gamma = 0,
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = 1,
  subsample = c(0.25, 0.5, 0.8)
)

#####################################################
# Preparing for forecast
X_train <- as.matrix(train_regressor_1)
y_train <- train_set$no_outlier_sales

X_test <- as.matrix(test_regressor_1)
y_test <- test_set$no_outlier_sales

# Setting up the train control
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Training the model with hyperparameter tuning
tuned_xgb_model <- train(
  x = X_train,
  y = y_train,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = param_grid,
  metric = "RMSE"
)

# printing the best parameters
print(tuned_xgb_model$bestTune)

y_pred <- predict(tuned_xgb_model, newdata = X_test)


#####################################################
# Preparing for forecast
X_train2 <- as.matrix(train_regressor_2)
y_train2 <- train_set$no_outlier_sales

X_test2 <- as.matrix(test_regressor_2)
y_test2 <- test_set$no_outlier_sales

# Setting up the train control
train_control2 <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Training the model with hyperparameter tuning
tuned_xgb_model2 <- train(
  x = X_train2,
  y = y_train2,
  method = "xgbTree",
  trControl = train_control2,
  tuneGrid = param_grid,
  metric = "RMSE"
)

# printing the best parameters
print(tuned_xgb_model2$bestTune)

y_pred2 <- predict(tuned_xgb_model2, newdata = X_test2)


#####################################################
# Preparing for forecast
X_train3 <- as.matrix(train_regressor_3)
y_train3 <- train_set$no_outlier_sales

X_test3 <- as.matrix(test_regressor_3)
y_test3 <- test_set$no_outlier_sales

# Setting up the train control
train_control3 <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Training the model with hyperparameter tuning
tuned_xgb_model3 <- train(
  x = X_train3,
  y = y_train3,
  method = "xgbTree",
  trControl = train_control3,
  tuneGrid = param_grid,
  metric = "RMSE"
)

# printing the best parameters
print(tuned_xgb_model3$bestTune)

y_pred3 <- predict(tuned_xgb_model3, newdata = X_test3)


#####################################################
# Preparing for forecast
X_train4 <- as.matrix(train_regressor_4)
y_train4 <- train_set$no_outlier_sales

X_test4 <- as.matrix(test_regressor_4)
y_test4 <- test_set$no_outlier_sales

# Setting up the train control
train_control4 <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Training the model with hyperparameter tuning
tuned_xgb_model4 <- train(
  x = X_train4,
  y = y_train4,
  method = "xgbTree",
  trControl = train_control4,
  tuneGrid = param_grid,
  metric = "RMSE"
)

# printing the best parameters
print(tuned_xgb_model4$bestTune)

y_pred4 <- predict(tuned_xgb_model4, newdata = X_test4)


#####################################################
# Preparing for forecast
X_train5 <- as.matrix(train_regressor_5)
y_train5 <- train_set$no_outlier_sales

X_test5 <- as.matrix(test_regressor_5)
y_test5 <- test_set$no_outlier_sales

# Setting up the train control
train_control5 <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Training the model with hyperparameter tuning
tuned_xgb_model5 <- train(
  x = X_train5,
  y = y_train5,
  method = "xgbTree",
  trControl = train_control5,
  tuneGrid = param_grid,
  metric = "RMSE"
)

# printing the best parameters
print(tuned_xgb_model5$bestTune)

y_pred5 <- predict(tuned_xgb_model5, newdata = X_test5)


#####################################################
# 2. Model Evaluation
#####################################################
xgb_forecast_df <- data.frame(
  date = test_set$date,
  sales = test_set$no_outlier_sales,
  y_pred = y_pred,
  y_pred2 = y_pred2,
  y_pred3 = y_pred3,
  y_pred4 = y_pred4,
  y_pred5 = y_pred5
)


# 3.2. Evaluating the model
mae_values <- c()
mape_values <- c()
rmse_values <- c()

for (col in names(xgb_forecast_df)[3:ncol(xgb_forecast_df)]) {
  mae_values <- c(mae_values, mae(xgb_forecast_df$sales, xgb_forecast_df[[col]]))
  mape_values <- c(mape_values, mape(xgb_forecast_df$sales, xgb_forecast_df[[col]]))
  rmse_values <- c(rmse_values, rmse(xgb_forecast_df$sales, xgb_forecast_df[[col]]))
}

xgb_evaluation_set_metrics <- data.frame(
  Model = names(xgb_forecast_df)[3:ncol(xgb_forecast_df)],
  MAE = mae_values,
  MAPE = mape_values,
  RMSE = rmse_values
)
