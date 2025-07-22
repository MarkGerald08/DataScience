#####################################################
# 1. Checking the data
#####################################################
head(train_set)

str(train_set)

summary(train_set)


#####################################################
# 2.1. Forecasting with Prophet without regressor
#####################################################
# 2.1.1 Preparing a prophet dataframe
prophet_df <- data.frame(ds = as.Date(train_set$date), y = as.numeric(train_set$no_outlier_sales))

# 2.1.3. Initializing the model
prophet_model <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")

# 2.1.4. Fitting the model
prophet_model <- fit.prophet(prophet_model, prophet_df)

# 2.1.5. Creating future dataframe
validation_future_sales <- make_future_dataframe(prophet_model, periods = horizon, freq = "day")

# 2.1.6. Prophet forecasting
prophet_forecast <- predict(prophet_model, validation_future_sales)

# 2.1.7. Visualizing the forecast
plot(prophet_model, prophet_forecast)


#####################################################
# 2.2. Forecasting with Prophet with regressor
#####################################################
# 2.2.1. Preparing a prophet dataframe with external regressors
prophet_df1 <- data.frame(ds = as.Date(train_set$date), y = as.numeric(train_set$no_outlier_sales))
prophet_df1$external_reg <- as.matrix(train_regressor_1)[1:nrow(prophet_df1)]

# 2.2.2. Handling missing values
prophet_df1$external_reg[is.na(prophet_df1$external_reg)] <- 0

# 2.2.3. Initializing the model
prophet_model1 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_model1 <- add_regressor(prophet_model1, "external_reg")

# 2.2.4. Fitting the model
prophet_model1 <- fit.prophet(prophet_model1, prophet_df1)

# 2.2.5. Creating future dataframe
future_sales1 <- make_future_dataframe(prophet_model1, periods = nrow(test_set), freq = "day")
future_sales1$external_reg <- as.matrix(test_regressor_1)[1:nrow(future_sales1)]

future_sales1$external_reg[is.na(future_sales1$external_reg)] <- 0

# 2.2.6. Prophet forecasting
prophet_forecast1 <- predict(prophet_model1, future_sales1)

# 2.2.7. Visualizing the forecast
plot(prophet_model1, prophet_forecast1)


#####################################################
# 2.2. Forecasting with Prophet with regressor
#####################################################
# 2.2.1. Preparing a prophet dataframe with external regressors
prophet_df2 <- data.frame(ds = as.Date(train_set$date), y = as.numeric(train_set$no_outlier_sales))
prophet_df2$external_reg <- as.matrix(train_regressor_2)[1:nrow(prophet_df2)]

# 2.2.2. Handling missing values
prophet_df2$external_reg[is.na(prophet_df2$external_reg)] <- 0

# 2.2.3. Initializing the model
prophet_model2 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_model2 <- add_regressor(prophet_model2, "external_reg")

# 2.2.4. Fitting the model
prophet_model2 <- fit.prophet(prophet_model2, prophet_df2)

# 2.2.5. Creating future dataframe
future_sales2 <- make_future_dataframe(prophet_model2, periods = nrow(test_set), freq = "day")
future_sales2$external_reg <- as.matrix(test_regressor_2)[1:nrow(future_sales2)]

future_sales2$external_reg[is.na(future_sales2$external_reg)] <- 0

# 2.2.6. Prophet forecasting
prophet_forecast2 <- predict(prophet_model2, future_sales2)

# 2.2.7. Visualizing the forecast
plot(prophet_model2, prophet_forecast2)


#####################################################
# 2.2. Forecasting with Prophet with regressor
#####################################################
# 2.2.1. Preparing a prophet dataframe with external regressors
prophet_df3 <- data.frame(ds = as.Date(train_set$date), y = as.numeric(train_set$no_outlier_sales))
prophet_df3$external_reg <- as.matrix(train_regressor_3)[1:nrow(prophet_df3)]

# 2.2.2. Handling missing values
prophet_df3$external_reg[is.na(prophet_df3$external_reg)] <- 0

# 2.2.3. Initializing the model
prophet_model3 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_model3 <- add_regressor(prophet_model3, "external_reg")

# 2.2.4. Fitting the model
prophet_model3 <- fit.prophet(prophet_model3, prophet_df3)

# 2.2.5. Creating future dataframe
future_sales3 <- make_future_dataframe(prophet_model3, periods = nrow(test_set), freq = "day")
future_sales3$external_reg <- as.matrix(test_regressor_3)[1:nrow(future_sales3)]

future_sales3$external_reg[is.na(future_sales3$external_reg)] <- 0

# 2.2.6. Prophet forecasting
prophet_forecast3 <- predict(prophet_model3, future_sales3)

# 2.2.7. Visualizing the forecast
plot(prophet_model3, prophet_forecast3)


#####################################################
# 2.2. Forecasting with Prophet with regressor
#####################################################
# 2.2.1. Preparing a prophet dataframe with external regressors
prophet_df4 <- data.frame(ds = as.Date(train_set$date), y = as.numeric(train_set$no_outlier_sales))
prophet_df4$external_reg <- as.matrix(train_regressor_4)[1:nrow(prophet_df4)]

# 2.2.2. Handling missing values
prophet_df4$external_reg[is.na(prophet_df4$external_reg)] <- 0

# 2.2.3. Initializing the model
prophet_model4 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_model4 <- add_regressor(prophet_model4, "external_reg")

# 2.2.4. Fitting the model
prophet_model4 <- fit.prophet(prophet_model4, prophet_df4)

# 2.2.5. Creating future dataframe
future_sales4 <- make_future_dataframe(prophet_model4, periods = nrow(test_set), freq = "day")
future_sales4$external_reg <- as.matrix(test_regressor_4)[1:nrow(future_sales4)]

future_sales4$external_reg[is.na(future_sales4$external_reg)] <- 0

# 2.2.6. Prophet forecasting
prophet_forecast4 <- predict(prophet_model4, future_sales4)

# 2.2.7. Visualizing the forecast
plot(prophet_model4, prophet_forecast4)


#####################################################
# 2.2. Forecasting with Prophet with regressor
#####################################################
# 2.2.1. Preparing a prophet dataframe with external regressors
prophet_df5 <- data.frame(ds = as.Date(train_set$date), y = as.numeric(train_set$no_outlier_sales))
prophet_df5$external_reg <- as.matrix(train_regressor_5)[1:nrow(prophet_df5)]

# 2.2.2. Handling missing values
prophet_df5$external_reg[is.na(prophet_df5$external_reg)] <- 0

# 2.2.3. Initializing the model
prophet_model5 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative", yearly.seasonality = TRUE)
prophet_model5 <- add_regressor(prophet_model5, "external_reg")

# 2.2.4. Fitting the model
prophet_model5 <- fit.prophet(prophet_model5, prophet_df5)

# 2.2.5. Creating future dataframe
future_sales5 <- make_future_dataframe(prophet_model5, periods = nrow(test_set), freq = "day")
future_sales5$external_reg <- as.matrix(test_regressor_5)[1:nrow(future_sales5)]

future_sales5$external_reg[is.na(future_sales5$external_reg)] <- 0

# 2.2.6. Prophet forecasting
prophet_forecast5 <- predict(prophet_model5, future_sales5)

# 2.2.7. Visualizing the forecast
plot(prophet_model5, prophet_forecast5)


#####################################################
# Model Evaluation
#####################################################
prophet_forecast$ds <- as.Date(prophet_forecast$ds)
prophet_forecast1$ds <- as.Date(prophet_forecast1$ds)
prophet_forecast2$ds <- as.Date(prophet_forecast2$ds)
prophet_forecast3$ds <- as.Date(prophet_forecast3$ds)
prophet_forecast4$ds <- as.Date(prophet_forecast4$ds)
prophet_forecast5$ds <- as.Date(prophet_forecast5$ds)


prophet_forecast_df <- data.frame(
  date = test_set$date,
  sales = test_set$no_outlier_sales,
  prophet = prophet_forecast[which(prophet_forecast$ds %in% test_set$date), "yhat"],
  prophet_1 = prophet_forecast1[which(prophet_forecast1$ds %in% test_set$date), "yhat"],
  prophet_2 = prophet_forecast2[which(prophet_forecast2$ds %in% test_set$date), "yhat"],
  prophet_3 = prophet_forecast3[which(prophet_forecast3$ds %in% test_set$date), "yhat"],
  prophet_4 = prophet_forecast4[which(prophet_forecast4$ds %in% test_set$date), "yhat"],
  prophet_5 = prophet_forecast5[which(prophet_forecast5$ds %in% test_set$date), "yhat"]
)

# 3.2. Evaluating the model
mae_values <- c()
mape_values <- c()
rmse_values <- c()

for (col in names(prophet_forecast_df)[3:ncol(prophet_forecast_df)]) {
  mae_values <- c(mae_values, mae(prophet_forecast_df$sales, prophet_forecast_df[[col]]))
  mape_values <- c(mape_values, mape(prophet_forecast_df$sales, prophet_forecast_df[[col]]))
  rmse_values <- c(rmse_values, rmse(prophet_forecast_df$sales, prophet_forecast_df[[col]]))
}

prophet_evaluation_set_metrics <- data.frame(
  Model = names(prophet_forecast_df)[3:ncol(prophet_forecast_df)],
  MAE = mae_values,
  MAPE = mape_values,
  RMSE = rmse_values
)
