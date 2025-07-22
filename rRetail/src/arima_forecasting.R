#####################################################
# 1. Checking the data
#####################################################
head(train_set)

str(train_set)

summary(train_set)


#####################################################
# 3. ARIMA forecasting
#####################################################
# 3.1. Preparing for forecast
horizon = nrow(test_set)


#####################################################
# 4. Implementing Auto ARIMA model
#####################################################
# 4.1. Adding external regressor.
auto_model <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  max.p = 5, max.q = 7,
)

summary(auto_model) # AIC=8872.03   AICc=8872.09   BIC=8903.75 | ARIMA(3,1,2)

# 4.1.2. Checking the residual
checkresiduals(auto_model)

# 4.1.3. SARIMA forecast with external regressor
auto_model_forecast <- forecast(
  auto_model,
  h = horizon
)

plot(auto_model_forecast)


#####################################################
# 4.1. Adding external regressor.
auto_model_1 <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  max.p = 5, max.q = 7,
  xreg = as.matrix(train_regressor_1)
)

summary(auto_model_1) # AIC=8517.31   AICc=8517.43   BIC=8564.7 | ARIMA(5,1,0)

# 4.1.2. Checking the residual
checkresiduals(auto_model_1)

# 4.1.3. SARIMA forecast with external regressor
auto_model_forecast_1 <- forecast(
  auto_model_1,
  xreg = as.matrix(test_regressor_1),
  h = horizon
)

plot(auto_model_forecast_1)

#####################################################
# 4.1. Adding external regressor.
auto_model_2 <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  max.p = 5, max.q = 7,
  xreg = as.matrix(train_regressor_2)
)

summary(auto_model_2) # AIC=8621.03   AICc=8621.16   BIC=8668.43 | ARIMA(5,1,0)

# 4.1.2. Checking the residual
checkresiduals(auto_model_2)

# 4.1.3. SARIMA forecast with external regressor
auto_model_forecast_2 <- forecast(
  auto_model_2,
  xreg = as.matrix(test_regressor_2),
  h = horizon
)

plot(auto_model_forecast_2)

#####################################################
# 4.1. Adding external regressor.
auto_model_3 <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  max.p = 5, max.q = 7,
  xreg = as.matrix(train_regressor_3)
)

summary(auto_model_3) # AIC=4971.66   AICc=4971.81   BIC=5024.47 | ARIMA(5,0,0)

# 4.1.2. Checking the residual
checkresiduals(auto_model_3)

# 4.1.3. SARIMA forecast with external regressor
auto_model_forecast_3 <- forecast(
  auto_model_3,
  xreg = as.matrix(test_regressor_3),
  h = horizon
)

plot(auto_model_forecast_3)

#####################################################
# 4.1. Adding external regressor.
auto_model_4 <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  max.p = 5, max.q = 7,
  xreg = as.matrix(train_regressor_4)
)

summary(auto_model_4) # AIC=8584.08   AICc=8584.5   BIC=8673.94 | ARIMA(1,1,4)

# 4.1.2. Checking the residual
checkresiduals(auto_model_4)

# 4.1.3. SARIMA forecast with external regressor
auto_model_forecast_4 <- forecast(
  auto_model_4,
  xreg = as.matrix(test_regressor_4),
  h = horizon
)

plot(auto_model_forecast_4)

#####################################################
auto_model_5 <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = FALSE,
  max.p = 5, max.q = 7,
  xreg = as.matrix(train_regressor_5)
)

summary(auto_model_5) # AIC=4188.34   AICc=4189.12   BIC=4309.46 | ARIMA(5,1,0)

# 4.1.2. Checking the residual
checkresiduals(auto_model_5)

# 4.1.3. SARIMA forecast with external regressor
auto_model_forecast_5 <- forecast(
  auto_model_5,
  xreg = as.matrix(test_regressor_5),
  h = horizon
)

plot(auto_model_forecast_5)


#################################################################
###################################################### ######  ##
# 5. Implementing ARIMA model                       ## ######  ##
###################################################### ######  ##
#################################################################
# 4.1. Adding external regressor.
arima1_model <- Arima(
  train_ts,
  order = c(2,0,0),
)

summary(arima1_model) # AIC=9164.7   AICc=9164.73   BIC=9185.85

# 4.1.2. Checking the residual
checkresiduals(arima1_model)

# 4.1.3. SARIMA forecast with external regressor
arima1_forecast <- forecast(
  arima1_model,
  h = horizon
)

plot(arima1_forecast)

#####################################################
arima1_model1 <- Arima(
  train_ts,
  order = c(2,0,0),
  xreg = as.matrix(train_regressor_1)
)

summary(arima1_model1) # AIC=8191.7   AICc=8191.78   BIC=8228.57

# 4.1.2. Checking the residual
checkresiduals(arima1_model1)

# 4.1.3. SARIMA forecast with external regressor
arima1_forecast1 <- forecast(
  arima1_model1,
  xreg = as.matrix(test_regressor_1),
  h = horizon
)

plot(arima1_forecast1)

#####################################################
# 4.1. Adding external regressor.
arima1_model2 <- Arima(
  train_ts,
  order = c(2,0,0),
  xreg = as.matrix(train_regressor_2)
)

summary(arima1_model2) # AIC=8729.53   AICc=8729.61   BIC=8766.4

# 4.1.2. Checking the residual
checkresiduals(arima1_model2)

# 4.1.3. SARIMA forecast with external regressor
arima1_forecast2 <- forecast(
  arima1_model2,
  xreg = as.matrix(test_regressor_2),
  h = horizon
)

plot(arima1_forecast2)

#####################################################
# 4.1. Adding external regressor.
arima1_model3 <- Arima(
  train_ts,
  order = c(2,0,0),
  xreg = as.matrix(train_regressor_3)
)

summary(arima1_model3) # AIC=4990.7   AICc=4990.78   BIC=5027.67

# 4.1.2. Checking the residual
checkresiduals(arima1_model3)

# 4.1.3. SARIMA forecast with external regressor
arima1_forecast3 <- forecast(
  arima1_model3,
  xreg = as.matrix(test_regressor_3),
  h = horizon
)

plot(arima1_forecast3)

#####################################################
# 4.1. Adding external regressor.
arim1_model4 <- Arima(
  train_ts,
  order = c(2,0,0),
  xreg = as.matrix(train_regressor_4)
)

summary(arim1_model4) # AIC=8736.34   AICc=8736.67   BIC=8815.64

# 4.1.2. Checking the residual
checkresiduals(arim1_model4)

# 4.1.3. SARIMA forecast with external regressor
arima1_forecast4 <- forecast(
  arim1_model4,
  xreg = as.matrix(test_regressor_4),
  h = horizon
)

plot(arima1_forecast4)

#####################################################
arima1_model5 <- Arima(
  train_ts,
  order = c(2,0,0),
  xreg = as.matrix(train_regressor_5)
)

summary(arima1_model5) # AIC=4482.84   AICc=4483.49   BIC=4593.44

# 4.1.2. Checking the residual
checkresiduals(arima1_model5)

# 4.1.3. SARIMA forecast with external regressor
arima1_forecast5 <- forecast(
  arima1_model5,
  xreg = as.matrix(test_regressor_5),
  h = horizon
)

plot(arima1_forecast5)


#####################################################
# 6. Implementing ARIMA model
#####################################################
# 4.1. Adding external regressor.
arima2_model <- Arima(
  train_ts,
  order = c(2,0,1)
)

summary(arima2_model) # AIC=8906.92   AICc=8906.97   BIC=8933.36

# 4.1.2. Checking the residual
checkresiduals(arima2_model)

# 4.1.3. SARIMA forecast with external regressor
arima2_forecast <- forecast(
  arima2_model,
  h = horizon
)

plot(arima2_forecast)

#####################################################
arima2_model1 <- Arima(
  train_ts,
  order = c(2,0,1),
  xreg = as.matrix(train_regressor_1)
)

summary(arima2_model1) # AIC=8016.84   AICc=8016.94   BIC=8058.98 

# 4.1.2. Checking the residual
checkresiduals(arima2_model1)

# 4.1.3. SARIMA forecast with external regressor
arima2_forecast1 <- forecast(
  arima2_model1,
  xreg = as.matrix(test_regressor_1),
  h = horizon
)

plot(arima2_forecast1)

#####################################################
# 4.1. Adding external regressor.
arima2_model2 <- Arima(
  train_ts,
  order = c(2,0,1),
  xreg = as.matrix(train_regressor_2)
)

summary(arima2_model2) # AIC=8501.65   AICc=8501.75   BIC=8543.78

# 4.1.2. Checking the residual
checkresiduals(arima2_model2)

# 4.1.3. SARIMA forecast with external regressor
arima2_forecast2 <- forecast(
  arima2_model2,
  xreg = as.matrix(test_regressor_2),
  h = horizon
)

plot(arima2_forecast2)

#####################################################
# 4.1. Adding external regressor.
arima2_model3 <- Arima(
  train_ts,
  order = c(2,0,1),
  xreg = as.matrix(train_regressor_3)
)

summary(arima2_model3) # AIC=4979.5   AICc=4979.6   BIC=5021.76

# 4.1.2. Checking the residual
checkresiduals(arima2_model3)

# 4.1.3. SARIMA forecast with external regressor
arima2_forecast3 <- forecast(
  arima2_model3,
  xreg = as.matrix(test_regressor_3),
  h = horizon
)

plot(arima2_forecast3)

#####################################################
# 4.1. Adding external regressor.
arim2_model4 <- Arima(
  train_ts,
  order = c(2,0,1),
  xreg = as.matrix(train_regressor_4)
)

summary(arim2_model4) # AIC=8592.96   AICc=8593.34   BIC=8677.55

# 4.1.2. Checking the residual
checkresiduals(arim2_model4)

# 4.1.3. SARIMA forecast with external regressor
arima2_forecast4 <- forecast(
  arim2_model4,
  xreg = as.matrix(test_regressor_4),
  h = horizon
)

plot(arima2_forecast4)

#####################################################
arima2_model5 <- Arima(
  train_ts,
  order = c(2,0,1),
  xreg = as.matrix(train_regressor_5)
)

summary(arima2_model5) # AIC=4483.63   AICc=4484.35   BIC=4599.5

# 4.1.2. Checking the residual
checkresiduals(arima2_model5)

# 4.1.3. SARIMA forecast with external regressor
arima2_forecast5 <- forecast(
  arima2_model5,
  xreg = as.matrix(test_regressor_5),
  h = horizon
)

plot(arima2_forecast5)



#####################################################
# 6. Implementing ARIMA model
#####################################################
# 4.1. Adding external regressor.
arima3_model <- Arima(
  train_ts,
  order = c(2,0,2)
)

summary(arima3_model) # AIC=8905.46   AICc=8905.52   BIC=8937.18

# 4.1.2. Checking the residual
checkresiduals(arima3_model)

# 4.1.3. SARIMA forecast with external regressor
arima3_forecast <- forecast(
  arima3_model,
  h = horizon
)

plot(arima3_forecast)

#####################################################

arima3_model1 <- Arima(
  train_ts,
  order = c(2,0,2),
  xreg = as.matrix(train_regressor_1)
)

summary(arima3_model1) # AIC=8005.91   AICc=8006.04   BIC=8053.31

# 4.1.2. Checking the residual
checkresiduals(arima3_model1)

# 4.1.3. SARIMA forecast with external regressor
arima3_forecast1 <- forecast(
  arima3_model1,
  xreg = as.matrix(test_regressor_1),
  h = horizon
)

plot(arima3_forecast1)

#####################################################
# 4.1. Adding external regressor.
arima3_model2 <- Arima(
  train_ts,
  order = c(2,0,2),
  xreg = as.matrix(train_regressor_2)
)

summary(arima3_model2) # AIC=8502.55   AICc=8502.67   BIC=8549.95

# 4.1.2. Checking the residual
checkresiduals(arima3_model2)

# 4.1.3. SARIMA forecast with external regressor
arima3_forecast2 <- forecast(
  arima3_model2,
  xreg = as.matrix(test_regressor_2),
  h = horizon
)

plot(arima3_forecast2)

#####################################################
# 4.1. Adding external regressor.
arima3_model3 <- Arima(
  train_ts,
  order = c(2,0,2),
  xreg = as.matrix(train_regressor_3)
)

summary(arima3_model3) # AIC=4800.03   AICc=4800.16   BIC=4847.56

# 4.1.2. Checking the residual
checkresiduals(arima3_model3)

# 4.1.3. SARIMA forecast with external regressor
arima3_forecast3 <- forecast(
  arima3_model3,
  xreg = as.matrix(test_regressor_3),
  h = horizon
)

plot(arima3_forecast3)


#####################################################
# 4.1. Adding external regressor.
arim3_model4 <- Arima(
  train_ts,
  order = c(2,0,2),
  xreg = as.matrix(train_regressor_4)
)

summary(arim3_model4) # AIC=8593.23   AICc=8593.65   BIC=8683.1

# 4.1.2. Checking the residual
checkresiduals(arim3_model4)

# 4.1.3. SARIMA forecast with external regressor
arima3_forecast4 <- forecast(
  arim3_model4,
  xreg = as.matrix(test_regressor_4),
  h = horizon
)

plot(arima3_forecast4)

#####################################################
arima3_model5 <- Arima(
  train_ts,
  order = c(2,0,2),
  xreg = as.matrix(train_regressor_5)
)

summary(arima3_model5) # AIC=3859.38   AICc=3860.17   BIC=3980.52

# 4.1.2. Checking the residual
checkresiduals(arima3_model5)

# 4.1.3. SARIMA forecast with external regressor
arima3_forecast5 <- forecast(
  arima3_model5,
  xreg = as.matrix(test_regressor_5),
  h = horizon
)

plot(arima3_forecast5)


#####################################################
# 6. Implementing ARIMA model
#####################################################
# 4.1. Adding external regressor.
arima4_model <- Arima(
  train_ts,
  order = c(0,0,2)
)

summary(arima4_model) # AIC=9258.95   AICc=9258.98   BIC=9280.1

# 4.1.2. Checking the residual
checkresiduals(arima4_model)

# 4.1.3. SARIMA forecast with external regressor
arima4_forecast <- forecast(
  arima4_model,
  h = horizon
)

plot(arima4_forecast)

#####################################################

arima4_model1 <- Arima(
  train_ts,
  order = c(0,0,2),
  xreg = as.matrix(train_regressor_1)
)

summary(arima4_model1) # AIC=8190.78   AICc=8190.86   BIC=8227.65

# 4.1.2. Checking the residual
checkresiduals(arima4_model1)

# 4.1.3. SARIMA forecast with external regressor
arima4_forecast1 <- forecast(
  arima4_model1,
  xreg = as.matrix(test_regressor_1),
  h = horizon
)

plot(arima4_forecast1)

#####################################################
# 4.1. Adding external regressor.
arima4_model2 <- Arima(
  train_ts,
  order = c(0,0,2),
  xreg = as.matrix(train_regressor_2)
)

summary(arima4_model2) # AIC=8820.22   AICc=8820.3   BIC=8857.09

# 4.1.2. Checking the residual
checkresiduals(arima4_model2)

# 4.1.3. SARIMA forecast with external regressor
arima4_forecast2 <- forecast(
  arima4_model2,
  xreg = as.matrix(test_regressor_2),
  h = horizon
)

plot(arima4_forecast2)

#####################################################
# 4.1. Adding external regressor.
arima4_model3 <- Arima(
  train_ts,
  order = c(0,0,2),
  xreg = as.matrix(train_regressor_3)
)

summary(arima4_model3) # AIC=5000.14   AICc=5000.22   BIC=5037.11

# 4.1.2. Checking the residual
checkresiduals(arima4_model3)

# 4.1.3. SARIMA forecast with external regressor
arima4_forecast3 <- forecast(
  arima4_model3,
  xreg = as.matrix(test_regressor_3),
  h = horizon
)

plot(arima4_forecast3)

#####################################################
# 4.1. Adding external regressor.
arim4_model4 <- Arima(
  train_ts,
  order = c(0,0,2),
  xreg = as.matrix(train_regressor_4)
)

summary(arim4_model4) # AIC=8746.02   AICc=8746.35   BIC=8825.32

# 4.1.2. Checking the residual
checkresiduals(arim4_model4)

# 4.1.3. SARIMA forecast with external regressor
arima4_forecast4 <- forecast(
  arim4_model4,
  xreg = as.matrix(test_regressor_4),
  h = horizon
)

plot(arima4_forecast4)

#####################################################
arima4_model5 <- Arima(
  train_ts,
  order = c(0,0,2),
  xreg = as.matrix(train_regressor_5)
)

summary(arima4_model5) # AIC=4489.55   AICc=4490.21   BIC=4600.15

# 4.1.2. Checking the residual
checkresiduals(arima4_model5)

# 4.1.3. SARIMA forecast with external regressor
arima4_forecast5 <- forecast(
  arima4_model5,
  xreg = as.matrix(test_regressor_5),
  h = horizon
)

plot(arima4_forecast5)


#####################################################
# 6. Implementing ARIMA model
#####################################################
# 4.1. Adding external regressor.
arima5_model <- Arima(
  train_ts,
  order = c(1,0,2)
)

summary(arima5_model) # AIC=8905.87   AICc=8905.91   BIC=8932.31

# 4.1.2. Checking the residual
checkresiduals(arima5_model)

# 4.1.3. SARIMA forecast with external regressor
arima5_forecast <- forecast(
  arima5_model,
  h = horizon
)

plot(arima5_forecast)

#####################################################

arima5_model1 <- Arima(
  train_ts,
  order = c(1,0,2),
  xreg = as.matrix(train_regressor_1)
)

summary(arima5_model1) # AIC=8024.85   AICc=8024.96   BIC=8066.99

# 4.1.2. Checking the residual
checkresiduals(arima5_model1)

# 4.1.3. SARIMA forecast with external regressor
arima5_forecast1 <- forecast(
  arima5_model1,
  xreg = as.matrix(test_regressor_1),
  h = horizon
)

plot(arima5_forecast1)

#####################################################
# 4.1. Adding external regressor.
arima5_model2 <- Arima(
  train_ts,
  order = c(1,0,2),
  xreg = as.matrix(train_regressor_2)
)

summary(arima5_model2) # AIC=8501.65   AICc=8501.75   BIC=8543.78

# 4.1.2. Checking the residual
checkresiduals(arima5_model2)

# 4.1.3. SARIMA forecast with external regressor
arima5_forecast2 <- forecast(
  arima5_model2,
  xreg = as.matrix(test_regressor_2),
  h = horizon
)

plot(arima5_forecast2)

#####################################################
# 4.1. Adding external regressor.
arima5_model3 <- Arima(
  train_ts,
  order = c(1,0,2),
  xreg = as.matrix(train_regressor_3)
)

summary(arima5_model3) # AIC=4983.55   AICc=4983.65   BIC=5025.8

# 4.1.2. Checking the residual
checkresiduals(arima5_model3)

# 4.1.3. SARIMA forecast with external regressor
arima5_forecast3 <- forecast(
  arima5_model3,
  xreg = as.matrix(test_regressor_3),
  h = horizon
)

plot(arima5_forecast3)

#####################################################
# 4.1. Adding external regressor.
arim5_model4 <- Arima(
  train_ts,
  order = c(1,0,2),
  xreg = as.matrix(train_regressor_4)
)

summary(arim5_model4) # AIC=8591.63   AICc=8592.01   BIC=8676.22

# 4.1.2. Checking the residual
checkresiduals(arim5_model4)

# 4.1.3. SARIMA forecast with external regressor
arima5_forecast4 <- forecast(
  arim5_model4,
  xreg = as.matrix(test_regressor_4),
  h = horizon
)

plot(arima5_forecast4)

#####################################################
arima5_model5 <- Arima(
  train_ts,
  order = c(1,0,2),
  xreg = as.matrix(train_regressor_5)
)

summary(arima5_model5) # AIC=4471.44   AICc=4472.16   BIC=4587.31

# 4.1.2. Checking the residual
checkresiduals(arima5_model5)

# 4.1.3. SARIMA forecast with external regressor
arima5_forecast5 <- forecast(
  arima5_model5,
  xreg = as.matrix(test_regressor_5),
  h = horizon
)

plot(arima5_forecast5)


#####################################################
# Model Evaluation
#####################################################
arima_forecast_df <- data.frame(
  date = test_set$date,
  sales = test_set$no_outlier_sales,
  auto_model = auto_model_forecast$mean,
  auto_arima1 = auto_model_forecast_1$mean,
  auto_arima2 = auto_model_forecast_2$mean,
  auto_arima3 = auto_model_forecast_3$mean,
  auto_arima4 = auto_model_forecast_4$mean,
  auto_arima5 = auto_model_forecast_5$mean,
  arima1_model = arima1_forecast$mean,
  arima1_model1 = arima1_forecast1$mean,
  arima1_model2 = arima1_forecast2$mean,
  arima1_model3 = arima1_forecast3$mean,
  arima1_model4 = arima1_forecast4$mean,
  arima1_model5 = arima1_forecast5$mean,
  arima2_model = arima2_forecast$mean,
  arima2_model1 = arima2_forecast1$mean,
  arima2_model2 = arima2_forecast2$mean,
  arima2_model3 = arima2_forecast3$mean,
  arima2_model4 = arima2_forecast4$mean,
  arima2_model5 = arima2_forecast5$mean,
  arima3_model = arima3_forecast$mean,
  arima3_model1 = arima3_forecast1$mean,
  arima3_model2 = arima3_forecast2$mean,
  arima3_model3 = arima3_forecast3$mean,
  arima3_model4 = arima3_forecast4$mean,
  arima3_model5 = arima3_forecast5$mean,
  arima4_model = arima4_forecast1$mean,
  arima4_model1 = arima4_forecast1$mean,
  arima4_model2 = arima4_forecast2$mean,
  arima4_model3 = arima4_forecast3$mean,
  arima4_model4 = arima4_forecast4$mean,
  arima4_model5 = arima4_forecast5$mean,
  arima5_model = arima5_forecast$mean,
  arima5_model1 = arima5_forecast1$mean,
  arima5_model2 = arima5_forecast2$mean,
  arima5_model3 = arima5_forecast3$mean,
  arima5_model4 = arima5_forecast4$mean,
  arima5_model5 = arima5_forecast5$mean
)

# 3.2. Evaluating the model
mae_values <- c()
mape_values <- c()
rmse_values <- c()

for (col in names(arima_forecast_df)[3:ncol(arima_forecast_df)]) {
  mae_values <- c(mae_values, mae(arima_forecast_df$sales, arima_forecast_df[[col]]))
  mape_values <- c(mape_values, mape(arima_forecast_df$sales, arima_forecast_df[[col]]))
  rmse_values <- c(rmse_values, rmse(arima_forecast_df$sales, arima_forecast_df[[col]]))
}

arima_evaluation_set_metrics <- data.frame(
  Model = names(arima_forecast_df)[3:ncol(arima_forecast_df)],
  MAE = mae_values,
  MAPE = mape_values,
  RMSE = rmse_values
)


# Visualizing the result
ggplot(arima_forecast_df) +
  geom_line(aes(date, sales, color = "Sales")) +
  geom_line(aes(date, arima4_model1, color = "Forecast"), linewidth = 1) +
  scale_color_manual(
    values = c(
      "Sales" = "grey50",
      "Forecast" = "blue"),
    name = NULL) +
  labs(
    x = NULL,
    y = NULL
  ) +
  scale_x_date(
    breaks = scales::date_breaks("2 months"),
    labels = scales::date_format("%m-%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.1, 0.95),
    legend.background = element_rect(fill = "white", color = "white"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 15)
  ) 


