################################################################
# 1. Manual ARIMA Forecasting
################################################################
# 1.1. Manual forecast on Cases
cases1 <- Arima(cases_ts, order = c(1, 2, 3))
summary(cases1)  

cases2 <- Arima(cases_ts, order = c(2, 2, 2))
summary(cases2)

cases3 <- Arima(cases_ts, order = c(4, 2, 1))
summary(cases3)


# 1.2. Manual forecast on Deaths
deaths1 <- Arima(deaths_ts, order = c(1, 2, 4))
summary(deaths1)

deaths2 <- Arima(deaths_ts, order = c(2, 2, 2))
summary(deaths2)

deaths3 <- Arima(deaths_ts, order = c(2, 2, 4))
summary(deaths3)  


################################################################
# 2. Auto ARIMA Forecasting
################################################################
horizon <- 82

# 2.1. Cases: ARIMA forecast
# 2.1.1. Validation set
auto_cases <- auto.arima(cases_ts, stepwise = FALSE, max.p = 3, max.q = 7)
summary(auto_cases) # ARIMA(0,2,4)(0,1,0)[365] | AIC=6711.16

forecast_cases <- auto_cases %>% forecast(h = horizon)


# 2.1.2. Test set
auto_cases2 <- auto.arima(train_cases, stepwise = FALSE, max.p = 3, max.q = 7)
summary(auto_cases2) # ARIMA(0,2,4)(0,1,0)[365] | AIC=6762.01

forecast_cases2 <- auto_cases2 %>% forecast(h = horizon)


# 2.2. Deaths: ARIMA forecast
# 2.2.1 Validation set
auto_deaths <- auto.arima(deaths_ts, stepwise = FALSE, max.p = 3, max.q = 7)
summary(auto_deaths) # AIC=5881.18    

forecast_deaths <- forecast(auto_deaths, h = horizon)

# 2.2.2. Test set
auto_deaths2 <- auto.arima(train_deaths, stepwise = FALSE, max.p = 3, max.q = 7)
summary(auto_deaths2) # AIC=5881.18

forecast_deaths2 <- forecast(auto_deaths2, h = horizon)


################################################################
# 3. Model Evaluation
################################################################
# 3.1. Cases: Validation set
cases_arima_df <- data.frame(
  date = validation_data$date,
  cases = validation_data$cases,
  forecast_cases = forecast_cases$mean
)

mae(cases_arima_df$cases, cases_arima_df$forecast_cases)
mape(cases_arima_df$cases, cases_arima_df$forecast_cases)
rmse(cases_arima_df$cases, cases_arima_df$forecast_cases)


# 3.2. Cases: Test set
test_cases_arima_df <- data.frame(
  date = test_set$date,
  cases = test_set$cases,
  forecast_cases = forecast_cases2$mean
)

mae(test_cases_arima_df$cases, test_cases_arima_df$forecast_cases)
mape(test_cases_arima_df$cases, test_cases_arima_df$forecast_cases)
rmse(test_cases_arima_df$cases, test_cases_arima_df$forecast_cases)


# 3.3. Deaths: Validation set
deaths_arima_df <- data.frame(
  date = validation_data$date,
  deaths = validation_data$deaths,
  forecast_deaths = forecast_deaths$mean
)

mae(deaths_arima_df$deaths, deaths_arima_df$forecast_deaths)
mape(deaths_arima_df$deaths, deaths_arima_df$forecast_deaths)
rmse(deaths_arima_df$deaths, deaths_arima_df$forecast_deaths)


# 3.4. Deaths: test set
test_deaths_arima_df <- data.frame(
  date = test_set$date,
  deaths = test_set$deaths,
  forecast_deaths = forecast_deaths2$mean
)

mae(test_deaths_arima_df$deaths, test_deaths_arima_df$forecast_deaths)
mape(test_deaths_arima_df$deaths, test_deaths_arima_df$forecast_deaths)
rmse(test_deaths_arima_df$deaths, test_deaths_arima_df$forecast_deaths)


################################################################
# 3. Visualization
################################################################
# 3.1. Cases
# 3.1.1. Creating a dataframe for validation set forecast
validation_combined_data <- data.frame(
  date = c(train_data$date, validation_data$date),
  actual = c(train_data$cases, validation_data$cases),
  forecast = c(rep(NA, length(train_data$cases)), forecast_cases$mean),
  lower = c(rep(NA, length(train_data$cases)), forecast_cases$lower[,2]),
  upper = c(rep(NA, length(train_data$cases)), forecast_cases$upper[,2]),
  dataset = c(rep("Train", length(train_data$cases)), rep("Validation", length(validation_data$cases)))
)

tail(validation_combined_data, 20)

# 3.1.2. Creating a dataframe for test set forecast
test_combined_data <- data.frame(
  date = c(train_data$date, test_set$date),
  actual = c(train_data$cases, test_set$cases),
  forecast = c(rep(NA, length(train_data$cases)), forecast_cases2$mean),
  lower = c(rep(NA, length(train_data$cases)), forecast_cases2$lower[,2]),
  upper = c(rep(NA, length(train_data$cases)), forecast_cases2$upper[,2]),
  dataset = c(rep("Train", length(train_data$cases)), rep("Test", length(test_set$cases)))
)

# 3.1.3. Visualizing forecast
ggplot() +
  geom_line(aes(date, cases, color = "Cases"), data = daily_data, size = 1.2) +
  geom_line(aes(date, forecast, color = "Validation Forecast"), data = validation_combined_data, linetype = "dashed", size = 1) +
  geom_line(aes(date, forecast, color = "Test Forecast"), data = test_combined_data, linetype = "dashed", size = 1) +
  geom_ribbon(data = subset(validation_combined_data, dataset == "Validation"),
              aes(x = date, ymin = lower, ymax = upper), fill = "blue", alpha = 0.2) +
  geom_ribbon(data = subset(test_combined_data, dataset == "Test"),
              aes(x = date, ymin = lower, ymax = upper), fill = "red", alpha = 0.2) +
  labs(title = "COVID-19 Cases Vs. Forecast from ARIMA(0,2,4)(0,1,0)[365]",
       x = "Date", y = "Cases") +
  theme_minimal() +
  scale_color_manual(values = c("Cases" = "grey40", "Validation Forecast" = "blue",
                                "Test Forecast" = "red"), name = "Forecast Cases")


# 3.2. Deaths
# 3.2.1. Creating a dataframe for validation set forecast
deaths_validation_combined_data <- data.frame(
  date = c(train_data$date, validation_data$date),
  actual = c(train_data$deaths, validation_data$deaths),
  forecast = c(rep(NA, length(train_data$deaths)), forecast_deaths$mean),
  lower = c(rep(NA, length(train_data$deaths)), forecast_deaths$lower[,2]),
  upper = c(rep(NA, length(train_data$deaths)), forecast_deaths$upper[,2]),
  dataset = c(rep("Train", length(train_data$deaths)), rep("Validation", length(validation_data$deaths)))
)

# 3.2.2. Creating a dataframe for test set forecast
deaths_test_combined_data <- data.frame(
  date = c(train_data$date, test_set$date),
  actual = c(train_data$deaths, test_set$deaths),
  forecast = c(rep(NA, length(train_data$deaths)), forecast_deaths2$mean),
  lower = c(rep(NA, length(train_data$deaths)), forecast_deaths2$lower[,2]),
  upper = c(rep(NA, length(train_data$deaths)), forecast_deaths2$upper[,2]),
  dataset = c(rep("Train", length(train_data$deaths)), rep("Test", length(test_set$deaths)))
)

# 3.2.3. Visualizing forecast
ggplot() +
  geom_line(aes(date, deaths, color = "Deaths"), data = daily_data, size = 1.2) +
  geom_line(aes(date, forecast, color = "Validation Forecast"), data = deaths_validation_combined_data, linetype = "dashed", size = 1) +
  geom_line(aes(date, forecast, color = "Test Forecast"), data = deaths_test_combined_data, linetype = "dashed", size = 1) +
  geom_ribbon(data = subset(deaths_validation_combined_data, dataset == "Validation"),
              aes(x = date, ymin = lower, ymax = upper), fill = "blue", alpha = 0.2) +
  geom_ribbon(data = subset(deaths_test_combined_data, dataset == "Test"),
              aes(x = date, ymin = lower, ymax = upper), fill = "red", alpha = 0.2) +
  labs(title = "COVID-19 Deaths Vs. Forecast from ARIMA(3,2,2)",
       x = "Date", y = "Deaths") +
  theme_minimal() +
  scale_color_manual(values = c("Deaths" = "grey40", "Validation Forecast" = "blue",
                                "Test Forecast" = "red"), name = "Forecast Deaths")