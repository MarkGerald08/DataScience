################################################################
# 1. FB Prophet Forecasting
################################################################
h <- 82

# 1.1.: Prophet Forecast: Cases
# 1.1.1 Validation set
cases_train_1 <- train_data %>%
  rename(ds = date, y = cases)

cases_train_1$date <- as.Date(cases_train_1$ds)


prophet_cases_model_1 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_cases_model_1 <- fit.prophet(prophet_cases_model_1, cases_train_1)

# 1.1.2. Creating future data
validation_future_cases <- make_future_dataframe(prophet_cases_model_1, periods = horizon, freq = "day")

# 1.1.3. Forecasting cases
validation_prophet_forecast_cases <- predict(prophet_cases_model_1, validation_future_cases)

# 1.1.4. Visualizing the forecast
plot(
  prophet_cases_model_1, validation_prophet_forecast_cases,
  xlab = "Date", ylab = "COVID-19 Cases"
)


# 1.1.1 Test set
cases_train_2 <- train_set %>%
  rename(ds = date, y = cases)

cases_train_2$date <- as.Date(cases_train_2$ds)


prophet_cases_model_2 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_cases_model_2 <- fit.prophet(prophet_cases_model_2, cases_train_2)

# 1.1.2. Creating future data
test_future_cases <- make_future_dataframe(prophet_cases_model_2, periods = horizon, freq = "day")

# 1.1.3. Forecasting cases
test_prophet_forecast_cases <- predict(prophet_cases_model_2, test_future_cases)

# 1.1.4. Visualizing the forecast
plot(
  prophet_cases_model_2, test_prophet_forecast_cases,
  xlab = "Date", ylab = "COVID-19 Cases"
)


# 1.2.: Prophet Forecast: Deaths
# 1.2.1. Validation set
deaths_train <- train_data %>%
  rename(ds = date, y = deaths)

deaths_train$date <- as.Date(deaths_train$ds)

prophet_deaths_model <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_deaths_model <- fit.prophet(prophet_deaths_model, deaths_train)

# 1.2.1. Creating future data
validation_future_deaths <- make_future_dataframe(prophet_deaths_model, periods = horizon, freq = "day")

# 1.2.2. Forecasting cases
prophet_forecast_deaths <- predict(prophet_deaths_model, validation_future_deaths)

# 1.2.3. Visualizing the forecast
plot(
  prophet_deaths_model, prophet_forecast_deaths,
  xlab = "Date", ylab = "COVID-19 Deaths"
)


# 1.2.1. Test set
deaths_train_1 <- train_set %>%
  rename(ds = date, y = deaths)

deaths_train_1$date <- as.Date(deaths_train_1$ds)

prophet_deaths_model_1 <- prophet(daily.seasonality = TRUE, seasonality.mode = "multiplicative")
prophet_deaths_model_1 <- fit.prophet(prophet_deaths_model_1, deaths_train_1)

# 1.2.1. Creating future data
test_future_deaths <- make_future_dataframe(prophet_deaths_model_1, periods = horizon, freq = "day")

# 1.2.2. Forecasting cases
test_prophet_forecast_deaths <- predict(prophet_deaths_model_1, test_future_deaths)

# 1.2.3. Visualizing the forecast
plot(
  prophet_deaths_model_1, test_prophet_forecast_deaths,
  xlab = "Date", ylab = "COVID-19 Deaths"
)


################################################################
# 2. Model Evaluation
################################################################
# 2.1. Model Evaluation: Cases
# 2.1.2. Converting into date format
validation_data$date <- as.Date(validation_data$date)
test_set$date <- as.Date(test_set$date)

validation_prophet_forecast_cases$ds <- as.Date(validation_prophet_forecast_cases$ds)
test_prophet_forecast_cases$ds <- as.Date(test_prophet_forecast_cases$ds)

validation_prophet_forecast = validation_prophet_forecast_cases[which(validation_prophet_forecast_cases$ds %in% validation_data$date), "yhat"]
test_prophet_forecast = test_prophet_forecast_cases[which(test_prophet_forecast_cases$ds %in% test_set$date), "yhat"]

# Validation set
mae(validation_data$cases, validation_prophet_forecast)
mape(validation_data$cases, validation_prophet_forecast)
rmse(validation_data$cases, validation_prophet_forecast)

# Test set
mae(test_set$cases, test_prophet_forecast)
mape(test_set$cases, test_prophet_forecast)
rmse(test_set$cases, test_prophet_forecast)


# 2.1. Model Evaluation: Cases
# 2.1.2. Converting into date format
prophet_forecast_deaths$ds <- as.Date(prophet_forecast_deaths$ds)
test_prophet_forecast_deaths$ds <- as.Date(test_prophet_forecast_deaths$ds)

validation_prophet_forecast_deaths = prophet_forecast_deaths[which(prophet_forecast_deaths$ds %in% validation_data$date), "yhat"]
test_prophet_forecast = test_prophet_forecast_deaths[which(test_prophet_forecast_deaths$ds %in% test_set$date), "yhat"]

# Validation set
mae(validation_data$deaths, validation_prophet_forecast_deaths)
mape(validation_data$deaths, validation_prophet_forecast_deaths)
rmse(validation_data$deaths, validation_prophet_forecast_deaths)

# Test set
mae(test_set$deaths, test_prophet_forecast)
mape(test_set$deaths, test_prophet_forecast)
rmse(test_set$deaths, test_prophet_forecast)
################################################################
# 3. Visualizing the Forecast
################################################################
# 3.1. Cases
# 3.1.1. Creating a dataframe for validation set forecast
validation_combined_prophet <- data.frame(
  date = c(train_data$date, validation_data$date),
  actual = c(train_data$cases, validation_data$cases),
  forecast = c(rep(NA, length(train_data$cases)), validation_prophet_forecast_cases[which(validation_prophet_forecast_cases$ds %in% validation_data$date), "yhat"]),
  lower = c(rep(NA, length(train_data$cases)), validation_prophet_forecast_cases[which(validation_prophet_forecast_cases$ds %in% validation_data$date), "yhat_lower"]),
  upper = c(rep(NA, length(train_data$cases)), validation_prophet_forecast_cases[which(validation_prophet_forecast_cases$ds %in% validation_data$date), "yhat_upper"]),
  dataset = c(rep("Train", length(train_data$cases)), rep("Validation", length(validation_data$cases)))
)

tail(validation_combined_prophet, 20)

# 3.1.2. Creating a dataframe for test set forecast
test_combined_prophet <- data.frame(
  date = c(train_data$date, test_set$date),
  actual = c(train_data$cases, test_set$cases),
  forecast = c(rep(NA, length(train_data$cases)), test_prophet_forecast_cases[which(test_prophet_forecast_cases$ds %in% test_set$date), "yhat"]),
  lower = c(rep(NA, length(train_data$cases)), test_prophet_forecast_cases[which(test_prophet_forecast_cases$ds %in% test_set$date), "yhat_lower"]),
  upper = c(rep(NA, length(train_data$cases)), test_prophet_forecast_cases[which(test_prophet_forecast_cases$ds %in% test_set$date), "yhat_upper"]),
  dataset = c(rep("Train", length(train_data$cases)), rep("Test", length(test_set$cases)))
)

tail(test_combined_prophet, 82)

# 3.1.3. Visualizing forecast
ggplot() +
  geom_line(aes(date, cases, color = "Cases"), data = daily_data, size = 1.2) +
  geom_line(aes(date, forecast, color = "Validation Forecast"), data = validation_combined_prophet, linetype = "dashed", size = 1.2) +
  geom_line(aes(date, forecast, color = "Test Forecast"), data = test_combined_prophet, linetype = "dashed", size = 1.2) +
  geom_ribbon(data = subset(validation_combined_prophet, dataset == "Validation"),
              aes(x = date, ymin = lower, ymax = upper), fill = "blue", alpha = 0.2) +
  geom_ribbon(data = subset(test_combined_prophet, dataset == "Test"),
              aes(x = date, ymin = lower, ymax = upper), fill = "red", alpha = 0.2) +
  labs(title = "COVID-19 Cases Vs. Forecast from Prophet",
       x = "Date", y = "Cases") +
  theme_minimal() +
  scale_color_manual(values = c("Cases" = "grey40", "Validation Forecast" = "blue",
                                "Test Forecast" = "red"), name = "Forecast Cases")


# 3.2. Deaths
# 3.2.1. Creating a dataframe for validation set forecast
validation_combined_prophet_deaths <- data.frame(
  date = c(train_data$date, validation_data$date),
  actual = c(train_data$deaths, validation_data$deaths),
  forecast = c(rep(NA, length(train_data$deaths)), prophet_forecast_deaths[which(prophet_forecast_deaths$ds %in% validation_data$date), "yhat"]),
  lower = c(rep(NA, length(train_data$deaths)), prophet_forecast_deaths[which(prophet_forecast_deaths$ds %in% validation_data$date), "yhat_lower"]),
  upper = c(rep(NA, length(train_data$deaths)), prophet_forecast_deaths[which(prophet_forecast_deaths$ds %in% validation_data$date), "yhat_upper"]),
  dataset = c(rep("Train", length(train_data$deaths)), rep("Validation", length(validation_data$deaths)))
)


# 3.2.2. Creating a dataframe for test set forecast
test_combined_prophet_deaths <- data.frame(
  date = c(train_data$date, test_set$date),
  actual = c(train_data$deaths, test_set$deaths),
  forecast = c(rep(NA, length(train_data$deaths)), test_prophet_forecast_deaths[which(test_prophet_forecast_deaths$ds %in% test_set$date), "yhat"]),
  lower = c(rep(NA, length(train_data$deaths)), test_prophet_forecast_deaths[which(test_prophet_forecast_deaths$ds %in% test_set$date), "yhat_lower"]),
  upper = c(rep(NA, length(train_data$deaths)), test_prophet_forecast_deaths[which(test_prophet_forecast_deaths$ds %in% test_set$date), "yhat_upper"]),
  dataset = c(rep("Train", length(train_data$deaths)), rep("Test", length(test_set$deaths)))
)

tail(validation_combined_prophet_deaths)

# 3.2.3. Visualizing the forecast
ggplot() +
  geom_line(aes(date, deaths, color = "Deaths"), data = daily_data, size = 1.2) +
  geom_line(aes(date, forecast, color = "Validation Forecast"), data = validation_combined_prophet_deaths, linetype = "dashed", size = 1.2) +
  geom_line(aes(date, forecast, color = "Test Forecast"), data = test_combined_prophet_deaths, linetype = "dashed", size = 1.2) +
  geom_ribbon(data = subset(validation_combined_prophet_deaths, dataset == "Validation"),
              aes(x = date, ymin = lower, ymax = upper), fill = "blue", alpha = 0.2) +
  geom_ribbon(data = subset(test_combined_prophet_deaths, dataset == "Test"),
              aes(x = date, ymin = lower, ymax = upper), fill = "red", alpha = 0.2) +
  labs(title = "COVID-19 Deaths Vs. Forecast from Prophet",
       x = "Date", y = "Deaths") +
  theme_minimal() +
  scale_color_manual(values = c("Deaths" = "grey40", "Validation Forecast" = "blue",
                                "Test Forecast" = "red"), name = "Forecast Deaths")
