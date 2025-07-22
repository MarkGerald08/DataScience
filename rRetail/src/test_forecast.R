#####################################################
# Forecast on test set
#####################################################
head(test_set)

str(test_set)

summary(test_set)

# Visualizing the data
ggplot() +
  geom_line(aes(date, no_outlier_sales, color = "Train"), data = train_data, linewidth = .75) +
  geom_line(aes(date, no_outlier_sales, color = "Validation"), data = validation_data, linewidth = .75) +
  geom_line(aes(date, no_outlier_sales, color = "Test"), data = test_set, linewidth = .75) +
  scale_color_manual(values = c("Train" = "black", "Validation" = "blue",
                                "Test" = "red"),
                     name = "Data Splits") +
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
    axis.text.x = element_text(),
    legend.position = c(0.08, 0.9),
    #legend.background = element_rect(fill = "white", color = "white"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 15)
  ) +
  guides(color = guide_legend(ncol = 1))


#####################################################
# Preparing for test forecast
#####################################################
horizon = nrow(test_set)

# Preparing the external regressors.
test_reg <- as.matrix(
    test_set[, c(
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    "ewma_1", "ewma_2", "is_weekend")]
)

# Preparing a fourier terms
test_ts <- ts(test_set$no_outlier_sales, frequency = 365, start = c(2017, 7))

test_fterms <- fourier(test_ts, K = 5)


#####################################################
# Final Testing on Best Performing ARIMA
#####################################################
final_arima <- forecast(arima_model6, xreg = cbind(test_reg, test_fterms), h = horizon)
plot(final_arima) # ARIMA(1,0,2)


# Creating a final testing dataframe
final_test_df <- data.frame(
  date = test_set$date,
  sales = test_set$no_outlier_sales,
  forecast = final_arima$mean
)

# 3.2. Evaluating the model
mae_values <- c()
mape_values <- c()
rmse_values <- c()

for (col in names(final_test_df)[3:ncol(final_test_df)]) {
  mae_values <- c(mae_values, mae(final_test_df$sales, final_test_df[[col]]))
  mape_values <- c(mape_values, mape(final_test_df$sales, final_test_df[[col]]))
  rmse_values <- c(rmse_values, rmse(final_test_df$sales, final_test_df[[col]]))
}

final_test_metrics <- data.frame(
  Model = names(final_test_df)[3:ncol(final_test_df)],
  MAE = mae_values,
  MAPE = mape_values,
  RMSE = rmse_values
)


# Visualizing the final test
validation_df_forecast <- data.frame(
  date = validation_data$date,
  sales = validation_data$no_outlier_sales,
  validation_arima = arima_model6_forecast$mean,
  validation_lower = arima_model6_forecast$lower[,2],
  validation_upper = arima_model6_forecast$upper[,2],
  dataset = rep("Validation", length(validation_data$no_outlier_sales))
)

#####################################################
ggplot(validation_df_forecast) +
  geom_line(aes(date, sales, color = "Sales"), linewidth = .75) +
  geom_line(aes(date, validation_arima, color = "Forecast"), linewidth = .75) +
  geom_ribbon(data = subset(validation_df_forecast, dataset == "Validation"),
              aes(x = date, ymin = validation_lower, ymax = validation_upper), fill = "red",
              alpha = .4) +
  scale_color_manual(values = c("Sales" = "black", "Forecast" = "red"),
                     name = "Validation Forecast") +
  labs(
    title = NULL,
    x = "Date",
    y = NULL
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 month"),
    labels = scales::date_format("%m-%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.12, 0.96),
    #egend.background = element_rect(color = "black"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 12)
  ) +
  guides(color = guide_legend(ncol = 2))


#####################################################
final_df_forecast <- data.frame(
  date = test_set$date,
  sales = test_set$no_outlier_sales,
  final_arima = final_arima$mean,
  final_lower = final_arima$lower[,2],
  final_upper = final_arima$upper[,2],
  dataset = rep("Test", length(test_set$no_outlier_sales))
)

ggplot(final_df_forecast) +
  geom_line(aes(date, sales, color = "Sales"), linewidth = .75) +
  geom_line(aes(date, final_arima, color = "Forecast"), linewidth = .75) +
  geom_ribbon(data = subset(final_df_forecast, dataset == "Test"),
              aes(x = date, ymin = final_lower, ymax = final_upper), fill = "red",
              alpha = .2) +
  scale_color_manual(values = c("Sales" = "black", "Forecast" = "red"),
                     name = "ARIMA(1,0,2) Final Forecast") +
  labs(
    title = NULL,
    x = "Date",
    y = "Sales"
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 month"),
    labels = scales::date_format("%m-%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.13, 0.95),
    #egend.background = element_rect(color = "black"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 12)
  ) +
  guides(color = guide_legend(ncol = 2))


#####################################################
ggplot() +
  geom_line(aes(date, no_outlier_sales, color = "Sales"), data = dataset, linewidth =.7) +
  geom_line(aes(date, validation_arima, color = "Validation Forecast"), data = validation_df_forecast, linewidth =.7) +
  geom_line(aes(date, final_arima, color = "Test Forecast"), data = final_df_forecast, linewidth =.7) +
  geom_ribbon(data = subset(validation_df_forecast, dataset == "Validation"),
              aes(x = date, ymin = validation_lower, ymax = validation_upper), fill = "red",
              alpha = .4) +
  geom_ribbon(data = subset(final_df_forecast, dataset == "Test"),
              aes(x = date, ymin = final_lower, ymax = final_upper), fill = "orange",
              alpha = .4) +
  scale_color_manual(values = c("Sales" = "black", "Validation Forecast" = "red",
                                "Test Forecast" = "orange"),
                     name = "Supply Chain Demand Forecast") +
  geom_vline(xintercept = as.Date("2017-07-01"), color = "black", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = as.Date("2016-07-01"), color = "black", linetype = "dashed", linewidth = 1) +
  labs(
    title = NULL,
    x = "Date",
    y = NULL
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.22, 0.95),
    #egend.background = element_rect(color = "black"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 15)
  ) +
  guides(color = guide_legend(ncol = 3))

