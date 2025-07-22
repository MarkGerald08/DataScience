################################################################
# 1. Checking the data
################################################################
head(daily_data)

str(daily_data)

summary(daily_data)

# 1.1. Dropping unnecessary features
daily_data <- daily_data %>%
  select(-year, -month, -week, -day)


################################################################
# 2. Visualizing the data
################################################################
# 2.1. Visualizing the number of cases
ggplot() +
  geom_line(data = daily_data, aes(x = date, y = cases), linewidth=1) +
  labs(
    title = "COVID-19 Cases in U.S. States (2020-2023)",
    x = "",
    y = "Mean Aggregated Daily Cases"
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  )+
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text()
  )

# 2.2. Visualizing the number of deaths
ggplot() +
  geom_line(data = daily_data, aes(x = date, y = deaths), linewidth=1) +
  labs(
    title = "COVID-19 Deaths in U.S. States (2020-2023)",
    x = "",
    y = "Mean Aggregated Daily Deaths"
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  )+
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text()
  )


# Checking the relationship of the data
ggplot(data = daily_data, aes(x = cases, y= deaths)) +
  geom_point(alpha = 0.6) +
  #geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 0.5) +
  labs(
    title = "",
    x = "Cases",
    y = "Deaths"
  ) +
  theme_minimal()

plot(diff(daily_data$cases), diff(daily_data$deaths))



################################################################
# 3. Time Series Decomposition
################################################################
# 3.1. Preparing the time series data
cases_ts <- ts(daily_data$cases, frequency = 365, start = c(2020, 1, 21))
deaths_ts <- ts(daily_data$deaths, frequency = 365, start = c(2020, 1, 21))

# 3.2. Decomposing the time series
cases_decomp <- stl(cases_ts, s.window = "periodic")
deaths_decomp <- stl(deaths_ts, s.window = "periodic")

# 3.3. Plotting the decomposition
plot(cases_decomp, main = "Decompose Daily Cases")
plot(deaths_decomp, main = "Decompose Daily Deaths")


################################################################
# 4. Stationarity Check
################################################################
# 4.1. Performing Augment Dickey-Fuller test
print(adf.test(cases_ts))
# Augmented Dickey-Fuller Test

# data:  cases_ts
# Dickey-Fuller = -2.9947, Lag order = 10, p-value = 0.1573
# alternative hypothesis: stationary


print(adf.test(deaths_ts))
# Augmented Dickey-Fuller Test

# data:  deaths_ts
# Dickey-Fuller = -1.479, Lag order = 10, p-value = 0.7989
# alternative hypothesis: stationary


# 4.2. Differencing for stationarity
# 4.2.1. Differencing the cases
plot(diff(cases_ts), xlab = "Year", ylab = "diff(cases)", main = "U.S. States COVID-19 Cases Over Time")
print(adf.test(diff(cases_ts)))
# Augmented Dickey-Fuller Test

# data:  diff(cases_ts)
# Dickey-Fuller = -4.0138, Lag order = 10, p-value = 0.01
# alternative hypothesis: stationary


# Second differencing
plot(
  diff(cases_ts, differences = 2),
  xlab = "Year", ylab = "diff(cases)",
  main = "U.S. States COVID-19 Cases Over Time"
)
print(adf.test(diff(cases_ts, differences = 2)))
# Augmented Dickey-Fuller Test

# data:  diff(cases_ts, differences = 2)
# Dickey-Fuller = -7.2747, Lag order = 10, p-value = 0.01
# alternative hypothesis: stationary


# 4.2.2. Differencing the deaths
plot(
  diff(deaths_ts),
  xlab = "Year", ylab = "diff(deaths)",
  main = "U.S. States COVID-19 Deaths Over Time"
)
print(adf.test(diff(deaths_ts)))
# Augmented Dickey-Fuller Test

# data:  diff(deaths_ts)
# Dickey-Fuller = -3.3236, Lag order = 10, p-value = 0.06662
# alternative hypothesis: stationary


# Second Differencing
plot(
  diff(deaths_ts, differences = 2),
  xlab = "Year", ylab = "second diff(deaths)",
  main = "U.S. States COVID-19 Deaths Over Time"
)
print(adf.test(diff(deaths_ts, differences = 2)))
# Augmented Dickey-Fuller Test

# data:  diff(deaths_ts, differences = 2)
# Dickey-Fuller = -14.271, Lag order = 10, p-value = 0.01
# alternative hypothesis: stationary


################################################################
# 5. ACF and PACF
################################################################
# 5.1. ACF for cases
acf(cases_ts, main = "Cases", xlab = "", ylab = "Autocorrelation")
acf(diff(cases_ts), main = "1st diff(cases)", xlab = "", ylab = "Autocorrelation")
acf(diff(cases_ts, differences = 2), main = "2nd diff(cases)", xlab = "", ylab = "Autocorrelation")

# 5.2. PACF for cases
pacf(cases_ts, main = "Cases", xlab = "", ylab = "Partial Autocorrelation")
pacf(diff(cases_ts), main = "1st diff(Cases)", xlab = "", ylab = "Partial Autocorrelation")
pacf(diff(cases_ts, differences = 2), main = "2nd diff(Cases)", xlab = "", ylab = "Partial Autocorrelation")


# 5.3. ACF for deaths
acf(deaths_ts, main = "Deaths", xlab = "", ylab = "Autocorrelation")
acf(diff(deaths_ts), main = "1st diff(Deaths)", xlab = "", ylab = "Autocorrelation")
acf(diff(deaths_ts, differences = 2), main = "2nd diff(Deaths)", xlab = "", ylab = "Autocorrelation")

# 5.4. PACF for deaths
pacf(deaths_ts, main = "Deaths", xlab = "", ylab = "Partial Autocorrelation")
pacf(diff(deaths_ts), main = "1st diff(Deaths)", xlab = "", ylab = "Partial Autocorrelation")
pacf(diff(deaths_ts, differences = 2), main = "2nd diff(Deaths)", xlab = "", ylab = "Partial Autocorrelation")
