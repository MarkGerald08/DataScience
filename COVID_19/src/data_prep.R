################################################################
# 1. Preparing the data
################################################################
# 1.1. Splitting the data
test_start_date <- as.Date("2023-01-01")
train_set <- subset(daily_data, date < test_start_date)
test_set <- subset(daily_data, date >= test_start_date)

validation_start_date <- as.Date("2022-10-11")
train_data <- subset(train_set, date < validation_start_date)
validation_data <- subset(train_set, date >= validation_start_date)


# 1.2. Visualizing the split number of cases
ggplot() +
  geom_line(data = train_data, aes(x = date, y = cases, color = "Train set"), linewidth=1) +
  geom_line(data = validation_data, aes(x = date, y = cases, color = "Validation set"), linewidth=1) +
  geom_line(data = test_set, aes(x = date, y = cases, color = "Test set"), linewidth=1) +
  labs(
    title = "COVID-19 Cases: Train, Validation, and Test Split (2020-2023)",
    x = "Date",
    y = "Mean Aggregated Cases"
  ) +
  scale_color_manual(values = c("Train set" = "#12355B", "Validation set" = "red",
                                "Test set" = "blue"),
                     name = "Cases Split (2020-2023)") +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  )+
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text()
  )


# 1.3. Visualizing the split number of deaths
ggplot() +
  geom_line(data = train_data, aes(x = date, y = deaths, color = "Train set"), linewidth=1) +
  geom_line(data = validation_data, aes(x = date, y = deaths, color = "Validation set"), linewidth=1) +
  geom_line(data = test_set, aes(x = date, y = deaths, color = "Test set"), linewidth=1) +
  labs(
    title = "COVID-19 Deaths: Train, Validation, and Test Split (2020-2023)",
    x = "Date",
    y = "Mean Aggregated Cases"
  ) +
  scale_color_manual(values = c("Train set" = "#12355B", "Validation set" = "red",
                                "Test set" = "blue"),
                     name = "Deaths Split (2020-2023)") +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  )+
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text()
  )


# 1.4. Preparing time series data
cases_ts <- ts(train_data$cases, frequency = 365, start = c(2020, 1, 21), end = c(2022, 10, 10))
deaths_ts <- ts(train_data$deaths, frequency = 365, start = c(2020, 1, 21), end = c(2022, 10, 10))

train_cases <- ts(train_set$cases, frequency = 365, start = c(2020, 1, 21), end = c(2022, 12, 31))
train_deaths <- ts(train_set$deaths, frequency = 365, start = c(2020, 1, 21), end = c(2022, 12, 31))
