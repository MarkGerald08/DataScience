#####################################################
# Checking the data
#####################################################
head(dataset)

str(dataset)

summary(dataset)

summary(dataset$no_outlier_sales)
sd(dataset$no_outlier_sales)

# Checking for missing values
colSums(is.na(dataset))

# Checking for duplicated values
sum(duplicated(dataset))

# Visualizing the data
ggplot(dataset) +
  geom_line(aes(date, no_outlier_sales)) +
  #geom_smooth(aes(date, no_outlier_sales), color = "blue") +
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
# Feature Engineering
#####################################################
# Rolling means and standard deviation
dataset <- dataset %>%
  mutate(
    rolling_mean_7 = rollmean(no_outlier_sales, k = 7, fill = NA, align = "right"), # weekly trend
    rolling_mean_30 = rollmean(no_outlier_sales, k = 30, fill = NA, align = "right"), # monthly trend
    rolling_std_7 = rollapply(no_outlier_sales, width = 7, FUN = sd, fill = NA, align = "right"),
    rolling_std_30 = rollapply(no_outlier_sales, width = 30, FUN = sd, fill = NA, align = "right")
  )

# Exponential weighted moving average
dataset <- dataset %>%
  mutate(
    ewma_1 = EMA(no_outlier_sales, n = 1/0.3 - 1), # alpha = 0.3
    ewma_2 = EMA(no_outlier_sales, n = 1/0.1 - 1) # alpha = 0.1
  )

# Extracting is_weekend features
dataset <- dataset %>%
  mutate(
    is_weekend = ifelse(week %in% c("Sat", "Sun"), 1, 0)
  )


# Preparing a fourier
# Identifying optimal value for K
dataset_ts <- ts(dataset$no_outlier_sales, frequency = 365, start = c(2013, 1))

for (K in 1:5) {
  fourier_terms <- fourier(dataset_ts, K = K)
  model <- auto.arima(dataset_ts, xreg = fourier_terms)
  cat("K =", K, "AIC =", AIC(model), "\n")
}

# creating a fourier dataframe
fourier_terms_df <- data.frame(fourier(dataset_ts, K = 5))
fourier_terms_df$index <- seq_len(nrow(fourier_terms_df))
fourier_terms_df$index <- as.numeric(fourier_terms_df$index)

dataset <- cbind(dataset, fourier_terms_df)


# Visualizing the rolling means
ggplot(dataset) +
  geom_line(aes(date, no_outlier_sales, color = "sales"), size = .5) +
  geom_line(aes(date, rolling_mean_7, color = "rolling_mean_7"), size = .8) +
  geom_line(aes(date, rolling_mean_30, color = "rolling_mean_30"), size = .8) +
  #geom_line(aes(date, rolling_std_7), color = "blue", size = .8) +
  #geom_line(aes(date, rolling_std_30), color = "red", size = .8) +
  scale_color_manual(
    values = c("sales" = "black",
               "rolling_mean_7" = "blue",
               "rolling_mean_30" = "red"),
    name = NULL
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = NULL
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.1, 0.92),
    #legend.background = element_rect(fill = "white", color = "white"),
    legend.text = element_text(size = 12),
    #legend.title = element_text(face = "bold", size = 15)
  ) +
  guides(color = guide_legend(ncol = 1))


# Visualizing the rolling std
ggplot(dataset) +
  geom_line(aes(date, no_outlier_sales, color = "sales"), size = .5) +
  geom_line(aes(date, rolling_std_7, color = "rolling_std_7"), size = .8) +
  geom_line(aes(date, rolling_std_30, color = "rolling_std_30"), size = .8) +
  scale_color_manual(
    values = c("sales" = "black",
               "rolling_std_7" = "blue",
               "rolling_std_30" = "red"),
    name = NULL
  ) +
  labs(
    x = NULL,
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
    legend.position = c(0.1, 0.92),
    #legend.background = element_rect(fill = "white", color = "white"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 15)
  ) +
  guides(color = guide_legend(ncol = 1))


# Visualizing exponential weighted moving average
ggplot(dataset) +
  geom_line(aes(date, no_outlier_sales, color = "sales"), size = 0.5) +
  geom_line(aes(date, ewma_1, color = "ewma alpha = 0.3"), size = 0.8) +
  geom_line(aes(date, ewma_2, color = "ewma alpha = 0.1"), size = 0.8) +
  scale_color_manual(
    values = c("sales" = "black",
               "ewma alpha = 0.3" = "blue",
               "ewma alpha = 0.1" = "red"),
    name = NULL
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = NULL
  ) +
  scale_x_date(
    breaks = scales::date_breaks("1 year"),
    labels = scales::date_format("%Y")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=15),
    axis.text.x = element_text(),
    legend.position = c(0.1, 0.92),
    #legend.background = element_rect(fill = "NULL"white, color = "white"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 15)
  ) +
  guides(color = guide_legend(ncol = 1))


# Visualizing fourier terms
ggplot(fourier_terms_df) +
  geom_line(aes(index, S5.365)) +
  geom_line(aes(index, C5.365), color = "red") +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_minimal()


# Dropping unnecessary columns
dataset <- dataset %>%
  select(-sales, -year, -month, -week, -day)


#####################################################
# Splitting the data
#####################################################
test_start_date <- as.Date("2017-01-01")
train_set <- subset(dataset, date < test_start_date)
test_set <- subset(dataset, date >= test_start_date)

# Visualizing the partition
ggplot() +
  geom_line(aes(date, no_outlier_sales, color = "Train"), data = train_set) +
  geom_line(aes(date, no_outlier_sales, color = "Test"), data = test_set) +
  geom_vline(xintercept = as.Date("2017-01-01"), color = "black", linetype = "dashed", linewidth = 1) +
  scale_color_manual(
    values = c("Train" = "black", "Test" = "red"),
    name = "Data Splits"
  ) +
  labs(
    x = NULL,
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
    legend.position = c(0.09, 0.90),
    #legend.background = element_rect(fill = "NULL"white, color = "white"),
    legend.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 15)
  ) +
  guides(color = guide_legend(ncol = 1))

