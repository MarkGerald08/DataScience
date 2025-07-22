#####################################################
# Checking the data
#####################################################
head(train_set, 10) # data set with feature engineered variables.

str(train_set)

summary(train_set)

# Visualizing the data
ggplot(train_set) +
  geom_line(aes(date, no_outlier_sales)) +
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
# Time series decomposition
#####################################################
# Preparing the time series object
sales_ts <- ts(train_set$no_outlier_sales, frequency = 365, start = c(2013, 1))

# Decomposing the time series with STL
sales_decomp <- stl(sales_ts, s.window = "periodic")

# Visualizing the decomposition
autoplot(sales_decomp, xlab = "")


#####################################################
# Checking stationarity
#####################################################
# Performing Augmented Dickey-Fuller Test
print(adf.test(sales_ts))

# ADF Test result
# Test Statistics: -4.4279
# p-value: 0.01
# alternative hypothesis: stationary


#####################################################
# Visualizing Autocorrelation (ACF and PACF)
#####################################################
# Autocorrelation Function
acf(sales_ts, xlab = "", ylab = "", main = "acf(sales)")

# Applying first differencing
acf(diff(sales_ts), xlab = "", ylab = "", main = "acf(diff(sales))")

# Partial Autocorrelation Function
pacf(sales_ts, xlab = "", ylab = "", main = "pacf(sales)")

# Applying first differencing
pacf(diff(sales_ts), xlab = "", ylab = "", main = "pacf(diff(sales))")
