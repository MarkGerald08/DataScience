library(GPfit)
library(DiceKriging)


#####################################################
# 1. Checking the data
#####################################################
head(train_set)

str(train_set)

summary(train_set)

# 1.1. Visualizing the data
ggplot() +
  geom_line(aes(date, no_outlier_sales, color = "Train"), data = train_data, linewidth = .75) +
  geom_line(aes(date, no_outlier_sales, color = "Validation"), data = validation_data, linewidth = .75) +
  scale_color_manual(
    values = c("Train" = "black", "Validation" = "red"),
    name = NULL
  ) +
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
    legend.position = c(0.12, 0.95),
    legend.background = element_rect(fill = "white", color = "white"),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 15)
  ) +
  guides(color = guide_legend(ncol = 2))


#####################################################
# 2.1. Gaussian Process Modeling
#####################################################
# Train a gaussian process model
gaussian_model <- GP_fit()