################################################################
# Importing libraries
################################################################
library(ggplot2)
library(readxl)
library(dplyr)
library(lubridate)
library(tidyr)
library(forecast)
library(prophet)
library(Metrics)
library(gridExtra)
library(patchwork)
library(tseries)


################################################################
# Exploring the data.
# 1. Inspecting the data
################################################################
head(us_states)

str(us_states)

summary(us_states)


# Counting the states.
state_counts <- us_states %>%
  group_by(state) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

print(state_counts, n = 56)


################################################################
# 2. Handling missing values
################################################################
colSums(is.na(us_states)) # checking for missing values


################################################################
# 3. Visualizing the data
################################################################
# 3.1. Visualizing COVID-19 Cases and  Deaths Over Time.
# 3.1.1. Cases
ggplot() +
  geom_line(data = us_states, aes(x = date, y = cases), linewidth=1) +
  labs(
    title = "US States COVID-19 Cases",
    x = "",
    y = ""
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

# 3.1.2. Deaths
ggplot() +
  geom_line(data = us_states, aes(x = date, y = deaths), linewidth=1) +
  labs(
    title = "US States COVID-19 Deaths",
    x = "",
    y = ""
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


# 3.2. Visualizing COVID-19 cases and deaths distribution.
# 3.2.1. Cases
d_cases <- ggplot(us_states, aes(cases / 1e6)) +
  geom_histogram(fill = "white", color = "black") +
  labs(
    #title = "COVID-19 Cases Distribution",
    x = "Cases (in millions)",
    y = ""
  ) +
  theme_minimal()


# 3.2.2. Deaths
d_deaths <- ggplot(us_states, aes(deaths / 1e3)) +
  geom_histogram(fill = "white", color = "black") +
  labs(
    #title = "COVID-19 Deaths Distribution",
    x = "Deaths (in thousands)",
    y = ""
  ) +
  theme_minimal()

(d_cases + d_deaths) + 
  plot_annotation(
    title = "Distribution of COVID-19 Cases and Deaths Across U.S. States"
  )


# 3.3.2. Deaths
ggplot(global_trends, aes(x = date)) +
  geom_line(aes(y = deaths / 1e3)) +
  labs(
    title = "US States COVID-19 Deaths Over Time",
    x = "Date",
    y = "Deaths (in thousands)",
  ) +
  theme_minimal()


# 4.1. Visualizing the total number of cases and deaths by state
# 4.1.1. Summarizing the cases by states.
state_summary <- us_states %>%
  group_by(state) %>%
  summarise(
    total_Cases = sum(cases, na.rm = TRUE),
    total_deaths = sum(deaths, na.rm = TRUE)
  )

# 4.1.2. Plotting the total cases by states
ggplot(state_summary, aes(x = reorder(state, -total_Cases), y = total_Cases / 1e9)) +
  geom_bar(stat = "identity", fill = "black") +
  labs(title = "Total COVID-19 Cases by State",
       x = "",
       y = "Total Cases (in billions)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# 4.1.3. Plotting the total deaths by states
ggplot(state_summary, aes(x = reorder(state, -total_deaths), y = total_deaths / 1e6)) +
  geom_bar(stat = "identity", fill = "black") +
  labs(title = "Total COVID-19 Deaths by State",
       x = "",
       y = "Total Deaths (in millions)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


################################################################
# 5. Changing the frequency of the data
################################################################
daily_data <- us_states %>%
  mutate(date = floor_date(date, "day")) %>%
  group_by(date) %>%
  summarise(
    cases = mean(cases, na.rm = TRUE),
    deaths = mean(deaths, na.rm = TRUE)
  )


# 5.1. Visualizing the distribution of the new data.
# 5.1.2. Cases
p_cases <- ggplot(daily_data, aes(cases)) +
  geom_histogram(fill = "white", color = "black") +
  labs(
    #title = "Mean Distribution of Aggregated COVID-19 Cases Across U.s. States",
    x = "Cases", 
    y = ""
  ) +
  theme_minimal()

# 5.1.3. Deaths
p_deaths <- ggplot(daily_data, aes(deaths)) +
  geom_histogram(fill = "white", color = "black") +
  labs(
    #title = "Aggregated COVID-19 Deaths Distribution",
    x = "Deaths", 
    y = ""
  ) +
  theme_minimal()

(p_cases + p_deaths) + 
  plot_annotation(
    title = "Mean Distribution of Aggregated COVID-19 Cases and Deaths Across U.S. States"
  )


# 5.2. Extracting features
daily_data$year <- format(daily_data$date, "%Y")
daily_data$month <- format(daily_data$date, "%m")
daily_data$weekday <- format(daily_data$date, "%w")
daily_data$day <- format(daily_data$date, "%d")

# 5.2.3. Converting Sunday(0) to 7.
daily_data$weekday[daily_data$weekday == 0] <- 7

# 5.3. Visualizing the progress
# 5.3.1. Cases: Yearly progress
ggplot(daily_data, aes(x = as.factor(year), y = cases)) +
  geom_line(aes(group=1), size =1.5) +
  labs(title = NULL, x="Year") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# 5.3.2. Cases: Monthly progress
ggplot(daily_data, aes(x = as.factor(month), y = cases)) +
  geom_line(aes(group=1), size =1.5) +
  labs(title = NULL, x="Month") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# 5.3.3. Cases: Daily progress
ggplot(daily_data, aes(x = as.factor(day), y = cases)) +
  geom_line(aes(group=1), size =1.5) +
  labs(title = NULL, x="Day") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()


# 5.4.1. Deaths: Yearly progress
ggplot(daily_data, aes(x = as.factor(year), y = deaths)) +
  geom_line(aes(group=1), size =1.5) +
  labs(title = NULL, x="Year") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# 5.4.2. Deaths: Monthly progress
ggplot(daily_data, aes(x = as.factor(month), y = deaths)) +
  geom_line(aes(group=1), size =1.5) +
  labs(title = NULL, x="Month") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()

# 5.4.3. Deaths: Daily progress
ggplot(daily_data, aes(x = as.factor(day), y = deaths)) +
  geom_line(aes(group=1), size =1.5) +
  labs(title = NULL, x="Day") +
  theme( plot.title=element_text(vjust=3, size=15) ) + theme_minimal()


# 6.3. Visualizing the new data
# 6.3.1. Visualizing the number of cases
ggplot() +
  geom_line(data = daily_data, aes(x = date, y = cases), linewidth=1) +
  labs(
    title = "COVID-19 Cases in U.S. States (2020-2023)",
    x = "Date",
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

# 6.3.2. Visualizing the number of deaths
ggplot() +
  geom_line(data = daily_data, aes(x = date, y = deaths), linewidth=1) +
  labs(
    title = "COVID-19 Deaths in U.S. States (2020-2023)",
    x = "Date",
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

# 6.4. Visualizing extracted features using boxplots.
# 6.4.1 COVID-19 by years
ggplot(daily_data, aes(x = year, y = cases)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Yearly COVID-19 Cases (2020-2023)",
    x = "",
    y = ""
  ) +
  theme_minimal()

ggplot(daily_data, aes(x = year, y = deaths)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Yearly COVID-19 Deaths (2020-2023)",
    x = "",
    y = ""
  ) +
  theme_minimal()


# 6.4.2. COVID-19 by month
ggplot(daily_data, aes(x = month, y = cases)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Monthly COVID-19 Cases",
    x = "Month",
    y = ""
  ) +
  theme_minimal()

ggplot(daily_data, aes(x = month, y = deaths)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Monthly COVID-19 Deaths",
    x = "Month",
    y = ""
  ) +
  theme_minimal()


# 6.4.3. COVID-19 by week
ggplot(daily_data, aes(x = weekday, y = cases)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Weekly COVID-19 Cases",
    x = "Week",
    y = ""
  ) +
  theme_minimal()

# 6.4.4. COVID-19 by week
ggplot(daily_data, aes(x = weekday, y = deaths)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Weekly COVID-19 Deaths",
    x = "",
    y = ""
  ) +
  theme_minimal()


# 6.4.5. COVID-19 by day
ggplot(daily_data, aes(x = day, y = cases)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Daily COVID-19 Cases",
    x = "",
    y = ""
  ) +
  theme_minimal()

# 6.4.6 COVID-19 by day
ggplot(daily_data, aes(x = day, y = deaths)) +
  geom_boxplot() +
  labs(
    title = "Mean Aggregated Daily COVID-19 Deaths",
    x = "",
    y = ""
  ) +
  theme_minimal()