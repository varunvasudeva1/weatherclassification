library(tensorflow)
library(keras)
library(reticulate)
library(tidyverse)
library(caret)

use_condaenv("r-reticulate")
use_backend("plaidml")

weather <- read.csv("weatherHistory.csv")
head(weather)

weather <- weather %>%
  select(!(Formatted.Date|Summary|Daily.Summary|Loud.Cover)) %>%
  rename(precip = Precip.Type, temp = Temperature..C., atemp = Apparent.Temperature..C., humidity = Humidity, windspeed = Wind.Speed..km.h., windbearing = Wind.Bearing..degrees., visibility = Visibility..km., pressure = Pressure..millibars.) %>%
  filter(precip == "rain" | precip == "snow") %>%
  mutate(precip = as.factor(precip)) %>%
  drop_na()

# data allocation

train_prop <- 0.60
val_prop <- 0.20
test_prop <- 0.20

train_size <- floor(train_prop * nrow(weather))
val_size <- floor(val_prop * nrow(weather))
test_size <- floor(test_prop * nrow(weather))

train_indices <- sort(sample(seq_len(nrow(weather)), size = train_size))
temp <- setdiff(seq_len(nrow(weather)), train_indices)
val_indices <- sort(sample(temp, size = val_size))
test_indices <- setdiff(temp, val_indices)

train <- weather[train_indices, ]
validation <- weather[val_indices, ]
test <- weather[test_indices, ]

train_data <- select(train, !precip)
train_labels <- select(train, precip)
val_data <- select(validation, !precip)
val_labels <- select(validation, precip)
test_data <- select(test, !precip)
test_labels <- select(test, precip)

train_data <- data.matrix(train_data)
train_labels <- data.matrix(train_labels)
val_data <- data.matrix(val_data)
val_labels <- data.matrix(val_labels)
test_data <- data.matrix(test_data)
test_labels <- data.matrix(test_labels)

train_labels <- to_categorical(train_labels)
val_labels <- to_categorical(val_labels)
test_labels <- to_categorical(test_labels)

# neural network model

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(7)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 256, activation = 'sigmoid') %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)
network

history <- network %>% 
  fit(train_data, train_labels, epochs = 10, verbose = 2, validation_data = list(val_data, val_labels))
metrics <- network %>% 
  evaluate(test_data, test_labels)

plot(history)
metrics

save_model_tf(network, 'weather_nn')
save_model_weights_tf(network, 'weather_nn')
