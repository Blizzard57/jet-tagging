# install.packages("reticulate")
# install.packages("ggplot2", dependencies = TRUE)

library(reticulate)
library(ggplot2)
np <- import("numpy")

raw_data <- np$load("data/efp_computed/top_train_1.npz")
raw_data$files
x <- raw_data$f[["x"]]
y <- raw_data$f[["y"]]

efp1 <- x[, 1]
length(efp1)

