# Project 1
library(sigmoid)

setwd("C:/Users/willd/Desktop/Everything/C S 5033")

g_2_w <- read.csv("Gaussian 2d Wide.csv")
g_2_n <- read.csv("Gaussian 2d Narrow.csv")
g_2_o <- read.csv("Gaussian 2d Overlap.csv")

# Pick dataframe to learn from
data <- g_2_o

# X1's
x1 <- c(data[,1], data[,3])
# X2's
x2 <- c(data[,2], data[,4])
# X3's

# Classes
classes <- c(rep(0,99), rep(1,99))

# Range, for plotting
range <- rbind(c(min(x1), max(x1)), c(min(x2), max(x2)))
range <- c(min(range[,1]), max(range[,2]))

# Random starting weight vector
w_v <- runif(3,-0.1,.1)

# Learning rate
learning_rate <- 0.1
epoch <- 0

for(epoch in 1:50000){
  epoch <- epoch + 1
  loss <- 0
  
  for(i in 1:length(x1)) {
    
    # Create p vector for point i
    p <- c(1, x1[i], x2[i])
    # Determine target class value
    target <- classes[i]
    # Calculate sum net value
    net_i <- sum(p * w_v)
    # Perform sigmoidal transformation
    a <- sigmoid(net_i)
    # Calculate error
    error <- target - a
    # Calculate FFANN formula part
    g <- ((1-a)*a)
  
    if(error != 0){
      w_v <- w_v + (learning_rate * g * error * p)
      loss <- loss + (error^2)
    }
  }
  print(loss)
  if(loss < .2){
    break
  }
  # Plot epoch's points and dividing line
  intercept <- -w_v[1]/w_v[3]
  slope <- -w_v[2]/w_v[3]
  plot(x1, x2, col = as.factor(classes), xlim=c(range[1], range[2]), ylim=c(range[1],range[2]))
  abline(a = intercept, b=slope)
}