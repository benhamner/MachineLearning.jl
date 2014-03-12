library(BayesTree, quietly=TRUE, verbose=FALSE)

# Parameters and data loading
args       <- commandArgs(trailingOnly = TRUE)
outputFile <- args[1]
inputFile  <- args[2]
targetName <- args[3]

data    <- read.csv(inputFile)
x_train <- data[data$is_test!="true",]
x_test  <- data[data$is_test=="true",]
y_train <- as.matrix.data.frame(x_train[targetName])
x_train["is_test"]  <- NULL
x_test["is_test"]   <- NULL
x_train[targetName] <- NULL
x_test[targetName]  <- NULL

model <- bart(as.matrix.data.frame(x_train), as.vector(y_train), as.matrix.data.frame(x_test), ntree=10, printevery=100000, verbose=FALSE)

write.csv(model$yhat.test.mean, outputFile, row.names=FALSE)