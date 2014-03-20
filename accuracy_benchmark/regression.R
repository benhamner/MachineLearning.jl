# Parameters and data loading
args       <- commandArgs(trailingOnly = TRUE)
outputFile <- args[1]
inputFile  <- args[2]
targetName <- args[3]
modelName  <- args[4]

data    <- read.csv(inputFile)
x_train <- data[data$is_test!="true",]
x_test  <- data[data$is_test=="true",]
y_train <- as.matrix.data.frame(x_train[targetName])
x_train["is_test"]  <- NULL
x_test["is_test"]   <- NULL
x_train[targetName] <- NULL
x_test[targetName]  <- NULL

if (modelName=="BART") {
    library(BayesTree, quietly=TRUE, verbose=FALSE)
    model <- bart(as.matrix.data.frame(x_train), as.vector(y_train), as.matrix.data.frame(x_test), ntree=10, printevery=100000, verbose=FALSE)
    predictions <- model$yhat.test.mean
}
if (modelName=="Random Forest") {
    suppressMessages(library(randomForest, quietly=TRUE, verbose=FALSE))
    model <- randomForest(as.matrix.data.frame(x_train), as.vector(y_train), as.matrix.data.frame(x_test), ntree=100)
    predictions <- model$test$predicted
}

write.csv(predictions, outputFile, row.names=FALSE)