using Base.Test
using MachineLearning

require("linear_data.jl")

num_features=5
x, y = linear_data(2500, num_features)
x_train, y_train, x_test, y_test = split_train_test(x, y)

forest = fit(x_train, y_train, classification_forest_options(num_trees=10, display=true))
println(forest)
yhat = predict(forest, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy: ", acc)
@test acc>0.80

model = randn(10)
model[3] = 100.0
x     = randn(2500, 10)
y     = x*model
x_train, y_train, x_test, y_test = split_train_test(x, y)

forest = fit(x_train, y_train, regression_forest_options())
println(forest)
yhat = predict(forest, x_test)
correlation = cor(y_test, yhat)
println("Linear Pearson Correlation: ", correlation)
@test correlation>0.80