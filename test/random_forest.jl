using Base.Test
using MachineLearning

require("linear_data.jl")

num_features=5
x, y = linear_data(2500, num_features)
split = split_train_test(x, y)
acc = evaluate(split, classification_forest_options(num_trees=10, display=true), accuracy)
println("Linear Accuracy: ", acc)
@test acc>0.80

model = randn(10)
model[3] = 100.0
x     = randn(2500, 10)
y     = x*model
split = split_train_test(x, y)
correlation = evaluate(split, regression_forest_options(), cor)
println("Linear Pearson Correlation: ", correlation)
@test correlation>0.80
