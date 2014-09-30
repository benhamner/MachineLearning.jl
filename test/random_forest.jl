using Base.Test
using MachineLearning

num_features=5
x, y = MachineLearning.linear_data(2500, num_features)
split = split_train_test(x, y)
score = evaluate(split, classification_forest_options(num_trees=10, display=true), accuracy)
println("Linear Accuracy: ", score)
@test score>0.80

model = randn(10)
model[3] = 100.0
x     = randn(2500, 10)
y     = x*model
split = split_train_test(x, y)
score = evaluate(split, regression_forest_options(), cor)
println("Linear Pearson Correlation: ", score)
@test score>0.80
