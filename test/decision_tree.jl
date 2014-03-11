using Base.Test
using MachineLearning

require("linear_data.jl")

@test_approx_eq gini([1.0,0]) 0.0
@test_approx_eq gini([10.0,0]) 0.0
@test_approx_eq gini([0.0,0,10]) 0.0
@test_approx_eq gini([1.0,1]) 0.5
@test_approx_eq gini([1.0,1,1]) 2/3
@test_approx_eq gini([1.0,1,2]) 5/8

@test classification_split_location([1,1,1,1,2,2,2,2,2],2)==(0.0, 4)
@test classification_split_location([1,2],2)==(0.0, 1)
@test classification_split_location([2,1,1,1,2,2,2,2,2],2)==(3/18, 4)

x=randn(2500, 2)
y=int(map(i->x[i,1]>0 || x[i,2]>0, 1:size(x,1)))
x_train, y_train, x_test, y_test = split_train_test(x, y)
tree = fit(x_train, y_train, classification_tree_options())
println(tree)
yhat = predict(tree, x_test)
acc = accuracy(y_test, yhat)
println("Quadrant Accuracy: ", acc)
@test acc>0.9

num_features=5
x, y = linear_data(2500, num_features)
x_train, y_train, x_test, y_test = split_train_test(x, y)

tree = fit(x_train, y_train, classification_tree_options())
println(tree)
yhat = predict(tree, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy: ", acc)
@test acc>0.80

tree = fit(x_train, y_train, classification_tree_options(minimum_split_size=50))
println(tree)
yhat = predict(tree, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy: ", acc)
@test acc>0.80

x = [1.4, 2.3, 1.5, 10.7]
@test mean((x-mean(x)).^2)==streaming_mse(sum(x), sum(x.^2), 4)

for i=1:100
    x=randn(rand(1:20, 1)[1])
    @test_approx_eq mean((x-mean(x)).^2) streaming_mse(sum(x), sum(x.^2), length(x))
end

model = randn(10)
model[3] = 100.0
x     = randn(2500, 10)
y     = x*model
x_train, y_train, x_test, y_test = split_train_test(x, y)

tree = fit(x_train, y_train, regression_tree_options())
yhat = predict(tree, x_test)
correlation = cor(y_test, yhat)
println("Linear Pearson Correlation: ", correlation)
@test correlation>0.50