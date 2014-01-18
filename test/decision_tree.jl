using Base.Test
using MachineLearning

require("linear_data.jl")

@test_approx_eq gini([1.0,0]) 0.0
@test_approx_eq gini([10.0,0]) 0.0
@test_approx_eq gini([0.0,0,10]) 0.0
@test_approx_eq gini([1.0,1]) 0.5
@test_approx_eq gini([1.0,1,1]) 2/3
@test_approx_eq gini([1.0,1,2]) 5/8

@test split_location([1,1,1,1,2,2,2,2,2],2)==(0.0, 4)
@test split_location([1,2],2)==(0.0, 1)
@test split_location([2,1,1,1,2,2,2,2,2],2)==(3/8, 4)

num_features=5
x, y = linear_data(2500, num_features)
x_train, y_train, x_test, y_test = split_train_test(x, y)

tree = train(x_train, y_train, DecisionTreeOptions())
yhat = predict(tree, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy: ", acc)
@test acc>0.55