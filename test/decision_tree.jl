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
@test split_location([2,1,1,1,2,2,2,2,2],2)==(3/18, 4)

leaf1 = ClassificationLeaf([1.0,0.0])
leaf2 = ClassificationLeaf([0.5,0.5])
leaf3 = ClassificationLeaf([1.0,0.0])
leaf4 = ClassificationLeaf([1.0,0.0])
leaf5 = ClassificationLeaf([0.0,1.0,0.0])

branch1 = DecisionBranch(1, 0.1, leaf1, leaf2)
branch2 = DecisionBranch(1, 0.2, branch1, leaf3)
branch3 = DecisionBranch(1, 0.3, leaf4, leaf5)
branch4 = DecisionBranch(1, 0.4, branch2, branch3)

tree = ClassificationTree(branch4, [1,2], 1, ClassificationTreeOptions())

@test length(leaf1)==1
@test length(leaf2)==1
@test length(branch1)==3
@test length(branch2)==5
@test length(branch3)==3
@test length(branch4)==9
@test length(tree)==9

@test depth(leaf1)==1
@test depth(branch1)==2
@test depth(branch2)==3
@test depth(branch3)==2
@test depth(branch4)==4
@test depth(tree)==4

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