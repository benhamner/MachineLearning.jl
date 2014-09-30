using Base.Test
using MachineLearning

@test_approx_eq gini([1.0,0]) 0.0
@test_approx_eq gini([10.0,0]) 0.0
@test_approx_eq gini([0.0,0,10]) 0.0
@test_approx_eq gini([1.0,1]) 0.5
@test_approx_eq gini([1.0,1,1]) 2/3
@test_approx_eq gini([1.0,1,2]) 5/8

@test classification_split_location([1,1,1,1,2,2,2,2,2],2)==(0.0, 4)
@test classification_split_location([1,2],2)==(0.0, 1)
@test classification_split_location([2,1,1,1,2,2,2,2,2],2)==(3/18, 4)
@test regression_split_location([1.0:10])==(4.0,5)

x=randn(2500, 2)
y=int(map(i->x[i,1]>0 || x[i,2]>0, 1:size(x,1)))
split = split_train_test(x, y)
acc = evaluate(split, classification_tree_options(), accuracy)
println("Quadrant Accuracy: ", acc)
@test acc>0.9

num_features=5
x, y = MachineLearning.linear_data(2500, num_features)
split = split_cross_valid(x, y)
acc = evaluate(split, classification_tree_options(), accuracy)
println("Linear Accuracy: ", acc)
@test acc>0.80

acc = evaluate(split, classification_tree_options(minimum_split_size=50), accuracy)
println("Linear Accuracy: ", acc)
@test acc>0.80

x = [1.4, 2.3, 1.5, 10.7]
@test mean((x.-mean(x)).^2)==streaming_mse(sum(x), sum(x.^2), 4)

for i=1:100
    x=randn(rand(1:20, 1)[1])
    @test_approx_eq mean((x.-mean(x)).^2) streaming_mse(sum(x), sum(x.^2), length(x))
end

correlations = zeros(10)
for i=1:10
    model = randn(10)
    model[3] = 100.0
    x     = randn(2500, 10)
    y     = x*model
    split = split_train_test(x, y)
    correlations[i] = evaluate(split, regression_tree_options(), cor)
end
println("Linear Pearson Correlation: ", mean(correlations))
@test mean(correlations)>0.50

branch = DecisionBranch(1, 0.5, 
	DecisionBranch(2, 1.5, 
		DecisionBranch(3, 2.5, 
			MachineLearning.RegressionLeaf(10.0), 
			MachineLearning.RegressionLeaf(12.0)),
		MachineLearning.RegressionLeaf(20.0)), 
	MachineLearning.RegressionLeaf(30.0))

@test MachineLearning.pretty_string(branch) == "DecisionBranch(1, 0.500000, \n|DecisionBranch(2, 1.500000, \n||DecisionBranch(3, 2.500000, \n|||RegressionLeaf(10.0) \n|||RegressionLeaf(12.0)) \n||RegressionLeaf(20.0)) \n|RegressionLeaf(30.0))"
