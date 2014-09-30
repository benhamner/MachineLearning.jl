using Base.Test
using MachineLearning

num_features=5
x, y = MachineLearning.linear_data(2500, num_features)
split = split_train_test(0.0001*x, y)

net_opts = classification_net_options(learning_rate=10.0)
opts = ClassificationPipelineOptions(TransformerOptions[], net_opts)
acc = evaluate(split, opts, accuracy)
println("Linear Accuracy, Unnormalized: ", acc)
@test acc<0.70

opts = ClassificationPipelineOptions(TransformerOptions[ZmuvOptions()], net_opts)
acc = evaluate(split, opts, accuracy)
println("Linear Accuracy, Normalized: ", acc)
@test acc>0.80

x = randn(1000, 5)
m = [100.0, 10.0, -1.0, 0.1, 0.01]
y = x*m + randn()

split = split_train_test(0.0001*x, y)

net_opts = regression_net_options()
opts = RegressionPipelineOptions(TransformerOptions[], net_opts)
score = evaluate(split, opts, cor)
println("Linear Accuracy, Unnormalized: ", score)
@test score<0.70

opts = RegressionPipelineOptions(TransformerOptions[ZmuvOptions()], net_opts)
score = evaluate(split, opts, cor)
println("Linear Accuracy, Normalized: ", score)
@test score>0.80
