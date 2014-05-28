using Base.Test
using MachineLearning

require("linear_data.jl")

num_features=5
x, y = linear_data(2500, num_features)
split = split_train_test(0.0001*x, y)

net_opts = neural_net_options(learning_rate=10.0)
opts = PipelineOptionsAny([], net_opts)
acc = evaluate(split, opts, accuracy)
println("Linear Accuracy, Unnormalized: ", acc)
@test acc<0.55

opts = PipelineOptionsAny([ZmuvOptions()], net_opts)
acc = evaluate(split, opts, accuracy)
println("Linear Accuracy, Normalized: ", acc)
@test acc>0.80
