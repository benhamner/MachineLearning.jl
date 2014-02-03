using Base.Test
using MachineLearning

require("linear_data.jl")

num_features=5
x, y = linear_data(2500, num_features)
x_train, y_train, x_test, y_test = split_train_test(0.0001*x, y)

net_opts = neural_net_options(learning_rate=10.0)
opts = PipelineOptions([], net_opts)
pipeline = fit(x_train, y_train, opts)
yhat = predict(pipeline, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Unnormalized: ", acc)
@test acc<0.55

opts = PipelineOptions([ZmuvOptions()], net_opts)
pipeline = fit(x_train, y_train, opts)
yhat = predict(pipeline, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Normalized: ", acc)
@test acc>0.80