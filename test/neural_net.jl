using Base.Test
using MachineLearning

require("linear_data.jl")

x, y = linear_data(2500, 5)
x_train, y_train, x_test, y_test = split_train_test(x, y)
opts = neural_net_options(learning_rate=10.0)
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Valid Stop: ", acc)
@test acc>0.8

opts = neural_net_options(learning_rate=10.0, stop_criteria=StopAfterIteration(40))
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Preset Stop: ", acc)
@test acc>0.8

x = randn(2500, 5)
y = int(map(x->x>0.0, x[:,1]-x[:,2]+3*x[:,3]+x[:,4].*x[:,5]+0.2*randn(2500)))
x_train, y_train, x_test, y_test = split_train_test(x, y)

opts = neural_net_options(learning_rate=10.0)
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 1 Hidden Layer : ", acc)
@test acc>0.80

opts = neural_net_options(hidden_layers=[10;10], learning_rate=10.0)
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 2 Hidden Layers: ", acc)
@test acc>0.80

opts = neural_net_options(hidden_layers=Array(Int, 0), learning_rate=10.0)
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 0 Hidden Layers: ", acc)
@test acc>0.80