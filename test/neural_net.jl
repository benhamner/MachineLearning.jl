using Base.Test
using MachineLearning

require("linear_data.jl")

x, y = linear_data(2000, 5)
x_train, y_train, x_test, y_test = split_train_test(x, y)
opts = neural_net_options()
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy: ", acc)
@test acc>0.95

x = randn(2000, 5)
y = int(map(x->x>0.0, x[:,1]-x[:,2]+3*x[:,3]+x[:,4].*x[:,5]))
x_train, y_train, x_test, y_test = split_train_test(x, y)

opts = neural_net_options()
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 1 Hidden Layer : ", acc)
@test acc>0.80

opts = neural_net_options(hidden_layers=[10;10])
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 2 Hidden Layers: ", acc)
@test acc>0.80

opts = neural_net_options(hidden_layers=Array(Int, 0))
net = train(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 0 Hidden Layers: ", acc)
@test acc>0.80