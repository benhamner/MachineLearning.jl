using Base.Test
using MachineLearning

require("linear_data.jl")

xs, ys = linear_data(1000, 5)
opts = neural_net_options()
net = train(xs, ys, opts)
yhat = predict(net, xs)
accuracy = sum(map(x->x[1]==x[2], zip(ys, yhat)))/size(xs, 1)
println("Linear Accuracy: ", accuracy)
@test accuracy>0.95

xs = randn(1000, 5)
ys = int(map(x->x>0.0, xs[:,1]-xs[:,2]+3*xs[:,3]+xs[:,4].*xs[:,5]))
xts = randn(1000, 5)
yts = int(map(x->x>0.0, xts[:,1]-xts[:,2]+3*xts[:,3]+xts[:,4].*xts[:,5]))

opts = neural_net_options()
net = train(xs, ys, opts)
yhat = predict(net, xts)
accuracy = sum(map(x->x[1]==x[2], zip(yts, yhat)))/size(xts, 1)
println("Nonlinear Accuracy, 1 Hidden Layer : ", accuracy)
@test accuracy>0.90

opts = neural_net_options(hidden_layers=[10;10])
net = train(xs, ys, opts)
yhat = predict(net, xts)
accuracy = sum(map(x->x[1]==x[2], zip(yts, yhat)))/size(xts, 1)
println("Nonlinear Accuracy, 2 Hidden Layers: ", accuracy)
@test accuracy>0.90

opts = neural_net_options(hidden_layers=Array(Int, 0))
net = train(xs, ys, opts)
yhat = predict(net, xts)
accuracy = sum(map(x->x[1]==x[2], zip(yts, yhat)))/size(xts, 1)
println("Nonlinear Accuracy, 0 Hidden Layers: ", accuracy)
@test accuracy>0.80