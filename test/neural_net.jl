using Base.Test
using MachineLearning

require("linear_data.jl")

xs, ys = linear_data(1000, 5)
opts = NeuralNetOptions()
net = train(xs, ys, opts)
yhat = [int(predict(net, vec(xs[i,:]))[2]>=0.5) for i=1:size(xs,1)]
accuracy = sum(map(x->x[1]==x[2], zip(ys, yhat)))/size(xs, 1)
println("Linear Accuracy: ", accuracy)
@test accuracy>0.95

xs = randn(1000, 5)
ys = int(map(x->x>0.0, xs[:,1]-xs[:,2]+3*xs[:,3]+xs[:,4].*xs[:,5]))
opts = NeuralNetOptions()
net = train(xs, ys, opts)

xts = randn(1000, 5)
yts = int(map(x->x>0.0, xts[:,1]-xts[:,2]+3*xts[:,3]+xts[:,4].*xts[:,5]))
yhat = [int(predict(net, vec(xts[i,:]))[2]>=0.5) for i=1:size(xts,1)]
accuracy = sum(map(x->x[1]==x[2], zip(yts, yhat)))/size(xts, 1)
println("Nonlinear Accuracy: ", accuracy)
@test accuracy>0.90

#for i=1:10
#    println(i, " ", predict(net, vec(xts[i,:]))[2], " ", yts[i])
#end