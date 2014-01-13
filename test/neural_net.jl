using MachineLearning

xs = randn(10000, 5)
ys = int(map(x->x>0.0, xs[:,1]-xs[:,2]+3*xs[:,3]+xs[:,4].*xs[:,5]))
opts = NeuralNetOptions()
net = train(xs, ys, opts)

for i=1:10
    println(i, " ", predict(net, vec(xs[i,:]))[2], " ", ys[i])
end

xts = randn(10, 5)
yts = int(map(x->x>0.0, xts[:,1]-xts[:,2]+3*xts[:,3]+xts[:,4].*xts[:,5]))

for i=1:10
    println(i, " ", predict(net, vec(xts[i,:]))[2], " ", yts[i])
end