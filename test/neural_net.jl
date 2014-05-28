using Base.Test
using MachineLearning

require("linear_data.jl")

num_features=5
x, y = linear_data(2500, num_features)
split = split_train_test(x, y)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)

# Checking gradients
println("Checking Gradients")
opts = neural_net_options(hidden_layers=[3])
classes = sort(unique(y))
classes_map = Dict(classes, [1:length(classes)])
net = initialize_net(opts, classes, num_features)
temp = initialize_neural_net_temporary(net)
weights = net_to_weights(net)

actuals = one_hot(y, classes_map)
epsilon = 1e-4
gradients = copy(weights)
for i=1:length(weights)
    w1 = copy(weights)
    w2 = copy(weights)
    w1[i] -= epsilon
    w2[i] += epsilon
    cost_gradient_update_net!(net, x, actuals, weights, gradients, temp)
    c1 = cost(net, x, actuals, w1)
    c2 = cost(net, x, actuals, w2)
    err = abs(((c2-c1)/(2*epsilon)-gradients[i])/gradients[i])
    @test err<epsilon
end

println("Classification Tests")
opts = neural_net_options(learning_rate=10.0, track_cost=true)
net = fit(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Preset Stop: ", acc)
@test acc>0.7

opts = neural_net_options(learning_rate=10.0, train_method=:cg)
net = fit(x_train, y_train, opts)
for layer=net.layers
    println("Max Weight: ", maximum(layer.weights))
end
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Soph: ", acc)
@test acc>0.7

opts = neural_net_options(learning_rate=10.0, stop_criteria=StopAfterValidationErrorStopsImproving())
net = fit(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Linear Accuracy, Valid Stop: ", acc)
@test acc>0.7

x = randn(2500, 5)
y = int(map(x->x>0.0, x[:,1]-x[:,2]+3*x[:,3]+x[:,4].*x[:,5]+0.2*randn(2500)))
split = split_train_test(x, y)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)

opts = neural_net_options(learning_rate=10.0)
net = fit(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 1 Hidden Layer : ", acc)
@test acc>0.70

opts = neural_net_options(hidden_layers=[10;10], learning_rate=10.0)
net = fit(x_train, y_train, opts)
println(net)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 2 Hidden Layers: ", acc)
@test acc>0.70

opts = neural_net_options(hidden_layers=Array(Int, 0), learning_rate=10.0)
net = fit(x_train, y_train, opts)
yhat = predict(net, x_test)
acc = accuracy(y_test, yhat)
println("Nonlinear Accuracy, 0 Hidden Layers: ", acc)
@test acc>0.70
