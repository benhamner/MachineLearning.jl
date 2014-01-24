using .MachineLearning
using Optim
using StatsBase

type StopAfterIteration
    max_iteration::Int
end
StopAfterIteration() = StopAfterIteration(50)

type StopAfterValidationErrorStopsImproving
    validation_set_size::Float64
    validation_iteration_window_size::Int
    min_iteration::Int
    max_iteration::Int
end
StopAfterValidationErrorStopsImproving() = StopAfterValidationErrorStopsImproving(0.2, 2, 10, 1000)

NeuralNetStopCriteria = Union(StopAfterIteration, StopAfterValidationErrorStopsImproving)

type NeuralNetOptions <: SupervisedModelOptions
    bias_unit::Bool # include a bias unit that always outputs a +1
    hidden_layers::Vector{Int} # sizes of hidden layers
    train_method::Symbol
    learning_rate::Float64
    stop_criteria::NeuralNetStopCriteria
end

function neural_net_options(;bias_unit::Bool=true,
                            hidden_layers::Vector{Int}=[20],
                            train_method::Symbol=:sgd,
                            learning_rate::Float64=1.0,
                            stop_criteria::NeuralNetStopCriteria=StopAfterIteration())
    NeuralNetOptions(bias_unit, hidden_layers, train_method, learning_rate, stop_criteria)
end

type NeuralNetLayer
    weights::Matrix{Float64}
end

type NeuralNet <: ClassificationModel
    options::NeuralNetOptions
    layers::Vector{NeuralNetLayer}
    classes::Vector
end

sigmoid(z::Vector{Float64}) = 1/(1+exp(-z))
sigmoid_gradient(z::Vector{Float64}) = sigmoid(z) .* (1-sigmoid(z))

function one_hot(y::Vector, classes_map::Dict)
    values = zeros(length(y), length(classes_map))
    for i=1:length(y)
        values[i, classes_map[y[i]]] = 1.0
    end
    values
end

function fit(x::Matrix{Float64}, y::Vector, opts::NeuralNetOptions)
    num_features = size(x, 2)
    classes = sort(unique(y))
    classes_map = Dict(classes, [1:length(classes)])
    num_classes = length(classes)
    net = initialize_net(opts, classes, num_features)

    if opts.train_method==:sgd # stochastic gradient descent
        if typeof(opts.stop_criteria)==StopAfterIteration
            train_preset_stop!(net, x, one_hot(y, classes_map))
        elseif typeof(opts.stop_criteria)==StopAfterValidationErrorStopsImproving
            x_train, y_train, x_val, y_val = split_train_test(x, y, opts.stop_criteria.validation_set_size)
            train_valid_stop!(net, x_train, one_hot(y_train, classes_map), x_val, one_hot(y_val, classes_map))
        end
    else
        # use optimize from Optim.jl
        actuals = one_hot(y, classes_map)
        initial_weights = net_to_weights(net)

        f = weights -> cost(net, x, actuals, weights)
        g! = (weights, gradients) -> cost_gradient!(net, x, actuals, weights, gradients)
        res = optimize(f, g!, initial_weights, method=opts.train_method)
        weights_to_net!(res.minimum, net)
        net
    end
    net
end

function train_preset_stop!(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64})
    num_samples = size(x,1)
    for iter=1:net.options.stop_criteria.max_iteration
        for j=1:num_samples
            update_weights!(net, vec(x[j,:]), vec(actuals[j,:]), net.options.learning_rate, num_samples)
        end
    end
end

function train_valid_stop!(net::NeuralNet,
                           x_train::Matrix{Float64},
                           a_train::Matrix{Float64},
                           x_val::Matrix{Float64},
                           a_val::Matrix{Float64})
    num_samples = size(x_train,1)
    validation_scores = Array(Float64, 0)
    
    iteration = 0
    while iteration<net.options.stop_criteria.max_iteration
        iteration += 1
        for j=1:num_samples
            update_weights!(net, vec(x_train[j,:]), vec(a_train[j,:]), net.options.learning_rate, num_samples)
        end
        preds = predict_probs(net, x_val)
        err = mean_log_loss(a_val, preds)
        push!(validation_scores, err)
        if iteration>=2*net.options.stop_criteria.validation_iteration_window_size && iteration>net.options.stop_criteria.min_iteration
            ws = net.options.stop_criteria.validation_iteration_window_size
            if minimum(validation_scores[iteration-ws+1:iteration])>=maximum(validation_scores[iteration-2*ws+1:iteration-ws])
                break
            end
        end
    end
    println("Number of Iterations: ", iteration)
end

function predict_probs(net::NeuralNet, sample::Vector{Float64})
    state = sample
    for layer = net.layers
        if net.options.bias_unit==true
            state = [1.0;state]
        end

        state = sigmoid(layer.weights*state)
    end
    state
end

function predict_probs(net::NeuralNet, samples::Matrix{Float64})
    probs = Array(Float64, size(samples, 1), length(net.classes))
    for i=1:size(samples, 1)
        probs[i,:] = predict_probs(net, vec(samples[i,:]))
    end
    probs
end

function StatsBase.predict(net::NeuralNet, sample::Vector{Float64})
    probs = predict_probs(net, sample)
    net.classes[minimum(find(x->x==maximum(probs), probs))]
end

function StatsBase.predict(net::NeuralNet, samples::Matrix{Float64})
    [StatsBase.predict(net, vec(samples[i,:])) for i=1:size(samples,1)]
end

function update_weights!(net::NeuralNet, sample::Vector{Float64}, actual::Vector{Float64}, learning_rate::Float64, num_samples::Int)
    layer_gradients = cost_gradient(net, sample, actual)/num_samples
    regularization_gradients = regularization_gradient(net)/num_samples^2

    for i=1:length(net.layers)
        net.layers[i].weights -= learning_rate*(layer_gradients[i] + regularization_gradients[i])
    end
end

function initialize_layer(number_in::Int, number_out::Int)
    epsilon_init = sqrt(6) / sqrt(number_in + number_out)
    weights = 2.0 * (rand(number_out, number_in) - 0.5) * epsilon_init
    NeuralNetLayer(weights)
end

function initialize_net(opts::NeuralNetOptions, classes::Vector, num_features::Int)
    layers = Array(NeuralNetLayer, 0)
    if isempty(opts.hidden_layers)
        push!(layers, initialize_layer(num_features + (opts.bias_unit?1:0), length(classes)))
    else
        push!(layers, initialize_layer(num_features + (opts.bias_unit?1:0), opts.hidden_layers[1]))
        for i=1:length(opts.hidden_layers)-1
            push!(layers, initialize_layer(opts.hidden_layers[i] + (opts.bias_unit?1:0), opts.hidden_layers[i+1]))
        end
        push!(layers, initialize_layer(opts.hidden_layers[length(opts.hidden_layers)] + (opts.bias_unit?1:0), length(classes)))
    end
    NeuralNet(opts, layers, classes)
end

function weights_to_net!(weights::Vector{Float64}, net::NeuralNet)
    loc = 0
    for layer = net.layers
        layer.weights[:] = weights[loc+1:loc+length(layer.weights)]
        loc += length(layer.weights)
    end
    @assert loc==length(weights)
end

function net_to_weights(net::NeuralNet)
    weights = Array(Float64, 0)
    for layer=net.layers
        for w=layer.weights
            push!(weights, w)
        end
    end
    weights
end

function cost(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64})
    @assert size(x,1)==size(actuals,1)
    probs = predict_probs(net, x)
    err = mean_log_loss(actuals, probs)
    regularization = 0.0
    for layer=net.layers
        w = copy(layer.weights)
        if net.options.bias_unit
            w[:,1] = 0.0
        end
        regularization += sum(w.^2)/(2*size(x,1))
    end
    err + regularization
end

function cost(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64}, weights::Vector{Float64})
    weights_to_net!(weights, net)
    cost(net, x, actuals)
end

function cost_gradient(net::NeuralNet, sample::Vector{Float64}, actual::Vector{Float64})
    outputs = Array(Vector{Float64}, 0) # before passing through sigmoid
    activations = Array(Vector{Float64}, 0)
    push!(outputs, sample)
    push!(activations, sample)
    state = sample
    for layer = net.layers
        if net.options.bias_unit
            state = [1.0;state]
        end

        push!(outputs, layer.weights*state)
        state = sigmoid(outputs[length(outputs)])
        push!(activations, state)
    end

    deltas = activations[length(activations)] - actual
    layer_gradients = Array(Matrix{Float64},length(net.layers))
    for i=length(net.layers):-1:1
        gradient = deltas*(net.options.bias_unit?hcat(1,activations[i]'):activations[i]')
        if i>1
            deltas = net.layers[i].weights'*deltas
            if net.options.bias_unit
                deltas = deltas[2:length(deltas)]
            end
            deltas = deltas.*sigmoid_gradient(outputs[i])
        end
        layer_gradients[i]=gradient
    end
    layer_gradients
end

function regularization_gradient(net::NeuralNet)
    layer_gradients = [copy(layer.weights) for layer=net.layers]
    for i=1:length(layer_gradients)
        if net.options.bias_unit
            layer_gradients[i][:,1]=0.0
        end
    end
    layer_gradients
end

function cost_gradient!(net::NeuralNet, x::Matrix{Float64}, actuals::Matrix{Float64}, weights::Vector{Float64}, gradients::Vector{Float64})
    @assert size(x,1)==size(actuals,1)
    weights_to_net!(weights, net)
    gradients[:]=0.0
    layer_gradients = [0.0*layer.weights for layer=net.layers]
    for i=1:size(x,1)
        delta = cost_gradient(net, vec(x[i,:]), vec(actuals[i,:]))/size(x,1)
        layer_gradients += delta
    end
    regularization = regularization_gradient(net)
    layer_gradients += regularization/size(x,1)

    i=0
    for gradient=layer_gradients
        for g=gradient
            i += 1
            gradients[i] = g
        end
    end
end

function Base.show(io::IO, net::NeuralNet)
    info = join(["Neural Network",
                 @sprintf("    %d Hidden Layers",length(net.options.hidden_layers)),
                 @sprintf("    %d Classes",length(net.classes))], "\n")
    print(io, info)
end