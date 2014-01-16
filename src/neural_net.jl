using .MachineLearning
using Optim

type StopAfterIteration
    max_iteration::Int
end
StopAfterIteration() = StopAfterIteration(100)

type StopAfterValidationErrorStopsImproving
    validation_set_size::Float64
    validation_iteration_window_size::Int
    min_iteration::Int
    max_iteration::Int
end
StopAfterValidationErrorStopsImproving() = StopAfterValidationErrorStopsImproving(0.2, 2, 10, 1000)

NeuralNetStopCriteria = Union(StopAfterIteration, StopAfterValidationErrorStopsImproving)

type NeuralNetOptions
    bias_unit::Bool # include a bias unit that always outputs a +1
    hidden_layers::Vector{Int} # sizes of hidden layers
    learning_rate::Float64
    stop_criteria::NeuralNetStopCriteria
end

function neural_net_options(;bias_unit::Bool=true,
                            hidden_layers::Vector{Int}=[20],
                            learning_rate::Float64=1.0,
                            stop_criteria::NeuralNetStopCriteria=StopAfterValidationErrorStopsImproving())
    NeuralNetOptions(bias_unit, hidden_layers, learning_rate, stop_criteria)
end

type NeuralNetLayer
    weights::Array{Float64, 2}
end

type NeuralNet
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

function train(x::Array{Float64, 2}, y::Vector, opts::NeuralNetOptions)
    num_features = size(x, 2)
    classes = sort(unique(y))
    classes_map = Dict(classes, [1:length(classes)])
    num_classes = length(classes)
    net = initialize_net(opts, classes, num_features)

    if typeof(opts.stop_criteria)==StopAfterValidationErrorStopsImproving
        x, y, x_val, y_val = split_train_test(x, y, opts.stop_criteria.validation_set_size)
        actuals_val = one_hot(y_val, classes_map)
        validation_scores = Array(Float64, 0)
    end

    num_samples = size(x, 1)
    actuals = one_hot(y, classes_map)
    update_size = opts.learning_rate / num_samples

    iteration=0
    while true
        iteration += 1
        for j=1:num_samples
            update_weights!(net, vec(x[j,:]), vec(actuals[j,:]), update_size)
        end

        if iteration >= opts.stop_criteria.max_iteration
            break
        end
        if typeof(opts.stop_criteria)==StopAfterValidationErrorStopsImproving
            preds = predict_probs(net, x_val)
            err = mean_log_loss(actuals_val, preds)
            #predst = predict_probs(net, x)
            #errt = mean_log_loss(actuals, predst)
            #println("Iteration: ", iteration, " Val Error: ", err, "Train Err: ", errt)
            push!(validation_scores, err)
            if iteration>=2*opts.stop_criteria.validation_iteration_window_size && iteration>opts.stop_criteria.min_iteration
                ws = opts.stop_criteria.validation_iteration_window_size
                if minimum(validation_scores[iteration-ws+1:iteration])>=maximum(validation_scores[iteration-2*ws+1:iteration-ws])
                    break
                end
            end
        end
    end
    println("Number of Iterations: ", iteration)
    net
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

function predict_probs(net::NeuralNet, samples::Array{Float64, 2})
    probs = Array(Float64, size(samples, 1), length(net.classes))
    for i=1:size(samples, 1)
        probs[i,:] = predict_probs(net, vec(samples[i,:]))
    end
    probs
end

function predict(net::NeuralNet, sample::Vector{Float64})
    probs = predict_probs(net, sample)
    net.classes[minimum(find(x->x==maximum(probs), probs))]
end

function predict(net::NeuralNet, samples::Array{Float64, 2})
    [predict(net, vec(samples[i,:])) for i=1:size(samples,1)]
end

function update_weights!(net::NeuralNet, sample::Vector{Float64}, actual::Vector{Float64}, update_size::Float64)
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
    for i=length(net.layers):-1:1
        gradient = update_size*deltas*(net.options.bias_unit?hcat(1,activations[i]'):activations[i]')
        if i>1
            deltas = net.layers[i].weights'*deltas
            if net.options.bias_unit
            	deltas = deltas[2:length(deltas)]
            end
            deltas = deltas.*sigmoid_gradient(outputs[i])
        end
        net.layers[i].weights -= gradient
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

function cost(net::NeuralNet, x::Array{Float64,2}, actuals::Array{Float64,2}, weights::Vector{Float64})
    weights_to_net!(weights, net)
    probs = predict_probs(net, x)
    err = mean_log_loss(actuals, probs)
    err
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
    layer_gradients = Array(Array{Float64,2},length(net.layers))
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
    gradients = Array(Float64, 0)
    for gradient=layer_gradients
        for g=gradient
            push!(gradients, g)
        end
    end
    gradients
end

function cost_gradient!(net::NeuralNet, x::Array{Float64,2}, actuals::Array{Float64,2}, weights::Vector{Float64}, gradients::Vector{Float64})
    @assert size(x,1)==size(actuals,1)
    weights_to_net!(weights, net)
    gradients[:]=0.0
    for i=1:size(x,1)
        delta = cost_gradient(net, vec(x[i,:]), vec(actuals[i,:]))/size(x,1)
        for j=1:length(delta)
            gradients[j] += delta[j]
        end
    end
end

function train_soph(x::Array{Float64, 2}, y::Vector, opts::NeuralNetOptions)
    num_features = size(x, 2)
    classes = sort(unique(y))
    classes_map = Dict(classes, [1:length(classes)])
    net = initialize_net(opts, classes, num_features)
    actuals = one_hot(y, classes_map)

    initial_weights = net_to_weights(net)

    f = weights -> cost(net, x, actuals, weights)
    g! = (weights, gradients) -> cost_gradient!(net, x, actuals, weights, gradients)
    res = optimize(f, g!, initial_weights, method=:cg, show_trace=true)
    weights_to_net!(res.minimum, net)
    net
end